#!/usr/bin/env python3
"""One-dimensional Buckley--Leverett validation of the **P1 finite-element
solver used by ``two_phase_transport.solve_two_phase``** (the repaired
version): P1 CG, implicit Euler, Picard linearization of the fractional
flow, residual SUPG with characteristic speed ``fw'(Sw) u``, upwind
exterior states on the inlet/outlet facets, and the conservative bounded
limiter on ``[Swr, 1-Sor]``.

This 1-D script mirrors the 3-D solver's weak form, SUPG tau, boundary
numerical fluxes, Picard loop, limiter and mass ledger exactly, on an
interval mesh with a constant total velocity, and compares against the
exact Buckley--Leverett solution (Welge tangent + rarefaction + shock):

- rarefaction: x/t = fw'(S) for S in [S*, S_inj], fw'(S) from v_s down to 0;
- shock:       x = v_s t, S jumps from S* down to Swi,
  fw'(S*) = (fw(S*) - fw(Swi))/(S* - Swi), v_s = (fw(S*) - fw(Swi))/(S* - Swi);
- exact mass:  M(t) = Swi L + (fw(S_inj) - fw(Swi)) u t.

Run: conda run -n fenicsx --no-capture-output python -u validate_1d_buckley_leverett.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import ufl
from basix.ufl import element
from dolfinx import fem, mesh as dmesh
from mpi4py import MPI
from scipy.sparse.linalg import splu

from ddpnm_core.fem_utils import to_scipy_matrix
from two_phase_physics import (
    DEFAULT_COREY,
    conservative_bounded_limiter,
    fractional_flow,
    fractional_flow_derivative,
)

COREY = dict(DEFAULT_COREY)
SWI = 0.2        # initial saturation (== Swr; the repaired solver default)
S_INJ = 1.0 - COREY["sor"]   # 0.8, the repaired solver default inlet value


# ---------------------------------------------------------------------------
# Exact solution (Welge tangent + rarefaction + shock)
# ---------------------------------------------------------------------------


def welge_shock(corey: dict, swi: float, s_inj: float) -> tuple[float, float]:
    """(v_s, S*) with fw'(S*) = (fw(S*) - fw(Swi))/(S* - Swi)."""
    swr = corey["swr"]

    def f(s):
        return float(np.asarray(fractional_flow(s, **corey)).item())

    def fp(s):
        return float(np.asarray(fractional_flow_derivative(s, **corey)).item())

    f_swi = f(swi)

    def resid(s):
        return fp(s) - (f(s) - f_swi) / (s - swi)

    s_star = brentq(resid, swi + 1.0e-8, s_inj - 1.0e-8)
    v_s = (f(s_star) - f_swi) / (s_star - swi)
    return v_s, s_star


def rarefaction_position(s: float, corey: dict) -> float:
    return float(np.asarray(fractional_flow_derivative(s, **corey)).item())


def exact_profile(x: np.ndarray, t: float, corey: dict, swi: float, s_inj: float) -> np.ndarray:
    v_s, s_star = welge_shock(corey, swi, s_inj)
    x = np.asarray(x, dtype=float)
    s = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi <= 0.0:
            s[i] = s_inj
        elif xi >= v_s * t:
            s[i] = swi
        else:
            def g(ss):
                return rarefaction_position(ss, corey) - xi / t

            # fw'(S) decreases from v_s (at S*) down to fw'(S_inj) (at S_inj).
            if g(s_star) >= 0.0 >= g(s_inj - 1.0e-12):
                s[i] = brentq(g, s_star, s_inj - 1.0e-12)
            else:
                s[i] = s_star
    return s


# ---------------------------------------------------------------------------
# 1-D mirror of the repaired P1 finite-element solver
# ---------------------------------------------------------------------------


def solve_1d_blev(L: float, nx: int, u_val: float, dt: float, t_final: float,
                  corey: dict, swi: float = SWI, s_inj: float = S_INJ,
                  picard_max_iters: int = 6, picard_tol: float = 1.0e-6,
                  picard_relaxation: float = 1.0,
                  supg: bool = True, supg_factor: float = 0.5,
                  porosity: float = 1.0, diffusivity: float = 0.0) -> dict:
    msh = dmesh.create_interval(MPI.COMM_WORLD, nx, (0.0, L))
    cell = msh.basix_cell()
    C = fem.functionspace(msh, element("Lagrange", cell, 1))
    S_trial = ufl.TrialFunction(C)
    w = ufl.TestFunction(C)
    u = fem.Constant(msh, float(u_val))
    normal = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    dx = ufl.dx
    lower, upper = float(corey["swr"]), 1.0 - float(corey["sor"])
    swr, sor = float(corey["swr"]), float(corey["sor"])
    nw, no = float(corey["nw"]), float(corey["no"])
    mu_w, mu_o = float(corey["mu_w"]), float(corey["mu_o"])

    mass_matrix = to_scipy_matrix(fem.assemble_matrix(fem.form(S_trial * w * dx))).tocsr()
    mass_weights = np.asarray(mass_matrix @ np.ones(mass_matrix.shape[1]), dtype=float)
    total_volume = float(np.sum(mass_weights))
    diffusion_matrix = None
    if diffusivity > 0.0:
        diffusion_matrix = to_scipy_matrix(
            fem.assemble_matrix(fem.form(ufl.inner(ufl.grad(S_trial), ufl.grad(w)) * dx))
        ).tocsr()

    # Inlet (x=0, tag 1) and outlet (x=L, tag 2) facets.
    fdim = 0
    inlet_facets = np.asarray(
        dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0)),
        dtype=np.int32,
    )
    outlet_facets = np.asarray(
        dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], L)),
        dtype=np.int32,
    )
    boundary_facets = np.concatenate((inlet_facets, outlet_facets))
    boundary_values = np.concatenate(
        (np.full(len(inlet_facets), 1, dtype=np.int32),
         np.full(len(outlet_facets), 2, dtype=np.int32))
    )
    order = np.argsort(boundary_facets)
    facet_tags = dmesh.meshtags(msh, fdim, boundary_facets[order], boundary_values[order])
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    fw_fun = fem.Function(C)
    df_fun = fem.Function(C)
    convection_form = fem.form(fw_fun * u * w.dx(0) * dx)
    un = u * normal[0]
    un_pos = ufl.max_value(un, 0.0)
    un_neg = ufl.min_value(un, 0.0)
    fw_inlet_c = fem.Constant(msh, float(fractional_flow(np.asarray([s_inj]), swr, sor, nw, no, mu_w, mu_o)[0]))
    fw_backflow_c = fem.Constant(msh, float(fractional_flow(np.asarray([swi]), swr, sor, nw, no, mu_w, mu_o)[0]))
    inlet_numerical_flux = un_pos * fw_fun + un_neg * fw_inlet_c
    outlet_numerical_flux = un_pos * fw_fun + un_neg * fw_backflow_c
    boundary_vector_form = fem.form(inlet_numerical_flux * w * ds(1) + outlet_numerical_flux * w * ds(2))
    boundary_net_form = fem.form(inlet_numerical_flux * ds(1) + outlet_numerical_flux * ds(2))
    water_in_form = fem.form((-un_neg * fw_inlet_c) * ds(1) + (-un_neg * fw_backflow_c) * ds(2))
    water_out_form = fem.form((un_pos * fw_fun) * ds(1) + (un_pos * fw_fun) * ds(2))
    outlet_water_form = fem.form(un_pos * fw_fun * ds(2))
    outlet_total_form = fem.form(un_pos * ds(2))

    dt_constant = fem.Constant(msh, float(dt))
    phi_constant = fem.Constant(msh, float(porosity))
    characteristic_speed = ufl.sqrt(
        df_fun * df_fun * u * u + fem.Constant(msh, 1.0e-30)
    )
    tau = fem.Constant(msh, float(supg_factor)) / ufl.sqrt(
        (2.0 * phi_constant / dt_constant) ** 2
        + (2.0 * characteristic_speed / h) ** 2
        + fem.Constant(msh, 1.0e-30)
    )
    streamline_test = df_fun * u * w.dx(0)
    supg_mass_form = fem.form(tau * streamline_test * S_trial * dx)
    supg_speed_form = fem.form(tau * streamline_test * df_fun * u * S_trial.dx(0) * dx)

    def assemble_global(form) -> float:
        return float(msh.comm.allreduce(float(fem.assemble_scalar(form)), op=MPI.SUM))

    outlet_volume_rate = assemble_global(outlet_total_form)
    if outlet_volume_rate <= 1.0e-14:
        raise RuntimeError("no positive outlet flux")

    cold = np.full(C.dofmap.index_map.size_local * C.dofmap.index_map_bs, float(swi), dtype=float)

    times = [0.0]
    watercuts = [assemble_global(outlet_water_form) / outlet_volume_rate]
    masses = [float(porosity) * float(np.dot(mass_weights, cold))]
    budget_masses = [masses[0]]
    relative_balance_residuals = [0.0]
    picard_counts = [0]
    picard_converged = [True]
    mins = [float(np.min(cold))]
    maxs = [float(np.max(cold))]
    limiter_failed = []

    n_steps = int(np.ceil(t_final / dt))
    current_time = 0.0
    for step in range(1, n_steps + 1):
        dt_step = min(float(dt), float(t_final) - current_time)
        if dt_step <= 1.0e-14:
            break
        dt_constant.value = dt_step
        rhs_mass_matrix = (float(porosity) / dt_step) * mass_matrix
        a_fixed = rhs_mass_matrix.copy()
        if diffusion_matrix is not None:
            a_fixed = a_fixed + float(diffusivity) * diffusion_matrix
        lu_fixed = splu(a_fixed.tocsc()) if not supg else None

        s_iter = cold.copy()
        iters_used = 0
        converged = False
        candidate = cold.copy()
        net_boundary_rate_used = 0.0
        # Mirror of the 3-D solver's performance form: SUPG characteristic
        # speed frozen at fw'(S^n) for the step, one factorization per step.
        if supg:
            dfw_step = fractional_flow_derivative(cold, swr, sor, nw, no, mu_w, mu_o)
            df_fun.x.array[:] = dfw_step
            df_fun.x.scatter_forward()
            supg_mass_matrix = to_scipy_matrix(fem.assemble_matrix(supg_mass_form)).tocsr()
            supg_speed_matrix = to_scipy_matrix(fem.assemble_matrix(supg_speed_form)).tocsr()
            a_step = a_fixed + (float(porosity) / dt_step) * supg_mass_matrix
            a_step = a_step + supg_speed_matrix
            lu_step = splu(a_step.tocsc())
        else:
            supg_mass_matrix = None
            lu_step = lu_fixed
        for it in range(picard_max_iters):
            fw_vec = fractional_flow(s_iter, swr, sor, nw, no, mu_w, mu_o)
            fw_fun.x.array[:] = fw_vec
            fw_fun.x.scatter_forward()
            rhs = rhs_mass_matrix @ cold
            rhs = rhs + fem.assemble_vector(convection_form).array
            rhs = rhs - fem.assemble_vector(boundary_vector_form).array
            net_boundary_rate_used = assemble_global(boundary_net_form)
            if supg:
                rhs = rhs + (float(porosity) / dt_step) * (supg_mass_matrix @ cold)
            candidate = lu_step.solve(rhs)
            iters_used = it + 1
            if float(np.max(np.abs(candidate - s_iter))) <= picard_tol:
                converged = True
                break
            s_iter = (
                float(picard_relaxation) * candidate
                + (1.0 - float(picard_relaxation)) * s_iter
            )
        raw = candidate

        raw_mass = float(porosity) * float(np.dot(mass_weights, raw))
        try:
            limited, limiter_info = conservative_bounded_limiter(
                raw, mass_weights, raw_mass, lower, upper,
            )
            limiter_failed.append(False)
        except RuntimeError:
            limited = np.clip(raw, lower, upper)
            limiter_failed.append(True)
        cold = limited
        new_mass = float(porosity) * float(np.dot(mass_weights, cold))

        budget_masses.append(budget_masses[-1] + dt_step * net_boundary_rate_used)
        mass_balance_abs = float(new_mass - masses[-1] + dt_step * net_boundary_rate_used)
        scale = max(abs(new_mass - masses[-1]), abs(dt_step * net_boundary_rate_used), 1.0e-14)
        relative_balance_residuals.append(abs(mass_balance_abs) / scale)

        current_time += dt_step
        times.append(current_time)
        watercuts.append(assemble_global(outlet_water_form) / outlet_volume_rate)
        masses.append(new_mass)
        picard_counts.append(iters_used)
        picard_converged.append(converged)
        mins.append(float(np.min(cold)))
        maxs.append(float(np.max(cold)))

    return {
        "final_saturation": cold,
        "coords": C.tabulate_dof_coordinates(),
        "masses": masses,
        "budget_masses": budget_masses,
        "times": times,
        "relative_conservation_residual": relative_balance_residuals,
        "picard_counts": picard_counts,
        "picard_converged": picard_converged,
        "limiter_failed": limiter_failed,
        "total_volume": total_volume,
        "watercuts": watercuts,
        "outlet_volume_rate": outlet_volume_rate,
        "lower": lower,
        "upper": upper,
    }


# ---------------------------------------------------------------------------
# Metrics vs exact solution
# ---------------------------------------------------------------------------


def main() -> None:
    L, nx, u_val = 10.0, 400, 1.0
    dt, t_final = float(os.environ.get("VALIDATE_DT", "0.0025")), 1.0

    v_s, s_star = welge_shock(COREY, SWI, S_INJ)
    fw_swi = float(np.asarray(fractional_flow(SWI, **COREY)).item())
    fw_inj = float(np.asarray(fractional_flow(S_INJ, **COREY)).item())
    print(f"Welge: S* = {s_star:.4f}, v_s = {v_s:.4f}, fw(Swi) = {fw_swi:.4f}, fw(Sinj) = {fw_inj:.4f}")
    xf = v_s * t_final
    print(f"exact shock at t={t_final}: x = {xf:.4f} (domain L={L})")

    result = solve_1d_blev(L, nx, u_val, dt, t_final, COREY, swi=SWI, s_inj=S_INJ,
                              picard_relaxation=float(os.environ.get("VALIDATE_RELAX", "1.0")))
    s = result["final_saturation"]
    x = result["coords"][:, 0]
    h = x[1] - x[0]

    s_exact = exact_profile(x, t_final, COREY, SWI, S_INJ)
    # Shock location: first crossing of the S* iso-line (profile is monotone).
    below = np.flatnonzero(s < s_star)
    if len(below) == 0:
        print("FAIL: no S < S* dofs")
        return 1
    i0 = int(below[0])
    if i0 == 0:
        x_num = x[0]
    else:
        s0, s1 = float(s[i0 - 1]), float(s[i0])
        if abs(s1 - s0) <= 1.0e-14:
            x_num = x[i0]
        else:
            x_num = x[i0 - 1] + (s_star - s0) * (x[i0] - x[i0 - 1]) / (s1 - s0)
    print(f"numerical shock: x = {x_num:.4f}, exact = {xf:.4f}, "
          f"error = {x_num - xf:+.4f} ({(x_num - xf) / xf * 100:+.2f}%)")

    l1 = float(np.sum(np.abs(s - s_exact)) * h)
    print(f"profile L1 error: {l1:.4e} (={(l1 / (S_INJ - SWI) / h):.2f} cells of smearing)")

    m_exact = SWI * L + (fw_inj - fw_swi) * u_val * t_final
    m_num = result["masses"][-1]
    print(f"mass: num = {m_num:.6f}, exact = {m_exact:.6f}, rel error = {(m_num - m_exact) / m_exact:+.3e}")

    res = float(np.max(result["relative_conservation_residual"]))
    print(f"max relative conservation residual: {res:.3e}")
    print(f"Picard: mean iters {np.mean(result['picard_counts'][1:]):.2f}, "
          f"converged steps {sum(result['picard_converged'][1:])}/{len(result['times']) - 1}")
    print(f"limiter infeasible-target rejections: {sum(result['limiter_failed'])}")
    print(f"saturation range: [{s.min():.4f}, {s.max():.4f}] (physical [{result['lower']}, {result['upper']}])")
    below0 = int(np.count_nonzero(s < result["lower"] - 1.0e-12))
    above1 = int(np.count_nonzero(s > result["upper"] + 1.0e-12))
    print(f"out-of-bounds dofs: below {below0}, above {above1}")

    np.savez_compressed(
        PROJECT_DIR / "outputs" / "validate_1d_blev.npz",
        x=x, s_num=s, s_exact=s_exact, v_s=v_s, s_star=s_star,
        t_final=t_final, L=L, swi=SWI, s_inj=S_INJ,
    )

    ok = (abs(x_num - xf) / xf < 0.03) and (res < 1.0e-12) and (l1 < 0.10) \
        and (below0 == 0) and (above1 == 0)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

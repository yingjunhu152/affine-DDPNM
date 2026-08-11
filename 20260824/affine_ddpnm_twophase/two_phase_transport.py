"""Two-phase (Buckley--Leverett) saturation transport driven by a vertex
velocity field, in the same validation structure as the single-phase tracer.

Model: immiscible water--oil displacement, incompressible total velocity
``u`` (fixed, supplied as P1 vertex values), Corey relative permeabilities
and fractional flow ``fw(Sw)``.  The saturation satisfies

    dSw/dt + div(fw(Sw) u) - kappa Laplacian(Sw) = 0

with an upwind exterior state ``Sw = Sw_inlet`` on inflowing parts of
``x = 0``, a conservative outflow term on ``x = L``, and impermeable walls.
The discretization is P1 implicit Euler with Picard iteration, residual-based
SUPG using the characteristic speed ``fw'(Sw) u``, and a conservative limiter
on the physical interval ``[Swr, 1-Sor]``.

The solver helpers (limiter, mass bookkeeping, CSV output, plot style)
are ported from the single-phase tracer project and inlined below, so this
module is fully standalone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import ufl
from basix.ufl import element
from dolfinx import fem, mesh as dmesh
from mpi4py import MPI
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from ddpnm_core.fem_utils import to_scipy_matrix
from ddpnm_core.io import assign_p1_function, topology_vertex_coordinates
from two_phase_physics import (
    DEFAULT_COREY,
    conservative_bounded_limiter,
    crossing_time,
    effective_saturation,
    fractional_flow,
    fractional_flow_derivative,
    mass_balance_residual,
    phase_mobilities,
    recovery_factor,
    signed_time_error,
    validate_corey,
    weighted_average,
)

METHODS = ("FEM", "Classic-DDPNM-1", "NormalLinear-DDPNM-3", "Affine-DDPNM-9")

# ---------------------------------------------------------------------------
# Helpers ported from ``tracer_transport`` (the single-phase tracer project)
# and inlined here so this module is fully standalone.  The limiter and the
# mass bookkeeping are identical to the tracer's; only the solver itself is
# two-phase.
# ---------------------------------------------------------------------------


COLORS = {
    "FEM": "#2b2d42",
    "Classic-DDPNM-1": "#8d6e63",
    "NormalLinear-DDPNM-3": "#7B1FA2",
    "Affine-DDPNM-9": "#d1495b",
}


def nearest_indices(
    query_coords: np.ndarray,
    source_coords: np.ndarray,
    label: str,
    tolerance: float = 1.0e-8,
) -> tuple[np.ndarray, float]:
    tree = cKDTree(np.asarray(source_coords, dtype=float))
    distances, indices = tree.query(np.asarray(query_coords, dtype=float), k=1)
    max_distance = float(np.max(distances)) if len(np.atleast_1d(distances)) else 0.0
    if max_distance > tolerance:
        raise RuntimeError(
            f"{label} do not match within {tolerance:g}; max distance is {max_distance:.3e}"
        )
    return np.asarray(indices, dtype=np.int64), max_distance




def impose_dirichlet_rows(A: csr_matrix, dofs: np.ndarray) -> csr_matrix:
    A = A.tolil(copy=True)
    for dof in np.asarray(dofs, dtype=np.int64):
        A.rows[int(dof)] = [int(dof)]
        A.data[int(dof)] = [1.0]
    return A.tocsr()




def scalar_mass(values: np.ndarray, mass: csr_matrix) -> float:
    ones = np.ones(len(values), dtype=float)
    return float(ones @ (mass @ np.asarray(values, dtype=float)))




def mass_l2_relative(diff: np.ndarray, ref: np.ndarray, mass: csr_matrix) -> float:
    diff = np.asarray(diff, dtype=float)
    ref = np.asarray(ref, dtype=float)
    abs_sq = float(diff @ (mass @ diff))
    ref_sq = float(ref @ (mass @ ref))
    return float(np.sqrt(max(abs_sq, 0.0)) / max(np.sqrt(max(ref_sq, 0.0)), 1.0e-300))




def mass_l2_absolute(diff: np.ndarray, mass: csr_matrix) -> float:
    diff = np.asarray(diff, dtype=float)
    abs_sq = float(diff @ (mass @ diff))
    return float(np.sqrt(max(abs_sq, 0.0)))




def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2f3640",
            "axes.labelcolor": "#1f2933",
            "xtick.color": "#1f2933",
            "ytick.color": "#1f2933",
            "grid.color": "#d8dee9",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "legend.frameon": False,
        }
    )




def build_tracer_plot_grid(points: np.ndarray, cells: np.ndarray) -> pv.UnstructuredGrid:
    cells_flat = np.hstack([np.full((len(cells), 1), 4, dtype=np.int64), np.asarray(cells, dtype=np.int64)]).ravel()
    celltypes = np.full(len(cells), pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells_flat, celltypes, np.asarray(points, dtype=float))




def render_tracer_isosurface_panel(
    plotter: pv.Plotter,
    grid,
    shell,
    scalars: str,
    title: str,
    scalar_bar_title: str,
    cmap: str,
    clim: tuple[float, float],
) -> None:
    add_tracer_domain_box(plotter, grid.bounds)
    shell_opacity = 0.12 if scalar_bar_title == "c" else 0.30
    plotter.add_mesh(shell, color="#787878", opacity=shell_opacity, smooth_shading=True, show_edges=False)
    scalar_bar_args = {
        "title": scalar_bar_title,
        "vertical": True,
        "position_x": 0.87,
        "position_y": 0.18,
        "width": 0.045,
        "height": 0.60,
        "title_font_size": 15,
        "label_font_size": 12,
        "fmt": "%.1f",
        "color": "black",
    }
    contour = grid.contour(isosurfaces=tracer_iso_levels(clim, scalar_bar_title), scalars=scalars)
    if contour.n_points:
        render_mesh = plot_extract_surface(contour).smooth(
            n_iter=90, relaxation_factor=0.06, feature_smoothing=False, boundary_smoothing=True
        )
        opacity = 0.74 if scalar_bar_title == "c" else 0.60
    else:
        render_mesh = grid
        opacity = 0.55
    plotter.add_mesh(
        render_mesh, scalars=scalars, cmap=cmap, clim=clim, opacity=opacity,
        smooth_shading=True, show_edges=False, scalar_bar_args=scalar_bar_args,
    )
    plotter.add_text(title, position=(0.035, 0.925), font_size=14, color="black", viewport=True)
    set_tracer_paper_camera(plotter, grid.bounds)




def tracer_iso_levels(clim: tuple[float, float], scalar_bar_title: str = "") -> np.ndarray:
    lo, hi = clim
    if hi <= lo:
        hi = lo + 1.0
    if scalar_bar_title == "c":
        return np.linspace(0.25, 0.95, 6)
    return np.linspace(lo + 0.24 * (hi - lo), hi - 0.10 * (hi - lo), 5)




def tracer_error_range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if len(finite) == 0:
        return -16.0, -8.0
    lo = float(np.percentile(finite, 5.0))
    hi = float(np.percentile(finite, 99.2))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi




def plot_extract_surface(dataset):
    try:
        return dataset.extract_surface(algorithm="dataset_surface")
    except TypeError:
        return dataset.extract_surface()




def add_tracer_domain_box(plotter: pv.Plotter, bounds: tuple) -> None:
    cube = plot_extract_surface(pv.Cube(bounds=bounds))
    plotter.add_mesh(cube, color="#eeeeee", opacity=0.08, show_edges=False)




def set_tracer_paper_camera(plotter: pv.Plotter, bounds: tuple) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))
    scale = max(xmax - xmin, ymax - ymin, zmax - zmin)
    plotter.camera_position = [
        (center[0] + 4.10 * scale, center[1] - 4.00 * scale, center[2] + 3.10 * scale),
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.45)
    plotter.add_light(pv.Light(position=(center[0], center[1] - 2.0 * scale, center[2] + 4.0 * scale), intensity=0.55))
    plotter.add_light(pv.Light(position=(center[0] - 2.0 * scale, center[1] + 1.5 * scale, center[2] + 2.5 * scale), intensity=0.35))
# ---------------------------------------------------------------------------
# Two-phase saturation solver
# ---------------------------------------------------------------------------


def solve_two_phase(
    msh,
    u_vertices: np.ndarray,
    diffusivity: float = 0.0,
    porosity: float = 1.0,
    dt: float = 0.1,
    t_final: float = 30.0,
    sw_initial: float = 0.2,
    sw_inlet: float | None = None,
    corey: dict[str, float] | None = None,
    picard_max_iters: int = 6,
    picard_tol: float = 1.0e-6,
    picard_relaxation: float = 1.0,
    supg: bool = True,
    supg_factor: float = 0.5,
    snapshot_every: int | None = 10,
) -> dict[str, object]:
    """Solve the Buckley--Leverett saturation equation on *msh* with total
    velocity *u_vertices* (P1 vertex values, topology-vertex ordering).

    Returns the time history (water cut, recovery, mass bookkeeping,
    saturation bounds, Picard diagnostics), the final saturation on dofs and
    vertices, the mass matrix for reference-error metrics, and — if
    ``snapshot_every`` is set — the vertex-interpolated saturation field
    every ``snapshot_every`` steps (``snapshot_times`` /
    ``snapshot_saturation_vertices``).
    """
    corey = dict(DEFAULT_COREY, **(corey or {}))
    swr, sor = float(corey["swr"]), float(corey["sor"])
    nw, no = float(corey["nw"]), float(corey["no"])
    mu_w, mu_o = float(corey["mu_w"]), float(corey["mu_o"])
    validate_corey(swr, sor, nw, no, mu_w, mu_o)
    lower, upper = swr, 1.0 - sor
    if sw_inlet is None:
        sw_inlet = upper
    sw_inlet = float(sw_inlet)
    if porosity <= 0.0 or dt <= 0.0 or t_final < 0.0:
        raise ValueError("porosity and dt must be positive and t_final must be nonnegative")
    if diffusivity < 0.0:
        raise ValueError("diffusivity must be nonnegative")
    if not lower <= sw_initial <= upper:
        raise ValueError(f"sw_initial must lie in [{lower:g}, {upper:g}]")
    if not lower <= sw_inlet <= upper:
        raise ValueError(f"sw_inlet must lie in [{lower:g}, {upper:g}]")
    if picard_max_iters < 1 or not 0.0 < picard_relaxation <= 1.0:
        raise ValueError("invalid Picard iteration settings")

    cell = msh.basix_cell()
    C = fem.functionspace(msh, element("Lagrange", cell, 1))
    U = fem.functionspace(msh, element("Lagrange", cell, 1, shape=(3,)))
    u = fem.Function(U)
    assign_p1_function(u, np.asarray(u_vertices, dtype=float))
    u.x.scatter_forward()

    S_trial = ufl.TrialFunction(C)
    w = ufl.TestFunction(C)
    h = ufl.CellDiameter(msh)
    normal = ufl.FacetNormal(msh)
    dx = ufl.dx(domain=msh)

    mass_matrix = to_scipy_matrix(fem.assemble_matrix(fem.form(S_trial * w * dx))).tocsr()
    mass_weights = np.asarray(mass_matrix @ np.ones(mass_matrix.shape[1]), dtype=float)
    total_volume = float(np.sum(mass_weights))
    initial_water_mass = float(porosity) * total_volume * float(sw_initial)
    initial_oil_mass = float(porosity) * total_volume * (1.0 - float(sw_initial))

    diffusion_matrix = None
    if diffusivity > 0.0:
        diffusion_matrix = to_scipy_matrix(
            fem.assemble_matrix(fem.form(ufl.inner(ufl.grad(S_trial), ufl.grad(w)) * dx))
        ).tocsr()

    # Mark only the pressure inlet/outlet planes.  Sphere and lateral-wall
    # facets remain impermeable and therefore carry no transport flux.
    coords = C.tabulate_dof_coordinates()
    vertex_coords = topology_vertex_coordinates(msh)
    vertex_to_c_dof, concentration_map_distance = nearest_indices(
        vertex_coords, coords, "scalar saturation vertex output coordinates"
    )
    lx = float(np.max(vertex_coords[:, 0]))
    fdim = msh.topology.dim - 1
    inlet_facets = np.asarray(
        dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0)),
        dtype=np.int32,
    )
    outlet_facets = np.asarray(
        dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], lx)),
        dtype=np.int32,
    )
    if len(inlet_facets) == 0 or len(outlet_facets) == 0:
        raise RuntimeError("failed to locate inlet or outlet boundary facets")
    boundary_facets = np.concatenate((inlet_facets, outlet_facets))
    boundary_values = np.concatenate(
        (np.full(len(inlet_facets), 1, dtype=np.int32),
         np.full(len(outlet_facets), 2, dtype=np.int32))
    )
    order = np.argsort(boundary_facets)
    facet_tags = dmesh.meshtags(
        msh, fdim, boundary_facets[order], boundary_values[order]
    )
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    # Coefficients and forms are compiled once; only coefficient arrays and
    # the step-size constant change during the nonlinear iteration.
    fw_fun = fem.Function(C)
    df_fun = fem.Function(C)
    convection_form = fem.form(fw_fun * ufl.dot(u, ufl.grad(w)) * dx)
    un = ufl.dot(u, normal)
    un_pos = ufl.max_value(un, 0.0)
    un_neg = ufl.min_value(un, 0.0)
    fw_inlet = fem.Constant(
        msh, float(fractional_flow(np.asarray([sw_inlet]), swr, sor, nw, no, mu_w, mu_o)[0])
    )
    fw_backflow = fem.Constant(
        msh, float(fractional_flow(np.asarray([sw_initial]), swr, sor, nw, no, mu_w, mu_o)[0])
    )
    inlet_numerical_flux = un_pos * fw_fun + un_neg * fw_inlet
    outlet_numerical_flux = un_pos * fw_fun + un_neg * fw_backflow
    boundary_vector_form = fem.form(
        inlet_numerical_flux * w * ds(1) + outlet_numerical_flux * w * ds(2)
    )
    boundary_net_form = fem.form(
        inlet_numerical_flux * ds(1) + outlet_numerical_flux * ds(2)
    )
    water_in_form = fem.form(
        (-un_neg * fw_inlet) * ds(1) + (-un_neg * fw_backflow) * ds(2)
    )
    water_out_form = fem.form(
        (un_pos * fw_fun) * ds(1) + (un_pos * fw_fun) * ds(2)
    )
    outlet_water_form = fem.form(un_pos * fw_fun * ds(2))
    outlet_total_form = fem.form(un_pos * ds(2))

    dt_constant = fem.Constant(msh, float(dt))
    phi_constant = fem.Constant(msh, float(porosity))
    characteristic_speed = ufl.sqrt(
        df_fun * df_fun * ufl.inner(u, u) + fem.Constant(msh, 1.0e-30)
    )
    tau = fem.Constant(msh, float(supg_factor)) / ufl.sqrt(
        (2.0 * phi_constant / dt_constant) ** 2
        + (2.0 * characteristic_speed / h) ** 2
        + fem.Constant(msh, 1.0e-30)
    )
    streamline_test = df_fun * ufl.dot(u, ufl.grad(w))
    supg_mass_form = fem.form(tau * streamline_test * S_trial * dx)
    supg_speed_form = fem.form(
        tau
        * streamline_test
        * df_fun
        * ufl.dot(u, ufl.grad(S_trial))
        * dx
    )

    def assemble_global(form) -> float:
        local = float(fem.assemble_scalar(form))
        return float(msh.comm.allreduce(local, op=MPI.SUM))

    def set_fractional_flow(values: np.ndarray) -> None:
        fw_fun.x.array[:] = fractional_flow(values, swr, sor, nw, no, mu_w, mu_o)
        fw_fun.x.scatter_forward()

    outlet_volume_rate = assemble_global(outlet_total_form)
    if outlet_volume_rate <= 1.0e-14:
        raise RuntimeError("the supplied velocity has no positive outlet flux")

    cold = np.full(C.dofmap.index_map.size_local * C.dofmap.index_map_bs, sw_initial, dtype=float)
    cold_vec = cold.copy()
    set_fractional_flow(cold)

    times = [0.0]
    watercuts = [assemble_global(outlet_water_form) / outlet_volume_rate]
    masses = [float(porosity) * scalar_mass(cold, mass_matrix)]
    recoveries = [recovery_factor(masses[0], initial_water_mass, initial_oil_mass)]
    budget_masses = [masses[0]]
    raw_masses = [masses[0]]
    boundary_net_rates = [0.0]
    water_in_rates = [0.0]
    water_out_rates = [0.0]
    pore_volumes_injected = [0.0]
    mins = [float(np.min(cold))]
    maxs = [float(np.max(cold))]
    raw_mins = [float(np.min(cold))]
    raw_maxs = [float(np.max(cold))]
    raw_below_counts = [int(np.count_nonzero(cold < lower - 1.0e-12))]
    raw_above_counts = [int(np.count_nonzero(cold > upper + 1.0e-12))]
    picard_counts = [0]
    picard_converged = [True]
    absolute_balance_residual = [0.0]
    relative_balance_residual = [0.0]
    cumulative_budget_residual = [0.0]
    limiter_mass_residual = [0.0]

    n_steps = int(np.ceil(t_final / dt))
    snapshot_times = [0.0]
    snapshot_vertices = [np.asarray(cold[vertex_to_c_dof], dtype=float).copy()]
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
        water_in_rate_used = 0.0
        water_out_rate_used = 0.0
        # Performance: the SUPG mass form depends only on tau, u, w and
        # S_trial (assemble once per time step), and the SUPG characteristic
        # speed is frozen at fw'(S^n) for the whole step (frozen-coefficient
        # stabilization; the residual fractional-flow term fw(S^k) is still
        # updated every Picard iteration).  This turns the per-step cost into
        # one matrix assembly + one factorization plus cheap backsolves.
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
            # Conservative weak form with upwind exterior states at the
            # pressure inlet/outlet.  Keeping the boundary vector is what
            # makes the global water inventory follow the boundary flux.
            rhs = rhs + fem.assemble_vector(convection_form).array
            rhs = rhs - fem.assemble_vector(boundary_vector_form).array
            net_boundary_rate_used = assemble_global(boundary_net_form)
            water_in_rate_used = assemble_global(water_in_form)
            water_out_rate_used = assemble_global(water_out_form)
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
        craw = np.asarray(candidate, dtype=float)

        raw_integral = scalar_mass(craw, mass_matrix)
        raw_water_mass = float(porosity) * raw_integral
        raw_masses.append(raw_water_mass)
        raw_mins.append(float(np.min(craw)))
        raw_maxs.append(float(np.max(craw)))
        raw_below_counts.append(int(np.count_nonzero(craw < lower - 1.0e-12)))
        raw_above_counts.append(int(np.count_nonzero(craw > upper + 1.0e-12)))

        limited, limiter_info = conservative_bounded_limiter(
            craw,
            mass_weights,
            target_mass=raw_integral,
            lower=lower,
            upper=upper,
        )
        cold_vec = limited
        step_mass = float(porosity) * scalar_mass(cold_vec, mass_matrix)

        budget_masses.append(budget_masses[-1] - dt_step * net_boundary_rate_used)
        balance_abs, balance_rel = mass_balance_residual(
            step_mass,
            masses[-1],
            dt_step,
            net_boundary_rate_used,
            water_in_rate_used,
            water_out_rate_used,
        )

        current_time += dt_step
        set_fractional_flow(cold_vec)
        times.append(current_time)
        watercuts.append(assemble_global(outlet_water_form) / outlet_volume_rate)
        masses.append(step_mass)
        recoveries.append(recovery_factor(step_mass, initial_water_mass, initial_oil_mass))
        boundary_net_rates.append(net_boundary_rate_used)
        water_in_rates.append(water_in_rate_used)
        water_out_rates.append(water_out_rate_used)
        pore_volumes_injected.append(
            pore_volumes_injected[-1]
            + dt_step * water_in_rate_used / max(float(porosity) * total_volume, 1.0e-300)
        )
        mins.append(float(np.min(cold_vec)))
        maxs.append(float(np.max(cold_vec)))
        absolute_balance_residual.append(balance_abs)
        relative_balance_residual.append(balance_rel)
        cumulative_budget_residual.append(step_mass - budget_masses[-1])
        limiter_mass_residual.append(float(porosity) * float(limiter_info["mass_residual"]))
        picard_counts.append(iters_used)
        picard_converged.append(converged)
        if snapshot_every is not None and step % snapshot_every == 0:
            snapshot_times.append(current_time)
            snapshot_vertices.append(np.asarray(cold_vec[vertex_to_c_dof], dtype=float).copy())
        cold = cold_vec

    return {
        "history": {
            "time": np.asarray(times),
            "watercut": np.asarray(watercuts),
            "recovery": np.asarray(recoveries),
            "mass": np.asarray(masses),
            "budget_mass": np.asarray(budget_masses),
            "raw_mass": np.asarray(raw_masses),
            "boundary_water_net_outflow": np.asarray(boundary_net_rates),
            "water_inflow_rate": np.asarray(water_in_rates),
            "water_outflow_rate": np.asarray(water_out_rates),
            "pore_volumes_injected": np.asarray(pore_volumes_injected),
            "mass_balance_absolute_residual": np.asarray(absolute_balance_residual),
            "mass_balance_relative_residual": np.asarray(relative_balance_residual),
            "cumulative_budget_residual": np.asarray(cumulative_budget_residual),
            "limiter_mass_residual": np.asarray(limiter_mass_residual),
            "min_s": np.asarray(mins),
            "max_s": np.asarray(maxs),
            "raw_min_s_before_limiter": np.asarray(raw_mins),
            "raw_max_s_before_limiter": np.asarray(raw_maxs),
            "raw_below_lower_bound_before_limiter": np.asarray(raw_below_counts),
            "raw_above_upper_bound_before_limiter": np.asarray(raw_above_counts),
            "picard_iterations": np.asarray(picard_counts),
            "picard_converged": np.asarray(picard_converged, dtype=bool),
        },
        "final_saturation": cold_vec,
        "final_saturation_vertices": np.asarray(cold_vec[vertex_to_c_dof], dtype=float),
        "mass_matrix": mass_matrix,
        "outlet_flux_weight": outlet_volume_rate,
        "final_mass": masses[-1],
        "min_mass": min(masses),
        "max_mass": max(masses),
        "final_min": mins[-1],
        "final_max": maxs[-1],
        "final_below_lower_bound": int(np.count_nonzero(cold_vec < lower - 1.0e-12)),
        "final_above_upper_bound": int(np.count_nonzero(cold_vec > upper + 1.0e-12)),
        "raw_final_min_before_limiter": raw_mins[-1],
        "raw_final_max_before_limiter": raw_maxs[-1],
        "raw_final_below_lower_bound_before_limiter": raw_below_counts[-1],
        "raw_final_above_upper_bound_before_limiter": raw_above_counts[-1],
        "limiter_final_mass_residual": limiter_mass_residual[-1],
        "max_picard_iterations": int(np.max(picard_counts)),
        "nonconverged_picard_steps": int(np.count_nonzero(~np.asarray(picard_converged))),
        "saturation_bounds": (lower, upper),
        "concentration_vertex_mapping_max_distance": concentration_map_distance,
        "snapshot_times": np.asarray(snapshot_times),
        "snapshot_saturation_vertices": np.asarray(snapshot_vertices),
    }


# ---------------------------------------------------------------------------
# Reference-error bookkeeping
# ---------------------------------------------------------------------------


def add_two_phase_reference_errors(
    rows: list[dict[str, object]],
    histories: dict[str, dict[str, np.ndarray]],
    final_dof_fields: dict[str, np.ndarray],
    mass_matrix: csr_matrix,
    reference: str = "FEM",
) -> None:
    ref_curve = histories[reference]["watercut"]
    ref_time = histories[reference]["time"]
    ref_field = final_dof_fields[reference]
    ref_row = next(row for row in rows if row["method"] == reference)

    def time_l2_relative(signal: np.ndarray, ref: np.ndarray) -> float:
        numerator = float(np.trapezoid((signal - ref) ** 2, ref_time))
        denominator = float(np.trapezoid(ref ** 2, ref_time))
        if denominator <= 1.0e-24:
            return float("nan")
        return float(np.sqrt(max(numerator, 0.0) / denominator))

    for row in rows:
        method = str(row["method"])
        signal = histories[method]["watercut"]
        field_diff = final_dof_fields[method] - ref_field
        row["watercut_rel_l2_error"] = time_l2_relative(signal, ref_curve)
        row["watercut_abs_l1_error"] = float(np.trapezoid(np.abs(signal - ref_curve), ref_time))
        row["saturation_rel_l2_fem_integral_vs_fem"] = mass_l2_relative(
            field_diff, ref_field, mass_matrix
        )
        row["saturation_linf_abs_vs_fem"] = float(np.linalg.norm(field_diff, ord=np.inf))
        row["recovery_final_error_vs_fem"] = float(
            row["final_recovery"] - float(ref_row["final_recovery"])
        )
        row["recovery_rel_l2_error"] = time_l2_relative(
            histories[method]["recovery"], histories[reference]["recovery"]
        )
        row["watercut_t50_error"] = signed_time_error(row["watercut_t50"], ref_row["watercut_t50"])
        row["watercut_t90_error"] = signed_time_error(row["watercut_t90"], ref_row["watercut_t90"])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_watercut_curves(out: Path, histories: dict[str, dict[str, np.ndarray]]) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=220)
    for method, history in histories.items():
        ax.plot(
            history["time"], history["watercut"],
            color=COLORS.get(method, "#555555"),
            linestyle="-" if method == "FEM" else "--",
            linewidth=2.0, label=method,
        )
    ax.set_xlabel("time")
    ax.set_ylabel("flux-weighted outlet water cut")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.7)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_recovery_curves(out: Path, histories: dict[str, dict[str, np.ndarray]]) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=220)
    for method, history in histories.items():
        ax.plot(
            history["time"], history["recovery"],
            color=COLORS.get(method, "#555555"),
            linestyle="-" if method == "FEM" else "--",
            linewidth=2.0, label=method,
        )
    ax.set_xlabel("time")
    ax.set_ylabel("oil recovery factor")
    ax.grid(True, alpha=0.7)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_mass_balance_validation(out: Path, histories: dict[str, dict[str, np.ndarray]]) -> None:
    """Plot the independently accumulated flux budget and its residual."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=220)
    for method, history in histories.items():
        color = COLORS.get(method, "#555555")
        style = "-" if method == "FEM" else "--"
        axes[0].plot(history["time"], history["mass"], color=color,
                     linestyle=style, linewidth=2.0, label=f"{method} inventory")
        axes[0].plot(history["time"], history["budget_mass"], color=color,
                     linestyle=":", linewidth=1.2, alpha=0.8)
        axes[1].semilogy(
            history["time"],
            np.maximum(np.abs(history["cumulative_budget_residual"]), 1.0e-18),
            color=color, linestyle=style, linewidth=2.0, label=method,
        )
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("water inventory; dotted = flux budget")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("absolute cumulative balance residual")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.7)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_two_phase_error_summary(out: Path, rows: list[dict[str, object]]) -> None:
    set_plot_style()
    methods = [str(row["method"]) for row in rows if row["method"] != "FEM"]
    if not methods:
        return
    saturation = [float(row["saturation_rel_l2_fem_integral_vs_fem"]) for row in rows if row["method"] != "FEM"]
    watercut = [float(row["watercut_rel_l2_error"]) for row in rows if row["method"] != "FEM"]
    recovery = [float(row["recovery_rel_l2_error"]) for row in rows if row["method"] != "FEM"]
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=220)
    ax.bar(x - width, saturation, width, label="saturation field L2", color="#457b9d")
    watercut_plot = [value if np.isfinite(value) and value > 0.0 else np.nan for value in watercut]
    ax.bar(x, watercut_plot, width, label="water cut L2", color="#e76f51")
    ax.bar(x + width, recovery, width, label="recovery L2", color="#2a9d8f")
    ax.set_yscale("log")
    finite_values = [
        value for value in saturation + watercut + recovery
        if np.isfinite(value) and value > 0.0
    ]
    upper = max(finite_values, default=1.0e-1)
    ax.set_ylim(1.0e-8, max(1.0e-1, 2.0 * upper))
    ax.set_xticks(x, methods)
    ax.set_ylabel("relative error (log)")
    ax.grid(True, axis="y", which="both", alpha=0.7)
    ax.legend(ncols=3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_final_saturation(
    out: Path,
    points: np.ndarray,
    cells: np.ndarray,
    final_fields: dict[str, np.ndarray],
    methods: list[str],
    reference: str = "FEM",
) -> None:
    """Isosurface panels of the final saturation per method plus the
    log10 absolute error against the FEM-driven reference."""
    panels = list(methods)
    if "Affine-DDPNM-9" in final_fields and reference in final_fields:
        panels.append("log10 error")

    grid = build_tracer_plot_grid(points, cells)
    all_saturation = np.concatenate(
        [np.asarray(final_fields[method], dtype=float) for method in methods]
    )
    saturation_clim = (float(np.min(all_saturation)), float(np.max(all_saturation)))
    if saturation_clim[1] <= saturation_clim[0]:
        saturation_clim = (saturation_clim[0], saturation_clim[0] + 1.0)
    for method in methods:
        grid.point_data[f"saturation_{method}"] = np.asarray(final_fields[method], dtype=float)
    if "Affine-DDPNM-9" in final_fields and reference in final_fields:
        error = np.abs(
            np.asarray(final_fields["Affine-DDPNM-9"], dtype=float)
            - np.asarray(final_fields[reference], dtype=float)
        )
        grid.point_data["log10_saturation_error"] = np.log10(np.maximum(error, 1.0e-16))
    shell = plot_extract_surface(grid).smooth(
        n_iter=50, relaxation_factor=0.06, feature_smoothing=False, boundary_smoothing=True
    )

    pv.global_theme.font.family = "arial"
    plotter = pv.Plotter(off_screen=True, window_size=(640 * len(panels), 820), shape=(1, len(panels)), border=False)
    plotter.set_background("white")
    for index, panel in enumerate(panels):
        plotter.subplot(0, index)
        if panel == "log10 error":
            render_tracer_isosurface_panel(
                plotter, grid, shell, "log10_saturation_error", "Affine-DDPNM-9 error",
                "log10 error", "turbo",
                tracer_error_range(grid.point_data["log10_saturation_error"]),
            )
        else:
            render_tracer_isosurface_panel(
                plotter, grid, shell, f"saturation_{panel}", panel, "S", "viridis", saturation_clim,
            )
    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()


# ---------------------------------------------------------------------------
# CSV output (two-phase history)
# ---------------------------------------------------------------------------


def write_two_phase_history_csv(path: Path, histories: dict[str, dict[str, np.ndarray]]) -> None:
    rows: list[dict[str, object]] = []
    for method, history in histories.items():
        for i, t in enumerate(history["time"]):
            rows.append(
                {
                    "method": method,
                    "time": float(t),
                    "watercut": float(history["watercut"][i]),
                    "recovery": float(history["recovery"][i]),
                    "mass": float(history["mass"][i]),
                    "budget_mass": float(history["budget_mass"][i]),
                    "raw_mass": float(history["raw_mass"][i]),
                    "boundary_water_net_outflow": float(history["boundary_water_net_outflow"][i]),
                    "water_inflow_rate": float(history["water_inflow_rate"][i]),
                    "water_outflow_rate": float(history["water_outflow_rate"][i]),
                    "pore_volumes_injected": float(history["pore_volumes_injected"][i]),
                    "mass_balance_absolute_residual": float(history["mass_balance_absolute_residual"][i]),
                    "mass_balance_relative_residual": float(history["mass_balance_relative_residual"][i]),
                    "cumulative_budget_residual": float(history["cumulative_budget_residual"][i]),
                    "min_s": float(history["min_s"][i]),
                    "max_s": float(history["max_s"][i]),
                    "raw_min_s_before_limiter": float(history["raw_min_s_before_limiter"][i]),
                    "raw_max_s_before_limiter": float(history["raw_max_s_before_limiter"][i]),
                    "raw_below_lower_bound_before_limiter": int(history["raw_below_lower_bound_before_limiter"][i]),
                    "raw_above_upper_bound_before_limiter": int(history["raw_above_upper_bound_before_limiter"][i]),
                    "picard_iterations": int(history["picard_iterations"][i]),
                    "picard_converged": bool(history["picard_converged"][i]),
                    "limiter_mass_residual": float(history["limiter_mass_residual"][i]),
                }
            )
    write_csv(path, rows)

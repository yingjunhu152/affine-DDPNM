"""Six-arm random-27 experiment orchestration."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import Arm, Coupling, FlowMethod, Numerics, Physics
from .flow import FlowResult, build_flow_solver
from .geometry import PoreProjector, build_bentheimer, build_random27, enable_flow_dependencies
from .metrics import scalar_l2_relative, vector_l2_relative
from .transport import CahnHilliardTransport


@dataclass
class ArmResult:
    arm: str
    steps: int
    converged: bool
    wall_seconds: float
    flow_offline_seconds: float
    flow_seconds: float
    transport_seconds: float
    flow_solves: int
    newton_iterations: int
    sfi_iterations: int
    outlet_flux: float
    mass: float
    free_energy: float
    phi_min: float
    phi_max: float
    flow_global_unknowns: int
    max_flow_linear_residual: float
    flow_stabilization: str
    phi_l2_error_vs_fem: float = float("nan")
    velocity_l2_error_vs_fem: float = float("nan")


@dataclass
class _ArmFields:
    result: ArmResult
    phi: np.ndarray
    velocity: np.ndarray
    history: list[dict]


def _run_arm(arm, solver, transport, projector, physics, numerics) -> _ArmFields:
    started = time.perf_counter()
    phi = transport.initial_vertices()
    flow_seconds = 0.0
    transport_seconds = 0.0
    flow_solves = 0
    newton_total = 0
    sfi_total = 0
    max_flow_residual = 0.0
    history: list[dict] = []
    flow: FlowResult | None = None

    if arm.coupling is Coupling.FROZEN:
        pore_mu = projector.viscosity(phi, physics.viscosity_water, physics.viscosity_oil)
        flow = solver.solve(pore_mu)
        flow_seconds += flow.wall_seconds
        flow_solves += 1
        max_flow_residual = flow.relative_linear_residual

    for step in range(1, numerics.steps + 1):
        old_phi = phi.copy()
        if arm.coupling is Coupling.FROZEN:
            t0 = time.perf_counter()
            phi, diag = transport.advance(old_phi, flow.vertex_velocity)
            transport_seconds += time.perf_counter() - t0
            sfi_iterations = 0
            newton_total += diag.newton_iterations
        else:
            iterate = old_phi.copy()
            converged = False
            for outer in range(1, numerics.sfi_max_iterations + 1):
                pore_mu = projector.viscosity(
                    iterate, physics.viscosity_water, physics.viscosity_oil
                )
                flow = solver.solve(pore_mu)
                flow_seconds += flow.wall_seconds
                flow_solves += 1
                max_flow_residual = max(max_flow_residual, flow.relative_linear_residual)
                t0 = time.perf_counter()
                candidate, diag = transport.advance(
                    old_phi, flow.vertex_velocity, initial_guess_vertices=iterate
                )
                transport_seconds += time.perf_counter() - t0
                newton_total += diag.newton_iterations
                # The first solve is a time-step predictor, not a coupling
                # correction.  Damping it would make the physical time-step
                # change masquerade as a Picard error and create a long,
                # meaningless geometric tail.  Subsequent corrections use
                # the configured SFI relaxation.
                relaxed = candidate if outer == 1 else (
                    numerics.sfi_relaxation * candidate
                    + (1.0 - numerics.sfi_relaxation) * iterate
                )
                update = float(np.max(np.abs(relaxed - iterate)))
                iterate = relaxed
                if outer > 1 and update <= numerics.sfi_tolerance:
                    converged = True
                    break
            if not converged:
                raise RuntimeError(
                    f"{arm.name} SFI failed at step {step}; update={update:.3e}"
                )
            phi = candidate
            sfi_iterations = outer
            sfi_total += outer
        history.append({
            "step": step,
            "time": step * numerics.dt,
            "newton_iterations": diag.newton_iterations,
            "newton_residual_inf": diag.residual_inf,
            "line_search_reductions": diag.line_search_reductions,
            "sfi_iterations": sfi_iterations,
            "mass": diag.mass,
            "free_energy": diag.free_energy,
            "phi_min": diag.phi_min,
            "phi_max": diag.phi_max,
            "outlet_flux": flow.outlet_flux,
        })
        print(
            f"    {arm.name}: step {step}/{numerics.steps}, "
            f"Newton={diag.newton_iterations}, SFI={sfi_iterations}, "
            f"phi=[{diag.phi_min:.3f},{diag.phi_max:.3f}]",
            flush=True,
        )

    result = ArmResult(
        arm=arm.name,
        steps=numerics.steps,
        converged=True,
        wall_seconds=time.perf_counter() - started,
        flow_offline_seconds=float(getattr(solver, "offline_seconds", 0.0)),
        flow_seconds=flow_seconds,
        transport_seconds=transport_seconds,
        flow_solves=flow_solves,
        newton_iterations=newton_total,
        sfi_iterations=sfi_total,
        outlet_flux=flow.outlet_flux,
        mass=diag.mass,
        free_energy=diag.free_energy,
        phi_min=diag.phi_min,
        phi_max=diag.phi_max,
        flow_global_unknowns=flow.global_unknowns,
        max_flow_linear_residual=max_flow_residual,
        flow_stabilization=flow.stabilization,
    )
    return _ArmFields(result, phi.copy(), flow.vertex_velocity.copy(), history)


def run_six_geometry(
    geometry: str,
    arms: tuple[Arm, ...],
    physics: Physics,
    numerics: Numerics,
    out_dir: Path,
    mesh_file: Path | None = None,
) -> dict:
    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. Choose a new directory; old results are never overwritten."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if geometry == "random27":
        print("[geometry] building random-27 partition", flush=True)
        partition = build_random27(mesh_file=mesh_file or out_dir / "random27.msh")
        geometry_label = "random-27 sphere porous medium"
    elif geometry == "bentheimer":
        if mesh_file is None:
            raise ValueError("Bentheimer requires an existing --mesh-file")
        print(f"[geometry] loading inverted Bentheimer partition: {mesh_file}", flush=True)
        partition = build_bentheimer(mesh_file)
        geometry_label = "phase-inverted Bentheimer, Cartesian 3x3x3 partition"
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    projector = PoreProjector.from_partition(partition)
    requested_methods = tuple(dict.fromkeys(arm.flow for arm in arms))
    solvers = {}
    for method in requested_methods:
        print(f"[offline] preparing {method.value} flow solver", flush=True)
        solvers[method] = build_flow_solver(partition, physics, numerics, method)

    fields: dict[str, _ArmFields] = {}
    for arm in arms:
        print(f"[run] {arm.name}", flush=True)
        transport = CahnHilliardTransport(partition.mesh, physics, numerics)
        fields[arm.name] = _run_arm(
            arm, solvers[arm.flow], transport, projector, physics, numerics
        )

    for coupling in Coupling:
        fem_name = Arm(FlowMethod.FEM, coupling).name
        reference = fields.get(fem_name)
        if reference is None:
            continue
        for arm in arms:
            if arm.coupling is not coupling:
                continue
            current = fields[arm.name]
            current.result.phi_l2_error_vs_fem = scalar_l2_relative(
                current.phi, reference.phi, projector
            )
            current.result.velocity_l2_error_vs_fem = vector_l2_relative(
                current.velocity, reference.velocity, projector
            )

    rows = [asdict(fields[arm.name].result) for arm in arms]
    prefix = geometry
    with (out_dir / f"{prefix}_six_arms.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    enable_flow_dependencies()
    from ddpnm_core.io import topology_vertex_coordinates
    coordinates = topology_vertex_coordinates(partition.mesh)
    for arm in arms:
        current = fields[arm.name]
        np.savez_compressed(
            out_dir / f"{arm.name}_final.npz",
            phi=current.phi,
            velocity=current.velocity,
            coordinates=coordinates,
            cells=projector.cell_vertices,
            cell_labels=projector.labels,
        )
        with (out_dir / f"{arm.name}_history.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(current.history[0]))
            writer.writeheader()
            writer.writerows(current.history)
    report = {
        "geometry": geometry_label,
        "source_mesh": str(mesh_file.resolve()) if mesh_file is not None else str((out_dir / "random27.msh").resolve()),
        "transport_protocol": "one shared global conforming P1-P1 convex-splitting Cahn-Hilliard discretization",
        "flow_protocol": "Taylor-Hood FEM or local-response DDPNM; SFI updates porewise viscosity",
        "physics": physics.as_dict(),
        "numerics": numerics.as_dict(),
        "mesh": {
            "cells": int(len(partition.cell_labels)),
            "pores": int(len(projector.pore_volumes)),
            "interfaces": int(len(partition.interface_pairs)),
        },
        "results": rows,
    }
    (out_dir / f"{prefix}_six_arms.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def run_random27(arms, physics, numerics, out_dir, mesh_file=None) -> dict:
    return run_six_geometry(
        "random27", arms, physics, numerics, out_dir, mesh_file
    )


def run_bentheimer(arms, physics, numerics, out_dir, mesh_file) -> dict:
    return run_six_geometry(
        "bentheimer", arms, physics, numerics, out_dir, mesh_file
    )

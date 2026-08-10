#!/usr/bin/env python3
"""Affine-DDPNM driven single-phase tracer validation on the random-27 medium.

Pipeline (combining the two archived projects):

1. random-27 sphere partition mesh (frozen ``SPHERES`` packing, Voronoi
   throat faces, gmsh fragmentation) — geometry from ``affine_ddpnm_3d_random_porous``;
2. monolithic Taylor-Hood P2-P1 FEM Stokes reference — ``ddpnm_core`` solver;
3. Classic-DDPNM-1 / NormalLinear-DDPNM-3 (W1n) / Affine-DDPNM-9 Stokes
   solutions on the same mesh — nine-mode affine interface basis;
4. each Stokes velocity (P1 vertex values, two-sided interface average for
   DDPNM) drives the same transient tracer advection--diffusion solver —
   ported from ``stokes_tracer_hoddpnm`` (tracer_transport.py);
5. tracer metrics against the FEM-driven reference: breakthrough curve,
   final concentration field, mass balance, crossing times.

Run in the FEniCSx environment:

    conda run -n fenicsx --no-capture-output python run_affine_ddpnm_tracer.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from random_porous import build_partition
from affine_face_basis import (
    AffineFaceBasis,
    CompatibleClassicP0Basis,
    NormalLinearFaceBasis,
)
from ddpnm_core.assembler import InterfaceAssembler
from ddpnm_core.fem_utils import solve_reference
from ddpnm_core.io import topology_arrays
from ddpnm_core.library import build_response_library
from ddpnm_core.validation import finite_element_error_analysis
from ddpnm3d.solver import DdpnmSolution, LocalResponse, build_modes
from postprocess.fields import mixed_solution_to_p1, reconstruct_parent_vertices

import tracer_transport as tracer

METHODS = ("FEM", "Classic-DDPNM-1", "NormalLinear-DDPNM-3", "Affine-DDPNM-9")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Mesh (same defaults as the archived random-27 benchmark)
    parser.add_argument("--mesh-size", type=float, default=0.13)
    parser.add_argument("--sphere-size", type=float, default=0.065)
    parser.add_argument("--boundary-size", type=float, default=0.085)
    parser.add_argument("--interface-size", type=float, default=0.075)
    parser.add_argument("--sphere-band", type=float, default=0.15)
    parser.add_argument("--boundary-band", type=float, default=0.13)
    parser.add_argument("--interface-band", type=float, default=0.12)
    # Stokes
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--inlet-pressure", type=float, default=1.0)
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--pressure-stabilization", type=float, default=0.0)
    # Tracer
    parser.add_argument("--diffusivity", type=float, default=0.05)
    parser.add_argument("--porosity", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--t-final", type=float, default=30.0)
    parser.add_argument("--supg", action="store_true")
    parser.add_argument("--supg-factor", type=float, default=0.50)
    # Output
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "outputs" / "benchmark_tracer")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# DDPNM solution conversion (same as run_random_benchmark.py)
# ---------------------------------------------------------------------------


def reduced_solution(partition, library, system) -> DdpnmSolution:
    """Convert the core system to the 3-D DdpnmSolution API."""
    keys = system.global_keys
    key_to_dof = {key: dof for dof, key in enumerate(keys)}
    local_responses: list[LocalResponse] = []
    for entry in library.entries:
        matrix = entry.primitive_G
        scale = max(float(np.linalg.norm(matrix)), 1.0e-30)
        local_responses.append(
            LocalResponse(
                pore_id=int(entry.operator.pore_id),
                submesh=entry.operator.submesh,
                parent_cell_map=entry.operator.parent_cell_map,
                parent_vertex_map=entry.operator.parent_vertex_map,
                ports=entry.operator.ports,
                modes=build_modes(entry.operator.ports),
                W=entry.operator.W,
                G=matrix,
                responses=entry.primitive_responses,
                ndofs=entry.operator.ndofs,
                symmetry_error=entry.symmetry_error,
                kernel_error=float(
                    np.linalg.norm(matrix @ np.ones(matrix.shape[0])) / scale
                ),
            )
        )
    n_interfaces = len(partition.interface_pairs)
    return DdpnmSolution(
        interface_pressures=np.asarray(
            [
                system.coefficients[key_to_dof[(iid, "normal", "P0")]]
                for iid in range(n_interfaces)
            ]
        ),
        schur_matrix=system.schur_matrix,
        rhs=system.rhs,
        local_responses=local_responses,
        local_solutions=system.local_solutions,
        interface_flux_sums=np.asarray(
            [
                system.moment_residuals[key_to_dof[(iid, "normal", "P0")]]
                for iid in range(n_interfaces)
            ]
        ),
        boundary_fluxes=system.boundary_fluxes,
        min_schur_eigenvalue=system.min_schur_eigenvalue,
        max_mass_residual=float(np.max(np.abs(system.moment_residuals))),
    )


# ---------------------------------------------------------------------------
# Stokes metrics (archived convention: exact FE-integral broken-domain L2)
# ---------------------------------------------------------------------------


def tetrahedron_volumes(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    xyz = points[tetrahedra]
    matrices = np.stack(
        (xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0]),
        axis=1,
    )
    return np.abs(np.linalg.det(matrices)) / 6.0


def ddpnm_stokes_metrics(partition, solution, reference, volumes: np.ndarray) -> dict[str, float]:
    """Same metrics as the archived W1n benchmark
    (finite_element_error_analysis: exact P2-P1 integrals per pore,
    global-mean aligned pressure)."""
    metric, _u_cell, _p_cell = finite_element_error_analysis(
        partition, solution, reference, volumes
    )
    return {
        "velocity_relative_l2_error_vs_fem": float(metric["velocity_relative_l2"]),
        "velocity_relative_broken_h1_vs_fem": float(metric["velocity_relative_broken_h1_seminorm"]),
        "pressure_raw_relative_l2_error_vs_fem": float(metric["pressure_raw_relative_l2"]),
        "pressure_mean_aligned_relative_l2_error_vs_fem": float(metric["pressure_mean_aligned_relative_l2"]),
        "outlet_flux_relative_error_vs_fem": float(metric["outlet_flux_relative_error"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, dict[str, float]] = {}

    print("[1/7] Building the random-27 partition mesh ...")
    t0 = time.perf_counter()
    partition = build_partition(
        mesh_size=args.mesh_size,
        sphere_size=args.sphere_size,
        boundary_size=args.boundary_size,
        interface_size=args.interface_size,
        sphere_band=args.sphere_band,
        boundary_band=args.boundary_band,
        interface_band=args.interface_band,
        mesh_file=args.out_dir / "random_sphere_partition.msh",
    )
    mesh_seconds = time.perf_counter() - t0
    points, tetrahedra = topology_arrays(partition.mesh)
    n_interfaces = len(partition.interface_pairs)
    n_cells = len(tetrahedra)
    print(f"      tetrahedra={n_cells}, pores={len(np.unique(partition.cell_labels))}, interfaces={n_interfaces}")

    print("[2/7] Monolithic Taylor-Hood FEM Stokes reference ...")
    t0 = time.perf_counter()
    reference = solve_reference(
        partition.mesh,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
    )
    fem_seconds = time.perf_counter() - t0
    timings["FEM"] = {"offline_seconds": 0.0, "online_seconds": fem_seconds}
    u_fem, p_fem = mixed_solution_to_p1(reference.W, reference.solution)
    volumes = tetrahedron_volumes(points, tetrahedra)
    print(f"      dofs={reference.ndofs}, solve={fem_seconds:.3f} s, outlet flux={reference.boundary_fluxes['outlet']:.6e}")

    print("[3/7] Tracer driven by the FEM velocity ...")
    t0 = time.perf_counter()
    tracer_fem = tracer.solve_tracer(
        partition.mesh, u_fem,
        diffusivity=args.diffusivity, porosity=args.porosity,
        dt=args.dt, t_final=args.t_final, supg=args.supg, supg_factor=args.supg_factor,
    )
    timings["FEM"]["tracer_seconds"] = time.perf_counter() - t0
    print(f"      tracer={timings['FEM']['tracer_seconds']:.3f} s, "
          f"final outlet c={tracer_fem['history']['cout'][-1]:.4f}, t90={tracer.crossing_time(tracer_fem['history']['time'], tracer_fem['history']['cout'], 0.90):.3f}")

    rows: list[dict[str, object]] = []
    histories: dict[str, dict[str, np.ndarray]] = {}
    final_vertex_fields: dict[str, np.ndarray] = {}
    final_dof_fields: dict[str, np.ndarray] = {}
    vertex_velocities: dict[str, np.ndarray] = {}
    mass_matrix = tracer_fem["mass_matrix"]

    fem_row = {
        "method": "FEM",
        "stokes_solve_time_seconds": float(fem_seconds),
        "global_unknowns": int(reference.ndofs),
        "modes_per_interface": "monolithic",
        "velocity_relative_l2_error_vs_fem": 0.0,
        "velocity_relative_broken_h1_vs_fem": 0.0,
        "pressure_raw_relative_l2_error_vs_fem": 0.0,
        "pressure_mean_aligned_relative_l2_error_vs_fem": 0.0,
        "outlet_flux_relative_error_vs_fem": 0.0,
        "tracer_time_seconds": timings["FEM"]["tracer_seconds"],
    }
    rows.append(fem_row)
    histories["FEM"] = tracer_fem["history"]
    final_vertex_fields["FEM"] = tracer_fem["final_concentration_vertices"]
    final_dof_fields["FEM"] = tracer_fem["final_concentration"]
    vertex_velocities["FEM"] = np.asarray(u_fem, dtype=float)

    basis_factories = {
        "Classic-DDPNM-1": lambda p: CompatibleClassicP0Basis(),
        "NormalLinear-DDPNM-3": lambda p: NormalLinearFaceBasis(p),
        "Affine-DDPNM-9": lambda p: AffineFaceBasis(p),
    }
    for step, method in enumerate(("Classic-DDPNM-1", "NormalLinear-DDPNM-3", "Affine-DDPNM-9"), start=4):
        print(f"[{step}/7] {method} ...")
        t0 = time.perf_counter()
        basis = basis_factories[method](partition)
        library = build_response_library(
            partition, basis,
            viscosity=args.viscosity,
            inlet_pressure=args.inlet_pressure,
            outlet_pressure=args.outlet_pressure,
            pressure_stabilization=args.pressure_stabilization,
        )
        offline = time.perf_counter() - t0
        levels = np.full(n_interfaces, 2, dtype=np.int8) if method == "Affine-DDPNM-9" else np.zeros(n_interfaces, dtype=np.int8)
        t0 = time.perf_counter()
        system = InterfaceAssembler(library).assemble(levels)
        solution = reduced_solution(partition, library, system)
        online = time.perf_counter() - t0
        u_ddpnm, p_ddpnm, _vertex_counts = reconstruct_parent_vertices(partition, solution)
        t0 = time.perf_counter()
        history = tracer.solve_tracer(
            partition.mesh, u_ddpnm,
            diffusivity=args.diffusivity, porosity=args.porosity,
            dt=args.dt, t_final=args.t_final, supg=args.supg, supg_factor=args.supg_factor,
        )
        tracer_time = time.perf_counter() - t0

        row = {
            "method": method,
            "stokes_solve_time_seconds": float(offline + online),
            "offline_seconds": float(offline),
            "online_seconds": float(online),
            "global_unknowns": int(len(system.global_keys)),
            "modes_per_interface": 1 if method == "Classic-DDPNM-1" else (3 if method == "NormalLinear-DDPNM-3" else 9),
            "schur_symmetry_error": float(system.symmetry_error),
            "min_schur_eigenvalue": float(solution.min_schur_eigenvalue),
            "max_mass_residual": float(solution.max_mass_residual),
            "tracer_time_seconds": float(tracer_time),
            **ddpnm_stokes_metrics(partition, solution, reference, volumes),
        }
        rows.append(row)
        histories[method] = history["history"]
        final_vertex_fields[method] = history["final_concentration_vertices"]
        final_dof_fields[method] = history["final_concentration"]
        vertex_velocities[method] = np.asarray(u_ddpnm, dtype=float)
        print(
            f"      dofs={len(system.global_keys)}, offline={offline:.3f} s, online={online:.3f} s, "
            f"L2(u)={row['velocity_relative_l2_error_vs_fem']:.3%}, tracer={tracer_time:.3f} s"
        )
        del library, system, solution
        gc.collect()

    print("[7/7] Metrics and outputs ...")
    attach_tracer_metrics(rows, histories, final_dof_fields, mass_matrix, reference="FEM")
    _write_outputs(
        args, partition, points, tetrahedra, mesh_seconds, timings, rows,
        histories, final_vertex_fields, vertex_velocities,
    )


# ---------------------------------------------------------------------------
# Tracer metrics
# ---------------------------------------------------------------------------


def attach_tracer_metrics(
    rows: list[dict[str, object]],
    histories: dict[str, dict[str, np.ndarray]],
    final_dof_fields: dict[str, np.ndarray],
    mass_matrix,
    reference: str = "FEM",
) -> None:
    for row in rows:
        method = str(row["method"])
        history = histories[method]
        row["tracer_time_seconds"] = float(row.get("tracer_time_seconds", 0.0))
        row["mean_outlet_concentration"] = float(np.mean(history["cout"]))
        row["final_outlet_concentration"] = float(history["cout"][-1])
        row["final_concentration_min"] = float(history["min_c"][-1])
        row["final_concentration_max"] = float(history["max_c"][-1])
        row["raw_final_concentration_min_before_limiter"] = float(history["raw_min_c_before_limiter"][-1])
        row["raw_final_concentration_max_before_limiter"] = float(history["raw_max_c_before_limiter"][-1])
        row["raw_final_concentration_below_zero_before_limiter"] = int(history["raw_below_zero_before_limiter"][-1])
        row["raw_final_concentration_above_one_before_limiter"] = int(history["raw_above_one_before_limiter"][-1])
        row["limiter_final_mass_residual"] = float(history["limiter_mass_residual"][-1])
        row["max_mass_balance_residual_rate"] = float(np.max(np.abs(history["mass_balance_residual_rate"])))
        row["max_mass_balance_relative_residual"] = float(np.max(np.abs(history["mass_balance_relative_residual"])))
        row["max_limiter_mass_residual"] = float(np.max(np.abs(history["limiter_mass_residual"])))
        row["final_tracer_mass"] = float(history["mass"][-1])
        row["min_tracer_mass"] = float(np.min(history["mass"]))
        row["max_tracer_mass"] = float(np.max(history["mass"]))
        row["t10"] = tracer.crossing_time(history["time"], history["cout"], 0.10)
        row["t50"] = tracer.crossing_time(history["time"], history["cout"], 0.50)
        row["t90"] = tracer.crossing_time(history["time"], history["cout"], 0.90)
    tracer.add_reference_errors(rows, histories, final_dof_fields, mass_matrix, reference)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _write_outputs(
    args, partition, points, tetrahedra, mesh_seconds, timings, rows,
    histories, final_vertex_fields, vertex_velocities,
) -> None:
    out_dir = args.out_dir
    tracer.write_csv(out_dir / "tracer_metrics.csv", rows)
    tracer.write_history_csv(out_dir / "mass_balance_history.csv", histories)
    tracer.plot_breakthrough(out_dir / "breakthrough_curves.png", histories)
    tracer.plot_mass_balance(out_dir / "mass_balance_validation.png", histories)
    tracer.plot_error_summary(out_dir / "tracer_error_summary.png", rows)
    tracer.plot_final_concentration(
        out_dir / "final_concentration_and_error.png",
        points, tetrahedra, final_vertex_fields,
        [str(row["method"]) for row in rows], reference="FEM",
    )
    for row in rows:
        method = str(row["method"])
        save_method_vtu(
            out_dir / f"{method.lower().replace('-', '_')}_tracer_final.vtu",
            points, tetrahedra,
            vertex_velocities[method], final_vertex_fields[method],
        )

    np.savez_compressed(
        out_dir / "tracer_velocity_fields.npz",
        points=points,
        tetrahedra=tetrahedra,
        cell_labels=partition.cell_labels,
        interface_pairs=np.asarray(partition.interface_pairs, dtype=np.int32),
        interface_centers=partition.interface_centers,
        interface_normals=partition.interface_normals,
        interface_areas=partition.interface_areas,
        sphere_centers=partition.pore_seeds[:, :3],
        sphere_radii=partition.pore_seeds[:, 3],
        **{f"u_{method.replace('-', '_')}": vertex_velocities[method] for method in vertex_velocities},
        **{f"c_{method.replace('-', '_')}": final_vertex_fields[method] for method in final_vertex_fields},
    )

    summary = {
        "description": (
            "Single-phase tracer validation driven by DDPNM Stokes velocity fields: "
            "Classic-DDPNM-1 / NormalLinear-DDPNM-3 (W1n) / Affine-DDPNM-9 on the "
            "random-27 sphere medium, all compared against the monolithic FEM-driven tracer."
        ),
        "geometry": (
            "unit cube minus a frozen seed-20260804 packing of 27 random spheres "
            "(18 clipped wall spheres + 9 interior spheres); Voronoi throat faces"
        ),
        "stokes_model": "FEniCSx Taylor-Hood P2-P1 Stokes, pressure-driven (inlet=1, outlet=0), no-slip walls.",
        "ddpnm_model": (
            "per-interface modal traction spaces: {P0 n} (Classic, 1/face), "
            "{P0,P1_s,P1_t} x {n} (W1n, 3/face), {P0,P1_s,P1_t} x {n,t1,t2} (Affine, 9/face)"
        ),
        "tracer_model": (
            "transient P1 scalar advection-diffusion, inlet step c=1 at x=0, "
            "natural outlet/wall flux, implicit Euler + SUPG, conservative bounded limiter"
        ),
        "velocity_field_for_tracer": (
            "P1 vertex values; DDPNM fields use the two-sided interface average "
            "of the per-pore local solutions (reconstruct_parent_vertices)"
        ),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "counts": {
            "solid_spheres": 27,
            "pore_subdomains": int(len(np.unique(partition.cell_labels))),
            "interfaces": int(len(partition.interface_pairs)),
            "global_vertices": int(len(points)),
            "global_tetrahedra": int(len(tetrahedra)),
        },
        "common_mesh_seconds": float(mesh_seconds),
        "timings": timings,
        "stokes_cases": rows,
        "tracer_metrics": rows,
        "outputs": {
            "breakthrough_curves": str(out_dir / "breakthrough_curves.png"),
            "mass_balance_validation": str(out_dir / "mass_balance_validation.png"),
            "tracer_error_summary": str(out_dir / "tracer_error_summary.png"),
            "final_concentration_and_error": str(out_dir / "final_concentration_and_error.png"),
            "mass_balance_history_csv": str(out_dir / "mass_balance_history.csv"),
            "tracer_metrics_csv": str(out_dir / "tracer_metrics.csv"),
            "tracer_velocity_fields_npz": str(out_dir / "tracer_velocity_fields.npz"),
        },
    }
    (out_dir / "affine_ddpnm_tracer_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(out_dir / "TRACER_VALIDATION_REPORT.md", summary)
    print(f"Done: {out_dir.resolve()}")


def save_method_vtu(out: Path, points, tetrahedra, velocity: np.ndarray, concentration: np.ndarray) -> None:
    cells = np.hstack([np.full((len(tetrahedra), 1), 4, dtype=np.int64), np.asarray(tetrahedra, dtype=np.int64)]).ravel()
    celltypes = np.full(len(tetrahedra), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, np.asarray(points, dtype=float))
    grid.point_data["velocity"] = np.asarray(velocity, dtype=float)
    grid.point_data["concentration"] = np.asarray(concentration, dtype=float)
    grid.save(out)


def write_report(path: Path, summary: dict[str, object]) -> None:
    rows = summary["tracer_metrics"]
    lines = [
        "# Affine-DDPNM Driven Tracer Validation (random-27 medium)",
        "",
        "The same random-27 partition mesh is solved by the monolithic Taylor-Hood FEM and by "
        "three DDPNM interface-traction spaces (Classic-1, W1n-3, Affine-9).  Each Stokes "
        "velocity field then drives the identical transient tracer advection-diffusion model, "
        "and every tracer metric is reported against the FEM-driven reference.",
        "",
        "## Stokes Compression and Velocity Error",
        "",
        "| method | Stokes time (s) | global unknowns | modes/face | velocity rel L2 | velocity broken H1 | pressure aligned rel L2 | outlet flux rel err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {float(row['stokes_solve_time_seconds']):.3f} | "
            f"{row['global_unknowns']} | {row['modes_per_interface']} | "
            f"{float(row['velocity_relative_l2_error_vs_fem']):.3%} | "
            f"{float(row['velocity_relative_broken_h1_vs_fem']):.3%} | "
            f"{float(row['pressure_mean_aligned_relative_l2_error_vs_fem']):.3%} | "
            f"{float(row['outlet_flux_relative_error_vs_fem']):.3e} |"
        )
    lines.extend(
        [
            "",
            "## Tracer Metrics vs FEM-Driven Reference",
            "",
            "| method | tracer time (s) | breakthrough rel L2 | concentration rel L2 | final mass rel err | max balance residual | final range | raw limiter hits | t90 |",
            "|---|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        raw_hits = int(row["raw_final_concentration_below_zero_before_limiter"]) + int(row["raw_final_concentration_above_one_before_limiter"])
        lines.append(
            f"| {row['method']} | {row['tracer_time_seconds']:.3f} | "
            f"{row['breakthrough_rel_l2_error']:.3e} | "
            f"{row['concentration_rel_l2_fem_integral_vs_fem']:.3e} | "
            f"{row['final_mass_rel_error_vs_fem']:.3e} | "
            f"{row['max_mass_balance_relative_residual']:.3e} | "
            f"[{row['final_concentration_min']:.3f}, {row['final_concentration_max']:.3f}] | "
            f"{raw_hits} | "
            f"{row['t90']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Breakthrough curves: `{summary['outputs']['breakthrough_curves']}`",
            f"- Mass balance validation: `{summary['outputs']['mass_balance_validation']}`",
            f"- Error summary: `{summary['outputs']['tracer_error_summary']}`",
            f"- Final concentration and error: `{summary['outputs']['final_concentration_and_error']}`",
            f"- CSV metrics: `{summary['outputs']['tracer_metrics_csv']}`",
            f"- Mass history CSV: `{summary['outputs']['mass_balance_history_csv']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

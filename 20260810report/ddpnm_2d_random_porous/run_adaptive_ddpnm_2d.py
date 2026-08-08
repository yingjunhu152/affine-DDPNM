from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpnm2d.exact_schur import solve_exact_fe_schur
from ddpnm2d.geometry import PARTICLES, build_partition
from ddpnm_core.algebra import hierarchy_error
from ddpnm_core.io import topology_arrays
from ddpnm_core.reconstruction import mixed_solution_to_p1

from ddpnm2d.hierarchy import (
    reconstruct_hierarchy_vertices,
    run_adaptive_hierarchy,
    build_hierarchy_library,
    write_adaptive_report,
)


def plot_method_validation(partition, adaptive, reference_fields, method_errors, out_dir: Path) -> None:
    names = ["DDPNM", "DDPNMT", "HODDPNM", "adaptive_final"]
    labels = ["DDPNM", "DDPNMT", "HODDPNM", "Adaptive"]
    velocity = [method_errors[name]["velocity"] for name in names]
    pressure = [method_errors[name]["pressure"] for name in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    width = 0.36
    ax.bar(x - width / 2, velocity, width, label="velocity", color="#2878b5")
    ax.bar(x + width / 2, pressure, width, label="pressure", color="#f2a541")
    ax.axhline(adaptive.tolerance, color="#c62828", ls="--", label="hierarchical TOL")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("relative error to exact FE Schur")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "method_errors_to_exact_schur.png", dpi=220)
    plt.close(fig)

    points, cells = topology_arrays(partition.mesh)
    u_ref, _ = reference_fields
    solutions = [
        adaptive.initial_ddpnm,
        adaptive.full_ddpnmt,
        adaptive.full_hoddpnm,
        adaptive.final_solution,
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.2), constrained_layout=True)
    for ax, label, solution in zip(axes.ravel(), labels, solutions, strict=True):
        u, _ = reconstruct_hierarchy_vertices(partition, solution)
        point_error = np.linalg.norm(u - u_ref, axis=1)
        field = np.log10(np.maximum(point_error, 1.0e-12))
        artist = ax.tripcolor(points[:, 0], points[:, 1], cells, field, shading="gouraud", cmap="turbo")
        for px, py, radius in PARTICLES:
            ax.add_patch(plt.Circle((px, py), radius, facecolor="white", edgecolor="#777777", lw=0.5))
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[], title=f"{label}: log10 |u-u_FE|")
        fig.colorbar(artist, ax=ax, shrink=0.82)
    fig.savefig(out_dir / "method_velocity_error_fields.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive DDPNM -> DDPNMT -> HODDPNM on the fixed 2-D porous medium."
    )
    parser.add_argument("--mesh-size", type=float, default=0.04)
    parser.add_argument("--wall-size", type=float, default=0.018)
    parser.add_argument("--throat-size", type=float, default=0.010)
    parser.add_argument("--wall-band", type=float, default=0.065)
    parser.add_argument("--throat-band", type=float, default=0.045)
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--inlet-pressure", type=float, default=1.0)
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--pressure-stabilization", type=float, default=1.0e-10)
    parser.add_argument("--target-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--marking-theta", type=float, default=0.65)
    parser.add_argument("--max-marked-per-iteration", type=int, default=3)
    parser.add_argument("--max-iterations-per-phase", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/adaptive_hierarchy"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    print("[1/5] Building analytic saddle-cut, wall/throat-refined mesh...", flush=True)
    partition = build_partition(
        mesh_size=args.mesh_size,
        wall_size=args.wall_size,
        throat_size=args.throat_size,
        wall_band=args.wall_band,
        throat_band=args.throat_band,
    )
    print(
        f"      {len(set(partition.cell_labels.tolist()))} pores, "
        f"{len(partition.interface_pairs)} interfaces",
        flush=True,
    )
    print("[2/5] Factoring local Stokes systems and building P1 vector-traction library...", flush=True)
    library = build_hierarchy_library(
        partition,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
    )
    print(
        f"      maximum local response symmetry error "
        f"{max(r.symmetry_error for r in library.local_responses):.3e}",
        flush=True,
    )
    print("[3/5] Forming exact full Taylor-Hood FE-trace Schur complement...", flush=True)
    exact = solve_exact_fe_schur(
        partition,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
    )
    reference_fields = mixed_solution_to_p1(exact.W, exact.solution)
    print(
        f"      Schur size {len(exact.interface_dofs)}, "
        f"Schur-vs-monolithic difference {exact.monolithic_relative_difference:.3e}",
        flush=True,
    )
    print("[4/5] Running two-stage hierarchical adaptive promotion...", flush=True)
    adaptive = run_adaptive_hierarchy(
        library,
        tolerance=args.target_tolerance,
        marking_theta=args.marking_theta,
        max_marked_per_iteration=args.max_marked_per_iteration,
        max_iterations_per_phase=args.max_iterations_per_phase,
        reference_fields=reference_fields,
    )
    print(
        "      final interface levels DDPNM/DDPNMT/HODDPNM = "
        f"{[int((adaptive.final_solution.levels == i).sum()) for i in range(3)]}",
        flush=True,
    )
    print("[5/5] Writing algorithm box, convergence figures and validation report...", flush=True)
    parameters = {
        "mesh_size": args.mesh_size,
        "wall_size": args.wall_size,
        "throat_size": args.throat_size,
        "wall_band": args.wall_band,
        "throat_band": args.throat_band,
        "viscosity": args.viscosity,
        "inlet_pressure": args.inlet_pressure,
        "outlet_pressure": args.outlet_pressure,
        "pressure_stabilization": args.pressure_stabilization,
        "target_tolerance": args.target_tolerance,
        "marking_theta": args.marking_theta,
        "max_marked_per_iteration": args.max_marked_per_iteration,
        "max_iterations_per_phase": args.max_iterations_per_phase,
    }
    report = write_adaptive_report(library, adaptive, args.out_dir, parameters)
    method_errors = {}
    for name, solution in [
        ("DDPNM", adaptive.initial_ddpnm),
        ("DDPNMT", adaptive.full_ddpnmt),
        ("HODDPNM", adaptive.full_hoddpnm),
        ("adaptive_final", adaptive.final_solution),
    ]:
        fields = reconstruct_hierarchy_vertices(partition, solution)
        method_errors[name] = hierarchy_error(fields, reference_fields).__dict__
    plot_method_validation(partition, adaptive, reference_fields, method_errors, args.out_dir)
    report["errors_to_exact_FE_schur"] = method_errors
    report["exact_FE_trace_schur"] = {
        "total_mixed_dofs": int(len(exact.solution)),
        "fixed_velocity_dofs": int(len(exact.fixed_dofs)),
        "interface_trace_dofs": int(len(exact.interface_dofs)),
        "interior_dofs": int(len(exact.interior_dofs)),
        "schur_symmetry_error": exact.schur_symmetry_error,
        "schur_relative_residual": exact.schur_relative_residual,
        "interior_relative_residual": exact.interior_relative_residual,
        "global_relative_residual": exact.global_relative_residual,
        "relative_difference_from_monolithic": exact.monolithic_relative_difference,
        "note": "Exact correctness Schur; the mixed P2-P1 Schur matrix is symmetric but not required to be positive definite.",
    }
    elapsed = time.perf_counter() - started
    report["wall_time_seconds"] = elapsed
    (args.out_dir / "adaptive_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Done in {elapsed:.2f} s; final velocity/pressure errors to exact Schur = "
        f"{method_errors['adaptive_final']['velocity']:.3e} / "
        f"{method_errors['adaptive_final']['pressure']:.3e}",
        flush=True,
    )
    print(f"Outputs: {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare DDPNM/DDPNMT and cardinal cross-normal/cross-vector DDPNM."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from ddpnm_core.fem_utils import solve_reference
from ddpnm_core.io import topology_arrays
from ddpnm_core.validation import finite_element_error_analysis
from ddpnm3d.geometry import build_partition
from ddpnm3d.skeleton_stokes import solve_cross_tangential_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-size", type=float, default=0.20)
    parser.add_argument("--sphere-size", type=float, default=0.040)
    parser.add_argument("--boundary-size", type=float, default=0.070)
    parser.add_argument("--interface-size", type=float, default=0.055)
    parser.add_argument("--sphere-band", type=float, default=0.10)
    parser.add_argument("--boundary-band", type=float, default=0.08)
    parser.add_argument("--interface-band", type=float, default=0.08)
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--inlet-pressure", type=float, default=1.0)
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--pressure-stabilization", type=float, default=0.0)
    parser.add_argument("--alignment-penalty", type=float, default=3.0)
    parser.add_argument("--compliance-floor", type=float, default=1.0e-10)
    parser.add_argument("--energy-ridge", type=float, default=1.0e-10)
    parser.add_argument("--reference-iterative-threshold", type=int, default=100_000)
    parser.add_argument("--reference-rtol", type=float, default=1.0e-9)
    parser.add_argument("--reference-restart", type=int, default=60)
    parser.add_argument("--reference-maxiter", type=int, default=120)
    parser.add_argument("--reference-ilu-drop-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--reference-ilu-fill-factor", type=float, default=6.0)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/cross_ddpnmt")
    )
    return parser.parse_args()


def tetrahedron_volumes(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    xyz = points[tetrahedra]
    matrices = np.stack(
        (xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0]),
        axis=1,
    )
    return np.abs(np.linalg.det(matrices)) / 6.0


def system_diagnostics(solution) -> dict:
    inlet = float(solution.boundary_fluxes["inlet"])
    outlet = float(solution.boundary_fluxes["outlet"])
    return {
        "interface_unknowns": len(solution.global_keys),
        "minimum_schur_eigenvalue": solution.min_schur_eigenvalue,
        "relative_linear_residual": solution.relative_linear_residual,
        "maximum_interface_moment_residual": solution.max_moment_residual,
        "maximum_local_mass_residual": solution.max_local_mass_residual,
        "maximum_local_linear_residual": solution.max_local_linear_residual,
        "maximum_flux_divergence_discrepancy": (
            solution.max_flux_divergence_discrepancy
        ),
        "inlet_outward_flux": inlet,
        "outlet_outward_flux": outlet,
        "relative_mass_imbalance": abs(inlet + outlet)
        / max(abs(inlet), abs(outlet), 1.0e-30),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    print("[1/4] Building the common saddle partition and tetrahedral mesh ...")
    started = time.perf_counter()
    partition = build_partition(
        mesh_size=args.mesh_size,
        sphere_size=args.sphere_size,
        boundary_size=args.boundary_size,
        interface_size=args.interface_size,
        sphere_band=args.sphere_band,
        boundary_band=args.boundary_band,
        interface_band=args.interface_band,
        mesh_file=args.out_dir / "cross_ddpnmt_partition.msh",
    )
    timings["mesh"] = time.perf_counter() - started
    points, tetrahedra = topology_arrays(partition.mesh)
    volumes = tetrahedron_volumes(points, tetrahedra)
    print(f"      tetrahedra={len(tetrahedra)}, interfaces={len(partition.interface_pairs)}")

    print("[2/4] Building one vector nodal response library and four Schur systems ...")
    started = time.perf_counter()
    comparison = solve_cross_tangential_comparison(
        partition,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
        alignment_penalty=args.alignment_penalty,
        compliance_floor=args.compliance_floor,
        energy_ridge=args.energy_ridge,
    )
    timings["response_and_reduced_solves"] = time.perf_counter() - started
    solutions = {
        "DDPNM": comparison.ddpnm,
        "DDPNMT": comparison.ddpnmt,
        "Cross-DDPNM": comparison.cross_normal,
        "Cross-DDPNMT": comparison.cross_ddpnmt,
    }
    for name, solution in solutions.items():
        print(
            f"      {name}: dofs={len(solution.global_keys)}, "
            f"mass={solution.max_local_mass_residual:.2e}, "
            f"Schur residual={solution.relative_linear_residual:.2e}"
        )

    print("[3/4] Solving monolithic Taylor--Hood FEM on the identical mesh ...")
    started = time.perf_counter()
    reference = solve_reference(
        partition.mesh,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
        iterative_threshold=args.reference_iterative_threshold,
        iterative_rtol=args.reference_rtol,
        iterative_restart=args.reference_restart,
        iterative_maxiter=args.reference_maxiter,
        ilu_drop_tolerance=args.reference_ilu_drop_tolerance,
        ilu_fill_factor=args.reference_ilu_fill_factor,
    )
    timings["reference"] = time.perf_counter() - started

    print("[4/4] Computing strict same-mesh errors and writing audit data ...")
    started = time.perf_counter()
    metrics: dict[str, dict] = {}
    cell_arrays: dict[str, np.ndarray] = {}
    for name, solution in solutions.items():
        metric, velocity_cell, pressure_cell = finite_element_error_analysis(
            partition, solution, reference, volumes
        )
        metrics[name] = metric
        safe = name.lower().replace("-", "_")
        cell_arrays[f"velocity_error_cell_rms_{safe}"] = velocity_cell
        cell_arrays[f"pressure_error_cell_rms_{safe}"] = pressure_cell
        print(
            f"      {name}: L2(u)={metric['velocity_relative_l2']:.3%}, "
            f"H1b(u)={metric['velocity_relative_broken_h1_seminorm']:.3%}, "
            f"L2(p)={metric['pressure_raw_relative_l2']:.3%}, "
            f"flux={metric['outlet_flux_relative_error']:.3%}"
        )
    timings["validation"] = time.perf_counter() - started
    timings["total"] = time.perf_counter() - total_started

    with (args.out_dir / "method_error_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "interface_unknowns",
                "velocity_relative_L2",
                "velocity_relative_broken_H1",
                "pressure_raw_relative_L2",
                "outlet_flux_relative_error",
                "relative_mass_imbalance",
            ]
        )
        for name, solution in solutions.items():
            metric = metrics[name]
            diagnostics = system_diagnostics(solution)
            writer.writerow(
                [
                    name,
                    len(solution.global_keys),
                    metric["velocity_relative_l2"],
                    metric["velocity_relative_broken_h1_seminorm"],
                    metric["pressure_raw_relative_l2"],
                    metric["outlet_flux_relative_error"],
                    diagnostics["relative_mass_imbalance"],
                ]
            )

    np.savez_compressed(
        args.out_dir / "cross_ddpnmt_results.npz",
        points=points,
        tetrahedra=tetrahedra,
        cell_volumes=volumes,
        full_scalar_nodes_per_interface=np.asarray(
            [item.full_scalar_dofs for item in comparison.normal_diagnostics]
        ),
        cross_nodes_per_interface=np.asarray(
            [item.skeleton_scalar_dofs for item in comparison.normal_diagnostics]
        ),
        **cell_arrays,
    )
    np.savez_compressed(
        args.out_dir / "cross_vector_extension_matrices.npz",
        **{
            f"interface_{index:03d}": transform
            for index, transform in enumerate(comparison.vector_transforms)
        },
    )
    report = {
        "method": "DDPNM/DDPNMT versus cardinal Cross-DDPNM/Cross-DDPNMT",
        "spaces": {
            "DDPNM": "one constant normal traction per interface",
            "DDPNMT": "constant normal plus two constant tangential tractions",
            "Cross-DDPNM": "one cardinal normal traction per cross node",
            "Cross-DDPNMT": (
                "cardinal normal plus two cardinal tangential tractions per cross node"
            ),
        },
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "counts": {
            "tetrahedra": len(tetrahedra),
            "subdomains": 64,
            "interfaces": len(partition.interface_pairs),
            "full_face_vector_dofs_sum": int(
                sum(item.full_vector_dofs for item in comparison.vector_diagnostics)
            ),
            "cross_vector_dofs_sum": int(
                sum(item.active_vector_dofs for item in comparison.vector_diagnostics)
            ),
        },
        "extension": {
            "maximum_normal_cardinal_residual": float(
                max(item.constraint_residual for item in comparison.normal_diagnostics)
            ),
            "maximum_normal_constant_residual": float(
                max(
                    item.constant_reproduction_residual
                    for item in comparison.normal_diagnostics
                )
            ),
            "maximum_vector_cardinal_residual": float(
                max(item.cardinal_residual for item in comparison.vector_diagnostics)
            ),
            "maximum_vector_constant_residual": float(
                max(
                    item.constant_vector_reproduction_residual
                    for item in comparison.vector_diagnostics
                )
            ),
        },
        "systems": {
            name: system_diagnostics(solution)
            for name, solution in solutions.items()
        },
        "strict_errors_to_identical_mesh_FEM": metrics,
        "reference": {
            "mixed_dofs": reference.ndofs,
            "relative_linear_residual": reference.relative_linear_residual,
            "relative_mass_imbalance": reference.relative_mass_imbalance,
        },
        "timings_seconds": timings,
        "vector_interfaces": [
            asdict(item) for item in comparison.vector_diagnostics
        ],
    }
    (args.out_dir / "cross_ddpnmt_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done in {timings['total']:.1f} s: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

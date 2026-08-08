#!/usr/bin/env python3
"""Six-way same-mesh comparison including quasi-uniform interface points."""

from __future__ import annotations

import argparse
import csv
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
from ddpnm3d.geometry import SPHERE_CENTERS, SPHERE_RADIUS, build_partition
from ddpnm3d.skeleton_stokes import (
    _interface_triangles,
    solve_cross_uniform_tangential_comparison,
)
from ddpnm3d.visualization import evaluate_fem_ddpnm_slice


METHOD_NAMES = (
    "DDPNM",
    "DDPNMT",
    "Cross-DDPNM",
    "Cross-DDPNMT",
    "Uniform-DDPNM",
    "Uniform-DDPNMT",
)
METRIC_KEYS = (
    "velocity_relative_l2",
    "velocity_relative_broken_h1_seminorm",
    "pressure_raw_relative_l2",
    "outlet_flux_relative_error",
)


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
    parser.add_argument(
        "--sampling-factor",
        type=float,
        default=1.0,
        help="Ns=ceil(sampling_factor*sqrt(Nf)) per interface",
    )
    parser.add_argument(
        "--uniform-affine-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exactly reproduce {1,s,t} for every active traction component",
    )
    parser.add_argument("--compliance-floor", type=float, default=1.0e-10)
    parser.add_argument("--energy-ridge", type=float, default=1.0e-10)
    parser.add_argument("--slice-z", type=float, default=0.48)
    parser.add_argument("--reference-iterative-threshold", type=int, default=100_000)
    parser.add_argument("--reference-rtol", type=float, default=1.0e-9)
    parser.add_argument("--reference-restart", type=int, default=60)
    parser.add_argument("--reference-maxiter", type=int, default=120)
    parser.add_argument("--reference-ilu-drop-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--reference-ilu-fill-factor", type=float, default=6.0)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/uniform_point_ddpnmt")
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


def representative_face_arrays(partition, comparison) -> dict[str, np.ndarray]:
    """Return a median-size interface projected onto its own tangent plane."""
    sizes = np.asarray([len(item.full_nodes) for item in comparison.uniform_sets])
    order = np.argsort(sizes)
    interface_id = int(order[len(order) // 2])
    control = comparison.uniform_sets[interface_id]
    cross = comparison.skeletons[interface_id]
    nodes = np.asarray(control.full_nodes, dtype=np.int32)
    node_to_local = {int(node): index for index, node in enumerate(nodes)}
    relative = partition.mesh.geometry.x[nodes, :3] - control.saddle
    xy = np.column_stack((relative @ control.tangent_1, relative @ control.tangent_2))
    triangles = _interface_triangles(partition, interface_id)
    triangles_local = np.asarray(
        [[node_to_local[int(node)] for node in triangle] for triangle in triangles],
        dtype=np.int32,
    )
    cross_edges = []
    for arm in cross.arms:
        cross_edges.extend(
            (node_to_local[int(first)], node_to_local[int(second)])
            for first, second in zip(arm[:-1], arm[1:], strict=True)
        )
    return {
        "representative_interface_id": np.asarray(interface_id),
        "representative_face_xy": xy,
        "representative_triangles": triangles_local,
        "representative_cross_nodes": np.asarray(
            [node_to_local[int(node)] for node in cross.skeleton_nodes],
            dtype=np.int32,
        ),
        "representative_cross_edges": np.asarray(cross_edges, dtype=np.int32),
        "representative_uniform_nodes": np.asarray(
            [node_to_local[int(node)] for node in control.skeleton_nodes],
            dtype=np.int32,
        ),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    print("[1/4] Building the common 27-sphere mesh ...")
    started = time.perf_counter()
    partition = build_partition(
        mesh_size=args.mesh_size,
        sphere_size=args.sphere_size,
        boundary_size=args.boundary_size,
        interface_size=args.interface_size,
        sphere_band=args.sphere_band,
        boundary_band=args.boundary_band,
        interface_band=args.interface_band,
        mesh_file=args.out_dir / "uniform_point_partition.msh",
    )
    timings["mesh"] = time.perf_counter() - started
    points, tetrahedra = topology_arrays(partition.mesh)
    volumes = tetrahedron_volumes(points, tetrahedra)
    print(f"      tetrahedra={len(tetrahedra)}, interfaces={len(partition.interface_pairs)}")

    print("[2/4] Building one nodal library and six reduced Schur systems ...")
    started = time.perf_counter()
    comparison = solve_cross_uniform_tangential_comparison(
        partition,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
        alignment_penalty=args.alignment_penalty,
        sampling_factor=args.sampling_factor,
        uniform_affine_complete=args.uniform_affine_complete,
        compliance_floor=args.compliance_floor,
        energy_ridge=args.energy_ridge,
    )
    timings["response_and_reduced_solves"] = time.perf_counter() - started
    solutions = {
        "DDPNM": comparison.ddpnm,
        "DDPNMT": comparison.ddpnmt,
        "Cross-DDPNM": comparison.cross_normal,
        "Cross-DDPNMT": comparison.cross_ddpnmt,
        "Uniform-DDPNM": comparison.uniform_normal,
        "Uniform-DDPNMT": comparison.uniform_ddpnmt,
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
        metric["maximum_cell_velocity_rms_error"] = float(np.max(velocity_cell))
        metric["maximum_cell_pressure_rms_error"] = float(np.max(pressure_cell))
        safe = name.lower().replace("-", "_")
        cell_arrays[f"velocity_error_cell_rms_{safe}"] = velocity_cell
        cell_arrays[f"pressure_error_cell_rms_{safe}"] = pressure_cell
        print(
            f"      {name}: L2(u)={metric['velocity_relative_l2']:.3%}, "
            f"H1b(u)={metric['velocity_relative_broken_h1_seminorm']:.3%}, "
            f"L2(p)={metric['pressure_raw_relative_l2']:.3%}, "
            f"flux={metric['outlet_flux_relative_error']:.3%}"
        )

    print(f"      evaluating broken local fields on z={args.slice_z:.3f} ...")
    slice_arrays: dict[str, np.ndarray] = {}
    for method_index, (name, solution) in enumerate(solutions.items()):
        local_slice = evaluate_fem_ddpnm_slice(
            partition,
            solution,
            reference,
            points,
            tetrahedra,
            z_value=args.slice_z,
        )
        safe = name.lower().replace("-", "_")
        if method_index == 0:
            slice_arrays.update(
                {
                    "error_slice_z": local_slice["error_slice_z"],
                    "error_slice_points": local_slice["error_slice_points"],
                    "error_slice_triangles": local_slice["error_slice_triangles"],
                    "error_slice_parent_cells": local_slice[
                        "error_slice_parent_cells"
                    ],
                    "error_slice_u_fem": local_slice["error_slice_u_fem"],
                    "error_slice_p_fem": local_slice["error_slice_p_fem"],
                }
            )
        slice_arrays[f"error_slice_u_{safe}"] = local_slice[
            "error_slice_u_ddpnm"
        ]
        slice_arrays[f"error_slice_p_{safe}"] = local_slice[
            "error_slice_p_ddpnm"
        ]
        pointwise_velocity_error = np.linalg.norm(
            local_slice["error_slice_u_ddpnm"]
            - local_slice["error_slice_u_fem"],
            axis=1,
        )
        pointwise_pressure_error = np.abs(
            local_slice["error_slice_p_ddpnm"]
            - local_slice["error_slice_p_fem"]
        )
        metrics[name]["maximum_slice_pointwise_velocity_error"] = float(
            np.max(pointwise_velocity_error)
        )
        metrics[name]["maximum_slice_pointwise_pressure_error"] = float(
            np.max(pointwise_pressure_error)
        )
        print(
            f"      {name}: max cell RMS |du|="
            f"{metrics[name]['maximum_cell_velocity_rms_error']:.3e}, "
            f"max slice |du|="
            f"{metrics[name]['maximum_slice_pointwise_velocity_error']:.3e}, "
            f"max local mass={solution.max_local_mass_residual:.3e}"
        )
    timings["validation"] = time.perf_counter() - started
    timings["total"] = time.perf_counter() - total_started

    diagnostics = {
        name: system_diagnostics(solution) for name, solution in solutions.items()
    }
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
                "maximum_cell_velocity_rms_error",
                "maximum_slice_pointwise_velocity_error",
                "maximum_local_mass_residual",
                "relative_mass_imbalance",
            ]
        )
        for name in METHOD_NAMES:
            metric = metrics[name]
            writer.writerow(
                [
                    name,
                    len(solutions[name].global_keys),
                    *(metric[key] for key in METRIC_KEYS),
                    metric["maximum_cell_velocity_rms_error"],
                    metric["maximum_slice_pointwise_velocity_error"],
                    solutions[name].max_local_mass_residual,
                    diagnostics[name]["relative_mass_imbalance"],
                ]
            )

    uniform_counts = np.asarray(
        [item.target_count for item in comparison.uniform_sets], dtype=np.int32
    )
    full_counts = np.asarray(
        [len(item.full_nodes) for item in comparison.uniform_sets], dtype=np.int32
    )
    cross_counts = np.asarray(
        [len(item.skeleton_nodes) for item in comparison.skeletons], dtype=np.int32
    )
    error_matrix = np.asarray(
        [[metrics[name][key] for key in METRIC_KEYS] for name in METHOD_NAMES]
    )
    np.savez_compressed(
        args.out_dir / "uniform_point_ddpnmt_results.npz",
        points=points,
        tetrahedra=tetrahedra,
        cell_volumes=volumes,
        sphere_centers=SPHERE_CENTERS,
        sphere_radius=np.asarray([SPHERE_RADIUS]),
        method_names=np.asarray(METHOD_NAMES),
        uniform_affine_complete=np.asarray(
            [args.uniform_affine_complete], dtype=bool
        ),
        metric_keys=np.asarray(METRIC_KEYS),
        error_matrix=error_matrix,
        interface_unknowns=np.asarray(
            [len(solutions[name].global_keys) for name in METHOD_NAMES]
        ),
        maximum_cell_velocity_rms_error=np.asarray(
            [metrics[name]["maximum_cell_velocity_rms_error"] for name in METHOD_NAMES]
        ),
        maximum_slice_pointwise_velocity_error=np.asarray(
            [
                metrics[name]["maximum_slice_pointwise_velocity_error"]
                for name in METHOD_NAMES
            ]
        ),
        maximum_local_mass_residual=np.asarray(
            [solutions[name].max_local_mass_residual for name in METHOD_NAMES]
        ),
        full_nodes_per_interface=full_counts,
        cross_nodes_per_interface=cross_counts,
        uniform_nodes_per_interface=uniform_counts,
        uniform_covering_radius=np.asarray(
            [item.covering_radius for item in comparison.uniform_sets]
        ),
        uniform_minimum_separation=np.asarray(
            [item.minimum_separation for item in comparison.uniform_sets]
        ),
        **representative_face_arrays(partition, comparison),
        **slice_arrays,
        **cell_arrays,
    )

    report = {
        "method": (
            "affine-complete saddle-seeded geodesic FPS "
            "uniform-point DDPNM/DDPNMT"
            if args.uniform_affine_complete
            else "constant-complete saddle-seeded geodesic FPS DDPNM/DDPNMT"
        ),
        "sampling_rule": "Ns = ceil(sampling_factor*sqrt(Nf)); seed = saddle-nearest vertex; remaining points = geodesic farthest-point sampling",
        "uniform_exact_modes": (
            "{1,s,t} tensor {normal,tangent_1,tangent_2} (nine vector modes)"
            if args.uniform_affine_complete
            else "constant normal/tangential component modes only"
        ),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "counts": {
            "tetrahedra": len(tetrahedra),
            "subdomains": 64,
            "interfaces": len(partition.interface_pairs),
            "full_scalar_nodes_sum": int(np.sum(full_counts)),
            "cross_scalar_nodes_sum": int(np.sum(cross_counts)),
            "uniform_scalar_nodes_sum": int(np.sum(uniform_counts)),
            "uniform_nodes_per_face_min_mean_max": [
                int(np.min(uniform_counts)),
                float(np.mean(uniform_counts)),
                int(np.max(uniform_counts)),
            ],
        },
        "uniform_sampling": {
            "covering_radius_min_mean_max": [
                float(min(item.covering_radius for item in comparison.uniform_sets)),
                float(np.mean([item.covering_radius for item in comparison.uniform_sets])),
                float(max(item.covering_radius for item in comparison.uniform_sets)),
            ],
            "minimum_separation_min_mean_max": [
                float(min(item.minimum_separation for item in comparison.uniform_sets)),
                float(np.mean([item.minimum_separation for item in comparison.uniform_sets])),
                float(max(item.minimum_separation for item in comparison.uniform_sets)),
            ],
        },
        "extension": {
            "maximum_uniform_normal_cardinal_residual": float(
                max(item.constraint_residual for item in comparison.uniform_normal_diagnostics)
            ),
            "maximum_uniform_normal_constant_residual": float(
                max(item.constant_reproduction_residual for item in comparison.uniform_normal_diagnostics)
            ),
            "maximum_uniform_vector_cardinal_residual": float(
                max(item.cardinal_residual for item in comparison.uniform_vector_diagnostics)
            ),
            "maximum_uniform_vector_constant_residual": float(
                max(item.constant_vector_reproduction_residual for item in comparison.uniform_vector_diagnostics)
            ),
            "maximum_uniform_normal_exact_mode_residual": float(
                max(
                    item.constant_reproduction_residual
                    for item in comparison.uniform_normal_diagnostics
                )
            ),
            "maximum_uniform_vector_exact_mode_residual": float(
                max(
                    item.constant_vector_reproduction_residual
                    for item in comparison.uniform_vector_diagnostics
                )
            ),
            "uniform_residual_represents_affine_modes": bool(
                args.uniform_affine_complete
            ),
        },
        "systems": diagnostics,
        "strict_errors_to_identical_mesh_FEM": metrics,
        "reference": {
            "mixed_dofs": reference.ndofs,
            "relative_linear_residual": reference.relative_linear_residual,
            "relative_mass_imbalance": reference.relative_mass_imbalance,
        },
        "timings_seconds": timings,
    }
    (args.out_dir / "uniform_point_ddpnmt_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done in {timings['total']:.1f} s: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

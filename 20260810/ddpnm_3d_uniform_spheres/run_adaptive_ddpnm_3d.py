#!/usr/bin/env python3
"""Three-dimensional adaptive DDPNM -> DDPNMT -> HODDPNM on the
uniform 27-sphere porous cube."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from ddpnm_core.algebra import hierarchy_error
from ddpnm_core.constants import LEVEL_NAMES
from ddpnm_core.fem_utils import solve_reference
from ddpnm_core.io import topology_arrays
from ddpnm_core.validation import finite_element_error_analysis

from ddpnm3d.geometry import SPHERE_CENTERS, SPHERE_RADIUS, build_partition
from ddpnm3d.hierarchy import (
    build_hierarchy_library,
    reconstruct_hierarchy_vertices,
    run_adaptive_hierarchy,
)
from ddpnm3d.visualization import evaluate_fem_ddpnm_slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Three-dimensional adaptive DDPNM -> DDPNMT -> HODDPNM on the "
            "uniform 27-sphere porous cube."
        )
    )
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
    parser.add_argument("--target-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--marking-theta", type=float, default=0.65)
    parser.add_argument("--max-marked-per-iteration", type=int, default=12)
    parser.add_argument("--max-iterations-per-phase", type=int, default=40)
    parser.add_argument("--reference-iterative-threshold", type=int, default=100_000)
    parser.add_argument("--reference-rtol", type=float, default=1.0e-9)
    parser.add_argument("--reference-restart", type=int, default=60)
    parser.add_argument("--reference-maxiter", type=int, default=120)
    parser.add_argument("--reference-ilu-drop-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--reference-ilu-fill-factor", type=float, default=6.0)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/adaptive_hierarchy")
    )
    return parser.parse_args()


def tetrahedron_volumes(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    coordinates = points[tetrahedra]
    matrices = np.stack(
        (
            coordinates[:, 1] - coordinates[:, 0],
            coordinates[:, 2] - coordinates[:, 0],
            coordinates[:, 3] - coordinates[:, 0],
        ),
        axis=1,
    )
    return np.abs(np.linalg.det(matrices)) / 6.0


def solution_stats(solution) -> dict:
    return {
        "method": solution.method_name,
        "interface_unknowns": len(solution.global_keys),
        "minimum_schur_eigenvalue": solution.min_schur_eigenvalue,
        "schur_symmetry_error": solution.symmetry_error,
        "relative_linear_residual": solution.relative_linear_residual,
        "maximum_interface_moment_residual": solution.max_mass_residual,
        "inlet_outward_flux": float(solution.boundary_fluxes["inlet"]),
        "outlet_outward_flux": float(solution.boundary_fluxes["outlet"]),
        "relative_mass_imbalance": float(
            abs(solution.boundary_fluxes["inlet"] + solution.boundary_fluxes["outlet"])
            / max(abs(solution.boundary_fluxes["outlet"]), 1.0e-30)
        ),
    }


def write_algorithm_markdown(out_dir: Path, tolerance: float, theta: float) -> None:
    text = f"""# 算法：三维分层自适应 DD-PNM

**输入：** 最大球子区域、严格孔喉鞍点截面、局部 Taylor--Hood 矩阵、容忍值 `TOL={tolerance:.3g}`。

1. 对每个孔隙子区域组装并分解一次局部 Stokes 矩阵，建立所有界面牵引模式的局部响应库。
2. 所有内部界面初始化为 DDPNM：每个界面仅保留一个常数法向牵引自由度。
3. 计算完整 DDPNMT 比较解：每个界面保留常数法向牵引和两个常数切向牵引，共 3 个自由度。
4. 若当前解与 DDPNMT 的归一化速度或压力差超过 `TOL`，以界面上的层级差构造指标，并按 Dörfler 准则（`theta={theta:.2f}`）标记界面；每条标记界面只升一级。
5. 重新装配并求解凝聚后的界面 Schur 系统，重复"比较—标记—升级—求解"，直到第一阶段达到容忍值。
6. 计算完整 HODDPNM 比较解：在每个界面的局部坐标 `(s,t)` 上，对法向和两个切向牵引均使用 `{{1,s,t}}` 完整 P1 模式，共 9 个自由度。
7. 以完整 HODDPNM 为第二阶段比较解，重复 Dörfler 标记与逐级升级，直至层级差不超过 `TOL`，或所有界面均达到 HODDPNM。
8. 用最终混合阶界面解重构全部局部速度、压力；传统整体 FEM 只用于事后误差验证，不参与界面选择。

> 这里的 Schur 补是局部有限元内部自由度的静态凝聚。自适应过程只求解界面牵引系数，且三个界面空间严格嵌套。
"""
    (out_dir / "ADAPTIVE_3D_ALGORITHM.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("[1/6] Building the saddle-cut, locally refined tetrahedral mesh...", flush=True)
    mesh_started = time.perf_counter()
    partition = build_partition(
        mesh_size=args.mesh_size,
        sphere_size=args.sphere_size,
        boundary_size=args.boundary_size,
        interface_size=args.interface_size,
        sphere_band=args.sphere_band,
        boundary_band=args.boundary_band,
        interface_band=args.interface_band,
        mesh_file=out_dir / "adaptive_uniform_27_spheres_partition.msh",
    )
    mesh_time = time.perf_counter() - mesh_started
    points, tetrahedra = topology_arrays(partition.mesh)
    volumes = tetrahedron_volumes(points, tetrahedra)
    print(
        f"      {len(tetrahedra)} tetrahedra, 64 subdomains, "
        f"{len(partition.interface_pairs)} interfaces",
        flush=True,
    )

    print("[2/6] Factoring local Stokes systems and building the 3D P1 traction library...", flush=True)
    library_started = time.perf_counter()
    library = build_hierarchy_library(
        partition,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
    )
    library_time = time.perf_counter() - library_started
    print(
        f"      sum local mixed dofs {sum(r.ndofs for r in library.local_responses)}, "
        f"max response symmetry error {max(r.symmetry_error for r in library.local_responses):.3e}",
        flush=True,
    )

    print("[3/6] Running the two-stage hierarchical adaptive promotion...", flush=True)
    adaptive_started = time.perf_counter()
    adaptive = run_adaptive_hierarchy(
        library,
        tolerance=args.target_tolerance,
        marking_theta=args.marking_theta,
        max_marked_per_iteration=args.max_marked_per_iteration,
        max_iterations_per_phase=args.max_iterations_per_phase,
    )
    adaptive_time = time.perf_counter() - adaptive_started
    final_counts = [
        int(np.sum(adaptive.final_solution.levels == level)) for level in range(3)
    ]
    print(
        f"      final DDPNM/DDPNMT/HODDPNM interfaces {final_counts}; "
        f"difference to full HODDPNM {adaptive.final_error_to_hoddpnm.combined:.3e}",
        flush=True,
    )

    print("[4/6] Solving traditional monolithic Taylor-Hood FEM on the identical mesh...", flush=True)
    reference_started = time.perf_counter()
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
    reference_time = time.perf_counter() - reference_started
    print(
        f"      {reference.ndofs} dofs, {reference.iterations} iterations, "
        f"true residual {reference.relative_linear_residual:.3e}",
        flush=True,
    )

    print("[5/6] Computing strict P2-P1 errors for all hierarchy levels...", flush=True)
    validation_started = time.perf_counter()
    named_solutions = {
        "DDPNM": adaptive.initial_ddpnm,
        "DDPNMT": adaptive.full_ddpnmt,
        "HODDPNM": adaptive.full_hoddpnm,
        "adaptive_final": adaptive.final_solution,
    }
    metrics: dict[str, dict] = {}
    vertex_fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, solution in named_solutions.items():
        print(f"      validating {name}...", flush=True)
        metric, _, _ = finite_element_error_analysis(
            partition, solution, reference, volumes
        )
        metrics[name] = metric
        vertex_fields[name] = reconstruct_hierarchy_vertices(library, solution)
    slice_data = evaluate_fem_ddpnm_slice(
        partition,
        adaptive.final_solution,
        reference,
        points,
        tetrahedra,
        z_value=0.48,
    )
    validation_time = time.perf_counter() - validation_started

    print("[6/6] Writing adaptive history, report, fields and plotting data...", flush=True)
    with (out_dir / "adaptive_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "phase",
                "iteration",
                "velocity_difference",
                "pressure_difference",
                "combined_difference",
                "DDPNM_interfaces",
                "DDPNMT_interfaces",
                "HODDPNM_interfaces",
                "interface_unknowns",
                "marked_interfaces",
            ]
        )
        for item in adaptive.history:
            writer.writerow(
                [
                    item.phase,
                    item.iteration,
                    item.error.velocity,
                    item.error.pressure,
                    item.error.combined,
                    *item.counts,
                    item.interface_unknowns,
                    " ".join(str(interface) for interface in item.marked_interfaces),
                ]
            )

    with (out_dir / "method_error_metrics.csv").open(
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
                "pressure_mean_aligned_relative_L2",
                "outlet_flux_relative_error",
            ]
        )
        for name, solution in named_solutions.items():
            metric = metrics[name]
            writer.writerow(
                [
                    name,
                    len(solution.global_keys),
                    metric["velocity_relative_l2"],
                    metric["velocity_relative_broken_h1_seminorm"],
                    metric["pressure_raw_relative_l2"],
                    metric["pressure_mean_aligned_relative_l2"],
                    metric["outlet_flux_relative_error"],
                ]
            )

    parameters = vars(args).copy()
    parameters["out_dir"] = str(out_dir)
    report = {
        "method": "3D adaptive DDPNM -> DDPNMT -> HODDPNM",
        "geometry": "unit cube minus a uniform 3x3x3 array of 27 spheres",
        "interface_spaces": {
            "DDPNM": "one P0 normal-traction coefficient per interface",
            "DDPNMT": "P0 normal plus two P0 tangential-traction coefficients per interface",
            "HODDPNM": (
                "three vector components times the complete facewise P1 basis "
                "{1,s,t}: nine coefficients per interface; local FE interiors are Schur-condensed"
            ),
        },
        "adaptive_estimator": (
            "two-stage embedded hierarchy on reconstructed interface/vertex fields; "
            "traditional FEM is excluded from marking"
        ),
        "parameters": parameters,
        "counts": {
            "solid_spheres": 27,
            "subdomains": 64,
            "interfaces": len(partition.interface_pairs),
            "tetrahedra": len(tetrahedra),
            "vertices": len(points),
            "final_DDPNM_interfaces": final_counts[0],
            "final_DDPNMT_interfaces": final_counts[1],
            "final_HODDPNM_interfaces": final_counts[2],
        },
        "systems": {
            name: solution_stats(solution)
            for name, solution in named_solutions.items()
        },
        "final_error_to_full_HODDPNM": adaptive.final_error_to_hoddpnm.__dict__,
        "strict_errors_to_identical_mesh_FEM": metrics,
        "traditional_fem": {
            "mixed_dofs": reference.ndofs,
            "matrix_nonzeros": reference.matrix_nnz,
            "solver": reference.solver_method,
            "iterations": reference.iterations,
            "relative_linear_residual": reference.relative_linear_residual,
            "relative_mass_imbalance": reference.relative_mass_imbalance,
            "relative_energy_residual": reference.relative_energy_residual,
        },
        "iterations": [
            {
                "phase": item.phase,
                "iteration": item.iteration,
                "error": item.error.__dict__,
                "level_counts": list(item.counts),
                "interface_unknowns": item.interface_unknowns,
                "marked_interfaces": list(item.marked_interfaces),
            }
            for item in adaptive.history
        ],
        "timings_seconds": {
            "mesh": mesh_time,
            "local_response_library": library_time,
            "adaptive_hierarchy": adaptive_time,
            "traditional_fem": reference_time,
            "strict_validation": validation_time,
            "total": time.perf_counter() - started,
        },
    }
    (out_dir / "adaptive_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_algorithm_markdown(
        out_dir, args.target_tolerance, args.marking_theta
    )

    arrays: dict[str, np.ndarray] = {
        "points": points,
        "tetrahedra": tetrahedra,
        "sphere_centers": SPHERE_CENTERS,
        "sphere_radius": np.asarray([SPHERE_RADIUS]),
        "maximal_ball_centers": np.asarray(
            [ball.center for ball in partition.maximal_balls]
        ),
        "interface_pairs": np.asarray(partition.interface_pairs, dtype=np.int32),
        "interface_centers": partition.interface_centers,
        "interface_normals": partition.interface_normals,
        "interface_areas": partition.interface_areas,
        "final_levels": adaptive.final_solution.levels,
        "history_velocity": np.asarray([item.error.velocity for item in adaptive.history]),
        "history_pressure": np.asarray([item.error.pressure for item in adaptive.history]),
        "history_combined": np.asarray([item.error.combined for item in adaptive.history]),
        "history_counts": np.asarray([item.counts for item in adaptive.history], dtype=np.int32),
        "history_unknowns": np.asarray([item.interface_unknowns for item in adaptive.history]),
        "history_phase": np.asarray([item.phase for item in adaptive.history]),
        "target_tolerance": np.asarray([adaptive.tolerance]),
    }
    for name, (velocity, pressure) in vertex_fields.items():
        arrays[f"u_{name}"] = velocity
        arrays[f"p_{name}"] = pressure
    arrays.update(slice_data)
    np.savez_compressed(out_dir / "adaptive_3d_results.npz", **arrays)

    final_metric = metrics["adaptive_final"]
    print(
        "Done: adaptive/FEM relative errors "
        f"L2(u)={final_metric['velocity_relative_l2']:.2%}, "
        f"broken-H1(u)={final_metric['velocity_relative_broken_h1_seminorm']:.2%}, "
        f"L2(p)={final_metric['pressure_raw_relative_l2']:.2%}, "
        f"flow={final_metric['outlet_flux_relative_error']:.2%}",
        flush=True,
    )
    print(f"Outputs: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

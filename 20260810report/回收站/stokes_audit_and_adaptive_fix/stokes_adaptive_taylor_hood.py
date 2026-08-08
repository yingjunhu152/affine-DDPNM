from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy import ndimage as ndi
from scipy.sparse.linalg import LinearOperator, gmres, splu, spsolve
from scipy.spatial import cKDTree

import real_porous_hoddpnm_validation as base
import render_final_state_figures as final_state_renderer


METHODS = ("PNM", "DDPNM", "DDPNMT", "HODDPNM")
METHOD_LEVEL = {name: i for i, name in enumerate(METHODS)}
METHOD_COLOR = {
    "PNM": "#5c7cfa",
    "DDPNM": "#37b24d",
    "DDPNMT": "#f59f00",
    "HODDPNM": "#e03131",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume-npy", type=Path, default=Path("data/berea_100_to_300.npz"))
    parser.add_argument("--pore-value", type=float, default=1.0)
    parser.add_argument("--crop", default="20:36,150:166,30:46")
    parser.add_argument("--regions", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--upgrade-fraction", type=float, default=0.25)
    parser.add_argument(
        "--adaptive-strategy",
        choices=(
            "cumulative",
            "capped-high-error",
            "capped-monotone-high-error",
            "tolerance-driven",
            "proportional-interface",
            "staged-proportional-hoddpnm",
            "a-priori-stokes-gain",
        ),
        default="tolerance-driven",
    )
    parser.add_argument("--indicator", choices=("residual", "posterior-defect", "true-error", "geometry-prior", "spectral-geometry-prior", "stokes-spectral-geometry-prior"), default="residual")
    parser.add_argument("--reference-solve", choices=("none", "before", "after"), default="none")
    parser.add_argument("--error-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--error-metric", choices=("estimated-residual", "posterior-defect", "geometry-prior", "spectral-geometry-prior", "stokes-spectral-geometry-prior", "velocity", "pressure", "max", "combined"), default="estimated-residual")
    parser.add_argument("--dorfler-theta", type=float, default=0.65)
    parser.add_argument("--max-upgrades-per-cycle", type=int, default=3)
    parser.add_argument("--minimum-stop-stage", type=int, default=0)
    parser.add_argument("--initial-stage", type=int, default=0)
    parser.add_argument("--interface-fractions", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--ddpnmt-interface-fraction", type=float, default=0.10)
    parser.add_argument("--geometry-prior-weights", default="1.0,1.1,0.9,0.65,0.75,0.7,0.55")
    parser.add_argument("--geometry-stage-tail", default="1.0,0.62,0.42,0.28,0.16,0.07,0.015")
    parser.add_argument("--posterior-defect-weights", default="1.0,0.75,0.65,0.25")
    parser.add_argument("--spectral-boundary-max-nodes", type=int, default=72)
    parser.add_argument("--spectral-tail-modes", type=int, default=24)
    parser.add_argument("--spectral-weight", type=float, default=1.25)
    parser.add_argument("--spectral-ridge", type=float, default=1.0e-10)
    parser.add_argument("--proportional-node-set", choices=("interface-only", "interface-plus-interior"), default="interface-plus-interior")
    parser.add_argument("--pressure-indicator-weight", type=float, default=0.05)
    parser.add_argument("--hoddpnm-region-fraction", type=float, default=0.125)
    parser.add_argument("--ddpnmt-region-fraction", type=float, default=0.0)
    parser.add_argument("--ddpnm-region-fraction", type=float, default=0.25)
    parser.add_argument("--active-dof-cap", type=float, default=0.80)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--interface-thickness", type=float, default=1.05)
    parser.add_argument("--pressure-stabilization", type=float, default=1.0e-10)
    parser.add_argument("--restricted-solver", choices=("direct", "schur-gmres"), default="schur-gmres")
    parser.add_argument("--schur-preconditioner", choices=("ilu", "none"), default="ilu")
    parser.add_argument("--schur-rtol", type=float, default=1.0e-10)
    parser.add_argument("--schur-atol", type=float, default=1.0e-12)
    parser.add_argument("--schur-restart", type=int, default=80)
    parser.add_argument("--schur-maxiter", type=int, default=300)
    parser.add_argument("--ilu-drop-tol", type=float, default=1.0e-4)
    parser.add_argument("--ilu-fill-factor", type=float, default=12.0)
    parser.add_argument("--save-iteration-artifacts", action="store_true", help="Save cycle npz files, progression plots, and diagnostic iteration CSVs.")
    parser.add_argument("--skip-final-figures", action="store_true", help="Skip final-state microscopy-style PNG figures.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/final_state_figures"))
    args = parser.parse_args()
    if args.reference_solve == "none" and args.indicator == "true-error":
        raise SystemExit("--indicator true-error requires --reference-solve before or after; use --indicator residual for a FEM-independent adaptive run")
    if args.reference_solve == "none" and args.error_metric not in ("estimated-residual", "posterior-defect", "geometry-prior", "spectral-geometry-prior", "stokes-spectral-geometry-prior"):
        raise SystemExit("--error-metric velocity/pressure/max/combined requires a FEM reference; use a FEM-independent metric instead")
    if args.reference_solve == "after" and args.indicator == "true-error":
        raise SystemExit("--reference-solve after cannot be used with --indicator true-error because the reference is computed only after adaptation")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()
    memory_trace = {"start": base.process_memory_mib()}

    pore = load_pore(args.volume_npy, args.pore_value, parse_crop(args.crop))
    mesh = base.build_real_porous_mesh(pore, args.voxel_size, args.regions)
    assembled = base.assemble_taylor_hood_system(mesh, args.pressure_stabilization, args.interface_thickness)
    memory_trace["after_fenicsx_assembly"] = base.process_memory_mib()

    A = assembled["A"].tocsr()
    b = np.asarray(assembled["b"], dtype=float)

    reference = None
    fem_time = None
    if args.reference_solve == "before":
        t0 = time.perf_counter()
        reference = spsolve(A, b)
        fem_time = time.perf_counter() - t0
        memory_trace["after_fem_reference"] = base.process_memory_mib()

    dof_data = build_dof_data(mesh, assembled)
    dof_data["adaptive_strategy"] = args.adaptive_strategy
    dof_data["interface_fractions"] = np.asarray(parse_float_list(args.interface_fractions), dtype=float)
    dof_data["ddpnmt_interface_fraction"] = float(args.ddpnmt_interface_fraction)
    dof_data["proportional_node_set"] = args.proportional_node_set
    neighbor_regions = region_neighbors(mesh)
    geometry_prior = build_geometry_prior(mesh, dof_data, neighbor_regions, parse_float_sequence(args.geometry_prior_weights), parse_float_sequence(args.geometry_stage_tail))
    if args.indicator == "spectral-geometry-prior" or args.error_metric == "spectral-geometry-prior":
        geometry_prior = build_spectral_geometry_prior(
            A,
            dof_data,
            geometry_prior,
            assembled["fixed_dofs"],
            max_boundary_nodes=args.spectral_boundary_max_nodes,
            tail_modes=args.spectral_tail_modes,
            spectral_weight=args.spectral_weight,
            ridge=args.spectral_ridge,
        )
    if args.indicator == "stokes-spectral-geometry-prior" or args.error_metric == "stokes-spectral-geometry-prior":
        geometry_prior = build_stokes_spectral_geometry_prior(
            A,
            dof_data,
            geometry_prior,
            assembled["fixed_dofs"],
            max_boundary_nodes=args.spectral_boundary_max_nodes,
            tail_modes=args.spectral_tail_modes,
            spectral_weight=args.spectral_weight,
            ridge=args.spectral_ridge,
        )
    dof_data["geometry_prior"] = geometry_prior
    if args.indicator in ("geometry-prior", "spectral-geometry-prior", "stokes-spectral-geometry-prior"):
        dof_data["v_node_scores"] = geometry_prior["node_scores"]
    levels = np.full(args.regions, max(0, int(args.initial_stage)), dtype=np.int32)
    rows: list[dict[str, float | int]] = []
    posterior_rows: list[dict[str, float | int]] = []
    solutions: list[np.ndarray] = []
    residual_baseline = None
    geometry_prior_baseline = None
    posterior_defect_baseline = None

    for cycle in range(args.cycles + 1):
        active = active_dofs_for_levels(levels, dof_data, neighbor_regions, assembled["fixed_dofs"])
        t0 = time.perf_counter()
        solution, solver_info = solve_restricted_stokes(
            A,
            b,
            active,
            interface_mask=assembled["interface_mask"],
            fixed_dofs=assembled["fixed_dofs"],
            solver=args.restricted_solver,
            schur_preconditioner=args.schur_preconditioner,
            rtol=args.schur_rtol,
            atol=args.schur_atol,
            restart=args.schur_restart,
            maxiter=args.schur_maxiter,
            ilu_drop_tol=args.ilu_drop_tol,
            ilu_fill_factor=args.ilu_fill_factor,
        )
        solve_time = time.perf_counter() - t0
        cycle_memory = base.process_memory_mib()
        solutions.append(solution)

        algebraic_residual = A @ solution - b
        residual_rel_error = float(np.linalg.norm(algebraic_residual) / max(np.linalg.norm(b), 1.0e-300))
        if residual_baseline is None:
            residual_baseline = max(residual_rel_error, 1.0e-300)
        estimated_rel_error = float(residual_rel_error / residual_baseline)
        residual_node_scores = np.linalg.norm(np.abs(algebraic_residual[dof_data["v_node_dofs"]]), axis=1)
        if args.indicator in ("geometry-prior", "spectral-geometry-prior", "stokes-spectral-geometry-prior"):
            dof_data["v_node_scores"] = geometry_prior["node_scores"]
        else:
            dof_data["v_node_scores"] = residual_node_scores
        residual_eta = region_residual_indicator(A, b, solution, dof_data["mixed_region"], args.regions, assembled["fixed_dofs"])
        posterior_eta, posterior_defect_error, posterior_components, posterior_node_scores = region_posterior_defect_indicator(
            A,
            b,
            solution,
            active,
            dof_data,
            args.regions,
            assembled["fixed_dofs"],
            parse_float_sequence(args.posterior_defect_weights),
        )
        if posterior_defect_baseline is None:
            posterior_defect_baseline = max(posterior_defect_error, 1.0e-300)
        posterior_defect_relative = float(posterior_defect_error / posterior_defect_baseline)
        if args.indicator == "posterior-defect":
            dof_data["v_node_scores"] = posterior_node_scores
        geometry_eta, geometry_prior_error = region_geometry_prior_indicator(levels, dof_data)
        if geometry_prior_baseline is None:
            geometry_prior_baseline = max(geometry_prior_error, 1.0e-300)
        geometry_prior_relative = float(geometry_prior_error / geometry_prior_baseline)
        if reference is not None:
            velocity_error = (solution[assembled["mapV"]] - reference[assembled["mapV"]]).reshape((-1, 3))
            dof_data["v_node_scores"] = np.linalg.norm(velocity_error, axis=1) if args.indicator == "true-error" else residual_node_scores
            if args.indicator in ("geometry-prior", "spectral-geometry-prior", "stokes-spectral-geometry-prior"):
                dof_data["v_node_scores"] = geometry_prior["node_scores"]
            if args.indicator == "posterior-defect":
                dof_data["v_node_scores"] = posterior_node_scores
            velocity_ref = reference[assembled["mapV"]].reshape((-1, 3))
            pressure_error = solution[assembled["mapQ"]] - reference[assembled["mapQ"]]
            pressure_ref = reference[assembled["mapQ"]]
            velocity_rel_error = vector_rel_l2(velocity_error, velocity_ref)
            pressure_rel_error = scalar_rel_l2(pressure_error, pressure_ref)
            true_eta = region_true_error_indicator(
                velocity_error,
                pressure_error,
                dof_data,
                args.regions,
                pressure_weight=args.pressure_indicator_weight,
            )
        else:
            velocity_rel_error = float("nan")
            pressure_rel_error = float("nan")
            true_eta = np.zeros(args.regions, dtype=float)
        if args.indicator == "true-error":
            eta = true_eta
        elif args.indicator == "posterior-defect":
            eta = posterior_eta
        elif args.indicator in ("geometry-prior", "spectral-geometry-prior", "stokes-spectral-geometry-prior"):
            eta = geometry_eta
        else:
            eta = residual_eta
        counts = {name: int(np.count_nonzero(levels == METHOD_LEVEL[name])) for name in METHODS}
        if args.adaptive_strategy in ("proportional-interface", "staged-proportional-hoddpnm", "a-priori-stokes-gain"):
            counts = {
                "PNM": int(np.count_nonzero(levels == 0)),
                "DDPNM": int(np.count_nonzero(levels == 1)),
                "DDPNMT": int(np.count_nonzero(levels == 2)),
                "HODDPNM": int(np.count_nonzero(levels >= 3)),
            }
        if args.adaptive_strategy == "proportional-interface":
            fraction_stats = proportional_fraction_stats(levels, dof_data)
        elif args.adaptive_strategy in ("staged-proportional-hoddpnm", "a-priori-stokes-gain"):
            fraction_stats = staged_proportional_fraction_stats(levels, dof_data)
        else:
            fraction_stats = {}
        if args.error_metric == "posterior-defect":
            target_error = posterior_defect_relative
        elif args.error_metric in ("geometry-prior", "spectral-geometry-prior", "stokes-spectral-geometry-prior"):
            target_error = geometry_prior_relative
        else:
            target_error = stopping_error(velocity_rel_error, pressure_rel_error, estimated_rel_error, args.error_metric)
        active_count = int(np.count_nonzero(active))
        known_fixed_count = int(solver_info["n_fixed_known"])
        active_free_count = active_count - known_fixed_count
        row = {
            "cycle": int(cycle),
            "active_dofs": active_count,
            "active_dof_ratio": float(active_count / len(active)),
            "active_free_dofs": active_free_count,
            "active_free_dof_ratio": float(active_free_count / len(active)),
            "known_fixed_dofs": known_fixed_count,
            "pressure_time_seconds": float(solve_time),
            "working_set_mib": float(cycle_memory["working_set_mib"]),
            "peak_working_set_mib": float(cycle_memory["peak_working_set_mib"]),
            "pagefile_mib": float(cycle_memory["pagefile_mib"]),
            "peak_pagefile_mib": float(cycle_memory["peak_pagefile_mib"]),
            "velocity_rel_l2_error": float(velocity_rel_error),
            "pressure_rel_l2_error": float(pressure_rel_error),
            "target_error": float(target_error),
            "error_tolerance": float(args.error_tolerance),
            "converged_to_tolerance": bool(target_error <= args.error_tolerance),
            "mixed_residual_rel_l2": float(residual_rel_error),
            "estimated_residual_relative_to_initial": float(estimated_rel_error),
            "indicator": args.indicator,
            "max_residual_indicator": float(np.max(residual_eta)),
            "max_posterior_defect_indicator": float(np.max(posterior_eta)),
            "max_true_error_indicator": float(np.max(true_eta)),
            "max_geometry_prior_indicator": float(np.max(geometry_eta)),
            "posterior_defect_error": float(posterior_defect_error),
            "posterior_defect_relative_to_initial": float(posterior_defect_relative),
            "geometry_prior_error": float(geometry_prior_error),
            "geometry_prior_relative_to_initial": float(geometry_prior_relative),
            "solver": str(solver_info["solver"]),
            "schur_preconditioner": str(solver_info["preconditioner"]),
            "schur_boundary_dofs": int(solver_info["n_boundary"]),
            "schur_interior_dofs": int(solver_info["n_interior"]),
            "schur_converged": bool(solver_info.get("converged", True)),
            "schur_iterations": int(solver_info["iterations"]),
            "schur_gmres_info": int(solver_info["gmres_info"]),
            "schur_relative_residual": float(solver_info["relative_residual"]),
            "schur_operator_relative_residual": float(solver_info.get("schur_relative_residual", solver_info["relative_residual"])),
            **fraction_stats,
            **{f"n_{name}": counts[name] for name in METHODS},
        }
        rows.append(row)
        for component_row in posterior_components:
            posterior_rows.append({"cycle": int(cycle), **component_row})
        if args.save_iteration_artifacts:
            np.savez_compressed(
                args.out_dir / f"cycle_{cycle:02d}.npz",
                levels=levels,
                solution=solution,
                active=active,
                eta=eta,
            )
        stop_stage_ok = int(np.max(levels)) >= int(args.minimum_stop_stage)
        if cycle == args.cycles or (
            args.adaptive_strategy in ("tolerance-driven", "proportional-interface", "staged-proportional-hoddpnm", "a-priori-stokes-gain")
            and target_error <= args.error_tolerance
            and stop_stage_ok
        ):
            break
        levels = adaptive_update_levels(
            levels,
            eta,
            args,
            dof_data,
            neighbor_regions,
            assembled["fixed_dofs"],
        )

    final_solution = solutions[-1]
    if args.reference_solve == "after":
        t0 = time.perf_counter()
        reference = spsolve(A, b)
        fem_time = time.perf_counter() - t0
        memory_trace["after_fem_reference"] = base.process_memory_mib()

    if reference is not None:
        fields = base.vertex_output_fields(mesh, assembled, reference, final_solution)
        final_velocity_error = fields["hodd_velocity"] - fields["fem_velocity"]
        final_pressure_error = fields["hodd_pressure"] - fields["fem_pressure"]
        velocity_error_stats = base.vector_error_stats(final_velocity_error, fields["fem_velocity"])
        pressure_error_stats = base.scalar_error_stats(final_pressure_error, fields["fem_pressure"])
    else:
        fields = vertex_solution_fields(mesh, assembled, final_solution)
        velocity_error_stats = None
        pressure_error_stats = None
    base.save_vtu(mesh, fields, args.out_dir / "real_porous_hoddpnm_solution.vtu")
    final_figure_outputs = {}
    if not args.skip_final_figures:
        final_figure_outputs = final_state_renderer.render_final_state_figures(mesh, levels, fields, dof_data, args.out_dir)
    if args.save_iteration_artifacts:
        render_method_map(mesh, levels, args.out_dir / "final_region_method_map.png")
        render_progression(mesh, args.out_dir, rows, args.out_dir / "cycle_method_progression.png")
        plot_history(rows, args.out_dir / "adaptive_stokes_error_cost_history.png")
    memory_trace["end"] = base.process_memory_mib()

    final = rows[-1]
    if velocity_error_stats is not None and pressure_error_stats is not None:
        final["validation_velocity_rel_l2_error"] = float(velocity_error_stats["l2_rel"])
        final["validation_pressure_rel_l2_error"] = float(pressure_error_stats["l2_rel"])
    summary = {
        "method": "adaptive FEniCSx Taylor-Hood P2-P1 Stokes on real Berea pore mesh",
        "important_note": "Every adaptive level solves a restricted subspace of the same FEniCSx Taylor-Hood Stokes matrix. No graph pressure equation is used.",
        "linear_solver": {
            "restricted_solver": args.restricted_solver,
            "schur_preconditioner": args.schur_preconditioner,
            "ilu_drop_tol": args.ilu_drop_tol,
            "ilu_fill_factor": args.ilu_fill_factor,
            "schur_rtol": args.schur_rtol,
            "schur_atol": args.schur_atol,
            "schur_restart": args.schur_restart,
            "schur_maxiter": args.schur_maxiter,
        },
        "adaptive_policy": {
            "strategy": args.adaptive_strategy,
            "upgrade_fraction": args.upgrade_fraction,
            "hoddpnm_region_fraction": args.hoddpnm_region_fraction,
            "ddpnmt_region_fraction": args.ddpnmt_region_fraction,
            "ddpnm_region_fraction": args.ddpnm_region_fraction,
            "active_dof_cap": args.active_dof_cap,
            "indicator": args.indicator,
            "error_tolerance": args.error_tolerance,
            "error_metric": args.error_metric,
            "dorfler_theta": args.dorfler_theta,
            "max_upgrades_per_cycle": args.max_upgrades_per_cycle,
            "minimum_stop_stage": args.minimum_stop_stage,
            "initial_stage": args.initial_stage,
            "interface_fractions": parse_float_list(args.interface_fractions),
            "proportional_node_set": args.proportional_node_set,
            "pressure_indicator_weight": args.pressure_indicator_weight,
            "geometry_prior_weights": parse_float_sequence(args.geometry_prior_weights),
            "geometry_stage_tail": parse_float_sequence(args.geometry_stage_tail),
            "spectral_boundary_max_nodes": args.spectral_boundary_max_nodes,
            "spectral_tail_modes": args.spectral_tail_modes,
            "spectral_weight": args.spectral_weight,
            "spectral_ridge": args.spectral_ridge,
            "posterior_defect_weights": parse_float_sequence(args.posterior_defect_weights),
            "note": "tolerance-driven starts from all-PNM, checks a global error target, and upgrades high-indicator regions by one fidelity level using Doerfler marking until the target is reached or cycles are exhausted. proportional-interface upgrades the released interface-DOF fraction instead of only changing whole-region method levels.",
        },
        "level_meaning": {
            "PNM": "global low-order Stokes subspace: P1-like velocity vertices plus P1 pressure",
            "DDPNM": "adds all P2 velocity dofs in selected regions",
            "DDPNMT": "adds selected regions plus adjacent-interface P2 dofs",
            "HODDPNM": "adds selected regions with one-neighbor overlap",
        },
        "crop": args.crop,
        "domain_shape_voxels": list(mesh.domain_shape),
        "pore_voxels": int(len(mesh.pore_voxels)),
        "regions": int(args.regions),
        "n_vertices": int(len(mesh.coords)),
        "n_tets": int(len(mesh.cells)),
        "mixed_dofs": int(len(b)),
        "velocity_dofs": int(len(assembled["mapV"])),
        "pressure_dofs": int(len(assembled["mapQ"])),
        "timings_seconds": {
            "fem_sparse_direct_reference": float(fem_time) if fem_time is not None else None,
            "final_adaptive_restricted_stokes": float(final["pressure_time_seconds"]),
            "total_wall_time": float(time.perf_counter() - t_total),
        },
        "memory_trace_mib": memory_trace,
        "final": final,
        "history": rows if args.save_iteration_artifacts else [],
        "history_omitted_from_outputs": not args.save_iteration_artifacts,
        "geometry_prior": geometry_prior["summary"],
        "errors_velocity_vertices": velocity_error_stats,
        "errors_pressure_vertices": pressure_error_stats,
        "outputs": {
            "vtu": "real_porous_hoddpnm_solution.vtu",
            "final_figures": final_figure_outputs,
            "method_map_legacy": "final_region_method_map.png" if args.save_iteration_artifacts else None,
            "iteration_artifacts": "saved only with --save-iteration-artifacts",
        },
    }
    (args.out_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_iteration_artifacts:
        write_csv(args.out_dir / "adaptive_stokes_history.csv", rows)
        write_csv(args.out_dir / "geometry_region_prior.csv", geometry_prior["rows"])
        write_csv(args.out_dir / "posterior_defect_components.csv", posterior_rows)
    write_report(args.out_dir / "ADAPTIVE_STOKES_REPORT.md", summary)

    print(f"FEniCSx TH Stokes dofs: {len(b)}")
    print(f"regions: {args.regions}, pore voxels: {len(mesh.pore_voxels)}")
    if fem_time is None:
        print("FEM reference time: not computed")
    else:
        print(f"FEM reference time: {fem_time:.3f} s")
    print(f"final adaptive restricted Stokes time: {final['pressure_time_seconds']:.3f} s")
    print(f"final active dofs including known fixed: {final['active_dofs']} ({100.0 * final['active_dof_ratio']:.1f}%)")
    print(f"final active free dofs: {final['active_free_dofs']} ({100.0 * final['active_free_dof_ratio']:.1f}%)")
    print(f"known fixed dofs removed from solve: {final['known_fixed_dofs']}")
    print(f"final target error ({args.error_metric}): {final['target_error']:.3e}, tolerance: {args.error_tolerance:.3e}")
    print(f"converged to tolerance: {final['converged_to_tolerance']}")
    print(f"final restricted solver: {final['solver']} / {final['schur_preconditioner']}")
    print(f"final Schur GMRES iterations: {final['schur_iterations']}, residual: {final['schur_operator_relative_residual']:.3e}")
    print(f"final counts: {', '.join(f'{name}={final['n_' + name]}' for name in METHODS)}")
    printed_velocity_error = final.get("validation_velocity_rel_l2_error", final["velocity_rel_l2_error"])
    printed_pressure_error = final.get("validation_pressure_rel_l2_error", final["pressure_rel_l2_error"])
    print(f"velocity rel L2 error: {printed_velocity_error:.3e}")
    print(f"pressure rel L2 error: {printed_pressure_error:.3e}")
    print(f"wrote {args.out_dir / 'validation_summary.json'}")


def load_pore(path: Path, pore_value: float, crop: tuple[slice, slice, slice]) -> np.ndarray:
    arr = np.load(path)
    volume = arr[sorted(arr.files)[0]] if isinstance(arr, np.lib.npyio.NpzFile) else arr
    pore = np.asarray(volume[crop] == pore_value, dtype=bool)
    return base.keep_largest_pore_component(pore)


def build_dof_data(mesh, assembled: dict[str, object]) -> dict[str, np.ndarray]:
    map_v = np.asarray(assembled["mapV"], dtype=np.int64)
    map_q = np.asarray(assembled["mapQ"], dtype=np.int64)
    v_coords = np.asarray(assembled["V_coords"], dtype=float)
    q_coords = np.asarray(assembled["Q_coords"], dtype=float)
    v_region = point_regions(mesh, v_coords)
    q_region = point_regions(mesh, q_coords)
    v_node_dofs = map_v.reshape((-1, 3))
    vertex_nodes = is_grid_vertex(v_coords, mesh.voxel_size)
    interface_mask = np.asarray(assembled["interface_mask"], dtype=bool)
    interface_nodes = np.any(interface_mask[v_node_dofs], axis=1) & ~vertex_nodes
    p2_interior_nodes = (~vertex_nodes) & (~interface_nodes)
    mixed_region = -np.ones(int(max(map_v.max(), map_q.max()) + 1), dtype=np.int32)
    for node, region in enumerate(v_region):
        mixed_region[v_node_dofs[node]] = int(region)
    mixed_region[map_q] = q_region
    return {
        "map_v": map_v,
        "map_q": map_q,
        "v_coords": v_coords,
        "q_coords": q_coords,
        "v_node_dofs": v_node_dofs,
        "v_region": v_region,
        "q_region": q_region,
        "vertex_nodes": vertex_nodes,
        "interface_nodes": interface_nodes,
        "p2_interior_nodes": p2_interior_nodes,
        "mixed_region": mixed_region,
        "v_node_scores": np.zeros(len(v_coords), dtype=float),
    }


def vertex_solution_fields(mesh, assembled: dict[str, object], solution: np.ndarray) -> dict[str, np.ndarray]:
    v_coords = assembled["V_coords"]
    q_coords = assembled["Q_coords"]
    map_v = assembled["mapV"]
    map_q = assembled["mapQ"]
    tree_v = cKDTree(v_coords)
    _, nearest_v = tree_v.query(mesh.coords, k=1)
    tree_q = cKDTree(q_coords)
    _, nearest_q = tree_q.query(mesh.coords, k=1)
    velocity = solution[map_v].reshape((-1, 3))[nearest_v]
    pressure = solution[map_q][nearest_q]
    return {
        "hodd_velocity": velocity,
        "hodd_pressure": pressure,
        "estimated_velocity": velocity,
        "estimated_pressure": pressure,
    }


def point_regions(mesh, points: np.ndarray) -> np.ndarray:
    centers = (mesh.pore_voxels.astype(float) + 0.5) * mesh.voxel_size
    labels = mesh.voxel_labels[tuple(mesh.pore_voxels.T)]
    tree = cKDTree(centers)
    _, nearest = tree.query(points, k=1)
    return labels[nearest].astype(np.int32)


def is_grid_vertex(points: np.ndarray, voxel_size: float) -> np.ndarray:
    scaled = points / voxel_size
    return np.max(np.abs(scaled - np.round(scaled)), axis=1) < 1.0e-8


def region_neighbors(mesh) -> list[set[int]]:
    n_regions = int(mesh.voxel_labels.max()) + 1
    neighbors = [set() for _ in range(n_regions)]
    labels = mesh.voxel_labels
    for x, y, z in mesh.pore_voxels:
        r = int(labels[x, y, z])
        for nx, ny, nz in base.neighbor_voxels(int(x), int(y), int(z), mesh.domain_shape):
            s = int(labels[nx, ny, nz])
            if s >= 0 and s != r:
                neighbors[r].add(s)
                neighbors[s].add(r)
    return neighbors


def build_geometry_prior(mesh, dof_data: dict[str, np.ndarray], neighbors: list[set[int]], weights: list[float], stage_tail: list[float]) -> dict[str, object]:
    labels = mesh.voxel_labels
    pore_lookup = np.zeros(mesh.domain_shape, dtype=bool)
    pore_lookup[tuple(mesh.pore_voxels.T)] = True
    solid_distance = ndi.distance_transform_edt(pore_lookup) * mesh.voxel_size
    n_regions = int(labels.max()) + 1
    feature_names = [
        "solid_surface_density",
        "interface_density",
        "neighbor_degree",
        "bbox_void_deficit",
        "constriction_risk",
        "x_section_risk",
        "x_interface_cut_fraction",
    ]
    raw = np.zeros((n_regions, len(feature_names)), dtype=float)
    rows: list[dict[str, float | int]] = []

    for region in range(n_regions):
        voxels = mesh.pore_voxels[labels[tuple(mesh.pore_voxels.T)] == region]
        count = int(len(voxels))
        if count == 0:
            continue
        mins = voxels.min(axis=0)
        maxs = voxels.max(axis=0)
        extents = maxs - mins + 1
        bbox_volume = int(np.prod(extents))
        pore_fraction_bbox = float(count / max(bbox_volume, 1))
        surface_faces = 0
        interface_faces = 0
        x_interface_faces = 0
        for x, y, z in voxels:
            for nx, ny, nz in base.neighbor_voxels(int(x), int(y), int(z), mesh.domain_shape):
                if not pore_lookup[nx, ny, nz]:
                    surface_faces += 1
                    continue
                other = int(labels[nx, ny, nz])
                if other >= 0 and other != region:
                    interface_faces += 1
                    if nx != int(x):
                        x_interface_faces += 1
        dist_values = solid_distance[tuple(voxels.T)]
        p10_distance = float(np.percentile(dist_values, 10.0))
        mean_distance = float(np.mean(dist_values))
        _, x_counts = np.unique(voxels[:, 0], return_counts=True)
        x_section_risk = float(np.mean(x_counts) / max(float(np.min(x_counts)), 1.0)) if len(x_counts) else 0.0
        raw[region, :] = [
            surface_faces / max(6.0 * count, 1.0),
            interface_faces / max(6.0 * count, 1.0),
            len(neighbors[region]) / max(n_regions - 1, 1),
            1.0 - pore_fraction_bbox,
            1.0 / max(p10_distance, 0.5 * mesh.voxel_size),
            x_section_risk,
            x_interface_faces / max(interface_faces, 1),
        ]
        rows.append(
            {
                "region": int(region),
                "pore_voxels": count,
                "bbox_volume": bbox_volume,
                "bbox_pore_fraction": pore_fraction_bbox,
                "mean_solid_distance": mean_distance,
                "p10_solid_distance": p10_distance,
                "solid_surface_faces": int(surface_faces),
                "interface_faces": int(interface_faces),
                "x_interface_faces": int(x_interface_faces),
                "neighbor_count": int(len(neighbors[region])),
                **{name: float(raw[region, idx]) for idx, name in enumerate(feature_names)},
            }
        )

    normalized = np.column_stack([robust_normalize(raw[:, idx]) for idx in range(raw.shape[1])])
    w = np.asarray(weights, dtype=float)
    if len(w) < len(feature_names):
        w = np.pad(w, (0, len(feature_names) - len(w)), constant_values=1.0)
    w = np.maximum(w[: len(feature_names)], 0.0)
    if float(np.sum(w)) <= 0.0:
        w[:] = 1.0
    complexity = normalized @ (w / np.sum(w))
    complexity = 0.05 + 0.95 * robust_normalize(complexity)

    for row in rows:
        region = int(row["region"])
        row["geometry_complexity"] = float(complexity[region])
        for idx, name in enumerate(feature_names):
            row[f"normalized_{name}"] = float(normalized[region, idx])

    shifted = np.clip(np.floor(np.asarray(dof_data["v_coords"], dtype=float) / mesh.voxel_size).astype(int), 0, np.asarray(mesh.domain_shape) - 1)
    node_distance = solid_distance[tuple(shifted.T)]
    node_solid_risk = robust_normalize(1.0 / np.maximum(node_distance, 0.5 * mesh.voxel_size))
    node_region_risk = complexity[np.asarray(dof_data["v_region"], dtype=np.int32)]
    node_scores = 0.65 * node_solid_risk + 0.35 * robust_normalize(node_region_risk)

    return {
        "feature_names": feature_names,
        "weights": [float(v) for v in w],
        "stage_tail": [float(v) for v in stage_tail],
        "raw_features": raw,
        "normalized_features": normalized,
        "complexity": complexity,
        "node_scores": node_scores,
        "rows": rows,
        "summary": {
            "type": "geometry-only prior; no FEM solution or true-error information is used for marking",
            "feature_names": feature_names,
            "weights": [float(v) for v in w],
            "stage_tail": [float(v) for v in stage_tail],
            "complexity_min": float(np.min(complexity)),
            "complexity_max": float(np.max(complexity)),
            "complexity_mean": float(np.mean(complexity)),
        },
    }


def build_spectral_geometry_prior(
    A: sp.csr_matrix,
    dof_data: dict[str, np.ndarray],
    geometry_prior: dict[str, object],
    fixed_dofs: np.ndarray,
    max_boundary_nodes: int,
    tail_modes: int,
    spectral_weight: float,
    ridge: float,
) -> dict[str, object]:
    n_regions = len(np.asarray(geometry_prior["complexity"], dtype=float))
    geometry_complexity = np.asarray(geometry_prior["complexity"], dtype=float)
    base_node_scores = np.asarray(geometry_prior["node_scores"], dtype=float)
    fixed = np.asarray(fixed_dofs, dtype=np.int64)
    node_fixed = np.any(np.isin(np.asarray(dof_data["v_node_dofs"]), fixed), axis=1)
    spectral_raw = np.zeros((n_regions, 3), dtype=float)
    spectral_rows: list[dict[str, float | int | str]] = []

    for region in range(n_regions):
        region_nodes = np.flatnonzero((np.asarray(dof_data["v_region"], dtype=np.int32) == region) & (~node_fixed))
        interface_nodes = region_nodes[np.asarray(dof_data["interface_nodes"], dtype=bool)[region_nodes]]
        nonvertex_nodes = region_nodes[~np.asarray(dof_data["vertex_nodes"], dtype=bool)[region_nodes]]
        if len(region_nodes) == 0:
            spectral_rows.append(empty_spectral_row(region, "no_region_nodes"))
            continue
        status = "ok"
        if len(interface_nodes) < 4:
            candidate_nodes = nonvertex_nodes if len(nonvertex_nodes) else region_nodes
            status = "fallback_high_risk_region_nodes"
        else:
            candidate_nodes = interface_nodes
        if len(candidate_nodes) > max_boundary_nodes:
            order = candidate_nodes[np.argsort(base_node_scores[candidate_nodes])[::-1]]
            boundary_nodes = np.sort(order[:max_boundary_nodes])
            sampled_fraction = float(len(boundary_nodes) / len(candidate_nodes))
        else:
            boundary_nodes = candidate_nodes
            sampled_fraction = 1.0

        interior_nodes = np.setdiff1d(region_nodes, boundary_nodes, assume_unique=False)
        boundary_dofs = np.asarray(dof_data["v_node_dofs"])[boundary_nodes].ravel()
        interior_dofs = np.setdiff1d(np.asarray(dof_data["v_node_dofs"])[interior_nodes].ravel(), fixed, assume_unique=False)
        boundary_dofs = np.unique(boundary_dofs.astype(np.int64))
        interior_dofs = np.unique(interior_dofs.astype(np.int64))
        if len(boundary_dofs) < 6:
            row = empty_spectral_row(region, "too_few_boundary_dofs")
            row["spectral_boundary_nodes_total"] = int(len(candidate_nodes))
            row["spectral_boundary_nodes_sampled"] = int(len(boundary_nodes))
            row["spectral_boundary_sample_fraction"] = sampled_fraction
            row["spectral_boundary_dofs"] = int(len(boundary_dofs))
            row["spectral_interior_dofs"] = int(len(interior_dofs))
            spectral_rows.append(row)
            continue

        try:
            eigvals = local_velocity_schur_eigenvalues(A, boundary_dofs, interior_dofs, ridge)
            metrics = spectral_metrics(eigvals, tail_modes)
            spectral_raw[region, :] = [metrics["tail_ratio"], metrics["effective_rank_ratio"], metrics["condition_proxy"]]
            spectral_rows.append(
                {
                    "region": int(region),
                    "spectral_status": status,
                    "spectral_boundary_nodes_total": int(len(candidate_nodes)),
                    "spectral_boundary_nodes_sampled": int(len(boundary_nodes)),
                    "spectral_boundary_sample_fraction": sampled_fraction,
                    "spectral_boundary_dofs": int(len(boundary_dofs)),
                    "spectral_interior_dofs": int(len(interior_dofs)),
                    "spectral_positive_modes": int(metrics["positive_modes"]),
                    "steklov_tail_ratio": float(metrics["tail_ratio"]),
                    "steklov_effective_rank_ratio": float(metrics["effective_rank_ratio"]),
                    "steklov_condition_proxy": float(metrics["condition_proxy"]),
                    "steklov_lambda_min": float(metrics["lambda_min"]),
                    "steklov_lambda_max": float(metrics["lambda_max"]),
                }
            )
        except Exception as exc:
            spectral_rows.append(empty_spectral_row(region, f"failed:{type(exc).__name__}"))

    spectral_norm = np.column_stack([robust_normalize(spectral_raw[:, idx]) for idx in range(spectral_raw.shape[1])])
    spectral_complexity = 0.50 * spectral_norm[:, 0] + 0.30 * spectral_norm[:, 1] + 0.20 * spectral_norm[:, 2]
    spectral_complexity = 0.05 + 0.95 * robust_normalize(spectral_complexity)
    combined = geometry_complexity * (1.0 + max(float(spectral_weight), 0.0) * spectral_complexity)
    combined = 0.05 + 0.95 * robust_normalize(combined)

    node_region_spectral = spectral_complexity[np.asarray(dof_data["v_region"], dtype=np.int32)]
    node_region_combined = combined[np.asarray(dof_data["v_region"], dtype=np.int32)]
    node_scores = 0.45 * robust_normalize(base_node_scores) + 0.30 * robust_normalize(node_region_spectral) + 0.25 * robust_normalize(node_region_combined)

    rows_by_region = {int(row["region"]): dict(row) for row in geometry_prior["rows"]}
    spectral_by_region = {int(row["region"]): row for row in spectral_rows}
    merged_rows = []
    case_target_stages = np.zeros(n_regions, dtype=np.int32)
    for region in range(n_regions):
        row = rows_by_region.get(region, {"region": int(region)})
        row.update(spectral_by_region.get(region, empty_spectral_row(region, "missing")))
        row["spectral_complexity"] = float(spectral_complexity[region])
        row["spectral_geometry_complexity"] = float(combined[region])
        merged_rows.append(row)

    out = dict(geometry_prior)
    out["complexity"] = combined
    out["geometry_only_complexity"] = geometry_complexity
    out["spectral_complexity"] = spectral_complexity
    out["spectral_raw_features"] = spectral_raw
    out["node_scores"] = node_scores
    out["rows"] = merged_rows
    out["summary"] = {
        "type": "geometry plus local velocity Schur/Steklov spectral prior; no FEM solution or true-error information is used for marking",
        "geometry_feature_names": geometry_prior["feature_names"],
        "spectral_feature_names": ["steklov_tail_ratio", "steklov_effective_rank_ratio", "steklov_condition_proxy"],
        "weights": geometry_prior["weights"],
        "stage_tail": geometry_prior["stage_tail"],
        "spectral_weight": float(spectral_weight),
        "spectral_tail_modes": int(tail_modes),
        "spectral_boundary_max_nodes": int(max_boundary_nodes),
        "complexity_min": float(np.min(combined)),
        "complexity_max": float(np.max(combined)),
        "complexity_mean": float(np.mean(combined)),
        "spectral_complexity_min": float(np.min(spectral_complexity)),
        "spectral_complexity_max": float(np.max(spectral_complexity)),
        "spectral_complexity_mean": float(np.mean(spectral_complexity)),
    }
    return out


def build_stokes_spectral_geometry_prior(
    A: sp.csr_matrix,
    dof_data: dict[str, np.ndarray],
    geometry_prior: dict[str, object],
    fixed_dofs: np.ndarray,
    max_boundary_nodes: int,
    tail_modes: int,
    spectral_weight: float,
    ridge: float,
) -> dict[str, object]:
    n_regions = len(np.asarray(geometry_prior["complexity"], dtype=float))
    geometry_complexity = np.asarray(geometry_prior["complexity"], dtype=float)
    base_node_scores = np.asarray(geometry_prior["node_scores"], dtype=float)
    fixed = np.asarray(fixed_dofs, dtype=np.int64)
    node_fixed = np.any(np.isin(np.asarray(dof_data["v_node_dofs"]), fixed), axis=1)
    spectral_raw = np.zeros((n_regions, 4), dtype=float)
    spectral_rows: list[dict[str, float | int | str]] = []

    for region in range(n_regions):
        region_nodes = np.flatnonzero((np.asarray(dof_data["v_region"], dtype=np.int32) == region) & (~node_fixed))
        interface_nodes = region_nodes[np.asarray(dof_data["interface_nodes"], dtype=bool)[region_nodes]]
        nonvertex_nodes = region_nodes[~np.asarray(dof_data["vertex_nodes"], dtype=bool)[region_nodes]]
        pressure_dofs = np.setdiff1d(np.asarray(dof_data["map_q"])[np.asarray(dof_data["q_region"], dtype=np.int32) == region], fixed, assume_unique=False)
        if len(region_nodes) == 0:
            spectral_rows.append(empty_spectral_row(region, "no_region_nodes"))
            continue
        status = "ok"
        if len(interface_nodes) < 4:
            candidate_nodes = nonvertex_nodes if len(nonvertex_nodes) else region_nodes
            status = "fallback_high_risk_region_nodes"
        else:
            candidate_nodes = interface_nodes
        if len(candidate_nodes) > max_boundary_nodes:
            order = candidate_nodes[np.argsort(base_node_scores[candidate_nodes])[::-1]]
            boundary_nodes = np.sort(order[:max_boundary_nodes])
            sampled_fraction = float(len(boundary_nodes) / len(candidate_nodes))
        else:
            boundary_nodes = candidate_nodes
            sampled_fraction = 1.0

        interior_nodes = np.setdiff1d(region_nodes, boundary_nodes, assume_unique=False)
        boundary_dofs = np.unique(np.asarray(dof_data["v_node_dofs"])[boundary_nodes].ravel().astype(np.int64))
        interior_velocity_dofs = np.setdiff1d(np.asarray(dof_data["v_node_dofs"])[interior_nodes].ravel(), fixed, assume_unique=False).astype(np.int64)
        pressure_dofs = np.unique(pressure_dofs.astype(np.int64))
        if len(boundary_dofs) < 6:
            row = empty_spectral_row(region, "too_few_boundary_dofs")
            row["spectral_boundary_nodes_total"] = int(len(candidate_nodes))
            row["spectral_boundary_nodes_sampled"] = int(len(boundary_nodes))
            row["spectral_boundary_sample_fraction"] = sampled_fraction
            row["spectral_boundary_dofs"] = int(len(boundary_dofs))
            row["spectral_interior_dofs"] = int(len(interior_velocity_dofs))
            row["stokes_pressure_dofs"] = int(len(pressure_dofs))
            spectral_rows.append(row)
            continue

        try:
            eigvals, beta_proxy = local_stokes_schur_eigenvalues(A, boundary_dofs, interior_velocity_dofs, pressure_dofs, ridge)
            metrics = spectral_metrics(eigvals, tail_modes)
            infsup_risk = float(1.0 / max(beta_proxy, 1.0e-12))
            spectral_raw[region, :] = [metrics["tail_ratio"], metrics["effective_rank_ratio"], metrics["condition_proxy"], np.log10(max(infsup_risk, 1.0))]
            spectral_rows.append(
                {
                    "region": int(region),
                    "spectral_status": status,
                    "spectral_operator": "local_stokes_saddle_point_schur",
                    "spectral_boundary_nodes_total": int(len(candidate_nodes)),
                    "spectral_boundary_nodes_sampled": int(len(boundary_nodes)),
                    "spectral_boundary_sample_fraction": sampled_fraction,
                    "spectral_boundary_dofs": int(len(boundary_dofs)),
                    "spectral_interior_dofs": int(len(interior_velocity_dofs)),
                    "stokes_pressure_dofs": int(len(pressure_dofs)),
                    "stokes_beta_proxy": float(beta_proxy),
                    "stokes_infsup_risk": infsup_risk,
                    "spectral_positive_modes": int(metrics["positive_modes"]),
                    "steklov_tail_ratio": float(metrics["tail_ratio"]),
                    "steklov_effective_rank_ratio": float(metrics["effective_rank_ratio"]),
                    "steklov_condition_proxy": float(metrics["condition_proxy"]),
                    "steklov_lambda_min": float(metrics["lambda_min"]),
                    "steklov_lambda_max": float(metrics["lambda_max"]),
                }
            )
        except Exception as exc:
            row = empty_spectral_row(region, f"failed:{type(exc).__name__}")
            row["spectral_operator"] = "local_stokes_saddle_point_schur"
            row["stokes_pressure_dofs"] = int(len(pressure_dofs))
            spectral_rows.append(row)

    spectral_norm = np.column_stack([robust_normalize(spectral_raw[:, idx]) for idx in range(spectral_raw.shape[1])])
    spectral_complexity = 0.40 * spectral_norm[:, 0] + 0.25 * spectral_norm[:, 1] + 0.15 * spectral_norm[:, 2] + 0.20 * spectral_norm[:, 3]
    spectral_complexity = 0.05 + 0.95 * robust_normalize(spectral_complexity)
    combined = geometry_complexity * (1.0 + max(float(spectral_weight), 0.0) * spectral_complexity)
    combined = 0.05 + 0.95 * robust_normalize(combined)
    stage_tail = np.asarray(geometry_prior["stage_tail"], dtype=float)
    if len(stage_tail) < 7:
        stage_tail = np.pad(stage_tail, (0, 7 - len(stage_tail)), mode="edge")
    stage_prior_errors = combined[:, None] * stage_tail[:7][None, :]

    node_region_spectral = spectral_complexity[np.asarray(dof_data["v_region"], dtype=np.int32)]
    node_region_combined = combined[np.asarray(dof_data["v_region"], dtype=np.int32)]
    node_scores = 0.40 * robust_normalize(base_node_scores) + 0.35 * robust_normalize(node_region_spectral) + 0.25 * robust_normalize(node_region_combined)

    rows_by_region = {int(row["region"]): dict(row) for row in geometry_prior["rows"]}
    spectral_by_region = {int(row["region"]): row for row in spectral_rows}
    merged_rows = []
    case_target_stages = np.zeros(n_regions, dtype=np.int32)
    for region in range(n_regions):
        row = rows_by_region.get(region, {"region": int(region)})
        row.update(spectral_by_region.get(region, empty_spectral_row(region, "missing")))
        row["spectral_complexity"] = float(spectral_complexity[region])
        row["stokes_spectral_geometry_complexity"] = float(combined[region])
        row["spectral_geometry_complexity"] = float(combined[region])
        row["prior_error_PNM"] = float(stage_prior_errors[region, 0])
        row["prior_error_DDPNMT"] = float(stage_prior_errors[region, 2])
        row["prior_error_HODDPNM50"] = float(stage_prior_errors[region, 4])
        row["prior_error_HODDPNM100"] = float(stage_prior_errors[region, 6])
        geometry_case = classify_geometry_case(
            geometry_complexity=float(geometry_complexity[region]),
            spectral_complexity=float(spectral_complexity[region]),
            infsup_risk=float(row.get("stokes_infsup_risk", 0.0)),
        )
        row["geometry_case"] = geometry_case
        case_target_stages[region] = geometry_case_target_stage(geometry_case)
        row["case_target_stage"] = int(case_target_stages[region])
        merged_rows.append(row)

    out = dict(geometry_prior)
    out["complexity"] = combined
    out["geometry_only_complexity"] = geometry_complexity
    out["spectral_complexity"] = spectral_complexity
    out["spectral_raw_features"] = spectral_raw
    out["stage_prior_errors"] = stage_prior_errors
    out["case_target_stages"] = case_target_stages
    out["node_scores"] = node_scores
    out["rows"] = merged_rows
    out["summary"] = {
        "type": "geometry plus local Stokes saddle-point Schur/Steklov spectral prior; no FEM solution or true-error information is used for marking",
        "geometry_feature_names": geometry_prior["feature_names"],
        "spectral_feature_names": ["steklov_tail_ratio", "steklov_effective_rank_ratio", "steklov_condition_proxy", "stokes_infsup_risk"],
        "weights": geometry_prior["weights"],
        "stage_tail": geometry_prior["stage_tail"],
        "spectral_weight": float(spectral_weight),
        "spectral_tail_modes": int(tail_modes),
        "spectral_boundary_max_nodes": int(max_boundary_nodes),
        "complexity_min": float(np.min(combined)),
        "complexity_max": float(np.max(combined)),
        "complexity_mean": float(np.mean(combined)),
        "spectral_complexity_min": float(np.min(spectral_complexity)),
        "spectral_complexity_max": float(np.max(spectral_complexity)),
        "spectral_complexity_mean": float(np.mean(spectral_complexity)),
    }
    return out


def empty_spectral_row(region: int, status: str) -> dict[str, float | int | str]:
    return {
        "region": int(region),
        "spectral_status": status,
        "spectral_boundary_nodes_total": 0,
        "spectral_boundary_nodes_sampled": 0,
        "spectral_boundary_sample_fraction": 0.0,
        "spectral_boundary_dofs": 0,
        "spectral_interior_dofs": 0,
        "spectral_positive_modes": 0,
        "steklov_tail_ratio": 0.0,
        "steklov_effective_rank_ratio": 0.0,
        "steklov_condition_proxy": 0.0,
        "steklov_lambda_min": 0.0,
        "steklov_lambda_max": 0.0,
        "stokes_pressure_dofs": 0,
        "stokes_beta_proxy": 0.0,
        "stokes_infsup_risk": 0.0,
    }


def local_velocity_schur_eigenvalues(A: sp.csr_matrix, boundary_dofs: np.ndarray, interior_dofs: np.ndarray, ridge: float) -> np.ndarray:
    A_bb = A[boundary_dofs][:, boundary_dofs].tocsc()
    if len(interior_dofs) == 0:
        schur = A_bb.toarray()
    else:
        A_ii = A[interior_dofs][:, interior_dofs].tocsc()
        diag_scale = float(np.mean(np.abs(A_ii.diagonal()))) if A_ii.shape[0] else 1.0
        A_ii = A_ii + sp.eye(A_ii.shape[0], format="csc") * max(float(ridge) * max(diag_scale, 1.0), 1.0e-14)
        A_ib = A[interior_dofs][:, boundary_dofs].tocsc()
        A_bi = A[boundary_dofs][:, interior_dofs].tocsc()
        lu = splu(A_ii)
        solved = lu.solve(A_ib.toarray())
        schur = A_bb.toarray() - A_bi.toarray() @ solved
    schur = 0.5 * (schur + schur.T)
    diag = np.sqrt(np.maximum(np.abs(np.diag(schur)), 1.0e-14))
    normalized = schur / np.outer(diag, diag)
    normalized = 0.5 * (normalized + normalized.T)
    eigvals = np.linalg.eigvalsh(normalized)
    return eigvals[np.isfinite(eigvals)]


def local_stokes_schur_eigenvalues(
    A: sp.csr_matrix,
    boundary_dofs: np.ndarray,
    interior_velocity_dofs: np.ndarray,
    pressure_dofs: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, float]:
    boundary_dofs = np.asarray(boundary_dofs, dtype=np.int64)
    interior = np.unique(np.concatenate([np.asarray(interior_velocity_dofs, dtype=np.int64), np.asarray(pressure_dofs, dtype=np.int64)]))
    K_gg = A[boundary_dofs][:, boundary_dofs].tocsc()
    if len(interior) == 0:
        schur = K_gg.toarray()
        beta_proxy = 1.0
    else:
        K_ii = A[interior][:, interior].tocsc()
        diag_scale = float(np.mean(np.abs(K_ii.diagonal()))) if K_ii.shape[0] else 1.0
        regularization = max(float(ridge) * max(diag_scale, 1.0), 1.0e-14)
        K_ii = K_ii + sp.eye(K_ii.shape[0], format="csc") * regularization
        K_ig = A[interior][:, boundary_dofs].tocsc()
        K_gi = A[boundary_dofs][:, interior].tocsc()
        lu = splu(K_ii)
        solved = lu.solve(K_ig.toarray())
        schur = K_gg.toarray() - K_gi.toarray() @ solved
        beta_proxy = local_pressure_coupling_beta_proxy(A, interior_velocity_dofs, pressure_dofs)
    schur = 0.5 * (schur + schur.T)
    mass = np.full(len(boundary_dofs), 1.0, dtype=float)
    mass_inv_sqrt = 1.0 / np.sqrt(np.maximum(mass, 1.0e-14))
    generalized = schur * np.outer(mass_inv_sqrt, mass_inv_sqrt)
    generalized = 0.5 * (generalized + generalized.T)
    eigvals = np.linalg.eigvalsh(generalized)
    return eigvals[np.isfinite(eigvals)], float(beta_proxy)


def local_pressure_coupling_beta_proxy(A: sp.csr_matrix, velocity_dofs: np.ndarray, pressure_dofs: np.ndarray) -> float:
    velocity_dofs = np.asarray(velocity_dofs, dtype=np.int64)
    pressure_dofs = np.asarray(pressure_dofs, dtype=np.int64)
    if len(velocity_dofs) == 0 or len(pressure_dofs) == 0:
        return 1.0e-12
    B = A[pressure_dofs][:, velocity_dofs].toarray()
    if B.size == 0:
        return 1.0e-12
    row_norm = np.sqrt(np.sum(B * B, axis=1))
    col_norm = np.sqrt(np.sum(B * B, axis=0))
    scale = max(float(np.mean(row_norm[row_norm > 0.0])) if np.any(row_norm > 0.0) else 0.0, float(np.mean(col_norm[col_norm > 0.0])) if np.any(col_norm > 0.0) else 0.0, 1.0e-12)
    scaled = B / scale
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    positive = singular_values[singular_values > 1.0e-10]
    if len(positive) == 0:
        return 1.0e-12
    return float(np.min(positive))


def classify_geometry_case(geometry_complexity: float, spectral_complexity: float, infsup_risk: float) -> str:
    if geometry_complexity >= 0.70 and (spectral_complexity >= 0.60 or infsup_risk >= 40.0):
        return "C_bottleneck_or_stokes_unstable"
    if spectral_complexity >= 0.55 or geometry_complexity >= 0.45:
        return "B_interface_dominated"
    return "A_regular_open"


def geometry_case_target_stage(geometry_case: str) -> int:
    if geometry_case.startswith("C_"):
        return 6
    if geometry_case.startswith("B_"):
        return 4
    return 0


def spectral_metrics(eigvals: np.ndarray, tail_modes: int) -> dict[str, float | int]:
    positive = np.sort(eigvals[eigvals > 1.0e-12])[::-1]
    if len(positive) == 0:
        return {
            "positive_modes": 0,
            "tail_ratio": 0.0,
            "effective_rank_ratio": 0.0,
            "condition_proxy": 0.0,
            "lambda_min": 0.0,
            "lambda_max": 0.0,
        }
    weights = positive / max(float(np.sum(positive)), 1.0e-300)
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1.0e-300))))
    effective_rank_ratio = float(np.exp(entropy) / len(positive))
    cutoff = int(np.clip(tail_modes, 1, len(positive)))
    tail_ratio = float(np.sum(positive[cutoff:]) / max(float(np.sum(positive)), 1.0e-300))
    small = float(np.percentile(positive, 10.0))
    large = float(np.percentile(positive, 90.0))
    condition_proxy = float(np.log10(max(large / max(small, 1.0e-12), 1.0)))
    return {
        "positive_modes": int(len(positive)),
        "tail_ratio": tail_ratio,
        "effective_rank_ratio": effective_rank_ratio,
        "condition_proxy": condition_proxy,
        "lambda_min": float(np.min(positive)),
        "lambda_max": float(np.max(positive)),
    }


def robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros_like(values, dtype=float)
    lo = float(np.percentile(finite, 5.0))
    hi = float(np.percentile(finite, 95.0))
    if hi <= lo:
        hi = float(np.max(finite))
        lo = float(np.min(finite))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def region_geometry_prior_indicator(levels: np.ndarray, dof_data: dict[str, object]) -> tuple[np.ndarray, float]:
    prior = dof_data["geometry_prior"]
    complexity = np.asarray(prior["complexity"], dtype=float)
    stage_tail = np.asarray(prior["stage_tail"], dtype=float)
    if len(stage_tail) == 0:
        stage_tail = np.asarray([1.0], dtype=float)
    tail = stage_tail[np.minimum(np.asarray(levels, dtype=np.int32), len(stage_tail) - 1)]
    raw_eta = complexity * tail
    global_error = float(np.linalg.norm(raw_eta) / max(np.sqrt(len(raw_eta)), 1.0))
    eta = raw_eta / max(float(np.max(raw_eta)), 1.0e-300)
    return eta, global_error


def active_dofs_for_levels(levels: np.ndarray, dof_data: dict[str, np.ndarray], neighbors: list[set[int]], fixed_dofs: np.ndarray) -> np.ndarray:
    n_mixed = len(dof_data["mixed_region"])
    active = np.zeros(n_mixed, dtype=bool)
    active[np.asarray(fixed_dofs, dtype=np.int64)] = True
    active[dof_data["map_q"]] = True
    active[dof_data["v_node_dofs"][dof_data["vertex_nodes"]].ravel()] = True
    if dof_data.get("adaptive_strategy") == "proportional-interface":
        return active_dofs_for_interface_fractions(levels, dof_data, fixed_dofs)
    if dof_data.get("adaptive_strategy") in ("staged-proportional-hoddpnm", "a-priori-stokes-gain"):
        return active_dofs_for_staged_proportional_hoddpnm(levels, dof_data, fixed_dofs)
    for region, level in enumerate(levels):
        if level <= METHOD_LEVEL["PNM"]:
            continue
        patch = {int(region)}
        if level >= METHOD_LEVEL["DDPNMT"]:
            patch |= set(int(v) for v in neighbors[region])
        if level >= METHOD_LEVEL["HODDPNM"]:
            second = set()
            for item in patch:
                second |= set(int(v) for v in neighbors[item])
            patch |= second
        mask = np.isin(dof_data["v_region"], list(patch))
        active[dof_data["v_node_dofs"][mask].ravel()] = True
    return active


def active_dofs_for_staged_proportional_hoddpnm(levels: np.ndarray, dof_data: dict[str, np.ndarray], fixed_dofs: np.ndarray) -> np.ndarray:
    n_mixed = len(dof_data["mixed_region"])
    active = np.zeros(n_mixed, dtype=bool)
    active[np.asarray(fixed_dofs, dtype=np.int64)] = True
    active[dof_data["map_q"]] = True
    active[dof_data["v_node_dofs"][dof_data["vertex_nodes"]].ravel()] = True

    fractions = np.asarray(dof_data["interface_fractions"], dtype=float)
    hoddpnm_fractions = fractions[fractions > 0.0]
    if len(hoddpnm_fractions) == 0:
        hoddpnm_fractions = np.asarray([1.0], dtype=float)
    scores = np.asarray(dof_data.get("v_node_scores", np.zeros(len(dof_data["v_region"]))), dtype=float)
    ddpnmt_fraction = float(np.clip(dof_data.get("ddpnmt_interface_fraction", 0.10), 0.0, 1.0))

    for region, stage in enumerate(levels):
        stage = int(stage)
        region_nodes = dof_data["v_region"] == region
        if stage >= 1:
            interior = region_nodes & dof_data["p2_interior_nodes"]
            active[dof_data["v_node_dofs"][interior].ravel()] = True
        if stage >= 2:
            activate_ranked_interface_nodes(active, dof_data, scores, region, ddpnmt_fraction)
        if stage >= 3:
            fraction_index = min(stage - 3, len(hoddpnm_fractions) - 1)
            activate_ranked_interface_nodes(active, dof_data, scores, region, float(hoddpnm_fractions[fraction_index]))
    return active


def activate_ranked_interface_nodes(
    active: np.ndarray,
    dof_data: dict[str, np.ndarray],
    scores: np.ndarray,
    region: int,
    fraction: float,
) -> None:
    interface = np.flatnonzero((dof_data["v_region"] == region) & dof_data["interface_nodes"])
    if len(interface) == 0:
        return
    n_keep = int(np.ceil(float(np.clip(fraction, 0.0, 1.0)) * len(interface)))
    if n_keep <= 0:
        return
    order = interface[np.argsort(scores[interface])[::-1]]
    selected = order[:n_keep]
    active[dof_data["v_node_dofs"][selected].ravel()] = True


def active_dofs_for_interface_fractions(levels: np.ndarray, dof_data: dict[str, np.ndarray], fixed_dofs: np.ndarray) -> np.ndarray:
    n_mixed = len(dof_data["mixed_region"])
    active = np.zeros(n_mixed, dtype=bool)
    active[np.asarray(fixed_dofs, dtype=np.int64)] = True
    active[dof_data["map_q"]] = True
    active[dof_data["v_node_dofs"][dof_data["vertex_nodes"]].ravel()] = True

    fractions = np.asarray(dof_data["interface_fractions"], dtype=float)
    scores = np.asarray(dof_data.get("v_node_scores", np.zeros(len(dof_data["v_region"]))), dtype=float)
    for region, stage in enumerate(levels):
        stage = int(np.clip(stage, 0, len(fractions) - 1))
        if stage <= 0:
            continue
        region_nodes = dof_data["v_region"] == region
        if dof_data.get("proportional_node_set") == "interface-plus-interior":
            interior = region_nodes & dof_data["p2_interior_nodes"]
            active[dof_data["v_node_dofs"][interior].ravel()] = True

        interface = np.flatnonzero(region_nodes & dof_data["interface_nodes"])
        if len(interface) == 0:
            continue
        n_keep = int(np.ceil(float(np.clip(fractions[stage], 0.0, 1.0)) * len(interface)))
        if n_keep <= 0:
            continue
        order = interface[np.argsort(scores[interface])[::-1]]
        selected = order[:n_keep]
        active[dof_data["v_node_dofs"][selected].ravel()] = True
    return active


def solve_restricted_stokes(
    A: sp.csr_matrix,
    b: np.ndarray,
    active: np.ndarray,
    interface_mask: np.ndarray,
    fixed_dofs: np.ndarray,
    solver: str,
    schur_preconditioner: str,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    ilu_drop_tol: float,
    ilu_fill_factor: float,
) -> tuple[np.ndarray, dict[str, object]]:
    active_idx = np.flatnonzero(active)
    x = np.zeros_like(b, dtype=float)
    fixed_mask = np.zeros(A.shape[0], dtype=bool)
    fixed = np.unique(np.asarray(fixed_dofs, dtype=np.int64))
    fixed = fixed[(fixed >= 0) & (fixed < A.shape[0])]
    fixed_mask[fixed] = True
    known_mask = active & fixed_mask
    free_mask = active & ~fixed_mask
    known = np.flatnonzero(known_mask)
    free_idx = np.flatnonzero(free_mask)
    x_known = np.asarray(b[known], dtype=float)
    x[known] = x_known
    rhs_all = np.asarray(b, dtype=float).copy()
    if len(known):
        rhs_all -= A[:, known] @ x_known

    if solver == "direct":
        x[free_idx] = spsolve(A[free_idx][:, free_idx].tocsc(), rhs_all[free_idx])
        return x, {
            "solver": "restricted_sparse_direct_known_fixed",
            "preconditioner": "none",
            "n_boundary": 0,
            "n_interior": int(len(free_idx)),
            "n_fixed_known": int(len(known)),
            "converged": True,
            "iterations": 1,
            "gmres_info": 0,
            "relative_residual": float(np.linalg.norm(A[active_idx][:, active_idx] @ x[active_idx] - b[active_idx]) / max(np.linalg.norm(b[active_idx]), 1.0e-300)),
        }

    boundary_mask = free_mask & np.asarray(interface_mask, dtype=bool)
    interior_mask = free_mask & ~boundary_mask
    boundary = np.flatnonzero(boundary_mask)
    interior = np.flatnonzero(interior_mask)

    if len(boundary) == 0 or len(interior) == 0:
        x[free_idx] = spsolve(A[free_idx][:, free_idx].tocsc(), rhs_all[free_idx])
        return x, {
            "solver": "restricted_sparse_direct_known_fixed_fallback",
            "preconditioner": "none",
            "n_boundary": int(len(boundary)),
            "n_interior": int(len(free_idx)),
            "n_fixed_known": int(len(known)),
            "converged": True,
            "iterations": 1,
            "gmres_info": 0,
            "relative_residual": float(np.linalg.norm(A[active_idx][:, active_idx] @ x[active_idx] - b[active_idx]) / max(np.linalg.norm(b[active_idx]), 1.0e-300)),
        }

    Aii = A[interior][:, interior].tocsc()
    Aib = A[interior][:, boundary].tocsc()
    Abi = A[boundary][:, interior].tocsc()
    Abb = A[boundary][:, boundary].tocsc()
    bi = rhs_all[interior]
    bb = rhs_all[boundary]

    lu_i = splu(Aii)
    yi = lu_i.solve(bi)
    rhs = bb - Abi @ yi

    scale = base.schur_diagonal_scale(Aii, Aib, Abi, Abb)
    inv_scale = 1.0 / scale

    def schur_matvec(u_boundary: np.ndarray) -> np.ndarray:
        return Abb @ u_boundary - Abi @ lu_i.solve(Aib @ u_boundary)

    schur_operator = LinearOperator(
        Abb.shape,
        matvec=lambda u_boundary: scale * schur_matvec(scale * u_boundary),
        dtype=float,
    )
    preconditioner_operator, preconditioner_info = base.build_schur_preconditioner(
        Aii,
        Aib,
        Abi,
        Abb,
        schur_preconditioner,
        ilu_drop_tol,
        ilu_fill_factor,
    )
    if preconditioner_operator is not None:
        inner_preconditioner = preconditioner_operator
        preconditioner_operator = LinearOperator(
            Abb.shape,
            matvec=lambda u_boundary: inv_scale * (inner_preconditioner @ (inv_scale * u_boundary)),
            dtype=float,
        )
    residuals: list[float] = []
    gmres_rtol = min(float(rtol) * 1.0e-4, 1.0e-14)
    gmres_atol = min(float(atol) * 1.0e-4, 1.0e-16)
    scaled_rhs = scale * rhs
    yb, info = gmres(
        schur_operator,
        scaled_rhs,
        M=preconditioner_operator,
        rtol=gmres_rtol,
        atol=gmres_atol,
        restart=restart,
        maxiter=maxiter,
        callback=residuals.append,
        callback_type="pr_norm",
    )
    ub = scale * yb
    ui = yi - lu_i.solve(Aib @ ub)
    x[boundary] = ub
    x[interior] = ui
    schur_relative_residual = np.linalg.norm(schur_matvec(ub) - rhs) / max(np.linalg.norm(rhs), 1.0e-300)
    relative_residual = np.linalg.norm(A[active_idx][:, active_idx] @ x[active_idx] - b[active_idx]) / max(np.linalg.norm(b[active_idx]), 1.0e-300)
    residual_acceptance = max(10.0 * rtol, 1.0e-9)
    schur_converged = bool(info == 0 or schur_relative_residual <= residual_acceptance)
    if not np.isfinite(schur_relative_residual) or schur_relative_residual > residual_acceptance:
        raise RuntimeError(
            "restricted Stokes Schur GMRES did not converge: "
            f"info={info}, iterations={len(residuals)}, relative_residual={schur_relative_residual:.3e}"
        )
    return x, {
        "solver": "restricted_matrix_free_schur_gmres_known_fixed",
        "n_boundary": int(len(boundary)),
        "n_interior": int(len(interior)),
        "n_fixed_known": int(len(known)),
        "converged": schur_converged,
        "iterations": int(len(residuals)),
        "gmres_info": int(info),
        "relative_residual": float(relative_residual),
        "schur_relative_residual": float(schur_relative_residual),
        "schur_diagonal_scaling": True,
        "scaled_gmres_rtol": float(gmres_rtol),
        "scaled_gmres_atol": float(gmres_atol),
        "schur_scale_min": float(np.min(scale)),
        "schur_scale_max": float(np.max(scale)),
        **preconditioner_info,
    }


def region_residual_indicator(A: sp.csr_matrix, b: np.ndarray, solution: np.ndarray, mixed_region: np.ndarray, n_regions: int, fixed_dofs: np.ndarray) -> np.ndarray:
    residual = np.abs(A @ solution - b)
    residual[np.asarray(fixed_dofs, dtype=np.int64)] = 0.0
    eta = np.zeros(n_regions, dtype=float)
    for region in range(n_regions):
        dofs = np.flatnonzero(mixed_region == region)
        eta[region] = np.linalg.norm(residual[dofs]) / max(np.sqrt(len(dofs)), 1.0)
    return eta / max(float(eta.max()), 1.0e-300)


def region_posterior_defect_indicator(
    A: sp.csr_matrix,
    b: np.ndarray,
    solution: np.ndarray,
    active: np.ndarray,
    dof_data: dict[str, np.ndarray],
    n_regions: int,
    fixed_dofs: np.ndarray,
    weights: list[float],
) -> tuple[np.ndarray, float, list[dict[str, float | int]], np.ndarray]:
    residual = np.asarray(A @ solution - b, dtype=float)
    residual[np.asarray(fixed_dofs, dtype=np.int64)] = 0.0
    abs_residual = np.abs(residual)
    active = np.asarray(active, dtype=bool)
    w = np.asarray(weights, dtype=float)
    if len(w) < 4:
        w = np.pad(w, (0, 4 - len(w)), constant_values=1.0)
    w = np.maximum(w[:4], 0.0)
    if float(np.sum(w)) <= 0.0:
        w[:] = 1.0

    v_node_dofs = np.asarray(dof_data["v_node_dofs"], dtype=np.int64)
    velocity_dofs = np.asarray(dof_data["map_v"], dtype=np.int64)
    pressure_dofs = np.asarray(dof_data["map_q"], dtype=np.int64)
    node_residual = np.linalg.norm(abs_residual[v_node_dofs], axis=1)
    inactive_node_residual = np.linalg.norm(abs_residual[v_node_dofs] * (~active[v_node_dofs]), axis=1)
    interface_node_residual = node_residual * np.asarray(dof_data["interface_nodes"], dtype=bool)
    node_scores = 0.70 * inactive_node_residual + 0.30 * interface_node_residual

    fixed_mask = np.zeros_like(active, dtype=bool)
    fixed_mask[np.asarray(fixed_dofs, dtype=np.int64)] = True
    eta_raw = np.zeros(n_regions, dtype=float)
    component_rows: list[dict[str, float | int]] = []
    for region in range(n_regions):
        vmask = np.asarray(dof_data["v_region"], dtype=np.int32) == region
        qmask = np.asarray(dof_data["q_region"], dtype=np.int32) == region
        region_velocity_dofs = np.intersect1d(v_node_dofs[vmask].ravel(), velocity_dofs, assume_unique=False)
        region_pressure_dofs = pressure_dofs[qmask]
        region_mixed_dofs = np.concatenate([region_velocity_dofs, region_pressure_dofs])
        region_mixed_dofs = region_mixed_dofs[~fixed_mask[region_mixed_dofs]]

        inactive_velocity_dofs = region_velocity_dofs[(~active[region_velocity_dofs]) & (~fixed_mask[region_velocity_dofs])]
        interface_nodes = vmask & np.asarray(dof_data["interface_nodes"], dtype=bool)
        interface_velocity_dofs = v_node_dofs[interface_nodes].ravel()
        interface_velocity_dofs = interface_velocity_dofs[~fixed_mask[interface_velocity_dofs]]
        pressure_rows = region_pressure_dofs[~fixed_mask[region_pressure_dofs]]

        inactive_defect = norm_per_sqrt(abs_residual, inactive_velocity_dofs)
        interface_defect = norm_per_sqrt(abs_residual, interface_velocity_dofs)
        divergence_defect = norm_per_sqrt(abs_residual, pressure_rows)
        mixed_defect = norm_per_sqrt(abs_residual, region_mixed_dofs)
        eta_raw[region] = (
            w[0] * inactive_defect
            + w[1] * interface_defect
            + w[2] * divergence_defect
            + w[3] * mixed_defect
        )
        component_rows.append(
            {
                "region": int(region),
                "inactive_velocity_defect": float(inactive_defect),
                "interface_velocity_defect": float(interface_defect),
                "divergence_pressure_defect": float(divergence_defect),
                "mixed_region_defect": float(mixed_defect),
                "posterior_defect": float(eta_raw[region]),
                "inactive_velocity_dofs": int(len(inactive_velocity_dofs)),
                "interface_velocity_dofs": int(len(interface_velocity_dofs)),
                "pressure_rows": int(len(pressure_rows)),
            }
        )
    global_error = float(np.linalg.norm(eta_raw) / max(np.sqrt(n_regions), 1.0))
    eta = eta_raw / max(float(np.max(eta_raw)), 1.0e-300)
    return eta, global_error, component_rows, node_scores


def norm_per_sqrt(values: np.ndarray, dofs: np.ndarray) -> float:
    dofs = np.asarray(dofs, dtype=np.int64)
    if len(dofs) == 0:
        return 0.0
    return float(np.linalg.norm(values[dofs]) / max(np.sqrt(len(dofs)), 1.0))


def region_true_error_indicator(
    velocity_error: np.ndarray,
    pressure_error: np.ndarray,
    dof_data: dict[str, np.ndarray],
    n_regions: int,
    pressure_weight: float,
) -> np.ndarray:
    eta = np.zeros(n_regions, dtype=float)
    velocity_norm = np.linalg.norm(velocity_error, axis=1)
    for region in range(n_regions):
        vmask = dof_data["v_region"] == region
        qmask = dof_data["q_region"] == region
        v_part = np.linalg.norm(velocity_norm[vmask]) / max(np.sqrt(np.count_nonzero(vmask)), 1.0)
        p_part = np.linalg.norm(pressure_error[qmask]) / max(np.sqrt(np.count_nonzero(qmask)), 1.0)
        eta[region] = v_part + pressure_weight * p_part
    return eta / max(float(eta.max()), 1.0e-300)


def adaptive_update_levels(
    levels: np.ndarray,
    eta: np.ndarray,
    args: argparse.Namespace,
    dof_data: dict[str, np.ndarray],
    neighbors: list[set[int]],
    fixed_dofs: np.ndarray,
) -> np.ndarray:
    if args.adaptive_strategy == "cumulative":
        return adaptive_upgrade(levels, eta, args.upgrade_fraction)
    if args.adaptive_strategy == "proportional-interface":
        return proportional_interface_upgrade(
            levels,
            eta,
            dof_data,
            neighbors,
            fixed_dofs,
            active_dof_cap=args.active_dof_cap,
            dorfler_theta=args.dorfler_theta,
            max_upgrades_per_cycle=args.max_upgrades_per_cycle,
        )
    if args.adaptive_strategy == "staged-proportional-hoddpnm":
        return staged_proportional_hoddpnm_upgrade(
            levels,
            eta,
            dof_data,
            neighbors,
            fixed_dofs,
            active_dof_cap=args.active_dof_cap,
            dorfler_theta=args.dorfler_theta,
            max_upgrades_per_cycle=args.max_upgrades_per_cycle,
        )
    if args.adaptive_strategy == "a-priori-stokes-gain":
        return a_priori_stokes_gain_upgrade(
            levels,
            dof_data,
            neighbors,
            fixed_dofs,
            active_dof_cap=args.active_dof_cap,
            max_upgrades_per_cycle=args.max_upgrades_per_cycle,
        )
    if args.adaptive_strategy == "tolerance-driven":
        return tolerance_driven_upgrade(
            levels,
            eta,
            dof_data,
            neighbors,
            fixed_dofs,
            active_dof_cap=args.active_dof_cap,
            dorfler_theta=args.dorfler_theta,
            max_upgrades_per_cycle=args.max_upgrades_per_cycle,
        )
    if args.adaptive_strategy == "capped-monotone-high-error":
        return capped_monotone_high_error_levels(
            levels,
            eta,
            dof_data,
            neighbors,
            fixed_dofs,
            hoddpnm_fraction=args.hoddpnm_region_fraction,
            ddpnmt_fraction=args.ddpnmt_region_fraction,
            ddpnm_fraction=args.ddpnm_region_fraction,
            active_dof_cap=args.active_dof_cap,
        )
    return capped_high_error_levels(
        eta,
        dof_data,
        neighbors,
        fixed_dofs,
        hoddpnm_fraction=args.hoddpnm_region_fraction,
        ddpnmt_fraction=args.ddpnmt_region_fraction,
        ddpnm_fraction=args.ddpnm_region_fraction,
        active_dof_cap=args.active_dof_cap,
    )


def adaptive_upgrade(levels: np.ndarray, eta: np.ndarray, fraction: float) -> np.ndarray:
    out = levels.copy()
    candidates = np.flatnonzero(out < METHOD_LEVEL["HODDPNM"])
    if not len(candidates):
        return out
    n_mark = max(1, int(np.ceil(fraction * len(candidates))))
    order = candidates[np.argsort(eta[candidates])[::-1]]
    out[order[:n_mark]] += 1
    return out


def proportional_interface_upgrade(
    current_levels: np.ndarray,
    eta: np.ndarray,
    dof_data: dict[str, np.ndarray],
    neighbors: list[set[int]],
    fixed_dofs: np.ndarray,
    active_dof_cap: float,
    dorfler_theta: float,
    max_upgrades_per_cycle: int,
) -> np.ndarray:
    levels = current_levels.copy()
    max_stage = len(dof_data["interface_fractions"]) - 1
    candidates = np.flatnonzero(levels < max_stage)
    if not len(candidates):
        return levels

    weights = np.maximum(eta[candidates], 0.0) ** 2
    total = float(np.sum(weights))
    if total <= 0.0:
        marked = list(candidates[: max(1, max_upgrades_per_cycle)])
    else:
        order = candidates[np.argsort(weights)[::-1]]
        threshold = np.clip(dorfler_theta, 0.0, 1.0) * total
        cumulative = 0.0
        marked = []
        for region in order:
            marked.append(int(region))
            cumulative += float(eta[int(region)] ** 2)
            if cumulative >= threshold:
                break
        marked = marked[: max(1, int(max_upgrades_per_cycle))]

    base = np.zeros_like(levels)
    cap = max(
        int(np.count_nonzero(active_dofs_for_levels(base, dof_data, neighbors, fixed_dofs))),
        int(np.floor(np.clip(active_dof_cap, 0.0, 1.0) * len(dof_data["mixed_region"]))),
    )
    for region in marked:
        trial = levels.copy()
        trial[region] = min(max_stage, trial[region] + 1)
        active_count = int(np.count_nonzero(active_dofs_for_levels(trial, dof_data, neighbors, fixed_dofs)))
        if active_count <= cap:
            levels[region] = trial[region]
    return levels


def staged_proportional_hoddpnm_upgrade(
    current_levels: np.ndarray,
    eta: np.ndarray,
    dof_data: dict[str, np.ndarray],
    neighbors: list[set[int]],
    fixed_dofs: np.ndarray,
    active_dof_cap: float,
    dorfler_theta: float,
    max_upgrades_per_cycle: int,
) -> np.ndarray:
    levels = current_levels.copy()
    hoddpnm_fractions = np.asarray(dof_data["interface_fractions"], dtype=float)
    hoddpnm_fractions = hoddpnm_fractions[hoddpnm_fractions > 0.0]
    max_stage = 2 + max(1, len(hoddpnm_fractions))
    candidates = np.flatnonzero(levels < max_stage)
    if not len(candidates):
        return levels

    weights = np.maximum(eta[candidates], 0.0) ** 2
    total = float(np.sum(weights))
    if total <= 0.0:
        marked = list(candidates[: max(1, max_upgrades_per_cycle)])
    else:
        order = candidates[np.argsort(weights)[::-1]]
        threshold = np.clip(dorfler_theta, 0.0, 1.0) * total
        cumulative = 0.0
        marked = []
        for region in order:
            marked.append(int(region))
            cumulative += float(eta[int(region)] ** 2)
            if cumulative >= threshold:
                break
        marked = marked[: max(1, int(max_upgrades_per_cycle))]

    base = np.zeros_like(levels)
    cap = max(
        int(np.count_nonzero(active_dofs_for_levels(base, dof_data, neighbors, fixed_dofs))),
        int(np.floor(np.clip(active_dof_cap, 0.0, 1.0) * len(dof_data["mixed_region"]))),
    )
    for region in marked:
        trial = levels.copy()
        trial[region] = min(max_stage, trial[region] + 1)
        active_count = int(np.count_nonzero(active_dofs_for_levels(trial, dof_data, neighbors, fixed_dofs)))
        if active_count <= cap:
            levels[region] = trial[region]
    return levels


def a_priori_stokes_gain_upgrade(
    current_levels: np.ndarray,
    dof_data: dict[str, np.ndarray],
    neighbors: list[set[int]],
    fixed_dofs: np.ndarray,
    active_dof_cap: float,
    max_upgrades_per_cycle: int,
) -> np.ndarray:
    levels = current_levels.copy()
    hoddpnm_fractions = np.asarray(dof_data["interface_fractions"], dtype=float)
    hoddpnm_fractions = hoddpnm_fractions[hoddpnm_fractions > 0.0]
    max_stage = 2 + max(1, len(hoddpnm_fractions))
    candidates = np.flatnonzero(levels < max_stage)
    if not len(candidates):
        return levels

    prior = dof_data.get("geometry_prior", {})
    prior_table = np.asarray(prior.get("stage_prior_errors", np.empty((0, 0))), dtype=float)
    if prior_table.shape[0] != len(levels) or prior_table.shape[1] <= max_stage:
        complexity = np.asarray(prior.get("complexity", np.ones(len(levels))), dtype=float)
        stage_tail = np.asarray(prior.get("stage_tail", np.ones(max_stage + 1)), dtype=float)
        stage_tail = np.pad(stage_tail, (0, max(0, max_stage + 1 - len(stage_tail))), mode="edge")
        prior_table = complexity[:, None] * stage_tail[: max_stage + 1][None, :]
    case_target_stages = np.asarray(prior.get("case_target_stages", np.zeros(len(levels), dtype=np.int32)), dtype=np.int32)
    if len(case_target_stages) != len(levels):
        case_target_stages = np.zeros(len(levels), dtype=np.int32)

    current_active_count = int(np.count_nonzero(active_dofs_for_levels(levels, dof_data, neighbors, fixed_dofs)))
    base = np.zeros_like(levels)
    cap = max(
        int(np.count_nonzero(active_dofs_for_levels(base, dof_data, neighbors, fixed_dofs))),
        int(np.floor(np.clip(active_dof_cap, 0.0, 1.0) * len(dof_data["mixed_region"]))),
    )
    scored: list[tuple[float, int, int]] = []
    for region in candidates:
        region = int(region)
        next_stage = min(max_stage, int(levels[region]) + 1)
        risk_drop = float(max(prior_table[region, int(levels[region])] - prior_table[region, next_stage], 0.0))
        if int(levels[region]) < int(case_target_stages[region]):
            risk_drop *= 5.0
        trial = levels.copy()
        trial[region] = next_stage
        trial_active_count = int(np.count_nonzero(active_dofs_for_levels(trial, dof_data, neighbors, fixed_dofs)))
        added = max(trial_active_count - current_active_count, 1)
        scored.append((risk_drop / added, region, trial_active_count))

    scored.sort(reverse=True)
    for _, region, _ in scored[: max(1, int(max_upgrades_per_cycle))]:
        trial = levels.copy()
        trial[int(region)] = min(max_stage, trial[int(region)] + 1)
        active_count = int(np.count_nonzero(active_dofs_for_levels(trial, dof_data, neighbors, fixed_dofs)))
        if active_count <= cap:
            levels[int(region)] = trial[int(region)]
    return levels


def tolerance_driven_upgrade(
    current_levels: np.ndarray,
    eta: np.ndarray,
    dof_data: dict[str, np.ndarray],
    neighbors: list[set[int]],
    fixed_dofs: np.ndarray,
    active_dof_cap: float,
    dorfler_theta: float,
    max_upgrades_per_cycle: int,
) -> np.ndarray:
    levels = current_levels.copy()
    candidates = np.flatnonzero(levels < METHOD_LEVEL["HODDPNM"])
    if not len(candidates):
        return levels

    weights = np.maximum(eta[candidates], 0.0) ** 2
    total = float(np.sum(weights))
    if total <= 0.0:
        marked = list(candidates[: max(1, max_upgrades_per_cycle)])
    else:
        order = candidates[np.argsort(weights)[::-1]]
        threshold = np.clip(dorfler_theta, 0.0, 1.0) * total
        cumulative = 0.0
        marked = []
        for region in order:
            marked.append(int(region))
            cumulative += float(eta[int(region)] ** 2)
            if cumulative >= threshold:
                break
        marked = marked[: max(1, int(max_upgrades_per_cycle))]

    base = np.zeros_like(levels)
    cap = max(
        int(np.count_nonzero(active_dofs_for_levels(base, dof_data, neighbors, fixed_dofs))),
        int(np.floor(np.clip(active_dof_cap, 0.0, 1.0) * len(dof_data["mixed_region"]))),
    )
    for region in marked:
        trial = levels.copy()
        trial[region] += 1
        active_count = int(np.count_nonzero(active_dofs_for_levels(trial, dof_data, neighbors, fixed_dofs)))
        if active_count <= cap:
            levels[region] = trial[region]
    return levels


def capped_high_error_levels(
    eta: np.ndarray,
    dof_data: dict[str, np.ndarray],
    neighbors: list[set[int]],
    fixed_dofs: np.ndarray,
    hoddpnm_fraction: float,
    ddpnmt_fraction: float,
    ddpnm_fraction: float,
    active_dof_cap: float,
) -> np.ndarray:
    n_regions = len(eta)
    levels = np.zeros(n_regions, dtype=np.int32)
    order = list(np.argsort(eta)[::-1])
    cap = max(
        int(np.count_nonzero(active_dofs_for_levels(levels, dof_data, neighbors, fixed_dofs))),
        int(np.floor(np.clip(active_dof_cap, 0.0, 1.0) * len(dof_data["mixed_region"]))),
    )

    hodd_target = max(1, int(np.ceil(np.clip(hoddpnm_fraction, 0.0, 1.0) * n_regions)))
    ddpnmt_target = int(np.ceil(np.clip(ddpnmt_fraction, 0.0, 1.0) * n_regions))
    ddpnm_target = int(np.ceil(np.clip(ddpnm_fraction, 0.0, 1.0) * n_regions))

    assigned: set[int] = set()
    assigned |= greedy_assign_level(levels, order, assigned, METHOD_LEVEL["HODDPNM"], hodd_target, cap, dof_data, neighbors, fixed_dofs)
    assigned |= greedy_assign_level(levels, order, assigned, METHOD_LEVEL["DDPNMT"], ddpnmt_target, cap, dof_data, neighbors, fixed_dofs)
    assigned |= greedy_assign_level(levels, order, assigned, METHOD_LEVEL["DDPNM"], ddpnm_target, cap, dof_data, neighbors, fixed_dofs)
    return levels


def capped_monotone_high_error_levels(
    current_levels: np.ndarray,
    eta: np.ndarray,
    dof_data: dict[str, np.ndarray],
    neighbors: list[set[int]],
    fixed_dofs: np.ndarray,
    hoddpnm_fraction: float,
    ddpnmt_fraction: float,
    ddpnm_fraction: float,
    active_dof_cap: float,
) -> np.ndarray:
    n_regions = len(eta)
    levels = current_levels.copy()
    order = list(np.argsort(eta)[::-1])
    base = np.zeros(n_regions, dtype=np.int32)
    cap = max(
        int(np.count_nonzero(active_dofs_for_levels(base, dof_data, neighbors, fixed_dofs))),
        int(np.floor(np.clip(active_dof_cap, 0.0, 1.0) * len(dof_data["mixed_region"]))),
    )

    hodd_target = max(1, int(np.ceil(np.clip(hoddpnm_fraction, 0.0, 1.0) * n_regions)))
    ddpnmt_target = int(np.ceil(np.clip(ddpnmt_fraction, 0.0, 1.0) * n_regions))
    ddpnm_target = int(np.ceil(np.clip(ddpnm_fraction, 0.0, 1.0) * n_regions))

    promote_existing_or_new(levels, order, METHOD_LEVEL["HODDPNM"], hodd_target, cap, dof_data, neighbors, fixed_dofs)
    promote_existing_or_new(levels, order, METHOD_LEVEL["DDPNMT"], ddpnmt_target, cap, dof_data, neighbors, fixed_dofs)
    promote_existing_or_new(levels, order, METHOD_LEVEL["DDPNM"], ddpnm_target, cap, dof_data, neighbors, fixed_dofs)
    return levels


def promote_existing_or_new(
    levels: np.ndarray,
    order: list[int],
    level: int,
    target: int,
    cap: int,
    dof_data: dict[str, np.ndarray],
    neighbors: list[set[int]],
    fixed_dofs: np.ndarray,
) -> None:
    if target <= 0:
        return
    for region in order:
        if np.count_nonzero(levels == level) >= target:
            return
        region = int(region)
        if levels[region] >= level:
            continue
        trial = levels.copy()
        trial[region] = level
        active_count = int(np.count_nonzero(active_dofs_for_levels(trial, dof_data, neighbors, fixed_dofs)))
        if active_count <= cap:
            levels[region] = level


def greedy_assign_level(
    levels: np.ndarray,
    order: list[int],
    assigned: set[int],
    level: int,
    target: int,
    cap: int,
    dof_data: dict[str, np.ndarray],
    neighbors: list[set[int]],
    fixed_dofs: np.ndarray,
) -> set[int]:
    added: set[int] = set()
    for region in order:
        if len(added) >= target:
            break
        if int(region) in assigned:
            continue
        trial = levels.copy()
        trial[int(region)] = level
        active_count = int(np.count_nonzero(active_dofs_for_levels(trial, dof_data, neighbors, fixed_dofs)))
        if active_count <= cap:
            levels[int(region)] = level
            added.add(int(region))
    return added


def vector_rel_l2(diff: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1.0e-300))


def scalar_rel_l2(diff: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1.0e-300))


def stopping_error(velocity_error: float, pressure_error: float, residual_error: float, metric: str) -> float:
    if metric == "estimated-residual":
        return float(residual_error)
    if metric == "velocity":
        return float(velocity_error)
    if metric == "pressure":
        return float(pressure_error)
    if metric == "combined":
        return float(np.sqrt(velocity_error * velocity_error + pressure_error * pressure_error))
    return float(max(velocity_error, pressure_error))


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise SystemExit("--interface-fractions must contain at least one value")
    if values[0] != 0.0:
        values.insert(0, 0.0)
    return [float(np.clip(v, 0.0, 1.0)) for v in values]


def parse_float_sequence(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise SystemExit("expected at least one comma-separated float")
    return values


def proportional_fraction_stats(levels: np.ndarray, dof_data: dict[str, np.ndarray]) -> dict[str, float | int]:
    fractions = np.asarray(dof_data["interface_fractions"], dtype=float)
    stage_fraction = fractions[np.clip(levels, 0, len(fractions) - 1)]
    interface_total = int(np.count_nonzero(dof_data["interface_nodes"]))
    released = 0
    for region, stage in enumerate(levels):
        interface = np.flatnonzero((dof_data["v_region"] == region) & dof_data["interface_nodes"])
        if len(interface) == 0:
            continue
        frac = float(stage_fraction[region])
        released += int(np.ceil(frac * len(interface)))
    stats: dict[str, float | int] = {
        "mean_interface_fraction": float(np.mean(stage_fraction)) if len(stage_fraction) else 0.0,
        "released_interface_nodes": int(released),
        "total_interface_nodes": int(interface_total),
        "released_interface_node_ratio": float(released / max(interface_total, 1)),
    }
    for index, fraction in enumerate(fractions):
        stats[f"n_fraction_stage_{index}"] = int(np.count_nonzero(levels == index))
        stats[f"fraction_stage_{index}"] = float(fraction)
    return stats


def staged_proportional_fraction_stats(levels: np.ndarray, dof_data: dict[str, np.ndarray]) -> dict[str, float | int]:
    hoddpnm_fractions = np.asarray(dof_data["interface_fractions"], dtype=float)
    hoddpnm_fractions = hoddpnm_fractions[hoddpnm_fractions > 0.0]
    if len(hoddpnm_fractions) == 0:
        hoddpnm_fractions = np.asarray([1.0], dtype=float)
    ddpnmt_fraction = float(np.clip(dof_data.get("ddpnmt_interface_fraction", 0.10), 0.0, 1.0))
    interface_total = int(np.count_nonzero(dof_data["interface_nodes"]))
    released = 0
    stage_fraction_sum = 0.0
    for region, stage in enumerate(levels):
        interface = np.flatnonzero((dof_data["v_region"] == region) & dof_data["interface_nodes"])
        if int(stage) < 2 or len(interface) == 0:
            fraction = 0.0
        elif int(stage) == 2:
            fraction = ddpnmt_fraction
        else:
            fraction = float(hoddpnm_fractions[min(int(stage) - 3, len(hoddpnm_fractions) - 1)])
        stage_fraction_sum += fraction
        released += int(np.ceil(fraction * len(interface)))
    stats: dict[str, float | int] = {
        "mean_interface_fraction": float(stage_fraction_sum / max(len(levels), 1)),
        "released_interface_nodes": int(released),
        "total_interface_nodes": int(interface_total),
        "released_interface_node_ratio": float(released / max(interface_total, 1)),
        "n_stage_pnm": int(np.count_nonzero(levels == 0)),
        "n_stage_ddpnm": int(np.count_nonzero(levels == 1)),
        "n_stage_ddpnmt": int(np.count_nonzero(levels == 2)),
    }
    for index, fraction in enumerate(hoddpnm_fractions, start=3):
        stats[f"n_stage_hoddpnm_{int(round(100.0 * fraction))}"] = int(np.count_nonzero(levels == index))
        stats[f"fraction_stage_{index}"] = float(fraction)
    return stats


def render_method_map(mesh, levels: np.ndarray, out: Path) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1450, 1050))
    plotter.set_background("white")
    voxel_labels = mesh.voxel_labels[tuple(mesh.pore_voxels.T)]
    centers = (mesh.pore_voxels.astype(float) + 0.5) * mesh.voxel_size
    if int(np.max(levels)) > METHOD_LEVEL["HODDPNM"]:
        labels = {
            0: ("PNM", "#5c7cfa"),
            1: ("DDPNM", "#37b24d"),
            2: ("DDPNMT", "#f59f00"),
            3: ("H25", "#ff922b"),
            4: ("H50", "#e03131"),
            5: ("H75", "#9c36b5"),
            6: ("H100", "#364fc7"),
        }
    else:
        labels = {METHOD_LEVEL[method]: (method, METHOD_COLOR[method]) for method in METHODS}
    for level, (label, color) in labels.items():
        region_ids = np.flatnonzero(levels == level)
        if not len(region_ids):
            continue
        mask = np.isin(voxel_labels, region_ids)
        cloud = pv.PolyData(centers[mask])
        geom = pv.Cube(x_length=0.82, y_length=0.82, z_length=0.82)
        plotter.add_mesh(cloud.glyph(scale=False, orient=False, geom=geom), color=color, opacity=0.72, label=label)
    add_domain_box(plotter, np.asarray(mesh.domain_shape, dtype=float) * mesh.voxel_size)
    plotter.add_legend(size=(0.18, 0.18), bcolor="white", face=None)
    set_camera(plotter, np.asarray(mesh.domain_shape, dtype=float) * mesh.voxel_size)
    plotter.screenshot(str(out))
    plotter.close()


def render_progression(mesh, out_dir: Path, rows: list[dict[str, float | int]], out: Path) -> None:
    cycles = [0, 2, 4, 6, int(rows[-1]["cycle"])]
    cycles = sorted(set(c for c in cycles if (out_dir / f"cycle_{c:02d}.npz").exists()))
    panel_paths = []
    for cycle in cycles:
        levels = np.load(out_dir / f"cycle_{cycle:02d}.npz")["levels"]
        panel = out_dir / f"cycle_{cycle:02d}_method_map.png"
        render_method_map(mesh, levels, panel)
        panel_paths.append(panel)
    cols = 3
    nrows = int(np.ceil(len(panel_paths) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(14.0, 4.25 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for ax, path, cycle in zip(axes, panel_paths, cycles):
        ax.imshow(plt.imread(path))
        ax.set_title(f"cycle {cycle}", fontsize=13)
        ax.axis("off")
    for ax in axes[len(panel_paths) :]:
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94, wspace=0.01, hspace=0.08)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_history(rows: list[dict[str, float | int]], out: Path) -> None:
    cycles = [int(row["cycle"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    axes[0].semilogy(cycles, [row["velocity_rel_l2_error"] for row in rows], marker="o", label="velocity")
    axes[0].semilogy(cycles, [row["pressure_rel_l2_error"] for row in rows], marker="s", label="pressure")
    axes[0].set_xlabel("adaptive cycle")
    axes[0].set_ylabel("relative L2 error")
    axes[0].legend(frameon=False)
    axes[1].plot(cycles, [row["pressure_time_seconds"] for row in rows], marker="o")
    axes[1].set_xlabel("adaptive cycle")
    axes[1].set_ylabel("restricted Stokes solve time (s)")
    for method in METHODS:
        axes[2].plot(cycles, [row[f"n_{method}"] for row in rows], marker="o", label=method)
    axes[2].set_xlabel("adaptive cycle")
    axes[2].set_ylabel("regions")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def add_domain_box(plotter: pv.Plotter, shape: np.ndarray) -> None:
    sx, sy, sz = shape
    cube = pv.Cube(bounds=(0, sx, 0, sy, 0, sz)).extract_surface()
    plotter.add_mesh(cube, color="#efefef", opacity=0.055, show_edges=False)
    edges = cube.extract_feature_edges(boundary_edges=True, feature_edges=True, manifold_edges=False, non_manifold_edges=False)
    plotter.add_mesh(edges, color="#8d8d8d", opacity=0.65, line_width=2.0)


def set_camera(plotter: pv.Plotter, shape: np.ndarray) -> None:
    center = tuple(0.5 * shape)
    scale = float(max(shape))
    plotter.camera_position = [(1.55 * scale, -1.62 * scale, 1.25 * scale), center, (0.0, 0.0, 1.0)]
    plotter.enable_parallel_projection()
    plotter.camera.zoom(0.86)


def parse_crop(text: str) -> tuple[slice, slice, slice]:
    spans = []
    for part in text.split(","):
        start, stop = part.split(":")
        spans.append(slice(int(start), int(stop)))
    if len(spans) != 3:
        raise SystemExit("--crop must be x0:x1,y0:y1,z0:z1")
    return tuple(spans)


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, object]) -> None:
    final = summary["final"]
    reported_velocity_error = final.get("validation_velocity_rel_l2_error", final["velocity_rel_l2_error"])
    reported_pressure_error = final.get("validation_pressure_rel_l2_error", final["pressure_rel_l2_error"])
    fem_memory = summary["memory_trace_mib"].get("after_fem_reference")
    adaptive_memory = summary["memory_trace_mib"]["end"]
    fem_time = summary["timings_seconds"]["fem_sparse_direct_reference"]
    fem_time_text = f"`{fem_time:.6f} s`" if fem_time is not None else "`not computed`"
    fem_memory_text = (
        f"`{fem_memory['working_set_mib']:.3f} MiB` / `{fem_memory['peak_working_set_mib']:.3f} MiB`"
        if fem_memory is not None
        else "`not computed`"
    )
    lines = [
        "# Adaptive FEniCSx Taylor-Hood Stokes Report",
        "",
        "This run uses only the FEniCSx-assembled Taylor-Hood P2-P1 Stokes matrix.",
        "Adaptive levels choose restricted Stokes subspaces; no graph pressure equation is used.",
        "",
        f"- Crop: `{summary['crop']}`",
        f"- Pore voxels: `{summary['pore_voxels']}`",
        f"- Regions: `{summary['regions']}`",
        f"- Mixed dofs: `{summary['mixed_dofs']}`",
        f"- FEM direct reference time: {fem_time_text}",
        f"- FEM direct working / peak working memory: {fem_memory_text}",
        f"- Final adaptive restricted Stokes time: `{summary['timings_seconds']['final_adaptive_restricted_stokes']:.6f} s`",
        f"- Final adaptive working / peak working memory: `{adaptive_memory['working_set_mib']:.3f} MiB` / `{adaptive_memory['peak_working_set_mib']:.3f} MiB`",
        f"- Adaptive strategy: `{summary['adaptive_policy']['strategy']}`",
        f"- Error metric / tolerance: `{summary['adaptive_policy']['error_metric']}` / `{summary['adaptive_policy']['error_tolerance']:.6e}`",
        f"- Doerfler theta / max upgrades per cycle: `{summary['adaptive_policy']['dorfler_theta']:.3f}` / `{summary['adaptive_policy']['max_upgrades_per_cycle']}`",
        f"- Active DOF cap: `{summary['adaptive_policy']['active_dof_cap']:.3f}`",
        f"- Final active DOF ratio, including known fixed Dirichlet DOFs: `{final['active_dof_ratio']:.6f}`",
        f"- Final active free DOF ratio, after known-fixed elimination: `{final['active_free_dof_ratio']:.6f}`",
        f"- Known fixed DOFs removed from the restricted solve: `{final['known_fixed_dofs']}`",
        f"- Final target error: `{final['target_error']:.6e}`",
        f"- Converged to tolerance: `{final['converged_to_tolerance']}`",
        f"- Final velocity rel L2 error: `{reported_velocity_error:.6e}`",
        f"- Final pressure rel L2 error: `{reported_pressure_error:.6e}`",
        f"- Final method counts: PNM={final['n_PNM']}, DDPNM={final['n_DDPNM']}, DDPNMT={final['n_DDPNMT']}, HODDPNM={final['n_HODDPNM']}",
        f"- Schur iterations / operator residual: `{final['schur_iterations']}` / `{final['schur_operator_relative_residual']:.6e}`",
        "- Pressure DOF policy: all pressure DOFs are kept active, so this run demonstrates the restricted Stokes adaptive framework rather than strong compression efficiency.",
        f"- Iteration artifacts saved: `{not summary.get('history_omitted_from_outputs', False)}`",
    ]
    if "mean_interface_fraction" in final:
        lines.extend(
            [
                f"- Mean interface fraction: `{final['mean_interface_fraction']:.6f}`",
                f"- Released interface nodes: `{final['released_interface_nodes']}` / `{final['total_interface_nodes']}`",
                f"- Released interface node ratio: `{final['released_interface_node_ratio']:.6f}`",
            ]
        )
    final_figures = summary.get("outputs", {}).get("final_figures", {})
    if final_figures:
        lines.extend(["", "## Final-State Figures", ""])
        for label, rel_path in final_figures.items():
            if rel_path:
                lines.append(f"- `{label}`: `{Path(rel_path).name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

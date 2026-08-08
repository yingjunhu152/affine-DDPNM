#!/usr/bin/env python3
"""Same-mesh benchmark on the random-sphere 3-D porous medium.

Compares, on one identical tetrahedral mesh of the frozen random packing:

1. Classic-DDPNM: one constant-normal traction coefficient per interface;
2. Affine-DDPNM:   one face entity carrying the nine generalized modes
                   {1,s,t} x {n,t1,t2};
3. the exact finite-element trace Schur complement (correctness baseline:
                   must agree with the monolithic solve to roundoff);
4. the monolithic Taylor-Hood P2-P1 FEM reference.

Strict same-mesh errors are reported against both the monolithic FEM
solution and the exact FE-trace Schur solution (2-D-style format).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent
BASE_PROJECT_DIR = REPOSITORY_DIR / "ddpnm_3d_uniform_spheres"
if not BASE_PROJECT_DIR.exists():
    raise FileNotFoundError(f"Expected the base project at {BASE_PROJECT_DIR}.")
for import_root in (REPOSITORY_DIR, BASE_PROJECT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ddpnm_core.assembler import InterfaceAssembler
from ddpnm_core.fem_utils import solve_reference
from ddpnm_core.io import topology_arrays
from ddpnm_core.library import build_response_library
from ddpnm_core.trace_schur import solve_exact_fe_schur
from ddpnm_core.validation import finite_element_error_analysis
from ddpnm3d.solver import DdpnmSolution, LocalResponse, build_modes
from ddpnm3d.visualization import evaluate_fem_ddpnm_slice

from random_porous import SPHERES, build_partition
from affine_face_basis import (
    AffineFaceBasis,
    CompatibleClassicP0Basis,
    NormalLinearFaceBasis,
)
from watershed_partition import build_partition_watershed

METHODS = (
    "Classic-DDPNM-1",
    "NormalLinear-DDPNM-3",
    "Affine-DDPNM-9",
    "Exact-FE-Schur",
    "Monolithic-FEM",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition",
        choices=("voronoi", "watershed"),
        default="voronoi",
        help="partition drawing: Voronoi saddle planes (baseline) or "
        "clearance watershed basins (hand-off 5.3/5.4)",
    )
    parser.add_argument(
        "--with-exact-schur",
        action="store_true",
        help="run the exact dense FE-trace Schur even for the watershed "
        "partition (it is the default for Voronoi; for watershed it costs "
        "the dense solve and is only needed as a correctness re-check)",
    )
    parser.add_argument("--policy", default="walls_and_spheres")
    parser.add_argument("--abs-threshold", type=float, default=0.02)
    parser.add_argument("--rel-threshold", type=float, default=0.05)
    parser.add_argument("--mesh-size", type=float, default=0.13)
    parser.add_argument("--sphere-size", type=float, default=0.065)
    parser.add_argument("--boundary-size", type=float, default=0.085)
    parser.add_argument("--interface-size", type=float, default=0.075)
    parser.add_argument("--sphere-band", type=float, default=0.15)
    parser.add_argument("--boundary-band", type=float, default=0.13)
    parser.add_argument("--interface-band", type=float, default=0.12)
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--inlet-pressure", type=float, default=1.0)
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--pressure-stabilization", type=float, default=0.0)
    parser.add_argument("--reference-iterative-threshold", type=int, default=100_000)
    parser.add_argument("--reference-rtol", type=float, default=1.0e-9)
    parser.add_argument("--reference-restart", type=int, default=60)
    parser.add_argument("--reference-maxiter", type=int, default=150)
    parser.add_argument("--reference-ilu-drop-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--reference-ilu-fill-factor", type=float, default=6.0)
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_DIR / "outputs" / "benchmark"
    )
    return parser.parse_args()


def tetrahedron_volumes(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    xyz = points[tetrahedra]
    matrices = np.stack(
        (xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0]),
        axis=1,
    )
    return np.abs(np.linalg.det(matrices)) / 6.0


class _ReferenceLike:
    """Minimal reference object for error analysis of a Schur solution.

    The exact FE-trace Schur solution equals the monolithic solve to
    roundoff (1e-12), so its boundary fluxes are those of the FEM reference.
    """

    def __init__(self, W, solution, boundary_fluxes):
        self.W = W
        self.solution = solution
        self.boundary_fluxes = boundary_fluxes


def reduced_solution(partition, library, system) -> DdpnmSolution:
    """Convert the core system to the 3-D DdpnmSolution API (same as the
    affine benchmark: one face entity per interface, keys per global dof)."""
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


def error_metrics(partition, solution, reference, volumes) -> dict:
    metric, _velocity_cell, _pressure_cell = finite_element_error_analysis(
        partition, solution, reference, volumes
    )
    return metric


def _metric_entry(metric: dict) -> dict:
    return {
        "velocity_relative_l2": metric["velocity_relative_l2"],
        "velocity_relative_broken_h1": metric[
            "velocity_relative_broken_h1_seminorm"
        ],
        "pressure_relative_l2": metric["pressure_raw_relative_l2"],
        "pressure_aligned_relative_l2": metric[
            "pressure_mean_aligned_relative_l2"
        ],
        "outlet_flux_relative_error": metric["outlet_flux_relative_error"],
    }


def _algebraic_diagnostics(solution: DdpnmSolution) -> dict:
    """Conservation and algebra diagnostics of a reduced-method solution."""
    schur = np.asarray(solution.schur_matrix)
    scale = max(float(np.linalg.norm(schur)), 1.0e-30)
    return {
        "schur_symmetry_error": float(
            np.linalg.norm(schur - schur.T) / scale
        ),
        "min_schur_eigenvalue": float(solution.min_schur_eigenvalue),
        "max_mass_residual": float(solution.max_mass_residual),
        "boundary_fluxes": {
            str(key): float(value) for key, value in solution.boundary_fluxes.items()
        },
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, dict[str, float]] = {}
    method_data: dict[str, dict] = {}

    print("[1/6] Building the random-sphere partition mesh ...")
    started = time.perf_counter()
    if args.partition == "watershed":
        partition = build_partition_watershed(
            bulk_size=args.mesh_size,
            sphere_size=args.sphere_size,
            boundary_size=args.boundary_size,
            sphere_band=args.sphere_band,
            boundary_band=args.boundary_band,
            mesh_file=args.out_dir / "watershed_partition.msh",
            policy=args.policy,
            abs_threshold=args.abs_threshold,
            rel_threshold=args.rel_threshold,
        )
    else:
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
    mesh_seconds = time.perf_counter() - started
    points, tetrahedra = topology_arrays(partition.mesh)
    volumes = tetrahedron_volumes(points, tetrahedra)
    n_interfaces = len(partition.interface_pairs)
    print(
        f"      tetrahedra={len(tetrahedra)}, subdomains="
        f"{len(np.unique(partition.cell_labels))}, interfaces={n_interfaces}"
    )

    print("[2/6] Solving the monolithic Taylor--Hood FEM reference ...")
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
    fem_seconds = time.perf_counter() - started
    timings["Monolithic-FEM"] = {
        "offline_seconds": 0.0,
        "online_seconds": fem_seconds,
        "first_solve_seconds": fem_seconds,
    }
    method_data["Monolithic-FEM"] = {
        "global_unknowns": int(reference.ndofs),
        "modes_per_interface": "monolithic",
        "primitive_rhs_columns": 0,
        **_metric_entry(
            {
                "velocity_relative_l2": 0.0,
                "velocity_relative_broken_h1_seminorm": 0.0,
                "pressure_raw_relative_l2": 0.0,
                "pressure_mean_aligned_relative_l2": 0.0,
                "outlet_flux_relative_error": 0.0,
            }
        ),
    }
    print(f"      dofs={reference.ndofs}, solve={fem_seconds:.3f} s")

    print("[3/6] Exact FE-trace Schur complement (correctness baseline) ...")
    if args.partition == "watershed" and not args.with_exact_schur:
        # The dense exact-Schur implementation costs ~9 minutes at 15k
        # tetrahedra and OOMs near 20k; its correctness role was established
        # on the Voronoi partition (monolithic difference 1e-12).  The
        # watershed run keeps the monolithic FEM as the reference and skips
        # the dense Schur unless explicitly requested.
        print("      skipped for the watershed partition (--with-exact-schur to run)")
        schur_ref = None
    else:
        started = time.perf_counter()
        schur = solve_exact_fe_schur(
            partition,
            viscosity=args.viscosity,
            inlet_pressure=args.inlet_pressure,
            outlet_pressure=args.outlet_pressure,
            pressure_stabilization=1.0e-10,
        )
        schur_seconds = time.perf_counter() - started
        timings["Exact-FE-Schur"] = {
            "offline_seconds": 0.0,
            "online_seconds": schur_seconds,
            "first_solve_seconds": schur_seconds,
        }
        method_data["Exact-FE-Schur"] = {
            "global_unknowns": int(len(schur.interface_dofs)),
            "modes_per_interface": "full FE trace",
            "primitive_rhs_columns": 0,
            **_metric_entry(
                {
                    "velocity_relative_l2": 0.0,
                    "velocity_relative_broken_h1_seminorm": 0.0,
                    "pressure_raw_relative_l2": 0.0,
                    "pressure_mean_aligned_relative_l2": 0.0,
                    "outlet_flux_relative_error": 0.0,
                }
            ),
        }
        print(
            f"      interface trace dofs={len(schur.interface_dofs)}, "
            f"solve={schur_seconds:.3f} s, vs monolithic="
            f"{schur.monolithic_relative_difference:.2e}"
        )
        schur_ref = schur

    print("[4/6] Classic DDPNM: one P0 normal coefficient per face ...")
    started = time.perf_counter()
    classic_basis = CompatibleClassicP0Basis()
    classic_library = build_response_library(
        partition,
        classic_basis,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
    )
    classic_offline = time.perf_counter() - started
    classic_primitive_count = sum(
        len(entry.primitive_modes) for entry in classic_library.entries
    )
    started = time.perf_counter()
    classic_system = InterfaceAssembler(classic_library).assemble(
        np.zeros(n_interfaces, dtype=np.int8)
    )
    classic_value = reduced_solution(partition, classic_library, classic_system)
    classic_online = time.perf_counter() - started
    classic_metric = error_metrics(partition, classic_value, reference, volumes)
    timings["Classic-DDPNM-1"] = {
        "offline_seconds": classic_offline,
        "online_seconds": classic_online,
        "first_solve_seconds": classic_offline + classic_online,
    }
    method_data["Classic-DDPNM-1"] = {
        "global_unknowns": len(classic_system.global_keys),
        "modes_per_interface": 1,
        "primitive_rhs_columns": classic_primitive_count,
        **_metric_entry(classic_metric),
        **_algebraic_diagnostics(classic_value),
    }
    print(
        f"      dofs={len(classic_system.global_keys)}, "
        f"offline={classic_offline:.3f} s, online={classic_online:.3f} s, "
        f"L2(u)={classic_metric['velocity_relative_l2']:.3%}"
    )

    print("[5/6] Normal-linear DDPNM: three normal modes per face ...")
    started = time.perf_counter()
    normal_linear_basis = NormalLinearFaceBasis(partition)
    normal_linear_library = build_response_library(
        partition,
        normal_linear_basis,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
    )
    normal_linear_offline = time.perf_counter() - started
    normal_linear_primitive_count = sum(
        len(entry.primitive_modes) for entry in normal_linear_library.entries
    )
    started = time.perf_counter()
    normal_linear_system = InterfaceAssembler(normal_linear_library).assemble(
        np.zeros(n_interfaces, dtype=np.int8)
    )
    normal_linear_solution = reduced_solution(
        partition, normal_linear_library, normal_linear_system
    )
    normal_linear_online = time.perf_counter() - started
    normal_linear_metric = error_metrics(
        partition, normal_linear_solution, reference, volumes
    )
    timings["NormalLinear-DDPNM-3"] = {
        "offline_seconds": normal_linear_offline,
        "online_seconds": normal_linear_online,
        "first_solve_seconds": normal_linear_offline + normal_linear_online,
    }
    method_data["NormalLinear-DDPNM-3"] = {
        "global_unknowns": len(normal_linear_system.global_keys),
        "modes_per_interface": 3,
        "primitive_rhs_columns": normal_linear_primitive_count,
        **_metric_entry(normal_linear_metric),
        **_algebraic_diagnostics(normal_linear_solution),
    }
    print(
        f"      dofs={len(normal_linear_system.global_keys)}, "
        f"offline={normal_linear_offline:.3f} s, online={normal_linear_online:.3f} s, "
        f"L2(u)={normal_linear_metric['velocity_relative_l2']:.3%}"
    )

    print("[6/6] Affine DDPNM: one face entity carrying nine modes ...")
    started = time.perf_counter()
    affine_basis = AffineFaceBasis(partition)
    affine_library = build_response_library(
        partition,
        affine_basis,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
    )
    affine_offline = time.perf_counter() - started
    affine_primitive_count = sum(
        len(entry.primitive_modes) for entry in affine_library.entries
    )
    started = time.perf_counter()
    affine_system = InterfaceAssembler(affine_library).assemble(
        np.full(n_interfaces, 2, dtype=np.int8)
    )
    affine_solution = reduced_solution(partition, affine_library, affine_system)
    affine_online = time.perf_counter() - started
    affine_metric = error_metrics(partition, affine_solution, reference, volumes)
    timings["Affine-DDPNM-9"] = {
        "offline_seconds": affine_offline,
        "online_seconds": affine_online,
        "first_solve_seconds": affine_offline + affine_online,
    }
    method_data["Affine-DDPNM-9"] = {
        "global_unknowns": len(affine_system.global_keys),
        "modes_per_interface": 9,
        "primitive_rhs_columns": affine_primitive_count,
        **_metric_entry(affine_metric),
        **_algebraic_diagnostics(affine_solution),
    }
    print(
        f"      dofs={len(affine_system.global_keys)}, "
        f"offline={affine_offline:.3f} s, online={affine_online:.3f} s, "
        f"L2(u)={affine_metric['velocity_relative_l2']:.3%}"
    )

    # Errors of the reduced methods against the exact FE-trace Schur
    # solution (2-D style "errors_to_exact_FE_schur"); skipped when the
    # dense Schur was not run (watershed partition by default).
    errors_to_schur: dict[str, dict[str, float]] = {}
    if schur_ref is not None:
        for name, solution in [
            ("Classic-DDPNM-1", classic_value),
            ("NormalLinear-DDPNM-3", normal_linear_solution),
            ("Affine-DDPNM-9", affine_solution),
        ]:
            metric = error_metrics(
                partition,
                solution,
                _ReferenceLike(
                    schur_ref.W, schur_ref.solution, reference.boundary_fluxes
                ),
                volumes,
            )
            errors_to_schur[name] = _metric_entry(metric)

    fem_time = timings["Monolithic-FEM"]["first_solve_seconds"]
    for name in (
        "Classic-DDPNM-1",
        "NormalLinear-DDPNM-3",
        "Affine-DDPNM-9",
        "Exact-FE-Schur",
    ):
        if name not in method_data:
            continue
        first = timings[name]["first_solve_seconds"]
        method_data[name]["speedup_vs_fem_first_solve"] = fem_time / first

    _write_outputs(
        partition, args, mesh_seconds, timings, method_data, errors_to_schur,
        schur_ref, reference, points, tetrahedra, volumes,
        classic_library, classic_system, classic_value,
        normal_linear_library, normal_linear_system, normal_linear_solution,
        affine_library, affine_system, affine_solution,
    )


def _write_outputs(partition, args, mesh_seconds, timings, method_data,
                   errors_to_schur, schur_ref, reference, points, tetrahedra,
                   volumes, classic_library, classic_system, classic_value,
                   normal_linear_library, normal_linear_system,
                   normal_linear_solution,
                   affine_library, affine_system, affine_solution) -> None:
    out_dir = args.out_dir
    with (out_dir / "random_affine_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method", "global_unknowns", "modes_per_interface",
                "primitive_rhs_columns", "offline_seconds", "online_seconds",
                "first_solve_seconds", "speedup_vs_fem",
                "velocity_relative_L2", "velocity_relative_broken_H1",
                "pressure_relative_L2", "pressure_aligned_relative_L2",
                "outlet_flux_relative_error",
            ]
        )
        for name in METHODS:
            if name not in method_data:
                continue  # e.g. Exact-FE-Schur skipped for watershed
            values = method_data[name]
            timer = timings[name]
            writer.writerow(
                [
                    name, values["global_unknowns"],
                    values.get("modes_per_interface", "monolithic"),
                    values["primitive_rhs_columns"], timer["offline_seconds"],
                    timer["online_seconds"], timer["first_solve_seconds"],
                    values.get("speedup_vs_fem_first_solve", 0.0),
                    values["velocity_relative_l2"],
                    values["velocity_relative_broken_h1"],
                    values["pressure_relative_l2"],
                    values["pressure_aligned_relative_l2"],
                    values["outlet_flux_relative_error"],
                ]
            )

    slice_data = {
        name: evaluate_fem_ddpnm_slice(
            partition, solution, reference, points, tetrahedra, z_value=0.5
        )
        for name, solution in [
            ("classic", classic_value),
            ("normal_linear", normal_linear_solution),
            ("affine", affine_solution),
        ]
    }
    np.savez_compressed(
        out_dir / "random_benchmark_fields.npz",
        points=points,
        tetrahedra=tetrahedra,
        cell_centers=partition.cell_centers,
        cell_labels=partition.cell_labels,
        cell_volumes=volumes,
        facet_interface_ids=partition.facet_interface_ids,
        interface_pairs=np.asarray(partition.interface_pairs, dtype=np.int32),
        sphere_centers=SPHERES[:, :3],
        sphere_radii=SPHERES[:, 3],
        maximal_ball_centers=partition.pore_seeds[:, :3],
        maximal_ball_radii=partition.pore_seeds[:, 3],
        interface_centers=partition.interface_centers,
        interface_normals=partition.interface_normals,
        interface_areas=partition.interface_areas,
        interface_pressures_classic=classic_value.interface_pressures,
        interface_flux_sums_classic=classic_value.interface_flux_sums,
        schur_matrix_classic=classic_value.schur_matrix,
        interface_pressures_normal_linear=(
            normal_linear_solution.interface_pressures
        ),
        interface_flux_sums_normal_linear=(
            normal_linear_solution.interface_flux_sums
        ),
        schur_matrix_normal_linear=normal_linear_solution.schur_matrix,
        interface_pressures_affine=affine_solution.interface_pressures,
        interface_flux_sums_affine=affine_solution.interface_flux_sums,
        schur_matrix_affine=affine_solution.schur_matrix,
        **{
            f"{name}_{key}": value
            for name, data in slice_data.items()
            for key, value in data.items()
        },
    )
    report = {
        "benchmark": (
            "random-sphere 3-D medium: classic DDPNM vs single-entity "
            "nine-mode affine DDPNM vs exact FE-trace Schur vs monolithic FEM"
        ),
        "geometry": (
            "unit cube minus a frozen seed-20260804 packing of 27 random "
            "spheres (18 clipped wall spheres + 9 interior spheres); throat "
            "faces are the Voronoi faces of the Delaunay pairs, i.e. the "
            "saddle-plane sections bounded by the neighbouring saddle planes"
        ),
        "common_mesh_seconds": mesh_seconds,
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "counts": {
            "solid_spheres": int(len(SPHERES)),
            "pore_subdomains": int(len(np.unique(partition.cell_labels))),
            "interfaces": int(len(partition.interface_pairs)),
            "global_vertices": int(len(points)),
            "global_tetrahedra": int(len(tetrahedra)),
            "interface_surface_patches": partition.cad_counts[
                "interface_surface_patches"
            ],
        },
        "timings": timings,
        "methods": method_data,
        "errors_to_exact_fem": {
            name: {
                key: value
                for key, value in data.items()
                if key
                not in (
                    "global_unknowns",
                    "modes_per_interface",
                    "primitive_rhs_columns",
                    "speedup_vs_fem_first_solve",
                )
            }
            for name, data in method_data.items()
        },
        "errors_to_exact_fe_schur": errors_to_schur,
        "exact_fe_trace_schur": (
            {
                "total_mixed_dofs": int(len(schur_ref.fixed_dofs))
                + int(len(schur_ref.interface_dofs))
                + int(len(schur_ref.interior_dofs)),
                "interface_trace_dofs": int(len(schur_ref.interface_dofs)),
                "interior_dofs": int(len(schur_ref.interior_dofs)),
                "schur_symmetry_error": float(schur_ref.schur_symmetry_error),
                "schur_relative_residual": float(
                    schur_ref.schur_relative_residual
                ),
                "global_relative_residual": float(
                    schur_ref.global_relative_residual
                ),
                "monolithic_relative_difference": float(
                    schur_ref.monolithic_relative_difference
                ),
                "note": (
                    "Exact correctness Schur; must agree with the "
                    "monolithic solve to roundoff."
                ),
            }
            if schur_ref is not None
            else None
        ),
        "reference": {
            "mixed_dofs": int(reference.ndofs),
            "relative_linear_residual": reference.relative_linear_residual,
            "relative_mass_imbalance": reference.relative_mass_imbalance,
        },
    }
    (out_dir / "random_affine_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

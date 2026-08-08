#!/usr/bin/env python3
"""Same-mesh benchmark for classic and single-point affine DDPNM.

The affine method keeps one representative entity per interface.  Its nine
generalized coefficients multiply the facewise modes

    {1, s, t} x {normal, tangent_1, tangent_2}.

No uniform surface control points or nodal minimum-energy extension are used.
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
    raise FileNotFoundError(
        "Expected the unchanged base project at "
        f"{BASE_PROJECT_DIR}."
    )
for import_root in (REPOSITORY_DIR, BASE_PROJECT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ddpnm_core.assembler import InterfaceAssembler
from ddpnm_core.fem_utils import solve_reference
from ddpnm_core.io import topology_arrays
from ddpnm_core.library import build_response_library
from ddpnm_core.validation import finite_element_error_analysis
from ddpnm3d.geometry import build_partition
from ddpnm3d.solver import DdpnmSolution, LocalResponse, build_modes

from affine_face_basis import (
    AffineFaceBasis,
    CompatibleClassicP0Basis,
    NormalLinearFaceBasis,
)


METHODS = (
    "Classic-DDPNM-1",
    "NormalLinear-DDPNM-3",
    "Affine-DDPNM-9",
    "Monolithic-FEM",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-size", type=float, default=0.28)
    parser.add_argument("--sphere-size", type=float, default=0.070)
    parser.add_argument("--boundary-size", type=float, default=0.10)
    parser.add_argument("--interface-size", type=float, default=0.10)
    parser.add_argument("--sphere-band", type=float, default=0.10)
    parser.add_argument("--boundary-band", type=float, default=0.08)
    parser.add_argument("--interface-band", type=float, default=0.08)
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--inlet-pressure", type=float, default=1.0)
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--pressure-stabilization", type=float, default=0.0)
    parser.add_argument("--reference-iterative-threshold", type=int, default=100_000)
    parser.add_argument("--reference-rtol", type=float, default=1.0e-9)
    parser.add_argument("--reference-restart", type=int, default=60)
    parser.add_argument("--reference-maxiter", type=int, default=120)
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


def classic_solution(partition, library, system) -> DdpnmSolution:
    """Convert the independently timed core P0 system to the 3-D API."""
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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, dict[str, float]] = {}
    method_data: dict[str, dict] = {}

    print("[1/5] Building one common mesh ...")
    started = time.perf_counter()
    partition = build_partition(
        mesh_size=args.mesh_size,
        sphere_size=args.sphere_size,
        boundary_size=args.boundary_size,
        interface_size=args.interface_size,
        sphere_band=args.sphere_band,
        boundary_band=args.boundary_band,
        interface_band=args.interface_band,
        mesh_file=args.out_dir / "affine_ddpnm_partition.msh",
    )
    mesh_seconds = time.perf_counter() - started
    points, tetrahedra = topology_arrays(partition.mesh)
    volumes = tetrahedron_volumes(points, tetrahedra)
    n_interfaces = len(partition.interface_pairs)
    print(f"      tetrahedra={len(tetrahedra)}, interfaces={n_interfaces}")

    print("[2/5] Solving the monolithic Taylor--Hood FEM reference ...")
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
        "primitive_rhs_columns": 0,
        "velocity_relative_l2": 0.0,
        "velocity_relative_broken_h1": 0.0,
        "pressure_relative_l2": 0.0,
        "outlet_flux_relative_error": 0.0,
    }
    print(f"      dofs={reference.ndofs}, solve={fem_seconds:.3f} s")

    print("[3/5] Classic DDPNM: one P0 normal coefficient per face ...")
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
    classic_value = classic_solution(partition, classic_library, classic_system)
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
        "velocity_relative_l2": classic_metric["velocity_relative_l2"],
        "velocity_relative_broken_h1": classic_metric[
            "velocity_relative_broken_h1_seminorm"
        ],
        "pressure_relative_l2": classic_metric["pressure_raw_relative_l2"],
        "outlet_flux_relative_error": classic_metric[
            "outlet_flux_relative_error"
        ],
    }
    print(
        f"      dofs={len(classic_system.global_keys)}, "
        f"offline={classic_offline:.3f} s, online={classic_online:.3f} s, "
        f"L2(u)={classic_metric['velocity_relative_l2']:.3%}"
    )
    del classic_value, classic_system, classic_library
    gc.collect()

    print("[4/5] Normal-linear DDPNM: three normal modes per face ...")
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
    normal_linear_solution = classic_solution(
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
        "velocity_relative_l2": normal_linear_metric["velocity_relative_l2"],
        "velocity_relative_broken_h1": normal_linear_metric[
            "velocity_relative_broken_h1_seminorm"
        ],
        "pressure_relative_l2": normal_linear_metric["pressure_raw_relative_l2"],
        "outlet_flux_relative_error": normal_linear_metric[
            "outlet_flux_relative_error"
        ],
    }
    print(
        f"      dofs={len(normal_linear_system.global_keys)}, "
        f"offline={normal_linear_offline:.3f} s, online={normal_linear_online:.3f} s, "
        f"L2(u)={normal_linear_metric['velocity_relative_l2']:.3%}"
    )
    del normal_linear_solution, normal_linear_system, normal_linear_library
    gc.collect()

    print("[5/5] Affine DDPNM: one face entity carrying nine modes ...")
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
    affine_solution = classic_solution(partition, affine_library, affine_system)
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
        "velocity_relative_l2": affine_metric["velocity_relative_l2"],
        "velocity_relative_broken_h1": affine_metric[
            "velocity_relative_broken_h1_seminorm"
        ],
        "pressure_relative_l2": affine_metric["pressure_raw_relative_l2"],
        "outlet_flux_relative_error": affine_metric[
            "outlet_flux_relative_error"
        ],
    }
    print(
        f"      dofs={len(affine_system.global_keys)}, "
        f"offline={affine_offline:.3f} s, online={affine_online:.3f} s, "
        f"L2(u)={affine_metric['velocity_relative_l2']:.3%}"
    )

    fem_time = timings["Monolithic-FEM"]["first_solve_seconds"]
    for name in METHODS:
        first = timings[name]["first_solve_seconds"]
        method_data[name]["speedup_vs_fem_first_solve"] = fem_time / first

    with (args.out_dir / "affine_ddpnm_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method", "global_unknowns", "modes_per_interface",
                "primitive_rhs_columns", "offline_seconds", "online_seconds",
                "first_solve_seconds", "speedup_vs_fem",
                "velocity_relative_L2", "velocity_relative_broken_H1",
                "pressure_relative_L2", "outlet_flux_relative_error",
            ]
        )
        for name in METHODS:
            values = method_data[name]
            timer = timings[name]
            writer.writerow(
                [
                    name, values["global_unknowns"],
                    values.get("modes_per_interface", "monolithic"),
                    values["primitive_rhs_columns"], timer["offline_seconds"],
                    timer["online_seconds"], timer["first_solve_seconds"],
                    values["speedup_vs_fem_first_solve"],
                    values["velocity_relative_l2"],
                    values["velocity_relative_broken_h1"],
                    values["pressure_relative_l2"],
                    values["outlet_flux_relative_error"],
                ]
            )

    report = {
        "benchmark": "single-point classic DDPNM vs single-point nine-mode affine DDPNM vs FEM",
        "interface_space": {
            "classic": "span{1*n}",
            "normal_linear": "span{1,s,t} tensor n",
            "affine": "span{1,s,t} tensor span{n,t1,t2}",
            "uniform_surface_points_used": False,
        },
        "common_mesh_seconds": mesh_seconds,
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "counts": {
            "tetrahedra": len(tetrahedra),
            "subdomains": 64,
            "interfaces": n_interfaces,
        },
        "timings": timings,
        "methods": method_data,
        "reference": {
            "mixed_dofs": int(reference.ndofs),
            "relative_linear_residual": reference.relative_linear_residual,
        },
        "timing_scope": {
            "offline": "local factorizations and primitive response construction",
            "online": "global reduced assembly, solve, and local reconstruction",
            "excluded": "common mesh generation and post-processing validation",
        },
    }
    (args.out_dir / "affine_ddpnm_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Mesh-convergence verification of the DDPNM Stokes velocity/pressure errors.

The same random-27 sphere geometry is meshed at three self-similar
resolutions (bulk size 0.17 / 0.13 / 0.10; all six size and band
parameters scaled by the same factor).  On every level the monolithic
Taylor-Hood FEM Stokes reference and the three DDPNM interface-traction
spaces (Classic-1 / W1n-3 / Affine-9) are solved, and the relative L2
errors are measured against the same-mesh FEM reference with the archived
metric pipeline (``finite_element_error_analysis``).  Empirical convergence
orders p = -d log(err)/d log(h) are reported per error type per method
(consecutive-level pairs, coarse-to-fine span, and a 3-point log-log
regression with r^2).

Stokes only: the two-phase transport is not part of this study — the claim
under test is the velocity-error decay rate.  No snapshots are produced.

Run in the FEniCSx environment:

    conda run -n fenicsx --no-capture-output python run_mesh_convergence.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

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

from run_affine_ddpnm_twophase import (
    json_safe,
    reduced_solution,
    tetrahedron_volumes,
    ddpnm_stokes_metrics,
)

# Self-similar mesh family: every size/band parameter of the benchmark
# defaults (bulk 0.13 family) scaled by the same factor, so the three
# levels differ only in resolution.  Bulk sizes land on round numbers.
BASE_SIZES = {
    "mesh_size": 0.13,
    "sphere_size": 0.065,
    "boundary_size": 0.085,
    "interface_size": 0.075,
    "sphere_band": 0.15,
    "boundary_band": 0.13,
    "interface_band": 0.12,
}
LEVELS = (
    {"name": "coarse", "bulk": 0.17, "scale": 0.17 / 0.13},
    {"name": "medium", "bulk": 0.13, "scale": 1.0},
    {"name": "fine", "bulk": 0.10, "scale": 0.10 / 0.13},
)

METHODS = ("Classic-DDPNM-1", "NormalLinear-DDPNM-3", "Affine-DDPNM-9")
ERROR_KEYS = {
    "velocity_relative_l2_error_vs_fem": "velocity rel L2",
    "velocity_relative_broken_h1_vs_fem": "velocity broken H1",
    "pressure_mean_aligned_relative_l2_error_vs_fem": "pressure aligned rel L2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--inlet-pressure", type=float, default=1.0)
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--pressure-stabilization", type=float, default=0.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "mesh_convergence",
    )
    return parser.parse_args()


def level_sizes(level: dict[str, object]) -> dict[str, float]:
    scale = float(level["scale"])
    return {key: float(value) * scale for key, value in BASE_SIZES.items()}


def mean_cell_size(points: np.ndarray, tetrahedra: np.ndarray) -> tuple[float, int]:
    """Mean tetrahedron size h = mean(volume^(1/3)); also the cell count."""
    volumes = tetrahedron_volumes(points, tetrahedra)
    return float(np.mean(volumes ** (1.0 / 3.0))), len(volumes)


def empirical_orders(errors: dict[str, np.ndarray], h: np.ndarray) -> dict[str, dict[str, float]]:
    """Pairwise and regression orders p = -d log(err)/d log(h) per error key."""
    orders: dict[str, dict[str, float]] = {}
    for key, values in errors.items():
        entry: dict[str, object] = {}
        for a, b in zip(range(len(h) - 1), range(1, len(h))):
            entry[f"p_{LEVELS[a]['name']}_to_{LEVELS[b]['name']}"] = float(
                np.log(values[a] / values[b]) / np.log(h[a] / h[b])
            )
        entry["p_coarse_to_fine"] = float(
            np.log(values[0] / values[-1]) / np.log(h[0] / h[-1])
        )
        slope, intercept = np.polyfit(np.log(h), np.log(values), 1)
        residuals = np.log(values) - (slope * np.log(h) + intercept)
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((np.log(values) - np.mean(np.log(values))) ** 2))
        entry["regression_slope"] = float(slope)
        entry["regression_r_squared"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        orders[key] = entry
    return orders


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hs = np.empty(len(LEVELS))
    timings: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    level_rows: dict[str, dict[str, object]] = {}

    for index, level in enumerate(LEVELS):
        name = level["name"]
        sizes = level_sizes(level)
        print(f"[{name}] building partition mesh (bulk {level['bulk']}) ...")
        t0 = time.perf_counter()
        partition = build_partition(
            mesh_size=sizes["mesh_size"],
            sphere_size=sizes["sphere_size"],
            boundary_size=sizes["boundary_size"],
            interface_size=sizes["interface_size"],
            sphere_band=sizes["sphere_band"],
            boundary_band=sizes["boundary_band"],
            interface_band=sizes["interface_band"],
            mesh_file=args.out_dir / f"mesh_{name}.msh",
        )
        mesh_seconds = time.perf_counter() - t0
        points, tetrahedra = topology_arrays(partition.mesh)
        h, n_cells = mean_cell_size(points, tetrahedra)
        hs[index] = h
        n_interfaces = len(partition.interface_pairs)
        print(
            f"      tetrahedra={n_cells}, vertices={len(points)}, "
            f"interfaces={n_interfaces}, h={h:.5f}, mesh={mesh_seconds:.1f} s"
        )

        print(f"[{name}] FEM Stokes reference ...")
        t0 = time.perf_counter()
        reference = solve_reference(
            partition.mesh,
            viscosity=args.viscosity,
            inlet_pressure=args.inlet_pressure,
            outlet_pressure=args.outlet_pressure,
            pressure_stabilization=args.pressure_stabilization,
        )
        fem_seconds = time.perf_counter() - t0
        volumes = tetrahedron_volumes(points, tetrahedra)
        print(f"      dofs={reference.ndofs}, solve={fem_seconds:.3f} s")

        level_rows[name] = {"h": h, "n_cells": n_cells, "mesh_seconds": mesh_seconds}
        basis_factories = {
            "Classic-DDPNM-1": lambda p: CompatibleClassicP0Basis(),
            "NormalLinear-DDPNM-3": lambda p: NormalLinearFaceBasis(p),
            "Affine-DDPNM-9": lambda p: AffineFaceBasis(p),
        }
        for method in METHODS:
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
            levels = (
                np.full(n_interfaces, 2, dtype=np.int8)
                if method == "Affine-DDPNM-9"
                else np.zeros(n_interfaces, dtype=np.int8)
            )
            t0 = time.perf_counter()
            system = InterfaceAssembler(library).assemble(levels)
            solution = reduced_solution(partition, library, system)
            online = time.perf_counter() - t0
            metrics = ddpnm_stokes_metrics(partition, solution, reference, volumes)
            rows.append(
                {
                    "level": name,
                    "bulk_size": float(level["bulk"]),
                    "h_mean_cube_root_volume": h,
                    "n_tetrahedra": n_cells,
                    "method": method,
                    "global_unknowns": int(len(system.global_keys)),
                    "offline_seconds": float(offline),
                    "online_seconds": float(online),
                    **{key: metrics[key] for key in metrics},
                }
            )
            print(
                f"      {method}: dofs={len(system.global_keys)}, "
                f"offline={offline:.2f} s, online={online:.2f} s, "
                f"velL2={metrics['velocity_relative_l2_error_vs_fem']:.3%}, "
                f"H1={metrics['velocity_relative_broken_h1_vs_fem']:.3%}"
            )
            del library, system, solution

    orders_by_method: dict[str, dict[str, object]] = {}
    method_errors_map: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        method_errors = {key: np.array([r[key] for r in rows if r["method"] == method]) for key in ERROR_KEYS}
        method_errors_map[method] = method_errors
        orders_by_method[method] = empirical_orders(method_errors, hs)

    timings["levels"] = level_rows
    summary = {
        "description": (
            "Mesh-convergence verification of DDPNM Stokes errors on the random-27 "
            "medium: three self-similar mesh levels (bulk 0.17/0.13/0.10), errors vs "
            "same-mesh Taylor-Hood FEM reference, empirical orders per method."
        ),
        "base_sizes": BASE_SIZES,
        "levels": [
            {"name": level["name"], "bulk_size": level["bulk"], "scale": level["scale"]}
            for level in LEVELS
        ],
        "h_definition": "mean of per-tetrahedron volume^(1/3)",
        "order_definition": "p = -d log(err)/d log(h); regression in log-log space",
        "timings": timings,
        "errors_vs_h": {
            method: {
                level["name"]: {key: float(method_errors[key][i]) for key in ERROR_KEYS}
                for i, level in enumerate(LEVELS)
            }
            for method, method_errors in method_errors_map.items()
        },
        "orders": orders_by_method,
        "rows": rows,
    }
    (args.out_dir / "convergence_report.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    # CSV metrics (one row per method x level)
    csv_lines = [
        "level,method,bulk_size,h_mean_cube_root_volume,n_tetrahedra,global_unknowns,"
        "velocity_relative_l2_error_vs_fem,velocity_relative_broken_h1_vs_fem,"
        "pressure_mean_aligned_relative_l2_error_vs_fem,offline_seconds,online_seconds"
    ]
    for row in rows:
        csv_lines.append(
            f"{row['level']},{row['method']},{row['bulk_size']:.4f},{row['h_mean_cube_root_volume']:.6f},"
            f"{row['n_tetrahedra']},{row['global_unknowns']},"
            f"{row['velocity_relative_l2_error_vs_fem']:.8e},"
            f"{row['velocity_relative_broken_h1_vs_fem']:.8e},"
            f"{row['pressure_mean_aligned_relative_l2_error_vs_fem']:.8e},"
            f"{row['offline_seconds']:.3f},{row['online_seconds']:.3f}"
        )
    (args.out_dir / "convergence_metrics.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )
    print(f"Done: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

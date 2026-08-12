#!/usr/bin/env python3
"""Robustness check of the DDPNM mode-enrichment accuracy story on the
**uniform** 27-sphere geometry (3x3x3 regular packing, the archived
``ddpnm_3d_uniform_spheres`` construction), against the random-27 medium.

The uniform geometry module lives in the archive
(``20260810report/ddpnm_3d_uniform_spheres/ddpnm3d/geometry.py``) and is
self-contained (gmsh + dolfinx + scipy only), so it is loaded by file path
and driven through the same pipeline as the random-27 benchmark: monolithic
Taylor-Hood FEM reference + Classic-1 / W1n-3 / Affine-9 with the archived
metric pipeline (``finite_element_error_analysis``).  Mesh parameters match
the archived uniform benchmark (bulk 0.28 family) so the rerun can be
compared bit-for-bit against it.

Question: does the monotone mode-enrichment reduction (Classic ~65% /
W1n ~31% / Affine ~6.5% velocity rel L2 on random-27) hold on a second,
regular geometry -- and at what levels?

Run: conda run -n fenicsx --no-capture-output python run_uniform_robustness.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

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

# Archive uniform-27 geometry module (self-contained: gmsh/dolfinx/scipy).
UNIFORM_GEOMETRY_PY = (
    Path(r"D:\hu\tongjiproj\20260727\20260810report")
    / "ddpnm_3d_uniform_spheres" / "ddpnm3d" / "geometry.py"
)

# Archived uniform benchmark mesh family (affine_ddpnm_3d, bulk 0.28).
UNIFORM_SIZES = {
    "mesh_size": 0.28,
    "sphere_size": 0.070,
    "boundary_size": 0.10,
    "interface_size": 0.10,
    "sphere_band": 0.10,
    "boundary_band": 0.08,
    "interface_band": 0.08,
}

# Random-27 reference (archive benchmark_twophase, same metric pipeline).
RANDOM_27 = {
    "Classic-DDPNM-1": {"eps": 0.653203, "flux": 0.7272, "dofs": 114},
    "NormalLinear-DDPNM-3": {"eps": 0.307488, "flux": 0.1465, "dofs": 342},
    "Affine-DDPNM-9": {"eps": 0.065058, "flux": 0.0271, "dofs": 1026},
}
METHODS = ("Classic-DDPNM-1", "NormalLinear-DDPNM-3", "Affine-DDPNM-9")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--inlet-pressure", type=float, default=1.0)
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--pressure-stabilization", type=float, default=0.0)
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_DIR / "outputs" / "uniform_robustness",
    )
    return parser.parse_args()


def load_uniform_geometry():
    spec = importlib.util.spec_from_file_location("uniform_geometry", UNIFORM_GEOMETRY_PY)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module's @dataclass classes need the module
    # namespace in sys.modules during class creation.
    sys.modules["uniform_geometry"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    geometry = load_uniform_geometry()
    timings: dict[str, object] = {}
    rows: list[dict[str, object]] = []

    print("[1/3] uniform-27 partition mesh (bulk 0.28 family) ...")
    t0 = time.perf_counter()
    partition = geometry.build_partition(
        mesh_size=UNIFORM_SIZES["mesh_size"],
        sphere_size=UNIFORM_SIZES["sphere_size"],
        boundary_size=UNIFORM_SIZES["boundary_size"],
        interface_size=UNIFORM_SIZES["interface_size"],
        sphere_band=UNIFORM_SIZES["sphere_band"],
        boundary_band=UNIFORM_SIZES["boundary_band"],
        interface_band=UNIFORM_SIZES["interface_band"],
        mesh_file=args.out_dir / "uniform_partition.msh",
    )
    mesh_seconds = time.perf_counter() - t0
    points, tetrahedra = topology_arrays(partition.mesh)
    n_cells = len(tetrahedra)
    n_interfaces = len(partition.interface_pairs)
    print(f"      tetrahedra={n_cells}, vertices={len(points)}, "
          f"pores={len(np.unique(partition.cell_labels))}, interfaces={n_interfaces}, "
          f"mesh={mesh_seconds:.1f} s")

    print("[2/3] Taylor-Hood FEM reference ...")
    t0 = time.perf_counter()
    reference = solve_reference(
        partition.mesh,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
    )
    fem_seconds = time.perf_counter() - t0
    timings["FEM"] = {"online_seconds": fem_seconds}
    volumes = tetrahedron_volumes(points, tetrahedra)
    print(f"      dofs={reference.ndofs}, solve={fem_seconds:.3f} s, "
          f"outlet flux={reference.boundary_fluxes['outlet']:.6e}")

    print("[3/3] Classic / W1n / Affine ...")
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
                "method": method,
                "global_unknowns": int(len(system.global_keys)),
                "modes_per_interface": 1 if method == "Classic-DDPNM-1" else (3 if method == "NormalLinear-DDPNM-3" else 9),
                "offline_seconds": float(offline),
                "online_seconds": float(online),
                **{key: metrics[key] for key in metrics},
            }
        )
        timings[method] = {"offline_seconds": float(offline), "online_seconds": float(online)}
        print(
            f"      {method}: dofs={len(system.global_keys)}, "
            f"velL2={metrics['velocity_relative_l2_error_vs_fem']:.3%}, "
            f"H1={metrics['velocity_relative_broken_h1_vs_fem']:.3%}, "
            f"press={metrics['pressure_mean_aligned_relative_l2_error_vs_fem']:.3%}, "
            f"fluxErr={metrics['outlet_flux_relative_error_vs_fem']:.4f}"
        )
        del library, system, solution

    by_method = {r["method"]: r for r in rows}
    comparison = {
        method: {
            "uniform_eps": by_method[method]["velocity_relative_l2_error_vs_fem"],
            "uniform_flux": by_method[method]["outlet_flux_relative_error_vs_fem"],
            "random_eps": RANDOM_27[method]["eps"],
            "random_flux": RANDOM_27[method]["flux"],
            "eps_ratio_uniform_over_random": (
                by_method[method]["velocity_relative_l2_error_vs_fem"] / RANDOM_27[method]["eps"]
            ),
            "flux_ratio_uniform_over_random": (
                by_method[method]["outlet_flux_relative_error_vs_fem"] / RANDOM_27[method]["flux"]
            ),
        }
        for method in METHODS
    }

    summary = {
        "description": (
            "Robustness check: DDPNM mode-enrichment error story on the uniform 27-sphere "
            "(3x3x3 regular packing) vs the random-27 medium; same metric pipeline."
        ),
        "geometry": "uniform 27 spheres (grid 0.2/0.5/0.8, r=0.105), archived ddpnm_3d_uniform_spheres",
        "mesh_sizes": UNIFORM_SIZES,
        "counts": {
            "tetrahedra": n_cells,
            "vertices": int(len(points)),
            "pore_subdomains": int(len(np.unique(partition.cell_labels))),
            "interfaces": n_interfaces,
        },
        "common_mesh_seconds": mesh_seconds,
        "timings": timings,
        "rows": rows,
        "comparison_uniform_vs_random": comparison,
    }
    (args.out_dir / "uniform_robustness_report.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    csv_lines = [
        "method,global_unknowns,velocity_relative_l2_error_vs_fem,velocity_relative_broken_h1_vs_fem,"
        "pressure_mean_aligned_relative_l2_error_vs_fem,outlet_flux_relative_error_vs_fem,"
        "offline_seconds,online_seconds"
    ]
    for row in rows:
        csv_lines.append(
            f"{row['method']},{row['global_unknowns']},"
            f"{row['velocity_relative_l2_error_vs_fem']:.8e},"
            f"{row['velocity_relative_broken_h1_vs_fem']:.8e},"
            f"{row['pressure_mean_aligned_relative_l2_error_vs_fem']:.8e},"
            f"{row['outlet_flux_relative_error_vs_fem']:.8e},"
            f"{row['offline_seconds']:.3f},{row['online_seconds']:.6f}"
        )
    (args.out_dir / "uniform_robustness_metrics.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    # ---- comparison plot ----------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    labels = ["Classic-1", "W1n-3", "Affine-9"]
    x = np.arange(len(METHODS))
    w = 0.36
    for ax, (key, ylabel, title) in zip(
        axes,
        (
            ("eps", "velocity rel L2 error", "Velocity error: random vs uniform 27 spheres"),
            ("flux", "outlet flux rel error", "Flux error: random vs uniform 27 spheres"),
        ),
    ):
        ax.set_facecolor("#fcfcfb")
        random_vals = [RANDOM_27[m][key] for m in METHODS]
        uniform_vals = [comparison[m][f"uniform_{key}"] for m in METHODS]
        ax.bar(x - w / 2, random_vals, w, color="#2a78d6", label="random-27")
        ax.bar(x + w / 2, uniform_vals, w, color="#1baf7a", label="uniform-27")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel, color="#52514e")
        ax.set_title(title, color="#0b0b0b", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.6)
    fig.tight_layout()
    out_png = args.out_dir / "uniform_vs_random.png"
    fig.savefig(out_png, dpi=200, facecolor="#fcfcfb", bbox_inches="tight")
    print(f"Done: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

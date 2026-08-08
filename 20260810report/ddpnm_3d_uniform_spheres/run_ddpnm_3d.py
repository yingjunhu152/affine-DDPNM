#!/usr/bin/env python3
"""Solve the original 3D DDPNM on a unit cube containing a uniform
3x3x3 array of 27 solid spheres."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from ddpnm_core.config import DdpnmConfig
from ddpnm_core.fem_utils import solve_reference

from ddpnm3d.geometry import build_partition
from ddpnm3d.report import write_outputs
from ddpnm3d.solver import solve_ddpnm


def parse_args() -> DdpnmConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the original 3D DDPNM on a unit cube containing a uniform "
            "3x3x3 array of 27 solid spheres."
        )
    )
    # Override 3D-specific defaults
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
    parser.add_argument(
        "--with-reference",
        action="store_true",
        help="Also solve the monolithic P2-P1 Stokes reference problem.",
    )
    parser.add_argument("--reference-iterative-threshold", type=int, default=100_000)
    parser.add_argument("--reference-rtol", type=float, default=1.0e-9)
    parser.add_argument("--reference-restart", type=int, default=60)
    parser.add_argument("--reference-maxiter", type=int, default=120)
    parser.add_argument("--reference-ilu-drop-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--reference-ilu-fill-factor", type=float, default=6.0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/default"),
                       dest="output_dir")
    args = parser.parse_args()
    return DdpnmConfig(
        mesh_size=args.mesh_size,
        sphere_size=args.sphere_size,
        boundary_size=args.boundary_size,
        interface_size=args.interface_size,
        sphere_band=args.sphere_band,
        boundary_band=args.boundary_band,
        interface_band=args.interface_band,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
        output_dir=args.output_dir,
        with_reference=args.with_reference,
        reference_iterative_threshold=args.reference_iterative_threshold,
        reference_rtol=args.reference_rtol,
        reference_restart=args.reference_restart,
        reference_maxiter=args.reference_maxiter,
        reference_ilu_drop_tolerance=args.reference_ilu_drop_tolerance,
        reference_ilu_fill_factor=args.reference_ilu_fill_factor,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("[1/4] Computing maximal balls and generating the refined conforming tetrahedral mesh...")
    mesh_started = time.perf_counter()
    partition = build_partition(
        mesh_size=cfg.mesh_size,
        sphere_size=cfg.sphere_size,
        boundary_size=cfg.boundary_size,
        interface_size=cfg.interface_size,
        sphere_band=cfg.sphere_band,
        boundary_band=cfg.boundary_band,
        interface_band=cfg.interface_band,
        mesh_file=cfg.output_dir / "uniform_27_spheres_partition.msh",
    )
    mesh_time = time.perf_counter() - mesh_started
    print(
        f"      {len(partition.cell_labels)} tetrahedra, "
        f"{len(partition.maximal_balls)} maximal balls, "
        f"{len(np.unique(partition.cell_labels))} subdomains, "
        f"{len(partition.interface_pairs)} interfaces"
    )

    print("[2/4] Building local P2-P1 traction-to-flux maps and solving original DDPNM...")
    solve_started = time.perf_counter()
    solution = solve_ddpnm(
        partition,
        viscosity=cfg.viscosity,
        inlet_pressure=cfg.inlet_pressure,
        outlet_pressure=cfg.outlet_pressure,
        pressure_stabilization=cfg.pressure_stabilization,
    )
    solve_time = time.perf_counter() - solve_started
    print(
        f"      Schur size {solution.schur_matrix.shape[0]}, "
        f"max flux residual {solution.max_mass_residual:.3e}, "
        f"minimum eigenvalue {solution.min_schur_eigenvalue:.3e}"
    )

    reference = None
    if cfg.with_reference:
        print("[3/4] Solving the monolithic Taylor-Hood reference problem...")
        reference_started = time.perf_counter()
        reference = solve_reference(
            partition.mesh,
            viscosity=cfg.viscosity,
            inlet_pressure=cfg.inlet_pressure,
            outlet_pressure=cfg.outlet_pressure,
            pressure_stabilization=cfg.pressure_stabilization,
            iterative_threshold=cfg.reference_iterative_threshold,
            iterative_rtol=cfg.reference_rtol,
            iterative_restart=cfg.reference_restart,
            iterative_maxiter=cfg.reference_maxiter,
            ilu_drop_tolerance=cfg.reference_ilu_drop_tolerance,
            ilu_fill_factor=cfg.reference_ilu_fill_factor,
        )
        reference_time = time.perf_counter() - reference_started
        print(
            f"      {reference.ndofs} mixed dofs, {reference.solver_method}, "
            f"iterations {reference.iterations}, relative residual "
            f"{reference.relative_linear_residual:.3e}"
        )
    else:
        print("[3/4] Monolithic reference skipped (enable with --with-reference).")
        reference_time = 0.0

    print("[4/4] Reconstructing fields and writing mesh, CSV, JSON, NPZ and XDMF outputs...")
    report = write_outputs(
        partition, solution, cfg.output_dir, cfg.as_dict(), reference=reference
    )
    report["timings_seconds"] = {
        "mesh": mesh_time,
        "ddpnm": solve_time,
        "reference": reference_time,
        "total": time.perf_counter() - started,
    }
    with (cfg.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"Done in {report['timings_seconds']['total']:.2f} s")
    print(f"Raw outputs: {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute the exact FE trace Schur complement on the 3-D sphere pack and
benchmark all three reduced interface spaces against it."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from ddpnm3d.geometry import build_partition
from ddpnm3d.basis_3d import HierarchyBasis
from ddpnm_core.library import build_response_library
from ddpnm_core.assembler import InterfaceAssembler
from ddpnm3d.trace_schur import solve_exact_fe_schur, reduced_space_benchmark


def main() -> None:
    out_dir = Path("outputs/trace_schur")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  3-D Exact FE Trace Schur Complement & Reduced-Space Benchmark")
    print("=" * 64)

    # --- 1. Build partition ---
    print("\n[1/4] Building partition ...")
    t0 = time.perf_counter()
    partition = build_partition()
    n_pores = len(np.unique(partition.cell_labels))
    n_interfaces = len(partition.interface_pairs)
    print(f"  {n_pores} pores, {n_interfaces} interfaces")
    print(f"  partition time: {time.perf_counter() - t0:.1f} s")

    # --- 2. Exact FE trace Schur ---
    print("\n[2/4] Solving exact FE trace Schur complement ...")
    t0 = time.perf_counter()
    exact = solve_exact_fe_schur(partition)
    elapsed = time.perf_counter() - t0
    print(f"  interface DOFs: {len(exact.interface_dofs)}")
    print(f"  interior DOFs:  {len(exact.interior_dofs)}")
    print(f"  wall DOFs:      {len(exact.fixed_dofs)}")
    print(f"  Schur symmetry error:            {exact.schur_symmetry_error:.2e}")
    print(f"  Schur relative residual:         {exact.schur_relative_residual:.2e}")
    print(f"  Interior relative residual:      {exact.interior_relative_residual:.2e}")
    print(f"  Global relative residual:        {exact.global_relative_residual:.2e}")
    print(f"  Monolithic relative difference:  {exact.monolithic_relative_difference:.2e}")
    print(f"  solve time: {elapsed:.1f} s")

    # --- 3. Build response library ---
    print("\n[3/4] Building response library (9-mode affine basis) ...")
    t0 = time.perf_counter()
    basis = HierarchyBasis(partition)
    library = build_response_library(partition, basis)
    assembler = InterfaceAssembler(library)
    print(f"  library time: {time.perf_counter() - t0:.1f} s")

    # --- 4. Solve reduced spaces ---
    print("\n[4/4] Solving reduced spaces & benchmarking ...")
    space_names = {0: "P0-DDPNM", 1: "P0-vector-DDPNMT", 2: "P1-vector-HODDPNM"}
    systems: dict[str, object] = {}
    for level in (0, 1, 2):
        levels = np.full(n_interfaces, level, dtype=np.int8)
        system = assembler.assemble(levels)
        systems[space_names[level]] = system
        print(f"  {space_names[level]:22s}: {len(system.global_keys):4d} unknowns, "
              f"min eig = {system.min_schur_eigenvalue:.3e}")

    # --- Benchmark ---
    print("\n--- Reduced-space benchmark ---")
    results = reduced_space_benchmark(exact, library, systems)
    for name, metrics in results.items():
        print(f"  {name:22s}:  "
              f"energy_err = {metrics['schur_energy_relative_error']:.4e},  "
              f"trace_l2   = {metrics['trace_coefficient_relative_l2']:.4e}")

    # Monotonicity check
    energy_vals = [results[n]["schur_energy_relative_error"] for n in space_names.values()]
    monotone = all(
        energy_vals[i] >= energy_vals[i + 1] for i in range(len(energy_vals) - 1)
    )
    print(f"\n  Energy errors monotone (P0 >= P0-vec >= P1-vec): {monotone}")

    # --- Write report ---
    report = {
        "mesh": {
            "pores": n_pores,
            "interfaces": n_interfaces,
        },
        "exact_schur": {
            "interface_dofs": int(len(exact.interface_dofs)),
            "interior_dofs": int(len(exact.interior_dofs)),
            "wall_dofs": int(len(exact.fixed_dofs)),
            "schur_symmetry_error": exact.schur_symmetry_error,
            "schur_relative_residual": exact.schur_relative_residual,
            "interior_relative_residual": exact.interior_relative_residual,
            "global_relative_residual": exact.global_relative_residual,
            "monolithic_relative_difference": exact.monolithic_relative_difference,
        },
        "reduced_spaces": {
            name: {
                "unknowns": len(systems[name].global_keys),
                "min_schur_eigenvalue": systems[name].min_schur_eigenvalue,
                "max_moment_residual": float(
                    np.max(np.abs(systems[name].moment_residuals))
                ),
            }
            for name in space_names.values()
        },
        "benchmark": results,
        "energy_monotone": monotone,
    }
    report_path = out_dir / "trace_schur_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()

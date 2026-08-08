#!/usr/bin/env python3
"""Verify the 3-D exact trace Schur complement against the monolithic FEM solve."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ddpnm3d.geometry import build_partition
from ddpnm3d.trace_schur import solve_exact_fe_schur


TOL = 1.0e-10


def main() -> int:
    print("verify_trace_schur_3d ...")
    partition = build_partition()
    exact = solve_exact_fe_schur(partition)

    failures = 0

    def check(name: str, value: float, limit: float = TOL) -> None:
        nonlocal failures
        ok = value < limit
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name} = {value:.3e}  (limit {limit:.1e})")
        if not ok:
            failures += 1

    check("schur_symmetry_error", exact.schur_symmetry_error)
    check("schur_relative_residual", exact.schur_relative_residual)
    check("interior_relative_residual", exact.interior_relative_residual)
    check("global_relative_residual", exact.global_relative_residual)
    check("monolithic_relative_difference", exact.monolithic_relative_difference)

    # Check benchmark report if available
    report_path = Path("outputs/trace_schur/trace_schur_benchmark.json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        bench = report.get("benchmark", {})
        energy_errs = [
            bench[name]["schur_energy_relative_error"]
            for name in [
                "P0-DDPNM",
                "P0-vector-DDPNMT",
                "P1-vector-HODDPNM",
            ]
            if name in bench
        ]
        if len(energy_errs) == 3:
            monotone = energy_errs[0] >= energy_errs[1] >= energy_errs[2]
            status = "PASS" if monotone else "FAIL"
            print(f"  {status}: energy errors monotone  "
                  f"P0={energy_errs[0]:.3e} >= "
                  f"P0-vec={energy_errs[1]:.3e} >= "
                  f"P1-vec={energy_errs[2]:.3e}")
            if not monotone:
                failures += 1

    if failures:
        print(f"\n{failures} verification(s) FAILED.")
        return 1
    print("\nAll exact trace Schur verifications PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

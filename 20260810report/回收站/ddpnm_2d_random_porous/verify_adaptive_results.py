from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify adaptive DDPNM hierarchy invariants.")
    parser.add_argument(
        "report", type=Path, nargs="?",
        default=Path("outputs/adaptive_hierarchy/adaptive_report.json"),
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    counts = report["counts"]
    systems = report["systems"]
    exact = report["exact_FE_trace_schur"]
    checks = {
        "all interfaces classified": (
            counts["final_DDPNM_interfaces"]
            + counts["final_DDPNMT_interfaces"]
            + counts["final_HODDPNM_interfaces"]
            == counts["interfaces"]
        ),
        "nested interface dimensions": (
            systems["DDPNM"]["interface_unknowns"]
            < systems["DDPNMT"]["interface_unknowns"]
            < systems["HODDPNM"]["interface_unknowns"]
        ),
        "local Schur symmetry": report["local_response_library"]["maximum_symmetry_error"] < 1.0e-10,
        "all modal moment balances": max(
            item["maximum_interface_moment_residual"] for item in systems.values()
        ) < 1.0e-10,
        "adaptive target reached": (
            report["final_error_to_full_HODDPNM"]["combined"]
            <= report["target_tolerance"] * (1.0 + 1.0e-10)
        ),
        "exact Schur symmetry": exact["schur_symmetry_error"] < 1.0e-10,
        "exact Schur residual": exact["schur_relative_residual"] < 1.0e-10,
        "exact Schur reconstruction residual": exact["global_relative_residual"] < 1.0e-10,
        "Schur equals monolithic FEM": exact["relative_difference_from_monolithic"] < 1.0e-10,
        "full HODDPNM validation": report["errors_to_exact_FE_schur"]["HODDPNM"]["velocity"] < 0.02,
    }
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

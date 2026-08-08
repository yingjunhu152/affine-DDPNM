from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DD-PNM numerical invariants.")
    parser.add_argument("report", type=Path, nargs="?", default=Path("outputs/enriched/report.json"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    checks = {
        "one subdomain per analytic pore region": (
            report["counts"]["pore_subdomains"] == report["counts"]["analytic_throat_cuts"] - 8
        ),
        "constant plus linear interface modes": report["interface_system"]["modes_per_interface"] == 2,
        "positive Schur matrix": report["interface_system"]["minimum_eigenvalue"] > 0.0,
        "zeroth/first flux-moment conservation": report["interface_system"]["maximum_flux_balance_residual"] < 1.0e-9,
        "global linear residual": report["interface_system"]["relative_linear_residual"] < 1.0e-10,
        "local DtN symmetry": max(x["dtn_symmetry_error"] for x in report["local_subdomains"]) < 1.0e-8,
        "velocity validation": report["validation"]["vertex_velocity_relative_l2"] < 0.20,
        "pressure validation": report["validation"]["vertex_pressure_mean_aligned_relative_l2"] < 0.10,
    }
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

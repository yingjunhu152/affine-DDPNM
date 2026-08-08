from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    out_dir = parse_args().out_dir
    report = json.loads((out_dir / "adaptive_report.json").read_text(encoding="utf-8"))
    assert report["counts"]["interfaces"] == 144
    assert report["systems"]["DDPNM"]["interface_unknowns"] == 144
    assert report["systems"]["DDPNMT"]["interface_unknowns"] == 432
    assert report["systems"]["HODDPNM"]["interface_unknowns"] == 1296
    tolerance = report["parameters"]["target_tolerance"]
    assert report["final_error_to_full_HODDPNM"]["combined"] <= tolerance * (1.0 + 1.0e-8)
    for name in ("DDPNM", "DDPNMT", "HODDPNM", "adaptive_final"):
        system = report["systems"][name]
        assert system["relative_linear_residual"] < 1.0e-9
        assert system["maximum_interface_moment_residual"] < 1.0e-9
        assert system["relative_mass_imbalance"] < 1.0e-8
    fem = report["traditional_fem"]
    assert fem["relative_linear_residual"] < 1.0e-8
    assert fem["relative_mass_imbalance"] < 1.0e-7
    required = (
        "adaptive_3d_results.npz",
        "adaptive_history.csv",
        "ADAPTIVE_3D_ALGORITHM.md",
        "01_adaptive_convergence.png",
        "02_adaptive_final_interface_hierarchy.png",
        "03_method_errors_to_fem.png",
        "04_adaptive_fem_error_fields.png",
        "05_adaptive_algorithm_box.png",
    )
    for filename in required:
        assert (out_dir / filename).is_file(), filename
    print("Adaptive 3D verification passed.")


if __name__ == "__main__":
    main()

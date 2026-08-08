from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    out_dir = parser.parse_args().out_dir
    report = json.loads(
        (out_dir / "strict_2d_p0_report.json").read_text(encoding="utf-8")
    )
    assert report["counts"]["interfaces"] == 35
    assert report["counts"]["ddpnm_interface_unknowns"] == 35
    metrics = report["strict_validation"]
    assert metrics["same_parent_mesh_object"]
    assert metrics["quadrature_degree"] == 6
    assert metrics["maximum_global_to_local_dof_coordinate_mismatch"] < 1.0e-10
    assert metrics["ddpnm_relative_mass_imbalance"] < 1.0e-7
    assert metrics["fem_relative_mass_imbalance"] < 1.0e-7
    assert report["systems"]["ddpnm_maximum_interface_residual"] < 1.0e-9
    assert report["systems"]["fem_relative_linear_residual"] < 1.0e-8
    for filename in (
        "strict_2d_p0_error_data.npz",
        "strict_2d_p0_metrics.csv",
        "strict_2d_p0_ddpnm_error_fields.png",
    ):
        assert (out_dir / filename).is_file(), filename
    print("Strict 2D P0-DDPNM verification passed.")


if __name__ == "__main__":
    main()

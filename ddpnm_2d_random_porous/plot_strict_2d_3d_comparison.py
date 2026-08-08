from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("velocity_relative_l2", r"velocity $L^2$"),
    ("velocity_relative_broken_h1_seminorm", r"velocity broken-$H^1$"),
    ("pressure_raw_relative_l2", r"pressure raw $L^2$"),
    ("pressure_mean_aligned_relative_l2", r"pressure aligned $L^2$"),
    ("outlet_flux_relative_error", "outlet flux"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-2d", type=Path, required=True)
    parser.add_argument("--report-3d", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_2d = json.loads(args.report_2d.read_text(encoding="utf-8"))
    report_3d = json.loads(args.report_3d.read_text(encoding="utf-8"))
    values_2d = report_2d["strict_validation"]
    values_3d = report_3d["strict_errors_to_identical_mesh_FEM"]["DDPNM"]
    array_2d = np.asarray([values_2d[key] for key, _ in METRICS])
    array_3d = np.asarray([values_3d[key] for key, _ in METRICS])

    x = np.arange(len(METRICS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.22)
    bars_2d = ax.bar(x - width / 2, 100 * array_2d, width, label="2D random circles", color="#2878b5")
    bars_3d = ax.bar(x + width / 2, 100 * array_3d, width, label="3D regular spheres", color="#d1495b")
    ax.set_yscale("log")
    ax.set_xticks(x, [label for _, label in METRICS])
    ax.set_ylabel("strict relative error to identical-mesh FEM (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Original P0-DDPNM under one strict error definition")
    for bars in (bars_2d, bars_3d):
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5)
    fig.text(
        0.5,
        0.035,
        "Same norms and quadrature; geometries are not matched, so this is a case comparison rather than a pure dimensional study.",
        ha="center",
        fontsize=9.2,
    )
    fig.savefig(args.out_dir / "strict_2d_3d_error_comparison.png", dpi=260)
    plt.close(fig)

    with (args.out_dir / "strict_2d_3d_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "2D", "3D", "3D_to_2D_ratio"])
        for (key, _), value_2d, value_3d in zip(
            METRICS, array_2d, array_3d, strict=True
        ):
            writer.writerow([key, value_2d, value_3d, value_3d / value_2d])


if __name__ == "__main__":
    main()

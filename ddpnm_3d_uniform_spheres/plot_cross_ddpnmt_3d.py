#!/usr/bin/env python3
"""Paper-style comparison plots for DDPNM/DDPNMT and cross variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/cross_ddpnmt/cross_ddpnmt_report.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cross_ddpnmt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    methods = ["DDPNM", "DDPNMT", "Cross-DDPNM", "Cross-DDPNMT"]
    labels = ["DDPNM", "DDPNMT", "Cross\nDDPNM", "Cross\nDDPNMT"]
    colors = ["#8f9397", "#f28e2b", "#4e79a7", "#d62728"]
    metrics = report["strict_errors_to_identical_mesh_FEM"]
    systems = report["systems"]

    plt.rcParams.update(
        {"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8}
    )
    metric_specs = [
        ("velocity_relative_l2", r"velocity $L^2$"),
        ("velocity_relative_broken_h1_seminorm", r"broken-$H^1$"),
        ("pressure_raw_relative_l2", r"pressure $L^2$"),
        ("outlet_flux_relative_error", "outlet flux"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.7), constrained_layout=True)
    for axis, (field, title) in zip(axes.ravel(), metric_specs, strict=True):
        values = 100.0 * np.asarray([metrics[name][field] for name in methods])
        bars = axis.bar(np.arange(4), values, color=colors, width=0.72)
        axis.bar_label(bars, fmt="%.2f%%", padding=2, fontsize=8)
        axis.set_xticks(np.arange(4), labels)
        axis.set_ylabel("relative error (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        axis.set_ylim(0.0, max(values) * 1.18)
    figure.suptitle("Same-mesh comparison against monolithic Taylor--Hood FEM", fontsize=14)
    figure.savefig(
        args.out_dir / "01_ddpnm_ddpnmt_cross_error_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    dofs = np.asarray([systems[name]["interface_unknowns"] for name in methods])
    velocity = 100.0 * np.asarray(
        [metrics[name]["velocity_relative_l2"] for name in methods]
    )
    flux = 100.0 * np.asarray(
        [metrics[name]["outlet_flux_relative_error"] for name in methods]
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    for axis, values, ylabel, title in [
        (axes[0], velocity, r"velocity $L^2$ error (%)", "Velocity accuracy versus interface size"),
        (axes[1], flux, "outlet flux error (%)", "Flow-rate accuracy versus interface size"),
    ]:
        for index, name in enumerate(methods):
            axis.scatter(dofs[index], values[index], s=58, color=colors[index], zorder=3)
            axis.annotate(
                name,
                (dofs[index], values[index]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xscale("log")
        axis.set_xlabel("global interface unknowns (log scale)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.22)
    figure.savefig(
        args.out_dir / "02_accuracy_vs_interface_unknowns.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()

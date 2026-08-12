#!/usr/bin/env python3
"""Log-log mesh-convergence plot for the DDPNM Stokes error study.

Reads ``convergence_report.json`` (written by ``run_mesh_convergence.py``)
and draws one panel per error metric (velocity rel L2, velocity broken H1,
pressure aligned rel L2) with the three DDPNM series, dashed O(h)/O(h^2)
reference guides, and the fitted regression slope labelled per series.

Run in the FEniCSx environment (needs matplotlib):

    conda run -n fenicsx --no-capture-output python plot_mesh_convergence.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Validated categorical slots (light surface): blue / orange / aqua.
SERIES = {
    "Classic-DDPNM-1": {"color": "#2a78d6", "marker": "o"},
    "NormalLinear-DDPNM-3": {"color": "#eb6834", "marker": "s"},
    "Affine-DDPNM-9": {"color": "#1baf7a", "marker": "D"},
}
PANELS = [
    ("velocity_relative_l2_error_vs_fem", "velocity rel L2 error"),
    ("velocity_relative_broken_h1_vs_fem", "velocity broken H1 error"),
    ("pressure_mean_aligned_relative_l2_error_vs_fem", "pressure aligned rel L2 error"),
]

SURFACE = "#fcfcfb"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
GUIDE = "#c3c2b7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=PROJECT_DIR / "outputs" / "mesh_convergence" / "convergence_report.json"
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_DIR / "outputs" / "mesh_convergence" / "convergence_plot.png"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.data.read_text(encoding="utf-8"))
    levels = report["levels"]
    hs = np.asarray([report["timings"]["levels"][level["name"]]["h"] for level in levels], dtype=float)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(12.5, 4.0))
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, label) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
        ax.grid(True, which="both", color=GRIDLINE, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=SECONDARY, labelsize=9)
        ax.set_xlabel(r"mean cell size $h = \langle V^{1/3}\rangle$", color=SECONDARY, fontsize=10)
        ax.set_ylabel(label, color=SECONDARY, fontsize=10)
        ax.set_title(label, color=PRIMARY, fontsize=11, pad=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4g"))

        for method, style in SERIES.items():
            values = np.asarray(
                [report["errors_vs_h"][method][level["name"]][key] for level in levels], dtype=float
            )
            slope = report["orders"][method][key]["regression_slope"]
            line, = ax.plot(
                hs, values,
                color=style["color"],
                marker=style["marker"],
                markersize=9,
                markerfacecolor=style["color"],
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                linewidth=2.0,
                zorder=3,
            )
            # Direct slope label (relief for the sub-3:1 aqua series).
            mid = len(hs) // 2
            ax.annotate(
                f"p ≈ {slope:.2f}",
                xy=(hs[mid], values[mid]),
                xytext=(hs[mid] * 0.92, values[mid] * 1.45),
                color=style["color"],
                fontsize=9,
                zorder=4,
            )

        # Reference guides: O(h) anchored on the Classic coarse point,
        # O(h^2) anchored on the Affine coarse point.
        for order, anchor_method in ((1, "Classic-DDPNM-1"), (2, "Affine-DDPNM-9")):
            anchor_err = report["errors_vs_h"][anchor_method]["coarse"][key]
            guide_h = np.geomspace(hs[0], hs[-1], 50)
            guide_err = anchor_err * (guide_h / hs[0]) ** order
            ax.plot(
                guide_h, guide_err,
                linestyle=(0, (5, 4)),
                color=GUIDE,
                linewidth=1.2,
                zorder=1,
            )
            ax.text(
                guide_h[-1] * 1.02, guide_err[-1],
                rf"$O(h^{order})$",
                color=MUTED,
                fontsize=9,
                va="center",
            )

    handles = [
        plt.Line2D(
            [], [],
            color=style["color"], marker=style["marker"], markersize=9,
            markerfacecolor=style["color"], markeredgecolor=SURFACE,
            linewidth=2.0, label=method,
        )
        for method, style in SERIES.items()
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.985), fontsize=10, labelcolor=PRIMARY,
    )
    fig.suptitle(
        "DDPNM Stokes error vs mesh size (random-27 medium, same-mesh Taylor-Hood FEM reference)",
        color=PRIMARY, fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Done: {args.out.resolve()}")


if __name__ == "__main__":
    main()

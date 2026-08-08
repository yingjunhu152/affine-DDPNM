#!/usr/bin/env python3
"""Make a paper-style accuracy/cost comparison figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-csv",
        type=Path,
        default=PROJECT_DIR / "outputs" / "benchmark" / "affine_ddpnm_metrics.csv",
    )
    parser.add_argument(
        "--uniform-csv",
        type=Path,
        default=(
            PROJECT_DIR.parent
            / "ddpnm_3d_uniform_spheres"
            / "outputs"
            / "affine_uniform_point_smoke"
            / "method_error_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_DIR
            / "outputs"
            / "benchmark"
            / "affine_ddpnm_error_cost_comparison.png"
        ),
    )
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["method"]: row for row in rows}


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    args = parse_args()
    new = read_rows(args.new_csv)
    old = read_rows(args.uniform_csv)

    classic = new["Classic-DDPNM-1"]
    normal_linear = new["NormalLinear-DDPNM-3"]
    affine = new["Affine-DDPNM-9"]
    fem = new["Monolithic-FEM"]
    uniform = old["Uniform-DDPNMT"]

    colors = {
        "classic": "#7A7F87",
        "w1n": "#7B1FA2",
        "affine": "#1565C0",
        "uniform": "#2E9D65",
        "fem": "#B23A48",
    }
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))

    # (a) Error comparison.  Uniform-DDPNMT is retained as a historical
    # same-mesh reference, but it is not used by the new Affine-DDPNM run.
    error_labels = [r"$L^2(u)$", r"broken-$H^1(u)$", r"$L^2(p)$", "outlet flux"]
    error_keys_new = [
        "velocity_relative_L2",
        "velocity_relative_broken_H1",
        "pressure_relative_L2",
        "outlet_flux_relative_error",
    ]
    error_keys_old = [
        "velocity_relative_L2",
        "velocity_relative_broken_H1",
        "pressure_raw_relative_L2",
        "outlet_flux_relative_error",
    ]
    positions = np.arange(len(error_labels))
    width = 0.18
    classic_errors = 100.0 * np.asarray([value(classic, key) for key in error_keys_new])
    normal_linear_errors = 100.0 * np.asarray(
        [value(normal_linear, key) for key in error_keys_new]
    )
    affine_errors = 100.0 * np.asarray([value(affine, key) for key in error_keys_new])
    uniform_errors = 100.0 * np.asarray([value(uniform, key) for key in error_keys_old])
    ax = axes[0, 0]
    ax.bar(positions - 1.5 * width, classic_errors, width, color=colors["classic"], label="Classic DDPNM")
    ax.bar(positions - 0.5 * width, normal_linear_errors, width, color=colors["w1n"], label=r"$W_{1n}$ normal-linear")
    ax.bar(positions + 0.5 * width, affine_errors, width, color=colors["affine"], label="Affine DDPNM (new)")
    ax.bar(positions + 1.5 * width, uniform_errors, width, color=colors["uniform"], hatch="//", label="Uniform DDPNMT (kept)")
    ax.set_xticks(positions, error_labels)
    ax.set_ylabel("Relative error (%)")
    ax.set_title("(a) Same-mesh errors")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # (b) Global unknown counts.
    ax = axes[0, 1]
    dof_labels = [
        "Classic\nDDPNM",
        r"$W_{1n}$\nDDPNM",
        "Affine\nDDPNM",
        "Uniform\nDDPNMT",
        "Monolithic\nFEM",
    ]
    dofs = [
        int(classic["global_unknowns"]),
        int(normal_linear["global_unknowns"]),
        int(affine["global_unknowns"]),
        int(uniform["interface_unknowns"]),
        int(fem["global_unknowns"]),
    ]
    bars = ax.bar(
        np.arange(5),
        dofs,
        color=[
            colors["classic"],
            colors["w1n"],
            colors["affine"],
            colors["uniform"],
            colors["fem"],
        ],
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(5), dof_labels)
    ax.set_ylabel("Global unknowns (log scale)")
    ax.set_title("(b) Reduced-system size")
    ax.grid(axis="y", which="both", alpha=0.25)
    for bar, count in zip(bars, dofs, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, count * 1.13, f"{count:,}", ha="center", va="bottom", fontsize=9)

    # (c) Independently measured first-solve cost.  The old uniform timing
    # bundled six solves, so it is intentionally not shown as a comparable bar.
    ax = axes[1, 0]
    cost_labels = ["Classic DDPNM", r"$W_{1n}$ DDPNM", "Affine DDPNM", "Monolithic FEM"]
    rows = [classic, normal_linear, affine, fem]
    offline = np.asarray([value(row, "offline_seconds") for row in rows])
    online = np.asarray([value(row, "online_seconds") for row in rows])
    x = np.arange(4)
    ax.bar(x, offline, color="#8DB7E8", label="offline local library")
    ax.bar(x, online, bottom=offline, color="#174A7E", label="online solve")
    ax.set_xticks(x, cost_labels, rotation=8)
    ax.set_ylabel("Wall time (s)")
    ax.set_title("(c) Independent first-solve cost")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    for index, total in enumerate(offline + online):
        ax.text(index, total + 1.5, f"{total:.2f} s", ha="center", fontsize=9)

    # (d) Accuracy versus interface/global unknowns.
    ax = axes[1, 1]
    reduced_dofs = np.asarray(
        [dofs[0], dofs[1], dofs[2], dofs[3]], dtype=float
    )
    velocity_errors = np.asarray(
        [classic_errors[0], normal_linear_errors[0], affine_errors[0], uniform_errors[0]],
        dtype=float,
    )
    labels = ["Classic", r"$W_{1n}$", "Affine (new)", "Uniform (kept)"]
    point_colors = [colors["classic"], colors["w1n"], colors["affine"], colors["uniform"]]
    for x_value, y_value, label, color in zip(
        reduced_dofs, velocity_errors, labels, point_colors, strict=True
    ):
        ax.scatter(x_value, y_value, s=75, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(label, (x_value, y_value), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Interface unknowns (log scale)")
    ax.set_ylabel(r"Velocity relative $L^2$ error (%)")
    ax.set_title("(d) Accuracy per interface unknown")
    ax.grid(which="both", alpha=0.25)

    figure.suptitle(
        r"One interface entity: $\{1,s,t\}\otimes\{\mathbf{n},\mathbf{t}_1,\mathbf{t}_2\}$",
        fontsize=13,
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

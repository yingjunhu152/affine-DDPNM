from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


METRICS = (
    ("velocity_relative_l2", r"velocity $L^2$"),
    ("velocity_relative_broken_h1_seminorm", r"velocity broken-$H^1$"),
    ("pressure_raw_relative_l2", r"pressure raw $L^2$"),
    ("pressure_mean_aligned_relative_l2", r"pressure aligned $L^2$"),
    ("outlet_flux_relative_error", "outlet flux"),
)
METHODS = (
    ("DDPNM", "P0 normal (1/face)", "#4c78a8"),
    ("DDPNMT", "P0 vector (3/face)", "#f2a541"),
    ("HODDPNM", "P1 vector (9/face)", "#2a9d8f"),
)


def algorithm_box(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    ax.set_axis_off()
    ax.add_patch(
        Rectangle((0.025, 0.025), 0.95, 0.95, fill=False, lw=1.8, transform=ax.transAxes)
    )
    ax.plot([0.025, 0.975], [0.895, 0.895], color="black", lw=1.2, transform=ax.transAxes)
    ax.text(
        0.05,
        0.935,
        "Algorithm 1  Three-dimensional vector-affine HODDPNM-P1(9)",
        fontsize=16,
        fontweight="bold",
        va="center",
        transform=ax.transAxes,
    )
    lines = [
        r"Input: subdomains $\{\Omega_i\}$, saddle-cut interfaces $\{\Gamma_{ij}\}$, local P2-P1 FE systems.",
        r"1:  For every $\Gamma_{ij}$ construct $(\mathbf{n},\mathbf{t}_1,\mathbf{t}_2)$ and scaled coordinates $(s,t)$.",
        r"2:  Add the constant tangential modes $\mathbf{t}_1$ and $\mathbf{t}_2$ to the original mode $\mathbf{n}$.",
        r"3:  Add the linear normal modes $s\mathbf{n}$ and $t\mathbf{n}$.",
        r"4:  Add the linear tangential modes $s\mathbf{t}_1,t\mathbf{t}_1,s\mathbf{t}_2,t\mathbf{t}_2$.",
        r"5:  Hence $\mathcal{T}_{ij}^{(9)}=\mathrm{span}\{\mathbf{n},\mathbf{t}_1,\mathbf{t}_2,s\mathbf{n},t\mathbf{n},$",
        r"    $s\mathbf{t}_1,t\mathbf{t}_1,s\mathbf{t}_2,t\mathbf{t}_2\}$ (nine coefficients per interface).",
        r"6:  Factor each local Stokes matrix once and compute one response for every incident basis traction.",
        r"7:  Statically condense local FE interiors and assemble $S_9\boldsymbol{\lambda}_9=\mathbf{b}_9$.",
        r"8:  Solve the interface system and reconstruct $(\mathbf{u}_i,p_i)$ by local response superposition.",
        r"9:  Verify all interface moments, global mass balance, and the linear residual.",
        r"Output: the nine interface coefficients and the reconstructed subdomain velocity-pressure fields.",
    ]
    y = 0.845
    for index, line in enumerate(lines):
        ax.text(
            0.055,
            y,
            line,
            fontsize=12.6,
            va="top",
            transform=ax.transAxes,
            color="#1d1d1d",
        )
        y -= 0.064 if index not in (4, 5) else 0.058
    ax.plot([0.025, 0.975], [0.095, 0.095], color="black", lw=1.0, transform=ax.transAxes)
    ax.text(
        0.05,
        0.058,
        "This is a reduced nine-mode trace Schur complement, not the exact full finite-element trace Schur complement.",
        fontsize=10.8,
        style="italic",
        transform=ax.transAxes,
    )
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    metrics = report["strict_errors_to_identical_mesh_FEM"]

    algorithm_box(args.out_dir / "01_hoddpnm_p1_9_algorithm_box.png")

    x = np.arange(len(METRICS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.18)
    for method_index, (key, label, color) in enumerate(METHODS):
        values = 100 * np.asarray([metrics[key][metric] for metric, _ in METRICS])
        bars = ax.bar(
            x + (method_index - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8.3)
    ax.set_yscale("log")
    ax.set_xticks(x, [label for _, label in METRICS])
    ax.set_ylabel("strict relative error to identical-mesh FEM (%)")
    ax.set_title("Effect of the complete nine-mode vector-affine interface space")
    ax.set_ylim(top=70.0)
    ax.legend(frameon=False, ncol=1, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(args.out_dir / "02_interface_enrichment_errors.png", dpi=260)
    plt.close(fig)

    with (args.out_dir / "affine_hoddpnm_p1_9_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "unknowns_per_interface", *[key for key, _ in METRICS]])
        for method, _, _ in METHODS:
            writer.writerow(
                [
                    method,
                    {"DDPNM": 1, "DDPNMT": 3, "HODDPNM": 9}[method],
                    *[metrics[method][metric] for metric, _ in METRICS],
                ]
            )

    hodd = metrics["HODDPNM"]
    summary = {
        "method": "HODDPNM-P1(9), reduced vector-affine trace Schur complement",
        "mode_groups_per_interface": {
            "constant_vector": 3,
            "linear_normal": 2,
            "linear_tangential": 4,
            "total": 9,
        },
        "interfaces": report["counts"]["interfaces"],
        "global_interface_unknowns": report["systems"]["HODDPNM"]["interface_unknowns"],
        "system_diagnostics": report["systems"]["HODDPNM"],
        "strict_errors_to_identical_mesh_fem": hodd,
    }
    (args.out_dir / "affine_hoddpnm_p1_9_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

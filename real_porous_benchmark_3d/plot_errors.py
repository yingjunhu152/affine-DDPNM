#!/usr/bin/env python3
"""Error and cost plots for the real-porous benchmark.

Reads benchmark_report.json from an output directory.
Produces: error_bars.png, cost_comparison.png
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_all(out_dir: str):
    out = Path(out_dir)
    with open(out / "benchmark_report.json") as fh:
        report = json.load(fh)

    methods_dict = report["methods"]  # dict keyed by method name
    timings = report.get("timings", {})
    part_name = report["partition"]
    n_cells = report["mesh_cells"]
    n_ifaces = report.get("n_interfaces", "?")
    porosity = report["parameters"]["porosity"]

    # ── Figure 1: Error bars ─────────────────────────────────────
    _plot_error_bars(methods_dict, part_name, n_cells, porosity, out)

    # ── Figure 2: Cost comparison ────────────────────────────────
    _plot_cost(methods_dict, timings, part_name, out)

    # ── Figure 3: Summary table as figure ────────────────────────
    _plot_table(methods_dict, timings, part_name, n_cells, n_ifaces, porosity, out)

    print(f"Plots saved to {out}/")


def _plot_error_bars(methods, part_name, n_cells, porosity, out_dir):
    dd = methods  # already a dict
    order = ["Classic-DDPNM", "Affine-DDPNM", "HODDPNM-adaptive"]
    names = [n for n in order if n in dd]

    error_keys = [
        ("velocity_relative_l2", "Velocity L²"),
        ("velocity_relative_broken_h1", "Velocity broken-H¹"),
        ("pressure_relative_l2", "Pressure L²"),
        ("outlet_flux_relative_error", "Outlet flux"),
    ]

    n_groups = len(names)
    n_bars = len(error_keys)
    x = np.arange(n_groups)
    width = 0.2
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, (key, label) in enumerate(error_keys):
        vals = []
        for name in names:
            v = dd[name].get(key, float("nan"))
            if isinstance(v, (int, float)) and not np.isnan(v):
                vals.append(v * 100)
            else:
                vals.append(0)
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=colors[i], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                        f"{val:.1f}%", ha="center", fontsize=7, fontweight="bold",
                        rotation=90, va="bottom")

    ax.set_xticks(x + width * (n_bars - 1) / 2)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("relative error (%)", fontsize=11)
    ax.set_title(f"{part_name} partition  |  {n_cells:,} cells  |  "
                 f"porosity {porosity*100:.1f}%", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "error_bars.png", dpi=150)
    plt.close(fig)


def _plot_cost(methods, timings, part_name, out_dir):
    dd = methods  # already a dict
    order = ["Classic-DDPNM", "Affine-DDPNM", "HODDPNM-adaptive", "Monolithic-FEM"]
    names = [n for n in order if n in dd]
    times = [timings.get(n, {}).get("first_solve_seconds", 0) for n in names]
    mems = [timings.get(n, {}).get("peak_memory_mib", 0) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#95a5a6"]

    # Time
    bars1 = ax1.bar(range(len(names)), times, color=colors[:len(names)],
                    edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars1, times):
        if val and not (isinstance(val, float) and np.isnan(val)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                     f"{val:.1f}s", ha="center", fontsize=10, fontweight="bold")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("wall time (s)", fontsize=11)
    ax1.set_title(f"Compute time — {part_name}", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Memory
    bars2 = ax2.bar(range(len(names)), mems, color=colors[:len(names)],
                    edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars2, mems):
        if val and not (isinstance(val, float) and np.isnan(val)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mems)*0.02,
                     f"{val:.0f} MiB", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("peak memory (MiB)", fontsize=11)
    ax2.set_title(f"Peak memory — {part_name}", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "cost_comparison.png", dpi=150)
    plt.close(fig)


def _plot_table(methods, timings, part_name, n_cells, n_ifaces, porosity, out_dir):
    """Render a summary table as a matplotlib figure."""
    dd = methods  # already a dict
    order = ["Classic-DDPNM", "Affine-DDPNM", "HODDPNM-adaptive", "Monolithic-FEM"]
    names = [n for n in order if n in dd]

    fig, ax = plt.subplots(figsize=(12, 2.5 + 0.4 * len(names)))
    ax.axis("off")

    col_labels = ["Method", "Unknowns", "L² error", "H¹ error",
                  "Pressure L²", "Flux error", "Time", "Memory"]
    rows = []
    for name in names:
        d = dd[name]
        tm = timings.get(name, {})
        uk = d.get("global_unknowns", "-")
        if isinstance(uk, float): uk = int(uk)
        rows.append([
            name,
            str(uk),
            _pct(d.get("velocity_relative_l2")),
            _pct(d.get("velocity_relative_broken_h1")),
            _pct(d.get("pressure_relative_l2")),
            _pct(d.get("outlet_flux_relative_error")),
            f"{tm.get('first_solve_seconds', 0):.1f}s" if tm.get("first_solve_seconds") else "-",
            f"{tm.get('peak_memory_mib', 0):.0f} MiB" if tm.get("peak_memory_mib") else "-",
        ])

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center", colColours=["#2c3e50"]*len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_facecolor("#2c3e50")

    # Alternate row colors
    for i in range(len(names)):
        for j in range(len(col_labels)):
            cell = table[i+1, j]
            if i % 2 == 0:
                cell.set_facecolor("#ecf0f1")

    ax.set_title(f"{part_name} partition  |  {n_cells:,} cells  |  "
                 f"{n_ifaces} interfaces  |  "
                 f"porosity {porosity*100:.1f}%",
                 fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(out_dir / "summary_table.png", dpi=150)
    plt.close(fig)


def _pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, (int, float)) and v == 0.0:
        return "ref"
    return f"{float(v)*100:.2f}%"


if __name__ == "__main__":
    plot_all(sys.argv[1] if len(sys.argv) > 1 else "outputs/benchmark_grid")

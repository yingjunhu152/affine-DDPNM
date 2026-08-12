#!/usr/bin/env python3
"""Plots for the 1-D velocity-perturbation sensitivity study.

Reads ``sensitivity_report.json`` (written by ``run_1d_sensitivity.py``):
- panel A: dR vs eps for the coherent / structure / mixed families with the
  theory lines dR = 1.25*PVI*theta*eps and the 3-D empirical points
  (hollow diamond = throughput 1.25*dPVI, filled = measured, bar = leak);
- panel B: exact 3-D decomposition dR = 1.25*dPVI - leak per method;
- panel C: field L2 error vs eps for the pure-structure family (structure
  perturbs fields, not recovery).

Run: conda run -n fenicsx --no-capture-output python plot_1d_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

SURFACE, INK, SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
METHOD_COLORS = {
    "Classic-DDPNM-1": "#2a78d6",
    "NormalLinear-DDPNM-3": "#eb6834",
    "Affine-DDPNM-9": "#1baf7a",
}
PVI_REF = 0.23


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path,
        default=PROJECT_DIR / "outputs" / "sensitivity_1d" / "sensitivity_report.json",
    )
    parser.add_argument(
        "--out", type=Path,
        default=PROJECT_DIR / "outputs" / "sensitivity_1d" / "sensitivity_plot.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.data.read_text(encoding="utf-8"))
    rows = report["rows"]
    agg = report["aggregates"]

    def group(family: str, eps: float) -> list[dict]:
        return [r for r in rows if r["family"] == family and abs(r["eps"] - eps) < 1e-12]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=SECONDARY, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")

    # ---- panel A: sensitivity curves --------------------------------------
    ax = axes[0]
    for eps in (0.065, 0.6532):
        g = agg[f"structure_eps{eps}"]
        ax.errorbar(eps, g["dR_mean"], yerr=2 * g["dR_std"], fmt="o", color=MUTED,
                    markersize=6, capsize=3, label="structure (θ=0)" if eps == 0.065 else None)
    for eps in (0.0651, 0.3075, 0.6532):
        g = agg[f"mixed_eps{eps}"]
        ax.errorbar(eps, g["dR_mean"], yerr=2 * g["dR_std"], fmt="s", color=SECONDARY,
                    markersize=6, capsize=3, label="mixed (θ=0.45)" if eps == 0.0651 else None)
    for eps in (0.03, 0.065, 0.15, 0.3, 0.65):
        g = group("coherent", eps)[0]
        ax.plot(eps, g["dR"], "^", color=INK, markersize=7,
                label="coherent (θ=1)" if eps == 0.03 else None)
    ax.axhline(0.0, color="#c3c2b7", linewidth=1.0)
    eps_line = np.linspace(0.0, 0.66, 60)
    for theta, ls in ((0.45, "--"), (1.0, ":")):
        ax.plot(eps_line, 1.25 * PVI_REF * theta * eps_line, ls, color=MUTED, linewidth=1.4)
    ax.text(0.56, 1.25 * PVI_REF * 0.45 * 0.56 + 0.005, "1.25·PVI·θ·ε, θ=0.45",
            color=SECONDARY, fontsize=8)
    ax.text(0.56, 1.25 * PVI_REF * 1.0 * 0.56 - 0.018, "θ=1", color=SECONDARY, fontsize=8)
    for method, seed in report["empirical_3d"].items():
        color = METHOD_COLORS[method]
        eps, dR_tp, dR_meas = seed["eps"], seed["dR_throughput_125dPVI"], seed["dR_measured"]
        ax.plot(eps, dR_tp, "D", mfc="none", mec=color, markersize=8, zorder=5)
        ax.plot(eps, dR_meas, "D", color=color, markersize=8, zorder=5)
        ax.plot([eps, eps], [dR_meas, dR_tp], color=color, linewidth=1.2, zorder=4)
    ax.set_xlabel("velocity rel L2 error ε", color=SECONDARY)
    ax.set_ylabel("recovery deviation ΔR", color=SECONDARY)
    ax.set_title("1-D sensitivity (lines) vs 3-D points (diamonds)", color=INK, fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)

    # ---- panel B: 3-D exact decomposition ----------------------------------
    ax = axes[1]
    empirical = report["empirical_3d"]
    for i, method in enumerate(empirical):
        seed = empirical[method]
        tp, leak, net = seed["dR_throughput_125dPVI"], seed["leak"], seed["dR_measured"]
        ax.bar(i, tp, width=0.55, color="#2a78d6", label="1.25·ΔPVI (throughput)" if i == 0 else None)
        ax.bar(i, -leak, width=0.55, bottom=tp, color="#e34948",
               label="−leak (early water out)" if i == 0 else None)
        ax.plot(i, net, "o", color=INK, markersize=7, label="ΔR measured" if i == 0 else None)
        ax.text(i, 0.003, f"θ={seed['theta']:.2f}", color=SECONDARY, fontsize=8,
                ha="center")
    ax.axhline(0.0, color="#c3c2b7", linewidth=1.0)
    ax.set_xticks(range(len(empirical)))
    ax.set_xticklabels([m.replace("-DDPNM", "").replace("NormalLinear", "W1n") for m in empirical],
                       rotation=15, fontsize=8)
    ax.set_ylabel("recovery deviation", color=SECONDARY)
    ax.set_title("3-D: ΔR = 1.25·ΔPVI − leak (exact)", color=INK, fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)

    # ---- panel C: structure moves fields, not recovery ---------------------
    ax = axes[2]
    for eps in (0.065, 0.6532):
        g = group("structure", eps)
        g_mean = float(np.mean([r["field_rel_l2"] for r in g]))
        g_std = float(np.std([r["field_rel_l2"] for r in g]))
        ax.errorbar(eps, g_mean, yerr=2 * g_std, fmt="o",
                    color="#eb6834", markersize=7, capsize=3,
                    label="field rel L2" if eps == 0.065 else None)
    ax.plot([0.05, 0.68], [0.075, 0.46], ":", color=MUTED, linewidth=1.2)
    ax.text(0.30, 0.36, "field L2 ≈ 1.5·ε", color=SECONDARY, fontsize=8)
    for eps in (0.065, 0.6532):
        g = agg[f"structure_eps{eps}"]
        ax.errorbar(eps, g["dR_mean"], yerr=2 * g["dR_std"], fmt="s", color="#1baf7a",
                    markersize=7, capsize=3, label="ΔR (should be 0)" if eps == 0.065 else None)
    ax.axhline(0.0, color="#c3c2b7", linewidth=1.0)
    ax.set_xlabel("ε (structure family, δ(0)=0)", color=SECONDARY)
    ax.set_ylabel("error", color=SECONDARY)
    ax.set_title("Structure perturbs fields, not recovery", color=INK, fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Done: {args.out.resolve()}")


if __name__ == "__main__":
    main()

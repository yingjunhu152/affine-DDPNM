"""
Paper figure: Classic vs Affine DDPNM across three 3D geometries.
Panel (a) Velocity L2 error.  Panel (b) On-line speedup vs FEM (log scale).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np

# ── Colour scheme ──
C_BLUE   = "#2a78d6"
C_ORANGE = "#eb6834"
C_GRAY   = "#898781"
C_DARK   = "#0b0b0b"
C_GRID   = "#e1e0d9"
C_REF    = "#c3c2b7"
C_GREEN  = "#0ca30c"
C_RED    = "#d03b3b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": C_REF,
    "axes.linewidth": 0.6,
    "grid.color": C_GRID,
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
    "xtick.color": C_GRAY,
    "ytick.color": C_GRAY,
    "text.color": C_DARK,
    "axes.labelcolor": C_DARK,
})

# ── Data ──
geometries = ["Uniform\nspheres", "Random\nspheres", "100-sphere\ndense"]

classic_l2 = [21.08, 65.32, 95.09]
affine_l2  = [ 5.63,  6.51, 15.72]

classic_off = [11.22, 12.20, 67.7]
classic_on  = [ 0.006,  0.012,  0.227]
affine_off  = [13.66, 15.09, 89.2]
affine_on   = [ 0.198,  0.139, 13.83]
fem_time    = [81.45, 116.62, 23.8]

classic_total      = [a+b for a,b in zip(classic_off, classic_on)]
affine_total       = [a+b for a,b in zip(affine_off,  affine_on)]
classic_online_spd = [fem_time[i]/classic_on[i] for i in range(3)]
affine_online_spd  = [fem_time[i]/affine_on[i]  for i in range(3)]

n = len(geometries)
x = np.arange(n)
bar_w  = 0.30
gap    = 0.05
bar_w2 = 0.34

# ── Figure ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
fig.subplots_adjust(left=0.06, right=0.98, top=0.87, bottom=0.20, wspace=0.34)

# ══════════════════════════════════════════
# Panel (a): Velocity L² error
# ══════════════════════════════════════════
ax1.bar(x - bar_w/2 - gap/2, classic_l2, bar_w,
        color=C_BLUE, edgecolor="none", zorder=3)
ax1.bar(x + bar_w/2 + gap/2, affine_l2,  bar_w,
        color=C_ORANGE, edgecolor="none", zorder=3)

for i in range(n):
    ax1.text(x[i] - bar_w/2 - gap/2, classic_l2[i] + 2.0,
             f"{classic_l2[i]:.1f}%", ha="center", va="bottom", fontsize=7.0,
             color=C_GRAY, fontweight="bold")
    ax1.text(x[i] + bar_w/2 + gap/2, affine_l2[i] + 2.0,
             f"{affine_l2[i]:.1f}%", ha="center", va="bottom", fontsize=7.0,
             color=C_GRAY, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(geometries, fontsize=8, color=C_DARK)
ax1.set_ylabel("Velocity $L^2$ error (%)", fontsize=8.5, color=C_DARK)
ax1.set_ylim(0, 118)
ax1.yaxis.set_major_locator(mticker.MultipleLocator(20))
ax1.grid(axis="y", zorder=0)
ax1.set_title("(a)  Velocity $L^2$ error", fontsize=10.5, fontweight="bold",
              color=C_DARK, pad=8)

reductions = ["-73%", "-90%", "-83%"]
for i in range(n):
    mid_y = (classic_l2[i] + affine_l2[i]) / 2
    ax1.annotate(reductions[i],
                xy=(x[i], mid_y), fontsize=8, fontweight="bold",
                color=C_GREEN, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_GRID,
                          linewidth=0.6, alpha=0.88))

# ══════════════════════════════════════════
# Panel (b): On-line speedup vs FEM (log scale)
# ══════════════════════════════════════════

ax2.bar(x - bar_w2/2 - gap/2, classic_online_spd, bar_w2,
        color=C_BLUE, edgecolor="none", zorder=3,
        label="Classic-DDPNM  on-line")
ax2.bar(x + bar_w2/2 + gap/2, affine_online_spd,  bar_w2,
        color=C_ORANGE, edgecolor="none", zorder=3,
        label="Affine-DDPNM   on-line")

for i in range(n):
    cv, av = classic_online_spd[i], affine_online_spd[i]
    ax2.text(x[i] - bar_w2/2 - gap/2, cv * 1.25,
             f"{cv:.0f}x" if cv >= 100 else f"{cv:.1f}x",
             ha="center", va="bottom", fontsize=7.2, color=C_BLUE, fontweight="bold")
    ax2.text(x[i] + bar_w2/2 + gap/2, av * 1.25,
             f"{av:.0f}x" if av >= 10 else f"{av:.1f}x",
             ha="center", va="bottom", fontsize=7.2, color=C_ORANGE, fontweight="bold")

# FEM baseline
ax2.axhline(y=1.0, color=C_REF, linewidth=0.8, linestyle="--", zorder=0)
ax2.text(n - 0.6, 1.5, "FEM baseline", fontsize=6.8, color=C_GRAY, ha="right", va="bottom")

# shade slower-than-FEM zone
ax2.axhspan(0.18, 1.0, xmin=0.60, xmax=0.97, facecolor=C_RED, alpha=0.06, zorder=0, linewidth=0)
ax2.text(2.35, 0.42, "slower\nthan FEM", fontsize=6.5, color=C_RED, ha="center", va="center",
         style="italic")

ax2.set_yscale("log")
ax2.set_xticks(x)
ax2.set_xticklabels(geometries, fontsize=8, color=C_DARK)
ax2.set_ylabel("On-line speedup vs FEM  (log scale)", fontsize=8.5, color=C_DARK)
ax2.set_ylim(0.25, 50000)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}x" if v >= 1 else f"{v:.2f}x"))
ax2.grid(axis="y", zorder=0, which="major")
ax2.grid(axis="y", zorder=0, which="minor", linewidth=0.3, color=C_GRID, alpha=0.5)
ax2.set_title("(b)  On-line speedup vs monolithic FEM",
              fontsize=10.5, fontweight="bold", color=C_DARK, pad=8)

# Inset table: raw online times
inset = (
    "On-line solve time:\n"
    f"Classic    6 ms   12 ms   0.23 s\n"
    f"Affine     0.20 s  0.14 s  13.8 s\n"
    f"FEM         81 s    117 s   23.8 s"
)
ax2.text(0.02, 0.98, inset, transform=ax2.transAxes,
         fontsize=5.8, va="top", ha="left", color=C_GRAY,
         family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_GRID, linewidth=0.5, alpha=0.85))

# ── Shared legend ──
legend_elements = [
    Patch(facecolor=C_BLUE,   label="Classic-DDPNM  on-line"),
    Patch(facecolor=C_ORANGE, label="Affine-DDPNM   on-line"),
]
leg = fig.legend(handles=legend_elements, loc="upper center", ncol=2,
                 frameon=True, fontsize=8.2, handlelength=1.4, handleheight=1.2,
                 columnspacing=2.0, bbox_to_anchor=(0.5, 0.99))
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_edgecolor(C_GRID)

# ── Footer ──
fig.text(0.5, 0.01,
         "Same mesh for all methods.  Error: 6th-order quadrature, per-element, no interface averaging.  "
         "On-line speedup = t_FEM / t_on-line (Schur assembly + solve + reconstruction; offline library pre-built).  "
         "Green labels: L2 reduction Classic -> Affine.",
         ha="center", va="bottom", fontsize=6.4, color=C_GRAY, style="italic")

fig.text(0.5, 0.055,
         "Uniform: 12,203 cells, 144 ifaces  |  Random: 15,249 cells, 114 ifaces  |  100-sphere: 45,191 cells, 537 ifaces",
         ha="center", va="bottom", fontsize=6.6, color=C_GRAY)

# ── Save ──
out_dir = "d:/hu/tongjiproj/20260727/"
fig.savefig(out_dir + "fig_classic_vs_affine.pdf", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out_dir + "fig_classic_vs_affine.png", dpi=250, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved fig_classic_vs_affine.pdf and .png")
plt.close(fig)

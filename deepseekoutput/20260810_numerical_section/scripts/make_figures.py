#!/usr/bin/env python3
"""Generate the figures for the numerical-experiments paper section.

All figures are produced from the archived benchmark outputs
(outputs/benchmark_w1n of the three projects) and the computed
geometry statistics.  Run under the econ env (matplotlib):
    /d/Miniconda3/envs/econ/python.exe make_figures.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SECTION = Path(__file__).resolve().parent.parent
FIG = SECTION / "figures"
DATA = SECTION / "data"
FIG.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# validated categorical palette (dataviz reference instance, light mode)
# ----------------------------------------------------------------------------
C_BLUE = "#2a78d6"     # slot 1  -> Classic (W0n)
C_ORANGE = "#eb6834"   # slot 2  -> NormalLinear (W1n)
C_AQUA = "#1baf7a"     # slot 3  -> Affine (W1v)
C_VIOLET = "#4a3aa7"   # slot 7  (reserved, not used in series)
C_GRAY = "#52514e"     # neutral ink -> reference (FEM)
C_GRID = "#e1e0d9"
C_INK = "#0b0b0b"
C_MUTED = "#898781"
C_W1N_PURPLE = "#7B1FA2"  # colour used in the archived W1n experiment plot

METHOD_COLORS = {
    "W0n": C_BLUE,
    "W1n": C_ORANGE,
    "W1v": C_AQUA,
}
METHOD_LABELS = {
    "W0n": r"$W_{0n}$ (Classic)",
    "W1n": r"$W_{1n}$ (NormalLinear)",
    "W1v": r"$W_{1v}$ (Affine)",
}
GEOM_LABELS = {
    "Uniform-27": r"Uniform-27",
    "Random-27": r"Random-27",
    "Real-100": r"Real-100",
}
GEOM_MARKERS = {"Uniform-27": "o", "Random-27": "^", "Real-100": "s"}


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
def load_metrics() -> dict:
    """Return {geometry: {method: metrics}} with method keys W0n/W1n/W1v."""
    cols = {
        "l2": "velocity_relative_L2",
        "h1": "velocity_relative_broken_H1",
        "p": "pressure_relative_L2",
        "q": "outlet_flux_relative_error",
        "dofs": "global_unknowns",
        "modes": "modes_per_interface",
        "offline": "offline_seconds",
        "online": "online_seconds",
        "first": "first_solve_seconds",
        "speedup": "speedup_vs_fem",
        "mem": None,
    }
    out = {}
    files = {
        "Uniform-27": (
            REPO / "affine_ddpnm_3d/outputs/benchmark_w1n/affine_ddpnm_metrics.csv",
            {"velocity_relative_L2": "velocity_relative_L2"},
        ),
        "Random-27": (
            REPO / "affine_ddpnm_3d_random_porous/outputs/benchmark_w1n/random_affine_metrics.csv",
            {"velocity_relative_L2": "velocity_relative_L2"},
        ),
        "Real-100": (
            REPO / "real_porous_benchmark_3d/outputs/benchmark_w1n/benchmark_metrics.csv",
            {
                "velocity_relative_L2": "velocity_relative_l2",
                "velocity_relative_broken_H1": "velocity_relative_broken_h1",
                "pressure_relative_L2": "pressure_relative_l2",
                "outlet_flux_relative_error": "outlet_flux_relative_error",
                "offline_seconds": "offline_s",
                "online_seconds": "online_s",
                "first_solve_seconds": "total_s",
                "mem": "mem_mib",
            },
        ),
    }
    for geom, (path, rename) in files.items():
        with open(path, encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        header = [h.strip() for h in lines[0].split(",")]
        rows = [dict(zip(header, (v.strip() for v in l.split(",")))) for l in lines[1:]]
        method_map = {
            "Classic-DDPNM-1": "W0n",
            "Classic-DDPNM": "W0n",
            "NormalLinear-DDPNM-3": "W1n",
            "Affine-DDPNM-9": "W1v",
            "Affine-DDPNM": "W1v",
        }
        out[geom] = {}
        for row in rows:
            name = row["method"].strip()
            if name in method_map:
                key = method_map[name]
                m = {"name": name}
                for k, col in cols.items():
                    col = rename.get(col, col)
                    m[k] = float(row[col]) if col and row.get(col) not in (None, "") else None
                if m["modes"] is None:  # real CSV omits the modes column
                    m["modes"] = {"W0n": 1, "W1n": 3, "W1v": 9}[key]
                out[geom][key] = m
    return out


def load_random_report() -> dict:
    p = REPO / "affine_ddpnm_3d_random_porous/outputs/benchmark_w1n/random_affine_report.json"
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load_geometry_stats() -> dict:
    with open(DATA / "geometry_stats.json", encoding="utf-8") as fh:
        return {g["name"]: g for g in json.load(fh)["geometries"]}


def _extract_array(source: Path, var: str) -> np.ndarray:
    """Parse `VAR = np.asarray([...], dtype=...)` literal from a python source."""
    text = source.read_text(encoding="utf-8")
    m = re.search(rf"{var}\s*=\s*np\.asarray\(\s*(\[.*?\])\s*[,)]", text, re.S)
    if not m:
        raise RuntimeError(f"{var} not found in {source}")
    body = m.group(1).strip()
    return np.asarray(ast.literal_eval(body), dtype=float)


def load_spheres() -> dict:
    uniform_src = REPO / "ddpnm_3d_uniform_spheres/ddpnm3d/geometry.py"
    random_src = REPO / "affine_ddpnm_3d_random_porous/random_porous.py"
    real_src = REPO / "real_porous_benchmark_3d/geometry.py"
    text = uniform_src.read_text(encoding="utf-8")
    radius = float(re.search(r"SPHERE_RADIUS\s*=\s*([0-9.]+)", text).group(1))
    grid = _extract_array(uniform_src, "SPHERE_GRID")
    centers = np.asarray(
        [(a, b, c) for a in grid for b in grid for c in grid], dtype=float
    )
    rand = _extract_array(random_src, "SPHERES")
    real = _extract_array(real_src, "SPHERES")
    return {
        "Uniform-27": {"c": centers, "r": np.full(len(centers), radius)},
        "Random-27": {"c": rand[:, :3], "r": rand[:, 3]},
        "Real-100": {"c": real[:, :3], "r": real[:, 3]},
    }


METRICS = load_metrics()
GEOM_STATS = load_geometry_stats()
SPHERES = load_spheres()
RANDOM_REPORT = load_random_report()


# ----------------------------------------------------------------------------
# shared styling
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": C_GRID,
    "axes.linewidth": 0.8,
    "axes.labelcolor": C_INK,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "text.color": C_INK,
    "axes.titlecolor": C_INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
})


def geom_order():
    return ["Uniform-27", "Random-27", "Real-100"]


def method_order():
    return ["W0n", "W1n", "W1v"]


def log10ticks(ax):
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciBase(labelOnlyBase=False))


# ----------------------------------------------------------------------------
# Figure 1: geometry schematics (solid spheres)
# ----------------------------------------------------------------------------
def fig1_geometry():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6),
                             subplot_kw={"projection": "3d"})
    for ax, geom in zip(axes, geom_order()):
        sp = SPHERES[geom]
        ax.set_box_aspect((1, 1, 1))
        # unit cube wireframe
        for x0, x1 in [(0, 1)]:
            for yy in (0, 1):
                for zz in (0, 1):
                    ax.plot([x0, x1], [yy, yy], [zz, zz], color=C_GRID, lw=0.5)
            for xx in (0, 1):
                for zz in (0, 1):
                    ax.plot([xx, xx], [x0, x1], [zz, zz], color=C_GRID, lw=0.5)
                for yy in (0, 1):
                    ax.plot([xx, xx], [yy, yy], [x0, x1], color=C_GRID, lw=0.5)
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 12)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        for c, r in zip(sp["c"], sp["r"]):
            ax.plot_surface(
                c[0] + r * xs, c[1] + r * ys, c[2] + r * zs,
                color=C_BLUE, alpha=0.28, linewidth=0, rstride=1, cstride=1,
                shade=True,
            )
        ax.set_title(GEOM_LABELS[geom], pad=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor(C_GRID)
        ax.view_init(elev=22, azim=-55)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_geometry.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: accuracy vs modal budget (1 -> 3 -> 9 modes per interface)
# ----------------------------------------------------------------------------
def fig2_modal_budget():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, geom in zip(axes, geom_order()):
        g = METRICS[geom]
        xs = [g[k]["modes"] for k in method_order()]
        ys = [g[k]["l2"] for k in method_order()]
        ax.semilogy(xs, ys, "--", color=C_MUTED, lw=0.9, zorder=1)
        for x, y, k in zip(xs, ys, method_order()):
            ax.plot(x, y, "o", ms=8, mfc=METHOD_COLORS[k], mec="white", mew=0.9,
                    zorder=3)
            ax.annotate(
                METHOD_LABELS[k].split(" (")[0],
                (x, y), textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=7.5, color=C_INK,
            )
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 9])
        ax.set_xticklabels(["1", "3", "9"])
        ax.set_xlim(0.7, 13)
        ax.set_ylim(4e-2, 2.0)
        ax.set_title(GEOM_LABELS[geom])
        ax.set_xlabel(r"modes per interface $r$")
        if ax is axes[0]:
            ax.set_ylabel(r"relative $L^2$ error of $\mathbf{u}$")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:.0%}"))
    axes[0].annotate(r"$W_{0n}\to W_{1n}\to W_{1v}$",
                     xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8,
                     color=C_MUTED, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_modal_budget.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: four error metrics as dot plots across geometries (log y)
# ----------------------------------------------------------------------------
def fig3_metrics():
    metrics = [
        ("l2", r"(a) velocity, $L^2$"),
        ("h1", r"(b) velocity, broken $H^1$"),
        ("p", r"(c) pressure, $L^2$"),
        ("q", r"(d) outlet flux"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.2), sharey=False)
    geoms = geom_order()
    x = np.arange(len(geoms))
    for ax, (key, label) in zip(axes, metrics):
        for j, k in enumerate(method_order()):
            vals = [METRICS[g][k][key] for g in geoms]
            ax.plot(x + 0.16 * (j - 1), vals, "o", ms=7,
                    mfc=METHOD_COLORS[k], mec="white", mew=0.9,
                    label=METHOD_LABELS[k])
        ax.set_xticks(x)
        ax.set_xticklabels([GEOM_LABELS[g] for g in geoms], fontsize=8)
        ax.set_yscale("log")
        ax.set_title(label, fontsize=9)
        ax.set_ylim(2e-3, 4.0)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, p: f"{v:.0%}" if v >= 0.01 else f"{v:.0e}")
        )
    axes[0].set_ylabel("relative error")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=METHOD_COLORS[k],
                          mec="white", mew=0.9, label=METHOD_LABELS[k])
               for k in method_order()]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
               fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 4: FEM-relative observed error-reduction shares (100% stacked)
# ----------------------------------------------------------------------------
def fig4_shares():
    geoms = geom_order()
    normal_share, vector_share = [], []
    for g in geoms:
        e0n, e1n, e1v = (METRICS[g][k]["l2"] for k in method_order())
        denom = e0n - e1v
        normal_share.append((e0n - e1n) / denom)
        vector_share.append((e1n - e1v) / denom)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    y = np.arange(len(geoms))[::-1]
    ax.barh(y, normal_share, height=0.55, color=C_BLUE,
            label=r"normal-spatial step $W_{0n}\to W_{1n}$")
    ax.barh(y, vector_share, left=normal_share, height=0.55, color=C_AQUA,
            label=r"additional vectorial step $W_{1n}\to W_{1v}$")
    for i, (a, b) in enumerate(zip(normal_share, vector_share)):
        ax.text(a / 2, y[i], f"{a:.0%}", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        ax.text(a + b / 2, y[i], f"{b:.0%}", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([GEOM_LABELS[g] for g in geoms])
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"share of observed FEM-relative $L^2(\mathbf{u})$ error reduction")
    ax.grid(axis="x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1, fontsize=8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_shares.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 5: accuracy vs online cost (Pareto)
# ----------------------------------------------------------------------------
def fig5_pareto():
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    fem_online = {"Uniform-27": 77.251, "Random-27": 118.419, "Real-100": 14.525}
    for geom in geom_order():
        for k in method_order():
            m = METRICS[geom][k]
            ax.loglog(m["online"], m["l2"], GEOM_MARKERS[geom], ms=7,
                      mfc=METHOD_COLORS[k], mec="white", mew=0.8, zorder=3)
    for geom in geom_order():
        ax.loglog(fem_online[geom], 1.0, GEOM_MARKERS[geom], ms=9,
                  mfc="none", mec=C_GRAY, mew=1.4, zorder=4)
        ax.annotate(GEOM_LABELS[geom],
                    (fem_online[geom] * 1.25, 0.75), fontsize=8, color=C_MUTED)
    # method legend
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=METHOD_COLORS[k],
                          mec="white", mew=0.8, label=METHOD_LABELS[k].replace(" (", " ("))
               for k in method_order()]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec=C_GRAY,
                              mew=1.4, label="monolithic FEM (reference)"))
    geom_handles = [
        plt.Line2D([], [], marker=GEOM_MARKERS[g], ls="", color=C_MUTED,
                   label=GEOM_LABELS[g])
        for g in geom_order()
    ]
    ax.legend(handles=handles + [plt.Line2D([], [], ls="", label="")] + geom_handles,
              loc="upper right", fontsize=8, ncol=1)
    ax.set_xlabel("online solve time (s)")
    ax.set_ylabel(r"relative $L^2$ error of $\mathbf{u}$")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:.0%}"))
    ax.set_xlim(4e-3, 4e3)
    ax.set_ylim(4e-2, 2.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_pareto.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 6: Random-27 slice velocity fields and errors
# ----------------------------------------------------------------------------
def fig6_fields():
    npz_path = REPO / "affine_ddpnm_3d_random_porous/outputs/benchmark_w1n/random_benchmark_fields.npz"
    d = np.load(npz_path)
    sl = {
        "points": d["classic_error_slice_points"],
        "tri": d["classic_error_slice_triangles"],
    }
    xy = sl["points"][:, :2]
    tri = sl["tri"]
    u_fem = d["classic_error_slice_u_fem"]
    mag_fem = np.linalg.norm(u_fem, axis=1)
    panels = [
        ("reference", r"$|\mathbf{u}_{\rm FEM}|$", mag_fem, None),
        ("W0n", r"$|\mathbf{u}_{W_{0n}}-\mathbf{u}_{\rm FEM}|$",
         np.linalg.norm(d["classic_error_slice_u_ddpnm"] - u_fem, axis=1), "err"),
        ("W1n", r"$|\mathbf{u}_{W_{1n}}-\mathbf{u}_{\rm FEM}|$",
         np.linalg.norm(d["normal_linear_error_slice_u_ddpnm"] - u_fem, axis=1), "err"),
        ("W1v", r"$|\mathbf{u}_{W_{1v}}-\mathbf{u}_{\rm FEM}|$",
         np.linalg.norm(d["affine_error_slice_u_ddpnm"] - u_fem, axis=1), "err"),
    ]
    err_max = max(p[2].max() for p in panels if p[3] == "err")
    ref_max = mag_fem.max()
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4), sharey=True)
    for ax, (key, title, vals, kind) in zip(axes, panels):
        if kind == "err":
            vmax = err_max
            cmap = "Blues"
        else:
            vmax = ref_max
            cmap = "viridis"
        ax.tripcolor(xy[:, 0], xy[:, 1], tri, vals, cmap=cmap,
                     vmin=0.0, vmax=vmax, shading="gouraud")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    axes[0].set_ylabel("slice plane $z=0.5$ (unit cube)")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_fields.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig1_geometry()
    fig2_modal_budget()
    fig3_metrics()
    fig4_shares()
    fig5_pareto()
    fig6_fields()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()

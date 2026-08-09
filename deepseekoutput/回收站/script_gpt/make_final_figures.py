#!/usr/bin/env python3
"""Figures for the final numerical section (missing / replacement set).

Produces the figures specified in FIELD_DATA_TODO_20260809.md from the
existing benchmark outputs.  Figures whose data is not (yet) archived are
detected at run time, reported in the printed summary, and skipped - the
functions stay active and fire automatically as soon as the data appears
(e.g. W0v metrics in the CSVs, Random-27 interface flux keys, Real-100
W1n slice fields, multi-mesh / multi-seed / Schur-decomposition archives).

Outputs (figures/)  - status column: OK produced, SKIP missing data:
    fig2_modal_budget_fourspace.png        OK  (W0v pending -> 3-point path)
    fig3_metrics_fourspace.png             OK  (W0v pending -> 3 methods)
    fig4_equal_budget_W0v_vs_W1n.png       SKIP (no W0v run exists)
    fig5_fields_uniform27.png              OK
    fig5_fields_random27.png               OK
    fig5_fields_uniform_random.png         OK  (paper-ready 2x4 comparison)
    fig5_fields_real100.png                SKIP (current W1n benchmark saves no slice fields)
    fig5_interface_flux_modes.png          OK  (Uniform-27 row only)
    fig5_interface_flux_errors.png         OK  (for each archived interface-flux row)
    fig6_schur_energy_decomposition.png    SKIP (requires full broken-pressure C-DD)
    fig7_accuracy_cost_standardized.png    OK  (single timing runs; no spread)
    fig8_mesh_sensitivity.png              SKIP (no multi-mesh archive)
    fig9_random_ensemble.png               SKIP (no multi-seed archive)
    fig10_observed_shares_clean.png        OK
    fig1b_geometry_partition_interfaces.png OK

Run under the econ env:
    /d/Miniconda3/envs/econ/python.exe make_final_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from figlib import (
    BENCH,
    C_AQUA,
    C_BLUE,
    C_GRAY,
    C_GRID,
    C_INK,
    C_MUTED,
    C_ORANGE,
    C_VIOLET,
    FIG_DIR,
    GEOM_LABELS,
    GEOM_ORDER,
    GRID_PLANES,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    NESTED_PATHS,
    draw_cube,
    draw_spheres,
    load_fields,
    load_metrics,
    load_random_partition,
    load_spheres,
    present_methods,
    sci_colorbar,
    set_equal_cube,
    slice_field_grids,
    slice_sphere_cuts,
)

METRICS = load_metrics()
SPHERES = load_spheres()

STATUS = {}

# Timing scope in the currently archived benchmark files.  Keep this mapping
# synchronized with future re-runs.  The accuracy-cost figure prints the scope
# explicitly so that solve-only and assembly+solve timings are never silently
# compared as if they were identical measurements.
FEM_TIMING_SCOPE = {
    "Uniform-27": "assembly + solve",
    "Random-27": "assembly + solve",
    "Real-100": "solve only",
}


def mark(name: str, ok: bool, note: str = "") -> None:
    STATUS[name] = (ok, note)
    print(("[ OK ] " if ok else "[SKIP] ") + name + (f"  ({note})" if note else ""))


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------
def pct_format(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:.0%}"))


def panel_letter(ax, letter, x=0.5, y=-0.09):
    ax.text(
        x, y, f"({letter})",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=12, fontfamily="serif",
    )


def nested_lines(ax, x, g, methods, *, ls="-", lw=1.1, alpha=0.9, color=C_MUTED):
    """Connect nested method points only (never W0v <-> W1n)."""
    for path in NESTED_PATHS:
        pts = [(x[k], g[k]["l2"]) for k in path if k in g]
        if len(pts) == len(path):
            ax.plot(*zip(*pts), ls=ls, lw=lw, alpha=alpha, color=color, zorder=1)


def modal_x_positions(g: dict) -> dict[str, float]:
    """Display positions for modal-budget plots.

    W0v and W1n both have exactly r=3.  When both are present, they are given
    a very small visual jitter around the r=3 tick so that the equal-budget
    comparison remains legible.  The tick itself stays at the true value 3.
    """
    xs = {k: float(g[k]["modes"]) for k in g if k in METHOD_LABELS}
    if "W0v" in xs and "W1n" in xs:
        xs["W0v"] = 2.82
        xs["W1n"] = 3.18
    return xs


# ----------------------------------------------------------------------------
# fig2: modal budget, FEM-relative velocity L2 vs modes/interface r
# ----------------------------------------------------------------------------
def fig2_modal_budget_fourspace():
    name = "fig2_modal_budget_fourspace.png"
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, geom in zip(axes, GEOM_ORDER):
        g = METRICS[geom]
        xs = modal_x_positions(g)
        # legal nested paths only; W0v is drawn when present (r=3, dashed path)
        nested_lines(ax, xs, g, ["W0n", "W1n", "W1v"])
        if "W0v" in xs:
            nested_lines(ax, xs, g, ["W0n", "W0v", "W1v"], ls="--", lw=0.9)
        for k, x in xs.items():
            y = g[k]["l2"]
            ax.plot(x, y, "o", ms=8, mfc=METHOD_COLORS[k], mec="white", mew=0.9, zorder=3)
            ax.annotate(
                METHOD_LABELS[k], (x, y), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=7.5, color=C_INK,
            )
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 9])
        ax.set_xticklabels(["1", "3", "9"])
        ax.set_xlim(0.7, 13)
        ax.set_title(GEOM_LABELS[geom])
        ax.set_xlabel(r"modes per interface $r$")
        if ax is axes[0]:
            ax.set_ylabel(r"relative $L^2$ error of $\mathbf{u}$")
            pct_format(ax)
    if all("W0v" in METRICS[g] for g in GEOM_ORDER):
        axes[0].annotate(
            r"$W_{0v}$ and $W_{1n}$ both have $r=3$ (slightly jittered for visibility)",
            xy=(0.02, 0.03), xycoords="axes fraction", fontsize=7.2,
            color=C_MUTED, ha="left",
        )
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    mark(name, True)


# ----------------------------------------------------------------------------
# fig3: four error metrics, all methods, log y
# ----------------------------------------------------------------------------
def fig3_metrics_fourspace():
    name = "fig3_metrics_fourspace.png"
    metrics = [
        ("l2", r"(a) velocity, $L^2$"),
        ("h1", r"(b) velocity, broken $H^1$"),
        ("p", r"(c) pressure, $L^2$"),
        ("q", r"(d) outlet flux"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.3))
    x = np.arange(len(GEOM_ORDER))
    for ax, (key, label) in zip(axes, metrics):
        for j, geom in enumerate(GEOM_ORDER):
            g = METRICS[geom]
            ks = present_methods(METRICS, geom)
            offsets = 0.16 * (np.arange(len(ks)) - (len(ks) - 1) / 2.0)
            xpos = {k: x[j] + offsets[ks.index(k)] for k in ks}
            # nested chain per geometry (W0v never joined to W1n)
            for path in NESTED_PATHS:
                pts = [(xpos[k], g[k][key])
                       for k in path if k in ks]
                if len(pts) == len(path):
                    ax.plot(*zip(*pts), ls=":", lw=1.0, color=C_MUTED, zorder=1)
            for k in ks:
                ax.plot(xpos[k], g[k][key], "o", ms=7,
                        mfc=METHOD_COLORS[k], mec="white", mew=0.9, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([GEOM_LABELS[g] for g in GEOM_ORDER], fontsize=8)
        ax.set_yscale("log")
        ax.set_title(label, fontsize=9)
        vals = np.asarray([
            METRICS[g][k][key]
            for g in GEOM_ORDER
            for k in present_methods(METRICS, g)
        ], dtype=float)
        ax.set_ylim(0.25 * vals.min(), 4.0 * vals.max())
        # consistent percent ticks on every panel (3 significant digits)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, p: f"{v * 100:.3g}%")
        )
    axes[0].set_ylabel("relative error")
    available = [
        k for k in METHOD_ORDER if any(k in METRICS[g] for g in GEOM_ORDER)
    ]
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc=METHOD_COLORS[k], mec="white",
                   mew=0.9, label=METHOD_LABELS[k])
        for k in available
    ]
    fig.legend(handles=handles, loc="lower center", ncol=max(1, len(handles)),
               bbox_to_anchor=(0.5, -0.01), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    mark(name, True)


# ----------------------------------------------------------------------------
# fig4: equal-budget W0v vs W1n at r = 3 (needs a W0v run)
# ----------------------------------------------------------------------------
def fig4_equal_budget_W0v_vs_W1n():
    name = "fig4_equal_budget_W0v_vs_W1n.png"
    if not all("W0v" in METRICS[g] for g in GEOM_ORDER):
        mark(name, False, "no W0v (constant-vector) run exists in any benchmark output")
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, geom in zip(axes, GEOM_ORDER):
        g = METRICS[geom]
        e0n, e0v, e1n = g["W0n"]["l2"], g["W0v"]["l2"], g["W1n"]["l2"]
        w, h = 0.34, 0.62
        ax.bar([0], [e0v / e0n], width=w, color=C_VIOLET, label=METHOD_LABELS["W0v"])
        ax.bar([1], [e1n / e0n], width=w, color=C_ORANGE, label=METHOD_LABELS["W1n"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([METHOD_LABELS["W0v"], METHOD_LABELS["W1n"]], fontsize=8)
        ax.set_title(GEOM_LABELS[geom])
        ax.set_xlim(-0.6, 1.6)
    axes[0].set_ylabel(r"velocity $L^2$ error / $e_{0n}$")
    pct_format(axes[0])
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    mark(name, True)


# ----------------------------------------------------------------------------
# fig5: slice field panels per geometry (shared error color scale)
# ----------------------------------------------------------------------------
CMAP_SPEED = plt.matplotlib.colormaps["viridis"]
CMAP_ERR = plt.matplotlib.colormaps["magma"]
CMAP_FLUX = mpl.colors.LinearSegmentedColormap.from_list(
    "div_flux", ["#e34948", "#f5b3b1", "#f0efec", "#a9c8f2", "#2a78d6"]
)

ERR_METHODS = ["W0n", "W1n", "W1v"]


def _draw_field_panel(ax, grid, vmin, vmax, cmap, boundaries, spheres, radii, z):
    artist = ax.pcolormesh(
        grid["X"], grid["Y"], grid["values"], cmap=cmap,
        vmin=vmin, vmax=vmax, shading="auto"
    )
    ax.add_collection(
        plt.matplotlib.collections.LineCollection(
            boundaries, colors="#1f1f1f", linewidths=1.3, zorder=5
        )
    )
    slice_sphere_cuts(ax, spheres, radii, z, color="#3a3a3a")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=8)
    return artist


def uniform_slice_labels(slice_points, edges=(0.0, 0.2, 0.5, 0.8, 1.0)):
    """Cartesian grid cell index (0..63) of each slice vertex, from its
    coordinates (the Uniform-27 partition is the 4x4x4 cuboid grid)."""
    edges = np.asarray(edges)
    idx = np.clip(np.searchsorted(edges, slice_points, side="right") - 1, 0, 3)
    return idx[:, 0] * 16 + idx[:, 1] * 4 + idx[:, 2]


def _prepare_slice_grid(geom: str):
    """Return the smoothed, geometry-aware slice-grid payload for one case."""
    data = load_fields(geom)
    if data is None:
        return None
    if geom == "Uniform-27":
        sp = SPHERES[geom]
        return slice_field_grids(
            data,
            spheres=sp["c"], radii=sp["r"],
            vertex_labels=uniform_slice_labels(data["classic_error_slice_points"]),
        )
    return slice_field_grids(data)


def fig5_fields(geom: str):
    fname = {
        "Uniform-27": "fig5_fields_uniform27.png",
        "Random-27": "fig5_fields_random27.png",
        "Real-100": "fig5_fields_real100.png",
    }[geom]
    grid = _prepare_slice_grid(geom)
    if grid is None:
        mark(fname, False,
             "current benchmark archives no slice fields (Real-100: z_slice API bug)")
        return
    gr = grid["grids"]
    # One honest shared scale for all three error panels.  Use the full finite
    # maximum (not a hidden percentile clipping) so visual differences have a
    # direct quantitative meaning.
    err_max = max(float(np.nanmax(gr[f"err_{k}"])) for k in ERR_METHODS)

    fig = plt.figure(figsize=(14.2, 3.8))
    gs = fig.add_gridspec(
        1, 6, width_ratios=[1, 1, 1, 1, 0.055, 0.055],
        wspace=0.28,
    )
    axes = [fig.add_subplot(gs[0, c]) for c in range(4)]
    ref_cax, err_cax = fig.add_subplot(gs[0, 4]), fig.add_subplot(gs[0, 5])
    panels = [
        ("speed_ref", CMAP_SPEED, 0.0, grid["speed_max"], r"FEM: $|\mathbf{u}|$"),
        ("err_W0n", CMAP_ERR, 0.0, err_max, r"$W_{0n}$: $|\mathbf{u}-\mathbf{u}_{\rm FEM}|$"),
        ("err_W1n", CMAP_ERR, 0.0, err_max, r"$W_{1n}$: $|\mathbf{u}-\mathbf{u}_{\rm FEM}|$"),
        ("err_W1v", CMAP_ERR, 0.0, err_max, r"$W_{1v}$: $|\mathbf{u}-\mathbf{u}_{\rm FEM}|$"),
    ]
    for col, (key, cmap, vmin, vmax, ttl) in enumerate(panels):
        ax = axes[col]
        _draw_field_panel(
            ax, {**grid, "values": gr[key]}, vmin, vmax, cmap,
            grid["boundaries"], grid["spheres"], grid["radii"], grid["z"],
        )
        ax.set_title(ttl, fontsize=11)
        ax.tick_params(labelsize=7.5)
        if col == 0:
            ax.set_ylabel("$y$", fontsize=9)
        ax.set_xlabel("$x$", fontsize=9)
        panel_letter(ax, chr(ord("a") + col), y=-0.13)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, grid["speed_max"]),
                              cmap=CMAP_SPEED), cax=ref_cax)
    cb.set_label(r"$|\mathbf{u}_{\mathrm{FEM}}|$", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, err_max), cmap=CMAP_ERR),
        cax=err_cax)
    cb.set_label(r"$|\mathbf{u}-\mathbf{u}_{\mathrm{FEM}}|$", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    fig.savefig(FIG_DIR / fname, dpi=240, bbox_inches="tight")
    plt.close(fig)
    mark(fname, True)


def fig5_fields_uniform_random():
    """Paper-ready 2x4 comparison: regular versus irregular geometry."""
    name = "fig5_fields_uniform_random.png"
    geoms = ["Uniform-27", "Random-27"]
    prepared = {g: _prepare_slice_grid(g) for g in geoms}
    if any(prepared[g] is None for g in geoms):
        mark(name, False, "Uniform-27 and Random-27 slice archives are both required")
        return

    fig = plt.figure(figsize=(14.2, 7.0))
    gs = fig.add_gridspec(
        2, 6, width_ratios=[1, 1, 1, 1, 0.055, 0.055],
        wspace=0.28, hspace=0.30,
    )
    titles = [
        r"FEM: $|\mathbf{u}|$",
        r"$W_{0n}$: $|\mathbf{u}-\mathbf{u}_{\rm FEM}|$",
        r"$W_{1n}$: $|\mathbf{u}-\mathbf{u}_{\rm FEM}|$",
        r"$W_{1v}$: $|\mathbf{u}-\mathbf{u}_{\rm FEM}|$",
    ]
    letter = ord("a")
    for row, geom in enumerate(geoms):
        grid = prepared[geom]
        gr = grid["grids"]
        err_max = max(float(np.nanmax(gr[f"err_{k}"])) for k in ERR_METHODS)
        axes = [fig.add_subplot(gs[row, c]) for c in range(4)]
        ref_cax = fig.add_subplot(gs[row, 4])
        err_cax = fig.add_subplot(gs[row, 5])
        keys = ["speed_ref", "err_W0n", "err_W1n", "err_W1v"]
        for col, (ax, key) in enumerate(zip(axes, keys)):
            if key == "speed_ref":
                cmap, vmin, vmax = CMAP_SPEED, 0.0, grid["speed_max"]
            else:
                cmap, vmin, vmax = CMAP_ERR, 0.0, err_max
            _draw_field_panel(
                ax, {**grid, "values": gr[key]}, vmin, vmax, cmap,
                grid["boundaries"], grid["spheres"], grid["radii"], grid["z"],
            )
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            ax.set_xlabel("$x$", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(f"{geom}\n$y$", fontsize=8.5)
            panel_letter(ax, chr(letter), y=-0.14)
            letter += 1

        cb = fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(0.0, grid["speed_max"]), cmap=CMAP_SPEED
            ),
            cax=ref_cax,
        )
        cb.set_label(r"$|\mathbf{u}_{\mathrm{FEM}}|$", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(0.0, err_max), cmap=CMAP_ERR
            ),
            cax=err_cax,
        )
        cb.set_label(r"$|\mathbf{u}-\mathbf{u}_{\mathrm{FEM}}|$", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.savefig(FIG_DIR / name, dpi=240, bbox_inches="tight")
    plt.close(fig)
    mark(name, True)


def fig5_fields_all():
    fig5_fields("Uniform-27")
    fig5_fields("Random-27")
    fig5_fields("Real-100")
    fig5_fields_uniform_random()


# ----------------------------------------------------------------------------
# fig5b: normal flux q = u . n on representative interfaces (local s,t)
# ----------------------------------------------------------------------------
def fig5_interface_flux_modes():
    name = "fig5_interface_flux_modes.png"
    rows = {}
    pending = []
    for geom in ("Uniform-27", "Random-27"):
        data = load_fields(geom)
        if data is not None and "interface_q_fem" in data.files:
            rows[geom] = data
        else:
            pending.append(geom)
    if not rows:
        mark(name, False, "no geometry archives per-interface fluxes")
        return
    nrows = len(rows)
    note = f"rows available: {', '.join(rows)}" + (
        f"; pending: {', '.join(pending)} (interface export not archived)" if pending else "")
    print(f"  fig5_interface_flux_modes: {note}")
    fig = plt.figure(figsize=(13.8, 3.6 * nrows + 0.2))
    gs = fig.add_gridspec(nrows, 5, width_ratios=[1, 1, 1, 1, 0.07],
                          wspace=0.24, hspace=0.34)
    titles = [
        r"FEM: $q=\mathbf{u}\cdot\mathbf{n}$",
        r"$W_{0n}$: $q$",
        r"$W_{1n}$: $q$",
        r"$W_{1v}$: $q$",
    ]
    for row, (geom, d) in enumerate(rows.items()):
        st = d["interface_st"]
        tri = d["interface_triangles"]
        qs = [d["interface_q_fem"], d["interface_q_classic"],
              d["interface_q_normal_linear"], d["interface_q_affine"]]
        q_max = max(float(np.max(np.abs(q))) for q in qs)
        for col in range(4):
            ax = fig.add_subplot(gs[row, col])
            ax.tripcolor(st[:, 0], st[:, 1], tri, qs[col], cmap=CMAP_FLUX,
                         vmin=-q_max, vmax=q_max, shading="gouraud")
            ax.set_title(titles[col] if row == 0 else titles[col], fontsize=8.5)
            ax.set_aspect("equal")
            s0, s1 = float(st[:, 0].min()), float(st[:, 0].max())
            t0, t1 = float(st[:, 1].min()), float(st[:, 1].max())
            ax.set_xlim(s0, s1)
            ax.set_ylim(t0, t1)
            ax.set_xticks([s0, 0.5 * (s0 + s1), s1])
            ax.set_yticks([t0, 0.5 * (t0 + t1), t1])
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
            ax.tick_params(labelsize=7.5, colors=C_MUTED)
            ax.grid(False)
            if col == 0:
                ax.set_ylabel(f"{geom}\n$t$", fontsize=8.5)
            ax.set_xlabel("$s$", fontsize=8.5)
            panel_letter(ax, chr(ord("a") + row * 4 + col), y=-0.14)
        cax = fig.add_subplot(gs[row, 4])
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-q_max, q_max),
                                  cmap=CMAP_FLUX), cax=cax)
        cb.set_label(r"$q=\mathbf{u}\cdot\mathbf{n}$", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.savefig(FIG_DIR / name, dpi=240, bbox_inches="tight")
    plt.close(fig)
    mark(name, True)


def fig5_interface_flux_errors():
    """Absolute normal-flux errors on the same representative interfaces.

    This companion figure is deliberately error-focused because the four raw
    q-fields can look visually similar even when their interface errors differ
    materially.  Each geometry row uses one common error scale for W0n/W1n/W1v.
    """
    name = "fig5_interface_flux_errors.png"
    rows = {}
    pending = []
    for geom in ("Uniform-27", "Random-27"):
        data = load_fields(geom)
        if data is not None and "interface_q_fem" in data.files:
            rows[geom] = data
        else:
            pending.append(geom)
    if not rows:
        mark(name, False, "no geometry archives per-interface fluxes")
        return

    nrows = len(rows)
    fig = plt.figure(figsize=(10.6, 3.55 * nrows + 0.15))
    gs = fig.add_gridspec(nrows, 4, width_ratios=[1, 1, 1, 0.07],
                          wspace=0.25, hspace=0.34)
    titles = [
        r"$W_{0n}$: $|q-q_{\rm FEM}|$",
        r"$W_{1n}$: $|q-q_{\rm FEM}|$",
        r"$W_{1v}$: $|q-q_{\rm FEM}|$",
    ]
    letter = ord("a")
    for row, (geom, d) in enumerate(rows.items()):
        st = d["interface_st"]
        tri = d["interface_triangles"]
        q_ref = d["interface_q_fem"]
        errs = [
            np.abs(d["interface_q_classic"] - q_ref),
            np.abs(d["interface_q_normal_linear"] - q_ref),
            np.abs(d["interface_q_affine"] - q_ref),
        ]
        emax = max(float(np.max(e)) for e in errs)
        s0, s1 = float(st[:, 0].min()), float(st[:, 0].max())
        t0, t1 = float(st[:, 1].min()), float(st[:, 1].max())
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            ax.tripcolor(st[:, 0], st[:, 1], tri, errs[col], cmap=CMAP_ERR,
                         vmin=0.0, vmax=emax, shading="gouraud")
            ax.set_title(titles[col], fontsize=8.8)
            ax.set_aspect("equal")
            ax.set_xlim(s0, s1)
            ax.set_ylim(t0, t1)
            ax.set_xticks([s0, 0.5 * (s0 + s1), s1])
            ax.set_yticks([t0, 0.5 * (t0 + t1), t1])
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
            ax.tick_params(labelsize=7.3, colors=C_MUTED)
            ax.grid(False)
            ax.set_xlabel("$s$", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(f"{geom}\n$t$", fontsize=8.5)
            panel_letter(ax, chr(letter), y=-0.14)
            letter += 1
        cax = fig.add_subplot(gs[row, 3])
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(0.0, emax), cmap=CMAP_ERR
            ),
            cax=cax,
        )
        cb.set_label(r"$|q-q_{\rm FEM}|$", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.savefig(FIG_DIR / name, dpi=240, bbox_inches="tight")
    plt.close(fig)
    note = f"rows available: {', '.join(rows)}"
    if pending:
        note += f"; pending: {', '.join(pending)}"
    mark(name, True, note)


# ----------------------------------------------------------------------------
# fig6: Schur energy decomposition (requires full broken-pressure C-DD)
# ----------------------------------------------------------------------------
def fig6_schur_energy_decomposition():
    name = "fig6_schur_energy_decomposition.png"
    mark(name, False,
         "requires the full broken-pressure C-DD (Delta_N / Delta_{T|N} / "
         "affine residual) which no benchmark archives")


# ----------------------------------------------------------------------------
# fig7: accuracy vs cost, one panel per geometry (FEM = vertical reference)
# ----------------------------------------------------------------------------
def fig7_accuracy_cost_standardized():
    name = "fig7_accuracy_cost_standardized.png"
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, geom in zip(axes, GEOM_ORDER):
        g = METRICS[geom]
        fem_first = g["__fem__"]["first"]
        for k in present_methods(METRICS, geom):
            m = g[k]
            ax.loglog(m["online"], m["l2"], "o", ms=7, mfc=METHOD_COLORS[k],
                      mec="white", mew=0.8, zorder=3)
            ax.loglog(m["first"], m["l2"], "o", ms=7, mfc="white",
                      mec=METHOD_COLORS[k], mew=1.3, zorder=3)
            ax.plot([m["online"], m["first"]], [m["l2"], m["l2"]],
                    ls=":", color=METHOD_COLORS[k], lw=0.9, zorder=2)
        if fem_first:
            ax.axvline(fem_first, ls="--", color=C_GRAY, lw=1.2, zorder=4)
        ax.text(
            0.98, 0.97, f"FEM timing: {FEM_TIMING_SCOPE[geom]}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.8, color=C_MUTED,
        )
        ax.set_xlim(4e-3, 2.5e2)
        ax.set_title(GEOM_LABELS[geom])
        ax.set_xlabel("recorded wall time (s)")
        if ax is axes[0]:
            ax.set_ylabel(r"relative $L^2$ error of $\mathbf{u}$")
            pct_format(ax)
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc=METHOD_COLORS[k], mec="white",
                   mew=0.8, label=METHOD_LABELS[k])
        for k in ("W0n", "W1n", "W1v")
    ]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec=C_GRAY,
                              mew=1.3, label="first solve (incl. offline)"))
    handles.append(plt.Line2D([], [], ls="--", color=C_GRAY,
                              label="monolithic FEM timing (reference error = 0, off log axis)"))
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.03), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    mark(name, True, "single timing runs archived (no median/spread)")


# ----------------------------------------------------------------------------
# fig8 / fig9: mesh sensitivity / random ensemble (no archives)
# ----------------------------------------------------------------------------
def fig8_mesh_sensitivity():
    mark("fig8_mesh_sensitivity.png", False,
         "requires >=3 mesh levels for Uniform-27, none archived")


def fig9_random_ensemble():
    mark("fig9_random_ensemble.png", False,
         "requires several frozen Random-27 seeds, only seed 20260804 archived")


# ----------------------------------------------------------------------------
# fig10: observed error-reduction shares (100% stacked, legend outside)
# ----------------------------------------------------------------------------
def fig10_observed_shares_clean():
    name = "fig10_observed_shares_clean.png"
    geoms = GEOM_ORDER
    shares = []
    for g in geoms:
        e0n, e1n, e1v = (METRICS[g][k]["l2"] for k in ("W0n", "W1n", "W1v"))
        denom = e0n - e1v
        shares.append(((e0n - e1n) / denom, (e1n - e1v) / denom))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    y = np.arange(len(geoms))[::-1]
    normal = [s[0] for s in shares]
    vector = [s[1] for s in shares]
    ax.barh(y, normal, height=0.5, color=C_BLUE,
            label=r"normal-spatial step $W_{0n}\to W_{1n}$")
    ax.barh(y, vector, left=normal, height=0.5, color=C_AQUA,
            label=r"additional vectorial step $W_{1n}\to W_{1v}$")
    for i, (a, b) in enumerate(zip(normal, vector)):
        ax.text(a / 2, y[i], f"{a:.0%}", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        ax.text(a + b / 2, y[i], f"{b:.0%}", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([GEOM_LABELS[g] for g in geoms])
    ax.set_xlim(0, 1)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=7.6)
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.18, right=0.98)
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    mark(name, True)


# ----------------------------------------------------------------------------
# fig1b: solid geometry + partition/interface network per geometry
# ----------------------------------------------------------------------------
COLORS_24 = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377",
    "#BBBBBB", "#EE8866", "#44AA99", "#999933", "#882255", "#117733",
]


def _grid_interface_lines(planes: np.ndarray):
    """Segments of the finite-area internal interfaces of the 4x4x4 grid
    partition: the grid lines lying on the interior planes x/y/z in `planes`."""
    segs = []
    for axis in range(3):
        for v in planes:
            lo, hi = (0.0, 1.0), (0.0, 1.0)
            for u in planes:
                segs.append([_pt(axis, v, u, 0.0), _pt(axis, v, u, 1.0)])
                segs.append([_pt(axis, v, 0.0, u), _pt(axis, v, 1.0, u)])
    return np.asarray(segs)


def _pt(axis, v, a, b):
    p = [0.0, 0.0, 0.0]
    p[axis] = v
    rest = [a, b]
    p[(axis + 1) % 3] = rest[0]
    p[(axis + 2) % 3] = rest[1]
    return p


def _random_partition_lines(part):
    """Subdomain edges colored per label + interface network (pairs, centers)."""
    labels = np.asarray(part["cell_labels"], dtype=int)
    collections = []
    for label in np.unique(labels):
        cells = part["tetrahedra"][labels == label]
        edges = cells[:, np.asarray([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])]
        edges = edges.reshape(-1, 2)
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
        collections.append(
            (part["points"][edges], COLORS_24[label % len(COLORS_24)], 0.15, 0.30)
        )
    pair_lines = part["balls"][part["pairs"]]  # (N,2,3)
    return collections, pair_lines


def _grid_adjacency_graph(planes: np.ndarray):
    """Representative pore centers and adjacency edges for a Cartesian cut."""
    edges = np.concatenate(([0.0], np.asarray(planes, dtype=float), [1.0]))
    mids = 0.5 * (edges[:-1] + edges[1:])
    centers = np.asarray(
        [(mids[i], mids[j], mids[k]) for i in range(4) for j in range(4) for k in range(4)],
        dtype=float,
    )

    def idx(i, j, k):
        return i * 16 + j * 4 + k

    segs = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = centers[idx(i, j, k)]
                if i < 3:
                    segs.append([a, centers[idx(i + 1, j, k)]])
                if j < 3:
                    segs.append([a, centers[idx(i, j + 1, k)]])
                if k < 3:
                    segs.append([a, centers[idx(i, j, k + 1)]])
    return centers, np.asarray(segs)


def fig1b_geometry_partition():
    name = "fig1b_geometry_partition_interfaces.png"
    fig = plt.figure(figsize=(12.6, 7.4))
    axes = [[fig.add_subplot(2, 3, r * 3 + c + 1, projection="3d")
             for c in range(3)] for r in range(2)]
    rand_part = load_random_partition()
    rand_pairs = rand_part["balls"][rand_part["pairs"]]
    for c, geom in enumerate(GEOM_ORDER):
        sp = SPHERES[geom]
        ax = axes[0][c]
        draw_spheres(ax, sp["c"], sp["r"], color=C_BLUE, alpha=0.30, linewidth=0.0)
        draw_cube(ax, color=C_GRID, linewidth=0.5, alpha=0.8)
        set_equal_cube(ax)
        ax.set_title(GEOM_LABELS[geom], fontsize=10.5, pad=0)
        ax.text2D(0.5, -0.10, f"({chr(ord('a') + c)})", transform=ax.transAxes,
                  ha="center", va="top", fontsize=12, fontfamily="serif")
        ax = axes[1][c]
        draw_spheres(ax, sp["c"], sp["r"], color=C_GRAY, alpha=0.10, linewidth=0.0)
        if geom in GRID_PLANES:
            nodes, segs = _grid_adjacency_graph(GRID_PLANES[geom])
            ax.add_collection3d(
                Line3DCollection(segs, colors="#30343b",
                                 linewidths=0.75, alpha=0.88)
            )
            ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=4.5,
                       color="#30343b", alpha=0.65, depthshade=False)
        else:  # Random-27: same graph-only visual convention; no colored FE mesh
            ax.add_collection3d(
                Line3DCollection(rand_pairs, colors="#30343b",
                                 linewidths=0.75, alpha=0.88)
            )
            ax.scatter(rand_part["balls"][:, 0], rand_part["balls"][:, 1],
                       rand_part["balls"][:, 2], s=4.5, color="#30343b",
                       alpha=0.65, depthshade=False)
        draw_cube(ax, color=C_GRID, linewidth=0.5, alpha=0.8)
        set_equal_cube(ax)
        ax.text2D(0.5, -0.10, f"({chr(ord('d') + c)})", transform=ax.transAxes,
                  ha="center", va="top", fontsize=12, fontfamily="serif")
    fig.text(0.012, 0.76, "Solid sphere packing",
             rotation=90, va="center", ha="center", fontsize=11, fontstyle="italic")
    fig.text(0.012, 0.26, "Pore adjacency / interface graph",
             rotation=90, va="center", ha="center", fontsize=11, fontstyle="italic")
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    mark(name, True)


# ----------------------------------------------------------------------------
def main() -> None:
    fig2_modal_budget_fourspace()
    fig3_metrics_fourspace()
    fig4_equal_budget_W0v_vs_W1n()
    fig5_fields_all()
    fig5_interface_flux_modes()
    fig5_interface_flux_errors()
    fig6_schur_energy_decomposition()
    fig7_accuracy_cost_standardized()
    fig8_mesh_sensitivity()
    fig9_random_ensemble()
    fig10_observed_shares_clean()
    fig1b_geometry_partition()
    ok = sum(1 for _, (s, _) in STATUS.items() if s)
    print(f"\n{ok}/{len(STATUS)} figures written to {FIG_DIR}")
    missing = [k for k, (s, _) in STATUS.items() if not s]
    if missing:
        print("missing data:", ", ".join(missing))


if __name__ == "__main__":
    main()

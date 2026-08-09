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
    fig5_fields_real100.png                SKIP (current W1n benchmark saves no slice fields)
    fig5_interface_flux_modes.png          OK  (Uniform-27 row only)
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


# ----------------------------------------------------------------------------
# fig2: modal budget, FEM-relative velocity L2 vs modes/interface r
# ----------------------------------------------------------------------------
def fig2_modal_budget_fourspace():
    name = "fig2_modal_budget_fourspace.png"
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, geom in zip(axes, GEOM_ORDER):
        g = METRICS[geom]
        xs = {k: g[k]["modes"] for k in present_methods(METRICS, geom)}
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
    axes[0].annotate(
        r"$W_{0n}\to W_{1n}\to W_{1v}$ (nested)$\;\cdot\; W_{0v}$ at $r=3$ pending",
        xy=(0.02, 0.03), xycoords="axes fraction", fontsize=7.5,
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
            xx = x[j] + 0.16 * (np.arange(len(ks)) - 1)
            yy = [g[k][key] for k in ks]
            # nested chain per geometry (W0v never joined to W1n)
            for path in NESTED_PATHS:
                pts = [(x[j] + 0.16 * (ks.index(k) - 1), g[k][key])
                       for k in path if k in ks]
                if len(pts) == len(path):
                    ax.plot(*zip(*pts), ls=":", lw=1.0, color=C_MUTED, zorder=1)
            for k in ks:
                ax.plot(x[j] + 0.16 * (ks.index(k) - 1), g[k][key], "o", ms=7,
                        mfc=METHOD_COLORS[k], mec="white", mew=0.9, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([GEOM_LABELS[g] for g in GEOM_ORDER], fontsize=8)
        ax.set_yscale("log")
        ax.set_title(label, fontsize=9)
        vals = np.asarray([[METRICS[g][k][key] for k in present_methods(METRICS, g)]
                           for g in GEOM_ORDER]).ravel()
        ax.set_ylim(0.25 * vals.min(), 4.0 * vals.max())
        # consistent percent ticks on every panel (3 significant digits)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, p: f"{v * 100:.3g}%")
        )
    axes[0].set_ylabel("relative error")
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc=METHOD_COLORS[k], mec="white",
                   mew=0.9, label=METHOD_LABELS[k])
        for k in ("W0n", "W1n", "W1v")
    ]
    handles.append(plt.Line2D([], [], ls=":", color=C_MUTED, label="nested path"))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)
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
CMAP_ERR = plt.matplotlib.colormaps["turbo"]
CMAP_FLUX = mpl.colors.LinearSegmentedColormap.from_list(
    "div_flux", ["#e34948", "#f5b3b1", "#f0efec", "#a9c8f2", "#2a78d6"]
)

ERR_METHODS = ["W0n", "W1n", "W1v"]


def _draw_field_panel(ax, grid, vmin, vmax, cmap, boundaries, spheres, radii, z):
    artist = ax.pcolormesh(grid["X"], grid["Y"], grid["values"], cmap=cmap,
                           vmin=vmin, vmax=vmax)
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


def uniform_interface_trace_lines(centers, radii, z=0.5,
                                  planes=(0.2, 0.5, 0.8), n=400):
    """Analytic grid-plane traces of the Uniform-27 partition on the slice.

    The slice triangulation of this geometry is not conforming to the grid
    partition (see slice_field_grids), so the interface traces are drawn
    analytically as the lines x/y = 0.2, 0.5, 0.8, clipped to the fluid
    domain (portions inside sphere cross-sections removed).
    """
    xs = np.linspace(0.0, 1.0, n)
    segs = []
    for v in planes:
        for pts in (np.column_stack((np.full(n, v), xs)),
                    np.column_stack((xs, np.full(n, v)))):
            keep = np.ones(n, dtype=bool)
            for c, r in zip(centers, radii, strict=True):
                dz = abs(float(c[2]) - z)
                if dz >= float(r):
                    continue
                rho2 = float(r) ** 2 - dz ** 2
                d2 = (pts[:, 0] - c[0]) ** 2 + (pts[:, 1] - c[1]) ** 2
                keep &= d2 >= rho2
            idx = np.flatnonzero(keep)
            if len(idx) == 0:
                continue
            starts = np.r_[0, np.flatnonzero(np.diff(idx) > 1) + 1]
            for s in range(len(starts)):
                run = idx[starts[s]:(starts[s + 1] if s + 1 < len(starts) else len(idx))]
                if len(run) >= 3:
                    segs.append([pts[run[0]], pts[run[-1]]])
    return segs


def uniform_slice_tri_labels(slice_points, slice_triangles,
                             edges=(0.0, 0.2, 0.5, 0.8, 1.0)):
    """Cartesian grid cell index (0..63) of each slice triangle, from its
    centroid (the Uniform-27 partition is the 4x4x4 cuboid grid).

    Triangle-level labels (rather than per-vertex ones) keep the interface
    vertices inside both adjacent subdomains' interpolation vertex sets, so
    no white NaN band appears along the grid-plane traces.
    """
    edges = np.asarray(edges)
    cents = slice_points[slice_triangles].mean(axis=1)
    idx = np.clip(np.searchsorted(edges, cents, side="right") - 1, 0, 3)
    return idx[:, 0] * 16 + idx[:, 1] * 4 + idx[:, 2]


def fig5_fields(geom: str):
    fname = {
        "Uniform-27": "fig5_fields_uniform27.png",
        "Random-27": "fig5_fields_random27.png",
        "Real-100": "fig5_fields_real100.png",
    }[geom]
    data = load_fields(geom)
    if data is None:
        mark(fname, False,
             "current benchmark archives no slice fields (Real-100: z_slice API bug)")
        return
    if geom == "Uniform-27":
        sp = SPHERES[geom]
        grid = slice_field_grids(
            data,
            spheres=sp["c"], radii=sp["r"],
            global_interp=True,
            boundary_lines=uniform_interface_trace_lines(sp["c"], sp["r"], z=0.5),
        )
    else:  # Random-27: exact per-triangle parent cell from the archive
        tri_labels = np.asarray(data["cell_labels"])[
            data["classic_error_slice_parent_cells"][data["classic_error_slice_triangles"][:, 0]]
        ]
        grid = slice_field_grids(data, tri_labels=tri_labels)
    gr = grid["grids"]
    # One shared error scale, anchored on the largest-error method so every
    # panel reads: vmax = p90 of the W0n error field (its median then sits at
    # ~30% of the scale; a p98 vmax pushes W0n's bulk below ~10% and makes
    # the panel read as empty).
    err_w0n = gr["err_W0n"][~np.isnan(gr["err_W0n"])]
    err_max = float(np.percentile(err_w0n, 90.0))

    fig = plt.figure(figsize=(14.2, 3.8))
    # spacer column (5) keeps the two colorbar columns apart so their tick
    # numbers never collide
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 0.06, 0.14, 0.06],
                          wspace=0.26)
    axes = [fig.add_subplot(gs[0, c]) for c in range(4)]
    ref_cax, err_cax = fig.add_subplot(gs[0, 4]), fig.add_subplot(gs[0, 6])
    panels = [
        ("speed_ref", CMAP_SPEED, 0.0, grid["speed_max"], r"FEM: $|\mathbf{u}|$"),
        ("err_W0n", CMAP_ERR, 0.0, err_max, r"$W_{0n}$"),
        ("err_W1n", CMAP_ERR, 0.0, err_max, r"$W_{1n}$"),
        ("err_W1v", CMAP_ERR, 0.0, err_max, r"$W_{1v}$"),
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


def fig5_fields_all():
    fig5_fields("Uniform-27")
    fig5_fields("Random-27")
    fig5_fields("Real-100")


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
    # Two rows per geometry: q = u.n maps (shared diverging scale) and the
    # FEM-relative flux errors |q - q_FEM| (shared sequential scale).
    fig = plt.figure(figsize=(13.8, 6.6 * nrows + 0.3))
    gs = fig.add_gridspec(2 * nrows, 5,
                          width_ratios=[1, 1, 1, 1, 0.07],
                          wspace=0.26, hspace=0.45)
    for row, (geom, d) in enumerate(rows.items()):
        st = d["interface_st"]
        tri = d["interface_triangles"]
        q_fem = d["interface_q_fem"]
        qs = [q_fem, d["interface_q_classic"],
              d["interface_q_normal_linear"], d["interface_q_affine"]]
        q_max = max(float(np.max(np.abs(q))) for q in qs)
        errs = [np.abs(q - q_fem) for q in qs[1:]]
        err_max = float(np.percentile(np.concatenate(errs), 90.0))
        s0, s1 = float(st[:, 0].min()), float(st[:, 0].max())
        t0, t1 = float(st[:, 1].min()), float(st[:, 1].max())
        ticks_s = np.linspace(s0, s1, 5)
        ticks_t = np.linspace(t0, t1, 5)
        for col in range(4):
            r0 = 2 * row
            ax = fig.add_subplot(gs[r0, col])
            ax.tripcolor(st[:, 0], st[:, 1], tri, qs[col], cmap=CMAP_FLUX,
                         vmin=-q_max, vmax=q_max, shading="gouraud")
            ax.set_title(
                [r"FEM: $q=\mathbf{u}\cdot\mathbf{n}$",
                 r"$W_{0n}$: $q$", r"$W_{1n}$: $q$", r"$W_{1v}$: $q$"][col],
                fontsize=8.5)
            ax.set_aspect("equal")
            ax.set_xlim(s0, s1)
            ax.set_ylim(t0, t1)
            ax.set_xticks(ticks_s)
            ax.set_yticks(ticks_t)
            ax.tick_params(labelsize=7, colors=C_MUTED)
            ax.grid(False)
            if col == 0:
                ax.set_ylabel(f"{geom}\n$t$", fontsize=8.5)
            panel_letter(ax, chr(ord("a") + col), y=-0.16)
            if col == 0:
                ax.set_xlabel("$s$", fontsize=8.5)
        # error row: three FEM-relative flux-error panels, shared scale
        for col, (label, err) in enumerate(zip(("W0n", "W1n", "W1v"), errs, strict=True)):
            ax2 = fig.add_subplot(gs[2 * row + 1, col])
            ax2.tripcolor(st[:, 0], st[:, 1], tri, err, cmap=CMAP_ERR,
                          vmin=0.0, vmax=err_max, shading="gouraud")
            ax2.set_title(rf"${label}$: $|q-q_{{\mathrm{{FEM}}}}|$", fontsize=8.5)
            ax2.set_aspect("equal")
            ax2.set_xlim(s0, s1)
            ax2.set_ylim(t0, t1)
            ax2.set_xticks(ticks_s)
            ax2.set_yticks(ticks_t)
            ax2.tick_params(labelsize=7, colors=C_MUTED)
            ax2.grid(False)
            if col == 0:
                ax2.set_ylabel("$t$", fontsize=8.5)
            ax2.set_xlabel("$s$", fontsize=8.5)
            panel_letter(ax2, chr(ord("e") + col), y=-0.16)
        # colorbars: row 0 diverging q, row 1 sequential error
        cax = fig.add_subplot(gs[2 * row, 4])
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-q_max, q_max),
                                  cmap=CMAP_FLUX), cax=cax)
        cb.set_label(r"$q=\mathbf{u}\cdot\mathbf{n}$", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cax2 = fig.add_subplot(gs[2 * row + 1, 4])
        cb2 = fig.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, err_max),
                                  cmap=CMAP_ERR), cax=cax2)
        cb2.set_label(r"$|q-q_{\mathrm{FEM}}|$", fontsize=8)
        cb2.ax.tick_params(labelsize=7)
    fig.savefig(FIG_DIR / name, dpi=240, bbox_inches="tight")
    plt.close(fig)
    mark(name, True)


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
            ax.annotate(
                "FEM reference\nerror = 0 (off log axis)",
                (fem_first, 1.35), ha="center", fontsize=7, color=C_GRAY,
                xycoords=("data", "axes fraction"),
            )
        ax.set_xlim(4e-3, 2.5e2)
        ax.set_title(GEOM_LABELS[geom])
        ax.set_xlabel("solve time (s)")
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
                              label="monolithic FEM (error = 0)"))
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.text(
        0.5, -0.10,
        "FEM reference time: Uniform-27 / Random-27 = monolithic solve "
        "(online); Real-100 = total_s incl. assembly (different solver path)",
        ha="center", fontsize=7, color=C_MUTED,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    mark(name, True,
         "single timing runs; FEM timing scope differs across geometries (see note)")


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
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.30)
    y = np.arange(len(geoms))[::-1]
    normal = [s[0] for s in shares]
    vector = [s[1] for s in shares]
    ax.barh(y, normal, height=0.5, color=C_BLUE,
            label=r"normal step $W_{0n}\to W_{1n}$")
    ax.barh(y, vector, left=normal, height=0.5, color=C_AQUA,
            label=r"vectorial step $W_{1n}\to W_{1v}$")
    for i, (a, b) in enumerate(zip(normal, vector)):
        ax.text(a / 2, y[i], f"{a:.0%}", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        ax.text(a + b / 2, y[i], f"{b:.0%}", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([GEOM_LABELS[g] for g in geoms])
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"share of observed FEM-relative $L^2(\mathbf{u})$ error reduction",
                  labelpad=2)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.40), ncol=2, fontsize=8)
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    mark(name, True)


# ----------------------------------------------------------------------------
# fig1b: solid geometry + partition/interface network per geometry
# ----------------------------------------------------------------------------
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


def random_interface_edges(part):
    """Edges of the finite-area internal interfaces of the Random-27
    partition: mesh faces shared by two subdomains (drawn in the same
    wireframe style as the grid partitions of the other geometries)."""
    tets = part["tetrahedra"]
    labels = np.asarray(part["cell_labels"], dtype=int)
    face_labels = {}
    for t, lab in zip(tets, labels):
        for face in ((t[0], t[1], t[2]), (t[0], t[1], t[3]),
                     (t[0], t[2], t[3]), (t[1], t[2], t[3])):
            key = tuple(sorted(int(v) for v in face))
            face_labels.setdefault(key, set()).add(int(lab))
    segs = []
    for key, labs in face_labels.items():
        if len(labs) == 2:
            v = part["points"][list(key)]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                segs.append([v[a], v[b]])
    return np.asarray(segs)


def fig1b_geometry_partition():
    name = "fig1b_geometry_partition_interfaces.png"
    fig = plt.figure(figsize=(12.6, 7.4))
    axes = [[fig.add_subplot(2, 3, r * 3 + c + 1, projection="3d")
             for c in range(3)] for r in range(2)]
    rand_part = load_random_partition()
    rand_iface_edges = random_interface_edges(rand_part)
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
            segs = _grid_interface_lines(GRID_PLANES[geom])
        else:  # Random-27: interface triangles of the actual partition mesh
            segs = rand_iface_edges
        ax.add_collection3d(
            Line3DCollection(segs, colors="#30343b", linewidths=0.8, alpha=0.9)
        )
        draw_cube(ax, color=C_GRID, linewidth=0.5, alpha=0.8)
        set_equal_cube(ax)
        ax.text2D(0.5, -0.10, f"({chr(ord('d') + c)})", transform=ax.transAxes,
                  ha="center", va="top", fontsize=12, fontfamily="serif")
    fig.text(0.012, 0.76, "Solid sphere packing",
             rotation=90, va="center", ha="center", fontsize=11, fontstyle="italic")
    fig.text(0.012, 0.26, "Finite-area internal interfaces",
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

#!/usr/bin/env python3
"""Slice-plane error fields for the two-phase Buckley--Leverett run.

Same archived style as the tracer's ``slice_error_fields.png`` (ported from
``affine_ddpnm_3d_random_porous/plot_random_errors.py``): the tetra mesh is
cut at ``z = 0.5``, the P1 vertex fields are evaluated on the slice,
resampled on a regular grid with per-subdomain interpolation and masked
Gaussian smoothing (the jump across Voronoi interfaces is preserved),
sphere cross-sections are drawn as cut-outs, interface traces as black
lines, and every error panel gets an independent 98th-percentile color
scale with a sci-format colorbar.

Panels (2 x 3, z = 0.5):

- row 1: velocity error ``||u_ddpnm| - |u_FEM||`` per method;
- row 2: saturation error ``|S_ddpnm - S_FEM|`` per method.

Only reads ``outputs/benchmark_twophase/twophase_fields.npz`` — no need to
re-run the benchmark.

Run in the FEniCSx environment:

    conda run -n fenicsx --no-capture-output python -u plot_slice_errors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ddpnm3d.visualization import build_cellwise_slice

OUT_DIR = PROJECT_DIR / "outputs" / "benchmark_twophase"

METHODS = ("Classic-DDPNM-1", "NormalLinear-DDPNM-3", "Affine-DDPNM-9")
NPZ_KEY = {
    "Classic-DDPNM-1": "Classic_DDPNM_1",
    "NormalLinear-DDPNM-3": "NormalLinear_DDPNM_3",
    "Affine-DDPNM-9": "Affine_DDPNM_9",
}
Z_VALUE = 0.5
N_GRID = 500


# ---------------------------------------------------------------------------
# Ported helpers from the archived plot_random_errors.py
# ---------------------------------------------------------------------------


def sci_colorbar(fig, artist, ax):
    cb = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.025)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    cb.formatter = formatter
    cb.update_ticks()
    cb.ax.tick_params(labelsize=9, length=2.5)
    return cb


def _point_in_triangle(px, py, a, b, c):
    """Barycentric-side test (true on edges), vectorized over triangles."""
    d1 = (b[:, 0] - a[:, 0]) * (py - a[:, 1]) - (b[:, 1] - a[:, 1]) * (px - a[:, 0])
    d2 = (c[:, 0] - b[:, 0]) * (py - b[:, 1]) - (c[:, 1] - b[:, 1]) * (px - b[:, 0])
    d3 = (a[:, 0] - c[:, 0]) * (py - c[:, 1]) - (a[:, 1] - c[:, 1]) * (px - c[:, 0])
    neg = (d1 < 0.0) | (d2 < 0.0) | (d3 < 0.0)
    pos = (d1 > 0.0) | (d2 > 0.0) | (d3 > 0.0)
    return ~(neg & pos)


def _classify_slice_grid(
    slice_points, slice_triangles, vertex_labels, z_value, spheres, radii, n=N_GRID,
):
    """Assign every regular-grid point to its parent slice triangle/subdomain."""
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, ys)
    grid = np.column_stack((X.ravel(), Y.ravel()))
    tris_pts = slice_points[slice_triangles][:, :, :2]
    tlabels = vertex_labels[slice_triangles][:, 0]

    nbins = 128
    bin_size = 1.0 / nbins
    bin_tris = {}
    lo = tris_pts.min(axis=1)
    hi = tris_pts.max(axis=1)
    ib_lo = np.clip(np.floor(lo / bin_size).astype(int), 0, nbins - 1)
    ib_hi = np.clip(np.floor(hi / bin_size).astype(int), 0, nbins - 1)
    for t in range(len(slice_triangles)):
        for bx in range(ib_lo[t, 0], ib_hi[t, 0] + 1):
            for by in range(ib_lo[t, 1], ib_hi[t, 1] + 1):
                bin_tris.setdefault((bx, by), []).append(t)

    in_ball = np.zeros(len(grid), dtype=bool)
    for center, radius in zip(spheres, radii, strict=True):
        dz = abs(float(center[2]) - z_value)
        if dz >= float(radius):
            continue
        rho = float(np.sqrt(radius**2 - dz**2))
        d2 = (grid[:, 0] - center[0]) ** 2 + (grid[:, 1] - center[1]) ** 2
        in_ball |= d2 < rho**2

    cell_of_point = np.full(len(grid), -1, dtype=int)
    for i, (px, py) in enumerate(grid):
        if in_ball[i]:
            continue
        bx = min(int(px / bin_size), nbins - 1)
        by = min(int(py / bin_size), nbins - 1)
        candidates = bin_tris.get((bx, by), ())
        if not candidates:
            continue
        cand = np.asarray(candidates, dtype=int)
        inside = _point_in_triangle(
            px, py, tris_pts[cand, 0], tris_pts[cand, 1], tris_pts[cand, 2]
        )
        hit = cand[inside]
        if len(hit):
            cell_of_point[i] = tlabels[hit[0]]
    return X, Y, cell_of_point, cell_of_point >= 0


def _smooth_slice_grid(
    values, slice_points, vertex_labels, X, Y, cell_of_point, usable, n=N_GRID, sigma=0.8,
):
    """Per-subdomain linear interpolation + masked Gaussian smoothing."""
    grid = np.column_stack((X.ravel(), Y.ravel()))
    Z = np.full(len(grid), np.nan)
    for k in np.unique(vertex_labels):
        in_k = vertex_labels == k
        if int(in_k.sum()) < 4:
            continue
        interpolator = LinearNDInterpolator(slice_points[in_k][:, :2], values[in_k])
        sel = usable & (cell_of_point == k)
        Z[sel] = interpolator(grid[sel])

    Z2d = Z.reshape(n, n)
    smooth = np.full_like(Z2d, np.nan)
    for k in np.unique(vertex_labels):
        m = np.isfinite(Z2d) & (cell_of_point.reshape(n, n) == k)
        if not m.any():
            continue
        numerator = gaussian_filter(np.where(m, Z2d, 0.0), sigma)
        denominator = gaussian_filter(m.astype(float), sigma)
        smooth[m] = numerator[m] / np.maximum(denominator[m], 1.0e-12)
    return smooth


def _subdomain_boundary_lines(slice_points, slice_triangles, vertex_labels):
    """Interior slice edges shared by two subdomains = interface traces."""
    tlabels = vertex_labels[slice_triangles][:, 0]
    edge_triangles = {}
    for t, (i, j, k) in enumerate(slice_triangles):
        for a, b in ((i, j), (j, k), (k, i)):
            edge_triangles.setdefault(tuple(sorted((a, b))), []).append(t)
    lines = []
    for key, triangles in edge_triangles.items():
        if len(triangles) == 2 and tlabels[triangles[0]] != tlabels[triangles[1]]:
            lines.append([slice_points[key[0]][:2], slice_points[key[1]][:2]])
    return lines


# ---------------------------------------------------------------------------
# Slice evaluation of the P1 vertex fields
# ---------------------------------------------------------------------------


def _slice_groups(slice_points, parent_cells):
    """Contiguous (parent cell, slice-point index range) groups."""
    order = np.argsort(parent_cells, kind="stable")
    sorted_cells = parent_cells[order]
    bounds = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1], True])
    groups = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        groups.append((int(sorted_cells[start]), order[start:stop]))
    return groups


def evaluate_p1_on_slice(
    points, tetrahedra, values, slice_points, groups
) -> np.ndarray:
    """P1 vertex field -> slice-point values by per-tetra barycentric weights."""
    out = np.zeros((len(slice_points), values.shape[1]), dtype=float)
    for cell, indices in groups:
        verts = tetrahedra[cell]
        matrix = np.column_stack([points[verts], np.ones(4)])  # (4, 4)
        pad = np.column_stack([slice_points[indices], np.ones(len(indices))])
        weights = np.linalg.solve(matrix, pad.T).T  # (k, 4)
        out[indices] = weights @ np.asarray(values[verts], dtype=float)
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def plot_slice_error_fields(data, out_dir: Path) -> None:
    points = np.asarray(data["points"], dtype=float)
    tetrahedra = np.asarray(data["tetrahedra"], dtype=np.int64)
    cell_labels = np.asarray(data["cell_labels"], dtype=np.int32)
    spheres = data["sphere_centers"]
    radii = data["sphere_radii"]

    slice_points, slice_triangles, parent_cells = build_cellwise_slice(
        points, tetrahedra, Z_VALUE
    )
    groups = _slice_groups(slice_points, parent_cells)
    vertex_labels = cell_labels[parent_cells]

    u_fem = np.asarray(data["u_FEM"], dtype=float)
    s_fem = np.asarray(data["s_FEM"], dtype=float)
    u_fem_slice = evaluate_p1_on_slice(points, tetrahedra, u_fem, slice_points, groups)
    s_fem_slice = evaluate_p1_on_slice(
        points, tetrahedra, s_fem[:, None], slice_points, groups
    )[:, 0]
    speed_ref = np.linalg.norm(u_fem_slice, axis=1)

    fields: dict[str, np.ndarray] = {}
    for method in METHODS:
        key = NPZ_KEY[method]
        u_dd = evaluate_p1_on_slice(
            points, tetrahedra, np.asarray(data[f"u_{key}"], dtype=float), slice_points, groups
        )
        s_dd = evaluate_p1_on_slice(
            points, tetrahedra, np.asarray(data[f"s_{key}"], dtype=float)[:, None],
            slice_points, groups,
        )[:, 0]
        fields[f"speed_err_{method}"] = np.abs(np.linalg.norm(u_dd, axis=1) - speed_ref)
        fields[f"s_err_{method}"] = np.abs(s_dd - s_fem_slice)

    X, Y, cell_of_point, usable = _classify_slice_grid(
        slice_points, slice_triangles, vertex_labels, Z_VALUE, spheres, radii
    )
    grids = {
        key: _smooth_slice_grid(
            values, slice_points, vertex_labels, X, Y, cell_of_point, usable
        )
        for key, values in fields.items()
    }

    def limit(grid) -> float:
        finite = grid[~np.isnan(grid)]
        return float(np.percentile(finite, 98.0)) if len(finite) else 1.0

    entries = []
    for method in METHODS:
        entries.append(
            (
                grids[f"speed_err_{method}"],
                "turbo", 0.0, limit(grids[f"speed_err_{method}"]),
                f"{method}: $||u|-|u_{{\\mathrm{{FEM}}}}||$",
            )
        )
    for method in METHODS:
        entries.append(
            (
                grids[f"s_err_{method}"],
                "turbo", 0.0, limit(grids[f"s_err_{method}"]),
                f"{method}: $|S-S_{{\\mathrm{{FEM}}}}|$",
            )
        )

    boundary_segments = _subdomain_boundary_lines(
        slice_points, slice_triangles, vertex_labels
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.6))
    plt.subplots_adjust(
        left=0.045, right=0.97, bottom=0.075, top=0.965, wspace=0.32, hspace=0.22
    )
    for index, (ax, (values, cmap, vmin, vmax, title)) in enumerate(
        zip(axes.ravel(), entries, strict=True)
    ):
        artist = ax.pcolormesh(X, Y, values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.add_collection(
            LineCollection(boundary_segments, colors="#1f1f1f", linewidths=1.3, zorder=5)
        )
        # No sphere cross-section outlines: the field is cut directly by the
        # obstacles (NaN inside the spheres renders as clean white cut-outs).
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.text(
            0.5, -0.075, f"({chr(ord('a') + index)})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, fontfamily="serif",
        )
        sci_colorbar(fig, artist, ax)
    fig.text(
        0.012, 0.76, "Velocity error vs FEM",
        rotation=90, va="center", ha="center", fontsize=11, fontstyle="italic",
    )
    fig.text(
        0.012, 0.28, "Saturation error vs FEM",
        rotation=90, va="center", ha="center", fontsize=11, fontstyle="italic",
    )
    fig.text(
        0.5, 0.015, f"slice z = {Z_VALUE:g}; regular-grid resampling with masked "
        "per-subdomain smoothing; the field is cut directly by the solid spheres "
        "(no outlines) with Voronoi interface traces",
        ha="center", va="top", fontsize=8.5, color="#444444",
    )
    fig.savefig(out_dir / "slice_error_fields.png", dpi=240)
    plt.close(fig)


def main() -> None:
    data = np.load(OUT_DIR / "twophase_fields.npz")
    plot_slice_error_fields(data, OUT_DIR)
    print(f"wrote {OUT_DIR / 'slice_error_fields.png'}")


if __name__ == "__main__":
    main()

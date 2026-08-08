#!/usr/bin/env python3
"""Error-cloud and comparison figures for the random-sphere affine benchmark.

The plotting style follows the 3-D uniform folder (slice error fields with
sphere cut-outs, 3-D construction panels) and the 2-D folder (paper-style
field panels, strict 2-D/3-D comparison bars).

Outputs
-------
01_slice_error_fields.png   2x4 slice panels at z=0.5 (solutions + error fields)
02_geometry_and_partition.png  3-D construction panels
03_2d_3d_error_ratio.png    strict P0-DDPNM 2-D vs 3-D ratio bars
04_methods_vs_exact_schur.png  reduced methods vs exact FE-trace Schur
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter

COLORS = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377",
    "#BBBBBB", "#EE8866", "#44AA99", "#999933", "#882255", "#117733",
]

REPOSITORY_DIR = Path(__file__).resolve().parent.parent
STRICT_2D_REPORT = (
    REPOSITORY_DIR
    / "ddpnm_2d_random_porous"
    / "outputs"
    / "strict_p0_comparison"
    / "strict_2d_p0_report.json"
)

METRICS = (
    ("velocity_relative_l2", r"velocity $L^2$"),
    ("velocity_relative_broken_h1_seminorm", r"velocity broken-$H^1$"),
    ("pressure_raw_relative_l2", r"pressure raw $L^2$"),
    ("pressure_mean_aligned_relative_l2", r"pressure aligned $L^2$"),
    ("outlet_flux_relative_error", "outlet flux"),
)

# 3-D report uses the short names without the 2-D suffixes.
METRIC_3D_KEYS = {
    "velocity_relative_l2": "velocity_relative_l2",
    "velocity_relative_broken_h1_seminorm": "velocity_relative_broken_h1",
    "pressure_raw_relative_l2": "pressure_relative_l2",
    "pressure_mean_aligned_relative_l2": "pressure_aligned_relative_l2",
    "outlet_flux_relative_error": "outlet_flux_relative_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fields", type=Path, default=Path("outputs/benchmark/random_benchmark_fields.npz")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/benchmark/random_affine_report.json")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/benchmark"))
    return parser.parse_args()


def sci_colorbar(fig, artist, ax, signed: bool = False):
    cb = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.025)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    cb.formatter = formatter
    cb.update_ticks()
    cb.ax.tick_params(labelsize=9, length=2.5)
    return cb


def set_equal_cube(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-55)


def cube_edges(lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    vertices = np.asarray(list(np.ndindex(2, 2, 2)), dtype=float)
    vertices = lower + (upper - lower) * vertices
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(vertices[i] != vertices[j]) == 1:
                edges.append([vertices[i], vertices[j]])
    return np.asarray(edges)


def draw_cube(ax, color="#30343b", linewidth=0.8, alpha=0.9) -> None:
    ax.add_collection3d(
        Line3DCollection(cube_edges(), colors=color, linewidths=linewidth, alpha=alpha)
    )


def sphere_surface(center: np.ndarray, radius: float, nu=18, nv=10):
    u = np.linspace(0.0, 2.0 * np.pi, nu)
    v = np.linspace(0.0, np.pi, nv)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def draw_spheres(ax, centers, radii, color="#7f8288", alpha=0.55, linewidth=0.15):
    for center, value in zip(centers, radii, strict=True):
        x, y, z = sphere_surface(center, float(value))
        ax.plot_surface(
            x, y, z, color=color, alpha=alpha, linewidth=linewidth,
            edgecolor="#60636a" if linewidth else "none", shade=True,
        )


def slice_sphere_cuts(ax, spheres, radii, z_value, color="#404040"):
    for center, radius in zip(spheres, radii, strict=True):
        dz = abs(float(center[2]) - z_value)
        if dz >= float(radius):
            continue
        rho = float(np.sqrt(radius**2 - dz**2))
        circle = np.linspace(0.0, 2.0 * np.pi, 120)
        ax.plot(
            center[0] + rho * np.cos(circle),
            center[1] + rho * np.sin(circle),
            color=color, linewidth=0.9,
        )


def _point_in_triangle(px, py, a, b, c):
    """Barycentric-side test (true on edges), vectorized over triangles."""
    d1 = (b[:, 0] - a[:, 0]) * (py - a[:, 1]) - (b[:, 1] - a[:, 1]) * (px - a[:, 0])
    d2 = (c[:, 0] - b[:, 0]) * (py - b[:, 1]) - (c[:, 1] - b[:, 1]) * (px - b[:, 0])
    d3 = (a[:, 0] - c[:, 0]) * (py - c[:, 1]) - (a[:, 1] - c[:, 1]) * (px - c[:, 0])
    neg = (d1 < 0.0) | (d2 < 0.0) | (d3 < 0.0)
    pos = (d1 > 0.0) | (d2 > 0.0) | (d3 > 0.0)
    return ~(neg & pos)


def _classify_slice_grid(
    slice_points, slice_triangles, vertex_labels, z_value, spheres, radii, n=500,
):
    """Assign every regular-grid point to its parent slice triangle/subdomain.

    Uses a binning hash (triangle bounding boxes registered per bin), then
    exact point-in-triangle tests, so the classification follows the actual
    slice triangulation (with its sphere holes) rather than a fill-in
    Delaunay hull.  Points inside sphere cross-sections get label -1.
    """
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, ys)
    grid = np.column_stack((X.ravel(), Y.ravel()))
    tris_pts = slice_points[slice_triangles][:, :, :2]  # (T, 3, 2)
    tlabels = vertex_labels[slice_triangles][:, 0]  # slice tri -> subdomain

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
        inside = _point_in_triangle(px, py, tris_pts[cand, 0], tris_pts[cand, 1], tris_pts[cand, 2])
        hit = cand[inside]
        if len(hit):
            cell_of_point[i] = tlabels[hit[0]]
    return X, Y, cell_of_point, cell_of_point >= 0


def _smooth_slice_grid(
    values, slice_points, vertex_labels, X, Y, cell_of_point, usable, n=500, sigma=0.8,
):
    """Per-subdomain linear interpolation + light Gaussian smoothing.

    Each grid point is evaluated with the interpolant built from slice
    vertices of its own subdomain only, and the smoothing convolution is
    masked per subdomain (normalized convolution).  The discontinuity
    across Voronoi interfaces is therefore preserved while the field reads
    smooth inside every pore region.
    """
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


def plot_slice_error_fields(data, out_dir: Path) -> None:
    """2x4 panels: reference/affine solutions on top, classic/affine errors below.

    Fields are resampled on a regular grid and smoothed *inside* each
    subdomain only, so the fields read smooth within a pore region while
    the jump across Voronoi interfaces is kept and emphasized by the
    subdomain boundary lines.  The classic and affine error panels get
    independent color scales (98th percentiles), since the affine errors
    are about an order of magnitude smaller.
    """
    z_value = float(data["classic_error_slice_z"][0])
    slice_points = data["classic_error_slice_points"]
    slice_triangles = data["classic_error_slice_triangles"]
    spheres = data["sphere_centers"]
    radii = data["sphere_radii"]
    vertex_labels = np.asarray(data["cell_labels"])[data["classic_error_slice_parent_cells"]]

    speed_ref = np.linalg.norm(data["classic_error_slice_u_fem"], axis=1)
    p_ref = data["classic_error_slice_p_fem"]
    slice_fields = {"speed_ref": speed_ref, "p_ref": p_ref}
    for name in ("classic", "affine"):
        u_dd = data[f"{name}_error_slice_u_ddpnm"]
        p_dd = data[f"{name}_error_slice_p_ddpnm"]
        slice_fields[f"speed_dd_{name}"] = np.linalg.norm(u_dd, axis=1)
        slice_fields[f"p_dd_{name}"] = p_dd
        slice_fields[f"speed_err_{name}"] = np.abs(
            np.linalg.norm(u_dd, axis=1) - speed_ref
        )
        slice_fields[f"p_err_{name}"] = p_dd - p_ref

    n = 500
    X, Y, cell_of_point, usable = _classify_slice_grid(
        slice_points, slice_triangles, vertex_labels, z_value, spheres, radii, n=n
    )
    grids = {
        key: _smooth_slice_grid(
            values, slice_points, vertex_labels, X, Y, cell_of_point, usable, n=n
        )
        for key, values in slice_fields.items()
    }

    # Independent color scales: classic errors are ~10x larger than affine.
    speed_limit_classic = float(np.percentile(grids["speed_err_classic"][~np.isnan(grids["speed_err_classic"])], 98.0))
    speed_limit_affine = float(np.percentile(grids["speed_err_affine"][~np.isnan(grids["speed_err_affine"])], 98.0))
    p_limit_classic = float(np.percentile(np.abs(grids["p_err_classic"][~np.isnan(grids["p_err_classic"])]), 98.0))
    p_limit_affine = float(np.percentile(np.abs(grids["p_err_affine"][~np.isnan(grids["p_err_affine"])]), 98.0))

    fig, axes = plt.subplots(2, 4, figsize=(17.0, 8.6))
    plt.subplots_adjust(
        left=0.045, right=0.97, bottom=0.075, top=0.965, wspace=0.27, hspace=0.20
    )
    entries = [
        (grids["speed_ref"], "viridis", 0.0, float(np.max(speed_ref)), r"$|u|_{\mathrm{FEM}}$"),
        (grids["p_ref"], "viridis", float(p_ref.min()), float(p_ref.max()), r"$p_{\mathrm{FEM}}$"),
        (grids["speed_dd_affine"], "viridis", 0.0, float(np.max(speed_ref)), r"$|u|_{\mathrm{affine}}$"),
        (grids["p_dd_affine"], "viridis", float(p_ref.min()), float(p_ref.max()), r"$p_{\mathrm{affine}}$"),
        (grids["speed_err_classic"], "turbo", 0.0, speed_limit_classic, r"classic: $||u|-|u_{\mathrm{FEM}}||$"),
        (grids["p_err_classic"], "RdBu_r", -p_limit_classic, p_limit_classic, r"classic: $p-p_{\mathrm{FEM}}$"),
        (grids["speed_err_affine"], "turbo", 0.0, speed_limit_affine, r"affine: $||u|-|u_{\mathrm{FEM}}||$"),
        (grids["p_err_affine"], "RdBu_r", -p_limit_affine, p_limit_affine, r"affine: $p-p_{\mathrm{FEM}}$"),
    ]
    boundary_segments = _subdomain_boundary_lines(
        slice_points, slice_triangles, vertex_labels
    )
    for index, (ax, (values, cmap, vmin, vmax, title)) in enumerate(
        zip(axes.ravel(), entries, strict=True)
    ):
        artist = ax.pcolormesh(X, Y, values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.add_collection(
            LineCollection(
                boundary_segments, colors="#1f1f1f", linewidths=1.3, zorder=5
            )
        )
        slice_sphere_cuts(ax, spheres, radii, z_value, color="#3a3a3a")
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=10)
        ax.text(
            0.5, -0.075, f"({chr(ord('a') + index)})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, fontfamily="serif",
        )
        sci_colorbar(fig, artist, ax, signed=(index in (5, 7)))
    fig.text(
        0.012, 0.76, "Reference and affine solutions",
        rotation=90, va="center", ha="center", fontsize=11, fontstyle="italic",
    )
    fig.text(
        0.012, 0.28, "Error fields vs FEM",
        rotation=90, va="center", ha="center", fontsize=11, fontstyle="italic",
    )
    fig.savefig(out_dir / "01_slice_error_fields.png", dpi=240)
    plt.close(fig)


def plot_geometry(data, out_dir: Path) -> None:
    points = data["points"]
    tetrahedra = data["tetrahedra"]
    cell_labels = data["cell_labels"]
    spheres = data["sphere_centers"]
    radii = data["sphere_radii"]
    balls = data["maximal_ball_centers"]
    ball_radii = data["maximal_ball_radii"]
    pairs = data["interface_pairs"]

    fig = plt.figure(figsize=(18.0, 5.1), constrained_layout=True)
    axes = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]

    draw_spheres(axes[0], spheres, radii, alpha=0.72, linewidth=0.12)
    draw_cube(axes[0])
    axes[0].set_title("Computational geometry\n27 random spheres (seed 20260804)", fontsize=10.5)

    draw_spheres(axes[1], spheres, radii, color="#a5a7ab", alpha=0.14, linewidth=0.0)
    draw_spheres(axes[1], balls, ball_radii, color="#ef8a62", alpha=0.30, linewidth=0.20)
    for i, j in pairs:
        axes[1].plot(
            balls[[i, j], 0], balls[[i, j], 1], balls[[i, j], 2],
            color="#1464b4", linewidth=1.4,
        )
    axes[1].scatter(
        balls[:, 0], balls[:, 1], balls[:, 2], s=20, color="#b2182b", depthshade=False
    )
    draw_cube(axes[1], alpha=0.55)
    axes[1].set_title("Pore bodies and throat graph", fontsize=10.5)

    centers = data["interface_centers"]
    normals = data["interface_normals"]
    axes[2].quiver(
        centers[:, 0], centers[:, 1], centers[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        length=0.05, normalize=False, color="#1455a0", alpha=0.9, linewidth=1.1,
    )
    draw_spheres(axes[2], spheres, radii, color="#8c8f94", alpha=0.22, linewidth=0.0)
    axes[2].scatter(
        centers[:, 0], centers[:, 1], centers[:, 2],
        s=9, color="#b2182b", depthshade=False,
    )
    draw_cube(axes[2])
    axes[2].set_title("Voronoi throat interfaces\n(centers and normals)", fontsize=10.5)

    labels = np.asarray(cell_labels, dtype=int)
    for label in np.unique(labels):
        cells = tetrahedra[labels == label]
        edges = cells[:, np.asarray([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])].reshape(-1, 2)
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
        axes[3].add_collection3d(
            Line3DCollection(
                points[edges], colors=COLORS[label % len(COLORS)], linewidths=0.18, alpha=0.30
            )
        )
    draw_cube(axes[3], alpha=0.65)
    axes[3].set_title("Pore subdomains\n(colored tetrahedral regions)", fontsize=10.5)

    for ax in axes:
        set_equal_cube(ax)
        ax.text2D(0.50, -0.035, "", transform=ax.transAxes, ha="center", fontsize=12)
    fig.savefig(out_dir / "02_geometry_and_partition.png", dpi=250)
    plt.close(fig)


def plot_2d_3d_ratio(report, out_dir: Path) -> None:
    """Strict P0-DDPNM 2-D vs 3-D comparison bars with ratio labels."""
    values_2d = json.loads(STRICT_2D_REPORT.read_text(encoding="utf-8"))[
        "strict_validation"
    ]
    values_3d = report["methods"]["Classic-DDPNM-1"]
    array_2d = np.asarray([values_2d[key] for key, _ in METRICS])
    array_3d = np.asarray(
        [values_3d[METRIC_3D_KEYS[key]] for key, _ in METRICS]
    )

    x = np.arange(len(METRICS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.20)
    bars_2d = ax.bar(x - width / 2, 100.0 * array_2d, width, label="2-D random (17 circles)")
    bars_3d = ax.bar(x + width / 2, 100.0 * array_3d, width, label="3-D random (27 spheres)")
    for bars in (bars_2d, bars_3d):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=8.5,
            )
    for index, (key, _) in enumerate(METRICS):
        ratio = array_3d[index] / max(array_2d[index], 1.0e-15)
        ax.annotate(
            f"3-D/2-D = {ratio:.2f}",
            (x[index], 100.0 * max(array_2d[index], array_3d[index]) * 1.10),
            ha="center", va="bottom", fontsize=8.5, color="#881122",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRICS], fontsize=9)
    ax.set_ylabel("relative error / %")
    ax.set_ylim(0.0, 100.0 * max(array_2d.max(), array_3d.max()) * 1.45)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title(
        "Original P0-DDPNM: strict same-mesh errors, 2-D vs 3-D random media",
        fontsize=11.5,
    )
    fig.savefig(out_dir / "03_2d_3d_error_ratio.png", dpi=240)
    plt.close(fig)


def plot_methods_vs_schur(report, out_dir: Path) -> None:
    """Reduced methods vs the exact FE-trace Schur solution (2-D style)."""
    errors = report["errors_to_exact_fe_schur"]
    names = list(errors.keys())
    metrics = (
        ("velocity_relative_l2", r"velocity $L^2$"),
        ("velocity_relative_broken_h1", r"velocity broken-$H^1$"),
        ("pressure_relative_l2", r"pressure raw $L^2$"),
        ("outlet_flux_relative_error", "outlet flux"),
    )
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.18)
    for offset, name in zip((-width / 2, width / 2), names, strict=True):
        values = np.asarray([errors[name][key] for key, _ in metrics])
        bars = ax.bar(x + offset, 100.0 * values, width, label=name)
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=8.5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], fontsize=9)
    ax.set_ylabel("relative error vs exact FE-trace Schur / %")
    ax.legend(fontsize=10)
    ax.set_title("Reduced interface spaces vs the exact FE-trace Schur solution", fontsize=11.5)
    fig.savefig(out_dir / "04_methods_vs_exact_schur.png", dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.fields)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    plot_slice_error_fields(data, args.out_dir)
    plot_geometry(data, args.out_dir)
    plot_2d_3d_ratio(report, args.out_dir)
    plot_methods_vs_schur(report, args.out_dir)
    print(f"Figures written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Shared data layer and styling for the final numerical-section figures.

Style conventions follow the archived random-porous figure script
``affine_ddpnm_3d_random_porous/plot_random_errors.py`` (the source of
``outputs/benchmark/01_slice_error_fields.png``): slice fields rendered by
per-subdomain linear interpolation + masked Gaussian smoothing, sphere
cross-section outlines, subdomain boundary lines, scientific-notation
colorbars, serif panel letters, and the 2x4 layout constants.  The
categorical method palette is the validated dataviz reference instance also
used by ``deepseekoutput/20260810_numerical_section/scripts/make_figures.py``
(first three slots validate all-pairs; the violet slot is reserved for W0v
and only rendered when W0v data becomes available).

Method naming used throughout the numerical section:
    W0n  span{1} x n            constant normal     (Classic-DDPNM-1, r=1)
    W0v  span{1} x {n,t1,t2}    constant vector     (not run yet, r=3)
    W1n  span{1,s,t} x n        linear normal       (NormalLinear-DDPNM-3, r=3)
    W1v  span{1,s,t} x {n,t1,t2} affine nine-mode   (Affine-DDPNM-9, r=9)

Run under the econ env::

    /d/Miniconda3/envs/econ/python.exe make_final_figures.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter

# ----------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO = _SCRIPT_DIR.parent.parent.parent  # D:\\hu\\tongjiproj\\20260727
FIG_DIR = _SCRIPT_DIR.parent / "figures_gpt"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BENCH = {
    "Uniform-27": REPO / "affine_ddpnm_3d/outputs/benchmark_w1n",
    "Random-27": REPO / "affine_ddpnm_3d_random_porous/outputs/benchmark_w1n",
    "Real-100": REPO / "real_porous_benchmark_3d/outputs/benchmark_w1n",
}

# ----------------------------------------------------------------------------
# validated categorical palette (dataviz reference instance, light mode)
# ----------------------------------------------------------------------------
C_BLUE = "#2a78d6"     # slot 1  -> W0n
C_ORANGE = "#eb6834"   # slot 2  -> W1n
C_AQUA = "#1baf7a"     # slot 3  -> W1v
C_VIOLET = "#4a3aa7"   # slot 7  -> W0v (reserved; only drawn when data exists)
C_GRAY = "#52514e"     # neutral ink -> reference (FEM)
C_GRID = "#e1e0d9"
C_INK = "#0b0b0b"
C_MUTED = "#898781"

METHOD_COLORS = {
    "W0n": C_BLUE,
    "W0v": C_VIOLET,
    "W1n": C_ORANGE,
    "W1v": C_AQUA,
}
METHOD_LABELS = {
    "W0n": r"$W_{0n}$",
    "W0v": r"$W_{0v}$",
    "W1n": r"$W_{1n}$",
    "W1v": r"$W_{1v}$",
}
# W0n -> W0v -> W1v  and  W0n -> W1n -> W1v are the legal nested paths.
# W0v and W1n are generally *not* nested, so they are never connected.
NESTED_PATHS = (("W0n", "W0v", "W1v"), ("W0n", "W1n", "W1v"))
GEOM_LABELS = {"Uniform-27": "Uniform-27", "Random-27": "Random-27", "Real-100": "Real-100"}
GEOM_ORDER = ["Uniform-27", "Random-27", "Real-100"]
METHOD_ORDER = ["W0n", "W0v", "W1n", "W1v"]  # W0v drops out automatically when absent

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

# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
_COL_RENAMES = {
    "Uniform-27": {},
    "Random-27": {},
    "Real-100": {
        "velocity_relative_L2": "velocity_relative_l2",
        "velocity_relative_broken_H1": "velocity_relative_broken_h1",
        "pressure_relative_L2": "pressure_relative_l2",
        "outlet_flux_relative_error": "outlet_flux_relative_error",
        "offline_seconds": "offline_s",
        "online_seconds": "online_s",
        "first_solve_seconds": "total_s",
        "global_unknowns": "global_unknowns",
        "modes_per_interface": "modes_per_interface",
    },
}
_METHOD_MAP = {
    "Classic-DDPNM-1": "W0n",
    "Classic-DDPNM": "W0n",
    "NormalLinear-DDPNM-3": "W1n",
    "Affine-DDPNM-9": "W1v",
    "Affine-DDPNM": "W1v",
    # W0v (constant vector) would be mapped here once a run exists, e.g.
    # "VectorConstant-DDPNM-3": "W0v",
}


def load_metrics() -> dict:
    """Return {geometry: {method: metrics}} with method keys in METHOD_ORDER."""
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
    }
    out = {}
    for geom in GEOM_ORDER:
        csv_path = BENCH[geom] / {
            "Uniform-27": "affine_ddpnm_metrics.csv",
            "Random-27": "random_affine_metrics.csv",
            "Real-100": "benchmark_metrics.csv",
        }[geom]
        rows = []
        with open(csv_path, encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        header = [h.strip() for h in lines[0].split(",")]
        for line in lines[1:]:
            rows.append(dict(zip(header, (v.strip() for v in line.split(",")))))
        rename = _COL_RENAMES[geom]
        methods = {}
        fem_first = None
        for row in rows:
            name = row["method"].strip()
            key = _METHOD_MAP.get(name)
            if name == "Monolithic-FEM":
                col = rename.get(cols["first"], cols["first"])
                if row.get(col) not in (None, ""):
                    fem_first = float(row[col])
                continue
            if key is None:
                continue
            m = {}
            for k, col in cols.items():
                col = rename.get(col, col)
                m[k] = float(row[col]) if row.get(col) not in (None, "") else None
            if m["modes"] is None:  # Real-100 CSV omits the modes column
                m["modes"] = {"W0n": 1, "W0v": 3, "W1n": 3, "W1v": 9}[key]
            methods[key] = m
        methods["__fem__"] = {"first": fem_first}
        out[geom] = methods
    return out


def present_methods(metrics: dict, geom: str) -> list[str]:
    """Method keys actually available for a geometry (drops W0v if absent)."""
    return [k for k in METHOD_ORDER if k in metrics[geom]]


# ----------------------------------------------------------------------------
# slice fields
# ----------------------------------------------------------------------------
_FIELD_FILES = {
    "Uniform-27": "affine_benchmark_fields.npz",
    "Random-27": "random_benchmark_fields.npz",
    "Real-100": None,  # current W1n benchmark saves no slice fields (z_slice API bug)
}


def load_fields(geom: str):
    """Return the benchmark field archive for a geometry, or None if absent."""
    fname = _FIELD_FILES[geom]
    if fname is None:
        return None
    path = BENCH[geom] / fname
    if not path.is_file():
        return None
    return np.load(path)


def _point_in_triangle(px, py, a, b, c):
    """Barycentric-side test (true on edges), vectorized over triangles."""
    d1 = (b[:, 0] - a[:, 0]) * (py - a[:, 1]) - (b[:, 1] - a[:, 1]) * (px - a[:, 0])
    d2 = (c[:, 0] - b[:, 0]) * (py - b[:, 1]) - (c[:, 1] - b[:, 1]) * (px - b[:, 0])
    d3 = (a[:, 0] - c[:, 0]) * (py - c[:, 1]) - (a[:, 1] - c[:, 1]) * (px - c[:, 0])
    neg = (d1 < 0.0) | (d2 < 0.0) | (d3 < 0.0)
    pos = (d1 > 0.0) | (d2 > 0.0) | (d3 > 0.0)
    return ~(neg & pos)


def _classify_slice_grid(
    slice_points, slice_triangles, vertex_labels, z_value, spheres, radii, n=400,
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
    values, slice_points, vertex_labels, X, Y, cell_of_point, usable, n=400, sigma=0.8,
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


def slice_sphere_cuts(ax, spheres, radii, z_value, color="#3a3a3a"):
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


def slice_field_grids(data, n=400, spheres=None, radii=None, vertex_labels=None):
    """Build smooth regular-grid versions of the slice fields in ``data``.

    Returns dict of 2-D arrays keyed by 'speed_ref', 'err_W0n', 'err_W1n',
    'err_W1v' plus metadata (X, Y, boundary segments, sphere cuts, z).

    ``spheres``/``radii``/``vertex_labels`` may be passed explicitly for
    archives that do not carry them (the Uniform-27 npz has no sphere or
    cell-label arrays; the caller derives subdomain labels from the
    Cartesian grid cell of each slice vertex instead).
    """
    z_value = float(data["classic_error_slice_z"][0])
    slice_points = data["classic_error_slice_points"]
    slice_triangles = data["classic_error_slice_triangles"]
    if spheres is None:
        spheres = data["sphere_centers"]
    if radii is None:
        radii = data["sphere_radii"]
    if vertex_labels is None:
        vertex_labels = np.asarray(data["cell_labels"])[data["classic_error_slice_parent_cells"]]

    u_fem = data["classic_error_slice_u_fem"]
    speed_ref = np.linalg.norm(u_fem, axis=1)
    raw = {"speed_ref": speed_ref}
    for key, suffix in (("W0n", "classic"), ("W1n", "normal_linear"), ("W1v", "affine")):
        u_dd = data[f"{suffix}_error_slice_u_ddpnm"]
        # Pointwise vector-velocity error magnitude.  Do NOT use
        # | |u_DD| - |u_FEM| | here: that quantity can be much smaller than
        # ||u_DD-u_FEM|| and does not match the field-error label used in the
        # numerical section.
        raw[f"err_{key}"] = np.linalg.norm(u_dd - u_fem, axis=1)

    X, Y, cell_of_point, usable = _classify_slice_grid(
        slice_points, slice_triangles, vertex_labels, z_value, spheres, radii, n=n
    )
    grids = {
        key: _smooth_slice_grid(
            values, slice_points, vertex_labels, X, Y, cell_of_point, usable, n=n
        )
        for key, values in raw.items()
    }
    boundaries = _subdomain_boundary_lines(slice_points, slice_triangles, vertex_labels)
    return {
        "grids": grids,
        "X": X,
        "Y": Y,
        "boundaries": boundaries,
        "spheres": spheres,
        "radii": radii,
        "z": z_value,
        "speed_max": float(speed_ref.max()),
    }


def sci_colorbar(fig, artist, ax, labelsize=8, label=None):
    cb = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.025)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    cb.formatter = formatter
    cb.update_ticks()
    cb.ax.tick_params(labelsize=labelsize, length=2.5)
    if label:
        cb.set_label(label, fontsize=9)
    return cb


# ----------------------------------------------------------------------------
# geometry (fig1b)
# ----------------------------------------------------------------------------
def _extract_array(source: Path, var: str) -> np.ndarray:
    """Parse `VAR = np.asarray([...], dtype=...)` literal from a python source."""
    text = source.read_text(encoding="utf-8")
    m = re.search(rf"{var}\s*=\s*np\.asarray\(\s*(\[.*?\])\s*[,)]", text, re.S)
    if not m:
        raise RuntimeError(f"{var} not found in {source}")
    body = m.group(1).strip()
    return np.asarray(ast.literal_eval(body), dtype=float)


def load_spheres() -> dict:
    """Sphere centers/radii for all three geometries."""
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


# Grid-partition planes: Uniform-27 cuts at CELL_EDGES = [0, .2, .5, .8, 1];
# Real-100 cuts at GRID_PLANES = [.25, .5, .75] (both 4x4x4 = 64 cuboids,
# 144 finite-area internal interfaces).
GRID_PLANES = {
    "Uniform-27": np.asarray([0.2, 0.5, 0.8]),
    "Real-100": np.asarray([0.25, 0.5, 0.75]),
}


def load_random_partition() -> dict:
    """Full random-27 partition from the field archive (points/tets/balls/...)."""
    d = load_fields("Random-27")
    return {
        "points": d["points"],
        "tetrahedra": d["tetrahedra"],
        "cell_labels": d["cell_labels"],
        "balls": d["maximal_ball_centers"],
        "ball_radii": d["maximal_ball_radii"],
        "pairs": d["interface_pairs"],
        "centers": d["interface_centers"],
        "normals": d["interface_normals"],
    }


def draw_cube(ax, color="#30343b", linewidth=0.8, alpha=0.9) -> None:
    vertices = np.asarray(list(np.ndindex(2, 2, 2)), dtype=float)
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(vertices[i] != vertices[j]) == 1:
                edges.append([vertices[i], vertices[j]])
    ax.add_collection3d(
        Line3DCollection(np.asarray(edges), colors=color, linewidths=linewidth, alpha=alpha)
    )


def draw_spheres(ax, centers, radii, color="#7f8288", alpha=0.55, linewidth=0.15):
    u = np.linspace(0.0, 2.0 * np.pi, 18)
    v = np.linspace(0.0, np.pi, 10)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    for center, value in zip(centers, radii, strict=True):
        ax.plot_surface(
            center[0] + float(value) * xs,
            center[1] + float(value) * ys,
            center[2] + float(value) * zs,
            color=color, alpha=alpha, linewidth=linewidth,
            edgecolor="#60636a" if linewidth else "none", shade=True,
        )


def set_equal_cube(ax, elev=22.0, azim=-55.0) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=elev, azim=azim)

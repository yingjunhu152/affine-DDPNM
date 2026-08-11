#!/usr/bin/env python3
"""FEM-only two-phase field figures (from the last completed benchmark npz).

Writes to ``outputs/fem_plots/``:
- ``fem_water_saturation.png``  : final water saturation S_w, z = 0.5 slice;
- ``fem_oil_saturation.png``   : final oil saturation S_o = 1 - S_w;
- ``fem_velocity.png``         : speed |u| on the same slice.

Run in the FEniCSx environment:
    conda run -n fenicsx --no-capture-output python -u plot_fem_fields.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import plot_slice_errors as pse

OUT_DIR = PROJECT_DIR / "outputs" / "benchmark_twophase"
PLOT_DIR = PROJECT_DIR / "outputs" / "fem_plots"
Z_VALUE = 0.5
SW_LIM = (0.2, 0.8)


def panel(ax, X, Y, values, cmap, vmin, vmax, title, segments, cbar_ax=None, cbar_fig=None):
    artist = ax.pcolormesh(X, Y, values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.add_collection(LineCollection(segments, colors="#1f1f1f", linewidths=1.0, zorder=5))
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11)
    if cbar_ax is not None:
        pse.sci_colorbar(cbar_fig, artist, cbar_ax)


def main() -> None:
    data = np.load(OUT_DIR / "twophase_fields.npz")
    points = np.asarray(data["points"], dtype=float)
    tetrahedra = np.asarray(data["tetrahedra"], dtype=np.int64)
    cell_labels = np.asarray(data["cell_labels"], dtype=np.int32)
    spheres = data["sphere_centers"]
    radii = data["sphere_radii"]

    slice_points, slice_triangles, parent_cells = pse.build_cellwise_slice(
        points, tetrahedra, Z_VALUE
    )
    groups = pse._slice_groups(slice_points, parent_cells)
    vertex_labels = cell_labels[parent_cells]
    X, Y, cell_of_point, usable = pse._classify_slice_grid(
        slice_points, slice_triangles, vertex_labels, Z_VALUE, spheres, radii
    )
    segments = pse._subdomain_boundary_lines(slice_points, slice_triangles, vertex_labels)

    def smooth_field(vertex_values):
        on_slice = pse.evaluate_p1_on_slice(
            points, tetrahedra, np.asarray(vertex_values, dtype=float)[:, None],
            slice_points, groups,
        )[:, 0]
        return pse._smooth_slice_grid(
            on_slice, slice_points, vertex_labels, X, Y, cell_of_point, usable
        )

    s_w = smooth_field(data["s_FEM"])
    speed = smooth_field(np.linalg.norm(np.asarray(data["u_FEM"], dtype=float), axis=1))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # water saturation
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    panel(ax, X, Y, s_w, "viridis", SW_LIM[0], SW_LIM[1],
          "FEM: final water saturation $S_w$ (t = 30)", segments, ax, fig)
    fig.savefig(PLOT_DIR / "fem_water_saturation.png", dpi=200)
    plt.close(fig)

    # oil saturation
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    panel(ax, X, Y, 1.0 - s_w, "viridis", SW_LIM[0], SW_LIM[1],
          "FEM: final oil saturation $S_o = 1 - S_w$ (t = 30)", segments, ax, fig)
    fig.savefig(PLOT_DIR / "fem_oil_saturation.png", dpi=200)
    plt.close(fig)

    # speed
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    vmax = float(np.percentile(speed[~np.isnan(speed)], 98.0))
    panel(ax, X, Y, speed, "turbo", 0.0, vmax,
          "FEM: speed $|u|$ (z = 0.5)", segments, ax, fig)
    fig.savefig(PLOT_DIR / "fem_velocity.png", dpi=200)
    plt.close(fig)

    print(f"wrote {PLOT_DIR / 'fem_water_saturation.png'}")
    print(f"wrote {PLOT_DIR / 'fem_oil_saturation.png'}")
    print(f"wrote {PLOT_DIR / 'fem_velocity.png'}")


if __name__ == "__main__":
    main()

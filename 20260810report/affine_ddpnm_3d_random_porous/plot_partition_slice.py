#!/usr/bin/env python3
"""Partition geometry on the z ~ 0.5 slice: subregion traces and spheres.

Two bare panels, 01-style: (a) the Voronoi partition and (b) the
clearance-watershed partition.  Only the subdomain boundary lines
(interface traces on the slice) and the sphere cut-outs are drawn --
no field data.  Subregion/interface counts in the titles come from the
mesh-level arrays of each npz.

Data: outputs/benchmark/random_benchmark_fields.npz (frozen Voronoi)
and outputs/ablation_4way/watershed/random_benchmark_fields.npz.

Output: outputs/ablation_4way/partition_slice.png
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from plot_random_errors import slice_sphere_cuts

PROJECT_DIR = Path(__file__).resolve().parent
OUT = PROJECT_DIR / "outputs" / "ablation_4way" / "partition_slice.png"


def subdomain_interface_lines(
    slice_points, slice_triangles, spheres, radii, z_value
) -> list[list[np.ndarray]]:
    """Slice traces of the subdomain interfaces.

    The slice triangulation is non-conforming across subdomains (each
    subdomain is triangulated separately), so an interface trace is a
    contour edge (an edge used by exactly one slice triangle) that lies
    neither on the cube wall nor inside a sphere cut-out.  The two
    coincident contour edges of a non-conforming seam are both kept,
    which draws the seam once as a solid line.
    """
    counts: Counter[tuple[int, int]] = Counter()
    for t, (i, j, k) in enumerate(slice_triangles):
        for a, b in ((i, j), (j, k), (k, i)):
            counts[tuple(sorted((a, b)))] += 1

    lines = []
    for (a, b), count in counts.items():
        if count != 1:
            continue
        p, q = slice_points[a][:2], slice_points[b][:2]
        # Cube wall: both endpoints on the same wall plane x=0/1 or y=0/1.
        on_wall = any(
            abs(p[axis] - q[axis]) < 1e-9
            and (abs(p[axis]) < 1e-9 or abs(p[axis] - 1.0) < 1e-9)
            for axis in (0, 1)
        )
        if on_wall:
            continue
        # Sphere cut-out: the segment's midpoint lies inside some cut disk.
        mid = 0.5 * (p + q)
        in_cut = any(
            (abs(float(c[2]) - z_value) < float(r))
            and (
                np.hypot(mid[0] - c[0], mid[1] - c[1])
                < np.sqrt(float(r) ** 2 - (float(c[2]) - z_value) ** 2) - 1e-9
            )
            for c, r in zip(spheres, radii, strict=True)
        )
        if in_cut:
            continue
        lines.append([p, q])
    return lines

PANELS = [
    # (title, npz path, method used for the slice triangulation)
    (
        "Voronoi partition",
        PROJECT_DIR / "outputs" / "benchmark" / "random_benchmark_fields.npz",
        "classic",
    ),
    (
        "Clearance watershed partition",
        PROJECT_DIR / "outputs" / "ablation_4way" / "watershed" / "random_benchmark_fields.npz",
        "classic",
    ),
]


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    fig.subplots_adjust(
        left=0.05, right=0.985, bottom=0.085, top=0.93, wspace=0.13
    )

    for index, (title, npz_path, method) in enumerate(PANELS):
        data = np.load(npz_path)
        z_value = float(data[f"{method}_error_slice_z"][0])
        slice_points = data[f"{method}_error_slice_points"]
        slice_triangles = data[f"{method}_error_slice_triangles"]
        spheres = data["sphere_centers"]
        radii = data["sphere_radii"]
        vertex_labels = np.asarray(data["cell_labels"])[
            data[f"{method}_error_slice_parent_cells"]
        ]
        n_subdomains = int(len(np.unique(data["cell_labels"])))
        n_interfaces = int(len(data["interface_pairs"]))

        ax = axes[index]
        ax.add_collection(
            LineCollection(
                subdomain_interface_lines(
                    slice_points, slice_triangles, spheres, radii, z_value
                ),
                colors="#1f1f1f", linewidths=1.6, zorder=5,
            )
        )
        slice_sphere_cuts(ax, spheres, radii, z_value, color="#3a3a3a")
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=8)
        ax.set_title(
            f"{title}  ($z$={z_value:.1f}, {n_subdomains} subdomains, "
            f"{n_interfaces} interfaces)",
            fontsize=10,
        )
        ax.text(
            0.5, -0.10, f"({chr(ord('a') + index)})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, fontfamily="serif",
        )

    fig.savefig(OUT, dpi=240)
    plt.close(fig)
    print(f"Done: {OUT}")


if __name__ == "__main__":
    main()

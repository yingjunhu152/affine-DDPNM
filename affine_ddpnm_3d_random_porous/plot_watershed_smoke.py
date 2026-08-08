"""Visualise the watershed smoke outputs (2x2 figure).

Panels:
(a) clearance maxima (maximal balls) and interface centroids in 3-D;
(b) slice z ~ 0.5 through the fluid: basin labels + interface facets;
(c) merge-tree persistence scatter (birth vs death) with the threshold box;
(d) interface area and normal-dispersion histograms.

Reads ``outputs/watershed_smoke/watershed_smoke_fields.npz`` written by
``smoke_watershed_partition.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "watershed_smoke"


def _load() -> dict:
    with np.load(OUT_DIR / "watershed_smoke_fields.npz") as data:
        return {name: data[name] for name in data.files}


def _panel_maxima(ax, data: dict) -> None:
    """(a) Maximal balls + interface centroids."""
    centers = data["sphere_centers"]
    radii = data["sphere_radii"]
    balls = data["maximal_balls"]
    interface_centers = data["interface_centers"]
    from mpl_toolkits.mplot3d import art3d

    for c, r in zip(centers, radii, strict=True):
        ax.scatter(*c, s=8, color="0.65")
        circle = matplotlib.patches.Circle(
            c[:2], r, facecolor="0.85", edgecolor="0.4", linewidth=0.6, zorder=1
        )
        # Equatorial-circle glyphs (flat discs in the plane z = c[2]).
        ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=float(c[2]), zdir="z")
    ax.scatter(
        balls[:, 0], balls[:, 1], balls[:, 2],
        s=30, c="C0", marker="o", label="clearance maxima", zorder=3,
    )
    ax.scatter(
        interface_centers[:, 0], interface_centers[:, 1], interface_centers[:, 2],
        s=12, c="C3", marker="x", label="interface centroids", zorder=3,
    )
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("(a) maximal balls and interfaces")
    ax.legend(loc="upper right", fontsize=7)


def _panel_slice(ax, data: dict) -> None:
    """(b) Basin labels and interface facets on the z ~ 0.5 slice."""
    centers = data["cell_centers"]
    labels = data["cell_labels"]
    z_slice = 0.5
    half = 0.6 * np.median(np.diff(np.unique(centers[:, 2])))
    selected = np.flatnonzero(np.abs(centers[:, 2] - z_slice) <= half)
    n_pores = int(labels.max()) + 1
    scatter = ax.scatter(
        centers[selected, 0], centers[selected, 1],
        c=labels[selected], s=6, cmap="tab20", vmin=-0.5, vmax=n_pores - 0.5,
        linewidths=0,
    )
    # Mark slice cells adjacent to any interface facet (the npz stores facet
    # ids only, so adjacency is rebuilt from the tetrahedra).
    adjacent = _slice_interface_adjacency(data, selected)
    ax.scatter(
        centers[selected[adjacent], 0], centers[selected[adjacent], 1],
        s=3, c="k", linewidths=0, zorder=3,
    )
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"(b) basins on z={z_slice:g} ({n_pores} pores)")
    ax.set_aspect("equal")
    scatter.set_rasterized(True)


def _slice_interface_adjacency(data: dict, selected: np.ndarray) -> np.ndarray:
    """Bool mask over *selected*: cells whose shared facet is an interface."""
    tetrahedra = data["tetrahedra"]
    labels = data["cell_labels"]
    iface_ids = data["facet_interface_ids"]
    # Cell adjacency through facets: find cells sharing a facet whose label
    # differs — cheap loop over tetrahedra edges is not available, so use the
    # facet index arrays reconstructed from the tetrahedra.
    # A facet (sorted vertex triple) shared by two cells is an interface
    # facet when their labels differ; mark both cells on the slice.
    from collections import defaultdict

    facets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for cell, tet in enumerate(tetrahedra):
        for combo in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            key = tuple(sorted(int(tet[i]) for i in combo))
            facets[key].append(cell)
    adjacent = np.zeros(len(selected), dtype=bool)
    for key, cells in facets.items():
        if len(cells) != 2:
            continue
        a, b = cells
        if int(labels[a]) != int(labels[b]):
            pos = np.flatnonzero(selected == a)
            if len(pos):
                adjacent[pos[0]] = True
    return adjacent


def _panel_persistence(ax, data: dict) -> None:
    """(c) Merge-tree persistence: birth vs death per component."""
    table = data["persistence_table"]
    birth = table[:, 0]
    death = table[:, 1]
    persistence = table[:, 2]
    survivors = death >= birth  # top components: death == min clearance
    scatter = ax.scatter(
        death[~survivors], birth[~survivors],
        c=persistence[~survivors], s=10, cmap="viridis",
    )
    ax.scatter(death[survivors], birth[survivors], s=10, c="C3", marker="*")
    ax.set_xlabel("death level (saddle)")
    ax.set_ylabel("birth level (peak)")
    ax.set_title("(c) merge-tree persistence")
    ax.axline((0, 0), slope=1, color="0.6", linewidth=0.8)
    fig = ax.figure
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="persistence")


def _panel_interface_stats(ax, data: dict) -> None:
    """(d) Interface areas and normal dispersion."""
    areas = data["interface_areas"]
    dispersion = data["interface_normal_dispersion"]
    ax.hist(areas, bins=24, color="C0", alpha=0.7, label="area")
    ax.set_xlabel("interface area")
    ax.set_ylabel("count", color="C0")
    ax.twinx().hist(
        dispersion, bins=24, color="C3", alpha=0.5, label="normal dispersion"
    )
    ax.set_ylabel("normal dispersion", color="C3")
    ax.set_title("(d) interface areas and normal dispersion")


def main() -> None:
    data = _load()
    fig = plt.figure(figsize=(13, 11))
    ax_a = fig.add_subplot(2, 2, 1, projection="3d")
    ax_b = fig.add_subplot(2, 2, 2)
    ax_c = fig.add_subplot(2, 2, 3)
    ax_d = fig.add_subplot(2, 2, 4)
    _panel_maxima(ax_a, data)
    _panel_slice(ax_b, data)
    _panel_persistence(ax_c, data)
    _panel_interface_stats(ax_d, data)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "watershed_smoke_figure.png", dpi=160)
    print(f"Figure written: {OUT_DIR / 'watershed_smoke_figure.png'}")


if __name__ == "__main__":
    main()

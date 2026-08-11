#!/usr/bin/env python3
"""FEM-only two-phase snapshot frames: water + oil per time step.

Reads the FEM snapshot arrays (``s_FEM_snapshots`` + ``snapshot_times``)
from a completed benchmark npz and renders one frame per snapshot —
left panel: water saturation S_w, right panel: oil saturation S_o = 1 - S_w —
into ``outputs/fem_plots/frames/fem_t{time:05.1f}.png`` plus a GIF.

Run in the FEniCSx environment:
    conda run -n fenicsx --no-capture-output python -u plot_fem_frames.py \
        --data outputs/smoke_twophase3/twophase_fields.npz
"""

from __future__ import annotations

import argparse
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

PLOT_DIR = PROJECT_DIR / "outputs" / "fem_plots"
FRAME_DIR = PLOT_DIR / "frames"
Z_VALUE = 0.5
SW_LIM = (0.2, 0.8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=PROJECT_DIR / "outputs" / "smoke_twophase3" / "twophase_fields.npz")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1, help="take every Nth snapshot")
    args = parser.parse_args()

    data = np.load(args.data)
    points = np.asarray(data["points"], dtype=float)
    tetrahedra = np.asarray(data["tetrahedra"], dtype=np.int64)
    cell_labels = np.asarray(data["cell_labels"], dtype=np.int32)
    spheres = data["sphere_centers"]
    radii = data["sphere_radii"]

    snapshots = np.asarray(data["s_FEM_snapshots"], dtype=float)
    times = np.asarray(data["snapshot_times"], dtype=float)
    n_snap = len(times)

    slice_points, slice_triangles, parent_cells = pse.build_cellwise_slice(
        points, tetrahedra, Z_VALUE
    )
    groups = pse._slice_groups(slice_points, parent_cells)
    vertex_labels = cell_labels[parent_cells]
    X, Y, cell_of_point, usable = pse._classify_slice_grid(
        slice_points, slice_triangles, vertex_labels, Z_VALUE, spheres, radii
    )
    segments = pse._subdomain_boundary_lines(slice_points, slice_triangles, vertex_labels)

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    step = max(1, args.stride)
    n_render = n_snap if args.max_frames <= 0 else min(n_snap, args.max_frames)
    paths: list[Path] = []
    for i in range(0, n_render, step):
        t = float(times[i])
        on_slice = pse.evaluate_p1_on_slice(
            points, tetrahedra, snapshots[i][:, None], slice_points, groups
        )[:, 0]
        s_w = pse._smooth_slice_grid(
            on_slice, slice_points, vertex_labels, X, Y, cell_of_point, usable
        )
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))
        plt.subplots_adjust(left=0.05, right=0.96, bottom=0.08, top=0.88, wspace=0.12)
        for ax, (values, title) in zip(
            axes,
            ((s_w, "water saturation $S_w$"), (1.0 - s_w, "oil saturation $S_o = 1 - S_w$")),
            strict=True,
        ):
            artist = ax.pcolormesh(X, Y, values, cmap="viridis",
                                   vmin=SW_LIM[0], vmax=SW_LIM[1])
            ax.add_collection(
                LineCollection(segments, colors="#1f1f1f", linewidths=1.0, zorder=5)
            )
            ax.set_aspect("equal")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(labelsize=8)
            ax.set_title(title, fontsize=11)
            pse.sci_colorbar(fig, artist, ax)
        fig.suptitle(f"FEM  |  t = {t:5.2f}", fontsize=13)
        out = FRAME_DIR / f"fem_t{t:05.1f}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        paths.append(out)
        if (i + 1) % 5 == 0 or i == n_render - 1:
            print(f"  rendered {i + 1}/{n_render} (t={t:.2f})")

    try:
        import imageio.v2 as imageio

        gif_path = PLOT_DIR / "fem_frames.gif"
        imageio.mimsave(gif_path, [imageio.imread(p) for p in paths], duration=0.45)
        print(f"wrote {gif_path}")
    except Exception as exc:
        print(f"GIF skipped: {exc}")
    print(f"wrote {len(paths)} frames to {FRAME_DIR}")


if __name__ == "__main__":
    main()

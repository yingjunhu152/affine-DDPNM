#!/usr/bin/env python3
"""Two-phase snapshot frames: water and oil saturation fields per method.

Reads the snapshot arrays written by the benchmark
(``outputs/benchmark_twophase/twophase_fields.npz``:
``s_*_snapshots`` (n_snap, n_vertices) + ``snapshot_times``) and renders
one frame per snapshot:

- row 1: water saturation ``S_w`` (FEM / Classic-1 / W1n-3 / Affine-9);
- row 2: oil saturation ``S_o = 1 - S_w`` for the same methods.

Each panel is the z = 0.5 slice of the P1 vertex field, resampled on the
regular grid with the archived masked per-subdomain smoothing, sphere
cross-sections cut out and Voronoi interface traces drawn.  Frames are
written to ``outputs/benchmark_twophase/frames/twophase_t{time:05.1f}.png``
plus an animated GIF.

Run in the FEniCSx environment (needs the benchmark output only):

    conda run -n fenicsx --no-capture-output python -u plot_twophase_frames.py
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

import plot_slice_errors as pse  # shared slice pipeline helpers

OUT_DIR = PROJECT_DIR / "outputs" / "smoke_twophase3"
FRAME_DIR = OUT_DIR / "frames"  # default; overridden by --out-dir
METHODS = ("FEM", "Classic-DDPNM-1", "NormalLinear-DDPNM-3", "Affine-DDPNM-9")
NPZ_KEY = {
    "FEM": "FEM",
    "Classic-DDPNM-1": "Classic_DDPNM_1",
    "NormalLinear-DDPNM-3": "NormalLinear_DDPNM_3",
    "Affine-DDPNM-9": "Affine_DDPNM_9",
}
Z_VALUE = 0.5
SW_LIM = (0.2, 0.8)  # physical saturation interval [Swr, 1-Sor]


def render_frame(
    X, Y, cell_of_point, usable, boundary_segments,
    sat_fields: dict[str, np.ndarray], t: float, pvi: float,
    out: Path,
) -> None:
    """One frame: water saturation (row 1) and oil saturation (row 2)."""
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 7.2))
    plt.subplots_adjust(
        left=0.035, right=0.985, bottom=0.075, top=0.92, wspace=0.10, hspace=0.24
    )
    for row, phase in enumerate(("water", "oil")):
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            s_w = sat_fields[method]
            values = s_w if phase == "water" else 1.0 - s_w
            artist = ax.pcolormesh(X, Y, values, cmap="viridis", vmin=SW_LIM[0], vmax=SW_LIM[1])
            ax.add_collection(
                LineCollection(boundary_segments, colors="#1f1f1f", linewidths=1.0, zorder=5)
            )
            ax.set_aspect("equal")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(method, fontsize=10)
            if col == 0:
                ax.set_ylabel("oil" if row else "water", fontsize=10, rotation=90)
            if row == 1 and col == 3:
                pse.sci_colorbar(fig, artist, ax)
    fig.suptitle(f"t = {t:5.2f}   (injected PVI = {pvi:5.3f})", fontsize=12)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="render only the first N frames (0 = all)")
    args = parser.parse_args()
    out_dir = args.out_dir
    frame_dir = out_dir / "frames"

    data = np.load(out_dir / "twophase_fields.npz")
    points = np.asarray(data["points"], dtype=float)
    tetrahedra = np.asarray(data["tetrahedra"], dtype=np.int64)
    cell_labels = np.asarray(data["cell_labels"], dtype=np.int32)
    spheres = data["sphere_centers"]
    radii = data["sphere_radii"]

    snapshots = {m: np.asarray(data[f"s_{NPZ_KEY[m]}_snapshots"], dtype=float) for m in METHODS}
    times = np.asarray(data["snapshot_times"], dtype=float)
    n_snap = len(times)
    print(f"snapshots: {n_snap} frames at times {times[0]:.2f} .. {times[-1]:.2f}")

    slice_points, slice_triangles, parent_cells = pse.build_cellwise_slice(
        points, tetrahedra, Z_VALUE
    )
    groups = pse._slice_groups(slice_points, parent_cells)
    vertex_labels = cell_labels[parent_cells]
    X, Y, cell_of_point, usable = pse._classify_slice_grid(
        slice_points, slice_triangles, vertex_labels, Z_VALUE, spheres, radii
    )
    boundary_segments = pse._subdomain_boundary_lines(
        slice_points, slice_triangles, vertex_labels
    )

    frame_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    # injected pore volumes from the FEM history (twophase_history.csv)
    import csv

    with (out_dir / "twophase_history.csv").open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        pvi_by_time = {}
        for row in reader:
            if row["method"] == "FEM":
                pvi_by_time[float(row["time"])] = float(row["pore_volumes_injected"])
    n_render = n_snap if args.max_frames <= 0 else min(n_snap, args.max_frames)
    for i in range(n_render):
        t = float(times[i])
        sat_fields = {
            m: pse._smooth_slice_grid(
                pse.evaluate_p1_on_slice(
                    points, tetrahedra, snapshots[m][i][:, None], slice_points, groups
                )[:, 0],
                slice_points, vertex_labels, X, Y, cell_of_point, usable,
            )
            for m in METHODS
        }
        out = frame_dir / f"twophase_t{t:05.1f}.png"
        render_frame(
            X, Y, cell_of_point, usable, boundary_segments,
            sat_fields, t, pvi_by_time.get(t, float("nan")), out,
        )
        paths.append(out)
        if (i + 1) % 5 == 0 or i == n_render - 1:
            print(f"  rendered {i + 1}/{n_render} frames (t={t:.2f})")
    print(f"wrote {len(paths)} frames to {frame_dir}")

    # animated GIF
    try:
        import imageio.v2 as imageio

        gif_path = out_dir / "twophase_frames.gif"
        imageio.mimsave(gif_path, [imageio.imread(p) for p in paths], duration=0.45)
        print(f"wrote {gif_path}")
    except Exception as exc:  # imageio optional
        print(f"GIF skipped: {exc}")


if __name__ == "__main__":
    main()

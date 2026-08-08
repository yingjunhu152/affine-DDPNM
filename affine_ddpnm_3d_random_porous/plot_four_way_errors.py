#!/usr/bin/env python3
"""Four-way (partition x basis) slice speed panels, 01-style.

2x4 panels at z ~ 0.5 through the fluid, styled after
``outputs/benchmark/01_slice_error_fields.png`` (plot_random_errors.py):

- top row: reduced DDPNM speed |u| for the four combinations
  (Voronoi x {Classic, Affine}, watershed x {Classic, Affine}),
  viridis, one shared color scale (same physical flow);
- bottom row: absolute speed error ||u_ddpnm| - |u_fem||, turbo,
  per-panel 98th-percentile scale (classic errors ~5-10x affine);

fields are resampled on a regular grid and smoothed *inside* each
subdomain only, so the fields read smooth within a pore region while
the jump across interfaces is kept; interface traces (subdomain
boundary lines) and sphere cut-outs are drawn on every panel.

Data: outputs/benchmark/random_benchmark_fields.npz (frozen Voronoi)
and outputs/ablation_4way/watershed/random_benchmark_fields.npz.

Output: outputs/ablation_4way/four_way_speed_error.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from plot_random_errors import (
    _classify_slice_grid,
    _smooth_slice_grid,
    sci_colorbar,
    slice_sphere_cuts,
)
from plot_partition_slice import subdomain_interface_lines

PROJECT_DIR = Path(__file__).resolve().parent
N_GRID = 500
OUT = PROJECT_DIR / "outputs" / "ablation_4way" / "four_way_speed_error.png"

COMBOS = [
    # (drawing, basis, method key, report dir)
    ("Voronoi", "Classic", "classic", PROJECT_DIR / "outputs" / "benchmark"),
    ("Voronoi", "Affine", "affine", PROJECT_DIR / "outputs" / "benchmark"),
    ("Watershed", "Classic", "classic", PROJECT_DIR / "outputs" / "ablation_4way" / "watershed"),
    ("Watershed", "Affine", "affine", PROJECT_DIR / "outputs" / "ablation_4way" / "watershed"),
]


def report_l2(report_dir: Path, method: str) -> float:
    """Relative L2 speed error of the method on this mesh."""
    with (report_dir / "random_affine_report.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    key = "Classic-DDPNM-1" if method == "classic" else "Affine-DDPNM-9"
    return float(report["methods"][key]["velocity_relative_l2"])


def _load_combo(method: str, report_dir: Path) -> dict:
    data = np.load(report_dir / "random_benchmark_fields.npz")
    z_value = float(data[f"{method}_error_slice_z"][0])
    slice_points = data[f"{method}_error_slice_points"]
    slice_triangles = data[f"{method}_error_slice_triangles"]
    vertex_labels = np.asarray(data["cell_labels"])[
        data[f"{method}_error_slice_parent_cells"]
    ]
    speed_ref = np.linalg.norm(data[f"{method}_error_slice_u_fem"], axis=1)
    speed_dd = np.linalg.norm(data[f"{method}_error_slice_u_ddpnm"], axis=1)
    return {
        "z": z_value,
        "slice_points": slice_points,
        "slice_triangles": slice_triangles,
        "vertex_labels": vertex_labels,
        "speed_ref": speed_ref,
        "speed_dd": speed_dd,
        "spheres": data["sphere_centers"],
        "radii": data["sphere_radii"],
    }


def main() -> None:
    combos = [
        {
            "drawing": drawing,
            "basis": basis,
            "l2": report_l2(report_dir, method),
            "field": _load_combo(method, report_dir),
        }
        for drawing, basis, method, report_dir in COMBOS
    ]
    # One shared speed scale for the top row (same physical flow).
    speed_max = max(
        float(np.max(c["field"]["speed_ref"])) for c in combos
    )

    fig, axes = plt.subplots(2, 4, figsize=(17.0, 8.6))
    fig.subplots_adjust(
        left=0.045, right=0.97, bottom=0.075, top=0.965, wspace=0.27, hspace=0.20
    )

    for index, combo in enumerate(combos):
        field = combo["field"]
        X, Y, cell_of_point, usable = _classify_slice_grid(
            field["slice_points"], field["slice_triangles"], field["vertex_labels"],
            field["z"], field["spheres"], field["radii"], n=N_GRID,
        )
        grid_speed = _smooth_slice_grid(
            field["speed_dd"], field["slice_points"], field["vertex_labels"],
            X, Y, cell_of_point, usable, n=N_GRID,
        )
        error = np.abs(field["speed_dd"] - field["speed_ref"])
        grid_err = _smooth_slice_grid(
            error, field["slice_points"], field["vertex_labels"],
            X, Y, cell_of_point, usable, n=N_GRID,
        )
        err_limit = float(np.nanpercentile(grid_err, 98.0))

        title_top = f"{combo['drawing']} × {combo['basis']}"
        title_bot = (
            f"{combo['drawing']} × {combo['basis']}"
            f"  (rel. $L^2$ {combo['l2']:.1%})"
        )

        for row, (values, cmap, vmin, vmax, title) in enumerate(
            (
                (grid_speed, "viridis", 0.0, speed_max, title_top),
                (grid_err, "turbo", 0.0, err_limit, title_bot),
            )
        ):
            ax = axes[row, index]
            artist = ax.pcolormesh(X, Y, values, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.add_collection(
                LineCollection(
                    subdomain_interface_lines(
                        field["slice_points"], field["slice_triangles"],
                        field["spheres"], field["radii"], field["z"],
                    ),
                    colors="#1f1f1f", linewidths=1.3, zorder=5,
                )
            )
            slice_sphere_cuts(ax, field["spheres"], field["radii"], field["z"],
                              color="#3a3a3a")
            ax.set_aspect("equal")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(labelsize=8)
            ax.set_title(title, fontsize=10)
            ax.text(
                0.5, -0.075, f"({chr(ord('a') + row * 4 + index)})",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=12, fontfamily="serif",
            )
            sci_colorbar(fig, artist, ax, signed=False)

    fig.text(
        0.012, 0.76, "Reduced DDPNM speed $|u|$",
        rotation=90, va="center", ha="center", fontsize=11, fontstyle="italic",
    )
    fig.text(
        0.012, 0.28, "Speed errors vs FEM",
        rotation=90, va="center", ha="center", fontsize=11, fontstyle="italic",
    )
    fig.savefig(OUT, dpi=240)
    plt.close(fig)
    print(f"Done: {OUT}")


if __name__ == "__main__":
    main()

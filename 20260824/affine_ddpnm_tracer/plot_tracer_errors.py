#!/usr/bin/env python3
"""Error-field visualization for the affine-DDPNM tracer run.

Reads the saved run outputs (no benchmark rerun needed) and renders:

1. ``outputs/benchmark_tracer/error_field_cloud.png`` — spatial error
   fields on the tetrahedral cloud, 2 rows x 3 methods:

   - row 1: ``log10 |u_ddpnm - u_fem|``  (velocity error)
   - row 2: ``log10 |c_ddpnm - c_fem|``  (tracer concentration error)

   magma colormap, per-panel percentile color range, identical camera.

2. ``outputs/benchmark_tracer/tracer_error_summary.png`` — regenerates the
   error comparison chart with the readable y-axis (machine-precision
   mass-balance series removed, ylim tightened).

Run in the FEniCSx environment:

    conda run -n fenicsx --no-capture-output python -u plot_tracer_errors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import tracer_transport as tracer

OUT_DIR = PROJECT_DIR / "outputs" / "benchmark_tracer"

METHODS = ("Classic-DDPNM-1", "NormalLinear-DDPNM-3", "Affine-DDPNM-9")
NPZ_KEY = {
    "Classic-DDPNM-1": "Classic_DDPNM_1",
    "NormalLinear-DDPNM-3": "NormalLinear_DDPNM_3",
    "Affine-DDPNM-9": "Affine_DDPNM_9",
}


def log10_range(values: np.ndarray, floor: float) -> tuple[float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if len(finite) == 0:
        return np.log10(floor), np.log10(floor) + 2.0
    lo = float(np.percentile(finite, 5.0))
    hi = float(np.percentile(finite, 99.2))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def render_error_cloud(
    grid,
    velocity_error: dict[str, np.ndarray],
    concentration_error: dict[str, np.ndarray],
    out: Path,
) -> None:
    pv.global_theme.font.family = "arial"
    plotter = pv.Plotter(
        off_screen=True, window_size=(640 * 3, 720 * 2), shape=(2, 3), border=False
    )
    plotter.set_background("white")

    panels = [
        ("velocity", "log10 |Δu|", velocity_error, "magma", 1.0e-12),
        ("concentration", "log10 |Δc|", concentration_error, "magma", 1.0e-14),
    ]
    for row, (label, scalar_title, errors, cmap, floor) in enumerate(panels):
        for col, method in enumerate(METHODS):
            plotter.subplot(row, col)
            values = errors[method]
            log_values = np.log10(np.maximum(values, floor))
            clim = log10_range(log_values, floor)
            plotter.add_mesh(
                grid,
                scalars=log_values,
                cmap=cmap,
                clim=clim,
                opacity=0.72,
                smooth_shading=False,
                show_edges=False,
                scalar_bar_args={
                    "title": scalar_title,
                    "vertical": True,
                    "position_x": 0.88,
                    "position_y": 0.18,
                    "width": 0.045,
                    "height": 0.60,
                    "title_font_size": 15,
                    "label_font_size": 12,
                    "fmt": "%.1f",
                    "color": "black",
                },
            )
            plotter.add_text(
                f"{method} {label} error",
                position=(0.035, 0.925),
                font_size=14,
                color="black",
                viewport=True,
            )
            tracer.set_tracer_paper_camera(plotter, grid.bounds)

    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()


def main() -> None:
    data = np.load(OUT_DIR / "tracer_velocity_fields.npz")
    points = np.asarray(data["points"], dtype=float)
    tetrahedra = np.asarray(data["tetrahedra"], dtype=np.int64)
    u_fem = np.asarray(data["u_FEM"], dtype=float)
    c_fem = np.asarray(data["c_FEM"], dtype=float)

    velocity_error = {
        method: np.linalg.norm(np.asarray(data[f"u_{NPZ_KEY[method]}"], dtype=float) - u_fem, axis=1)
        for method in METHODS
    }
    concentration_error = {
        method: np.abs(np.asarray(data[f"c_{NPZ_KEY[method]}"], dtype=float) - c_fem)
        for method in METHODS
    }

    cells = np.hstack(
        [np.full((len(tetrahedra), 1), 4, dtype=np.int64), tetrahedra]
    ).ravel()
    celltypes = np.full(len(tetrahedra), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, points)

    render_error_cloud(grid, velocity_error, concentration_error, OUT_DIR / "error_field_cloud.png")

    report = json.loads((OUT_DIR / "affine_ddpnm_tracer_report.json").read_text(encoding="utf-8"))
    tracer.plot_error_summary(OUT_DIR / "tracer_error_summary.png", report["tracer_metrics"])

    for method in METHODS:
        print(
            f"{method}: velocity error max={np.max(velocity_error[method]):.3e}, "
            f"median={np.median(velocity_error[method]):.3e}; "
            f"concentration error max={np.max(concentration_error[method]):.3e}, "
            f"median={np.median(concentration_error[method]):.3e}"
        )
    print(f"wrote {OUT_DIR / 'error_field_cloud.png'}")
    print(f"wrote {OUT_DIR / 'tracer_error_summary.png'}")


if __name__ == "__main__":
    main()

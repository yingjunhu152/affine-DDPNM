from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.tri as mtri
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-style 2x2 DD-PNM error fields.")
    parser.add_argument(
        "--fields", type=Path, default=Path("outputs/default/ddpnm_2d_fields.npz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/default/05_paper_style_error_fields.png")
    )
    return parser.parse_args()


def sci_colorbar(fig, artist, ax, signed: bool = False):
    cb = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.025)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    cb.formatter = formatter
    cb.update_ticks()
    cb.ax.tick_params(labelsize=9, length=2.5)
    return cb


def main() -> None:
    args = parse_args()
    data = np.load(args.fields)
    points = data["points"]
    triangles = data["triangles"]
    u_dd = data["u_ddpnm"]
    p_dd = data["p_ddpnm_mean_aligned"]
    u_ref = data["u_reference"]
    p_ref = data["p_reference"]
    tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    speed_ref = np.linalg.norm(u_ref, axis=1)
    speed_dd = np.linalg.norm(u_dd, axis=1)
    speed_error = np.abs(speed_dd - speed_ref)
    pressure_error = p_dd - p_ref
    pressure_limit = max(float(np.max(np.abs(pressure_error))), 1.0e-15)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 9.0))
    plt.subplots_adjust(left=0.035, right=0.965, bottom=0.055, top=0.985, wspace=0.18, hspace=0.13)
    panels = [
        (speed_ref, "jet", None, None),
        (p_ref, "jet", 0.0, 1.0),
        (speed_error, "jet", 0.0, None),
        (pressure_error, "jet", -pressure_limit, pressure_limit),
    ]
    for index, (ax, (values, cmap, vmin, vmax)) in enumerate(zip(axes.ravel(), panels, strict=True)):
        artist = ax.tripcolor(
            tri, values, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax,
            rasterized=True,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        sci_colorbar(fig, artist, ax, signed=(index == 3))
        ax.text(
            0.5, -0.055, f"({chr(ord('a') + index)})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=14, fontfamily="serif",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=260, facecolor="white")
    plt.close(fig)
    print(args.output.resolve())
    print(f"max | |u_DD|-|u_FE| | = {np.max(speed_error):.6e}")
    print(f"pressure error range = [{np.min(pressure_error):.6e}, {np.max(pressure_error):.6e}]")


if __name__ == "__main__":
    main()

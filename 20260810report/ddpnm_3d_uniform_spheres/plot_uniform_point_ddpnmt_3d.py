#!/usr/bin/env python3
"""Paper-style figures for the uniform interface point experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import matplotlib.tri as mtri
import numpy as np


COLORS = ("#7a7a7a", "#4c78a8", "#e45756", "#72b7b2", "#f2a541", "#54a24b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/uniform_point_ddpnmt")
    )
    parser.add_argument(
        "--affine-labels",
        action="store_true",
        help="label the two uniform spaces as affine-complete",
    )
    return parser.parse_args()


def style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="0.88", linewidth=0.7, zorder=0)


def plot_interface_points(data, out_dir: Path) -> None:
    xy = data["representative_face_xy"]
    triangles = data["representative_triangles"]
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)
    cross_nodes = data["representative_cross_nodes"]
    cross_edges = data["representative_cross_edges"]
    uniform_nodes = data["representative_uniform_nodes"]
    interface_id = int(data["representative_interface_id"])

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    for axis in axes:
        axis.triplot(triangulation, color="0.74", linewidth=0.55, zorder=1)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    for first, second in cross_edges:
        axes[0].plot(
            xy[[first, second], 0], xy[[first, second], 1],
            color="#2b6cb0", linewidth=2.0, zorder=2,
        )
    axes[0].scatter(
        xy[cross_nodes, 0], xy[cross_nodes, 1], s=28,
        color="#d62728", edgecolor="white", linewidth=0.55, zorder=3,
    )
    axes[0].set_title(f"(a) Cross nodes: $N_s={len(cross_nodes)}$")

    axes[1].scatter(
        xy[uniform_nodes, 0], xy[uniform_nodes, 1], s=48,
        color="#f2a541", edgecolor="#6b3d00", linewidth=0.65, zorder=3,
    )
    axes[1].scatter(
        xy[uniform_nodes[0], 0], xy[uniform_nodes[0], 1], s=75,
        marker="*", color="#d62728", edgecolor="white", linewidth=0.55, zorder=4,
    )
    axes[1].set_title(f"(b) Geodesic FPS: $N_s={len(uniform_nodes)}$")
    figure.suptitle(
        f"Representative throat interface {interface_id}: same triangular mesh",
        fontsize=12,
    )
    legend = [
        Line2D([0], [0], marker="*", linestyle="none", color="#d62728", label="saddle seed"),
        Line2D([0], [0], marker="o", linestyle="none", color="#f2a541", label="FPS controls"),
    ]
    axes[1].legend(handles=legend, loc="best", frameon=False, fontsize=8)
    figure.savefig(out_dir / "01_cross_vs_uniform_interface_points.png", dpi=260)
    plt.close(figure)


def _display_names(data, affine_labels: bool) -> list[str]:
    names = [str(value) for value in data["method_names"]]
    if affine_labels:
        return [name.replace("Uniform-", "Affine-Uniform-") for name in names]
    return names


def plot_errors(data, out_dir: Path, affine_labels: bool = False) -> None:
    names = _display_names(data, affine_labels)
    errors = 100.0 * data["error_matrix"]
    titles = (
        r"Velocity relative $L^2$ error",
        r"Velocity broken-$H^1$ error",
        r"Pressure raw relative $L^2$ error",
        "Outlet flux relative error",
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    x = np.arange(len(names))
    for metric, (axis, title) in enumerate(zip(axes.ravel(), titles, strict=True)):
        bars = axis.bar(x, errors[:, metric], color=COLORS, width=0.74, zorder=2)
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5)
        axis.set_title(title)
        axis.set_ylabel("Relative error (%)")
        axis.set_xticks(x, names, rotation=24, ha="right", fontsize=8)
        axis.set_ylim(0.0, max(errors[:, metric]) * 1.18)
        style_axis(axis)
    figure.suptitle("Identical-mesh comparison against monolithic Taylor–Hood FEM")
    figure.savefig(out_dir / "02_six_method_error_comparison.png", dpi=260)
    plt.close(figure)


def plot_accuracy_cost(data, out_dir: Path, affine_labels: bool = False) -> None:
    names = _display_names(data, affine_labels)
    errors = 100.0 * data["error_matrix"]
    dofs = data["interface_unknowns"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.1), constrained_layout=True)
    labels = (r"Velocity relative $L^2$ error (%)", "Outlet flux relative error (%)")
    for metric, (axis, ylabel) in zip((0, 3), zip(axes, labels, strict=True), strict=True):
        for index, name in enumerate(names):
            axis.scatter(
                dofs[index], errors[index, metric], s=72, color=COLORS[index],
                edgecolor="white", linewidth=0.7, zorder=3, label=name,
            )
            axis.annotate(
                name.replace("Uniform-", "U-").replace("Cross-", "C-"),
                (dofs[index], errors[index, metric]), xytext=(4, 4),
                textcoords="offset points", fontsize=7,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Global interface unknowns (log scale)")
        axis.set_ylabel(ylabel)
        style_axis(axis)
    axes[0].set_title("(a) Velocity accuracy versus interface cost")
    axes[1].set_title("(b) Flux accuracy versus interface cost")
    figure.savefig(out_dir / "03_accuracy_vs_interface_unknowns.png", dpi=260)
    plt.close(figure)


def draw_sphere_sections(axis, centers, radius: float, z_value: float) -> None:
    for center in centers:
        distance = abs(float(center[2] - z_value))
        if distance >= radius:
            continue
        section_radius = np.sqrt(radius**2 - distance**2)
        axis.add_patch(
            plt.Circle(
                center[:2], section_radius, facecolor="white",
                edgecolor="none", zorder=5,
            )
        )


def plot_local_error_fields(
    data, out_dir: Path, affine_labels: bool = False
) -> None:
    """One four-panel FEM/method/error figure for every reduced space."""
    raw_names = [str(value) for value in data["method_names"]]
    names = _display_names(data, affine_labels)
    safe_names = [name.lower().replace("-", "_") for name in raw_names]
    points = data["error_slice_points"]
    triangles = data["error_slice_triangles"]
    u_fem = data["error_slice_u_fem"]
    p_fem = data["error_slice_p_fem"]
    z_value = float(data["error_slice_z"][0])
    centers = data["sphere_centers"]
    radius = float(data["sphere_radius"][0])
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    method_velocities = [data[f"error_slice_u_{safe}"] for safe in safe_names]
    method_pressures = [data[f"error_slice_p_{safe}"] for safe in safe_names]
    velocity_scale = max(
        float(np.max(np.linalg.norm(u_fem, axis=1))),
        *(float(np.max(np.linalg.norm(field, axis=1))) for field in method_velocities),
    )
    velocity_error_scale = max(
        float(np.max(np.linalg.norm(field - u_fem, axis=1)))
        for field in method_velocities
    )
    pressure_error_scale = max(
        float(np.max(np.abs(field - p_fem))) for field in method_pressures
    )
    cell_maxima = data["maximum_cell_velocity_rms_error"]
    slice_maxima = data["maximum_slice_pointwise_velocity_error"]
    mass_maxima = data["maximum_local_mass_residual"]

    for method_index, (name, u_method, p_method) in enumerate(
        zip(names, method_velocities, method_pressures, strict=True)
    ):
        velocity_error = np.linalg.norm(u_method - u_fem, axis=1)
        pressure_error = p_method - p_fem
        fields = (
            (
                np.linalg.norm(u_fem, axis=1),
                r"FEM velocity magnitude $|u_h^{FE}|$",
                "turbo", 0.0, velocity_scale, None,
            ),
            (
                np.linalg.norm(u_method, axis=1),
                rf"{name} velocity magnitude $|u_h^{{R}}|$",
                "turbo", 0.0, velocity_scale, None,
            ),
            (
                velocity_error,
                r"Local velocity error $|u_h^{R}-u_h^{FE}|$",
                "turbo", 0.0, velocity_error_scale, None,
            ),
            (
                pressure_error,
                r"Local pressure error $p_h^{R}-p_h^{FE}$",
                "RdBu_r", None, None,
                TwoSlopeNorm(
                    vmin=-pressure_error_scale,
                    vcenter=0.0,
                    vmax=pressure_error_scale,
                ),
            ),
        )
        figure, axes = plt.subplots(
            2, 2, figsize=(12.6, 10.1), constrained_layout=True
        )
        for panel, (axis, field_spec) in enumerate(
            zip(axes.flat, fields, strict=True)
        ):
            field, title, cmap, vmin, vmax, norm = field_spec
            artist = axis.tripcolor(
                triangulation,
                field,
                shading="gouraud",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                rasterized=True,
            )
            draw_sphere_sections(axis, centers, radius, z_value)
            axis.set(
                xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[]
            )
            axis.set_title(title, fontsize=12.0)
            axis.text(
                0.5, -0.055, f"({chr(97 + panel)})",
                transform=axis.transAxes, ha="center", va="top", fontsize=12.5,
            )
            colorbar = figure.colorbar(artist, ax=axis, shrink=0.88, pad=0.025)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 2))
            colorbar.formatter = formatter
            colorbar.update_ticks()
        figure.suptitle(
            f"{name} local error on the identical tetrahedral mesh, z={z_value:.2f}\n"
            f"max cell RMS |du|={cell_maxima[method_index]:.3e}; "
            f"max slice |du|={slice_maxima[method_index]:.3e}; "
            f"max local mass residual={mass_maxima[method_index]:.3e}",
            fontsize=13,
        )
        figure.savefig(
            out_dir / f"04_{method_index + 1:02d}_{safe_names[method_index]}_local_errors.png",
            dpi=260,
        )
        plt.close(figure)


def main() -> None:
    args = parse_args()
    path = args.out_dir / "uniform_point_ddpnmt_results.npz"
    with np.load(path, allow_pickle=False) as data:
        plot_interface_points(data, args.out_dir)
        plot_errors(data, args.out_dir, affine_labels=args.affine_labels)
        plot_accuracy_cost(data, args.out_dir, affine_labels=args.affine_labels)
        plot_local_error_fields(
            data, args.out_dir, affine_labels=args.affine_labels
        )
    print(f"Figures written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

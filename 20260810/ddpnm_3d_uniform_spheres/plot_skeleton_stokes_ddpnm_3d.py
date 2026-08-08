#!/usr/bin/env python3
"""Paper-style audit figure for the cross-skeleton Stokes extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/skeleton_stokes_ddpnm/skeleton_stokes_ddpnm_results.npz"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/skeleton_stokes_ddpnm")
    )
    return parser.parse_args()


def tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.eye(3)
    reference = candidates[int(np.argmin(np.abs(candidates @ normal)))]
    first = reference - float(reference @ normal) * normal
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    second /= np.linalg.norm(second)
    return first, second


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.input)
    representative = int(data["representative_interface"][0])
    saddle = data["representative_saddle"]
    tangent_1 = data["representative_tangent_1"]
    tangent_2 = data["representative_tangent_2"]
    coordinates = data["representative_coordinates"]
    relative = coordinates - saddle
    planar = np.column_stack((relative @ tangent_1, relative @ tangent_2))
    triangles = data["representative_triangles"]
    triangulation = mtri.Triangulation(planar[:, 0], planar[:, 1], triangles)
    skeleton_nodes = data["representative_skeleton_nodes"]
    center_node = int(data["representative_center_node"][0])
    normal_extension = data["representative_extension_normal"][:, 0]
    endpoint_extension = data["representative_extension_endpoint"][:, 0]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
        }
    )
    figure = plt.figure(figsize=(12.2, 8.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)

    axis_3d = figure.add_subplot(grid[0, 0], projection="3d")
    segments = data["skeleton_segments"]
    segment_interfaces = data["skeleton_segment_interfaces"]
    for segment, interface_id in zip(segments, segment_interfaces, strict=True):
        color = "#d62728" if int(interface_id) == representative else "#2c6db2"
        alpha = 0.95 if int(interface_id) == representative else 0.18
        linewidth = 1.8 if int(interface_id) == representative else 0.55
        axis_3d.plot(*segment.T, color=color, alpha=alpha, linewidth=linewidth)
    saddles = data["throat_saddles"]
    axis_3d.scatter(
        saddles[:, 0], saddles[:, 1], saddles[:, 2],
        s=4, color="#222222", alpha=0.22,
    )
    axis_3d.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xlabel="x", ylabel="y", zlabel="z")
    axis_3d.set_box_aspect((1, 1, 1))
    axis_3d.view_init(elev=23, azim=-52)
    axis_3d.set_title("(a) 144 saddle-centred cross skeletons")

    axis_mesh = figure.add_subplot(grid[0, 1])
    axis_mesh.triplot(triangulation, color="#a9adb3", linewidth=0.55)
    representative_segments = segments[segment_interfaces == representative]
    for segment in representative_segments:
        projected = np.column_stack(
            ((segment - saddle) @ tangent_1, (segment - saddle) @ tangent_2)
        )
        axis_mesh.plot(
            projected[:, 0], projected[:, 1], color="#ef3b2c", linewidth=2.0,
            zorder=3,
        )
    axis_mesh.scatter(
        planar[skeleton_nodes, 0], planar[skeleton_nodes, 1],
        s=24, facecolor="#ef3b2c", edgecolor="white", linewidth=0.45, zorder=4,
        label="cross dofs",
    )
    axis_mesh.scatter(
        planar[center_node, 0], planar[center_node, 1],
        s=70, marker="*", facecolor="#111111", edgecolor="white", linewidth=0.5,
        zorder=5, label="nearest saddle node",
    )
    axis_mesh.set_aspect("equal")
    axis_mesh.set_title(f"(b) Interface {representative}: full face and discrete cross")
    axis_mesh.set_xlabel(r"$s$")
    axis_mesh.set_ylabel(r"$t$")
    axis_mesh.legend(frameon=False, loc="best")

    axis_normal = figure.add_subplot(grid[1, 0])
    image_normal = axis_normal.tripcolor(
        triangulation, normal_extension, shading="gouraud", cmap="coolwarm"
    )
    axis_normal.triplot(triangulation, color="black", linewidth=0.16, alpha=0.25)
    axis_normal.scatter(
        planar[skeleton_nodes, 0], planar[skeleton_nodes, 1],
        s=10, facecolor="none", edgecolor="black", linewidth=0.5,
    )
    axis_normal.set_aspect("equal")
    axis_normal.set_title("(c) Unit centre-normal constraint: Stokes-energy extension")
    axis_normal.set_xlabel(r"$s$")
    axis_normal.set_ylabel(r"$t$")
    figure.colorbar(image_normal, ax=axis_normal, shrink=0.86)

    axis_tangent = figure.add_subplot(grid[1, 1])
    image_tangent = axis_tangent.tripcolor(
        triangulation, endpoint_extension, shading="gouraud", cmap="coolwarm"
    )
    axis_tangent.triplot(triangulation, color="black", linewidth=0.16, alpha=0.25)
    axis_tangent.scatter(
        planar[skeleton_nodes, 0], planar[skeleton_nodes, 1],
        s=10, facecolor="none", edgecolor="black", linewidth=0.5,
    )
    axis_tangent.set_aspect("equal")
    axis_tangent.set_title("(d) Endpoint normal fluctuation: Stokes-energy extension")
    axis_tangent.set_xlabel(r"$s$")
    axis_tangent.set_ylabel(r"$t$")
    figure.colorbar(image_tangent, ax=axis_tangent, shrink=0.86)

    output = args.out_dir / "01_cross_skeleton_and_stokes_extension.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)

    report_path = args.out_dir / "skeleton_stokes_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cross = report.get("validation")
        baseline = report.get("ddpnm_baseline_validation")
        if cross is not None and baseline is not None:
            labels = [r"velocity $L^2$", r"broken-$H^1$", "pressure $L^2$", "outlet flux"]
            fields = [
                "velocity_relative_l2",
                "velocity_relative_broken_h1_seminorm",
                "pressure_raw_relative_l2",
                "outlet_flux_relative_error",
            ]
            baseline_values = 100.0 * np.asarray([baseline[field] for field in fields])
            cross_values = 100.0 * np.asarray([cross[field] for field in fields])
            positions = np.arange(len(labels), dtype=float)
            width = 0.36
            figure, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
            first = axis.bar(
                positions - width / 2, baseline_values, width,
                color="#8b8d90", label="DDPNM (one P0 normal dof)",
            )
            second = axis.bar(
                positions + width / 2, cross_values, width,
                color="#2c6db2", label="cardinal cross-normal DDPNM",
            )
            axis.bar_label(first, fmt="%.2f%%", padding=2, fontsize=8)
            axis.bar_label(second, fmt="%.2f%%", padding=2, fontsize=8)
            axis.set_xticks(positions, labels)
            axis.set_ylabel("relative error (%)")
            axis.set_title("Same-mesh comparison against monolithic Taylor--Hood FEM")
            axis.grid(axis="y", alpha=0.22)
            axis.legend(frameon=False)
            figure.savefig(
                args.out_dir / "03_ddpnm_cross_fem_error_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(figure)

    full = data["full_normal_dofs_per_interface"]
    skeleton = data["active_normal_dofs_per_interface"]
    figure, axis = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    axis.scatter(full, skeleton, s=22, alpha=0.68, color="#2c6db2")
    xline = np.linspace(max(float(np.min(full)), 1.0), float(np.max(full)), 200)
    constant = float(np.median(skeleton / np.sqrt(np.maximum(full, 1))))
    axis.plot(xline, constant * np.sqrt(xline), "--", color="#d62728", linewidth=1.5,
              label=rf"$N_C\approx {constant:.2f}\sqrt{{N_f}}$")
    axis.plot(xline, xline, ":", color="#555555", linewidth=1.0, label="$N_C=N_f$")
    axis.set_xlabel("Full-face normal dofs $N_f=O(h^{-2})$")
    axis.set_ylabel("Active cross-normal dofs $N_C=O(h^{-1})$")
    axis.set_title("Interface-space scaling on the 144 throats")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(
        args.out_dir / "02_interface_dof_scaling.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter
import numpy as np


LEVEL_NAMES = ("DDPNM", "DDPNMT", "HODDPNM")
LEVEL_COLORS = ("#3b70b5", "#f2a541", "#d1495b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def chinese_font() -> FontProperties | None:
    path = Path("C:/Windows/Fonts/msyh.ttc")
    return FontProperties(fname=str(path)) if path.exists() else None


def draw_sphere_sections(ax, centers, radius, z_value) -> None:
    for center in centers:
        distance = abs(float(center[2] - z_value))
        if distance >= radius:
            continue
        section_radius = np.sqrt(radius**2 - distance**2)
        ax.add_patch(
            plt.Circle(
                center[:2], section_radius, facecolor="white", edgecolor="none", zorder=5
            )
        )


def plot_convergence(data, out_dir: Path) -> None:
    velocity = data["history_velocity"]
    pressure = data["history_pressure"]
    counts = data["history_counts"]
    unknowns = data["history_unknowns"]
    tolerance = float(data["target_tolerance"][0])
    x = np.arange(len(velocity))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    axes[0].semilogy(x, velocity, "o-", lw=1.8, label="velocity difference")
    axes[0].semilogy(x, pressure, "s-", lw=1.8, label="pressure difference")
    axes[0].axhline(tolerance, color="#b32025", ls="--", lw=1.4, label="TOL = 1%")
    axes[0].set(xlabel="adaptive iteration", ylabel="hierarchical relative difference")
    axes[0].grid(alpha=0.24)
    axes[0].legend(frameon=False, fontsize=9)
    bottom = np.zeros(len(x))
    for level, (name, color) in enumerate(zip(LEVEL_NAMES, LEVEL_COLORS, strict=True)):
        axes[1].bar(x, counts[:, level], bottom=bottom, color=color, label=name)
        bottom += counts[:, level]
    axes[1].set(xlabel="adaptive iteration", ylabel="number of interfaces")
    axes[1].legend(frameon=False, fontsize=9)
    axes[2].plot(x, unknowns, "o-", color="#4b4b4b", lw=1.8)
    axes[2].set(xlabel="adaptive iteration", ylabel="active interface unknowns")
    axes[2].grid(alpha=0.24)
    fig.suptitle("3D two-stage adaptive DDPNM hierarchy", fontsize=14)
    fig.savefig(out_dir / "01_adaptive_convergence.png", dpi=250)
    plt.close(fig)


def plot_final_hierarchy(data, out_dir: Path) -> None:
    balls = data["maximal_ball_centers"]
    pairs = data["interface_pairs"]
    levels = data["final_levels"]
    fig = plt.figure(figsize=(9.2, 8.1), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for interface_id, (first, second) in enumerate(pairs):
        xyz = balls[[first, second]]
        ax.plot(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            color=LEVEL_COLORS[int(levels[interface_id])], lw=3.0, alpha=0.95,
        )
    ax.scatter(balls[:, 0], balls[:, 1], balls[:, 2], s=24, c="#24282d", depthshade=False)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xlabel="x", ylabel="y", zlabel="z")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=23, azim=-57)
    handles = [
        plt.Line2D([0], [0], color=LEVEL_COLORS[i], lw=4, label=LEVEL_NAMES[i])
        for i in range(3)
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True)
    ax.set_title("Final adaptive interface hierarchy on the maximal-ball graph")
    fig.savefig(out_dir / "02_adaptive_final_interface_hierarchy.png", dpi=250)
    plt.close(fig)


def plot_method_errors(report, out_dir: Path) -> None:
    metrics = report["strict_errors_to_identical_mesh_FEM"]
    names = ("DDPNM", "DDPNMT", "HODDPNM", "adaptive_final")
    labels = ("DDPNM", "DDPNMT", "HODDPNM", "Adaptive")
    keys = (
        "velocity_relative_l2",
        "velocity_relative_broken_h1_seminorm",
        "pressure_raw_relative_l2",
        "outlet_flux_relative_error",
    )
    quantity_labels = (r"velocity $L^2$", r"velocity broken-$H^1$", r"pressure $L^2$", "outlet flux")
    colors = ("#2878b5", "#9b59b6", "#f2a541", "#d1495b")
    x = np.arange(len(names))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.2, 5.4), constrained_layout=True)
    for index, (key, label, color) in enumerate(zip(keys, quantity_labels, colors, strict=True)):
        values = [metrics[name][key] for name in names]
        ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("relative error to identical-mesh FEM")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Accuracy gained by 3D interface enrichment and adaptivity")
    fig.savefig(out_dir / "03_method_errors_to_fem.png", dpi=250)
    plt.close(fig)


def plot_adaptive_error_fields(data, out_dir: Path) -> None:
    points = data["error_slice_points"]
    triangles = data["error_slice_triangles"]
    u_fem = data["error_slice_u_fem"]
    p_fem = data["error_slice_p_fem"]
    u_adaptive = data["error_slice_u_ddpnm"]
    p_adaptive = data["error_slice_p_ddpnm"]
    z_value = float(data["error_slice_z"][0])
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    velocity_error = np.linalg.norm(u_adaptive - u_fem, axis=1)
    pressure_error = p_adaptive - p_fem
    pressure_limit = max(float(np.max(np.abs(pressure_error))), 1.0e-14)
    fields = (
        (np.linalg.norm(u_fem, axis=1), r"FEM velocity magnitude $|u_h^{FE}|$", "turbo", None),
        (np.linalg.norm(u_adaptive, axis=1), r"Adaptive velocity magnitude $|u_h^{A}|$", "turbo", None),
        (velocity_error, r"Velocity error $|u_h^{A}-u_h^{FE}|$", "turbo", None),
        (
            pressure_error,
            r"Pressure error $p_h^{A}-p_h^{FE}$",
            "RdBu_r",
            TwoSlopeNorm(vmin=-pressure_limit, vcenter=0.0, vmax=pressure_limit),
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.1), constrained_layout=True)
    centers = data["sphere_centers"]
    radius = float(data["sphere_radius"][0])
    for panel, (ax, (field, title, cmap, norm)) in enumerate(zip(axes.flat, fields, strict=True)):
        artist = ax.tripcolor(
            triangulation, field, shading="gouraud", cmap=cmap, norm=norm, rasterized=True
        )
        draw_sphere_sections(ax, centers, radius, z_value)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
        ax.set_title(title, fontsize=12.2)
        ax.text(0.5, -0.055, f"({chr(97 + panel)})", transform=ax.transAxes, ha="center", va="top", fontsize=12.5)
        colorbar = fig.colorbar(artist, ax=ax, shrink=0.88, pad=0.025)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        colorbar.formatter = formatter
        colorbar.update_ticks()
    fig.suptitle(
        "Traditional FEM and adaptive DDPNM on the identical tetrahedral mesh\n"
        f"cell-sided slice z={z_value:.2f}; no averaging across subdomain interfaces",
        fontsize=13,
    )
    fig.savefig(out_dir / "04_adaptive_fem_error_fields.png", dpi=260)
    plt.close(fig)


def plot_algorithm_box(report, out_dir: Path) -> None:
    tolerance = report["parameters"]["target_tolerance"]
    theta = report["parameters"]["marking_theta"]
    lines = (
        "算法 1　三维分层自适应 DD-PNM",
        "输入：最大球子区域、严格孔喉截面、局部 Taylor–Hood 矩阵、目标容忍值 TOL",
        "1:  对每个子区域仅分解一次局部 Stokes 矩阵，建立 P1 向量牵引响应库",
        "2:  所有界面置为 DDPNM：1 个 P0 法向牵引自由度",
        "3:  以完整 DDPNMT（P0 法向 + 两个 P0 切向）作为第一层比较解",
        "4:  while max(速度层级差, 压力层级差) > TOL do",
        "5:      计算界面层级差指标；按 Dörfler 准则标记；标记界面只升一级",
        "6:      重新装配并求解凝聚后的界面 Schur 系统",
        "7:  end while",
        "8:  以完整 HODDPNM（3 分量 × {1,s,t}，每界面 9 自由度）作为第二层比较解",
        "9:  重复第 4–7 行，直到达到 TOL 或所有界面升至 HODDPNM",
        "10: 重构局部场；整体 FEM 仅用于事后验证，不参与标记",
        f"参数：TOL={tolerance:.3g}，Dörfler θ={theta:.2f}",
    )
    font = chinese_font()
    fig, ax = plt.subplots(figsize=(12.0, 7.2), constrained_layout=True)
    ax.axis("off")
    ax.add_patch(
        plt.Rectangle((0.02, 0.035), 0.96, 0.93, transform=ax.transAxes, facecolor="#fbfbf8", edgecolor="#20252b", lw=1.5)
    )
    ax.text(0.05, 0.925, lines[0], transform=ax.transAxes, va="top", fontsize=15, fontweight="bold", fontproperties=font)
    ax.plot([0.05, 0.95], [0.865, 0.865], transform=ax.transAxes, color="#20252b", lw=0.9)
    ax.text(0.055, 0.825, "\n".join(lines[1:]), transform=ax.transAxes, va="top", fontsize=11.0, linespacing=1.45, fontproperties=font)
    fig.savefig(out_dir / "05_adaptive_algorithm_box.png", dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    report = json.loads(args.report.read_text(encoding="utf-8"))
    plot_convergence(data, args.out_dir)
    plot_final_hierarchy(data, args.out_dir)
    plot_method_errors(report, args.out_dir)
    plot_adaptive_error_fields(data, args.out_dir)
    plot_algorithm_box(report, args.out_dir)
    print(f"Adaptive figures written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

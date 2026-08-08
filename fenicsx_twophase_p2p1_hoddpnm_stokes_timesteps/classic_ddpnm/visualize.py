from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .geometry import Network
from .local_stokes_ho import canonical_tangent_axes


def plot_network(network: Network, out: Path, title: str = "Irregular PNM Network") -> None:
    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    centers = np.asarray([p.center for p in network.pores], dtype=float)
    radii = np.asarray([p.radius for p in network.pores], dtype=float)

    draw_spheres(ax, centers, radii)
    draw_interfaces(ax, network, color="#2f6f9f", linewidth=0.8, alpha=0.62)
    draw_boundary_ports(ax, network)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=8, c="#1f2933", depthshade=True)

    ax.set_title(title)
    set_axes_equal(ax, centers)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-52)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_hoddpnm_pressure_nodes(
    network: Network,
    result_json: Path,
    out: Path,
    title: str = "Normal HODDPNM Interface Node Pressure",
) -> None:
    data = json.loads(result_json.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(11, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    centers = np.asarray([p.center for p in network.pores], dtype=float)
    radii = np.asarray([p.radius for p in network.pores], dtype=float)

    draw_spheres(ax, centers, radii, alpha=0.08)
    draw_interfaces(ax, network, color="#4b5563", linewidth=0.5, alpha=0.32)
    draw_boundary_ports(ax, network)

    points = []
    pressures = []
    for item in data["interface_pressure_dofs"]:
        interface_id = int(item["interface_id"])
        a, b = network.interfaces[interface_id]
        center = 0.5 * (centers[a] + centers[b])
        direction = centers[b] - centers[a]
        direction = direction / np.linalg.norm(direction)
        e1, e2 = canonical_tangent_axes(direction)
        s, t = item["node_key"]
        points.append(center + float(s) * e1 + float(t) * e2)
        pressures.append(float(item["pressure"]))

    pts = np.asarray(points, dtype=float)
    pressures = np.asarray(pressures, dtype=float)
    sc = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=pressures,
        cmap="viridis",
        s=8,
        linewidths=0.0,
        depthshade=False,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.72)
    cbar.set_label("interface pressure")

    ax.set_title(title)
    set_axes_equal(ax, centers)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-52)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_interface_error_cloud(
    network: Network,
    points: np.ndarray,
    errors: np.ndarray,
    out: Path,
    title: str = "HODDPNM - FEM Interface Error",
) -> None:
    fig = plt.figure(figsize=(11, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    centers = np.asarray([p.center for p in network.pores], dtype=float)
    radii = np.asarray([p.radius for p in network.pores], dtype=float)

    draw_spheres(ax, centers, radii, alpha=0.07)
    draw_interfaces(ax, network, color="#4b5563", linewidth=0.45, alpha=0.28)
    draw_boundary_ports(ax, network)

    vmax = float(np.max(np.abs(errors))) if len(errors) else 1.0
    vmax = max(vmax, 1.0e-14)
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=errors,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        s=9,
        linewidths=0.0,
        depthshade=False,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.72)
    cbar.set_label("p_HODDPNM - p_FEM")

    ax.set_title(title)
    set_axes_equal(ax, centers)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-52)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_interface_error_cloud_v2(
    network: Network,
    points: np.ndarray,
    errors: np.ndarray,
    out: Path,
    title: str = "HODDPNM - FEM Error Cloud",
) -> None:
    fig = plt.figure(figsize=(13.5, 7.2), dpi=200)
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    centers = np.asarray([p.center for p in network.pores], dtype=float)
    abs_error = np.abs(errors)
    clip = float(np.percentile(abs_error, 98.0)) if len(abs_error) else 1.0
    clip = max(clip, 1.0e-14)
    sizes = 8.0 + 70.0 * np.sqrt(np.clip(abs_error / clip, 0.0, 1.0))

    draw_interfaces(ax3d, network, color="#1f2937", linewidth=0.5, alpha=0.18)
    ax3d.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        s=12,
        c="#111827",
        alpha=0.62,
        depthshade=False,
    )
    sc = ax3d.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=errors,
        cmap="RdBu_r",
        vmin=-clip,
        vmax=clip,
        s=sizes,
        alpha=0.92,
        linewidths=0.18,
        edgecolors="#f8fafc",
        depthshade=False,
    )
    draw_boundary_ports(ax3d, network)
    ax3d.set_title("3D interface-node error", pad=12)
    set_axes_equal(ax3d, centers)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.view_init(elev=24, azim=-48)
    ax3d.grid(False)
    ax3d.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax3d.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax3d.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))

    for a, b in network.interfaces:
        ax2d.plot(
            centers[[a, b], 0],
            centers[[a, b], 1],
            color="#94a3b8",
            linewidth=0.45,
            alpha=0.34,
            zorder=1,
        )
    ax2d.scatter(
        centers[:, 0],
        centers[:, 1],
        s=12,
        c="#111827",
        alpha=0.55,
        zorder=2,
    )
    ax2d.scatter(
        points[:, 0],
        points[:, 1],
        c=errors,
        cmap="RdBu_r",
        vmin=-clip,
        vmax=clip,
        s=sizes,
        alpha=0.9,
        linewidths=0.2,
        edgecolors="#ffffff",
        zorder=3,
    )
    ax2d.set_title("x-y projection, size = |error|")
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y")
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.grid(True, color="#e5e7eb", linewidth=0.5)
    ax2d.spines["top"].set_visible(False)
    ax2d.spines["right"].set_visible(False)

    cbar = fig.colorbar(sc, ax=[ax3d, ax2d], pad=0.035, shrink=0.8)
    cbar.set_label("p_HODDPNM - p_FEM")
    fig.suptitle(title, y=0.98, fontsize=15)
    fig.text(
        0.52,
        0.035,
        f"Linf={np.max(abs_error):.3e}, mean |e|={np.mean(abs_error):.3e}, color clipped at p98={clip:.3e}",
        ha="center",
        va="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.0, 0.06, 0.96, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def draw_spheres(ax, centers: np.ndarray, radii: np.ndarray, alpha: float = 0.16) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 18)
    v = np.linspace(0.0, np.pi, 10)
    for center, radius in zip(centers, radii):
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color="#8ecae6", alpha=alpha, linewidth=0.0, shade=True)


def draw_interfaces(ax, network: Network, color: str, linewidth: float, alpha: float) -> None:
    centers = np.asarray([p.center for p in network.pores], dtype=float)
    segments = [(centers[a], centers[b]) for a, b in network.interfaces]
    collection = Line3DCollection(segments, colors=color, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(collection)


def draw_boundary_ports(ax, network: Network) -> None:
    for pore in network.pores:
        center = np.asarray(pore.center, dtype=float)
        for port in pore.ports:
            if port.kind not in {"inlet", "outlet"}:
                continue
            normal = np.asarray(port.normal, dtype=float)
            color = "#d7263d" if port.kind == "inlet" else "#2a9d8f"
            ax.quiver(
                center[0],
                center[1],
                center[2],
                0.32 * normal[0],
                0.32 * normal[1],
                0.32 * normal[2],
                color=color,
                linewidth=1.0,
                arrow_length_ratio=0.25,
            )


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

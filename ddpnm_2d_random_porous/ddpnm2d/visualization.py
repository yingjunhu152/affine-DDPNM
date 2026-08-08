"""Two-dimensional DD-PNM matplotlib visualization helpers."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.tri as mtri
import numpy as np
from basix.ufl import element
from dolfinx import fem, io
from mpi4py import MPI
from ddpnm_core.io import topology_arrays, assign_p1_function
from .geometry import PARTICLES
from .solver import DdpnmSolution


def _coordinate_lookup(coords: np.ndarray) -> dict[tuple[float, float], int]:
    return {
        (round(float(x), 12), round(float(y), 12)): i
        for i, (x, y) in enumerate(coords[:, :2])
    }


def interface_segments(partition: PartitionData) -> list[np.ndarray]:
    msh = partition.mesh
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    segments = []
    for facet, interface_id in enumerate(partition.facet_interface_ids):
        if interface_id >= 0:
            segments.append(msh.geometry.x[f2v.links(facet), :2])
    return segments


def unique_mesh_edges(cells: np.ndarray) -> np.ndarray:
    edges = np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def draw_particles(ax, fill: str = "white", edge: str = "#66686d") -> None:
    for x, y, radius in PARTICLES:
        ax.add_patch(plt.Circle((x, y), radius, facecolor=fill, edgecolor=edge, lw=0.8, zorder=6))


def style_domain_axes(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_discrete_mesh(partition: PartitionData, out_dir: Path) -> None:
    points, cells = topology_arrays(partition.mesh)
    edges = unique_mesh_edges(cells)
    mesh_segments = points[edges]
    interfaces = interface_segments(partition)
    narrowest = int(np.argmin(partition.interface_half_lengths))
    center = partition.interface_centers[narrowest]
    radius = max(0.105, 3.8 * partition.interface_half_lengths[narrowest])
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.1), constrained_layout=True)
    for ax in axes:
        ax.add_collection(
            LineCollection(mesh_segments, colors="#6f747a", linewidths=0.28, zorder=1)
        )
        if interfaces:
            ax.add_collection(
                LineCollection(interfaces, colors="#085cc7", linewidths=1.45, zorder=4)
            )
        draw_particles(ax)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Locally refined conforming mesh")
    axes[1].set_xlim(max(0, center[0]-radius), min(1, center[0]+radius))
    axes[1].set_ylim(max(0, center[1]-radius), min(1, center[1]+radius))
    axes[1].scatter(points[:, 0], points[:, 1], s=5, c="#202124", zorder=5, label="P1 geometry / pressure nodes")
    midpoints = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    axes[1].scatter(midpoints[:, 0], midpoints[:, 1], s=4, c="#e97817", zorder=3, label="P2 velocity edge nodes")
    axes[1].set_title(f"Narrow-throat zoom (interface {narrowest})")
    axes[1].legend(loc="upper right", fontsize=7.5, framealpha=0.92)
    fig.savefig(out_dir / "00_analytic_refined_discrete_mesh.png", dpi=260)
    plt.close(fig)


def plot_partition(partition: PartitionData, out_dir: Path) -> None:
    points, cells = topology_arrays(partition.mesh)
    triang = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    ax.tripcolor(
        triang,
        facecolors=partition.cell_labels,
        cmap="tab20",
        edgecolors="none",
        linewidth=0.18,
        shading="flat",
    )
    segments = interface_segments(partition)
    if segments:
        ax.add_collection(LineCollection(segments, colors="#0c58c7", linewidths=1.5, zorder=5))
    draw_particles(ax)
    ax.scatter(
        partition.pore_seeds[:, 0], partition.pore_seeds[:, 1],
        s=22, c="#d62728", edgecolors="white", linewidths=0.5, zorder=7,
        label="distance-map maxima",
    )
    style_domain_axes(ax)
    ax.set_title("Analytic saddle-cut pore subdomains and internal interfaces")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(out_dir / "01_geometry_and_subdomains.png", dpi=240)
    plt.close(fig)


def plot_fields(
    partition: PartitionData,
    u_dd: np.ndarray,
    p_dd: np.ndarray,
    out_dir: Path,
) -> None:
    points, cells = topology_arrays(partition.mesh)
    triang = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    fields = [(np.linalg.norm(u_dd, axis=1), "Velocity magnitude |u|", "viridis"),
              (p_dd, "Local pressure π (vertex-averaged at interfaces)", "coolwarm")]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7), constrained_layout=True)
    for ax, (values, title, cmap) in zip(axes, fields, strict=True):
        artist = ax.tripcolor(triang, values, shading="gouraud", cmap=cmap)
        draw_particles(ax)
        style_domain_axes(ax)
        ax.set_title(title)
        fig.colorbar(artist, ax=ax, shrink=0.84)
    fig.suptitle("2D DD-PNM reconstructed fields")
    fig.savefig(out_dir / "02_ddpnm_reconstructed_fields.png", dpi=240)
    plt.close(fig)


def plot_validation(
    partition: PartitionData,
    u_dd: np.ndarray,
    p_dd: np.ndarray,
    u_ref: np.ndarray,
    p_ref: np.ndarray,
    out_dir: Path,
) -> tuple[dict[str, float], np.ndarray]:
    points, cells = topology_arrays(partition.mesh)
    triang = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    shift = float(np.mean(p_ref - p_dd))
    p_aligned = p_dd + shift
    u_error = np.linalg.norm(u_dd - u_ref, axis=1)
    p_error = np.abs(p_aligned - p_ref)
    velocity_relative = float(np.linalg.norm(u_dd-u_ref) / max(np.linalg.norm(u_ref), 1e-30))
    pressure_relative = float(np.linalg.norm(p_aligned-p_ref) / max(np.linalg.norm(p_ref-np.mean(p_ref)), 1e-30))
    metrics = {
        "vertex_velocity_relative_l2": velocity_relative,
        "vertex_pressure_mean_aligned_relative_l2": pressure_relative,
        "pressure_alignment_shift": shift,
        "vertex_velocity_max_abs": float(np.max(u_error)),
        "vertex_pressure_max_abs": float(np.max(p_error)),
    }
    items = [
        (np.linalg.norm(u_ref,axis=1), "Reference |u|", "viridis"),
        (np.linalg.norm(u_dd,axis=1), "DD-PNM |u|", "viridis"),
        (u_error, "|u_DD - u_ref|", "magma"),
        (p_ref, "Reference p", "coolwarm"),
        (p_aligned, "DD-PNM π (mean-aligned)", "coolwarm"),
        (p_error, "|π_DD - p_ref|", "magma"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.1), constrained_layout=True)
    for ax, (values, title, cmap) in zip(axes.ravel(), items, strict=True):
        artist = ax.tripcolor(triang, values, shading="gouraud", cmap=cmap)
        draw_particles(ax)
        style_domain_axes(ax)
        ax.set_title(title, fontsize=10)
        fig.colorbar(artist, ax=ax, shrink=0.77)
    fig.suptitle(
        f"Full Taylor-Hood reference vs DD-PNM: velocity rel. {velocity_relative:.3e}, "
        f"pressure rel. {pressure_relative:.3e}"
    )
    fig.savefig(out_dir / "03_reference_comparison.png", dpi=240)
    plt.close(fig)
    return metrics, p_aligned


def plot_interface_diagnostics(solution: DdpnmSolution, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    axes[0].spy(np.abs(solution.schur_matrix) > 1.0e-14, markersize=4, color="#245da8")
    axes[0].set_title("Global interface Schur sparsity")
    axes[0].set_xlabel("interface index")
    axes[0].set_ylabel("interface index")
    axes[1].bar(
        np.arange(len(solution.interface_flux_sums)),
        np.abs(solution.interface_flux_sums),
        color="#3d77b5",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Absolute flux-balance residual")
    axes[1].set_xlabel("interface index")
    axes[1].set_ylabel("|flux-moment residual|")
    fig.savefig(out_dir / "04_interface_system_diagnostics.png", dpi=220)
    plt.close(fig)


def write_xdmf_fields(
    partition: PartitionData,
    u_dd: np.ndarray,
    p_dd: np.ndarray,
    u_ref: np.ndarray,
    p_ref: np.ndarray,
    out_dir: Path,
) -> None:
    msh = partition.mesh
    cell = msh.basix_cell()
    V = fem.functionspace(msh, element("Lagrange", cell, 1, shape=(2,)))
    Q = fem.functionspace(msh, element("Lagrange", cell, 1))
    functions = []
    for name, space, values in [
        ("u_ddpnm", V, u_dd), ("p_ddpnm", Q, p_dd),
        ("u_reference", V, u_ref), ("p_reference", Q, p_ref),
    ]:
        f = fem.Function(space)
        f.name = name
        assign_p1_function(f, values)
        functions.append(f)
    with io.XDMFFile(MPI.COMM_SELF, str(out_dir / "ddpnm_2d_fields.xdmf"), "w") as xdmf:
        xdmf.write_mesh(msh)
        for f in functions:
            xdmf.write_function(f)

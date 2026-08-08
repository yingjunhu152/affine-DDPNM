from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import matplotlib.tri as mtri
import numpy as np


COLORS = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#EE8866",
    "#44AA99",
    "#999933",
    "#882255",
    "#117733",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the 3D original DDPNM result.")
    parser.add_argument(
        "--input", type=Path, default=Path("outputs/default/ddpnm_3d_results.npz")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/default"))
    return parser.parse_args()


def set_equal_cube(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-55)


def cube_edges(lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    vertices = np.asarray(
        list(np.ndindex(2, 2, 2)), dtype=float
    )
    vertices = lower + (upper - lower) * vertices
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(vertices[i] != vertices[j]) == 1:
                edges.append([vertices[i], vertices[j]])
    return np.asarray(edges)


def draw_cube(ax, color="#30343b", linewidth=0.8, alpha=0.9) -> None:
    ax.add_collection3d(
        Line3DCollection(cube_edges(), colors=color, linewidths=linewidth, alpha=alpha)
    )


def set_local_box(ax, lower: float, upper: float) -> None:
    margin = 0.025 * (upper - lower)
    ax.set_xlim(lower - margin, upper + margin)
    ax.set_ylim(lower - margin, upper + margin)
    ax.set_zlim(lower - margin, upper + margin)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-55)


def sphere_surface(center: np.ndarray, radius: float, nu=18, nv=10):
    u = np.linspace(0.0, 2.0 * np.pi, nu)
    v = np.linspace(0.0, np.pi, nv)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def draw_spheres(
    ax,
    centers: np.ndarray,
    radius: float | np.ndarray,
    color="#7f8288",
    alpha=0.55,
    linewidth=0.15,
) -> None:
    radii = np.full(len(centers), float(radius)) if np.ndim(radius) == 0 else np.asarray(radius)
    for center, value in zip(centers, radii, strict=True):
        x, y, z = sphere_surface(center, float(value))
        ax.plot_surface(
            x,
            y,
            z,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            edgecolor="#60636a" if linewidth else "none",
            shade=True,
        )


def unique_tetra_edges(tetrahedra: np.ndarray) -> np.ndarray:
    pairs = np.asarray([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    edges = tetrahedra[:, pairs].reshape(-1, 2)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def triangle_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def outer_triangle_mask(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    coords = points[triangles]
    mask = np.zeros(len(triangles), dtype=bool)
    for axis in range(3):
        mask |= np.all(np.abs(coords[:, :, axis]) <= 2.0e-8, axis=1)
        mask |= np.all(np.abs(coords[:, :, axis] - 1.0) <= 2.0e-8, axis=1)
    return mask


def plot_four_stage(data, out_dir: Path) -> None:
    points = data["points"]
    tetrahedra = data["tetrahedra"]
    labels = data["cell_labels"]
    spheres = data["sphere_centers"]
    sphere_radius = float(data["sphere_radius"][0])
    balls = data["maximal_ball_centers"]
    ball_radii = data["maximal_ball_radii"]
    pairs = data["throat_pairs"]
    interface_triangles = data["interface_triangles"]
    interface_ids = data["interface_triangle_ids"]
    boundary_triangles = data["boundary_triangles"]

    fig = plt.figure(figsize=(18.0, 5.1), constrained_layout=True)
    axes = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]

    draw_spheres(axes[0], spheres, sphere_radius, alpha=0.78, linewidth=0.12)
    draw_cube(axes[0])
    axes[0].set_title("Computational geometry\n27 uniform solid spheres", fontsize=10.5)

    draw_spheres(axes[1], spheres, sphere_radius, color="#a5a7ab", alpha=0.16, linewidth=0.0)
    draw_spheres(
        axes[1], balls, ball_radii, color="#ef8a62", alpha=0.22, linewidth=0.20
    )
    for i, j in pairs:
        axes[1].plot(
            balls[[i, j], 0],
            balls[[i, j], 1],
            balls[[i, j], 2],
            color="#1464b4",
            linewidth=1.7,
        )
    axes[1].scatter(
        balls[:, 0], balls[:, 1], balls[:, 2], s=23, color="#b2182b", depthshade=False
    )
    draw_cube(axes[1], alpha=0.55)
    axes[1].set_title("Maximal-ball graph\n64 pore bodies and 144 throats", fontsize=10.5)

    draw_spheres(axes[2], spheres, sphere_radius, color="#8c8f94", alpha=0.28, linewidth=0.0)
    for interface_id in range(len(pairs)):
        triangles = interface_triangles[interface_ids == interface_id]
        axes[2].add_collection3d(
            Poly3DCollection(
                points[triangles],
                facecolors=COLORS[interface_id % len(COLORS)],
                edgecolors="#1455a0",
                linewidths=0.17,
                alpha=0.35,
            )
        )
    axes[2].scatter(
        balls[:, 0], balls[:, 1], balls[:, 2], s=18, color="#b2182b", depthshade=False
    )
    draw_cube(axes[2])
    axes[2].set_title("Maximal-ball subdomains\nconforming saddle interfaces", fontsize=10.5)

    selected_tetrahedra = tetrahedra[np.all(data["cell_centers"] <= 0.5, axis=1)]
    edges = unique_tetra_edges(selected_tetrahedra)
    axes[3].add_collection3d(
        Line3DCollection(
            points[edges], colors="#74777d", linewidths=0.24, alpha=0.48
        )
    )
    outer_mask = outer_triangle_mask(points, boundary_triangles)
    sphere_triangles = boundary_triangles[~outer_mask]
    sphere_edges = triangle_edges(sphere_triangles)
    axes[3].add_collection3d(
        Line3DCollection(
            points[sphere_edges], colors="#3f4248", linewidths=0.25, alpha=0.44
        )
    )
    interface_centroids = points[interface_triangles].mean(axis=1)
    local_interfaces = interface_triangles[np.all(interface_centroids <= 0.5 + 1.0e-10, axis=1)]
    axes[3].add_collection3d(
        Poly3DCollection(
            points[local_interfaces],
            facecolors="#4e8bd1",
            edgecolors="#0756a5",
            linewidths=0.24,
            alpha=0.22,
        )
    )
    draw_cube(axes[3], alpha=0.65)
    axes[3].set_title("Body-fitted tetrahedral mesh\ncut-away view of one cube octant", fontsize=10.5)

    for index, ax in enumerate(axes):
        set_equal_cube(ax)
        ax.text2D(0.50, -0.035, f"({chr(97 + index)})", transform=ax.transAxes, ha="center", fontsize=12)
    fig.savefig(out_dir / "01_ddpnm3d_four_stage_construction.png", dpi=260)
    plt.close(fig)


def plot_mesh_detail(data, out_dir: Path) -> None:
    points = data["points"]
    tetrahedra = data["tetrahedra"]
    labels = data["cell_labels"]
    centers = data["cell_centers"]
    diameters = data["cell_diameters"]
    spheres = data["sphere_centers"]
    sphere_radius = float(data["sphere_radius"][0])
    boundary_triangles = data["boundary_triangles"]
    interface_triangles = data["interface_triangles"]
    interface_ids = data["interface_triangle_ids"]

    fig = plt.figure(figsize=(17.4, 5.8), constrained_layout=True)
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    selected_label = 21
    selected = tetrahedra[labels == selected_label]
    edges = unique_tetra_edges(selected)
    ax0.add_collection3d(
        Line3DCollection(points[edges], colors="#565a60", linewidths=0.27, alpha=0.58)
    )
    local_interface_ids = np.flatnonzero(
        np.any(data["throat_pairs"] == selected_label, axis=1)
    )
    triangles = interface_triangles[np.isin(interface_ids, local_interface_ids)]
    ax0.add_collection3d(
        Poly3DCollection(
            points[triangles],
            facecolors="#80b1d3",
            edgecolors="#0756a5",
            linewidths=0.25,
            alpha=0.24,
        )
    )
    ax0.add_collection3d(
        Line3DCollection(
            cube_edges(0.2, 0.5), colors="#30343b", linewidths=0.8, alpha=0.9
        )
    )
    set_local_box(ax0, 0.2, 0.5)
    ax0.set_title(f"Volumetric tetrahedra in local region $\\Omega_{{{selected_label}}}$")

    z_mask = np.all(np.abs(points[interface_triangles, 2] - 0.5) <= 2.0e-8, axis=1)
    z_triangles = interface_triangles[z_mask]
    edges2d = triangle_edges(z_triangles)
    ax1.add_collection(
        LineCollection(points[edges2d, :2], colors="#51555c", linewidths=0.35)
    )
    for center in spheres[np.abs(spheres[:, 2] - 0.5) < 1.0e-12]:
        ax1.add_patch(
            plt.Circle(
                center[:2], sphere_radius, facecolor="white", edgecolor="#33363b", lw=0.65
            )
        )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Conforming mesh on the saddle plane $z=0.5$")

    sphere_distance = np.min(
        np.linalg.norm(centers[:, None, :] - spheres[None, :, :], axis=2) - sphere_radius,
        axis=1,
    )
    outer_distance = np.minimum(np.min(centers, axis=1), np.min(1.0 - centers, axis=1))
    interface_distance = np.min(
        np.abs(centers[:, :, None] - np.asarray([0.2, 0.5, 0.8])[None, None, :]),
        axis=(1, 2),
    )
    sphere_half_band = 0.5 * float(data["mesh_sphere_band"][0])
    boundary_half_band = 0.5 * float(data["mesh_boundary_band"][0])
    interface_half_band = 0.5 * float(data["mesh_interface_band"][0])
    zones = [
        diameters[sphere_distance <= sphere_half_band],
        diameters[outer_distance <= boundary_half_band],
        diameters[interface_distance <= interface_half_band],
        diameters[
            (sphere_distance >= 0.10)
            & (outer_distance >= 0.08)
            & (interface_distance >= 0.08)
        ],
    ]
    ax2.boxplot(
        zones,
        tick_labels=["sphere", "outer wall", "interface", "bulk"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#9ecae1", "edgecolor": "#355f8a"},
        medianprops={"color": "#b2182b", "linewidth": 1.5},
    )
    ax2.set_ylabel("tetrahedron diameter $h_K$")
    ax2.set_title("Geometric local refinement audit")
    ax2.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "02_tetrahedral_mesh_and_refinement.png", dpi=260)
    plt.close(fig)


def plot_fields(data, out_dir: Path) -> None:
    points = data["points"]
    triangles = data["interface_triangles"]
    mask = np.all(np.abs(points[triangles, 2] - 0.5) <= 2.0e-8, axis=1)
    triangles = triangles[mask]
    # The DDPNM solution is stored piecewise by subdomain.  A shared vertex can
    # therefore carry two independent traces.  This arithmetic trace average is
    # deliberately used only to make a readable continuous slice plot.
    velocity = data["u_ddpnm_trace_average_visualization"]
    pressure = data["p_ddpnm_trace_average_visualization"]
    values = [
        (np.linalg.norm(velocity, axis=1), "Velocity magnitude $|u|$", "viridis"),
        (velocity[:, 0], "Streamwise velocity $u_x$", "coolwarm"),
        (pressure, "Trace-averaged pressure $p$", "coolwarm"),
    ]
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), constrained_layout=True)
    sphere_centers = data["sphere_centers"]
    radius = float(data["sphere_radius"][0])
    for ax, (field, title, cmap) in zip(axes, values, strict=True):
        artist = ax.tripcolor(triangulation, field, shading="gouraud", cmap=cmap)
        for center in sphere_centers[np.abs(sphere_centers[:, 2] - 0.5) < 1.0e-12]:
            ax.add_patch(
                plt.Circle(center[:2], radius, facecolor="white", edgecolor="none", zorder=5)
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        fig.colorbar(artist, ax=ax, shrink=0.82)
    fig.suptitle(
        "Original 3D DDPNM on $z=0.5$ "
        "(two-sided trace average for visualization only)"
    )
    fig.savefig(out_dir / "03_ddpnm_fields_central_plane.png", dpi=250)
    plt.close(fig)


def plot_diagnostics(data, out_dir: Path) -> None:
    S = data["schur_matrix"]
    rhs = data["schur_rhs"]
    pressures = data["interface_pressures"]
    residuals = data["interface_flux_sums"]
    balls = data["maximal_ball_centers"]
    pairs = data["throat_pairs"]
    fig = plt.figure(figsize=(15.3, 4.7), constrained_layout=True)
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    image = ax0.imshow(S, cmap="RdBu_r", aspect="equal")
    ax0.set_title(f"{len(pressures)} x {len(pressures)} interface Schur matrix")
    ax0.set_xlabel("interface unknown")
    ax0.set_ylabel("interface equation")
    fig.colorbar(image, ax=ax0, shrink=0.78)
    ax1.bar(np.arange(len(residuals)), np.abs(residuals), color="#4477aa")
    ax1.set_yscale("log")
    ax1.set_xlabel("interface id")
    ax1.set_ylabel("absolute flux residual")
    ax1.set_title(
        f"Conservation: max = {np.max(np.abs(residuals)):.2e}\n"
        f"linear residual = {np.linalg.norm(S @ pressures-rhs)/max(np.linalg.norm(rhs),1e-30):.2e}"
    )
    norm = plt.Normalize(float(np.min(pressures)), float(np.max(pressures)))
    cmap = plt.get_cmap("coolwarm")
    for edge_id, (i, j) in enumerate(pairs):
        ax2.plot(
            balls[[i, j], 0],
            balls[[i, j], 1],
            balls[[i, j], 2],
            color=cmap(norm(pressures[edge_id])),
            linewidth=4.0,
        )
    ax2.scatter(balls[:, 0], balls[:, 1], balls[:, 2], s=28, color="#20242a")
    draw_cube(ax2, alpha=0.45)
    set_equal_cube(ax2)
    ax2.set_title("Interface traction on the pore graph")
    fig.savefig(out_dir / "04_interface_system_diagnostics.png", dpi=240)
    plt.close(fig)


def plot_fem_ddpnm_errors(data, out_dir: Path) -> None:
    if "error_slice_points" not in data or not len(data["error_slice_points"]):
        return
    points = data["error_slice_points"]
    triangles = data["error_slice_triangles"]
    u_fem = data["error_slice_u_fem"]
    p_fem = data["error_slice_p_fem"]
    u_ddpnm = data["error_slice_u_ddpnm"]
    p_ddpnm = data["error_slice_p_ddpnm"]
    z_value = float(data["error_slice_z"][0])
    velocity_error = np.linalg.norm(u_ddpnm - u_fem, axis=1)
    pressure_error = p_ddpnm - p_fem
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    maximum_pressure_error = max(float(np.max(np.abs(pressure_error))), 1.0e-14)
    fields = [
        (np.linalg.norm(u_fem, axis=1), r"FEM velocity magnitude $|u_h^{\rm FE}|$", "turbo", None),
        (p_fem, r"FEM pressure $p_h^{\rm FE}$", "turbo", None),
        (velocity_error, r"Velocity error $|u_h^{\rm DD}-u_h^{\rm FE}|$", "turbo", None),
        (
            pressure_error,
            r"Pressure error $p_h^{\rm DD}-p_h^{\rm FE}$",
            "RdBu_r",
            TwoSlopeNorm(
                vmin=-maximum_pressure_error,
                vcenter=0.0,
                vmax=maximum_pressure_error,
            ),
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.1), constrained_layout=True)
    sphere_centers = data["sphere_centers"]
    sphere_radius = float(data["sphere_radius"][0])
    for panel, (ax, (field, title, cmap, norm)) in enumerate(
        zip(axes.flat, fields, strict=True)
    ):
        artist = ax.tripcolor(
            triangulation,
            field,
            shading="gouraud",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        for center in sphere_centers:
            distance = abs(float(center[2] - z_value))
            if distance >= sphere_radius:
                continue
            section_radius = np.sqrt(sphere_radius**2 - distance**2)
            ax.add_patch(
                plt.Circle(
                    center[:2],
                    section_radius,
                    facecolor="white",
                    edgecolor="none",
                    zorder=5,
                )
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12.2)
        ax.text(
            0.5,
            -0.055,
            f"({chr(97 + panel)})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12.5,
        )
        colorbar = fig.colorbar(artist, ax=ax, shrink=0.88, pad=0.025)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        colorbar.formatter = formatter
        colorbar.update_ticks()
    fig.suptitle(
        "Traditional FEM reference and original DDPNM error on the identical mesh\n"
        f"cell-sided slice $z={z_value:.2f}$; no averaging across DDPNM interfaces",
        fontsize=13.0,
    )
    fig.savefig(out_dir / "05_fem_ddpnm_error_fields.png", dpi=260)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input) as data:
        plot_four_stage(data, args.out_dir)
        plot_mesh_detail(data, args.out_dir)
        plot_fields(data, args.out_dir)
        plot_diagnostics(data, args.out_dir)
        plot_fem_ddpnm_errors(data, args.out_dir)
    print(f"Plots written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

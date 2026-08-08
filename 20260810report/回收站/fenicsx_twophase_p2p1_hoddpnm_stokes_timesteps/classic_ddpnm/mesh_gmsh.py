from __future__ import annotations

import numpy as np
import ufl
from basix.ufl import element
from dolfinx import mesh as dmesh
from mpi4py import MPI

from .geometry import PORT_TAG_BASE, WALL_TAG, Pore, normalize, spherical_distance


def build_pore_mesh(
    pore: Pore,
    h: float = 0.22,
    port_half_width: float = 0.42,
    throat_radius: float = 0.13,
    stub_length: float = 0.46,
):
    """Create a tagged local pore-throat mesh.

    The local DDPNM response should see an open cross-section for every port.
    A closed ball with spherical-cap Neumann patches gives very noisy and
    non-physical flux matrices in 3D. This mesh therefore uses the union of a
    spherical pore body and short cylindrical stubs in every port direction.
    The end disk of each stub is tagged as the pressure interface; all other
    exterior facets are no-slip wall.
    """
    center = np.asarray(pore.center, dtype=float)
    radius = pore.radius
    directions = [np.asarray(port.normal, dtype=float) for port in pore.ports]
    h_eff = min(h, throat_radius * 0.9, radius * 0.65)
    bbox_radius = radius + stub_length + h_eff
    axes = [
        np.arange(center[axis] - bbox_radius, center[axis] + bbox_radius + 0.5 * h_eff, h_eff)
        for axis in range(3)
    ]

    all_coords = np.asarray(
        [(x, y, z) for z in axes[2] for y in axes[1] for x in axes[0]],
        dtype=np.float64,
    )
    nx, ny, nz = len(axes[0]), len(axes[1]), len(axes[2])

    raw_cells: list[list[int]] = []
    for iz in range(nz - 1):
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                centroid = np.array(
                    [
                        0.5 * (axes[0][ix] + axes[0][ix + 1]),
                        0.5 * (axes[1][iy] + axes[1][iy + 1]),
                        0.5 * (axes[2][iz] + axes[2][iz + 1]),
                    ],
                    dtype=float,
                )
                if not point_in_local_domain(
                    centroid,
                    center,
                    radius,
                    directions,
                    throat_radius,
                    stub_length,
                ):
                    continue
                v000 = grid_id(ix, iy, iz, nx, ny)
                v001 = grid_id(ix + 1, iy, iz, nx, ny)
                v010 = grid_id(ix, iy + 1, iz, nx, ny)
                v011 = grid_id(ix + 1, iy + 1, iz, nx, ny)
                v100 = grid_id(ix, iy, iz + 1, nx, ny)
                v101 = grid_id(ix + 1, iy, iz + 1, nx, ny)
                v110 = grid_id(ix, iy + 1, iz + 1, nx, ny)
                v111 = grid_id(ix + 1, iy + 1, iz + 1, nx, ny)
                raw_cells.extend(
                    [
                        [v000, v001, v011, v111],
                        [v000, v011, v010, v111],
                        [v000, v010, v110, v111],
                        [v000, v110, v100, v111],
                        [v000, v100, v101, v111],
                        [v000, v101, v001, v111],
                    ]
                )

    if not raw_cells:
        raise RuntimeError(f"Local mesh for pore {pore.id} is empty.")

    raw_cells_array = np.asarray(raw_cells, dtype=np.int64)
    used = np.unique(raw_cells_array.ravel())
    remap = -np.ones(len(all_coords), dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    coords = all_coords[used]
    cell_array = orient_tets(coords, remap[raw_cells_array])

    domain = ufl.Mesh(element("Lagrange", "tetrahedron", 1, shape=(3,)))
    msh = dmesh.create_mesh(MPI.COMM_SELF, cell_array, domain, coords)

    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    exterior = dmesh.exterior_facet_indices(msh.topology)
    facet_to_vertices = msh.topology.connectivity(fdim, 0)

    facet_info: list[tuple[int, np.ndarray, list[tuple[float, float]]]] = []
    max_axial = [-np.inf for _ in pore.ports]
    for facet in exterior:
        vertices = facet_to_vertices.links(facet)
        midpoint = msh.geometry.x[vertices].mean(axis=0)
        port_measures: list[tuple[float, float]] = []
        for local_index, _ in enumerate(pore.ports):
            direction = directions[local_index]
            axial = float(np.dot(midpoint - center, direction))
            radial = np.linalg.norm((midpoint - center) - axial * direction)
            port_measures.append((axial, radial))
            if radial <= throat_radius * 1.25:
                max_axial[local_index] = max(max_axial[local_index], axial)
        facet_info.append((int(facet), midpoint, port_measures))

    tag_facets: list[int] = []
    tag_values: list[int] = []
    tagged_counts = [0 for _ in pore.ports]
    for facet, _, port_measures in facet_info:
        tag = WALL_TAG
        for local_index, port in enumerate(pore.ports):
            axial, radial = port_measures[local_index]
            if (
                axial >= max_axial[local_index] - 0.25 * h_eff
                and radial <= throat_radius * 1.25
            ):
                tag = PORT_TAG_BASE + local_index
                tagged_counts[local_index] += 1
                break
        tag_facets.append(int(facet))
        tag_values.append(tag)

    # Oblique cylindrical stubs cut through the Cartesian background grid as a
    # stair-step surface. Rarely, no exterior facet midpoint lands close enough
    # to the ideal flat end disk. In that case, tag the best available exterior
    # facets near the furthest axial section so every PNM port has DOFs.
    tag_by_facet = {facet: value for facet, value in zip(tag_facets, tag_values)}
    for local_index, count in enumerate(tagged_counts):
        if count > 0:
            continue
        candidates: list[tuple[float, int]] = []
        for facet, _, port_measures in facet_info:
            axial, radial = port_measures[local_index]
            if axial <= 0.0:
                continue
            score = axial - 3.0 * max(0.0, radial - throat_radius)
            candidates.append((score, facet))
        for _, facet in sorted(candidates, reverse=True)[:8]:
            tag_by_facet[facet] = PORT_TAG_BASE + local_index

    tag_facets = sorted(tag_by_facet)
    tag_values = [tag_by_facet[facet] for facet in tag_facets]

    order = np.argsort(tag_facets)
    facet_tags = dmesh.meshtags(
        msh,
        fdim,
        np.asarray(tag_facets, dtype=np.int32)[order],
        np.asarray(tag_values, dtype=np.int32)[order],
    )
    return msh, facet_tags


def point_in_local_domain(
    point: np.ndarray,
    center: np.ndarray,
    radius: float,
    directions: list[np.ndarray],
    throat_radius: float,
    stub_length: float,
) -> bool:
    relative = point - center
    if np.linalg.norm(relative) <= radius:
        return True
    for direction in directions:
        axial = float(np.dot(relative, direction))
        if axial < 0.0 or axial > stub_length:
            continue
        radial = np.linalg.norm(relative - axial * direction)
        if radial <= throat_radius:
            return True
    return False


def grid_id(ix: int, iy: int, iz: int, nx: int, ny: int) -> int:
    return (iz * ny + iy) * nx + ix


def fibonacci_sphere(samples: int) -> np.ndarray:
    i = np.arange(samples, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / samples
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = phi * i
    return np.column_stack((np.cos(theta) * r, np.sin(theta) * r, z))


def unique_rows(points: np.ndarray) -> np.ndarray:
    rounded = np.round(points, decimals=12)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def tetra_volume(coords: np.ndarray) -> float:
    return abs(np.linalg.det(np.column_stack((coords[1] - coords[0], coords[2] - coords[0], coords[3] - coords[0])))) / 6.0


def orient_tets(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    oriented = cells.copy()
    for i, tet in enumerate(oriented):
        mat = np.column_stack((coords[tet[1]] - coords[tet[0]], coords[tet[2]] - coords[tet[0]], coords[tet[3]] - coords[tet[0]]))
        if np.linalg.det(mat) < 0:
            oriented[i, [2, 3]] = oriented[i, [3, 2]]
    return oriented

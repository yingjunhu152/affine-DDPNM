from __future__ import annotations

import numpy as np

from ddpnm_core.reconstruction import mixed_solution_functions
from ddpnm_core.solver_types import ReferenceSolution

from .geometry import PartitionData
from .solver import DdpnmSolution


def build_cellwise_slice(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    z_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect each tetrahedron independently, retaining duplicate interface traces."""
    edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    slice_points: list[np.ndarray] = []
    slice_cells: list[int] = []
    slice_triangles: list[list[int]] = []
    tolerance = 1.0e-12
    for cell, vertices in enumerate(tetrahedra):
        coordinates = points[vertices]
        signed = coordinates[:, 2] - z_value
        if np.min(signed) > tolerance or np.max(signed) < -tolerance:
            continue
        polygon: list[np.ndarray] = []

        def append_unique(point: np.ndarray) -> None:
            if not any(np.linalg.norm(point - existing) <= 1.0e-11 for existing in polygon):
                polygon.append(point)

        for local_vertex in range(4):
            if abs(signed[local_vertex]) <= tolerance:
                append_unique(coordinates[local_vertex].copy())
        for first, second in edges:
            if signed[first] * signed[second] < -(tolerance**2):
                fraction = -signed[first] / (signed[second] - signed[first])
                append_unique(
                    coordinates[first]
                    + fraction * (coordinates[second] - coordinates[first])
                )
        if len(polygon) < 3:
            continue
        if len(polygon) > 4:
            raise RuntimeError("A plane cut produced more than four tetrahedron vertices.")
        polygon_array = np.asarray(polygon, dtype=float)
        center = np.mean(polygon_array, axis=0)
        angles = np.arctan2(
            polygon_array[:, 1] - center[1], polygon_array[:, 0] - center[0]
        )
        polygon_array = polygon_array[np.argsort(angles)]
        first_index = len(slice_points)
        for point in polygon_array:
            slice_points.append(point)
            slice_cells.append(cell)
        center_index = len(slice_points)
        slice_points.append(center)
        slice_cells.append(cell)
        for index in range(len(polygon_array)):
            slice_triangles.append(
                [
                    center_index,
                    first_index + index,
                    first_index + (index + 1) % len(polygon_array),
                ]
            )
    return (
        np.asarray(slice_points, dtype=float),
        np.asarray(slice_triangles, dtype=np.int32),
        np.asarray(slice_cells, dtype=np.int32),
    )


def evaluate_fem_ddpnm_slice(
    partition: PartitionData,
    solution: DdpnmSolution,
    reference: ReferenceSolution,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    z_value: float = 0.48,
) -> dict[str, np.ndarray]:
    slice_points, slice_triangles, parent_cells = build_cellwise_slice(
        points, tetrahedra, z_value
    )
    u_fem_function, p_fem_function = mixed_solution_functions(
        reference.W, reference.solution
    )
    u_fem = np.asarray(
        u_fem_function.eval(slice_points, parent_cells), dtype=float
    ).reshape(-1, 3)
    p_fem = np.asarray(
        p_fem_function.eval(slice_points, parent_cells), dtype=float
    ).reshape(-1)

    n_cells = len(tetrahedra)
    response_by_parent = np.full(n_cells, -1, dtype=np.int32)
    local_cell_by_parent = np.full(n_cells, -1, dtype=np.int32)
    for response_index, response in enumerate(solution.local_responses):
        parent = np.asarray(response.parent_cell_map, dtype=np.int32)
        response_by_parent[parent] = response_index
        local_cell_by_parent[parent] = np.arange(len(parent), dtype=np.int32)
    if np.any(response_by_parent < 0):
        raise RuntimeError("Slice evaluation has unmapped parent tetrahedra.")

    u_ddpnm = np.empty_like(u_fem)
    p_ddpnm = np.empty_like(p_fem)
    point_responses = response_by_parent[parent_cells]
    for response_index, (response, local_solution) in enumerate(
        zip(solution.local_responses, solution.local_solutions, strict=True)
    ):
        selected = np.flatnonzero(point_responses == response_index)
        if not len(selected):
            continue
        u_local, p_local = mixed_solution_functions(response.W, local_solution)
        local_cells = local_cell_by_parent[parent_cells[selected]]
        u_ddpnm[selected] = np.asarray(
            u_local.eval(slice_points[selected], local_cells), dtype=float
        ).reshape(-1, 3)
        p_ddpnm[selected] = np.asarray(
            p_local.eval(slice_points[selected], local_cells), dtype=float
        ).reshape(-1)

    return {
        "error_slice_z": np.asarray([z_value], dtype=float),
        "error_slice_points": slice_points,
        "error_slice_triangles": slice_triangles,
        "error_slice_parent_cells": parent_cells,
        "error_slice_u_fem": u_fem,
        "error_slice_p_fem": p_fem,
        "error_slice_u_ddpnm": u_ddpnm,
        "error_slice_p_ddpnm": p_ddpnm,
    }


def _interface_facets(
    partition: PartitionData, interface_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (facet_ids, one parent cell per facet) of a partition interface.

    The parent cell is the lower-indexed of the two cells sharing the facet,
    which fixes a deterministic evaluation side for the interface traces.
    """
    mesh = partition.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, 0)
    mesh.topology.create_connectivity(fdim, tdim)
    f2v = mesh.topology.connectivity(fdim, 0)
    f2c = mesh.topology.connectivity(fdim, tdim)
    facet_ids = np.flatnonzero(partition.facet_interface_ids == interface_id)
    parent_cells = np.asarray(
        [min(int(c) for c in f2c.links(int(fid))) for fid in facet_ids],
        dtype=np.int32,
    )
    return facet_ids, parent_cells


def fem_interface_fluxes(partition: PartitionData, reference: ReferenceSolution) -> np.ndarray:
    """Signed FEM normal flux  sum_facets (u.n) area  per partition interface."""
    mesh = partition.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, 0)
    mesh.topology.create_connectivity(fdim, tdim)
    f2v = mesh.topology.connectivity(fdim, 0)
    f2c = mesh.topology.connectivity(fdim, tdim)
    u_fem_function, _p_fem = mixed_solution_functions(reference.W, reference.solution)
    x = mesh.geometry.x
    fluxes = np.zeros(len(partition.interface_pairs))
    for interface_id in range(len(fluxes)):
        facet_ids, parent_cells = _interface_facets(partition, interface_id)
        if not len(facet_ids):
            continue
        vertices = np.asarray(
            [list(f2v.links(int(fid))) for fid in facet_ids], dtype=np.int32
        )
        tri = x[vertices]
        u = np.asarray(
            u_fem_function.eval(tri.mean(axis=1), parent_cells), dtype=float
        ).reshape(-1, 3)
        areas = (
            np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            / 2.0
        )
        normal = partition.interface_normals[interface_id]
        fluxes[interface_id] = float(np.sum((u * normal).sum(axis=1) * areas))
    return fluxes


def evaluate_fem_ddpnm_interface(
    partition: PartitionData,
    solution: DdpnmSolution,
    reference: ReferenceSolution,
    interface_id: int,
) -> dict[str, np.ndarray]:
    """Evaluate u.n of FEM and of one DDPNM solution on the facets of one interface.

    Returns the values on deduplicated facet vertices together with the
    normalized in-plane coordinates ``(s, t)`` of the interface, scaled so
    that ``(s, t)`` lies in ``[-1, 1]^2`` (the vertex with the largest
    in-plane radius sits on the boundary).
    """
    mesh = partition.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, 0)
    mesh.topology.create_connectivity(fdim, tdim)
    f2v = mesh.topology.connectivity(fdim, 0)
    f2c = mesh.topology.connectivity(fdim, tdim)
    x = mesh.geometry.x
    facet_ids, parent_cells = _interface_facets(partition, interface_id)
    if not len(facet_ids):
        raise RuntimeError(f"Interface {interface_id} has no facets.")
    index: dict[int, int] = {}
    points: list[np.ndarray] = []
    triangles: list[list[int]] = []
    vertex_parent_cells: list[int] = []
    for fid, parent_cell in zip(facet_ids, parent_cells):
        local = []
        for vertex in f2v.links(int(fid)):
            if int(vertex) not in index:
                index[int(vertex)] = len(points)
                points.append(x[int(vertex)].copy())
                # the normal trace u.n is continuous across the interface,
                # so the parent cell of the first referencing facet fixes a
                # deterministic evaluation side for every vertex
                vertex_parent_cells.append(int(parent_cell))
            local.append(index[int(vertex)])
        triangles.append(local)
    points3d = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int32)
    parent_cells = np.asarray(vertex_parent_cells, dtype=np.int32)
    normal = partition.interface_normals[interface_id]
    axis = int(np.argmin(np.abs(normal)))
    e1 = np.cross(np.eye(3)[axis], normal)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    center = points3d.mean(axis=0)
    st = np.column_stack(((points3d - center) @ e1, (points3d - center) @ e2))
    radius = float(np.max(np.linalg.norm(st, axis=1)))
    st /= radius

    u_fem_function, _p_fem = mixed_solution_functions(reference.W, reference.solution)
    u_fem = np.asarray(
        u_fem_function.eval(points3d, parent_cells), dtype=float
    ).reshape(-1, 3)

    n_cells = mesh.topology.index_map(tdim).size_local
    response_by_parent = np.full(n_cells, -1, dtype=np.int32)
    local_cell_by_parent = np.full(n_cells, -1, dtype=np.int32)
    for response_index, response in enumerate(solution.local_responses):
        parent = np.asarray(response.parent_cell_map, dtype=np.int32)
        response_by_parent[parent] = response_index
        local_cell_by_parent[parent] = np.arange(len(parent), dtype=np.int32)
    if np.any(response_by_parent[parent_cells] < 0):
        raise RuntimeError("Interface evaluation has unmapped parent tetrahedra.")
    u_ddpnm = np.empty_like(u_fem)
    point_responses = response_by_parent[parent_cells]
    for response_index, (response, local_solution) in enumerate(
        zip(solution.local_responses, solution.local_solutions, strict=True)
    ):
        selected = np.flatnonzero(point_responses == response_index)
        if not len(selected):
            continue
        u_local, _p_local = mixed_solution_functions(response.W, local_solution)
        local_cells = local_cell_by_parent[parent_cells[selected]]
        u_ddpnm[selected] = np.asarray(
            u_local.eval(points3d[selected], local_cells), dtype=float
        ).reshape(-1, 3)

    return {
        "interface_points": points3d,
        "interface_st": st,
        "interface_triangles": triangles,
        "interface_parent_cells": parent_cells,
        "interface_normal": normal,
        "interface_center": center,
        "interface_q_fem": (u_fem * normal).sum(axis=1),
        "interface_q_ddpnm": (u_ddpnm * normal).sum(axis=1),
    }


def interface_contour_segments(
    partition: PartitionData, z_value: float
) -> np.ndarray:
    """Line segments where the partition interface surfaces cross the plane z=z_value.

    Returns an ``(n, 2, 3)`` array; the figure scripts draw these as the
    interface outline on slice plots.
    """
    mesh = partition.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    x = mesh.geometry.x
    tolerance = 1.0e-12
    segments: list[np.ndarray] = []
    for interface_id in range(len(partition.interface_pairs)):
        facet_ids = np.flatnonzero(partition.facet_interface_ids == interface_id)
        for fid in facet_ids:
            tri = x[list(f2v.links(int(fid)))]
            signed = tri[:, 2] - z_value
            if np.min(signed) > tolerance or np.max(signed) < -tolerance:
                continue
            crossings: list[np.ndarray] = []
            for first, second in ((0, 1), (1, 2), (2, 0)):
                if signed[first] * signed[second] < -(tolerance**2):
                    fraction = -signed[first] / (signed[second] - signed[first])
                    crossings.append(tri[first] + fraction * (tri[second] - tri[first]))
            if len(crossings) == 2:
                segments.append(np.stack(crossings))
    return np.asarray(segments, dtype=float)

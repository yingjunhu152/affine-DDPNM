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

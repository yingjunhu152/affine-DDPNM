"""Cellwise slice extraction for 3-D → 2-D cross sections."""

from __future__ import annotations

import numpy as np


def build_cellwise_slice(
    points: np.ndarray,
    cells: np.ndarray,
    z_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect each simplex independently, retaining duplicate interface traces.

    Works for both tetrahedra (3-D) and triangles (2-D line cuts).
    """
    n_vertices = cells.shape[1]  # 3 for tri, 4 for tet
    if n_vertices == 3:
        # 2-D: line cut through triangles at y = z_value
        # Use y coordinate as the cut axis
        edges = ((0, 1), (0, 2), (1, 2))
        cut_axis = 1  # y-axis
    elif n_vertices == 4:
        edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        cut_axis = 2  # z-axis
    else:
        raise ValueError("Cells must be triangles or tetrahedra.")

    slice_points: list[np.ndarray] = []
    slice_cells: list[int] = []
    slice_segments: list[list[int]] = []
    tolerance = 1.0e-12

    for cell, vertices in enumerate(cells):
        coordinates = points[vertices]
        signed = coordinates[:, cut_axis] - z_value
        if np.min(signed) > tolerance or np.max(signed) < -tolerance:
            continue

        # Collect intersection points
        intersections: list[np.ndarray] = []

        def append_unique(point: np.ndarray) -> None:
            if not any(np.linalg.norm(point - existing) <= 1.0e-11
                       for existing in intersections):
                intersections.append(point)

        for local_vertex in range(n_vertices):
            if abs(signed[local_vertex]) <= tolerance:
                append_unique(coordinates[local_vertex].copy())

        for first, second in edges:
            if signed[first] * signed[second] < -(tolerance**2):
                fraction = -signed[first] / (signed[second] - signed[first])
                append_unique(
                    coordinates[first]
                    + fraction * (coordinates[second] - coordinates[first])
                )

        if n_vertices == 4:
            # Tetrahedron: polygon → triangulate
            if len(intersections) < 3:
                continue
            poly = np.asarray(intersections, dtype=float)
            center = np.mean(poly, axis=0)
            angles = np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0])
            poly = poly[np.argsort(angles)]
            first_idx = len(slice_points)
            for pt in poly:
                slice_points.append(pt)
                slice_cells.append(cell)
            center_idx = len(slice_points)
            slice_points.append(center)
            slice_cells.append(cell)
            for idx in range(len(poly)):
                slice_segments.append([
                    center_idx, first_idx + idx,
                    first_idx + (idx + 1) % len(poly),
                ])
        else:
            # Triangle: line segment
            if len(intersections) != 2:
                continue
            first_idx = len(slice_points)
            for pt in intersections:
                slice_points.append(pt)
                slice_cells.append(cell)
            slice_segments.append([first_idx, first_idx + 1])

    return (
        np.asarray(slice_points, dtype=float),
        np.asarray(slice_segments, dtype=np.int32),
        np.asarray(slice_cells, dtype=np.int32),
    )


def evaluate_fem_ddpnm_slice(
    partition,
    solution,
    reference,
    points: np.ndarray,
    cells: np.ndarray,
    z_value: float = 0.48,
) -> dict[str, np.ndarray]:
    """Evaluate FEM and DDPNM fields on a cellwise slice and return error arrays."""
    from ddpnm_core.reconstruction import mixed_solution_functions as _msf

    slice_pts, slice_tris, parent_cells = build_cellwise_slice(points, cells, z_value)
    u_fem_fn, p_fem_fn = _msf(reference.W, reference.solution)
    gdim = reference.W.mesh.geometry.dim
    u_fem = np.asarray(u_fem_fn.eval(slice_pts, parent_cells), dtype=float).reshape(-1, gdim)
    p_fem = np.asarray(p_fem_fn.eval(slice_pts, parent_cells), dtype=float).reshape(-1)

    n_cells = len(cells)
    response_by_parent = np.full(n_cells, -1, dtype=np.int32)
    local_cell_by_parent = np.full(n_cells, -1, dtype=np.int32)
    for ridx, resp in enumerate(solution.local_responses):
        parent = np.asarray(resp.parent_cell_map, dtype=np.int32)
        response_by_parent[parent] = ridx
        local_cell_by_parent[parent] = np.arange(len(parent), dtype=np.int32)
    if np.any(response_by_parent < 0):
        raise RuntimeError("Slice evaluation has unmapped parent cells.")

    u_ddpnm = np.empty_like(u_fem)
    p_ddpnm = np.empty_like(p_fem)
    point_responses = response_by_parent[parent_cells]
    for ridx, (resp, local_sol) in enumerate(
        zip(solution.local_responses, solution.local_solutions, strict=True)
    ):
        selected = np.flatnonzero(point_responses == ridx)
        if not len(selected):
            continue
        u_loc, p_loc = _msf(resp.W, local_sol)
        local_cells_sel = local_cell_by_parent[parent_cells[selected]]
        u_ddpnm[selected] = np.asarray(
            u_loc.eval(slice_pts[selected], local_cells_sel), dtype=float
        ).reshape(-1, gdim)
        p_ddpnm[selected] = np.asarray(
            p_loc.eval(slice_pts[selected], local_cells_sel), dtype=float
        ).reshape(-1)

    return {
        "error_slice_z": np.asarray([z_value], dtype=float),
        "error_slice_points": slice_pts,
        "error_slice_triangles": slice_tris,
        "error_slice_parent_cells": parent_cells,
        "error_slice_u_fem": u_fem,
        "error_slice_p_fem": p_fem,
        "error_slice_u_ddpnm": u_ddpnm,
        "error_slice_p_ddpnm": p_ddpnm,
    }

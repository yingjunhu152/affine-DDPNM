"""Mesh quality auditing for 2-D and 3-D DDPNM partitions."""

from __future__ import annotations

import numpy as np


def simplex_diameters_and_quality(
    points: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (diameters, quality, volumes) for simplices (triangles or tetrahedra).

    For triangles, quality = 4√3 · area / (sum of squared edge lengths).
    For tetrahedra, quality = 12 · (3·V)^{2/3} / (sum of squared edge lengths).
    """
    n_vertices = cells.shape[1]  # 3 for tri, 4 for tet
    if n_vertices not in (3, 4):
        raise ValueError("Cells must be triangles or tetrahedra.")

    # Edge pairs
    if n_vertices == 3:
        vertex_pairs = [(0, 1), (0, 2), (1, 2)]
    else:
        vertex_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    coords = points[cells]
    edges = np.stack(
        [coords[:, j] - coords[:, i] for i, j in vertex_pairs], axis=1
    )
    lengths_sq = np.sum(edges * edges, axis=2)
    diameters = np.sqrt(np.max(lengths_sq, axis=1))

    if n_vertices == 3:
        # Area = 0.5 * |cross product|
        cross = np.cross(
            coords[:, 1] - coords[:, 0],
            coords[:, 2] - coords[:, 0],
        )
        volumes = 0.5 * np.abs(cross)
        quality = 4.0 * np.sqrt(3.0) * volumes / np.sum(lengths_sq, axis=1)
    else:
        matrix = np.stack(
            [coords[:, 1] - coords[:, 0],
             coords[:, 2] - coords[:, 0],
             coords[:, 3] - coords[:, 0]], axis=1
        )
        volumes = np.abs(np.linalg.det(matrix)) / 6.0
        quality = 12.0 * np.power(3.0 * volumes, 2.0 / 3.0) / np.sum(lengths_sq, axis=1)

    return diameters, quality, volumes

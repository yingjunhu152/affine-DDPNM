"""2D-specific validation: wraps ddpnm_core.validation with 2D diagnostics."""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem

from ddpnm_core.io import topology_arrays
from ddpnm_core.reconstruction import mixed_solution_functions
from ddpnm_core.validation import (
    finite_element_error_analysis as _core_finite_element_error_analysis,
)


def finite_element_error_analysis(
    partition,
    solution,
    reference,
    quadrature_degree: int = 6,
) -> tuple[dict, dict]:
    """Strict broken-domain P2-P1 errors + 2D per-cell diagnostics.

    Returns ``(metrics, fields)`` where *fields* is a dict containing
    ``cell_areas``, ``velocity_error_cell_rms``, ``pressure_error_cell_rms``,
    ``pressure_error_cell_mean``, ``ddpnm_speed_cell_centroid``,
    ``fem_speed_cell_centroid``.
    """
    points, triangles = topology_arrays(partition.mesh)
    # compute triangle areas
    coords = points[triangles]
    cross = np.cross(
        coords[:, 1] - coords[:, 0],
        coords[:, 2] - coords[:, 0],
    )
    cell_areas = 0.5 * np.abs(cross)
    if np.any(cell_areas <= 0.0):
        raise RuntimeError("Non-positive triangle area in the error analysis.")

    metrics, velocity_cell_rms, pressure_cell_rms = (
        _core_finite_element_error_analysis(
            partition, solution, reference, cell_areas, quadrature_degree
        )
    )

    # 2D-specific per-cell speed and mean pressure error diagnostics
    u_reference, p_reference = mixed_solution_functions(
        reference.W, reference.solution
    )
    n_cells = len(triangles)
    ddpnm_speed = np.zeros(n_cells, dtype=float)
    fem_speed = np.zeros(n_cells, dtype=float)
    pressure_error_mean = np.zeros(n_cells, dtype=float)
    assigned = np.zeros(n_cells, dtype=bool)

    for response, local_solution in zip(
        solution.local_responses, solution.local_solutions, strict=True
    ):
        u_ddpnm, p_ddpnm = mixed_solution_functions(response.W, local_solution)
        from ddpnm_core.validation import restrict_same_mesh_nodal_function
        u_fem, _ = restrict_same_mesh_nodal_function(
            u_reference, u_ddpnm.function_space
        )
        p_fem, _ = restrict_same_mesh_nodal_function(
            p_reference, p_ddpnm.function_space
        )
        parent_cells = np.asarray(response.parent_cell_map, dtype=np.int32)
        centroid = np.mean(
            response.submesh.geometry.x[
                : response.submesh.topology.index_map(0).size_local, :
            ],
            axis=0,
        )
        # Cell centroids for speed evaluation
        tdim = response.submesh.topology.dim
        response.submesh.topology.create_connectivity(tdim, 0)
        c2v = response.submesh.topology.connectivity(tdim, 0)
        n_local = response.submesh.topology.index_map(tdim).size_local
        local_centroids = np.empty((n_local, response.submesh.geometry.dim))
        for cell in range(n_local):
            local_centroids[cell] = response.submesh.geometry.x[
                c2v.links(cell), :
            ].mean(axis=0)

        ddpnm_speed[parent_cells] = np.linalg.norm(
            np.asarray(
                u_ddpnm.eval(local_centroids, np.arange(n_local, dtype=np.int32)),
                dtype=float,
            ).reshape(-1, 2),
            axis=1,
        )
        fem_speed[parent_cells] = np.linalg.norm(
            np.asarray(
                u_fem.eval(local_centroids, np.arange(n_local, dtype=np.int32)),
                dtype=float,
            ).reshape(-1, 2),
            axis=1,
        )
        dp = p_ddpnm - p_fem
        for cell in range(n_local):
            pressure_error_mean[parent_cells[cell]] = float(
                dp.eval(
                    local_centroids[cell : cell + 1],
                    np.array([cell], dtype=np.int32),
                )[0]
            )
        assigned[parent_cells] = True

    if not np.all(assigned):
        raise RuntimeError("Some parent cells missing from 2D error diagnostics.")

    fields = {
        "cell_areas": cell_areas,
        "velocity_error_cell_rms": velocity_cell_rms,
        "pressure_error_cell_rms": pressure_cell_rms,
        "pressure_error_cell_mean": pressure_error_mean,
        "ddpnm_speed_cell_centroid": ddpnm_speed,
        "fem_speed_cell_centroid": fem_speed,
    }
    return metrics, fields

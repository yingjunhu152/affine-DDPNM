"""Field reconstruction from local DD-PNM solutions to global vertex arrays."""

from __future__ import annotations

import numpy as np
from basix.ufl import element
from dolfinx import fem
from scipy.spatial import cKDTree

from ddpnm_core.io import topology_vertex_coordinates


def p1_vertex_values(function: fem.Function, msh, components: int) -> np.ndarray:
    """Extract P1 dof values at topology vertices via cKDTree nearest match."""
    gdim = msh.geometry.dim
    vertex_coords = topology_vertex_coordinates(msh)
    dof_coords = function.function_space.tabulate_dof_coordinates()[:, :gdim]
    raw = np.asarray(function.x.array, dtype=float)
    if components == 1:
        values = raw.reshape(-1, 1)
    elif len(raw) == len(dof_coords) * components:
        values = raw.reshape(len(dof_coords), components)
    else:
        raise RuntimeError(
            f"Unexpected P1 vector layout: {len(raw)} values and "
            f"{len(dof_coords)} coordinate rows."
        )
    distances, indices = cKDTree(dof_coords).query(vertex_coords, k=1)
    if float(np.max(distances)) > 1.0e-9:
        raise RuntimeError(
            f"P1 vertex-to-dof mismatch: max distance {np.max(distances):.3e}."
        )
    result = values[np.asarray(indices, dtype=np.int32)]
    return result[:, 0] if components == 1 else result


def mixed_solution_functions(
    W: fem.FunctionSpace, solution: np.ndarray
) -> tuple[fem.Function, fem.Function]:
    wh = fem.Function(W)
    wh.x.array[:] = solution
    return wh.sub(0).collapse(), wh.sub(1).collapse()


def mixed_solution_to_p1(W, solution: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    msh = W.mesh
    gdim = msh.geometry.dim
    uh, ph = mixed_solution_functions(W, solution)
    cell = msh.basix_cell()
    V1 = fem.functionspace(msh, element("Lagrange", cell, 1, shape=(gdim,)))
    u1 = fem.Function(V1)
    u1.interpolate(uh)
    return p1_vertex_values(u1, msh, gdim), p1_vertex_values(ph, msh, 1)


def reconstruct_parent_vertices(
    partition, solution
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a two-sided vertex average strictly for plotting convenience."""
    msh = partition.mesh
    gdim = msh.geometry.dim
    n_vertices = msh.topology.index_map(0).size_local
    u_sum = np.zeros((n_vertices, gdim), dtype=float)
    p_sum = np.zeros(n_vertices, dtype=float)
    counts = np.zeros(n_vertices, dtype=np.int32)
    for response, local_solution in zip(
        solution.local_responses, solution.local_solutions, strict=True
    ):
        u_local, p_local = mixed_solution_to_p1(response.W, local_solution)
        for local_vertex, parent_vertex in enumerate(response.parent_vertex_map):
            u_sum[parent_vertex] += u_local[local_vertex]
            p_sum[parent_vertex] += p_local[local_vertex]
            counts[parent_vertex] += 1
    if np.any(counts == 0):
        raise RuntimeError(
            "Some global vertices were not reconstructed from local fields."
        )
    return u_sum / counts[:, None], p_sum / counts, counts


def reference_parent_vertices(reference) -> tuple[np.ndarray, np.ndarray]:
    return mixed_solution_to_p1(reference.W, reference.solution)


def reconstruct_piecewise_p1_cell_vertices(
    partition, solution, parent_tetrahedra: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell P1 vertex representation without interface averaging."""
    n_cells = len(parent_tetrahedra)
    u_cell = np.zeros((n_cells, 3), dtype=float)
    p_cell = np.zeros(n_cells, dtype=float)
    cell_to_local: dict[int, tuple[int, int]] = {}
    for response_index, response in enumerate(solution.local_responses):
        for local_cell, parent_cell in enumerate(response.parent_cell_map):
            cell_to_local[int(parent_cell)] = (response_index, local_cell)
    for parent_cell in range(n_cells):
        if parent_cell in cell_to_local:
            ridx, lcell = cell_to_local[parent_cell]
            response = solution.local_responses[ridx]
            local_sol = solution.local_solutions[ridx]
            u_local, p_local = mixed_solution_to_p1(response.W, local_sol)
            vertices = parent_tetrahedra[parent_cell]
            valid = vertices >= 0
            if np.all(valid):
                u_cell[parent_cell] = np.mean(u_local[valid], axis=0)
                p_cell[parent_cell] = np.mean(p_local[valid])
    return u_cell, p_cell

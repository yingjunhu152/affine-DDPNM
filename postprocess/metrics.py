"""Error norms, comparison metrics, and adaptive marking utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from basix.ufl import element
from dolfinx import fem, mesh as dmesh
from scipy.spatial import cKDTree

from postprocess.fields import mixed_solution_functions


# ---------------------------------------------------------------------------
# HierarchyError
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HierarchyError:
    velocity: float
    pressure: float
    combined: float
    pressure_shift: float


def hierarchy_error(
    candidate: tuple[np.ndarray, np.ndarray],
    target: tuple[np.ndarray, np.ndarray],
) -> HierarchyError:
    """Relative L² differences of velocity and mean-shifted pressure."""
    u, p = candidate
    ut, pt = target
    pressure_shift = float(np.mean(p - pt))
    velocity = float(np.linalg.norm(u - ut) / max(np.linalg.norm(ut), 1.0e-30))
    pressure = float(
        np.linalg.norm(p - pressure_shift - pt) / max(np.linalg.norm(pt), 1.0e-30)
    )
    return HierarchyError(
        velocity=velocity,
        pressure=pressure,
        combined=max(velocity, pressure),
        pressure_shift=pressure_shift,
    )


# ---------------------------------------------------------------------------
# Dörfler marking
# ---------------------------------------------------------------------------

def _dorfler_mark(
    indicators: np.ndarray,
    candidates: np.ndarray,
    theta: float,
    max_marked: int,
) -> tuple[int, ...]:
    """Dörfler marking: select candidates whose squared indicators
    dominate ``theta`` of the total, up to ``max_marked``."""
    if len(candidates) == 0:
        return ()
    values = indicators[candidates] ** 2
    order = np.argsort(values)[::-1]
    total = float(np.sum(values))
    if total <= 1.0e-30:
        return (int(candidates[order[0]]),)
    marked: list[int] = []
    accumulated = 0.0
    for local_index in order:
        marked.append(int(candidates[local_index]))
        accumulated += float(values[local_index])
        if accumulated >= theta * total or len(marked) >= max_marked:
            break
    return tuple(marked)


def _level_counts(levels: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(np.sum(levels == level)) for level in range(3))


# ---------------------------------------------------------------------------
# Strict broken-domain FE error analysis
# ---------------------------------------------------------------------------

def restrict_same_mesh_nodal_function(
    source: fem.Function, target_space: fem.FunctionSpace
) -> tuple[fem.Function, float]:
    """Restrict a Lagrange function to a conforming submesh by nodal matching."""
    gdim = source.function_space.mesh.geometry.dim
    source_coords = source.function_space.tabulate_dof_coordinates()[:, :gdim]
    target_coords = target_space.tabulate_dof_coordinates()[:, :gdim]
    distances, indices = cKDTree(source_coords).query(target_coords, k=1)
    maximum_distance = float(np.max(distances)) if len(distances) else 0.0
    if maximum_distance > 1.0e-10:
        raise RuntimeError(
            f"Global-to-local FE dof mismatch: max distance {maximum_distance:.3e}."
        )
    source_bs = source.function_space.dofmap.index_map_bs
    target_bs = target_space.dofmap.index_map_bs
    if source_bs != target_bs:
        raise RuntimeError("Global and local finite-element block sizes differ.")
    target = fem.Function(target_space)
    target.x.array.reshape(-1, target_bs)[:] = source.x.array.reshape(
        -1, source_bs
    )[np.asarray(indices, dtype=np.int32)]
    return target, maximum_distance


def _dg0_integrals(
    expression, submesh: dmesh.Mesh, quadrature_degree: int
) -> np.ndarray:
    Q0 = fem.functionspace(submesh, element("DG", submesh.basix_cell(), 0))
    test = ufl.TestFunction(Q0)
    dx = ufl.Measure(
        "dx", domain=submesh, metadata={"quadrature_degree": quadrature_degree}
    )
    vector = fem.assemble_vector(fem.form(expression * test * dx))
    return np.asarray(vector.array, dtype=float).copy()


def finite_element_error_analysis(
    partition,
    solution,
    reference,
    parent_volumes: np.ndarray,
    quadrature_degree: int = 6,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Strict broken-domain P2-P1 errors against the monolithic FEM solution.

    Returns ``(metrics, velocity_cell_rms, pressure_cell_rms)``.
    """
    u_reference, p_reference = mixed_solution_functions(
        reference.W, reference.solution
    )
    n_cells = partition.mesh.topology.index_map(
        partition.mesh.topology.dim
    ).size_local
    velocity_error_integrals = np.zeros(n_cells, dtype=float)
    pressure_error_integrals = np.zeros(n_cells, dtype=float)
    velocity_reference_integrals = np.zeros(n_cells, dtype=float)
    pressure_reference_integrals = np.zeros(n_cells, dtype=float)
    assigned = np.zeros(n_cells, dtype=bool)
    velocity_h1_error_sq = 0.0
    velocity_h1_reference_sq = 0.0
    ddpnm_divergence_sq = 0.0
    fem_divergence_sq = 0.0
    pressure_error_integral = 0.0
    pressure_reference_integral = 0.0
    maximum_mapping_distance = 0.0

    for response, local_solution in zip(
        solution.local_responses, solution.local_solutions, strict=True
    ):
        u_ddpnm, p_ddpnm = mixed_solution_functions(response.W, local_solution)
        u_fem, velocity_distance = restrict_same_mesh_nodal_function(
            u_reference, u_ddpnm.function_space
        )
        p_fem, pressure_distance = restrict_same_mesh_nodal_function(
            p_reference, p_ddpnm.function_space
        )
        maximum_mapping_distance = max(
            maximum_mapping_distance, velocity_distance, pressure_distance
        )
        du = u_ddpnm - u_fem
        dp = p_ddpnm - p_fem
        parent_cells = np.asarray(response.parent_cell_map, dtype=np.int32)
        local_velocity_error = _dg0_integrals(
            ufl.inner(du, du), response.submesh, quadrature_degree
        )
        local_pressure_error = _dg0_integrals(
            dp * dp, response.submesh, quadrature_degree
        )
        local_velocity_reference = _dg0_integrals(
            ufl.inner(u_fem, u_fem), response.submesh, quadrature_degree
        )
        local_pressure_reference = _dg0_integrals(
            p_fem * p_fem, response.submesh, quadrature_degree
        )
        for values in (
            local_velocity_error,
            local_pressure_error,
            local_velocity_reference,
            local_pressure_reference,
        ):
            if len(values) != len(parent_cells):
                raise RuntimeError("DG0 error integrals do not match local cells.")
        velocity_error_integrals[parent_cells] = local_velocity_error
        pressure_error_integrals[parent_cells] = local_pressure_error
        velocity_reference_integrals[parent_cells] = local_velocity_reference
        pressure_reference_integrals[parent_cells] = local_pressure_reference
        assigned[parent_cells] = True

        dx = ufl.Measure(
            "dx",
            domain=response.submesh,
            metadata={"quadrature_degree": quadrature_degree},
        )
        velocity_h1_error_sq += float(
            fem.assemble_scalar(
                fem.form(ufl.inner(ufl.grad(du), ufl.grad(du)) * dx)
            )
        )
        velocity_h1_reference_sq += float(
            fem.assemble_scalar(
                fem.form(ufl.inner(ufl.grad(u_fem), ufl.grad(u_fem)) * dx)
            )
        )
        ddpnm_divergence_sq += float(
            fem.assemble_scalar(fem.form(ufl.div(u_ddpnm) ** 2 * dx))
        )
        fem_divergence_sq += float(
            fem.assemble_scalar(fem.form(ufl.div(u_fem) ** 2 * dx))
        )
        pressure_error_integral += float(
            fem.assemble_scalar(fem.form(dp * dx))
        )
        pressure_reference_integral += float(
            fem.assemble_scalar(fem.form(p_fem * dx))
        )

    if not np.all(assigned):
        raise RuntimeError("Some parent cells are missing from the exact error analysis.")
    if np.any(parent_volumes <= 0.0):
        raise RuntimeError("Non-positive parent cell volume in the error analysis.")

    velocity_error_sq = float(np.sum(velocity_error_integrals))
    pressure_error_sq = float(np.sum(pressure_error_integrals))
    velocity_reference_sq = float(np.sum(velocity_reference_integrals))
    pressure_reference_sq = float(np.sum(pressure_reference_integrals))
    volume = float(np.sum(parent_volumes))
    pressure_mean_difference = pressure_error_integral / volume
    pressure_reference_mean = pressure_reference_integral / volume
    pressure_mean_aligned_error_sq = max(
        pressure_error_sq - pressure_error_integral**2 / volume, 0.0
    )
    pressure_centered_reference_sq = max(
        pressure_reference_sq - pressure_reference_integral**2 / volume, 0.0
    )
    ddpnm_outlet = float(solution.boundary_fluxes["outlet"])
    fem_outlet = float(reference.boundary_fluxes["outlet"])

    metrics: dict = {
        "same_parent_mesh_object": bool(reference.W.mesh is partition.mesh),
        "quadrature_degree": int(quadrature_degree),
        "maximum_global_to_local_dof_coordinate_mismatch": maximum_mapping_distance,
        "velocity_absolute_l2": float(np.sqrt(max(velocity_error_sq, 0.0))),
        "velocity_reference_l2": float(np.sqrt(max(velocity_reference_sq, 0.0))),
        "velocity_relative_l2": float(
            np.sqrt(velocity_error_sq / max(velocity_reference_sq, 1.0e-30))
        ),
        "velocity_relative_broken_h1_seminorm": float(
            np.sqrt(velocity_h1_error_sq / max(velocity_h1_reference_sq, 1.0e-30))
        ),
        "pressure_raw_absolute_l2": float(np.sqrt(max(pressure_error_sq, 0.0))),
        "pressure_raw_relative_l2": float(
            np.sqrt(pressure_error_sq / max(pressure_reference_sq, 1.0e-30))
        ),
        "pressure_mean_difference_ddpnm_minus_fem": float(pressure_mean_difference),
        "pressure_reference_mean": float(pressure_reference_mean),
        "pressure_mean_aligned_relative_l2": float(
            np.sqrt(
                pressure_mean_aligned_error_sq
                / max(pressure_centered_reference_sq, 1.0e-30)
            )
        ),
        "ddpnm_broken_divergence_l2": float(np.sqrt(max(ddpnm_divergence_sq, 0.0))),
        "fem_divergence_l2": float(np.sqrt(max(fem_divergence_sq, 0.0))),
        "ddpnm_outlet_flux": ddpnm_outlet,
        "fem_outlet_flux": fem_outlet,
        "outlet_flux_relative_error": float(
            abs(ddpnm_outlet - fem_outlet) / max(abs(fem_outlet), 1.0e-30)
        ),
        "cell_error_integral_sum_velocity": velocity_error_sq,
        "cell_error_integral_sum_pressure": pressure_error_sq,
    }
    velocity_cell_rms = np.sqrt(
        np.maximum(velocity_error_integrals, 0.0) / parent_volumes
    )
    pressure_cell_rms = np.sqrt(
        np.maximum(pressure_error_integrals, 0.0) / parent_volumes
    )
    return metrics, velocity_cell_rms, pressure_cell_rms

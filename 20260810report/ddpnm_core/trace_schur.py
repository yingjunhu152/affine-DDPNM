"""Exact finite-element trace Schur complement.

This module computes the *exact* Schur complement of the full Taylor–Hood
Stokes system, partitioning degrees of freedom into wall (fixed), interface
trace, and interior.  It serves as the reference oracle against which all
reduced interface spaces (P0-DDPNM, P0-vector, P1-vector) are benchmarked.

Originally written for 2-D (``ddpnm2d/exact_schur.py``); now generalised to
both dimensions and shared through ``ddpnm_core``.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import ufl
from dolfinx import fem
from scipy.sparse.linalg import MatrixRankWarning, splu, spsolve

from ddpnm_core.constants import WALL_TAG
from ddpnm_core.fem_utils import (
    global_boundary_tags,
    mixed_stokes_space,
    to_numpy_vector,
    to_scipy_matrix,
)


@dataclass
class ExactSchurSolution:
    """Result of the exact FE-trace Schur complement solve."""

    W: fem.FunctionSpace
    solution: np.ndarray
    monolithic_solution: np.ndarray
    schur_matrix: np.ndarray
    schur_rhs: np.ndarray
    fixed_dofs: np.ndarray
    interface_dofs: np.ndarray
    interior_dofs: np.ndarray
    interface_dofs_by_interface: tuple[np.ndarray, ...]
    schur_symmetry_error: float
    schur_relative_residual: float
    interior_relative_residual: float
    global_relative_residual: float
    monolithic_relative_difference: float


# ---------------------------------------------------------------------------
# DOF partitioning
# ---------------------------------------------------------------------------

def _wall_parent_dofs(
    W: fem.FunctionSpace, wall_facets: np.ndarray
) -> tuple[np.ndarray, fem.DirichletBC]:
    """Velocity DOFs on wall facets → fixed list + Dirichlet BC."""
    V0, _ = W.sub(0).collapse()
    wall_map = fem.locate_dofs_topological(
        (W.sub(0), V0), W.mesh.topology.dim - 1, wall_facets
    )
    zero = fem.Function(V0)
    bc = fem.dirichletbc(zero, wall_map, W.sub(0))
    if isinstance(wall_map, tuple):
        wall_parent = np.asarray(wall_map[0], dtype=np.int32)
    else:
        wall_parent = np.asarray(wall_map, dtype=np.int32)
        if wall_parent.ndim == 2:
            wall_parent = (
                wall_parent[0]
                if wall_parent.shape[0] == 2 and wall_parent.shape[1] != 2
                else wall_parent[:, 0]
            )
    return np.unique(wall_parent), bc


def _interface_parent_dofs(
    partition,
    W: fem.FunctionSpace,
    fixed_dofs: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Per-interface free Taylor–Hood trace DOFs (velocity + pressure)."""
    fdim = partition.mesh.topology.dim - 1
    fixed = set(int(i) for i in fixed_dofs)
    groups: list[np.ndarray] = []
    for interface_id in range(len(partition.interface_pairs)):
        facets = np.flatnonzero(
            partition.facet_interface_ids == interface_id
        ).astype(np.int32)
        velocity = np.asarray(
            fem.locate_dofs_topological(W.sub(0), fdim, facets), dtype=np.int32
        ).ravel()
        pressure = np.asarray(
            fem.locate_dofs_topological(W.sub(1), fdim, facets), dtype=np.int32
        ).ravel()
        dofs = np.unique(np.concatenate([velocity, pressure]))
        dofs = np.asarray(
            [int(i) for i in dofs if int(i) not in fixed], dtype=np.int32
        )
        if len(dofs) == 0:
            raise RuntimeError(
                f"Interface {interface_id} has no free Taylor–Hood trace DOFs."
            )
        groups.append(dofs)
    return tuple(groups)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def solve_exact_fe_schur(
    partition,
    viscosity: float = 1.0,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
    pressure_stabilization: float = 1.0e-10,
) -> ExactSchurSolution:
    """Solve the global P2-P1 Stokes system by explicit FE-trace Schur complement.

    The Schur unknowns are all free P2 velocity and P1 pressure trace DOFs
    on analytic throat interfaces.  This is a correctness implementation,
    not a speed claim.
    """
    msh = partition.mesh
    tags = global_boundary_tags(msh)
    W = mixed_stokes_space(msh)
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    dx = ufl.dx(domain=msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    normal = ufl.FacetNormal(msh)
    a = (
        viscosity * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        - p * ufl.div(v) * dx
        - q * ufl.div(u) * dx
        - pressure_stabilization * p * q * dx
    )
    L = (
        -inlet_pressure * ufl.dot(normal, v) * ds(2)
        - outlet_pressure * ufl.dot(normal, v) * ds(3)
    )
    wall_dofs, bc = _wall_parent_dofs(W, tags.find(WALL_TAG))
    a_form = fem.form(a)
    A = to_scipy_matrix(fem.assemble_matrix(a_form, bcs=[bc])).tocsr()
    b_vector = fem.assemble_vector(fem.form(L))
    fem.apply_lifting(b_vector.array, [a_form], [[bc]])
    fem.set_bc(b_vector.array, [bc])
    b = to_numpy_vector(b_vector)

    groups = _interface_parent_dofs(partition, W, wall_dofs)
    interface = np.unique(np.concatenate(groups)).astype(np.int32)
    all_dofs = np.arange(A.shape[0], dtype=np.int32)
    free_mask = np.ones(A.shape[0], dtype=bool)
    free_mask[wall_dofs] = False
    interface_mask = np.zeros(A.shape[0], dtype=bool)
    interface_mask[interface] = True
    interior = all_dofs[free_mask & ~interface_mask]

    if len(interior) + len(interface) + len(wall_dofs) != A.shape[0]:
        raise RuntimeError(
            "Fixed/interface/interior Schur partition is incomplete."
        )
    if np.intersect1d(interface, interior).size:
        raise RuntimeError(
            "Interface and interior Schur DOF sets overlap."
        )

    Aii = A[np.ix_(interior, interior)].tocsc()
    Aib = A[np.ix_(interior, interface)].tocsc()
    Abi = A[np.ix_(interface, interior)].tocsr()
    Abb = A[np.ix_(interface, interface)].toarray()
    bi = b[interior]
    bb = b[interface]

    try:
        factor = splu(Aii)
    except RuntimeError as exc:
        raise RuntimeError(
            "The exact-Schur interior block is singular."
        ) from exc

    yi = factor.solve(bi)
    transfer = factor.solve(Aib.toarray())
    S = Abb - Abi @ transfer
    g = bb - Abi @ yi
    xb = np.linalg.solve(S, g)
    xi = yi - transfer @ xb

    solution = np.zeros(A.shape[0], dtype=float)
    solution[interface] = xb
    solution[interior] = xi
    solution[wall_dofs] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        monolithic = np.asarray(spsolve(A, b), dtype=float)

    if not np.all(np.isfinite(monolithic)):
        raise RuntimeError(
            "Monolithic comparison solve returned non-finite values."
        )

    schur_scale = max(float(np.linalg.norm(S)), 1.0e-30)
    schur_residual = float(
        np.linalg.norm(S @ xb - g) / max(np.linalg.norm(g), 1.0e-30)
    )
    interior_residual = float(
        np.linalg.norm(Aii @ xi + Aib @ xb - bi)
        / max(np.linalg.norm(bi), 1.0)
    )
    free = np.concatenate([interior, interface])
    global_residual = float(
        np.linalg.norm((A @ solution - b)[free])
        / max(np.linalg.norm(b[free]), 1.0)
    )
    comparison = float(
        np.linalg.norm(solution - monolithic)
        / max(np.linalg.norm(monolithic), 1.0e-30)
    )

    return ExactSchurSolution(
        W=W,
        solution=solution,
        monolithic_solution=monolithic,
        schur_matrix=np.asarray(S),
        schur_rhs=np.asarray(g),
        fixed_dofs=wall_dofs,
        interface_dofs=interface,
        interior_dofs=interior,
        interface_dofs_by_interface=groups,
        schur_symmetry_error=float(np.linalg.norm(S - S.T) / schur_scale),
        schur_relative_residual=schur_residual,
        interior_relative_residual=interior_residual,
        global_relative_residual=global_residual,
        monolithic_relative_difference=comparison,
    )

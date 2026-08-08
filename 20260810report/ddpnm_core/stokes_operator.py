"""Dimension-agnostic local Stokes operator.

Every local Stokes solve — whether for the classic P0-DDPNM (solver.py) or
the hierarchical primitive library (hierarchy.py), in 2D or 3D — assembles
the identical Taylor–Hood bilinear form and wall boundary conditions.  This
module captures that shared operator in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from ddpnm_core.constants import PORT_TAG_BASE, WALL_TAG
from ddpnm_core.fem_utils import (
    build_local_submesh,
    mixed_stokes_space,
    to_numpy_vector,
    to_scipy_matrix,
)
from ddpnm_core.solver_types import PortInfo


@dataclass
class LocalStokesOperator:
    """Factorized Taylor–Hood Stokes operator on a single pore submesh.

    The bilinear form is ::

        a = viscosity · ⟨∇u, ∇v⟩ − p∇·v − q∇·u − stab · p·q

    with zero Dirichlet BCs for velocity on ``WALL_TAG`` facets.  The
    operator owns the assembled sparse matrix **A** and its SuperLU
    factorisation so that multiple right-hand sides can be solved cheaply.
    """

    pore_id: int
    submesh: dmesh.Mesh
    parent_cell_map: np.ndarray
    parent_vertex_map: np.ndarray
    ports: tuple[PortInfo, ...]
    facet_tags: dmesh.MeshTags
    W: fem.FunctionSpace
    normal: ufl.FacetNormal
    a_form: fem.Form
    bcs: list[fem.DirichletBC]
    A: csr_matrix
    factor: splu
    ndofs: int

    @property
    def dim(self) -> int:
        """Geometric dimension of the pore subdomain."""
        return self.submesh.geometry.dim

    def assemble_load(self, expression) -> np.ndarray:
        """Assemble *expression* · v ds, apply lifting and BCs, return rhs."""
        L_form = fem.form(expression)
        b = fem.assemble_vector(L_form)
        fem.apply_lifting(b.array, [self.a_form], [self.bcs])
        fem.set_bc(b.array, self.bcs)
        return to_numpy_vector(b)

    def solve(self, loads: np.ndarray) -> np.ndarray:
        """Solve for each right-hand-side column in *loads*.

        *loads* shape = ``(ndofs, k)``.
        """
        if loads.ndim == 1:
            return self.factor.solve(loads)
        return self.factor.solve(loads)


def build_local_stokes_operator(
    partition,
    pore_id: int,
    viscosity: float = 1.0,
    pressure_stabilization: float = 0.0,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
) -> LocalStokesOperator:
    """Create and factorize the local Stokes operator for *pore_id*."""
    submesh, parent_cells, parent_vertices, ports, facet_tags = build_local_submesh(
        partition, pore_id, inlet_pressure, outlet_pressure
    )
    W = mixed_stokes_space(submesh)
    (u, p_var) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    dx = ufl.dx(domain=submesh)
    ds_measure = ufl.Measure("ds", domain=submesh, subdomain_data=facet_tags)
    n = ufl.FacetNormal(submesh)

    a = (
        viscosity * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        - p_var * ufl.div(v) * dx
        - q * ufl.div(u) * dx
        - pressure_stabilization * p_var * q * dx
    )
    a_form = fem.form(a)

    wall_facets = facet_tags.find(WALL_TAG)
    V0, _ = W.sub(0).collapse()
    wall_dofs = fem.locate_dofs_topological(
        (W.sub(0), V0), submesh.topology.dim - 1, wall_facets
    )
    zero = fem.Function(V0)
    bcs = [fem.dirichletbc(zero, wall_dofs, W.sub(0))]

    A = to_scipy_matrix(fem.assemble_matrix(a_form, bcs=bcs))
    try:
        factor = splu(A.tocsc())
    except RuntimeError as exc:
        raise RuntimeError(
            f"Local Stokes matrix is singular in pore {pore_id}; "
            "increase pressure_stabilization slightly."
        ) from exc

    return LocalStokesOperator(
        pore_id=pore_id,
        submesh=submesh,
        parent_cell_map=parent_cells,
        parent_vertex_map=parent_vertices,
        ports=ports,
        facet_tags=facet_tags,
        W=W,
        normal=n,
        a_form=a_form,
        bcs=bcs,
        A=A,
        factor=factor,
        ndofs=A.shape[0],
    )

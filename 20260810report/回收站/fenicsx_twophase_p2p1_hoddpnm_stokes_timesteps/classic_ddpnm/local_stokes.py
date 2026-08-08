from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem
from scipy.sparse.linalg import spsolve

from .geometry import PORT_TAG_BASE, WALL_TAG, Pore
from .linalg import to_numpy_vector, to_scipy_matrix
from .mesh_gmsh import build_pore_mesh


@dataclass
class LocalResponse:
    pore_id: int
    port_ids: list[int]
    port_kinds: list[str]
    port_pressures: list[float | None]
    G: np.ndarray
    responses: np.ndarray
    ndofs: int


def solve_local_responses(
    pore: Pore,
    h: float = 0.09,
    port_half_width: float = 0.18,
    pressure_stabilization: float = 1.0e-10,
) -> LocalResponse:
    """Solve one Stokes response problem per port pressure basis."""
    domain, facet_tags = build_pore_mesh(pore, h=h, port_half_width=port_half_width)
    cell = domain.basix_cell()
    velocity_el = element("Lagrange", cell, 2, shape=(domain.geometry.dim,))
    pressure_el = element("Lagrange", cell, 1)
    W = fem.functionspace(domain, mixed_element([velocity_el, pressure_el]))

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    dx = ufl.dx(domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    n = ufl.FacetNormal(domain)

    a = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        - p * ufl.div(v) * dx
        - q * ufl.div(u) * dx
        - pressure_stabilization * p * q * dx
    )

    wall_facets = facet_tags.find(WALL_TAG)
    V0, _ = W.sub(0).collapse()
    wall_dofs = fem.locate_dofs_topological(
        (W.sub(0), V0), domain.topology.dim - 1, wall_facets
    )
    zero_u = fem.Function(V0)
    bcs = [fem.dirichletbc(zero_u, wall_dofs, W.sub(0))]

    A = to_scipy_matrix(fem.assemble_matrix(fem.form(a), bcs=bcs))
    port_count = len(pore.ports)
    responses = []
    flux = np.zeros((port_count, port_count), dtype=float)

    for source_index, port in enumerate(pore.ports):
        traction = ufl.as_vector([-component for component in port.normal])
        L = ufl.dot(traction, v) * ds(PORT_TAG_BASE + source_index)
        b = to_numpy_vector(fem.assemble_vector(fem.form(L)))
        fem.apply_lifting(b, [fem.form(a)], [bcs])
        fem.set_bc(b, bcs)

        sol = spsolve(A, b)
        responses.append(sol)

        wh = fem.Function(W)
        wh.x.array[:] = sol
        uh = wh.sub(0).collapse()

        for target_index in range(port_count):
            form = fem.form(ufl.dot(uh, n) * ds(PORT_TAG_BASE + target_index))
            flux[target_index, source_index] = fem.assemble_scalar(form)

    return LocalResponse(
        pore_id=pore.id,
        port_ids=[p.id for p in pore.ports],
        port_kinds=[p.kind for p in pore.ports],
        port_pressures=[p.pressure for p in pore.ports],
        G=flux,
        responses=np.column_stack(responses) if responses else np.empty((A.shape[0], 0)),
        ndofs=A.shape[0],
    )

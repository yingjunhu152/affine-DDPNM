from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem
from scipy.sparse.linalg import splu

from .geometry import PORT_TAG_BASE, WALL_TAG, Pore
from .linalg import to_numpy_vector, to_scipy_matrix
from .mesh_gmsh import build_pore_mesh


@dataclass
class PortNodes:
    port_index: int
    port_id: int
    kind: str
    pressure: float | None
    normal: tuple[float, float, float]
    dofs: np.ndarray
    coords: np.ndarray
    params: np.ndarray


@dataclass
class LocalHOResponse:
    pore_id: int
    port_nodes: list[PortNodes]
    G: np.ndarray
    responses: np.ndarray
    ndofs: int


def solve_local_ho_responses(
    pore: Pore,
    h: float = 0.28,
    port_half_width: float = 0.55,
    pressure_stabilization: float = 1.0e-10,
    throat_radius: float = 0.13,
    stub_length: float = 0.46,
) -> LocalHOResponse:
    """Solve nodal normal-pressure interface responses for one pore.

    Each pressure node on a port end disk defines one interface pressure basis
    phi_m. For oblique PNM throats the Cartesian background mesh can cut a
    stair-step end face with different raw FE nodes on the two adjacent pores,
    so the interface basis is evaluated from a stable 3x3 reference-disk
    template in the port's tangent frame.
    """
    domain, facet_tags = build_pore_mesh(
        pore,
        h=h,
        port_half_width=port_half_width,
        throat_radius=throat_radius,
        stub_length=stub_length,
    )
    cell = domain.basix_cell()
    velocity_el = element("Lagrange", cell, 2, shape=(domain.geometry.dim,))
    pressure_el = element("Lagrange", cell, 1)
    W = fem.functionspace(domain, mixed_element([velocity_el, pressure_el]))
    S = fem.functionspace(domain, ("Lagrange", 1))

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
    solver = splu(A.tocsc())

    port_nodes = collect_port_nodes(
        domain,
        facet_tags,
        S,
        pore,
        throat_radius=throat_radius,
        stub_length=stub_length,
    )
    basis_meta: list[tuple[int, int]] = []
    for port_pos, nodes in enumerate(port_nodes):
        for local_node_pos in range(len(nodes.params)):
            basis_meta.append((port_pos, local_node_pos))

    nbasis = len(basis_meta)
    responses = []
    flux = np.zeros((nbasis, nbasis), dtype=float)

    phi_expressions: list[ufl.core.expr.Expr] = []
    for port_pos, local_node_pos in basis_meta:
        port = pore.ports[port_pos]
        phi_expressions.append(
            template_phi(
                domain,
                normal=np.asarray(port.normal, dtype=float),
                center=np.asarray(pore.center, dtype=float),
                param=port_nodes[port_pos].params[local_node_pos],
                stub_length=stub_length,
                node_span=0.9 * throat_radius,
            )
        )

    for source_index, (source_port_pos, _) in enumerate(basis_meta):
        port = pore.ports[source_port_pos]
        phi = phi_expressions[source_index]
        traction = -phi * ufl.as_vector([component for component in port.normal])
        L = ufl.dot(traction, v) * ds(PORT_TAG_BASE + source_port_pos)
        b = to_numpy_vector(fem.assemble_vector(fem.form(L)))
        fem.apply_lifting(b, [fem.form(a)], [bcs])
        fem.set_bc(b, bcs)

        sol = solver.solve(b)
        responses.append(sol)

        wh = fem.Function(W)
        wh.x.array[:] = sol
        uh = wh.sub(0).collapse()

        for target_index, (target_port_pos, _) in enumerate(basis_meta):
            phi_out = phi_expressions[target_index]
            form = fem.form(
                ufl.dot(uh, n)
                * phi_out
                * ds(PORT_TAG_BASE + target_port_pos)
            )
            flux[target_index, source_index] = fem.assemble_scalar(form)

    return LocalHOResponse(
        pore_id=pore.id,
        port_nodes=port_nodes,
        G=flux,
        responses=np.column_stack(responses) if responses else np.empty((A.shape[0], 0)),
        ndofs=A.shape[0],
    )


def collect_port_nodes(
    domain,
    facet_tags,
    S,
    pore: Pore,
    throat_radius: float,
    stub_length: float,
) -> list[PortNodes]:
    port_nodes: list[PortNodes] = []
    params = template_params(node_span=0.9 * throat_radius)
    for port_index, port in enumerate(pore.ports):
        facets = facet_tags.find(PORT_TAG_BASE + port_index)
        dofs = fem.locate_dofs_topological(S, domain.topology.dim - 1, facets)
        dofs = np.asarray(sorted(set(int(d) for d in dofs)), dtype=np.int32)
        node_coords = template_coords(
            np.asarray(pore.center, dtype=float),
            np.asarray(port.normal, dtype=float),
            params,
            stub_length=stub_length,
        )
        port_nodes.append(
            PortNodes(
                port_index=port_index,
                port_id=port.id,
                kind=port.kind,
                pressure=port.pressure,
                normal=port.normal,
                dofs=dofs,
                coords=node_coords,
                params=params.copy(),
            )
        )
    return port_nodes


def template_params(node_span: float) -> np.ndarray:
    values = np.asarray([-node_span, 0.0, node_span], dtype=float)
    return np.asarray([(s, t) for s in values for t in values], dtype=float)


def template_coords(
    center: np.ndarray,
    normal: np.ndarray,
    params: np.ndarray,
    stub_length: float,
) -> np.ndarray:
    e1, e2 = canonical_tangent_axes(normal)
    n = normal / np.linalg.norm(normal)
    origin = center + stub_length * n
    return origin + params[:, :1] * e1 + params[:, 1:] * e2


def template_phi(
    domain,
    normal: np.ndarray,
    center: np.ndarray,
    param: np.ndarray,
    stub_length: float,
    node_span: float,
):
    e1, e2 = canonical_tangent_axes(normal)
    n = normal / np.linalg.norm(normal)
    origin = center + stub_length * n
    x = ufl.SpatialCoordinate(domain)
    rel = ufl.as_vector([x[i] - float(origin[i]) for i in range(3)])
    s = sum(float(e1[i]) * rel[i] for i in range(3))
    t = sum(float(e2[i]) * rel[i] for i in range(3))
    return lagrange_1d(s, float(param[0]), node_span) * lagrange_1d(t, float(param[1]), node_span)


def lagrange_1d(coord, center: float, span: float):
    eps = 1.0e-12
    if center < -eps:
        return coord * (coord - span) / (2.0 * span * span)
    if center > eps:
        return coord * (coord + span) / (2.0 * span * span)
    return 1.0 - coord * coord / (span * span)


def port_plane_params(coords: np.ndarray, normal: np.ndarray, center: np.ndarray) -> np.ndarray:
    e1, e2 = canonical_tangent_axes(normal)
    origin = coords.mean(axis=0)
    rel = coords - origin
    return np.column_stack((rel @ e1, rel @ e2))


def canonical_tangent_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal / np.linalg.norm(normal)
    base = n.copy()
    for value in base:
        if abs(value) > 1.0e-12:
            if value < 0.0:
                base = -base
            break
    axis = np.eye(3)[int(np.argmin(np.abs(base)))]
    e1 = np.cross(base, axis)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(base, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2

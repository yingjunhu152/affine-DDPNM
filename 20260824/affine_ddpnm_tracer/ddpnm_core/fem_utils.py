"""Dimension-agnostic FEniCSx/dolfinx utilities for DD-PNM.

These functions use ``msh.geometry.dim`` instead of hardcoded ``3`` / ``2``
so they work identically for 2D and 3D meshes.
"""

from __future__ import annotations

import warnings

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh as dmesh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import (
    LinearOperator,
    MatrixRankWarning,
    gmres,
    spilu,
    splu,
    spsolve,
)

from .constants import WALL_TAG, PORT_TAG_BASE

# ---------------------------------------------------------------------------
# Cache: avoid rebuilding the parent facet lookup 100× per library build
# ---------------------------------------------------------------------------

_parent_facet_lookup_cache: dict[int, dict[tuple[int, ...], int]] = {}


# ---------------------------------------------------------------------------
# Sparse / vector helpers (identical in 2D and 3D)
# ---------------------------------------------------------------------------


def to_scipy_matrix(matrix) -> csr_matrix:
    if hasattr(matrix, "to_scipy"):
        return matrix.to_scipy().tocsr()
    return csr_matrix(matrix.to_dense())


def to_numpy_vector(vector) -> np.ndarray:
    if hasattr(vector, "array"):
        return np.asarray(vector.array, dtype=float).copy()
    return np.asarray(vector, dtype=float).copy()


# ---------------------------------------------------------------------------
# Mesh topology helpers
# ---------------------------------------------------------------------------


def parent_facet_lookup(msh: dmesh.Mesh) -> dict[tuple[int, ...], int]:
    mesh_id = id(msh)
    cached = _parent_facet_lookup_cache.get(mesh_id)
    if cached is not None:
        return cached
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    n_facets = msh.topology.index_map(fdim).size_local
    result: dict[tuple[int, ...], int] = {}
    for facet in range(n_facets):
        result[tuple(sorted(int(v) for v in f2v.links(facet)))] = facet
    _parent_facet_lookup_cache[mesh_id] = result
    return result


# ---------------------------------------------------------------------------
# Submesh construction (dimension-agnostic)
# ---------------------------------------------------------------------------


def build_local_submesh(
    partition,  # PartitionData-like protocol
    pore_id: int,
    inlet_pressure: float,
    outlet_pressure: float,
) -> tuple[dmesh.Mesh, np.ndarray, np.ndarray, tuple, dmesh.MeshTags]:
    """Create a local submesh for one pore region.

    Returns ``(submesh, mapped_cells, mapped_vertices, ports, facet_tags)``.
    Uses ``msh.geometry.dim`` so the same code works for 2D triangles and
    3D tetrahedra.
    """
    from .solver_types import PortInfo

    parent = partition.mesh
    tdim = parent.topology.dim
    fdim = tdim - 1
    gdim = parent.geometry.dim  # 2 or 3

    cells = np.flatnonzero(partition.cell_labels == pore_id).astype(np.int32)
    if not len(cells):
        raise RuntimeError(f"Pore {pore_id} has no cells.")

    submesh, cell_map, vertex_map, _ = dmesh.create_submesh(parent, tdim, cells)
    submesh.topology.create_connectivity(fdim, 0)
    submesh.topology.create_connectivity(fdim, tdim)

    lookup = parent_facet_lookup(parent)

    n_sub_vertices = submesh.topology.index_map(0).size_local
    mapped_vertices = np.asarray(
        vertex_map.sub_topology_to_topology(
            np.arange(n_sub_vertices, dtype=np.int32), False
        ),
        dtype=np.int32,
    )
    n_sub_cells = submesh.topology.index_map(tdim).size_local
    mapped_cells = np.asarray(
        cell_map.sub_topology_to_topology(
            np.arange(n_sub_cells, dtype=np.int32), False
        ),
        dtype=np.int32,
    )

    sub_f2v = submesh.topology.connectivity(fdim, 0)
    exterior = dmesh.exterior_facet_indices(submesh.topology)

    by_interface: dict[int, list[int]] = {}
    inlet: list[int] = []
    outlet: list[int] = []
    walls: list[int] = []
    parent_by_sub: dict[int, int] = {}

    for sub_facet in exterior:
        sub_vertices = sub_f2v.links(int(sub_facet))
        key = tuple(sorted(int(mapped_vertices[v]) for v in sub_vertices))
        parent_facet = lookup.get(key)
        if parent_facet is None:
            raise RuntimeError(
                "Could not map a local boundary facet to the parent mesh."
            )
        parent_by_sub[int(sub_facet)] = int(parent_facet)
        interface_id = int(partition.facet_interface_ids[parent_facet])
        if interface_id >= 0:
            by_interface.setdefault(interface_id, []).append(int(sub_facet))
            continue
        midpoint = submesh.geometry.x[sub_vertices, :gdim].mean(axis=0)
        if midpoint[0] <= 1.0e-8:
            inlet.append(int(sub_facet))
        elif midpoint[0] >= 1.0 - 1.0e-8:
            outlet.append(int(sub_facet))
        else:
            walls.append(int(sub_facet))

    ports: list[PortInfo] = []
    port_facets: list[list[int]] = []
    for interface_id in sorted(by_interface):
        facets = sorted(by_interface[interface_id])
        ports.append(
            PortInfo(
                kind="interface",
                global_interface=interface_id,
                pressure=None,
                parent_facets=tuple(parent_by_sub[f] for f in facets),
            )
        )
        port_facets.append(facets)
    if inlet:
        ports.append(
            PortInfo(
                kind="inlet",
                global_interface=None,
                pressure=float(inlet_pressure),
                parent_facets=tuple(parent_by_sub[f] for f in sorted(inlet)),
            )
        )
        port_facets.append(sorted(inlet))
    if outlet:
        ports.append(
            PortInfo(
                kind="outlet",
                global_interface=None,
                pressure=float(outlet_pressure),
                parent_facets=tuple(parent_by_sub[f] for f in sorted(outlet)),
            )
        )
        port_facets.append(sorted(outlet))
    if not ports:
        raise RuntimeError(f"Pore {pore_id} has no coupling or pressure port.")

    tag_by_facet = {int(f): WALL_TAG for f in exterior}
    for local_port, facets in enumerate(port_facets):
        for facet in facets:
            tag_by_facet[int(facet)] = PORT_TAG_BASE + local_port
    indices = np.asarray(sorted(tag_by_facet), dtype=np.int32)
    values = np.asarray([tag_by_facet[int(f)] for f in indices], dtype=np.int32)
    tags = dmesh.meshtags(submesh, fdim, indices, values)
    return submesh, mapped_cells, mapped_vertices, tuple(ports), tags


# ---------------------------------------------------------------------------
# Mixed function space (dimension-agnostic)
# ---------------------------------------------------------------------------

def mixed_stokes_space(msh: dmesh.Mesh) -> fem.FunctionSpace:
    """Taylor–Hood [P2]^d–P1 mixed space.  The velocity dimension is
    inferred from ``msh.geometry.dim``."""
    gdim = msh.geometry.dim
    cell = msh.basix_cell()
    velocity = element("Lagrange", cell, 2, shape=(gdim,))
    pressure = element("Lagrange", cell, 1)
    return fem.functionspace(msh, mixed_element([velocity, pressure]))


# ---------------------------------------------------------------------------
# Global boundary tagging (dimension-agnostic)
# ---------------------------------------------------------------------------

def global_boundary_tags(msh: dmesh.Mesh) -> dmesh.MeshTags:
    """Tag exterior facets: 2 = inlet (x≈0), 3 = outlet (x≈1), else WALL."""
    fdim = msh.topology.dim - 1
    gdim = msh.geometry.dim
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    exterior = dmesh.exterior_facet_indices(msh.topology)
    values: list[int] = []
    for facet in exterior:
        midpoint = msh.geometry.x[f2v.links(int(facet)), :gdim].mean(axis=0)
        if midpoint[0] <= 1.0e-8:
            values.append(2)
        elif midpoint[0] >= 1.0 - 1.0e-8:
            values.append(3)
        else:
            values.append(WALL_TAG)
    order = np.argsort(exterior)
    return dmesh.meshtags(
        msh,
        fdim,
        np.asarray(exterior, dtype=np.int32)[order],
        np.asarray(values, dtype=np.int32)[order],
    )


# ---------------------------------------------------------------------------
# Monolithic FEM reference solver (dimension-agnostic)
# ---------------------------------------------------------------------------

def solve_reference(
    msh: dmesh.Mesh,
    viscosity: float = 1.0,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
    pressure_stabilization: float = 0.0,
    iterative_threshold: int = 100_000,
    iterative_rtol: float = 1.0e-9,
    iterative_restart: int = 60,
    iterative_maxiter: int = 120,
    ilu_drop_tolerance: float = 2.0e-3,
    ilu_fill_factor: float = 6.0,
):
    """Solve the monolithic Taylor–Hood Stokes problem.

    Returns a ``ReferenceSolution`` (imported from the calling package's
    solver module for its specific dataclass).
    """
    from ddpnm_core.solver_types import ReferenceSolution

    gdim = msh.geometry.dim

    tags = global_boundary_tags(msh)
    W = mixed_stokes_space(msh)
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    dx = ufl.dx(domain=msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    n = ufl.FacetNormal(msh)
    a = (
        viscosity * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        - p * ufl.div(v) * dx
        - q * ufl.div(u) * dx
        - pressure_stabilization * p * q * dx
    )
    L = (
        -inlet_pressure * ufl.dot(n, v) * ds(2)
        - outlet_pressure * ufl.dot(n, v) * ds(3)
    )
    wall_facets = tags.find(WALL_TAG)
    V0, _ = W.sub(0).collapse()
    wall_dofs = fem.locate_dofs_topological(
        (W.sub(0), V0), msh.topology.dim - 1, wall_facets
    )
    zero = fem.Function(V0)
    bcs = [fem.dirichletbc(zero, wall_dofs, W.sub(0))]
    a_form = fem.form(a)
    A = to_scipy_matrix(fem.assemble_matrix(a_form, bcs=bcs))
    b = fem.assemble_vector(fem.form(L))
    fem.apply_lifting(b.array, [a_form], [bcs])
    fem.set_bc(b.array, bcs)
    rhs = to_numpy_vector(b)
    residual_history: list[float] = []

    if A.shape[0] < iterative_threshold:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            solution = spsolve(A, rhs)
        solver_method = "SciPy SuperLU sparse direct"
        iterations = 1
        final_preconditioned_residual = 0.0
    else:
        V, velocity_to_mixed = W.sub(0).collapse()
        Q, pressure_to_mixed = W.sub(1).collapse()
        velocity_to_mixed = np.asarray(velocity_to_mixed, dtype=np.int32)
        pressure_to_mixed = np.asarray(pressure_to_mixed, dtype=np.int32)
        velocity_block = A[velocity_to_mixed][:, velocity_to_mixed].tocsr()
        block_size = V.dofmap.index_map_bs
        if block_size != gdim or len(velocity_to_mixed) % block_size:
            raise RuntimeError(
                "Unexpected vector P2 layout in the FEM preconditioner."
            )
        scalar_stiffness = velocity_block[0::block_size, 0::block_size].tocsc()
        scalar_ilu = spilu(
            scalar_stiffness,
            drop_tol=ilu_drop_tolerance,
            fill_factor=ilu_fill_factor,
            permc_spec="COLAMD",
        )
        pressure_trial = ufl.TrialFunction(Q)
        pressure_test = ufl.TestFunction(Q)
        pressure_mass = to_scipy_matrix(
            fem.assemble_matrix(
                fem.form(pressure_trial * pressure_test * ufl.dx(domain=msh))
            )
        )
        pressure_factor = splu(pressure_mass.tocsc())

        def apply_preconditioner(vector: np.ndarray) -> np.ndarray:
            result = np.zeros_like(vector)
            velocity_rhs = vector[velocity_to_mixed].reshape(-1, block_size)
            velocity_result = np.column_stack(
                [scalar_ilu.solve(velocity_rhs[:, component])
                 for component in range(gdim)]
            )
            result[velocity_to_mixed] = velocity_result.reshape(-1)
            result[pressure_to_mixed] = pressure_factor.solve(
                vector[pressure_to_mixed]
            )
            return result

        preconditioner = LinearOperator(
            A.shape, matvec=apply_preconditioner, dtype=np.float64
        )

        def record_residual(value: float) -> None:
            residual_history.append(float(value))

        solution, info = gmres(
            A,
            rhs,
            M=preconditioner,
            rtol=iterative_rtol,
            atol=0.0,
            restart=iterative_restart,
            maxiter=iterative_maxiter,
            callback=record_residual,
            callback_type="pr_norm",
        )
        if info != 0:
            raise RuntimeError(
                f"Preconditioned GMRES did not converge (info={info}, "
                f"iterations={len(residual_history)})."
            )
        solver_method = (
            "SciPy GMRES with scalar-P2 ILU and exact P1-mass block preconditioner"
        )
        iterations = len(residual_history)
        final_preconditioned_residual = (
            residual_history[-1] if residual_history else 0.0
        )

    if not np.all(np.isfinite(solution)):
        raise RuntimeError(
            "The monolithic reference solve returned non-finite values."
        )
    relative_linear_residual = float(
        np.linalg.norm(A @ solution - rhs) / max(np.linalg.norm(rhs), 1.0e-30)
    )
    if relative_linear_residual > max(1.0e-8, 20.0 * iterative_rtol):
        raise RuntimeError(
            f"The monolithic FEM relative residual is {relative_linear_residual:.3e}."
        )

    wh = fem.Function(W)
    wh.x.array[:] = solution
    uh = wh.sub(0).collapse()
    inlet_flux = float(
        fem.assemble_scalar(fem.form(ufl.dot(uh, n) * ds(2)))
    )
    outlet_flux = float(
        fem.assemble_scalar(fem.form(ufl.dot(uh, n) * ds(3)))
    )
    net_flux = inlet_flux + outlet_flux
    mass_scale = max(abs(inlet_flux), abs(outlet_flux), 1.0e-30)
    energy_dissipation = float(
        viscosity
        * fem.assemble_scalar(
            fem.form(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx(domain=msh))
        )
    )
    boundary_power = float(
        -inlet_pressure * inlet_flux - outlet_pressure * outlet_flux
    )
    relative_energy_residual = float(
        abs(energy_dissipation - boundary_power)
        / max(abs(boundary_power), 1.0e-30)
    )
    return ReferenceSolution(
        W=W,
        solution=np.asarray(solution, dtype=float),
        boundary_fluxes={"inlet": inlet_flux, "outlet": outlet_flux},
        relative_mass_imbalance=float(abs(net_flux) / mass_scale),
        relative_linear_residual=relative_linear_residual,
        energy_dissipation=energy_dissipation,
        boundary_power=boundary_power,
        relative_energy_residual=relative_energy_residual,
        ndofs=int(A.shape[0]),
        matrix_nnz=int(A.nnz),
        solver_method=solver_method,
        iterations=iterations,
        final_preconditioned_residual=float(final_preconditioned_residual),
    )

"""2D DDPNM solver: local Stokes response + global Schur assembly.

This module now delegates to the unified ``ddpnm_core`` pipeline while
preserving the original public API (``interface_order``, ``interface_coefficients``
shape ``(n, modes_per_interface)``, ``mode_shape``, ``solve_reference``, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from ddpnm_core.constants import PORT_TAG_BASE, WALL_TAG
from ddpnm_core.fem_utils import (
    global_boundary_tags,
    mixed_stokes_space,
    to_numpy_vector,
    to_scipy_matrix,
)
from ddpnm_core.solver_types import PortInfo, ReferenceSolution

from .geometry import PartitionData


# ---------------------------------------------------------------------------
# Dataclasses — preserved exactly
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeInfo:
    port_index: int
    degree: int
    global_dof: int | None
    known_coefficient: float | None


@dataclass
class LocalResponse:
    pore_id: int
    submesh: dmesh.Mesh
    parent_cell_map: np.ndarray
    parent_vertex_map: np.ndarray
    ports: tuple[PortInfo, ...]
    modes: tuple[ModeInfo, ...]
    W: fem.FunctionSpace
    G: np.ndarray
    responses: np.ndarray
    ndofs: int
    symmetry_error: float
    kernel_error: float


@dataclass
class DdpnmSolution:
    interface_coefficients: np.ndarray  # shape (n_interfaces, modes_per_interface)
    schur_matrix: np.ndarray
    rhs: np.ndarray
    local_responses: list[LocalResponse]
    local_solutions: list[np.ndarray]
    interface_flux_moment_sums: np.ndarray
    interface_order: int
    boundary_fluxes: dict[str, float]
    min_schur_eigenvalue: float
    max_mass_residual: float

    @property
    def interface_pressures(self) -> np.ndarray:
        return self.interface_coefficients.ravel()

    @property
    def interface_flux_sums(self) -> np.ndarray:
        return self.interface_flux_moment_sums.ravel()


# ---------------------------------------------------------------------------
# Classic helpers (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def build_modes(ports: tuple[PortInfo, ...], interface_order: int) -> tuple[ModeInfo, ...]:
    modes: list[ModeInfo] = []
    modes_per_interface = interface_order + 1
    for port_index, port in enumerate(ports):
        if port.kind == "interface":
            for degree in range(modes_per_interface):
                modes.append(ModeInfo(
                    port_index=port_index,
                    degree=degree,
                    global_dof=modes_per_interface * int(port.global_interface) + degree,
                    known_coefficient=None,
                ))
        else:
            modes.append(ModeInfo(
                port_index=port_index,
                degree=0,
                global_dof=None,
                known_coefficient=float(port.pressure),
            ))
    return tuple(modes)


def mode_shape(partition, ports, mode, x):
    """Closed-form interface polynomial shape (preserved for external callers)."""
    if mode.degree == 0:
        return ufl.as_ufl(1.0)
    interface_id = int(ports[mode.port_index].global_interface)
    center = partition.interface_centers[interface_id]
    tangent = partition.interface_tangents[interface_id]
    half_length = float(partition.interface_half_lengths[interface_id])
    coordinate = (
        (x[0] - float(center[0])) * float(tangent[0])
        + (x[1] - float(center[1])) * float(tangent[1])
    ) / half_length
    if mode.degree == 1:
        return coordinate
    raise ValueError(f"Unsupported interface polynomial degree {mode.degree}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_ddpnm(
    partition: PartitionData,
    viscosity: float = 1.0,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
    pressure_stabilization: float = 1.0e-10,
    interface_order: int = 1,
) -> DdpnmSolution:
    """Solve DDPNM with polynomial normal-traction modes.

    Delegates to the unified ``ddpnm_core`` pipeline via
    ``PolynomialNormalBasis``.
    """
    if interface_order not in (0, 1):
        raise ValueError("This implementation supports interface order 0 or 1.")

    from ddpnm_core.library import build_response_library
    from ddpnm_core.assembler import InterfaceAssembler
    from ddpnm2d.basis_2d import PolynomialNormalBasis

    n_interfaces = len(partition.interface_pairs)
    modes_per_interface = interface_order + 1
    n_unknowns = modes_per_interface * n_interfaces

    basis = PolynomialNormalBasis(partition, interface_order)
    library = build_response_library(
        partition, basis, viscosity=viscosity,
        inlet_pressure=inlet_pressure, outlet_pressure=outlet_pressure,
        pressure_stabilization=pressure_stabilization,
    )
    assembler = InterfaceAssembler(library)
    levels = np.zeros(n_interfaces, dtype=np.int8)
    system = assembler.assemble(levels)

    keys = system.global_keys
    key_to_dof = {k: d for d, k in enumerate(keys)}

    # Convert entries → LocalResponse
    local_responses: list[LocalResponse] = []
    for entry in library.entries:
        ports = entry.operator.ports
        modes = build_modes(ports, interface_order)
        G = entry.primitive_G
        scale = max(float(np.linalg.norm(G)), 1.0e-30)
        constant_coeffs = np.asarray(
            [1.0 if mode.degree == 0 else 0.0 for mode in modes]
        )
        local_responses.append(LocalResponse(
            pore_id=entry.operator.pore_id,
            submesh=entry.operator.submesh,
            parent_cell_map=entry.operator.parent_cell_map,
            parent_vertex_map=entry.operator.parent_vertex_map,
            ports=ports,
            modes=modes,
            W=entry.operator.W,
            G=G,
            responses=entry.primitive_responses,
            ndofs=entry.operator.ndofs,
            symmetry_error=entry.symmetry_error,
            kernel_error=float(np.linalg.norm(G @ constant_coeffs) / scale),
        ))

    # Extract coefficients and flux moments
    coefficients_flat = np.zeros(n_unknowns) if n_unknowns else np.empty(0)
    for interface_id in range(n_interfaces):
        for degree in range(modes_per_interface):
            if interface_order == 0:
                key = (interface_id, "normal", "P0")
            else:
                key = (interface_id, "normal", f"P{degree}")
            if key in key_to_dof:
                dof = key_to_dof[key]
                gid = modes_per_interface * interface_id + degree
                coefficients_flat[gid] = system.coefficients[dof]

    coefficients = coefficients_flat.reshape(n_interfaces, modes_per_interface)

    flux_sums = np.zeros((n_interfaces, modes_per_interface), dtype=float)
    for interface_id in range(n_interfaces):
        for degree in range(modes_per_interface):
            if interface_order == 0:
                key = (interface_id, "normal", "P0")
            else:
                key = (interface_id, "normal", f"P{degree}")
            if key in key_to_dof:
                dof = key_to_dof[key]
                flux_sums[interface_id, degree] = system.moment_residuals[dof]

    return DdpnmSolution(
        interface_coefficients=coefficients,
        schur_matrix=system.schur_matrix,
        rhs=system.rhs,
        local_responses=local_responses,
        local_solutions=system.local_solutions,
        interface_flux_moment_sums=flux_sums,
        interface_order=interface_order,
        boundary_fluxes=system.boundary_fluxes,
        min_schur_eigenvalue=system.min_schur_eigenvalue,
        max_mass_residual=float(np.max(np.abs(flux_sums))) if n_unknowns else 0.0,
    )


def solve_reference(
    msh: dmesh.Mesh,
    viscosity: float = 1.0,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
    pressure_stabilization: float = 1.0e-10,
) -> ReferenceSolution:
    """Monolithic Taylor–Hood reference solve (unchanged from original)."""
    tags = global_boundary_tags(msh)
    W = mixed_stokes_space(msh)
    (u, p_var) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    dx = ufl.dx(domain=msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    n = ufl.FacetNormal(msh)
    a = (
        viscosity * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        - p_var * ufl.div(v) * dx
        - q * ufl.div(u) * dx
        - pressure_stabilization * p_var * q * dx
    )
    L = -inlet_pressure * ufl.dot(n, v) * ds(2) - outlet_pressure * ufl.dot(n, v) * ds(3)
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
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        solution = spsolve(A, rhs)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("Reference Stokes solve returned non-finite values.")
    solution = np.asarray(solution, dtype=float)
    wh = fem.Function(W)
    wh.x.array[:] = solution
    uh = wh.sub(0).collapse()
    inlet_flux = float(fem.assemble_scalar(fem.form(ufl.dot(uh, n) * ds(2))))
    outlet_flux = float(fem.assemble_scalar(fem.form(ufl.dot(uh, n) * ds(3))))
    boundary_fluxes = {"inlet": inlet_flux, "outlet": outlet_flux}
    relative_mass_imbalance = float(
        abs(inlet_flux + outlet_flux) / max(abs(outlet_flux), 1.0e-30)
    )
    residual = A @ solution - rhs
    relative_linear_residual = float(
        np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0e-30)
    )
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
        solution=solution,
        boundary_fluxes=boundary_fluxes,
        relative_mass_imbalance=relative_mass_imbalance,
        relative_linear_residual=relative_linear_residual,
        energy_dissipation=energy_dissipation,
        boundary_power=boundary_power,
        relative_energy_residual=relative_energy_residual,
        ndofs=len(solution),
        matrix_nnz=int(A.nnz),
        solver_method="SciPy SuperLU sparse direct",
        iterations=1,
        final_preconditioned_residual=0.0,
    )

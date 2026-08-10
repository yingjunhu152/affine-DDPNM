"""3D DDPNM solver: local Stokes response + global Schur assembly.

This module now delegates to the unified ``ddpnm_core`` pipeline while
preserving the original public API (dataclass field names, default
``pressure_stabilization=0.0``, ``kernel_error`` per response, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ddpnm_core.solver_types import PortInfo

from .geometry import PartitionData


# ---------------------------------------------------------------------------
# Dataclasses — preserved exactly for backward compatibility
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeInfo:
    port_index: int
    global_dof: int | None
    known_coefficient: float | None


@dataclass
class LocalResponse:
    pore_id: int
    submesh: object  # dolfinx.mesh.Mesh
    parent_cell_map: np.ndarray
    parent_vertex_map: np.ndarray
    ports: tuple[PortInfo, ...]
    modes: tuple[ModeInfo, ...]
    W: object  # fem.FunctionSpace
    G: np.ndarray
    responses: np.ndarray
    ndofs: int
    symmetry_error: float
    kernel_error: float


@dataclass
class DdpnmSolution:
    interface_pressures: np.ndarray
    schur_matrix: np.ndarray
    rhs: np.ndarray
    local_responses: list[LocalResponse]
    local_solutions: list[np.ndarray]
    interface_flux_sums: np.ndarray
    boundary_fluxes: dict[str, float]
    min_schur_eigenvalue: float
    max_mass_residual: float


# ---------------------------------------------------------------------------
# Public API — delegates to ddpnm_core
# ---------------------------------------------------------------------------

def build_modes(ports: tuple[PortInfo, ...]) -> tuple[ModeInfo, ...]:
    """Build the classic P0-DDPNM mode list (one constant normal traction per port)."""
    modes: list[ModeInfo] = []
    for port_index, port in enumerate(ports):
        if port.kind == "interface":
            modes.append(ModeInfo(
                port_index=port_index,
                global_dof=int(port.global_interface),
                known_coefficient=None,
            ))
        else:
            modes.append(ModeInfo(
                port_index=port_index,
                global_dof=None,
                known_coefficient=float(port.pressure),
            ))
    return tuple(modes)


def solve_ddpnm(
    partition: PartitionData,
    viscosity: float = 1.0,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
    pressure_stabilization: float = 0.0,
) -> DdpnmSolution:
    """Solve original DDPNM: one constant normal traction per interface.

    Delegates to the unified ``ddpnm_core`` pipeline via the
    ``ClassicP0Basis``.
    """
    from ddpnm_core.library import build_response_library
    from ddpnm_core.assembler import InterfaceAssembler
    from ddpnm3d.basis_3d import ClassicP0Basis

    n_interfaces = len(partition.interface_pairs)
    basis = ClassicP0Basis()
    library = build_response_library(
        partition, basis, viscosity=viscosity,
        inlet_pressure=inlet_pressure, outlet_pressure=outlet_pressure,
        pressure_stabilization=pressure_stabilization,
    )
    assembler = InterfaceAssembler(library)
    levels = np.zeros(n_interfaces, dtype=np.int8)
    system = assembler.assemble(levels)

    # Build key → dof map for extraction
    keys = system.global_keys
    key_to_dof = {k: d for d, k in enumerate(keys)}

    # Convert LocalLibraryEntry → LocalResponse
    local_responses: list[LocalResponse] = []
    for entry in library.entries:
        ports = entry.operator.ports
        modes = build_modes(ports)
        G = entry.primitive_G
        scale = max(float(np.linalg.norm(G)), 1.0e-30)
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
            kernel_error=float(np.linalg.norm(G @ np.ones(G.shape[0])) / scale),
        ))

    # Extract interface pressures and flux sums
    interface_pressures = np.asarray([
        system.coefficients[key_to_dof[(i, "normal", "P0")]]
        for i in range(n_interfaces)
    ])
    interface_flux_sums = np.asarray([
        system.moment_residuals[key_to_dof[(i, "normal", "P0")]]
        for i in range(n_interfaces)
    ])

    return DdpnmSolution(
        interface_pressures=interface_pressures,
        schur_matrix=system.schur_matrix,
        rhs=system.rhs,
        local_responses=local_responses,
        local_solutions=system.local_solutions,
        interface_flux_sums=interface_flux_sums,
        boundary_fluxes=system.boundary_fluxes,
        min_schur_eigenvalue=system.min_schur_eigenvalue,
        max_mass_residual=float(np.max(np.abs(system.moment_residuals)))
        if len(system.moment_residuals) else 0.0,
    )

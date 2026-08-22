"""Response library: factorize once, solve for many subspace choices.

A :class:`ResponseLibrary` stores, for every pore, the factorized local Stokes
operator and the full primitive response data (load matrix **B**, response
matrix **R**, flux matrix **G** = BᵀR).  Once built, any subspace (level
vector) can be queried via :class:`~ddpnm_core.assembler.InterfaceAssembler`
without re-factorizing.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np

from ddpnm_core.basis import InterfaceBasis, PrimitiveMode
from ddpnm_core.stokes_operator import (
    LocalStokesOperator,
    build_local_stokes_operator,
)


@dataclass
class LocalLibraryEntry:
    """Per-pore primitive response data."""

    operator: LocalStokesOperator
    primitive_modes: tuple[PrimitiveMode, ...]
    primitive_loads: np.ndarray  # B, shape (ndofs, n_primitive)
    primitive_responses: np.ndarray  # R = A⁻¹B, shape (ndofs, n_primitive)
    primitive_G: np.ndarray  # BᵀR, hierarchy sign convention
    symmetry_error: float


@dataclass
class ResponseLibrary:
    """Complete primitive response library for one partition + basis."""

    partition: object  # PartitionData
    basis: InterfaceBasis
    entries: list[LocalLibraryEntry]
    interface_nodes: tuple[tuple[int, ...], ...]  # parent vertices per interface
    viscosity: float
    pressure_stabilization: float
    inlet_pressure: float
    outlet_pressure: float
    velocity_degree: int = 2
    pressure_gradient_stabilization: float = 0.0
    viscous_form: str = "gradient"


def build_response_library(
    partition,
    basis: InterfaceBasis,
    viscosity: float = 1.0,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
    pressure_stabilization: float = 0.0,
    retain_responses: bool = True,
    velocity_degree: int = 2,
    pressure_gradient_stabilization: float = 0.0,
    viscous_form: str = "gradient",
) -> ResponseLibrary:
    """Build the primitive response library for *partition* and *basis*.

    Each local Stokes matrix is factorized exactly once.  The returned
    library can be queried for arbitrarily many subspace choices.  Set
    ``retain_responses=False`` when only condensed Schur data are required:
    this releases each large local right-hand-side/response block immediately
    after forming its exact primitive compliance matrix ``G``.
    """
    pore_ids = sorted(int(label) for label in np.unique(partition.cell_labels))
    entries: list[LocalLibraryEntry] = []

    for pore_id in pore_ids:
        operator = build_local_stokes_operator(
            partition, pore_id, viscosity, pressure_stabilization,
            inlet_pressure, outlet_pressure,
            velocity_degree=velocity_degree,
            pressure_gradient_stabilization=pressure_gradient_stabilization,
            viscous_form=viscous_form,
        )
        specs = basis.primitive_specs(partition, operator)
        modes = tuple(spec.mode for spec in specs)

        # Assemble all primitive load columns
        columns = [spec.load(operator) for spec in specs]
        B = np.column_stack(columns)

        # Block solve and flux matrix
        R = operator.solve(B)
        G = B.T @ R  # hierarchy sign convention: G = Bᵀ A⁻¹ B

        scale = max(float(np.linalg.norm(G)), 1.0e-30)
        entries.append(LocalLibraryEntry(
            operator=operator,
            primitive_modes=modes,
            primitive_loads=B if retain_responses else np.empty((operator.ndofs, 0)),
            primitive_responses=R if retain_responses else np.empty((operator.ndofs, 0)),
            primitive_G=G,
            symmetry_error=float(np.linalg.norm(G - G.T) / scale),
        ))
        # Release per-pore temporaries to avoid C-heap exhaustion on large meshes
        del B, R, columns, specs, modes
        if pore_id % 10 == 0:
            gc.collect()

    # Interface nodes — basis may provide them, otherwise compute generically
    if hasattr(basis, "interface_nodes"):
        interface_nodes = basis.interface_nodes
    else:
        from ddpnm_core.basis import _interface_nodes
        interface_nodes = _interface_nodes(partition)

    return ResponseLibrary(
        partition=partition,
        basis=basis,
        entries=entries,
        interface_nodes=interface_nodes,
        viscosity=viscosity,
        pressure_stabilization=pressure_stabilization,
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure,
        velocity_degree=velocity_degree,
        pressure_gradient_stabilization=pressure_gradient_stabilization,
        viscous_form=viscous_form,
    )

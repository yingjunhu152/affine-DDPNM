"""Shared dataclass types for DD-PNM solvers.

These are the common types used by both 2D and 3D solvers.  Dimension-specific
extensions (e.g.  ``ModeInfo`` with a ``degree`` field in 2D) live in the
respective dimension packages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dolfinx import fem, mesh as dmesh


@dataclass(frozen=True)
class PortInfo:
    kind: str
    global_interface: int | None
    pressure: float | None
    parent_facets: tuple[int, ...]


@dataclass
class ReferenceSolution:
    W: fem.FunctionSpace
    solution: np.ndarray
    boundary_fluxes: dict[str, float]
    relative_mass_imbalance: float
    relative_linear_residual: float
    energy_dissipation: float
    boundary_power: float
    relative_energy_residual: float
    ndofs: int
    matrix_nnz: int
    solver_method: str
    iterations: int
    final_preconditioned_residual: float
    converged: bool = True

"""Interface basis protocol and primitive-mode types.

An :class:`InterfaceBasis` encodes everything dimension- and enrichment-
specific: how many modes per level, what scalar shapes and vector directions
constitute each primitive load, and how primitives combine into active global
unknowns.  P0-DDPNM, P0-vector-DDPNM and P1-vector-HODDPNM are just three
concrete bases with different :meth:`InterfaceBasis.active_indices` answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrimitiveMode:
    """One primitive traction mode on a local pore port.

    A primitive is the finest-granularity load the response library stores.
    Active (global) unknowns are combinations of primitives — either a simple
    selection (3D affine basis) or a linear combination (2D nodal-hat sum for
    the P0 constant mode).
    """

    port_index: int
    component: str  # "normal" | "tangent" | "tangent_1" | "tangent_2"
    polynomial: str  # "P0" | "P1" | "P1_s" | "P1_t" | "nodal"
    interface_id: int | None
    node_index: int | None  # only for nodal primitives
    known_coefficient: float | None  # prescribed for inlet / outlet ports


@dataclass(frozen=True)
class PrimitiveSpec:
    """A primitive mode paired with its assembled load vector factory."""

    mode: PrimitiveMode
    load: Callable[["LocalStokesOperator"], np.ndarray]
    # callable receives the factorized operator, returns the rhs column


# ---------------------------------------------------------------------------
# Basis protocol
# ---------------------------------------------------------------------------

class InterfaceBasis(Protocol):
    """Protocol for an interface traction basis.

    Concrete bases live in the dimension packages (``basis_2d.py``,
    ``basis_3d.py``).  The core never imports from those packages; it only
    calls the methods below.
    """

    name: str
    component_names: tuple[str, ...]
    level_dofs: tuple[int, int, int]  # unknowns per interface at levels 0 / 1 / 2

    def primitive_specs(
        self, partition, operator: "LocalStokesOperator"
    ) -> tuple[PrimitiveSpec, ...]:
        """Build all primitive load specs for *operator*.

        Called once per pore during :func:`build_response_library`.
        """
        ...

    def active_indices(
        self,
        primitive_modes: tuple[PrimitiveMode, ...],
        port_index: int,
        level: int,
    ) -> tuple[int, ...]:
        """Return the indices into *primitive_modes* that are active on
        *port_index* at the given hierarchy *level*.

        The returned order defines the local ordering of the active unknowns
        for that port — they must be consistent with :meth:`global_keys`.
        """
        ...

    def active_transform(
        self,
        primitive_modes: tuple[PrimitiveMode, ...],
        port_index: int,
        level: int,
    ) -> np.ndarray | None:
        """Optional: a ``(n_primitive, n_active)`` combination matrix.

        Return ``None`` (the default) when each active mode is a direct
        selection of one primitive (the 3D affine case).  Return a matrix
        when active modes are linear combinations of primitives (the 2D
        nodal-hat sum for the P0 constant).
        """
        return None

    def global_keys(
        self, level: int, interface_id: int
    ) -> tuple[tuple, ...]:
        """Ordered global-key tuples for *interface_id* at *level*.

        Each key is a hashable tuple that serves as the dictionary key in
        the global Schur assembly.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers shared by concrete bases
# ---------------------------------------------------------------------------

def _interface_nodes(partition) -> tuple[tuple[int, ...], ...]:
    """Return per-interface tuple of parent-vertex indices.

    Concrete bases can call this as part of their setup.  2D sorts vertices
    along the interface tangent; 3D keeps raw facet→vertex order (ordering is
    irrelevant for affine modes).
    """
    msh = partition.mesh
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    groups: list[tuple[int, ...]] = []
    for interface_id in range(len(partition.interface_pairs)):
        facets = np.flatnonzero(partition.facet_interface_ids == interface_id)
        vertices = np.unique(
            np.concatenate([f2v.links(int(facet)) for facet in facets])
        )
        groups.append(tuple(int(vertex) for vertex in vertices))
    return tuple(groups)

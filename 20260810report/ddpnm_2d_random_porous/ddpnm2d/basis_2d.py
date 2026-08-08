"""2-D interface traction bases.

Provides two concrete :class:`~ddpnm_core.basis.InterfaceBasis` implementations:

* :class:`PolynomialNormalBasis` — ``interface_order`` ∈ {0, 1}, using
  closed-form UFL polynomial shapes (the classic 2-D solver).
* :class:`HierarchyBasis` — P1 nodal-hat primitives with level-dependent
  summation transforms (the 2-D adaptive hierarchy).

Level semantics (HierarchyBasis)
--------------------------------
===========  =====  ============================================
Level        dofs   active modes
===========  =====  ============================================
0 (DDPNM)    1      normal_constant (sum of all nodal hat normals)
1 (DDPNMT)   2      normal_constant + tangent_constant
2 (HODDPNM)  2N     per-node normal + tangent (N = interface nodes)
===========  =====  ============================================
"""

from __future__ import annotations

import numpy as np
import ufl
from basix.ufl import element
from dolfinx import fem

from ddpnm_core.basis import (
    InterfaceBasis,
    PrimitiveMode,
    PrimitiveSpec,
    _interface_nodes,
)
from ddpnm_core.constants import PORT_TAG_BASE, WALL_TAG


# ---------------------------------------------------------------------------
# Sorted interface nodes (2-D: order along tangent)
# ---------------------------------------------------------------------------

def _sorted_interface_nodes(partition) -> tuple[tuple[int, ...], ...]:
    """Interface vertices ordered along the interface tangent."""
    msh = partition.mesh
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    result: list[tuple[int, ...]] = []
    for interface_id in range(len(partition.interface_pairs)):
        facets = np.flatnonzero(partition.facet_interface_ids == interface_id)
        vertices = np.unique(np.concatenate([f2v.links(int(f)) for f in facets]))
        xy = msh.geometry.x[vertices, :2]
        tangent = partition.interface_tangents[interface_id]
        order = np.argsort(xy @ tangent)
        result.append(tuple(int(v) for v in vertices[order]))
    return tuple(result)


# ---------------------------------------------------------------------------
# PolynomialNormalBasis — classic 2-D solver (interface_order 0 or 1)
# ---------------------------------------------------------------------------

class PolynomialNormalBasis:
    """Polynomial normal-traction basis for the classic 2-D DDPNM solver.

    ``interface_order=0`` → one constant normal traction per interface.
    ``interface_order=1`` → constant + linear (P1) normal traction.

    The shape functions are closed-form UFL expressions, identical to the
    original ``mode_shape``.
    """

    name = "2D-polynomial-normal"
    component_names = ("normal",)

    def __init__(self, partition, interface_order: int = 1):
        if interface_order not in (0, 1):
            raise ValueError("interface_order must be 0 or 1.")
        self._order = interface_order
        self._modes_per_interface = interface_order + 1
        self._partition = partition
        self.interface_nodes = _sorted_interface_nodes(partition)
        self.level_dofs = (
            self._modes_per_interface,
            self._modes_per_interface,
            self._modes_per_interface,
        )

    @property
    def interface_order(self) -> int:
        return self._order

    def primitive_specs(self, partition, operator):
        specs: list[PrimitiveSpec] = []
        x = ufl.SpatialCoordinate(operator.submesh)
        n = operator.normal
        (_, q_var) = ufl.TestFunctions(operator.W)
        (v_expr, _) = ufl.split(ufl.TestFunction(operator.W))

        for port_index, port in enumerate(operator.ports):
            tag = PORT_TAG_BASE + port_index
            if port.kind != "interface":
                # Prescribed boundary — constant normal
                mode = PrimitiveMode(
                    port_index=port_index, component="normal", polynomial="P0",
                    interface_id=None, node_index=None,
                    known_coefficient=float(port.pressure),
                )
                L = -ufl.dot(n, v_expr) * ufl.ds(tag, domain=operator.submesh,
                                                  subdomain_data=operator.facet_tags)
                specs.append(PrimitiveSpec(mode=mode, load=lambda op, e=L: op.assemble_load(e)))
                continue

            interface_id = int(port.global_interface)
            center = partition.interface_centers[interface_id]
            tangent = partition.interface_tangents[interface_id]
            half = float(partition.interface_half_lengths[interface_id])
            coordinate = (
                (x[0] - float(center[0])) * float(tangent[0])
                + (x[1] - float(center[1])) * float(tangent[1])
            ) / half

            shapes = [ufl.as_ufl(1.0)]
            if self._order >= 1:
                shapes.append(coordinate)

            for degree, shape in enumerate(shapes):
                global_dof = self._modes_per_interface * interface_id + degree
                mode = PrimitiveMode(
                    port_index=port_index, component="normal",
                    polynomial=f"P{degree}", interface_id=interface_id,
                    node_index=None, known_coefficient=None,
                )
                L = -shape * ufl.dot(n, v_expr) * ufl.ds(tag, domain=operator.submesh,
                                                          subdomain_data=operator.facet_tags)
                specs.append(PrimitiveSpec(mode=mode, load=lambda op, e=L: op.assemble_load(e)))

        return tuple(specs)

    def active_indices(self, primitive_modes, port_index, level):
        return tuple(
            i for i, m in enumerate(primitive_modes) if m.port_index == port_index
        )

    def global_keys(self, level, interface_id):
        if self._order == 0:
            return ((interface_id, "normal", "P0"),)
        return ((interface_id, "normal", "P0"), (interface_id, "normal", "P1"))

    def active_transform(self, primitive_modes, port_index, level):
        return None


# ---------------------------------------------------------------------------
# HierarchyBasis — 2-D nodal P1 hats with summation transforms
# ---------------------------------------------------------------------------

class HierarchyBasis:
    """P1 nodal-hat hierarchy for 2-D interfaces.

    Primitives are per-node P1 Lagrange hat functions in the normal and
    tangent components.  For levels 0 and 1 the constant modes are formed
    by summing the hats (partition of unity); for level 2 each hat is used
    individually.
    """

    name = "2D-nodal-hierarchy"
    component_names = ("normal", "tangent")
    level_dofs = (1, 2, -1)  # level 2 depends on n_nodes

    def __init__(self, partition):
        self._partition = partition
        self.interface_nodes = _sorted_interface_nodes(partition)

    def primitive_specs(self, partition, operator):
        specs: list[PrimitiveSpec] = []
        submesh = operator.submesh
        n = operator.normal
        tangent = ufl.as_vector((-n[1], n[0]))
        (_, q_var) = ufl.TestFunctions(operator.W)
        (v_expr, _) = ufl.split(ufl.TestFunction(operator.W))

        # Build P1 Lagrange space for hat functions
        submesh.topology.create_connectivity(0, submesh.topology.dim)
        Q = fem.functionspace(submesh, element("Lagrange", submesh.basix_cell(), 1))
        parent_to_local = {int(p): l for l, p in enumerate(operator.parent_vertex_map)}

        for port_index, port in enumerate(operator.ports):
            tag = PORT_TAG_BASE + port_index
            if port.kind != "interface":
                mode = PrimitiveMode(
                    port_index=port_index, component="normal", polynomial="nodal",
                    interface_id=None, node_index=None,
                    known_coefficient=float(port.pressure),
                )
                L = -ufl.dot(n, v_expr) * ufl.ds(tag, domain=submesh,
                                                  subdomain_data=operator.facet_tags)
                specs.append(PrimitiveSpec(mode=mode, load=lambda op, e=L: op.assemble_load(e)))
                continue

            interface_id = int(port.global_interface)
            nodes = self.interface_nodes[interface_id]

            for component in ("normal", "tangent"):
                direction = -n if component == "normal" else tangent
                for node_position, parent_vertex in enumerate(nodes):
                    local_vertex = parent_to_local.get(int(parent_vertex))
                    if local_vertex is None:
                        raise RuntimeError(
                            f"Interface {interface_id} node {parent_vertex} "
                            f"absent from pore submesh."
                        )
                    mode = PrimitiveMode(
                        port_index=port_index, component=component,
                        polynomial="nodal", interface_id=interface_id,
                        node_index=node_position, known_coefficient=None,
                    )

                    # Build the hat function
                    phi = fem.Function(Q)
                    dofs = fem.locate_dofs_topological(
                        Q, 0, np.asarray([local_vertex], dtype=np.int32)
                    )
                    phi.x.array[dofs] = 1.0

                    L = phi * ufl.dot(direction, v_expr) * ufl.ds(
                        tag, domain=submesh, subdomain_data=operator.facet_tags
                    )
                    specs.append(PrimitiveSpec(mode=mode, load=lambda op, e=L: op.assemble_load(e)))

        return tuple(specs)

    def active_indices(self, primitive_modes, port_index, level):
        """Return indices of active primitives for a port at the given level."""
        port_primitives = [
            (i, m) for i, m in enumerate(primitive_modes) if m.port_index == port_index
        ]
        if not port_primitives:
            return ()

        # Check if this is a boundary port
        if port_primitives[0][1].interface_id is None:
            return (port_primitives[0][0],)

        if level <= 1:
            # All primitives on this port are active (they get combined via transform)
            return tuple(i for i, _ in port_primitives)
        else:
            # Level 2: each primitive is individually active
            return tuple(i for i, _ in port_primitives)

    def active_transform(self, primitive_modes, port_index, level):
        """Build summation transform for constant modes at levels 0/1."""
        port_primitives = [
            (i, m) for i, m in enumerate(primitive_modes) if m.port_index == port_index
        ]
        if not port_primitives:
            return None

        if port_primitives[0][1].interface_id is None:
            return None  # boundary — no transform needed

        normal_indices = [
            i for i, m in port_primitives if m.component == "normal"
        ]
        tangent_indices = [
            i for i, m in port_primitives if m.component == "tangent"
        ]

        n_prim = len(port_primitives)
        if level == 0:
            # One active mode: sum of all normal hats
            col = np.zeros(n_prim)
            for idx in normal_indices:
                local_idx = next(j for j, (pi, _) in enumerate(port_primitives) if pi == idx)
                col[local_idx] = 1.0
            return col.reshape(-1, 1)
        elif level == 1:
            # Two active modes: sum of normal hats, sum of tangent hats
            transform = np.zeros((n_prim, 2))
            for idx in normal_indices:
                local_idx = next(j for j, (pi, _) in enumerate(port_primitives) if pi == idx)
                transform[local_idx, 0] = 1.0
            for idx in tangent_indices:
                local_idx = next(j for j, (pi, _) in enumerate(port_primitives) if pi == idx)
                transform[local_idx, 1] = 1.0
            return transform
        else:
            return None  # Level 2: identity

    def global_keys(self, level, interface_id):
        if level == 0:
            return ((interface_id, "normal_constant"),)
        if level == 1:
            return ((interface_id, "normal_constant"), (interface_id, "tangent_constant"))
        if level >= 2:
            nodes = self.interface_nodes[interface_id]
            keys: list[tuple] = []
            for node_position in range(len(nodes)):
                keys.append((interface_id, "normal_node", node_position))
                keys.append((interface_id, "tangent_node", node_position))
            return tuple(keys)
        raise ValueError(f"Invalid hierarchy level {level}.")

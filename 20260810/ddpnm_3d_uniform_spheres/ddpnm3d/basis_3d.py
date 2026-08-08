"""3-D affine interface traction bases.

Provides the nine-mode facewise-affine traction space (``HierarchyBasis``)
and the original one-mode constant-normal-traction subspace
(``ClassicP0Basis``).  Both are concrete :class:`~ddpnm_core.basis.InterfaceBasis`
implementations.

Level semantics
---------------
===========  =====  ===========================================
Level        dofs   active modes
===========  =====  ===========================================
0 (DDPNM)    1      (normal, P0)
1 (DDPNMT)   3      all P0 components {n, t₁, t₂}
2 (HODDPNM)  9      full affine space {P0, P1ₛ, P1ₜ} × {n, t₁, t₂}
===========  =====  ===========================================
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem

from ddpnm_core.basis import (
    InterfaceBasis,
    PrimitiveMode,
    PrimitiveSpec,
    _interface_nodes,
)
from ddpnm_core.constants import PORT_TAG_BASE


COMPONENTS = ("normal", "tangent_1", "tangent_2")
POLYNOMIALS = ("P0", "P1_s", "P1_t")

AFFINE_MODE_GROUPS = {
    "constant_vector": (
        ("normal", "P0"),
        ("tangent_1", "P0"),
        ("tangent_2", "P0"),
    ),
    "linear_normal": (
        ("normal", "P1_s"),
        ("normal", "P1_t"),
    ),
    "linear_tangential": (
        ("tangent_1", "P1_s"),
        ("tangent_1", "P1_t"),
        ("tangent_2", "P1_s"),
        ("tangent_2", "P1_t"),
    ),
}


def affine_interface_mode_keys(interface_id: int) -> tuple[tuple, ...]:
    """The nine ordered keys of the full facewise P1 traction space."""
    return tuple(
        (int(interface_id), component, polynomial)
        for group in AFFINE_MODE_GROUPS.values()
        for component, polynomial in group
    )


# ---------------------------------------------------------------------------
# Tangent-frame helpers
# ---------------------------------------------------------------------------

def _tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=float)
    reference = (
        np.asarray([1.0, 0.0, 0.0])
        if abs(float(normal[0])) < 0.85
        else np.asarray([0.0, 1.0, 0.0])
    )
    tangent_1 = np.cross(normal, reference)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(normal, tangent_1)
    tangent_2 /= np.linalg.norm(tangent_2)
    return tangent_1, tangent_2


def _interface_frames_and_scales(
    partition, interface_nodes: tuple[tuple[int, ...], ...]
) -> tuple[np.ndarray, np.ndarray]:
    tangents = np.empty((len(interface_nodes), 2, 3), dtype=float)
    scales = np.empty((len(interface_nodes), 2), dtype=float)
    coordinates = partition.mesh.geometry.x[:, :3]
    for interface_id, vertices in enumerate(interface_nodes):
        tangent_1, tangent_2 = _tangent_frame(
            partition.interface_normals[interface_id]
        )
        tangents[interface_id, 0] = tangent_1
        tangents[interface_id, 1] = tangent_2
        relative = coordinates[np.asarray(vertices)] - partition.interface_centers[
            interface_id
        ]
        scales[interface_id, 0] = max(
            float(np.max(np.abs(relative @ tangent_1))), 1.0e-12
        )
        scales[interface_id, 1] = max(
            float(np.max(np.abs(relative @ tangent_2))), 1.0e-12
        )
    return tangents, scales


# ---------------------------------------------------------------------------
# HierarchyBasis — the full 9-mode affine space
# ---------------------------------------------------------------------------

class HierarchyBasis:
    """Facewise-affine (P1) traction basis for 3-D interfaces.

    Each interface carries 9 modes: :math:`\\{1, s, t\\}` scalar shapes
    times :math:`\\{n, t_1, t_2\\}` vector directions.  The constant-vector
    subspace (3 modes) is level 1; the single normal-constant mode is level 0.

    The scalar shape is represented as ``c₀ + c₁·x + c₂·y + c₃·z`` and the
    direction as ``d + dn·n``, both baked into run-time ``fem.Constant``
    coefficients so the same FFCx load kernel is reused across every
    tetrahedral submesh.
    """

    name = "3D-affine-hierarchy"
    component_names = COMPONENTS
    level_dofs = (1, 3, 9)

    def __init__(self, partition):
        interface_nodes = _interface_nodes(partition)
        tangents, scales = _interface_frames_and_scales(partition, interface_nodes)
        self.interface_nodes = interface_nodes
        self.interface_tangents = tangents
        self.interface_scales = scales

    # --- primitive_specs ---------------------------------------------------

    def primitive_specs(self, partition, operator):
        """Return the runtime-Constant coefficient specs for all primitives.

        One compiled load form per port — the FFCx kernel is reused across
        all primitive modes on the same port by mutating fem.Constant values,
        exactly as in the original _primitive_mode_specs optimization.
        """
        specs: list[PrimitiveSpec] = []
        pore_id = operator.pore_id
        ports = operator.ports
        submesh = operator.submesh

        # Pre-compute per-interface geometry lookups
        interface_data: dict[int, dict] = {}
        for port_index, port in enumerate(ports):
            if port.kind != "interface":
                continue
            iid = int(port.global_interface)
            if iid in interface_data:
                continue
            pair = partition.interface_pairs[iid]
            side_sign = 1.0 if pore_id == pair[0] else -1.0
            if pore_id not in pair:
                raise RuntimeError(f"Pore {pore_id} not incident on interface {iid}.")
            t1 = self.interface_tangents[iid, 0]
            t2 = self.interface_tangents[iid, 1]
            center = partition.interface_centers[iid]
            s1, s2 = self.interface_scales[iid]
            interface_data[iid] = {
                "side_sign": side_sign,
                "scalar": {
                    "P0": np.asarray([1.0, 0.0, 0.0, 0.0]),
                    "P1_s": np.concatenate(([-float(center @ t1) / s1], t1 / s1)),
                    "P1_t": np.concatenate(([-float(center @ t2) / s2], t2 / s2)),
                },
                "dirs": {
                    "normal": (np.zeros(3), -1.0),
                    "tangent_1": (side_sign * t1, 0.0),
                    "tangent_2": (side_sign * t2, 0.0),
                },
            }

        # Build one compiled form per port with mutable Constants
        for port_index, port in enumerate(ports):
            tag = PORT_TAG_BASE + port_index
            sc = fem.Constant(submesh, np.zeros(4))
            dc = fem.Constant(submesh, np.zeros(3))
            nc = fem.Constant(submesh, 0.0)
            x = ufl.SpatialCoordinate(submesh)
            ln = ufl.FacetNormal(submesh)
            scalar_shape = sc[0] + sc[1] * x[0] + sc[2] * x[1] + sc[3] * x[2]
            direction_expr = dc + nc * ln
            (v_expr, _) = ufl.split(ufl.TestFunction(operator.W))
            L_expr = scalar_shape * ufl.dot(direction_expr, v_expr) * ufl.ds(
                tag, domain=submesh, subdomain_data=operator.facet_tags
            )
            load_form = fem.form(L_expr)

            if port.kind != "interface":
                mode = PrimitiveMode(
                    port_index=port_index, component="normal", polynomial="P0",
                    interface_id=None, node_index=None,
                    known_coefficient=float(port.pressure),
                )
                sc.value[:] = [1.0, 0.0, 0.0, 0.0]
                dc.value[:] = np.zeros(3)
                nc.value = -1.0
                specs.append(PrimitiveSpec(mode=mode,
                    load=_constant_load(operator, sc, dc, nc, load_form)))
                continue

            iid = int(port.global_interface)
            data = interface_data[iid]
            for component in COMPONENTS:
                direction_vec, normal_coef = data["dirs"][component]
                for polynomial in POLYNOMIALS:
                    mode = PrimitiveMode(
                        port_index=port_index, component=component,
                        polynomial=polynomial, interface_id=iid,
                        node_index=None, known_coefficient=None,
                    )
                    c = data["scalar"][polynomial].copy()
                    d = direction_vec.copy()
                    nf = normal_coef
                    specs.append(PrimitiveSpec(mode=mode,
                        load=_constant_load(operator, sc, dc, nc, load_form,
                                            c, d, nf)))

        return tuple(specs)

    # --- active_indices ----------------------------------------------------

    def active_indices(self, primitive_modes, port_index, level):
        indices: list[int] = []
        for idx, mode in enumerate(primitive_modes):
            if mode.port_index != port_index:
                continue
            if mode.interface_id is None:
                indices.append(idx)
                continue
            if level == 0:
                if mode.component == "normal" and mode.polynomial == "P0":
                    indices.append(idx)
            elif level == 1:
                if mode.polynomial == "P0":
                    indices.append(idx)
            else:
                indices.append(idx)
        return tuple(indices)

    # --- global_keys --------------------------------------------------------

    def global_keys(self, level, interface_id):
        if level == 0:
            return ((interface_id, "normal", "P0"),)
        if level == 1:
            return tuple(
                (interface_id, component, "P0") for component in COMPONENTS
            )
        if level >= 2:
            return affine_interface_mode_keys(interface_id)
        raise ValueError(f"Invalid hierarchy level {level}.")


# ---------------------------------------------------------------------------
# ClassicP0Basis — original DDPNM (one constant normal traction)
# ---------------------------------------------------------------------------

class ClassicP0Basis:
    """Classic P0-DDPNM basis: one constant normal traction per interface.

    This is mathematically identical to :class:`HierarchyBasis` at uniform
    level 0 but is provided as a stand-alone basis for the original solver
    interface where ``interface_order`` is not a concept.
    """

    name = "3D-classic-P0"
    component_names = ("normal",)
    level_dofs = (1, 1, 1)

    def primitive_specs(self, partition, operator):
        specs: list[PrimitiveSpec] = []
        submesh = operator.submesh
        for port_index, port in enumerate(operator.ports):
            tag = PORT_TAG_BASE + port_index
            sc = fem.Constant(submesh, np.zeros(4))
            dc = fem.Constant(submesh, np.zeros(3))
            nc = fem.Constant(submesh, 0.0)
            x = ufl.SpatialCoordinate(submesh)
            ln = ufl.FacetNormal(submesh)
            scalar_shape = sc[0] + sc[1] * x[0] + sc[2] * x[1] + sc[3] * x[2]
            direction_expr = dc + nc * ln
            (v_expr, _) = ufl.split(ufl.TestFunction(operator.W))
            L_expr = scalar_shape * ufl.dot(direction_expr, v_expr) * ufl.ds(
                tag, domain=submesh, subdomain_data=operator.facet_tags
            )
            load_form = fem.form(L_expr)

            if port.kind == "interface":
                mode = PrimitiveMode(
                    port_index=port_index, component="normal", polynomial="P0",
                    interface_id=int(port.global_interface), node_index=None,
                    known_coefficient=None,
                )
            else:
                mode = PrimitiveMode(
                    port_index=port_index, component="normal", polynomial="P0",
                    interface_id=None, node_index=None,
                    known_coefficient=float(port.pressure),
                )
            sc.value[:] = [1.0, 0.0, 0.0, 0.0]
            dc.value[:] = np.zeros(3)
            nc.value = -1.0
            specs.append(PrimitiveSpec(mode=mode,
                load=_constant_load(operator, sc, dc, nc, load_form)))
        return tuple(specs)

    def active_indices(self, primitive_modes, port_index, level):
        # All primitives are always active (one per port).
        return tuple(
            i for i, m in enumerate(primitive_modes) if m.port_index == port_index
        )

    def global_keys(self, level, interface_id):
        return ((interface_id, "normal", "P0"),)


# ---------------------------------------------------------------------------
# Reusable load assembler (constant-coefficient kernel)
# ---------------------------------------------------------------------------

def _constant_load(operator, sc, dc, nc, load_form,
                   coeffs=None, direction=None, normal_coef=None):
    """Return a zero-argument callable that mutates the Constants, assembles,
    applies lifting/BCs, and returns the numpy rhs column.

    The *load_form* is pre-compiled once per port; *sc*, *dc*, *nc* are the
    mutable fem.Constant placeholders embedded in it.
    """
    if coeffs is not None:
        sc.value[:] = coeffs
    if direction is not None:
        dc.value[:] = direction
    if normal_coef is not None:
        nc.value = normal_coef

    def _assemble(_op=None):
        b = fem.assemble_vector(load_form)
        fem.apply_lifting(b.array, [operator.a_form], [operator.bcs])
        fem.set_bc(b.array, operator.bcs)
        from ddpnm_core.fem_utils import to_numpy_vector
        return to_numpy_vector(b)

    return _assemble

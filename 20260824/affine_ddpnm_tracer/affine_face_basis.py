"""Isolated one-entity-per-face affine basis for the Affine-DDPNM study."""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem

from ddpnm_core.basis import PrimitiveMode, PrimitiveSpec
from ddpnm_core.constants import PORT_TAG_BASE
from ddpnm_core.fem_utils import to_numpy_vector
from ddpnm3d.basis_3d import (
    COMPONENTS,
    POLYNOMIALS,
    ClassicP0Basis,
    HierarchyBasis,
)


class CompatibleClassicP0Basis(ClassicP0Basis):
    """Classic basis with the direct-selection hook used by the new core."""

    def active_transform(self, primitive_modes, port_index, level):
        return None


class AffineFaceBasis(HierarchyBasis):
    """Nine generalized modes attached to one representative face entity.

    This is not a nodal or uniform-point space.  On interface ``f`` the nine
    unknown coefficients multiply

        {1, s_f, t_f} x {n_f, tangent_1_f, tangent_2_f}.

    The representative point is the interface center, where ``s_f=t_f=0``.
    """

    name = "3D-single-entity-affine-DDPNM"

    def active_transform(self, primitive_modes, port_index, level):
        return None

    def primitive_specs(self, partition, operator):
        specs: list[PrimitiveSpec] = []
        pore_id = operator.pore_id
        ports = operator.ports
        submesh = operator.submesh

        interface_data: dict[int, dict] = {}
        for port in ports:
            if port.kind != "interface":
                continue
            interface_id = int(port.global_interface)
            if interface_id in interface_data:
                continue
            pair = partition.interface_pairs[interface_id]
            side_sign = 1.0 if pore_id == pair[0] else -1.0
            tangent_1 = self.interface_tangents[interface_id, 0]
            tangent_2 = self.interface_tangents[interface_id, 1]
            center = partition.interface_centers[interface_id]
            scale_s, scale_t = self.interface_scales[interface_id]
            interface_data[interface_id] = {
                "scalar": {
                    "P0": np.asarray([1.0, 0.0, 0.0, 0.0]),
                    "P1_s": np.concatenate(
                        ([-float(center @ tangent_1) / scale_s], tangent_1 / scale_s)
                    ),
                    "P1_t": np.concatenate(
                        ([-float(center @ tangent_2) / scale_t], tangent_2 / scale_t)
                    ),
                },
                "directions": {
                    "normal": (np.zeros(3), -1.0),
                    "tangent_1": (side_sign * tangent_1, 0.0),
                    "tangent_2": (side_sign * tangent_2, 0.0),
                },
            }

        for port_index, port in enumerate(ports):
            tag = PORT_TAG_BASE + port_index
            scalar_constant = fem.Constant(submesh, np.zeros(4))
            direction_constant = fem.Constant(submesh, np.zeros(3))
            normal_constant = fem.Constant(submesh, 0.0)
            x = ufl.SpatialCoordinate(submesh)
            local_normal = ufl.FacetNormal(submesh)
            scalar_shape = (
                scalar_constant[0]
                + scalar_constant[1] * x[0]
                + scalar_constant[2] * x[1]
                + scalar_constant[3] * x[2]
            )
            direction = direction_constant + normal_constant * local_normal
            velocity_test, _ = ufl.split(ufl.TestFunction(operator.W))
            load_form = fem.form(
                scalar_shape
                * ufl.dot(direction, velocity_test)
                * ufl.ds(
                    tag,
                    domain=submesh,
                    subdomain_data=operator.facet_tags,
                )
            )

            if port.kind != "interface":
                mode = PrimitiveMode(
                    port_index=port_index,
                    component="normal",
                    polynomial="P0",
                    interface_id=None,
                    node_index=None,
                    known_coefficient=float(port.pressure),
                )
                specs.append(
                    PrimitiveSpec(
                        mode=mode,
                        load=_runtime_load(
                            operator,
                            load_form,
                            scalar_constant,
                            direction_constant,
                            normal_constant,
                            np.asarray([1.0, 0.0, 0.0, 0.0]),
                            np.zeros(3),
                            -1.0,
                        ),
                    )
                )
                continue

            interface_id = int(port.global_interface)
            data = interface_data[interface_id]
            for component in COMPONENTS:
                direction_vector, normal_coefficient = data["directions"][component]
                for polynomial in POLYNOMIALS:
                    mode = PrimitiveMode(
                        port_index=port_index,
                        component=component,
                        polynomial=polynomial,
                        interface_id=interface_id,
                        node_index=None,
                        known_coefficient=None,
                    )
                    specs.append(
                        PrimitiveSpec(
                            mode=mode,
                            load=_runtime_load(
                                operator,
                                load_form,
                                scalar_constant,
                                direction_constant,
                                normal_constant,
                                data["scalar"][polynomial],
                                direction_vector,
                                normal_coefficient,
                            ),
                        )
                    )
        return tuple(specs)


class NormalLinearFaceBasis(HierarchyBasis):
    """Three normal-only generalized modes attached to one representative
    face entity.  On interface ``f`` the three unknown coefficients
    multiply

        {1, s_f, t_f} x {n_f}.

    The representative point is the interface center, where ``s_f=t_f=0``.
    This is the W_{1n} control space: it isolates the effect of adding
    linear-in-s/t variation to the classical constant-normal mode, with
    no tangential traction components.
    """

    name = "3D-single-entity-normal-linear-DDPNM"

    def active_transform(self, primitive_modes, port_index, level):
        return None

    def primitive_specs(self, partition, operator):
        specs: list[PrimitiveSpec] = []
        pore_id = operator.pore_id
        ports = operator.ports
        submesh = operator.submesh

        interface_data: dict[int, dict] = {}
        for port in ports:
            if port.kind != "interface":
                continue
            interface_id = int(port.global_interface)
            if interface_id in interface_data:
                continue
            pair = partition.interface_pairs[interface_id]
            side_sign = 1.0 if pore_id == pair[0] else -1.0
            tangent_1 = self.interface_tangents[interface_id, 0]
            tangent_2 = self.interface_tangents[interface_id, 1]
            center = partition.interface_centers[interface_id]
            scale_s, scale_t = self.interface_scales[interface_id]
            interface_data[interface_id] = {
                "scalar": {
                    "P0": np.asarray([1.0, 0.0, 0.0, 0.0]),
                    "P1_s": np.concatenate(
                        ([-float(center @ tangent_1) / scale_s], tangent_1 / scale_s)
                    ),
                    "P1_t": np.concatenate(
                        ([-float(center @ tangent_2) / scale_t], tangent_2 / scale_t)
                    ),
                },
                "directions": {
                    "normal": (np.zeros(3), -1.0),
                },
            }

        for port_index, port in enumerate(ports):
            tag = PORT_TAG_BASE + port_index
            scalar_constant = fem.Constant(submesh, np.zeros(4))
            direction_constant = fem.Constant(submesh, np.zeros(3))
            normal_constant = fem.Constant(submesh, 0.0)
            x = ufl.SpatialCoordinate(submesh)
            local_normal = ufl.FacetNormal(submesh)
            scalar_shape = (
                scalar_constant[0]
                + scalar_constant[1] * x[0]
                + scalar_constant[2] * x[1]
                + scalar_constant[3] * x[2]
            )
            direction = direction_constant + normal_constant * local_normal
            velocity_test, _ = ufl.split(ufl.TestFunction(operator.W))
            load_form = fem.form(
                scalar_shape
                * ufl.dot(direction, velocity_test)
                * ufl.ds(
                    tag,
                    domain=submesh,
                    subdomain_data=operator.facet_tags,
                )
            )

            if port.kind != "interface":
                mode = PrimitiveMode(
                    port_index=port_index,
                    component="normal",
                    polynomial="P0",
                    interface_id=None,
                    node_index=None,
                    known_coefficient=float(port.pressure),
                )
                specs.append(
                    PrimitiveSpec(
                        mode=mode,
                        load=_runtime_load(
                            operator,
                            load_form,
                            scalar_constant,
                            direction_constant,
                            normal_constant,
                            np.asarray([1.0, 0.0, 0.0, 0.0]),
                            np.zeros(3),
                            -1.0,
                        ),
                    )
                )
                continue

            interface_id = int(port.global_interface)
            data = interface_data[interface_id]
            for component, (direction_vector, normal_coefficient) in data["directions"].items():
                for polynomial in POLYNOMIALS:
                    mode = PrimitiveMode(
                        port_index=port_index,
                        component=component,
                        polynomial=polynomial,
                        interface_id=interface_id,
                        node_index=None,
                        known_coefficient=None,
                    )
                    specs.append(
                        PrimitiveSpec(
                            mode=mode,
                            load=_runtime_load(
                                operator,
                                load_form,
                                scalar_constant,
                                direction_constant,
                                normal_constant,
                                data["scalar"][polynomial],
                                direction_vector,
                                normal_coefficient,
                            ),
                        )
                    )
        return tuple(specs)

    def active_indices(self, primitive_modes, port_index, level):
        # All primitives are active: this basis emits exactly the W_{1n}
        # target space (3 modes per interface + 1 known boundary mode).
        del level
        return tuple(
            i
            for i, m in enumerate(primitive_modes)
            if m.port_index == port_index
        )

    def global_keys(self, level, interface_id):
        del level
        return tuple(
            (interface_id, "normal", polynomial)
            for polynomial in POLYNOMIALS
        )


class VectorConstantFaceBasis(HierarchyBasis):
    """Three constant-vector modes per interface: W_{0v} = span{1} x {n, t1, t2}.

    On interface ``f`` the three unknown coefficients multiply

        {1} x {n_f, tangent_1_f, tangent_2_f}.

    The representative point is the interface center, where ``s_f=t_f=0``.
    This is the W_{0v} control space: the tangential constant modes, which
    complement W_{1n} in the nested path W_{0n} -> W_{0v} -> W_{1v}.
    """

    name = "3D-single-entity-vector-constant-DDPNM"

    def active_transform(self, primitive_modes, port_index, level):
        return None

    def primitive_specs(self, partition, operator):
        specs: list[PrimitiveSpec] = []
        pore_id = operator.pore_id
        ports = operator.ports
        submesh = operator.submesh

        interface_data: dict[int, dict] = {}
        for port in ports:
            if port.kind != "interface":
                continue
            interface_id = int(port.global_interface)
            if interface_id in interface_data:
                continue
            pair = partition.interface_pairs[interface_id]
            side_sign = 1.0 if pore_id == pair[0] else -1.0
            tangent_1 = self.interface_tangents[interface_id, 0]
            tangent_2 = self.interface_tangents[interface_id, 1]
            interface_data[interface_id] = {
                "directions": {
                    "normal": (np.zeros(3), -1.0),
                    "tangent_1": (side_sign * tangent_1, 0.0),
                    "tangent_2": (side_sign * tangent_2, 0.0),
                },
            }

        for port_index, port in enumerate(ports):
            tag = PORT_TAG_BASE + port_index
            scalar_constant = fem.Constant(submesh, np.zeros(4))
            direction_constant = fem.Constant(submesh, np.zeros(3))
            normal_constant = fem.Constant(submesh, 0.0)
            x = ufl.SpatialCoordinate(submesh)
            local_normal = ufl.FacetNormal(submesh)
            scalar_shape = (
                scalar_constant[0]
                + scalar_constant[1] * x[0]
                + scalar_constant[2] * x[1]
                + scalar_constant[3] * x[2]
            )
            direction = direction_constant + normal_constant * local_normal
            velocity_test, _ = ufl.split(ufl.TestFunction(operator.W))
            load_form = fem.form(
                scalar_shape
                * ufl.dot(direction, velocity_test)
                * ufl.ds(
                    tag,
                    domain=submesh,
                    subdomain_data=operator.facet_tags,
                )
            )

            if port.kind != "interface":
                mode = PrimitiveMode(
                    port_index=port_index,
                    component="normal",
                    polynomial="P0",
                    interface_id=None,
                    node_index=None,
                    known_coefficient=float(port.pressure),
                )
                specs.append(
                    PrimitiveSpec(
                        mode=mode,
                        load=_runtime_load(
                            operator,
                            load_form,
                            scalar_constant,
                            direction_constant,
                            normal_constant,
                            np.asarray([1.0, 0.0, 0.0, 0.0]),
                            np.zeros(3),
                            -1.0,
                        ),
                    )
                )
                continue

            interface_id = int(port.global_interface)
            data = interface_data[interface_id]
            for component, (direction_vector, normal_coefficient) in data["directions"].items():
                mode = PrimitiveMode(
                    port_index=port_index,
                    component=component,
                    polynomial="P0",
                    interface_id=interface_id,
                    node_index=None,
                    known_coefficient=None,
                )
                specs.append(
                    PrimitiveSpec(
                        mode=mode,
                        load=_runtime_load(
                            operator,
                            load_form,
                            scalar_constant,
                            direction_constant,
                            normal_constant,
                            np.asarray([1.0, 0.0, 0.0, 0.0]),
                            direction_vector,
                            normal_coefficient,
                        ),
                    )
                )
        return tuple(specs)

    def active_indices(self, primitive_modes, port_index, level):
        # All primitives are active: this basis emits exactly the W_{0v}
        # target space (3 modes per interface + 1 known boundary mode).
        del level
        return tuple(
            i
            for i, m in enumerate(primitive_modes)
            if m.port_index == port_index
        )

    def global_keys(self, level, interface_id):
        del level
        return tuple(
            (interface_id, component, "P0")
            for component in COMPONENTS
        )


def _runtime_load(
    operator,
    load_form,
    scalar_constant,
    direction_constant,
    normal_constant,
    scalar_coefficients,
    direction_coefficients,
    normal_coefficient,
):
    """Make a response load that restores its own coefficients every call."""
    scalar_values = np.asarray(scalar_coefficients, dtype=float).copy()
    direction_values = np.asarray(direction_coefficients, dtype=float).copy()
    normal_value = float(normal_coefficient)

    def assemble_load(_operator=None):
        scalar_constant.value[:] = scalar_values
        direction_constant.value[:] = direction_values
        normal_constant.value = normal_value
        vector = fem.assemble_vector(load_form)
        fem.apply_lifting(vector.array, [operator.a_form], [operator.bcs])
        fem.set_bc(vector.array, operator.bcs)
        return to_numpy_vector(vector)

    return assemble_load

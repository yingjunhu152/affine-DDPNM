"""Cross-skeleton, minimum-Stokes-energy interface reduction for 3-D DDPNM.

This module is deliberately self-contained inside the 3-D demonstration
project.  It does not modify :mod:`ddpnm_core` and it preserves the DDPNM
workflow:

1. factorise one local Taylor--Hood Stokes operator per pore;
2. build local traction-to-velocity response maps;
3. assemble a small global system of generalised interface tractions;
4. reconstruct independent local Stokes fields.

The only change is the interface traction space.  A full facewise P1 nodal
space is first used as an offline primitive space.  On every throat face a
mesh-conforming cross is routed through the analytic clearance saddle.  The
cross values are extended to the complete face by minimising the two-sided
discrete Stokes/Steklov energy.  The resulting basis has O(h^-1) unknowns per
face instead of O(h^-2) full-face nodal unknowns.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable

import numpy as np
import ufl
from dolfinx import fem
from scipy.sparse import coo_matrix, csr_matrix, eye as sparse_eye
from scipy.sparse.linalg import eigsh, spsolve

from ddpnm_core.basis import PrimitiveMode, PrimitiveSpec, _interface_nodes
from ddpnm_core.constants import PORT_TAG_BASE
from ddpnm_core.fem_utils import to_numpy_vector
from ddpnm_core.library import ResponseLibrary, build_response_library
from ddpnm3d.solver import LocalResponse


COMPONENTS = ("normal", "tangent_1", "tangent_2")


@dataclass(frozen=True)
class CrossSkeleton:
    """One mesh-conforming cross on a throat interface."""

    interface_id: int
    full_nodes: tuple[int, ...]
    saddle: np.ndarray
    center_node: int
    tangent_1: np.ndarray
    tangent_2: np.ndarray
    endpoints: tuple[int, int, int, int]
    arms: tuple[tuple[int, ...], ...]
    skeleton_nodes: tuple[int, ...]


@dataclass(frozen=True)
class UniformControlSet:
    """Quasi-uniform interface vertices selected by geodesic FPS.

    The saddle-nearest mesh vertex is always the first control point.  The
    remaining vertices are chosen by deterministic geodesic farthest-point
    sampling on the interface triangle graph.  ``skeleton_nodes`` is kept as
    the common name consumed by the minimum-energy extension routines.
    """

    interface_id: int
    full_nodes: tuple[int, ...]
    saddle: np.ndarray
    center_node: int
    tangent_1: np.ndarray
    tangent_2: np.ndarray
    skeleton_nodes: tuple[int, ...]
    target_count: int
    covering_radius: float
    minimum_separation: float


@dataclass(frozen=True)
class ExtensionDiagnostics:
    interface_id: int
    full_scalar_dofs: int
    skeleton_scalar_dofs: int
    full_normal_dofs: int
    active_normal_dofs: int
    cross_normal_dofs: int
    conservative_p0_dofs: int
    reduction_ratio: float
    constraint_residual: float
    constant_reproduction_residual: float
    energy_condition: float
    compliance_positive_rank_side_0: int
    compliance_positive_rank_side_1: int


@dataclass(frozen=True)
class VectorExtensionDiagnostics:
    interface_id: int
    full_scalar_nodes: int
    skeleton_scalar_nodes: int
    full_vector_dofs: int
    active_vector_dofs: int
    reduction_ratio: float
    cardinal_residual: float
    constant_vector_reproduction_residual: float
    energy_condition: float
    positive_rank_side_0: int
    positive_rank_side_1: int


@dataclass
class SkeletonDdpnmSolution:
    """DDPNM result with a variable number of skeleton modes per interface."""

    coefficients: np.ndarray
    global_keys: tuple[tuple, ...]
    schur_matrix: csr_matrix
    rhs: np.ndarray
    local_responses: list[LocalResponse]
    local_solutions: list[np.ndarray]
    moment_residuals: np.ndarray
    boundary_fluxes: dict[str, float]
    min_schur_eigenvalue: float
    max_moment_residual: float
    max_local_mass_residual: float
    max_local_linear_residual: float
    max_flux_divergence_discrepancy: float
    relative_linear_residual: float


@dataclass
class CrossTangentialComparison:
    ddpnm: SkeletonDdpnmSolution
    ddpnmt: SkeletonDdpnmSolution
    cross_normal: SkeletonDdpnmSolution
    cross_ddpnmt: SkeletonDdpnmSolution
    skeletons: tuple[CrossSkeleton, ...]
    normal_transforms: tuple[np.ndarray, ...]
    vector_transforms: tuple[np.ndarray, ...]
    normal_diagnostics: tuple[ExtensionDiagnostics, ...]
    vector_diagnostics: tuple[VectorExtensionDiagnostics, ...]


@dataclass
class CrossUniformTangentialComparison:
    """Six nested spaces assembled from one full nodal response library."""

    ddpnm: SkeletonDdpnmSolution
    ddpnmt: SkeletonDdpnmSolution
    cross_normal: SkeletonDdpnmSolution
    cross_ddpnmt: SkeletonDdpnmSolution
    uniform_normal: SkeletonDdpnmSolution
    uniform_ddpnmt: SkeletonDdpnmSolution
    skeletons: tuple[CrossSkeleton, ...]
    uniform_sets: tuple[UniformControlSet, ...]
    cross_normal_transforms: tuple[np.ndarray, ...]
    cross_vector_transforms: tuple[np.ndarray, ...]
    uniform_normal_transforms: tuple[np.ndarray, ...]
    uniform_vector_transforms: tuple[np.ndarray, ...]
    cross_normal_diagnostics: tuple[ExtensionDiagnostics, ...]
    cross_vector_diagnostics: tuple[VectorExtensionDiagnostics, ...]
    uniform_normal_diagnostics: tuple[ExtensionDiagnostics, ...]
    uniform_vector_diagnostics: tuple[VectorExtensionDiagnostics, ...]


def _assemble_load(operator, load_form) -> np.ndarray:
    vector = fem.assemble_vector(load_form)
    fem.apply_lifting(vector.array, [operator.a_form], [operator.bcs])
    fem.set_bc(vector.array, operator.bcs)
    return to_numpy_vector(vector)


def _boundary_load(operator, load_form):
    def assemble(_operator=None):
        return _assemble_load(operator, load_form)

    return assemble


def _nodal_load(
    operator,
    scalar_function,
    scalar_dof: int,
    direction_constant,
    direction: np.ndarray,
    load_form,
):
    stored_direction = np.asarray(direction, dtype=float).copy()

    def assemble(_operator=None):
        scalar_function.x.array[:] = 0.0
        scalar_function.x.array[int(scalar_dof)] = 1.0
        scalar_function.x.scatter_forward()
        direction_constant.value[:] = stored_direction
        return _assemble_load(operator, load_form)

    return assemble


def _nodal_normal_load(
    operator,
    scalar_function,
    scalar_dof: int,
    load_form,
):
    """Assemble one nodal pressure load using the exact local facet normal."""

    def assemble(_operator=None):
        scalar_function.x.array[:] = 0.0
        scalar_function.x.array[int(scalar_dof)] = 1.0
        scalar_function.x.scatter_forward()
        return _assemble_load(operator, load_form)

    return assemble


def _fallback_tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    candidates = np.eye(3)
    reference = candidates[int(np.argmin(np.abs(candidates @ normal)))]
    tangent_1 = reference - float(reference @ normal) * normal
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(normal, tangent_1)
    tangent_2 /= np.linalg.norm(tangent_2)
    return tangent_1, tangent_2


def _principal_tangent_frame(
    coordinates: np.ndarray,
    normal: np.ndarray,
    saddle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA frame in the throat plane with a deterministic circular fallback."""
    fallback_1, fallback_2 = _fallback_tangent_frame(normal)
    relative = np.asarray(coordinates, dtype=float) - np.asarray(saddle, dtype=float)
    planar = np.column_stack((relative @ fallback_1, relative @ fallback_2))
    covariance = planar.T @ planar / max(len(planar), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    scale = max(float(eigenvalues[-1]), 1.0e-30)
    if float(abs(eigenvalues[-1] - eigenvalues[0]) / scale) < 1.0e-3:
        return fallback_1, fallback_2
    axis = eigenvectors[:, -1]
    tangent_1 = axis[0] * fallback_1 + axis[1] * fallback_2
    tangent_1 /= np.linalg.norm(tangent_1)
    if float(tangent_1 @ fallback_1) < 0.0:
        tangent_1 *= -1.0
    tangent_2 = np.cross(np.asarray(normal, dtype=float), tangent_1)
    tangent_2 /= np.linalg.norm(tangent_2)
    return tangent_1, tangent_2


class FullNodalStokesBasis:
    """Offline full facewise-P1 vector traction basis.

    Every interface vertex carries three traction modes in the canonical
    frame ``(-n, t1, t2)`` on the first pore and the opposite frame on the
    second pore.  This sign convention gives equal-and-opposite physical
    tractions for a shared global coefficient.
    """

    name = "3D-full-nodal-offline-Stokes"
    component_names = COMPONENTS
    level_dofs = (-1, -1, -1)

    def __init__(self, partition):
        self.interface_nodes = _interface_nodes(partition)
        self.interface_frames = np.empty((len(self.interface_nodes), 3, 3))
        coordinates = partition.mesh.geometry.x[:, :3]
        for interface_id, nodes in enumerate(self.interface_nodes):
            saddle = np.asarray(partition.throats[interface_id].saddle, dtype=float)
            # The strict interface is the saddle plane stored by the throat.
            # A line between adjacent maximal-ball centres is generally
            # oblique near the external boundary because the two balls have
            # different radii; it must not be used as the face normal.
            normal = np.asarray(partition.throats[interface_id].normal, dtype=float)
            tangent_1, tangent_2 = _principal_tangent_frame(
                coordinates[np.asarray(nodes, dtype=np.int32)], normal, saddle
            )
            self.interface_frames[interface_id, 0] = -normal
            self.interface_frames[interface_id, 1] = tangent_1
            self.interface_frames[interface_id, 2] = tangent_2

    def primitive_specs(self, partition, operator):
        specs: list[PrimitiveSpec] = []
        submesh = operator.submesh
        submesh.topology.create_connectivity(0, submesh.topology.dim)
        pore_id = int(operator.pore_id)
        parent_to_local = {
            int(parent): int(local)
            for local, parent in enumerate(operator.parent_vertex_map)
        }
        scalar_space = fem.functionspace(submesh, ("Lagrange", 1))
        (test_velocity, _) = ufl.split(ufl.TestFunction(operator.W))
        ds = ufl.Measure(
            "ds", domain=submesh, subdomain_data=operator.facet_tags
        )
        local_normal = ufl.FacetNormal(submesh)

        for port_index, port in enumerate(operator.ports):
            tag = PORT_TAG_BASE + port_index
            if port.kind != "interface":
                form = fem.form(
                    -ufl.dot(local_normal, test_velocity) * ds(tag)
                )
                specs.append(
                    PrimitiveSpec(
                        mode=PrimitiveMode(
                            port_index=port_index,
                            component="normal",
                            polynomial="P0",
                            interface_id=None,
                            node_index=None,
                            known_coefficient=float(port.pressure),
                        ),
                        load=_boundary_load(operator, form),
                    )
                )
                continue

            interface_id = int(port.global_interface)
            pair = partition.interface_pairs[interface_id]
            if pore_id == pair[0]:
                side_sign = 1.0
            elif pore_id == pair[1]:
                side_sign = -1.0
            else:
                raise RuntimeError(
                    f"Pore {pore_id} is not incident on interface {interface_id}."
                )

            scalar_function = fem.Function(scalar_space)
            direction_constant = fem.Constant(submesh, np.zeros(3, dtype=float))
            tangent_form = fem.form(
                scalar_function
                * ufl.dot(direction_constant, test_velocity)
                * ds(tag)
            )
            normal_form = fem.form(
                -scalar_function
                * ufl.dot(local_normal, test_velocity)
                * ds(tag)
            )
            for node_index, parent_vertex in enumerate(
                self.interface_nodes[interface_id]
            ):
                local_vertex = parent_to_local[int(parent_vertex)]
                scalar_dofs = np.asarray(
                    fem.locate_dofs_topological(
                        scalar_space, 0, np.asarray([local_vertex], dtype=np.int32)
                    ),
                    dtype=np.int32,
                ).ravel()
                if len(scalar_dofs) != 1:
                    raise RuntimeError(
                        "The scalar P1 interface hat does not have exactly one vertex dof."
                    )
                for component_index, component in enumerate(COMPONENTS):
                    direction = side_sign * self.interface_frames[
                        interface_id, component_index
                    ]
                    if component == "normal":
                        load = _nodal_normal_load(
                            operator,
                            scalar_function,
                            int(scalar_dofs[0]),
                            normal_form,
                        )
                    else:
                        load = _nodal_load(
                            operator,
                            scalar_function,
                            int(scalar_dofs[0]),
                            direction_constant,
                            direction,
                            tangent_form,
                        )
                    specs.append(
                        PrimitiveSpec(
                            mode=PrimitiveMode(
                                port_index=port_index,
                                component=component,
                                polynomial="nodal",
                                interface_id=interface_id,
                                node_index=node_index,
                                known_coefficient=None,
                            ),
                            load=load,
                        )
                    )
        return tuple(specs)

    def active_indices(self, primitive_modes, port_index, level):
        return tuple(
            index
            for index, mode in enumerate(primitive_modes)
            if mode.port_index == port_index
        )

    def active_transform(self, primitive_modes, port_index, level):
        return None

    def global_keys(self, level, interface_id):
        return tuple(
            (int(interface_id), "full", int(node), component)
            for node in self.interface_nodes[int(interface_id)]
            for component in COMPONENTS
        )


def _interface_triangles(partition, interface_id: int) -> np.ndarray:
    mesh = partition.mesh
    facet_dimension = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facet_dimension, 0)
    facet_to_vertex = mesh.topology.connectivity(facet_dimension, 0)
    facets = np.flatnonzero(
        np.asarray(partition.facet_interface_ids) == int(interface_id)
    )
    return np.asarray(
        [facet_to_vertex.links(int(facet)) for facet in facets], dtype=np.int32
    )


def _face_graph(
    triangles: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[dict[int, dict[int, float]], tuple[int, ...]]:
    edge_counts: dict[tuple[int, int], int] = {}
    graph: dict[int, dict[int, float]] = {}
    for triangle in triangles:
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            edge = (min(first, second), max(first, second))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            length = float(np.linalg.norm(coordinates[first] - coordinates[second]))
            graph.setdefault(first, {})[second] = length
            graph.setdefault(second, {})[first] = length
    boundary = sorted(
        {node for edge, count in edge_counts.items() if count == 1 for node in edge}
    )
    return graph, tuple(boundary)


def _single_source_geodesic_distances(
    graph: dict[int, dict[int, float]],
    start: int,
) -> dict[int, float]:
    """Dijkstra distances on one triangulated interface graph."""
    start = int(start)
    if start not in graph:
        raise ValueError(f"Start vertex {start} is not in the interface graph.")
    distances = {node: float("inf") for node in graph}
    distances[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor, length in graph[node].items():
            candidate = distance + float(length)
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, int(neighbor)))
    return distances


def _uniform_target_count(
    full_node_count: int,
    sampling_factor: float = 1.0,
) -> int:
    """Return ``ceil(factor*sqrt(N_f))``, clipped to ``[1, N_f]``."""
    full_node_count = int(full_node_count)
    if full_node_count < 1:
        raise ValueError("An interface must contain at least one mesh vertex.")
    if sampling_factor <= 0.0:
        raise ValueError("The uniform sampling factor must be positive.")
    requested = int(np.ceil(float(sampling_factor) * np.sqrt(full_node_count)))
    return min(full_node_count, max(1, requested))


def _geodesic_farthest_points(
    graph: dict[int, dict[int, float]],
    start: int,
    count: int,
) -> tuple[tuple[int, ...], float, float]:
    """Saddle-seeded deterministic geodesic farthest-point sampling.

    At every step, select the vertex whose distance to the current sample is
    maximal.  Mesh-index tie breaking makes repeated runs bitwise stable.
    The returned radii measure final covering and pairwise separation.
    """
    if not graph:
        raise ValueError("Cannot sample an empty interface graph.")
    count = min(len(graph), max(1, int(count)))
    start = int(start)
    selected = [start]
    selected_set = {start}
    nearest = _single_source_geodesic_distances(graph, start)
    insertion_separations: list[float] = []
    while len(selected) < count:
        candidates = (node for node in graph if node not in selected_set)
        next_node = max(candidates, key=lambda node: (nearest[node], -int(node)))
        separation = float(nearest[next_node])
        selected.append(int(next_node))
        selected_set.add(int(next_node))
        insertion_separations.append(separation)
        distances = _single_source_geodesic_distances(graph, int(next_node))
        for node in graph:
            nearest[node] = min(nearest[node], distances[node])

    covering_radius = float(max(nearest.values()))
    if not np.isfinite(covering_radius):
        raise RuntimeError("The interface triangle graph is disconnected.")
    minimum_separation = (
        float(min(insertion_separations)) if insertion_separations else 0.0
    )
    return tuple(selected), covering_radius, minimum_separation


def _shortest_aligned_path(
    graph: dict[int, dict[int, float]],
    coordinates: np.ndarray,
    start: int,
    candidates: Iterable[int],
    direction: np.ndarray,
    saddle: np.ndarray,
    alignment_penalty: float,
) -> tuple[int, ...]:
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    candidate_list = list(candidates)
    candidate_list.sort(
        key=lambda node: float((coordinates[int(node)] - saddle) @ direction),
        reverse=True,
    )
    candidate_set = set(int(node) for node in candidate_list)
    distances = {int(start): 0.0}
    previous: dict[int, int] = {}
    queue: list[tuple[float, int]] = [(0.0, int(start))]
    target = None
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances.get(node):
            continue
        if node in candidate_set and node != int(start):
            # Prefer extreme nodes: accept only from the leading candidate band.
            leading = candidate_list[: max(1, min(6, len(candidate_list)))]
            if node in leading:
                target = node
                break
        for neighbor, length in graph[node].items():
            edge_direction = coordinates[neighbor] - coordinates[node]
            edge_direction /= max(float(np.linalg.norm(edge_direction)), 1.0e-30)
            alignment = abs(float(edge_direction @ direction))
            weight = length * (1.0 + alignment_penalty * (1.0 - alignment) ** 2)
            new_distance = distance + weight
            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                previous[neighbor] = node
                heapq.heappush(queue, (new_distance, neighbor))
    if target is None:
        reachable = [node for node in candidate_list if node in distances and node != start]
        if not reachable:
            raise RuntimeError("No interface-boundary node is reachable from the saddle node.")
        target = reachable[0]
    path = [int(target)]
    while path[-1] != int(start):
        path.append(previous[path[-1]])
    path.reverse()
    return tuple(path)


def build_cross_skeletons(
    partition,
    basis: FullNodalStokesBasis,
    alignment_penalty: float = 3.0,
) -> tuple[CrossSkeleton, ...]:
    """Route four interface-mesh paths through the nearest saddle vertex."""
    coordinates = partition.mesh.geometry.x[:, :3]
    skeletons: list[CrossSkeleton] = []
    for interface_id, full_nodes in enumerate(basis.interface_nodes):
        triangles = _interface_triangles(partition, interface_id)
        graph, boundary = _face_graph(triangles, coordinates)
        saddle = np.asarray(partition.throats[interface_id].saddle, dtype=float)
        full_array = np.asarray(full_nodes, dtype=np.int32)
        center_node = int(
            full_array[
                np.argmin(np.linalg.norm(coordinates[full_array] - saddle, axis=1))
            ]
        )
        tangent_1 = basis.interface_frames[interface_id, 1]
        tangent_2 = basis.interface_frames[interface_id, 2]
        directions = (tangent_1, -tangent_1, tangent_2, -tangent_2)
        arms = tuple(
            _shortest_aligned_path(
                graph,
                coordinates,
                center_node,
                boundary,
                direction,
                saddle,
                alignment_penalty,
            )
            for direction in directions
        )
        skeleton_nodes = tuple(sorted({node for arm in arms for node in arm}))
        skeletons.append(
            CrossSkeleton(
                interface_id=interface_id,
                full_nodes=tuple(int(node) for node in full_nodes),
                saddle=saddle,
                center_node=center_node,
                tangent_1=np.asarray(tangent_1, dtype=float).copy(),
                tangent_2=np.asarray(tangent_2, dtype=float).copy(),
                endpoints=tuple(int(arm[-1]) for arm in arms),
                arms=arms,
                skeleton_nodes=skeleton_nodes,
            )
        )
    return tuple(skeletons)


def build_uniform_control_sets(
    partition,
    basis: FullNodalStokesBasis,
    sampling_factor: float = 1.0,
) -> tuple[UniformControlSet, ...]:
    """Select ``ceil(factor*sqrt(N_f))`` quasi-uniform vertices per face."""
    coordinates = partition.mesh.geometry.x[:, :3]
    control_sets: list[UniformControlSet] = []
    for interface_id, full_nodes in enumerate(basis.interface_nodes):
        triangles = _interface_triangles(partition, interface_id)
        graph, _boundary = _face_graph(triangles, coordinates)
        saddle = np.asarray(partition.throats[interface_id].saddle, dtype=float)
        full_array = np.asarray(full_nodes, dtype=np.int32)
        center_node = int(
            full_array[
                np.argmin(np.linalg.norm(coordinates[full_array] - saddle, axis=1))
            ]
        )
        target_count = _uniform_target_count(len(full_nodes), sampling_factor)
        selected, covering_radius, minimum_separation = (
            _geodesic_farthest_points(graph, center_node, target_count)
        )
        control_sets.append(
            UniformControlSet(
                interface_id=interface_id,
                full_nodes=tuple(int(node) for node in full_nodes),
                saddle=saddle,
                center_node=center_node,
                tangent_1=np.asarray(
                    basis.interface_frames[interface_id, 1], dtype=float
                ).copy(),
                tangent_2=np.asarray(
                    basis.interface_frames[interface_id, 2], dtype=float
                ).copy(),
                skeleton_nodes=selected,
                target_count=target_count,
                covering_radius=covering_radius,
                minimum_separation=minimum_separation,
            )
        )
    return tuple(control_sets)


def _face_mass_matrix(partition, skeleton: CrossSkeleton) -> np.ndarray:
    coordinates = partition.mesh.geometry.x[:, :3]
    triangles = _interface_triangles(partition, skeleton.interface_id)
    node_to_local = {node: index for index, node in enumerate(skeleton.full_nodes)}
    mass = np.zeros((len(skeleton.full_nodes), len(skeleton.full_nodes)))
    reference = np.asarray(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]
    )
    for triangle in triangles:
        xyz = coordinates[triangle]
        area = 0.5 * float(
            np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0]))
        )
        local = [node_to_local[int(node)] for node in triangle]
        mass[np.ix_(local, local)] += (area / 12.0) * reference
    return mass


def _positive_energy(matrix: np.ndarray, relative_floor: float) -> tuple[np.ndarray, int]:
    """Return the positive-semidefinite Stokes response energy.

    The interface unknowns are *traction coefficients*.  Their local Stokes
    energy is therefore ``q.T @ G @ q`` with the compliance/response matrix
    ``G = B.T @ A^{-1} @ B`` itself.  Inverting ``G`` would instead be the
    metric for prescribed velocity traces and is not the DDPNM variable used
    here.  Only tiny non-positive roundoff eigenvalues are removed.
    """
    symmetric = 0.5 * (np.asarray(matrix) + np.asarray(matrix).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0e-30)
    floor = max(relative_floor * scale, 1.0e-14)
    positive_rank = int(np.count_nonzero(eigenvalues > floor))
    clipped = np.maximum(eigenvalues, 0.0)
    energy = (eigenvectors * clipped[None, :]) @ eigenvectors.T
    return 0.5 * (energy + energy.T), positive_rank


def _face_primitive_indices(entry, interface_id: int) -> tuple[int, ...]:
    indices = tuple(
        index
        for index, mode in enumerate(entry.primitive_modes)
        if mode.interface_id == int(interface_id)
    )
    return indices


def _constant_preserving_extension(
    energy: np.ndarray,
    restriction: np.ndarray,
    cross_constants: np.ndarray | None = None,
    full_constants: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Cardinal minimum-energy extension preserving supplied constant modes.

    For the scalar case the optional matrices are omitted and the single
    all-ones mode is preserved.  For a vector interface they contain the
    three constant component fields in cross and full-face coordinates.
    """
    harmonic_columns = np.linalg.solve(energy, restriction.T)
    coarse_energy_inverse = restriction @ harmonic_columns
    extension = harmonic_columns @ np.linalg.solve(
        coarse_energy_inverse, np.eye(restriction.shape[0])
    )
    if cross_constants is None:
        cross_constants = np.ones((restriction.shape[0], 1), dtype=float)
    if full_constants is None:
        full_constants = np.ones((energy.shape[0], 1), dtype=float)
    cross_constants = np.asarray(cross_constants, dtype=float)
    full_constants = np.asarray(full_constants, dtype=float)
    if cross_constants.shape[0] != restriction.shape[0]:
        raise ValueError("Cross constant modes do not match the restriction rows.")
    if full_constants.shape[0] != energy.shape[0]:
        raise ValueError("Full constant modes do not match the energy rows.")
    if cross_constants.shape[1] != full_constants.shape[1]:
        raise ValueError("Cross/full constant-mode counts differ.")
    constant_defect = full_constants - extension @ cross_constants
    extension += constant_defect @ np.linalg.pinv(cross_constants)
    cardinal_residual = float(
        np.linalg.norm(restriction @ extension - np.eye(restriction.shape[0]))
    )
    constant_residual = float(
        np.linalg.norm(extension @ cross_constants - full_constants)
    )
    return extension, cardinal_residual, constant_residual


def _face_affine_coordinate_modes(partition, skeleton) -> np.ndarray:
    """Return the scaled facewise affine basis ``{1,s,t}`` at all vertices."""
    coordinates = partition.mesh.geometry.x[
        np.asarray(skeleton.full_nodes, dtype=np.int32), :3
    ]
    relative = coordinates - np.asarray(skeleton.saddle, dtype=float)
    first = relative @ np.asarray(skeleton.tangent_1, dtype=float)
    second = relative @ np.asarray(skeleton.tangent_2, dtype=float)
    scale = max(
        float(np.max(np.abs(first))),
        float(np.max(np.abs(second))),
        1.0e-14,
    )
    return np.column_stack(
        (np.ones(len(coordinates)), first / scale, second / scale)
    )


def _vector_affine_modes(scalar_modes: np.ndarray) -> np.ndarray:
    """Lift ``{1,s,t}`` to nine node-major vector component modes."""
    scalar_modes = np.asarray(scalar_modes, dtype=float)
    modes = np.zeros((3 * len(scalar_modes), 9), dtype=float)
    for node, values in enumerate(scalar_modes):
        for component in range(3):
            modes[3 * node + component, 3 * component : 3 * component + 3] = values
    return modes


def build_minimum_energy_extensions(
    partition,
    full_library: ResponseLibrary,
    skeletons: tuple[CrossSkeleton, ...],
    compliance_floor: float = 1.0e-10,
    energy_ridge: float = 1.0e-10,
    affine_complete: bool = False,
) -> tuple[tuple[np.ndarray, ...], tuple[ExtensionDiagnostics, ...]]:
    """Build two-sided Stokes-energy extension matrices for all interfaces."""
    entries_by_pore = {
        int(entry.operator.pore_id): entry for entry in full_library.entries
    }
    transforms: list[np.ndarray] = []
    diagnostics: list[ExtensionDiagnostics] = []
    for skeleton in skeletons:
        interface_id = skeleton.interface_id
        pair = partition.interface_pairs[interface_id]
        expected = tuple(
            (node_index, component)
            for node_index in range(len(skeleton.full_nodes))
            for component in COMPONENTS
        )
        side_compliances: list[np.ndarray] = []
        ranks: list[int] = []
        for pore_id in pair:
            entry = entries_by_pore[int(pore_id)]
            indices = _face_primitive_indices(entry, interface_id)
            ordered = tuple(
                (
                    int(entry.primitive_modes[index].node_index),
                    entry.primitive_modes[index].component,
                )
                for index in indices
            )
            if ordered != expected:
                raise RuntimeError(
                    f"Primitive ordering mismatch on interface {interface_id}."
                )
            block = entry.primitive_G[np.ix_(indices, indices)]
            side_compliances.append(0.5 * (block + block.T))

        normal_positions = np.arange(0, 3 * len(skeleton.full_nodes), 3, dtype=np.int32)
        energy = np.zeros(
            (len(skeleton.full_nodes), len(skeleton.full_nodes)), dtype=float
        )
        for compliance in side_compliances:
            normal_compliance = compliance[np.ix_(normal_positions, normal_positions)]
            response_energy, rank = _positive_energy(
                normal_compliance, compliance_floor
            )
            ranks.append(rank)
            energy += response_energy
        energy = 0.5 * (energy + energy.T)
        ridge = energy_ridge * max(
            float(np.trace(energy)) / max(len(energy), 1), 1.0e-30
        )
        energy += ridge * np.eye(len(energy))

        node_to_local = {
            node: index for index, node in enumerate(skeleton.full_nodes)
        }
        skeleton_positions = [
            node_to_local[node] for node in skeleton.skeleton_nodes
        ]
        selected = np.asarray(skeleton_positions, dtype=np.int32)
        restriction = np.zeros((len(selected), len(energy)))
        restriction[np.arange(len(selected)), selected] = 1.0
        # The raw harmonic extension is cardinal on the cross, but it need
        # not reproduce a constant on the whole face.  Correct it inside the
        # nullspace of the restriction.  This is the minimum-energy
        # correction subject to both
        #
        #     R E = I            (nodal/cardinal cross coefficients),
        #     E 1 = 1            (exact original DDPNM P0 mode).
        #
        # Consequently there are exactly N_cross unknowns, not N_cross + 1,
        # and each global coefficient is the pressure at one cross node.
        if affine_complete:
            full_exact_modes = _face_affine_coordinate_modes(
                partition, skeleton
            )
            cross_exact_modes = full_exact_modes[selected]
            if np.linalg.matrix_rank(cross_exact_modes) != 3:
                raise RuntimeError(
                    f"Selected points on interface {interface_id} do not span "
                    "the affine face space {1,s,t}."
                )
        else:
            cross_exact_modes = np.ones((len(selected), 1), dtype=float)
            full_exact_modes = np.ones((len(energy), 1), dtype=float)
        harmonic_transform, residual, constant_residual = (
            _constant_preserving_extension(
                energy,
                restriction,
                cross_constants=cross_exact_modes,
                full_constants=full_exact_modes,
            )
        )
        transform = np.zeros((3 * len(energy), harmonic_transform.shape[1]))
        transform[normal_positions, :] = harmonic_transform
        eigenvalues = np.linalg.eigvalsh(energy)
        positive = eigenvalues[eigenvalues > 1.0e-14]
        condition = float(positive[-1] / positive[0]) if len(positive) else float("inf")
        transforms.append(transform)
        diagnostics.append(
            ExtensionDiagnostics(
                interface_id=interface_id,
                full_scalar_dofs=len(skeleton.full_nodes),
                skeleton_scalar_dofs=len(skeleton.skeleton_nodes),
                full_normal_dofs=len(skeleton.full_nodes),
                active_normal_dofs=len(skeleton.skeleton_nodes),
                cross_normal_dofs=len(skeleton.skeleton_nodes),
                conservative_p0_dofs=1,
                reduction_ratio=float(
                    len(skeleton.skeleton_nodes)
                    / max(len(skeleton.full_nodes), 1)
                ),
                constraint_residual=residual,
                constant_reproduction_residual=constant_residual,
                energy_condition=condition,
                compliance_positive_rank_side_0=ranks[0],
                compliance_positive_rank_side_1=ranks[1],
            )
        )
    return tuple(transforms), tuple(diagnostics)


def build_vector_minimum_energy_extensions(
    partition,
    full_library: ResponseLibrary,
    skeletons: tuple[CrossSkeleton, ...],
    compliance_floor: float = 1.0e-10,
    energy_ridge: float = 1.0e-10,
    affine_complete: bool = False,
) -> tuple[tuple[np.ndarray, ...], tuple[VectorExtensionDiagnostics, ...]]:
    """Extend three-component cross-node tractions to each complete face.

    The columns are ordered node-major as ``(normal, tangent_1, tangent_2)``.
    They are cardinal on the cross and exactly reproduce all three constant
    vector modes, so uniform Cross-DDPNMT strictly contains DDPNMT.
    """
    entries_by_pore = {
        int(entry.operator.pore_id): entry for entry in full_library.entries
    }
    transforms: list[np.ndarray] = []
    diagnostics: list[VectorExtensionDiagnostics] = []
    for skeleton in skeletons:
        interface_id = skeleton.interface_id
        pair = partition.interface_pairs[interface_id]
        expected = tuple(
            (node_index, component)
            for node_index in range(len(skeleton.full_nodes))
            for component in COMPONENTS
        )
        size = 3 * len(skeleton.full_nodes)
        energy = np.zeros((size, size), dtype=float)
        ranks: list[int] = []
        for pore_id in pair:
            entry = entries_by_pore[int(pore_id)]
            indices = _face_primitive_indices(entry, interface_id)
            ordered = tuple(
                (
                    int(entry.primitive_modes[index].node_index),
                    entry.primitive_modes[index].component,
                )
                for index in indices
            )
            if ordered != expected:
                raise RuntimeError(
                    f"Primitive ordering mismatch on vector interface {interface_id}."
                )
            compliance = entry.primitive_G[np.ix_(indices, indices)]
            response_energy, rank = _positive_energy(
                compliance, compliance_floor
            )
            energy += response_energy
            ranks.append(rank)
        energy = 0.5 * (energy + energy.T)
        ridge = energy_ridge * max(
            float(np.trace(energy)) / max(len(energy), 1), 1.0e-30
        )
        energy += ridge * np.eye(len(energy))

        node_to_local = {
            node: index for index, node in enumerate(skeleton.full_nodes)
        }
        selected_nodes = [node_to_local[node] for node in skeleton.skeleton_nodes]
        selected = np.asarray(
            [3 * node + component for node in selected_nodes for component in range(3)],
            dtype=np.int32,
        )
        restriction = np.zeros((len(selected), len(energy)), dtype=float)
        restriction[np.arange(len(selected)), selected] = 1.0
        if affine_complete:
            full_scalar_modes = _face_affine_coordinate_modes(
                partition, skeleton
            )
            cross_scalar_modes = full_scalar_modes[selected_nodes]
            if np.linalg.matrix_rank(cross_scalar_modes) != 3:
                raise RuntimeError(
                    f"Selected vector points on interface {interface_id} do not "
                    "span the affine face space {1,s,t}."
                )
            cross_constants = _vector_affine_modes(cross_scalar_modes)
            full_constants = _vector_affine_modes(full_scalar_modes)
        else:
            cross_constants = np.zeros((len(selected), 3), dtype=float)
            full_constants = np.zeros((len(energy), 3), dtype=float)
            for node in range(len(skeleton.skeleton_nodes)):
                for component in range(3):
                    cross_constants[3 * node + component, component] = 1.0
            for node in range(len(skeleton.full_nodes)):
                for component in range(3):
                    full_constants[3 * node + component, component] = 1.0
        transform, cardinal_residual, constant_residual = (
            _constant_preserving_extension(
                energy,
                restriction,
                cross_constants=cross_constants,
                full_constants=full_constants,
            )
        )
        eigenvalues = np.linalg.eigvalsh(energy)
        positive = eigenvalues[eigenvalues > 1.0e-14]
        condition = (
            float(positive[-1] / positive[0]) if len(positive) else float("inf")
        )
        transforms.append(transform)
        diagnostics.append(
            VectorExtensionDiagnostics(
                interface_id=interface_id,
                full_scalar_nodes=len(skeleton.full_nodes),
                skeleton_scalar_nodes=len(skeleton.skeleton_nodes),
                full_vector_dofs=3 * len(skeleton.full_nodes),
                active_vector_dofs=3 * len(skeleton.skeleton_nodes),
                reduction_ratio=float(
                    len(skeleton.skeleton_nodes)
                    / max(len(skeleton.full_nodes), 1)
                ),
                cardinal_residual=cardinal_residual,
                constant_vector_reproduction_residual=constant_residual,
                energy_condition=condition,
                positive_rank_side_0=ranks[0],
                positive_rank_side_1=ranks[1],
            )
        )
    return tuple(transforms), tuple(diagnostics)


class SkeletonStokesBasis:
    """Active O(h^-1) DDPNM basis represented in the full nodal primitives."""

    name = "3D-cross-skeleton-minimum-Stokes-energy"
    component_names = ("normal",)
    level_dofs = (-1, -1, -1)

    def __init__(
        self,
        full_basis: FullNodalStokesBasis,
        skeletons: tuple[CrossSkeleton, ...],
        transforms: tuple[np.ndarray, ...],
    ):
        self.interface_nodes = full_basis.interface_nodes
        self.interface_frames = full_basis.interface_frames
        self.skeletons = skeletons
        self.transforms = transforms

    def primitive_specs(self, partition, operator):
        raise RuntimeError(
            "SkeletonStokesBasis reuses an already-built full nodal response library."
        )

    def active_indices(self, primitive_modes, port_index, level):
        return tuple(
            index
            for index, mode in enumerate(primitive_modes)
            if mode.port_index == port_index
        )

    def active_transform(self, primitive_modes, port_index, level):
        indices = self.active_indices(primitive_modes, port_index, level)
        if not indices:
            return None
        mode = primitive_modes[indices[0]]
        if mode.interface_id is None:
            return None
        return self.transforms[int(mode.interface_id)]

    def global_keys(self, level, interface_id):
        skeleton = self.skeletons[int(interface_id)]
        return tuple(
            (int(interface_id), "cross", int(node), "normal")
            for node in skeleton.skeleton_nodes
        )

    def active_components(self, interface_id):
        return ("normal",) * len(
            self.skeletons[int(interface_id)].skeleton_nodes
        )


class VectorSkeletonStokesBasis(SkeletonStokesBasis):
    """Three traction components at every cardinal cross node."""

    name = "3D-cross-skeleton-vector-minimum-Stokes-energy"
    component_names = COMPONENTS

    def global_keys(self, level, interface_id):
        del level
        skeleton = self.skeletons[int(interface_id)]
        return tuple(
            (int(interface_id), "cross", int(node), component)
            for node in skeleton.skeleton_nodes
            for component in COMPONENTS
        )

    def active_components(self, interface_id):
        return tuple(
            component
            for _node in self.skeletons[int(interface_id)].skeleton_nodes
            for component in COMPONENTS
        )


class UniformPointStokesBasis(SkeletonStokesBasis):
    """Normal traction values at quasi-uniform geodesic FPS points."""

    name = "3D-uniform-points-normal-minimum-Stokes-energy"

    def global_keys(self, level, interface_id):
        del level
        control_set = self.skeletons[int(interface_id)]
        return tuple(
            (int(interface_id), "uniform", int(node), "normal")
            for node in control_set.skeleton_nodes
        )


class UniformVectorPointStokesBasis(VectorSkeletonStokesBasis):
    """Normal plus two tangential values at quasi-uniform FPS points."""

    name = "3D-uniform-points-vector-minimum-Stokes-energy"

    def global_keys(self, level, interface_id):
        del level
        control_set = self.skeletons[int(interface_id)]
        return tuple(
            (int(interface_id), "uniform", int(node), component)
            for node in control_set.skeleton_nodes
            for component in COMPONENTS
        )


class ConstantInterfaceStokesBasis:
    """P0 normal or P0 vector subspaces represented in nodal primitives."""

    level_dofs = (1, 3, 3)

    def __init__(
        self,
        full_basis: FullNodalStokesBasis,
        components: tuple[str, ...],
    ):
        if any(component not in COMPONENTS for component in components):
            raise ValueError("Unknown constant interface component.")
        self.name = "3D-constant-" + "-".join(components)
        self.component_names = components
        self.interface_nodes = full_basis.interface_nodes
        self.interface_frames = full_basis.interface_frames
        self.components = components
        component_to_index = {name: index for index, name in enumerate(COMPONENTS)}
        transforms: list[np.ndarray] = []
        for nodes in self.interface_nodes:
            transform = np.zeros((3 * len(nodes), len(components)), dtype=float)
            for column, component in enumerate(components):
                transform[component_to_index[component] :: 3, column] = 1.0
            transforms.append(transform)
        self.transforms = tuple(transforms)

    def primitive_specs(self, partition, operator):
        raise RuntimeError("ConstantInterfaceStokesBasis reuses the nodal library.")

    def active_indices(self, primitive_modes, port_index, level):
        del level
        return tuple(
            index
            for index, mode in enumerate(primitive_modes)
            if mode.port_index == port_index
        )

    def active_transform(self, primitive_modes, port_index, level):
        del level
        indices = self.active_indices(primitive_modes, port_index, 0)
        if not indices:
            return None
        mode = primitive_modes[indices[0]]
        if mode.interface_id is None:
            return None
        return self.transforms[int(mode.interface_id)]

    def global_keys(self, level, interface_id):
        del level
        return tuple(
            (int(interface_id), component, "P0")
            for component in self.components
        )

    def active_components(self, interface_id):
        del interface_id
        return self.components


def _block_diagonal_transform(
    blocks: list[tuple[tuple[int, ...], np.ndarray]],
    total_primitives: int,
) -> np.ndarray:
    columns = sum(block.shape[1] for _, block in blocks)
    transform = np.zeros((total_primitives, columns))
    offset = 0
    for indices, block in blocks:
        transform[np.asarray(indices, dtype=np.int32), offset : offset + block.shape[1]] = block
        offset += block.shape[1]
    return transform


def assemble_skeleton_ddpnm(
    full_library: ResponseLibrary,
    basis: SkeletonStokesBasis,
) -> SkeletonDdpnmSolution:
    """Sparse global DDPNM assembly using the extended skeleton modes."""
    partition = full_library.partition
    global_keys = tuple(
        key
        for interface_id in range(len(partition.interface_pairs))
        for key in basis.global_keys(0, interface_id)
    )
    key_to_dof = {key: index for index, key in enumerate(global_keys)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs = np.zeros(len(global_keys))
    reconstruction: list[dict] = []
    local_responses: list[LocalResponse] = []

    for entry in full_library.entries:
        blocks: list[tuple[tuple[int, ...], np.ndarray]] = []
        active_info: list[dict] = []
        for port_index, port in enumerate(entry.operator.ports):
            indices = tuple(
                index
                for index, mode in enumerate(entry.primitive_modes)
                if mode.port_index == port_index
            )
            if port.kind == "interface":
                interface_id = int(port.global_interface)
                block = basis.transforms[interface_id]
                keys = basis.global_keys(0, interface_id)
                components = basis.active_components(interface_id)
                if len(keys) != block.shape[1] or len(components) != len(keys):
                    raise RuntimeError(
                        f"Active key/component mismatch on interface {interface_id}."
                    )
                blocks.append((indices, block))
                active_info.extend(
                    {
                        "dof": key_to_dof[key],
                        "known": None,
                        "port_index": port_index,
                        "component": component,
                    }
                    for key, component in zip(keys, components, strict=True)
                )
            else:
                if len(indices) != 1:
                    raise RuntimeError("A physical boundary port must have one P0 mode.")
                blocks.append((indices, np.ones((1, 1))))
                active_info.append(
                    {
                        "dof": -1,
                        "known": float(port.pressure),
                        "port_index": port_index,
                        "component": "normal",
                    }
                )

        transform = _block_diagonal_transform(
            blocks, len(entry.primitive_modes)
        )
        reduced_loads = entry.primitive_loads @ transform
        reduced_responses = entry.operator.solve(reduced_loads)
        local_g = reduced_loads.T @ reduced_responses
        local_g = 0.5 * (local_g + local_g.T)
        unknown = [i for i, info in enumerate(active_info) if info["dof"] >= 0]
        known = [
            i
            for i, info in enumerate(active_info)
            if info["dof"] < 0 and info["known"] is not None
        ]
        global_dofs = np.asarray(
            [active_info[index]["dof"] for index in unknown], dtype=np.int32
        )
        block = local_g[np.ix_(unknown, unknown)]
        grid_row, grid_column = np.meshgrid(global_dofs, global_dofs, indexing="ij")
        rows.extend(grid_row.ravel().tolist())
        columns.extend(grid_column.ravel().tolist())
        values.extend(block.ravel().tolist())
        if known:
            known_values = np.asarray(
                [active_info[index]["known"] for index in known], dtype=float
            )
            rhs[global_dofs] -= local_g[np.ix_(unknown, known)] @ known_values

        reconstruction.append(
            {
                "responses": reduced_responses,
                "loads": reduced_loads,
                "g": local_g,
                "active_info": active_info,
            }
        )
        scale = max(float(np.linalg.norm(local_g)), 1.0e-30)
        local_responses.append(
            LocalResponse(
                pore_id=int(entry.operator.pore_id),
                submesh=entry.operator.submesh,
                parent_cell_map=entry.operator.parent_cell_map,
                parent_vertex_map=entry.operator.parent_vertex_map,
                ports=entry.operator.ports,
                modes=tuple(),
                W=entry.operator.W,
                G=local_g,
                responses=reduced_responses,
                ndofs=entry.operator.ndofs,
                symmetry_error=float(np.linalg.norm(local_g - local_g.T) / scale),
                kernel_error=0.0,
            )
        )

    schur = coo_matrix(
        (values, (rows, columns)), shape=(len(global_keys), len(global_keys))
    ).tocsr()
    schur = 0.5 * (schur + schur.T)
    coefficients = np.asarray(spsolve(schur.tocsc(), rhs), dtype=float)
    if not np.all(np.isfinite(coefficients)):
        diagonal_scale = max(
            float(np.mean(np.abs(schur.diagonal()))), 1.0e-30
        )
        regularized = schur + (1.0e-12 * diagonal_scale) * sparse_eye(
            schur.shape[0], format="csr"
        )
        coefficients = np.asarray(spsolve(regularized.tocsc(), rhs), dtype=float)
        if not np.all(np.isfinite(coefficients)):
            raise RuntimeError("The skeleton DDPNM Schur system is singular.")
    linear_residual = schur @ coefficients - rhs
    relative_linear_residual = float(
        np.linalg.norm(linear_residual) / max(np.linalg.norm(rhs), 1.0e-30)
    )
    try:
        minimum_eigenvalue = float(
            eigsh(schur, k=1, which="SA", return_eigenvectors=False)[0]
        )
    except Exception:
        minimum_eigenvalue = float("nan")

    local_solutions: list[np.ndarray] = []
    moment_residuals = np.zeros(len(global_keys))
    local_mass_residuals: list[float] = []
    local_linear_residuals: list[float] = []
    flux_divergence_discrepancies: list[float] = []
    boundary_fluxes = {"inlet": 0.0, "outlet": 0.0}
    for entry, data in zip(full_library.entries, reconstruction, strict=True):
        active_info = data["active_info"]
        local_coefficients = np.asarray(
            [
                coefficients[info["dof"]]
                if info["dof"] >= 0
                else float(info["known"])
                for info in active_info
            ],
            dtype=float,
        )
        local_solution = data["responses"] @ local_coefficients
        local_load = data["loads"] @ local_coefficients
        local_solutions.append(local_solution)
        moments = data["g"] @ local_coefficients
        # Because the interface nodal basis is a partition of unity, summing
        # all moment rows on a port gives its ordinary total normal flux.
        # Summing over every port of one pore must therefore vanish.
        flux_moment = float(
            sum(
                moments[index]
                for index, info in enumerate(active_info)
                if info["component"] == "normal"
            )
        )
        local_mass_residuals.append(abs(flux_moment))
        algebraic_residual = entry.operator.A @ local_solution - local_load
        local_linear_residuals.append(
            float(
                np.linalg.norm(algebraic_residual)
                / max(np.linalg.norm(local_load), 1.0e-30)
            )
        )
        _, pressure_to_mixed = entry.operator.W.sub(1).collapse()
        constant_pressure_test = np.zeros(entry.operator.ndofs, dtype=float)
        constant_pressure_test[np.asarray(pressure_to_mixed, dtype=np.int32)] = 1.0
        discrete_divergence_moment = float(
            constant_pressure_test @ (entry.operator.A @ local_solution)
        )
        flux_divergence_discrepancies.append(
            abs(flux_moment - discrete_divergence_moment)
        )
        for index, info in enumerate(active_info):
            if info["dof"] >= 0:
                moment_residuals[info["dof"]] += moments[index]
            else:
                port = entry.operator.ports[info["port_index"]]
                if port.kind in boundary_fluxes:
                    boundary_fluxes[port.kind] -= float(moments[index])

    return SkeletonDdpnmSolution(
        coefficients=coefficients,
        global_keys=global_keys,
        schur_matrix=schur,
        rhs=rhs,
        local_responses=local_responses,
        local_solutions=local_solutions,
        moment_residuals=moment_residuals,
        boundary_fluxes=boundary_fluxes,
        min_schur_eigenvalue=minimum_eigenvalue,
        max_moment_residual=float(np.max(np.abs(moment_residuals))),
        max_local_mass_residual=float(max(local_mass_residuals, default=0.0)),
        max_local_linear_residual=float(max(local_linear_residuals, default=0.0)),
        max_flux_divergence_discrepancy=float(
            max(flux_divergence_discrepancies, default=0.0)
        ),
        relative_linear_residual=relative_linear_residual,
    )


def solve_cross_skeleton_ddpnm(
    partition,
    viscosity: float = 1.0,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
    pressure_stabilization: float = 0.0,
    alignment_penalty: float = 3.0,
    compliance_floor: float = 1.0e-10,
    energy_ridge: float = 1.0e-10,
):
    """Complete offline/online cross-skeleton DDPNM workflow."""
    full_basis = FullNodalStokesBasis(partition)
    full_library = build_response_library(
        partition,
        full_basis,
        viscosity=viscosity,
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure,
        pressure_stabilization=pressure_stabilization,
    )
    skeletons = build_cross_skeletons(
        partition, full_basis, alignment_penalty=alignment_penalty
    )
    transforms, diagnostics = build_minimum_energy_extensions(
        partition,
        full_library,
        skeletons,
        compliance_floor=compliance_floor,
        energy_ridge=energy_ridge,
    )
    # The online solve needs the primitive loads but not the full primitive
    # response vectors or full G matrices.  Release those two large offline
    # arrays before constructing the reduced response library.
    for entry in full_library.entries:
        entry.primitive_responses = np.empty((entry.operator.ndofs, 0))
        entry.primitive_G = np.empty((0, 0))
    skeleton_basis = SkeletonStokesBasis(full_basis, skeletons, transforms)
    solution = assemble_skeleton_ddpnm(full_library, skeleton_basis)
    return solution, skeletons, transforms, diagnostics


def solve_cross_tangential_comparison(
    partition,
    viscosity: float = 1.0,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
    pressure_stabilization: float = 0.0,
    alignment_penalty: float = 3.0,
    compliance_floor: float = 1.0e-10,
    energy_ridge: float = 1.0e-10,
) -> CrossTangentialComparison:
    """Build one nodal library and solve the four nested interface spaces."""
    full_basis = FullNodalStokesBasis(partition)
    full_library = build_response_library(
        partition,
        full_basis,
        viscosity=viscosity,
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure,
        pressure_stabilization=pressure_stabilization,
    )
    skeletons = build_cross_skeletons(
        partition, full_basis, alignment_penalty=alignment_penalty
    )
    normal_transforms, normal_diagnostics = build_minimum_energy_extensions(
        partition,
        full_library,
        skeletons,
        compliance_floor=compliance_floor,
        energy_ridge=energy_ridge,
    )
    vector_transforms, vector_diagnostics = (
        build_vector_minimum_energy_extensions(
            partition,
            full_library,
            skeletons,
            compliance_floor=compliance_floor,
            energy_ridge=energy_ridge,
        )
    )

    # All four online spaces reuse the same primitive load library and local
    # factorisations.  The full primitive responses and G blocks are no longer
    # needed after both extension families have been constructed.
    for entry in full_library.entries:
        entry.primitive_responses = np.empty((entry.operator.ndofs, 0))
        entry.primitive_G = np.empty((0, 0))

    ddpnm_basis = ConstantInterfaceStokesBasis(full_basis, ("normal",))
    ddpnmt_basis = ConstantInterfaceStokesBasis(full_basis, COMPONENTS)
    cross_normal_basis = SkeletonStokesBasis(
        full_basis, skeletons, normal_transforms
    )
    cross_vector_basis = VectorSkeletonStokesBasis(
        full_basis, skeletons, vector_transforms
    )
    return CrossTangentialComparison(
        ddpnm=assemble_skeleton_ddpnm(full_library, ddpnm_basis),
        ddpnmt=assemble_skeleton_ddpnm(full_library, ddpnmt_basis),
        cross_normal=assemble_skeleton_ddpnm(full_library, cross_normal_basis),
        cross_ddpnmt=assemble_skeleton_ddpnm(full_library, cross_vector_basis),
        skeletons=skeletons,
        normal_transforms=normal_transforms,
        vector_transforms=vector_transforms,
        normal_diagnostics=normal_diagnostics,
        vector_diagnostics=vector_diagnostics,
    )


def solve_cross_uniform_tangential_comparison(
    partition,
    viscosity: float = 1.0,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
    pressure_stabilization: float = 0.0,
    alignment_penalty: float = 3.0,
    sampling_factor: float = 1.0,
    uniform_affine_complete: bool = True,
    compliance_floor: float = 1.0e-10,
    energy_ridge: float = 1.0e-10,
) -> CrossUniformTangentialComparison:
    """Compare constants, the cross, and uniform FPS points on one mesh.

    All six spaces reuse one full nodal traction-response library.  Thus the
    reported differences are caused only by the interface reduction space.
    """
    full_basis = FullNodalStokesBasis(partition)
    full_library = build_response_library(
        partition,
        full_basis,
        viscosity=viscosity,
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure,
        pressure_stabilization=pressure_stabilization,
    )
    skeletons = build_cross_skeletons(
        partition, full_basis, alignment_penalty=alignment_penalty
    )
    uniform_sets = build_uniform_control_sets(
        partition, full_basis, sampling_factor=sampling_factor
    )

    cross_normal_transforms, cross_normal_diagnostics = (
        build_minimum_energy_extensions(
            partition,
            full_library,
            skeletons,
            compliance_floor=compliance_floor,
            energy_ridge=energy_ridge,
        )
    )
    cross_vector_transforms, cross_vector_diagnostics = (
        build_vector_minimum_energy_extensions(
            partition,
            full_library,
            skeletons,
            compliance_floor=compliance_floor,
            energy_ridge=energy_ridge,
        )
    )
    uniform_normal_transforms, uniform_normal_diagnostics = (
        build_minimum_energy_extensions(
            partition,
            full_library,
            uniform_sets,
            compliance_floor=compliance_floor,
            energy_ridge=energy_ridge,
            affine_complete=uniform_affine_complete,
        )
    )
    uniform_vector_transforms, uniform_vector_diagnostics = (
        build_vector_minimum_energy_extensions(
            partition,
            full_library,
            uniform_sets,
            compliance_floor=compliance_floor,
            energy_ridge=energy_ridge,
            affine_complete=uniform_affine_complete,
        )
    )

    for entry in full_library.entries:
        entry.primitive_responses = np.empty((entry.operator.ndofs, 0))
        entry.primitive_G = np.empty((0, 0))

    ddpnm_basis = ConstantInterfaceStokesBasis(full_basis, ("normal",))
    ddpnmt_basis = ConstantInterfaceStokesBasis(full_basis, COMPONENTS)
    cross_normal_basis = SkeletonStokesBasis(
        full_basis, skeletons, cross_normal_transforms
    )
    cross_vector_basis = VectorSkeletonStokesBasis(
        full_basis, skeletons, cross_vector_transforms
    )
    uniform_normal_basis = UniformPointStokesBasis(
        full_basis, uniform_sets, uniform_normal_transforms
    )
    uniform_vector_basis = UniformVectorPointStokesBasis(
        full_basis, uniform_sets, uniform_vector_transforms
    )
    return CrossUniformTangentialComparison(
        ddpnm=assemble_skeleton_ddpnm(full_library, ddpnm_basis),
        ddpnmt=assemble_skeleton_ddpnm(full_library, ddpnmt_basis),
        cross_normal=assemble_skeleton_ddpnm(full_library, cross_normal_basis),
        cross_ddpnmt=assemble_skeleton_ddpnm(full_library, cross_vector_basis),
        uniform_normal=assemble_skeleton_ddpnm(
            full_library, uniform_normal_basis
        ),
        uniform_ddpnmt=assemble_skeleton_ddpnm(
            full_library, uniform_vector_basis
        ),
        skeletons=skeletons,
        uniform_sets=uniform_sets,
        cross_normal_transforms=cross_normal_transforms,
        cross_vector_transforms=cross_vector_transforms,
        uniform_normal_transforms=uniform_normal_transforms,
        uniform_vector_transforms=uniform_vector_transforms,
        cross_normal_diagnostics=cross_normal_diagnostics,
        cross_vector_diagnostics=cross_vector_diagnostics,
        uniform_normal_diagnostics=uniform_normal_diagnostics,
        uniform_vector_diagnostics=uniform_vector_diagnostics,
    )

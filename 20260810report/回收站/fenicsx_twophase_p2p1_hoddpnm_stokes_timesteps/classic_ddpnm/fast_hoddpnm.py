from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Network
from .local_stokes_ho import canonical_tangent_axes


@dataclass
class FastHODof:
    interface_id: int
    node_key: tuple[float, float]
    index: int
    point: tuple[float, float, float]


@dataclass
class FastHOResult:
    pressures: np.ndarray
    G: np.ndarray
    rhs: np.ndarray
    dofs: list[FastHODof]


def solve_fast_normal_ho_network(
    network: Network,
    throat_radius: float = 0.13,
    node_span_factor: float = 0.9,
    boundary_length_floor: float = 0.12,
) -> FastHOResult:
    """Fast normal-only HODDPNM with 3x3 interface nodal pressures.

    This keeps the HODDPNM global unknown structure on interface faces while
    replacing each expensive local Stokes response with a pore-pressure
    Schur complement: q_i = c_i (p_i - p_pore), sum_i q_i = 0.
    """
    centers = np.asarray([p.center for p in network.pores], dtype=float)
    node_params, node_weights = reference_disk_nodes(throat_radius * node_span_factor)
    interface_points = make_interface_node_points(network, node_params)

    dofs: list[FastHODof] = []
    dof_map: dict[tuple[int, int], int] = {}
    for interface_id, points in enumerate(interface_points):
        for node_index, point in enumerate(points):
            dof_map[(interface_id, node_index)] = len(dofs)
            dofs.append(
                FastHODof(
                    interface_id=interface_id,
                    node_key=(float(node_params[node_index, 0]), float(node_params[node_index, 1])),
                    index=len(dofs),
                    point=(float(point[0]), float(point[1]), float(point[2])),
                )
            )

    n_unknowns = len(dofs)
    G = np.zeros((n_unknowns, n_unknowns), dtype=float)
    rhs = np.zeros(n_unknowns, dtype=float)
    interface_lookup = {tuple(sorted(edge)): idx for idx, edge in enumerate(network.interfaces)}

    for pore in network.pores:
        entries: list[tuple[int | None, float | None, float]] = []
        pore_center = np.asarray(pore.center, dtype=float)
        for port in pore.ports:
            if port.kind == "interface":
                interface_id = interface_lookup[tuple(sorted((pore.id, int(port.neighbor))))]
                a, b = network.interfaces[interface_id]
                midpoint = 0.5 * (centers[a] + centers[b])
                length = max(float(np.linalg.norm(midpoint - pore_center)), boundary_length_floor)
                for node_index, weight in enumerate(node_weights):
                    entries.append((dof_map[(interface_id, node_index)], None, float(weight / length)))
            elif port.kind in {"inlet", "outlet"}:
                length = max(distance_to_x_boundary(pore_center, port.kind), boundary_length_floor)
                for weight in node_weights:
                    entries.append((None, float(port.pressure), float(weight / length)))

        if not entries:
            continue
        conductances = np.asarray([item[2] for item in entries], dtype=float)
        denom = float(conductances.sum())
        local = np.diag(conductances) - np.outer(conductances, conductances) / denom

        for i, (row_gid, _, _) in enumerate(entries):
            if row_gid is None:
                continue
            for j, (col_gid, known_pressure, _) in enumerate(entries):
                value = local[i, j]
                if col_gid is None:
                    rhs[row_gid] -= value * float(known_pressure)
                else:
                    G[row_gid, col_gid] += value

    pressures = np.linalg.solve(G, rhs)
    return FastHOResult(pressures=pressures, G=G, rhs=rhs, dofs=dofs)


def reference_disk_nodes(node_span: float) -> tuple[np.ndarray, np.ndarray]:
    coords_1d = np.asarray([-node_span, 0.0, node_span], dtype=float)
    simpson_1d = np.asarray([1.0, 4.0, 1.0], dtype=float)
    params = []
    weights = []
    for i, s in enumerate(coords_1d):
        for j, t in enumerate(coords_1d):
            params.append((s, t))
            weights.append(simpson_1d[i] * simpson_1d[j])
    params_array = np.asarray(params, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    weights_array *= np.pi * (node_span / 0.9) ** 2 / weights_array.sum()
    return params_array, weights_array


def make_interface_node_points(network: Network, params: np.ndarray) -> list[np.ndarray]:
    centers = np.asarray([p.center for p in network.pores], dtype=float)
    points: list[np.ndarray] = []
    for a, b in network.interfaces:
        midpoint = 0.5 * (centers[a] + centers[b])
        direction = centers[b] - centers[a]
        direction = direction / np.linalg.norm(direction)
        e1, e2 = canonical_tangent_axes(direction)
        points.append(midpoint + params[:, :1] * e1 + params[:, 1:] * e2)
    return points


def distance_to_x_boundary(center: np.ndarray, kind: str, domain_size: float = 5.0) -> float:
    if kind == "inlet":
        return float(center[0])
    return float(domain_size - center[0])

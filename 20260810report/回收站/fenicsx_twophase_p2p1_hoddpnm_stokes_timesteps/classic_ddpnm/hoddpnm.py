from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Network
from .local_stokes_ho import LocalHOResponse, solve_local_ho_responses


@dataclass
class HODof:
    interface_id: int
    node_key: tuple[float, float]
    index: int


@dataclass
class HODdpnmResult:
    pressures: np.ndarray
    G: np.ndarray
    rhs: np.ndarray
    local_responses: list[LocalHOResponse]
    dofs: list[HODof]
    node_counts_by_interface: dict[int, int]


def solve_ho_network(
    network: Network,
    h: float = 0.28,
    port_half_width: float = 0.55,
    node_match_tol: float = 0.075,
    global_stabilization: float = 1.0e-10,
) -> HODdpnmResult:
    local = [
        solve_local_ho_responses(pore, h=h, port_half_width=port_half_width)
        for pore in network.pores
    ]

    interface_nodes = build_interface_node_clusters(local, node_match_tol=node_match_tol)
    dof_map: dict[tuple[int, int], int] = {}
    dofs: list[HODof] = []
    for interface_id in sorted(interface_nodes):
        for node_index, param in enumerate(interface_nodes[interface_id]):
            dof_map[(interface_id, node_index)] = len(dofs)
            dofs.append(
                HODof(
                    interface_id=interface_id,
                    node_key=rounded_param_key(param),
                    index=len(dofs),
                )
            )

    n_unknowns = len(dofs)
    G = np.zeros((n_unknowns, n_unknowns), dtype=float)
    rhs = np.zeros(n_unknowns, dtype=float)

    for response in local:
        row_entries = basis_entries(response, interface_nodes)
        col_entries = row_entries
        for row_local, row_entry in enumerate(row_entries):
            row_gid = global_or_known(row_entry, dof_map)
            if row_gid is None:
                continue
            for col_local, col_entry in enumerate(col_entries):
                value = response.G[row_local, col_local]
                col_gid = global_or_known(col_entry, dof_map)
                if col_gid is None:
                    known = col_entry["pressure"]
                    if known is None:
                        raise ValueError("Boundary high-order node is missing known pressure.")
                    rhs[row_gid] -= value * float(known)
                else:
                    G[row_gid, col_gid] += value

    solve_matrix = G.copy()
    if global_stabilization > 0.0 and n_unknowns > 0:
        scale = max(float(np.linalg.norm(G, ord=np.inf)), 1.0)
        solve_matrix = solve_matrix + global_stabilization * scale * np.eye(n_unknowns)
    try:
        pressures = np.linalg.solve(solve_matrix, rhs)
    except np.linalg.LinAlgError:
        pressures = np.linalg.lstsq(solve_matrix, rhs, rcond=1.0e-12)[0]
    node_counts: dict[int, int] = {}
    for dof in dofs:
        node_counts[dof.interface_id] = node_counts.get(dof.interface_id, 0) + 1

    return HODdpnmResult(
        pressures=pressures,
        G=G,
        rhs=rhs,
        local_responses=local,
        dofs=dofs,
        node_counts_by_interface=node_counts,
    )


def build_interface_node_clusters(
    local: list[LocalHOResponse],
    node_match_tol: float,
) -> dict[int, np.ndarray]:
    points_by_interface: dict[int, list[np.ndarray]] = {}
    for response in local:
        for nodes in response.port_nodes:
            if nodes.port_id >= 0:
                points_by_interface.setdefault(nodes.port_id, []).extend(nodes.params)

    clusters: dict[int, np.ndarray] = {}
    for interface_id, point_list in points_by_interface.items():
        points = np.asarray(point_list, dtype=float)
        centers: list[np.ndarray] = []
        for point in points[np.lexsort((points[:, 1], points[:, 0]))]:
            if not centers:
                centers.append(point.copy())
                continue
            distances = np.asarray([np.linalg.norm(point - center) for center in centers])
            best = int(np.argmin(distances))
            if distances[best] <= node_match_tol:
                centers[best] = 0.5 * (centers[best] + point)
            else:
                centers.append(point.copy())
        clusters[interface_id] = np.asarray(
            sorted(centers, key=lambda x: (round(float(x[0]), 10), round(float(x[1]), 10))),
            dtype=float,
        )
    return clusters


def basis_entries(
    response: LocalHOResponse,
    interface_nodes: dict[int, np.ndarray],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for nodes in response.port_nodes:
        for param in nodes.params:
            node_index = None
            if nodes.port_id >= 0:
                node_index = nearest_interface_node(interface_nodes[nodes.port_id], param)
            entries.append(
                {
                    "port_id": nodes.port_id,
                    "node_index": node_index,
                    "pressure": nodes.pressure,
                }
            )
    return entries


def global_or_known(
    entry: dict[str, object],
    dof_map: dict[tuple[int, int], int],
) -> int | None:
    port_id = int(entry["port_id"])
    if port_id < 0:
        return None
    return dof_map[(port_id, int(entry["node_index"]))]


def nearest_interface_node(nodes: np.ndarray, param: np.ndarray) -> int:
    distances = np.linalg.norm(nodes - param, axis=1)
    return int(np.argmin(distances))


def rounded_param_key(param: np.ndarray) -> tuple[float, float]:
    return (float(np.round(param[0], 10)), float(np.round(param[1], 10)))

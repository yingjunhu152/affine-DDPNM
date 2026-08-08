from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, distance_matrix


WALL_TAG = 1
PORT_TAG_BASE = 100


@dataclass(frozen=True)
class Port:
    id: int
    neighbor: int | None
    kind: str
    normal: tuple[float, float, float]
    pressure: float | None = None


@dataclass(frozen=True)
class Pore:
    id: int
    ijk: tuple[int, int, int]
    center: tuple[float, float, float]
    radius: float
    ports: tuple[Port, ...]


@dataclass(frozen=True)
class Network:
    pores: tuple[Pore, ...]
    interfaces: tuple[tuple[int, int], ...]
    inlet_pressure: float
    outlet_pressure: float


def make_regular_network(
    n: int = 5,
    domain_size: float = 5.0,
    radius: float = 0.23,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
) -> Network:
    """Build a regular n by n by n pore network.

    The pores are non-overlapping balls. Interior interfaces are between
    nearest neighbors, with interface normals aligned with center-to-center
    directions. External inlet/outlet ports live on the x-min/x-max sides
    with a tiny eccentric tilt so the geometry is not perfectly degenerate.
    """
    margin = 0.55
    grid = np.linspace(margin, domain_size - margin, n)

    def pid(i: int, j: int, k: int) -> int:
        return (i * n + j) * n + k

    centers = {
        (i, j, k): (float(grid[k]), float(grid[j]), float(grid[i]))
        for i in range(n)
        for j in range(n)
        for k in range(n)
    }
    interfaces: list[tuple[int, int]] = []
    port_lists: dict[int, list[Port]] = {
        pid(i, j, k): [] for i in range(n) for j in range(n) for k in range(n)
    }

    next_unknown = 0

    for i in range(n):
        for j in range(n):
            for k in range(n):
                a = pid(i, j, k)
                if k + 1 < n:
                    b = pid(i, j, k + 1)
                    interfaces.append((a, b))
                    add_pair_ports(port_lists, centers, (i, j, k), (i, j, k + 1), next_unknown, n)
                    next_unknown += 1
                if j + 1 < n:
                    b = pid(i, j + 1, k)
                    interfaces.append((a, b))
                    add_pair_ports(port_lists, centers, (i, j, k), (i, j + 1, k), next_unknown, n)
                    next_unknown += 1
                if i + 1 < n:
                    b = pid(i + 1, j, k)
                    interfaces.append((a, b))
                    add_pair_ports(port_lists, centers, (i, j, k), (i + 1, j, k), next_unknown, n)
                    next_unknown += 1

    for i in range(n):
        for j in range(n):
            tilt = np.array([0.0, 0.045 * ((i % 2) - 0.5), 0.045 * ((j % 2) - 0.5)])
            left = pid(i, j, 0)
            right = pid(i, j, n - 1)
            port_lists[left].append(
                make_port(
                    id=-1,
                    neighbor=None,
                    kind="inlet",
                    normal=normalize(np.array([-1.0, 0.0, 0.0]) + tilt),
                    pressure=inlet_pressure,
                )
            )
            port_lists[right].append(
                make_port(
                    id=-2,
                    neighbor=None,
                    kind="outlet",
                    normal=normalize(np.array([1.0, 0.0, 0.0]) - tilt),
                    pressure=outlet_pressure,
                )
            )

    pores = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                a = pid(i, j, k)
                ports = tuple(sorted(port_lists[a], key=sort_key))
                pores.append(
                    Pore(
                        id=a,
                        ijk=(i, j, k),
                        center=centers[(i, j, k)],
                        radius=radius,
                        ports=ports,
                    )
                )

    return Network(
        pores=tuple(pores),
        interfaces=tuple(interfaces),
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure,
    )


def make_irregular_network(
    n_pores: int = 125,
    domain_size: float = 5.0,
    radius: float = 0.17,
    min_gap: float = 0.08,
    seed: int = 7,
    target_degree: int = 5,
    max_edge_length: float = 1.18,
    n_boundary_ports: int = 18,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
) -> Network:
    """Build an irregular 3D PNM-style network of non-overlapping balls."""
    rng = np.random.default_rng(seed)
    centers = generate_nonoverlapping_centers(
        n_pores=n_pores,
        domain_size=domain_size,
        radius=radius,
        min_gap=min_gap,
        rng=rng,
    )
    edges = select_pnm_edges(
        centers,
        target_degree=target_degree,
        max_edge_length=max_edge_length,
    )

    port_lists: dict[int, list[Port]] = {i: [] for i in range(n_pores)}
    interfaces: list[tuple[int, int]] = []
    for interface_id, (a, b) in enumerate(edges):
        interfaces.append((a, b))
        normal_ab = normalize(centers[b] - centers[a])
        port_lists[a].append(make_port(interface_id, b, "interface", normal_ab))
        port_lists[b].append(make_port(interface_id, a, "interface", -normal_ab))

    inlet_ids = np.argsort(centers[:, 0])[: min(n_boundary_ports, n_pores)]
    outlet_ids = np.argsort(-centers[:, 0])[: min(n_boundary_ports, n_pores)]

    for pore_id in inlet_ids:
        direction = normalize(
            np.array(
                [
                    -1.0,
                    0.10 * (centers[pore_id, 1] - domain_size / 2.0),
                    0.10 * (centers[pore_id, 2] - domain_size / 2.0),
                ]
            )
        )
        port_lists[int(pore_id)].append(
            make_port(-1, None, "inlet", direction, pressure=inlet_pressure)
        )

    for pore_id in outlet_ids:
        direction = normalize(
            np.array(
                [
                    1.0,
                    0.10 * (centers[pore_id, 1] - domain_size / 2.0),
                    0.10 * (centers[pore_id, 2] - domain_size / 2.0),
                ]
            )
        )
        port_lists[int(pore_id)].append(
            make_port(-2, None, "outlet", direction, pressure=outlet_pressure)
        )

    pores = tuple(
        Pore(
            id=i,
            ijk=(-1, -1, -1),
            center=tuple(float(x) for x in centers[i]),
            radius=radius,
            ports=tuple(sorted(port_lists[i], key=sort_key)),
        )
        for i in range(n_pores)
    )
    return Network(
        pores=pores,
        interfaces=tuple(interfaces),
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure,
    )


def generate_nonoverlapping_centers(
    n_pores: int,
    domain_size: float,
    radius: float,
    min_gap: float,
    rng: np.random.Generator,
) -> np.ndarray:
    margin = radius + 0.35
    min_dist = 2.0 * radius + min_gap
    centers: list[np.ndarray] = []
    attempts = 0
    while len(centers) < n_pores and attempts < 250000:
        attempts += 1
        candidate = rng.uniform(margin, domain_size - margin, size=3)
        if all(np.linalg.norm(candidate - center) >= min_dist for center in centers):
            centers.append(candidate)
    if len(centers) != n_pores:
        raise RuntimeError(f"Could only place {len(centers)} pores out of {n_pores}.")
    return np.asarray(centers, dtype=float)


def select_pnm_edges(
    centers: np.ndarray,
    target_degree: int,
    max_edge_length: float,
) -> tuple[tuple[int, int], ...]:
    edge_set: set[tuple[int, int]] = set()
    delaunay = Delaunay(centers)
    for simplex in delaunay.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                a, b = sorted((int(simplex[i]), int(simplex[j])))
                if np.linalg.norm(centers[a] - centers[b]) <= max_edge_length:
                    edge_set.add((a, b))

    distances = distance_matrix(centers, centers)
    np.fill_diagonal(distances, np.inf)
    for a in range(len(centers)):
        for b in np.argsort(distances[a])[:target_degree]:
            if distances[a, b] <= max_edge_length * 1.18:
                edge_set.add(tuple(sorted((a, int(b)))))

    return tuple(sorted(connect_components(edge_set, distances)))


def connect_components(
    edge_set: set[tuple[int, int]],
    distances: np.ndarray,
) -> set[tuple[int, int]]:
    parent = list(range(distances.shape[0]))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edge_set:
        union(a, b)

    while len({find(i) for i in range(len(parent))}) > 1:
        comps: dict[int, list[int]] = {}
        for i in range(len(parent)):
            comps.setdefault(find(i), []).append(i)
        comp_ids = list(comps)
        best: tuple[float, int, int] | None = None
        for ci in range(len(comp_ids)):
            for cj in range(ci + 1, len(comp_ids)):
                for a in comps[comp_ids[ci]]:
                    b = min(comps[comp_ids[cj]], key=lambda x: distances[a, x])
                    item = (float(distances[a, b]), a, b)
                    if best is None or item < best:
                        best = item
        if best is None:
            break
        _, a, b = best
        edge_set.add(tuple(sorted((a, b))))
        union(a, b)
    return edge_set


def add_pair_ports(
    port_lists: dict[int, list[Port]],
    centers: dict[tuple[int, int, int], tuple[float, float, float]],
    ia: tuple[int, int, int],
    ib: tuple[int, int, int],
    interface_id: int,
    n: int,
) -> None:
    aid = (ia[0] * n + ia[1]) * n + ia[2]
    bid = (ib[0] * n + ib[1]) * n + ib[2]
    normal_ab = normalize(np.asarray(centers[ib]) - np.asarray(centers[ia]))
    port_lists[aid].append(make_port(interface_id, bid, "interface", normal_ab))
    port_lists[bid].append(make_port(interface_id, aid, "interface", -normal_ab))


def make_port(
    id: int,
    neighbor: int | None,
    kind: str,
    normal: np.ndarray,
    pressure: float | None = None,
) -> Port:
    return Port(
        id=id,
        neighbor=neighbor,
        kind=kind,
        normal=tuple(float(x) for x in normalize(normal)),
        pressure=pressure,
    )


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector.")
    return v / norm


def sort_key(port: Port) -> tuple[float, float, float]:
    return port.normal


def spherical_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0))
    return float(np.arccos(dot))

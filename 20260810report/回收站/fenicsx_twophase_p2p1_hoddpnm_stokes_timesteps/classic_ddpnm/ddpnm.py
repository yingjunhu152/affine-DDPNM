from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Network
from .local_stokes import LocalResponse, solve_local_responses


@dataclass
class DdpnmResult:
    pressures: np.ndarray
    G: np.ndarray
    rhs: np.ndarray
    local_responses: list[LocalResponse]


def solve_network(network: Network, h: float = 0.09, port_half_width: float = 0.18) -> DdpnmResult:
    local = [
        solve_local_responses(pore, h=h, port_half_width=port_half_width)
        for pore in network.pores
    ]

    n_unknowns = len(network.interfaces)
    G = np.zeros((n_unknowns, n_unknowns), dtype=float)
    rhs = np.zeros(n_unknowns, dtype=float)

    for response in local:
        for row_local, row_id in enumerate(response.port_ids):
            if row_id < 0:
                continue
            for col_local, col_id in enumerate(response.port_ids):
                value = response.G[row_local, col_local]
                if col_id >= 0:
                    G[row_id, col_id] += value
                else:
                    p_known = response.port_pressures[col_local]
                    if p_known is None:
                        raise ValueError(f"Known boundary port {col_id} has no pressure.")
                    rhs[row_id] -= value * p_known

    pressures = np.linalg.solve(G, rhs)
    return DdpnmResult(pressures=pressures, G=G, rhs=rhs, local_responses=local)


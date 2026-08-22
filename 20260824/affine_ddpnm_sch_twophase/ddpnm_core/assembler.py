"""Interface assembler: local G matrices → global Schur system.

The :class:`InterfaceAssembler` is the dimension-agnostic merge of the four
Schur-assembly routines currently duplicated across 2D/3D ``solver.py`` and
``hierarchy.py``.  It uses a single sign convention (the hierarchy's
``S += G[uu]; rhs -= G[uk] @ p_known``) and reconstructs local solutions and
per-mode moment residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from ddpnm_core.library import ResponseLibrary


@dataclass
class SchurSystem:
    """The result of assembling and solving the global Schur complement."""

    levels: np.ndarray
    global_keys: tuple[tuple, ...]
    coefficients: np.ndarray
    schur_matrix: np.ndarray
    rhs: np.ndarray
    local_solutions: list[np.ndarray]
    moment_residuals: np.ndarray  # per global key
    boundary_fluxes: dict[str, float]
    min_schur_eigenvalue: float
    symmetry_error: float
    relative_linear_residual: float


class InterfaceAssembler:
    """Assemble and solve the condensed interface system for a given level vector.

    Parameters
    ----------
    library: ResponseLibrary
        Pre-built primitive response library.
    """

    def __init__(self, library: ResponseLibrary):
        self._library = library
        self._basis = library.basis

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def keys(self, levels: np.ndarray) -> tuple[tuple, ...]:
        """Ordered global keys for *levels*."""
        ninterfaces = len(self._library.partition.interface_pairs)
        levels_arr = np.asarray(levels, dtype=np.int8)
        if levels_arr.shape != (ninterfaces,):
            raise ValueError(
                f"Expected {ninterfaces} interface levels, got {levels_arr.shape}."
            )
        return self._global_keys(levels_arr)

    def assemble(
        self, levels: np.ndarray | list[int] | tuple[int, ...],
        compute_min_eigenvalue: bool = True,
    ) -> SchurSystem:
        """Assemble, symmetrize, solve and reconstruct.

        ``compute_min_eigenvalue=False`` skips the dense ``eigvalsh`` of the
        Schur matrix (O(n^3), diagnostic only) — the two-phase SFI loops call
        this once per iterate, where the eigendecomposition dominates the
        wall time without affecting the solution.
        """
        levels_arr = np.asarray(levels, dtype=np.int8).copy()
        ninterfaces = len(self._library.partition.interface_pairs)
        if levels_arr.shape != (ninterfaces,):
            raise ValueError(
                f"Expected {ninterfaces} interface levels, got {levels_arr.shape}."
            )

        keys = self._global_keys(levels_arr)
        key_to_dof = {key: dof for dof, key in enumerate(keys)}
        n_global = len(keys)

        # Sparse Schur assembly: accumulate (row, col, value) triples
        coo_rows: list[int] = []
        coo_cols: list[int] = []
        coo_vals: list[float] = []
        rhs_vec = np.zeros(n_global, dtype=float)
        rec_data: list[dict] = []

        for entry in self._library.entries:
            primitives = entry.primitive_modes
            ports = entry.operator.ports

            # 1. Collect active primitive indices per port and compute transforms
            port_active_indices: list[tuple[int, ...]] = []
            port_transforms: list[np.ndarray | None] = []
            all_active: list[int] = []

            for port_index, port in enumerate(ports):
                lvl = (
                    int(levels_arr[int(port.global_interface)])
                    if port.kind == "interface" else 0
                )
                indices = self._basis.active_indices(primitives, port_index, lvl)
                port_active_indices.append(indices)
                all_active.extend(indices)
                port_transforms.append(
                    self._basis.active_transform(primitives, port_index, lvl)
                )

            if not all_active:
                rec_data.append({"responses": None, "active_info": [], "G": None})
                continue

            active_arr = np.asarray(all_active, dtype=np.int32)

            # 2. Build per-port active-mode → global-key mapping
            active_info: list[dict] = []  # {key, known, dof, port_index}
            for port_index, (port, indices, transform) in enumerate(
                zip(ports, port_active_indices, port_transforms, strict=True)
            ):
                if port.kind == "interface":
                    lvl = int(levels_arr[int(port.global_interface)])
                    iid = int(port.global_interface)
                    port_keys = list(self._basis.global_keys(lvl, iid))
                else:
                    port_keys = [None]  # single boundary mode

                if transform is not None:
                    # transform maps primitives (n_p) → active (n_a)
                    n_active = transform.shape[1]
                    for j in range(n_active):
                        key = port_keys[j] if j < len(port_keys) else None
                        if key is not None and key in key_to_dof:
                            active_info.append({
                                "key": key, "known": None,
                                "dof": key_to_dof[key], "port_index": port_index,
                            })
                        else:
                            active_info.append({
                                "key": None,
                                "known": float(port.pressure) if port.kind != "interface" else None,
                                "dof": -1, "port_index": port_index,
                            })
                else:
                    # Each active primitive maps 1:1 to a global key
                    for j, prim_idx in enumerate(indices):
                        mode = primitives[prim_idx]
                        if mode.interface_id is not None:
                            key = (int(mode.interface_id), mode.component, mode.polynomial)
                            if key in key_to_dof:
                                active_info.append({
                                    "key": key, "known": None,
                                    "dof": key_to_dof[key], "port_index": port_index,
                                })
                            else:
                                # Inactive mode at this level — shouldn't happen
                                active_info.append({
                                    "key": key, "known": None, "dof": -1, "port_index": port_index,
                                })
                        else:
                            active_info.append({
                                "key": None,
                                "known": float(mode.known_coefficient),
                                "dof": -1, "port_index": port_index,
                            })

            # 3. Apply transforms to get reduced G and response matrix
            # Build block-diagonal transform if any port has one
            has_transform = any(t is not None for t in port_transforms)
            if has_transform:
                transform_full = _block_diag(port_transforms, port_active_indices,
                                             len(primitives))
                G = transform_full.T @ entry.primitive_G[np.ix_(active_arr, active_arr)] @ transform_full
                responses = entry.primitive_responses[:, active_arr] @ transform_full
            else:
                G = entry.primitive_G[np.ix_(active_arr, active_arr)]
                responses = entry.primitive_responses[:, active_arr]

            # 4. Assemble into S and rhs
            unknown = [i for i, am in enumerate(active_info) if am["dof"] >= 0]
            known = [i for i, am in enumerate(active_info) if am["dof"] < 0 and am["known"] is not None]
            dofs = [active_info[i]["dof"] for i in unknown]

            if unknown:
                G_uu = G[np.ix_(unknown, unknown)]
                for ui, di in enumerate(dofs):
                    for uj, dj in enumerate(dofs):
                        val = float(G_uu[ui, uj])
                        if abs(val) > 1e-30:
                            coo_rows.append(di); coo_cols.append(dj); coo_vals.append(val)
                if known:
                    known_vals = np.asarray([float(active_info[i]["known"]) for i in known])
                    rhs_vec[dofs] -= G[np.ix_(unknown, known)] @ known_vals

            rec_data.append({"responses": responses, "active_info": active_info, "G": G})

        # 5. Build sparse Schur, symmetrize and solve
        if n_global > 0 and coo_rows:
            S_sparse = csr_matrix((coo_vals, (coo_rows, coo_cols)),
                                  shape=(n_global, n_global))
            S_sparse.sum_duplicates()
            S_sparse = 0.5 * (S_sparse + S_sparse.T)
            coefficients = spsolve(S_sparse, rhs_vec)
            residual = S_sparse @ coefficients - rhs_vec
            rel_res = float(
                np.linalg.norm(residual) / max(np.linalg.norm(rhs_vec), 1.0e-30)
            )
            if compute_min_eigenvalue:
                S = S_sparse.toarray()  # dense, diagnostic only
                min_eig = float(np.linalg.eigvalsh(S)[0])
            else:
                S = S_sparse.toarray()
                min_eig = float("nan")
        elif n_global == 0:
            coefficients = np.empty(0)
            min_eig = float("nan")
            rel_res = 0.0
            S = np.empty((0, 0))
        else:
            # No nonzeros — degenerate case
            S = np.zeros((n_global, n_global))
            coefficients = np.zeros(n_global)
            min_eig = 0.0
            rel_res = 0.0

        # 6. Reconstruct local solutions and accumulate residuals
        local_solutions: list[np.ndarray] = []
        moment_residuals = np.zeros(n_global, dtype=float)
        boundary_fluxes: dict[str, float] = {"inlet": 0.0, "outlet": 0.0}

        for entry, rd in zip(self._library.entries, rec_data, strict=True):
            if rd["responses"] is None or len(rd["active_info"]) == 0:
                local_solutions.append(np.zeros(entry.operator.ndofs))
                continue

            local_coeffs = np.empty(len(rd["active_info"]), dtype=float)
            for i, am in enumerate(rd["active_info"]):
                if am["dof"] >= 0:
                    local_coeffs[i] = coefficients[am["dof"]]
                elif am["known"] is not None:
                    local_coeffs[i] = float(am["known"])
                else:
                    local_coeffs[i] = 0.0

            local_solutions.append(rd["responses"] @ local_coeffs)
            moments = rd["G"] @ local_coeffs

            for i, am in enumerate(rd["active_info"]):
                if am["dof"] >= 0:
                    moment_residuals[am["dof"]] += moments[i]
                elif am.get("port_index") is not None:
                    port = entry.operator.ports[am["port_index"]]
                    if port.kind in boundary_fluxes:
                        boundary_fluxes[port.kind] -= float(moments[i])

        return SchurSystem(
            levels=levels_arr,
            global_keys=keys,
            coefficients=coefficients,
            schur_matrix=S,
            rhs=rhs_vec,
            local_solutions=local_solutions,
            moment_residuals=moment_residuals,
            boundary_fluxes=boundary_fluxes,
            min_schur_eigenvalue=float(min_eig) if n_global else float("nan"),
            symmetry_error=float(np.linalg.norm(S - S.T) / max(np.linalg.norm(S), 1e-30)),
            relative_linear_residual=rel_res,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _global_keys(self, levels_arr: np.ndarray) -> tuple[tuple, ...]:
        all_keys: list[tuple] = []
        for interface_id, level in enumerate(levels_arr):
            all_keys.extend(self._basis.global_keys(int(level), interface_id))
        return tuple(all_keys)


def _block_diag(transforms, active_indices_per_port, n_total_prim):
    """Build a block-diagonal transform matrix.

    Each port contributes a transform T_i of shape (n_prim_i, n_active_i).
    The output is (n_total_active_prims, n_total_active) where
    n_total_active_prims is the total number of active primitives across all ports.
    """
    # Count total active primitives and total active modes
    total_prim = sum(len(indices) for indices in active_indices_per_port)
    total_active = sum(
        (t.shape[1] if t is not None else len(indices))
        for t, indices in zip(transforms, active_indices_per_port, strict=True)
    )
    result = np.zeros((total_prim, total_active))
    prim_offset = 0
    active_offset = 0
    for indices, transform in zip(active_indices_per_port, transforms, strict=True):
        n_prim = len(indices)
        if transform is not None:
            n_active = transform.shape[1]
            result[prim_offset:prim_offset + n_prim, active_offset:active_offset + n_active] = transform
        else:
            n_active = n_prim
            result[prim_offset:prim_offset + n_prim, active_offset:active_offset + n_active] = np.eye(n_prim)
        prim_offset += n_prim
        active_offset += n_active
    return result

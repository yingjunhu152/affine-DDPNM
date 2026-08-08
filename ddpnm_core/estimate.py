"""Residual-based adaptive estimator.

Replaces the old ``_interface_indicators`` (vertex field-difference averaging)
with five direct measurements on each interface:

1. **velocity jump** — ‖u⁺ − u⁻‖_{L²(Γᵢ)}
2. **normal flux residual** — per-interface normal-P0 moment residual
3. **tangential moment residual** — per-interface active tangential moment norm
4. **inactive mode residual** — per-interface unmodelled higher-order moments
5. **area-weighted norm** — each component divided by √|Γᵢ| for dimensionless ηᵢ

All computations are pure NumPy — no new UFL forms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Interface measure (facet length / area)
# ---------------------------------------------------------------------------

def interface_measure(partition) -> np.ndarray:
    """Per-interface facet area (3D) or edge length (2D).

    Computed from mesh geometry by summing the (d-1)-volume of each
    facet belonging to the interface.
    """
    msh = partition.mesh
    gdim = msh.geometry.dim
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    x = msh.geometry.x
    n_interfaces = len(partition.interface_pairs)
    measures = np.zeros(n_interfaces, dtype=float)
    for interface_id in range(n_interfaces):
        facets = np.flatnonzero(partition.facet_interface_ids == interface_id)
        area = 0.0
        for facet in facets:
            vertices = x[f2v.links(int(facet))[:fdim + 1], :gdim]
            if fdim == 1:  # 2D: edge length
                area += float(np.linalg.norm(vertices[1] - vertices[0]))
            elif fdim == 2:  # 3D: triangle area
                ab = vertices[1] - vertices[0]
                ac = vertices[2] - vertices[0]
                area += 0.5 * float(np.linalg.norm(np.cross(ab, ac)))
        measures[interface_id] = area
    return measures


# ---------------------------------------------------------------------------
# 1. Two-sided velocity jump
# ---------------------------------------------------------------------------

def two_sided_velocity_jump(
    library,
    system,
) -> np.ndarray:
    """Per-interface ‖u⁺ − u⁻‖_{L²(Γᵢ)} / √|Γᵢ|.

    Evaluates both incident pore velocities at interface vertices using the
    per-pore P1 reconstruction, then computes the RMS of the difference.
    For P1 elements on straight facets the vertex-based mean is exact to
    quadrature error.
    """
    n_interfaces = len(library.partition.interface_pairs)
    measures = interface_measure(library.partition)
    jumps = np.zeros(n_interfaces, dtype=float)

    # Build per-pore vertex velocities (non-averaged)
    pore_velocity: dict[int, np.ndarray] = {}
    for entry, local_sol in zip(library.entries, system.local_solutions, strict=True):
        pore_id = entry.operator.pore_id
        u_local, _ = _p1_vertex_fields(entry.operator.W, local_sol)
        # Map to parent vertices
        parent = np.asarray(entry.operator.parent_vertex_map, dtype=np.int32)
        n_parent = library.partition.mesh.topology.index_map(0).size_local
        gdim = u_local.shape[1]
        u_parent = np.zeros((n_parent, gdim), dtype=float)
        u_parent[parent] = u_local
        pore_velocity[pore_id] = u_parent

    for interface_id, (pore_a, pore_b) in enumerate(library.partition.interface_pairs):
        if pore_a not in pore_velocity or pore_b not in pore_velocity:
            continue
        vertices = np.asarray(library.interface_nodes[interface_id], dtype=np.int32)
        if len(vertices) == 0:
            continue
        u_a = pore_velocity[pore_a][vertices]
        u_b = pore_velocity[pore_b][vertices]
        diff = u_a - u_b
        # Vertex-mean L² norm, exact for P1 on straight facets
        l2_sq = float(np.mean(np.sum(diff ** 2, axis=1)))
        jumps[interface_id] = np.sqrt(max(l2_sq, 0.0)) / max(
            np.sqrt(measures[interface_id]), 1.0e-30
        )
    return jumps


# ---------------------------------------------------------------------------
# 2. Normal flux residual
# ---------------------------------------------------------------------------

def normal_flux_residual(
    library,
    system,
) -> np.ndarray:
    """Per-interface |normal-P0 moment residual| / √|Γᵢ|."""
    n_interfaces = len(library.partition.interface_pairs)
    measures = interface_measure(library.partition)
    residuals = np.zeros(n_interfaces, dtype=float)
    key_to_dof = {k: d for d, k in enumerate(system.global_keys)}
    for interface_id in range(n_interfaces):
        # Try both 2D and 3D key formats
        for key in (
            (interface_id, "normal", "P0"),
            (interface_id, "normal_constant"),
        ):
            dof = key_to_dof.get(key)
            if dof is not None:
                residuals[interface_id] = float(
                    abs(system.moment_residuals[dof])
                ) / max(np.sqrt(measures[interface_id]), 1.0e-30)
                break
    return residuals


# ---------------------------------------------------------------------------
# 3. Tangential moment residual
# ---------------------------------------------------------------------------

def tangential_moment_residual(
    library,
    system,
) -> np.ndarray:
    """Per-interface norm of active tangential-mode moment residuals / √|Γᵢ|."""
    n_interfaces = len(library.partition.interface_pairs)
    measures = interface_measure(library.partition)
    residuals = np.zeros(n_interfaces, dtype=float)

    # Tangential components by key format
    tang_2d = {"tangent_constant", "tangent_node"}
    tang_3d = {"tangent_1", "tangent_2"}

    key_to_dof = {k: d for d, k in enumerate(system.global_keys)}
    for interface_id in range(n_interfaces):
        sq_sum = 0.0
        for key, dof in key_to_dof.items():
            if key[0] != interface_id:
                continue
            # Check if tangential
            if len(key) >= 2:
                component = key[1] if isinstance(key[1], str) else ""
                if component in tang_2d or component in tang_3d:
                    sq_sum += float(system.moment_residuals[dof]) ** 2
        residuals[interface_id] = np.sqrt(sq_sum) / max(
            np.sqrt(measures[interface_id]), 1.0e-30
        )
    return residuals


# ---------------------------------------------------------------------------
# 4. Inactive higher-order mode residual
# ---------------------------------------------------------------------------

def inactive_mode_residual(
    library,
    system,
) -> np.ndarray:
    """Per-interface norm of *inactive* primitive-mode moment residuals / √|Γᵢ|.

    For each pore we extend the solved active coefficients back to the full
    primitive space, compute ``primitive_G @ full_coeffs``, and accumulate
    the moments for primitives that were *not* active.  No extra solves are
    needed.
    """
    n_interfaces = len(library.partition.interface_pairs)
    measures = interface_measure(library.partition)
    eps = 1.0e-30
    inactive_sq = np.zeros(n_interfaces, dtype=float)

    key_to_dof = {k: d for d, k in enumerate(system.global_keys)}
    basis = library.basis

    for entry, local_sol in zip(library.entries, system.local_solutions, strict=True):
        primitives = entry.primitive_modes
        n_prim = len(primitives)

        # Determine active primitives
        is_active = np.zeros(n_prim, dtype=bool)
        active_keys: list = []
        for port in entry.operator.ports:
            port_index = entry.operator.ports.index(port)
            lvl = (
                int(system.levels[int(port.global_interface)])
                if port.kind == "interface" else 0
            )
            indices = list(basis.active_indices(primitives, port_index, lvl))
            for idx in indices:
                is_active[idx] = True
                m = primitives[idx]
                if m.interface_id is not None:
                    active_keys.append(
                        (int(m.interface_id), m.component, m.polynomial)
                        if hasattr(m, "polynomial") else
                        (int(m.interface_id), m.component)
                    )

        if not np.any(is_active):
            continue

        # Build full primitive coefficients from active solution
        full_coeffs = np.zeros(n_prim, dtype=float)
        active_arr = np.flatnonzero(is_active).astype(np.int32)
        G_active = entry.primitive_G[np.ix_(active_arr, active_arr)]

        # Solve for active coefficients from local solution
        # local_sol = R_active @ c_active → c_active = least_squares
        R_active = entry.primitive_responses[:, active_arr]
        try:
            c_active, _res, _rank, _sv = np.linalg.lstsq(R_active, local_sol, rcond=None)
        except np.linalg.LinAlgError:
            continue
        full_coeffs[active_arr] = c_active

        # Full moments
        full_moments = entry.primitive_G @ full_coeffs

        # Accumulate squared moments for inactive primitives
        for idx in range(n_prim):
            if is_active[idx]:
                continue
            m = primitives[idx]
            if m.interface_id is None:
                continue
            iid = int(m.interface_id)
            inactive_sq[iid] += float(full_moments[idx]) ** 2

    return np.sqrt(inactive_sq) / np.maximum(np.sqrt(measures), eps)


# ---------------------------------------------------------------------------
# Combined estimator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidualComponents:
    """The four direct interface residual measurements."""

    velocity_jump: np.ndarray
    normal_flux: np.ndarray
    tangential_moment: np.ndarray
    inactive_mode: np.ndarray
    measures: np.ndarray


def residual_indicators(
    library,
    system,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> tuple[np.ndarray, ResidualComponents]:
    """Per-interface combined residual indicator.

    .. math::

        η_i = \\sqrt{\\sum_k w_k \\cdot \\text{comp}_k[i]^2}

    where the four components are area-weighted (each divided by √|Γ_i|)
    so the result is dimensionless.
    """
    wv, wn, wt, wi = weights
    vj = two_sided_velocity_jump(library, system)
    nf = normal_flux_residual(library, system)
    tm = tangential_moment_residual(library, system)
    im = inactive_mode_residual(library, system)
    measures = interface_measure(library.partition)

    eta_sq = wv * vj ** 2 + wn * nf ** 2 + wt * tm ** 2 + wi * im ** 2
    return np.sqrt(eta_sq), ResidualComponents(
        velocity_jump=vj,
        normal_flux=nf,
        tangential_moment=tm,
        inactive_mode=im,
        measures=measures,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _p1_vertex_fields(W, solution_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_vertices, p_vertices) P1 arrays for a single local solution."""
    from ddpnm_core.reconstruction import mixed_solution_to_p1
    return mixed_solution_to_p1(W, solution_vector)

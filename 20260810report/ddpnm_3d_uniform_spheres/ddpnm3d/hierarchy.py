"""3-D adaptive DDPNM hierarchy.

This module now delegates to the unified ``ddpnm_core`` pipeline while
preserving the original public API (dataclass field names, default
``pressure_stabilization=0.0``, 3D field names like ``interface_moment_residuals``,
etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dolfinx import mesh as dmesh
from dolfinx import fem

from ddpnm_core.algebra import (
    HierarchyError,
    _dorfler_mark,
    _interface_indicators,
    _level_counts,
    hierarchy_error,
)
from ddpnm_core.constants import LEVEL_NAMES
from ddpnm_core.reconstruction import mixed_solution_to_p1
from ddpnm_core.solver_types import PortInfo

from .geometry import PartitionData

# Re-export dimension-specific constants and helpers
from .basis_3d import (
    COMPONENTS,
    POLYNOMIALS,
    AFFINE_MODE_GROUPS,
    affine_interface_mode_keys,
    _interface_nodes,
    _tangent_frame,
    _interface_frames_and_scales,
)

# Backward-compatible alias for code that may import _primitive_mode_specs
# from hierarchy.  The logic now lives in HierarchyBasis.primitive_specs.
def _primitive_mode_specs(partition, pore_id, ports, interface_tangents, interface_scales):
    """Deprecated thin wrapper — use HierarchyBasis.primitive_specs directly."""
    from ddpnm3d.basis_3d import HierarchyBasis as HB
    raise DeprecationWarning(
        "_primitive_mode_specs is deprecated. Use HierarchyBasis directly."
    )

# Tolerances / limits — preserved from original
DEFAULT_TOLERANCE = 1.0e-2
DEFAULT_MARKING_THETA = 0.65
DEFAULT_MAX_MARKED = 12
DEFAULT_MAX_ITERATIONS = 40


# ---------------------------------------------------------------------------
# Dataclasses — preserved exactly for backward compatibility
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrimitiveMode:
    port_index: int
    component: str
    polynomial: str
    interface_id: int | None
    known_coefficient: float | None


@dataclass
class HierarchyLocalResponse:
    pore_id: int
    submesh: dmesh.Mesh
    parent_cell_map: np.ndarray
    parent_vertex_map: np.ndarray
    ports: tuple[PortInfo, ...]
    primitive_modes: tuple[PrimitiveMode, ...]
    W: fem.FunctionSpace
    primitive_loads: np.ndarray
    primitive_responses: np.ndarray
    primitive_G: np.ndarray
    ndofs: int
    symmetry_error: float


@dataclass
class HierarchyLibrary:
    partition: PartitionData
    local_responses: list[HierarchyLocalResponse]
    interface_nodes: tuple[tuple[int, ...], ...]
    interface_tangents: np.ndarray
    interface_scales: np.ndarray
    viscosity: float
    pressure_stabilization: float
    inlet_pressure: float
    outlet_pressure: float


@dataclass
class HierarchySolution:
    levels: np.ndarray
    method_name: str
    global_keys: tuple[tuple, ...]
    coefficients: np.ndarray
    interface_pressures: np.ndarray
    schur_matrix: np.ndarray
    rhs: np.ndarray
    local_responses: list[HierarchyLocalResponse]
    local_solutions: list[np.ndarray]
    interface_flux_sums: np.ndarray
    interface_moment_residuals: np.ndarray
    boundary_fluxes: dict[str, float]
    min_schur_eigenvalue: float
    symmetry_error: float
    relative_linear_residual: float
    max_mass_residual: float


@dataclass
class AdaptiveIteration:
    phase: str
    iteration: int
    error: HierarchyError
    counts: tuple[int, int, int]
    interface_unknowns: int
    marked_interfaces: tuple[int, ...]


@dataclass
class AdaptiveHierarchyResult:
    initial_ddpnm: HierarchySolution
    full_ddpnmt: HierarchySolution
    full_hoddpnm: HierarchySolution
    final_solution: HierarchySolution
    history: list[AdaptiveIteration]
    tolerance: float
    marking_theta: float
    final_error_to_hoddpnm: HierarchyError


# ---------------------------------------------------------------------------
# _mode_is_active / _mode_key / _global_keys — kept as thin wrappers
# ---------------------------------------------------------------------------

def _mode_is_active(mode: PrimitiveMode, level: int) -> bool:
    if mode.interface_id is None:
        return True
    if level == 0:
        return mode.component == "normal" and mode.polynomial == "P0"
    if level == 1:
        return mode.polynomial == "P0"
    if level == 2:
        return True
    raise ValueError(f"Invalid hierarchy level {level}.")


def _mode_key(mode: PrimitiveMode) -> tuple:
    if mode.interface_id is None:
        raise ValueError("A prescribed boundary mode has no global key.")
    return (int(mode.interface_id), mode.component, mode.polynomial)


def _global_keys(levels: np.ndarray) -> tuple[tuple, ...]:
    keys: list[tuple] = []
    for interface_id, level_value in enumerate(levels):
        level = int(level_value)
        if level == 0:
            keys.append((interface_id, "normal", "P0"))
        elif level == 1:
            keys.extend((interface_id, component, "P0") for component in COMPONENTS)
        elif level == 2:
            keys.extend(affine_interface_mode_keys(interface_id))
        else:
            raise ValueError(f"Invalid hierarchy level {level}.")
    return tuple(keys)


# ---------------------------------------------------------------------------
# Core functions — delegate to ddpnm_core
# ---------------------------------------------------------------------------

def build_hierarchy_library(
    partition: PartitionData,
    viscosity: float = 1.0,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
    pressure_stabilization: float = 0.0,
) -> HierarchyLibrary:
    """Build the primitive response library (one factorization per pore)."""
    from ddpnm_core.library import build_response_library
    from ddpnm3d.basis_3d import HierarchyBasis as HB

    basis = HB(partition)
    core_lib = build_response_library(
        partition, basis, viscosity=viscosity,
        inlet_pressure=inlet_pressure, outlet_pressure=outlet_pressure,
        pressure_stabilization=pressure_stabilization,
    )

    # Convert core entries → HierarchyLocalResponse
    local_responses: list[HierarchyLocalResponse] = []
    for entry in core_lib.entries:
        pm_3d = tuple(
            PrimitiveMode(
                port_index=m.port_index,
                component=m.component,
                polynomial=m.polynomial,
                interface_id=m.interface_id,
                known_coefficient=m.known_coefficient,
            )
            for m in entry.primitive_modes
        )
        local_responses.append(HierarchyLocalResponse(
            pore_id=entry.operator.pore_id,
            submesh=entry.operator.submesh,
            parent_cell_map=entry.operator.parent_cell_map,
            parent_vertex_map=entry.operator.parent_vertex_map,
            ports=entry.operator.ports,
            primitive_modes=pm_3d,
            W=entry.operator.W,
            primitive_loads=entry.primitive_loads,
            primitive_responses=entry.primitive_responses,
            primitive_G=entry.primitive_G,
            ndofs=entry.operator.ndofs,
            symmetry_error=entry.symmetry_error,
        ))

    library = HierarchyLibrary(
        partition=partition,
        local_responses=local_responses,
        interface_nodes=basis.interface_nodes,
        interface_tangents=basis.interface_tangents,
        interface_scales=basis.interface_scales,
        viscosity=viscosity,
        pressure_stabilization=pressure_stabilization,
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure,
    )
    # Cache the core library so solve_hierarchy doesn't re-factorize
    object.__setattr__(library, "_core", core_lib)
    return library


def solve_hierarchy(
    library: HierarchyLibrary,
    levels: np.ndarray | list[int] | tuple[int, ...],
) -> HierarchySolution:
    """Solve the global Schur system for the given interface levels.

    Uses the cached core library (``library._core``) if available;
    otherwise rebuilds it.
    """
    from ddpnm_core.assembler import InterfaceAssembler

    levels_array = np.asarray(levels, dtype=np.int8).copy()
    ninterfaces = len(library.partition.interface_pairs)
    if levels_array.shape != (ninterfaces,):
        raise ValueError(
            f"Expected {ninterfaces} interface levels, got {levels_array.shape}."
        )

    core_lib = _ensure_core_library(library)
    assembler = InterfaceAssembler(core_lib)
    system = assembler.assemble(levels_array)

    return _to_hierarchy_solution(library, system, levels_array)


def _ensure_core_library(library: HierarchyLibrary):
    """Return the cached core ResponseLibrary, building it if necessary."""
    cached = getattr(library, "_core", None)
    if cached is not None:
        return cached
    core = _to_core_library(library)
    object.__setattr__(library, "_core", core)
    return core


def _to_core_library(library: HierarchyLibrary):
    """Build a core ResponseLibrary from the wrapper library.

    Rebuilds LocalStokesOperators (re-factorizes).  Called only once per
    library; the result is cached on ``library._core``.
    """
    from ddpnm_core.library import ResponseLibrary, LocalLibraryEntry
    from ddpnm_core.basis import PrimitiveMode as CorePM
    from ddpnm_core.stokes_operator import build_local_stokes_operator

    core_entries = []
    for resp in library.local_responses:
        core_pm = tuple(
            CorePM(
                port_index=m.port_index,
                component=m.component,
                polynomial=m.polynomial,
                interface_id=m.interface_id,
                node_index=None,
                known_coefficient=m.known_coefficient,
            )
            for m in resp.primitive_modes
        )
        operator = build_local_stokes_operator(
            library.partition, resp.pore_id,
            library.viscosity, library.pressure_stabilization,
            library.inlet_pressure, library.outlet_pressure,
        )
        core_entries.append(LocalLibraryEntry(
            operator=operator,
            primitive_modes=core_pm,
            primitive_loads=resp.primitive_loads,
            primitive_responses=resp.primitive_responses,
            primitive_G=resp.primitive_G,
            symmetry_error=resp.symmetry_error,
        ))

    from ddpnm3d.basis_3d import HierarchyBasis as HB
    basis = HB(library.partition)
    return ResponseLibrary(
        partition=library.partition,
        basis=basis,
        entries=core_entries,
        interface_nodes=library.interface_nodes,
        viscosity=library.viscosity,
        pressure_stabilization=library.pressure_stabilization,
        inlet_pressure=library.inlet_pressure,
        outlet_pressure=library.outlet_pressure,
    )


def _to_hierarchy_solution(
    library: HierarchyLibrary,
    system,
    levels_array: np.ndarray,
) -> HierarchySolution:
    """Convert SchurSystem → HierarchySolution with 3D field names."""
    ninterfaces = len(library.partition.interface_pairs)
    keys = system.global_keys
    key_to_dof = {k: d for d, k in enumerate(keys)}

    interface_pressures = np.asarray([
        system.coefficients[key_to_dof[(i, "normal", "P0")]]
        for i in range(ninterfaces)
    ])

    # Per-interface normal flux sums (3D convention: -moments)
    interface_flux_sums = np.zeros(ninterfaces, dtype=float)
    for dof, key in enumerate(keys):
        iid, comp, poly = key
        if comp == "normal" and poly == "P0":
            interface_flux_sums[iid] -= system.moment_residuals[dof]

    method_name = (
        LEVEL_NAMES[int(levels_array[0])]
        if np.all(levels_array == levels_array[0])
        else "adaptive-DDPNM/DDPNMT/HODDPNM"
    )

    return HierarchySolution(
        levels=levels_array,
        method_name=method_name,
        global_keys=keys,
        coefficients=system.coefficients,
        interface_pressures=interface_pressures,
        schur_matrix=system.schur_matrix,
        rhs=system.rhs,
        local_responses=library.local_responses,
        local_solutions=system.local_solutions,
        interface_flux_sums=interface_flux_sums,
        interface_moment_residuals=system.moment_residuals,
        boundary_fluxes=system.boundary_fluxes,
        min_schur_eigenvalue=system.min_schur_eigenvalue,
        symmetry_error=system.symmetry_error,
        relative_linear_residual=system.relative_linear_residual,
        max_mass_residual=(
            float(np.max(np.abs(system.moment_residuals)))
            if len(system.moment_residuals) else 0.0
        ),
    )


def reconstruct_hierarchy_vertices(
    library: HierarchyLibrary, solution: HierarchySolution
) -> tuple[np.ndarray, np.ndarray]:
    """P1 vertex reconstruction by averaging incident pore fields."""
    nvertices = library.partition.mesh.topology.index_map(0).size_local
    u_sum = np.zeros((nvertices, 3), dtype=float)
    p_sum = np.zeros(nvertices, dtype=float)
    counts = np.zeros(nvertices, dtype=np.int32)
    for response, vector in zip(
        solution.local_responses, solution.local_solutions, strict=True
    ):
        u_local, p_local = mixed_solution_to_p1(response.W, vector)
        parent = np.asarray(response.parent_vertex_map, dtype=np.int32)
        u_sum[parent] += u_local
        p_sum[parent] += p_local
        counts[parent] += 1
    if np.any(counts == 0):
        raise RuntimeError("Hierarchy reconstruction missed parent vertices.")
    return u_sum / counts[:, None], p_sum / counts


def run_adaptive_hierarchy(
    library: HierarchyLibrary,
    tolerance: float = DEFAULT_TOLERANCE,
    marking_theta: float = DEFAULT_MARKING_THETA,
    max_marked_per_iteration: int = DEFAULT_MAX_MARKED,
    max_iterations_per_phase: int = DEFAULT_MAX_ITERATIONS,
) -> AdaptiveHierarchyResult:
    """Two-phase adaptive hierarchy (DDPNM→DDPNMT→HODDPNM)."""
    ninterfaces = len(library.partition.interface_pairs)
    ddpnm = solve_hierarchy(library, np.zeros(ninterfaces, dtype=np.int8))
    ddpnmt = solve_hierarchy(library, np.ones(ninterfaces, dtype=np.int8))
    hoddpnm = solve_hierarchy(library, np.full(ninterfaces, 2, dtype=np.int8))

    target_data = (
        ("DDPNM_to_DDPNMT",
         reconstruct_hierarchy_vertices(library, ddpnmt), 1),
        ("DDPNMT_to_HODDPNM",
         reconstruct_hierarchy_vertices(library, hoddpnm), 2),
    )

    levels = np.zeros(ninterfaces, dtype=np.int8)
    current = ddpnm
    history: list[AdaptiveIteration] = []

    for phase, target_fields, target_level in target_data:
        for iteration in range(max_iterations_per_phase + 1):
            current_fields = reconstruct_hierarchy_vertices(library, current)
            error = hierarchy_error(current_fields, target_fields)
            if error.combined <= tolerance:
                history.append(AdaptiveIteration(
                    phase, iteration, error, _level_counts(levels),
                    len(current.global_keys), (),
                ))
                break
            candidates = np.flatnonzero(levels < target_level)
            if len(candidates) == 0:
                history.append(AdaptiveIteration(
                    phase, iteration, error, _level_counts(levels),
                    len(current.global_keys), (),
                ))
                break
            # Use residual-based estimator (Phase 2)
            from ddpnm_core.estimate import residual_indicators
            from ddpnm_core.assembler import InterfaceAssembler
            core_lib = _ensure_core_library(library)
            assembler = InterfaceAssembler(core_lib)
            core_system = assembler.assemble(current.levels)
            indicators, _components = residual_indicators(core_lib, core_system)
            marked = _dorfler_mark(
                indicators, candidates, marking_theta, max_marked_per_iteration
            )
            history.append(AdaptiveIteration(
                phase, iteration, error, _level_counts(levels),
                len(current.global_keys), marked,
            ))
            for interface_id in marked:
                levels[interface_id] = min(int(levels[interface_id]) + 1, target_level)
            current = solve_hierarchy(library, levels)
        else:
            raise RuntimeError(
                f"Adaptive phase {phase} exceeded its iteration limit."
            )

    final_fields = reconstruct_hierarchy_vertices(library, current)
    hodd_fields = reconstruct_hierarchy_vertices(library, hoddpnm)
    return AdaptiveHierarchyResult(
        initial_ddpnm=ddpnm,
        full_ddpnmt=ddpnmt,
        full_hoddpnm=hoddpnm,
        final_solution=current,
        history=history,
        tolerance=tolerance,
        marking_theta=marking_theta,
        final_error_to_hoddpnm=hierarchy_error(final_fields, hodd_fields),
    )

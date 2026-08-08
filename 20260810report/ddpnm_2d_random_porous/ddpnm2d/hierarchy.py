"""2-D adaptive DDPNM hierarchy.

This module now delegates to the unified ``ddpnm_core`` pipeline while
preserving the original public API (dataclass fields, ``HierarchyLibrary``
field names, output functions, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
import numpy as np
from dolfinx import fem, mesh as dmesh

from .geometry import PARTICLES, PartitionData
from ddpnm_core.algebra import (
    HierarchyError,
    _dorfler_mark,
    _interface_indicators,
    _level_counts,
    hierarchy_error,
)
from ddpnm_core.constants import LEVEL_NAMES
from ddpnm_core.io import topology_arrays
from ddpnm_core.reconstruction import mixed_solution_to_p1
from ddpnm_core.solver_types import PortInfo


# ---------------------------------------------------------------------------
# Dataclasses — preserved exactly
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrimitiveMode:
    port_index: int
    component: str
    interface_id: int | None
    interface_node: int | None
    known_coefficient: float | None


@dataclass(frozen=True)
class ActiveMode:
    key: tuple | None
    known_coefficient: float | None
    label: str


@dataclass
class HierarchyLocalResponse:
    pore_id: int
    submesh: dmesh.Mesh
    parent_vertex_map: np.ndarray
    ports: tuple[PortInfo, ...]
    interface_nodes: dict[int, tuple[int, ...]]
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
    schur_matrix: np.ndarray
    rhs: np.ndarray
    local_responses: list[HierarchyLocalResponse]
    local_solutions: list[np.ndarray]
    flux_residuals: np.ndarray
    min_schur_eigenvalue: float
    symmetry_error: float
    relative_linear_residual: float
    max_flux_residual: float


@dataclass
class AdaptiveIteration:
    phase: str
    iteration: int
    error: HierarchyError
    counts: tuple[int, int, int]
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
    final_error_to_reference: HierarchyError | None


# ---------------------------------------------------------------------------
# Interface node helpers
# ---------------------------------------------------------------------------

def _interface_nodes(partition: PartitionData) -> tuple[tuple[int, ...], ...]:
    msh = partition.mesh
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    result: list[tuple[int, ...]] = []
    for interface_id in range(len(partition.interface_pairs)):
        facets = np.flatnonzero(partition.facet_interface_ids == interface_id)
        vertices = np.unique(np.concatenate([f2v.links(int(f)) for f in facets]))
        xy = msh.geometry.x[vertices, :2]
        tangent = partition.interface_tangents[interface_id]
        order = np.argsort(xy @ tangent)
        result.append(tuple(int(v) for v in vertices[order]))
    return tuple(result)


# ---------------------------------------------------------------------------
# Global keys and active transforms (2D-specific logic)
# ---------------------------------------------------------------------------

def _global_keys(
    levels: np.ndarray, interface_nodes: tuple[tuple[int, ...], ...]
) -> tuple[tuple, ...]:
    keys: list[tuple] = []
    for interface_id, level in enumerate(levels):
        if int(level) == 0:
            keys.append((interface_id, "normal_constant"))
        elif int(level) == 1:
            keys.extend(
                [(interface_id, "normal_constant"), (interface_id, "tangent_constant")]
            )
        elif int(level) == 2:
            for node_position in range(len(interface_nodes[interface_id])):
                keys.append((interface_id, "normal_node", node_position))
                keys.append((interface_id, "tangent_node", node_position))
        else:
            raise ValueError(f"Invalid hierarchy level {level} on interface {interface_id}.")
    return tuple(keys)


def _active_transform(
    response: HierarchyLocalResponse,
    levels: np.ndarray,
) -> tuple[np.ndarray, tuple[ActiveMode, ...]]:
    columns: list[np.ndarray] = []
    active: list[ActiveMode] = []
    nprimitive = len(response.primitive_modes)
    for port_index, port in enumerate(response.ports):
        primitive = [
            i for i, mode in enumerate(response.primitive_modes)
            if mode.port_index == port_index
        ]
        if port.kind != "interface":
            column = np.zeros(nprimitive)
            column[primitive[0]] = 1.0
            columns.append(column)
            active.append(
                ActiveMode(None, float(port.pressure), f"{port.kind}:normal")
            )
            continue
        interface_id = int(port.global_interface)
        level = int(levels[interface_id])
        normal_indices = [
            i for i in primitive if response.primitive_modes[i].component == "normal"
        ]
        tangent_indices = [
            i for i in primitive if response.primitive_modes[i].component == "tangent"
        ]
        if level <= 1:
            normal_constant = np.zeros(nprimitive)
            normal_constant[normal_indices] = 1.0
            columns.append(normal_constant)
            active.append(
                ActiveMode(
                    (interface_id, "normal_constant"), None,
                    f"interface-{interface_id}:normal-P0",
                )
            )
            if level == 1:
                tangent_constant = np.zeros(nprimitive)
                tangent_constant[tangent_indices] = 1.0
                columns.append(tangent_constant)
                active.append(
                    ActiveMode(
                        (interface_id, "tangent_constant"), None,
                        f"interface-{interface_id}:tangent-P0",
                    )
                )
        else:
            for node_position, (normal_index, tangent_index) in enumerate(
                zip(normal_indices, tangent_indices, strict=True)
            ):
                normal_node = np.zeros(nprimitive)
                normal_node[normal_index] = 1.0
                columns.append(normal_node)
                active.append(
                    ActiveMode(
                        (interface_id, "normal_node", node_position), None,
                        f"interface-{interface_id}:normal-P1-{node_position}",
                    )
                )
                tangent_node = np.zeros(nprimitive)
                tangent_node[tangent_index] = 1.0
                columns.append(tangent_node)
                active.append(
                    ActiveMode(
                        (interface_id, "tangent_node", node_position), None,
                        f"interface-{interface_id}:tangent-P1-{node_position}",
                    )
                )
    return np.column_stack(columns), tuple(active)


# ---------------------------------------------------------------------------
# build_hierarchy_library — delegates to core
# ---------------------------------------------------------------------------

def build_hierarchy_library(
    partition: PartitionData,
    viscosity: float = 1.0,
    inlet_pressure: float = 1.0,
    outlet_pressure: float = 0.0,
    pressure_stabilization: float = 1.0e-10,
) -> HierarchyLibrary:
    """Build the primitive response library (one factorization per pore)."""
    from ddpnm_core.library import build_response_library
    from ddpnm2d.basis_2d import HierarchyBasis as HB

    nodes = _interface_nodes(partition)
    basis = HB(partition)
    core_lib = build_response_library(
        partition, basis, viscosity=viscosity,
        inlet_pressure=inlet_pressure, outlet_pressure=outlet_pressure,
        pressure_stabilization=pressure_stabilization,
    )

    # Convert core entries → 2D HierarchyLocalResponse
    local_responses: list[HierarchyLocalResponse] = []
    for entry in core_lib.entries:
        pm_2d = tuple(
            PrimitiveMode(
                port_index=m.port_index,
                component=m.component,
                interface_id=m.interface_id,
                interface_node=m.node_index,
                known_coefficient=m.known_coefficient,
            )
            for m in entry.primitive_modes
        )
        # Build per-interface local node map
        local_nodes: dict[int, tuple[int, ...]] = {}
        parent_to_local = {int(p): l for l, p in enumerate(entry.operator.parent_vertex_map)}
        for iid, global_nodes in enumerate(nodes):
            local = tuple(
                parent_to_local[int(v)] for v in global_nodes
                if int(v) in parent_to_local
            )
            if local:
                local_nodes[iid] = local

        local_responses.append(HierarchyLocalResponse(
            pore_id=entry.operator.pore_id,
            submesh=entry.operator.submesh,
            parent_vertex_map=entry.operator.parent_vertex_map,
            ports=entry.operator.ports,
            interface_nodes=local_nodes,
            primitive_modes=pm_2d,
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
        interface_nodes=nodes,
        viscosity=viscosity,
        pressure_stabilization=pressure_stabilization,
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure,
    )
    object.__setattr__(library, "_core", core_lib)
    return library


# ---------------------------------------------------------------------------
# solve_hierarchy
# ---------------------------------------------------------------------------

def solve_hierarchy(
    library: HierarchyLibrary,
    levels: np.ndarray | list[int] | tuple[int, ...],
) -> HierarchySolution:
    """Solve the global Schur system using the original 2D transform logic."""
    levels_array = np.asarray(levels, dtype=np.int8).copy()
    ninterfaces = len(library.partition.interface_pairs)
    if levels_array.shape != (ninterfaces,):
        raise ValueError(
            f"Expected {ninterfaces} interface levels, got {levels_array.shape}."
        )

    keys = _global_keys(levels_array, library.interface_nodes)
    key_to_dof = {key: i for i, key in enumerate(keys)}
    S = np.zeros((len(keys), len(keys)), dtype=float)
    rhs = np.zeros(len(keys), dtype=float)
    local_data: list[tuple[np.ndarray, tuple[ActiveMode, ...], np.ndarray, np.ndarray]] = []

    for response in library.local_responses:
        transform, active = _active_transform(response, levels_array)
        G = transform.T @ response.primitive_G @ transform
        resp_mat = response.primitive_responses @ transform
        unknown = [i for i, mode in enumerate(active) if mode.key is not None]
        known = [i for i, mode in enumerate(active) if mode.key is None]
        dofs = [key_to_dof[active[i].key] for i in unknown]
        if unknown:
            S[np.ix_(dofs, dofs)] += G[np.ix_(unknown, unknown)]
            if known:
                known_coefficients = np.asarray(
                    [float(active[i].known_coefficient) for i in known]
                )
                rhs[dofs] -= G[np.ix_(unknown, known)] @ known_coefficients
        local_data.append((resp_mat, active, G, np.asarray(dofs, dtype=np.int32)))

    S = 0.5 * (S + S.T)
    coefficients = np.linalg.solve(S, rhs) if len(keys) else np.empty(0)
    eigenvalues = np.linalg.eigvalsh(S) if len(keys) else np.empty(0)
    local_solutions: list[np.ndarray] = []
    flux = np.zeros(len(keys), dtype=float)
    for response, (resp_mat, active, G, _) in zip(
        library.local_responses, local_data, strict=True
    ):
        local_coefficients = np.empty(len(active), dtype=float)
        for i, mode in enumerate(active):
            local_coefficients[i] = (
                coefficients[key_to_dof[mode.key]]
                if mode.key is not None
                else float(mode.known_coefficient)
            )
        local_solutions.append(resp_mat @ local_coefficients)
        moments = G @ local_coefficients
        for i, mode in enumerate(active):
            if mode.key is not None:
                flux[key_to_dof[mode.key]] += moments[i]
    residual = S @ coefficients - rhs
    relative_linear_residual = float(
        np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0e-30)
    )
    method_name = (
        LEVEL_NAMES[int(levels_array[0])]
        if np.all(levels_array == levels_array[0])
        else "adaptive-DDPNM/DDPNMT/HODDPNM"
    )
    return HierarchySolution(
        levels=levels_array,
        method_name=method_name,
        global_keys=keys,
        coefficients=coefficients,
        schur_matrix=S,
        rhs=rhs,
        local_responses=library.local_responses,
        local_solutions=local_solutions,
        flux_residuals=flux,
        min_schur_eigenvalue=float(eigenvalues[0]) if len(eigenvalues) else float("nan"),
        symmetry_error=float(np.linalg.norm(S - S.T) / max(np.linalg.norm(S), 1e-30)),
        relative_linear_residual=relative_linear_residual,
        max_flux_residual=float(np.max(np.abs(flux))) if len(flux) else 0.0,
    )


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_hierarchy_vertices(
    partition: PartitionData, solution: HierarchySolution
) -> tuple[np.ndarray, np.ndarray]:
    nvertices = partition.mesh.topology.index_map(0).size_local
    u_sum = np.zeros((nvertices, 2), dtype=float)
    p_sum = np.zeros(nvertices, dtype=float)
    counts = np.zeros(nvertices, dtype=np.int32)
    for response, vector in zip(
        solution.local_responses, solution.local_solutions, strict=True
    ):
        u_local, p_local = mixed_solution_to_p1(response.W, vector)
        for local_vertex, parent_vertex in enumerate(response.parent_vertex_map):
            u_sum[parent_vertex] += u_local[local_vertex]
            p_sum[parent_vertex] += p_local[local_vertex]
            counts[parent_vertex] += 1
    if np.any(counts == 0):
        raise RuntimeError("Hierarchy reconstruction missed parent vertices.")
    return u_sum / counts[:, None], p_sum / counts


# ---------------------------------------------------------------------------
# Adaptive loop
# ---------------------------------------------------------------------------

def run_adaptive_hierarchy(
    library: HierarchyLibrary,
    tolerance: float = 1.0e-2,
    marking_theta: float = 0.65,
    max_marked_per_iteration: int = 3,
    max_iterations_per_phase: int = 20,
    reference_fields: tuple[np.ndarray, np.ndarray] | None = None,
) -> AdaptiveHierarchyResult:
    ninterfaces = len(library.interface_nodes)
    ddpnm = solve_hierarchy(library, np.zeros(ninterfaces, dtype=np.int8))
    ddpnmt = solve_hierarchy(library, np.ones(ninterfaces, dtype=np.int8))
    hoddpnm = solve_hierarchy(library, np.full(ninterfaces, 2, dtype=np.int8))
    targets = {
        "DDPNM_to_DDPNMT": (ddpnmt, reconstruct_hierarchy_vertices(library.partition, ddpnmt), 1),
        "DDPNMT_to_HODDPNM": (hoddpnm, reconstruct_hierarchy_vertices(library.partition, hoddpnm), 2),
    }
    levels = np.zeros(ninterfaces, dtype=np.int8)
    current = ddpnm
    history: list[AdaptiveIteration] = []
    for phase, (_, target_fields, target_level) in targets.items():
        for iteration in range(max_iterations_per_phase + 1):
            current_fields = reconstruct_hierarchy_vertices(library.partition, current)
            error = hierarchy_error(current_fields, target_fields)
            if error.combined <= tolerance:
                history.append(
                    AdaptiveIteration(phase, iteration, error, _level_counts(levels), ())
                )
                break
            candidates = np.flatnonzero(levels < target_level)
            if len(candidates) == 0:
                history.append(
                    AdaptiveIteration(phase, iteration, error, _level_counts(levels), ())
                )
                break
            # Use residual-based estimator (Phase 2)
            from ddpnm_core.estimate import residual_indicators
            from ddpnm_core.assembler import InterfaceAssembler
            core_lib = getattr(library, "_core", None)
            if core_lib is not None:
                assembler = InterfaceAssembler(core_lib)
                core_system = assembler.assemble(current.levels)
                indicators, _components = residual_indicators(core_lib, core_system)
            else:
                indicators = _interface_indicators(library, current_fields, target_fields)
            marked = _dorfler_mark(indicators, candidates, marking_theta, max_marked_per_iteration)
            history.append(
                AdaptiveIteration(phase, iteration, error, _level_counts(levels), marked)
            )
            for interface_id in marked:
                levels[interface_id] = min(int(levels[interface_id]) + 1, target_level)
            current = solve_hierarchy(library, levels)
        else:
            raise RuntimeError(f"Adaptive phase {phase} exceeded its iteration limit.")

    final_fields = reconstruct_hierarchy_vertices(library.partition, current)
    hodd_fields = reconstruct_hierarchy_vertices(library.partition, hoddpnm)
    final_error = hierarchy_error(final_fields, hodd_fields)
    reference_error = (
        hierarchy_error(final_fields, reference_fields)
        if reference_fields is not None else None
    )
    return AdaptiveHierarchyResult(
        initial_ddpnm=ddpnm,
        full_ddpnmt=ddpnmt,
        full_hoddpnm=hoddpnm,
        final_solution=current,
        history=history,
        tolerance=tolerance,
        marking_theta=marking_theta,
        final_error_to_hoddpnm=final_error,
        final_error_to_reference=reference_error,
    )


# ---------------------------------------------------------------------------
# Output — unchanged from original
# ---------------------------------------------------------------------------

def _draw_particles(ax) -> None:
    for x, y, radius in PARTICLES:
        ax.add_patch(plt.Circle((x, y), radius, facecolor="white", edgecolor="#60646b", lw=0.7))


def plot_adaptive_results(
    partition: PartitionData,
    result: AdaptiveHierarchyResult,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    history = result.history
    x = np.arange(len(history))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    axes[0].semilogy(x, [h.error.velocity for h in history], "o-", label="velocity")
    axes[0].semilogy(x, [h.error.pressure for h in history], "s-", label="pressure")
    axes[0].axhline(result.tolerance, color="#c62828", ls="--", label="target tolerance")
    axes[0].set_xlabel("adaptive iteration")
    axes[0].set_ylabel("hierarchical relative difference")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    counts = np.asarray([h.counts for h in history])
    colors = ["#4c78a8", "#f2a541", "#d1495b"]
    bottom = np.zeros(len(history))
    for level, name in LEVEL_NAMES.items():
        axes[1].bar(x, counts[:, level], bottom=bottom, color=colors[level], label=name)
        bottom += counts[:, level]
    axes[1].set_xlabel("adaptive iteration")
    axes[1].set_ylabel("number of interfaces")
    axes[1].legend(frameon=False)
    fig.savefig(out_dir / "adaptive_convergence.png", dpi=220)
    plt.close(fig)

    points, cells = topology_arrays(partition.mesh)
    cell_levels = np.zeros(len(cells), dtype=np.int8)
    incident: dict[int, list[int]] = {i: [] for i in range(len(np.unique(partition.cell_labels)))}
    for interface_id, pair in enumerate(partition.interface_pairs):
        incident[pair[0]].append(interface_id)
        incident[pair[1]].append(interface_id)
    for pore_id, interfaces in incident.items():
        if interfaces:
            cell_levels[partition.cell_labels == pore_id] = int(
                np.max(result.final_solution.levels[interfaces])
            )
    fig, ax = plt.subplots(figsize=(7.3, 7.0), constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(colors)
    ax.tripcolor(points[:, 0], points[:, 1], cells, facecolors=cell_levels, cmap=cmap, vmin=-0.5, vmax=2.5)
    _draw_particles(ax)
    for interface_id, nodes in enumerate(_interface_nodes(partition)):
        xy = partition.mesh.geometry.x[np.asarray(nodes), :2]
        ax.plot(xy[:, 0], xy[:, 1], color=colors[int(result.final_solution.levels[interface_id])], lw=2.0)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[], title="Final adaptive interface hierarchy")
    handles = [plt.Line2D([0], [0], color=colors[i], lw=4, label=LEVEL_NAMES[i]) for i in range(3)]
    ax.legend(handles=handles, loc="upper right", frameon=True)
    fig.savefig(out_dir / "adaptive_final_hierarchy.png", dpi=220)
    plt.close(fig)


def write_algorithm_box(out_dir: Path, tolerance: float, theta: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "算法：二维分层自适应 DD-PNM",
        "输入：解析孔区划分、局部 P2-P1 矩阵、目标容忍值 TOL",
        "1. 每个局部 Stokes 矩阵只分解一次。",
        "2. 构造界面 P1 法向/切向单位响应库。",
        "3. 0 级 DDPNM：每条界面一个常数法向牵引自由度。",
        "4. 与 1 级 DDPNMT 比较：增加一个常数切向牵引自由度。",
        "5. 当 max(速度相对差, 压力相对差) > TOL 时：",
        "     用层级差指标和 Dörfler 准则标记界面；",
        "     被标记界面只升一级，重新装配并求解 Schur 系统。",
        "6. 与 2 级 HODDPNM 比较：界面 P1 节点法向+切向牵引。",
        "7. 重复 标记 → 升级 → Schur 求解，直到达到 TOL 或全部升至 2 级。",
        "8. 重构局部场，验证全部界面矩残差及单体有限元误差。",
        f"默认参数：TOL={tolerance:.3g}，Dörfler θ={theta:.2f}，每轮最多升级 3 条界面。",
    ]
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    chinese = FontProperties(fname=str(font_path)) if font_path.exists() else None
    fig, ax = plt.subplots(figsize=(11.6, 7.1), constrained_layout=True)
    ax.axis("off")
    ax.add_patch(
        plt.Rectangle((0.02, 0.03), 0.96, 0.94, transform=ax.transAxes,
                      facecolor="#fbfbf8", edgecolor="#20252b", linewidth=1.5)
    )
    ax.text(0.05, 0.94, lines[0], transform=ax.transAxes, va="top", ha="left",
            fontsize=15, fontweight="bold", fontproperties=chinese)
    ax.plot([0.05, 0.95], [0.885, 0.885], transform=ax.transAxes, color="#20252b", lw=0.9)
    ax.text(0.055, 0.85, "\n".join(lines[1:]), transform=ax.transAxes, va="top", ha="left",
            fontsize=11.3, linespacing=1.45, fontproperties=chinese)
    fig.savefig(out_dir / "adaptive_algorithm_box.png", dpi=220)
    plt.close(fig)
    markdown = """# 算法：二维分层自适应 DD-PNM

**输入：**解析孔区划分、局部 Taylor--Hood 矩阵、目标容忍值 `TOL`。

1. 每个局部 Stokes 矩阵只分解一次，并建立 P1 法向/切向界面响应库。
2. 所有界面初始化为 0 级 DDPNM：只保留 `P0` 法向牵引。
3. 以完整 DDPNMT（`P0` 法向 + `P0` 切向）作为第一层富集比较。
4. 若归一化速度或压力差超过 `TOL`，用层级差指标和 Dörfler 准则标记界面；每条被标记界面只升一级，然后重新装配凝聚界面系统。
5. 以完整 HODDPNM（`P1` 节点法向 + `P1` 节点切向）作为第二层比较，重复"标记—升级—Schur 求解"。
6. 当两个归一化场差均不超过 `TOL`，或全部界面达到 HODDPNM 时停止。
7. 重构局部场，并验证界面矩残差和相对于单体有限元的误差。
"""
    (out_dir / "ADAPTIVE_ALGORITHM.md").write_text(markdown, encoding="utf-8")


def write_adaptive_report(
    library: HierarchyLibrary,
    result: AdaptiveHierarchyResult,
    out_dir: Path,
    parameters: dict,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_adaptive_results(library.partition, result, out_dir)
    write_algorithm_box(out_dir, result.tolerance, result.marking_theta)
    with (out_dir / "adaptive_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "phase", "iteration", "velocity_difference", "pressure_difference",
            "combined_difference", "ddpnm_interfaces", "ddpnmt_interfaces",
            "hoddpnm_interfaces", "marked_interfaces",
        ])
        for item in result.history:
            writer.writerow([
                item.phase, item.iteration, item.error.velocity, item.error.pressure,
                item.error.combined, *item.counts,
                " ".join(str(i) for i in item.marked_interfaces),
            ])
    def solution_stats(solution: HierarchySolution) -> dict:
        return {
            "interface_unknowns": len(solution.global_keys),
            "minimum_schur_eigenvalue": solution.min_schur_eigenvalue,
            "schur_symmetry_error": solution.symmetry_error,
            "relative_linear_residual": solution.relative_linear_residual,
            "maximum_interface_moment_residual": solution.max_flux_residual,
        }
    report = {
        "method": "adaptive DDPNM -> DDPNMT -> HODDPNM hierarchy",
        "interface_spaces": {
            "DDPNM": "one P0 normal traction coefficient per interface",
            "DDPNMT": "P0 normal plus P0 tangential traction per interface",
            "HODDPNM": "P1 nodal normal plus P1 nodal tangential traction; local Schur condensation",
        },
        "parameters": parameters,
        "target_tolerance": result.tolerance,
        "marking_theta": result.marking_theta,
        "counts": {
            "subdomains": len(np.unique(library.partition.cell_labels)),
            "interfaces": len(library.partition.interface_pairs),
            "final_DDPNM_interfaces": int(np.sum(result.final_solution.levels == 0)),
            "final_DDPNMT_interfaces": int(np.sum(result.final_solution.levels == 1)),
            "final_HODDPNM_interfaces": int(np.sum(result.final_solution.levels == 2)),
        },
        "systems": {
            "DDPNM": solution_stats(result.initial_ddpnm),
            "DDPNMT": solution_stats(result.full_ddpnmt),
            "HODDPNM": solution_stats(result.full_hoddpnm),
            "adaptive_final": solution_stats(result.final_solution),
        },
        "local_response_library": {
            "maximum_symmetry_error": float(
                max(response.symmetry_error for response in library.local_responses)
            ),
            "sum_local_mixed_dofs": int(
                sum(response.ndofs for response in library.local_responses)
            ),
        },
        "final_error_to_full_HODDPNM": result.final_error_to_hoddpnm.__dict__,
        "final_error_to_monolithic_FEM": (
            result.final_error_to_reference.__dict__
            if result.final_error_to_reference is not None else None
        ),
        "iterations": [
            {
                "phase": h.phase,
                "iteration": h.iteration,
                "error": h.error.__dict__,
                "level_counts": list(h.counts),
                "marked_interfaces": list(h.marked_interfaces),
            }
            for h in result.history
        ],
        "interpretation": (
            "A difference above tolerance promotes the marked interface to the richer "
            "space. Retaining DDPNM when that difference is large would reverse the "
            "usual adaptive accuracy logic. The comparison is a hierarchical (fine-level) "
            "estimator, analogous to the oracle diagnostic branch of the 3D reference."
        ),
    }
    (out_dir / "adaptive_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report

"""Single-phase tracer transport driven by a vertex velocity field.

Ported from the archived ``stokes_tracer_hoddpnm`` project (20260810report/
回收站) and adapted to run on an existing dolfinx mesh: the tracer function
spaces are built directly on ``msh`` and the velocity is injected as P1
vertex values (the same continuous field the Stokes metrics are computed
against).

Model: transient advection--diffusion of a scalar concentration with an
inlet step ``c = 1`` at ``x = 0``, natural outlet/wall flux, implicit
Euler time stepping with streamline-upwind stabilization and a
conservative bounded limiter that preserves the raw step mass in ``[0, 1]``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import ufl
from basix.ufl import element
from dolfinx import fem
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from ddpnm_core.fem_utils import to_scipy_matrix
from ddpnm_core.io import assign_p1_function, topology_vertex_coordinates

METHODS = ("FEM", "Classic-DDPNM-1", "NormalLinear-DDPNM-3", "Affine-DDPNM-9")

COLORS = {
    "FEM": "#2b2d42",
    "Classic-DDPNM-1": "#8d6e63",
    "NormalLinear-DDPNM-3": "#7B1FA2",
    "Affine-DDPNM-9": "#d1495b",
}


# ---------------------------------------------------------------------------
# Tracer solver
# ---------------------------------------------------------------------------


def solve_tracer(
    msh,
    u_vertices: np.ndarray,
    diffusivity: float = 0.05,
    porosity: float = 1.0,
    dt: float = 0.1,
    t_final: float = 20.0,
    supg: bool = True,
    supg_factor: float = 0.5,
) -> dict[str, object]:
    """Solve transient scalar advection--diffusion on *msh* with velocity
    *u_vertices* (P1 vertex values, topology-vertex ordering).

    Returns the time history, final concentration on dofs and vertices, and
    the tracer mass matrix (needed for reference-error metrics).
    """
    cell = msh.basix_cell()
    C = fem.functionspace(msh, element("Lagrange", cell, 1))
    U = fem.functionspace(msh, element("Lagrange", cell, 1, shape=(3,)))
    c = ufl.TrialFunction(C)
    w = ufl.TestFunction(C)
    u = fem.Function(U)
    assign_p1_function(u, np.asarray(u_vertices, dtype=float))
    u.x.scatter_forward()

    h = ufl.CellDiameter(msh)
    speed = ufl.sqrt(ufl.inner(u, u) + fem.Constant(msh, 1.0e-14))
    mass_form = fem.form(c * w * ufl.dx)
    mass_matrix = to_scipy_matrix(fem.assemble_matrix(mass_form)).tocsr()
    mass_step_matrix = (float(porosity) / float(dt)) * mass_matrix
    rhs_operator = fem.Constant(msh, float(porosity / dt)) * c * w * ufl.dx
    operator = (
        fem.Constant(msh, float(porosity / dt)) * c * w * ufl.dx
        + fem.Constant(msh, float(diffusivity)) * ufl.inner(ufl.grad(c), ufl.grad(w)) * ufl.dx
        + ufl.dot(u, ufl.grad(c)) * w * ufl.dx
    )
    if supg:
        tau = fem.Constant(msh, float(supg_factor)) * h / (2.0 * speed)
        streamline_test = ufl.dot(u, ufl.grad(w))
        operator += tau * streamline_test * (
            fem.Constant(msh, float(porosity / dt)) * c + ufl.dot(u, ufl.grad(c))
        ) * ufl.dx
        rhs_operator += tau * streamline_test * fem.Constant(msh, float(porosity / dt)) * c * ufl.dx

    A = to_scipy_matrix(fem.assemble_matrix(fem.form(operator))).tocsr()
    transport_matrix = (A - mass_step_matrix).tocsr()
    rhs_matrix = to_scipy_matrix(fem.assemble_matrix(fem.form(rhs_operator))).tocsr()

    coords = C.tabulate_dof_coordinates()
    vertex_coords = topology_vertex_coordinates(msh)
    # P1 dofs coincide with topology vertices; map both ways and verify.
    vertex_to_c_dof, concentration_map_distance = nearest_indices(
        vertex_coords, coords, "scalar concentration vertex output coordinates"
    )
    dof_to_vertex, dof_vertex_distance = nearest_indices(
        coords, vertex_coords, "scalar concentration dof coordinates"
    )
    lx = float(np.max(coords[:, 0]))
    inlet_dofs = np.asarray(
        fem.locate_dofs_geometrical(C, lambda x: np.isclose(x[0], 0.0)),
        dtype=np.int64,
    )
    outlet_dofs = np.flatnonzero(np.isclose(coords[:, 0], lx))
    if len(outlet_dofs) == 0:
        xmax = float(np.max(coords[:, 0]))
        outlet_dofs = np.flatnonzero(np.isclose(coords[:, 0], xmax))

    A_bc = impose_dirichlet_rows(A, inlet_dofs)
    lu = splu(A_bc.tocsc())
    mass_weights = np.asarray(mass_matrix @ np.ones(mass_matrix.shape[1]), dtype=float)
    cold_vec = np.zeros(C.dofmap.index_map.size_local * C.dofmap.index_map_bs, dtype=float)
    cold_vec[inlet_dofs] = 1.0

    tree = cKDTree(vertex_coords)
    _, nearest_velocity_nodes = tree.query(coords[outlet_dofs], k=1)
    outlet_weights = np.maximum(u_vertices[nearest_velocity_nodes, 0], 0.0)
    if float(np.sum(outlet_weights)) <= 1.0e-14:
        outlet_weights = np.ones(len(outlet_dofs), dtype=float)
    outlet_flux_weight = float(np.sum(outlet_weights))

    times = [0.0]
    cout = [weighted_average(cold_vec[outlet_dofs], outlet_weights)]
    masses = [scalar_mass(cold_vec, mass_matrix)]
    budget_masses = [masses[0]]
    raw_masses = [masses[0]]
    mins = [float(np.min(cold_vec))]
    maxs = [float(np.max(cold_vec))]
    raw_mins = [float(np.min(cold_vec))]
    raw_maxs = [float(np.max(cold_vec))]
    raw_below_counts = [int(np.count_nonzero(cold_vec < -1.0e-12))]
    raw_above_counts = [int(np.count_nonzero(cold_vec > 1.0 + 1.0e-12))]
    mass_rate = [0.0]
    transport_rate = [0.0]
    source_rate = [0.0]
    balance_residual_rate = [0.0]
    relative_balance_residual = [0.0]
    limiter_mass_change_abs = [0.0]
    limiter_mass_residual = [0.0]

    n_steps = int(np.ceil(t_final / dt))
    free_mask = np.ones(len(cold_vec), dtype=bool)
    free_mask[inlet_dofs] = False
    for step in range(1, n_steps + 1):
        rhs = rhs_matrix @ cold_vec
        rhs_bc = rhs.copy()
        rhs_bc[inlet_dofs] = 1.0
        craw = lu.solve(rhs_bc)
        craw[inlet_dofs] = 1.0

        raw_mass = scalar_mass(craw, mass_matrix)
        raw_masses.append(raw_mass)
        raw_mins.append(float(np.min(craw)))
        raw_maxs.append(float(np.max(craw)))
        raw_below_counts.append(int(np.count_nonzero(craw < -1.0e-12)))
        raw_above_counts.append(int(np.count_nonzero(craw > 1.0 + 1.0e-12)))

        limited, limiter_info = conservative_bounded_limiter(
            craw, mass_weights, inlet_dofs, raw_mass
        )
        limited[inlet_dofs] = 1.0
        residual = A @ craw - rhs
        step_mass_rate = (raw_mass - masses[-1]) / dt
        step_transport_rate = float(np.sum(transport_matrix @ craw))
        step_source_rate = float(np.sum(residual[inlet_dofs]))
        step_balance = float(np.sum(residual[free_mask]))
        scale = max(abs(step_mass_rate) + abs(step_transport_rate) + abs(step_source_rate), 1.0e-300)

        cold_vec = limited
        times.append(float(min(step * dt, t_final)))
        cout.append(weighted_average(cold_vec[outlet_dofs], outlet_weights))
        masses.append(scalar_mass(cold_vec, mass_matrix))
        budget_masses.append(budget_masses[-1] + dt * (step_source_rate - step_transport_rate))
        mins.append(float(np.min(cold_vec)))
        maxs.append(float(np.max(cold_vec)))
        mass_rate.append(step_mass_rate)
        transport_rate.append(step_transport_rate)
        source_rate.append(step_source_rate)
        balance_residual_rate.append(step_balance)
        relative_balance_residual.append(abs(step_balance) / scale)
        limiter_mass_change_abs.append(float(limiter_info["mass_change_abs"]))
        limiter_mass_residual.append(float(limiter_info["mass_residual"]))

    return {
        "history": {
            "time": np.asarray(times),
            "cout": np.asarray(cout),
            "mass": np.asarray(masses),
            "budget_mass": np.asarray(budget_masses),
            "raw_mass": np.asarray(raw_masses),
            "mass_rate": np.asarray(mass_rate),
            "transport_rate": np.asarray(transport_rate),
            "dirichlet_source_rate": np.asarray(source_rate),
            "mass_balance_residual_rate": np.asarray(balance_residual_rate),
            "mass_balance_relative_residual": np.asarray(relative_balance_residual),
            "limiter_mass_change_abs": np.asarray(limiter_mass_change_abs),
            "limiter_mass_residual": np.asarray(limiter_mass_residual),
            "min_c": np.asarray(mins),
            "max_c": np.asarray(maxs),
            "raw_min_c_before_limiter": np.asarray(raw_mins),
            "raw_max_c_before_limiter": np.asarray(raw_maxs),
            "raw_below_zero_before_limiter": np.asarray(raw_below_counts),
            "raw_above_one_before_limiter": np.asarray(raw_above_counts),
        },
        "final_concentration": cold_vec,
        "final_concentration_vertices": np.asarray(cold_vec[vertex_to_c_dof], dtype=float),
        "mass_matrix": mass_matrix,
        "outlet_flux_weight": outlet_flux_weight,
        "final_mass": masses[-1],
        "min_mass": min(masses),
        "max_mass": max(masses),
        "final_min": mins[-1],
        "final_max": maxs[-1],
        "final_below_zero": int(np.count_nonzero(cold_vec < -1.0e-12)),
        "final_above_one": int(np.count_nonzero(cold_vec > 1.0 + 1.0e-12)),
        "raw_final_min_before_limiter": raw_mins[-1],
        "raw_final_max_before_limiter": raw_maxs[-1],
        "raw_final_below_zero_before_limiter": raw_below_counts[-1],
        "raw_final_above_one_before_limiter": raw_above_counts[-1],
        "limiter_final_mass_residual": limiter_mass_residual[-1],
        "concentration_vertex_mapping_max_distance": concentration_map_distance,
        "dof_to_vertex_mapping_max_distance": dof_vertex_distance,
    }


def conservative_bounded_limiter(
    values: np.ndarray, weights: np.ndarray, fixed_dofs: np.ndarray, target_mass: float
) -> tuple[np.ndarray, dict[str, float]]:
    """Clip to [0, 1] then rescale the adjustable dofs to restore the raw mass."""
    out = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    fixed = np.zeros(len(out), dtype=bool)
    fixed[np.asarray(fixed_dofs, dtype=np.int64)] = True
    out[fixed] = np.clip(values[fixed], 0.0, 1.0)
    adjustable = ~fixed

    for _ in range(32):
        residual = float(target_mass - np.dot(weights, out))
        if abs(residual) <= 1.0e-12 * max(abs(target_mass), 1.0):
            break
        if residual > 0.0:
            idx = adjustable & (out < 1.0 - 1.0e-14)
            capacity = weights[idx] * (1.0 - out[idx])
            total_capacity = float(np.sum(capacity))
            if total_capacity <= 1.0e-300:
                break
            theta = min(1.0, residual / total_capacity)
            out[idx] += theta * (1.0 - out[idx])
        else:
            idx = adjustable & (out > 1.0e-14)
            capacity = weights[idx] * out[idx]
            total_capacity = float(np.sum(capacity))
            if total_capacity <= 1.0e-300:
                break
            theta = min(1.0, -residual / total_capacity)
            out[idx] -= theta * out[idx]

    final_residual = float(target_mass - np.dot(weights, out))
    return out, {
        "mass_residual": final_residual,
        "mass_change_abs": float(np.dot(weights, np.abs(out - np.clip(values, 0.0, 1.0)))),
    }


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def nearest_indices(
    query_coords: np.ndarray,
    source_coords: np.ndarray,
    label: str,
    tolerance: float = 1.0e-8,
) -> tuple[np.ndarray, float]:
    tree = cKDTree(np.asarray(source_coords, dtype=float))
    distances, indices = tree.query(np.asarray(query_coords, dtype=float), k=1)
    max_distance = float(np.max(distances)) if len(np.atleast_1d(distances)) else 0.0
    if max_distance > tolerance:
        raise RuntimeError(
            f"{label} do not match within {tolerance:g}; max distance is {max_distance:.3e}"
        )
    return np.asarray(indices, dtype=np.int64), max_distance


def impose_dirichlet_rows(A: csr_matrix, dofs: np.ndarray) -> csr_matrix:
    A = A.tolil(copy=True)
    for dof in np.asarray(dofs, dtype=np.int64):
        A.rows[int(dof)] = [int(dof)]
        A.data[int(dof)] = [1.0]
    return A.tocsr()


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    denom = max(float(np.sum(weights)), 1.0e-300)
    return float(np.dot(values, weights) / denom)


def crossing_time(time_values: np.ndarray, signal: np.ndarray, level: float) -> float:
    above = np.flatnonzero(signal >= level)
    if len(above) == 0:
        return float("nan")
    idx = int(above[0])
    if idx == 0:
        return float(time_values[0])
    t0, t1 = float(time_values[idx - 1]), float(time_values[idx])
    y0, y1 = float(signal[idx - 1]), float(signal[idx])
    if abs(y1 - y0) <= 1.0e-14:
        return t1
    return float(t0 + (level - y0) * (t1 - t0) / (y1 - y0))


def scalar_mass(values: np.ndarray, mass: csr_matrix) -> float:
    ones = np.ones(len(values), dtype=float)
    return float(ones @ (mass @ np.asarray(values, dtype=float)))


def mass_l2_relative(diff: np.ndarray, ref: np.ndarray, mass: csr_matrix) -> float:
    diff = np.asarray(diff, dtype=float)
    ref = np.asarray(ref, dtype=float)
    abs_sq = float(diff @ (mass @ diff))
    ref_sq = float(ref @ (mass @ ref))
    return float(np.sqrt(max(abs_sq, 0.0)) / max(np.sqrt(max(ref_sq, 0.0)), 1.0e-300))


def mass_l2_absolute(diff: np.ndarray, mass: csr_matrix) -> float:
    diff = np.asarray(diff, dtype=float)
    abs_sq = float(diff @ (mass @ diff))
    return float(np.sqrt(max(abs_sq, 0.0)))


def signed_time_error(value: float, ref_value: float) -> float:
    if value != value or ref_value != ref_value:
        return float("nan")
    return float(value - ref_value)


# ---------------------------------------------------------------------------
# CSV output (ported from the archived comparison script)
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_history_csv(path: Path, histories: dict[str, dict[str, np.ndarray]]) -> None:
    import csv

    rows: list[dict[str, object]] = []
    for method, history in histories.items():
        for i, t in enumerate(history["time"]):
            rows.append(
                {
                    "method": method,
                    "time": float(t),
                    "outlet_concentration": float(history["cout"][i]),
                    "mass": float(history["mass"][i]),
                    "budget_mass": float(history["budget_mass"][i]),
                    "raw_mass": float(history["raw_mass"][i]),
                    "mass_rate": float(history["mass_rate"][i]),
                    "transport_rate": float(history["transport_rate"][i]),
                    "dirichlet_source_rate": float(history["dirichlet_source_rate"][i]),
                    "mass_balance_residual_rate": float(history["mass_balance_residual_rate"][i]),
                    "mass_balance_relative_residual": float(history["mass_balance_relative_residual"][i]),
                    "limiter_mass_change_abs": float(history["limiter_mass_change_abs"][i]),
                    "limiter_mass_residual": float(history["limiter_mass_residual"][i]),
                    "min_c": float(history["min_c"][i]),
                    "max_c": float(history["max_c"][i]),
                    "raw_min_c_before_limiter": float(history["raw_min_c_before_limiter"][i]),
                    "raw_max_c_before_limiter": float(history["raw_max_c_before_limiter"][i]),
                    "raw_below_zero_before_limiter": int(history["raw_below_zero_before_limiter"][i]),
                    "raw_above_one_before_limiter": int(history["raw_above_one_before_limiter"][i]),
                }
            )
    write_csv(path, rows)


# ---------------------------------------------------------------------------
# Reference-error bookkeeping (ported from the archived comparison script)
# ---------------------------------------------------------------------------


def add_reference_errors(
    rows: list[dict[str, object]],
    histories: dict[str, dict[str, np.ndarray]],
    final_dof_fields: dict[str, np.ndarray],
    mass_matrix: csr_matrix,
    reference: str = "FEM",
) -> None:
    ref_curve = histories[reference]["cout"]
    ref_time = histories[reference]["time"]
    ref_row = next(row for row in rows if row["method"] == reference)
    ref_field = final_dof_fields[reference]
    ref_mass = ref_row["final_tracer_mass"]
    for row in rows:
        method = str(row["method"])
        signal = histories[method]["cout"]
        curve_diff = signal - ref_curve
        field_diff = final_dof_fields[method] - ref_field
        row["breakthrough_rel_l2_error"] = float(
            np.linalg.norm(curve_diff) / max(np.linalg.norm(ref_curve), 1.0e-300)
        )
        row["breakthrough_abs_l1_error"] = float(np.trapezoid(np.abs(curve_diff), ref_time))
        row["concentration_l2_abs_fem_integral_vs_fem"] = mass_l2_absolute(
            field_diff, mass_matrix
        )
        row["concentration_rel_l2_fem_integral_vs_fem"] = mass_l2_relative(
            field_diff, ref_field, mass_matrix
        )
        row["concentration_linf_abs_vs_fem"] = float(np.linalg.norm(field_diff, ord=np.inf))
        row["final_mass_error_vs_fem"] = float(row["final_tracer_mass"] - ref_mass)
        row["final_mass_rel_error_vs_fem"] = float(
            (row["final_tracer_mass"] - ref_mass) / max(abs(float(ref_mass)), 1.0e-300)
        )
        row["t10_error"] = signed_time_error(row["t10"], ref_row["t10"])
        row["t50_error"] = signed_time_error(row["t50"], ref_row["t50"])
        row["t90_error"] = signed_time_error(row["t90"], ref_row["t90"])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2f3640",
            "axes.labelcolor": "#1f2933",
            "xtick.color": "#1f2933",
            "ytick.color": "#1f2933",
            "grid.color": "#d8dee9",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def plot_breakthrough(out: Path, histories: dict[str, dict[str, np.ndarray]]) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=220)
    for method, history in histories.items():
        ax.plot(
            history["time"],
            history["cout"],
            color=COLORS.get(method, "#555555"),
            linestyle="-" if method == "FEM" else "--",
            linewidth=2.0,
            label=method,
        )
    ax.set_xlabel("time")
    ax.set_ylabel("flux-weighted outlet concentration")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.7)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_mass_balance(out: Path, histories: dict[str, dict[str, np.ndarray]]) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=220)
    for method, history in histories.items():
        color = COLORS.get(method, "#555555")
        axes[0].plot(history["time"], history["mass"], color=color, linewidth=2.0, label=f"{method} mass")
        axes[0].plot(
            history["time"], history["budget_mass"], color=color, linewidth=1.4, linestyle=":", label=f"{method} budget"
        )
        axes[1].semilogy(
            history["time"][1:],
            np.maximum(np.abs(history["mass_balance_relative_residual"][1:]), 1.0e-18),
            color=color,
            linewidth=2.0,
            label=method,
        )
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("total tracer mass")
    axes[0].grid(True, alpha=0.7)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("relative mass-balance residual")
    axes[1].grid(True, which="both", alpha=0.7)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_error_summary(out: Path, rows: list[dict[str, object]]) -> None:
    set_plot_style()
    methods = [str(row["method"]) for row in rows if row["method"] != "FEM"]
    if not methods:
        return
    concentration = [float(row["concentration_rel_l2_fem_integral_vs_fem"]) for row in rows if row["method"] != "FEM"]
    breakthrough = [float(row["breakthrough_rel_l2_error"]) for row in rows if row["method"] != "FEM"]
    mass = [abs(float(row["final_mass_rel_error_vs_fem"])) for row in rows if row["method"] != "FEM"]
    balance = [float(row["max_mass_balance_relative_residual"]) for row in rows if row["method"] != "FEM"]
    x = np.arange(len(methods))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=220)
    ax.bar(x - 1.5 * width, concentration, width, label="field L2", color="#457b9d")
    ax.bar(x - 0.5 * width, breakthrough, width, label="breakthrough L2", color="#e76f51")
    ax.bar(x + 0.5 * width, mass, width, label="final mass", color="#2a9d8f")
    ax.bar(x + 1.5 * width, balance, width, label="mass balance", color="#6d597a")
    ax.set_yscale("log")
    ax.set_xticks(x, methods)
    ax.set_ylabel("relative error")
    ax.grid(True, axis="y", which="both", alpha=0.7)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_final_concentration(
    out: Path,
    points: np.ndarray,
    cells: np.ndarray,
    final_fields: dict[str, np.ndarray],
    methods: list[str],
    reference: str = "FEM",
) -> None:
    """Isosurface panels of the final concentration per method plus the
    log10 absolute error against the FEM-driven reference."""
    panels = list(methods)
    if "Affine-DDPNM-9" in final_fields and reference in final_fields:
        panels.append("log10 error")

    grid = build_tracer_plot_grid(points, cells)
    for method in methods:
        grid.point_data[f"concentration_{method}"] = np.asarray(final_fields[method], dtype=float)
    if "Affine-DDPNM-9" in final_fields and reference in final_fields:
        error = np.abs(
            np.asarray(final_fields["Affine-DDPNM-9"], dtype=float) - np.asarray(final_fields[reference], dtype=float)
        )
        grid.point_data["log10_concentration_error"] = np.log10(np.maximum(error, 1.0e-16))
    shell = plot_extract_surface(grid).smooth(
        n_iter=50, relaxation_factor=0.06, feature_smoothing=False, boundary_smoothing=True
    )

    pv.global_theme.font.family = "arial"
    plotter = pv.Plotter(off_screen=True, window_size=(640 * len(panels), 820), shape=(1, len(panels)), border=False)
    plotter.set_background("white")
    for index, panel in enumerate(panels):
        plotter.subplot(0, index)
        if panel == "log10 error":
            render_tracer_isosurface_panel(
                plotter, grid, shell, "log10_concentration_error", "Affine-DDPNM-9 error",
                "log10 error", "turbo", tracer_error_range(grid.point_data["log10_concentration_error"]),
            )
        else:
            render_tracer_isosurface_panel(
                plotter, grid, shell, f"concentration_{panel}", panel, "c", "viridis", (0.0, 1.0),
            )
    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()


def build_tracer_plot_grid(points: np.ndarray, cells: np.ndarray) -> pv.UnstructuredGrid:
    cells_flat = np.hstack([np.full((len(cells), 1), 4, dtype=np.int64), np.asarray(cells, dtype=np.int64)]).ravel()
    celltypes = np.full(len(cells), pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells_flat, celltypes, np.asarray(points, dtype=float))


def render_tracer_isosurface_panel(
    plotter: pv.Plotter,
    grid,
    shell,
    scalars: str,
    title: str,
    scalar_bar_title: str,
    cmap: str,
    clim: tuple[float, float],
) -> None:
    add_tracer_domain_box(plotter, grid.bounds)
    shell_opacity = 0.12 if scalar_bar_title == "c" else 0.30
    plotter.add_mesh(shell, color="#787878", opacity=shell_opacity, smooth_shading=True, show_edges=False)
    scalar_bar_args = {
        "title": scalar_bar_title,
        "vertical": True,
        "position_x": 0.87,
        "position_y": 0.18,
        "width": 0.045,
        "height": 0.60,
        "title_font_size": 15,
        "label_font_size": 12,
        "fmt": "%.1f",
        "color": "black",
    }
    contour = grid.contour(isosurfaces=tracer_iso_levels(clim, scalar_bar_title), scalars=scalars)
    if contour.n_points:
        render_mesh = plot_extract_surface(contour).smooth(
            n_iter=90, relaxation_factor=0.06, feature_smoothing=False, boundary_smoothing=True
        )
        opacity = 0.74 if scalar_bar_title == "c" else 0.60
    else:
        render_mesh = grid
        opacity = 0.55
    plotter.add_mesh(
        render_mesh, scalars=scalars, cmap=cmap, clim=clim, opacity=opacity,
        smooth_shading=True, show_edges=False, scalar_bar_args=scalar_bar_args,
    )
    plotter.add_text(title, position=(0.035, 0.925), font_size=14, color="black", viewport=True)
    set_tracer_paper_camera(plotter, grid.bounds)


def tracer_iso_levels(clim: tuple[float, float], scalar_bar_title: str = "") -> np.ndarray:
    lo, hi = clim
    if hi <= lo:
        hi = lo + 1.0
    if scalar_bar_title == "c":
        return np.linspace(0.25, 0.95, 6)
    return np.linspace(lo + 0.24 * (hi - lo), hi - 0.10 * (hi - lo), 5)


def tracer_error_range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if len(finite) == 0:
        return -16.0, -8.0
    lo = float(np.percentile(finite, 5.0))
    hi = float(np.percentile(finite, 99.2))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def plot_extract_surface(dataset):
    try:
        return dataset.extract_surface(algorithm="dataset_surface")
    except TypeError:
        return dataset.extract_surface()


def add_tracer_domain_box(plotter: pv.Plotter, bounds: tuple) -> None:
    cube = plot_extract_surface(pv.Cube(bounds=bounds))
    plotter.add_mesh(cube, color="#eeeeee", opacity=0.08, show_edges=False)


def set_tracer_paper_camera(plotter: pv.Plotter, bounds: tuple) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))
    scale = max(xmax - xmin, ymax - ymin, zmax - zmin)
    plotter.camera_position = [
        (center[0] + 4.10 * scale, center[1] - 4.00 * scale, center[2] + 3.10 * scale),
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.45)
    plotter.add_light(pv.Light(position=(center[0], center[1] - 2.0 * scale, center[2] + 4.0 * scale), intensity=0.55))
    plotter.add_light(pv.Light(position=(center[0] - 2.0 * scale, center[1] + 1.5 * scale, center[2] + 2.5 * scale), intensity=0.35))

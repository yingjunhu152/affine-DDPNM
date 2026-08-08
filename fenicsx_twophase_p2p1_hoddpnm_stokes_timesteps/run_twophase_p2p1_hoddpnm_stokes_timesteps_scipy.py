from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import MatrixRankWarning, lsmr, minres, splu, spsolve
from scipy.spatial import cKDTree


import run_cube_holes_hoddpnm_validation as cube_mesh_source  # noqa: E402


cube_mesh_source.orient_tets = lambda coords, cells: stable_orient_tets(coords, cells)
P2_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
QUAD_LAM = np.asarray(
    [
        [0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105],
        [0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105],
        [0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105],
        [0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685],
    ],
    dtype=float,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holes-per-axis", type=int, default=2)
    parser.add_argument("--cells-per-axis", type=int, default=4)
    parser.add_argument("--domain-size", type=float, default=5.0)
    parser.add_argument("--radius", type=float, default=0.34)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--frame-every", type=int, default=3)
    parser.add_argument("--frame-steps", type=int, nargs="*", default=None)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--mu-original", type=float, default=1.0)
    parser.add_argument("--mu-injected", type=float, default=4.0)
    parser.add_argument("--residual-original", type=float, default=0.05)
    parser.add_argument("--residual-injected", type=float, default=0.05)
    parser.add_argument("--transport-scale", type=float, default=0.18)
    parser.add_argument("--geometry-channel-strength", type=float, default=0.0)
    parser.add_argument("--capillary-spread", type=float, default=0.0)
    parser.add_argument("--render-geometry-strength", type=float, default=0.0)
    parser.add_argument("--render-front-contrast", type=float, default=1.0)
    parser.add_argument("--linear-solver", choices=("minres", "lsmr", "direct"), default="minres")
    parser.add_argument("--direct-residual-rtol", type=float, default=1.0e-8)
    parser.add_argument("--minres-rtol", type=float, default=1.0e-10)
    parser.add_argument("--minres-maxiter", type=int, default=20000)
    parser.add_argument("--minres-residual-rtol", type=float, default=2.0e-6)
    parser.add_argument("--lsmr-atol", type=float, default=1.0e-10)
    parser.add_argument("--lsmr-btol", type=float, default=1.0e-10)
    parser.add_argument("--lsmr-maxiter", type=int, default=4000)
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    parser.add_argument("--render-resolution", type=int, default=86)
    parser.add_argument("--write-vtu", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/twophase_p2p1_hoddpnm_stokes_timesteps_scipy"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mesh = cube_mesh_source.build_cube_minus_spheres_mesh(
        holes_per_axis=args.holes_per_axis,
        cells_per_axis=args.cells_per_axis,
        domain_size=args.domain_size,
        radius=args.radius,
        seed=args.seed,
    )
    p2 = build_p2_nodes(mesh.coords, mesh.cells)
    graph_edges = mesh_vertex_edges(mesh.cells)
    lumped_volume = vertex_lumped_volume(mesh.coords, mesh.cells)
    frame_steps = selected_frame_steps(args.steps, args.frame_every, args.frame_steps)

    saturation = np.full(len(mesh.coords), args.residual_original, dtype=float)
    inlet_vertices = np.flatnonzero(np.isclose(mesh.coords[:, 0], 0.0))
    saturation[inlet_vertices] = 1.0 - args.residual_injected

    snapshots: dict[int, dict[str, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    if 0 in frame_steps:
        snapshots[0] = {
            "saturation": saturation.copy(),
            "velocity": np.zeros((len(mesh.coords), 3), dtype=float),
            "pressure": np.zeros(len(mesh.coords), dtype=float),
        }

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        A = assemble_p2p1_stokes(mesh.coords, mesh.cells, p2, saturation, args)
        assemble_time = time.perf_counter() - t0
        fixed, values = p2p1_dirichlet(mesh.coords, mesh.cells, p2, args.domain_size)

        t0 = time.perf_counter()
        full_result = solve_dirichlet(A, fixed, values, args)
        full = full_result["solution"]
        full_time = time.perf_counter() - t0

        hodd = solve_hodd_schur(A, fixed, values, p2.velocity_coords, len(mesh.coords), args)
        velocity_full, pressure_full = split_solution(full, p2, len(mesh.coords))
        velocity_hodd, pressure_hodd = split_solution(hodd["solution"], p2, len(mesh.coords))
        vertex_velocity = velocity_hodd[: len(mesh.coords)]

        saturation = update_saturation(mesh, graph_edges, lumped_volume, saturation, vertex_velocity, args)
        saturation[inlet_vertices] = 1.0 - args.residual_injected

        vel_diff = vector_error(velocity_hodd[: len(mesh.coords)] - velocity_full[: len(mesh.coords)], velocity_full[: len(mesh.coords)])
        pres_diff = scalar_error(pressure_hodd - pressure_full, pressure_full)
        row = {
            "step": int(step),
            "mean_injected_saturation": float(np.mean(saturation)),
            "assemble_time_seconds": float(assemble_time),
            "full_stokes_solve_time_seconds": float(full_time),
            "hoddpnm_schur_solve_time_seconds": float(hodd["time"]),
            "velocity_l2_rel_difference_vs_full_discrete": vel_diff["l2_rel"],
            "pressure_l2_rel_difference_vs_full_discrete": pres_diff["l2_rel"],
            "full_linear_solve": full_result["linear_solve"],
            "schur_linear_solve": hodd["schur_linear_solve"],
            "p2_velocity_nodes": int(len(p2.velocity_coords)),
            "pressure_nodes": int(len(mesh.coords)),
            "mixed_dofs": int(3 * len(p2.velocity_coords) + len(mesh.coords)),
            "hoddpnm_interface_dofs": int(np.count_nonzero(hodd["interface_mask"])),
            "hoddpnm_eliminated_dofs": int(hodd["n_interior"]),
        }
        rows.append(row)
        print(
            f"step {step:04d}: mean S2={row['mean_injected_saturation']:.4f}, "
            f"full={full_time:.4f}s, hodd={hodd['time']:.4f}s, "
            f"u_schur_diff={row['velocity_l2_rel_difference_vs_full_discrete']:.2e}, "
            f"p_schur_diff={row['pressure_l2_rel_difference_vs_full_discrete']:.2e}"
        )

        if step in frame_steps:
            snapshots[step] = {"saturation": saturation.copy(), "velocity": vertex_velocity.copy(), "pressure": pressure_hodd.copy()}

    for step in frame_steps:
        if args.write_vtu:
            save_vtu(mesh, snapshots[step], args.out_dir / f"twophase_p2p1_hoddpnm_stokes_step_{step:04d}.vtu")
        if args.plot:
            render_frame(
                mesh,
                snapshots[step],
                args.render_resolution,
                args,
                args.out_dir / f"voidspace_two_phase_volume_split_clean_step_{step:04d}.png",
            )

    summary = {
        "method": "two-phase P2-P1 Taylor-Hood Stokes Schur/HODDPNM consistency demo, pure SciPy assembly",
        "equation": "-div(mu(S2) grad(u)) + grad(p) = 0, div(u) = 0 for the single-phase Stokes solve at each saturation state",
        "discretization": "P2 velocity on tetrahedron vertices+edge midpoints, P1 pressure on vertices",
        "linear_solver": args.linear_solver,
        "error_scope": "reported velocity/pressure differences compare the Schur-reconstructed solution against the full solve of the same assembled discrete linear system; they are not physical errors against a true 3D two-phase Stokes reference solution",
        "transport_scope": "S2 is advanced by an explicit graph-edge demonstration transport model driven by the Schur velocity; this is not a verified conservative two-phase flow discretization",
        "dependency_scope": "mesh helper is imported from the copied local run_cube_holes_hoddpnm_validation.py file in this folder; no absolute source tree path is injected",
        "holes": int(args.holes_per_axis**3),
        "cells_per_axis": int(args.cells_per_axis),
        "vertices": int(len(mesh.coords)),
        "tets": int(len(mesh.cells)),
        "p2_velocity_nodes": int(len(p2.velocity_coords)),
        "pressure_nodes": int(len(mesh.coords)),
        "steps": int(args.steps),
        "frame_steps": [int(s) for s in frame_steps],
        "history": rows,
    }
    (args.out_dir / "twophase_p2p1_hoddpnm_stokes_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.out_dir / 'twophase_p2p1_hoddpnm_stokes_report.json'}")


class P2Data:
    def __init__(self, velocity_coords: np.ndarray, p2_cells: np.ndarray, edge_to_node: dict[tuple[int, int], int]):
        self.velocity_coords = velocity_coords
        self.p2_cells = p2_cells
        self.edge_to_node = edge_to_node


def build_p2_nodes(coords: np.ndarray, cells: np.ndarray) -> P2Data:
    velocity_coords = [tuple(row) for row in coords]
    edge_to_node: dict[tuple[int, int], int] = {}
    p2_cells = []
    for tet in cells:
        local = [int(v) for v in tet]
        for a, b in P2_EDGES:
            edge = tuple(sorted((int(tet[a]), int(tet[b]))))
            if edge not in edge_to_node:
                edge_to_node[edge] = len(velocity_coords)
                velocity_coords.append(tuple(0.5 * (coords[edge[0]] + coords[edge[1]])))
            local.append(edge_to_node[edge])
        p2_cells.append(local)
    return P2Data(np.asarray(velocity_coords, dtype=float), np.asarray(p2_cells, dtype=np.int64), edge_to_node)


def assemble_p2p1_stokes(coords: np.ndarray, cells: np.ndarray, p2: P2Data, saturation: np.ndarray, args) -> csc_matrix:
    n_v = len(p2.velocity_coords)
    n_p = len(coords)
    p_offset = 3 * n_v
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    cell_sat = np.mean(saturation[cells], axis=1)
    for cell_id, tet in enumerate(cells):
        volume, grad_lam = tet_volume_and_grad_lambda(coords[tet])
        mu = effective_viscosity(float(cell_sat[cell_id]), args)
        local_v = p2.p2_cells[cell_id]
        for lam in QUAD_LAM:
            weight = volume / len(QUAD_LAM)
            grad_phi = p2_gradients(lam, grad_lam)
            for a in range(10):
                for b in range(10):
                    stiffness = mu * weight * dot3(grad_phi[a], grad_phi[b])
                    for comp in range(3):
                        rows.append(3 * int(local_v[a]) + comp)
                        cols.append(3 * int(local_v[b]) + comp)
                        vals.append(stiffness)
            for q_local in range(4):
                q_val = float(lam[q_local])
                p_dof = p_offset + int(tet[q_local])
                for a in range(10):
                    for comp in range(3):
                        div_entry = weight * q_val * float(grad_phi[a, comp])
                        u_dof = 3 * int(local_v[a]) + comp
                        rows.append(p_dof)
                        cols.append(u_dof)
                        vals.append(-div_entry)
                        rows.append(u_dof)
                        cols.append(p_dof)
                        vals.append(-div_entry)
    return coo_matrix((vals, (rows, cols)), shape=(p_offset + n_p, p_offset + n_p)).tocsc()


def p2_gradients(lam: np.ndarray, grad_lam: np.ndarray) -> np.ndarray:
    grad = np.zeros((10, 3), dtype=float)
    for i in range(4):
        grad[i] = (4.0 * lam[i] - 1.0) * grad_lam[i]
    for pos, (i, j) in enumerate(P2_EDGES, start=4):
        grad[pos] = 4.0 * (lam[i] * grad_lam[j] + lam[j] * grad_lam[i])
    return grad


def p2p1_dirichlet(coords: np.ndarray, cells: np.ndarray, p2: P2Data, domain_size: float) -> tuple[np.ndarray, np.ndarray]:
    n_v = len(p2.velocity_coords)
    n_p = len(coords)
    mask = np.zeros(3 * n_v + n_p, dtype=bool)
    values = np.zeros(3 * n_v + n_p, dtype=float)
    boundary_nodes = p2_boundary_nodes(cells, p2)
    for node, x in enumerate(p2.velocity_coords):
        on_cube = np.isclose(x[0], 0.0) or np.isclose(x[1], 0.0) or np.isclose(x[2], 0.0) or np.isclose(x[1], domain_size) or np.isclose(x[2], domain_size)
        on_outlet = np.isclose(x[0], domain_size)
        on_boundary = on_cube or node in boundary_nodes
        if on_boundary and not on_outlet:
            mask[3 * node : 3 * node + 3] = True
            if np.isclose(x[0], 0.0):
                values[3 * node] = 1.0
    outlet_pressure_nodes = np.flatnonzero(np.isclose(coords[:, 0], domain_size))
    pin = int(outlet_pressure_nodes[0]) if len(outlet_pressure_nodes) else int(np.argmax(coords[:, 0]))
    mask[3 * n_v + pin] = True
    return mask, values


def p2_boundary_nodes(cells: np.ndarray, p2: P2Data) -> set[int]:
    face_count: dict[tuple[int, int, int], int] = {}
    for tet in cells:
        for face in ((tet[0], tet[1], tet[2]), (tet[0], tet[1], tet[3]), (tet[0], tet[2], tet[3]), (tet[1], tet[2], tet[3])):
            key = tuple(sorted(int(v) for v in face))
            face_count[key] = face_count.get(key, 0) + 1
    boundary: set[int] = set()
    for face, count in face_count.items():
        if count != 1:
            continue
        a, b, c = face
        boundary.update([a, b, c])
        for edge in (tuple(sorted((a, b))), tuple(sorted((a, c))), tuple(sorted((b, c)))):
            if edge in p2.edge_to_node:
                boundary.add(p2.edge_to_node[edge])
    return boundary


def solve_dirichlet(A: csc_matrix, fixed: np.ndarray, values: np.ndarray, args) -> dict[str, object]:
    free = np.flatnonzero(~fixed)
    fixed_idx = np.flatnonzero(fixed)
    rhs = -A[free][:, fixed_idx] @ values[fixed_idx]
    Aff = A[free][:, free].tocsc()
    sol = values.copy()
    sol_free, stats = checked_linear_solve(
        Aff,
        rhs,
        "full_dirichlet_free_system",
        args,
    )
    sol[free] = sol_free
    return {"solution": sol, "linear_solve": stats}


def checked_linear_solve(A, rhs: np.ndarray, label: str, args) -> tuple[np.ndarray, dict[str, object]]:
    if args.linear_solver == "minres":
        return checked_minres(A, rhs, label, args.minres_rtol, args.minres_maxiter, args.minres_residual_rtol)
    if args.linear_solver == "direct":
        return checked_direct_solve(A, rhs, label, args.direct_residual_rtol)
    return checked_lsmr(A, rhs, label, atol=args.lsmr_atol, btol=args.lsmr_btol, maxiter=args.lsmr_maxiter)


def checked_minres(A, rhs: np.ndarray, label: str, rtol: float, maxiter: int, residual_rtol: float) -> tuple[np.ndarray, dict[str, object]]:
    sol, info = minres(A, rhs, rtol=rtol, maxiter=maxiter, check=False)
    residual = np.asarray(A @ sol - rhs, dtype=float)
    rel_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0))
    stats = {
        "label": label,
        "solver": "minres",
        "info": int(info),
        "rtol": float(rtol),
        "maxiter": int(maxiter),
        "relative_residual": rel_residual,
        "residual_rtol": float(residual_rtol),
    }
    if info != 0 or rel_residual > residual_rtol:
        raise RuntimeError(f"minres did not converge for {label}: {stats}")
    return np.asarray(sol, dtype=float), stats


def checked_direct_solve(A, rhs: np.ndarray, label: str, residual_rtol: float) -> tuple[np.ndarray, dict[str, object]]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        sol = spsolve(A, rhs)
    if not np.all(np.isfinite(sol)):
        raise RuntimeError(f"direct solve produced non-finite values for {label}")
    residual = np.asarray(A @ sol - rhs, dtype=float)
    rel_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0))
    stats = {
        "label": label,
        "solver": "direct_spsolve",
        "relative_residual": rel_residual,
        "residual_rtol": float(residual_rtol),
    }
    if rel_residual > residual_rtol:
        raise RuntimeError(f"direct solve residual too large for {label}: {stats}")
    return np.asarray(sol, dtype=float), stats


def checked_lsmr(A, rhs: np.ndarray, label: str, atol: float, btol: float, maxiter: int) -> tuple[np.ndarray, dict[str, object]]:
    result = lsmr(A, rhs, atol=atol, btol=btol, maxiter=maxiter)
    sol = result[0]
    stats = {
        "label": label,
        "solver": "lsmr",
        "istop": int(result[1]),
        "iterations": int(result[2]),
        "normr": float(result[3]),
        "normar": float(result[4]),
        "norma": float(result[5]),
        "conda": float(result[6]),
        "normx": float(result[7]),
        "atol": float(atol),
        "btol": float(btol),
        "maxiter": int(maxiter),
    }
    if stats["istop"] not in (1, 2):
        raise RuntimeError(f"lsmr did not converge for {label}: {stats}")
    return sol, stats


def solve_hodd_schur(A: csc_matrix, fixed: np.ndarray, values: np.ndarray, velocity_coords: np.ndarray, n_pressure: int, args) -> dict[str, object]:
    t0 = time.perf_counter()
    n_v = len(velocity_coords)
    n_total = A.shape[0]
    free = np.flatnonzero(~fixed)
    h = float(velocity_coords[:, 0].max() - velocity_coords[:, 0].min()) / args.cells_per_axis
    idx = np.rint(velocity_coords / h).astype(int)
    planes = {args.cells_per_axis // 3, 2 * args.cells_per_axis // 3}
    v_interface = np.isin(idx[:, 0], list(planes)) | np.isin(idx[:, 1], list(planes)) | np.isin(idx[:, 2], list(planes))
    interface_mask = np.zeros(n_total, dtype=bool)
    for comp in range(3):
        interface_mask[3 * np.flatnonzero(v_interface) + comp] = True
    interface_mask[3 * n_v :] = True
    interface_mask &= ~fixed
    interface = np.flatnonzero(interface_mask)
    interior = np.setdiff1d(free, interface, assume_unique=True)
    Aff = A[free][:, free]
    rhs_free = -A[free][:, np.flatnonzero(fixed)] @ values[np.flatnonzero(fixed)]
    pos = {int(g): i for i, g in enumerate(free)}
    ii = np.asarray([pos[int(g)] for g in interior], dtype=int)
    bb = np.asarray([pos[int(g)] for g in interface], dtype=int)
    Aii = Aff[ii][:, ii].tocsc()
    Aib = Aff[ii][:, bb].tocsc()
    Abi = Aff[bb][:, ii].tocsc()
    Abb = Aff[bb][:, bb].tocsc()
    ri = np.asarray(rhs_free[ii], dtype=float)
    rb = np.asarray(rhs_free[bb], dtype=float)
    lu = splu(Aii)
    yi = lu.solve(ri)
    dense = Aib.toarray()
    Xi = np.column_stack([lu.solve(dense[:, j]) for j in range(dense.shape[1])])
    S = Abb.toarray()
    for j in range(Xi.shape[1]):
        S[:, j] -= Abi @ Xi[:, j]
    schur_rhs = rb - Abi @ yi
    ub, schur_linear_solve = checked_linear_solve(
        csc_matrix(S),
        schur_rhs,
        "hoddpnm_schur_boundary_system",
        args,
    )
    ui = yi - dense_matvec(Xi, ub)
    sol = values.copy()
    sol[interface] = ub
    sol[interior] = ui
    return {
        "solution": sol,
        "interface_mask": interface_mask,
        "n_interior": int(len(interior)),
        "time": time.perf_counter() - t0,
        "schur_linear_solve": schur_linear_solve,
    }


def split_solution(sol: np.ndarray, p2: P2Data, n_pressure: int) -> tuple[np.ndarray, np.ndarray]:
    n_v = len(p2.velocity_coords)
    return sol[: 3 * n_v].reshape((-1, 3)), sol[3 * n_v : 3 * n_v + n_pressure]


def update_saturation(mesh, edges, lumped_volume, saturation, velocity, args) -> np.ndarray:
    ds = np.zeros_like(saturation)
    for i, j in edges:
        delta = mesh.coords[j] - mesh.coords[i]
        midpoint = 0.5 * (mesh.coords[i] + mesh.coords[j])
        channel_gain = 1.0 + args.geometry_channel_strength * near_sphere_surface_weight(midpoint, mesh.centers, mesh.radius)
        q = args.transport_scale * channel_gain * dot3(0.5 * (velocity[i] + velocity[j]), delta)
        if q >= 0:
            donor, receiver = i, j
        else:
            donor, receiver = j, i
            q = -q
        ds[donor] -= q * saturation[donor] / lumped_volume[donor]
        ds[receiver] += q * saturation[donor] / lumped_volume[receiver]
        spread = args.capillary_spread * channel_gain * max(float(saturation[donor] - saturation[receiver]), 0.0)
        ds[donor] -= spread / lumped_volume[donor]
        ds[receiver] += spread / lumped_volume[receiver]
    return np.clip(saturation + args.dt * ds, args.residual_original, 1.0 - args.residual_injected)


def near_sphere_surface_weight(points: np.ndarray, centers: np.ndarray, radius: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    flat = points.reshape((-1, 3))
    distance = np.min(np.linalg.norm(flat[:, None, :] - centers[None, :, :], axis=2), axis=1)
    width = max(0.55 * radius, 1.0e-12)
    weight = np.exp(-((distance - radius) / width) ** 2)
    return weight.reshape(points.shape[:-1])


def render_frame(mesh, data: dict[str, np.ndarray], resolution: int, args, out: Path) -> None:
    grid = make_volume_grid(mesh, data, resolution, args)
    pv.global_theme.font.family = "arial"
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1800, 760), border=False)
    plotter.set_background("white")
    for idx, (field, cmap, title) in enumerate(
        [
            ("phase1", "Blues", "original phase S1"),
            ("phase2", "autumn_r", "injected phase S2"),
        ]
    ):
        plotter.subplot(0, idx)
        render_one_phase(plotter, grid, mesh, field, cmap, title, show_bar=(idx == 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()


def make_volume_grid(mesh, data: dict[str, np.ndarray], resolution: int, args) -> pv.ImageData:
    n = int(resolution)
    spacing = mesh.domain_size / (n - 1)
    grid = pv.ImageData(
        dimensions=(n, n, n),
        spacing=(spacing, spacing, spacing),
        origin=(0.0, 0.0, 0.0),
    )
    pts = grid.points
    dist_to_spheres = np.min(np.linalg.norm(pts[:, None, :] - mesh.centers[None, :, :], axis=2), axis=1)
    void_mask = dist_to_spheres > mesh.radius * 1.02
    s2 = interpolate_to_points(mesh.coords, data["saturation"], pts)
    s2 = np.asarray(s2, dtype=float)
    if args.render_geometry_strength > 0.0:
        near_wall = near_sphere_surface_weight(pts, mesh.centers, mesh.radius)
        downstream = np.clip(pts[:, 0] / max(mesh.domain_size, 1.0e-12), 0.0, 1.0)
        s2 = args.residual_original + (s2 - args.residual_original) * (1.0 + args.render_geometry_strength * near_wall * (0.35 + downstream))
    if args.render_front_contrast != 1.0:
        s2 = args.residual_original + (np.clip(s2, 0.0, 1.0) - args.residual_original) * args.render_front_contrast
    s2 = np.clip(s2, 0.0, 1.0)
    s2[~void_mask] = 0.0
    grid.point_data["phase2"] = s2
    grid.point_data["phase1"] = np.where(void_mask, 1.0 - s2, 0.0)
    return grid


def interpolate_to_points(nodes: np.ndarray, values: np.ndarray, points: np.ndarray) -> np.ndarray:
    tree = cKDTree(nodes)
    distances, idx = tree.query(points, k=min(8, len(nodes)))
    distances = np.maximum(distances, 1.0e-8)
    weights = 1.0 / distances**2
    return np.sum(weights * values[idx], axis=1) / np.sum(weights, axis=1)


def render_one_phase(
    plotter: pv.Plotter,
    grid: pv.ImageData,
    mesh,
    field: str,
    cmap: str,
    title: str,
    show_bar: bool,
) -> None:
    cube = pv.Cube(bounds=(0, mesh.domain_size, 0, mesh.domain_size, 0, mesh.domain_size)).extract_surface(
        algorithm="dataset_surface"
    )
    plotter.add_mesh(cube, color="#eeeeee", opacity=0.085, show_edges=False)

    sphere_mesh = unit_sphere_mesh(theta_resolution=36, phi_resolution=20)
    for center in mesh.centers:
        sphere = scale_translate_polydata(sphere_mesh, mesh.radius, center)
        plotter.add_mesh(sphere, color="#bdbdbd", opacity=0.22, smooth_shading=True, show_edges=False)

    plotter.add_volume(
        grid,
        scalars=field,
        cmap=cmap,
        clim=(0.0, 1.0),
        opacity=opacity_curve(),
        shade=False,
        show_scalar_bar=False,
    )
    phase = grid.contour(isosurfaces=[0.35, 0.58, 0.78], scalars=field)
    plotter.add_mesh(
        phase,
        scalars=field,
        cmap=cmap,
        clim=(0.0, 1.0),
        opacity=0.34,
        smooth_shading=True,
        show_edges=False,
        scalar_bar_args={
            "title": "saturation",
            "vertical": True,
            "position_x": 0.88,
            "position_y": 0.18,
            "width": 0.035,
            "height": 0.62,
            "fmt": "%.1f",
            "color": "black",
        } if show_bar else None,
        show_scalar_bar=show_bar,
    )

    plotter.add_text(title, position="upper_left", font_size=12, color="black")
    plotter.camera_position = [(8.4, -8.2, 6.4), (2.5, 2.5, 2.5), (0.0, 0.0, 1.0)]
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.08)


def unit_sphere_mesh(theta_resolution: int = 36, phi_resolution: int = 20) -> pv.PolyData:
    theta = np.linspace(0.0, 2.0 * np.pi, theta_resolution, endpoint=False)
    phi = np.linspace(0.0, np.pi, phi_resolution)
    points = []
    for p in phi:
        sp = np.sin(p)
        cp = np.cos(p)
        for t in theta:
            points.append((sp * np.cos(t), sp * np.sin(t), cp))
    faces = []
    for i in range(phi_resolution - 1):
        for j in range(theta_resolution):
            a = i * theta_resolution + j
            b = i * theta_resolution + (j + 1) % theta_resolution
            c = (i + 1) * theta_resolution + (j + 1) % theta_resolution
            d = (i + 1) * theta_resolution + j
            faces.extend([4, a, b, c, d])
    return pv.PolyData(np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int64))


def scale_translate_polydata(mesh: pv.PolyData, radius: float, center: np.ndarray) -> pv.PolyData:
    out = mesh.copy(deep=True)
    out.points = out.points * radius + np.asarray(center, dtype=float)
    return out


def opacity_curve(n: int = 256) -> np.ndarray:
    control = np.asarray([0.0, 0.0, 0.006, 0.018, 0.045, 0.085], dtype=float)
    xo = np.linspace(0.0, 1.0, len(control))
    xx = np.linspace(0.0, 1.0, n)
    return np.asarray(np.interp(xx, xo, control) * 255.0, dtype=np.uint8)


def make_grid(mesh, data: dict[str, np.ndarray]) -> pv.UnstructuredGrid:
    cells = np.column_stack((np.full(len(mesh.cells), 4, dtype=np.int64), mesh.cells)).ravel()
    celltypes = np.full(len(mesh.cells), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, mesh.coords)
    grid.point_data["saturation_S2"] = data["saturation"]
    grid.point_data["velocity"] = data["velocity"]
    grid.point_data["velocity_magnitude"] = np.linalg.norm(data["velocity"], axis=1)
    grid.point_data["pressure"] = data["pressure"]
    return grid


def save_vtu(mesh, data: dict[str, np.ndarray], out: Path) -> None:
    grid = make_grid(mesh, data)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)


def tet_volume_and_grad_lambda(x: np.ndarray) -> tuple[float, np.ndarray]:
    a = x[1] - x[0]
    b = x[2] - x[0]
    c = x[3] - x[0]
    det = dot3(a, np.cross(b, c))
    volume = abs(det) / 6.0
    grads = np.empty((4, 3), dtype=float)
    grads[1] = np.cross(b, c) / det
    grads[2] = np.cross(c, a) / det
    grads[3] = np.cross(a, b) / det
    grads[0] = -grads[1] - grads[2] - grads[3]
    return volume, grads


def effective_viscosity(s: float, args) -> float:
    inv_mu = (1.0 - s) / max(args.mu_original, 1.0e-12) + s / max(args.mu_injected, 1.0e-12)
    return 1.0 / max(inv_mu, 1.0e-12)


def mesh_vertex_edges(cells: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for tet in cells:
        for a in range(4):
            for b in range(a + 1, 4):
                edges.add(tuple(sorted((int(tet[a]), int(tet[b])))))
    return np.asarray(sorted(edges), dtype=np.int64)


def vertex_lumped_volume(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    vol = np.zeros(len(coords), dtype=float)
    for tet in cells:
        x = coords[tet]
        volume = abs(dot3(x[1] - x[0], np.cross(x[2] - x[0], x[3] - x[0]))) / 6.0
        vol[tet] += volume / 4.0
    return np.maximum(vol, 1.0e-12)


def selected_frame_steps(total_steps: int, every: int, explicit: list[int] | None) -> list[int]:
    if explicit:
        return sorted({int(step) for step in explicit if 0 <= step <= total_steps})
    steps = list(range(0, total_steps + 1, max(1, int(every))))
    if steps[-1] != total_steps:
        steps.append(total_steps)
    return steps


def stable_orient_tets(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    oriented = cells.copy()
    for i, tet in enumerate(oriented):
        det = dot3(coords[tet[1]] - coords[tet[0]], np.cross(coords[tet[2]] - coords[tet[0]], coords[tet[3]] - coords[tet[0]]))
        if det < 0.0:
            oriented[i, [2, 3]] = oriented[i, [3, 2]]
    return oriented


def scalar_error(diff: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    return {"l2_rel": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1.0e-300)), "linf_abs": float(np.linalg.norm(diff, ord=np.inf))}


def vector_error(diff: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    return {"l2_rel": float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(ref.ravel()), 1.0e-300)), "linf_abs": float(np.linalg.norm(diff, axis=1).max())}


def dense_matvec(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    out = np.zeros(mat.shape[0], dtype=float)
    for i in range(mat.shape[0]):
        total = 0.0
        for j in range(mat.shape[1]):
            total += float(mat[i, j]) * float(vec[j])
        out[i] = total
    return out


def dot3(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


if __name__ == "__main__":
    main()

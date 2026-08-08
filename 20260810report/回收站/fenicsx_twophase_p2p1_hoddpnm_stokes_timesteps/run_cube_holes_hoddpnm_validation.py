from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu, spsolve


@dataclass
class MeshData:
    coords: np.ndarray
    cells: np.ndarray
    centers: np.ndarray
    radius: float
    cells_per_axis: int
    domain_size: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holes-per-axis", type=int, default=3)
    parser.add_argument("--cells-per-axis", type=int, default=15)
    parser.add_argument("--domain-size", type=float, default=5.0)
    parser.add_argument("--radius", type=float, default=0.34)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cube_holes_27_validation"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mesh = build_cube_minus_spheres_mesh(
        holes_per_axis=args.holes_per_axis,
        cells_per_axis=args.cells_per_axis,
        domain_size=args.domain_size,
        radius=args.radius,
        seed=args.seed,
    )
    A = assemble_stokes(mesh.coords, mesh.cells)
    dirichlet, values = stokes_dirichlet_dofs(mesh.coords, mesh.cells, mesh.domain_size)
    full = solve_dirichlet_system(A, dirichlet, values)
    hodd = solve_hoddpnm_schur(A, mesh.coords, dirichlet, values, args.cells_per_axis)

    diff = hodd["solution"] - full["solution"]
    free = ~dirichlet
    interface = hodd["interface_mask"]
    fem_velocity = full["solution"][: 3 * len(mesh.coords)].reshape((-1, 3))
    hodd_velocity = hodd["solution"][: 3 * len(mesh.coords)].reshape((-1, 3))
    fem_pressure = full["solution"][3 * len(mesh.coords) :]
    hodd_pressure = hodd["solution"][3 * len(mesh.coords) :]
    velocity_diff = hodd_velocity - fem_velocity
    pressure_diff = hodd_pressure - fem_pressure
    velocity_interface = interface[: 3 * len(mesh.coords)].reshape((-1, 3)).any(axis=1)
    pressure_interface = interface[3 * len(mesh.coords) :]
    data = {
        "method": "exact-static-condensation-HODDPNM-Stokes-on-cube-minus-spheres",
        "equation": "stabilized P1-P1 Stokes",
        "holes": int(args.holes_per_axis**3),
        "cells_per_axis": args.cells_per_axis,
        "n_vertices": int(len(mesh.coords)),
        "n_tets": int(len(mesh.cells)),
        "mixed_dofs": int(len(full["solution"])),
        "velocity_dofs": int(3 * len(mesh.coords)),
        "pressure_dofs": int(len(mesh.coords)),
        "fem_free_dofs": int(np.count_nonzero(free)),
        "hoddpnm_interface_dofs": int(np.count_nonzero(interface)),
        "hoddpnm_interior_dofs_eliminated": int(hodd["n_interior"]),
        "hoddpnm_schur_condition_number": float(np.linalg.cond(hodd["schur_matrix"])),
        "errors_all_free_mixed": error_stats(diff[free], full["solution"][free]),
        "errors_velocity_nodes": vector_error_stats(velocity_diff, fem_velocity),
        "errors_pressure_nodes": error_stats(pressure_diff, fem_pressure),
        "errors_velocity_interface": vector_error_stats(
            velocity_diff[velocity_interface],
            fem_velocity[velocity_interface],
        ),
        "errors_pressure_interface": error_stats(
            pressure_diff[pressure_interface],
            fem_pressure[pressure_interface],
        ),
        "pressure_range": {
            "fem_min": float(fem_pressure.min()),
            "fem_max": float(fem_pressure.max()),
            "hoddpnm_min": float(hodd_pressure.min()),
            "hoddpnm_max": float(hodd_pressure.max()),
        },
        "velocity_range": {
            "fem_norm_min": float(np.linalg.norm(fem_velocity, axis=1).min()),
            "fem_norm_max": float(np.linalg.norm(fem_velocity, axis=1).max()),
            "hoddpnm_norm_min": float(np.linalg.norm(hodd_velocity, axis=1).min()),
            "hoddpnm_norm_max": float(np.linalg.norm(hodd_velocity, axis=1).max()),
        },
        "sphere_centers": mesh.centers.tolist(),
        "sphere_radius": mesh.radius,
    }
    (args.out_dir / "validation_summary.json").write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
    save_vtk(mesh, full["solution"], hodd["solution"], diff, args.out_dir / "cube_holes_solution.vtu")
    plot_pyvista(mesh, full["solution"], diff, args.out_dir / "cube_holes_hoddpnm_error.png")

    print(f"holes: {data['holes']}")
    print(f"vertices: {data['n_vertices']}")
    print(f"tets: {data['n_tets']}")
    print(f"FEM free dofs: {data['fem_free_dofs']}")
    print(f"HODDPNM interface dofs: {data['hoddpnm_interface_dofs']}")
    print(f"HODDPNM eliminated interior dofs: {data['hoddpnm_interior_dofs_eliminated']}")
    print(f"Schur condition number: {data['hoddpnm_schur_condition_number']:.3e}")
    print(f"mixed free L2 rel error: {data['errors_all_free_mixed']['l2_rel']:.3e}")
    print(f"velocity node L2 rel error: {data['errors_velocity_nodes']['l2_rel']:.3e}")
    print(f"pressure node L2 rel error: {data['errors_pressure_nodes']['l2_rel']:.3e}")
    print(f"wrote {args.out_dir / 'validation_summary.json'}")
    print(f"wrote {args.out_dir / 'cube_holes_solution.vtu'}")
    print(f"wrote {args.out_dir / 'cube_holes_hoddpnm_error.png'}")


def build_cube_minus_spheres_mesh(
    holes_per_axis: int,
    cells_per_axis: int,
    domain_size: float,
    radius: float,
    seed: int,
) -> MeshData:
    centers = irregular_centers(holes_per_axis, domain_size, radius, seed)
    axis = np.linspace(0.0, domain_size, cells_per_axis + 1)
    all_coords = np.asarray(
        [
            (axis[i], axis[j], axis[k])
            for i in range(cells_per_axis + 1)
            for j in range(cells_per_axis + 1)
            for k in range(cells_per_axis + 1)
        ],
        dtype=float,
    )
    kept: list[list[int]] = []
    for i in range(cells_per_axis):
        for j in range(cells_per_axis):
            for k in range(cells_per_axis):
                centroid = np.asarray(
                    [
                        0.5 * (axis[i] + axis[i + 1]),
                        0.5 * (axis[j] + axis[j + 1]),
                        0.5 * (axis[k] + axis[k + 1]),
                    ],
                    dtype=float,
                )
                if np.any(np.linalg.norm(centers - centroid, axis=1) <= radius):
                    continue
                v000 = grid_id(i, j, k, cells_per_axis)
                v001 = grid_id(i, j, k + 1, cells_per_axis)
                v010 = grid_id(i, j + 1, k, cells_per_axis)
                v011 = grid_id(i, j + 1, k + 1, cells_per_axis)
                v100 = grid_id(i + 1, j, k, cells_per_axis)
                v101 = grid_id(i + 1, j, k + 1, cells_per_axis)
                v110 = grid_id(i + 1, j + 1, k, cells_per_axis)
                v111 = grid_id(i + 1, j + 1, k + 1, cells_per_axis)
                kept.extend(
                    [
                        [v000, v001, v011, v111],
                        [v000, v011, v010, v111],
                        [v000, v010, v110, v111],
                        [v000, v110, v100, v111],
                        [v000, v100, v101, v111],
                        [v000, v101, v001, v111],
                    ]
                )
    raw_cells = np.asarray(kept, dtype=np.int64)
    used = np.unique(raw_cells.ravel())
    remap = -np.ones(len(all_coords), dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    coords = all_coords[used]
    cells = keep_largest_cell_component(orient_tets(coords, remap[raw_cells]))
    used = np.unique(cells.ravel())
    remap2 = -np.ones(len(coords), dtype=np.int64)
    remap2[used] = np.arange(len(used), dtype=np.int64)
    coords = coords[used]
    cells = remap2[cells]
    return MeshData(coords, cells, centers, radius, cells_per_axis, domain_size)


def irregular_centers(n: int, domain_size: float, radius: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    margin = max(0.55, 1.35 * radius)
    target = n**3
    centers: list[np.ndarray] = []
    min_distance = 2.45 * radius
    attempts = 0
    while len(centers) < target and attempts < 200000:
        attempts += 1
        point = rng.uniform(margin, domain_size - margin, size=3)
        if centers and min(np.linalg.norm(point - center) for center in centers) < min_distance:
            continue
        centers.append(point)
    if len(centers) != target:
        raise RuntimeError(f"Could only place {len(centers)} non-overlapping spheres out of {target}.")
    return np.asarray(centers, dtype=float)


def assemble_stokes(
    coords: np.ndarray,
    cells: np.ndarray,
    viscosity: float = 1.0,
    pressure_stabilization: float = 1.0e-3,
) -> csc_matrix:
    rows = []
    cols = []
    vals = []
    n_vertices = len(coords)
    pressure_offset = 3 * n_vertices
    for tet in cells:
        x = coords[tet]
        mat = np.ones((4, 4), dtype=float)
        mat[:, 1:] = x
        det = np.linalg.det(mat)
        volume = abs(det) / 6.0
        inv = np.linalg.inv(mat)
        grads = inv[1:, :].T
        ke = viscosity * volume * (grads @ grads.T)
        mp = pressure_stabilization * volume / 20.0 * (np.ones((4, 4)) + np.eye(4))
        for a in range(4):
            for b in range(4):
                for comp in range(3):
                    rows.append(3 * int(tet[a]) + comp)
                    cols.append(3 * int(tet[b]) + comp)
                    vals.append(float(ke[a, b]))

                p_row = pressure_offset + int(tet[a])
                p_col = pressure_offset + int(tet[b])
                rows.append(p_row)
                cols.append(p_col)
                vals.append(float(-mp[a, b]))

                for comp in range(3):
                    div_entry = volume * grads[b, comp] / 4.0
                    u_dof = 3 * int(tet[b]) + comp
                    p_dof = pressure_offset + int(tet[a])
                    rows.append(p_dof)
                    cols.append(u_dof)
                    vals.append(float(-div_entry))
                    rows.append(u_dof)
                    cols.append(p_dof)
                    vals.append(float(-div_entry))
    n = 4 * len(coords)
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()


def stokes_dirichlet_dofs(
    coords: np.ndarray,
    cells: np.ndarray,
    domain_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    tol = 1.0e-12
    n_vertices = len(coords)
    boundary_nodes = exterior_nodes(cells)
    inlet = np.isclose(coords[:, 0], 0.0, atol=tol)
    outlet = np.isclose(coords[:, 0], domain_size, atol=tol)
    wall = boundary_nodes & ~outlet

    mask = np.zeros(4 * n_vertices, dtype=bool)
    values = np.zeros(4 * n_vertices, dtype=float)
    for node in np.flatnonzero(wall):
        mask[3 * node : 3 * node + 3] = True
    for node in np.flatnonzero(inlet):
        mask[3 * node] = True
        mask[3 * node + 1] = True
        mask[3 * node + 2] = True
        values[3 * node] = 1.0
    pressure_pin = 3 * n_vertices + int(np.flatnonzero(outlet)[0])
    mask[pressure_pin] = True
    values[pressure_pin] = 0.0
    return mask, values


def exterior_nodes(cells: np.ndarray) -> np.ndarray:
    face_count: dict[tuple[int, int, int], int] = {}
    for tet in cells:
        for face in (
            (tet[0], tet[1], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]),
            (tet[1], tet[2], tet[3]),
        ):
            key = tuple(sorted(int(v) for v in face))
            face_count[key] = face_count.get(key, 0) + 1
    n = int(cells.max()) + 1
    mask = np.zeros(n, dtype=bool)
    for face, count in face_count.items():
        if count == 1:
            mask[list(face)] = True
    return mask


def solve_dirichlet_system(A: csc_matrix, dirichlet: np.ndarray, values: np.ndarray) -> dict[str, np.ndarray]:
    free = np.flatnonzero(~dirichlet)
    fixed = np.flatnonzero(dirichlet)
    rhs = -A[free][:, fixed] @ values[fixed]
    sol_free = spsolve(A[free][:, free], rhs)
    sol = values.copy()
    sol[free] = sol_free
    return {"solution": sol}


def solve_hoddpnm_schur(
    A: csc_matrix,
    coords: np.ndarray,
    dirichlet: np.ndarray,
    values: np.ndarray,
    cells_per_axis: int,
) -> dict[str, object]:
    free = np.flatnonzero(~dirichlet)
    fixed = np.flatnonzero(dirichlet)
    n_vertices = len(coords)
    h = float(coords[:, 0].max() - coords[:, 0].min()) / cells_per_axis
    indices = np.rint(coords / h).astype(int)
    planes = {cells_per_axis // 3, 2 * cells_per_axis // 3}
    node_interface = (
        np.isin(indices[:, 0], list(planes))
        | np.isin(indices[:, 1], list(planes))
        | np.isin(indices[:, 2], list(planes))
    )
    interface_global_mask = np.zeros(4 * n_vertices, dtype=bool)
    for comp in range(3):
        interface_global_mask[3 * np.flatnonzero(node_interface) + comp] = True
    interface_global_mask[3 * n_vertices :] = node_interface
    interface_global_mask &= ~dirichlet
    interface = np.flatnonzero(interface_global_mask)
    interior = np.setdiff1d(free, interface, assume_unique=True)

    Aff = A[free][:, free]
    rhs_free = -A[free][:, fixed] @ values[fixed]
    free_pos = {int(g): i for i, g in enumerate(free)}
    i_pos = np.asarray([free_pos[int(g)] for g in interior], dtype=int)
    b_pos = np.asarray([free_pos[int(g)] for g in interface], dtype=int)

    Aii = Aff[i_pos][:, i_pos].tocsc()
    Aib = Aff[i_pos][:, b_pos].tocsc()
    Abi = Aff[b_pos][:, i_pos].tocsc()
    Abb = Aff[b_pos][:, b_pos].tocsc()
    ri = np.asarray(rhs_free[i_pos], dtype=float)
    rb = np.asarray(rhs_free[b_pos], dtype=float)

    lu = splu(Aii)
    x_i_rhs = lu.solve(ri)
    x_i_cols = lu.solve(Aib.toarray())
    schur = Abb.toarray() - Abi @ x_i_cols
    schur_rhs = rb - Abi @ x_i_rhs
    ub = np.linalg.solve(schur, schur_rhs)
    ui = x_i_rhs - x_i_cols @ ub

    sol = values.copy()
    sol[interface] = ub
    sol[interior] = ui
    return {
        "solution": sol,
        "interface_mask": interface_global_mask,
        "n_interior": len(interior),
        "schur_matrix": schur,
    }


def error_stats(diff: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    return {
        "l2_abs": float(np.linalg.norm(diff)),
        "l2_rel": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1.0e-300)),
        "linf_abs": float(np.linalg.norm(diff, ord=np.inf)),
        "mean_abs": float(np.mean(np.abs(diff))),
        "median_abs": float(np.median(np.abs(diff))),
    }


def vector_error_stats(diff: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff_norm = np.linalg.norm(diff, axis=1)
    ref_norm = np.linalg.norm(ref, axis=1)
    return {
        "l2_abs": float(np.linalg.norm(diff.ravel())),
        "l2_rel": float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(ref.ravel()), 1.0e-300)),
        "linf_abs": float(diff_norm.max()),
        "linf_rel": float(diff_norm.max() / max(ref_norm.max(), 1.0e-300)),
        "mean_abs": float(np.mean(diff_norm)),
        "median_abs": float(np.median(diff_norm)),
    }


def save_vtk(mesh: MeshData, fem: np.ndarray, hodd: np.ndarray, diff: np.ndarray, out: Path) -> None:
    cells = np.column_stack((np.full(len(mesh.cells), 4, dtype=np.int64), mesh.cells)).ravel()
    celltypes = np.full(len(mesh.cells), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, mesh.coords)
    n = len(mesh.coords)
    fem_velocity = fem[: 3 * n].reshape((-1, 3))
    hodd_velocity = hodd[: 3 * n].reshape((-1, 3))
    fem_pressure = fem[3 * n :]
    hodd_pressure = hodd[3 * n :]
    grid.point_data["fem_velocity"] = fem_velocity
    grid.point_data["hoddpnm_velocity"] = hodd_velocity
    grid.point_data["velocity_magnitude"] = np.linalg.norm(fem_velocity, axis=1)
    grid.point_data["velocity_abs_error"] = np.linalg.norm(hodd_velocity - fem_velocity, axis=1)
    grid.point_data["fem_pressure"] = fem_pressure
    grid.point_data["hoddpnm_pressure"] = hodd_pressure
    grid.point_data["pressure_abs_error"] = np.abs(hodd_pressure - fem_pressure)
    grid.point_data["abs_error"] = np.sqrt(
        grid.point_data["velocity_abs_error"] ** 2 + grid.point_data["pressure_abs_error"] ** 2
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)


def plot_pyvista(mesh: MeshData, fem: np.ndarray, diff: np.ndarray, out: Path) -> None:
    cells = np.column_stack((np.full(len(mesh.cells), 4, dtype=np.int64), mesh.cells)).ravel()
    celltypes = np.full(len(mesh.cells), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, mesh.coords)
    n = len(mesh.coords)
    velocity_diff = diff[: 3 * n].reshape((-1, 3))
    pressure_diff = diff[3 * n :]
    grid.point_data["abs_error"] = np.sqrt(np.linalg.norm(velocity_diff, axis=1) ** 2 + pressure_diff**2)
    surf = grid.extract_surface().smooth(n_iter=25, relaxation_factor=0.05)
    clipped = grid.clip(normal=(0, -1, 0), origin=(2.5, 2.55, 2.5)).extract_surface().smooth(
        n_iter=25,
        relaxation_factor=0.05,
    )
    vmax = max(float(np.percentile(np.abs(diff), 99.5)), 1.0e-14)
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1500, 560), border=False)
    plotter.set_background("white")
    bar = {
        "title": "|HODDPNM - FEM|",
        "vertical": True,
        "position_x": 0.88,
        "position_y": 0.18,
        "width": 0.05,
        "height": 0.62,
        "fmt": "%.1e",
        "color": "black",
    }
    for idx, obj in enumerate([surf, clipped]):
        plotter.subplot(0, idx)
        plotter.add_mesh(
            obj,
            scalars="abs_error",
            cmap="viridis",
            clim=(0.0, vmax),
            smooth_shading=True,
            show_edges=False,
            opacity=0.65 if idx == 0 else 1.0,
            scalar_bar_args=bar if idx == 1 else None,
            show_scalar_bar=idx == 1,
        )
        plotter.camera_position = [(8.0, -7.6, 5.3), (2.5, 2.5, 2.5), (0.0, 0.0, 1.0)]
        plotter.camera.zoom(1.28 if idx == 0 else 1.38)
        plotter.enable_parallel_projection()
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()


def grid_id(i: int, j: int, k: int, n: int) -> int:
    return (i * (n + 1) + j) * (n + 1) + k


def orient_tets(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    out = cells.copy()
    for i, tet in enumerate(out):
        mat = np.column_stack((coords[tet[1]] - coords[tet[0]], coords[tet[2]] - coords[tet[0]], coords[tet[3]] - coords[tet[0]]))
        if np.linalg.det(mat) < 0.0:
            out[i, [2, 3]] = out[i, [3, 2]]
    return out


def keep_largest_cell_component(cells: np.ndarray) -> np.ndarray:
    face_to_cells: dict[tuple[int, int, int], list[int]] = {}
    for ci, tet in enumerate(cells):
        faces = (
            (tet[0], tet[1], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]),
            (tet[1], tet[2], tet[3]),
        )
        for face in faces:
            key = tuple(sorted(int(v) for v in face))
            face_to_cells.setdefault(key, []).append(ci)
    adjacency = [[] for _ in range(len(cells))]
    for owners in face_to_cells.values():
        if len(owners) < 2:
            continue
        for a in owners:
            for b in owners:
                if a != b:
                    adjacency[a].append(b)
    seen = np.zeros(len(cells), dtype=bool)
    components: list[list[int]] = []
    for start in range(len(cells)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            item = stack.pop()
            comp.append(item)
            for nb in adjacency[item]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        components.append(comp)
    largest = max(components, key=len)
    return cells[np.asarray(largest, dtype=np.int64)]


if __name__ == "__main__":
    main()

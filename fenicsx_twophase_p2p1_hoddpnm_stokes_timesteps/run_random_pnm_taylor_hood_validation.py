from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI
from scipy.sparse.linalg import LinearOperator, gmres, spilu, splu, spsolve
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.distance import cdist

from classic_ddpnm.linalg import to_numpy_vector, to_scipy_matrix
from run_cube_holes_hoddpnm_validation import (
    MeshData,
    build_cube_minus_spheres_mesh,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holes-per-axis", type=int, default=3)
    parser.add_argument("--cells-per-axis", type=int, default=9)
    parser.add_argument("--domain-size", type=float, default=5.0)
    parser.add_argument("--radius", type=float, default=0.34)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--target-degree", type=int, default=4)
    parser.add_argument("--max-edge-length", type=float, default=2.35)
    parser.add_argument("--pressure-stabilization", type=float, default=1.0e-10)
    parser.add_argument("--pressure-boundary-mode", choices=("interface-anchors", "all"), default="interface-anchors")
    parser.add_argument("--schur-solver", choices=("gmres", "dense"), default="gmres")
    parser.add_argument("--schur-rtol", type=float, default=1.0e-7)
    parser.add_argument("--schur-atol", type=float, default=0.0)
    parser.add_argument("--schur-maxiter", type=int, default=600)
    parser.add_argument("--schur-restart", type=int, default=80)
    parser.add_argument("--schur-preconditioner", choices=("ilu", "diag", "none"), default="ilu")
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/random_27_pnm_taylor_hood_validation"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mesh = build_cube_minus_spheres_mesh(
        holes_per_axis=args.holes_per_axis,
        cells_per_axis=args.cells_per_axis,
        domain_size=args.domain_size,
        radius=args.radius,
        seed=args.seed,
    )
    pnm_edges = select_pnm_edges(mesh.centers, args.target_degree, args.max_edge_length)
    total_timer = time.perf_counter()
    assembly_timer = time.perf_counter()
    assembled = assemble_taylor_hood_system(mesh, pnm_edges, args)
    assembly_time = time.perf_counter() - assembly_timer
    full_timer = time.perf_counter()
    fem_solution = spsolve(assembled["A"], assembled["b"])
    full_solve_time = time.perf_counter() - full_timer
    hodd_timer = time.perf_counter()
    hodd = solve_schur_hoddpnm(
        assembled["A"],
        assembled["b"],
        assembled["interface_mask"],
        assembled["fixed_dofs"],
        args,
    )
    hoddpnm_solve_time = time.perf_counter() - hodd_timer
    total_time = time.perf_counter() - total_timer
    hodd_solution = hodd["solution"]
    diff = hodd_solution - fem_solution

    vertex_fields = vertex_output_fields(mesh, assembled, fem_solution, hodd_solution)
    velocity_error_all = diff[assembled["mapV"]].reshape((-1, 3))
    pressure_error_all = diff[assembled["mapQ"]]
    velocity_ref_all = fem_solution[assembled["mapV"]].reshape((-1, 3))
    pressure_ref_all = fem_solution[assembled["mapQ"]]
    hodd_pressure_all = hodd_solution[assembled["mapQ"]]
    pressure_error_aligned = mean_aligned_difference(hodd_solution[assembled["mapQ"]], fem_solution[assembled["mapQ"]])
    fem_pressure_zero_mean = pressure_ref_all - np.mean(pressure_ref_all)
    hodd_pressure_zero_mean = hodd_pressure_all - np.mean(hodd_pressure_all)

    data = {
        "method": "FEniCSx Taylor-Hood P2-P1 Stokes HODDPNM via Schur complement",
        "equation": "-Delta u + grad p = 0, div u = 0",
        "assembly": "FEniCSx/dolfinx UFL assembly with Basix P2 vector velocity and P1 scalar pressure",
        "error_scope": "Schur/full consistency for the same FEniCSx-assembled discrete matrix; not a physical error against a higher-accuracy Stokes or two-phase reference",
        "reduction_scope": "known Dirichlet dofs are removed first; the active Schur boundary contains only free PNM-interface velocity dofs and free retained pressure dofs, while free interior dofs are statically reconstructed",
        "timestep_scope": "steady Stokes reduction validation only; this script is not the two-phase timestep solver",
        "conditioning_scope": "large Schur condition numbers and large absolute pressure ranges indicate a numerically ill-conditioned reduced pressure system; small relative Schur/full differences should be interpreted together with these scale diagnostics",
        "pressure_stabilization": args.pressure_stabilization,
        "pressure_boundary_mode": args.pressure_boundary_mode,
        "schur_solver": hodd["schur_solver"],
        "schur_preconditioner": hodd["schur_preconditioner"],
        "dense_schur_used": bool(hodd["dense_schur_used"]),
        "schur_iterations": hodd["schur_iterations"],
        "schur_relative_residual": hodd["schur_relative_residual"],
        "assembly_time_seconds": float(assembly_time),
        "full_solve_time_seconds": float(full_solve_time),
        "hoddpnm_solve_time_seconds": float(hoddpnm_solve_time),
        "total_time_seconds": float(total_time),
        "holes": int(args.holes_per_axis**3),
        "cells_per_axis": args.cells_per_axis,
        "n_vertices": int(len(mesh.coords)),
        "n_tets": int(len(mesh.cells)),
        "mixed_dofs": int(len(fem_solution)),
        "velocity_p2_scalar_nodes": int(len(assembled["V_coords"])),
        "velocity_dofs": int(len(assembled["mapV"])),
        "pressure_dofs": int(len(assembled["mapQ"])),
        "velocity_interface_dofs": int(assembled["interface_stats"]["velocity_interface_dofs"]),
        "pressure_boundary_dofs": int(assembled["interface_stats"]["pressure_boundary_dofs"]),
        "pressure_interface_dofs": int(assembled["interface_stats"]["pressure_interface_dofs"]),
        "pressure_anchor_dofs": int(assembled["interface_stats"]["pressure_anchor_dofs"]),
        "pressure_interior_dofs_eliminated": int(assembled["interface_stats"]["pressure_interior_dofs_eliminated"]),
        "hoddpnm_interface_dofs": int(np.count_nonzero(assembled["interface_mask"])),
        "hoddpnm_active_schur_dofs": int(hodd["n_boundary"]),
        "hoddpnm_active_dof_ratio": float(hodd["n_boundary"] / len(fem_solution)),
        "hoddpnm_known_fixed_dofs": int(hodd["n_fixed_known"]),
        "hoddpnm_free_interior_dofs_eliminated": int(hodd["n_interior"]),
        "hoddpnm_interior_dofs_eliminated": int(hodd["n_interior"]),
        "pnm_edges": int(len(pnm_edges)),
        "pnm_target_degree": args.target_degree,
        "pnm_max_edge_length": args.max_edge_length,
        "schur_condition_number": None if hodd["schur_matrix"] is None else float(np.linalg.cond(hodd["schur_matrix"])),
        "errors_mixed_all": scalar_error_stats(diff, fem_solution),
        "errors_velocity_p2_all": vector_error_stats(velocity_error_all, velocity_ref_all),
        "errors_pressure_p1_all": scalar_error_stats(pressure_error_all, pressure_ref_all),
        "errors_pressure_p1_mean_aligned": scalar_error_stats(pressure_error_aligned, pressure_ref_all - np.mean(pressure_ref_all)),
        "errors_velocity_vertices": vector_error_stats(
            vertex_fields["hodd_velocity"] - vertex_fields["fem_velocity"],
            vertex_fields["fem_velocity"],
        ),
        "errors_pressure_vertices": scalar_error_stats(
            vertex_fields["hodd_pressure"] - vertex_fields["fem_pressure"],
            vertex_fields["fem_pressure"],
        ),
        "velocity_range": {
            "fem_vertex_min": float(np.linalg.norm(vertex_fields["fem_velocity"], axis=1).min()),
            "fem_vertex_max": float(np.linalg.norm(vertex_fields["fem_velocity"], axis=1).max()),
        },
        "pressure_range": {
            "fem_min": float(vertex_fields["fem_pressure"].min()),
            "fem_max": float(vertex_fields["fem_pressure"].max()),
            "fem_mean_p1": float(np.mean(pressure_ref_all)),
            "hodd_mean_p1": float(np.mean(hodd_pressure_all)),
            "fem_zero_mean_min_p1": float(fem_pressure_zero_mean.min()),
            "fem_zero_mean_max_p1": float(fem_pressure_zero_mean.max()),
            "hodd_zero_mean_min_p1": float(hodd_pressure_zero_mean.min()),
            "hodd_zero_mean_max_p1": float(hodd_pressure_zero_mean.max()),
        },
        "sphere_centers": mesh.centers.tolist(),
        "sphere_radius": mesh.radius,
    }
    (args.out_dir / "validation_summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    save_vtu(mesh, vertex_fields, args.out_dir / "cube_holes_taylor_hood_solution.vtu")
    if args.plot:
        plot_error_cloud(args.out_dir / "cube_holes_taylor_hood_solution.vtu", args.out_dir / "cube_holes_taylor_hood_error.png")

    print(f"holes: {data['holes']}")
    print(f"vertices: {data['n_vertices']}")
    print(f"tets: {data['n_tets']}")
    print(f"mixed dofs: {data['mixed_dofs']}")
    print(f"P2 velocity dofs: {data['velocity_dofs']}")
    print(f"P1 pressure dofs: {data['pressure_dofs']}")
    print(f"velocity interface dofs: {data['velocity_interface_dofs']}")
    print(f"pressure boundary dofs: {data['pressure_boundary_dofs']}")
    print(f"pressure eliminated interior dofs: {data['pressure_interior_dofs_eliminated']}")
    print(f"HODDPNM interface dofs before fixed removal: {data['hoddpnm_interface_dofs']}")
    print(f"HODDPNM active Schur dofs: {data['hoddpnm_active_schur_dofs']}")
    print(f"HODDPNM active ratio: {data['hoddpnm_active_dof_ratio']:.1%}")
    print(f"HODDPNM known fixed dofs removed: {data['hoddpnm_known_fixed_dofs']}")
    print(f"eliminated free interior dofs: {data['hoddpnm_free_interior_dofs_eliminated']}")
    print(f"Schur solver: {data['schur_solver']}")
    print(f"Schur preconditioner: {data['schur_preconditioner']}")
    print(f"dense Schur used: {data['dense_schur_used']}")
    print(f"Schur iterations: {data['schur_iterations']}")
    print(f"Schur relative residual: {data['schur_relative_residual']:.3e}")
    print(f"assembly time: {data['assembly_time_seconds']:.4f} s")
    print(f"full solve time: {data['full_solve_time_seconds']:.4f} s")
    print(f"HODDPNM solve time: {data['hoddpnm_solve_time_seconds']:.4f} s")
    print(f"total time: {data['total_time_seconds']:.4f} s")
    if data["schur_condition_number"] is not None:
        print(f"Schur condition number: {data['schur_condition_number']:.3e}")
    print(f"pressure range: [{data['pressure_range']['fem_min']:.3e}, {data['pressure_range']['fem_max']:.3e}]")
    print(f"pressure mean: {data['pressure_range']['fem_mean_p1']:.3e}")
    print(
        "zero-mean pressure range: "
        f"[{data['pressure_range']['fem_zero_mean_min_p1']:.3e}, "
        f"{data['pressure_range']['fem_zero_mean_max_p1']:.3e}]"
    )
    print(f"velocity P2 L2 rel error: {data['errors_velocity_p2_all']['l2_rel']:.3e}")
    print(f"pressure P1 L2 rel error: {data['errors_pressure_p1_all']['l2_rel']:.3e}")
    print(f"pressure mean-aligned L2 rel error: {data['errors_pressure_p1_mean_aligned']['l2_rel']:.3e}")
    print(f"mixed L2 rel error: {data['errors_mixed_all']['l2_rel']:.3e}")
    print(f"wrote {args.out_dir / 'validation_summary.json'}")
    print(f"wrote {args.out_dir / 'cube_holes_taylor_hood_solution.vtu'}")
    if args.plot:
        print(f"wrote {args.out_dir / 'cube_holes_taylor_hood_error.png'}")


def assemble_taylor_hood_system(
    mesh: MeshData,
    pnm_edges: list[tuple[int, int]],
    args,
) -> dict[str, object]:
    domain_ufl = ufl.Mesh(element("Lagrange", "tetrahedron", 1, shape=(3,)))
    msh = dmesh.create_mesh(MPI.COMM_SELF, mesh.cells, domain_ufl, mesh.coords)
    cell = msh.basix_cell()
    velocity_el = element("Lagrange", cell, 2, shape=(3,))
    pressure_el = element("Lagrange", cell, 1)
    W = fem.functionspace(msh, mixed_element([velocity_el, pressure_el]))
    V, mapV = W.sub(0).collapse()
    Q, mapQ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a = fem.form(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
        - fem.Constant(msh, args.pressure_stabilization) * p * q * ufl.dx
    )
    L = fem.form(fem.Constant(msh, 0.0) * q * ufl.dx)

    bcs, fixed_dofs = taylor_hood_bcs(
        msh,
        W,
        V,
        Q,
        mapQ,
        mesh.domain_size,
        mesh.centers,
        mesh.radius,
        mesh.domain_size / mesh.cells_per_axis,
    )
    A = to_scipy_matrix(fem.assemble_matrix(a, bcs=bcs))
    b = to_numpy_vector(fem.assemble_vector(L))
    fem.apply_lifting(b, [a], [bcs])
    fem.set_bc(b, bcs)

    interface_mask, interface_stats = taylor_hood_interface_mask(
        W.dofmap.index_map.size_local * W.dofmap.index_map_bs,
        np.asarray(mapV, dtype=np.int64),
        V.tabulate_dof_coordinates(),
        np.asarray(mapQ, dtype=np.int64),
        Q.tabulate_dof_coordinates(),
        mesh.coords,
        mesh.cells,
        mesh.domain_size,
        mesh.cells_per_axis,
        mesh.centers,
        pnm_edges,
        args.pressure_boundary_mode,
    )
    return {
        "A": A,
        "b": b,
        "interface_mask": interface_mask,
        "fixed_dofs": fixed_dofs,
        "mapV": np.asarray(mapV, dtype=np.int64),
        "mapQ": np.asarray(mapQ, dtype=np.int64),
        "V_coords": V.tabulate_dof_coordinates(),
        "Q_coords": Q.tabulate_dof_coordinates(),
        "interface_stats": interface_stats,
    }


def taylor_hood_bcs(
    msh,
    W,
    V,
    Q,
    mapQ: np.ndarray,
    domain_size: float,
    centers: np.ndarray,
    radius: float,
    h: float,
):
    def inlet(x):
        return np.isclose(x[0], 0.0)

    def wall(x):
        points = x.T
        nearest_sphere = np.min(np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2), axis=1)
        on_sphere_wall = np.abs(nearest_sphere - radius) <= 0.95 * h
        on_cube_wall = (
            np.isclose(x[1], 0.0)
            | np.isclose(x[1], domain_size)
            | np.isclose(x[2], 0.0)
            | np.isclose(x[2], domain_size)
            | np.isclose(x[0], 0.0)
        )
        return on_cube_wall | on_sphere_wall

    u_wall = fem.Function(V)
    u_wall.x.array[:] = 0.0
    u_in = fem.Function(V)
    u_in.x.array[:] = 0.0
    u_in.x.array.reshape((-1, 3))[:, 0] = 1.0

    wall_dofs = fem.locate_dofs_geometrical((W.sub(0), V), wall)
    inlet_dofs = fem.locate_dofs_geometrical((W.sub(0), V), inlet)
    bcs = [
        fem.dirichletbc(u_wall, wall_dofs, W.sub(0)),
        fem.dirichletbc(u_in, inlet_dofs, W.sub(0)),
    ]

    q_coords = Q.tabulate_dof_coordinates()
    outlet_candidates = np.flatnonzero(np.isclose(q_coords[:, 0], domain_size))
    if len(outlet_candidates) == 0:
        outlet_candidates = np.asarray([int(np.argmax(q_coords[:, 0]))], dtype=int)
    target = np.asarray([domain_size, 0.5 * domain_size, 0.5 * domain_size])
    q_local = int(outlet_candidates[np.argmin(np.linalg.norm(q_coords[outlet_candidates] - target, axis=1))])
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    pressure_dofs = [np.asarray([int(mapQ[q_local])], dtype=np.int32), np.asarray([q_local], dtype=np.int32)]
    bcs.append(fem.dirichletbc(p0, pressure_dofs, W.sub(1)))

    fixed = []
    for dofs in (wall_dofs, inlet_dofs):
        fixed.extend(np.asarray(dofs[0], dtype=np.int64).tolist())
    fixed.append(int(mapQ[q_local]))
    return bcs, np.unique(np.asarray(fixed, dtype=np.int64))


def taylor_hood_interface_mask(
    n_mixed: int,
    mapV: np.ndarray,
    V_coords: np.ndarray,
    mapQ: np.ndarray,
    Q_coords: np.ndarray,
    mesh_coords: np.ndarray,
    mesh_cells: np.ndarray,
    domain_size: float,
    cells_per_axis: int,
    centers: np.ndarray,
    pnm_edges: list[tuple[int, int]],
    pressure_boundary_mode: str,
) -> tuple[np.ndarray, dict[str, int | str]]:
    mask = np.zeros(n_mixed, dtype=bool)
    h = domain_size / cells_per_axis
    v_node_mask = pnm_interface_nodes(V_coords, centers, pnm_edges, h)
    v_dof_mask = expand_vector_mask(v_node_mask, len(mapV), vector_dim=3)
    q_interface_mask = pnm_interface_nodes(Q_coords, centers, pnm_edges, h)
    if pressure_boundary_mode == "all":
        q_anchor_mask = np.zeros(len(Q_coords), dtype=bool)
        q_node_mask = np.ones(len(Q_coords), dtype=bool)
    else:
        q_anchor_mask = pressure_component_anchors(Q_coords, mesh_coords, mesh_cells, q_interface_mask)
        q_node_mask = q_interface_mask | q_anchor_mask
    mask[mapV] = v_dof_mask
    mask[mapQ] = q_node_mask
    stats = {
        "pressure_boundary_mode": pressure_boundary_mode,
        "velocity_interface_dofs": int(np.count_nonzero(v_dof_mask)),
        "pressure_boundary_dofs": int(np.count_nonzero(q_node_mask)),
        "pressure_interface_dofs": int(np.count_nonzero(q_interface_mask)),
        "pressure_anchor_dofs": int(np.count_nonzero(q_anchor_mask)),
        "pressure_interior_dofs_eliminated": int(len(Q_coords) - np.count_nonzero(q_node_mask)),
    }
    return mask, stats


def expand_vector_mask(node_mask: np.ndarray, n_dofs: int, vector_dim: int) -> np.ndarray:
    if len(node_mask) == n_dofs:
        return node_mask
    if len(node_mask) * vector_dim == n_dofs:
        return np.repeat(node_mask, vector_dim)
    raise ValueError(
        f"Cannot expand velocity interface mask: {len(node_mask)} coordinate masks for {n_dofs} velocity dofs."
    )


def pressure_component_anchors(
    Q_coords: np.ndarray,
    mesh_coords: np.ndarray,
    mesh_cells: np.ndarray,
    pressure_interface_mask: np.ndarray,
) -> np.ndarray:
    vertex_to_q = cKDTree(Q_coords).query(mesh_coords, k=1)[1].astype(np.int64)
    neighbors: list[set[int]] = [set() for _ in range(len(Q_coords))]
    for tet in mesh_cells:
        q_tet = [int(vertex_to_q[int(v)]) for v in tet]
        for a in range(4):
            for b in range(a + 1, 4):
                ia, ib = q_tet[a], q_tet[b]
                neighbors[ia].add(ib)
                neighbors[ib].add(ia)

    interior = ~pressure_interface_mask
    seen = np.zeros(len(Q_coords), dtype=bool)
    anchors = np.zeros(len(Q_coords), dtype=bool)
    for start in np.flatnonzero(interior):
        if seen[start]:
            continue
        stack = [int(start)]
        component: list[int] = []
        seen[start] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for nb in neighbors[node]:
                if interior[nb] and not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        coords = Q_coords[component]
        centroid = np.mean(coords, axis=0)
        anchor = component[int(np.argmin(np.linalg.norm(coords - centroid, axis=1)))]
        anchors[anchor] = True
    return anchors


def pnm_interface_nodes(
    coords: np.ndarray,
    centers: np.ndarray,
    edges: list[tuple[int, int]],
    h: float,
) -> np.ndarray:
    mask = np.zeros(len(coords), dtype=bool)
    thickness = 0.65 * h
    for a, b in edges:
        ca = centers[a]
        cb = centers[b]
        direction = cb - ca
        length = float(np.linalg.norm(direction))
        if length <= 1.0e-14:
            continue
        normal = direction / length
        midpoint = 0.5 * (ca + cb)
        rel = coords - midpoint
        axial = np.abs(rel @ normal)
        radial = np.linalg.norm(rel - (rel @ normal)[:, None] * normal, axis=1)
        mask |= (axial <= thickness) & (radial <= 0.42 * length)
    return mask


def select_pnm_edges(
    centers: np.ndarray,
    target_degree: int,
    max_edge_length: float,
) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    tri = Delaunay(centers)
    for simplex in tri.simplices:
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = sorted((int(simplex[i]), int(simplex[j])))
                if np.linalg.norm(centers[a] - centers[b]) <= max_edge_length:
                    edges.add((a, b))
    distances = cdist(centers, centers)
    np.fill_diagonal(distances, np.inf)
    for a in range(len(centers)):
        for b in np.argsort(distances[a])[:target_degree]:
            edge = tuple(sorted((int(a), int(b))))
            edges.add(edge)
    return sorted(connect_components(centers, edges))


def connect_components(centers: np.ndarray, edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    parent = list(range(len(centers)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    distances = cdist(centers, centers)
    while len({find(i) for i in range(len(centers))}) > 1:
        best = None
        best_distance = np.inf
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if find(i) == find(j):
                    continue
                if distances[i, j] < best_distance:
                    best_distance = distances[i, j]
                    best = (i, j)
        assert best is not None
        edge = tuple(sorted(best))
        edges.add(edge)
        union(*edge)
    return edges


def solve_schur_hoddpnm(A, b, interface_mask: np.ndarray, fixed_dofs: np.ndarray, args) -> dict[str, object]:
    n = A.shape[0]
    interface_mask = np.asarray(interface_mask, dtype=bool).copy()
    fixed = np.unique(np.asarray(fixed_dofs, dtype=np.int64))
    fixed = fixed[(fixed >= 0) & (fixed < n)]

    fixed_mask = np.zeros(n, dtype=bool)
    fixed_mask[fixed] = True
    boundary_mask = interface_mask & ~fixed_mask
    interior_mask = ~(boundary_mask | fixed_mask)

    boundary = np.flatnonzero(boundary_mask)
    interior = np.flatnonzero(interior_mask)
    known = np.flatnonzero(fixed_mask)
    x_known = np.asarray(b[known], dtype=float)

    rhs_all = np.asarray(b, dtype=float).copy()
    if len(known):
        rhs_all -= A[:, known] @ x_known

    Aii = A[interior][:, interior].tocsc()
    Aib = A[interior][:, boundary].tocsc()
    Abi = A[boundary][:, interior].tocsc()
    Abb = A[boundary][:, boundary].tocsc()
    bi = rhs_all[interior]
    bb = rhs_all[boundary]

    lu = splu(Aii)
    yi = lu.solve(bi)
    rhs = bb - Abi @ yi

    if args.schur_solver == "dense":
        Xi = lu.solve(Aib.toarray())
        S = Abb.toarray() - Abi @ Xi
        ub = np.linalg.solve(S, rhs)
        schur_relative_residual = float(np.linalg.norm(S @ ub - rhs) / max(np.linalg.norm(rhs), 1.0e-300))
        schur_iterations = None
        dense_schur_used = True
    else:
        S = None

        def schur_matvec(x: np.ndarray) -> np.ndarray:
            return Abb @ x - Abi @ lu.solve(Aib @ x)

        operator = LinearOperator((len(boundary), len(boundary)), matvec=schur_matvec, dtype=float)
        preconditioner = schur_preconditioner(Abb, args.schur_preconditioner)
        iterations = {"count": 0}

        def callback(_residual) -> None:
            iterations["count"] += 1

        ub, info = gmres(
            operator,
            rhs,
            rtol=args.schur_rtol,
            atol=args.schur_atol,
            restart=args.schur_restart,
            maxiter=args.schur_maxiter,
            M=preconditioner,
            callback=callback,
            callback_type="pr_norm",
        )
        schur_relative_residual = float(np.linalg.norm(schur_matvec(ub) - rhs) / max(np.linalg.norm(rhs), 1.0e-300))
        if info != 0:
            raise RuntimeError(
                f"matrix-free Schur GMRES did not converge: info={info}, "
                f"relative_residual={schur_relative_residual:.3e}, iterations={iterations['count']}"
            )
        schur_iterations = int(iterations["count"])
        dense_schur_used = False

    ui = yi - lu.solve(Aib @ ub)

    sol = np.zeros(A.shape[0], dtype=float)
    sol[known] = x_known
    sol[boundary] = ub
    sol[interior] = ui
    return {
        "solution": sol,
        "schur_matrix": S,
        "n_boundary": int(len(boundary)),
        "n_interior": len(interior),
        "n_fixed_known": int(len(known)),
        "schur_solver": args.schur_solver,
        "schur_preconditioner": args.schur_preconditioner if args.schur_solver == "gmres" else None,
        "dense_schur_used": dense_schur_used,
        "schur_iterations": schur_iterations,
        "schur_relative_residual": schur_relative_residual,
    }


def schur_preconditioner(Abb, mode: str):
    if mode == "none":
        return None
    if mode == "ilu":
        ilu = spilu(Abb.tocsc(), drop_tol=1.0e-4, fill_factor=8.0)
        return LinearOperator(Abb.shape, matvec=ilu.solve, dtype=float)
    diag = np.asarray(Abb.diagonal(), dtype=float)
    scale = np.where(np.abs(diag) > 1.0e-14, 1.0 / diag, 1.0)
    return LinearOperator((len(scale), len(scale)), matvec=lambda x: scale * x, dtype=float)


def mean_aligned_difference(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (candidate - np.mean(candidate)) - (reference - np.mean(reference))


def vertex_output_fields(mesh: MeshData, assembled: dict[str, object], fem_solution: np.ndarray, hodd_solution: np.ndarray) -> dict[str, np.ndarray]:
    V_coords = assembled["V_coords"]
    Q_coords = assembled["Q_coords"]
    mapV = assembled["mapV"]
    mapQ = assembled["mapQ"]
    fem_v_p2 = fem_solution[mapV].reshape((-1, 3))
    hodd_v_p2 = hodd_solution[mapV].reshape((-1, 3))
    fem_p = fem_solution[mapQ]
    hodd_p = hodd_solution[mapQ]

    v_tree = cKDTree(V_coords)
    q_tree = cKDTree(Q_coords)
    _, v_idx = v_tree.query(mesh.coords, k=1)
    _, q_idx = q_tree.query(mesh.coords, k=1)
    return {
        "fem_velocity": fem_v_p2[v_idx],
        "hodd_velocity": hodd_v_p2[v_idx],
        "velocity_abs_error": np.linalg.norm(hodd_v_p2[v_idx] - fem_v_p2[v_idx], axis=1),
        "velocity_magnitude": np.linalg.norm(fem_v_p2[v_idx], axis=1),
        "fem_pressure": fem_p[q_idx],
        "hodd_pressure": hodd_p[q_idx],
        "pressure_abs_error": np.abs(hodd_p[q_idx] - fem_p[q_idx]),
    }


def save_vtu(mesh: MeshData, fields: dict[str, np.ndarray], out: Path) -> None:
    cells = np.column_stack((np.full(len(mesh.cells), 4, dtype=np.int64), mesh.cells)).ravel()
    celltypes = np.full(len(mesh.cells), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, mesh.coords)
    for name, values in fields.items():
        grid.point_data[name] = values
    grid.point_data["abs_error"] = np.sqrt(fields["velocity_abs_error"] ** 2 + fields["pressure_abs_error"] ** 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)


def plot_error_cloud(vtu: Path, out: Path) -> None:
    grid = pv.read(vtu)
    full = grid.extract_surface().smooth(n_iter=30, relaxation_factor=0.05)
    cut = grid.clip(normal=(0, -1, 0), origin=(2.5, 2.55, 2.5)).extract_surface().smooth(
        n_iter=30,
        relaxation_factor=0.05,
    )
    vmax = max(float(np.percentile(grid.point_data["abs_error"], 99.5)), 1.0e-14)
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1500, 560), border=False)
    plotter.set_background("white")
    bar = {
        "title": "|error|",
        "vertical": True,
        "position_x": 0.89,
        "position_y": 0.18,
        "width": 0.045,
        "height": 0.62,
        "fmt": "%.1e",
        "color": "black",
    }
    for idx, obj in enumerate([full, cut]):
        plotter.subplot(0, idx)
        plotter.add_mesh(
            obj,
            scalars="abs_error",
            cmap="viridis",
            clim=(0.0, vmax),
            smooth_shading=True,
            show_edges=False,
            opacity=0.58 if idx == 0 else 1.0,
            scalar_bar_args=bar if idx == 1 else None,
            show_scalar_bar=idx == 1,
        )
        plotter.camera_position = [(8.0, -7.6, 5.3), (2.5, 2.5, 2.5), (0.0, 0.0, 1.0)]
        plotter.camera.zoom(1.28 if idx == 0 else 1.38)
        plotter.enable_parallel_projection()
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()


def scalar_error_stats(diff: np.ndarray, ref: np.ndarray) -> dict[str, float]:
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


if __name__ == "__main__":
    main()

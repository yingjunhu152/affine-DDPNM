from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from ctypes import wintypes

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI
from scipy.sparse.linalg import LinearOperator, gmres, spilu, splu, spsolve
from scipy.spatial import cKDTree

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from classic_ddpnm.linalg import to_numpy_vector, to_scipy_matrix  # noqa: E402


@dataclass
class VoxelMeshData:
    coords: np.ndarray
    cells: np.ndarray
    pore_voxels: np.ndarray
    voxel_labels: np.ndarray
    interface_centers: np.ndarray
    voxel_size: float
    domain_shape: tuple[int, int, int]

    @property
    def domain_size(self) -> tuple[float, float, float]:
        nx, ny, nz = self.domain_shape
        return (nx * self.voxel_size, ny * self.voxel_size, nz * self.voxel_size)


def process_memory_mib() -> dict[str, float] | None:
    if sys.platform != "win32":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("Kernel32.dll")
    psapi = ctypes.WinDLL("Psapi.dll")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        return None
    scale = 1024.0 * 1024.0
    return {
        "working_set_mib": float(counters.WorkingSetSize / scale),
        "peak_working_set_mib": float(counters.PeakWorkingSetSize / scale),
        "pagefile_mib": float(counters.PagefileUsage / scale),
        "peak_pagefile_mib": float(counters.PeakPagefileUsage / scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume-npy", type=Path, help="Path to a binary/segmented 3D .npy or .npz volume.")
    parser.add_argument("--volume-raw", type=Path, help="Path to a binary/segmented 3D raw volume.")
    parser.add_argument("--raw-shape", type=int, nargs=3, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--raw-dtype", default="uint8")
    parser.add_argument("--raw-order", choices=("xyz", "zyx"), default="zyx")
    parser.add_argument("--pore-value", type=float, default=0.0)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--pore-below-threshold", action="store_true")
    parser.add_argument("--invert-pore-mask", action="store_true")
    parser.add_argument("--crop", type=str, help="Crop as x0:x1,y0:y1,z0:z1 before meshing.")
    parser.add_argument("--crop-size", type=int, default=36, help="Centered crop size if --crop is omitted.")
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--regions", type=int, default=12)
    parser.add_argument("--interface-thickness", type=float, default=1.05)
    parser.add_argument("--pressure-interface-thickness", type=float, default=2.0)
    parser.add_argument("--pressure-stabilization", type=float, default=1.0e-6)
    parser.add_argument("--hoddpnm-solver", choices=("gmres", "exact"), default="gmres")
    parser.add_argument("--schur-rtol", type=float, default=1.0e-10)
    parser.add_argument("--schur-atol", type=float, default=1.0e-12)
    parser.add_argument("--schur-restart", type=int, default=80)
    parser.add_argument("--schur-maxiter", type=int, default=300)
    parser.add_argument("--schur-preconditioner", choices=("ilu", "none"), default="ilu")
    parser.add_argument("--ilu-drop-tol", type=float, default=1.0e-4)
    parser.add_argument("--ilu-fill-factor", type=float, default=12.0)
    parser.add_argument("--skip-condition-number", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/real_porous_hoddpnm_validation"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_total0 = time.perf_counter()
    memory_trace = {"start": process_memory_mib()}
    t0 = time.perf_counter()
    pore = load_pore_mask(args)
    pore = center_crop_or_slice(pore, args.crop, args.crop_size)
    mesh = build_real_porous_mesh(
        pore,
        voxel_size=args.voxel_size,
        n_regions=args.regions,
    )
    geometry_time = time.perf_counter() - t0
    memory_trace["after_geometry_and_decomposition"] = process_memory_mib()
    t0 = time.perf_counter()
    assembled = assemble_taylor_hood_system(
        mesh,
        args.pressure_stabilization,
        args.interface_thickness,
        args.pressure_interface_thickness,
    )
    assembly_time = time.perf_counter() - t0
    memory_trace["after_fem_assembly"] = process_memory_mib()

    t0 = time.perf_counter()
    fem_solution = spsolve(assembled["A"], assembled["b"])
    fem_time = time.perf_counter() - t0
    memory_trace["after_fem_sparse_direct_solve"] = process_memory_mib()
    t0 = time.perf_counter()
    hodd = solve_schur_hoddpnm(
        assembled["A"],
        assembled["b"],
        assembled["interface_mask"],
        assembled["fixed_dofs"],
        solver=args.hoddpnm_solver,
        rtol=args.schur_rtol,
        atol=args.schur_atol,
        restart=args.schur_restart,
        maxiter=args.schur_maxiter,
        preconditioner=args.schur_preconditioner,
        ilu_drop_tol=args.ilu_drop_tol,
        ilu_fill_factor=args.ilu_fill_factor,
    )
    hodd_time = time.perf_counter() - t0
    memory_trace["after_hoddpnm_schur_solve"] = process_memory_mib()
    hodd_solution = hodd["solution"]
    diff = hodd_solution - fem_solution

    velocity_error = diff[assembled["mapV"]].reshape((-1, 3))
    pressure_error = diff[assembled["mapQ"]]
    velocity_ref = fem_solution[assembled["mapV"]].reshape((-1, 3))
    pressure_ref = fem_solution[assembled["mapQ"]]
    fields = vertex_output_fields(mesh, assembled, fem_solution, hodd_solution)

    schur_condition_number = None
    if not args.skip_condition_number and hodd["schur_matrix"] is not None:
        schur_condition_number = float(np.linalg.cond(hodd["schur_matrix"]))

    summary = {
        "method": "real porous medium Taylor-Hood P2-P1 Stokes HODDPNM via Schur complement",
        "numerical_structure": "same FEM assembly + interface/interior partition + Schur reconstruction as the random-sphere HODDPNM validation",
        "hoddpnm_solver": hodd["solver"],
        "hoddpnm_solver_info": hodd["solver_info"],
        "region_decomposition": "geodesic Voronoi partition on the connected real pore voxel graph",
        "domain_shape_voxels": list(mesh.domain_shape),
        "voxel_size": mesh.voxel_size,
        "pore_voxels": int(len(mesh.pore_voxels)),
        "regions": int(args.regions),
        "region_interface_faces": int(len(mesh.interface_centers)),
        "n_vertices": int(len(mesh.coords)),
        "n_tets": int(len(mesh.cells)),
        "mixed_dofs": int(len(fem_solution)),
        "velocity_p2_scalar_nodes": int(len(assembled["V_coords"])),
        "velocity_dofs": int(len(assembled["mapV"])),
        "pressure_dofs": int(len(assembled["mapQ"])),
        "hoddpnm_active_dofs": int(hodd["n_boundary"]),
        "hoddpnm_active_dof_ratio": float(hodd["n_boundary"] / len(fem_solution)),
        "hoddpnm_interface_dofs": int(np.count_nonzero(assembled["interface_mask"])),
        "hoddpnm_known_fixed_dofs": int(hodd["n_fixed_known"]),
        "hoddpnm_velocity_interface_dofs": int(assembled["hoddpnm_velocity_interface_dofs"]),
        "hoddpnm_pressure_interface_dofs": int(assembled["hoddpnm_pressure_interface_dofs"]),
        "hoddpnm_pressure_interior_dofs": int(assembled["hoddpnm_pressure_interior_dofs"]),
        "hoddpnm_interior_dofs_eliminated": int(hodd["n_interior"]),
        "schur_condition_number": schur_condition_number,
        "timings_seconds": {
            "geometry_and_decomposition": float(geometry_time),
            "fem_assembly": float(assembly_time),
            "fem_sparse_direct_solve": float(fem_time),
            "hoddpnm_schur_solve": float(hodd_time),
            "total_before_output": float(time.perf_counter() - t_total0),
        },
        "memory_trace_mib": memory_trace,
        "errors_mixed_all": scalar_error_stats(diff, fem_solution),
        "errors_velocity_p2_all": vector_error_stats(velocity_error, velocity_ref),
        "errors_pressure_p1_all": scalar_error_stats(pressure_error, pressure_ref),
        "errors_velocity_vertices": vector_error_stats(
            fields["hodd_velocity"] - fields["fem_velocity"],
            fields["fem_velocity"],
        ),
        "errors_pressure_vertices": scalar_error_stats(
            fields["hodd_pressure"] - fields["fem_pressure"],
            fields["fem_pressure"],
        ),
    }
    (args.out_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_vtu(mesh, fields, args.out_dir / "real_porous_hoddpnm_solution.vtu")
    plot_error_cloud(args.out_dir / "real_porous_hoddpnm_solution.vtu", args.out_dir / "real_porous_hoddpnm_error.png")

    print(f"pore voxels: {summary['pore_voxels']}")
    print(f"regions: {summary['regions']}")
    print(f"region interface faces: {summary['region_interface_faces']}")
    print(f"vertices: {summary['n_vertices']}")
    print(f"tets: {summary['n_tets']}")
    print(f"mixed dofs: {summary['mixed_dofs']}")
    print(f"HODDPNM active Schur dofs: {summary['hoddpnm_active_dofs']}")
    print(f"HODDPNM active ratio: {summary['hoddpnm_active_dof_ratio']:.1%}")
    print(f"HODDPNM interface dofs before fixed removal: {summary['hoddpnm_interface_dofs']}")
    print(f"HODDPNM known fixed dofs removed: {summary['hoddpnm_known_fixed_dofs']}")
    print(f"eliminated interior dofs: {summary['hoddpnm_interior_dofs_eliminated']}")
    print(f"FEM direct solve time: {fem_time:.3f} s")
    print(f"HODDPNM Schur time ({hodd['solver']}): {hodd_time:.3f} s")
    print(f"HODDPNM solver info: {hodd['solver_info']}")
    after_fem_memory = memory_trace.get("after_fem_sparse_direct_solve") or {}
    after_hodd_memory = memory_trace.get("after_hoddpnm_schur_solve") or {}
    print(f"FEM checkpoint working set: {after_fem_memory.get('working_set_mib', float('nan')):.1f} MiB")
    print(f"HODDPNM checkpoint working set: {after_hodd_memory.get('working_set_mib', float('nan')):.1f} MiB")
    print(f"velocity P2 L2 rel error: {summary['errors_velocity_p2_all']['l2_rel']:.3e}")
    print(f"pressure P1 L2 rel error: {summary['errors_pressure_p1_all']['l2_rel']:.3e}")
    print(f"wrote {args.out_dir / 'validation_summary.json'}")
    print(f"wrote {args.out_dir / 'real_porous_hoddpnm_solution.vtu'}")


def load_pore_mask(args) -> np.ndarray:
    if args.volume_npy is None and args.volume_raw is None:
        raise SystemExit(
            "Provide a real segmented volume with --volume-npy or --volume-raw. "
            "See DATA_SOURCES.md for Berea sandstone download notes."
        )
    if args.volume_npy is not None:
        arr = np.load(args.volume_npy)
        if isinstance(arr, np.lib.npyio.NpzFile):
            first = sorted(arr.files)[0]
            volume = arr[first]
        else:
            volume = arr
    else:
        if args.raw_shape is None:
            raise SystemExit("--raw-shape NX NY NZ is required with --volume-raw.")
        raw = np.fromfile(args.volume_raw, dtype=np.dtype(args.raw_dtype))
        expected = int(np.prod(args.raw_shape))
        if raw.size != expected:
            raise SystemExit(f"raw size {raw.size} does not match --raw-shape product {expected}.")
        if args.raw_order == "zyx":
            volume = raw.reshape((args.raw_shape[2], args.raw_shape[1], args.raw_shape[0])).transpose(2, 1, 0)
        else:
            volume = raw.reshape(tuple(args.raw_shape))

    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise SystemExit(f"Expected a 3D volume, got shape {volume.shape}.")
    if args.threshold is None:
        pore = volume == args.pore_value
    elif args.pore_below_threshold:
        pore = volume < args.threshold
    else:
        pore = volume > args.threshold
    if args.invert_pore_mask:
        pore = ~pore
    return keep_largest_pore_component(np.asarray(pore, dtype=bool))


def center_crop_or_slice(pore: np.ndarray, crop: str | None, crop_size: int) -> np.ndarray:
    if crop:
        spans = []
        for item in crop.split(","):
            a, b = item.split(":")
            spans.append(slice(int(a), int(b)))
        if len(spans) != 3:
            raise SystemExit("--crop must have three ranges: x0:x1,y0:y1,z0:z1")
        out = pore[spans[0], spans[1], spans[2]]
    else:
        size = min(crop_size, *pore.shape)
        start = [(n - size) // 2 for n in pore.shape]
        out = pore[
            start[0] : start[0] + size,
            start[1] : start[1] + size,
            start[2] : start[2] + size,
        ]
    out = keep_largest_pore_component(out)
    if not out.any():
        raise SystemExit("The selected crop contains no connected pore space.")
    return out


def build_real_porous_mesh(pore: np.ndarray, voxel_size: float, n_regions: int) -> VoxelMeshData:
    pore_voxels = np.argwhere(pore)
    voxel_labels, interface_centers = decompose_pore_regions(pore, n_regions, voxel_size)

    nx, ny, nz = pore.shape
    raw_coords = np.asarray(
        [(i, j, k) for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)],
        dtype=float,
    )
    raw_coords *= voxel_size
    cells = []
    for i, j, k in pore_voxels:
        v000 = grid_id(i, j, k, ny, nz)
        v001 = grid_id(i, j, k + 1, ny, nz)
        v010 = grid_id(i, j + 1, k, ny, nz)
        v011 = grid_id(i, j + 1, k + 1, ny, nz)
        v100 = grid_id(i + 1, j, k, ny, nz)
        v101 = grid_id(i + 1, j, k + 1, ny, nz)
        v110 = grid_id(i + 1, j + 1, k, ny, nz)
        v111 = grid_id(i + 1, j + 1, k + 1, ny, nz)
        cells.extend(
            [
                [v000, v001, v011, v111],
                [v000, v011, v010, v111],
                [v000, v010, v110, v111],
                [v000, v110, v100, v111],
                [v000, v100, v101, v111],
                [v000, v101, v001, v111],
            ]
        )
    raw_cells = np.asarray(cells, dtype=np.int64)
    used = np.unique(raw_cells.ravel())
    remap = -np.ones(len(raw_coords), dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    coords = raw_coords[used]
    cells = orient_tets(coords, remap[raw_cells])
    return VoxelMeshData(
        coords=coords,
        cells=cells,
        pore_voxels=pore_voxels,
        voxel_labels=voxel_labels,
        interface_centers=interface_centers,
        voxel_size=voxel_size,
        domain_shape=tuple(int(x) for x in pore.shape),
    )


def decompose_pore_regions(pore: np.ndarray, n_regions: int, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    pore_voxels = np.argwhere(pore)
    seeds = farthest_point_seeds(pore_voxels, min(n_regions, len(pore_voxels)))
    labels = -np.ones(pore.shape, dtype=np.int32)
    q: deque[tuple[int, int, int]] = deque()
    for label, seed in enumerate(seeds):
        x, y, z = (int(v) for v in seed)
        labels[x, y, z] = label
        q.append((x, y, z))
    while q:
        x, y, z = q.popleft()
        label = labels[x, y, z]
        for nx, ny, nz in neighbor_voxels(x, y, z, pore.shape):
            if not pore[nx, ny, nz] or labels[nx, ny, nz] >= 0:
                continue
            labels[nx, ny, nz] = label
            q.append((nx, ny, nz))

    interface_centers = []
    for x, y, z in pore_voxels:
        here = labels[x, y, z]
        for nx, ny, nz in neighbor_voxels(int(x), int(y), int(z), pore.shape):
            if not pore[nx, ny, nz]:
                continue
            there = labels[nx, ny, nz]
            if there >= 0 and there != here:
                center = (np.asarray([x, y, z], dtype=float) + np.asarray([nx, ny, nz], dtype=float) + 1.0) * 0.5
                interface_centers.append(center * voxel_size)
    if interface_centers:
        interface = np.unique(np.round(np.asarray(interface_centers), decimals=12), axis=0)
    else:
        interface = np.empty((0, 3), dtype=float)
    return labels, interface


def farthest_point_seeds(points: np.ndarray, n_seeds: int) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    first = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))
    seeds = [points[first]]
    dist2 = np.sum((points - seeds[0]) ** 2, axis=1)
    for _ in range(1, n_seeds):
        idx = int(np.argmax(dist2))
        seeds.append(points[idx])
        dist2 = np.minimum(dist2, np.sum((points - points[idx]) ** 2, axis=1))
    return np.asarray(seeds, dtype=np.int64)


def assemble_taylor_hood_system(
    mesh: VoxelMeshData,
    pressure_stabilization: float,
    interface_thickness: float,
    pressure_interface_thickness: float | None = None,
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
        - fem.Constant(msh, pressure_stabilization) * p * q * ufl.dx
    )
    L = fem.form(fem.Constant(msh, 0.0) * q * ufl.dx)

    bcs, fixed_dofs = taylor_hood_bcs(msh, W, V, Q, np.asarray(mapQ, dtype=np.int64), mesh)
    A = to_scipy_matrix(fem.assemble_matrix(a, bcs=bcs))
    b = to_numpy_vector(fem.assemble_vector(L))
    fem.apply_lifting(b, [a], [bcs])
    fem.set_bc(b, bcs)

    interface_mask = taylor_hood_interface_mask(
        W.dofmap.index_map.size_local * W.dofmap.index_map_bs,
        np.asarray(mapV, dtype=np.int64),
        V.tabulate_dof_coordinates(),
        np.asarray(mapQ, dtype=np.int64),
        Q.tabulate_dof_coordinates(),
        mesh,
        interface_thickness,
        interface_thickness if pressure_interface_thickness is None else pressure_interface_thickness,
    )
    velocity_interface = interface_mask[np.asarray(mapV, dtype=np.int64)]
    pressure_interface = interface_mask[np.asarray(mapQ, dtype=np.int64)]
    return {
        "A": A,
        "b": b,
        "interface_mask": interface_mask,
        "fixed_dofs": fixed_dofs,
        "mapV": np.asarray(mapV, dtype=np.int64),
        "mapQ": np.asarray(mapQ, dtype=np.int64),
        "V_coords": V.tabulate_dof_coordinates(),
        "Q_coords": Q.tabulate_dof_coordinates(),
        "hoddpnm_velocity_interface_dofs": int(np.count_nonzero(velocity_interface)),
        "hoddpnm_pressure_interface_dofs": int(np.count_nonzero(pressure_interface)),
        "hoddpnm_pressure_interior_dofs": int(len(mapQ) - np.count_nonzero(pressure_interface)),
    }


def taylor_hood_bcs(msh, W, V, Q, mapQ: np.ndarray, mesh: VoxelMeshData):
    lx, ly, lz = mesh.domain_size
    voxel_size = mesh.voxel_size

    def inlet(x):
        return np.isclose(x[0], 0.0)

    def solid_or_lateral_wall(x):
        points = x.T
        on_lateral = (
            np.isclose(points[:, 1], 0.0)
            | np.isclose(points[:, 1], ly)
            | np.isclose(points[:, 2], 0.0)
            | np.isclose(points[:, 2], lz)
        )
        near_solid = near_solid_boundary(points, mesh, 0.62 * voxel_size)
        return on_lateral | (near_solid & ~np.isclose(points[:, 0], 0.0) & ~np.isclose(points[:, 0], lx))

    u_wall = fem.Function(V)
    u_wall.x.array[:] = 0.0
    u_in = fem.Function(V)
    u_in.x.array[:] = 0.0
    u_in.x.array.reshape((-1, 3))[:, 0] = 1.0

    wall_dofs = fem.locate_dofs_geometrical((W.sub(0), V), solid_or_lateral_wall)
    inlet_dofs = fem.locate_dofs_geometrical((W.sub(0), V), inlet)
    bcs = [
        fem.dirichletbc(u_wall, wall_dofs, W.sub(0)),
        fem.dirichletbc(u_in, inlet_dofs, W.sub(0)),
    ]

    q_coords = Q.tabulate_dof_coordinates()
    outlet_candidates = np.flatnonzero(np.isclose(q_coords[:, 0], lx))
    if len(outlet_candidates) == 0:
        outlet_candidates = np.asarray([int(np.argmax(q_coords[:, 0]))], dtype=int)
    target = np.asarray([lx, 0.5 * ly, 0.5 * lz])
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


def near_solid_boundary(points: np.ndarray, mesh: VoxelMeshData, tolerance: float) -> np.ndarray:
    shifted = np.clip(np.floor(points / mesh.voxel_size).astype(int), 0, np.asarray(mesh.domain_shape) - 1)
    pore_lookup = np.zeros(mesh.domain_shape, dtype=bool)
    pore_lookup[tuple(mesh.pore_voxels.T)] = True
    solidish = np.zeros(len(points), dtype=bool)
    for n, ijk in enumerate(shifted):
        x, y, z = (int(v) for v in ijk)
        for nx, ny, nz in neighbor_voxels(x, y, z, mesh.domain_shape):
            if not pore_lookup[nx, ny, nz]:
                solidish[n] = True
                break
    return solidish


def taylor_hood_interface_mask(
    n_mixed: int,
    mapV: np.ndarray,
    V_coords: np.ndarray,
    mapQ: np.ndarray,
    Q_coords: np.ndarray,
    mesh: VoxelMeshData,
    interface_thickness: float,
    pressure_interface_thickness: float,
) -> np.ndarray:
    mask = np.zeros(n_mixed, dtype=bool)
    if len(mesh.interface_centers) == 0:
        v_node_mask = np.zeros(len(V_coords), dtype=bool)
        q_node_mask = np.zeros(len(Q_coords), dtype=bool)
    else:
        tree = cKDTree(mesh.interface_centers)
        dist, _ = tree.query(V_coords, k=1, workers=-1)
        v_node_mask = dist <= interface_thickness * mesh.voxel_size
        dist_q, _ = tree.query(Q_coords, k=1, workers=-1)
        q_node_mask = dist_q <= pressure_interface_thickness * mesh.voxel_size
    mask[mapV] = np.repeat(v_node_mask, 3)
    mask[mapQ] = q_node_mask
    return mask


def solve_schur_hoddpnm(
    A,
    b,
    interface_mask: np.ndarray,
    fixed_dofs: np.ndarray,
    solver: str,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    preconditioner: str,
    ilu_drop_tol: float,
    ilu_fill_factor: float,
) -> dict[str, object]:
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

    if solver == "exact":
        Xi = lu.solve(Aib.toarray())
        S = Abb.toarray() - Abi @ Xi
        ub = np.linalg.solve(S, rhs)
        solver_info = {
            "type": "dense_exact_schur",
            "converged": True,
            "iterations": 1,
            "gmres_info": 0,
            "relative_residual": 0.0,
            "preconditioner": "none",
        }
        ui = yi - Xi @ ub
    else:
        scale = schur_diagonal_scale(Aii, Aib, Abi, Abb)
        inv_scale = 1.0 / scale

        matvec_count = 0

        def schur_matvec(x: np.ndarray) -> np.ndarray:
            nonlocal matvec_count
            matvec_count += 1
            return Abb @ x - Abi @ lu.solve(Aib @ x)

        schur_operator = LinearOperator(
            Abb.shape,
            matvec=lambda x: scale * schur_matvec(scale * x),
            dtype=float,
        )
        preconditioner_operator, preconditioner_info = build_schur_preconditioner(
            Aii,
            Aib,
            Abi,
            Abb,
            preconditioner,
            ilu_drop_tol,
            ilu_fill_factor,
        )
        if preconditioner_operator is not None:
            inner_preconditioner = preconditioner_operator
            preconditioner_operator = LinearOperator(
                Abb.shape,
                matvec=lambda x: inv_scale * (inner_preconditioner @ (inv_scale * x)),
                dtype=float,
        )
        residuals = []
        gmres_rtol = min(float(rtol) * 1.0e-4, 1.0e-14)
        gmres_atol = min(float(atol) * 1.0e-4, 1.0e-16)
        scaled_rhs = scale * rhs
        yb, info = gmres(
            schur_operator,
            scaled_rhs,
            rtol=gmres_rtol,
            atol=gmres_atol,
            restart=restart,
            maxiter=maxiter,
            M=preconditioner_operator,
            callback=residuals.append,
            callback_type="pr_norm",
        )
        ub = scale * yb
        ui = yi - lu.solve(Aib @ ub)
        final_residual = np.linalg.norm(schur_matvec(ub) - rhs) / max(np.linalg.norm(rhs), 1.0e-30)
        solver_info = {
            "type": f"matrix_free_{solver}_schur",
            "converged": bool(info == 0),
            "iterations": int(len(residuals)),
            "schur_matvecs": int(matvec_count),
            "gmres_info": int(info),
            "relative_residual": float(final_residual),
            "rtol": float(rtol),
            "atol": float(atol),
            "restart": int(restart),
            "maxiter": int(maxiter),
            "schur_diagonal_scaling": True,
            "scaled_gmres_rtol": float(gmres_rtol),
            "scaled_gmres_atol": float(gmres_atol),
            "schur_scale_min": float(np.min(scale)),
            "schur_scale_max": float(np.max(scale)),
            **preconditioner_info,
        }
        if info != 0 or not np.isfinite(final_residual) or final_residual > max(10.0 * rtol, 1.0e-9):
            raise RuntimeError(
                "HODDPNM Schur GMRES did not converge: "
                f"info={info}, iterations={len(residuals)}, relative_residual={final_residual:.3e}"
            )
        S = None

    sol = np.zeros(n, dtype=float)
    sol[known] = x_known
    sol[boundary] = ub
    sol[interior] = ui
    return {
        "solution": sol,
        "schur_matrix": S,
        "n_boundary": int(len(boundary)),
        "n_interior": len(interior),
        "n_fixed_known": int(len(known)),
        "solver": solver,
        "solver_info": solver_info,
    }


def schur_diagonal_scale(Aii, Aib, Abi, Abb) -> np.ndarray:
    diag = Aii.diagonal().astype(float)
    small = np.abs(diag) < 1.0e-14
    diag[small] = np.where(diag[small] < 0.0, -1.0e-14, 1.0e-14)
    inv_diag = 1.0 / diag
    correction_diag = np.asarray(Abi.multiply(Aib.T).dot(inv_diag), dtype=float).reshape(-1)
    approximate_diag = np.asarray(Abb.diagonal(), dtype=float) - correction_diag
    magnitude = np.maximum(np.abs(approximate_diag), 1.0e-14)
    return 1.0 / np.sqrt(magnitude)


def build_schur_preconditioner(
    Aii,
    Aib,
    Abi,
    Abb,
    preconditioner: str,
    ilu_drop_tol: float,
    ilu_fill_factor: float,
) -> tuple[LinearOperator | None, dict[str, object]]:
    if preconditioner == "none":
        return None, {"preconditioner": "none"}

    diag = Aii.diagonal().astype(float)
    small = np.abs(diag) < 1.0e-14
    diag[small] = np.where(diag[small] < 0.0, -1.0e-14, 1.0e-14)
    approximate_schur = (Abb - Abi @ (sp.diags(1.0 / diag, format="csc") @ Aib)).tocsc()
    try:
        ilu = spilu(approximate_schur, drop_tol=ilu_drop_tol, fill_factor=ilu_fill_factor)
    except RuntimeError as exc:
        return None, {
            "preconditioner": "ilu_failed_none_used",
            "preconditioner_error": str(exc),
        }

    return (
        LinearOperator(Abb.shape, matvec=ilu.solve, dtype=float),
        {
            "preconditioner": "ilu_of_diagonal_approximate_schur",
            "ilu_drop_tol": float(ilu_drop_tol),
            "ilu_fill_factor": float(ilu_fill_factor),
            "approximate_schur_nnz": int(approximate_schur.nnz),
        },
    )


def vertex_output_fields(mesh: VoxelMeshData, assembled: dict[str, object], fem_solution: np.ndarray, hodd_solution: np.ndarray) -> dict[str, np.ndarray]:
    V_coords = assembled["V_coords"]
    Q_coords = assembled["Q_coords"]
    mapV = assembled["mapV"]
    mapQ = assembled["mapQ"]
    tree_v = cKDTree(V_coords)
    _, nearest_v = tree_v.query(mesh.coords, k=1)
    tree_q = cKDTree(Q_coords)
    _, nearest_q = tree_q.query(mesh.coords, k=1)

    fem_velocity = fem_solution[mapV].reshape((-1, 3))[nearest_v]
    hodd_velocity = hodd_solution[mapV].reshape((-1, 3))[nearest_v]
    fem_pressure = fem_solution[mapQ][nearest_q]
    hodd_pressure = hodd_solution[mapQ][nearest_q]
    return {
        "fem_velocity": fem_velocity,
        "hodd_velocity": hodd_velocity,
        "velocity_error": hodd_velocity - fem_velocity,
        "fem_pressure": fem_pressure,
        "hodd_pressure": hodd_pressure,
        "pressure_error": hodd_pressure - fem_pressure,
    }


def save_vtu(mesh: VoxelMeshData, fields: dict[str, np.ndarray], out: Path) -> None:
    cells = np.hstack([np.full((len(mesh.cells), 1), 4, dtype=np.int64), mesh.cells]).ravel()
    celltypes = np.full(len(mesh.cells), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, mesh.coords)
    for name, values in fields.items():
        grid.point_data[name] = values
    grid.save(out)


def plot_error_cloud(vtu: Path, out: Path) -> None:
    grid = pv.read(vtu)
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 1100))
    grid.point_data["velocity_error_norm"] = np.linalg.norm(grid.point_data["velocity_error"], axis=1)
    plotter.add_mesh(grid, scalars="velocity_error_norm", cmap="magma", opacity=0.72, show_edges=False)
    plotter.add_axes()
    plotter.camera_position = "iso"
    plotter.screenshot(out)
    plotter.close()


def keep_largest_pore_component(pore: np.ndarray) -> np.ndarray:
    labels = -np.ones(pore.shape, dtype=np.int32)
    best_label = -1
    best_size = 0
    current = 0
    for start in np.argwhere(pore):
        sx, sy, sz = (int(v) for v in start)
        if labels[sx, sy, sz] >= 0:
            continue
        q: deque[tuple[int, int, int]] = deque([(sx, sy, sz)])
        labels[sx, sy, sz] = current
        size = 0
        while q:
            x, y, z = q.popleft()
            size += 1
            for nx, ny, nz in neighbor_voxels(x, y, z, pore.shape):
                if pore[nx, ny, nz] and labels[nx, ny, nz] < 0:
                    labels[nx, ny, nz] = current
                    q.append((nx, ny, nz))
        if size > best_size:
            best_size = size
            best_label = current
        current += 1
    return labels == best_label


def neighbor_voxels(x: int, y: int, z: int, shape: tuple[int, int, int]):
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
            yield nx, ny, nz


def grid_id(i: int, j: int, k: int, ny: int, nz: int) -> int:
    return (i * (ny + 1) + j) * (nz + 1) + k


def orient_tets(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    out = cells.copy()
    for idx, tet in enumerate(out):
        x = coords[tet]
        vol = np.linalg.det(np.column_stack((x[1] - x[0], x[2] - x[0], x[3] - x[0])))
        if vol < 0.0:
            out[idx, [0, 1]] = out[idx, [1, 0]]
    return out


def scalar_error_stats(diff: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    ref = np.asarray(ref, dtype=float)
    l2 = float(np.linalg.norm(diff))
    ref_l2 = float(np.linalg.norm(ref))
    return {
        "l2_abs": l2,
        "l2_rel": float(l2 / max(ref_l2, 1.0e-300)),
        "linf_abs": float(np.linalg.norm(diff, ord=np.inf)) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "median_abs": float(np.median(np.abs(diff))) if diff.size else 0.0,
    }


def vector_error_stats(diff: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff_norm = np.linalg.norm(diff, axis=1) if diff.size else np.asarray([], dtype=float)
    ref_norm = np.linalg.norm(ref, axis=1) if ref.size else np.asarray([], dtype=float)
    l2 = float(np.linalg.norm(diff_norm))
    ref_l2 = float(np.linalg.norm(ref_norm))
    return {
        "l2_abs": l2,
        "l2_rel": float(l2 / max(ref_l2, 1.0e-300)),
        "linf_abs": float(diff_norm.max()) if len(diff_norm) else 0.0,
        "linf_rel": float(diff_norm.max() / max(ref_norm.max(), 1.0e-300)) if len(diff_norm) else 0.0,
        "mean_abs": float(diff_norm.mean()) if len(diff_norm) else 0.0,
        "median_abs": float(np.median(diff_norm)) if len(diff_norm) else 0.0,
    }


if __name__ == "__main__":
    main()

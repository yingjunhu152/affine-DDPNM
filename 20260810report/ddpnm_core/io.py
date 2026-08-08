"""Dimension-agnostic I/O utilities: topology arrays, field assignment, XDMF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from basix.ufl import element
from dolfinx import fem, io, mesh as dmesh
from mpi4py import MPI
from scipy.spatial import cKDTree


def topology_vertex_coordinates(msh: dmesh.Mesh) -> np.ndarray:
    """Coordinates indexed by topology vertex id, independent of geometry ordering."""
    gdim = msh.geometry.dim
    n_vertices = msh.topology.index_map(0).size_local
    msh.topology.create_connectivity(0, msh.topology.dim)
    vertex_ids = np.arange(n_vertices, dtype=np.int32)
    geometry_ids = np.asarray(
        dmesh.entities_to_geometry(msh, 0, vertex_ids, False), dtype=np.int32
    ).reshape(-1)
    if len(geometry_ids) != n_vertices:
        raise RuntimeError("Unexpected topology-to-geometry vertex map.")
    return np.asarray(msh.geometry.x[geometry_ids, :gdim], dtype=float)


def topology_arrays(msh: dmesh.Mesh) -> tuple[np.ndarray, np.ndarray]:
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    n_cells = msh.topology.index_map(tdim).size_local
    cells = np.asarray([c2v.links(cell) for cell in range(n_cells)], dtype=np.int32)
    points = topology_vertex_coordinates(msh)
    return points, cells


def assign_p1_function(function: fem.Function, values: np.ndarray) -> None:
    gdim = function.function_space.mesh.geometry.dim
    coords = function.function_space.tabulate_dof_coordinates()[:, :gdim]
    msh = function.function_space.mesh
    vertex_coords = topology_vertex_coordinates(msh)
    distances, vertex_indices = cKDTree(vertex_coords).query(coords, k=1)
    if float(np.max(distances)) > 1.0e-9:
        raise RuntimeError(
            f"P1 dof-to-vertex mismatch: max distance {np.max(distances):.3e}."
        )
    if values.ndim == 1:
        function.x.array[:] = values[np.asarray(vertex_indices, dtype=np.int32)]
    else:
        components = values.shape[1]
        function.x.array.reshape(len(coords), components)[:] = values[
            np.asarray(vertex_indices, dtype=np.int32)
        ]


def assign_dg0_function(function: fem.Function, values: np.ndarray) -> None:
    space = function.function_space
    n_cells = space.mesh.topology.index_map(space.mesh.topology.dim).size_local
    if len(values) != n_cells:
        raise RuntimeError("DG0 values do not match the number of local cells.")
    block_size = space.dofmap.index_map_bs
    components = 1 if values.ndim == 1 else values.shape[1]
    if block_size != components:
        raise RuntimeError(
            f"Unexpected DG0 block size {block_size} for {components} components."
        )
    for cell in range(n_cells):
        dofs = space.dofmap.cell_dofs(cell)
        if len(dofs) != 1:
            raise RuntimeError("A DG0 element should have one block dof per cell.")
        start = block_size * int(dofs[0])
        if components == 1:
            function.x.array[start] = values[cell]
        else:
            function.x.array[start : start + block_size] = values[cell]


def write_xdmf_fields(
    msh: dmesh.Mesh,
    u_visual: np.ndarray,
    p_visual: np.ndarray,
    u_cell: np.ndarray,
    p_cell: np.ndarray,
    out_dir: Path,
    filename: str = "ddpnm_fields.xdmf",
    u_ref: np.ndarray | None = None,
    p_ref: np.ndarray | None = None,
    velocity_error_rms: np.ndarray | None = None,
    pressure_error_rms: np.ndarray | None = None,
) -> None:
    """Write P1 visualization and DG0 cell-mean / error fields to XDMF."""
    gdim = msh.geometry.dim
    cell = msh.basix_cell()
    V = fem.functionspace(msh, element("Lagrange", cell, 1, shape=(gdim,)))
    Q = fem.functionspace(msh, element("Lagrange", cell, 1))
    Vdg = fem.functionspace(msh, element("DG", cell, 0, shape=(gdim,)))
    Qdg = fem.functionspace(msh, element("DG", cell, 0))

    entries: list[tuple[str, fem.FunctionSpace, np.ndarray]] = [
        ("u_ddpnm_trace_average_visualization", V, u_visual),
        ("p_ddpnm_trace_average_visualization", Q, p_visual),
    ]
    if u_ref is not None and p_ref is not None:
        entries.extend(
            [("u_reference", V, u_ref), ("p_reference", Q, p_ref)]
        )
    functions = []
    for name, space, values in entries:
        func = fem.Function(space)
        func.name = name
        assign_p1_function(func, values)
        functions.append(func)
    for name, space, values in [
        ("u_ddpnm_piecewise_p1_cell_mean", Vdg, u_cell),
        ("p_ddpnm_piecewise_p1_cell_mean", Qdg, p_cell),
    ]:
        func = fem.Function(space)
        func.name = name
        assign_dg0_function(func, values)
        functions.append(func)
    if velocity_error_rms is not None and pressure_error_rms is not None:
        for name, values in [
            ("velocity_error_cell_rms", velocity_error_rms),
            ("pressure_error_cell_rms", pressure_error_rms),
        ]:
            func = fem.Function(Qdg)
            func.name = name
            assign_dg0_function(func, values)
            functions.append(func)

    with io.XDMFFile(MPI.COMM_SELF, str(out_dir / filename), "w") as xdmf:
        xdmf.write_mesh(msh)
        for func in functions:
            xdmf.write_function(func)

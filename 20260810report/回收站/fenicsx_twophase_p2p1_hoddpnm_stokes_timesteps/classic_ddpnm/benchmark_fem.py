from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from basix.ufl import element
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse.linalg import spsolve

from .geometry import Network
from .linalg import to_numpy_vector, to_scipy_matrix


@dataclass
class FemBenchmarkResult:
    pressures: np.ndarray
    mesh_ndofs: int
    mesh_cells: int
    inlet_dofs: int
    outlet_dofs: int
    vertex_coordinates: np.ndarray
    vertex_pressures: np.ndarray


def solve_pressure_benchmark(
    network: Network,
    domain_size: float = 5.0,
    cells_per_axis: int = 28,
    throat_radius: float = 0.13,
) -> FemBenchmarkResult:
    """Solve a connected-domain scalar pressure FEM benchmark.

    The benchmark domain is the union of all spherical pore bodies, straight
    cylindrical throats between neighboring pore centers, and inlet/outlet
    cylinders running from boundary planes to the first/last pore columns.
    It solves -div(grad p) = 0 with Dirichlet pressure on x-min/x-max openings
    and natural no-flux conditions elsewhere.
    """
    coords, cells = build_structured_union_mesh(
        network=network,
        domain_size=domain_size,
        cells_per_axis=cells_per_axis,
        throat_radius=throat_radius,
    )
    domain = ufl.Mesh(element("Lagrange", "tetrahedron", 1, shape=(3,)))
    msh = dmesh.create_mesh(MPI.COMM_SELF, cells, domain, coords)
    V = fem.functionspace(msh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = fem.form(ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(fem.Constant(msh, 0.0) * v * ufl.dx)

    inlet_facets, outlet_facets = boundary_facets_by_x(msh, domain_size)
    inlet_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, inlet_facets)
    outlet_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, outlet_facets)
    if len(inlet_dofs) == 0 or len(outlet_dofs) == 0:
        raise RuntimeError(
            "Benchmark mesh has no inlet/outlet Dirichlet dofs. "
            "Increase cells_per_axis or throat_radius."
        )

    bc_in = fem.dirichletbc(
        fem.Constant(msh, float(network.inlet_pressure)),
        inlet_dofs,
        V,
    )
    bc_out = fem.dirichletbc(
        fem.Constant(msh, float(network.outlet_pressure)),
        outlet_dofs,
        V,
    )
    bcs = [bc_in, bc_out]

    A = to_scipy_matrix(fem.assemble_matrix(a, bcs=bcs))
    b = to_numpy_vector(fem.assemble_vector(L))
    fem.apply_lifting(b, [a], [bcs])
    fem.set_bc(b, bcs)

    solution = spsolve(A, b)
    uh = fem.Function(V)
    uh.x.array[:] = solution

    dof_coords = V.tabulate_dof_coordinates()
    return FemBenchmarkResult(
        pressures=solution,
        mesh_ndofs=V.dofmap.index_map.size_local * V.dofmap.index_map_bs,
        mesh_cells=msh.topology.index_map(msh.topology.dim).size_local,
        inlet_dofs=len(inlet_dofs),
        outlet_dofs=len(outlet_dofs),
        vertex_coordinates=dof_coords,
        vertex_pressures=solution.copy(),
    )


def sample_pressures_at_interfaces(
    result: FemBenchmarkResult,
    network: Network,
) -> np.ndarray:
    interpolator = LinearNDInterpolator(result.vertex_coordinates, result.vertex_pressures)
    centers = {pore.id: np.asarray(pore.center, dtype=float) for pore in network.pores}
    points = np.asarray(
        [0.5 * (centers[a] + centers[b]) for a, b in network.interfaces],
        dtype=float,
    )
    values = np.asarray(interpolator(points), dtype=float)
    if np.any(np.isnan(values)):
        missing = int(np.count_nonzero(np.isnan(values)))
        raise RuntimeError(f"Could not interpolate FEM pressure at {missing} interfaces.")
    return values


def sample_pressures_at_points(
    result: FemBenchmarkResult,
    points: np.ndarray,
) -> np.ndarray:
    interpolator = LinearNDInterpolator(result.vertex_coordinates, result.vertex_pressures)
    values = np.asarray(interpolator(points), dtype=float)
    if np.any(np.isnan(values)):
        missing = int(np.count_nonzero(np.isnan(values)))
        raise RuntimeError(f"Could not interpolate FEM pressure at {missing} sample points.")
    return values


def build_structured_union_mesh(
    network: Network,
    domain_size: float,
    cells_per_axis: int,
    throat_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, domain_size, cells_per_axis + 1)
    pore_centers = np.asarray([p.center for p in network.pores], dtype=float)
    pore_radii = np.asarray([p.radius for p in network.pores], dtype=float)
    segments = make_segments(network, domain_size)

    kept_tets: list[list[int]] = []
    for i in range(cells_per_axis):
        for j in range(cells_per_axis):
            for k in range(cells_per_axis):
                centroid = np.array(
                    [
                        0.5 * (axis[k] + axis[k + 1]),
                        0.5 * (axis[j] + axis[j + 1]),
                        0.5 * (axis[i] + axis[i + 1]),
                    ],
                    dtype=float,
                )
                if not point_in_union(centroid, pore_centers, pore_radii, segments, throat_radius):
                    continue
                v000 = grid_id(i, j, k, cells_per_axis)
                v001 = grid_id(i, j, k + 1, cells_per_axis)
                v010 = grid_id(i, j + 1, k, cells_per_axis)
                v011 = grid_id(i, j + 1, k + 1, cells_per_axis)
                v100 = grid_id(i + 1, j, k, cells_per_axis)
                v101 = grid_id(i + 1, j, k + 1, cells_per_axis)
                v110 = grid_id(i + 1, j + 1, k, cells_per_axis)
                v111 = grid_id(i + 1, j + 1, k + 1, cells_per_axis)
                kept_tets.extend(
                    [
                        [v000, v001, v011, v111],
                        [v000, v011, v010, v111],
                        [v000, v010, v110, v111],
                        [v000, v110, v100, v111],
                        [v000, v100, v101, v111],
                        [v000, v101, v001, v111],
                    ]
                )

    if not kept_tets:
        raise RuntimeError("Structured benchmark mesh is empty.")

    all_coords = np.asarray(
        [(axis[k], axis[j], axis[i]) for i in range(cells_per_axis + 1) for j in range(cells_per_axis + 1) for k in range(cells_per_axis + 1)],
        dtype=np.float64,
    )
    raw_cells = np.asarray(kept_tets, dtype=np.int64)
    used = np.unique(raw_cells.ravel())
    remap = -np.ones(len(all_coords), dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    coords = all_coords[used]
    cells = orient_tets(coords, remap[raw_cells])
    return coords, cells


def make_segments(network: Network, domain_size: float) -> list[tuple[np.ndarray, np.ndarray]]:
    centers = {pore.id: np.asarray(pore.center, dtype=float) for pore in network.pores}
    segments = [(centers[a], centers[b]) for a, b in network.interfaces]
    for pore in network.pores:
        for port in pore.ports:
            if port.kind == "inlet":
                boundary = np.asarray([0.0, pore.center[1], pore.center[2]], dtype=float)
                segments.append((boundary, centers[pore.id]))
            elif port.kind == "outlet":
                boundary = np.asarray([domain_size, pore.center[1], pore.center[2]], dtype=float)
                segments.append((centers[pore.id], boundary))
    return segments


def point_in_union(
    point: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    segments: list[tuple[np.ndarray, np.ndarray]],
    throat_radius: float,
) -> bool:
    if np.any(np.linalg.norm(centers - point, axis=1) <= radii):
        return True
    radius2 = throat_radius * throat_radius
    return any(point_segment_distance2(point, a, b) <= radius2 for a, b in segments)


def point_segment_distance2(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.dot(point - a, point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.dot(point - closest, point - closest))


def grid_id(i: int, j: int, k: int, cells_per_axis: int) -> int:
    n = cells_per_axis + 1
    return (i * n + j) * n + k


def orient_tets(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    oriented = cells.copy()
    for idx, tet in enumerate(oriented):
        mat = np.column_stack(
            (
                coords[tet[1]] - coords[tet[0]],
                coords[tet[2]] - coords[tet[0]],
                coords[tet[3]] - coords[tet[0]],
            )
        )
        if np.linalg.det(mat) < 0:
            oriented[idx, [2, 3]] = oriented[idx, [3, 2]]
    return oriented


def boundary_facets_by_x(msh, domain_size: float) -> tuple[np.ndarray, np.ndarray]:
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    exterior = dmesh.exterior_facet_indices(msh.topology)
    facet_to_vertices = msh.topology.connectivity(fdim, 0)
    coords = msh.geometry.x
    h = domain_size / max(1, round(domain_size / min_positive_spacing(coords[:, 0])))
    tol = max(1.0e-10, 0.51 * h)

    inlet: list[int] = []
    outlet: list[int] = []
    for facet in exterior:
        midpoint = coords[facet_to_vertices.links(facet)].mean(axis=0)
        if midpoint[0] <= tol:
            inlet.append(int(facet))
        elif midpoint[0] >= domain_size - tol:
            outlet.append(int(facet))
    return np.asarray(inlet, dtype=np.int32), np.asarray(outlet, dtype=np.int32)


def min_positive_spacing(values: np.ndarray) -> float:
    unique = np.unique(np.round(values, decimals=12))
    diff = np.diff(np.sort(unique))
    positive = diff[diff > 0.0]
    if len(positive) == 0:
        return 1.0
    return float(positive.min())

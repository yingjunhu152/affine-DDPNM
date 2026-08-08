from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import gmsh
import numpy as np
from dolfinx import mesh as dmesh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from scipy.optimize import brentq


DOMAIN_MIN = 0.0
DOMAIN_MAX = 1.0
SPHERE_GRID = np.asarray([0.20, 0.50, 0.80], dtype=float)
PARTITION_PLANES = SPHERE_GRID.copy()
CELL_EDGES = np.asarray([0.0, *SPHERE_GRID, 1.0], dtype=float)
SPHERE_RADIUS = 0.105
SPHERE_CENTERS = np.asarray(
    list(product(SPHERE_GRID, SPHERE_GRID, SPHERE_GRID)), dtype=float
)


@dataclass(frozen=True)
class MaximalBall:
    center: tuple[float, float, float]
    radius: float
    touching_spheres: tuple[int, ...]
    touching_boundaries: tuple[str, ...]
    lattice_cell: tuple[int, int, int]


@dataclass(frozen=True)
class Throat:
    pore_i: int
    pore_j: int
    saddle: tuple[float, float, float]
    normal: tuple[float, float, float]
    clearance: float
    touching_spheres: tuple[int, ...]


@dataclass
class PartitionData:
    mesh: dmesh.Mesh
    cell_centers: np.ndarray
    cell_clearance: np.ndarray
    cell_labels: np.ndarray
    maximal_balls: tuple[MaximalBall, ...]
    throats: tuple[Throat, ...]
    interface_pairs: tuple[tuple[int, int], ...]
    facet_interface_ids: np.ndarray
    interface_centers: np.ndarray
    interface_normals: np.ndarray
    interface_areas: np.ndarray
    mesh_parameters: dict[str, float]
    cad_counts: dict[str, int]

    @property
    def pore_seeds(self) -> np.ndarray:
        return np.asarray(
            [(*ball.center, ball.radius) for ball in self.maximal_balls], dtype=float
        )


def clearance(points: np.ndarray) -> np.ndarray:
    """Distance to the closest solid sphere or outer cube boundary."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    result = np.minimum(np.min(pts, axis=1), np.min(1.0 - pts, axis=1))
    for center in SPHERE_CENTERS:
        result = np.minimum(
            result, np.linalg.norm(pts - center[None, :], axis=1) - SPHERE_RADIUS
        )
    return result


def _boundary_constrained_radius(boundary_axes: int, dimension: int) -> float:
    """Solve the equal-clearance condition for a boundary-adjacent void cell."""
    if boundary_axes == 0:
        return float(np.sqrt(dimension) * 0.15 - SPHERE_RADIUS)
    interior_axes = dimension - boundary_axes

    def equation(value: float) -> float:
        grain_distance = np.sqrt(
            boundary_axes * (0.20 - value) ** 2
            + interior_axes * 0.15**2
        )
        return float(grain_distance - SPHERE_RADIUS - value)

    return float(brentq(equation, 0.0, 0.20, xtol=1.0e-14, rtol=1.0e-14))


def _cell_coordinate(index: int, boundary_radius: float) -> float:
    if index == 0:
        return boundary_radius
    if index == 3:
        return 1.0 - boundary_radius
    return 0.5 * float(CELL_EDGES[index] + CELL_EDGES[index + 1])


def _touching_outer_boundaries(center: np.ndarray, radius: float) -> tuple[str, ...]:
    names = ("x0", "y0", "z0", "x1", "y1", "z1")
    distances = np.asarray([*center, *(1.0 - center)], dtype=float)
    return tuple(
        names[i] for i in np.flatnonzero(np.abs(distances - radius) <= 2.0e-10)
    )


def maximal_balls_from_uniform_lattice() -> tuple[MaximalBall, ...]:
    """Compute all 4x4x4 maximal empty balls, including boundary void cells.

    The three grain-centre planes on every coordinate axis divide the cube into
    four clearance cells.  Interior cells are constrained only by grains;
    boundary cells are constrained jointly by grains and one, two or three cube
    walls.  Equal-clearance equations give the exact centres and radii.
    """
    balls: list[MaximalBall] = []
    for ix, iy, iz in product(range(4), repeat=3):
        lattice_cell = (ix, iy, iz)
        boundary_axes = sum(index in (0, 3) for index in lattice_cell)
        radius = _boundary_constrained_radius(boundary_axes, dimension=3)
        center = np.asarray(
            [
                _cell_coordinate(ix, radius),
                _cell_coordinate(iy, radius),
                _cell_coordinate(iz, radius),
            ],
            dtype=float,
        )
        grain_clearances = (
            np.linalg.norm(SPHERE_CENTERS - center[None, :], axis=1)
            - SPHERE_RADIUS
        )
        measured_radius = float(
            min(np.min(grain_clearances), np.min(center), np.min(1.0 - center))
        )
        if abs(measured_radius - radius) > 2.0e-10:
            raise RuntimeError(
                f"Analytic maximal-ball radius mismatch in cell {lattice_cell}."
            )
        touching = tuple(
            int(i)
            for i in np.flatnonzero(np.abs(grain_clearances - radius) <= 1.0e-10)
        )
        touching_boundaries = _touching_outer_boundaries(center, radius)
        if radius <= 0.0 or len(touching) + len(touching_boundaries) < 4:
            raise RuntimeError("The prescribed lattice did not produce a valid maximal ball.")
        balls.append(
            MaximalBall(
                center=tuple(float(x) for x in center),
                radius=radius,
                touching_spheres=touching,
                touching_boundaries=touching_boundaries,
                lattice_cell=lattice_cell,
            )
        )
    if len(balls) != 64:
        raise RuntimeError(f"Expected 64 maximal balls, found {len(balls)}.")
    return tuple(balls)


def maximal_ball_graph(balls: tuple[MaximalBall, ...]) -> tuple[Throat, ...]:
    """Connect the 4x4x4 six-neighbour graph and compute exact saddle points."""
    cell_to_id = {ball.lattice_cell: i for i, ball in enumerate(balls)}
    throats: list[Throat] = []
    for cell, i in sorted(cell_to_id.items(), key=lambda item: item[1]):
        for axis in range(3):
            if cell[axis] >= 3:
                continue
            neighbor = list(cell)
            neighbor[axis] += 1
            j = cell_to_id[tuple(neighbor)]
            normal = np.zeros(3, dtype=float)
            normal[axis] = 1.0
            tangential_axes = [value for value in range(3) if value != axis]
            tangential_cells = [cell[value] for value in tangential_axes]
            boundary_axes = sum(index in (0, 3) for index in tangential_cells)
            throat_radius = _boundary_constrained_radius(boundary_axes, dimension=2)
            saddle = np.empty(3, dtype=float)
            saddle[axis] = float(PARTITION_PLANES[cell[axis]])
            for tangent_axis, tangent_cell in zip(
                tangential_axes, tangential_cells, strict=True
            ):
                saddle[tangent_axis] = _cell_coordinate(tangent_cell, throat_radius)
            grain_clearances = (
                np.linalg.norm(SPHERE_CENTERS - saddle[None, :], axis=1)
                - SPHERE_RADIUS
            )
            measured_radius = float(
                min(np.min(grain_clearances), np.min(saddle), np.min(1.0 - saddle))
            )
            if abs(measured_radius - throat_radius) > 2.0e-10:
                raise RuntimeError(f"Analytic throat radius mismatch for edge {(i, j)}.")
            touching = tuple(
                int(k)
                for k in np.flatnonzero(
                    np.abs(grain_clearances - throat_radius) <= 1.0e-10
                )
            )
            throats.append(
                Throat(
                    pore_i=i,
                    pore_j=j,
                    saddle=tuple(float(x) for x in saddle),
                    normal=tuple(float(x) for x in normal),
                    clearance=throat_radius,
                    touching_spheres=touching,
                )
            )
    if len(throats) != 144:
        raise RuntimeError(f"Expected 144 six-neighbour throats, found {len(throats)}.")
    return tuple(sorted(throats, key=lambda throat: (throat.pore_i, throat.pore_j)))


def _add_plane_surface(axis: int, coordinate: float) -> int:
    if axis == 0:
        xyz = [
            (coordinate, 0.0, 0.0),
            (coordinate, 1.0, 0.0),
            (coordinate, 1.0, 1.0),
            (coordinate, 0.0, 1.0),
        ]
    elif axis == 1:
        xyz = [
            (0.0, coordinate, 0.0),
            (1.0, coordinate, 0.0),
            (1.0, coordinate, 1.0),
            (0.0, coordinate, 1.0),
        ]
    elif axis == 2:
        xyz = [
            (0.0, 0.0, coordinate),
            (1.0, 0.0, coordinate),
            (1.0, 1.0, coordinate),
            (0.0, 1.0, coordinate),
        ]
    else:
        raise ValueError(axis)
    points = [gmsh.model.occ.addPoint(*point) for point in xyz]
    lines = [
        gmsh.model.occ.addLine(points[k], points[(k + 1) % 4]) for k in range(4)
    ]
    loop = gmsh.model.occ.addCurveLoop(lines)
    return gmsh.model.occ.addPlaneSurface([loop])


def _octant_label_from_point(point: np.ndarray) -> int:
    ix, iy, iz = (
        int(np.searchsorted(PARTITION_PLANES, point[k], side="right"))
        for k in range(3)
    )
    return 16 * ix + 4 * iy + iz


def _outer_side(surface: int, tolerance: float = 5.0e-7) -> str | None:
    center = np.asarray(gmsh.model.occ.getCenterOfMass(2, surface), dtype=float)
    if abs(center[0]) < tolerance:
        return "inlet"
    if abs(center[0] - 1.0) < tolerance:
        return "outlet"
    if (
        abs(center[1]) < tolerance
        or abs(center[1] - 1.0) < tolerance
        or abs(center[2]) < tolerance
        or abs(center[2] - 1.0) < tolerance
    ):
        return "outer_wall"
    return None


def _distance_threshold(
    surfaces: list[int], size_min: float, size_max: float, band: float
) -> int | None:
    if not surfaces:
        return None
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", surfaces)
    gmsh.model.mesh.field.setNumber(distance, "Sampling", 120)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", size_min)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", size_max)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", band)
    return threshold


def generate_partition_mesh(
    bulk_size: float = 0.20,
    sphere_size: float = 0.040,
    boundary_size: float = 0.070,
    interface_size: float = 0.055,
    sphere_band: float = 0.10,
    boundary_band: float = 0.08,
    interface_band: float = 0.08,
    mesh_file: Path | None = None,
) -> tuple[
    dmesh.Mesh,
    dmesh.MeshTags,
    tuple[MaximalBall, ...],
    tuple[Throat, ...],
    dict[str, int],
]:
    """Build the sphere-subtracted, 64-region conforming tetrahedral mesh."""
    balls = maximal_balls_from_uniform_lattice()
    throats = maximal_ball_graph(balls)
    pair_to_interface = {
        (throat.pore_i, throat.pore_j): i for i, throat in enumerate(throats)
    }

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("ddpnm_3d_uniform_27_spheres")
        box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        grains = [
            gmsh.model.occ.addSphere(float(x), float(y), float(z), SPHERE_RADIUS)
            for x, y, z in SPHERE_CENTERS
        ]
        fluid, _ = gmsh.model.occ.cut(
            [(3, box)],
            [(3, tag) for tag in grains],
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()
        split_surfaces = [
            _add_plane_surface(axis, float(coordinate))
            for axis in range(3)
            for coordinate in PARTITION_PLANES
        ]
        gmsh.model.occ.fragment(
            fluid,
            [(2, tag) for tag in split_surfaces],
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()

        volumes = [tag for _, tag in gmsh.model.getEntities(3)]
        volumes_by_label: dict[int, list[int]] = {i: [] for i in range(len(balls))}
        volume_to_label: dict[int, int] = {}
        for volume in volumes:
            center = np.asarray(gmsh.model.occ.getCenterOfMass(3, volume), dtype=float)
            label = _octant_label_from_point(center)
            volumes_by_label[label].append(volume)
            volume_to_label[volume] = label
        missing = [label for label, tags in volumes_by_label.items() if not tags]
        if missing:
            raise RuntimeError(f"Gmsh partition is missing pore regions {missing}.")
        for label, tags in volumes_by_label.items():
            physical = gmsh.model.addPhysicalGroup(3, tags, tag=1000 + label)
            gmsh.model.setPhysicalName(3, physical, f"pore_{label:02d}")

        sphere_surfaces: list[int] = []
        cube_surfaces: dict[str, list[int]] = {
            "inlet": [],
            "outlet": [],
            "outer_wall": [],
        }
        interface_surfaces: dict[int, list[int]] = {
            i: [] for i in range(len(throats))
        }
        orphan_surfaces = 0
        for _, surface in gmsh.model.getEntities(2):
            upward, _ = gmsh.model.getAdjacencies(2, surface)
            adjacent = [int(volume) for volume in upward if int(volume) in volume_to_label]
            if len(adjacent) == 2:
                pair = tuple(sorted(volume_to_label[volume] for volume in adjacent))
                interface_id = pair_to_interface.get(pair)
                if interface_id is None:
                    raise RuntimeError(f"Unexpected adjacent pore pair {pair}.")
                interface_surfaces[interface_id].append(surface)
            elif len(adjacent) == 1:
                surface_type = gmsh.model.getType(2, surface).lower()
                if "sphere" in surface_type:
                    sphere_surfaces.append(surface)
                elif "plane" in surface_type:
                    side = _outer_side(surface)
                    if side is None:
                        raise RuntimeError(
                            f"Planar exterior surface {surface} is not on the outer cube."
                        )
                    cube_surfaces[side].append(surface)
                else:
                    raise RuntimeError(
                        f"Unexpected exterior CAD surface type {surface_type!r}."
                    )
            elif len(adjacent) == 0:
                orphan_surfaces += 1
            else:
                raise RuntimeError(
                    f"Surface {surface} has {len(adjacent)} adjacent fluid volumes."
                )

        expected_counts = {
            "sphere_surface_patches": 216,
            "inlet_surface_patches": 16,
            "outlet_surface_patches": 16,
            "outer_wall_surface_patches": 64,
            "interface_surface_patches": 144,
        }
        measured_counts = {
            "sphere_surface_patches": len(sphere_surfaces),
            "inlet_surface_patches": len(cube_surfaces["inlet"]),
            "outlet_surface_patches": len(cube_surfaces["outlet"]),
            "outer_wall_surface_patches": len(cube_surfaces["outer_wall"]),
            "interface_surface_patches": sum(
                len(tags) for tags in interface_surfaces.values()
            ),
        }
        for name, expected in expected_counts.items():
            if measured_counts[name] != expected:
                raise RuntimeError(
                    f"Expected {expected} {name}, found {measured_counts[name]}."
                )
        if any(len(tags) != 1 for tags in interface_surfaces.values()):
            bad = [i for i, tags in interface_surfaces.items() if len(tags) != 1]
            raise RuntimeError(f"Interfaces without exactly one CAD patch: {bad}.")
        cad_counts = {
            **measured_counts,
            "orphan_tool_surface_patches": orphan_surfaces,
            "fluid_volumes": len(volumes),
        }

        for name, tag, surfaces in [
            ("sphere_walls", 1, sphere_surfaces),
            ("inlet", 2, cube_surfaces["inlet"]),
            ("outlet", 3, cube_surfaces["outlet"]),
            ("outer_walls", 4, cube_surfaces["outer_wall"]),
        ]:
            if surfaces:
                physical = gmsh.model.addPhysicalGroup(2, surfaces, tag=tag)
                gmsh.model.setPhysicalName(2, physical, name)
        for interface_id, surfaces in interface_surfaces.items():
            physical = gmsh.model.addPhysicalGroup(
                2, surfaces, tag=2000 + interface_id
            )
            gmsh.model.setPhysicalName(2, physical, f"interface_{interface_id:02d}")

        fields = [
            _distance_threshold(
                sphere_surfaces, sphere_size, bulk_size, sphere_band
            ),
            _distance_threshold(
                cube_surfaces["inlet"]
                + cube_surfaces["outlet"]
                + cube_surfaces["outer_wall"],
                boundary_size,
                bulk_size,
                boundary_band,
            ),
            _distance_threshold(
                [surface for tags in interface_surfaces.values() for surface in tags],
                interface_size,
                bulk_size,
                interface_band,
            ),
        ]
        active_fields = [field for field in fields if field is not None]
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", active_fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

        gmsh.option.setNumber(
            "Mesh.MeshSizeMin", min(sphere_size, boundary_size, interface_size)
        )
        gmsh.option.setNumber("Mesh.MeshSizeMax", bulk_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)
        if mesh_file is not None:
            mesh_file.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(mesh_file.resolve()))
        data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=3)
        if data.cell_tags is None:
            raise RuntimeError("Gmsh did not return physical pore-volume tags.")
        return data.mesh, data.cell_tags, balls, throats, cad_counts
    finally:
        gmsh.finalize()


def cell_geometry(msh: dmesh.Mesh) -> tuple[np.ndarray, np.ndarray]:
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    n_cells = msh.topology.index_map(tdim).size_local
    centers = np.empty((n_cells, 3), dtype=float)
    for cell in range(n_cells):
        centers[cell] = msh.geometry.x[c2v.links(cell), :3].mean(axis=0)
    return centers, clearance(centers)


def normalize_cell_tags(cell_tags: dmesh.MeshTags, n_cells: int) -> np.ndarray:
    raw = np.full(n_cells, -1, dtype=np.int32)
    raw[cell_tags.indices] = cell_tags.values
    if np.any(raw < 0):
        raise RuntimeError("Some fluid tetrahedra have no pore-region physical tag.")
    labels = raw - 1000
    if np.any(labels < 0) or np.any(labels >= 64):
        raise RuntimeError(f"Unexpected physical volume tags: {sorted(np.unique(raw))}.")
    return labels.astype(np.int32)


def find_interfaces(
    msh: dmesh.Mesh, labels: np.ndarray
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    f2c = msh.topology.connectivity(fdim, tdim)
    n_facets = msh.topology.index_map(fdim).size_local
    pairs: set[tuple[int, int]] = set()
    facet_pairs: list[tuple[int, int] | None] = [None] * n_facets
    for facet in range(n_facets):
        cells = f2c.links(facet)
        if len(cells) != 2:
            continue
        a, b = int(labels[cells[0]]), int(labels[cells[1]])
        if a != b:
            pair = (min(a, b), max(a, b))
            pairs.add(pair)
            facet_pairs[facet] = pair
    ordered = tuple(sorted(pairs))
    pair_to_id = {pair: i for i, pair in enumerate(ordered)}
    facet_ids = np.full(n_facets, -1, dtype=np.int32)
    for facet, pair in enumerate(facet_pairs):
        if pair is not None:
            facet_ids[facet] = pair_to_id[pair]
    return ordered, facet_ids


def interface_geometry(
    msh: dmesh.Mesh,
    facet_ids: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    balls: tuple[MaximalBall, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    centers = np.zeros((len(pairs), 3), dtype=float)
    normals = np.zeros((len(pairs), 3), dtype=float)
    areas = np.zeros(len(pairs), dtype=float)
    seeds = np.asarray([ball.center for ball in balls], dtype=float)
    for interface_id, (a, b) in enumerate(pairs):
        facets = np.flatnonzero(facet_ids == interface_id)
        if not len(facets):
            raise RuntimeError(f"Interface {interface_id} has no mesh facets.")
        for facet in facets:
            triangle = msh.geometry.x[f2v.links(int(facet)), :3]
            area = 0.5 * float(
                np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
            )
            centers[interface_id] += area * triangle.mean(axis=0)
            areas[interface_id] += area
        centers[interface_id] /= areas[interface_id]
        direction = seeds[b] - seeds[a]
        normals[interface_id] = direction / np.linalg.norm(direction)
    return centers, normals, areas


def build_partition(
    mesh_size: float = 0.20,
    sphere_size: float = 0.040,
    boundary_size: float = 0.070,
    interface_size: float = 0.055,
    sphere_band: float = 0.10,
    boundary_band: float = 0.08,
    interface_band: float = 0.08,
    mesh_file: Path | None = None,
) -> PartitionData:
    msh, cell_tags, balls, throats, cad_counts = generate_partition_mesh(
        bulk_size=mesh_size,
        sphere_size=sphere_size,
        boundary_size=boundary_size,
        interface_size=interface_size,
        sphere_band=sphere_band,
        boundary_band=boundary_band,
        interface_band=interface_band,
        mesh_file=mesh_file,
    )
    centers, cell_clearance = cell_geometry(msh)
    labels = normalize_cell_tags(cell_tags, len(centers))
    pairs, facet_ids = find_interfaces(msh, labels)
    expected_pairs = tuple(sorted((t.pore_i, t.pore_j) for t in throats))
    if pairs != expected_pairs:
        raise RuntimeError(
            f"CAD interface graph {pairs} does not match maximal-ball graph {expected_pairs}."
        )
    interface_centers, interface_normals, interface_areas = interface_geometry(
        msh, facet_ids, pairs, balls
    )
    return PartitionData(
        mesh=msh,
        cell_centers=centers,
        cell_clearance=cell_clearance,
        cell_labels=labels,
        maximal_balls=balls,
        throats=throats,
        interface_pairs=pairs,
        facet_interface_ids=facet_ids,
        interface_centers=interface_centers,
        interface_normals=interface_normals,
        interface_areas=interface_areas,
        mesh_parameters={
            "bulk_size": mesh_size,
            "sphere_size": sphere_size,
            "boundary_size": boundary_size,
            "interface_size": interface_size,
            "sphere_band": sphere_band,
            "boundary_band": boundary_band,
            "interface_band": interface_band,
        },
        cad_counts=cad_counts,
    )

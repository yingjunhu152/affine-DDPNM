from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np
from dolfinx import mesh as dmesh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from scipy.spatial import Delaunay


# Exact particle realization used in the four-panel figure generated with seed
# 20260802. Coordinates are normalized to the unit square. Boundary particles
# deliberately extend outside the square and are clipped by the sample window.
PARTICLES = np.asarray(
    [
        (-0.025000000000, 0.100000000000, 0.145000000000),
        (0.310000000000, -0.020000000000, 0.115000000000),
        (0.720000000000, -0.025000000000, 0.135000000000),
        (1.035000000000, 0.120000000000, 0.155000000000),
        (-0.030000000000, 0.550000000000, 0.145000000000),
        (1.030000000000, 0.580000000000, 0.135000000000),
        (0.120000000000, 1.030000000000, 0.155000000000),
        (0.630000000000, 1.035000000000, 0.165000000000),
        (0.516662023180, 0.558084178298, 0.100665971996),
        (0.412212863712, 0.219942972094, 0.059762386225),
        (0.306120272148, 0.722710117180, 0.093650812401),
        (0.200757990595, 0.321502976092, 0.091168296428),
        (0.682661167231, 0.250664375584, 0.067195923857),
        (0.749919165440, 0.662971023340, 0.078565350203),
        (0.858975918274, 0.827226326783, 0.057802603839),
        (0.721216152710, 0.456233946715, 0.067402000641),
        (0.898977237350, 0.363928216045, 0.058511863168),
    ],
    dtype=float,
)


@dataclass(frozen=True)
class AnalyticCut:
    particle_i: int
    particle_j: int
    start: tuple[float, float]
    end: tuple[float, float]
    saddle: tuple[float, float]
    half_width: float


@dataclass
class PartitionData:
    mesh: dmesh.Mesh
    cell_centers: np.ndarray
    cell_clearance: np.ndarray
    pore_seeds: np.ndarray
    cell_labels: np.ndarray
    interface_pairs: tuple[tuple[int, int], ...]
    facet_interface_ids: np.ndarray
    interface_centers: np.ndarray
    interface_tangents: np.ndarray
    interface_half_lengths: np.ndarray
    analytic_cuts: tuple[AnalyticCut, ...]
    mesh_parameters: dict[str, float]


def clearance(points: np.ndarray) -> np.ndarray:
    """Distance to the closest solid or outer sample boundary."""
    pts = np.atleast_2d(points)
    c = np.minimum.reduce(
        [pts[:, 0], 1.0 - pts[:, 0], pts[:, 1], 1.0 - pts[:, 1]]
    )
    for x, y, radius in PARTICLES:
        c = np.minimum(c, np.hypot(pts[:, 0] - x, pts[:, 1] - y) - radius)
    return c


def analytic_throat_cuts() -> tuple[AnalyticCut, ...]:
    """Construct strict circular-particle throat sections from Delaunay neighbors.

    For neighboring circles i and j, the interface is the straight gap segment on
    their center line. Its center is the equal-distance saddle of the distance-to-solid
    function and its endpoints are the nearest points on the two circle boundaries.
    """
    triangulation = Delaunay(PARTICLES[:, :2])
    edges: set[tuple[int, int]] = set()
    for simplex in triangulation.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                edges.add(tuple(sorted((int(simplex[a]), int(simplex[b])))))
    cuts: list[AnalyticCut] = []
    for i, j in sorted(edges):
        a, b = PARTICLES[i], PARTICLES[j]
        delta = b[:2] - a[:2]
        distance = float(np.linalg.norm(delta))
        tangent = delta / distance
        gap = distance - float(a[2]) - float(b[2])
        if gap <= 1.0e-7:
            continue
        start = a[:2] + tangent * a[2]
        end = b[:2] - tangent * b[2]
        saddle_distance_from_a = 0.5 * (distance + float(a[2]) - float(b[2]))
        saddle = a[:2] + tangent * saddle_distance_from_a
        midpoint = 0.5 * (start + end)
        if not np.all(midpoint >= -1.0e-10) or not np.all(midpoint <= 1.0 + 1.0e-10):
            continue
        # Reject a nominal Delaunay edge if its gap segment crosses a third particle.
        valid = True
        for k, particle in enumerate(PARTICLES):
            if k in (i, j):
                continue
            samples = np.linspace(0.04, 0.96, 13)[:, None] * end + np.linspace(0.96, 0.04, 13)[:, None] * start
            if np.any(np.linalg.norm(samples - particle[:2], axis=1) < particle[2] - 1.0e-9):
                valid = False
                break
        if valid:
            cuts.append(
                AnalyticCut(
                    particle_i=i,
                    particle_j=j,
                    start=(float(start[0]), float(start[1])),
                    end=(float(end[0]), float(end[1])),
                    saddle=(float(saddle[0]), float(saddle[1])),
                    half_width=0.5 * gap,
                )
            )
    return tuple(cuts)


def _configure_distance_refinement(
    circle_curves: list[int],
    throat_curves: list[int],
    bulk_size: float,
    wall_size: float,
    throat_size: float,
    wall_band: float,
    throat_band: float,
) -> None:
    fields: list[int] = []
    if circle_curves:
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "CurvesList", circle_curves)
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 120)
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", wall_size)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", bulk_size)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", wall_band)
        fields.append(threshold)
    if throat_curves:
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "CurvesList", throat_curves)
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 80)
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", throat_size)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", bulk_size)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", throat_band)
        fields.append(threshold)
    if fields:
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)


def generate_analytic_partition_mesh(
    bulk_size: float = 0.04,
    wall_size: float = 0.018,
    throat_size: float = 0.010,
    wall_band: float = 0.065,
    throat_band: float = 0.045,
) -> tuple[dmesh.Mesh, dmesh.MeshTags, tuple[AnalyticCut, ...]]:
    """Create an OCC-fragmented conforming mesh with analytic throat cuts."""
    cuts = analytic_throat_cuts()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("analytic_saddle_cut_porous_medium_2d")
        rectangle = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0)
        disks = [
            gmsh.model.occ.addDisk(float(x), float(y), 0.0, float(r), float(r))
            for x, y, r in PARTICLES
        ]
        void_entities, _ = gmsh.model.occ.cut(
            [(2, rectangle)], [(2, tag) for tag in disks],
            removeObject=True, removeTool=True,
        )
        gmsh.model.occ.synchronize()
        cut_lines: list[int] = []
        for cut in cuts:
            p0 = gmsh.model.occ.addPoint(cut.start[0], cut.start[1], 0.0)
            p1 = gmsh.model.occ.addPoint(cut.end[0], cut.end[1], 0.0)
            cut_lines.append(gmsh.model.occ.addLine(p0, p1))
        gmsh.model.occ.fragment(
            void_entities, [(1, tag) for tag in cut_lines],
            removeObject=True, removeTool=True,
        )
        gmsh.model.occ.synchronize()

        surfaces = gmsh.model.getEntities(2)
        if len(surfaces) < 2:
            raise RuntimeError("Analytic throat curves did not split the pore space.")
        for local_id, (_, surface) in enumerate(sorted(surfaces), start=1):
            physical = gmsh.model.addPhysicalGroup(2, [surface], 1000 + local_id)
            gmsh.model.setPhysicalName(2, physical, f"pore_{local_id:03d}")

        circle_curves: list[int] = []
        throat_curves: list[int] = []
        for _, curve in gmsh.model.getEntities(1):
            curve_type = gmsh.model.getType(1, curve).lower()
            if "circle" in curve_type or "ellipse" in curve_type:
                circle_curves.append(curve)
                continue
            bbox = gmsh.model.getBoundingBox(1, curve)
            xmin, ymin, _, xmax, ymax, _ = bbox
            on_outer = (
                abs(xmin) < 1.0e-8 and abs(xmax) < 1.0e-8
                or abs(xmin - 1.0) < 1.0e-8 and abs(xmax - 1.0) < 1.0e-8
                or abs(ymin) < 1.0e-8 and abs(ymax) < 1.0e-8
                or abs(ymin - 1.0) < 1.0e-8 and abs(ymax - 1.0) < 1.0e-8
            )
            if not on_outer:
                throat_curves.append(curve)

        _configure_distance_refinement(
            circle_curves, throat_curves, bulk_size, wall_size, throat_size,
            wall_band, throat_band,
        )
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(wall_size, throat_size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", bulk_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=2)
        if data.cell_tags is None:
            raise RuntimeError("Gmsh did not return pore-region cell tags.")
        return data.mesh, data.cell_tags, cuts
    finally:
        gmsh.finalize()


def cell_geometry(msh: dmesh.Mesh) -> tuple[np.ndarray, np.ndarray]:
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    n_cells = msh.topology.index_map(tdim).size_local
    centers = np.empty((n_cells, 2), dtype=float)
    for cell in range(n_cells):
        centers[cell] = msh.geometry.x[c2v.links(cell), :2].mean(axis=0)
    return centers, clearance(centers)


def normalize_cell_tags(cell_tags: dmesh.MeshTags, n_cells: int) -> np.ndarray:
    raw = np.full(n_cells, -1, dtype=np.int32)
    raw[cell_tags.indices] = cell_tags.values
    if np.any(raw < 0):
        raise RuntimeError("Some pore-space cells have no analytic subdomain tag.")
    unique = sorted(int(value) for value in np.unique(raw))
    mapping = {value: i for i, value in enumerate(unique)}
    return np.asarray([mapping[int(value)] for value in raw], dtype=np.int32)


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
    msh: dmesh.Mesh, facet_ids: np.ndarray, n_interfaces: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    centers = np.empty((n_interfaces, 2), dtype=float)
    tangents = np.empty((n_interfaces, 2), dtype=float)
    half_lengths = np.empty(n_interfaces, dtype=float)
    for interface_id in range(n_interfaces):
        facets = np.flatnonzero(facet_ids == interface_id)
        vertices = np.unique(np.concatenate([f2v.links(int(f)) for f in facets]))
        points = msh.geometry.x[vertices, :2]
        delta = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        a, b = np.unravel_index(int(np.argmax(distances)), distances.shape)
        start, end = points[a], points[b]
        if tuple(end) < tuple(start):
            start, end = end, start
        vector = end - start
        length = float(np.linalg.norm(vector))
        centers[interface_id] = 0.5 * (start + end)
        tangents[interface_id] = vector / length
        half_lengths[interface_id] = 0.5 * length
    return centers, tangents, half_lengths


def pore_seeds_from_regions(
    centers: np.ndarray, cell_clearance: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    seeds = []
    for label in sorted(int(x) for x in np.unique(labels)):
        cells = np.flatnonzero(labels == label)
        cell = int(cells[np.argmax(cell_clearance[cells])])
        seeds.append((centers[cell, 0], centers[cell, 1], cell_clearance[cell]))
    return np.asarray(seeds, dtype=float)


def build_partition(
    mesh_size: float = 0.04,
    max_pores: int | None = None,
    wall_size: float = 0.018,
    throat_size: float = 0.010,
    wall_band: float = 0.065,
    throat_band: float = 0.045,
) -> PartitionData:
    # max_pores is retained only for command-line compatibility with the baseline;
    # strict analytic cuts determine the number of pore regions automatically.
    del max_pores
    msh, cell_tags, cuts = generate_analytic_partition_mesh(
        bulk_size=mesh_size,
        wall_size=wall_size,
        throat_size=throat_size,
        wall_band=wall_band,
        throat_band=throat_band,
    )
    centers, cell_clearance = cell_geometry(msh)
    labels = normalize_cell_tags(cell_tags, len(centers))
    pairs, facet_ids = find_interfaces(msh, labels)
    iface_centers, iface_tangents, iface_half_lengths = interface_geometry(
        msh, facet_ids, len(pairs)
    )
    seeds = pore_seeds_from_regions(centers, cell_clearance, labels)
    return PartitionData(
        mesh=msh,
        cell_centers=centers,
        cell_clearance=cell_clearance,
        pore_seeds=seeds,
        cell_labels=labels,
        interface_pairs=pairs,
        facet_interface_ids=facet_ids,
        interface_centers=iface_centers,
        interface_tangents=iface_tangents,
        interface_half_lengths=iface_half_lengths,
        analytic_cuts=cuts,
        mesh_parameters={
            "bulk_size": mesh_size,
            "wall_size": wall_size,
            "throat_size": throat_size,
            "wall_band": wall_band,
            "throat_band": throat_band,
        },
    )

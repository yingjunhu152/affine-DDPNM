"""Random-sphere 3-D partition following the ddpnm_2d_random_porous recipe.

Geometry generation mirrors the 2-D project exactly:

1. a fixed-seed random packing is frozen as the ``SPHERES`` table (18
   boundary spheres deliberately extend outside the unit cube and are
   clipped by the sample window, 9 interior spheres);
2. Delaunay-neighbouring spheres define candidate pore throats; for every
   valid pair the exact center-line gap segment between the two sphere
   walls is constructed, whose equal-clearance point is the analytic
   throat saddle (a nominal pair is rejected when the gap segment crosses
   a third sphere);
3. each valid throat is cut by the planar face of the saddle plane
   perpendicular to the center line.  A 1-D gap segment cannot separate a
   volume, so the 3-D interface is the part of that plane belonging to the
   pair, i.e. the Voronoi face of the two sphere centres (bounded by the
   neighbouring saddle planes and the cube wall) — the same construction
   whose regular-grid special case the 3-D uniform folder uses;
4. the faces are embedded in the OpenCASCADE fluid volume and the whole
   domain is fragmented once, giving one conforming tetrahedral mesh with
   analytic subdomain interfaces (no mesh conversion error).

The generated partition provides the same :class:`PartitionData` fields as
``ddpnm3d.geometry`` so the shared ``ddpnm_core`` solver pipeline applies
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import gmsh
import numpy as np
from dolfinx import mesh as dmesh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from scipy.spatial import Delaunay, Voronoi

# Exact particle realization generated with seed 20260804.  Boundary spheres
# deliberately extend outside the cube and are clipped by the sample window.
SPHERES = np.asarray(
    [
        (-0.043446823437, 0.255098198896, 0.579697492716, 0.115647834689),
        (-0.022525418045, 0.677830945616, 0.219223768473, 0.118409228932),
        (-0.043781618863, 0.349694676678, 0.222024474857, 0.110938933607),
        (1.025008485930, 0.219943441316, 0.638457436843, 0.127327508536),
        (1.041188247603, 0.741012904832, 0.356876999359, 0.128707458160),
        (1.023692742419, 0.694507435115, 0.763739294844, 0.120930222968),
        (0.557883620054, -0.043900624072, 0.677069367654, 0.127935933194),
        (0.255561032155, -0.035369496929, 0.265725628023, 0.122692495239),
        (0.751519743114, -0.027254035026, 0.394198434795, 0.133351556208),
        (0.211711346721, 1.023722295521, 0.543351782392, 0.117185822135),
        (0.538536537859, 1.023941674985, 0.519863980489, 0.115264323036),
        (0.264203481210, 1.027830944860, 0.253600686685, 0.111410857067),
        (0.764156614542, 0.651078306967, -0.033161305707, 0.123270789319),
        (0.245717562709, 0.319303250620, -0.034430838380, 0.125602419068),
        (0.723490293832, 0.383092875425, -0.035408066318, 0.111147219385),
        (0.458669960743, 0.720323126604, 1.037336934465, 0.117644865875),
        (0.762755875480, 0.629324556256, 1.020522753714, 0.113140079974),
        (0.587050996920, 0.374669048605, 1.040390431139, 0.110876776447),
        (0.653367023935, 0.404527200765, 0.698334496808, 0.084105352717),
        (0.533084816370, 0.197341275614, 0.288681924368, 0.078758358654),
        (0.773093711751, 0.840918133612, 0.630971887756, 0.081652336082),
        (0.406145288086, 0.239228181591, 0.713831951804, 0.087890567256),
        (0.779471058760, 0.863882348867, 0.332126373971, 0.085157373964),
        (0.844048515165, 0.352413257282, 0.402109280550, 0.091526031678),
        (0.381479890328, 0.528245434800, 0.515974504698, 0.082165066727),
        (0.406608917449, 0.688567638892, 0.154576686664, 0.094716969786),
        (0.518787505536, 0.719575248996, 0.635622256484, 0.080040058493),
    ],
    dtype=float,
)


def set_spheres(spheres: np.ndarray) -> None:
    """Replace the frozen packing for one reproducible realization.

    The partition code intentionally reads the module-level table so all
    geometric operations (clearance, Voronoi faces and CAD construction)
    use exactly the same realization.  This small explicit hook is used by
    the multi-seed principle experiment; it avoids copying the solver for
    every random geometry.
    """
    values = np.asarray(spheres, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) < 4:
        raise ValueError("sphere realization must have shape (n, 4) with n >= 4")
    if not np.all(np.isfinite(values)) or np.any(values[:, 3] <= 0.0):
        raise ValueError("sphere realization contains non-finite centers or non-positive radii")
    global SPHERES
    SPHERES = np.ascontiguousarray(values)


def load_spheres_json(path: str | Path) -> None:
    """Load a realization written by the multi-seed campaign generator."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    set_spheres(payload["spheres"] if isinstance(payload, dict) else payload)

# Fictitious seeds far outside the cube bound every real cell of the Voronoi
# diagram so all ridge polygons of real-real pairs are finite.
DUMMY_SEEDS = np.asarray(
    [
        (-2.5, 0.5, 0.5),
        (3.5, 0.5, 0.5),
        (0.5, -2.5, 0.5),
        (0.5, 3.5, 0.5),
        (0.5, 0.5, -2.5),
        (0.5, 0.5, 3.5),
    ],
    dtype=float,
)

MIN_FACE_AREA = 1.0e-12
MIN_FACE_CLEARANCE = 1.0e-4
CLIP_BOX_HALF_WIDTH = 0.6  # generous trim: the fluid solid trims faces itself


@dataclass(frozen=True)
class Throat:
    pore_i: int
    pore_j: int
    saddle: tuple[float, float, float]
    normal: tuple[float, float, float]
    clearance: float
    gap_length: float


@dataclass
class PartitionData:
    mesh: dmesh.Mesh
    cell_centers: np.ndarray
    cell_clearance: np.ndarray
    cell_labels: np.ndarray
    maximal_balls: tuple[tuple[float, float, float, float], ...]
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
        return np.asarray([(*ball[:3], ball[3]) for ball in self.maximal_balls])


def sphere_clearance(points: np.ndarray) -> np.ndarray:
    """Distance to the closest solid sphere (cube boundary not included)."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    result = np.full(len(pts), np.inf, dtype=float)
    for x, y, z, r in SPHERES:
        result = np.minimum(
            result, np.linalg.norm(pts - np.asarray([x, y, z])[None, :], axis=1) - r
        )
    return result


def clearance(points: np.ndarray) -> np.ndarray:
    """Distance to the closest solid sphere or outer cube boundary."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    result = np.minimum(np.min(pts, axis=1), np.min(1.0 - pts, axis=1))
    return np.minimum(result, sphere_clearance(pts))


def _gap_segment(i: int, j: int) -> tuple[np.ndarray, np.ndarray, float] | None:
    a, b = SPHERES[i], SPHERES[j]
    delta = b[:3] - a[:3]
    distance = float(np.linalg.norm(delta))
    tangent = delta / distance
    gap = distance - a[3] - b[3]
    if gap <= 1.0e-7:
        return None
    start = a[:3] + tangent * a[3]
    end = b[:3] - tangent * b[3]
    return start, end, gap


def _segment_crosses_third_sphere(i: int, j: int, start: np.ndarray, end: np.ndarray) -> bool:
    samples = np.linspace(0.04, 0.96, 13)[:, None] * end + np.linspace(
        0.96, 0.04, 13
    )[:, None] * start
    for k, particle in enumerate(SPHERES):
        if k in (i, j):
            continue
        if np.any(
            np.linalg.norm(samples - particle[:3][None, :], axis=1)
            < particle[3] - 1.0e-9
        ):
            return True
    return False


def analytic_throat_candidates() -> list[tuple[int, int, np.ndarray, float, float]]:
    """Delaunay-neighbour pairs with valid center-line gap segments.

    Returns ``(i, j, saddle, gap, gap_length)`` for every valid throat,
    mirroring the 2-D recipe (reject pairs whose gap segment crosses a
    third sphere).  No midpoint-of-cube rejection is used: a 3-D Voronoi
    face may have its saddle outside the cube while still cutting through
    the cube interior, and the mesh-based pair graph decides the actual
    interfaces.
    """
    triangulation = Delaunay(SPHERES[:, :3])
    edges: set[tuple[int, int]] = set()
    for simplex in triangulation.simplices:
        for a in range(4):
            for b in range(a + 1, 4):
                edges.add(tuple(sorted((int(simplex[a]), int(simplex[b])))))
    result: list[tuple[int, int, np.ndarray, float, float]] = []
    for i, j in sorted(edges):
        segment = _gap_segment(i, j)
        if segment is None:
            continue
        start, end, gap = segment
        # Equal-clearance point: t - r_a = (d - t) - r_b  =>  t = (d + r_a - r_b)/2,
        # which lies at the midpoint of the wall-to-wall gap segment.
        saddle = 0.5 * (start + end)
        if _segment_crosses_third_sphere(i, j, start, end):
            continue
        result.append((i, j, saddle, gap, float(np.linalg.norm(end - start))))
    return result


# ---------------------------------------------------------------------------
# Voronoi throat faces
# ---------------------------------------------------------------------------

def _clip_polygon_to_box(polygon: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman clip of a convex polygon to a large bounding box.

    The box is only a safety trim against numerical blow-up of far ridge
    vertices; the fluid solid trims each face to the cube itself during the
    OCC fragment, which keeps neighbouring Voronoi faces conforming.
    """
    lo = -CLIP_BOX_HALF_WIDTH
    hi = 1.0 + CLIP_BOX_HALF_WIDTH
    current = polygon
    for axis in range(3):
        for bound in (lo, hi):
            def inside(point, axis=axis, bound=bound) -> bool:
                return point[axis] >= bound - 1.0e-12 if bound == lo else point[axis] <= bound + 1.0e-12

            def intersect(p, q, axis=axis, bound=bound):
                value_p, value_q = p[axis], q[axis]
                t = (bound - value_p) / (value_q - value_p)
                return p + t * (q - p)

            output: list[np.ndarray] = []
            for k in range(len(current)):
                p, q = current[k], current[(k + 1) % len(current)]
                if inside(p):
                    output.append(p)
                    if not inside(q):
                        output.append(intersect(p, q))
                elif inside(q):
                    output.append(intersect(p, q))
            current = np.asarray(output, dtype=float) if output else np.empty((0, 3))
            if len(current) == 0:
                return current
    return current


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
    total = 0.0
    for k in range(len(polygon)):
        a, b = polygon[k], polygon[(k + 1) % len(polygon)]
        total += float(np.dot(normal, np.cross(a, b)))
    return 0.5 * abs(total) / max(float(np.linalg.norm(normal)), 1.0e-30)


def voronoi_throat_faces() -> tuple[list[tuple[int, int]], list[np.ndarray], list[Throat]]:
    """Build the clipped planar faces of all valid Delaunay pairs.

    The face of pair (i, j) is the Voronoi ridge polygon of the two sphere
    centres clipped to the unit cube.  Pairs rejected by the gap-segment
    tests of :func:`analytic_throat_candidates` are skipped; faces whose
    clipped part has no clearance to the solid obstacles are skipped too.
    """
    candidates = analytic_throat_candidates()
    all_points = np.vstack([SPHERES[:, :3], DUMMY_SEEDS])
    vor = Voronoi(all_points)
    faces: list[np.ndarray] = []
    used_pairs: list[tuple[int, int]] = []
    throat_records: list[Throat] = []
    for ridge_index, (sites, vertices) in enumerate(
        zip(vor.ridge_points, vor.ridge_vertices, strict=True)
    ):
        a, b = int(sites[0]), int(sites[1])
        if a >= 27 and b >= 27:
            continue  # dummy-dummy face: would cross the cube interior
        if -1 in vertices:
            continue  # unbounded: cannot happen with the dummy seeds
        polygon = vor.vertices[np.asarray(vertices, dtype=int)]
        # Keep every non-degenerate face, real-real and real-dummy alike.
        # A Voronoi face's boundary is shared with neighbouring faces, so
        # dropping one face leaves its neighbours' boundary edges dangling
        # in free space and the OCC fragment does not split the volume.
        # Real-dummy faces lie outside the cube (the fluid solid trims them
        # during the fragment) but their edges close the boundary shells of
        # the wall-adjacent pores.  Only a genuine intersection with a solid
        # sphere disqualifies a face.
        if len(polygon) < 3 or _polygon_area(polygon) < MIN_FACE_AREA:
            continue
        if float(np.min(sphere_clearance(polygon))) < -1.0e-6:
            continue  # pathological: face would cut through a solid sphere
        faces.append(polygon)
        if a >= 27 or b >= 27:
            continue  # real-dummy: shell closure only, not a throat
        pair = (min(a, b), max(a, b))
        used_pairs.append(pair)

    # Reject candidate throats whose face is not a real-real Voronoi ridge,
    # then attach the throat records in pair order.
    face_for_pair: dict[tuple[int, int], int] = {
        pair: i for i, pair in enumerate(used_pairs)
    }
    for i, j, saddle, gap, gap_length in candidates:
        face_index = face_for_pair.get((i, j))
        if face_index is None:
            continue
        polygon = faces[face_index]
        if sphere_clearance(saddle[None, :]).item() < MIN_FACE_CLEARANCE:
            continue
        normal = (SPHERES[j, :3] - SPHERES[i, :3]) / float(
            np.linalg.norm(SPHERES[j, :3] - SPHERES[i, :3])
        )
        throat_records.append(
            Throat(
                pore_i=i,
                pore_j=j,
                saddle=tuple(float(x) for x in saddle),
                normal=tuple(float(x) for x in normal),
                clearance=float(gap),
                gap_length=float(gap_length),
            )
        )
    return used_pairs, faces, throat_records


# ---------------------------------------------------------------------------
# OCC mesh generation
# ---------------------------------------------------------------------------

def _add_planar_face(polygon: np.ndarray) -> int:
    points = [gmsh.model.occ.addPoint(float(p[0]), float(p[1]), float(p[2])) for p in polygon]
    lines = [
        gmsh.model.occ.addLine(points[k], points[(k + 1) % len(points)])
        for k in range(len(points))
    ]
    loop = gmsh.model.occ.addCurveLoop(lines)
    return gmsh.model.occ.addPlaneSurface([loop])


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


def generate_partition_mesh(
    bulk_size: float = 0.10,
    sphere_size: float = 0.045,
    boundary_size: float = 0.060,
    interface_size: float = 0.055,
    sphere_band: float = 0.12,
    boundary_band: float = 0.10,
    interface_band: float = 0.09,
    mesh_file=None,
) -> tuple[dmesh.Mesh, dmesh.MeshTags, tuple[Throat, ...], dict[str, int]]:
    """Build the sphere-subtracted, Voronoi-partitioned conforming mesh."""
    used_pairs, faces, throats = voronoi_throat_faces()
    n_pores = len(SPHERES)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("affine_ddpnm_3d_random_porous")
        box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        grains = [
            gmsh.model.occ.addSphere(float(x), float(y), float(z), float(r))
            for x, y, z, r in SPHERES
        ]
        fluid, _ = gmsh.model.occ.cut(
            [(3, box)],
            [(3, tag) for tag in grains],
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()
        cut_surfaces = [_add_planar_face(polygon) for polygon in faces]
        gmsh.model.occ.fragment(
            fluid,
            [(2, tag) for tag in cut_surfaces],
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()

        volumes = [tag for _, tag in gmsh.model.getEntities(3)]
        volume_to_label: dict[int, int] = {}
        labels_by_volume: dict[int, list[int]] = {i: [] for i in range(n_pores)}
        for volume in volumes:
            center = np.asarray(gmsh.model.occ.getCenterOfMass(3, volume), dtype=float)
            label = int(
                np.argmin(np.linalg.norm(SPHERES[:, :3] - center[None, :], axis=1))
            )
            volume_to_label[volume] = label
            labels_by_volume[label].append(volume)
        missing = [label for label, tags in labels_by_volume.items() if not tags]
        if missing:
            raise RuntimeError(f"Gmsh partition is missing pore regions {missing}.")
        if len(volumes) != n_pores:
            raise RuntimeError(
                f"Expected {n_pores} pore volumes from the fragment, "
                f"found {len(volumes)}."
            )
        for label, tags in labels_by_volume.items():
            physical = gmsh.model.addPhysicalGroup(3, tags, tag=1000 + label)
            gmsh.model.setPhysicalName(3, physical, f"pore_{label:03d}")

        pair_to_interface = {pair: i for i, pair in enumerate(used_pairs)}
        sphere_surfaces: list[int] = []
        cube_surfaces: dict[str, list[int]] = {
            "inlet": [],
            "outlet": [],
            "outer_wall": [],
        }
        interface_surfaces: dict[int, list[int]] = {
            i: [] for i in range(len(used_pairs))
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

        cad_counts = {
            "fluid_volumes": len(volumes),
            "pore_regions": n_pores,
            "interface_candidates": len(used_pairs),
            "interface_surface_patches": sum(
                len(tags) for tags in interface_surfaces.values()
            ),
            "sphere_surface_patches": len(sphere_surfaces),
            "inlet_surface_patches": len(cube_surfaces["inlet"]),
            "outlet_surface_patches": len(cube_surfaces["outlet"]),
            "outer_wall_surface_patches": len(cube_surfaces["outer_wall"]),
            "orphan_tool_surface_patches": orphan_surfaces,
        }
        # A throat face may legitimately be split into several CAD patches
        # (trimmed by the cube boundary or neighbouring faces); every patch
        # is tagged with the same interface id and the mesh facets below
        # carry the pair via facet_interface_ids, so the solver sees one
        # interface per pair regardless of the patch count.

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
            gmsh.model.setPhysicalName(2, physical, f"interface_{interface_id:03d}")

        fields = [
            _distance_threshold(sphere_surfaces, sphere_size, bulk_size, sphere_band),
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
        return data.mesh, data.cell_tags, tuple(throats), cad_counts
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
    if np.any(labels < 0) or np.any(labels >= len(SPHERES)):
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    centers = np.zeros((len(pairs), 3), dtype=float)
    normals = np.zeros((len(pairs), 3), dtype=float)
    areas = np.zeros(len(pairs), dtype=float)
    for interface_id, (a, b) in enumerate(pairs):
        facets = np.flatnonzero(facet_ids == interface_id)
        if not len(facets):
            raise RuntimeError(f"Interface {interface_id} has no mesh facets.")
        for facet in facets:
            triangle = msh.geometry.x[f2v.links(int(facet)), :3]
            area = 0.5 * float(
                np.linalg.norm(
                    np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                )
            )
            centers[interface_id] += area * triangle.mean(axis=0)
            areas[interface_id] += area
        centers[interface_id] /= areas[interface_id]
        direction = SPHERES[b, :3] - SPHERES[a, :3]
        normals[interface_id] = direction / np.linalg.norm(direction)
    return centers, normals, areas


def build_partition(
    mesh_size: float = 0.10,
    sphere_size: float = 0.045,
    boundary_size: float = 0.060,
    interface_size: float = 0.055,
    sphere_band: float = 0.12,
    boundary_band: float = 0.10,
    interface_band: float = 0.09,
    mesh_file=None,
) -> PartitionData:
    msh, cell_tags, throats, cad_counts = generate_partition_mesh(
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
    # The mesh-based interface graph is the ground truth; the analytic
    # throat records are attached per pair where they exist.
    throat_by_pair = {(t.pore_i, t.pore_j): t for t in throats}
    ordered_throats = tuple(
        throat_by_pair[p] for p in pairs if p in throat_by_pair
    )
    interface_centers, interface_normals, interface_areas = interface_geometry(
        msh, facet_ids, pairs
    )
    maximal_balls: list[tuple[float, float, float, float]] = []
    for label in sorted(int(x) for x in np.unique(labels)):
        cells = np.flatnonzero(labels == label)
        cell = int(cells[np.argmax(cell_clearance[cells])])
        maximal_balls.append(
            (centers[cell, 0], centers[cell, 1], centers[cell, 2], cell_clearance[cell])
        )
    return PartitionData(
        mesh=msh,
        cell_centers=centers,
        cell_clearance=cell_clearance,
        cell_labels=labels,
        maximal_balls=tuple(maximal_balls),
        throats=ordered_throats,
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

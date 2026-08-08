"""High-porosity random sphere packing — three partition methods.

1. Voronoi  — embed Voronoi faces via OCC fragment (like random_porous.py)
2. Watershed — clearance-based basin labelling (like watershed_partition.py)
3. Grid     — 4×4×4=64 subdomains via OCC cutting planes ("平凡傻瓜分法")

All three produce a PartitionData-compatible object consumable by ddpnm_core.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gmsh
import numpy as np
from dolfinx import mesh as dmesh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from scipy.spatial import Delaunay, Voronoi

# ---------------------------------------------------------------------------
# Frozen sphere packing (seed 20260805, 60 spheres, porosity ~93.9%)
# ---------------------------------------------------------------------------
SPHERES = np.asarray([
    (-0.021487614682, 0.468331371797, 0.219343631223, 0.066969753600),
    (-0.020872531581, 0.125932414768, 0.585447085966, 0.059814987033),
    (-0.048553517779, 0.299654338557, 0.779867868594, 0.058434742586),
    (1.038684665032, 0.782078678124, 0.553583500805, 0.058034692997),
    (1.044576205664, 0.443370871909, 0.198644941463, 0.080719996873),
    (1.040176769229, 0.439454814258, 0.503419272720, 0.063261866108),
    (0.549184197412, -0.028721585953, 0.547209625031, 0.058485080043),
    (0.268462973660, -0.022174601321, 0.277380238498, 0.080816807132),
    (0.731995168274, -0.044158889357, 0.154178948570, 0.058849083896),
    (0.752114775424, 1.043948621413, 0.646231263459, 0.080387110595),
    (0.660398567039, 1.023017648056, 0.129297686668, 0.057985980017),
    (0.143836473552, 1.038943248352, 0.491810962977, 0.094267859617),
    (0.423409709274, 0.270678330011, -0.028305331895, 0.058239529276),
    (0.541474045524, 0.805600921198, -0.049715969490, 0.092650269888),
    (0.248286836240, 0.517304476691, -0.025686284105, 0.092304420165),
    (0.714470735468, 0.850147800426, 1.026321137116, 0.083153807128),
    (0.332315119086, 0.707450752164, 1.039606305627, 0.089295013626),
    (0.753064702783, 0.243640343971, 1.036194033840, 0.085519076009),
    (0.342475454316, 0.485973093592, 0.810530664730, 0.082603904054),
    (0.645910188387, 0.581901875648, 0.372972849942, 0.062723702656),
    (0.760244239549, 0.820808173352, 0.253098010707, 0.083152544878),
    (0.134781709695, 0.516131258203, 0.372172853722, 0.081470498640),
    (0.457007913662, 0.744179637189, 0.185701077650, 0.092944224011),
    (0.398989188943, 0.769589850408, 0.566893544527, 0.067789954011),
    (0.625221725716, 0.292140563728, 0.881580686434, 0.059272408780),
    (0.272317384911, 0.269779139186, 0.839716530396, 0.058064671074),
    (0.602038259440, 0.249061306986, 0.216482008205, 0.057835177795),
    (0.491836766634, 0.102379067625, 0.580843606988, 0.065120269431),
    (0.292072049851, 0.880435902233, 0.834287295997, 0.093786415308),
    (0.117036113434, 0.642352932613, 0.681651658722, 0.058240176328),
    (0.791399443518, 0.103409213969, 0.560883156846, 0.058905756577),
    (0.780110859628, 0.406174910050, 0.243549845383, 0.087678860657),
    (0.285563723209, 0.218994550466, 0.684322068377, 0.063181736989),
    (0.722402850903, 0.262464360894, 0.561065897825, 0.089805723070),
    (0.341704112139, 0.243208229400, 0.408032470570, 0.085906097128),
    (0.389443981029, 0.513621929080, 0.285981068193, 0.096540994393),
    (0.175097040556, 0.213243096686, 0.596006033231, 0.062436338021),
    (0.553984567262, 0.747211345868, 0.465353151913, 0.059590624036),
    (0.428960811312, 0.352514195358, 0.292005787171, 0.061144296204),
    (0.869347608298, 0.101349453795, 0.176070809864, 0.062761943740),
    (0.654197134948, 0.889830616158, 0.716405359097, 0.056470435416),
    (0.878881384006, 0.652449214325, 0.659090771038, 0.086645033287),
    (0.252051598842, 0.610769948872, 0.181829068464, 0.059993925742),
    (0.785828306346, 0.218193258346, 0.249071718904, 0.062384790734),
    (0.616077940392, 0.579883856462, 0.112996118645, 0.064154926247),
    (0.761724031280, 0.196856047518, 0.714896803509, 0.057902431268),
    (0.298204973347, 0.893873074260, 0.486828772588, 0.080345150763),
    (0.158232023584, 0.309624067312, 0.781460865783, 0.066178466900),
    (0.146725275152, 0.309789430374, 0.305613238821, 0.061410026326),
    (0.764483470837, 0.626965126269, 0.819023766118, 0.067052695468),
    (0.465553225424, 0.800932662200, 0.837995871380, 0.064512716466),
    (0.749639493776, 0.288301491133, 0.840246442372, 0.059232356581),
    (0.653402191362, 0.723369458904, 0.705684715072, 0.065801882001),
    (0.413380995499, 0.626039336640, 0.869281778737, 0.055380497666),
    (0.146190371285, 0.480843477874, 0.110567114624, 0.061486350626),
    (0.686829645017, 0.444746533119, 0.660948969195, 0.067288090707),
    (0.482743507943, 0.444733544866, 0.667629913044, 0.058210325846),
    (0.692002828153, 0.816393325203, 0.402518215374, 0.062806675926),
    (0.884891017955, 0.885656077678, 0.646533372050, 0.065291802718),
    (0.470492334282, 0.610799468261, 0.517103613492, 0.063560423757),
    (0.200654569472, 0.651342622823, 0.522686131530, 0.065566484457),
    (0.222811773635, 0.098194566828, 0.128356175470, 0.057971904180),
    (0.113870520907, 0.784447629181, 0.701492826150, 0.065034179983),
    (0.145888371112, 0.392173845290, 0.581323993370, 0.061157498573),
    (0.733259317053, 0.830464260416, 0.837330134889, 0.064913283004),
    (0.369474225545, 0.393874690466, 0.591244908102, 0.055078893137),
    (0.572496898529, 0.571372128681, 0.682787583113, 0.059363395428),
    (0.805183895809, 0.670149527822, 0.137793297631, 0.066958489695),
    (0.634354426343, 0.398581141774, 0.338269459226, 0.065141813871),
    (0.486942065219, 0.524736048180, 0.873025561970, 0.056262944908),
    (0.102413607612, 0.597833660459, 0.866388754178, 0.055772314163),
    (0.523210368390, 0.196114529551, 0.780505142990, 0.055240409319),
    (0.129148876697, 0.764761305128, 0.117982047755, 0.083060409001),
    (0.804705371959, 0.248713336937, 0.400495881496, 0.067425592301),
    (0.623426315499, 0.205175525615, 0.431320929063, 0.060174554105),
    (0.182138651450, 0.514390083399, 0.668749422459, 0.062973542693),
    (0.379211157124, 0.333598701034, 0.717359854911, 0.058740417399),
    (0.890786395625, 0.340221385895, 0.622836101673, 0.092892022250),
    (0.354258886781, 0.090243111228, 0.731930010774, 0.066962504365),
    (0.161285193423, 0.192640035273, 0.898703694748, 0.061816173734),
    (0.809753614235, 0.867222886806, 0.500256362004, 0.084155316344),
    (0.667052640796, 0.666304110225, 0.535175747881, 0.066996103821),
    (0.665326289046, 0.133635261200, 0.215420655412, 0.059179645090),
    (0.631395910549, 0.605074707734, 0.842737691903, 0.061066803023),
    (0.899674472694, 0.171532115073, 0.881720062798, 0.066740908634),
    (0.204968435486, 0.128381270698, 0.315691951107, 0.066613497257),
    (0.096358799473, 0.316175489700, 0.130604842296, 0.056300494168),
    (0.544861683877, 0.442438900517, 0.535362806900, 0.056197655260),
    (0.907118989987, 0.454040989162, 0.372257535976, 0.067917937809),
    (0.462603730943, 0.547314381168, 0.086567888684, 0.058996768197),
    (0.431877078501, 0.218711725171, 0.086816536756, 0.059084393464),
    (0.413369566706, 0.100666621060, 0.368080865176, 0.063725960000),
    (0.402530286703, 0.100301626790, 0.888945243214, 0.080261115887),
    (0.152271254953, 0.787002617087, 0.481222106478, 0.064094960281),
    (0.447797405782, 0.370014989323, 0.906033398644, 0.083284359788),
    (0.669294251044, 0.113469937947, 0.915065737707, 0.066582119999),
    (0.216683873466, 0.858684852874, 0.267478505384, 0.082730595736),
    (0.210167727851, 0.695756978103, 0.326563665916, 0.059199916259),
    (0.908376149841, 0.870932572520, 0.313911705416, 0.063736264973),
    (0.568070782323, 0.892783330214, 0.507529313947, 0.064650408315),
], dtype=float)

N_SPHERES = len(SPHERES)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def sphere_clearance(points: np.ndarray) -> np.ndarray:
    """Distance to the closest solid sphere surface (cube boundary excluded)."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    result = np.full(len(pts), np.inf, dtype=float)
    for x, y, z, r in SPHERES:
        result = np.minimum(
            result, np.linalg.norm(pts - np.asarray([x, y, z])[None, :], axis=1) - r
        )
    return result


# ===================================================================
# Partition 1: Voronoi (adapted from affine_ddpnm_3d_random_porous)
# ===================================================================

DUMMY_SEEDS_V = np.asarray([
    (-2.5, 0.5, 0.5), (3.5, 0.5, 0.5),
    (0.5, -2.5, 0.5), (0.5, 3.5, 0.5),
    (0.5, 0.5, -2.5), (0.5, 0.5, 3.5),
], dtype=float)

MIN_FACE_AREA = 1e-12
MIN_FACE_CLEARANCE = 1e-4
CLIP_BOX_HALF = 0.6


@dataclass(frozen=True)
class Throat:
    pore_i: int; pore_j: int
    saddle: tuple[float, float, float]
    normal: tuple[float, float, float]
    clearance: float; gap_length: float


@dataclass
class PartitionData:
    mesh: dmesh.Mesh
    cell_centers: np.ndarray
    cell_clearance: np.ndarray
    cell_labels: np.ndarray
    maximal_balls: tuple
    throats: tuple
    interface_pairs: tuple
    facet_interface_ids: np.ndarray
    interface_centers: np.ndarray
    interface_normals: np.ndarray
    interface_areas: np.ndarray
    mesh_parameters: dict
    cad_counts: dict

    @property
    def pore_seeds(self):
        return np.asarray([(*b[:3], b[3]) for b in self.maximal_balls])


def _gap_segment(i, j):
    a, b = SPHERES[i], SPHERES[j]
    delta = b[:3] - a[:3]
    dist = float(np.linalg.norm(delta))
    tangent = delta / dist
    gap = dist - a[3] - b[3]
    if gap <= 1e-7:
        return None
    return a[:3] + tangent * a[3], b[:3] - tangent * b[3], gap


def _segment_crosses_third(i, j, start, end):
    samples = np.linspace(0.04, 0.96, 13)[:, None] * end + np.linspace(0.96, 0.04, 13)[:, None] * start
    for k, p in enumerate(SPHERES):
        if k in (i, j):
            continue
        if np.any(np.linalg.norm(samples - p[:3][None, :], axis=1) < p[3] - 1e-9):
            return True
    return False


def _analytic_throat_candidates():
    tri = Delaunay(SPHERES[:, :3])
    edges = set()
    for s in tri.simplices:
        for a in range(4):
            for b in range(a + 1, 4):
                edges.add(tuple(sorted((int(s[a]), int(s[b])))))
    result = []
    for i, j in sorted(edges):
        seg = _gap_segment(i, j)
        if seg is None:
            continue
        start, end, gap = seg
        saddle = 0.5 * (start + end)
        if _segment_crosses_third(i, j, start, end):
            continue
        result.append((i, j, saddle, gap, float(np.linalg.norm(end - start))))
    return result


def _clip_polygon_to_box(polygon):
    lo, hi = -CLIP_BOX_HALF, 1.0 + CLIP_BOX_HALF
    current = polygon
    for axis in range(3):
        for bound in (lo, hi):
            def inside(p, ax=axis, b=bound):
                return p[ax] >= b - 1e-12 if b == lo else p[ax] <= b + 1e-12
            def intersect(p, q, ax=axis, b=bound):
                t = (b - p[ax]) / (q[ax] - p[ax])
                return p + t * (q - p)
            out = []
            for k in range(len(current)):
                p_, q_ = current[k], current[(k + 1) % len(current)]
                if inside(p_):
                    out.append(p_)
                    if not inside(q_):
                        out.append(intersect(p_, q_))
                elif inside(q_):
                    out.append(intersect(p_, q_))
            current = np.asarray(out, dtype=float) if out else np.empty((0, 3))
            if len(current) == 0:
                return current
    return current


def _polygon_area(polygon):
    if len(polygon) < 3:
        return 0.0
    n = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
    total = sum(float(np.dot(n, np.cross(polygon[k], polygon[(k + 1) % len(polygon)])))
                for k in range(len(polygon)))
    return 0.5 * abs(total) / max(float(np.linalg.norm(n)), 1e-30)


def _voronoi_throat_faces():
    candidates = _analytic_throat_candidates()
    all_pts = np.vstack([SPHERES[:, :3], DUMMY_SEEDS_V])
    vor = Voronoi(all_pts)
    faces, used_pairs, throat_records = [], [], []
    for sites, verts in zip(vor.ridge_points, vor.ridge_vertices):
        a, b = int(sites[0]), int(sites[1])
        if a >= N_SPHERES and b >= N_SPHERES:
            continue
        if -1 in verts:
            continue
        polygon = vor.vertices[np.asarray(verts, dtype=int)]
        if len(polygon) < 3 or _polygon_area(polygon) < MIN_FACE_AREA:
            continue
        if float(np.min(sphere_clearance(polygon))) < -1e-6:
            continue
        faces.append(polygon)
        if a >= N_SPHERES or b >= N_SPHERES:
            continue
        used_pairs.append((min(a, b), max(a, b)))

    face_for_pair = {p: i for i, p in enumerate(used_pairs)}
    for i, j, saddle, gap, gap_len in candidates:
        fi = face_for_pair.get((i, j))
        if fi is None:
            continue
        poly = faces[fi]
        if sphere_clearance(saddle[None, :]).item() < MIN_FACE_CLEARANCE:
            continue
        normal = (SPHERES[j, :3] - SPHERES[i, :3])
        normal /= float(np.linalg.norm(normal))
        throat_records.append(Throat(i, j, tuple(float(x) for x in saddle),
                                     tuple(float(x) for x in normal), float(gap), float(gap_len)))
    return used_pairs, faces, throat_records


def _add_planar_face(polygon):
    pts = [gmsh.model.occ.addPoint(float(p[0]), float(p[1]), float(p[2])) for p in polygon]
    lines = [gmsh.model.occ.addLine(pts[k], pts[(k + 1) % len(pts)]) for k in range(len(pts))]
    loop = gmsh.model.occ.addCurveLoop(lines)
    return gmsh.model.occ.addPlaneSurface([loop])


def _distance_field(surfaces, sz_min, sz_max, band):
    if not surfaces:
        return None
    d = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(d, "SurfacesList", surfaces)
    gmsh.model.mesh.field.setNumber(d, "Sampling", 120)
    t = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(t, "InField", d)
    gmsh.model.mesh.field.setNumber(t, "SizeMin", sz_min)
    gmsh.model.mesh.field.setNumber(t, "SizeMax", sz_max)
    gmsh.model.mesh.field.setNumber(t, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(t, "DistMax", band)
    return t


def _outer_side(surface, tol=5e-7):
    c = np.asarray(gmsh.model.occ.getCenterOfMass(2, surface), dtype=float)
    if abs(c[0]) < tol: return "inlet"
    if abs(c[0] - 1.0) < tol: return "outlet"
    if (abs(c[1]) < tol or abs(c[1] - 1.0) < tol or
        abs(c[2]) < tol or abs(c[2] - 1.0) < tol):
        return "outer_wall"
    return None


# ===================================================================
# Weighted Voronoi (Laguerre / Power Diagram) — face polygon generation
# ===================================================================

DUMMY_SEEDS_W = np.asarray([
    [-3.5, 0.5, 0.5, 0.0], [4.5, 0.5, 0.5, 0.0],
    [0.5, -3.5, 0.5, 0.0], [0.5, 4.5, 0.5, 0.0],
    [0.5, 0.5, -3.5, 0.0], [0.5, 0.5, 4.5, 0.0],
], dtype=float)


def _plane_local_frame(normal, origin):
    """Return 2-D local frame (t1, t2) on a plane for 3D↔2D mapping."""
    n = normal / float(np.linalg.norm(normal))
    if abs(n[2]) < 0.9:
        t1 = np.cross(n, [0.0, 0.0, 1.0])
    else:
        t1 = np.cross(n, [1.0, 0.0, 0.0])
    t1 /= float(np.linalg.norm(t1))
    t2 = np.cross(n, t1)
    return n, t1, t2


def _to_2d(pts_3d, origin, t1, t2):
    """Project 3-D points (N,3) onto local 2-D frame."""
    d = pts_3d - origin[None, :]
    return np.column_stack([np.dot(d, t1), np.dot(d, t2)])


def _to_3d(pts_2d, origin, t1, t2):
    """Lift 2-D points (N,2) back to 3-D on the plane."""
    return origin[None, :] + pts_2d[:, 0:1] * t1[None, :] + pts_2d[:, 1:2] * t2[None, :]


def _clip_polygon_2d_by_half_plane(poly_2d, p0_2d, dir_2d, inside_normal_2d):
    """Sutherland–Hodgman clip a 2-D polygon against a half-plane.

    The half-plane boundary is the line p0_2d + t*dir_2d.
    Points on the side toward *inside_normal_2d* (a unit vector perpendicular
    to the line) are considered inside and kept.
    """
    if len(poly_2d) < 3:
        return poly_2d

    d = inside_normal_2d / float(np.linalg.norm(inside_normal_2d))

    def _inside(p):
        return float(np.dot(p - p0_2d, d)) >= -1e-12

    def _intersect(p, q):
        # Line segment p→q crossing the boundary line
        dp = float(np.dot(p - p0_2d, d))
        dq = float(np.dot(q - p0_2d, d))
        t = dp / max(abs(dp - dq), 1e-30)
        return p + t * (q - p)

    out = []
    for k in range(len(poly_2d)):
        p, q = poly_2d[k], poly_2d[(k + 1) % len(poly_2d)]
        if _inside(p):
            out.append(p)
            if not _inside(q):
                out.append(_intersect(p, q))
        elif _inside(q):
            out.append(_intersect(p, q))

    return np.asarray(out, dtype=float) if out else np.empty((0, 2))


def _initial_square_3d(face_center, normal, half_size=2.0):
    """Large square on a plane; serves as seed polygon for half-space clipping."""
    n, t1, t2 = _plane_local_frame(normal, face_center)
    half = half_size
    corners_2d = np.array([
        [-half, -half], [half, -half], [half, half], [-half, half]
    ])
    return _to_3d(corners_2d, face_center, t1, t2)


def _weighted_voronoi_throat_faces():
    """Compute weighted-Voronoi face polygons via 4-D Delaunay circumcenters.

    Uses the 4-D circumcenters of the regular triangulation to produce
    **globally consistent** polygon vertices (shared exactly by neighbouring
    faces), then clips each face polygon against the domain box.

    Returns ``(used_pairs, faces, throat_records)``.
    """
    from collections import defaultdict

    all_spheres = np.vstack([SPHERES, DUMMY_SEEDS_W])
    n_real = len(SPHERES)
    n_all = len(all_spheres)
    centers_all = all_spheres[:, :3]
    radii_all = all_spheres[:, 3]

    # ── 1. 4-D Delaunay = regular triangulation ──
    w_lift = np.sum(centers_all**2, axis=1) - radii_all**2
    lifted = np.column_stack([centers_all, w_lift])
    tri = Delaunay(lifted)
    n_simplex = len(tri.simplices)
    print(f"  weighted-Voronoi Delaunay: {n_simplex} 4-simplices")

    # ── 2. 4-D circumcenters → 3-D power-diagram vertices ──
    # For each 4-simplex {v0..v4}, solve  2⟨c, vi−v0⟩ = |vi|²−|v0|²  (i=1..4)
    power_vertices_3d = np.zeros((n_simplex, 3))
    for s_idx in range(n_simplex):
        verts = lifted[tri.simplices[s_idx]]  # (5, 4)
        v0 = verts[0]
        A = 2.0 * (verts[1:] - v0).T  # (4, 4)
        b = np.sum(verts[1:]**2, axis=1) - np.sum(v0**2)
        try:
            circum_4d = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            power_vertices_3d[s_idx] = np.nan
            continue
        power_vertices_3d[s_idx] = circum_4d[:3]

    # ── 3. Build face polygons from simplex→pair mappings ──
    # Map (i,j) → list of 3-D vertices (from simplices containing both i,j)
    pair_vertices: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for s_idx in range(n_simplex):
        if np.any(~np.isfinite(power_vertices_3d[s_idx])):
            continue
        v3 = power_vertices_3d[s_idx]
        sv = [int(v) for v in tri.simplices[s_idx]]
        for ai in range(5):
            for bi in range(ai + 1, 5):
                a, b = sv[ai], sv[bi]
                if a >= n_all or b >= n_all:
                    continue
                pair = (min(a, b), max(a, b))
                pair_vertices[pair].append(v3.copy())

    # ── 4. Ownership filter (same as prototype) ──
    w_sq_all = np.sum(centers_all**2, axis=1) - radii_all**2

    def _closest_two_power(pt):
        scores = 2.0 * pt @ centers_all.T - w_sq_all
        order = np.argsort(-scores)
        return int(order[0]), int(order[1])

    pairs_filtered = []
    for (a, b), verts in pair_vertices.items():
        if len(verts) < 3:
            continue
        ca, cb = centers_all[a], centers_all[b]
        ra, rb = radii_all[a], radii_all[b]
        delta = cb - ca
        dist = float(np.linalg.norm(delta))
        if dist < 1e-10:
            continue
        unit = delta / dist
        shift = (ra**2 - rb**2) / (2.0 * dist)
        fc = 0.5 * (ca + cb) + shift * unit
        k1, k2 = _closest_two_power(fc)
        if {k1, k2} == {a, b}:
            pairs_filtered.append((a, b))
    print(f"  pairs with ≥3 vertices: {len(pair_vertices)}")
    print(f"  after ownership filter: {len(pairs_filtered)}")

    # ── 5. Order vertices and clip to box per face ──
    faces = []
    used_pairs = []
    for i, j in pairs_filtered:
        if i >= n_real and j >= n_real:
            continue  # dummy–dummy

        verts = pair_vertices[(i, j)]
        if len(verts) < 3:
            continue

        # Order vertices by angle on the bisector plane
        ci, cj = centers_all[i], centers_all[j]
        ri, rj = radii_all[i], radii_all[j]
        delta = cj - ci
        dist = max(float(np.linalg.norm(delta)), 1e-30)
        unit = delta / dist
        shift = (ri**2 - rj**2) / (2.0 * dist)
        fc = 0.5 * (ci + cj) + shift * unit

        _, t1, t2 = _plane_local_frame(unit, fc)
        verts_2d = _to_2d(np.array(verts), fc, t1, t2)
        centroid_2d = verts_2d.mean(axis=0)
        angles = np.arctan2(verts_2d[:, 1] - centroid_2d[1],
                            verts_2d[:, 0] - centroid_2d[0])
        order = np.argsort(angles)
        ordered_verts = np.array(verts)[order]

        # Clip against domain box
        polygon_3d = _clip_polygon_to_box(ordered_verts)
        if len(polygon_3d) < 3:
            continue
        # Project vertices onto the bisector plane to fix numerical drift
        polygon_3d = polygon_3d - np.outer(np.dot(polygon_3d - fc, unit), unit)
        area = _polygon_area(polygon_3d)
        if area < MIN_FACE_AREA:
            continue

        faces.append(polygon_3d)
        if i < n_real and j < n_real:
            used_pairs.append((min(i, j), max(i, j)))

    print(f"  polygons after box clip: {len(faces)} ({len(used_pairs)} real–real)")

    # ── 6. Throat records ──
    face_for_pair = {p: idx for idx, p in enumerate(used_pairs)}
    candidates = _analytic_throat_candidates()
    throat_records = []
    for i_th, j_th, saddle, gap, gap_len in candidates:
        fi = face_for_pair.get((i_th, j_th))
        if fi is None:
            continue
        if sphere_clearance(np.asarray(saddle).reshape(1, -1)).item() < MIN_FACE_CLEARANCE:
            continue
        normal = SPHERES[j_th, :3] - SPHERES[i_th, :3]
        normal /= float(np.linalg.norm(normal))
        throat_records.append(Throat(i_th, j_th, tuple(float(x) for x in saddle),
                                     tuple(float(x) for x in normal),
                                     float(gap), float(gap_len)))
    print(f"  throat records: {len(throat_records)}")

    return used_pairs, faces, throat_records


def build_partition_voronoi(
    mesh_size=0.08, sphere_size=0.03, boundary_size=0.05,
    interface_size=0.04, sphere_band=0.10, boundary_band=0.08,
    interface_band=0.07, mesh_file=None,
) -> PartitionData:
    """Voronoi-face partition. Embeds Voronoi polygons in OCC, fragments once."""
    used_pairs, faces, throats = _voronoi_throat_faces()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("real_porous_voronoi")
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        grains = [gmsh.model.occ.addSphere(float(x), float(y), float(z), float(r))
                  for x, y, z, r in SPHERES]
        fluid, _ = gmsh.model.occ.cut([(3, box)], [(3, t) for t in grains],
                                      removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        cut_surfs = [_add_planar_face(p) for p in faces]
        gmsh.model.occ.fragment(fluid, [(2, t) for t in cut_surfs],
                                removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        volumes = [t for _, t in gmsh.model.getEntities(3)]
        vol_to_label = {}
        for vol in volumes:
            c = np.asarray(gmsh.model.occ.getCenterOfMass(3, vol), dtype=float)
            vol_to_label[vol] = int(np.argmin(np.linalg.norm(SPHERES[:, :3] - c[None, :], axis=1)))
        for label in range(N_SPHERES):
            tags = [v for v, l in vol_to_label.items() if l == label]
            if not tags:
                raise RuntimeError(f"Missing pore region {label}")
            ph = gmsh.model.addPhysicalGroup(3, tags, tag=1000 + label)
            gmsh.model.setPhysicalName(3, ph, f"pore_{label:03d}")

        pair_to_iface = {p: i for i, p in enumerate(used_pairs)}
        sphere_surfs, cube_surfs = [], {"inlet": [], "outlet": [], "outer_wall": []}
        iface_surfs = {i: [] for i in range(len(used_pairs))}

        for _, surf in gmsh.model.getEntities(2):
            up, _ = gmsh.model.getAdjacencies(2, surf)
            adj = [int(v) for v in up if int(v) in vol_to_label]
            if len(adj) == 2:
                pair = tuple(sorted(vol_to_label[v] for v in adj))
                iid = pair_to_iface.get(pair)
                if iid is not None:
                    iface_surfs[iid].append(surf)
            elif len(adj) == 1:
                st = gmsh.model.getType(2, surf).lower()
                if "sphere" in st:
                    sphere_surfs.append(surf)
                elif "plane" in st:
                    side = _outer_side(surf)
                    if side:
                        cube_surfs[side].append(surf)

        for name, tag, surfs in [
            ("sphere_walls", 1, sphere_surfs),
            ("inlet", 2, cube_surfs["inlet"]),
            ("outlet", 3, cube_surfs["outlet"]),
            ("outer_walls", 4, cube_surfs["outer_wall"]),
        ]:
            if surfs:
                ph = gmsh.model.addPhysicalGroup(2, surfs, tag=tag)
                gmsh.model.setPhysicalName(2, ph, name)
        for iid, surfs in iface_surfs.items():
            if surfs:
                ph = gmsh.model.addPhysicalGroup(2, surfs, tag=2000 + iid)
                gmsh.model.setPhysicalName(2, ph, f"iface_{iid:03d}")

        fields = [_distance_field(sphere_surfs, sphere_size, mesh_size, sphere_band),
                  _distance_field(cube_surfs["inlet"] + cube_surfs["outlet"] + cube_surfs["outer_wall"],
                                  boundary_size, mesh_size, boundary_band),
                  _distance_field([s for ss in iface_surfs.values() for s in ss],
                                  interface_size, mesh_size, interface_band)]
        active = [f for f in fields if f is not None]
        mn = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(mn, "FieldsList", active)
        gmsh.model.mesh.field.setAsBackgroundMesh(mn)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(sphere_size, boundary_size, interface_size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)
        if mesh_file:
            mesh_file.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(mesh_file.resolve()))
        data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=3)
        if data.cell_tags is None:
            raise RuntimeError("No physical volume tags.")
        return _postprocess_partition(data.mesh, data.cell_tags,
                                      tuple(throats), used_pairs, mesh_size)
    finally:
        gmsh.finalize()


# ===================================================================
# Partition W-occ: Weighted Voronoi via OCC (half-space intersection faces)
# ===================================================================

def build_partition_weighted_voronoi_occ(
    mesh_size=0.12, sphere_size=0.05, boundary_size=0.07,
    interface_size=0.06, sphere_band=0.14, boundary_band=0.12,
    interface_band=0.10, mesh_file=None,
) -> PartitionData:
    """Weighted Voronoi partition via OCC fragment.

    Computes power-diagram faces via 4-D Delaunay + half-space clipping,
    embeds them in OCC, and fragments once (same pattern as standard Voronoi).
    """
    used_pairs, faces, throats = _weighted_voronoi_throat_faces()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("real_porous_weighted_voronoi")
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        grains = [gmsh.model.occ.addSphere(float(x), float(y), float(z), float(r))
                  for x, y, z, r in SPHERES]
        fluid, _ = gmsh.model.occ.cut([(3, box)], [(3, t) for t in grains],
                                      removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        # Build OCC surfaces and cut one-by-one (avoids fragment combinatorics)
        cut_surfs = []
        for idx, p in enumerate(faces):
            try:
                cut_surfs.append(_add_planar_face(p))
            except Exception:
                pass
        print(f"  OCC faces: {len(cut_surfs)}/{len(faces)} embedded")
        for idx, sf in enumerate(cut_surfs):
            try:
                fluid, _ = gmsh.model.occ.cut(fluid, [(2, sf)],
                                              removeObject=True, removeTool=True)
            except Exception:
                pass
            if idx % 50 == 0:
                gmsh.model.occ.synchronize()
        gmsh.model.occ.synchronize()

        volumes = [t for _, t in gmsh.model.getEntities(3)]
        vol_to_label = {}
        for vol in volumes:
            c = np.asarray(gmsh.model.occ.getCenterOfMass(3, vol), dtype=float)
            vol_to_label[vol] = int(np.argmin(np.linalg.norm(SPHERES[:, :3] - c[None, :], axis=1)))
        missing_labels = []
        for label in range(N_SPHERES):
            tags = [v for v, l in vol_to_label.items() if l == label]
            if not tags:
                missing_labels.append(label)
                continue  # boundary spheres may lose their cell
            ph = gmsh.model.addPhysicalGroup(3, tags, tag=1000 + label)
            gmsh.model.setPhysicalName(3, ph, f"pore_{label:03d}")
        if missing_labels:
            print(f"  WARNING: Missing pore regions: {missing_labels}")

        pair_to_iface = {p: i for i, p in enumerate(used_pairs)}
        sphere_surfs, cube_surfs = [], {"inlet": [], "outlet": [], "outer_wall": []}
        iface_surfs = {i: [] for i in range(len(used_pairs))}

        for _, surf in gmsh.model.getEntities(2):
            up, _ = gmsh.model.getAdjacencies(2, surf)
            adj = [int(v) for v in up if int(v) in vol_to_label]
            if len(adj) == 2:
                pair = tuple(sorted(vol_to_label[v] for v in adj))
                iid = pair_to_iface.get(pair)
                if iid is not None:
                    iface_surfs[iid].append(surf)
            elif len(adj) == 1:
                st = gmsh.model.getType(2, surf).lower()
                if "sphere" in st:
                    sphere_surfs.append(surf)
                elif "plane" in st:
                    side = _outer_side(surf)
                    if side:
                        cube_surfs[side].append(surf)

        for name, tag, surfs in [
            ("sphere_walls", 1, sphere_surfs),
            ("inlet", 2, cube_surfs["inlet"]),
            ("outlet", 3, cube_surfs["outlet"]),
            ("outer_walls", 4, cube_surfs["outer_wall"]),
        ]:
            if surfs:
                ph = gmsh.model.addPhysicalGroup(2, surfs, tag=tag)
                gmsh.model.setPhysicalName(2, ph, name)
        for iid, surfs in iface_surfs.items():
            if surfs:
                ph = gmsh.model.addPhysicalGroup(2, surfs, tag=2000 + iid)
                gmsh.model.setPhysicalName(2, ph, f"iface_{iid:03d}")

        fields = [_distance_field(sphere_surfs, sphere_size, mesh_size, sphere_band),
                  _distance_field(cube_surfs["inlet"] + cube_surfs["outlet"] + cube_surfs["outer_wall"],
                                  boundary_size, mesh_size, boundary_band),
                  _distance_field([s for ss in iface_surfs.values() for s in ss],
                                  interface_size, mesh_size, interface_band)]
        active = [f for f in fields if f is not None]
        mn = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(mn, "FieldsList", active)
        gmsh.model.mesh.field.setAsBackgroundMesh(mn)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(sphere_size, boundary_size, interface_size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)
        if mesh_file:
            mesh_file.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(mesh_file.resolve()))
        data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=3)
        if data.cell_tags is None:
            raise RuntimeError("No physical volume tags.")
        return _postprocess_partition(data.mesh, data.cell_tags,
                                      tuple(throats), used_pairs, mesh_size)
    finally:
        gmsh.finalize()


# ===================================================================
# Partition W (labeling-based, DEPRECATED — kept for reference)
# ===================================================================

def build_partition_weighted_voronoi(
    mesh_size=0.12, sphere_size=0.05, boundary_size=0.07,
    sphere_band=0.14, boundary_band=0.12, mesh_file=None,
) -> PartitionData:
    """Weighted Voronoi partition — mesh labelling by power distance.

    Generates an unpartitioned fluid mesh (no internal cuts), labels each
    cell by argmin_k power_distance(x_c, sphere_k), and extracts interfaces
    from mesh facets between differently-labelled cells.

    Avoids OCC fragment entirely (same approach as watershed).
    Interface normals come from mesh facet geometry.
    """
    from watershed_partition import (
        generate_unpartitioned_mesh, cell_centers, cell_adjacency,
        split_interface_components, compute_interface_geometry,
    )

    # Generate unpartitioned mesh (reuse watershed's mesh generator)
    msh = generate_unpartitioned_mesh(
        bulk_size=mesh_size, sphere_size=sphere_size,
        boundary_size=boundary_size, sphere_band=sphere_band,
        boundary_band=boundary_band, mesh_file=mesh_file,
    )
    centers_mesh = cell_centers(msh)
    n_cells = len(centers_mesh)

    # Label cells by power distance
    sph_centers = SPHERES[:, :3]
    sph_w = np.sum(sph_centers**2, axis=1) - SPHERES[:, 3]**2

    labels = np.full(n_cells, -1, dtype=np.int32)
    batch = 10000
    for start in range(0, n_cells, batch):
        end = min(start + batch, n_cells)
        pts = centers_mesh[start:end]
        scores = 2.0 * pts @ sph_centers.T - sph_w[None, :]
        labels[start:end] = np.argmax(scores, axis=1).astype(np.int32)

    # Extract interfaces from mesh facets
    pairs, facet_ids = split_interface_components(msh, labels)

    # Interface geometry from mesh facets
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    n_facets = msh.topology.index_map(fdim).size_local
    facet_bary = np.empty((n_facets, 3))
    for f in range(n_facets):
        facet_bary[f] = msh.geometry.x[f2v.links(f), :3].mean(axis=0)
    facet_clearance = sphere_clearance(facet_bary)

    iface_centers, iface_normals, iface_areas, dispersion, saddles = (
        compute_interface_geometry(msh, facet_ids, pairs, labels, facet_clearance)
    )

    # Maximal balls per region
    # Clearance from solid (sphere surfaces + cube walls)
    cell_clearance = np.full(n_cells, np.inf)
    for x, y, z, r in SPHERES:
        cell_clearance = np.minimum(
            cell_clearance,
            np.linalg.norm(centers_mesh - np.array([x, y, z])[None, :], axis=1) - r)
    cell_clearance = np.minimum(cell_clearance, np.minimum(
        np.min(centers_mesh, axis=1), 1.0 - np.min(centers_mesh, axis=1)))

    n_regions = int(labels.max()) + 1
    max_balls = np.zeros((n_regions, 4))
    for lbl in range(n_regions):
        cells = np.flatnonzero(labels == lbl)
        best = int(cells[np.argmax(cell_clearance[cells])])
        max_balls[lbl] = (*centers_mesh[best], cell_clearance[best])

    return PartitionData(
        mesh=msh, cell_centers=centers_mesh, cell_clearance=cell_clearance,
        cell_labels=labels.astype(np.int32),
        maximal_balls=tuple(tuple(b) for b in max_balls), throats=(),
        interface_pairs=pairs, facet_interface_ids=facet_ids,
        interface_centers=iface_centers, interface_normals=iface_normals,
        interface_areas=iface_areas,
        mesh_parameters={"mesh_size": mesh_size},
        cad_counts={"n_regions": n_regions, "method": "weighted_voronoi_labeling"},
    )


# ===================================================================
# Partition 2: Grid (4×4×4=64 cuboids, "平凡傻瓜分法")
# ===================================================================

GRID_PLANES = [0.25, 0.50, 0.75]


def build_partition_grid(
    mesh_size=0.08, sphere_size=0.03, boundary_size=0.05,
    sphere_band=0.10, boundary_band=0.08, mesh_file=None,
) -> PartitionData:
    """Grid partition: cut the cube with 3×3 planes → 4×4×4=64 subdomains."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("real_porous_grid")
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        grains = [gmsh.model.occ.addSphere(float(x), float(y), float(z), float(r))
                  for x, y, z, r in SPHERES]
        fluid, _ = gmsh.model.occ.cut([(3, box)], [(3, t) for t in grains],
                                      removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        # Create cutting planes
        cut_surfs = []
        for axis, vals in enumerate([GRID_PLANES, GRID_PLANES, GRID_PLANES]):
            for v in vals:
                if axis == 0:
                    pts = [(v, 0, 0), (v, 1, 0), (v, 1, 1), (v, 0, 1)]
                elif axis == 1:
                    pts = [(0, v, 0), (1, v, 0), (1, v, 1), (0, v, 1)]
                else:
                    pts = [(0, 0, v), (1, 0, v), (1, 1, v), (0, 1, v)]
                gmsh_pts = [gmsh.model.occ.addPoint(*p) for p in pts]
                lines = [gmsh.model.occ.addLine(gmsh_pts[k], gmsh_pts[(k + 1) % 4])
                         for k in range(4)]
                loop = gmsh.model.occ.addCurveLoop(lines)
                cut_surfs.append(gmsh.model.occ.addPlaneSurface([loop]))

        gmsh.model.occ.fragment(fluid, [(2, t) for t in cut_surfs],
                                removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        # Label volumes by grid cell (ix, iy, iz)
        volumes = [t for _, t in gmsh.model.getEntities(3)]
        vol_to_label = {}
        for vol in volumes:
            c = np.asarray(gmsh.model.occ.getCenterOfMass(3, vol), dtype=float)
            ix = min(int(c[0] / 0.25), 3)
            iy = min(int(c[1] / 0.25), 3)
            iz = min(int(c[2] / 0.25), 3)
            vol_to_label[vol] = ix * 16 + iy * 4 + iz  # 0..63

        n_cells = 64
        for label in range(n_cells):
            tags = [v for v, l in vol_to_label.items() if l == label]
            if not tags:
                raise RuntimeError(f"Missing grid cell {label}")
            ph = gmsh.model.addPhysicalGroup(3, tags, tag=1000 + label)
            gmsh.model.setPhysicalName(3, ph, f"cell_{label:03d}")

        # Surface tagging
        sphere_surfs, cube_surfs = [], {"inlet": [], "outlet": [], "outer_wall": []}
        for _, surf in gmsh.model.getEntities(2):
            up, _ = gmsh.model.getAdjacencies(2, surf)
            adj = [int(v) for v in up if int(v) in vol_to_label]
            st = gmsh.model.getType(2, surf).lower()
            if len(adj) == 1:
                if "sphere" in st:
                    sphere_surfs.append(surf)
                elif "plane" in st:
                    side = _outer_side(surf)
                    if side:
                        cube_surfs[side].append(surf)

        for name, tag, surfs in [
            ("sphere_walls", 1, sphere_surfs),
            ("inlet", 2, cube_surfs["inlet"]),
            ("outlet", 3, cube_surfs["outlet"]),
            ("outer_walls", 4, cube_surfs["outer_wall"]),
        ]:
            if surfs:
                ph = gmsh.model.addPhysicalGroup(2, surfs, tag=tag)
                gmsh.model.setPhysicalName(2, ph, name)

        fields = [_distance_field(sphere_surfs, sphere_size, mesh_size, sphere_band),
                  _distance_field(cube_surfs["inlet"] + cube_surfs["outlet"] + cube_surfs["outer_wall"],
                                  boundary_size, mesh_size, boundary_band)]
        active = [f for f in fields if f is not None]
        mn = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(mn, "FieldsList", active)
        gmsh.model.mesh.field.setAsBackgroundMesh(mn)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(sphere_size, boundary_size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)
        if mesh_file:
            mesh_file.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(mesh_file.resolve()))
        data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=3)
        if data.cell_tags is None:
            raise RuntimeError("No physical volume tags.")
        return _postprocess_partition(data.mesh, data.cell_tags, (), [], mesh_size)
    finally:
        gmsh.finalize()


# ===================================================================
# Partition 3: Watershed (re-exported from watershed_partition.py)
# ===================================================================

def build_partition_watershed(
    mesh_size=0.08, sphere_size=0.03, boundary_size=0.05,
    sphere_band=0.10, boundary_band=0.08, mesh_file=None,
    policy="walls_and_spheres", abs_threshold=0.015, rel_threshold=0.04,
):
    """Watershed clearance-basin partition. Thin wrapper — delegates to
    the watershed_partition module from affine_ddpnm_3d_random_porous."""
    import sys
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    # Use the SPHERES from *this* module by temporarily replacing the import
    import affine_ddpnm_3d_random_porous.watershed_partition as _wp
    import affine_ddpnm_3d_random_porous.random_porous as _rp

    # Patch SPHERES
    _orig_rp_spheres = _rp.SPHERES
    _rp.SPHERES = SPHERES
    try:
        return _wp.build_partition_watershed(
            bulk_size=mesh_size, sphere_size=sphere_size,
            boundary_size=boundary_size, sphere_band=sphere_band,
            boundary_band=boundary_band, mesh_file=mesh_file,
            policy=policy, abs_threshold=abs_threshold,
            rel_threshold=rel_threshold,
        )
    finally:
        _rp.SPHERES = _orig_rp_spheres


# ===================================================================
# Shared post-processing
# ===================================================================

def _postprocess_partition(msh, cell_tags, throats, pairs, mesh_size):
    """Extract PartitionData from a tagged mesh (common to Voronoi + Grid)."""
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    n_cells = msh.topology.index_map(tdim).size_local
    centers = np.empty((n_cells, 3))
    for c in range(n_cells):
        centers[c] = msh.geometry.x[c2v.links(c), :3].mean(axis=0)

    # cell clearance
    clearance = np.full(n_cells, np.inf)
    for x, y, z, r in SPHERES:
        clearance = np.minimum(
            clearance, np.linalg.norm(centers - np.asarray([x, y, z])[None, :], axis=1) - r)
    clearance = np.minimum(clearance, np.minimum(np.min(centers, axis=1), 1.0 - np.min(centers, axis=1)))

    # labels
    raw = np.full(n_cells, -1, dtype=np.int32)
    raw[cell_tags.indices] = cell_tags.values
    labels = raw - 1000

    # interfaces from mesh facets
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    f2c = msh.topology.connectivity(fdim, tdim)
    n_facets = msh.topology.index_map(fdim).size_local
    found_pairs = set()
    facet_pair = [None] * n_facets
    for f in range(n_facets):
        cells = f2c.links(f)
        if len(cells) != 2:
            continue
        a, b = int(labels[cells[0]]), int(labels[cells[1]])
        if a != b:
            p = (min(a, b), max(a, b))
            found_pairs.add(p)
            facet_pair[f] = p
    ordered = tuple(sorted(found_pairs))
    pair_to_id = {p: i for i, p in enumerate(ordered)}
    facet_ids = np.full(n_facets, -1, dtype=np.int32)
    for f, p in enumerate(facet_pair):
        if p is not None:
            facet_ids[f] = pair_to_id[p]

    # interface geometry — use mesh facet normals (works for all partition types)
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    iface_centers = np.zeros((len(ordered), 3))
    iface_normals = np.zeros((len(ordered), 3))
    iface_areas = np.zeros(len(ordered))
    for iid, (a, b) in enumerate(ordered):
        facets = np.flatnonzero(facet_ids == iid)
        normal_sum = np.zeros(3)
        for f in facets:
            tri = msh.geometry.x[f2v.links(int(f)), :3]
            fn = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            area = 0.5 * float(np.linalg.norm(fn))
            unit = fn / max(float(np.linalg.norm(fn)), 1e-30) if area > 0 else np.zeros(3)
            iface_centers[iid] += area * tri.mean(axis=0)
            iface_areas[iid] += area
            normal_sum += area * unit
        iface_centers[iid] /= max(iface_areas[iid], 1e-30)
        if float(np.linalg.norm(normal_sum)) > 1e-30:
            iface_normals[iid] = normal_sum / np.linalg.norm(normal_sum)

    # maximal balls
    max_balls = []
    for label in sorted(set(int(l) for l in labels)):
        cells = np.flatnonzero(labels == label)
        best = int(cells[np.argmax(clearance[cells])])
        max_balls.append((centers[best, 0], centers[best, 1], centers[best, 2], clearance[best]))

    return PartitionData(
        mesh=msh, cell_centers=centers, cell_clearance=clearance,
        cell_labels=labels.astype(np.int32),
        maximal_balls=tuple(max_balls), throats=throats,
        interface_pairs=ordered, facet_interface_ids=facet_ids,
        interface_centers=iface_centers, interface_normals=iface_normals,
        interface_areas=iface_areas,
        mesh_parameters={"mesh_size": mesh_size}, cad_counts={"n_regions": len(set(int(l) for l in labels))},
    )

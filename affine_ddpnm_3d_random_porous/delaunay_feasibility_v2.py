"""Delaunay tetrahedron partition — refined geometric correctness screening (v2).

Pure numpy/scipy/matplotlib.

Improvements over v1:
  1. 4th-sphere penetration: only count spheres whose intersection circle on
     the face plane actually overlaps the triangle.
  2. Per-face positive-clearance connected-component analysis.
  3. Per-face viability flag: positive area, single connected passage.
  4. Accurate constriction (min clearance on the fluid passage).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

# ---------------------------------------------------------------------------
# Frozen sphere realisation (seed 20260804), copied from random_porous.py
# ---------------------------------------------------------------------------
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

CUBE_CORNERS = np.array(
    [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
    dtype=float,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "delaunay_feasibility"

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _triangle_centroid_normal(a, b, c):
    centroid = (a + b + c) / 3.0
    normal = np.cross(b - a, c - a)
    area = 0.5 * float(np.linalg.norm(normal))
    if area > 0:
        normal /= 2.0 * area
    return centroid, area, normal


def _sample_triangle_grid(a, b, c, n_side=25):
    """Uniform barycentric grid over triangle ABC. Returns (n_pts, 3), (n_pts, 3) barycentric.

    n_side controls density: ~ n_side*(n_side+1)/2 points.
    """
    pts = []
    bary = []
    for i in range(n_side + 1):
        for j in range(n_side + 1 - i):
            u = i / n_side
            v = j / n_side
            w = 1.0 - u - v
            pts.append(w * a + v * b + u * c)  # w*A + v*B + u*C
            bary.append([w, v, u])
    if not pts:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.asarray(pts), np.asarray(bary)


def _build_grid_adjacency(bary_coords, n_side, tol=1e-10):
    """Build adjacency list for the barycentric grid. Two points are adjacent
    if they share an edge in the sub-triangulation of the grid."""
    # Build a lookup from barycentric (i,j,k) to point index.
    # Each grid point is (w, v, u) = (k/n_side, j/n_side, i/n_side)
    # where i in [0,n_side], j in [0,n_side-i].
    n = len(bary_coords)
    # Map: (i, j) -> point index
    ij_to_idx = {}
    for idx, (w, v, u) in enumerate(bary_coords):
        i_ = int(round(u * n_side))
        j_ = int(round(v * n_side))
        ij_to_idx[(i_, j_)] = idx

    adj = [[] for _ in range(n)]
    for idx, (w, v, u) in enumerate(bary_coords):
        i_ = int(round(u * n_side))
        j_ = int(round(v * n_side))
        for di, dj in [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]:
            ni, nj = i_ + di, j_ + dj
            nbr = ij_to_idx.get((ni, nj))
            if nbr is not None:
                adj[idx].append(nbr)
    return adj


def _sphere_clearance(pts, centers, radii):
    """Per-point minimum clearance to all spheres."""
    result = np.full(len(pts), np.inf, dtype=float)
    for c, r in zip(centers, radii):
        result = np.minimum(result, np.linalg.norm(pts - c[None, :], axis=1) - r)
    return result


def _find_connected_components(adj, mask):
    """Find connected components of masked nodes. Returns component labels."""
    n = len(mask)
    labels = np.full(n, -1, dtype=int)
    comp_id = 0
    for seed in range(n):
        if not mask[seed] or labels[seed] != -1:
            continue
        stack = [seed]
        labels[seed] = comp_id
        while stack:
            v = stack.pop()
            for nb in adj[v]:
                if mask[nb] and labels[nb] == -1:
                    labels[nb] = comp_id
                    stack.append(nb)
        comp_id += 1
    return labels, comp_id


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def refined_interface_analysis(n_side=25):
    """Refined per-interface analysis with correct geometric screening.

    Parameters
    ----------
    n_side : int
        Sampling grid density per triangle edge (~ n_side²/2 points per face).

    Returns
    -------
    dict with all statistics.
    """
    centers = SPHERES[:, :3].copy()
    radii = SPHERES[:, 3].copy()
    n_spheres = len(centers)

    tri = Delaunay(centers)
    n_tets = len(tri.simplices)
    print(f"Delaunay tetrahedra: {n_tets}")

    # Extract shared faces
    interfaces = []
    seen = set()
    for i in range(n_tets):
        for j in range(4):
            nbr = int(tri.neighbors[i, j])
            if nbr == -1:
                continue
            key = (min(i, nbr), max(i, nbr))
            if key in seen:
                continue
            seen.add(key)
            face_verts = [
                int(tri.simplices[i, (j + 1) % 4]),
                int(tri.simplices[i, (j + 2) % 4]),
                int(tri.simplices[i, (j + 3) % 4]),
            ]
            other_i = int(tri.simplices[i, j])
            other_nbr_candidates = set(int(v) for v in tri.simplices[nbr]) - set(face_verts)
            other_nbr = other_nbr_candidates.pop() if other_nbr_candidates else -1
            a, b, c = centers[face_verts[0]], centers[face_verts[1]], centers[face_verts[2]]
            centroid, area, normal = _triangle_centroid_normal(a, b, c)
            # The 5 (or 4 if other_i==other_nbr) sphere indices involved
            involved = set(face_verts) | {other_i, other_nbr}
            involved.discard(-1)
            interfaces.append({
                "face_verts": face_verts,
                "other_a": other_i,
                "other_b": other_nbr,
                "involved_spheres": sorted(involved),
                "centroid": centroid.tolist(),
                "area": float(area),
                "normal": normal.tolist(),
                "tri_verts_xyz": [a.tolist(), b.tolist(), c.tolist()],
            })

    n_interfaces = len(interfaces)
    print(f"Shared interfaces: {n_interfaces}")

    # --- Per-face refined analysis ---
    print(f"Sampling each face with {n_side}×{n_side} grid...")
    for idx, iface in enumerate(interfaces):
        fv = iface["face_verts"]
        a, b, c = centers[fv[0]], centers[fv[1]], centers[fv[2]]
        samples, bary = _sample_triangle_grid(a, b, c, n_side)
        n_samples = len(samples)

        if n_samples == 0:
            iface.update({
                "n_samples": 0, "pos_area_frac": 0.0, "n_components": 0,
                "max_clearance": -1.0, "min_pos_clearance": np.inf,
                "intruders": [], "intruder_count": 0,
                "viable": False, "viable_reason": "degenerate face",
            })
            continue

        # Full clearance to all 27 spheres
        clearance = _sphere_clearance(samples, centers, radii)
        pos_mask = clearance > 0.0
        n_pos = int(pos_mask.sum())
        pos_frac = n_pos / n_samples if n_samples > 0 else 0.0

        # Connected components of the positive-clearance region
        adj = _build_grid_adjacency(bary, n_side)
        comp_labels, n_comps = _find_connected_components(adj, pos_mask)

        # Identify intruding spheres: for each sphere NOT in the two adjacent
        # tets, check if its intersection circle overlaps the triangle.
        involved = set(iface["involved_spheres"])
        _, _, normal = _triangle_centroid_normal(a, b, c)
        intruders = []
        for k in range(n_spheres):
            if k in involved:
                continue
            ck = centers[k]
            rk = radii[k]
            # Distance from centre to face plane
            dist = abs(float(np.dot(ck - a, normal)))
            if dist >= rk - 1e-10:
                continue  # sphere doesn't reach the plane
            # Intersection circle on the plane
            circle_r = np.sqrt(max(0.0, rk**2 - dist**2))
            proj = ck - float(np.dot(ck - a, normal)) * normal
            # Does this circle overlap the triangle?
            if _circle_overlaps_triangle(proj, circle_r, a, b, c, samples, clearance, k):
                intruders.append({
                    "sphere_idx": k,
                    "center": ck.tolist(),
                    "radius": float(rk),
                    "dist_to_plane": float(dist),
                    "circle_radius": float(circle_r),
                })
        n_intruders = len(intruders)

        max_cl = float(clearance.max())
        min_pos_cl = float(clearance[pos_mask].min()) if n_pos > 0 else np.inf

        # Viability criteria:
        # - At least 1% of face area has positive clearance
        # - The positive region is a single connected component
        # - (No intruders = clean; with intruders = viable but holed)
        viable = pos_frac >= 0.01 and n_comps == 1
        if pos_frac < 0.01:
            reason = "blocked (< 1% open area)"
        elif n_comps == 0:
            reason = "no positive clearance"
        elif n_comps > 1:
            reason = f"fragmented ({n_comps} components)"
        else:
            reason = "ok" if n_intruders == 0 else f"viable ({n_intruders} holes)"

        iface.update({
            "n_samples": n_samples,
            "pos_area_frac": round(pos_frac, 6),
            "n_components": n_comps,
            "max_clearance": round(max_cl, 6),
            "min_pos_clearance": round(min_pos_cl, 6) if n_pos > 0 else None,
            "intruders": intruders,
            "intruder_count": n_intruders,
            "viable": viable,
            "viable_reason": reason,
        })

    # --- Summary statistics ---
    viable_count = sum(1 for f in interfaces if f["viable"])
    blocked_count = sum(1 for f in interfaces if not f["viable"] and f["pos_area_frac"] < 0.01)
    fragmented_count = sum(1 for f in interfaces if not f["viable"] and f["n_components"] > 1)

    total_intruders = sum(f["intruder_count"] for f in interfaces)
    faces_with_intruders = sum(1 for f in interfaces if f["intruder_count"] > 0)
    clean_faces = n_interfaces - faces_with_intruders

    max_clearances = np.array([f["max_clearance"] for f in interfaces])
    min_pos_clearances = np.array([
        f["min_pos_clearance"] for f in interfaces if f["min_pos_clearance"] is not None
    ])

    print(f"\n--- Refined results (n_side={n_side}) ---")
    print(f"Viable interfaces:       {viable_count}/{n_interfaces}")
    print(f"  Blocked (< 1% open):   {blocked_count}")
    print(f"  Fragmented (>1 comp):  {fragmented_count}")
    print(f"Faces with 4th-sphere intruders: {faces_with_intruders}/{n_interfaces}")
    print(f"  Clean faces (no intruders):    {clean_faces}/{n_interfaces}")
    print(f"  Total intruder instances:      {total_intruders}")
    print(f"Max clearance:  min={max_clearances.min():.4f}  med={float(np.median(max_clearances)):.4f}  max={max_clearances.max():.4f}")
    if len(min_pos_clearances) > 0:
        print(f"Min pos clearance (constriction): min={min_pos_clearances.min():.6f}  med={float(np.median(min_pos_clearances)):.6f}  max={min_pos_clearances.max():.6f}")

    # Convex hull check
    hull = ConvexHull(centers)
    hull_ok = np.all(
        hull.equations[:, :3] @ CUBE_CORNERS.T + hull.equations[:, 3:4] <= 1e-10, axis=0
    )
    corners_in = int(hull_ok.sum())

    return {
        "n_spheres": n_spheres,
        "n_tetrahedra": n_tets,
        "n_interfaces": n_interfaces,
        "grid_n_side": n_side,
        "viable_count": viable_count,
        "blocked_count": blocked_count,
        "fragmented_count": fragmented_count,
        "faces_with_intruders": faces_with_intruders,
        "clean_faces": clean_faces,
        "total_intruders": total_intruders,
        "max_clearance_min": float(max_clearances.min()),
        "max_clearance_median": float(np.median(max_clearances)),
        "max_clearance_max": float(max_clearances.max()),
        "min_pos_clearance_min": float(min_pos_clearances.min()) if len(min_pos_clearances) > 0 else None,
        "min_pos_clearance_median": float(np.median(min_pos_clearances)) if len(min_pos_clearances) > 0 else None,
        "min_pos_clearance_max": float(min_pos_clearances.max()) if len(min_pos_clearances) > 0 else None,
        "voronoi_saddle_median": 0.0073,
        "corners_inside_hull": corners_in,
        "interfaces": interfaces,
        "_centers": centers,
        "_radii": radii,
        "_tri": tri,
        "_hull": hull,
        "_hull_ok": hull_ok,
    }


def _circle_overlaps_triangle(center, radius, a, b, c, samples, clearance, sphere_idx):
    """Check if sphere k's intersection circle on the plane overlaps the triangle.

    The circle is at *center* with *radius* on the face plane.  Returns True if
    the circular disc meaningfully reduces the positive-clearance region of the
    triangle (i.e. some points in the triangle that WOULD have positive
    clearance to the 3 vertex spheres are inside this disc).

    Approach: check whether this sphere *alone* would exclude any sample point
    that has positive clearance to all OTHER spheres.
    """
    tri_verts = np.asarray([a, b, c])
    # Quick reject: if the circle center is more than radius + triangle_bbox
    # from the triangle centroid, it can't overlap.
    tri_centroid = tri_verts.mean(axis=0)
    tri_extent = max(np.linalg.norm(v - tri_centroid) for v in tri_verts)
    dist_center_to_tri = float(np.linalg.norm(center - tri_centroid))
    if dist_center_to_tri > radius + tri_extent + 1e-10:
        return False

    # Direct check: for every sample point, clearance to THIS sphere.
    # If a point has positive clearance to all other spheres but negative
    # clearance to this sphere, the sphere is intruding.
    for i, pt in enumerate(samples):
        cl_to_this = float(np.linalg.norm(pt - center)) - radius
        cl_all_except_this = clearance[i]  # already includes this sphere
        # Recompute clearance without this sphere:
        # Wait, clearance already includes this sphere. Let me recompute.
        pass

    # Simpler: compute clearance to this sphere at each sample.
    # A point is "affected" by this sphere if clearance_to_this_sphere < 0
    # AND the point is inside or near the triangle.
    cl_to_sphere = np.linalg.norm(samples - center[None, :], axis=1) - radius
    # Point is inside the disc
    inside_disc = cl_to_sphere < -1e-10
    if not np.any(inside_disc):
        return False
    # The point must also be within the triangle (all samples are, by construction)
    # AND must have been in the positive-clearance region to other spheres
    # to count as "the sphere creates a new hole".

    # Compute clearance to all spheres EXCEPT this one
    other_clearance = _sphere_clearance_except(samples, SPHERES[:, :3], SPHERES[:, 3], sphere_idx)
    # A point counts as "intruded" if: it has positive clearance to all other
    # spheres BUT is inside this sphere's disc. This means this sphere is the
    # one creating the blockage at that point.
    intruded = inside_disc & (other_clearance > 0)
    return bool(np.any(intruded))


def _sphere_clearance_except(pts, centers, radii, except_idx):
    """Clearance to all spheres EXCEPT except_idx."""
    result = np.full(len(pts), np.inf, dtype=float)
    for k, (c, r) in enumerate(zip(centers, radii)):
        if k == except_idx:
            continue
        result = np.minimum(result, np.linalg.norm(pts - c[None, :], axis=1) - r)
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_refined(report, out_dir):
    centers = report["_centers"]
    radii = report["_radii"]
    interfaces = report["interfaces"]

    fig = plt.figure(figsize=(20, 16))

    # --- 3D view: faces coloured by viability ---
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    _draw_cube(ax1)
    for iface in interfaces:
        tri_xyz = np.asarray(iface["tri_verts_xyz"])
        if not iface["viable"]:
            if iface["n_components"] == 0 or iface["pos_area_frac"] < 0.01:
                color, alpha, label = "#e74c3c", 0.8, "blocked"
            else:
                color, alpha, label = "#f39c12", 0.7, "fragmented"
        elif iface["intruder_count"] > 0:
            color, alpha, label = "#f1c40f", 0.5, "holed"
        else:
            color, alpha, label = "#2ecc71", 0.4, "clean"
        ax1.plot_trisurf(tri_xyz[:, 0], tri_xyz[:, 1], tri_xyz[:, 2],
                         color=color, alpha=alpha, shade=True, antialiased=True)
    ax1.set_title("Interface viability\n"
                  f"green=clean  yellow=holed  orange=fragmented  red=blocked")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    _set_axes_equal(ax1, centers)

    # --- 3D view: one representative face with sampling grid ---
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    _draw_cube(ax2)
    # Pick a clean face and a blocked face
    clean = [f for f in interfaces if f["viable"] and f["intruder_count"] == 0]
    blocked = [f for f in interfaces if not f["viable"]]
    show_faces = clean[:3] + blocked[:2]
    for iface in show_faces:
        tri_xyz = np.asarray(iface["tri_verts_xyz"])
        if iface["viable"]:
            color = "#2ecc71" if iface["intruder_count"] == 0 else "#f1c40f"
            alpha = 0.5
        else:
            color, alpha = "#e74c3c", 0.7
        ax2.plot_trisurf(tri_xyz[:, 0], tri_xyz[:, 1], tri_xyz[:, 2],
                         color=color, alpha=alpha, shade=True)
    # Draw sphere wireframes
    _draw_spheres_wireframe(ax2, centers, radii, n_theta=12, n_phi=6)
    ax2.set_title("Sample faces + spheres")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    _set_axes_equal(ax2, centers)

    # --- Histogram: positive area fraction ---
    ax3 = fig.add_subplot(2, 2, 3)
    pos_fracs = [f["pos_area_frac"] for f in interfaces]
    ax3.hist(pos_fracs, bins=30, color="steelblue", edgecolor="white")
    ax3.axvline(0.01, color="red", lw=1, linestyle="--", label="1% threshold")
    ax3.set_xlabel("positive-clearance area fraction")
    ax3.set_ylabel("interface count")
    ax3.set_title(f"Positive-clearance area fraction ({len(interfaces)} interfaces)")
    ax3.legend()

    # --- Bar chart: summary ---
    ax4 = fig.add_subplot(2, 2, 4)
    cats = ["clean", "holed\n(viable)", "fragmented", "blocked"]
    clean_n = report["clean_faces"]
    holed_n = report["viable_count"] - report["clean_faces"]
    frag_n = report["fragmented_count"]
    blocked_n = report["blocked_count"]
    values = [clean_n, holed_n, frag_n, blocked_n]
    colors = ["#2ecc71", "#f1c40f", "#f39c12", "#e74c3c"]
    bars = ax4.bar(cats, values, color=colors, edgecolor="black")
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha="center", fontsize=11, fontweight="bold")
    ax4.set_ylabel("interface count")
    ax4.set_title(f"Interface classification ({report['n_interfaces']} total)\n"
                  f"grid {report['grid_n_side']}×{report['grid_n_side']}")

    fig.tight_layout()
    fig.savefig(out_dir / "delaunay_refined_v2.png", dpi=150)
    plt.close(fig)
    print("Saved delaunay_refined_v2.png")


def _draw_cube(ax):
    edges = [
        ((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 0, 1)),
        ((1, 1, 0), (1, 0, 0)), ((1, 1, 0), (0, 1, 0)), ((1, 1, 0), (1, 1, 1)),
        ((1, 0, 1), (1, 0, 0)), ((1, 0, 1), (0, 0, 1)), ((1, 0, 1), (1, 1, 1)),
        ((0, 1, 1), (0, 1, 0)), ((0, 1, 1), (0, 0, 1)), ((0, 1, 1), (1, 1, 1)),
    ]
    for (x0, y0, z0), (x1, y1, z1) in edges:
        ax.plot([x0, x1], [y0, y1], [z0, z1], color="black", lw=0.5, alpha=0.3)


def _draw_spheres_wireframe(ax, centers, radii, n_theta=12, n_phi=6):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    for (cx, cy, cz), r in zip(centers, radii):
        x_eq = cx + r * np.cos(theta)
        y_eq = cy + r * np.sin(theta)
        ax.plot(x_eq, y_eq, np.full_like(theta, cz), color="gray", lw=0.2, alpha=0.25)
        for p in phi[1:-1]:
            rr = r * np.sin(p)
            ax.plot(cx + rr * np.cos(theta), cy + rr * np.sin(theta),
                    np.full_like(theta, cz + r * np.cos(p)),
                    color="gray", lw=0.15, alpha=0.2)


def _set_axes_equal(ax, pts):
    mid = pts.mean(axis=0)
    max_range = 0.5 * (pts.max(axis=0) - pts.min(axis=0)).max() * 1.15
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Delaunay refined geometric correctness screening (v2)")
    print(f"output dir: {OUT_DIR}")
    print("=" * 60)

    report = refined_interface_analysis(n_side=25)

    # Save JSON (strip internal ndarray keys)
    json_report = {k: v for k, v in report.items() if not k.startswith("_")}
    with open(OUT_DIR / "delaunay_report_v2.json", "w", encoding="utf-8") as fh:
        json.dump(json_report, fh, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"Saved delaunay_report_v2.json")

    # Plot
    plot_refined(report, OUT_DIR)

    # Summary markdown
    _write_summary_v2(report)
    print("\nDone.")


def _write_summary_v2(report):
    lines = [
        "# Delaunay tetrahedron partition — refined geometric screening (v2)",
        "",
        f"Sampling grid: {report['grid_n_side']}×{report['grid_n_side']} per triangular face",
        f"Delaunay tetrahedra: {report['n_tetrahedra']}",
        f"Shared interfaces: {report['n_interfaces']}",
        "",
        "## Interface classification",
        "",
        f"| class | count | pct |",
        f"|---|---|---|",
        f"| **clean** (viable, no intruders) | {report['clean_faces']} | {report['clean_faces']/report['n_interfaces']*100:.1f}% |",
        f"| **holed** (viable, with intruders) | {report['viable_count'] - report['clean_faces']} | {(report['viable_count'] - report['clean_faces'])/report['n_interfaces']*100:.1f}% |",
        f"| **fragmented** (>1 positive-clearance component) | {report['fragmented_count']} | {report['fragmented_count']/report['n_interfaces']*100:.1f}% |",
        f"| **blocked** (< 1% open area) | {report['blocked_count']} | {report['blocked_count']/report['n_interfaces']*100:.1f}% |",
        "",
        "## Intruder statistics",
        f"- Faces with ≥1 intruder: **{report['faces_with_intruders']}/{report['n_interfaces']}**",
        f"- Clean faces (no intruders): **{report['clean_faces']}/{report['n_interfaces']}**",
        f"- Total intruder instances: {report['total_intruders']}",
        "",
        "## Clearance statistics",
        f"- Max clearance (saddle): med={report['max_clearance_median']:.4f} (Voronoi ref=0.0073)",
        f"- Min positive clearance (constriction): med={report['min_pos_clearance_median']:.6f}" if report['min_pos_clearance_median'] is not None else "",
        f"- Convex hull corners inside: {report['corners_inside_hull']}/8",
        "",
        "## Decision",
    ]

    if report["viable_count"] == report["n_interfaces"] and report["clean_faces"] == report["n_interfaces"]:
        lines.append("✅ All interfaces geometrically correct — no intruders, all viable.")
    elif report["viable_count"] == report["n_interfaces"]:
        lines.append(f"⚠️ All interfaces viable but {report['faces_with_intruders']} have 4th-sphere holes.")
    else:
        lines.append(f"⚠️ {report['n_interfaces'] - report['viable_count']} interfaces failed viability: "
                     f"{report['blocked_count']} blocked, {report['fragmented_count']} fragmented.")

    with open(OUT_DIR / "delaunay_summary_v2.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved delaunay_summary_v2.md")


if __name__ == "__main__":
    main()

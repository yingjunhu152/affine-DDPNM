"""Delaunay tetrahedron partition feasibility check for random spheres.

Pure numpy/scipy/matplotlib — no FEniCSx, no Gmsh.

Analyses the "trivial partition method" (handoff §5.4): partition the fluid
domain by the Delaunay tetrahedralisation of the 27 sphere centres.  Each
tetrahedron is one subdomain cell; each shared triangular face between two
adjacent tetrahedra is an interface whose three vertices are sphere centres
and whose plane passes through those three centres — the direct analogue of
the regular-grid 4×4×4 partition for the random geometry.

Produces the four statistics required by handoff §5.4⑤:
  1. tetrahedron count and shared-face count
  2. number of faces whose plane is penetrated by a 4th sphere
  3. number of faces with no fluid channel (3 vertex discs cover the triangle)
  4. per-face saddle clearance distribution (compare Voronoi median 0.0073)

Also checks convex-hull coverage of the unit cube and adds 8 cube-corner
dummy points when the hull leaves cube corners uncovered.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

# ---------------------------------------------------------------------------
# Frozen sphere realisation (seed 20260804), copied from random_porous.py
# so this script stays standalone — no FEniCSx/Gmsh imports needed.
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
# Helpers
# ---------------------------------------------------------------------------


def _triangle_geometry(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Centroid, area, unit normal of triangle ABC (CCW orientation)."""
    centroid = (a + b + c) / 3.0
    normal = np.cross(b - a, c - a)
    area = 0.5 * float(np.linalg.norm(normal))
    if area > 0:
        normal /= 2.0 * area
    return centroid, area, normal


def _point_in_triangle(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> bool:
    """Barycentric test: *p* must be a 3-element array."""
    v0, v1, v2 = c - a, b - a, p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-30:
        return False
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return v >= -1e-12 and w >= -1e-12 and v + w <= 1.0 + 1e-12


def _sample_triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray, n: int = 200):
    """Uniform grid samples over triangle ABC.  Returns (n_pts, 3)."""
    n_side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(n_side + 1):
        for j in range(n_side + 1 - i):
            u = i / n_side
            v = j / n_side
            if u + v > 1.0 + 1e-12:
                continue
            w = 1.0 - u - v
            pts.append(w * a + v * b + u * c)
    return np.asarray(pts) if pts else np.empty((0, 3))


def _sphere_clearance(pts: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """|P - centre| - r for every sphere; min over spheres (per query point)."""
    result = np.full(len(pts), np.inf, dtype=float)
    for c, r in zip(centers, radii):
        result = np.minimum(result, np.linalg.norm(pts - c[None, :], axis=1) - r)
    return result


def _face_clearance_3vertex(
    samples: np.ndarray, tri_verts: np.ndarray, tri_radii: np.ndarray
) -> np.ndarray:
    """Clearance to the three vertex spheres only (for the fluid-channel check)."""
    result = np.full(len(samples), np.inf, dtype=float)
    for v, r in zip(tri_verts, tri_radii):
        result = np.minimum(result, np.linalg.norm(samples - v[None, :], axis=1) - r)
    return result


# ---------------------------------------------------------------------------
# 1. Delaunay triangulation
# ---------------------------------------------------------------------------

def run_delaunay_analysis() -> dict:
    centers = SPHERES[:, :3].copy()
    radii = SPHERES[:, 3].copy()
    n_spheres = len(centers)

    tri = Delaunay(centers)
    n_tets = len(tri.simplices)
    print(f"Delaunay tetrahedra: {n_tets}")

    # --- 2. Extract shared faces (interfaces between adjacent tets) ---
    # neighbours[i, j] = neighbour idx or -1 (convex hull face)
    interfaces: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for i in range(n_tets):
        for j in range(4):
            nbr = int(tri.neighbors[i, j])
            if nbr == -1:
                continue
            key = (min(i, nbr), max(i, nbr))
            if key in seen:
                continue
            seen.add(key)
            # face j is opposite vertex j of simplex i
            face_verts_idx = [
                int(tri.simplices[i, (j + 1) % 4]),
                int(tri.simplices[i, (j + 2) % 4]),
                int(tri.simplices[i, (j + 3) % 4]),
            ]
            # The other vertex in each tet
            other_i = int(tri.simplices[i, j])
            # find which face of nbr matches (its vertices minus ours gives the
            # other vertex in nbr)
            other_nbr_candidates = set(int(v) for v in tri.simplices[nbr]) - set(face_verts_idx)
            other_nbr = other_nbr_candidates.pop() if other_nbr_candidates else -1

            a, b, c = centers[face_verts_idx]
            centroid, area, normal = _triangle_geometry(a, b, c)
            interfaces.append(
                {
                    "tet_a": i,
                    "tet_b": nbr,
                    "face_verts": face_verts_idx,          # 3 sphere-centre indices
                    "other_a": other_i,                     # non-face vertex of tet a
                    "other_b": other_nbr,                   # non-face vertex of tet b
                    "centroid": centroid,
                    "area": area,
                    "normal": normal,
                }
            )

    n_interfaces = len(interfaces)
    print(f"Shared interfaces: {n_interfaces}")

    # --- 3. Convex hull coverage of the unit cube ---
    hull = ConvexHull(centers)
    hull_ok = np.all(hull.equations[:, :3] @ CUBE_CORNERS.T + hull.equations[:, 3:4] <= 1e-10, axis=0)
    print(f"\nCube corners inside convex hull: {hull_ok.sum()}/8")
    for corner, ok in zip(CUBE_CORNERS, hull_ok):
        status = "inside" if ok else "OUTSIDE"
        print(f"  {corner}  {status}")

    # Also check a grid of points on each cube face to assess coverage gaps
    face_missing = []
    for face_axis in range(3):
        for face_val in (0.0, 1.0):
            grid = np.linspace(0.05, 0.95, 10)
            pts = np.array(np.meshgrid(grid, grid, indexing="ij")).reshape(2, -1).T
            face_pts = np.zeros((len(pts), 3))
            axes = [0, 1, 2]
            axes.remove(face_axis)
            face_pts[:, axes[0]] = pts[:, 0]
            face_pts[:, axes[1]] = pts[:, 1]
            face_pts[:, face_axis] = face_val
            inside = np.all(
                hull.equations[:, :3] @ face_pts.T + hull.equations[:, 3:4] <= 1e-10, axis=0
            )
            n_outside = int((~inside).sum())
            if n_outside > 0:
                face_missing.append(
                    {"axis": face_axis, "value": face_val, "n_outside_of_100": n_outside}
                )
    print(f"\nCube face sampling (10×10 grid per face):")
    for item in face_missing:
        axis_name = ["x", "y", "z"][item["axis"]]
        print(f"  {axis_name}={item['value']}: {item['n_outside_of_100']}/100 pts outside hull")

    # --- 3b. Delaunay with 8 cube-corner dummies ---
    centers_with_dummies = np.vstack([centers, CUBE_CORNERS])
    tri_dummy = Delaunay(centers_with_dummies)
    n_tets_dummy = len(tri_dummy.simplices)

    # Count shared faces among "real" tets only (exclude dummies from interface
    # count, but count real–dummy interfaces as boundary faces)
    dummy_start = n_spheres  # indices n_spheres .. n_spheres+7 are dummies
    n_real_interfaces = 0
    n_boundary_interfaces = 0  # real–dummy shared faces
    for i in range(n_tets_dummy):
        for j in range(4):
            nbr = int(tri_dummy.neighbors[i, j])
            if nbr == -1 or i >= nbr:
                continue
            verts_i = set(int(v) for v in tri_dummy.simplices[i])
            verts_nbr = set(int(v) for v in tri_dummy.simplices[nbr])
            has_dummy_i = any(v >= dummy_start for v in verts_i)
            has_dummy_nbr = any(v >= dummy_start for v in verts_nbr)
            if has_dummy_i and has_dummy_nbr:
                continue  # dummy–dummy: skip
            if has_dummy_i or has_dummy_nbr:
                n_boundary_interfaces += 1
            else:
                n_real_interfaces += 1

    print(f"\nWith 8 cube-corner dummies:")
    print(f"  tetrahedra: {n_tets_dummy}  (was {n_tets})")
    print(f"  real–real interfaces: {n_real_interfaces}  (was {n_interfaces})")
    print(f"  real–dummy boundary faces: {n_boundary_interfaces}")

    # --- 4. Per-interface pitfall checks ---
    # 4a. 4th-sphere penetration
    # 4b. No fluid channel (covered by 3 vertex discs)
    # 4c. Saddle clearance

    penetrated_count = 0
    no_channel_count = 0
    saddle_clearances = np.zeros(len(interfaces))
    max_clearance_pts = np.zeros((len(interfaces), 3))

    for idx, iface in enumerate(interfaces):
        fv = iface["face_verts"]
        a, b, c = centers[fv[0]], centers[fv[1]], centers[fv[2]]
        ra, rb, rc = radii[fv[0]], radii[fv[1]], radii[fv[2]]
        tri_verts = np.asarray([a, b, c])
        tri_radii = np.asarray([ra, rb, rc])

        # All sphere indices involved in the two adjacent tets (up to 5)
        involved = set(fv) | {iface["other_a"], iface["other_b"]}
        involved.discard(-1)

        # --- 4a. 4th-sphere plane penetration ---
        # Check: for any sphere k NOT in the two adjacent tets, is
        # dist(center_k, plane) < r_k?  If so the sphere intersects the
        # interface plane, potentially creating a hole.
        centroid, area, normal = iface["centroid"], iface["area"], iface["normal"]
        penetrated = False
        for k in range(n_spheres):
            if k in involved:
                continue
            dist_to_plane = abs(float(np.dot(centers[k] - centroid, normal)))
            if dist_to_plane < radii[k] - 1e-10:
                penetrated = True
                break
        if penetrated:
            penetrated_count += 1
            iface["penetrated"] = True
        else:
            iface["penetrated"] = False

        # --- 4b. Fluid channel check ---
        # Sample the triangle and check clearance to the 3 vertex spheres.
        samples = _sample_triangle(a, b, c, n=300)
        if len(samples) == 0:
            clearance_3v = np.array([-1.0])
        else:
            clearance_3v = _face_clearance_3vertex(samples, tri_verts, tri_radii)
        max_cl_3v = float(clearance_3v.max()) if len(clearance_3v) else -1.0
        if max_cl_3v <= 0.0:
            no_channel_count += 1
            iface["no_channel"] = True
        else:
            iface["no_channel"] = False
        iface["max_clearance_3v"] = max_cl_3v

        # --- 4c. Saddle clearance (all 27 spheres) ---
        if len(samples) > 0:
            full_clearance = _sphere_clearance(samples, centers, radii)
            best_i = int(np.argmax(full_clearance))
            saddle_clearances[idx] = float(full_clearance[best_i])
            max_clearance_pts[idx] = samples[best_i]
        else:
            saddle_clearances[idx] = -1.0

    print(f"\nPer-interface pitfalls:")
    print(f"  4th-sphere penetrated:  {penetrated_count}/{n_interfaces}")
    print(f"  no fluid channel (3-disc cover): {no_channel_count}/{n_interfaces}")
    print(f"  saddle clearance: min={saddle_clearances.min():.6f}  "
          f"median={float(np.median(saddle_clearances)):.6f}  "
          f"max={saddle_clearances.max():.6f}")
    print(f"  Voronoi reference: median saddle displacement = 0.0073")

    # Additional: clearance at triangle centroid (quick diagnostic)
    centroid_clearances = np.zeros(len(interfaces))
    for idx, iface in enumerate(interfaces):
        centroid_clearances[idx] = float(
            _sphere_clearance(iface["centroid"].reshape(1, 3), centers, radii)[0]
        )
    print(f"  centroid clearance: min={centroid_clearances.min():.6f}  "
          f"median={float(np.median(centroid_clearances)):.6f}  "
          f"max={centroid_clearances.max():.6f}")

    # --- 5. Saddle clearance histogram bins ---
    bins = [-0.05, -0.02, 0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25]
    hist, _ = np.histogram(saddle_clearances, bins=bins)

    return {
        "n_spheres": n_spheres,
        "n_tetrahedra": n_tets,
        "n_tetrahedra_with_dummies": n_tets_dummy,
        "n_interfaces": n_interfaces,
        "n_real_interfaces_with_dummies": n_real_interfaces,
        "n_boundary_interfaces_with_dummies": n_boundary_interfaces,
        "corners_inside_hull": int(hull_ok.sum()),
        "corners_outside_hull": int((~hull_ok).sum()),
        "face_coverage": face_missing,
        "penetrated_count": penetrated_count,
        "no_channel_count": no_channel_count,
        "saddle_clearance_min": float(saddle_clearances.min()),
        "saddle_clearance_median": float(np.median(saddle_clearances)),
        "saddle_clearance_max": float(saddle_clearances.max()),
        "centroid_clearance_min": float(centroid_clearances.min()),
        "centroid_clearance_median": float(np.median(centroid_clearances)),
        "centroid_clearance_max": float(centroid_clearances.max()),
        "voronoi_saddle_median": 0.0073,
        "saddle_histogram_bins": bins,
        "saddle_histogram_counts": hist.tolist(),
        "interfaces": [
            {
                "face_verts": iface["face_verts"],
                "area": float(iface["area"]),
                "penetrated": iface["penetrated"],
                "no_channel": iface["no_channel"],
                "max_clearance_3v": float(iface["max_clearance_3v"]),
                "saddle_clearance": float(saddle_clearances[i]),
            }
            for i, iface in enumerate(interfaces)
        ],
        # raw data for plotting
        "_tri": tri,
        "_centers": centers,
        "_radii": radii,
        "_interfaces": interfaces,
        "_saddle_clearances": saddle_clearances,
        "_max_clearance_pts": max_clearance_pts,
        "_hull": hull,
        "_hull_ok": hull_ok,
        "_tri_dummy": tri_dummy,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_delaunay_partition(report: dict, out_dir: Path) -> None:
    centers = report["_centers"]
    radii = report["_radii"]
    tri = report["_tri"]
    interfaces = report["_interfaces"]
    saddle_clearances = report["_saddle_clearances"]
    hull = report["_hull"]
    hull_ok = report["_hull_ok"]
    tri_dummy = report["_tri_dummy"]

    # --- Figure 1: Delaunay wireframe + spheres + cube ---
    fig = plt.figure(figsize=(18, 8))

    # 1a. Delaunay edges only (no dummies)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    _draw_cube(ax1)
    # Delaunay edges
    for simplex in tri.simplices:
        for a_idx in range(4):
            for b_idx in range(a_idx + 1, 4):
                a = centers[simplex[a_idx]]
                b = centers[simplex[b_idx]]
                ax1.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                         color="gray", lw=0.3, alpha=0.5)
    # Sphere centres
    ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                c="black", s=8, zorder=5)
    # Corners: green = inside hull, red = outside
    for corner, ok in zip(CUBE_CORNERS, hull_ok):
        color = "green" if ok else "red"
        ax1.scatter(*corner, c=color, s=60, marker="s", zorder=6, edgecolors="black")
    ax1.set_title(f"Delaunay wireframe ({len(tri.simplices)} tets)\n"
                  f"green corners = inside hull, red = outside")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    _set_axes_equal(ax1, centers)

    # 1b. Interfaces colour-coded by penalties
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _draw_cube(ax2)
    for idx, iface in enumerate(interfaces):
        fv = iface["face_verts"]
        a, b, c = centers[fv[0]], centers[fv[1]], centers[fv[2]]
        tri_pts = np.asarray([a, b, c])
        if iface["no_channel"]:
            color, alpha, label = "red", 0.7, "no channel"
        elif iface["penetrated"]:
            color, alpha, label = "orange", 0.5, "penetrated"
        else:
            color, alpha, label = "steelblue", 0.35, "clean"
        ax2.plot_trisurf(tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2],
                         color=color, alpha=alpha, shade=True)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="red", alpha=0.7, label=f"no channel ({report['no_channel_count']})"),
        Patch(facecolor="orange", alpha=0.5, label=f"4th-sphere penetrated ({report['penetrated_count']})"),
        Patch(facecolor="steelblue", alpha=0.35, label=f"clean ({report['n_interfaces'] - report['penetrated_count'] - report['no_channel_count']})"),
    ]
    ax2.legend(handles=legend_elements, loc="upper left", fontsize=8)
    ax2.set_title(f"Interfaces ({report['n_interfaces']} faces)\n"
                  f"red=no fluid channel, orange=penetrated, blue=clean")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    _set_axes_equal(ax2, centers)

    fig.tight_layout()
    fig.savefig(out_dir / "delaunay_partition.png", dpi=150)
    plt.close(fig)
    print("Saved delaunay_partition.png")

    # --- Figure 2: With cube-corner dummies ---
    fig2 = plt.figure(figsize=(16, 7))

    ax3 = fig2.add_subplot(1, 2, 1, projection="3d")
    _draw_cube(ax3)
    for simplex in tri_dummy.simplices:
        pts = np.vstack([centers, CUBE_CORNERS])
        for a_idx in range(4):
            for b_idx in range(a_idx + 1, 4):
                a = pts[simplex[a_idx]]
                b = pts[simplex[b_idx]]
                ax3.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                         color="gray", lw=0.3, alpha=0.4)
    ax3.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                c="black", s=8, zorder=5)
    ax3.scatter(CUBE_CORNERS[:, 0], CUBE_CORNERS[:, 1], CUBE_CORNERS[:, 2],
                c="green", s=50, marker="s", zorder=6, edgecolors="black")
    ax3.set_title(f"With 8 cube-corner dummies\n"
                  f"({report['n_tetrahedra_with_dummies']} tets, "
                  f"{report['n_real_interfaces_with_dummies']} real+"
                  f"{report['n_boundary_interfaces_with_dummies']} bdy interfaces)")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    _set_axes_equal(ax3, np.vstack([centers, CUBE_CORNERS]))

    # Histogram of saddle clearances vs Voronoi
    ax4 = fig2.add_subplot(1, 2, 2)
    voronoi_ref = report["voronoi_saddle_median"]
    ax4.hist(saddle_clearances, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax4.axvline(voronoi_ref, color="orange", lw=2, linestyle="--",
                label=f"Voronoi median ({voronoi_ref})")
    delaunay_median = report["saddle_clearance_median"]
    ax4.axvline(delaunay_median, color="red", lw=2, linestyle="-",
                label=f"Delaunay median ({delaunay_median:.4f})")
    ax4.set_xlabel("saddle clearance")
    ax4.set_ylabel("interface count")
    ax4.set_title(f"Saddle clearance distribution ({len(interfaces)} interfaces)\n"
                  f"Delaunay median={delaunay_median:.4f}  "
                  f"Voronoi median={voronoi_ref}")
    ax4.legend()

    fig2.tight_layout()
    fig2.savefig(out_dir / "delaunay_dummies_and_saddles.png", dpi=150)
    plt.close(fig2)
    print("Saved delaunay_dummies_and_saddles.png")

    # --- Figure 3: 3D spheres + selected interfaces (semi-transparent spheres) ---
    fig3 = plt.figure(figsize=(14, 12))
    ax5 = fig3.add_subplot(1, 1, 1, projection="3d")
    _draw_cube(ax5)

    # Draw a few representative spheres as wireframes
    _draw_spheres_wireframe(ax5, centers, radii, n_theta=16, n_phi=8)

    # Show clean interfaces
    for idx, iface in enumerate(interfaces):
        fv = iface["face_verts"]
        tri_pts = np.asarray([centers[fv[0]], centers[fv[1]], centers[fv[2]]])
        if iface["no_channel"]:
            color, alpha = "#e74c3c", 0.8
        elif iface["penetrated"]:
            color, alpha = "#f39c12", 0.6
        else:
            color, alpha = "#3498db", 0.3
        ax5.plot_trisurf(tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2],
                         color=color, alpha=alpha, shade=True)

    # Saddle points (max clearance on each interface)
    saddle_pts = report["_max_clearance_pts"]
    ax5.scatter(saddle_pts[:, 0], saddle_pts[:, 1], saddle_pts[:, 2],
                c="darkgreen", s=3, alpha=0.6)

    ax5.set_title(f"Spheres + Delaunay interfaces + saddle points\n"
                  f"{len(interfaces)} interfaces, {report['n_tetrahedra']} tets")
    ax5.set_xlabel("x"); ax5.set_ylabel("y"); ax5.set_zlabel("z")
    _set_axes_equal(ax5, centers)

    fig3.tight_layout()
    fig3.savefig(out_dir / "delaunay_spheres_interfaces.png", dpi=150)
    plt.close(fig3)
    print("Saved delaunay_spheres_interfaces.png")


def _draw_cube(ax: plt.Axes) -> None:
    """Wireframe of the unit cube."""
    edges = [
        ((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 0, 1)),
        ((1, 1, 0), (1, 0, 0)), ((1, 1, 0), (0, 1, 0)), ((1, 1, 0), (1, 1, 1)),
        ((1, 0, 1), (1, 0, 0)), ((1, 0, 1), (0, 0, 1)), ((1, 0, 1), (1, 1, 1)),
        ((0, 1, 1), (0, 1, 0)), ((0, 1, 1), (0, 0, 1)), ((0, 1, 1), (1, 1, 1)),
    ]
    for (x0, y0, z0), (x1, y1, z1) in edges:
        ax.plot([x0, x1], [y0, y1], [z0, z1], color="black", lw=0.5, alpha=0.4)


def _draw_spheres_wireframe(ax, centers, radii, n_theta=16, n_phi=8):
    """Lightweight sphere wireframes."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    for (cx, cy, cz), r in zip(centers, radii):
        # equator
        x_eq = cx + r * np.cos(theta)
        y_eq = cy + r * np.sin(theta)
        z_eq = np.full_like(theta, cz)
        ax.plot(x_eq, y_eq, z_eq, color="gray", lw=0.3, alpha=0.3)
        # meridian
        for p in phi[1:-1]:
            rr = r * np.sin(p)
            zz = cz + r * np.cos(p)
            x_mer = cx + rr * np.cos(theta)
            y_mer = cy + rr * np.sin(theta)
            z_mer = np.full_like(theta, zz)
            ax.plot(x_mer, y_mer, z_mer, color="gray", lw=0.2, alpha=0.2)


def _set_axes_equal(ax: plt.Axes, pts: np.ndarray) -> None:
    """Equal aspect ratio across all three axes."""
    mid = pts.mean(axis=0)
    max_range = 0.5 * (pts.max(axis=0) - pts.min(axis=0)).max() * 1.1
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Delaunay tetrahedron feasibility check")
    print(f"output dir: {OUT_DIR}")
    print("=" * 60)

    report = run_delaunay_analysis()

    # Save report (strip internal ndarray keys for JSON)
    json_report = {k: v for k, v in report.items() if not k.startswith("_")}
    report_path = OUT_DIR / "delaunay_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(json_report, fh, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\nSaved {report_path}")

    # Plots
    plot_delaunay_partition(report, OUT_DIR)

    # Summary markdown
    _write_summary(report)

    print("\nDone.")


def _write_summary(report: dict) -> None:
    lines = [
        "# Delaunay tetrahedron partition — feasibility check",
        "",
        f"**Sphere centres**: {report['n_spheres']} (18 boundary + 9 interior)",
        f"**Delaunay tetrahedra**: {report['n_tetrahedra']}",
        f"**Shared interfaces**: {report['n_interfaces']}",
        "",
        "## Convex hull coverage",
        f"- Cube corners inside hull: **{report['corners_inside_hull']}/8**",
        f"- With 8 corner dummies: {report['n_tetrahedra_with_dummies']} tets, "
        f"{report['n_real_interfaces_with_dummies']} real–real + "
        f"{report['n_boundary_interfaces_with_dummies']} real–dummy interfaces",
        "",
        "## Interface pitfalls",
        f"- 4th-sphere penetrated: **{report['penetrated_count']}/{report['n_interfaces']}**",
        f"- No fluid channel (3-disc cover): **{report['no_channel_count']}/{report['n_interfaces']}**",
        "",
        "## Saddle clearance",
        f"- min = {report['saddle_clearance_min']:.6f}",
        f"- **median = {report['saddle_clearance_median']:.6f}**",
        f"- max = {report['saddle_clearance_max']:.6f}",
        f"- Voronoi reference median = {report['voronoi_saddle_median']}",
        "",
        "## Centroid clearance (quick diagnostic)",
        f"- min = {report['centroid_clearance_min']:.6f}",
        f"- median = {report['centroid_clearance_median']:.6f}",
        f"- max = {report['centroid_clearance_max']:.6f}",
        "",
        "## Saddle clearance histogram bins",
    ]
    bins = report["saddle_histogram_bins"]
    counts = report["saddle_histogram_counts"]
    lines.append("| bin | count |")
    lines.append("|---|---|")
    for i in range(len(bins) - 1):
        lines.append(f"| [{bins[i]:.3f}, {bins[i+1]:.3f}) | {counts[i]} |")

    lines += [
        "",
        "## Decision",
    ]
    if report["no_channel_count"] > 0:
        lines.append(f"- ⚠️ {report['no_channel_count']} interfaces have **no fluid channel** — "
                     "faces completely blocked by their 3 vertex discs.")
    if report["penetrated_count"] > 0:
        lines.append(f"- ⚠️ {report['penetrated_count']} interfaces are **penetrated by a 4th sphere** — "
                     "the face plane cuts through an uninvolved sphere, creating a hole.")
    if report["corners_inside_hull"] < 8:
        lines.append(f"- ⚠️ {8 - report['corners_inside_hull']} cube corners are **outside the convex hull** — "
                     "need dummy corner points or shell closure.")

    lines.append(f"- Saddle clearance median {report['saddle_clearance_median']:.4f} "
                 f"vs Voronoi {report['voronoi_saddle_median']}")

    md_path = OUT_DIR / "delaunay_summary.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()

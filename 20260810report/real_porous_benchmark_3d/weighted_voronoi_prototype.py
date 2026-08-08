"""Weighted Voronoi (Laguerre / Power Diagram) — pure numpy prototype.

1. Generate dense bimodal sphere packing (80–120 spheres, porosity ~85%)
2. 4-D convex hull → regular triangulation → weighted Voronoi interfaces
3. Visualise interfaces + saddle clearance distribution

No FEniCSx / Gmsh — pure numpy + scipy + matplotlib.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "weighted_voronoi"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# 1. Generate dense bimodal sphere packing
# ===================================================================

def generate_dense_packing(
    seed: int = 20260806,
    target: int = 120,
    r_small: tuple[float, float] = (0.040, 0.055),
    r_large: tuple[float, float] = (0.065, 0.085),
    large_fraction: float = 0.30,
    min_gap: float = 0.012,
    boundary_count: int = 24,
) -> np.ndarray:
    """Rejection-sampled dense bimodal packing.

    Returns (n, 4) array: (x, y, z, r).  Boundary spheres extend outside [0,1]³.
    """
    rng = np.random.default_rng(seed)
    spheres: list[tuple[float, float, float, float]] = []

    def _mg(c: np.ndarray, r: float) -> float:
        best = 1e9
        for sx, sy, sz, sr in spheres:
            d = float(np.linalg.norm(c - np.array([sx, sy, sz])))
            best = min(best, d - r - sr)
        return best

    # --- Boundary spheres ---
    faces = [(0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0), (2, -1.0), (2, 1.0)]
    placed = 0
    tries = 0
    while placed < boundary_count and tries < 300_000:
        tries += 1
        axis, sign = faces[placed // (boundary_count // 6)]
        center = rng.uniform(0.12, 0.88, size=3)
        if rng.random() < large_fraction:
            radius = float(rng.uniform(*r_large))
        else:
            radius = float(rng.uniform(*r_small))
        offset = rng.uniform(0.020, 0.050)
        center[axis] = (0.0 if sign < 0 else 1.0) + sign * offset
        if _mg(center, radius) < min_gap:
            continue
        spheres.append((float(center[0]), float(center[1]), float(center[2]), radius))
        placed += 1

    # --- Interior spheres ---
    placed = 0
    tries = 0
    interior_target = target - boundary_count
    while placed < interior_target and tries < 500_000:
        tries += 1
        if rng.random() < large_fraction:
            radius = float(rng.uniform(*r_large))
        else:
            radius = float(rng.uniform(*r_small))
        center = rng.uniform(0.08, 0.92, size=3)
        if _mg(center, radius) < min_gap:
            continue
        spheres.append((float(center[0]), float(center[1]), float(center[2]), radius))
        placed += 1

    table = np.asarray(spheres, dtype=float)
    return table


# ===================================================================
# 2. Weighted Voronoi via 4-D convex hull
# ===================================================================

def weighted_voronoi_faces(spheres: np.ndarray) -> dict:
    """Extract weighted Voronoi interfaces from 4-D Delaunay triangulation.

    Algorithm (Aurenhammer 1987):
      1. Lift (x, y, z, r) → (x, y, z, w = x²+y²+z²−r²) in ℝ⁴
      2. Delaunay triangulation of the lifted 4-D points
         → this is the REGULAR (weighted Delaunay) triangulation
      3. Each 4-simplex has 5 vertices. Adjacent simplices share a 3-D facet.
         The 1 vertex unique to each simplex = (i, j) pair that shares a
         weighted Voronoi face.
      4. Interface plane: power bisector of (i, j)

    Returns dict with keys: pairs, face_normals, face_centers, face_displacements.
    """
    n = len(spheres)
    centers = spheres[:, :3]
    radii = spheres[:, 3]
    w = np.sum(centers**2, axis=1) - radii**2

    lifted = np.column_stack([centers, w])  # (n, 4)

    # 4-D Delaunay → regular triangulation
    tri = Delaunay(lifted)
    n_simplices = len(tri.simplices)
    print(f"  Delaunay: {n_simplices} 4-simplices (each has 5 vertices)")

    # tri.simplices shape: (n_simplices, 5)
    # tri.neighbors shape: (n_simplices, 5)
    # neighbor j of simplex i is opposite vertex j of simplex i

    pairs_set: set[tuple[int, int]] = set()
    for i in range(n_simplices):
        si = set(int(v) for v in tri.simplices[i])
        for j in range(5):
            nbr_idx = tri.neighbors[i, j]
            if nbr_idx == -1:
                continue  # convex hull boundary face — skip
            if i >= nbr_idx:
                continue  # deduplicate

            sn = set(int(v) for v in tri.simplices[nbr_idx])
            # The two simplices share the 4 vertices opposite their differing vertices
            # Vertex j of simplex i is unique to i; the corresponding unique vertex
            # in simplex nbr is whatever's not in si ∩ sn
            common = si & sn
            if len(common) != 4:
                continue  # should always be 4 for 4D Delaunay

            diff_i = si - common  # vertex unique to simplex i
            diff_n = sn - common  # vertex unique to simplex nbr

            if len(diff_i) != 1 or len(diff_n) != 1:
                continue
            a, b = int(diff_i.pop()), int(diff_n.pop())
            if a >= n or b >= n:
                continue
            pairs_set.add((min(a, b), max(a, b)))

    pairs_raw = sorted(pairs_set)
    print(f"  Raw Delaunay-adjacent pairs: {len(pairs_raw)}")

    # Filter: keep only pairs where the bisector centre really belongs to
    # the weighted Voronoi face, i.e. i and j are the two closest spheres in
    # power distance at the bisector centre.
    w_sq = np.sum(centers**2, axis=1) - radii**2

    def _closest_two_power(pt: np.ndarray) -> tuple[int, int]:
        scores = 2.0 * pt @ centers.T - w_sq
        order = np.argsort(-scores)  # descending
        return int(order[0]), int(order[1])

    pairs_filtered = []
    for a, b in pairs_raw:
        ci, cj = centers[a], centers[b]
        ri, rj = radii[a], radii[b]
        delta = cj - ci
        dist = float(np.linalg.norm(delta))
        if dist < 1e-10:
            continue
        unit = delta / dist
        shift = (ri**2 - rj**2) / (2.0 * dist)
        fc = 0.5 * (ci + cj) + shift * unit
        k1, k2 = _closest_two_power(fc)
        if {k1, k2} == {a, b}:
            pairs_filtered.append((a, b))

    pairs = pairs_filtered
    print(f"  After filtering (bisector ownership): {len(pairs)}")

    # For each pair, compute the power bisector plane
    face_normals = np.zeros((len(pairs), 3))
    face_centers = np.zeros((len(pairs), 3))
    face_displacements = np.zeros(len(pairs))  # shift from standard Voronoi midpoint

    for idx, (i, j) in enumerate(pairs):
        ci, cj = centers[i], centers[j]
        ri, rj = radii[i], radii[j]
        delta = cj - ci
        dist = float(np.linalg.norm(delta))
        unit = delta / dist if dist > 0 else np.array([1.0, 0.0, 0.0])

        # Standard Voronoi midpoint
        voronoi_mid = 0.5 * (ci + cj)
        # Power diagram shift
        shift = (ri**2 - rj**2) / (2.0 * dist)
        # Weighted Voronoi face center (point on the bisector plane)
        wv_center = voronoi_mid + shift * unit

        face_normals[idx] = unit
        face_centers[idx] = wv_center
        face_displacements[idx] = shift

    return {
        "pairs": pairs,
        "normals": face_normals,
        "centers": face_centers,
        "displacements": face_displacements,
        "n_simplices": n_simplices,
        "n_spheres": n,
        "_tri": tri,
    }


# ===================================================================
# 3. Saddle clearance on each face
# ===================================================================

def face_saddle_clearance(
    face_center: np.ndarray,
    face_normal: np.ndarray,
    spheres: np.ndarray,
) -> float:
    """Maximum clearance on the weighted Voronoi face plane (all spheres).

    Samples a disc around face_center, returns max clearance.
    This is the "throat size" — how wide the passage is at this interface.
    """
    centers = spheres[:, :3]
    radii = spheres[:, 3]

    # Sample points on the plane in a disc
    # Build two tangent vectors
    if abs(face_normal[2]) < 0.9:
        t1 = np.cross(face_normal, [0, 0, 1])
    else:
        t1 = np.cross(face_normal, [1, 0, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(face_normal, t1)

    n_r = 15; n_theta = 20
    samples = []
    for ir in range(n_r):
        r = (ir + 1) / n_r * 0.25  # up to radius 0.25
        for it in range(n_theta):
            theta = 2 * np.pi * it / n_theta
            pt = face_center + r * (np.cos(theta) * t1 + np.sin(theta) * t2)
            samples.append(pt)
    pts = np.asarray(samples)

    clearance = np.full(len(pts), np.inf)
    for c, r in zip(centers, radii):
        clearance = np.minimum(clearance, np.linalg.norm(pts - c[None, :], axis=1) - r)
    return float(np.max(clearance))


# ===================================================================
# 4. Cell-labelling via half-space membership
# ===================================================================

def label_cells_by_power_diagram(
    sample_pts: np.ndarray,
    spheres: np.ndarray,
    wv_result: dict,
) -> np.ndarray:
    """Assign each sample point to its weighted Voronoi cell.

    Cell of sphere k = {x : |x−c_k|²−r_k² ≤ |x−c_i|²−r_i² for all i}.
    This is the power distance comparison.
    """
    n_pts = len(sample_pts)
    n_spheres = len(spheres)
    centers = spheres[:, :3]
    radii = spheres[:, 3]
    w_sq = np.sum(centers**2, axis=1) - radii**2  # the 4th coordinate

    # For each point, find argmin of power distance:
    # power_dist(x, sphere_k) = |x−c_k|² − r_k²
    # = |x|² − 2⟨x,c_k⟩ + |c_k|² − r_k²
    # = |x|² − 2⟨x,c_k⟩ + w_k
    # argmin_k power_dist = argmax_k (2⟨x,c_k⟩ − w_k)
    # since |x|² is common.

    labels = np.full(n_pts, -1, dtype=np.int32)
    batch_size = 5000
    for start in range(0, n_pts, batch_size):
        end = min(start + batch_size, n_pts)
        batch = sample_pts[start:end]
        # scores[b, k] = 2⟨x_b, c_k⟩ − w_k
        scores = 2.0 * batch @ centers.T - w_sq[None, :]
        labels[start:end] = np.argmax(scores, axis=1).astype(np.int32)

    return labels


# ===================================================================
# 5. Visualisation
# ===================================================================

def visualise(spheres: np.ndarray, wv_result: dict, saddles: np.ndarray) -> None:
    """3-D plot: spheres + weighted Voronoi interfaces + saddle points."""
    centers = spheres[:, :3]
    radii = spheres[:, 3]
    pairs = wv_result["pairs"]
    normals = wv_result["normals"]
    face_centers = wv_result["centers"]
    displacements = wv_result["displacements"]

    # ── Figure 1: 3-D interfaces ──
    fig = plt.figure(figsize=(16, 13))

    ax1 = fig.add_subplot(2, 2, (1, 2), projection="3d")
    _draw_cube(ax1)

    # Draw interface faces as small squares
    for idx, (i, j) in enumerate(pairs):
        fc = face_centers[idx]
        fn = normals[idx]
        # Build tangent vectors
        if abs(fn[2]) < 0.9:
            t1 = np.cross(fn, [0, 0, 1])
        else:
            t1 = np.cross(fn, [1, 0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(fn, t1)

        # Small square on the plane
        half = 0.06
        corners = np.array([
            fc - half*t1 - half*t2,
            fc + half*t1 - half*t2,
            fc + half*t1 + half*t2,
            fc - half*t1 + half*t2,
        ])
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        tri1 = [corners[0], corners[1], corners[2]]
        tri2 = [corners[0], corners[2], corners[3]]
        color = plt.cm.turbo(idx / len(pairs))
        ax1.add_collection3d(Poly3DCollection([tri1, tri2], alpha=0.5,
                              facecolor=color, edgecolor="black", linewidth=0.2))

    # Draw sphere wireframes (subset for clarity)
    for k in range(len(spheres)):
        if k % 3 == 0:  # every 3rd sphere
            _draw_sphere_wireframe(ax1, centers[k], radii[k])

    ax1.set_title(f"Weighted Voronoi — {len(pairs)} interfaces\n"
                  f"{len(spheres)} spheres, bimodal radii")
    _set_axes_equal_3d(ax1)

    # ── Figure 1b: Saddle clearance histogram ──
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.hist(saddles, bins=30, color="steelblue", edgecolor="white")
    ax2.axvline(np.median(saddles), color="red", lw=2, linestyle="--",
                label=f"median = {np.median(saddles):.4f}")
    ax2.set_xlabel("saddle clearance")
    ax2.set_ylabel("interface count")
    ax2.set_title("Saddle clearance distribution")
    ax2.legend()

    # ── Figure 1c: Displacement from standard Voronoi ──
    ax3 = fig.add_subplot(2, 2, 4)
    abs_disp = np.abs(displacements)
    ax3.hist(abs_disp, bins=30, color="darkorange", edgecolor="white")
    ax3.axvline(np.median(abs_disp), color="red", lw=2, linestyle="--",
                label=f"median = {np.median(abs_disp):.4f}")
    ax3.set_xlabel("|displacement| from Voronoi midpoint")
    ax3.set_ylabel("interface count")
    ax3.set_title("Weight shift vs standard Voronoi")
    ax3.legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "weighted_voronoi_overview.png", dpi=200)
    plt.close(fig)
    print(f"  Saved weighted_voronoi_overview.png")

    # ── Figure 2: Slice at z=0.5 with cell labels ──
    _plot_slice_labels(spheres, wv_result, saddles)


def _plot_slice_labels(spheres, wv_result, saddles):
    """2-D slice at z=0.5 showing weighted Voronoi cell boundaries."""
    z_val = 0.5
    n_side = 150
    xs = np.linspace(0, 1, n_side)
    ys = np.linspace(0, 1, n_side)
    X, Y = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([X.ravel(), Y.ravel(), np.full(n_side*n_side, z_val)])

    labels = label_cells_by_power_diagram(grid_pts, spheres, wv_result)
    label_grid = labels.reshape(n_side, n_side)

    fig, ax = plt.subplots(figsize=(10, 9))
    # Use a qualitative colormap
    n_labels = int(labels.max()) + 1
    cmap = plt.cm.tab20 if n_labels <= 20 else plt.cm.gist_ncar
    im = ax.pcolormesh(X, Y, label_grid, cmap=cmap, shading="auto", alpha=0.8)

    # Draw sphere cross-sections
    for (cx, cy, cz, r) in spheres:
        dz = abs(cz - z_val)
        if dz < r:
            r_cut = np.sqrt(max(0, r**2 - dz**2))
            circle = plt.Circle((cx, cy), r_cut, fill=True,
                               color="white", edgecolor="black", linewidth=0.5)
            ax.add_patch(circle)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"z = {z_val} slice — weighted Voronoi cells\n"
                 f"{n_labels} cells, {len(spheres)} spheres")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "weighted_voronoi_slice.png", dpi=200)
    plt.close(fig)
    print(f"  Saved weighted_voronoi_slice.png")


# ===================================================================
# Helpers
# ===================================================================

def _draw_cube(ax):
    edges = [
        ((0,0,0),(1,0,0)),((0,0,0),(0,1,0)),((0,0,0),(0,0,1)),
        ((1,1,0),(1,0,0)),((1,1,0),(0,1,0)),((1,1,0),(1,1,1)),
        ((1,0,1),(1,0,0)),((1,0,1),(0,0,1)),((1,0,1),(1,1,1)),
        ((0,1,1),(0,1,0)),((0,1,1),(0,0,1)),((0,1,1),(1,1,1)),
    ]
    for (x0,y0,z0),(x1,y1,z1) in edges:
        ax.plot([x0,x1],[y0,y1],[z0,z1],color="black",lw=0.5,alpha=0.3)


def _draw_sphere_wireframe(ax, center, radius, n_th=12, n_ph=6):
    th = np.linspace(0, 2*np.pi, n_th)
    ph = np.linspace(0, np.pi, n_ph)
    cx, cy, cz = center
    # equator
    ax.plot(cx+radius*np.cos(th), cy+radius*np.sin(th), np.full_like(th,cz),
            color="gray", lw=0.3, alpha=0.3)
    # meridians
    for p in ph[1:-1]:
        rr = radius * np.sin(p); zz = cz + radius * np.cos(p)
        ax.plot(cx+rr*np.cos(th), cy+rr*np.sin(th), np.full_like(th,zz),
                color="gray", lw=0.2, alpha=0.2)


def _set_axes_equal_3d(ax):
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1); ax.set_zlim(-0.1, 1.1)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("Weighted Voronoi (Laguerre / Power Diagram) prototype")
    print("=" * 60)

    # 1. Dense packing
    print("\n--- 1. Generating dense bimodal packing ---")
    spheres = generate_dense_packing(seed=20260806)
    n = len(spheres)
    v_solid = float(np.sum(4.0 / 3.0 * np.pi * spheres[:, 3]**3))
    porosity = 1.0 - v_solid
    small_mask = spheres[:, 3] < 0.05
    print(f"  Spheres: {n} ({small_mask.sum()} small + {(~small_mask).sum()} large)")
    print(f"  Radii: [{spheres[:,3].min():.4f}, {spheres[:,3].max():.4f}]")
    print(f"  Porosity: {porosity*100:.1f}%")
    print(f"  r_max - r_min = {spheres[:,3].max()-spheres[:,3].min():.4f}")

    # 2. Weighted Voronoi
    print("\n--- 2. Computing weighted Voronoi (4-D convex hull) ---")
    wv = weighted_voronoi_faces(spheres)

    # 3. Saddle clearance analysis
    print("\n--- 3. Saddle clearance analysis ---")
    saddles = np.array([
        face_saddle_clearance(wv["centers"][i], wv["normals"][i], spheres)
        for i in range(len(wv["pairs"]))
    ])
    abs_disp = np.abs(wv["displacements"])
    print(f"  Saddle clearance: min={saddles.min():.6f}  "
          f"median={np.median(saddles):.6f}  max={saddles.max():.6f}")
    print(f"  Displacement |shift|: min={abs_disp.min():.6f}  "
          f"median={np.median(abs_disp):.6f}  max={abs_disp.max():.6f}")

    # 4. Visualise
    print("\n--- 4. Visualisation ---")
    visualise(spheres, wv, saddles)

    # 5. Save data
    print("\n--- 5. Saving ---")
    np.savez(OUT_DIR / "weighted_voronoi_data.npz",
             sphere_centers=spheres[:, :3], sphere_radii=spheres[:, 3],
             pairs=np.array(wv["pairs"]),
             face_centers=wv["centers"], face_normals=wv["normals"],
             displacements=wv["displacements"], saddles=saddles)

    summary = {
        "n_spheres": int(n),
        "porosity": float(porosity),
        "r_min": float(spheres[:, 3].min()),
        "r_max": float(spheres[:, 3].max()),
        "n_interfaces": int(len(wv["pairs"])),
        "n_4simplices": int(wv["n_simplices"]),
        "saddle_median": float(np.median(saddles)),
        "saddle_min": float(saddles.min()),
        "saddle_max": float(saddles.max()),
        "displacement_median": float(np.median(abs_disp)),
        "displacement_max": float(abs_disp.max()),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved {OUT_DIR}/weighted_voronoi_data.npz")
    print(f"  Saved {OUT_DIR}/summary.json")
    print("\nDone.")


if __name__ == "__main__":
    main()

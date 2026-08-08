"""One-shot generator for the frozen random-sphere realization.

Mirrors ddpnm_2d_random_porous: a fixed-seed random packing with boundary
particles deliberately extending outside the unit cube (clipped by the
sample window) plus interior particles.  The produced table is frozen into
``random_porous.py`` so every run uses the exact same geometry.

Radius range and minimum gap obey  r_max - r_min <= gap/2, which guarantees
that every Voronoi throat face of the packing stays clear of every sphere
(face points are at least d(i,j)/2 away from any centre).
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(20260804)

spheres: list[tuple[float, float, float, float]] = []
MIN_GAP = 0.050  # global minimum gap (boundary-boundary pairs included)


def min_gap(candidate: np.ndarray, radius: float) -> float:
    best = 1.0e9
    for x, y, z, r in spheres:
        d = float(np.linalg.norm(candidate - np.asarray([x, y, z])))
        best = min(best, d - radius - r)
    return best


# --- 18 boundary spheres: centered outside one cube face, clipped by window ---
FACES = [  # (axis, sign)
    (0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0), (2, -1.0), (2, 1.0),
]
placed = 0
tries = 0
while placed < 18 and tries < 400000:
    tries += 1
    axis, sign = FACES[placed // 3]
    center = RNG.uniform(0.20, 0.80, size=3)
    radius = float(RNG.uniform(0.115, 0.125))
    center[axis] = (0.0 if sign < 0 else 1.0) + sign * RNG.uniform(0.020, 0.045)
    if min_gap(center, radius) < MIN_GAP:
        continue
    spheres.append((float(center[0]), float(center[1]), float(center[2]), radius))
    placed += 1
if placed < 18:
    raise RuntimeError(f"Only placed {placed} boundary spheres after {tries} tries.")


# --- 9 interior spheres: rejection sampling with the same minimum gap ---
placed = 0
tries = 0
while placed < 9 and tries < 200000:
    tries += 1
    radius = float(RNG.uniform(0.105, 0.115))
    center = RNG.uniform(0.13, 0.87, size=3)
    if min_gap(center, radius) < MIN_GAP:
        continue
    spheres.append((float(center[0]), float(center[1]), float(center[2]), radius))
    placed += 1
if placed < 9:
    raise RuntimeError(f"Only placed {placed} interior spheres after {tries} tries.")

table = np.asarray(spheres, dtype=float)
print(f"n_spheres={len(table)}")
print(f"interior_r=[{table[18:, 3].min():.4f}, {table[18:, 3].max():.4f}]")
print(f"boundary_r=[{table[:18, 3].min():.4f}, {table[:18, 3].max():.4f}]")
gaps: list[float] = []
for i in range(len(table)):
    for j in range(i + 1, len(table)):
        d = float(np.linalg.norm(table[i, :3] - table[j, :3]))
        gaps.append(d - table[i, 3] - table[j, 3])
gaps = np.asarray(gaps)
print(f"min pairwise gap = {gaps.min():.4f}   (need >= {MIN_GAP})")
print(f"r_max - r_min = {table[:, 3].max() - table[:, 3].min():.4f}   (need <= {MIN_GAP/2:.4f})")
inside = np.all((table[:, :3] >= -0.01) & (table[:, :3] <= 1.01), axis=1)
print(f"spheres with center inside cube: {inside.sum()} / {len(table)}")

print("\nSPHERES = np.asarray([")
for x, y, z, r in table:
    print(f"    ({x:.12f}, {y:.12f}, {z:.12f}, {r:.12f}),")
print("], dtype=float)")

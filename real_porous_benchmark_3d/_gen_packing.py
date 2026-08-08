"""Generate a 50-sphere uniform-radius packing — Voronoi-safe.

r ∈ [0.050, 0.058] → r_max − r_min ≤ 0.008 ≤ gap/2 = 0.015
All three partitions (Voronoi, watershed, grid) guaranteed valid.
"""

import numpy as np

RNG = np.random.default_rng(20260806)
MIN_GAP = 0.030
RADIUS_RANGE = (0.050, 0.058)
TARGET = 50
BOUNDARY = 18  # 3 per face

spheres = []


def _min_gap(candidate, radius):
    best = 1e9
    for x, y, z, r in spheres:
        d = float(np.linalg.norm(candidate - np.asarray([x, y, z])))
        best = min(best, d - radius - r)
    return best


FACES = [(0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0), (2, -1.0), (2, 1.0)]
placed = 0
tries = 0
while placed < BOUNDARY and tries < 200_000:
    tries += 1
    axis, sign = FACES[placed // (BOUNDARY // 6)]
    center = RNG.uniform(0.15, 0.85, size=3)
    radius = float(RNG.uniform(*RADIUS_RANGE))
    offset = RNG.uniform(0.025, 0.050)
    center[axis] = (0.0 if sign < 0 else 1.0) + sign * offset
    if _min_gap(center, radius) < MIN_GAP:
        continue
    spheres.append((float(center[0]), float(center[1]), float(center[2]), radius))
    placed += 1
print(f"Boundary: {placed}/{BOUNDARY} ({tries} tries)")

placed = 0
tries = 0
while placed < TARGET - BOUNDARY and tries < 300_000:
    tries += 1
    radius = float(RNG.uniform(*RADIUS_RANGE))
    center = RNG.uniform(0.10, 0.90, size=3)
    if _min_gap(center, radius) < MIN_GAP:
        continue
    spheres.append((float(center[0]), float(center[1]), float(center[2]), radius))
    placed += 1
print(f"Interior: {placed}/{TARGET - BOUNDARY} ({tries} tries)")

table = np.asarray(spheres, dtype=float)
n = len(table)
v = float(np.sum(4.0 / 3.0 * np.pi * table[:, 3] ** 3))
por = 1.0 - v

gaps = [float(np.linalg.norm(table[i,:3]-table[j,:3])) - table[i,3] - table[j,3]
        for i in range(n) for j in range(i+1,n)]
gaps_arr = np.asarray(gaps)

print(f"\nTotal: {n} spheres  |  solid={v:.4f}  |  porosity={por*100:.1f}%")
print(f"Radii: [{table[:,3].min():.4f}, {table[:,3].max():.4f}]")
print(f"r_max−r_min={table[:,3].max()-table[:,3].min():.4f} ≤ {MIN_GAP/2:.4f}  {'✓' if table[:,3].max()-table[:,3].min()<=MIN_GAP/2 else '✗'}")
print(f"Min gap: {gaps_arr.min():.4f} ≥ {MIN_GAP}  {'✓' if gaps_arr.min()>=MIN_GAP else '✗'}")

print(f"\nSPHERES = np.asarray([")
for x, y, z, r in table:
    print(f"    ({x:.12f}, {y:.12f}, {z:.12f}, {r:.12f}),")
print(f"], dtype=float)")

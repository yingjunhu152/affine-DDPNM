"""Quick validation: run weighted-Voronoi partition and check correctness."""
import sys, time, numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ.parent))
sys.path.insert(0, str(PROJ))

from geometry import (
    build_partition_weighted_voronoi_occ, SPHERES, N_SPHERES,
    _weighted_voronoi_throat_faces,
)

print("=" * 60)
print("Weighted Voronoi partition validation")
print("=" * 60)

# ── 1. Face generation ──
t0 = time.perf_counter()
used_pairs, faces, throats = _weighted_voronoi_throat_faces()
t_face = time.perf_counter() - t0
print(f"\nFace generation: {t_face:.1f}s")
print(f"  Real-real pairs: {len(used_pairs)}")
print(f"  Total faces (incl. real-dummy): {len(faces)}")

# Check connectivity: which spheres appear in used_pairs
connected = set()
for i, j in used_pairs:
    connected.add(i)
    connected.add(j)
isolated = set(range(N_SPHERES)) - connected
print(f"  Connected spheres: {len(connected)}/{N_SPHERES}")
if isolated:
    print(f"  WARNING: Isolated spheres: {sorted(isolated)}")

# ── 2. Mesh generation (no FEM) ──
print("\n─ Mesh generation ─")
t0 = time.perf_counter()
try:
    p = build_partition_weighted_voronoi_occ(
        mesh_size=0.12, sphere_size=0.05, boundary_size=0.07,
        interface_size=0.06, sphere_band=0.14, boundary_band=0.12,
        interface_band=0.10,
    )
except Exception as e:
    print(f"FATAL: {e}")
    sys.exit(1)
t_mesh = time.perf_counter() - t0
mesh = p.mesh
nc = mesh.topology.index_map(mesh.topology.dim).size_local
ni = len(p.interface_pairs)
labels = np.asarray(p.cell_labels)
nr = len(set(int(l) for l in labels))

print(f"\nMesh: {nc} cells, {ni} interfaces, {nr} regions ({t_mesh:.1f}s)")

# ── 3. Check every sphere has at least one cell ──
present_labels = set(int(l) for l in labels)
missing = sorted(set(range(N_SPHERES)) - present_labels)
print(f"  Present labels: {len(present_labels)}/{N_SPHERES}")
if missing:
    print(f"  MISSING pore regions: {missing}")
else:
    print(f"  All {N_SPHERES} spheres have at least one cell ✓")

# ── 4. Interface pair coverage ──
mesh_pairs = set(p.interface_pairs)
weighted_pairs = set((min(a,b), max(a,b)) for a,b in used_pairs)
missing_pairs = weighted_pairs - mesh_pairs
extra_pairs = mesh_pairs - weighted_pairs
print(f"\n  Interface pairs: {len(mesh_pairs)} (expected ~{len(used_pairs)})")
if missing_pairs:
    print(f"  Missing from mesh: {len(missing_pairs)}")
if extra_pairs:
    print(f"  Extra in mesh: {len(extra_pairs)}")

# ── 5. Per-sphere interface counts ──
iface_count = {i: 0 for i in range(N_SPHERES)}
for a, b in mesh_pairs:
    iface_count[a] = iface_count.get(a, 0) + 1
    iface_count[b] = iface_count.get(b, 0) + 1
no_iface = [k for k, v in iface_count.items() if v == 0]
avg_iface = np.mean(list(iface_count.values()))
print(f"\n  Avg interfaces/sphere: {avg_iface:.1f}")
if no_iface:
    print(f"  Spheres with NO interfaces: {no_iface}")

print(f"\n{'OK' if not missing and not no_iface else 'ISSUES FOUND'}")

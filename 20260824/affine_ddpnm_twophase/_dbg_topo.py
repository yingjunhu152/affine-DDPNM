import sys
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
from random_porous import build_partition
from ddpnm_core.io import topology_arrays

# same parameters as run_affine_ddpnm_twophase.py
partition = build_partition(
    mesh_size=0.13, sphere_size=0.065, boundary_size=0.085, interface_size=0.075,
    sphere_band=0.15, boundary_band=0.13, interface_band=0.12,
    mesh_file=PROJECT_DIR / "outputs" / "benchmark_twophase" / "random_sphere_partition.msh",
)
msh = partition.mesh
points, tetra = topology_arrays(msh)
print(f"cells={len(tetra)}")

tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
msh.topology.create_connectivity(tdim, fdim)
msh.topology.create_connectivity(tdim, 0)
msh.topology.create_connectivity(fdim, 0)
n_cells = msh.topology.index_map(tdim).size_local
n_facets = msh.topology.index_map(fdim).size_local
print(f"n_cells={n_cells} n_facets={n_facets}")

f2c = msh.topology.connectivity(fdim, tdim)
f2v = msh.topology.connectivity(fdim, 0)
bad = []
for f in range(n_facets):
    cl = f2c.links(f)
    if len(cl) > 2:
        bad.append((f, "many", list(cl)))
    for c in cl:
        if c >= n_cells or c < 0:
            bad.append((f, "cell", int(c)))
    for v in f2v.links(f):
        if v >= msh.geometry.x.shape[0] or v < 0:
            bad.append((f, "vert", int(v)))
print("bad links:", bad[:10], "count", len(bad))

# connectivity arrays (compressed) max index
ca = f2c.array
print("f2c array shape:", ca.shape, "max:", ca.max() if len(ca) else None, ">= n_cells:", bool(len(ca) and ca.max() >= n_cells))
print("f2c offsets:", f2c.offsets[:6], "...", f2c.offsets[-3:])

print("geom points:", msh.geometry.x.shape)
print("topo verts:", msh.topology.index_map(0).size_local)
print("f2v max:", f2v.array.max() if len(f2v.array) else None)
# cell_to_facets sanity
c2f = msh.topology.connectivity(tdim, fdim)
ca2 = c2f.array
print("c2f shape:", ca2.shape, "max:", ca2.max() if len(ca2) else None, ">= n_facets:", bool(len(ca2) and ca2.max() >= n_facets))

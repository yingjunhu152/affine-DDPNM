"""Per-basin exterior composition of the watershed partition.

Counts, per basin, the exterior facets on solid walls (sphere walls + lateral
cube faces), on the open inlet/outlet, and the interface facets.  A basin with
zero solid-wall facets has no velocity Dirichlet boundary in the local Stokes
solve — its local operator is floating (rigid-mode nullspace) and the plain
splu solve produces the catastrophic G entries seen in the ablation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from dolfinx import mesh as dmesh

REPOSITORY_DIR = Path(__file__).resolve().parent.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from watershed_partition import build_partition_watershed


def main() -> None:
    print("Building the watershed partition ...", flush=True)
    partition = build_partition_watershed(
        bulk_size=0.13, sphere_size=0.065, boundary_size=0.085,
        sphere_band=0.15, boundary_band=0.13,
        policy="walls_and_spheres", abs_threshold=0.02, rel_threshold=0.05,
    )
    msh = partition.mesh
    labels = partition.cell_labels
    n_pores = int(labels.max()) + 1
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    f2c = msh.topology.connectivity(fdim, tdim)
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    exterior = dmesh.exterior_facet_indices(msh.topology)

    wall_count = np.zeros(n_pores, dtype=int)
    open_count = np.zeros(n_pores, dtype=int)  # inlet/outlet
    interface_count = np.zeros(n_pores, dtype=int)
    for facet in exterior:
        cells = f2c.links(int(facet))
        if len(cells) != 1:
            continue
        pore = int(labels[cells[0]])
        interface_id = int(partition.facet_interface_ids[facet])
        if interface_id >= 0:
            interface_count[pore] += 1
            continue
        verts = msh.geometry.x[f2v.links(int(facet)), :3]
        midpoint = verts.mean(axis=0)
        if midpoint[0] <= 1.0e-8 or midpoint[0] >= 1.0 - 1.0e-8:
            open_count[pore] += 1
        else:
            wall_count[pore] += 1

    floating = np.flatnonzero(wall_count == 0)
    print(f"basins={n_pores}, floating (no solid wall facets)={len(floating)}")
    print(f"floating basins: {floating.tolist()}")
    for pore in floating:
        cells = int(np.sum(labels == pore))
        print(
            f"  pore {pore}: cells={cells}, wall={wall_count[pore]}, "
            f"open={open_count[pore]}, interface_facets={interface_count[pore]}"
        )
    print("wall-facet counts of all basins (min/med/max):",
          wall_count.min(), np.median(wall_count), wall_count.max())
    print("DIAG DONE")


if __name__ == "__main__":
    main()

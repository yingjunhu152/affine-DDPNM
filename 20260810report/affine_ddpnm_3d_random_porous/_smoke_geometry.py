"""Smoke test for the random-sphere partition geometry (coarse mesh)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY_DIR = Path(__file__).resolve().parent.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from random_porous import (
    _polygon_area,
    analytic_throat_candidates,
    build_partition,
    clearance,
    voronoi_throat_faces,
)

candidates = analytic_throat_candidates()
print(f"Delaunay candidates after gap/third-sphere/window tests: {len(candidates)}")
pairs, faces, throats = voronoi_throat_faces()
print(f"valid throat faces: {len(faces)}")
areas = [_polygon_area(f) for f in faces]
print(f"face areas: min={min(areas):.5f} max={max(areas):.5f}")

partition = build_partition(
    mesh_size=0.14,
    sphere_size=0.07,
    boundary_size=0.09,
    interface_size=0.08,
    sphere_band=0.14,
    boundary_band=0.12,
    interface_band=0.11,
)
msh = partition.mesh
tdim = msh.topology.dim
n_cells = msh.topology.index_map(tdim).size_local
print(f"\nmesh: cells={n_cells}, interfaces={len(partition.interface_pairs)}")
print(f"cell labels: {len(np.unique(partition.cell_labels))} regions")
print(f"cad_counts: {partition.cad_counts}")
print(f"interface areas: min={partition.interface_areas.min():.5f} "
      f"max={partition.interface_areas.max():.5f}")
print(f"maximal balls: {len(partition.maximal_balls)}")
print("min interface clearance:",
      float(np.min(clearance(partition.interface_centers))))

if n_cells < 1000:
    raise SystemExit("mesh too coarse -> smoke test failed")
print("SMOKE OK")

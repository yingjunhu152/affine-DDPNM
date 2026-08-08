"""Diagnostic: why did the per-throat Hausdorff matching find zero facets?

Rebuilds the watershed facet list from the saved formal mesh, then for every
Voronoi interface pair reports how many watershed interface facets lie within
``margin`` of the face polygon (2-D polygon distance, not anchor distance) and
what their nearest-sphere pairs are — to decide whether the sphere-pair filter
over-restricts the correspondence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from dolfinx import mesh as dmesh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

REPOSITORY_DIR = Path(__file__).resolve().parent.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from random_porous import SPHERES, voronoi_throat_faces
from compare_partitions_formal import (
    polygon_frame,
    point_to_polygon_distance,
    clip_polygon_to_cube,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "watershed_formal"
MARGIN = 0.06


def main() -> None:
    print("Rebuilding the formal-mesh watershed partition ...")
    from watershed_partition import build_partition_watershed
    partition = build_partition_watershed(
        mesh_file=OUT_DIR / "watershed_formal_mesh.msh",
        policy="walls_and_spheres",
        abs_threshold=0.02,
        rel_threshold=0.05,
        bulk_size=0.10,
        sphere_size=0.040,
        boundary_size=0.055,
        sphere_band=0.12,
        boundary_band=0.10,
    )
    msh = partition.mesh
    labels = partition.cell_labels
    facet_ids = partition.facet_interface_ids
    cell_centers = partition.cell_centers
    nearest = np.argmin(
        np.linalg.norm(cell_centers[:, None, :] - SPHERES[None, :, :3], axis=2), axis=1
    )

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    msh.topology.create_connectivity(fdim, tdim)
    f2c = msh.topology.connectivity(fdim, tdim)
    centroids, pairs = [], []
    for facet in np.flatnonzero(facet_ids >= 0):
        verts = msh.geometry.x[f2v.links(int(facet)), :3]
        cells = f2c.links(int(facet))
        centroids.append(verts.mean(axis=0))
        pairs.append(tuple(sorted((int(nearest[cells[0]]), int(nearest[cells[1]])))))
    centroids = np.asarray(centroids, dtype=float)
    pairs = np.asarray(pairs, dtype=np.int64)
    print(f"interface facets: {len(centroids)}")

    used_pairs, faces, throats = voronoi_throat_faces()
    face_for_pair = {pair: i for i, pair in enumerate(used_pairs)}
    vor_fields = np.load(
        Path(__file__).resolve().parent / "outputs" / "benchmark" / "random_benchmark_fields.npz"
    )
    vor_pairs = [tuple(int(x) for x in pair) for pair in vor_fields["interface_pairs"]]
    print(f"voronoi pairs: {len(vor_pairs)}")

    from collections import Counter
    n_with_candidates = 0
    for pair in vor_pairs:
        face_index = face_for_pair.get(pair)
        if face_index is None:
            print(f"pair {pair}: no face"); continue
        polygon = clip_polygon_to_cube(faces[face_index])
        if len(polygon) < 3:
            print(f"pair {pair}: polygon empty after cube clip"); continue
        normal, u, v = polygon_frame(polygon)
        dist = point_to_polygon_distance(centroids, polygon, normal, u, v)
        cand = np.flatnonzero(dist <= MARGIN)
        if len(cand) == 0:
            print(f"pair {pair}: NO candidates (min dist {dist.min():.4f})")
            continue
        n_with_candidates += 1
        cnt = Counter(tuple(int(x) for x in pairs[c]) for c in cand)
        top = cnt.most_common(3)
        print(f"pair {pair}: {len(cand)} candidates, min dist {dist[cand].min():.4f}, "
              f"sphere-pairs {top} (match={cnt.get(pair, 0)})")
    print(f"pairs with candidates: {n_with_candidates}/{len(vor_pairs)}")


if __name__ == "__main__":
    main()

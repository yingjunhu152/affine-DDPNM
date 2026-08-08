#!/usr/bin/env python3
"""Compute geometry/discretization statistics for the three benchmark geometries.

Reads the saved MSH partitions (uniform, random) with meshio and rebuilds the
real-porous grid partition with the benchmark's own builder (no saved mesh).
Outputs a JSON consumed by the paper tables and figures.

Run with the fenicsx env:
    export PATH="/d/Miniconda3/envs/fenicsx:/d/Miniconda3/envs/fenicsx/Library/bin:..."
    python geometry_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)


def tetra_volume(a, b, c, d):
    return abs(np.dot(b - a, np.cross(c - a, d - a))) / 6.0


def stats_from_msh(msh_path: Path, name: str, known_mixed_dofs: int) -> dict:
    import meshio

    m = meshio.read(str(msh_path))
    points = np.asarray(m.points[:, :3], dtype=float)
    tris = np.asarray(m.cells_dict["triangle"], dtype=np.int64)
    # gmsh writes one cell block per physical entity; concatenate the tetra
    # blocks and their volume tags (tags are 1000 + region label)
    tets_parts, labels_parts = [], []
    for blk, phys in zip(m.cells, m.cell_data["gmsh:physical"]):
        if blk.type == "tetra":
            tets_parts.append(np.asarray(blk.data, dtype=np.int64))
            labels_parts.append(np.asarray(phys, dtype=np.int64).ravel())
    tets = np.vstack(tets_parts)
    labels = np.concatenate(labels_parts) - 1000

    n_cells = len(tets)
    n_vert = len(points)
    edges = np.unique(
        np.sort(tets[:, [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]].reshape(-1, 2), axis=1),
        axis=0,
    )
    n_edge = len(edges)
    vel_dofs = 3 * (n_vert + n_edge)  # P2 Lagrange on tets: vertex + edge nodes
    pres_dofs = n_vert                # P1
    mixed = vel_dofs + pres_dofs
    assert mixed == known_mixed_dofs, f"{name}: predicted {mixed} != reported {known_mixed_dofs}"

    # pore volumes
    vols = np.array([tetra_volume(*points[t]) for t in tets])
    pore_vols = np.zeros(int(labels.max()) + 1)
    np.add.at(pore_vols, labels, vols)

    # interfaces: facets shared by two differently labelled tets
    tris_sorted = np.sort(tris, axis=1)
    tri_keys = {tuple(map(int, t)): i for i, t in enumerate(tris_sorted)}
    tri_to_tets: dict[tuple, list] = {}
    for ci, t in enumerate(tets):
        for tri in (
            tuple(sorted(map(int, (t[0], t[1], t[2])))),
            tuple(sorted(map(int, (t[0], t[1], t[3])))),
            tuple(sorted(map(int, (t[0], t[2], t[3])))),
            tuple(sorted(map(int, (t[1], t[2], t[3])))),
        ):
            tri_to_tets.setdefault(tri, []).append(ci)

    areas_by_pair: dict[tuple, float] = {}
    verts_of_pair: dict[tuple, list] = {}
    for key, cs in tri_to_tets.items():
        if len(cs) != 2:
            continue
        a, b = int(labels[cs[0]]), int(labels[cs[1]])
        if a != b:
            tri = tris_sorted[tri_keys[key]]
            pa = 0.5 * float(
                np.linalg.norm(
                    np.cross(points[tri[1]] - points[tri[0]], points[tri[2]] - points[tri[0]])
                )
            )
            pair = (min(a, b), max(a, b))
            areas_by_pair[pair] = areas_by_pair.get(pair, 0.0) + pa
            verts_of_pair.setdefault(pair, []).extend(points[list(key)])
    pairs = sorted(areas_by_pair)
    iface_areas = np.array([areas_by_pair[p] for p in pairs])

    # aspect ratios via PCA on each interface patch's vertices: the two
    # largest eigenvalues span the tangent plane, so a/b = sqrt(lam2/lam1) >= 1
    aspect = []
    for pair in pairs:
        vv = np.asarray(verts_of_pair[pair])
        vv = vv - vv.mean(axis=0)
        cov = vv.T @ vv / len(vv)
        w = np.linalg.eigvalsh(cov)
        aspect.append(float(np.sqrt(w[2] / max(w[1], 1e-30))))
    aspect = np.asarray(aspect)

    return {
        "name": name,
        "cells": int(n_cells),
        "vertices": int(n_vert),
        "edges": int(n_edge),
        "fem_velocity_dofs": int(vel_dofs),
        "fem_pressure_dofs": int(pres_dofs),
        "fem_mixed_dofs": int(mixed),
        "n_pores": int(len(pore_vols)),
        "n_interfaces": int(len(pairs)),
        "mean_ports_per_pore": float(2 * len(pairs) / len(pore_vols)),
        "pore_volume_cv": float(np.std(pore_vols) / np.mean(pore_vols)),
        "interface_area_cv": float(np.std(iface_areas) / np.mean(iface_areas)),
        "interface_aspect_ratio_mean": float(np.mean(aspect)),
        "interface_aspect_ratio_cv": float(np.std(aspect) / np.mean(aspect)),
        "total_fluid_volume": float(pore_vols.sum()),
    }


def real_stats() -> dict:
    """Rebuild the real-porous grid partition (the benchmark saved no mesh)."""
    sys.path.insert(0, str(REPO / "real_porous_benchmark_3d"))
    from geometry import SPHERES, build_partition_grid
    from ddpnm_core.io import topology_vertex_coordinates

    partition = build_partition_grid(
        mesh_size=0.12, sphere_size=0.05, boundary_size=0.07,
        sphere_band=0.14, boundary_band=0.12,
    )
    msh = partition.mesh
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    n_cells = msh.topology.index_map(tdim).size_local
    tets = np.asarray([list(c2v.links(c)) for c in range(n_cells)], dtype=np.int64)
    points = topology_vertex_coordinates(msh)
    n_vert = len(points)
    edges = np.unique(
        np.sort(tets[:, [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]].reshape(-1, 2), axis=1),
        axis=0,
    )
    n_edge = len(edges)
    vel_dofs = 3 * (n_vert + n_edge)
    pres_dofs = n_vert
    labels = np.asarray(partition.cell_labels)
    vols = np.array([tetra_volume(*points[t]) for t in tets])
    pore_vols = np.zeros(int(labels.max()) + 1)
    np.add.at(pore_vols, labels, vols)
    iface_areas = np.asarray(partition.interface_areas)
    n_pairs = len(partition.interface_pairs)
    # aspect ratios via PCA on each interface patch's vertices
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    facet_ids = np.asarray(partition.facet_interface_ids)
    n_facets = msh.topology.index_map(fdim).size_local
    verts_of_pair = {p: [] for p in partition.interface_pairs}
    for facet in range(n_facets):
        iid = int(facet_ids[facet])
        if iid >= 0:
            verts_of_pair[partition.interface_pairs[iid]].extend(
                [x[:3] for x in msh.geometry.x[f2v.links(facet)]]
            )
    aspect = []
    for pair in partition.interface_pairs:
        vv = np.asarray(verts_of_pair[pair])
        vv = vv - vv.mean(axis=0)
        cov = vv.T @ vv / len(vv)
        w = np.linalg.eigvalsh(cov)
        aspect.append(float(np.sqrt(w[2] / max(w[1], 1e-30))))
    aspect = np.asarray(aspect)
    return {
        "name": "Real-100",
        "cells": int(n_cells),
        "vertices": int(n_vert),
        "edges": int(n_edge),
        "fem_velocity_dofs": int(vel_dofs),
        "fem_pressure_dofs": int(pres_dofs),
        "fem_mixed_dofs": int(vel_dofs + pres_dofs),
        "n_pores": int(len(pore_vols)),
        "n_interfaces": int(n_pairs),
        "mean_ports_per_pore": float(2 * n_pairs / len(pore_vols)),
        "pore_volume_cv": float(np.std(pore_vols) / np.mean(pore_vols)),
        "interface_area_cv": float(np.std(iface_areas) / np.mean(iface_areas)),
        "interface_aspect_ratio_mean": float(np.mean(aspect)),
        "interface_aspect_ratio_cv": float(np.std(aspect) / np.mean(aspect)),
        "total_fluid_volume": float(pore_vols.sum()),
        "sphere_count": int(len(SPHERES)),
    }


def main() -> None:
    uniform = stats_from_msh(
        REPO / "affine_ddpnm_3d/outputs/benchmark_w1n/affine_ddpnm_partition.msh",
        "Uniform-27", known_mixed_dofs=63955,
    )
    random = stats_from_msh(
        REPO / "affine_ddpnm_3d_random_porous/outputs/benchmark_w1n/random_sphere_partition.msh",
        "Random-27", known_mixed_dofs=75954,
    )
    real = real_stats()
    out = {"geometries": [uniform, random, real]}
    (OUT / "geometry_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

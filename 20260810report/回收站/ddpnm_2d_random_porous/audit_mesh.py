from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ddpnm2d.geometry import build_partition


def stats(values: list[float] | np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    return {
        "count": int(len(x)),
        "minimum": float(np.min(x)),
        "mean": float(np.mean(x)),
        "maximum": float(np.max(x)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DD-PNM parent/local mesh geometry.")
    parser.add_argument("--mesh-size", type=float, default=0.04)
    parser.add_argument("--wall-size", type=float, default=0.018)
    parser.add_argument("--throat-size", type=float, default=0.010)
    parser.add_argument("--wall-band", type=float, default=0.065)
    parser.add_argument("--throat-band", type=float, default=0.045)
    parser.add_argument("--output", type=Path, default=Path("outputs/enriched/mesh_audit.json"))
    args = parser.parse_args()
    partition = build_partition(
        mesh_size=args.mesh_size,
        wall_size=args.wall_size,
        throat_size=args.throat_size,
        wall_band=args.wall_band,
        throat_band=args.throat_band,
    )
    msh = partition.mesh
    tdim, fdim = msh.topology.dim, msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(tdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    f2c = msh.topology.connectivity(fdim, tdim)
    c2v = msh.topology.connectivity(tdim, 0)
    n_facets = msh.topology.index_map(fdim).size_local
    n_cells = msh.topology.index_map(tdim).size_local

    interface_lengths: dict[int, list[float]] = {}
    interface_vertices: dict[int, set[int]] = {}
    solid_wall_edges: list[float] = []
    outer_edges: list[float] = []
    interior_edges: list[float] = []
    for facet in range(n_facets):
        vertices = f2v.links(facet)
        xy = msh.geometry.x[vertices, :2]
        length = float(np.linalg.norm(xy[1] - xy[0]))
        iid = int(partition.facet_interface_ids[facet])
        if iid >= 0:
            interface_lengths.setdefault(iid, []).append(length)
            interface_vertices.setdefault(iid, set()).update(int(v) for v in vertices)
        if len(f2c.links(facet)) == 2:
            interior_edges.append(length)
        else:
            midpoint = xy.mean(axis=0)
            on_outer = (
                midpoint[0] <= 1.0e-8 or midpoint[0] >= 1.0 - 1.0e-8
                or midpoint[1] <= 1.0e-8 or midpoint[1] >= 1.0 - 1.0e-8
            )
            (outer_edges if on_outer else solid_wall_edges).append(length)

    near_sizes: list[float] = []
    far_sizes: list[float] = []
    for cell in range(n_cells):
        vertices = c2v.links(cell)
        xy = msh.geometry.x[vertices, :2]
        a = xy[1] - xy[0]
        b = xy[2] - xy[0]
        area = abs(float(a[0] * b[1] - a[1] * b[0])) / 2.0
        equivalent_h = float(np.sqrt(4.0 * area / np.sqrt(3.0)))
        if partition.cell_clearance[cell] < 0.055:
            near_sizes.append(equivalent_h)
        elif partition.cell_clearance[cell] > 0.11:
            far_sizes.append(equivalent_h)

    per_interface = []
    for interface_id, pair in enumerate(partition.interface_pairs):
        lengths = interface_lengths.get(interface_id, [])
        vertices = interface_vertices.get(interface_id, set())
        per_interface.append({
            "interface_id": interface_id,
            "pore_pair": list(pair),
            "facet_count": len(lengths),
            "unique_mesh_vertices": len(vertices),
            "polyline_length": float(sum(lengths)),
        })
    report = {
        "mesh_strategy": "analytic saddle-cut conforming mesh with wall/throat distance fields",
        "explicit_circle_refinement": True,
        "explicit_throat_refinement": True,
        "solution_adaptive_refinement": False,
        "requested_mesh_size": args.mesh_size,
        "requested_wall_size": args.wall_size,
        "requested_throat_size": args.throat_size,
        "global_cells": n_cells,
        "global_vertices": int(msh.topology.index_map(0).size_local),
        "solid_circle_boundary_edge_lengths": stats(solid_wall_edges),
        "outer_boundary_edge_lengths": stats(outer_edges),
        "interior_edge_lengths": stats(interior_edges),
        "near_solid_equivalent_triangle_size": stats(near_sizes),
        "far_from_solid_equivalent_triangle_size": stats(far_sizes),
        "minimum_interface_facets": min(x["facet_count"] for x in per_interface),
        "minimum_interface_vertices": min(x["unique_mesh_vertices"] for x in per_interface),
        "interfaces": per_interface,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

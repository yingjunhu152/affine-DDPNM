"""Coarse-mesh watershed partition smoke (hand-off section 13, experiment A).

Builds the unpartitioned fluid mesh of the frozen 27-sphere packing at
coarse resolution, runs the persistence-filtered watershed partition under
the documented boundary policy, verifies the topology invariants and writes
the full topological audit (raw maxima, pore basins, volumes, coordination
numbers, interface components, persistence) to
``outputs/watershed_smoke/`` as npz + JSON + Markdown.  No Stokes/Schur
solve is attempted: geometry and topology first.

The pore count is not compared to the 27 solid spheres; a different count
is the expected outcome of the geometric pore definition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_DIR = Path(__file__).resolve().parent.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from ddpnm_core.io import topology_arrays

from random_porous import SPHERES
from watershed_partition import (
    build_partition_watershed,
    build_superlevel_merge_tree,
    cell_adjacency,
    cell_centers,
    check_topology_invariants,
    clearance_from_policy,
    select_persistent_maxima,
    tetrahedron_volumes,
    watershed_labels,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "watershed_smoke"

# Coarse parameters mirroring _smoke_geometry.py.
MESH_PARAMS = dict(
    bulk_size=0.14,
    sphere_size=0.07,
    boundary_size=0.09,
    sphere_band=0.14,
    boundary_band=0.12,
)


def persistence_table(partition) -> np.ndarray:
    """(birth, death, persistence) rows of every merge-tree component."""
    rows = [
        (c.birth_value, c.death_level, c.persistence)
        for c in partition.components
    ]
    return np.asarray(rows, dtype=float)


def audit_counts(labels: np.ndarray, interface_pairs) -> dict:
    n_pores = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=n_pores)
    coordination = np.zeros(n_pores, dtype=int)
    for a, b in interface_pairs:
        coordination[a] += 1
        coordination[b] += 1
    return {
        "n_pores": n_pores,
        "pore_volume_counts": sizes.tolist(),
        "coordination_numbers": coordination.tolist(),
        "coordination_distribution": {
            str(int(k)): int(v)
            for k, v in sorted(
                ((c, int(np.sum(coordination == c))) for c in range(1, int(coordination.max()) + 1))
            )
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building the unpartitioned fluid mesh (coarse) ...")
    partition = build_partition_watershed(
        mesh_file=OUT_DIR / "watershed_mesh.msh",
        **MESH_PARAMS,
    )
    msh = partition.mesh
    points, tetrahedra = topology_arrays(msh)
    volumes = tetrahedron_volumes(points, tetrahedra)
    n_cells = len(tetrahedra)
    print(f"      cells={n_cells}, basins={int(partition.cell_labels.max()) + 1}, "
          f"interfaces={len(partition.interface_pairs)}")

    print("[2/4] Topology invariants ...")
    invariants = check_topology_invariants(partition, volumes)
    for name, ok in invariants.items():
        print(f"      {name}: {ok}")
    if not all(invariants.values()):
        raise SystemExit("Topology invariants failed -> smoke test failed")

    print("[3/4] Experiment A: topological audit ...")
    # Raw maxima and persistence distribution (once per boundary policy).
    report: dict = {
        "solid_spheres": int(len(SPHERES)),
        "mesh": {"cells": n_cells, "points": len(points)},
        "policy": partition.mesh_parameters["policy"],
        "raw_local_maxima": len(partition.components),
        "persistence": {
            "birth": [float(c.birth_value) for c in partition.components],
            "death": [float(c.death_level) for c in partition.components],
            "persistence": [float(c.persistence) for c in partition.components],
        },
        "audit": audit_counts(partition.cell_labels, partition.interface_pairs),
        "interface_stats": {
            "n_interfaces": int(len(partition.interface_pairs)),
            "areas_min": float(partition.interface_areas.min()),
            "areas_max": float(partition.interface_areas.max()),
            "areas_median": float(np.median(partition.interface_areas)),
            "normal_dispersion_max": float(partition.interface_normal_dispersion.max()),
            "normal_dispersion_median": float(
                np.median(partition.interface_normal_dispersion)
            ),
            "saddle_value_min": float(partition.interface_saddle_value.min()),
            "saddle_value_max": float(partition.interface_saddle_value.max()),
        },
        "invariants": {name: bool(ok) for name, ok in invariants.items()},
    }

    # Threshold scan on the fixed merge tree: pore count over the
    # (abs, rel) grid — a stable interval is the physical pore count.
    centers = cell_centers(msh)
    adjacency = cell_adjacency(msh)
    threshold_scan: dict[str, list[dict]] = {}
    for abs_threshold in (0.01, 0.02, 0.04):
        for rel_threshold in (0.05, 0.1):
            markers = select_persistent_maxima(
                partition.components, abs_threshold, rel_threshold
            )
            labels = watershed_labels(
                partition.components, markers, n_cells,
                partition.cell_clearance, adjacency,
            )
            threshold_scan[f"{abs_threshold}/{rel_threshold}"] = {
                "n_basins": int(labels.max()) + 1,
                "n_markers": int(markers.sum()),
            }
    report["threshold_scan"] = threshold_scan

    # Boundary-policy sensitivity: policy A (all six cube faces) vs the
    # documented policy B, at the same thresholds.
    policy_counts: dict[str, int] = {}
    for policy in ("walls_and_spheres", "cube_and_spheres"):
        clearance = clearance_from_policy(centers, policy)
        components = build_superlevel_merge_tree(clearance, adjacency)
        markers = select_persistent_maxima(components, 0.02, 0.05)
        labels = watershed_labels(components, markers, n_cells, clearance, adjacency)
        policy_counts[policy] = int(labels.max()) + 1
    report["policy_sensitivity_n_basins"] = policy_counts

    print("[4/4] Writing outputs ...")
    np.savez_compressed(
        OUT_DIR / "watershed_smoke_fields.npz",
        points=points,
        tetrahedra=tetrahedra,
        cell_centers=partition.cell_centers,
        cell_clearance=partition.cell_clearance,
        cell_labels=partition.cell_labels,
        cell_volumes=volumes,
        facet_interface_ids=partition.facet_interface_ids,
        interface_pairs=np.asarray(partition.interface_pairs, dtype=np.int32),
        interface_centers=partition.interface_centers,
        interface_normals=partition.interface_normals,
        interface_areas=partition.interface_areas,
        interface_normal_dispersion=partition.interface_normal_dispersion,
        interface_saddle_value=partition.interface_saddle_value,
        maximal_balls=partition.maximal_balls,
        pore_peak_clearance=partition.pore_peak_clearance,
        pore_persistence=partition.pore_persistence,
        persistence_table=persistence_table(partition),
        sphere_centers=SPHERES[:, :3],
        sphere_radii=SPHERES[:, 3],
    )
    (OUT_DIR / "watershed_smoke_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = _markdown_report(report)
    (OUT_DIR / "watershed_smoke_report.md").write_text(md, encoding="utf-8")
    print(f"Done: {OUT_DIR.resolve()}")
    print("SMOKE OK")


def _markdown_report(report: dict) -> str:
    lines = [
        "# Watershed partition smoke report (experiment A)",
        "",
        f"- solid spheres: **{report['solid_spheres']}**",
        f"- mesh cells: **{report['mesh']['cells']}**",
        f"- boundary policy: `{report['policy']}` (spheres + lateral walls; "
        "inlet/outlet open)",
        f"- raw clearance local maxima: **{report['raw_local_maxima']}**",
        "",
        "## Pores after persistence filtering",
        "",
        f"- pore basins: **{report['audit']['n_pores']}** "
        "(not constrained to equal the sphere count)",
        f"- basin sizes: {report['audit']['pore_volume_counts']}",
        f"- coordination numbers: {report['audit']['coordination_distribution']}",
        "",
        "## Interfaces",
        "",
        f"- interface components: **{report['interface_stats']['n_interfaces']}**",
        f"- areas: min {report['interface_stats']['areas_min']:.4f}, "
        f"median {report['interface_stats']['areas_median']:.4f}, "
        f"max {report['interface_stats']['areas_max']:.4f}",
        f"- normal dispersion: median {report['interface_stats']['normal_dispersion_median']:.4f}, "
        f"max {report['interface_stats']['normal_dispersion_max']:.4f}",
        f"- saddle values: min {report['interface_stats']['saddle_value_min']:.4f}, "
        f"max {report['interface_stats']['saddle_value_max']:.4f}",
        "",
        "## Threshold scan (abs / rel -> basins)",
        "",
        "| abs | rel | basins | markers |",
        "|-----|-----|--------|---------|",
    ]
    for key, value in report["threshold_scan"].items():
        abs_t, rel_t = key.split("/")
        lines.append(
            f"| {abs_t} | {rel_t} | {value['n_basins']} | {value['n_markers']} |"
        )
    lines += [
        "",
        "## Boundary-policy sensitivity",
        "",
    ]
    for policy, count in report["policy_sensitivity_n_basins"].items():
        lines.append(f"- `{policy}`: {count} basins")
    lines += [
        "",
        "## Topology invariants",
        "",
    ]
    for name, ok in report["invariants"].items():
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

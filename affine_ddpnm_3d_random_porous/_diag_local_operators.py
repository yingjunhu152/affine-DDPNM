"""Diagnostic: locate the ill-posed local operators of the watershed partition.

Rebuilds the watershed partition at the benchmark parameters, builds the
Classic response library (the same code path that produced the 1e14 Schur
eigenvalues) and reports per-pore diagnostics: cell count, port structure,
wall facet count, response/G norms, and the largest-G pores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY_DIR = Path(__file__).resolve().parent.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from ddpnm_core.library import build_response_library
from ddpnm_core.stokes_operator import build_local_stokes_operator
from watershed_partition import build_partition_watershed

sys.path.insert(0, str(REPOSITORY_DIR / "ddpnm_3d_uniform_spheres"))
from affine_face_basis import CompatibleClassicP0Basis


def main() -> None:
    print("[1/3] Building the watershed partition (benchmark parameters) ...",
          flush=True)
    partition = build_partition_watershed(
        bulk_size=0.13, sphere_size=0.065, boundary_size=0.085,
        sphere_band=0.15, boundary_band=0.13,
        policy="walls_and_spheres", abs_threshold=0.02, rel_threshold=0.05,
    )
    n_pores = int(np.max(partition.cell_labels)) + 1
    print(f"      basins={n_pores}, interfaces={len(partition.interface_pairs)}",
          flush=True)

    print("[2/3] Building local operators ...", flush=True)
    basis = CompatibleClassicP0Basis()
    library = build_response_library(
        partition, basis, viscosity=1.0, inlet_pressure=1.0,
        outlet_pressure=0.0, pressure_stabilization=0.0,
    )

    print("[3/3] Per-pore diagnostics ...", flush=True)
    rows = []
    for entry in library.entries:
        op = entry.operator
        n_wall = 0
        n_inlet = 0
        n_outlet = 0
        n_interface_ports = 0
        n_interface_facets = 0
        for port in op.ports:
            if port.kind == "interface":
                n_interface_ports += 1
                n_interface_facets += len(port.parent_facets)
            elif port.kind == "inlet":
                n_inlet += 1
            elif port.kind == "outlet":
                n_outlet += 1
        scale = max(float(np.linalg.norm(entry.primitive_G)), 1.0e-30)
        rows.append(
            {
                "pore": op.pore_id,
                "n_cells": int(op.submesh.topology.index_map(op.submesh.topology.dim).size_local),
                "n_wall_facets": len(op.facet_tags.find(4)),
                "n_interface_ports": n_interface_ports,
                "n_interface_facets": n_interface_facets,
                "n_inlet": n_inlet,
                "n_outlet": n_outlet,
                "G_norm": float(np.linalg.norm(entry.primitive_G)),
                "G_max": float(np.abs(entry.primitive_G).max()),
            }
        )
    rows.sort(key=lambda r: -r["G_norm"])
    print(f"{'pore':>4} {'cells':>5} {'walls':>5} {'if_ports':>8} "
          f"{'if_facets':>9} {'in/out':>7} {'|G|':>12} {'max|G|':>12}")
    for row in rows[:12]:
        print(
            f"{row['pore']:>4} {row['n_cells']:>5} {row['n_wall_facets']:>5} "
            f"{row['n_interface_ports']:>8} {row['n_interface_facets']:>9} "
            f"{row['n_inlet']}/{row['n_outlet']:>4} "
            f"{row['G_norm']:>12.4e} {row['G_max']:>12.4e}"
        )
    print("...")
    for row in rows[-3:]:
        print(
            f"{row['pore']:>4} {row['n_cells']:>5} {row['n_wall_facets']:>5} "
            f"{row['n_interface_ports']:>8} {row['n_interface_facets']:>9} "
            f"{row['n_inlet']}/{row['n_outlet']:>4} "
            f"{row['G_norm']:>12.4e} {row['G_max']:>12.4e}"
        )
    normals = np.asarray(rows)
    print(f"n pores with |G| > 1e4: "
          f"{int(np.sum([r['G_norm'] > 1e4 for r in rows]))}")
    print(f"n pores with 0 wall facets: "
          f"{int(np.sum([r['n_wall_facets'] == 0 for r in rows]))}")
    print("DIAG DONE")


if __name__ == "__main__":
    main()

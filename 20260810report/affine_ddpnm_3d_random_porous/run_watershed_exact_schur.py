"""Exact FE-trace Schur correctness baseline on the watershed partition (5.4).

The dense exact-Schur implementation was established as the correctness
baseline on the Voronoi partition (monolithic difference 1.05e-12).  This
script re-runs it on the merged watershed partition (82 basins, 369
interfaces) so the watershed DDPNM numbers carry the same roundoff-level
correctness statement.  Not an efficiency claim: the dense implementation is
slow by design.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_DIR = Path(__file__).resolve().parent.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from ddpnm_core.fem_utils import solve_reference
from ddpnm_core.trace_schur import solve_exact_fe_schur
from watershed_partition import build_partition_watershed

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "ablation_4way" / "watershed"


def main() -> None:
    print("[1/3] Building the watershed partition ...", flush=True)
    partition = build_partition_watershed(
        bulk_size=0.13, sphere_size=0.065, boundary_size=0.085,
        sphere_band=0.15, boundary_band=0.13,
        policy="walls_and_spheres", abs_threshold=0.02, rel_threshold=0.05,
    )
    print(
        f"      basins={int(np.max(partition.cell_labels)) + 1}, "
        f"interfaces={len(partition.interface_pairs)}",
        flush=True,
    )

    print("[2/3] Monolithic FEM reference ...", flush=True)
    started = time.perf_counter()
    reference = solve_reference(
        partition.mesh, viscosity=1.0, inlet_pressure=1.0, outlet_pressure=0.0,
        pressure_stabilization=0.0,
        iterative_threshold=100_000, iterative_rtol=1.0e-9,
        iterative_restart=60, iterative_maxiter=150,
        ilu_drop_tolerance=2.0e-3, ilu_fill_factor=6.0,
    )
    fem_seconds = time.perf_counter() - started
    print(f"      dofs={reference.ndofs}, solve={fem_seconds:.3f} s", flush=True)

    print("[3/3] Exact dense FE-trace Schur ...", flush=True)
    started = time.perf_counter()
    schur = solve_exact_fe_schur(
        partition,
        viscosity=1.0,
        inlet_pressure=1.0,
        outlet_pressure=0.0,
        pressure_stabilization=1.0e-10,
    )
    schur_seconds = time.perf_counter() - started
    report = {
        "partition": "watershed (merged)",
        "total_mixed_dofs": int(len(schur.fixed_dofs))
        + int(len(schur.interface_dofs))
        + int(len(schur.interior_dofs)),
        "interface_trace_dofs": int(len(schur.interface_dofs)),
        "interior_dofs": int(len(schur.interior_dofs)),
        "schur_symmetry_error": float(schur.schur_symmetry_error),
        "schur_relative_residual": float(schur.schur_relative_residual),
        "global_relative_residual": float(schur.global_relative_residual),
        "monolithic_relative_difference": float(
            schur.monolithic_relative_difference
        ),
        "schur_seconds": float(schur_seconds),
        "fem_seconds": float(fem_seconds),
    }
    print(
        f"      trace dofs={report['interface_trace_dofs']}, "
        f"solve={schur_seconds:.1f} s, vs monolithic="
        f"{report['monolithic_relative_difference']:.2e}",
        flush=True,
    )
    (OUT_DIR / "watershed_exact_schur.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done: {OUT_DIR / 'watershed_exact_schur.json'}")


if __name__ == "__main__":
    main()

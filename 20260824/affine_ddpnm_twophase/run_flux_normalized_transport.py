#!/usr/bin/env python3
"""Counterfactual flux-normalization experiment for the DDPNM recovery bias.

The recovery bias decomposes exactly as ``dR = 1.25*dPVI - leak``
(``outputs/sensitivity_1d/SENSITIVITY_REPORT.md``): the dominant term is
the DDPNM total-flux error amplified by the mass-balance law.  This script
rescales each archived DDPNM P1 velocity field so its inlet volume flux
matches the FEM field's (``alpha = Q_FEM / Q_method``), reruns the
identical two-phase transport, and checks that the recovery deviation
collapses to the small leak correction (throughput term removed).

The velocity fields come from ``twophase_fields.npz`` and the benchmark
mesh is loaded from the archived ``random_sphere_partition.msh`` (exact
geometry/quadrature: the inlet flux reproduces the transport history
bit-for-bit), so no Stokes solve or mesh generation is needed.  The FEM
rerun doubles as a self-consistency check (R must reproduce 0.2791).

Run: conda run -n fenicsx --no-capture-output python run_flux_normalized_transport.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh as dmesh

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import two_phase_transport as tp
from ddpnm_core.io import assign_p1_function
from two_phase_physics import crossing_time

# Archived benchmark results (same metric pipeline).
ARCHIVE = {
    "Classic_DDPNM_1": {"dR": 0.1488, "leak": 0.0625, "pvi": 0.3998, "r": 0.4279},
    "NormalLinear_DDPNM_3": {"dR": 0.0280, "leak": 0.0143, "pvi": 0.2646, "r": 0.3071},
    "Affine_DDPNM_9": {"dR": 0.0060, "leak": 0.0020, "pvi": 0.2372, "r": 0.2851},
}
R_FEM_ARCHIVE = 0.2791
METHOD_KEYS = ("FEM", "Classic_DDPNM_1", "NormalLinear_DDPNM_3", "Affine_DDPNM_9")
DISPLAY = {
    "FEM": "FEM",
    "Classic_DDPNM_1": "Classic-DDPNM-1",
    "NormalLinear_DDPNM_3": "NormalLinear-DDPNM-3",
    "Affine_DDPNM_9": "Affine-DDPNM-9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fields",
        type=Path,
        default=PROJECT_DIR / "outputs" / "benchmark_twophase" / "twophase_fields.npz",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=PROJECT_DIR / "outputs" / "benchmark_twophase" / "random_sphere_partition.msh",
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--t-final", type=float, default=30.0)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_DIR / "outputs" / "flux_normalized",
    )
    return parser.parse_args()


def load_benchmark_mesh(path: Path):
    """Load the original benchmark mesh from the archived .msh (exact
    geometry and quadrature: inlet flux reproduces the transport history)."""
    from dolfinx.io import gmsh as gmshio

    result = gmshio.read_from_msh(str(path), MPI.COMM_SELF, gdim=3)
    return result[0]


def inlet_volume_flux(msh, u_vertices: np.ndarray) -> float:
    """Total inlet volume flux int_{inlet} -min(u.n,0) ds of a P1 vertex field
    (same facet tagging and quadrature as the transport solver)."""
    fdim = msh.topology.dim - 1
    inlet_facets = np.asarray(
        dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0)),
        dtype=np.int32,
    )
    outlet_facets = np.asarray(
        dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0)),
        dtype=np.int32,
    )
    if len(inlet_facets) == 0 or len(outlet_facets) == 0:
        raise RuntimeError("failed to locate inlet or outlet boundary facets")
    boundary_facets = np.concatenate((inlet_facets, outlet_facets))
    boundary_values = np.concatenate(
        (np.full(len(inlet_facets), 1, dtype=np.int32),
         np.full(len(outlet_facets), 2, dtype=np.int32))
    )
    order = np.argsort(boundary_facets)
    facet_tags = dmesh.meshtags(msh, fdim, boundary_facets[order], boundary_values[order])
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    U = fem.functionspace(msh, ("Lagrange", 1, (3,)))
    u = fem.Function(U)
    assign_p1_function(u, np.asarray(u_vertices, dtype=float))
    u.x.scatter_forward()
    un = ufl.dot(u, ufl.FacetNormal(msh))
    inflow = fem.form(ufl.min_value(un, 0.0) * ds(1))
    return float(-fem.assemble_scalar(inflow))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    data = np.load(args.fields)
    print(f"[1/4] loading benchmark mesh from {args.mesh.name} ...")
    msh = load_benchmark_mesh(args.mesh)
    n_cells = msh.topology.index_map(msh.topology.dim).size_local
    print(f"      cells={n_cells} (archive 15249)")

    print("[2/4] inlet fluxes of the archived fields ...")
    fluxes: dict[str, float] = {}
    for key in METHOD_KEYS:
        fluxes[key] = inlet_volume_flux(msh, data[f"u_{key}"])
        print(f"      {DISPLAY[key]}: Q_in = {fluxes[key]:.6f}")
    alpha = {key: 1.0 for key in METHOD_KEYS}
    for key in ("Classic_DDPNM_1", "NormalLinear_DDPNM_3", "Affine_DDPNM_9"):
        alpha[key] = fluxes["FEM"] / fluxes[key]
        print(f"      {DISPLAY[key]}: alpha = {alpha[key]:.5f}")

    print("[3/4] two-phase transport with flux-normalized fields (snapshot_every=10) ...")
    histories: dict[str, dict] = {}
    for key in METHOD_KEYS:
        u_norm = alpha[key] * np.asarray(data[f"u_{key}"], dtype=float)
        t0 = time.perf_counter()
        result = tp.solve_two_phase(
            msh, u_norm,
            diffusivity=0.0, porosity=1.0,
            dt=args.dt, t_final=args.t_final,
            sw_initial=0.2, sw_inlet=None,
            corey=None,
            picard_max_iters=6, picard_tol=1.0e-6,
            picard_relaxation=1.0,
            supg=True, supg_factor=0.50,
            snapshot_every=args.snapshot_every,
        )
        seconds = time.perf_counter() - t0
        h = result["history"]
        histories[key] = h
        print(
            f"      {DISPLAY[key]}: R={h['recovery'][-1]:.4f} PVI={h['pore_volumes_injected'][-1]:.4f} "
            f"({seconds:.0f} s)"
        )

    print("[4/4] comparison vs archive ...")
    r_fem = float(histories["FEM"]["recovery"][-1])
    print(f"      FEM rerun R = {r_fem:.4f} (archive 0.2791); "
          f"delta = {r_fem - R_FEM_ARCHIVE:+.5f}")
    rows: list[dict[str, object]] = []
    for key in ("Classic_DDPNM_1", "NormalLinear_DDPNM_3", "Affine_DDPNM_9"):
        h = histories[key]
        r_norm = float(h["recovery"][-1])
        pvi_norm = float(h["pore_volumes_injected"][-1])
        dR_norm = r_norm - r_fem
        arch = ARCHIVE[key]
        rows.append(
            {
                "method": DISPLAY[key],
                "alpha": alpha[key],
                "recovery_normalized": r_norm,
                "pvi_normalized": pvi_norm,
                "dR_normalized": dR_norm,
                "dR_original": arch["dR"],
                "leak_original": arch["leak"],
                "predicted_dR_norm_approx_minus_leak": -arch["leak"],
                "watercut_t50_normalized": crossing_time(h["time"], h["watercut"], 0.50),
                "s_min": float(h["min_s"][-1]),
                "s_max": float(h["max_s"][-1]),
                "max_balance_resid": float(np.max(np.abs(h["mass_balance_relative_residual"]))),
            }
        )
        print(
            f"      {DISPLAY[key]}: dR_norm={dR_norm:+.4f} (original {arch['dR']:+.4f}, "
            f"leak {arch['leak']:.4f})  PVI_norm={pvi_norm:.4f}"
        )

    summary = {
        "description": (
            "Counterfactual: DDPNM fields rescaled to the FEM total flux, same two-phase transport; "
            "the recovery bias should collapse to the leak correction."
        ),
        "fem_rerun_recovery": r_fem,
        "fem_archive_recovery": R_FEM_ARCHIVE,
        "fem_rebuild_delta": r_fem - R_FEM_ARCHIVE,
        "inlet_fluxes": fluxes,
        "alpha": alpha,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - t0_all,
    }
    from run_affine_ddpnm_twophase import json_safe

    (args.out_dir / "flux_normalized_report.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    csv_lines = [
        "method,alpha,recovery_normalized,pvi_normalized,dR_normalized,dR_original,"
        "leak_original,watercut_t50_normalized"
    ]
    for row in rows:
        csv_lines.append(
            f"{row['method']},{row['alpha']:.6f},{row['recovery_normalized']:.6f},"
            f"{row['pvi_normalized']:.6f},{row['dR_normalized']:.6f},"
            f"{row['dR_original']:.6f},{row['leak_original']:.6f},"
            f"{row['watercut_t50_normalized']}"
        )
    (args.out_dir / "flux_normalized_metrics.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )
    print(f"Done: {args.out_dir.resolve()} ({time.perf_counter() - t0_all:.0f} s)")


if __name__ == "__main__":
    main()

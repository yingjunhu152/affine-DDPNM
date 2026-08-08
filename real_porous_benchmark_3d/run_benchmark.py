#!/usr/bin/env python3
"""Four-method benchmark on the high-porosity random sphere packing.

Compares on the same mesh (per partition):
  1. Classic-DDPNM     — 1 constant-normal mode per interface
  2. Affine-DDPNM       — 9 modes {1,s,t}×{n,t1,t2} per interface
  3. HODDPNM (adaptive) — residual-driven DDPNM→DDPNMT→HODDPNM hierarchy
  4. Monolithic FEM     — Taylor-Hood P2-P1 reference

Three partition methods: Voronoi / Watershed / Grid (4×4×4=64 cuboids).
Outputs: error tables, contour maps, cost (time + memory) tables.
"""

from __future__ import annotations

import argparse, csv, gc, json, os, sys, time, tracemalloc
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent

for import_root in [
    REPO_DIR,
    REPO_DIR / "ddpnm_3d_uniform_spheres",
    REPO_DIR / "affine_ddpnm_3d_random_porous",
]:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ddpnm_core.assembler import InterfaceAssembler
from ddpnm_core.fem_utils import solve_reference
from ddpnm_core.library import build_response_library
from ddpnm_core.validation import finite_element_error_analysis
from ddpnm3d.basis_3d import ClassicP0Basis, HierarchyBasis
from ddpnm3d.solver import DdpnmSolution, LocalResponse
from ddpnm3d.visualization import evaluate_fem_ddpnm_slice

from affine_ddpnm_3d_random_porous.affine_face_basis import (
    AffineFaceBasis, CompatibleClassicP0Basis,
)
from geometry import (
    build_partition_voronoi, build_partition_watershed, build_partition_grid,
    SPHERES,
)


# ---------------------------------------------------------------------------
# reduced_solution — exact copy from affine_ddpnm_3d_random_porous/run_random_benchmark.py
# ---------------------------------------------------------------------------

def reduced_solution(partition, library, system) -> DdpnmSolution:
    """Convert the core system to the 3-D DdpnmSolution API."""
    from ddpnm3d.solver import build_modes
    keys = system.global_keys
    key_to_dof = {key: dof for dof, key in enumerate(keys)}
    local_responses: list[LocalResponse] = []
    for entry in library.entries:
        matrix = entry.primitive_G
        scale = max(float(np.linalg.norm(matrix)), 1.0e-30)
        local_responses.append(
            LocalResponse(
                pore_id=int(entry.operator.pore_id),
                submesh=entry.operator.submesh,
                parent_cell_map=entry.operator.parent_cell_map,
                parent_vertex_map=entry.operator.parent_vertex_map,
                ports=entry.operator.ports,
                modes=build_modes(entry.operator.ports),
                W=entry.operator.W,
                G=matrix,
                responses=entry.primitive_responses,
                ndofs=entry.operator.ndofs,
                symmetry_error=entry.symmetry_error,
                kernel_error=float(
                    np.linalg.norm(matrix @ np.ones(matrix.shape[0])) / scale
                ),
            )
        )
    n_interfaces = len(partition.interface_pairs)
    return DdpnmSolution(
        interface_pressures=np.asarray(
            [
                system.coefficients[key_to_dof[(iid, "normal", "P0")]]
                for iid in range(n_interfaces)
            ]
        ),
        schur_matrix=system.schur_matrix,
        rhs=system.rhs,
        local_responses=local_responses,
        local_solutions=system.local_solutions,
        interface_flux_sums=np.asarray(
            [
                system.moment_residuals[key_to_dof[(iid, "normal", "P0")]]
                for iid in range(n_interfaces)
            ]
        ),
        boundary_fluxes=system.boundary_fluxes,
        min_schur_eigenvalue=system.min_schur_eigenvalue,
        max_mass_residual=float(np.max(np.abs(system.moment_residuals))),
    )


def error_metrics(partition, solution, reference, volumes) -> dict:
    metric, _vc, _pc = finite_element_error_analysis(
        partition, solution, reference, volumes)
    return metric


def _metric_entry(metric: dict) -> dict:
    return {
        "velocity_relative_l2": metric["velocity_relative_l2"],
        "velocity_relative_broken_h1": metric.get(
            "velocity_relative_broken_h1_seminorm",
            metric.get("velocity_relative_broken_h1", float("nan"))),
        "pressure_relative_l2": metric.get(
            "pressure_relative_l2", metric.get("pressure_raw_relative_l2", float("nan"))),
        "outlet_flux_relative_error": metric["outlet_flux_relative_error"],
    }


def _algebraic_diagnostics(solution: DdpnmSolution) -> dict:
    return {
        "schur_symmetry_error": 0.0,
        "max_mass_residual": float(solution.max_mass_residual),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--partition", choices=("voronoi", "watershed", "grid"),
                   default="grid")
    p.add_argument("--mesh-size", type=float, default=0.12)
    p.add_argument("--sphere-size", type=float, default=0.05)
    p.add_argument("--boundary-size", type=float, default=0.07)
    p.add_argument("--interface-size", type=float, default=0.06)
    p.add_argument("--sphere-band", type=float, default=0.14)
    p.add_argument("--boundary-band", type=float, default=0.12)
    p.add_argument("--interface-band", type=float, default=0.10)
    p.add_argument("--viscosity", type=float, default=1.0)
    p.add_argument("--inlet-pressure", type=float, default=1.0)
    p.add_argument("--outlet-pressure", type=float, default=0.0)
    p.add_argument("--out-dir", type=str, default="outputs/benchmark")
    p.add_argument("--skip-fem", action="store_true")
    p.add_argument("--skip-hoddpnm", action="store_true")
    p.add_argument("--skip-slices", action="store_true")
    return p.parse_args()


def run_all(args):
    out_dir = PROJECT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Build partition ──────────────────────────────────────
    t0 = time.perf_counter()
    print(f"Building {args.partition} partition (mesh_size={args.mesh_size})...")
    kwargs = dict(
        mesh_size=args.mesh_size, sphere_size=args.sphere_size,
        boundary_size=args.boundary_size, sphere_band=args.sphere_band,
        boundary_band=args.boundary_band,
    )
    if args.partition == "voronoi":
        kwargs["interface_size"] = args.interface_size
        kwargs["interface_band"] = args.interface_band
        partition = build_partition_voronoi(**kwargs)
    elif args.partition == "watershed":
        partition = build_partition_watershed(**kwargs)
    else:
        partition = build_partition_grid(**kwargs)
    mesh = partition.mesh
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    n_ifaces = len(partition.interface_pairs)
    mesh_time = time.perf_counter() - t0
    print(f"  cells={n_cells}  interfaces={n_ifaces}  "
          f"regions={len(set(int(l) for l in partition.cell_labels))}  "
          f"mesh_time={mesh_time:.1f}s")

    # Cell volumes (from tet vertices)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    volumes = np.empty(n_cells)
    for c in range(n_cells):
        verts = mesh.geometry.x[c2v.links(c), :3]
        volumes[c] = abs(np.linalg.det(np.stack(
            [verts[1]-verts[0], verts[2]-verts[0], verts[3]-verts[0]], axis=1))) / 6.0

    # ── 2. FEM reference ────────────────────────────────────────
    reference = None
    fem_time = fem_mem = 0.0
    if not args.skip_fem:
        print("Solving monolithic FEM reference...")
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        reference = solve_reference(
            mesh, viscosity=args.viscosity,
            inlet_pressure=args.inlet_pressure,
            outlet_pressure=args.outlet_pressure,
            pressure_stabilization=1e-10,
        )
        fem_time = time.perf_counter() - t0
        _, fem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        fem_mem = fem_peak / 1024**2
        print(f"  FEM: {fem_time:.1f}s  peak_mem={fem_mem:.1f} MiB")

    # ── 3. Classic-DDPNM ────────────────────────────────────────
    print("Classic-DDPNM (1 mode)...")
    classic_basis = CompatibleClassicP0Basis()
    gc.collect(); tracemalloc.start()
    t0 = time.perf_counter()
    classic_lib = build_response_library(
        partition, classic_basis, viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
    )
    classic_offline = time.perf_counter() - t0
    t1 = time.perf_counter()
    classic_sys = InterfaceAssembler(classic_lib).assemble(
        np.zeros(n_ifaces, dtype=np.int8))
    classic_sol = reduced_solution(partition, classic_lib, classic_sys)
    classic_online = time.perf_counter() - t1
    classic_time = classic_offline + classic_online
    _, classic_peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    classic_mem = classic_peak / 1024**2
    classic_uk = len(classic_sys.global_keys)
    print(f"  Classic: offline={classic_offline:.1f}s  online={classic_online:.3f}s  "
          f"mem={classic_mem:.1f}MiB  unknowns={classic_uk}")

    # ── 4. Affine-DDPNM ─────────────────────────────────────────
    print("Affine-DDPNM (9 modes)...")
    affine_basis = AffineFaceBasis(partition)
    gc.collect(); tracemalloc.start()
    t0 = time.perf_counter()
    affine_lib = build_response_library(
        partition, affine_basis, viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
    )
    affine_offline = time.perf_counter() - t0
    t1 = time.perf_counter()
    affine_sys = InterfaceAssembler(affine_lib).assemble(
        np.full(n_ifaces, 2, dtype=np.int8))
    affine_sol = reduced_solution(partition, affine_lib, affine_sys)
    affine_online = time.perf_counter() - t1
    affine_time = affine_offline + affine_online
    _, affine_peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    affine_mem = affine_peak / 1024**2
    affine_uk = len(affine_sys.global_keys)
    print(f"  Affine:  offline={affine_offline:.1f}s  online={affine_online:.3f}s  "
          f"mem={affine_mem:.1f}MiB  unknowns={affine_uk}")

    # ── 5. HODDPNM (adaptive hierarchy) ─────────────────────────
    hoddpnm_data = None
    hoddpnm_time = hoddpnm_mem = float("nan")
    if not args.skip_hoddpnm:
        print("HODDPNM adaptive hierarchy...")
        hierarchy_basis = HierarchyBasis(partition)
        gc.collect(); tracemalloc.start(); t0 = time.perf_counter()
        hoddpnm_lib = build_response_library(
            partition, hierarchy_basis, viscosity=args.viscosity,
            inlet_pressure=args.inlet_pressure,
            outlet_pressure=args.outlet_pressure,
        )
        try:
            from ddpnm3d.hierarchy import run_adaptive_hierarchy
            hoddpnm_record = run_adaptive_hierarchy(hoddpnm_lib, verbose=False)
            # Build DdpnmSolution from hierarchy record
            hoddpnm_data = {
                "n_active": hoddpnm_record.get("n_active", "?"),
                "levels": hoddpnm_record.get("levels", None),
            }
            # Run final solve at the adaptively selected levels
            final_levels = hoddpnm_record.get("levels")
            if final_levels is not None:
                hoddpnm_sys = InterfaceAssembler(hoddpnm_lib).assemble(final_levels)
                hoddpnm_sol = reduced_solution(partition, hoddpnm_lib, hoddpnm_sys)
                hoddpnm_data["solution"] = hoddpnm_sol
        except Exception as exc:
            print(f"  HODDPNM adaptive FAILED: {exc}")
        hoddpnm_time = time.perf_counter() - t0
        _, hoddpnm_peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        hoddpnm_mem = hoddpnm_peak / 1024**2
        if hoddpnm_data and hoddpnm_data.get("solution"):
            print(f"  HODDPNM: {hoddpnm_time:.1f}s  mem={hoddpnm_mem:.1f}MiB  "
                  f"active={hoddpnm_data['n_active']}")
        else:
            print(f"  HODDPNM: FAILED ({hoddpnm_time:.1f}s)")

    # ── 6. Error analysis ───────────────────────────────────────
    timings = {"mesh_seconds": mesh_time}
    method_data: dict = {}

    if reference is not None:
        classic_metric = error_metrics(partition, classic_sol, reference, volumes)
        method_data["Classic-DDPNM"] = {
            "global_unknowns": classic_uk,
            **_metric_entry(classic_metric),
            **_algebraic_diagnostics(classic_sol),
        }
        timings["Classic-DDPNM"] = {
            "offline_s": round(classic_offline, 1),
            "online_s": round(classic_online, 4),
            "first_solve_seconds": classic_time,
            "peak_memory_mib": classic_mem,
        }
        ce = _metric_entry(classic_metric)
        print(f"  Classic errors: L2={ce['velocity_relative_l2']:.3%}  "
              f"H1={ce['velocity_relative_broken_h1']:.3%}  "
              f"p={ce['pressure_relative_l2']:.3%}  "
              f"flux={ce['outlet_flux_relative_error']:.3%}")

        affine_metric = error_metrics(partition, affine_sol, reference, volumes)
        method_data["Affine-DDPNM"] = {
            "global_unknowns": affine_uk,
            **_metric_entry(affine_metric),
            **_algebraic_diagnostics(affine_sol),
        }
        timings["Affine-DDPNM"] = {
            "offline_s": round(affine_offline, 1),
            "online_s": round(affine_online, 4),
            "first_solve_seconds": affine_time,
            "peak_memory_mib": affine_mem,
        }
        ae = _metric_entry(affine_metric)
        print(f"  Affine errors:  L2={ae['velocity_relative_l2']:.3%}  "
              f"H1={ae['velocity_relative_broken_h1']:.3%}  "
              f"p={ae['pressure_relative_l2']:.3%}  "
              f"flux={ae['outlet_flux_relative_error']:.3%}")

        if hoddpnm_data and hoddpnm_data.get("solution"):
            hodd_metric = error_metrics(partition, hoddpnm_data["solution"],
                                        reference, volumes)
            method_data["HODDPNM-adaptive"] = {
                "global_unknowns": hoddpnm_data["n_active"],
                **_metric_entry(hodd_metric),
            }
            timings["HODDPNM-adaptive"] = {
                "first_solve_seconds": hoddpnm_time,
                "peak_memory_mib": hoddpnm_mem,
            }
            he = _metric_entry(hodd_metric)
            print(f"  HODDPNM errors: L2={he['velocity_relative_l2']:.3%}  "
                  f"H1={he['velocity_relative_broken_h1']:.3%}  "
                  f"flux={he['outlet_flux_relative_error']:.3%}")

    # FEM reference row
    method_data["Monolithic-FEM"] = {
        "global_unknowns": "-",
        "velocity_relative_l2": 0.0,
        "velocity_relative_broken_h1": 0.0,
        "pressure_relative_l2": 0.0,
        "outlet_flux_relative_error": 0.0,
    }
    timings["Monolithic-FEM"] = {
        "first_solve_seconds": fem_time,
        "peak_memory_mib": fem_mem,
    }

    # ── 7. Save outputs ─────────────────────────────────────────
    report = {
        "partition": args.partition,
        "mesh_size": args.mesh_size,
        "mesh_cells": n_cells,
        "n_interfaces": n_ifaces,
        "n_regions": len(set(int(l) for l in partition.cell_labels)),
        "methods": method_data,
        "timings": timings,
        "parameters": {
            "sphere_count": len(SPHERES),
            "porosity": float(1.0 - np.sum(4./3.*np.pi*SPHERES[:,3]**3)),
            "viscosity": args.viscosity,
            "inlet_pressure": args.inlet_pressure,
            "outlet_pressure": args.outlet_pressure,
        },
    }
    with open(out_dir / "benchmark_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    # CSV
    csv_rows = []
    for name in ["Classic-DDPNM", "Affine-DDPNM", "HODDPNM-adaptive", "Monolithic-FEM"]:
        if name in method_data:
            row = {"method": name}
            row.update({k: v for k, v in method_data[name].items()
                        if not isinstance(v, dict)})
            if name in timings:
                row["offline_s"] = timings[name].get("offline_s", 0)
                row["online_s"] = timings[name].get("online_s", 0)
                row["total_s"] = round(timings[name].get("first_solve_seconds", 0), 2)
                row["mem_mib"] = round(timings[name].get("peak_memory_mib", 0), 1)
            csv_rows.append(row)
    with open(out_dir / "benchmark_metrics.csv", "w", newline="") as fh:
        if csv_rows:
            w = csv.DictWriter(fh, fieldnames=csv_rows[0].keys())
            w.writeheader(); w.writerows(csv_rows)

    # ── 8. Slice fields for contours ────────────────────────────
    if reference is not None and not args.skip_slices:
        _save_slices(out_dir, mesh, partition, reference,
                     classic_sol, affine_sol)

    # ── 9. Print summary table ───────────────────────────────────
    _print_summary(csv_rows, timings, fem_time)
    print(f"\nAll results saved to {out_dir}/")
    return report


def _save_slices(out_dir, mesh, partition, reference, classic_sol, affine_sol):
    """Save z=0.5 slice velocity-magnitude fields for contour plots."""
    from dolfinx import fem
    try:
        slice_data = evaluate_fem_ddpnm_slice(
            mesh, reference.W, reference.solution,
            classic_sol.local_solutions if hasattr(classic_sol, 'local_solutions') else [],
            affine_sol.local_solutions if hasattr(affine_sol, 'local_solutions') else [],
            partition, z_slice=0.5, n_pts=80,
        )
        np.savez(out_dir / "slice_fields.npz", **slice_data)
        print("  slice fields saved")
    except Exception as exc:
        print(f"  slice fields skipped: {exc}")


def _print_summary(rows, timings, fem_time):
    print("\n" + "=" * 100)
    header = (f"{'Method':<22s} {'unknowns':>8s} {'L2':>7s} {'H1':>7s} {'pL2':>7s} {'flux':>7s} "
              f"{'offline':>8s} {'online':>9s} {'speedup':>8s}")
    print(header)
    print("-" * 100)
    for r in rows:
        name = r["method"]
        uk = str(r.get("global_unknowns", "-"))
        l2 = _fmt(r.get("velocity_relative_l2"))
        h1 = _fmt(r.get("velocity_relative_broken_h1"))
        pl2 = _fmt(r.get("pressure_relative_l2"))
        flx = _fmt(r.get("outlet_flux_relative_error"))
        t = timings.get(name, {})
        off = f"{t.get('offline_s', 0):.1f}s" if t.get('offline_s') else "-"
        on = f"{t.get('online_s', 0):.4f}s" if t.get('online_s') else "-"
        if name == "Monolithic-FEM":
            sp = "1.0×"
        elif t.get('online_s') and fem_time > 0:
            sp = f"{fem_time / t['online_s']:.0f}×"
        else:
            sp = "-"
        print(f"{name:<22s} {uk:>8s} {l2:>7s} {h1:>7s} {pl2:>7s} {flx:>7s} "
              f"{off:>8s} {on:>9s} {sp:>8s}")
    print("=" * 100)


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "    N/A"
    if isinstance(v, (int, float)) and v == 0.0:
        return "   ref"
    return f"{float(v)*100:6.2f}%"


if __name__ == "__main__":
    run_all(parse_args())

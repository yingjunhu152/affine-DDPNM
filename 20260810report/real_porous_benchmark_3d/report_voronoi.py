#!/usr/bin/env python3
"""Generate complete Voronoi benchmark report with off/online timing."""
import sys, time, gc, tracemalloc, json, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ddpnm_3d_uniform_spheres"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "affine_ddpnm_3d_random_porous"))

from geometry import build_partition_voronoi, SPHERES
from ddpnm_core.fem_utils import solve_reference
from ddpnm_core.library import build_response_library
from ddpnm_core.assembler import InterfaceAssembler
from ddpnm_core.validation import finite_element_error_analysis
from affine_ddpnm_3d_random_porous.affine_face_basis import CompatibleClassicP0Basis, AffineFaceBasis
from ddpnm3d.solver import DdpnmSolution, LocalResponse, build_modes

OUT = Path(__file__).resolve().parent / "outputs" / "benchmark_voronoi"
OUT.mkdir(parents=True, exist_ok=True)

# --- Partition ---
print("Building Voronoi partition..."); t0 = time.perf_counter()
p = build_partition_voronoi(mesh_size=0.12, sphere_size=0.05, boundary_size=0.07,
    interface_size=0.06, sphere_band=0.14, boundary_band=0.12, interface_band=0.10)
mesh = p.mesh; n_ifaces = len(p.interface_pairs)
nc = mesh.topology.index_map(mesh.topology.dim).size_local
nr = len(set(int(l) for l in p.cell_labels))
mesh_t = time.perf_counter() - t0
print(f"  {nc} cells, {n_ifaces} interfaces, {nr} regions ({mesh_t:.1f}s)")

# --- FEM ---
print("FEM reference..."); gc.collect(); tracemalloc.start(); t0 = time.perf_counter()
ref = solve_reference(mesh, viscosity=1.0, inlet_pressure=1.0, outlet_pressure=0.0,
                      pressure_stabilization=1e-10)
fem_t = time.perf_counter() - t0; _, fem_m = tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"  {fem_t:.1f}s, {fem_m/1024**2:.0f}MiB")

# Cell volumes
td = mesh.topology.dim; mesh.topology.create_connectivity(td, 0); c2 = mesh.topology.connectivity(td, 0)
vols = np.empty(nc)
for c in range(nc):
    vv = mesh.geometry.x[c2.links(c), :3]
    vols[c] = abs(np.linalg.det(np.stack([vv[1]-vv[0], vv[2]-vv[0], vv[3]-vv[0]], axis=1))) / 6.0

def reduced_sol(lib, sys, ni):
    keys = sys.global_keys; k2d = {k: d for d, k in enumerate(keys)}
    lrs = [LocalResponse(
        pore_id=int(e.operator.pore_id), submesh=e.operator.submesh,
        parent_cell_map=e.operator.parent_cell_map,
        parent_vertex_map=e.operator.parent_vertex_map,
        ports=e.operator.ports, modes=build_modes(e.operator.ports),
        W=e.operator.W, G=e.primitive_G, responses=e.primitive_responses,
        ndofs=e.operator.ndofs, symmetry_error=e.symmetry_error,
        kernel_error=float(np.linalg.norm(e.primitive_G @ np.ones(e.primitive_G.shape[0]))
            / max(float(np.linalg.norm(e.primitive_G)), 1e-30))) for e in lib.entries]
    return DdpnmSolution(
        interface_pressures=np.array([sys.coefficients[k2d[(iid, "normal", "P0")]]
                                      for iid in range(ni)]),
        schur_matrix=sys.schur_matrix, rhs=sys.rhs, local_responses=lrs,
        local_solutions=sys.local_solutions,
        interface_flux_sums=np.array([sys.moment_residuals[k2d[(iid, "normal", "P0")]]
                                      for iid in range(ni)]),
        boundary_fluxes=sys.boundary_fluxes,
        min_schur_eigenvalue=sys.min_schur_eigenvalue,
        max_mass_residual=float(np.max(np.abs(sys.moment_residuals))))

# --- Classic ---
print("Classic-DDPNM..."); gc.collect(); tracemalloc.start()
t0 = time.perf_counter()
cl = build_response_library(p, CompatibleClassicP0Basis(), viscosity=1.0,
                            inlet_pressure=1.0, outlet_pressure=0.0)
c_off = time.perf_counter() - t0
t1 = time.perf_counter()
cs = InterfaceAssembler(cl).assemble(np.zeros(n_ifaces, dtype=np.int8))
c_on = time.perf_counter() - t1
c_sol = reduced_sol(cl, cs, n_ifaces)
_, cp = tracemalloc.get_traced_memory(); tracemalloc.stop()
cm, _, _ = finite_element_error_analysis(p, c_sol, ref, vols)
print(f"  offline={c_off:.1f}s  online={c_on:.4f}s  mem={cp/1024**2:.0f}MiB  "
      f"unknowns={len(cs.global_keys)}")
print(f"  L2(u)={cm['velocity_relative_l2']:.3%}  "
      f"H1={cm['velocity_relative_broken_h1_seminorm']:.3%}  "
      f"p={cm['pressure_raw_relative_l2']:.3%}  "
      f"flux={cm['outlet_flux_relative_error']:.3%}")

# --- Affine ---
print("Affine-DDPNM..."); gc.collect(); tracemalloc.start()
t0 = time.perf_counter()
al = build_response_library(p, AffineFaceBasis(p), viscosity=1.0,
                            inlet_pressure=1.0, outlet_pressure=0.0)
a_off = time.perf_counter() - t0
t1 = time.perf_counter()
a_sys = InterfaceAssembler(al).assemble(np.full(n_ifaces, 2, dtype=np.int8))
a_on = time.perf_counter() - t1
a_sol = reduced_sol(al, a_sys, n_ifaces)
_, ap = tracemalloc.get_traced_memory(); tracemalloc.stop()
am, _, _ = finite_element_error_analysis(p, a_sol, ref, vols)
print(f"  offline={a_off:.1f}s  online={a_on:.4f}s  mem={ap/1024**2:.0f}MiB  "
      f"unknowns={len(a_sys.global_keys)}")
print(f"  L2(u)={am['velocity_relative_l2']:.3%}  "
      f"H1={am['velocity_relative_broken_h1_seminorm']:.3%}  "
      f"p={am['pressure_raw_relative_l2']:.3%}  "
      f"flux={am['outlet_flux_relative_error']:.3%}")

# --- Save report ---
porosity = float(1.0 - np.sum(4.0 / 3.0 * np.pi * SPHERES[:, 3] ** 3))
report = {
    "partition": "voronoi",
    "mesh_cells": int(nc), "n_interfaces": int(n_ifaces),
    "n_regions": int(nr), "sphere_count": len(SPHERES), "porosity": porosity,
    "parameters": {"mesh_size": 0.12, "viscosity": 1.0,
                   "inlet_pressure": 1.0, "outlet_pressure": 0.0},
    "methods": {
        "Classic-DDPNM": {
            "global_unknowns": int(len(cs.global_keys)),
            "velocity_relative_l2": float(cm["velocity_relative_l2"]),
            "velocity_relative_broken_h1": float(cm["velocity_relative_broken_h1_seminorm"]),
            "pressure_relative_l2": float(cm["pressure_raw_relative_l2"]),
            "outlet_flux_relative_error": float(cm["outlet_flux_relative_error"]),
        },
        "Affine-DDPNM": {
            "global_unknowns": int(len(a_sys.global_keys)),
            "velocity_relative_l2": float(am["velocity_relative_l2"]),
            "velocity_relative_broken_h1": float(am["velocity_relative_broken_h1_seminorm"]),
            "pressure_relative_l2": float(am["pressure_raw_relative_l2"]),
            "outlet_flux_relative_error": float(am["outlet_flux_relative_error"]),
        },
        "Monolithic-FEM": {
            "global_unknowns": "-", "velocity_relative_l2": 0,
            "velocity_relative_broken_h1": 0, "pressure_relative_l2": 0,
            "outlet_flux_relative_error": 0,
        },
    },
    "timings": {
        "mesh_s": round(mesh_t, 1),
        "Classic-DDPNM": {"offline_s": round(c_off, 1), "online_s": round(c_on, 4),
                          "total_s": round(c_off + c_on, 1),
                          "peak_memory_mib": round(cp / 1024**2, 0)},
        "Affine-DDPNM": {"offline_s": round(a_off, 1), "online_s": round(a_on, 4),
                         "total_s": round(a_off + a_on, 1),
                         "peak_memory_mib": round(ap / 1024**2, 0)},
        "Monolithic-FEM": {"total_s": round(fem_t, 1),
                           "peak_memory_mib": round(fem_m / 1024**2, 0)},
    },
}
with open(OUT / "benchmark_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved {OUT}/benchmark_report.json")

# --- Print table ---
print()
print("=" * 110)
print(f"{'Method':<22s} {'unknowns':>8s} {'L2(u)':>8s} {'H1(u)':>8s} "
      f"{'L2(p)':>8s} {'flux':>8s} {'offline':>8s} {'online':>10s} "
      f"{'spd':>6s} {'mem':>8s}")
print("-" * 110)
for name, m, off, on, mem in [
    ("Classic-DDPNM", cm, c_off, c_on, cp),
    ("Affine-DDPNM", am, a_off, a_on, ap),
]:
    uk = report["methods"][name]["global_unknowns"]
    print(f"{name:<22s} {uk:>8d} "
          f"{m['velocity_relative_l2']*100:>7.2f}% "
          f"{m['velocity_relative_broken_h1_seminorm']*100:>7.2f}% "
          f"{m['pressure_raw_relative_l2']*100:>7.2f}% "
          f"{m['outlet_flux_relative_error']*100:>7.2f}% "
          f"{off:>7.1f}s {on:>9.4f}s {fem_t/on:>5.0f}x "
          f"{mem/1024**2:>7.0f}MiB")
print(f"{'Monolithic-FEM':<22s} {'-':>8s} {'ref':>8s} {'ref':>8s} "
      f"{'ref':>8s} {'ref':>8s} {'-':>8s} {fem_t:>9.1f}s {'1x':>6s} "
      f"{fem_m/1024**2:>7.0f}MiB")
print("=" * 110)
print(f"\nPorosity: {porosity*100:.1f}%  |  {len(SPHERES)} spheres  |  "
      f"r_max-r_min = {SPHERES[:,3].max()-SPHERES[:,3].min():.4f}")

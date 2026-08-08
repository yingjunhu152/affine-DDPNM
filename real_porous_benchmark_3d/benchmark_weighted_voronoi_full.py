#!/usr/bin/env python3
"""Complete 100-sphere Weighted-Voronoi benchmark — error table + cost table + contour plots."""
import sys, time, gc, tracemalloc, json, numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parent
for r in [PROJ.parent, PROJ.parent/"ddpnm_3d_uniform_spheres", PROJ.parent/"affine_ddpnm_3d_random_porous"]:
    if str(r) not in sys.path: sys.path.insert(0, str(r))

from geometry import build_partition_weighted_voronoi_occ, SPHERES
from ddpnm_core.fem_utils import solve_reference
from ddpnm_core.library import build_response_library
from ddpnm_core.assembler import InterfaceAssembler
from ddpnm_core.validation import finite_element_error_analysis
from affine_ddpnm_3d_random_porous.affine_face_basis import CompatibleClassicP0Basis, AffineFaceBasis
from ddpnm3d.solver import DdpnmSolution, LocalResponse, build_modes
from ddpnm3d.visualization import evaluate_fem_ddpnm_slice

OUT = PROJ / "outputs" / "benchmark_weighted_voronoi_100"
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Partition ──
print("=== 100-sphere Weighted Voronoi benchmark ===")
print(f"{len(SPHERES)} spheres, porosity={1-np.sum(4/3*np.pi*SPHERES[:,3]**3):.1%}")
print(f"r=[{SPHERES[:,3].min():.3f},{SPHERES[:,3].max():.3f}], r_max-r_min={SPHERES[:,3].max()-SPHERES[:,3].min():.3f}")
t0=time.perf_counter()
p = build_partition_weighted_voronoi_occ(mesh_size=0.12,sphere_size=0.05,boundary_size=0.07,
    interface_size=0.06,sphere_band=0.14,boundary_band=0.12,interface_band=0.10)
mesh=p.mesh; ni=len(p.interface_pairs); nc=mesh.topology.index_map(mesh.topology.dim).size_local
nr=len(set(int(l) for l in p.cell_labels))
mt=time.perf_counter()-t0
print(f"mesh: {nc}cells {ni}interfaces {nr}regions ({mt:.1f}s)")

# ── 2. FEM ──
print("FEM..."); gc.collect(); tracemalloc.start(); t0=time.perf_counter()
ref=solve_reference(mesh,viscosity=1.0,inlet_pressure=1.0,outlet_pressure=0.0,pressure_stabilization=1e-10)
ft=time.perf_counter()-t0; _,fm=tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"  {ft:.1f}s {fm/1024**2:.0f}MiB")

# volumes
td=mesh.topology.dim; mesh.topology.create_connectivity(td,0); c2=mesh.topology.connectivity(td,0)
vols=np.empty(nc)
for c in range(nc):
    vv=mesh.geometry.x[c2.links(c),:3]
    vols[c]=abs(np.linalg.det(np.stack([vv[1]-vv[0],vv[2]-vv[0],vv[3]-vv[0]],axis=1)))/6.0

def mk_sol(lib,sys,ni):
    keys=sys.global_keys; k2d={k:d for d,k in enumerate(keys)}
    lrs=[LocalResponse(pore_id=int(e.operator.pore_id),submesh=e.operator.submesh,
        parent_cell_map=e.operator.parent_cell_map,parent_vertex_map=e.operator.parent_vertex_map,
        ports=e.operator.ports,modes=build_modes(e.operator.ports),W=e.operator.W,G=e.primitive_G,
        responses=e.primitive_responses,ndofs=e.operator.ndofs,symmetry_error=e.symmetry_error,
        kernel_error=float(np.linalg.norm(e.primitive_G@np.ones(e.primitive_G.shape[0]))/
            max(float(np.linalg.norm(e.primitive_G)),1e-30))) for e in lib.entries]
    return DdpnmSolution(
        interface_pressures=np.array([sys.coefficients[k2d[(iid,'normal','P0')]] for iid in range(ni)]),
        schur_matrix=sys.schur_matrix,rhs=sys.rhs,local_responses=lrs,local_solutions=sys.local_solutions,
        interface_flux_sums=np.array([sys.moment_residuals[k2d[(iid,'normal','P0')]] for iid in range(ni)]),
        boundary_fluxes=sys.boundary_fluxes,min_schur_eigenvalue=sys.min_schur_eigenvalue,
        max_mass_residual=float(np.max(np.abs(sys.moment_residuals))))

# ── 3. Classic ──
print("Classic..."); gc.collect(); tracemalloc.start()
t0=time.perf_counter()
cl=build_response_library(p,CompatibleClassicP0Basis(),viscosity=1.0,inlet_pressure=1.0,outlet_pressure=0.0)
coff=time.perf_counter()-t0
_, coff_mem = tracemalloc.get_traced_memory(); tracemalloc.stop()
t1=time.perf_counter(); tracemalloc.start()
cs=InterfaceAssembler(cl).assemble(np.zeros(ni,dtype=np.int8)); con=time.perf_counter()-t1
csol=mk_sol(cl,cs,ni); _,con_mem=tracemalloc.get_traced_memory(); tracemalloc.stop()
cm,_,_=finite_element_error_analysis(p,csol,ref,vols)
cuk=len(cs.global_keys)
print(f"  off={coff:.1f}s on={con:.4f}s off_mem={coff_mem/1024**2:.0f}MiB on_mem={con_mem/1024**2:.0f}MiB uk={cuk}")
print(f"  L2(u)={cm['velocity_relative_l2']:.3%} H1={cm['velocity_relative_broken_h1_seminorm']:.3%} p={cm['pressure_raw_relative_l2']:.3%} flux={cm['outlet_flux_relative_error']:.3%}")

# ── 4. Affine ──
del cl, cs  # free Classic factorizations (keep csol/cm for slices)
from ddpnm_core.fem_utils import _parent_facet_lookup_cache
_parent_facet_lookup_cache.clear()
gc.collect()
print("Affine..."); tracemalloc.start()
t0=time.perf_counter()
al=build_response_library(p,AffineFaceBasis(p),viscosity=1.0,inlet_pressure=1.0,outlet_pressure=0.0)
aoff=time.perf_counter()-t0
_, aoff_mem = tracemalloc.get_traced_memory(); tracemalloc.stop()
t1=time.perf_counter(); tracemalloc.start()
a_sys=InterfaceAssembler(al).assemble(np.full(ni,2,dtype=np.int8)); aon=time.perf_counter()-t1
asol=mk_sol(al,a_sys,ni); _,aon_mem=tracemalloc.get_traced_memory(); tracemalloc.stop()
am,_,_=finite_element_error_analysis(p,asol,ref,vols)
auk=len(a_sys.global_keys)
print(f"  off={aoff:.1f}s on={aon:.4f}s off_mem={aoff_mem/1024**2:.0f}MiB on_mem={aon_mem/1024**2:.0f}MiB uk={auk}")
print(f"  L2(u)={am['velocity_relative_l2']:.3%} H1={am['velocity_relative_broken_h1_seminorm']:.3%} p={am['pressure_raw_relative_l2']:.3%} flux={am['outlet_flux_relative_error']:.3%}")

# ── 5. Slice fields ──
print("Slice...")
tetra=np.zeros((nc,4),dtype=np.int32)
for c in range(nc): tetra[c]=c2.links(c)
pts=mesh.geometry.x.copy()
csd=evaluate_fem_ddpnm_slice(p,csol,ref,pts,tetra,z_value=0.50)
asd=evaluate_fem_ddpnm_slice(p,asol,ref,pts,tetra,z_value=0.50)
np.savez(OUT/"slice_fields.npz",
    slice_points=csd["error_slice_points"],slice_triangles=csd["error_slice_triangles"],
    vertex_labels=np.asarray(p.cell_labels)[csd["error_slice_parent_cells"]],
    parent_cells=csd["error_slice_parent_cells"],
    sphere_centers=SPHERES[:,:3],sphere_radii=SPHERES[:,3],
    fem_speed=np.linalg.norm(csd["error_slice_u_fem"],axis=1),
    fem_pres=csd["error_slice_p_fem"],
    classic_err=np.abs(np.linalg.norm(csd["error_slice_u_ddpnm"],axis=1)-np.linalg.norm(csd["error_slice_u_fem"],axis=1)),
    classic_pres_err=np.abs(csd["error_slice_p_ddpnm"]-csd["error_slice_p_fem"]),
    affine_err=np.abs(np.linalg.norm(asd["error_slice_u_ddpnm"],axis=1)-np.linalg.norm(asd["error_slice_u_fem"],axis=1)),
    affine_pres_err=np.abs(asd["error_slice_p_ddpnm"]-asd["error_slice_p_fem"]),
    classic_l2=float(cm["velocity_relative_l2"]),affine_l2=float(am["velocity_relative_l2"]),
    classic_pl2=float(cm["pressure_raw_relative_l2"]),affine_pl2=float(am["pressure_raw_relative_l2"]),
    classic_flux=float(cm["outlet_flux_relative_error"]),affine_flux=float(am["outlet_flux_relative_error"]))
print("  saved slice_fields.npz")

# ── 6. Save report ──
por=float(1-np.sum(4/3*np.pi*SPHERES[:,3]**3))
report={
    "partition":"weighted_voronoi","mesh_cells":int(nc),"n_interfaces":int(ni),"n_regions":int(nr),
    "sphere_count":len(SPHERES),"porosity":por,
    "r_min":float(SPHERES[:,3].min()),"r_max":float(SPHERES[:,3].max()),
    "r_max_minus_r_min":float(SPHERES[:,3].max()-SPHERES[:,3].min()),
    "parameters":{"mesh_size":0.12,"viscosity":1.0,"inlet_pressure":1.0,"outlet_pressure":0.0},
    "methods":{
        "Classic-DDPNM":{"global_unknowns":cuk,
            "velocity_relative_l2":float(cm["velocity_relative_l2"]),
            "velocity_relative_broken_h1":float(cm["velocity_relative_broken_h1_seminorm"]),
            "pressure_relative_l2":float(cm["pressure_raw_relative_l2"]),
            "outlet_flux_relative_error":float(cm["outlet_flux_relative_error"])},
        "Affine-DDPNM":{"global_unknowns":auk,
            "velocity_relative_l2":float(am["velocity_relative_l2"]),
            "velocity_relative_broken_h1":float(am["velocity_relative_broken_h1_seminorm"]),
            "pressure_relative_l2":float(am["pressure_raw_relative_l2"]),
            "outlet_flux_relative_error":float(am["outlet_flux_relative_error"])},
        "Monolithic-FEM":{"global_unknowns":"-","velocity_relative_l2":0,"velocity_relative_broken_h1":0,"pressure_relative_l2":0,"outlet_flux_relative_error":0}},
    "timings":{
        "mesh_s":round(mt,1),
        "Classic-DDPNM":{"offline_s":round(coff,1),"online_s":round(con,4),"total_s":round(coff+con,1),
            "offline_memory_mib":round(coff_mem/1024**2,0),"online_memory_mib":round(con_mem/1024**2,0)},
        "Affine-DDPNM":{"offline_s":round(aoff,1),"online_s":round(aon,4),"total_s":round(aoff+aon,1),
            "offline_memory_mib":round(aoff_mem/1024**2,0),"online_memory_mib":round(aon_mem/1024**2,0)},
        "Monolithic-FEM":{"total_s":round(ft,1),"peak_memory_mib":round(fm/1024**2,0)}}}
with open(OUT/"benchmark_report.json","w") as f: json.dump(report,f,indent=2)
print(f"saved {OUT}/benchmark_report.json")

# ── 7. Print tables ──
def _mem_str(mib):
    if mib >= 1024: return f"{mib/1024:.1f}GiB"
    return f"{int(mib)}MiB"
print(); print("="*120)
print(f"{'Method':<20s} {'uk':>6s} {'L2(u)':>8s} {'H1(u)':>8s} {'L2(p)':>8s} {'flux':>8s} {'off_t':>7s} {'on_t':>9s} {'off_mem':>9s} {'on_mem':>9s} {'spd':>5s}")
print("-"*120)
for n,m,t in [("Classic-DDPNM",cm,(coff,con,coff_mem,con_mem)),("Affine-DDPNM",am,(aoff,aon,aoff_mem,aon_mem))]:
    print(f"{n:<20s} {report['methods'][n]['global_unknowns']:>6d} {m['velocity_relative_l2']*100:>7.2f}% {m['velocity_relative_broken_h1_seminorm']*100:>7.2f}% {m['pressure_raw_relative_l2']*100:>7.2f}% {m['outlet_flux_relative_error']*100:>7.2f}% {t[0]:>6.1f}s {t[1]:>8.4f}s {_mem_str(t[2]):>9s} {_mem_str(t[3]):>9s} {ft/t[1]:>4.0f}x")
print(f"{'Monolithic-FEM':<20s} {'-':>6s} {'ref':>8s} {'ref':>8s} {'ref':>8s} {'ref':>8s} {'-':>7s} {ft:>8.1f}s {'-':>9s} {_mem_str(fm):>9s} {'1x':>5s}")
print("="*120)
print(f"\nporosity={por:.1%} | r_max-r_min={SPHERES[:,3].max()-SPHERES[:,3].min():.4f} | weighted Voronoi (power diagram)")
print(f"mesh: {int(nc)} cells, {int(ni)} interfaces | outputs: {OUT}/")

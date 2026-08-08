#!/usr/bin/env python3
"""Error contour plots at z=0.5 — exact ddpnm3d style.

Per-partition 1×3 panel figure: FEM reference (viridis), Classic error (turbo),
Affine error (turbo), with subdomain interface lines + sphere cut-outs.

Imports and reuses the exact same helper functions as the ddpnm3d project:
_classify_slice_grid, _smooth_slice_grid, subdomain_interface_lines,
slice_sphere_cuts, sci_colorbar.
"""

from __future__ import annotations

import argparse, gc, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent

for root in [REPO_DIR, REPO_DIR / "ddpnm_3d_uniform_spheres",
             REPO_DIR / "affine_ddpnm_3d_random_porous"]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

# Reuse the exact same helpers from the ddpnm3d plotting modules
from plot_random_errors import (
    _classify_slice_grid, _smooth_slice_grid, slice_sphere_cuts, sci_colorbar,
)
from plot_partition_slice import subdomain_interface_lines
from ddpnm3d.visualization import evaluate_fem_ddpnm_slice

from geometry import SPHERES

N_GRID = 500  # fine grid matching ddpnm3d style
SIGMA = 4.0


# ---------------------------------------------------------------------------
# Per-partition runner
# ---------------------------------------------------------------------------

def build_and_evaluate(partition_name: str, out_dir: Path):
    """Build partition, run FEM+Classic+Affine, evaluate slice, return data."""
    from geometry import (build_partition_voronoi, build_partition_weighted_voronoi,
                          build_partition_watershed, build_partition_grid)
    from ddpnm_core.fem_utils import solve_reference
    from ddpnm_core.reconstruction import mixed_solution_functions
    from affine_ddpnm_3d_random_porous.affine_face_basis import (
        CompatibleClassicP0Basis, AffineFaceBasis,
    )
    from ddpnm_core.library import build_response_library
    from ddpnm_core.assembler import InterfaceAssembler

    kw = dict(mesh_size=0.12, sphere_size=0.05, boundary_size=0.07,
              sphere_band=0.14, boundary_band=0.12)

    print(f"=== {partition_name} ===")
    t0 = time.perf_counter()

    # ── Mesh ──
    if partition_name == "grid":
        p = build_partition_grid(**kw)
    elif partition_name == "voronoi":
        p = build_partition_voronoi(interface_size=0.06, interface_band=0.10, **kw)
    elif partition_name == "wvoronoi":
        p = build_partition_weighted_voronoi(**kw)
    else:
        import affine_ddpnm_3d_random_porous.random_porous as _rp
        import affine_ddpnm_3d_random_porous.watershed_partition as _wp
        _orig = _rp.SPHERES; _rp.SPHERES = SPHERES
        p = _wp.build_partition_watershed(
            bulk_size=kw["mesh_size"], sphere_size=kw["sphere_size"],
            boundary_size=kw["boundary_size"], sphere_band=kw["sphere_band"],
            boundary_band=kw["boundary_band"], policy="walls_and_spheres",
            abs_threshold=0.03, rel_threshold=0.06)
        _rp.SPHERES = _orig

    mesh = p.mesh
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    n_ifaces = len(p.interface_pairs)
    print(f"  mesh: {n_cells} cells, {n_ifaces} interfaces "
          f"({time.perf_counter()-t0:.1f}s)")

    # ── FEM ──
    print("  FEM..."); gc.collect(); t0 = time.perf_counter()
    ref = solve_reference(mesh, viscosity=1.0, inlet_pressure=1.0,
                          outlet_pressure=0.0, pressure_stabilization=1e-10)
    ft = time.perf_counter() - t0
    u_fem_fn, _ = mixed_solution_functions(ref.W, ref.solution)
    print(f"    {ft:.1f}s")

    # ── Tetrahedra + vertices for slice evaluation ──
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    n_c = mesh.topology.index_map(tdim).size_local
    tetrahedra = np.zeros((n_c, 4), dtype=np.int32)
    for c in range(n_c):
        tetrahedra[c] = c2v.links(c)
    mesh_pts = mesh.geometry.x.copy()

    # ── Classic ──
    print("  Classic..."); gc.collect()
    cb = CompatibleClassicP0Basis()
    clib = build_response_library(p, cb, viscosity=1.0, inlet_pressure=1.0,
                                  outlet_pressure=0.0)
    csys = InterfaceAssembler(clib).assemble(np.zeros(n_ifaces, dtype=np.int8))
    csol = _build_solution(clib, csys, n_ifaces)
    classic_slice = evaluate_fem_ddpnm_slice(
        p, csol, ref, mesh_pts, tetrahedra, z_value=0.50)
    classic_speed = np.linalg.norm(classic_slice["error_slice_u_fem"], axis=1)
    classic_err = np.abs(
        np.linalg.norm(classic_slice["error_slice_u_ddpnm"], axis=1) - classic_speed)
    c_l2 = float(np.sqrt(
        np.sum(classic_err**2) / max(np.sum(classic_speed**2), 1e-30)))
    print(f"    L2={c_l2:.3%}")

    # ── Affine ──
    print("  Affine..."); gc.collect()
    ab = AffineFaceBasis(p)
    alib = build_response_library(p, ab, viscosity=1.0, inlet_pressure=1.0,
                                  outlet_pressure=0.0)
    asys = InterfaceAssembler(alib).assemble(np.full(n_ifaces, 2, dtype=np.int8))
    asol = _build_solution(alib, asys, n_ifaces)
    affine_slice = evaluate_fem_ddpnm_slice(
        p, asol, ref, mesh_pts, tetrahedra, z_value=0.50)
    affine_speed = np.linalg.norm(affine_slice["error_slice_u_fem"], axis=1)
    affine_err = np.abs(
        np.linalg.norm(affine_slice["error_slice_u_ddpnm"], axis=1) - affine_speed)
    a_l2 = float(np.sqrt(
        np.sum(affine_err**2) / max(np.sum(affine_speed**2), 1e-30)))
    print(f"    L2={a_l2:.3%}")

    # ── Extract slice data for the grid interpolation pipeline ──
    # Use the classic slice triangulation for consistency (same mesh)
    slice_pts = classic_slice["error_slice_points"]
    slice_tris = classic_slice["error_slice_triangles"]
    parent_cells = classic_slice["error_slice_parent_cells"]
    vertex_labels = np.asarray(p.cell_labels)[parent_cells]

    all_spheres = SPHERES
    sphere_centers = all_spheres[:, :3].copy()
    sphere_radii = all_spheres[:, 3].copy()

    # FEM speed + pressure at slice vertices
    fem_speed_vert = np.linalg.norm(classic_slice["error_slice_u_fem"], axis=1)
    fem_pres_vert = classic_slice["error_slice_p_fem"]
    classic_pres_err = np.abs(classic_slice["error_slice_p_ddpnm"] - fem_pres_vert)
    affine_pres_err = np.abs(affine_slice["error_slice_p_ddpnm"] - fem_pres_vert)

    # Pressure L2 errors
    cp_l2 = float(np.sqrt(np.sum(classic_pres_err**2) / max(np.sum(fem_pres_vert**2), 1e-30)))
    ap_l2 = float(np.sqrt(np.sum(affine_pres_err**2) / max(np.sum(fem_pres_vert**2), 1e-30)))
    print(f"    p-L2: Classic={cp_l2:.3%}  Affine={ap_l2:.3%}")

    # Flux errors from full FEM error analysis
    from ddpnm_core.validation import finite_element_error_analysis
    # Build cell volumes for error analysis
    tdim = mesh.topology.dim; mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0); n_c = mesh.topology.index_map(tdim).size_local
    vols = np.empty(n_c)
    for c in range(n_c):
        vv = mesh.geometry.x[c2v.links(c),:3]
        vols[c] = abs(np.linalg.det(np.stack([vv[1]-vv[0],vv[2]-vv[0],vv[3]-vv[0]],axis=1)))/6.0
    cm,_,_ = finite_element_error_analysis(p, csol, ref, vols)
    am,_,_ = finite_element_error_analysis(p, asol, ref, vols)
    c_flux = cm["outlet_flux_relative_error"]
    a_flux = am["outlet_flux_relative_error"]
    c_pl2_full = cm.get("pressure_raw_relative_l2", cm.get("pressure_relative_l2", float("nan")))
    a_pl2_full = am.get("pressure_raw_relative_l2", am.get("pressure_relative_l2", float("nan")))
    print(f"    flux: Classic={c_flux:.3%}  Affine={a_flux:.3%}")
    print(f"    p-L2 (full): Classic={c_pl2_full:.3%}  Affine={a_pl2_full:.3%}")

    npz_path = out_dir / f"slice_{partition_name}.npz"
    np.savez(npz_path,
             slice_points=slice_pts, slice_triangles=slice_tris,
             vertex_labels=vertex_labels, parent_cells=parent_cells,
             sphere_centers=sphere_centers, sphere_radii=sphere_radii,
             fem_speed=fem_speed_vert, fem_pres=fem_pres_vert,
             classic_err=classic_err, classic_pres_err=classic_pres_err,
             affine_err=affine_err, affine_pres_err=affine_pres_err,
             classic_l2=c_l2, affine_l2=a_l2,
             classic_pl2=c_pl2_full, affine_pl2=a_pl2_full,
             classic_flux=c_flux, affine_flux=a_flux)
    print(f"  Saved {npz_path.name}")
    return str(npz_path)


def _build_solution(lib, sys, ni):
    from ddpnm3d.solver import DdpnmSolution, LocalResponse, build_modes
    keys = sys.global_keys; k2d = {k: d for d, k in enumerate(keys)}
    lrs = [
        LocalResponse(
            pore_id=int(e.operator.pore_id), submesh=e.operator.submesh,
            parent_cell_map=e.operator.parent_cell_map,
            parent_vertex_map=e.operator.parent_vertex_map,
            ports=e.operator.ports, modes=build_modes(e.operator.ports),
            W=e.operator.W, G=e.primitive_G, responses=e.primitive_responses,
            ndofs=e.operator.ndofs, symmetry_error=e.symmetry_error,
            kernel_error=float(
                np.linalg.norm(e.primitive_G @ np.ones(e.primitive_G.shape[0]))
                / max(float(np.linalg.norm(e.primitive_G)), 1e-30)))
        for e in lib.entries
    ]
    return DdpnmSolution(
        interface_pressures=np.array(
            [sys.coefficients[k2d[(iid, "normal", "P0")]] for iid in range(ni)]),
        schur_matrix=sys.schur_matrix, rhs=sys.rhs, local_responses=lrs,
        local_solutions=sys.local_solutions,
        interface_flux_sums=np.array(
            [sys.moment_residuals[k2d[(iid, "normal", "P0")]] for iid in range(ni)]),
        boundary_fluxes=sys.boundary_fluxes,
        min_schur_eigenvalue=sys.min_schur_eigenvalue,
        max_mass_residual=float(np.max(np.abs(sys.moment_residuals))))


# ---------------------------------------------------------------------------
# Plotting (ddpnm3d style)
# ---------------------------------------------------------------------------

def plot_one(npz_path: str, out_dir: Path):
    """2×3 panel figure — velocity (top row) + pressure (bottom row)."""
    data = np.load(npz_path)
    slice_pts = data["slice_points"]
    slice_tris = data["slice_triangles"]
    vertex_labels = data["vertex_labels"]
    sphere_c = data["sphere_centers"]
    sphere_r = data["sphere_radii"]
    fem_speed = data["fem_speed"]
    fem_pres = data["fem_pres"]
    classic_err = data["classic_err"]
    classic_pres_err = data["classic_pres_err"]
    affine_err = data["affine_err"]
    affine_pres_err = data["affine_pres_err"]
    c_l2 = float(data["classic_l2"])
    a_l2 = float(data["affine_l2"])
    c_pl2 = float(data["classic_pl2"])
    a_pl2 = float(data["affine_pl2"])
    c_flux = float(data["classic_flux"])
    a_flux = float(data["affine_flux"])

    z_val = 0.50
    part_name = Path(npz_path).stem.replace("slice_", "")

    # ── Classify onto fine grid + strong smooth (no visible mesh seams) ──
    X, Y, cell_of_point, usable = _classify_slice_grid(
        slice_pts, slice_tris, vertex_labels, z_val, sphere_c, sphere_r, n=N_GRID)

    def _smooth(vals):
        return _smooth_slice_grid(vals, slice_pts, vertex_labels,
                                  X, Y, cell_of_point, usable, n=N_GRID, sigma=SIGMA)

    fem_v_grid  = _smooth(fem_speed);      classic_v_grid = _smooth(classic_err)
    affine_v_grid = _smooth(affine_err);   fem_p_grid     = _smooth(fem_pres)
    classic_p_grid = _smooth(classic_pres_err); affine_p_grid = _smooth(affine_pres_err)

    v_max = float(np.nanmax(fem_v_grid))
    cv_max = float(np.nanpercentile(np.abs(classic_v_grid), 98.0))
    av_max = float(np.nanpercentile(np.abs(affine_v_grid), 98.0))
    p_max = float(np.nanmax(np.abs(fem_p_grid)))
    cp_max = float(np.nanpercentile(np.abs(classic_p_grid), 98.0))
    ap_max = float(np.nanpercentile(np.abs(affine_p_grid), 98.0))

    fig, axes = plt.subplots(2, 3, figsize=(17.0, 11.0))
    fig.subplots_adjust(left=0.045, right=0.97, bottom=0.06, top=0.96,
                        wspace=0.27, hspace=0.22)

    panels_2d = [
        (0, 0, fem_v_grid, "viridis", 0.0, v_max, f"FEM $|u|$", False),
        (0, 1, classic_v_grid, "turbo", 0.0, cv_max, f"Classic $|u|$ err  $L^2$={c_l2:.1%}", False),
        (0, 2, affine_v_grid, "turbo", 0.0, av_max, f"Affine $|u|$ err  $L^2$={a_l2:.1%}", False),
        (1, 0, fem_p_grid, "RdBu_r", -p_max, p_max, f"FEM $p$", True),
        (1, 1, classic_p_grid, "turbo", 0.0, cp_max, f"Classic $|p|$ err  $L^2$={c_pl2:.1%}  flux={c_flux:.1%}", False),
        (1, 2, affine_p_grid, "turbo", 0.0, ap_max, f"Affine $|p|$ err  $L^2$={a_pl2:.1%}  flux={a_flux:.1%}", False),
    ]

    for row, col, values, cmap, vmin, vmax, title, signed in panels_2d:
        ax = axes[row, col]
        artist = ax.imshow(values, extent=[0, 1, 0, 1], origin="lower",
                           cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                           interpolation="bilinear")
        slice_sphere_cuts(ax, sphere_c, sphere_r, z_val, color="#cccccc")
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0]); ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=10)
        ax.text(0.5, -0.075, f"({chr(ord('a') + row * 3 + col)})",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=12, fontfamily="serif")
        sci_colorbar(fig, artist, ax, signed=signed)

    fig.text(0.012, 0.78, "velocity", rotation=90, va="center", ha="center",
             fontsize=11, fontstyle="italic")
    fig.text(0.012, 0.30, "pressure", rotation=90, va="center", ha="center",
             fontsize=11, fontstyle="italic")
    fig.suptitle(f"{part_name} partition — $z=0.5$ slice", fontsize=13,
                 fontweight="bold", y=0.99)
    fig.savefig(out_dir / f"slice_errors_{part_name}.png", dpi=240)
    plt.close(fig)
    print(f"  Saved slice_errors_{part_name}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", choices=("grid", "voronoi", "wvoronoi", "watershed"),
                    default="grid")
    ap.add_argument("--out-dir", default="outputs/slices")
    ap.add_argument("--build-only", action="store_true",
                    help="only build+save NPZ, skip plotting")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    npz = build_and_evaluate(args.partition, out)
    if not args.build_only:
        plot_one(npz, out)


if __name__ == "__main__":
    main()

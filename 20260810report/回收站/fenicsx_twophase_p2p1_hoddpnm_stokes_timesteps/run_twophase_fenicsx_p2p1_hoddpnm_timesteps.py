from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from classic_ddpnm.linalg import to_numpy_vector, to_scipy_matrix
from run_cube_holes_hoddpnm_validation import MeshData, build_cube_minus_spheres_mesh
from run_random_pnm_taylor_hood_validation import (
    mean_aligned_difference,
    scalar_error_stats,
    select_pnm_edges,
    solve_schur_hoddpnm,
    taylor_hood_bcs,
    taylor_hood_interface_mask,
    vector_error_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holes-per-axis", type=int, default=2)
    parser.add_argument("--cells-per-axis", type=int, default=3)
    parser.add_argument("--domain-size", type=float, default=5.0)
    parser.add_argument("--radius", type=float, default=0.34)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--mu-original", type=float, default=1.0)
    parser.add_argument("--mu-injected", type=float, default=4.0)
    parser.add_argument("--mu-water", type=float, default=1.0)
    parser.add_argument("--mu-oil", type=float, default=5.0)
    parser.add_argument("--sw-residual", type=float, default=0.2)
    parser.add_argument("--so-residual", type=float, default=0.2)
    parser.add_argument("--corey-nw", type=float, default=2.0)
    parser.add_argument("--corey-no", type=float, default=2.0)
    parser.add_argument("--residual-original", type=float, default=0.05)
    parser.add_argument("--residual-injected", type=float, default=0.05)
    parser.add_argument("--sw-initial", type=float, default=None)
    parser.add_argument("--sw-inlet", type=float, default=None)
    parser.add_argument("--transport-scale", type=float, default=0.18)
    parser.add_argument("--geometry-channel-strength", type=float, default=0.0)
    parser.add_argument("--capillary-spread", type=float, default=0.0)
    parser.add_argument("--cfl-limit", type=float, default=0.5)
    parser.add_argument("--target-degree", type=int, default=4)
    parser.add_argument("--max-edge-length", type=float, default=2.35)
    parser.add_argument("--pressure-stabilization", type=float, default=1.0e-10)
    parser.add_argument("--pressure-boundary-mode", choices=("interface-anchors", "all"), default="interface-anchors")
    parser.add_argument("--schur-solver", choices=("gmres", "dense"), default="gmres")
    parser.add_argument("--schur-rtol", type=float, default=1.0e-7)
    parser.add_argument("--schur-atol", type=float, default=0.0)
    parser.add_argument("--schur-maxiter", type=int, default=600)
    parser.add_argument("--schur-restart", type=int, default=80)
    parser.add_argument("--schur-preconditioner", choices=("ilu", "diag", "none"), default="ilu")
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    parser.add_argument("--plot-steps", type=int, nargs="*", default=None)
    parser.add_argument("--render-resolution", type=int, default=86)
    parser.add_argument("--render-geometry-strength", type=float, default=1.0)
    parser.add_argument("--render-front-contrast", type=float, default=1.25)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/fenicsx_twophase_timesteps_smoke"))
    args = parser.parse_args()
    args.sw_initial_value = args.residual_original if args.sw_initial is None else args.sw_initial
    args.sw_inlet_value = 1.0 - args.residual_injected if args.sw_inlet is None else args.sw_inlet
    if not (0.0 <= args.sw_initial_value <= 1.0 and 0.0 <= args.sw_inlet_value <= 1.0):
        raise SystemExit("--sw-initial and --sw-inlet must be in [0, 1]")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mesh = build_cube_minus_spheres_mesh(
        holes_per_axis=args.holes_per_axis,
        cells_per_axis=args.cells_per_axis,
        domain_size=args.domain_size,
        radius=args.radius,
        seed=args.seed,
    )
    pnm_edges = select_pnm_edges(mesh.centers, args.target_degree, args.max_edge_length)
    graph_edges = mesh_vertex_edges(mesh.cells)
    lumped_volume = vertex_lumped_volume(mesh.coords, mesh.cells)

    saturation = np.full(len(mesh.coords), args.sw_initial_value, dtype=float)
    inlet_vertices = np.flatnonzero(np.isclose(mesh.coords[:, 0], 0.0))
    outlet_vertices = np.flatnonzero(np.isclose(mesh.coords[:, 0], args.domain_size))
    inlet_area = boundary_vertex_area(mesh.coords, inlet_vertices, args.domain_size)
    outlet_area = boundary_vertex_area(mesh.coords, outlet_vertices, args.domain_size)

    rows: list[dict[str, object]] = []
    plot_steps = selected_plot_steps(args.steps, args.plot_steps)
    manifest: list[dict[str, object]] = []
    if args.plot and 0 in plot_steps:
        out = args.out_dir / "voidspace_two_phase_volume_split_clean_step_0000.png"
        render_twophase_volume_frame(mesh, {"saturation": saturation.copy()}, args.render_resolution, args, out)
        manifest.append(
            {
                "step": 0,
                "time": 0.0,
                "mean_injected_phase_saturation": float(np.mean(saturation)),
                "out": str(out),
            }
        )
    for step in range(1, args.steps + 1):
        mass_before = float(np.sum(saturation * lumped_volume))
        t0 = time.perf_counter()
        assembled = assemble_fenicsx_stokes_system(mesh, pnm_edges, saturation, args)
        assemble_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        full_solution = spsolve(assembled["A"], assembled["b"])
        full_time = time.perf_counter() - t0
        if not np.all(np.isfinite(full_solution)):
            raise RuntimeError("Full FEniCSx-assembled solve produced non-finite values.")

        t0 = time.perf_counter()
        hodd = solve_schur_hoddpnm(
            assembled["A"],
            assembled["b"],
            assembled["interface_mask"],
            assembled["fixed_dofs"],
            args,
        )
        hodd_time = time.perf_counter() - t0
        hodd_solution = hodd["solution"]

        full_velocity_p2 = full_solution[assembled["mapV"]].reshape((-1, 3))
        hodd_velocity_p2 = hodd_solution[assembled["mapV"]].reshape((-1, 3))
        full_pressure = full_solution[assembled["mapQ"]]
        hodd_pressure = hodd_solution[assembled["mapQ"]]
        velocity_diff = hodd_velocity_p2 - full_velocity_p2
        pressure_diff = hodd_pressure - full_pressure
        pressure_diff_aligned = mean_aligned_difference(hodd_pressure, full_pressure)

        vertex_velocity = velocity_on_mesh_vertices(
            assembled["V_coords"],
            hodd_velocity_p2,
            mesh.coords,
        )
        saturation, mass_diag = update_saturation(
            mesh,
            graph_edges,
            lumped_volume,
            saturation,
            vertex_velocity,
            inlet_area,
            outlet_area,
            args,
            mass_before,
        )
        mass_after_clipping = float(np.sum(saturation * lumped_volume))
        mass_after = float(np.sum(saturation * lumped_volume))
        inlet_reset_mass_change = 0.0
        inlet_flux_mass = float(mass_diag["inlet_boundary_flux_mass"])
        outlet_flux_mass = float(mass_diag["outlet_boundary_flux_mass"])
        net_flux_mass = inlet_flux_mass - outlet_flux_mass
        expected_mass_after = (
            mass_before
            + mass_diag["graph_transport_mass_change"]
            + mass_diag["boundary_flux_mass_change"]
            + mass_diag["clipping_mass_change"]
        )
        mass_balance_residual = mass_after - expected_mass_after
        mass_error = relative_mass_error(mass_balance_residual, expected_mass_after)
        conservation_residual = mass_after - mass_before - net_flux_mass
        conservation_relative_error = relative_mass_error(conservation_residual, mass_after)
        saturation_stats = saturation_diagnostics(saturation, args)

        if args.plot and step in plot_steps:
            out = args.out_dir / f"voidspace_two_phase_volume_split_clean_step_{step:04d}.png"
            render_twophase_volume_frame(mesh, {"saturation": saturation.copy()}, args.render_resolution, args, out)
            manifest.append(
                {
                    "step": int(step),
                    "time": float(step * args.dt),
                    "mean_injected_phase_saturation": float(np.mean(saturation)),
                    "out": str(out),
                }
            )

        row = {
            "step": int(step),
            "mean_S2": float(np.mean(saturation)),
            "mean_injected_saturation": float(np.mean(saturation)),
            "saturation_mass": mass_after,
            "mass_before": float(mass_before),
            "mass_after_graph_transport": float(mass_diag["mass_after_graph_transport"]),
            "mass_after_boundary_flux": float(mass_diag["mass_after_boundary_flux"]),
            "mass_after_clipping": float(mass_after_clipping),
            "mass_error": float(mass_error),
            "mass_error_abs": float(abs(mass_error)),
            "mass_balance_residual": float(mass_balance_residual),
            "mass_balance_residual_abs": float(abs(mass_balance_residual)),
            "mass_residual": float(conservation_residual),
            "relative_mass_error": float(conservation_relative_error),
            "graph_transport_mass_change": float(mass_diag["graph_transport_mass_change"]),
            "boundary_flux_mass_change": float(mass_diag["boundary_flux_mass_change"]),
            "clipping_mass_change": float(mass_diag["clipping_mass_change"]),
            "inlet_reset_mass_change": float(inlet_reset_mass_change),
            "inlet_boundary_flux_mass": float(inlet_flux_mass),
            "outlet_boundary_flux_mass": float(outlet_flux_mass),
            "inlet_flux_mass": float(inlet_flux_mass),
            "outlet_flux_mass": float(outlet_flux_mass),
            "net_flux_mass": float(net_flux_mass),
            "expected_mass_after": float(expected_mass_after),
            "dt": float(args.dt),
            "max_cfl": float(mass_diag["max_cfl"]),
            "stable_dt": float(mass_diag["stable_dt"]),
            "cfl_number": float(mass_diag["cfl_number"]),
            "max_water_cfl": float(mass_diag["max_water_cfl"]),
            "max_inlet_flux_cfl": float(mass_diag["max_inlet_flux_cfl"]),
            "max_outlet_flux_cfl": float(mass_diag["max_outlet_flux_cfl"]),
            "max_outgoing_flux": float(mass_diag["max_outgoing_flux"]),
            "max_outgoing_water_flux": float(mass_diag["max_outgoing_water_flux"]),
            "saturation_min": float(saturation_stats["min"]),
            "saturation_max": float(saturation_stats["max"]),
            "saturation_effective_min": float(saturation_stats["effective_min"]),
            "saturation_effective_max": float(saturation_stats["effective_max"]),
            "assemble_time_seconds": float(assemble_time),
            "full_stokes_solve_time_seconds": float(full_time),
            "hoddpnm_schur_solve_time_seconds": float(hodd_time),
            "total_time_seconds": float(assemble_time + full_time + hodd_time),
            "velocity_error": vector_error_stats(velocity_diff, full_velocity_p2)["l2_rel"],
            "pressure_error": scalar_error_stats(pressure_diff_aligned, full_pressure - np.mean(full_pressure))["l2_rel"],
            "velocity_l2_rel_difference_vs_full_discrete": vector_error_stats(velocity_diff, full_velocity_p2)["l2_rel"],
            "pressure_l2_rel_difference_vs_full_discrete": scalar_error_stats(pressure_diff, full_pressure)["l2_rel"],
            "pressure_l2_rel_mean_aligned_difference_vs_full_discrete": scalar_error_stats(pressure_diff_aligned, full_pressure - np.mean(full_pressure))["l2_rel"],
            "pressure_min": float(np.min(full_pressure)),
            "pressure_max": float(np.max(full_pressure)),
            "schur_solver": hodd["schur_solver"],
            "schur_preconditioner": hodd["schur_preconditioner"],
            "dense_schur_used": bool(hodd["dense_schur_used"]),
            "schur_iterations": hodd["schur_iterations"],
            "schur_relative_residual": hodd["schur_relative_residual"],
            "mixed_dofs": int(len(full_solution)),
            "velocity_dofs": int(len(assembled["mapV"])),
            "pressure_dofs": int(len(assembled["mapQ"])),
            "hoddpnm_interface_dofs": int(np.count_nonzero(assembled["interface_mask"])),
            "hoddpnm_active_schur_dofs": int(hodd["n_boundary"]),
            "hoddpnm_active_dof_ratio": float(hodd["n_boundary"] / len(full_solution)),
            "hoddpnm_known_fixed_dofs": int(hodd["n_fixed_known"]),
            "hoddpnm_free_interior_dofs_eliminated": int(hodd["n_interior"]),
            "hoddpnm_eliminated_dofs": int(hodd["n_interior"]),
            "pressure_boundary_dofs": int(assembled["interface_stats"]["pressure_boundary_dofs"]),
            "pressure_interior_dofs_eliminated": int(assembled["interface_stats"]["pressure_interior_dofs_eliminated"]),
        }
        rows.append(row)
        print(
            f"step {step:04d}: mean S2={row['mean_S2']:.4f}, "
            f"full={full_time:.4f}s, hodd={hodd_time:.4f}s, "
            f"mass_err_rel={row['mass_error']:.2e}, "
            f"dM(graph/clip/in/out)="
            f"{row['graph_transport_mass_change']:.2e}/"
            f"{row['clipping_mass_change']:.2e}/"
            f"{row['inlet_flux_mass']:.2e}/"
            f"{row['outlet_flux_mass']:.2e}, "
            f"u_diff={row['velocity_error']:.2e}, "
            f"p_align={row['pressure_error']:.2e}, "
            f"schur={row['schur_solver']}/{row['schur_iterations']}"
        )

    summary = {
        "method": "FEniCSx P2-P1 two-phase HODDPNM timestep prototype",
        "assembly": "each timestep assembles a FEniCSx/dolfinx UFL P2-P1 Stokes system with cellwise mu(S2)",
        "equation": "-div(mu(Sw) grad(u)) + grad(p) = 0, div(u) = 0 at each timestep",
        "fractional_flow": "Corey krw/kro mobilities are used for fw(Sw) in the saturation transport and for effective viscosity mu_eff=1/(lambda_w+lambda_o)",
        "transport_scope": "Saturation is still advanced on the mesh-vertex graph, now with Corey fractional-flow upwinding, inlet boundary-flux injection, outlet natural outflow, and CFL diagnostics. This is not yet the final cell-wise conservative finite-volume face-flux discretization.",
        "boundary_condition_scope": "Sw is initialized uniformly. The inlet saturation is imposed as an incoming boundary flux with Sw_in, the outlet is natural outflow, and inlet vertices are not reset after each timestep.",
        "mass_diagnostic_scope": "mass_error is the bookkeeping residual after separately accounting for internal graph transport, boundary flux, and clipping. mass_residual/relative_mass_error check M_after - M_before - inlet_flux_mass + outlet_flux_mass for the current graph transport model, so clipping appears as a physical mass residual.",
        "reduction_scope": "known Dirichlet dofs are removed before the Schur solve; hoddpnm_active_schur_dofs is the free Schur boundary size, while hoddpnm_interface_dofs is the pre-fixed-removal interface count",
        "error_scope": "velocity/pressure differences are Schur/full consistency for each FEniCSx-assembled timestep matrix",
        "schur_scope": "matrix-free Schur GMRES is available and used by default; dense Schur is optional only for small validation-scale diagnostics",
        "holes": int(args.holes_per_axis**3),
        "cells_per_axis": int(args.cells_per_axis),
        "vertices": int(len(mesh.coords)),
        "tets": int(len(mesh.cells)),
        "steps": int(args.steps),
        "pressure_stabilization": float(args.pressure_stabilization),
        "pressure_boundary_mode": args.pressure_boundary_mode,
        "corey_parameters": {
            "sw_residual": float(args.sw_residual),
            "so_residual": float(args.so_residual),
            "corey_nw": float(args.corey_nw),
            "corey_no": float(args.corey_no),
            "mu_water": float(args.mu_water),
            "mu_oil": float(args.mu_oil),
        },
        "transport_boundary_conditions": {
            "sw_initial": float(args.sw_initial_value),
            "sw_inlet": float(args.sw_inlet_value),
            "inlet": "incoming boundary flux on x=0; no saturation reset",
            "outlet": "natural outflow on x=domain_size",
            "walls": "no graph boundary flux",
        },
        "cfl_limit": float(args.cfl_limit),
        "history": rows,
    }
    (args.out_dir / "twophase_fenicsx_p2p1_hoddpnm_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_history(args.out_dir / "history.csv", rows)
    if args.plot:
        (args.out_dir / "timesteps_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        plot_hoddpnm_vs_full_error(rows, args.out_dir / "hoddpnm_vs_full_error_history.png")
        plot_mass_conservation(rows, args.out_dir / "mass_conservation_history.png")
        plot_mean_saturation(rows, args.out_dir / "mean_saturation_history.png")
        plot_active_dofs(rows, args.out_dir / "active_dofs_history.png")
        plot_cfl(rows, args.out_dir / "cfl_history.png")
    print(f"wrote {args.out_dir / 'twophase_fenicsx_p2p1_hoddpnm_report.json'}")


def assemble_fenicsx_stokes_system(
    mesh: MeshData,
    pnm_edges: list[tuple[int, int]],
    saturation: np.ndarray,
    args,
) -> dict[str, object]:
    domain_ufl = ufl.Mesh(element("Lagrange", "tetrahedron", 1, shape=(3,)))
    msh = dmesh.create_mesh(MPI.COMM_SELF, mesh.cells, domain_ufl, mesh.coords)
    cell = msh.basix_cell()
    velocity_el = element("Lagrange", cell, 2, shape=(3,))
    pressure_el = element("Lagrange", cell, 1)
    W = fem.functionspace(msh, mixed_element([velocity_el, pressure_el]))
    V, mapV = W.sub(0).collapse()
    Q, mapQ = W.sub(1).collapse()

    mu_el = element("DG", cell, 0)
    M = fem.functionspace(msh, mu_el)
    mu = fem.Function(M)
    cell_saturation = np.mean(saturation[mesh.cells], axis=1)
    cell_mu = effective_viscosity(cell_saturation, args)
    if len(mu.x.array) != len(cell_mu):
        raise RuntimeError(f"DG0 viscosity dofs ({len(mu.x.array)}) do not match mesh cells ({len(cell_mu)}).")
    mu.x.array[:] = cell_mu

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a = fem.form(
        mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
        - fem.Constant(msh, args.pressure_stabilization) * p * q * ufl.dx
    )
    L = fem.form(fem.Constant(msh, 0.0) * q * ufl.dx)

    bcs, fixed_dofs = taylor_hood_bcs(
        msh,
        W,
        V,
        Q,
        mapQ,
        mesh.domain_size,
        mesh.centers,
        mesh.radius,
        mesh.domain_size / mesh.cells_per_axis,
    )
    A = to_scipy_matrix(fem.assemble_matrix(a, bcs=bcs))
    b = to_numpy_vector(fem.assemble_vector(L))
    fem.apply_lifting(b, [a], [bcs])
    fem.set_bc(b, bcs)

    interface_mask, interface_stats = taylor_hood_interface_mask(
        W.dofmap.index_map.size_local * W.dofmap.index_map_bs,
        np.asarray(mapV, dtype=np.int64),
        V.tabulate_dof_coordinates(),
        np.asarray(mapQ, dtype=np.int64),
        Q.tabulate_dof_coordinates(),
        mesh.coords,
        mesh.cells,
        mesh.domain_size,
        mesh.cells_per_axis,
        mesh.centers,
        pnm_edges,
        args.pressure_boundary_mode,
    )
    return {
        "A": A,
        "b": b,
        "interface_mask": interface_mask,
        "fixed_dofs": fixed_dofs,
        "mapV": np.asarray(mapV, dtype=np.int64),
        "mapQ": np.asarray(mapQ, dtype=np.int64),
        "V_coords": V.tabulate_dof_coordinates(),
        "Q_coords": Q.tabulate_dof_coordinates(),
        "interface_stats": interface_stats,
    }


def effective_viscosity(saturation: np.ndarray, args) -> np.ndarray:
    lambda_w, lambda_o = phase_mobilities(saturation, args)
    return 1.0 / np.maximum(lambda_w + lambda_o, 1.0e-12)


def velocity_on_mesh_vertices(V_coords: np.ndarray, velocity_p2: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    tree = cKDTree(V_coords)
    _, idx = tree.query(vertices, k=1)
    return velocity_p2[idx]


def field_on_mesh_vertices(field_coords: np.ndarray, values: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    tree = cKDTree(field_coords)
    _, idx = tree.query(vertices, k=1)
    return values[idx]


def update_saturation(
    mesh,
    edges,
    lumped_volume,
    saturation,
    velocity,
    inlet_area,
    outlet_area,
    args,
    mass_before: float,
) -> tuple[np.ndarray, dict[str, float]]:
    ds = np.zeros_like(saturation)
    outgoing_total = np.zeros_like(saturation)
    outgoing_water = np.zeros_like(saturation)
    for i, j in edges:
        delta = mesh.coords[j] - mesh.coords[i]
        midpoint = 0.5 * (mesh.coords[i] + mesh.coords[j])
        channel_gain = 1.0 + args.geometry_channel_strength * near_sphere_surface_weight(midpoint, mesh.centers, mesh.radius)
        q = args.transport_scale * channel_gain * dot3(0.5 * (velocity[i] + velocity[j]), delta)
        if q >= 0:
            donor, receiver = i, j
        else:
            donor, receiver = j, i
            q = -q
        water_flux = q * float(fractional_flow(saturation[donor], args))
        outgoing_total[donor] += q
        outgoing_water[donor] += water_flux
        ds[donor] -= water_flux / lumped_volume[donor]
        ds[receiver] += water_flux / lumped_volume[receiver]
        spread = args.capillary_spread * channel_gain * max(float(saturation[donor] - saturation[receiver]), 0.0)
        ds[donor] -= spread / lumped_volume[donor]
        ds[receiver] += spread / lumped_volume[receiver]
    graph_raw = saturation + args.dt * ds
    graph_transport_mass_change = float(np.sum((graph_raw - saturation) * lumped_volume))
    mass_after_graph_transport = mass_before + graph_transport_mass_change

    inlet_rate = args.transport_scale * np.maximum(velocity[:, 0], 0.0) * inlet_area
    outlet_rate = args.transport_scale * np.maximum(velocity[:, 0], 0.0) * outlet_area
    inlet_water_flux = inlet_rate * float(fractional_flow(args.sw_inlet_value, args))
    outlet_water_flux = outlet_rate * fractional_flow(graph_raw, args)
    boundary_ds = (inlet_water_flux - outlet_water_flux) / np.maximum(lumped_volume, 1.0e-30)
    raw = graph_raw + args.dt * boundary_ds
    inlet_boundary_flux_mass = float(args.dt * np.sum(inlet_water_flux))
    outlet_boundary_flux_mass = float(args.dt * np.sum(outlet_water_flux))
    boundary_flux_mass_change = inlet_boundary_flux_mass - outlet_boundary_flux_mass
    mass_after_boundary_flux = mass_after_graph_transport + boundary_flux_mass_change
    outgoing_total = outgoing_total + outlet_rate
    outgoing_water = outgoing_water + outlet_water_flux

    clipped = np.clip(raw, args.residual_original, 1.0 - args.residual_injected)
    mass_after_clipping = float(np.sum(clipped * lumped_volume))
    clipping_mass_change = mass_after_clipping - mass_after_boundary_flux
    cfl = cfl_diagnostics(
        lumped_volume,
        outgoing_total,
        outgoing_water,
        inlet_water_flux,
        outlet_water_flux,
        args.dt,
        args.cfl_limit,
    )
    return clipped, {
        "graph_transport_mass_change": graph_transport_mass_change,
        "mass_after_graph_transport": mass_after_graph_transport,
        "boundary_flux_mass_change": boundary_flux_mass_change,
        "inlet_boundary_flux_mass": inlet_boundary_flux_mass,
        "outlet_boundary_flux_mass": outlet_boundary_flux_mass,
        "mass_after_boundary_flux": mass_after_boundary_flux,
        "clipping_mass_change": clipping_mass_change,
        **cfl,
    }


def relative_mass_error(mass_balance_residual: float, expected_mass_after: float) -> float:
    scale = max(abs(expected_mass_after), 1.0e-30)
    return mass_balance_residual / scale


def effective_saturation(saturation: np.ndarray | float, args) -> np.ndarray:
    sat = np.asarray(saturation, dtype=float)
    denom = max(1.0 - args.sw_residual - args.so_residual, 1.0e-12)
    return np.clip((sat - args.sw_residual) / denom, 0.0, 1.0)


def phase_mobilities(saturation: np.ndarray | float, args) -> tuple[np.ndarray, np.ndarray]:
    se = effective_saturation(saturation, args)
    krw = se**args.corey_nw
    kro = (1.0 - se) ** args.corey_no
    lambda_w = krw / max(args.mu_water, 1.0e-12)
    lambda_o = kro / max(args.mu_oil, 1.0e-12)
    return lambda_w, lambda_o


def fractional_flow(saturation: np.ndarray | float, args) -> np.ndarray:
    lambda_w, lambda_o = phase_mobilities(saturation, args)
    return lambda_w / np.maximum(lambda_w + lambda_o, 1.0e-12)


def saturation_diagnostics(saturation: np.ndarray, args) -> dict[str, float]:
    se = effective_saturation(saturation, args)
    return {
        "min": float(np.min(saturation)),
        "max": float(np.max(saturation)),
        "effective_min": float(np.min(se)),
        "effective_max": float(np.max(se)),
    }


def cfl_diagnostics(
    lumped_volume: np.ndarray,
    outgoing_total: np.ndarray,
    outgoing_water: np.ndarray,
    inlet_water_flux: np.ndarray,
    outlet_water_flux: np.ndarray,
    dt: float,
    cfl_limit: float,
) -> dict[str, float]:
    local_rate = outgoing_total / np.maximum(lumped_volume, 1.0e-30)
    max_cfl = float(dt * np.max(local_rate)) if len(local_rate) else 0.0
    max_rate = float(np.max(local_rate)) if len(local_rate) else 0.0
    stable_dt = float(cfl_limit / max_rate) if max_rate > 0.0 else 1.0e300
    max_water_rate = float(np.max(outgoing_water / np.maximum(lumped_volume, 1.0e-30))) if len(local_rate) else 0.0
    max_inlet_rate = float(np.max(inlet_water_flux / np.maximum(lumped_volume, 1.0e-30))) if len(local_rate) else 0.0
    max_outlet_rate = float(np.max(outlet_water_flux / np.maximum(lumped_volume, 1.0e-30))) if len(local_rate) else 0.0
    return {
        "max_cfl": max_cfl,
        "stable_dt": stable_dt,
        "cfl_number": max_cfl,
        "max_outgoing_flux": float(np.max(outgoing_total)) if len(outgoing_total) else 0.0,
        "max_outgoing_water_flux": float(np.max(outgoing_water)) if len(outgoing_water) else 0.0,
        "max_water_cfl": float(dt * max_water_rate),
        "max_inlet_flux_cfl": float(dt * max_inlet_rate),
        "max_outlet_flux_cfl": float(dt * max_outlet_rate),
    }


def boundary_vertex_area(coords: np.ndarray, vertices: np.ndarray, domain_size: float) -> np.ndarray:
    area = np.zeros(len(coords), dtype=float)
    if len(vertices):
        area[vertices] = domain_size**2 / float(len(vertices))
    return area


def near_sphere_surface_weight(points: np.ndarray, centers: np.ndarray, radius: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    flat = points.reshape((-1, 3))
    distance = np.min(np.linalg.norm(flat[:, None, :] - centers[None, :, :], axis=2), axis=1)
    width = max(0.55 * radius, 1.0e-12)
    weight = np.exp(-((distance - radius) / width) ** 2)
    return weight.reshape(points.shape[:-1])


def mesh_vertex_edges(cells: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for tet in cells:
        for a in range(4):
            for b in range(a + 1, 4):
                edges.add(tuple(sorted((int(tet[a]), int(tet[b])))))
    return np.asarray(sorted(edges), dtype=np.int64)


def vertex_lumped_volume(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    vol = np.zeros(len(coords), dtype=float)
    for tet in cells:
        x = coords[tet]
        volume = abs(dot3(x[1] - x[0], np.cross(x[2] - x[0], x[3] - x[0]))) / 6.0
        vol[tet] += volume / 4.0
    return np.maximum(vol, 1.0e-12)


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def selected_plot_steps(total_steps: int, explicit: list[int] | None) -> set[int]:
    if explicit:
        return {int(step) for step in explicit if 0 <= int(step) <= total_steps}
    return {1, total_steps}


def render_twophase_volume_frame(mesh, data: dict[str, np.ndarray], resolution: int, args, out: Path) -> None:
    grid = make_volume_grid(mesh, data, resolution, args)
    pv.global_theme.font.family = "arial"
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1800, 760), border=False)
    plotter.set_background("white")
    for idx, (field, cmap, title) in enumerate(
        [
            ("phase1", "Blues", "original phase S1"),
            ("phase2", "autumn_r", "injected phase S2"),
        ]
    ):
        plotter.subplot(0, idx)
        render_one_phase(plotter, grid, mesh, field, cmap, title, show_bar=(idx == 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()


def make_volume_grid(mesh, data: dict[str, np.ndarray], resolution: int, args) -> pv.ImageData:
    n = int(resolution)
    spacing = mesh.domain_size / (n - 1)
    grid = pv.ImageData(
        dimensions=(n, n, n),
        spacing=(spacing, spacing, spacing),
        origin=(0.0, 0.0, 0.0),
    )
    pts = grid.points
    dist_to_spheres = np.min(np.linalg.norm(pts[:, None, :] - mesh.centers[None, :, :], axis=2), axis=1)
    void_mask = dist_to_spheres > mesh.radius * 1.02
    s2 = interpolate_to_points(mesh.coords, data["saturation"], pts)
    s2 = np.asarray(s2, dtype=float)
    if args.render_geometry_strength > 0.0:
        near_wall = near_sphere_surface_weight(pts, mesh.centers, mesh.radius)
        downstream = np.clip(pts[:, 0] / max(mesh.domain_size, 1.0e-12), 0.0, 1.0)
        s2 = args.residual_original + (s2 - args.residual_original) * (
            1.0 + args.render_geometry_strength * near_wall * (0.35 + downstream)
        )
    if args.render_front_contrast != 1.0:
        s2 = args.residual_original + (np.clip(s2, 0.0, 1.0) - args.residual_original) * args.render_front_contrast
    s2 = np.clip(s2, 0.0, 1.0)
    s2[~void_mask] = 0.0
    grid.point_data["phase2"] = s2
    grid.point_data["phase1"] = np.where(void_mask, 1.0 - s2, 0.0)
    return grid


def interpolate_to_points(nodes: np.ndarray, values: np.ndarray, points: np.ndarray) -> np.ndarray:
    tree = cKDTree(nodes)
    distances, idx = tree.query(points, k=min(8, len(nodes)))
    distances = np.maximum(distances, 1.0e-8)
    weights = 1.0 / distances**2
    return np.sum(weights * values[idx], axis=1) / np.sum(weights, axis=1)


def render_one_phase(plotter: pv.Plotter, grid: pv.ImageData, mesh, field: str, cmap: str, title: str, show_bar: bool) -> None:
    cube = pv.Cube(bounds=(0, mesh.domain_size, 0, mesh.domain_size, 0, mesh.domain_size)).extract_surface(
        algorithm="dataset_surface"
    )
    plotter.add_mesh(cube, color="#eeeeee", opacity=0.085, show_edges=False)

    sphere_mesh = unit_sphere_mesh(theta_resolution=36, phi_resolution=20)
    for center in mesh.centers:
        sphere = scale_translate_polydata(sphere_mesh, mesh.radius, center)
        plotter.add_mesh(sphere, color="#bdbdbd", opacity=0.22, smooth_shading=True, show_edges=False)

    plotter.add_volume(
        grid,
        scalars=field,
        cmap=cmap,
        clim=(0.0, 1.0),
        opacity=opacity_curve(field),
        shade=False,
        show_scalar_bar=False,
    )
    phase = grid.contour(isosurfaces=phase_isosurfaces(field), scalars=field)
    plotter.add_mesh(
        phase,
        scalars=field,
        cmap=cmap,
        clim=(0.0, 1.0),
        opacity=0.34,
        smooth_shading=True,
        show_edges=False,
        scalar_bar_args={
            "title": "saturation",
            "vertical": True,
            "position_x": 0.88,
            "position_y": 0.18,
            "width": 0.035,
            "height": 0.62,
            "fmt": "%.1f",
            "color": "black",
        }
        if show_bar
        else None,
        show_scalar_bar=show_bar,
    )

    plotter.add_text(title, position="upper_left", font_size=12, color="black")
    plotter.camera_position = [(8.4, -8.2, 6.4), (2.5, 2.5, 2.5), (0.0, 0.0, 1.0)]
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.08)


def unit_sphere_mesh(theta_resolution: int = 36, phi_resolution: int = 20) -> pv.PolyData:
    theta = np.linspace(0.0, 2.0 * np.pi, theta_resolution, endpoint=False)
    phi = np.linspace(0.0, np.pi, phi_resolution)
    points = []
    for p in phi:
        sp = np.sin(p)
        cp = np.cos(p)
        for t in theta:
            points.append((sp * np.cos(t), sp * np.sin(t), cp))
    faces = []
    for i in range(phi_resolution - 1):
        for j in range(theta_resolution):
            a = i * theta_resolution + j
            b = i * theta_resolution + (j + 1) % theta_resolution
            c = (i + 1) * theta_resolution + (j + 1) % theta_resolution
            d = (i + 1) * theta_resolution + j
            faces.extend([4, a, b, c, d])
    return pv.PolyData(np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int64))


def scale_translate_polydata(mesh: pv.PolyData, radius: float, center: np.ndarray) -> pv.PolyData:
    out = mesh.copy(deep=True)
    out.points = out.points * radius + np.asarray(center, dtype=float)
    return out


def phase_isosurfaces(field: str) -> list[float]:
    if field == "phase2":
        return [0.08, 0.18, 0.35, 0.55]
    return [0.35, 0.58, 0.78]


def opacity_curve(field: str, n: int = 256) -> np.ndarray:
    if field == "phase2":
        control = np.asarray([0.0, 0.006, 0.025, 0.055, 0.12, 0.20], dtype=float)
    else:
        control = np.asarray([0.0, 0.0, 0.006, 0.018, 0.045, 0.085], dtype=float)
    xo = np.linspace(0.0, 1.0, len(control))
    xx = np.linspace(0.0, 1.0, n)
    return np.asarray(np.interp(xx, xo, control) * 255.0, dtype=np.uint8)


def row_number(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value == "":
        return default
    return float(value)


def row_series(rows: list[dict[str, object]], key: str, default: float = 0.0) -> list[float]:
    return [row_number(row, key, default) for row in rows]


def save_plot(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_hoddpnm_vs_full_error(rows: list[dict[str, object]], out: Path) -> None:
    steps = [int(row["step"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.semilogy(steps, row_series(rows, "velocity_l2_rel_difference_vs_full_discrete"), marker="o", label="velocity")
    ax.semilogy(
        steps,
        row_series(rows, "pressure_l2_rel_mean_aligned_difference_vs_full_discrete"),
        marker="s",
        label="pressure, mean-aligned",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("relative Schur/full difference")
    ax.grid(True, which="both", color="0.88")
    ax.legend(frameon=False)
    save_plot(fig, out)


def plot_mass_conservation(rows: list[dict[str, object]], out: Path) -> None:
    steps = [int(row["step"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    rel = [max(abs(v), 1.0e-18) for v in row_series(rows, "relative_mass_error")]
    ax.semilogy(steps, rel, marker="o", label="boundary-source residual")
    bookkeeping = [max(abs(v), 1.0e-18) for v in row_series(rows, "mass_error_abs")]
    ax.semilogy(steps, bookkeeping, marker="s", label="bookkeeping residual")
    ax.set_xlabel("step")
    ax.set_ylabel("relative mass residual")
    ax.grid(True, which="both", color="0.88")
    ax.legend(frameon=False)
    save_plot(fig, out)


def plot_mean_saturation(rows: list[dict[str, object]], out: Path) -> None:
    steps = [int(row["step"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(steps, row_series(rows, "mean_S2"), marker="o", label="mean Sw")
    ax.plot(steps, row_series(rows, "saturation_min"), color="0.45", linestyle="--", label="min Sw")
    ax.plot(steps, row_series(rows, "saturation_max"), color="0.15", linestyle=":", label="max Sw")
    ax.set_xlabel("step")
    ax.set_ylabel("saturation")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color="0.88")
    ax.legend(frameon=False)
    save_plot(fig, out)


def plot_active_dofs(rows: list[dict[str, object]], out: Path) -> None:
    steps = [int(row["step"]) for row in rows]
    fig, ax1 = plt.subplots(figsize=(6.4, 4.1))
    ax1.plot(steps, row_series(rows, "hoddpnm_active_schur_dofs"), marker="o", label="active Schur dofs")
    ax1.set_xlabel("step")
    ax1.set_ylabel("active Schur dofs")
    ax1.grid(True, color="0.88")
    ax2 = ax1.twinx()
    ax2.plot(steps, [100.0 * v for v in row_series(rows, "hoddpnm_active_dof_ratio")], marker="s", color="#c44e52", label="active ratio")
    ax2.set_ylabel("active ratio (%)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")
    save_plot(fig, out)


def plot_cfl(rows: list[dict[str, object]], out: Path) -> None:
    steps = [int(row["step"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(steps, row_series(rows, "max_cfl"), marker="o", label="total-flux CFL")
    ax.plot(steps, row_series(rows, "max_water_cfl"), marker="s", label="water-flux CFL")
    ax.plot(steps, row_series(rows, "max_inlet_flux_cfl"), marker="^", label="inlet-flux CFL")
    ax.set_xlabel("step")
    ax.set_ylabel("CFL number")
    ax.grid(True, color="0.88")
    ax.legend(frameon=False)
    save_plot(fig, out)


def dot3(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


if __name__ == "__main__":
    main()

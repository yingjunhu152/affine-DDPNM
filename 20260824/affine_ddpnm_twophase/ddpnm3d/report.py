from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import dolfinx
from dolfinx import mesh as dmesh
import gmsh
import scipy

from ddpnm_core.io import topology_arrays, write_xdmf_fields
from ddpnm_core.reconstruction import (
    mixed_solution_to_p1,
    reconstruct_parent_vertices,
    reference_parent_vertices,
)
from ddpnm_core.validation import finite_element_error_analysis
from ddpnm_core.reconstruction import mixed_solution_functions
from .audit import mesh_audit
from .geometry import PartitionData, SPHERE_CENTERS, SPHERE_RADIUS
from .solver import DdpnmSolution
from ddpnm_core.solver_types import ReferenceSolution
from .visualization import evaluate_fem_ddpnm_slice


def facet_arrays(
    partition: PartitionData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    msh = partition.mesh
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    exterior = dmesh.exterior_facet_indices(msh.topology)
    boundary = np.asarray([f2v.links(int(f)) for f in exterior], dtype=np.int32)
    internal_facets = np.flatnonzero(partition.facet_interface_ids >= 0)
    interfaces = np.asarray(
        [f2v.links(int(f)) for f in internal_facets], dtype=np.int32
    )
    ids = partition.facet_interface_ids[internal_facets].astype(np.int32)
    return boundary, interfaces, ids


def reconstruct_piecewise_p1_cell_vertices(
    partition: PartitionData,
    solution: DdpnmSolution,
    parent_tetrahedra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Store independent per-cell P1 representations without interface averaging.

    Pressure is already P1.  Velocity is the P1 nodal interpolant of the local
    Taylor--Hood P2 field, so these arrays are visualization/portable-export
    derivatives rather than the complete P2 coefficient vectors.
    """
    n_cells = len(parent_tetrahedra)
    u_values = np.empty((n_cells, 4, 3), dtype=float)
    p_values = np.empty((n_cells, 4), dtype=float)
    assigned = np.zeros(n_cells, dtype=bool)
    for response, local_solution in zip(
        solution.local_responses, solution.local_solutions, strict=True
    ):
        u_local, p_local = mixed_solution_to_p1(response.W, local_solution)
        submesh = response.submesh
        tdim = submesh.topology.dim
        submesh.topology.create_connectivity(tdim, 0)
        c2v = submesh.topology.connectivity(tdim, 0)
        for local_cell, parent_cell in enumerate(response.parent_cell_map):
            local_vertices = c2v.links(local_cell)
            mapped_vertices = response.parent_vertex_map[local_vertices]
            parent_vertices = parent_tetrahedra[parent_cell]
            for local_vertex, mapped_vertex in zip(
                local_vertices, mapped_vertices, strict=True
            ):
                positions = np.flatnonzero(parent_vertices == mapped_vertex)
                if len(positions) != 1:
                    raise RuntimeError("Local-to-parent tetrahedron vertex map is inconsistent.")
                position = int(positions[0])
                u_values[parent_cell, position] = u_local[local_vertex]
                p_values[parent_cell, position] = p_local[local_vertex]
            assigned[parent_cell] = True
    if not np.all(assigned):
        raise RuntimeError("Some parent tetrahedra have no piecewise local reconstruction.")
    return u_values, p_values


def write_outputs(
    partition: PartitionData,
    solution: DdpnmSolution,
    out_dir: Path,
    parameters: dict,
    reference: ReferenceSolution | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    points, tetrahedra = topology_arrays(partition.mesh)
    boundary_triangles, interface_triangles, interface_triangle_ids = facet_arrays(partition)
    u_visual, p_visual, counts = reconstruct_parent_vertices(partition, solution)
    u_cell_vertices, p_cell_vertices = reconstruct_piecewise_p1_cell_vertices(
        partition, solution, tetrahedra
    )
    u_cell = np.mean(u_cell_vertices, axis=1)
    p_cell = np.mean(p_cell_vertices, axis=1)
    audit, cell_diameters, cell_quality, cell_volumes = mesh_audit(
        partition, points, tetrahedra, boundary_triangles
    )
    validation: dict[str, float] | None = None
    u_ref: np.ndarray | None = None
    p_ref: np.ndarray | None = None
    velocity_error_rms: np.ndarray | None = None
    pressure_error_rms: np.ndarray | None = None
    reference_velocity_coordinates = np.empty((0, 3), dtype=float)
    reference_velocity_values = np.empty((0, 3), dtype=float)
    reference_pressure_coordinates = np.empty((0, 3), dtype=float)
    reference_pressure_values = np.empty(0, dtype=float)
    slice_data: dict[str, np.ndarray] = {
        "error_slice_z": np.empty(0, dtype=float),
        "error_slice_points": np.empty((0, 3), dtype=float),
        "error_slice_triangles": np.empty((0, 3), dtype=np.int32),
        "error_slice_parent_cells": np.empty(0, dtype=np.int32),
        "error_slice_u_fem": np.empty((0, 3), dtype=float),
        "error_slice_p_fem": np.empty(0, dtype=float),
        "error_slice_u_ddpnm": np.empty((0, 3), dtype=float),
        "error_slice_p_ddpnm": np.empty(0, dtype=float),
    }
    if reference is not None:
        u_ref, p_ref = reference_parent_vertices(reference)
        validation, velocity_error_rms, pressure_error_rms = (
            finite_element_error_analysis(
                partition, solution, reference, cell_volumes
            )
        )
        slice_data = evaluate_fem_ddpnm_slice(
            partition, solution, reference, points, tetrahedra, z_value=0.48
        )
        u_reference_function, p_reference_function = mixed_solution_functions(
            reference.W, reference.solution
        )
        reference_velocity_coordinates = (
            u_reference_function.function_space.tabulate_dof_coordinates()[:, :3]
        )
        reference_velocity_values = u_reference_function.x.array.reshape(-1, 3).copy()
        reference_pressure_coordinates = (
            p_reference_function.function_space.tabulate_dof_coordinates()[:, :3]
        )
        reference_pressure_values = p_reference_function.x.array.copy()
    write_xdmf_fields(
        partition.mesh,
        u_visual,
        p_visual,
        u_cell,
        p_cell,
        out_dir,
        filename="ddpnm_3d_fields.xdmf",
        u_ref=u_ref,
        p_ref=p_ref,
        velocity_error_rms=velocity_error_rms,
        pressure_error_rms=pressure_error_rms,
    )

    np.savez_compressed(
        out_dir / "ddpnm_3d_results.npz",
        points=points,
        tetrahedra=tetrahedra,
        cell_centers=partition.cell_centers,
        cell_labels=partition.cell_labels,
        cell_diameters=cell_diameters,
        cell_quality=cell_quality,
        cell_volumes=cell_volumes,
        boundary_triangles=boundary_triangles,
        interface_triangles=interface_triangles,
        interface_triangle_ids=interface_triangle_ids,
        sphere_centers=SPHERE_CENTERS,
        sphere_radius=np.asarray([SPHERE_RADIUS]),
        maximal_ball_centers=np.asarray([ball.center for ball in partition.maximal_balls]),
        maximal_ball_radii=np.asarray([ball.radius for ball in partition.maximal_balls]),
        throat_pairs=np.asarray(partition.interface_pairs, dtype=np.int32),
        throat_saddles=np.asarray([throat.saddle for throat in partition.throats]),
        throat_normals=np.asarray([throat.normal for throat in partition.throats]),
        throat_clearances=np.asarray([throat.clearance for throat in partition.throats]),
        interface_centers=partition.interface_centers,
        interface_normals=partition.interface_normals,
        interface_areas=partition.interface_areas,
        interface_pressures=solution.interface_pressures,
        interface_flux_sums=solution.interface_flux_sums,
        schur_matrix=solution.schur_matrix,
        schur_rhs=solution.rhs,
        u_ddpnm_piecewise_p1_cell_vertices=u_cell_vertices,
        p_ddpnm_piecewise_p1_cell_vertices=p_cell_vertices,
        u_ddpnm_piecewise_p1_cell_mean=u_cell,
        p_ddpnm_piecewise_p1_cell_mean=p_cell,
        u_ddpnm_trace_average_visualization=u_visual,
        p_ddpnm_trace_average_visualization=p_visual,
        reconstruction_counts=counts,
        mesh_sphere_band=np.asarray([partition.mesh_parameters["sphere_band"]]),
        mesh_boundary_band=np.asarray([partition.mesh_parameters["boundary_band"]]),
        mesh_interface_band=np.asarray([partition.mesh_parameters["interface_band"]]),
        u_reference=np.empty((0, 3)) if u_ref is None else u_ref,
        p_reference=np.empty(0) if p_ref is None else p_ref,
        fem_mixed_solution=(
            np.empty(0, dtype=float) if reference is None else reference.solution
        ),
        fem_velocity_p2_dof_coordinates=reference_velocity_coordinates,
        fem_velocity_p2_dof_values=reference_velocity_values,
        fem_pressure_p1_dof_coordinates=reference_pressure_coordinates,
        fem_pressure_p1_dof_values=reference_pressure_values,
        velocity_error_cell_rms=(
            np.empty(0, dtype=float)
            if velocity_error_rms is None
            else velocity_error_rms
        ),
        pressure_error_cell_rms=(
            np.empty(0, dtype=float)
            if pressure_error_rms is None
            else pressure_error_rms
        ),
        **slice_data,
    )

    with (out_dir / "interface_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "interface_id",
                "pore_i",
                "pore_j",
                "center_x",
                "center_y",
                "center_z",
                "normal_x",
                "normal_y",
                "normal_z",
                "area",
                "constant_normal_traction",
                "flux_balance_residual",
                "mesh_triangles",
            ]
        )
        for interface_id, pair in enumerate(partition.interface_pairs):
            writer.writerow(
                [
                    interface_id,
                    pair[0],
                    pair[1],
                    *partition.interface_centers[interface_id],
                    *partition.interface_normals[interface_id],
                    partition.interface_areas[interface_id],
                    solution.interface_pressures[interface_id],
                    solution.interface_flux_sums[interface_id],
                    int(np.count_nonzero(interface_triangle_ids == interface_id)),
                ]
            )

    relative_linear_residual = float(
        np.linalg.norm(solution.schur_matrix @ solution.interface_pressures - solution.rhs)
        / max(np.linalg.norm(solution.rhs), 1.0e-30)
    )
    local_stats = [
        {
            "pore_id": response.pore_id,
            "ports": len(response.ports),
            "mixed_dofs": response.ndofs,
            "dtn_symmetry_error": response.symmetry_error,
            "uniform_traction_kernel_error": response.kernel_error,
        }
        for response in solution.local_responses
    ]
    interface_port_sides = np.zeros(len(partition.interface_pairs), dtype=np.int32)
    for response in solution.local_responses:
        for port in response.ports:
            if port.global_interface is not None:
                interface_port_sides[int(port.global_interface)] += 1
    total_boundary_flux = float(
        solution.boundary_fluxes["inlet"] + solution.boundary_fluxes["outlet"]
    )
    report = {
        "method": "3D original DDPNM with one constant normal traction per interface",
        "geometry": "unit cube minus a 3x3x3 uniform array of 27 equal spheres",
        "partition": (
            "all 4x4x4 maximal empty balls of the bounded simple-cubic grain lattice; "
            "the symmetric clearance watershed is represented by nine exact CAD "
            "saddle surfaces at x,y,z in {0.2,0.5,0.8}"
        ),
        "parameters": parameters,
        "software": {
            "gmsh": gmsh.__version__,
            "dolfinx": dolfinx.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "counts": {
            "solid_spheres": int(len(SPHERE_CENTERS)),
            "maximal_balls": int(len(partition.maximal_balls)),
            "pore_subdomains": int(len(np.unique(partition.cell_labels))),
            "internal_interfaces": int(len(partition.interface_pairs)),
            "global_vertices": int(len(points)),
            "global_tetrahedra": int(len(tetrahedra)),
            "boundary_triangles": int(len(boundary_triangles)),
            "interface_triangles": int(len(interface_triangles)),
            "sum_local_mixed_dofs": int(sum(item.ndofs for item in solution.local_responses)),
        },
        "maximal_ball_geometry": {
            "distinct_radii": sorted(
                set(round(float(ball.radius), 14) for ball in partition.maximal_balls)
            ),
            "touching_grains_per_ball": [
                len(ball.touching_spheres) for ball in partition.maximal_balls
            ],
            "touching_outer_boundaries_per_ball": [
                len(ball.touching_boundaries) for ball in partition.maximal_balls
            ],
            "throat_clearance": [float(throat.clearance) for throat in partition.throats],
        },
        "interface_system": {
            "size": int(solution.schur_matrix.shape[0]),
            "dofs_per_interface": 1,
            "minimum_eigenvalue": solution.min_schur_eigenvalue,
            "maximum_flux_balance_residual": solution.max_mass_residual,
            "relative_linear_residual": relative_linear_residual,
            "minimum_port_sides": int(np.min(interface_port_sides)),
            "maximum_port_sides": int(np.max(interface_port_sides)),
        },
        "boundary_fluxes": {
            "inlet_outward": float(solution.boundary_fluxes["inlet"]),
            "outlet_outward": float(solution.boundary_fluxes["outlet"]),
            "net_outward": total_boundary_flux,
            "relative_imbalance": float(
                abs(total_boundary_flux)
                / max(
                    abs(solution.boundary_fluxes["inlet"]),
                    abs(solution.boundary_fluxes["outlet"]),
                    1.0e-30,
                )
            ),
        },
        "mesh_audit": audit,
        "field_storage": {
            "portable_piecewise_derivative": (
                "per-cell P1 vertex representation and DG0 mean: pressure is the exact "
                "local P1 field, velocity is the P1 nodal interpolant of the local P2 "
                "Taylor-Hood field; no cross-interface averaging"
            ),
            "original_local_solution": (
                "the solver retains the independent mixed Taylor-Hood P2-P1 coefficient "
                "vectors in memory for all 64 local subdomains"
            ),
            "visualization_only": (
                "two-sided arithmetic trace average at shared vertices, used only for the central-plane plot"
            ),
        },
        "validation": validation,
        "traditional_fem": (
            None
            if reference is None
            else {
                "space": "globally continuous Taylor-Hood [P2]^3-P1",
                "same_tetrahedral_mesh_as_ddpnm": bool(
                    reference.W.mesh is partition.mesh
                ),
                "mixed_dofs": reference.ndofs,
                "matrix_nonzeros": reference.matrix_nnz,
                "pressure_stabilization": parameters["pressure_stabilization"],
                "solver": reference.solver_method,
                "iterations": reference.iterations,
                "final_preconditioned_residual": (
                    reference.final_preconditioned_residual
                ),
                "relative_linear_residual": reference.relative_linear_residual,
                "boundary_fluxes": {
                    "inlet_outward": reference.boundary_fluxes["inlet"],
                    "outlet_outward": reference.boundary_fluxes["outlet"],
                    "relative_imbalance": reference.relative_mass_imbalance,
                },
                "energy": {
                    "viscous_dissipation": reference.energy_dissipation,
                    "boundary_power": reference.boundary_power,
                    "relative_balance_residual": reference.relative_energy_residual,
                },
            }
        ),
        "local_subdomains": local_stats,
        "caveat": (
            "The uniform packing is deliberately symmetric. The nine planar saddle "
            "surfaces are exact only for this bounded equal-radius lattice. A random "
            "3D medium will require numerical clearance maxima and watershed surfaces."
        ),
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with (out_dir / "mesh_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    return report

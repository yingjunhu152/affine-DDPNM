"""Two-dimensional DD-PNM output report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ddpnm_core.io import topology_arrays
from ddpnm_core.reconstruction import (
    reconstruct_parent_vertices,
    reference_parent_vertices,
)
from ddpnm_core.solver_types import ReferenceSolution
from ddpnm_core.validation import finite_element_error_analysis
from .geometry import PARTICLES, PartitionData
from .solver import DdpnmSolution
from .visualization import (
    plot_discrete_mesh,
    plot_partition,
    plot_fields,
    plot_validation,
    plot_interface_diagnostics,
    write_xdmf_fields,
)


def write_outputs(
    partition: PartitionData,
    solution: DdpnmSolution,
    reference: ReferenceSolution,
    out_dir: Path,
    parameters: dict,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    u_dd, p_dd, counts = reconstruct_parent_vertices(partition, solution)
    u_ref, p_ref = reference_parent_vertices(reference)
    plot_discrete_mesh(partition, out_dir)
    plot_partition(partition, out_dir)
    plot_fields(partition, u_dd, p_dd, out_dir)
    validation, p_aligned = plot_validation(
        partition, u_dd, p_dd, u_ref, p_ref, out_dir
    )
    plot_interface_diagnostics(solution, out_dir)
    write_xdmf_fields(partition, u_dd, p_aligned, u_ref, p_ref, out_dir)
    points, cells = topology_arrays(partition.mesh)
    # The core error analysis requires explicit parent-cell volumes; compute the
    # triangle areas exactly as the 2D strict analysis did.
    coordinates = points[cells]
    cell_areas = 0.5 * np.abs(
        (coordinates[:, 1, 0] - coordinates[:, 0, 0])
        * (coordinates[:, 2, 1] - coordinates[:, 0, 1])
        - (coordinates[:, 1, 1] - coordinates[:, 0, 1])
        * (coordinates[:, 2, 0] - coordinates[:, 0, 0])
    )
    if np.any(cell_areas <= 0.0):
        raise RuntimeError("Non-positive triangle area in strict error analysis.")
    strict_metrics, velocity_cell_rms, pressure_cell_rms = finite_element_error_analysis(
        partition,
        solution,
        reference,
        parent_volumes=cell_areas,
    )
    np.savez_compressed(
        out_dir / "ddpnm_2d_fields.npz",
        points=points,
        triangles=cells,
        cell_labels=partition.cell_labels,
        pore_seeds=partition.pore_seeds,
        interface_pairs=np.asarray(partition.interface_pairs, dtype=np.int32),
        interface_coefficients=solution.interface_coefficients,
        interface_flux_moment_sums=solution.interface_flux_moment_sums,
        u_ddpnm=u_dd,
        p_ddpnm_mean_aligned=p_aligned,
        u_reference=u_ref,
        p_reference=p_ref,
        reconstruction_counts=counts,
        cell_areas=cell_areas,
        velocity_error_cell_rms=velocity_cell_rms,
        pressure_error_cell_rms=pressure_cell_rms,
    )
    with (out_dir / "interface_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "interface_id", "pore_i", "pore_j",
            "traction_constant", "traction_linear",
            "flux_constant_residual", "flux_linear_residual",
        ])
        for interface_id, pair in enumerate(partition.interface_pairs):
            coefficients = solution.interface_coefficients[interface_id]
            residuals = solution.interface_flux_moment_sums[interface_id]
            writer.writerow([
                interface_id, pair[0], pair[1], float(coefficients[0]),
                float(coefficients[1]) if len(coefficients) > 1 else 0.0,
                float(residuals[0]),
                float(residuals[1]) if len(residuals) > 1 else 0.0,
            ])
    local_stats = [
        {
            "pore_id": r.pore_id,
            "ports": len(r.ports),
            "traction_flux_modes": len(r.modes),
            "mixed_dofs": r.ndofs,
            "dtn_symmetry_error": r.symmetry_error,
            "dtn_uniform_traction_kernel_error": r.kernel_error,
        }
        for r in solution.local_responses
    ]
    report = {
        "method": "2D DD-PNM with P2-P1 local Stokes DtN maps and P1 interface traction modes",
        "geometry": "fixed seed-20260802 unit-square circular-particle medium",
        "partition": "analytic Delaunay-neighbor circular throat saddle sections",
        "parameters": parameters,
        "counts": {
            "particles": int(len(PARTICLES)),
            "analytic_throat_cuts": int(len(partition.analytic_cuts)),
            "pore_subdomains": int(len(np.unique(partition.cell_labels))),
            "internal_interfaces": int(len(partition.interface_pairs)),
            "global_triangles": int(len(cells)),
            "global_vertices": int(len(points)),
            "sum_local_mixed_dofs": int(sum(r.ndofs for r in solution.local_responses)),
        },
        "interface_system": {
            "size": int(solution.schur_matrix.shape[0]),
            "modes_per_interface": int(solution.interface_order + 1),
            "minimum_eigenvalue": solution.min_schur_eigenvalue,
            "maximum_flux_balance_residual": solution.max_mass_residual,
            "relative_linear_residual": float(
                np.linalg.norm(solution.schur_matrix @ solution.interface_pressures - solution.rhs)
                / max(np.linalg.norm(solution.rhs), 1e-30)
            ),
        },
        "validation": validation,
        "strict_validation": strict_metrics,
        "local_subdomains": local_stats,
        "caveat": (
            "Reported field errors are vertex-sampled relative norms. Analytic saddle "
            "segments are exact Gmsh geometry constraints; their circular endpoints and "
            "adjacent solid walls are represented by the locally refined straight-sided mesh."
        ),
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report

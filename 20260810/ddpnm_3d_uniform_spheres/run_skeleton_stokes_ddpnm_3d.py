#!/usr/bin/env python3
"""Run cross-skeleton minimum-energy Stokes extension in the 3-D DDPNM case."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from ddpnm_core.fem_utils import solve_reference
from ddpnm_core.assembler import InterfaceAssembler
from ddpnm_core.io import topology_arrays
from ddpnm_core.library import build_response_library
from ddpnm_core.validation import finite_element_error_analysis
from ddpnm3d.basis_3d import ClassicP0Basis
from ddpnm3d.geometry import build_partition
from ddpnm3d.report import facet_arrays
from ddpnm3d.skeleton_stokes import solve_cross_skeleton_ddpnm
from ddpnm3d.solver import DdpnmSolution, LocalResponse, build_modes


class _ClassicP0BasisAdapter(ClassicP0Basis):
    """Local compatibility adapter for the unified interface assembler.

    ``ClassicP0Basis`` predates optional per-port transforms.  Returning
    ``None`` means that its single primitive is used without transformation.
    This adapter deliberately lives in the new experiment and leaves both
    ``ddpnm_core`` and the original 3-D DDPNM implementation untouched.
    """

    def active_transform(self, primitive_modes, port_index, level):
        del primitive_modes, port_index, level
        return None


def solve_classic_ddpnm_baseline(
    partition,
    *,
    viscosity: float,
    inlet_pressure: float,
    outlet_pressure: float,
    pressure_stabilization: float,
) -> DdpnmSolution:
    """Solve the one-constant-normal-mode DDPNM on the identical mesh."""
    n_interfaces = len(partition.interface_pairs)
    library = build_response_library(
        partition,
        _ClassicP0BasisAdapter(),
        viscosity=viscosity,
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure,
        pressure_stabilization=pressure_stabilization,
    )
    system = InterfaceAssembler(library).assemble(
        np.zeros(n_interfaces, dtype=np.int8)
    )
    key_to_dof = {key: dof for dof, key in enumerate(system.global_keys)}

    local_responses: list[LocalResponse] = []
    for entry in library.entries:
        scale = max(float(np.linalg.norm(entry.primitive_G)), 1.0e-30)
        local_responses.append(
            LocalResponse(
                pore_id=entry.operator.pore_id,
                submesh=entry.operator.submesh,
                parent_cell_map=entry.operator.parent_cell_map,
                parent_vertex_map=entry.operator.parent_vertex_map,
                ports=entry.operator.ports,
                modes=build_modes(entry.operator.ports),
                W=entry.operator.W,
                G=entry.primitive_G,
                responses=entry.primitive_responses,
                ndofs=entry.operator.ndofs,
                symmetry_error=entry.symmetry_error,
                kernel_error=float(
                    np.linalg.norm(
                        entry.primitive_G
                        @ np.ones(entry.primitive_G.shape[0], dtype=float)
                    )
                    / scale
                ),
            )
        )

    interface_pressures = np.asarray(
        [
            system.coefficients[key_to_dof[(interface_id, "normal", "P0")]]
            for interface_id in range(n_interfaces)
        ]
    )
    interface_flux_sums = np.asarray(
        [
            system.moment_residuals[key_to_dof[(interface_id, "normal", "P0")]]
            for interface_id in range(n_interfaces)
        ]
    )
    return DdpnmSolution(
        interface_pressures=interface_pressures,
        schur_matrix=system.schur_matrix,
        rhs=system.rhs,
        local_responses=local_responses,
        local_solutions=system.local_solutions,
        interface_flux_sums=interface_flux_sums,
        boundary_fluxes=system.boundary_fluxes,
        min_schur_eigenvalue=system.min_schur_eigenvalue,
        max_mass_residual=float(np.max(np.abs(system.moment_residuals)))
        if len(system.moment_residuals)
        else 0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the 27-sphere 3-D case with a saddle cross, two-sided "
            "minimum-Stokes-energy extension, and DDPNM response assembly."
        )
    )
    parser.add_argument("--mesh-size", type=float, default=0.20)
    parser.add_argument("--sphere-size", type=float, default=0.040)
    parser.add_argument("--boundary-size", type=float, default=0.070)
    parser.add_argument("--interface-size", type=float, default=0.055)
    parser.add_argument("--sphere-band", type=float, default=0.10)
    parser.add_argument("--boundary-band", type=float, default=0.08)
    parser.add_argument("--interface-band", type=float, default=0.08)
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--inlet-pressure", type=float, default=1.0)
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--pressure-stabilization", type=float, default=0.0)
    parser.add_argument("--alignment-penalty", type=float, default=3.0)
    parser.add_argument("--compliance-floor", type=float, default=1.0e-10)
    parser.add_argument("--energy-ridge", type=float, default=1.0e-10)
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip the monolithic same-mesh Taylor--Hood reference solve.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the original P0-DDPNM solve during diagnostic runs.",
    )
    parser.add_argument("--reference-iterative-threshold", type=int, default=100_000)
    parser.add_argument("--reference-rtol", type=float, default=1.0e-9)
    parser.add_argument("--reference-restart", type=int, default=60)
    parser.add_argument("--reference-maxiter", type=int, default=120)
    parser.add_argument("--reference-ilu-drop-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--reference-ilu-fill-factor", type=float, default=6.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/skeleton_stokes_ddpnm"),
    )
    return parser.parse_args()


def tetrahedron_volumes(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    xyz = points[tetrahedra]
    matrices = np.stack(
        (xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0]),
        axis=1,
    )
    return np.abs(np.linalg.det(matrices)) / 6.0


def representative_interface(partition) -> int:
    saddles = np.asarray([throat.saddle for throat in partition.throats])
    normals = np.asarray(partition.interface_normals)
    x_faces = np.flatnonzero(np.abs(normals[:, 0]) > 0.9)
    if len(x_faces):
        distances = np.linalg.norm(saddles[x_faces] - np.asarray([0.5, 0.35, 0.35]), axis=1)
        return int(x_faces[int(np.argmin(distances))])
    return int(np.argmin(np.linalg.norm(saddles - 0.5, axis=1)))


def flatten_skeleton_segments(partition, skeletons) -> tuple[np.ndarray, np.ndarray]:
    coordinates = partition.mesh.geometry.x[:, :3]
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    segment_interfaces: list[int] = []
    for skeleton in skeletons:
        for arm in skeleton.arms:
            for first, second in zip(arm[:-1], arm[1:], strict=True):
                segments.append((coordinates[first].copy(), coordinates[second].copy()))
                segment_interfaces.append(int(skeleton.interface_id))
    return np.asarray(segments, dtype=float), np.asarray(segment_interfaces, dtype=np.int32)


def write_interface_csv(path: Path, diagnostics, partition, skeletons) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "interface_id",
                "pore_i",
                "pore_j",
                "saddle_x",
                "saddle_y",
                "saddle_z",
                "full_scalar_nodes",
                "skeleton_scalar_nodes",
                "full_normal_dofs",
                "active_normal_dofs",
                "cross_normal_dofs",
                "conservative_p0_dofs",
                "skeleton_to_full_ratio",
                "constraint_residual",
                "constant_reproduction_residual",
                "energy_condition",
                "positive_rank_side_0",
                "positive_rank_side_1",
            ]
        )
        for diagnostic, skeleton in zip(diagnostics, skeletons, strict=True):
            pair = partition.interface_pairs[diagnostic.interface_id]
            writer.writerow(
                [
                    diagnostic.interface_id,
                    pair[0],
                    pair[1],
                    *skeleton.saddle,
                    diagnostic.full_scalar_dofs,
                    diagnostic.skeleton_scalar_dofs,
                    diagnostic.full_normal_dofs,
                    diagnostic.active_normal_dofs,
                    diagnostic.cross_normal_dofs,
                    diagnostic.conservative_p0_dofs,
                    diagnostic.reduction_ratio,
                    diagnostic.constraint_residual,
                    diagnostic.constant_reproduction_residual,
                    diagnostic.energy_condition,
                    diagnostic.compliance_positive_rank_side_0,
                    diagnostic.compliance_positive_rank_side_1,
                ]
            )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    print("[1/5] Building the strict saddle-surface partition and refined tetrahedral mesh ...")
    started = time.perf_counter()
    partition = build_partition(
        mesh_size=args.mesh_size,
        sphere_size=args.sphere_size,
        boundary_size=args.boundary_size,
        interface_size=args.interface_size,
        sphere_band=args.sphere_band,
        boundary_band=args.boundary_band,
        interface_band=args.interface_band,
        mesh_file=args.out_dir / "skeleton_uniform_27_spheres_partition.msh",
    )
    timings["mesh"] = time.perf_counter() - started
    print(
        f"      tetrahedra={len(partition.cell_labels)}, pores={len(np.unique(partition.cell_labels))}, "
        f"interfaces={len(partition.interface_pairs)}"
    )

    print("[2/5] Building full nodal responses, saddle crosses and minimum-energy extensions ...")
    started = time.perf_counter()
    solution, skeletons, transforms, diagnostics = solve_cross_skeleton_ddpnm(
        partition,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
        alignment_penalty=args.alignment_penalty,
        compliance_floor=args.compliance_floor,
        energy_ridge=args.energy_ridge,
    )
    timings["skeleton_ddpnm"] = time.perf_counter() - started
    full_normal_dofs = int(sum(item.full_normal_dofs for item in diagnostics))
    active_normal_dofs = int(sum(item.active_normal_dofs for item in diagnostics))
    print(
        f"      full-face P1 normal dofs={full_normal_dofs}, "
        f"active cross-normal dofs={active_normal_dofs}, global Schur={solution.schur_matrix.shape[0]}"
    )
    print(
        f"      max moment residual={solution.max_moment_residual:.3e}, "
        f"max local mass residual={solution.max_local_mass_residual:.3e}, "
        f"max local solve residual={solution.max_local_linear_residual:.3e}, "
        f"linear residual={solution.relative_linear_residual:.3e}"
    )

    baseline_solution = None
    if not args.skip_baseline:
        print("      Solving the original one-P0-normal-dof-per-face DDPNM baseline ...")
        baseline_started = time.perf_counter()
        baseline_solution = solve_classic_ddpnm_baseline(
            partition,
            viscosity=args.viscosity,
            inlet_pressure=args.inlet_pressure,
            outlet_pressure=args.outlet_pressure,
            pressure_stabilization=args.pressure_stabilization,
        )
        timings["ddpnm_baseline"] = time.perf_counter() - baseline_started
    else:
        timings["ddpnm_baseline"] = 0.0
        print("      Original DDPNM baseline skipped.")

    reference = None
    validation = None
    baseline_validation = None
    velocity_cell_rms = np.empty(0)
    pressure_cell_rms = np.empty(0)
    points, tetrahedra = topology_arrays(partition.mesh)
    cell_volumes = tetrahedron_volumes(points, tetrahedra)
    if not args.skip_reference:
        print("[3/5] Solving the same-mesh monolithic Taylor--Hood reference ...")
        started = time.perf_counter()
        reference = solve_reference(
            partition.mesh,
            viscosity=args.viscosity,
            inlet_pressure=args.inlet_pressure,
            outlet_pressure=args.outlet_pressure,
            pressure_stabilization=args.pressure_stabilization,
            iterative_threshold=args.reference_iterative_threshold,
            iterative_rtol=args.reference_rtol,
            iterative_restart=args.reference_restart,
            iterative_maxiter=args.reference_maxiter,
            ilu_drop_tolerance=args.reference_ilu_drop_tolerance,
            ilu_fill_factor=args.reference_ilu_fill_factor,
        )
        timings["reference"] = time.perf_counter() - started
        print("[4/5] Computing strict L2, broken-H1, pressure and flux errors ...")
        validation, velocity_cell_rms, pressure_cell_rms = finite_element_error_analysis(
            partition, solution, reference, cell_volumes
        )
        if baseline_solution is not None:
            baseline_validation, _, _ = finite_element_error_analysis(
                partition, baseline_solution, reference, cell_volumes
            )
        print(
            f"      cross velocity L2={validation['velocity_relative_l2']:.3%}, "
            f"broken-H1={validation['velocity_relative_broken_h1_seminorm']:.3%}, "
            f"pressure raw L2={validation['pressure_raw_relative_l2']:.3%}, "
            f"flux={validation['outlet_flux_relative_error']:.3%}"
        )
        if baseline_validation is not None:
            print(
                f"      DDPNM velocity L2={baseline_validation['velocity_relative_l2']:.3%}, "
                f"broken-H1={baseline_validation['velocity_relative_broken_h1_seminorm']:.3%}, "
                f"pressure raw L2={baseline_validation['pressure_raw_relative_l2']:.3%}, "
                f"flux={baseline_validation['outlet_flux_relative_error']:.3%}"
            )
    else:
        timings["reference"] = 0.0
        print("[3/5] Reference solve skipped.")
        print("[4/5] Strict FEM error analysis skipped.")

    print("[5/5] Writing the reproducible audit, extension data and plotting input ...")
    boundary_triangles, interface_triangles, interface_triangle_ids = facet_arrays(partition)
    segments, segment_interfaces = flatten_skeleton_segments(partition, skeletons)
    representative = representative_interface(partition)
    representative_skeleton = skeletons[representative]
    representative_transform = transforms[representative]
    full_nodes = np.asarray(representative_skeleton.full_nodes, dtype=np.int32)
    node_to_local = {int(node): index for index, node in enumerate(full_nodes)}
    rep_triangles_global = interface_triangles[interface_triangle_ids == representative]
    rep_triangles = np.asarray(
        [[node_to_local[int(node)] for node in triangle] for triangle in rep_triangles_global],
        dtype=np.int32,
    )
    center_cross_index = representative_skeleton.skeleton_nodes.index(
        representative_skeleton.center_node
    )
    center_normal_column = center_cross_index
    endpoint_cross_index = max(
        range(len(representative_skeleton.skeleton_nodes)),
        key=lambda index: np.linalg.norm(
            points[representative_skeleton.skeleton_nodes[index]]
            - representative_skeleton.saddle
        ),
    )
    endpoint_normal_column = endpoint_cross_index
    extension_normal = representative_transform[:, center_normal_column].reshape(-1, 3)
    extension_endpoint = representative_transform[:, endpoint_normal_column].reshape(-1, 3)

    np.savez_compressed(
        args.out_dir / "skeleton_stokes_ddpnm_results.npz",
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        interface_triangles=interface_triangles,
        interface_triangle_ids=interface_triangle_ids,
        cell_labels=partition.cell_labels,
        cell_volumes=cell_volumes,
        throat_saddles=np.asarray([throat.saddle for throat in partition.throats]),
        throat_normals=np.asarray(partition.interface_normals),
        skeleton_segments=segments,
        skeleton_segment_interfaces=segment_interfaces,
        full_scalar_nodes_per_interface=np.asarray(
            [item.full_scalar_dofs for item in diagnostics], dtype=np.int32
        ),
        skeleton_scalar_nodes_per_interface=np.asarray(
            [item.skeleton_scalar_dofs for item in diagnostics], dtype=np.int32
        ),
        full_normal_dofs_per_interface=np.asarray(
            [item.full_normal_dofs for item in diagnostics], dtype=np.int32
        ),
        active_normal_dofs_per_interface=np.asarray(
            [item.active_normal_dofs for item in diagnostics], dtype=np.int32
        ),
        coefficients=solution.coefficients,
        rhs=solution.rhs,
        moment_residuals=solution.moment_residuals,
        velocity_error_cell_rms=velocity_cell_rms,
        pressure_error_cell_rms=pressure_cell_rms,
        representative_interface=np.asarray([representative], dtype=np.int32),
        representative_nodes=full_nodes,
        representative_coordinates=points[full_nodes],
        representative_triangles=rep_triangles,
        representative_skeleton_nodes=np.asarray(
            [node_to_local[node] for node in representative_skeleton.skeleton_nodes],
            dtype=np.int32,
        ),
        representative_saddle=representative_skeleton.saddle,
        representative_tangent_1=representative_skeleton.tangent_1,
        representative_tangent_2=representative_skeleton.tangent_2,
        representative_center_node=np.asarray(
            [node_to_local[representative_skeleton.center_node]], dtype=np.int32
        ),
        representative_extension_normal=extension_normal,
        representative_extension_endpoint=extension_endpoint,
    )
    np.savez_compressed(
        args.out_dir / "minimum_energy_extension_matrices.npz",
        **{f"interface_{index:03d}": transform for index, transform in enumerate(transforms)},
    )
    write_interface_csv(
        args.out_dir / "skeleton_interface_audit.csv",
        diagnostics,
        partition,
        skeletons,
    )

    scaling_proxy = np.asarray(
        [
            item.skeleton_scalar_dofs / max(np.sqrt(item.full_scalar_dofs), 1.0)
            for item in diagnostics
        ]
    )
    timings["total"] = time.perf_counter() - total_started
    report = {
        "method": "3D DDPNM with saddle cross skeleton and two-sided minimum-Stokes-energy extension",
        "framework": {
            "unchanged": [
                "maximal-ball pore graph and strict saddle interfaces",
                "one local Taylor--Hood Stokes factorisation per pore",
                "local traction-to-velocity response library",
                "global DDPNM moment-continuity assembly",
                "independent local-field reconstruction",
            ],
            "changed": (
                "the facewise constant/affine traction space is replaced by an "
                "an O(h^-1) cardinal scalar-normal cross space extended over each "
                "full face by the two-sided discrete Stokes response energy; the "
                "extensions form a partition of unity, so their coefficients are "
                "cross-node pressures and their sum exactly contains DDPNM P0"
            ),
        },
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "counts": {
            "pores": int(len(np.unique(partition.cell_labels))),
            "interfaces": int(len(partition.interface_pairs)),
            "tetrahedra": int(len(tetrahedra)),
            "full_face_normal_dofs_sum": full_normal_dofs,
            "active_cross_normal_dofs_sum": active_normal_dofs,
            "global_skeleton_unknowns": int(len(solution.global_keys)),
            "skeleton_to_full_ratio": float(
                active_normal_dofs / max(full_normal_dofs, 1)
            ),
            "mean_skeleton_over_sqrt_full_scalar": float(np.mean(scaling_proxy)),
            "max_skeleton_over_sqrt_full_scalar": float(np.max(scaling_proxy)),
        },
        "linear_system": {
            "shape": list(solution.schur_matrix.shape),
            "nonzeros": int(solution.schur_matrix.nnz),
            "minimum_eigenvalue": solution.min_schur_eigenvalue,
            "relative_linear_residual": solution.relative_linear_residual,
            "maximum_moment_residual": solution.max_moment_residual,
            "maximum_local_mass_residual": solution.max_local_mass_residual,
            "maximum_local_linear_residual": solution.max_local_linear_residual,
            "maximum_flux_divergence_discrepancy": (
                solution.max_flux_divergence_discrepancy
            ),
        },
        "extension": {
            "maximum_constraint_residual": float(
                max(item.constraint_residual for item in diagnostics)
            ),
            "maximum_constant_reproduction_residual": float(
                max(item.constant_reproduction_residual for item in diagnostics)
            ),
            "maximum_energy_condition": float(
                max(item.energy_condition for item in diagnostics)
            ),
            "representative_interface": representative,
            "interpretation": (
                "each column is cardinal at one cross node; the columns sum to the "
                "constant face function and minimize the corrected two-sided energy"
            ),
        },
        "boundary_fluxes": {
            "inlet_outward": float(solution.boundary_fluxes["inlet"]),
            "outlet_outward": float(solution.boundary_fluxes["outlet"]),
            "relative_imbalance": float(
                abs(solution.boundary_fluxes["inlet"] + solution.boundary_fluxes["outlet"])
                / max(
                    abs(solution.boundary_fluxes["inlet"]),
                    abs(solution.boundary_fluxes["outlet"]),
                    1.0e-30,
                )
            ),
        },
        "validation": validation,
        "ddpnm_baseline_validation": baseline_validation,
        "reference": None
        if reference is None
        else {
            "mixed_dofs": int(reference.ndofs),
            "solver": reference.solver_method,
            "iterations": int(reference.iterations),
            "relative_linear_residual": float(reference.relative_linear_residual),
        },
        "timings_seconds": timings,
        "interfaces": [asdict(item) for item in diagnostics],
    }
    (args.out_dir / "skeleton_stokes_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    gc.collect()
    print(f"Done in {timings['total']:.1f} s. Outputs: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

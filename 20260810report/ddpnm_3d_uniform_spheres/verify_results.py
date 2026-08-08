from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a 3D original DDPNM run.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/default"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with (args.out_dir / "report.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    with np.load(args.out_dir / "ddpnm_3d_results.npz") as data:
        assert report["counts"]["solid_spheres"] == 27
        assert report["counts"]["maximal_balls"] == 64
        assert report["counts"]["pore_subdomains"] == 64
        assert report["counts"]["internal_interfaces"] == 144
        assert report["interface_system"]["size"] == 144
        assert report["interface_system"]["dofs_per_interface"] == 1
        assert report["interface_system"]["minimum_eigenvalue"] > 0.0
        assert report["interface_system"]["maximum_flux_balance_residual"] < 1.0e-10
        assert report["interface_system"]["relative_linear_residual"] < 1.0e-10
        assert report["interface_system"]["minimum_port_sides"] == 2
        assert report["interface_system"]["maximum_port_sides"] == 2
        assert report["parameters"]["pressure_stabilization"] == 0.0
        assert report["boundary_fluxes"]["relative_imbalance"] < 1.0e-10
        assert max(
            item["dtn_symmetry_error"] for item in report["local_subdomains"]
        ) < 1.0e-10
        assert max(
            item["uniform_traction_kernel_error"]
            for item in report["local_subdomains"]
        ) < 1.0e-10

        audit = report["mesh_audit"]
        assert audit["cad_entities"] == {
            "sphere_surface_patches": 216,
            "inlet_surface_patches": 16,
            "outlet_surface_patches": 16,
            "outer_wall_surface_patches": 64,
            "interface_surface_patches": 144,
            # These are remnants of the nine finite cutting tools outside the
            # fluid volumes.  They have no adjacent 3-D entity and are neither
            # meshed nor assigned to a physical boundary group.
            "orphan_tool_surface_patches": 324,
            "fluid_volumes": 64,
        }
        assert audit["interfaces"]["minimum_triangles"] >= 8
        assert audit["mean_ratio_quality"]["minimum"] > 0.05
        sphere_radius = float(data["sphere_radius"][0])
        sphere_resolution = report["parameters"]["sphere_size"] / sphere_radius
        assert audit["fluid_volume"]["relative_error"] < max(
            0.03, 0.06 * sphere_resolution
        )
        surface = audit["boundary_surfaces"]
        assert surface["triangles_per_sphere"]["minimum"] >= 40
        assert surface["sphere_discrete_to_analytic_area_ratio"]["minimum"] > (
            1.0 - max(0.10, 0.20 * sphere_resolution)
        )
        assert surface["sphere_surface_edges"]["maximum_sagitta_over_radius"] < 0.20

        expected_shapes = {
            "u_ddpnm_piecewise_p1_cell_vertices": (
                report["counts"]["global_tetrahedra"], 4, 3
            ),
            "p_ddpnm_piecewise_p1_cell_vertices": (
                report["counts"]["global_tetrahedra"], 4
            ),
            "u_ddpnm_piecewise_p1_cell_mean": (
                report["counts"]["global_tetrahedra"], 3
            ),
            "p_ddpnm_piecewise_p1_cell_mean": (
                report["counts"]["global_tetrahedra"],
            ),
            "u_ddpnm_trace_average_visualization": (
                report["counts"]["global_vertices"], 3
            ),
            "p_ddpnm_trace_average_visualization": (
                report["counts"]["global_vertices"],
            ),
        }
        for name, shape in expected_shapes.items():
            assert data[name].shape == shape
            assert np.all(np.isfinite(data[name]))
        assert np.linalg.norm(
            data["schur_matrix"] @ data["interface_pressures"] - data["schur_rhs"]
        ) / max(np.linalg.norm(data["schur_rhs"]), 1.0e-30) < 1.0e-10
        pairs = data["throat_pairs"]
        balls = data["maximal_ball_centers"]
        saddles = data["throat_saddles"]
        normals = data["throat_normals"]
        clearances = data["throat_clearances"]
        sphere_centers = data["sphere_centers"]
        sphere_radius = float(data["sphere_radius"][0])
        planes = np.asarray([0.2, 0.5, 0.8])
        for interface_id, (pore_i, pore_j) in enumerate(pairs):
            cell_i = np.asarray(
                [pore_i // 16, (pore_i % 16) // 4, pore_i % 4], dtype=int
            )
            cell_j = np.asarray(
                [pore_j // 16, (pore_j % 16) // 4, pore_j % 4], dtype=int
            )
            changed = np.flatnonzero(cell_i != cell_j)
            assert len(changed) == 1
            axis = int(changed[0])
            assert cell_j[axis] == cell_i[axis] + 1
            expected_normal = np.zeros(3)
            expected_normal[axis] = 1.0
            assert np.allclose(normals[interface_id], expected_normal, atol=1.0e-12)
            assert abs(saddles[interface_id, axis] - planes[cell_i[axis]]) < 1.0e-12
            measured_clearance = min(
                float(np.min(saddles[interface_id])),
                float(np.min(1.0 - saddles[interface_id])),
                float(
                    np.min(
                        np.linalg.norm(
                            sphere_centers - saddles[interface_id][None, :], axis=1
                        )
                        - sphere_radius
                    )
                ),
            )
            assert abs(measured_clearance - clearances[interface_id]) < 1.0e-10

        if report["parameters"]["with_reference"]:
            fem = report["traditional_fem"]
            validation = report["validation"]
            assert fem is not None and validation is not None
            assert fem["same_tetrahedral_mesh_as_ddpnm"]
            assert validation["same_parent_mesh_object"]
            assert fem["mixed_dofs"] == len(data["fem_mixed_solution"])
            assert fem["mixed_dofs"] > report["counts"]["global_tetrahedra"]
            assert fem["matrix_nonzeros"] > fem["mixed_dofs"]
            assert fem["pressure_stabilization"] == 0.0
            reference_rtol = report["parameters"]["reference_rtol"]
            algebraic_tolerance = max(1.0e-8, 20.0 * reference_rtol)
            assert fem["relative_linear_residual"] < algebraic_tolerance
            assert fem["boundary_fluxes"]["inlet_outward"] < 0.0
            assert fem["boundary_fluxes"]["outlet_outward"] > 0.0
            assert (
                fem["boundary_fluxes"]["relative_imbalance"]
                < algebraic_tolerance
            )
            assert (
                fem["energy"]["relative_balance_residual"]
                < algebraic_tolerance
            )
            assert validation["quadrature_degree"] >= 4
            assert (
                validation["maximum_global_to_local_dof_coordinate_mismatch"]
                < 1.0e-10
            )
            for name in (
                "velocity_relative_l2",
                "velocity_relative_broken_h1_seminorm",
                "pressure_raw_relative_l2",
                "pressure_mean_aligned_relative_l2",
                "outlet_flux_relative_error",
                "ddpnm_broken_divergence_l2",
                "fem_divergence_l2",
            ):
                assert np.isfinite(validation[name]) and validation[name] >= 0.0
            n_cells = report["counts"]["global_tetrahedra"]
            for name in ("velocity_error_cell_rms", "pressure_error_cell_rms"):
                assert data[name].shape == (n_cells,)
                assert np.all(np.isfinite(data[name]))
                assert np.all(data[name] >= 0.0)
            assert data["u_reference"].shape == (
                report["counts"]["global_vertices"],
                3,
            )
            assert data["p_reference"].shape == (
                report["counts"]["global_vertices"],
            )
            assert np.all(np.isfinite(data["fem_mixed_solution"]))
            assert data["fem_velocity_p2_dof_values"].shape[1] == 3
            assert data["fem_pressure_p1_dof_values"].shape == (
                report["counts"]["global_vertices"],
            )
            assert len(data["error_slice_points"]) > 0
            assert len(data["error_slice_triangles"]) > 0
            assert abs(float(data["error_slice_z"][0]) - 0.48) < 1.0e-12
            assert np.all(np.isfinite(data["error_slice_u_fem"]))
            assert np.all(np.isfinite(data["error_slice_p_fem"]))
            assert np.all(np.isfinite(data["error_slice_u_ddpnm"]))
            assert np.all(np.isfinite(data["error_slice_p_ddpnm"]))
        else:
            assert report.get("traditional_fem") is None
            assert report["validation"] is None

    xdmf_path = args.out_dir / "ddpnm_3d_fields.xdmf"
    h5_path = args.out_dir / "ddpnm_3d_fields.h5"
    assert xdmf_path.is_file() and h5_path.is_file()
    attribute_names = {
        item.attrib["Name"]
        for item in ET.parse(xdmf_path).getroot().iter("Attribute")
    }
    assert {
        "u_ddpnm_trace_average_visualization",
        "p_ddpnm_trace_average_visualization",
        "u_ddpnm_piecewise_p1_cell_mean",
        "p_ddpnm_piecewise_p1_cell_mean",
    }.issubset(attribute_names)
    if report["parameters"]["with_reference"]:
        assert {
            "u_reference",
            "p_reference",
            "velocity_error_cell_rms",
            "pressure_error_cell_rms",
        }.issubset(attribute_names)
        assert (args.out_dir / "05_fem_ddpnm_error_fields.png").is_file()

    zones = report["mesh_audit"]["zones"]
    target_sizes = {
        "near_sphere": report["parameters"]["sphere_size"],
        "near_outer_boundary": report["parameters"]["boundary_size"],
        "near_interface": report["parameters"]["interface_size"],
    }
    for name, target in target_sizes.items():
        local_h = zones[name]["median_diameter"]
        assert zones[name]["tetrahedra"] > 0
        assert np.isfinite(local_h)
        # A tetrahedron diameter is larger than its characteristic edge size.
        assert local_h < 3.5 * target
    print("3D original DDPNM verification passed.")
    print(
        "counts: 27 spheres, 64 maximal balls/subdomains, "
        "144 interfaces and 144 global interface unknowns"
    )
    print(
        f"max flux residual = "
        f"{report['interface_system']['maximum_flux_balance_residual']:.3e}"
    )


if __name__ == "__main__":
    main()

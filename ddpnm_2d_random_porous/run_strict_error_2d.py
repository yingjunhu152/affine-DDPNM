from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter
import numpy as np

from ddpnm2d.geometry import PARTICLES, build_partition
from ddpnm_core.io import topology_arrays
from ddpnm_core.reconstruction import (
    reconstruct_parent_vertices,
    reference_parent_vertices,
)

from ddpnm2d.solver import solve_ddpnm, solve_reference
from ddpnm2d.validation import finite_element_error_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict identical-mesh P2-P1 error analysis for 2D P0-DDPNM."
    )
    parser.add_argument("--mesh-size", type=float, default=0.04)
    parser.add_argument("--wall-size", type=float, default=0.018)
    parser.add_argument("--throat-size", type=float, default=0.010)
    parser.add_argument("--wall-band", type=float, default=0.065)
    parser.add_argument("--throat-band", type=float, default=0.045)
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--inlet-pressure", type=float, default=1.0)
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--pressure-stabilization", type=float, default=1.0e-10)
    parser.add_argument("--quadrature-degree", type=int, default=6)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/strict_p0_comparison")
    )
    return parser.parse_args()


def draw_particles(ax) -> None:
    for x, y, radius in PARTICLES:
        ax.add_patch(
            plt.Circle(
                (x, y), radius, facecolor="white", edgecolor="none", zorder=5
            )
        )


def plot_strict_error_fields(
    points: np.ndarray,
    triangles: np.ndarray,
    fields: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    pressure_error = fields["pressure_error_cell_mean"]
    pressure_limit = max(float(np.max(np.abs(pressure_error))), 1.0e-14)
    panels = (
        (
            fields["fem_speed_cell_centroid"],
            r"FEM velocity magnitude $|u_h^{\rm FE}|$",
            "turbo",
            None,
        ),
        (
            fields["ddpnm_speed_cell_centroid"],
            r"P0-DDPNM velocity magnitude $|u_h^{\rm DD}|$",
            "turbo",
            None,
        ),
        (
            fields["velocity_error_cell_rms"],
            r"Cell RMS velocity error",
            "turbo",
            None,
        ),
        (
            pressure_error,
            r"Cell-mean pressure error $p_h^{\rm DD}-p_h^{\rm FE}$",
            "RdBu_r",
            TwoSlopeNorm(vmin=-pressure_limit, vcenter=0.0, vmax=pressure_limit),
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.0), constrained_layout=True)
    for panel, (ax, (values, title, cmap, norm)) in enumerate(
        zip(axes.flat, panels, strict=True)
    ):
        artist = ax.tripcolor(
            triangulation,
            facecolors=values,
            shading="flat",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        draw_particles(ax)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
        ax.set_title(title, fontsize=12.0)
        ax.text(
            0.5,
            -0.05,
            f"({chr(97 + panel)})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12.5,
        )
        colorbar = fig.colorbar(artist, ax=ax, shrink=0.88, pad=0.025)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        colorbar.formatter = formatter
        colorbar.update_ticks()
    fig.suptitle(
        "Strict 2D P0-DDPNM error on the identical Taylor–Hood mesh\n"
        "cellwise broken-domain evaluation; no interface trace averaging",
        fontsize=13.2,
    )
    fig.savefig(out_dir / "strict_2d_p0_ddpnm_error_fields.png", dpi=260)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("[1/5] Building the analytic saddle-cut refined 2D mesh...", flush=True)
    partition = build_partition(
        mesh_size=args.mesh_size,
        wall_size=args.wall_size,
        throat_size=args.throat_size,
        wall_band=args.wall_band,
        throat_band=args.throat_band,
    )
    points, triangles = topology_arrays(partition.mesh)
    print(
        f"      {len(triangles)} triangles, "
        f"{len(np.unique(partition.cell_labels))} subdomains, "
        f"{len(partition.interface_pairs)} interfaces",
        flush=True,
    )
    print("[2/5] Solving original P0-DDPNM (one normal traction/interface)...", flush=True)
    ddpnm = solve_ddpnm(
        partition,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
        interface_order=0,
    )
    print(
        f"      Schur size {ddpnm.schur_matrix.shape[0]}, "
        f"maximum interface residual {ddpnm.max_mass_residual:.3e}",
        flush=True,
    )
    print("[3/5] Solving monolithic P2-P1 FEM on the identical mesh...", flush=True)
    reference = solve_reference(
        partition.mesh,
        viscosity=args.viscosity,
        inlet_pressure=args.inlet_pressure,
        outlet_pressure=args.outlet_pressure,
        pressure_stabilization=args.pressure_stabilization,
    )
    print(
        f"      {reference.ndofs} dofs, residual {reference.relative_linear_residual:.3e}, "
        f"mass imbalance {reference.relative_mass_imbalance:.3e}",
        flush=True,
    )
    print("[4/5] Integrating strict broken-domain P2-P1 errors...", flush=True)
    metrics, fields = finite_element_error_analysis(
        partition,
        ddpnm,
        reference,
        quadrature_degree=args.quadrature_degree,
    )
    u_dd, p_dd, _ = reconstruct_parent_vertices(partition, ddpnm)
    u_fem, p_fem = reference_parent_vertices(reference)
    pressure_shift = float(np.mean(p_dd - p_fem))
    vertex_metrics = {
        "velocity_relative_l2": float(
            np.linalg.norm(u_dd - u_fem) / max(np.linalg.norm(u_fem), 1.0e-30)
        ),
        "pressure_mean_aligned_relative_l2": float(
            np.linalg.norm(p_dd - pressure_shift - p_fem)
            / max(np.linalg.norm(p_fem), 1.0e-30)
        ),
        "pressure_shift": pressure_shift,
    }
    print("[5/5] Writing strict metrics, portable fields and paper-style plot...", flush=True)
    plot_strict_error_fields(points, triangles, fields, args.out_dir)
    np.savez_compressed(
        args.out_dir / "strict_2d_p0_error_data.npz",
        points=points,
        triangles=triangles,
        cell_labels=partition.cell_labels,
        **fields,
    )
    parameters = vars(args).copy()
    parameters["out_dir"] = str(args.out_dir)
    report = {
        "method": "2D original P0-DDPNM versus identical-mesh monolithic FEM",
        "error_definition": (
            "sixth-order cell integration of independent local P2-P1 fields; "
            "no averaging across interfaces"
        ),
        "parameters": parameters,
        "counts": {
            "particles": len(PARTICLES),
            "subdomains": int(len(np.unique(partition.cell_labels))),
            "interfaces": len(partition.interface_pairs),
            "triangles": len(triangles),
            "vertices": len(points),
            "ddpnm_interface_unknowns": int(ddpnm.schur_matrix.shape[0]),
            "fem_mixed_dofs": reference.ndofs,
        },
        "strict_validation": metrics,
        "legacy_vertex_sampled_validation": vertex_metrics,
        "systems": {
            "ddpnm_maximum_interface_residual": ddpnm.max_mass_residual,
            "ddpnm_minimum_schur_eigenvalue": ddpnm.min_schur_eigenvalue,
            "fem_relative_linear_residual": reference.relative_linear_residual,
            "fem_relative_mass_imbalance": reference.relative_mass_imbalance,
        },
        "wall_time_seconds": time.perf_counter() - started,
        "comparison_warning": (
            "These norms are directly comparable to the 3D strict norms, but the "
            "2D random-circle and 3D regular-sphere geometries are not matched benchmarks."
        ),
    }
    (args.out_dir / "strict_2d_p0_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.out_dir / "strict_2d_p0_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
    print(
        "Strict errors: "
        f"L2(u)={metrics['velocity_relative_l2']:.2%}, "
        f"broken-H1(u)={metrics['velocity_relative_broken_h1_seminorm']:.2%}, "
        f"raw L2(p)={metrics['pressure_raw_relative_l2']:.2%}, "
        f"mean-aligned L2(p)={metrics['pressure_mean_aligned_relative_l2']:.2%}, "
        f"flow={metrics['outlet_flux_relative_error']:.2%}",
        flush=True,
    )
    print(f"Outputs: {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

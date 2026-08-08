from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("outputs_presentation")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    summarize_preconditioners()
    summarize_pressure_stabilization()
    summarize_size_timings()


def summarize_preconditioners() -> None:
    out_dir = ROOT / "01_reduction_validation"
    rows = []
    for path in sorted(out_dir.glob("preconditioner_*_27holes_c4/validation_summary.json")):
        data = load_json(path)
        rows.append(
            {
                "case": path.parent.name,
                "preconditioner": data.get("schur_preconditioner"),
                "holes": data.get("holes"),
                "cells_per_axis": data.get("cells_per_axis"),
                "mixed_dofs": data.get("mixed_dofs"),
                "hoddpnm_interface_dofs": data.get("hoddpnm_interface_dofs"),
                "hoddpnm_active_schur_dofs": data.get("hoddpnm_active_schur_dofs"),
                "hoddpnm_active_dof_ratio": data.get("hoddpnm_active_dof_ratio"),
                "hoddpnm_known_fixed_dofs": data.get("hoddpnm_known_fixed_dofs"),
                "hoddpnm_free_interior_dofs_eliminated": data.get("hoddpnm_free_interior_dofs_eliminated"),
                "hoddpnm_interior_dofs_eliminated": data.get("hoddpnm_interior_dofs_eliminated"),
                "schur_iterations": data.get("schur_iterations"),
                "schur_relative_residual": data.get("schur_relative_residual"),
                "full_solve_time_seconds": data.get("full_solve_time_seconds"),
                "hoddpnm_solve_time_seconds": data.get("hoddpnm_solve_time_seconds"),
                "velocity_l2_rel": data["errors_velocity_p2_all"]["l2_rel"],
                "pressure_mean_aligned_l2_rel": data["errors_pressure_p1_mean_aligned"]["l2_rel"],
            }
        )
    write_csv(out_dir / "preconditioner_comparison.csv", rows)
    if rows:
        labels = [str(row["preconditioner"]) for row in rows]
        iterations = [float(row["schur_iterations"] or 0.0) for row in rows]
        residuals = [float(row["schur_relative_residual"] or 0.0) for row in rows]
        fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
        ax1.bar(labels, iterations, color="#4f8bc9", alpha=0.82)
        ax1.set_ylabel("GMRES iterations")
        ax1.set_xlabel("Schur preconditioner")
        ax2 = ax1.twinx()
        ax2.semilogy(labels, residuals, marker="o", color="#c44e52", linewidth=2)
        ax2.set_ylabel("relative residual")
        ax1.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "preconditioner_comparison.png", dpi=180)
        plt.close(fig)


def summarize_pressure_stabilization() -> None:
    out_dir = ROOT / "04_reference_error"
    rows = []
    for path in sorted(out_dir.glob("pressure_stab_*/validation_summary.json")):
        data = load_json(path)
        pressure = data.get("pressure_range", {})
        rows.append(
            {
                "case": path.parent.name,
                "pressure_stabilization": data.get("pressure_stabilization"),
                "holes": data.get("holes"),
                "cells_per_axis": data.get("cells_per_axis"),
                "mixed_dofs": data.get("mixed_dofs"),
                "pressure_mean": pressure.get("fem_mean_p1"),
                "pressure_min": pressure.get("fem_min"),
                "pressure_max": pressure.get("fem_max"),
                "zero_mean_pressure_min": pressure.get("fem_zero_mean_min_p1"),
                "zero_mean_pressure_max": pressure.get("fem_zero_mean_max_p1"),
                "velocity_l2_rel": data["errors_velocity_p2_all"]["l2_rel"],
                "pressure_l2_rel_raw": data["errors_pressure_p1_all"]["l2_rel"],
                "pressure_l2_rel_mean_aligned": data["errors_pressure_p1_mean_aligned"]["l2_rel"],
                "schur_iterations": data.get("schur_iterations"),
                "schur_relative_residual": data.get("schur_relative_residual"),
                "full_solve_time_seconds": data.get("full_solve_time_seconds"),
                "hoddpnm_solve_time_seconds": data.get("hoddpnm_solve_time_seconds"),
            }
        )
    rows.sort(key=lambda row: float(row["pressure_stabilization"] or 0.0), reverse=True)
    write_csv(out_dir / "pressure_stabilization_sensitivity.csv", rows)
    if rows:
        x = [float(row["pressure_stabilization"]) for row in rows]
        pressure_err = [float(row["pressure_l2_rel_mean_aligned"]) for row in rows]
        velocity_err = [float(row["velocity_l2_rel"]) for row in rows]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.loglog(x, pressure_err, marker="s", label="pressure, mean-aligned")
        ax.loglog(x, velocity_err, marker="o", label="velocity")
        ax.invert_xaxis()
        ax.set_xlabel("pressure stabilization")
        ax.set_ylabel("relative Schur/full difference")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "pressure_stabilization_sensitivity.png", dpi=180)
        plt.close(fig)


def summarize_size_timings() -> None:
    out_dir = ROOT / "01_reduction_validation"
    rows = []
    for path in sorted(out_dir.glob("size_h*_c*/validation_summary.json")):
        data = load_json(path)
        rows.append(
            {
                "case": path.parent.name,
                "holes": data.get("holes"),
                "cells_per_axis": data.get("cells_per_axis"),
                "mixed_dofs": data.get("mixed_dofs"),
                "hoddpnm_interface_dofs": data.get("hoddpnm_interface_dofs"),
                "hoddpnm_active_schur_dofs": data.get("hoddpnm_active_schur_dofs"),
                "hoddpnm_active_dof_ratio": data.get("hoddpnm_active_dof_ratio"),
                "hoddpnm_known_fixed_dofs": data.get("hoddpnm_known_fixed_dofs"),
                "hoddpnm_free_interior_dofs_eliminated": data.get("hoddpnm_free_interior_dofs_eliminated"),
                "hoddpnm_interior_dofs_eliminated": data.get("hoddpnm_interior_dofs_eliminated"),
                "assembly_time_seconds": data.get("assembly_time_seconds"),
                "full_solve_time_seconds": data.get("full_solve_time_seconds"),
                "hoddpnm_solve_time_seconds": data.get("hoddpnm_solve_time_seconds"),
                "total_time_seconds": data.get("total_time_seconds"),
                "schur_iterations": data.get("schur_iterations"),
                "schur_relative_residual": data.get("schur_relative_residual"),
                "pressure_mean_aligned_l2_rel": data["errors_pressure_p1_mean_aligned"]["l2_rel"],
            }
        )
    rows.sort(key=lambda row: (int(row["holes"] or 0), int(row["cells_per_axis"] or 0)))
    write_csv(out_dir / "size_timing_comparison.csv", rows)
    if rows:
        labels = [f"{row['holes']}h-c{row['cells_per_axis']}" for row in rows]
        assembly = [float(row["assembly_time_seconds"] or 0.0) for row in rows]
        full = [float(row["full_solve_time_seconds"] or 0.0) for row in rows]
        hodd = [float(row["hoddpnm_solve_time_seconds"] or 0.0) for row in rows]
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.semilogy(labels, assembly, marker="^", label="assembly")
        ax.semilogy(labels, full, marker="o", label="full solve")
        ax.semilogy(labels, hodd, marker="s", label="HODDPNM Schur")
        ax.set_xlabel("case")
        ax.set_ylabel("seconds")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        fig.autofmt_xdate(rotation=25)
        fig.tight_layout()
        fig.savefig(out_dir / "size_timing_comparison.png", dpi=180)
        plt.close(fig)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

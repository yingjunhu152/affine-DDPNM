from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dirs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--name", default="pressure_compressed_validation_summary")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [case_row(case_dir) for case_dir in args.case_dirs]
    write_csv(args.out_dir / f"{args.name}.csv", rows)
    write_markdown(args.out_dir / f"{args.name}.md", rows)
    print(f"wrote {args.out_dir / f'{args.name}.csv'}")
    print(f"wrote {args.out_dir / f'{args.name}.md'}")


def case_row(case_dir: Path) -> dict[str, object]:
    summary_path = case_dir / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["fem_integral_validation"]
    solver = summary["hoddpnm_solver_info"]
    return {
        "case": case_dir.name,
        "pore_voxels": summary["pore_voxels"],
        "mixed_dofs": summary["mixed_dofs"],
        "active_boundary_dofs": summary.get("hoddpnm_active_boundary_dofs", summary["hoddpnm_interface_dofs"]),
        "eliminated_interior_dofs": summary["hoddpnm_interior_dofs_eliminated"],
        "pressure_boundary_dofs": summary["hoddpnm_pressure_boundary_dofs"],
        "pressure_eliminated_dofs": summary["hoddpnm_pressure_interior_dofs_eliminated"],
        "gmres_iterations": solver.get("iterations", ""),
        "preconditioner": solver.get("preconditioner", ""),
        "schur_residual": solver.get("relative_residual", ""),
        "velocity_l2_rel": metrics["velocity_l2_rel"],
        "pressure_l2_rel": metrics["pressure_l2_rel"],
        "converged": solver.get("converged", ""),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "case",
        "pore_voxels",
        "mixed_dofs",
        "active_boundary_dofs",
        "eliminated_interior_dofs",
        "pressure_boundary_dofs",
        "pressure_eliminated_dofs",
        "gmres_iterations",
        "preconditioner",
        "schur_residual",
        "velocity_l2_rel",
        "pressure_l2_rel",
        "converged",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Pressure-Compressed HODDPNM Stokes Validation Summary",
        "",
        "All rows are same-discrete-system FEniCSx Taylor-Hood P2-P1 solver-equivalence checks.",
        "The HODDPNM solve uses matrix-free GMRES Schur by default; dense exact Schur is not the formal reporting path.",
        "",
        "| case | pore voxels | mixed dofs | active boundary dofs | eliminated interior dofs | pressure boundary dofs | pressure eliminated dofs | GMRES iterations | preconditioner | Schur residual | velocity L2 rel | pressure L2 rel | converged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {pore_voxels} | {mixed_dofs} | {active_boundary_dofs} | {eliminated_interior_dofs} | "
            "{pressure_boundary_dofs} | {pressure_eliminated_dofs} | {gmres_iterations} | {preconditioner} | {schur_residual} | "
            "{velocity_l2_rel} | {pressure_l2_rel} | {converged} |".format(
                **{key: format_value(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "Use `velocity_l2_rel`, `pressure_l2_rel`, and `schur_residual` as the main quantitative evidence.",
            "The table does not claim analytic or high-resolution physical true error.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6e}"
    return str(value)


if __name__ == "__main__":
    main()

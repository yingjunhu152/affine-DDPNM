from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CASES = {
    "coarse": {
        "holes_per_axis": 2,
        "cells_per_axis": 3,
        "steps": 20,
        "plot_steps": [1, 10, 20],
    },
    "medium": {
        "holes_per_axis": 3,
        "cells_per_axis": 4,
        "steps": 20,
        "plot_steps": [1, 10, 20],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", choices=sorted(DEFAULT_CASES), default=["coarse", "medium"])
    parser.add_argument("--out-root", type=Path, default=Path("outputs_presentation/02_twophase_timesteps/minimal_validation"))
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--cfl-limit", type=float, default=0.5)
    parser.add_argument("--transport-scale", type=float, default=0.18)
    parser.add_argument("--sw-initial", type=float, default=0.2)
    parser.add_argument("--sw-inlet", type=float, default=1.0)
    parser.add_argument("--mu-water", type=float, default=1.0)
    parser.add_argument("--mu-oil", type=float, default=5.0)
    parser.add_argument("--sw-residual", type=float, default=0.2)
    parser.add_argument("--so-residual", type=float, default=0.2)
    parser.add_argument("--corey-nw", type=float, default=2.0)
    parser.add_argument("--corey-no", type=float, default=2.0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_name in args.cases:
        case = DEFAULT_CASES[case_name]
        run_dir = args.out_root / f"{case_name}_{case['holes_per_axis'] ** 3}holes_c{case['cells_per_axis']}_steps{case['steps']}"
        run_timestep_case(args, case, run_dir)
        rows.append(summarize_case(case_name, case, run_dir))

    write_csv(args.out_root / "minimal_validation_summary.csv", rows)
    (args.out_root / "minimal_validation_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(args.out_root / "minimal_validation_summary.md", rows)
    print(f"wrote {args.out_root / 'minimal_validation_summary.csv'}")
    print(f"wrote {args.out_root / 'minimal_validation_summary.md'}")


def run_timestep_case(args, case: dict[str, object], run_dir: Path) -> None:
    residual_injected = 1.0 - args.sw_inlet
    if not (0.0 <= residual_injected <= 1.0):
        raise SystemExit("--sw-inlet must be in [0, 1]")
    cmd = [
        str(args.python),
        "run_twophase_fenicsx_p2p1_hoddpnm_timesteps.py",
        "--holes-per-axis",
        str(case["holes_per_axis"]),
        "--cells-per-axis",
        str(case["cells_per_axis"]),
        "--steps",
        str(case["steps"]),
        "--plot",
        "--plot-steps",
        *[str(step) for step in case["plot_steps"]],
        "--dt",
        str(args.dt),
        "--cfl-limit",
        str(args.cfl_limit),
        "--transport-scale",
        str(args.transport_scale),
        "--residual-original",
        str(args.sw_initial),
        "--residual-injected",
        str(residual_injected),
        "--sw-initial",
        str(args.sw_initial),
        "--sw-inlet",
        str(args.sw_inlet),
        "--mu-water",
        str(args.mu_water),
        "--mu-oil",
        str(args.mu_oil),
        "--sw-residual",
        str(args.sw_residual),
        "--so-residual",
        str(args.so_residual),
        "--corey-nw",
        str(args.corey_nw),
        "--corey-no",
        str(args.corey_no),
        "--pressure-boundary-mode",
        "interface-anchors",
        "--schur-solver",
        "gmres",
        "--schur-preconditioner",
        "ilu",
        "--out-dir",
        str(run_dir),
    ]
    run(cmd)


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def summarize_case(case_name: str, case: dict[str, object], run_dir: Path) -> dict[str, object]:
    history = read_history(run_dir / "history.csv")
    first = history[0]
    last = history[-1]
    return {
        "case": case_name,
        "run_dir": str(run_dir),
        "holes": int(case["holes_per_axis"]) ** 3,
        "cells_per_axis": int(case["cells_per_axis"]),
        "steps": int(case["steps"]),
        "mean_sw_step1": value(first, "mean_S2"),
        "mean_sw_final": value(last, "mean_S2"),
        "saturation_min_over_time": min(value(row, "saturation_min") for row in history),
        "saturation_max_over_time": max(value(row, "saturation_max") for row in history),
        "max_relative_mass_error": max(abs(value(row, "relative_mass_error")) for row in history),
        "max_bookkeeping_mass_error": max(abs(value(row, "mass_error")) for row in history),
        "max_cfl": max(value(row, "max_cfl") for row in history),
        "min_stable_dt": min(value(row, "stable_dt") for row in history),
        "max_velocity_error": max(value(row, "velocity_error") for row in history),
        "max_pressure_error": max(value(row, "pressure_error") for row in history),
        "mixed_dofs": int(float(last["mixed_dofs"])),
        "hoddpnm_active_schur_dofs": int(float(last["hoddpnm_active_schur_dofs"])),
        "hoddpnm_active_dof_ratio": value(last, "hoddpnm_active_dof_ratio"),
        "schur_iterations_max": max(int(float(row["schur_iterations"])) for row in history),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Minimal Two-Phase Validation",
        "",
        "These cube-minus-sphere cases exercise the Corey graph-transport timestep, inlet boundary-flux injection, natural outlet outflow, CFL diagnostics, mass diagnostics, and full-FEM/HODDPNM Stokes comparison.",
        "",
        "| case | holes | cpa | steps | final mean Sw | Sw range | max rel mass err | max CFL | active Schur dofs | max velocity err | max pressure err |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['holes']} | {row['cells_per_axis']} | {row['steps']} | "
            f"{float(row['mean_sw_final']):.4f} | "
            f"[{float(row['saturation_min_over_time']):.3f}, {float(row['saturation_max_over_time']):.3f}] | "
            f"{float(row['max_relative_mass_error']):.2e} | "
            f"{float(row['max_cfl']):.3f} | "
            f"{row['hoddpnm_active_schur_dofs']} ({100.0 * float(row['hoddpnm_active_dof_ratio']):.2f}%) | "
            f"{float(row['max_velocity_error']):.2e} | "
            f"{float(row['max_pressure_error']):.2e} |"
        )
    lines.extend(
        [
            "",
            "Note: inlet saturation is imposed through boundary flux, not by resetting boundary vertices. This is still the graph-edge Corey transport validation path; the stricter cell-wise finite-volume face-flux transport remains the next model upgrade.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

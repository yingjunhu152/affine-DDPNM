"""Run the non-Korteweg verification and production campaign.

The campaign is restartable: a completed, valid JSON report is skipped, while
an incomplete non-empty directory is rejected instead of being overwritten.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT / "outputs" / "baseline_campaign_20260822"
DEFAULT_BENTHEIMER = (
    PROJECT.parent / "affine_ddpnm_twophase" / "outputs" / "experiment_v2"
    / "bentheimer_inverted_cartesian_mesh_c6" / "bentheimer_voxel_pore_mesh.msh"
)


def _tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _valid_report(path: Path, expected_arms: int) -> bool:
    if not path.is_file():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("results", [])
    return len(rows) == expected_arms and all(row.get("converged") for row in rows)


def _run(script: str, arguments: list[str], out_dir: Path, report_name: str, arms: int) -> None:
    report = out_dir / report_name
    if _valid_report(report, arms):
        print(f"[campaign] skip completed {out_dir.name}", flush=True)
        return
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Incomplete non-empty campaign directory: {out_dir}")
    command = [sys.executable, "-u", str(PROJECT / script), *arguments, "--out-dir", str(out_dir)]
    print("[campaign]", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)
    if not _valid_report(report, arms):
        raise RuntimeError(f"Campaign run did not produce a valid report: {report}")


def time_step_stage(root: Path, bentheimer: Path) -> None:
    for geometry, script, extra in (
        ("random27", "run_random27_six.py", []),
        ("bentheimer", "run_bentheimer_six.py", ["--mesh-file", str(bentheimer)]),
    ):
        for dt in (1.0, 0.5, 0.25):
            out_dir = root / "time_step" / f"{geometry}_dt_{_tag(dt)}"
            _run(
                script,
                ["--arms", "FEM-frozen", "--dt", str(dt), "--t-final", "1", *extra],
                out_dir,
                f"{geometry}_six_arms.json",
                1,
            )


def pod_stage(root: Path, bentheimer: Path) -> None:
    for tolerance in (1.0e-6, 1.0e-8, 1.0e-10):
        out_dir = root / "pod" / f"bentheimer_tol_{_tag(tolerance)}"
        _run(
            "run_bentheimer_six.py",
            [
                "--arms", "FEM-frozen,Affine-frozen",
                "--dt", "0.1", "--t-final", "0.1",
                "--affine-pod-tol", str(tolerance),
                "--mesh-file", str(bentheimer),
            ],
            out_dir,
            "bentheimer_six_arms.json",
            2,
        )


def production_stage(root: Path, bentheimer: Path) -> None:
    for dt in (1.0, 0.5):
        _run(
            "run_random27_six.py",
            ["--arms", "all", "--dt", str(dt), "--t-final", "6"],
            root / "production" / f"random27_dt_{_tag(dt)}",
            "random27_six_arms.json",
            6,
        )
        _run(
            "run_bentheimer_six.py",
            [
                "--arms", "all", "--dt", str(dt), "--t-final", "6",
                "--mesh-file", str(bentheimer),
            ],
            root / "production" / f"bentheimer_dt_{_tag(dt)}",
            "bentheimer_six_arms.json",
            6,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("time-step", "pod", "production", "all"), default="all")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bentheimer-mesh", type=Path, default=DEFAULT_BENTHEIMER)
    args = parser.parse_args()
    if not args.bentheimer_mesh.is_file():
        raise FileNotFoundError(args.bentheimer_mesh)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage in ("time-step", "all"):
        time_step_stage(args.root, args.bentheimer_mesh)
    if args.stage in ("pod", "all"):
        pod_stage(args.root, args.bentheimer_mesh)
    if args.stage in ("production", "all"):
        production_stage(args.root, args.bentheimer_mesh)


if __name__ == "__main__":
    main()

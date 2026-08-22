"""Run the six preliminary inverted-Bentheimer experiments."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

from schbench.config import Arm, InitialProfile, Numerics, Physics, ViscousForm
from schbench.experiment import run_bentheimer
from schbench.geometry import DEFAULT_BENTHEIMER_MESH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="all")
    parser.add_argument("--mesh-file", type=Path, default=DEFAULT_BENTHEIMER_MESH)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--t-final", type=float, default=6.0)
    parser.add_argument("--newton-tol", type=float, default=1.0e-8)
    parser.add_argument("--newton-max", type=int, default=18)
    parser.add_argument("--sfi-tol", type=float, default=1.0e-4)
    parser.add_argument("--sfi-max", type=int, default=12)
    parser.add_argument("--sfi-relaxation", type=float, default=0.65)
    parser.add_argument("--mobility", type=float, default=2.0e-4)
    parser.add_argument("--surface-tension", type=float, default=2.0e-3)
    parser.add_argument("--epsilon-factor", type=float, default=1.5)
    parser.add_argument("--affine-pod-tol", type=float, default=1.0e-8)
    parser.add_argument(
        "--viscous-form", choices=[item.value for item in ViscousForm],
        default=ViscousForm.GRADIENT.value,
    )
    parser.add_argument(
        "--initial-profile", choices=[item.value for item in InitialProfile],
        default=InitialProfile.DISCONTINUOUS.value,
    )
    parser.add_argument("--initial-transition-length", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "bentheimer_rewrite")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mesh_file.is_file():
        raise FileNotFoundError(args.mesh_file)
    physics = Physics(
        mobility=args.mobility,
        surface_tension=args.surface_tension,
        epsilon_factor=args.epsilon_factor,
        initial_profile=InitialProfile(args.initial_profile),
        initial_transition_length=args.initial_transition_length,
    )
    numerics = Numerics(
        dt=args.dt,
        t_final=args.t_final,
        newton_tolerance=args.newton_tol,
        newton_max_iterations=args.newton_max,
        sfi_tolerance=args.sfi_tol,
        sfi_max_iterations=args.sfi_max,
        sfi_relaxation=args.sfi_relaxation,
        affine_pod_tolerance=args.affine_pod_tol,
        viscous_form=ViscousForm(args.viscous_form),
    )
    report = run_bentheimer(
        Arm.parse_many(args.arms), physics, numerics, args.out_dir, args.mesh_file
    )
    print(json.dumps(report["results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

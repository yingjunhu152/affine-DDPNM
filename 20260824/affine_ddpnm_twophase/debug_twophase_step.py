#!/usr/bin/env python3
"""Run one conservative transport step and print the mass ledger.

This diagnostic uses only modules shipped in this project.  It is intended to
be run in the same FEniCSx environment as the benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ddpnm_core.fem_utils import solve_reference
from postprocess.fields import mixed_solution_to_p1
from random_porous import build_partition
from two_phase_transport import solve_two_phase


def run_case(partition, velocity: np.ndarray, supg: bool) -> None:
    result = solve_two_phase(
        partition.mesh,
        velocity,
        dt=0.1,
        t_final=0.1,
        sw_initial=0.2,
        sw_inlet=0.8,
        picard_max_iters=20,
        supg=supg,
    )
    history = result["history"]
    print(f"SUPG={supg}")
    print(f"  initial/final water mass: {history['mass'][0]:.12e} / {history['mass'][-1]:.12e}")
    print(f"  flux-budget water mass:   {history['budget_mass'][-1]:.12e}")
    print(f"  step balance residual:    {history['mass_balance_absolute_residual'][-1]:.3e}")
    print(f"  cumulative residual:      {history['cumulative_budget_residual'][-1]:.3e}")
    print(f"  limiter mass residual:    {history['limiter_mass_residual'][-1]:.3e}")
    print(f"  saturation range:         [{history['min_s'][-1]:.6f}, {history['max_s'][-1]:.6f}]")
    print(f"  raw range:                [{history['raw_min_s_before_limiter'][-1]:.6f}, "
          f"{history['raw_max_s_before_limiter'][-1]:.6f}]")
    print(f"  Picard converged:         {bool(history['picard_converged'][-1])}")


def main() -> None:
    mesh_file = PROJECT_DIR / "outputs" / "debug_twophase" / "random_sphere_partition.msh"
    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    partition = build_partition(mesh_file=mesh_file)
    reference = solve_reference(partition.mesh)
    velocity, _pressure = mixed_solution_to_p1(reference.W, reference.solution)
    run_case(partition, velocity, supg=False)
    run_case(partition, velocity, supg=True)


if __name__ == "__main__":
    main()

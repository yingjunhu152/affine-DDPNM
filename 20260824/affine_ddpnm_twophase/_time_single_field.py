"""Time the two-phase transport for a single velocity field (300 steps),
to check whether the Affine-9 field is intrinsically slower than the FEM
field (the full benchmark reported 850 s vs 548 s; the smoke run showed no
difference).  Run with the fenicsx environment."""
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from random_porous import build_partition
import two_phase_transport as tp

field = sys.argv[1] if len(sys.argv) > 1 else "u_FEM"
out_dir = PROJECT_DIR / "outputs" / "benchmark_twophase"

partition = build_partition(
    mesh_size=0.13, sphere_size=0.065, boundary_size=0.085, interface_size=0.075,
    sphere_band=0.15, boundary_band=0.13, interface_band=0.12,
    mesh_file=out_dir / "random_sphere_partition.msh",
)
data = np.load(out_dir / "twophase_fields.npz")
u = data[field]
print(f"field {field}: shape {u.shape}")

t0 = time.perf_counter()
result = tp.solve_two_phase(partition.mesh, u, t_final=30.0, dt=0.1)
elapsed = time.perf_counter() - t0
print(f"{field}: {elapsed:.1f} s for 300 steps ({elapsed / 300:.3f} s/step), "
      f"recovery={result['history']['recovery'][-1]:.4f}, balance={result['history']['mass_balance_relative_residual'][-1]:.2e}")

"""FEM-only two-phase run with per-step snapshots (t = 0..30), written to
outputs/fem_snapshots/twophase_fields.npz.  Reuses the FEM velocity field
from the last completed benchmark npz (no Stokes re-solve)."""
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from random_porous import build_partition
import two_phase_transport as tp

src = np.load(PROJECT_DIR / "outputs" / "benchmark_twophase" / "twophase_fields.npz")
u_fem = src["u_FEM"]

partition = build_partition(
    mesh_size=0.13, sphere_size=0.065, boundary_size=0.085, interface_size=0.075,
    sphere_band=0.15, boundary_band=0.13, interface_band=0.12,
    mesh_file=PROJECT_DIR / "outputs" / "benchmark_twophase" / "random_sphere_partition.msh",
)

t0 = time.perf_counter()
result = tp.solve_two_phase(
    partition.mesh, u_fem, t_final=30.0, dt=0.1,
    snapshot_every=1,  # one snapshot per step -> 301 frames
)
print(f"FEM two-phase: {time.perf_counter() - t0:.1f} s, recovery={result['history']['recovery'][-1]:.4f}")

out_dir = PROJECT_DIR / "outputs" / "fem_snapshots"
out_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_dir / "twophase_fields.npz",
    points=src["points"], tetrahedra=src["tetrahedra"],
    cell_labels=src["cell_labels"],
    interface_pairs=src["interface_pairs"], interface_centers=src["interface_centers"],
    interface_normals=src["interface_normals"], interface_areas=src["interface_areas"],
    sphere_centers=src["sphere_centers"], sphere_radii=src["sphere_radii"],
    u_FEM=np.asarray(u_fem, dtype=float),
    s_FEM=result["final_saturation_vertices"],
    s_FEM_snapshots=result["snapshot_saturation_vertices"],
    snapshot_times=result["snapshot_times"],
)
print(f"wrote {out_dir / 'twophase_fields.npz'} with {len(result['snapshot_times'])} snapshots")

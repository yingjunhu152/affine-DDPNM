"""Two-dimensional DD-PNM reproduction for a fixed porous geometry."""

from .geometry import PARTICLES, build_partition
from .solver import solve_ddpnm, solve_reference
from .hierarchy import build_hierarchy_library, run_adaptive_hierarchy
from .exact_schur import solve_exact_fe_schur

__all__ = [
    "PARTICLES", "build_partition", "solve_ddpnm", "solve_reference",
    "build_hierarchy_library", "run_adaptive_hierarchy", "solve_exact_fe_schur",
]

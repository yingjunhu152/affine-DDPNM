"""2-D exact FE trace Schur complement — re-export shim.

The implementation has moved to :mod:`ddpnm_core.trace_schur`.  This module
exists for backward compatibility so that ``from ddpnm2d.exact_schur import
solve_exact_fe_schur`` continues to work.
"""

from ddpnm_core.trace_schur import (
    ExactSchurSolution,
    _wall_parent_dofs,
    _interface_parent_dofs,
    solve_exact_fe_schur,
)

# Re-export for backward-compatible access as ddpnm2d.exact_schur.*
__all__ = [
    "ExactSchurSolution",
    "_wall_parent_dofs",
    "_interface_parent_dofs",
    "solve_exact_fe_schur",
]

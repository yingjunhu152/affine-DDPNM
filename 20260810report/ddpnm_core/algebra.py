"""Algebraic utilities — re-export shim. See ``postprocess.metrics``."""

from postprocess.metrics import (
    HierarchyError,
    hierarchy_error,
    _dorfler_mark,
    _level_counts,
)

# Deprecated — use ddpnm_core.estimate.residual_indicators instead
def _interface_indicators(library, current, target):
    """Deprecated interface-vertex averaging estimator.

    Use :func:`ddpnm_core.estimate.residual_indicators` for the new
    residual-based estimator, which provides velocity jump, normal flux
    residual, tangential moment residual, and inactive-mode residual
    measurements.
    """
    import warnings
    import numpy as np
    warnings.warn(
        "_interface_indicators is deprecated. Use residual_indicators from "
        "ddpnm_core.estimate.",
        DeprecationWarning, stacklevel=2,
    )
    u, p = current
    ut, pt = target
    pressure_shift = float(np.mean(p - pt))
    du = u - ut
    dp = p - pressure_shift - pt
    unorm = max(float(np.linalg.norm(ut)), 1.0e-30)
    pnorm = max(float(np.linalg.norm(pt)), 1.0e-30)
    indicators = np.empty(len(library.interface_nodes), dtype=float)
    for interface_id, vertices in enumerate(library.interface_nodes):
        ids = np.asarray(vertices, dtype=np.int32)
        indicators[interface_id] = np.sqrt(
            np.linalg.norm(du[ids]) ** 2 / unorm**2
            + np.linalg.norm(dp[ids]) ** 2 / pnorm**2
        )
    return indicators

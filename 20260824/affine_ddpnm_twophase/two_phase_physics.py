"""Pure numerical helpers for the Buckley--Leverett transport solver.

This module deliberately has no FEniCSx dependency, which makes the Corey
closure, limiter, and scalar diagnostics testable in a plain Python runtime.
"""

from __future__ import annotations

import numpy as np


DEFAULT_COREY = {
    "swr": 0.2,
    "sor": 0.2,
    "nw": 2.0,
    "no": 2.0,
    "mu_w": 1.0,
    "mu_o": 5.0,
}


def validate_corey(swr: float, sor: float, nw: float, no: float,
                   mu_w: float, mu_o: float) -> None:
    if swr < 0.0 or sor < 0.0 or swr + sor >= 1.0:
        raise ValueError("Corey residual saturations require 0 <= Swr, Sor and Swr + Sor < 1")
    if nw < 1.0 or no < 1.0:
        raise ValueError("Corey exponents must be at least one")
    if mu_w <= 0.0 or mu_o <= 0.0:
        raise ValueError("Phase viscosities must be positive")


def effective_saturation(values, swr: float, sor: float) -> np.ndarray:
    if swr < 0.0 or sor < 0.0 or swr + sor >= 1.0:
        raise ValueError("invalid residual saturations")
    return np.clip(
        (np.asarray(values, dtype=float) - swr) / (1.0 - swr - sor),
        0.0,
        1.0,
    )


def phase_mobilities(values, swr: float, sor: float, nw: float, no: float,
                     mu_w: float, mu_o: float) -> tuple[np.ndarray, np.ndarray]:
    validate_corey(swr, sor, nw, no, mu_w, mu_o)
    se = effective_saturation(values, swr, sor)
    lam_w = np.power(se, nw) / mu_w
    lam_o = np.power(1.0 - se, no) / mu_o
    return lam_w, lam_o


def fractional_flow(values, swr: float = DEFAULT_COREY["swr"],
                    sor: float = DEFAULT_COREY["sor"],
                    nw: float = DEFAULT_COREY["nw"],
                    no: float = DEFAULT_COREY["no"],
                    mu_w: float = DEFAULT_COREY["mu_w"],
                    mu_o: float = DEFAULT_COREY["mu_o"]) -> np.ndarray:
    """Return the Corey water fractional flow evaluated pointwise."""
    lam_w, lam_o = phase_mobilities(values, swr, sor, nw, no, mu_w, mu_o)
    return lam_w / np.maximum(lam_w + lam_o, 1.0e-300)


def fractional_flow_derivative(values, swr: float = DEFAULT_COREY["swr"],
                               sor: float = DEFAULT_COREY["sor"],
                               nw: float = DEFAULT_COREY["nw"],
                               no: float = DEFAULT_COREY["no"],
                               mu_w: float = DEFAULT_COREY["mu_w"],
                               mu_o: float = DEFAULT_COREY["mu_o"]) -> np.ndarray:
    """Return ``dfw/dSw`` using the quotient rule.

    The derivative is zero outside the physical Corey interval because the
    effective saturation is clipped there.  Inside the interval,

        df/dSe = (lambda_w' lambda_o - lambda_w lambda_o') / lambda_t**2.
    """
    validate_corey(swr, sor, nw, no, mu_w, mu_o)
    values = np.asarray(values, dtype=float)
    se = effective_saturation(values, swr, sor)
    lam_w = np.power(se, nw) / mu_w
    lam_o = np.power(1.0 - se, no) / mu_o
    dlam_w = nw * np.power(se, nw - 1.0) / mu_w
    dlam_o = -no * np.power(1.0 - se, no - 1.0) / mu_o
    derivative = (
        (dlam_w * lam_o - lam_w * dlam_o)
        / np.maximum((lam_w + lam_o) ** 2, 1.0e-300)
        / (1.0 - swr - sor)
    )
    physical = (values >= swr) & (values <= 1.0 - sor)
    return np.where(physical, derivative, 0.0)


def conservative_bounded_limiter(
    values: np.ndarray,
    weights: np.ndarray,
    target_mass: float,
    lower: float,
    upper: float,
    fixed_dofs: np.ndarray | None = None,
    fixed_values: np.ndarray | None = None,
    tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Enforce bounds while preserving a feasible weighted mass exactly.

    Unlike the original limiter, this routine refuses an infeasible target
    instead of silently returning a bounded field with a different mass.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("values and weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("mass-lumping weights must be nonnegative")
    if not lower < upper:
        raise ValueError("limiter lower bound must be smaller than upper bound")

    out = np.clip(values, lower, upper)
    fixed = np.zeros(len(out), dtype=bool)
    if fixed_dofs is not None:
        fixed[np.asarray(fixed_dofs, dtype=np.int64)] = True
    if fixed_values is not None:
        fixed_values = np.asarray(fixed_values, dtype=float)
        if fixed_values.shape != values.shape:
            raise ValueError("fixed_values must have the same shape as values")
        out[fixed] = np.clip(fixed_values[fixed], lower, upper)

    adjustable = ~fixed
    fixed_mass = float(np.dot(weights[fixed], out[fixed]))
    min_mass = fixed_mass + float(np.sum(weights[adjustable])) * lower
    max_mass = fixed_mass + float(np.sum(weights[adjustable])) * upper
    scale = max(abs(target_mass), abs(min_mass), abs(max_mass), 1.0)
    tol_abs = tolerance * scale
    if target_mass < min_mass - tol_abs or target_mass > max_mass + tol_abs:
        raise RuntimeError(
            f"bounded limiter target mass {target_mass:.16e} is outside "
            f"[{min_mass:.16e}, {max_mass:.16e}]"
        )

    target = min(max(float(target_mass), min_mass), max_mass)
    clipped = out.copy()
    for _ in range(64):
        residual = float(target - np.dot(weights, out))
        if abs(residual) <= tol_abs:
            break
        if residual > 0.0:
            active = adjustable & (out < upper - 1.0e-14)
            capacity = weights[active] * (upper - out[active])
            total_capacity = float(np.sum(capacity))
            if total_capacity <= 1.0e-300:
                break
            theta = min(1.0, residual / total_capacity)
            out[active] += theta * (upper - out[active])
        else:
            active = adjustable & (out > lower + 1.0e-14)
            capacity = weights[active] * (out[active] - lower)
            total_capacity = float(np.sum(capacity))
            if total_capacity <= 1.0e-300:
                break
            theta = min(1.0, -residual / total_capacity)
            out[active] -= theta * (out[active] - lower)

    final_residual = float(target_mass - np.dot(weights, out))
    if abs(final_residual) > 10.0 * tol_abs:
        raise RuntimeError(
            f"bounded limiter failed to restore mass; residual={final_residual:.3e}"
        )
    return out, {
        "mass_residual": final_residual,
        "mass_change_abs": float(np.dot(weights, np.abs(out - clipped))),
        "target_mass": float(target_mass),
    }


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(values, weights) / denom)


def crossing_time(time_values: np.ndarray, signal: np.ndarray, level: float) -> float:
    time_values = np.asarray(time_values, dtype=float)
    signal = np.asarray(signal, dtype=float)
    above = np.flatnonzero(signal >= level)
    if len(above) == 0:
        return float("nan")
    idx = int(above[0])
    if idx == 0:
        return float(time_values[0])
    t0, t1 = float(time_values[idx - 1]), float(time_values[idx])
    y0, y1 = float(signal[idx - 1]), float(signal[idx])
    if abs(y1 - y0) <= 1.0e-14:
        return t1
    return float(t0 + (level - y0) * (t1 - t0) / (y1 - y0))


def signed_time_error(value: float, ref_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(ref_value):
        return float("nan")
    return float(value - ref_value)


def recovery_factor(water_mass: float, initial_water_mass: float,
                    initial_oil_mass: float) -> float:
    if initial_oil_mass <= 0.0:
        raise ValueError("initial oil mass must be positive")
    return float((water_mass - initial_water_mass) / initial_oil_mass)


def mass_balance_residual(new_mass: float, old_mass: float, dt: float,
                          net_boundary_outflow: float, inflow: float,
                          outflow: float) -> tuple[float, float]:
    """Return absolute and scaled residual of one conservative time step."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    absolute = float(new_mass - old_mass + dt * net_boundary_outflow)
    scale = max(
        abs(new_mass - old_mass),
        dt * (abs(inflow) + abs(outflow)),
        1.0e-14,
    )
    return absolute, float(abs(absolute) / scale)

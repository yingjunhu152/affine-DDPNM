from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from two_phase_physics import (  # noqa: E402
    conservative_bounded_limiter,
    crossing_time,
    fractional_flow,
    fractional_flow_derivative,
    mass_balance_residual,
    recovery_factor,
)


class CoreyTests(unittest.TestCase):
    def test_fractional_flow_is_monotone_and_bounded(self) -> None:
        saturation = np.linspace(0.0, 1.0, 1001)
        flow = fractional_flow(saturation)
        self.assertGreaterEqual(float(np.min(flow)), 0.0)
        self.assertLessEqual(float(np.max(flow)), 1.0)
        self.assertTrue(np.all(np.diff(flow) >= -1.0e-14))
        self.assertEqual(float(flow[0]), 0.0)
        self.assertEqual(float(flow[-1]), 1.0)

    def test_analytic_derivative_matches_centered_difference(self) -> None:
        saturation = np.linspace(0.22, 0.78, 31)
        epsilon = 1.0e-7
        finite_difference = (
            fractional_flow(saturation + epsilon)
            - fractional_flow(saturation - epsilon)
        ) / (2.0 * epsilon)
        analytic = fractional_flow_derivative(saturation)
        np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-6, atol=2.0e-8)
        self.assertTrue(np.all(analytic >= 0.0))


class LimiterTests(unittest.TestCase):
    def test_bounds_and_mass_are_preserved(self) -> None:
        values = np.asarray([-0.8, 0.25, 0.60, 1.7, 0.41])
        weights = np.asarray([0.2, 0.7, 0.4, 0.3, 0.5])
        target = float(np.dot(weights, values))
        limited, info = conservative_bounded_limiter(
            values, weights, target_mass=target, lower=0.2, upper=0.8
        )
        self.assertGreaterEqual(float(np.min(limited)), 0.2 - 1.0e-14)
        self.assertLessEqual(float(np.max(limited)), 0.8 + 1.0e-14)
        self.assertAlmostEqual(float(np.dot(weights, limited)), target, places=12)
        self.assertLess(abs(float(info["mass_residual"])), 1.0e-12)

    def test_infeasible_mass_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            conservative_bounded_limiter(
                np.asarray([0.2, 0.8]),
                np.ones(2),
                target_mass=2.0,
                lower=0.2,
                upper=0.8,
            )


class DiagnosticTests(unittest.TestCase):
    def test_crossing_time_interpolates(self) -> None:
        result = crossing_time(
            np.asarray([0.0, 1.0, 2.0]),
            np.asarray([0.0, 0.25, 0.75]),
            0.5,
        )
        self.assertAlmostEqual(result, 1.5)

    def test_recovery_starts_at_zero(self) -> None:
        self.assertEqual(recovery_factor(0.2, 0.2, 0.8), 0.0)
        self.assertAlmostEqual(recovery_factor(0.36, 0.2, 0.8), 0.2)

    def test_mass_ledger_uses_boundary_flux(self) -> None:
        old_mass = 0.20
        dt = 0.1
        inflow = 0.03
        outflow = 0.01
        net_outflow = outflow - inflow
        new_mass = old_mass - dt * net_outflow
        absolute, relative = mass_balance_residual(
            new_mass, old_mass, dt, net_outflow, inflow, outflow
        )
        self.assertAlmostEqual(absolute, 0.0, places=15)
        self.assertLess(relative, 1.0e-14)


if __name__ == "__main__":
    unittest.main()

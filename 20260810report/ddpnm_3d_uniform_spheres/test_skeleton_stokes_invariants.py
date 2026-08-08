#!/usr/bin/env python3
"""Algebraic invariants of the cardinal cross-skeleton extension."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from ddpnm3d.skeleton_stokes import (
    _constant_preserving_extension,
    _geodesic_farthest_points,
    _single_source_geodesic_distances,
    _uniform_target_count,
    _vector_affine_modes,
)


class UniformGeodesicSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph: dict[int, dict[int, float]] = {
            node: {} for node in range(25)
        }
        for row in range(5):
            for column in range(5):
                node = 5 * row + column
                for other_row, other_column in (
                    (row + 1, column),
                    (row, column + 1),
                ):
                    if other_row < 5 and other_column < 5:
                        other = 5 * other_row + other_column
                        self.graph[node][other] = 1.0
                        self.graph[other][node] = 1.0

    def test_sqrt_target_count(self) -> None:
        self.assertEqual(_uniform_target_count(25), 5)
        self.assertEqual(_uniform_target_count(26), 6)
        self.assertEqual(_uniform_target_count(4, sampling_factor=0.5), 1)
        self.assertEqual(_uniform_target_count(4, sampling_factor=4.0), 4)

    def test_sampling_is_exact_unique_seeded_and_deterministic(self) -> None:
        selected, covering, separation = _geodesic_farthest_points(
            self.graph, start=12, count=5
        )
        self.assertEqual(selected, (12, 0, 4, 20, 24))
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(covering, 2.0)
        self.assertEqual(separation, 4.0)

    def test_reported_covering_radius_is_the_true_graph_radius(self) -> None:
        selected, covering, _separation = _geodesic_farthest_points(
            self.graph, start=12, count=5
        )
        true_radius = max(
            min(
                _single_source_geodesic_distances(self.graph, sample)[node]
                for sample in selected
            )
            for node in self.graph
        )
        self.assertEqual(covering, true_radius)


class ConstantPreservingExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(20260803)
        factor = generator.normal(size=(17, 17))
        self.energy = factor.T @ factor + 0.4 * np.eye(17)
        self.selected = np.asarray([0, 3, 6, 10, 13, 16], dtype=np.int32)
        self.restriction = np.zeros((len(self.selected), 17), dtype=float)
        self.restriction[np.arange(len(self.selected)), self.selected] = 1.0
        self.extension, self.cardinal_residual, self.constant_residual = (
            _constant_preserving_extension(self.energy, self.restriction)
        )

    def test_exactly_one_unknown_per_cross_node(self) -> None:
        self.assertEqual(self.extension.shape, (17, len(self.selected)))

    def test_cross_coefficients_are_nodal_values(self) -> None:
        coefficients = np.asarray([0.2, -0.1, 0.8, 0.4, -0.3, 0.7])
        face_values = self.extension @ coefficients
        np.testing.assert_allclose(
            self.restriction @ face_values, coefficients, atol=2.0e-12
        )
        self.assertLess(self.cardinal_residual, 2.0e-12)

    def test_original_ddpnm_constant_is_reproduced(self) -> None:
        constant_pressure = 0.731
        face_values = self.extension @ (
            constant_pressure * np.ones(len(self.selected))
        )
        np.testing.assert_allclose(
            face_values,
            constant_pressure * np.ones(17),
            atol=2.0e-12,
        )
        self.assertLess(self.constant_residual, 2.0e-12)

    def test_extension_is_energy_minimal_under_admissible_variations(self) -> None:
        generator = np.random.default_rng(41)
        perturbation = generator.normal(size=self.extension.shape)
        perturbation[self.selected, :] = 0.0
        perturbation[:, -1] = -np.sum(perturbation[:, :-1], axis=1)
        self.assertLess(
            np.linalg.norm(self.restriction @ perturbation), 1.0e-13
        )
        self.assertLess(
            np.linalg.norm(perturbation @ np.ones(len(self.selected))), 1.0e-13
        )
        baseline = float(np.trace(self.extension.T @ self.energy @ self.extension))
        perturbed = self.extension + 1.0e-3 * perturbation
        perturbed_energy = float(np.trace(perturbed.T @ self.energy @ perturbed))
        self.assertGreaterEqual(perturbed_energy, baseline - 2.0e-11)

    def test_three_scalar_affine_modes_are_exactly_reproduced(self) -> None:
        first = np.linspace(-1.0, 1.0, self.energy.shape[0])
        full_modes = np.column_stack((np.ones(len(first)), first, first**2))
        selected_modes = full_modes[self.selected]
        self.assertEqual(np.linalg.matrix_rank(selected_modes), 3)
        extension, cardinal, affine = _constant_preserving_extension(
            self.energy,
            self.restriction,
            selected_modes,
            full_modes,
        )
        np.testing.assert_allclose(
            extension @ selected_modes, full_modes, atol=3.0e-12
        )
        np.testing.assert_allclose(
            self.restriction @ extension,
            np.eye(len(self.selected)),
            atol=3.0e-12,
        )
        self.assertLess(cardinal, 3.0e-12)
        self.assertLess(affine, 3.0e-12)
        generator = np.random.default_rng(77)
        perturbation = generator.normal(size=extension.shape)
        perturbation[self.selected, :] = 0.0
        projector = np.eye(len(self.selected)) - selected_modes @ np.linalg.pinv(
            selected_modes
        )
        perturbation = perturbation @ projector
        self.assertLess(
            np.linalg.norm(perturbation @ selected_modes), 2.0e-12
        )
        baseline = float(np.trace(extension.T @ self.energy @ extension))
        perturbed = extension + 1.0e-3 * perturbation
        self.assertGreaterEqual(
            float(np.trace(perturbed.T @ self.energy @ perturbed)),
            baseline - 3.0e-11,
        )


class VectorConstantPreservingExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(912)
        n_full_nodes = 9
        self.selected_nodes = np.asarray([0, 2, 5, 8], dtype=np.int32)
        size = 3 * n_full_nodes
        factor = generator.normal(size=(size, size))
        self.energy = factor.T @ factor + 0.7 * np.eye(size)
        selected = np.asarray(
            [3 * node + component for node in self.selected_nodes for component in range(3)],
            dtype=np.int32,
        )
        self.restriction = np.zeros((len(selected), size), dtype=float)
        self.restriction[np.arange(len(selected)), selected] = 1.0
        self.cross_constants = np.zeros((len(selected), 3), dtype=float)
        self.full_constants = np.zeros((size, 3), dtype=float)
        for node in range(len(self.selected_nodes)):
            for component in range(3):
                self.cross_constants[3 * node + component, component] = 1.0
        for node in range(n_full_nodes):
            for component in range(3):
                self.full_constants[3 * node + component, component] = 1.0
        self.extension, self.cardinal_residual, self.constant_residual = (
            _constant_preserving_extension(
                self.energy,
                self.restriction,
                self.cross_constants,
                self.full_constants,
            )
        )

    def test_vector_cross_coefficients_are_cardinal(self) -> None:
        coefficients = np.arange(self.extension.shape[1], dtype=float) / 7.0
        np.testing.assert_allclose(
            self.restriction @ (self.extension @ coefficients),
            coefficients,
            atol=3.0e-12,
        )
        self.assertLess(self.cardinal_residual, 3.0e-12)

    def test_three_constant_vector_modes_are_reproduced(self) -> None:
        amplitudes = np.asarray([0.4, -0.7, 1.2])
        cross_values = self.cross_constants @ amplitudes
        expected = self.full_constants @ amplitudes
        np.testing.assert_allclose(
            self.extension @ cross_values, expected, atol=3.0e-12
        )
        self.assertLess(self.constant_residual, 3.0e-12)

    def test_nine_vector_affine_modes_are_exactly_reproduced(self) -> None:
        coordinates = np.asarray(
            [(column, row) for row in (-1.0, 0.0, 1.0) for column in (-1.0, 0.0, 1.0)]
        )
        scalar_modes = np.column_stack(
            (np.ones(len(coordinates)), coordinates[:, 0], coordinates[:, 1])
        )
        selected_scalar_modes = scalar_modes[self.selected_nodes]
        self.assertEqual(np.linalg.matrix_rank(selected_scalar_modes), 3)
        full_modes = _vector_affine_modes(scalar_modes)
        selected_modes = _vector_affine_modes(selected_scalar_modes)
        extension, cardinal, affine = _constant_preserving_extension(
            self.energy,
            self.restriction,
            selected_modes,
            full_modes,
        )
        np.testing.assert_allclose(
            extension @ selected_modes, full_modes, atol=5.0e-12
        )
        np.testing.assert_allclose(
            self.restriction @ extension,
            np.eye(self.restriction.shape[0]),
            atol=5.0e-12,
        )
        self.assertLess(cardinal, 5.0e-12)
        self.assertLess(affine, 5.0e-12)


if __name__ == "__main__":
    unittest.main()

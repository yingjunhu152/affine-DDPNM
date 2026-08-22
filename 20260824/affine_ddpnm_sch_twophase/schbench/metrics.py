"""Mesh-weighted comparison metrics for the six experiment arms."""

from __future__ import annotations

import numpy as np


def scalar_l2_relative(a: np.ndarray, reference: np.ndarray, projector) -> float:
    delta = np.asarray(a) - np.asarray(reference)
    cells = projector.cell_vertices
    weights = projector.cell_volumes
    numerator = np.sum(weights * np.mean(delta[cells] ** 2, axis=1))
    denominator = np.sum(weights * np.mean(np.asarray(reference)[cells] ** 2, axis=1))
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))


def vector_l2_relative(a: np.ndarray, reference: np.ndarray, projector) -> float:
    delta2 = np.sum((np.asarray(a) - np.asarray(reference)) ** 2, axis=1)
    reference2 = np.sum(np.asarray(reference) ** 2, axis=1)
    cells = projector.cell_vertices
    weights = projector.cell_volumes
    numerator = np.sum(weights * np.mean(delta2[cells], axis=1))
    denominator = np.sum(weights * np.mean(reference2[cells], axis=1))
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))

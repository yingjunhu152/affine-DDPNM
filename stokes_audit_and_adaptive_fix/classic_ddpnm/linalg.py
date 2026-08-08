from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def to_scipy_matrix(matrix) -> csr_matrix:
    if hasattr(matrix, "to_scipy"):
        return matrix.to_scipy().tocsr()
    return csr_matrix(matrix.to_dense())


def to_numpy_vector(vector) -> np.ndarray:
    if hasattr(vector, "array"):
        return np.asarray(vector.array, dtype=float).copy()
    return np.asarray(vector, dtype=float).copy()

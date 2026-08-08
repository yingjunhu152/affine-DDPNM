"""Diagnostic: why is the watershed Schur system ill-posed?

Compares the spectra of the stored Schur matrices and flags outlier rows.
"""

from __future__ import annotations

import numpy as np

for label, path in [
    ("voronoi", "outputs/benchmark/random_benchmark_fields.npz"),
    ("watershed", "outputs/ablation_4way/watershed/random_benchmark_fields.npz"),
]:
    data = np.load(path)
    for method in ("classic", "affine"):
        schur = data[f"schur_matrix_{method}"]
        ev = np.linalg.eigvalsh(schur)
        print(
            f"{label} {method}: shape={schur.shape}, "
            f"eig min={ev.min():.3e}, max={ev.max():.3e}, "
            f"|S|max={np.abs(schur).max():.3e}"
        )
        row_max = np.abs(schur).max(axis=1)
        i = int(np.argmax(row_max))
        print(f"   worst row {i}: max={row_max[i]:.3e}, diag={abs(schur[i, i]):.3e}")
        pairs = data["interface_pairs"]
        print(f"   interface_pairs[{i}] = {list(pairs[i])}")
        # Rows whose diagonal is not the dominant entry (suspicious coupling)
        diag_dominant = np.abs(np.diag(schur)) >= 0.5 * row_max
        print(f"   rows without diag dominance: {int(np.sum(~diag_dominant))}")

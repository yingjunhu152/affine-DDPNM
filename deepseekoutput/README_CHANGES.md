# Revised figure scripts

This package contains revised versions of `figlib.py` and `make_final_figures.py`.
The figure-generation pipeline itself was **not run** here; only a syntax compilation check was performed.

Key changes:

1. Corrected slice velocity-error field from `abs(|u_DD|-|u_FEM|)` to the mathematically correct pointwise vector error `||u_DD-u_FEM||`.
2. Fixed the field-panel GridSpec bug that caused the reference colorbar to overlap the W0n panel.
3. Replaced the rainbow/turbo error map with a sequential `magma` map and uses one full-range shared error scale for W0n/W1n/W1v.
4. Added a paper-ready 2x4 Uniform-27 vs Random-27 field/error comparison: `fig5_fields_uniform_random.png`.
5. Added numeric `(s,t)` ticks to representative-interface flux plots.
6. Added `fig5_interface_flux_errors.png` showing `|q-q_FEM|` for W0n/W1n/W1v on the same representative interfaces.
7. Cleaned modal-budget figure; W0v/W1n receive a small display jitter only when both r=3 data sets exist.
8. Centered method offsets in the four-metric figure and made the legend dynamically include W0v when it becomes available.
9. Cleaned the observed-share figure: legend moved above, redundant x-label removed.
10. Accuracy-cost figure now explicitly states the archived FEM timing scope per geometry and labels the x-axis as recorded wall time.
11. Reworked the geometry/network figure so the lower row uses one consistent pore-adjacency/interface-graph convention; the Random-27 colored tetrahedral-mesh rendering was removed.
12. Existing SKIP behavior for missing W0v, full C-DD, mesh ensemble, random ensemble, and Real-100 slice data is preserved.

Current archived timing scope encoded in the script:
- Uniform-27: FEM assembly + solve
- Random-27: FEM assembly + solve
- Real-100: FEM solve only

Update `FEM_TIMING_SCOPE` after you rerun standardized timings.

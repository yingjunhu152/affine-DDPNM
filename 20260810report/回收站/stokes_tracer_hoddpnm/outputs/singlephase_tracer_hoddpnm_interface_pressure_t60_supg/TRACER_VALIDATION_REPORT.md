# Single-Phase Stokes-Tracer HODDPNM Validation

This run removes adaptive method levels and compares FEM-Stokes tracer against a fixed HODDPNM Schur/static-condensation solve of the same FEniCSx Taylor-Hood matrix.

HODDPNM is the Schur-complement/static-condensation path: known Dirichlet dofs are removed first, interior velocity/interior pressure unknowns are eliminated, the free interface velocity/interface-pressure Schur problem is solved, and the full field is reconstructed.

The current implementation demonstrates clear dof compression and acceleration using Schur diagonal scaling plus modest pressure stabilization. The Schur iteration count is still O(10^2), so a scalable Schur preconditioner remains the next optimization target.

## Stokes Compression

| method | Stokes time (s) | active dofs | active ratio | Krylov iterations | Schur matvecs | Schur residual | known fixed dofs | eliminated free interior dofs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FEM | 7.810 | 42532 | 100.0% |  |  |  |  |  |
| HODDPNM | 2.944 | 11521 | 27.1% | 251 | 255 | 5.959e-11 | 21940 | 9071 |

## Main Metrics

| method | tracer time (s) | breakthrough rel L2 | concentration rel L2 | final mass rel err | max balance residual | final range | raw limiter hits | t90 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| FEM | 0.270 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 3.571e-14 | [0.149, 1.000] | 151 | 19.965 |
| HODDPNM | 0.193 | 2.722e-13 | 4.994e-13 | 2.149e-13 | 4.589e-14 | [0.149, 1.000] | 151 | 19.965 |

## Files

- Breakthrough curves: `outputs\singlephase_tracer_hoddpnm_interface_pressure_t60_supg\breakthrough_curves.png`
- Mass balance validation: `outputs\singlephase_tracer_hoddpnm_interface_pressure_t60_supg\mass_balance_validation.png`
- Error summary: `outputs\singlephase_tracer_hoddpnm_interface_pressure_t60_supg\tracer_error_summary.png`
- Final concentration and error: `outputs\singlephase_tracer_hoddpnm_interface_pressure_t60_supg\final_concentration_and_error.png`
- CSV metrics: `outputs\singlephase_tracer_hoddpnm_interface_pressure_t60_supg\tracer_metrics.csv`
- Mass history CSV: `outputs\singlephase_tracer_hoddpnm_interface_pressure_t60_supg\mass_balance_history.csv`

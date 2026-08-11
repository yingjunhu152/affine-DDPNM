# Affine-DDPNM Driven Two-Phase Validation (random-27 medium)

The same random-27 partition mesh is solved by the monolithic Taylor-Hood FEM and by three DDPNM interface-traction spaces (Classic-1, W1n-3, Affine-9).  Each Stokes total velocity field then drives the identical Buckley--Leverett water--oil saturation transport (Corey fractional flow), and every two-phase metric is reported against the FEM-driven reference.

## Stokes Compression and Velocity Error

| method | Stokes time (s) | global unknowns | modes/face | velocity rel L2 | velocity broken H1 | pressure aligned rel L2 | outlet flux rel err |
|---|---:|---:|---:|---:|---:|---:|---:|
| FEM | 116.033 | 75954 | monolithic | 0.000% | 0.000% | 0.000% | 0.000e+00 |
| Classic-DDPNM-1 | 7.864 | 114 | 1 | 65.320% | 85.852% | 21.957% | 7.272e-01 |
| NormalLinear-DDPNM-3 | 9.243 | 342 | 3 | 30.749% | 38.965% | 6.944% | 1.465e-01 |
| Affine-DDPNM-9 | 12.643 | 1026 | 9 | 6.506% | 17.556% | 4.360% | 2.706e-02 |

## Two-Phase Metrics vs FEM-Driven Reference

| method | two-phase time (s) | saturation rel L2 | water cut rel L2 | recovery rel L2 | final recovery | final recovery err | final range | raw limiter hits | watercut t50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FEM | 54.544 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.0289 | 0.000e+00 | [0.200, 0.800] | 221 | nan |
| Classic-DDPNM-1 | 54.816 | 1.920e-01 | 6.631e+01 | 7.321e-01 | 0.0500 | 2.112e-02 | [0.200, 0.800] | 250 | nan |
| NormalLinear-DDPNM-3 | 55.189 | 5.967e-02 | 1.653e+00 | 1.463e-01 | 0.0331 | 4.221e-03 | [0.200, 0.800] | 213 | nan |
| Affine-DDPNM-9 | 54.110 | 1.181e-02 | 1.288e-01 | 2.757e-02 | 0.0296 | 7.955e-04 | [0.200, 0.800] | 213 | nan |

## Conservation and Nonlinear-Solve Audit

| method | max step balance abs | max cumulative budget abs | limiter mass abs | nonconverged Picard steps | final PVI |
|---|---:|---:|---:|---:|---:|
| FEM | 8.619e-17 | 8.327e-16 | 8.327e-17 | 7 | 0.023 |
| Classic-DDPNM-1 | 1.269e-16 | 5.551e-16 | 5.551e-17 | 20 | 0.040 |
| NormalLinear-DDPNM-3 | 8.554e-17 | 1.943e-16 | 5.551e-17 | 29 | 0.026 |
| Affine-DDPNM-9 | 1.134e-16 | 7.216e-16 | 8.327e-17 | 21 | 0.024 |

## Files

- Water cut curves: `outputs\smoke_twophase2\watercut_curves.png`
- Recovery curves: `outputs\smoke_twophase2\recovery_curves.png`
- Mass-balance validation: `outputs\smoke_twophase2\mass_balance_validation.png`
- Error summary: `outputs\smoke_twophase2\twophase_error_summary.png`
- Final saturation and error: `outputs\smoke_twophase2\final_saturation_and_error.png`
- CSV metrics: `outputs\smoke_twophase2\twophase_metrics.csv`
- Saturation history CSV: `outputs\smoke_twophase2\twophase_history.csv`

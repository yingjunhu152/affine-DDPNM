# Affine-DDPNM Driven Two-Phase Validation (random-27 medium)

The same random-27 partition mesh is solved by the monolithic Taylor-Hood FEM and by three DDPNM interface-traction spaces (Classic-1, W1n-3, Affine-9).  Each Stokes total velocity field then drives the identical Buckley--Leverett water--oil saturation transport (Corey fractional flow), and every two-phase metric is reported against the FEM-driven reference.

## Stokes Compression and Velocity Error

| method | Stokes time (s) | global unknowns | modes/face | velocity rel L2 | velocity broken H1 | pressure aligned rel L2 | outlet flux rel err |
|---|---:|---:|---:|---:|---:|---:|---:|
| FEM | 401.955 | 75954 | monolithic | 0.000% | 0.000% | 0.000% | 0.000e+00 |
| Classic-DDPNM-1 | 27.517 | 114 | 1 | 65.320% | 85.852% | 21.957% | 7.272e-01 |
| NormalLinear-DDPNM-3 | 32.031 | 342 | 3 | 30.749% | 38.965% | 6.944% | 1.465e-01 |
| Affine-DDPNM-9 | 42.548 | 1026 | 9 | 6.506% | 17.556% | 4.360% | 2.706e-02 |

## Two-Phase Metrics vs FEM-Driven Reference

| method | two-phase time (s) | saturation rel L2 | water cut rel L2 | recovery rel L2 | final recovery | final recovery err | final range | raw limiter hits | watercut t50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FEM | 8.279 | 0.000e+00 | 0.000e+00 | 0.000e+00 | -0.0024 | 0.000e+00 | [0.000, 1.000] | 582 | nan |
| Classic-DDPNM-1 | 8.243 | 2.204e-01 | 7.826e+00 | 1.242e-01 | -0.0009 | 1.490e-03 | [0.000, 1.000] | 556 | nan |
| NormalLinear-DDPNM-3 | 8.521 | 1.765e-01 | 9.994e-01 | 8.462e-02 | -0.0014 | 1.015e-03 | [0.000, 1.000] | 575 | nan |
| Affine-DDPNM-9 | 3.687 | 1.645e-01 | 1.000e+00 | 1.512e-02 | -0.0022 | 1.814e-04 | [0.000, 1.000] | 556 | nan |

## Files

- Water cut curves: `outputs\smoke_twophase\watercut_curves.png`
- Recovery curves: `outputs\smoke_twophase\recovery_curves.png`
- Error summary: `outputs\smoke_twophase\twophase_error_summary.png`
- Final saturation and error: `outputs\smoke_twophase\final_saturation_and_error.png`
- CSV metrics: `outputs\smoke_twophase\twophase_metrics.csv`
- Saturation history CSV: `outputs\smoke_twophase\twophase_history.csv`

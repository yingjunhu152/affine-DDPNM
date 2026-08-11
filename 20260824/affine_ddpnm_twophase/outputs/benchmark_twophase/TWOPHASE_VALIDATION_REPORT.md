# Affine-DDPNM Driven Two-Phase Validation (random-27 medium)

The same random-27 partition mesh is solved by the monolithic Taylor-Hood FEM and by three DDPNM interface-traction spaces (Classic-1, W1n-3, Affine-9).  Each Stokes total velocity field then drives the identical Buckley--Leverett water--oil saturation transport (Corey fractional flow), and every two-phase metric is reported against the FEM-driven reference.

## Stokes Compression and Velocity Error

| method | Stokes time (s) | global unknowns | modes/face | velocity rel L2 | velocity broken H1 | pressure aligned rel L2 | outlet flux rel err |
|---|---:|---:|---:|---:|---:|---:|---:|
| FEM | 340.904 | 75954 | monolithic | 0.000% | 0.000% | 0.000% | 0.000e+00 |
| Classic-DDPNM-1 | 23.651 | 114 | 1 | 65.320% | 85.852% | 21.957% | 7.272e-01 |
| NormalLinear-DDPNM-3 | 27.137 | 342 | 3 | 30.749% | 38.965% | 6.944% | 1.465e-01 |
| Affine-DDPNM-9 | 12.609 | 1026 | 9 | 6.506% | 17.556% | 4.360% | 2.706e-02 |

## Two-Phase Metrics vs FEM-Driven Reference

| method | two-phase time (s) | saturation rel L2 | water cut rel L2 | recovery rel L2 | final recovery | final recovery err | final range | raw limiter hits | watercut t50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FEM | 1455.326 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.2791 | 0.000e+00 | [0.201, 0.800] | 452 | nan |
| Classic-DDPNM-1 | 1449.445 | 3.954e-01 | 2.269e+00 | 6.157e-01 | 0.4279 | 1.488e-01 | [0.204, 0.800] | 807 | nan |
| NormalLinear-DDPNM-3 | 1055.928 | 1.417e-01 | 9.622e-01 | 1.271e-01 | 0.3071 | 2.802e-02 | [0.201, 0.800] | 482 | nan |
| Affine-DDPNM-9 | 1283.083 | 7.511e-02 | 1.520e-01 | 2.526e-02 | 0.2851 | 6.003e-03 | [0.201, 0.800] | 456 | nan |

## Conservation and Nonlinear-Solve Audit

| method | max step balance abs | max cumulative budget abs | limiter mass abs | nonconverged Picard steps | final PVI |
|---|---:|---:|---:|---:|---:|
| FEM | 2.894e-16 | 1.832e-15 | 1.665e-16 | 35 | 0.231 |
| Classic-DDPNM-1 | 2.459e-16 | 2.665e-15 | 1.665e-16 | 249 | 0.400 |
| NormalLinear-DDPNM-3 | 2.182e-16 | 2.054e-15 | 1.665e-16 | 154 | 0.265 |
| Affine-DDPNM-9 | 2.220e-16 | 2.276e-15 | 1.110e-16 | 80 | 0.237 |

## Files

- Water cut curves: `outputs\benchmark_twophase\watercut_curves.png`
- Recovery curves: `outputs\benchmark_twophase\recovery_curves.png`
- Mass-balance validation: `outputs\benchmark_twophase\mass_balance_validation.png`
- Error summary: `outputs\benchmark_twophase\twophase_error_summary.png`
- Final saturation and error: `outputs\benchmark_twophase\final_saturation_and_error.png`
- CSV metrics: `outputs\benchmark_twophase\twophase_metrics.csv`
- Saturation history CSV: `outputs\benchmark_twophase\twophase_history.csv`

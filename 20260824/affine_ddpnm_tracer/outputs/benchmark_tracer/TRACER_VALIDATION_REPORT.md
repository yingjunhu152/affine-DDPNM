# Affine-DDPNM Driven Tracer Validation (random-27 medium)

The same random-27 partition mesh is solved by the monolithic Taylor-Hood FEM and by three DDPNM interface-traction spaces (Classic-1, W1n-3, Affine-9).  Each Stokes velocity field then drives the identical transient tracer advection-diffusion model, and every tracer metric is reported against the FEM-driven reference.

## Stokes Compression and Velocity Error

| method | Stokes time (s) | global unknowns | modes/face | velocity rel L2 | velocity broken H1 | pressure aligned rel L2 | outlet flux rel err |
|---|---:|---:|---:|---:|---:|---:|---:|
| FEM | 106.210 | 75954 | monolithic | 0.000% | 0.000% | 0.000% | 0.000e+00 |
| Classic-DDPNM-1 | 7.757 | 114 | 1 | 65.320% | 85.852% | 21.957% | 7.272e-01 |
| NormalLinear-DDPNM-3 | 8.801 | 342 | 3 | 30.749% | 38.965% | 6.944% | 1.465e-01 |
| Affine-DDPNM-9 | 12.367 | 1026 | 9 | 6.506% | 17.556% | 4.360% | 2.706e-02 |

## Tracer Metrics vs FEM-Driven Reference

| method | tracer time (s) | breakthrough rel L2 | concentration rel L2 | final mass rel err | max balance residual | final range | raw limiter hits | t90 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| FEM | 0.425 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 4.117e-13 | [0.974, 1.000] | 0 | 19.629 |
| Classic-DDPNM-1 | 0.385 | 6.332e-03 | 1.302e-03 | 1.182e-03 | 3.966e-13 | [0.976, 1.000] | 0 | 19.294 |
| NormalLinear-DDPNM-3 | 0.377 | 2.978e-03 | 6.024e-04 | 5.465e-04 | 2.740e-13 | [0.975, 1.000] | 0 | 19.473 |
| Affine-DDPNM-9 | 0.377 | 5.065e-04 | 1.026e-04 | 9.326e-05 | 3.885e-13 | [0.974, 1.000] | 0 | 19.603 |

## Files

- Breakthrough curves: `D:\hu\tongjiproj\20260727\20260824\affine_ddpnm_tracer\outputs\benchmark_tracer\breakthrough_curves.png`
- Mass balance validation: `D:\hu\tongjiproj\20260727\20260824\affine_ddpnm_tracer\outputs\benchmark_tracer\mass_balance_validation.png`
- Error summary: `D:\hu\tongjiproj\20260727\20260824\affine_ddpnm_tracer\outputs\benchmark_tracer\tracer_error_summary.png`
- Final concentration and error: `D:\hu\tongjiproj\20260727\20260824\affine_ddpnm_tracer\outputs\benchmark_tracer\final_concentration_and_error.png`
- CSV metrics: `D:\hu\tongjiproj\20260727\20260824\affine_ddpnm_tracer\outputs\benchmark_tracer\tracer_metrics.csv`
- Mass history CSV: `D:\hu\tongjiproj\20260727\20260824\affine_ddpnm_tracer\outputs\benchmark_tracer\mass_balance_history.csv`

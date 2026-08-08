# Tolerance-Driven Adaptive Stokes HODDPNM Report

## Algorithm

The adaptive logic was changed from a fixed-cycle promotion rule to a tolerance-driven rule.

Initial state:

- all regions start as `PNM`;
- the equation is still the FEniCSx Taylor-Hood P2-P1 Stokes matrix;
- the linear solver is restricted Schur-GMRES with ILU approximate Schur preconditioner.

Stopping rule:

- compute the global target error after each solve;
- default target error is velocity relative L2 error;
- stop when `target_error <= error_tolerance`.

Region ranking:

- compute a region indicator `eta_r`;
- in the validation run, `eta_r` uses the true local velocity/pressure error against the FEM reference;
- for a production algorithm this should be replaced by a residual or posterior estimator.

Upgrade rule:

- sort candidate regions by `eta_r^2` from large to small;
- use Doerfler marking with `theta = 0.65`;
- select regions until the selected cumulative indicator energy reaches `65%` of the candidate total;
- at most `3` regions are upgraded per cycle;
- each selected region is promoted by one level only:
  `PNM -> DDPNM -> DDPNMT -> HODDPNM`;
- promotions are monotone; a refined region is never downgraded;
- active DOFs are capped by `active_dof_cap` when requested.

This answers the design choices:

- once per cycle, upgrade at most `3` regions;
- region order is descending local indicator energy;
- upgrade is one fidelity level at a time, not a direct jump to HODDPNM;
- stop as soon as the error bound is reached.

## Code and run script

Updated solver:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\stokes_adaptive_taylor_hood.py`

Reproducible run script:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\run_tolerance_driven_adaptive_berea.ps1`

## Result: tolerance 1e-5

Output:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\outputs\adaptive_stokes_berea_16_r16_tol1e5_trueerr`

Final state:

| item | value |
| --- | ---: |
| Error tolerance | 1.000e-05 |
| Final velocity relative L2 error | 5.535e-06 |
| Final pressure relative L2 error | 1.209e-13 |
| Converged to tolerance | true |
| Active DOFs | 42241 / 42532 |
| Active DOF ratio | 99.3158% |
| Final adaptive Schur-GMRES time | 1.594 s |
| Schur-GMRES iterations | 42 |
| Final regions | PNM=3, DDPNM=9, DDPNMT=4, HODDPNM=0 |

Cycle history:

| cycle | active ratio | velocity error | regions |
| ---: | ---: | ---: | --- |
| 0 | 62.4% | 1.547e+01 | PNM=16 |
| 1 | 66.2% | 3.142e+00 | PNM=14, DDPNM=2 |
| 2 | 67.6% | 2.980e+00 | PNM=13, DDPNM=3 |
| 3 | 68.9% | 2.432e+00 | PNM=12, DDPNM=4 |
| 4 | 69.9% | 2.202e+00 | PNM=11, DDPNM=5 |
| 5 | 77.7% | 1.001e+00 | PNM=8, DDPNM=8 |
| 6 | 84.4% | 7.319e-01 | PNM=7, DDPNM=8, DDPNMT=1 |
| 7 | 90.6% | 1.662e-01 | PNM=5, DDPNM=9, DDPNMT=2 |
| 8 | 91.3% | 1.617e-01 | PNM=4, DDPNM=10, DDPNMT=2 |
| 9 | 99.3% | 5.535e-06 | PNM=3, DDPNM=9, DDPNMT=4 |

Main figure:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\outputs\adaptive_stokes_berea_16_r16_tol1e5_trueerr\paper_velocity_pressure_error_isosurfaces\berea_velocity_pressure_log_error_isosurfaces.png`

## Result: tolerance 1e-6

Output:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\outputs\adaptive_stokes_berea_16_r16_tol1e6_trueerr`

Final state:

| item | value |
| --- | ---: |
| Error tolerance | 1.000e-06 |
| Final velocity relative L2 error | 8.269e-13 |
| Final pressure relative L2 error | 1.788e-15 |
| Converged to tolerance | true |
| Active DOFs | 42532 / 42532 |
| Active DOF ratio | 100% |

Interpretation:

With the current subspace definition, `1e-6` is strict enough that the method promotes until the active space becomes the full Taylor-Hood system. Therefore `1e-5` is currently the more useful non-full-space validation tolerance for this Berea crop.

## Caveat

This is a validation-oracle adaptive run because `--indicator true-error` uses the FEM reference to rank local errors. This is acceptable for diagnosing and validating the adaptive mechanism, but it is not yet the final standalone algorithm. For publication, the next step should replace `true-error` with a computable residual or posterior estimator.

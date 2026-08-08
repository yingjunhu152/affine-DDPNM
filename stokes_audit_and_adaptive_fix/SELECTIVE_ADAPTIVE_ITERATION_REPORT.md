# Selective Adaptive HODDPNM Iteration Report

## Purpose

The goal of this iteration was to prevent the adaptive Stokes solver from promoting all regions to the full Taylor-Hood P2-P1 space. HODDPNM should act only on high-error regions, with the rest of the domain left at lower fidelity.

The implementation was updated in:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\stokes_adaptive_taylor_hood.py`

## Implemented changes

- Added `--adaptive-strategy capped-high-error`.
- Added `--adaptive-strategy capped-monotone-high-error`.
- Added active DOF cap through `--active-dof-cap`.
- Added region selection by `--indicator residual` and validation-oracle `--indicator true-error`.
- Added monotone promotion so selected high-error regions are not demoted in later cycles.
- Kept the linear solve as FEniCSx Taylor-Hood Stokes restricted Schur-GMRES with ILU approximate Schur preconditioner.

## Iteration results

| case | indicator | active DOFs | active ratio | time (s) | velocity rel L2 | pressure rel L2 | regions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| full preconditioned adaptive baseline | residual/cumulative | 42532 | 100.0% | 4.497 | 8.269e-13 | 1.788e-15 | PNM=0, DDPNM=9, DDPNMT=0, HODDPNM=7 |
| capped high-error, cap 80% | residual | 33889 | 79.7% | 2.238 | 2.730e+00 | 3.712e-01 | PNM=11, DDPNM=4, DDPNMT=0, HODDPNM=1 |
| capped high-error, cap 90% | residual | 38080 | 89.5% | 3.363 | 1.133e+00 | 2.721e-01 | PNM=7, DDPNM=8, DDPNMT=0, HODDPNM=1 |
| capped high-error, cap 97% | residual | 40948 | 96.3% | 4.921 | 1.156e+00 | 4.241e-02 | PNM=3, DDPNM=10, DDPNMT=0, HODDPNM=3 |
| capped high-error, true-error, cap 90% | true-error | 38065 | 89.5% | 3.331 | 8.902e-01 | 2.721e-01 | PNM=7, DDPNM=8, DDPNMT=0, HODDPNM=1 |
| capped high-error, true-error, cap 99.5%, non-monotone | true-error | 41971 | 98.7% | 4.932 | 4.475e-01 | 7.098e-08 | PNM=1, DDPNM=12, DDPNMT=0, HODDPNM=3 |
| capped monotone high-error, true-error, cap 99.5% | true-error | 42241 | 99.3% | 5.441 | 5.535e-06 | 1.209e-13 | PNM=1, DDPNM=12, DDPNMT=0, HODDPNM=3 |

## Main finding

The monotone high-error strategy fixes the oscillation. With `cap=99.5%`, it keeps HODDPNM on only three high-error regions and reaches a velocity relative L2 error of `5.535e-06`.

However, this is not yet a strong adaptive result: it still needs `99.3%` of the mixed DOFs and is slower than the full preconditioned HODDPNM baseline on this small Berea case.

## Diagnosis

The current selective implementation removes unselected high-order velocity DOFs from the approximation space by setting them inactive. For this Berea Stokes problem, those P2 velocity DOFs are globally important. Leaving even a few regions at the current PNM level can create large velocity errors.

Therefore, the next algorithmic step should not be more parameter tuning. It should replace inactive high-order DOF truncation with one of:

- local static condensation for unselected regions;
- local reduced Stokes basis functions for PNM/DDPNM regions;
- residual correction solves on selected patches;
- a true multilevel Schur formulation where lower-fidelity regions still contribute an approximate condensed response instead of zeroing their P2 modes.

## Key output

Best selective monotone run:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\outputs\adaptive_stokes_berea_16_r16_selective_monotone_trueerr_cap995`

Main error figure:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\outputs\adaptive_stokes_berea_16_r16_selective_monotone_trueerr_cap995\paper_velocity_pressure_error_isosurfaces\berea_velocity_pressure_log_error_isosurfaces.png`

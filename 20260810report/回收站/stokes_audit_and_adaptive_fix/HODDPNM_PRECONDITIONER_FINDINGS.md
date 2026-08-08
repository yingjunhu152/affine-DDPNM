# HODDPNM Preconditioner Findings

## Existing preconditioned implementations found

1. `D:\hu\tongjiproj\FENICSX\fenicsx_real_porous_hoddpnm_berea_comparison\real_porous_hoddpnm_validation.py`

   This is the directly reusable real-Berea implementation. It solves the FEniCSx Taylor-Hood P2-P1 Stokes system with HODDPNM Schur complement, using matrix-free GMRES on the Schur operator. The implemented preconditioner is `ilu_of_diagonal_approximate_schur`.

2. `D:\hu\tongjiproj\FENICSX\fenicsx_random_sphere_holes_pnm_stokes\run_random_pnm_taylor_hood_preconditioned_schur.py`

   This is the more experimental preconditioner testbed. It contains multiple Schur preconditioners, including local DtN, patch Schur, edge-patch Schur, two-level graph, and PNM saddle/coarse variants.

## Preconditioner attached to adaptive Stokes HODDPNM

Updated file:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\stokes_adaptive_taylor_hood.py`

The adaptive solver now keeps the same FEniCSx-assembled Taylor-Hood P2-P1 Stokes matrix and changes only the restricted linear solve:

- before: restricted sparse direct solve;
- now: restricted matrix-free Schur GMRES;
- preconditioner: ILU of diagonal approximate Schur complement;
- default CLI: `--restricted-solver schur-gmres --schur-preconditioner ilu`;
- ILU parameters: `drop_tol=1e-4`, `fill_factor=12`.

## Verified Berea adaptive run

Output folder:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\outputs\adaptive_stokes_berea_16_r16_preconditioned`

Main results:

| item | value |
| --- | ---: |
| Mixed DOFs | 42532 |
| FEM sparse direct reference time | 26.040091 s |
| Final adaptive HODDPNM Schur-GMRES time | 4.497073 s |
| Final Schur GMRES iterations | 43 |
| Final Schur relative residual | 3.401643e-12 |
| Final velocity relative L2 error | 8.268750e-13 |
| Final pressure relative L2 error | 1.787850e-15 |
| Final region counts | PNM=0, DDPNM=9, DDPNMT=0, HODDPNM=7 |

Main figure:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\outputs\adaptive_stokes_berea_16_r16_preconditioned\paper_velocity_pressure_error_isosurfaces\berea_velocity_pressure_log_error_isosurfaces.png`

Summary file:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\outputs\adaptive_stokes_berea_16_r16_preconditioned\validation_summary.json`

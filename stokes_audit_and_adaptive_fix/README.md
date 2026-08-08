# Stokes Audit and Adaptive Fix

This directory is the corrected Stokes validation line for the Berea adaptive
PNM/DDPNM/HODDPNM experiments.

The governing rule is simple:

- Acceptable Stokes results must solve the FEniCSx Taylor-Hood P2-P1 Stokes
  system, or a restricted/substructured solve of the same assembled matrix.
- Graph pressure, tracer, and two-phase graph prototypes are not Stokes
  validation results.
- Oracle experiments that use FEM true error are diagnostic only.
- Computable adaptive experiments must be judged by validation against FEM,
  not by their internal indicator alone.

## Start Here

Read these files first:

1. `STOKES_AUDIT_REPORT.md` - audit rule and corrected Stokes scope.
2. `EXPERIMENT_TAXONOMY.md` - classification of each experiment family.
3. `REPORT_INDEX.md` - reading order for the top-level reports.

Core code:

- `stokes_adaptive_taylor_hood.py` - adaptive restricted Stokes driver.
- `real_porous_hoddpnm_validation.py` - Berea mesh, Taylor-Hood assembly,
  direct FEM reference, and known-fixed HODDPNM Schur solve utilities.
- `render_final_state_figures.py` - the single final-state plotting program.

Canonical input:

- `data/berea_100_to_300.npz`
- crop: `20:36,150:166,30:46`
- regions: `16`

## Current Disk State

This checkout is intentionally cleaned. The only output folder currently kept
under `outputs/` is:

- `outputs/final_state_figures/`

The historical result folders named in the reports are not present on disk in
this checkout. Treat those paths as recorded case names unless they are
regenerated locally.

## Recorded Mainline

Use these as historical narrative anchors recorded by the reports:

| role | case/report | status |
| --- | --- | --- |
| Correctness baseline | `outputs/berea_16_r16_pure_hoddpnm100` | full HODDPNM100, machine-precision agreement with FEM |
| Corrected Schur-GMRES baseline | `outputs/adaptive_stokes_berea_16_r16_preconditioned` | equation-correct, validates preconditioned restricted Stokes machinery |
| Oracle adaptive diagnostic | `outputs/adaptive_stokes_berea_16_r16_tol1e5_trueerr` | reaches `1e-5` velocity error, but uses FEM true error for marking |
| Best computable posterior attempt | `outputs/adaptive_stokes_berea_16_r16_posterior_defect_cap995` | computable indicator; validation velocity error remains about `1.95e-2` |

## Final Output Contract

New runs should show only final-state visual evidence by default. Iteration
cycles are internal diagnostics and are not written unless
`--save-iteration-artifacts` is passed.

Default final figures are written under `outputs/final_state_figures/`:

- `final_region_methods.png` - final per-region method/stage assignment.
- `final_velocity_magnitude.png` - final velocity magnitude field.
- `final_pressure.png` - final pressure field.
- `final_velocity_log_error.png` - final velocity error field.
- `final_pressure_log_error.png` - final pressure error field.
- `final_state_overview.png` - combined final-state visual summary.

The figure style follows
`D:\hu\tongjiproj\FENICSX\fenicsx_irregular_hoddpnm\outputs\cube_holes_27_taylor_hood_random_style_errors\cells_9_pressure_error_isosurface.png`:
white background, parallel 3D camera, translucent grey pore/solid geometry,
turbo log-error isosurfaces, compact upper-left titles, and a thin right-side
PyVista scalar bar.

To generate velocity and pressure error fields, the run must include a FEM
reference solve, usually `--reference-solve after` for non-oracle indicators or
`--reference-solve before` when using `--indicator true-error`.

## What Not To Claim

Do not present these as final standalone adaptive algorithms:

- `true-error` indicator cases: they need an FEM reference to choose upgrades.
- geometry-prior and spectral-prior cases: their internal targets can converge
  while validation velocity error remains large.
- smoke, spectrum-check, rerun-memory, and no-FEM smoke folders: these are
  implementation checks, not final numerical evidence.

## Present Diagnosis

The corrected code is Stokes-equation-valid, but the adaptive algorithm is not
yet a strong efficiency result. Current low-order regions are represented by
deactivating selected P2 velocity DOFs in the Taylor-Hood space. This is a valid
restricted-subspace experiment, but it is not yet a locally condensed physical
PNM/DDPNM response. As a result, a small number of unupgraded regions can still
dominate the velocity error, and accurate oracle runs typically activate about
`99%` of the mixed DOFs.

The adaptive restricted solve currently keeps all pressure DOFs active. This is
useful for stability, but it weakens any compression claim; report active DOF
ratios with that convention stated.

The next algorithmic step should be local static condensation or local reduced
Stokes bases for low-fidelity regions, followed by a posterior estimator that
correlates with FEM velocity error.

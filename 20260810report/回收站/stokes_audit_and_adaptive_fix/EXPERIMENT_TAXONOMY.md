# Experiment Taxonomy

This file separates the experiment families so the project narrative is not
buried under smoke runs and parameter sweeps.

Note: this checkout keeps only `outputs/final_state_figures/` on disk. The
output paths below are historical case identifiers recorded by the reports;
regenerate a case before citing it as locally reproducible evidence.

## A. Validated Correctness Baselines

These cases establish that the corrected line solves the intended Stokes system.

| case | role | use |
| --- | --- | --- |
| `outputs/berea_16_r16_pure_hoddpnm100` | full HODDPNM100 / full active interface release | cite as the clean FEM-equivalent baseline |
| `outputs/adaptive_stokes_berea_16_r16_preconditioned` | corrected adaptive Stokes with Schur-GMRES + ILU preconditioner | cite for corrected restricted Stokes machinery |
| `outputs/adaptive_stokes_berea_16_r16` | earlier corrected direct/restricted baseline | historical baseline; prefer the preconditioned run when discussing solver performance |

Main evidence:

- Pure HODDPNM100: active DOF ratio `1.0`, velocity relative L2 error
  `9.739e-13`, pressure relative L2 error `1.788e-15`.
- Preconditioned corrected baseline: final velocity relative L2 error
  `8.269e-13`, pressure relative L2 error `1.788e-15`.

## B. Oracle Diagnostic Experiments

These cases validate the adaptive mechanism, but they are not standalone
adaptive algorithms because marking uses FEM true error.

| case | result | use |
| --- | --- | --- |
| `outputs/adaptive_stokes_berea_16_r16_tol1e5_trueerr` | velocity relative L2 `5.535e-6`, active ratio `0.993` | best clean oracle tolerance-driven diagnostic |
| `outputs/adaptive_stokes_berea_16_r16_tol1e6_trueerr` | reaches machine precision, active ratio `1.0` | shows strict tolerance promotes to full space |
| `outputs/adaptive_stokes_berea_16_r16_proportional_interface_tol1e5` | velocity relative L2 `5.535e-6`, active ratio `0.993` | validates gradual interface-release idea |
| `outputs/adaptive_stokes_berea_16_r16_selective_monotone_trueerr_cap995` | velocity relative L2 `5.535e-6`, active ratio `0.993` | shows monotone marking fixes oscillation |
| `outputs/adaptive_stokes_berea_16_r16_selective_trueerr_*` | mixed accuracy | historical oracle sweeps; do not use as mainline |

Use these only with wording such as: "oracle diagnostic using FEM true-error
marking." Do not call them production adaptive estimators.

## C. Computable Adaptive Candidates

These use indicators available without a precomputed FEM reference for marking.
They are research candidates, not final results.

| family/case | status | interpretation |
| --- | --- | --- |
| `posterior_defect_cap90/cap98/cap995` | best computable posterior family so far | improves monotonically with cap, but `cap995` still has validation velocity error about `1.95e-2` |
| `staged_residual_tol1e5_*` | computable residual family | internal residual target can converge while validation velocity error remains large unless almost full space is reached |
| `geometry_prior_staged*` | computable geometry prior | geometry target is not reliable as an error estimator |
| `spectral_geometry_*` | computable spectral-geometry prior | spectral features do not yet predict velocity error sufficiently |
| `stokes_spectral_*` | computable Stokes spectral prior | same limitation as spectral geometry; useful for diagnostics, not final claims |
| `apriori_stokes_gain_*` | a priori gain prototype | converges its prior target, but validation velocity error remains too large |

Acceptable wording: "candidate estimator under development" or "negative
evidence for the current estimator." Avoid claiming these are validated adaptive
algorithms.

## D. Failed Or Superseded Experiments

These should not be used as main numerical evidence.

| pattern | reason |
| --- | --- |
| `*_smoke` | quick execution checks only |
| `*_spectrum_check*` | spectral diagnostic checks only |
| `*_rerun_memory` | memory reproduction, not a distinct method |
| `selective_cap80/cap90/cap97` | residual/non-oracle capped runs with poor velocity accuracy |
| non-monotone selective true-error runs | superseded by monotone strategy |
| no-FEM smoke outputs | cannot validate velocity/pressure error |

## Reporting Rule

Every new output folder should be assigned one of these labels:

- `baseline`
- `oracle-diagnostic`
- `computable-candidate`
- `smoke-or-check`
- `failed-or-superseded`

Every new report should state:

- whether marking used FEM true error;
- whether FEM was used only after adaptation for validation;
- active DOF ratio;
- velocity and pressure relative L2 errors against FEM, when available;
- whether the result is citeable as a Stokes validation result.

# Report Index

Use this index instead of reading the top-level reports alphabetically.

Current disk state: `outputs/` has been cleaned to retain only
`outputs/final_state_figures/`. Reports that name older `outputs/...` case
folders are preserving historical results, not asserting that those folders are
present in this checkout.

## Primary Narrative

Read these first.

| report | why it matters |
| --- | --- |
| `STOKES_AUDIT_REPORT.md` | defines the audit rule: only FEniCSx Taylor-Hood Stokes or restricted solves of the same matrix count |
| `HODDPNM_PRECONDITIONER_FINDINGS.md` | records the Schur-GMRES + ILU preconditioner attached to the corrected adaptive Stokes driver |
| `GEOMETRY_ADAPTIVE_VS_HODDPNM100_COMPARISON.md` | compares adaptive geometry-prior runs against pure HODDPNM100 and makes the cost caveat explicit |

## Oracle Diagnostics

These reports are useful for understanding what the adaptive mechanism can do
when local error information is idealized.

| report | status |
| --- | --- |
| `TOLERANCE_DRIVEN_ADAPTIVE_REPORT.md` | primary true-error tolerance diagnostic |
| `PROPORTIONAL_INTERFACE_HODDPNM_REPORT.md` | validates gradual interface-release mechanics under true-error marking |
| `SELECTIVE_ADAPTIVE_ITERATION_REPORT.md` | shows monotone marking fixes earlier selective oscillation, but still needs about `99%` active DOFs |

## Computable Estimator Attempts

These are the current standalone-adaptive research line. They are not final
successes yet.

| report | status |
| --- | --- |
| `POSTERIOR_DEFECT_ADAPTIVE_REPORT.md` | best computable posterior family so far; validation error still too high |
| `GEOMETRY_PRIOR_STAGED_HODDPNM_REPORT.md` | geometry-only prior; converges its own target but not reliable velocity error |
| `SPECTRAL_GEOMETRY_PRIOR_HODDPNM_REPORT.md` | spectral geometry prior; useful diagnostic but insufficient accuracy |
| `A_PRIORI_STOKES_GAIN_ADAPTIVE_REPORT.md` | a priori Stokes-gain prototype; prior target does not yet track velocity error |
| `A_PRIORI_GEOMETRY_STEKLOV_UPGRADE_RULE.md` | theoretical design note for geometry/Steklov upgrade logic |

## Negative Evidence To Keep, Not Lead With

These reports are valuable because they explain why earlier routes should not
be the main story.

| report | lesson |
| --- | --- |
| `GEOMETRY_ADAPTIVE_VS_HODDPNM100_COMPARISON.md` | adaptive iterations are not cheaper than pure HODDPNM100 when counted honestly |
| `SELECTIVE_ADAPTIVE_ITERATION_REPORT.md` | leaving low-order regions as deactivated P2 modes is too crude for strong savings |
| `SPECTRAL_GEOMETRY_PRIOR_HODDPNM_REPORT.md` | local geometry/spectrum alone does not yet predict global velocity error |

## Reporting Labels

Use the same labels as `EXPERIMENT_TAXONOMY.md`:

- `baseline`
- `oracle-diagnostic`
- `computable-candidate`
- `smoke-or-check`
- `failed-or-superseded`

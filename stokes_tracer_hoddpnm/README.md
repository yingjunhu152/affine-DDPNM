# Single-Phase Stokes-Tracer HODDPNM Validation

This folder validates single-phase tracer transport driven by a real-geometry FEniCSx Taylor-Hood P2-P1 Stokes velocity field.

The folder is self-contained for the formal run. It includes the local Stokes/HODDPNM support module, the small `classic_ddpnm.linalg` helper, and the cropped Berea input archive under `data\`.

Formal comparison:

- `FEM`: direct sparse solve of the full FEniCSx P2-P1 Stokes matrix.
- `HODDPNM`: known-fixed Schur-complement/static-condensation solve of the same matrix. Dirichlet dofs are removed from the active solve, and only free interface velocity dofs plus free interface P1 pressure dofs are retained as active Schur unknowns.

Adaptive method levels are intentionally removed from this folder. The HODDPNM result should be interpreted as a Schur/static-condensation validation, not as PNM/DDPNM/DDPNMT adaptive selection. The pressure interface thickness is controlled separately from the velocity interface thickness so the Schur boundary can retain interface pressure without reverting to all P1 pressure dofs.

The formal script enforces `--max-active-ratio 0.50` by default. The current Berea crop gives 11,521 active Schur dofs out of 42,532 full mixed dofs, i.e. 27.1%, instead of the older 33,461/42,532 case where fixed boundary dofs were incorrectly counted as active. Schur diagonal scaling and a modest pressure stabilization of `1e-6` are applied before GMRES, reducing the formal run from 935 to about 251 GMRES iterations and from 14.10 s to roughly 7 s for the HODDPNM Stokes solve on the current machine. This is a compressed and accelerated HODDPNM validation; the remaining O(10^2) Schur iterations show that a scalable Schur preconditioner remains the next optimization target.

## Run

```powershell
cd D:\hu\tongjiproj\FENICSX\stokes_tracer_hoddpnm
.\run_berea_tracer_step1.ps1
```

The formal output directory is:

```text
outputs\singlephase_tracer_hoddpnm_interface_pressure_t60_supg
```

Legacy smoke and earlier comparison outputs are archived outside `outputs\` under:

```text
outputs_archived_20260725_legacy
outputs_archived_20260725_experiments
```

## Main Outputs

- `TRACER_VALIDATION_REPORT.md`
- `tracer_summary.json`
- `tracer_metrics.csv`
- `stokes_velocity_cases.csv`
- `mass_balance_history.csv`
- `breakthrough_curves.png`
- `mass_balance_validation.png`
- `tracer_error_summary.png`
- `final_concentration_and_error.png`
- `fem_tracer_final.vtu`
- `hoddpnm_tracer_final.vtu`

## Validation Metrics

- Stokes velocity and pressure errors against FEM.
- HODDPNM active Schur dofs, active ratio, known fixed dofs, interface velocity/interface-pressure dof counts, and eliminated interior pressure dofs.
- HODDPNM Schur iterations, true reconstructed Schur residual, and Stokes solve time.
- Outlet breakthrough curve error against FEM-driven tracer.
- Final concentration-field integral L2 and Linf errors.
- Algebraic finite-element mass-balance residual at every time step.
- Bounded limiter diagnostics for the concentration range `[0, 1]`.


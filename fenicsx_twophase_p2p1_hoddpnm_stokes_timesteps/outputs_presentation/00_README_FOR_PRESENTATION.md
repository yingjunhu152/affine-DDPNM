# Presentation Output Index

Last organized: 2026-07-26 01:14, after the inlet/outlet boundary-condition update.

Use these folders for presentation:

- `01_stokes_reduction_validation/`
  - Schur/full FEM consistency validation, preconditioner comparison, and size timing.
  - Key figures: `preconditioner_comparison.png`, `size_timing_comparison.png`.
- `02_twophase_main_case/`
  - Main 27-hole, c4, 10-step FEniCSx two-phase timestep case.
  - Uses Corey mobility, inlet boundary-flux injection, natural outlet outflow, no inlet saturation reset.
  - Key files: `history.csv`, `twophase_fenicsx_p2p1_hoddpnm_report.json`, PyVista-style `voidspace_two_phase_volume_split_clean_step_XXXX.png` frames, and standard history plots.
- `03_minimal_twophase_validation/`
  - Compact cube-minus-sphere validation batch with coarse and medium cases.
  - Key file: `minimal_validation_summary.md`.
- `04_pressure_stabilization/`
  - Pressure-stabilization sensitivity results.
  - Report zero-mean pressure ranges and mean-aligned pressure errors, not raw pressure magnitude as a physical benchmark.
- `05_selected_figures/`
  - Small hand-picked figure folder for quick slide building.
  - The main timestep figures are `03_main_twophase_step_0000.png`, `04_main_twophase_step_0005.png`, and `05_main_twophase_step_0010.png`.

Archived or not recommended for presentation:

- `_archive_smoke_and_old/`
  - Smoke tests, temporary outputs, old mass/visualization figures, and the old root `outputs/` contents.
- The root `outputs/` folder is intentionally empty after cleanup and should not be shown as formal results.

Important reporting caveats:

- HODDPNM errors compare Schur reconstruction with the full solve of the same assembled FEniCSx P2-P1 matrix.
- The two-phase transport is still graph-edge Corey transport with boundary-flux injection; it is not yet the final cell-wise conservative finite-volume face-flux discretization.
- `inlet_reset_mass_change` is retained as a legacy CSV column, but normalized inlet/outlet runs should have `inlet_reset_mass_change = 0`.
- Use `inlet_boundary_flux_mass`, `outlet_boundary_flux_mass`, `net_flux_mass`, `relative_mass_error`, and CFL fields for boundary-flux diagnostics.

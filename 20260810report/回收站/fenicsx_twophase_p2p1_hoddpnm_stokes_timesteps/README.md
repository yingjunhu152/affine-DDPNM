# FEniCSx P2-P1 Schur Reduction And SciPy Two-Phase Demo

This folder contains three separate lines:

- a FEniCSx-assembled P2-P1 Taylor-Hood Stokes Schur/HODDPNM reduction validation;
- a FEniCSx-assembled P2-P1 timestep prototype that reassembles Stokes from Corey `mu_eff(Sw)` at every step and writes its own timestep plots;
- a pure-SciPy timestep visualization demo.

It does not replace the older PNM two-phase folders.

Use this for Schur/full reduction validation:

`run_random_pnm_taylor_hood_validation.py`

Use this for the unified FEniCSx timestep prototype:

`run_twophase_fenicsx_p2p1_hoddpnm_timesteps.py`

Use this only for original-style timestep pictures:

`run_twophase_p2p1_hoddpnm_stokes_timesteps_scipy.py`

The FEniCSx timestep prototype now puts `Sw^n -> mu_eff(Sw^n) -> FEniCSx P2-P1 Stokes assembly -> full/HODDPNM solve -> u_h^n -> Sw^{n+1}` in one script and writes timestep field PNGs plus `history.csv`. Its saturation transport is still a graph-edge update, now with Corey fractional-flow upwinding, inlet boundary-flux injection, outlet natural outflow, and CFL diagnostics. It is not yet a conservative finite-volume two-phase PDE validation.

## Model

The Stokes solve uses continuous finite elements:

```text
-div(mu_eff(Sw) grad(u)) + grad(p) = 0
div(u) = 0
```

with:

- velocity: P2 Taylor-Hood vector field;
- pressure: P1 Taylor-Hood scalar field;
- `mu_eff(Sw) = 1 / (lambda_w(Sw) + lambda_o(Sw))`, with Corey relative permeabilities;
- `fw(Sw) = lambda_w(Sw) / (lambda_w(Sw) + lambda_o(Sw))` for upwind saturation transport;
- HODDPNM: Schur/static-condensation reconstruction for the same assembled discrete linear system;
- saturation update: empirical vertex graph-edge transport driven by the HODDPNM Stokes velocity, now using Corey fractional-flow upwinding, incoming x=0 boundary flux with `Sw_in`, and natural x=L outflow.

This is no longer the pore-network pressure equation. The reported velocity/pressure differences are not physical errors against a true 3D two-phase reference solution; they compare the Schur-reconstructed solution with the full solve of the same discrete matrix. The inlet saturation is no longer imposed by resetting boundary vertices after each step; it enters through an incoming boundary flux. The saturation transport is still not the final cell-wise finite-volume face-flux discretization.

## Files

- `run_twophase_p2p1_hoddpnm_stokes_timesteps_scipy.py`: runnable P2-P1 timestep consistency demo and PyVista renderer, assembled directly with SciPy. It imports the copied mesh helper from this folder and no longer injects a hard-coded `D:\...` source path.
- `run_twophase_fenicsx_p2p1_hoddpnm_timesteps.py`: FEniCSx timestep prototype. Each step assembles a P2-P1 Stokes matrix with DG0 cellwise Corey `mu_eff(Sw)`, solves full and HODDPNM systems, then advances `Sw` with graph-edge Corey fractional-flow transport plus inlet/outlet boundary fluxes. With `--plot`, it writes PyVista-style two-panel voidspace phase-rendering PNGs plus standard history plots and history tables.
- `run_minimal_twophase_validation.py`: batch runner for a minimal cube-minus-sphere two-phase validation set. It runs coarse and medium cases and writes PyVista-style timestep frames plus CSV/JSON/Markdown summaries.
- `run_cube_holes_hoddpnm_validation.py`: copied cube-minus-sphere mesh utility and stabilized P1-P1 Stokes Schur consistency demo. It is not a P2-P1 Taylor-Hood physical error validator.
- `run_random_pnm_taylor_hood_validation.py`: FEniCSx/dolfinx UFL assembly for Taylor-Hood P2-P1 Stokes. Its reference is still the same discrete FEM system, but the default HODDPNM interface now eliminates pressure interior dofs and keeps only PNM-interface pressure dofs plus one pressure anchor per eliminated pressure component.
- `make_presentation_summaries.py`: collects official JSON outputs under `outputs_presentation/` into CSV tables and presentation plots.
- `classic_ddpnm/`: copied local conversion helpers used to convert FEniCSx matrices/vectors to SciPy.

## FEniCSx P2-P1 Reduction Run

This is the preferred entry point for the FEniCSx assembly and stronger reduction-meaning question:

```powershell
cd D:\hu\tongjiproj\FENICSX\fenicsx_twophase_p2p1_hoddpnm_stokes_timesteps
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
D:\Miniconda3\Scripts\conda.exe run -n fenicsx --no-capture-output python run_random_pnm_taylor_hood_validation.py --holes-per-axis 3 --cells-per-axis 4 --pressure-boundary-mode interface-anchors --schur-solver gmres --schur-preconditioner ilu --out-dir outputs_presentation\01_reduction_validation\preconditioner_ilu_27holes_c4
```

The important report fields are:

- `assembly`: confirms FEniCSx/dolfinx UFL assembly with Basix P2 vector velocity and P1 scalar pressure.
- `pressure_boundary_mode`: default `interface-anchors`.
- `pressure_boundary_dofs`: pressure dofs retained in the pre-fixed-removal Schur interface mask.
- `pressure_interior_dofs_eliminated`: pressure dofs actually eliminated.
- `velocity_interface_dofs`: velocity dofs retained in the Schur boundary.
- `hoddpnm_interface_dofs`: interface dofs before removing known fixed Dirichlet dofs.
- `hoddpnm_active_schur_dofs`: free active Schur unknowns after known fixed dofs are removed.
- `hoddpnm_active_dof_ratio`: active Schur dofs divided by mixed dofs.
- `hoddpnm_known_fixed_dofs`: Dirichlet dofs removed as known values before the Schur solve.
- `hoddpnm_free_interior_dofs_eliminated`: free interior mixed dofs reconstructed by static condensation.
- `schur_solver`: `gmres` by default.
- `schur_preconditioner`: `ilu`, `diag`, or `none`; default is `ilu`.
- `dense_schur_used`: `false` for the matrix-free Schur path.
- `schur_iterations`, `schur_relative_residual`: GMRES diagnostics.
- `assembly_time_seconds`, `full_solve_time_seconds`, `hoddpnm_solve_time_seconds`, `total_time_seconds`: separated timing fields.
- `errors_pressure_p1_mean_aligned`: pressure error after removing the pressure gauge constant from both solutions.

Use `--pressure-boundary-mode all` only as the old baseline where all pressure dofs are retained.

The default `--pressure-stabilization 1e-10` is a tiny pressure regularization used to keep the reduced-pressure Schur interior blocks invertible. A smoke test with `--pressure-stabilization 0` produced an exactly singular matrix/interior block in this current setup, so do not describe this version as an unstabilized pure Taylor-Hood Schur reduction. Pressure is gauge-dependent, so report the zero-mean pressure range and mean-aligned pressure error instead of overemphasizing the raw absolute pressure scale.

This line is steady Stokes only. It does not advance saturation and does not generate two-phase timestep frames.

The old dense Schur path is still available with `--schur-solver dense`, but it is validation-scale only:

```text
Xi = A_ii^{-1} A_ib
S = A_bb - A_bi Xi
```

The default GMRES path applies the Schur complement matrix-free:

```text
Sx = A_bb x - A_bi A_ii^{-1} A_ib x
```

so it avoids `Aib.toarray()` and does not explicitly store dense `S`.

## FEniCSx Two-Phase Timestep Prototype

```powershell
cd D:\hu\tongjiproj\FENICSX\fenicsx_twophase_p2p1_hoddpnm_stokes_timesteps
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
D:\Miniconda3\Scripts\conda.exe run -n fenicsx --no-capture-output python run_twophase_fenicsx_p2p1_hoddpnm_timesteps.py --holes-per-axis 2 --cells-per-axis 3 --steps 2 --pressure-boundary-mode interface-anchors --schur-solver gmres --out-dir outputs\fenicsx_twophase_timesteps_8holes_c3_gmres
```

For presentation-scale output:

```powershell
D:\Miniconda3\Scripts\conda.exe run -n fenicsx --no-capture-output python run_twophase_fenicsx_p2p1_hoddpnm_timesteps.py --holes-per-axis 3 --cells-per-axis 4 --steps 10 --plot --plot-steps 1 5 10 --pressure-boundary-mode interface-anchors --schur-solver gmres --schur-preconditioner ilu --out-dir outputs_presentation\02_twophase_timesteps\fenicsx_27holes_c4_steps10
```

This is the unified line where saturation changes and FEniCSx P2-P1 HODDPNM are in the same timestep loop. It writes `twophase_fenicsx_p2p1_hoddpnm_report.json`, `history.csv`, and timestep field PNGs.
The pressure timestep PNGs plot the HODDPNM-reconstructed pressure after removing its mean.

The `history.csv` table contains:

- `step`, `mean_S2`, `mass_error`, `velocity_error`, `pressure_error`, `schur_iterations`;
- mass bookkeeping fields: `mass_before`, `mass_after_graph_transport`, `mass_after_boundary_flux`, `mass_after_clipping`, `saturation_mass`, `expected_mass_after`;
- mass-change fields: `graph_transport_mass_change`, `boundary_flux_mass_change`, `clipping_mass_change`, `inlet_reset_mass_change`;
- boundary-source mass fields for the current graph transport model: `mass_residual`, `relative_mass_error`, `inlet_boundary_flux_mass`, `outlet_boundary_flux_mass`, `inlet_flux_mass`, `outlet_flux_mass`, `net_flux_mass`;
- CFL fields: `dt`, `max_cfl`, `stable_dt`, `cfl_number`, `max_water_cfl`, `max_inlet_flux_cfl`, `max_outlet_flux_cfl`;
- saturation bounds: `saturation_min`, `saturation_max`, `saturation_effective_min`, `saturation_effective_max`;
- Schur size fields: `hoddpnm_interface_dofs`, `hoddpnm_active_schur_dofs`, `hoddpnm_active_dof_ratio`, `hoddpnm_known_fixed_dofs`, `hoddpnm_free_interior_dofs_eliminated`;
- `assemble_time_seconds`, `full_stokes_solve_time_seconds`, `hoddpnm_schur_solve_time_seconds`, `total_time_seconds`;
- pressure error is mean-aligned, while the raw pressure range is kept only as a scale diagnostic.

The `mass_error` column is now a relative bookkeeping residual after separately accounting for graph transport mass change, boundary-flux mass change, and clipping mass change. `inlet_reset_mass_change` is kept as a legacy column but is zero in the normalized inlet/outlet boundary-condition path. `relative_mass_error` checks `M_after - M_before - inlet_flux_mass + outlet_flux_mass`; if clipping changes mass, this residual exposes it. These plots are still graph-transport diagnostics; strict finite-volume two-phase conservation requires the next cell-wise face-flux update.

## Minimal Two-Phase Validation

Run the compact cube-minus-sphere validation batch with:

```powershell
cd D:\hu\tongjiproj\FENICSX\fenicsx_twophase_p2p1_hoddpnm_stokes_timesteps
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
D:\Miniconda3\Scripts\conda.exe run -n fenicsx --no-capture-output python run_minimal_twophase_validation.py
```

Default cases:

- `coarse`: `holes_per_axis=2`, `cells_per_axis=3`, `steps=20`;
- `medium`: `holes_per_axis=3`, `cells_per_axis=4`, `steps=20`.

The script uses uniform `Sw_initial=0.2`, incoming x=0 boundary-flux saturation `Sw_inlet=1.0`, natural x=L outflow, no inlet saturation reset, `dt=0.08`, `transport_scale=0.18`, Corey parameters `Swr=0.2`, `Sor=0.2`, `nw=no=2`, `mu_w=1`, `mu_o=5`, and `gmres/ilu` for the HODDPNM Schur solve. It writes one run folder per case plus:

- `outputs_presentation\02_twophase_timesteps\minimal_validation\minimal_validation_summary.csv`
- `outputs_presentation\02_twophase_timesteps\minimal_validation\minimal_validation_summary.json`
- `outputs_presentation\02_twophase_timesteps\minimal_validation\minimal_validation_summary.md`

Each run folder contains `history.csv`, `twophase_fenicsx_p2p1_hoddpnm_report.json`, field snapshots for steps `1`, `10`, and `20`, plus the standard history plots. This is a reproducible small-case validation harness for saturation-front behavior, saturation bounds, CFL margin, boundary-flux mass diagnostics, and full-FEM/HODDPNM consistency. It is still graph-edge Corey transport; do not present it as the final conservative FV face-flux discretization.

## Presentation Outputs

Final presentation-use outputs are separated under `outputs_presentation/`:

- `00_README_FOR_PRESENTATION.md`: teacher-facing index and reporting caveats.
- `01_stokes_reduction_validation/`: FEniCSx P2-P1 reduced Schur validation, preconditioner comparison, and size timing comparison.
- `02_twophase_main_case/`: main 27-hole, c4, 10-step two-phase boundary-flux timestep case.
- `03_minimal_twophase_validation/`: compact coarse/medium cube-minus-sphere two-phase validation batch.
- `04_pressure_stabilization/`: pressure-stabilization sensitivity outputs and mean-aligned pressure-error comparison.
- `05_selected_figures/`: copied high-priority figures for quick slide building.
- `_archive_smoke_and_old/`: smoke tests, temporary outputs, old mass/visualization figures, and the previous root `outputs/` contents.

The older formal source folders `01_reduction_validation/`, `02_twophase_timesteps/`, and `04_reference_error/` are retained for script compatibility, but the cleaner folder names above are the recommended presentation entry points.

Regenerate the summary tables and plots with:

```powershell
D:\Miniconda3\Scripts\conda.exe run -n fenicsx --no-capture-output python make_presentation_summaries.py
```

## SciPy Timestep Visualization Run

```powershell
cd D:\hu\tongjiproj\FENICSX\fenicsx_twophase_p2p1_hoddpnm_stokes_timesteps
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
& D:\Miniconda3\envs\fenicsx\python.exe run_twophase_p2p1_hoddpnm_stokes_timesteps_scipy.py --holes-per-axis 3 --cells-per-axis 5 --steps 6 --frame-steps 0 1 2 3 4 5 6 --dt 0.14 --transport-scale 0.75 --geometry-channel-strength 1.25 --capillary-spread 0.0005 --render-geometry-strength 1.0 --render-front-contrast 1.25 --render-resolution 90 --out-dir outputs\scipy_twophase_visual_demo_27holes_cells5_original_style
```

Outputs:

`outputs\scipy_twophase_visual_demo_27holes_cells5_original_style`

The folder contains:

- `twophase_p2p1_hoddpnm_stokes_report.json`
- `voidspace_two_phase_volume_split_clean_step_XXXX.png`
- optional VTU files if `--write-vtu` is used.

## Useful Parameters

- `--holes-per-axis`: default `2` for a small smoke test; use `3` for 27 holes.
- `--cells-per-axis`: default `4`; increase gradually.
- `--steps`: number of time steps.
- `--frame-steps 0 1 2 4 8 12`: explicit frame selection.
- `--mu-original`, `--mu-injected`: phase viscosities.
- `--dt`, `--transport-scale`: saturation update controls.
- `--geometry-channel-strength`: increases graph transport along pore edges near sphere surfaces, making the injected phase enter the irregular void region more visibly.
- `--capillary-spread`: small saturation-front spreading term along graph edges.
- `--render-resolution`: PyVista volume-grid resolution used for the original-style split render.
- `--render-geometry-strength`, `--render-front-contrast`: visualization-only controls that make the volume-rendered interface easier to see without changing the Stokes solve.
- `--linear-solver minres`: default checked solver for the full and Schur boundary systems.
- `--minres-residual-rtol`: accepted actual relative residual threshold; default `2e-6` for this demo, not machine precision.
- `--no-plot`: write only JSON report, no PNG rendering.
- `--write-vtu`: also write VTU files for ParaView/PyVista inspection.

## Interpretation Limits

- `velocity_l2_rel_difference_vs_full_discrete` and `pressure_l2_rel_difference_vs_full_discrete` compare HODDPNM Schur reconstruction with the full solve of the same assembled P2-P1 linear system.
- These fields are consistency differences, not convergence rates or physical errors versus an analytical/over-refined 3D two-phase Stokes solution.
- The steady FEniCSx reduction line, unified FEniCSx timestep prototype, and SciPy visual demo should be reported separately.
- The script now records `full_linear_solve` and `schur_linear_solve` metadata in the JSON report and raises an error if the selected solver does not meet the accepted stop/residual criteria.
- The explicit saturation update is a graph-edge demonstration transport model. It is not a verified conservative two-phase flow discretization.
- Known fixed Dirichlet dofs are removed before the Schur solve. Report `hoddpnm_active_schur_dofs` and `hoddpnm_active_dof_ratio` as the active reduced-system size; report `hoddpnm_interface_dofs` only as the pre-fixed-removal interface count.
- The current 27-hole timestep case should be presented as a Schur/full consistency and diagnostic-quality timestep demonstration unless the active Schur dof ratio from the regenerated output is small enough for the specific efficiency claim being made.
- Raw pressure magnitude is strongly affected by the pressure stabilization. Use zero-mean pressure ranges and mean-aligned pressure errors; do not present raw pressure means or raw absolute pressure magnitude as physical pressure benchmarks.

## Environment Note

The FEniCSx reduction script requires `dolfinx`, `basix`, `ufl`, `mpi4py`, and a working local MPI runtime. In the current Windows environment, direct execution through `D:\Miniconda3\envs\fenicsx\python.exe` can fail during `MPI_Init_thread`, while `D:\Miniconda3\Scripts\conda.exe run -n fenicsx --no-capture-output python ...` initializes MPI correctly. Use the `conda run` form for FEniCSx scripts.

The SciPy timestep visualization script does not require Basix or FEniCSx and remains useful for plot generation, but it should not be cited as the FEniCSx assembly path.

## Verified Runs

Formal outputs now live under `outputs_presentation/`.

For `outputs_presentation\02_twophase_main_case`:

- geometry: `holes_per_axis=3`, `cells_per_axis=4`, `steps=10`;
- mixed dofs: `2288`; velocity dofs: `2163`; pressure dofs: `125`;
- pre-fixed-removal HODDPNM interface dofs: `874`;
- active Schur dofs after known fixed removal: `31` (`1.35%` of mixed dofs);
- known fixed Dirichlet dofs removed before Schur: `2146`;
- free interior dofs reconstructed by static condensation: `111`;
- pressure Schur boundary dofs: `31`; pressure interior dofs eliminated: `94`;
- Schur solver/preconditioner: `gmres/ilu`; dense Schur used: `false`; iterations: `7` at every recorded step;
- Corey parameters: `Swr=0.2`, `Sor=0.2`, `nw=no=2`, `mu_w=1`, `mu_o=5`;
- boundary transport: uniform initial `Sw=0.05`, incoming x=0 boundary-flux saturation `Sw_in=0.95`, natural x=L outflow, no inlet saturation reset;
- mean S2 increases from `0.0566` to `0.1070`;
- saturation stays in `[0.05, 0.6525]`;
- velocity Schur/full difference is about `6.92e-9`;
- mean-aligned pressure Schur/full difference is about `5.36e-9`;
- graph-model `relative_mass_error` stays at roundoff; `inlet_reset_mass_change=0`, inlet boundary flux mass is about `0.36` per step, and outlet flux is zero over this short 10-step run;
- `max_cfl` is about `0.1106` and `max_inlet_flux_cfl` is about `0.0885`, both below the default `--cfl-limit 0.5`;
- PyVista-style two-panel phase-rendering images exist for steps `0` through `10`: `voidspace_two_phase_volume_split_clean_step_XXXX.png`.
- standard history plots are generated directly by `run_twophase_fenicsx_p2p1_hoddpnm_timesteps.py`: HODDPNM/full errors, mass diagnostics, mean saturation, active dofs, and CFL.

For `outputs_presentation\03_minimal_twophase_validation`:

- `coarse_8holes_c3_steps20`: final mean Sw `0.2880`, saturation range `[0.200, 0.886]`, max `relative_mass_error` `1.42e-16`, max CFL `0.062`, active Schur dofs `7` (`0.64%`), max velocity/pressure differences `4.88e-7` / `7.35e-8`;
- `medium_27holes_c4_steps20`: final mean Sw `0.2852`, saturation range `[0.200, 1.000]`, max `relative_mass_error` `1.74e-4`, max CFL `0.111`, active Schur dofs `31` (`1.35%`), max velocity/pressure differences `7.12e-9` / `5.37e-9`;
- the medium case shows a small nonzero boundary-source relative mass residual once clipping appears, while `mass_error` bookkeeping stays at zero. This is a diagnostic result, not a strict FV conservation claim.

For pressure-stabilization sensitivity in `outputs_presentation\04_pressure_stabilization`:

- `1e-8`: pressure mean about `3.79e7`, zero-mean pressure range about `[-1.28e8, 2.09e8]`, mean-aligned pressure error `5.37e-9`;
- `1e-10`: pressure mean about `3.79e9`, zero-mean pressure range about `[-1.28e10, 2.09e10]`, mean-aligned pressure error `5.38e-9`;
- `1e-12`: pressure mean about `3.79e11`, zero-mean pressure range about `[-1.28e12, 2.09e12]`, mean-aligned pressure error `1.31e-8`.

This confirms the raw pressure scale grows roughly as stabilization is tightened. Report the zero-mean pressure range and mean-aligned pressure error; do not present raw pressure magnitude as a physical pressure benchmark.

For Schur preconditioner comparison on the 27-hole c4 case:

- all three runs have `31` active Schur dofs (`1.35%`) after removing `2146` known fixed dofs;
- `none`: 14 GMRES iterations, residual `4.41e-8`, HODDPNM time `0.0083 s`;
- `diag`: 11 GMRES iterations, residual `9.53e-8`, HODDPNM time `0.0033 s`;
- `ilu`: 7 GMRES iterations, residual `3.04e-8`, HODDPNM time `0.0035 s`.

The known-fixed Schur path now gives a genuinely small active Schur system for this boundary-heavy validation mesh. This should still be described as Schur/full consistency plus first solver-size diagnostics, not as a final scalable solver study.

For size timing comparison in `outputs_presentation\01_stokes_reduction_validation\size_timing_comparison.csv`:

- cases were run for `holes_per_axis=2,3,4` and `cells_per_axis=3,4`;
- active Schur ratios in these validation-size cases range from about `0.64%` to `2.46%` after known fixed dofs are removed;
- each JSON separates `assembly_time_seconds`, `full_solve_time_seconds`, `hoddpnm_solve_time_seconds`, `schur_iterations`, and `total_time_seconds`;
- `size_timing_comparison.png` is log-scale and should be used to discuss scaling trends cautiously, since these meshes are still validation-size.

Generated summary artifacts:

- `outputs_presentation\01_stokes_reduction_validation\preconditioner_comparison.csv`
- `outputs_presentation\01_stokes_reduction_validation\preconditioner_comparison.png`
- `outputs_presentation\01_stokes_reduction_validation\size_timing_comparison.csv`
- `outputs_presentation\01_stokes_reduction_validation\size_timing_comparison.png`
- `outputs_presentation\04_pressure_stabilization\pressure_stabilization_sensitivity.csv`
- `outputs_presentation\04_pressure_stabilization\pressure_stabilization_sensitivity.png`

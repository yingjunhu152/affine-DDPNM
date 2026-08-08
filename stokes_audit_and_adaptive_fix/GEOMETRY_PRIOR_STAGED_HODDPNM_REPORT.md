# Geometry-Prior Staged HODDPNM Prototype

This is a first algorithmic prototype for FEM-independent adaptive region selection.
The adaptive marking step does not use a FEM reference solution. FEM can still be
computed with `--reference-solve after`, but only for final validation.

## Algorithm

Each region receives a geometry complexity score from voxel-only features:

- solid surface density
- inter-region interface density
- number of neighboring regions
- local bounding-box void deficit
- constriction risk from distance to solid
- x-direction cross-section risk
- x-normal interface cut fraction

The staged residual risk is then modeled as

```text
eta_i(stage) = geometry_complexity_i * tail(stage)
```

with the current prototype tail

```text
PNM: 1.0
DDPNM: 0.62
DDPNMT: 0.42
HODDPNM-25%: 0.28
HODDPNM-50%: 0.16
HODDPNM-75%: 0.07
HODDPNM-100%: 0.015
```

At each cycle, regions are ranked by this geometry-prior indicator and marked by
Dorfler marking. A marked region is upgraded by one stage:

```text
PNM -> DDPNM -> DDPNMT -> HODDPNM-25% -> HODDPNM-50% -> HODDPNM-75% -> HODDPNM-100%
```

Within proportional HODDPNM stages, interface nodes are released in geometry-risk
order: nodes near solid boundaries and inside high-complexity regions are released
first.

## Main Validation Run

Command:

```powershell
.\run_geometry_prior_staged_berea.ps1
```

Output directory:

```text
outputs\adaptive_stokes_berea_16_r16_geometry_prior_staged_cap100
```

Key result:

- Mixed TH Stokes DOFs: `42532`
- Final active DOFs: `41998` (`98.744%`)
- Final adaptive restricted solve time: `1.552 s`
- FEM reference time, computed after adaptation only: `7.402 s`
- Geometry-prior target: `1.854750e-02`
- Geometry-prior tolerance: `2.000000e-02`
- Converged to geometry tolerance: `True`
- Final stages: PNM `0`, DDPNM `0`, DDPNMT `0`, HODDPNM `16`
- HODDPNM stage split: H25 `1`, H50 `1`, H75 `1`, H100 `13`
- Released interface nodes: `4983 / 5253` (`94.860%`)
- Validation velocity relative L2 error: `2.349754e-01`
- Validation pressure relative L2 error: `1.047771e-08`

## Important Interpretation

This is not yet a reliable a priori error estimator. It is only a geometry-driven
selection skeleton. The pressure is already close to FEM in this run, but the
velocity error remains too large even after most interface DOFs are released.

That means the next mathematical step should not be more hand tuning of these
weights. The next serious step should add a local interface spectral indicator
or a local Schur/Steklov surrogate, so that the geometry score predicts interface
trace complexity rather than only voxel roughness.

## Produced Files

- `geometry_region_prior.csv`: per-region geometry features and complexity scores
- `adaptive_stokes_history.csv`: staged adaptive history
- `validation_summary.json`: full run summary and validation data
- `final_region_method_map.png`: final region-stage map
- `cycle_method_progression.png`: staged evolution
- `real_porous_hoddpnm_solution.vtu`: final solution fields

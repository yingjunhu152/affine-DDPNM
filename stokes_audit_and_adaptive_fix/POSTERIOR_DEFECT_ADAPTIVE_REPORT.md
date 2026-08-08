# Posterior-Defect Adaptive Stokes HODDPNM

This is the first a posteriori prototype after abandoning the pure geometry/a-priori route.

## Numerical Model

- Equation: 3D incompressible Stokes.
- Discretization: FEniCSx Taylor-Hood mixed element, P2 velocity and P1 pressure.
- Geometry: real Berea voxel pore domain, crop `20:36,150:166,30:46`.
- Region count: 16.
- Adaptive ladder: `PNM -> DDPNM -> DDPNMT -> HODDPNM25 -> HODDPNM50 -> HODDPNM75 -> HODDPNM100`.
- Restricted solver: Schur GMRES with ILU preconditioner.

## Posterior Indicator

The adaptive selection no longer uses the FEM reference solution.  The FEM solution is only computed after the adaptive loop for validation.

For each region, the indicator is assembled from the current restricted solution defect

```text
r = A u_h - b
```

where `A` and `b` are the full Taylor-Hood Stokes matrix and vector.  The regional score combines

- inactive velocity DOF defect,
- interface velocity DOF defect,
- pressure/divergence equation defect,
- full mixed regional defect.

Default weights:

```text
inactive velocity : interface velocity : pressure/divergence : mixed region
1.0 : 0.75 : 0.65 : 0.25
```

Region upgrades use Doerfler marking on this posterior-defect score.  Within HODDPNM stages, released interface nodes are ranked by local node defect, so HODDPNM does not open all interface DOFs at once.

## Current Results

| case | active DOFs | active ratio | final regions | velocity rel L2 error vs FEM | pressure rel L2 error vs FEM | final restricted solve | adaptive solve sum | FEM-after solve | total wall time | end working memory | peak memory |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pure HODDPNM100 | 42532 | 1.000 | HODDPNM=16 | 9.739e-13 | 1.788e-15 | 1.498 s | 1.498 s | 7.729 s | 12.928 s | 363.508 MiB | 985.512 MiB |
| Posterior cap90 | 37966 | 0.893 | PNM=2, DDPNM=1, HODDPNM=13 | 6.491e-01 | 5.032e-08 | 3.913 s | 33.234 s | 31.720 s | 97.453 s | 463.949 MiB | 994.734 MiB |
| Posterior cap98 | 41353 | 0.972 | PNM=1, DDPNM=1, HODDPNM=14 | 1.144e-01 | 6.650e-10 | 4.757 s | 64.726 s | 34.556 s | 133.339 s | 465.598 MiB | 996.414 MiB |
| Posterior cap995 | 42163 | 0.991 | DDPNM=1, HODDPNM=15 | 1.950e-02 | 3.521e-11 | 5.000 s | 92.389 s | 33.833 s | 161.208 s | 466.965 MiB | 998.145 MiB |

## Interpretation

The a posteriori defect indicator is clearly better than the failed pure geometry-prior route: increasing the active DOF cap monotonically improves the validated velocity error from about `6.49e-1` to `1.95e-2`.

However, this residual-defect prototype is not yet good enough as a final paper algorithm.  It can strongly reduce the algebraic defect while still leaving a small number of low-order regions that dominate the velocity error.  At `99.5%` active DOFs, one DDPNM region remains because of the cap, and that single region still leaves about `2%` velocity error.

The next mathematically cleaner posterior route should be a hierarchical a posteriori estimator: solve a local enriched correction on candidate regions or patches, then mark by the local correction energy rather than by the raw residual alone.

## Generated Files

- Main code: `stokes_adaptive_taylor_hood.py`
- Repro script: `run_posterior_defect_adaptive_berea.ps1`
- Smoke output: `outputs/adaptive_stokes_berea_16_r16_posterior_defect_smoke`
- Main posterior output: `outputs/adaptive_stokes_berea_16_r16_posterior_defect_cap995`
- History CSV: `outputs/adaptive_stokes_berea_16_r16_posterior_defect_cap995/adaptive_stokes_history.csv`
- Posterior component CSV: `outputs/adaptive_stokes_berea_16_r16_posterior_defect_cap995/posterior_defect_components.csv`
- Method progression image: `outputs/adaptive_stokes_berea_16_r16_posterior_defect_cap995/cycle_method_progression.png`
- Final region map: `outputs/adaptive_stokes_berea_16_r16_posterior_defect_cap995/final_region_method_map.png`

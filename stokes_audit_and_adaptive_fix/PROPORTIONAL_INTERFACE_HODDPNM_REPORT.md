# Proportional-Interface HODDPNM Adaptive Report

## Motivation

The previous tolerance-driven adaptive run still changed regions by discrete method levels. This was too coarse. The new idea is to treat HODDPNM refinement as a released-interface-DOF fraction instead of a pure region upgrade.

The new strategy is implemented in:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\stokes_adaptive_taylor_hood.py`

Run script:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\run_proportional_interface_adaptive_berea.ps1`

## Algorithm

Strategy name:

`proportional-interface`

State variable:

- each region owns an interface fraction stage;
- default fractions are `0, 0.25, 0.5, 0.75, 1.0`;
- this means a region can gradually release 0%, 25%, 50%, 75%, or 100% of its selected interface velocity nodes.

Base DOFs:

- pressure DOFs are always active;
- grid-vertex velocity DOFs are always active;
- fixed boundary DOFs are always active.

Proportional DOF release:

- for a region with fraction stage greater than zero:
  - non-interface P2 velocity nodes in that region are active;
  - interface P2 velocity nodes are sorted by local velocity-error score;
  - only the top fraction of interface nodes is activated;
  - for each selected velocity node, the current implementation releases all three Cartesian components.

Important caveat:

- this is a Cartesian-component prototype;
- it does not yet transform velocity DOFs into local normal/tangential interface bases;
- the next refinement should release `u.n`, `u.t1`, and `u.t2` components selectively.

Upgrade rule:

- compute local region indicator `eta_r`;
- validation run uses `true-error` against the FEM reference;
- sort by `eta_r^2`;
- use Doerfler marking with `theta = 0.65`;
- upgrade at most `3` regions per cycle;
- each marked region increases its interface fraction by one stage only;
- no demotion is allowed.

## Berea result, tolerance 1e-5

Output folder:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\outputs\adaptive_stokes_berea_16_r16_proportional_interface_tol1e5`

Final result:

| item | value |
| --- | ---: |
| FEM direct reference time | 8.144 s |
| Final adaptive solve time | 1.570 s |
| Mixed DOFs | 42532 |
| Active DOFs | 42241 |
| Active DOF ratio | 99.316% |
| Velocity relative L2 error | 5.535e-06 |
| Pressure relative L2 error | 1.209e-13 |
| Schur-GMRES iterations | 42 |
| Mean interface fraction | 0.750 |
| Released interface nodes | 4239 / 5253 |
| Released interface-node ratio | 80.697% |
| Fraction stages | 0%: 1 region, 25%: 1 region, 50%: 1 region, 75%: 7 regions, 100%: 6 regions |

## Iteration history

| cycle | active % | solve time (s) | velocity error | mean interface fraction | released interface nodes | fraction stages |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 62.40 | 0.438 | 1.547e+01 | 0.000 | 0 | 0%:16 |
| 1 | 64.97 | 0.478 | 3.162e+00 | 0.031 | 129 | 0%:14, 25%:2 |
| 2 | 65.79 | 0.460 | 3.053e+00 | 0.047 | 206 | 0%:13, 25%:3 |
| 3 | 66.83 | 0.550 | 2.550e+00 | 0.063 | 260 | 0%:12, 25%:4 |
| 4 | 67.63 | 0.569 | 2.252e+00 | 0.078 | 310 | 0%:11, 25%:5 |
| 5 | 71.89 | 0.635 | 1.326e+00 | 0.125 | 620 | 0%:9, 25%:6, 50%:1 |
| 6 | 75.54 | 0.799 | 1.089e+00 | 0.172 | 855 | 0%:8, 25%:6, 50%:1, 75%:1 |
| 7 | 78.27 | 0.860 | 8.770e-01 | 0.219 | 1064 | 0%:6, 25%:8, 50%:1, 100%:1 |
| 8 | 79.69 | 1.014 | 8.820e-01 | 0.266 | 1341 | 0%:6, 25%:6, 50%:2, 75%:1, 100%:1 |
| 9 | 80.52 | 1.122 | 8.213e-01 | 0.313 | 1670 | 0%:6, 25%:5, 50%:1, 75%:3, 100%:1 |
| 10 | 81.90 | 1.213 | 7.379e-01 | 0.359 | 1871 | 0%:5, 25%:6, 50%:1, 75%:1, 100%:3 |
| 11 | 83.77 | 1.245 | 6.317e-01 | 0.406 | 2208 | 0%:5, 25%:4, 50%:2, 75%:2, 100%:3 |
| 12 | 85.15 | 1.587 | 5.725e-01 | 0.453 | 2502 | 0%:5, 25%:2, 50%:3, 75%:3, 100%:3 |
| 13 | 88.21 | 2.418 | 2.514e-01 | 0.500 | 2777 | 0%:4, 25%:3, 50%:1, 75%:5, 100%:3 |
| 14 | 89.53 | 1.285 | 3.031e-01 | 0.547 | 2943 | 0%:3, 25%:2, 50%:3, 75%:5, 100%:3 |
| 15 | 90.56 | 1.160 | 1.827e-01 | 0.594 | 3212 | 0%:3, 25%:1, 50%:3, 75%:5, 100%:4 |
| 16 | 94.77 | 1.490 | 3.535e-02 | 0.641 | 3532 | 0%:2, 25%:2, 50%:1, 75%:7, 100%:4 |
| 17 | 95.66 | 1.586 | 2.915e-02 | 0.672 | 3795 | 0%:2, 25%:1, 50%:2, 75%:6, 100%:5 |
| 18 | 98.57 | 1.581 | 2.134e-03 | 0.703 | 3985 | 0%:1, 25%:2, 50%:1, 75%:7, 100%:5 |
| 19 | 99.02 | 1.652 | 6.931e-03 | 0.719 | 4049 | 0%:1, 25%:1, 50%:2, 75%:7, 100%:5 |
| 20 | 99.13 | 1.564 | 2.225e-04 | 0.734 | 4175 | 0%:1, 25%:1, 50%:2, 75%:6, 100%:6 |
| 21 | 99.32 | 1.570 | 5.535e-06 | 0.750 | 4239 | 0%:1, 25%:1, 50%:1, 75%:7, 100%:6 |

## Interpretation

This validates the proposed proportional idea: the adaptive variable is now a release fraction, and the error decreases through many smoother steps instead of a few coarse whole-region jumps.

However, this Berea crop still needs a high active-DOF ratio to reach `1e-5`. The reason is that this prototype still releases Cartesian velocity nodes, not local normal/tangential components. A stronger next version should build local interface frames and activate only the relevant normal/tangential components at each selected interface point.

Main error figure:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix\outputs\adaptive_stokes_berea_16_r16_proportional_interface_tol1e5\paper_velocity_pressure_error_isosurfaces\berea_velocity_pressure_log_error_isosurfaces.png`

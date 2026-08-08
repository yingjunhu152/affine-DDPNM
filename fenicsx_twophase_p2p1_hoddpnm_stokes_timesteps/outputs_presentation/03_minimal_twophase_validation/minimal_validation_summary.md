# Minimal Two-Phase Validation

These cube-minus-sphere cases exercise the Corey graph-transport timestep, inlet boundary-flux injection, natural outlet outflow, CFL diagnostics, mass diagnostics, and full-FEM/HODDPNM Stokes comparison.

| case | holes | cpa | steps | final mean Sw | Sw range | max rel mass err | max CFL | active Schur dofs | max velocity err | max pressure err |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| coarse | 8 | 3 | 20 | 0.2880 | [0.200, 0.886] | 1.42e-16 | 0.062 | 7 (0.64%) | 4.88e-07 | 7.35e-08 |
| medium | 27 | 4 | 20 | 0.2852 | [0.200, 1.000] | 1.74e-04 | 0.111 | 31 (1.35%) | 7.12e-09 | 5.37e-09 |

Note: inlet saturation is imposed through boundary flux, not by resetting boundary vertices. This is still the graph-edge Corey transport validation path; the stricter cell-wise finite-volume face-flux transport remains the next model upgrade.

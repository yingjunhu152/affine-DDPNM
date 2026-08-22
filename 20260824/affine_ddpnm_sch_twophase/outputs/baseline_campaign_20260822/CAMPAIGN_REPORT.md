# Non-Korteweg baseline campaign report

All requested time-step, POD and production runs completed with finite fields and converged solvers.

## Main production result (dt=0.5, t=6)

| Geometry | Arm | phi L2 vs FEM | velocity L2 vs FEM | flux rel. error | online/s | offline/s |
|---|---|---:|---:|---:|---:|---:|
| random27 | FEM-frozen | 0 | 0 | 0 | 42.51 | 0.61 |
| random27 | FEM-SFI | 0 | 0 | 0 | 405.12 | 0.61 |
| random27 | Classic-frozen | 0.0375508 | 0.587548 | 0.750546 | 28.07 | 32.87 |
| random27 | Classic-SFI | 0.0382376 | 0.594784 | 0.758132 | 56.77 | 32.87 |
| random27 | Affine-frozen | 0.00207548 | 0.0555337 | 0.0281583 | 28.38 | 50.88 |
| random27 | Affine-SFI | 0.00213399 | 0.0558631 | 0.0284186 | 57.73 | 50.88 |
| bentheimer | FEM-frozen | 0 | 0 | 0 | 80.59 | 1.59 |
| bentheimer | FEM-SFI | 0 | 0 | 0 | 852.75 | 1.59 |
| bentheimer | Classic-frozen | 0.00658596 | 0.787149 | 1.20862 | 40.64 | 46.85 |
| bentheimer | Classic-SFI | 0.00664392 | 0.788368 | 1.21048 | 80.07 | 46.85 |
| bentheimer | Affine-frozen | 0.00135487 | 0.187416 | 0.111598 | 40.94 | 72.62 |
| bentheimer | Affine-SFI | 0.00136864 | 0.187595 | 0.111903 | 85.92 | 72.62 |

## Time-step diagnosis

- random27 final min(phi): dt=1: -1.346140, dt=0.5: -1.413214, dt=0.25: -1.453105.
- bentheimer final min(phi): dt=1: -1.328044, dt=0.5: -1.371019, dt=0.25: -1.397307.

The undershoot grows rather than disappears under time-step refinement while mass and free energy approach stable values. It is therefore attributed primarily to the continuous-P1 Galerkin advection of a strong inlet jump, not to an overly large time step.

## POD diagnosis

All tolerances 1e-6, 1e-8 and 1e-10 retain 720/747 directions and give identical velocity error, outlet flux and residual. The default 1e-8 lies inside a clear threshold plateau.

## Time refinement of final fields (dt=1 versus dt=0.5)

| Geometry | Arm | phi relative difference | velocity relative difference |
|---|---|---:|---:|
| random27 | FEM-frozen | 0.00385449 | 0 |
| random27 | FEM-SFI | 0.00385735 | 0.000108137 |
| random27 | Classic-frozen | 0.00448878 | 0 |
| random27 | Classic-SFI | 0.00449991 | 0.000104607 |
| random27 | Affine-frozen | 0.00388038 | 0 |
| random27 | Affine-SFI | 0.00388339 | 0.000108328 |
| bentheimer | FEM-frozen | 0.00359116 | 0 |
| bentheimer | FEM-SFI | 0.00359391 | 0.000165006 |
| bentheimer | Classic-frozen | 0.00368083 | 0 |
| bentheimer | Classic-SFI | 0.00368396 | 0.000178338 |
| bentheimer | Affine-frozen | 0.00360758 | 0 |
| bentheimer | Affine-SFI | 0.00361043 | 0.000166242 |

## Scope limitation

This campaign intentionally excludes Korteweg capillary forcing. It is a viscosity-coupled Stokes--Cahn--Hilliard baseline, not full Model H. No clipping was applied to hide phase-field undershoot.

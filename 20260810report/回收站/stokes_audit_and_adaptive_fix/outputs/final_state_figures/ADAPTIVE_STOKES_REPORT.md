# Adaptive FEniCSx Taylor-Hood Stokes Report

This run uses only the FEniCSx-assembled Taylor-Hood P2-P1 Stokes matrix.
Adaptive levels choose restricted Stokes subspaces; no graph pressure equation is used.

- Crop: `20:36,150:166,30:46`
- Pore voxels: `1304`
- Regions: `16`
- Mixed dofs: `42532`
- FEM direct reference time: `9.417497 s`
- FEM direct working / peak working memory: `228.988 MiB` / `990.176 MiB`
- Final adaptive restricted Stokes time: `0.218713 s`
- Final adaptive working / peak working memory: `432.078 MiB` / `990.176 MiB`
- Adaptive strategy: `staged-proportional-hoddpnm`
- Error metric / tolerance: `posterior-defect` / `1.000000e-10`
- Doerfler theta / max upgrades per cycle: `0.850` / `4`
- Active DOF cap: `0.995`
- Final active DOF ratio, including known fixed Dirichlet DOFs: `0.668273`
- Final active free DOF ratio, after known-fixed elimination: `0.152426`
- Known fixed DOFs removed from the restricted solve: `21940`
- Final target error: `5.742777e-06`
- Converged to tolerance: `False`
- Final velocity rel L2 error: `5.045072e+00`
- Final pressure rel L2 error: `1.882445e-06`
- Final method counts: PNM=8, DDPNM=8, DDPNMT=0, HODDPNM=0
- Schur iterations / operator residual: `35` / `5.349076e-13`
- Pressure DOF policy: all pressure DOFs are kept active, so this run demonstrates the restricted Stokes adaptive framework rather than strong compression efficiency.
- Iteration artifacts saved: `False`
- Mean interface fraction: `0.000000`
- Released interface nodes: `0` / `5253`
- Released interface node ratio: `0.000000`

## Final-State Figures

- `final_region_methods`: `final_region_methods.png`
- `final_velocity_magnitude`: `final_velocity_magnitude.png`
- `final_pressure`: `final_pressure.png`
- `final_velocity_log_error`: `final_velocity_log_error.png`
- `final_pressure_log_error`: `final_pressure_log_error.png`
- `final_state_overview`: `final_state_overview.png`

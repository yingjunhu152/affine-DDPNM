# Adaptive FEniCSx Taylor-Hood Stokes Report

This run uses only the FEniCSx-assembled Taylor-Hood P2-P1 Stokes matrix.
Adaptive levels choose restricted Stokes subspaces; no graph pressure equation is used.

- Crop: `20:36,150:166,30:46`
- Pore voxels: `1304`
- Regions: `16`
- Mixed dofs: `42532`
- FEM direct reference time: `7.742588 s`
- FEM direct working / peak working memory: `229.266 MiB` / `990.785 MiB`
- Final adaptive restricted Stokes time: `0.525144 s`
- Final adaptive working / peak working memory: `427.645 MiB` / `990.785 MiB`
- Adaptive strategy: `tolerance-driven`
- Error metric / tolerance: `estimated-residual` / `1.000000e-05`
- Doerfler theta / max upgrades per cycle: `0.650` / `3`
- Active DOF cap: `0.800`
- Final active DOF ratio, including known fixed Dirichlet DOFs: `0.688728`
- Final active free DOF ratio, after known-fixed elimination: `0.172882`
- Known fixed DOFs removed from the restricted solve: `21940`
- Final target error: `8.397385e-06`
- Converged to tolerance: `True`
- Final velocity rel L2 error: `5.798901e+00`
- Final pressure rel L2 error: `6.861394e-06`
- Final method counts: PNM=12, DDPNM=4, DDPNMT=0, HODDPNM=0
- Schur iterations / operator residual: `133` / `1.255012e-12`
- Pressure DOF policy: all pressure DOFs are kept active, so this run demonstrates the restricted Stokes adaptive framework rather than strong compression efficiency.
- Iteration artifacts saved: `False`

## Final-State Figures

- `final_region_methods`: `final_region_methods.png`
- `final_velocity_magnitude`: `final_velocity_magnitude.png`
- `final_pressure`: `final_pressure.png`
- `final_velocity_log_error`: `final_velocity_log_error.png`
- `final_pressure_log_error`: `final_pressure_log_error.png`
- `final_state_overview`: `final_state_overview.png`

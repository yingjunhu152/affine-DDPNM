# Pressure-Compressed HODDPNM Stokes Validation Summary

All rows are same-discrete-system FEniCSx Taylor-Hood P2-P1 solver-equivalence checks.
The HODDPNM solve uses matrix-free GMRES Schur by default; dense exact Schur is not the formal reporting path.

| case | pore voxels | mixed dofs | active boundary dofs | eliminated interior dofs | pressure boundary dofs | pressure eliminated dofs | GMRES iterations | preconditioner | Schur residual | velocity L2 rel | pressure L2 rel | converged |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| real_3d_berea_16_r6_exact_schur | 1304 | 42532 | 3976 | 16616 | 274 | 1830 | 4 | exact_schur_lu_validation | 1.392608e-15 | 1.292042e-09 | 1.384249e-14 | True |
| aggressive_t025_none_scaled | 1304 | 42532 | 405 | 20187 | 0 | 2104 | 27 | none | 7.927448e-11 | 2.343730e-11 | 1.306183e-14 | True |
| t0p30_p075_min50_none_scaled | 1304 | 42532 | 679 | 19913 | 274 | 1830 | 701 | none | 8.676480e-03 | 2.245559e-04 | 5.308397e-12 | True |
| t0p30_p0p30_min100_none_scaled | 1304 | 42532 | 505 | 20087 | 100 | 2004 | 1842 | none | 2.061767e-06 | 4.513603e-07 | 1.284536e-14 | True |
| t0p30_p0p30_min200_none_scaled | 1304 | 42532 | 605 | 19987 | 200 | 1904 | 1804 | none | 4.916628e-03 | 2.726494e-04 | 5.980517e-12 | True |
| t0p30_p0p30_min50_none_scaled | 1304 | 42532 | 455 | 20137 | 50 | 2054 | 153 | none | 2.104560e-06 | 6.311402e-07 | 7.215890e-15 | True |
| t0p30_p0p30_min55_none_scaled | 1304 | 42532 | 460 | 20132 | 55 | 2049 | 237 | none | 1.516878e-06 | 5.424037e-07 | 2.355556e-14 | True |
| t0p30_p0p30_min60_none_scaled | 1304 | 42532 | 465 | 20127 | 60 | 2044 | 395 | none | 1.690594e-06 | 5.708683e-07 | 1.421541e-14 | True |
| t0p35_p075_min50_none_scaled | 1304 | 42532 | 679 | 19913 | 274 | 1830 | 701 | none | 8.676480e-03 | 2.245559e-04 | 5.308397e-12 | True |
| t0p35_p0p35_min50_none_scaled | 1304 | 42532 | 455 | 20137 | 50 | 2054 | 153 | none | 2.104560e-06 | 6.311402e-07 | 7.215890e-15 | True |
| t0p40_p075_min50_none_scaled | 1304 | 42532 | 679 | 19913 | 274 | 1830 | 701 | none | 8.676480e-03 | 2.245559e-04 | 5.308397e-12 | True |
| t0p40_p0p40_min50_none_scaled | 1304 | 42532 | 455 | 20137 | 50 | 2054 | 153 | none | 2.104560e-06 | 6.311402e-07 | 7.215890e-15 | True |
| t0p45_p075_min50_none_scaled | 1304 | 42532 | 679 | 19913 | 274 | 1830 | 701 | none | 8.676480e-03 | 2.245559e-04 | 5.308397e-12 | True |
| t0p45_p0p45_min50_none_scaled | 1304 | 42532 | 455 | 20137 | 50 | 2054 | 153 | none | 2.104560e-06 | 6.311402e-07 | 7.215890e-15 | True |

Use `velocity_l2_rel`, `pressure_l2_rel`, and `schur_residual` as the main quantitative evidence.
The table does not claim analytic or high-resolution physical true error.

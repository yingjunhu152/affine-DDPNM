# Balanced Efficiency Sweep Summary

Sweep target: find a pressure-anchored matrix-free Schur case with nonzero pressure boundary DOFs, useful accuracy, and HODDPNM time below the direct FEM solve.

- ideal target met: no
- closest pressure-anchored faster case: `02_efficiency_trials/balanced_sweep/t0p30_p0p30_min55_none_scaled`
- metrics: active dofs 460, pressure boundary dofs 55, GMRES iterations 237, FEM 8.172 s, HODDPNM 5.249 s, speedup 1.557x, velocity L2 rel 5.424e-07

| case | v-thick | p-thick | min-p | active | p-boundary | iter | FEM s | HODD s | speedup | vel L2 rel | perm-like rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t0p25_p0p25_min50_none_scaled | 0.250 | 0.250 | 50 | 455 | 50 | 153 | 8.192 | 3.397 | 2.412 | 6.311e-07 | 3.618e-08 |
| t0p30_p075_min50_none_scaled | 0.300 | 0.750 | 50 | 679 | 274 | 701 | 9.134 | 13.778 | 0.663 | 2.246e-04 | 3.938e-06 |
| t0p30_p0p30_min100_none_scaled | 0.300 | 0.300 | 100 | 505 | 100 | 1842 | 7.948 | 29.274 | 0.271 | 4.514e-07 | 1.919e-09 |
| t0p30_p0p30_min200_none_scaled | 0.300 | 0.300 | 200 | 605 | 200 | 1804 | 7.893 | 31.523 | 0.250 | 2.726e-04 | 2.725e-06 |
| t0p30_p0p30_min50_none_scaled | 0.300 | 0.300 | 50 | 455 | 50 | 153 | 7.933 | 3.467 | 2.288 | 6.311e-07 | 3.618e-08 |
| t0p30_p0p30_min55_none_scaled | 0.300 | 0.300 | 55 | 460 | 55 | 237 | 8.172 | 5.249 | 1.557 | 5.424e-07 | 6.143e-08 |
| t0p30_p0p30_min60_none_scaled | 0.300 | 0.300 | 60 | 465 | 60 | 395 | 7.958 | 7.206 | 1.104 | 5.709e-07 | 5.384e-08 |
| t0p35_p075_min50_none_scaled | 0.350 | 0.750 | 50 | 679 | 274 | 701 | 8.491 | 22.812 | 0.372 | 2.246e-04 | 3.938e-06 |
| t0p35_p0p35_min50_none_scaled | 0.350 | 0.350 | 50 | 455 | 50 | 153 | 8.109 | 3.715 | 2.183 | 6.311e-07 | 3.618e-08 |
| t0p40_p075_min50_none_scaled | 0.400 | 0.750 | 50 | 679 | 274 | 701 | 8.184 | 11.187 | 0.732 | 2.246e-04 | 3.938e-06 |
| t0p40_p0p40_min50_none_scaled | 0.400 | 0.400 | 50 | 455 | 50 | 153 | 7.876 | 3.437 | 2.292 | 6.311e-07 | 3.618e-08 |
| t0p45_p075_min50_none_scaled | 0.450 | 0.750 | 50 | 679 | 274 | 701 | 7.932 | 11.152 | 0.711 | 2.246e-04 | 3.938e-06 |
| t0p45_p0p45_min50_none_scaled | 0.450 | 0.450 | 50 | 455 | 50 | 153 | 7.838 | 3.863 | 2.029 | 6.311e-07 | 3.618e-08 |

Conclusion: this sweep did not find the full ideal middle point. The 55-pressure-anchor run is faster than FEM and has nonzero pressure boundary DOFs, but its velocity L2 relative error is about 5e-7 and GMRES needs 237 iterations. Larger pressure interfaces or 100-200 pressure anchors increase the active Schur size and iteration count enough to erase the speed benefit. This supports adding a scalable Schur preconditioner before making a strong efficiency claim.

# Affine-complete uniform-point DDPNM(T): same-mesh results

## Controlled experiment

- Geometry: unit cube containing the uniform 3x3x3 array of 27 spheres.
- Partition: 64 non-overlapping subdomains and 144 strict saddle interfaces.
- Common mesh: 12,203 tetrahedra.
- FEM reference: monolithic Taylor--Hood solve with 63,955 mixed unknowns on
  exactly the same mesh.
- Uniform controls: 4--6 vertices per face, mean 5, selected by saddle-seeded
  geodesic farthest-point sampling.
- Only the uniform extension constraints changed.  DDPNM, DDPNMT and both
  cross spaces are numerically identical to the previous experiment.

## Global errors

| Method | Interface dofs | velocity L2 | velocity broken-H1 | raw pressure L2 | outlet flux |
|---|---:|---:|---:|---:|---:|
| DDPNM | 144 | 21.084% | 44.829% | 3.208% | 16.463% |
| DDPNMT | 432 | 19.957% | 43.326% | 3.250% | 14.849% |
| Cross-DDPNM | 864 | 20.528% | 44.367% | 3.156% | 16.040% |
| Cross-DDPNMT | 2,592 | 19.327% | 43.311% | 3.708% | 13.374% |
| Affine-Uniform-DDPNM | 720 | 7.807% | 23.360% | **1.809%** | 2.215% |
| Affine-Uniform-DDPNMT | 2,160 | **5.466%** | **21.590%** | 2.313% | **0.023%** |

## Maximum local errors and conservation

| Method | max cell RMS velocity error | max slice pointwise velocity error | max local mass residual |
|---|---:|---:|---:|
| DDPNM | 2.574e-3 | 2.702e-3 | 2.073e-16 |
| DDPNMT | 2.561e-3 | 2.684e-3 | 2.511e-16 |
| Cross-DDPNM | 2.566e-3 | 2.608e-3 | 1.849e-16 |
| Cross-DDPNMT | 2.612e-3 | 2.664e-3 | 2.119e-16 |
| Affine-Uniform-DDPNM | 1.011e-3 | 1.103e-3 | 1.822e-16 |
| Affine-Uniform-DDPNMT | **6.394e-4** | **8.242e-4** | **1.593e-16** |

## Effect of affine completeness

Relative to the previous constant-only uniform extension, with exactly the
same point locations and global unknown counts:

| Quantity | Constant-only Uniform-DDPNMT | Affine-complete | change |
|---|---:|---:|---:|
| velocity L2 | 19.534% | 5.466% | -72.0% |
| velocity broken-H1 | 42.687% | 21.590% | -49.4% |
| raw pressure L2 | 3.353% | 2.313% | -31.0% |
| outlet flux | 13.797% | 0.023% | -99.8% |
| max cell RMS velocity | 2.584e-3 | 6.394e-4 | -75.3% |
| max slice velocity | 2.869e-3 | 8.242e-4 | -71.3% |

The maximum normal affine reproduction residual is `6.99e-15`; the maximum
nine-mode vector residual is `5.14e-14`; the maximum cardinal residual is
`1.44e-15`.  Local mass conservation remains at roundoff level.

This controlled comparison confirms that the earlier failure was primarily
a basis-completeness problem, not a failure of the O(h^-1) point count or of
the quasi-uniform point distribution.  Sparse values plus constant-only
minimum-energy extension suppressed important smooth face gradients.  Exact
reproduction of `{1,s,t}` restores those gradients and yields a large error
reduction without adding any interface unknowns.

The remaining 5.47% velocity L2 and 21.59% broken-H1 errors motivate the next
stage: residual-driven online interface modes targeted at the unresolved
full-face velocity jump, rather than further geometry-only point enrichment.

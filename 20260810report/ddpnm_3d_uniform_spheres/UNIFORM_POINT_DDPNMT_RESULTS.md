# Uniform interface point experiment

## Setup

- Geometry: unit cube with 27 uniformly arranged spherical obstacles.
- Partition: 64 non-overlapping maximal-ball subdomains and 144 strict
  saddle-plane interfaces.
- Mesh: one common body-fitted mesh with 12,203 tetrahedra.
- FEM reference: monolithic Taylor--Hood solve on exactly the same mesh,
  63,955 mixed unknowns.
- Uniform sampling: `N_s=ceil(sqrt(N_f))`, seeded at the saddle-nearest
  interface vertex, followed by geodesic farthest-point sampling.
- All six reduced methods reuse exactly the same full nodal local response
  library and local Stokes factorisations.

## Strict same-mesh errors

| Method | Interface dofs | velocity L2 | velocity broken-H1 | raw pressure L2 | outlet flux |
|---|---:|---:|---:|---:|---:|
| DDPNM | 144 | 21.084% | 44.829% | 3.208% | 16.463% |
| DDPNMT | 432 | 19.957% | 43.326% | 3.250% | 14.849% |
| Cross-DDPNM | 864 | 20.528% | 44.367% | 3.156% | 16.040% |
| Cross-DDPNMT | 2,592 | **19.327%** | 43.311% | 3.708% | **13.374%** |
| Uniform-DDPNM | 720 | 20.779% | 44.214% | **3.131%** | 15.966% |
| Uniform-DDPNMT | 2,160 | 19.534% | **42.687%** | 3.353% | 13.797% |

## Interpretation

The proposed uniform points are not a universal replacement for the cross,
but they are a more balanced reduced interface set.

- The uniform spaces use 16.7% fewer interface unknowns than their cross
  counterparts (720 versus 864, and 2,160 versus 2,592).
- Uniform-DDPNM is slightly worse than Cross-DDPNM in velocity L2 by 0.251
  percentage points, but is better in broken-H1 by 0.154 points, pressure by
  0.024 points, and flux by 0.074 points.
- Uniform-DDPNMT is slightly worse than Cross-DDPNMT in velocity L2 by 0.207
  points and flux by 0.423 points, but improves broken-H1 by 0.624 points and
  raw pressure by 0.355 points.
- Relative to DDPNMT, Uniform-DDPNMT improves velocity L2 by 0.422 points,
  broken-H1 by 0.639 points, and flux by 1.052 points, at five times the
  interface unknown count; its pressure error increases by 0.103 points.

The strongest outcome is therefore not a dramatic velocity-L2 reduction.
It is that quasi-uniform surface coverage gives the best broken-H1 result and
much better pressure behaviour than the cross-vector space while using fewer
unknowns.  The cross remains best for the tested global velocity-L2 and
outlet-flux objectives.

## Algebraic audit

- Uniform points per face: minimum 4, mean 5, maximum 6.
- Sum of full face vertices: 2,952; cross points: 864; uniform points: 720.
- Maximum cardinal extension residual: `1.52e-15`.
- Maximum constant-vector reproduction residual: `3.82e-15`.
- Maximum local mass residual among the uniform methods: `1.18e-16`.
- Global Schur relative residuals: `8.42e-16` (normal) and `1.33e-15`
  (three-component).

These residuals show that the comparison is not contaminated by failed
conservation constraints or inaccurate Schur solves.

## Maximum local errors

The velocity maxima below are absolute errors.  The first is the maximum of
the cellwise RMS velocity error over all 12,203 tetrahedra.  The second is the
maximum pointwise velocity error on the broken, cell-sided slice `z=0.48`.
No averaging is applied across subdomain interfaces.

| Method | max cell RMS velocity error | max slice pointwise velocity error | max local mass residual |
|---|---:|---:|---:|
| DDPNM | 2.574e-3 | 2.702e-3 | 2.073e-16 |
| DDPNMT | **2.561e-3** | 2.684e-3 | 2.511e-16 |
| Cross-DDPNM | 2.566e-3 | **2.608e-3** | 1.849e-16 |
| Cross-DDPNMT | 2.612e-3 | 2.664e-3 | 2.119e-16 |
| Uniform-DDPNM | 2.621e-3 | 2.726e-3 | **1.167e-16** |
| Uniform-DDPNMT | 2.584e-3 | 2.869e-3 | 1.176e-16 |

The maximum local velocity error is therefore not monotone with the number
of interface unknowns.  Uniform-DDPNMT improves the global velocity L2 and
broken-H1 norms relative to DDPNMT, but its slice maximum is larger.  The
enrichment redistributes the error without controlling the worst local
hotspot.  The four-panel fields show that these hotspots are concentrated
near sphere walls, narrow passages, and broken subdomain traces.  In
contrast, every local mass residual is at roundoff level, so local mass
conservation is not responsible for the velocity peaks.

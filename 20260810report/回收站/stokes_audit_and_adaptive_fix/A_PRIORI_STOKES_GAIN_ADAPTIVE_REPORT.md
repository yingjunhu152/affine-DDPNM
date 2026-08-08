# A Priori Stokes-Gain Adaptive Prototype

This run implements the requested Stokes-based a priori adaptive logic. It uses
the same global FEniCSx Taylor-Hood P2-P1 Stokes matrix as the FEM reference.
The adaptive selector does not use a FEM true solution. `--reference-solve after`
is used only after adaptation to measure validation error.

## Local Stokes Spectral Indicator

For each region, local velocity trace DOFs are placed in `Gamma`. Local interior
velocity DOFs and local pressure DOFs are placed in `I`. The local operator is
the mixed Taylor-Hood Stokes saddle-point block

```text
K_II = [ A_II   B_I^T ]
       [ B_I   -C_i  ]
```

The code forms a local Stokes Schur/Steklov operator

```text
S_i = K_GammaGamma - K_GammaI K_II^{-1} K_IGamma
```

and computes spectral features from its eigenvalues:

- `steklov_tail_ratio`
- `steklov_effective_rank_ratio`
- `steklov_condition_proxy`
- `stokes_infsup_risk`

This is not the old velocity-block proxy. Pressure DOFs are included in the
local Schur elimination.

## A Priori Error Estimates

Each region gets method-specific prior errors:

```text
R_i(PNM)
R_i(DDPNMT)
R_i(HODDPNM50)
R_i(HODDPNM100)
```

They are stored in `geometry_region_prior.csv` as:

- `prior_error_PNM`
- `prior_error_DDPNMT`
- `prior_error_HODDPNM50`
- `prior_error_HODDPNM100`

The prototype uses

```text
R_i(method) = C_i * tail(method)
```

where

```text
C_i = normalize(G_i * (1 + spectral_weight * S_i))
```

and `G_i` is the geometry proxy for

```text
c1 / r_min_i^2
+ c2 / delta_min_i^2
+ c3 kappa_i^2
+ c4 tau_i
+ c5 N_conn_i
+ c6 A_Gamma_i^pore
```

## Three Geometry Cases

The code classifies every region:

- `A_regular_open`: target stage `0`, PNM
- `B_interface_dominated`: target stage `4`, HODDPNM50
- `C_bottleneck_or_stokes_unstable`: target stage `6`, HODDPNM100

The upgrade rule is

```text
Gain_i = [R_i(current) - R_i(next)] / added_DOF_i
```

Regions below their A/B/C target stage receive a priority multiplier, then the
largest gains are upgraded, with at most 3 region upgrades per cycle and the
active DOF cap enforced.

## Cap95 Result

Output directory:

```text
outputs\adaptive_stokes_berea_16_r16_apriori_stokes_casegain_cap95
```

Main result:

- Final cycle: `27`
- Final active DOFs: `40309 / 42532`
- Active ratio: `0.947733`
- Final restricted solve time: `5.359051 s`
- Cumulative restricted solve time: `52.204112 s`
- FEM-after validation time: `10.328292 s`
- Final stages: PNM `2`, DDPNM `0`, DDPNMT `0`, HODDPNM `14`
- Validation velocity relative L2 error: `2.202754e+00`
- Validation pressure relative L2 error: `2.525127e-01`

## Interpretation

The requested Stokes/TH/infsup-aware structure is now present in the code, but
the present constants are not yet reliable enough. The scheme classified two
regions as regular-open A regions and left them at PNM; the validation error
shows that this is too aggressive.

This is a useful prototype but not yet a publishable estimator. The next fix is
to make the geometry constants less permissive, or to compute stronger local
constants such as a real trace mass generalized eigenproblem and a sharper local
inf-sup constant.

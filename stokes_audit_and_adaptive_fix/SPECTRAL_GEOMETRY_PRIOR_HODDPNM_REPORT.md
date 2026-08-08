# Spectral-Geometry Staged HODDPNM Prototype

This version adds a local Schur/Steklov spectral proxy to the geometry-prior
adaptive selector. FEM is not used for adaptive marking. If `--reference-solve
after` is enabled, FEM is computed only after the adaptive run for validation.

## Region Selection

Each region starts with a geometry score

```text
G_i = c1 / r_min_i^2
    + c2 / delta_min_i^2
    + c3 kappa_i^2
    + c4 tau_i
    + c5 N_conn_i
    + c6 A_Gamma_i^pore
```

The current implementation uses voxel proxies for this formula:

- `solid_surface_density`
- `interface_density`
- `neighbor_degree`
- `bbox_void_deficit`
- `constriction_risk`
- `x_section_risk`
- `x_interface_cut_fraction`

Then it computes a local velocity Schur/Steklov proxy. For each region:

1. choose local trace candidate nodes, preferring inter-region interface nodes;
2. exclude fixed boundary nodes from the spectral trace set;
3. sample at most `--spectral-boundary-max-nodes` high-risk trace nodes;
4. build a local velocity Schur complement

```text
S_Gamma = A_bb - A_bi A_ii^{-1} A_ib
```

5. compute eigenvalue-based features:

- `steklov_tail_ratio`
- `steklov_effective_rank_ratio`
- `steklov_condition_proxy`

The final prior complexity is

```text
C_i = normalize( G_i * (1 + spectral_weight * spectral_complexity_i) )
```

At adaptive cycle `n`, the upgrade indicator is

```text
eta_i = C_i * tail(stage_i)
```

with stage tail

```text
PNM: 1.0
DDPNM: 0.62
DDPNMT: 0.42
HODDPNM-25%: 0.28
HODDPNM-50%: 0.16
HODDPNM-75%: 0.07
HODDPNM-100%: 0.015
```

Regions are sorted by `eta_i^2`; Dorfler marking with `theta=0.65` selects the
dominant contributors, with at most 3 region upgrades per cycle. Each marked
region advances only one stage:

```text
PNM -> DDPNM -> DDPNMT -> HODDPNM-25% -> HODDPNM-50% -> HODDPNM-75% -> HODDPNM-100%
```

## Cap95 Validation Run

Command:

```powershell
.\run_spectral_geometry_prior_staged_berea.ps1
```

Output directory:

```text
outputs\adaptive_stokes_berea_16_r16_spectral_geometry_cap95
```

Key results:

- Final cycle: `30`
- Final active DOFs: `40351 / 42532` (`94.872%`)
- Final restricted solve time: `1.190970 s`
- Cumulative adaptive restricted solve time: `19.327021 s`
- FEM-after validation time: `7.495881 s`
- Final region counts: PNM `0`, DDPNM `0`, DDPNMT `2`, HODDPNM `14`
- Validation velocity relative L2 error: `5.724802e-01`
- Validation pressure relative L2 error: `3.473808e-08`

Compared with pure HODDPNM100:

- pure HODDPNM100 active DOFs: `42532 / 42532`
- pure HODDPNM100 restricted solve time: `1.498426 s`
- pure HODDPNM100 velocity relative L2 error: `9.739472e-13`
- pure HODDPNM100 pressure relative L2 error: `1.787850e-15`

## Current Interpretation

The spectral-geometry prior now has the right algorithmic shape: region ranking
is no longer purely hand-built geometry, but includes a local interface Schur
spectrum. It also gives a faster final restricted solve than pure HODDPNM100 in
the cap95 run.

However, this prototype is not yet accurate enough. The velocity error remains
too large. The likely reason is that this version uses only the velocity block
as a Steklov proxy, not the full local Stokes saddle-point operator with pressure
and an interface mass matrix. The next mathematical implementation should use a
generalized local Stokes interface eigenproblem rather than this velocity-only
Schur proxy.

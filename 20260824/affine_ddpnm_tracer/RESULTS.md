# RESULTS — Affine-DDPNM driven tracer on the random-27 medium

Run: `2026-08-11`, `outputs/benchmark_tracer/` (self-contained rerun log:
`run_tracer.log`).

Geometry: unit cube minus the frozen 27-sphere packing (seed-20260804,
18 clipped wall spheres + 9 interior spheres); 15249 tetrahedra, 3483
vertices, 27 pore regions, 114 Voronoi throat interfaces.

Stokes: Taylor-Hood P2-P1, pressure-driven (inlet=1, outlet=0), no-slip
walls, 75954 mixed dofs.
Tracer: P1 advection-diffusion, inlet step c=1 at x=0, natural outlet/wall
flux, implicit Euler + SUPG (factor 0.5), dt=0.1, t=30, diffusivity=0.05,
porosity=1, conservative bounded limiter.

## 1. Stokes (exact FE-integral metrics, identical convention to the
   8/10 W1n benchmark — all three methods reproduce the archived numbers)

| method | unknowns | modes/face | Stokes time (s) | velocity rel L2 | broken H1 | pressure aligned rel L2 | outlet flux rel err |
|---|---:|---:|---:|---:|---:|---:|---:|
| Monolithic-FEM | 75954 | monolithic | 106.21 | 0 | 0 | 0 | 0 |
| Classic-DDPNM-1 | 114 | 1 | 7.76 | 65.32% | 85.85% | 21.96% | 7.272e-01 |
| NormalLinear-DDPNM-3 (W1n) | 342 | 3 | 8.80 | 30.75% | 38.97% | 6.94% | 1.465e-01 |
| Affine-DDPNM-9 | 1026 | 9 | 12.37 | 6.51% | 17.56% | 4.36% | 2.706e-02 |

Cross-check: identical mesh + libraries ⇒ identical archived numbers
(65.32 / 30.75 / 6.51 %, fluxes 0.7272 / 0.1465 / 0.0271).

## 2. Tracer vs FEM-driven reference

| method | breakthrough rel L2 | concentration rel L2 | final mass rel err | max balance residual | t90 | t90 err |
|---|---:|---:|---:|---:|---:|---:|
| FEM | 0 | 0 | 0 | 4.1e-13 | 19.629 | 0 |
| Classic-DDPNM-1 | 6.332e-03 | 1.302e-03 | 1.182e-03 | 4.0e-13 | 19.294 | -0.335 |
| NormalLinear-DDPNM-3 | 2.978e-03 | 6.024e-04 | 5.465e-04 | 2.7e-13 | 19.473 | -0.156 |
| Affine-DDPNM-9 | 5.065e-04 | 1.026e-04 | 9.326e-05 | 3.9e-13 | 19.603 | -0.026 |

Tracer wall time ≈ 0.4 s per method (200–300 implicit steps on 3483 P1
dofs).

## 3. Interpretation

- **Propagation is monotone and near-linear.**  Velocity rel L2
  65.32% → 30.75% → 6.51% maps to concentration-field rel L2
  1.30e-3 → 6.02e-4 → 1.03e-4 (a constant transfer factor ≈ 2e-3 per unit
  velocity error).  Breakthrough and final-mass errors track the same
  factor.
- **Flux conservation is by construction.**  The Schur system *is* the
  per-interface flux-balance equations (9 moment equations per interface
  for Affine); with divergence-free local solves the global inlet=outlet
  balance follows, so every tracer mass-balance residual sits at
  roundoff (~4e-13) for all methods including Classic.  The reported
  outlet-flux error is a *model* error of the restricted traction space,
  not a conservation violation.
- **Affine-9 is the only space reaching transport-level accuracy here**:
  concentration error 1e-4 (below the discretization error of the tracer
  itself), breakthrough t90 off by 0.026 time units; Classic arrives
  0.34 time units early with a 0.6% mass bias.

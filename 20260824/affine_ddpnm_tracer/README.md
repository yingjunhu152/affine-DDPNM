# Affine-DDPNM Driven Single-Phase Tracer Validation (random-27 medium)

Combines the two archived projects:

- **geometry + Stokes DDPNM**: `affine_ddpnm_3d_random_porous` (random-27
  sphere medium, Voronoi throat faces, nine-mode `AffineFaceBasis`);
- **tracer transport**: `stokes_tracer_hoddpnm` (回收站) — transient P1
  advection--diffusion with inlet step, SUPG, and a conservative bounded
  limiter.

The question answered: how does the velocity error of each DDPNM interface
space (Classic-1 / W1n-3 / Affine-9) propagate into a downstream transport
quantity (breakthrough curve, final concentration field, mass balance)?

## Pipeline

1. Build the random-27 partition mesh (frozen `SPHERES`, gmsh OCC fragment,
   Voronoi throat faces) — `random_porous.py`.
2. Solve the monolithic Taylor-Hood P2-P1 Stokes reference — `ddpnm_core`.
3. Solve Classic-DDPNM-1 / NormalLinear-DDPNM-3 / Affine-DDPNM-9 on the
   same mesh — `ddpnm_core` library + `affine_face_basis.py`.
4. Every Stokes velocity is projected to P1 vertex values and drives the
   same tracer model — `tracer_transport.py`.  DDPNM fields use the
   two-sided interface average of the per-pore local solutions.
5. All tracer metrics are reported against the FEM-driven reference:
   breakthrough curve L2, final concentration field L2 (mass-weighted),
   final mass error, algebraic mass-balance residual, t10/t50/t90 crossing
   times, limiter diagnostics.

## Run

```powershell
cd D:\hu\tongjiproj\20260727\20260824\affine_ddpnm_tracer
conda run -n fenicsx --no-capture-output python -u run_affine_ddpnm_tracer.py
```

Outputs are written to `outputs\benchmark_tracer\`:

- `TRACER_VALIDATION_REPORT.md`, `affine_ddpnm_tracer_report.json`
- `tracer_metrics.csv`, `mass_balance_history.csv`
- `breakthrough_curves.png`, `mass_balance_validation.png`,
  `tracer_error_summary.png`, `final_concentration_and_error.png`
- `slice_error_fields.png` (2x3 slice-plane error fields at z = 0.5:
  velocity error and tracer concentration error per method, archived
  `plot_random_errors.py` style — per-subdomain grid resampling, masked
  smoothing, sphere cut-outs, Voronoi interface traces)
- `tracer_velocity_fields.npz` (vertex velocity + final concentration per
  method), per-method VTU files, `random_sphere_partition.msh`

Reproduce the plots from saved outputs (no benchmark rerun):

```powershell
conda run -n fenicsx --no-capture-output python -u plot_slice_errors.py
```

Note: `tracer_error_summary.png` deliberately excludes the mass-balance
series (machine precision ~1e-13 for every method), which flattened the
O(1e-5..1e-2) error differences on a shared log axis; mass conservation is
shown in `mass_balance_validation.png` instead.

## Structure

| File / folder | Origin | Role |
|---|---|---|
| `run_affine_ddpnm_tracer.py` | new | main benchmark script |
| `tracer_transport.py` | ported from `stokes_tracer_hoddpnm` | tracer solver, limiter, plots, CSV |
| `random_porous.py` | copied from `affine_ddpnm_3d_random_porous` | random-27 geometry + partition |
| `affine_face_basis.py` | copied from `affine_ddpnm_3d_random_porous` | Classic / W1n / Affine interface bases |
| `ddpnm_core/` | copied from `20260727/ddpnm_core` | local Stokes operators, response library, Schur assembly, FEM reference |
| `ddpnm3d/` | copied from `ddpnm_3d_uniform_spheres` | 3-D solver API (DdpnmSolution) and basis helpers |
| `postprocess/` | copied from `20260727/postprocess` | field reconstruction (P1 vertex projection, two-sided average) |

## Notes

- Stokes metrics use the exact FE-integral broken-domain metric of the
  archived W1n benchmark (`ddpnm_core.validation.finite_element_error_analysis`:
  per-pore P2-P1 integrals, global-mean aligned pressure).  On this mesh all
  three methods reproduce the archived numbers bit-for-bit.
- The DDPNM local Stokes solves are pressure-ambiguous up to a constant per
  pore; the reported pressure error is mean-aligned.  Velocity and fluxes
  are unaffected (the primitive flux matrices annihilate the
  constant-pressure kernel), and interface flux balance is enforced by the
  Schur system by construction (machine precision).
- No adaptive method levels are used; each method is a fixed modal interface
  space (1 / 3 / 9 unknowns per interface).

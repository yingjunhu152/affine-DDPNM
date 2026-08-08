# FENICSX Stokes Audit Report

Audit target: every flow solve used in the Stokes paper line must be a FEniCSx Taylor-Hood `P2-P1` Stokes solve or a restricted/substructured solve of the same assembled Stokes matrix. Graph pressure/tracer prototypes must not be presented as Stokes results.

## Corrected Adaptive Stokes Case

Folder:

`D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix`

Main script:

`stokes_adaptive_taylor_hood.py`

Status:

- Uses FEniCSx Taylor-Hood `P2-P1` Stokes assembly.
- Uses the same assembled Stokes matrix for all adaptive levels.
- No graph pressure equation is used.
- Outputs velocity and pressure error against FEM direct Stokes reference.

Result:

- Crop: `20:36,150:166,30:46`
- Regions: `16`
- Mixed dofs: `42532`
- FEM direct reference time: `28.0864768 s`
- Final adaptive restricted Stokes time: `30.9379754 s`
- Final method counts: `PNM=0, DDPNM=9, DDPNMT=0, HODDPNM=7`
- Final velocity relative L2 error: `1.666384236e-13`
- Final pressure relative L2 error: `8.443833889e-15`

Important limitation:

This corrected adaptive implementation is equation-correct but not yet performance-good. The current restricted-subspace design reaches the full Taylor-Hood space once all regions are upgraded to `DDPNM`, so the final error drops to machine precision and the restricted solve can be slower than the direct reference. It is a valid Stokes adaptive prototype, not yet the final efficient algorithm.

## Project Audit

| Folder | Status | Notes |
| --- | --- | --- |
| `fenicsx_classic_ddpnm` | Pass for Stokes line | Contains FEniCSx/local Stokes components. Keep checking individual demos before citation. |
| `fenicsx_irregular_hoddpnm` | Pass for Stokes validation scripts | Contains Taylor-Hood validation scripts and plotting helpers. |
| `fenicsx_random_sphere_holes_pnm_stokes` | Pass for Stokes validation scripts | Contains Taylor-Hood random sphere Stokes validation and preconditioned Schur runs. |
| `fenicsx_real_porous_hoddpnm_berea_comparison` | Pass for fixed HODDPNM Stokes | `real_porous_hoddpnm_validation.py` is FEniCSx Taylor-Hood Stokes. `real_berea_tracer_hoddpnm.py` is graph tracer and is not a Stokes validation. |
| `adaptive_multifidelity_berea_pnm_ddpnm_hoddpnm` | Fail for Stokes line | Deprecated. Graph pressure/tracer prototype, no velocity error. |
| `adaptive_realistic_berea_16_multiregion` | Fail for Stokes line | Deprecated. Graph pressure/tracer prototype, no velocity error. |
| `fenicsx_singlephase_tracer_hoddpnm` | Fail for Stokes line | Graph tracer prototype. Can be reused only after carrier velocity comes from FEniCSx Taylor-Hood Stokes. |
| `fenicsx_twophase_hoddpnm` | Fail for Stokes line | Graph two-phase prototype. Not a continuous Stokes/FEniCSx Taylor-Hood solve. |
| `_inspect_fenicsx_real_porous_hoddpnm_berea_comparison` | Ignore | Inspection/copy folder, not a primary project. |

## Actions Taken

- Created corrected adaptive Stokes implementation in `stokes_audit_and_adaptive_fix`.
- Generated velocity/pressure error VTU and isosurface figures for the corrected adaptive Stokes case.
- Added warning banners to graph adaptive/tracer/two-phase README files.
- Kept graph prototypes available, but explicitly excluded them from Stokes validation claims.

## Required Rule Going Forward

For the Stokes paper line, a result is acceptable only if the output summary states:

- `FEniCSx Taylor-Hood P2-P1 Stokes`, or
- `restricted/substructured solve of the same FEniCSx Taylor-Hood Stokes matrix`,

and includes:

- velocity error,
- pressure error,
- FEM direct Stokes reference or clearly defined Stokes reference solve.

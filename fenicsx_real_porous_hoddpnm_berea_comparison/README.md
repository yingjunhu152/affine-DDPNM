# Real Berea P2-P1 Stokes + HODDPNM Validation

This folder is organized as a real 3D Berea porous-medium experiment with two
separate claims:

1. **Correctness validation:** HODDPNM reconstruction matches the same
   discrete Taylor-Hood FEM Stokes system.
2. **Efficiency exploration:** matrix-free Schur solves can be faster after
   aggressive interface compression, but the current pressure-anchored cases do
   not yet justify a strong general efficiency claim.

The script uses the known-fixed Schur partition: Dirichlet velocity DOFs and
the pressure gauge DOF are removed before the active Schur unknowns are formed.

## Output Layout

Current retained outputs are:

```text
outputs/
  01_correctness_validation/
    real_3d_berea_16_r6_exact_schur/
  02_efficiency_trials/
    aggressive_t025_none_scaled/
    balanced_sweep/
    balanced_sweep_summary.csv
    balanced_sweep_summary.md
  03_figures_for_presentation/
    correctness_exact_schur/
    efficiency_aggressive_t025_none_scaled/
    balanced_t0p30_min55_pressure_anchored/
```

Use `01_correctness_validation` to discuss solver equivalence. Use
`02_efficiency_trials` only as an efficiency-potential study.

## Key Results

| case | active Schur DOFs | pressure boundary DOFs | GMRES iter | FEM solve | HODDPNM Schur | velocity L2 rel | pressure L2 rel |
|---|---:|---:|---:|---:|---:|---:|---:|
| correctness exact-Schur | 3976 | 274 | 4 | 8.12 s | 24.72 s | 1.29e-9 | 1.38e-14 |
| aggressive speed trial | 405 | 0 | 27 | 9.02 s | 2.39 s | 2.34e-11 | 1.31e-14 |
| closest pressure-anchored speed trial | 460 | 55 | 237 | 8.17 s | 5.25 s | 5.42e-7 | 2.36e-14 |

Interpretation:

- The exact-Schur case is the main correctness result. It is accurate, but it is
  intentionally not efficient because it factors a dense Schur matrix.
- The aggressive `t=0.25` no-preconditioner case is fast and accurate, but it
  keeps zero pressure boundary DOFs, so present it as an aggressive static
  condensation speed trial.
- The pressure-anchored sweep did not find the ideal middle point. The best
  faster case keeps 55 pressure boundary DOFs, but its velocity L2 error is
  about `5e-7` and GMRES needs 237 iterations. Cases with 100-200 pressure
  anchors or a wider pressure interface lose the speed advantage.

## Physical Diagnostics

`validation_summary.json` now includes a `physical_diagnostics` block with:

- inlet and outlet flux proxy;
- pressure-drop proxy;
- permeability-like proxy;
- continuity row residual;
- velocity magnitude range;
- FEM-vs-HODDPNM relative errors for the proxy quantities.

These are solver-comparison diagnostics, not calibrated laboratory
permeability measurements. The closest pressure-anchored speed trial gives a
permeability-like relative error of `6.14e-8` and HODDPNM continuity row
residual `2.26e-12`.

## New Interface Controls

The main script supports separate velocity and pressure interface selection:

```powershell
--interface-thickness 0.30
--pressure-interface-thickness 0.30
--min-pressure-boundary-dofs 55
```

`--pressure-interface-thickness` defaults to the velocity interface thickness.
`--min-pressure-boundary-dofs` forces the nearest free pressure DOFs to remain
in the active Schur boundary when the geometric pressure interface would be too
small.

## Reproduce

Run the retained correctness and aggressive efficiency cases:

```powershell
.\run_berea_pressure_compressed_validation.ps1
```

Run a single pressure-anchored trial:

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
D:\Miniconda3\Scripts\conda.exe run -n fenicsx --no-capture-output python real_porous_hoddpnm_validation.py `
  --volume-npy data\berea_100_to_300.npz `
  --pore-value 1 `
  --crop 20:36,150:166,30:46 `
  --regions 6 `
  --interface-thickness 0.30 `
  --pressure-interface-thickness 0.30 `
  --min-pressure-boundary-dofs 55 `
  --hoddpnm-solver gmres `
  --schur-preconditioner none `
  --schur-rtol 1e-10 `
  --skip-condition-number `
  --out-dir outputs\02_efficiency_trials\balanced_sweep\t0p30_p0p30_min55_none_scaled
```

Render presentation figures for any case:

```powershell
D:\Miniconda3\Scripts\conda.exe run -n fenicsx --no-capture-output python render_berea_efficiency_isosurfaces.py `
  --case-dir outputs\02_efficiency_trials\balanced_sweep\t0p30_p0p30_min55_none_scaled `
  --out-dir outputs\03_figures_for_presentation\balanced_t0p30_min55_pressure_anchored
```

The complete sweep table is in
`outputs\02_efficiency_trials\balanced_sweep_summary.md` and
`outputs\02_efficiency_trials\balanced_sweep_summary.csv`.

## Positioning For Reporting

Recommended wording:

> On a real 3D Berea pore geometry, the P2-P1 Stokes HODDPNM Schur
> reconstruction is solver-equivalent to the full FEM system.

Efficiency wording should remain cautious:

> Matrix-free Schur compression shows speed potential, including faster
> pressure-anchored trials, but robust efficiency still needs a scalable Schur
> preconditioner for larger pressure-interface sets.

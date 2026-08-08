$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

D:\Miniconda3\Scripts\conda.exe run -n fenicsx --no-capture-output python stokes_adaptive_taylor_hood.py `
  --volume-npy data\berea_100_to_300.npz `
  --pore-value 1 `
  --crop 20:36,150:166,30:46 `
  --regions 16 `
  --cycles 30 `
  --adaptive-strategy a-priori-stokes-gain `
  --reference-solve after `
  --indicator stokes-spectral-geometry-prior `
  --error-metric stokes-spectral-geometry-prior `
  --error-tolerance 0.05 `
  --minimum-stop-stage 4 `
  --max-upgrades-per-cycle 3 `
  --active-dof-cap 0.95 `
  --interface-fractions 0.25,0.5,0.75,1.0 `
  --ddpnmt-interface-fraction 0.10 `
  --geometry-prior-weights 1.0,1.1,0.9,0.65,0.75,0.7,0.55 `
  --geometry-stage-tail 1.0,0.62,0.42,0.28,0.16,0.07,0.015 `
  --spectral-boundary-max-nodes 36 `
  --spectral-tail-modes 12 `
  --spectral-weight 1.25 `
  --restricted-solver schur-gmres `
  --schur-preconditioner ilu `
  --out-dir outputs\adaptive_stokes_berea_16_r16_apriori_stokes_casegain_cap95

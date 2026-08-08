$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

D:\Miniconda3\Scripts\conda.exe run -n fenicsx --no-capture-output python stokes_adaptive_taylor_hood.py `
  --volume-npy data\berea_100_to_300.npz `
  --pore-value 1 `
  --crop 20:36,150:166,30:46 `
  --regions 16 `
  --cycles 0 `
  --adaptive-strategy staged-proportional-hoddpnm `
  --initial-stage 6 `
  --reference-solve after `
  --indicator geometry-prior `
  --error-metric geometry-prior `
  --error-tolerance 0.0 `
  --minimum-stop-stage 6 `
  --active-dof-cap 1.0 `
  --interface-fractions 0.25,0.5,0.75,1.0 `
  --ddpnmt-interface-fraction 0.10 `
  --restricted-solver schur-gmres `
  --schur-preconditioner ilu `
  --out-dir outputs\berea_16_r16_pure_hoddpnm100

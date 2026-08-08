$ErrorActionPreference = "Stop"

$Python = "D:\Miniconda3\Scripts\conda.exe"
$EnvName = "fenicsx"
$Script = "stokes_adaptive_taylor_hood.py"

Set-Location "D:\hu\tongjiproj\FENICSX\stokes_audit_and_adaptive_fix"

& $Python run -n $EnvName --no-capture-output python $Script `
  --volume-npy data\berea_100_to_300.npz `
  --pore-value 1 `
  --crop 20:36,150:166,30:46 `
  --regions 16 `
  --cycles 28 `
  --adaptive-strategy staged-proportional-hoddpnm `
  --reference-solve after `
  --indicator posterior-defect `
  --error-metric posterior-defect `
  --error-tolerance 1e-10 `
  --minimum-stop-stage 6 `
  --dorfler-theta 0.85 `
  --max-upgrades-per-cycle 4 `
  --active-dof-cap 0.995 `
  --interface-fractions 0.25,0.5,0.75,1.0 `
  --ddpnmt-interface-fraction 0.10 `
  --restricted-solver schur-gmres `
  --schur-preconditioner ilu `
  --out-dir outputs\adaptive_stokes_berea_16_r16_posterior_defect_cap995

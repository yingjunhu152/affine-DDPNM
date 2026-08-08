$ErrorActionPreference = "Stop"

$python = "D:\Miniconda3\Scripts\conda.exe"

& $python run -n fenicsx --no-capture-output python stokes_adaptive_taylor_hood.py `
  --volume-npy data\berea_100_to_300.npz `
  --pore-value 1 `
  --crop 20:36,150:166,30:46 `
  --regions 16 `
  --cycles 12 `
  --adaptive-strategy tolerance-driven `
  --reference-solve before `
  --indicator true-error `
  --error-metric velocity `
  --error-tolerance 1e-5 `
  --dorfler-theta 0.65 `
  --max-upgrades-per-cycle 3 `
  --active-dof-cap 0.995 `
  --restricted-solver schur-gmres `
  --schur-preconditioner ilu `
  --out-dir outputs\adaptive_stokes_berea_16_r16_tol1e5_trueerr

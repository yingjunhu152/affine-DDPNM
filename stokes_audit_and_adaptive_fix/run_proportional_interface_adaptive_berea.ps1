$ErrorActionPreference = "Stop"

$python = "D:\Miniconda3\Scripts\conda.exe"

& $python run -n fenicsx --no-capture-output python stokes_adaptive_taylor_hood.py `
  --volume-npy data\berea_100_to_300.npz `
  --pore-value 1 `
  --crop 20:36,150:166,30:46 `
  --regions 16 `
  --cycles 24 `
  --adaptive-strategy proportional-interface `
  --reference-solve before `
  --indicator true-error `
  --error-metric velocity `
  --error-tolerance 1e-5 `
  --dorfler-theta 0.65 `
  --max-upgrades-per-cycle 3 `
  --active-dof-cap 0.995 `
  --interface-fractions 0,0.25,0.5,0.75,1.0 `
  --proportional-node-set interface-plus-interior `
  --restricted-solver schur-gmres `
  --schur-preconditioner ilu `
  --out-dir outputs\adaptive_stokes_berea_16_r16_proportional_interface_tol1e5

$ErrorActionPreference = "Stop"

$Conda = "D:\Miniconda3\Scripts\conda.exe"
$EnvName = "fenicsx"
$Crop = "20:36,150:166,30:46"
$CaseDir = "outputs\adaptive_stokes_berea_16_r16"

& $Conda run -n $EnvName --no-capture-output python stokes_adaptive_taylor_hood.py `
  --volume-npy data\berea_100_to_300.npz `
  --pore-value 1 `
  --crop $Crop `
  --regions 16 `
  --cycles 8 `
  --reference-solve after `
  --upgrade-fraction 0.25 `
  --out-dir $CaseDir

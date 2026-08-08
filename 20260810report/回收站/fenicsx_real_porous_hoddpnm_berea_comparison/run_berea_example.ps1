$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

$python = "D:\Miniconda3\Scripts\conda.exe"
$volume = "data\berea_100_to_300.npz"

if (-not (Test-Path -LiteralPath $volume)) {
    throw "Missing $volume. Download a segmented Berea sandstone volume and save/convert it there. See DATA_SOURCES.md."
}

& $python run -n fenicsx --no-capture-output python real_porous_hoddpnm_validation.py `
    --volume-npy $volume `
    --pore-value 1 `
    --crop 20:28,150:158,30:38 `
    --regions 2 `
    --hoddpnm-solver gmres `
    --schur-preconditioner exact-schur `
    --skip-condition-number `
    --out-dir outputs\validation_stokes_pressure_compressed_8_r2

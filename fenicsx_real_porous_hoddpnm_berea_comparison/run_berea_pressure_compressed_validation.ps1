$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

$Conda = "D:\Miniconda3\Scripts\conda.exe"
$EnvName = "fenicsx"
$Volume = "data\berea_100_to_300.npz"

if (-not (Test-Path -LiteralPath $Volume)) {
    throw "Missing $Volume. See DATA_SOURCES.md."
}

function Run-StokesCase {
    param(
        [string]$CaseDir,
        [string]$Crop,
        [int]$Regions,
        [double]$InterfaceThickness,
        [string]$Preconditioner,
        [string]$ExtraArgs = "",
        [int]$Restart = 80,
        [int]$MaxIter = 400
    )

    $argList = @(
        "run", "-n", $EnvName, "--no-capture-output", "python", "real_porous_hoddpnm_validation.py",
        "--volume-npy", $Volume,
        "--pore-value", "1",
        "--crop", $Crop,
        "--regions", "$Regions",
        "--interface-thickness", "$InterfaceThickness",
        "--hoddpnm-solver", "gmres",
        "--schur-preconditioner", $Preconditioner,
        "--schur-restart", "$Restart",
        "--schur-maxiter", "$MaxIter",
        "--skip-condition-number",
        "--out-dir", $CaseDir
    )
    if ($ExtraArgs.Trim().Length -gt 0) {
        $argList += $ExtraArgs.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
    }
    & $Conda @argList
}

$CorrectnessCase = "outputs\01_correctness_validation\real_3d_berea_16_r6_exact_schur"
$AggressiveCase = "outputs\02_efficiency_trials\aggressive_t025_none_scaled"

Run-StokesCase `
    -CaseDir $CorrectnessCase `
    -Crop "20:36,150:166,30:46" `
    -Regions 6 `
    -InterfaceThickness 1.05 `
    -Preconditioner "exact-schur"

Run-StokesCase `
    -CaseDir $AggressiveCase `
    -Crop "20:36,150:166,30:46" `
    -Regions 6 `
    -InterfaceThickness 0.25 `
    -Preconditioner "none"

& $Conda run -n $EnvName --no-capture-output python render_berea_efficiency_isosurfaces.py `
    --case-dir $CorrectnessCase `
    --out-dir "outputs\03_figures_for_presentation\correctness_exact_schur" `
    --crop "20:36,150:166,30:46"

& $Conda run -n $EnvName --no-capture-output python render_berea_efficiency_isosurfaces.py `
    --case-dir $AggressiveCase `
    --out-dir "outputs\03_figures_for_presentation\efficiency_aggressive_t025_none_scaled" `
    --crop "20:36,150:166,30:46"

foreach ($LegacyPng in @(
    (Join-Path $CorrectnessCase "real_porous_hoddpnm_error.png"),
    (Join-Path $AggressiveCase "real_porous_hoddpnm_error.png")
)) {
    if (Test-Path -LiteralPath $LegacyPng) {
        Remove-Item -LiteralPath $LegacyPng -Force
    }
}

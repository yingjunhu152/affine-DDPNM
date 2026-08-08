$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:XDG_CACHE_HOME = Join-Path $ProjectDir ".cache"

$Conda = "D:\Miniconda3\Scripts\conda.exe"
$PlotPython = "D:\Miniconda3\envs\econ\python.exe"
if (-not (Test-Path -LiteralPath $Conda)) {
    throw "Required conda executable not found: $Conda"
}
if (-not (Test-Path -LiteralPath $PlotPython)) {
    throw "Required plotting Python not found: $PlotPython"
}

$OutputDir = Join-Path $ProjectDir "outputs\default"
for ($ArgumentIndex = 0; $ArgumentIndex -lt $args.Count; $ArgumentIndex++) {
    if ($args[$ArgumentIndex] -eq "--out-dir" -and $ArgumentIndex + 1 -lt $args.Count) {
        $RequestedOutput = $args[$ArgumentIndex + 1]
        if ([System.IO.Path]::IsPathRooted($RequestedOutput)) {
            $OutputDir = $RequestedOutput
        }
        else {
            $OutputDir = [System.IO.Path]::GetFullPath((Join-Path $ProjectDir $RequestedOutput))
        }
    }
    elseif ($args[$ArgumentIndex] -like "--out-dir=*") {
        $RequestedOutput = $args[$ArgumentIndex].Substring("--out-dir=".Length)
        if ([System.IO.Path]::IsPathRooted($RequestedOutput)) {
            $OutputDir = $RequestedOutput
        }
        else {
            $OutputDir = [System.IO.Path]::GetFullPath((Join-Path $ProjectDir $RequestedOutput))
        }
    }
}

& $Conda run -n fenicsx --no-capture-output `
    python run_ddpnm_3d.py @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PlotPython plot_results.py `
    --input (Join-Path $OutputDir "ddpnm_3d_results.npz") `
    --out-dir $OutputDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PlotPython verify_results.py --out-dir $OutputDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

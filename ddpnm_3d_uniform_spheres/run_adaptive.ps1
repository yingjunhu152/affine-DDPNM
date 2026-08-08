$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:XDG_CACHE_HOME = Join-Path $ProjectDir ".cache"

$Conda = "D:\Miniconda3\Scripts\conda.exe"
$PlotPython = "D:\Miniconda3\envs\econ\python.exe"
$OutputDir = Join-Path $ProjectDir "outputs\adaptive_hierarchy"

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
}

& $Conda run -n fenicsx --no-capture-output python run_adaptive_ddpnm_3d.py @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PlotPython plot_adaptive_results.py `
    --input (Join-Path $OutputDir "adaptive_3d_results.npz") `
    --report (Join-Path $OutputDir "adaptive_report.json") `
    --out-dir $OutputDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PlotPython verify_adaptive_results.py --out-dir $OutputDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

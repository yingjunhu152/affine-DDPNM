$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:XDG_CACHE_HOME = Join-Path $ProjectDir ".cache"
$Conda = "D:\Miniconda3\Scripts\conda.exe"
$OutputDir = Join-Path $ProjectDir "outputs\strict_p0_comparison"

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

& $Conda run -n fenicsx --no-capture-output python run_strict_error_2d.py @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Conda run -n fenicsx --no-capture-output python verify_strict_error_2d.py --out-dir $OutputDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

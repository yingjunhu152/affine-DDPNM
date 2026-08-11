$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

conda run -n fenicsx --no-capture-output python preflight.py
if ($LASTEXITCODE -ne 0) { throw "FEniCSx benchmark preflight failed." }

conda run -n fenicsx --no-capture-output python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Pure numerical tests failed." }

New-Item -ItemType Directory -Force outputs\benchmark_twophase | Out-Null
conda run -n fenicsx --no-capture-output python -u run_affine_ddpnm_twophase.py `
  --out-dir outputs\benchmark_twophase 2>&1 | Tee-Object outputs\benchmark_twophase\run.log
if ($LASTEXITCODE -ne 0) { throw "Benchmark failed; see outputs\benchmark_twophase\run.log." }

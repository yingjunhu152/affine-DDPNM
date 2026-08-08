$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
& "D:\Miniconda3\Scripts\conda.exe" run -n fenicsx --no-capture-output `
    python run_ddpnm_2d.py @args

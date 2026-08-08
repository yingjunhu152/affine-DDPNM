$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

& "D:\Miniconda3\Scripts\conda.exe" run -n fenicsx --no-capture-output python run_berea_tracer_comparison.py `
  --pore-value 1 `
  --crop 20:36,150:166,30:46 `
  --regions 16 `
  --interface-thickness 1.05 `
  --pressure-interface-thickness 2.0 `
  --diffusivity 0.05 `
  --dt 0.20 `
  --t-final 60.0 `
  --supg `
  --methods FEM,HODDPNM `
  --out-dir outputs\singlephase_tracer_hoddpnm_interface_pressure_t60_supg

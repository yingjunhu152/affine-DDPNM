$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$Conda = "D:\Miniconda3\Scripts\conda.exe"
$Script = "real_porous_hoddpnm_validation.py"
$BaseArgs = @(
  "run", "-n", "fenicsx", "--no-capture-output", "python", $Script,
  "--volume-npy", "data\berea_100_to_300.npz",
  "--pore-value", "1",
  "--crop", "20:36,150:166,30:46",
  "--regions", "6",
  "--hoddpnm-solver", "gmres",
  "--schur-preconditioner", "none",
  "--schur-rtol", "1e-10",
  "--skip-condition-number",
  "--skip-visualization"
)

$Cases = @(
  @{ Name = "t0p30_p0p30_min50_none_scaled"; Interface = "0.30"; Pressure = "0.30"; MinPressure = "50" },
  @{ Name = "t0p35_p0p35_min50_none_scaled"; Interface = "0.35"; Pressure = "0.35"; MinPressure = "50" },
  @{ Name = "t0p40_p0p40_min50_none_scaled"; Interface = "0.40"; Pressure = "0.40"; MinPressure = "50" },
  @{ Name = "t0p45_p0p45_min50_none_scaled"; Interface = "0.45"; Pressure = "0.45"; MinPressure = "50" },
  @{ Name = "t0p30_p0p30_min55_none_scaled"; Interface = "0.30"; Pressure = "0.30"; MinPressure = "55" },
  @{ Name = "t0p30_p0p30_min60_none_scaled"; Interface = "0.30"; Pressure = "0.30"; MinPressure = "60" },
  @{ Name = "t0p30_p0p30_min100_none_scaled"; Interface = "0.30"; Pressure = "0.30"; MinPressure = "100" },
  @{ Name = "t0p30_p0p30_min200_none_scaled"; Interface = "0.30"; Pressure = "0.30"; MinPressure = "200" },
  @{ Name = "t0p30_p075_min50_none_scaled"; Interface = "0.30"; Pressure = "0.75"; MinPressure = "50" },
  @{ Name = "t0p35_p075_min50_none_scaled"; Interface = "0.35"; Pressure = "0.75"; MinPressure = "50" },
  @{ Name = "t0p40_p075_min50_none_scaled"; Interface = "0.40"; Pressure = "0.75"; MinPressure = "50" },
  @{ Name = "t0p45_p075_min50_none_scaled"; Interface = "0.45"; Pressure = "0.75"; MinPressure = "50" }
)

foreach ($Case in $Cases) {
  $OutDir = "outputs\02_efficiency_trials\balanced_sweep\$($Case.Name)"
  & $Conda @BaseArgs `
    "--interface-thickness" $Case.Interface `
    "--pressure-interface-thickness" $Case.Pressure `
    "--min-pressure-boundary-dofs" $Case.MinPressure `
    "--out-dir" $OutDir
}

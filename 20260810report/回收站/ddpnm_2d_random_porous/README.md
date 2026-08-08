# 2D DD-PNM reproduction on the fixed random porous medium

This directory implements the two-dimensional domain-decomposition pore-network
method described in Section 2.4 and Algorithm 2.1 of `DDPNM.pdf`. It uses the exact
17-circle porous geometry generated earlier with random seed `20260802`.
The original four-panel construction figure is preserved under `assets/` for comparison.

## Adaptive DDPNM / DDPNMT / HODDPNM hierarchy

`run_adaptive_ddpnm_2d.py` adds the three nested physical interface spaces used
by the DDPNMT/HODDPNM reference implementation:

| level | method | unknowns on one internal interface |
|---:|---|---|
| 0 | DDPNM | one constant normal pressure/traction coefficient |
| 1 | DDPNMT | the DDPNM coefficient plus one constant tangential traction coefficient |
| 2 | HODDPNM | P1 nodal normal and tangential traction coefficients |

The former `interface_order=1` option in the baseline program is a linearly varying
**normal** traction and is deliberately not called DDPNMT. The new DDPNMT load is a
true vector traction in the local tangential direction.

Every local Taylor--Hood matrix is factored once. Its P1 vector-traction response
library produces the Steklov/Schur interface operator for any mixture of levels 0,
1 and 2. In addition, the adaptive driver implements the real-3D folder's explicit
full-FE-trace correctness Schur complement

```text
S = A_GG - A_GI A_II^{-1} A_IG,
g = b_G  - A_GI A_II^{-1} b_I.
```

That full P2--P1 trace solve must agree with the monolithic solve to roundoff. It is
a correctness baseline, not an efficiency claim.

The adaptive state starts with every interface at DDPNM. It compares against the
DDPNMT enriched solution, marks by a hierarchical field-difference indicator with
Dörfler marking, and promotes each selected interface by one level. The process is
then repeated against HODDPNM. Levels only increase; they never decrease. A difference
larger than the tolerance therefore triggers promotion. Keeping DDPNM in that case
would leave the same discrepancy unchanged and stall the algorithm.

Default adaptive controls are:

```text
target hierarchical tolerance = 1e-2 (1 percent)
Dörfler theta                 = 0.65
maximum interfaces per cycle = 3
maximum cycles per phase     = 30
```

Run the adaptive experiment with:

```powershell
.\run_adaptive.ps1
```

For example, change the hierarchical tolerance with
`.\run_adaptive.ps1 --target-tolerance 0.005`.

Important adaptive outputs are:

- `adaptive_algorithm_box.png` and `ADAPTIVE_ALGORITHM.md`;
- `adaptive_convergence.png`;
- `adaptive_final_hierarchy.png`;
- `method_errors_to_exact_schur.png`;
- `method_velocity_error_fields.png`;
- `adaptive_history.csv` and `adaptive_report.json`.

## What is implemented

1. Gmsh meshes the unit square after subtracting the fixed circular particles.
2. Delaunay-neighbouring particles define candidate pore throats. For every valid
   pair, the code constructs the exact center-line gap segment between the two circle
   walls; its equal-clearance point is the analytic throat saddle.
3. These segments are embedded into the CAD model before meshing. Consequently every
   subdomain boundary is a conforming, straight saddle cross-section rather than a
   staircase assembled from pre-existing mesh facets.
4. Every subdomain uses a Taylor-Hood P2-P1 discretization of steady Stokes flow.
   Geometry-based size fields refine the mesh near every circular wall and still more
   strongly near every throat cross-section.
5. The resulting local traction-to-flux matrices are assembled into the global
   interface Schur system. Every internal interface has two traction unknowns: a
   constant coefficient and a coefficient multiplying the normalized linear coordinate
   along the interface.
6. Local velocity and pressure fields are reconstructed from the stored unit responses.
7. A monolithic Taylor-Hood solution on the uncut pore space provides a reference.
   The program reports interface mass residuals, DtN symmetry diagnostics, Schur
   eigenvalues, and vertex-sampled field errors.

Both the zeroth flux moment (net flux) and the first flux moment are balanced across
every internal interface. The refinement is prescribed from geometry; it is local but
is not a posteriori solution-adaptive refinement.

## Run

From PowerShell:

```powershell
cd D:\hu\tongjiproj\20260727\ddpnm_2d_random_porous
.\run.ps1
```

Useful options:

```powershell
.\run.ps1 --mesh-size 0.05 --wall-size 0.025 --throat-size 0.014 --out-dir outputs\coarse
.\run.ps1 --mesh-size 0.03 --wall-size 0.014 --throat-size 0.008 --out-dir outputs\refined
```

The script uses `D:\Miniconda3\envs\fenicsx`, which already contains FEniCSx,
Gmsh, SciPy, NumPy and Matplotlib.

Verify the default run's conservation, SPD, DtN symmetry and reference-error bounds:

```powershell
D:\Miniconda3\envs\fenicsx\python.exe verify_results.py
```

## Main outputs

- `00_analytic_refined_discrete_mesh.png`: full mesh plus a throat-level zoom showing
  P1 geometry/pressure nodes and P2 velocity edge nodes.
- `01_geometry_and_subdomains.png`: analytic saddle cuts and subdomain labels.
- `02_ddpnm_reconstructed_fields.png`: reconstructed DD-PNM velocity and pressure.
- `03_reference_comparison.png`: monolithic/reference comparison and error fields.
- `04_interface_system_diagnostics.png`: Schur sparsity and flux-balance residuals.
- `05_paper_style_error_fields.png`: paper-style FE fields and point-wise DD-PNM errors.
- `interface_results.csv`: one row per internal interface.
- `report.json`: reproducible parameters and numerical diagnostics.
- `ddpnm_2d_fields.xdmf`: ParaView-readable reconstructed and reference fields.
- `ddpnm_2d_fields.npz`: mesh and fields for Python post-processing.

Regenerate the paper-style error figure and audit mesh/interface resolution with:

```powershell
D:\Miniconda3\envs\fenicsx\python.exe plot_paper_style_errors.py
.\run.ps1  # only needed if the default fields have not yet been generated
D:\Miniconda3\Scripts\conda.exe run -n fenicsx --no-capture-output python audit_mesh.py
```

## Interpretation

This is the paper-style DD-PNM, not merely static condensation of a monolithic
matrix: each local operator is built by independent Stokes solves and the global
unknowns are constant-plus-linear normal tractions on pore interfaces. The very small
pressure mass term controlled by `--pressure-stabilization` is a numerical gauge
regularization; the default is `1e-10`, matching the existing FEniCSx prototypes in
the parent project.

# Real Berea Taylor-Hood HODDPNM Validation

This report separates raw DOF-vector diagnostics from FEM integral solver-equivalence metrics.

## Scope

solver-equivalence check against the same discrete FEniCSx Taylor-Hood system; not an analytic or high-resolution true-error study

## Preferred Metrics

| metric | value | meaning |
|---|---:|---|
| gmres_iterations | 1.842000e+03 | GMRES callback iterations for the matrix-free Schur solve |
| gmres_relative_residual | 2.061767e-06 | relative residual of the reported Schur solve |
| active_boundary_dofs | 5.050000e+02 | free active Schur boundary unknown count after removing known Dirichlet/gauge dofs |
| known_fixed_dofs | 2.194000e+04 | known Dirichlet/gauge dofs removed before forming the active Schur unknown set |
| eliminated_interior_dofs | 2.008700e+04 | interior unknown count eliminated by local Schur reconstruction |
| velocity_l2_rel | 4.513603e-07 | FEM mass-matrix integral relative L2 error for P2 velocity |
| velocity_h1_seminorm_rel | 5.938514e-07 | FEM stiffness-matrix relative H1 seminorm error for P2 velocity |
| pressure_l2_rel | 1.284536e-14 | FEM mass-matrix integral relative L2 error for pinned-gauge P1 pressure |
| pressure_l2_mean_aligned_rel | 1.124705e-14 | FEM mass-matrix pressure L2 error after removing the mass-weighted mean pressure shift |
| pressure_eliminated_dofs | 2.004000e+03 | P1 pressure unknown count eliminated inside subregions |
| inlet_flux_proxy_rel_error | 0.000000e+00 | relative difference between FEM and HODDPNM inlet flux proxy |
| pressure_drop_proxy_rel_error | 3.007275e-15 | relative difference between FEM and HODDPNM pressure-drop proxy |
| permeability_like_rel_error | 1.918732e-09 | relative difference in proxy effective-permeability quantity Q L / (A delta-p) |
| hoddpnm_continuity_row_residual_l2 | 5.931357e-10 | L2 norm of mixed-system pressure-row residual for HODDPNM solution |

## Notes

- Solver converged: `True`
- Solver type: `matrix_free_gmres_schur_known_fixed`
- Pressure gauge: pressure is compared in the pinned-gauge system imposed by the FEM boundary condition; mean-aligned pressure diagnostics are also reported
- Schur partition: known Dirichlet/gauge dofs are removed first; only free pressure dofs near decomposition interfaces are kept in the active Schur boundary, while subregion-interior pressure dofs are eliminated
- The exact-Schur preconditioner is a dense validation aid; these timings and memory numbers should not be used to claim HODDPNM speed or memory savings.
- Legacy DOF-vector errors remain in `validation_summary.json` only for continuity with older runs.
- The VTU/PNG error fields are visualization outputs; use `validation_metrics.csv` for quantitative reporting.

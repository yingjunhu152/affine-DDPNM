# Real Berea Taylor-Hood HODDPNM Validation

This report separates raw DOF-vector diagnostics from FEM integral solver-equivalence metrics.

## Scope

solver-equivalence check against the same discrete FEniCSx Taylor-Hood system; not an analytic or high-resolution true-error study

## Preferred Metrics

| metric | value | meaning |
|---|---:|---|
| gmres_iterations | 2.700000e+01 | GMRES callback iterations for the matrix-free Schur solve |
| gmres_relative_residual | 7.927448e-11 | relative residual of the reported Schur solve |
| active_boundary_dofs | 4.050000e+02 | free active Schur boundary unknown count after removing known Dirichlet/gauge dofs |
| known_fixed_dofs | 2.194000e+04 | known Dirichlet/gauge dofs removed before forming the active Schur unknown set |
| eliminated_interior_dofs | 2.018700e+04 | interior unknown count eliminated by local Schur reconstruction |
| velocity_l2_rel | 2.343730e-11 | FEM mass-matrix integral relative L2 error for P2 velocity |
| velocity_h1_seminorm_rel | 3.966056e-11 | FEM stiffness-matrix relative H1 seminorm error for P2 velocity |
| pressure_l2_rel | 1.306183e-14 | FEM mass-matrix integral relative L2 error for pinned-gauge P1 pressure |
| pressure_l2_mean_aligned_rel | 1.309732e-14 | FEM mass-matrix pressure L2 error after removing the mass-weighted mean pressure shift |
| pressure_eliminated_dofs | 2.104000e+03 | P1 pressure unknown count eliminated inside subregions |
| inlet_flux_proxy_rel_error | 0.000000e+00 | relative difference between FEM and HODDPNM inlet flux proxy |
| pressure_drop_proxy_rel_error | 3.759093e-16 | relative difference between FEM and HODDPNM pressure-drop proxy |
| permeability_like_rel_error | 1.200248e-12 | relative difference in proxy effective-permeability quantity Q L / (A delta-p) |
| hoddpnm_continuity_row_residual_l2 | 3.241399e-12 | L2 norm of mixed-system pressure-row residual for HODDPNM solution |

## Notes

- Solver converged: `True`
- Solver type: `matrix_free_gmres_schur_known_fixed`
- Pressure gauge: pressure is compared in the pinned-gauge system imposed by the FEM boundary condition; mean-aligned pressure diagnostics are also reported
- Schur partition: known Dirichlet/gauge dofs are removed first; only free pressure dofs near decomposition interfaces are kept in the active Schur boundary, while subregion-interior pressure dofs are eliminated
- The exact-Schur preconditioner is a dense validation aid; these timings and memory numbers should not be used to claim HODDPNM speed or memory savings.
- Legacy DOF-vector errors remain in `validation_summary.json` only for continuity with older runs.
- The VTU/PNG error fields are visualization outputs; use `validation_metrics.csv` for quantitative reporting.

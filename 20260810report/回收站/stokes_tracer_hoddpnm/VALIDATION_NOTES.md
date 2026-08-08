# Stokes-Tracer Validation Notes

This folder is now scoped to single-phase tracer validation only.

It is self-contained for the formal run: `real_porous_hoddpnm_validation.py`, the minimal `classic_ddpnm` helper, and `data\berea_100_to_300.npz` live in this folder.

The comparison is:

- `FEM`: direct sparse solve of the FEniCSx Taylor-Hood P2-P1 Stokes system, followed by tracer transport.
- `HODDPNM`: known-fixed Schur-complement/static-condensation solve of the same FEniCSx Stokes matrix. Dirichlet dofs are removed as known values before Schur assembly; the active solve retains only free interface velocity dofs and free interface P1 pressure dofs, followed by tracer transport.

No adaptive method levels, PNM, DDPNM, DDPNMT, HODDPNM50, or HODDPNM100 cases are used in the formal run. The pressure interface thickness is separate from the velocity interface thickness; this keeps pressure reduced while avoiding an under-resolved pressure Schur boundary.

Key output fields:

- `velocity_rel_l2_fem_integral_vs_fem` and `pressure_rel_l2_fem_integral_vs_fem` validate the HODDPNM Stokes field against FEM.
- `active_dofs` and `active_dof_ratio` are the actual free Schur unknowns after removing known Dirichlet dofs. `hoddpnm_known_fixed_dofs`, `hoddpnm_velocity_interface_dofs`, `hoddpnm_pressure_interface_dofs`, and `hoddpnm_pressure_interior_dofs` document the known-fixed interface-pressure Schur partition.
- The formal comparison uses `--max-active-ratio 0.50` as a guard against regressions to weak compression.
- GMRES is run on a diagonally scaled Schur operator, then checked against the true unscaled Schur residual. Non-converged or falsely scaled-converged runs raise an error instead of feeding a bad Stokes field into the tracer solve.
- The formal pressure stabilization is `1e-6`. This reduces Schur iterations substantially while keeping HODDPNM-vs-FEM Stokes and tracer discrepancies at roundoff scale for the same stabilized system.
- The present result should be described as compressed and accelerated, not as a final scalable solver: O(10^2) Schur iterations are still high for large-scale use, so the next numerical task is a stronger scalable Schur preconditioner.
- `breakthrough_rel_l2_error` validates the outlet tracer curve against the FEM-driven reference.
- `concentration_rel_l2_fem_integral_vs_fem` and `concentration_linf_abs_vs_fem` validate the final spatial tracer field.
- `mass_balance_residual_rate` and `mass_balance_relative_residual` report the algebraic finite-element mass balance at every time step.
- `budget_mass` is the mass reconstructed from the cumulative source-minus-transport budget.
- `raw_final_concentration_*_before_limiter` fields report the unconstrained linear-solve output immediately before the bounded limiter.
- `limiter_*` fields report whether the bounded `[0, 1]` limiter preserved the raw step mass.

The tracer velocity and output fields are explicitly remapped between DOLFINx dof ordering and the mesh vertex ordering before assembly, VTU output, and PyVista plotting.


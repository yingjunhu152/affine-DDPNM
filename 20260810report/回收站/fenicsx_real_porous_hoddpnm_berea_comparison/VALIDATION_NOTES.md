# Real Berea Validation Notes

Old generated outputs and mixed-scope reports were removed.

Use these files from a new run:

- `validation_summary.json`: full machine-readable Stokes validation summary.
- `validation_metrics.csv`: preferred quantitative Stokes metrics.
- `VALIDATION_REPORT.md`: human-readable Stokes validation report.
- `real_porous_hoddpnm_solution.vtu` and `real_porous_hoddpnm_error.png`: visualization outputs only.

Preferred Stokes metrics:

- `velocity_l2_rel`: FEM mass-matrix integral relative L2 error for P2 velocity.
- `velocity_h1_seminorm_rel`: FEM stiffness-matrix relative H1 seminorm error for P2 velocity.
- `pressure_l2_rel`: FEM mass-matrix integral relative L2 error for pinned-gauge P1 pressure.
- `pressure_l2_mean_aligned_rel`: pressure L2 error after removing the mass-weighted mean pressure shift.

Legacy `np.linalg.norm` DOF-vector errors are still written to `validation_summary.json` for continuity, but they should not be used as the main physical error claim.

`prototype_graph_tracer/real_berea_tracer_hoddpnm.py` is a voxel-graph pressure/tracer prototype. Its outputs stay under `prototype_graph_tracer/outputs/` and include pre-clip tracer bounds and mass diagnostics, but it is not a FEniCSx Taylor-Hood P2-P1 Stokes tracer validation.

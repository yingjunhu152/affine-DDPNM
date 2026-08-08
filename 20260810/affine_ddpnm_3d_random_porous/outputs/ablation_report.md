# Four-combination ablation: partition x basis (hand-off 5.3)

- Voronoi: 15249 cells, 27 regions, 114 interfaces
- Watershed: 6586 cells, 82 basins, 369 interfaces
- Monolithic FEM: 75954 dofs (116.6 s) / 38243 dofs (27.5 s)

| metric | VxClassic | VxAffine | WxClassic | WxAffine |
|---|---|---|---|---|
| global_interface_unknowns | 114 | 1026 | 369 | 3321 |
| velocity_relative_l2 | 0.6532 | 0.06506 | 0.4355 | 0.1131 |
| velocity_relative_broken_h1 | 0.8585 | 0.1756 | 0.6396 | 0.3073 |
| pressure_relative_l2 | 0.103 | 0.02028 | 0.06723 | 0.03932 |
| outlet_flux_relative_error | 0.7272 | 0.02706 | 0.4015 | 0.07751 |
| first_solve_seconds | 12.22 | 15.23 | 17.35 | 25.01 |
| speedup_vs_fem_first_solve | 9.547 | 7.66 | 1.586 | 1.1 |
| schur_symmetry_error | 0 | 0 | 0 | 0 |
| max_mass_residual | nan | nan | 2.828e-16 | 8.267e-13 |
| min_schur_eigenvalue | nan | nan | 2.92e-06 | -2.969e-18 |
| inlet_outlet_flux_imbalance | nan | nan | 2.017e-14 | 3.99e-11 |

## Caveats

- **meshes_differ**: The two partitions are meshed differently by construction: the Voronoi mesh embeds the saddle planes as internal cuts (with an interface size field), the watershed mesh has no cuts and no interface refinement.  Same geometry, same sphere/boundary fields.
- **watershed_exact_schur_skipped**: The exact dense FE-trace Schur was not rerun for the watershed partition; correctness was established on the Voronoi partition (monolithic difference ~1e-12).

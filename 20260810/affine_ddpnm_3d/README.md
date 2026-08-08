# Affine-DDPNM 3-D experiment

This directory is deliberately separate from `ddpnm_3d_uniform_spheres`.
The original affine-uniform implementation and its results are not modified.

The benchmark compares, on one identical tetrahedral mesh:

1. Classic DDPNM with one constant-normal coefficient per interface;
2. Affine DDPNM with one interface entity carrying the nine generalized modes
   `{1,s,t} x {n,t1,t2}`;
3. the monolithic Taylor--Hood FEM reference.

No uniform surface samples, point-selection algorithm, or nodal minimum-energy
extension is used in this experiment.

Run in the FEniCSx environment:

```powershell
conda run -n fenicsx --no-capture-output python run_affine_ddpnm_benchmark.py
```

Outputs are written to `outputs/benchmark`.

`affine_face_basis.py` contains the isolated single-entity nine-mode basis.
`plot_affine_ddpnm_comparison.py` creates the paper-style accuracy/cost figure.
See `RESULTS.md` for the measured comparison and timing-scope notes.

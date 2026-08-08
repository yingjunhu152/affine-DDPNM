# Geometry Adaptive vs Pure HODDPNM100

Both runs use the same Berea crop, 16 regions, FEniCSx Taylor-Hood P2-P1 Stokes
matrix, and Schur-GMRES with ILU preconditioning.

FEM is computed only after the run for validation and timing reference.

| case | cycles | active DOFs | active ratio | final solve time | cumulative adaptive solve time | total wall time | FEM-after time | peak working memory | velocity rel L2 | pressure rel L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometry-prior adaptive | 30 | 41998 | 0.987445 | 1.552000 s | 19.564073 s | 35.769554 s | 7.401607 s | 996.512 MiB | 2.349754e-01 | 1.047771e-08 |
| pure HODDPNM100 | 0 | 42532 | 1.000000 | 1.498426 s | 1.498426 s | 12.927675 s | 7.728529 s | 985.512 MiB | 9.739472e-13 | 1.787850e-15 |

## Interpretation

The current geometry-prior adaptive prototype is not faster than pure
HODDPNM100 when the cost is counted honestly over all adaptive iterations.

The final adaptive solve is close to pure HODDPNM100 in size and time:

- adaptive final solve: `1.552000 s`
- pure HODDPNM100 solve: `1.498426 s`

But adaptive spent many intermediate solves:

- adaptive cumulative solve time: `19.564073 s`
- pure HODDPNM100 single solve time: `1.498426 s`

This means the present geometry indicator is only a structural prototype. It
pushes almost every interface DOF open before its geometry risk is small, so it
does not yet deliver a useful accuracy-cost tradeoff.

The next version should replace the hand-built geometry score with a local
interface spectral indicator, so that the method can stop earlier while still
protecting velocity accuracy.

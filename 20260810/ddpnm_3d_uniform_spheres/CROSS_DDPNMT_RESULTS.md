# DDPNMT and cardinal Cross-DDPNMT results

Verification mesh: 12,203 tetrahedra, 64 subdomains and 144 interfaces.  All
four reduced methods and the monolithic Taylor--Hood reference use the same
parent mesh.

| method | interface dofs | velocity L2 | broken-H1 | pressure raw L2 | outlet flux |
|---|---:|---:|---:|---:|---:|
| DDPNM | 144 | 21.084% | 44.829% | 3.208% | 16.463% |
| DDPNMT | 432 | 19.957% | 43.326% | 3.250% | 14.849% |
| Cross-DDPNM | 864 | 20.528% | 44.367% | 3.156% | 16.040% |
| Cross-DDPNMT | 2,592 | **19.327%** | **43.311%** | 3.708% | **13.374%** |

The tangential constants give a larger velocity improvement than normal-only
cross enrichment.  Combining both produces the best velocity and flow-rate
results, while the pressure L2 error increases.  The increase remains after
pressure mean alignment, so it is not a pressure-gauge shift; it is a genuine
pressure reconstruction trade-off of the high-dimensional vector cross
space.

All four systems preserve mass to numerical precision.  For Cross-DDPNMT:

```text
global interface unknowns                   2592
maximum vector cardinal residual            2.53e-15
maximum three-constant reproduction residual 4.18e-15
maximum local mass residual                 2.12e-16
relative global Schur residual              1.58e-15
relative inlet/outlet mass imbalance        2.73e-14
```

Thus the velocity/pressure behavior is a property of the tested interface
space, not a failure of conservation or linear algebra.


# Cardinal cross-normal DDPNM: implementation audit

## Corrected formulation

For every throat face `f`, the cross has `N_C,f` mesh nodes and exactly
`N_C,f` scalar normal-pressure unknowns.  The extension matrix satisfies

```text
R_C,f E_f = I,          E_f 1_C = 1_f.
```

Thus the coefficients are cross-node pressures and the original constant
DDPNM mode is contained exactly in their span.  The global row associated
with cross node `r` is

```text
integral_Gamma_f Phi_f,r
    (u_i dot n_i + u_j dot n_j) dS = 0.
```

## Bugs found in the earlier prototype

1. It used one extra P0 mode plus zero-mean cross modes, so it had
   `N_C,f + 1` coefficients and they were not the requested nodal pressures.
2. It used the line joining adjacent maximal-ball centres as the face normal.
   Near the external boundary the maximal balls have unequal radii, so this
   line is oblique although the throat is a coordinate plane.  The supposed
   normal load therefore contained a tangential component.
3. The second bug caused a maximum local mass residual of about `4.77e-5`
   and an apparent global boundary imbalance of about `15%`.

The corrected implementation uses `-phi * FacetNormal` for every nodal normal
traction.  On the diagnostic mesh it gives

```text
maximum cardinal residual             1.04e-15
maximum constant-reproduction residual 1.07e-15
maximum local Stokes solve residual     7.92e-15
maximum local mass residual             2.00e-16
maximum global interface moment residual 1.14e-18
```

## Same-mesh FEM comparison

Coarse verification mesh: 12,203 tetrahedra, 64 pore subdomains, 144 throat
interfaces.  Both reduced methods and the monolithic Taylor--Hood FEM use the
same parent mesh.

| metric | DDPNM | cardinal cross-normal DDPNM | change |
|---|---:|---:|---:|
| velocity relative L2 | 21.084% | 20.528% | -0.556 percentage points |
| velocity broken-H1 | 44.829% | 44.367% | -0.462 percentage points |
| pressure raw relative L2 | 3.208% | 3.156% | -0.052 percentage points |
| outlet flux relative error | 16.463% | 16.040% | -0.423 percentage points |

The corrected cross method is consistently better than original DDPNM, so
the earlier 33% velocity result was not the result of this intended method.
However, the improvement is small: normal-only cross enrichment does not
recover the two tangential velocity/traction components needed for a strong
3-D interface approximation.


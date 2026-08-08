# Cross-skeleton minimum-energy Stokes DDPNM

The implementation remains a DDPNM method.  The maximal-ball partition,
strict throat saddle sections, local Taylor--Hood Stokes solves, response
library, global interface moment equations and local reconstruction are
unchanged.  Only the interface approximation space is replaced.

```text
Algorithm: Cross-skeleton minimum-energy Stokes DDPNM

Input:
    Conforming tetrahedral partition {Omega_i};
    strict throat faces {Gamma_f} and clearance saddles {x_f^s};
    target interface mesh size h; viscosity mu.

Offline stage:
1. For each throat face Gamma_f:
   1.1 Build the face triangulation and its edge graph.
   1.2 Select the mesh vertex nearest x_f^s as the discrete saddle node.
   1.3 Compute two in-plane principal directions t_f^1 and t_f^2.
   1.4 Route four direction-weighted shortest paths from the saddle node to
       the face boundary along +/-t_f^1 and +/-t_f^2.
   1.5 Let C_f be the union of the four paths.  The number of scalar cross
       nodes is N_C,f = O(h^-1).

2. For every pore Omega_i:
   2.1 Assemble and factorise the local P2-P1 Stokes operator A_i once.
   2.2 On every incident face build the full P1 nodal traction primitives
       B_i,f.  The normal component uses the exact local FEniCSx facet
       normal, never the line joining two maximal-ball centres.
   2.3 Compute the local response and compliance matrices
           R_i,f = A_i^{-1} B_i,f,
           G_i,f = B_i,f^T R_i,f.

3. For every internal face f=(i,j):
   3.1 Extract the normal-normal response blocks and form the two-sided
       traction-response energy
           H_f = G_i,f^{nn} + G_j,f^{nn} + epsilon I.
   3.2 Build the nodal cross restriction R_C,f.
   3.3 Compute the raw minimum-energy extension
           E_f^0 = H_f^{-1} R_C,f^T
                   (R_C,f H_f^{-1} R_C,f^T)^{-1}.
   3.4 Correct E_f^0 inside ker(R_C,f) to obtain E_f satisfying
           R_C,f E_f = I,          E_f 1_C = 1_f.
       The first identity makes every global coefficient exactly one cross
       node pressure.  The second makes the nodal basis a partition of unity,
       so the original DDPNM constant-normal mode belongs exactly to the
       cross space without adding an extra unknown.
   3.5 Embed E_f only in the normal rows of B_i,f.  Do not add tangential
       interface unknowns.  Each face therefore has exactly N_C,f dofs.

Online DDPNM stage:
4. For every internal face Gamma_f=(Omega_i,Omega_j) and every cross basis
   Phi_f,r, impose the generalized normal-flux equation
       integral_Gamma_f Phi_f,r
       (u_i dot n_i + u_j dot n_j) dS = 0.
5. Assemble these equations into the sparse global DDPNM moment system
           S_C lambda_C = b_C
   from the reduced local response matrices.
6. Solve for the cross-node pressure coefficients lambda_C.
7. Reconstruct every independent local Stokes solution and verify:
       cardinal interpolation R_C,f E_f = I,
       constant reproduction E_f 1_C = 1_f,
       local algebraic Stokes residual,
       local total-flux/divergence identity,
       interface moment conservation,
       inlet/outlet flux balance,
       same-mesh FEM L2 and broken-H1 errors.

Output:
    Reconstructed DDPNM velocity and pressure;
    cross geometry and extension matrices;
    exactly N_C,f=O(h^-1) cross-node interface unknowns instead of
    O(h^-2) full-face nodes.
```

Important interpretation: the cross is a control skeleton, not a replacement
for the physical throat face.  Every active basis function still has support
on the complete face after the Stokes-energy extension.

# Algorithm: DDPNMT and cardinal Cross-DDPNMT

```text
Input:
    non-overlapping maximal-ball subdomains {Omega_i};
    strict saddle throat faces {Gamma_f};
    conforming tetrahedral mesh; viscosity mu.

Offline local-response stage:
1. For every face Gamma_f, construct the saddle-centred mesh cross C_f and
   collect its N_C,f nodes.
2. Define an exact orthonormal face frame (n_f,t_f^1,t_f^2).  The normal
   traction is assembled with the local FEniCSx FacetNormal; the two tangent
   directions use equal-and-opposite signs on the adjacent pore domains.
3. For every pore Omega_i, factorise its P2-P1 Stokes matrix A_i once.
4. For every incident face node and every component c in {n,t1,t2}, assemble
   the full nodal traction load B_i,f^(node,c) and response
       U_i,f^(node,c) = A_i^(-1) B_i,f^(node,c).
5. Form the two-sided vector response energy H_f from the adjacent response
   blocks.  Let R_C,f restrict full-face vector values to the cross.
6. Compute the raw cardinal extension
       E_f^0 = H_f^(-1) R_C,f^T
               (R_C,f H_f^(-1) R_C,f^T)^(-1).
7. Let C_C and C_F contain the three constant component fields on the cross
   and full face.  Apply the minimum correction
       E_f = E_f^0 + (C_F - E_f^0 C_C) C_C^+,
   so that
       R_C,f E_f = I,             E_f C_C = C_F.

Four nested comparison spaces:
8. DDPNM:       one constant normal coefficient per face.
9. DDPNMT:      constant coefficients in (n,t1,t2), three per face.
10. Cross-DDPNM: one cardinal normal coefficient per cross node.
11. Cross-DDPNMT: cardinal coefficients in (n,t1,t2), 3 N_C,f per face.

Global DDPNM assembly:
12. For every active face basis Phi_f,r^c, impose the generalized velocity
    continuity equation
       integral_Gamma_f Phi_f,r^c
       (u_i - u_j) dot d_f^c dS = 0,
    where d_f^c is n_f, t_f^1 or t_f^2.
13. Assemble all equations into the condensed interface system
       S lambda = b,
    solve for the interface traction coefficients, and reconstruct every
    independent local Stokes field by linear response superposition.
14. Verify cardinal/constant reproduction, local mass balance, interface
    moment residuals and the same-mesh FEM errors.

Output:
    DDPNM, DDPNMT, Cross-DDPNM and Cross-DDPNMT velocity/pressure fields;
    interface coefficients, conservation diagnostics and FEM errors.
```

The constant normal combination in both cross spaces is exactly the original
DDPNM mode.  The three constant vector combinations in Cross-DDPNMT are
exactly DDPNMT.  Hence the comparisons are nested at the interface-space
level; the two tangential equations improve tangential velocity matching but
do not alter the ordinary total-flux conservation equation.


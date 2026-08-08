# Uniform-point DDPNM / DDPNMT algorithm

The interface reduction remains inside the DDPNM framework.  Only the active
traction space on each two-dimensional throat face is changed.

```text
Algorithm 1  Saddle-seeded geodesic-FPS DDPNM(T)

Input : non-overlapping pore subdomains {Omega_i}; strict saddle interfaces
        {Gamma_f}; common conforming tetrahedral mesh; tolerance parameters.
Output: piecewise Stokes field (u_h, p_h) and interface coefficients lambda.

Offline stage
 1: for every pore Omega_i, assemble and factorise its local Taylor--Hood
    Stokes operator A_i;
 2: for every interface Gamma_f, extract the triangular face graph G_f and
    its N_f mesh vertices;
 3: set N_s(f) = ceil(sqrt(N_f)); choose the vertex nearest the analytic
    throat saddle as x_1;
 4: for k = 2,...,N_s(f), choose
         x_k = arg max_{x in G_f} min_{1 <= j < k} d_Gf(x,x_j),
    using geodesic edge distance and deterministic mesh-index tie breaking;
 5: build the full nodal traction response matrix on both pores adjacent to
    Gamma_f, and form the two-sided Stokes response energy H_f;
 6: construct the cardinal minimum-energy extension E_f by
         E_f = arg min_E trace(E^T H_f E)
         subject to R_f E_f = I and E_f C_s = C_f,
    where R_f restricts a full face field to the selected points.  C_s,C_f
    enforce exact reproduction of the constant normal mode (DDPNM) or all
    three constant vector modes (DDPNMT).

Online stage
 7: for every pore, project its local response matrix to the active columns
    of E_f;
 8: assemble the global DDPNM interface moment/flux-continuity system
         S lambda = b;
 9: solve for the selected-point normal coefficients (Uniform-DDPNM), or
    normal plus two tangential coefficients (Uniform-DDPNMT);
10: reconstruct every local Stokes field by the linear combination of its
    pressure-driven response functions, then join the broken solution.
```

For a quasi-uniform triangular face, `N_f = O(h^-2)` and
`N_s = ceil(sqrt(N_f)) = O(h^-1)`.  Hence the method uses a line-like number
of interface coefficients without prescribing one particular cross curve.


# Affine-complete uniform-point DDPNM(T)

```text
Algorithm 2  Affine-complete geodesic-FPS DDPNM(T)

Input : non-overlapping subdomains {Omega_i}; strict saddle interfaces
        {Gamma_f}; conforming tetrahedral mesh; local Taylor--Hood operators.
Output: broken Stokes field (u_h,p_h) and selected-point traction coefficients.

Offline stage
 1: factorise one local Stokes operator A_i for every subdomain Omega_i;
 2: on each interface Gamma_f with N_f vertices, set
        N_s(f) = ceil(sqrt(N_f));
 3: select the saddle-nearest vertex first, then add vertices by deterministic
    geodesic farthest-point sampling until N_s(f) points have been selected;
 4: form scaled face coordinates (s,t) in the strict saddle plane and the
    scalar affine matrices
        C_f = [1,s,t] on every face vertex,
        C_s = R_f C_f on the selected vertices;
 5: build the two-sided discrete Stokes/Steklov energy H_f from the two
    adjacent local response operators;
 6: compute the cardinal affine-complete minimum-energy extension E_f:
        minimise    trace(E_f^T H_f E_f)
        subject to  R_f E_f = I,
                    E_f C_s = C_f                 (normal version),
    or
                    E_f (C_s tensor I_3)
                    = C_f tensor I_3              (three-component version);
    hence the vector space exactly contains the nine modes
        {1,s,t} tensor {normal,tangent_1,tangent_2}.

Online stage
 7: project each local traction-response matrix through E_f;
 8: assemble the global DDPNM moment/weak-trace-continuity system S lambda=b;
 9: solve for all selected-point coefficients and reconstruct each local
    Stokes field by linear combination of the precomputed responses;
10: optionally evaluate the unresolved full-face velocity jump to drive a
    later residual-based online enrichment stage.
```

The point count remains `O(h^-1)` per face, while exact affine completeness
prevents the sparse cardinal extension from discarding the dominant smooth
surface modes.  Setting `--no-uniform-affine-complete` recovers the previous
constant-only extension for controlled comparisons.


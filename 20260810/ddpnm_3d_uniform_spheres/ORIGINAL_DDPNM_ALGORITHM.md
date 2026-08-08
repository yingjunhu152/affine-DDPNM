┌──────────────────────────────────────────────────────────────────────┐
│ Algorithm 1  Three-dimensional original DDPNM                       │
├──────────────────────────────────────────────────────────────────────┤
│ Input: cube Ω, 27 solid spheres, p_in, p_out, viscosity μ, sizes h. │
│ Output: interface tractions λ_f and reconstructed local fields.      │
├──────────────────────────────────────────────────────────────────────┤
│ 1. Construct the clearance d(x) to all grains and cube boundaries.  │
│ 2. Compute all 4x4x4 maximal empty balls of the bounded domain.      │
│ 3. Connect neighbouring maximal balls; locate each throat saddle.    │
│ 4. Imprint 144 equal-clearance interfaces and fragment Ω={Ω_i}.     │
│ 5. Generate one conforming tetrahedral mesh, refined near sphere     │
│    walls, the outer cube boundary and internal interfaces.           │
│ 6. For each Ω_i, assemble the Taylor–Hood P2–P1 Stokes matrix A_i.  │
│ 7. For each local port f, solve A_i w_if=b_if for a unit constant    │
│    normal traction and measure every port flux to form G_i.          │
│ 8. Assemble S_Γ λ=r_Γ by requiring Σ_i ∫_{Γ_f}u_i·n_i dS=0.         │
│ 9. Solve the one-unknown-per-interface system and reconstruct        │
│    x_i=Σ_f λ_f w_if plus the known inlet/outlet responses.           │
│10. Verify interface flux conservation and local DtN symmetry.       │
└──────────────────────────────────────────────────────────────────────┘

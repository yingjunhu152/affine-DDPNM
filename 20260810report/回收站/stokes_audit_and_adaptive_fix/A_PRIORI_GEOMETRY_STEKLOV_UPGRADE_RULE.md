# A Priori Geometry-Steklov Upgrade Rule

This note records the first mathematically defensible upgrade logic for
PNM, DDPNMT, HODDPNM50, and HODDPNM100 without using a global FEM reference
solution during adaptation.

## Local Stokes Interface Operator

For a subregion `Omega_i`, split local velocity DOFs into interior `I` and
interface trace `Gamma`. Keep local pressure DOFs inside the Stokes block:

```text
K_II = [ A_II   B_I^T ]
       [ B_I   -C_i  ]

K_IG = [ A_IG ]
       [ B_G  ]

K_GI = [ A_GI  B_G^T ]

K_GG = A_GG
```

The local Stokes Steklov/Schur operator is

```text
S_i = K_GG - K_GI K_II^{-1} K_IG
```

and the generalized interface eigenproblem is

```text
S_i phi_k = lambda_k M_Gamma phi_k
```

where `M_Gamma` is an interface mass matrix. This includes pressure and local
inf-sup stability through the saddle-point inverse `K_II^{-1}`.

## Error Form

Let `V_h` be the full Taylor-Hood FEM space and `V_m` be the restricted
PNM/DDPNMT/HODDPNM space. For each region,

```text
||u_h - u_m||_A + ||p_h - p_m||_Q
  <= C_stab,i * E_trace,i(m_i) * B_i
```

where:

- `C_stab,i` depends only on local Stokes stability constants,
  mainly Korn, Poincare, trace extension, and inf-sup constants;
- `E_trace,i(m_i)` is the unresolved interface trace approximation factor;
- `B_i` is the size of local boundary/forcing data.

The adaptive decision does not need the FEM solution. It uses the computable
prior

```text
R_i(method) = C_geo,i * C_stab,i * E_trace,i(method) * B_i
```

and upgrades the region with the largest expected error reduction per added DOF:

```text
gain_i = [R_i(current) - R_i(next)] / [DOF_i(next) - DOF_i(current)]
```

## Method-Specific Priors

Let `P_M` be the projection from the full interface trace space onto the trace
space used by method `M`. Define

```text
E_trace,i(M)^2
  = spectral_radius( (I-P_M)^T S_i (I-P_M), M_Gamma )
```

or, for a mode-based approximation,

```text
E_trace,i(M)^2 ~= sum_{k not represented by M} lambda_k * a_k^2
```

If the trace coefficients `a_k` are unknown, use a conservative spectral tail
proxy from the generalized eigenvalues and local geometry.

### PNM

PNM keeps only the coarsest trace behavior. Its prior is

```text
R_i(PNM) = C_geo,i * C_stab,i * E_trace,i(PNM) * B_i
```

This is large when the region has narrow throats, high tortuosity, many
connected interface components, or slow Steklov spectral decay.

### DDPNMT

DDPNMT adds the low-dimensional interface correction, interpreted as normal and
tangential/transverse trace modes per interface. Its prior is

```text
R_i(DDPNMT) = C_geo,i * C_stab,i * E_trace,i(DDPNMT) * B_i
```

It should be selected when the spectral tail drops sharply after the first few
interface modes. In that case, a small number of directional trace modes captures
most of the interface response.

### HODDPNM50

HODDPNM50 releases roughly half of the high-risk interface trace DOFs. Its prior
is

```text
R_i(HODDPNM50) = C_geo,i * C_stab,i * E_trace,i(HODDPNM50) * B_i
```

It is appropriate when the first few modes are not enough, but the Steklov tail
has moderate decay and the interface complexity is localized.

### HODDPNM100

HODDPNM100 releases the full interface trace space. Relative to the full FEM
trace space,

```text
E_trace,i(HODDPNM100) ~= 0
```

so the remaining error is only the ordinary FEM discretization error:

```text
R_i(HODDPNM100) ~= C_FEM,i h_i^s B_i
```

This should be used when the local Stokes stability constant is bad or the
Steklov spectral tail decays slowly.

## Geometry-Only Constants

A fully explicit universal constant for arbitrary voxel porous media is not
available in practice. But under uniform Lipschitz and mesh-shape assumptions,
the Stokes estimate can be controlled by constants depending only on local
geometry:

```text
C_stab,i = F(C_P,i, C_K,i, beta_i^{-1}, C_trace,i, C_ext,i)
```

where:

- `C_P,i`: local Poincare constant;
- `C_K,i`: Korn constant;
- `beta_i`: local Stokes inf-sup constant;
- `C_trace,i`: trace constant;
- `C_ext,i`: divergence-free or Stokes extension constant.

For computation, use geometry proxies:

```text
C_geo,i =
  c1 / r_min,i^2
+ c2 / delta_min,i^2
+ c3 kappa_i^2
+ c4 tau_i
+ c5 N_conn,i
+ c6 A_Gamma,i^pore
```

and optionally compute `beta_i`, `C_P,i`, and `E_trace,i` from small local
generalized eigenproblems.

## Three Geometry Cases

### Case A: regular/open region

Criteria:

```text
C_geo small, beta_i not small, Steklov tail fast
```

Decision:

```text
PNM is acceptable; upgrade only if gain_i is high.
```

### Case B: interface-dominated region

Criteria:

```text
C_geo moderate, A_Gamma^pore large, N_conn large, first Steklov modes dominant
```

Decision:

```text
PNM -> DDPNMT, or DDPNMT -> HODDPNM50 if residual trace tail remains large.
```

### Case C: bottleneck / unstable Stokes region

Criteria:

```text
r_min small, delta_min small, tau large, beta_i small, Steklov tail slow
```

Decision:

```text
upgrade toward HODDPNM100 first.
```

## Upgrade Rule

For every candidate region and candidate next method, compute

```text
Gain_i = [R_i(current_method) - R_i(next_method)] / added_DOF_i
```

Upgrade the regions with largest `Gain_i`, subject to the active DOF cap.
This is stronger than simply upgrading the largest-error region because it
measures expected accuracy per added degree of freedom.

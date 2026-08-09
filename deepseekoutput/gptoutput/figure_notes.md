# Final numerical section — figure notes (2026-08-09)

Scripts: `deepseekoutput/gptoutput/scripts/` (figlib.py = data layer + shared
style; make_final_figures.py = all figure functions).
Outputs: `deepseekoutput/gptoutput/figures/`.

Regenerate: `/d/Miniconda3/envs/econ/python.exe deepseekoutput/gptoutput/scripts/make_final_figures.py`

Method naming: W0n = Classic-DDPNM-1 (`span{1}⊗n`), W1n = NormalLinear-DDPNM-3
(`span{1,s,t}⊗n`), W1v = Affine-DDPNM-9 (`span{1,s,t}⊗{n,t1,t2}`),
W0v = constant vector `span{1}⊗{n,t1,t2}` — **no W0v run exists yet**; all
figure functions detect and render it as soon as it appears in the metrics CSVs.

Style: learned from `affine_ddpnm_3d_random_porous/plot_random_errors.py`
(source of `outputs/benchmark/01_slice_error_fields.png`): slice fields via
per-subdomain linear interpolation + masked Gaussian smoothing, sphere
cross-section outlines, subdomain boundary lines, sci colorbar, serif panel
letters. Categorical palette = validated dataviz reference (blue/orange/aqua,
violet reserved for W0v).

---

## Produced (8 figures)

### fig2_modal_budget_fourspace.png
FEM-relative velocity L2 error vs modes/interface r = 1, 3, 9, log-y,
one panel per geometry (Uniform-27 / Random-27 / Real-100), shared y-range.
Only legal nested paths are connected: currently the single path
W0n → W1n → W1v (annotation notes W0v at r=3 pending). When a W0v run exists,
the figure additionally shows W0v at r=3 (violet) and connects the dashed
path W0n → W0v → W1v; W0v and W1n are never connected (they are generally
non-nested).
Data: `outputs/benchmark_w1n/*_metrics.csv` (velocity_relative_L2).

### fig3_metrics_fourspace.png
Four panels: velocity L2, velocity broken-H1, pressure L2, outlet-flux error.
All methods per geometry, log-y with a **consistent percent tick format on
every panel** (3 significant digits), dotted lines only along nested chains
(W0n → W1n → W1v); W0v would be added automatically when data exists and is
never joined to W1n. Data: same CSVs (velocity_relative_broken_H1,
pressure_relative_L2, outlet_flux_relative_error).

### fig5_fields_uniform27.png / fig5_fields_random27.png
Slice z = 0.5. Panels: (a) FEM |u| (own viridis scale); (b)(c)(d) error
|u − u_FEM| for W0n / W1n / W1v with **one shared color scale** — vmax = p90
of the W0n error field (a p98 vmax would push W0n's bulk below ~10% of the
scale and make its panel read as empty; with p90, W0n's median sits at ~30%
of the scale and W1n/W1v residual structure stays visible). Turbo colormap,
sphere cross-sections and subdomain interface traces overlaid, x/y
coordinates shown. The shared scale visualizes the mechanism: W1n leaves
visible residual structure, W1v reduces it substantially.
**Layout note (fixed):** both colorbars sit in two dedicated columns at the
right of the 4 panels, separated by a spacer column so their tick numbers
never collide (width_ratios [1,1,1,1,0.06,0.14,0.06]); an earlier interleaved
layout collided the W0n panel with the reference colorbar slot — verified via
a full bbox audit that no axes/colorbar/text overlap remains in any figure.
**Interfaces (fixed):** the Uniform-27 slice triangulation is not conforming
to the grid partition (triangles cross the partition planes), which made
per-subdomain interpolation produce white NaN bands along the interfaces and
left the traces undrawn. Uniform-27 now interpolates globally over all slice
vertices (NaN only inside sphere holes, 31%) and draws the grid-plane traces
analytically (x/y = 0.2/0.5/0.8, clipped to the fluid domain). Random-27 uses
exact per-triangle parent-cell labels from the archive.
Data: `affine_ddpnm_3d/outputs/benchmark_w1n/affine_benchmark_fields.npz`,
`affine_ddpnm_3d_random_porous/outputs/benchmark_w1n/random_benchmark_fields.npz`.
Note: the Uniform-27 archive lacks sphere/cell-label arrays; subdomain labels
are derived from the Cartesian grid cell of each slice vertex
(CELL_EDGES = 0/0.2/0.5/0.8/1, 4×4×4 = 64 cuboids).

### fig5_interface_flux_modes.png
Representative interface from Uniform-27 (interface 56): **two rows** —
top row q(s,t) = u·n for FEM / W0n / W1n / W1v on one common diverging scale
with numeric (s,t) ticks; bottom row the FEM-relative flux errors
|q − q_FEM| for W0n / W1n / W1v on one shared sequential scale (vmax = p90
of the pooled flux errors). Random-27 (and Real-100) rows pending: their
archives do not export per-interface flux keys.

### fig7_accuracy_cost_standardized.png
One panel per geometry. Reduced methods only: filled point = online solve,
hollow point = first solve (incl. offline), dotted connector. FEM is drawn as
a vertical dashed reference line labelled "FEM reference, error = 0
(off log axis)" — it is never plotted as a nonzero-error point. Data: single
timing runs per method (offline_seconds / online_seconds / first_solve);
**no repeated timing runs exist, so no median/spread is shown**.
**Caveat (noted on the figure): FEM timing scope differs across geometries —
Uniform-27 / Random-27 report the monolithic solve time (online), Real-100
reports total_s incl. assembly (different solver path). Unify the timing
protocol before final use.**

### fig10_observed_shares_clean.png
100% horizontal stacked bars of the observed FEM-relative velocity-L2 error
reduction along W0n → W1n → W1v (blue: normal-spatial step, aqua: additional
vectorial step). Legend in two columns outside the plotting area, below the
axis, verified non-overlapping with the x label.
**Caption: these are path-dependent observed shares along W0n → W1n → W1v,
not causal or theorem-level contributions to the total error reduction.**

### fig1b_geometry_partition_interfaces.png
Two rows × three geometries, consistent camera/scale/orientation:
row 1 solid sphere packing, row 2 **finite-area internal interfaces in the
same wireframe style for all three geometries** — Cartesian 4×4×4 grid
partitions (planes 0.2/0.5/0.8 and 0.25/0.5/0.75; 144 interfaces each) for
Uniform-27 / Real-100, and the interface triangles extracted from the actual
Random-27 tetrahedral mesh (faces shared by two subdomains) — so the
solid-packing irregularity and the partition irregularity can be compared
fairly across geometries.

---

## Missing data (5 figures — scripts ready, auto-activate when data lands)

| Figure | Needed data | Where it would come from |
|---|---|---|
| fig4_equal_budget_W0v_vs_W1n.png | W0v (constant vector, r=3) run | a `VectorConstant`-style benchmark on all three geometries |
| fig5_fields_real100.png | Real-100 W1n slice fields | fixing the `z_slice` API mismatch (`run_benchmark.py` vs `ddpnm3d/visualization.py`, handoff §4.1) and re-running |
| fig6_schur_energy_decomposition.png | full broken-pressure C-DD (Δ_N, Δ_{T\|N}, affine residual) | not archived by any benchmark |
| fig8_mesh_sensitivity.png | ≥3 mesh levels for Uniform-27 | new mesh-sensitivity runs |
| fig9_random_ensemble.png | several frozen Random-27 seeds | only seed 20260804 archived |

---

## Captions (LaTeX-ready)

Fig2: FEM-relative velocity L2 error against the interface modal budget r per
geometry; only nested spaces are joined (W0n → W1n → W1v; W0v at r = 3
pending). Fig3: the four error metrics across methods and geometries.
Fig5 (fields): velocity magnitude and FEM-relative error slices at z = 0.5;
all error panels share one color scale. Fig5 (flux): normal flux on a
representative interface of Uniform-27. Fig7: accuracy versus solve time;
FEM is a reference line at zero error. Fig10: observed error-reduction shares
(path-dependent, not causal). Fig1b: solid packing and partition/interface
network for the three geometries.

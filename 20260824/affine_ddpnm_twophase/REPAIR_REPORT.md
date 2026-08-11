# Two-phase solver repair report

## Correctness repairs

1. Corrected the sign of the analytic Corey fractional-flow derivative and
   verified it against centered finite differences.
2. Replaced the whole-plane strong saturation condition with upwind numerical
   fluxes on the pressure inlet and outlet.  Reverse-flow portions now use an
   exterior state instead of being incorrectly pinned.
3. Restored the conservative outlet boundary term in the weak form.
4. Corrected SUPG to use the characteristic velocity `fw'(Sw) u`, including
   both derivative factors and the correct transient right-hand-side sign.
5. Changed the default injection value and limiter interval to the physical
   Corey range `[Swr, 1-Sor]`.
6. Initialized the domain uniformly, so recovery starts at zero.
7. Replaced the self-fulfilling limiter residual with the independent balance

   `M_new - M_old + dt * (Q_water,out - Q_water,in)`.

8. Replaced nodal outlet weighting with facet-integrated positive-flux water
   cut and added injected pore-volume tracking.
9. Added Picard convergence counts, strict JSON output, a mass-budget figure,
   and explicit rejection of infeasible limiter targets.
10. Removed all invalid `tp.tt` references, repaired the standalone diagnostic,
    and retained old outputs only under `outputs/legacy_invalid_20260811/`.

## Verification completed in the repair environment

- Seven pure numerical tests pass: Corey monotonicity, analytic derivative,
  limiter bounds/mass, infeasible-target rejection, crossing-time interpolation,
  zero initial recovery, and boundary-flux mass bookkeeping.
- Every Python source file passes byte-code compilation.
- The full benchmark entry point was started but could not import its runtime:
  the repair container has no PyVista, DOLFINx, UFL, Basix, Gmsh, or MPI4Py.
  Consequently it produced no new physical benchmark result here.

Run `run_benchmark.ps1` in the documented `fenicsx` conda environment to perform
the dependency preflight, repeat the tests, execute the full benchmark, and
capture its log.

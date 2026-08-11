# Invalid pre-fix outputs

These files are retained only to document the failure that motivated the
solver repair.  They must not be cited as physical or numerical validation.

The smoke run used a fractional-flow derivative with the wrong sign, omitted
the conservative outlet boundary flux, measured only limiter mass restoration,
and initialized recovery from a boundary-pinned field.  The benchmark run then
stopped at an invalid `tp.tt` attribute reference before producing results.

Run `run_affine_ddpnm_twophase.py` with the repaired source to create a new
`outputs/benchmark_twophase/` directory.

"""Immutable configuration for DD-PNM simulations (2D and 3D)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DdpnmConfig:
    """All parameters for a DD-PNM run.

    Dimension-agnostic physics parameters are always present.  Geometry
    sizing parameters are optional; the run scripts assert that the
    relevant subset is not ``None``.
    """

    # -- physics ---------------------------------------------------------------
    viscosity: float = 1.0
    inlet_pressure: float = 1.0
    outlet_pressure: float = 0.0
    pressure_stabilization: float = 0.0

    # -- output ----------------------------------------------------------------
    output_dir: Path = field(default_factory=lambda: Path("outputs/default"))

    # -- shared mesh -----------------------------------------------------------
    mesh_size: float = 0.04

    # -- 3D geometry -----------------------------------------------------------
    sphere_size: float | None = None
    boundary_size: float | None = None
    interface_size: float | None = None
    sphere_band: float | None = None
    boundary_band: float | None = None
    interface_band: float | None = None

    # -- 2D geometry -----------------------------------------------------------
    wall_size: float | None = None
    throat_size: float | None = None
    wall_band: float | None = None
    throat_band: float | None = None
    interface_order: int | None = None

    # -- reference solver ------------------------------------------------------
    with_reference: bool = False
    reference_iterative_threshold: int = 100_000
    reference_rtol: float = 1.0e-9
    reference_restart: int = 60
    reference_maxiter: int = 120
    reference_ilu_drop_tolerance: float = 2.0e-3
    reference_ilu_fill_factor: float = 6.0

    # -- adaptive hierarchy -----------------------------------------------------
    target_tolerance: float = 1.0e-2
    marking_theta: float = 0.65
    max_marked_per_iteration: int = 12
    max_iterations_per_phase: int = 40

    # -- helpers ----------------------------------------------------------------

    def as_dict(self) -> dict:
        """Return all non-None fields as a plain dict (suitable for JSON)."""
        result: dict = {}
        for field_def in self.__dataclass_fields__.values():
            value = getattr(self, field_def.name)
            if isinstance(value, Path):
                value = str(value)
            result[field_def.name] = value
        return result

    @classmethod
    def from_cli(cls, description: str = "") -> "DdpnmConfig":
        """Parse command-line arguments and return a frozen config.

        The returned config always has every field populated (defaults are used
        for any argument the caller did not register).
        """
        parser = argparse.ArgumentParser(description=description)
        _add_arguments(parser)
        args = parser.parse_args()
        return cls(
            **{
                field_def.name: getattr(args, field_def.name)
                for field_def in cls.__dataclass_fields__.values()
            }
        )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register every DdpnmConfig field as a CLI flag."""
    # physics
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--inlet-pressure", type=float, default=1.0)
    parser.add_argument("--outlet-pressure", type=float, default=0.0)
    parser.add_argument("--pressure-stabilization", type=float, default=0.0)
    # output
    parser.add_argument("--out-dir", type=Path, dest="output_dir",
                       default=Path("outputs/default"))
    # shared mesh
    parser.add_argument("--mesh-size", type=float, default=0.04)
    # 3D geometry
    parser.add_argument("--sphere-size", type=float, default=None)
    parser.add_argument("--boundary-size", type=float, default=None)
    parser.add_argument("--interface-size", type=float, default=None)
    parser.add_argument("--sphere-band", type=float, default=None)
    parser.add_argument("--boundary-band", type=float, default=None)
    parser.add_argument("--interface-band", type=float, default=None)
    # 2D geometry
    parser.add_argument("--wall-size", type=float, default=None)
    parser.add_argument("--throat-size", type=float, default=None)
    parser.add_argument("--wall-band", type=float, default=None)
    parser.add_argument("--throat-band", type=float, default=None)
    parser.add_argument("--interface-order", type=int, default=None)
    # reference
    parser.add_argument("--with-reference", action="store_true")
    parser.add_argument("--reference-iterative-threshold", type=int, default=100_000)
    parser.add_argument("--reference-rtol", type=float, default=1.0e-9)
    parser.add_argument("--reference-restart", type=int, default=60)
    parser.add_argument("--reference-maxiter", type=int, default=120)
    parser.add_argument("--reference-ilu-drop-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--reference-ilu-fill-factor", type=float, default=6.0)
    # adaptive
    parser.add_argument("--target-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--marking-theta", type=float, default=0.65)
    parser.add_argument("--max-marked-per-iteration", type=int, default=12)
    parser.add_argument("--max-iterations-per-phase", type=int, default=40)

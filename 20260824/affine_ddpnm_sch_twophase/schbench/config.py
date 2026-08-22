"""Typed configuration and experiment-arm definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class FlowMethod(StrEnum):
    FEM = "FEM"
    CLASSIC = "Classic"
    AFFINE = "Affine"


class Coupling(StrEnum):
    FROZEN = "frozen"
    SFI = "SFI"


class ViscousForm(StrEnum):
    GRADIENT = "gradient"
    SYMMETRIC = "symmetric"


class InitialProfile(StrEnum):
    DISCONTINUOUS = "discontinuous"
    EXPONENTIAL = "exponential"


@dataclass(frozen=True)
class Arm:
    flow: FlowMethod
    coupling: Coupling

    @property
    def name(self) -> str:
        return f"{self.flow.value}-{self.coupling.value}"

    @classmethod
    def all(cls) -> tuple["Arm", ...]:
        return tuple(cls(flow, coupling) for flow in FlowMethod for coupling in Coupling)

    @classmethod
    def parse_many(cls, value: str) -> tuple["Arm", ...]:
        if value.strip().lower() == "all":
            return cls.all()
        lookup = {arm.name.lower(): arm for arm in cls.all()}
        requested = [item.strip().lower() for item in value.split(",") if item.strip()]
        unknown = [item for item in requested if item not in lookup]
        if unknown:
            raise ValueError(f"Unknown arm(s): {unknown}; choices are {sorted(lookup)}")
        return tuple(lookup[item] for item in requested)


@dataclass(frozen=True)
class Physics:
    mobility: float = 2.0e-4
    surface_tension: float = 2.0e-3
    epsilon_factor: float = 1.5
    viscosity_water: float = 1.0
    viscosity_oil: float = 5.0
    inlet_pressure: float = 1.0
    outlet_pressure: float = 0.0
    phi_initial: float = -1.0
    phi_inlet: float = 1.0
    initial_profile: InitialProfile = InitialProfile.DISCONTINUOUS
    initial_transition_length: float = 0.05

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class Numerics:
    dt: float = 0.25
    t_final: float = 1.0
    newton_tolerance: float = 1.0e-8
    newton_max_iterations: int = 18
    line_search_min: float = 1.0 / 128.0
    sfi_tolerance: float = 1.0e-4
    sfi_max_iterations: int = 12
    sfi_relaxation: float = 0.65
    pressure_stabilization: float = 1.0e-10
    affine_pod_tolerance: float = 1.0e-8
    viscous_form: ViscousForm = ViscousForm.GRADIENT

    @property
    def steps(self) -> int:
        ratio = self.t_final / self.dt
        rounded = round(ratio)
        if abs(ratio - rounded) > 1.0e-10:
            raise ValueError("t_final must be an integer multiple of dt")
        if rounded < 1:
            raise ValueError("At least one time step is required")
        return int(rounded)

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)

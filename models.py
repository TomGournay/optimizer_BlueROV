from dataclasses import dataclass

import numpy as np


# 6-DOF conventions used throughout the project:
# inertial/world frame: NED = [north, east, down]
# body frame: x forward, y starboard/right, z downward
#
# eta = [north, east, down, roll, pitch, yaw] in the inertial NED frame
# nu = [surge, sway, heave, roll_rate, pitch_rate, yaw_rate] in the body frame
# tau = [X, Y, Z, K, M, N] body-frame forces and moments


@dataclass(frozen=True)
class Design:
    """Design variables for one simulation or optimization candidate.

    Values are stored generically so design variables can be added by extending
    the design configuration. The vectorized thruster layout uses raw 3D
    components that are normalized by allocation.py before entering physics.
    """

    names: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError("design names and values must have the same length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("design names must be unique")

    def get(self, name: str) -> float:
        """Return a design variable by name."""

        try:
            index = self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown design variable: {name}") from exc
        return self.values[index]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values))


@dataclass(frozen=True)
class ControlParams:
    """Parameters used by the selected control mode to compute motor commands."""

    mode: str
    names: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError("control names and values must have the same length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("control names must be unique")

    def get(self, name: str) -> float:
        """Return a control parameter by name."""

        try:
            index = self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown control parameter: {name}") from exc
        return self.values[index]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values))


@dataclass(frozen=True)
class VehicleGeometry:
    """Fixed intrinsic geometry of the vehicle."""

    thruster_positions: np.ndarray
    thruster_sphere_radius: float

    @property
    def n_thrusters(self) -> int:
        return self.thruster_positions.shape[0]


@dataclass(frozen=True)
class VehicleDynamicsParameters:
    """Fixed dynamic parameters used by the 6-DOF equations."""

    center_of_mass: np.ndarray
    center_of_buoyancy: np.ndarray
    rigid_body_mass_matrix: np.ndarray
    added_mass_matrix: np.ndarray
    linear_damping: np.ndarray
    quadratic_damping: np.ndarray
    weight: float
    buoyancy: float

    @property
    def mass_matrix(self) -> np.ndarray:
        return self.rigid_body_mass_matrix + self.added_mass_matrix


@dataclass(frozen=True)
class VehicleModel:
    """Complete fixed vehicle model assembled from geometry and dynamics."""

    geometry: VehicleGeometry
    dynamics: VehicleDynamicsParameters


@dataclass(frozen=True)
class SimulationResult:
    """Output of a simulated trajectory."""

    time: np.ndarray
    eta: np.ndarray
    nu: np.ndarray
    motor_commands: np.ndarray
    motor_thrusts: np.ndarray
    wrenches: np.ndarray


@dataclass(frozen=True)
class OptimizationResult:
    """Output of the optimization loop."""

    design: Design
    control_params: ControlParams
    cost: float
    simulation: SimulationResult

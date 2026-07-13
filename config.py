from dataclasses import dataclass, field
from typing import Literal

import numpy as np


ObjectiveMode = str
ControlMode = Literal["piecewise_constant", "pid_pose"]

N_DESIGN_THRUSTERS = 6
DESIGN_VECTOR_AXES = ("x", "y", "z")


def vectorized_thruster_design_names(
    n_thrusters: int = N_DESIGN_THRUSTERS,
) -> tuple[str, ...]:
    """Return raw position and direction vector component names."""

    return tuple(
        name
        for motor_id in range(n_thrusters)
        for name in (
            *(f"pos_{axis}_m{motor_id}" for axis in DESIGN_VECTOR_AXES),
            *(f"dir_{axis}_m{motor_id}" for axis in DESIGN_VECTOR_AXES),
        )
    )


def default_thruster_position_vectors() -> np.ndarray:
    """Return six default raw position vectors on the unit sphere."""

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )


def default_thruster_direction_vectors() -> np.ndarray:
    """Return a deterministic full-rank default direction set."""

    return np.array(
        [
            [0.053543, -0.483986, -0.873436],
            [-0.190199, 0.769754, -0.609346],
            [0.575425, 0.305400, 0.758695],
            [-0.746158, -0.082759, 0.660605],
            [-0.852098, -0.350176, 0.388981],
            [-0.430419, 0.856988, -0.283392],
        ],
        dtype=float,
    )


def default_vectorized_thruster_design_values() -> np.ndarray:
    """Return default raw design values grouped motor by motor."""

    positions = default_thruster_position_vectors()
    directions = default_thruster_direction_vectors()
    values = np.concatenate((positions, directions), axis=1)
    return values.reshape(-1)


@dataclass(frozen=True)
class DesignConfig:
    """Names and bounds for design variables."""

    names: tuple[str, ...] = field(default_factory=vectorized_thruster_design_names)
    default_values: np.ndarray = field(
        default_factory=default_vectorized_thruster_design_values
    )
    lower_bounds: np.ndarray = field(
        default_factory=lambda: -np.ones(6 * N_DESIGN_THRUSTERS)
    )
    upper_bounds: np.ndarray = field(
        default_factory=lambda: np.ones(6 * N_DESIGN_THRUSTERS)
    )
    min_direction_outward_dot: float = -0.10
    min_allocation_singular_value: float = 0.15
    min_thruster_spacing: float = 0.08

    @property
    def n_parameters(self) -> int:
        return len(self.names)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        lower_bounds = np.asarray(self.lower_bounds, dtype=float)
        upper_bounds = np.asarray(self.upper_bounds, dtype=float)

        if lower_bounds.shape != (self.n_parameters,):
            raise ValueError("design lower_bounds must match design names")
        if upper_bounds.shape != (self.n_parameters,):
            raise ValueError("design upper_bounds must match design names")
        if np.any(lower_bounds > upper_bounds):
            raise ValueError("design lower_bounds must be <= upper_bounds")

        return list(zip(lower_bounds.tolist(), upper_bounds.tolist()))

    @property
    def defaults(self) -> np.ndarray:
        """Return validated default design values."""

        default_values = np.asarray(self.default_values, dtype=float)
        lower_bounds = np.asarray(self.lower_bounds, dtype=float)
        upper_bounds = np.asarray(self.upper_bounds, dtype=float)

        if default_values.shape != (self.n_parameters,):
            raise ValueError("design default_values must match design names")
        if np.any(default_values < lower_bounds) or np.any(default_values > upper_bounds):
            raise ValueError("design default_values must be inside design bounds")

        return default_values


@dataclass(frozen=True)
class ControlConfig:
    """Settings for the motor command parametrization."""

    mode: ControlMode = "piecewise_constant"
    n_segments: int = 5
    u_min: float = -1.0
    u_max: float = 1.0
    pid_integral_limit: float = 5.0

    # Future controller modes can add their own parameterization here.
    # Example:
    #
    # mode: ControlMode = "pid_pose"
    #
    # pid_pose optimizes PID gains that track cfg.objective.target_pose.
    # It computes a desired wrench, allocates it to
    # thrusters, then converts desired thrusts back to normalized commands.


@dataclass(frozen=True)
class SimulationConfig:
    """Time grid and initial conditions.

    Positions are expressed in the inertial NED frame:
    [north, east, down].
    """

    duration: float = 15.0
    dt: float = 0.10
    initial_eta: np.ndarray = field(default_factory=lambda: np.zeros(6))
    initial_nu: np.ndarray = field(default_factory=lambda: np.zeros(6))

    @property
    def n_steps(self) -> int:
        return int(round(self.duration / self.dt)) + 1


@dataclass(frozen=True)
class ObjectiveConfig:
    """Active mission objective.

    Available modes are registered in cost.py.
    """

    mode: ObjectiveMode = "forward_progress"
    target_position: np.ndarray = field(default_factory=lambda: np.array([5.0, 0.0, 0.0]))
    target_attitude: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_time: float | None = None
    target_tolerance: float = 0.10
    target_position_weight: float = 1.0
    target_attitude_weight: float = 1.0
    require_station_keeping: bool = False
    station_keeping_window: float = 2.0
    station_time_objective: bool = False
    station_position_tolerance: float = 0.15
    station_attitude_tolerance: float = float(np.deg2rad(5.0))
    station_linear_velocity_tolerance: float = 0.05
    station_angular_velocity_tolerance: float = float(np.deg2rad(3.0))

    @property
    def target_pose(self) -> np.ndarray:
        """Return target eta = [x, y, z, phi, theta, psi]."""

        target_position = np.asarray(self.target_position, dtype=float)
        target_attitude = np.asarray(self.target_attitude, dtype=float)

        if target_position.shape != (3,):
            raise ValueError("target_position must have shape (3,)")
        if target_attitude.shape != (3,):
            raise ValueError("target_attitude must have shape (3,)")

        return np.concatenate((target_position, target_attitude))


@dataclass(frozen=True)
class CostWeights:
    """Weights used to combine trajectory penalties into one scalar cost."""

    objective: float = 1.0
    target_arrival_time: float = 1.0
    drift: float = 10.0
    attitude: float = 5.0
    energy: float = 0.1
    smoothness: float = 0.1
    station_position: float = 5.0
    station_attitude: float = 2.0
    station_linear_velocity: float = 10.0
    station_angular_velocity: float = 5.0
    station_success_time: float = 1.0
    station_failure: float = 10.0
    allocation_quality: float = 50.0
    inward_direction: float = 100.0
    position_spacing: float = 10.0

@dataclass(frozen=True)
class OptimizerConfig:
    """Numerical settings for the optimizer."""

    max_iterations: int =   50
    population_size: int =   10
    random_seed: int = 2


@dataclass(frozen=True)
class ProblemConfig:
    """Top-level configuration object passed through the problem."""

    design: DesignConfig = field(default_factory=DesignConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    cost: CostWeights = field(default_factory=CostWeights)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

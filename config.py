from dataclasses import dataclass, field
from typing import Literal

import numpy as np


ObjectiveMode = str
ControlMode = Literal["piecewise_constant", "pid_pose"]


@dataclass(frozen=True)
class DesignConfig:
    """Names and bounds for design variables."""

    names: tuple[str, ...] = ("alpha",)
    lower_bounds: np.ndarray = field(default_factory=lambda: np.array([np.deg2rad(0.0)]))
    upper_bounds: np.ndarray = field(default_factory=lambda: np.array([np.deg2rad(90.0)]))

    # Example with several design variables:
    #
    # names: tuple[str, ...] = ("alpha1", "alpha2")
    # lower_bounds: np.ndarray = field(
    #     default_factory=lambda: np.array([np.deg2rad(0.0), np.deg2rad(0.0)])
    # )
    # upper_bounds: np.ndarray = field(
    #     default_factory=lambda: np.array([np.deg2rad(90.0), np.deg2rad(90.0)])
    # )
    #
    # optimization.py will automatically use len(names) to size and decode z.
    # The new variables must still be used explicitly in the physics, e.g.
    # design.get("alpha1") and design.get("alpha2") inside allocation.py.

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

    duration: float = 8.0
    dt: float = 0.05
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

@dataclass(frozen=True)
class OptimizerConfig:
    """Numerical settings for the optimizer."""

    max_iterations: int =   5
    population_size: int =   2
    random_seed: int = 1


@dataclass(frozen=True)
class ProblemConfig:
    """Top-level configuration object passed through the problem."""

    design: DesignConfig = field(default_factory=DesignConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    cost: CostWeights = field(default_factory=CostWeights)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

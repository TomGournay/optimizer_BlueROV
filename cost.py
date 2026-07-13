from collections.abc import Callable

import numpy as np

from config import ProblemConfig
from models import SimulationResult

ObjectiveCostFunction = Callable[[SimulationResult, ProblemConfig], float]
TARGET_POSE_OBJECTIVE_MODES = ("target_position", "target_pose")


def wrap_angle_error(error: np.ndarray) -> np.ndarray:
    """Wrap angular errors to [-pi, pi]."""

    return (error + np.pi) % (2.0 * np.pi) - np.pi


def forward_progress_cost(simulation: SimulationResult) -> float:
    """Reward final north progress by returning its negative value.

    Optimizers minimize J. Returning -north_final means that trajectories going
    farther along the north axis get a lower cost.
    """

    north_final = simulation.eta[-1, 0]
    return float(-north_final)


def final_position_cost(
    simulation: SimulationResult,
    target_position: np.ndarray,
) -> float:
    """Return squared final position error in the inertial NED frame."""

    target_position = np.asarray(target_position, dtype=float)
    if target_position.shape != (3,):
        raise ValueError("target_position must have shape (3,)")

    final_position = simulation.eta[-1, :3]
    error = final_position - target_position
    return float(error @ error)


def target_sample_index(simulation: SimulationResult, cfg: ProblemConfig) -> int:
    """Return the simulation index used for target-pose error evaluation."""

    if cfg.objective.target_time is None:
        return simulation.time.size - 1

    target_time = float(cfg.objective.target_time)
    start_time = float(simulation.time[0])
    end_time = float(simulation.time[-1])

    if target_time < start_time or target_time > end_time:
        raise ValueError(
            f"target_time must be inside the simulation interval "
            f"[{start_time}, {end_time}]"
        )

    return int(np.argmin(np.abs(simulation.time - target_time)))


def target_pose_errors(
    simulation: SimulationResult,
    cfg: ProblemConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return position and attitude errors at the configured target time."""

    sample_index = target_sample_index(simulation, cfg)
    target_pose = cfg.objective.target_pose

    position_error = simulation.eta[sample_index, :3] - target_pose[:3]
    attitude_error = wrap_angle_error(simulation.eta[sample_index, 3:6] - target_pose[3:6])

    return position_error, attitude_error


def target_position_error_cost(simulation: SimulationResult, cfg: ProblemConfig) -> float:
    """Return squared position error at the configured target time."""

    position_error, _ = target_pose_errors(simulation, cfg)
    return float(position_error @ position_error)


def target_attitude_error_cost(simulation: SimulationResult, cfg: ProblemConfig) -> float:
    """Return squared attitude error at the configured target time."""

    _, attitude_error = target_pose_errors(simulation, cfg)
    return float(attitude_error @ attitude_error)


def target_pose_cost(simulation: SimulationResult, cfg: ProblemConfig) -> float:
    """Return weighted 6D target-pose error at the configured target time."""

    position_error, attitude_error = target_pose_errors(simulation, cfg)
    position_cost = float(position_error @ position_error)
    attitude_cost_value = float(attitude_error @ attitude_error)

    return float(
        cfg.objective.target_position_weight * position_cost
        + cfg.objective.target_attitude_weight * attitude_cost_value
    )


def forward_progress_objective(
    simulation: SimulationResult,
    cfg: ProblemConfig,
) -> float:
    """Objective wrapper for maximizing final north progress."""

    return forward_progress_cost(simulation)


def target_position_objective(
    simulation: SimulationResult,
    cfg: ProblemConfig,
) -> float:
    """Objective wrapper for minimizing target pose error.

    This keeps the historical objective name usable. With the new target_pose
    settings, target_position is a backwards-compatible alias.
    """

    return target_pose_cost(simulation, cfg)


def target_pose_objective(
    simulation: SimulationResult,
    cfg: ProblemConfig,
) -> float:
    """Objective wrapper for minimizing target pose error."""

    return target_pose_cost(simulation, cfg)


OBJECTIVE_COSTS: dict[str, ObjectiveCostFunction] = {
    "forward_progress": forward_progress_objective,
    "target_position": target_position_objective,
    "target_pose": target_pose_objective,
}


def available_objective_modes() -> tuple[str, ...]:
    """Return objective modes registered in this module."""

    return tuple(OBJECTIVE_COSTS.keys())


def target_arrival_time_cost(simulation: SimulationResult, cfg: ProblemConfig) -> float:
    """Penalize late arrival to the target position.

    This component is active only in target-pose modes without a fixed
    target_time. If the target is reached within cfg.objective.target_tolerance,
    the cost is the first arrival time. If not reached, the cost is the full
    simulation duration plus the final remaining distance.
    """

    if cfg.objective.mode not in TARGET_POSE_OBJECTIVE_MODES:
        return 0.0
    if cfg.objective.target_time is not None:
        return 0.0

    target_position = np.asarray(cfg.objective.target_position, dtype=float)
    if target_position.shape != (3,):
        raise ValueError("target_position must have shape (3,)")

    distances = np.linalg.norm(simulation.eta[:, :3] - target_position, axis=1)
    reached_indices = np.flatnonzero(distances <= cfg.objective.target_tolerance)

    if reached_indices.size > 0:
        first_reached = reached_indices[0]
        return float(simulation.time[first_reached])

    return float(simulation.time[-1] + distances[-1])


def objective_cost(simulation: SimulationResult, cfg: ProblemConfig) -> float:
    """Return the active mission objective selected in cfg.objective.mode."""

    try:
        objective_function = OBJECTIVE_COSTS[cfg.objective.mode]
    except KeyError as exc:
        valid_modes = ", ".join(available_objective_modes())
        raise ValueError(
            f"unknown objective mode: {cfg.objective.mode}. "
            f"Available modes: {valid_modes}"
        ) from exc

    return objective_function(simulation, cfg)


def drift_cost(simulation: SimulationResult) -> float:
    """Penalize motion away from the desired north-axis translation."""

    east = simulation.eta[:, 1]
    down = simulation.eta[:, 2]
    return float(np.mean(east**2 + down**2))


def attitude_cost(simulation: SimulationResult) -> float:
    """Penalize roll, pitch, and yaw over the trajectory."""

    attitude = simulation.eta[:, 3:6]
    return float(np.mean(np.sum(attitude**2, axis=1)))


def energy_cost(simulation: SimulationResult, cfg: ProblemConfig) -> float:
    """Penalize motor command effort over time."""

    command_norm_squared = np.sum(simulation.motor_commands**2, axis=1)
    return float(np.sum(command_norm_squared) * cfg.simulation.dt)


def smoothness_cost(simulation: SimulationResult) -> float:
    """Penalize jumps between successive motor command samples."""

    command_differences = np.diff(simulation.motor_commands, axis=0)
    return float(np.sum(command_differences**2))


def cost_components(
    simulation: SimulationResult,
    cfg: ProblemConfig,
) -> dict[str, float]:
    """Return unweighted cost components for diagnostics."""

    objective_components = {
        mode: objective_function(simulation, cfg)
        for mode, objective_function in OBJECTIVE_COSTS.items()
    }

    return {
        "objective": objective_cost(simulation, cfg),
        **objective_components,
        "target_position_error": target_position_error_cost(simulation, cfg),
        "target_attitude_error": target_attitude_error_cost(simulation, cfg),
        "target_arrival_time": target_arrival_time_cost(simulation, cfg),
        "drift": drift_cost(simulation),
        "attitude": attitude_cost(simulation),
        "energy": energy_cost(simulation, cfg),
        "smoothness": smoothness_cost(simulation),
    }


def trajectory_cost(simulation: SimulationResult, cfg: ProblemConfig) -> float:
    """Return the weighted scalar objective J for one simulated trajectory."""

    components = cost_components(simulation, cfg)
    weights = cfg.cost
    drift_weight = weights.drift
    attitude_weight = weights.attitude

    if cfg.objective.mode in TARGET_POSE_OBJECTIVE_MODES:
        drift_weight = 0.0
        attitude_weight = 0.0

    return float(
        weights.objective * components["objective"]
        + weights.target_arrival_time * components["target_arrival_time"]
        + drift_weight * components["drift"]
        + attitude_weight * components["attitude"]
        + weights.energy * components["energy"]
        + weights.smoothness * components["smoothness"]
    )

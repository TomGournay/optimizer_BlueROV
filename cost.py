from collections.abc import Callable

import numpy as np

from allocation import (
    allocation_quality_penalty,
    inward_direction_penalty,
    position_spacing_penalty,
)
from config import ProblemConfig
from models import Design, SimulationResult, VehicleModel

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


def station_keeping_sample_indices(
    simulation: SimulationResult,
    cfg: ProblemConfig,
) -> np.ndarray:
    """Return sample indices used to evaluate station-keeping behavior."""

    if cfg.objective.mode not in TARGET_POSE_OBJECTIVE_MODES:
        return np.array([], dtype=int)
    if not cfg.objective.require_station_keeping:
        return np.array([], dtype=int)

    window = float(cfg.objective.station_keeping_window)
    if window <= 0.0:
        raise ValueError("station_keeping_window must be positive")

    time = np.asarray(simulation.time, dtype=float)

    if cfg.objective.target_time is not None:
        target_index = target_sample_index(simulation, cfg)
        target_time = float(time[target_index])
        after_end = min(float(time[-1]), target_time + window)
        after_indices = np.flatnonzero((time >= target_time) & (time <= after_end))

        if after_indices.size > 1:
            return after_indices

        before_start = max(float(time[0]), target_time - window)
        return np.flatnonzero((time >= before_start) & (time <= target_time))

    target_position = np.asarray(cfg.objective.target_position, dtype=float)
    if target_position.shape != (3,):
        raise ValueError("target_position must have shape (3,)")

    distances = np.linalg.norm(simulation.eta[:, :3] - target_position, axis=1)
    reached_indices = np.flatnonzero(distances <= cfg.objective.target_tolerance)

    if reached_indices.size > 0:
        start_time = float(time[reached_indices[0]])
        end_time = min(float(time[-1]), start_time + window)
        return np.flatnonzero((time >= start_time) & (time <= end_time))

    start_time = max(float(time[0]), float(time[-1]) - window)
    return np.flatnonzero(time >= start_time)


def station_keeping_cost_components(
    simulation: SimulationResult,
    cfg: ProblemConfig,
) -> dict[str, float]:
    """Return station-keeping penalties near or after target arrival."""

    zero_components = {
        "station_position": 0.0,
        "station_attitude": 0.0,
        "station_linear_velocity": 0.0,
        "station_angular_velocity": 0.0,
    }
    sample_indices = station_keeping_sample_indices(simulation, cfg)
    if sample_indices.size == 0:
        return zero_components

    target_position = np.asarray(cfg.objective.target_position, dtype=float)
    target_attitude = np.asarray(cfg.objective.target_attitude, dtype=float)
    if target_position.shape != (3,):
        raise ValueError("target_position must have shape (3,)")
    if target_attitude.shape != (3,):
        raise ValueError("target_attitude must have shape (3,)")

    position_error = simulation.eta[sample_indices, :3] - target_position
    attitude_error = wrap_angle_error(
        simulation.eta[sample_indices, 3:6] - target_attitude
    )
    linear_velocity = simulation.nu[sample_indices, :3]
    angular_velocity = simulation.nu[sample_indices, 3:6]

    return {
        "station_position": float(np.mean(np.sum(position_error**2, axis=1))),
        "station_attitude": float(np.mean(np.sum(attitude_error**2, axis=1))),
        "station_linear_velocity": float(np.mean(np.sum(linear_velocity**2, axis=1))),
        "station_angular_velocity": float(
            np.mean(np.sum(angular_velocity**2, axis=1))
        ),
    }


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


def design_cost_components(
    design: Design | None,
    vehicle: VehicleModel | None,
    cfg: ProblemConfig,
) -> dict[str, float]:
    """Return unweighted penalties attached to the optimized thruster layout."""

    if design is None or vehicle is None:
        return {
            "allocation_quality": 0.0,
            "inward_direction": 0.0,
            "position_spacing": 0.0,
        }

    return {
        "allocation_quality": allocation_quality_penalty(
            design=design,
            vehicle=vehicle,
            min_singular_value=cfg.design.min_allocation_singular_value,
        ),
        "inward_direction": inward_direction_penalty(
            design=design,
            vehicle=vehicle,
            min_outward_dot=cfg.design.min_direction_outward_dot,
        ),
        "position_spacing": position_spacing_penalty(
            design=design,
            vehicle=vehicle,
            min_spacing=cfg.design.min_thruster_spacing,
        ),
    }


def cost_components(
    simulation: SimulationResult,
    cfg: ProblemConfig,
    design: Design | None = None,
    vehicle: VehicleModel | None = None,
) -> dict[str, float]:
    """Return unweighted cost components for diagnostics."""

    objective_components = {
        mode: objective_function(simulation, cfg)
        for mode, objective_function in OBJECTIVE_COSTS.items()
    }
    layout_components = design_cost_components(design, vehicle, cfg)
    station_components = station_keeping_cost_components(simulation, cfg)

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
        **station_components,
        **layout_components,
    }


def trajectory_cost(
    simulation: SimulationResult,
    cfg: ProblemConfig,
    design: Design | None = None,
    vehicle: VehicleModel | None = None,
) -> float:
    """Return the weighted scalar objective J for one simulated trajectory."""

    components = cost_components(simulation, cfg, design, vehicle)
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
        + weights.station_position * components["station_position"]
        + weights.station_attitude * components["station_attitude"]
        + weights.station_linear_velocity * components["station_linear_velocity"]
        + weights.station_angular_velocity * components["station_angular_velocity"]
        + weights.allocation_quality * components["allocation_quality"]
        + weights.inward_direction * components["inward_direction"]
        + weights.position_spacing * components["position_spacing"]
    )

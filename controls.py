import numpy as np

from allocation import allocate_thruster_forces_optimized
from config import ProblemConfig
from dynamics import rotation_body_to_ned, transformation_matrix
from models import ControlParams, VehicleModel
from thrusters import inverse_thrust_law, thrust_law


PID_AXES = ("north", "east", "down", "roll", "pitch", "yaw")


def segment_indices_for_time(time: np.ndarray, duration: float, n_segments: int) -> np.ndarray:
    """Map each time sample to its piecewise-constant control segment."""

    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    if duration <= 0.0:
        raise ValueError("duration must be positive")

    normalized_time = np.asarray(time, dtype=float) / duration
    normalized_time = np.clip(normalized_time, 0.0, 1.0)
    indices = np.floor(normalized_time * n_segments).astype(int)
    return np.minimum(indices, n_segments - 1)


def piecewise_constant_parameter_names(
    n_segments: int,
    n_thrusters: int,
) -> tuple[str, ...]:
    """Return names for piecewise-constant motor command parameters."""

    return tuple(
        f"u_s{segment_id}_m{motor_id}"
        for segment_id in range(n_segments)
        for motor_id in range(n_thrusters)
    )


def pid_pose_parameter_names() -> tuple[str, ...]:
    """Return parameter names for PID pose control."""

    return tuple(
        f"{gain}_{axis}"
        for gain in ("kp", "ki", "kd")
        for axis in PID_AXES
    )


def control_parameter_names(
    cfg: ProblemConfig,
    vehicle: VehicleModel,
) -> tuple[str, ...]:
    """Return optimizer parameter names for the selected control mode."""

    if cfg.control.mode == "piecewise_constant":
        return piecewise_constant_parameter_names(
            n_segments=cfg.control.n_segments,
            n_thrusters=vehicle.geometry.n_thrusters,
        )
    if cfg.control.mode == "pid_pose":
        return pid_pose_parameter_names()

    raise ValueError(f"unknown control mode: {cfg.control.mode}")


def n_control_parameters(cfg: ProblemConfig, vehicle: VehicleModel) -> int:
    """Return the number of scalar control parameters."""

    return len(control_parameter_names(cfg, vehicle))


def control_bounds(
    cfg: ProblemConfig,
    vehicle: VehicleModel,
) -> list[tuple[float, float]]:
    """Return optimizer bounds for the selected control mode."""

    if cfg.control.mode == "piecewise_constant":
        return [
            (cfg.control.u_min, cfg.control.u_max)
            for _ in range(n_control_parameters(cfg, vehicle))
        ]
    if cfg.control.mode == "pid_pose":
        kp_bounds = [(0.0, 50.0) for _ in PID_AXES]
        ki_bounds = [(0.0, 10.0) for _ in PID_AXES]
        kd_bounds = [(0.0, 30.0) for _ in PID_AXES]
        return kp_bounds + ki_bounds + kd_bounds

    raise ValueError(f"unknown control mode: {cfg.control.mode}")


def decode_control_params(
    flat_parameters: np.ndarray,
    cfg: ProblemConfig,
    vehicle: VehicleModel,
) -> ControlParams:
    """Convert a flat optimizer vector slice into ControlParams."""

    names = control_parameter_names(cfg, vehicle)
    values = np.asarray(flat_parameters, dtype=float)

    if values.size != len(names):
        raise ValueError(f"expected {len(names)} control values, got {values.size}")

    return ControlParams(
        mode=cfg.control.mode,
        names=names,
        values=tuple(values.tolist()),
    )


def piecewise_constant_segment_commands(
    control_params: ControlParams,
    cfg: ProblemConfig,
    vehicle: VehicleModel,
) -> np.ndarray:
    """Return segment command matrix for piecewise-constant control."""

    if control_params.mode != "piecewise_constant":
        raise ValueError("control_params mode must be piecewise_constant")

    values = control_params.as_array()
    expected_size = cfg.control.n_segments * vehicle.geometry.n_thrusters

    if values.size != expected_size:
        raise ValueError(f"expected {expected_size} control values, got {values.size}")

    return values.reshape((cfg.control.n_segments, vehicle.geometry.n_thrusters))


def piecewise_constant_commands(
    control_params: ControlParams,
    time: np.ndarray,
    cfg: ProblemConfig,
    vehicle: VehicleModel,
) -> np.ndarray:
    """Build motor commands u(t) from segment-wise command parameters."""

    segment_commands = piecewise_constant_segment_commands(control_params, cfg, vehicle)
    segment_indices = segment_indices_for_time(
        time=time,
        duration=cfg.simulation.duration,
        n_segments=cfg.control.n_segments,
    )
    return segment_commands[segment_indices]


def default_control_params(cfg: ProblemConfig, vehicle: VehicleModel) -> ControlParams:
    """Return deterministic demo parameters for the selected control mode."""

    names = control_parameter_names(cfg, vehicle)

    if cfg.control.mode == "piecewise_constant":
        values = np.zeros(len(names), dtype=float)
        return ControlParams(mode=cfg.control.mode, names=names, values=tuple(values.tolist()))

    if cfg.control.mode == "pid_pose":
        kp = np.array([8.0, 8.0, 8.0, 4.0, 4.0, 4.0], dtype=float)
        ki = np.zeros(6, dtype=float)
        kd = np.array([6.0, 6.0, 6.0, 2.0, 2.0, 2.0], dtype=float)
        values = np.concatenate((kp, ki, kd))
        return ControlParams(mode=cfg.control.mode, names=names, values=tuple(values.tolist()))

    raise ValueError(f"unknown control mode: {cfg.control.mode}")


def initial_control_runtime_state(cfg: ProblemConfig) -> dict[str, np.ndarray]:
    """Return mutable runtime state used by feedback controllers."""

    if cfg.control.mode == "pid_pose":
        return {
            "integral_error": np.zeros(6, dtype=float),
            "allocated_forces": np.array([], dtype=float),
        }
    return {}


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def pid_pose_gains(control_params: ControlParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return kp, ki, kd arrays for pose PID control."""

    values = control_params.as_array()
    if values.size != 18:
        raise ValueError(f"pid_pose expects 18 gains, got {values.size}")

    return values[:6], values[6:12], values[12:18]


def pid_pose_command(
    time: float,
    state: np.ndarray,
    control_params: ControlParams,
    cfg: ProblemConfig,
    vehicle: VehicleModel,
    allocation_matrix: np.ndarray,
    control_state: dict[str, np.ndarray],
    dt: float,
) -> np.ndarray:
    """Compute a PID pose-control motor command."""

    del time

    eta = np.asarray(state[:6], dtype=float)
    nu = np.asarray(state[6:], dtype=float)

    reference_eta = np.asarray(cfg.objective.target_pose, dtype=float)

    error = reference_eta - eta
    error[3:6] = wrap_angle(error[3:6])

    integral_error = control_state.setdefault("integral_error", np.zeros(6, dtype=float))
    integral_error += error * dt
    integral_error[:] = np.clip(
        integral_error,
        -cfg.control.pid_integral_limit,
        cfg.control.pid_integral_limit,
    )

    eta_dot = transformation_matrix(eta) @ nu
    error_dot = -eta_dot

    kp, ki, kd = pid_pose_gains(control_params)
    generalized_effort = kp * error + ki * integral_error + kd * error_dot

    roll, pitch, yaw = eta[3:6]
    ned_to_body = rotation_body_to_ned(roll, pitch, yaw).T
    force_body = ned_to_body @ generalized_effort[:3]
    moment_body = generalized_effort[3:6]
    tau_desired = np.concatenate((force_body, moment_body))

    thrust_limits = thrust_law(np.array([cfg.control.u_min, cfg.control.u_max]))
    force_min = float(np.min(thrust_limits))
    force_max = float(np.max(thrust_limits))

    initial_forces = control_state.get("allocated_forces")
    if initial_forces is not None and initial_forces.size != allocation_matrix.shape[1]:
        initial_forces = None

    desired_thrusts = allocate_thruster_forces_optimized(
        allocation_matrix=allocation_matrix,
        tau_desired=tau_desired,
        force_min=force_min,
        force_max=force_max,
        initial_forces=initial_forces,
    )
    control_state["allocated_forces"] = desired_thrusts.copy()
    return inverse_thrust_law(desired_thrusts)


def motor_command_at_time(
    time: float,
    state: np.ndarray,
    control_params: ControlParams,
    cfg: ProblemConfig,
    vehicle: VehicleModel,
    allocation_matrix: np.ndarray | None = None,
    control_state: dict[str, np.ndarray] | None = None,
    dt: float | None = None,
) -> np.ndarray:
    """Return the motor command at one simulation time and state."""

    if control_params.mode == "piecewise_constant":
        segment_commands = piecewise_constant_segment_commands(
            control_params,
            cfg,
            vehicle,
        )
        segment_id = segment_indices_for_time(
            np.array([time]),
            cfg.simulation.duration,
            cfg.control.n_segments,
        )[0]
        return segment_commands[segment_id].copy()

    if control_params.mode == "pid_pose":
        if allocation_matrix is None:
            raise ValueError("pid_pose requires allocation_matrix")
        if control_state is None:
            raise ValueError("pid_pose requires control_state")
        if dt is None:
            raise ValueError("pid_pose requires dt")

        return pid_pose_command(
            time=time,
            state=state,
            control_params=control_params,
            cfg=cfg,
            vehicle=vehicle,
            allocation_matrix=allocation_matrix,
            control_state=control_state,
            dt=dt,
        )

    raise ValueError(f"unknown control mode: {control_params.mode}")

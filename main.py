import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from allocation import (
    allocate_thruster_forces_optimized,
    allocation_singular_values,
    build_allocation_matrix,
    direction_outward_alignment,
    thruster_directions,
    thruster_positions,
)
from config import ProblemConfig
from controls import control_parameter_names, default_control_params
from cost import (
    TARGET_POSE_OBJECTIVE_MODES,
    available_objective_modes,
    cost_components,
    target_sample_index,
    trajectory_cost,
)
from models import ControlParams, Design
from optimization import run_optimization
from plots import save_optimization_plots, save_simulation_plots
from simulation import simulate
from thrusters import inverse_thrust_law, thrust_law
from vehicle import default_vehicle_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run 6-thruster vectorized design-control simulation or optimization."
    )
    parser.add_argument(
        "mode",
        choices=("simulate", "optimize"),
        nargs="?",
        default="simulate",
        help="Run one demonstration simulation or the full optimizer.",
    )
    parser.add_argument(
        "--objective",
        choices=available_objective_modes(),
        default=None,
        help="Override the objective mode from config.py.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("piecewise_constant", "pid_pose"),
        default=None,
        help="Override the control mode from config.py.",
    )
    parser.add_argument(
        "--target-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Target inertial position [x/north, y/east, z/down].",
    )
    parser.add_argument(
        "--target-attitude",
        nargs=3,
        type=float,
        metavar=("PHI", "THETA", "PSI"),
        help="Target attitude [roll, pitch, yaw] in radians.",
    )
    parser.add_argument(
        "--target-attitude-deg",
        nargs=3,
        type=float,
        metavar=("PHI", "THETA", "PSI"),
        help="Target attitude [roll, pitch, yaw] in degrees.",
    )
    parser.add_argument(
        "--target-pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "PHI", "THETA", "PSI"),
        help="Target pose [x, y, z, roll, pitch, yaw], angles in radians.",
    )
    parser.add_argument(
        "--target-pose-deg",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "PHI", "THETA", "PSI"),
        help="Target pose [x, y, z, roll, pitch, yaw], angles in degrees.",
    )
    parser.add_argument(
        "--target-time",
        type=float,
        default=None,
        help="Time at which the target pose should be imposed. Defaults to final time.",
    )
    parser.add_argument(
        "--station-keeping",
        action="store_true",
        help="Penalize motion and target error over a holding window near arrival.",
    )
    parser.add_argument(
        "--station-window",
        type=float,
        default=None,
        help="Station-keeping window duration in seconds.",
    )
    parser.add_argument(
        "--station-time-objective",
        action="store_true",
        help="Minimize the first time at which station keeping succeeds.",
    )
    parser.add_argument(
        "--station-position-tolerance",
        type=float,
        default=None,
        help="Position tolerance used to declare station-keeping success.",
    )
    parser.add_argument(
        "--station-attitude-tolerance-deg",
        type=float,
        default=None,
        help="Attitude tolerance in degrees used to declare station-keeping success.",
    )
    parser.add_argument(
        "--station-linear-velocity-tolerance",
        type=float,
        default=None,
        help="Linear velocity tolerance used to declare station-keeping success.",
    )
    parser.add_argument(
        "--station-angular-velocity-tolerance-deg",
        type=float,
        default=None,
        help="Angular velocity tolerance in deg/s used to declare station-keeping success.",
    )
    parser.add_argument(
        "--command",
        type=float,
        default=0.5,
        help="Forward desired wrench fraction used for simulate mode.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override simulation duration.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override simulation time step.",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=None,
        help="Override number of piecewise-constant control segments.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override optimizer maximum iterations.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Override optimizer population size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override optimizer random seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where plots are saved.",
    )
    parser.add_argument(
        "--candidate-log",
        default=None,
        help="JSONL path for optimizer candidate logs. Defaults to output-dir/candidate_log.jsonl.",
    )
    parser.add_argument(
        "--no-candidate-log",
        action="store_true",
        help="Disable optimizer candidate logging.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not save plots.",
    )
    args = parser.parse_args()

    full_pose_args = [args.target_pose is not None, args.target_pose_deg is not None]
    partial_pose_args = [
        args.target_position is not None,
        args.target_attitude is not None,
        args.target_attitude_deg is not None,
    ]

    if sum(full_pose_args) > 1:
        parser.error("use only one of --target-pose and --target-pose-deg")
    if sum(full_pose_args) == 1 and any(partial_pose_args):
        parser.error("--target-pose cannot be combined with partial target arguments")
    if args.target_attitude is not None and args.target_attitude_deg is not None:
        parser.error("use only one of --target-attitude and --target-attitude-deg")
    if args.station_window is not None and args.station_window <= 0.0:
        parser.error("--station-window must be positive")
    if (
        args.station_position_tolerance is not None
        and args.station_position_tolerance <= 0.0
    ):
        parser.error("--station-position-tolerance must be positive")
    if (
        args.station_attitude_tolerance_deg is not None
        and args.station_attitude_tolerance_deg <= 0.0
    ):
        parser.error("--station-attitude-tolerance-deg must be positive")
    if (
        args.station_linear_velocity_tolerance is not None
        and args.station_linear_velocity_tolerance <= 0.0
    ):
        parser.error("--station-linear-velocity-tolerance must be positive")
    if (
        args.station_angular_velocity_tolerance_deg is not None
        and args.station_angular_velocity_tolerance_deg <= 0.0
    ):
        parser.error("--station-angular-velocity-tolerance-deg must be positive")

    return args


def config_from_args(args: argparse.Namespace) -> ProblemConfig:
    """Build a ProblemConfig with command-line overrides."""

    cfg = ProblemConfig()

    has_target_override = any(
        value is not None
        for value in (
            args.target_position,
            args.target_attitude,
            args.target_attitude_deg,
            args.target_pose,
            args.target_pose_deg,
            args.target_time,
        )
    )

    if args.objective is not None or has_target_override:
        objective_updates = {}
        if args.objective is not None:
            objective_updates["mode"] = args.objective
        elif has_target_override:
            objective_updates["mode"] = "target_pose"

        if args.target_pose is not None:
            target_pose = np.array(args.target_pose, dtype=float)
            objective_updates["target_position"] = target_pose[:3]
            objective_updates["target_attitude"] = target_pose[3:6]

        if args.target_pose_deg is not None:
            target_pose = np.array(args.target_pose_deg, dtype=float)
            objective_updates["target_position"] = target_pose[:3]
            objective_updates["target_attitude"] = np.deg2rad(target_pose[3:6])

        if args.target_position is not None:
            objective_updates["target_position"] = np.array(args.target_position, dtype=float)
        if args.target_attitude is not None:
            objective_updates["target_attitude"] = np.array(args.target_attitude, dtype=float)
        if args.target_attitude_deg is not None:
            objective_updates["target_attitude"] = np.deg2rad(
                np.array(args.target_attitude_deg, dtype=float)
            )
        if args.target_time is not None:
            objective_updates["target_time"] = args.target_time

        cfg = replace(
            cfg,
            objective=replace(cfg.objective, **objective_updates),
        )

    station_updates = {}
    if args.station_keeping or args.station_time_objective:
        station_updates["require_station_keeping"] = True
    if args.station_time_objective:
        station_updates["station_time_objective"] = True
    if args.station_window is not None:
        station_updates["station_keeping_window"] = args.station_window
    if args.station_position_tolerance is not None:
        station_updates["station_position_tolerance"] = args.station_position_tolerance
    if args.station_attitude_tolerance_deg is not None:
        station_updates["station_attitude_tolerance"] = np.deg2rad(
            args.station_attitude_tolerance_deg
        )
    if args.station_linear_velocity_tolerance is not None:
        station_updates["station_linear_velocity_tolerance"] = (
            args.station_linear_velocity_tolerance
        )
    if args.station_angular_velocity_tolerance_deg is not None:
        station_updates["station_angular_velocity_tolerance"] = np.deg2rad(
            args.station_angular_velocity_tolerance_deg
        )
    if station_updates:
        cfg = replace(
            cfg,
            objective=replace(cfg.objective, **station_updates),
        )

    if args.duration is not None:
        cfg = replace(
            cfg,
            simulation=replace(cfg.simulation, duration=args.duration),
        )

    if args.dt is not None:
        cfg = replace(
            cfg,
            simulation=replace(cfg.simulation, dt=args.dt),
        )

    if args.segments is not None:
        cfg = replace(
            cfg,
            control=replace(cfg.control, n_segments=args.segments),
        )

    if args.control_mode is not None:
        cfg = replace(
            cfg,
            control=replace(cfg.control, mode=args.control_mode),
        )

    optimizer_updates = {}
    if args.max_iterations is not None:
        optimizer_updates["max_iterations"] = args.max_iterations
    if args.population_size is not None:
        optimizer_updates["population_size"] = args.population_size
    if args.seed is not None:
        optimizer_updates["random_seed"] = args.seed
    if optimizer_updates:
        cfg = replace(cfg, optimizer=replace(cfg.optimizer, **optimizer_updates))

    return cfg


def design_from_defaults(cfg: ProblemConfig) -> Design:
    """Build a deterministic vectorized thruster design for simulate mode."""

    return Design(
        names=cfg.design.names,
        values=tuple(cfg.design.defaults.tolist()),
    )


def forward_demo_control_params(
    cfg: ProblemConfig,
    vehicle,
    design: Design,
    command_magnitude: float,
) -> ControlParams:
    """Build a simple open-loop command that requests forward body force."""

    if cfg.control.mode != "piecewise_constant":
        return default_control_params(cfg, vehicle)

    allocation_matrix = build_allocation_matrix(design, vehicle)
    thrust_limits = thrust_law(np.array([cfg.control.u_min, cfg.control.u_max]))
    force_min = float(np.min(thrust_limits))
    force_max = float(np.max(thrust_limits))

    tau_desired = np.zeros(6, dtype=float)
    tau_desired[0] = command_magnitude * force_max
    desired_thrusts = allocate_thruster_forces_optimized(
        allocation_matrix=allocation_matrix,
        tau_desired=tau_desired,
        force_min=force_min,
        force_max=force_max,
    )
    command = inverse_thrust_law(desired_thrusts)
    commands = np.tile(command, (cfg.control.n_segments, 1))

    return ControlParams(
        mode=cfg.control.mode,
        names=control_parameter_names(cfg, vehicle),
        values=tuple(commands.reshape(-1).tolist()),
    )


def print_design_summary(design: Design, vehicle) -> None:
    """Print normalized thruster positions, directions, and allocation quality."""

    positions = thruster_positions(design, vehicle)
    directions = thruster_directions(design, vehicle)
    alignment = direction_outward_alignment(design, vehicle)
    singular_values = allocation_singular_values(design, vehicle)

    print(f"Design variables: {len(design.values)} raw vector components")
    print("Thruster layout [body frame, same axes as NED at zero attitude]:")
    for motor_id, (position, direction, dot_value) in enumerate(
        zip(positions, directions, alignment)
    ):
        position_text = np.array2string(position, precision=4, suppress_small=True)
        direction_text = np.array2string(direction, precision=4, suppress_small=True)
        print(
            f"  M{motor_id}: pos={position_text}, "
            f"dir={direction_text}, outward_dot={dot_value:.4f}"
        )
    print("Allocation singular values:")
    print("  " + np.array2string(singular_values, precision=4, suppress_small=True))


def print_simulation_summary(
    simulation,
    cfg: ProblemConfig,
    design: Design,
    vehicle,
    cost: float,
) -> None:
    """Print key simulation diagnostics."""

    final_eta = simulation.eta[-1]
    components = cost_components(simulation, cfg, design, vehicle)

    print_design_summary(design, vehicle)

    print("Final eta [north, east, down, roll, pitch, yaw]:")
    print("  " + np.array2string(final_eta, precision=4, suppress_small=True))

    if cfg.objective.mode in TARGET_POSE_OBJECTIVE_MODES:
        target_index = target_sample_index(simulation, cfg)
        target_time = simulation.time[target_index]
        target_eta = simulation.eta[target_index]
        print(f"Eta at target time {target_time:.3f} s:")
        print("  " + np.array2string(target_eta, precision=4, suppress_small=True))
        print("Target eta [x, y, z, roll, pitch, yaw]:")
        print(
            "  "
            + np.array2string(
                cfg.objective.target_pose,
                precision=4,
                suppress_small=True,
            )
        )
        print("Drift and attitude trajectory penalties are disabled for this objective.")
        if cfg.objective.require_station_keeping:
            print(
                "Station keeping enabled over "
                f"{cfg.objective.station_keeping_window:.3f} s."
            )
        if cfg.objective.station_time_objective:
            print("Station time objective enabled:")
            print(
                "  success tolerances: "
                f"position <= {cfg.objective.station_position_tolerance:.4f} m, "
                f"attitude <= {np.rad2deg(cfg.objective.station_attitude_tolerance):.3f} deg, "
                f"linear velocity <= {cfg.objective.station_linear_velocity_tolerance:.4f} m/s, "
                f"angular velocity <= "
                f"{np.rad2deg(cfg.objective.station_angular_velocity_tolerance):.3f} deg/s"
            )

    print("Cost components:")
    for name, value in components.items():
        print(f"  {name}: {value:.6f}")

    print(f"Total cost: {cost:.6f}")


def run_simulation_mode(cfg: ProblemConfig, args: argparse.Namespace) -> None:
    """Run one deterministic demonstration simulation."""

    vehicle = default_vehicle_model()
    design = design_from_defaults(cfg)
    control_params = forward_demo_control_params(
        cfg=cfg,
        vehicle=vehicle,
        design=design,
        command_magnitude=args.command,
    )

    simulation = simulate(design, control_params, cfg, vehicle)
    cost = trajectory_cost(simulation, cfg, design, vehicle)

    print("Mode: simulate")
    print(f"Objective mode: {cfg.objective.mode}")
    print(f"Control mode: {cfg.control.mode}")
    print_simulation_summary(simulation, cfg, design, vehicle, cost)

    if not args.no_plots:
        save_simulation_plots(simulation, Path(args.output_dir), design, vehicle)
        print(f"Plots saved in: {Path(args.output_dir).resolve()}")


def run_optimization_mode(cfg: ProblemConfig, args: argparse.Namespace) -> None:
    """Run differential evolution optimization."""

    vehicle = default_vehicle_model()
    candidate_log_path = None
    if not args.no_candidate_log:
        if args.candidate_log is not None:
            candidate_log_path = Path(args.candidate_log)
        else:
            candidate_log_path = Path(args.output_dir) / "candidate_log.jsonl"

    result = run_optimization(cfg, vehicle, candidate_log_path=candidate_log_path)

    print("Mode: optimize")
    print(f"Objective mode: {cfg.objective.mode}")
    print(f"Control mode: {cfg.control.mode}")
    if candidate_log_path is not None:
        print(f"Candidate log saved in: {candidate_log_path.resolve()}")
    print_simulation_summary(result.simulation, cfg, result.design, vehicle, result.cost)

    if not args.no_plots:
        save_optimization_plots(result, vehicle, Path(args.output_dir))
        print(f"Plots saved in: {Path(args.output_dir).resolve()}")


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)

    if args.mode == "simulate":
        run_simulation_mode(cfg, args)
    elif args.mode == "optimize":
        run_optimization_mode(cfg, args)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

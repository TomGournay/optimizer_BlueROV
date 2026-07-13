import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from allocation import thruster_directions
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
from vehicle import default_vehicle_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run BlueROV2 design-control simulation or optimization."
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
        "--alpha-deg",
        type=float,
        default=45.0,
        help="Alpha value used for simulate mode.",
    )
    parser.add_argument(
        "--command",
        type=float,
        default=0.5,
        help="Horizontal command magnitude used for simulate mode.",
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


def design_from_alpha(cfg: ProblemConfig, alpha_rad: float) -> Design:
    """Build a Design for simulate mode using alpha and midpoint defaults."""

    if "alpha" not in cfg.design.names:
        raise ValueError("the current allocation model requires a design variable named 'alpha'")

    lower_bounds = np.asarray(cfg.design.lower_bounds, dtype=float)
    upper_bounds = np.asarray(cfg.design.upper_bounds, dtype=float)
    values = 0.5 * (lower_bounds + upper_bounds)
    values[cfg.design.names.index("alpha")] = alpha_rad

    return Design(names=cfg.design.names, values=tuple(values.tolist()))


def forward_demo_control_params(
    cfg: ProblemConfig,
    vehicle,
    design: Design,
    command_magnitude: float,
) -> ControlParams:
    """Build a simple open-loop command that pushes horizontal thrusters forward."""

    if cfg.control.mode != "piecewise_constant":
        return default_control_params(cfg, vehicle)

    commands = np.zeros((cfg.control.n_segments, vehicle.geometry.n_thrusters), dtype=float)
    directions = thruster_directions(design.alpha, vehicle)

    for motor_id in vehicle.geometry.horizontal_thruster_ids:
        forward_sign = np.sign(directions[motor_id, 0])
        commands[:, motor_id] = command_magnitude * forward_sign

    return ControlParams(
        mode=cfg.control.mode,
        names=control_parameter_names(cfg, vehicle),
        values=tuple(commands.reshape(-1).tolist()),
    )


def print_simulation_summary(
    simulation,
    cfg: ProblemConfig,
    design: Design,
    cost: float,
) -> None:
    """Print key simulation diagnostics."""

    final_eta = simulation.eta[-1]
    components = cost_components(simulation, cfg)

    print("Design variables:")
    for name, value in design.as_dict().items():
        print(f"  {name}: {value:.6f} rad ({np.rad2deg(value):.3f} deg)")

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

    print("Cost components:")
    for name, value in components.items():
        print(f"  {name}: {value:.6f}")

    print(f"Total cost: {cost:.6f}")


def run_simulation_mode(cfg: ProblemConfig, args: argparse.Namespace) -> None:
    """Run one deterministic demonstration simulation."""

    vehicle = default_vehicle_model()
    design = design_from_alpha(cfg, np.deg2rad(args.alpha_deg))
    control_params = forward_demo_control_params(
        cfg=cfg,
        vehicle=vehicle,
        design=design,
        command_magnitude=args.command,
    )

    simulation = simulate(design, control_params, cfg, vehicle)
    cost = trajectory_cost(simulation, cfg)

    print("Mode: simulate")
    print(f"Objective mode: {cfg.objective.mode}")
    print(f"Control mode: {cfg.control.mode}")
    print_simulation_summary(simulation, cfg, design, cost)

    if not args.no_plots:
        save_simulation_plots(simulation, Path(args.output_dir))
        print(f"Plots saved in: {Path(args.output_dir).resolve()}")


def run_optimization_mode(cfg: ProblemConfig, args: argparse.Namespace) -> None:
    """Run differential evolution optimization."""

    vehicle = default_vehicle_model()
    result = run_optimization(cfg, vehicle)

    print("Mode: optimize")
    print(f"Objective mode: {cfg.objective.mode}")
    print(f"Control mode: {cfg.control.mode}")
    print_simulation_summary(result.simulation, cfg, result.design, result.cost)

    if not args.no_plots:
        save_optimization_plots(result, Path(args.output_dir))
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

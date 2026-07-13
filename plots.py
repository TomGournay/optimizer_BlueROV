from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from allocation import direction_outward_alignment, thruster_directions, thruster_positions
from models import Design, OptimizationResult, SimulationResult, VehicleModel


def plot_trajectory_ned(simulation: SimulationResult, output_path: str | Path) -> None:
    """Save a north-east trajectory plot."""

    output_path = Path(output_path)
    fig, ax = plt.subplots()
    ax.plot(simulation.eta[:, 1], simulation.eta[:, 0])
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_title("Horizontal trajectory")
    ax.grid(True)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_position_time(simulation: SimulationResult, output_path: str | Path) -> None:
    """Save north, east, down positions over time."""

    output_path = Path(output_path)
    fig, ax = plt.subplots()
    ax.plot(simulation.time, simulation.eta[:, 0], label="north")
    ax.plot(simulation.time, simulation.eta[:, 1], label="east")
    ax.plot(simulation.time, simulation.eta[:, 2], label="down")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [m]")
    ax.set_title("Position in NED frame")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_attitude_time(simulation: SimulationResult, output_path: str | Path) -> None:
    """Save roll, pitch, yaw over time."""

    output_path = Path(output_path)
    fig, ax = plt.subplots()
    ax.plot(simulation.time, simulation.eta[:, 3], label="roll")
    ax.plot(simulation.time, simulation.eta[:, 4], label="pitch")
    ax.plot(simulation.time, simulation.eta[:, 5], label="yaw")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angle [rad]")
    ax.set_title("Attitude")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_motor_commands(simulation: SimulationResult, output_path: str | Path) -> None:
    """Save normalized motor commands over time."""

    output_path = Path(output_path)
    fig, ax = plt.subplots()
    for motor_id in range(simulation.motor_commands.shape[1]):
        ax.step(
            simulation.time,
            simulation.motor_commands[:, motor_id],
            where="post",
            label=f"motor {motor_id}",
        )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("command [-]")
    ax.set_title("Motor commands")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_motor_commands_stacked(simulation: SimulationResult, output_path: str | Path) -> None:
    """Save one motor command subplot per thruster."""

    output_path = Path(output_path)
    n_motors = simulation.motor_commands.shape[1]
    fig_height = max(6.0, 1.35 * n_motors)
    fig, axes = plt.subplots(
        n_motors,
        1,
        sharex=True,
        figsize=(10.0, fig_height),
    )

    if n_motors == 1:
        axes = [axes]

    for motor_id, ax in enumerate(axes):
        ax.step(
            simulation.time,
            simulation.motor_commands[:, motor_id],
            where="post",
            linewidth=1.5,
        )
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"M{motor_id}")
        ax.grid(True)

    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Motor commands by thruster", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_motor_commands_heatmap(simulation: SimulationResult, output_path: str | Path) -> None:
    """Save a motor command heatmap for overlap/sign diagnostics."""

    output_path = Path(output_path)
    n_motors = simulation.motor_commands.shape[1]
    fig, ax = plt.subplots(figsize=(10.0, max(4.0, 0.45 * n_motors + 2.0)))

    image = ax.imshow(
        simulation.motor_commands.T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[
            float(simulation.time[0]),
            float(simulation.time[-1]),
            -0.5,
            n_motors - 0.5,
        ],
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )

    ax.set_xlabel("time [s]")
    ax.set_ylabel("motor id")
    ax.set_title("Motor command heatmap")
    ax.set_yticks(range(n_motors))
    ax.set_yticklabels([f"M{i}" for i in range(n_motors)])
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("command [-]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_thruster_geometry_3d(
    design: Design,
    vehicle: VehicleModel,
    output_path: str | Path,
) -> None:
    """Save a 3D visualization of optimized thruster positions and directions."""

    output_path = Path(output_path)
    positions = thruster_positions(design, vehicle)
    directions = thruster_directions(design, vehicle)
    alignment = direction_outward_alignment(design, vehicle)
    radius = float(vehicle.geometry.thruster_sphere_radius)

    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 24)
    sphere_x = radius * np.outer(np.cos(u), np.sin(v))
    sphere_y = radius * np.outer(np.sin(u), np.sin(v))
    sphere_z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(
        sphere_x,
        sphere_y,
        sphere_z,
        rstride=3,
        cstride=3,
        color="0.65",
        alpha=0.35,
        linewidth=0.6,
    )

    axis_length = 1.35 * radius
    ax.quiver(0.0, 0.0, 0.0, axis_length, 0.0, 0.0, color="tab:red", linewidth=2.0)
    ax.quiver(0.0, 0.0, 0.0, 0.0, axis_length, 0.0, color="tab:green", linewidth=2.0)
    ax.quiver(0.0, 0.0, 0.0, 0.0, 0.0, axis_length, color="tab:blue", linewidth=2.0)
    ax.text(1.08 * axis_length, 0.0, 0.0, "N / x", color="tab:red")
    ax.text(0.0, 1.08 * axis_length, 0.0, "E / y", color="tab:green")
    ax.text(0.0, 0.0, 1.08 * axis_length, "D / z", color="tab:blue")

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=45,
        color="tab:blue",
        depthshade=True,
        label="motor position",
    )

    arrow_length = 0.45 * radius
    for motor_id, (position, direction, dot_value) in enumerate(
        zip(positions, directions, alignment)
    ):
        color = "tab:orange" if dot_value >= 0.0 else "tab:red"
        ax.quiver(
            position[0],
            position[1],
            position[2],
            arrow_length * direction[0],
            arrow_length * direction[1],
            arrow_length * direction[2],
            color=color,
            linewidth=2.0,
            arrow_length_ratio=0.22,
            normalize=False,
        )
        label_position = position + 0.08 * radius * position / np.linalg.norm(position)
        ax.text(
            label_position[0],
            label_position[1],
            label_position[2],
            f"M{motor_id}",
            color="0.15",
        )

    limit = 1.65 * radius
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.invert_zaxis()
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("N / body x [m]")
    ax.set_ylabel("E / body y [m]")
    ax.set_zlabel("D / body z [m]")
    ax.set_title("Optimized 6-thruster geometry")
    ax.view_init(elev=22.0, azim=-45.0)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_simulation_plots(
    simulation: SimulationResult,
    output_dir: str | Path = "outputs",
    design: Design | None = None,
    vehicle: VehicleModel | None = None,
) -> None:
    """Save the standard plot set for one simulation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_trajectory_ned(simulation, output_dir / "trajectory_ned.png")
    plot_position_time(simulation, output_dir / "position_time.png")
    plot_attitude_time(simulation, output_dir / "attitude_time.png")
    plot_motor_commands(simulation, output_dir / "motor_commands.png")
    plot_motor_commands_stacked(simulation, output_dir / "motor_commands_stacked.png")
    plot_motor_commands_heatmap(simulation, output_dir / "motor_commands_heatmap.png")
    if design is not None and vehicle is not None:
        plot_thruster_geometry_3d(design, vehicle, output_dir / "thruster_geometry_3d.png")


def save_optimization_plots(
    result: OptimizationResult,
    vehicle: VehicleModel,
    output_dir: str | Path = "outputs",
) -> None:
    """Save the standard plot set for an optimization result."""

    save_simulation_plots(result.simulation, output_dir, result.design, vehicle)

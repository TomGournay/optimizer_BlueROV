from pathlib import Path

import matplotlib.pyplot as plt

from models import OptimizationResult, SimulationResult


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


def save_simulation_plots(
    simulation: SimulationResult,
    output_dir: str | Path = "outputs",
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


def save_optimization_plots(
    result: OptimizationResult,
    output_dir: str | Path = "outputs",
) -> None:
    """Save the standard plot set for an optimization result."""

    save_simulation_plots(result.simulation, output_dir)

import numpy as np


# Odd polynomial thrust law for normalized motor commands u in [-1, 1].
# Positive u produces positive thrust along the thruster direction epsilon_i.
THRUST_POLYNOMIAL_COEFFICIENTS = {
    9: -140.3,
    7: 389.9,
    5: -404.1,
    3: 176.0,
    1: 8.9,
}


def saturate_motor_commands(commands: np.ndarray) -> np.ndarray:
    """Clamp normalized motor commands to their physical range [-1, 1]."""

    return np.clip(commands, -1.0, 1.0)


def thrust_law(commands: np.ndarray) -> np.ndarray:
    """Convert normalized motor commands into scalar thrust forces.

    The input and output have the same shape. The command is saturated before
    evaluating the polynomial.
    """

    u = saturate_motor_commands(np.asarray(commands, dtype=float))
    thrust = np.zeros_like(u, dtype=float)

    for power, coefficient in THRUST_POLYNOMIAL_COEFFICIENTS.items():
        thrust += coefficient * u**power

    return thrust


def inverse_thrust_law(thrusts: np.ndarray, n_grid: int = 2001) -> np.ndarray:
    """Approximate normalized commands that best realize desired thrusts.

    The polynomial is not strictly monotonic over [-1, 1], so this uses a dense
    grid search instead of a direct interpolation.
    """

    thrusts = np.asarray(thrusts, dtype=float)
    command_grid = np.linspace(-1.0, 1.0, n_grid)
    thrust_grid = thrust_law(command_grid)

    flat_thrusts = thrusts.reshape(-1)
    commands = np.empty_like(flat_thrusts, dtype=float)

    for index, thrust in enumerate(flat_thrusts):
        nearest = np.argmin(np.abs(thrust_grid - thrust))
        commands[index] = command_grid[nearest]

    return commands.reshape(thrusts.shape)

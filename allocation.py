import numpy as np
from scipy.optimize import minimize

from models import VehicleModel


HORIZONTAL_FORWARD_SIGNS = np.array([ 1.0,  1.0, -1.0, -1.0])
HORIZONTAL_LATERAL_SIGNS = np.array([ 1.0, -1.0, -1.0,  1.0])


VERTICAL_SIGNS = np.array([-1.0,  1.0,  1.0, -1.0])

def horizontal_thruster_directions(alpha: float) -> np.ndarray:
    """Return unit directions for the four horizontal thrusters.

    The directions follow the horizontal thruster order defined in geometry.py:
    front-left, front-right, rear-left, rear-right.
    """

    directions = np.zeros((4, 3), dtype=float)
    directions[:, 0] = HORIZONTAL_FORWARD_SIGNS * np.cos(alpha)
    directions[:, 1] = HORIZONTAL_LATERAL_SIGNS * np.sin(alpha)
    return directions


def vertical_thruster_directions() -> np.ndarray:
    """Return unit directions for the four vertical thrusters.

    In the body convention used here, z points downward. A positive vertical
    motor thrust therefore produces a positive Z force.
    """

    directions = np.zeros((4, 3), dtype=float)
    directions[:, 2] = VERTICAL_SIGNS * 1.0
    return directions


def thruster_directions(alpha: float, vehicle: VehicleModel) -> np.ndarray:
    """Return one unit direction vector per thruster."""

    directions = np.zeros_like(vehicle.geometry.thruster_positions, dtype=float)
    directions[list(vehicle.geometry.horizontal_thruster_ids)] = horizontal_thruster_directions(
        alpha
    )
    directions[list(vehicle.geometry.vertical_thruster_ids)] = vertical_thruster_directions()
    return directions


def moment_arms_from_center_of_mass(vehicle: VehicleModel) -> np.ndarray:
    """Return r_i vectors from the center of mass to each thruster."""

    return vehicle.geometry.thruster_positions - vehicle.dynamics.center_of_mass


def build_allocation_matrix(alpha: float, vehicle: VehicleModel) -> np.ndarray:
    """Build the 6 x n thruster allocation matrix T(alpha).

    Geometry, thrust directions, and the resulting wrench are expressed in the
    body frame. Each column maps one scalar thruster force into a body-frame
    wrench:

        t_i = [epsilon_i, r_i x epsilon_i]

    The wrench order is [X, Y, Z, K, M, N].
    """

    directions = thruster_directions(alpha, vehicle)
    moment_arms = moment_arms_from_center_of_mass(vehicle)
    moments = np.cross(moment_arms, directions)
    return np.vstack((directions.T, moments.T))


def default_axis_weights(allocation_matrix: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """Return normalization weights for each wrench axis.

    Axes with larger allocation authority are scaled down so forces and moments
    with different units can be handled in one least-squares objective.
    """

    axis_scales = np.sum(np.abs(allocation_matrix), axis=1)
    weights = np.zeros_like(axis_scales, dtype=float)
    controllable = axis_scales > epsilon
    weights[controllable] = 1.0 / axis_scales[controllable]
    return weights


def allocate_thruster_forces_optimized(
    allocation_matrix: np.ndarray,
    tau_desired: np.ndarray,
    force_min: float,
    force_max: float,
    axis_weights: np.ndarray | None = None,
    initial_forces: np.ndarray | None = None,
    effort_weight: float = 1e-8,
) -> np.ndarray:
    """Allocate desired wrench to bounded thruster forces.

    Solves:

        min_f 0.5 ||W (T f - tau_desired)||^2 + 0.5 rho ||f||^2
        s.t.  force_min <= f_i <= force_max

    This replaces the unconstrained pseudo-inverse allocation and respects
    motor force limits directly.
    """

    T = np.asarray(allocation_matrix, dtype=float)
    tau = np.asarray(tau_desired, dtype=float)

    if T.ndim != 2:
        raise ValueError("allocation_matrix must be 2D")
    if tau.shape != (T.shape[0],):
        raise ValueError("tau_desired size must match allocation_matrix rows")

    n_motors = T.shape[1]

    if axis_weights is None:
        axis_weights = default_axis_weights(T)
    axis_weights = np.asarray(axis_weights, dtype=float)
    if axis_weights.shape != (T.shape[0],):
        raise ValueError("axis_weights size must match allocation_matrix rows")

    weighted_T = axis_weights[:, None] * T
    weighted_tau = axis_weights * tau

    if initial_forces is None:
        initial_forces, *_ = np.linalg.lstsq(weighted_T, weighted_tau, rcond=None)
    else:
        initial_forces = np.asarray(initial_forces, dtype=float)
        if initial_forces.shape != (n_motors,):
            raise ValueError("initial_forces must have shape (n_motors,)")

    initial_forces = np.clip(initial_forces, force_min, force_max)

    def objective(forces: np.ndarray) -> float:
        residual = weighted_T @ forces - weighted_tau
        return float(0.5 * residual @ residual + 0.5 * effort_weight * forces @ forces)

    def gradient(forces: np.ndarray) -> np.ndarray:
        residual = weighted_T @ forces - weighted_tau
        return weighted_T.T @ residual + effort_weight * forces

    result = minimize(
        objective,
        initial_forces,
        jac=gradient,
        method="SLSQP",
        bounds=[(force_min, force_max)] * n_motors,
        options={"ftol": 1e-10, "maxiter": 200, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"bounded thrust allocation failed: {result.message}")

    return result.x

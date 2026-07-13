import numpy as np
from scipy.optimize import minimize

from models import Design, VehicleModel


DESIGN_VECTOR_AXES = ("x", "y", "z")
MIN_VECTOR_NORM = 1e-9


def design_vector(design: Design, prefix: str, motor_id: int) -> np.ndarray:
    """Return one raw 3D vector from design variables."""

    return np.array(
        [design.get(f"{prefix}_{axis}_m{motor_id}") for axis in DESIGN_VECTOR_AXES],
        dtype=float,
    )


def normalize_vectors(vectors: np.ndarray, label: str) -> np.ndarray:
    """Return row-wise unit vectors, rejecting near-zero rows."""

    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"{label} must have shape (n, 3)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"{label} must be finite")

    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms < MIN_VECTOR_NORM):
        raise ValueError(f"{label} contains a near-zero vector")

    return vectors / norms[:, None]


def thruster_position_unit_vectors(design: Design, vehicle: VehicleModel) -> np.ndarray:
    """Return one unit sphere location vector per thruster."""

    raw_positions = np.vstack(
        [
            design_vector(design, "pos", motor_id)
            for motor_id in range(vehicle.geometry.n_thrusters)
        ]
    )
    return normalize_vectors(raw_positions, "thruster position vectors")


def thruster_positions(design: Design, vehicle: VehicleModel) -> np.ndarray:
    """Return thruster positions projected onto the vehicle bounding sphere."""

    radius = float(vehicle.geometry.thruster_sphere_radius)
    if radius <= 0.0:
        raise ValueError("thruster_sphere_radius must be positive")

    return radius * thruster_position_unit_vectors(design, vehicle)


def thruster_directions(design: Design, vehicle: VehicleModel) -> np.ndarray:
    """Return one optimized unit direction vector per thruster."""

    raw_directions = np.vstack(
        [
            design_vector(design, "dir", motor_id)
            for motor_id in range(vehicle.geometry.n_thrusters)
        ]
    )
    return normalize_vectors(raw_directions, "thruster direction vectors")


def direction_outward_alignment(design: Design, vehicle: VehicleModel) -> np.ndarray:
    """Return dot(direction, outward sphere normal) for each thruster."""

    outward_normals = thruster_position_unit_vectors(design, vehicle)
    directions = thruster_directions(design, vehicle)
    return np.sum(outward_normals * directions, axis=1)


def moment_arms_from_center_of_mass(design: Design, vehicle: VehicleModel) -> np.ndarray:
    """Return r_i vectors from the center of mass to each thruster."""

    return thruster_positions(design, vehicle) - vehicle.dynamics.center_of_mass


def build_allocation_matrix(design: Design, vehicle: VehicleModel) -> np.ndarray:
    """Build the 6 x n thruster allocation matrix T(design).

    Geometry, thrust directions, and the resulting wrench are expressed in the
    body frame. Each column maps one scalar thruster force into a body-frame
    wrench:

        t_i = [epsilon_i, r_i x epsilon_i]

    The wrench order is [X, Y, Z, K, M, N].
    """

    directions = thruster_directions(design, vehicle)
    moment_arms = moment_arms_from_center_of_mass(design, vehicle)
    moments = np.cross(moment_arms, directions)
    return np.vstack((directions.T, moments.T))


def allocation_singular_values(design: Design, vehicle: VehicleModel) -> np.ndarray:
    """Return singular values of the design allocation matrix."""

    return np.linalg.svd(build_allocation_matrix(design, vehicle), compute_uv=False)


def allocation_quality_penalty(
    design: Design,
    vehicle: VehicleModel,
    min_singular_value: float,
) -> float:
    """Penalize allocation matrices with weak 6-DOF authority."""

    if min_singular_value <= 0.0:
        return 0.0

    singular_values = allocation_singular_values(design, vehicle)
    shortfall = max(0.0, min_singular_value - float(singular_values[-1]))
    return float((shortfall / min_singular_value) ** 2)


def inward_direction_penalty(
    design: Design,
    vehicle: VehicleModel,
    min_outward_dot: float,
) -> float:
    """Penalize directions that point too far toward the vehicle interior."""

    alignment = direction_outward_alignment(design, vehicle)
    shortfall = np.maximum(0.0, min_outward_dot - alignment)
    return float(shortfall @ shortfall)


def position_spacing_penalty(
    design: Design,
    vehicle: VehicleModel,
    min_spacing: float,
) -> float:
    """Penalize optimized thruster positions that collapse together."""

    if min_spacing <= 0.0:
        return 0.0

    positions = thruster_positions(design, vehicle)
    penalty = 0.0
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            shortfall = max(0.0, min_spacing - distance)
            penalty += (shortfall / min_spacing) ** 2

    return float(penalty)


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

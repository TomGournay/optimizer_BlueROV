import numpy as np

from models import VehicleModel


def rotation_body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the rotation matrix from body frame to inertial NED frame."""

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def angular_rate_transform(roll: float, pitch: float) -> np.ndarray:
    """Map body angular rates [p, q, r] to Euler angle rates."""

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)

    if abs(cp) < 1e-8:
        raise ValueError("pitch is too close to +/- 90 degrees for Euler angles")

    tp = np.tan(pitch)

    return np.array(
        [
            [1.0, sr * tp, cr * tp],
            [0.0, cr, -sr],
            [0.0, sr / cp, cr / cp],
        ],
        dtype=float,
    )


def transformation_matrix(eta: np.ndarray) -> np.ndarray:
    """Return J(eta), such that eta_dot = J(eta) @ nu."""

    roll, pitch, yaw = eta[3:6]
    transform = np.zeros((6, 6), dtype=float)
    transform[:3, :3] = rotation_body_to_ned(roll, pitch, yaw)
    transform[3:, 3:] = angular_rate_transform(roll, pitch)
    return transform


def damping_force(nu: np.ndarray, vehicle: VehicleModel) -> np.ndarray:
    """Return linear + quadratic damping forces in the body frame."""

    linear = vehicle.dynamics.linear_damping @ nu
    quadratic = vehicle.dynamics.quadratic_damping @ (np.abs(nu) * nu)
    return linear + quadratic


def weight_buoyancy_wrench(eta: np.ndarray, vehicle: VehicleModel) -> np.ndarray:
    """Return the physical body-frame wrench applied by weight and buoyancy."""

    roll, pitch, yaw = eta[3:6]
    body_to_ned = rotation_body_to_ned(roll, pitch, yaw)
    ned_to_body = body_to_ned.T

    weight_ned = np.array([0.0, 0.0, vehicle.dynamics.weight], dtype=float)
    buoyancy_ned = np.array([0.0, 0.0, -vehicle.dynamics.buoyancy], dtype=float)

    weight_body = ned_to_body @ weight_ned
    buoyancy_body = ned_to_body @ buoyancy_ned

    weight_moment = np.cross(vehicle.dynamics.center_of_mass, weight_body)
    buoyancy_moment = np.cross(
        vehicle.dynamics.center_of_buoyancy,
        buoyancy_body,
    )

    force = weight_body + buoyancy_body
    moment = weight_moment + buoyancy_moment
    return np.concatenate((force, moment))


def restoring_wrench(eta: np.ndarray, vehicle: VehicleModel) -> np.ndarray:
    """Return the restoring term g(eta) for the 6-DOF equations.

    The physical weight/buoyancy wrench is applied to the vehicle, while
    Fossen's equation writes the restoring term on the left-hand side:

        M nu_dot + ... + g(eta) = tau

    Therefore g(eta) is the opposite of the physical weight/buoyancy wrench.
    """

    return -weight_buoyancy_wrench(eta, vehicle)


def state_derivative(state: np.ndarray, tau: np.ndarray, vehicle: VehicleModel) -> np.ndarray:
    """Return the 12-state derivative for the simplified 6-DOF model."""

    eta = np.asarray(state[:6], dtype=float)
    nu = np.asarray(state[6:], dtype=float)
    tau = np.asarray(tau, dtype=float)

    eta_dot = transformation_matrix(eta) @ nu
    damping = damping_force(nu, vehicle)
    restoring = restoring_wrench(eta, vehicle)

    nu_dot = np.linalg.solve(
        vehicle.dynamics.mass_matrix,
        tau - damping - restoring,
    )

    return np.concatenate((eta_dot, nu_dot))

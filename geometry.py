import numpy as np

from models import VehicleGeometry


# Vehicle geometry is expressed in the body frame:
# x forward, y starboard/right, z downward.
#
# This body frame is consistent with the inertial NED convention:
# when roll = pitch = yaw = 0, body x aligns with north, body y aligns with
# east, and body z aligns with down.
#
# The vectorized frame uses six thrusters. Their default locations are the
# vertices of an octahedron on the vehicle bounding sphere; optimization can
# move them on that sphere through the position design variables.


N_THRUSTERS = 6
THRUSTER_SPHERE_RADIUS = 0.25

THRUSTER_NAMES = tuple(f"thruster_{motor_id}" for motor_id in range(N_THRUSTERS))


def default_vehicle_geometry() -> VehicleGeometry:
    """Return the fixed intrinsic geometry for the six-thruster frame.

    The stored positions are only a deterministic default layout and define the
    number of thrusters plus the sphere radius. During optimization, allocation.py
    rebuilds the actual positions from the design variables and projects them
    onto the same sphere.
    """

    unit_positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    thruster_positions = THRUSTER_SPHERE_RADIUS * unit_positions

    return VehicleGeometry(
        thruster_positions=thruster_positions,
        thruster_sphere_radius=THRUSTER_SPHERE_RADIUS,
    )

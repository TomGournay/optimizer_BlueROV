import numpy as np

from models import VehicleGeometry


# Vehicle geometry is expressed in the body frame:
# x forward, y starboard/right, z downward.
#
# This body frame is consistent with the inertial NED convention:
# when roll = pitch = yaw = 0, body x aligns with north, body y aligns with
# east, and body z aligns with down.
#
# Thruster order used by this project:
# 0: horizontal front-left
# 1: horizontal front-right
# 2: horizontal rear-left
# 3: horizontal rear-right
# 4: vertical front-left
# 5: vertical front-right
# 6: vertical rear-left
# 7: vertical rear-right


THRUSTER_NAMES = (
    "horizontal_front_left",
    "horizontal_front_right",
    "horizontal_rear_left",
    "horizontal_rear_right",
    "vertical_front_left",
    "vertical_front_right",
    "vertical_rear_left",
    "vertical_rear_right",
)


HORIZONTAL_THRUSTER_IDS = (0, 1, 2, 3)
VERTICAL_THRUSTER_IDS = (4, 5, 6, 7)


def default_vehicle_geometry() -> VehicleGeometry:
    """Return the fixed intrinsic geometry for the first BlueROV2 model.

    The dimensions are deliberately simple starting values. They define where
    forces are applied, not the thruster orientations or dynamic reference
    points. Orientations that depend on the design variable alpha are computed
    in allocation.py.
    """

    x_h = 0.20
    y_h = 0.16
    z_h = 0.00

    x_v = 0.16
    y_v = 0.13
    z_v = 0.00

    thruster_positions = np.array(
        [
            [x_h, -y_h, z_h],
            [x_h, y_h, z_h],
            [-x_h, -y_h, z_h],
            [-x_h, y_h, z_h],
            [x_v, -y_v, z_v],
            [x_v, y_v, z_v],
            [-x_v, -y_v, z_v],
            [-x_v, y_v, z_v],
        ],
        dtype=float,
    )

    return VehicleGeometry(
        thruster_positions=thruster_positions,
        horizontal_thruster_ids=HORIZONTAL_THRUSTER_IDS,
        vertical_thruster_ids=VERTICAL_THRUSTER_IDS,
    )

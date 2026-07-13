import numpy as np

from models import VehicleDynamicsParameters


def default_vehicle_dynamics_parameters() -> VehicleDynamicsParameters:
    """Return first-pass 6-DOF dynamic parameters for the vehicle.

    These values are placeholders for the initial software architecture. They
    should be replaced by identified BlueROV2 parameters before interpreting
    simulation results physically.
    """

    mass = 11.5
    inertia_xx = 0.16
    inertia_yy = 0.16
    inertia_zz = 0.16

    rigid_body_mass_matrix = np.diag(
        [mass, mass, mass, inertia_xx, inertia_yy, inertia_zz]
    )

    added_mass_matrix = np.diag([5.5, 12.7, 14.6, 0.12, 0.12, 0.12])

    linear_damping = np.diag([4.0, 6.0, 7.0, 0.3, 0.3, 0.3])
    quadratic_damping = np.diag([18.0, 22.0, 25.0, 1.0, 1.0, 1.0])

    center_of_mass = np.zeros(3, dtype=float)
    center_of_buoyancy = np.zeros(3, dtype=float)

    gravity = 9.81
    weight = mass * gravity
    buoyancy = weight

    return VehicleDynamicsParameters(
        center_of_mass=center_of_mass,
        center_of_buoyancy=center_of_buoyancy,
        rigid_body_mass_matrix=rigid_body_mass_matrix,
        added_mass_matrix=added_mass_matrix,
        linear_damping=linear_damping,
        quadratic_damping=quadratic_damping,
        weight=weight,
        buoyancy=buoyancy,
    )

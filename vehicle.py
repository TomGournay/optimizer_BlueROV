from geometry import default_vehicle_geometry
from models import VehicleModel
from parameters_dynamic import default_vehicle_dynamics_parameters


def default_vehicle_model() -> VehicleModel:
    """Assemble the fixed vehicle geometry and dynamic parameters."""

    return VehicleModel(
        geometry=default_vehicle_geometry(),
        dynamics=default_vehicle_dynamics_parameters(),
    )

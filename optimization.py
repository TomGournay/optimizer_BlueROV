import numpy as np
from scipy.optimize import differential_evolution

from config import ProblemConfig
from controls import control_bounds, decode_control_params, n_control_parameters
from cost import trajectory_cost
from models import ControlParams, Design, OptimizationResult, VehicleModel
from simulation import simulate


def n_design_parameters(cfg: ProblemConfig) -> int:
    """Return the number of scalar design parameters."""

    return cfg.design.n_parameters


def n_command_parameters(cfg: ProblemConfig, vehicle: VehicleModel) -> int:
    """Return the number of scalar control parameters in the optimizer vector."""

    return n_control_parameters(cfg, vehicle)


def n_optimization_parameters(cfg: ProblemConfig, vehicle: VehicleModel) -> int:
    """Return the total size of z = [design_parameters, command_parameters]."""

    return n_design_parameters(cfg) + n_command_parameters(cfg, vehicle)


def decode_z(
    z: np.ndarray,
    cfg: ProblemConfig,
    vehicle: VehicleModel,
) -> tuple[Design, ControlParams]:
    """Decode z = [design_parameters, flat_command_parameters]."""

    z = np.asarray(z, dtype=float)
    n_design = n_design_parameters(cfg)
    expected_size = n_optimization_parameters(cfg, vehicle)

    if z.size != expected_size:
        raise ValueError(f"expected z size {expected_size}, got {z.size}")

    design_values = tuple(z[:n_design].tolist())
    flat_control_parameters = z[n_design:]

    design = Design(names=cfg.design.names, values=design_values)
    control_params = decode_control_params(
        flat_parameters=flat_control_parameters,
        cfg=cfg,
        vehicle=vehicle,
    )
    return design, control_params


def build_bounds(
    cfg: ProblemConfig,
    vehicle: VehicleModel,
) -> list[tuple[float, float]]:
    """Return optimizer bounds for z = [design_parameters, command_parameters]."""

    design_bounds = cfg.design.bounds
    return design_bounds + control_bounds(cfg, vehicle)


def objective(
    z: np.ndarray,
    cfg: ProblemConfig,
    vehicle: VehicleModel,
) -> float:
    """Return the scalar optimization objective for one candidate z."""

    try:
        design, control_params = decode_z(z, cfg, vehicle)
        simulation = simulate(design, control_params, cfg, vehicle)
        cost = trajectory_cost(simulation, cfg)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return float("inf")

    if not np.isfinite(cost):
        return float("inf")

    return float(cost)


def run_optimization(
    cfg: ProblemConfig,
    vehicle: VehicleModel,
) -> OptimizationResult:
    """Run differential evolution and return the best simulated candidate."""

    result = differential_evolution(
        func=objective,
        bounds=build_bounds(cfg, vehicle),
        args=(cfg, vehicle),
        maxiter=cfg.optimizer.max_iterations,
        popsize=cfg.optimizer.population_size,
        seed=cfg.optimizer.random_seed,
        polish=False,
    )

    design, control_params = decode_z(result.x, cfg, vehicle)
    simulation = simulate(design, control_params, cfg, vehicle)

    return OptimizationResult(
        design=design,
        control_params=control_params,
        cost=float(result.fun),
        simulation=simulation,
    )

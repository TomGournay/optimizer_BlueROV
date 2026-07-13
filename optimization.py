import json
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from config import ProblemConfig
from controls import control_bounds, decode_control_params, n_control_parameters
from cost import cost_components, trajectory_cost
from models import ControlParams, Design, OptimizationResult, VehicleModel
from simulation import simulate


def jsonable(value: Any) -> Any:
    """Convert numpy and dataclass values into strict JSON-compatible values."""

    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return None
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


class CandidateLogger:
    """Write one JSON line for each optimizer candidate evaluation."""

    def __init__(
        self,
        path: str | Path,
        cfg: ProblemConfig,
        vehicle: VehicleModel,
    ) -> None:
        self.path = Path(path)
        self.cfg = cfg
        self.vehicle = vehicle
        self.evaluation_count = 0
        self._file = None

    def __enter__(self) -> "CandidateLogger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self.write(
            {
                "type": "run_start",
                "objective": self.cfg.objective,
                "cost": self.cfg.cost,
                "design": self.cfg.design,
                "control": self.cfg.control,
                "simulation": self.cfg.simulation,
                "optimizer": self.cfg.optimizer,
                "n_thrusters": self.vehicle.geometry.n_thrusters,
            }
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, record: dict[str, Any]) -> None:
        """Append one JSON record."""

        if self._file is None:
            raise RuntimeError("candidate logger is not open")

        json.dump(jsonable(record), self._file, ensure_ascii=True, allow_nan=False)
        self._file.write("\n")
        self._file.flush()

    def log_candidate(
        self,
        status: str,
        cost: float | None,
        components: dict[str, float] | None,
        design: Design | None,
        control_params: ControlParams | None,
        error: str | None = None,
    ) -> None:
        """Append one candidate evaluation record."""

        self.evaluation_count += 1
        record: dict[str, Any] = {
            "type": "candidate",
            "evaluation": self.evaluation_count,
            "status": status,
            "cost": cost,
            "components": components or {},
        }

        if error is not None:
            record["error"] = error
        if design is not None:
            record["design"] = design.as_dict()
        if control_params is not None:
            record["control"] = control_params.as_dict()

        self.write(record)

    def log_run_end(self, success: bool, best_cost: float | None) -> None:
        """Append the final optimization summary."""

        self.write(
            {
                "type": "run_end",
                "success": success,
                "best_cost": best_cost,
                "n_evaluations": self.evaluation_count,
            }
        )


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
    candidate_logger: CandidateLogger | None = None,
) -> float:
    """Return the scalar optimization objective for one candidate z."""

    design = None
    control_params = None
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            design, control_params = decode_z(z, cfg, vehicle)
            simulation = simulate(design, control_params, cfg, vehicle)
            components = cost_components(simulation, cfg, design, vehicle)
            cost = trajectory_cost(simulation, cfg, design, vehicle)
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        if candidate_logger is not None:
            candidate_logger.log_candidate(
                status="invalid",
                cost=None,
                components=None,
                design=design,
                control_params=control_params,
                error=str(exc) or exc.__class__.__name__,
            )
        return float("inf")

    if not np.isfinite(cost):
        if candidate_logger is not None:
            candidate_logger.log_candidate(
                status="invalid",
                cost=None,
                components=None,
                design=design,
                control_params=control_params,
                error="candidate cost is not finite",
            )
        return float("inf")

    if candidate_logger is not None:
        candidate_logger.log_candidate(
            status="ok",
            cost=float(cost),
            components=components,
            design=design,
            control_params=control_params,
        )

    return float(cost)


def run_optimization(
    cfg: ProblemConfig,
    vehicle: VehicleModel,
    candidate_log_path: str | Path | None = None,
) -> OptimizationResult:
    """Run differential evolution and return the best simulated candidate."""

    logger_context = (
        nullcontext(None)
        if candidate_log_path is None
        else CandidateLogger(candidate_log_path, cfg, vehicle)
    )

    with logger_context as candidate_logger:
        result = differential_evolution(
            func=objective,
            bounds=build_bounds(cfg, vehicle),
            args=(cfg, vehicle, candidate_logger),
            maxiter=cfg.optimizer.max_iterations,
            popsize=cfg.optimizer.population_size,
            seed=cfg.optimizer.random_seed,
            polish=False,
        )

        success = bool(np.isfinite(result.fun))
        if candidate_logger is not None:
            candidate_logger.log_run_end(
                success=success,
                best_cost=float(result.fun) if success else None,
            )

    if not np.isfinite(result.fun):
        raise RuntimeError(
            "optimization did not find a feasible candidate. "
            "Try lowering PID gain bounds, increasing design penalty weights, "
            "or relaxing the target/time settings."
        )

    design, control_params = decode_z(result.x, cfg, vehicle)
    simulation = simulate(design, control_params, cfg, vehicle)

    return OptimizationResult(
        design=design,
        control_params=control_params,
        cost=float(result.fun),
        simulation=simulation,
    )

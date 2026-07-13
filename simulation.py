import numpy as np

from allocation import build_allocation_matrix
from config import ProblemConfig
from controls import initial_control_runtime_state, motor_command_at_time
from dynamics import state_derivative
from models import ControlParams, Design, SimulationResult, VehicleModel
from thrusters import thrust_law


def simulation_time_grid(cfg: ProblemConfig) -> np.ndarray:
    """Return the simulation time grid."""

    return np.linspace(
        0.0,
        cfg.simulation.duration,
        cfg.simulation.n_steps,
    )


def initial_state(cfg: ProblemConfig) -> np.ndarray:
    """Return the initial 12-state vector [eta, nu]."""

    eta0 = np.asarray(cfg.simulation.initial_eta, dtype=float)
    nu0 = np.asarray(cfg.simulation.initial_nu, dtype=float)

    if eta0.shape != (6,):
        raise ValueError("initial_eta must have shape (6,)")
    if nu0.shape != (6,):
        raise ValueError("initial_nu must have shape (6,)")

    return np.concatenate((eta0, nu0))


def rk4_step(state: np.ndarray, tau: np.ndarray, dt: float, vehicle: VehicleModel) -> np.ndarray:
    """Advance the state by one RK4 step with constant tau over the step."""

    k1 = state_derivative(state, tau, vehicle)
    k2 = state_derivative(state + 0.5 * dt * k1, tau, vehicle)
    k3 = state_derivative(state + 0.5 * dt * k2, tau, vehicle)
    k4 = state_derivative(state + dt * k3, tau, vehicle)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(
    design: Design,
    control_params: ControlParams,
    cfg: ProblemConfig,
    vehicle: VehicleModel,
) -> SimulationResult:
    """Simulate the vehicle response for one design and command history."""

    time = simulation_time_grid(cfg)
    allocation_matrix = build_allocation_matrix(design.alpha, vehicle)
    state = initial_state(cfg)
    control_state = initial_control_runtime_state(cfg)

    n_steps = time.size
    eta = np.zeros((n_steps, 6), dtype=float)
    nu = np.zeros((n_steps, 6), dtype=float)
    motor_commands = np.zeros((n_steps, vehicle.geometry.n_thrusters), dtype=float)
    motor_thrusts = np.zeros_like(motor_commands, dtype=float)
    wrenches = np.zeros((n_steps, 6), dtype=float)

    for k in range(n_steps):
        command_k = motor_command_at_time(
            time=time[k],
            state=state,
            control_params=control_params,
            cfg=cfg,
            vehicle=vehicle,
            allocation_matrix=allocation_matrix,
            control_state=control_state,
            dt=cfg.simulation.dt,
        )
        if command_k.shape != (vehicle.geometry.n_thrusters,):
            raise ValueError("motor command must have shape (n_thrusters,)")

        thrust_k = thrust_law(command_k)
        tau_k = allocation_matrix @ thrust_k

        eta[k] = state[:6]
        nu[k] = state[6:]
        motor_commands[k] = command_k
        motor_thrusts[k] = thrust_k
        wrenches[k] = tau_k

        if k < n_steps - 1:
            state = rk4_step(state, tau_k, cfg.simulation.dt, vehicle)

    return SimulationResult(
        time=time,
        eta=eta,
        nu=nu,
        motor_commands=motor_commands,
        motor_thrusts=motor_thrusts,
        wrenches=wrenches,
    )

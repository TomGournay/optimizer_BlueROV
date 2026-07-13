# Vectorized 6-thruster design-control co-optimization

This project builds a modular Python model for co-optimizing an underwater robot
with six vectorized thrusters distributed on the vehicle bounding sphere.

## Current objective

The objective is selected in `config.py` with:

```python
ProblemConfig().objective.mode
```

Available modes are registered once in `cost.py`:

```text
forward_progress
target_position
target_pose
```

The default mode is open-loop performance:

```text
maximize final north progress during the simulation time
```

Because the optimizer minimizes a scalar cost, this is implemented as:

```text
forward_progress_cost = -north_final
```

To switch to target tracking later, set:

```python
objective = ObjectiveConfig(
    mode="target_pose",
    target_position=np.array([5.0, 0.0, 0.0]),
    target_attitude=np.deg2rad(np.array([0.0, 0.0, 30.0])),
    target_time=4.0,
)
```

The total cost also keeps the existing penalties:

```text
J =
  w_objective  * active_objective
+ w_arrival    * target_arrival_time
+ w_drift      * mean(east^2 + down^2)
+ w_attitude   * mean(roll^2 + pitch^2 + yaw^2)
+ w_energy     * integral(||u||^2)
+ w_smoothness * sum(||u_k - u_{k-1}||^2)
+ w_station_p  * station_position
+ w_station_a  * station_attitude
+ w_station_v  * station_linear_velocity
+ w_station_w  * station_angular_velocity
+ w_alloc      * allocation_quality
+ w_inward     * inward_direction
+ w_spacing    * position_spacing
```

The design penalties discourage singular allocation matrices, thruster
directions that point too far inward, and optimized motor locations that collapse
onto one another.

For target-pose objectives, the global drift and attitude trajectory penalties
are disabled automatically because lateral motion and nonzero attitude can be
part of the requested maneuver.

For `forward_progress`:

```text
active_objective = -north_final
```

For `target_position` or `target_pose`:

```text
active_objective =
    target_position_weight * ||position(t*) - target_position||^2
  + target_attitude_weight * ||wrap(attitude(t*) - target_attitude)||^2
```

If `ObjectiveConfig.target_time` is `None`, `t*` is the final simulation time.
Otherwise, `t*` is the closest simulation sample to `target_time`.

When no fixed `target_time` is set, `target_arrival_time` also rewards faster
arrival. The robot is considered to have reached the target when:

```text
||position(t) - target_position|| <= target_tolerance
```

where `target_tolerance` is configured in `ObjectiveConfig`.

Station keeping can be enabled for `target_position` and `target_pose`
objectives. It penalizes target error and velocity over a holding window near
arrival, so the optimizer prefers solutions that arrive, slow down, and stay
stable instead of only crossing the target once.

## Frames and state convention

Inertial frame:

```text
NED = [north, east, down]
```

Body frame:

```text
x forward
y starboard/right
z down
```

State and wrench conventions:

```text
eta = [north, east, down, roll, pitch, yaw]
nu  = [surge, sway, heave, p, q, r]
tau = [X, Y, Z, K, M, N]
```

## File roles

```text
models.py
  Shared dataclasses: Design, ControlParams, VehicleGeometry,
  VehicleDynamicsParameters, VehicleModel, SimulationResult,
  OptimizationResult.

config.py
  Numerical problem settings: design variable names/bounds, control bounds, time grid,
  initial state, objective mode, cost weights, optimizer settings.

geometry.py
  Fixed geometry defaults: six thruster names, bounding-sphere radius, and a
  deterministic default layout on that sphere.

parameters_dynamic.py
  Dynamic vehicle parameters: center of mass, center of buoyancy, mass
  matrices, damping, weight, buoyancy.

vehicle.py
  Assembles geometry and dynamic parameters into one VehicleModel.

allocation.py
  Builds T(design): optimized thruster positions, optimized unit directions,
  moment arms, r_i x epsilon_i, layout penalties, and the final allocation
  matrix.

thrusters.py
  Converts normalized motor commands u in [-1, 1] into scalar thrust forces.

controls.py
  Defines control parameterization modes. The current mode is
  piecewise_constant. This file owns control parameter names, bounds,
  decoding, and motor command evaluation u_k = controller(...).

dynamics.py
  Simplified 6-DOF dynamics: eta_dot = J(eta) nu and
  M nu_dot = tau - damping - restoring.

simulation.py
  Runs the time simulation: u_k -> thrust -> tau_k -> RK4 integration.

cost.py
  Computes trajectory cost components and the weighted scalar objective J.

optimization.py
  Build z = [design parameters, command parameters], decode z, build bounds,
  evaluate objective, and run scipy.optimize.differential_evolution.

plots.py
  Save trajectory, position, attitude, motor command, and 3D thruster-geometry
  plots to image files.
```

## Planned files

```text
```

## Tutorial

### Run a demonstration simulation

Use this first to check that the model, dynamics, cost, and plots work:

```bash
python main.py simulate
```

By default this runs one open-loop forward command with:

```text
the default six-thruster vectorized design
forward desired wrench fraction = 0.5
```

Useful options:

```bash
python main.py simulate --command 0.4
python main.py simulate --duration 5 --dt 0.02
python main.py simulate --no-plots
```

### Run the optimizer

For a quick smoke test:

```bash
python main.py optimize --max-iterations 2 --population-size 3
```

For the default optimizer settings from `config.py`:

```bash
python main.py optimize
```

The optimizer can be expensive because the current vector is:

```text
z = [design variables, command parameters]
```

With the default configuration:

```text
36 design variables + 5 segments * 6 motors = 66 variables
```

The 36 design variables are grouped by motor:

```text
pos_x_mi, pos_y_mi, pos_z_mi
dir_x_mi, dir_y_mi, dir_z_mi
```

The raw position vector is normalized and projected onto the bounding sphere.
The raw direction vector is normalized into the thruster force direction.

The control part of `z` is generated from `ControlConfig.mode`, so future
controllers can use a different number of parameters without changing
`optimization.py`.

### Switch objective modes

Default objective:

```bash
python main.py optimize --objective forward_progress
```

This maximizes final north progress by minimizing:

```text
-north_final
```

Target pose objective:

```bash
python main.py optimize --objective target_pose --target-pose-deg 5 2 -1 10 0 45 --target-time 4
```

This minimizes:

```text
target_position_weight * ||position(t*) - target_position||^2
+ target_attitude_weight * ||wrap(attitude(t*) - target_attitude)||^2
```

Angles passed with `--target-pose-deg` or `--target-attitude-deg` are converted
to radians internally. You can also set position and attitude separately:

```bash
python main.py optimize --objective target_pose --target-position 5 2 -1 --target-attitude-deg 10 0 45 --target-time 4
```

When `--target-time` is omitted, the target pose is evaluated at the final
simulation time. In that case, the target objective also includes a fast-arrival
component:

```text
target_arrival_time = first time inside target_tolerance
```

If the target is not reached:

```text
target_arrival_time = simulation_duration + final_remaining_distance
```

Tune it in `config.py` with:

```python
ObjectiveConfig.target_tolerance
CostWeights.target_arrival_time
```

Enable station keeping to penalize residual target error and motion near arrival:

```bash
python main.py optimize --objective target_position --target-position 5 0 0 --station-keeping
python main.py optimize --objective target_pose --target-pose-deg 5 0 0 0 0 30 --station-keeping --station-window 3
```

The window is chosen after first arrival when the target tolerance is reached.
If the target is not reached, the final window is used. When `target_time` is
set, the window starts at that target time when possible.

Tune it in `config.py` with:

```python
ObjectiveConfig.require_station_keeping
ObjectiveConfig.station_keeping_window
CostWeights.station_position
CostWeights.station_attitude
CostWeights.station_linear_velocity
CostWeights.station_angular_velocity
```

### Add a new objective mode

Add one function in `cost.py` with the signature:

```python
def my_objective(simulation: SimulationResult, cfg: ProblemConfig) -> float:
    ...
```

Then register it in the `OBJECTIVE_COSTS` dictionary:

```python
OBJECTIVE_COSTS = {
    "forward_progress": forward_progress_objective,
    "target_position": target_position_objective,
    "target_pose": target_pose_objective,
    "my_objective": my_objective,
}
```

After that, it is automatically available from the CLI:

```bash
python main.py optimize --objective my_objective
```

### Change command parametrization

Change the number of piecewise-constant command segments:

```bash
python main.py optimize --segments 3
```

Or edit `ControlConfig` in `config.py`:

```python
mode = "piecewise_constant"
n_segments = 5
u_min = -1.0
u_max = 1.0
```

### Add a new controller mode

The current controller mode is:

```python
ControlConfig.mode = "piecewise_constant"
```

The first feedback controller mode is also available:

```python
ControlConfig.mode = "pid_pose"
```

From the command line:

```bash
python main.py simulate --control-mode pid_pose --objective target_pose --target-pose-deg 5 0 0 0 0 30
python main.py optimize --control-mode pid_pose --objective target_pose --target-pose-deg 5 0 0 0 0 30
```

`pid_pose` optimizes or uses PID gains:

```text
kp_north, kp_east, kp_down, kp_roll, kp_pitch, kp_yaw
ki_north, ki_east, ki_down, ki_roll, ki_pitch, ki_yaw
kd_north, kd_east, kd_down, kd_roll, kd_pitch, kd_yaw
```

The controller tracks:

```text
eta = cfg.objective.target_pose
```

It computes a desired body wrench, allocates it through `T(design)`, and
converts desired thrusts to normalized motor commands.

The wrench-to-thruster allocation does not use an unconstrained pseudo-inverse.
It solves a bounded weighted least-squares problem:

```text
min_f 0.5 ||W (T(design) f - tau_desired)||^2 + 0.5 rho ||f||^2
subject to f_min <= f_i <= f_max
```

where `f_min` and `f_max` come from `thrust_law(u_min)` and
`thrust_law(u_max)`. This makes the controller respect motor force limits during
allocation instead of clipping only after an unconstrained solve.

To add another controller, for example `feedback_pd`, the intended workflow is:

1. Add the mode name to `ControlMode` in `config.py`.
2. Add parameter naming and bounds in `controls.py`.
3. Add the command law in `controls.py`, usually inside `motor_command_at_time`.
4. Use the current state if needed:

```python
u_k = controller(time, state, control_params, cfg, vehicle)
```

`simulation.py` already calls:

```python
motor_command_at_time(time[k], state, control_params, cfg, vehicle)
```

inside the simulation loop, so feedback controllers can depend on the current
state. `optimization.py` already gets the number of control parameters and their
bounds from `controls.py`, so it does not need to be edited when a new control
mode has a different parameter count.

### Vectorized thruster design variables

The default `DesignConfig` optimizes six motors with one raw position vector and
one raw direction vector per motor:

```text
M0: pos_x_m0, pos_y_m0, pos_z_m0, dir_x_m0, dir_y_m0, dir_z_m0
...
M5: pos_x_m5, pos_y_m5, pos_z_m5, dir_x_m5, dir_y_m5, dir_z_m5
```

`allocation.py` normalizes each position vector and projects it onto the
vehicle bounding sphere. It also normalizes each direction vector before
building the allocation matrix:

```text
t_i = [epsilon_i, r_i x epsilon_i]
```

The design cost includes:

```text
allocation_quality
  penalizes weak minimum singular value of T(design)

inward_direction
  penalizes dot(direction_i, outward_normal_i) below min_direction_outward_dot

position_spacing
  penalizes motor locations closer than min_thruster_spacing
```

Tune the corresponding thresholds in `DesignConfig` and their weights in
`CostWeights`.

### Outputs

Plots are saved in `outputs/` by default:

```text
trajectory_ned.png
position_time.png
attitude_time.png
motor_commands.png
motor_commands_stacked.png
motor_commands_heatmap.png
thruster_geometry_3d.png
```

## Plot usage

Once a `SimulationResult` or `OptimizationResult` is available, plots can be
saved with:

```python
from plots import save_simulation_plots, save_optimization_plots

save_simulation_plots(simulation, output_dir="outputs", design=design, vehicle=vehicle)
save_optimization_plots(result, vehicle, output_dir="outputs")
```

The generated files are:

```text
trajectory_ned.png
position_time.png
attitude_time.png
motor_commands.png
motor_commands_stacked.png
motor_commands_heatmap.png
thruster_geometry_3d.png
```

For motor diagnostics, prefer:

```text
motor_commands_stacked.png
  one subplot per motor, so overlapping curves cannot hide each other

motor_commands_heatmap.png
  color-coded motor/time matrix, useful to spot identical or opposite commands

thruster_geometry_3d.png
  3D bounding sphere, motor positions, thrust direction vectors, and the NED
  axes that coincide with the body axes at zero attitude
```

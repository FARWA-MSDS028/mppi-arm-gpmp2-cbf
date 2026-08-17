# GPMP2 + MPPI + CBF-QP Safe Motion Planning (Franka Panda, Simulation)

A layered, safety-aware motion planning and control pipeline for a
7-DOF Franka Panda arm, implemented and tested entirely in **MuJoCo
simulation**. The robot plans and executes a trajectory from a home
position to a target, actively avoiding a known obstacle, with a
mathematically enforced safety guarantee at the lowest level.

> **Important:** This project has been developed and verified **only
> in simulation**. It has never been run on a real robot. See
> "Moving to Real Hardware" below before attempting to deploy this on
> actual Franka hardware.

---

## Pipeline Overview
GPMP2 (global path planner)
|
v
MPPI (local sampling-based controller)
|
v
CBF-QP (hard safety filter)
|
v
Robot Execution
|
v
Feasibility Extraction
|
v
Conflict Factor Detection
|
v
Covariance Steering
|
v
Update GPMP2 (replan) -----> repeats
### 1. GPMP2 (Gaussian Process Motion Planner 2)
Plans a full trajectory in one shot by solving a nonlinear
least-squares optimization problem: minimize a combination of
"smoothness" and "distance from the obstacle," solved via
Levenberg-Marquardt using the [GTSAM](https://gtsam.org/) library.
Replans periodically from the robot's current real state.

### 2. MPPI (Model Predictive Path Integral Control)
A sampling-based local controller. At every control step, it samples
many random candidate short-horizon trajectories around GPMP2's
reference, scores each one, and blends them into a single
weighted-average command.

### 3. CBF-QP (Control Barrier Function, solved as a Quadratic Program)
The actual hard safety guarantee. Before any command reaches the
robot, this layer checks whether it satisfies a safety condition
`h(x) >= 0` (a mathematical distance-from-danger function), and if
not, finds the smallest possible correction that restores safety,
solved via [OSQP](https://osqp.org/) every control step. Uses a
**High-Order CBF (HOCBF)** formulation, required because the
obstacle-distance function has relative degree 2 under this system's
dynamics (a standard technical requirement for this class of problem).

### 4. Conflict Factor Detection + Covariance Steering
When the robot repeatedly approaches danger, the system inserts a
"conflict factor" back into GPMP2's own optimization graph (forcing
future replans to actively avoid that region) and temporarily shrinks
MPPI's random exploration range near that danger zone, slowly
restoring full exploration once the situation stabilizes.

### Multi-Rate, Parallel Execution
GPMP2, MPPI, and the CBF-QP/robot-execution loop each run as
**independent threads at their own natural rate** (GPMP2: slow,
fraction of a second to several seconds per solve; MPPI: ~10 Hz;
robot/safety loop: up to 500 Hz), matching how a real robotic system
must be structured -- a slow global planner cannot be allowed to block
a fast safety loop.

---

## What Has Been Verified

- The robot reliably reaches the target (goal error below threshold)
  across many independent test runs.
- The CBF-QP dramatically reduces unsafe motion compared to running
  MPPI with no safety filter (directly measured, not assumed).
- The three-thread architecture runs correctly in parallel, with the
  robot/safety loop never blocked by the slower planning stages.
- **Known limitation:** in the complete real-time system, the safety
  value `h(x)` can very briefly dip slightly negative (typically
  -0.001 to -0.004) during the fastest part of the motion, before
  recovering. This was investigated with a full parameter sweep and
  traced to a known, documented limitation of enforcing a continuous-
  time safety constraint at discrete time intervals rather than
  continuously -- a legitimate topic for further work, not an
  unexplained bug.

---

## Setup

```bash
git clone https://github.com/FARWA-MSDS028/mppi-arm-gpmp2-cbf.git
cd mppi-arm-gpmp2-cbf
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Robot model assets (not included in this repo -- too large)

Download the Franka Panda model from MuJoCo Menagerie:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie
mkdir -p assets/assets
cp -r /tmp/menagerie/franka_emika_panda/assets/* assets/assets/
```

(`assets/panda.xml` itself is already included in this repo.)

---

## Running It

**Full closed-loop pipeline (headless, prints results):**
```bash
python main.py --mjcf assets/panda.xml --cycles 20 \
    --obstacle-x 0.20 --obstacle-y 0.09 --obstacle-z 0.85 \
    --obstacle-radius 0.08 --d-safe 0.03 --gpmp2-eps 0.02 \
    --tau-conflict 0.05 --tau-safe 0.15 --lambda-cbf 0.3
```

**Live 3D visual demo (opens a MuJoCo window):**
```bash
python demo_dashboard.py --mjcf assets/panda.xml --cycles 20 \
    --obstacle-x 0.20 --obstacle-y 0.09 --obstacle-z 0.85 \
    --obstacle-radius 0.08 --d-safe 0.03 --gpmp2-eps 0.02 \
    --tau-conflict 0.05 --tau-safe 0.15 --lambda-cbf 0.3
```

The demo shows: white sphere (home), orange sphere (obstacle), green
sphere (target), a yellow trail of the robot's real executed path
(turning red wherever the CBF-QP made a safety correction), and pink
markers where a conflict was detected. It holds at home briefly while
GPMP2 plans, then moves, then freezes and prints `TARGET REACHED` once
the goal is reached. Close the window or press Ctrl+C to exit.

A recorded video demo is available separately (shared outside this
repository due to file size).

---

## Which Files Matter

**Core pipeline** (this is the actual project):
main.py
planner/factor_graph.py, gpmp2_planner.py, conflict_factor.py, isam_update.py
controller/mppi.py, cost.py, sampling.py, covariance.py
cbf/barrier.py, qp_solver.py, feasibility.py
robot/franka.py, mujoco_env.py
assets/panda.xml
requirements.txt
**Live visual demo:**
demo_dashboard.py
demo/dashboard_state.py, threaded_pipeline.py, gpmp2_process.py
**Everything else** (files starting with `test_`, `check_`, `diagnose_`,
`debug_`, `step`, `find_`, `stage`, `sweep_`, `verify_`, `compare_`,
`fix_`) are one-off debugging and verification scripts used during
development. They document how specific bugs were found and fixed,
but are not required to run or understand the final pipeline.

---

## Moving to Real Hardware -- Read Before Attempting

This is simulation-only. Several real, non-trivial gaps must be
addressed before this can run on physical hardware:

1. **No real robot interface exists in this code.** A real Panda is
   controlled via `libfranka` (C++) or `franka_ros`/`franka_ros2`,
   over Franka's FCI (Franka Control Interface, up to 1kHz). None of
   this project's Python/MuJoCo code will run on real hardware as-is
   -- a real hardware bridge must be written.

2. **Control mode must be explicitly verified.** This simulation's
   Panda model uses **position-controlled actuators with fixed
   internal gains**, not raw torque control -- discovered through
   direct testing during development. On real hardware, the chosen
   control mode (position, velocity, or torque) must be confirmed and
   the CBF-QP's dynamics model (`cbf/barrier.py`, `franka_dynamics`)
   must match it exactly, or the safety guarantee will not hold.

3. **The obstacle in this project is purely virtual** -- a
   mathematical value used only in the planner/CBF calculations, never
   a real physical object the simulator could collide with. A real
   deployment needs an actual perception system (camera/depth sensor)
   or precisely measured, fixed obstacle geometry, plus real
   self-collision handling -- none of which exists in this codebase.

4. **All tuned parameters (`d_safe`, `alpha_gamma`, sample counts,
   thread rates) were calibrated against simulated dynamics and this
   simulation's specific control loop timing.** They are a starting
   point only and must be re-verified against the real robot's actual
   dynamics and control loop rate before being trusted for real
   safety-critical operation.

---

## Dependencies

See `requirements.txt`. Notably: [GTSAM](https://gtsam.org/) (GPMP2
optimization), [OSQP](https://osqp.org/) (CBF-QP solver),
[MuJoCo](https://mujoco.org/) (simulation and visualization).

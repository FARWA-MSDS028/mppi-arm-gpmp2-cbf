"""
stage_validation_v2.py
Runs every stage BOTH with the real obstacle present (matching your
request to include it at every stage) and reports Stage 1 as a
COMPARISON baseline rather than a hard gate -- naive MPPI (no GPMP2
prior) is expected to struggle on a large reach; that's the reason
GPMP2 guidance exists. Stages 2 and 3 are what actually need to succeed.
"""
from __future__ import annotations
import numpy as np, mujoco

from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from cbf.barrier import DistanceBarrier

from experiments.baseline_mppi import run_baseline_mppi
from experiments.gpmp2_mppi import run_gpmp2_mppi
from experiments.safe_mppi import run_safe_mppi

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
SUCCESS_THRESHOLD = 0.15

OBSTACLE_CENTER = (0.20, 0.09, 0.85)
OBSTACLE_RADIUS = 0.08
D_SAFE = 0.03


def make_env_franka_sdf():
    env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=OBSTACLE_CENTER,
                          obstacle_radius=OBSTACLE_RADIUS)
    franka = FrankaModel(env.model, mujoco.MjData(env.model))
    sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
    env.reset(Q0)
    return env, franka, sdf


def report(stage_name, goal_err, min_clear, extra=""):
    h = min_clear - D_SAFE
    success = goal_err < SUCCESS_THRESHOLD
    safe = h >= 0
    print(f"\n{'='*72}\n{stage_name}\n{'='*72}")
    print(f"  final goal_err:      {goal_err:.4f}  (threshold: {SUCCESS_THRESHOLD})")
    print(f"  min clearance d(x):  {min_clear:.4f}")
    print(f"  worst h(q)=d-d_safe: {h:.4f}  ({'SAFE the whole run' if safe else 'VIOLATED at some point'})")
    if extra:
        print(f"  {extra}")
    print(f"  REACHED TARGET: {success}   STAYED SAFE: {safe}")
    return success, safe


print("Running all 3 stages WITH the real obstacle present. Stage 1 is "
      "reported as a comparison baseline (naive MPPI, no GPMP2 guidance) "
      "-- it is not expected to necessarily succeed; Stages 2-3 are the "
      "ones that need to.\n")

# ---- Stage 1 ----
env, franka, sdf = make_env_franka_sdf()
log1 = run_baseline_mppi(env, franka, sdf, q_goal=Q_GOAL, rng_seed=0)
report("STAGE 1 (comparison baseline): MPPI alone, WITH obstacle",
       log1["goal_error"][-1], min(log1["dist"]))

# ---- Stage 2 ----
env, franka, sdf = make_env_franka_sdf()
log2, theta_q = run_gpmp2_mppi(env, franka, sdf, q0=Q0, q_goal=Q_GOAL, rng_seed=0)
ok2, safe2 = report("STAGE 2: GPMP2 -> MPPI, WITH obstacle",
                     log2["goal_error"][-1], min(log2["dist"]))

# ---- Stage 3 ----
env, franka, sdf = make_env_franka_sdf()
barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                           sdf=sdf, dof=DOF, d_safe=D_SAFE)
h0 = barrier.forward(np.concatenate([Q0, np.zeros(DOF)]))
print(f"\nPre-flight: h(q0) = {h0:.4f}  ({'SAFE START' if h0 >= 0 else 'UNSAFE START'})")
log3, feas_log = run_safe_mppi(env, franka, sdf, barrier, q0=Q0, q_goal=Q_GOAL, rng_seed=0)
n_unsafe = sum(log3["unsafe_flags"])
n_total = len(log3["unsafe_flags"])
extra3 = f"MPPI proposed unsafe controls on {n_unsafe}/{n_total} steps (CBF-QP corrected each)"
ok3, safe3 = report("STAGE 3: GPMP2 -> MPPI -> CBF-QP, WITH obstacle",
                     log3["goal_error"][-1], min(log3["dist"]), extra3)

print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
print(f"  Stage 1 (baseline, comparison only): goal_err={log1['goal_error'][-1]:.4f}")
print(f"  Stage 2 (GPMP2+MPPI):  reached={ok2}  safe={safe2}  goal_err={log2['goal_error'][-1]:.4f}")
print(f"  Stage 3 (+CBF-QP):     reached={ok3}  safe={safe3}  goal_err={log3['goal_error'][-1]:.4f}")

"""
test_baseline_lower_sigma.py
Same as before, but with MPPI's sampling noise reduced -- testing
whether the jitter (not the plateau, which is confirmed fixed) is
caused by sigma=0.3 being calibrated for the old, wrong torque-based
model instead of direct position control.
"""
import numpy as np
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from experiments.baseline_mppi import run_baseline_mppi

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
SUCCESS_THRESHOLD = 0.15

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(5.0, 5.0, 5.0),
                       obstacle_radius=0.01)
franka = FrankaModel(env.model, env.data)
sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
env.reset(Q0)

for test_sigma in [0.05, 0.02]:
    env.reset(Q0)
    log = run_baseline_mppi(env, franka, sdf, q_goal=Q_GOAL, rng_seed=0, sigma=test_sigma)
    goal_err = log["goal_error"][-1]
    min_err = min(log["goal_error"])
    print(f"\nsigma={test_sigma}: final={goal_err:.4f}  min={min_err:.4f}  "
          f"SUCCESS={goal_err < SUCCESS_THRESHOLD}")
    print(f"  last 10 steps: {[round(e, 3) for e in log['goal_error'][-10:]]}")

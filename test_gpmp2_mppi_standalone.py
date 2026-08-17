"""
test_gpmp2_mppi_standalone.py
Stage 2 standalone: GPMP2 as MPPI's prior, still no CBF. Tests whether
GPMP2's actual path planning (not just MPPI's soft cost) meaningfully
improves safety over baseline, using the corrected sigma=0.002.
"""
import numpy as np
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from experiments.gpmp2_mppi import run_gpmp2_mppi

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
SUCCESS_THRESHOLD = 0.15
OBSTACLE_CENTER = (0.20, 0.09, 0.85)
OBSTACLE_RADIUS = 0.08
D_SAFE = 0.03

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=OBSTACLE_CENTER,
                       obstacle_radius=OBSTACLE_RADIUS)
franka = FrankaModel(env.model, env.data)
sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
env.reset(Q0)

log, theta_q = run_gpmp2_mppi(env, franka, sdf, q0=Q0, q_goal=Q_GOAL, rng_seed=0, sigma=0.002)

goal_err = log["goal_error"][-1]
min_clear = min(log["dist"])
h = min_clear - D_SAFE
print(f"final goal_err: {goal_err:.4f}  (success: {goal_err < SUCCESS_THRESHOLD})")
print(f"min_clear: {min_clear:.4f}   worst h(x): {h:.4f}  (safe: {h >= 0})")

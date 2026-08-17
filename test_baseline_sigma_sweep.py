"""
test_baseline_sigma_sweep.py
Narrows in on the best sigma for the corrected position-control model.
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

print(f"{'sigma':>8}{'final_err':>12}{'min_err':>10}{'last10_mean':>14}{'last10_std':>12}")
for test_sigma in [0.005, 0.01, 0.015, 0.02, 0.03]:
    env.reset(Q0)
    log = run_baseline_mppi(env, franka, sdf, q_goal=Q_GOAL, rng_seed=0, sigma=test_sigma)
    goal_err = log["goal_error"][-1]
    min_err = min(log["goal_error"])
    last10 = log["goal_error"][-10:]
    print(f"{test_sigma:>8}{goal_err:>12.4f}{min_err:>10.4f}{np.mean(last10):>14.4f}{np.std(last10):>12.4f}")

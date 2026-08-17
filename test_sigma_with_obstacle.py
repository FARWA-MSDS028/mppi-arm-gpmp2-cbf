"""
test_sigma_with_obstacle.py
Same sweep, but WITH the real obstacle present -- checking not just
goal_err, but whether MPPI still explores enough to route around danger.
A sigma that's great for pure reaching but too tight for obstacle
avoidance would show up here as small goal_err alongside a bad (very
negative) min_clear.
"""
import numpy as np
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from experiments.baseline_mppi import run_baseline_mppi

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

print(f"{'sigma':>8}{'final_err':>12}{'min_clear':>12}{'worst_h(x)':>12}{'success':>10}")
for test_sigma in [0.002, 0.005, 0.01, 0.02, 0.05]:
    env.reset(Q0)
    log = run_baseline_mppi(env, franka, sdf, q_goal=Q_GOAL, rng_seed=0, sigma=test_sigma)
    goal_err = log["goal_error"][-1]
    min_clear = min(log["dist"])
    h = min_clear - D_SAFE
    print(f"{test_sigma:>8}{goal_err:>12.4f}{min_clear:>12.4f}{h:>12.4f}{str(goal_err < SUCCESS_THRESHOLD):>10}")

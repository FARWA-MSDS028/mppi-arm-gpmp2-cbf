"""
test_baseline_standalone.py
Standalone test of experiments/baseline_mppi.py alone, after the
position-actuator fix. No GPMP2, no CBF -- isolates whether MPPI's own
raw rollouts, now correctly interpreted as position commands, can reach
the goal on their own.
"""
import numpy as np
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from experiments.baseline_mppi import run_baseline_mppi

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
SUCCESS_THRESHOLD = 0.15

# No obstacle -- isolate MPPI's own reachability first, same as the
# earlier staged-validation approach.
env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(5.0, 5.0, 5.0),
                       obstacle_radius=0.01)
franka = FrankaModel(env.model, env.data)
sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
env.reset(Q0)

log = run_baseline_mppi(env, franka, sdf, q_goal=Q_GOAL, rng_seed=0)

goal_err = log["goal_error"][-1]
min_err = min(log["goal_error"])
step_first_success = next((i for i, e in enumerate(log["goal_error"]) if e < SUCCESS_THRESHOLD), None)

print(f"{'step':>6}{'goal_err':>12}")
for i in range(0, len(log["goal_error"]), 20):
    print(f"{i:>6}{log['goal_error'][i]:>12.4f}")
print(f"{len(log['goal_error'])-1:>6}{log['goal_error'][-1]:>12.4f}")

print(f"\nFinal goal_err: {goal_err:.4f}")
print(f"Min goal_err ever reached: {min_err:.4f}")
print(f"First step under threshold ({SUCCESS_THRESHOLD}): {step_first_success}")
print(f"SUCCESS: {goal_err < SUCCESS_THRESHOLD}")

"""
find_safer_goal.py
Searches for a goal position close to the original Q_GOAL, but with
more safety margin from the obstacle, so the robot can reach it fully
without the CBF having to hold it back at the very end.

Strategy: try small changes to each joint of the original goal, one at
a time, and see which change increases h(x) the most -- then combine
the most helpful changes into one new goal, checking joint limits and
overall distance from the original goal so the task stays close to
what you originally wanted.
"""
import numpy as np
import mujoco
from robot.franka import DOF, FrankaModel
from planner.factor_graph import SignedDistanceField
from cbf.barrier import DistanceBarrier
from robot.mujoco_env import MujocoFrankaEnv

JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

obstacle_center = (0.20, 0.09, 0.85)
obstacle_radius = 0.08
d_safe = 0.03

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=obstacle_center, obstacle_radius=obstacle_radius)
franka = FrankaModel(env.model, mujoco.MjData(env.model))
sdf = SignedDistanceField(np.array([obstacle_center]), np.array([obstacle_radius]))
barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii, sdf=sdf, dof=DOF, d_safe=d_safe)

q_goal_original = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
h_original = barrier.forward(np.concatenate([q_goal_original, np.zeros(DOF)]))
print(f"Original goal h(x): {h_original:.4f}")
print()

# Try changing each joint, one at a time, by a small step in both
# directions, and see which single change helps the most.
print(f"{'joint':>6}{'direction':>10}{'new_h(x)':>12}{'change':>10}")
best_moves = []
for joint_idx in range(DOF):
    for direction in [-1, +1]:
        for step_size in [0.02, 0.05, 0.08]:
            q_test = q_goal_original.copy()
            q_test[joint_idx] += direction * step_size
            if q_test[joint_idx] < JOINT_LOWER[joint_idx] or q_test[joint_idx] > JOINT_UPPER[joint_idx]:
                continue
            h_test = barrier.forward(np.concatenate([q_test, np.zeros(DOF)]))
            change = h_test - h_original
            best_moves.append((change, joint_idx, direction, step_size, h_test))

best_moves.sort(reverse=True)
print("Top 10 single-joint changes that improve safety margin the most:")
print(f"{'joint':>6}{'direction':>10}{'step':>8}{'new_h(x)':>12}{'improvement':>14}")
for change, joint_idx, direction, step_size, h_test in best_moves[:10]:
    print(f"{joint_idx+1:>6}{'+' if direction > 0 else '-':>10}{step_size:>8}{h_test:>12.4f}{change:>14.4f}")

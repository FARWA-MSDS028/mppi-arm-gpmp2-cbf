"""
find_safer_goal_combo.py
Combines the best single-joint changes together (checking the REAL,
combined h(x), not just adding numbers, since the true system is not
simply additive) to find an actual safe goal close to the original one.
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

def h_of(q):
    return barrier.forward(np.concatenate([q, np.zeros(DOF)]))

print(f"Original goal h(x): {h_of(q_goal_original):.4f}")
print()

# Try combining the top few helpful moves together, at a few different
# strengths, and check the REAL combined h(x) each time.
combos = [
    {"name": "joint2+0.08, joint4-0.08", "changes": {1: 0.08, 3: -0.08}},
    {"name": "joint2+0.10, joint4-0.10", "changes": {1: 0.10, 3: -0.10}},
    {"name": "joint2+0.08, joint4-0.08, joint3+0.08", "changes": {1: 0.08, 3: -0.08, 2: 0.08}},
    {"name": "joint2+0.10, joint4-0.10, joint3+0.10", "changes": {1: 0.10, 3: -0.10, 2: 0.10}},
    {"name": "joint2+0.12, joint4-0.12, joint3+0.10", "changes": {1: 0.12, 3: -0.12, 2: 0.10}},
    {"name": "joint2+0.15, joint4-0.10", "changes": {1: 0.15, 3: -0.10}},
    {"name": "joint2+0.10, joint4-0.08, joint1+0.08", "changes": {1: 0.10, 3: -0.08, 0: 0.08}},
]

print(f"{'combo':<45}{'h(x)':>10}{'distance_from_original':>25}")
for combo in combos:
    q_test = q_goal_original.copy()
    for joint_idx, delta in combo["changes"].items():
        q_test[joint_idx] += delta
    q_test = np.clip(q_test, JOINT_LOWER, JOINT_UPPER)
    h_test = h_of(q_test)
    dist = float(np.max(np.abs(q_test - q_goal_original)))
    safe_mark = "SAFE" if h_test >= 0 else "still unsafe"
    print(f"{combo['name']:<45}{h_test:>10.4f}{dist:>25.4f}   {safe_mark}")
    print(f"    full position: {np.round(q_test, 4)}")

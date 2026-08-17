"""
check_constraint_force_magnitude.py
Compares the magnitude of self-collision constraint forces against
actuator forces, to see whether self-collision is strong enough to
meaningfully resist commanded motion -- the likely explanation for the
earlier "commanded full goal position, but barely moved" mystery.
"""
import numpy as np, mujoco
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import DOF

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(0.20, 0.09, 0.85), obstacle_radius=0.08)
env.reset(np.zeros(DOF))
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])

print(f"{'step':>6}{'ncon':>6}{'|qfrc_constraint|':>20}{'|qfrc_actuator|':>18}{'qvel_norm':>12}")
for step in range(10):
    env.step(Q_GOAL)
    qfrc_c = np.linalg.norm(env.data.qfrc_constraint[:DOF])
    qfrc_a = np.linalg.norm(env.data.qfrc_actuator[:DOF])
    qvel_n = np.linalg.norm(env.data.qvel[:DOF])
    print(f"{step:>6}{env.data.ncon:>6}{qfrc_c:>20.4f}{qfrc_a:>18.4f}{qvel_n:>12.4f}")

print(f"\nIf |qfrc_constraint| is comparable to or larger than |qfrc_actuator|,")
print(f"self-collision is genuinely fighting the commanded motion.")

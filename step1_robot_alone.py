"""
step1_robot_alone.py
Simplest possible test: send the goal position directly to the robot,
no GPMP2, no MPPI, no CBF, no threads. We already proved this works
(goal_err=0.0073) earlier tonight -- this just re-confirms it still
works with the current files, before we test anything more complex.
"""
import numpy as np
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import DOF

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(5.0, 5.0, 5.0), obstacle_radius=0.01)
q0 = np.zeros(DOF)
q_goal = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
env.reset(q0)

for step in range(300):
    env.substep(q_goal)
    if step % 50 == 0:
        q = env.get_state()[:DOF]
        err = float(np.max(np.abs(q - q_goal)))
        print(f"step {step:>4}  goal_err={err:.4f}")

q = env.get_state()[:DOF]
print(f"Final goal_err: {float(np.max(np.abs(q - q_goal))):.4f}")

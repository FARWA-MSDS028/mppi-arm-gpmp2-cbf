"""
diagnose_position_control.py
Tests the simplified fix directly: since the actuators are position
servos (confirmed), skip computed-torque control entirely and just pass
the desired position straight through, clipped to real joint limits.
No MPPI, no GPMP2, no CBF yet -- pure "step straight toward q_goal" to
isolate whether this alone fixes the stuck-joint plateau.
"""
import numpy as np
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import DOF

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(5.0, 5.0, 5.0),
                       obstacle_radius=0.01)

q0 = np.zeros(DOF)
q_goal = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
env.reset(q0)

JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

n_steps = 300  # same budget as the very first (misleading) reachability test
errs = []
for step in range(n_steps):
    x = env.get_state()
    q = x[:DOF]
    # Directly command the goal position, clipped to real joint limits --
    # MuJoCo's own internal servo (kp=4500/3500/2000) handles tracking.
    q_des = np.clip(q_goal, JOINT_LOWER, JOINT_UPPER)
    env.step(q_des)
    e_now = float(np.linalg.norm(q - q_goal, ord=np.inf))
    errs.append(e_now)
    if step % 50 == 0 or step == n_steps - 1:
        print(f"step {step:>4}  goal_err = {e_now:.4f}")

print(f"\nFinal goal_err: {errs[-1]:.4f}")
print(f"Min goal_err: {min(errs):.4f} at step {errs.index(min(errs))}")
print(f"REACHED (< 0.15): {'YES' if min(errs) < 0.15 else 'NO'}")

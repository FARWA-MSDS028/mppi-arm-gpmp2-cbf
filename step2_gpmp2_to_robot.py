"""
step2_gpmp2_to_robot.py
GPMP2 solves ONE plan with a long enough horizon to really reach the
goal, then the robot follows that plan directly (no MPPI in between).
No threads. This tests whether GPMP2's plan, on its own, actually
leads the robot to the target.
"""
import numpy as np
import mujoco
from robot.franka import DOF, FrankaModel
from planner.factor_graph import SignedDistanceField
from planner.gpmp2_planner import GPMP2Planner
from robot.mujoco_env import MujocoFrankaEnv

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(0.20, 0.09, 0.85), obstacle_radius=0.08)
franka = FrankaModel(env.model, mujoco.MjData(env.model))
sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))

q0 = np.zeros(DOF)
q_goal = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
env.reset(q0)

Qc = 0.5 * np.eye(DOF)
planner = GPMP2Planner(dof=DOF, dt=0.05, Qc=Qc, sdf=sdf, fk_fn=franka.fk,
                        sphere_offsets=franka.sphere_radii, eps=0.02, sigma_obs=0.02)

N_horizon = 60  # long enough to actually reach the goal in one plan
theta0 = np.concatenate([q0, np.zeros(DOF)])
theta_goal = np.concatenate([q_goal, np.zeros(DOF)])
result = planner.plan(theta0, theta_goal, N=N_horizon)
theta_star = result.theta_star[:, :DOF]

print(f"GPMP2 final planned waypoint: {np.round(theta_star[-1], 4)}")
print(f"Q_GOAL:                        {np.round(q_goal, 4)}")

for i, q_des in enumerate(theta_star):
    for _ in range(25):  # match the real control_dt (0.05s = 25 physics ticks)
        env.substep(q_des)
    if i % 10 == 0:
        q = env.get_state()[:DOF]
        err = float(np.max(np.abs(q - q_goal)))
        print(f"waypoint {i:>3}  goal_err={err:.4f}")

q = env.get_state()[:DOF]
print(f"Final goal_err: {float(np.max(np.abs(q - q_goal))):.4f}")

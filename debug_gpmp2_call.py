"""
debug_gpmp2_call.py
Calls GPMP2Planner.plan() the EXACT same way demo/threaded_pipeline.py's
gpmp2_thread_fn does, but with no threads -- so we can see the real
error and print shapes right before the crash.
"""
import numpy as np
import mujoco
from robot.franka import DOF, FrankaModel
from planner.factor_graph import SignedDistanceField
from planner.gpmp2_planner import GPMP2Planner

mjcf_path = "assets/panda.xml"
obstacle_center = (0.20, 0.09, 0.85)
obstacle_radius = 0.08
gpmp2_eps = 0.02

model = mujoco.MjModel.from_xml_path(mjcf_path)
franka = FrankaModel(model, mujoco.MjData(model))
sdf = SignedDistanceField(np.array([obstacle_center]), np.array([obstacle_radius]))

Qc = 0.5 * np.eye(DOF)
planner = GPMP2Planner(dof=DOF, dt=0.05, Qc=Qc, sdf=sdf, fk_fn=franka.fk,
                        sphere_offsets=franka.sphere_radii, eps=gpmp2_eps,
                        sigma_obs=0.02)

q0 = np.zeros(DOF)
q_goal = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
theta0 = np.concatenate([q0, np.zeros(DOF)])
theta_goal = np.concatenate([q_goal, np.zeros(DOF)])

print("theta0 shape:", theta0.shape)
print("theta_goal shape:", theta_goal.shape)
print("Qc shape:", Qc.shape)
print("sphere_offsets shape:", franka.sphere_radii.shape)
print("DOF:", DOF)
print("N_horizon: 30")

result = planner.plan(theta0, theta_goal, N=30, init_trajectory=None)
print("SUCCESS -- final_error:", result.final_error)

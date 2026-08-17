"""
debug_gpmp2_thread_only.py
Runs ONLY the GPMP2 thread (no MPPI, no robot thread) for a short
time, to confirm the velocity-slicing fix works across MULTIPLE
solves in a row -- this is the exact case that used to crash on the
second solve.
"""
import time
import numpy as np
import mujoco
import threading

from robot.franka import DOF, FrankaModel
from planner.factor_graph import SignedDistanceField
from planner.gpmp2_planner import GPMP2Planner
from demo.dashboard_state import DashboardState
from demo.threaded_pipeline import PipelineSharedState, gpmp2_thread_fn

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

state = DashboardState()
shared = PipelineSharedState()
shared.set_robot_state(q0, np.zeros(DOF))

t = threading.Thread(target=gpmp2_thread_fn,
                      args=(shared, state, planner, q_goal, DOF, 30),
                      daemon=True)
t.start()

print("Running for 10 seconds -- watching for multiple successful solves...")
for i in range(10):
    time.sleep(1)
    snap = state.snapshot()
    print(f"t={i+1}s  gpmp2_status={snap['gpmp2_status']}  "
          f"version={snap['gpmp2_version']}  "
          f"solve_time={snap['gpmp2_solve_time_s']}")

shared.stop = True
print("Done -- if version kept increasing with no crash, the fix works.")

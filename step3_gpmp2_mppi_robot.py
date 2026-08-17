"""
step3_gpmp2_mppi_robot.py
GPMP2 solves ONE plan, then MPPI is used to follow that plan step by
step (same as demo/threaded_pipeline.py tries to do), sending MPPI's
output directly to the robot. No CBF, no threads. This isolates
whether the problem is in how MPPI is being called.
"""
import numpy as np
import mujoco
from robot.franka import DOF, FrankaModel
from planner.factor_graph import SignedDistanceField
from planner.gpmp2_planner import GPMP2Planner
from robot.mujoco_env import MujocoFrankaEnv
from controller.mppi import MPPIController

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(0.20, 0.09, 0.85), obstacle_radius=0.08)
franka = FrankaModel(env.model, mujoco.MjData(env.model))
sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))

q0 = np.zeros(DOF)
q_goal = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
env.reset(q0)

Qc = 0.5 * np.eye(DOF)
N_horizon = 60
planner = GPMP2Planner(dof=DOF, dt=0.05, Qc=Qc, sdf=sdf, fk_fn=franka.fk,
                        sphere_offsets=franka.sphere_radii, eps=0.02, sigma_obs=0.02)

theta0 = np.concatenate([q0, np.zeros(DOF)])
theta_goal = np.concatenate([q_goal, np.zeros(DOF)])
result = planner.plan(theta0, theta_goal, N=N_horizon)
theta_ref = result.theta_star[:, :DOF]  # (N_horizon+1, DOF)

mppi = MPPIController(lam=1.0, dt=0.05, dof=DOF, sdf=sdf, eps_margin=0.15,
                       sigma_obs=0.02, lambda_cbf=1.0, fk_batch_fn=franka.fk_batch,
                       sphere_radii=franka.sphere_radii)
rng = np.random.default_rng(0)
Sigma = 0.002 ** 2 * np.eye(DOF)
K_inv_diag_full = np.ones((N_horizon + 1, DOF))

def barrier_batch_fn(V):
    from cbf.barrier import DistanceBarrier
    N_, T_, _ = V.shape
    return np.zeros((N_, T_))  # placeholder, no real barrier needed for this test

for step in range(N_horizon):
    theta_remaining = theta_ref[step:]
    K_remaining = K_inv_diag_full[step:]
    if len(theta_remaining) < 2:
        break
    mppi_result = mppi.step(theta_remaining, Sigma, 200, K_remaining, barrier_batch_fn, rng)
    u_mppi = mppi_result.u_mppi[0]
    for _ in range(25):
        env.substep(u_mppi)
    if step % 10 == 0:
        q = env.get_state()[:DOF]
        err = float(np.max(np.abs(q - q_goal)))
        print(f"step {step:>3}  u_mppi[0]={u_mppi[0]:.4f}  goal_err={err:.4f}")

q = env.get_state()[:DOF]
print(f"Final goal_err: {float(np.max(np.abs(q - q_goal))):.4f}")

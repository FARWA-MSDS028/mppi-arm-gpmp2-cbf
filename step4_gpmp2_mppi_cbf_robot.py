"""
step4_gpmp2_mppi_cbf_robot.py
GPMP2 -> MPPI -> CBF-QP -> Robot, still no threads. This adds the last
missing piece from step3, to test whether the CBF-QP is the reason the
threaded pipeline is not moving toward the goal.
"""
import numpy as np
import mujoco
from robot.franka import DOF, FrankaModel
from planner.factor_graph import SignedDistanceField
from planner.gpmp2_planner import GPMP2Planner
from robot.mujoco_env import MujocoFrankaEnv
from controller.mppi import MPPIController
from cbf.barrier import DistanceBarrier, hocbf_lie_derivatives, franka_dynamics
from cbf.qp_solver import CBFQPSolver

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(0.20, 0.09, 0.85), obstacle_radius=0.08)
franka = FrankaModel(env.model, mujoco.MjData(env.model))
sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                           sdf=sdf, dof=DOF, d_safe=0.03)

JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
qp = CBFQPSolver(m=DOF, u_min=JOINT_LOWER, u_max=JOINT_UPPER, alpha_gamma=100.0)

def gravity_fn(q): return franka.gravity(q)
def coriolis_fn(q, qdot): return franka.coriolis_times_qdot(q, qdot)
def M_fn(q): return franka.mass_matrix(q)
def f_fn(xi):
    fi, _ = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
    return fi
def g_fn(xi):
    _, gi = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
    return gi

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
theta_ref = result.theta_star[:, :DOF]

mppi = MPPIController(lam=1.0, dt=0.05, dof=DOF, sdf=sdf, eps_margin=0.15,
                       sigma_obs=0.02, lambda_cbf=1.0, fk_batch_fn=franka.fk_batch,
                       sphere_radii=franka.sphere_radii)
rng = np.random.default_rng(0)
Sigma = 0.002 ** 2 * np.eye(DOF)
K_inv_diag_full = np.ones((N_horizon + 1, DOF))

def barrier_batch_fn(V):
    N_, T_, _ = V.shape
    h = np.zeros((N_, T_))
    for i in range(N_):
        for t in range(T_):
            x = np.concatenate([V[i, t], np.zeros(DOF)])
            h[i, t] = barrier.forward(x)
    return h

for step in range(N_horizon):
    theta_remaining = theta_ref[step:]
    K_remaining = K_inv_diag_full[step:]
    if len(theta_remaining) < 2:
        break
    mppi_result = mppi.step(theta_remaining, Sigma, 200, K_remaining, barrier_batch_fn, rng)
    u_mppi = mppi_result.u_mppi[0]

    q, qdot = env.get_state()[:DOF], env.get_state()[DOF:]
    x = np.concatenate([q, qdot])
    psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x, f_fn, g_fn, alpha0=1.0)
    qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)

    for _ in range(25):
        env.substep(qp_result.u_safe)

    if step % 10 == 0:
        q = env.get_state()[:DOF]
        err = float(np.max(np.abs(q - q_goal)))
        print(f"step {step:>3}  u_mppi[0]={u_mppi[0]:.4f}  u_safe[0]={qp_result.u_safe[0]:.4f}  "
              f"goal_err={err:.4f}  h(x)={h0:.4f}")

q = env.get_state()[:DOF]
print(f"Final goal_err: {float(np.max(np.abs(q - q_goal))):.4f}")

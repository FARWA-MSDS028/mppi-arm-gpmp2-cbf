"""
diagnose_mppi_cbf_tugofwar.py
Runs the real safe_mppi pipeline for 30 steps, printing q before/after,
u_mppi (proposed) vs u_safe (corrected), and net displacement per step
-- to see directly whether the CBF is holding the robot at a standstill
(corrections cancel MPPI's forward samples) or actively pushing it
backward, now that self-collision and the actuator-model fix are both
in place.
"""
import numpy as np, mujoco
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from planner.gpmp2_planner import GPMP2Planner
from cbf.barrier import DistanceBarrier, hocbf_lie_derivatives, franka_dynamics, closest_clearance
from cbf.qp_solver import CBFQPSolver, detect_unsafe
from controller.mppi import MPPIController

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
OBSTACLE_CENTER = (0.20, 0.09, 0.85)
OBSTACLE_RADIUS = 0.08
D_SAFE = 0.03
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=OBSTACLE_CENTER,
                       obstacle_radius=OBSTACLE_RADIUS)
franka = FrankaModel(env.model, mujoco.MjData(env.model))
sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                           sdf=sdf, dof=DOF, d_safe=D_SAFE)
qp = CBFQPSolver(m=DOF, u_min=JOINT_LOWER, u_max=JOINT_UPPER, alpha_gamma=1.0)
env.reset(Q0)

def gravity_fn(q): return franka.gravity(q)
def coriolis_fn(q, qdot): return franka.coriolis_times_qdot(q, qdot)
def M_fn(q): return franka.mass_matrix(q)
def f_fn(xi):
    fi, _ = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
    return fi
def g_fn(xi):
    _, gi = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
    return gi

theta0 = np.concatenate([Q0, np.zeros(DOF)])
theta_goal = np.concatenate([Q_GOAL, np.zeros(DOF)])
Qc = 0.5 * np.eye(DOF)
planner = GPMP2Planner(dof=DOF, dt=0.05, Qc=Qc, sdf=sdf, fk_fn=franka.fk,
                        sphere_offsets=franka.sphere_radii, eps=0.02, sigma_obs=0.02)
gpmp2_result = planner.plan(theta0, theta_goal, N=30)
theta_q = gpmp2_result.theta_star[:, :DOF]

def barrier_batch_fn(V):
    N_, T_, _ = V.shape
    h = np.zeros((N_, T_))
    for i in range(N_):
        for t in range(T_):
            x = np.concatenate([V[i, t], np.zeros(DOF)])
            h[i, t] = barrier.forward(x)
    return h

mppi = MPPIController(lam=1.0, dt=0.05, dof=DOF, sdf=sdf, eps_margin=0.15,
                       sigma_obs=0.02, lambda_cbf=1.0, fk_batch_fn=franka.fk_batch,
                       sphere_radii=franka.sphere_radii)
rng = np.random.default_rng(0)
Sigma = 0.002**2 * np.eye(DOF)
K_inv_diag = np.ones((31, DOF))

print(f"{'step':>5}{'q0_before':>12}{'u_mppi[0]':>12}{'u_safe[0]':>12}{'q0_after':>12}{'net_disp[0]':>14}")
for step in range(30):
    x = env.get_state()
    q = x[:DOF]
    result = mppi.step(theta_q[step:], Sigma, 200, K_inv_diag[step:], barrier_batch_fn, rng)
    u_mppi_pos = result.u_mppi[0]
    u_mppi = np.clip(u_mppi_pos, JOINT_LOWER, JOINT_UPPER)

    psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x, f_fn, g_fn, alpha0=1.0)
    qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
    env.step(qp_result.u_safe)

    q_after = env.get_state()[:DOF]
    net_disp = q_after[0] - q[0]
    print(f"{step:>5}{q[0]:>12.4f}{u_mppi[0]:>12.4f}{qp_result.u_safe[0]:>12.4f}"
          f"{q_after[0]:>12.4f}{net_disp:>14.5f}")

print(f"\nFinal q: {np.round(env.get_state()[:DOF], 4)}")
print(f"Target:  {np.round(Q_GOAL, 4)}")

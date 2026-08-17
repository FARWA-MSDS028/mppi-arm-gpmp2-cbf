"""
check_h_along_mppi_only.py
Re-runs Experiment A (MPPI only, no CBF) and logs h(x) throughout --
checking whether MPPI's own uncorrected trajectory would have violated
safety near the goal (meaning the CBF's sustained fight in Experiment B
is genuinely necessary), or stayed safe on its own (meaning the CBF is
overreacting to something that wasn't really a threat).
"""
import numpy as np, mujoco
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from planner.gpmp2_planner import GPMP2Planner
from cbf.barrier import DistanceBarrier, hocbf_lie_derivatives, franka_dynamics
from controller.mppi import MPPIController

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
OBSTACLE_CENTER = (0.20, 0.09, 0.85)
OBSTACLE_RADIUS = 0.08
D_SAFE = 0.03
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
N_SUBSTEPS = 25
N_STEPS = 30

model = mujoco.MjModel.from_xml_path("assets/panda.xml")
data = mujoco.MjData(model)
data.qpos[:DOF] = Q0
franka = FrankaModel(model, mujoco.MjData(model))
sdf = SignedDistanceField(np.array([OBSTACLE_CENTER]), np.array([OBSTACLE_RADIUS]))
barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                           sdf=sdf, dof=DOF, d_safe=D_SAFE)

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
gpmp2_result = planner.plan(theta0, theta_goal, N=N_STEPS)
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
K_inv_diag = np.ones((N_STEPS + 1, DOF))

print(f"{'step':>5}{'h(x)':>10}{'goal_err':>10}")
for step in range(N_STEPS):
    q, qdot = data.qpos[:DOF].copy(), data.qvel[:DOF].copy()
    result = mppi.step(theta_q[step:], Sigma, 200, K_inv_diag[step:], barrier_batch_fn, rng)
    u_mppi = np.clip(result.u_mppi[0], JOINT_LOWER, JOINT_UPPER)
    for _ in range(N_SUBSTEPS):
        data.ctrl[:DOF] = u_mppi
        mujoco.mj_step(model, data)
    x_now = np.concatenate([data.qpos[:DOF], data.qvel[:DOF]])
    _, _, _, h0 = hocbf_lie_derivatives(barrier, x_now, f_fn, g_fn, alpha0=1.0)
    err = float(np.linalg.norm(data.qpos[:DOF] - Q_GOAL, ord=np.inf))
    print(f"{step:>5}{h0:>10.5f}{err:>10.4f}")

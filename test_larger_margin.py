"""
test_fast_cbf_loop.py
Same as verify_substep_fix.py, but with the CBF-QP re-solved every
physics substep instead of once per control period -- testing whether
decoupling the safety-enforcement rate from the planning rate restores
h(x) >= 0 while keeping the goal-reaching progress the substep fix
already proved works.
"""
import numpy as np, mujoco
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from planner.gpmp2_planner import GPMP2Planner
from cbf.barrier import DistanceBarrier, hocbf_lie_derivatives, franka_dynamics
from cbf.qp_solver import CBFQPSolver
from controller.mppi import MPPIController

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
OBSTACLE_CENTER = (0.20, 0.09, 0.85)
OBSTACLE_RADIUS = 0.08
D_SAFE = 0.08
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
N_SUBSTEPS = 25

model = mujoco.MjModel.from_xml_path("assets/panda.xml")
data = mujoco.MjData(model)
data.qpos[:DOF] = Q0

def substep(u):
    data.ctrl[:DOF] = u
    mujoco.mj_step(model, data)

franka = FrankaModel(model, mujoco.MjData(model))
sdf = SignedDistanceField(np.array([OBSTACLE_CENTER]), np.array([OBSTACLE_RADIUS]))
barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                           sdf=sdf, dof=DOF, d_safe=D_SAFE)
qp = CBFQPSolver(m=DOF, u_min=JOINT_LOWER, u_max=JOINT_UPPER, alpha_gamma=1.0)

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

worst_h = float("inf")
for step in range(30):
    q, qdot = data.qpos[:DOF].copy(), data.qvel[:DOF].copy()
    x = np.concatenate([q, qdot])
    result = mppi.step(theta_q[step:], Sigma, 200, K_inv_diag[step:], barrier_batch_fn, rng)
    u_mppi = np.clip(result.u_mppi[0], JOINT_LOWER, JOINT_UPPER)

    # Fast inner safety loop -- re-solve CBF-QP every physics substep
    for _ in range(N_SUBSTEPS):
        x_now = np.concatenate([data.qpos[:DOF], data.qvel[:DOF]])
        psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x_now, f_fn, g_fn, alpha0=1.0)
        worst_h = min(worst_h, h0)
        qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
        substep(qp_result.u_safe)

    if step % 5 == 0 or step == 29:
        err = float(np.linalg.norm(data.qpos[:DOF] - Q_GOAL, ord=np.inf))
        print(f"step {step:>4}  q0={data.qpos[0]:.4f}  goal_err={err:.4f}  worst_h_so_far={worst_h:.4f}")

final_err = float(np.linalg.norm(data.qpos[:DOF] - Q_GOAL, ord=np.inf))
print(f"\nFinal goal_err: {final_err:.4f}   Worst h(x) over entire run: {worst_h:.4f}  "
      f"(safe: {worst_h >= 0})")

"""
verify_qp_constraint_satisfaction.py
Uses raw MuJoCo directly (matching diagnose_h_dip_shape.py's exact
pattern -- one physics tick per QP solve), fast-forwards to the known
boundary-crossing point (step 9->10), then directly checks:
1. Does u_safe actually satisfy the QP's own constraint inequality
   Lg_psi1 . u >= -alpha_gamma*psi1 - Lf_psi1 ?
2. Does the constraint's DRIFT-ONLY prediction of psi1's next value
   (built from f_fn alone, no control) match what actually happens
   under the real, CONTROLLED u_safe ?
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
D_SAFE = 0.03
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
ALPHA_GAMMA = 1.0
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
qp = CBFQPSolver(m=DOF, u_min=JOINT_LOWER, u_max=JOINT_UPPER, alpha_gamma=ALPHA_GAMMA)

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

# ---- Fast-forward through steps 0-8 exactly as before ----
for step in range(9):
    q, qdot = data.qpos[:DOF].copy(), data.qvel[:DOF].copy()
    x = np.concatenate([q, qdot])
    result = mppi.step(theta_q[step:], Sigma, 200, K_inv_diag[step:], barrier_batch_fn, rng)
    u_mppi = np.clip(result.u_mppi[0], JOINT_LOWER, JOINT_UPPER)
    for sub in range(N_SUBSTEPS):
        x_now = np.concatenate([data.qpos[:DOF], data.qvel[:DOF]])
        psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x_now, f_fn, g_fn, alpha0=1.0)
        qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
        substep(qp_result.u_safe)

# ---- Step 9, substep 24 -> Step 10, substep 0: the exact boundary-crossing moment ----
step = 9
q, qdot = data.qpos[:DOF].copy(), data.qvel[:DOF].copy()
x = np.concatenate([q, qdot])
result = mppi.step(theta_q[step:], Sigma, 200, K_inv_diag[step:], barrier_batch_fn, rng)
u_mppi = np.clip(result.u_mppi[0], JOINT_LOWER, JOINT_UPPER)

print(f"{'sub':>4}{'h':>10}{'psi1':>10}{'LHS=Lg.u':>12}{'RHS=-a*psi1-Lf':>16}{'CONSTRAINT OK?':>16}")
for sub in range(N_SUBSTEPS):
    x_now = np.concatenate([data.qpos[:DOF], data.qvel[:DOF]])
    psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x_now, f_fn, g_fn, alpha0=1.0)
    qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
    u_safe = qp_result.u_safe

    LHS = float(Lg_psi1 @ u_safe)
    RHS = -ALPHA_GAMMA * psi1 - Lf_psi1
    ok = LHS >= RHS - 1e-6  # small numerical tolerance

    # Also: what psi1 does the constraint PREDICT for the next instant,
    # vs what we'll actually measure after substepping?
    predicted_psi1_dot = LHS + Lf_psi1  # = Lg.u + Lf = psi1_dot under this u
    predicted_next_psi1 = psi1 + predicted_psi1_dot * model.opt.timestep

    print(f"{sub:>4}{h0:>10.5f}{psi1:>10.5f}{LHS:>12.4f}{RHS:>16.4f}{str(ok):>16}")
    substep(u_safe)

    x_after = np.concatenate([data.qpos[:DOF], data.qvel[:DOF]])
    psi1_after, _, _, h0_after = hocbf_lie_derivatives(barrier, x_after, f_fn, g_fn, alpha0=1.0)
    if sub in (0, 1, 2):
        print(f"      predicted next psi1={predicted_next_psi1:.5f}   "
              f"actual next psi1={psi1_after:.5f}   "
              f"prediction error={abs(predicted_next_psi1 - psi1_after):.5f}")

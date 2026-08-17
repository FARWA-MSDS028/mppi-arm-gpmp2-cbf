"""
test_mppi_vs_cbf_intervention.py
Frozen alpha_gamma=100 (confirmed: smallest value achieving
unsafe_substeps=0/750). Logs ||u_safe - u_mppi|| every substep to
directly test whether the ~11.5cm final error is caused by CBF still
fighting MPPI, or MPPI itself simply stopping short.
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
N_SUBSTEPS = 25
ALPHA_GAMMA = 100.0
N_STEPS = 30

def setup():
    model = mujoco.MjModel.from_xml_path("assets/panda.xml")
    data = mujoco.MjData(model)
    data.qpos[:DOF] = Q0
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
    return model, data, franka, sdf, barrier, qp, f_fn, g_fn, planner, theta_q, mppi, barrier_batch_fn

def substep(model, data, u):
    data.ctrl[:DOF] = u
    mujoco.mj_step(model, data)

# ---- Experiment A: MPPI only, no CBF at all ----
print("=" * 60, "\nEXPERIMENT A: MPPI only (no CBF)\n", "=" * 60)
model, data, franka, sdf, barrier, qp, f_fn, g_fn, planner, theta_q, mppi, barrier_batch_fn = setup()
rng = np.random.default_rng(0)
Sigma = 0.002**2 * np.eye(DOF)
K_inv_diag = np.ones((N_STEPS + 1, DOF))
for step in range(N_STEPS):
    q, qdot = data.qpos[:DOF].copy(), data.qvel[:DOF].copy()
    result = mppi.step(theta_q[step:], Sigma, 200, K_inv_diag[step:], barrier_batch_fn, rng)
    u_mppi = np.clip(result.u_mppi[0], JOINT_LOWER, JOINT_UPPER)
    for _ in range(N_SUBSTEPS):
        substep(model, data, u_mppi)
err_A = float(np.linalg.norm(data.qpos[:DOF] - Q_GOAL, ord=np.inf))
print(f"Final goal_err (MPPI only): {err_A:.4f}")

# ---- Experiment B: MPPI + CBF, logging ||u_safe - u_mppi|| ----
print("\n" + "=" * 60, "\nEXPERIMENT B: MPPI + CBF (alpha_gamma=100)\n", "=" * 60)
model, data, franka, sdf, barrier, qp, f_fn, g_fn, planner, theta_q, mppi, barrier_batch_fn = setup()
rng = np.random.default_rng(0)
diffs = []
for step in range(N_STEPS):
    q, qdot = data.qpos[:DOF].copy(), data.qvel[:DOF].copy()
    result = mppi.step(theta_q[step:], Sigma, 200, K_inv_diag[step:], barrier_batch_fn, rng)
    u_mppi = np.clip(result.u_mppi[0], JOINT_LOWER, JOINT_UPPER)
    for _ in range(N_SUBSTEPS):
        x_now = np.concatenate([data.qpos[:DOF], data.qvel[:DOF]])
        psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x_now, f_fn, g_fn, alpha0=1.0)
        qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
        diff = float(np.linalg.norm(qp_result.u_safe - u_mppi))
        diffs.append(diff)
        substep(model, data, qp_result.u_safe)
    if step % 5 == 0 or step == N_STEPS - 1:
        recent = diffs[-N_SUBSTEPS:]
        print(f"step {step:>4}  mean||u*-u_mppi|| (this step)={np.mean(recent):.4f}  "
              f"max={np.max(recent):.4f}")
err_B = float(np.linalg.norm(data.qpos[:DOF] - Q_GOAL, ord=np.inf))
print(f"\nFinal goal_err (MPPI+CBF): {err_B:.4f}")
print(f"Overall mean ||u*-u_mppi||: {np.mean(diffs):.4f}   max: {np.max(diffs):.4f}")
print(f"Last 5 steps' worth of substeps -- mean ||u*-u_mppi||: {np.mean(diffs[-5*N_SUBSTEPS:]):.4f}")

print(f"\n{'='*60}\nCOMPARISON: A (MPPI only) err={err_A:.4f}  vs  B (MPPI+CBF) err={err_B:.4f}")
print(f"If these are close, CBF is NOT the cause of the residual error.")

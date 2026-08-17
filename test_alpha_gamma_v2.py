"""
test_alpha_gamma_v2.py
Re-tests alpha_gamma sweep with CORRECT substep timing (the earlier
sweep, before the mujoco_env.py fix, ran on 25x-compressed time and is
now known to be invalid). Tests whether a larger alpha_gamma reduces
how deep/long the h<0 dip runs, per the recovery-speed hypothesis.
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

def run_test(alpha_gamma_test, n_steps=30):
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
    qp = CBFQPSolver(m=DOF, u_min=JOINT_LOWER, u_max=JOINT_UPPER, alpha_gamma=alpha_gamma_test)

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
    K_inv_diag = np.ones((n_steps + 1, DOF))

    worst_h = float("inf")
    n_unsafe_substeps = 0
    total_substeps = 0
    for step in range(n_steps):
        q, qdot = data.qpos[:DOF].copy(), data.qvel[:DOF].copy()
        x = np.concatenate([q, qdot])
        result = mppi.step(theta_q[step:], Sigma, 200, K_inv_diag[step:], barrier_batch_fn, rng)
        u_mppi = np.clip(result.u_mppi[0], JOINT_LOWER, JOINT_UPPER)
        for _ in range(N_SUBSTEPS):
            x_now = np.concatenate([data.qpos[:DOF], data.qvel[:DOF]])
            psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x_now, f_fn, g_fn, alpha0=1.0)
            worst_h = min(worst_h, h0)
            if h0 < 0:
                n_unsafe_substeps += 1
            total_substeps += 1
            qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
            substep(qp_result.u_safe)

    final_err = float(np.linalg.norm(data.qpos[:DOF] - Q_GOAL, ord=np.inf))
    print(f"alpha_gamma={alpha_gamma_test:>6}  final_goal_err={final_err:.4f}  "
          f"worst_h={worst_h:.5f}  unsafe_substeps={n_unsafe_substeps}/{total_substeps}")

for ag in [1.0, 5.0, 20.0, 50.0]:
    run_test(ag)

"""
experiments/safe_mppi.py
==========================
Experiments 3 & 4 (spec):
  3. MPPI before CBF: unsafe trajectory/controls, barrier value, collision
     prediction (Stage 4's UnsafeDetection, recorded pre-QP).
  4. After CBF-QP: safe controls, barrier maintained, safety margins, QP
     intervention, control comparison (cbf/qp_solver.CBFQPResult).

Runs GPMP2+MPPI exactly as experiments/gpmp2_mppi.py, but now routes
u_mppi through the CBF-QP filter every step, logging both the pre-QP
(unsafe) and post-QP (safe) quantities so plots/main.py can render the
side-by-side comparison the spec requires.
"""

from __future__ import annotations
import numpy as np

from planner.gpmp2_planner import GPMP2Planner
from planner.factor_graph import SignedDistanceField
from controller.mppi import MPPIController
from cbf.barrier import DistanceBarrier, lie_derivatives, franka_dynamics, closest_clearance, hocbf_lie_derivatives
from cbf.qp_solver import CBFQPSolver, detect_unsafe
from cbf.feasibility import FeasibilityLog
from robot.franka import FrankaModel, DOF

JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def run_safe_mppi(env, franka: FrankaModel, sdf: SignedDistanceField,
                    barrier: DistanceBarrier, q0: np.ndarray, q_goal: np.ndarray,
                    T: int = 30, N: int = 200, n_steps: int = 150,
                    lam: float = 1.0, sigma: float = 0.002, rng_seed: int = 0,
                    gpmp2_eps: float = 0.15, gpmp2_sigma_obs: float = 0.02):
    rng = np.random.default_rng(rng_seed)
    theta0 = np.concatenate([q0, np.zeros(DOF)])
    theta_goal = np.concatenate([q_goal, np.zeros(DOF)])

    Qc = 0.5 * np.eye(DOF)
    planner = GPMP2Planner(dof=DOF, dt=0.05, Qc=Qc, sdf=sdf, fk_fn=franka.fk,
                            sphere_offsets=franka.sphere_radii,
                            eps=gpmp2_eps, sigma_obs=gpmp2_sigma_obs)
    gpmp2_result = planner.plan(theta0, theta_goal, N=T)
    theta_q = gpmp2_result.theta_star[:, :DOF]

    Sigma = sigma ** 2 * np.eye(DOF)
    K_inv_diag = np.ones((T + 1, DOF))

    def barrier_batch_fn(V):
        N_, T_, _ = V.shape
        h = np.zeros((N_, T_))
        for i in range(N_):
            for t in range(T_):
                x = np.concatenate([V[i, t], np.zeros(DOF)])
                h[i, t] = barrier.forward(x)
        return h

    mppi = MPPIController(lam=lam, dt=0.05, dof=DOF, sdf=sdf, eps_margin=0.15,
                           sigma_obs=0.02, lambda_cbf=1.0, fk_batch_fn=franka.fk_batch,
                           sphere_radii=franka.sphere_radii)
    qp = CBFQPSolver(m=DOF, u_min=JOINT_LOWER, u_max=JOINT_UPPER, alpha_gamma=100.0)
    feas_log = FeasibilityLog()

    # Real control-affine dynamics f(x), g(x) for the CBF Lie derivatives
    # (Step 3.2) -- was previously a f=0,g=I placeholder here; now matches
    # the fix already applied in main.py.
    def gravity_fn(q_):
        return franka.gravity(q_)

    def coriolis_fn(q_, qdot_):
        return franka.coriolis_times_qdot(q_, qdot_)

    def M_fn(q_):
        return franka.mass_matrix(q_)

    def f_fn(xi):
        fi, _ = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
        return fi

    def g_fn(xi):
        _, gi = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
        return gi

    q = q0.copy()
    log = {"q": [], "u_mppi": [], "u_safe": [], "h": [], "unsafe_flags": [],
           "predicted_hdot": [], "intervention": [], "cost_history": [],
           "goal_error": [], "dist": []}

    for step in range(n_steps):
        k = min(step, T - 1)
        x = env.get_state()
        result = mppi.step(theta_q[k:], Sigma, N, K_inv_diag[k:], barrier_batch_fn, rng)
        u_mppi_pos = result.u_mppi[0]  # DESIRED POSITION -- and now it stays one.
        # MuJoCo's actuators are position servos (confirmed by direct
        # testing), not torque motors -- no conversion needed. u_mppi IS
        # the position MPPI wants, clipped to real joint limits, and the
        # CBF-QP below finds the closest SAFE position to it.
        u_mppi = np.clip(u_mppi_pos, JOINT_LOWER, JOINT_UPPER)

        # HOCBF fix (see cbf/barrier.py::hocbf_lie_derivatives docstring):
        # h0(x)=d(q)-d_safe has relative degree 2 under second-order
        # dynamics, so the plain Lg_h0 from lie_derivatives() is
        # IDENTICALLY ZERO -- the CBF-QP could never correct anything,
        # regardless of the "unsafe" flag above. psi1/Lf_psi1/Lg_psi1 are
        # used for the real QP constraint; h0 is kept for honest logging.
        psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x, f_fn, g_fn, alpha0=1.0)
        d_obs = closest_clearance(franka.fk, franka.sphere_radii, sdf, q)
        unsafe = detect_unsafe(u_mppi, Lf_psi1, Lg_psi1, psi1, alpha_gamma=100.0,
                                d_obstacle=d_obs, h0_physical=h0)

        qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
        # u_safe is already a torque command (the QP operates in torque
        # space, bounded by TAU_MAX) -- execute it directly. The previous
        # np.clip(qp_result.u_safe - q, -5, 5) treated it as a position
        # AGAIN after the QP had already produced a torque, which both
        # discarded the QP's actual output magnitude and reintroduced the
        # position/torque units mismatch this fix addresses.
        env.step(qp_result.u_safe)
        q = env.get_state()[:DOF]

        feas_log.record(step, unsafe, qp_result)
        log["q"].append(q.copy())
        log["u_mppi"].append(u_mppi.copy())
        log["u_safe"].append(qp_result.u_safe.copy())
        log["h"].append(h0)
        log["unsafe_flags"].append(unsafe.is_unsafe)
        log["predicted_hdot"].append(unsafe.predicted_hdot)
        log["intervention"].append(qp_result.intervention_magnitude)
        log["cost_history"].append(float(np.mean(result.costs)))
        log["goal_error"].append(float(np.linalg.norm(q - q_goal, ord=np.inf)))
        log["dist"].append(d_obs)

    return log, feas_log

"""
experiments/gpmp2_mppi.py
============================
Experiment 2 (spec): GPMP2 + MPPI. Runs GPMP2 once to get theta*, then
uses it as the MPPI sampling mean every control cycle (Step 2.1), and
records the same metrics as baseline_mppi.py plus the sampling-variance
reduction so plots/main.py can render:
  - trajectory comparison (vs. baseline_mppi)
  - sample distribution (rollouts around theta* vs around prev tape)
  - cost convergence
"""

from __future__ import annotations
import numpy as np

from planner.gpmp2_planner import GPMP2Planner
from planner.factor_graph import SignedDistanceField
from controller.mppi import MPPIController
from cbf.barrier import closest_clearance
from robot.franka import FrankaModel, DOF

JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def run_gpmp2_mppi(env, franka: FrankaModel, sdf: SignedDistanceField,
                     q0: np.ndarray, q_goal: np.ndarray, T: int = 30,
                     N: int = 200, n_steps: int = 150, lam: float = 1.0,
                     sigma: float = 0.002, rng_seed: int = 0,
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

    def barrier_batch_fn(V):  # no CBF term in this experiment (isolates the GPMP2 effect)
        return np.full(V.shape[:2], 10.0)  # large h => negligible cbf_soft_cost

    mppi = MPPIController(lam=lam, dt=0.05, dof=DOF, sdf=sdf, eps_margin=0.15,
                           sigma_obs=0.02, lambda_cbf=0.0, fk_batch_fn=franka.fk_batch,
                           sphere_radii=franka.sphere_radii)

    # Computed-torque control (see main.py's matching comment) -- Kp tapers
    # from base to wrist joints, Kd is critically damped per joint.

    log = {"q": [], "dist": [], "u": [], "cost_history": [], "sample_std": [], "goal_error": [],
           "V_snapshot": None, "sampling_mean_snapshot": None, "weighted_mean_snapshot": None}
    q = q0.copy()
    for step in range(n_steps):
        k = min(step, T - 1)
        result = mppi.step(theta_q[k:], Sigma, N, K_inv_diag[k:], barrier_batch_fn, rng)
        u0 = result.u_mppi[0]
        u_torque = np.clip(u0, JOINT_LOWER, JOINT_UPPER)
        env.step(u_torque)
        q = env.get_state()[:DOF]
        d = closest_clearance(franka.fk, franka.sphere_radii, sdf, q)

        log["q"].append(q.copy())
        log["dist"].append(d)
        log["u"].append(u_torque.copy())
        log["cost_history"].append(float(np.mean(result.costs)))
        log["sample_std"].append(np.std(result.V[:, 0, :]))
        log["goal_error"].append(float(np.linalg.norm(q - q_goal, ord=np.inf)))
        log["V_snapshot"] = result.V                    # overwritten each step; last
        log["sampling_mean_snapshot"] = theta_q[k:]      # step's rollout cloud kept
        log["weighted_mean_snapshot"] = result.u_mppi    # for plot_rollout_vs_mean

    return log, theta_q

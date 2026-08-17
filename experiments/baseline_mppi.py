"""
experiments/baseline_mppi.py
==============================
Experiment 1 (spec): Baseline MPPI -- ordinary MPPI with
V_i = mu_prev + eps_i (recursive tape mean, NOT the GPMP2 mean), no CBF
safety filter. Produces the trajectory/control/distance/goal-convergence/
obstacle-avoidance figures used as the point of comparison for
Experiment 2 (GPMP2+MPPI) and Experiment 4 (after CBF-QP).
"""

from __future__ import annotations
import numpy as np

from planner.factor_graph import SignedDistanceField
from controller.sampling import sample_trajectories
from controller.cost import gp_smoothness_cost, obstacle_cost, total_cost
from cbf.barrier import closest_clearance
from robot.franka import FrankaModel, DOF

JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def run_baseline_mppi(env, franka: FrankaModel, sdf: SignedDistanceField,
                        q_goal: np.ndarray, T: int = 30, N: int = 200,
                        n_steps: int = 150, lam: float = 1.0,
                        sigma: float = 0.002, rng_seed: int = 0):
    """
    Ordinary receding-horizon MPPI: at every control step, sample around
    the PREVIOUS control tape (shifted), NOT around a GPMP2 reference.
    This is the "Vi = previous trajectory + eps" behavior the spec
    explicitly says the GPMP2-guided variant must NOT use -- kept here
    only as the baseline comparator.
    """
    rng = np.random.default_rng(rng_seed)
    q = env.get_state()[:DOF]
    prev_tape = np.tile(q_goal, (T, 1))  # naive straight-to-goal init
    Sigma = sigma ** 2 * np.eye(DOF)
    K_inv_diag = np.ones((T, DOF))

    # Computed-torque control (see main.py's matching comment) -- Kp tapers
    # from base to wrist joints, Kd is critically damped per joint.

    log = {"q": [], "dist": [], "u": [], "cost_history": [], "goal_error": [],
           "V_snapshot": None, "sampling_mean_snapshot": None, "weighted_mean_snapshot": None}
    for step in range(n_steps):
        V, eps = sample_trajectories(prev_tape, Sigma, N, rng)
        gp_c = gp_smoothness_cost(V, prev_tape, K_inv_diag)  # regularizer only
        V_pos = franka.fk_batch(V)
        obs_c = obstacle_cost(V_pos, sdf, eps=0.15, sigma_obs=0.02,
                               sphere_radii=franka.sphere_radii)
        costs = total_cost(gp_c, obs_c, np.zeros_like(obs_c), lambda_cbf=0.0)

        beta = np.min(costs)
        w = np.exp(-(costs - beta) / lam)
        w /= np.sum(w)
        u_tape = np.tensordot(w, V, axes=(0, 0))

        u0 = u_tape[0]
        # u0 IS the control now -- MuJoCo's actuators are position servos
        # (confirmed by direct testing), not torque motors. No computed-
        # torque law needed; just clip to real joint limits.
        u_torque = np.clip(u0, JOINT_LOWER, JOINT_UPPER)
        env.step(u_torque)
        q = env.get_state()[:DOF]
        d = closest_clearance(franka.fk, franka.sphere_radii, sdf, q)

        log["q"].append(q.copy())
        log["dist"].append(d)
        log["u"].append(u_torque.copy())
        log["cost_history"].append(float(np.mean(costs)))
        log["goal_error"].append(float(np.linalg.norm(q - q_goal, ord=np.inf)))
        log["V_snapshot"] = V                       # overwritten each step; last
        log["sampling_mean_snapshot"] = prev_tape    # step's rollout cloud kept
        log["weighted_mean_snapshot"] = u_tape       # for plot_rollout_vs_mean

        # shift tape (ordinary MPPI's recursive mean update)
        prev_tape = np.vstack([u_tape[1:], u_tape[-1:]])

    return log

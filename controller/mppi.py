"""
controller/mppi.py
====================
Implements Steps 2.3-2.4: softmax importance weights and the weighted-
average optimal control.

    w_i = exp(-1/lambda J(V_i)) / sum_j exp(-1/lambda J(V_j))
    u_mppi = sum_i w_i * V_i

Numerically stabilized softmax (subtract min cost before exponentiating)
to avoid overflow -- mathematically identical, standard log-sum-exp trick.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from controller.sampling import sample_trajectories
from controller.cost import gp_smoothness_cost, obstacle_cost, cbf_soft_cost, total_cost


@dataclass
class MPPIResult:
    u_mppi: np.ndarray        # (T, n) weighted-average control tape
    weights: np.ndarray       # (N,) softmax importance weights
    costs: np.ndarray         # (N,) raw per-rollout costs
    V: np.ndarray             # (N, T, n) sampled rollouts
    eps: np.ndarray           # (N, T, n) sampled noise


class MPPIController:
    """
    Time complexity per call: O(N*T*n) for sampling + weighting,
    plus the cost of obstacle_cost's forward-kinematics/SDF evaluation
    which dominates in practice: O(N*T*M) (M = # collision spheres).
    Memory complexity: O(N*T*n).
    """

    def __init__(self, lam: float, dt: float, dof: int, sdf, eps_margin: float,
                 sigma_obs: float, lambda_cbf: float, fk_batch_fn,
                 sphere_radii=None):
        """
        fk_batch_fn  : callable V (N,T,dof) -> (N,T,M,3) collision-sphere
                       world positions (vectorized forward kinematics,
                       robot/franka.py)
        sphere_radii : (M,) robot collision-sphere radii
                       (robot/franka.py FrankaModel.sphere_radii) -- passed
                       through to controller.cost.obstacle_cost so MPPI's
                       clearance check matches the GPMP2 obstacle factor's
                       (planner/factor_graph.py) exactly. None => point
                       robot (radius 0), which under-estimates risk.
        """
        self.lambda_mppi = lam  # MPPI softmax temperature -- named distinctly
        # from the CBF-QP's KKT multiplier lambda* (cbf/qp_solver.py's
        # `dual_variables`), which is an unrelated quantity that just
        # happens to share the Greek letter in the math notation.
        self.dt = dt
        self.dof = dof
        self.sdf = sdf
        self.eps_margin = eps_margin
        self.sigma_obs = sigma_obs
        self.lambda_cbf = lambda_cbf
        self.fk_batch_fn = fk_batch_fn
        self.sphere_radii = sphere_radii

    def step(self, theta_gpmp2: np.ndarray, Sigma_t: np.ndarray, N: int,
             K_inv_diag: np.ndarray, barrier_batch_fn, rng: np.random.Generator) -> MPPIResult:
        """
        theta_gpmp2 : (T, n) GPMP2 reference (sampling mean, Step 2.1)
        Sigma_t     : (n,n) or (T,n,n) sampling covariance
        K_inv_diag  : (T, n) diagonal GP precision for gp_smoothness_cost
        barrier_batch_fn : callable V (N,T,n) -> h_values (N,T) barrier
                           values along every rollout (cbf/barrier.py)
        """
        V, eps = sample_trajectories(theta_gpmp2, Sigma_t, N, rng)

        gp_c = gp_smoothness_cost(V, theta_gpmp2, K_inv_diag)
        V_pos = self.fk_batch_fn(V[..., :self.dof])
        obs_c = obstacle_cost(V_pos, self.sdf, self.eps_margin, self.sigma_obs,
                               sphere_radii=self.sphere_radii)
        h_vals = barrier_batch_fn(V)
        cbf_c = cbf_soft_cost(h_vals)

        costs = total_cost(gp_c, obs_c, cbf_c, self.lambda_cbf)

        # Step 2.3: numerically stable softmax
        beta = np.min(costs)
        w_unnorm = np.exp(-(costs - beta) / self.lambda_mppi)
        weights = w_unnorm / np.sum(w_unnorm)

        # Step 2.4: weighted average
        u_mppi = np.tensordot(weights, V, axes=(0, 0))  # (T, n)

        return MPPIResult(u_mppi=u_mppi, weights=weights, costs=costs, V=V, eps=eps)

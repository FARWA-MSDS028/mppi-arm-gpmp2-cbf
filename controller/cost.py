"""
controller/cost.py
====================
Implements Step 2.2:

    J(V_i) = phi(V_i^T)                         [terminal cost]
           + sum_{t=0}^{T-1} q(V_i^t)            [running cost]

    q(V_i^t) = 1/2 ||V_i^t - mu_t||^2_K          [GP smoothness]
             + 1/2 ||h(V_i^t)||^2_Sigma_obs       [obstacle]
             + lambda_cbf * CBF_cost(V_i^t)       [safety, novel]

Each term reuses the SAME primitives as the GPMP2 factor graph
(planner/factor_graph.py: hinge_cost, SignedDistanceField) so the MPPI
running cost and the GPMP2 objective are mathematically consistent, as
required ("mathematically consistent... implementation").
"""

from __future__ import annotations
import numpy as np
from planner.factor_graph import hinge_cost, SignedDistanceField


def gp_smoothness_cost(V: np.ndarray, mu: np.ndarray, K_inv_diag: np.ndarray) -> np.ndarray:
    """
    1/2 ||V_t - mu_t||^2_K  approximated with a diagonal precision
    K_inv_diag (per-dimension inverse-variance weights taken from the
    diagonal of the GPMP2 marginal-covariance inverse) for O(N*T*n)
    evaluation instead of a full quadratic form per rollout.

    V          : (N, T, n)
    mu         : (T, n)      -- theta_gpmp2 reference
    K_inv_diag : (T, n)      -- per-timestep diagonal precision

    Returns (N, T) per-timestep cost.
    """
    diff = V - mu[None, :, :]
    return 0.5 * np.sum(diff * diff * K_inv_diag[None, :, :], axis=2)


def obstacle_cost(V_positions: np.ndarray, sdf: SignedDistanceField,
                   eps: float, sigma_obs: float,
                   sphere_radii: np.ndarray = None) -> np.ndarray:
    """
    1/2 ||h(V_t)||^2_Sigma_obs  with Sigma_obs = sigma_obs^2 * I.

    V_positions  : (N, T, M, 3) workspace positions of the M collision
                   spheres for every rollout/timestep (from forward
                   kinematics, robot/franka.py)
    sphere_radii : (M,) robot sphere radii (robot/franka.py's
                   `sphere_radii`); subtracted from the SDF query so this
                   matches planner/factor_graph.py's obstacle factor
                   exactly (same clearance definition in both the
                   long-horizon planner and the reactive MPPI layer, as
                   required for mathematical consistency). Defaults to
                   zero radius (point robot) if not provided.

    Returns (N, T) cost.
    """
    N, T, M, _ = V_positions.shape
    radii = sphere_radii if sphere_radii is not None else np.zeros(M)
    cost = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            h = np.zeros(M)
            for j in range(M):
                d = sdf.distance(V_positions[i, t, j], robot_radius=radii[j])
                c, _ = hinge_cost(d, eps)
                h[j] = c
            cost[i, t] = 0.5 * np.sum(h ** 2) / (sigma_obs ** 2)
    return cost


def cbf_soft_cost(h_values: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """
    CBF_cost(V_t): soft (differentiable-free, rollout-level) penalty
    that discourages the sampler from proposing rollouts that would
    later require large CBF-QP intervention. Implemented as a hinge on
    the barrier value itself:

        CBF_cost = max(0, margin - h(x_t))^2

    h_values : (N, T) barrier value h_theta(x) evaluated along each
               rollout (cbf/barrier.py)
    """
    return np.maximum(0.0, margin - h_values) ** 2


def total_cost(gp_cost: np.ndarray, obs_cost: np.ndarray, cbf_cost: np.ndarray,
               lambda_cbf: float, terminal_cost: np.ndarray | None = None) -> np.ndarray:
    """
    J(V_i) = terminal + sum_t [ gp_cost + obs_cost + lambda_cbf * cbf_cost ]

    All *_cost inputs are (N, T) except terminal_cost which is (N,).
    Returns (N,) total cost per rollout.
    """
    running = np.sum(gp_cost + obs_cost + lambda_cbf * cbf_cost, axis=1)
    if terminal_cost is not None:
        running = running + terminal_cost
    return running

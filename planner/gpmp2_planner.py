"""
planner/gpmp2_planner.py
=========================
Implements Step 1.4 (MAP inference) of the specification:

    theta* = argmin_theta  1/2||theta-mu||^2_K + 1/2||h(theta)||^2_Sigma_obs

solved by Levenberg-Marquardt (GTSAM's native optimizer performs the
linearize -> solve normal-equations -> iterate loop shown in the spec:

    (K^-1 + H^T Sigma_obs^-1 H) delta_theta* = K^-1(mu-theta_bar) - H^T Sigma_obs^-1 h(theta_bar)

This is exactly what GTSAM's LevenbergMarquardtOptimizer does internally
when the graph is composed of Gaussian PriorFactors + the CustomFactors
built in factor_graph.py -- we do not reimplement the linear algebra,
we assemble the *same* graph and let GTSAM's sparse solver (which
factorizes the same information matrix analytically) do it, which is
mathematically identical to hand-rolling the normal equations but
numerically far more robust (QR/Cholesky on the Bayes Tree).

Output
------
GPMP2Result:
    theta_star : (N+1, 2*dof) MAP trajectory
    covariances: list of (2*dof,2*dof) marginal covariances per pose
                 (from the Gaussian Bayes Tree / joint marginal, used
                 downstream as Sigma_0 for MPPI, Step 2.1)
"""

from __future__ import annotations
import numpy as np
import gtsam
from dataclasses import dataclass

from planner.factor_graph import build_gpmp2_graph, SignedDistanceField


@dataclass
class GPMP2Result:
    theta_star: np.ndarray          # (N+1, 2*dof)
    covariances: list               # length N+1, each (2*dof, 2*dof)
    keys: list
    graph: gtsam.NonlinearFactorGraph
    values: gtsam.Values
    iterations: int
    final_error: float


class GPMP2Planner:
    """
    Encapsulates graph construction + LM optimization + marginal
    covariance extraction.

    Time complexity: each LM iteration is O(N * dof^3) for the banded
    Cholesky solve (GP-prior factors give a block-tridiagonal
    information matrix, exploited natively by GTSAM's variable
    elimination ordering).
    Memory complexity: O(N * dof^2) for the sparse Bayes Tree.
    """

    def __init__(self, dof: int, dt: float, Qc: np.ndarray,
                 sdf: SignedDistanceField, fk_fn, sphere_offsets,
                 eps: float = 0.15, sigma_obs: float = 0.02,
                 start_cov_scale: float = 1e-4, goal_cov_scale: float = 1e-4):
        self.dof = dof
        self.dt = dt
        self.Qc = Qc
        self.sdf = sdf
        self.fk_fn = fk_fn
        self.sphere_offsets = sphere_offsets
        self.eps = eps
        self.sigma_obs = sigma_obs
        self.K0 = start_cov_scale * np.eye(2 * dof)
        self.KN = goal_cov_scale * np.eye(2 * dof)

    def plan(self, theta0: np.ndarray, theta_goal: np.ndarray, N: int,
              max_iters: int = 100, init_trajectory: np.ndarray = None) -> GPMP2Result:
        """
        init_trajectory : optional (N+1, 2*dof) initial guess for LM,
                           e.g. the previous cycle's converged theta_star
                           (shifted). WITHOUT this, every call re-solves
                           from a fresh straight-line interpolation
                           between theta0 and theta_goal -- since the
                           obstacle-avoidance problem is non-convex (at
                           least two qualitatively different routes exist,
                           left vs. right of an obstacle), a fresh solve
                           can converge to a DIFFERENT route than the
                           previous cycle chose, causing the planned
                           trajectory to flip back and forth cycle-to-
                           cycle -- this was a real, confirmed cause of
                           multi-cycle goal-error OSCILLATION (not just
                           slow convergence) in testing. Passing the
                           previous trajectory as init_trajectory keeps
                           LM anchored near the same local minimum instead
                           of being free to re-choose a different route
                           every cycle.
        """
        graph, keys, init = build_gpmp2_graph(
            N, self.dt, self.dof, self.Qc, theta0, theta_goal,
            self.K0, self.KN, self.fk_fn, self.sphere_offsets,
            self.sdf, self.eps, self.sigma_obs, init_trajectory=init_trajectory)

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(max_iters)
        params.setRelativeErrorTol(1e-6)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
        result = optimizer.optimize()

        theta_star = np.stack([result.atVector(k) for k in keys], axis=0)

        # Marginal covariances (Step 1.1 K, propagated to posterior) --
        # used as the initial MPPI sampling covariance Sigma_0 (Block 2).
        marginals = gtsam.Marginals(graph, result)
        covs = [marginals.marginalCovariance(k) for k in keys]

        return GPMP2Result(
            theta_star=theta_star,
            covariances=covs,
            keys=keys,
            graph=graph,
            values=result,
            iterations=optimizer.iterations(),
            final_error=graph.error(result),
        )

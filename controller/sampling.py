"""
controller/sampling.py
========================
Implements Step 2.1 exactly:

    Ordinary MPPI:      V_i = mu_prev + eps_i,        eps_i ~ N(0, Sigma)
    GPMP2-guided MPPI:  V_i = theta*_GPMP2 + eps_i,    eps_i ~ N(0, Sigma_t)

The key structural change from vanilla MPPI (which recursively reuses
its own previous control tape as the sampling mean) is that here the
mean is *replaced every planning cycle* by the fresh GPMP2 reference
trajectory theta*_GPMP2 -- this couples the reactive sampling layer to
the long-horizon planner.
"""

from __future__ import annotations
import numpy as np


def sample_trajectories(theta_gpmp2: np.ndarray, Sigma_t: np.ndarray,
                         N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Inputs
    ------
    theta_gpmp2 : (T, n) GPMP2 reference trajectory used as sampling MEAN
                  (n = control/action dimension being perturbed, e.g. m=7
                  torques or the joint-position block of theta, per the
                  user's cost design in controller/cost.py)
    Sigma_t     : (n, n) time-varying covariance (identical across the
                  horizon here; a (T,n,n) tensor is also accepted for a
                  per-timestep covariance schedule)
    N           : number of MPPI rollouts
    rng         : numpy random Generator (explicit for reproducibility)

    Returns
    -------
    V : (N, T, n) sampled trajectories, V_i = theta_gpmp2 + eps_i

    Time complexity: O(N*T*n) to draw + add.
    Memory complexity: O(N*T*n).
    """
    T, n = theta_gpmp2.shape
    if Sigma_t.ndim == 2:
        L = np.linalg.cholesky(Sigma_t + 1e-9 * np.eye(n))
        eps = rng.standard_normal((N, T, n)) @ L.T
    else:
        # (T, n, n) per-timestep covariance
        eps = np.zeros((N, T, n))
        for t in range(T):
            Lt = np.linalg.cholesky(Sigma_t[t] + 1e-9 * np.eye(n))
            eps[:, t, :] = rng.standard_normal((N, n)) @ Lt.T
    V = theta_gpmp2[None, :, :] + eps
    return V, eps

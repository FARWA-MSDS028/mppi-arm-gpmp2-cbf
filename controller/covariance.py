"""
controller/covariance.py
==========================
Implements Block 5A exactly.

Step 5A.1 -- intervention history (windowed outer-product average):
    delta_u_bar = (1/W) sum_{tau=t-W}^{t} delta_u_tau delta_u_tau^T

Step 5A.2 -- covariance update:
    P_dangerous  = delta_u_bar delta_u_bar^T / ||delta_u_bar||_F^2
    Sigma_target = Sigma_0 - eta * P_dangerous
    Sigma_{t+1}  = (1-beta) Sigma_t + beta * Sigma_target

  Single-step practical rule actually used online (also given in spec):
    Sigma_{t+1} = Sigma_t - eta * ||delta_u_t|| * delta_u_t delta_u_t^T
    Sigma_{t+1} = max(Sigma_{t+1}, Sigma_min * I)   [PSD / min-variance clip]

Both forms are implemented; `update_windowed` follows Step 5A.1+5A.2's
EMA-toward-a-danger-projected-target formulation, `update_online`
follows the single-sample practical rule. The closed loop
(experiments/adaptive_loop.py) uses `update_online` every control cycle
and `update_windowed` at the end of each planning cycle (matching how
the two formulas are presented in sequence in the spec).
"""

from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass, field


def _project_psd_floor(Sigma: np.ndarray, sigma_min: float,
                        sigma_max: float = None, fallback: np.ndarray = None) -> np.ndarray:
    """
    Clip Sigma to be PSD with a minimum-variance floor:
        Sigma <- clip(Sigma, sigma_min, sigma_max)   (elementwise on eigenvalues)

    Implemented via eigen-decomposition since the update is a rank-1
    downdate that can otherwise lose positive-definiteness.

    Hardening (fixes a real bug): `np.clip(nan, a_min, a_max)` returns
    `nan` unchanged -- it does NOT floor NaN to `a_min`. If an upstream
    intervention magnitude explodes (e.g. from a QP numerical blowup)
    and NaN/Inf enters `Sigma`, the eigenvalue clip silently let NaN
    through before, so the "positive-definite" guarantee only held for
    finite inputs. This version explicitly checks for non-finite values
    first and falls back to a known-good matrix (`fallback`, typically
    Sigma0) rather than propagating NaN. An optional `sigma_max` caps
    eigenvalues from above too -- the original only floored a minimum,
    so nothing stopped Sigma from growing without bound if some other
    update path pushed eigenvalues up (defense in depth alongside fixing
    the actual source of any blowup upstream).
    """
    if fallback is None:
        fallback = sigma_min * np.eye(Sigma.shape[0])
    if not np.all(np.isfinite(Sigma)):
        return fallback.copy()

    Sigma_sym = 0.5 * (Sigma + Sigma.T)
    w, V = np.linalg.eigh(Sigma_sym)
    if not np.all(np.isfinite(w)):
        return fallback.copy()
    w_clipped = np.clip(w, sigma_min, sigma_max)
    result = (V * w_clipped) @ V.T
    if not np.all(np.isfinite(result)):
        return fallback.copy()
    return result


@dataclass
class CovarianceSteering:
    """
    n         : action/perturbation dimension (matches Sigma_t in MPPI sampling)
    eta       : learning rate
    beta      : EMA blend factor for the windowed update
    W         : window length for the intervention history
    sigma_min : minimum-variance floor (PSD safeguard)
    sigma_max : maximum-variance ceiling (defense-in-depth PSD safeguard;
                also caps how far one abnormally large intervention can
                distort Sigma_t before the next few updates recover it)
    max_intervention_norm : hard clip on ||delta_u_t|| before it enters
                the update rule -- if an upstream numerical issue (e.g. a
                QP scaling/units problem) produces an enormous
                intervention, this stops it from poisoning Sigma_t with
                a single pathological update; the underlying cause
                still needs fixing, this only prevents it from cascading
                into a broken covariance matrix.
    """
    n: int
    Sigma0: np.ndarray
    eta: float = 0.05
    beta: float = 0.3
    W: int = 20
    sigma_min: float = 1e-4
    sigma_max: float = 10.0
    max_intervention_norm: float = 50.0

    Sigma_t: np.ndarray = field(init=False)
    history: deque = field(init=False)
    trace_log: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.Sigma_t = self.Sigma0.copy()
        self.history = deque(maxlen=self.W)
        self.trace_log.append(self.Sigma_t.copy())
    def reset(self):
        """
        Reset covariance steering to its initial exploration confidence.
        """
        self.Sigma_t = self.Sigma0.copy()
        self.history.clear()
    # ---- Step 5A.2, practical single-sample rule ------------------------
    def update_online(self, delta_u_t: np.ndarray) -> np.ndarray:
        """
        Sigma_{t+1} = Sigma_t - eta ||delta_u_t|| delta_u_t delta_u_t^T
        clipped to keep PSD / min variance.

        Time complexity O(n^2) (outer product) + O(n^3) for the
        eigen-clip safeguard.
        """
        norm = np.linalg.norm(delta_u_t)
        if norm > self.max_intervention_norm or not np.isfinite(norm):
            # Defensive clip -- see class docstring. Scale delta_u_t down
            # to the max norm rather than dropping the update entirely,
            # so direction information isn't discarded, just magnitude.
            if np.isfinite(norm) and norm > 0:
                delta_u_t = delta_u_t * (self.max_intervention_norm / norm)
            else:
                delta_u_t = np.zeros_like(delta_u_t)
            norm = self.max_intervention_norm
        outer = np.outer(delta_u_t, delta_u_t)
        Sigma_next = self.Sigma_t - self.eta * norm * outer
        Sigma_next = _project_psd_floor(Sigma_next, self.sigma_min, self.sigma_max,
                                         fallback=self.Sigma0)
        self.Sigma_t = Sigma_next
        self.history.append(delta_u_t.copy())
        self.trace_log.append(self.Sigma_t.copy())
        return self.Sigma_t

    # ---- Steps 5A.1 + 5A.2, windowed EMA-toward-target -------------------
    def update_windowed(self) -> np.ndarray:
        """
        Uses the last W recorded delta_u samples to build the danger
        projection P_dangerous and blend Sigma_t toward
        Sigma_target = Sigma0 - eta * P_dangerous via an EMA with rate beta.
        """
        if len(self.history) == 0:
            return self.Sigma_t
        stacked = np.stack(list(self.history), axis=0)  # (W', n)
        delta_u_bar = np.mean(
            np.einsum('wi,wj->wij', stacked, stacked), axis=0)  # (n,n)
        fro2 = np.sum(delta_u_bar * delta_u_bar)
        if fro2 < 1e-12:
            P_dangerous = np.zeros((self.n, self.n))
        else:
            P_dangerous = (delta_u_bar @ delta_u_bar) / fro2
        Sigma_target = self.Sigma0 - self.eta * P_dangerous
        Sigma_target = _project_psd_floor(Sigma_target, self.sigma_min, self.sigma_max,
                                           fallback=self.Sigma0)
        self.Sigma_t = (1 - self.beta) * self.Sigma_t + self.beta * Sigma_target
        self.Sigma_t = _project_psd_floor(self.Sigma_t, self.sigma_min, self.sigma_max,
                                           fallback=self.Sigma0)
        self.trace_log.append(self.Sigma_t.copy())
        return self.Sigma_t
    def eigen_history(self):
        """Returns list of eigenvalue arrays over time, for Experiment 6
        ('covariance matrices, eigenvalues') plotting."""
        return [np.sort(np.linalg.eigvalsh(S))[::-1] for S in self.trace_log]

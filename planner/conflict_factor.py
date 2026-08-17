"""
planner/conflict_factor.py
============================
Implements Block 5B, Steps 5B.1-5B.2 exactly.

Step 5B.1 -- trigger condition:
    if ||delta_u_t|| > tau_conflict  AND  h_theta(x_t) < tau_safe:
        x_t is a "conflict state"

Step 5B.2 -- Gaussian conflict factor:
    f_conflict(theta_i; x_danger, sigma_conflict, w) =
        exp{ -w / (2 sigma_conflict^2) * ||theta_i - x_danger||^2 }

This is a plain isotropic Gaussian factor on a SINGLE pose (the pose
nearest in the trajectory to the timestep where the conflict was
detected), with the intervention magnitude w = ||delta_u_t|| controlling
the factor's precision (stronger intervention -> tighter/weightier
penalty pulling the trajectory away from x_danger).

Also implements the alternative CBF-value formulation (Step 5B.2,
"Alternative formulation"):
    f_conflict(theta_i) = exp{ -1/2 [max(0,-h_theta(theta_i))]^2 * w }
which is exposed as `conflict_factor_cbf_variant` and used when a
direct barrier-based penalty (rather than a distance-to-danger-point
penalty) is preferred.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

import gtsam
from gtsam import CustomFactor, noiseModel


@dataclass
class ConflictEvent:
    """Record of a single detected conflict (Step 5B.1)."""
    timestep: int
    x_danger: np.ndarray      # state theta at time of conflict
    intervention_norm: float  # w = ||delta_u_t||
    barrier_value: float      # h_theta(x_t)


def detect_conflict(delta_u: np.ndarray, h_value: float,
                     tau_conflict: float, tau_safe: float) -> bool:
    """
    Step 5B.1 trigger condition.

    Inputs
    ------
    delta_u      : u* - u_mppi at this timestep (R^m)
    h_value      : barrier value h_theta(x_t)
    tau_conflict : minimum intervention-magnitude threshold
    tau_safe     : safety-margin threshold

    Returns True iff both conditions hold simultaneously.
    Time complexity: O(m).
    """
    return (np.linalg.norm(delta_u) > tau_conflict) and (h_value < tau_safe)


def make_conflict_factor(key_i, x_danger: np.ndarray, sigma_conflict: float,
                          w: float) -> CustomFactor:
    """
    f_conflict(theta_i) = exp{ -w/(2 sigma_conflict^2) ||theta_i - x_danger||^2 }

    Implemented as a CustomFactor with error e = sqrt(w) / sigma_conflict * (theta_i - x_danger),
    so that -1/2 ||e||^2 reproduces the target exponent exactly (GTSAM
    factors are always expressed as exp{-1/2 ||error||^2_Sigma}; folding
    sqrt(w)/sigma_conflict into the error term realizes an isotropic
    noise model of covariance (sigma_conflict^2 / w) I).

    Time complexity O(d) per evaluation, d = dim(theta_i).
    """
    dim = x_danger.shape[0]
    scale = np.sqrt(w) / sigma_conflict

    def error_func(this: CustomFactor, values: gtsam.Values, jacobians):
        th = values.atVector(this.keys()[0])
        err = scale * (th - x_danger)
        if jacobians is not None:
            jacobians[0] = scale * np.eye(dim)
        return err

    noise = noiseModel.Unit.Create(dim)  # error already pre-scaled
    return CustomFactor(noise, [key_i], error_func)


def conflict_factor_cbf_variant(key_i, barrier_fn, w: float) -> CustomFactor:
    """
    Alternative formulation (Step 5B.2):
        f_conflict(theta_i) = exp{ -1/2 [max(0,-h_theta(theta_i))]^2 * w }

    barrier_fn: callable theta_i (R^d) -> (h: float, grad_h: (d,))
                (learned CBF value + its gradient, cbf/barrier.py)
    """
    def error_func(this: CustomFactor, values: gtsam.Values, jacobians):
        th = values.atVector(this.keys()[0])
        h, grad_h = barrier_fn(th)
        viol = max(0.0, -h)
        err = np.array([np.sqrt(w) * viol])
        if jacobians is not None:
            # d(err)/d(theta) = sqrt(w) * d(viol)/d(theta)
            # d(viol)/d(theta) = -grad_h if h<0 else 0
            dviol = -grad_h if h < 0 else np.zeros_like(grad_h)
            jacobians[0] = (np.sqrt(w) * dviol).reshape(1, -1)
        return err

    noise = noiseModel.Unit.Create(1)
    return CustomFactor(noise, [key_i], error_func)


class ConflictFactorManager:
    """
    Bookkeeping object that accumulates ConflictEvents over a run and
    turns them into factors ready for insertion via iSAM2
    (planner/isam_update.py, Step 5B.4). Keeping this separate from the
    graph itself means the SAME manager can be inspected for
    plots/main.py Experiment 7 (conflict map).
    """

    def __init__(self, tau_conflict: float, tau_safe: float,
                 sigma_conflict: float = 0.3):
        self.tau_conflict = tau_conflict
        self.tau_safe = tau_safe
        self.sigma_conflict = sigma_conflict
        self.events: list[ConflictEvent] = []

    def check_and_record(self, timestep: int, x_t: np.ndarray,
                          delta_u: np.ndarray, h_value: float) -> Optional[ConflictEvent]:
        if detect_conflict(delta_u, h_value, self.tau_conflict, self.tau_safe):
            ev = ConflictEvent(timestep=timestep, x_danger=x_t.copy(),
                                intervention_norm=float(np.linalg.norm(delta_u)),
                                barrier_value=h_value)
            self.events.append(ev)
            return ev
        return None

    def factor_for_event(self, key_i, ev: ConflictEvent) -> CustomFactor:
        return make_conflict_factor(key_i, ev.x_danger, self.sigma_conflict,
                                     ev.intervention_norm)

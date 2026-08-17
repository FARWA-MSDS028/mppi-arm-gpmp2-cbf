"""
cbf/feasibility.py
=====================
Implements Block 4 / Stage 7 exactly: extracts and stores everything
available from the QP solve.

Info 1: barrier value        h_t = h_theta(x_t)
Info 2: intervention magnitude ||delta_u_t|| = ||u*_t - u_mppi,t||
Info 3: active constraints / dual variables lambda*_t
Info 4: Lie derivative values Lf h, Lg h

This module owns a running FeasibilityLog used by both the covariance
steering update (controller/covariance.py) and the conflict-factor
trigger (planner/conflict_factor.py), and is the data source for
Experiment 5's plots (active constraints, intervention magnitude, dual
variables, barrier values, slack, safety margin).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from cbf.qp_solver import CBFQPResult, UnsafeDetection


@dataclass
class FeasibilitySample:
    t: int
    barrier_value: float
    intervention: np.ndarray
    intervention_magnitude: float
    dual_variables: np.ndarray
    active_constraints: list
    Lf_h: float
    Lg_h: np.ndarray
    slack: float
    safety_margin: float
    was_unsafe_pre_qp: bool


class FeasibilityLog:
    """O(1) append per control cycle; O(T) memory for a T-step episode."""

    def __init__(self):
        self.samples: list[FeasibilitySample] = []

    def record(self, t: int, unsafe: UnsafeDetection, qp: CBFQPResult):
        self.samples.append(FeasibilitySample(
            t=t,
            barrier_value=qp.safety_margin,
            intervention=qp.intervention,
            intervention_magnitude=qp.intervention_magnitude,
            dual_variables=qp.dual_variables,
            active_constraints=qp.active_constraints,
            Lf_h=unsafe.Lf_h,
            Lg_h=unsafe.Lg_h,
            slack=qp.slack,
            safety_margin=qp.safety_margin,
            was_unsafe_pre_qp=unsafe.is_unsafe,
        ))

    # ---- convenience array views for plotting/analysis -------------------
    def barrier_trace(self) -> np.ndarray:
        return np.array([s.barrier_value for s in self.samples])

    def intervention_trace(self) -> np.ndarray:
        return np.array([s.intervention_magnitude for s in self.samples])

    def slack_trace(self) -> np.ndarray:
        return np.array([s.slack for s in self.samples])

    def active_constraint_counts(self) -> np.ndarray:
        return np.array([len(s.active_constraints) for s in self.samples])

    def dual_norm_trace(self) -> np.ndarray:
        return np.array([np.linalg.norm(s.dual_variables) for s in self.samples])

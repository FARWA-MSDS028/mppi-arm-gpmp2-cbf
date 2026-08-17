"""
controller/tracking.py
=========================
Low-level PD/gravity-compensated tracking controller that converts a
desired joint-POSITION command (MPPI's raw output, sampled around a
GPMP2 position trajectory) into an actual joint TORQUE command.

This fixes a real units mismatch: MPPI samples V_i = theta_GPMP2 + eps_i
in POSITION space (rad, magnitude ~0.1-3), but cbf.qp_solver.CBFQPSolver
and the physical actuators operate in TORQUE space (Nm, bounded by
robot.franka.TAU_MAX ~ tens of Nm). Feeding a position-scale number
directly into a torque-scale CBF-QP means genuinely dangerous torques
(shown by direct testing to require magnitude ~50+ Nm to violate the
barrier at a typical configuration) are never actually proposed by
MPPI, no matter how obstacle-averse MPPI's own cost function is tuned
(lambda_cbf) -- the candidate "u" is simply never large enough in
magnitude to reach the danger region in torque space. That was the
real reason the full closed loop showed zero CBF/conflict intervention
across an entire lambda_cbf sweep.

Fix: track MPPI's position output with a PD (+ gravity compensation)
law, and let the CBF-QP filter THIS torque, not the raw position
value. u_mppi remains a meaningful "desired configuration" signal;
this module is what turns it into something the actuators (and the
CBF-QP, which models actuator-space dynamics) actually operate on.
"""
from __future__ import annotations
import numpy as np


def pd_torque(q_des: np.ndarray, q: np.ndarray, qdot: np.ndarray,
              Kp, Kd, gravity_comp: np.ndarray = None) -> np.ndarray:
    """
    tau = Kp*(q_des - q) - Kd*qdot + gravity_comp(q)

    Standard joint-space PD tracking law with optional gravity
    compensation. This is what actually gets tested by the CBF-QP and
    (after CBF filtering) applied to the plant -- MPPI's raw position
    sample is only ever a SETPOINT for this controller, never a torque
    command by itself.

    Inputs
    ------
    q_des        : (dof,) desired joint position (MPPI's u_mppi)
    q, qdot      : (dof,) current joint state
    Kp, Kd       : scalar or (dof,) proportional/derivative gains
    gravity_comp : (dof,) optional gravity compensation torque
                   (robot/franka.py FrankaModel.gravity(q)) -- without
                   this the PD law alone must fight gravity too, which
                   distorts the tracking-error-to-torque relationship
                   and makes Kp/Kd harder to reason about physically.

    Returns (dof,) torque command, NOT yet clipped to actuator limits
    (the CBF-QP's box constraints handle that; this function's output
    is exactly what "u_mppi" means from the CBF-QP's point of view).

    Time complexity: O(dof).
    """
    tau = Kp * (q_des - q) - Kd * qdot
    if gravity_comp is not None:
        tau = tau + gravity_comp
    return tau

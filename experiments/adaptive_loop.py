"""
experiments/adaptive_loop.py
===============================
Experiments 5-8 (spec): thin wrapper around main.run_closed_loop that
adds the bookkeeping needed for:

  5. Feasibility Information plots (active constraints, intervention
     magnitude, dual variables, barrier values, slack, safety margin)
     -- already logged by cbf.feasibility.FeasibilityLog inside the loop.
  6. Covariance Steering plots (sampling before/after, covariance
     matrices, eigenvalues, trajectory improvement) -- from
     controller.covariance.CovarianceSteering.trace_log / eigen_history().
  7. Conflict Factors plots (where added, updated factor graph, iSAM2
     updates, before/after trajectory) -- from
     planner.conflict_factor.ConflictFactorManager.events and the
     theta_star history recorded here.
  8. Full framework: success/failure + timing, recorded per cycle.
"""

from __future__ import annotations
import time
import numpy as np

from main import run_closed_loop


def run_full_framework(mjcf_path: str, n_planning_cycles: int = 20, **kwargs):
    t0 = time.time()
    history, feas_log, cov_steer = run_closed_loop(
        mjcf_path=mjcf_path, n_planning_cycles=n_planning_cycles, **kwargs)
    elapsed = time.time() - t0

    q_traj = np.array(history["q"])
    h_traj = np.array(history["h"])

    summary = {
        "wall_clock_s": elapsed,
        "n_control_steps": len(history["q"]),
        "n_conflicts_inserted": len(history["conflicts"]),
        "min_barrier_value": float(np.min(h_traj)) if len(h_traj) else None,
        "final_joint_error_inf_norm": None,  # filled by caller with q_goal
    }
    return history, feas_log, cov_steer, summary

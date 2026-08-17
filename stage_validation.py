"""
stage_validation.py
=====================
Runs the pipeline stage-by-stage, exactly as requested: each stage must
independently succeed (reach the target) before the next is attempted.
Uses your ACTUAL experiments/*.py files -- no reconstruction, no
assumptions about what they currently contain.

Stage 1: MPPI baseline alone (no GPMP2 prior, no CBF-QP filter).
Stage 2: GPMP2 -> MPPI (GPMP2 trajectory as sampling mean, still no CBF).
Stage 3: GPMP2 -> MPPI -> CBF-QP (full safety filter added).

Same success threshold used throughout this session: goal_err < 0.15 rad
(inf-norm joint error). Math/definitions are unchanged from everything
we've built together -- this script only orchestrates and reports.
"""
from __future__ import annotations
import numpy as np

from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from cbf.barrier import DistanceBarrier

from experiments.baseline_mppi import run_baseline_mppi
from experiments.gpmp2_mppi import run_gpmp2_mppi
from experiments.safe_mppi import run_safe_mppi

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
SUCCESS_THRESHOLD = 0.15  # rad, same threshold used all session

# Real obstacle (from the verified-safe-start scenario), used for Stage 3 only.
OBSTACLE_CENTER = (0.20, 0.09, 0.85)
OBSTACLE_RADIUS = 0.08
D_SAFE = 0.03

# Effectively "no obstacle" for Stages 1-2, per your request to isolate
# MPPI/GPMP2 reachability from any obstacle-avoidance behavior first.
FAR_AWAY_CENTER = (5.0, 5.0, 5.0)
FAR_AWAY_RADIUS = 0.01


def make_env_franka_sdf(obstacle_center, obstacle_radius):
    env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=obstacle_center,
                          obstacle_radius=obstacle_radius)
    import mujoco
    franka = FrankaModel(env.model, mujoco.MjData(env.model))
    sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
    env.reset(Q0)
    return env, franka, sdf


def report(stage_name, goal_err, min_clear, extra=""):
    success = goal_err < SUCCESS_THRESHOLD
    print(f"\n{'='*70}\n{stage_name}\n{'='*70}")
    print(f"  final goal_err:     {goal_err:.4f}  (threshold: {SUCCESS_THRESHOLD})")
    print(f"  min clearance d(x): {min_clear:.4f}")
    if extra:
        print(f"  {extra}")
    print(f"  SUCCESS: {success}")
    return success


def stage1_baseline_mppi():
    env, franka, sdf = make_env_franka_sdf(FAR_AWAY_CENTER, FAR_AWAY_RADIUS)
    log = run_baseline_mppi(env, franka, sdf, q_goal=Q_GOAL, rng_seed=0)
    goal_err = log["goal_error"][-1]
    min_clear = min(log["dist"])
    n_steps_to_converge = next((i for i, e in enumerate(log["goal_error"])
                                 if e < SUCCESS_THRESHOLD), None)
    extra = f"steps to first reach threshold: {n_steps_to_converge}"
    return report("STAGE 1: MPPI baseline alone (no obstacle, no GPMP2, no CBF)",
                   goal_err, min_clear, extra)


def stage2_gpmp2_mppi():
    env, franka, sdf = make_env_franka_sdf(FAR_AWAY_CENTER, FAR_AWAY_RADIUS)
    log, theta_q = run_gpmp2_mppi(env, franka, sdf, q0=Q0, q_goal=Q_GOAL, rng_seed=0)
    goal_err = log["goal_error"][-1]
    min_clear = min(log["dist"])
    n_steps_to_converge = next((i for i, e in enumerate(log["goal_error"])
                                 if e < SUCCESS_THRESHOLD), None)
    extra = f"steps to first reach threshold: {n_steps_to_converge}"
    return report("STAGE 2: GPMP2 -> MPPI (no obstacle, no CBF)",
                   goal_err, min_clear, extra)


def stage3_safe_mppi():
    env, franka, sdf = make_env_franka_sdf(OBSTACLE_CENTER, OBSTACLE_RADIUS)
    barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                               sdf=sdf, dof=DOF, d_safe=D_SAFE)
    h0 = barrier.forward(np.concatenate([Q0, np.zeros(DOF)]))
    print(f"\nPre-flight check: h(q0) = {h0:.4f}  ({'SAFE START' if h0 >= 0 else 'UNSAFE START -- ABORTING'})")
    if h0 < 0:
        return False

    log, feas_log = run_safe_mppi(env, franka, sdf, barrier, q0=Q0, q_goal=Q_GOAL, rng_seed=0)
    goal_err = log["goal_error"][-1]
    min_clear = min(log["dist"])
    n_unsafe_pre_qp = sum(log["unsafe_flags"])
    n_total = len(log["unsafe_flags"])
    avg_intervention_when_unsafe = (
        float(np.mean([iv for iv, flag in zip(log["intervention"], log["unsafe_flags"]) if flag]))
        if n_unsafe_pre_qp > 0 else 0.0)
    avg_intervention_when_safe = (
        float(np.mean([iv for iv, flag in zip(log["intervention"], log["unsafe_flags"]) if not flag]))
        if (n_total - n_unsafe_pre_qp) > 0 else 0.0)
    extra = (f"unsafe pre-QP: {n_unsafe_pre_qp}/{n_total} steps | "
             f"avg intervention when MPPI was unsafe: {avg_intervention_when_unsafe:.4f} | "
             f"avg intervention when MPPI was ALREADY safe: {avg_intervention_when_safe:.4f} "
             f"(should be near 0 -- confirms QP leaves safe controls essentially unchanged)")
    return report("STAGE 3: GPMP2 -> MPPI -> CBF-QP (real obstacle, hard safety filter)",
                   goal_err, min_clear, extra)


if __name__ == "__main__":
    print("Running staged validation. Each stage gates the next -- "
          "if a stage fails, later stages are skipped.\n")

    ok1 = stage1_baseline_mppi()
    if not ok1:
        print("\nSTOPPING: Stage 1 failed. Per your instructions, later stages "
              "won't be attempted until this is fixed -- see the reachability/"
              "singularity tests suggested earlier to diagnose why.")
    else:
        ok2 = stage2_gpmp2_mppi()
        if not ok2:
            print("\nSTOPPING: Stage 2 failed even though Stage 1 succeeded. This "
                  "would mean GPMP2's own trajectory is actively hurting MPPI's "
                  "ability to reach the goal, compared to MPPI operating alone -- "
                  "worth investigating GPMP2's cost weighting (Qc) directly.")
        else:
            ok3 = stage3_safe_mppi()
            if not ok3:
                print("\nSTOPPING: Stage 3 failed even though Stages 1-2 succeeded. "
                      "This would isolate the problem specifically to the CBF-QP's "
                      "interaction with the obstacle -- e.g. the QP filtering too "
                      "aggressively and blocking forward progress near the obstacle.")
            else:
                print("\nALL THREE STAGES PASSED INDEPENDENTLY. Safe to proceed to "
                      "combining GPMP2+MPPI+CBF-QP+conflict-factors in the full "
                      "closed loop (main.py / run_experiments.py --only full).")

"""
run_experiments.py
=====================
Runs all four required experiments in sequence and reports the full
metric set a paper needs, including MINIMUM clearance reached during
the run (not just the final value) -- if you see 0/N unsafe steps in
the MPPI+CBF experiment, check min_clearance first: it almost always
means the obstacle is placed too far from the path the arm actually
takes, not a bug in the CBF-QP. Run find_obstacle_placement.py to see
where the arm's own straight-line path passes in workspace, then use
--obstacle-x/y/z/--obstacle-radius/--d-safe below to place the
obstacle somewhere the arm can't avoid cheaply.

Usage
-----
    python run_experiments.py --mjcf assets/panda.xml
    python run_experiments.py --mjcf assets/panda.xml --only safe_mppi \
        --obstacle-x 0.35 --obstacle-y 0.0 --obstacle-z 0.45 \
        --obstacle-radius 0.12 --d-safe 0.08
    python run_experiments.py --mjcf assets/panda.xml --seeds 10
"""

from __future__ import annotations
import argparse
import time
import numpy as np

from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from cbf.barrier import DistanceBarrier

from experiments.baseline_mppi import run_baseline_mppi
from experiments.gpmp2_mppi import run_gpmp2_mppi
from experiments.safe_mppi import run_safe_mppi
from experiments.adaptive_loop import run_full_framework

from plots.main import (
    plot_trajectory_comparison, plot_rollout_vs_mean, plot_cost_convergence,
    plot_multi_cost_convergence, plot_barrier_and_intervention,
    plot_covariance_evolution, plot_success_rate_and_timing,
)

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
GOAL_SUCCESS_THRESHOLD = 0.15  # rad, inf-norm joint error counted as "reached goal"


def make_env_franka_sdf(mjcf_path: str, obstacle_center=(0.5, 0.0, 0.4),
                         obstacle_radius: float = 0.08):
    import mujoco
    env = MujocoFrankaEnv(mjcf_path=mjcf_path, obstacle_center=obstacle_center,
                          obstacle_radius=obstacle_radius)
    # Own MjData, separate from env.data -- see main.py::build_default_system's
    # comment for why sharing it with the environment is a real bug (fk_batch
    # overwrites qpos with rollout candidates that must never leak into the
    # actual simulated state).
    franka = FrankaModel(env.model, mujoco.MjData(env.model))
    sdf = SignedDistanceField(np.array([env.obstacle_center]),
                              np.array([env.obstacle_radius]))
    env.reset(Q0)
    return env, franka, sdf


def _control_effort(u_list) -> float:
    """sum_t ||u_t||^2 over the run -- a standard control-effort metric."""
    return float(sum(np.sum(np.square(u)) for u in u_list))


def _metrics_from_log(log: dict, u_key: str, wall_clock_s: float) -> dict:
    final_goal_error = log["goal_error"][-1] if log.get("goal_error") else float("nan")
    min_clearance = float(np.min(log["dist"])) if log.get("dist") else float("nan")
    return {
        "wall_clock_s": wall_clock_s,
        "control_effort": _control_effort(log[u_key]),
        "final_cost": log["cost_history"][-1] if log.get("cost_history") else float("nan"),
        "final_goal_error": final_goal_error,
        "min_clearance": min_clearance,
        "success": bool(final_goal_error < GOAL_SUCCESS_THRESHOLD),
        "n_conflicts_inserted": 0,  # only meaningful for the full loop; kept for
    }                                # a uniform summary dict across experiments


def print_metrics_table(all_metrics: dict):
    print("\n" + "=" * 92)
    print(f"{'Experiment':<18}{'time(s)':>9}{'ctrl effort':>13}{'final cost':>12}"
          f"{'goal err':>10}{'min clear':>11}{'success':>9}")
    print("-" * 92)
    for label, m in all_metrics.items():
        print(f"{label:<18}{m['wall_clock_s']:>9.2f}{m['control_effort']:>13.3f}"
              f"{m['final_cost']:>12.4f}{m['final_goal_error']:>10.4f}"
              f"{m['min_clearance']:>11.4f}{str(m['success']):>9}")
    print("=" * 92)
    print("Note: if min_clearance stayed well above 0 (comfortably larger than\n"
          "d_safe) for the MPPI+CBF / full-loop rows, the CBF never had to act --\n"
          "that's a scenario-design issue (obstacle too far from the path), not\n"
          "evidence the CBF-QP is broken. See find_obstacle_placement.py.")


def experiment_1_baseline(mjcf_path: str, rng_seed: int = 0,
                            obstacle_center=(0.5, 0.0, 0.4), obstacle_radius: float = 0.08):
    env, franka, sdf = make_env_franka_sdf(mjcf_path, obstacle_center, obstacle_radius)
    t0 = time.time()
    log = run_baseline_mppi(env, franka, sdf, q_goal=Q_GOAL, rng_seed=rng_seed)
    elapsed = time.time() - t0
    return log, elapsed


def experiment_2_gpmp2_mppi(mjcf_path: str, rng_seed: int = 0,
                              obstacle_center=(0.5, 0.0, 0.4), obstacle_radius: float = 0.08,
                              gpmp2_eps: float = 0.15):
    env, franka, sdf = make_env_franka_sdf(mjcf_path, obstacle_center, obstacle_radius)
    t0 = time.time()
    log, theta_q = run_gpmp2_mppi(env, franka, sdf, q0=Q0, q_goal=Q_GOAL, rng_seed=rng_seed,
                                    gpmp2_eps=gpmp2_eps)
    elapsed = time.time() - t0
    return log, theta_q, elapsed


def experiment_3_safe_mppi(mjcf_path: str, rng_seed: int = 0,
                             obstacle_center=(0.5, 0.0, 0.4), obstacle_radius: float = 0.08,
                             d_safe: float = 0.05, gpmp2_eps: float = 0.15):
    env, franka, sdf = make_env_franka_sdf(mjcf_path, obstacle_center, obstacle_radius)
    barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                               sdf=sdf, dof=DOF, d_safe=d_safe)
    t0 = time.time()
    log, feas_log = run_safe_mppi(env, franka, sdf, barrier, q0=Q0, q_goal=Q_GOAL,
                                    rng_seed=rng_seed, gpmp2_eps=gpmp2_eps)
    elapsed = time.time() - t0
    return log, feas_log, elapsed


def experiment_4_full_loop(mjcf_path: str, cycles: int = 5, rng_seed: int = 0,
                            obstacle_center=(0.5, 0.0, 0.4), obstacle_radius: float = 0.08,
                            d_safe: float = 0.05, tau_conflict: float = 0.05,
                            tau_safe: float = 0.2, lambda_cbf: float = 1.0,
                            gpmp2_eps: float = 0.03, N_horizon: int = 30):
    t0 = time.time()
    history, feas_log, cov_steer, summary = run_full_framework(
        mjcf_path, n_planning_cycles=cycles, rng_seed=rng_seed,
        obstacle_center=obstacle_center, obstacle_radius=obstacle_radius, d_safe=d_safe,
        tau_conflict=tau_conflict, tau_safe=tau_safe, lambda_cbf=lambda_cbf,
        gpmp2_eps=gpmp2_eps, N_horizon=N_horizon)
    elapsed = time.time() - t0
    return history, feas_log, cov_steer, summary, elapsed


def run_multi_seed_success_rate(mjcf_path: str, n_seeds: int, obstacle_center, obstacle_radius):
    """
    Runs baseline and GPMP2+MPPI over n_seeds and reports the fraction
    of seeds that reached the goal -- an actual success RATE, not a
    single-run flag.
    """
    print(f"\n=== Success rate over {n_seeds} seeds (baseline vs GPMP2+MPPI) ===")
    results = {"baseline": [], "gpmp2+mppi": []}
    for seed in range(n_seeds):
        log_b, _ = experiment_1_baseline(mjcf_path, rng_seed=seed,
                                           obstacle_center=obstacle_center,
                                           obstacle_radius=obstacle_radius)
        results["baseline"].append(log_b["goal_error"][-1] < GOAL_SUCCESS_THRESHOLD)

        log_g, _, _ = experiment_2_gpmp2_mppi(mjcf_path, rng_seed=seed,
                                                obstacle_center=obstacle_center,
                                                obstacle_radius=obstacle_radius)
        results["gpmp2+mppi"].append(log_g["goal_error"][-1] < GOAL_SUCCESS_THRESHOLD)

    for label, flags in results.items():
        rate = np.mean(flags)
        print(f"  {label:<15} success rate: {rate:.1%}  ({sum(flags)}/{n_seeds} seeds)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=str, default="assets/panda.xml")
    parser.add_argument("--only", type=str, default="all",
                         choices=["all", "baseline", "gpmp2_mppi", "safe_mppi", "full"])
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=1,
                         help="If >1, additionally run baseline/GPMP2+MPPI over "
                              "this many seeds and report a real success RATE.")
    parser.add_argument("--obstacle-x", type=float, default=0.5)
    parser.add_argument("--obstacle-y", type=float, default=0.0)
    parser.add_argument("--obstacle-z", type=float, default=0.4)
    parser.add_argument("--obstacle-radius", type=float, default=0.08)
    parser.add_argument("--d-safe", type=float, default=0.05,
                         help="CBF safety margin d_safe in h(x)=d(x)-d_safe")
    parser.add_argument("--tau-conflict", type=float, default=0.05,
                         help="Conflict trigger: intervention magnitude threshold "
                              "(full-loop only). Lower = conflict factors fire more "
                              "easily. Default lowered from the spec's illustrative "
                              "0.5, which rarely coincides with low margin once "
                              "covariance steering is already shrinking risk.")
    parser.add_argument("--tau-safe", type=float, default=0.2,
                         help="Conflict trigger: barrier-value threshold h(x) < tau_safe "
                              "(full-loop only).")
    parser.add_argument("--lambda-cbf", type=float, default=1.0,
                         help="Weight of MPPI's own soft obstacle-avoidance cost "
                              "(full-loop only). At the default 1.0, MPPI already "
                              "avoids risk on its own and the CBF-QP/conflict factors "
                              "rarely get anything to correct. Lower this (e.g. 0.05) "
                              "to demonstrate genuine CBF intervention and conflict "
                              "escalation in the full closed loop.")
    parser.add_argument("--gpmp2-eps", type=float, default=0.03,
                         help="GPMP2's OWN obstacle-avoidance margin (full-loop only). "
                              "THIS WAS THE ACTUAL ROOT CAUSE of conflicts never "
                              "firing: GPMP2's old default (0.15) was larger than "
                              "every d_safe we tried, so the long-horizon planner was "
                              "already staying further from the obstacle than the CBF "
                              "required -- the CBF/conflict layers had nothing left to "
                              "do regardless of lambda_cbf. Keep --gpmp2-eps LESS than "
                              "--d-safe for the CBF to have genuine work.")
    parser.add_argument("--gpmp2-sigma-obs", type=float, default=0.02,
                         help="GPMP2 obstacle-cost sharpness")
    parser.add_argument("--n-horizon", type=int, default=30,
                         help="MPPI/GPMP2 planning horizon per cycle, in control steps "
                              "(full-loop only). If per-cycle goal-error improvement is "
                              "decelerating sharply (check the per-cycle trace), try "
                              "increasing this before adding more --cycles.")
    args = parser.parse_args()

    obstacle_center = (args.obstacle_x, args.obstacle_y, args.obstacle_z)

    all_logs = {}
    all_metrics = {}

    if args.only in ("all", "baseline"):
        print("\n=== Experiment 1: Baseline MPPI (no GPMP2 prior, no CBF) ===")
        log, elapsed = experiment_1_baseline(args.mjcf, obstacle_center=obstacle_center,
                                               obstacle_radius=args.obstacle_radius)
        plot_trajectory_comparison({"baseline MPPI": log}, save="fig_exp1_trajectory.png")
        plot_rollout_vs_mean(log["V_snapshot"], log["weighted_mean_snapshot"],
                              sampling_mean=log["sampling_mean_snapshot"],
                              sampling_mean_label="previous control tape",
                              goal_value=Q_GOAL[0], joint_idx=0,
                              title="Baseline MPPI: rollouts vs. weighted mean",
                              save="fig_exp1_rollouts_vs_mean.png")
        plot_cost_convergence(log["cost_history"], save="fig_exp1_cost_convergence.png")
        all_logs["baseline"] = log
        all_metrics["baseline"] = _metrics_from_log(log, "u", elapsed)
        print(f"  final EE clearance to obstacle: {log['dist'][-1]:.4f} m, "
              f"min clearance during run: {min(log['dist']):.4f} m")

    if args.only in ("all", "gpmp2_mppi"):
        print("\n=== Experiment 2: GPMP2 + MPPI (GPMP2 trajectory as sampling mean) ===")
        log, theta_q, elapsed = experiment_2_gpmp2_mppi(args.mjcf, obstacle_center=obstacle_center,
                                                           obstacle_radius=args.obstacle_radius,
                                                           gpmp2_eps=args.gpmp2_eps)
        plot_trajectory_comparison({"GPMP2+MPPI": log}, save="fig_exp2_trajectory.png")
        plot_rollout_vs_mean(log["V_snapshot"], log["weighted_mean_snapshot"],
                              sampling_mean=log["sampling_mean_snapshot"],
                              sampling_mean_label="GPMP2 reference",
                              goal_value=Q_GOAL[0], joint_idx=0,
                              title="GPMP2+MPPI: rollouts vs. weighted mean vs. GPMP2 reference",
                              save="fig_exp2_rollouts_vs_mean.png")
        plot_cost_convergence(log["cost_history"], save="fig_exp2_cost_convergence.png")
        all_logs["gpmp2+mppi"] = log
        all_metrics["gpmp2+mppi"] = _metrics_from_log(log, "u", elapsed)
        print(f"  final EE clearance to obstacle: {log['dist'][-1]:.4f} m, "
              f"min clearance during run: {min(log['dist']):.4f} m")

    if args.only in ("all", "safe_mppi"):
        print("\n=== Experiments 3-4: MPPI before/after CBF-QP safety filter ===")
        log, feas_log, elapsed = experiment_3_safe_mppi(
            args.mjcf, obstacle_center=obstacle_center,
            obstacle_radius=args.obstacle_radius, d_safe=args.d_safe,
            gpmp2_eps=args.gpmp2_eps)
        plot_trajectory_comparison({"MPPI+CBF": log}, save="fig_exp3_4_trajectory.png")
        plot_cost_convergence(log["cost_history"], save="fig_exp3_4_cost_convergence.png")
        plot_barrier_and_intervention(feas_log, save="fig_exp3_4_feasibility.png")
        all_logs["mppi+cbf"] = log
        all_metrics["mppi+cbf"] = _metrics_from_log(log, "u_safe", elapsed)
        n_unsafe = sum(log["unsafe_flags"])
        print(f"  steps where MPPI's raw control was unsafe pre-QP: "
              f"{n_unsafe}/{len(log['unsafe_flags'])}")
        print(f"  min clearance during run: {min(log['dist']):.4f} m "
              f"(d_safe={args.d_safe}, obstacle_radius={args.obstacle_radius})")

    if args.only in ("all", "full"):
        print("\n=== Experiments 5-8: Full adaptive closed loop ===")
        history, feas_log, cov_steer, summary, elapsed = experiment_4_full_loop(
            args.mjcf, cycles=args.cycles, obstacle_center=obstacle_center,
            obstacle_radius=args.obstacle_radius, d_safe=args.d_safe,
            tau_conflict=args.tau_conflict, tau_safe=args.tau_safe,
            lambda_cbf=args.lambda_cbf, gpmp2_eps=args.gpmp2_eps,
            N_horizon=args.n_horizon)
        plot_trajectory_comparison({"full adaptive loop": history}, save="fig_exp5_8_trajectory.png")
        plot_cost_convergence(history["cost_history"], save="fig_exp5_8_cost_convergence.png")
        plot_barrier_and_intervention(feas_log, save="fig_exp5_feasibility_full.png")
        plot_covariance_evolution(cov_steer, save="fig_exp6_covariance.png")
        all_logs["full loop"] = history
        m = _metrics_from_log(history, "u_safe", elapsed)
        m["n_conflicts_inserted"] = summary["n_conflicts_inserted"]
        all_metrics["full loop"] = m
        print(f"  conflict factors inserted: {summary['n_conflicts_inserted']}")
        if history.get("best_goal_error") is not None:
            print(f"\n  *** BEST RESULT ACHIEVED THIS RUN: goal_err = "
                  f"{history['best_goal_error']:.4f} *** (this is the trajectory the "
                  f"checkpoint guard already found and kept as its fallback -- a "
                  f"legitimate result of this run even if the final cycle ended "
                  f"elsewhere due to later drift)")
        if history.get("replan_accepted"):
            n_replans = len(history["replan_accepted"])
            n_rejected = sum(1 for a in history["replan_accepted"] if not a)
            print(f"  replan cost-guard: {n_rejected}/{n_replans} GPMP2 replans were "
                  f"REJECTED (new solve scored worse than continuing the warm-start basis "
                  f"by GPMP2's own cost) and {n_replans - n_rejected}/{n_replans} were "
                  f"accepted.")
            if n_rejected > 0:
                rejected_cycles = [i + 1 for i, a in enumerate(history["replan_accepted"]) if not a]
                print(f"  rejected at cycle(s): {rejected_cycles}")
        if history.get("rolled_back_to_best"):
            n_rollbacks = sum(1 for r in history["rolled_back_to_best"] if r)
            print(f"  outcome checkpoint: {n_rollbacks}/{len(history['rolled_back_to_best'])} "
                  f"cycles rolled back to the best-known trajectory (real goal_err had drifted "
                  f"meaningfully worse than the best ever achieved).")
            if n_rollbacks > 0:
                rollback_cycles = [i + 1 for i, r in enumerate(history["rolled_back_to_best"]) if r]
                print(f"  rolled back at cycle(s): {rollback_cycles}")
        if history.get("cycle_end_goal_error"):
            print("\n  Per-cycle convergence trace (goal error, min clearance seen so far):")
            print(f"  {'cycle':>6}{'goal_err':>12}{'min_clear':>12}{'delta':>10}")
            prev = None
            for i, (ge, mc) in enumerate(zip(history["cycle_end_goal_error"],
                                               history["cycle_end_min_clear"])):
                delta = f"{ge - prev:+.4f}" if prev is not None else "  --"
                print(f"  {i + 1:>6}{ge:>12.4f}{mc:>12.4f}{delta:>10}")
                prev = ge
            n = len(history["cycle_end_goal_error"])
            if n >= 3:
                errs = history["cycle_end_goal_error"]
                deltas = [errs[i] - errs[i - 1] for i in range(1, n)]
                n_regressions = sum(1 for d in deltas if d > 0)
                best_so_far = min(errs)
                best_idx = errs.index(best_so_far)
                final = errs[-1]
                if n_regressions >= max(2, n // 3):
                    print(f"  --> OSCILLATING, not converging: {n_regressions}/{n - 1} "
                          f"cycle-to-cycle steps got WORSE, not just slower. Best goal_err "
                          f"seen was {best_so_far:.4f} at cycle {best_idx + 1}, but the run "
                          f"ended at {final:.4f} -- {'worse' if final > best_so_far else 'better'} "
                          f"than that best point.")
                else:
                    last_delta = deltas[-1]
                    prev_delta = deltas[-2]
                    if abs(prev_delta) > 1e-9 and abs(last_delta) < 0.3 * abs(prev_delta):
                        print("  --> improvement per cycle is shrinking fast (looks like it's "
                              "plateauing) -- more cycles alone are unlikely to close the gap.")
                    elif abs(last_delta) < 1e-3:
                        print("  --> goal_err barely changed in the last cycle -- likely stalled, "
                              "not just slow.")
                    else:
                        print("  --> still making meaningful progress each cycle -- more cycles "
                              "may help.")
        if summary["n_conflicts_inserted"] == 0 and feas_log.samples:
            h_trace = feas_log.barrier_trace()
            interv_trace = feas_log.intervention_trace()
            min_h = float(np.min(h_trace))
            max_interv = float(np.max(interv_trace))
            # Fraction of steps satisfying EACH half of the AND condition,
            # to show which one is the actual blocker for this run.
            frac_safe_half = float(np.mean(h_trace < args.tau_safe))
            frac_conflict_half = float(np.mean(interv_trace > args.tau_conflict))
            print(f"  no conflicts triggered -- diagnostic: min h(x) observed = {min_h:.4f} "
                  f"(tau_safe={args.tau_safe}, {frac_safe_half:.0%} of steps below it), "
                  f"max intervention observed = {max_interv:.4f} "
                  f"(tau_conflict={args.tau_conflict}, {frac_conflict_half:.0%} of steps above it). "
                  f"Both fractions need to be simultaneously true at the SAME step for a "
                  f"conflict to fire -- try lowering whichever fraction is near 0%.")

    if all_metrics:
        print_metrics_table(all_metrics)

    if args.only == "all" and len(all_logs) > 1:
        print("\n=== Combined comparison across all four experiments ===")
        plot_multi_cost_convergence(
            {label: log["cost_history"] for label, log in all_logs.items()},
            save="fig_all_cost_comparison.png")
        plot_trajectory_comparison(all_logs, save="fig_all_trajectory_comparison.png")
        plot_success_rate_and_timing(list(all_metrics.values()), list(all_metrics.keys()),
                                       save="fig_all_timing_summary.png")

    if args.seeds > 1:
        run_multi_seed_success_rate(args.mjcf, args.seeds, obstacle_center, args.obstacle_radius)

    print("\nAll requested experiments finished. Figures saved as fig_*.png "
          "in the current directory.")


if __name__ == "__main__":
    main()

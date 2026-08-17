"""
plots/main.py
===============
Publication-quality figure generation for every experiment listed in
the spec's "Results Required" / "Visualization" sections. Each
function takes the log dict / FeasibilityLog / CovarianceSteering
objects produced by experiments/*.py and saves a matplotlib figure.

Kept dependency-light (matplotlib only) so it runs anywhere numpy runs,
independent of gtsam/mujoco/osqp availability.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "savefig.bbox": "tight",
})


def plot_trajectory_comparison(logs: dict, joint_idx: int = 0, save="trajectory_comparison.png"):
    """logs: {label: log_dict} with log_dict['q'] a list of (dof,) arrays."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, log in logs.items():
        q = np.array(log["q"])
        ax.plot(q[:, joint_idx], label=label)
    ax.set_xlabel("control step"); ax.set_ylabel(f"joint {joint_idx} position (rad)")
    ax.set_title("Trajectory comparison"); ax.legend()
    fig.savefig(save); plt.close(fig)


def plot_rollout_vs_mean(V: np.ndarray, weighted_mean: np.ndarray,
                           sampling_mean: np.ndarray = None,
                           sampling_mean_label: str = "sampling mean",
                           goal_value: float = None, joint_idx: int = 0,
                           title: str = "", save: str = "rollout_vs_mean.png"):
    """
    THE figure that demonstrates why GPMP2-guided sampling helps: plots,
    over the full horizon (not just one timestep), every sampled MPPI
    rollout as a thin translucent line, the MPPI weighted-mean control
    tape as a bold solid line, the sampling-mean trajectory (GPMP2
    reference for the guided variant, or the previous control tape for
    baseline MPPI) as a bold dashed line, and the goal joint value as a
    horizontal reference line.

    V              : (N, T, n) sampled rollouts (MPPIResult.V)
    weighted_mean  : (T, n) MPPIResult.u_mppi -- the actual MPPI output
    sampling_mean  : (T, n) or None -- what V was centered on (theta_GPMP2
                     for the guided variant, prev_tape for baseline)
    goal_value     : scalar goal value for `joint_idx`, drawn as a
                     horizontal dashed reference line, or None to omit
    joint_idx      : which of the n dimensions to plot (default 0)
    """
    N, T, n = V.shape
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i in range(N):
        ax.plot(V[i, :, joint_idx], color="tab:blue", alpha=0.05, linewidth=0.8)
    # one labeled line so the legend shows the rollouts without N legend entries
    ax.plot([], [], color="tab:blue", alpha=0.4, linewidth=1.2, label="sampled rollouts")

    if sampling_mean is not None:
        ax.plot(sampling_mean[:, joint_idx], color="tab:green", linewidth=2.2,
                linestyle="--", label=sampling_mean_label)

    ax.plot(weighted_mean[:, joint_idx], color="tab:red", linewidth=2.4,
            label="MPPI weighted mean (u_mppi)")

    if goal_value is not None:
        ax.axhline(goal_value, color="k", linestyle=":", linewidth=1.5, label="goal")

    ax.set_xlabel("horizon step"); ax.set_ylabel(f"joint {joint_idx} value (rad)")
    ax.set_title(title or "Sampled rollouts vs. weighted mean vs. reference")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(save); plt.close(fig)


def plot_sample_distribution(V: np.ndarray, theta_mean: np.ndarray, save="sample_distribution.png"):
    """V: (N,T,n) rollouts; theta_mean: (T,n) mean trajectory. Plots the
    t=0 slice's first two dims as a scatter cloud around the mean."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(V[:, 0, 0], V[:, 0, 1], s=6, alpha=0.4, label="rollouts")
    ax.scatter([theta_mean[0, 0]], [theta_mean[0, 1]], c="red", marker="x",
               s=80, label="GPMP2 mean")
    ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")
    ax.set_title("MPPI sample distribution"); ax.legend()
    fig.savefig(save); plt.close(fig)


def plot_multi_cost_convergence(cost_histories: dict, save="cost_comparison_all.png"):
    """
    cost_histories: {label: list_of_mean_cost_per_step}. Overlays every
    experiment's cost curve on one axis so they are directly comparable
    (fixes the earlier inconsistency where only some experiments logged
    cost at all).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, hist in cost_histories.items():
        ax.plot(hist, label=label)
    ax.set_xlabel("control step"); ax.set_ylabel("mean rollout cost")
    ax.set_title("Cost comparison across experiments"); ax.legend()
    fig.savefig(save); plt.close(fig)


def plot_cost_convergence(cost_history, save="cost_convergence.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cost_history)
    ax.set_xlabel("control step"); ax.set_ylabel("mean rollout cost")
    ax.set_title("MPPI cost convergence")
    fig.savefig(save); plt.close(fig)


def plot_barrier_and_intervention(feas_log, save="feasibility.png"):
    h = feas_log.barrier_trace()
    interv = feas_log.intervention_trace()
    slack = feas_log.slack_trace()
    active = feas_log.active_constraint_counts()
    duals = feas_log.dual_norm_trace()

    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    axes[0, 0].plot(h); axes[0, 0].axhline(0, color="k", ls="--"); axes[0, 0].set_title("Barrier value h(x)")
    axes[0, 1].plot(interv); axes[0, 1].set_title("Intervention magnitude ||delta u||")
    axes[1, 0].plot(slack); axes[1, 0].set_title("Slack")
    axes[1, 1].plot(active); axes[1, 1].set_title("# active constraints")
    axes[2, 0].plot(duals); axes[2, 0].set_title("Dual variable norm")
    axes[2, 1].plot(h, label="h(x)"); axes[2, 1].plot(interv, label="||delta u||")
    axes[2, 1].set_title("Barrier vs. intervention"); axes[2, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("control step")
    fig.tight_layout(); fig.savefig(save); plt.close(fig)


def plot_covariance_evolution(cov_steer, save="covariance_evolution.png"):
    eig_hist = cov_steer.eigen_history()
    eig_arr = np.array(eig_hist)  # (steps, n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j in range(eig_arr.shape[1]):
        axes[0].plot(eig_arr[:, j], label=f"eig {j}")
    axes[0].set_title("Covariance eigenvalues over time")
    axes[0].set_xlabel("update index"); axes[0].legend(fontsize=6)

    traces = [np.trace(S) for S in cov_steer.trace_log]
    axes[1].plot(traces)
    axes[1].set_title("tr(Sigma_t) (overall sampling spread)")
    axes[1].set_xlabel("update index")
    fig.tight_layout(); fig.savefig(save); plt.close(fig)


def plot_conflict_map(events, theta_before: np.ndarray, theta_after: np.ndarray,
                        save="conflict_map.png"):
    """events: list[ConflictEvent]. theta_before/after: (T, dof) position
    trajectories pre/post conflict-factor insertion."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(theta_before[:, 0], theta_before[:, 1], label="before conflict factors")
    ax.plot(theta_after[:, 0], theta_after[:, 1], label="after conflict factors")
    for ev in events:
        ax.scatter([ev.x_danger[0]], [ev.x_danger[1]], c="red", marker="*", s=100)
    ax.set_title("Conflict map: trajectory before/after iSAM2 update")
    ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1"); ax.legend()
    fig.savefig(save); plt.close(fig)


def plot_success_rate_and_timing(summaries: list[dict], labels: list[str],
                                    save="summary_timing.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    times = [s["wall_clock_s"] for s in summaries]
    axes[0].bar(labels, times)
    axes[0].set_ylabel("wall-clock time (s)"); axes[0].set_title("Execution time")

    n_conflicts = [s["n_conflicts_inserted"] for s in summaries]
    axes[1].bar(labels, n_conflicts)
    axes[1].set_ylabel("# conflict factors inserted"); axes[1].set_title("Conflict interventions")
    fig.tight_layout(); fig.savefig(save); plt.close(fig)

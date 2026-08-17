"""
diagnose_convergence_loss.py
Compares GPMP2's PLANNED barrier value (h along its own proposed
trajectory) against the ACTUAL executed h(q) for cycles 14-20, to test
whether GPMP2 keeps proposing routes inside the gap between gpmp2_eps
and d_safe that the CBF then has to keep fighting.
"""
import numpy as np
from main import run_closed_loop, build_default_system
from cbf.barrier import DistanceBarrier

PLANNED = {}   # cycle -> planned h trace along theta_star_new
EXECUTED = {}  # cycle -> list of executed h(q) values

def on_stage(stage, info):
    if stage == "Update GPMP2":
        cycle = info["cycle"]
        PLANNED[cycle] = info  # theta_star_new is here; barrier eval done after run
    elif stage == "Robot Execution":
        cycle = info["cycle"]
        EXECUTED.setdefault(cycle, []).append(info)

history, feas_log, cov_steer = run_closed_loop(
    mjcf_path="assets/panda.xml", n_planning_cycles=20,
    obstacle_center=(0.20, 0.09, 0.85), obstacle_radius=0.08, d_safe=0.03,
    tau_conflict=0.05, tau_safe=0.15, lambda_cbf=0.3, gpmp2_eps=0.02,
    on_stage=on_stage)

# Rebuild the same barrier used internally to evaluate GPMP2's PLANNED h
_, franka, sdf, barrier, _ = build_default_system(
    "assets/panda.xml", obstacle_center=(0.20, 0.09, 0.85),
    obstacle_radius=0.08, d_safe=0.03)

print(f"{'cycle':>6}{'planned_min_h':>16}{'planned_frac_unsafe':>18}")
for cycle in sorted(PLANNED.keys()):
    if 12 <= cycle <= 22:
        theta_star = PLANNED[cycle]["theta_star_new"]
        h_vals = np.array([barrier.forward(row) for row in theta_star])
        frac_neg = float(np.mean(h_vals < 0))  # fraction of planned points inside the eps/d_safe gap
        print(f"{cycle:>6}{np.min(h_vals):>16.4f}{frac_neg:>18.1%}")

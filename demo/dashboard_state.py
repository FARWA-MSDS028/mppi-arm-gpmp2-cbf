"""
demo/dashboard_state.py
=========================
Thread-safe shared state for the live dashboard.
"""
from __future__ import annotations
import threading
import time
import numpy as np

PIPELINE_STAGES = [
    "GPMP2", "MPPI", "CBF-QP", "Robot Execution",
    "Feasibility Extraction", "Conflict Factor",
    "Covariance Steering", "Update GPMP2",
]

STAGE_COLORS = {
    "gpmp2": "#1f6feb",
    "mppi_sample": "#8b8b8b",
    "mppi_best": "#2ea043",
    "rejected": "#da3633",
    "robot_path": "#d4a72c",
    "obstacle": "#e8590c",
    "goal": "#2ea043",
    "home": "#f0f6fc",
    "cbf_correction": "#da3633",   # red — path segments where CBF-QP actively corrected
    "conflict_marker": "#ff2d55",  # pink — points where a conflict event was recorded
    "covariance": "#8250df",       # purple — uncertainty sphere from covariance steering
}


class DashboardState:
    def __init__(self):
        self.lock = threading.Lock()

        self.current_stage = None
        self.stage_entered_at = time.time()
        self.cycle = 0
        self.control_step = 0

        self.gpmp2_theta = None
        self.gpmp2_theta_old = None
        self.gpmp2_iterations = None
        self.gpmp2_final_error = None
        self.gpmp2_time_s = None
        self.gpmp2_accepted = None
        self.gpmp2_rolled_back = None

        self.mppi_rollouts = None
        self.mppi_best = None
        self.mppi_costs = None
        self.mppi_n_samples = None
        self.mppi_sampling_mean = None

        self.qp_unsafe = None
        self.qp_intervention = None
        self.qp_solve_status = None
        self.qp_n_unsafe_total = 0
        self.qp_n_corrected_total = 0
        self.qp_min_h_seen = None

        self.q = None
        self.qdot = None
        self.u_safe = None
        self.ee_position = None
        self.sim_time = None
        self.goal_error = None
        self.control_freq_hz = None
        self._last_exec_wall_time = None

        self.active_constraints = []
        self.safety_margin = None
        self.feasibility_bar = 0.0

        self.conflict_score = 0.0
        self.conflict_active = False
        self.n_conflicts_total = 0
        self.tau_conflict = None
        self.tau_safe = None

        self.sigma_t = None
        self.sigma_scale = None
        self.sigma_eigenvalues = None

        self.q_home = None
        self.q_goal = None
        self.obstacle_center = None
        self.obstacle_radius = None
        self.ee_home = None
        self.ee_goal = None

        self.ee_path = []
        self.conflict_markers = []  # list of ee positions where a conflict event fired

        self.paused = False
        self.show_mppi = True
        self.show_gpmp2 = True
        self.show_covariance = True
        self.show_feasibility = True
        self.show_stats = True
        self.reset_requested = False
        self.quit_requested = False

        # ---- Multi-rate thread status ----
        self.gpmp2_status = "IDLE"
        self.gpmp2_solve_time_s = None
        self.gpmp2_version = 0
        self.mppi_status = "IDLE"
        self.mppi_period_s = None
        self.mppi_compute_time_s = None
        self.mppi_reference_version = None
        self.robot_status = "IDLE"
        self.robot_period_s = None
        self.cbf_active = False

    def set_stage(self, stage_name: str):
        with self.lock:
            self.current_stage = stage_name
            self.stage_entered_at = time.time()

    def set_thread_status(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
    def append_ee_path(self, p, cbf_active: bool = False, max_len: int = 2000):
        with self.lock:
            self.ee_path.append((np.asarray(p).copy(), cbf_active))
            if len(self.ee_path) > max_len:
                self.ee_path.pop(0)

    def push_conflict_marker(self, p, max_len: int = 200):
        with self.lock:
            self.conflict_markers.append(np.asarray(p).copy())
            if len(self.conflict_markers) > max_len:
                self.conflict_markers.pop(0)
    def snapshot(self) -> dict:
        with self.lock:
            return {
                "current_stage": self.current_stage,
                "cycle": self.cycle,
                "control_step": self.control_step,
                "gpmp2_theta": None if self.gpmp2_theta is None else self.gpmp2_theta.copy(),
                "gpmp2_theta_old": None if self.gpmp2_theta_old is None else self.gpmp2_theta_old.copy(),
                "gpmp2_iterations": self.gpmp2_iterations,
                "gpmp2_final_error": self.gpmp2_final_error,
                "gpmp2_time_s": self.gpmp2_time_s,
                "gpmp2_accepted": self.gpmp2_accepted,
                "gpmp2_rolled_back": self.gpmp2_rolled_back,
                "mppi_rollouts": None if self.mppi_rollouts is None else self.mppi_rollouts.copy(),
                "mppi_best": None if self.mppi_best is None else self.mppi_best.copy(),
                "mppi_costs": None if self.mppi_costs is None else self.mppi_costs.copy(),
                "mppi_n_samples": self.mppi_n_samples,
                "qp_unsafe": self.qp_unsafe,
                "qp_intervention": self.qp_intervention,
                "qp_solve_status": self.qp_solve_status,
                "qp_n_unsafe_total": self.qp_n_unsafe_total,
                "qp_n_corrected_total": self.qp_n_corrected_total,
                "qp_min_h_seen": self.qp_min_h_seen,
                "q": None if self.q is None else self.q.copy(),
                "qdot": None if self.qdot is None else self.qdot.copy(),
                "ee_position": None if self.ee_position is None else self.ee_position.copy(),
                "sim_time": self.sim_time,
                "goal_error": self.goal_error,
                "control_freq_hz": self.control_freq_hz,
                "active_constraints": list(self.active_constraints),
                "safety_margin": self.safety_margin,
                "feasibility_bar": self.feasibility_bar,
                "conflict_score": self.conflict_score,
                "conflict_active": self.conflict_active,
                "n_conflicts_total": self.n_conflicts_total,
                "sigma_scale": self.sigma_scale,
                "sigma_eigenvalues": None if self.sigma_eigenvalues is None else self.sigma_eigenvalues.copy(),
                "q_home": self.q_home,
                "q_goal": self.q_goal,
                "obstacle_center": self.obstacle_center,
                "obstacle_radius": self.obstacle_radius,
                "ee_home": self.ee_home,
                "ee_goal": self.ee_goal,
                "ee_path": list(self.ee_path),
                "conflict_markers": list(self.conflict_markers),
                "paused": self.paused,
                "show_mppi": self.show_mppi,
                "show_gpmp2": self.show_gpmp2,
                "show_covariance": self.show_covariance,
                "show_feasibility": self.show_feasibility,
                "show_stats": self.show_stats,
                "gpmp2_status": self.gpmp2_status,
                "gpmp2_solve_time_s": self.gpmp2_solve_time_s,
                "gpmp2_version": self.gpmp2_version,
                "mppi_status": self.mppi_status,
                "mppi_period_s": self.mppi_period_s,
                "mppi_compute_time_s": self.mppi_compute_time_s,
                "mppi_reference_version": self.mppi_reference_version,
                "robot_status": self.robot_status,
                "robot_period_s": self.robot_period_s,
                "cbf_active": self.cbf_active,
            }

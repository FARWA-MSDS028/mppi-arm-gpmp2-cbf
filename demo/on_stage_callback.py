from __future__ import annotations
import time
import numpy as np

def make_on_stage_callback(state, franka=None):
    def on_stage(stage: str, info: dict):
        if stage == "GPMP2":
            with state.lock:
                state.gpmp2_theta = np.asarray(info["theta_star"])
                state.gpmp2_version += 1
                state.gpmp2_iterations = info.get("iterations")
                state.gpmp2_final_error = info.get("final_error")
                state.cycle = info.get("cycle", state.cycle)
            state.set_stage("GPMP2")

        elif stage == "MPPI":
            result = info["mppi_result"]
            with state.lock:
                state.mppi_rollouts = np.asarray(result.V)
                state.mppi_version += 1
                state.mppi_best = np.asarray(result.u_mppi)
                state.mppi_costs = np.asarray(result.costs)
                state.mppi_n_samples = result.V.shape[0]
                state.control_step = info.get("k", state.control_step)
                state.cycle = info.get("cycle", state.cycle)
            state.set_stage("MPPI")

        elif stage == "CBF-QP":
            qp_result = info["qp_result"]
            unsafe = info["unsafe"]
            with state.lock:
                state.qp_unsafe = bool(unsafe.is_unsafe)
                state.qp_intervention = float(qp_result.intervention_magnitude)
                state.qp_solve_status = qp_result.solve_status
                if state.qp_unsafe:
                    state.qp_n_unsafe_total += 1
                if qp_result.intervention_magnitude > 1e-6:
                    state.qp_n_corrected_total += 1
                state.active_constraints = list(qp_result.active_constraints)
                if state.qp_min_h_seen is None or unsafe.h_value < state.qp_min_h_seen:
                    state.qp_min_h_seen = float(unsafe.h_value)
            state.set_stage("CBF-QP")

        elif stage == "Robot Execution":
            q = np.asarray(info["q"])
            u_safe = np.asarray(info["u_safe"])
            t = info.get("t", 0)
            now = time.time()
            with state.lock:
                state.q = q
                state.ee_position = np.asarray(info.get("ee_position")) if info.get("ee_position") is not None else state.ee_position
                state.sim_time = t * 0.05
                if state.q_goal is not None:
                    state.goal_error = float(np.max(np.abs(q - state.q_goal)))
                if state._last_exec_wall_time is not None:
                    dt_wall = now - state._last_exec_wall_time
                    if dt_wall > 1e-6:
                        state.control_freq_hz = 1.0 / dt_wall
                state._last_exec_wall_time = now
                state.cycle = info.get("cycle", state.cycle)
                state.control_step = info.get("k", state.control_step)
            if state.ee_position is not None:
                state.append_ee_path(state.ee_position)
            elif franka is not None:
                centers, _ = franka.fk(q)
                state.append_ee_path(centers[-1])
            state.set_stage("Robot Execution")
            with state.lock:
                if state.single_step_pending:
                    state.single_step_pending = False
                    state.paused = True
            while True:
                with state.lock:
                    is_paused = state.paused
                    should_quit = state.quit_requested
                if not is_paused or should_quit:
                    break
                time.sleep(0.05)

        elif stage == "Feasibility Extraction":
            sample = info.get("sample")
            with state.lock:
                if sample is not None:
                    h_val = getattr(sample, "barrier_value", None)
                    if h_val is not None:
                        state.safety_margin = float(h_val)
                        if state.qp_min_h_seen is None or h_val < state.qp_min_h_seen:
                            state.qp_min_h_seen = float(h_val)
                        scale = state.tau_safe if state.tau_safe else 0.2
                        state.feasibility_bar = float(np.clip(1.0 - (h_val / scale), 0.0, 1.0))
            state.set_stage("Feasibility Extraction")

        elif stage == "Conflict Factor":
            event = info.get("event")
            with state.lock:
                state.tau_conflict = info.get("tau_conflict", state.tau_conflict)
                state.tau_safe = info.get("tau_safe", state.tau_safe)
                state.n_conflicts_total = info.get("n_conflicts_so_far", state.n_conflicts_total)
                state.conflict_active = event is not None
                intervention = info.get("intervention", 0.0)
                h0 = info.get("h0", None)
                tau_c = state.tau_conflict or 0.05
                tau_s = state.tau_safe or 0.2
                interv_score = np.clip(intervention / max(tau_c, 1e-6), 0.0, 1.0)
                h_score = 0.0 if h0 is None else np.clip(1.0 - (h0 / max(tau_s, 1e-6)), 0.0, 1.0)
                state.conflict_score = float(0.5 * interv_score + 0.5 * h_score)
            state.set_stage("Conflict Factor")

        elif stage == "Covariance Steering":
            with state.lock:
                eigvals = info.get("eigenvalues")
                if eigvals is not None:
                    state.sigma_scale = float(np.mean(eigvals))
                state.cycle = info.get("cycle", state.cycle)
            state.set_stage("Covariance Steering")

        elif stage == "Update GPMP2":
            with state.lock:
                state.gpmp2_theta = np.asarray(info.get("theta_star_new")) \
                    if info.get("theta_star_new") is not None else state.gpmp2_theta
                state.gpmp2_version += 1
                state.gpmp2_iterations = info.get("iterations", state.gpmp2_iterations)
                state.gpmp2_final_error = info.get("final_error", state.gpmp2_final_error)
                state.gpmp2_accepted = info.get("accepted", state.gpmp2_accepted)
                state.gpmp2_rolled_back = info.get("rolled_back", state.gpmp2_rolled_back)
                state.cycle = info.get("cycle", state.cycle)
            state.set_stage("Update GPMP2")

    return on_stage

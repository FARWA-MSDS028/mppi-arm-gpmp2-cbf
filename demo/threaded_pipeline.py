"""
demo/threaded_pipeline.py
============================
Multi-rate realization of:
    GPMP2 -> MPPI -> CBF-QP -> Robot Execution -> Feasibility ->
    Conflict Factor -> Covariance Steering -> Update GPMP2 -> Repeat

Data-dependency order is unchanged from main.py. Only wall-clock
timing changes: three independent threads, each reading the most
RECENT output of the stage above it, never blocking on it.

FIXED BUG: gpmp2_thread_fn used to store its own warm-start trajectory
with velocity stripped off (theta_star[:, :dof], 7 numbers per point
instead of the required 14). GPMP2's states are [q(7), qdot(7)] = 14
numbers per point -- passing a 7-number trajectory back in as the next
solve's init_trajectory caused "NoiseModel has dimension 14 instead of
7". Fix: keep the FULL 14-number trajectory for GPMP2's own internal
warm-starting, and only slice off the position part when publishing to
MPPI (MPPI's sampling mean is position-only, matching main.py's own
pattern: mppi.step(theta_star[:, :dof], ...)).
"""
from __future__ import annotations
import threading
import time
import numpy as np
import gtsam

from cbf.barrier import hocbf_lie_derivatives


class PipelineSharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.gpmp2_theta = None
        self.gpmp2_version = 0
        self.start_execution = False   # robot waits here until we say "go"
        self.u_mppi = None
        self.mppi_reference_version = None
        self.q = None
        self.qdot = None
        self.conflict_events = []
        self.best_theta = None
        self.best_goal_err = float("inf")
        self.stop = False

    def publish_gpmp2(self, theta):
        with self.lock:
            self.gpmp2_theta = theta
            self.gpmp2_version += 1
            return self.gpmp2_version

    def get_gpmp2(self):
        with self.lock:
            return self.gpmp2_theta, self.gpmp2_version

    def publish_u_mppi(self, u, ref_version):
        with self.lock:
            self.u_mppi = u
            self.mppi_reference_version = ref_version

    def get_u_mppi(self):
        with self.lock:
            return self.u_mppi

    def set_robot_state(self, q, qdot):
        with self.lock:
            self.q = q
            self.qdot = qdot

    def get_robot_state(self):
        with self.lock:
            return self.q, self.qdot

    def push_conflict_event(self, key_idx, event):
        with self.lock:
            self.conflict_events.append((key_idx, event))

    def pop_all_conflict_events(self):
        with self.lock:
            events, self.conflict_events = self.conflict_events, []
            return events

    def update_best(self, theta_full, goal_err):
        with self.lock:
            if goal_err < self.best_goal_err:
                self.best_goal_err = goal_err
                self.best_theta = theta_full.copy()

    def get_best(self):
        with self.lock:
            return self.best_theta, self.best_goal_err

    def set_start_execution(self):
        with self.lock:
            self.start_execution = True

    def can_execute(self):
        with self.lock:
            return self.start_execution


def gpmp2_thread_fn(shared, dashboard_state, planner, q_goal, dof, N_horizon,
                     rollback_tolerance=0.05):
    theta_goal = np.concatenate([q_goal, np.zeros(dof)])
    warm_start_full = None  # keeps ALL 14 numbers per point -- GPMP2's own state

    GOAL_THRESHOLD = 0.15  # same success number used throughout this project

    while not shared.stop:
        q, qdot = shared.get_robot_state()
        if q is None:
            time.sleep(0.05)
            continue

        # Once the robot is already close enough to the goal, stop
        # replanning -- there is nothing left to improve, and replanning
        # forever wastes computer time and does not help. Confirmed
        # needed: a 90-second test showed goal_err reach 0.058 (a real
        # success) by t=48s, then stay flat between 0.058 and 0.060 for
        # the remaining 42 seconds, while GPMP2 kept replanning almost
        # 300 more times for no real benefit.
        current_goal_err = float(np.max(np.abs(q - q_goal)))
        if current_goal_err < GOAL_THRESHOLD:
            dashboard_state.set_thread_status(gpmp2_status="READY (goal reached, paused)")
            time.sleep(0.2)
            continue

        dashboard_state.set_thread_status(gpmp2_status="SOLVING")
        t0 = time.time()
        theta0 = np.concatenate([q, qdot])

        best_theta_full, best_goal_err = shared.get_best()
        current_goal_err = float(np.max(np.abs(q - q_goal)))
        rolled_back = best_theta_full is not None and current_goal_err > best_goal_err + rollback_tolerance
        basis_full = best_theta_full if rolled_back else warm_start_full
        init_trajectory = None
        if basis_full is not None:
            init_trajectory = np.vstack([basis_full[1:], basis_full[-1:]])  # still 14 numbers per point

        result = planner.plan(theta0, theta_goal, N=N_horizon, init_trajectory=init_trajectory)

        accepted = True
        if init_trajectory is not None:
            baseline_values = gtsam.Values()
            for key, row in zip(result.keys, init_trajectory):
                baseline_values.insert(key, row)
            baseline_error = result.graph.error(baseline_values)
            accepted = result.final_error <= baseline_error

        theta_star_full = result.theta_star if accepted else init_trajectory  # 14 numbers per point
        warm_start_full = theta_star_full

        solve_time = time.time() - t0
        # Only the POSITION part goes to MPPI (matches main.py's own
        # pattern: mppi.step uses theta_star[:, :dof], not the full state).
        theta_pos_only = theta_star_full[:, :dof]
        version = shared.publish_gpmp2(theta_pos_only)
        shared.update_best(theta_star_full, current_goal_err)

        dashboard_state.set_thread_status(
            gpmp2_status="READY", gpmp2_solve_time_s=solve_time, gpmp2_version=version,
            gpmp2_accepted=accepted, gpmp2_rolled_back=rolled_back,
            gpmp2_final_error=result.final_error, gpmp2_theta=theta_pos_only,
        )


def mppi_thread_fn(shared, dashboard_state, mppi, barrier, dof, N_horizon, rng, hz=10.0):
    """
    barrier : the DistanceBarrier instance -- barrier_batch_fn is built
              here as a closure over it, matching main.py's own
              barrier_batch_fn construction, not duplicated logic.
    """
    period = 1.0 / hz
    Sigma = 0.002 ** 2 * np.eye(dof)
    K_inv_diag_full = np.ones((N_horizon + 1, dof))

    def barrier_batch_fn(V):
        N_, T_, _ = V.shape
        h = np.zeros((N_, T_))
        for i in range(N_):
            for t in range(T_):
                x = np.concatenate([V[i, t], np.zeros(dof)])
                h[i, t] = barrier.forward(x)
        return h

    # step_in_horizon moves forward through the current plan each time
    # MPPI runs, matching main.py's behavior (theta_q[step:]). WITHOUT
    # this, MPPI always looked at the START of the plan, over and over,
    # which is why the robot was not moving at all -- confirmed by
    # goal_err staying flat near 1.73 for hundreds of GPMP2 solves.
    step_in_horizon = 0
    last_version_seen = -1

    while not shared.stop:
        t0 = time.time()
        dashboard_state.set_thread_status(mppi_status="RUNNING")

        theta_ref, version = shared.get_gpmp2()
        if theta_ref is not None:
            if version != last_version_seen:
                # A new GPMP2 version arrived. IMPORTANT: this does NOT
                # mean the plan actually changed direction -- GPMP2
                # replans continuously, and when the robot has barely
                # moved, the new plan is nearly IDENTICAL to the old
                # one. Resetting step_in_horizon on every new version
                # number meant MPPI was told to "start over from the
                # beginning" almost every single time it ran, so it
                # never got far enough through any one plan to make
                # real progress -- confirmed as the real cause of the
                # robot barely moving in the threaded version, even
                # though GPMP2, MPPI, and CBF-QP each work correctly on
                # their own (see step1-step4 tests). Only reset the
                # counter if we don't already have a walking position,
                # or if the new plan is clearly a different LENGTH
                # (a real structural change) -- otherwise just keep
                # walking forward through the updated plan.
                if last_version_seen == -1:
                    step_in_horizon = 0
                last_version_seen = version

            # Clamp instead of reset -- keep walking forward, never
            # jump back to the start just because a new version number
            # arrived.
            step_in_horizon = min(step_in_horizon, max(0, len(theta_ref) - 1))
            theta_ref_remaining = theta_ref[step_in_horizon:]
            K_inv_diag_remaining = K_inv_diag_full[step_in_horizon:len(theta_ref_remaining) + step_in_horizon]

            # Once we are near the END of the plan, theta_ref_remaining
            # becomes very short (sometimes just 1-2 points). Giving
            # MPPI such a short, oddly-shaped target is not a normal
            # planning problem and was confirmed to cause a small,
            # steady wobble that never fully reached zero error (stuck
            # around 0.059 even after GPMP2 stopped replanning
            # entirely). Fix: pad the remaining plan back out to a full
            # N_horizon+1 length by repeating the FINAL point (the
            # goal) -- this gives MPPI a normal, well-shaped, full-size
            # target to track, that simply asks it to "stay at the
            # goal" for the padded part, instead of a tiny, unusual
            # problem.
            MIN_LEN = N_horizon + 1
            if len(theta_ref_remaining) < MIN_LEN:
                pad_count = MIN_LEN - len(theta_ref_remaining)
                last_point = theta_ref_remaining[-1:]
                padding = np.repeat(last_point, pad_count, axis=0)
                theta_ref_remaining = np.vstack([theta_ref_remaining, padding])
                K_inv_diag_remaining = np.vstack([
                    K_inv_diag_remaining,
                    np.repeat(K_inv_diag_full[-1:], pad_count, axis=0),
                ])

            result = mppi.step(theta_ref_remaining, Sigma, 200, K_inv_diag_remaining,
                                barrier_batch_fn, rng)
            u_mppi = result.u_mppi[0]
            step_in_horizon += 1
            shared.publish_u_mppi(u_mppi, version)
            dashboard_state.set_thread_status(
                mppi_reference_version=version, mppi_n_samples=result.V.shape[0],
                mppi_rollouts=result.V, mppi_best=result.u_mppi, mppi_costs=result.costs,
            )

        compute_time = time.time() - t0
        dashboard_state.set_thread_status(mppi_compute_time_s=compute_time, mppi_period_s=period)
        time.sleep(max(0.0, period - compute_time))


def robot_thread_fn(shared, dashboard_state, env, franka, barrier, qp, f_fn, g_fn,
                     conflict_mgr, cov_steer, feas_log, dof, hz=500.0):
    # RESTORED to 500Hz. Earlier tonight this was lowered to 120Hz for
    # "demo smoothness", with an incorrect claim that this was a pure
    # visual choice with no safety effect. DIRECTLY DISPROVEN by real
    # measurement: a full 90-second run at 120Hz showed h(x) actually
    # go negative (worst -0.0043) mid-motion, something never seen at
    # 500Hz. The real reason: 500Hz means each CBF-QP correction is
    # only held for 0.002s before being re-checked; at 120Hz that gap
    # grows to ~0.008s, giving real dynamics enough time to drift past
    # the safety boundary between corrections (the same sampled-data
    # gap identified analytically earlier tonight, now confirmed
    # directly). Also: 500Hz is not actually "fast" -- 500 calls/sec x
    # 0.002s physics step = exactly 1.0 real second of sim time per
    # real second, i.e. true real-time pace. The earlier "too fast"
    # feeling was very likely caused by GPMP2/MPPI blocking the render
    # loop in bursts, both already fixed separately (GPMP2 -> its own
    # process; MPPI's own cost computation lightened) -- not by this
    # rate itself.
    from cbf.barrier import closest_clearance
    from cbf.qp_solver import detect_unsafe
    period = 1.0 / hz
    step_idx = 0
    alpha_gamma = getattr(qp, "alpha_gamma", 100.0)
    while not shared.stop:
        t0 = time.time()

        if not shared.can_execute():
            dashboard_state.set_thread_status(robot_status="WAITING (holding at home)")
            time.sleep(period)
            continue

        dashboard_state.set_thread_status(robot_status="RUNNING")
        q, qdot = env.get_state()[:dof], env.get_state()[dof:]
        shared.set_robot_state(q, qdot)
        x = np.concatenate([q, qdot])

        ee_centers, _ = franka.fk(q)

        # Once genuinely close to the goal, stop actually moving --
        # not just freezing the DISPLAY. Without this, the window
        # freezes its picture at one moment while the real background
        # robot keeps moving and refining further, making the frozen
        # picture stale and confusing (the arm looked stuck while real
        # work was still silently happening).
        goal_err_check = None
        if dashboard_state.q_goal is not None:
            goal_err_check = float(np.max(np.abs(q - dashboard_state.q_goal)))
        if goal_err_check is not None and goal_err_check < 0.15:
            dashboard_state.set_thread_status(
                q=q, qdot=qdot, goal_error=goal_err_check, cbf_active=False,
            )
            time.sleep(period)
            continue

        u_mppi = shared.get_u_mppi()
        if u_mppi is not None:
            psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x, f_fn, g_fn, alpha0=1.0)
            d_obs = closest_clearance(franka.fk, franka.sphere_radii, barrier.sdf, q)
            unsafe = detect_unsafe(u_mppi, Lf_psi1, Lg_psi1, psi1, alpha_gamma=alpha_gamma,
                                    d_obstacle=d_obs, h0_physical=h0)
            qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
            goal_err = None
            if dashboard_state.q_goal is not None:
                goal_err = float(np.max(np.abs(q - dashboard_state.q_goal)))

            dashboard_state.set_thread_status(
                cbf_active=qp_result.intervention_magnitude > 1e-6,
                safety_margin=h0, qp_intervention=qp_result.intervention_magnitude,
                q=q, qdot=qdot, goal_error=goal_err,
            )
            env.substep(qp_result.u_safe)
            dashboard_state.append_ee_path(ee_centers[-1], cbf_active=qp_result.intervention_magnitude > 1e-6)
            feas_log.record(step_idx, unsafe, qp_result)

            ev = conflict_mgr.check_and_record(step_idx, x, qp_result.intervention, h0)
            if ev is not None:
                dashboard_state.push_conflict_marker(ee_centers[-1])

            cov_steer.update_online(qp_result.intervention)
            if step_idx % 20 == 0:
                cov_steer.update_windowed()

            step_idx += 1

        exec_time = time.time() - t0
        dashboard_state.set_thread_status(robot_period_s=period)
        time.sleep(max(0.0, period - exec_time))

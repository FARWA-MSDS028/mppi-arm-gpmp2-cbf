"""
demo/gpmp2_process.py
========================
Runs GPMP2 in its OWN SEPARATE PROCESS (not a thread). A process has
its own separate Python, with its own separate processing lock -- so
it can NEVER block the main window's ability to respond, no matter how
long a solve takes. This directly fixes the "MuJoCo is not responding"
freezes and the jumpy/skipping robot motion seen in the threaded
version, both caused by GPMP2 (a C++ library) holding Python's shared
lock for its entire solve time.

Two parts:
  - gpmp2_process_fn   : runs INSIDE the separate process.
  - gpmp2_bridge_thread_fn : a small, lightweight thread in the MAIN
    process that feeds robot-state into the separate process and
    publishes results back into the shared dashboard state -- this
    thread does almost no work itself, so it cannot cause blocking.

Logic matches demo/threaded_pipeline.py's gpmp2_thread_fn exactly
(same warm-start, same accept-if-better guard, same pause-once-goal-
reached behavior) -- only the process/thread boundary changed.
"""
from __future__ import annotations
import time
import numpy as np


def gpmp2_process_fn(q_in_queue, theta_out_queue, mjcf_path, obstacle_center,
                      obstacle_radius, d_safe, gpmp2_eps, q_goal, dof, N_horizon,
                      rollback_tolerance=0.05):
    """Runs entirely inside the separate process."""
    import mujoco
    import gtsam
    from robot.franka import FrankaModel
    from planner.factor_graph import SignedDistanceField
    from planner.gpmp2_planner import GPMP2Planner

    GOAL_THRESHOLD = 0.15

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    franka = FrankaModel(model, mujoco.MjData(model))
    sdf = SignedDistanceField(np.array([obstacle_center]), np.array([obstacle_radius]))
    Qc = 0.5 * np.eye(dof)
    planner = GPMP2Planner(dof=dof, dt=0.05, Qc=Qc, sdf=sdf, fk_fn=franka.fk,
                            sphere_offsets=franka.sphere_radii, eps=gpmp2_eps,
                            sigma_obs=0.02)

    theta_goal = np.concatenate([q_goal, np.zeros(dof)])
    warm_start_full = None
    best_theta_full = None
    best_goal_err = float("inf")

    while True:
        item = q_in_queue.get()  # blocks here until the main process sends something
        if item is None:  # shutdown signal
            break
        q, qdot = item

        current_goal_err = float(np.max(np.abs(q - q_goal)))
        if current_goal_err < GOAL_THRESHOLD:
            theta_out_queue.put({"paused": True})
            continue

        theta0 = np.concatenate([q, qdot])
        rolled_back = best_theta_full is not None and current_goal_err > best_goal_err + rollback_tolerance
        basis_full = best_theta_full if rolled_back else warm_start_full
        init_trajectory = np.vstack([basis_full[1:], basis_full[-1:]]) if basis_full is not None else None

        t0 = time.time()
        result = planner.plan(theta0, theta_goal, N=N_horizon, init_trajectory=init_trajectory)
        solve_time = time.time() - t0

        accepted = True
        if init_trajectory is not None:
            baseline_values = gtsam.Values()
            for key, row in zip(result.keys, init_trajectory):
                baseline_values.insert(key, row)
            baseline_error = result.graph.error(baseline_values)
            accepted = result.final_error <= baseline_error

        theta_star_full = result.theta_star if accepted else init_trajectory
        warm_start_full = theta_star_full
        if current_goal_err < best_goal_err:
            best_goal_err = current_goal_err
            best_theta_full = theta_star_full.copy()

        theta_out_queue.put({
            "paused": False,
            "theta": theta_star_full[:, :dof],
            "solve_time": solve_time,
            "accepted": accepted,
            "rolled_back": rolled_back,
            "final_error": float(result.final_error),
        })


def gpmp2_bridge_thread_fn(shared, dashboard_state, q_in_queue, theta_out_queue, dof):
    """
    Lives in the MAIN process, as a lightweight thread. Feeds the
    current robot state to the separate GPMP2 process and publishes
    results back -- does almost no work itself, so cannot cause the
    freezing that running GPMP2 directly did.
    """
    version = 0
    while not shared.stop:
        q, qdot = shared.get_robot_state()
        if q is None:
            time.sleep(0.05)
            continue

        dashboard_state.set_thread_status(gpmp2_status="SOLVING")
        q_in_queue.put((q, qdot))
        result = theta_out_queue.get()  # blocks until the process finishes this solve

        if shared.stop:
            break

        if result.get("paused"):
            dashboard_state.set_thread_status(gpmp2_status="READY (goal reached, paused)")
            time.sleep(0.2)
            continue

        version += 1
        shared.publish_gpmp2(result["theta"])
        dashboard_state.set_thread_status(
            gpmp2_status="READY", gpmp2_solve_time_s=result["solve_time"],
            gpmp2_version=version, gpmp2_accepted=result["accepted"],
            gpmp2_rolled_back=result["rolled_back"],
            gpmp2_final_error=result["final_error"], gpmp2_theta=result["theta"],
        )

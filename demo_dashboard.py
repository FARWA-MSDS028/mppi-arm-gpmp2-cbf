"""
demo_dashboard.py
====================
Single-window live demonstration of:
    GPMP2 -> MPPI -> CBF-QP -> Robot Execution -> Feasibility ->
    Conflict Factor -> Covariance Steering -> Update GPMP2 -> Repeat

Three threads, each at its own speed:
    - GPMP2 thread    : slow (fraction of a second to a few seconds)
    - MPPI thread     : medium (about 10 times per second)
    - Robot/CBF thread: fast (about 500 times per second)

Only ONE window opens: the MuJoCo 3D view. Live status and metrics
are printed to the terminal instead of a second window.

Run:
    python demo_dashboard.py --mjcf assets/panda.xml --cycles 20 \
        --obstacle-x 0.20 --obstacle-y 0.09 --obstacle-z 0.85 \
        --obstacle-radius 0.08 --d-safe 0.03 --gpmp2-eps 0.02 \
        --tau-conflict 0.05 --tau-safe 0.15 --lambda-cbf 0.3
"""
from __future__ import annotations
import argparse
import threading
import multiprocessing as mp
import time
import numpy as np
import mujoco
import mujoco.viewer

from demo.dashboard_state import DashboardState, STAGE_COLORS
from demo.gpmp2_process import gpmp2_process_fn, gpmp2_bridge_thread_fn
from demo.threaded_pipeline import (
    PipelineSharedState, gpmp2_thread_fn, mppi_thread_fn, robot_thread_fn,
)


def _hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return np.array([r, g, b, alpha], dtype=np.float32)


def _add_sphere(scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                         np.asarray(pos, dtype=np.float64), np.eye(3).flatten(), rgba)
    scn.ngeom += 1


def _add_capsule_segment(scn, p_from, p_to, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    p_from = np.asarray(p_from, dtype=np.float64)
    p_to = np.asarray(p_to, dtype=np.float64)
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                         np.zeros(3), np.eye(3).flatten(), rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, p_from, p_to)
    scn.ngeom += 1


def _draw_joint_trajectory(scn, franka, theta_dof, rgba, radius=0.004, stride=1):
    if theta_dof is None or len(theta_dof) < 2:
        return
    pts = []
    for i in range(0, len(theta_dof), stride):
        centers, _ = franka.fk(theta_dof[i])
        pts.append(centers[-1])
    for a, b in zip(pts[:-1], pts[1:]):
        _add_capsule_segment(scn, a, b, radius, rgba)


def render_scene(viewer, snap, franka):
    scn = viewer.user_scn
    scn.ngeom = 0

    if snap["ee_home"] is not None:
        _add_sphere(scn, snap["ee_home"], 0.04, _hex_to_rgba(STAGE_COLORS["home"]))
    if snap["ee_goal"] is not None:
        _add_sphere(scn, snap["ee_goal"], 0.05, _hex_to_rgba(STAGE_COLORS["goal"]))
    if snap["obstacle_center"] is not None:
        _add_sphere(scn, snap["obstacle_center"],
                    max(snap["obstacle_radius"] or 0.08, 0.08),
                    _hex_to_rgba(STAGE_COLORS["obstacle"], alpha=0.85))

    if snap["show_gpmp2"] and snap["gpmp2_theta"] is not None:
        _draw_joint_trajectory(scn, franka, snap["gpmp2_theta"][:, :franka.dof],
                                _hex_to_rgba(STAGE_COLORS["gpmp2"]), radius=0.005, stride=3)

    if snap["show_mppi"] and snap["mppi_rollouts"] is not None:
        rollouts = snap["mppi_rollouts"]
        costs = snap["mppi_costs"]
        # Reduced from 20 rollouts (stride 2) to 5 rollouts (stride 4) --
        # the old version computed forward kinematics for 600+ points
        # EVERY FRAME on the same thread that keeps the window
        # responsive, which was the real cause of MuJoCo freezing when
        # you tried to zoom or drag.
        n_show = min(5, rollouts.shape[0])
        idx = np.argsort(costs)[:n_show] if costs is not None else range(n_show)
        for i in idx:
            _draw_joint_trajectory(scn, franka, rollouts[i][:, :franka.dof],
                                    _hex_to_rgba(STAGE_COLORS["mppi_sample"], alpha=0.25),
                                    radius=0.002, stride=4)
        if snap["mppi_best"] is not None:
            _draw_joint_trajectory(scn, franka, snap["mppi_best"][:, :franka.dof],
                                    _hex_to_rgba(STAGE_COLORS["mppi_best"]), radius=0.004, stride=3)
    path = snap["ee_path"]
    if len(path) >= 2:
        stride = max(1, len(path) // 200)
        pts = path[::stride]
        for (a, a_cbf), (b, b_cbf) in zip(pts[:-1], pts[1:]):
            color = STAGE_COLORS["cbf_correction"] if (a_cbf or b_cbf) else STAGE_COLORS["robot_path"]
            _add_capsule_segment(scn, a, b, 0.003, _hex_to_rgba(color))

    for marker in snap["conflict_markers"]:
        _add_sphere(scn, marker, 0.012, _hex_to_rgba(STAGE_COLORS["conflict_marker"], alpha=0.9))

    if snap["show_covariance"] and snap.get("ee_position") is not None and snap.get("sigma_scale"):
        radius = 0.05 + 0.3 * min(float(snap["sigma_scale"]), 1.0)  # visual scale, tune as needed
        _add_sphere(scn, snap["ee_position"], radius, _hex_to_rgba(STAGE_COLORS["covariance"], alpha=0.15))

def print_status_line(snap):
    gpmp2_flag = snap["gpmp2_status"] if snap["gpmp2_status"] else "IDLE   "
    gpmp2_flag = f"{gpmp2_flag:<7}"
    mppi_flag = "RUN" if snap["mppi_status"] == "RUNNING" else "IDL"
    robot_flag = "RUN" if snap["robot_status"] == "RUNNING" else "IDL"
    cbf_flag = "ACTIVE  " if snap["cbf_active"] else "inactive"

    goal_err = f"{snap['goal_error']:.4f}" if snap["goal_error"] is not None else "----"
    h_val = f"{snap['safety_margin']:.4f}" if snap["safety_margin"] is not None else "----"
    gpmp2_v = snap["gpmp2_version"]
    conflict = f"{snap['conflict_score']:.2f}"

    line = (f"\r[GPMP2 {gpmp2_flag} v{gpmp2_v:>4}] "
            f"[MPPI {mppi_flag}] [Robot {robot_flag}] [CBF {cbf_flag}] "
            f"goal_err={goal_err} h(x)={h_val} conflict={conflict}   ")
    print(line, end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=str, default="assets/panda.xml")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--obstacle-x", type=float, default=0.20)
    parser.add_argument("--obstacle-y", type=float, default=0.09)
    parser.add_argument("--obstacle-z", type=float, default=0.85)
    parser.add_argument("--obstacle-radius", type=float, default=0.08)
    parser.add_argument("--d-safe", type=float, default=0.03)
    parser.add_argument("--gpmp2-eps", type=float, default=0.02)
    parser.add_argument("--tau-conflict", type=float, default=0.05)
    parser.add_argument("--tau-safe", type=float, default=0.15)
    parser.add_argument("--lambda-cbf", type=float, default=0.3)
    parser.add_argument("--alpha-gamma", type=float, default=100.0)
    parser.add_argument("--n-horizon", type=int, default=30)
    args = parser.parse_args()

    from main import build_default_system
    from robot.franka import DOF, FrankaModel
    from cbf.barrier import franka_dynamics, DistanceBarrier
    from cbf.qp_solver import CBFQPSolver
    from planner.gpmp2_planner import GPMP2Planner
    from planner.conflict_factor import ConflictFactorManager
    from controller.covariance import CovarianceSteering
    from controller.mppi import MPPIController
    from cbf.feasibility import FeasibilityLog

    obstacle_center = (args.obstacle_x, args.obstacle_y, args.obstacle_z)
    q0 = np.zeros(DOF)
    q_goal = np.array([0.4, -0.2, 0.3, -1.9, 0.1, 1.6, 0.5])

    env, franka_robot, sdf, barrier_robot, _ = build_default_system(
        args.mjcf, obstacle_center=obstacle_center,
        obstacle_radius=args.obstacle_radius, d_safe=args.d_safe)
    env.reset(q0)

    franka_gpmp2 = FrankaModel(env.model, mujoco.MjData(env.model))
    franka_mppi = FrankaModel(env.model, mujoco.MjData(env.model))
    barrier_mppi = DistanceBarrier(fk_fn=franka_mppi.fk, sphere_radii=franka_mppi.sphere_radii,
                                    sdf=sdf, dof=DOF, d_safe=args.d_safe)
    # A FOURTH private FrankaModel, used ONLY by the main thread for
    # drawing trajectory lines. Reusing franka_robot here caused a real
    # crash: the main thread's rendering calls (franka.fk inside
    # render_scene) were writing into the same MjData that the robot
    # thread's gravity()/mass_matrix()/coriolis() calls were also
    # writing into, at the same time -- MuJoCo detected the corrupted,
    # inconsistent state and crashed with a constraint-allocation error.
    franka_render = FrankaModel(env.model, mujoco.MjData(env.model))

    JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    qp = CBFQPSolver(m=DOF, u_min=JOINT_LOWER, u_max=JOINT_UPPER, alpha_gamma=args.alpha_gamma)

    def gravity_fn(q): return franka_robot.gravity(q)
    def coriolis_fn(q, qdot): return franka_robot.coriolis_times_qdot(q, qdot)
    def M_fn(q): return franka_robot.mass_matrix(q)
    def f_fn(xi):
        fi, _ = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
        return fi
    def g_fn(xi):
        _, gi = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
        return gi

    Qc = 0.5 * np.eye(DOF)
    planner = GPMP2Planner(dof=DOF, dt=0.05, Qc=Qc, sdf=sdf, fk_fn=franka_gpmp2.fk,
                            sphere_offsets=franka_gpmp2.sphere_radii, eps=args.gpmp2_eps,
                            sigma_obs=0.02)
    mppi = MPPIController(lam=1.0, dt=0.05, dof=DOF, sdf=sdf, eps_margin=0.15,
                           sigma_obs=0.02, lambda_cbf=args.lambda_cbf,
                           fk_batch_fn=franka_mppi.fk_batch, sphere_radii=franka_mppi.sphere_radii)
    conflict_mgr = ConflictFactorManager(tau_conflict=args.tau_conflict, tau_safe=args.tau_safe)
    Sigma0 = 0.002 ** 2 * np.eye(DOF)
    cov_steer = CovarianceSteering(n=DOF, Sigma0=Sigma0, eta=0.05, beta=0.3, W=20)
    feas_log = FeasibilityLog()

    dashboard_state = DashboardState()
    shared = PipelineSharedState()

    ee_home, _ = franka_robot.fk(q0)
    ee_goal, _ = franka_robot.fk(q_goal)
    dashboard_state.set_thread_status(
        q_home=q0, q_goal=q_goal, obstacle_center=np.array(obstacle_center),
        obstacle_radius=args.obstacle_radius, ee_home=ee_home[-1], ee_goal=ee_goal[-1],
    )
    shared.set_robot_state(q0, np.zeros(DOF))
    rng = np.random.default_rng(0)

    # GPMP2 now runs in its OWN SEPARATE PROCESS (see demo/gpmp2_process.py)
    # instead of a thread -- this was necessary to fix real, confirmed
    # freezing ("MuJoCo is not responding") and jumpy/skipping robot
    # motion, both caused by GPMP2 (a C++ library) blocking Python's
    # shared processing lock for its entire solve time. A process has
    # its own separate lock, so it can never block the window.
    q_in_queue = mp.Queue()
    theta_out_queue = mp.Queue()
    gpmp2_proc = mp.Process(
        target=gpmp2_process_fn,
        args=(q_in_queue, theta_out_queue, args.mjcf, obstacle_center,
              args.obstacle_radius, args.d_safe, args.gpmp2_eps, q_goal, DOF, args.n_horizon),
        daemon=True,
    )
    gpmp2_proc.start()

    t1 = threading.Thread(target=gpmp2_bridge_thread_fn,
                           args=(shared, dashboard_state, q_in_queue, theta_out_queue, DOF),
                           daemon=True)
    t2 = threading.Thread(target=mppi_thread_fn,
                           args=(shared, dashboard_state, mppi, barrier_mppi, DOF, args.n_horizon, rng),
                           daemon=True)
    t3 = threading.Thread(target=robot_thread_fn,
                           args=(shared, dashboard_state, env, franka_robot, barrier_robot, qp, f_fn, g_fn,
                                 conflict_mgr, cov_steer, feas_log, DOF),
                           daemon=True)
    t1.start(); t2.start(); t3.start()

    def key_callback(keycode):
        chr_key = chr(keycode) if 0 <= keycode < 256 else ""
        if chr_key == " ":
            dashboard_state.paused = not dashboard_state.paused
        elif chr_key.lower() == "p":
            dashboard_state.show_mppi = not dashboard_state.show_mppi
        elif chr_key.lower() == "g":
            dashboard_state.show_gpmp2 = not dashboard_state.show_gpmp2

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    GOAL_THRESHOLD = 0.15  # same success threshold used throughout the project
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Set a good, wide starting camera ONCE, automatically, so no
        # mouse zooming or dragging is ever needed. This matters because
        # GPMP2's solver blocks Python's shared processing lock during
        # each solve -- if you try to zoom/drag AT THAT EXACT MOMENT,
        # the click gets stuck waiting and Ubuntu shows "not responding"
        # (the program is not actually broken, just busy). Avoiding the
        # need to touch the mouse at all avoids this entirely.
        viewer.cam.lookat[:] = [0.3, 0.0, 0.6]   # roughly the middle of the workspace
        viewer.cam.distance = 1.8                 # zoomed out enough to see everything
        viewer.cam.azimuth = 135                  # angled view, not straight-on
        viewer.cam.elevation = -20                # slightly looking down

        last_print_time = 0.0
        reached = False
        announced = False
        try:
            while viewer.is_running():
                snap = dashboard_state.snapshot()

                if not shared.can_execute():
                    if snap["gpmp2_theta"] is not None and not hasattr(main, "_revealed"):
                        main._revealed = time.time()
                        print("\nSTAGE: GPMP2 path planned. Holding at HOME...")
                    if hasattr(main, "_revealed") and time.time() - main._revealed > 2.5:
                        shared.set_start_execution()
                        print("STAGE: MPPI + CBF-QP now executing.\n")

                # Once reached, freeze the robot exactly where it is --
                # stop reading new commands from the threads, but keep
                # rendering the window and keep it fully responsive.
                if not reached and not snap["paused"] and snap["q"] is not None:
                    data.qpos[:len(snap["q"])] = snap["q"]
                    mujoco.mj_forward(model, data)
                render_scene(viewer, snap, franka_render)
                viewer.sync()

                now = time.time()
                if now - last_print_time > 0.2:
                    print_status_line(snap)
                    last_print_time = now

                if not reached and snap["goal_error"] is not None and snap["goal_error"] < GOAL_THRESHOLD:
                    reached = True

                if reached and not announced:
                    print(f"\n\nTARGET REACHED -- goal_err = {snap['goal_error']:.4f}")
                    print("Robot frozen at goal. Window stays open -- "
                          "close it or press Ctrl+C in the terminal when you are done.")
                    announced = True

                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            print("\n\nCtrl+C received -- shutting down.")

    # Clean shutdown, in order, ONLY when the window is actually closed
    # or Ctrl+C is pressed -- not automatically the instant the target
    # is reached. This ordering (threads first, THEN the separate
    # process, THEN print) also fixes the segmentation fault that used
    # to happen when everything was torn down abruptly at once.
    print("Stopping threads...")
    shared.stop = True
    time.sleep(0.3)  # give threads a moment to notice shared.stop and exit their loops
    q_in_queue.put(None)
    gpmp2_proc.join(timeout=2.0)
    if gpmp2_proc.is_alive():
        gpmp2_proc.terminate()
    print("Stopped cleanly.")


if __name__ == "__main__":
    main()

"""
test_alpha_gamma_real_pipeline.py
Runs the REAL, complete threaded pipeline (live GPMP2 replanning, real
MPPI, real CBF-QP, at 500Hz) for each alpha_gamma value, tracking
worst h(x) seen -- this time using the actual full system, the same
one proven trustworthy by test_step_in_horizon.py, not a simplified
stand-in.
"""
import time
import numpy as np
import mujoco
import threading

from robot.franka import DOF, FrankaModel
from planner.factor_graph import SignedDistanceField
from planner.gpmp2_planner import GPMP2Planner
from cbf.barrier import DistanceBarrier, franka_dynamics
from cbf.qp_solver import CBFQPSolver
from controller.mppi import MPPIController
from planner.conflict_factor import ConflictFactorManager
from controller.covariance import CovarianceSteering
from cbf.feasibility import FeasibilityLog
from robot.mujoco_env import MujocoFrankaEnv
from demo.dashboard_state import DashboardState
from demo.threaded_pipeline import PipelineSharedState, gpmp2_thread_fn, mppi_thread_fn, robot_thread_fn

mjcf_path = "assets/panda.xml"
obstacle_center = (0.20, 0.09, 0.85)
obstacle_radius = 0.08
d_safe = 0.03
q0 = np.zeros(DOF)
q_goal = np.array([0.4, -0.2, 0.3, -1.9, 0.1, 1.6, 0.5])
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def run_with_alpha_gamma(alpha_gamma, run_seconds=90):
    env = MujocoFrankaEnv(mjcf_path=mjcf_path, obstacle_center=obstacle_center, obstacle_radius=obstacle_radius)
    franka_robot = FrankaModel(env.model, mujoco.MjData(env.model))
    franka_gpmp2 = FrankaModel(env.model, mujoco.MjData(env.model))
    franka_mppi = FrankaModel(env.model, mujoco.MjData(env.model))
    sdf = SignedDistanceField(np.array([obstacle_center]), np.array([obstacle_radius]))
    barrier_robot = DistanceBarrier(fk_fn=franka_robot.fk, sphere_radii=franka_robot.sphere_radii, sdf=sdf, dof=DOF, d_safe=d_safe)
    barrier_mppi = DistanceBarrier(fk_fn=franka_mppi.fk, sphere_radii=franka_mppi.sphere_radii, sdf=sdf, dof=DOF, d_safe=d_safe)
    qp = CBFQPSolver(m=DOF, u_min=JOINT_LOWER, u_max=JOINT_UPPER, alpha_gamma=alpha_gamma)

    def gravity_fn(q): return franka_robot.gravity(q)
    def coriolis_fn(q, qdot): return franka_robot.coriolis_times_qdot(q, qdot)
    def M_fn(q): return franka_robot.mass_matrix(q)
    def f_fn(xi):
        fi, _ = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
        return fi
    def g_fn(xi):
        _, gi = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
        return gi

    env.reset(q0)
    Qc = 0.5 * np.eye(DOF)
    planner = GPMP2Planner(dof=DOF, dt=0.05, Qc=Qc, sdf=sdf, fk_fn=franka_gpmp2.fk,
                            sphere_offsets=franka_gpmp2.sphere_radii, eps=0.02, sigma_obs=0.02)
    mppi = MPPIController(lam=1.0, dt=0.05, dof=DOF, sdf=sdf, eps_margin=0.15,
                           sigma_obs=0.02, lambda_cbf=0.3, fk_batch_fn=franka_mppi.fk_batch,
                           sphere_radii=franka_mppi.sphere_radii)
    conflict_mgr = ConflictFactorManager(tau_conflict=0.05, tau_safe=0.15)
    cov_steer = CovarianceSteering(n=DOF, Sigma0=0.002**2*np.eye(DOF), eta=0.05, beta=0.3, W=20)
    feas_log = FeasibilityLog()

    state = DashboardState()
    shared = PipelineSharedState()
    shared.set_robot_state(q0, np.zeros(DOF))
    state.q_goal = q_goal
    rng = np.random.default_rng(0)

    t1 = threading.Thread(target=gpmp2_thread_fn, args=(shared, state, planner, q_goal, DOF, 30), daemon=True)
    t2 = threading.Thread(target=mppi_thread_fn, args=(shared, state, mppi, barrier_mppi, DOF, 30, rng), daemon=True)
    t3 = threading.Thread(target=robot_thread_fn,
                           args=(shared, state, env, franka_robot, barrier_robot, qp, f_fn, g_fn,
                                 conflict_mgr, cov_steer, feas_log, DOF), daemon=True)
    t1.start(); t2.start(); t3.start()
    shared.set_start_execution()

    worst_h = float("inf")
    final_goal_err = None
    start_time = time.time()
    while time.time() - start_time < run_seconds:
        time.sleep(0.5)
        snap = state.snapshot()
        if snap["safety_margin"] is not None:
            worst_h = min(worst_h, snap["safety_margin"])
        if snap["goal_error"] is not None:
            final_goal_err = snap["goal_error"]

    shared.stop = True
    time.sleep(0.3)
    return worst_h, final_goal_err


for ag in [3000, 6000, 10000, 20000]:
    worst, final_err = run_with_alpha_gamma(ag)
    status = "SAFE" if worst >= 0 else "UNSAFE"
    print(f"alpha_gamma={ag:>6}  worst_h={worst:.4f}  final_goal_err={final_err}  {status}")

"""
test_conflict_and_covariance.py
Same threaded pipeline as before, but this time prints conflict count
and covariance size over time, so we can SEE whether Stage 6 (Conflict
Factor) and Stage 7 (Covariance Steering) are really doing anything
during a real run, not just sitting in the code unused.
"""
import time
import numpy as np
import mujoco
import threading

from robot.franka import DOF, FrankaModel
from planner.factor_graph import SignedDistanceField
from planner.gpmp2_planner import GPMP2Planner
from cbf.barrier import DistanceBarrier
from cbf.qp_solver import CBFQPSolver
from controller.mppi import MPPIController
from planner.conflict_factor import ConflictFactorManager
from controller.covariance import CovarianceSteering
from cbf.feasibility import FeasibilityLog
from robot.mujoco_env import MujocoFrankaEnv
from demo.dashboard_state import DashboardState
from demo.threaded_pipeline import PipelineSharedState, gpmp2_thread_fn, mppi_thread_fn, robot_thread_fn
from cbf.barrier import franka_dynamics

mjcf_path = "assets/panda.xml"
obstacle_center = (0.20, 0.09, 0.85)
obstacle_radius = 0.08
d_safe = 0.03

env = MujocoFrankaEnv(mjcf_path=mjcf_path, obstacle_center=obstacle_center, obstacle_radius=obstacle_radius)
franka_robot = FrankaModel(env.model, mujoco.MjData(env.model))
franka_gpmp2 = FrankaModel(env.model, mujoco.MjData(env.model))
franka_mppi = FrankaModel(env.model, mujoco.MjData(env.model))
sdf = SignedDistanceField(np.array([obstacle_center]), np.array([obstacle_radius]))
barrier_robot = DistanceBarrier(fk_fn=franka_robot.fk, sphere_radii=franka_robot.sphere_radii, sdf=sdf, dof=DOF, d_safe=d_safe)
barrier_mppi = DistanceBarrier(fk_fn=franka_mppi.fk, sphere_radii=franka_mppi.sphere_radii, sdf=sdf, dof=DOF, d_safe=d_safe)

JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
qp = CBFQPSolver(m=DOF, u_min=JOINT_LOWER, u_max=JOINT_UPPER, alpha_gamma=100.0)

def gravity_fn(q): return franka_robot.gravity(q)
def coriolis_fn(q, qdot): return franka_robot.coriolis_times_qdot(q, qdot)
def M_fn(q): return franka_robot.mass_matrix(q)
def f_fn(xi):
    fi, _ = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
    return fi
def g_fn(xi):
    _, gi = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
    return gi

q0 = np.zeros(DOF)
q_goal = np.array([0.4, -0.2, 0.3, -1.9, 0.1, 1.6, 0.5])
env.reset(q0)

Qc = 0.5 * np.eye(DOF)
planner = GPMP2Planner(dof=DOF, dt=0.05, Qc=Qc, sdf=sdf, fk_fn=franka_gpmp2.fk,
                        sphere_offsets=franka_gpmp2.sphere_radii, eps=0.02, sigma_obs=0.02)
mppi = MPPIController(lam=1.0, dt=0.05, dof=DOF, sdf=sdf, eps_margin=0.15,
                       sigma_obs=0.02, lambda_cbf=0.3, fk_batch_fn=franka_mppi.fk_batch,
                       sphere_radii=franka_mppi.sphere_radii)
conflict_mgr = ConflictFactorManager(tau_conflict=0.05, tau_safe=0.15)
Sigma0 = 0.002 ** 2 * np.eye(DOF)
cov_steer = CovarianceSteering(n=DOF, Sigma0=Sigma0, eta=0.05, beta=0.3, W=20)
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

print("Running for 90 seconds, watching conflict count and covariance size...")
prev_n_conflicts = 0
for i in range(90):
    time.sleep(1)
    n_conflicts = len(conflict_mgr.events)
    sigma_trace = float(np.trace(cov_steer.Sigma_t))
    new_conflicts = n_conflicts - prev_n_conflicts
    prev_n_conflicts = n_conflicts
    snap = state.snapshot()
    print(f"t={i+1:>3}s  goal_err={snap['goal_error']}  "
          f"conflicts_total={n_conflicts}  new_this_second={new_conflicts}  "
          f"covariance_trace={sigma_trace:.6f}")

shared.stop = True
print("Done.")
print(f"\nTotal conflict factors recorded during this run: {len(conflict_mgr.events)}")
print(f"Starting covariance trace: {float(np.trace(Sigma0)):.6f}")
print(f"Final covariance trace:    {float(np.trace(cov_steer.Sigma_t)):.6f}")

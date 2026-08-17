"""
test_final_plateau_cause.py
Same setup as test_step_in_horizon.py, but prints h(x) and whether the
CBF is actively correcting, alongside goal_err, during the final
plateau -- to test whether the robot is stopping just short of the
exact goal because getting any closer would violate safety (a real,
correct tradeoff), rather than a leftover bug.
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

# First, check directly: is the EXACT goal position itself safe?
h_at_goal = barrier_robot.forward(np.concatenate([q_goal, np.zeros(DOF)]))
print(f"h(x) AT THE EXACT GOAL POSITION: {h_at_goal:.4f}  "
      f"({'SAFE -- goal itself is fine' if h_at_goal >= 0 else 'UNSAFE -- the goal itself is inside the danger zone!'})")
print()

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

print("Running for 90 seconds, watching h(x) and CBF activity during the final plateau...")
for i in range(90):
    time.sleep(1)
    snap = state.snapshot()
    if i >= 55:  # only print the interesting final part in detail
        print(f"t={i+1}s  goal_err={snap['goal_error']}  h(x)={snap['safety_margin']}  "
              f"cbf_active={snap['cbf_active']}  intervention={snap['qp_intervention']}")

shared.stop = True
print("Done.")

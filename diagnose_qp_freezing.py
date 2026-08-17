"""
diagnose_qp_freezing.py
Prints the raw CBF quantities (h, psi1, Lf_psi1, Lg_psi1 magnitude) and
compares u_mppi (proposed) vs u_safe (QP-corrected) at several steps, to
see exactly how the QP is reacting to the new, much larger Kp-scaled
dynamics -- is it freezing near q (near-zero movement allowed), or
something else?
"""
import numpy as np
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from cbf.barrier import DistanceBarrier, hocbf_lie_derivatives, franka_dynamics
from cbf.qp_solver import CBFQPSolver, detect_unsafe
from controller.mppi import MPPIController
from planner.gpmp2_planner import GPMP2Planner

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
OBSTACLE_CENTER = (0.20, 0.09, 0.85)
OBSTACLE_RADIUS = 0.08
D_SAFE = 0.03
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=OBSTACLE_CENTER,
                       obstacle_radius=OBSTACLE_RADIUS)
franka = FrankaModel(env.model, env.data)
sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                           sdf=sdf, dof=DOF, d_safe=D_SAFE)
qp = CBFQPSolver(m=DOF, u_min=JOINT_LOWER, u_max=JOINT_UPPER, alpha_gamma=1.0)
env.reset(Q0)

def gravity_fn(q): return franka.gravity(q)
def coriolis_fn(q, qdot): return franka.coriolis_times_qdot(q, qdot)
def M_fn(q): return franka.mass_matrix(q)
def f_fn(xi):
    fi, _ = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
    return fi
def g_fn(xi):
    _, gi = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
    return gi

# Just move straight toward the goal (no MPPI needed to see the QP's behavior)
for step in range(10):
    x = env.get_state()
    q, qdot = x[:DOF], x[DOF:]
    u_mppi = np.clip(Q_GOAL, JOINT_LOWER, JOINT_UPPER)  # simple direct proposal

    psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x, f_fn, g_fn, alpha0=1.0)
    qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)

    print(f"\nstep {step}:")
    print(f"  h0={h0:.4f}  psi1={psi1:.4f}  Lf_psi1={Lf_psi1:.4f}  ||Lg_psi1||={np.linalg.norm(Lg_psi1):.4f}")
    print(f"  u_mppi (proposed): {np.round(u_mppi, 3)}")
    print(f"  u_safe (QP output): {np.round(qp_result.u_safe, 3)}")
    print(f"  q (current):        {np.round(q, 3)}")
    print(f"  intervention magnitude: {qp_result.intervention_magnitude:.4f}")
    print(f"  solve_status: {qp_result.solve_status}")

    env.step(qp_result.u_safe)

"""
check_env_step_and_fixed_vs_moving.py
1. Directly inspects how much simulated time one env.step() call
   advances (by checing sim time before/after).
2. Compares the SAME CBF pipeline under a FIXED target (like the
   earlier successful test) vs the MOVING target from GPMP2/MPPI --
   isolating whether target-shifting, not the CBF, is the real cause.
"""
import numpy as np, mujoco
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from cbf.barrier import DistanceBarrier, hocbf_lie_derivatives, franka_dynamics
from cbf.qp_solver import CBFQPSolver

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
OBSTACLE_CENTER = (0.20, 0.09, 0.85)
OBSTACLE_RADIUS = 0.08
D_SAFE = 0.03
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# ---- Part 1: how much sim time does one env.step() call advance? ----
env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=OBSTACLE_CENTER,
                       obstacle_radius=OBSTACLE_RADIUS)
env.reset(Q0)
t_before = env.data.time
env.step(np.zeros(DOF))
t_after = env.data.time
print(f"Sim time before one env.step() call: {t_before:.4f}")
print(f"Sim time after one env.step() call:  {t_after:.4f}")
print(f"Time advanced per call: {t_after - t_before:.4f}  (model timestep is 0.002)")
print(f"=> Substeps per env.step() call: {(t_after - t_before) / env.model.opt.timestep:.1f}\n")

# ---- Part 2: FIXED target through the real CBF pipeline (no MPPI/GPMP2) ----
franka = FrankaModel(env.model, mujoco.MjData(env.model))
sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                           sdf=sdf, dof=DOF, d_safe=D_SAFE)
qp = CBFQPSolver(m=DOF, u_min=JOINT_LOWER, u_max=JOINT_UPPER, alpha_gamma=1.0)

def gravity_fn(q): return franka.gravity(q)
def coriolis_fn(q, qdot): return franka.coriolis_times_qdot(q, qdot)
def M_fn(q): return franka.mass_matrix(q)
def f_fn(xi):
    fi, _ = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
    return fi
def g_fn(xi):
    _, gi = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
    return gi

env.reset(Q0)
print(f"{'step':>5}{'q0':>10}{'u_safe[0]':>12}")
for step in range(30):
    x = env.get_state()
    q = x[:DOF]
    u_mppi = np.clip(Q_GOAL, JOINT_LOWER, JOINT_UPPER)  # FIXED target, full goal, every step
    psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x, f_fn, g_fn, alpha0=1.0)
    qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
    env.step(qp_result.u_safe)
    if step % 5 == 0 or step == 29:
        print(f"{step:>5}{q[0]:>10.4f}{qp_result.u_safe[0]:>12.4f}")

print(f"\nFinal q[0] with FIXED target + CBF: {env.get_state()[0]:.4f}  (target: {Q_GOAL[0]})")

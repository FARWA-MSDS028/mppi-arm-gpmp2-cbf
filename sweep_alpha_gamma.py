"""
sweep_alpha_gamma.py
Tests whether alpha_gamma=1.0 is miscalibrated for the new,
much-larger-magnitude position-servo dynamics (Lg_psi1 ~980-989,
vs whatever much smaller scale it was implicitly tuned for before).
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

def run_test(alpha_gamma, n_steps=60):
    env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=OBSTACLE_CENTER,
                           obstacle_radius=OBSTACLE_RADIUS)
    franka = FrankaModel(env.model, mujoco.MjData(env.model))
    sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
    barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                               sdf=sdf, dof=DOF, d_safe=D_SAFE)
    qp = CBFQPSolver(m=DOF, u_min=JOINT_LOWER, u_max=JOINT_UPPER, alpha_gamma=alpha_gamma)
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

    interventions = []
    for step in range(n_steps):
        x = env.get_state()
        q = x[:DOF]
        u_mppi = np.clip(q + 0.05 * (Q_GOAL - q), JOINT_LOWER, JOINT_UPPER)
        psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x, f_fn, g_fn, alpha0=1.0)
        qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
        interventions.append(qp_result.intervention_magnitude)
        env.step(qp_result.u_safe)

    final_err = float(np.linalg.norm(env.get_state()[:DOF] - Q_GOAL, ord=np.inf))
    print(f"alpha_gamma={alpha_gamma:>8}  final_intervention={interventions[-1]:.4f}  "
          f"final_goal_err={final_err:.4f}")

for ag in [1.0, 10.0, 50.0, 200.0, 1000.0]:
    run_test(ag)

"""
diagnose_feasibility.py
Runs longer, tracking qdot growth alongside intervention magnitude and
solve_status -- direct test of the feasibility hypothesis. Also compares
against a SMOOTHED reference (small incremental steps toward the goal,
like GPMP2/MPPI would actually provide) to see if the aggressive
constant-full-goal command was itself the problem.
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

def run_test(label, command_fn, n_steps=40):
    env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=OBSTACLE_CENTER,
                           obstacle_radius=OBSTACLE_RADIUS)
    franka = FrankaModel(env.model, mujoco.MjData(env.model))
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

    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"{'step':>5}{'qdot_norm':>12}{'intervention':>14}{'status':>14}")
    for step in range(n_steps):
        x = env.get_state()
        q, qdot = x[:DOF], x[DOF:]
        u_mppi = command_fn(q, step)
        psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x, f_fn, g_fn, alpha0=1.0)
        qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
        if step % 5 == 0 or step == n_steps - 1:
            print(f"{step:>5}{np.linalg.norm(qdot):>12.4f}{qp_result.intervention_magnitude:>14.4f}"
                  f"{qp_result.solve_status:>14}")
        env.step(qp_result.u_safe)
    final_err = float(np.linalg.norm(env.get_state()[:DOF] - Q_GOAL, ord=np.inf))
    print(f"Final goal_err: {final_err:.4f}")

# Test A: constant full-goal command (what we just tested -- likely aggressive)
run_test("TEST A: constant full-goal command (aggressive)",
          lambda q, step: np.clip(Q_GOAL, JOINT_LOWER, JOINT_UPPER))

# Test B: smoothed incremental command (small step toward goal each time)
run_test("TEST B: smoothed incremental command (5% of remaining distance per step)",
          lambda q, step: np.clip(q + 0.05 * (Q_GOAL - q), JOINT_LOWER, JOINT_UPPER))

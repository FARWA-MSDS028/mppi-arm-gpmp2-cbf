"""
test_qp_intervention.py
==========================
Standalone sanity check, isolated from any scenario-design question:
constructs the exact same barrier/dynamics/QP objects main.py uses,
then feeds the QP a DELIBERATELY ADVERSARIAL control (chosen to
minimize psi1_dot, i.e. actively push toward the constraint boundary)
at a state where h0 < 0. If intervention_magnitude comes back nonzero
here, the QP mechanism itself is proven correct -- whatever happens in
the full closed loop after that is a scenario/physics question, not a
QP-correctness question.

Usage: python test_qp_intervention.py --mjcf assets/panda.xml
"""
from __future__ import annotations
import argparse
import numpy as np

from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF, TAU_MAX
from planner.factor_graph import SignedDistanceField
from cbf.barrier import DistanceBarrier, hocbf_lie_derivatives, franka_dynamics
from cbf.qp_solver import CBFQPSolver, detect_unsafe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=str, default="assets/panda.xml")
    args = parser.parse_args()

    env = MujocoFrankaEnv(mjcf_path=args.mjcf, obstacle_center=(0.22, 0.08, 0.92),
                           obstacle_radius=0.12)
    import mujoco
    franka = FrankaModel(env.model, mujoco.MjData(env.model))
    sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
    barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                               sdf=sdf, dof=DOF, d_safe=0.10)

    q0 = np.zeros(DOF)
    env.reset(q0)
    x = env.get_state()

    def gravity_fn(q):
        return franka.gravity(q)

    def coriolis_fn(q, qdot):
        return franka.coriolis_times_qdot(q, qdot)

    def M_fn(q):
        return franka.mass_matrix(q)

    def f_fn(xi):
        fi, _ = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
        return fi

    def g_fn(xi):
        _, gi = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
        return gi

    psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x, f_fn, g_fn, alpha0=1.0)
    print(f"State: h0={h0:.4f}, psi1={psi1:.4f}, Lf_psi1={Lf_psi1:.4f}, "
          f"||Lg_psi1||={np.linalg.norm(Lg_psi1):.4f}")

    qp = CBFQPSolver(m=DOF, u_min=-TAU_MAX, u_max=TAU_MAX, alpha_gamma=1.0)

    # Test 1: a "do-nothing" control (u=0). Given Lf_psi1 already large
    # and positive here, this is likely STILL safe (confirms the
    # "gravity already retreating" reading of the debug output).
    u_zero = np.zeros(DOF)
    qp_result = qp.solve(u_zero, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
    print(f"\nTest 1 (u=0):"
          f" intervention={qp_result.intervention_magnitude:.6f}"
          f" status={qp_result.solve_status}")

    # Test 2: an ADVERSARIAL control chosen to actively MINIMIZE psi1_dot
    # (push as hard as possible in the -Lg_psi1 direction, i.e. exactly
    # the direction that makes the constraint hardest to satisfy) at
    # max torque. This should force psi1_dot below the safety threshold
    # if left uncorrected -- if the QP is working, intervention should
    # now be clearly nonzero.
    if np.linalg.norm(Lg_psi1) > 1e-8:
        u_adversarial = -TAU_MAX * (Lg_psi1 / np.linalg.norm(Lg_psi1))
    else:
        u_adversarial = -TAU_MAX
    predicted_hdot = Lf_psi1 + float(Lg_psi1 @ u_adversarial)
    print(f"\nTest 2 (adversarial u, worst-case direction at max torque):"
          f"\n  u_adversarial norm={np.linalg.norm(u_adversarial):.2f}"
          f"\n  predicted psi1_dot under this u = {predicted_hdot:.4f}"
          f" (negative = would violate the class-K bound)")
    qp_result2 = qp.solve(u_adversarial, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
    print(f"  intervention={qp_result2.intervention_magnitude:.6f}"
          f" status={qp_result2.solve_status}"
          f" active_constraints={qp_result2.active_constraints}")

    if qp_result2.intervention_magnitude > 1e-4:
        print("\n==> QP mechanism CONFIRMED working: it corrected the "
              "adversarial control. The 0.0000 intervention you saw in "
              "the full closed loop reflects that u_mppi (and even u=0) "
              "already happened to be safe at those states, not a QP bug.")
    else:
        print("\n==> QP still did not intervene even against an adversarial "
              "control -- this points to a remaining issue (check "
              "predicted psi1_dot above: if it's still >= the class-K "
              "bound even at max adversarial torque, the constraint is "
              "genuinely non-binding at this state, which would itself "
              "be worth double-checking against TAU_MAX's scale).")


if __name__ == "__main__":
    main()

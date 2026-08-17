"""
find_obstacle_placement.py
=============================
Diagnostic: prints the workspace (x,y,z) position of every collision
sphere along the straight-line joint-space interpolation from Q0 to
Q_GOAL, so you can pick an obstacle center that the arm ACTUALLY passes
near -- instead of guessing coordinates blind.

Run this BEFORE re-running the CBF/full-loop experiments after seeing
0/150 unsafe steps. Zero interventions almost always means the
obstacle just isn't close to the path GPMP2/MPPI actually take, not a
bug in the CBF-QP itself.

Usage
-----
    python find_obstacle_placement.py --mjcf assets/panda.xml

Then look at the printed table: pick the (x,y,z) row with the smallest
spread across spheres/timesteps as your new --obstacle-x/y/z, and set
--obstacle-radius large enough (and/or --d-safe large enough) that the
margin (obstacle surface to nearest sphere, minus d_safe) is small or
negative at that point -- i.e. deliberately put the obstacle where the
arm's own reach naturally passes.
"""

from __future__ import annotations
import argparse
import numpy as np

from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=str, default="assets/panda.xml")
    parser.add_argument("--n-samples", type=int, default=11,
                         help="number of interpolation points between Q0 and Q_GOAL")
    args = parser.parse_args()

    # Obstacle placement doesn't matter for this diagnostic -- we're
    # only using the env to get a MuJoCo model/data pair for FK.
    env = MujocoFrankaEnv(mjcf_path=args.mjcf)
    import mujoco
    franka = FrankaModel(env.model, mujoco.MjData(env.model))

    print(f"{'alpha':>6}  {'sphere_idx':>10}  {'x':>8}  {'y':>8}  {'z':>8}")
    print("-" * 50)

    all_centers = []
    for alpha in np.linspace(0.0, 1.0, args.n_samples):
        q = (1 - alpha) * Q0 + alpha * Q_GOAL
        centers, _ = franka.fk(q)  # (M, 3)
        all_centers.append(centers)
        for j, c in enumerate(centers):
            print(f"{alpha:6.2f}  {j:10d}  {c[0]:8.3f}  {c[1]:8.3f}  {c[2]:8.3f}")

    all_centers = np.stack(all_centers)  # (n_samples, M, 3)
    mean_pos = all_centers.mean(axis=(0, 1))
    print("\n" + "=" * 50)
    print(f"Mean sphere position across the whole path: "
          f"({mean_pos[0]:.3f}, {mean_pos[1]:.3f}, {mean_pos[2]:.3f})")

    # Also report the hand/end-effector position at each alpha, since
    # that's usually the most intuitive "where does the arm go" signal.
    print("\nEnd-effector (hand) position at each alpha:")
    for alpha in np.linspace(0.0, 1.0, args.n_samples):
        q = (1 - alpha) * Q0 + alpha * Q_GOAL
        franka.fk(q)  # updates internal mj_data via _set_q + mj_kinematics
        import mujoco
        mujoco.mj_kinematics(env.model, env.data)
        ee = env.data.xpos[env.model.body("hand").id].copy()
        print(f"  alpha={alpha:.2f}: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")

    print("\nSuggested next step: pick an obstacle center close to one of the\n"
          "positions above (ideally one near the MIDDLE of the path, alpha~0.4-0.6,\n"
          "since GPMP2/MPPI will otherwise route around an obstacle near either\n"
          "endpoint fairly cheaply), then rerun with e.g.:\n"
          "  python run_experiments.py --mjcf assets/panda.xml --only safe_mppi \\\n"
          "      --obstacle-x <x> --obstacle-y <y> --obstacle-z <z> \\\n"
          "      --obstacle-radius 0.12 --d-safe 0.08")


if __name__ == "__main__":
    main()

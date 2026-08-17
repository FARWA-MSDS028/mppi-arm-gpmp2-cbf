"""
test_safe_mppi_standalone.py
Stage 3 standalone: GPMP2 -> MPPI -> CBF-QP, with the corrected
position-actuator CBF math and sigma=0.002. This is the real test of
your safety requirement: does the hard constraint actually keep h(x) >= 0
where Stage 2 landed right at the boundary?
"""
import numpy as np
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from cbf.barrier import DistanceBarrier
from experiments.safe_mppi import run_safe_mppi

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
SUCCESS_THRESHOLD = 0.15
OBSTACLE_CENTER = (0.20, 0.09, 0.85)
OBSTACLE_RADIUS = 0.08
D_SAFE = 0.03

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=OBSTACLE_CENTER,
                       obstacle_radius=OBSTACLE_RADIUS)
franka = FrankaModel(env.model, env.data)
sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                           sdf=sdf, dof=DOF, d_safe=D_SAFE)
env.reset(Q0)

h0 = barrier.forward(np.concatenate([Q0, np.zeros(DOF)]))
print(f"Pre-flight: h(q0) = {h0:.4f}  ({'SAFE START' if h0 >= 0 else 'UNSAFE START'})")

log, feas_log = run_safe_mppi(env, franka, sdf, barrier, q0=Q0, q_goal=Q_GOAL, rng_seed=0, sigma=0.002)

goal_err = log["goal_error"][-1]
min_clear = min(log["dist"])
h = min_clear - D_SAFE
n_unsafe = sum(log["unsafe_flags"])
n_total = len(log["unsafe_flags"])

print(f"\nfinal goal_err: {goal_err:.4f}  (success: {goal_err < SUCCESS_THRESHOLD})")
print(f"min_clear: {min_clear:.4f}   worst h(x): {h:.6f}  (safe: {h >= -1e-6})")
print(f"MPPI proposed unsafe on {n_unsafe}/{n_total} steps -- CBF-QP corrected each")

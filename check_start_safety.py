import numpy as np, mujoco
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from cbf.barrier import DistanceBarrier

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml",
                       obstacle_center=(0.20, 0.09, 0.85), obstacle_radius=0.08)
franka = FrankaModel(env.model, mujoco.MjData(env.model))
sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                           sdf=sdf, dof=DOF, d_safe=0.03)

q0 = np.zeros(DOF)
x0 = np.concatenate([q0, np.zeros(DOF)])
h0 = barrier.forward(x0)
print(f"h(q0) = {h0:.4f}  -->  {'SAFE START' if h0 >= 0 else 'STILL UNSAFE, need to adjust'}")

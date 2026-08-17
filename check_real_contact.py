"""
check_real_contact.py
Checks whether the obstacle is a real collision-enabled MuJoCo body
generating actual contact forces, which would explain motion being
throttled despite u_mppi == u_safe == full goal position every step.
"""
import numpy as np
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import DOF

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(0.20, 0.09, 0.85),
                       obstacle_radius=0.08)
env.reset(np.zeros(DOF))

print(f"Number of geoms in the model: {env.model.ngeom}")
print(f"Timestep (env.model.opt.timestep): {env.model.opt.timestep}")

# Step a few times with a strong direct command and watch contacts + qacc
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
for step in range(5):
    env.step(Q_GOAL)
    print(f"\nstep {step}: active contacts = {env.data.ncon}")
    if env.data.ncon > 0:
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            import mujoco
            g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            print(f"  CONTACT: {g1} <-> {g2}  dist={c.dist:.5f}")
    print(f"  qacc: {np.round(env.data.qacc[:DOF], 4)}")
    print(f"  qvel: {np.round(env.data.qvel[:DOF], 4)}")

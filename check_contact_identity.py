"""
check_contact_identity.py
1. Confirms whether the obstacle is EVER inserted into the physical
   MuJoCo model (by checking if ngeom changes with different obstacle
   parameters).
2. Identifies exactly which body pairs are generating the real contacts
   seen earlier, by NAME (not "None <-> None"), to determine if they're
   robot self-collision rather than arm-obstacle collision.
"""
import numpy as np, mujoco
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import DOF

# ---- Test 1: does the obstacle actually change the physical model? ----
env_a = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(0.20, 0.09, 0.85), obstacle_radius=0.08)
env_b = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(5.0, 5.0, 5.0), obstacle_radius=0.5)
print(f"ngeom with obstacle_radius=0.08 at (0.20,0.09,0.85): {env_a.model.ngeom}")
print(f"ngeom with obstacle_radius=0.5  at (5.0,5.0,5.0):    {env_b.model.ngeom}")
print(f"SAME MODEL (obstacle is purely virtual/Python-side): {env_a.model.ngeom == env_b.model.ngeom}")

# ---- Test 2: identify the real contacts by body NAME ----
env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(0.20, 0.09, 0.85), obstacle_radius=0.08)
env.reset(np.zeros(DOF))
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
env.step(Q_GOAL)

print(f"\nActive contacts: {env.data.ncon}")
for i in range(env.data.ncon):
    c = env.data.contact[i]
    body1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, env.model.geom_bodyid[c.geom1])
    body2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, env.model.geom_bodyid[c.geom2])
    print(f"  geom{c.geom1}(body={body1}) <-> geom{c.geom2}(body={body2})  dist={c.dist:.5f}")

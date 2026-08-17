"""
compare_barrier_vs_real_geometry.py
Directly compares our own DistanceBarrier's d(x) against MuJoCo's
authoritative real geom-to-geom distance (mj_geomDistance), at the exact
same configuration -- the definitive test of whether our safety math
corresponds to real physical geometry, or has been silently miscalibrated.
"""
import numpy as np, mujoco
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF
from planner.factor_graph import SignedDistanceField
from cbf.barrier import DistanceBarrier

OBSTACLE_CENTER = (0.20, 0.09, 0.85)
OBSTACLE_RADIUS = 0.08
D_SAFE = 0.03

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=OBSTACLE_CENTER,
                       obstacle_radius=OBSTACLE_RADIUS)
franka = FrankaModel(env.model, env.data)
sdf = SignedDistanceField(np.array([env.obstacle_center]), np.array([env.obstacle_radius]))
barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                           sdf=sdf, dof=DOF, d_safe=D_SAFE)

q0 = np.zeros(DOF)
env.reset(q0)

# ---- Our own barrier's answer ----
x0 = np.concatenate([q0, np.zeros(DOF)])
d_ours = barrier._closest(q0)[0] if hasattr(barrier, "_closest") else None
h_ours = barrier.forward(x0)
print(f"Our DistanceBarrier: h(q0) = {h_ours:.4f}  =>  implies d(q0) = {h_ours + D_SAFE:.4f}")

# ---- Find the obstacle geom in the real model ----
print(f"\nAll geoms in the model (name, type, size):")
obstacle_geom_id = None
for i in range(env.model.ngeom):
    name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)
    gtype = env.model.geom_type[i]
    size = env.model.geom_size[i]
    body_id = env.model.geom_bodyid[i]
    body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    tag = ""
    if name and "obstacle" in (name or "").lower():
        obstacle_geom_id = i
        tag = "  <-- OBSTACLE?"
    print(f"  id={i:>3} name={str(name):>20} body={str(body_name):>20} type={gtype} size={size}{tag}")

if obstacle_geom_id is None:
    print("\nCould not auto-find obstacle geom by name -- please tell me which id above is the obstacle.")
else:
    print(f"\nUsing obstacle geom id={obstacle_geom_id}, size={env.model.geom_size[obstacle_geom_id]}")
    print(f"(we PASSED obstacle_radius={OBSTACLE_RADIUS} -- does the actual geom size match?)")

    # ---- Real MuJoCo distance from EVERY arm geom to the obstacle ----
    mujoco.mj_forward(env.model, env.data)
    print(f"\nReal MuJoCo distance from each arm geom to the obstacle geom:")
    min_real_dist = float("inf")
    min_geom_name = None
    for i in range(env.model.ngeom):
        if i == obstacle_geom_id:
            continue
        body_id = env.model.geom_bodyid[i]
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None or "panda" not in str(body_name).lower() and "link" not in str(body_name).lower() and "hand" not in str(body_name).lower():
            continue  # skip non-arm geoms (floor, etc.)
        dist = mujoco.mj_geomDistance(env.model, env.data, i, obstacle_geom_id, 10.0, None)
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)
        print(f"  geom {i} ({name}, body={body_name}): real distance = {dist:.5f}")
        if dist < min_real_dist:
            min_real_dist = dist
            min_geom_name = name

    print(f"\n{'='*60}")
    print(f"Our barrier says d(q0)  = {h_ours + D_SAFE:.4f}")
    print(f"Real MuJoCo says d(q0)  = {min_real_dist:.4f}  (closest geom: {min_geom_name})")
    print(f"DISCREPANCY: {abs((h_ours + D_SAFE) - min_real_dist):.4f} m")
    print(f"{'='*60}")

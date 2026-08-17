"""
diagnose_stuck_joint.py
Identifies which specific joint(s) are stuck at the plateau, and checks
the two most likely physical explanations: a joint sitting at its hard
mechanical limit, or self-collision between links.
"""
import numpy as np, mujoco
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF, TAU_MAX

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(5.0, 5.0, 5.0),
                       obstacle_radius=0.01)
franka = FrankaModel(env.model, mujoco.MjData(env.model))

q0 = np.zeros(DOF)
q_goal = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
env.reset(q0)

Kp = np.array([80, 80, 80, 60, 40, 30, 20], dtype=float)
Kd = 2.0 * np.sqrt(Kp)

for step in range(1000):  # run well past the plateau point (~step 300)
    x = env.get_state()
    q, qdot = x[:DOF], x[DOF:]
    e = q_goal - q
    qdd_des = Kp * e - Kd * qdot
    M = franka.mass_matrix(q)
    C = franka.coriolis_times_qdot(q, qdot)
    G = franka.gravity(q)
    u = np.clip(M @ qdd_des + C + G, -TAU_MAX, TAU_MAX)
    env.step(u)

x_final = env.get_state()
q_final, qdot_final = x_final[:DOF], x_final[DOF:]

# ---- Per-joint breakdown ----
print(f"{'joint':>8}{'q_final':>10}{'q_goal':>10}{'error':>10}{'lower':>10}{'upper':>10}{'AT LIMIT?':>12}")
for i in range(DOF):
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
    lo, hi = env.model.jnt_range[jid]
    at_limit = (abs(q_final[i] - lo) < 0.01) or (abs(q_final[i] - hi) < 0.01)
    print(f"{i+1:>8}{q_final[i]:>10.4f}{q_goal[i]:>10.4f}{q_goal[i]-q_final[i]:>10.4f}"
          f"{lo:>10.4f}{hi:>10.4f}{str(at_limit):>12}")

# ---- Self-collision check ----
print(f"\nActive contacts at final settled state: {env.data.ncon}")
if env.data.ncon > 0:
    print("Contact details (geom1, geom2, distance -- negative = actual penetration):")
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        print(f"  {g1_name} <-> {g2_name}: dist={c.dist:.5f}")
else:
    print("No active contacts -- self-collision is NOT the cause.")

# ---- Final torque required to HOLD this position (should be small if truly settled) ----
M = franka.mass_matrix(q_final)
G = franka.gravity(q_final)
print(f"\nGravity-compensation torque at final pose: {np.round(G, 3)}")
print(f"TAU_MAX:                                    {TAU_MAX}")
print(f"Fraction of TAU_MAX used just to hold here:  {np.round(np.abs(G) / TAU_MAX, 3)}")

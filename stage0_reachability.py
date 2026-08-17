import numpy as np, mujoco
from robot.mujoco_env import MujocoFrankaEnv
from robot.franka import FrankaModel, DOF, TAU_MAX

env = MujocoFrankaEnv(mjcf_path="assets/panda.xml", obstacle_center=(5.0, 5.0, 5.0),
                       obstacle_radius=0.01)
franka = FrankaModel(env.model, mujoco.MjData(env.model))  # <-- fixed: separate data

q0 = np.zeros(DOF)
q_goal = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
env.reset(q0)

Kp = np.array([80, 80, 80, 60, 40, 30, 20], dtype=float)
Kd = 2.0 * np.sqrt(Kp)

n_steps = 300
errs = []
for step in range(n_steps):
    x = env.get_state()
    q, qdot = x[:DOF], x[DOF:]
    e = q_goal - q
    qdd_des = Kp * e - Kd * qdot
    M = franka.mass_matrix(q)
    C = franka.coriolis_times_qdot(q, qdot)
    G = franka.gravity(q)
    u = np.clip(M @ qdd_des + C + G, -TAU_MAX, TAU_MAX)
    env.step(u)
    errs.append(float(np.linalg.norm(q - q_goal, ord=np.inf)))

print(f"goal_err after {n_steps} steps: {errs[-1]:.4f}")
print(f"min goal_err reached: {min(errs):.4f} at step {errs.index(min(errs))}")
print(f"REACHABLE: {'YES' if min(errs) < 0.15 else 'NO'}")

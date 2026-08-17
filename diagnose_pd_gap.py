"""
diagnose_pd_gap.py
Tests three things at once, using your ACTUAL franka.py / mujoco_env.py:
1. Is the trend still improving, or has it genuinely plateaued? (longer
   run, trend printed periodically)
2. Is torque actually saturating at TAU_MAX? (tracks fraction of
   joint-steps pinned at the limit)
3. Is FrankaModel's dynamics (M, C, G) actually synchronized with the
   REAL q/qdot MuJoCo is executing, cross-checked against raw MuJoCo
   computed independently on a fresh MjData for the same q/qdot?
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

print(f"TAU_MAX per joint: {TAU_MAX}")
print(f"\n{'step':>6}{'goal_err':>12}{'max|u|/TAU_MAX':>18}{'n_joints_saturated':>20}")

n_steps = 1500
errs = []
sat_fractions = []
raw_check_done = False

for step in range(n_steps):
    x = env.get_state()
    q, qdot = x[:DOF], x[DOF:]
    e = q_goal - q
    qdd_des = Kp * e - Kd * qdot

    M = franka.mass_matrix(q)
    C = franka.coriolis_times_qdot(q, qdot)
    G = franka.gravity(q)

    # --- Cross-check #3: verify against RAW MuJoCo on a completely
    # independent MjData, for the SAME q/qdot -- only done once, at a
    # representative mid-trajectory point, to keep this fast.
    if not raw_check_done and step == 50:
        raw_data = mujoco.MjData(env.model)
        raw_data.qpos[:DOF] = q
        raw_data.qvel[:DOF] = qdot
        mujoco.mj_forward(env.model, raw_data)
        raw_G = raw_data.qfrc_bias.copy()[:DOF]  # bias = C(q,qdot)*qdot + G(q) combined in MuJoCo's convention
        # FrankaModel's C + G should match MuJoCo's combined bias force at this q,qdot
        combined_franka = C + G
        diff = np.abs(combined_franka - raw_G)
        print(f"\n  [Cross-check @ step 50] FrankaModel's (C+G) vs raw MuJoCo qfrc_bias:")
        print(f"    FrankaModel: {np.round(combined_franka, 3)}")
        print(f"    Raw MuJoCo:  {np.round(raw_G, 3)}")
        print(f"    Max diff:    {np.max(diff):.6f}  ({'MATCH' if np.max(diff) < 0.01 else 'MISMATCH -- FrankaModel is desynced!'})\n")
        raw_check_done = True

    u_unclipped = M @ qdd_des + C + G
    u = np.clip(u_unclipped, -TAU_MAX, TAU_MAX)
    n_saturated = int(np.sum(np.abs(u_unclipped) >= TAU_MAX - 1e-6))
    max_ratio = float(np.max(np.abs(u_unclipped) / TAU_MAX))

    env.step(u)
    e_now = float(np.linalg.norm(q - q_goal, ord=np.inf))
    errs.append(e_now)
    sat_fractions.append(n_saturated)

    if step % 100 == 0 or step == n_steps - 1:
        print(f"{step:>6}{e_now:>12.4f}{max_ratio:>18.2f}{n_saturated:>20d}")

print(f"\nFinal goal_err: {errs[-1]:.4f}")
print(f"Min goal_err ever: {min(errs):.4f} at step {errs.index(min(errs))}")
print(f"Average number of joints saturated per step: {np.mean(sat_fractions):.2f} / {DOF}")
print(f"Steps with ANY joint saturated: {sum(1 for s in sat_fractions if s > 0)}/{n_steps}")

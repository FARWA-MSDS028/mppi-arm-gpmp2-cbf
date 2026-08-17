"""
test_exclude_self_collision.py (properly time-matched)
Substeps 25 times per iteration (dt=0.05 / timestep=0.002) to match
MujocoFrankaEnv.step()'s real simulated time per call -- the previous
version only ran 0.4s total instead of the intended ~10s.
"""
import numpy as np, mujoco, os
from robot.franka import DOF

Q0 = np.zeros(DOF)
Q_GOAL = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

with open("assets/panda.xml") as f:
    xml_text = f.read()

exclude_block = '''
  <contact>
    <exclude body1="link5" body2="hand"/>
    <exclude body1="left_finger" body2="right_finger"/>
  </contact>
'''
insert_pos = xml_text.index(">", xml_text.index("<mujoco")) + 1
xml_text_modified = xml_text[:insert_pos] + exclude_block + xml_text[insert_pos:]

temp_path = "assets/panda_test_exclude.xml"
with open(temp_path, "w") as f:
    f.write(xml_text_modified)

model = mujoco.MjModel.from_xml_path(temp_path)
data = mujoco.MjData(model)
data.qpos[:DOF] = Q0
mujoco.mj_forward(model, data)

print(f"model timestep: {model.opt.timestep}  -- substepping 25x per iteration to match dt=0.05")

for step in range(200):
    q = data.qpos[:DOF].copy()
    ctrl = np.clip(q + 0.05 * (Q_GOAL - q), JOINT_LOWER, JOINT_UPPER)
    data.ctrl[:DOF] = ctrl
    for _ in range(25):  # <-- the fix: 25 physics substeps = one real dt=0.05 control period
        mujoco.mj_step(model, data)
    if step % 40 == 0 or step == 199:
        err = float(np.linalg.norm(data.qpos[:DOF] - Q_GOAL, ord=np.inf))
        print(f"step {step:>4}  goal_err={err:.4f}  ncon={data.ncon}")

final_err = float(np.linalg.norm(data.qpos[:DOF] - Q_GOAL, ord=np.inf))
print(f"\nFinal goal_err with self-collision EXCLUDED (properly time-matched): {final_err:.4f}")

os.remove(temp_path)

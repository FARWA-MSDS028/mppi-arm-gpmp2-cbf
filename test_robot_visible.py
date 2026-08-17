"""
test_robot_visible.py
Isolates whether the robot renders at all with our camera setup --
completely separate from any custom-geom or threading logic. Saves a
single PNG to inspect directly.
"""
import numpy as np
import mujoco
import cv2

model = mujoco.MjModel.from_xml_path("assets/panda.xml")
data = mujoco.MjData(model)

# Pose the arm at a clearly non-zero, visually obvious configuration
data.qpos[:7] = [0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5]
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)

cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)
cam.lookat[:] = [0.3, 0.0, 0.5]
cam.distance = 1.6
cam.azimuth = 120
cam.elevation = -20

renderer.update_scene(data, camera=cam)
img = renderer.render()
cv2.imwrite("robot_test.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("Saved robot_test.png -- open it and check if the robot appears.")

# Also print where the robot bodies actually are in world space, to
# sanity-check against the camera's lookat point.
for name in ["link1", "link4", "hand"]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    print(f"{name} world position: {data.xpos[bid]}")

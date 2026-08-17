"""
robot/mujoco_env.py
=====================
Implements Stage 6: robot execution.  Loads the Franka Panda MuJoCo
model, steps the simulator with ONLY the CBF-QP-filtered control u*
(never u_mppi directly), and exposes rendering hooks.

Uses `robot.franka.resolve_arm_indices` to find the 7 arm joints'
qpos/qvel addresses and the 7 arm actuators' ids BY NAME, so reset(),
step(), and get_state() write/read the correct slots even if the MJCF
also defines gripper joints/actuators (this fixes a real bug from the
previous version, which assumed the arm occupies qpos[:7]/ctrl[:7]).

MuJoCo's official Menagerie Franka Panda XML
(`franka_emika_panda/panda.xml`) is the expected asset; place it at
`assets/panda.xml` relative to the project root, or point
`MJCF_PATH` to your local copy.
"""

from __future__ import annotations
import os
import numpy as np

from robot.franka import resolve_arm_indices, DOF

MJCF_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "assets", "panda.xml")


class MujocoFrankaEnv:
    """
    Time complexity per step: MuJoCo's mj_step is O(dof) for this small
    articulated chain.
    """
    def __init__(self, mjcf_path: str = MJCF_PATH_DEFAULT,
                 obstacle_center=(0.5, 0.0, 0.4), obstacle_radius=0.08,
                 target_pos=(0.5, 0.3, 0.5), dt: float = 0.002,
                 control_dt: float = 0.05, gripper_qpos: float = 0.04):
        # control_dt is the CONTROL period every GPMP2Planner/
        # MPPIController construction elsewhere in this codebase assumes
        # (dt=0.05). dt (0.002) is the fine-grained PHYSICS timestep --
        # needed separately because these position-servo actuators are
        # very stiff (kp up to 4500) and would be numerically unstable
        # at a larger physics step. Before this fix, step() called
        # mj_step() exactly ONCE per call, silently advancing only dt
        # (0.002s) instead of control_dt (0.05s) -- a 25x time
        # compression present in every experiment run this session.
        import mujoco
        if not os.path.exists(mjcf_path):
            raise FileNotFoundError(
                f"Franka MJCF not found at {mjcf_path}. Download MuJoCo "
                f"Menagerie's franka_emika_panda/panda.xml and place it there, "
                f"or pass mjcf_path=... explicitly.")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)

        idx = resolve_arm_indices(self.model)
        self.qpos_adr = idx["qpos_adr"]
        self.qvel_adr = idx["qvel_adr"]
        self.actuator_ids = idx["actuator_ids"]

        self.obstacle_center = np.array(obstacle_center)
        self.obstacle_radius = obstacle_radius
        self.target_pos = np.array(target_pos)
        self.dt = dt
        self.control_dt = control_dt
        self.n_substeps = round(control_dt / dt)  # = 25 for the defaults above
        self.gripper_qpos = gripper_qpos  # fixed-open gripper; not actuated by our loop

    def reset(self, q0: np.ndarray, qdot0: np.ndarray = None):
        import mujoco
        self.data.qpos[self.qpos_adr] = q0
        if qdot0 is not None:
            self.data.qvel[self.qvel_adr] = qdot0
        mujoco.mj_forward(self.model, self.data)

    def step(self, u_star: np.ndarray):
        """
        Execute ONLY u* (Stage 6: "Execute only u* inside MuJoCo").
        u_star : (7,) desired joint POSITION command (post CBF-QP
                 filtering; confirmed by direct testing that these
                 actuators are position servos, not torque motors),
                 written to the 7 arm actuators BY ID, not by a
                 positional slice.

        Advances n_substeps physics steps (fine-grained dt) to cover one
        full control_dt period -- matching the dt=0.05 assumed by every
        GPMP2Planner/MPPIController construction elsewhere. u_star is
        held constant across all substeps (a standard zero-order-hold),
        exactly like the successful verify_substep_fix.py test.
        """
        import mujoco
        self.data.ctrl[self.actuator_ids] = u_star
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def substep(self, u_star: np.ndarray):
        """
        Executes exactly ONE physics tick (self.dt = 0.002s), no
        internal loop -- for callers that need to re-check the CBF
        constraint and re-correct faster than one full control_dt
        (0.05s) period. Confirmed necessary by test_fast_cbf_loop.py:
        holding a single CBF-QP solution across all 25 substeps let the
        robot drift past the safety boundary within one hold period,
        even though the control was safe at the instant it was
        computed. Used by demo/threaded_pipeline.py's robot_thread_fn
        to run the CBF-QP + robot execution loop at ~500Hz, matching
        that proven-safe configuration.

        u_star : (7,) desired joint POSITION command (post CBF-QP
                 filtering), same convention as step().
        """
        import mujoco
        self.data.ctrl[self.actuator_ids] = u_star
        mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def get_state(self):
        """Returns [q_arm (7,), qdot_arm (7,)] -- the arm's DOF only,
        regardless of how many extra (e.g. gripper) DOF the model has."""
        q = self.data.qpos[self.qpos_adr].copy()
        qdot = self.data.qvel[self.qvel_adr].copy()
        return np.concatenate([q, qdot])

    def ee_position(self, ee_body_name: str = "hand") -> np.ndarray:
        bid = self.model.body(ee_body_name).id
        return self.data.xpos[bid].copy()

    # ---- Rendering (Stage 6 visualization requirements) -----------------
    def render_frame(self, width=640, height=480):
        """Offscreen RGB render for building trajectory/animation figures."""
        import mujoco
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data)
        img = renderer.render()
        renderer.close()
        return img

    def launch_interactive_viewer(self):
        """Opens MuJoCo's passive viewer for live inspection."""
        import mujoco.viewer
        return mujoco.viewer.launch_passive(self.model, self.data)

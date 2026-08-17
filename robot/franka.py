"""
robot/franka.py
=================
Franka Panda forward-kinematics + dynamics accessors, built on top of
the MuJoCo model (robot/mujoco_env.py). All downstream modules
(GPMP2 obstacle factor, MPPI cost, CBF dynamics) call through this
file so there is exactly ONE source of truth for the robot model.

IMPORTANT -- fixes a real bug from the previous version: the arm's 7
DOF are NOT assumed to occupy qpos[:7] / qvel[:7] / ctrl[:7] anymore.
Most Panda MJCFs (including MuJoCo Menagerie's) have additional
gripper joints/actuators (finger_joint1, finger_joint2, and their
actuator(s)), so nq/nv/nu can be 8, 9, or more depending on the model
variant. Every arm-joint/actuator index used below is resolved BY NAME
once at __init__ time via `resolve_arm_indices`, and all reads/writes
use that explicit index array (fancy indexing) instead of a slice.
This is what makes the code robust whether the model reports nu=7,
nu=8 (one combined gripper actuator), or nu=9.

Collision geometry: the Panda is approximated by swept spheres along
the kinematic chain (interpolated along each link segment), matching
GPMP2's "robot as spheres" collision model (Step 1.3, obstacle
factor: s_j = robot spheres).
"""

from __future__ import annotations
import numpy as np

DOF = 7

# Panda joint limits (rad), from Franka's data sheet, joints 1-7 in order.
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
TAU_MAX = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)  # Nm, joints 1-7

ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]      # joint1..joint7
ARM_ACTUATOR_NAMES = [f"actuator{i}" for i in range(1, 8)]  # actuator1..actuator7


def resolve_arm_indices(model):
    """
    Resolves, BY NAME, the 7 arm joints' qpos/qvel addresses and the 7
    arm actuators' ids -- regardless of how many other joints/actuators
    (gripper fingers, etc.) the MJCF defines or in what order.

    Returns a dict:
        qpos_adr    : (7,) int array, indices into data.qpos
        qvel_adr    : (7,) int array, indices into data.qvel / data.qfrc_*
        actuator_ids: (7,) int array, indices into data.ctrl

    Raises a clear error (rather than silently mis-indexing) if the
    expected joint/actuator names are not found, so a differently-named
    MJCF fails loudly instead of quietly writing to the wrong joints.
    """
    qpos_adr = np.zeros(DOF, dtype=int)
    qvel_adr = np.zeros(DOF, dtype=int)
    for i, name in enumerate(ARM_JOINT_NAMES):
        try:
            j = model.joint(name)
        except KeyError as e:
            raise KeyError(
                f"Could not find joint '{name}' in the MJCF. This model's "
                f"arm joints are named differently -- edit ARM_JOINT_NAMES "
                f"in robot/franka.py to match your panda.xml.") from e
        qpos_adr[i] = j.qposadr[0]
        qvel_adr[i] = j.dofadr[0]

    actuator_ids = np.zeros(DOF, dtype=int)
    for i, name in enumerate(ARM_ACTUATOR_NAMES):
        try:
            a = model.actuator(name)
        except KeyError as e:
            raise KeyError(
                f"Could not find actuator '{name}' in the MJCF. Check your "
                f"panda.xml's <actuator> names and update ARM_ACTUATOR_NAMES "
                f"in robot/franka.py accordingly (e.g. some variants name "
                f"them 'panda_joint1_actuator' or similar).") from e
        actuator_ids[i] = a.id

    return {"qpos_adr": qpos_adr, "qvel_adr": qvel_adr, "actuator_ids": actuator_ids}


class FrankaModel:
    """
    Wraps a mujoco.MjModel/MjData pair (robot/mujoco_env.py) to expose:
      - forward kinematics for collision spheres:  fk(q) -> (centers, Jlin)
      - batched forward kinematics for MPPI rollouts: fk_batch(Q)
      - mass matrix M(q), gravity G(q), Coriolis C(q,qdot)qdot (for cbf/barrier.py)

    Time complexity: MuJoCo's mj_kinematics/mj_comPos are O(dof) per
    call (single kinematic chain); mj_fullM (mass matrix) is O(nv^2)
    over the full model, then sliced down to the 7 arm DOF.
    """

    # Approximate swept-sphere radius per link (m). Panda links taper from
    # the base (~8cm) to the wrist/hand (~5cm); these are conservative
    # (slightly oversized) placeholders -- replace with the exact radii
    # from your collision mesh / URDF <collision> geometry if available.
    LINK_RADII_DEFAULT = [0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05]

    def __init__(self, mj_model, mj_data, num_spheres_per_link: int = 2,
                 link_radii: list = None):
        self.model = mj_model
        self.data = mj_data
        self.n_per_link = num_spheres_per_link
        self.dof = DOF

        idx = resolve_arm_indices(mj_model)
        self.qpos_adr = idx["qpos_adr"]
        self.qvel_adr = idx["qvel_adr"]
        self.actuator_ids = idx["actuator_ids"]

        # Body ids for the 7 arm links + hand, resolved once at init.
        self.link_body_ids = [mj_model.body(f"link{i}").id for i in range(1, 8)]
        self.end_body_id = mj_model.body("hand").id
        radii = link_radii if link_radii is not None else self.LINK_RADII_DEFAULT
        self.sphere_radii = np.array(
            [radii[i] for i in range(len(self.link_body_ids)) for _ in range(self.n_per_link)])

    # ---- internal helpers: write q/qdot into the CORRECT qpos/qvel slots
    def _set_q(self, q: np.ndarray):
        self.data.qpos[self.qpos_adr] = q

    def _set_qdot(self, qdot: np.ndarray):
        self.data.qvel[self.qvel_adr] = qdot

    def _segment_endpoints(self, bid_from, bid_to):
        p_from = self.data.xpos[bid_from].copy()
        p_to = self.data.xpos[bid_to].copy()
        return p_from, p_to

    # ---- single-configuration FK (used by GPMP2 obstacle factor) --------
    def fk(self, q: np.ndarray):
        """
        Returns (centers: (M,3), Jlin: (M,3,dof)) -- world-frame sphere
        centers, interpolated along each link SEGMENT (link_i to the next
        link's origin, or to the hand body for the last link), and their
        linear Jacobians wrt q via MuJoCo's mj_jac, sliced down to the
        actual 7 arm DOF columns (self.qvel_adr) -- NOT the first 7
        columns of the full Jacobian, which may belong to gripper DOF
        if those come first in the model.
        """
        import mujoco
        self._set_q(q)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        next_ids = self.link_body_ids[1:] + [self.end_body_id]

        centers = []
        jacs = []
        for bid, bid_next in zip(self.link_body_ids, next_ids):
            p_from, p_to = self._segment_endpoints(bid, bid_next)
            for k in range(self.n_per_link):
                alpha = (k + 0.5) / self.n_per_link
                p = (1 - alpha) * p_from + alpha * p_to
                centers.append(p)
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, jacp, jacr, p, bid)
                jacs.append(jacp[:, self.qvel_adr])  # <-- correct columns, not [:dof]
        return np.stack(centers), np.stack(jacs)

    def num_spheres(self) -> int:
        return len(self.link_body_ids) * self.n_per_link

    # ---- batched FK for MPPI rollouts (positions only, no Jacobian) -----
    def fk_batch(self, Q: np.ndarray) -> np.ndarray:
        """
        Q : (N, T, dof)
        Returns (N, T, M, 3) sphere centers.
        """
        import mujoco
        N, T, dof = Q.shape
        M = self.num_spheres()
        out = np.zeros((N, T, M, 3))
        next_ids = self.link_body_ids[1:] + [self.end_body_id]
        for i in range(N):
            for t in range(T):
                self._set_q(Q[i, t])
                mujoco.mj_kinematics(self.model, self.data)
                idx = 0
                for bid, bid_next in zip(self.link_body_ids, next_ids):
                    p_from, p_to = self._segment_endpoints(bid, bid_next)
                    for k in range(self.n_per_link):
                        alpha = (k + 0.5) / self.n_per_link
                        out[i, t, idx] = (1 - alpha) * p_from + alpha * p_to
                        idx += 1
        return out

    # ---- dynamics terms for cbf/barrier.py --------------------------------
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        import mujoco
        self._set_q(q)
        mujoco.mj_forward(self.model, self.data)
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, M_full)
        return M_full[np.ix_(self.qvel_adr, self.qvel_adr)]

    def gravity(self, q: np.ndarray) -> np.ndarray:
        import mujoco
        self._set_q(q)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_bias[self.qvel_adr].copy()

    def coriolis_times_qdot(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        import mujoco
        self._set_q(q)
        self._set_qdot(qdot)
        mujoco.mj_forward(self.model, self.data)
        bias_arm = self.data.qfrc_bias[self.qvel_adr].copy()
        return bias_arm - self.gravity(q)

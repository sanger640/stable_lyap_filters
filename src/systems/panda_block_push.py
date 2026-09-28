"""Panda end-effector pushing one upright Jenga block to a planar goal."""
from pathlib import Path

import numpy as np


PUSH_XML = r"""
<mujoco model="panda_block_push">
  <include file="panda.xml"/>
  <statistic center="0.5 0 0.48" extent="0.65"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.16 0.16 0.16" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <body name="table" pos="0.5 0 0">
      <geom name="table_geom" size="0.24 0.34 0.025" type="box"
            rgba="0.88 0.88 0.90 1" pos="0 0 0.4" mass="1000"
            contype="1" conaffinity="1"/>
    </body>
    <body name="push_block" pos="0 0 0">
      <joint name="push_block_free" type="free"/>
      <geom name="push_block_geom" size="0.0125 0.01 0.0375" type="box"
            rgba="0.92 0.20 0.12 1" mass="0.05" friction="0.62 0.02 0.002"/>
    </body>
    <body name="goal" pos="0.635 0 0.428">
      <geom name="goal_geom" type="box" size="0.043 0.032 0.002"
            rgba="0.12 0.90 0.25 0.48" contype="0" conaffinity="0"/>
    </body>
    <camera name="cam_fixed" pos="0.94 -0.02 1.10" mode="targetbody" target="table"/>
  </worldbody>
  <keyframe>
    <key name="home"
         qpos="0 -0.93 0 -2.77 0 2.72 0.7853 0.04 0.04
               0.535 0 0.463 1 0 0 0"
         ctrl="0 -0.92 0 -2.77 0 2.72 0.7853 110"/>
  </keyframe>
</mujoco>
"""


def write_push_xml(panda_asset_directory, destination=None):
    """Write the task XML beside ``panda.xml`` so all mesh includes resolve."""
    directory = Path(panda_asset_directory)
    path = Path(destination) if destination else directory / "panda_block_push.xml"
    path.write_text(PUSH_XML)
    return path


class PandaBlockPush:
    """Deterministic operational-space Panda controller and one free Jenga block."""

    def __init__(self, xml_path, width=640, height=480, render=True):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("attachment_site").id
        self.block_id = self.model.body("push_block").id
        self.block_geom_id = self.model.geom("push_block_geom").id
        self.robot_body_ids = set(range(self.block_id)) - {0, self.model.body("table").id}
        self.steps_per_action = int(round(.1 / self.model.opt.timestep))
        self.orientation = np.zeros(4)
        self.target = np.zeros(3)
        self.gripper = 0.0
        self.renderer = (mujoco.Renderer(self.model, height=height, width=width)
                         if render else None)
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state_size = mujoco.mj_stateSize(self.model, self.state_spec)
        self.peak_tilt_deg = 0.0
        self.reset()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def reset(self, block_xy=(.535, 0.), block_yaw=0.):
        m, d, mj = self.model, self.data, self.mj
        mj.mj_resetDataKeyframe(m, d, 0)
        joint = m.joint("push_block_free")
        adr = int(joint.qposadr[0])
        d.qpos[adr:adr + 3] = [float(block_xy[0]), float(block_xy[1]), .4625]
        d.qpos[adr + 3:adr + 7] = [np.cos(block_yaw/2), 0., 0., np.sin(block_yaw/2)]
        d.qvel[:] = 0.; d.ctrl[:] = 0.
        mj.mj_forward(m, d)
        self.target = d.site_xpos[self.site].copy()
        mj.mju_mat2Quat(self.orientation, d.site_xmat[self.site])
        self.gripper = 0.0
        self.peak_tilt_deg = 0.0
        # Let the block and robot settle while preserving their initial targets.
        for _ in range(10 * self.steps_per_action):
            self._control(); mj.mj_step(m, d)
            self.peak_tilt_deg = max(self.peak_tilt_deg, self.block_tilt_deg())

    def snapshot(self):
        state = np.empty(self.state_size)
        self.mj.mj_getState(self.model, self.data, state, self.state_spec)
        return (state, self.data.ctrl.copy(), self.target.copy(), float(self.gripper),
                float(self.peak_tilt_deg))

    def restore(self, snapshot):
        state, ctrl, target, gripper, peak_tilt = snapshot
        self.mj.mj_setState(self.model, self.data, state, self.state_spec)
        self.data.ctrl[:] = ctrl; self.target = target.copy(); self.gripper = gripper
        self.peak_tilt_deg = float(peak_tilt)
        self.mj.mj_forward(self.model, self.data)

    def _control(self):
        m, d, mj = self.model, self.data, self.mj
        current_matrix = d.site_xmat[self.site].reshape(3, 3)
        target_matrix = np.zeros(9); mj.mju_quat2Mat(target_matrix, self.orientation)
        error = target_matrix.reshape(3, 3) @ current_matrix.T
        dr = .5 * np.array([error[2, 1]-error[1, 2], error[0, 2]-error[2, 0],
                            error[1, 0]-error[0, 1]])
        jac = np.zeros((6, m.nv)); mj.mj_jacSite(m, d, jac[:3], jac[3:], self.site)
        velocity = jac @ d.qvel
        wrench = np.hstack([500*(self.target-d.site_xpos[self.site])-30*velocity[:3],
                            20*dr-.5*velocity[3:]])
        d.ctrl[:7] = (jac.T @ wrench - .1*d.qvel)[:7]
        d.ctrl[7] = self.gripper

    def step(self, action):
        action = np.asarray(action, float)
        if action.shape != (4,):
            raise ValueError("action must be absolute EE xyz plus gripper command")
        self.target = action[:3].copy()
        self.gripper = 0.0 if action[3] > .9 else 110.0 if action[3] < -.9 else self.gripper
        for _ in range(self.steps_per_action):
            self._control(); self.mj.mj_step(self.model, self.data)
            self.peak_tilt_deg = max(self.peak_tilt_deg, self.block_tilt_deg())

    def proprio(self):
        closed = 1.0 if self.gripper < 50 else -1.0
        return np.r_[self.data.site_xpos[self.site], closed]

    def block_pose(self):
        return self.data.body("push_block").xpos.copy(), self.data.body("push_block").xquat.copy()

    def block_tilt_deg(self):
        rotation = self.data.body("push_block").xmat.reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1., 1.))))

    def robot_contact(self):
        for contact in self.data.contact[:self.data.ncon]:
            bodies = {int(self.model.geom_bodyid[contact.geom1]),
                      int(self.model.geom_bodyid[contact.geom2])}
            if self.block_id in bodies and any(body in self.robot_body_ids for body in bodies):
                return True
        return False

    def outcome(self, goal=(.635, 0.)):
        position, quaternion = self.block_pose()
        distance = float(np.linalg.norm(position[:2] - np.asarray(goal)))
        tilt = self.block_tilt_deg()
        toppled = bool(self.peak_tilt_deg >= 45.)
        supported = bool(position[2] > .41 and .26 < position[0] < .74 and abs(position[1]) < .34)
        return {"success": bool(distance <= .025 and tilt <= 10. and supported and not toppled),
                "failure": toppled,
                "supported": supported, "goal_distance_m": distance,
                "tilt_deg": tilt, "peak_tilt_deg": float(self.peak_tilt_deg),
                "block_position_m": position.tolist()}

    def render(self):
        if self.renderer is None:
            raise RuntimeError("rendering was disabled for this simulator")
        self.renderer.update_scene(self.data, camera="cam_fixed")
        return self.renderer.render().copy()

"""Minimal MuJoCo mechanism benchmarks for cross-task regime monitoring.

These are deliberately not Jenga scenes and do not include a full robot arm. The commanded tool
pose is the action, isolating the contact mechanisms the universal monitor must preserve:
stick/slide/edge-fall during pushing and free/contact/insert/jam during insertion.
"""
from dataclasses import dataclass

import numpy as np


PUSHING_XML = r"""
<mujoco model="planar pushing benchmark">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>
  <default>
    <geom condim="4" friction="0.75 0.02 0.002" solref="0.01 1"/>
    <joint damping="3"/>
  </default>
  <worldbody>
    <light pos="0 -0.4 1.2" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="overview" pos="0 -0.78 0.62" xyaxes="1 0 0 0 0.62 0.78"/>
    <geom name="floor" type="plane" pos="0 0 -0.16" size="2 2 .1" rgba=".12 .14 .18 1"/>
    <geom name="table" type="box" pos="0 0 0" size=".42 .27 .05"
          rgba=".42 .31 .22 1" friction="0.32 0.01 0.001"/>
    <geom name="goal" type="box" pos=".275 0 .052" size=".055 .085 .002"
          rgba=".15 .85 .30 .45" contype="0" conaffinity="0"/>
    <body name="object" pos="-.08 0 .085">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size=".034 .034 .034" mass=".16"
            rgba=".92 .24 .16 1" friction="0.40 0.01 0.001"/>
    </body>
    <body name="pusher" pos="-.27 0 .105">
      <joint name="push_x" type="slide" axis="1 0 0" range="0 .78"/>
      <joint name="push_y" type="slide" axis="0 1 0" range="-.18 .18"/>
      <geom type="box" size=".019 .075 .065" mass="1" rgba=".18 .42 .95 1"
            friction="1.1 0.02 0.002"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pusher_x" joint="push_x" kp="4500" kv="120" ctrlrange="0 .78"/>
    <position name="pusher_y" joint="push_y" kp="4500" kv="120" ctrlrange="-.18 .18"/>
  </actuator>
</mujoco>
"""


INSERTION_XML = r"""
<mujoco model="peg insertion benchmark">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>
  <default>
    <geom condim="4" friction="0.8 0.03 0.003" solref="0.006 1"/>
    <joint damping="4"/>
  </default>
  <worldbody>
    <light pos="-.2 -.4 1.1" dir=".2 .25 -1" diffuse=".9 .9 .9"/>
    <camera name="overview" pos=".42 -.62 .38" xyaxes=".83 .56 0 -.25 .37 .89"/>
    <geom name="floor" type="plane" pos="0 0 -.01" size="1 1 .1" rgba=".12 .14 .18 1"/>
    <geom name="socket_base" type="box" pos="0 0 .012" size=".16 .16 .012"
          rgba=".30 .33 .38 1"/>
    <geom name="socket_left" type="box" pos="-.087 0 .055" size=".062 .13 .043"
          rgba=".62 .66 .72 1"/>
    <geom name="socket_right" type="box" pos=".087 0 .055" size=".062 .13 .043"
          rgba=".62 .66 .72 1"/>
    <geom name="socket_front" type="box" pos="0 -.087 .055" size=".025 .062 .043"
          rgba=".62 .66 .72 1"/>
    <geom name="socket_back" type="box" pos="0 .087 .055" size=".025 .062 .043"
          rgba=".62 .66 .72 1"/>
    <geom name="target" type="box" pos="0 0 .018" size=".024 .024 .003"
          rgba=".15 .85 .30 .5" contype="0" conaffinity="0"/>
    <body name="peg" pos="0 0 .205">
      <joint name="peg_x" type="slide" axis="1 0 0" range="-.05 .05"/>
      <joint name="peg_y" type="slide" axis="0 1 0" range="-.05 .05"/>
      <joint name="peg_z" type="slide" axis="0 0 1" range="-.16 .02"/>
      <joint name="peg_yaw" type="hinge" axis="0 0 1" range="-.35 .35"/>
      <geom name="peg_geom" type="box" pos="0 0 0" size=".020 .020 .060" mass=".35"
            rgba=".92 .56 .12 1" friction=".65 .02 .002"/>
      <geom name="tool" type="cylinder" pos="0 0 .09" size=".030 .032" mass=".3"
            rgba=".18 .42 .95 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="peg_x_ctrl" joint="peg_x" kp="850" kv="65" ctrlrange="-.05 .05"/>
    <position name="peg_y_ctrl" joint="peg_y" kp="850" kv="65" ctrlrange="-.05 .05"/>
    <position name="peg_z_ctrl" joint="peg_z" kp="1100" kv="80" ctrlrange="-.16 .02"/>
    <position name="peg_yaw_ctrl" joint="peg_yaw" kp="180" kv="12" ctrlrange="-.35 .35"/>
  </actuator>
</mujoco>
"""


@dataclass(frozen=True)
class MechanismOutcome:
    success: bool
    failure: bool
    mode: str
    values: dict


class ContactBenchmark:
    """Small deterministic controlled-tool MuJoCo environment."""

    xml = None
    camera = "overview"

    def __init__(self, width=640, height=480):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.steps_per_action = int(round(.1 / self.model.opt.timestep))
        # Integration state includes solver warm-start. Counterfactual branches must not influence
        # the subsequently executed nominal trajectory through hidden simulator state.
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state_size = mujoco.mj_stateSize(self.model, self.state_spec)

    def close(self):
        self.renderer.close()

    def reset(self):
        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)

    def snapshot(self):
        state = np.empty(self.state_size)
        self.mj.mj_getState(self.model, self.data, state, self.state_spec)
        return state

    def restore(self, state):
        self.mj.mj_setState(self.model, self.data, np.asarray(state, float), self.state_spec)
        self.mj.mj_forward(self.model, self.data)

    def step(self, control):
        self.data.ctrl[:] = np.asarray(control, float)
        for _ in range(self.steps_per_action):
            self.mj.mj_step(self.model, self.data)

    def render(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()


class PushingBenchmark(ContactBenchmark):
    """Push a free cube into a goal without sending it over the table edge."""

    xml = PUSHING_XML

    def reset(self, object_y=0.0):
        super().reset()
        joint = self.model.joint("object_free")
        adr = int(joint.qposadr[0])
        self.data.qpos[adr:adr + 3] = [-.08, float(object_y), .085]
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        self.data.qpos[int(self.model.joint("push_x").qposadr[0])] = 0
        self.data.qpos[int(self.model.joint("push_y").qposadr[0])] = 0
        self.data.ctrl[:] = [0, 0]
        self.mj.mj_forward(self.model, self.data)

    def object_position(self):
        return self.data.body("object").xpos.copy()

    def outcome(self):
        position = self.object_position()
        in_goal = .22 <= position[0] <= .34 and abs(position[1]) <= .085 and position[2] > .045
        fallen = position[2] < .02 or abs(position[0]) > .45 or abs(position[1]) > .31
        mode = "edge_fall" if fallen else "goal_slide" if in_goal else "slide_or_deflect"
        return MechanismOutcome(bool(in_goal), bool(fallen), mode,
                                {"object_position_m": position.tolist()})


class InsertionBenchmark(ContactBenchmark):
    """Insert a tight rectangular peg; lateral/yaw error creates rim contact and jamming."""

    xml = INSERTION_XML

    def reset(self):
        super().reset()
        self.data.ctrl[:] = [0, 0, 0, 0]
        self.mj.mj_forward(self.model, self.data)

    def peg_position(self):
        return self.data.body("peg").xpos.copy()

    def outcome(self):
        position = self.peg_position()
        inserted = position[2] < .105 and abs(position[0]) < .005 and abs(position[1]) < .005
        commanded_down = self.data.ctrl[2] < -.10
        jammed = commanded_down and position[2] > .13
        mode = "inserted" if inserted else "rim_jam" if jammed else "free_or_contact"
        return MechanismOutcome(bool(inserted), bool(jammed), mode,
                                {"peg_position_m": position.tolist(),
                                 "command": self.data.ctrl.copy().tolist()})

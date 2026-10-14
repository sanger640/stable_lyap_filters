"""Phase 2 procedural scenes: rows (F1), stacks (F2), leaning (F3, held out), scripted pick/push, harm label.

Frozen protocol: results/jenga/phase2_generality_protocol.json (amendment 1). Same Panda, table, controller, 0.1 s
control step and close camera as the three-block task; the three named blocks are replaced by N free objects
obj0..objN-1 (obj0 = target). `pilot` runs the amendment-1 pilot (visibility, settling, fork rate) on pilot seeds
that are never reused.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

TABLE_TOP = 0.425
TABLE_X, TABLE_Y = (0.3, 0.7), (-0.4, 0.4)
CENTER = np.array([0.5, 0.0])
PALETTE = [(0.9, 0.1, 0.1), (0.8, 0.6, 0.4), (0.2, 0.5, 0.9), (0.2, 0.7, 0.3),
           (0.95, 0.8, 0.2), (0.6, 0.3, 0.7), (0.95, 0.5, 0.1), (0.3, 0.3, 0.3)]
HOME_QPOS = "0 -0.93 0 -2.77 0 2.72 0.7853 0.04 0.04"
HOME_CTRL = "0 -0.92 0 -2.77 0 2.72 0.7853 110"
TARGET_MOVED_EVENT = True    # T3 sets False (the pushed block's displacement is the task, not an event)
AT_REST_INCLUDE_TARGET = False   # T3 sets True
VALID_PERTURBATION = True    # data correction 2026-10-13 (see Phase2Sim.perturb); False reproduces older data
HOLD_CHECK_STEPS = 38        # stability check horizon (control steps) for valid perturbations
REST_MOVE_MM, REST_TILT_DEG = 10.0, 10.0   # a chunk start is 'at rest' if the unperturbed scene, robot holding, stays within these
PILOT = {"seed_base": 990_000, "scenes": 30, "futures": 8}
RANGES = {   # amendment 1; the pilot may adjust only the action ranges ("action") and family sizes
    "box_half": ((0.008, 0.016), (0.008, 0.014), (0.020, 0.045)),
    "cyl": ((0.008, 0.014), (0.020, 0.045)),
    "slab_half": ((0.025, 0.040), (0.012, 0.018), (0.005, 0.008)),
    "mass": (0.025, 0.120), "friction": (0.5, 1.0),
    "action": {"offset_mm": 12.0, "push_yaw_deg": 25.0, "depth_mm": 8.0, "approach_mm": (60.0, 120.0),
               "step_mm": (5.0, 12.0), "push_mm": (30.0, 80.0), "prepush_mm": (20.0, 40.0)},
    "action_F2": {"offset_mm": 5.0, "push_mm": (30.0, 45.0)},   # amendment 2 pilot adjustment, last step (35%, then 32% > 30%)
    "F1": {"n": (2, 5), "gap_mm": (0.0, 15.0), "row_yaw_deg": 10.0, "max_row_mm": 150.0},
    "F2": {"stacks": (1, 2), "height": (2, 3), "offset_mm": 8.0, "neighbours": (0, 2), "neighbour_gap_mm": (0.0, 20.0)},
    "F3": {"lean_deg": (10.0, 30.0), "neighbours": (0, 2), "wall_prob": 0.3},
}


# ------------------------------------------------------------------------------------------- scene sampling
def _quat_z(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _object(rng, shape):
    if shape == "box":
        size = [rng.uniform(*r) for r in RANGES["box_half"]]
    elif shape == "cylinder":
        size = [rng.uniform(*r) for r in RANGES["cyl"]]
    else:
        size = [rng.uniform(*r) for r in RANGES["slab_half"]]
    return {"shape": shape, "size": size, "mass": float(rng.uniform(*RANGES["mass"])),
            "friction": float(rng.uniform(*RANGES["friction"])), "rgba": [*PALETTE[rng.integers(len(PALETTE))], 1.0]}


def half_extent(o):
    """(half x, half y, half z) of the object's axis-aligned box when upright and unrotated."""
    if o["shape"] == "cylinder":
        return np.array([o["size"][0], o["size"][0], o["size"][1]])
    return np.array(o["size"])


def _row(rng, n, origin, yaw, gap, shapes=("box", "cylinder")):
    """n upright objects in a row along the rotated y axis, centred on origin."""
    objs = [_object(rng, shapes[rng.integers(len(shapes))]) for _ in range(n)]
    widths = [2 * half_extent(o)[1] for o in objs]
    gaps = [rng.uniform(*gap) / 1000 for _ in range(n - 1)]
    total = sum(widths) + sum(gaps)
    y = -total / 2
    axis = np.array([-np.sin(yaw), np.cos(yaw)])
    for k, o in enumerate(objs):
        y += widths[k] / 2
        xy = origin + axis * y + rng.uniform(-0.002, 0.002, 2)
        o["pos"] = [float(xy[0]), float(xy[1]), TABLE_TOP + half_extent(o)[2] + 0.0005]
        o["quat"] = _quat_z(yaw + np.radians(rng.uniform(-3, 3))).tolist()
        y += widths[k] / 2 + (gaps[k] if k < n - 1 else 0)
    return objs, total


def sample_scene(family, seed):
    """-> scene dict (objects with obj0 = target, optional static wall, primitive). Deterministic in seed."""
    if family == "T2":                                      # task T2 (push past tall objects), eval/t2_push.py
        import t2_push
        return t2_push.sample_t2(seed)
    if family == "T3":                                      # task T3 (push a Jenga block into a zone), eval/t3_push.py
        import t3_push
        return t3_push.sample_t3(seed)
    rng = np.random.default_rng([20261200, {"F1": 1, "F2": 2, "F3": 3}[family], seed])
    wall = None
    if family == "F1":
        r = RANGES["F1"]
        while True:
            n = int(rng.integers(r["n"][0], r["n"][1] + 1))
            yaw = np.radians(rng.uniform(-r["row_yaw_deg"], r["row_yaw_deg"]))
            objs, total = _row(rng, n, CENTER + rng.uniform(-0.01, 0.01, 2), yaw, r["gap_mm"])
            if total * 1000 <= r["max_row_mm"]:
                break
        t = int(rng.integers(n))
        primitive = "pick"
    elif family == "F2":
        r = RANGES["F2"]
        objs, tops, bases = [], [], []
        n_stacks = int(rng.integers(r["stacks"][0], r["stacks"][1] + 1))
        xs = [0.0] if n_stacks == 1 else [-0.03, 0.03]
        for sx in xs:
            base_xy = CENTER + np.array([0.0, sx]) + rng.uniform(-0.008, 0.008, 2)
            z, prev = TABLE_TOP, None
            for level in range(int(rng.integers(r["height"][0], r["height"][1] + 1))):
                shape = ["box", "slab"][rng.integers(2)] if level == 0 else ["box", "slab", "cylinder"][rng.integers(3)]
                o = _object(rng, shape)
                if shape == "box" and level > 0:            # lie boxes flat on upper levels so stacks are not towers
                    o["size"] = sorted(o["size"], reverse=True)
                he = half_extent(o)
                off = rng.uniform(-r["offset_mm"], r["offset_mm"], 2) / 1000 if level > 0 else np.zeros(2)
                o["pos"] = [float(base_xy[0] + off[0]), float(base_xy[1] + off[1]), z + he[2] + 0.0005]
                o["quat"] = _quat_z(np.radians(rng.uniform(-8, 8))).tolist()
                z += 2 * he[2] + 0.0005
                objs.append(o); prev = o
            tops.append(len(objs) - 1); bases.append(len(objs) - 1 - level)
        for _ in range(int(rng.integers(r["neighbours"][0], r["neighbours"][1] + 1))):
            o = _object(rng, ["box", "cylinder"][rng.integers(2)])
            side = rng.choice([-1, 1])
            edge = max(abs(p["pos"][1] - CENTER[1]) + half_extent(p)[1] for p in objs)
            y = CENTER[1] + side * (edge + rng.uniform(*r["neighbour_gap_mm"]) / 1000 + half_extent(o)[1])
            o["pos"] = [float(CENTER[0] + rng.uniform(-0.01, 0.01)), float(y), TABLE_TOP + half_extent(o)[2] + 0.0005]
            o["quat"] = _quat_z(np.radians(rng.uniform(-5, 5))).tolist()
            objs.append(o)
        primitive = "pick" if rng.random() < 0.5 else "push"
        t = int(rng.choice(tops)) if primitive == "pick" else int(rng.integers(len(objs)))
    else:
        r = RANGES["F3"]
        lean = np.radians(rng.uniform(*r["lean_deg"]))
        leaner = _object(rng, "box")
        lh = half_extent(leaner)
        use_wall = rng.random() < r["wall_prob"]
        side = rng.choice([-1, 1])                          # support is on this side (y) of the leaner
        objs = []
        if use_wall:
            support_face = CENTER[1] + side * 0.02
            wall = {"pos": [float(CENTER[0]), float(support_face + side * 0.005), TABLE_TOP + 0.03],
                    "size": [0.05, 0.005, 0.03]}
        else:
            support = _object(rng, "box")
            support["size"][2] = max(support["size"][2], lh[2] * np.cos(lean) + 0.005)
            sh = half_extent(support)
            sy = CENTER[1] + side * 0.02
            support["pos"] = [float(CENTER[0]), float(sy), TABLE_TOP + sh[2] + 0.0005]
            support["quat"] = [1.0, 0.0, 0.0, 0.0]
            support_face = sy - side * sh[1]
            objs.append(support)
        # leaner: rotated about x by `lean` toward the support; its upper corner touches the support face
        tilt = -side * lean
        # exact placement: centre = pivot + R(tilt) @ (0, 0, h) shifted so the far top corner meets the face
        R = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
        corners = np.array([[0, sy_, sz_] for sy_ in (-lh[1], lh[1]) for sz_ in (-lh[2], lh[2])]) @ R.T
        lowest = corners[:, 2].min()
        reach = (corners[:, 1] * side).max()
        leaner["pos"] = [float(CENTER[0]), float(support_face - side * (reach + 0.0005)), float(TABLE_TOP - lowest + 0.0005)]
        leaner["quat"] = [float(np.cos(tilt / 2)), float(np.sin(tilt / 2)), 0.0, 0.0]
        objs.insert(0, leaner)
        far = min(o["pos"][1] - side * half_extent(o)[1] for o in objs) if side > 0 else \
            max(o["pos"][1] - side * half_extent(o)[1] for o in objs)
        for _ in range(int(rng.integers(r["neighbours"][0], r["neighbours"][1] + 1))):
            o = _object(rng, ["box", "cylinder"][rng.integers(2)])
            y = far - side * (rng.uniform(0.0, 0.015) + half_extent(o)[1] + 0.002)
            o["pos"] = [float(CENTER[0] + rng.uniform(-0.01, 0.01)), float(y), TABLE_TOP + half_extent(o)[2] + 0.0005]
            o["quat"] = _quat_z(np.radians(rng.uniform(-5, 5))).tolist()
            far = y - side * half_extent(o)[1]
            objs.append(o)
        primitive = "pick" if rng.random() < 0.5 else "push"
        t = int(rng.integers(len(objs)))
    objs.insert(0, objs.pop(t))                             # obj0 = target
    return {"family": family, "seed": int(seed), "objects": objs, "wall": wall, "primitive": primitive,
            "action_seed": int(rng.integers(2 ** 31))}


# ------------------------------------------------------------------------------------------- MJCF + sim
def scene_xml(scene):
    geoms = []
    for k, o in enumerate(scene["objects"]):
        size = " ".join(f"{v:.5f}" for v in o["size"])
        typ = "cylinder" if o["shape"] == "cylinder" else "box"
        rgba = " ".join(f"{v:.3f}" for v in o["rgba"])
        geoms.append(f'''    <body name="obj{k}" pos="0 0 0">
      <joint type="free"/>
      <geom name="obj{k}_g" size="{size}" type="{typ}" rgba="{rgba}" mass="{o['mass']:.5f}" friction="{o['friction']:.3f} 0.005 0.0001"/>
    </body>''')
    if scene.get("zone"):                                   # T3 goal zone: visual only (no collision), flush with the table
        z = scene["zone"]
        geoms.append(f'''    <geom name="zone" type="cylinder" size="{z['radius']:.5f} 0.0003" pos="{z['pos'][0]:.5f} {z['pos'][1]:.5f} {TABLE_TOP + 0.0003:.5f}" rgba="0.2 0.75 0.3 0.55" contype="0" conaffinity="0"/>''')
    if scene["wall"]:
        w = scene["wall"]
        geoms.append(f'''    <body name="wall" pos="{w['pos'][0]:.5f} {w['pos'][1]:.5f} {w['pos'][2]:.5f}">
      <geom size="{' '.join(f'{v:.5f}' for v in w['size'])}" type="box" rgba="0.55 0.55 0.6 1"/>
    </body>''')
    table_friction = f' friction="{scene["table_friction"]:.3f} 0.005 0.0001"' if "table_friction" in scene else ""   # T3
    qpos = HOME_QPOS + "".join(" " + " ".join(f"{v:.6f}" for v in [*o["pos"], *o["quat"]]) for o in scene["objects"])
    return f'''<mujoco model="phase2_{scene['family']}_{scene['seed']}">
  <include file="panda.xml"/>
  <statistic center="0.3 0 0.4" extent="0.8"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <body name="table" pos="0.5 0 0">
      <geom size="0.2 0.4 0.025" type="box" rgba="0.9 0.9 0.9 1" pos="0 0 0.4" mass="1000" contype="1" conaffinity="1"{table_friction}/>
    </body>
{chr(10).join(geoms)}
    <camera name="cam_fixed" pos="0.9 0 1.2" mode="targetbody" target="table"/>
  </worldbody>
  <keyframe>
    <key name="home" qpos="{qpos}" ctrl="{HOME_CTRL}"/>
  </keyframe>
</mujoco>
'''


def make_sim(scene):
    """Phase2Sim for one scene (XML written next to the extracted panda.xml, removed after loading)."""
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    base = Path(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="p2_")))
    path = base.parent / f"p2_{scene['family']}_{scene['seed']}_{os.getpid()}.xml"
    path.write_text(scene_xml(scene))
    try:
        sim = Phase2Sim(path, len(scene["objects"]))
    finally:
        path.unlink(missing_ok=True)
    return sim


def _base():
    from jenga_short_held_tails import DirectJengaSim
    return DirectJengaSim


class Phase2Sim(_base()):
    """DirectJengaSim with N generic objects: obj0 is the target; block_ids = the non-target objects."""

    def __init__(self, xml, n):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("attachment_site").id
        self.tracked_block_ids = [self.model.body(f"obj{k}").id for k in range(n)]
        self.block_ids = self.tracked_block_ids[1:]
        self.table_id = self.model.body("table").id
        self.steps_per_action = int(round(0.1 / self.model.opt.timestep))
        self.orientation, self.target, self.gripper = np.zeros(4), np.zeros(3), 110.0
        self.renderer = self.segment_renderer = None
        self.state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.state_size = mujoco.mj_stateSize(self.model, self.state_spec)
        self.qadr = [self.model.jnt_qposadr[self.model.body_jntadr[b]] for b in self.tracked_block_ids]

    def close(self):
        pass

    def reset(self, seed=None):
        m, d, mj = self.model, self.data, self.mj
        mj.mj_resetDataKeyframe(m, d, 0)
        mj.mj_forward(m, d)
        self.target = d.site_xpos[self.site].copy()
        mj.mju_mat2Quat(self.orientation, d.site_xmat[self.site])
        self.gripper = 110.0

    def max_tilt(self):
        return 0.0                                   # not used in Phase 2 (harm comes from object_state traces)

    def object_state(self):
        """(N, 7) world position + quaternion of every object."""
        return np.concatenate([self.data.xpos[self.tracked_block_ids], self.data.xquat[self.tracked_block_ids]], 1)

    def perturb(self, position, rotation):
        """Perception-error pose noise on every object. With VALID_PERTURBATION (default, data correction
        2026-10-13) the noise is planar (x, y, yaw) and is shrunk (x1, 0.75, 0.5, 0.25, 0) until the perturbed scene
        is physically valid and at rest like the observed one: no new overlap beyond 0.5 mm and, over the full 3.8 s
        horizon with the robot holding its target, no object moving more than 5 mm (or tilting more than 5 deg) beyond
        what the unperturbed scene does. Without the flag: the
        original full 6-D noise (reproduces data generated before 2026-10-13)."""
        if not VALID_PERTURBATION:
            return self._apply(position, rotation, planar=False)
        base = self.snapshot()
        key = hash(base[0].tobytes())                         # the unperturbed scene is shared by all its futures
        if getattr(self, "_base_key", None) != key:
            pen0 = self._penetration()
            move0, tilt0 = self._hold_motion(HOLD_CHECK_STEPS)
            self._base_key, self._base_check = key, (pen0, move0, tilt0)
        pen0, move0, tilt0 = self._base_check
        self.restore(base)
        for scale in (1.0, 0.75, 0.5, 0.25, 0.0):
            self.restore(base)
            if scale > 0:
                self._apply(position * scale, rotation * scale, planar=True)
            if scale == 0.0:
                self.last_perturb_scale = 0.0
                return
            if self._penetration() > pen0 + 0.0005:
                continue
            after = self.snapshot()
            move, tilt = self._hold_motion(HOLD_CHECK_STEPS, limit=(move0 + 0.005, tilt0 + 5.0))
            self.restore(after)
            if np.all(move <= move0 + 0.005) and np.all(tilt <= tilt0 + 5.0):
                self.last_perturb_scale = scale
                return

    def _apply(self, position, rotation, planar):
        from scipy.spatial.transform import Rotation
        d = self.data
        for k, adr in enumerate(self.qadr):
            dp = np.array(position[k], float)
            rv = np.array(rotation[k], float)
            if planar:
                dp[2] = 0.0; rv = np.array([0.0, 0.0, rv[2]])
            d.qpos[adr:adr + 3] += dp
            w, x, y, z = d.qpos[adr + 3:adr + 7]
            q = (Rotation.from_rotvec(rv) * Rotation.from_quat([x, y, z, w])).as_quat()
            d.qpos[adr + 3:adr + 7] = [q[3], q[0], q[1], q[2]]
        self.mj.mj_forward(self.model, d)

    def _penetration(self):
        """Deepest overlap (m) of any contact involving an object."""
        ids = set(self.tracked_block_ids)
        deep = 0.0
        for c in self.data.contact[:self.data.ncon]:
            if int(self.model.geom_bodyid[c.geom1]) in ids or int(self.model.geom_bodyid[c.geom2]) in ids:
                deep = max(deep, -float(c.dist))
        return deep

    def _hold_motion(self, steps=5, limit=None):
        """Per-object displacement (m) and tilt change (deg) over `steps` control steps holding the robot target;
        with `limit` (per-object move, tilt bounds) it stops early once an object is 20 mm / 20 deg beyond a bound (an
        object that far gone does not come back, so the draw is rejected either way)."""
        s0 = self.object_state()
        hold = np.r_[self.target, 0.0]
        for i in range(steps):
            self.execute(hold)
            if limit is not None and i % 5 == 4:
                s1 = self.object_state()
                m = np.linalg.norm(s1[:, :3] - s0[:, :3], axis=1); t = np.abs(tilt_deg(s1[:, 3:]) - tilt_deg(s0[:, 3:]))
                if np.any(m > limit[0] + 0.02) or np.any(t > limit[1] + 20.0):   # far beyond: will not come back
                    return m, t
        s1 = self.object_state()
        return (np.linalg.norm(s1[:, :3] - s0[:, :3], axis=1),
                np.abs(tilt_deg(s1[:, 3:]) - tilt_deg(s0[:, 3:])))


def at_rest(sim):
    """True if, with the robot holding its target for the full horizon, no NON-TARGET object of the UNPERTURBED scene moves more
    than REST_MOVE_MM or tilts more than REST_TILT_DEG (otherwise motion is ongoing and forks cannot be attributed to
    the next action). Leaves the simulator state unchanged."""
    snap = sim.snapshot(); warm = sim.data.qacc_warmstart.copy()
    move, tilt = sim._hold_motion(HOLD_CHECK_STEPS)
    sim.restore(snap); sim.data.qacc_warmstart[:] = warm
    k0 = 0 if AT_REST_INCLUDE_TARGET else 1                # non-target objects (T3, a single pushed block: all)
    return bool(np.all(move[k0:] * 1000 <= REST_MOVE_MM) and np.all(tilt[k0:] <= REST_TILT_DEG))


def settle(sim, family):
    """1.0 s at home. -> (ok, max displacement mm, max tilt change deg)."""
    s0 = sim.object_state()
    hold = np.r_[sim.target, 0.0]
    for _ in range(10):
        sim.execute(hold)
    s1 = sim.object_state()
    disp = float(np.linalg.norm(s1[:, :3] - s0[:, :3], axis=1).max() * 1000)
    dtilt = float(np.abs(tilt_deg(s1[:, 3:]) - tilt_deg(s0[:, 3:])).max())
    ok = disp <= 5.0 if family == "F3" else (disp <= 5.0 and dtilt <= 5.0)
    return ok, disp, dtilt


def tilt_deg(quats):
    """Angle between each body's z axis and world z (deg); quats (..., 4) wxyz."""
    w, x, y, z = np.moveaxis(np.asarray(quats), -1, 0)
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


# ------------------------------------------------------------------------------------------- visibility
def supporting(sim):
    """Objects that support another (amendment 2): in contact with another object whose centre is higher."""
    ids = {b: k for k, b in enumerate(sim.tracked_block_ids)}
    out = np.zeros(len(ids), bool)
    for c in sim.data.contact[:sim.data.ncon]:
        a, b = int(sim.model.geom_bodyid[c.geom1]), int(sim.model.geom_bodyid[c.geom2])
        if a in ids and b in ids:
            lower = a if sim.data.xpos[a][2] < sim.data.xpos[b][2] else b
            out[ids[lower]] = True
    return out


def visible_ok(sim, vis, alone):
    """Amendment 2: >= 60% visible, or >= 10% for an object that supports another (stacks)."""
    need = np.where(supporting(sim), 0.10, 0.60)
    return bool(np.all(np.asarray(vis) >= need) and min(alone) > 0)


def visibility(sim):
    """Fraction of each object's unoccluded pixels that are visible in the close-camera true-start render."""
    import mujoco
    from jenga_render_frames import close_camera
    r = mujoco.Renderer(sim.model, height=384, width=384)
    r.enable_segmentation_rendering()
    cam = close_camera(mujoco)
    geom_of = [int(np.flatnonzero(sim.model.geom_bodyid == b)[0]) for b in sim.tracked_block_ids]

    def counts():
        r.update_scene(sim.data, camera=cam)
        seg = r.render()
        is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        return [int((is_geom & (seg[..., 0] == g)).sum()) for g in geom_of]
    full = counts()
    snap = sim.snapshot()
    alone = []
    for k in range(len(geom_of)):
        for j, adr in enumerate(sim.qadr):            # move every other object far below the floor
            if j != k:
                sim.data.qpos[adr + 2] = -5.0 - j
        mujoco.mj_forward(sim.model, sim.data)
        alone.append(counts()[k])
        sim.restore(snap)
    r.close()
    return [f / a if a else 0.0 for f, a in zip(full, alone)], alone


# ------------------------------------------------------------------------------------------- primitives
def primitive_actions(sim, scene):
    """4-D commands (xyz, gripper +1 close / -1 open / 0 hold) for the scene's PICK or PUSH, randomised."""
    a = dict(RANGES["action"], **RANGES.get(f"action_{scene['family']}", {}), **scene.get("action_override", {}))
    rng = np.random.default_rng(scene["action_seed"])
    tgt = scene["objects"][0]
    pos = sim.object_state()[0, :3]
    top = pos[2] + tilted_half_height(sim, 0)
    home = sim.data.site_xpos[sim.site].copy()
    step = rng.uniform(*a["step_mm"]) / 1000
    off = rng.uniform(-a["offset_mm"], a["offset_mm"], 2) / 1000
    cmds = [np.r_[home, -1.0]]

    def go(p, grip=0.0):
        cur = cmds[-1][:3]
        n = max(1, int(np.ceil(np.linalg.norm(p - cur) / step)))
        for i in range(1, n + 1):
            cmds.append(np.r_[cur + (p - cur) * i / n, grip])
    if scene["primitive"] == "pick":
        above = np.r_[pos[:2] + off, top + rng.uniform(*a["approach_mm"]) / 1000]
        grasp_z = top - 0.012 + rng.uniform(-a["depth_mm"], a["depth_mm"]) / 1000
        grasp_z = max(grasp_z, TABLE_TOP + 0.008)
        go(above); go(np.r_[above[:2], grasp_z])
        for _ in range(5):
            cmds.append(np.r_[above[:2], grasp_z, 1.0])
        go(np.r_[above[:2], grasp_z + 0.08], 0.0)
    else:
        yaw = np.radians(rng.uniform(-a["push_yaw_deg"], a["push_yaw_deg"])) + rng.choice([0.0, np.pi / 2, -np.pi / 2])
        if "push_yaw" in scene:                                 # T2 task: fixed nominal direction (random draws above kept)
            yaw = scene["push_yaw"] + np.radians(rng.uniform(-scene["push_jitter_deg"], scene["push_jitter_deg"]))
        u = np.array([np.cos(yaw), np.sin(yaw)])               # push direction (0 = away from the robot, +x)
        reach = float(np.abs(half_extent(tgt)[:2]).max())
        if "push_reach_m" in scene:                             # T3: contact face distance along the push direction
            reach = scene["push_reach_m"]
        if "push_site_offset_xy" in scene:                      # T3: centre the leading finger (not the site) on the block
            off = off + np.asarray(scene["push_site_offset_xy"])
        pre = pos[:2] + off - u * (reach + rng.uniform(*a["prepush_mm"]) / 1000)
        z = TABLE_TOP + min(0.015, 0.8 * (top - TABLE_TOP))
        if "push_height_mm" in scene:                           # T3: fingertip height above the table (tip vs slide)
            z = TABLE_TOP + scene["push_height_mm"] / 1000
        cmds.append(np.r_[home, 1.0])                          # close the gripper to push with the fingertips
        go(np.r_[pre, top + 0.05], 0.0); go(np.r_[pre, z])
        travel = rng.uniform(*a["push_mm"]) / 1000
        if "zone" in scene:                                     # T3: push so the block centre would end at the zone centre
            travel = float(np.dot(np.asarray(scene["zone"]["pos"][:2]) - pos[:2], u)) + scene.get("push_overshoot_m", 0.0)
        go(np.r_[pos[:2] + off + u * (travel - reach), z])
        go(np.r_[cmds[-1][:2], top + 0.06])
    cmds += [cmds[-1].copy() for _ in range(8)]                # short hold so the last chunk has a full 8 commands
    for c in cmds[1:]:
        if c[3] == -1.0:
            c[3] = 0.0
    return np.asarray(cmds, np.float32)


def tilted_half_height(sim, k):
    """Vertical half extent of object k in its current pose (approximate: corners of its box hull)."""
    import mujoco
    g = int(np.flatnonzero(sim.model.geom_bodyid == sim.tracked_block_ids[k])[0])
    size = sim.model.geom_size[g]
    he = np.array([size[0], size[0], size[1]]) if sim.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER else size
    R = sim.data.xmat[sim.tracked_block_ids[k]].reshape(3, 3)
    return float(np.abs(R[2]) @ he)


# ------------------------------------------------------------------------------------------- futures + harm
def pooled_error_model():
    import visual_pipeline_panel3 as p3
    e = json.loads(p3.ERROR_MODEL.read_text())
    pc = np.asarray(e["position_covariance_m2"])
    pos = sum(pc[3 * k:3 * k + 3, 3 * k:3 * k + 3] for k in range(3)) / 3
    rot = sum(np.asarray(c) for c in e["rotation_covariance_rad2_per_block"]) / 3
    return pos, rot


def draw_error(n, pos_cov, rot_cov, seed, *key):
    rng = np.random.default_rng([seed, *key])
    return rng.multivariate_normal(np.zeros(3), pos_cov, n), rng.multivariate_normal(np.zeros(3), rot_cov, n)


def future(sim, snapshot, chunk, noise, error, hold=30):
    """One future: true state + per-object pose error + executed chunk with noise, then hold. -> (8+hold, N, 7)."""
    sim.restore(snapshot)
    sim.perturb(*error)
    acts = np.asarray(chunk, np.float32).copy()
    acts[:, :3] -= noise
    out = []
    for act in list(acts) + [chunk[-1]] * hold:
        sim.execute(act); out.append(sim.object_state())
    return np.stack(out)


def events(trace, start):
    """Any-outcome events (primary label from 2026-10-13): trace (T, N, 7), start (N, 7) -> (N, 3) bool per object
    INCLUDING the target: tilt change >= 45 deg, off the table top, centre moved >= 30 mm (end of the hold)."""
    end = trace[-1]
    tilt = np.abs(tilt_deg(end[:, 3:]) - tilt_deg(start[:, 3:])) >= 45.0
    off = (end[:, 0] < TABLE_X[0]) | (end[:, 0] > TABLE_X[1]) | (end[:, 1] < TABLE_Y[0]) | (end[:, 1] > TABLE_Y[1]) \
        | (end[:, 2] < TABLE_TOP - 0.01)
    moved = np.linalg.norm(end[:, :3] - start[:, :3], axis=1) >= 0.030
    if not TARGET_MOVED_EVENT:                              # T3: the pushed block moves by design; no 'moved' event for it
        moved[0] = False
    return np.stack([tilt, off, moved], 1)


def split_count(event_list):
    """Futures' event arrays -> count of the most split (object, event) cell: the one maximising min(c, K - c).
    Any-outcome fork iff 2 <= c <= K - 2; safe iff c in {0, K}."""
    cnt = np.sum(event_list, 0).ravel(); k = len(event_list)
    return int(cnt[np.argmax(np.minimum(cnt, k - cnt))])


def harm(trace, start):
    """trace (T, N, 7), start (N, 7) -> True if any NON-target object tilts >= 45 deg from its start tilt, leaves the
    table top, or its centre moves >= 30 mm (end of the hold)."""
    end = trace[-1, 1:]; s = start[1:]
    tilt = np.abs(tilt_deg(end[:, 3:]) - tilt_deg(s[:, 3:])) >= 45.0
    off = (end[:, 0] < TABLE_X[0]) | (end[:, 0] > TABLE_X[1]) | (end[:, 1] < TABLE_Y[0]) | (end[:, 1] > TABLE_Y[1]) \
        | (end[:, 2] < TABLE_TOP - 0.01)
    moved = np.linalg.norm(end[:, :3] - s[:, :3], axis=1) >= 0.030
    return bool((tilt | off | moved).any())


# ------------------------------------------------------------------------------------------- pilot
def pilot_scene(job):
    import traceback
    family, seed = job
    try:
        return family, seed, _pilot_scene(family, seed), None
    except Exception:
        return family, seed, None, traceback.format_exc()


def _pilot_scene(family, seed):
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import chunk_starts
    scene = sample_scene(family, seed)
    sim = make_sim(scene)
    sim.reset()
    ok, disp, dtilt = settle(sim, family)              # visibility is judged on the settled start (contacts exist)
    vis, alone = visibility(sim)
    row = {"n_objects": len(scene["objects"]), "primitive": scene["primitive"], "min_visible": float(min(vis)),
           "visible_ok": visible_ok(sim, vis, alone), "settle_ok": bool(ok),
           "settle_disp_mm": disp, "settle_dtilt_deg": dtilt}
    if not (row["visible_ok"] and ok):
        return row
    acts = primitive_actions(sim, scene)
    snippets = np.load(um.BENCH)["snippets"]
    pos_cov, rot_cov = pooled_error_model()
    starts = [s for s in chunk_starts(len(acts)) if s + 8 <= len(acts)]
    harm_counts = []
    for index, act in enumerate(acts):
        if index in starts:
            snap = sim.snapshot(); warm = sim.data.qacc_warmstart.copy()
            start = sim.object_state()
            n = len(scene["objects"])
            h = [harm(future(sim, snap, acts[index:index + 8], snippets[k],
                             draw_error(n, pos_cov, rot_cov, PILOT["seed_base"], seed, index, k)), start)
                 for k in range(PILOT["futures"])]
            harm_counts.append(int(sum(h)))
            sim.restore(snap); sim.data.qacc_warmstart[:] = warm
        sim.execute(act)
    row.update({"commands": len(acts), "chunk_starts": len(starts), "harm_counts": harm_counts})
    return row


def pilot(workers):
    import multiprocessing as mp
    jobs = [(f, PILOT["seed_base"] + i) for f in ("F1", "F2", "F3") for i in range(PILOT["scenes"])]
    rows, failures = {}, []
    with mp.get_context("spawn").Pool(workers) as pool:
        for family, seed, row, err in pool.imap_unordered(pilot_scene, jobs):
            if err:
                failures.append((family, seed)); print(err, flush=True)
            else:
                rows[f"{family}:{seed}"] = row
    summary = {}
    for f in ("F1", "F2", "F3"):
        rs = [r for k, r in rows.items() if k.startswith(f)]
        valid = [r for r in rs if r["visible_ok"] and r["settle_ok"]]
        hc = [c for r in valid for c in r["harm_counts"]]
        summary[f] = {"scenes": len(rs), "visible_fail": sum(not r["visible_ok"] for r in rs),
                      "settle_fail": sum(r["visible_ok"] and not r["settle_ok"] for r in rs), "valid": len(valid),
                      "chunk_starts": len(hc), "fork_rate_1_to_7_of_8": float(np.mean([0 < c < 8 for c in hc])) if hc else None,
                      "always_harm_rate": float(np.mean([c == 8 for c in hc])) if hc else None,
                      "median_commands": float(np.median([r["commands"] for r in valid])) if valid else None}
    out = {"pilot": PILOT, "ranges": RANGES, "summary": summary, "failures": failures, "rows": rows}
    (ROOT / "results/jenga/phase2_pilot.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)


def snapshot_images(n_per_family=4):
    """Close-camera true-start renders of the first pilot scenes (for a visual check only)."""
    import mujoco
    from PIL import Image
    from jenga_render_frames import close_camera
    tiles = []
    for f in ("F1", "F2", "F3"):
        row = []
        for i in range(n_per_family):
            scene = sample_scene(f, PILOT["seed_base"] + i)
            sim = make_sim(scene); sim.reset(); settle(sim, f)
            r = mujoco.Renderer(sim.model, height=256, width=256)
            r.update_scene(sim.data, camera=close_camera(mujoco)); row.append(r.render().copy()); r.close()
        tiles.append(np.concatenate(row, 1))
    path = ROOT / "results/jenga/phase2_pilot_scenes.png"
    Image.fromarray(np.concatenate(tiles, 0)).save(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("images", "pilot"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not json.loads((ROOT / "results/jenga/phase2_generality_protocol.json").read_text())["amendment_1"]["status"].startswith("frozen"):
        raise SystemExit("amendment 1 not frozen")
    if args.stage == "images":
        print(snapshot_images())
    else:
        pilot(args.workers)


if __name__ == "__main__":
    main()

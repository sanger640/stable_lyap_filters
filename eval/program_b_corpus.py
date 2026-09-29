"""Program B generic rigid-body interaction corpus (frozen protocol + addendum 1).

Random convex objects (boxes, cylinders, wedges) on the Jenga table, arranged as stacks, rows,
leaning pairs, piles and scatters, pushed, poked, pulled, swept and grasped by the same Panda arm
with the same operational-space controller as DirectJengaSim. No Jenga scene file is loaded, the
real Jenga layout is rejected, and no label of any kind is recorded.

Each trajectory runs 100 control steps (10 Hz, 50 x 2 ms substeps) and is recorded every 10
substeps (20 ms), so the 100 ms series is an exact subsample. A seeded 25% of trajectories also
get one branch point with 8 AR(1)-residual branches of 8 + 12 control steps. Every trajectory logs
new-contact onsets at substep resolution. With ``--audit``, it also logs, at every 20 ms frame,
the minimum geom distance (<= 10 mm) of every body pair involving an object. That audit is what
addendum 1's timestep rule reads.

Splits: episodes 0-4999 train, 5000-5099 validation, 5100-5199 test.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

OUTPUT = ROOT / "results/program_b/corpus"
SIM_ARCHIVE = ROOT / "vendor/panda_express_sim.tar"
BASE_SEED = 20260929
SPLITS = {"train": (0, 5000), "validation": (5000, 5100), "test": (5100, 5200)}

TABLE_TOP = 0.425
TABLE_CENTER = np.array([0.5, 0.0])
TABLE_HALF = np.array([0.2, 0.4])
WORKSPACE_X = (0.40, 0.60)
WORKSPACE_Y = (-0.14, 0.14)
SITE_MIN_Z = TABLE_TOP + 0.020          # fingertips sit ~17 mm below the attachment site
SITE_MAX_Z = 0.75
TRANSIT_SPEED = 0.020                   # m per control step through free space
HOME_QPOS = {"joint1": 0.0, "joint2": -0.93, "joint3": 0.0, "joint4": -2.77, "joint5": 0.0,
             "joint6": 2.72, "joint7": 0.7853, "finger_joint1": 0.04, "finger_joint2": 0.04}

CONTROL_STEPS = 100
SUBSTEPS = 50
RECORD_EVERY = 10
FRAMES_PER_STEP = SUBSTEPS // RECORD_EVERY
ONSET_GAP = 10
AUDIT_DISTMAX = 0.010
SIZE_RANGE = (0.010, 0.120)
DENSITY_RANGE = (300.0, 1500.0)
FRICTION_RANGE = (0.3, 1.2)

BRANCH_FRACTION = 0.25
BRANCH_WINDOW = (10, 80)
BRANCHES = 8
BRANCH_ACTION = 8
BRANCH_HOLD = 12
RESIDUAL_SIGMA = 0.0032
RESIDUAL_RHO = 0.8

JENGA_DIMS = np.array([0.020, 0.025, 0.075])
TABLE, FLOOR, HAND, LEFT_FINGER, RIGHT_FINGER, ARM = 100, 101, 102, 103, 104, 105


# ---------------------------------------------------------------------------------------------
# Objects and scene layout


def random_quaternion(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def yaw_quaternion(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_multiply(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def log_uniform(rng, low, high, size=None):
    return np.exp(rng.uniform(np.log(low), np.log(high), size))


def sample_object(rng):
    kind = rng.choice(["box", "box", "cylinder", "wedge"])
    if kind == "box":
        dims = log_uniform(rng, *SIZE_RANGE, 3)                       # full extents x, y, z
    elif kind == "cylinder":
        dims = np.array([log_uniform(rng, *SIZE_RANGE) / 2, log_uniform(rng, *SIZE_RANGE)])
    else:
        dims = log_uniform(rng, *SIZE_RANGE, 3)                       # base x, depth y, height z
    return {"kind": str(kind), "dims": [float(v) for v in dims],
            "density": float(rng.uniform(*DENSITY_RANGE)),
            "friction": float(rng.uniform(*FRICTION_RANGE))}


def object_vertices(obj):
    """Vertices in the body frame (a 32-gon for cylinders; wedges are right triangular prisms)."""
    d = np.asarray(obj["dims"])
    if obj["kind"] == "box":
        h = d / 2
        return np.array([[sx * h[0], sy * h[1], sz * h[2]]
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    if obj["kind"] == "cylinder":
        radius, height = d
        angle = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        ring = np.stack([radius * np.cos(angle), radius * np.sin(angle)], 1)
        return np.concatenate([np.c_[ring, np.full(32, -height / 2)],
                               np.c_[ring, np.full(32, height / 2)]])
    x, y, z = d / 2
    return np.array([[-x, -y, -z], [x, -y, -z], [-x, -y, z],
                     [-x, y, -z], [x, y, -z], [-x, y, z]])


def resting_orientation(obj, rng):
    """A stable-looking resting face with random yaw."""
    tilt = np.array([1.0, 0.0, 0.0, 0.0])
    if obj["kind"] in ("box", "wedge"):
        axis = rng.integers(3)
        if axis == 0:
            tilt = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
        elif axis == 1:
            tilt = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    elif obj["kind"] == "cylinder" and rng.random() < .5:
        tilt = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    return quat_multiply(yaw_quaternion(rng.uniform(0, 2 * np.pi)), tilt)


def extent(obj, quat):
    """(min, max) of the rotated vertices along x, y, z."""
    world = object_vertices(obj) @ quat_matrix(quat).T
    return world.min(0), world.max(0)


def layout_group(objects, kind, center, rng):
    """Place a group of objects; returns [(pos, quat)] before settling."""
    placements = []
    if kind == "stack":
        base = TABLE_TOP
        below = None
        for obj in objects:
            quat = resting_orientation(obj, rng)
            low, high = extent(obj, quat)
            xy = center.copy()
            if below is not None:
                xy = below + rng.uniform(-.3, .3, 2) * max(1e-3, min(high[:2] - low[:2]) / 2)
            placements.append((np.r_[xy, base - low[2] + 1e-3], quat))
            base += high[2] - low[2] + 1e-3
            below = xy
    elif kind == "row":
        heading = rng.uniform(0, np.pi)
        direction = np.array([np.cos(heading), np.sin(heading)])
        cursor = 0.0
        for number, obj in enumerate(objects):
            quat = resting_orientation(obj, rng)
            low, high = extent(obj, quat)
            half = float(np.max(np.abs(np.r_[low[:2], high[:2]])))
            if number:
                cursor += half + rng.uniform(0.0, 0.03)
            xy = center + direction * cursor
            placements.append((np.r_[xy, TABLE_TOP - low[2] + 1e-3], quat))
            cursor += half
    elif kind == "lean":
        heading = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(heading), np.sin(heading)])
        first, rest = objects[0], objects[1:]
        quat = resting_orientation(first, rng)
        low, high = extent(first, quat)
        placements.append((np.r_[center, TABLE_TOP - low[2] + 1e-3], quat))
        offset = float(np.max(np.abs(np.r_[low[:2], high[:2]])))
        for obj in rest:
            angle = rng.uniform(np.radians(20), np.radians(50))
            axis = np.r_[-direction[1], direction[0], 0.0]
            lean = np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]
            quat = quat_multiply(lean, resting_orientation(obj, rng))
            low2, high2 = extent(obj, quat)
            xy = center + direction * (offset + float(np.max(np.abs(np.r_[low2[:2], high2[:2]]))) * .6)
            placements.append((np.r_[xy, TABLE_TOP - low2[2] + 2e-3], quat))
    elif kind == "pile":
        height = TABLE_TOP + 0.02
        for obj in objects:
            quat = random_quaternion(rng)
            low, high = extent(obj, quat)
            xy = center + rng.uniform(-0.02, 0.02, 2)
            placements.append((np.r_[xy, height - low[2]], quat))
            height += high[2] - low[2] + 0.01
    else:                                                               # scatter
        for obj in objects:
            quat = resting_orientation(obj, rng)
            low, _ = extent(obj, quat)
            xy = center + rng.uniform(-0.06, 0.06, 2)
            placements.append((np.r_[xy, TABLE_TOP - low[2] + 1e-3], quat))
    return placements


def sample_scene(rng):
    count = int(rng.integers(2, 13))
    objects = [sample_object(rng) for _ in range(count)]
    groups, start = [], 0
    while start < count:
        size = int(min(count - start, rng.integers(1, 5)))
        kinds = ["scatter"] if size == 1 else ["stack", "row", "lean", "pile", "scatter"]
        groups.append((str(rng.choice(kinds)), list(range(start, start + size))))
        start += size
    placements = [None] * count
    centers = []
    for kind, members in groups:
        for _ in range(50):
            center = np.array([rng.uniform(*WORKSPACE_X), rng.uniform(*WORKSPACE_Y)])
            if all(np.linalg.norm(center - c) > 0.07 for c in centers):
                break
        centers.append(center)
        for index, placement in zip(members, layout_group([objects[i] for i in members], kind,
                                                          center, rng)):
            placements[index] = placement
    for obj, (pos, quat) in zip(objects, placements):
        obj["init_pos"] = [float(v) for v in pos]
        obj["init_quat"] = [float(v) for v in quat]
    return {"objects": objects, "groups": [{"kind": k, "members": m} for k, m in groups]}


def is_jenga_layout(objects, positions, quats, tolerance=0.2, line_tolerance=0.010):
    """True if >= 3 upright Jenga-sized boxes have centres within 10 mm of a common line."""
    candidates = []
    for obj, pos, quat in zip(objects, positions, quats):
        if obj["kind"] != "box":
            continue
        dims = np.asarray(obj["dims"])
        if not np.all(np.abs(np.sort(dims) - JENGA_DIMS) <= tolerance * JENGA_DIMS):
            continue
        long_axis = quat_matrix(quat)[:, int(np.argmax(dims))]
        if abs(long_axis[2]) > np.cos(np.radians(20)):                 # standing on its end
            candidates.append(np.asarray(pos[:2]))
    if len(candidates) < 3:
        return False
    points = np.asarray(candidates)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            direction = points[j] - points[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            normal = np.array([-direction[1], direction[0]]) / norm
            if np.sum(np.abs((points - points[i]) @ normal) <= line_tolerance) >= 3:
                return True
    return False


def scene_xml(scene):
    assets, bodies = [], []
    for number, obj in enumerate(scene["objects"]):
        pos = " ".join(f"{v:.6f}" for v in obj["init_pos"])
        quat = " ".join(f"{v:.6f}" for v in obj["init_quat"])
        common = (f'density="{obj["density"]:.3f}" friction="{obj["friction"]:.4f} 0.005 0.0001" '
                  f'name="obj{number}_geom"')
        d = obj["dims"]
        if obj["kind"] == "box":
            geom = f'<geom type="box" size="{d[0] / 2:.6f} {d[1] / 2:.6f} {d[2] / 2:.6f}" {common}/>'
        elif obj["kind"] == "cylinder":
            geom = f'<geom type="cylinder" size="{d[0]:.6f} {d[1] / 2:.6f}" {common}/>'
        else:
            vertex = " ".join(f"{v:.6f}" for v in object_vertices(obj).ravel())
            assets.append(f'<mesh name="obj{number}_mesh" vertex="{vertex}"/>')
            geom = f'<geom type="mesh" mesh="obj{number}_mesh" {common}/>'
        bodies.append(f'<body name="obj{number}" pos="{pos}" quat="{quat}">'
                      f'<freejoint name="obj{number}_joint"/>{geom}</body>')
    return f"""<mujoco model="program_b_generic">
  <include file="panda.xml"/>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    <geom name="floor" size="0 0 0.05" type="plane"/>
    <body name="table" pos="0.5 0 0">
      <geom size="0.2 0.4 0.025" type="box" pos="0 0 0.4" mass="1000" contype="1" conaffinity="1"/>
    </body>
    {''.join(bodies)}
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------------------------------------
# Simulator (same arm, controller and substep count as DirectJengaSim)


class GenericSim:
    def __init__(self, xml_path, objects):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.site = m.site("attachment_site").id
        self.steps_per_action = int(round(0.1 / m.opt.timestep))
        assert self.steps_per_action == SUBSTEPS
        self.objects = objects
        self.object_bodies = [m.body(f"obj{i}").id for i in range(len(objects))]
        self.robot_bodies = [m.body(n).id for n in ("hand", "left_finger", "right_finger")]
        code = {body: i for i, body in enumerate(self.object_bodies)}
        code.update({m.body("table").id: TABLE, 0: FLOOR, self.robot_bodies[0]: HAND,
                     self.robot_bodies[1]: LEFT_FINGER, self.robot_bodies[2]: RIGHT_FINGER})
        self.body_code = np.array([code.get(b, ARM) for b in range(m.nbody)])
        self.collision_geoms = np.flatnonzero((m.geom_contype != 0) | (m.geom_conaffinity != 0))
        self.target = np.zeros(3)
        self.orientation = np.zeros(4)
        self.gripper = 110.0
        self.substep = 0
        self.last_seen = {}

    def reset(self):
        m, d, mj = self.model, self.data, self.mj
        mj.mj_resetData(m, d)
        for name, value in HOME_QPOS.items():
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = value
        d.ctrl[:] = 0.0
        mj.mj_forward(m, d)
        self.target = d.site_xpos[self.site].copy()
        self.target = np.array([0.5, 0.0, 0.60])                      # settle moves the hand here
        mj.mju_mat2Quat(self.orientation, d.site_xmat[self.site])
        self.gripper = 110.0
        self.substep = 0
        self.last_seen = {}

    def _control(self):
        m, d, mj = self.model, self.data, self.mj
        current = d.site_xmat[self.site].reshape(3, 3)
        target_matrix = np.zeros(9)
        mj.mju_quat2Mat(target_matrix, self.orientation)
        error = target_matrix.reshape(3, 3) @ current.T
        dr = np.array([error[2, 1] - error[1, 2], error[0, 2] - error[2, 0],
                       error[1, 0] - error[0, 1]]) * 0.5
        jac = np.zeros((6, m.nv))
        mj.mj_jacSite(m, d, jac[:3], jac[3:], self.site)
        velocity = jac @ d.qvel
        wrench = np.hstack([500 * (self.target - d.site_xpos[self.site]) - 30 * velocity[:3],
                            20 * dr - 0.5 * velocity[3:]])
        d.ctrl[:7] = (jac.T @ wrench - 0.1 * d.qvel)[:7]
        d.ctrl[7] = self.gripper

    def settle(self, substeps=750):
        for _ in range(substeps):
            self._control()
            self.mj.mj_step(self.model, self.data)

    def contact_pairs(self):
        pairs = set()
        for contact in self.data.contact[:self.data.ncon]:
            a = int(self.body_code[self.model.geom_bodyid[contact.geom1]])
            b = int(self.body_code[self.model.geom_bodyid[contact.geom2]])
            if min(a, b) < TABLE and a != b:
                pairs.add((min(a, b), max(a, b)))
        return pairs

    def body_pair_distances(self, distmax=AUDIT_DISTMAX):
        """Minimum geom distance (<= distmax) for every body pair involving an object."""
        m, d, mj = self.model, self.data, self.mj
        geoms = self.collision_geoms
        codes = self.body_code[m.geom_bodyid[geoms]]
        positions, radii = d.geom_xpos[geoms], m.geom_rbound[geoms]
        best = {}
        fromto = np.zeros(6)
        for i in np.flatnonzero(codes < TABLE):
            near = np.linalg.norm(positions - positions[i], axis=1) - radii - radii[i] <= distmax
            for j in np.flatnonzero(near):
                a, b = int(codes[i]), int(codes[j])
                if a == b or (b < TABLE and j <= i):
                    continue
                distance = mj.mj_geomDistance(m, d, int(geoms[i]), int(geoms[j]), distmax, fromto)
                if distance <= distmax:
                    key = (min(a, b), max(a, b))
                    best[key] = min(best.get(key, np.inf), float(distance))
        return best

    def start_recording(self):
        """Contacts already present are not onsets; count substeps from here."""
        self.substep = 0
        self.last_seen = {pair: 0 for pair in self.contact_pairs()}

    def snapshot(self):
        state = np.empty(self.mj.mj_stateSize(self.model, self.mj.mjtState.mjSTATE_FULLPHYSICS))
        self.mj.mj_getState(self.model, self.data, state, self.mj.mjtState.mjSTATE_FULLPHYSICS)
        return state, self.data.ctrl.copy(), self.target.copy(), float(self.gripper)

    def restore(self, snapshot):
        state, ctrl, target, gripper = snapshot
        self.mj.mj_setState(self.model, self.data, state, self.mj.mjtState.mjSTATE_FULLPHYSICS)
        self.data.ctrl[:] = ctrl
        self.target = target.copy()
        self.gripper = gripper
        self.mj.mj_forward(self.model, self.data)

    def frame(self):
        d = self.data
        objects = np.concatenate([d.xpos[self.object_bodies], d.xquat[self.object_bodies]], 1)
        robot = np.concatenate([d.xpos[self.robot_bodies], d.xquat[self.robot_bodies]], 1)
        m = self.model
        fingers = [d.qpos[m.jnt_qposadr[m.joint(n).id]] for n in ("finger_joint1", "finger_joint2")]
        return objects, robot, d.site_xpos[self.site].copy(), np.asarray(fingers)

    def execute(self, action, recorder=None, audit=False):
        action = np.asarray(action, float)
        self.target = np.clip(action[:3], [0.25, -0.35, SITE_MIN_Z], [0.75, 0.35, SITE_MAX_Z])
        if action[3] > 0.9:
            self.gripper = 0.0
        elif action[3] < -0.9:
            self.gripper = 110.0
        for number in range(self.steps_per_action):
            self._control()
            self.mj.mj_step(self.model, self.data)
            self.substep += 1
            if recorder is not None:
                for pair in self.contact_pairs():
                    previous = self.last_seen.get(pair)
                    if previous is None or previous < self.substep - ONSET_GAP:
                        recorder["onsets"].append((self.substep, *pair))
                    self.last_seen[pair] = self.substep
                if (number + 1) % RECORD_EVERY == 0:
                    record_frame(recorder, self, audit)


def new_recorder():
    return {"objects": [], "robot": [], "site": [], "fingers": [], "onsets": [], "audit": []}


def record_frame(recorder, sim, audit):
    objects, robot, site, fingers = sim.frame()
    recorder["objects"].append(objects); recorder["robot"].append(robot)
    recorder["site"].append(site); recorder["fingers"].append(fingers)
    if audit:
        frame = len(recorder["objects"]) - 1
        recorder["audit"].extend((frame, a, b, dist)
                                 for (a, b), dist in sim.body_pair_distances().items())


def stack_recorder(recorder):
    return {
        "objects": np.asarray(recorder["objects"], np.float64),
        "robot": np.asarray(recorder["robot"], np.float64),
        "site": np.asarray(recorder["site"], np.float64),
        "fingers": np.asarray(recorder["fingers"], np.float64),
        "onsets": np.asarray(recorder["onsets"], np.int64).reshape(-1, 3),
        "audit": np.asarray(recorder["audit"], np.float64).reshape(-1, 4),
    }


# ---------------------------------------------------------------------------------------------
# Behaviours: random pushes, pokes, pulls, sweeps and grasps


def interpolate(start, goal, speed):
    distance = np.linalg.norm(goal - start)
    count = max(1, int(np.ceil(distance / speed)))
    return [start + (goal - start) * (k + 1) / count for k in range(count)]


def plan_behaviour(sim, rng, current):
    objects, _, _, _ = sim.frame()
    index = int(rng.integers(len(objects)))
    center = objects[index, :3]
    low, high = extent(sim.objects[index], objects[index, 3:])
    radius = float(np.max(np.abs(np.r_[low[:2], high[:2]])))
    tops = [pose[2] + extent(obj, pose[3:])[1][2] for obj, pose in zip(sim.objects, objects)]
    safe = float(np.clip(max(tops) + 0.03 + 0.017, SITE_MIN_Z, SITE_MAX_Z))  # fingertips clear
    speed = float(log_uniform(rng, 0.002, 0.020))
    height = float(np.clip(center[2] + rng.uniform(-0.3, 0.3) * (high[2] - low[2]),
                           SITE_MIN_Z, SITE_MAX_Z))
    heading = rng.uniform(0, 2 * np.pi)
    u = np.array([np.cos(heading), np.sin(heading), 0.0])
    kind = str(rng.choice(["push", "poke", "pull", "sweep", "grasp", "move"],
                          p=[.3, .15, .15, .1, .2, .1]))
    grip = float(rng.choice([1.0, -1.0]))
    waypoints = []
    if kind in ("push", "poke", "pull"):
        side = -1.0 if kind != "pull" else 1.0
        pre = np.r_[center[:2] + side * (radius + 0.03) * u[:2], height]
        travel = rng.uniform(0.02, 0.10) if kind != "poke" else rng.uniform(0.005, 0.02)
        end = pre - side * (radius + 0.03 + travel) * u if kind == "pull" else pre + (0.03 + travel) * u
        waypoints = [("transit", np.r_[pre[:2], safe]), pre, end]
        if kind == "poke":
            waypoints.append(pre)
        waypoints.append(("transit", np.r_[waypoints[-1][:2], safe]))
    elif kind == "sweep":
        z = SITE_MIN_Z + rng.uniform(0.0, 0.02)
        start = np.r_[center[:2] - (radius + 0.08) * u[:2], z]
        waypoints = [("transit", np.r_[start[:2], safe]), start, start + (2 * radius + 0.16) * u,
                     ("transit", np.r_[(start + (2 * radius + 0.16) * u)[:2], safe])]
    elif kind == "grasp":
        grip = -1.0
        above = np.r_[center[:2], safe]
        grasp = np.r_[center[:2], float(np.clip(center[2], SITE_MIN_Z, SITE_MAX_Z))]
        lift = grasp + np.r_[rng.uniform(-0.05, 0.05, 2), rng.uniform(0.03, 0.08)]
        waypoints = [("transit", above), grasp, ("close", 3), lift, ("open", 2),
                     ("transit", np.r_[lift[:2], safe])]
    else:
        waypoints = [("transit", np.r_[rng.uniform(*WORKSPACE_X), rng.uniform(*WORKSPACE_Y),
                                       rng.uniform(safe, max(safe, 0.60))])]
    commands, position = [], np.asarray(current, float).copy()
    for waypoint in waypoints:
        step = speed
        if isinstance(waypoint, tuple) and waypoint[0] in ("close", "open"):
            command = 1.0 if waypoint[0] == "close" else -1.0
            commands.extend([np.r_[position, command]] + [np.r_[position, 0.0]] * (waypoint[1] - 1))
            continue
        if isinstance(waypoint, tuple):                                 # ("transit", point)
            step, waypoint = TRANSIT_SPEED, waypoint[1]
        goal = np.clip(waypoint, [0.25, -0.35, SITE_MIN_Z], [0.75, 0.35, SITE_MAX_Z])
        for point in interpolate(position, goal, step):
            commands.append(np.r_[point, grip if not commands else 0.0])
        position = goal
    return kind, commands


def ar1_residual(rng, steps=BRANCH_ACTION, sigma=RESIDUAL_SIGMA, rho=RESIDUAL_RHO):
    residual = np.zeros((steps, 3))
    residual[0] = rng.normal(0, sigma, 3)
    for t in range(1, steps):
        residual[t] = rho * residual[t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, sigma, 3)
    return residual


# ---------------------------------------------------------------------------------------------
# Episodes


def build(scene, directory, name):
    path = Path(directory) / f"{name}.xml"
    path.write_text(scene_xml(scene))
    sim = GenericSim(path, scene["objects"])
    sim.reset()
    return sim


def settled_ok(sim):
    objects, _, _, _ = sim.frame()
    d = sim.data
    speeds = np.linalg.norm(d.cvel[sim.object_bodies][:, 3:], axis=1)
    on_table = np.all(objects[:, 2] > TABLE_TOP - 0.005)
    inside = np.all(np.abs(objects[:, :2] - TABLE_CENTER) < TABLE_HALF - 0.01)
    penetration = min([c.dist for c in d.contact[:d.ncon]] + [0.0])
    return bool(on_table and inside and speeds.max() < 0.02 and penetration > -0.003)


def generate_episode(episode, directory, audit=False, steps=CONTROL_STEPS):
    rng = np.random.default_rng(BASE_SEED + episode)
    for attempt in range(40):
        scene = sample_scene(rng)
        sim = build(scene, directory, f"ep{episode}")
        sim.settle()
        objects, _, _, _ = sim.frame()
        if settled_ok(sim) and not is_jenga_layout(scene["objects"], objects[:, :3],
                                                     objects[:, 3:]):
            break
    else:
        raise RuntimeError(f"episode {episode}: no valid scene after 40 attempts")
    settle_state = sim.data.qpos.copy(), sim.data.qvel.copy()
    branched = rng.random() < BRANCH_FRACTION
    branch_step = int(rng.integers(BRANCH_WINDOW[0], BRANCH_WINDOW[1] + 1)) if branched else -1
    if steps < BRANCH_WINDOW[1] + BRANCH_ACTION:                       # short test episodes only
        branched, branch_step = False, -1

    recorder = new_recorder()
    sim.start_recording()
    record_frame(recorder, sim, audit)
    actions, behaviours, queue = [], [], []
    while len(actions) < steps:
        if not queue:
            kind, queue = plan_behaviour(sim, rng, sim.target)
            behaviours.append((len(actions), kind))
        action = queue.pop(0)
        sim.execute(action, recorder, audit)
        actions.append(action)
    actions = np.asarray(actions, np.float64)
    result = {"scene": scene, "attempts": attempt + 1, "behaviours": behaviours,
              "actions": actions, "main": stack_recorder(recorder), "branch_step": branch_step}

    if branched:
        replay = build(scene, directory, f"ep{episode}_replay")
        replay.settle()
        if not (np.array_equal(replay.data.qpos, settle_state[0])
                and np.array_equal(replay.data.qvel, settle_state[1])):
            raise RuntimeError(f"episode {episode}: settle replay is not deterministic")
        for action in actions[:branch_step]:
            replay.execute(action)
        snapshot = replay.snapshot()
        branches, residuals = [], []
        for _ in range(BRANCHES):
            replay.restore(snapshot)
            residual = ar1_residual(rng)
            branch_actions = actions[branch_step:branch_step + BRANCH_ACTION].copy()
            branch_actions[:, :3] += residual
            hold = np.r_[branch_actions[-1, :3], 0.0]
            branch_actions = np.concatenate([branch_actions, np.repeat(hold[None], BRANCH_HOLD, 0)])
            branch_recorder = new_recorder()
            replay.start_recording()
            record_frame(branch_recorder, replay, False)
            for action in branch_actions:
                replay.execute(action, branch_recorder, False)
            branches.append((branch_actions, stack_recorder(branch_recorder)))
            residuals.append(residual)
        result["branches"] = branches
        result["residuals"] = np.asarray(residuals)
    return result


def save_episode(result, path, episode, split):
    arrays = {"actions": result["actions"], "branch_step": np.int64(result["branch_step"])}
    for key, value in result["main"].items():
        arrays[f"main_{key}"] = value
    if "branches" in result:
        arrays["branch_actions"] = np.stack([b[0] for b in result["branches"]])
        for key in ("objects", "robot", "site", "fingers"):
            arrays[f"branch_{key}"] = np.stack([b[1][key] for b in result["branches"]])
        arrays["branch_residuals"] = result["residuals"]
    meta = {"episode": episode, "split": split, "seed": BASE_SEED + episode,
            "scene": result["scene"], "attempts": result["attempts"],
            "behaviours": result["behaviours"], "record_every_substeps": RECORD_EVERY,
            "substeps_per_action": SUBSTEPS, "quaternion_order": "wxyz",
            "robot_bodies": ["hand", "left_finger", "right_finger"],
            "body_codes": {"objects": "0..n-1", "table": TABLE, "floor": FLOOR, "hand": HAND,
                           "left_finger": LEFT_FINGER, "right_finger": RIGHT_FINGER, "arm": ARM}}
    arrays["meta"] = np.asarray(json.dumps(meta))
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


_WORKER_DIR = None


def _init_worker(archive):
    global _WORKER_DIR
    from jenga_short_held_tails import extract_sim
    temp = tempfile.mkdtemp(prefix="program_b_corpus_")
    _WORKER_DIR = extract_sim(archive, temp).parent


def _job(args):
    episode, split, output, audit = args
    path = Path(output) / split / f"ep{episode:05d}.npz"
    if path.exists():
        return episode, 0.0, True
    start = time.time()
    save_episode(generate_episode(episode, _WORKER_DIR, audit), path, episode, split)
    return episode, time.time() - start, False


def split_of(episode):
    for name, (first, stop) in SPLITS.items():
        if first <= episode < stop:
            return name
    raise ValueError(f"episode {episode} outside every split")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--audit", action="store_true",
                        help="log body-pair distances every 20 ms (validation timestep rule)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("simulator generation is capped at four workers")
    episodes = list(range(args.start, args.start + args.count))
    for split in {split_of(e) for e in episodes}:
        (Path(args.output) / split).mkdir(parents=True, exist_ok=True)
    jobs = [(e, split_of(e), args.output, args.audit) for e in episodes]
    start, times = time.time(), []
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(str(SIM_ARCHIVE),)) as pool:
        for number, (episode, seconds, skipped) in enumerate(pool.map(_job, jobs), 1):
            if not skipped:
                times.append(seconds)
            if number % 10 == 0 or number == len(jobs):
                print(f"{number}/{len(jobs)} episodes, {time.time() - start:.0f} s wall, "
                      f"median {np.median(times) if times else 0:.1f} s/episode", flush=True)
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = Path(args.output) / f"manifest_{args.start}_{args.count}.json"
    manifest.write_text(json.dumps({
        "start": args.start, "count": args.count, "workers": args.workers, "audit": args.audit,
        "generator_sha256": code, "wall_seconds": time.time() - start,
        "median_episode_seconds": float(np.median(times)) if times else None,
        "episodes_per_hour": len(times) / max(time.time() - start, 1e-9) * 3600,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

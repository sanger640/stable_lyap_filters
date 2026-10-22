"""Custom second task for the filter (user, 2026-10-22: "develop a custom test environment in MuJoCo ... similar and
relatively simple but different enough to claim generalisability"): PRECARIOUS PLACEMENT.

Same robot, controller, action format ([x, y, z, gripper] absolute target, 0.1 s per action), camera and execution-noise
model as the Jenga task (eval/jenga_short_held_tails.DirectJengaSim). Different contact mode and failure:
  Jenga  = extract a block from between two neighbours; failure = a neighbour topples.
  Pillar = pick a cube from the table and set it on top of a narrow upright pillar, then release and retreat;
           failure = the cube falls off or the pillar tips over.
Scripted expert (used for the quick fork check and as the demonstrator for a Diffusion Policy later) in expert_plan().

usage: python3 eval/pillar_task.py demo --episodes 4      (runs the expert, prints outcomes)
       python3 eval/pillar_task.py fork --episodes 16 --noise 2 --branch-noise 2 --out results/pillar/fork_quick.json
       python3 eval/pillar_task.py video --episodes 2 --out f.mp4
"""
import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

TABLE_Z = 0.425
CAM = "0.5 -0.55 0.62"                                                      # world-model camera: fixed, side-on to the pillar
_W = float(os.environ.get("PILLAR_HALF_W", 0.0125))                     # pillar half-width (task difficulty)
GEOM = dict(pillar_half=(_W, _W, 0.05), cube_half=0.015, pillar_mass=0.05, cube_mass=0.03, friction=0.6)

SCENE = """<mujoco model="panda_pillar_setup">
  <include file="panda.xml"/>
  <statistic center="0.3 0 0.4" extent="0.8"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/><rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <body name="table" pos="0.5 0 0">
      <geom size="0.2 0.4 0.025" type="box" rgba="0.9 0.9 0.9 1" pos="0 0 0.4" mass="1000" contype="1" conaffinity="1"/>
    </body>
    <body name="cube" pos="0.45 0.08 {cube_z}">
      <joint type="free"/>
      <geom size="{ch} {ch} {ch}" type="box" rgba="0.9 0.1 0.1 1" mass="{cube_mass}" friction="{friction} 0.005 0.0001"/>
    </body>
    <body name="pillar" pos="0.5 -0.05 {pillar_z}">
      <joint type="free"/>
      <geom size="{px} {py} {pz}" type="box" rgba="0.8 0.6 0.4 1" mass="{pillar_mass}" friction="{friction} 0.005 0.0001"/>
    </body>
    <body name="cam_target" pos="0.5 -0.02 0.49"/>
    <camera name="cam_fixed" pos="{cam}" mode="targetbody" target="cam_target" fovy="{fovy}"/>
  </worldbody>
  <keyframe>
    <key name="home" qpos="0 -0.93 0 -2.77 0 2.72 0.7853 0.04 0.04
                           0.45 0.08 {cube_z} 1 0 0 0
                           0.5 -0.05 {pillar_z} 1 0 0 0"
         ctrl="0 -0.92 0 -2.77 0 2.72 0.7853 110"/>
  </keyframe>
</mujoco>
"""


def scene_xml(geom=GEOM):
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import extract_sim
    jenga_xml = Path(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="pillar_")))
    px, py, pz = geom["pillar_half"]; ch = geom["cube_half"]
    xml = SCENE.format(px=px, py=py, pz=pz, ch=ch, cube_z=TABLE_Z + ch + 0.001, pillar_z=TABLE_Z + pz + 0.001,
                       cube_mass=geom["cube_mass"], pillar_mass=geom["pillar_mass"], friction=geom["friction"],
                       cam=os.environ.get("PILLAR_CAM", CAM), fovy=os.environ.get("PILLAR_FOVY", "45"))
    tag = "_".join(f"{v}" for v in (px, py, pz, ch, geom["cube_mass"], geom["pillar_mass"], geom["friction"],
                                     os.environ.get("PILLAR_CAM", CAM).replace(" ", ","), os.environ.get("PILLAR_FOVY", "45")))
    out = jenga_xml.parent / f"panda_pillar_setup_{tag}.xml"
    if not out.exists():
        out.write_text(xml)
    return out


class PillarSim:
    """DirectJengaSim with the pillar scene: tracked bodies = [cube, pillar]; block_ids = [pillar] (max_tilt = pillar)."""

    def __new__(cls, geom=GEOM):
        import jenga_short_held_tails as J
        names = J.BLOCK_NAMES
        J.BLOCK_NAMES = ("cube", "pillar")
        try:
            sim = J.DirectJengaSim(str(scene_xml(geom)))
        finally:
            J.BLOCK_NAMES = names
        sim.geom = geom
        sim.reset = lambda seed, s=sim: reset(s, seed)
        return sim


def reset(sim, seed):
    """Keyframe + randomised layout: pillar at (0.50 +- 0.02, -0.05 +- 0.02), cube at (0.45 +- 0.02, 0.08 +- 0.02), small yaws."""
    m, d, mj = sim.model, sim.data, sim.mj
    mj.mj_resetDataKeyframe(m, d, 0)
    rng = np.random.default_rng(int(seed))
    for name in ("cube", "pillar"):
        adr = m.jnt_qposadr[m.body_jntadr[m.body(name).id]]
        d.qpos[adr:adr + 2] += rng.uniform(-0.02, 0.02, 2)
        yaw = np.radians(rng.uniform(-10.0, 10.0)); d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mj.mj_forward(m, d)
    sim.target = d.site_xpos[sim.site].copy(); mj.mju_mat2Quat(sim.orientation, d.site_xmat[sim.site])
    sim.gripper = 110.0
    for _ in range(5):                                                        # settle
        sim.execute(np.r_[sim.target, -1.0])


def body_pos(sim, name):
    return sim.data.xpos[sim.model.body(name).id].copy()


def expert_plan(sim, approach=0.06, place_clear=0.004, speed=0.012):
    """Waypoints of the scripted placement: above cube -> down -> close -> lift -> above pillar -> descend until the cube
    clears the pillar top by place_clear -> open -> retreat up. Returned as a list of [x, y, z, grip] actions, each a
    straight-line step of at most `speed` metres (0.1 s per action, as in Jenga)."""
    g = sim.geom; ch, pz = g["cube_half"], g["pillar_half"][2]
    c, p = body_pos(sim, "cube"), body_pos(sim, "pillar")
    grasp_off = 0.0                                                           # attachment site at the cube centre when grasped
    pillar_top = p[2] + pz
    waypoints = [(np.r_[c[:2], c[2] + approach], -1), (np.r_[c[:2], c[2] + grasp_off], -1), ("close", 1),
                 (np.r_[c[:2], c[2] + 0.12], 1), (np.r_[p[:2], pillar_top + ch + 0.10], 1),
                 (np.r_[p[:2], pillar_top + ch + place_clear], 1), ("open", -1), (np.r_[p[:2], pillar_top + ch + 0.10], -1)]
    acts, cur = [], sim.data.site_xpos[sim.site].copy()
    for w, grip in waypoints:
        if isinstance(w, str):
            acts += [np.r_[cur, grip]] * 6                                    # 0.6 s for the gripper to close / open
            continue
        n = max(1, int(np.ceil(np.linalg.norm(w - cur) / speed)))
        acts += [np.r_[cur + (w - cur) * (k + 1) / n, grip] for k in range(n)]
        cur = w
    return np.array(acts, np.float32)


def outcome(sim, pillar0):
    """success = cube resting on the pillar top (within the top's footprint, z above it) and the pillar upright (< 20 deg)
    and not moved > 2 cm; failure modes: cube_fell (cube below the top), pillar_tipped (> 45 deg)."""
    g = sim.geom; c, p = body_pos(sim, "cube"), body_pos(sim, "pillar")
    tilt = sim.block_diagnostics()["tilt"]                                   # [cube, pillar]
    top = p[2] + g["pillar_half"][2]
    on_top = c[2] > top and np.linalg.norm(c[:2] - p[:2]) < g["pillar_half"][0] + g["cube_half"]
    return {"success": bool(on_top and tilt[1] < 20 and np.linalg.norm(p[:2] - pillar0[:2]) < 0.02),
            "cube_fell": bool(c[2] < top - 0.01), "pillar_tipped": bool(tilt[1] > 45),
            "cube_z": float(c[2]), "pillar_tilt": float(tilt[1]), "offset_mm": float(np.linalg.norm(c[:2] - p[:2]) * 1000)}


def run_episode(sim, seed, noise_scale=0.0, record=False, settle=15):
    """Expert closed loop with Jenga's execution noise (snippets x noise_scale added to the target, per 8-action chunk)."""
    import uncertainty_monitor_dev as um
    reset(sim, seed)
    snippets = np.load(um.BENCH)["snippets"]; rng = np.random.default_rng([20261022, seed])
    plan, frames = expert_plan(sim), []
    pillar0 = body_pos(sim, "pillar")
    for i, a in enumerate(plan):
        if i % 8 == 0:
            nz = snippets[rng.integers(len(snippets))] * noise_scale
        a = a.copy(); a[:3] += nz[i % 8]
        sim.execute(a)
        if record:
            frames.append(sim.render())
    for _ in range(settle):
        sim.execute(np.r_[sim.target, -1.0])
        if record:
            frames.append(sim.render())
    return outcome(sim, pillar0), frames, len(plan)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.0); ap.add_argument("--out", default="pillar.mp4")
    ap.add_argument("--branch-noise", type=float, default=2.0); ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    if a.cmd == "fork":                                                       # fork --episodes 16 --noise 2 --branch-noise 2 --out f.json
        return fork_check(a.episodes, a.noise, a.branch_noise, a.procs, a.out)
    sim = PillarSim()
    if a.cmd == "demo":
        res = [run_episode(sim, s, a.noise)[0] for s in range(a.episodes)]
        for s, r in enumerate(res):
            print(s, r)
        print("success", np.mean([r["success"] for r in res]))
    elif a.cmd == "video":                                                    # frames -> .npz (encode with tools/encode_mp4)
        vids = []
        for s in range(1000, 1000 + a.episodes):
            r, fr, n = run_episode(sim, s, a.noise, record=True); print(s, r, n)
            lab = "SUCCESS" if r["success"] else ("CUBE FELL" if r["cube_fell"] else "PILLAR TIPPED")
            vids += [banner(f, f"Pillar task, scripted expert, 2x Jenga execution noise   episode {s}   {lab}", r["success"]) for f in fr]
        np.savez_compressed(a.out, frames=np.stack(vids))
    elif a.cmd == "forkvideo":                                                # a real HOLD fork from results/pillar/fork_w8.json
        np.savez_compressed(a.out, frames=fork_clip(sim))



def banner(img, txt, ok):
    import cv2
    bar = np.full((28, img.shape[1], 3), (40, 150, 60) if ok else (190, 50, 40), np.uint8)
    cv2.putText(bar, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def fork_clip(sim):
    """Re-run the episode of a HOLD fork up to its chunk, then replay the chunk + hold with fresh 2x noise until one
    replay keeps the cube on top and one drops it; render exact | stays | falls side by side."""
    import json
    import uncertainty_monitor_dev as um
    rows = json.loads((ROOT / "results/pillar/fork_w8.json").read_text())["rows"]
    snippets = np.load(um.BENCH)["snippets"]
    for r in sorted((r for r in rows if r["hold_fork"]), key=lambda r: abs(r["hold_n_bad"] - 16)):
        seed, c0 = r["seed"], r["t"]
        reset(sim, seed); rng = np.random.default_rng([20261022, seed]); plan = expert_plan(sim)
        nz_main = [snippets[rng.integers(len(snippets))] * 2.0 for _ in range(len(plan) // 8 + 1)]
        for i in range(c0):
            a = plan[i].copy(); a[:3] += nz_main[i // 8][i % 8]; sim.execute(a)
        snap = sim.snapshot(); brng = np.random.default_rng(5); vids, ends = [], []
        for k in range(24):
            sim.restore(snap); fr = [sim.render()]
            nz = 0 if k == 0 else snippets[brng.integers(len(snippets))] * 2.0
            ch = plan[c0:c0 + 8].copy(); ch[:, :3] += nz if np.ndim(nz) == 0 else nz[:len(ch)]
            for a in list(ch) + [np.r_[ch[-1, :3], ch[-1, 3]]] * 20:
                sim.execute(a); fr.append(sim.render())
            vids.append(fr); ends.append(body_pos(sim, "cube")[2])
        ex = ends[0]
        fall = [k for k in range(1, 24) if ends[k] < ex - 0.02]; stay = [k for k in range(1, 24) if abs(ends[k] - ex) < 0.01]
        if fall and stay:
            i, j = stay[0], fall[0]
            out = [np.concatenate([banner(vids[0][t], "exact chunk", True), banner(vids[i][t], "noise draw A: cube stays", True),
                                   banner(vids[j][t], "noise draw B: cube falls", False)], 1) for t in range(len(vids[0]))]
            print("fork clip from episode", seed, "chunk start", c0); return np.stack([f for f in out for _ in range(3)])
    raise RuntimeError("no clip found")


def fork_job(args):
    """One episode: expert closed loop with noise_main; at every 8th action (chunk start) branch 33 ways:
    exact + 32 with fresh noise snippets x noise_branch on the next 8 actions; then either HOLD (20 actions at the last
    target, gripper unchanged) or REST (the rest of the expert plan without noise + 15 settle). Labels: task outcome."""
    seed, noise_main, noise_branch = args
    import uncertainty_monitor_dev as um
    os.environ.setdefault("MUJOCO_GL", "egl")
    sim = PillarSim(); reset(sim, seed)
    snippets = np.load(um.BENCH)["snippets"]; rng = np.random.default_rng([20261022, seed]); brng = np.random.default_rng([7, seed])
    plan = expert_plan(sim); pillar0 = body_pos(sim, "pillar"); rows = []
    nz_main = [snippets[rng.integers(len(snippets))] * noise_main for _ in range(len(plan) // 8 + 1)]
    for c0 in range(0, len(plan), 8):
        snap = sim.snapshot(); res = {"hold": [], "rest": []}; ends = []
        for mode in ("hold", "rest"):
            for k in range(33):
                sim.restore(snap)
                nz = 0 if k == 0 else snippets[brng.integers(len(snippets))] * noise_branch
                ch = plan[c0:c0 + 8].copy(); ch[:, :3] += nz if np.ndim(nz) == 0 else nz[:len(ch)]
                for a in ch:
                    sim.execute(a)
                tail = [np.r_[sim.target, ch[-1, 3]]] * 20 if mode == "hold" else list(plan[c0 + 8:]) + [np.r_[plan[-1, :3], -1.0]] * 15
                for a in tail:
                    sim.execute(a)
                if mode == "hold":                                        # hold: vs the exact replay's end (generic, as StackThree)
                    ends.append((body_pos(sim, "cube"), body_pos(sim, "pillar"), float(sim.block_diagnostics()["tilt"][1])))
                    res[mode].append(None)
                else:
                    res[mode].append(not outcome(sim, pillar0)["success"])
        sim.restore(snap)
        (c_ex, p_ex, t_ex) = ends[0]
        bad = [bool(c[2] < c_ex[2] - 0.02 or tl > t_ex + 45) for c, p, tl in ends[1:]]
        good = [bool(np.linalg.norm(c - c_ex) < 0.01 and np.linalg.norm(p - p_ex) < 0.01) for c, p, tl in ends[1:]]
        row = {"seed": seed, "t": c0, "T": len(plan), "hold_n_bad": int(sum(bad)), "hold_n_good": int(sum(good)),
               "hold_fork": bool(sum(bad) >= 2 and sum(good) >= 2)}
        nb = int(sum(res["rest"][1:])); row.update({"rest_n_bad": nb, "rest_bad_exact": bool(res["rest"][0]), "rest_fork": 2 <= nb <= 30})
        rows.append(row)
        ch = plan[c0:c0 + 8].copy(); ch[:, :3] += nz_main[c0 // 8][:len(ch)]   # the main (executed) episode continues
        for a in ch:
            sim.execute(a)
    return rows


def fork_check(episodes, noise_main, noise_branch, procs, out):
    import json
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(procs) as pool:
        rows = [r for rs in pool.map(fork_job, [(s, noise_main, noise_branch) for s in range(1000, 1000 + episodes)]) for r in rs]
    res = {"episodes": episodes, "noise_main": noise_main, "noise_branch": noise_branch, "points": len(rows)}
    for mode in ("hold", "rest"):
        nb = [r[f"{mode}_n_bad"] for r in rows]; fk = sum(r[f"{mode}_fork"] for r in rows); anyb = sum(x > 0 for x in nb)
        res[mode] = {"forks": fk, "points_any_bad": anyb, "all_or_nearly_all_bad": sum(x >= 31 for x in nb),
                     "pass": bool(fk >= 15 and fk >= 0.25 * max(anyb, 1) and fk >= 0.05 * len(rows))}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"result": res, "rows": rows}, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

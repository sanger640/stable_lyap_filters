"""Task T3: push an upright Jenga block into a goal zone without tipping it (user choice 2026-10-14; replaces T2 as the
paper's second task, T2 kept as backup).

One upright Jenga block (25 x 20 x 75 mm, as in the three-block task) on the table; a green goal zone (visual only)
60-110 mm ahead along the push direction (+y, image-lateral, as T2). The robot closes the gripper and pushes with the
fingertips at a randomised height; the push is aimed to bring the block centre to the zone centre. Whether the block
slides or tips depends on contact height vs friction (tips roughly when height > depth / (2 mu)), so perception error
(block pose) and execution noise (fingertip xyz) split outcomes near the critical height.

Labels: unsafe = the block tips (>= 45 deg) or leaves the table (any-outcome events with the block's own displacement
excluded, TARGET_MOVED_EVENT = False); task success = block centre ends inside the zone, upright.
`pilot` reports visibility, settling, fork rate and success rate on pilot seeds (never reused); `images` a strip.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_scenes as ps  # noqa: E402

ps.TARGET_MOVED_EVENT = False          # importing this module (any T3 worker) switches the event rule
ps.AT_REST_INCLUDE_TARGET = True

T3 = {"block_half": (0.0125, 0.010, 0.0375), "mass": (0.04, 0.06), "friction": (0.33, 0.37), "yaw_deg": 10.0,
      # pilot 3: tipping is set by height x friction (tips above ~9.5 mm); friction is invisible to the camera, so it is
      # held ~fixed (a known material) and forks come from pose/execution error near the critical height (~27 mm)
      "table_friction": 0.1,   # MuJoCo takes the max of the two geoms' friction: keep the table below the block's range
      "zone_ahead_mm": (60.0, 110.0), "zone_radius_mm": (25.0, 30.0), "push_height_mm": (18.0, 36.0),   # fingertip-site height (the fingers cannot go below ~13 mm); around the ~27 mm critical height
      "finger_lead_m": 0.026,          # leading finger's outer face is ~26 mm ahead of the fingertip site along +y
      "finger_x_m": 0.0,               # lateral aim offset: shifting it (-9..+19 mm) or biasing the push direction did not
                                       # remove the ~+16-19 mm sideways drift (the block yaws off the fingertip); the zone
                                       # radius is 25-30 mm instead (calibration 2026-10-14)
      "overshoot_m": 0.010,            # calibrated: sliding blocks stopped ~10 mm short
      "push_jitter_deg": 6.0, "action": {"offset_mm": 4.0, "prepush_mm": (15.0, 25.0), "step_mm": (3.0, 6.0)}}
# pilot 1 (table friction 1.0 dominated, heights 5-45 mm, 5-12 mm/command): 88% tipped, 0% in zone
PILOT_BASE = 990_300                  # pilot 4 (pilots 1-3: 990_000, 990_100, 990_100 recalibrated)


def sample_t3(seed):
    rng = np.random.default_rng([20261410, seed])
    block = {"shape": "box", "size": list(T3["block_half"]), "mass": float(rng.uniform(*T3["mass"])),
             "friction": float(rng.uniform(*T3["friction"])), "rgba": [0.85, 0.15, 0.12, 1.0]}
    bx, by = ps.CENTER[0] + rng.uniform(-0.01, 0.01), ps.CENTER[1] - 0.05 + rng.uniform(-0.005, 0.005)
    block["pos"] = [float(bx), float(by), ps.TABLE_TOP + T3["block_half"][2] + 0.0005]
    block["quat"] = ps._quat_z(np.radians(rng.uniform(-T3["yaw_deg"], T3["yaw_deg"]))).tolist()
    zone = {"pos": [float(bx + rng.uniform(-0.005, 0.005)), float(by + rng.uniform(*T3["zone_ahead_mm"]) / 1000)],
            "radius": float(rng.uniform(*T3["zone_radius_mm"]) / 1000)}
    return {"family": "T3", "seed": int(seed), "objects": [block], "wall": None, "primitive": "push", "zone": zone,
            "table_friction": T3["table_friction"],
            "push_yaw": float(np.pi / 2), "push_jitter_deg": T3["push_jitter_deg"], "action_override": T3["action"],
            "push_height_mm": float(rng.uniform(*T3["push_height_mm"])), "push_reach_m": T3["block_half"][1] + T3["finger_lead_m"],
            "push_site_offset_xy": [T3["finger_x_m"], 0.0], "push_overshoot_m": T3["overshoot_m"],
            "action_seed": int(rng.integers(2 ** 31))}


def in_zone(scene, end_state):
    """Task success: block centre inside the zone and upright (< 45 deg)."""
    z = scene["zone"]
    d = np.linalg.norm(end_state[0, :2] - np.asarray(z["pos"]))
    return bool(d <= z["radius"] and ps.tilt_deg(end_state[0, 3:]) < 45.0)


def pilot_scene(seed):
    import traceback
    try:
        return seed, _pilot(seed), None
    except Exception:
        return seed, None, traceback.format_exc()


def _pilot(seed):
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import chunk_starts
    scene = sample_t3(seed)
    sim = ps.make_sim(scene); sim.reset()
    ok, _, _ = ps.settle(sim, "T3")
    vis, alone = ps.visibility(sim)
    row = {"visible_ok": ps.visible_ok(sim, vis, alone), "settle_ok": bool(ok), "push_height_mm": scene["push_height_mm"],
           "friction": scene["objects"][0]["friction"]}
    if not (row["visible_ok"] and ok):
        return row
    acts = ps.primitive_actions(sim, scene)
    snippets = np.load(um.BENCH)["snippets"]; pos_cov, rot_cov = ps.pooled_error_model()
    starts = [s for s in chunk_starts(len(acts)) if s + 8 <= len(acts)]
    splits, rest = [], []
    for index, act in enumerate(acts):
        if index in starts:
            if not ps.at_rest(sim):
                rest.append(False)
            else:
                rest.append(True)
                snap = sim.snapshot(); warm = sim.data.qacc_warmstart.copy(); start = sim.object_state()
                evs = [ps.events(ps.future(sim, snap, acts[index:index + 8], snippets[k],
                                           ps.draw_error(1, pos_cov, rot_cov, PILOT_BASE, seed, index, k)), start) for k in range(8)]
                splits.append(ps.split_count(evs))
                sim.restore(snap); sim.data.qacc_warmstart[:] = warm
        sim.execute(act)
    for _ in range(20):
        sim.execute(np.r_[sim.target, 0.0])
    end = sim.object_state()
    row.update({"commands": len(acts), "splits": splits, "not_at_rest": int(sum(not r for r in rest)),
                "nominal_tipped": bool(ps.tilt_deg(end[0, 3:]) >= 45), "nominal_in_zone": in_zone(scene, end)})
    return row


def pilot(n=48, workers=12):
    import multiprocessing as mp
    rows = {}
    with mp.get_context("spawn").Pool(workers) as p:
        for seed, row, err in p.imap_unordered(pilot_scene, [PILOT_BASE + i for i in range(n)]):
            if err:
                print(err, flush=True)
            else:
                rows[seed] = row
    valid = [r for r in rows.values() if r["visible_ok"] and r["settle_ok"]]
    sp = [c for r in valid for c in r["splits"]]
    out = {"scenes": len(rows), "valid": len(valid), "chunk_starts": len(sp),
           "fork_rate_1_to_7_of_8": float(np.mean([0 < c < 8 for c in sp])) if sp else None,
           "scenes_with_a_fork": float(np.mean([any(0 < c < 8 for c in r["splits"]) for r in valid])) if valid else None,
           "nominal_tipped": float(np.mean([r["nominal_tipped"] for r in valid])) if valid else None,
           "nominal_in_zone": float(np.mean([r["nominal_in_zone"] for r in valid])) if valid else None,
           "not_at_rest_starts": int(sum(r["not_at_rest"] for r in valid)),
           "tip_vs_height": sorted((round(r["push_height_mm"]), round(r["friction"], 2), r["nominal_tipped"]) for r in valid),
           "rows": rows}
    (ROOT / "results/jenga/t3_pilot.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "tip_vs_height")}, indent=1), flush=True)
    print("height, friction, tipped:", out["tip_vs_height"], flush=True)


def images(n=4):
    import mujoco
    from PIL import Image
    from jenga_render_frames import close_camera
    tiles = []
    for i in range(n):
        sc = sample_t3(PILOT_BASE + i); sim = ps.make_sim(sc); sim.reset(); ps.settle(sim, "T3")
        r = mujoco.Renderer(sim.model, height=256, width=256)
        for when in ("start", "end"):
            if when == "end":
                for a in ps.primitive_actions(sim, sc):
                    sim.execute(a)
                for _ in range(20):
                    sim.execute(np.r_[sim.target, 0.0])
            r.update_scene(sim.data, camera=close_camera(mujoco)); tiles.append(r.render().copy())
        r.close()
    Image.fromarray(np.concatenate([np.concatenate(tiles[2 * k:2 * k + 2], 0) for k in range(n)], 1)).save(ROOT / "results/jenga/t3_scenes.png")


if __name__ == "__main__":
    {"pilot": pilot, "images": images}[sys.argv[1]]()

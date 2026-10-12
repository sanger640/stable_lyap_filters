"""Task T2: push a box past tall, narrow objects (paper experiment E4). Fixed task structure, randomised poses/sizes.

Scenes reuse eval/phase2_scenes.py (MJCF builder, Phase2Sim, push primitive, harm label). The push direction is fixed
(+y, image-lateral) with small jitter; one or two tall cylinders/slabs stand beside the path with 0-15 mm clearance.
`pilot` reports visibility, settling and the fork rate on pilot seeds (never reused).
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

T2 = {"box_half": ((0.016, 0.022), (0.016, 0.022), (0.014, 0.022)), "box_mass": (0.08, 0.20),
      "tall_r": (0.008, 0.012), "tall_hh": (0.035, 0.055), "tall_mass": (0.02, 0.06), "clear_mm": (0.0, 15.0),
      "push_jitter_deg": 10.0, "action": {"offset_mm": 6.0, "push_mm": (40.0, 70.0), "prepush_mm": (15.0, 25.0)}}
PILOT_BASE = 980_000


def sample_t2(seed):
    rng = np.random.default_rng([20261310, seed])
    box = {"shape": "box", "size": [float(rng.uniform(*r)) for r in T2["box_half"]],
           "mass": float(rng.uniform(*T2["box_mass"])), "friction": float(rng.uniform(0.5, 1.0)),
           "rgba": [*ps.PALETTE[rng.integers(len(ps.PALETTE))], 1.0]}
    bx, by = ps.CENTER[0] + rng.uniform(-0.01, 0.01), ps.CENTER[1] - 0.035 + rng.uniform(-0.005, 0.005)
    box["pos"] = [float(bx), float(by), ps.TABLE_TOP + box["size"][2] + 0.0005]
    box["quat"] = ps._quat_z(np.radians(rng.uniform(-5, 5))).tolist()
    objs = [box]
    sides = [1] if rng.random() < 0.5 else [-1, 1]
    if len(sides) == 1:
        sides = [int(rng.choice([-1, 1]))]
    for side in sides:
        r, hh = rng.uniform(*T2["tall_r"]), rng.uniform(*T2["tall_hh"])
        o = {"shape": "cylinder", "size": [float(r), float(hh)], "mass": float(rng.uniform(*T2["tall_mass"])),
             "friction": float(rng.uniform(0.5, 1.0)), "rgba": [*ps.PALETTE[rng.integers(len(ps.PALETTE))], 1.0]}
        x = bx + side * (box["size"][0] + r + rng.uniform(*T2["clear_mm"]) / 1000)
        y = by + rng.uniform(0.02, 0.06)                    # beside the path, ahead of the box
        o["pos"] = [float(x), float(y), ps.TABLE_TOP + hh + 0.0005]; o["quat"] = [1.0, 0.0, 0.0, 0.0]
        objs.append(o)
    return {"family": "T2", "seed": int(seed), "objects": objs, "wall": None, "primitive": "push",
            "push_yaw": float(np.pi / 2), "push_jitter_deg": T2["push_jitter_deg"], "action_override": T2["action"],
            "action_seed": int(rng.integers(2 ** 31))}


def pilot_scene(seed):
    import traceback
    try:
        return seed, _pilot(seed), None
    except Exception:
        return seed, None, traceback.format_exc()


def _pilot(seed):
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import chunk_starts
    scene = sample_t2(seed)
    sim = ps.make_sim(scene); sim.reset()
    ok, disp, dtilt = ps.settle(sim, "T2")
    vis, alone = ps.visibility(sim)
    row = {"visible_ok": ps.visible_ok(sim, vis, alone), "settle_ok": bool(ok), "n_tall": len(scene["objects"]) - 1}
    if not (row["visible_ok"] and ok):
        return row
    acts = ps.primitive_actions(sim, scene)
    snippets = np.load(um.BENCH)["snippets"]; pos_cov, rot_cov = ps.pooled_error_model()
    starts = [s for s in chunk_starts(len(acts)) if s + 8 <= len(acts)]
    counts = []
    for index, act in enumerate(acts):
        if index in starts:
            snap = sim.snapshot(); warm = sim.data.qacc_warmstart.copy(); start = sim.object_state()
            counts.append(int(sum(ps.harm(ps.future(sim, snap, acts[index:index + 8], snippets[k],
                                                    ps.draw_error(len(scene["objects"]), pos_cov, rot_cov, PILOT_BASE, seed, index, k)), start)
                                  for k in range(8))))
            sim.restore(snap); sim.data.qacc_warmstart[:] = warm
        sim.execute(act)
    row.update({"commands": len(acts), "harm_counts": counts})
    return row


def pilot(n=40, workers=8):
    import multiprocessing as mp
    rows = {}
    with mp.get_context("spawn").Pool(workers) as p:
        for seed, row, err in p.imap_unordered(pilot_scene, [PILOT_BASE + i for i in range(n)]):
            if err:
                print(err, flush=True)
            else:
                rows[seed] = row
    valid = [r for r in rows.values() if r["visible_ok"] and r["settle_ok"]]
    hc = [c for r in valid for c in r["harm_counts"]]
    scene_fork = [any(0 < c < 8 for c in r["harm_counts"]) for r in valid]
    out = {"scenes": len(rows), "valid": len(valid), "chunk_starts": len(hc),
           "fork_rate_1_to_7_of_8": float(np.mean([0 < c < 8 for c in hc])) if hc else None,
           "always_harm_rate": float(np.mean([c == 8 for c in hc])) if hc else None,
           "scenes_with_a_fork": float(np.mean(scene_fork)) if valid else None, "rows": rows}
    (ROOT / "results/jenga/t2_pilot.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)


def images(n=4):
    import mujoco
    from PIL import Image
    from jenga_render_frames import close_camera
    tiles = []
    for i in range(n):
        sc = sample_t2(PILOT_BASE + i); sim = ps.make_sim(sc); sim.reset(); ps.settle(sim, "T2")
        r = mujoco.Renderer(sim.model, height=256, width=256); r.update_scene(sim.data, camera=close_camera(mujoco))
        tiles.append(r.render().copy()); r.close()
    Image.fromarray(np.concatenate(tiles, 1)).save(ROOT / "results/jenga/t2_scenes.png")


if __name__ == "__main__":
    {"pilot": pilot, "images": images}[sys.argv[1]]()

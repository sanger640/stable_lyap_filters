"""Task 2, knock-down (results/jenga/knock_protocol.json): the Jenga scene with the opposite goal for the middle block.
The robot must push the red middle block over (an INTENDED irreversible change) without toppling either neighbour (an
unintended one). Same robot, controller, fixed camera, blocks and execution-noise model as Jenga; only the policy and the
success rule differ. The gripper points along +x with its fingers closed, so it is a ~1.5 cm wide horizontal pusher;
its tips sit ~1 cm ahead of the attachment site.

Policy: a scripted expert with Jenga's interface (16 absolute [x, y, z, gripper] targets per decision, the first 8
executed, re-planned every chunk). It perceives the red block's pose with an error (per-episode offset + per-decision
jitter), as the pillar task's expert (docs/PILLAR_TASK.md).

usage: python3 eval/knock_task.py pilot <depth_mm> <n> <seed0> [noise]   (exploratory)
       python3 eval/knock_task.py select                                (registered width rule, seeds 40000+)
       python3 eval/knock_task.py fork <n> <seed0> [control]             (registered fork check)"""
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / "results/knock"
TOPPLE_DEG = 45.0
STEP_M = 0.012                       # <= 12 mm per 0.1 s target, as the pillar expert
TIP_M = 0.010                        # fingertips ahead of the attachment site along +x
HALF_X, TOP_Z = 0.0125, 0.0375       # block half-depth along x, half-height
PUSH_BELOW_TOP = 0.010               # site height: 1 cm below the block's top
PERC_SIGMA = 0.0018                  # per-axis perception sd (2.5 mm 2-D error, Jenga's measured head)
MAX_CHUNKS = 15                      # 12 s
DEPTH = float(os.environ.get("KNOCK_DEPTH_MM", 40)) / 1000


def make_sim():
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    return DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="knock_"))))


def tilts(sim):
    """tilt (deg) of [middle, left, right]"""
    out = []
    for b in sim.tracked_block_ids:
        R = sim.data.xmat[b].reshape(3, 3)
        out.append(float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))))
    return np.array(out)


def settle(sim):
    for _ in range(10):
        sim.execute(np.r_[sim.target, 1.0].astype(np.float32))      # blocks land; gripper closes (it stays closed)


class Expert:
    """Re-plans from the current state every chunk. Phases: approach (rise, move behind the block, descend) -> push along
    +x by DEPTH past the perceived near face -> retreat once the block is pushed (or has fallen) -> hold."""

    def __init__(self, ep_seed, depth=None, sigma=PERC_SIGMA):
        self.rng = np.random.default_rng([20261104, int(ep_seed)])
        self.bias = self.rng.normal(0, sigma, 2); self.sigma = sigma
        self.depth = DEPTH if depth is None else depth
        self.pushed = False

    def perceive(self, sim):
        p = sim.data.xpos[sim.tracked_block_ids[0]][:2].copy()
        return p + self.bias + self.rng.normal(0, self.sigma / 3, 2)

    def plan(self, sim):
        site = sim.data.site_xpos[sim.site].copy()
        px, py = self.perceive(sim)
        z0 = sim._z0
        zp = z0 + TOP_Z - PUSH_BELOW_TOP
        face = px - HALF_X - TIP_M                                   # site x with the tips touching the near face
        red_tilt = tilts(sim)[0]
        if red_tilt >= TOPPLE_DEG or site[0] >= face + self.depth - 0.002:
            self.pushed = True
        if self.pushed:                                              # retreat back and up, then hold
            wps = [np.array([face - 0.05, py, zp + 0.04])]
        elif site[0] < face - 0.004 and (abs(site[1] - py) > 0.003 or abs(site[2] - zp) > 0.003):
            above = np.array([face - 0.02, py, zp + 0.03])          # line up behind the block, above push height
            if abs(site[1] - py) > 0.003 or site[2] > zp + 0.01:
                wps = [above, np.array([face - 0.02, py, zp])]
            else:
                wps = [np.array([face - 0.02, py, zp])]
        else:                                                        # push along +x
            wps = [np.array([face + self.depth, py, zp])]
        tg, cur = [], site.copy()
        for w in wps:
            while len(tg) < 16:
                d = w - cur; n = np.linalg.norm(d)
                if n < 1e-4:
                    break
                cur = cur + d * min(1.0, STEP_M / n); tg.append(cur.copy())
        while len(tg) < 16:
            tg.append(cur.copy())
        return np.c_[np.array(tg[:16]), np.ones(16)].astype(np.float32)


def episode(ep, depth=None, noise_scale=2.0, record=None, chunk_hook=None):
    depth = DEPTH if depth is None else depth
    import uncertainty_monitor_dev as um
    snippets = np.load(um.BENCH)["snippets"]; nrng = np.random.default_rng([20261502, ep])
    sim = make_sim()
    try:
        sim.reset(int(ep)); settle(sim)
        sim._z0 = float(sim.data.xpos[sim.tracked_block_ids[0]][2])
        pol = Expert(ep, depth)
        peak_n = 0.0; red_t = None
        for k in range(MAX_CHUNKS):
            plan = pol.plan(sim); chunk = plan[:8].copy(); info = {}
            if record is not None:
                record.append({"k": k, "snap": sim.snapshot(), "chunk": chunk.copy(), "plan": plan.copy(),
                               "pushed": pol.pushed})
            if chunk_hook is not None:
                chunk, info = chunk_hook(sim, chunk, k, plan)
            noise = snippets[nrng.integers(len(snippets))] * noise_scale
            for j in range(8):
                a = chunk[j].copy(); a[:3] -= noise[j]; sim.execute(a)
                t = tilts(sim); peak_n = max(peak_n, float(t[1:].max()))
                if red_t is None and t[0] >= TOPPLE_DEG:
                    red_t = round((k * 8 + j + 1) * 0.1, 1)
            if pol.pushed and red_t is not None and k >= 2 and np.linalg.norm(sim.data.site_xpos[sim.site][0] - chunk[-1][0]) < 0.01:
                break
        for _ in range(20):                                          # settle 2 s with the arm holding
            sim.execute(np.r_[sim.target, 1.0].astype(np.float32))
            t = tilts(sim); peak_n = max(peak_n, float(t[1:].max()))
        t = tilts(sim)
        red_down = bool(t[0] >= TOPPLE_DEG)
        res = {"ep": int(ep), "chunks": k + 1, "red_down": red_down, "red_down_s": red_t, "red_final_tilt": round(float(t[0]), 1),
               "neighbour_toppled": bool(peak_n >= TOPPLE_DEG), "peak_neighbour_tilt": round(peak_n, 1)}
        res["success"] = red_down and not res["neighbour_toppled"]
        return res
    finally:
        sim.close()


def pilot(depth_mm, n, seed0, nz=2.0):
    rows = [episode(seed0 + i, depth_mm / 1000, nz) for i in range(n)]
    s = {"depth_mm": depth_mm, "noise": nz, "n": n, "success": sum(r["success"] for r in rows),
         "red_down": sum(r["red_down"] for r in rows), "neighbour_toppled": sum(r["neighbour_toppled"] for r in rows)}
    print(json.dumps(s), flush=True)
    return s, rows


def select():
    """Registered depth rule (results/jenga/knock_protocol.json stage0.depth_rule)."""
    res = {}
    for d in (40, 45, 50, 55):
        s, rows = pilot(d, 40, 40000, 2.0); res[d] = {**s, "rate": s["success"] / 40}
    band = [d for d in res if 0.6 <= res[d]["rate"] <= 0.9]
    pick = max(band) if band else min(res, key=lambda d: min(abs(res[d]["rate"] - 0.6), abs(res[d]["rate"] - 0.9)))
    out = {"protocol": "results/jenga/knock_protocol.json", "grid": res, "chosen_depth_mm": pick, "in_band": bool(band)}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "select.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"chosen_depth_mm": pick, "in_band": bool(band)}), flush=True)


def fork(n, seed0, control=False):
    """Registered fork check: 32 copies per chunk with fresh 2x snippets (none in the control), + 20 hold steps."""
    import uncertainty_monitor_dev as um
    snippets = np.load(um.BENCH)["snippets"]
    depth = json.loads((OUT / "select.json").read_text())["chosen_depth_mm"] / 1000
    pts, eps = [], []
    for e in range(seed0, seed0 + n):
        rec = []; res = episode(e, depth, 2.0, record=rec); eps.append(res)
        sim = make_sim(); rng = np.random.default_rng([20261105, e])
        try:
            for r in rec:
                nb, rd = [], []
                for c in range(32):
                    sim.restore(r["snap"]); sim._z0 = None
                    noise = np.zeros((8, 3)) if control else snippets[rng.integers(len(snippets))] * 2.0
                    peak = float(tilts(sim)[1:].max())
                    for j in range(8):
                        a = r["chunk"][j].copy(); a[:3] -= noise[j]; sim.execute(a); peak = max(peak, float(tilts(sim)[1:].max()))
                    for _ in range(20):
                        sim.execute(np.r_[sim.target, 1.0].astype(np.float32)); peak = max(peak, float(tilts(sim)[1:].max()))
                    nb.append(peak >= TOPPLE_DEG); rd.append(tilts(sim)[0] >= TOPPLE_DEG)
                pts.append({"ep": e, "k": r["k"], "neigh": int(sum(nb)), "red": int(sum(rd)), "pushed": bool(r["pushed"])})
        finally:
            sim.close()
        print(e, res["success"], [(p["neigh"], p["red"]) for p in pts if p["ep"] == e], flush=True)
    P = len(pts); anyb = [p for p in pts if p["neigh"] > 0]; fk = [p for p in pts if 0 < p["neigh"] < 32]
    redany = [p for p in pts if p["red"] > 0]
    out = {"protocol": "results/jenga/knock_protocol.json", "control": control, "depth_mm": depth * 1000, "episodes": n,
           "episode_success": sum(r["success"] for r in eps), "episode_neighbour_toppled": sum(r["neighbour_toppled"] for r in eps),
           "points": P, "points_any_neighbour": len(anyb), "forks": len(fk), "all_bad": sum(p["neigh"] == 32 for p in pts),
           "fork_share": round(len(fk) / P, 4), "fork_share_of_any": round(len(fk) / max(len(anyb), 1), 4),
           "red_points_any": len(redany), "red_points_all32": sum(p["red"] == 32 for p in redany),
           "red_points_some": sum(0 < p["red"] < 32 for p in redany)}
    if not control:
        sel = json.loads((OUT / "select.json").read_text())
        out["bars"] = {"B1_success_in_band": sel["in_band"], "B2_neighbour_topple_ge5pct": out["episode_neighbour_toppled"] / n >= 0.05,
                       "B3_forks_ge50pct_of_any": out["fork_share_of_any"] >= 0.5, "B4_forks_ge3pct": out["fork_share"] >= 0.03}
    (OUT / f"fork{'_control' if control else ''}.json").write_text(json.dumps({**out, "points_rows": pts, "episodes_rows": eps}, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "pilot":
        s, rows = pilot(float(a[1]), int(a[2]), int(a[3]), float(a[4]) if len(a) > 4 else 2.0)
        for r in rows[:5]:
            print(r)
    elif a[0] == "select":
        select()
    elif a[0] == "fork":
        fork(int(a[1]), int(a[2]), len(a) > 3 and a[3] == "control")

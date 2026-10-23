"""Optimisation safety filter on the full-token DINO-WM (docs/FILTER_DEEP_DIVE.md step 4). Task-agnostic, label-free.

Monitor (eval/tokmon.py): sensitivity of the imagined outcome to action perturbations only. Fires on the policy's plan if
  h < H_SAFE (its futures fork at <= 2x the robot's execution noise) or g > G_MAX (sensitivity gain) or o > O_MAX
  (outside-hand disturbance). G_MAX, O_MAX: label-free percentiles over the policy's own plans (calibrate stage).
Filter, CBF-style optimisation over the POLICY'S ACTIONS (no hold / primitive / backup strategies):
  min_theta  cost(edit_theta(plan))  s.t.  h >= H_SAFE,  g <= G_MAX,  o <= O_MAX
  edit_theta = time-warp (speed v, pause k) + smooth xyz knot offsets (steps 1, 5, 10, 15), boxed to the range the world
  model was trained on (eval/tok_data.py edits) = trust region; cost = opt2's mm^2 + ETA * second differences on steps 1-8.
  Solver: CEM over theta (search: g, o at scale 1 with 16 futures, draws 0-15; violation-ranked). Certificate on the 4
  finals: h at all scales + g + o with 32 FRESH draws (32-63). Cheapest certified; if none certifies, the least-violating
  final (slack, as a CBF-QP with slack), never a fixed fallback.
usage: python3 eval/tokfilter.py calibrate <model.pt> ; run <dev|holdout|relpass> <model.pt> <tag> ; report <set> <tag>
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
OUT = ROOT / "results/jenga/tokfilter"
H_SAFE = float(np.log2(3.0))
KNOTS = (1, 5, 10, 15)
LO = np.r_[0.3, 0.0, [-15.0] * 12]                 # v, k, 4 knots x (x, y, z) mm (z clipped to half below)
HI = np.r_[1.2, 4.0, [15.0] * 12]
IDENT = np.r_[1.0, 0.0, [0.0] * 12]
CEM = {"POP": 24, "ITERS": 3, "ELITES": 6, "FINAL": 4, "SD": np.r_[0.25, 1.5, [5.0] * 12]}
ETA = 4.0


def edit(p16, th):
    """p16 (16, 4), th (S, 14) -> (S, 16, 4): time-warp then knot offsets (z at half), gripper on the policy's clock."""
    th = np.atleast_2d(th); S = len(th); out = np.repeat(p16[None].astype(np.float64), S, 0)
    t = np.arange(16)
    for c in range(S):
        v, k = th[c, 0], th[c, 1]
        tau = np.clip(v * np.maximum(t - k, 0), 0, 15); i = np.clip(np.floor(tau).astype(int), 0, 14); w = (tau - i)[:, None]
        q = p16[i, :3] * (1 - w) + p16[i + 1, :3] * w
        kn = th[c, 2:].reshape(4, 3) * [1, 1, 0.5] / 1000
        off = np.stack([np.interp(t, KNOTS, kn[:, d], left=0.0) for d in range(3)], 1); off[0] = 0
        out[c, :, :3] = q + off
    out[:, 0, :3] = p16[0, :3]
    return out.astype(np.float32)


def cost(q, p):
    mm = (q[1:9, :3] - p[1:9, :3]) * 1000
    sd = np.diff(np.vstack([np.zeros((1, 3)), mm]), 2, axis=0)
    return float((mm ** 2).sum() + ETA * (sd ** 2).sum()) / 100


class Filter:
    def __init__(self, model_path, calib=None):
        import tokmon
        self.mon = tokmon.TokMon(model_path)
        c = json.loads(Path(calib or OUT / "calib.json").read_text()) if (calib or (OUT / "calib.json").exists()) else {}
        self.G_MAX, self.O_MAX = c.get("G_MAX"), c.get("O_MAX")

    def viol(self, r, j):
        """Normalised constraint violation (0 = feasible) of candidate j in score dict r."""
        v = max(0.0, r["g"][j] / self.G_MAX - 1) + max(0.0, r["o"][j] / self.O_MAX - 1)
        if "h_cert" in r:
            v += max(0.0, H_SAFE - r["h_cert"][j])
        return v

    def decide(self, vis0, prop0, p16, site, seed):
        mon = self.mon; keep = mon.hand_keep(site, p16); t0 = time.time()
        nom = mon.score(vis0, prop0, p16[None], keep, n=32)
        row = {"h_nom": float(nom["h"][0]), "g_nom": float(nom["g"][0]), "o_nom": float(nom["o"][0])}
        row["trig"] = bool(nom["h"][0] < H_SAFE or nom["g"][0] > self.G_MAX or nom["o"][0] > self.O_MAX)
        if not row["trig"]:
            row["kind"] = "pass"; row["sec"] = round(time.time() - t0, 1)
            return p16, row
        rng = np.random.default_rng(seed); mu, sd, pool = IDENT.copy(), CEM["SD"].copy(), []
        for it in range(CEM["ITERS"]):
            T = np.clip(mu + sd * rng.standard_normal((CEM["POP"], len(mu))), LO, HI); T[0] = mu if it else IDENT
            Q = edit(p16, T); r = mon.score(vis0, prop0, Q, keep, n=16, scales=(1.0,))
            key = [(self.viol(r, c), cost(Q[c], p16)) for c in range(len(T))]
            order = sorted(range(len(T)), key=lambda c: key[c])
            el = T[order[:CEM["ELITES"]]]; mu, sd = el.mean(0), np.maximum(el.std(0), 0.03 * (HI - LO))
            pool += [(key[c], T[c]) for c in order[:CEM["FINAL"]]]
        pool.sort(key=lambda x: x[0]); finals = np.array([th for _, th in pool[:CEM["FINAL"]]])
        Q = edit(p16, finals)
        rc = mon.score(vis0, prop0, Q, keep, n=32, offset=32)
        rc["h_cert"] = rc["h"]; vi = np.array([self.viol(rc, j) for j in range(len(Q))]); co = np.array([cost(q, p16) for q in Q])
        if (vi == 0).any():
            j = int(np.flatnonzero(vi == 0)[np.argmin(co[vi == 0])]); row["kind"] = "certified"
        else:
            j = int(np.argmin(vi)); row["kind"] = "slack"
        row.update({"theta": [round(float(x), 3) for x in finals[j]], "viol": round(float(vi[j]), 4), "cost": round(float(co[j]), 2),
                    "h_out": float(rc["h"][j]), "g_out": float(rc["g"][j]), "o_out": float(rc["o"][j]), "sec": round(time.time() - t0, 1)})
        return Q[j], row


def _state(C, i, sim, mon_frame):
    sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
    prop = np.r_[sim.data.site_xpos[sim.site], float(sim.gripper)].astype(np.float32)
    return mon_frame(sim), prop, sim.data.site_xpos[sim.site].copy()


def calibrate(model_path, n=300):
    """G_MAX = 90th, O_MAX = 99th percentile of g, o of the policy's own plan at random Stage 4 policy-alone decisions."""
    import cem_filter as cf
    import inspector_data as insp
    import tokmon
    mon = tokmon.TokMon(model_path); fr = cf.Monitor.__new__(cf.Monitor); fr.renderer = None; sim = insp._sim()
    run = ROOT / "results/program_b/policy_rollouts/paired_record_s4"; rng = np.random.default_rng(20261060); rows = []
    files = sorted(run.glob("ep*.npz"))
    for f in [files[i] for i in rng.choice(len(files), n, replace=True)]:
        with np.load(f) as S:
            d = int(rng.integers(len(S["chunk_index"]))); C = {k: S[k][d:d + 1] for k in ("state", "ctrl", "target", "gripper", "plan16")}
        frame, prop, site = _state(C, 0, sim, lambda s: cf.Monitor.frame(fr, s))
        vis0, prop0 = mon.start(frame, prop); p16 = C["plan16"][0].copy(); p16[0, :3] = C["target"][0]
        r = mon.score(vis0, prop0, p16[None], mon.hand_keep(site, p16), n=32)
        rows.append({"h": float(r["h"][0]), "g": float(r["g"][0]), "o": float(r["o"][0])})
    g = np.array([r["g"] for r in rows]); o = np.array([r["o"] for r in rows]); h = np.array([r["h"] for r in rows])
    out = {"G_MAX": float(np.percentile(g, 90)), "O_MAX": float(np.percentile(o, 99)), "n": n, "fork_rate": float((h < H_SAFE).mean()),
           "source": "label-free: 90th pct of g, 99th pct of o over the policy's own plans, random Stage 4 policy-alone decisions",
           "model": str(model_path)}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "calib.json").write_text(json.dumps(out, indent=1)); print(out)


def run(name, model_path, tag):
    import bench2 as b2
    import cem_filter as cf
    import inspector_data as insp
    C, R = b2.load(name); n = int(os.environ.get("TOK_LIMIT", len(R)))
    F = Filter(model_path); fr = cf.Monitor.__new__(cf.Monitor); fr.renderer = None; sim = insp._sim(); rows = []
    for i in range(n):
        frame, prop, site = _state(C, i, sim, lambda s: cf.Monitor.frame(fr, s))
        vis0, prop0 = F.mon.start(frame, prop)
        p16 = C["plan16"][i].copy(); p16[0, :3] = C["target"][i]
        plan, row = F.decide(vis0, prop0, p16, site, 7000 + i)
        rows.append({"i": i, **row, "chunk": plan[1:9].astype(np.float32).tolist()})
        if i % 10 == 0:
            print(name, i, row["kind"], row.get("sec"), flush=True)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"run_{name}_{tag}.json").write_text(json.dumps({"rows": rows}))


def report(name, tag):
    import bench2 as b2
    rows = json.loads((OUT / f"run_{name}_{tag}.json").read_text())["rows"]; C, R = b2.load(name); n = len(rows)
    tok = np.array(b2.truth_chunks(name, [(r["i"], np.asarray(r["chunk"], np.float32)) for r in rows]))
    pol = np.array(b2.truth_chunks(name, [(i, C["nominal"][i]) for i in range(n)]))
    hold = np.array(b2.truth_chunks(name, [(i, b2.hold_chunk(C, i)) for i in range(n)]))
    out = {"set": name, "decisions": n, "kinds": {k: sum(r["kind"] == k for r in rows) for k in ("pass", "certified", "slack")},
           "neighbour_risk": {"tok": float(tok[:, 0].mean()), "policy": float(pol[:, 0].mean()), "hold": float(hold[:, 0].mean())},
           "red_fall_risk": {"tok": float(tok[:, 1].mean()), "policy": float(pol[:, 1].mean()), "hold": float(hold[:, 1].mean())}}
    if name != "relpass":
        sel = np.array(b2.truth_chunks(name, [(i, C["executed"][i]) for i in range(n)])); out["neighbour_risk"]["selection"] = float(sel[:, 0].mean())
    g = OUT.parent.parent / "program_b/bench2" / f"opt3_{name}_g_cem.json"
    if g.exists():
        G = json.loads(g.read_text())["rows"][:n]
        gr = np.array(b2.truth_chunks(name, [(r["i"], np.asarray(r["chunk"], np.float32)) for r in G])); out["neighbour_risk"]["g_cem"] = float(gr[:, 0].mean())
        d = tok[:, 0] - gr[:, 0]; out["tok_minus_gcem"] = {"mean": float(d.mean()), "better": int((d < 0).sum()), "worse": int((d > 0).sum())}
    d = tok[:, 0] - pol[:, 0]; out["tok_minus_policy"] = {"mean": float(d.mean()), "better": int((d < 0).sum()), "worse": int((d > 0).sum())}
    (OUT / f"report_{name}_{tag}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    st = sys.argv[1]
    {"calibrate": lambda: calibrate(sys.argv[2]), "run": lambda: run(*sys.argv[2:5]), "report": lambda: report(*sys.argv[2:4])}[st]()

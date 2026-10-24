"""Exploratory (docs/FILTER_DEEP_DIVE.md 5f, release falls): the fork test cannot see a failure that happens in EVERY
imagined future (no spread), and the outside-hand barrier masks the carried block. Label-free stability readout: after the
imagined robot has stopped (the hold tail, model steps HOLD10 -> HOLD30), a stable scene should stop changing.
  settle      = max over futures of the mean squared latent change HOLD10 -> HOLD30 (whole frame)
  settle_mean = mean over futures of the same;  settle_hand / settle_out = restricted to / outside the hand mask
Decisions: benchmark v2 dev + holdout (all), candidates policy / hold / G-CEM / O-CEM's 5 finals (cached one-chunk truth),
model_OIL, 64 futures at scale 1, common random numbers across candidates. Reports AUC vs red fall and neighbour topple
(per candidate, by gripper phase) and within-decision concordance.
usage: python3 eval/settle_check.py run | analyse -> results/jenga/deep_dive/settle_rows.json, settle_check.json
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CBF_WM", str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"))
os.environ.setdefault("CBF_CALIB", "results/jenga/cbf_calibration_OIL.json")
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
import bench2 as b2  # noqa: E402
from red_readout import p16_of  # noqa: E402

D = ROOT / "results/jenga/deep_dive"
KEYS = ("settle", "settle_mean", "settle_hand", "settle_out")


def readouts(cbf, sim, z0, plans, seed, n_fut=64):
    import cbf_barrier as cb
    import cbf_filter as cbff
    import step3_world_model as wm
    mon, torch = cbf.mon, cbf.mon.torch
    C = len(plans); r = {k: np.zeros(C) for k in KEYS}
    d_ = np.stack([np.concatenate([(p * [1, 1, 0]).reshape(9), (q * [0, 0, 1]).reshape(9)])
                   for p, q in (mon.errors.sample(seed, 0, 0, j) for j in range(n_fut))])
    st = mon.g(z0[None].expand(n_fut, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32))
    acts = []
    for p in plans:
        base = cbff.horizon(p, cbff.CFG["N_LOOK"])
        for j in range(n_fut):
            a = base.copy()
            a[:cbff.CFG["N_LOOK"], :3] -= np.concatenate([mon.snippets[j % 64], mon.snippets[(j + 1) % 64]])[:cbff.CFG["N_LOOK"]]
            acts.append(a)
    pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(C * n_fut, wm.STEPS, -1)
    z = st.float().repeat(C, 1, 1, 1); ends = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for t in range(wm.STEPS):
            z = mon.model.step(z, pairs[:, t]).float()
            if t in (wm.HOLD10, wm.HOLD30):
                ends[t] = z.reshape(C, n_fut, 28, 28, 8)
    for c in range(C):
        pts = np.vstack([sim.data.site_xpos[sim.site][None], np.asarray(plans[c])[1:16, :3]])
        mh = torch.as_tensor(cb.hand_mask(pts), device="cuda").float()
        diff = (ends[wm.HOLD30][c] - ends[wm.HOLD10][c]).pow(2).mean(-1)          # (n_fut, 28, 28)
        full = diff.mean((1, 2))
        r["settle"][c], r["settle_mean"][c] = float(full.max()), float(full.mean())
        r["settle_hand"][c] = float(((diff * mh).sum((1, 2)) / mh.sum().clamp_min(1)).max())
        r["settle_out"][c] = float(((diff * (1 - mh)).sum((1, 2)) / (1 - mh).sum().clamp_min(1)).max())
    return {k: v.tolist() for k, v in r.items()}


def run():
    import torch
    import cbf_filter as cbff
    import cem_filter as cf
    import inspector_data as insp
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim(); rows = []
    for s in ("dev", "holdout"):
        C, _ = b2.load(s); cache = json.loads((b2.OUT / f"truth_{s}.json").read_text())
        O = json.loads((b2.OUT / f"opt4_{s}_o1.json").read_text())["rows"]
        G = json.loads((b2.OUT / f"opt3_{s}_g_cem.json").read_text())["rows"]
        for i in range(len(O)):
            chunks = {"policy": C["nominal"][i], "hold": b2.hold_chunk(C, i), "g_cem": np.asarray(G[i]["chunk"], np.float32)}
            chunks.update({f"o{j}": np.asarray(ch, np.float32) for j, ch in enumerate(O[i]["chunks"])})
            names = [k for k in chunks if b2._hash(i, np.asarray(chunks[k], np.float32)) in cache]
            truth = [cache[b2._hash(i, np.asarray(chunks[k], np.float32))] for k in names]
            sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
            z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
            mon.rng = np.random.default_rng([20261128, i])
            ro = readouts(cbf, sim, z0, np.stack([p16_of(C, i, chunks[k]) for k in names]), int(mon.rng.integers(2 ** 31)))
            p16 = C["plan16"][i]; closed = bool(C["gripper"][i] < 50)
            rows.append({"set": s, "i": i, "closed": closed, "release": bool(closed and (p16[1:9, 3] < -0.9).any()),
                         "names": names, "neighbour": [t[0] for t in truth], "red": [t[1] for t in truth], **ro})
            print("settle", s, i, flush=True)
    (D / "settle_rows.json").write_text(json.dumps(rows))


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); pos, neg = x[y > 0.5], x[y <= 0.5]
    if not len(pos) or not len(neg):
        return None
    return round(float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())), 3)


def analyse():
    R = json.loads((D / "settle_rows.json").read_text()); out = {"decisions": len(R), "exploratory": True}
    phases = {"all": lambda r: True, "closed": lambda r: r["closed"], "release": lambda r: r["release"],
              "open": lambda r: not r["closed"]}
    for ph, f in phases.items():
        rs = [r for r in R if f(r)]; res = {"decisions": len(rs)}
        for tgt in ("red", "neighbour"):
            for k in KEYS:
                xs = [v for r in rs for v in r[k]]; ys = [v for r in rs for v in r[tgt]]
                res[f"{tgt}_auc_{k}"] = auc(xs, ys)                               # risk-weighted: y = P(fail) > 0.5? use soft below
                agree = tot = 0
                for r in rs:
                    v, y = np.array(r[k]), np.array(r[tgt])
                    for a in range(len(v)):
                        for b in range(a + 1, len(v)):
                            if y[a] != y[b] and v[a] != v[b]:
                                agree += (v[a] < v[b]) == (y[a] < y[b]); tot += 1
                res[f"{tgt}_conc_{k}"] = round(agree / max(tot, 1), 3); res[f"{tgt}_pairs"] = tot
            # soft AUC: policy candidate, any-failure (truth > 0) vs none, across decisions
            pol = [r["names"].index("policy") for r in rs]
            for k in KEYS:
                res[f"{tgt}_policy_auc_any_{k}"] = auc([r[k][p] for r, p in zip(rs, pol)], [float(r[tgt][p] > 0) for r, p in zip(rs, pol)])
        out[ph] = res
    (D / "settle_check.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else analyse()

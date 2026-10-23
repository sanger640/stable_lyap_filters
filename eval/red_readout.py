"""Exploratory (docs/FILTER_DEEP_DIVE.md 5e): with the gripper closed, holding still has far lower one-chunk red-fall risk
than G-CEM's fix (benchmark v2: dev 6.1% vs 14.6%, holdout 2.5% vs 9.0%), but G's certificate masks out the hand region,
where the carried block is. Does any LABEL-FREE imagined readout rank candidate chunks by real red-fall risk there?
Decisions: benchmark v2 dev + holdout with the gripper closed. Candidates per decision (all with cached truth): policy,
hold, G-CEM's chunk, O-CEM's 5 finals, each as chunk + hold tail. Readouts (model_OIL, 64 futures, common random numbers
across candidates, scale 1): o (outside-hand, the barrier), o_full, o_hand (max over futures of the mean squared latent
change in that region), hand_spread (variance across futures of the hand-region latent: does the held object's outcome
fork?), h (fork distance at scales 0.5 / 1 / 2 / 3, higher = safer).
usage: python3 eval/red_readout.py run | analyse  -> results/jenga/deep_dive/red_readout_rows.json, red_readout.json
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

D = ROOT / "results/jenga/deep_dive"
SC = (0.5, 1.0, 2.0, 3.0)


def p16_of(C, i, chunk):
    p = np.zeros((16, 4), np.float32); p[0] = np.r_[C["target"][i], C["nominal"][i][0, 3]]
    p[1:9] = chunk; p[9:] = p[8]
    return p


def readouts(cbf, sim, z0, plans, seed, n_fut=64):
    import cbf_barrier as cb
    import cbf_filter as cbff
    import cem_filter as cf
    import step3_world_model as wm
    mon, torch = cbf.mon, cbf.mon.torch
    C = len(plans); split = np.zeros((C, len(SC)), bool); r = {k: np.zeros(C) for k in ("o", "o_full", "o_hand", "hand_spread")}
    for si, s in enumerate(SC):
        d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (q * [0, 0, 1] * s).reshape(9)])
                       for p, q in (mon.errors.sample(seed, 0, 0, j) for j in range(n_fut))])
        st = mon.g(z0[None].expand(n_fut, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32))
        acts = []
        for p in plans:
            base = cbff.horizon(p, cbff.CFG["N_LOOK"])
            for j in range(n_fut):
                a = base.copy()
                a[:cbff.CFG["N_LOOK"], :3] -= np.concatenate([mon.snippets[j % 64], mon.snippets[(j + 1) % 64]])[:cbff.CFG["N_LOOK"]] * s
                acts.append(a)
        pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(C * n_fut, wm.STEPS, -1)
        z = st.float().repeat(C, 1, 1, 1); ends = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(wm.STEPS):
                z = mon.model.step(z, pairs[:, t]).float()
                if t in (wm.HOLD10, wm.HOLD30):
                    ends[t] = z.reshape(C, n_fut, -1)
        ok10, _, _ = cf.split_test_gpu(ends[wm.HOLD10]); ok30, _, _ = cf.split_test_gpu(ends[wm.HOLD30])
        split[:, si] = (ok10 & ok30).cpu().numpy()
        if s == 1.0:
            for c in range(C):
                pts = np.vstack([sim.data.site_xpos[sim.site][None], np.asarray(plans[c])[1:16, :3]])
                hm = cb.hand_mask(pts); m = torch.as_tensor(~hm, device="cuda").float(); mh = 1 - m
                e = ends[wm.HOLD30][c].reshape(n_fut, 28, 28, 8)
                diff = (e - z0[None]).pow(2).mean(-1)
                for name, mm in (("o", m), ("o_full", torch.ones_like(m)), ("o_hand", mh)):
                    r[name][c] = float(((diff * mm).sum((1, 2)) / mm.sum().clamp_min(1)).max())
                r["hand_spread"][c] = float((e.var(0).mean(-1) * mh).sum() / mh.sum().clamp_min(1))
    h = np.array([np.log2(SC[np.flatnonzero(sp)[0]]) if sp.any() else np.log2(4.0) for sp in split])
    return {**{k: v.tolist() for k, v in r.items()}, "h": h.tolist()}


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
            if not C["gripper"][i] < 50:
                continue
            chunks = {"policy": C["nominal"][i], "hold": b2.hold_chunk(C, i), "g_cem": np.asarray(G[i]["chunk"], np.float32)}
            chunks.update({f"o{j}": np.asarray(ch, np.float32) for j, ch in enumerate(O[i]["chunks"])})
            names = list(chunks)
            truth = [cache[b2._hash(i, np.asarray(chunks[k], np.float32))] for k in names]
            sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
            z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
            mon.rng = np.random.default_rng([20261124, i])
            ro = readouts(cbf, sim, z0, np.stack([p16_of(C, i, chunks[k]) for k in names]), int(mon.rng.integers(2 ** 31)))
            rows.append({"set": s, "i": i, "names": names, "neighbour": [t[0] for t in truth], "red": [t[1] for t in truth], **ro})
            print("redro", s, i, flush=True)
    (D / "red_readout_rows.json").write_text(json.dumps(rows))


def analyse():
    R = json.loads((D / "red_readout_rows.json").read_text()); out = {"decisions": len(R), "exploratory": True}
    keys = ("o", "o_full", "o_hand", "hand_spread", "neg_h")
    for r in R:
        r["neg_h"] = [-x for x in r["h"]]
    for s in ("dev", "holdout", "both"):
        rs = [r for r in R if s in ("both", r["set"])]; res = {"decisions": len(rs)}
        for k in keys:
            agree = tot = 0
            for r in rs:
                v, y = np.array(r[k]), np.array(r["red"])
                for a in range(len(v)):
                    for b in range(a + 1, len(v)):
                        if y[a] != y[b] and v[a] != v[b]:
                            agree += (v[a] < v[b]) == (y[a] < y[b]); tot += 1
            pick = [int(np.argmin(r[k])) for r in rs]
            res[k] = {"within_decision_concordance_red": round(agree / max(tot, 1), 3), "pairs": tot,
                      "red_of_argmin": round(float(np.mean([r["red"][p] for r, p in zip(rs, pick)])), 4),
                      "neighbour_of_argmin": round(float(np.mean([r["neighbour"][p] for r, p in zip(rs, pick)])), 4),
                      "argmin_is_hold_share": round(float(np.mean([r["names"][p] == "hold" for r, p in zip(rs, pick)])), 3)}
        for name in ("policy", "hold", "g_cem"):
            res[f"red_{name}"] = round(float(np.mean([r["red"][r["names"].index(name)] for r in rs])), 4)
        res["red_oracle_best"] = round(float(np.mean([min(r["red"]) for r in rs])), 4)
        out[s] = res
    (D / "red_readout.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else analyse()

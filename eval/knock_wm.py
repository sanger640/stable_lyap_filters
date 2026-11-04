"""KNOCK stage 1 (results/jenga/knock_protocol.json stage1): does model_OIL, zero-shot, imagine the knock-down task's
neighbour forks without inventing forks from the intended red topple?
usage: python3 eval/knock_wm.py run [n] [seed0] -> results/knock/wm_rows<SUF>.json;  grade -> results/knock/wm_result<SUF>.json
       (KNOCK_WM_SUFFIX = SUF, e.g. _knk with CBF_WM = model_KNK);  train -> model_KNK (stage 1b);  error -> latent errors"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CBF_WM", str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"))
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import knock_task as K  # noqa: E402

OUT = ROOT / "results/knock"
SUF = os.environ.get("KNOCK_WM_SUFFIX", "")
KNK = ROOT / "results/program_b/knock_v1"
OIL_PT = ROOT / "results/program_b/oil_v1/model/model_OIL.pt"


def real_labels(sim, r, snippets, rng):
    nb, rd = 0, 0
    for c in range(32):
        sim.restore(r["snap"])
        noise = snippets[rng.integers(len(snippets))] * 2.0
        peak = float(K.tilts(sim)[1:].max())
        for j in range(8):
            a = r["chunk"][j].copy(); a[:3] -= noise[j]; sim.execute(a); peak = max(peak, float(K.tilts(sim)[1:].max()))
        for _ in range(20):
            sim.execute(np.r_[sim.target, 1.0].astype(np.float32)); peak = max(peak, float(K.tilts(sim)[1:].max()))
        nb += peak >= K.TOPPLE_DEG; rd += K.tilts(sim)[0] >= K.TOPPLE_DEG
    return int(nb), int(rd)


def wm_stats(mon, cbff, wm, hm, z0, plan, s, n_fut=64):
    torch = mon.torch
    seed = int(mon.rng.integers(2 ** 31))
    d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (q * [0, 0, 1] * s).reshape(9)])
                   for p, q in (mon.errors.sample(seed, 0, 0, j) for j in range(n_fut))])
    st = mon.g(z0[None].expand(n_fut, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32))
    base = cbff.horizon(plan, cbff.CFG["N_LOOK"]); acts = []
    for j in range(n_fut):
        a = base.copy()
        a[:cbff.CFG["N_LOOK"], :3] -= np.concatenate([mon.snippets[j % 64], mon.snippets[(j + 1) % 64]])[:cbff.CFG["N_LOOK"]] * s
        acts.append(a)
    pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(n_fut, wm.STEPS, -1)
    z = st.float(); ends = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for t in range(wm.STEPS):
            z = mon.model.step(z, pairs[:, t]).float()
            if t in (wm.HOLD10, wm.HOLD30):
                ends[t] = z.reshape(1, n_fut, -1)
    a10, a30 = hm.split_stats(ends[wm.HOLD10]), hm.split_stats(ends[wm.HOLD30])
    return {"ok10": bool(a10["ok"][0]), "ok30": bool(a30["ok"][0]), "D10": float(a10["D"][0]), "D30": float(a30["D"][0])}


def run(n=40, seed0=42000):
    import torch
    import cem_filter as cf
    import cbf_filter as cbff
    import hj_margin as hm
    import step3_world_model as wm
    import uncertainty_monitor_dev as um
    snippets = np.load(um.BENCH)["snippets"]
    mon = cf.Monitor(); rows = []
    for e in range(seed0, seed0 + n):
        rec = []; res = K.episode(e, 0.05, 2.0, record=rec)
        sim = K.make_sim(); rng = np.random.default_rng([20261106, e])
        try:
            for r in rec:
                sim.restore(r["snap"])
                red_up = bool(K.tilts(sim)[0] < K.TOPPLE_DEG)
                mon.rng = np.random.default_rng([20261107, e, r["k"]])
                z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
                plan17 = np.vstack([np.r_[sim.target, 1.0][None], r["plan"]]).astype(np.float32)   # PLAN_START = 1
                w = {f"s{s:g}": wm_stats(mon, cbff, wm, hm, z0, plan17, s) for s in (1.0, 2.0)}
                nb, rd = real_labels(sim, r, snippets, rng)
                rows.append({"ep": e, "k": r["k"], "neigh": nb, "red": rd, "red_up": red_up, "pushed": r["pushed"], **w})
        finally:
            sim.close()
        print(e, res["success"], [(x["neigh"], x["red"], round(x["s2"]["D30"], 2), x["s2"]["ok10"] and x["s2"]["ok30"]) for x in rows if x["ep"] == e], flush=True)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"wm_rows{SUF}.json").write_text(json.dumps(rows))


def grade():
    from sklearn.metrics import roc_auc_score
    R = json.loads((OUT / f"wm_rows{SUF}.json").read_text())
    fork = np.array([0 < r["neigh"] < 32 for r in R])
    intended = np.array([r["neigh"] == 0 and r["red"] == 32 and r["red_up"] for r in R])
    out = {"protocol": "results/jenga/knock_protocol.json stage1", "states": len(R), "forks": int(fork.sum()), "intended_only": int(intended.sum())}
    for s in ("s1", "s2"):
        D = np.array([r[s]["D30"] for r in R]); fl = np.array([r[s]["ok10"] and r[s]["ok30"] for r in R])
        out[s] = {"auroc_D30_vs_fork": round(float(roc_auc_score(fork, D)), 4),
                  "flag_rate_forks": round(float(fl[fork].mean()), 4), "flag_rate_nonfork": round(float(fl[~fork].mean()), 4),
                  "flag_rate_intended_only": round(float(fl[intended].mean()), 4) if intended.any() else None}
    out["W1"] = out["s2"]["auroc_D30_vs_fork"] >= 0.70
    out["W2"] = out["s2"]["flag_rate_intended_only"] is not None and out["s2"]["flag_rate_intended_only"] <= 0.20
    out["pass"] = bool(out["W1"] and out["W2"])
    (OUT / f"wm_result{SUF}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def train():
    """stage 1b: eval/s2s.py's recipe with the knock-down futures in place of the shifted-physics ones."""
    import closedloop_train as ct
    import fast_train as ft
    import phase1d_models as p1d
    import step3b_models as sm
    sm.DATA = ROOT / "results/program_b/threeblock_v3"
    ct._patch()
    P = ROOT / "results/program_b"
    groups = [p1d.load([KNK / "train"], proprio=False), p1d.load([P / "oil_v1/train"], proprio=False),
              p1d.load([P / "cf_v1/train"], proprio=False), p1d.load([P / "closedloop_v1/train_selected"], proprio=False)]
    states = [s for g in groups for s in g]
    sizes = [sum(len(s["z"]) for s in g) for g in groups]; edges = np.cumsum([0] + sizes)
    pools = [np.arange(edges[k], edges[k + 1]) for k in range(4)]
    share = np.array([3, 1, 1, 1])

    def sampler(rng, b):
        n = np.floor(b * share / share.sum()).astype(int); n[0] += b - n.sum()
        return np.concatenate([rng.choice(pools[k], n[k]) for k in range(4)])
    print("futures knock / oil / cf / closed-loop", sizes, flush=True); (KNK / "model").mkdir(parents=True, exist_ok=True)
    ft.train("KNK", KNK / "model", states, sampler, steps=3000, init=OIL_PT, lr=ft.LR * 0.3)


def error():
    import torch
    import phase1d_models as p1d
    import step3_world_model as wm
    models = {"OIL": wm.WorldModel.load(OIL_PT), "KNK": wm.WorldModel.load(KNK / "model/model_KNK.pt")}
    sets = {"knock_val": KNK / "val", "jenga_val": ROOT / "results/program_b/closedloop_v1/val"}
    out = {}
    for sname, folder in sets.items():
        states = p1d.load([folder], proprio=False)
        z0k = torch.cat([s["z0k"] for s in states]).float().cuda(); tgt = torch.cat([s["z"] for s in states]).float().cuda()
        act = torch.cat([s["actions"] for s in states]).cuda()
        out[sname] = {"futures": len(z0k)}
        for mname, m in models.items():
            err = []
            for i in range(0, len(z0k), 256):
                cur = z0k[i:i + 256]; e = []
                with torch.no_grad():
                    for t in range(wm.STEPS):
                        cur = m.step(cur, act[i:i + 256, t]).float()
                        e.append(((cur - tgt[i:i + 256, t]) ** 2).mean((1, 2, 3)))
                err.append(torch.stack(e, 1).cpu())
            E = torch.cat(err).numpy()
            out[sname][mname] = {"all_steps": round(float(E.mean()), 5), "hold30": round(float(E[:, wm.HOLD30].mean()), 5)}
    (OUT / "knk_error.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    {"run": lambda: run(*(int(x) for x in a[1:])), "grade": grade, "train": train, "error": error}[a[0]]()

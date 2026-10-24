"""HJ v2 stage B (docs/HJ_V2.md): the closed-loop policy's safety value V(z) = expected min over future decisions of the
label-free margin l (stage A), learned offline from logged trajectories (PNCBF policy value + lambda-Reachability
multi-step targets). Higher V = safer.
  target y_t = min{l_t, ..., l_{t+n-1}, b},  n ~ Geom(1 - LAMBDA) truncated at the episode end,
  b = V_targ(z_{t+n}) with prob DELTA^n (else ignored), episode end = no further term.
Net: small conv on the 28x28x8 latent + proprio (gripper xyz, closed), twin critics (min for bootstrap), EMA target.
usage: HJ_MARGIN=<stage-A key> python3 eval/hj_value2.py train <run> [<run> ...]  -> results/jenga/hj2/value_<HJ_MARGIN>.pt
       HJ_MARGIN=... python3 eval/hj_value2.py score <run>   -> V per decision (results/jenga/hj2/vscore_<run>.json)
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
DATA = ROOT / "results/jenga/hj2/data"
OUT = ROOT / "results/jenga/hj2"
CFG = {"LAMBDA": 0.9, "DELTA": 0.97, "EPOCHS": 40, "BATCH": 256, "LR": 3e-4, "TAU": 0.01, "SEED": 0, "CLIP": 3.0}


def margin_of(stats, scales, key):
    """Recompute a stage-A margin from the raw stat array (scales x 8) of one decision."""
    import hj_margin as hm
    names = ["ok", "D", "gain", "spread"]
    st = {str(float(s)): {e: {q: float(stats[j][4 * ei + qi]) if q != "ok" else bool(stats[j][4 * ei + qi] > 0.5)
                              for qi, q in enumerate(names)} for ei, e in enumerate(("10", "30"))} for j, s in enumerate(scales)}
    hm.SC = tuple(float(s) for s in scales)
    return hm.margins({"stats": st})[key]


def load(runs, key):
    eps = []
    for run in runs:
        for f in sorted((DATA / run).glob("ep*.npz")):
            with np.load(f) as d:
                m = np.array([margin_of(s, d["scales"], key) for s in d["stats"]], np.float32)
                eps.append({"run": run, "ep": int(f.stem[2:]), "z": d["z"].astype(np.float32), "p": d["proprio"],
                            "l": m, "k": d["chunk_index"], "kt": int(d["topple_chunk"])})
    return eps


class Net:
    def __init__(self):
        import torch
        import torch.nn as nn
        class V(nn.Module):
            def __init__(s):
                super().__init__()
                s.c = nn.Sequential(nn.Conv2d(8, 32, 3, 2, 1), nn.SiLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
                                    nn.Conv2d(64, 64, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
                s.h = nn.Sequential(nn.Linear(64 + 4, 128), nn.SiLU(), nn.Linear(128, 1))
            def forward(s, z, p):
                return s.h(torch.cat([s.c(z.permute(0, 3, 1, 2)), p], 1))[:, 0]
        self.torch = torch
        self.q = nn.ModuleList([V(), V()]).cuda()
        self.t = nn.ModuleList([V(), V()]).cuda(); self.t.load_state_dict(self.q.state_dict())

    def v(self, z, p, target=False):
        m = self.t if target else self.q
        return self.torch.minimum(m[0](z, p), m[1](z, p))


def train(runs):
    import torch
    key = os.environ["HJ_MARGIN"]; rng = np.random.default_rng(CFG["SEED"]); torch.manual_seed(CFG["SEED"])
    eps = load(runs, key)
    allm = np.concatenate([e["l"] for e in eps]); med, iqr = float(np.median(allm)), float(np.subtract(*np.percentile(allm, [75, 25])))
    for e in eps:
        e["l"] = np.clip((e["l"] - med) / max(iqr, 1e-6), -CFG["CLIP"], CFG["CLIP"]).astype(np.float32)
    idx = [(i, t) for i, e in enumerate(eps) for t in range(len(e["l"]))]
    va = set(rng.choice(len(eps), max(1, len(eps) // 10), replace=False).tolist())          # 10% episodes: early stopping
    tr = [x for x in idx if x[0] not in va]; vl = [x for x in idx if x[0] in va]
    net = Net(); opt = torch.optim.AdamW(net.q.parameters(), lr=CFG["LR"], weight_decay=1e-4)
    Zt = [torch.as_tensor(e["z"], device="cuda") for e in eps]; Pt = [torch.as_tensor(e["p"], device="cuda") for e in eps]

    def batch(items):
        z, p, y_parts, bz, bp, bmask, disc = [], [], [], [], [], [], []
        for i, t in items:
            e = eps[i]; T = len(e["l"]); n = min(int(rng.geometric(1 - CFG["LAMBDA"])), T - t)
            z.append(Zt[i][t]); p.append(Pt[i][t]); y_parts.append(float(e["l"][t:t + n].min()))
            boot = t + n < T and rng.random() < CFG["DELTA"] ** n
            j = min(t + n, T - 1); bz.append(Zt[i][j]); bp.append(Pt[i][j]); bmask.append(boot)
        f = lambda xs: torch.stack(xs)
        return f(z), f(p), torch.tensor(y_parts, device="cuda"), f(bz), f(bp), torch.tensor(bmask, device="cuda")

    def loss_of(items, train_mode):
        z, p, ymin, bz, bp, bm = batch(items)
        with torch.no_grad():
            b = net.v(bz, bp, target=True)
        y = torch.where(bm, torch.minimum(ymin, b), ymin)
        return sum(((q(z, p) - y) ** 2).mean() for q in net.q)

    best, best_state, log = 1e9, None, []
    for ep_i in range(CFG["EPOCHS"]):
        perm = rng.permutation(len(tr)); tl = []
        for s in range(0, len(perm), CFG["BATCH"]):
            l = loss_of([tr[j] for j in perm[s:s + CFG["BATCH"]]], True)
            opt.zero_grad(); l.backward(); opt.step(); tl.append(float(l))
            with torch.no_grad():
                for a, b in zip(net.t.parameters(), net.q.parameters()):
                    a.mul_(1 - CFG["TAU"]).add_(CFG["TAU"] * b)
        with torch.no_grad():
            vls = float(np.mean([float(loss_of(vl[s:s + 1024], False)) for s in range(0, len(vl), 1024)]))
        log.append({"epoch": ep_i, "train": float(np.mean(tl)), "val": vls}); print(log[-1], flush=True)
        if vls < best:
            best, best_state = vls, {k: v.detach().clone() for k, v in net.q.state_dict().items()}
    torch.save({"state": best_state, "cfg": CFG, "margin": key, "norm": [med, iqr], "runs": runs, "log": log},
               OUT / f"value_{key}.pt")
    print("saved", OUT / f"value_{key}.pt", "best val", best, flush=True)


def load_value(key):
    import torch
    ck = torch.load(OUT / f"value_{key}.pt", weights_only=False); net = Net(); net.q.load_state_dict(ck["state"]); net.q.eval()
    return net, ck


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2:])


def gate_b():
    """Gate B (docs/HJ_V2.md): held-out policy-alone episodes of paired_record_ocl (the trigger-table episodes: topple windows
    + every 4th episode without a topple; not used in training). tau = the V quantile giving the trend rule's alarm rate
    (12.7%) on the TRAINING record-arm decisions of episodes without a topple; alarm = V(z_k) < tau. Report alarm rate on
    the held-out safe decisions, warned-within-8 and lead >= 3 share on the held-out topple episodes, vs the trend rule
    (12.7% / 87%)."""
    import torch
    import margin_check as mc
    key = os.environ["HJ_MARGIN"]; net, ck = load_value(key)
    vals = []
    for run in ck["runs"]:
        if "record" not in run:
            continue
        for f in sorted((DATA / run).glob("ep*.npz")):
            with np.load(f) as d:
                if int(d["topple_chunk"]) >= 0:
                    continue
                with torch.no_grad():
                    vals += net.v(torch.as_tensor(d["z"].astype(np.float32), device="cuda"), torch.as_tensor(d["proprio"], device="cuda")).cpu().tolist()
    tau = float(np.quantile(vals, 0.127))
    H = DATA / "paired_record_ocl"
    A = json.loads((ROOT / "results/jenga/outcome_audit_ocl.json").read_text())["paired_record_ocl"]["rows"]
    top = {r["ep"] for r in A if r["logged_neighbour_toppled"]}; safe = sorted({r["ep"] for r in A} - top)[::4]
    sr, leads, n_t, rows = [], [], 0, []
    for e in sorted(top | set(safe)):
        with np.load(H / f"ep{e}.npz") as d:
            with torch.no_grad():
                v = net.v(torch.as_tensor(d["z"].astype(np.float32), device="cuda"), torch.as_tensor(d["proprio"], device="cuda")).cpu().numpy()
            ks, kt = d["chunk_index"].astype(int), int(d["topple_chunk"])
        rows += [{"ep": e, "k": int(k), "v": float(x), "kt": kt} for k, x in zip(ks, v)]
        if e in top:
            if kt < 0:
                continue
            n_t += 1; hits = [k for k, x in zip(ks, v) if kt - 8 <= k <= kt and x < tau]
            if hits:
                leads.append(kt - min(hits))
        else:
            sr += [bool(x < tau) for x in v]
    out = {"margin": key, "tau": tau, "alarm_rate_heldout_safe": round(float(np.mean(sr)), 4), "topple_episodes": n_t,
           "warned_within_8": len(leads), "lead_ge_3_share_of_topple_eps": round(sum(l >= 3 for l in leads) / max(n_t, 1), 3),
           "lead_median": float(np.median(leads)) if leads else None, "trend_reference": {"alarm_rate": 0.127, "lead_ge_3_share_of_topple_eps": 0.85}}
    pos = np.array([r["v"] for r in rows if r["kt"] >= 0 and 1 <= r["kt"] - r["k"] <= 3]); neg = np.array([r["v"] for r in rows if r["ep"] in safe])
    out["auc_1-3_before"] = float((pos[:, None] < neg[None]).mean()) if len(pos) else None
    pos = np.array([r["v"] for r in rows if r["kt"] >= 0 and 4 <= r["kt"] - r["k"] <= 8])
    out["auc_4-8_before"] = float((pos[:, None] < neg[None]).mean()) if len(pos) else None
    (OUT / f"gateB_{key}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def trend_alarms(rows_dir):
    """The trend rule (docs/LITERATURE_PLAN.md: relative d <= 2x or o > O_MAX, OR d fell since the previous decision and
    relative d <= 4x) on margin_check rows -> {(ep, k): alarm}."""
    O_MAX = json.loads((ROOT / "results/jenga/cbf_calibration_OIL.json").read_text())["O_MAX"]
    M = {(m["ep"], m["k"]): m for f in sorted(rows_dir.glob("margin_rows_*.jsonl")) for m in map(json.loads, f.read_text().splitlines())}
    first = lambda fl: next((i for i, v in enumerate(fl) if v), 5)
    rel = lambda m, mi: first(m["plan"]) <= mi and first(m["plan"]) < first(m["hold"])
    by, al = {}, {}
    for e, k in M:
        by.setdefault(e, []).append(k)
    for e, ks in by.items():
        p_ = None
        for k in sorted(ks):
            m = M[(e, k)]
            al[(e, k)] = bool(rel(m, 2) or m["o"] > O_MAX or (p_ is not None and first(m["plan"]) < first(p_["plan"]) and rel(m, 4)))
            p_ = m
    return al


def gate_b2():
    """Gate B on HELD-OUT episodes (results/jenga/hj2/stageB_protocol.json): policy-alone arm of the margin test
    (paired_record_m3; never used for any design), the margin_check decision set (topple windows kt-10..kt + every 4th
    episode without a topple). Trend rule computed on the same decisions (eval/margin_check.py MC_RUN=paired_record_m3).
    Primary: at V's threshold giving the trend rule's alarm rate on these safe decisions, share of topple episodes warned
    >= 3 chunks ahead (within kt-8..kt) by V > the trend rule's."""
    import torch
    key = os.environ["HJ_MARGIN"]; net, ck = load_value(key)
    al = trend_alarms(ROOT / "results/jenga/deep_dive/m3")
    A = json.loads((ROOT / "results/jenga/outcome_audit_m3.json").read_text())["paired_record_m3"]["rows"]
    top = {r["ep"] for r in A if r["logged_neighbour_toppled"]}; safe = set(sorted({r["ep"] for r in A} - top)[::4])
    V = {}
    for e in sorted(top | safe):
        with np.load(DATA / "paired_record_m3" / f"ep{e}.npz") as d:
            with torch.no_grad():
                v = net.v(torch.as_tensor(d["z"].astype(np.float32), device="cuda"), torch.as_tensor(d["proprio"], device="cuda")).cpu().numpy()
            for k, x in zip(d["chunk_index"].astype(int), v):
                if (e, int(k)) in al:
                    V[(e, int(k))] = float(x)
    kt = {e: next((c["chunk"] for c in json.loads((ROOT / f"results/program_b/policy_rollouts/paired_record_m3/ep{e}.json").read_text())["log"]
                   if c.get("topple_started")), None) for e in top}
    keys = sorted(V)
    sv = np.array([V[q] for q in keys if q[0] in safe]); s_al = np.array([al[q] for q in keys if q[0] in safe])
    rate = float(s_al.mean()); tau = float(np.quantile(sv, rate))
    def lead_share(alarm):
        n, ge3 = 0, 0
        for e in top:
            if kt[e] is None:
                continue
            n += 1; hits = [k for (ee, k) in keys if ee == e and kt[e] - 8 <= k <= kt[e] and alarm((ee, k))]
            ge3 += bool(hits) and kt[e] - min(hits) >= 3
        return ge3 / max(n, 1), n
    tr_share, n_t = lead_share(lambda q: al[q]); v_share, _ = lead_share(lambda q: V[q] < tau)
    pos = np.array([V[q] for q in keys if q[0] in top and kt[q[0]] is not None and 1 <= kt[q[0]] - q[1] <= 3])
    pos46 = np.array([V[q] for q in keys if q[0] in top and kt[q[0]] is not None and 4 <= kt[q[0]] - q[1] <= 6])
    out = {"margin": key, "decisions": len(keys), "topple_episodes": n_t, "trend_alarm_rate_safe": round(rate, 4), "tau": tau,
           "V_alarm_rate_safe": round(float((sv < tau).mean()), 4), "trend_lead_ge3_share": round(tr_share, 3),
           "V_lead_ge3_share": round(v_share, 3), "auc_V_1-3": float((pos[:, None] < sv[None]).mean()),
           "auc_V_4-6": float((pos46[:, None] < sv[None]).mean()), "PRIMARY_pass": bool(v_share > tr_share)}
    (OUT / f"gateB2_{key}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__" and sys.argv[1] == "gate":
    gate_b()
if __name__ == "__main__" and sys.argv[1] == "gate2":
    gate_b2()

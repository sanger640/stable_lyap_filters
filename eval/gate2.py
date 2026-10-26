"""GATE2: raise the irreversibility gate's recall (results/jenga/gate2_protocol.json), offline. Candidates (all label-free):
  base   max_r_2 > THR                     (lfci000's gate; CF3)
  A      max_r_4 > THR                     (the same reversal residual at 4x perception + execution errors)
  B(v)   max_r_2 > THR  OR  V(z, g) < v    (or the label-free value value_lf_alea.pt on the real latent is below v)
  AB(v)  max_r_4 > THR  OR  V(z, g) < v,   v in (0.0, -0.1, -0.2), THR = 1.3356
Dev = held-out m3 decisions (results/jenga/cf2_rows.npz: r_2, r_4); confirmation = CF3's fresh s4 + h2h2 decisions
(cf3_rows_*.npz at 2x, cf3s4_rows_*.npz at 4x from CF_S=4 CF_TAG=cf3s4 eval/cf_conf.py run). g = [target, plan[0, gripper]].
Labels grade only.
usage: python3 eval/gate2.py values | dev | conf
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / "results/jenga"
THR, VS = 1.3356, (0.0, -0.1, -0.2)
CONF_RUNS = ("paired_record_s4", "paired_record_h2h2")


def _V(run_, eps_, is_):
    """V(z, g) for decisions (episode stems, decision indices) of one run."""
    import torch
    import lsf as L
    vf = L.ValueFn("value_lf_alea.pt"); out = np.zeros(len(eps_), np.float32); cache = {}
    for n, (e, i) in enumerate(zip(eps_, is_)):
        stem = f"ep{int(e)}"
        if stem not in cache:
            cache.clear()
            with np.load(ROOT / "results/jenga/hj2/data" / run_ / f"{stem}.npz") as d:
                hz = d["z"].astype(np.float32)
            with np.load(ROOT / "results/program_b/policy_rollouts" / run_ / f"{stem}.npz") as d:
                cache[stem] = (hz, d["plan16"], d["target"])
        hz, plan16, target = cache[stem]
        z = torch.as_tensor(hz[int(i)][None], device="cuda").float()
        g = torch.as_tensor(np.r_[target[int(i)], plan16[int(i)][0, 3]][None], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            out[n] = float(vf.V(z, g))
    return out


def values():
    R = np.load(OUT / "cf2_rows.npz")
    np.save(OUT / "gate2_V_m3.npy", _V("paired_record_m3", R["ep"], R["i"]))
    C = _conf_rows(need4=False)
    v = np.zeros(len(C["ep"]), np.float32)
    for j, run_ in enumerate(CONF_RUNS):
        w = C["run"] == j; v[w] = _V(run_, C["ep"][w], C["i"][w])
    np.save(OUT / "gate2_V_conf.npy", v); print("values saved", flush=True)


def _conf_rows(need4=True):
    def load(prefix):
        fs = sorted(OUT.glob(f"{prefix}_rows_*.npz"))
        return {k: np.concatenate([np.load(f)[k] for f in fs]) for k in np.load(fs[0]).files}
    C = load("cf3")
    if need4:
        C4 = load("cf3s4")
        key = lambda D: list(zip(D["run"], D["ep"], D["i"]))
        idx = {t: n for n, t in enumerate(key(C4))}
        C["r_4"] = C4["r"][[idx[t] for t in key(C)]]
    C["r_2"] = C["r"]
    return C


def grade(r2, r4, V, kt, k, cl):
    pre = (kt >= 0) & (kt - k >= 1) & (kt - k <= 3); safe = kt < 0; cl = cl > 0
    m2, m4 = r2.max(1), r4.max(1)
    cands = {"base": m2 > THR, "A": m4 > THR}
    for v in VS:
        cands[f"B({v:g})"] = (m2 > THR) | (V < v); cands[f"AB({v:g})"] = (m4 > THR) | (V < v)
    out = {}
    for n, f in cands.items():
        out[n] = {"recall_pre": round(float(f[pre].mean()), 4), "safe_closed": round(float(f[safe & cl].mean()), 4),
                  "safe_open": round(float(f[safe & ~cl].mean()), 4),
                  "recall_by_lead": [round(float(f[pre & (kt - k == d)].mean()), 4) for d in (1, 2, 3)]}
    out["_V_quantiles_safe"] = np.quantile(V[safe], [0.01, 0.05, 0.1, 0.5]).round(3).tolist()
    out["_V_below_0_pre"] = round(float((V[pre] < 0).mean()), 4)
    return out


def dev():
    R = np.load(OUT / "cf2_rows.npz"); V = np.load(OUT / "gate2_V_m3.npy")
    g = grade(R["r_2"], R["r_4"], V, R["kt"], R["k"], R["closed"])
    ok = [n for n, v in g.items() if not n.startswith("_") and n != "base" and v["safe_closed"] <= 0.05 and v["safe_open"] <= 0.15]
    sel = max(ok, key=lambda n: (g[n]["recall_pre"], -g[n]["safe_closed"])) if ok else None
    go = bool(sel and g[sel]["recall_pre"] >= g["base"]["recall_pre"] + 0.10)
    out = {"decisions": int(len(R["kt"])), "grades": g, "eligible": ok, "selected": sel, "GO_to_confirmation": go}
    (OUT / "gate2_dev.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def conf():
    sel = json.loads((OUT / "gate2_dev.json").read_text())["selected"]
    C = _conf_rows(); V = np.load(OUT / "gate2_V_conf.npy")
    g = grade(C["r_2"], C["r_4"], V, C["kt"], C["k"], C["closed"])
    s = g[sel]
    out = {"decisions": int(len(C["kt"])), "selected": sel, "grades": g,
           "GATE2_pass": bool(s["recall_pre"] >= g["base"]["recall_pre"] + 0.10 and s["safe_closed"] <= 0.05 and s["safe_open"] <= 0.15)}
    (OUT / "gate2_conf.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"values": values, "dev": dev, "conf": conf}[sys.argv[1]]()

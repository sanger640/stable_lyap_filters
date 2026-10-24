"""Baseline monitor (docs/LITERATURE_REVIEW.md 10.2 item 2): is the fork (bimodality) needed, or does the SCALAR SPREAD of
the same imagined futures do as well? Same decisions and the SAME 64 futures per scale as the fork-test rows of
eval/margin_check.py (same seed), plan and hold, scales 0.5-4: from the HOLD10 / HOLD30 endings, the variance along the
principal axis ("spread", the axis the split test uses) and the total variance ("tvar", trace / dim), both per latent dim.
Alarm rules at the fork test's safe alarm rate: score_plan > tau and score_plan - score_hold > tau (relative), per scale.
usage: WM_COMPILE=1 python3 eval/spread_check.py run | analyse   (MC_RUN / MC_TAG as margin_check)
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
import margin_check as mc  # noqa: E402

OUT = ROOT / "results/jenga/deep_dive" / os.environ.get("SP_OUT", f"spread_rows{mc.TAG}.jsonl")
KEYS = os.environ.get("SP_KEYS")                                  # other decision sets (e.g. deep_dive/redfall_rows.jsonl)


def stats(cbf, z0, plans, n_fut=64):
    import cbf_filter as cbff
    import step3_world_model as wm
    mon, torch = cbf.mon, cbf.mon.torch
    C = len(plans); seed = int(mon.rng.integers(2 ** 31)); res = {}                 # one seed, as margin_check.split_flags
    for s in mc.SC:
        d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (r * [0, 0, 1] * s).reshape(9)])
                       for p, r in (mon.errors.sample(seed, 0, 0, j) for j in range(n_fut))])
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
        r = {}
        for t, e in ((10, ends[wm.HOLD10]), (30, ends[wm.HOLD30])):
            x = e.double(); X = x - x.mean(1, keepdim=True); Dm = x.shape[2]
            w = torch.linalg.eigvalsh(X @ X.transpose(1, 2))
            r[f"spread{t}"] = (w[:, -1] / n_fut / Dm).float().cpu().numpy().tolist()
            r[f"tvar{t}"] = ((X ** 2).sum((1, 2)) / n_fut / Dm).float().cpu().numpy().tolist()
        res[str(s)] = r
    return res


def run():
    import torch
    import cem_filter as cf
    import cbf_filter as cbff
    import inspector_data as insp
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    files = [ROOT / KEYS] if KEYS else sorted(mc.DT.glob("margin_rows_*.jsonl"))
    keys = {(r["ep"], r["k"]) for f in files for r in map(json.loads, f.read_text().splitlines())}
    done = {(r["ep"], r["k"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    with OUT.open("a") as fh:
        for ep in sorted({e for e, _ in keys}):
            with np.load(mc.RUN / f"ep{ep}.npz") as z:
                S = {k: z[k] for k in z.files}
            for d in range(len(S["chunk_index"])):
                k = int(S["chunk_index"][d])
                if (ep, k) not in keys or (ep, k) in done:
                    continue
                sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
                mon.rng = np.random.default_rng([20261100, ep, k])                      # = margin_check seed 0
                z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
                plan = S["plan16"][d].copy(); plan[0, :3] = S["target"][d]
                hold = np.repeat(np.r_[S["target"][d], 0.0][None].astype(np.float32), 16, 0)
                fh.write(json.dumps({"ep": ep, "k": k, "stats": stats(cbf, z0, np.stack([plan, hold]))}) + "\n"); fh.flush()
            print("spread", ep, flush=True)


def analyse():
    import seed_pool_check as sp
    rows = {(r["ep"], r["k"]): r for r in map(json.loads, OUT.read_text().splitlines())}
    M = {(r["ep"], r["k"]): r for f in sorted(mc.DT.glob("margin_rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}
    keys = sorted(set(rows) & set(M))
    A = json.loads(mc.AUDIT[0].read_text())[mc.AUDIT[1]]["rows"]
    topple = {r["ep"]: r["logged_neighbour_toppled"] for r in A}
    kt = {e: next((c["chunk"] for c in json.loads((mc.RUN / f"ep{e}.json").read_text())["log"] if c.get("topple_started")), None)
          for e in {k[0] for k in keys}}
    safe = np.array([not topple.get(k[0]) for k in keys])
    O_MAX = json.loads((ROOT / os.environ["CBF_CALIB"]).read_text())["O_MAX"]

    def evaluate(fire):
        by = {}
        for (e, k), f in zip(keys, fire):
            by.setdefault(e, {})[k] = bool(f)
        per, n_w = {}, 0
        for e, ks in by.items():
            if topple.get(e) and kt.get(e) is not None:
                hits = [k for k in sorted(ks) if kt[e] - 8 <= k <= kt[e] and ks[k]]
                per[e] = bool(hits) and kt[e] - hits[0] >= 3; n_w += bool(hits)
        evaluate.last = per
        return {"alarm_rate_safe": round(float(np.mean(np.asarray(fire)[safe])), 4), "topple_episodes": len(per),
                "warned_within_8": n_w, "lead_ge3_share_of_topples": round(float(np.mean(list(per.values()))), 3)}

    hp = np.array([sp.h_of(M[k]["plan"]) for k in keys]); hh = np.array([sp.h_of(M[k]["hold"]) for k in keys])
    o = np.array([M[k]["o"] for k in keys])
    out = {"decisions": len(keys), "exploratory": mc.TAG != "_m3", "fork_3x": evaluate((hp < np.minimum(hh, 2.0) - 1e-9) | (o > O_MAX))}
    per_fork = evaluate.last; rate = out["fork_3x"]["alarm_rate_safe"]; best = None
    for s in mc.SC:
        for stat in ("spread10", "spread30", "tvar10", "tvar30"):
            P = np.log(np.array([rows[k]["stats"][str(s)][stat][0] for k in keys]) + 1e-12)
            H = np.log(np.array([rows[k]["stats"][str(s)][stat][1] for k in keys]) + 1e-12)
            for kind, sc in (("plan", P), ("rel", P - H)):
                tau = float(np.quantile(sc[safe], 1 - rate)); r = evaluate(sc > tau)
                name = f"{stat}_{kind}_s{s}"; out[name] = r
                if best is None or r["lead_ge3_share_of_topples"] > best[1]["lead_ge3_share_of_topples"]:
                    best = (name, r, dict(evaluate.last))
    out["best_spread_rule"] = best[0]
    E = sorted(per_fork); x = np.array([per_fork[e] for e in E], float) - np.array([best[2][e] for e in E], float)
    rng = np.random.default_rng(20261124); bs = np.array([x[rng.integers(len(x), size=len(x))].mean() for _ in range(10000)])
    out["fork_minus_best_spread"] = {"pp": round(100 * float(x.mean()), 1), "lower95_one_sided_pp": round(100 * float(np.percentile(bs, 5)), 1)}
    (ROOT / f"results/jenga/deep_dive/spread_check{mc.TAG}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else v.get("lead_ge3_share_of_topples", v)) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    {"run": run, "analyse": analyse}[sys.argv[1]]()

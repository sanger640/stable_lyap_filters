"""Exploratory (docs/LITERATURE_PLAN.md, error-adaptive margin; "When World Models Lie", arXiv 2609.34300): does the world
model's own one-chunk prediction error (label-free, observable online) rise before topples, earlier than the fork alarm?
Policy-alone episodes of the O-CEM closed-loop test (paired_record_ocl), same episodes as eval/margin_check.py (topple
episodes + every 4th episode without a topple). For consecutive decisions k -> k+1: z_k, z_{k+1} = latents of the saved
states; prediction = model_OIL from z_k under the executed chunk, 16 execution-noise draws (x EXEC_NOISE_SCALE);
e = mean squared distance of z_{k+1} to the prediction mean, r = e / mean squared spread of the draws.
usage: python3 eval/wm_err_check.py run | analyse   -> results/jenga/deep_dive/wmerr_rows.jsonl, wmerr_check.json
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
RUN = ROOT / "results/program_b/policy_rollouts/paired_record_ocl"
D = ROOT / "results/jenga/deep_dive"
OUT = D / "wmerr_rows.jsonl"
N = 16


def episodes():
    A = json.loads((ROOT / "results/jenga/outcome_audit_ocl.json").read_text())["paired_record_ocl"]["rows"]
    top = {r["ep"] for r in A if r["logged_neighbour_toppled"]}
    safe = sorted({r["ep"] for r in A} - top)[::4]
    kt = {}
    for e in top | set(safe):
        L = json.loads((RUN / f"ep{e}.json").read_text())["log"]
        kt[e] = next((c["chunk"] for c in L if c.get("topple_started")), None)
    return top, safe, kt


def run():
    import torch
    import cem_filter as cf
    import inspector_data as insp
    import step3_world_model as wm
    mon = cf.Monitor(); sim = insp._sim(); scale = float(os.environ["EXEC_NOISE_SCALE"])
    done = {(r["ep"], r["k"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    top, safe, kt = episodes()
    with OUT.open("a") as fh:
        for ep in sorted(top | set(safe)):
            with np.load(RUN / f"ep{ep}.npz") as z:
                S = {k: z[k] for k in z.files}
            ks = S["chunk_index"].astype(int)
            lat = {}
            for d in range(len(ks) - 1):
                k = int(ks[d])
                if ks[d + 1] != k + 1 or (ep, k) in done or (ep in top and (kt[ep] is None or k + 1 > kt[ep])):
                    continue
                for dd in (d, d + 1):
                    if dd not in lat:
                        sim.restore((S["state"][dd], S["ctrl"][dd], S["target"][dd], float(S["gripper"][dd])))
                        lat[dd] = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
                rng = np.random.default_rng([20261111, ep, k])
                snip = mon.snippets[rng.choice(len(mon.snippets), N, replace=False)] * scale
                acts = wm.panel_actions(S["executed"][d], snip)                          # (N, 38, 4)
                pairs = torch.as_tensor(acts, device="cuda").reshape(N, wm.STEPS, -1)
                x = lat[d][None].expand(N, -1, -1, -1).contiguous()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for t in range(4):                                                  # 8 executed actions = 4 model steps
                        x = mon.model.step(x, pairs[:, t]).float()
                mu = x.mean(0)
                e = float((lat[d + 1] - mu).pow(2).mean()); sp = float((x - mu).pow(2).mean())
                fh.write(json.dumps({"ep": ep, "k": k, "e": e, "spread": sp, "r": e / max(sp, 1e-12)}) + "\n"); fh.flush()
            print("wmerr", ep, flush=True)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def analyse():
    R = [json.loads(l) for l in OUT.read_text().splitlines()]
    top, safe, kt = episodes()
    M = {}
    for f in sorted(D.glob("margin_rows_*.jsonl")):
        for l in f.read_text().splitlines():
            m = json.loads(l); M[(m["ep"], m["k"])] = m
    out = {"transitions": len(R), "exploratory": True}
    for key in ("e", "r"):
        neg = [x[key] for x in R if x["ep"] in safe]
        thr = float(np.quantile(neg, 0.95))
        res = {"safe_q95": thr}
        for lo, hi in ((1, 1), (2, 3), (4, 6), (7, 10)):                               # transitions ending lo..hi chunks before topple
            pos = [x[key] for x in R if x["ep"] in top and kt[x["ep"]] is not None and lo <= kt[x["ep"]] - (x["k"] + 1) <= hi]
            res[f"auc_{lo}-{hi}_before"] = auc(pos, neg); res[f"n_{lo}-{hi}"] = len(pos)
        leads, n_t = [], 0
        for e in top:
            if kt[e] is None:
                continue
            n_t += 1
            hits = [x["k"] + 1 for x in R if x["ep"] == e and kt[e] - 8 <= x["k"] + 1 <= kt[e] and x[key] > thr]
            if hits:
                leads.append(kt[e] - min(hits))
        res["topple_eps"] = n_t; res["warned_within_8_at_safe_q95"] = len(leads)
        res["lead_ge_3_share"] = round(float(np.mean([v >= 3 for v in leads])), 3) if leads else None
        # does the error carry information beyond the fork alarm? AUC among decisions where the plan does NOT fork at <= 2x
        nf = lambda x: (x["ep"], x["k"] + 1) in M and not any(M[(x["ep"], x["k"] + 1)]["plan"][:3])
        pos = [x[key] for x in R if x["ep"] in top and kt[x["ep"]] is not None and 1 <= kt[x["ep"]] - (x["k"] + 1) <= 6 and nf(x)]
        neg2 = [x[key] for x in R if x["ep"] in safe and nf(x)]
        res["auc_1-6_before_among_no_fork_2x"] = auc(pos, neg2); res["n_pos_no_fork"] = len(pos)
        out[key] = res
    (D / "wmerr_check.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else analyse()

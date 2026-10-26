"""CF2: time-irreversibility read of the fork, offline (results/jenga/cf2_protocol.json). Same decisions, noise draws and
grading as eval/cf_fork.py (CF1). Per noise draw j (64 futures, model_OIL, perception + execution noise at scale s):
  hold branch:     the current target held for all 38 control steps
  reverse branch:  the policy's plan (the fork's 15 look-ahead steps, with the draw's execution noise), then the same
                   targets and gripper commands in reverse order back to the current target (noise-free), then hold
  r_j = 98th percentile over the latent patches OUTSIDE the plan's hand path of mean_ch |f_rev_j - f_hold_j| at the end
An action that can be undone (pull the block out and push it back) leaves r small; one that cannot (a neighbour fell)
leaves it large. Background motion common to both branches cancels. P_irr = share of draws with r_j > tau, tau = q98
of r_j over pre-grasp decisions (label-free). Labels for grading only.
usage: python3 eval/cf_rev.py run -> results/jenga/cf2_rows.npz;  python3 eval/cf_rev.py grade -> results/jenga/cf2_result.json
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import cf_fork as C

OUT = ROOT / "results/jenga"


def reverse_plan(plan, cur, n_look, start):
    """(16, 4) plan -> (16 + n_look, 4) targets: the original 16 (the fork reads steps start .. start + n_look - 1), then
    those n_look targets in reverse order ending at the current target (gripper commands reversed with them)."""
    seg = plan[start:start + n_look]
    back = np.concatenate([seg[::-1][1:], cur[None]])
    return np.concatenate([plan[:start + n_look], back])


def run():
    import torch
    import cbf_barrier as cb
    import cbf_filter as cbff
    import cem_filter as cf
    import step3b_models as sm
    import step3_world_model as wm
    import visual_pipeline_panel3 as p3
    import uncertainty_monitor_dev as um
    model = wm.WorldModel.load(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); model.net.eval()
    G = sm.G.load(cf.V3 / "G.pt"); E = p3.ErrorModel(); snip = np.load(um.BENCH)["snippets"]
    S0, NL = cbff.CFG["PLAN_START"], cbff.CFG["N_LOOK"]
    total = 2 * wm.STEPS
    sel = C.select(); cache = {}
    res = {k: [] for k in ("ep", "i", "kt", "k", "closed", "pregrasp")}
    for s in C.SCALES:
        res[f"r_{s:g}"] = []
    for n, (stem, i, kt, k, cl) in enumerate(sel):
        if stem not in cache:
            cache.clear()
            with np.load(C.HJ / f"{stem}.npz") as d:
                hz, hp = d["z"].astype(np.float32), d["proprio"]
            with np.load(C.PR / f"{stem}.npz") as d:
                plan16, target = d["plan16"], d["target"]
            cl_idx = np.flatnonzero(hp[:, 3] > 0); fc = cl_idx[0] if len(cl_idx) else len(hp)
            cache[stem] = (hz, hp, plan16, target, fc)
        hz, hp, plan16, target, fc = cache[stem]
        z0 = torch.as_tensor(hz[i], device="cuda")
        plan = np.asarray(plan16[i], np.float32)
        cur = np.r_[target[i], plan[0, 3]].astype(np.float32)
        rev = reverse_plan(plan, cur, NL, S0)
        rev = np.concatenate([rev[S0:], np.repeat(rev[-1:], total, 0)])[:total]          # from the fork's first step
        hold = np.repeat(cur[None], total, 0)
        pts = np.vstack([hp[i, :3][None], plan[S0:S0 + NL, :3]])
        m = torch.as_tensor(~cb.hand_mask(pts), device="cuda")
        for key, v in (("ep", int(stem[2:])), ("i", i), ("kt", kt), ("k", k), ("closed", cl), ("pregrasp", float(i < fc))):
            res[key].append(v)
        seed = C.SEED + 1000 * int(stem[2:]) + i
        for s in C.SCALES:
            d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (r * [0, 0, 1] * s).reshape(9)])
                           for p, r in (E.sample(seed, 0, 0, j) for j in range(C.N_FUT))])
            st = G(z0[None].expand(C.N_FUT, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32)).float()
            acts = []
            for base in (rev, hold):
                for j in range(C.N_FUT):
                    a = base.copy()
                    a[:NL, :3] -= np.concatenate([snip[j % 64], snip[(j + 1) % 64]])[:NL] * s
                    acts.append(a)
            pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(2 * C.N_FUT, wm.STEPS, -1)
            z = st.repeat(2, 1, 1, 1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t in range(wm.STEPS):
                    z = model.step(z, pairs[:, t]).float()
            zr, zh = z[:C.N_FUT].float(), z[C.N_FUT:].float()
            res[f"r_{s:g}"].append(torch.quantile((zr - zh).abs().mean(-1)[:, m], 0.98, dim=1).cpu().numpy())
        if n % 100 == 0:
            print("cf2", n, len(sel), flush=True)
    np.savez_compressed(OUT / "cf2_rows.npz", **{k: np.asarray(v) for k, v in res.items()})
    print("saved", flush=True)


def grade():
    R = dict(np.load(OUT / "cf2_rows.npz"))
    kt, k, cl, pg = R["kt"], R["k"], R["closed"] > 0, R["pregrasp"] > 0
    pre = (kt >= 0) & (kt - k >= 1) & (kt - k <= 3); safe = kt < 0
    out = {"decisions": int(len(kt)), "stats": {}}
    for s in C.SCALES:
        D = R[f"r_{s:g}"]; tau = float(np.quantile(D[pg].ravel(), 0.98))
        for name, r in ((f"P_irr_{s:g}", (D > tau).mean(1)), (f"mean_r_{s:g}", D.mean(1)), (f"max_r_{s:g}", D.max(1))):
            out["stats"][name] = C._grade(r, pre, safe, cl)
        out["stats"][f"P_irr_{s:g}"]["tau"] = tau
    cf1 = json.loads((OUT / "cf1_result.json").read_text())
    ref = cf1["reference_spread_safe_closed_flag_at_50pct"]
    out["reference_spread_safe_closed_flag_at_50pct"] = ref
    out["GATE_CF2_pass"] = bool(any(v["auc"] >= 0.85 and v["safe_closed_flag_at_50pct"] <= ref / 2
                                    for n, v in out["stats"].items() if n.startswith("P_irr_")))
    (OUT / "cf2_result.json").write_text(json.dumps(out, indent=1))
    for n, v in sorted(out["stats"].items(), key=lambda x: x[1]["safe_closed_flag_at_50pct"]):
        print(f"{n:12s} AUC {v['auc']:.3f}  @50% recall: safe {v['safe_flag_at_50pct']:.3f} safe&closed {v['safe_closed_flag_at_50pct']:.3f}")
    print("GATE_CF2_pass", out["GATE_CF2_pass"])


if __name__ == "__main__":
    {"run": run, "grade": grade}[sys.argv[1]]()

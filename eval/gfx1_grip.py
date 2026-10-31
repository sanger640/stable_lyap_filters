"""GFX1 part G0 (results/jenga/gfx1_protocol.json, offline): is holding the gripper in the undo test why the value cannot see
red-block falls? On CF3's decisions (irrm1 rows; red-fall chunk from redm1_rows) the undo test of the policy's own plan, as
eval/irrm1.py (do NL control steps, undo back to the current target, hold; vs hold; 64 draws at 2x, CF3's seeds), with
  pol:  the plan's gripper commands (reversed with the plan, as CF3 / irrm1)
  held: the same xyz, gripper held at the current command (as the value's labels, eval/irrm2.py)
for NL 8 and 16; patch residual over the hand-masked patches (as irrm1) and over all patches; q98 (CF3's statistic) and max.
Decisions: all pre-red-fall (1-3 chunks before), all pre-topple (1-3 chunks), 1200 clean safe (seeded subsample).
usage: SHARD=i/n python3 eval/gfx1_grip.py run -> results/jenga/gfx1_grip_rows_<i>.npz;  python3 eval/gfx1_grip.py grade -> gfx1_grip.json"""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / "results/jenga"
S, THR, NLS, N_CLEAN = 2.0, 1.3356, (8, 16), 1200
VARS = [f"{g}{nl}_{m}_{st}" for nl in NLS for g in ("pol", "held") for m in ("mask", "all") for st in ("q98", "max")]


def rows():
    fs = sorted(OUT.glob("irrm1_rows_*.npz"))
    R = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("run", "ep", "i", "kt", "k", "closed", "r8")}
    Q = np.load(OUT / "redm1_rows.npz"); kr = {(int(a), int(b), int(c)): int(d) for a, b, c, d in zip(Q["run"], Q["ep"], Q["k"], Q["kr"])}
    R["kr"] = np.array([kr.get((int(a), int(b), int(c)), -1) for a, b, c in zip(R["run"], R["ep"], R["k"])])
    kt, k, KR = R["kt"], R["k"], R["kr"]
    pre_red = (KR >= 0) & (KR - k >= 1) & (KR - k <= 3); pre_top = (kt >= 0) & (kt - k >= 1) & (kt - k <= 3); clean = (kt < 0) & (KR < 0)
    rng = np.random.default_rng(20261031)
    ci = np.sort(rng.choice(np.flatnonzero(clean), N_CLEAN, replace=False))
    sel = np.sort(np.r_[np.flatnonzero(pre_red), np.flatnonzero(pre_top & ~pre_red), ci])
    return R, sel, pre_red, pre_top, clean


def run():
    import torch
    import cbf_barrier as cb
    import cbf_filter as cbff
    import cem_filter as cf
    import cf_conf as CC
    import cf_fork as C
    import cf_rev as V
    import step3b_models as sm
    import step3_world_model as wm
    import visual_pipeline_panel3 as p3
    import uncertainty_monitor_dev as um
    i_sh, n_sh = map(int, os.environ.get("SHARD", "0/1").split("/"))
    model = wm.WorldModel.load(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); model.net.eval()
    G = sm.G.load(cf.V3 / "G.pt"); E = p3.ErrorModel(); snip = np.load(um.BENCH)["snippets"]
    S0 = cbff.CFG["PLAN_START"]; total = 2 * wm.STEPS
    R, sel, *_ = rows(); sel = sel[i_sh::n_sh]
    lim = int(os.environ.get("GFX1_LIMIT", len(sel))); sel = sel[:lim]
    res = {"idx": [], "r": []}; cache = {}
    for n, idx in enumerate(sel):
        run_, stem, i = CC.RUNS[int(R["run"][idx])], f"ep{int(R['ep'][idx])}", int(R["i"][idx])
        if (run_, stem) not in cache:
            cache.clear()
            with np.load(ROOT / "results/jenga/hj2/data" / run_ / f"{stem}.npz") as d:
                hz, hp = d["z"].astype(np.float32), d["proprio"]
            with np.load(ROOT / "results/program_b/policy_rollouts" / run_ / f"{stem}.npz") as d:
                plan16, target = d["plan16"], d["target"]
            cache[(run_, stem)] = (hz, hp, plan16, target)
        hz, hp, plan16, target = cache[(run_, stem)]
        z0 = torch.as_tensor(hz[i], device="cuda"); plan = np.asarray(plan16[i], np.float32)
        cur = np.r_[target[i], plan[0, 3]].astype(np.float32)
        seed = C.SEED + 1000 * int(stem[2:]) + i                       # = CF3's / irrm1's draws for this decision
        d_ = np.stack([np.concatenate([(p * [1, 1, 0] * S).reshape(9), (r * [0, 0, 1] * S).reshape(9)])
                       for p, r in (E.sample(seed, 0, 0, j) for j in range(C.N_FUT))])
        st = G(z0[None].expand(C.N_FUT, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32)).float()
        hold = np.repeat(cur[None], total, 0)
        acts, masks = [], []
        for nl in NLS:                                                 # per horizon: pol, held, hold (same noise, as irrm1)
            rev = V.reverse_plan(plan, cur, nl, S0)
            rev = np.concatenate([rev[S0:], np.repeat(rev[-1:], total, 0)])[:total]
            held = rev.copy(); held[:, 3] = cur[3]
            masks.append(torch.as_tensor(~cb.hand_mask(np.vstack([hp[i, :3][None], plan[S0:S0 + nl, :3]])), device="cuda"))
            for base in (rev, held, hold):
                for j in range(C.N_FUT):
                    a = base.copy(); a[:nl, :3] -= np.concatenate([snip[j % 64], snip[(j + 1) % 64]])[:nl] * S; acts.append(a)
        B = 3 * len(NLS)
        pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(B * C.N_FUT, wm.STEPS, -1)
        z = st.repeat(B, 1, 1, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(wm.STEPS):
                z = model.step(z, pairs[:, t]).float()
        z = z.float().reshape(B, C.N_FUT, *z.shape[1:])
        out = []
        for b, nl in enumerate(NLS):
            for g in (0, 1):                                           # pol, held (vs this horizon's hold branch)
                dz = (z[3 * b + g] - z[3 * b + 2]).abs().mean(-1).reshape(C.N_FUT, -1)
                for m in (masks[b].reshape(-1), torch.ones_like(masks[b].reshape(-1))):
                    x = dz[:, m.bool()]
                    out += [torch.quantile(x, 0.98, dim=1).cpu().numpy(), x.max(1).values.cpu().numpy()]
        res["idx"].append(int(idx)); res["r"].append(np.stack(out).astype(np.float32))
        if n % 100 == 0:
            print("gfx1 grip", i_sh, n, len(sel), flush=True)
    np.savez_compressed(OUT / f"gfx1_grip_rows_{i_sh}.npz", idx=np.asarray(res["idx"]), r=np.asarray(res["r"]), vars=np.array(VARS))
    print("saved", flush=True)


def grade():
    from sklearn.metrics import roc_auc_score
    R, _, pre_red, pre_top, clean = rows()
    fs = sorted(OUT.glob("gfx1_grip_rows_*.npz"))
    idx = np.concatenate([np.load(f)["idx"] for f in fs]); r = np.concatenate([np.load(f)["r"] for f in fs]).max(2)   # (n, var) r_max
    o = np.argsort(idx); idx, r = idx[o], r[o]
    pr, pt, cl = pre_red[idx], pre_top[idx], clean[idx]; closed = R["closed"][idx] > 0
    auc = lambda pos: round(float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(cl.sum())], np.r_[s[pos], s[cl]])), 4)
    out = {"decisions": int(len(idx)), "pre_red": int(pr.sum()), "pre_topple": int(pt.sum()), "clean": int(cl.sum()), "tau": THR, "variants": {}}
    for v, name in enumerate(VARS):
        s = r[:, v]
        out["variants"][name] = {"pre_red_auc": auc(pr), "pre_red_recall": round(float((s[pr] > THR).mean()), 4),
                                 "pre_red_median": round(float(np.median(s[pr])), 4),
                                 "pre_topple_auc": auc(pt), "pre_topple_recall": round(float((s[pt] > THR).mean()), 4),
                                 "clean_flag_closed": round(float((s[cl & closed] > THR).mean()), 4),
                                 "clean_flag_open": round(float((s[cl & ~closed] > THR).mean()), 4)}
    ref = R["r8"][idx].max(1); s8 = r[:, VARS.index("pol8_mask_q98")]
    out["repro_pol8_mask_q98_vs_irrm1_r8"] = {"max_abs": float(np.abs(s8 - ref).max()), "median_abs": float(np.median(np.abs(s8 - ref))),
                                              "flag_agreement": float(((s8 > THR) == (ref > THR)).mean())}
    out["gate_G"] = {}
    for nl in NLS:
        a, b = out["variants"][f"pol{nl}_mask_q98"], out["variants"][f"held{nl}_mask_q98"]
        d_rec, d_auc = a["pre_red_recall"] - b["pre_red_recall"], a["pre_red_auc"] - b["pre_red_auc"]
        out["gate_G"][f"NL{nl}"] = {"recall_gain_pp": round(100 * d_rec, 2), "auc_gain": round(d_auc, 4), "pass": bool(d_rec >= 0.10 and d_auc >= 0.03)}
    out["gate_G_pass"] = any(v["pass"] for v in out["gate_G"].values())
    (OUT / "gfx1_grip.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"run": run, "grade": grade}[sys.argv[1]]()

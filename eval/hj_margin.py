"""HJ v2 stage A (docs/HJ_V2.md): a smooth, continuous, label-free failure margin l(z) from the fork test on the HOLD plan
at the current state. Per decision and scale s in SC: the split test's continuous statistics on the imagined hold-10 and
hold-30 endings (model_OIL, 64 futures, perception + execution errors x s): Ashman D, BIC gain (bic1 - bic2), and the
variance along the principal axis (spread). Dev data = the episodes of eval/margin_check.py (paired_record_ocl: topple
windows kt-10..kt + every 4th episode without a topple).
usage: python3 eval/hj_margin.py run [shard nshard] | analyse  -> results/jenga/hj2/margin_rows_<shard>.jsonl, margin_A.json
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
SC = (0.5, 1.0, 2.0, 3.0, 4.0, 8.0)
D = ROOT / "results/jenga/hj2"


def split_stats(e):
    """As cem_filter.split_test_gpu on e (C, n, Dim) -> dict of (C,) arrays: ok, D (Ashman), gain (bic1 - bic2),
    spread (variance along the principal axis per latent dimension)."""
    import torch
    x = e.double(); C, n, Dm = x.shape
    X = x - x.mean(1, keepdim=True)
    w, V = torch.linalg.eigh(X @ X.transpose(1, 2))
    y = V[:, :, -1] * w[:, -1:].clamp_min(0).sqrt()
    ys, order = y.sort(1)
    i = torch.arange(1, n, device=x.device, dtype=x.dtype)[None]
    cs = ys.cumsum(1); tot = cs[:, -1:]
    m1 = cs[:, :-1] / i; m2 = (tot - cs[:, :-1]) / (n - i)
    between = i * (n - i) / n * (m1 - m2) ** 2
    between = torch.where((i >= 2) & (n - i >= 2), between, torch.full_like(between, -1.0))
    split = between.argmax(1) + 1
    low = torch.arange(n, device=x.device)[None] < split[:, None]
    lab = torch.zeros(C, n, device=x.device, dtype=torch.long); lab.scatter_(1, order, (~low).long())
    ya = torch.where(lab == 0, y, torch.nan); yb = torch.where(lab == 1, y, torch.nan)
    na = (lab == 0).sum(1).double(); nb = (lab == 1).sum(1).double()
    ma, mb = ya.nanmean(1), yb.nanmean(1)
    va = ((ya - ma[:, None]) ** 2).nansum(1) / na; vb = ((yb - mb[:, None]) ** 2).nansum(1) / nb
    d = np.sqrt(2) * (ma - mb).abs() / (va + vb).sqrt()
    n2ll = lambda m, v: m * (torch.log(2 * np.pi * v.clamp_min(1e-12)) + 1)
    bic1 = n2ll(torch.full_like(na, n), y.var(1, unbiased=False)) + 2 * np.log(n)
    bic2 = n2ll(na, va) + n2ll(nb, vb) - 2 * (na * torch.log(na / n) + nb * torch.log(nb / n)) + 5 * np.log(n)
    ok = (bic2 < bic1) & (d > 2) & (torch.minimum(na, nb) >= 2)
    f = lambda t: t.float().cpu().numpy()
    return {"ok": f(ok).astype(bool), "D": f(d), "gain": f(bic1 - bic2), "spread": f(w[:, -1] / n / Dm)}


def hold_stats(mon, cbff, wm, z0, target, n_fut=64):
    """Hold plan from the current state at every scale -> {scale: {ending: stats}} (one plan)."""
    torch = mon.torch
    hold = np.repeat(np.r_[target, 0.0][None].astype(np.float32), 16, 0)
    seed = int(mon.rng.integers(2 ** 31)); out = {}
    for s in SC:
        d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (r * [0, 0, 1] * s).reshape(9)])
                       for p, r in (mon.errors.sample(seed, 0, 0, j) for j in range(n_fut))])
        st = mon.g(z0[None].expand(n_fut, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32))
        base = cbff.horizon(hold, cbff.CFG["N_LOOK"]); acts = []
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
        out[s] = {"10": {k: v[0].item() for k, v in split_stats(ends[wm.HOLD10]).items()},
                  "30": {k: v[0].item() for k, v in split_stats(ends[wm.HOLD30]).items()}}
    return out


def run(shard=0, nshard=1):
    import torch
    import cem_filter as cf
    import cbf_filter as cbff
    import inspector_data as insp
    import margin_check as mc
    import step3_world_model as wm
    mon = cf.Monitor(); sim = insp._sim(); D.mkdir(parents=True, exist_ok=True)
    out = D / f"margin_rows_{shard}.jsonl"
    done = {(r["ep"], r["k"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    A = json.loads((ROOT / "results/jenga/outcome_audit_ocl.json").read_text())["paired_record_ocl"]["rows"]
    top = {r["ep"] for r in A if r["logged_neighbour_toppled"]}
    safe = sorted({r["ep"] for r in A} - top)[::4]
    eps = sorted(top | set(safe))[shard::nshard]
    with out.open("a") as fh:
        for ep in eps:
            with np.load(mc.RUN / f"ep{ep}.npz") as z:
                S = {k: z[k] for k in z.files}
            kt = next((c["chunk"] for c in json.loads((mc.RUN / f"ep{ep}.json").read_text())["log"] if c.get("topple_started")), None)
            for d in range(len(S["chunk_index"])):
                k = int(S["chunk_index"][d])
                if (ep, k) in done or (ep in top and (kt is None or not kt - 10 <= k <= kt)):
                    continue
                sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
                mon.rng = np.random.default_rng([20261120, ep, k])
                z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
                st = hold_stats(mon, cbff, wm, z0, S["target"][d])
                fh.write(json.dumps({"ep": ep, "k": k, "kt": kt, "stats": {str(s): v for s, v in st.items()}}) + "\n"); fh.flush()
            print("hjmargin", ep, flush=True)


def margins(r):
    """Candidate margins (higher = safer) from one row's stats."""
    st = r["stats"]; S = [str(s) for s in SC]
    Gj = {s: min(st[s]["10"]["gain"], st[s]["30"]["gain"]) for s in S}            # BIC gain, both endings (> 0 = two modes)
    okj = {s: st[s]["10"]["ok"] and st[s]["30"]["ok"] for s in S}
    first = next((i for i, s in enumerate(S) if okj[s]), len(S))
    h_bin = float(np.log2(SC[first])) if first < len(S) else float(np.log2(16.0))
    # continuous fork distance: log2 scale where the joint BIC gain first crosses 0 (log-linear interpolation), cap log2 16
    # (Ashman D is NOT usable: the best 2-way split of a unimodal 64-sample cloud already gives D ~ 3 > 2)
    ls = np.log2(SC); gv = np.array([Gj[s] for s in S]); h_c = float(np.log2(16.0))
    for j in range(len(S)):
        if gv[j] > 0:
            h_c = float(ls[0]) if j == 0 else float(ls[j - 1] + (0 - gv[j - 1]) / max(gv[j] - gv[j - 1], 1e-6) * (ls[j] - ls[j - 1]))
            break
    return {"h_bin": h_bin, "h_cont": h_c, "negG_2": -Gj["2.0"], "negG_4": -Gj["4.0"],
            "negG_max": -max(Gj.values()),
            "neg_log_sens_2": -float(np.log(max(st["2.0"]["30"]["spread"], 1e-12) / 4.0)),
            "neg_log_sens_4": -float(np.log(max(st["4.0"]["30"]["spread"], 1e-12) / 16.0))}


def analyse():
    R = [json.loads(l) for f in sorted(D.glob("margin_rows_*.jsonl")) for l in f.read_text().splitlines()]
    for r in R:
        r["m"] = margins(r)
    keys = list(R[0]["m"])
    safe = [r for r in R if r["kt"] is None]; by = {}
    for r in safe:
        by.setdefault(r["ep"], []).append(r)
    out = {"rows": len(R), "safe_decisions": len(safe), "topple_episodes": len({r["ep"] for r in R if r["kt"] is not None}), "margins": {}}
    for key in keys:
        v = np.array([r["m"][key] for r in safe]); iqr = float(np.subtract(*np.percentile(v, [75, 25])))
        steps = [abs(b["m"][key] - a["m"][key]) for rs in by.values() for a, b in zip(sorted(rs, key=lambda x: x["k"]), sorted(rs, key=lambda x: x["k"])[1:]) if b["k"] == a["k"] + 1]
        res = {"distinct_values_safe": int(len(np.unique(np.round(v, 4)))), "iqr": iqr,
               "smooth_median_step_over_iqr": float(np.median(steps) / iqr) if iqr > 0 else None,
               "smooth_mean_step_over_sd": float(np.mean(steps) / v.std()) if v.std() > 0 else None}
        for lo, hi in ((1, 3), (4, 6), (7, 10)):
            pos = np.array([r["m"][key] for r in R if r["kt"] is not None and lo <= r["kt"] - r["k"] <= hi])
            res[f"auc_{lo}-{hi}_before"] = float(((-pos[:, None] > -v[None]).mean() + 0.5 * (pos[:, None] == v[None]).mean())) if len(pos) else None
        out["margins"][key] = res
    (D / "margin_A.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    run(*(int(x) for x in sys.argv[2:4])) if sys.argv[1] == "run" else analyse()

"""IFD1 (results/jenga/ifd1_protocol.json, offline diagnosis): why does the corrected irreversible-fragility value (GFX1 fx,
value_nsif16_alea.pt, flat target: consistent irreversibility -> 0) topple more than ours90 at 1x noise?
Hypothesis: at 1x the moves before those topples are irreversible in MOST imagined futures (median draw > tau), so the flat
target exempted them as 'consistent' (intended) instead of flagging them as fragile.
Decisions: 1-3 chunks before every logged neighbour topple in the H2H5 runs of fx and ours90 (1x and 2x). For each, the
latent is re-encoded from the saved sim state (fixed camera, as eval/hj_data2.py), then the undo test of the policy's own
plan is run exactly as the value's labels (eval/irrm2.py: gripper held, 64 draws, CF3's error model and snippets) at
NL 8 and 16, imagined noise 2x (the labels) and 1x (reported), over all patches (primary, user 2026-11-01: no hand mask)
and outside the hand mask (as fx's labels were made, reported).
usage: python3 eval/ifd1.py run -> results/jenga/ifd1_rows.npz;  python3 eval/ifd1.py grade -> results/jenga/ifd1_result.json"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CBF_WM", str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"))
os.environ.setdefault("CBF_CALIB", "results/jenga/cbf_calibration_OIL.json")
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
PR, OUT = ROOT / "results/program_b/policy_rollouts", ROOT / "results/jenga"
RUNS = ("paired_lfc000_h5fx_1x", "paired_lfc000_h5o90_1x", "paired_lfc000_h5fx_2x", "paired_lfc000_h5o90_2x")
NLS, SCALES, THR = (8, 16), (2.0, 1.0), 1.3356
VARS = [f"nl{nl}_s{s:g}_{m}" for nl in NLS for s in SCALES for m in ("all", "mask")]


def topples(run):
    out = {}
    for f in sorted((PR / run).glob("ep*.json")):
        kt = next((c["chunk"] for c in json.loads(f.read_text())["log"] if c.get("topple_started")), -1)
        if kt >= 0:
            out[int(f.stem[2:])] = int(kt)
    return out


def run():
    lim = int(os.environ.get("IFD1_LIMIT", 10**9)); n_dec = 0
    import torch
    import cbf_barrier as cb
    import cbf_filter as cbff
    import cem_filter as cf
    import cf_fork as C
    import cf_rev as V
    import inspector_data as insp
    import step3b_models as sm
    import step3_world_model as wm
    import visual_pipeline_panel3 as p3
    import uncertainty_monitor_dev as um
    mon = cf.Monitor(); sim = insp._sim(); model = mon.model
    G = sm.G.load(cf.V3 / "G.pt"); E = p3.ErrorModel(); snip = np.load(um.BENCH)["snippets"]
    S0, total = cbff.CFG["PLAN_START"], 2 * wm.STEPS
    res = {k: [] for k in ("run", "ep", "k", "kt", "closed", "r")}
    for ri, run_ in enumerate(RUNS):
        for ep, kt in topples(run_).items():
            with np.load(PR / run_ / f"ep{ep}.npz") as d:
                S = {k: d[k] for k in d.files}
            for k in (kt - 3, kt - 2, kt - 1):
                hit = np.flatnonzero(S["chunk_index"] == k)
                if k < 0 or not len(hit):
                    continue
                i = int(hit[0]); n_dec += 1
                if n_dec > lim:
                    break
                sim.restore((S["state"][i], S["ctrl"][i], S["target"][i], float(S["gripper"][i])))
                mon.rng = np.random.default_rng([20261121, ep, k])
                z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
                site = np.asarray(sim.data.site_xpos[sim.site], np.float32)
                plan = np.asarray(S["plan16"][i], np.float32); cur = np.r_[S["target"][i], plan[0, 3]].astype(np.float32)
                seed = C.SEED + 1000 * ep + i
                out = []
                for nl in NLS:
                    rev = V.reverse_plan(plan, cur, nl, S0)
                    rev = np.concatenate([rev[S0:], np.repeat(rev[-1:], total, 0)])[:total]; rev[:, 3] = cur[3]
                    hold = np.repeat(cur[None], total, 0)
                    mask = torch.as_tensor(~cb.hand_mask(np.vstack([site[None], plan[S0:S0 + nl, :3]])), device="cuda").reshape(-1)
                    for s in SCALES:
                        d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (r * [0, 0, 1] * s).reshape(9)])
                                       for p, r in (E.sample(seed, 0, 0, j) for j in range(C.N_FUT))])
                        st = G(z0[None].expand(C.N_FUT, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32)).float()
                        acts = []
                        for base in (rev, hold):
                            for j in range(C.N_FUT):
                                a = base.copy(); a[:nl, :3] -= np.concatenate([snip[j % 64], snip[(j + 1) % 64]])[:nl] * s; acts.append(a)
                        pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(2 * C.N_FUT, wm.STEPS, -1)
                        z = st.repeat(2, 1, 1, 1)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            for t in range(wm.STEPS):
                                z = model.step(z, pairs[:, t]).float()
                        z = z.float().reshape(2, C.N_FUT, *z.shape[1:])
                        dz = (z[0] - z[1]).abs().mean(-1).reshape(C.N_FUT, -1)
                        out += [torch.quantile(dz, 0.98, dim=1).cpu().numpy(), torch.quantile(dz[:, mask], 0.98, dim=1).cpu().numpy()]
                for key, v in (("run", ri), ("ep", ep), ("k", k), ("kt", kt), ("closed", float(S["gripper"][i]) < 50)):
                    res[key].append(v)
                res["r"].append(np.stack(out).astype(np.float32))
            print("ifd1", run_, ep, flush=True)
    np.savez_compressed(OUT / "ifd1_rows.npz", vars=np.array(VARS), runs=np.array(RUNS), **{k: np.asarray(v) for k, v in res.items()})
    print("saved", flush=True)


def grade():
    D = np.load(OUT / "ifd1_rows.npz"); r = D["r"]; run_, ep = D["run"], D["ep"]
    top = {n: set(topples(n)) for n in RUNS}
    out = {"protocol": "results/jenga/ifd1_protocol.json", "tau": THR, "groups": {}}
    for ri, name in enumerate(RUNS):
        sel = run_ == ri; g = {"decisions": int(sel.sum()), "episodes": int(len(set(ep[sel])))}
        for v, var in enumerate(VARS):
            x = r[sel, v]; mx, md = x.max(1), np.median(x, 1)
            g[var] = {"consistent": round(float((md > THR).mean()), 3), "fragile": round(float(((mx > THR) & (md <= THR)).mean()), 3),
                      "none": round(float((mx <= THR).mean()), 3)}
        out["groups"][name] = g
    disc = sorted(top[RUNS[0]] - top[RUNS[1]])                                   # fx topples at 1x, ours90 does not
    v = VARS.index("nl16_s2_all")
    per = {}
    for e in disc:
        x = r[(run_ == 0) & (ep == e), v]
        mx, md = x.max(1), np.median(x, 1)
        per[int(e)] = {"decisions": int(len(x)), "consistent": int((md > THR).sum()), "fragile": int(((mx > THR) & (md <= THR)).sum()),
                       "none": int((mx <= THR).sum()), "exempted": bool((md > THR).any() and not ((mx > THR) & (md <= THR)).any())}
    n_ex = sum(p["exempted"] for p in per.values())
    out["discordant_fx_1x"] = {"episodes": len(disc), "per_episode": per, "exempted": n_ex}
    need = (len(disc) + 1) // 2 + (1 if len(disc) % 2 == 0 else 0)                # strict majority
    out["H_exempt"] = {"rule": "strict majority of the discordant 1x episodes exempted (nl16, imagined 2x, all patches)",
                       "needed": need, "supported": bool(n_ex >= need)}
    (OUT / "ifd1_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"run": run, "grade": grade}[sys.argv[1]]()

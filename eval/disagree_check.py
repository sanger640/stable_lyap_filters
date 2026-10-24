"""Baseline monitor (exploratory): epistemic uncertainty = world-model ENSEMBLE DISAGREEMENT (model_OIL + the 4 ens_v1
certifier models), vs the fork test, as an early-warning alarm at a matched alarm rate. Same decisions as
eval/margin_check.py (policy-alone episodes; default paired_record_ocl = dev). At each decision, the policy's plan and
hold are rolled through all 5 models from the same 16 perturbed starts / execution draws at scale 1 (as the fork test);
disagreement = mean over futures of the across-model variance of the HOLD30 latent, over the whole frame (d_all) and
outside the hand (d_out, the o mask). Rules: d_plan > tau and d_plan - d_hold > tau (relative, as the trigger).
usage: WM_COMPILE=1 python3 eval/disagree_check.py run | analyse   (MC_RUN / MC_TAG as margin_check)
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

N_FUT = 16
OUT = ROOT / "results/jenga/deep_dive" / f"disagree_rows{mc.TAG}.jsonl"


def disagreement(models, cbf, sim, z0, plans, seed):
    import cbf_filter as cbff
    import step3_world_model as wm
    mon, cb, torch = cbf.mon, cbf.cb, cbf.mon.torch
    C = len(plans)
    d_ = np.stack([np.concatenate([(p * [1, 1, 0]).reshape(9), (r * [0, 0, 1]).reshape(9)])
                   for p, r in (mon.errors.sample(seed, 0, 0, j) for j in range(N_FUT))])
    st = mon.g(z0[None].expand(N_FUT, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32))
    acts = []
    for p in plans:
        base = cbff.horizon(p, cbff.CFG["N_LOOK"])
        for j in range(N_FUT):
            a = base.copy()
            a[:cbff.CFG["N_LOOK"], :3] -= np.concatenate([mon.snippets[j % 64], mon.snippets[(j + 1) % 64]])[:cbff.CFG["N_LOOK"]]
            acts.append(a)
    pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(C * N_FUT, wm.STEPS, -1)
    ends = []
    for m in models:
        z = st.float().repeat(C, 1, 1, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(wm.HOLD30 + 1):
                z = m.step(z, pairs[:, t]).float()
        ends.append(z.reshape(C, N_FUT, 28, 28, 8))
    E = torch.stack(ends)                                                  # (M, C, F, 28, 28, 8)
    var = E.var(0).mean(-1)                                                # (C, F, 28, 28)
    d_all, d_out = [], []
    for c in range(C):
        pts = np.vstack([sim.data.site_xpos[sim.site][None],
                         np.asarray(plans[c])[cbff.CFG["PLAN_START"]:cbff.CFG["PLAN_START"] + cbff.CFG["N_LOOK"], :3]])
        msk = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
        d_all.append(float(var[c].mean()))
        d_out.append(float(((var[c] * msk).sum((1, 2)) / msk.sum().clamp_min(1)).mean()))
    return d_all, d_out


def run():
    import torch
    import cem_filter as cf
    import cbf_filter as cbff
    import inspector_data as insp
    import step3_world_model as wm
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    models = [mon.model] + [wm.WorldModel.load(ROOT / f"results/program_b/ens_v1/model_ENS{k}.pt") for k in range(4)]
    keys = {(r["ep"], r["k"]) for f in sorted(mc.DT.glob("margin_rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}
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
                z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
                plan = S["plan16"][d].copy(); plan[0, :3] = S["target"][d]
                hold = np.repeat(np.r_[S["target"][d], 0.0][None].astype(np.float32), 16, 0)
                da, do = disagreement(models, cbf, sim, z0, np.stack([plan, hold]), 20261124 + 7919 * ep + k)
                fh.write(json.dumps({"ep": ep, "k": k, "d_all": da, "d_out": do}) + "\n"); fh.flush()
            print("dis", ep, flush=True)


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
        leads, n_t, per = [], 0, {}
        for e, ks in by.items():
            if topple.get(e) and kt.get(e) is not None:
                n_t += 1
                hits = [k for k in sorted(ks) if kt[e] - 8 <= k <= kt[e] and ks[k]]
                per[e] = bool(hits) and kt[e] - hits[0] >= 3
                if hits:
                    leads.append(kt[e] - hits[0])
        evaluate.last = per
        return {"alarm_rate_safe": round(float(np.mean(np.asarray(fire)[safe])), 4), "topple_episodes": n_t, "warned_within_8": len(leads),
                "lead_ge3_share_of_topples": round(sum(x >= 3 for x in leads) / max(n_t, 1), 3)}

    hp = np.array([sp.h_of(M[k]["plan"]) for k in keys]); hh = np.array([sp.h_of(M[k]["hold"]) for k in keys])
    o = np.array([M[k]["o"] for k in keys])
    fork = (hp < np.minimum(hh, 2.0) - 1e-9) | (o > O_MAX)
    out = {"decisions": len(keys), "exploratory": mc.TAG != "_m3", "fork_3x": evaluate(fork)}
    per_fork = evaluate.last
    rate = out["fork_3x"]["alarm_rate_safe"]
    for name, s in (("d_all_plan", [rows[k]["d_all"][0] for k in keys]), ("d_out_plan", [rows[k]["d_out"][0] for k in keys]),
                    ("d_all_rel", [rows[k]["d_all"][0] - rows[k]["d_all"][1] for k in keys]),
                    ("d_out_rel", [rows[k]["d_out"][0] - rows[k]["d_out"][1] for k in keys])):
        s = np.asarray(s); tau = float(np.quantile(s[safe], 1 - rate))
        out[name + "_matched"] = {"tau": tau, **evaluate(s > tau)}
        if name == "d_all_plan":                                           # primary (results/jenga/disagree_protocol.json)
            E = sorted(per_fork); x = np.array([per_fork[e] for e in E], float) - np.array([evaluate.last[e] for e in E], float)
            rng = np.random.default_rng(20261124); bs = np.array([x[rng.integers(len(x), size=len(x))].mean() for _ in range(10000)])
            out["PRIMARY"] = {"fork_minus_disagreement_pp": round(100 * float(x.mean()), 1), "lower95_one_sided_pp": round(100 * float(np.percentile(bs, 5)), 1),
                              "fork_only": int((x > 0).sum()), "disagreement_only": int((x < 0).sum())}
            out["PRIMARY"]["pass"] = bool(out["PRIMARY"]["fork_minus_disagreement_pp"] >= 10 and out["PRIMARY"]["lower95_one_sided_pp"] > 0)
        out[name + "_or_o_matched"] = None
        for q in np.linspace(0.5, 1.0, 201):                               # + the same o barrier, matched rate
            t2 = float(np.quantile(s[safe], q)); f = (s > t2) | (o > O_MAX)
            if np.mean(f[safe]) <= rate + 1e-9:
                out[name + "_or_o_matched"] = {"tau": t2, **evaluate(f)}; break
    (ROOT / f"results/jenga/deep_dive/disagree_check{mc.TAG}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"run": run, "analyse": analyse}[sys.argv[1]]()

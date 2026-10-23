"""Exploratory: red-block falls (the target block knocked over or dropped) are now the largest failure class under G-CEM
(ocl test: 19 before + 34 after extraction vs 22 neighbour topples, policy 21 + 25), and the filter does not reduce them.
Does the monitor see them coming? Policy-alone episodes of the O-CEM closed-loop test (paired_record_ocl).
Fall decision = the saved decision whose chunk contains the first step with red tilt >= 45 deg (eval/outcome_audit.replay).
Rows: decisions df-8..df of every red-fall episode without a neighbour topple, and all decisions of every 6th episode with
neither (reference). Per decision: split flags of the plan and of hold at SC (as eval/margin_check.py) and three
disturbance readouts of the plan at scale 1: o (outside the hand, the barrier), o_full (whole frame), o_hand (hand region).
usage: python3 eval/redfall_check.py run | analyse  -> results/jenga/deep_dive/redfall_rows.jsonl, redfall_check.json
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

RUN, D, SC = mc.RUN, mc.D, mc.SC
OUT = D / "redfall_rows.jsonl"


def flags(cbf, sim, z0, plans, n_fut=64):
    """As margin_check.split_flags, plus o_full / o_hand of plan 0 at scale 1."""
    import cbf_filter as cbff
    import cem_filter as cf
    import step3_world_model as wm
    mon, cb, torch = cbf.mon, cbf.cb, cbf.mon.torch
    C = len(plans); seed = int(mon.rng.integers(2 ** 31)); split = np.zeros((C, len(SC)), bool); o = {}
    for si, s in enumerate(SC):
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
        ok10, _, _ = cf.split_test_gpu(ends[wm.HOLD10]); ok30, _, _ = cf.split_test_gpu(ends[wm.HOLD30])
        split[:, si] = (ok10 & ok30).cpu().numpy()
        if s == 1.0:
            pts = np.vstack([sim.data.site_xpos[sim.site][None],
                             np.asarray(plans[0])[cbff.CFG["PLAN_START"]:cbff.CFG["PLAN_START"] + cbff.CFG["N_LOOK"], :3]])
            m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
            diff = (ends[wm.HOLD30][0].reshape(n_fut, 28, 28, 8) - z0[None]).pow(2).mean(-1)
            for name, mm in (("o", m), ("o_full", torch.ones_like(m)), ("o_hand", 1 - m)):
                o[name] = float(((diff * mm).sum((1, 2)) / mm.sum().clamp_min(1)).max())
    return split, o


def episodes():
    A = json.loads((ROOT / "results/jenga/outcome_audit_ocl.json").read_text())["paired_record_ocl"]["rows"]
    fall = {r["ep"]: ("before" if r["red_fell_before_extraction"] else "after") for r in A
            if not r["logged_neighbour_toppled"] and (r["red_fell_before_extraction"] or r["red_fell_after_extraction"])}
    ref = sorted(r["ep"] for r in A if not r["logged_neighbour_toppled"] and r["ep"] not in fall)[::6]
    return fall, ref


def run():
    import torch
    import cem_filter as cf
    import cbf_filter as cbff
    import inspector_data as insp
    import outcome_audit as oa
    import plausibility as pl
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    done = {(r["ep"], r["k"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    fall, ref = episodes()
    with OUT.open("a") as fh:
        for ep in sorted(set(fall) | set(ref)):
            with np.load(RUN / f"ep{ep}.npz") as z:
                S = {k: z[k] for k in z.files}
            df = None
            if ep in fall:
                _, TT, _, _ = oa.replay(sim, S, pl.episode_noises(ep, int(S["chunk_index"][-1]) + 1))
                df = int(np.argmax(TT >= 45)) // 8
            for d in range(len(S["chunk_index"])):
                k = int(S["chunk_index"][d])
                if (ep, k) in done or (df is not None and not df - 8 <= d <= df):
                    continue
                sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
                mon.rng = np.random.default_rng([20261100, ep, k])                         # = margin_check's draws
                z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
                plan = S["plan16"][d].copy(); plan[0, :3] = S["target"][d]
                hold = np.repeat(np.r_[S["target"][d], 0.0][None].astype(np.float32), 16, 0)
                split, o = flags(cbf, sim, z0, np.stack([plan, hold]))
                fh.write(json.dumps({"ep": ep, "k": k, "d": d, "df": df, "kind": fall.get(ep, "ref"), "plan": split[0].tolist(),
                                     "hold": split[1].tolist(), **o}) + "\n"); fh.flush()
            print("redfall", ep, fall.get(ep, "ref"), flush=True)


def analyse():
    R = [json.loads(l) for l in OUT.read_text().splitlines()]
    O_MAX = json.loads((ROOT / os.environ["CBF_CALIB"]).read_text())["O_MAX"]
    first = lambda fl: next((i for i, v in enumerate(fl) if v), len(SC))
    rel = lambda r, mi: first(r["plan"]) <= mi and first(r["plan"]) < first(r["hold"])
    ref = [r for r in R if r["kind"] == "ref"]
    out = {"rows": len(R), "fall_episodes": {k: len({r["ep"] for r in R if r["kind"] == k}) for k in ("before", "after")},
           "reference_decisions": len(ref), "exploratory": True}
    rules = {"G trigger (relative d<=2x or o>O_MAX)": lambda r: rel(r, 2) or r["o"] > O_MAX,
             "relative d<=4x or o>O_MAX": lambda r: rel(r, 4) or r["o"] > O_MAX,
             "plan forks at <=2x (absolute)": lambda r: first(r["plan"]) <= 2,
             "plan forks at <=4x (absolute)": lambda r: first(r["plan"]) <= 4}
    for q in (0.9, 0.95):
        for key in ("o_full", "o_hand"):
            t = float(np.quantile([r[key] for r in ref], q))
            rules[f"{key} > reference q{q}"] = (lambda key, t: (lambda r: r[key] > t))(key, t)
    for name, f in rules.items():
        res = {"alarm_rate_reference": round(float(np.mean([f(r) for r in ref])), 4)}
        for kind in ("before", "after"):
            eps = sorted({r["ep"] for r in R if r["kind"] == kind}); leads = []
            for e in eps:
                hits = [r["d"] for r in R if r["ep"] == e and f(r)]
                if hits:
                    leads.append(r_df(R, e) - min(hits))
            res[kind] = {"episodes": len(eps), "warned_within_8": len(leads),
                         "lead_ge_2": int(sum(v >= 2 for v in leads)), "lead_median": float(np.median(leads)) if leads else None}
        out[name] = res
    for key in ("o", "o_full", "o_hand"):
        neg = np.array([r[key] for r in ref])
        for kind in ("before", "after"):
            pos = np.array([r[key] for r in R if r["kind"] == kind and 1 <= r["df"] - r["d"] <= 4])
            out[f"auc_{key}_{kind}_1-4_before"] = float((pos[:, None] > neg[None]).mean()) if len(pos) else None
    (D / "redfall_check.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def r_df(R, e):
    return next(r["df"] for r in R if r["ep"] == e)


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else analyse()

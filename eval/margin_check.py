"""Exploratory (docs/FILTER_DEEP_DIVE.md section 5): would a LARGER barrier margin (fork test at 3x / 4x the execution and
perception errors) warn earlier before the topples that the current trigger (fork at <= 2x) catches too late?
Policy-alone episodes of the O-CEM closed-loop test (paired_record_ocl, 300 episodes, every decision state saved).
At every decision: split flags of the policy's plan and of hold at scales SC (model_OIL, 64 futures, as cbf.score),
plus o of the plan. Analysis: per trigger rule, alarm rate on episodes without a topple, and on topple episodes the lead
(chunks between the first alarm in the 8 chunks before the topple and the topple start).
usage: python3 eval/margin_check.py run [shard nshard] ; analyse
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
SC = (0.5, 1.0, 2.0, 3.0, 4.0)
RUN = ROOT / "results/program_b/policy_rollouts" / os.environ.get("MC_RUN", "paired_record_ocl")   # MC_RUN / MC_TAG: other
TAG = os.environ.get("MC_TAG", "")                                   # policy-alone runs (held-out gate sets)
DT = ROOT / "results/jenga/deep_dive" / (TAG.strip("_") if TAG else "")   # tagged runs write to their own folder
SEED = int(os.environ.get("MC_SEED", 20261100))                     # other seeds (retest): MC_SEED + MC_OUT folder
if os.environ.get("MC_OUT"):
    DT = ROOT / "results/jenga/deep_dive" / os.environ["MC_OUT"]
AUDIT = (ROOT / f"results/jenga/outcome_audit_{TAG.strip('_') or 'ocl'}.json", RUN.name)
D = ROOT / "results/jenga/deep_dive"


def split_flags(cbf, sim, z0, plans, n_fut=64):
    import cbf_filter as cbff
    import cem_filter as cf
    import step3_world_model as wm
    mon, cb, torch = cbf.mon, cbf.cb, cbf.mon.torch
    C = len(plans); seed = int(mon.rng.integers(2 ** 31)); split = np.zeros((C, len(SC)), bool); o = np.zeros(C)
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
            for c in range(C):
                pts = np.vstack([sim.data.site_xpos[sim.site][None],
                                 np.asarray(plans[c])[cbff.CFG["PLAN_START"]:cbff.CFG["PLAN_START"] + cbff.CFG["N_LOOK"], :3]])
                m = torch.as_tensor(~cb.hand_mask(pts), device="cuda")
                diff = (ends[wm.HOLD30][c].reshape(n_fut, 28, 28, 8) - z0[None]).pow(2).mean(-1)
                o[c] = float(((diff * m).sum((1, 2)) / m.sum().clamp_min(1)).max())
    return split, o


def run(shard=0, nshard=1):
    import torch
    import cem_filter as cf
    import cbf_filter as cbff
    import inspector_data as insp
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    if os.environ.get("TASK"):                                         # cluster array (py_array.sbatch)
        shard, nshard = (int(x) for x in os.environ["TASK"].split("/"))
    out = DT / f"margin_rows_{shard}.jsonl"; done = set(); DT.mkdir(parents=True, exist_ok=True)
    if out.exists():
        done = {(r["ep"], r["k"]) for r in map(json.loads, out.read_text().splitlines())}
    A = json.loads(AUDIT[0].read_text())[AUDIT[1]]["rows"]
    top = {r["ep"] for r in A if r["logged_neighbour_toppled"]}
    safe = sorted({r["ep"] for r in A} - top)[::4]                      # alarm rate: every 4th episode without a topple
    files = [RUN / f"ep{e}.npz" for e in sorted(top | set(safe))][shard::nshard]
    with out.open("a") as fh:
        for f in files:
            ep = int(f.stem[2:])
            with np.load(f) as z:
                S = {k: z[k] for k in z.files}
            kt = next((c["chunk"] for c in json.loads((RUN / f"ep{ep}.json").read_text())["log"] if c.get("topple_started")), None)
            for d in range(len(S["chunk_index"])):
                k = int(S["chunk_index"][d])
                if (ep, k) in done or (ep in top and (kt is None or not kt - 10 <= k <= kt)):
                    continue
                sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
                mon.rng = np.random.default_rng([SEED, ep, k])
                z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
                plan = S["plan16"][d].copy(); plan[0, :3] = S["target"][d]
                hold = np.repeat(np.r_[S["target"][d], 0.0][None].astype(np.float32), 16, 0)
                split, o = split_flags(cbf, sim, z0, np.stack([plan, hold]))
                fh.write(json.dumps({"ep": ep, "k": k, "plan": split[0].tolist(), "hold": split[1].tolist(), "o": float(o[0])}) + "\n")
                fh.flush()
            print("margin", ep, flush=True)


def analyse():
    import cbf_filter as cbff
    O_MAX = json.loads((ROOT / os.environ["CBF_CALIB"]).read_text())["O_MAX"]
    R = [json.loads(l) for f in sorted(D.glob("margin_rows_*.jsonl")) for l in f.read_text().splitlines()]
    A = json.loads((ROOT / "results/jenga/outcome_audit_ocl.json").read_text())["paired_record_ocl"]["rows"]
    topple = {r["ep"]: r["logged_neighbour_toppled"] for r in A}
    kt = {}
    for e in topple:
        L = json.loads((RUN / f"ep{e}.json").read_text())["log"]
        kt[e] = next((c["chunk"] for c in L if c.get("topple_started")), None)
    by = {}
    for r in R:
        by.setdefault(r["ep"], {})[r["k"]] = r
    out = {"decisions": len(R), "episodes": len(by), "exploratory": True}
    for m, name in ((3, "fork_le_2x (current, absolute)"), (4, "fork_le_3x"), (5, "fork_le_4x")):
        alarm = (lambda mm: (lambda r: bool(any(r["plan"][:mm])) or r["o"] > O_MAX))(m)
        safe = [alarm(r) for e, ks in by.items() if not topple.get(e) for r in ks.values()]
        leads, caught = [], 0; n_t = 0
        for e, ks in by.items():
            if not topple.get(e) or kt.get(e) is None:
                continue
            n_t += 1
            hits = [k for k in sorted(ks) if kt[e] - 8 <= k <= kt[e] and alarm(ks[k])]
            if hits:
                caught += 1; leads.append(kt[e] - hits[0])
        out[name] = {"alarm_rate_safe_episodes": round(float(np.mean(safe)), 4), "topple_episodes": n_t,
                     "warned_within_8": caught, "lead_median": float(np.median(leads)) if leads else None,
                     "lead_ge_3_share": round(float(np.mean([x >= 3 for x in leads])), 3) if leads else None,
                     "lead_hist": {int(v): int(c) for v, c in zip(*np.unique(leads, return_counts=True))}}
    (D / "margin_check.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*(int(x) for x in sys.argv[2:4]))
    else:
        analyse()

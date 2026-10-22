"""Pillar filter failure diagnostic (exploratory, after the failed universality test; docs/PILLAR_TASK.md "Diagnostic").
Replays the policy-alone (record) episodes 5000-5299 deterministically and, at chosen moments, measures the TRUE
episode-failure risk of one intervention: execute option X for this one chunk, then the policy alone to the episode's end,
under 16 noise draws (common across options). Options: pol (the policy's chunk), g (G-CEM's chunk, recomputed here),
hold (hold one chunk, with execution noise as in the test), still (hold one chunk, no execution noise).
Moments: A = every decision the record arm's trigger flagged (all options); B = every decision of the 33 failed episodes
and of 33 random successful ones (pol only; for the trigger's recall).
usage: python3 eval/pillar_diag.py prep ; eval [procs] ; report
"""
import copy
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PILLAR_HALF_W", "0.008"); os.environ.setdefault("PILLAR_PERC", "0.002")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
REC = ROOT / "results/pillar/filter/record"
OUT = ROOT / "results/pillar/diag"
STORE = Path.home() / "wksp/pillar_diag"
DRAWS, SEEDS = 16, range(5000, 5300)


def is_fail(oc):
    return bool(oc["cube_fell"] or oc["pillar_tipped"])


def choose():
    rec = {s: json.loads((REC / f"s{s}.json").read_text()) for s in SEEDS}
    failed = [s for s in SEEDS if is_fail(rec[s]["outcome"])]
    ok = [s for s in SEEDS if not is_fail(rec[s]["outcome"])]
    ctrl = sorted(np.random.default_rng(5).choice(ok, len(failed), replace=False).tolist())
    want = {}
    for s in SEEDS:
        ks = {r["k"] for r in rec[s]["log"] if r["trig"]}
        if s in failed or s in ctrl:
            ks |= {r["k"] for r in rec[s]["log"]}
        if ks:
            want[s] = ks
    return rec, failed, ctrl, want


def stage_prep():
    """GPU: replay record episodes, store state + policy state + options at the chosen moments."""
    import pillar_filter as F
    import pillar_task as P
    import pillar_wm as W
    import uncertainty_monitor_dev as um
    rec, failed, ctrl, want = choose()
    tau = json.loads((F.OUT / "tau.json").read_text())["tau"]
    sc = F.Scorer(); snippets = np.load(um.BENCH)["snippets"]; STORE.mkdir(parents=True, exist_ok=True)
    for s, ks in want.items():
        f = STORE / f"s{s}.pkl"
        if f.exists():
            continue
        trig = {r["k"] for r in rec[s]["log"] if r["trig"]}
        sim = P.PillarSim(); P.reset(sim, s); pol = P.ExpertPolicy(sim, P.PERC, s); cam = W.Cam(sim)
        rng = np.random.default_rng([20261502, s]); pillar0 = P.body_pos(sim, "pillar"); moments = []
        for k in range(30):
            plan = pol.plan(); chunk = plan[:8].copy()
            if k in ks:
                m = {"k": k, "snap": sim.snapshot(), "pol_state": copy.deepcopy((pol.rng.bit_generator.state, pol.opened_on_pillar, pol.closing)),
                     "triggered": k in trig, "options": {"pol": chunk}}
                if k in trig:
                    log = []
                    hook = F.make_hook("g", sc, cam, tau, log, s, rng=np.random.default_rng([99, s, k]), force=True)
                    m["options"]["g"] = hook(sim, plan, k); m["g_kind"] = log[-1].get("kind")
                    hold = np.repeat(np.r_[sim.target, plan[0, 3]][None], 8, 0).astype(np.float32)
                    m["options"]["hold"] = hold; m["options"]["still"] = hold
                moments.append(m)
            nz = snippets[rng.integers(len(snippets))] * F.NOISE_SCALE
            for j, a in enumerate(chunk):
                a = a.copy(); a[:3] -= nz[j]; sim.execute(a)
            if pol.opened_on_pillar and sim.data.site_xpos[sim.site][2] > P.body_pos(sim, "pillar")[2] + 0.12:
                break
        for _ in range(15):
            sim.execute(np.r_[sim.target, -1.0])
        assert P.outcome(sim, pillar0)["success"] == rec[s]["outcome"]["success"], f"replay mismatch seed {s}"
        f.write_bytes(pickle.dumps({"seed": s, "pillar0": pillar0, "moments": moments}))
    print("prep done", len(want), flush=True)


def eval_job(path):
    import pillar_task as P
    import uncertainty_monitor_dev as um
    out = OUT / "rows" / (Path(path).stem + ".json")
    if out.exists():
        return
    D = pickle.loads(Path(path).read_bytes()); s = D["seed"]
    snippets = np.load(um.BENCH)["snippets"]; sim = P.PillarSim(); P.reset(sim, s); pol = P.ExpertPolicy(sim, P.PERC, s)
    rows = []
    for m in D["moments"]:
        k, row = m["k"], {"seed": s, "k": m["k"], "triggered": m["triggered"], "g_kind": m.get("g_kind")}
        for name, opt in m["options"].items():
            fails = []
            for j in range(DRAWS):
                sim.restore(m["snap"]); st = m["pol_state"]
                pol.rng.bit_generator.state = copy.deepcopy(st[0]); pol.opened_on_pillar, pol.closing = st[1], st[2]
                r = np.random.default_rng([77, s, k, j]); nz = snippets[r.integers(len(snippets))] * 2.0
                if name == "still":
                    nz = np.zeros_like(nz)
                for jj, a in enumerate(opt):
                    a = np.array(a, np.float32); a[:3] -= nz[jj]; sim.execute(a)
                for kk in range(k + 1, 30):
                    if pol.opened_on_pillar and sim.data.site_xpos[sim.site][2] > P.body_pos(sim, "pillar")[2] + 0.12:
                        break
                    nz = snippets[r.integers(len(snippets))] * 2.0
                    for jj, a in enumerate(pol.plan()[:8]):
                        a = a.copy(); a[:3] -= nz[jj]; sim.execute(a)
                for _ in range(15):
                    sim.execute(np.r_[sim.target, -1.0])
                fails.append(is_fail(P.outcome(sim, D["pillar0"])))
            row[f"risk_{name}"] = float(np.mean(fails))
        rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rows))


def stage_eval(procs=16):
    import multiprocessing as mp
    paths = sorted(str(p) for p in STORE.glob("s*.pkl"))
    with mp.get_context("spawn").Pool(int(procs)) as pool:
        for i, _ in enumerate(pool.imap_unordered(eval_job, paths), 1):
            if i % 20 == 0:
                print("eval", i, len(paths), flush=True)
    print("eval done", flush=True)


def stage_report():
    from wm_libero import auroc
    rec, failed, ctrl, want = choose()
    rows = [r for f in sorted((OUT / "rows").glob("*.json")) for r in json.loads(f.read_text())]
    dmap = {(s, r["k"]): r["d"] for s in SEEDS for r in rec[s]["log"]}
    A = [r for r in rows if r["triggered"]]
    out = {"A_triggered_moments": len(A)}
    for o in ("pol", "g", "hold", "still"):
        out[f"A_mean_risk_{o}"] = float(np.mean([r[f"risk_{o}"] for r in A]))
    for o in ("g", "hold", "still"):
        dd = np.array([r[f"risk_{o}"] - r["risk_pol"] for r in A])
        out[f"A_{o}_minus_pol"] = {"mean": float(dd.mean()), "better": int((dd < 0).sum()), "worse": int((dd > 0).sum())}
    hi = [r for r in A if r["risk_pol"] >= 0.25]
    out["A_high_risk_triggered (risk_pol >= 0.25)"] = len(hi)
    for o in ("pol", "g", "hold", "still"):
        out[f"A_high_mean_risk_{o}"] = float(np.mean([r[f"risk_{o}"] for r in hi])) if hi else None
    B = [r for r in rows if r["seed"] in failed or r["seed"] in ctrl]
    risk = np.array([r["risk_pol"] for r in B]); d = np.array([dmap[(r["seed"], r["k"])] for r in B]); tr = np.array([r["triggered"] for r in B])
    out["B_moments"] = len(B)
    out["B_auroc_d_vs_risk>=0.25"] = auroc(d, risk >= 0.25) if (risk >= 0.25).any() else None
    out["B_risky_moments (>= 0.25)"] = int((risk >= 0.25).sum())
    out["B_recall_trigger_on_risky"] = float(tr[risk >= 0.25].mean()) if (risk >= 0.25).any() else None
    out["B_precision_trigger (risky among triggered)"] = float((risk[tr] >= 0.25).mean()) if tr.any() else None
    # the moment in each failed episode where the risk jumps: first moment with risk_pol >= 0.5
    jump = []
    for s in failed:
        rs = sorted([r for r in B if r["seed"] == s], key=lambda r: r["k"])
        first = next((r for r in rs if r["risk_pol"] >= 0.5), None)
        jump.append(None if first is None else first["triggered"])
    out["B_failed_eps_first_risk>=0.5_triggered"] = {"yes": jump.count(True), "no": jump.count(False), "never_>=0.5": jump.count(None)}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "report.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    st = sys.argv[1]
    {"prep": stage_prep, "eval": lambda: stage_eval(*sys.argv[2:3]), "report": stage_report}[st]()

"""O-CEM (docs/FILTER_DEEP_DIVE.md "O-CEM", pre-registered): the trigger stays the fork test; the FIX optimises the
policy's chunk to minimise the imagined DISTURBANCE (continuous barrier) instead of picking the cheapest fork-free chunk.

Why (deep dive 1g): within a state the fork readout h is a poor selector (choosing by h first among {policy, edit, hold}:
5.96% real risk vs policy 5.06%), while the continuous outside-hand disturbance o is a good one (4.37%, oracle 2.78%).
Formulation: theta in formulation G's box (speed, pause, amplitude, back-off; edits of the policy's OWN chunk; stopping
is inside the family, so no hold fallback is needed):
  min_theta  O(theta) + LAMBDA * cost(theta),  O = imagined disturbance of chunk + hold tail (cbf.score o at scale 1),
  search with model_OIL (32 futures, common draws), final choice among the 4 finals + the unmodified chunk by the
  PESSIMISTIC O over the certifier ensemble (OIL + ENS0-3, 64 fresh futures each, max over models).
usage: OPT2_CERT=OIL+ENS python3 eval/opt4.py run <set> <tag> [lambda] ; truth <set> <tag> ; report <set> <tag> <lambdas>
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
import opt3  # noqa: E402
import opt2  # noqa: E402
import bench2 as b2  # noqa: E402

LAMBDAS = (0.0, 0.001, 0.003, 0.01)


def chunk_plan(q16):
    """The executed chunk + hold tail as a 16-step plan (what the truth runs)."""
    p = np.asarray(q16, np.float32).copy(); p[9:] = p[8]
    return p


def solve(P, seed, lam):
    rng = np.random.default_rng(seed); mu, sd = opt3.IDENTITY.copy(), np.array(opt3.CFG["CEM_SD"]); pool = []
    for it in range(opt3.CFG["CEM_ITERS"]):
        T = np.clip(mu + sd * rng.standard_normal((opt3.CFG["CEM_POP"], 4)), opt3.LO, opt3.HI)
        if it == 0:
            T[0] = opt3.IDENTITY
        state = P.mon.rng.bit_generator.state
        _, o = P.score([chunk_plan(P.plan(th)) for th in T], opt2.CFG["N_SEARCH"])
        P.mon.rng.bit_generator.state = state
        key = o + lam * np.array([P.cost(th) for th in T])
        order = np.argsort(key); el = T[order[:opt3.CFG["CEM_ELITES"]]]
        mu, sd = el.mean(0), np.maximum(el.std(0), 0.02 * (opt3.HI - opt3.LO))
        pool += [(float(key[c]), T[c]) for c in order[:opt3.CFG["CEM_FINAL"]]]
    pool.sort(key=lambda x: x[0]); finals = [opt3.IDENTITY] + [th for _, th in pool[:opt3.CFG["CEM_FINAL"]]]
    plans = [chunk_plan(P.plan(th)) for th in finals]
    O = []
    for m in P.cert_models:                                      # pessimistic over the ensemble, fresh draws
        if m is not None:
            saved, P.mon.model = P.mon.model, m
        _, o = P.score(plans, opt2.CFG["N_CERT"]); O.append(o)
        if m is not None:
            P.mon.model = saved
    O = np.max(np.stack(O), 0); costs = np.array([P.cost(th) for th in finals])
    return finals, O, costs


def run(name, tag, lam=0.003):
    import torch
    import cbf_filter as cbff
    import cem_filter as cf
    import inspector_data as insp
    import step3_world_model as wm
    cbff.CFG["H_SAFE"] = opt2.HS; lam = float(lam)
    C, R = b2.load(name); n = int(os.environ.get("OPT2_LIMIT", len(R)))
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    certs = [None] + [wm.WorldModel.load(ROOT / f"results/program_b/ens_v1/model_ENS{k}.pt") for k in range(4)]
    out, t0 = [], time.time()
    for i in range(n):
        t1 = time.time()
        sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        P = opt3.Problem(mon, cbf, sim, torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda"),
                         np.vstack([np.r_[C["target"][i], C["plan16"][i][0, 3]][None], C["plan16"][i][1:]]), opt2.HS, certs)
        P.cbf_O_MAX = cbff.CFG["O_MAX"]
        finals, O, costs = solve(P, 7000 + i, lam)
        out.append({"i": i, "finals": [[round(float(x), 4) for x in th] for th in finals], "O": O.tolist(), "cost": costs.tolist(),
                    "chunks": [P.plan(th)[1:9].astype(np.float32).tolist() for th in finals], "seconds": round(time.time() - t1, 1)})
        if i % 10 == 0:
            print(f"O-CEM {name} {i + 1}/{n} {time.time() - t0:.0f}s", flush=True)
    (b2.OUT / f"opt4_{name}_{tag}.json").write_text(json.dumps({"lambda_search": lam, "rows": out}))


def choose(r, lam):
    return int(np.argmin(np.array(r["O"]) + lam * np.array(r["cost"])))


def truth(name, tag):
    rows = json.loads((b2.OUT / f"opt4_{name}_{tag}.json").read_text())["rows"]
    b2.truth_chunks(name, [(r["i"], np.asarray(ch, np.float32)) for r in rows for ch in r["chunks"]])


def report(name, tag, lams=None):
    rows = json.loads((b2.OUT / f"opt4_{name}_{tag}.json").read_text())["rows"]; C, R = b2.load(name); n = len(rows)
    tr = lambda chs: np.array(b2.truth_chunks(name, chs))[:, 0]
    pol = tr([(i, C["nominal"][i]) for i in range(n)]); hold = tr([(i, b2.hold_chunk(C, i)) for i in range(n)])
    out = {"set": name, "decisions": n, "policy": float(pol.mean()), "hold": float(hold.mean())}
    g = b2.OUT / (f"opt3_{name}_g_cem.json" if name in ("dev", "holdout") else f"opt3_{name}_abs.json")
    if g.exists():
        G = json.loads(g.read_text())["rows"][:n]; gr = tr([(r["i"], np.asarray(r["chunk"], np.float32)) for r in G]); out["g_cem"] = float(gr.mean())
    rng = np.random.default_rng(20261070)
    for lam in (lams or LAMBDAS):
        pick = tr([(r["i"], np.asarray(r["chunks"][choose(r, lam)], np.float32)) for r in rows])
        res = {"risk": float(pick.mean()), "unchanged_share": float(np.mean([choose(r, lam) == 0 for r in rows]))}
        for ref, v in (("policy", pol), ("hold", hold)) + ((("g_cem", gr),) if g.exists() else ()):
            d = pick - v; bs = np.array([d[rng.integers(n, size=n)].mean() for _ in range(10000)])
            res[f"minus_{ref}"] = {"mean": float(d.mean()), "upper95": float(np.percentile(bs, 95)), "p_less": float((bs >= 0).mean()),
                                   "better": int((d < 0).sum()), "worse": int((d > 0).sum())}
        out[f"lambda_{lam}"] = res
    (b2.OUT / f"report_opt4_{name}_{tag}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "run":
        run(*sys.argv[2:5])
    elif st == "truth":
        truth(*sys.argv[2:4])
    else:
        report(sys.argv[2], sys.argv[3], [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None)

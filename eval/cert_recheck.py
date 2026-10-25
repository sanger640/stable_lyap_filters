"""Exploratory (not a gate): is psel's certificate a noisy pass at the boundary (winner's curse), and what does a fix cost on
a full GPU? In paired_cbfps3m_psel, neighbour topples in the fix's own chunk follow 0.80% of certified fixes with ONE certified
sample vs 0.08% with 8-16, and 1.9% of fallback_best fixes (small counts). Cases: intervened decisions stratified by the
logged n_certified (1 / 2-3 / 4-7 / 8-16) plus fallback_best, 20 each, seed 20261025. From the exact saved state (as
eval/grip_bench.py): 16 fresh policy samples, the deployed rule with certificate futures A (timed), then the same 16 samples
+ hold certified again on two fresh future sets B1, B2. Reported per stratum: P(the pick passes fresh), mean fresh pass of
the other certified samples, fresh pass of samples A rejected, and seconds per fix on this GPU.
usage: python3 eval/cert_recheck.py run | analyse  -> results/jenga/deep_dive/cert_recheck.jsonl / cert_recheck.json
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import grip_bench as gb  # noqa: E402

OUT = ROOT / "results/jenga/deep_dive/cert_recheck.jsonl"
SEED = 20261025
BINS = (("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8-16", 8, 16))


def cases():
    pool = {b[0]: [] for b in BINS}; pool["fallback_best"] = []
    for f in sorted(gb.D.glob("ep*.npz")):
        e = int(f.stem[2:]); L = {c["chunk"]: c for c in json.loads((gb.D / f"ep{e}.json").read_text())["log"]}
        with np.load(f) as S:
            iv, ci = S["intervened"].astype(bool), S["chunk_index"]
        for d in np.flatnonzero(iv):
            c = L.get(int(ci[d]))
            if d < 1 or c is None or not c.get("intervened"):
                continue
            if c["fix_kind"] == "fallback_best":
                pool["fallback_best"].append((e, int(d)))
            elif c["fix_kind"] == "certified":
                pool[next(b for b, lo, hi in BINS if lo <= c["n_certified"] <= hi)].append((e, int(d)))
    rng = np.random.default_rng(SEED)
    return [(s, *pool[s][i]) for s in pool for i in rng.choice(len(pool[s]), min(len(pool[s]), 20), replace=False)]


def select(P, p16, samples, hold16):                 # gb.select (the deployed rule), also returning the certificate vector
    import opt2
    ok, hc, oc = P.certify(list(samples) + [hold16])
    costs = np.array([opt2.Problem.cost(P, c[1:9, :3] - p16[1:9, :3]) for c in samples]); okc = ok[:-1]
    if okc.any():
        j = int(np.flatnonzero(okc)[np.argmin(costs[okc])]); return samples[j], "certified", j, int(okc.sum()), ok
    if ok[-1]:
        return hold16, "fallback_hold", -1, 0, ok
    j = int(np.lexsort((oc[:-1], -hc[:-1]))[0]); return samples[j], "fallback_best", j, 0, ok


def recheck(sim, S, d, hist, pol, mon, cbf, certs, e):
    import torch
    import opt3
    import policy_rollout as pr
    import cbf_filter as cbff
    mon.rng = np.random.default_rng([SEED, e, d])
    z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
    p16 = np.asarray(S["plan16"][d], np.float32).copy(); p16[0, :3] = sim.target
    hold16 = np.repeat(np.r_[sim.target, p16[0, 3]][None], 16, 0).astype(np.float32)
    h, o = cbf.score(sim, z0, np.stack([p16, hold16]), 64); h_now = float(h[1])
    P = opt3.Problem(mon, cbf, sim, z0, p16, h_now, certs); P.cbf_O_MAX = cbff.CFG["O_MAX"]
    torch.manual_seed(int(SEED + 7 + 1000 * e + d))
    samples = np.asarray(pr.policy_chunk(pol, hist, n=16), np.float32)
    mon.rng = np.random.default_rng([SEED, e, d, 1])
    torch.cuda.synchronize(); t0 = time.time()
    _, kind, j, nc, okA = select(P, p16, samples, hold16)
    torch.cuda.synchronize(); sec = time.time() - t0
    fresh = []
    for r in (2, 3):
        mon.rng = np.random.default_rng([SEED, e, d, r]); fresh.append(P.certify(list(samples) + [hold16])[0].tolist())
    return {"kind": kind, "choice": j, "n_cert": nc, "okA": okA.tolist(), "fresh": fresh, "seconds_fix": round(sec, 2)}


def run():
    import plausibility as pl
    import fix_counterfactual as fc
    import policy_rollout as pr
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    done = {(r["ep"], r["d"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    mon, cbf, certs = gb.setup()
    pol = pr.load_policy()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="cr_")))); cams = pr.Cameras(sim)
    for ci, (stratum, e, d) in enumerate(cases()):
        if (e, d) in done:
            continue
        with np.load(gb.D / f"ep{e}.npz") as f:
            S = {k: f[k] for k in f.files}
        noises = pl.episode_noises(e, gb.LAST + 2, scale=fc.NOISE)
        sim.restore((S["state"][d - 1], S["ctrl"][d - 1], S["target"][d - 1], float(S["gripper"][d - 1])))
        hist = []
        for j, a in enumerate(S["executed"][d - 1]):
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[int(S["chunk_index"][d - 1])][j]; sim.execute(a)
            if j >= 6:
                hist.append(cams.obs())
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        row = {"stratum": stratum, "ep": e, "d": d, **recheck(sim, S, d, hist, pol, mon, cbf, certs, e)}
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(ci, stratum, e, d, row["kind"], row["n_cert"], row["seconds_fix"], flush=True)
    cams.close(); sim.close(); os._exit(0)


def analyse():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    out = {}
    for s in [b[0] for b in BINS] + ["fallback_best"]:
        R = [r for r in rows if r["stratum"] == s]
        if not R:
            continue
        pick, other, rej, ncert = [], [], [], []
        for r in R:
            okA, F = np.array(r["okA"][:16]), np.array([f[:16] for f in r["fresh"]], float)
            ncert.append(int(okA.sum()))
            if r["kind"] == "certified":
                pick.append(F[:, r["choice"]].mean())
                oth = [i for i in np.flatnonzero(okA) if i != r["choice"]]
                if oth:
                    other.append(F[:, oth].mean())
            if (~okA).any():
                rej.append(F[:, ~okA].mean())
        out[s] = {"cases": len(R), "recomputed_certified_share": round(float(np.mean([r["kind"] == "certified" for r in R])), 3),
                  "recomputed_n_cert_median": float(np.median(ncert)),
                  "pick_fresh_pass": round(float(np.mean(pick)), 3) if pick else None,
                  "other_certified_fresh_pass": round(float(np.mean(other)), 3) if other else None,
                  "rejected_fresh_pass": round(float(np.mean(rej)), 3) if rej else None,
                  "seconds_fix_median": float(np.median([r["seconds_fix"] for r in R]))}
    out["gpu"] = "RTX 3090 (local), Fir runs use MIG 1g.10gb"
    (ROOT / "results/jenga/deep_dive/cert_recheck.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    run() if sys.argv[1:] == ["run"] else analyse()

"""Offline gripper bench (results/jenga/gripbench_protocol.json): should the psel fix keep the policy's GRIPPER command?
psel picks among 16 of the policy's own samples, gripper column included; G-CEM keeps the plan's gripper (OPT3_GRIP=fixed).
In paired_cbfps3m_psel ~21% of fixes made while the red block is held pick a sample that OPENS the gripper where the plan
keeps it closed (eval/psel_grip_check log). Cases: every such decision (intervened, gripper closed at the decision, the
executed fix has gripper < 0 at a step where the policy's chunk has > 0). From the exact saved state and 2-frame history
(as eval/fix_bench.py), first chunk per branch, then the policy alone to the episode's end (chunk 59):
  nominal  the policy's own chunk
  logged   the fix as executed in the run
  psel     recomputed: 16 fresh policy samples, certified (OIL+ENS, 64 futures), closest certified (the deployed rule)
  pselg    the SAME 16 samples with the plan's gripper column, same certificate and rule (common random numbers)
Outcome per branch, to chunk 59: neighbour topple, red fall (tilt >= 45), extracted (ever, from the episode's rest pose),
v4b-like success = extracted and no red fall and no topple.
usage: SHARD=i/n python3 eval/grip_bench.py run -> results/jenga/deep_dive/gripbench/rows_<i>.jsonl
       python3 eval/grip_bench.py analyse      -> results/jenga/gripbench_result.json
"""
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import fix_counterfactual as fc  # noqa: E402

RUN = "paired_cbfps3m_psel"
D = ROOT / "results/program_b/policy_rollouts" / RUN
OUT = ROOT / "results/jenga/deep_dive/gripbench"
SEED, LAST = 20261026, 59


def cases():
    out = []
    for f in sorted(D.glob("ep*.npz")):
        with np.load(f) as S:
            iv, g, nom, exe = S["intervened"].astype(bool), S["gripper"], S["nominal"][:, :, 3], S["executed"][:, :, 3]
            rel = ((nom > 0) & (exe < 0)).any(1) & iv & (g < 55)
            out += [(int(f.stem[2:]), int(d)) for d in np.flatnonzero(rel) if d >= 1]
    return out


def setup():
    os.environ["CBF_WM"] = str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); os.environ["OPT2_CERT"] = "OIL+ENS"
    os.environ["OPT3_GRIP"] = "fixed"
    import cbf_barrier
    import cbf_filter as cbff
    cbf_barrier.SCALES = (0.5, 1.0, 2.0, 3.0, 4.0)
    cbff.CALIB = ROOT / "results/jenga/cbf_calibration_OIL.json"
    cbff.CFG["H_SAFE"] = float(np.log2(4.0)); cbff.CFG["FIX"] = "psel"
    import cem_filter as cf
    import step3_world_model as wm
    mon = cf.Monitor(); cbf = cbff.CBF(mon)
    certs = [None] + [wm.WorldModel.load(ROOT / f"results/program_b/ens_v1/model_ENS{e}.pt") for e in range(4)]
    return mon, cbf, certs


def select(P, p16, samples, hold16):
    import opt2
    ok, hc, oc = P.certify(list(samples) + [hold16])
    costs = np.array([opt2.Problem.cost(P, c[1:9, :3] - p16[1:9, :3]) for c in samples]); okc = ok[:-1]
    if okc.any():
        j = int(np.flatnonzero(okc)[np.argmin(costs[okc])]); return samples[j], "certified", j, int(okc.sum())
    if ok[-1]:
        return hold16, "fallback_hold", -1, 0
    j = int(np.lexsort((oc[:-1], -hc[:-1]))[0]); return samples[j], "fallback_best", j, 0


def first_chunks(sim, S, d, hist, pol, mon, cbf, certs, e):
    import torch
    import cbf_filter as cbff
    import opt3
    import policy_rollout as pr
    PS = pr.PLAN_START
    mon.rng = np.random.default_rng([SEED, e, d])
    z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
    p16 = np.asarray(S["plan16"][d], np.float32).copy(); p16[0, :3] = sim.target
    hold16 = np.repeat(np.r_[sim.target, p16[0, 3]][None], 16, 0).astype(np.float32)
    h, o = cbf.score(sim, z0, np.stack([p16, hold16]), 64); h_now = float(h[1])
    P = opt3.Problem(mon, cbf, sim, z0, p16, h_now, certs); P.cbf_O_MAX = cbff.CFG["O_MAX"]
    torch.manual_seed(int(SEED + 7 + 1000 * e + d))
    samples = np.asarray(pr.policy_chunk(pol, hist, n=16), np.float32)
    samples_g = samples.copy(); samples_g[:, :, 3] = p16[:, 3]
    out = {"nominal": S["nominal"][d], "logged": S["executed"][d]}; info = {"h_nom_re": float(h[0]), "h_now_re": h_now}
    for name, sm in (("psel", samples), ("pselg", samples_g)):
        mon.rng = np.random.default_rng([SEED, e, d, 1])                  # same certificate futures for both
        best, kind, j, nc = select(P, p16, sm, hold16)
        out[name] = best[PS:PS + 8].copy()
        info.update({f"{name}_kind": kind, f"{name}_choice": j, f"{name}_n_certified": nc,
                     f"{name}_opens": bool(((p16[PS:PS + 8, 3] > 0) & (best[PS:PS + 8, 3] < 0)).any())})
    return out, info


def branch(sim, cams, pol, S, d, e, hist0, first, noises, xy0, z_rest):
    import torch
    import policy_rollout as pr
    sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
    hist = list(hist0); k0 = int(S["chunk_index"][d])
    res = {"extract": None, "red_fall": None, "topple": None, "held_end": None}
    torch.manual_seed(int(SEED + 1000 * e + d))
    for k in range(k0, LAST + 1):
        chunk = first if k == k0 else pr.policy_chunk(pol, hist, full=True)[pr.PLAN_START:pr.PLAN_START + 8]
        for j, a in enumerate(chunk):
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[k][j]; sim.execute(a); hist.append(cams.obs())
            p = sim.data.xpos[sim.tracked_block_ids[0]]; t = sim.block_diagnostics()["tilt"]
            if res["extract"] is None and (p[2] - z_rest >= 0.02 or p[0] - xy0[0] <= -0.02):
                res["extract"] = k
            if res["red_fall"] is None and float(t[0]) >= 45:
                res["red_fall"] = k
            if res["topple"] is None and float(np.max(t[1:])) >= 45:
                res["topple"] = k
        hist = hist[-2:]
        if res["topple"] is not None or res["red_fall"] is not None:
            break
    res["held_end"] = bool(sim.gripper < 55)
    return res


def run():
    import plausibility as pl
    import policy_rollout as pr
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    shard, nshard = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
    OUT.mkdir(parents=True, exist_ok=True); fout = OUT / f"rows_{shard}.jsonl"
    done = {(r["ep"], r["d"]) for r in map(json.loads, fout.read_text().splitlines())} if fout.exists() else set()
    aud = {r["ep"]: r for r in json.loads((ROOT / "results/jenga/outcome_audit_psel.json").read_text())[RUN]["rows"]}
    mon, cbf, certs = setup()
    pol = pr.load_policy()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="gb_")))); cams = pr.Cameras(sim)
    left = int(os.environ.get("LIMIT", 10 ** 9))
    for ci, (e, d) in enumerate(cases()):
        if ci % nshard != shard or (e, d) in done:
            continue
        if left <= 0:
            break
        left -= 1
        with np.load(D / f"ep{e}.npz") as f:
            S = {k: f[k] for k in f.files}
        k0 = int(S["chunk_index"][d])
        noises = pl.episode_noises(e, LAST + 2, scale=fc.NOISE)
        sim.restore((S["state"][0], S["ctrl"][0], S["target"][0], float(S["gripper"][0])))
        xy0 = sim.data.xpos[sim.tracked_block_ids[0]][:2].copy(); z_rest = float(sim.data.xpos[sim.tracked_block_ids[0]][2])
        sim.restore((S["state"][d - 1], S["ctrl"][d - 1], S["target"][d - 1], float(S["gripper"][d - 1])))
        hist = []
        for j, a in enumerate(S["executed"][d - 1]):
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[int(S["chunk_index"][d - 1])][j]; sim.execute(a)
            if j >= 6:
                hist.append(cams.obs())
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        firsts, info = first_chunks(sim, S, d, hist, pol, mon, cbf, certs, e)
        x = aud[e]
        row = {"ep": e, "d": d, "chunk": k0, "episode_outcome": "topple" if x["logged_neighbour_toppled"] else
               "red_before" if x["red_fell_before_extraction"] else "timeout" if not x["extracted"] else
               "red_after" if x["red_fell_after_extraction"] else "ok", **info}
        for name, first in firsts.items():
            row[name] = branch(sim, cams, pol, S, d, e, hist, first, noises, xy0, z_rest)
        with open(fout, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(ci, row, flush=True)
    cams.close(); sim.close(); os._exit(0)


def ok(r):                                          # extracted (detected at any branch step, from the episode's rest pose,
    return r["extract"] is not None and r["red_fall"] is None and r["topple"] is None   # so a block already out counts at once)


def analyse():
    rows = [json.loads(l) for f in sorted(OUT.glob("rows_*.jsonl")) for l in f.read_text().splitlines()]
    rng = np.random.default_rng(20261026)
    B = ("nominal", "logged", "psel", "pselg")
    succ = {b: np.array([ok(r[b]) for r in rows], float) for b in B}
    top = {b: np.array([r[b]["topple"] is not None for r in rows], float) for b in B}
    red = {b: np.array([r[b]["red_fall"] is not None for r in rows], float) for b in B}

    def bound(x, q):
        return round(100 * float(np.percentile([x[rng.integers(len(x), size=len(x))].mean() for _ in range(10000)], q)), 2)
    ds, dt = succ["pselg"] - succ["psel"], top["pselg"] - top["psel"]
    out = {"cases": len(rows), "rates_pct": {b: {"success": round(100 * succ[b].mean(), 1), "topple": round(100 * top[b].mean(), 1),
                                                 "red_fall": round(100 * red[b].mean(), 1)} for b in B},
           "psel_opens": int(sum(r["psel_opens"] for r in rows)), "pselg_opens": int(sum(r["pselg_opens"] for r in rows)),
           "identical_choice": int(sum(r["psel_choice"] == r["pselg_choice"] and r["psel_kind"] == r["pselg_kind"] for r in rows)),
           "PRIMARY_success_pselg_minus_psel": {"diff_pp": round(100 * ds.mean(), 2), "lower95_one_sided_pp": bound(ds, 5)},
           "PRIMARY_topple_pselg_minus_psel": {"diff_pp": round(100 * dt.mean(), 2), "upper95_one_sided_pp": bound(dt, 95)},
           "by_episode_outcome": {k: sum(r["episode_outcome"] == k for r in rows) for k in ("ok", "timeout", "red_after", "topple", "red_before")}}
    out["PRIMARY_pass"] = bool(out["PRIMARY_success_pselg_minus_psel"]["lower95_one_sided_pp"] > 0 and out["PRIMARY_topple_pselg_minus_psel"]["upper95_one_sided_pp"] <= 3.0)
    (ROOT / "results/jenga/gripbench_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    run() if sys.argv[1:] == ["run"] else analyse()

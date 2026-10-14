"""Final test on fresh panels 6 and 7 with corrected ground truth (results/jenga/phase1e_protocol.json,
final_test_panels_v2_2026_10_14).

  truth  CPU pool: per panel state, at-rest test + truth_A / truth_B (64 futures each, original seeds) with the
         corrected perturbation -> results/program_b/panel{6,7}_v2/<state>.npz
  grade  GPU: monitors vs labels -> results/jenga/panels_v2_result.json
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

PANELS = {
    "panel6": {"dir": ROOT / "results/program_b/panel6", "RESET_BASE": 10000,
               "SEEDS": {"monitor_d": 20261092, "truth_A": 20261094, "truth_B": 20261095}},
    "panel7": {"dir": ROOT / "results/program_b/panel7", "RESET_BASE": 11000,
               "SEEDS": {"monitor_d": 20261112, "truth_A": 20261114, "truth_B": 20261115}},
}
V3 = ROOT / "results/program_b/threeblock_v3"
RESULT = ROOT / "results/jenga/panels_v2_result.json"


def _check_seeds():
    """Panel-7 seeds are read from its generator so they cannot drift."""
    import wm_readout_fix_panel7 as w7
    PANELS["panel7"]["SEEDS"] = {k: w7.CONST["SEEDS"][k] for k in ("monitor_d", "truth_A", "truth_B")}
    PANELS["panel7"]["RESET_BASE"] = w7.CONST["RESET_BASE"]


def truth_job(job):
    import traceback
    try:
        return job, _truth(job), None
    except Exception:
        return job, None, traceback.format_exc()


def _truth(job):
    panel, name = job
    _check_seeds()
    import program_b_reference as pr
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    cfg = PANELS[panel]
    out = ROOT / f"results/program_b/{panel}_v2" / f"{name}.npz"
    if out.exists():
        return True
    assert pr.VALID_PERTURBATION
    e, i = name[2:].split("_s"); e, i = int(e), int(i)
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB); actions = np.asarray(replay.episode(e).actions, np.float32); replay.close()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="pv2_"))))
    try:
        sim.reset(e + cfg["RESET_BASE"])
        for k, a in enumerate(actions):
            if k == i:
                break
            sim.execute(a)
        with np.load(cfg["dir"] / f"{name}.npz") as f:
            assert np.allclose(f["chunk"], actions[i:i + 8])
        rest = pr.blocks_at_rest(sim)
        snap = sim.snapshot(); tilt = sim.block_diagnostics()["tilt"][1:].copy()
        key = (e + cfg["RESET_BASE"], i); chunk = actions[i:i + 8]
        tr = {nm: np.stack([um.rollout(sim, snap, chunk, snippets[j], [errors.sample(cfg["SEEDS"][nm], *key, j)])
                            for j in range(64)]).astype(np.float32) for nm in ("truth_A", "truth_B")}
    finally:
        sim.close()
    np.savez_compressed(out.with_suffix(".tmp.npz"), at_rest=rest, start_tilt=tilt, **tr)
    out.with_suffix(".tmp.npz").replace(out)
    return True


def stage_truth(workers=int(os.environ.get("PV_WORKERS", 16))):
    jobs = []
    for panel, cfg in PANELS.items():
        (ROOT / f"results/program_b/{panel}_v2").mkdir(parents=True, exist_ok=True)
        jobs += [(panel, p.stem) for p in sorted(cfg["dir"].glob("ep*.npz")) if not p.name.endswith(".tmp.npz")]
    fails = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (job, ok, err) in enumerate(pool.imap_unordered(truth_job, jobs), 1):
            if err:
                fails.append(job); print(err, flush=True)
            if n % 20 == 0 or n == len(jobs):
                print(f"truth {n}/{len(jobs)}", flush=True)
    if fails:
        raise SystemExit(f"failures: {fails}")


def stage_grade():
    _check_seeds()
    import torch
    import phase1d_models as p1d  # noqa: F401
    import phase2_hybrid as hy
    import step3_world_model as wm
    import step3b_models as sm
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from fresh_fragility_panel import topple_count
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings
    from phase2_amend4 import alarm
    from regrade_any_outcome import events_three_block_v2, labels
    from step3_data import Latent
    res1e = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())
    v3 = json.loads((ROOT / "results/jenga/threeblock_v3_result.json").read_text())["floors"]
    old_us = wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_US.pt"); old_tau = res1e["US"]["tau"]
    old_g = sm.G.load(ROOT / "results/program_b/step3b/G.pt")
    us3 = wm.WorldModel.load(V3 / "model_US.pt"); us3.net.eval(); g3 = sm.G.load(V3 / "G.pt")
    s = torch.load(V3 / "model_HYB_T1.pt", map_location="cuda")
    head = hy.build_head(s["action_dim"]).cuda(); head.load_state_dict(s["state_dict"]); head.eval()
    latent, errors, snippets = Latent(), p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    rows = []
    with torch.no_grad():
        for panel, cfg in PANELS.items():
            for n, path in enumerate(sorted((ROOT / f"results/program_b/{panel}_v2").glob("ep*.npz"))):
                if path.name.endswith(".tmp.npz"):
                    continue
                name = path.stem; e, i = name[2:].split("_s"); key = (int(e) + cfg["RESET_BASE"], int(i))
                with np.load(path) as t, np.load(cfg["dir"] / f"{name}.npz") as f:
                    trA, trB, rest, tilt = t["truth_A"], t["truth_B"], bool(t["at_rest"]), t["start_tilt"]
                    exact, frame, chunk = f["exact"], f["close_frame"], f["chunk"]
                fA, sA = labels(events_three_block_v2(trA)); kA = topple_count(trA, tilt)
                fB, _ = labels(events_three_block_v2(trB)); fE, _ = labels(events_three_block_v2(exact))
                z0 = torch.as_tensor(latent(frame[None])[0], device="cuda").float()
                draws = [errors.sample(cfg["SEEDS"]["monitor_d"], *key, j) for j in range(64)]
                d_raw = torch.tensor(np.stack([np.concatenate([p.reshape(9), r.reshape(9)]) for p, r in draws]), dtype=torch.float32)
                d_pl = torch.tensor(np.stack([np.concatenate([(p * [1, 1, 0]).reshape(9), (r * [0, 0, 1]).reshape(9)]) for p, r in draws]), dtype=torch.float32)
                acts = wm.panel_actions(chunk, snippets)
                z64 = z0[None].expand(64, -1, -1, -1).contiguous()
                o10, o30 = sm.roll(old_us, old_g(z64, d_raw), acts)
                u10, u30 = sm.roll(us3, g3(z64, d_pl), acts)
                gen = torch.Generator(device="cuda").manual_seed(hy.MONITOR_SEED + 50_000 + len(rows))
                fut = hy.sample(head, us3, z64, us3._act(torch.as_tensor(acts, device="cuda")), s["residual_scale"], gen)
                h10 = fut[:, 0].reshape(64, -1).double().cpu().numpy(); h30 = fut[:, 1].reshape(64, -1).double().cpu().numpy()
                rows.append({"panel": panel, "state": name, "at_rest": rest, "fork": fA, "safe": sA,
                             "harm": 2 <= kA <= 62, "harm_safe": kA == 0,
                             "alarms": {"v3_hybrid_H30_floor": bool(alarm(h10, h30, "H30_floor", v3["HYB_new"])),
                                        "v3_hybrid_BOTH": bool(alarm(h10, h30, "BOTH")),
                                        "v3_plain_BOTH": bool(alarm(u10, u30, "BOTH")),
                                        "old_plain_H30_floor": bool(alarm(o10, o30, "H30_floor", old_tau)),
                                        "mujoco_probe_exact_any": bool(fE),
                                        "mujoco_probe_exact_harm": bool(decide(*trace_endings(exact, NON_ROBOT["jenga"]))[0]),
                                        "oracle_truth_B": bool(fB)}})
                torch.cuda.empty_cache()

    def wilson(k, n):
        if n == 0:
            return [None, None]
        p, z = k / n, 1.96
        c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return [round(c - h, 3), round(c + h, 3)]
    out = {"excluded_not_at_rest": sum(not r["at_rest"] for r in rows)}
    for lab, pk, nk in (("any_outcome_gripper_relative", "fork", "safe"), ("harm", "harm", "harm_safe")):
        out[lab] = {}
        for scope in ("panel6", "panel7", "pooled"):
            sub = [r for r in rows if r["at_rest"] and (r[pk] or r[nk]) and (scope == "pooled" or r["panel"] == scope)]
            y = np.array([r[pk] for r in sub]); res = {"forks": int(y.sum()), "safe": int((~y).sum())}
            for m in sub[0]["alarms"]:
                a = np.array([r["alarms"][m] for r in sub]); kc, kf = int((a & y).sum()), int((a & ~y).sum())
                res[m] = {"caught": round(kc / max(y.sum(), 1), 3), "false_alarm": round(kf / max((~y).sum(), 1), 3),
                          "score": round(kc / max(y.sum(), 1) - kf / max((~y).sum(), 1), 3),
                          "caught_ci": wilson(kc, int(y.sum())), "false_alarm_ci": wilson(kf, int((~y).sum()))}
            out[lab][scope] = res
    out["rows"] = rows
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)


if __name__ == "__main__":
    {"truth": stage_truth, "grade": stage_grade}[sys.argv[1]]()

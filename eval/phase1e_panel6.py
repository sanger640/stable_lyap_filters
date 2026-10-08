"""Phase 1e addendum 2: fresh panel 6 (the gate's confirmation panel) for US, with cell E and R_fast beside.

Generation reuses eval/step3b_panel5.py unchanged with panel-6 constants (same patching as eval/wm_readout_fix_panel7.py);
grading as eval/phase1e_panel7.py plus the exact-start MuJoCo probe and the second oracle run.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
PROTOCOL = ROOT / "results/jenga/phase1e_protocol.json"
OUT = ROOT / "results/program_b/panel6"
RESULT = ROOT / "results/jenga/phase1e_panel6_result.json"
CONST = {"RESET_BASE": 10000, "EPISODES": 80, "SELECT_SEED": 20261090,
         "SEEDS": {"monitor_d": 20261092, "truth_A": 20261094, "truth_B": 20261095, "hat": 20261096}}


def _patch():
    import step3b_panel5 as p5
    for k, v in CONST.items():
        setattr(p5, k, v)
    p5.OUT = OUT
    return p5


def worker(job):
    return _patch().worker(job)


def init():
    _patch()._set_base()


def generate(workers):
    p5 = _patch()
    OUT.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    import fresh_fragility_panel as fp
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    screen_file = OUT / "screen.json"
    if not screen_file.exists():
        replay = JengaReplay(DEFAULT_LMDB); ids = list(replay.episode_ids); replay.close()
        episodes = sorted(np.random.default_rng(CONST["SELECT_SEED"]).choice(ids, CONST["EPISODES"], replace=False).tolist(),
                          key=int)
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=init) as pool:
            screen = {e: s for e, s in pool.map(fp.chain_episode, [(e, set(), str(OUT)) for e in episodes])}
        screen_file.write_text(json.dumps(screen) + "\n")
    panel, screen, screened = p5.screen_panel(workers)
    print(f"screened {screened} states; panel {len(panel)}", flush=True)
    by_episode = {}
    for e, i in panel:
        by_episode.setdefault(e, set()).add(i)
    todo = [(e, idx) for e, idx in sorted(by_episode.items(), key=lambda x: int(x[0]))
            if not all((OUT / f"ep{e}_s{i}.npz").exists() for i in idx)]
    failures = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for number, (e, error) in enumerate(pool.imap_unordered(worker, todo), 1):
            print(f"{number}/{len(todo)} ep{e}" + (" FAILED" if error else ""), flush=True)
            if error:
                failures.append(e); print(error, flush=True)
    if failures:
        raise SystemExit(f"failed episodes: {failures}")
    return panel, screen, screened


def grade(panel, screen, screened):
    import torch
    import phase1_models as p1
    import step3_world_model as wm
    import step3b_models as sm
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from counterfactual_monitor import persistent_mode_alarm
    from fresh_fragility_panel import topple_count
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings
    from outcome_modes import two_mode_test
    from step3_data import Latent
    res = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())
    models = {"US": (wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_US.pt"), res["US"]["tau"]),
              "R_fast": (wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_R_fast.pt"), res["R_fast"]["tau"]),
              "E": (wm.WorldModel.load(p1.OUT / "model_E.pt"), res["E_reference"]["tau"])}
    snippets = np.load(um.BENCH)["snippets"]
    latent, errors, g = Latent(), p3.ErrorModel(), sm.G.load(sm.DATA / "G.pt")
    rows = []
    for e, i in sorted(panel, key=lambda k: (int(k[0]), k[1])):
        key = (int(e) + CONST["RESET_BASE"], i)
        tilt = np.array(screen[e][str(i)]["start_tilt"])
        with np.load(OUT / f"ep{e}_s{i}.npz") as f:
            k = topple_count(f["truth_A"], tilt)
            z0 = torch.as_tensor(latent(f["close_frame"][None])[0], device=sm.DEVICE).float()
            d = torch.tensor(np.stack([np.concatenate([p.reshape(9), r.reshape(9)]) for p, r in
                                       (errors.sample(CONST["SEEDS"]["monitor_d"], *key, j) for j in range(64))]),
                             dtype=torch.float32)
            acts = wm.panel_actions(f["chunk"], snippets)
            row = {"state": f"{e}:{i}", "fork": bool(2 <= k <= 62), "lean": bool(tilt.max() >= 10.0),
                   "exact": bool(decide(*trace_endings(f["exact"], NON_ROBOT["jenga"]))[0]),
                   "truth_B": bool(decide(*trace_endings(f["truth_B"], NON_ROBOT["jenga"]))[0])}
            for name, (model, tau) in models.items():
                e10, e30 = sm.roll(model, g(z0[None].expand(64, -1, -1, -1), d), acts)
                r0 = bool(persistent_mode_alarm(e10, e30).alarm)
                row[f"{name}_R0"] = r0
                row[f"{name}_R1"] = bool(r0 and two_mode_test(e30, 0.0).separation_rms_mm > tau)
            rows.append(row)
        torch.cuda.empty_cache()
    fork = np.array([r["fork"] for r in rows]); lean = np.array([r["lean"] for r in rows])
    out = {"protocol": "phase1e addendum 2", "screened": screened, "states": len(rows), "forks": int(fork.sum()),
           "lean_forks": int((fork & lean).sum()), "contact_forks": int((fork & ~lean).sum())}
    for col in ("US_R0", "US_R1", "R_fast_R0", "R_fast_R1", "E_R0", "E_R1", "exact", "truth_B"):
        a = np.array([r[col] for r in rows])
        out[col] = {"caught": round(float((a & fork).sum() / fork.sum()), 4),
                    "false_alarm": round(float((a & ~fork).sum() / (~fork).sum()), 4),
                    "forks_caught": int((a & fork).sum()), "false_alarms": int((a & ~fork).sum()),
                    "lean": f"{int((a & fork & lean).sum())}/{int((fork & lean).sum())}",
                    "contact": f"{int((a & fork & ~lean).sum())}/{int((fork & ~lean).sum())}"}
    out["US_passes_panel6_gates"] = bool(out["US_R0"]["caught"] >= 0.80 and out["US_R0"]["false_alarm"] <= 0.20)
    out["rows"] = rows
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("generate", "grade"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if "addendum_2" not in json.loads(PROTOCOL.read_text()):
        raise SystemExit("addendum 2 not recorded")
    if args.stage == "generate":
        generate(args.workers)
    else:
        if "R_fast" not in json.loads((ROOT / "results/jenga/phase1e_result.json").read_text()):
            raise SystemExit("R_fast not scored yet")
        grade(*generate(args.workers))


if __name__ == "__main__":
    main()

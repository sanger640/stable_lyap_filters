"""Fresh panel 7 for the WM readout fix (R1 model-error floor) on cell E.

Frozen protocol: results/jenga/wm_readout_fix_protocol.json (confirmation + addendum 2). Generation reuses
eval/step3b_panel5.py unchanged with panel-7 constants (reset +11000, 80 episodes, selection seed 20261111, new
draw seeds); grading runs cell E with G starts from independent monitor perception draws, R0 vs R1.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
PROTOCOL = ROOT / "results/jenga/wm_readout_fix_protocol.json"
OUT = ROOT / "results/program_b/panel7"
RESULT = ROOT / "results/jenga/wm_readout_fix_panel7_result.json"
CONST = {"RESET_BASE": 11000, "EPISODES": 80, "SELECT_SEED": 20261111,
         "SEEDS": {"monitor_d": 20261112, "truth_A": 20261114, "truth_B": 20261115, "hat": 20261116}}


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
    tau = json.loads((ROOT / "results/jenga/wm_readout_fix_result.json").read_text())["cell_E"]["tau"]
    keys = sorted(panel, key=lambda k: (int(k[0]), k[1]))
    snippets = np.load(um.BENCH)["snippets"]
    latent, errors = Latent(), p3.ErrorModel()
    g = sm.G.load(sm.DATA / "G.pt")
    model = wm.WorldModel.load(p1.OUT / "model_E.pt")
    rows = []
    for e, i in keys:
        key = (int(e) + CONST["RESET_BASE"], i)
        with np.load(OUT / f"ep{e}_s{i}.npz") as f:
            fork = 2 <= topple_count(f["truth_A"], np.array(screen[e][str(i)]["start_tilt"])) <= 62
            z0 = torch.as_tensor(latent(f["close_frame"][None])[0], device=sm.DEVICE).float()
            d = torch.tensor(np.stack([np.concatenate([p.reshape(9), r.reshape(9)]) for p, r in
                                       (errors.sample(CONST["SEEDS"]["monitor_d"], *key, j) for j in range(64))]),
                             dtype=torch.float32)
            e10, e30 = sm.roll(model, g(z0[None].expand(64, -1, -1, -1), d), wm.panel_actions(f["chunk"], snippets))
            r0 = bool(persistent_mode_alarm(e10, e30).alarm)
            sep = float(two_mode_test(e30, 0.0).separation_rms_mm)
            rows.append({"state": f"{e}:{i}", "fork": bool(fork), "R0": r0, "R1": bool(r0 and sep > tau), "separation": sep,
                         "exact": bool(decide(*trace_endings(f["exact"], NON_ROBOT["jenga"]))[0]),
                         "truth_B": bool(decide(*trace_endings(f["truth_B"], NON_ROBOT["jenga"]))[0])})
        torch.cuda.empty_cache()
    fork = np.array([r["fork"] for r in rows])

    def grade_(name):
        a = np.array([r[name] for r in rows])
        return {"caught": float((a & fork).sum() / fork.sum()), "false_alarm": float((a & ~fork).sum() / (~fork).sum()),
                "forks_caught": int((a & fork).sum()), "false_alarms": int((a & ~fork).sum())}
    out = {"protocol": "wm-readout-fix (confirmation, addendum 2)", "screened": screened, "states": len(rows),
           "forks": int(fork.sum()), "tau": tau, **{n: grade_(n) for n in ("R0", "R1", "exact", "truth_B")}}
    out["R1_confirmed"] = bool(out["R1"]["false_alarms"] <= 0.5 * out["R0"]["false_alarms"]
                               and out["R0"]["forks_caught"] - out["R1"]["forks_caught"] <= 2)
    out["rows"] = rows
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("states", "forks", "R0", "R1", "R1_confirmed")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    grade(*generate(args.workers))


if __name__ == "__main__":
    main()

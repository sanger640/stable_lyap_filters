"""T2 (push past tall objects) data, selection set H_T2, model and grading, reusing the Phase-2 code paths.

Frozen protocol: results/jenga/t2_protocol.json.  Stages: train_data | h | model | grade
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import numpy as np  # noqa: E402

import phase2_data as pdata  # noqa: E402
import phase2_models as pm  # noqa: E402

T2DIR = ROOT / "results/program_b/t2"
H_SEEDS = range(500_000, 500_800)


def _patch():
    pdata.OUT = T2DIR; pm.OUT = T2DIR
    pdata.FAMILIES = ("T2",)


def screen_job(job):
    _patch(); return pdata.screen_job(job)


def render_job(job):
    _patch(); return pdata.render_job(job)


def h_screen_job(job):
    _patch(); return pm.h2_screen_job(job)


def h_render_job(job):
    _patch(); return pm.h2_render_job(job)


def train_data(screen_workers=14, render_workers=5):
    _patch()
    for d in ("screen", "train"):
        (T2DIR / d).mkdir(parents=True, exist_ok=True)
    screens = pdata.pool_run(screen_job, [("T2", s) for s in range(3000)], screen_workers, "t2 screen")
    pdata.SEEDS = dict(pdata.SEEDS, select=20261320)
    pdata.PER_FAMILY, pdata.DISAGREE_CAP = 4800, 2400
    roles, summary = pdata.select(screens)
    (T2DIR / "selection.json").write_text(json.dumps({"summary": summary, "roles": {f"{f}:{s}": v for (f, s), v in roles.items()}}) + "\n")
    print(json.dumps(summary), flush=True)
    pdata.pool_run(render_job, [(f, s, r) for (f, s), r in sorted(roles.items())], render_workers, "t2 render")
    print("t2 train data done", flush=True)


def h(screen_workers=14, render_workers=5):
    _patch()
    for d in ("h2_screen", "h2"):
        (T2DIR / d).mkdir(parents=True, exist_ok=True)
    screens = pdata.pool_run(h_screen_job, [("T2", s) for s in H_SEEDS], screen_workers, "h screen")
    rows = [(f, s, int(i), c) for (f, s), r in sorted(screens.items()) if not r["skipped"] for i, c in r["harm_counts"].items()]
    rng = np.random.default_rng(20261321)
    forks = [r for r in rows if 2 <= r[3] <= 62]; safe = [r for r in rows if r[3] == 0]
    forks = [forks[i] for i in sorted(rng.permutation(len(forks))[:80])]
    safe = [safe[i] for i in sorted(rng.permutation(len(safe))[:80])]
    by = {}
    for f, s, i, _ in forks + safe:
        by.setdefault((f, s), set()).add(i)
    summary = {"scenes": len(screens), "chunk_starts": len(rows), "forks_available": sum(2 <= r[3] <= 62 for r in rows),
               "h_forks": len(forks), "h_safe": len(safe)}
    (T2DIR / "h_selection.json").write_text(json.dumps(summary) + "\n")
    print(json.dumps(summary), flush=True)
    pdata.pool_run(h_render_job, [(f, s, idx) for (f, s), idx in sorted(by.items())], render_workers, "h render")
    print("t2 h done", flush=True)


def model():
    _patch()
    import torch
    import phase1e_models as p1e
    import phase2_amend4 as a4
    states, names = pm.load_train()
    pm.fit_sampler(states)                                   # Gauss-TE on T2 deltas -> T2DIR/Gauss_TE_phase2.pt
    if not (T2DIR / "modes.json").exists():
        (T2DIR / "modes.json").write_text(json.dumps(p1e.detect_modes(states, names)) + "\n")
    del states
    a4.train_arm("US2_long")


def grade():
    _patch()
    import phase2_amend4 as a4
    a4.RESULT = ROOT / "results/jenga/t2_h_result.json"
    if not a4.RESULT.exists():
        a4.RESULT.write_text(json.dumps({"A_selected_readout": "BOTH"}) + "\n")   # protocol: BOTH is the primary readout
    a4.grade("T2_H")


if __name__ == "__main__":
    {"train_data": train_data, "h": h, "model": model, "grade": grade}[sys.argv[1]]()

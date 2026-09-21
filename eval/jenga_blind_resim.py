"""Re-simulate the systematically missed forks with dense logging (PLAN_NEXT.md Phase 2, step 5).

The Stage-0 oracle cache keeps block poses at hold steps 5/10/20/29/30 only, so it cannot say WHEN
the real counterfactual trajectories part or what physical event parts them. This re-runs each
selected fork from its exact state through all 64 probes at 1x, recording the full 61-dim state
(pose, velocity, contact flags, gripper) five times per control step -- every 10 of the 50
physics substeps, via `jenga_state_data.execute_recording`, which reproduces
`DirectJengaSim.execute` bit for bit.

Two groups, because a mechanism only explains blindness if it is RARER among forks the model does
catch:

  blind     forks robust-blind under either operating point (q >= 0.8, jenga_blind_freq.py)
  control   usually-detected forks (q < 0.3 under both), each matched to one blind fork on the
            number of probes that topple -- narrow branches were the obvious confound

Reproducibility. `DirectJengaSim.snapshot()` saves mjSTATE_FULLPHYSICS, which excludes the solver
warm start, and a restore leaves kinematics consistent with the restored state where an
uninterrupted replay reads them one substep late. So restore-then-continue is never bit-identical to
not interrupting. Stage 0 restored and continued at every probed chunk start, and its replays
drifted: its start states differ from the clean-replay ones in 182/214 test states (median 0.04 mm,
up to 52 mm, all drifts over 1 mm in quiet states). Here every chunk start is reached by an
UNINTERRUPTED replay and probing never continues a replay, so the dense traces start from exactly
the frozen start state the model is evaluated from.

Each probe is also run under three warm-start conditions -- restored with the snapshot (the dense
traces), zeroed, and carried from the previous probe -- and every probe's topple outcome is compared
with the frozen Stage-0 label. On the 34 forks the toppling set is IDENTICAL across all three
conditions: the branches are physical, not solver artefacts. Residual disagreement with Stage 0
(typically 0-2 of 64 probes) comes from its drifted start states.

The start state must equal the frozen start state exactly, or the run aborts.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

EVERY = 10                     # physics substeps between readings: 5 readings per control step
SCALE = "1.0"


def select(freq_path, controls_per_blind=1):
    rows = json.loads(Path(freq_path).read_text())["scales"][SCALE]["rows"]
    blind = [r for r in rows
             if r["class"] == "robust_blind" or r["class_matched"] == "robust_blind"]
    pool = [r for r in rows if r["class"] == "usually_detected"
            and r["class_matched"] == "usually_detected"]
    chosen, used = [], set()
    for b in sorted(blind, key=lambda r: r["topple_count"]):
        for _ in range(controls_per_blind):
            options = [r for r in pool if (r["episode_id"], r["chunk_start"]) not in used]
            if not options:
                break
            best = min(options, key=lambda r: (abs(r["topple_count"] - b["topple_count"]),
                                               r["q"], r["episode_id"], r["chunk_start"]))
            used.add((best["episode_id"], best["chunk_start"]))
            chosen.append(best)
    tag = lambda r, g: dict(r, group=g)  # noqa: E731
    return [tag(r, "blind") for r in blind] + [tag(r, "control") for r in chosen]


def simulate_episode(job):
    """One episode: walk to each selected chunk start, then run all 64 probes densely."""
    episode_id, forks, xml, windows = job
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim
    from jenga_state_data import execute_recording, step_state
    from state_dynamics import neighbour_tilt_deg

    def toppled(final, start):
        eligible = neighbour_tilt_deg(start[None])[0] < 45.0
        return bool(((neighbour_tilt_deg(final[None])[0] >= 45.0) & eligible).any())

    replay = JengaReplay(DEFAULT_LMDB)
    episode = replay.episode(episode_id)
    replay.close()
    sim = DirectJengaSim(xml)
    out = {}
    try:
        # Pass 1: an UNINTERRUPTED replay, snapshotting each chunk start. Restoring and then
        # continuing a replay is never bit-identical to not interrupting it (the restore leaves
        # forward-consistent kinematics where the replay reads them one substep late, and drops
        # the solver warm start), which is how Stage 0's replays drifted. So probing never
        # happens mid-replay.
        sim.reset(int(episode_id) + 1000)                         # batch-3 reset seed base
        starts = {f["chunk_start"] for f in forks}
        clean = {}
        for step, action in enumerate(episode.actions):
            if step in starts:
                clean[step] = (sim.snapshot(), sim.data.qacc_warmstart.copy(), step_state(sim))
                if len(clean) == len(starts):
                    break
            sim.execute(action)
        # Pass 2: every probe from the clean snapshot, warm start restored with it.
        for step in sorted(clean):
            snapshot, warm, start = clean[step]
            traces, labels = [], {"snapshot": [], "zero": [], "carry": []}
            # snapshot: warm start restored with the state -> deterministic dense traces
            for window in windows[step]:
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm
                trace = []
                for future in window[2:]:
                    trace.extend(execute_recording(sim, future, EVERY))
                traces.append(np.stack(trace))
                labels["snapshot"].append(toppled(trace[-1], start))
            # zero and carry: final outcome only
            for condition in ("zero", "carry"):
                for window in windows[step]:
                    sim.restore(snapshot)
                    if condition == "zero":
                        sim.data.qacc_warmstart[:] = 0.0
                    for future in window[2:]:
                        sim.execute(future)
                    labels[condition].append(toppled(step_state(sim), start))
            out[step] = {"start": start, "traces": np.stack(traces).astype(np.float32),
                         "labels": {k: np.asarray(v) for k, v in labels.items()}}
    finally:
        sim.close()
    return episode_id, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default=str(ROOT / "results/jenga/blind_freq_gnn_n5.json"))
    ap.add_argument("--out", default=str(ROOT / "results/jenga/blind_resim"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from jenga_bench import BENCH_FILE, verify
    from jenga_short_held_tails import extract_sim
    verify()
    bench = np.load(BENCH_FILE, allow_pickle=False)
    keys = list(zip(bench["test_episode"].tolist(), bench["test_chunk"].tolist()))
    index = {k: i for i, k in enumerate(keys)}
    k_scale = [str(s) for s in json.loads(Path(BENCH_FILE).with_name("manifest.json")
                                          .read_text())["eval_config"]["scales"]].index(SCALE)
    starts_frozen, windows_frozen = bench["test_start"], bench["test_windows"]
    real_frozen = bench["test_real_mm"]
    # Per-probe Stage-0 topple labels at the end of the hold (the frozen oracle's own verdicts).
    from jenga_stage0_noise_oracle import RECORD_AT
    stage0_rows = json.loads((ROOT / "results/jenga/holdout3_stage0.json").read_text())["rows"]
    stage0_index = {(r["episode_id"], r["chunk_start"]): i for i, r in enumerate(stage0_rows)}
    stage0_topple = np.load(ROOT / "results/jenga/holdout3_stage0_cache.npz",
                            allow_pickle=False)["topple"]
    hold_at = RECORD_AT.index(30)

    forks = select(args.freq)
    by_episode = {}
    for f in forks:
        by_episode.setdefault(f["episode_id"], []).append(f)
    print(f"{sum(f['group'] == 'blind' for f in forks)} blind + "
          f"{sum(f['group'] == 'control' for f in forks)} control forks over "
          f"{len(by_episode)} episodes, {EVERY}-substep readings", flush=True)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jenga_resim_") as temp:
        xml = str(extract_sim(ROOT / "vendor/panda_express_sim.tar", temp))
        jobs = []
        for e, fs in by_episode.items():
            windows = {f["chunk_start"]: windows_frozen[index[(e, f["chunk_start"])], k_scale]
                       for f in fs}
            jobs.append((e, fs, xml, windows))
        results = {}
        with ProcessPoolExecutor(args.workers) as pool:
            for done, (episode_id, data) in enumerate(pool.map(simulate_episode, jobs), 1):
                results[episode_id] = data
                print(f"  episode {done}/{len(jobs)} ({episode_id})", flush=True)

    manifest = []
    for f in forks:
        i = index[(f["episode_id"], f["chunk_start"])]
        data = results[f["episode_id"]][f["chunk_start"]]
        # Reproduction checks against the frozen benchmark, before anything is written.
        if not np.allclose(data["start"], starts_frozen[i], atol=1e-6):
            raise SystemExit(f"{f['episode_id']}:{f['chunk_start']} start state differs from "
                             f"the frozen benchmark")
        ending = 1000 * (data["traces"][:, -1, 0:9] - data["start"][None, 0:9])
        worst = float(np.abs(ending - real_frozen[i, k_scale]).max())
        frozen = stage0_topple[stage0_index[(f["episode_id"], f["chunk_start"])], k_scale, :,
                               hold_at]
        agreement = {}
        for condition, labels in data["labels"].items():
            agreement[condition] = {
                "topples": int(labels.sum()),
                "probes_agreeing_with_stage0": int((labels == frozen).sum()),
                "toppled_here_not_in_stage0": int((labels & ~frozen).sum()),
                "toppled_in_stage0_not_here": int((~labels & frozen).sum())}
        stable_set = all(np.array_equal(data["labels"][c], data["labels"]["snapshot"])
                         for c in ("zero", "carry"))
        name = f"ep{f['episode_id']}_step{f['chunk_start']}.npz"
        np.savez_compressed(out_dir / name, start=data["start"], traces=data["traces"],
                            windows=windows_frozen[i, k_scale], stage0_topple=frozen,
                            **{f"topple_{c}": v for c, v in data["labels"].items()})
        manifest.append({"file": name, "group": f["group"], "episode_id": f["episode_id"],
                         "chunk_start": f["chunk_start"], "q": f["q"],
                         "q_matched": f["q_matched"], "topple_count": f["topple_count"],
                         "stage0_topples": int(frozen.sum()),
                         "real_spread_mm": f["real_spread_mm"],
                         "max_ending_diff_vs_stage0_mm": worst,
                         "warmstart_conditions": agreement,
                         "toppling_set_identical_across_warmstart": stable_set})
    (out_dir / "index.json").write_text(json.dumps(
        {"readings_per_control_step": 50 // EVERY, "scale": SCALE, "forks": manifest},
        indent=1) + "\n")
    for m in manifest:
        w = m["warmstart_conditions"]
        print(f"  {m['group']:7s} ep{m['episode_id']:>3s} step{m['chunk_start']:>4d}  stage0 "
              f"{m['stage0_topples']:2d}/64  snapshot {w['snapshot']['topples']:2d} zero "
              f"{w['zero']['topples']:2d} carry {w['carry']['topples']:2d}  agree(snapshot) "
              f"{w['snapshot']['probes_agreeing_with_stage0']}/64  same set: "
              f"{m['toppling_set_identical_across_warmstart']}", flush=True)


if __name__ == "__main__":
    main()

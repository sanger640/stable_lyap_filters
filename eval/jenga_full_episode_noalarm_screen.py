"""Screen remaining successful-pick episodes for at least one physical alarm."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_local_delta_probes import probe_scalars  # noqa: E402
from jenga_multipeak_oracle import original_physical_responses  # noqa: E402
from jenga_short_held_tails import extract_sim  # noqa: E402
from jenga_trajectory_oracle import collect, read_cache, score_rows, write_cache  # noqa: E402


def choose_sentinels(all_block_path, known_alarm_episodes, extra_success_episodes=()):
    data = dict(np.load(all_block_path, allow_pickle=False))
    ids = data["episode_ids"].astype(str); starts = data["chunk_starts"].astype(int)
    lift = data["tail5_max_z"][:, 0] - data["start_position"][:, 0, 2]
    lateral = np.linalg.norm(data["tail5_end_position"][:, 0, :2]
                             - data["start_position"][:, 0, :2], axis=1)
    contact = np.any(data["tail5_contact_seen"][:, 10:12], axis=1)
    metadata, successes = [], []
    for episode in sorted(set(ids), key=int):
        indices = np.flatnonzero(ids == episode)
        picks = indices[(lift[indices] >= .025) & (lateral[indices] >= .02)]
        extra = str(episode) in extra_success_episodes
        if not extra and (not len(picks) or np.max(data["tail0_peak_tilt"][indices, 1:]) >= 45):
            continue
        successes.append(str(episode))
        if str(episode) in known_alarm_episodes:
            continue
        contacts = indices[contact[indices]]
        if not len(contacts):
            continue
        # Check a nominal neighbor contact nearest to the first visible red-block pick.
        anchor = int(starts[picks[0]]) if len(picks) else int(starts[indices[np.argmax(lift[indices])]])
        selected = int(contacts[np.argmin(np.abs(starts[contacts] - anchor))])
        metadata.append({"episode_id": str(episode), "chunk_start": int(starts[selected]),
                         "center_offset": 0.0, "stratum": "contact_sentinel",
                         "first_pick_chunk": anchor})
    return metadata, successes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--all-block-cache", default=str(ROOT / "results/jenga/all_block_tails_cache.npz"))
    ap.add_argument("--contact-result", default=str(ROOT / "results/jenga/contact_pick_check.json"))
    ap.add_argument("--full-audit", default=str(ROOT / "results/jenga/full_episode_alarm_audit.json"))
    ap.add_argument("--prior-screen")
    ap.add_argument("--additional-audit")
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/full_episode_sentinel_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/full_episode_noalarm_screen.json"))
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--extra-success", nargs="*", default=[])
    args = ap.parse_args()
    contact_rows = json.loads(Path(args.contact_result).read_text())["rows"]
    audit = json.loads(Path(args.full_audit).read_text())
    known = {r["episode_id"] for r in contact_rows if r["alarm"]}
    known |= {ep for ep, summary in audit["episodes"].items()
              if summary["successful_pick_proxy"] and summary["alarm_count"]}
    if args.prior_screen:
        prior = json.loads(Path(args.prior_screen).read_text())
        known |= {r["episode_id"] for r in prior["sentinel_rows"] if r["alarm"]}
    if args.additional_audit:
        additional = json.loads(Path(args.additional_audit).read_text())
        known |= {ep for ep, summary in additional["episodes"].items()
                  if summary["alarm_count"]}
    selected, successes = choose_sentinels(args.all_block_cache, known,
                                            set(args.extra_success))
    scalars = probe_scalars(50)
    if args.reuse_cache:
        metadata, scalars, trajectories, outcomes = read_cache(args.cache)
    else:
        replay = JengaReplay(args.lmdb)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_full_episode_sentinel_") as temp:
                metadata, trajectories, outcomes = collect(
                    selected, replay, extract_sim(args.sim_archive, temp), scalars, .10)
        finally:
            replay.close()
        write_cache(args.cache, metadata, scalars, trajectories, outcomes)
    scale = robust_component_scale(original_physical_responses(args.original_cache))
    rows = score_rows(scalars, trajectories, outcomes, metadata, scale)
    found = known | {r["episode_id"] for r in rows if r["alarm"]}
    unresolved = sorted(set(successes) - found, key=int)
    result = {"successful_pick_episodes_screened": len(successes),
              "episodes_with_at_least_one_verified_alarm": len(set(successes) & found),
              "unresolved_episodes_not_proven_quiet": unresolved,
              "known_alarms_from_contact_pick_or_full_audit": sorted(known, key=int),
              "sentinel_rows": rows}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sentinel_rows"}, indent=2))


if __name__ == "__main__":
    main()

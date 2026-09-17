"""Audit all nominal episodes and cross-check verified physical alarm decisions."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_full_episode_alarm_audit import nominal_episode  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, extract_sim  # noqa: E402

ALARM_RESULTS = (
    "contact_pick_check.json", "full_episode_alarm_audit.json",
    "full_episode_ep4_audit.json", "full_episode_noalarm_screen.json",
    "full_episode_extra_noalarm_screen.json",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--result-dir", default=str(ROOT / "results/jenga"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/nominal_success_scan.json"))
    args = ap.parse_args()
    alarmed = set()
    for name in ALARM_RESULTS:
        data = json.loads((Path(args.result_dir) / name).read_text())
        rows = data.get("rows", data.get("sentinel_rows", []))
        alarmed.update(r["episode_id"] for r in rows if r["alarm"])
    replay = JengaReplay(args.lmdb)
    try:
        with tempfile.TemporaryDirectory(prefix="jenga_nominal_success_") as temp:
            sim = DirectJengaSim(extract_sim(args.sim_archive, temp))
            try:
                episodes = {}
                for number, episode_id in enumerate(replay.episode_ids):
                    episodes[episode_id] = nominal_episode(sim, replay.episode(episode_id))
                    if (number + 1) % 20 == 0:
                        print(f"nominal screened {number + 1}", flush=True)
            finally:
                sim.close()
    finally:
        replay.close()
    success = sorted([ep for ep, value in episodes.items()
                      if value["successful_pick_proxy"]], key=int)
    missing = sorted(set(success) - alarmed, key=int)
    result = {"protocol": {"successful_pick_proxy":
                           "post-settle red lift >=2.5 cm and lateral motion >=2 cm; "
                           "nominal neighbor peak tilt <45 deg",
                           "alarm_evidence_files": list(ALARM_RESULTS)},
              "episode_count": len(episodes), "successful_pick_episodes": len(success),
              "successful_pick_ids": success,
              "successful_pick_with_verified_alarm": len(success) - len(missing),
              "successful_pick_without_verified_alarm": missing,
              "episodes": episodes}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()

"""Descriptive analysis of the immutable one-time Regime Monitor v0 TEST result.

This script cannot change alarms. It reports candidate attrition, binomial and episode-clustered
uncertainty, DEV-to-TEST change, and episode-level intervention burden in a separate artifact.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_regime_monitor_v0_freeze import sha256_file, verify as verify_v0  # noqa: E402

TEST_RESULT = ROOT / "results/jenga/regime_monitor_v0/test_ground_truth.json"
OUTPUT = ROOT / "results/jenga/regime_monitor_v0/test_ground_truth_analysis.json"


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(centre - radius), float(centre + radius)]


def clustered_interval(rows, field="alarm", resamples=10000, seed=0):
    """Episode-clustered percentile interval for a binary state-level rate."""
    episodes = sorted({row["episode_id"] for row in rows}, key=int)
    grouped = {episode: [bool(row[field]) for row in rows if row["episode_id"] == episode]
               for episode in episodes}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        sample = rng.choice(episodes, len(episodes), replace=True)
        observations = [value for episode in sample for value in grouped[str(episode)]]
        values.append(np.mean(observations))
    return [float(x) for x in np.quantile(values, [.025, .975])]


def analyse(result):
    rows = result["rows"]
    classes = ("topple_fork", "quiet", "nudge_fork", "small_split", "weak")
    by_class = {}
    for name in classes:
        selected = [row for row in rows if row["class"] == name]
        alarms = sum(row["alarm"] for row in selected)
        initial = sum(row["initial_alarm"] for row in selected)
        boundary = sum(row["boundary_refined_alarm"] for row in selected)
        by_class[name] = {
            "states": len(selected),
            "initial_candidates": initial,
            "boundary_candidates": boundary,
            "v0_alarms": alarms,
            "initial_rate": float(initial / len(selected)) if selected else None,
            "boundary_rate": float(boundary / len(selected)) if selected else None,
            "v0_rate": float(alarms / len(selected)) if selected else None,
            "v0_wilson_95": wilson(alarms, len(selected)),
            "v0_episode_clustered_95": clustered_interval(selected) if selected else [None, None],
        }
    episode_ids = sorted({row["episode_id"] for row in rows}, key=int)
    alarmed_episodes = {row["episode_id"] for row in rows if row["alarm"]}
    all_alarms = sum(row["alarm"] for row in rows)
    return {
        "by_class": by_class,
        "stage_attrition": {
            "topple_fork": {"missed_before_candidate": 84 - 80,
                            "lost_at_boundary": 80 - 78,
                            "lost_at_commitment_persistence": 78 - 55},
            "quiet": {"not_candidate": 113 - 18,
                      "removed_at_boundary": 18 - 16,
                      "removed_at_commitment_persistence": 16 - 4},
        },
        "dev_to_test": {
            "topple_fork_rate": {"dev": 18 / 23, "test": 55 / 84,
                                 "test_minus_dev": 55 / 84 - 18 / 23},
            "quiet_branch_rate": {"dev": 7 / 89, "test": 4 / 113,
                                  "test_minus_dev": 4 / 113 - 7 / 89},
        },
        "intervention_burden": {
            "states": len(rows), "alarmed_states": int(all_alarms),
            "state_alarm_rate": float(all_alarms / len(rows)),
            "episodes": len(episode_ids), "episodes_with_alarm": len(alarmed_episodes),
            "episode_alarm_rate": float(len(alarmed_episodes) / len(episode_ids)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(TEST_RESULT))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    if source.resolve() != TEST_RESULT.resolve() or output.resolve() != OUTPUT.resolve():
        raise SystemExit("analysis uses only the canonical immutable TEST paths")
    if output.exists():
        raise SystemExit(f"refusing to overwrite TEST analysis: {output}")
    manifest = verify_v0()
    result = json.loads(source.read_text())
    if result["regime_monitor_v0_sha256"] != manifest["protocol_sha256"]:
        raise SystemExit("TEST result does not belong to frozen Regime Monitor v0")
    analysis = {
        "source": str(source.relative_to(ROOT)), "source_sha256": sha256_file(source),
        "regime_monitor_v0_sha256": manifest["protocol_sha256"],
        "selection_policy": "descriptive only; no rule or alarm changed after TEST",
        **analyse(result),
    }
    output.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()

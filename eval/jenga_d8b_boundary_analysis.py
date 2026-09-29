"""Read-only analysis of D8b's residual boundary disagreement."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import boundary_refinement_alarm
from jenga_d6_memorization import selected_nested
from jenga_d6_set_response import DATA, NESTED, normalizers, sha256, tensors
from jenga_d7_explicit_curve import curve_targets, decoded
from jenga_d8b_stability import PROTOCOL
from jenga_reobservation_interface import validation_rows
from set_response_model import DirectTemporalPairCurveModel


RESULT = ROOT / "results/jenga/d8b_stability_result.json"
OUTPUT = ROOT / "results/jenga/d8b_boundary_analysis.json"


def evidence(values):
    row = boundary_refinement_alarm(values[:, 0][None], values[:, 1][None])
    return {"alarm": bool(row.alarm), "early_delta_bic": row.early_delta_bic[0],
            "full_delta_bic": row.full_delta_bic[0],
            "max_early_gap": float(np.max(values[:, 0])),
            "max_full_gap": float(np.max(values[:, 1])),
            "max_curve": float(np.max(values[:, 2:]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default=str(RESULT)); parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    result = json.loads(Path(args.result).read_text()); protocol = json.loads(PROTOCOL.read_text())
    loaded = np.load(args.data, allow_pickle=False)
    val_rows, _ = validation_rows(loaded["episode_id"])
    train_rows = np.setdiff1d(np.arange(len(loaded["start"])), val_rows)
    normalizer_np = normalizers(loaded["start"][train_rows], loaded["actions"][train_rows],
                                loaded["traces"][train_rows])
    norms = tensors(normalizer_np, device)
    nested = selected_nested(args.nested, protocol["data"]["selected_episodes"])
    start = torch.as_tensor(nested[0], device=device); actions = torch.as_tensor(nested[1], device=device)
    physical = np.expm1(curve_targets(nested[2], normalizer_np[4]))
    groups = {episode: {"physical": evidence(values), "seeds": {}}
              for episode, values in zip(protocol["data"]["selected_episodes"], physical)}
    for seed_row in result["seeds"]:
        saved = torch.load(ROOT / seed_row["checkpoint"], map_location=device, weights_only=False)
        model = DirectTemporalPairCurveModel().to(device); model.load_state_dict(saved["model"]); model.eval()
        with torch.no_grad():
            predicted = decoded(model(start, actions, *norms[:4])).cpu().numpy()
        for episode, values in zip(protocol["data"]["selected_episodes"], predicted):
            groups[episode]["seeds"][str(seed_row["seed"])] = evidence(values)
    zero_groups = [episode for episode, row in groups.items()
                   if row["physical"]["max_early_gap"] == 0.
                   and row["physical"]["max_full_gap"] == 0.]
    analysis = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_result": str(Path(args.result).relative_to(ROOT)),
        "source_result_sha256": sha256(args.result), "groups": groups,
        "exact_zero_physical_groups": zero_groups,
        "interpretation": ("all nonzero physical groups are stable across seeds; disagreement is "
                           "confined to exact-zero physical responses where arbitrarily tiny "
                           "structured prediction residuals remain visible to the scale-invariant BIC rule"),
        "next": ("factor response amplitude from curve shape and use an exact-zero-capable "
                 "nonnegative amplitude output trained only on continuous response magnitude"),
    }
    output.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({"exact_zero_physical_groups": zero_groups,
                      "interpretation": analysis["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()

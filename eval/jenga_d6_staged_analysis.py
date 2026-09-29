"""Read-only mechanism summary for the completed D6 staged overfit audit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from boundary_scale_loss import boundary_scale_terms
from jenga_d6_memorization import selected_nested
from jenga_d6_set_response import DATA, NESTED, normalizers, tensors, sha256
from jenga_reobservation_interface import validation_rows
from set_response_model import DirectSetResponseModel, reconstruct_states


RESULT = ROOT / "results/jenga/d6_staged_overfit_result.json"
OUTPUT = ROOT / "results/jenga/d6_staged_overfit_analysis.json"


def nested_components(arm, checkpoint, nested, norms, full_scale, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DirectSetResponseModel(set_conditioned=arm == "set_conditioned").to(device)
    model.load_state_dict(saved["model"]); model.eval()
    start = torch.as_tensor(nested[0], device=device)
    actions = torch.as_tensor(nested[1], device=device)
    truth = torch.as_tensor(nested[2], device=device)
    groups = len(start)
    flat_actions = actions.reshape(groups, -1, actions.shape[-2], 4)
    flat_truth = truth.reshape(groups, -1, truth.shape[-2], 61)
    with torch.no_grad():
        estimate = model(start, flat_actions, *norms[:4])
        predicted = reconstruct_states(estimate, start, flat_truth, norms[4]).reshape_as(truth)
        terms = boundary_scale_terms(predicted, truth, full_scale)
    return {name: float(getattr(terms, name))
            for name in ("response", "effect", "global_scale", "local_scale")}


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
    result = json.loads(Path(args.result).read_text())
    loaded = np.load(args.data, allow_pickle=False)
    val_rows, _ = validation_rows(loaded["episode_id"])
    train_rows = np.setdiff1d(np.arange(len(loaded["start"])), val_rows)
    norms = tensors(normalizers(loaded["start"][train_rows], loaded["actions"][train_rows],
                                loaded["traces"][train_rows]), device)
    full_scale = torch.ones(61, device=device); full_scale[:45] = norms[4]
    episodes = result["protocol"]["data"]["selected_episodes"]
    nested = selected_nested(args.nested, episodes)
    arms = {}
    for arm, content in result["arms"].items():
        stages = content["stages_by_name"]
        final = stages["add_nested"]
        checkpoint = ROOT / final["checkpoint"]
        arms[arm] = {
            "stage_table": {name: {
                "response": stage["evaluation"]["continuous"]["response"],
                "topology": stage["evaluation"]["continuous"]["topology"],
                "commitment": stage["evaluation"]["continuous"]["commitment"],
                "candidate_agreement": stage["evaluation"]["candidate"]["agreement"],
                "partition_agreement": stage["evaluation"]["candidate"]["mean_partition_agreement"],
                "nested_boundary_agreement": stage["evaluation"]["nested"]["boundary"]["agreement"],
                "nested_final_agreement": stage["evaluation"]["nested"]["alarm"]["agreement"],
                "gate_passes": stage["gate"]["passes"],
            } for name, stage in stages.items()},
            "nested_components_final": nested_components(
                arm, checkpoint, nested, norms, full_scale, device),
            "nested_entry_weighted_gradient_norms": {
                name: values["weighted"] for name, values in
                stages["add_nested"]["gradient_before"]["gradient_norms"].items()},
            "nested_entry_gradient_cosines":
                stages["add_nested"]["gradient_before"]["gradient_cosines"],
        }
    analysis = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_result": str(Path(args.result).relative_to(ROOT)),
        "source_result_sha256": sha256(args.result),
        "arms": arms,
        "interpretation": {
            "raw_response": "fixed raw-only stages underfit, but the set arm later passes all continuous gates under curriculum, so decoder impossibility is not supported",
            "topology_commitment": "set-conditioned add_commitment passes response, topology, and commitment gates; these relational objectives help rather than destroy fitting",
            "nested": "adding nested supervision preserves continuous neighborhood fit but fails exact candidate and nested monitor decisions",
            "mechanism": "at nested-stage entry its weighted gradient dominates and conflicts with commitment; the aggregate D3 loss can become small without preserving the threshold-sensitive final regime decision",
            "next": "replace aggregate nested regression with an explicit continuous pair-curve representation and balanced per-group/per-level reconstruction, then require exact same-example decisions before held-out evaluation"
        }
    }
    output.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()

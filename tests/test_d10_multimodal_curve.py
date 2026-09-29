from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d10_multimodal_curve import mixture_nll, pilot_gate


def test_mixture_nll_rewards_a_component_matching_the_complete_curve():
    target = torch.ones(2, 3, 4)
    bad = torch.zeros(2, 3, 3, 4)
    good = bad.clone(); good[:, 1] = target
    logits = torch.zeros(2, 3)
    assert mixture_nll(good, logits, target, .5) < mixture_nll(bad, logits, target, .5)


def test_mixture_nll_uses_component_probabilities():
    target = torch.ones(1, 2, 3)
    curves = torch.stack([target, torch.zeros_like(target)], dim=1)
    correct = torch.tensor([[5., -5.]])
    wrong = torch.tensor([[-5., 5.]])
    assert mixture_nll(curves, correct, target, .5) < mixture_nll(curves, wrong, target, .5)


def test_multimodal_gate_requires_specificity_recall_error_and_symmetry():
    def arm(mse, boundary_added, final_added, boundary_recall, final_recall):
        return {"mean_mse": mse, "nested": {
            "boundary": {"added_positive_rate": boundary_added, "recall": boundary_recall},
            "alarm": {"added_positive_rate": final_added, "recall": final_recall}}}
    config = {"boundary_added_positive_reduction_from_snapshot": .1,
              "final_added_positive_reduction_from_snapshot": .1,
              "maximum_boundary_recall_loss_from_snapshot": .05,
              "maximum_final_recall_loss_from_snapshot": .05,
              "endpoint_and_level_symmetry_max_abs": 1e-5}
    snapshot = arm(.6, .5, .4, .9, .8)
    mixture = arm(.5, .35, .25, .86, .76)
    assert pilot_gate(snapshot, mixture, 1e-6, config)["passes"]
    mixture["nested"]["alarm"]["recall"] = .7
    assert not pilot_gate(snapshot, mixture, 1e-6, config)["passes"]

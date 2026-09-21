"""The D2 intervention-consistency terms must reward exactly what they claim to, and nothing more."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))
from jenga_w6_simple import CONTINUOUS, intervention_terms  # noqa: E402

SCALE = torch.ones(len(CONTINUOUS))


def branches(g=4, b=5, h=6, seed=0):
    torch.manual_seed(seed)
    return torch.randn(g, b, h, 61)


def test_all_terms_vanish_on_a_perfect_prediction():
    truth = branches()
    terms = intervention_terms(truth.clone(), truth, SCALE, SCALE, quiet_tau=1e9)
    assert all(float(v) == 0.0 for v in terms.values())


def test_a_bias_shared_by_every_branch_is_caught_by_response_not_difference():
    """The Phase 2 failure: every branch under-delivers by the same amount."""
    truth = branches()
    biased = truth.clone(); biased[..., CONTINUOUS] -= 0.5
    terms = intervention_terms(biased, truth, SCALE, SCALE, quiet_tau=0.0)
    assert float(terms["response"]) > 0.2
    assert float(terms["difference"]) < 1e-10          # the effect between branches is unchanged


def test_a_wrong_intervention_effect_is_caught_by_difference():
    truth = branches()
    wrong = truth.clone(); wrong[:, 1:, :, CONTINUOUS] *= 0.0         # branches collapse to 0
    assert float(intervention_terms(wrong, truth, SCALE, SCALE, 0.0)["difference"]) > 0.1


def test_quiet_term_penalises_divergence_only_where_reality_does_not_diverge():
    truth = branches()
    truth[:, 1:] = truth[:, :1]                                       # interventions change nothing
    invented = truth.clone(); invented[:, 1:, :, CONTINUOUS] += 0.3   # model invents a branch
    terms = intervention_terms(invented, truth, SCALE, SCALE, quiet_tau=0.1)
    assert float(terms["quiet"]) > 0.05
    # Where the real effect is large, the quiet term is silent.
    real = branches(seed=1)
    assert float(intervention_terms(real.clone(), real, SCALE, SCALE, 1e-6)["quiet"]) == 0.0


def test_terms_use_only_the_generic_continuous_state():
    """Contacts and the gripper closed flag must not enter the objective."""
    truth = branches()
    other = [i for i in range(61) if i not in CONTINUOUS]
    changed = truth.clone(); changed[..., other] += 5.0
    terms = intervention_terms(changed, truth, SCALE, SCALE, quiet_tau=1e9)
    assert all(float(v) == 0.0 for v in terms.values())


def test_batched_branch_terms_equal_the_looped_version():
    """The speed-up must not change the D1 objective: same loss and ratio, on CPU and GPU."""
    from jenga_w6_simple import branch_terms, branch_terms_batched
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        torch.manual_seed(0)
        groups, branches, steps = 32, 5, 6
        predicted = torch.randn(groups * branches, steps, 61, device=device, requires_grad=True)
        truth = torch.randn(groups * branches, steps, 61, device=device)
        scale = torch.rand(61, device=device) + 0.5
        loop, loop_ratio = branch_terms(predicted, truth, [branches] * groups, scale)
        batched, batched_ratio = branch_terms_batched(predicted, truth, branches, scale)
        g_loop = torch.autograd.grad(loop, predicted)[0]
        g_batched = torch.autograd.grad(batched, predicted)[0]
        assert torch.allclose(loop, batched, rtol=1e-6, atol=0), device
        assert torch.allclose(g_loop, g_batched, rtol=1e-5, atol=1e-12), device
        assert abs(loop_ratio - batched_ratio) < 1e-6 * abs(loop_ratio)

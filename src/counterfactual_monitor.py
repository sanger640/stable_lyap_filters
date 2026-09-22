"""Calibration-free counterfactual regime monitor.

The monitor consumes two snapshots of the *same* execution-noise counterfactual rollouts.  It
raises an alarm only when both snapshots contain the same statistically supported two-mode split.
It has no fit/calibrate method and accepts no examples of safe or failed behaviour.
"""
from dataclasses import asdict, dataclass

import numpy as np

from outcome_modes import same_partition, two_mode_test


@dataclass(frozen=True)
class CounterfactualAlarm:
    alarm: bool
    persistent_partition: bool
    early: dict
    late: dict


def persistent_mode_alarm(early_endings, late_endings):
    """Decide whether counterfactual futures form a persistent regime split.

    ``early_endings`` and ``late_endings`` have shape ``(probes, features)`` and must preserve
    probe order.  The rule is coordinate-scale invariant except for numerical precision: BIC and
    Ashman's D compare separation with within-mode variation, and no absolute distance enters.

    Constants are structural, not fitted: BIC penalises the extra Gaussian; Ashman's D > 2 is the
    standard separated-bimodality criterion; each mode needs at least two probes; at most one probe
    may switch sides between the two times.
    """
    early_x = np.asarray(early_endings, float)
    late_x = np.asarray(late_endings, float)
    if early_x.ndim != 2 or late_x.ndim != 2 or early_x.shape != late_x.shape:
        raise ValueError("early and late endings must have matching (probes, features) shape")
    early = two_mode_test(early_x, perturbation_mm=0.0)
    late = two_mode_test(late_x, perturbation_mm=0.0)

    def supported(test):
        return test.bic_prefers_two and test.ashman_d > 2 and test.minority >= 2

    persistent = supported(early) and supported(late) and same_partition(
        early.labels, late.labels)
    return CounterfactualAlarm(bool(persistent), bool(persistent), early.summary(), late.summary())


def asdict_without_labels(result):
    """JSON-ready public result (mode labels are intentionally not part of the runtime API)."""
    return asdict(result)

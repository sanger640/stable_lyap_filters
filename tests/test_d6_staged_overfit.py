from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from jenga_d6_staged_overfit import diagnosis, stage_gate


def metrics(response=0.0, topology=0.0, commitment=0.0, agreement=1.0):
    return {
        "continuous": {"response": response, "topology": topology, "commitment": commitment},
        "candidate": {"agreement": agreement, "mean_partition_agreement": agreement},
        "nested": {"boundary": {"agreement": agreement}, "alarm": {"agreement": agreement}},
        "checks": {"permutation_max_abs": 0.0},
    }


def test_stage_gate_adds_requirements_progressively():
    spread = {"action": 1.0, "early_hold": 1.0, "late_hold": 1.0}
    assert stage_gate("one_state_response", metrics(response=0.004), spread)["passes"]
    assert not stage_gate("one_state_response", metrics(response=0.006), spread)["passes"]
    assert stage_gate("add_commitment", metrics(), spread)["passes"]
    assert not stage_gate("add_commitment", metrics(commitment=0.02), spread)["passes"]
    assert stage_gate("add_nested", metrics(), spread)["passes"]
    assert not stage_gate("add_nested", metrics(agreement=0.8), spread)["passes"]


def test_diagnosis_reports_first_failed_stage_per_arm():
    stages = [{"name": name} for name in (
        "one_state_response", "eight_state_response", "add_topology",
        "add_commitment", "add_nested")]
    def arm(failure):
        return {"stages": stages, "stages_by_name": {
            item["name"]: {"gate": {"passes": item["name"] != failure}} for item in stages}}
    result = diagnosis({"independent_direct": arm("add_topology"),
                        "set_conditioned": arm("add_topology")})
    assert result["first_failed_stage"] == {
        "independent_direct": "add_topology", "set_conditioned": "add_topology"}
    assert result["conclusion"] == "relational_objective_interference"


def test_diagnosis_recognizes_nested_bottleneck_after_continuous_pass():
    stages = [{"name": name} for name in (
        "one_state_response", "eight_state_response", "add_topology",
        "add_commitment", "add_nested")]
    def arm():
        return {"stages": stages, "stages_by_name": {
            item["name"]: {"gate": {"passes": item["name"] == "add_commitment"}}
            for item in stages}}
    result = diagnosis({"independent_direct": arm(), "set_conditioned": arm()})
    assert result["conclusion"] == (
        "neighborhood_geometry_representable_nested_consequence_bottleneck")

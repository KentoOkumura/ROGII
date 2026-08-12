from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp267_well_segment_candidate_divergence_signature_cluster_on_exp265"
)


def test_exp267_contract_disables_post_audit_training_and_inference() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())

    assert config["experiment"]["route"] == "ensemble"
    assert config["execution"]["run_approved"] is False
    assert config["execution"]["stage_a_total_boosters"] == 0
    assert config["model"]["conditional_stage_b"]["enabled"] is False
    assert config["model"]["conditional_stage_b"]["planned_cpu_boosters"] == 10
    assert config["execution"]["inference_enabled"] is False
    assert config["features"]["expected_feature_count"] == 18

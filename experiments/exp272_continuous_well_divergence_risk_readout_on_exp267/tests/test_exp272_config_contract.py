from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp272_continuous_well_divergence_risk_readout_on_exp267"


def test_exp272_config_keeps_zero_booster_and_inference_disabled() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())

    assert config["experiment"]["route"] == "ensemble"
    assert config["axes"]["primary"] == "fixed_range_gap_axis"
    assert config["axes"]["sensitivity_decision_role"] == (
        "report_only_cannot_rescue_primary_guard"
    )
    assert config["model"]["variants"] == 0
    assert config["model"]["lightgbm_configs"] == 0
    assert config["model"]["folds_trained"] == 0
    assert config["model"]["total_boosters"] == 0
    assert config["execution"]["run_approved"] is False
    assert config["execution"]["inference_enabled"] is False
    assert config["execution"]["submission_enabled"] is False
    assert config["inference"]["enabled"] is False
    assert config["inference"]["create_submission"] is False

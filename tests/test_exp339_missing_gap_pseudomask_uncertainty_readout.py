from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp339_missing_gap_pseudomask_uncertainty_readout"
TRAIN_SOURCE = EXP_DIR / (
    "exp339_missing_gap_pseudomask_uncertainty_readout_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp339_missing_gap_pseudomask_uncertainty_readout_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP339_IMPORT_ONLY")
    os.environ["EXP339_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP339_IMPORT_ONLY", None)
        else:
            os.environ["EXP339_IMPORT_ONLY"] = previous


def load_config(module):
    return module.read_yaml(EXP_DIR / "config.yaml")


def make_profile(gr: np.ndarray, well: str = "well-a") -> dict:
    values = np.asarray(gr, dtype=np.float64)
    return {
        "well_id": well,
        "prefix_rows": len(values),
        "gr": values,
        "finite_gr": np.isfinite(values),
    }


def test_stage0_contract_is_zero_hmm_zero_model_and_run_approved() -> None:
    module = load_module(TRAIN_SOURCE, "exp339_contract")
    config = load_config(module)
    module.validate_scientific_contract(config)
    counts = module.get_nested(config, "execution_contract")
    assert counts == {
        "scientific_readouts": 1,
        "control_readouts": 2,
        "hmm_well_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
    }
    assert module.get_nested(config, "execution.implementation_approved") is True
    assert module.get_nested(config, "execution.kaggle_push_approved") is True
    assert module.get_nested(config, "execution.run_stage_0") is True
    module.validate_scientific_contract(config, require_run_approval=True)
    disabled = deepcopy(config)
    disabled["execution"]["run_stage_0"] = False
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        module.validate_scientific_contract(disabled, require_run_approval=True)
    broken = deepcopy(config)
    broken["model"]["uncertainty_table"]["shrinkage_support_k"] = 100
    with pytest.raises(ValueError, match="shrinkage_support_k"):
        module.validate_scientific_contract(broken)
    broken_gate = deepcopy(config)
    broken_gate["model"]["pass_requires_all"]["minimum_distinct_wells_pooled"] = 699
    with pytest.raises(ValueError, match="promotion gate contract"):
        module.validate_scientific_contract(broken_gate)


def test_truth_free_loader_and_known_prefix_contract(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp339_truth_free")
    pd.DataFrame(
        {
            "GR": [10.0, np.nan, 12.0, 13.0, 14.0],
            "TVT_input": [100.0, 101.0, 102.0, np.nan, np.nan],
            "TVT": [100.0, 101.0, 102.0, 103.0, 104.0],
            "error": [0.0, 0.0, 0.0, 99.0, 99.0],
        }
    ).to_csv(tmp_path / "a__horizontal_well.csv", index=False)
    horizontal = module.load_horizontal_without_truth("a", tmp_path)
    assert list(horizontal.columns) == ["GR", "TVT_input"]
    profile = module.build_known_prefix_profile("a", horizontal)
    assert profile["prefix_rows"] == 3
    assert profile["finite_gr"].tolist() == [True, False, True]

    noncontiguous = horizontal.copy()
    noncontiguous.loc[1, "TVT_input"] = np.nan
    noncontiguous.loc[2, "TVT_input"] = 102.0
    with pytest.raises(ValueError, match="contiguous known prefix"):
        module.build_known_prefix_profile("a", noncontiguous)


def test_natural_missing_histogram_clips_at_64_and_uses_fixed_bins() -> None:
    module = load_module(TRAIN_SOURCE, "exp339_histogram")
    gr = np.ones(150, dtype=np.float64)
    gr[3:5] = np.nan
    gr[20:90] = np.nan
    profiles = {"a": make_profile(gr, "a")}
    inventory = module.natural_missing_inventory(profiles, ["a"], 0)
    assert inventory["raw_run_length"].tolist() == [2, 70]
    assert inventory["clipped_run_length"].tolist() == [2, 64]
    assert inventory["length_bin"].tolist() == ["L01_03", "L32_64"]
    histogram = module.build_natural_missing_histogram(inventory)
    assert histogram["natural_run_count"].sum() == 2
    assert histogram.groupby("length_bin")["natural_run_share_within_bin"].sum().eq(1.0).all()


def test_real_and_circular_plans_are_deterministic_matched_and_nonoverlapping() -> None:
    module = load_module(TRAIN_SOURCE, "exp339_plans")
    profile = make_profile(np.linspace(1.0, 300.0, 300), "well-a")
    histogram = pd.DataFrame(
        {
            "outer_fold": [0, 0, 0],
            "length_bin": ["L01_03", "L04_07", "L04_07"],
            "clipped_run_length": [2, 4, 7],
            "natural_run_count": [8, 3, 5],
            "natural_wells": [5, 3, 4],
            "natural_run_share_within_bin": [1.0, 0.375, 0.625],
        }
    )
    first = module.select_real_gap_plan(
        profile,
        histogram,
        outer_fold=0,
        role="outer_valid",
        maximum_slots=4,
    )
    second = module.select_real_gap_plan(
        profile,
        histogram,
        outer_fold=0,
        role="outer_valid",
        maximum_slots=4,
    )
    assert first == second
    assert len(first) == 8
    assert max(pd.Series([row["length_bin"] for row in first]).value_counts()) == 4
    module.validate_gap_plan(first, profile)
    circular = module.build_circular_control_plan(first, profile)
    module.validate_gap_plan(circular, profile)
    module.validate_matched_plans(first, circular)
    assert any(
        real["start_row"] != control["start_row"]
        for real, control in zip(first, circular, strict=True)
    )


def test_interpolation_is_frozen_before_hidden_gr_late_join() -> None:
    module = load_module(TRAIN_SOURCE, "exp339_freeze")
    profile = make_profile(np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0]), "well-a")
    plan = [
        {
            "outer_fold": 0,
            "role": "outer_valid",
            "placement": "real",
            "well_id": "well-a",
            "length_bin": "L01_03",
            "slot": 0,
            "gap_id": "gap-a",
            "gap_length": 3,
            "start_row": 2,
            "stop_row_exclusive": 5,
            "selection_sha256": "x",
            "control_unchanged": False,
        }
    ]
    predictions, hidden = module.build_interpolation_predictions(profile, plan)
    assert "hidden_raw_gr" not in predictions
    assert "interpolation_error" not in predictions
    np.testing.assert_allclose(predictions["interpolated_gr"], [7.0, 13.0, 19.0])
    frozen_sha = module.dataframe_content_sha(predictions)
    audit = module.attach_hidden_gr_after_prediction_freeze(
        predictions, hidden, frozen_prediction_sha256=frozen_sha
    )
    np.testing.assert_allclose(audit["hidden_raw_gr"], [4.0, 9.0, 16.0])
    np.testing.assert_allclose(audit["interpolation_error"], [-3.0, -4.0, -3.0])
    with pytest.raises(ValueError, match="must be frozen"):
        module.attach_hidden_gr_after_prediction_freeze(
            predictions, hidden, frozen_prediction_sha256="0" * 64
        )


def test_hierarchical_table_uses_outer_train_only_and_fixed_two_level_shrinkage() -> None:
    module = load_module(TRAIN_SOURCE, "exp339_table")
    train = pd.DataFrame(
        {
            "outer_fold": [0, 0, 0, 0],
            "role": ["outer_train"] * 4,
            "placement": ["real"] * 4,
            "length_bin": ["L01_03", "L01_03", "L04_07", "L04_07"],
            "distance_bin": ["D01_01"] * 4,
            "squared_error": [1.0, 1.0, 9.0, 9.0],
        }
    )
    valid = pd.DataFrame(
        {
            "outer_fold": [0],
            "role": ["outer_valid"],
            "placement": ["real"],
            "length_bin": ["L01_03"],
            "distance_bin": ["D01_01"],
            "squared_error": [1_000_000.0],
        }
    )
    first = module.fit_uncertainty_table(
        pd.concat([train, valid], ignore_index=True),
        outer_fold=0,
        placement="real",
        support_k=2,
    )
    second = module.fit_uncertainty_table(
        pd.concat([train, valid.assign(squared_error=2_000_000.0)], ignore_index=True),
        outer_fold=0,
        placement="real",
        support_k=2,
    )
    pd.testing.assert_frame_equal(first, second)
    cell = first.loc[
        first["length_bin"].eq("L01_03") & first["distance_bin"].eq("D01_01")
    ].iloc[0]
    assert cell["global_variance"] == pytest.approx(5.0)
    assert cell["length_variance"] == pytest.approx(3.0)
    assert cell["predicted_variance"] == pytest.approx(2.0)


def test_stage0_gate_is_strict_and_keeps_exp341_disabled() -> None:
    module = load_module(TRAIN_SOURCE, "exp339_gate")
    config = deepcopy(load_config(module))
    gates = config["model"]["pass_requires_all"]
    gates["minimum_pseudogap_coverage_each_fold"] = 1.0
    gates["minimum_wells_each_fold"] = 2
    gates["minimum_distinct_wells_pooled"] = 10
    real_rows = []
    circular_rows = []
    plan_rows = []
    fold_rows = []
    for fold in range(5):
        for row, (length, variance) in enumerate(((1, 0.5), (64, 1.5))):
            common = {
                "outer_fold": fold,
                "role": "outer_valid",
                "gap_length": length,
                "predicted_variance": variance,
                "squared_error": variance,
                "primary_nll": 1.0,
                "global_nll": 2.0,
            }
            real_rows.append({**common, "placement": "real"})
            circular_rows.append({**common, "placement": "circular", "primary_nll": 3.0})
            plan_rows.append(
                {
                    "outer_fold": fold,
                    "role": "outer_valid",
                    "placement": "real",
                    "well_id": f"w{fold}_{row}",
                }
            )
        fold_rows.append(
            {
                "outer_fold": fold,
                "valid_wells": 2,
                "selected_valid_wells": 2,
                "well_coverage": 1.0,
                "primary_better_than_global": True,
                "variance_calibration_ratio": 1.0,
                "gap_length_sigma_spearman": 1.0,
                "real_better_than_circular": True,
            }
        )
    scored = pd.DataFrame(real_rows + circular_rows)
    plans = pd.DataFrame(plan_rows)
    folds = pd.DataFrame(fold_rows)
    gate = module.evaluate_stage0_gate(scored, plans, folds, config)
    assert gate["passed"] is True
    assert gate["exp341_enabled"] is False
    broken = scored.copy()
    broken.loc[broken["placement"].eq("real"), "primary_nll"] = 4.0
    assert module.evaluate_stage0_gate(broken, plans, folds, config)["passed"] is False


def test_inference_is_fail_closed_and_sources_are_not_thin_wrappers() -> None:
    inference = load_module(INFERENCE_SOURCE, "exp339_inference")
    config = load_config(inference)
    contract = inference.validate_disabled_inference(config)
    assert contract["hmm_well_runs"] == 0
    assert contract["inference_enabled"] is False
    with pytest.raises(
        RuntimeError, match="HMM integration, inference, and submission are disabled"
    ):
        inference.stop_disabled_inference(config)

    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert "def select_real_gap_plan" in train_text
    assert "def fit_uncertainty_table" in train_text
    assert "def evaluate_stage0_gate" in train_text
    assert "def run_stage0_experiment" in train_text
    assert "from settings import" not in train_text
    assert "Path(__file__)" not in train_text
    assert "Path(__file__)" not in inference_text
    assert "def run_exact_hmm" not in train_text

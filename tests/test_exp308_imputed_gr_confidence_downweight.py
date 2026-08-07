from __future__ import annotations

import importlib.util
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp308_imputed_gr_confidence_downweight"
TRAIN_SOURCE = EXP_DIR / "exp308_imputed_gr_confidence_downweight_compact_selfcontained_train.py"
INFERENCE_SOURCE = (
    EXP_DIR / "exp308_imputed_gr_confidence_downweight_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP308_IMPORT_ONLY")
    os.environ["EXP308_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP308_IMPORT_ONLY", None)
        else:
            os.environ["EXP308_IMPORT_ONLY"] = previous


def test_frozen_contract_records_one_variant_and_zero_training_cost() -> None:
    module = load_module(TRAIN_SOURCE, "exp308_contract")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    module.validate_scientific_contract(config)
    assert module.get_nested(config, "model.active_variants") == [
        "missing_distance_half8_floor025"
    ]
    assert module.get_nested(config, "model.execution_counts") == {
        "variants": 1,
        "hmm_well_runs": 773,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "boosters": 0,
        "parent_control_retraining": False,
    }
    assert module.get_nested(config, "execution.implementation_approved") is True
    assert module.get_nested(config, "execution.kaggle_push_approved") is False
    with pytest.raises(RuntimeError, match="parent exp307 dependency"):
        module.validate_scientific_contract(config, require_run_approval=True)


def test_missing_distance_confidence_matches_preregistered_formula() -> None:
    module = load_module(TRAIN_SOURCE, "exp308_weight")
    raw = np.array([10.0, np.nan, np.nan, np.nan, 20.0, np.nan, np.nan, np.nan, np.nan])
    audit = module.build_missing_distance_confidence(raw)
    assert audit["nearest_finite_row_distance"].tolist() == [0, 1, 2, 1, 0, 1, 2, 3, 4]
    assert np.array_equal(audit["confidence_weight"][[0, 4]], np.ones(2))
    expected = np.exp2(-np.array([1, 2, 1, 1, 2, 3, 4], dtype=float) / 8.0)
    assert np.array_equal(audit["confidence_weight"][audit["raw_gr_missing"]], expected)
    assert audit["gap_bucket"].tolist() == [
        "observed",
        "gap_1_3",
        "gap_1_3",
        "gap_1_3",
        "observed",
        "gap_4_15",
        "gap_4_15",
        "gap_4_15",
        "gap_4_15",
    ]


def test_long_gap_floor_and_no_finite_fallback_are_exact() -> None:
    module = load_module(TRAIN_SOURCE, "exp308_floor")
    raw = np.r_[1.0, np.full(40, np.nan), 2.0]
    audit = module.build_missing_distance_confidence(raw)
    assert audit["confidence_weight"].min() == 0.25
    assert set(audit["gap_bucket"][1:-1]) == {"gap_16_plus"}
    fallback = module.build_missing_distance_confidence(np.full(7, np.nan))
    assert fallback["no_finite_gr_fallback"] is True
    assert np.array_equal(fallback["nearest_finite_row_distance"], np.full(7, -1))
    assert np.array_equal(fallback["confidence_weight"], np.full(7, 0.25))
    leading = module.build_missing_distance_confidence(np.array([np.nan, np.nan, 1.0]))
    assert leading["nearest_finite_row_distance"].tolist() == [2, 1, 0]


def test_parent_interpolation_parity_preserves_values_and_parent_formula() -> None:
    module = load_module(TRAIN_SOURCE, "exp308_interpolation")
    raw = np.array([np.nan, 10.0, np.nan, 16.0, np.nan])
    actual = module.parent_interpolated_gr(raw, np.array([30.0, 40.0]))
    expected = (
        pd.Series(raw).interpolate(limit_direction="both").fillna(35.0).to_numpy(np.float64)
    )
    assert np.array_equal(actual, expected)
    assert actual[[1, 3]].tolist() == [10.0, 16.0]


def test_weighted_emission_changes_only_row_precision() -> None:
    module = load_module(TRAIN_SOURCE, "exp308_emission")
    observed = np.array([10.0, 20.0])
    states = np.array([5.0, 15.0, 25.0])
    base = module.build_weighted_gaussian_emission(
        observed, states, 10.0, np.ones(2), 600.0
    )
    weighted = module.build_weighted_gaussian_emission(
        observed, states, 10.0, np.array([1.0, 0.25]), 600.0
    )
    assert np.array_equal(weighted[0], base[0])
    assert np.array_equal(weighted[1], np.float32(0.25) * base[1])


def test_synthetic_weighted_exact_hmm_is_finite_and_normalized() -> None:
    module = load_module(TRAIN_SOURCE, "exp308_hmm")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    rows = 28
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float) * 10.0,
            "Z": np.arange(rows, dtype=float) * 0.25,
            "GR": np.r_[np.linspace(40.0, 58.0, 24), [59.0, np.nan, np.nan, 62.0]],
            "TVT_input": np.r_[100.0 + np.arange(24, dtype=float), [np.nan] * 4],
        }
    )
    typewell = pd.DataFrame(
        {"TVT": np.linspace(80.0, 150.0, 141), "GR": np.linspace(35.0, 65.0, 141)}
    )
    prepared = module.prepare_hmm_inputs(horizontal, typewell, config)
    result = module.run_exact_hmm_variant(prepared, 20.0, config)
    assert result["mean"].shape == (4,)
    assert np.isfinite(result["mean"]).all()
    assert np.isfinite(result["std"]).all()
    assert result["posterior_row_sum_max_abs_error"] < 1e-6


def test_horizontal_loader_never_reads_truth(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp308_truth_free")
    pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "Z": [0.0, 0.1],
            "GR": [50.0, np.nan],
            "TVT_input": [100.0, np.nan],
            "TVT": [100.0, 101.0],
        }
    ).to_csv(tmp_path / "a__horizontal_well.csv", index=False)
    frame = module.load_horizontal_without_truth("a", tmp_path)
    assert list(frame.columns) == ["MD", "Z", "GR", "TVT_input"]
    assert "TVT" not in frame.columns


def test_parent_dependency_preflight_passes_then_fails_closed(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp308_parent")
    config = deepcopy(module.read_yaml(EXP_DIR / "config.yaml"))
    config["validation"].update({"expected_rows": 4, "expected_wells": 2})
    config["data"]["expected_raw_well_identity_sha256"] = "r" * 64
    spec = config["data"]["parent_exp307"]
    spec["candidates"] = [str(tmp_path)]
    prediction = pd.DataFrame(
        {
            "id": ["a_0", "a_1", "b_0", "b_1"],
            "well_id": ["a", "a", "b", "b"],
            "row_idx": [0, 1, 0, 1],
            "finite_mad_primary_hmm_tvt": [1.0, 2.0, 3.0, 4.0],
        }
    )
    scale = pd.DataFrame({"well_id": ["a", "b"], "finite_mad": [20.0, 30.0]})
    prediction_path = tmp_path / spec["prediction_filename"]
    scale_path = tmp_path / spec["scale_audit_filename"]
    prediction.to_csv(prediction_path, index=False, compression="gzip")
    scale.to_csv(scale_path, index=False, compression="gzip")
    contract = {"scientific_contract_sha256": "c" * 64}
    manifest = {"raw_train": {"content_sha256": "r" * 64}}
    promotion = {
        "passed": True,
        "primary_direct_gate": {"candidate_rmse": 1.0},
        "fixed_likpf_50_50_guard": {"candidate_rmse": 0.9},
    }
    contract_path = tmp_path / spec["scientific_contract_filename"]
    manifest_path = tmp_path / spec["input_control_manifest_filename"]
    promotion_path = tmp_path / spec["promotion_gate_filename"]
    contract_path.write_text(json.dumps(contract))
    manifest_path.write_text(json.dumps(manifest))
    promotion_path.write_text(json.dumps(promotion))
    spec.update(
        {
            "expected_scientific_contract_sha256": "c" * 64,
            "expected_input_control_manifest_sha256": module.sha256_path(manifest_path),
            "expected_prediction_decompressed_sha256": module.inspect_gzip_csv(prediction_path)[
                "decompressed_sha256"
            ],
            "expected_scale_audit_decompressed_sha256": module.inspect_gzip_csv(scale_path)[
                "decompressed_sha256"
            ],
            "expected_promotion_gate_sha256": module.sha256_path(promotion_path),
        }
    )
    config["references"]["parent_exp307_prediction_decompressed_sha256"] = spec[
        "expected_prediction_decompressed_sha256"
    ]
    config["references"]["parent_exp307_rmse"] = 1.0
    config["references"]["parent_exp307_likpf_50_50_rmse"] = 0.9
    summary = {
        "status": spec["required_status"],
        "promotion_gate": {"passed": True},
        "scientific_contract_sha256": "c" * 64,
        "input_control_manifest_sha256": module.sha256_path(manifest_path),
    }
    (tmp_path / spec["summary_filename"]).write_text(json.dumps(summary))
    report = module.preflight_parent_exp307(config)
    assert report["checks"] and all(report["checks"].values())
    promotion_path.write_text(
        json.dumps(
            {
                "passed": False,
                "primary_direct_gate": {"candidate_rmse": 1.0},
                "fixed_likpf_50_50_guard": {"candidate_rmse": 0.9},
            }
        )
    )
    spec["expected_promotion_gate_sha256"] = module.sha256_path(promotion_path)
    with pytest.raises(RuntimeError, match="dependency failed closed"):
        module.preflight_parent_exp307(config)


def test_truth_attachment_requires_prediction_and_weight_freeze() -> None:
    module = load_module(TRAIN_SOURCE, "exp308_freeze")
    with pytest.raises(RuntimeError, match="frozen prediction"):
        module._require_frozen_prediction({})
    module._require_frozen_prediction(
        {
            "frozen_before_truth_attachment": True,
            "decompressed_sha256": "a" * 64,
            "content_sha256": "a" * 64,
        }
    )


def test_promotion_gate_uses_parent_and_all_frozen_guards() -> None:
    module = load_module(TRAIN_SOURCE, "exp308_gate")
    config = deepcopy(module.read_yaml(EXP_DIR / "config.yaml"))
    config["validation"].update({"expected_rows": 12, "expected_wells": 3})
    config["model"]["execution_counts"]["hmm_well_runs"] = 3
    config["references"].update(
        {
            "parent_exp307_rmse": 1.0,
            "likpf_rmse": 1.0,
            "parent_exp307_likpf_50_50_rmse": 1.0,
        }
    )
    variant = "missing_distance_half8_floor025"
    missing = np.array([False, True, False, True] * 3)
    distance = np.array([0, 1, 0, 2] * 3)
    buckets = np.array(
        ["observed", "gap_1_3", "observed", "gap_4_15"] * 2
        + ["observed", "gap_16_plus", "observed", "gap_1_3"]
    )
    frame = pd.DataFrame(
        {
            "well_id": np.repeat(["a", "b", "c"], 4),
            "fold": np.arange(12) % 5,
            "true_tvt": np.zeros(12),
            "md_since": np.full(12, 1200.0),
            "parent_hmm_tvt": np.ones(12),
            "likpf_mean": np.ones(12),
            "parent_hmm_likpf_50_50": np.ones(12),
            f"{variant}_hmm_tvt": np.full(12, 0.8),
            f"{variant}_likpf_50_50": np.full(12, 0.9),
            "hidden_like_spatial": np.ones(12, dtype=bool),
            "hidden_like_typewell_purged": np.ones(12, dtype=bool),
            "raw_gr_finite": ~missing,
            "raw_gr_missing": missing,
            "nearest_finite_row_distance": distance,
            "confidence_weight": np.where(missing, np.exp2(-distance / 8.0), 1.0),
            "gap_bucket": buckets,
        }
    )
    paired, by_well = module.build_paired_metrics(frame, config)
    runtime = pd.DataFrame(
        {
            "well_id": ["a", "b", "c"],
            "variant": [variant] * 3,
            "posterior_row_sum_max_abs_error": np.zeros(3),
        }
    )
    weight_audit = frame[
        [
            "well_id",
            "raw_gr_finite",
            "raw_gr_missing",
            "nearest_finite_row_distance",
            "confidence_weight",
        ]
    ].copy()
    preflight = {
        "raw_train": {"content_sha256": "a" * 64},
        "controls": {"parent_exp307": {"checks": {"all_parent_checks": True}}},
    }
    gate = module.evaluate_promotion_gate(
        paired, by_well, frame, runtime, weight_audit, preflight, 1.0, config
    )
    assert gate["passed"] is True
    assert gate["primary_direct_gate"]["folds_improved"] == 5
    assert gate["technical_gate"]["gap_bucket_readout_complete"] is True
    assert gate["fixed_likpf_50_50_guard"]["passed"] is True


def test_inference_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp308_inference")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    contract = module.validate_disabled_inference(config)
    assert contract["inference_enabled"] is False
    assert contract["inference_create_submission"] is False
    assert contract["execution_create_submission"] is False
    with pytest.raises(RuntimeError, match="inference and submission are disabled"):
        module.stop_disabled_inference(config)


def test_sources_are_self_contained_notebook_safe_and_not_thin() -> None:
    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert "def build_missing_distance_confidence" in train_text
    assert "def build_weighted_gaussian_emission" in train_text
    assert "def _hmm2_fb" in train_text
    assert "def preflight_parent_exp307" in train_text
    assert "def evaluate_promotion_gate" in train_text
    assert "Path(__file__)" not in train_text
    assert "Path(__file__)" not in inference_text
    assert "submission.csv" not in train_text
    assert train_text.count("# %% [markdown]") >= 10

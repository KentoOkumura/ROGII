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
EXP_DIR = ROOT / "experiments" / "exp305_tempered_raw_smoothed_exact_hmm_emission"
TRAIN_SOURCE = (
    EXP_DIR / "exp305_tempered_raw_smoothed_exact_hmm_emission_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR / "exp305_tempered_raw_smoothed_exact_hmm_emission_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP305_IMPORT_ONLY")
    os.environ["EXP305_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP305_IMPORT_ONLY", None)
        else:
            os.environ["EXP305_IMPORT_ONLY"] = previous


def test_frozen_contract_has_one_variant_and_zero_training_cost() -> None:
    module = load_module(TRAIN_SOURCE, "exp305_contract")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    module.validate_scientific_contract(config)
    assert module.get_nested(config, "model.active_scientific_variants") == [
        "tempered_raw_swt_beta015"
    ]
    assert module.get_nested(config, "model.hmm.hmm_well_runs") == 773
    assert module.get_nested(config, "model.lightgbm_config_count") == 0
    assert module.get_nested(config, "model.fold_training_count") == 0
    assert module.get_nested(config, "model.booster_count") == 0
    assert module.get_nested(config, "model.parent_control_retraining") is False
    assert module.get_nested(config, "execution.kaggle_push_approved") is True
    assert module.get_nested(config, "execution.run_train") is True
    assert module.get_nested(config, "execution.run_inference") is False
    assert module.get_nested(config, "execution.create_submission") is False


def test_tempered_emission_is_exact_fixed_log_likelihood_mixture() -> None:
    module = load_module(TRAIN_SOURCE, "exp305_emission")
    observed_raw = np.array([10.0, 12.0])
    state_raw = np.array([9.0, 11.0, 14.0])
    observed_swt = np.array([10.5, 11.5])
    state_swt = np.array([9.5, 11.0, 13.0])
    tempered, raw, swt = module.build_tempered_emission(
        observed_raw,
        state_raw,
        observed_swt,
        state_swt,
        2.0,
    )
    np.testing.assert_allclose(tempered, 0.85 * raw + 0.15 * swt, rtol=0.0, atol=1e-7)
    with pytest.raises(ValueError, match="only raw_weight"):
        module.build_tempered_emission(
            observed_raw,
            state_raw,
            observed_swt,
            state_swt,
            2.0,
            raw_weight=0.80,
            swt_weight=0.20,
        )


def test_exp304_preflight_reads_silent_fallback_from_denoised_manifest(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp305_exp304_preflight")
    series_path = tmp_path / "series.csv.gz"
    pd.DataFrame(
        {
            "series_kind": ["horizontal", "typewell"],
            "well_id": ["a", "a"],
            "position": [0, 0],
            "coordinate": [0.0, 0.0],
            "original_missing": [False, False],
            "raw_gr": [1.0, 1.0],
            "swt_db4_l3_gr": [1.0, 1.0],
        }
    ).to_csv(
        series_path,
        index=False,
        compression="gzip",
    )
    series_report = module.inspect_gzip_csv(series_path)
    scientific_sha = "a" * 64
    raw_identity_sha = "b" * 64
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "series_artifact": {
                    "raw_sha256": series_report["raw_sha256"],
                    "decompressed_sha256": series_report["decompressed_sha256"],
                    "content_sha256": series_report["decompressed_sha256"],
                },
                "silent_fallback_count": 0,
            }
        )
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "selected_denoiser": "swt_db4_l3",
                "technical_gate": {"passed": True},
            }
        )
    )
    (tmp_path / "scientific_contract.json").write_text(
        json.dumps({"scientific_contract_sha256": scientific_sha})
    )
    (tmp_path / "input_manifest.json").write_text(
        json.dumps(
            {
                "raw_train": {
                    "well_file_identity_content_sha256": raw_identity_sha,
                }
            }
        )
    )
    config = {
        "data": {
            "exp304_selected_series": {
                "candidates": [str(tmp_path)],
                "filename": series_path.name,
                "manifest_filename": "manifest.json",
                "summary_filename": "summary.json",
                "scientific_contract_filename": "scientific_contract.json",
                "input_manifest_filename": "input_manifest.json",
                "require_nonzero_file_size": True,
                "expected_data_rows": 2,
                "expected_content_sha256": series_report["decompressed_sha256"],
                "require_manifest_raw_decompressed_content_sha_match": True,
                "expected_selected_denoiser": "swt_db4_l3",
                "expected_silent_fallback_count": 0,
                "expected_scientific_contract_sha256": scientific_sha,
                "expected_raw_well_identity_sha256": raw_identity_sha,
            }
        }
    }
    report = module.preflight_exp304(config)
    assert report["silent_fallback_count"] == 0
    assert report["series"]["columns"] == [
        "series_kind",
        "well_id",
        "position",
        "coordinate",
        "original_missing",
        "raw_gr",
        "swt_db4_l3_gr",
    ]


def test_saved_likpf_delta_is_reconstructed_as_absolute_tvt() -> None:
    module = load_module(TRAIN_SOURCE, "exp305_likpf_delta")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    control_spec = module.get_nested(config, "data.saved_controls")
    exp072 = pd.DataFrame(
        {
            "last_known_tvt": [100.0, 200.0],
            "likpf_mean_d": [1.5, -2.0],
        }
    )
    observed = module.materialize_saved_likpf_tvt(exp072, control_spec)
    np.testing.assert_allclose(observed, [101.5, 198.0], rtol=0.0, atol=0.0)
    invalid = deepcopy(control_spec)
    invalid["likpf_prediction_representation"] = "unknown"
    with pytest.raises(ValueError, match="unsupported saved likPF representation"):
        module.materialize_saved_likpf_tvt(exp072, invalid)


def test_synthetic_tempered_hmm_returns_normalized_finite_posterior_mean() -> None:
    module = load_module(TRAIN_SOURCE, "exp305_synthetic_hmm")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(8, dtype=float) * 10.0,
            "Z": np.arange(8, dtype=float) * 0.25,
            "GR": np.linspace(40.0, 55.0, 8),
            "TVT_input": [100.0, 101.0, 102.0, 103.0, np.nan, np.nan, np.nan, np.nan],
        }
    )
    horizontal_series = pd.DataFrame(
        {
            "raw_gr": np.linspace(40.0, 55.0, 8),
            "swt_db4_l3_gr": np.linspace(40.5, 54.5, 8),
        }
    )
    typewell_series = pd.DataFrame(
        {
            "coordinate": np.linspace(80.0, 130.0, 101),
            "raw_gr": np.linspace(35.0, 60.0, 101),
            "swt_db4_l3_gr": np.linspace(35.5, 59.5, 101),
        }
    )
    prepared = module.prepare_tempered_hmm_inputs(
        horizontal,
        horizontal_series,
        typewell_series,
        config,
    )
    result = module.run_tempered_hmm(prepared, config)
    assert result["mean"].shape == (4,)
    assert np.isfinite(result["mean"]).all()
    assert np.isfinite(result["std"]).all()
    assert result["posterior_row_sum_max_abs_error"] < 1e-6


def test_truth_attachment_requires_prediction_content_sha_freeze() -> None:
    module = load_module(TRAIN_SOURCE, "exp305_freeze")
    with pytest.raises(RuntimeError, match="frozen prediction"):
        module._require_frozen_prediction({})
    module._require_frozen_prediction({"decompressed_sha256": "a" * 64, "content_sha256": "a" * 64})


def test_fixed_paired_gate_requires_direct_and_blend_improvement() -> None:
    module = load_module(TRAIN_SOURCE, "exp305_gate")
    config = deepcopy(module.read_yaml(EXP_DIR / "config.yaml"))
    config["validation"]["expected_rows"] = 10
    config["validation"]["expected_wells"] = 2
    config["model"]["hmm"]["hmm_well_runs"] = 2
    frame = pd.DataFrame(
        {
            "well_id": ["a"] * 5 + ["b"] * 5,
            "fold": np.arange(10) % 5,
            "true_tvt": np.zeros(10),
            "md_since": np.full(10, 1200.0),
            "tempered_hmm_tvt": np.full(10, 0.8),
            "raw_hmm_tvt": np.ones(10),
            "likpf_mean": np.ones(10),
            "hidden_like_spatial": np.ones(10, dtype=bool),
            "hidden_like_typewell_purged": np.ones(10, dtype=bool),
        }
    )
    frame["tempered_likpf_50_50"] = 0.5 * (frame["tempered_hmm_tvt"] + frame["likpf_mean"])
    frame["raw_hmm_likpf_50_50"] = 0.5 * (frame["raw_hmm_tvt"] + frame["likpf_mean"])
    config["audit"]["saved_baselines"].update(
        {
            "raw_hmm_rmse": 1.0,
            "saved_likpf_rmse": 1.0,
            "raw_hmm_likpf_50_50_rmse": 1.0,
        }
    )
    paired, by_well = module.build_paired_metrics(frame, config)
    by_well_runtime = pd.DataFrame({"silent_fallback_count": [0, 0]})
    preflight = {
        "exp304": {
            "selected_denoiser": "swt_db4_l3",
            "silent_fallback_count": 0,
        }
    }
    gate = module.evaluate_promotion_gate(
        paired,
        by_well,
        frame,
        by_well_runtime,
        preflight,
        1.0,
        config,
    )
    assert gate["passed"] is True
    assert gate["comparison_gates"]["direct"]["folds_improved"] == 5
    assert gate["comparison_gates"]["blend"]["folds_improved"] == 5


def test_inference_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp305_inference")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    contract = module.validate_disabled_inference(config)
    assert contract["inference_enabled"] is False
    assert contract["inference_create_submission"] is False
    assert contract["execution_create_submission"] is False
    with pytest.raises(RuntimeError, match="inference and submission are disabled"):
        module.stop_disabled_inference(config)


def test_sources_are_not_thin_entrypoints_and_are_not_file_dependent() -> None:
    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert "def build_tempered_emission" in train_text
    assert "def _hmm2_fb" in train_text
    assert "def generate_and_freeze_predictions" in train_text
    assert "def evaluate_promotion_gate" in train_text
    assert "Path(__file__)" not in train_text
    assert "Path(__file__)" not in inference_text
    assert "submission.csv" not in train_text

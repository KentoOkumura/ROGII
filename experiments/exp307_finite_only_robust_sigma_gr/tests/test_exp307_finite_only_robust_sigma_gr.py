from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp307_finite_only_robust_sigma_gr"
TRAIN_SOURCE = EXP_DIR / "exp307_finite_only_robust_sigma_gr_compact_selfcontained_train.py"
INFERENCE_SOURCE = EXP_DIR / "exp307_finite_only_robust_sigma_gr_compact_selfcontained_inference.py"


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP307_IMPORT_ONLY")
    os.environ["EXP307_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP307_IMPORT_ONLY", None)
        else:
            os.environ["EXP307_IMPORT_ONLY"] = previous


def test_frozen_contract_records_two_variants_and_zero_training_cost() -> None:
    module = load_module(TRAIN_SOURCE, "exp307_contract")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    module.validate_scientific_contract(config)
    assert module.get_nested(config, "model.active_variants") == [
        "finite_std_diagnostic",
        "finite_mad_primary",
    ]
    counts = module.get_nested(config, "model.execution_counts")
    assert counts == {
        "variants": 2,
        "hmm_well_runs": 1546,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "boosters": 0,
        "parent_control_retraining": False,
    }
    assert module.get_nested(config, "execution.implementation_approved") is True
    assert module.get_nested(config, "execution.kaggle_push_approved") is True
    assert module.get_nested(config, "execution.run_scale_audit") is True
    assert module.get_nested(config, "execution.run_hmm") is True
    module.validate_scientific_contract(config, require_run_approval=True)
    broken = deepcopy(config)
    broken["model"]["fixed_hmm"]["output"] = "posterior_mode"
    with pytest.raises(ValueError, match="fixed_hmm.output"):
        module.validate_scientific_contract(broken)


def test_finite_scale_excludes_missing_gr_and_matches_std_and_mad() -> None:
    module = load_module(TRAIN_SOURCE, "exp307_scale")
    known_count = 25
    tvt_input = np.r_[np.arange(known_count, dtype=float), [np.nan, np.nan]]
    typewell_tvt = np.arange(40, dtype=float)
    typewell_gr = 50.0 + typewell_tvt
    residual = np.linspace(-4.0, 4.0, known_count)
    horizontal_gr = np.r_[50.0 + np.arange(known_count) + residual, [70.0, 71.0]]
    horizontal_gr[3] = np.nan
    frame = pd.DataFrame({"TVT_input": tvt_input, "GR": horizontal_gr})
    audit = module.compute_prefix_scale_audit(frame, typewell_tvt, typewell_gr)
    expected = np.delete(residual, 3)
    assert audit["finite_pair_count"] == 24
    assert audit["missing_known_gr_count"] == 1
    assert audit["finite_std_raw"] == pytest.approx(np.std(expected, ddof=0))
    expected_mad = 1.4826 * np.median(np.abs(expected - np.median(expected)))
    assert audit["finite_mad_raw"] == pytest.approx(expected_mad)
    assert audit["finite_std"] == 10.0
    assert audit["finite_mad"] == 10.0
    assert audit["current_zero_fill_std_raw"] != pytest.approx(audit["finite_std_raw"])


def test_scale_fallback_and_clip_are_fail_closed() -> None:
    module = load_module(TRAIN_SOURCE, "exp307_scale_guards")
    short = pd.DataFrame(
        {
            "TVT_input": np.r_[np.arange(10, dtype=float), np.nan],
            "GR": np.r_[np.linspace(-1000.0, 1000.0, 10), 0.0],
        }
    )
    typewell_tvt = np.arange(20, dtype=float)
    typewell_gr = np.zeros(20)
    fallback = module.compute_prefix_scale_audit(short, typewell_tvt, typewell_gr)
    assert fallback["finite_pair_count"] == 10
    assert fallback["finite_std"] == 30.0
    assert fallback["finite_mad"] == 30.0
    assert fallback["finite_std_fallback"] is True
    assert fallback["finite_mad_fallback"] is True

    enough = pd.DataFrame(
        {
            "TVT_input": np.r_[np.arange(25, dtype=float), np.nan],
            "GR": np.r_[np.linspace(-1000.0, 1000.0, 25), 0.0],
        }
    )
    clipped = module.compute_prefix_scale_audit(enough, np.arange(40, dtype=float), np.zeros(40))
    assert clipped["finite_std"] == 60.0
    assert clipped["finite_mad"] == 60.0
    assert clipped["finite_std_clip_high"] is True
    assert clipped["finite_mad_clip_high"] is True


def test_horizontal_loader_never_reads_truth(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp307_truth_free_loader")
    pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "Z": [0.0, 0.1],
            "GR": [50.0, 51.0],
            "TVT_input": [100.0, np.nan],
            "TVT": [100.0, 101.0],
        }
    ).to_csv(tmp_path / "a__horizontal_well.csv", index=False)
    frame = module.load_horizontal_without_truth("a", tmp_path)
    assert list(frame.columns) == ["MD", "Z", "GR", "TVT_input"]
    assert "TVT" not in frame.columns


def test_synthetic_exact_hmm_returns_finite_normalized_posterior_mean() -> None:
    module = load_module(TRAIN_SOURCE, "exp307_synthetic_hmm")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    rows = 28
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float) * 10.0,
            "Z": np.arange(rows, dtype=float) * 0.25,
            "GR": np.linspace(40.0, 60.0, rows),
            "TVT_input": np.r_[100.0 + np.arange(24, dtype=float), [np.nan] * 4],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(80.0, 150.0, 141),
            "GR": np.linspace(35.0, 65.0, 141),
        }
    )
    prepared = module.prepare_hmm_inputs(horizontal, typewell, config)
    result = module.run_exact_hmm_variant(prepared, prepared["scale_audit"]["finite_mad"], config)
    assert result["mean"].shape == (4,)
    assert np.isfinite(result["mean"]).all()
    assert np.isfinite(result["std"]).all()
    assert result["posterior_row_sum_max_abs_error"] < 1e-6


def test_truth_attachment_requires_prediction_and_scale_freeze() -> None:
    module = load_module(TRAIN_SOURCE, "exp307_freeze")
    with pytest.raises(RuntimeError, match="frozen prediction"):
        module._require_frozen_prediction({})
    module._require_frozen_prediction(
        {
            "frozen_before_truth_attachment": True,
            "decompressed_sha256": "a" * 64,
            "content_sha256": "a" * 64,
        }
    )


def test_control_and_assignment_preflight_checks_all_frozen_sha(
    tmp_path: Path,
) -> None:
    module = load_module(TRAIN_SOURCE, "exp307_preflight")
    config = deepcopy(module.read_yaml(EXP_DIR / "config.yaml"))
    wells = ["a", "b"]
    identities = [(well, row_idx) for well in wells for row_idx in range(5)]
    saved_hmm_path = tmp_path / "saved_hmm.csv.gz"
    saved_exp072_path = tmp_path / "saved_exp072.csv.gz"
    fold_path = tmp_path / "fold.csv.gz"
    hidden_path = tmp_path / "hidden.csv"
    pd.DataFrame(
        {
            "id": [f"{well}_{row_idx}" for well, row_idx in identities],
            "well": [well for well, _ in identities],
            "hmm_mean_tvt": np.ones(10),
        }
    ).to_csv(saved_hmm_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "id": [f"{well}_{row_idx}" for well, row_idx in identities],
            "well": [well for well, _ in identities],
            "md_since": np.arange(10, dtype=float),
            "last_known_tvt": np.linspace(100.0, 200.0, 10),
            "likpf_mean_d": np.ones(10),
        }
    ).to_csv(saved_exp072_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "well_id": [well for well, _ in identities],
            "row_idx": [row_idx for _, row_idx in identities],
            "suffix_offset": [row_idx for _, row_idx in identities],
            "fold": np.arange(10) % 5,
            "tvt_geop": np.zeros(10),
            "tvt_true": np.zeros(10),
        }
    ).to_csv(fold_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "well_id": wells,
            "verification_like_spatial_role": ["valid", "train"],
            "verification_like_typewell_purged_role": ["train", "valid"],
        }
    ).to_csv(hidden_path, index=False)
    config["validation"].update({"expected_rows": 10, "expected_wells": 2})
    config["data"]["saved_controls"].update(
        {
            "hmm_cache_filename": saved_hmm_path.name,
            "exp072_cache_filename": saved_exp072_path.name,
            "candidates": [str(tmp_path)],
        }
    )
    config["references"].update(
        {
            "exact_hmm_prediction_decompressed_sha256": module.inspect_gzip_csv(saved_hmm_path)[
                "decompressed_sha256"
            ],
            "exp209_feature_cache_decompressed_sha256": module.inspect_gzip_csv(saved_exp072_path)[
                "decompressed_sha256"
            ],
        }
    )
    config["data"]["fold_assignment"].update(
        {
            "filename": fold_path.name,
            "candidates": [str(tmp_path)],
            "expected_decompressed_sha256": module.inspect_gzip_csv(fold_path)[
                "decompressed_sha256"
            ],
        }
    )
    config["data"]["hidden_like_assignment"].update(
        {
            "filename": hidden_path.name,
            "candidates": [str(tmp_path)],
            "expected_sha256": module.sha256_path(hidden_path),
        }
    )
    report = module.preflight_controls_and_assignments(config)
    assert report["fold_assignment"]["well_ids"] == wells
    assert "likpf_mean_d" in report["saved_exp072"]["columns"]
    assert "likpf_mean" not in report["saved_exp072"]["columns"]
    broken = deepcopy(config)
    broken["references"]["exact_hmm_prediction_decompressed_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="saved exp209 HMM"):
        module.preflight_controls_and_assignments(broken)


def test_saved_likpf_delta_is_materialized_as_absolute_tvt() -> None:
    module = load_module(TRAIN_SOURCE, "exp307_likpf_materialization")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    control = module.get_nested(config, "data.saved_controls")
    frame = pd.DataFrame(
        {
            "last_known_tvt": [100.0, 250.5],
            "likpf_mean_d": [12.25, -3.0],
        }
    )
    observed = module.materialize_saved_likpf_tvt(frame, control)
    np.testing.assert_allclose(observed, [112.25, 247.5])


def test_primary_gate_uses_mad_and_blend_non_regression_only() -> None:
    module = load_module(TRAIN_SOURCE, "exp307_gate")
    config = deepcopy(module.read_yaml(EXP_DIR / "config.yaml"))
    config["validation"]["expected_rows"] = 10
    config["validation"]["expected_wells"] = 2
    config["model"]["execution_counts"]["hmm_well_runs"] = 4
    config["references"].update(
        {
            "exact_hmm_rmse": 1.0,
            "likpf_rmse": 1.0,
            "exact_hmm_likpf_50_50_rmse": 1.0,
        }
    )
    frame = pd.DataFrame(
        {
            "well_id": ["a"] * 5 + ["b"] * 5,
            "fold": np.arange(10) % 5,
            "true_tvt": np.zeros(10),
            "md_since": np.full(10, 1200.0),
            "raw_hmm_tvt": np.ones(10),
            "likpf_mean": np.ones(10),
            "raw_hmm_likpf_50_50": np.ones(10),
            "finite_std_diagnostic_hmm_tvt": np.full(10, 0.9),
            "finite_mad_primary_hmm_tvt": np.full(10, 0.8),
            "finite_std_diagnostic_likpf_50_50": np.full(10, 0.95),
            "finite_mad_primary_likpf_50_50": np.full(10, 0.9),
            "hidden_like_spatial": np.ones(10, dtype=bool),
            "hidden_like_typewell_purged": np.ones(10, dtype=bool),
        }
    )
    paired, by_well = module.build_paired_metrics(frame, config)
    runtime = pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "variant": [
                "finite_std_diagnostic",
                "finite_mad_primary",
                "finite_std_diagnostic",
                "finite_mad_primary",
            ],
            "posterior_row_sum_max_abs_error": np.zeros(4),
        }
    )
    scale_audit = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "current_zero_fill_std": [30.0, 30.0],
            "finite_std": [20.0, 20.0],
            "finite_mad": [15.0, 15.0],
        }
    )
    preflight = {"raw_train": {"content_sha256": "a" * 64}}
    gate = module.evaluate_promotion_gate(
        paired,
        by_well,
        frame,
        runtime,
        scale_audit,
        preflight,
        1.0,
        config,
    )
    assert gate["passed"] is True
    assert gate["primary_direct_gate"]["folds_improved"] == 5
    assert gate["fixed_likpf_50_50_guard"]["passed"] is True
    assert gate["diagnostic_can_promote"] is False


def test_inference_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp307_inference")
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
    assert "def compute_prefix_scale_audit" in train_text
    assert "def _hmm2_fb" in train_text
    assert "def generate_and_freeze_predictions" in train_text
    assert "def evaluate_promotion_gate" in train_text
    assert "from exact_hmm_smoother import" not in train_text
    assert "Path(__file__)" not in train_text
    assert "Path(__file__)" not in inference_text
    assert "submission.csv" not in train_text

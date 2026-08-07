from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp346_exp209_observed_only_finite_sigma_gr_hmm"
TRAIN_SOURCE = (
    EXP_DIR / "exp346_exp209_observed_only_finite_sigma_gr_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR / "exp346_exp209_observed_only_finite_sigma_gr_hmm_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP346_IMPORT_ONLY")
    os.environ["EXP346_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP346_IMPORT_ONLY", None)
        else:
            os.environ["EXP346_IMPORT_ONLY"] = previous


def test_frozen_contract_records_one_variant_and_closed_postrun_state() -> None:
    module = load_module(TRAIN_SOURCE, "exp346_contract")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    module.validate_scientific_contract(config)
    assert module.get_nested(config, "model.active_variants") == [
        "observed_finite_std_missing_exp209_sigma"
    ]
    assert module.get_nested(config, "execution_contract") == {
        "schedule_audits": 1,
        "scientific_variants": 1,
        "hmm_well_runs": 773,
        "model_configs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "boosters": 0,
        "parent_control_retraining": False,
    }
    assert module.get_nested(config, "execution.implementation_approved") is True
    assert module.get_nested(config, "execution.kaggle_push_approved") is False
    assert module.get_nested(config, "execution.run_scale_schedule_audit") is False
    assert module.get_nested(config, "execution.run_hmm") is False
    assert (
        module.get_nested(config, "experiment.status")
        == "train_side_observed_only_finite_sigma_gate_failed_closed"
    )
    with pytest.raises(RuntimeError, match="not approved"):
        module.validate_scientific_contract(config, require_run_approval=True)
    approved = deepcopy(config)
    approved["execution"]["kaggle_push_approved"] = True
    approved["execution"]["run_scale_schedule_audit"] = True
    approved["execution"]["run_hmm"] = True
    module.validate_scientific_contract(approved, require_run_approval=True)
    broken = deepcopy(config)
    broken["model"]["fixed_hmm"]["output"] = "posterior_mode"
    with pytest.raises(ValueError, match="fixed_hmm.output"):
        module.validate_scientific_contract(broken)


def _synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame]:
    known_count = 25
    eval_count = 5
    typewell_tvt = np.arange(80, dtype=float)
    typewell_gr = 50.0 + typewell_tvt
    residual = np.linspace(-30.0, 30.0, known_count)
    known_gr = 50.0 + np.arange(known_count, dtype=float) + residual
    known_gr[3] = np.nan
    eval_gr = np.array([78.0, np.nan, 80.0, np.nan, 82.0])
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(known_count + eval_count, dtype=float) * 10.0,
            "Z": np.arange(known_count + eval_count, dtype=float) * 0.2,
            "GR": np.r_[known_gr, eval_gr],
            "TVT_input": np.r_[100.0 + np.arange(known_count, dtype=float), [np.nan] * eval_count],
        }
    )
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})
    return horizontal, typewell


def test_scale_schedule_uses_observed_std_only_on_raw_finite_rows() -> None:
    module = load_module(TRAIN_SOURCE, "exp346_schedule")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    horizontal, typewell = _synthetic_well()
    audit = module.compute_prefix_scale_audit(
        horizontal,
        typewell["TVT"].to_numpy(float),
        typewell["GR"].to_numpy(float),
    )
    known = horizontal["TVT_input"].notna().to_numpy()
    typewell_at_known = np.interp(
        horizontal.loc[known, "TVT_input"], typewell["TVT"], typewell["GR"]
    )
    known_gr = horizontal.loc[known, "GR"].to_numpy(float)
    finite = np.isfinite(known_gr)
    expected_observed = np.std(known_gr[finite] - typewell_at_known[finite], ddof=0)
    expected_base = np.std(np.where(finite, known_gr, 0.0) - typewell_at_known, ddof=0)
    assert audit["finite_pair_count"] == 24
    assert audit["observed_finite_std_raw"] == pytest.approx(expected_observed)
    assert audit["exp209_zero_fill_sigma_raw"] == pytest.approx(expected_base)

    prepared = module.prepare_hmm_inputs(horizontal, typewell, config)
    np.testing.assert_array_equal(prepared["raw_gr_observed"], [True, False, True, False, True])
    expected_schedule = np.where(
        prepared["raw_gr_observed"],
        audit["observed_finite_std"],
        audit["exp209_zero_fill_sigma"],
    )
    np.testing.assert_allclose(prepared["sigma_schedule"], expected_schedule)
    assert prepared["evaluation_missing_fraction"] == pytest.approx(0.4)


def test_insufficient_finite_pairs_fall_back_to_exp209_sigma_for_whole_well() -> None:
    module = load_module(TRAIN_SOURCE, "exp346_fallback")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(14, dtype=float),
            "Z": np.arange(14, dtype=float) * 0.1,
            "GR": np.r_[np.linspace(-100.0, 100.0, 10), [1.0, np.nan, 2.0, np.nan]],
            "TVT_input": np.r_[np.arange(10, dtype=float), [np.nan] * 4],
        }
    )
    typewell = pd.DataFrame({"TVT": np.arange(30, dtype=float), "GR": np.zeros(30, dtype=float)})
    prepared = module.prepare_hmm_inputs(horizontal, typewell, config)
    audit = prepared["scale_audit"]
    assert audit["observed_finite_std_fallback"] is True
    assert audit["observed_finite_std"] == audit["exp209_zero_fill_sigma"]
    np.testing.assert_array_equal(
        prepared["sigma_schedule"],
        np.full(4, audit["exp209_zero_fill_sigma"]),
    )


def test_row_scheduled_exact_hmm_preserves_missing_emission_parity() -> None:
    module = load_module(TRAIN_SOURCE, "exp346_hmm")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    horizontal, typewell = _synthetic_well()
    prepared = module.prepare_hmm_inputs(horizontal, typewell, config)
    result = module.run_exact_hmm_variant(prepared, prepared["sigma_schedule"], config)
    assert result["mean"].shape == (5,)
    assert np.isfinite(result["mean"]).all()
    assert np.isfinite(result["std"]).all()
    assert result["posterior_row_sum_max_abs_error"] < 1.0e-6
    assert result["raw_missing_emission_max_abs_diff_vs_exp209"] == 0.0


def test_horizontal_loader_never_reads_unknown_suffix_truth(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp346_truth_free_loader")
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


def test_truth_attachment_requires_each_target_free_artifact_freeze() -> None:
    module = load_module(TRAIN_SOURCE, "exp346_freeze")
    with pytest.raises(RuntimeError, match="frozen prediction"):
        module._require_frozen_prediction({})
    module._require_frozen_prediction(
        {
            "frozen_before_truth_attachment": True,
            "decompressed_sha256": "a" * 64,
            "content_sha256": "a" * 64,
        }
    )


def test_control_and_assignment_preflight_checks_frozen_sha(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp346_preflight")
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
            "expected_hmm_prediction_decompressed_sha256": module.inspect_gzip_csv(saved_hmm_path)[
                "decompressed_sha256"
            ],
            "expected_exp072_cache_decompressed_sha256": module.inspect_gzip_csv(saved_exp072_path)[
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
    broken = deepcopy(config)
    broken["data"]["saved_controls"]["expected_hmm_prediction_decompressed_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="saved exp209 HMM"):
        module.preflight_controls_and_assignments(broken)


def test_saved_likpf_delta_is_materialized_as_absolute_tvt() -> None:
    module = load_module(TRAIN_SOURCE, "exp346_likpf")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    control = module.get_nested(config, "data.saved_controls")
    frame = pd.DataFrame({"last_known_tvt": [100.0, 250.5], "likpf_mean_d": [12.25, -3.0]})
    np.testing.assert_allclose(module.materialize_saved_likpf_tvt(frame, control), [112.25, 247.5])


def test_promotion_gate_requires_observed_gain_and_all_non_regression_scopes() -> None:
    module = load_module(TRAIN_SOURCE, "exp346_gate")
    config = deepcopy(module.read_yaml(EXP_DIR / "config.yaml"))
    wells = ["a", "b", "c", "d", "e"]
    raw_patterns = {
        "a": [True] * 5,
        "b": [True] * 5,
        "c": [False, True, True, True, True],
        "d": [False, False, True, True, True],
        "e": [False] * 5,
    }
    rows: list[dict[str, object]] = []
    for well in wells:
        pattern = raw_patterns[well]
        missing_fraction = 1.0 - float(np.mean(pattern))
        for fold, observed in enumerate(pattern):
            candidate = 0.8 if observed else 1.0
            rows.append(
                {
                    "well_id": well,
                    "fold": fold,
                    "true_tvt": 0.0,
                    "md_since": [100.0, 500.0, 1200.0, 1400.0, 1600.0][fold],
                    "raw_hmm_tvt": 1.0,
                    "likpf_mean": 1.0,
                    "raw_hmm_likpf_50_50": 1.0,
                    "raw_gr_observed": observed,
                    "evaluation_missing_fraction": missing_fraction,
                    "hidden_like_spatial": True,
                    "hidden_like_typewell_purged": True,
                    f"{module.VARIANT_ORDER[0]}_hmm_tvt": candidate,
                    f"{module.VARIANT_ORDER[0]}_likpf_50_50": 0.5 * candidate + 0.5,
                }
            )
    frame = pd.DataFrame(rows)
    config["validation"].update({"expected_rows": len(frame), "expected_wells": len(wells)})
    config["execution_contract"]["hmm_well_runs"] = len(wells)
    config["references"].update(
        {
            "exp209_raw_hmm_rmse": 1.0,
            "exp209_likpf_rmse": 1.0,
            "exp209_hmm_likpf_50_50_rmse": 1.0,
        }
    )
    paired, by_well = module.build_paired_metrics(frame, config)
    runtime = pd.DataFrame(
        {
            "well_id": wells,
            "variant": [module.VARIANT_ORDER[0]] * len(wells),
            "posterior_row_sum_max_abs_error": np.zeros(len(wells)),
            "raw_missing_emission_max_abs_diff_vs_exp209": np.zeros(len(wells)),
            "raw_observed_rows": [sum(raw_patterns[well]) for well in wells],
            "raw_missing_rows": [5 - sum(raw_patterns[well]) for well in wells],
        }
    )
    scale_audit = pd.DataFrame(
        {
            "well_id": wells,
            "exp209_zero_fill_sigma": np.full(len(wells), 30.0),
            "observed_finite_std": np.full(len(wells), 20.0),
            "observed_finite_std_fallback": np.zeros(len(wells), dtype=bool),
        }
    )
    preflight = {"raw_train": {"content_sha256": "a" * 64}}
    gate = module.evaluate_promotion_gate(
        paired, by_well, frame, runtime, scale_audit, preflight, 1.0, config
    )
    assert gate["passed"] is True
    assert gate["primary_direct_gate"]["folds_improved"] == 5
    assert gate["primary_direct_gate"]["raw_observed_improvement_ft"] >= 0.05
    assert all(gate["primary_direct_gate"]["required_non_regression_scopes"].values())
    assert gate["technical_gate"]["raw_missing_emission_schedule_parity_passed"] is True

    broken_frame = frame.copy()
    missing = ~broken_frame["raw_gr_observed"]
    broken_frame.loc[missing, f"{module.VARIANT_ORDER[0]}_hmm_tvt"] = 1.1
    broken_frame.loc[missing, f"{module.VARIANT_ORDER[0]}_likpf_50_50"] = 1.05
    broken_paired, broken_by_well = module.build_paired_metrics(broken_frame, config)
    broken_gate = module.evaluate_promotion_gate(
        broken_paired,
        broken_by_well,
        broken_frame,
        runtime,
        scale_audit,
        preflight,
        1.0,
        config,
    )
    assert broken_gate["passed"] is False
    assert (
        broken_gate["primary_direct_gate"]["required_non_regression_scopes"]["raw_gr_missing"]
        is False
    )


def test_inference_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp346_inference")
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
    assert "raw_mask_scale_schedule" in train_text
    assert "def evaluate_promotion_gate" in train_text
    assert "from exact_hmm_smoother import" not in train_text
    assert "Path(__file__)" not in train_text
    assert "Path(__file__)" not in inference_text
    assert "submission.csv" not in train_text

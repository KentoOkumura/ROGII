from __future__ import annotations

import importlib.util
import json
import math
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp338_exp209_well_adaptive_transition_noise"
TRAIN_SOURCE = EXP_DIR / (
    "exp338_exp209_well_adaptive_transition_noise_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp338_exp209_well_adaptive_transition_noise_compact_selfcontained_inference.py"
)
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    / "exact_hmm_smoother.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP338_IMPORT_ONLY")
    os.environ["EXP338_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP338_IMPORT_ONLY", None)
        else:
            os.environ["EXP338_IMPORT_ONLY"] = previous


def load_config(module):
    return module.read_yaml(EXP_DIR / "config.yaml")


def test_contract_records_one_approved_train_candidate_only() -> None:
    module = load_module(TRAIN_SOURCE, "exp338_contract")
    config = load_config(module)
    module.validate_scientific_contract(config)
    assert module.get_nested(config, "model.active_variants") == [
        "robust_prefix_rate_diffusion_on_exp209"
    ]
    assert module.get_nested(config, "execution_contract") == {
        "scientific_variants": 1,
        "hmm_well_runs": 773,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "boosters": 0,
        "parent_control_retraining": False,
    }
    assert module.get_nested(config, "execution.implementation_approved") is True
    assert module.get_nested(config, "execution.kaggle_push_approved") is True
    assert module.get_nested(config, "execution.run_transition_audit") is True
    assert module.get_nested(config, "execution.run_hmm") is True
    assert module.get_nested(config, "execution.run_inference") is False
    assert module.get_nested(config, "execution.create_submission") is False
    module.validate_scientific_contract(config, require_run_approval=True)

    unapproved = deepcopy(config)
    unapproved["execution"]["kaggle_push_approved"] = False
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        module.validate_scientific_contract(unapproved, require_run_approval=True)


def test_parent_metrics_contract_matches_the_kaggle_output_schema() -> None:
    module = load_module(TRAIN_SOURCE, "exp338_parent_metrics")
    config = load_config(module)
    saved = module.get_nested(config, "data.saved_exp209")
    expected_hmm_sha = saved["expected_hmm_decompressed_sha256"]
    expected_likpf_sha = saved["expected_likpf_decompressed_sha256"]
    parent_metrics = {
        "experiment": "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation",
        "status": "implemented_pending_kaggle_review",
        "rows": 3_783_989,
        "wells": 773,
        "metric_parity": {
            "checked": True,
            "actual_best_candidate": "blend_likpf_hmm_w500",
            "expected_best_candidate": "blend_likpf_hmm_w500",
            "best_candidate_matches": True,
            "best_rmse_abs_diff": 3.8105997788306922e-06,
            "best_rmse_matches": False,
        },
        "reference_parity": {
            "exp205_hmm_train_features": {
                "expected_decompressed_sha256": expected_hmm_sha,
                "generated_decompressed_sha256": expected_hmm_sha,
                "matches_expected_decompressed_sha256": True,
                "matches_reference_decompressed_sha256": True,
            },
            "exp072_train_features": {
                "generated_decompressed_sha256": expected_likpf_sha,
                "matches_reference_decompressed_sha256": False,
            },
        },
    }
    report = module.validate_parent_metrics_contract(
        parent_metrics,
        raw_sha256=saved["expected_metrics_sha256"],
        saved=saved,
        expected_rows=3_783_989,
        expected_wells=773,
    )
    assert report["hmm_exact_parity_passed"] is True
    assert report["metric_parity_accepted"] is True
    assert report["accepted_metric_abs_tolerance"] == pytest.approx(1.0e-5)

    excessive_difference = deepcopy(parent_metrics)
    excessive_difference["metric_parity"]["best_rmse_abs_diff"] = 1.1e-5
    with pytest.raises(ValueError, match="exact-parity dependency mismatch"):
        module.validate_parent_metrics_contract(
            excessive_difference,
            raw_sha256=saved["expected_metrics_sha256"],
            saved=saved,
            expected_rows=3_783_989,
            expected_wells=773,
        )


def test_hidden_like_role_contract_accepts_exp115_purged_exclusions() -> None:
    module = load_module(TRAIN_SOURCE, "exp338_hidden_roles")
    config = load_config(module)
    hidden = module.get_nested(config, "data.hidden_like_assignment")
    assignment_path = (
        ROOT
        / "experiments"
        / "exp115_hidden_like_spatial_holdout_from_ppt"
        / "artifacts"
        / hidden["filename"]
    )
    assert module.sha256_path(assignment_path) == hidden["expected_sha256"]
    role_columns = list(hidden["role_columns"].values())
    assignment = pd.read_csv(assignment_path, usecols=["well_id", *role_columns])
    report = module.validate_hidden_like_role_contract(assignment, hidden)
    assert report == {
        "hidden_like_spatial": {"train": 573, "valid": 200},
        "hidden_like_typewell_purged": {
            "purged_train_excluded": 16,
            "train": 557,
            "valid": 200,
        },
    }

    invalid = assignment.copy()
    invalid.loc[invalid.index[0], role_columns[0]] = "purged_train_excluded"
    with pytest.raises(ValueError, match="hidden_like_spatial"):
        module.validate_hidden_like_role_contract(invalid, hidden)


def test_transition_scale_matches_the_preregistered_formula_without_step_floor() -> None:
    module = load_module(TRAIN_SOURCE, "exp338_transition_formula")
    rows = 34
    step = 0.25
    md = np.arange(rows, dtype=float) * step
    rates = 0.02 + 0.0004 * np.square(np.arange(rows - 1, dtype=float))
    u = np.r_[100.0, 100.0 + np.cumsum(step * rates)]
    frame = pd.DataFrame({"MD": md, "Z": np.zeros(rows), "TVT_input": u})
    audit = module.compute_prefix_transition_scale_audit(frame)
    innovations = np.diff(rates) / math.sqrt(step)
    raw = 1.4826 * np.median(np.abs(innovations - np.median(innovations)))
    alpha = len(innovations) / (len(innovations) + 100.0)
    unclip = math.exp(alpha * math.log(max(raw, 1.0e-6)) + (1.0 - alpha) * math.log(0.002))
    assert audit["valid_rate_pairs"] == rows - 1
    assert audit["valid_rate_innovations"] == rows - 2
    assert audit["raw_sig_r"] == pytest.approx(raw)
    assert audit["shrinkage_alpha"] == pytest.approx(alpha)
    assert audit["unclipped_sig_r"] == pytest.approx(unclip)
    assert audit["sig_r"] == pytest.approx(np.clip(unclip, 0.001, 0.004))
    assert audit["sig_r_fallback"] is False


def test_transition_scale_fallback_and_clip_are_fixed() -> None:
    module = load_module(TRAIN_SOURCE, "exp338_transition_guards")
    short = pd.DataFrame(
        {
            "MD": np.arange(10, dtype=float),
            "Z": np.zeros(10),
            "TVT_input": np.arange(10, dtype=float),
        }
    )
    fallback = module.compute_prefix_transition_scale_audit(short)
    assert fallback["valid_rate_innovations"] == 8
    assert fallback["sig_r"] == 0.002
    assert fallback["sig_r_fallback"] is True

    rows = 40
    constant_rate = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float),
            "Z": np.zeros(rows),
            "TVT_input": np.arange(rows, dtype=float) * 0.03,
        }
    )
    clipped = module.compute_prefix_transition_scale_audit(constant_rate)
    assert clipped["valid_rate_innovations"] == rows - 2
    assert clipped["sig_r"] == 0.001
    assert clipped["sig_r_clip_low"] is True


def test_exp209_sigma_uses_zero_fill_population_std_and_unit_weight() -> None:
    module = load_module(TRAIN_SOURCE, "exp338_exp209_sigma")
    typewell_tvt = np.arange(40, dtype=float)
    typewell_gr = 50.0 + typewell_tvt
    tvt_input = np.r_[np.arange(25, dtype=float), [np.nan, np.nan]]
    observed = np.r_[50.0 + np.arange(25, dtype=float), [70.0, np.nan]]
    observed[3] = np.nan
    horizontal = pd.DataFrame({"TVT_input": tvt_input, "GR": observed})
    audit = module.compute_exp209_zero_fill_sigma_audit(horizontal, typewell_tvt, typewell_gr)
    expected_residual = np.zeros(25)
    expected_residual[3] = -(50.0 + 3.0)
    expected_raw = np.std(expected_residual, ddof=0)
    assert audit["sigma_gr_raw"] == pytest.approx(expected_raw)
    assert audit["sigma_gr"] == pytest.approx(np.clip(expected_raw, 10.0, 60.0))
    assert audit["missing_known_gr_rows"] == 1
    assert audit["observation_weight"] == 1.0
    assert audit["affine_a"] == 1.0 and audit["affine_b"] == 0.0


def test_candidate_reproduces_exp209_observation_and_state_grammar() -> None:
    module = load_module(TRAIN_SOURCE, "exp338_synthetic_hmm")
    parent = load_module(PARENT_SOURCE, "exp209_parent_exact_hmm")
    config = load_config(module)
    rows = 36
    known_rows = 30
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float) * 10.0,
            "Z": np.arange(rows, dtype=float) * 0.2,
            "GR": np.linspace(40.0, 62.0, rows),
            "TVT_input": np.r_[100.0 + np.arange(known_rows, dtype=float), [np.nan] * 6],
        }
    )
    horizontal.loc[[4, 32, 33], "GR"] = np.nan
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(80.0, 170.0, 181),
            "GR": np.linspace(35.0, 67.0, 181),
        }
    )
    prepared = module.prepare_hmm_inputs(horizontal, typewell, config)
    sig_r = float(prepared["transition_audit"]["sig_r"])
    candidate = module.run_exact_hmm_variant(
        prepared,
        float(prepared["sigma_audit"]["sigma_gr"]),
        sig_r,
        config,
    )
    parent_result = parent.run_hmm2(
        horizontal,
        typewell,
        step=0.35,
        n_rates=41,
        rate_span=0.10,
        sig_r=sig_r,
        sig_p=0.02,
        emission="gauss",
        lam=1.0,
        sigma_mode="std",
        start_sig=0.75,
        r0_sig=0.01,
        band_pad=100.0,
        mom=0.998,
        rate_center="zero",
        return_post=True,
    )
    np.testing.assert_allclose(candidate["mean"], parent_result["mean_eval"], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(candidate["std"], parent_result["std_eval"], rtol=0.0, atol=1e-10)
    assert prepared["sigma_audit"]["sigma_gr"] == pytest.approx(parent_result["prefix_sigma"])
    assert np.array_equal(prepared["observation_weight"], np.ones(6))
    assert candidate["posterior_row_sum_max_abs_error"] < 1.0e-6


def test_horizontal_loader_and_late_truth_boundary_are_fail_closed(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp338_truth_free")
    pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "Z": [0.0, 0.1],
            "GR": [50.0, np.nan],
            "TVT_input": [100.0, np.nan],
            "TVT": [100.0, 101.0],
            "error": [0.0, 99.0],
        }
    ).to_csv(tmp_path / "a__horizontal_well.csv", index=False)
    frame = module.load_horizontal_without_truth("a", tmp_path)
    assert list(frame.columns) == ["MD", "Z", "GR", "TVT_input"]
    with pytest.raises(RuntimeError, match="frozen prediction"):
        module._require_frozen_prediction({})


def test_exp209_dependency_preflight_checks_cache_and_metrics_sha(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp338_dependency")
    config = deepcopy(load_config(module))
    wells = ["a", "b"]
    identities = [(well, row_idx) for well in wells for row_idx in range(5)]
    hmm_path = tmp_path / "hmm.csv.gz"
    likpf_path = tmp_path / "likpf.csv.gz"
    fold_path = tmp_path / "fold.csv.gz"
    hidden_path = tmp_path / "hidden.csv"
    metrics_path = tmp_path / "metrics.json"
    pd.DataFrame(
        {
            "id": [f"{well}_{row}" for well, row in identities],
            "well": [well for well, _ in identities],
            "hmm_mean_tvt": np.ones(10),
        }
    ).to_csv(hmm_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "id": [f"{well}_{row}" for well, row in identities],
            "well": [well for well, _ in identities],
            "md_since": np.arange(10, dtype=float),
            "last_known_tvt": np.ones(10),
            "likpf_mean_d": np.zeros(10),
        }
    ).to_csv(likpf_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "well_id": [well for well, _ in identities],
            "row_idx": [row for _, row in identities],
            "suffix_offset": [row for _, row in identities],
            "fold": np.arange(10) % 2,
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
    hmm_sha = module.inspect_gzip_csv(hmm_path)["decompressed_sha256"]
    likpf_sha = module.inspect_gzip_csv(likpf_path)["decompressed_sha256"]
    metrics_path.write_text(
        json.dumps(
            {
                "experiment": "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation",
                "route": "pf_beam",
                "status": "completed",
                "rows": 10,
                "wells": 2,
                "metric_parity": {
                    "checked": True,
                    "actual_best_candidate": "blend_likpf_hmm_w500",
                    "expected_best_candidate": "blend_likpf_hmm_w500",
                    "best_candidate_matches": True,
                    "best_rmse_abs_diff": 3.0e-6,
                    "best_rmse_matches": False,
                },
                "reference_parity": {
                    "exp205_hmm_train_features": {
                        "expected_decompressed_sha256": hmm_sha,
                        "generated_decompressed_sha256": hmm_sha,
                        "matches_expected_decompressed_sha256": True,
                        "matches_reference_decompressed_sha256": True,
                    },
                    "exp072_train_features": {
                        "generated_decompressed_sha256": likpf_sha,
                        "matches_reference_decompressed_sha256": False,
                    },
                },
            }
        )
    )
    config["validation"].update(
        {"expected_rows": 10, "expected_wells": 2, "expected_folds": [0, 1]}
    )
    saved = config["data"]["saved_exp209"]
    saved.update(
        {
            "hmm_cache_filename": hmm_path.name,
            "likpf_cache_filename": likpf_path.name,
            "metrics_filename": metrics_path.name,
            "candidates": [str(tmp_path)],
            "metrics_candidates": [str(tmp_path)],
            "expected_hmm_decompressed_sha256": hmm_sha,
            "expected_likpf_decompressed_sha256": likpf_sha,
            "expected_metrics_sha256": module.sha256_path(metrics_path),
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
    assert report["parent_metrics"]["hmm_exact_parity_passed"] is True
    assert report["saved_hmm"]["decompressed_sha256"] == hmm_sha
    broken = deepcopy(config)
    broken["data"]["saved_exp209"]["expected_hmm_decompressed_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="HMM prediction decompressed SHA mismatch"):
        module.preflight_controls_and_assignments(broken)


def test_promotion_gate_requires_all_tail_and_transition_guards() -> None:
    module = load_module(TRAIN_SOURCE, "exp338_gate")
    config = deepcopy(load_config(module))
    config["validation"].update({"expected_rows": 10, "expected_wells": 2})
    config["execution_contract"]["hmm_well_runs"] = 2
    config["references"].update(
        {
            "exp209_raw_hmm_rmse": 1.0,
            "exp209_likpf_rmse": 1.0,
            "exp209_hmm_likpf_50_50_rmse": 1.0,
        }
    )
    scopes = [
        "overall",
        *[f"fold_{fold}" for fold in range(5)],
        "md_since_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ]
    paired = pd.DataFrame(
        [
            {
                "variant": module.VARIANT_ORDER[0],
                "comparison": comparison,
                "scope": scope,
                "candidate_rmse": 0.8 if comparison == "direct" else 0.9,
                "control_rmse": 1.0,
                "delta_rmse_candidate_minus_control": -0.2 if comparison == "direct" else -0.1,
                "improvement_ft": 0.2 if comparison == "direct" else 0.1,
            }
            for comparison in ("direct", "fixed_likpf_50_50")
            for scope in scopes
        ]
    )
    by_well = pd.DataFrame(
        [
            {
                "variant": module.VARIANT_ORDER[0],
                "comparison": comparison,
                "well_id": well,
                "candidate_rmse": 0.8 if comparison == "direct" else 0.9,
                "control_rmse": 1.0,
                "delta_rmse_candidate_minus_control": -0.2 if comparison == "direct" else -0.1,
            }
            for comparison in ("direct", "fixed_likpf_50_50")
            for well in ("a", "b")
        ]
    )
    frame = pd.DataFrame(
        {
            "well_id": ["a"] * 5 + ["b"] * 5,
            "true_tvt": np.zeros(10),
            "parent_hmm_tvt": np.ones(10),
            "likpf_mean": np.ones(10),
            f"{module.VARIANT_ORDER[0]}_hmm_tvt": np.full(10, 0.8),
            f"{module.VARIANT_ORDER[0]}_likpf_50_50": np.full(10, 0.9),
        }
    )
    runtime = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "variant": [module.VARIANT_ORDER[0]] * 2,
            "posterior_row_sum_max_abs_error": [0.0, 0.0],
        }
    )
    audit = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "exp209_sigma_gr": [20.0, 21.0],
            "exp209_observation_weight": [1.0, 1.0],
            "sig_r": [0.0018, 0.0022],
            "sig_r_fallback": [False, False],
            "sig_r_clip_low": [False, False],
            "sig_r_clip_high": [False, False],
        }
    )
    preflight = {
        "raw_train": {"content_sha256": "a" * 64},
        "controls": {
            "parent_metrics": {"hmm_exact_parity_passed": True},
            "saved_hmm": {"decompressed_sha256": "b" * 64},
        },
    }
    gate = module.evaluate_promotion_gate(
        paired, by_well, frame, runtime, audit, preflight, 1.0, config
    )
    assert gate["passed"] is True
    audit.loc[0, "sig_r_fallback"] = True
    failed = module.evaluate_promotion_gate(
        paired, by_well, frame, runtime, audit, preflight, 1.0, config
    )
    assert failed["passed"] is False
    assert failed["technical_gate"]["sig_r_fallback_fraction"] == 0.5


def test_inference_is_explicitly_disabled() -> None:
    module = load_module(INFERENCE_SOURCE, "exp338_inference")
    config = module.read_yaml(EXP_DIR / "config.yaml")
    contract = module.validate_disabled_inference(config)
    assert contract["mode"] == "disabled_until_train_side_promotion_and_separate_approval"
    with pytest.raises(RuntimeError, match="inference and submission are disabled"):
        module.stop_disabled_inference(config)


def test_sources_are_self_contained_notebook_safe_and_single_change() -> None:
    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert "def compute_exp209_zero_fill_sigma_audit" in train_text
    assert "def compute_prefix_transition_scale_audit" in train_text
    assert "def _hmm2_fb" in train_text
    assert "def compute_prefix_scale_audit" not in train_text
    assert "def compute_missing_gr_confidence" not in train_text
    assert "from settings import" not in train_text
    assert "Path(__file__)" not in train_text
    assert "Path(__file__)" not in inference_text

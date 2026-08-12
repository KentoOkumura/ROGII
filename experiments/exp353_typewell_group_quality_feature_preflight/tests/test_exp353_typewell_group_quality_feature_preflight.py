from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp353_typewell_group_quality_feature_preflight"
TRAIN_SOURCE = EXP_DIR / (
    "exp353_typewell_group_quality_feature_preflight_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp353_typewell_group_quality_feature_preflight_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    payload = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def train_module():
    return load_module(TRAIN_SOURCE, "exp353_train_contract")


def test_config_locks_zero_booster_stage_0_and_disabled_stage_1(train_module) -> None:
    config = load_config()
    train_module.validate_scientific_contract(config)
    train_module.validate_run_approval(config)
    assert config["experiment"]["route"] == "ml_model"
    assert config["execution_contract"]["stage_0"] == {
        "preflight_variants": 1,
        "negative_controls": 1,
        "folds": 5,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    assert config["execution_contract"]["stage_1_if_pass"]["boosters"] == 15
    assert config["execution"]["run_stage_1"] is False
    changed = load_config()
    changed["model"]["calibration"]["minimum_peer_wells"] = 2
    with pytest.raises(ValueError, match="minimum_peer_wells"):
        train_module.validate_scientific_contract(changed)
    unapproved = load_config()
    unapproved["execution"]["run_stage_0"] = False
    with pytest.raises(RuntimeError, match="disabled"):
        train_module.validate_run_approval(unapproved)
    stage_1 = load_config()
    stage_1["execution"]["run_stage_1"] = True
    with pytest.raises((ValueError, RuntimeError), match="Stage 1"):
        train_module.validate_run_approval(stage_1)


def test_fold_manifest_exactly_reconstructs_sklearn_groupkfold(train_module) -> None:
    row_counts = pd.DataFrame(
        {
            "well_id": ["w5", "w1", "w4", "w0", "w3", "w2", "w6"],
            "rows": [11, 9, 7, 13, 4, 6, 3],
        }
    )
    actual = train_module.build_exp148_fold_manifest(row_counts, 3)
    expanded_groups = np.concatenate(
        [
            np.repeat(well_id, rows)
            for well_id, rows in sorted(
                zip(row_counts["well_id"], row_counts["rows"], strict=True)
            )
        ]
    )
    expected: dict[str, int] = {}
    splitter = GroupKFold(n_splits=3)
    dummy = np.zeros((len(expanded_groups), 1), dtype=np.float32)
    for fold, (_, valid_idx) in enumerate(
        splitter.split(dummy, groups=expanded_groups)
    ):
        for well_id in np.unique(expanded_groups[valid_idx]):
            expected[str(well_id)] = fold
    assert actual.set_index("well_id")["fold"].to_dict() == expected


def test_stable_shuffle_and_feature_freeze_are_order_invariant(train_module) -> None:
    wells = ["wa", "wb", "wc", "wd", "we"]
    lookup = {"wa": "g1", "wb": "g1", "wc": "g2", "wd": "g3", "we": "g3"}
    first = train_module.stable_shuffled_group_lookup(wells, lookup, fold=2)
    second = train_module.stable_shuffled_group_lookup(
        list(reversed(wells)), lookup, fold=2
    )
    assert first == second
    assert sorted(first.values()) == sorted(lookup.values())
    assert any(first[well] != lookup[well] for well in wells)

    base = pd.DataFrame(
        {
            "fold": [0, 0],
            "well_id": ["wb", "wa"],
            "group_id": ["g1", "g1"],
            "control": ["real_native_group"] * 2,
            "fallback_reason": ["exact_outer_train_group"] * 2,
            "prior_source_wells_sha256": ["a" * 64] * 2,
            "fit_well_overlap": [0, 0],
            "outer_valid_truth_rows_before_feature_freeze": [0, 0],
            **{feature: [1.0, 2.0] for feature in train_module.FEATURE_COLUMNS},
        }
    )
    first_frozen, first_sha = train_module.add_freeze_hashes(base)
    second_frozen, second_sha = train_module.add_freeze_hashes(base.iloc[::-1])
    assert first_sha == second_sha
    pd.testing.assert_frame_equal(
        first_frozen.sort_values("well_id").reset_index(drop=True),
        second_frozen.sort_values("well_id").reset_index(drop=True),
    )


def test_group_prior_exact_then_global_fallback_is_finite(train_module) -> None:
    config = load_config()
    quality = pd.DataFrame(
        {
            "fold": [0, 0, 0],
            "well_id": ["wa", "wb", "wc"],
            "support_rows": [50, 70, 90],
            "fit_available": [True, True, True],
            "bias_at_gr50": [-4.0, 2.0, 8.0],
            "residual_sigma_mad": [2.0, 4.0, 6.0],
            "fit_rmse": [3.0, 5.0, 7.0],
            "fallback_reason": [None, None, None],
        }
    )
    source_lookup = {"wa": "ga", "wb": "ga", "wc": "gb"}
    priors, global_prior = train_module.aggregate_group_quality_priors(
        quality,
        source_lookup,
        fold=0,
        control="real_native_group",
        config=config,
    )
    features = train_module.build_valid_feature_rows(
        ["wv_exact", "wv_unseen"],
        {"wv_exact": "ga", "wv_unseen": "gz"},
        priors,
        global_prior,
        fold=0,
        control="real_native_group",
        fit_wells=list(source_lookup),
    )
    selected = features.set_index("well_id")
    assert selected.loc["wv_exact", "typewell_group_prior_available"] == 1.0
    assert selected.loc["wv_unseen", "typewell_group_prior_available"] == 0.0
    assert np.isfinite(features[train_module.FEATURE_COLUMNS].to_numpy()).all()
    assert selected.loc["wv_exact", "typewell_group_bias_abs_gr50"] == pytest.approx(1.0)


def synthetic_scored_inputs(train_module):
    feature_rows = []
    error_rows = []
    for fold in range(5):
        for offset in range(8):
            well_id = f"w{fold}_{offset}"
            error_rows.append(
                {
                    "well_id": well_id,
                    "fold": fold,
                    "exp148_well_rmse_ft": 1.0 + offset,
                }
            )
            for control in train_module.CONTROL_NAMES:
                sigma = (
                    1.0 + offset
                    if control == "real_native_group"
                    else 8.0 - offset
                )
                feature_rows.append(
                    {
                        "fold": fold,
                        "well_id": well_id,
                        "group_id": f"g{offset}",
                        "control": control,
                        "fallback_reason": "exact_outer_train_group",
                        "prior_source_wells_sha256": "a" * 64,
                        "fit_well_overlap": 0,
                        "outer_valid_truth_rows_before_feature_freeze": 0,
                        "typewell_group_log_support_wells": 2.0,
                        "typewell_group_log_support_rows": 5.0,
                        "typewell_group_residual_sigma": sigma,
                        "typewell_group_fit_rmse": sigma,
                        "typewell_group_bias_abs_gr50": sigma,
                        "typewell_group_prior_available": 1.0,
                    }
                )
    features, freeze_sha = train_module.add_freeze_hashes(pd.DataFrame(feature_rows))
    return features, pd.DataFrame(error_rows), freeze_sha


def test_late_error_attachment_and_all_gate_semantics(train_module) -> None:
    config = load_config()
    features, error, freeze_sha = synthetic_scored_inputs(train_module)
    with pytest.raises(ValueError, match="complete feature freeze"):
        train_module.attach_error_and_compute_readout(features, error, "short")
    scored, metrics, quartile = train_module.attach_error_and_compute_readout(
        features,
        error,
        freeze_sha,
    )
    gate = train_module.evaluate_stage_0_gate(features, metrics, quartile, config)
    assert gate["passed"]
    assert gate["positive_folds"] == 5
    assert gate["real_minus_shuffle_spearman"] > 1.0
    assert gate["q4_minus_q1_exp148_well_rmse_ft"] > 0.25
    assert len(scored) == len(features)

    failed_metrics = metrics.copy()
    pooled_real = failed_metrics["control"].eq("real_native_group") & failed_metrics[
        "fold"
    ].astype(str).eq("pooled")
    failed_metrics.loc[
        pooled_real, "residual_sigma_vs_exp148_well_rmse_spearman"
    ] = 0.0
    failed_gate = train_module.evaluate_stage_0_gate(
        features,
        failed_metrics,
        quartile,
        config,
    )
    assert not failed_gate["passed"]
    assert not failed_gate["checks"][
        "minimum_residual_sigma_vs_exp148_well_rmse_spearman"
    ]


def test_inference_contract_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp353_inference_contract")
    config = load_config()
    contract = module.validate_disabled_inference(config)
    assert not contract["inference_enabled"]
    assert not contract["create_submission"]
    config["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        module.validate_disabled_inference(config)

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp360_typewell_reference_shift_zncc_confidence_readout"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp360_typewell_reference_shift_zncc_confidence_readout_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp360_typewell_reference_shift_zncc_confidence_readout_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp360_train_test")


@pytest.fixture
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_shift_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = 200
    known_rows = 40
    true_tvt = 50.0 + 0.5 * np.arange(rows, dtype=np.float64)

    def gr_curve(tvt: np.ndarray) -> np.ndarray:
        scaled = tvt / 40.0
        return scaled**3 + 0.1 * scaled**5

    horizontal = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=np.float64),
            "GR": gr_curve(true_tvt),
            "TVT_input": np.r_[
                true_tvt[:known_rows], np.full(rows - known_rows, np.nan)
            ],
        }
    )
    typewell_tvt = np.linspace(0.0, 250.0, 5001)
    typewell = pd.DataFrame(
        {"TVT": typewell_tvt, "GR": gr_curve(typewell_tvt)}
    )
    row_idx = np.arange(known_rows, rows, dtype=np.int64)
    oof_safe = pd.DataFrame(
        {
            "well_id": "well-a",
            "row_idx": row_idx,
            "suffix_offset": np.arange(len(row_idx), dtype=np.int64),
            "fold": 0,
            "tvt_geop": true_tvt[row_idx] - 10.0,
        }
    )
    return oof_safe, horizontal, typewell


def test_scientific_contract_is_exactly_the_approved_zero_booster_readout(
    train, config
) -> None:
    train.validate_scientific_contract(config)
    broken = copy.deepcopy(config)
    broken["data"]["block_size"] = 256
    with pytest.raises(ValueError, match="512"):
        train.validate_scientific_contract(broken)
    broken = copy.deepcopy(config)
    broken["execution_contract"]["boosters"] = 1
    with pytest.raises(ValueError, match="zero-booster"):
        train.validate_scientific_contract(broken)


def test_typewell_reference_shift_sign_and_raw_finite_zncc(train, config) -> None:
    oof_safe, horizontal, typewell = synthetic_shift_inputs()
    scores, manifest = train.score_well_target_free_zncc(
        oof_safe, horizontal, typewell, config
    )
    top = scores.loc[scores["valid"]].sort_values(
        ["zncc", "shift_slot"], ascending=[False, True], kind="mergesort"
    ).iloc[0]
    assert top["shift_ft"] == pytest.approx(10.0)
    assert top["zncc"] == pytest.approx(1.0, abs=1e-10)
    assert manifest["blocks"] == 1
    assert scores["finite_pair_count"].eq(len(oof_safe)).all()


def test_horizontal_safe_loader_never_materializes_truth(train, tmp_path) -> None:
    path = tmp_path / "well__horizontal_well.csv"
    pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "GR": [10.0, 11.0],
            "TVT_input": [5.0, np.nan],
            "TVT": [5.0, 6.0],
        }
    ).to_csv(path, index=False)
    safe = train.load_horizontal_safe(path)
    assert safe.columns.tolist() == ["MD", "GR", "TVT_input"]
    assert "TVT" not in safe.columns


def test_zncc_rejects_truth_columns_and_invalidates_low_variance(train, config) -> None:
    oof_safe, horizontal, typewell = synthetic_shift_inputs()
    leaked = oof_safe.assign(tvt_true=oof_safe["tvt_geop"])
    with pytest.raises(ValueError, match="forbidden"):
        train.score_well_target_free_zncc(leaked, horizontal, typewell, config)
    constant = horizontal.copy()
    constant.loc[oof_safe["row_idx"], "GR"] = 1.0
    scores, _ = train.score_well_target_free_zncc(
        oof_safe, constant, typewell, config
    )
    assert not scores["valid"].any()
    assert scores["zncc"].eq(-1.0).all()


def test_exact_ties_prefer_zero_then_small_negative_shift(train) -> None:
    scores = np.ones(13, dtype=np.float64)
    valid = np.ones(13, dtype=bool)
    ranked = train.tie_resolved_valid_slots(scores, valid)
    assert train.EXPECTED_SHIFTS[ranked[:4]].tolist() == [0.0, -2.0, 2.0, -5.0]


def test_stable_permutation_is_deterministic_and_preserves_valid_scores(train) -> None:
    scores = np.linspace(-0.9, 0.9, 13)
    valid = np.ones(13, dtype=bool)
    first = train.stable_valid_score_permutation(
        scores, valid, well_id="well-a", block_id=7
    )
    second = train.stable_valid_score_permutation(
        scores, valid, well_id="well-a", block_id=7
    )
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), np.sort(scores))
    assert not np.array_equal(first, scores)


def test_primary_and_matched_control_features_use_the_same_blocks(train, config) -> None:
    oof_safe, horizontal, typewell = synthetic_shift_inputs()
    zncc, _ = train.score_well_target_free_zncc(
        oof_safe, horizontal, typewell, config
    )
    raw = zncc.drop(columns=["zncc", "valid", "finite_pair_count"]).copy()
    raw["likelihood_mean"] = -np.square(raw["shift_ft"] - 5.0)
    raw["likelihood_rank"] = (
        raw["likelihood_mean"].rank(method="first", ascending=False).astype(int)
    )
    features = train.build_target_free_block_features(zncc, raw)
    assert len(features) == 1
    row = features.iloc[0]
    assert bool(row["core_supported"])
    assert row["top1_shift_ft"] == pytest.approx(10.0)
    assert np.isfinite(row[f"risk__{train.PRIMARY_FAMILY}"])
    assert np.isfinite(row[f"raw_gaussian_risk__{train.PRIMARY_FAMILY}"])
    assert np.isfinite(row[f"permutation_risk__{train.PRIMARY_FAMILY}"])


def test_truth_ledger_fails_closed_before_freeze(train) -> None:
    ledger = train.TruthAccessLedger()
    with pytest.raises(ValueError, match="before target-free freeze"):
        ledger.register_truth_access()
    assert ledger.count_before_freeze == 1
    clean = train.TruthAccessLedger()
    clean.mark_frozen()
    clean.register_truth_access()
    assert clean.count_before_freeze == 0


def passing_metric_frames(train) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scope_rows = []
    fold_rows = []
    auc_by_variant = {
        "real_zncc": 0.70,
        "historical_raw_gaussian": 0.65,
        "stable_permutation": 0.64,
    }
    for variant, auc in auc_by_variant.items():
        for scope in (
            "pooled",
            "distance_1000_plus",
            "hidden_like_spatial",
            "hidden_like_typewell_purged",
        ):
            scope_rows.append(
                {
                    "variant": variant,
                    "family": train.PRIMARY_FAMILY,
                    "scope": scope,
                    "q4_minus_q1_mean_exp264_block_rmse": 0.8
                    if scope == "pooled"
                    else 0.2,
                    "q4_minus_q1_median_exp264_block_rmse": 0.3,
                    "row_weighted_abs_error_ge_10ft_auc": auc,
                }
            )
        for fold in range(5):
            fold_rows.append(
                {
                    "variant": variant,
                    "family": train.PRIMARY_FAMILY,
                    "fold": fold,
                    "q4_minus_q1_mean_exp264_block_rmse": 0.2,
                    "q4_minus_q1_median_exp264_block_rmse": 0.1,
                    "row_weighted_abs_error_ge_10ft_auc": auc,
                }
            )
    boundaries = pd.DataFrame(
        {
            "variant": ["real_zncc"] * 5,
            "family": [train.PRIMARY_FAMILY] * 5,
            "fold": np.arange(5),
            "q25_risk_boundary": np.zeros(5),
            "q75_risk_boundary": np.ones(5),
        }
    )
    return pd.DataFrame(scope_rows), pd.DataFrame(fold_rows), boundaries


def test_primary_gate_is_fail_closed_and_cannot_be_rescued(train, config) -> None:
    scopes, folds, boundaries = passing_metric_frames(train)
    freeze = {
        "technical_passed": True,
        "technical_checks": {"truth_access_count_zero": True},
    }
    gate, decision = train.evaluate_fixed_gate(
        scopes, folds, boundaries, freeze, config
    )
    assert gate["passed"] is True
    assert decision["action"].startswith("propose_separate_addonly")
    broken = scopes.copy()
    broken.loc[
        broken["variant"].eq("real_zncc") & broken["scope"].eq("pooled"),
        "row_weighted_abs_error_ge_10ft_auc",
    ] = 0.59
    gate, decision = train.evaluate_fixed_gate(
        broken, folds, boundaries, freeze, config
    )
    assert gate["passed"] is False
    assert decision["action"] == "close_zncc_confidence_branch_without_rescue"


def test_inference_is_explicitly_disabled(config) -> None:
    inference = load_module(INFERENCE_SOURCE, "exp360_inference_test")
    contract = inference.validate_disabled_inference(config)
    assert contract["readout_families"] == 6
    assert contract["models"] == 0
    assert contract["hmm_well_runs"] == 0
    with pytest.raises(RuntimeError, match="inference, and submission are disabled"):
        inference.stop_disabled_inference(config)

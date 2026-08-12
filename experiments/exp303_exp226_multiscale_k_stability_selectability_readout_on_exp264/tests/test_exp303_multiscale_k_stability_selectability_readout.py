from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments/exp303_exp226_multiscale_k_stability_selectability_readout_on_exp264"
SOURCE = (
    EXP_DIR
    / (
        "exp303_exp226_multiscale_k_stability_selectability_readout_on_exp264_"
        "compact_selfcontained_train.py"
    )
)


def load_module():
    spec = importlib.util.spec_from_file_location("exp303_train", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train():
    return load_module()


@pytest.fixture(scope="module")
def config():
    with (EXP_DIR / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


def test_config_records_consumed_kaggle_run_authorization(config):
    assert all(config["dependencies"]["status"].values())
    execution = config["execution"]
    assert execution["implementation"] is True
    assert execution["implementation_authorized"] is True
    assert execution["kaggle_execution_authorized"] is False
    assert execution["kaggle_execution_authorization_consumed_at"] == "2026-07-21 13:42:32 JST"
    assert execution["fixed_readout_variants_if_implemented"] == 1
    assert execution["outer_evaluation_folds_if_implemented"] == 5
    assert execution["lightgbm_config_count"] == 0
    assert execution["trained_fold_count"] == 0
    assert execution["total_boosters"] == 0
    assert execution["candidate_regeneration"] is False
    assert execution["parent_or_control_retraining"] is False
    assert config["runtime"]["kaggle"]["train_run_on_push"] is False
    assert config["runtime"]["kaggle"]["train_kernel_version"] == 1
    assert config["runtime"]["kaggle"]["train_kernel_id_no"] == 128080983


def test_config_pins_corrected_and_multiscale_inputs(config):
    expected = config["data"]["exp302_predictions"]["expected"]
    assert expected["exp226_k12"]["prediction_content_sha256"] == (
        "c3d7dfe20ad3b8c7d6d5220023bbb4526fb90d10cc73f01e612db847af70da63"
    )
    assert expected["exp226_k16"]["decompressed_sha256"] == (
        "709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609"
    )
    assert expected["exp226_k24"]["prediction_content_sha256"] == (
        "dca92e8f21d3b8b33d1543fe3df0bf586be3a2604b76ee1bf19fa84a327f06ef"
    )
    assert config["data"]["exp264_stage_c_candidate_score"]["expected_sha256"] == (
        "a10b7848127f01bef522f4b17dfd1640c9784956892dc24fc1159e3869500abc"
    )
    assert config["data"]["exp302_predictions"]["freeze_manifest"]["expected_file_sha256"] == (
        "bd80a4e02ddda8222821f4186adfa264ce1382504168be47ed51e2e5c04b6919"
    )


def test_streaming_prediction_hash_matches_full_parsed_frame(train, tmp_path):
    frame = pd.DataFrame(
        {
            "id": pd.Series(["a_0", "a_1", "b_0"], dtype="string"),
            "well": pd.Series(["a", "a", "b"], dtype="string"),
            "well_row_idx": pd.Series([0, 1, 0], dtype="int32"),
            "outer_fold": pd.Series([0, 0, 1], dtype="int64"),
            "variant": pd.Series(["exp226_k12"] * 3, dtype="string"),
            "candidate_tvt": pd.Series([1.0, 2.0, 3.0], dtype="float64"),
        }
    )
    path = tmp_path / "prediction.csv.gz"
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    parsed = train.read_exp302_variant(path, "exp226_k12")
    expected = train.frame_content_sha256(parsed)
    seen = []
    rows, actual = train.stream_exp302_variant(
        path,
        "exp226_k12",
        expected,
        lambda chunk, start: seen.append((start, len(chunk))),
        chunk_rows=2,
    )
    assert rows == 3
    assert actual == expected
    assert seen == [(0, 2), (2, 1)]


def _synthetic_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a_0", "a_1", "a_2", "a_3", "b_0", "b_1", "b_2", "b_3"],
            "well_id": ["a"] * 4 + ["b"] * 4,
            "row_idx": [0, 1, 2, 3, 0, 1, 2, 3],
            "suffix_offset": [0, 1, 2, 3, 0, 1, 2, 3],
            "fold": [0] * 4 + [1] * 4,
            "MD": np.arange(8, dtype=float),
            "exp226_k12_prediction": [0, 1, 2, 3, 4, 5, 6, 7],
            "exp226_k16_prediction": [1, 2, 3, 4, 5, 6, 7, 8],
            "exp226_k24_prediction": [3, 4, 5, 6, 8, 9, 10, 11],
            "exp264_stage_c_selected_candidate_id": ["exp226_k16"] * 8,
            "exp264_stage_c_selected_hard_prediction": np.arange(8, dtype=float),
        }
    )


def test_fixed_features_follow_preregistered_formula(train):
    features = train.build_fixed_row_features(_synthetic_rows(), h128=2, boundary_radius=0)
    assert np.allclose(features["level_spread_ft"], [3.0] * 4 + [4.0] * 4)
    assert np.allclose(features["k16_midpoint_deviation_ft"], [0.5] * 4 + [1.0] * 4)
    assert np.allclose(features["outer_asymmetry_ft"], [-1.0] * 4 + [-2.0] * 4)
    assert not features["direction_agreement"].any()
    assert features["k_order_monotone"].all()
    assert np.allclose(features["slope_spread_ft"], 0.0)
    assert not (set(features.columns) & train.FORBIDDEN_PRE_FREEZE_COLUMNS)


def test_h128_slope_and_segment_boundary_helpers_are_fixed(train):
    assert np.isclose(train.ols_slope_normalized(np.array([2.0, 4.0, 6.0])), 2.0)
    mask = train.segment_boundary_mask(24, [12], radius=0)
    assert np.array_equal(np.flatnonzero(mask), np.arange(2, 24, 2))
    expanded = train.segment_boundary_mask(24, [12], radius=1)
    assert expanded.sum() > mask.sum()


def test_empirical_percentile_uses_reference_only_and_average_ties(train):
    reference = np.array([0.0, 1.0, 1.0, 3.0])
    values = np.array([-1.0, 1.0, 2.0, 4.0])
    observed = train.empirical_percentile(reference, values)
    assert np.allclose(observed, [0.0, 0.5, 0.75, 1.0])
    changed_valid = train.empirical_percentile(reference, np.array([1000.0]))
    assert np.allclose(changed_valid, [1.0])


def test_forbidden_truth_columns_fail_closed(train):
    train.assert_no_forbidden_columns(["well_id", "row_idx", "candidate_tvt"])
    with pytest.raises(ValueError, match="forbidden"):
        train.assert_no_forbidden_columns(["well_id", "TVT"])
    with pytest.raises(ValueError, match="forbidden"):
        train.assert_no_forbidden_columns(["actual_abs_error"])


def test_h512_block_score_is_row_p90(train):
    features = train.build_fixed_row_features(_synthetic_rows(), h128=2, boundary_radius=0)
    scores = np.arange(len(features), dtype=float) / 10.0
    blocks = train.build_h512_blocks(features, scores, horizon=3)
    first = blocks.iloc[0]
    assert first["well_id"] == "a"
    assert first["rows"] == 3
    assert np.isclose(first["multiscale_k_instability_score"], np.quantile([0.0, 0.1, 0.2], 0.9))
    assert features["h512_block_id"].notna().all()


def test_truth_ledger_refuses_access_before_freeze(train):
    ledger = train.TruthAccessLedger()
    with pytest.raises(ValueError, match="before target-free freeze"):
        ledger.register_truth_access()
    assert ledger.count_before_freeze == 1
    ledger.mark_frozen()
    ledger.register_truth_access()
    assert ledger.count_before_freeze == 1


def test_auc_and_fixed_quintile_ties(train):
    labels = np.array([False, False, True, True])
    assert train.roc_auc_binary(labels, np.array([0.0, 0.0, 1.0, 1.0])) == 1.0
    assert train.roc_auc_binary(np.array([True, True]), np.array([0.0, 1.0])) is None
    frame = pd.DataFrame(
        {
            "multiscale_k_instability_score": np.arange(10, dtype=float),
            "positive_label": [False] * 5 + [True] * 5,
            "k16_benefit_ft": np.arange(10, dtype=float) / 10.0,
        }
    )
    summary = train.fixed_quintile_summary(frame)
    assert summary["positive_rate_lift"] == float("inf")
    assert summary["mean_k16_benefit_delta_ft"] > 0.25


def test_fixed_readout_can_pass_only_all_preregistered_guards(train, config):
    rows = []
    blocks = []
    hidden_wells = set()
    for fold in range(5):
        for polarity in (0, 1):
            block_id = len(blocks)
            well = f"w{fold}_{polarity}"
            hidden_wells.add(well)
            score = float(polarity)
            rows.append(
                {
                    "h512_block_id": block_id,
                    "exp226_k16_prediction": 0.0,
                    "exp264_stage_c_selected_hard_prediction": 1.0 if polarity else 0.1,
                }
            )
            blocks.append(
                {
                    "block_id": block_id,
                    "well_id": well,
                    "block_index": 0,
                    "rows": 1,
                    "fold": fold,
                    "first_suffix_offset": 0,
                    "last_suffix_offset": 0,
                    "min_md_since": 1200.0,
                    "max_md_since": 1200.0,
                    "multiscale_k_instability_score": score,
                }
            )
    features = pd.DataFrame(rows)
    frozen_blocks = pd.DataFrame(blocks)
    hidden = {
        "hidden_like_spatial": hidden_wells,
        "hidden_like_typewell_purged": hidden_wells,
    }
    _, scopes, folds, _, decision = train.build_post_freeze_readout(
        features, frozen_blocks, np.zeros(len(features)), hidden, config
    )
    assert decision["scientific_passed"] is True
    assert decision["folds_with_auc_above_0p5"] == 5
    assert scopes.set_index("scope").loc["pooled_h512_blocks", "auc"] == 1.0
    assert folds["auc"].eq(1.0).all()


def test_source_is_self_contained_and_canonical_placeholder_is_preserved():
    source = SOURCE.read_text()
    assert "__file__" not in source
    assert "from settings import" not in source
    assert "from src" not in source
    assert (
        EXP_DIR / "exp303_exp226_multiscale_k_stability_selectability_readout_on_exp264_train.ipynb"
    ).exists()
    assert "compact_selfcontained_train.py" in SOURCE.name

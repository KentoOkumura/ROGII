from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit"
)
TRAIN_PATH = (
    EXP_DIR
    / (
        "exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit_"
        "compact_selfcontained_train.py"
    )
)
INFERENCE_PATH = (
    EXP_DIR
    / (
        "exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit_"
        "compact_selfcontained_inference.py"
    )
)


def _load_module(path: Path, name: str):
    os.environ["EXP298_IMPORT_ONLY"] = "1"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train = _load_module(TRAIN_PATH, "exp298_train")
inference = _load_module(INFERENCE_PATH, "exp298_inference")


@pytest.fixture
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def _keys(well_lengths: list[int]) -> pd.DataFrame:
    records: list[dict] = []
    for well_position, length in enumerate(well_lengths):
        well = f"well_{well_position}"
        for row in range(length):
            records.append(
                {
                    "id": f"{well}_{row}",
                    "well": well,
                    "well_row_idx": row,
                    "outer_fold": well_position % 5,
                    "md_since": float(row + 1001),
                }
            )
    return pd.DataFrame(records)


def _bank(tmp_path: Path, keys: pd.DataFrame, values: np.ndarray):
    path = tmp_path / "candidate_bank.f32"
    memory = np.memmap(path, mode="w+", dtype="float32", shape=values.shape)
    memory[:] = values.astype(np.float32)
    memory.flush()
    bank = train.CandidateBank(
        keys=keys,
        candidate_ids=train.EXPECTED_CANDIDATE_ORDER,
        values=memory,
        values_path=path,
        primitive_ids=train.EXPECTED_CANDIDATE_ORDER[:6],
        manifest={},
        manifest_path=tmp_path / "cache_manifest.json",
        key_content_sha256=train.frame_content_sha256(
            keys[train.VALUE_KEY_COLUMNS]
        ),
        candidate_content_sha256="",
        coverage_by_candidate={
            candidate: 1.0 for candidate in train.EXPECTED_CANDIDATE_ORDER
        },
        sample_parity=pd.DataFrame(),
        input_evidence=[],
    )
    bank.candidate_content_sha256 = train.candidate_bank_content_sha256(bank, 3)
    return bank


def _components(tmp_path: Path, keys: pd.DataFrame, values: np.ndarray):
    path = tmp_path / "components.f64"
    memory = np.memmap(path, mode="w+", dtype="float64", shape=values.shape)
    memory[:] = values.astype(np.float64)
    memory.flush()
    return train.ComponentBundle(
        component_ids=train.COMPONENT_IDS,
        values=memory,
        values_path=path,
        content_sha256=train.component_content_sha256(
            keys, train.COMPONENT_IDS, memory, 3
        ),
        source_path=tmp_path / "exp226.csv.gz",
        source_decompressed_sha256="source",
        source_physical_columns=(*train.EXP226_ALLOWLIST, "tvt_true", "error"),
        source_loaded_columns=train.EXP226_ALLOWLIST,
        source_fold_crosswalk={"source_0__evaluation_0": len(keys)},
        alias_max_abs_ft=0.0,
        input_evidence=[],
    )


def test_config_locks_zero_booster_implementation_contract(config: dict) -> None:
    assert config["experiment"]["route"] == "pf_beam"
    assert config["execution"]["implementation"] is True
    assert config["execution"]["active_audit_contracts"] == 1
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["evaluation_fold_count"] == 5
    assert config["execution"]["trained_fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["hmm_pf_well_runs"] == 0
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["kaggle_execution_authorized"] is True
    assert config["execution"]["kaggle_push_approved"] is True
    assert tuple(config["candidate_bank"]["order"]) == train.EXPECTED_CANDIDATE_ORDER
    assert tuple(
        config["data"]["exp226_oof"]["required_allowlisted_columns"]
    ) == train.EXP226_ALLOWLIST
    assert config["audit"]["tie_policy"]["candidate_order"] == (
        "exp293_order_then_exp226_pre_u"
    )
    assert config["audit"]["block_partition"]["affine_eligibility"] == {
        "minimum_selected_rows": 2,
        "singleton_policy": (
            "exclude_from_affine_metric_rank_win_and_unique_best_denominators"
        ),
        "preserve_exp293_block_id_boundary_and_sha": True,
        "require_singleton_counts_in_outputs": True,
        "candidate_independent_exclusion": True,
    }


def test_downstream_contract_sha_and_branch_order_are_fixed(config: dict) -> None:
    assert train.sha256_file(EXP_DIR / "downstream_branch_contract.md") == (
        train.EXPECTED_DOWNSTREAM_CONTRACT_SHA256
    )
    assert train.downstream_contract_sha256() == train.EXPECTED_DOWNSTREAM_CONTRACT_SHA256
    contract = (EXP_DIR / "downstream_branch_contract.md").read_text()
    assert "exp298 FAIL" in contract
    assert "Stage 2 FAIL" in contract
    assert "Stage 4は自動開始しない" in contract
    assert config["downstream"]["stage2_if_exp298_pass"].startswith("exp226_")


def test_affine_quotient_removes_only_block_intercept_and_slope() -> None:
    keys = _keys([5])
    layout = train.build_block_assignments(keys, [5]).layouts["h5"]
    error = 7.0 + 3.0 * layout.row_coordinate
    stats = train.aggregate_quotient(error, layout)
    assert stats.offset_sse[0] > 0.0
    assert stats.affine_sse[0] == pytest.approx(0.0, abs=1e-10)
    nonlinear = error + np.square(layout.row_coordinate)
    nonlinear_stats = train.aggregate_quotient(nonlinear, layout)
    assert nonlinear_stats.affine_sse[0] > 0.0


def test_final_single_row_block_is_excluded_without_affine_fallback() -> None:
    keys = _keys([5])
    layout = train.build_block_assignments(keys, [4]).layouts["h4"]
    stats = train.aggregate_quotient(np.arange(5, dtype=np.float64), layout)
    assert layout.group_rows.tolist() == [4, 1]
    assert stats.affine_valid.tolist() == [True, False]
    assert np.isnan(stats.affine_sse[1])
    record = train._metric_record(
        "exp226_pre_u",
        "h4",
        "overall",
        stats,
        np.ones(layout.n_groups, dtype=bool),
        group_well=layout.group_well,
    )
    assert record["affine_eligible_rows"] == 4
    assert record["affine_valid_rows"] == 4
    assert record["affine_valid_eligible_row_fraction"] == 1.0
    assert record["affine_invalid_eligible_blocks"] == 0
    assert record["affine_excluded_singleton_blocks"] == 1
    assert record["affine_excluded_singleton_rows"] == 1
    assert record["affine_excluded_singleton_wells"] == 1


def test_difference_metrics_do_not_cross_block_boundaries() -> None:
    keys = _keys([3, 3])
    layout = train.build_block_assignments(keys, [3]).layouts["h3"]
    error = np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
    first_sse, first_rows, second_sse, second_rows = train.difference_group_metrics(
        error, layout
    )
    assert first_rows.tolist() == [2, 2]
    assert first_sse.tolist() == [2.0, 2.0]
    assert second_rows.tolist() == [1, 1]
    assert second_sse.tolist() == [0.0, 0.0]


def test_component_content_sha_detects_value_mutation(tmp_path: Path) -> None:
    keys = _keys([3])
    values = np.arange(9, dtype=np.float64).reshape(3, 3)
    components = _components(tmp_path, keys, values)
    before = components.content_sha256
    components.values[0, 1] += 1.0
    components.values.flush()
    after = train.component_content_sha256(keys, train.COMPONENT_IDS, components.values, 2)
    assert after != before


def test_primary_pre_u_ranks_first_after_affine_quotient(
    tmp_path: Path, config: dict
) -> None:
    keys = _keys([5, 5, 5, 5, 5])
    truth = np.tile(np.array([10.0, 11.0, 12.0, 13.0, 14.0]), 5)
    curved_error = np.tile(np.array([0.0, 2.0, 0.0, 2.0, 999.0]), 5)
    bank_values = np.column_stack(
        [truth - curved_error - 0.1 * position for position in range(12)]
    )
    bank = _bank(tmp_path, keys, bank_values)
    pre_u_error = np.tile(np.array([0.0, 1.0, 2.0, 3.0, 999.0]), 5)
    component_values = np.column_stack(
        [truth - curved_error, truth - pre_u_error, bank_values[:, 0]]
    )
    components = _components(tmp_path, keys, component_values)
    assignments = train.build_block_assignments(keys, [4])
    pooled, folds, scopes, blocks, by_well = train.build_quotient_readouts(
        bank,
        components,
        assignments,
        truth,
        {
            "hidden_like_spatial": {"well_0", "well_1"},
            "hidden_like_typewell_purged": {"well_2", "well_3"},
        },
        config,
    )
    primary = pooled[
        pooled["candidate_id"].eq("exp226_pre_u")
        & pooled["horizon"].eq("h4")
    ].iloc[0]
    assert primary["affine_quotient_rmse"] == pytest.approx(0.0, abs=1e-8)
    assert primary["affine_quotient_rank"] == 1
    assert primary["rows"] == 25
    assert primary["affine_eligible_rows"] == 20
    assert primary["affine_excluded_singleton_blocks"] == 5
    assert primary["affine_excluded_singleton_rows"] == 5
    assert primary["affine_excluded_singleton_wells"] == 5
    assert primary["block_win_fraction"] == 1.0
    assert primary["strict_unique_best_block_fraction"] == 1.0
    primary_blocks = blocks[
        blocks["candidate_id"].eq("exp226_pre_u")
        & blocks["horizon"].eq("h4")
    ]
    assert len(primary_blocks) == 10
    assert primary_blocks["affine_eligible"].sum() == 5
    assert primary_blocks["excluded_singleton"].sum() == 5
    assert not primary_blocks.loc[
        primary_blocks["excluded_singleton"], "is_best"
    ].any()
    assert not primary_blocks.loc[
        primary_blocks["excluded_singleton"], "is_unique_best"
    ].any()
    assert "exp226_post_u" not in set(
        blocks.loc[blocks["horizon"].eq("h4"), "candidate_id"]
    )
    assert len(folds[folds["horizon"].eq("h4")]) == 15 * 5
    assert set(scopes["scope"]) == {
        "1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    }
    assert len(by_well[by_well["horizon"].eq("h4")]) == 15 * 5


def test_execution_has_explicit_notebook_and_kaggle_approval(config: dict) -> None:
    train.validate_execution_contract(config, require_kaggle_approval=False)
    train.validate_execution_contract(config, require_kaggle_approval=True)


def test_expected_outputs_exclude_coefficients_predictions_and_submission(
    config: dict,
) -> None:
    artifacts = config["audit"]["expected_artifacts_if_implemented"]
    forbidden_tokens = ("coefficient", "corrected", "prediction", "submission")
    assert not any(token in name for name in artifacts for token in forbidden_tokens)
    assert config["audit"]["quotient"]["persist_coefficients"] is False
    assert config["audit"]["quotient"]["persist_corrected_prediction"] is False


def test_inference_is_fail_closed(config: dict) -> None:
    checks = inference.validate_disabled_inference(config)
    assert all(checks.values())
    with pytest.raises(RuntimeError, match="intentionally disabled"):
        inference.stop_without_inference(config)


def test_compact_sources_are_notebook_safe() -> None:
    assert "__file__" not in TRAIN_PATH.read_text()
    assert "__file__" not in INFERENCE_PATH.read_text()
    assert 'SUMMARY["support"]' not in TRAIN_PATH.read_text()
    assert 'SUMMARY["decision"]' in TRAIN_PATH.read_text()

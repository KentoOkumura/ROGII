from __future__ import annotations

import copy
import gzip
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp293_physics_only_candidate_bank_headroom_contract"
TRAIN_PATH = (
    EXP_DIR
    / "exp293_physics_only_candidate_bank_headroom_contract_compact_selfcontained_train.py"
)
INFERENCE_PATH = (
    EXP_DIR
    / "exp293_physics_only_candidate_bank_headroom_contract_compact_selfcontained_inference.py"
)


def _load_module(path: Path, name: str):
    os.environ["EXP293_IMPORT_ONLY"] = "1"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train = _load_module(TRAIN_PATH, "exp293_train")
inference = _load_module(INFERENCE_PATH, "exp293_inference")


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
                    "md_since": float(row + 1),
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


def test_config_and_downstream_branch_contract_are_fixed(config: dict) -> None:
    assert tuple(config["candidate_bank"]["order"]) == train.EXPECTED_CANDIDATE_ORDER
    assert config["experiment"]["route"] == "pf_beam"
    assert config["execution"]["active_audit_contracts"] == 1
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["trained_fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["downstream"]["branch_after_exp293"]["support_pass"].startswith(
        "stage2_"
    )
    assert config["downstream"]["branch_after_exp293"]["support_fail"].startswith(
        "stage4_"
    )
    assert "no_automatic_stage4" in config["downstream"]["branch_after_stage2"][
        "fail"
    ]
    downstream = (EXP_DIR / "downstream_branch_contract.md").read_text()
    assert train.sha256_file(EXP_DIR / "downstream_branch_contract.md") == (
        train.EXPECTED_DOWNSTREAM_CONTRACT_SHA256
    )
    assert train.downstream_contract_sha256() == (
        train.EXPECTED_DOWNSTREAM_CONTRACT_SHA256
    )
    assert "Stage 4へ自動分岐しない" in downstream
    assert "PASS後も直接Stage 2へ進まず" in downstream


def test_formula_materialization_uses_fixed_float32_order(
    tmp_path: Path, config: dict
) -> None:
    n_rows = 4
    values = np.memmap(
        tmp_path / "formula.f32",
        mode="w+",
        dtype="float32",
        shape=(n_rows, 12),
    )
    values[:] = np.nan
    position = {name: idx for idx, name in enumerate(train.EXPECTED_CANDIDATE_ORDER)}
    primitives = {
        "exp226_k16": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "selfgr_hmm_a070": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        "likpf_mean": np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32),
        "exact_hmm": np.array([13.0, 14.0, 15.0, 16.0], dtype=np.float32),
        "pf_ancc": np.array([17.0, 18.0, 19.0, 20.0], dtype=np.float32),
        "beam_mean": np.array([21.0, 22.0, 23.0, 24.0], dtype=np.float32),
    }
    for candidate, value in primitives.items():
        values[:, position[candidate]] = value
    train._materialize_formulas(values, position, config)
    expected_pair = np.float32(0.5) * (
        primitives["exp226_k16"] + primitives["selfgr_hmm_a070"]
    )
    np.testing.assert_array_equal(
        values[:, position["exp226_k16__selfgr_hmm_a070"]], expected_pair
    )
    expected_fixed = (
        np.float32(0.5) * primitives["exp226_k16"]
        + np.float32(0.25) * primitives["likpf_mean"]
        + np.float32(0.25) * primitives["exact_hmm"]
    ).astype(np.float32)
    np.testing.assert_array_equal(
        values[:, position["exp226_w500_50_50"]], expected_fixed
    )


def test_block_assignment_keeps_final_short_blocks() -> None:
    keys = _keys([130, 257])
    assignments = train.build_block_assignments(keys, [128, 256, 512])
    assert assignments.layouts["h128"].n_groups == 5
    assert assignments.layouts["h256"].n_groups == 3
    assert assignments.layouts["h512"].n_groups == 2
    assert assignments.layouts["whole_well"].n_groups == 2
    assert assignments.layouts["h128"].group_rows.tolist() == [128, 2, 128, 128, 1]
    assert assignments.layouts["h256"].group_rows.tolist() == [130, 256, 1]
    assert assignments.layouts["h512"].group_rows.tolist() == [130, 257]


def test_block_assignment_rejects_noncontiguous_well() -> None:
    keys = _keys([2, 2])
    shuffled = pd.concat([keys.iloc[:1], keys.iloc[2:], keys.iloc[1:2]], ignore_index=True)
    with pytest.raises(ValueError, match="not contiguous"):
        train.build_block_assignments(shuffled, [2])


def test_chunked_row_and_block_oracles_respect_candidate_order(
    tmp_path: Path, config: dict
) -> None:
    keys = _keys([4, 4])
    keys["md_since"] = np.tile([1.0, 2.0, 1001.0, 1002.0], 2)
    values = np.full((8, 12), 20.0, dtype=np.float32)
    values[:, 0] = np.tile([0.0, 0.0, 10.0, 10.0], 2)
    values[:, 1] = np.tile([10.0, 10.0, 0.0, 0.0], 2)
    values[:, 11] = 8.0
    bank = _bank(tmp_path, keys, values)
    assignments = train.build_block_assignments(keys, [2, 4])
    local_config = copy.deepcopy(config)
    local_config["audit"]["work_chunk_rows"] = 3
    state = train.compute_oracle_state(
        bank, np.zeros(8, dtype=np.float64), assignments, local_config
    )
    assert np.all(state.row_best_sse == 0.0)
    assert np.all(state.group_sse["h2"].min(axis=1) == 0.0)
    assert np.all(np.argmin(state.group_sse["h4"], axis=1) == 0)
    h4_rmse = np.sqrt(state.group_sse["h4"].min(axis=1).sum() / 8)
    assert h4_rmse == pytest.approx(np.sqrt(50.0))
    assert np.sqrt(state.anchor_row_sse.mean()) == pytest.approx(8.0)
    oracle, folds, subgroups, choices = train.build_metric_frames(
        bank,
        assignments,
        state,
        {
            "hidden_like_spatial": {"well_0"},
            "hidden_like_typewell_purged": {"well_1"},
        },
        local_config,
    )
    assert oracle.loc[
        oracle["scope"].eq("overall") & oracle["granularity"].eq("h2"),
        "oracle_rmse",
    ].item() == pytest.approx(0.0)
    assert oracle.loc[
        oracle["scope"].eq("overall") & oracle["granularity"].eq("h4"),
        "oracle_rmse",
    ].item() == pytest.approx(np.sqrt(50.0))
    assert set(folds["scope"]) == {"fold_0", "fold_1", "fold_2", "fold_3", "fold_4"}
    assert set(subgroups["scope"]) == {
        "1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    }
    assert not choices.empty
    by_well = train.build_by_well_metrics(
        bank, assignments, state, local_config
    )
    assert by_well["h2_oracle_rmse"].tolist() == [0.0, 0.0]


def test_required_recovery_formula_and_invalid_denominator() -> None:
    anchor_sse = 8.0**2 * 100
    oracle_sse = 5.0**2 * 100
    expected = (8.0**2 - 6.5**2) / (8.0**2 - 5.0**2)
    assert train.required_headroom_recovery(
        anchor_sse, oracle_sse, 100, 6.5
    ) == pytest.approx(expected)
    assert np.isnan(
        train.required_headroom_recovery(anchor_sse, anchor_sse, 100, 6.5)
    )


def test_support_pass_and_fail_route_are_deterministic(
    tmp_path: Path, config: dict
) -> None:
    keys = _keys([1, 1, 1, 1, 1])
    bank = _bank(tmp_path, keys, np.zeros((5, 12), dtype=np.float32))
    contract = tmp_path / "contract.json"
    manifest = tmp_path / "bank_manifest.json"
    block = tmp_path / "block.csv.gz"
    contract.write_text("{}\n")
    manifest.write_text("{}\n")
    block.write_bytes(gzip.compress(b"id\n", mtime=0))
    freeze = train.FreezeEvidence(
        contract_path=contract,
        contract_file_sha256=train.sha256_file(contract),
        bank_manifest_path=manifest,
        bank_manifest_file_sha256=train.sha256_file(manifest),
        candidate_content_sha256=bank.candidate_content_sha256,
        block_assignment_path=block,
        block_assignment_file_sha256=train.sha256_file(block),
        block_assignment_decompressed_sha256=train.sha256_decompressed_gzip(block),
        target_free_input_evidence_sha256="evidence",
        truth_access_count_before_freeze=0,
    )
    local_config = copy.deepcopy(config)
    local_config["validation"]["expected_rows"] = 5
    local_config["validation"]["expected_wells"] = 5
    pooled = pd.DataFrame(
        [
            {
                "scope": "overall",
                "granularity": "h512",
                "rows": 5,
                "anchor_rmse": 8.0,
                "oracle_rmse": 5.0,
                "required_recovery_to_6p5": 0.55,
            }
        ]
    )
    folds = pd.DataFrame(
        [
            {
                "scope": f"fold_{fold}",
                "granularity": "h512",
                "rows": 1,
                "anchor_rmse": 8.0,
                "oracle_rmse": 6.0,
                "required_recovery_to_6p5": 0.8,
            }
            for fold in range(5)
        ]
    )
    subgroups = pd.DataFrame(
        [
            {
                "scope": scope,
                "granularity": "h512",
                "rows": 1,
                "anchor_rmse": 8.0,
                "oracle_rmse": 6.0,
                "required_recovery_to_6p5": 0.8,
            }
            for scope in (
                "1000_plus",
                "hidden_like_spatial",
                "hidden_like_typewell_purged",
            )
        ]
    )
    parity = pd.DataFrame([{"passed": True}])
    passed = train.evaluate_support_decision(
        bank, freeze, pooled, folds, subgroups, parity, local_config
    )
    assert passed["support_passed"] is True
    assert passed["next_branch"].startswith("stage2_")

    failed_pooled = pooled.copy()
    failed_pooled.loc[0, "oracle_rmse"] = 5.6
    failed = train.evaluate_support_decision(
        bank,
        freeze,
        failed_pooled,
        folds,
        subgroups,
        parity,
        local_config,
    )
    assert failed["support_passed"] is False
    assert failed["next_branch"].startswith("stage4_")


def test_freeze_verification_detects_candidate_mutation(
    tmp_path: Path, config: dict
) -> None:
    keys = _keys([2])
    bank = _bank(tmp_path, keys, np.zeros((2, 12), dtype=np.float32))
    contract = tmp_path / "contract.json"
    manifest = tmp_path / "bank_manifest.json"
    block = tmp_path / "block.csv.gz"
    contract.write_text("{}\n")
    manifest.write_text("{}\n")
    block.write_bytes(gzip.compress(b"id\n", mtime=0))
    freeze = train.FreezeEvidence(
        contract_path=contract,
        contract_file_sha256=train.sha256_file(contract),
        bank_manifest_path=manifest,
        bank_manifest_file_sha256=train.sha256_file(manifest),
        candidate_content_sha256=bank.candidate_content_sha256,
        block_assignment_path=block,
        block_assignment_file_sha256=train.sha256_file(block),
        block_assignment_decompressed_sha256=train.sha256_decompressed_gzip(block),
        target_free_input_evidence_sha256="evidence",
        truth_access_count_before_freeze=0,
    )
    train.verify_freeze_before_truth(bank, freeze, 1)
    bank.values[0, 0] = np.float32(1.0)
    bank.values.flush()
    with pytest.raises(ValueError, match="candidate bank changed"):
        train.verify_freeze_before_truth(bank, freeze, 1)


def test_target_free_freeze_writes_deterministic_block_artifact(
    tmp_path: Path, config: dict
) -> None:
    keys = _keys([3, 1])
    bank = _bank(tmp_path, keys, np.zeros((4, 12), dtype=np.float32))
    bank.sample_parity = pd.DataFrame(
        [{"candidate_id": "exp226_w500_50_50", "passed": True}]
    )
    assignments = train.build_block_assignments(keys, [128, 256, 512])
    freeze = train.freeze_target_free_contract(
        bank,
        assignments,
        {
            "phase": "target_free",
            "source": "synthetic_hidden_like",
            "path": "synthetic",
        },
        config,
        tmp_path,
    )
    assert freeze.truth_access_count_before_freeze == 0
    assert freeze.block_assignment_path.exists()
    assert freeze.bank_manifest_path.exists()
    train.verify_freeze_before_truth(bank, freeze, 2)


def test_truth_columns_are_rejected_from_candidate_partition() -> None:
    train.reject_forbidden_candidate_columns(["id", "candidate_tvt"])
    with pytest.raises(ValueError, match="forbidden"):
        train.reject_forbidden_candidate_columns(["id", "true_tvt"])
    with pytest.raises(ValueError, match="forbidden"):
        train.reject_forbidden_candidate_columns(["id", "oracle_label"])


def test_inference_is_fail_closed_and_writes_no_submission(config: dict) -> None:
    checks = inference.validate_disabled_inference(config)
    assert all(checks.values())
    with pytest.raises(RuntimeError, match="intentionally disabled"):
        inference.stop_without_inference(config)
    assert "submission.csv" not in config["audit"]["expected_artifacts"]

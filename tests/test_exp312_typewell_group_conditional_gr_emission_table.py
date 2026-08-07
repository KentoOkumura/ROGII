from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp312_typewell_group_conditional_gr_emission_table"
TRAIN_SOURCE = EXP_DIR / (
    "exp312_typewell_group_conditional_gr_emission_table_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp312_typewell_group_conditional_gr_emission_table_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_contract_fixes_deployable12_zero_booster_and_parent_override() -> None:
    module = load_module(TRAIN_SOURCE, "exp312_contract")
    config = load_config()
    module.validate_scientific_contract(config)
    assert tuple(config["candidate_bank"]["order"]) == module.EXPECTED_CANDIDATE_ORDER
    assert config["validation"]["parent_gate_override"]["retained_failures"] == [
        "group_loo_fit_rmse_r2",
        "worst_well_delta",
    ]
    config["execution_contract"]["decoder_runs"] = 1
    with pytest.raises(ValueError, match="decoder_runs"):
        module.validate_scientific_contract(config)


def test_weighted_quantile_respects_equal_well_weights() -> None:
    module = load_module(TRAIN_SOURCE, "exp312_quantile")
    values = np.array([0.0, 0.0, 0.0, 100.0])
    row_weights = np.ones(4)
    equal_well_weights = np.array([1 / 3, 1 / 3, 1 / 3, 1.0])
    unweighted = module.weighted_quantile(values, row_weights, [0.5])[0]
    equal_well = module.weighted_quantile(values, equal_well_weights, [0.5])[0]
    assert unweighted == pytest.approx(0.0)
    assert equal_well > unweighted


def test_hierarchical_table_shrinks_and_uses_frozen_fallback_order() -> None:
    module = load_module(TRAIN_SOURCE, "exp312_table")
    config = load_config()
    n = 80
    records = pd.DataFrame(
        {
            "fold": 0,
            "well_id": np.repeat(["a", "b"], n),
            "group_id": np.repeat(["g1", "g2"], n),
            "typewell_gr": np.tile(np.linspace(0.0, 100.0, n), 2),
            "abs_gradient": np.tile(np.linspace(0.0, 3.0, n), 2),
            "missing_flag": np.zeros(2 * n, dtype=np.int8),
            "residual": np.concatenate([np.full(n, 10.0), np.full(n, -10.0)]),
            "weight": np.full(2 * n, 1.0 / n),
        }
    )
    gr_edges = np.linspace(10.0, 90.0, 9)
    gradient_edges = np.array([1.0, 2.0])
    table = module.fit_hierarchical_table(
        records,
        gr_edges,
        gradient_edges,
        fold=0,
        control="real",
        config=config,
    )
    group_row = table[(table["level"] == "group_unconditional") & (table["group_id"] == "g1")].iloc[
        0
    ]
    assert 0.0 < group_row["shrinkage_alpha"] < 1.0
    assert 0.0 < group_row["location"] < group_row["location_raw"]
    arrays = module.compile_table_arrays(table, ["g1", "g2"], 10, 3)
    location, scale, level = module.lookup_distribution(
        arrays,
        np.array([["g1"]]),
        np.array([[0]]),
        np.array([[0]]),
        np.array([[1]]),
    )
    assert np.isfinite(location).all() and np.isfinite(scale).all()
    assert level.item() in {0, 1, 2}


def test_group_shuffle_is_deterministic_and_preserves_label_multiset() -> None:
    module = load_module(TRAIN_SOURCE, "exp312_shuffle")
    wells = [f"w{index}" for index in range(12)]
    groups = {well: f"g{index // 3}" for index, well in enumerate(wells)}
    first = module.shuffled_group_lookup(wells, groups, fold=2)
    second = module.shuffled_group_lookup(list(reversed(wells)), groups, fold=2)
    assert first == second
    assert sorted(first.values()) == sorted(groups.values())
    assert any(first[well] != groups[well] for well in wells)


def test_tvt_shift_index_is_deterministic_within_well_and_count_matched() -> None:
    module = load_module(TRAIN_SOURCE, "exp312_shift")
    config = load_config()
    keys = pd.DataFrame(
        {
            "well": np.repeat(["a", "b"], [20, 25]),
            "id": [f"id_{index}" for index in range(45)],
        }
    )
    first = module.stable_shift_source_index(keys, config)
    second = module.stable_shift_source_index(keys.copy(), config)
    np.testing.assert_array_equal(first, second)
    for _, index in keys.groupby("well").indices.items():
        positions = np.asarray(index)
        assert sorted(first[positions].tolist()) == sorted(positions.tolist())
        assert not np.array_equal(first[positions], positions)


def test_rank_order_ties_and_truth_nearest_rank_are_candidate_order_stable() -> None:
    module = load_module(TRAIN_SOURCE, "exp312_rank")
    scores = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    order = module._rank_orders(scores)
    np.testing.assert_array_equal(order[0], [0, 1, 2])
    rank_orders = np.stack([order, order], axis=1)
    truth = np.array([1, 2], dtype=np.uint8)
    ranks = module.true_candidate_rank_positions(rank_orders, truth)
    np.testing.assert_array_equal(ranks, [[1, 1], [1, 1]])


def test_freeze_verification_rejects_rank_tampering(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp312_freeze")
    rank_path = tmp_path / "ranks.u1"
    rank_path.write_bytes(bytes([0, 1, 2]))
    freeze_path = tmp_path / "freeze.json"
    payload = {
        "outer_valid_truth_access_count_before_freeze": 0,
        "candidate_bank_content_sha256": "bank-sha",
        "emission_table_content_sha256": "table-sha",
    }
    freeze_path.write_text(json.dumps(payload))
    bank = type("Bank", (), {"candidate_content_sha256": "bank-sha"})()
    frozen = module.FrozenFold(
        fold=0,
        valid_positions=np.array([], dtype=np.int64),
        rank_path=rank_path,
        rank_sha256=module.sha256_file(rank_path),
        table_sha256="table-sha",
        freeze_path=freeze_path,
        freeze_sha256=module.sha256_file(freeze_path),
        gr_edges=np.array([]),
        gradient_edges=np.array([]),
        bin_edges=pd.DataFrame(),
        emission_tables=pd.DataFrame(),
        fallback=pd.DataFrame(),
    )
    module.verify_frozen_fold(frozen, bank)
    rank_path.write_bytes(bytes([2, 1, 0]))
    with pytest.raises(ValueError, match="rank order changed"):
        module.verify_frozen_fold(frozen, bank)


def test_small_fold_scores_freeze_before_late_truth_join(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp312_small_fold")
    config = load_config()
    candidate_ids = ("c0", "c1", "c2")
    wells = [f"w{index}" for index in range(10)]
    values_path = tmp_path / "values.f32"
    values = np.memmap(values_path, mode="w+", dtype="float32", shape=(10, 3))
    values[:] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    values.flush()
    keys = pd.DataFrame(
        {
            "id": [f"{well}_0" for well in wells],
            "well": wells,
            "well_row_idx": np.zeros(10, dtype=np.int32),
            "outer_fold": np.arange(10) % 2,
            "md_since": np.ones(10),
        }
    )
    bank = module.CandidateBank(
        keys=keys,
        candidate_ids=candidate_ids,
        values=values,
        values_path=values_path,
        manifest={},
        manifest_path=tmp_path / "manifest.json",
        key_content_sha256="keys",
        candidate_content_sha256="candidate-sha",
        sample_parity=pd.DataFrame(),
        input_evidence=[],
    )
    contexts = {}
    for well in wells:
        horizontal_path = tmp_path / f"{well}__horizontal_well.csv"
        pd.DataFrame({"TVT": [1.0], "GR": [10.0]}).to_csv(horizontal_path, index=False)
        contexts[well] = module.WellContext(
            well_id=well,
            horizontal_path=horizontal_path,
            typewell_path=tmp_path / f"{well}__typewell.csv",
            horizontal_gr=np.array([10.0]),
            horizontal_missing=np.array([False]),
            typewell_tvt=np.array([0.0, 2.0]),
            typewell_gr=np.array([0.0, 20.0]),
            typewell_abs_gradient=np.array([10.0, 10.0]),
        )
    hidden = pd.DataFrame(
        {
            "well_id": wells,
            "verification_like_spatial_role": ["valid"] * 10,
            "verification_like_typewell_purged_role": ["valid"] * 10,
        }
    )
    target_free = module.TargetFreeContext(
        wells=contexts,
        observed_gr=np.full(10, 10.0, dtype=np.float32),
        missing_flag=np.zeros(10, dtype=np.uint8),
        candidate_typewell_gr=np.tile(np.array([0.0, 10.0, 20.0], dtype=np.float32), (10, 1)),
        candidate_abs_gradient=np.full((10, 3), 10.0, dtype=np.float32),
        shift_source_index=np.arange(10),
        hidden_like=hidden,
        input_evidence=[],
    )
    fold_by_well = {well: index % 2 for index, well in enumerate(wells)}
    group_by_well = {well: "g0" for well in wells}
    parent = module.ParentContract(
        summary_path=tmp_path / "summary.json",
        fold_manifest=pd.DataFrame(),
        group_membership=pd.DataFrame(),
        group_by_well=group_by_well,
        fold_by_well=fold_by_well,
        input_evidence=[],
    )
    frozen = module.score_fold_target_free(bank, target_free, parent, 0, config, tmp_path)
    assert (
        json.loads(frozen.freeze_path.read_text())["outer_valid_truth_access_count_before_freeze"]
        == 0
    )
    readout = module.load_outer_valid_truth_after_freeze(bank, target_free, parent, frozen, config)
    assert len(readout.rank_positions) == 5
    assert set(readout.metrics["variant"]) == set(module.RANK_VARIANTS)


def test_inference_contract_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp312_inference")
    config = load_config()
    contract = module.validate_disabled_inference(config)
    assert not contract["inference_enabled"]
    assert not contract["create_submission"]
    config["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        module.validate_disabled_inference(config)

from __future__ import annotations

import importlib.util
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp289_fault_aware_transductive_geological_potential"
TRAIN_SOURCE = (
    EXP_DIR / "exp289_fault_aware_transductive_geological_potential_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp289_fault_aware_transductive_geological_potential_compact_selfcontained_inference.py"
)


def load_module(path: Path = TRAIN_SOURCE, name: str = "exp289_train"):
    previous = os.environ.get("EXP289_IMPORT_ONLY")
    os.environ["EXP289_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP289_IMPORT_ONLY", None)
        else:
            os.environ["EXP289_IMPORT_ONLY"] = previous


def load_config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_horizontal(rows: int = 96, known_rows: int = 48) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    z = -1000.0 - 0.2 * index
    tvt = 1200.0 + 0.3 * index + 0.001 * np.square(index)
    surface = z + tvt
    tvt_input = tvt.copy()
    tvt_input[known_rows:] = np.nan
    frame = pd.DataFrame(
        {
            "MD": index,
            "X": 100.0 + index,
            "Y": 200.0 + 0.5 * index,
            "Z": z,
            "ANCC": surface + 10.0,
            "ASTNU": surface + 20.0,
            "ASTNL": surface + 30.0,
            "EGFDU": surface + 40.0,
            "EGFDL": surface + 50.0,
            "BUDA": surface + 60.0,
            "TVT": tvt,
            "GR": 80.0 + np.sin(index),
            "TVT_input": tvt_input,
        }
    )
    return frame


def test_config_is_one_stage0_variant_and_zero_boosters() -> None:
    module = load_module(name="exp289_contract")
    config = load_config()
    module.validate_scientific_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["stages"]["stage0"]["active_variants"] == 1
    assert config["stages"]["stage0"]["ml_configs"] == 0
    assert config["stages"]["stage0"]["trained_folds"] == 0
    assert config["stages"]["stage0"]["boosters"] == 0
    assert config["execution"]["control_or_parent_retraining"] is False
    assert config["execution"]["kaggle_push_approved"] is True
    assert (
        config["execution"]["kaggle_push_approval_source"]
        == "user_message_run_exp289_2026_07_19"
    )
    assert config["inference"]["enabled"] is False


def test_outer_valid_loader_never_materializes_truth_or_formation(tmp_path: Path) -> None:
    module = load_module(name="exp289_target_safe")
    path = tmp_path / "well-a__horizontal_well.csv"
    synthetic_horizontal().to_csv(path, index=False)
    loaded = module.load_target_safe_horizontal(path)
    assert tuple(loaded.columns) == module.TARGET_SAFE_COLUMNS
    assert not module.TARGET_FORBIDDEN_COLUMNS.intersection(loaded.columns)
    module.validate_target_safe_frame(loaded)
    with pytest.raises(ValueError, match="forbidden columns"):
        module.validate_target_safe_frame(loaded.assign(TVT=1.0))
    second_path = tmp_path / "well-b__horizontal_well.csv"
    synthetic_horizontal(rows=80, known_rows=40).to_csv(second_path, index=False)
    nodes = module.build_target_nodes(
        {"well-a": path, "well-b": second_path}, ["well-a", "well-b"], load_config()
    )
    assert set(nodes["well_id"]) == {"well-a", "well-b"}
    assert not module.TARGET_FORBIDDEN_COLUMNS.intersection(nodes.columns)


def test_node_sampling_includes_anchor_and_final_row() -> None:
    module = load_module(name="exp289_sampling")
    config = load_config()
    frame = synthetic_horizontal(rows=97, known_rows=49)[list(module.TARGET_SAFE_COLUMNS)]
    indices = module.sampled_row_indices(frame, config)
    assert 48 in indices
    assert 96 in indices
    assert 0 in indices
    assert np.array_equal(indices, np.unique(indices))


def test_source_loader_excludes_all_missing_ancc_but_fails_on_partial(
    tmp_path: Path,
) -> None:
    module = load_module(name="exp289_missing_source_formation")
    config = load_config()
    valid_path = tmp_path / "valid__horizontal_well.csv"
    missing_path = tmp_path / "missing__horizontal_well.csv"
    partial_path = tmp_path / "partial__horizontal_well.csv"
    valid = synthetic_horizontal()
    valid.to_csv(valid_path, index=False)
    valid.assign(ANCC=np.nan).to_csv(missing_path, index=False)
    partial = valid.copy()
    partial.loc[10, "ANCC"] = np.nan
    partial.to_csv(partial_path, index=False)

    assert module.load_source_horizontal(missing_path).empty
    nodes = module.build_source_nodes(
        {"missing": missing_path, "valid": valid_path}, ["missing", "valid"], config
    )
    assert set(nodes["well_id"]) == {"valid"}
    with pytest.raises(ValueError, match="partially non-finite outer-train ANCC"):
        module.load_source_horizontal(partial_path)


def test_cross_well_neighbor_order_is_stable() -> None:
    module = load_module(name="exp289_graph")
    config = deepcopy(load_config())
    points = []
    wells = []
    rows = []
    for well_index, well in enumerate(("a", "b", "c", "d")):
        for row in range(4):
            points.append([10.0 * well_index, float(row)])
            wells.append(well)
            rows.append(row)
    xy = np.asarray(points, dtype=float)
    rows_array = np.asarray(rows, dtype=np.int64)
    first_indices, first_distances = module.query_cross_well_neighbors(
        xy,
        wells,
        xy,
        wells,
        rows_array,
        config,
        exclude_same_well=True,
    )
    second_indices, second_distances = module.query_cross_well_neighbors(
        xy,
        wells,
        xy,
        wells,
        rows_array,
        config,
        exclude_same_well=True,
    )
    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_array_equal(first_distances, second_distances)
    assert first_indices.shape == (16, 12)
    for row_index, selected in enumerate(first_indices):
        assert all(wells[value] != wells[row_index] for value in selected)


def test_target_free_freeze_precedes_exp226_truth_attachment(tmp_path: Path) -> None:
    module = load_module(name="exp289_freeze")
    config = deepcopy(load_config())
    config["validation"]["expected_rows"] = 8
    config["validation"]["expected_wells"] = 2
    graph = pd.DataFrame(
        {
            "fold": [0, 1],
            "target_forbidden_column_hits": [0, 0],
            "source_target_overlap": [0, 0],
            "truth_access_before_risk_freeze": [0, 0],
        }
    )
    node = pd.DataFrame(
        {
            "fold": [0, 1],
            "well_id": ["a", "b"],
            "MD": [1.0, 1.0],
            "row_idx": [1, 1],
            "fault_risk": [0.2, 0.8],
        }
    )
    well = pd.DataFrame(
        {
            "fold": [0, 1],
            "well_id": ["a", "b"],
            "suffix_fault_risk_p90": [0.2, 0.8],
            "truth_attached": [False, False],
        }
    )
    frozen = module.freeze_target_free_outputs(graph, node, well)
    assert set(frozen) == set(module.FROZEN_HASH_KEYS)
    oof = pd.DataFrame(
        {
            "well_id": ["a"] * 4 + ["b"] * 4,
            "fold": [0] * 4 + [1] * 4,
            "error": [1.0] * 4 + [12.0] * 4,
        }
    )
    oof_path = tmp_path / "oof.csv.gz"
    oof.to_csv(oof_path, index=False, compression="gzip")
    with pytest.raises(ValueError, match="frozen content SHA"):
        module.load_exp226_bias_readout(oof_path, {"a": 0, "b": 1}, well, config, frozen_hashes={})
    merged, metrics = module.load_exp226_bias_readout(
        oof_path, {"a": 0, "b": 1}, well, config, frozen_hashes=frozen
    )
    assert merged.loc[merged["well_id"].eq("b"), "abs_bias_ge_10"].item()
    assert set(metrics["scope"]) == {"overall", "fold_0", "fold_1"}


def test_formation_identity_is_post_freeze_and_exact(tmp_path: Path) -> None:
    module = load_module(name="exp289_identity")
    paths = {}
    fold_by_well = {}
    for index, well in enumerate(("a", "b")):
        path = tmp_path / f"{well}__horizontal_well.csv"
        synthetic_horizontal(rows=40, known_rows=20).to_csv(path, index=False)
        paths[well] = path
        fold_by_well[well] = index
    with pytest.raises(ValueError, match="frozen content SHA"):
        module.build_formation_identity_audit(paths, fold_by_well, frozen_hashes={})
    frozen = {key: key for key in module.FROZEN_HASH_KEYS}
    audit = module.build_formation_identity_audit(paths, fold_by_well, frozen_hashes=frozen)
    overall = audit.loc[audit["scope"].eq("overall")]
    assert len(overall) == 6
    assert np.allclose(overall["identity_error_rmse"], 0.0, atol=1e-12)
    assert np.allclose(overall["delta_correlation"], 1.0, atol=1e-12)


def test_risk_transform_and_guard_direction() -> None:
    module = load_module(name="exp289_guard")
    risk = module.risk_transform(np.asarray([0.0, 4.0, 8.0]), 0.0, 1.0, 4.0)
    assert risk[0] == pytest.approx(0.0)
    assert risk[1] == pytest.approx(0.5)
    assert risk[2] > risk[1]
    assert module.count_true_episodes(np.asarray([False, True, True, False, True])) == 2


def test_inference_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp289_inference")
    contract = module.validate_disabled_inference(load_config())
    assert contract["inference_enabled"] is False
    assert contract["create_submission"] is False
    with pytest.raises(RuntimeError, match="Stage 0 fault-topology association"):
        module.fail_closed()

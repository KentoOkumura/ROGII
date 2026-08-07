from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp301_gauge_invariant_multiformation_edge_potential"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp301_gauge_invariant_multiformation_edge_potential_compact_selfcontained_train.py"
)


def load_module(name: str = "exp301_train"):
    previous = os.environ.get("EXP301_IMPORT_ONLY")
    os.environ["EXP301_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, TRAIN_SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP301_IMPORT_ONLY", None)
        else:
            os.environ["EXP301_IMPORT_ONLY"] = previous


EXP301 = load_module()


@pytest.fixture
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_raw(rows: int = 33, known_rows: int = 16) -> pd.DataFrame:
    index = np.arange(rows, dtype=np.float64)
    z = -1000.0 - index
    tvt = 1200.0 + 1.1 * index + 0.0001 * np.square(index)
    u = z + tvt
    tvt_input = tvt.copy()
    tvt_input[known_rows:] = np.nan
    return pd.DataFrame(
        {
            "MD": index,
            "X": 1000.0 + 10.0 * index,
            "Y": 2000.0 + 3.0 * index,
            "Z": z,
            "ANCC": u + 10.0,
            "ASTNU": u + 20.0,
            "ASTNL": u + 30.0,
            "EGFDU": u + 40.0,
            "EGFDL": u + 50.0,
            "BUDA": u + 60.0,
            "TVT": tvt,
            "GR": 80.0 + np.sin(index),
            "TVT_input": tvt_input,
        }
    )


def test_config_fixes_one_variant_zero_boosters_and_separate_run_gate(config: dict) -> None:
    EXP301.validate_execution_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["execution"]["implementation"] is True
    assert config["execution"]["implementation_authorized"] is True
    assert config["execution"]["kaggle_execution_authorized"] is False
    assert (
        config["execution"]["kaggle_execution_authorization_source"]
        == "user_message_run_exp301"
    )
    assert config["execution"]["kaggle_run_completed"] is True
    assert config["outcome"]["stage0_passed"] is False
    assert config["outcome"]["stage1_executed"] is False
    assert config["execution"]["active_scientific_variants_if_implemented"] == 1
    assert config["execution"]["outer_evaluation_folds_if_implemented"] == 5
    assert config["execution"]["inner_lambda_candidates_per_outer_fold"] == 3
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["control_or_parent_retraining"] is False
    assert config["execution"]["inference"] is False
    assert config["execution"]["submission"] is False


def test_valid_column_poison_does_not_enter_safe_loader(tmp_path: Path) -> None:
    path = tmp_path / "well-a__horizontal_well.csv"
    first = synthetic_raw()
    first.to_csv(path, index=False)
    safe_first = EXP301.load_query_safe_horizontal(path)
    first_hash = EXP301.safe_query_content_sha256(safe_first)

    poisoned = first.copy()
    poisoned["TVT"] = 1.0e12 + np.arange(len(poisoned))
    poisoned["GR"] = -1.0e12
    for position, column in enumerate(EXP301.FORMATION_COLUMNS):
        poisoned[column] = (position + 1) * 1.0e9
    poisoned.to_csv(path, index=False)
    safe_second = EXP301.load_query_safe_horizontal(path)

    pd.testing.assert_frame_equal(safe_first, safe_second)
    assert EXP301.safe_query_content_sha256(safe_second) == first_hash
    assert not EXP301.FORBIDDEN_QUERY_COLUMNS.intersection(safe_second.columns)
    with pytest.raises(ValueError, match="rejected forbidden"):
        EXP301.load_query_safe_horizontal(path, (*EXP301.QUERY_RAW_COLUMNS, "TVT"))


def test_prepare_outer_fold_passes_filtered_edge_frame_to_constraint_builder(
    tmp_path: Path, config: dict
) -> None:
    donor_path = tmp_path / "donor__horizontal_well.csv"
    query_path = tmp_path / "query__horizontal_well.csv"
    synthetic_raw().to_csv(donor_path, index=False)
    synthetic_raw().to_csv(query_path, index=False)

    prepared = EXP301.prepare_outer_fold(
        0,
        {"donor": 1, "query": 0},
        {"donor": donor_path, "query": query_path},
        config,
    )

    assert isinstance(prepared.edges, pd.DataFrame)
    assert {"x_start", "y_start", "x_end", "y_end", "response", "scale"}.issubset(
        prepared.edges.columns
    )
    assert prepared.edges["solver_eligible"].all()
    assert prepared.edge_matrix.shape[0] == len(prepared.edges)


def test_formation_permutation_invariance() -> None:
    raw = synthetic_raw(rows=65, known_rows=32)
    frame = raw[list(EXP301.SOURCE_RAW_COLUMNS)].copy()
    frame.insert(0, "row_index", np.arange(len(frame), dtype=np.int64))
    original = EXP301.build_well_edges("well", frame, stride=16, minimum_finite=3)

    permuted = frame.copy()
    values = permuted[list(EXP301.FORMATION_COLUMNS)].to_numpy(copy=True)
    permuted[list(EXP301.FORMATION_COLUMNS)] = values[:, [5, 2, 4, 1, 3, 0]]
    second = EXP301.build_well_edges("well", permuted, stride=16, minimum_finite=3)
    np.testing.assert_array_equal(original["response"], second["response"])
    np.testing.assert_array_equal(original["scale"], second["scale"])
    np.testing.assert_array_equal(
        original["solver_eligible"], second["solver_eligible"]
    )


def _affine_problem():
    trajectory = np.array(
        [[0.25, 0.25], [2.25, 0.25], [2.25, 2.25], [0.25, 2.25], [0.25, 0.25]],
        dtype=np.float64,
    )
    grid = EXP301.build_active_grid([trajectory], spacing=1.0)
    start = trajectory[:-1]
    end = trajectory[1:]
    response = 2.0 * (end[:, 0] - start[:, 0]) - 3.0 * (end[:, 1] - start[:, 1])
    edges = pd.DataFrame(
        {
            "x_start": start[:, 0],
            "y_start": start[:, 1],
            "x_end": end[:, 0],
            "y_end": end[:, 1],
        }
    )
    matrix = EXP301.edge_constraint_matrix(grid, edges)
    regularizer = EXP301.build_second_difference_matrix(grid)
    gauge = EXP301.build_gauge_matrix(grid)
    result = EXP301.solve_huber_potential(
        matrix,
        response,
        np.ones(len(response), dtype=np.float64),
        regularizer,
        gauge,
        lambda_value=0.1,
        huber_delta=100.0,
        maximum_iterations=20,
        relative_tolerance=1.0e-9,
    )
    return trajectory, grid, result


def test_affine_field_recovery() -> None:
    trajectory, grid, result = _affine_problem()
    predicted = EXP301.evaluate_bilinear(
        grid, result.potential, trajectory[:, 0], trajectory[:, 1]
    )
    expected = 2.0 * trajectory[:, 0] - 3.0 * trajectory[:, 1]
    np.testing.assert_allclose(
        predicted - predicted[0], expected - expected[0], rtol=0.0, atol=1.0e-8
    )
    assert result.converged


def test_gauge_shift_invariance() -> None:
    trajectory, grid, result = _affine_problem()
    base = EXP301.evaluate_bilinear(
        grid, result.potential, trajectory[:, 0], trajectory[:, 1]
    )
    shifted = EXP301.evaluate_bilinear(
        grid, result.potential + 1234.5, trajectory[:, 0], trajectory[:, 1]
    )
    np.testing.assert_allclose(
        shifted - shifted[0], base - base[0], rtol=0.0, atol=1.0e-10
    )


def test_outer_fold_and_same_name_exclusion() -> None:
    fold_map = {"a": 0, "b": 0, "c": 1, "d": 2, "e": 3, "f": 4}
    donors = EXP301.donor_wells_for_fold(fold_map, 0)
    assert donors == ["c", "d", "e", "f"]
    assert not set(donors).intersection({"a", "b"})
    assert EXP301.query_wells_for_fold(fold_map, 0) == ["a", "b"]
    assert EXP301.inference_donor_wells(["a", "b", "c"], "b") == ["a", "c"]


def test_query_component_without_donor_fails_coverage() -> None:
    donor_trajectory = np.array([[0.25, 0.25], [1.25, 0.25]])
    query_trajectory = np.array([[100.25, 100.25], [101.25, 100.25]])
    grid = EXP301.build_active_grid([donor_trajectory, query_trajectory], spacing=1.0)
    edges = pd.DataFrame(
        {
            "x_start": [0.25],
            "y_start": [0.25],
            "x_end": [1.25],
            "y_end": [0.25],
        }
    )
    matrix = EXP301.edge_constraint_matrix(grid, edges)
    query = pd.DataFrame(
        {
            "well_id": ["q", "q"],
            "row_index": [0, 1],
            "MD": [0.0, 1.0],
            "X": query_trajectory[:, 0],
            "Y": query_trajectory[:, 1],
            "Z": [-1000.0, -1001.0],
            "TVT_input": [1200.0, np.nan],
        }
    )
    basis, donor, rows, supported = EXP301.query_component_coverage(
        grid, matrix, {"q": query}
    )
    assert basis == 1.0
    assert donor == 0.0
    assert rows == 2
    assert supported == 0
    assert EXP301.active_component_donor_coverage(grid, matrix) < 1.0


def test_content_sha_is_stable_and_order_sensitive() -> None:
    frame = pd.DataFrame({"well": ["a", "b"], "value": [1.0, 2.0]})
    first = EXP301.frame_content_sha256(frame)
    second = EXP301.frame_content_sha256(frame.copy())
    reordered = EXP301.frame_content_sha256(frame.iloc[::-1].reset_index(drop=True))
    assert first == second
    assert first != reordered
    array_first = EXP301.array_content_sha256(np.arange(5, dtype=np.int64), context=("x",))
    array_second = EXP301.array_content_sha256(np.arange(5, dtype=np.int64), context=("x",))
    assert array_first == array_second


def test_h512_add_one_diagnostic_does_not_persist_oracle_prediction(
    tmp_path: Path, config: dict
) -> None:
    local = copy.deepcopy(config)
    local["candidate_novelty_audit"]["block_horizon_rows"] = 2
    keys = pd.DataFrame(
        {
            "id": ["a_0", "a_1", "b_0", "b_1"],
            "well": ["a", "a", "b", "b"],
            "well_row_idx": [0, 1, 0, 1],
            "outer_fold": [0, 0, 1, 1],
            "md_since": [1.0, 2.0, 1.0, 2.0],
        }
    )
    values_path = tmp_path / "bank.f32"
    values = np.memmap(values_path, mode="w+", dtype="float32", shape=(4, 12))
    values[:] = 10.0
    values.flush()
    bank = EXP301.CandidateBank(
        keys=keys,
        candidate_ids=tuple(local["candidate_novelty_audit"]["candidate_order"]),
        values=values,
        values_path=values_path,
        manifest_path=tmp_path / "manifest.json",
        candidate_content_sha256="stable",
        input_evidence=[],
    )
    oof = pd.DataFrame(
        {
            "id": keys["id"],
            "well_id": keys["well"],
            "row_index": keys["well_row_idx"],
            "outer_fold": keys["outer_fold"],
            "MD": keys["md_since"],
            "md_since": keys["md_since"],
            "tvt_pred_exp301": np.zeros(4),
        }
    )
    metrics, summary = EXP301.evaluate_candidate_novelty(
        bank, oof, np.zeros(4), local
    )
    assert summary["oracle_rmse_improvement"] == 10.0
    assert summary["strict_unique_best_block_fraction"] == 1.0
    assert summary["oracle_prediction_persisted"] is False
    assert set(metrics["scope"]) == {"pooled", "fold_0", "fold_1", "fold_2", "fold_3", "fold_4"}

from __future__ import annotations

import os
import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

EXPERIMENT_DIR = Path(
    "experiments/exp383_all_tvt_stratigraphic_vector_drift_field"
)
TRAIN_SOURCE = EXPERIMENT_DIR / (
    "exp383_all_tvt_stratigraphic_vector_drift_field_"
    "compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXPERIMENT_DIR / (
    "exp383_all_tvt_stratigraphic_vector_drift_field_"
    "compact_selfcontained_inference.py"
)
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"


def load_namespace(path: Path = TRAIN_SOURCE) -> dict[str, object]:
    previous = os.environ.get("EXP383_IMPORT_ONLY")
    os.environ["EXP383_IMPORT_ONLY"] = "1"
    try:
        return runpy.run_path(str(path))
    finally:
        if previous is None:
            os.environ.pop("EXP383_IMPORT_ONLY", None)
        else:
            os.environ["EXP383_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def module() -> dict[str, object]:
    return load_namespace()


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(value, dict)
    return value


def test_execution_contract_records_failed_preflight_and_blocks_rerun(
    module: dict[str, object],
    config: dict,
) -> None:
    module["validate_execution_contract"](config, require_kaggle_authorization=False)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["experiment"]["status"] == "stage0_resource_fail_closed"
    assert config["execution"]["implementation_authorized"] is True
    assert (
        config["execution"]["implementation_approval_source"]
        == "user_message_implement_exp383_2026_07_24"
    )
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["canonical_inference_notebook_adopted"] is True
    assert config["execution"]["kaggle_execution_authorized"] is False
    assert config["execution"]["full_run_authorized"] is True
    assert config["execution"]["full_run_effective"] is False
    assert config["execution"]["current_mode"] == "stage0_resource_preflight"
    assert (
        config["execution"]["preflight_result"]["status"]
        == "failed_code_and_resource_gate"
    )
    assert config["runtime"]["scientific_candidates"] == 1
    assert config["runtime"]["reporting_folds"] == 5
    assert config["runtime"]["fitted_models"] == 0
    assert config["runtime"]["hmm_runs"] == 0
    assert config["runtime"]["pf_runs"] == 0
    assert config["runtime"]["beam_runs"] == 0
    assert config["runtime"]["lightgbm_boosters"] == 0
    assert config["runtime"]["replay_parent_control"] is False
    with pytest.raises(RuntimeError, match="Kaggle execution is not authorized"):
        module["validate_execution_contract"](
            config, require_kaggle_authorization=True
        )
    full = deepcopy(config)
    full["execution"]["current_mode"] = "full_run"
    with pytest.raises(RuntimeError, match="Kaggle execution is not authorized"):
        module["validate_execution_contract"](
            full, require_kaggle_authorization=True
        )


def test_fold_assignment_is_stable_and_parent_parity_is_checked(
    module: dict[str, object],
) -> None:
    wells = [f"w{index:03d}" for index in range(31)]
    first = module["assign_group_folds"](wells, 5, 42)
    second = module["assign_group_folds"](list(reversed(wells)), 5, 42)
    assert first == second
    parent = pd.DataFrame(
        [
            {
                "well_id": well,
                "fold": fold,
                "row_idx": 0,
                "suffix_offset": 0,
                "tvt_pred": 1.0,
            }
            for well, fold in first.items()
        ]
    )
    module["validate_fold_identity"](first, parent)
    broken = parent.copy()
    broken.loc[0, "fold"] = (int(broken.loc[0, "fold"]) + 1) % 5
    with pytest.raises(ValueError, match="fold identity mismatch"):
        module["validate_fold_identity"](first, broken)


def test_role_read_guard_rejects_valid_truth_and_formation(
    module: dict[str, object],
) -> None:
    ledger = module["RoleReadLedger"]()
    ledger.record_target(["MD", "X", "Y", "Z", "TVT_input"], 10)
    assert ledger.target_safe_rows == 10
    with pytest.raises(ValueError, match="forbidden columns"):
        ledger.record_target(["MD", "TVT", "ANCC"], 10)
    with pytest.raises(ValueError, match="leaked"):
        ledger.record_role_overlap(["source", "target"], ["target"])


def _surface_points(module: dict[str, object], count: int = 25) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(count):
        x = float((index % 5) * 100.0)
        y = float((index // 5) * 100.0)
        row: dict[str, object] = {
            "fold": 0,
            "role": "outer_train",
            "well_id": f"w{index:02d}",
            "row_idx": index,
            "MD": float(index),
            "X": x,
            "Y": y,
        }
        for formation_index, name in enumerate(module["FORMATION_NAMES"]):
            row[name] = (
                1000.0
                + 10.0 * formation_index
                + 0.5 * x
                - 0.25 * y
            )
        rows.append(row)
    return pd.DataFrame(rows)


def test_surface_plane_recovers_value_and_self_excludes(
    module: dict[str, object],
    config: dict,
) -> None:
    points = _surface_points(module)
    query = pd.Series(
        {
            "fold": 0,
            "role": "outer_train",
            "well_id": "w00",
            "row_idx": 99,
            "query_id": "q0",
            "MD": 99.0,
            "X": 175.0,
            "Y": 125.0,
        }
    )
    distance = np.sqrt(
        np.square(points["X"].to_numpy(float) - float(query["X"]))
        + np.square(points["Y"].to_numpy(float) - float(query["Y"]))
    )
    result = module["fit_surface_record"](points, distance, query, config)
    assert result is not None
    assert result["surface_unique_wells"] == 24
    expected = 1000.0 + 0.5 * 175.0 - 0.25 * 125.0
    assert result["surface_ANCC"] == pytest.approx(expected, abs=0.1)
    assert result["surface_grad_x_ANCC"] == pytest.approx(0.5, abs=1.0e-3)
    assert result["surface_grad_y_ANCC"] == pytest.approx(-0.25, abs=1.0e-3)


def test_all_tvt_windows_recover_fixed_planar_rate(
    module: dict[str, object],
    config: dict,
) -> None:
    md = np.arange(0.0, 1200.0)
    s_true = 200.0 + 0.4 * md
    frame = pd.DataFrame(
        {
            "fold": 0,
            "role": "outer_train",
            "well_id": "source",
            "row_idx": np.arange(len(md)),
            "MD": md,
            "X": 10.0 + 0.2 * md,
            "Y": 30.0 - 0.1 * md,
            "Z": -1000.0 - 0.8 * md,
            "TVT": s_true - (-1000.0 - 0.8 * md),
        }
    )
    windows = module["build_well_donor_windows"](frame, config)
    assert set(windows["window_scale_ft"].unique()) == {64.0, 256.0, 1024.0}
    assert np.median(windows["rate_true"]) == pytest.approx(0.4, abs=1.0e-10)
    assert np.median(windows["tangent_x"]) == pytest.approx(0.2, abs=1.0e-10)
    assert np.median(windows["tangent_y"]) == pytest.approx(-0.1, abs=1.0e-10)


def test_multiscale_surface_join_uses_unique_donor_node_id(
    module: dict[str, object],
) -> None:
    donors = pd.DataFrame(
        {
            "fold": [0, 0],
            "role": ["outer_train", "outer_train"],
            "well_id": ["source", "source"],
            "row_idx": [64, 64],
            "MD": [64.0, 64.0],
            "window_scale_ft": [64.0, 256.0],
        }
    )
    donors = module["attach_donor_query_ids"](donors)
    assert donors["query_id"].nunique() == 2
    surface = donors[
        ["fold", "role", "well_id", "row_idx", "query_id", "MD"]
    ].copy()
    surface["surface_available"] = True
    surface["surface_ANCC"] = [100.0, 200.0]
    joined = module["attach_surface_results_to_donors"](donors, surface)
    assert len(joined) == len(donors)
    assert joined["surface_ANCC"].tolist() == [100.0, 200.0]


def _surface_frame(module: dict[str, object], rows: int) -> pd.DataFrame:
    frame = pd.DataFrame(index=np.arange(rows))
    for index, name in enumerate(module["FORMATION_NAMES"]):
        frame[f"surface_{name}"] = 100.0 + index * 10.0 + np.arange(rows)
        frame[f"surface_grad_x_{name}"] = 0.5 + index * 0.01
        frame[f"surface_grad_y_{name}"] = -0.2 + index * 0.01
        frame[f"surface_variance_{name}"] = 1.0 + index
    return frame


def test_signature_is_exactly_29_dimensions(
    module: dict[str, object],
) -> None:
    frame = _surface_frame(module, 4)
    signature = module["raw_signature_matrix"](
        frame, np.asarray([300.0, 301.0, 302.0, 303.0])
    )
    assert signature.shape == (4, 29)
    assert np.isfinite(signature).all()


def _vector_donors(module: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(32):
        x = float((index % 8) * 100)
        y = float((index // 8) * 100)
        absolute = 200.0 + 1.5 * x - 0.75 * y
        row: dict[str, object] = {
            "fold": 0,
            "role": "outer_train",
            "well_id": f"w{index:02d}",
            "row_idx": index,
            "query_id": f"d{index}",
            "MD": float(index),
            "X": x,
            "Y": y,
            "Z": -1000.0,
            "window_scale_ft": 256.0,
            "S_true": absolute,
            "tangent_x": 1.0,
            "tangent_y": 0.0,
            "rate_true": 1.5,
            "window_residual_variance": 1.0,
        }
        for column in module["SIGNATURE_COLUMNS"]:
            row[column] = 0.0
        for formation_index, name in enumerate(module["FORMATION_NAMES"]):
            row[f"surface_{name}"] = (
                50.0 + formation_index * 10.0 + 0.25 * x + 0.1 * y
            )
            row[f"surface_grad_x_{name}"] = 0.25
            row[f"surface_grad_y_{name}"] = 0.1
            row[f"surface_variance_{name}"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def _vector_query(module: dict[str, object]) -> pd.Series:
    x = 275.0
    y = 125.0
    row: dict[str, object] = {
        "fold": 0,
        "role": "outer_valid",
        "well_id": "target",
        "row_idx": 1,
        "query_id": "q0",
        "MD": 64.0,
        "X": x,
        "Y": y,
        "Z": -1000.0,
        "TVT_input": np.nan,
        "tangent_x": 1.0,
        "tangent_y": 0.0,
        "base_path_s": 100.0,
        "fallback_rate": 0.5,
    }
    for column in module["SIGNATURE_COLUMNS"]:
        row[column] = 0.0
    for formation_index, name in enumerate(module["FORMATION_NAMES"]):
        row[f"surface_{name}"] = (
            50.0 + formation_index * 10.0 + 0.25 * x + 0.1 * y
        )
        row[f"surface_grad_x_{name}"] = 0.25
        row[f"surface_grad_y_{name}"] = 0.1
        row[f"surface_variance_{name}"] = 1.0
    return pd.Series(row)


def test_vector_field_recovers_planar_absolute_and_rate(
    module: dict[str, object],
    config: dict,
) -> None:
    donors = _vector_donors(module)
    query = _vector_query(module)
    distance = np.sqrt(
        np.square(donors["X"].to_numpy(float) - float(query["X"]))
        + np.square(donors["Y"].to_numpy(float) - float(query["Y"]))
    )
    result = module["fit_vector_field_record"](
        donors,
        distance,
        query,
        config,
        exclude_same_well=False,
    )
    assert result is not None
    expected = 200.0 + 1.5 * float(query["X"]) - 0.75 * float(query["Y"])
    assert result["field_absolute_s"] == pytest.approx(expected, abs=0.1)
    assert result["field_rate"] == pytest.approx(1.5, abs=2.0e-3)
    assert result["field_unique_wells"] >= 24
    assert result["field_support_ess"] > 0.0


def _calibration_query() -> pd.DataFrame:
    md = np.arange(0.0, 641.0, 64.0)
    return pd.DataFrame(
        {
            "fold": 0,
            "role": "outer_valid",
            "well_id": "target",
            "row_idx": np.arange(len(md)),
            "query_id": [f"q{index}" for index in range(len(md))],
            "MD": md,
            "Z": -1000.0,
            "TVT_input": np.where(md <= 128.0, 1103.0 + 0.5 * md, np.nan),
            "field_available": True,
            "field_absolute_s": 100.0 + 0.5 * md,
            "field_rate": 0.5,
            "field_absolute_variance": 1.0,
            "field_rate_variance": 1.0,
            "field_support_ess": 32.0,
            "field_unique_wells": 24.0,
            "field_condition_number": 1.0e4,
            "field_surface_variance": 1.0,
            "surface_variance_reference": 10.0,
            "base_path_s": 120.0 + 0.5 * md,
            "fallback_rate": 0.25,
        }
    )


def test_prefix_calibration_uses_every_finite_prefix_row(
    module: dict[str, object],
    config: dict,
) -> None:
    query = _calibration_query()
    md = np.arange(0.0, 200.0)
    target = pd.DataFrame(
        {
            "well_id": "target",
            "MD": md,
            "Z": -1000.0,
            "TVT_input": 1103.0 + 0.5 * md,
        }
    )
    calibrated, record = module["calibrate_prefix_for_well"](
        query, target, config
    )
    assert record["known_prefix_rows"] == 200
    assert record["prefix_bias_ft"] == pytest.approx(3.0, abs=1.0e-10)
    assert np.allclose(
        calibrated["calibrated_absolute_s"],
        calibrated["field_absolute_s"] + 3.0,
    )
    assert (calibrated["field_confidence"] > 0).all()


def test_path_solver_hard_constrains_known_query_rows(
    module: dict[str, object],
    config: dict,
) -> None:
    query = _calibration_query()
    query["calibrated_absolute_s"] = 103.0 + 0.5 * query["MD"]
    query["final_rate"] = 0.5
    solution, status = module["solve_path_for_well"](query, config)
    assert status == "vector_field"
    known = np.isfinite(query["TVT_input"])
    expected_known = query.loc[known, "TVT_input"] + query.loc[known, "Z"]
    assert np.allclose(solution[known], expected_known)
    assert np.allclose(solution, 103.0 + 0.5 * query["MD"], atol=1.0e-7)


def test_path_solver_fails_closed_to_exp226_for_field_gap(
    module: dict[str, object],
    config: dict,
) -> None:
    query = _calibration_query()
    query["calibrated_absolute_s"] = query["field_absolute_s"]
    query["final_rate"] = 0.5
    query.loc[3, "field_available"] = False
    solution, status = module["solve_path_for_well"](query, config)
    assert status == "coverage_exp226_fallback"
    assert np.array_equal(solution, query["base_path_s"].to_numpy(float))


def test_target_free_hash_is_order_stable_and_truth_join_requires_freeze(
    module: dict[str, object],
) -> None:
    frame = pd.DataFrame(
        {
            "fold": [0, 0],
            "well_id": ["b", "a"],
            "row_idx": [2, 1],
            "value": [1.0, 2.0],
        }
    )
    first = module["frame_content_sha256"](
        frame, sort_columns=("fold", "well_id", "row_idx")
    )
    second = module["frame_content_sha256"](
        frame.iloc[::-1], sort_columns=("fold", "well_id", "row_idx")
    )
    assert first == second
    prediction = pd.DataFrame(
        {
            "fold": [0],
            "well_id": ["a"],
            "row_idx": [1],
            "MD": [10.0],
            "Z": [-1000.0],
            "exp226_prediction": [1100.0],
            "exp383_prediction": [1099.0],
            "exp383_path_status": ["vector_field"],
            "distance_from_anchor": [1000.0],
        }
    )
    truth = pd.DataFrame(
        {
            "fold": [0],
            "well_id": ["a"],
            "row_idx": [1],
            "tvt_true": [1098.0],
        }
    )
    roles = pd.DataFrame(
        {
            "well_id": ["a"],
            "verification_like_spatial_role": ["valid"],
            "verification_like_typewell_purged_role": ["train"],
        }
    )
    with pytest.raises(ValueError, match="before target-free"):
        module["late_join_truth"](
            prediction,
            truth,
            roles,
            {},
            module["RoleReadLedger"](),
        )
    joined = module["late_join_truth"](
        prediction,
        truth,
        roles,
        {"prediction": "a" * 64},
        module["RoleReadLedger"](),
    )
    assert bool(joined["hidden_like_spatial"].iloc[0])
    assert not bool(joined["hidden_like_typewell_purged"].iloc[0])


def test_stage1_gate_uses_direct_control_and_fixed_scopes(
    module: dict[str, object],
    config: dict,
) -> None:
    rows = []
    for fold in range(5):
        for index in range(4):
            truth = 100.0 + index
            rows.append(
                {
                    "fold": fold,
                    "well_id": f"w{fold}",
                    "row_idx": index,
                    "tvt_true": truth,
                    "exp226_prediction": truth + 2.0,
                    "exp383_prediction": truth + (0.5 if index % 2 == 0 else -0.5),
                    "distance_from_anchor": 1100.0 if index >= 2 else 100.0,
                    "hidden_like_spatial": True,
                    "hidden_like_typewell_purged": True,
                }
            )
    scored = pd.DataFrame(rows)
    metrics, by_well, tail = module["build_stage1_readout"](scored)
    result = module["evaluate_stage1"](metrics, scored, config)
    assert result["passed"]
    assert result["observed"]["positive_folds"] == 5
    assert len(by_well) == 5
    assert tail["improved_wells"] == 5


def test_inference_remains_fail_closed(config: dict) -> None:
    inference = load_namespace(INFERENCE_SOURCE)
    contract = inference["validate_disabled_inference"](config)
    assert contract["inference_enabled"] is False
    assert contract["submission_enabled"] is False
    with pytest.raises(RuntimeError, match="fail-closed"):
        inference["run_inference"]()


def test_contract_rejects_accidental_control_replay(
    module: dict[str, object],
    config: dict,
) -> None:
    broken = deepcopy(config)
    broken["runtime"]["replay_parent_control"] = True
    with pytest.raises(ValueError, match="must not be regenerated"):
        module["validate_execution_contract"](
            broken, require_kaggle_authorization=False
        )

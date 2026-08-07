from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT_NAME = "exp508_exp413_public_trajectory_postprocess_audit"
SOURCE = HERE / f"{EXPERIMENT_NAME}_compact_selfcontained_train.py"


def load_source() -> ModuleType:
    previous = os.environ.get("EXP508_IMPORT_ONLY")
    os.environ["EXP508_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("exp508_train", SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP508_IMPORT_ONLY", None)
        else:
            os.environ["EXP508_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def train() -> ModuleType:
    return load_source()


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_static_contract_records_completed_cpu_run_without_inference_approval(
    train: ModuleType, config: dict
) -> None:
    contract = yaml.safe_load((HERE / "postprocess_contract.yaml").read_text())
    observed = train.validate_static_contract(config, contract)
    assert config["experiment"]["route"] == "ml_model"
    assert config["experiment"]["status"] == "stage_a_complete_fail_closed"
    assert config["authorization"]["implementation_approved"] is True
    assert config["authorization"]["canonical_notebook_adoption_approved"] is True
    assert config["authorization"]["kaggle_package_approved"] is True
    assert config["authorization"]["kaggle_run_approved"] is True
    assert config["authorization"]["inference_implementation_approved"] is False
    assert config["authorization"]["submission_approved"] is False
    assert config["execution"]["stage_a"] == {
        "run_approved": True,
        "enabled": True,
        "completed": True,
        "passed": False,
        "decision": "FAIL_CLOSE_WITHOUT_SG_GRID_WARMUP_ROUTER_OR_GATE_RESCUE",
        "import_only_env": "EXP508_IMPORT_ONLY",
    }
    assert observed["cost"] == {
        "scientific_primary_variants": 1,
        "report_only_controls": 2,
        "trained_models": 0,
        "lightgbm_configs": 0,
        "total_boosters": 0,
        "hmm_runs": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "parent_or_control_retraining": 0,
        "gpu_runs": 0,
    }


def test_stage_a_authorization_is_fail_closed(train: ModuleType, config: dict) -> None:
    with pytest.raises(RuntimeError, match="Kaggle private CPU"):
        train.require_stage_a_authorization(config)
    original = train.is_kaggle_runtime
    train.is_kaggle_runtime = lambda: True
    try:
        train.require_stage_a_authorization(config)
    finally:
        train.is_kaggle_runtime = original


def test_public_source_and_postprocess_contract_are_sha_pinned(
    train: ModuleType, config: dict
) -> None:
    evidence = train.verify_public_source_and_contract(config)
    assert evidence["source"]["observed_sha256"] == (
        "39b477f3687fed5c1679fc30b82a2980906bf1910df189696065800eb2b1f3ad"
    )
    assert evidence["source"]["required_snippets_passed"] is True
    assert evidence["postprocess_contract"]["sha256"] == (
        "58afa464063e998f2c4853eb7df2a68784a226ee625a00a8cf1e328280c7dd58"
    )


def test_sg61_p3_matches_scipy_per_well_and_preserves_source_order(
    train: ModuleType,
) -> None:
    wells = pd.Series(["b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a"])
    values = np.array([0.0, 10.0, 1.0, 9.0, 4.0, 7.0, 9.0, 6.0, 16.0, 6.0, 25.0, 7.0])
    observed, audit = train.savgol_by_well(wells, values)
    expected = values.copy()
    for well in ("b", "a"):
        positions = np.flatnonzero(wells.to_numpy() == well)
        expected[positions] = savgol_filter(values[positions], 5, 3)
    assert np.allclose(observed, expected)
    assert audit["well"].tolist() == ["b", "a"]
    assert audit["effective_window"].tolist() == [5, 5]
    assert audit["filter_applied"].tolist() == [True, True]


def test_sg_short_well_rule_and_parameter_grid_rejection(train: ModuleType) -> None:
    wells = pd.Series(["short"] * 4 + ["five"] * 5)
    values = np.arange(9, dtype=np.float64) ** 2
    observed, audit = train.savgol_by_well(wells, values)
    assert np.array_equal(observed[:4], values[:4])
    assert np.allclose(observed[4:], savgol_filter(values[4:], 5, 3))
    assert audit.set_index("well").loc["short", "effective_window"] == 3
    assert not bool(audit.set_index("well").loc["short", "filter_applied"])
    with pytest.raises(ValueError, match="only the frozen SG61/p3"):
        train.savgol_by_well(wells, values, window_length=59)


def test_tau85_is_exact_and_nonselectable(train: ModuleType) -> None:
    md = np.array([0.0, 85.0, 170.0])
    last = np.array([100.0, 100.0, 100.0])
    prediction = np.array([110.0, 110.0, 110.0])
    expected = last + (1.0 - np.exp(-md / 85.0)) * (prediction - last)
    assert np.allclose(train.tau85_warmup(md, last, prediction), expected)
    assert train.tau85_warmup(md, last, prediction)[0] == last[0]
    with pytest.raises(ValueError, match="only the frozen report-only tau=85"):
        train.tau85_warmup(md, last, prediction, tau_ft=50.0)


def synthetic_oof() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        well = f"w{fold}"
        last = 100.0 + fold
        for row_idx in range(2):
            actual = last + 1.0 + row_idx
            rows.append(
                {
                    "id": f"{well}_{row_idx}",
                    "well": well,
                    "md_since": float(row_idx * 1000),
                    "last_known_tvt": last,
                    "outer_fold": fold,
                    train_prediction_column(): actual + 0.1,
                    "target": actual - last,
                    "actual_tvt": actual,
                    "error_forbidden": 0.1,
                }
            )
    return pd.DataFrame(rows)


def train_prediction_column() -> str:
    return "scale5_x1p0_full_replacement__lgb_mean__pred_tvt"


def test_truth_free_loader_uses_exact_allowlist_and_truth_late_checks_alignment(
    train: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "stage_d_oof_predictions.parquet"
    source = synthetic_oof()
    source.to_parquet(path, index=False)
    frozen, manifest = train.load_truth_free_parent_oof(
        path,
        expected_sha256=train.sha256_file(path),
        expected_rows=10,
        expected_wells=5,
        expected_folds=[0, 1, 2, 3, 4],
    )
    assert manifest["loaded_columns"] == list(train.TRUTH_FREE_OOF_COLUMNS)
    assert manifest["truth_or_error_columns_loaded"] == 0
    assert "target" not in frozen and "actual_tvt" not in frozen and "error_forbidden" not in frozen
    truth, truth_manifest = train.load_truth_late(path, frozen)
    assert np.allclose(truth, source["actual_tvt"])
    assert truth_manifest["loaded_after_prediction_freeze"] is True

    reordered = source.iloc[::-1].reset_index(drop=True)
    reordered.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="row order differs"):
        train.load_truth_late(path, frozen)


def test_truth_free_prediction_schema_and_report_only_order(train: ModuleType) -> None:
    source = synthetic_oof()
    frame = pd.DataFrame(
        {
            "id": source["id"],
            "well": source["well"],
            "row_idx": np.tile([0, 1], 5),
            "fold": source["outer_fold"].astype(np.int8),
            "md_since": source["md_since"],
            "last_known_tvt": source["last_known_tvt"],
            train.CONTROL_COLUMN: source[train_prediction_column()],
        }
    )
    predictions, manifest, audit = train.generate_truth_free_predictions(frame)
    assert tuple(predictions.columns) == train.PREDICTION_FREEZE_COLUMNS
    assert manifest["truth_or_error_columns"] == 0
    assert audit["filter_applied"].sum() == 0
    assert np.array_equal(predictions[train.PRIMARY_COLUMN], predictions[train.CONTROL_COLUMN])
    report = train.score_report_only(
        predictions,
        source["actual_tvt"].to_numpy(np.float64),
        primary_decision_sha256="frozen-primary-decision-sha",
    )
    assert report["primary_decision_sha256"] == "frozen-primary-decision-sha"
    assert report["selection_or_rescue_performed"] is False
    assert all(
        not item["selectable"] and not item["may_rescue_primary"]
        for item in report["candidates"].values()
    )


def passing_gate_inputs(train: ModuleType) -> tuple[dict, dict[str, pd.DataFrame]]:
    summary = {
        "pooled": {"gain_ft": 0.02},
        "tail": {"delta_p95_ft": 0.10, "worst_delta_ft": 0.20},
        "prediction_start": {
            "abs_correction_p95_ft": 0.40,
            "abs_correction_max_ft": 1.50,
        },
    }
    tables = {
        "fold": pd.DataFrame(
            {"delta_sg_minus_exp413": [-0.01, 0.0, -0.02, -0.01, 0.01]}
        ),
        "scope": pd.DataFrame(
            {
                "scope": list(train.FIXED_SCOPE_ORDER),
                "delta_sg_minus_exp413": [0.01, 0.00, -0.01, 0.02, 0.01],
            }
        ),
    }
    return summary, tables


def test_primary_gate_is_strict_all_and_with_fixed_fail_decision(
    train: ModuleType, config: dict
) -> None:
    summary, tables = passing_gate_inputs(train)
    passed = train.build_primary_gate(summary, tables, {"all": True}, config)
    assert passed["passed"] is True
    assert passed["inference_automatically_approved"] is False
    failed_summary = copy.deepcopy(summary)
    failed_summary["tail"]["worst_delta_ft"] = 0.2500001
    failed = train.build_primary_gate(failed_summary, tables, {"all": True}, config)
    assert failed["passed"] is False
    assert failed["decision"] == "FAIL_CLOSE_WITHOUT_SG_GRID_WARMUP_ROUTER_OR_GATE_RESCUE"
    assert failed["report_only_may_rescue_primary"] is False


def test_source_has_no_model_physics_router_or_submission_execution_path() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    forbidden_calls = (
        "LGBMRegressor(",
        "CatBoostRegressor(",
        "run_particle_filter(",
        "run_pf(",
        "run_beam(",
        "submission.csv",
        "fit_well_router(",
    )
    assert all(call not in text for call in forbidden_calls)
    assert "Path(__file__)" not in text

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE
    / "exp511_exp413_transductive_k16_neighbor_rate_postprocess_"
    "compact_selfcontained_train.py"
)


def load_source() -> ModuleType:
    previous = os.environ.get("EXP511_IMPORT_ONLY")
    os.environ["EXP511_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("exp511_train", SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP511_IMPORT_ONLY", None)
        else:
            os.environ["EXP511_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def train() -> ModuleType:
    return load_source()


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_static_contract_records_train_run_authorization(
    train: ModuleType, config: dict
) -> None:
    observed = train.validate_static_contract(config)
    assert config["experiment"]["route"] == "ensemble"
    assert config["experiment"]["status"] == "stage_a_complete_fail_closed"
    assert config["authorization"]["implementation_approved"] is True
    assert config["authorization"]["canonical_notebook_adoption_approved"] is True
    assert config["authorization"]["kaggle_package_approved"] is True
    assert config["authorization"]["kaggle_run_approved"] is True
    assert config["authorization"]["inference_implementation_approved"] is False
    assert config["authorization"]["submission_approved"] is False
    assert config["implementation"]["compact_selfcontained_candidate_created"] is True
    assert config["implementation"]["dedicated_contract_tests_created"] is True
    assert config["implementation"]["canonical_train_notebook_is_template_placeholder"] is False
    assert observed["cost"] == {
        "scientific_primary_variants": 1,
        "report_only_variants": 0,
        "trained_models": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "total_boosters": 0,
        "hmm_runs": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "parent_or_control_retraining": 0,
        "gpu_runs": 0,
    }
    assert config["execution"]["stage_a"]["run_approved"] is True
    assert config["execution"]["stage_a"]["enabled"] is True
    assert config["execution"]["stage_a"]["completed"] is True
    assert config["execution"]["stage_a"]["kernel_version"] == 4


def test_static_contract_rejects_parameter_or_authorization_drift(
    train: ModuleType, config: dict
) -> None:
    changed = copy.deepcopy(config)
    changed["postprocess"]["primary"]["alpha"] = 0.10
    with pytest.raises(ValueError, match="fixed primary changed"):
        train.validate_static_contract(changed)
    changed = copy.deepcopy(config)
    changed["authorization"]["inference_implementation_approved"] = True
    with pytest.raises(ValueError, match="train-only authorization boundary"):
        train.validate_static_contract(changed)


def test_stage_a_is_fail_closed_without_separate_run_approval(
    train: ModuleType, config: dict
) -> None:
    with pytest.raises(RuntimeError, match="Kaggle private CPU"):
        train.require_stage_a_authorization(config)
    original = train.is_kaggle_runtime
    train.is_kaggle_runtime = lambda: True
    try:
        with pytest.raises(RuntimeError, match="already complete and closed"):
            train.require_stage_a_authorization(config)
    finally:
        train.is_kaggle_runtime = original


def test_k16_basis_and_zero_intercept_coefficient_recovery(
    train: ModuleType,
) -> None:
    segment = train.exact_k16_segment_ids(32)
    np.testing.assert_array_equal(segment, np.repeat(np.arange(16), 2))
    phi = train.cumulative_rate_basis(32)
    assert phi.shape == (32, 16)
    expected = np.linspace(-0.08, 0.07, 16)
    z = -np.arange(33, dtype=np.float64)
    u = np.arange(1, 33, dtype=np.float64)
    prediction = np.concatenate(
        [[100.0], 100.0 + u + phi @ expected]
    )
    observed, observed_phi, residual, diagnostics = (
        train.solve_smoothed_coefficients(
            prediction,
            z,
            rho=0.0,
        )
    )
    np.testing.assert_allclose(observed, expected, atol=1.0e-12)
    np.testing.assert_allclose(observed_phi @ observed, residual, atol=1.0e-12)
    assert diagnostics["rank"] == 16
    with pytest.raises(ValueError, match="fixed to K16"):
        train.exact_k16_segment_ids(32, segments=12)


def make_state(train: ModuleType, *, well: str = "target") -> object:
    rows = 33
    segment = train.exact_k16_segment_ids(rows - 1)
    phi = train.cumulative_rate_basis(rows - 1)
    x = np.arange(rows, dtype=np.float64)
    y = np.zeros(rows, dtype=np.float64)
    row_idx = np.arange(rows, dtype=np.int64)
    contract = train.K16Contract(theta0_deg=0.0, minimum_unique_donor_wells=1)
    midpoint, projection, source_row = train.segment_geometry(
        x, y, row_idx, segment, contract
    )
    return train.WellK16(
        fold=0,
        well=well,
        positions=np.arange(rows, dtype=np.int64),
        row_idx=row_idx,
        prediction=np.arange(rows, dtype=np.float64),
        phi=phi,
        coefficients=np.zeros(16, dtype=np.float64),
        segment_id=segment,
        segment_mid_xy=midpoint,
        segment_projection=projection,
        segment_source_row=source_row,
    )


def test_local_linear_is_stable_and_excludes_self(train: ModuleType) -> None:
    state = make_state(train)
    rows: list[dict[str, object]] = []
    for well_index in range(10):
        donor_well = "target" if well_index == 0 else f"d{well_index:02d}"
        value = 1_000_000.0 if donor_well == "target" else 3.5
        for segment in range(16):
            rows.append(
                {
                    "fold": 0,
                    "donor_well": donor_well,
                    "donor_segment": segment,
                    "source_row": segment,
                    "x": float(segment),
                    "y": float(well_index * 10),
                    "projection": 1.0,
                    "predicted_k16_coefficient": value,
                    "normalized_field_rate": value,
                    "field_eligible": True,
                }
            )
    field = pd.DataFrame(rows, columns=train.FIELD_COLUMNS)
    contract = train.K16Contract(
        theta0_deg=0.0,
        minimum_unique_donor_wells=1,
    )
    first = train.local_linear_consensus(field, state, contract)
    shuffled = train.local_linear_consensus(
        field.sample(frac=1.0, random_state=9).reset_index(drop=True),
        state,
        contract,
    )
    np.testing.assert_allclose(first.values, 3.5, atol=1.0e-10)
    np.testing.assert_allclose(first.values, shuffled.values, atol=1.0e-12)
    assert np.all(first.self_donor_segments == 0)
    assert np.all(first.selected_segments == 50)

    extreme = field[field["donor_well"].ne("target")].copy()
    extreme["x"] += 1.0e9
    extreme["y"] += 1.0e9
    far = train.local_linear_consensus(extreme, state, contract)
    assert np.all(far.finite)
    assert np.all(np.isfinite(far.effective_segments))
    assert np.all(far.effective_segments > 0.0)


def synthetic_batch(
    train: ModuleType,
    *,
    wells: int,
    rows: int = 33,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    phi = train.cumulative_rate_basis(rows - 1)
    for well_index in range(wells):
        well = f"w{well_index:02d}"
        coefficient = np.full(16, 0.01 * well_index)
        z = -np.arange(rows, dtype=np.float64)
        u = np.arange(1, rows, dtype=np.float64)
        prediction = np.concatenate(
            [[100.0 + well_index], 100.0 + well_index + u + phi @ coefficient]
        )
        theta = np.radians(118.4)
        along = np.arange(rows, dtype=np.float64) * 10.0
        x = well_index * 100.0 + along * np.cos(theta)
        y = well_index * 30.0 + along * np.sin(theta)
        for row_idx in range(rows):
            records.append(
                {
                    "id": f"{well}_{row_idx}",
                    "well": well,
                    "row_idx": row_idx,
                    "fold": 0,
                    "pred_tvt": prediction[row_idx],
                    "X": x[row_idx],
                    "Y": y[row_idx],
                    "Z": z[row_idx],
                }
            )
    all_rows = pd.DataFrame(records)
    base = all_rows[["id", "well", "row_idx", "fold", "pred_tvt"]].copy()
    allowlist = all_rows[list(train.PREDICTION_ALLOWLIST)].copy()
    return base, allowlist


def test_small_transductive_batch_is_identity_at_fixed_support(
    train: ModuleType,
) -> None:
    base, allowlist = synthetic_batch(train, wells=3)
    predictions, field, support, manifest = train.generate_truth_free_predictions(
        base,
        allowlist,
        train.K16Contract(),
    )
    assert len(field) == 3 * 16
    assert len(support) == 3 * 16
    assert not support["supported"].any()
    assert support["unique_donor_wells"].max() == 2
    np.testing.assert_array_equal(
        predictions[train.PRIMARY_COLUMN],
        predictions[train.CONTROL_COLUMN],
    )
    np.testing.assert_array_equal(predictions[train.CORRECTION_COLUMN], 0.0)
    assert manifest["first_score_row_abs_correction_max_ft"] == 0.0


def test_supported_postprocess_preserves_start_and_cap(train: ModuleType) -> None:
    base, allowlist = synthetic_batch(train, wells=10)
    contract = train.K16Contract(
        minimum_unique_donor_wells=1,
        alpha=10.0,
        correction_cap_ft=0.25,
    )
    predictions, _field, support, _manifest = (
        train.generate_truth_free_predictions(base, allowlist, contract)
    )
    assert support["supported"].any()
    first = np.asarray(
        [
            positions[0]
            for positions in predictions.groupby("well", sort=False).indices.values()
        ]
    )
    np.testing.assert_array_equal(
        predictions.iloc[first][train.CORRECTION_COLUMN], 0.0
    )
    assert predictions[train.CORRECTION_COLUMN].abs().max() <= 0.25


def synthetic_oof(train: ModuleType) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        well = f"w{fold}"
        for row_idx in range(17):
            last = 100.0 + fold
            actual = last + row_idx
            rows.append(
                {
                    "id": f"{well}_{row_idx}",
                    "well": well,
                    "outer_fold": fold,
                    train.PREDICTION_COLUMN: actual + 0.1,
                    "md_since": float(row_idx * 100.0),
                    "last_known_tvt": last,
                    "target": actual - last,
                    "actual_tvt": actual,
                    "error_forbidden": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_truth_free_loader_is_exact_and_truth_requires_freeze(
    train: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "stage_d_oof_predictions.parquet"
    source = synthetic_oof(train)
    source.to_parquet(path, index=False)
    frozen, manifest = train.load_truth_free_parent_oof(
        path,
        expected_sha256=train.sha256_file(path),
        expected_rows=len(source),
        expected_wells=5,
        expected_folds=[0, 1, 2, 3, 4],
    )
    assert manifest["loaded_columns"] == list(train.TRUTH_FREE_OOF_COLUMNS)
    assert manifest["truth_or_error_columns_loaded"] == 0
    assert "target" not in frozen and "actual_tvt" not in frozen
    predictions = pd.DataFrame(
        {
            "id": frozen["id"],
            "well": frozen["well"],
            "row_idx": frozen["row_idx"],
            "fold": frozen["fold"],
            train.CONTROL_COLUMN: frozen["pred_tvt"],
            train.PRIMARY_COLUMN: frozen["pred_tvt"],
            train.CORRECTION_COLUMN: 0.0,
        }
    )
    with pytest.raises(ValueError, match="frozen prediction SHA"):
        train.load_truth_late(
            path,
            predictions,
            prediction_freeze_sha256="",
        )
    truth, truth_manifest = train.load_truth_late(
        path,
        predictions,
        prediction_freeze_sha256="frozen-sha",
    )
    np.testing.assert_allclose(truth["actual_tvt"], source["actual_tvt"])
    assert truth_manifest["loaded_after_prediction_freeze"] is True


def test_raw_geometry_loader_uses_only_xyz_and_preserves_order(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "train"
    raw_root.mkdir()
    rows: list[dict[str, object]] = []
    for well_index in range(2):
        well = f"g{well_index}"
        raw = pd.DataFrame(
            {
                "X": np.arange(20, dtype=np.float64) + 100.0 * well_index,
                "Y": np.arange(20, dtype=np.float64) + 200.0 * well_index,
                "Z": -np.arange(20, dtype=np.float64),
                "TVT": np.arange(20, dtype=np.float64) + 10_000.0,
                "ANCC": np.arange(20, dtype=np.float64) + 20_000.0,
            }
        )
        raw.to_csv(raw_root / f"{well}__horizontal_well.csv", index=False)
        for row_idx in range(17):
            rows.append(
                {
                    "id": f"{well}_{row_idx}",
                    "well": well,
                    "row_idx": row_idx,
                    "fold": well_index,
                    "pred_tvt": 100.0 + row_idx,
                }
            )
    base = pd.DataFrame(rows)
    changed = copy.deepcopy(config)
    changed["data"]["raw_geometry"]["root_patterns"] = [str(raw_root)]
    monkeypatch.delenv("EXP511_RAW_TRAIN_DIR", raising=False)
    allowlist, manifest = train.attach_raw_geometry_allowlist(base, changed)
    assert tuple(allowlist.columns) == train.PREDICTION_ALLOWLIST
    assert manifest["loaded_columns"] == ["X", "Y", "Z"]
    assert manifest["forbidden_columns_loaded"] == 0
    assert manifest["files"] == 2
    pd.testing.assert_frame_equal(
        allowlist[["well", "row_idx", "fold"]],
        base[["well", "row_idx", "fold"]],
        check_dtype=False,
    )
    assert "TVT" not in allowlist and "ANCC" not in allowlist


def test_raw_geometry_resolver_accepts_kaggle_competition_mount_fallback(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kaggle_input = tmp_path / "kaggle" / "input"
    raw_root = (
        kaggle_input
        / "competitions"
        / "rogii-wellbore-geology-prediction"
        / "train"
    )
    raw_root.mkdir(parents=True)
    for well in ("g0", "g1"):
        pd.DataFrame({"X": [0.0], "Y": [1.0], "Z": [2.0]}).to_csv(
            raw_root / f"{well}__horizontal_well.csv", index=False
        )
    changed = copy.deepcopy(config)
    changed["data"]["raw_geometry"]["root_patterns"] = [
        str(tmp_path / "missing")
    ]
    monkeypatch.setattr(train, "KAGGLE_INPUT_ROOT", kaggle_input)
    monkeypatch.delenv("EXP511_RAW_TRAIN_DIR", raising=False)

    resolved, by_well = train.resolve_raw_train_dir(changed, {"g0", "g1"})

    assert resolved == raw_root
    assert set(by_well) == {"g0", "g1"}


def passing_gate_inputs(train: ModuleType) -> tuple[dict, dict[str, pd.DataFrame]]:
    summary = {
        "pooled": {"gain_ft": 0.02},
        "tail": {"delta_p95_ft": 0.10, "worst_delta_ft": 0.20},
        "continuity": {
            "first_score_row_abs_correction_max_ft": 0.0,
            "all_row_abs_correction_max_ft": 0.25,
        },
    }
    rows = [
        {
            "kind": "fold",
            "scope": str(fold),
            "delta_exp511_minus_exp413": value,
        }
        for fold, value in enumerate([-0.01, 0.0, -0.02, -0.01, 0.01])
    ]
    rows.extend(
        {
            "kind": "scope",
            "scope": scope,
            "delta_exp511_minus_exp413": value,
        }
        for scope, value in zip(
            train.FIXED_SCOPE_ORDER,
            [0.01, 0.00, -0.01, 0.02, 0.01],
            strict=True,
        )
    )
    return summary, {"pooled_fold_scope": pd.DataFrame(rows)}


def test_promotion_gate_is_strict_all_and(
    train: ModuleType, config: dict
) -> None:
    summary, tables = passing_gate_inputs(train)
    passed = train.build_promotion_gate(summary, tables, {"all": True}, config)
    assert passed["passed"] is True
    assert passed["inference_automatically_approved"] is False
    failed_summary = copy.deepcopy(summary)
    failed_summary["tail"]["worst_delta_ft"] = 0.2500001
    failed = train.build_promotion_gate(
        failed_summary, tables, {"all": True}, config
    )
    assert failed["passed"] is False
    assert failed["decision"] == (
        "FAIL_CLOSE_WITHOUT_ALPHA_CLIP_K_BANDWIDTH_RHO_THETA_SUPPORT_"
        "FADE_SCOPE_OR_GATE_RESCUE"
    )
    assert failed["same_oof_rescue_performed"] is False


def test_source_has_no_model_physics_router_inference_or_submission_path() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    forbidden_calls = (
        "LGBMRegressor(",
        "CatBoostRegressor(",
        "XGBRegressor(",
        "run_particle_filter(",
        "run_pf(",
        "run_hmm(",
        "run_beam(",
        "fit_selector(",
        "fit_router(",
        "submission.csv",
    )
    assert all(call not in text for call in forbidden_calls)
    assert "Path(__file__)" not in text

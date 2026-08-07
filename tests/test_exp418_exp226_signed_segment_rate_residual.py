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
EXP_DIR = ROOT / "experiments" / "exp418_exp226_signed_segment_rate_residual"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp418_exp226_signed_segment_rate_residual_compact_selfcontained_train.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP418_IMPORT_ONLY")
    os.environ["EXP418_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP418_IMPORT_ONLY", None)
        else:
            os.environ["EXP418_IMPORT_ONLY"] = previous


train = load_module(TRAIN_SOURCE, "exp418_train")


@pytest.fixture
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_nested(
    *,
    wells: int = 2,
    rows_per_well: int = 32,
    outer_fold: int = 0,
    role: str = "outer_valid",
) -> pd.DataFrame:
    records: list[dict] = []
    for well_index in range(wells):
        segment = train.exact_k16_segment_ids(rows_per_well)
        for offset in range(rows_per_well):
            records.append(
                {
                    "outer_fold": outer_fold,
                    "role": role,
                    "inner_fold": -1 if role == "outer_valid" else 0,
                    "well_id": f"well_{well_index}",
                    "row_idx": 1000 * well_index + offset,
                    "suffix_offset": offset,
                    "segment_id": int(segment[offset]),
                    "tvt_pred": 12000.0 + 10.0 * well_index + offset,
                }
            )
    return pd.DataFrame(records, columns=train.NESTED_COLUMNS)


def test_config_records_consumed_stage0_execution_authorization(
    config: dict,
) -> None:
    preview = train.validate_implementation_contract(config)
    assert config["experiment"]["route"] == "ensemble"
    assert config["experiment"]["status"] == "stage0_v1_technical_fail_closed"
    assert config["implementation"]["scope"] == "stage_0_and_stage_1_train"
    assert config["implementation"]["compact_train_candidate_created"] is True
    assert config["implementation"]["canonical_notebook_is_placeholder"] is False
    assert config["implementation"]["kaggle_package_created"] is True
    assert config["implementation"]["kaggle_execution_enabled"] is False
    assert config["execution_contract"]["selected_stage"] == "none"
    assert config["execution_contract"]["stage_0"]["boosters"] == 0
    stage1 = config["execution_contract"][
        "stage_1_if_stage_0_pass_and_separately_approved"
    ]
    assert stage1 == {
        "active_variants": 1,
        "model_configs": 1,
        "outer_folds": 5,
        "boosters": 5,
        "gpu": False,
        "exp226_fits": 0,
        "parent_control_retraining": False,
        "pf_hmm_beam_regeneration": 0,
    }
    assert config["execution_contract"]["stage_0_run_authorization_consumed"] is True
    assert preview["stage_0_run_approved"] is False
    assert preview["stage_1_run_approved"] is False
    with pytest.raises(RuntimeError, match="stage_0 execution is not authorized"):
        train.validate_implementation_contract(config, execution_stage="stage_0")
    with pytest.raises(RuntimeError, match="stage_1 execution is not authorized"):
        train.validate_implementation_contract(config, execution_stage="stage_1")


def test_contract_rejects_intercept_clip_or_exp226_fit(config: dict) -> None:
    changed = copy.deepcopy(config)
    changed["rate_target"]["solver"]["intercept"] = True
    with pytest.raises(ValueError, match="implementation contract changed"):
        train.validate_implementation_contract(changed)
    changed = copy.deepcopy(config)
    changed["correction"]["clipping"] = "clip_10"
    with pytest.raises(ValueError, match="implementation contract changed"):
        train.validate_implementation_contract(changed)
    changed = copy.deepcopy(config)
    changed["execution_contract"][
        "stage_1_if_stage_0_pass_and_separately_approved"
    ]["exp226_fits"] = 25
    with pytest.raises(ValueError, match="implementation contract changed"):
        train.validate_implementation_contract(changed)


def test_exact_k16_assignment_and_destination_interval_basis() -> None:
    segment = train.exact_k16_segment_ids(32)
    np.testing.assert_array_equal(segment, np.repeat(np.arange(16), 2))
    basis = train.cumulative_rate_basis(segment)
    assert basis.shape == (32, 16)
    np.testing.assert_array_equal(basis[0], 0.0)
    assert basis[1, 0] == 1.0
    assert basis[2, 0] == 1.0
    assert basis[2, 1] == 1.0
    assert np.linalg.matrix_rank(basis) == 16
    with pytest.raises(ValueError, match="fixed to K16"):
        train.exact_k16_segment_ids(32, k_segments=12)


def test_rate_target_recovers_signed_rates_and_zero_anchor() -> None:
    segment = train.exact_k16_segment_ids(64)
    expected_rates = np.linspace(-0.08, 0.07, 16, dtype=np.float64)
    residual = train.cumulative_rate_basis(segment) @ expected_rates
    solution = train.solve_zero_intercept_rates(residual, segment)
    np.testing.assert_allclose(solution.rates, expected_rates, atol=1e-12)
    np.testing.assert_allclose(solution.correction, residual, atol=1e-12)
    assert solution.correction[0] == 0.0
    assert solution.rank == 16
    assert solution.integration_max_abs_diff <= 1e-12


def test_sequential_integration_has_no_boundary_level_broadcast() -> None:
    segment = train.exact_k16_segment_ids(32)
    rates = np.arange(16, dtype=np.float64) / 100.0
    correction, parity = train.integrate_predicted_rates(segment, rates)
    assert correction[0] == 0.0
    assert correction[1] == pytest.approx(0.0)
    assert correction[2] == pytest.approx(0.01)
    assert correction[3] == pytest.approx(0.02)
    assert parity <= 1e-12


def test_truth_is_rejected_until_freeze_exists() -> None:
    rows = synthetic_nested()
    truth = rows[["well_id", "row_idx"]].copy()
    truth["tvt_true"] = rows["tvt_pred"] + 1.0
    with pytest.raises(ValueError, match="frozen truth-free contract"):
        train.attach_truth_after_freeze(
            rows, truth, truth_free_contract_sha256=""
        )
    joined = train.attach_truth_after_freeze(
        rows, truth, truth_free_contract_sha256="frozen"
    )
    assert joined["tvt_true"].notna().all()


def test_rate_target_rows_use_one_lstsq_per_well() -> None:
    rows = synthetic_nested()
    expected_rates = np.linspace(-0.04, 0.05, 16)
    truth_parts: list[pd.DataFrame] = []
    for _well_id, part in rows.groupby("well_id", sort=True):
        correction = train.cumulative_rate_basis(
            part["segment_id"].to_numpy()
        ) @ expected_rates
        output = part.copy()
        output["tvt_true"] = part["tvt_pred"].to_numpy() + correction
        truth_parts.append(output)
    joined = pd.concat(truth_parts, ignore_index=True)
    row_output, targets, diagnostics = train.build_rate_target_rows(joined)
    assert len(targets) == 32
    np.testing.assert_allclose(
        targets.loc[targets["well_id"].eq("well_0"), "segment_rate_target"],
        expected_rates,
        atol=1e-12,
    )
    assert diagnostics["rank_min"] == 16
    assert diagnostics["first_row_correction_abs_max_ft"] == 0.0
    assert diagnostics["matrix_vs_sequential_integration_abs_max_ft"] <= 1e-12
    np.testing.assert_allclose(
        row_output["tvt_pred_rate_oracle"], row_output["tvt_true"], atol=1e-12
    )


def test_segment_aggregation_preserves_exp333_features_and_rate_target() -> None:
    nested = synthetic_nested(wells=1, rows_per_well=32)
    surface = nested[["well_id", "row_idx"]].copy()
    surface["md_since"] = nested["suffix_offset"].to_numpy(float) * 10.0
    surface["feature_mean"] = np.tile([1.0, 3.0], 16)
    surface["feature_all_nan"] = np.nan
    expected_rates = np.linspace(-0.04, 0.05, 16)
    truth = nested[["well_id", "row_idx"]].copy()
    truth["tvt_true"] = nested["tvt_pred"].to_numpy() + (
        train.cumulative_rate_basis(nested["segment_id"].to_numpy())
        @ expected_rates
    )
    segments = train.aggregate_rate_segments(
        nested,
        surface,
        truth,
        ("feature_mean", "feature_all_nan"),
        truth_free_contract_sha256="frozen",
    )
    assert len(segments) == 16
    assert segments["segment_row_count"].eq(2).all()
    np.testing.assert_allclose(segments["feature_mean"], 2.0)
    assert segments["feature_all_nan"].isna().all()
    np.testing.assert_allclose(
        segments["segment_rate_target"], expected_rates, atol=1e-12
    )
    np.testing.assert_allclose(segments["segment_md_span"], 10.0)
    np.testing.assert_allclose(segments["exp226_pred_end_minus_start"], 1.0)
    assert segments["basis_rank"].eq(16).all()


def test_integrate_valid_rows_maps_segment_rates_continuously() -> None:
    nested = synthetic_nested(wells=1, rows_per_well=32)
    surface = nested[["well_id", "row_idx"]].copy()
    surface["md_since"] = nested["suffix_offset"].to_numpy(float) * 10.0
    target_rates = np.linspace(-0.04, 0.05, 16)
    truth = nested[["well_id", "row_idx"]].copy()
    truth["tvt_true"] = nested["tvt_pred"].to_numpy() + (
        train.cumulative_rate_basis(nested["segment_id"].to_numpy())
        @ target_rates
    )
    valid_segments = pd.DataFrame(
        {
            "well_id": "well_0",
            "segment_id": np.arange(16),
            "segment_rate_target": target_rates,
        }
    )
    output, diagnostics = train.integrate_valid_rows(
        nested,
        valid_segments,
        target_rates,
        surface,
        truth,
        truth_free_contract_sha256="frozen",
    )
    np.testing.assert_allclose(output["tvt_pred_stage1"], output["tvt_true"])
    assert output["rate_correction_pred"].iloc[0] == 0.0
    assert diagnostics["first_row_correction_abs_max_ft"] == 0.0
    assert diagnostics["matrix_vs_sequential_integration_abs_max_ft"] <= 1e-12


def test_stage0_oracle_gate_requires_all_fixed_checks(config: dict) -> None:
    rows = synthetic_nested(wells=5, rows_per_well=32)
    rows["outer_fold"] = rows["well_id"].str.rsplit("_", n=1).str[-1].astype(int)
    expected_rates = np.linspace(-0.4, 0.5, 16)
    truth = rows[["well_id", "row_idx"]].copy()
    truth["tvt_true"] = rows["tvt_pred"].to_numpy() + np.concatenate(
        [
            train.cumulative_rate_basis(part["segment_id"].to_numpy())
            @ expected_rates
            for _well, part in rows.groupby("well_id", sort=True)
        ]
    )
    changed = copy.deepcopy(config)
    changed["data"]["exp226_oof"]["expected_rmse"] = train.rmse(
        truth["tvt_true"], rows["tvt_pred"]
    )
    changed["stage_0_gate"]["minimum_oracle_gain_vs_exp226_ft"] = 0.1
    changed["stage_0_gate"]["minimum_fold_gain_vs_exp226_ft"] = 0.1
    readout = train.build_stage0_oracle_readout(
        rows,
        truth,
        changed,
        truth_free_contract_sha256="frozen",
        enforce_expected_counts=False,
    )
    assert readout.summary["decision"] == "PASS_STAGE0"
    assert readout.summary["technical_pass"] is True
    assert readout.summary["scientific_pass"] is True
    failed = copy.deepcopy(changed)
    failed["stage_0_gate"]["maximum_first_row_correction_abs_ft"] = -1.0
    readout = train.build_stage0_oracle_readout(
        rows,
        truth,
        failed,
        truth_free_contract_sha256="frozen",
        enforce_expected_counts=False,
    )
    assert readout.summary["decision"] == "FAIL_CLOSE_BRANCH"


def test_balanced_sign_accuracy_is_macro_recall() -> None:
    target = np.array([-2.0, -1.0, 1.0, 2.0])
    prediction = np.array([-1.0, 1.0, 1.0, 1.0])
    assert train.balanced_sign_accuracy(target, prediction) == pytest.approx(0.75)


def test_stage1_gate_requires_rate_and_tail_checks(config: dict) -> None:
    changed = copy.deepcopy(config)
    changed["stage_1_gate"]["maximum_pooled_rmse"] = 1.1
    changed["stage_1_gate"]["minimum_gain_vs_exp228_ft"] = 0.0
    changed["stage_1_gate"]["minimum_gain_vs_exp333_ft"] = 0.0
    row_records: list[dict] = []
    segment_records: list[dict] = []
    hidden_records: list[dict] = []
    for fold in range(5):
        well_id = f"well_{fold}"
        hidden_records.append(
            {
                "well_id": well_id,
                "verification_like_spatial_role": "valid",
                "verification_like_typewell_purged_role": "valid",
            }
        )
        for md_since in (100.0, 1200.0):
            row_records.append(
                {
                    "outer_fold": fold,
                    "well_id": well_id,
                    "tvt_true": 10.0,
                    "tvt_pred": 8.0,
                    "tvt_pred_stage1": 9.0,
                    "md_since": md_since,
                    "boundary_band_pm8": True,
                }
            )
        segment_records.extend(
            [
                {
                    "outer_fold": fold,
                    "well_id": well_id,
                    "segment_id": 0,
                    "segment_row_count": 2,
                    "segment_rate_target": -1.0,
                    "segment_rate_pred": -0.8,
                },
                {
                    "outer_fold": fold,
                    "well_id": well_id,
                    "segment_id": 1,
                    "segment_row_count": 2,
                    "segment_rate_target": 1.0,
                    "segment_rate_pred": 0.8,
                },
            ]
        )
    result = train.evaluate_stage1_outputs(
        pd.DataFrame(row_records),
        pd.DataFrame(segment_records),
        pd.DataFrame(hidden_records),
        {
            "first_row_correction_abs_max_ft": 0.0,
            "matrix_vs_sequential_integration_abs_max_ft": 0.0,
        },
        changed,
    )
    assert result["decision"] == "PASS_STAGE1"
    assert result["rate_sign_balanced_accuracy_passed_folds"] == 5
    regressed = pd.DataFrame(row_records)
    regressed.loc[regressed["md_since"].eq(100.0), "tvt_pred_stage1"] = 7.0
    failed = train.evaluate_stage1_outputs(
        regressed,
        pd.DataFrame(segment_records),
        pd.DataFrame(hidden_records),
        {
            "first_row_correction_abs_max_ft": 0.0,
            "matrix_vs_sequential_integration_abs_max_ft": 0.0,
        },
        changed,
    )
    assert failed["gate_checks"]["near_0_250_nonworse"] is False
    assert failed["decision"] == "FAIL_CLOSE_BRANCH"


def test_stage1_requires_sha_frozen_stage0_pass(config: dict) -> None:
    with pytest.raises(RuntimeError, match="Stage 0 summary SHA"):
        train.load_stage0_pass_evidence(config)


def test_source_never_calls_exp226_fit_or_regeneration() -> None:
    source = TRAIN_SOURCE.read_text()
    assert "build_fields(" not in source
    assert "fit_kappa(" not in source
    assert "generate_strict_nested_predictions" not in source
    assert "from inputs.exp226_source" not in source
    assert '"exp226_fits": 0' in source

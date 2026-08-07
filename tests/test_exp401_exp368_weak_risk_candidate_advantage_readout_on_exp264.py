from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = (
    ROOT
    / "experiments"
    / "exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264"
)
TRAIN_SOURCE = (
    EXPERIMENT_DIR
    / "exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264_"
    "compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXPERIMENT_DIR
    / "exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264_"
    "compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp401_train_contract")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp401_inference_contract")


@pytest.fixture(scope="module")
def config(train):
    return train.load_experiment_config(EXPERIMENT_DIR)


def test_frozen_scientific_contract_and_run_boundary(train, config) -> None:
    contract = train.validate_scientific_contract(config)

    assert contract["anchor"] == "likpf_mean"
    assert len(contract["candidate_order"]) == 12
    assert len(contract["primary_domain"]) == 11
    assert len(contract["secondary_domain"]) == 7
    assert contract["execution_counts"]["boosters"] == 0
    assert contract["execution_counts"]["pf_runs"] == 0
    assert contract["execution_counts"]["prediction_rows"] == 0
    assert len(contract["scientific_contract_sha256"]) == 64
    with pytest.raises(PermissionError, match="run_stage_0 must be true"):
        train.validate_scientific_contract(config, require_run_approval=True)

    scheduled = copy.deepcopy(config)
    scheduled["execution"]["run_stage_0"] = True
    approved_contract = train.validate_scientific_contract(
        scheduled, require_run_approval=True
    )
    assert approved_contract == contract

    revoked = copy.deepcopy(scheduled)
    revoked["execution"]["stage_0_run_approved"] = False
    with pytest.raises(PermissionError, match="run is not approved"):
        train.validate_scientific_contract(revoked, require_run_approval=True)
    assert json.loads(
        json.dumps({"gate": np.bool_(True)}, default=train._json_default)
    ) == {"gate": True}


def test_inference_is_fail_closed(inference) -> None:
    config = inference.load_config(EXPERIMENT_DIR)
    status = inference.validate_inference_is_disabled(config)

    assert status["implementation_scope"] == "stage_0_implementation_only"
    assert status["stage_0_implemented"] is True
    assert status["stage_1_implemented"] is False
    assert status["inference_enabled"] is False
    assert status["run_inference"] is False
    assert status["create_submission"] is False


def test_overlap_row_risk_uses_every_covering_block_and_keeps_tail(train) -> None:
    blocks = pd.DataFrame(
        {
            "well_id": ["w0"] * 3,
            "block_id": [0, 1, 2],
            "start_suffix_offset": [0, 2, 4],
            "stop_suffix_offset_exclusive": [3, 5, 5],
            "start_row_idx": [10, 12, 14],
            "end_row_idx": [12, 14, 14],
            "block_row_count": [3, 3, 1],
            "weak_posterior_mean": [0.2, 0.6, 1.0],
            "circular_weak_score": [1.0, 0.6, 0.2],
            "circular_offset_blocks": [1, 1, 1],
        }
    )

    rows = train.aggregate_overlapping_block_risk(
        blocks, block_rows=3, stride_rows=2
    )

    assert rows["row_idx"].tolist() == [10, 11, 12, 13, 14]
    assert rows["suffix_offset"].tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(
        rows[train.RISK_FEATURE_COLUMN],
        [0.2, 0.2, 0.4, 0.6, 0.8],
    )
    np.testing.assert_allclose(
        rows[train.CIRCULAR_FEATURE_COLUMN],
        [1.0, 1.0, 0.8, 0.6, 0.4],
    )
    assert rows["covering_block_count"].tolist() == [1, 1, 2, 1, 2]


def test_truth_access_is_fail_closed_until_all_feature_evidence_is_frozen(
    train,
) -> None:
    ledger = train.TruthAccessLedger()

    ledger.record_target_free_projection(["well_id", "row_idx", "fold"])
    ledger.record_target_free_projection(["pred_abs_error", "p_within10"])
    train.assert_target_free_columns(
        ["pred_abs_error", "p_within10"],
        stage="saved strict-nested score",
    )
    with pytest.raises(ValueError, match="truth/error columns"):
        train.assert_target_free_columns(
            ["abs_error"],
            stage="realized error",
        )
    with pytest.raises(RuntimeError, match="truth/error columns"):
        ledger.record_target_free_projection(["well_id", "tvt_true"])
    assert ledger.truth_columns_read_before_freeze == 1
    with pytest.raises(RuntimeError, match="early truth/error read"):
        ledger.freeze_features(
            {
                "feature_schema_sha256": "a" * 64,
                "feature_content_sha256": "b" * 64,
                "selector_surface_content_sha256": "c" * 64,
                "scientific_contract_sha256": "d" * 64,
            }
        )

    clean = train.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="requires frozen"):
        clean.read_late_truth(["well_id", "row_idx", "tvt_true"])
    clean.freeze_features(
        {
            "feature_schema_sha256": "a" * 64,
            "feature_content_sha256": "b" * 64,
            "selector_surface_content_sha256": "c" * 64,
            "scientific_contract_sha256": "d" * 64,
        }
    )
    clean.read_late_truth(["well_id", "row_idx", "tvt_true"])
    assert clean.late_truth_columns_read == ["well_id", "row_idx", "tvt_true"]


def _write_small_selector_parquet(
    path: Path,
    candidates: list[str],
) -> None:
    records = []
    score_rows = [
        np.array([1.0, 1.0, 0.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
        np.array([4.0, 4.0, 2.0, 4.0, 4.0, 0.6, 0.5, 4.0, 4.0, 4.0, 4.0, 0.6]),
    ]
    for row_number, (well, row_idx, generation_fold) in enumerate(
        [("w0", 10, 1), ("w1", 20, 0)]
    ):
        for code, candidate in enumerate(candidates):
            records.append(
                {
                    "id": f"{well}_{row_idx}",
                    "well": well,
                    "outer_fold": generation_fold,
                    "candidate_id": candidate,
                    "candidate_tvt": float(100 + row_number * 10 + code),
                    "pred_abs_error": float(score_rows[row_number][code]),
                    "p_within10": 0.5,
                    "downstream_outer_fold": generation_fold,
                    "nested_model_count": 4,
                }
            )
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(records)), path)


def test_selector_scan_separates_domains_and_uses_declared_tie_order(
    train,
    config,
    tmp_path: Path,
) -> None:
    test_config = copy.deepcopy(config)
    candidates = test_config["candidate_contract"]["candidate_order"]
    test_config["validation"]["expected_candidate_long_rows"] = 24
    test_config["validation"]["expected_folds"] = [0, 1]
    score_path = tmp_path / "scores.parquet"
    _write_small_selector_parquet(score_path, candidates)
    row_risk = pd.DataFrame(
        {
            "well_id": ["w0", "w1"],
            "row_idx": [10, 20],
            "fold": np.array([0, 1], dtype=np.int8),
        }
    )

    surface, candidate_values, audit = (
        train.scan_strict_nested_selector_surface(
            score_path,
            row_risk,
            test_config,
            tmp_path,
        )
    )

    primary = "primitive_pair_bank"
    secondary = "primitive_fixed_bank"
    assert surface[f"{primary}__nominated_code"].tolist() == [0, 6]
    assert surface[f"{secondary}__nominated_code"].tolist() == [0, 5]
    assert surface[f"{primary}__selector_margin"].tolist() == pytest.approx(
        [-0.5, 1.5]
    )
    assert surface[f"{secondary}__selector_margin"].tolist() == pytest.approx(
        [-0.5, 1.4]
    )
    assert audit["candidate_long_rows"] == 24
    assert audit["candidates_per_row"] == 12
    assert audit["selector_generation_vs_reporting_fold_mismatch_rows"] == 2
    assert audit["selector_generation_vs_reporting_fold_mismatch_wells"] == 2
    assert len(audit["selector_surface_content_sha256"]) == 64
    np.testing.assert_allclose(candidate_values[0], np.arange(100, 112))
    candidate_values_path = Path(audit["candidate_values_path"])
    del candidate_values
    candidate_values_path.unlink()


def test_crossfit_bins_and_tie_aware_auc(train) -> None:
    labels = np.array([False, True, False, True])
    scores = np.array([0.1, 0.9, 0.5, 0.5])
    assert train.roc_auc_binary(labels, scores) == pytest.approx(0.875)

    values = np.arange(10, dtype=float)
    folds = np.array([0, 1] * 5, dtype=np.int8)
    bins, boundaries = train.crossfit_quantile_bins(
        values, folds, quantiles=[0.5]
    )
    assert boundaries[0] == [5.0]
    assert boundaries[1] == [4.0]
    assert bins.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    quartiles, quartile_boundaries = train.crossfit_extreme_quartiles(
        values, folds
    )
    assert quartile_boundaries[0] == [3.0, 7.0]
    assert quartile_boundaries[1] == [2.0, 6.0]
    assert quartiles.tolist() == [1, 1, 1, 0, 0, 0, 0, 4, 4, 4]


def test_freeze_then_late_truth_candidate_advantage_readout(
    train,
    config,
    tmp_path: Path,
) -> None:
    rows = 8
    row_risk = pd.DataFrame(
        {
            "well_id": ["w0"] * 4 + ["w1"] * 4,
            "row_idx": [10, 11, 12, 13, 20, 21, 22, 23],
            "suffix_offset": [0, 1, 2, 3] * 2,
            "fold": np.array([0] * 4 + [1] * 4, dtype=np.int8),
            train.RISK_FEATURE_COLUMN: np.array(
                [0.1, 0.8, 0.2, 0.9, 0.15, 0.85, 0.25, 0.95],
                dtype=np.float32,
            ),
            train.CIRCULAR_FEATURE_COLUMN: np.array(
                [0.9, 0.2, 0.8, 0.1, 0.85, 0.15, 0.75, 0.05],
                dtype=np.float32,
            ),
            "covering_block_count": np.ones(rows, dtype=np.int8),
        }
    )
    recovering = np.array([False, True] * 4)
    surface = {}
    for domain in ("primitive_pair_bank", "primitive_fixed_bank"):
        surface[f"{domain}__nominated_code"] = np.zeros(rows, dtype=np.int8)
        surface[f"{domain}__nominated_tvt"] = np.where(
            recovering, 105.0, 120.0
        ).astype(np.float32)
        surface[f"{domain}__nominated_pred_abs_error"] = np.linspace(
            1.0, 2.0, rows, dtype=np.float32
        )
        surface[f"{domain}__anchor_tvt"] = np.full(
            rows, 115.0, dtype=np.float32
        )
        surface[f"{domain}__anchor_pred_abs_error"] = np.linspace(
            2.0, 3.0, rows, dtype=np.float32
        )
        surface[f"{domain}__selector_margin"] = np.ones(
            rows, dtype=np.float32
        )
    candidate_values = np.full((rows, 12), 130.0, dtype=np.float32)
    candidate_values[:, 2] = 115.0
    candidate_values[:, 0] = np.where(recovering, 105.0, 120.0)
    contract = train.validate_scientific_contract(config)
    selector_audit = {"selector_surface_content_sha256": "a" * 64}
    ledger = train.TruthAccessLedger()

    frozen = train.freeze_target_free_surface(
        row_risk,
        surface,
        selector_audit,
        contract,
        tmp_path,
        ledger,
    )
    assert ledger.frozen is True
    assert len(frozen["evidence"]["feature_content_sha256"]) == 64

    truth_path = tmp_path / "truth.csv.gz"
    pd.DataFrame(
        {
            "well_id": row_risk["well_id"],
            "row_idx": row_risk["row_idx"],
            "tvt_true": np.full(rows, 100.0),
        }
    ).to_csv(truth_path, index=False)
    hidden_path = tmp_path / "hidden.csv"
    pd.DataFrame(
        {
            "well_id": ["w0", "w1"],
            "verification_like_spatial_role": ["valid", "valid"],
            "verification_like_typewell_purged_role": ["valid", "valid"],
        }
    ).to_csv(hidden_path, index=False)
    late = train.load_late_truth_and_roles(
        frozen["row_risk"],
        {"fold_truth": truth_path, "hidden_like": hidden_path},
        config,
        ledger,
    )
    metrics, nominations = train.build_scope_metrics(
        frozen["row_risk"],
        late,
        surface,
        candidate_values,
        frozen,
        config,
    )

    primary = metrics.loc[
        metrics["domain"].eq("primitive_pair_bank")
        & metrics["scope"].eq("overall")
    ].iloc[0]
    assert primary["cohort_rows"] == rows
    assert primary["positive_rows"] == 4
    assert primary["negative_rows"] == 4
    assert primary["real_nominated_recovery10_auc"] == pytest.approx(1.0)
    assert primary["circular_nominated_recovery10_auc"] == pytest.approx(0.0)
    assert nominations["all_rows"].sum() == rows * 2
    assert ledger.truth_columns_read_before_freeze == 0
    assert ledger.late_truth_columns_read == ["well_id", "row_idx", "tvt_true"]


def _passing_metric_rows() -> pd.DataFrame:
    rows = []
    scopes = [
        "overall",
        "fold_0",
        "fold_1",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ]
    for domain in ("primitive_pair_bank", "primitive_fixed_bank"):
        for scope in scopes:
            primary = domain == "primitive_pair_bank"
            rows.append(
                {
                    "domain": domain,
                    "scope": scope,
                    "positive_rows": 2,
                    "negative_rows": 2,
                    "real_nominated_recovery10_auc": 0.65 if primary else 0.55,
                    "circular_nominated_recovery10_auc": 0.55 if primary else 0.54,
                    "real_minus_circular_auc": 0.10 if primary else 0.01,
                    "margin_conditional_auc": 0.60,
                    "q4_minus_q1_mean_realized_advantage_ft": 0.60 if primary else 0.10,
                }
            )
    return pd.DataFrame(rows)


def test_gate_is_exact_all_and_without_auc_inversion(train, config) -> None:
    test_config = copy.deepcopy(config)
    test_config["validation"]["expected_folds"] = [0, 1]
    technical = test_config["stage_0"]["technical_gate_all_required"]
    technical.update(
        {
            "expected_rows": 4,
            "expected_wells": 2,
            "expected_blocks": 2,
            "expected_folds": 2,
            "expected_candidate_long_rows": 48,
            "minimum_positive_rows_per_required_scope": 1,
            "minimum_negative_rows_per_required_scope": 1,
        }
    )
    primary_gate = test_config["stage_0"]["scientific_gate_all_required"][
        "primary_domain"
    ]
    primary_gate["minimum_folds_with_real_auc_strictly_above_0p50"] = 2
    primary_gate[
        "minimum_folds_with_margin_conditional_auc_strictly_above_0p50"
    ] = 2
    primary_gate["minimum_folds_with_positive_q4_minus_q1_advantage"] = 2
    row_risk = pd.DataFrame(
        {
            "well_id": ["w0", "w0", "w1", "w1"],
            "fold": np.array([0, 0, 1, 1], dtype=np.int8),
            train.RISK_FEATURE_COLUMN: np.array([0.1, 0.2, 0.8, 0.9]),
            train.CIRCULAR_FEATURE_COLUMN: np.array([0.9, 0.8, 0.2, 0.1]),
        }
    )
    selector_audit = {
        "candidate_long_rows": 48,
        "candidates_per_row": 12,
        "primary_domain_codes": list(range(11)),
        "secondary_domain_codes": [0, 1, 2, 3, 4, 5, 11],
        "covered_rows": 4,
    }
    hash_value = "a" * 64
    preflight = {
        "input_sha256": {
            "block": hash_value,
            "stage_c_score_schema_sha256": hash_value,
        }
    }
    frozen = {
        "feature_schema_sha256": hash_value,
        "feature_content_sha256": hash_value,
        "selector_surface_content_sha256": hash_value,
        "scientific_contract_sha256": hash_value,
    }
    readout = {"metrics": {"sha256": hash_value}}
    ledger = train.TruthAccessLedger()

    passing = train.evaluate_gates(
        metrics=_passing_metric_rows(),
        row_risk=row_risk,
        blocks=pd.DataFrame({"block": [0, 1]}),
        selector_audit=selector_audit,
        preflight=preflight,
        frozen_evidence=frozen,
        readout_reports=readout,
        config=test_config,
        ledger=ledger,
    )
    assert passing["technical_passed"] is True
    assert passing["scientific_passed"] is True
    assert passing["stage_0_passed"] is True

    failed_metrics = _passing_metric_rows()
    failed_metrics.loc[
        failed_metrics["domain"].eq("primitive_pair_bank")
        & failed_metrics["scope"].eq("overall"),
        "real_nominated_recovery10_auc",
    ] = 0.40
    failed = train.evaluate_gates(
        metrics=failed_metrics,
        row_risk=row_risk,
        blocks=pd.DataFrame({"block": [0, 1]}),
        selector_audit=selector_audit,
        preflight=preflight,
        frozen_evidence=frozen,
        readout_reports=readout,
        config=test_config,
        ledger=ledger,
    )
    assert failed["scientific_checks"]["primary_pooled_auc"] is False
    assert failed["stage_0_passed"] is False
    assert failed["decision"] == "stage_0_failed_close_without_rescue"


def test_notebook_sources_are_not_file_relative_or_model_creating() -> None:
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()

    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "import lightgbm" not in train_source.lower()
    assert ".fit(" not in train_source
    assert "particle_filter" not in train_source
    assert "submission.csv" not in train_source
    assert "sample_submission" not in inference_source
    assert "run_stage_0(" in train_source

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = "exp417_scale5_seed_aggregation_promotion_audit"
EXP_DIR = ROOT / "experiments" / EXP
TRAIN_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
INFERENCE_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.py"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train() -> ModuleType:
    return load_module(TRAIN_SOURCE, "exp417_train_contract")


@pytest.fixture(scope="module")
def inference() -> ModuleType:
    return load_module(INFERENCE_SOURCE, "exp417_inference_contract")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def synthetic_parent_contract(train: ModuleType) -> dict:
    payload = {
        "experiment": "exp404_scale5_sigma_gr_likelihood_pf_ablation",
        "primary_control": "likpf_scale_5_x1p0",
        "primary_candidate": "likpf_scale_5_x1p3",
        "parity_columns": ["likpf_mean_x1p0", "likpf_mean_x1p3"],
        "pf": {
            "particles": 500,
            "seeds": 128,
            "primary_seed_weighting_scale": 5.0,
            "arithmetic_mean_enabled_for_parity_only": True,
        },
        "gr_scale": {
            "variants": {
                "gs_x1p0": {"multiplier": 1.0},
            }
        },
    }
    payload["scientific_contract_sha256"] = train.mapping_sha256(payload)
    return payload


def synthetic_prediction() -> pd.DataFrame:
    truth = np.array([100.0, 101.0, 110.0, 111.0])
    frame = pd.DataFrame(
        {
            "id": ["w0_0", "w0_1", "w1_0", "w1_1"],
            "well_id": ["w0", "w0", "w1", "w1"],
            "row_idx": [0, 1, 0, 1],
            "suffix_offset": [0, 1, 0, 1],
            "last_known_tvt": [99.0, 99.0, 109.0, 109.0],
            "md_since": [100.0, 1100.0, 200.0, 1200.0],
            "raw_gr_observed": [True, False, True, False],
            "well_missing_fraction": [0.5, 0.5, 0.1, 0.1],
            "likpf_scale_5_x1p0": truth + 0.2,
            "likpf_scale_5_x1p3": truth + 0.4,
            "likpf_mean_x1p0": truth + 1.0,
            "likpf_mean_x1p3": truth + 0.8,
        }
    )
    frame["id"] = frame["id"].astype(object)
    frame["well_id"] = frame["well_id"].astype(object)
    for column in ("row_idx", "suffix_offset"):
        frame[column] = frame[column].astype(np.int64)
    frame["raw_gr_observed"] = frame["raw_gr_observed"].astype(bool)
    for column in (
        "last_known_tvt",
        "md_since",
        "well_missing_fraction",
        "likpf_scale_5_x1p0",
        "likpf_scale_5_x1p3",
        "likpf_mean_x1p0",
        "likpf_mean_x1p3",
    ):
        frame[column] = frame[column].astype(np.float64)
    return frame


def synthetic_audit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": ["w0", "w1"],
            "status": ["ok", "ok"],
            "gs_base": [20.0, 30.0],
            "gs_x1p0": [20.0, 30.0],
            "seed_base": [123, 456],
            "seed_base_x1p0": [123, 456],
            "pf_well_runs": [2, 2],
            "seeds": [128, 128],
            "particles": [500, 500],
        }
    )


def frozen_bundle(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> tuple[dict, pd.DataFrame, dict, object, dict]:
    local_config = copy.deepcopy(config)
    local_config["validation"]["expected_rows"] = 4
    local_config["validation"]["expected_wells"] = 2
    local_config["validation"]["expected_folds"] = [0, 1]
    prediction = synthetic_prediction()
    audit = synthetic_audit()
    prediction_path = tmp_path / "predictions.csv.gz.bin"
    audit_path = tmp_path / "audit.csv"
    prediction.to_csv(
        prediction_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    audit.to_csv(audit_path, index=False)
    spec = local_config["data"]["exp404_frozen_predictions"]
    spec["expected_logical_sha256"] = train.dataframe_content_sha(
        prediction,
        ["id", "well_id", "row_idx", *train.PARENT_PREDICTION_COLUMNS],
    )
    spec["expected_schema_sha256"] = train.dataframe_schema_sha(prediction)
    parent_contract = synthetic_parent_contract(train)
    preflight = {
        "paths": {
            "exp404_predictions": str(prediction_path),
            "exp404_audit": str(audit_path),
        },
        "reports": {
            "exp404_predictions": train.inspect_gzip_csv(prediction_path),
            "exp404_audit": {"raw_sha256": train.sha256_path(audit_path)},
        },
        "parent_contract": {
            "passed": True,
            "recorded_sha256": parent_contract["scientific_contract_sha256"],
            "recomputed_sha256": parent_contract["scientific_contract_sha256"],
        },
        "all_input_sha_matches": True,
    }
    ledger = train.TruthAccessLedger()
    loaded, loaded_audit, frozen = train.load_and_freeze_prediction_identity(
        preflight,
        local_config,
        ledger,
    )
    pd.testing.assert_frame_equal(loaded, prediction)
    pd.testing.assert_frame_equal(loaded_audit, audit)
    return local_config, loaded, preflight, ledger, frozen


def test_frozen_contract_is_zero_pf_zero_model_and_terminally_closed(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)

    assert contract["primary_control"] == "likpf_mean_x1p0"
    assert contract["primary_candidate"] == "likpf_scale_5_x1p0"
    assert contract["aggregation"]["candidate"]["temperature"] == 5.0
    assert contract["execution_counts"] == {
        "saved_candidate_readouts": 1,
        "scientific_candidates": 1,
        "pf_well_runs": 0,
        "parent_pf_control_reruns": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "reporting_folds": 5,
    }
    assert len(contract["scientific_contract_sha256"]) == 64
    with pytest.raises(RuntimeError, match="not approved"):
        train.validate_scientific_contract(config, require_run_approval=True)


def test_parent_contract_proves_same_x1p0_bank_and_fixed_scale5(
    train: ModuleType,
    config: dict,
) -> None:
    local_config = copy.deepcopy(config)
    payload = synthetic_parent_contract(train)
    local_config["data"]["exp404_frozen_predictions"]["expected_scientific_contract_sha256"] = (
        payload["scientific_contract_sha256"]
    )

    report = train.validate_parent_scientific_contract(payload, local_config)

    assert report["passed"] is True
    assert report["same_x1p0_pf_call_readouts"] == [
        "likpf_mean_x1p0",
        "likpf_scale_5_x1p0",
    ]
    broken = copy.deepcopy(payload)
    broken["pf"]["primary_seed_weighting_scale"] = 8.0
    with pytest.raises(ValueError, match="contract mismatch"):
        train.validate_parent_scientific_contract(broken, local_config)


def test_prediction_identity_freezes_before_truth_and_validates_same_bank(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    _, _, _, ledger, frozen = frozen_bundle(train, config, tmp_path)

    assert ledger.prediction_frozen is True
    assert all(value == 0 for value in ledger.report()["before_freeze"].values())
    assert frozen["same_bank_evidence"]["passed"] is True
    assert frozen["same_bank_evidence"]["stage_a_pf_well_runs"] == 0
    assert len(frozen["stage_a_logical_content_sha256"]) == 64
    assert frozen["control_column"] == "likpf_mean_x1p0"
    assert frozen["candidate_column"] == "likpf_scale_5_x1p0"


def test_late_join_reads_truth_fold_and_roles_only_after_freeze(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    local_config, prediction, preflight, ledger, frozen = frozen_bundle(
        train,
        config,
        tmp_path,
    )
    truth = np.array([100.0, 101.0, 110.0, 111.0])
    exp072 = pd.DataFrame(
        {
            "id": prediction["id"],
            "well": prediction["well_id"],
            "last_known_tvt": truth - 1.0,
            "likpf_mean_d": np.full(4, 2.0),
        }
    )
    hmm = pd.DataFrame(
        {
            "id": prediction["id"],
            "well": prediction["well_id"],
            "hmm_mean_tvt": truth + 0.6,
        }
    )
    fold = pd.DataFrame(
        {
            "well_id": prediction["well_id"],
            "row_idx": prediction["row_idx"],
            "suffix_offset": prediction["suffix_offset"],
            "fold": [0, 0, 1, 1],
            "tvt_true": truth,
        }
    )
    hidden = pd.DataFrame(
        {
            "well_id": ["w0", "w1"],
            "verification_like_spatial_role": ["valid", "train"],
            "verification_like_typewell_purged_role": ["train", "valid"],
        }
    )
    paths = {
        "exp072_control": tmp_path / "exp072.csv.gz",
        "exp209_hmm_control": tmp_path / "hmm.csv.gz",
        "fold_assignment": tmp_path / "fold.csv.gz",
        "hidden_like_assignment": tmp_path / "hidden.csv",
    }
    exp072.to_csv(paths["exp072_control"], index=False, compression="gzip")
    hmm.to_csv(paths["exp209_hmm_control"], index=False, compression="gzip")
    fold.to_csv(paths["fold_assignment"], index=False, compression="gzip")
    hidden.to_csv(paths["hidden_like_assignment"], index=False)
    preflight["paths"].update({key: str(value) for key, value in paths.items()})
    local_config["data"]["hidden_like_assignment"]["expected_role_counts"] = {
        "hidden_like_spatial": {"train": 1, "valid": 1},
        "hidden_like_typewell_purged": {"train": 1, "valid": 1},
    }

    frame, report = train.load_late_readout_frame(
        prediction,
        frozen,
        preflight,
        local_config,
        ledger,
    )

    assert report["truth_attached_after_prediction_freeze"] is True
    assert ledger.unknown_suffix_tvt_rows_after_freeze == 4
    assert ledger.fold_rows_after_freeze == 4
    assert ledger.hidden_like_role_rows_after_freeze == 2
    np.testing.assert_allclose(frame["saved_exp072_likpf_mean"], truth + 1.0)
    np.testing.assert_allclose(
        frame[train.CONTROL_BLEND],
        0.5 * (truth + 0.6) + 0.5 * (truth + 1.0),
    )
    np.testing.assert_allclose(
        frame[train.CANDIDATE_BLEND],
        0.5 * (truth + 0.6) + 0.5 * (truth + 0.2),
    )


def synthetic_metric_frame(train: ModuleType) -> pd.DataFrame:
    rows = 12
    truth = np.linspace(100.0, 111.0, rows)
    frame = pd.DataFrame(
        {
            "id": [f"w{i // 6}_{i}" for i in range(rows)],
            "well_id": ["w0"] * 6 + ["w1"] * 6,
            "true_tvt": truth,
            "fold": [0] * 6 + [1] * 6,
            "raw_gr_observed": [True, False] * 6,
            "well_missing_fraction": [0.4] * 6 + [0.1] * 6,
            "md_since": [1100.0, 1200.0] + [100.0] * 10,
            "hidden_like_spatial": [True] + [False] * 11,
            "hidden_like_typewell_purged": [False, True] + [False] * 10,
            "saved_exp072_likpf_mean": truth + 1.0,
            train.HMM_CONTROL: truth + 0.6,
            train.PRIMARY_CONTROL: truth + 1.0,
            train.PRIMARY_CANDIDATE: truth + 0.2,
        }
    )
    frame[train.CONTROL_BLEND] = 0.5 * frame[train.HMM_CONTROL] + 0.5 * frame[train.PRIMARY_CONTROL]
    frame[train.CANDIDATE_BLEND] = (
        0.5 * frame[train.HMM_CONTROL] + 0.5 * frame[train.PRIMARY_CANDIDATE]
    )
    return frame


def test_metrics_and_gate_enforce_direct_scope_tail_and_fixed_blend(
    train: ModuleType,
    config: dict,
) -> None:
    frame = synthetic_metric_frame(train)
    paired, by_well = train.build_metric_outputs(frame)
    local_config = copy.deepcopy(config)
    local_config["validation"]["expected_rows"] = len(frame)
    local_config["validation"]["expected_wells"] = 2
    local_config["validation"]["expected_folds"] = [0, 1]
    local_config["guards"]["scientific"]["minimum_improved_folds"] = 2
    ledger = train.TruthAccessLedger()
    ledger.mark_frozen()
    frozen = {
        "parent_logical_content_sha256": "a" * 64,
        "stage_a_logical_content_sha256": "b" * 64,
        "same_bank_evidence": {"passed": True},
    }
    gate = train.evaluate_promotion_gate(
        frame,
        paired,
        by_well,
        {"passed": True, "checks": {}},
        {"all_input_sha_matches": True},
        frozen,
        ledger,
        1.0,
        local_config,
    )

    direct = paired.loc[
        paired["comparison"].eq("direct_scale5_vs_arithmetic") & paired["scope"].eq("overall")
    ].iloc[0]
    assert direct["improvement_ft"] == pytest.approx(0.8)
    assert gate["technical_gate"]["passed"] is True
    assert gate["scientific_gate"]["passed"] is True
    assert gate["passed"] is True
    assert gate["scientific_gate"]["improved_folds"] == 2
    assert gate["scientific_gate"]["fixed_hmm_likpf_50_50_delta_rmse_ft"] < 0.0

    frame[train.CANDIDATE_BLEND] = frame["true_tvt"] + 2.0
    paired, by_well = train.build_metric_outputs(frame)
    failed = train.evaluate_promotion_gate(
        frame,
        paired,
        by_well,
        {"passed": True, "checks": {}},
        {"all_input_sha_matches": True},
        frozen,
        ledger,
        1.0,
        local_config,
    )
    assert failed["scientific_gate"]["checks"]["fixed_hmm_likpf_50_50_non_regression"] is False
    assert failed["passed"] is False


def test_parity_metrics_are_technical_only(
    train: ModuleType,
    config: dict,
) -> None:
    frame = synthetic_metric_frame(train)
    local_config = copy.deepcopy(config)
    truth = frame["true_tvt"].to_numpy(np.float64)
    local_config["data"]["exp072_control"]["expected_rmse_ft"] = train.rmse(
        truth,
        frame["saved_exp072_likpf_mean"].to_numpy(np.float64),
    )
    local_config["data"]["exp209_hmm_control"]["expected_rmse_ft"] = train.rmse(
        truth,
        frame[train.HMM_CONTROL].to_numpy(np.float64),
    )
    local_config["data"]["exp209_hmm_control"]["expected_fixed_hmm_likpf_50_50_rmse_ft"] = (
        train.rmse(
            truth,
            frame[train.CONTROL_BLEND].to_numpy(np.float64),
        )
    )

    parity = train.build_parity_metrics(frame, local_config)

    assert parity["policy"] == "saved_control_technical_parity_only"
    assert parity["passed"] is True
    assert set(parity["checks"]) == {
        "saved_exp072_mean_reference",
        "exp404_arithmetic_vs_exp072_reference",
        "exp404_arithmetic_vs_saved_exp072",
        "saved_exp209_hmm_reference",
        "fixed_exp209_hmm_arithmetic_50_50_reference",
    }


def test_inference_remains_fail_closed_after_stage_a_approval(
    inference: ModuleType,
    config: dict,
) -> None:
    status = inference.validate_inference_is_disabled(config)

    assert status["implementation_scope"] == "train_side_saved_oof_promotion_audit_only"
    assert status["stage_a_run_approved"] is True
    assert status["inference_enabled"] is False
    assert status["selected_candidate"] is None
    assert status["inference_approved"] is False
    assert status["submission_approved"] is False


def test_compact_sources_are_not_file_relative_or_submission_creating() -> None:
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()

    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "sample_submission" not in train_source
    assert "sample_submission" not in inference_source
    assert "run_full_experiment(CONFIG)" in train_source
    assert (EXP_DIR / f"{EXP}_train.ipynb").exists()
    assert (EXP_DIR / f"{EXP}_inference.ipynb").exists()

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

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp297_prefix_calibrated_latent_registration_gr_evidence"
TRAIN_PATH = EXP_DIR / (
    "exp297_prefix_calibrated_latent_registration_gr_evidence_compact_selfcontained_train.py"
)
INFERENCE_PATH = EXP_DIR / (
    "exp297_prefix_calibrated_latent_registration_gr_evidence_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    os.environ["EXP297_IMPORT_ONLY"] = "1"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train = load_module(TRAIN_PATH, "exp297_train")
inference = load_module(INFERENCE_PATH, "exp297_inference")


@pytest.fixture
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def test_fixed_stage2_contract_and_execution_boundary(config: dict) -> None:
    train.validate_scientific_contract(config)
    train.validate_execution_contract(config, require_run=False)
    inference.validate_fail_closed(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert tuple(config["candidate_bank"]["order"]) == train.EXPECTED_CANDIDATES
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["trained_fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["hmm_pf_well_runs"] == 0
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["execution"]["kaggle_run_completed"] is True
    assert config["execution"]["one_run_authorization_consumed"] is True
    assert config["execution"]["stage2_decision"] == "FAIL_STOP_NO_STAGE4"
    assert config["execution"]["stage3_authorized"] is False
    assert config["execution"]["stage4_authorized"] is False
    assert config["inference"]["enabled"] is False
    assert train.BLOCK_ASSIGNMENT_DTYPES == {
        "id": object,
        "well": object,
        "well_row_idx": "int32",
        "outer_fold": "int8",
        "md_since": "float32",
        "well_code": "int32",
        "h128_group": "int32",
        "h256_group": "int32",
        "h512_group": "int32",
        "whole_well_group": "int32",
    }

    changed = copy.deepcopy(config)
    changed["audit"]["registration"]["step_ft"] = 1.0
    with pytest.raises(ValueError, match="scientific contract mismatch"):
        train.validate_scientific_contract(changed)
    with pytest.raises(PermissionError, match="separately approved"):
        train.validate_execution_contract(config, require_run=True)
    approved = copy.deepcopy(config)
    approved["execution"]["kaggle_push_approved"] = True
    train.validate_execution_contract(approved, require_run=True)
    unapproved = copy.deepcopy(approved)
    unapproved["execution"]["kaggle_push_approved"] = False
    with pytest.raises(PermissionError, match="separately approved"):
        train.validate_execution_contract(unapproved, require_run=True)


def test_target_free_loader_rejects_truth_columns() -> None:
    train.reject_truth_columns(["MD", "GR", "TVT_input"])
    for column in ("TVT", "true_tvt", "target", "oracle_candidate", "abs_error"):
        with pytest.raises(ValueError, match="forbidden truth"):
            train.reject_truth_columns(["MD", "GR", "TVT_input", column])


def test_prefix_huber_calibration_recovers_affine_map(config: dict) -> None:
    typewell_tvt = np.linspace(0.0, 2000.0, 4001)
    typewell_gr = 70.0 + 20.0 * np.sin(typewell_tvt / 4.0)
    tvt_input = np.linspace(100.0, 400.0, 500)
    reference = np.interp(tvt_input, typewell_tvt, typewell_gr)
    horizontal_gr = 1.6 * reference + 7.0
    horizontal_gr[0] += 100.0
    result = train.robust_affine_calibration(
        horizontal_gr, tvt_input, typewell_tvt, typewell_gr, config
    )
    assert result.valid
    assert result.slope == pytest.approx(1.6, abs=0.01)
    assert result.intercept == pytest.approx(7.0, abs=1.0)
    assert 10.0 <= result.residual_scale <= 60.0
    assert 1.0 <= result.derivative_scale <= 30.0

    invalid = train.robust_affine_calibration(
        horizontal_gr[:20], tvt_input[:20], typewell_tvt, typewell_gr, config
    )
    assert not invalid.valid
    assert invalid.reason == "prefix_pairs_below_minimum"


def test_registration_is_observation_only_and_chain_rule_is_exact() -> None:
    typewell_tvt = np.linspace(0.0, 1000.0, 2001)
    typewell_gr = typewell_tvt**2 / 1000.0
    path = np.linspace(200.0, 260.0, 128)[:, None]
    calibration = train.CalibrationResult(True, "ok", 2.0, 2.0, 3.0, 10.0, 1.0, 128, 10.0, 1.0)
    deltas = np.array([-10.0, 0.0, 10.0])
    reference, derivative = train.registered_forward_matrices(
        path, typewell_tvt, typewell_gr, calibration, deltas
    )
    expected = 2.0 * np.interp(path[:, 0], typewell_tvt, typewell_gr) + 3.0
    np.testing.assert_allclose(reference[:, 1], expected, rtol=0, atol=1.0e-5)
    midpoint = (path[:-1, 0] + path[1:, 0]) / 2.0
    expected_derivative = 2.0 * (2.0 * midpoint / 1000.0) * np.diff(path[:, 0])
    np.testing.assert_allclose(derivative[:, 1], expected_derivative, atol=2.0e-3)
    np.testing.assert_array_equal(path[:, 0], np.linspace(200.0, 260.0, 128))


def test_shuffle_is_stable_and_preserves_nan_mask(config: dict) -> None:
    observed = np.arange(128, dtype=np.float64)
    observed[[3, 19, 77]] = np.nan
    first, first_offset = train.shuffled_preserve_nan_mask(observed, "well-a", config)
    second, second_offset = train.shuffled_preserve_nan_mask(observed, "well-a", config)
    np.testing.assert_array_equal(first, second)
    assert first_offset == second_offset
    np.testing.assert_array_equal(np.isnan(first), np.isnan(observed))
    np.testing.assert_array_equal(
        np.sort(first[np.isfinite(first)]), np.sort(observed[np.isfinite(observed)])
    )


def test_posterior_prefers_generating_state_and_preserves_probability(config: dict) -> None:
    rows = 128
    deltas = train.registration_grid(config)
    states = len(train.EXPECTED_CANDIDATES) * len(deltas)
    x = np.arange(rows, dtype=np.float64)
    observed = 70.0 + 25.0 * np.sin(0.35 * x)
    reference = np.empty((rows, states), dtype=np.float32)
    for state in range(states):
        phase = 0.15 + 0.012 * state
        reference[:, state] = 70.0 + 25.0 * np.sin(0.35 * x + phase)
    center_state = len(deltas) // 2
    reference[:, center_state] = observed
    derivative = np.diff(reference.astype(np.float64), axis=0).astype(np.float32)
    calibration = train.CalibrationResult(True, "ok", 1.0, 1.0, 0.0, 10.0, 1.0, 128, 20.0, 1.0)
    blocks = [train.BlockSlice(0, 0, rows)]
    raw, ncc, dscore, eligible = train.block_component_matrices(
        observed, reference, derivative, blocks, calibration, config
    )
    posterior = train.posterior_from_components(raw, ncc, dscore, eligible, deltas, config)
    assert posterior.candidate.shape == (1, 12)
    assert int(np.argmax(posterior.candidate[0])) == 0
    assert posterior.eligible_states[0] > 0
    assert posterior.reliable_probability[0] > 0
    assert posterior.unreliable_probability[0] > 0
    assert posterior.candidate[0].sum() == pytest.approx(1.0, abs=1.0e-6)
    assert posterior.joint_reliable[0].sum() == pytest.approx(
        posterior.reliable_probability[0], abs=1.0e-6
    )


def test_invalid_evidence_falls_back_only_to_safe_candidate() -> None:
    posterior = train.fallback_posterior(3, 12, 21)
    assert np.all(posterior.candidate[:, train.SAFE_INDEX] == 1.0)
    assert np.all(np.delete(posterior.candidate, train.SAFE_INDEX, axis=1) == 0.0)
    assert np.all(posterior.unreliable_probability == 1.0)
    assert np.all(posterior.joint_reliable == 0.0)


def test_frozen_evidence_detects_post_freeze_mutation(tmp_path: Path) -> None:
    evidence = tmp_path / "posterior.npy"
    contract = tmp_path / "contract.json"
    evidence.write_bytes(b"fixed")
    contract.write_text("{}\n")
    freeze = train.FrozenEvidence(
        paths=(evidence,),
        file_sha256={str(evidence): train.sha256_file(evidence)},
        contract_path=contract,
        contract_file_sha256=train.sha256_file(contract),
        truth_access_count_before_freeze=0,
    )
    train.verify_frozen_evidence(freeze)
    evidence.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed before truth"):
        train.verify_frozen_evidence(freeze)


def decision_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for control in ("real", "shuffle"):
        for fold in (-1, 0, 1, 2, 3, 4):
            rows.append(
                {
                    "control": control,
                    "horizon_rows": 256,
                    "scope": "pooled" if fold == -1 else "fold",
                    "fold": fold,
                    "expected_sse": 50.0 if control == "real" else 60.0,
                    "expected_rmse": 5.0 if control == "real" else 6.0,
                    "headroom_recovery": 0.4 if control == "real" else 0.2,
                }
            )
    rows.append(
        {
            "control": "real",
            "horizon_rows": 512,
            "scope": "pooled",
            "fold": -1,
            "expected_sse": 52.0,
            "expected_rmse": 5.2,
            "headroom_recovery": 0.36,
        }
    )
    subgroups = pd.DataFrame(
        {
            "subgroup": [
                "md_since_1000_plus",
                "hidden_like_spatial",
                "hidden_like_typewell_purged",
            ],
            "anchor_nonregression": [True, True, True],
        }
    )
    return pd.DataFrame(rows), subgroups


def test_stage2_pass_and_fail_routes_are_fixed(config: dict, tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    contract = tmp_path / "contract"
    evidence.write_text("evidence")
    contract.write_text("contract")
    freeze = train.FrozenEvidence(
        (evidence,),
        {str(evidence): train.sha256_file(evidence)},
        contract,
        train.sha256_file(contract),
        0,
    )
    metrics, subgroups = decision_frames()
    passed = train.stage2_decision(config, metrics, subgroups, freeze)
    assert passed["decision"] == "PASS_STAGE3"
    failed_metrics = metrics.copy()
    failed_metrics.loc[
        failed_metrics["control"].eq("real")
        & failed_metrics["horizon_rows"].eq(256)
        & failed_metrics["scope"].eq("pooled"),
        "headroom_recovery",
    ] = 0.34
    failed = train.stage2_decision(config, failed_metrics, subgroups, freeze)
    assert failed["decision"] == "FAIL_STOP_NO_STAGE4"


def test_inference_notebook_always_stops(config: dict) -> None:
    inference.validate_fail_closed(config)
    with pytest.raises(RuntimeError, match="forbidden"):
        inference.stop_inference()

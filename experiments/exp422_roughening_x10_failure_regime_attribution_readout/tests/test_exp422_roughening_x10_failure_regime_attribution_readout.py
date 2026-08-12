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

ROOT = Path(__file__).resolve().parents[3]
EXP = "exp422_roughening_x10_failure_regime_attribution_readout"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train() -> ModuleType:
    previous = os.environ.get("EXP422_IMPORT_ONLY")
    os.environ["EXP422_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp422_train_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP422_IMPORT_ONLY", None)
        else:
            os.environ["EXP422_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def synthetic_audit() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(15):
        eval_rows = 10 + index
        seeds = 128
        particles = 500
        rows.append(
            {
                "well_id": f"w{index:02d}",
                "status": "ok",
                "prefix_rows": 20 + index,
                "prefix_gr_missing_rows": index % 4,
                "eval_rows": eval_rows,
                "eval_raw_gr_missing_rows": index % 7,
                "seeds": seeds,
                "particles": particles,
                "seed_loglik_mean_per_row": -10.0 - index,
                "seed_loglik_best_per_row": -9.5 - index / 2.0,
                "resampling_count_total": seeds * (index + 1),
                "minimum_ess_mean": 450.0 - 10.0 * index,
                "position_clip_count_total": seeds * (index % 3),
                "seed_prediction_std_mean": 0.1 + index / 10.0,
                "gr_scale_clipped": 10.0 + index,
            }
        )
    return pd.DataFrame(rows)


def synthetic_reporting() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": [f"w{index:02d}" for index in range(15)],
            "row_idx": np.zeros(15, dtype=np.int64),
            "suffix_offset": np.zeros(15, dtype=np.int64),
            "fold": np.asarray([index % 5 for index in range(15)], dtype=np.int64),
        }
    )


def test_frozen_contract_is_zero_pf_zero_model_and_run_is_approved_fail_closed(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)

    assert contract["target_free_freeze"]["recovery_components"] == [
        "resampling_rate",
        "ess_collapse",
        "seed_prediction_dispersion",
        "seed_likelihood_gap",
    ]
    assert contract["target_free_freeze"]["damage_components"] == [
        "eval_missing_fraction",
        "suffix_horizon",
    ]
    assert contract["target_free_freeze"]["primary_target_cell"] == train.TARGET_CELL
    assert contract["execution_counts"] == {
        "saved_output_readout_contracts": 1,
        "new_prediction_rows": 0,
        "scientific_pf_variants": 0,
        "candidate_pf_well_runs": 0,
        "parent_pf_control_reruns": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "reporting_folds": 5,
    }
    assert len(contract["scientific_contract_sha256"]) == 64
    approved = train.validate_scientific_contract(
        config,
        require_run_approval=True,
    )
    assert approved["scientific_contract_sha256"] == contract["scientific_contract_sha256"]
    broken = copy.deepcopy(config)
    broken["execution"]["audit_run_approved"] = False
    with pytest.raises(RuntimeError, match="not approved"):
        train.validate_scientific_contract(broken, require_run_approval=True)


def test_raw_diagnostic_formulas_are_exact_and_finite(train: ModuleType) -> None:
    audit = synthetic_audit().iloc[:1].copy()
    audit.loc[0, "prefix_rows"] = 20
    audit.loc[0, "prefix_gr_missing_rows"] = 5
    audit.loc[0, "eval_rows"] = 10
    audit.loc[0, "eval_raw_gr_missing_rows"] = 2
    audit.loc[0, "seeds"] = 4
    audit.loc[0, "particles"] = 100
    audit.loc[0, "resampling_count_total"] = 20
    audit.loc[0, "minimum_ess_mean"] = 25
    audit.loc[0, "seed_prediction_std_mean"] = 3
    audit.loc[0, "seed_loglik_mean_per_row"] = -5
    audit.loc[0, "seed_loglik_best_per_row"] = -2
    audit.loc[0, "position_clip_count_total"] = 8

    observed = train.build_raw_diagnostics(audit).iloc[0]

    assert observed["resampling_rate"] == pytest.approx(0.5)
    assert observed["ess_collapse"] == pytest.approx(0.75)
    assert observed["seed_prediction_dispersion"] == pytest.approx(np.log1p(3.0))
    assert observed["seed_likelihood_gap"] == pytest.approx(np.log1p(3.0))
    assert observed["eval_missing_fraction"] == pytest.approx(0.2)
    assert observed["suffix_horizon"] == pytest.approx(np.log1p(10.0))
    assert observed["position_clip_rate"] == pytest.approx(0.2)
    assert observed["prefix_missing_fraction"] == pytest.approx(0.25)

    broken = audit.copy()
    broken["eval_rows"] = 0
    with pytest.raises(ValueError, match="denominator"):
        train.build_raw_diagnostics(broken)


def test_parent_logical_sha_uses_exact_exp416_freeze_columns(
    train: ModuleType,
) -> None:
    assert train.PARENT_LOGICAL_COLUMNS == (
        "id",
        "well_id",
        "row_idx",
        train.PRIMARY_CANDIDATE,
    )
    assert set(train.PARENT_LOGICAL_COLUMNS) < set(train.PARENT_PREDICTION_COLUMNS)


def test_outer_fold_ecdf_and_medians_are_target_free_and_fold_safe(
    train: ModuleType,
    config: dict,
) -> None:
    audit = synthetic_audit()
    reporting = synthetic_reporting()
    first = train.build_fold_safe_regime_features(audit, reporting, config)

    changed = audit.copy()
    changed.loc[changed["well_id"].isin(["w00", "w05", "w10"]), "resampling_count_total"] *= 5
    second = train.build_fold_safe_regime_features(changed, reporting, config)

    first_fold0 = first.loc[first["fold"].eq(0)].sort_values("well_id")
    second_fold0 = second.loc[second["fold"].eq(0)].sort_values("well_id")
    np.testing.assert_array_equal(
        first_fold0["recovery_pressure_outer_median"].to_numpy(),
        second_fold0["recovery_pressure_outer_median"].to_numpy(),
    )
    assert not np.array_equal(
        first_fold0["resampling_rate_ecdf"].to_numpy(),
        second_fold0["resampling_rate_ecdf"].to_numpy(),
    )
    assert set(first["regime_cell"].unique()) <= {
        "high_recovery_pressure__low_damage_exposure",
        "high_recovery_pressure__high_damage_exposure",
        "low_recovery_pressure__low_damage_exposure",
        "low_recovery_pressure__high_damage_exposure",
    }
    assert first["well_id"].is_unique


def test_row_scope_uses_only_fixed_quartile_gr_and_1000ft_bins(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    audit = synthetic_audit()
    reporting_wells = synthetic_reporting()
    features = train.build_fold_safe_regime_features(audit, reporting_wells, config)
    prediction_rows: list[dict[str, object]] = []
    reporting_rows: list[dict[str, object]] = []
    for well_index in range(15):
        well = f"w{well_index:02d}"
        fold = well_index % 5
        for offset in range(5):
            prediction_rows.append(
                {
                    "id": f"{well}_{offset}",
                    "well_id": well,
                    "row_idx": offset,
                    "suffix_offset": offset,
                    "last_known_tvt": 100.0,
                    "md_since": [0.0, 999.9, 1000.0, 1200.0, 1500.0][offset],
                    "raw_gr_observed": offset % 2 == 0,
                    train.PRIMARY_CANDIDATE: np.float32(100.0 + offset),
                }
            )
            reporting_rows.append(
                {
                    "well_id": well,
                    "row_idx": offset,
                    "suffix_offset": offset,
                    "fold": fold,
                }
            )
    prediction = pd.DataFrame(prediction_rows)
    reporting = pd.DataFrame(reporting_rows)
    scope = train.build_row_scope_freeze(prediction, reporting, features)

    assert scope["normalized_suffix_progress"].drop_duplicates().tolist() == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert scope["suffix_progress_quartile"].drop_duplicates().tolist() == [
        "q1_000_025",
        "q2_025_050",
        "q3_050_075",
        "q4_075_100",
    ]
    assert set(scope["raw_gr_status"]) == {"observed", "missing"}
    assert set(scope["md_since_1000_ft"]) == {"below", "at_or_above"}

    ledger = train.OutcomeAccessLedger()
    _, _, frozen, paths = train.freeze_target_free_regime(
        prediction,
        audit,
        reporting,
        config,
        tmp_path,
        ledger,
    )
    assert ledger.regime_frozen is True
    assert all(
        value == 0
        for value in frozen["outcome_access_ledger_at_freeze"][
            "outcome_before_freeze"
        ].values()
    )
    assert all(path.exists() for path in paths.values())
    assert len(frozen["feature_logical_content_sha256"]) == 64
    assert len(frozen["row_scope_logical_content_sha256"]) == 64


def test_late_outcome_ledger_fails_closed_before_freeze(train: ModuleType) -> None:
    ledger = train.OutcomeAccessLedger()
    with pytest.raises(RuntimeError, match="requires a frozen"):
        ledger.require_frozen()
    ledger.truth_rows_before_freeze = 1
    with pytest.raises(RuntimeError, match="outcome rows"):
        ledger.mark_frozen()


def test_within_fold_permutation_is_deterministic_and_directional(
    train: ModuleType,
) -> None:
    score = np.arange(25, dtype=np.float64)
    gain = score * 0.25 + np.asarray([index % 5 for index in range(25)]) * 0.01
    folds = np.asarray([index % 5 for index in range(25)], dtype=np.int64)

    first = train.within_fold_permutation_p(
        score,
        gain,
        folds,
        direction="positive",
        permutations=256,
        label="recovery_pressure_score",
    )
    second = train.within_fold_permutation_p(
        score,
        gain,
        folds,
        direction="positive",
        permutations=256,
        label="recovery_pressure_score",
    )

    assert first == second
    assert first["observed_rho"] > 0.99
    assert first["one_sided_p"] <= 0.025
    assert len(set(first["fold_seeds"].values())) == 5


def test_parent_terminal_failure_cannot_be_reclassified(
    train: ModuleType,
    config: dict,
) -> None:
    parent_contract = {
        "scientific_contract_sha256": config["data"]["exp416_merge_output"][
            "expected_scientific_contract_sha256"
        ]
    }
    decision = config["data"]["exp416_terminal_contract"]["decision"]
    status = config["data"]["exp416_terminal_contract"]["status"]
    manifest_sha = config["data"]["exp416_merge_output"][
        "expected_artifact_manifest_raw_sha256"
    ]
    parent_gate = {"passed": False, "decision": decision}
    parent_summary = {
        "status": status,
        "artifact_manifest_sha256": manifest_sha,
        "gate": {"passed": False, "decision": decision},
    }

    report = train.validate_parent_terminal_contract(
        parent_contract,
        parent_gate,
        parent_summary,
        config,
    )

    assert report["passed"] is True
    assert report["parent_reclassified"] is False
    broken = copy.deepcopy(parent_gate)
    broken["passed"] = True
    with pytest.raises(ValueError, match="terminal contract"):
        train.validate_parent_terminal_contract(
            parent_contract,
            broken,
            parent_summary,
            config,
        )


def test_scientific_gate_is_strict_and_cannot_override_technical_failure(
    train: ModuleType,
    config: dict,
) -> None:
    local = copy.deepcopy(config)
    local["validation"]["expected_rows"] = 20
    local["validation"]["expected_wells"] = 10
    local["validation"]["expected_persistent_episode_wells"] = 3
    local["validation"]["expected_persistent_episodes"] = 4
    local["validation"]["expected_persistent_episode_rows"] = 40
    local["guards"]["technical"]["require_expected_rows"] = 20
    local["guards"]["technical"]["require_expected_wells"] = 10
    local["guards"]["technical"]["require_well_audit_rows"] = 10
    local["guards"]["technical"]["require_by_well_metric_rows"] = 10
    local["guards"]["technical"]["require_persistent_episode_count"] = 4
    local["guards"]["technical"]["require_persistent_episode_wells"] = 3
    local["guards"]["technical"]["require_persistent_episode_rows"] = 40

    wells = [f"w{index}" for index in range(10)]
    target_wells = set(wells[:5])
    frame = pd.DataFrame(
        {
            "id": [f"w{index // 2}_{index % 2}" for index in range(20)],
            "well_id": [f"w{index // 2}" for index in range(20)],
            "row_idx": [index % 2 for index in range(20)],
            "fold": [index // 2 % 5 for index in range(20)],
            train.TRUE_TVT: np.full(20, 100.0),
            train.PRIMARY_CONTROL: np.full(20, 101.0),
            train.PRIMARY_CANDIDATE: np.asarray(
                [
                    100.5 if f"w{index // 2}" in target_wells else 102.0
                    for index in range(20)
                ]
            ),
        }
    )
    by_well = pd.DataFrame(
        {
            "well_id": wells,
            "improvement_ft": [0.5] * 5 + [-1.0] * 5,
            "is_primary_target_cell": [True] * 5 + [False] * 5,
        }
    )
    episodes = pd.DataFrame(
        {
            "episode_id": [f"e{index}" for index in range(4)],
            "well_id": ["w0", "w1", "w2", "w0"],
            "rows": [10] * 4,
        }
    )
    overall_metrics = pd.DataFrame(
        [
            {
                "scope_type": "overall",
                "scope": "overall",
                "candidate_rmse": 1.5,
                "control_rmse": 1.0,
                "improvement_ft": -0.5,
            }
        ]
    )
    regime_rows = [
        {
            "scope_type": "regime",
            "regime_cell": train.TARGET_CELL,
            "fold": np.nan,
            "improvement_ft": 0.5,
        }
    ]
    regime_rows.extend(
        {
            "scope_type": "regime_fold",
            "regime_cell": train.TARGET_CELL,
            "fold": fold,
            "improvement_ft": 0.5,
        }
        for fold in range(5)
    )
    score_rows: list[dict[str, object]] = []
    for score, pooled_rho, fold_rho in (
        ("recovery_pressure_score", 0.5, 0.5),
        ("damage_exposure_score", -0.5, -0.5),
    ):
        score_rows.append(
            {"score": score, "scope": "pooled", "fold": np.nan, "rho": pooled_rho}
        )
        score_rows.extend(
            {"score": score, "scope": "fold", "fold": fold, "rho": fold_rho}
            for fold in range(5)
        )
    permutation = {
        "recovery_pressure_score": {"one_sided_p": 0.01},
        "damage_exposure_score": {"one_sided_p": 0.01},
    }
    episode_summary = {
        "episodes": 4,
        "wells": 3,
        "sse_reduction_fraction": 0.10,
        "share_of_positive_episode_sse_reduction": 0.60,
    }
    freeze = {
        "outcome_access_ledger_at_freeze": {
            "prefreeze_safe_rows": {"well_audit": 10},
            "outcome_before_freeze": {
                "truth": 0,
                "control": 0,
                "by_well": 0,
                "persistent_episode": 0,
            },
        },
        **{
            key: "a" * 64
            for key in (
                "feature_schema_sha256",
                "feature_logical_content_sha256",
                "assignment_schema_sha256",
                "assignment_logical_content_sha256",
                "row_scope_schema_sha256",
                "row_scope_logical_content_sha256",
            )
        },
    }
    parity = {
        "candidate_absolute_difference_ft": 0.0,
        "control_absolute_difference_ft": 0.0,
        "candidate_minus_control_absolute_difference_ft": 0.0,
        "by_well_max_absolute_differences_ft": {
            "candidate_rmse": 0.0,
            "control_rmse": 0.0,
            "improvement_ft": 0.0,
        },
    }
    preflight = {
        "all_input_sha_matches": True,
        "parent_terminal_contract": {
            "passed": True,
            "parent_reclassified": False,
            "decision": "roughening_x10_rejected_close_without_rescue",
        },
        "source_kernel": {"version": 2, "id_no": 128_912_230},
    }

    gate = train.evaluate_attribution_gate(
        preflight,
        freeze,
        parity,
        frame,
        by_well,
        episodes,
        overall_metrics,
        pd.DataFrame(regime_rows),
        pd.DataFrame(score_rows),
        permutation,
        episode_summary,
        local,
    )
    assert gate["technical_gate"]["passed"] is True
    assert gate["scientific_gate"]["passed"] is True
    assert gate["passed"] is True
    assert gate["parent_decision_remains_unchanged"] is True
    assert gate["policy_ready"] is False

    broken = copy.deepcopy(preflight)
    broken["all_input_sha_matches"] = False
    failed = train.evaluate_attribution_gate(
        broken,
        freeze,
        parity,
        frame,
        by_well,
        episodes,
        overall_metrics,
        pd.DataFrame(regime_rows),
        pd.DataFrame(score_rows),
        permutation,
        episode_summary,
        local,
    )
    assert failed["scientific_gate"]["passed"] is True
    assert failed["technical_gate"]["passed"] is False
    assert failed["passed"] is False
    assert (
        failed["decision"]
        == "no_reproducible_target_free_regime_close_attribution_branch"
    )


def test_compact_source_is_not_thin_and_is_not_file_dependent() -> None:
    text = SOURCE.read_text()
    assert text.count("# %% [markdown]") >= 11
    assert "def build_fold_safe_regime_features" in text
    assert "def freeze_target_free_regime" in text
    assert "def load_late_outcomes" in text
    assert "def within_fold_permutation_p" in text
    assert "def evaluate_attribution_gate" in text
    assert "def run_full_experiment" in text
    assert "Path(__file__)" not in text
    assert "__file__" not in text

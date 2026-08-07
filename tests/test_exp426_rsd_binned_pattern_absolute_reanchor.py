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

ROOT = Path(__file__).resolve().parents[1]
EXP = "exp426_rsd_binned_pattern_absolute_reanchor"
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
    previous = os.environ.get("EXP426_IMPORT_ONLY")
    os.environ["EXP426_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp426_stage_a_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP426_IMPORT_ONLY", None)
        else:
            os.environ["EXP426_IMPORT_ONLY"] = previous


@pytest.fixture
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_well(train: ModuleType) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_rows = 340
    known_rows = 60
    tvt = 20.0 + 0.5 * np.arange(total_rows, dtype=np.float64)

    def gr_curve(values: np.ndarray) -> np.ndarray:
        scaled = values / 75.0
        return 45.0 + 13.0 * np.sin(values / 11.0) + 7.0 * scaled**2 + 2.0 * scaled**3

    horizontal = pd.DataFrame(
        {
            "MD": np.arange(total_rows, dtype=np.float64),
            "GR": gr_curve(tvt),
            "TVT_input": np.r_[tvt[:known_rows], np.full(total_rows - known_rows, np.nan)],
        }
    )
    typewell_tvt = np.arange(0.0, 260.0, 0.1)
    typewell = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": gr_curve(typewell_tvt),
        }
    )
    row_idx = np.arange(known_rows, total_rows, dtype=np.int64)
    safe = pd.DataFrame(
        {
            "well_id": "well-a",
            "row_idx": row_idx,
            "suffix_offset": np.arange(len(row_idx), dtype=np.int64),
            "fold": 0,
            "tvt_pred": tvt[row_idx] - 10.0,
        }
    )
    return safe, horizontal, typewell


def test_stage_a_contract_is_implementation_ready_and_zero_model(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)

    assert contract["stage"] == "stage_a_absolute_datum_identifiability"
    assert contract["base_path"] == "exp226_final_tvt_pred"
    assert contract["offsets_ft"] == train.OFFSETS_FT.tolist()
    assert contract["execution_counts"] == {
        "primary_scores": 1,
        "descriptive_scores": 2,
        "matched_controls": 3,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_runs": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    assert len(contract["scientific_contract_sha256"]) == 64

    with pytest.raises(RuntimeError, match="not approved"):
        train.validate_scientific_contract(config, require_run_approval=True)
    approved_config = copy.deepcopy(config)
    approved_config["execution"]["run_stage_a"] = True
    approved = train.validate_scientific_contract(
        approved_config,
        require_run_approval=True,
    )
    assert approved["scientific_contract_sha256"] == contract["scientific_contract_sha256"]
    unapproved = copy.deepcopy(config)
    unapproved["execution"]["kaggle_execution_authorized"] = False
    with pytest.raises(RuntimeError, match="not approved"):
        train.validate_scientific_contract(
            unapproved,
            require_run_approval=True,
        )

    broken = copy.deepcopy(config)
    broken["model"]["stage_b"]["enabled"] = True
    with pytest.raises(ValueError, match="Stage B and Stage C"):
        train.validate_scientific_contract(broken)


def test_rsd_binning_uses_raw_finite_gr_and_no_typewell_extrapolation(
    train: ModuleType,
) -> None:
    typewell_tvt = np.arange(0.0, 20.0, 0.1)
    typewell_gr = 5.0 + np.sin(typewell_tvt) + typewell_tvt**2
    candidate = np.arange(1.0, 17.0, 0.5)
    observed = np.interp(candidate, typewell_tvt, typewell_gr)
    observed[3] = np.nan

    score = train.rsd_binned_pattern_score(
        observed,
        candidate,
        typewell_tvt,
        typewell_gr,
        bin_width_ft=0.5,
        bin_origin_ft=0.0,
        minimum_raw_points=16,
        minimum_paired_bins=16,
        minimum_std=1.0e-6,
        correlation_clip_epsilon=1.0e-6,
    )
    assert score["valid"] is True
    assert score["raw_finite_points"] == 31
    assert score["paired_bins"] == 31
    assert score["pearson"] > 0.999

    unsupported = train.rsd_binned_pattern_score(
        observed,
        candidate + 100.0,
        typewell_tvt,
        typewell_gr,
        bin_width_ft=0.5,
        bin_origin_ft=0.0,
        minimum_raw_points=16,
        minimum_paired_bins=16,
        minimum_std=1.0e-6,
        correlation_clip_epsilon=1.0e-6,
    )
    assert unsupported["valid"] is False
    assert unsupported["paired_bins"] == 0


def test_target_free_score_bank_identifies_known_shift(
    train: ModuleType,
    config: dict,
) -> None:
    safe, horizontal, typewell = synthetic_well(train)
    scores, manifest = train.score_well_target_free(safe, horizontal, typewell, config)

    assert manifest["blocks"] == 1
    assert manifest["evaluation_rows"] == len(safe)
    assert len(scores) == 13
    top = scores.loc[scores["rsd_valid"]].sort_values("rsd_rank").iloc[0]
    assert top["offset_ft"] == pytest.approx(10.0)
    assert top["rsd_pearson"] > 0.999
    assert scores["raw_finite_gr_points"].eq(len(safe)).all()
    assert scores["rsd_top3"].sum() == 3
    assert train.validate_score_bank_structure(scores) == {
        "required_columns": True,
        "duplicate_identity_zero": True,
        "canonical_order": True,
        "finite_score_storage": True,
        "thirteen_offsets_per_block": True,
        "fixed_offset_order": True,
        "rank_permutations": True,
        "top3_masks_consistent": True,
    }
    well_manifest = pd.DataFrame([manifest])
    input_manifest = pd.DataFrame(
        [
            {
                "well_id": "well-a",
                "horizontal_raw_sha256": "a" * 64,
                "typewell_raw_sha256": "b" * 64,
            }
        ]
    )
    freeze = train.build_target_free_freeze(
        scores,
        well_manifest,
        input_manifest,
        config=config,
        runtime_seconds=1.0,
        peak_memory_gb=0.1,
        probe_logical_sha_match=True,
        truth_ledger=train.TruthAccessLedger(),
        strict_inventory=False,
    )
    assert freeze["technical_passed"] is True
    assert len(freeze["score_content_sha256"]) == 64
    assert len(freeze["input_manifest_content_sha256"]) == 64
    assert len(freeze["config_content_sha256"]) == 64


def test_tie_order_and_stable_permutation_are_deterministic(
    train: ModuleType,
) -> None:
    scores = np.ones(13, dtype=np.float64)
    valid = np.ones(13, dtype=bool)
    ranks, order = train.rank_scores(scores, valid)
    assert train.OFFSETS_FT[order[:5]].tolist() == [0.0, -2.0, 2.0, -5.0, 5.0]
    assert ranks[np.flatnonzero(train.OFFSETS_FT == 0.0)[0]] == 1

    first_scores, first_valid = train.stable_score_label_permutation(
        np.arange(13, dtype=np.float64),
        valid,
        well_id="well-a",
        block_id=3,
    )
    second_scores, second_valid = train.stable_score_label_permutation(
        np.arange(13, dtype=np.float64),
        valid,
        well_id="well-a",
        block_id=3,
    )
    np.testing.assert_array_equal(first_scores, second_scores)
    np.testing.assert_array_equal(first_valid, second_valid)
    np.testing.assert_array_equal(np.sort(first_scores), np.arange(13))
    assert not np.array_equal(first_scores, np.arange(13))


def test_truth_and_hidden_role_access_fail_closed_before_freeze(
    train: ModuleType,
) -> None:
    ledger = train.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="before target-free freeze"):
        ledger.register_truth_access(7)
    with pytest.raises(RuntimeError, match="before target-free freeze"):
        ledger.register_hidden_role_access(3)
    assert ledger.truth_rows_before_freeze == 7
    assert ledger.hidden_role_rows_before_freeze == 3

    clean = train.TruthAccessLedger()
    clean.mark_frozen("a" * 64)
    clean.register_truth_access(7)
    clean.register_hidden_role_access(3)
    assert clean.truth_rows_before_freeze == 0
    assert clean.hidden_role_rows_before_freeze == 0
    assert clean.truth_rows_after_freeze == 7
    assert clean.hidden_role_rows_after_freeze == 3


def test_discrete_oracle_and_post_freeze_replay_use_fixed_offsets(
    train: ModuleType,
    config: dict,
) -> None:
    safe, horizontal, typewell = synthetic_well(train)
    score_bank, _ = train.score_well_target_free(safe, horizontal, typewell, config)
    true_tvt = safe["tvt_pred"].to_numpy(np.float64) + 10.0
    truth = safe[["well_id", "row_idx"]].assign(tvt_true=true_tvt)
    hidden = pd.DataFrame(
        {
            "well_id": ["well-a"],
            "verification_like_spatial_role": ["valid"],
            "verification_like_typewell_purged_role": ["valid"],
        }
    )
    readout = train.build_post_freeze_block_readout(
        score_bank,
        safe,
        truth,
        hidden,
        config,
    )
    primary = readout.loc[readout["variant"].eq(train.PRIMARY_VARIANT)].iloc[0]
    assert primary["oracle_offset_ft"] == pytest.approx(10.0)
    assert primary["selected_offset_ft"] == pytest.approx(10.0)
    assert bool(primary["top1_exact"])
    assert bool(primary["top3_coverage"])
    assert primary["replay_sse"] == pytest.approx(0.0, abs=1e-9)
    assert bool(primary["hidden_like_spatial"])
    assert bool(primary["hidden_like_typewell_purged"])


def passing_metric_frames(train: ModuleType) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    exact = {
        train.PRIMARY_VARIANT: 0.70,
        "raw_pointwise_pearson": 0.60,
        "raw_gaussian": 0.58,
        "stable_permutation": 0.40,
    }
    parent_rmse = 9.427109596582213
    for variant in train.VARIANTS:
        for scope in (
            "pooled",
            "distance_1000_plus",
            "hidden_like_spatial",
            "hidden_like_typewell_purged",
        ):
            scope_rows.append(
                {
                    "variant": variant,
                    "scope": scope,
                    "top1_exact": exact[variant],
                    "top3_coverage": 0.75,
                    "direction_accuracy": 0.70,
                    "parent_rmse": parent_rmse,
                    "replay_rmse": parent_rmse - 0.2,
                    "replay_gain_ft": 0.2,
                }
            )
        for fold in range(5):
            fold_rows.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "scope": "pooled",
                    "top1_exact": exact[variant],
                    "top3_coverage": 0.75,
                    "direction_accuracy": 0.70,
                    "parent_rmse": parent_rmse,
                    "replay_rmse": parent_rmse - 0.2,
                    "replay_gain_ft": 0.2,
                }
            )
    return pd.DataFrame(scope_rows), pd.DataFrame(fold_rows)


def test_stage_a_gate_is_fail_closed_and_never_implements_later_stages(
    train: ModuleType,
    config: dict,
) -> None:
    scopes, folds = passing_metric_frames(train)
    gate, decision = train.evaluate_stage_a_gate(
        scopes,
        folds,
        {"technical_passed": True},
        config,
    )
    assert gate["passed"] is True
    assert decision == {
        "action": "request_separate_stage_b_implementation_authorization",
        "stage_b_implemented": False,
        "stage_c_implemented": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }

    broken = scopes.copy()
    broken.loc[
        broken["variant"].eq(train.PRIMARY_VARIANT) & broken["scope"].eq("pooled"),
        "top1_exact",
    ] = 0.20
    gate, decision = train.evaluate_stage_a_gate(
        broken,
        folds,
        {"technical_passed": True},
        config,
    )
    assert gate["passed"] is False
    assert decision["action"].startswith("close_stage_a_without")
    assert decision["stage_b_implemented"] is False
    assert decision["stage_c_implemented"] is False


def test_canonical_train_is_adopted_and_inference_remains_placeholder(
    config: dict,
) -> None:
    canonical_train = (EXP_DIR / f"{EXP}_train.ipynb").read_text()
    canonical_inference = (EXP_DIR / f"{EXP}_inference.ipynb").read_text()
    source = SOURCE.read_text()

    assert config["experiment"]["status"] == "completed_stage_a_technical_failed_closed"
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["implementation_authorized"] is True
    assert config["execution"]["kaggle_package_authorized"] is True
    assert config["execution"]["kaggle_push_authorized"] is True
    assert config["execution"]["kaggle_execution_authorized"] is True
    assert config["execution"]["run_stage_a"] is False
    assert config["execution"]["stage_a_completed"] is True
    assert "__file__" not in source
    assert "Stage A scientific contract" in canonical_train
    assert "run_stage_a_experiment" in canonical_train
    assert "shutil.copyfile(sample_submission, submission_path)" in canonical_inference

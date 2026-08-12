from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp359_exp226_window_likelihood_on_exp281"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp359_exp226_window_likelihood_on_exp281_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp359_exp226_window_likelihood_on_exp281_compact_selfcontained_inference.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train() -> ModuleType:
    return load_module(TRAIN_SOURCE, "exp359_train_test")


@pytest.fixture(scope="module")
def config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(value, dict)
    return value


def synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    known_rows = 80
    suffix_rows = 800
    total_rows = known_rows + suffix_rows
    row_idx = np.arange(total_rows)
    typewell_tvt = np.arange(850.0, 1250.1, 0.1)

    def gr_curve(tvt: np.ndarray) -> np.ndarray:
        return (
            70.0
            + 12.0 * np.sin(tvt / 7.0)
            + 4.0 * np.sin(tvt / 1.3)
            + 0.01 * (tvt - 1000.0)
        )

    prefix_tvt = 992.0 + 0.1 * np.arange(known_rows)
    suffix_geop = 1000.0 + 0.1 * np.arange(suffix_rows)
    tvt_input = np.full(total_rows, np.nan)
    tvt_input[:known_rows] = prefix_tvt
    horizontal_tvt = np.concatenate([prefix_tvt, suffix_geop + 5.0])
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(total_rows, dtype=float) * 0.5,
            "GR": gr_curve(horizontal_tvt),
            "TVT_input": tvt_input,
        }
    )
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": gr_curve(typewell_tvt)})
    oof = pd.DataFrame(
        {
            "well_id": "well0001",
            "row_idx": row_idx[known_rows:],
            "suffix_offset": np.arange(suffix_rows),
            "fold": 0,
            "tvt_geop": suffix_geop,
        }
    )
    return oof, horizontal, typewell


def test_config_freezes_stage0_and_keeps_expensive_routes_off(
    train: ModuleType,
    config: dict,
) -> None:
    train.validate_scientific_contract(config, require_run_approval=False)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["implementation"]["scope"] == "stage_0_rank_audit_only"
    assert config["implementation"]["canonical_notebook_adopted"] is True
    assert config["implementation"]["stage_1_implemented"] is False
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["run_stage_1"] is False
    assert config["execution"]["run_inference"] is False
    assert config["execution"]["create_submission"] is False
    assert config["execution_contract"]["stage_0"] == {
        "scientific_scores": 1,
        "saved_control_scores": 1,
        "reporting_folds": 5,
        "hmm_well_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    if config["execution"]["run_stage_0"]:
        train.validate_scientific_contract(config, require_run_approval=True)
    else:
        with pytest.raises(PermissionError):
            train.validate_scientific_contract(config, require_run_approval=True)


def test_profile_score_normalization_and_lambda_are_fixed(
    train: ModuleType,
    config: dict,
) -> None:
    oof, horizontal, typewell = synthetic_well()
    scores, manifest = train.score_well_window_target_free(
        oof,
        horizontal,
        typewell,
        config,
    )
    assert manifest["candidate_windows"] == 3
    assert manifest["eligible_windows"] == 3
    assert len(scores) == 3 * 13
    assert scores.groupby("window_id").size().eq(13).all()
    assert scores["eligible_window"].all()
    assert np.isfinite(
        scores[
            [
                "raw_score",
                "normalized_score",
                "potential_score",
                "posterior_sd_ft",
                "window_lambda",
            ]
        ].to_numpy()
    ).all()
    for _, part in scores.groupby("window_id"):
        assert part["normalized_score"].mean() == pytest.approx(0.0, abs=1e-10)
        assert part["normalized_score"].std(ddof=0) == pytest.approx(1.0, abs=1e-10)
        assert part["window_lambda"].nunique() == 1
        assert part["window_lambda"].iloc[0] == pytest.approx(
            0.25
            * np.clip(1.1 - 0.12 * part["posterior_sd_ft"].iloc[0], 0.3, 1.0)
        )
        top_shift = part.sort_values("potential_rank").iloc[0]["shift_ft"]
        assert top_shift == pytest.approx(5.0)


def test_negative_control_permutation_is_stable_and_not_identity(
    train: ModuleType,
) -> None:
    first = train.stable_score_permutation("well", 7, "abc", 13)
    second = train.stable_score_permutation("well", 7, "abc", 13)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(13))
    assert not np.array_equal(first, np.arange(13))


def test_target_free_scoring_rejects_truth_columns(
    train: ModuleType,
    config: dict,
) -> None:
    oof, horizontal, typewell = synthetic_well()
    oof["tvt_true"] = oof["tvt_geop"] + 5.0
    with pytest.raises(ValueError, match="forbidden"):
        train.score_well_window_target_free(oof, horizontal, typewell, config)


def test_saved_control_alignment_and_truth_late_readout(
    train: ModuleType,
    config: dict,
) -> None:
    oof, horizontal, typewell = synthetic_well()
    window_scores, _ = train.score_well_window_target_free(
        oof,
        horizontal,
        typewell,
        config,
    )
    blocks = sorted(window_scores["control_block_id"].unique())
    control_rows = []
    for block in blocks:
        control_score = -np.abs(train.EXPECTED_SHIFTS - 2.0)
        control_rank = train.rank_descending(control_score)
        for slot, shift in enumerate(train.EXPECTED_SHIFTS):
            control_rows.append(
                {
                    "well_id": "well0001",
                    "fold": 0,
                    "block_id": int(block),
                    "block_start_suffix_offset": int(block * 512),
                    "block_end_suffix_offset": int(block * 512 + 511),
                    "shift_slot": slot,
                    "shift_ft": shift,
                    "likelihood_mean": control_score[slot],
                    "likelihood_rank": control_rank[slot],
                }
            )
    bundle, technical = train.align_saved_control_to_windows(
        window_scores,
        pd.DataFrame(control_rows),
    )
    assert technical["saved_control_rank_parity"] == 1.0
    assert technical["row_identity_coverage"] == 1.0
    truth = oof[["well_id", "row_idx"]].copy()
    truth["tvt_true"] = oof["tvt_geop"] + 5.0
    readout = train.build_truth_readout(bundle, oof, truth, config)
    assert len(readout) == 3
    assert set(readout["nearest_shift_ft"]) == {5.0}
    assert readout["window_top1_hit"].all()
    assert (readout["window_mrr"] > readout["control_mrr"]).all()


def test_frozen_gate_requires_all_directions(
    train: ModuleType,
    config: dict,
) -> None:
    scope_names = [
        "overall",
        "long_tail_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ]
    scope_metrics = pd.DataFrame(
        {
            "scope": scope_names,
            "window_minus_control_mrr": [0.02] * 4,
            "window_minus_control_top3_rate": [0.02] * 4,
        }
    )
    fold_metrics = pd.DataFrame(
        {
            "fold": [0, 1, 2, 3, 4],
            "window_minus_control_mrr": [0.02] * 5,
            "window_minus_control_top3_rate": [0.02] * 5,
            "window_minus_shuffle_mrr": [0.01] * 5,
            "window_minus_shuffle_top3_rate": [0.01] * 5,
        }
    )
    technical = {
        "score_finite_coverage": 1.0,
        "row_identity_coverage": 1.0,
        "saved_control_rank_parity": 1.0,
    }
    guard = train.evaluate_guard(
        technical,
        scope_metrics,
        fold_metrics,
        eligible_window_fraction=0.5,
        config=config,
    )
    assert guard["passed"] is True
    degraded = fold_metrics.copy()
    degraded.loc[4, "window_minus_shuffle_mrr"] = 0.0
    failed = train.evaluate_guard(
        technical,
        scope_metrics,
        degraded,
        eligible_window_fraction=0.5,
        config=config,
    )
    assert failed["passed"] is False
    assert failed["checks"]["real_above_shuffle_all_folds"] is False


def test_compact_sources_are_not_thin_and_canonical_train_is_adopted() -> None:
    train_text = TRAIN_SOURCE.read_text()
    assert train_text.count("# %% [markdown]") >= 10
    assert "def score_well_window_target_free" in train_text
    assert "def build_truth_readout" in train_text
    assert "def evaluate_guard" in train_text
    assert "__file__" not in train_text
    assert "from settings import" not in train_text
    canonical_train = json.loads(
        (EXP_DIR / "exp359_exp226_window_likelihood_on_exp281_train.ipynb").read_text()
    )
    compact_train = json.loads(
        (
            EXP_DIR
            / "exp359_exp226_window_likelihood_on_exp281_compact_selfcontained_train.ipynb"
        ).read_text()
    )
    assert len(canonical_train["cells"]) == 21
    assert [cell["source"] for cell in canonical_train["cells"]] == [
        cell["source"] for cell in compact_train["cells"]
    ]


def test_inference_candidate_refuses_prediction(config: dict) -> None:
    inference = load_module(INFERENCE_SOURCE, "exp359_inference_test")
    contract = inference.validate_disabled_inference(config)
    assert contract["status"] == "inference_disabled"
    with pytest.raises(RuntimeError, match="Stage 0 produces only"):
        inference.refuse_inference(config)

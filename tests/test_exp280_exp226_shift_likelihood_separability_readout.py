from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp280_exp226_shift_likelihood_separability_readout"
TRAIN_SOURCE = EXP_DIR / "exp280_exp226_shift_likelihood_separability_readout_train.py"
INFERENCE_SOURCE = EXP_DIR / "exp280_exp226_shift_likelihood_separability_readout_inference.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_target_free_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    known_rows = 30
    eval_rows = 100
    rows = known_rows + eval_rows
    true_tvt = 100.0 + 0.2 * np.arange(rows, dtype=float)
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float),
            "GR": 2.0 * true_tvt,
            "TVT_input": np.r_[true_tvt[:known_rows], np.full(eval_rows, np.nan)],
        }
    )
    typewell_tvt = np.linspace(0.0, 300.0, 1201)
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": 2.0 * typewell_tvt})
    row_idx = np.arange(known_rows, rows, dtype=np.int64)
    oof_safe = pd.DataFrame(
        {
            "well_id": "well_a",
            "row_idx": row_idx,
            "suffix_offset": np.arange(eval_rows, dtype=np.int64),
            "fold": 0,
            "tvt_geop": true_tvt[row_idx] - 10.0,
        }
    )
    truth = pd.DataFrame(
        {"well_id": "well_a", "row_idx": row_idx, "tvt_true": true_tvt[row_idx]}
    )
    return oof_safe, horizontal, typewell, truth


def test_scientific_contract_is_exactly_the_approved_readout() -> None:
    module = load_module(TRAIN_SOURCE, "exp280_train_contract")
    config = load_config()
    module.validate_scientific_contract(config)
    config["audit"]["block_rows"] = 256
    with pytest.raises(ValueError, match="512-row"):
        module.validate_scientific_contract(config)


def test_target_free_likelihood_ranks_positive_ten_shift_first() -> None:
    module = load_module(TRAIN_SOURCE, "exp280_train_score")
    oof_safe, horizontal, typewell, _ = synthetic_target_free_inputs()
    scores, manifest = module.score_well_target_free(
        oof_safe, horizontal, typewell, load_config()
    )
    selected = scores.loc[scores["likelihood_rank"] == 1]
    assert len(selected) == 1
    assert selected["shift_ft"].iloc[0] == 10.0
    assert manifest["blocks"] == 1
    assert manifest["score_finite_coverage"] == 1.0
    assert scores["shuffled_likelihood_rank"].nunique() == 13


def test_truth_is_attached_only_after_score_and_recovers_rank() -> None:
    module = load_module(TRAIN_SOURCE, "exp280_train_truth")
    oof_safe, horizontal, typewell, truth = synthetic_target_free_inputs()
    scores, _ = module.score_well_target_free(oof_safe, horizontal, typewell, load_config())
    readout, episodes = module.build_truth_readout(scores, oof_safe, truth, load_config())
    assert episodes.empty
    assert len(readout) == 1
    row = readout.iloc[0]
    assert row["nearest_shift_ft"] == 10.0
    assert row["nearest_shift_rank"] == 1
    assert bool(row["top1_hit"])
    assert bool(row["top3_hit"])
    assert bool(row["sign_match"])
    assert row["top1_regret_rmse"] == pytest.approx(0.0)


def test_target_free_scoring_rejects_exp226_truth_or_error_columns() -> None:
    module = load_module(TRAIN_SOURCE, "exp280_train_leakage")
    oof_safe, horizontal, typewell, _ = synthetic_target_free_inputs()
    oof_safe["tvt_true"] = oof_safe["tvt_geop"]
    with pytest.raises(ValueError, match="forbidden exp226 columns"):
        module.score_well_target_free(oof_safe, horizontal, typewell, load_config())


def test_persistent_offset_episode_requires_128_consecutive_rows() -> None:
    module = load_module(TRAIN_SOURCE, "exp280_train_episode")
    error = np.r_[np.full(127, 11.0), 0.0, np.full(128, -12.0), np.zeros(5)]
    mask, episodes = module.persistent_offset_episodes(
        error,
        np.arange(len(error)),
        threshold_ft=10.0,
        minimum_consecutive_rows=128,
    )
    assert len(episodes) == 1
    assert episodes[0]["episode_rows"] == 128
    assert episodes[0]["median_signed_base_error_ft"] == -12.0
    assert int(mask.sum()) == 128


def test_inference_contract_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp280_inference")
    config = load_config()
    contract = module.validate_disabled_inference(config)
    assert not contract["inference_enabled"]
    assert not contract["create_submission"]
    config["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        module.validate_disabled_inference(config)


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
EXP_DIR = ROOT / "experiments" / "exp321_z_only_residual_gr_correction_ladder"
TRAIN_PATH = (
    EXP_DIR
    / "exp321_z_only_residual_gr_correction_ladder_compact_selfcontained_train.py"
)
INFERENCE_PATH = (
    EXP_DIR
    / "exp321_z_only_residual_gr_correction_ladder_compact_selfcontained_inference.py"
)


def _load_module(path: Path, name: str):
    os.environ["EXP321_IMPORT_ONLY"] = "1"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train = _load_module(TRAIN_PATH, "exp321_train")
inference = _load_module(INFERENCE_PATH, "exp321_inference")


@pytest.fixture
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def _synthetic_target_free_inputs(
    *, suffix_rows: int = 12
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    known_rows = 6
    total_rows = known_rows + suffix_rows
    z = np.arange(total_rows, dtype=np.float64)
    known_tvt = 100.0 - z[:known_rows]
    tvt_input = np.r_[known_tvt, np.full(suffix_rows, np.nan)]
    anchor_tvt = known_tvt[-1]
    tvt_z = anchor_tvt - (z[known_rows:] - z[known_rows - 1])
    # The suffix GR is deliberately aligned to tvt_z + 10 so Stage B must rank +10.
    gr = np.r_[2.0 * known_tvt, 2.0 * (tvt_z + 10.0)]
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(total_rows, dtype=np.float64),
            "Z": z,
            "GR": gr,
            "TVT_input": tvt_input,
        }
    )
    row_idx = np.arange(known_rows, total_rows, dtype=np.int64)
    oof = pd.DataFrame(
        {
            "well_id": "well_a",
            "fold": 0,
            "row_idx": row_idx,
            "suffix_offset": np.arange(suffix_rows, dtype=np.int64),
            "tvt_geop": tvt_z + np.linspace(-1.0, 1.0, suffix_rows),
        }
    )
    typewell_tvt = np.linspace(0.0, 150.0, 601)
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": 2.0 * typewell_tvt})
    return oof, horizontal, typewell


def test_config_locks_stage_ab_only_zero_model_contract(config: dict) -> None:
    train.validate_scientific_contract(config, require_kaggle_approval=False)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["scope"] == "stage_ab_only"
    assert config["implementation"]["stage_c_implemented"] is False
    assert config["stage_c_window_gr_correction"]["enabled"] is False
    assert config["execution_contract"]["stage_ab"]["active_variants"] == 1
    assert config["execution_contract"]["stage_ab"]["model_configs"] == 0
    assert config["execution_contract"]["stage_ab"]["trained_folds"] == 0
    assert config["execution_contract"]["stage_ab"]["boosters"] == 0
    assert config["execution_contract"]["parent_control_retraining"] is False
    assert config["execution_contract"]["kaggle_push_approved"] is True
    train.validate_scientific_contract(config, require_kaggle_approval=True)


def test_z_only_path_uses_last_known_anchor_and_fixed_minus_delta_z(config: dict) -> None:
    oof, horizontal, _ = _synthetic_target_free_inputs()
    path, manifest = train.build_z_only_target_free(oof, horizontal, config)
    expected = horizontal["TVT_input"].iloc[5] - (
        horizontal["Z"].iloc[6:].to_numpy() - horizontal["Z"].iloc[5]
    )
    np.testing.assert_allclose(path["tvt_z"], expected)
    assert manifest["last_known_row_idx"] == 5
    assert manifest["row_identity_coverage"] == 1.0
    assert set(path.columns) == set(train.TARGET_FREE_PATH_COLUMNS)
    assert not {"TVT", "tvt_true", "error"}.intersection(path.columns)


def test_z_only_path_fails_closed_on_noncontiguous_prefix_or_missing_suffix_z(
    config: dict,
) -> None:
    oof, horizontal, _ = _synthetic_target_free_inputs()
    broken_prefix = horizontal.copy()
    broken_prefix.loc[2, "TVT_input"] = np.nan
    with pytest.raises(ValueError, match="contiguous known prefix"):
        train.build_z_only_target_free(oof, broken_prefix, config)
    broken_z = horizontal.copy()
    broken_z.loc[oof["row_idx"].iloc[0], "Z"] = np.nan
    with pytest.raises(ValueError, match="suffix Z"):
        train.build_z_only_target_free(oof, broken_z, config)


def test_stage_b_exp280_parity_score_ranks_positive_ten_shift(config: dict) -> None:
    oof, horizontal, typewell = _synthetic_target_free_inputs()
    path, _ = train.build_z_only_target_free(oof, horizontal, config)
    scores, manifest = train.score_z_only_shifts_target_free(
        path, horizontal, typewell, config
    )
    selected = scores.loc[scores["likelihood_rank"] == 1]
    assert len(selected) == 1
    assert selected["shift_ft"].iloc[0] == 10.0
    assert scores["shuffled_likelihood_rank"].nunique() == 13
    assert manifest["score_finite_coverage"] == 1.0
    assert not {"TVT", "tvt_true", "error"}.intersection(scores.columns)


def test_truth_join_requires_freeze_and_stage_b_recovers_nearest_shift(
    config: dict,
) -> None:
    oof, horizontal, typewell = _synthetic_target_free_inputs()
    path, _ = train.build_z_only_target_free(oof, horizontal, config)
    scores, _ = train.score_z_only_shifts_target_free(path, horizontal, typewell, config)
    truth = path[["well_id", "row_idx"]].copy()
    truth["tvt_true"] = path["tvt_z"] + 10.0
    with pytest.raises(ValueError, match="frozen target-free contract"):
        train.attach_truth_after_freeze(path, truth, target_free_contract_sha256="")
    joined = train.attach_truth_after_freeze(
        path, truth, target_free_contract_sha256="frozen_sha"
    )
    readout = train.build_stage_b_block_readout(scores, joined, config)
    assert len(readout) == 1
    assert readout["nearest_shift_ft"].iloc[0] == 10.0
    assert readout["nearest_shift_rank"].iloc[0] == 1
    assert bool(readout["top1_hit"].iloc[0])


def test_stage_a_affine_quotient_removes_only_intercept_and_slope(config: dict) -> None:
    parts: list[pd.DataFrame] = []
    for fold in range(5):
        n_rows = 16
        x = np.arange(n_rows, dtype=np.float64)
        truth = 100.0 + x
        z_residual = 2.0 + 0.1 * x
        geop_residual = 2.0 + 0.1 * x + 0.02 * np.square(x - x.mean())
        parts.append(
            pd.DataFrame(
                {
                    "well_id": f"well_{fold}",
                    "fold": fold,
                    "row_idx": x.astype(np.int64),
                    "suffix_offset": x.astype(np.int64),
                    "md_since_ft": x + 1001.0,
                    "tvt_z": truth - z_residual,
                    "tvt_geop": truth - geop_residual,
                    "block_h128": 0,
                    "block_h256": 0,
                    "block_h512": 0,
                    "tvt_true": truth,
                }
            )
        )
    joined = pd.concat(parts, ignore_index=True)
    blocks = train.build_stage_a_block_readout(joined, config)
    metrics = train.build_stage_a_metrics(blocks)
    h512 = metrics.loc[
        (metrics["horizon_rows"] == 512) & (metrics["scope"] == "overall")
    ].iloc[0]
    assert h512["z_affine_quotient_rmse"] == pytest.approx(0.0, abs=1e-8)
    assert h512["geop_affine_quotient_rmse"] > 0.0
    assert h512["z_affine_sse_explained_fraction"] == pytest.approx(1.0)
    gate = train.evaluate_stage_a_gate(
        metrics,
        row_identity_coverage=1.0,
        well_identity_coverage=1.0,
        finite_prediction_coverage=1.0,
        config=config,
    )
    assert gate["passed"]
    assert gate["folds_meeting_relative_shape"] == {"h256": 5, "h512": 5}


def test_stage_a_singleton_is_excluded_from_affine_metric() -> None:
    stats = train._residual_block_stats(np.array([3.0]))
    assert not stats["affine_valid"]
    assert np.isnan(stats["affine_sse"])
    assert stats["offset_sse"] == 0.0


def test_stage_b_gate_requires_strict_exp280_and_scope_improvement(config: dict) -> None:
    local = copy.deepcopy(config)
    local["validation"]["expected_rows"] = 5
    fold_metrics = pd.DataFrame(
        [
            {
                "fold": fold,
                "top1": 0.30,
                "top3": 0.60,
                "mrr": 0.50,
                "sign": 0.60,
                "shuffled_top1": 0.10,
                "shuffled_top3": 0.20,
                "shuffled_mrr": 0.20,
                "shuffled_sign": 0.40,
            }
            for fold in range(5)
        ]
    )
    scope_metrics = pd.DataFrame(
        [
            {
                "scope": scope,
                "top1": 0.30,
                "top3": 0.60,
                "mrr": 0.50,
                "sign": 0.60,
                "shuffled_top1": 0.10,
                "shuffled_top3": 0.20,
                "shuffled_mrr": 0.20,
                "shuffled_sign": 0.40,
                "bank_range_coverage": 1.0,
                "maximum_quantization_error_ft": 0.0,
            }
            for scope in (
                "overall",
                "long_tail_1000_plus",
                "hidden_like_spatial",
                "hidden_like_typewell_purged",
            )
        ]
    )
    score = pd.DataFrame(
        {"likelihood_mean": [0.0], "shuffled_likelihood_mean": [0.0]}
    )
    readout = pd.DataFrame({"block_row_count": np.ones(5, dtype=np.int64)})
    gate = train.evaluate_stage_b_gate(
        score, readout, scope_metrics, fold_metrics, local
    )
    assert gate["passed"]
    scope_metrics.loc[scope_metrics["scope"] == "overall", "top1"] = local[
        "stage_b_shift_separability"
    ]["exp280_pooled_reference"]["top1"]
    failed = train.evaluate_stage_b_gate(
        score, readout, scope_metrics, fold_metrics, local
    )
    assert not failed["passed"]
    assert not failed["checks"]["top1_strictly_above_exp280_pooled"]


def test_inference_and_stage_c_are_fail_closed(config: dict) -> None:
    checks = inference.validate_disabled_inference(config)
    assert all(checks.values())
    with pytest.raises(RuntimeError, match="intentionally disabled"):
        inference.stop_without_inference(config)
    changed = copy.deepcopy(config)
    changed["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        inference.validate_disabled_inference(changed)


def test_compact_sources_are_notebook_safe() -> None:
    assert "__file__" not in TRAIN_PATH.read_text()
    assert "__file__" not in INFERENCE_PATH.read_text()
    assert "# ## Contents" in TRAIN_PATH.read_text()
    assert "run_stage_ab(CONFIG)" in TRAIN_PATH.read_text()

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp304_gr_denoiser_emission_separability_readout"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp304_gr_denoiser_emission_separability_readout_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp304_gr_denoiser_emission_separability_readout_compact_selfcontained_inference.py"
)


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


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    known_rows = 30
    total_rows = 130
    true_tvt = 100.0 + 0.2 * np.arange(total_rows, dtype=float)
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(total_rows, dtype=float),
            "GR": 2.0 * true_tvt,
            "TVT_input": np.r_[
                true_tvt[:known_rows],
                np.full(total_rows - known_rows, np.nan),
            ],
        }
    )
    typewell_tvt = np.linspace(0.0, 300.0, 1201)
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": 2.0 * typewell_tvt})
    row_idx = np.arange(known_rows, total_rows, dtype=np.int64)
    oof_safe = pd.DataFrame(
        {
            "well_id": "well_a",
            "row_idx": row_idx,
            "suffix_offset": np.arange(len(row_idx), dtype=np.int64),
            "fold": 0,
            "tvt_geop": true_tvt[row_idx] - 10.0,
        }
    )
    return oof_safe, horizontal, typewell


def test_scientific_contract_fixes_denoisers_and_zero_decode() -> None:
    module = load_module(TRAIN_SOURCE, "exp304_contract")
    config = load_config()
    module.validate_scientific_contract(config)
    config["audit"]["denoisers"]["swt_db4_l3"]["level"] = 2
    with pytest.raises(ValueError, match="db4 level-3"):
        module.validate_scientific_contract(config)


def test_robust_rts_is_deterministic_finite_and_reduces_a_spike() -> None:
    module = load_module(TRAIN_SOURCE, "exp304_rts")
    coordinate = np.arange(128, dtype=float)
    clean = 50.0 + 0.2 * coordinate + np.sin(coordinate / 5.0)
    observed = clean.copy()
    observed[60] += 20.0
    spec = load_config()["audit"]["denoisers"]["robust_rts"]
    first, variance, status = module.robust_rts_smooth(observed, coordinate, spec)
    second, second_variance, second_status = module.robust_rts_smooth(
        observed, coordinate, spec
    )
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(variance, second_variance, rtol=0.0, atol=0.0)
    assert status == second_status
    assert status["converged"]
    assert np.isfinite(first).all() and np.isfinite(variance).all()
    assert abs(first[60] - clean[60]) < abs(observed[60] - clean[60])


def test_l1_trend_admm_converges_and_reduces_second_difference_energy() -> None:
    module = load_module(TRAIN_SOURCE, "exp304_l1")
    coordinate = np.arange(128, dtype=float)
    observed = 20.0 + 0.1 * coordinate + 0.8 * np.sin(coordinate / 3.0)
    observed[40] += 5.0
    spec = load_config()["audit"]["denoisers"]["l1_trend"]
    smoothed, status = module.l1_trend_smooth(observed, spec)
    assert status["converged"]
    assert np.isfinite(smoothed).all()
    assert np.sum(np.abs(module.second_difference(smoothed))) < np.sum(
        np.abs(module.second_difference(observed))
    )


def test_target_free_variants_keep_raw_plus_ten_shift_rank_and_no_truth() -> None:
    module = load_module(TRAIN_SOURCE, "exp304_score")
    oof_safe, horizontal, typewell = synthetic_inputs()
    scores, series, statuses, distortion, manifests = module.score_well_target_free(
        oof_safe, horizontal, typewell, load_config()
    )
    assert not {"tvt_true", "error", "abs_error", "formation"}.intersection(
        scores.columns
    )
    raw_top = scores.loc[
        (scores["variant"] == "raw") & (scores["likelihood_rank"] == 1)
    ]
    assert len(raw_top) == 1
    assert raw_top["shift_ft"].iloc[0] == 10.0
    assert {"raw_gr", "robust_rts_gr", "robust_rts_posterior_variance"}.issubset(
        series.columns
    )
    assert set(statuses["variant"]) == {
        "raw",
        "robust_rts",
        "swt_db4_l3",
        "l1_trend",
    }
    assert set(distortion["variant"]) == {"robust_rts", "swt_db4_l3", "l1_trend"}
    assert set(manifests["variant"]).issubset(set(scores["variant"]))


def test_truth_loader_requires_all_three_frozen_hashes(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp304_truth_freeze")
    path = tmp_path / "oof.csv.gz"
    pd.DataFrame(
        {"well_id": ["a"], "row_idx": [1], "tvt_true": [2.0]}
    ).to_csv(path, index=False, compression="gzip")
    config = load_config()
    with pytest.raises(ValueError, match="complete frozen"):
        module.load_exp226_truth(
            path,
            config,
            frozen_evidence={"target_free_score_content_sha256": "only-one"},
        )


def test_quality_gate_selects_only_preregistered_passing_denoiser() -> None:
    module = load_module(TRAIN_SOURCE, "exp304_gate")
    config = load_config()
    required_scopes = [
        "overall",
        "md_since_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
        "typewell_gr_abs_gradient_top10pct",
    ]
    scope_rows = []
    for variant, mrr, top3, top1, gap in (
        ("raw", 0.30, 0.40, 0.20, -1.0),
        ("robust_rts", 0.32, 0.42, 0.20, -0.5),
    ):
        for scope in required_scopes:
            scope_rows.append(
                {
                    "variant": variant,
                    "scope": scope,
                    "mrr": mrr,
                    "top3_rate": top3,
                    "top1_rate": top1,
                    "truth_minus_best_decoy_gap_mean": gap,
                }
            )
    fold_rows = []
    for fold in range(5):
        fold_rows.extend(
            [
                {
                    "variant": "raw",
                    "scope": f"fold_{fold}",
                    "fold": fold,
                    "mrr": 0.30,
                    "top3_rate": 0.40,
                    "shuffled_mrr": 0.20,
                    "shuffled_top3_rate": 0.25,
                },
                {
                    "variant": "robust_rts",
                    "scope": f"fold_{fold}",
                    "fold": fold,
                    "mrr": 0.32,
                    "top3_rate": 0.42,
                    "shuffled_mrr": 0.20,
                    "shuffled_top3_rate": 0.25,
                },
            ]
        )
    result = module.evaluate_quality_gate(
        pd.DataFrame(scope_rows),
        pd.DataFrame(fold_rows),
        {"valid_denoisers": ["robust_rts"]},
        config,
    )
    assert result["passed"]
    assert result["selected_denoiser"] == "robust_rts"


def test_inference_contract_remains_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp304_inference")
    config = load_config()
    contract = module.validate_disabled_inference(config)
    assert not contract["inference_enabled"]
    assert not contract["create_submission"]
    config["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        module.validate_disabled_inference(config)

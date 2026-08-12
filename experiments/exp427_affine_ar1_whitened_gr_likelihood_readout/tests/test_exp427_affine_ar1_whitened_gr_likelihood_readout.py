from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp427_affine_ar1_whitened_gr_likelihood_readout"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp427_affine_ar1_whitened_gr_likelihood_readout_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp427_affine_ar1_whitened_gr_likelihood_readout_compact_selfcontained_inference.py"
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
    return load_module(TRAIN_SOURCE, "exp427_train_test")


@pytest.fixture(scope="module")
def config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(value, dict)
    return value


def synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    known_rows = 96
    suffix_rows = 640
    total_rows = known_rows + suffix_rows
    typewell_tvt = np.linspace(900.0, 1200.0, 6001)

    def gr_curve(tvt: np.ndarray) -> np.ndarray:
        return 70.0 + 18.0 * np.sin(tvt / 5.0) + 7.0 * np.sin(tvt / 1.7)

    prefix_tvt = 980.0 + 0.2 * np.arange(known_rows)
    suffix_geop = 1000.0 + 0.15 * np.arange(suffix_rows)
    tvt_input = np.full(total_rows, np.nan)
    tvt_input[:known_rows] = prefix_tvt
    observed_tvt = np.concatenate([prefix_tvt, suffix_geop + 5.0])
    raw_gr = 4.0 + 1.08 * gr_curve(observed_tvt)
    raw_gr[known_rows + 40 : known_rows + 44] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(total_rows, dtype=float) * 0.5,
            "GR": raw_gr,
            "TVT_input": tvt_input,
        }
    )
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": gr_curve(typewell_tvt)})
    oof = pd.DataFrame(
        {
            "well_id": "well0001",
            "row_idx": np.arange(known_rows, total_rows, dtype=np.int64),
            "suffix_offset": np.arange(suffix_rows, dtype=np.int64),
            "fold": 0,
            "tvt_geop": suffix_geop,
        }
    )
    return oof, horizontal, typewell


def test_config_records_completed_stage0_and_disables_rerun(
    train: ModuleType, config: dict
) -> None:
    contract = train.validate_scientific_contract(
        config, require_run_approval=False
    )
    assert config["experiment"]["status"] == "stage_0_completed_gate_failed_closed"
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["stage_0_implemented"] is True
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert config["implementation"]["canonical_inference_notebook_adopted"] is False
    assert config["execution"]["implementation_approved"] is True
    assert config["execution"]["kaggle_package_approved"] is False
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["execution"]["run_stage_0"] is False
    assert config["execution"]["stage_0_completed"] is True
    assert (
        config["execution"]["stage_0_decision"]
        == "stage_0_failed_close_without_rescue"
    )
    assert config["runtime"]["kaggle"]["train_run_on_push"] is False
    assert config["execution_contract"]["stage_0"] == {
        "scientific_primary_scores": 1,
        "diagnostic_ablation_scores": 2,
        "matched_control_scores": 1,
        "saved_control_scores": 1,
        "reporting_folds": 5,
        "hmm_well_runs": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "gpu_runs": 0,
    }
    assert len(contract["scientific_contract_sha256"]) == 64
    with pytest.raises(PermissionError, match="not approved"):
        train.validate_scientific_contract(config, require_run_approval=True)


def test_prefix_affine_posterior_uses_finite_known_prefix_only(
    train: ModuleType, config: dict
) -> None:
    _, horizontal, typewell = synthetic_well()
    horizontal.loc[3, "GR"] = np.nan
    posterior, residuals = train.fit_prefix_affine_posterior(
        horizontal,
        typewell,
        well_id="well0001",
        fold=0,
        config=config,
    )
    assert posterior["known_prefix_rows"] == 96
    assert posterior["finite_prefix_pairs"] == 95
    assert posterior["affine_eligible"] is True
    assert posterior["posterior_intercept"] == pytest.approx(4.0, abs=0.5)
    assert posterior["posterior_slope"] == pytest.approx(1.08, abs=0.01)
    assert posterior["sigma"] == pytest.approx(10.0)
    assert 3 not in residuals["row_idx"].tolist()
    assert np.isfinite(residuals["residual"]).all()


def test_ar1_yule_walker_respects_missing_run_boundaries(
    train: ModuleType,
) -> None:
    rng = np.random.default_rng(7)
    residual = np.empty(180)
    residual[0] = 1.0
    for index in range(1, len(residual)):
        residual[index] = 0.55 * residual[index - 1] + rng.normal(0.0, 0.2)
    row_idx = np.r_[np.arange(90), np.arange(91, 181)]
    frame = pd.DataFrame({"row_idx": row_idx, "residual": residual})
    estimate = train.estimate_well_ar1(
        frame, minimum_pairs=64, rho_clip=[-0.8, 0.8]
    )
    assert estimate["lag1_pair_count"] == 178
    assert estimate["contiguous_run_count"] == 2
    assert estimate["ar1_evaluable"] is True
    assert estimate["rho_clipped"] == pytest.approx(0.55, abs=0.15)


def test_fold_ar1_prior_excludes_every_outer_valid_well(
    train: ModuleType, config: dict
) -> None:
    rows = []
    residuals = {}
    rng = np.random.default_rng(11)
    for fold in range(5):
        for offset in range(2):
            well = f"w{fold}{offset}"
            values = np.empty(120)
            values[0] = 0.5
            for index in range(1, len(values)):
                values[index] = 0.4 * values[index - 1] + rng.normal(0.0, 0.3)
            residuals[well] = pd.DataFrame(
                {
                    "well_id": well,
                    "fold": fold,
                    "row_idx": np.arange(len(values)),
                    "residual": values,
                }
            )
            rows.append(
                {
                    "well_id": well,
                    "fold": fold,
                    "affine_eligible": True,
                }
            )
    per_well, priors = train.build_fold_ar1_priors(
        pd.DataFrame(rows), residuals, config
    )
    assert len(per_well) == 10
    assert priors["source_wells"].eq(8).all()
    assert priors["outer_valid_source_overlap"].eq(0).all()
    assert (priors["rho_fold"].abs() < 0.8).all()


def test_rank2_woodbury_matches_dense_reference(train: ModuleType) -> None:
    assert train.dense_woodbury_parity() <= 1.0e-8
    rng = np.random.default_rng(21)
    x = rng.normal(75.0, 10.0, size=23)
    y = 3.0 + 1.04 * x + rng.normal(0.0, 4.0, size=23)
    mean = np.asarray([1.0, 1.0])
    covariance = np.asarray([[4.0, -0.01], [-0.01, 0.003]])
    woodbury = train.gaussian_predictive_logpdf_woodbury(
        y, x, mean, covariance, sigma=12.0, rho=0.5
    )
    dense = train.gaussian_predictive_logpdf_dense(
        y, x, mean, covariance, sigma=12.0, rho=0.5
    )
    assert woodbury == pytest.approx(dense, abs=1.0e-10)


def test_factorial_block_scoring_is_raw_finite_and_fixed(
    train: ModuleType, config: dict
) -> None:
    oof, horizontal, typewell = synthetic_well()
    posterior, _ = train.fit_prefix_affine_posterior(
        horizontal,
        typewell,
        well_id="well0001",
        fold=0,
        config=config,
    )
    scores, negative, eligibility, manifest = train.score_well_target_free(
        oof,
        horizontal,
        typewell,
        posterior,
        rho_fold=0.35,
        config=config,
    )
    assert manifest["blocks"] == 2
    assert manifest["eligible_blocks"] == 2
    assert len(eligibility) == 2
    assert eligibility["eligible_block"].all()
    assert eligibility.loc[eligibility["block_id"].eq(0), "finite_gr_rows"].iloc[0] == 508
    assert len(scores) == 2 * 4 * 13
    assert len(negative) == 2 * 13
    assert scores.groupby(["block_id", "variant"]).size().eq(13).all()
    assert np.isfinite(scores["score"]).all()
    assert scores.groupby(["block_id", "variant"])["rank"].apply(
        lambda values: sorted(values.tolist()) == list(range(1, 14))
    ).all()
    saved = scores.loc[
        scores["variant"].eq("identity_iid_matched"),
        [
            "well_id",
            "fold",
            "block_id",
            "block_start_suffix_offset",
            "block_end_suffix_offset",
            "shift_slot",
            "shift_ft",
            "score",
            "rank",
        ],
    ].rename(columns={"score": "likelihood_mean", "rank": "likelihood_rank"})
    aligned, technical = train.align_saved_control(
        scores, negative, eligibility, saved
    )
    assert len(aligned) == 2 * 13
    assert technical["row_identity_coverage"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="negative control"):
        train.align_saved_control(scores, negative.iloc[:-1], eligibility, saved)


def test_ineligible_well_returns_typed_empty_score_tables(
    train: ModuleType, config: dict
) -> None:
    oof, horizontal, typewell = synthetic_well()
    posterior, _ = train.fit_prefix_affine_posterior(
        horizontal,
        typewell,
        well_id="well0001",
        fold=0,
        config=config,
    )
    suffix_rows = oof["row_idx"].to_numpy(np.int64)
    horizontal.loc[suffix_rows, "GR"] = np.nan
    scores, negative, eligibility, manifest = train.score_well_target_free(
        oof,
        horizontal,
        typewell,
        posterior,
        rho_fold=0.35,
        config=config,
    )
    assert scores.empty
    assert negative.empty
    assert {"well_id", "block_id", "variant", "shift_slot", "score", "rank"} <= set(
        scores.columns
    )
    assert {
        "well_id",
        "block_id",
        "variant",
        "shift_slot",
        "source_shift_slot",
        "score",
        "rank",
    } <= set(negative.columns)
    assert len(eligibility) == 2
    assert not eligibility["eligible_block"].any()
    assert manifest["eligible_blocks"] == 0


def test_target_free_scoring_rejects_truth_columns(
    train: ModuleType, config: dict
) -> None:
    oof, horizontal, typewell = synthetic_well()
    posterior, _ = train.fit_prefix_affine_posterior(
        horizontal,
        typewell,
        well_id="well0001",
        fold=0,
        config=config,
    )
    oof["tvt_true"] = oof["tvt_geop"] + 5.0
    with pytest.raises(ValueError, match="forbidden"):
        train.score_well_target_free(
            oof,
            horizontal,
            typewell,
            posterior,
            rho_fold=0.35,
            config=config,
        )
    horizontal["TVT"] = 0.0
    with pytest.raises(ValueError, match="horizontal TVT"):
        train.score_well_target_free(
            oof.drop(columns="tvt_true"),
            horizontal,
            typewell,
            posterior,
            rho_fold=0.35,
            config=config,
        )


def test_negative_control_is_stable_and_keyed_by_fold_block(
    train: ModuleType,
) -> None:
    first = train.stable_score_permutation("well", 2, 7, 13, seed=42)
    second = train.stable_score_permutation("well", 2, 7, 13, seed=42)
    changed = train.stable_score_permutation("well", 3, 7, 13, seed=42)
    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == list(range(13))
    assert not np.array_equal(first, np.arange(13))
    assert not np.array_equal(first, changed)


def test_manifest_content_hash_ignores_runtime_paths(train: ModuleType) -> None:
    left = {
        "input": {"path": "/kaggle/input/a.csv", "content_sha256": "a" * 64},
        "rows": 10,
    }
    right = {
        "input": {"path": "/tmp/local/a.csv", "content_sha256": "a" * 64},
        "rows": 10,
    }
    assert train.mapping_sha256(train.logical_manifest_payload(left)) == (
        train.mapping_sha256(train.logical_manifest_payload(right))
    )


def test_truth_access_ledger_fails_closed_before_freeze(train: ModuleType) -> None:
    ledger = train.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="requires a frozen"):
        ledger.require_frozen()
    with pytest.raises(RuntimeError, match="truth access"):
        ledger.register_truth(10)
    assert ledger.truth_rows_before_freeze == 10
    ledger = train.TruthAccessLedger()
    ledger.mark_frozen("a" * 64)
    ledger.register_truth(10)
    ledger.register_hidden_roles(3)
    assert ledger.truth_rows_after_freeze == 10
    assert ledger.hidden_rows_after_freeze == 3


def _metric_tables(train: ModuleType) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = {
        "identity_iid_matched": (0.40, 0.50, 2.0),
        "affine_iid": (0.44, 0.52, 1.8),
        "identity_ar1": (0.44, 0.52, 1.8),
        "affine_ar1": (0.47, 0.56, 1.0),
        "saved_exp280": (0.389626, 0.452421, 2.0),
        "affine_ar1_shuffled": (0.20, 0.25, 3.0),
    }
    factorial = pd.DataFrame(
        [
            {
                "scope": "overall",
                "family": family,
                "mrr": metric[0],
                "top3_rate": metric[1],
                "top1_regret_rmse_p90": metric[2],
            }
            for family, metric in values.items()
        ]
    )
    fold = pd.DataFrame(
        [
            {
                "scope": f"fold_{fold_id}",
                "fold": fold_id,
                "family": family,
                "mrr": metric[0],
                "top3_rate": metric[1],
            }
            for fold_id in range(5)
            for family, metric in values.items()
        ]
    )
    scopes = pd.DataFrame(
        [
            {
                "scope": scope,
                "family": family,
                "mrr": metric[0],
                "top3_rate": metric[1],
            }
            for scope in (
                "overall",
                "long_tail_1000_plus",
                "hidden_like_spatial",
                "hidden_like_typewell_purged",
            )
            for family, metric in values.items()
        ]
    )
    return factorial, fold, scopes


def test_and_gate_requires_factorial_controls_stress_and_shuffle(
    train: ModuleType, config: dict
) -> None:
    factorial, fold, scopes = _metric_tables(train)
    technical = {
        "score_finite_coverage": 1.0,
        "row_identity_coverage": 1.0,
        "minimum_candidate_count_per_eligible_block_variant": 13,
        "maximum_candidate_count_per_eligible_block_variant": 13,
        "eligible_well_fraction": 0.95,
        "eligible_block_fraction": 0.80,
        "affine_eligible_well_fraction": 0.95,
        "maximum_outer_valid_rho_source_overlap": 0,
        "maximum_abs_fold_rho": 0.7,
        "dense_woodbury_max_abs_error": 1.0e-12,
        "truth_rows_before_freeze": 0,
        "hidden_rows_before_freeze": 0,
        "runtime_seconds": 100.0,
        "peak_rss_gb": 2.0,
    }
    gate = train.evaluate_gates(technical, factorial, fold, scopes, config)
    assert gate["passed"] is True
    degraded = scopes.copy()
    mask = degraded["scope"].eq("hidden_like_spatial") & degraded["family"].eq(
        "affine_ar1"
    )
    degraded.loc[mask, "mrr"] = 0.30
    failed = train.evaluate_gates(technical, factorial, fold, degraded, config)
    assert failed["passed"] is False
    assert (
        failed["scientific_checks"][
            "hidden_like_spatial_mrr_vs_both_controls"
        ]
        is False
    )


def test_inference_candidate_refuses_prediction(config: dict) -> None:
    inference = load_module(INFERENCE_SOURCE, "exp427_inference_test")
    contract = inference.validate_disabled_inference(config)
    assert contract["status"] == "inference_disabled"
    with pytest.raises(RuntimeError, match="requires a separate experiment"):
        inference.refuse_inference(config)


def test_compact_train_is_adopted_and_inference_stays_fail_closed() -> None:
    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert train_text.count("# %% [markdown]") >= 12
    assert "def fit_prefix_affine_posterior" in train_text
    assert "def gaussian_predictive_logpdf_woodbury" in train_text
    assert "def score_well_target_free" in train_text
    assert "def evaluate_gates" in train_text
    assert "__file__" not in train_text
    assert "from settings import" not in train_text
    assert "__file__" not in inference_text
    compact_train = (
        EXP_DIR
        / "exp427_affine_ar1_whitened_gr_likelihood_readout_compact_selfcontained_train.ipynb"
    )
    compact_inference = (
        EXP_DIR
        / "exp427_affine_ar1_whitened_gr_likelihood_readout_compact_selfcontained_inference.ipynb"
    )
    assert compact_train.exists()
    assert compact_inference.exists()
    canonical_train = json.loads(
        (
            EXP_DIR
            / "exp427_affine_ar1_whitened_gr_likelihood_readout_train.ipynb"
        ).read_text()
    )
    canonical_inference = json.loads(
        (
            EXP_DIR
            / "exp427_affine_ar1_whitened_gr_likelihood_readout_inference.ipynb"
        ).read_text()
    )
    compact_train_json = json.loads(compact_train.read_text())
    assert canonical_train["cells"] == compact_train_json["cells"]
    assert "affine + AR(1) whitened GR likelihood readout" in "".join(
        canonical_train["cells"][0]["source"]
    )
    assert "Submission creation" in "".join(
        cell_source
        for cell in canonical_inference["cells"]
        for cell_source in cell["source"]
    )


def test_contract_rejects_route_or_execution_expansion(
    train: ModuleType, config: dict
) -> None:
    changed = copy.deepcopy(config)
    changed["experiment"]["route"] = "ensemble"
    with pytest.raises(ValueError, match="experiment.route"):
        train.validate_scientific_contract(changed)
    changed = copy.deepcopy(config)
    changed["execution"]["run_pf"] = True
    with pytest.raises(ValueError, match="execution.run_pf"):
        train.validate_scientific_contract(changed)

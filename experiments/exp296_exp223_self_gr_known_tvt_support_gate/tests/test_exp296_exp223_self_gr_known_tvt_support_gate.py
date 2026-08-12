from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp296_exp223_self_gr_known_tvt_support_gate"
TRAIN_SOURCE = EXP_DIR / (
    "exp296_exp223_self_gr_known_tvt_support_gate_compact_selfcontained_train.py"
)
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp223_joint_typewell_self_gr_hmm_likelihood_probe"
    / "exact_hmm_smoother.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP296 = load_module(TRAIN_SOURCE, "exp296_support_gate_contract")
EXP223 = load_module(PARENT_SOURCE, "exp223_parent_for_exp296_contract")


def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def self_gr_config() -> dict:
    return dict(load_config()["model"]["self_gr_emission"])


def synthetic_well(
    *,
    known_min: float = 40.0,
    known_max: float = 140.0,
    known_rows: int = 60,
    eval_rows: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = known_rows + eval_rows
    index = np.arange(rows, dtype=np.float64)
    tvt_input = np.full(rows, np.nan, dtype=np.float64)
    tvt_input[:known_rows] = np.linspace(known_min, known_max, known_rows)
    horizontal = pd.DataFrame(
        {
            "MD": 10_000.0 + index,
            "Z": -8_000.0 - 0.1 * index,
            "GR": 70.0 + 12.0 * np.sin(index / 4.0) + 4.0 * np.cos(index / 9.0),
            "TVT_input": tvt_input,
        }
    )
    typewell_tvt = np.linspace(0.0, 100.0, 401)
    typewell = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": 70.0 + 12.0 * np.sin(typewell_tvt / 4.0),
        }
    )
    return horizontal, typewell


def fake_forward_backward(emission, *_args):
    values = np.asarray(emission, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    posterior = np.exp(shifted)
    posterior /= posterior.sum(axis=1, keepdims=True)
    return posterior, float(np.sum(values))


def test_config_fixes_one_approved_kaggle_cpu_variant() -> None:
    config = load_config()
    EXP296.validate_scientific_contract(config)
    assert config["experiment"]["status"].startswith("kaggle_cpu_")
    assert config["model"]["planned_variants"] == 1
    assert config["model"]["planned_hmm_well_runs"] == 773
    assert config["model"]["lightgbm_configs"] == 0
    assert config["model"]["trained_folds"] == 0
    assert config["model"]["boosters"] == 0
    assert config["execution"]["implementation"] is True
    assert config["execution"]["run_control"] is False
    assert config["execution"]["run_variant"] is True
    assert config["execution"]["kaggle_cpu_push_approved"] is True
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["run_inference"] is False
    assert config["execution"]["write_submission"] is False

    unsafe = copy.deepcopy(config)
    unsafe["model"]["support_gate"]["padding_tvt"] = 0.35
    with pytest.raises(ValueError, match="support gate contract changed"):
        EXP296.validate_scientific_contract(unsafe)


def test_candidate_state_support_mask_is_inclusive_and_uses_all_finite_known_tvt() -> None:
    grid = np.asarray([9.0, 10.0, 10.5, 12.0, 13.0])
    tvt_input = np.asarray([np.nan, 12.0, 10.0, np.nan, 11.0])
    mask, lower, upper = EXP296.build_candidate_state_support_mask(grid, tvt_input)
    assert lower == 10.0
    assert upper == 12.0
    assert mask.tolist() == [False, True, True, True, False]


def test_support_gate_preserves_inside_bits_and_sets_outside_to_exact_zero() -> None:
    boost = np.asarray(
        [[-0.0, 0.25, 0.75, 1.0], [0.125, 0.5, 0.875, 0.0]],
        dtype=np.float32,
    )
    grid = np.asarray([0.0, 1.0, 2.0, 3.0])
    gated, support, lower, upper = EXP296.apply_known_tvt_support_gate(
        boost,
        grid,
        np.asarray([1.0, 2.0, np.nan]),
    )
    assert lower == 1.0 and upper == 2.0
    assert support.tolist() == [False, True, True, False]
    assert np.array_equal(gated[:, support], boost[:, support])
    assert np.max(np.abs(gated[:, ~support])) == 0.0
    assert gated.dtype == boost.dtype


def test_no_finite_known_tvt_makes_all_false_neutral_mask() -> None:
    grid = np.arange(5, dtype=np.float64)
    boost = np.ones((3, 5), dtype=np.float32)
    gated, support, lower, upper = EXP296.apply_known_tvt_support_gate(
        boost,
        grid,
        np.full(7, np.nan),
    )
    assert not support.any()
    assert np.max(np.abs(gated)) == 0.0
    assert np.isnan(lower) and np.isnan(upper)


def test_exp223_self_gr_surface_is_exactly_preserved_before_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    horizontal, typewell = synthetic_well()
    grid = np.arange(40.0, 141.0, 1.0)
    eval_index = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    peak = np.full(len(eval_index), 80.0)
    # exp223 used fillna(method=...), removed by the current pandas version.
    # exp296's bfill/ffill implementation is semantically identical; patch
    # only that compatibility layer so the unchanged parent surface can run.
    monkeypatch.setattr(EXP223, "build_gr_window_descriptors", EXP296.build_gr_window_descriptors)
    parent = EXP223.build_self_gr_likelihood_surface(
        horizontal,
        eval_index,
        grid,
        peak,
        self_gr_config(),
    )
    candidate = EXP296.build_self_gr_likelihood_surface(
        horizontal,
        eval_index,
        grid,
        peak,
        self_gr_config(),
    )
    for key in (
        "centered_logl",
        "quality",
        "peak_tvt",
        "peak_gap",
        "typewell_agreement",
        "valid",
    ):
        np.testing.assert_array_equal(candidate[key], parent[key])
    assert candidate["prefix_anchor_count"] == parent["prefix_anchor_count"]


def test_all_true_gate_has_exp223_prediction_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    horizontal, typewell = synthetic_well(known_min=40.0, known_max=140.0)
    monkeypatch.setattr(EXP223, "_hmm2_fb", fake_forward_backward)
    monkeypatch.setattr(EXP296, "_hmm2_fb", fake_forward_backward)
    monkeypatch.setattr(EXP223, "build_gr_window_descriptors", EXP296.build_gr_window_descriptors)
    kwargs = {
        "step": 1.0,
        "n_rates": 5,
        "rate_span": 0.10,
        "sig_r": 0.002,
        "sig_p": 0.02,
        "df": 4.0,
        "emission": "gauss",
        "lam": 1.0,
        "sigma_mode": "std",
        "start_sig": 0.75,
        "r0_sig": 0.01,
        "band_pad": 100.0,
        "mom": 0.998,
        "rate_center": "zero",
        "self_gr_config": self_gr_config(),
        "self_gr_alpha": 0.07,
        "self_gr_clip": 1.0,
        "self_gr_mode": "boost_only",
    }
    parent = EXP223.run_hmm2(horizontal, typewell, **kwargs)
    candidate = EXP296.run_hmm2_known_tvt_support_gate(
        horizontal,
        typewell,
        **kwargs,
        return_debug_matrices=True,
    )
    assert candidate["support_mask"].all()
    np.testing.assert_array_equal(candidate["mean_eval"], parent["mean_eval"])
    np.testing.assert_array_equal(candidate["std_eval"], parent["std_eval"])
    assert candidate["loglik"] == parent["loglik"]


def test_final_prediction_is_not_clipped_to_known_range(monkeypatch: pytest.MonkeyPatch) -> None:
    horizontal, typewell = synthetic_well(known_min=40.0, known_max=80.0)

    def outside_posterior(emission, *_args):
        posterior = np.zeros_like(emission, dtype=np.float64)
        posterior[:, -1] = 1.0
        return posterior, 0.0

    monkeypatch.setattr(EXP296, "_hmm2_fb", outside_posterior)
    result = EXP296.run_hmm2_known_tvt_support_gate(
        horizontal,
        typewell,
        step=1.0,
        n_rates=5,
        self_gr_config=self_gr_config(),
        return_debug_matrices=True,
    )
    assert result["outside_support_contribution_max_abs"] == 0.0
    assert result["inside_support_boost_delta_max_abs"] == 0.0
    assert np.all(result["mean_eval"] > result["known_tvt_max"])
    assert np.all(result["posterior_outside_support_mass"] == 1.0)


def test_no_known_tvt_fallback_keeps_base_hmm_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    horizontal, typewell = synthetic_well()
    horizontal["TVT_input"] = np.nan
    monkeypatch.setattr(EXP296, "_hmm2_fb", fake_forward_backward)
    result = EXP296.run_hmm2_known_tvt_support_gate(
        horizontal,
        typewell,
        step=1.0,
        n_rates=5,
        self_gr_config=self_gr_config(),
        return_debug_matrices=True,
    )
    assert result["no_known_tvt_fallback"] is True
    assert result["support_state_count"] == 0
    assert np.max(np.abs(result["gated_boost"])) == 0.0
    assert np.isfinite(result["mean_eval"]).all()
    np.testing.assert_allclose(result["posterior_outside_support_mass"], 1.0)


def test_generation_loader_never_reads_horizontal_truth_or_candidate_columns(
    tmp_path: Path,
) -> None:
    well = "well-a"
    horizontal, typewell = synthetic_well()
    horizontal.assign(
        TVT=np.arange(len(horizontal), dtype=float) + 1000.0,
        prediction=np.arange(len(horizontal), dtype=float),
        error=-1.0,
    ).to_csv(tmp_path / f"{well}__horizontal_well.csv", index=False)
    typewell.to_csv(tmp_path / f"{well}__typewell.csv", index=False)
    loaded_horizontal, loaded_typewell = EXP296.load_generation_well(well, tmp_path)
    assert loaded_horizontal.columns.tolist() == list(EXP296.GENERATION_HORIZONTAL_COLUMNS)
    assert "TVT" not in loaded_horizontal.columns
    assert "prediction" not in loaded_horizontal.columns
    assert "error" not in loaded_horizontal.columns
    assert loaded_typewell.columns.tolist() == list(EXP296.GENERATION_TYPEWELL_COLUMNS)


def test_train_dir_resolver_requires_expected_inventory_and_skips_public_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kaggle_input = tmp_path / "kaggle-input"
    public_test = kaggle_input / "rogii-wellbore-geology-prediction" / "test"
    competition_train = (
        kaggle_input / "competitions" / "rogii-wellbore-geology-prediction" / "train"
    )
    public_test.mkdir(parents=True)
    competition_train.mkdir(parents=True)
    (public_test / "demo__horizontal_well.csv").touch()
    (competition_train / "well-a__horizontal_well.csv").touch()
    (competition_train / "well-b__horizontal_well.csv").touch()
    config = copy.deepcopy(load_config())
    config["data"]["train_dir"] = "missing/raw/train"
    config["comparison"]["saved_control_wells"] = 2
    monkeypatch.setattr(EXP296, "KAGGLE_INPUT_ROOT", kaggle_input)
    monkeypatch.setattr(EXP296, "project_root", lambda: tmp_path)
    assert EXP296.resolve_train_dir(config) == competition_train.resolve()


def test_hmm_runtime_kwargs_exclude_manifest_source_and_fail_on_unknown_key() -> None:
    config = load_config()
    kwargs = EXP296.hmm_runtime_kwargs(config)
    assert "source" not in kwargs
    assert set(kwargs) == EXP296.HMM_RUNTIME_KEYS
    unsafe = copy.deepcopy(config)
    unsafe["model"]["hmm"]["unknown_runtime_key"] = 1
    with pytest.raises(ValueError, match="unexpected HMM runtime config keys"):
        EXP296.hmm_runtime_kwargs(unsafe)


def test_full_runner_requires_separate_kaggle_cpu_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    monkeypatch.setenv("EXPERIMENT_ALLOW_LOCAL", "1")
    disabled_variant = copy.deepcopy(config)
    disabled_variant["execution"]["run_variant"] = False
    with pytest.raises(RuntimeError, match="execution.run_variant is false"):
        EXP296.run_full_experiment(disabled_variant)
    approved_variant = copy.deepcopy(config)
    approved_variant["execution"]["kaggle_cpu_push_approved"] = False
    with pytest.raises(RuntimeError, match="Kaggle CPU push is not approved"):
        EXP296.run_full_experiment(approved_variant)


def test_truth_late_metric_and_hard_gate_directions_are_conjunctive() -> None:
    config = load_config()
    config["validation"]["hard_gates"]["technical"]["input_wells"] = 5
    rows: list[dict] = []
    for fold in range(5):
        for row_index in range(4):
            rows.append(
                {
                    "id": f"well-{fold}_{row_index}",
                    "well": f"well-{fold}",
                    "row_index": row_index,
                    "reporting_fold": fold,
                    "last_known_tvt": 0.0,
                    "control_last_known_tvt": 0.0,
                    "md_since": 1000.0 + row_index,
                    "true_tvt": 0.0,
                    "known_tvt_min": -1.0,
                    "known_tvt_max": 1.0,
                    "true_tvt_inside_known_range": row_index % 2 == 0,
                    "distance_bucket": "1000_plus",
                    EXP296.PREDICTION_COLUMN: 0.0,
                    "exp223_control_tvt": 2.0,
                    "posterior_outside_support_mass": 0.5,
                }
            )
    scored = pd.DataFrame(rows)
    overall = EXP296.overall_metrics(scored)
    folds = EXP296.grouped_metrics(scored, "reporting_fold", "reporting_fold")
    distance = EXP296.grouped_metrics(scored, "distance_bucket", "distance_bucket")
    scopes = EXP296.known_range_scope_metrics(scored)
    hidden = pd.DataFrame(
        [
            *EXP296.metric_rows(
                scored,
                scope="hidden_like",
                scope_value="verification_like_spatial",
            ),
            *EXP296.metric_rows(
                scored,
                scope="hidden_like",
                scope_value="verification_like_typewell_purged",
            ),
        ]
    )
    by_well = EXP296.by_well_metrics(scored)
    steps = EXP296.step_delta_metrics(scored)
    well_manifest = pd.DataFrame(
        {
            "outside_support_contribution_max_abs": [0.0] * 5,
            "inside_support_boost_delta_max_abs": [0.0] * 5,
        }
    )
    gates = EXP296.evaluate_hard_gates(
        generated=scored,
        well_manifest=well_manifest,
        overall=overall,
        folds=folds,
        distance=distance,
        scopes=scopes,
        hidden=hidden,
        by_well=by_well,
        steps=steps,
        join_audit={"row_identity_exact": True},
        control_manifest={"sha_exact": True},
        config=config,
    )
    assert gates["technical_passed"] is True
    assert gates["performance_passed"] is True
    assert gates["passed"] is True

    failed_overall = overall.copy()
    failed_overall.loc[failed_overall["candidate"] == EXP296.VARIANT, "delta_rmse_vs_exp223"] = 0.1
    failed = EXP296.evaluate_hard_gates(
        generated=scored,
        well_manifest=well_manifest,
        overall=failed_overall,
        folds=folds,
        distance=distance,
        scopes=scopes,
        hidden=hidden,
        by_well=by_well,
        steps=steps,
        join_audit={"row_identity_exact": True},
        control_manifest={"sha_exact": True},
        config=config,
    )
    assert failed["passed"] is False


def test_compact_source_is_self_contained_and_does_not_authorize_inference() -> None:
    source = TRAIN_SOURCE.read_text()
    assert "from settings import" not in source
    assert "from exact_hmm_smoother import" not in source
    assert "Path(__file__)" not in source
    assert "def apply_known_tvt_support_gate(" in source
    assert "def run_full_experiment(" in source
    assert "submission.csv" not in source
    assert not (
        EXP_DIR / "exp296_exp223_self_gr_known_tvt_support_gate_compact_selfcontained_inference.py"
    ).exists()

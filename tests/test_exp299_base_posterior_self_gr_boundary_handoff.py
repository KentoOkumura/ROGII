from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp299_base_posterior_self_gr_boundary_handoff"
TRAIN_SOURCE = EXP_DIR / (
    "exp299_base_posterior_self_gr_boundary_handoff_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp299_base_posterior_self_gr_boundary_handoff_compact_selfcontained_inference.py"
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


EXP299 = load_module(TRAIN_SOURCE, "exp299_handoff_contract")
INFERENCE = load_module(INFERENCE_SOURCE, "exp299_inference_contract")
EXP223 = load_module(PARENT_SOURCE, "exp223_parent_for_exp299_contract")


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
    typewell_tvt = np.linspace(0.0, 180.0, 721)
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


def test_config_locks_completed_two_pass_cpu_run_contract() -> None:
    config = load_config()
    EXP299.validate_scientific_contract(config)
    assert (
        config["experiment"]["status"]
        == "kaggle_cpu_v2_completed_train_side_guard_failed_closed"
    )
    assert config["model"]["planned_scientific_variants"] == 1
    assert config["model"]["planned_base_hmm_well_runs"] == 773
    assert config["model"]["planned_variant_hmm_well_runs"] == 773
    assert config["model"]["planned_total_hmm_well_runs"] == 1546
    assert config["model"]["lightgbm_configs"] == 0
    assert config["model"]["trained_folds"] == 0
    assert config["model"]["boosters"] == 0
    assert config["execution"]["implementation"] is True
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["run_train"] is True
    assert config["execution"]["run_variant"] is True
    assert config["execution"]["kaggle_cpu_push_approved"] is False
    assert config["execution"]["one_run_authorization_consumed"] is True
    assert config["execution"]["kaggle_kernel_version"] == 2
    assert config["execution"]["kaggle_kernel_id_no"] == 127957958
    assert config["execution"]["kaggle_kernel_status"] == "COMPLETE"
    assert config["execution"]["kaggle_execution_finished"] is True
    assert config["execution"]["kaggle_execution_succeeded"] is True
    assert config["execution"]["kaggle_v2_executed_hmm_well_runs"] == 1546
    assert config["execution"]["kaggle_v2_base_parity_passed"] is True
    assert config["execution"]["kaggle_v2_performance_passed"] is False
    assert config["execution"]["rerun_approved"] is False
    assert config["execution"]["rerun_authorization_consumed"] is True
    assert config["execution"]["run_inference"] is False
    assert config["execution"]["write_submission"] is False

    unsafe = copy.deepcopy(config)
    unsafe["model"]["handoff"]["boundary_fade_width_multiplier"] = 2.0
    with pytest.raises(ValueError, match="handoff contract changed"):
        EXP299.validate_scientific_contract(unsafe)


def test_support_is_inclusive_and_uses_all_finite_visible_prefix_tvt() -> None:
    support, lower, upper = EXP299.build_candidate_state_support_mask(
        np.asarray([9.0, 10.0, 10.5, 12.0, 13.0]),
        np.asarray([np.nan, 12.0, 10.0, np.nan, 11.0]),
    )
    assert lower == 10.0
    assert upper == 12.0
    assert support.tolist() == [False, True, True, True, False]


def test_conditional_handoff_is_exact_zero_outside_and_preserves_support_mass() -> None:
    grid = np.asarray([0.0, 10.0, 20.0, 30.0])
    posterior = np.asarray(
        [
            [0.05, 0.65, 0.20, 0.10],
            [0.10, 0.20, 0.65, 0.05],
        ]
    )
    boost = np.asarray([[0.9, 0.1, 1.0, 0.8], [0.7, 1.0, 0.2, 0.6]])
    result = EXP299.compute_boundary_handoff_contribution(
        posterior,
        boost,
        np.ones(2),
        grid,
        np.asarray([10.0, 20.0]),
        alpha=0.07,
        fade_width_tvt=12.0,
    )
    contribution = result["contribution"]
    support = result["support_mask"]
    assert np.max(np.abs(contribution[:, ~support])) == 0.0
    assert np.max(result["conditional_support_mass_relative_error"]) <= 1e-6
    assert np.any(contribution[:, support] > 0.0)
    assert np.any(contribution[:, support] < 0.0)
    before = posterior[:, support].sum(axis=1)
    after = (posterior[:, support] * np.exp(contribution[:, support])).sum(axis=1)
    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-9)


def test_base_mean_outside_or_on_boundary_neutralizes_every_state() -> None:
    grid = np.asarray([0.0, 10.0, 20.0, 30.0])
    posterior = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    result = EXP299.compute_boundary_handoff_contribution(
        posterior,
        np.ones_like(posterior),
        np.ones(3),
        grid,
        np.asarray([10.0, 20.0]),
        alpha=0.07,
        fade_width_tvt=12.0,
    )
    assert result["row_gate"].tolist() == [0.0, 0.0, 0.0]
    assert np.max(np.abs(result["contribution"])) == 0.0
    assert result["base_mean_outside_or_boundary_contribution_max_abs"] == 0.0


def test_no_finite_known_tvt_neutralizes_all_self_gr() -> None:
    posterior = np.asarray([[0.2, 0.3, 0.5]])
    result = EXP299.compute_boundary_handoff_contribution(
        posterior,
        np.ones_like(posterior),
        np.ones(1),
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([np.nan, np.nan]),
        alpha=0.07,
        fade_width_tvt=12.0,
    )
    assert not result["support_mask"].any()
    assert np.max(np.abs(result["contribution"])) == 0.0
    assert np.isnan(result["known_tvt_min"])
    assert np.isnan(result["known_tvt_max"])


def test_exp223_self_gr_surface_is_preserved_before_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    horizontal, _ = synthetic_well()
    grid = np.arange(40.0, 141.0, 1.0)
    eval_index = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    peak = np.full(len(eval_index), 80.0)
    monkeypatch.setattr(EXP223, "build_gr_window_descriptors", EXP299.build_gr_window_descriptors)
    parent = EXP223.build_self_gr_likelihood_surface(
        horizontal, eval_index, grid, peak, self_gr_config()
    )
    candidate = EXP299.build_self_gr_likelihood_surface(
        horizontal, eval_index, grid, peak, self_gr_config()
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


def test_two_pass_runner_uses_only_frozen_base_posterior_for_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    horizontal, typewell = synthetic_well()
    emissions: list[np.ndarray] = []

    def recording_forward_backward(emission, *_args):
        emissions.append(np.asarray(emission, dtype=np.float64).copy())
        return fake_forward_backward(emission)

    monkeypatch.setattr(EXP299, "_hmm2_fb", recording_forward_backward)
    result = EXP299.run_hmm2_base_posterior_handoff(
        horizontal,
        typewell,
        step=1.0,
        n_rates=5,
        self_gr_config=self_gr_config(),
        return_debug_matrices=True,
    )
    assert len(emissions) == 2
    np.testing.assert_array_equal(emissions[0], result["base_emission_ll"])
    np.testing.assert_allclose(
        emissions[1],
        result["base_emission_ll"] + result["contribution"],
        rtol=0.0,
        atol=1e-7,
    )
    expected = EXP299.compute_boundary_handoff_contribution(
        result["base_posterior"],
        result["exp223_boost"],
        result["self_gr_quality"],
        result["grid"],
        horizontal["TVT_input"].to_numpy(np.float64),
        alpha=0.07,
        fade_width_tvt=12.0,
    )
    np.testing.assert_array_equal(result["contribution"], expected["contribution"])


def test_generation_loader_drops_truth_prediction_and_error_columns(tmp_path: Path) -> None:
    horizontal, typewell = synthetic_well()
    horizontal.assign(TVT=999.0, prediction=1.0, error=-1.0).to_csv(
        tmp_path / "well-a__horizontal_well.csv", index=False
    )
    typewell.to_csv(tmp_path / "well-a__typewell.csv", index=False)
    loaded_horizontal, loaded_typewell = EXP299.load_generation_well("well-a", tmp_path)
    assert loaded_horizontal.columns.tolist() == list(EXP299.GENERATION_HORIZONTAL_COLUMNS)
    assert loaded_typewell.columns.tolist() == list(EXP299.GENERATION_TYPEWELL_COLUMNS)
    assert "TVT" not in loaded_horizontal
    assert "prediction" not in loaded_horizontal
    assert "error" not in loaded_horizontal


def test_exp209_parity_uses_float32_persisted_base_prediction_contract() -> None:
    persisted_float32 = np.asarray([11236.02, 12862.77], dtype=np.float32)
    generated = pd.DataFrame(
        {
            "id": ["a_1", "b_2"],
            "well": ["a", "b"],
            EXP299.BASE_PREDICTION_COLUMN: persisted_float32,
        }
    )
    # These are the short decimal strings pandas writes for the float32 values
    # above, parsed back as float64 by read_csv.
    reference = pd.DataFrame(
        {
            "id": ["a_1", "b_2"],
            "well": ["a", "b"],
            "exp209_base_hmm_mean_tvt": [11236.02, 12862.77],
        }
    )
    raw_decimal_delta = np.max(
        np.abs(
            persisted_float32.astype(np.float64)
            - reference["exp209_base_hmm_mean_tvt"].to_numpy(np.float64)
        )
    )
    assert raw_decimal_delta > 1e-5
    parity = EXP299.compare_base_hmm_reference(generated, reference, atol_ft=1e-5)
    assert parity["passed"] is True
    assert parity["max_abs_delta_ft"] == 0.0
    assert parity["reference_csv_decimal_to_float32_max_abs_delta_ft"] > 1e-5
    changed = reference.copy()
    changed.loc[1, "exp209_base_hmm_mean_tvt"] += 0.1
    with pytest.raises(ValueError, match="exp209 base-HMM parity failed"):
        EXP299.compare_base_hmm_reference(generated, changed, atol_ft=1e-5)


def test_full_runner_requires_notebook_adoption_run_and_cpu_push_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    monkeypatch.setenv("EXPERIMENT_ALLOW_LOCAL", "1")
    not_adopted = copy.deepcopy(config)
    not_adopted["execution"]["canonical_train_notebook_adopted"] = False
    with pytest.raises(RuntimeError, match="canonical train Notebook is not adopted"):
        EXP299.run_full_experiment(not_adopted)
    not_train = copy.deepcopy(config)
    not_train["execution"]["run_train"] = False
    with pytest.raises(RuntimeError, match="execution.run_train is false"):
        EXP299.run_full_experiment(not_train)
    not_variant = copy.deepcopy(config)
    not_variant["execution"]["run_variant"] = False
    with pytest.raises(RuntimeError, match="execution.run_variant is false"):
        EXP299.run_full_experiment(not_variant)
    with pytest.raises(RuntimeError, match="Kaggle CPU push is not approved"):
        EXP299.run_full_experiment(config)


def test_performance_and_technical_gates_are_conjunctive() -> None:
    config = copy.deepcopy(load_config())
    technical = config["validation"]["hard_gates"]["technical"]
    technical["input_wells"] = 5
    technical["output_rows"] = 20
    rows: list[dict] = []
    for fold in range(5):
        for row_index in range(4):
            truth = 2.0 if row_index == 3 else 0.0
            rows.append(
                {
                    "id": f"well-{fold}_{row_index}",
                    "well": f"well-{fold}",
                    "row_index": row_index,
                    "reporting_fold": fold,
                    "last_known_tvt": 0.0,
                    "control_last_known_tvt": 0.0,
                    "md_since": 1000.0 + row_index,
                    "true_tvt": truth,
                    "known_tvt_min": -1.0,
                    "known_tvt_max": 1.0,
                    "true_tvt_inside_known_range": row_index != 3,
                    "upper_boundary_scope": (
                        "upper_outside_0_12" if row_index == 3 else "not_upper_outside"
                    ),
                    "distance_bucket": "1000_plus",
                    EXP299.PREDICTION_COLUMN: truth,
                    "exp223_control_tvt": truth + 2.0,
                    "posterior_outside_support_mass": 0.5,
                }
            )
    scored = pd.DataFrame(rows)
    overall = EXP299.overall_metrics(scored)
    folds = EXP299.grouped_metrics(scored, "reporting_fold", "reporting_fold")
    distance = EXP299.grouped_metrics(scored, "distance_bucket", "distance_bucket")
    scopes = EXP299.known_range_scope_metrics(scored)
    hidden = pd.DataFrame(
        [
            *EXP299.metric_rows(
                scored, scope="hidden_like", scope_value="verification_like_spatial"
            ),
            *EXP299.metric_rows(
                scored,
                scope="hidden_like",
                scope_value="verification_like_typewell_purged",
            ),
        ]
    )
    by_well = EXP299.by_well_metrics(scored)
    steps = EXP299.step_delta_metrics(scored)
    manifest = pd.DataFrame(
        {
            "outside_support_contribution_max_abs": [0.0] * 5,
            "base_mean_outside_or_boundary_contribution_max_abs": [0.0] * 5,
            "row_gate_min": [0.0] * 5,
            "row_gate_max": [1.0] * 5,
            "conditional_support_mass_relative_error_max": [0.0] * 5,
            "base_hmm_well_runs": [1] * 5,
            "variant_hmm_well_runs": [1] * 5,
            "total_hmm_well_runs": [2] * 5,
        }
    )
    technical["base_hmm_well_runs"] = 5
    technical["variant_hmm_well_runs"] = 5
    technical["total_hmm_well_runs"] = 10
    gates = EXP299.evaluate_hard_gates(
        generated=scored,
        well_manifest=manifest,
        overall=overall,
        folds=folds,
        distance=distance,
        scopes=scopes,
        hidden=hidden,
        by_well=by_well,
        steps=steps,
        join_audit={"row_identity_exact": True},
        control_manifest={"sha_exact": True},
        base_parity={"row_identity_exact": True, "max_abs_delta_ft": 0.0},
        base_reference_manifest={"sha_exact": True},
        config=config,
    )
    assert gates["technical_passed"] is True
    assert gates["performance_passed"] is True
    assert gates["passed"] is True

    failed_overall = overall.copy()
    failed_overall.loc[
        failed_overall["candidate"] == EXP299.VARIANT, "delta_rmse_vs_exp223"
    ] = 0.1
    failed = EXP299.evaluate_hard_gates(
        generated=scored,
        well_manifest=manifest,
        overall=failed_overall,
        folds=folds,
        distance=distance,
        scopes=scopes,
        hidden=hidden,
        by_well=by_well,
        steps=steps,
        join_audit={"row_identity_exact": True},
        control_manifest={"sha_exact": True},
        base_parity={"row_identity_exact": True, "max_abs_delta_ft": 0.0},
        base_reference_manifest={"sha_exact": True},
        config=config,
    )
    assert failed["passed"] is False


def test_compact_sources_are_self_contained_and_inference_is_fail_closed() -> None:
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "from settings import" not in train_source
    assert "from exact_hmm_smoother import" not in train_source
    assert "Path(__file__)" not in train_source
    assert "def compute_boundary_handoff_contribution(" in train_source
    assert "def run_hmm2_base_posterior_handoff(" in train_source
    assert "def run_full_experiment(" in train_source
    assert "Path(__file__)" not in inference_source
    contract = INFERENCE.validate_inference_disabled(load_config())
    assert contract["run_inference"] is False
    assert contract["write_submission"] is False
    with pytest.raises(RuntimeError, match="intentionally fail-closed"):
        INFERENCE.run_inference(load_config())

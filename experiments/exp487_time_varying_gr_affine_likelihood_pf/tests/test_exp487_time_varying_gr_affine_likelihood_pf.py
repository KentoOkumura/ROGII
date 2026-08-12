from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP = "exp487_time_varying_gr_affine_likelihood_pf"
EXP_DIR = ROOT / "experiments" / EXP
TRAIN_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
INFERENCE_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.py"
COMPACT_TRAIN = EXP_DIR / f"{EXP}_compact_selfcontained_train.ipynb"
COMPACT_INFERENCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.ipynb"
CANONICAL_TRAIN = EXP_DIR / f"{EXP}_train.ipynb"
CONFIG_PATH = EXP_DIR / "config.yaml"
EXP404_SOURCE = (
    ROOT
    / "experiments"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_train.py"
)
EXP350_DIR = (
    ROOT
    / "experiments"
    / "exp350_exp345_bidirectional_gr_affine_smoother"
)
EXP350_SOURCE = (
    EXP350_DIR
    / "exp350_exp345_bidirectional_gr_affine_smoother_compact_selfcontained_train.py"
)
FIXED32_MANIFEST = (
    ROOT
    / "experiments"
    / "exp411_predictive_filtered_rate_innovation_destick"
    / "assets"
    / "stage0_fixed32_manifest.csv"
)


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train() -> ModuleType:
    previous = os.environ.get("EXP487_IMPORT_ONLY")
    os.environ["EXP487_IMPORT_ONLY"] = "1"
    try:
        return load_module(TRAIN_SOURCE, "exp487_train_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP487_IMPORT_ONLY", None)
        else:
            os.environ["EXP487_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def inference() -> ModuleType:
    previous = os.environ.get("EXP487_IMPORT_ONLY")
    os.environ["EXP487_IMPORT_ONLY"] = "1"
    try:
        return load_module(INFERENCE_SOURCE, "exp487_inference_guard")
    finally:
        if previous is None:
            os.environ.pop("EXP487_IMPORT_ONLY", None)
        else:
            os.environ["EXP487_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def exp404() -> ModuleType:
    previous = os.environ.get("EXP404_IMPORT_ONLY")
    os.environ["EXP404_IMPORT_ONLY"] = "1"
    try:
        return load_module(EXP404_SOURCE, "exp404_parent_for_exp487")
    finally:
        if previous is None:
            os.environ.pop("EXP404_IMPORT_ONLY", None)
        else:
            os.environ["EXP404_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def exp350() -> ModuleType:
    return load_module(EXP350_SOURCE, "exp350_parent_for_exp487")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_horizontal(rows: int = 260, prefix_rows: int = 220) -> pd.DataFrame:
    md = 1000.0 + np.arange(rows, dtype=np.float64)
    tvt = 100.0 + 0.1 * np.arange(rows, dtype=np.float64)
    typewell_gr = 50.0 + 20.0 * np.sin(tvt / 12.0)
    raw_gr = 1.2 * typewell_gr + 5.0
    tvt_input = tvt.copy()
    tvt_input[prefix_rows:] = np.nan
    return pd.DataFrame(
        {
            "MD": md,
            "Z": np.zeros(rows, dtype=np.float64),
            "GR": raw_gr,
            "TVT_input": tvt_input,
        }
    )


def synthetic_typewell() -> pd.DataFrame:
    tvt = np.linspace(50.0, 260.0, 1200)
    return pd.DataFrame(
        {
            "TVT": tvt,
            "GR": 50.0 + 20.0 * np.sin(tvt / 12.0),
        }
    )


def test_stage0_is_complete_and_all_execution_is_disabled(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)
    assert (
        config["experiment"]["status"]
        == "stage0_completed_all_pass_pending_stage1_approval"
    )
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["implementation_approval_received"] is True
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert config["implementation"]["stage_0_implemented"] is True
    assert config["implementation"]["stage_1_implemented"] is True
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["execution"]["stage_0_execution_approved"] is False
    assert config["execution"]["selected_stage"] is None
    assert config["execution"]["run_stage_0"] is False
    assert config["execution"]["run_stage_1"] is False
    assert config["execution"]["stage_0_completed"] is True
    assert config["execution"]["stage_0_all_gates_pass"] is True
    assert (
        config["execution"]["stage_1_eligible_pending_separate_user_approval"]
        is True
    )
    assert contract["active_variants"] == [
        "causal_ekf_affine_emission",
        "bidirectional_rts_affine_emission",
    ]
    assert contract["pf"]["particles"] == 500
    assert contract["pf"]["seeds"] == 128
    assert contract["pf"]["temperature"] == 5.0
    assert contract["pf"]["sigma_rescaled_by_affine_a"] is False
    assert len(contract["scientific_contract_sha256"]) == 64
    with pytest.raises(RuntimeError, match="Kaggle push is not approved"):
        train.validate_scientific_contract(
            config,
            require_run_approval=True,
        )


def test_execution_counts_are_exact_and_zero_control(
    train: ModuleType,
    config: dict,
) -> None:
    assert train.validate_execution_contract(config) == {
        "scientific_variants": 2,
        "stage_0_candidate_pf_well_runs": 64,
        "stage_0_seed_well_trajectories": 8192,
        "stage_0_particle_starts": 4096000,
        "stage_1_candidate_pf_well_runs": 1546,
        "stage_1_seed_well_trajectories": 197888,
        "stage_1_particle_starts": 98944000,
        "control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }


def test_contract_rejects_parameter_rescue(
    train: ModuleType,
    config: dict,
) -> None:
    broken = copy.deepcopy(config)
    broken["model"]["fixed_from_exp404"]["particles"] = 1000
    with pytest.raises(ValueError, match="particles"):
        train.validate_scientific_contract(broken)

    broken = copy.deepcopy(config)
    broken["model"]["affine_state_common"]["slope_bounds"] = [0.2, 5.0]
    with pytest.raises(ValueError, match="slope_bounds"):
        train.validate_scientific_contract(broken)

    broken = copy.deepcopy(config)
    broken["guards"]["forbidden"].remove("same_oof_rescue")
    with pytest.raises(ValueError, match="forbidden-rescue"):
        train.validate_scientific_contract(broken)


def test_robust_prefix_fit_and_identity_fallback(
    train: ModuleType,
    config: dict,
) -> None:
    x = np.linspace(20.0, 100.0, 200)
    y = 1.4 * x - 7.0
    y[-1] = 10000.0
    fit = train.robust_affine_fit(x, y, config)
    assert fit["valid"] is True
    assert fit["scale_a"] == pytest.approx(1.4, abs=1.0e-10)
    assert fit["intercept_b"] == pytest.approx(-7.0, abs=1.0e-9)
    assert np.isfinite(fit["covariance"]).all()

    fallback = train.robust_affine_fit(
        np.ones(50, dtype=np.float64),
        np.arange(50, dtype=np.float64),
        config,
    )
    assert fallback["valid"] is False
    assert fallback["fallback_reason"] == "insufficient_typewell_gr_std"
    assert fallback["scale_a"] == 1.0
    assert fallback["intercept_b"] == 0.0


def test_pf_input_preparation_matches_exp404(
    train: ModuleType,
    exp404: ModuleType,
) -> None:
    horizontal = synthetic_horizontal(rows=24, prefix_rows=12)
    horizontal.loc[[2, 15, 16], "GR"] = np.nan
    typewell = synthetic_typewell()
    observed = train.prepare_likelihood_pf_inputs(horizontal, typewell)
    expected = exp404.prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        multiplier=1.0,
        grid_step=0.2,
    )
    for key in (
        "eval_indices",
        "eval_md",
        "eval_z",
        "eval_gr",
        "raw_gr_observed",
        "grid_gr",
    ):
        np.testing.assert_array_equal(observed[key], expected[key])
    for key in (
        "last_known_tvt",
        "last_known_position",
        "initial_rate",
        "grid_minimum",
        "grid_step",
    ):
        assert observed[key] == expected[key]
    assert observed["scale_audit"]["candidate_scale"] == expected["scale_audit"][
        "candidate_scale"
    ]


def test_identity_affine_kernel_is_bitwise_exp404_parity(
    train: ModuleType,
    exp404: ModuleType,
) -> None:
    md = np.arange(1.0, 8.0, dtype=np.float64)
    z = np.linspace(0.0, 0.6, len(md), dtype=np.float64)
    gr = np.asarray([50.0, 52.0, 54.0, 53.0, 51.0, 50.0, 49.0])
    grid_gr = np.linspace(40.0, 70.0, 151)
    expected = exp404._pf_lik_allseeds(
        md,
        z,
        gr,
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        48,
        6,
        12345,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    observed = train._pf_affine_allseeds(
        md,
        z,
        gr,
        np.ones(len(md), dtype=np.float64),
        np.zeros(len(md), dtype=np.float64),
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        48,
        6,
        12345,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
        0.01,
        100.0,
        600.0,
    )
    for index in range(5):
        assert np.array_equal(observed[index], expected[index])


def test_dynamic_affine_emission_is_active_and_sigma_is_not_rescaled(
    train: ModuleType,
) -> None:
    contract = train.dynamic_affine_emission_contract()
    assert contract["pass"] is True
    assert contract["sigma_rescaled_by_a"] is False
    md = np.arange(1.0, 8.0, dtype=np.float64)
    z = np.linspace(0.0, 0.6, len(md), dtype=np.float64)
    gr = np.asarray([50.0, 52.0, 54.0, 53.0, 51.0, 50.0, 49.0])
    grid_gr = np.linspace(40.0, 70.0, 151)
    identity = train._pf_affine_allseeds(
        md,
        z,
        gr,
        np.ones(len(md)),
        np.zeros(len(md)),
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        64,
        8,
        4321,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
        0.01,
        100.0,
        600.0,
    )
    affine = train._pf_affine_allseeds(
        md,
        z,
        gr,
        np.full(len(md), 1.2),
        np.full(len(md), 5.0),
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        64,
        8,
        4321,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
        0.01,
        100.0,
        600.0,
    )
    assert np.isfinite(affine[0]).all()
    assert not np.array_equal(identity[0], affine[0])


def test_causal_schedule_missing_skip_and_rts_contract(
    train: ModuleType,
    config: dict,
) -> None:
    horizontal = synthetic_horizontal()
    typewell = synthetic_typewell()
    prepared = train.prepare_likelihood_pf_inputs(horizontal, typewell)
    eval_indices = prepared["eval_indices"]
    horizontal.loc[eval_indices[:5], "GR"] = np.nan
    prepared = train.prepare_likelihood_pf_inputs(horizontal, typewell)
    base_mean = 100.0 + 0.1 * eval_indices
    base_std = np.ones(len(eval_indices), dtype=np.float64)
    forward, audit, context = train.forward_affine_schedule(
        horizontal,
        typewell,
        prepared,
        base_mean,
        base_std,
        {"q_intercept": 1.0e-6, "q_log_scale": 1.0e-10},
        config,
    )
    assert not forward["raw_gr_update"].iloc[:5].any()
    assert forward["raw_gr_update"].iloc[5:].all()
    assert audit["fallback"] is False
    assert forward["affine_scale_a"].between(0.25, 4.0).all()
    rts, rts_audit = train.bidirectional_rts_schedule(
        forward,
        audit,
        context,
        config,
    )
    parity = train.forward_schedule_parity(forward, rts)
    assert parity["passed"] is True
    assert rts.iloc[-1]["affine_intercept_b"] == pytest.approx(
        forward.iloc[-1]["filtered_intercept_b"],
        abs=1.0e-12,
    )
    assert rts_audit["terminal_state_max_abs_error"] == 0.0
    assert rts_audit["terminal_covariance_max_abs_error"] == 0.0
    assert rts_audit["covariance_minimum_eigenvalue_before_floor"] >= -1.0e-8
    causal_decorated = train.decorate_schedule(
        forward,
        well="synthetic-well",
        kind=train.CAUSAL_VARIANT,
    )
    rts_decorated = train.decorate_schedule(
        rts,
        well="synthetic-well",
        kind=train.RTS_VARIANT,
    )
    assert causal_decorated.columns.tolist() == list(train.SCHEDULE_COLUMNS)
    assert rts_decorated.columns.tolist() == [
        *train.SCHEDULE_COLUMNS,
        *train.RTS_EXTRA_COLUMNS,
    ]
    assert causal_decorated.columns[:4].tolist() == [
        "id",
        "well_id",
        "row_idx",
        "suffix_offset",
    ]


def test_forward_and_rts_schedules_match_exp350_reference(
    train: ModuleType,
    exp350: ModuleType,
    config: dict,
) -> None:
    horizontal = synthetic_horizontal()
    typewell = synthetic_typewell()
    horizontal.loc[[225, 226, 240], "GR"] = np.nan
    prepared487 = train.prepare_likelihood_pf_inputs(horizontal, typewell)
    config350 = exp350.read_yaml(EXP350_DIR / "config.yaml")
    prepared350 = exp350.prepare_hmm_inputs(horizontal, typewell, config350)
    eval_indices = prepared487["eval_indices"]
    np.testing.assert_array_equal(prepared350["eval_index"], eval_indices)
    base_mean = 100.0 + 0.1 * eval_indices
    base_std = np.linspace(0.5, 1.5, len(eval_indices))
    process = {"q_intercept": 1.0e-6, "q_log_scale": 1.0e-10}
    observed_forward, observed_audit, observed_context = (
        train.forward_affine_schedule(
            horizontal,
            typewell,
            prepared487,
            base_mean,
            base_std,
            process,
            config,
        )
    )
    expected_forward, expected_audit, expected_context = (
        exp350.forward_affine_schedule(
            horizontal,
            typewell,
            prepared350,
            base_mean,
            base_std,
            process,
            config350,
        )
    )
    common_forward = [
        "row_idx",
        "affine_scale_a",
        "affine_intercept_b",
        "raw_gr_update",
        "predictive_nll_identity",
        "predictive_nll_affine",
        "observation_variance",
        "predicted_intercept_b",
        "predicted_log_scale_a",
        "predicted_p00",
        "predicted_p01",
        "predicted_p11",
        "filtered_intercept_b",
        "filtered_log_scale_a",
        "filtered_p00",
        "filtered_p01",
        "filtered_p11",
    ]
    pd.testing.assert_frame_equal(
        observed_forward[common_forward],
        expected_forward[common_forward],
        check_exact=True,
    )
    assert observed_audit["forward_boundary_jump_sigma"] == expected_audit[
        "forward_boundary_jump_sigma"
    ]
    observed_rts, observed_rts_audit = train.bidirectional_rts_schedule(
        observed_forward,
        observed_audit,
        observed_context,
        config,
    )
    expected_rts, expected_rts_audit = exp350.bidirectional_rts_schedule(
        expected_forward,
        expected_audit,
        expected_context,
        config350,
    )
    common_rts = [
        "row_idx",
        "affine_scale_a",
        "affine_intercept_b",
        "smoothed_log_scale_a",
        "smoothed_p00",
        "smoothed_p01",
        "smoothed_p11",
        "raw_gr_update",
        "predictive_nll_identity",
    ]
    pd.testing.assert_frame_equal(
        observed_rts[common_rts],
        expected_rts[common_rts],
        check_exact=True,
    )
    for key in (
        "terminal_state_max_abs_error",
        "terminal_covariance_max_abs_error",
        "covariance_minimum_eigenvalue_before_floor",
        "covariance_contraction_max_positive_eigenvalue",
        "output_scale_clip_fraction",
    ):
        assert observed_rts_audit[key] == expected_rts_audit[key]


def test_one_well_target_free_decode_smoke(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    well = "synthetic-well"
    horizontal = synthetic_horizontal()
    typewell = synthetic_typewell()
    horizontal_with_truth = horizontal.copy()
    horizontal_with_truth["TVT"] = 100.0 + 0.1 * np.arange(len(horizontal))
    horizontal_with_truth.to_csv(
        tmp_path / f"{well}__horizontal_well.csv",
        index=False,
    )
    typewell.to_csv(tmp_path / f"{well}__typewell.csv", index=False)
    prepared = train.prepare_likelihood_pf_inputs(horizontal, typewell)
    eval_indices = prepared["eval_indices"]
    saved_base = pd.DataFrame(
        {
            "well_id": well,
            "row_idx": eval_indices,
            "base_mean": 100.0 + 0.1 * eval_indices,
            "base_std": np.ones(len(eval_indices), dtype=np.float64),
        }
    )
    local_config = copy.deepcopy(config)
    local_config["model"]["fixed_from_exp404"]["particles"] = 8
    local_config["model"]["fixed_from_exp404"]["seeds"] = 3
    result = train.decode_target_free_well(
        well,
        tmp_path,
        saved_base,
        {"q_intercept": 1.0e-6, "q_log_scale": 1.0e-10},
        local_config,
    )
    assert len(result.prediction) == len(eval_indices)
    assert result.causal_schedule.columns.tolist() == list(
        train.SCHEDULE_COLUMNS
    )
    assert result.rts_schedule.columns.tolist() == [
        *train.SCHEDULE_COLUMNS,
        *train.RTS_EXTRA_COLUMNS,
    ]
    assert result.audit["candidate_pf_well_runs"] == 2
    assert result.audit["seed_well_trajectories"] == 6
    assert result.audit["particle_starts"] == 48
    assert result.audit["forward_parity_pass"] is True
    ledger = train.LeakageLedger(expected_variant_wells=2)
    frozen_outputs = train.freeze_target_free_outputs(
        [result],
        tmp_path / "artifacts",
        ledger=ledger,
        stage="synthetic",
        expected_rows=len(eval_indices),
        expected_wells=1,
    )
    predictions, causal_schedule, rts_schedule = frozen_outputs[:3]
    frozen = frozen_outputs[-1]
    train.verify_frozen_content(
        predictions,
        causal_schedule,
        rts_schedule,
        frozen,
    )
    assert frozen["frozen_before_truth_attachment"] is True
    assert frozen["truth_access_ledger_at_freeze"]["all_frozen"] is True
    control_source = pd.DataFrame(
        {
            "id": predictions["id"],
            "well_id": predictions["well_id"],
            "row_idx": predictions["row_idx"],
            "likpf_scale_5_x1p0": predictions[train.CAUSAL_PREDICTION].astype(
                np.float32
            ),
            "likpf_scale_5_x1p3": predictions[train.RTS_PREDICTION].astype(
                np.float32
            ),
            "likpf_mean_x1p0": predictions[train.CAUSAL_PREDICTION].astype(
                np.float32
            ),
            "likpf_mean_x1p3": predictions[train.RTS_PREDICTION].astype(
                np.float32
            ),
        }
    )
    control_path = tmp_path / "control.csv.gz"
    control_artifact = train.write_deterministic_gzip_csv(
        control_source,
        control_path,
    )
    control_spec = local_config["data"]["saved_control"]
    control_spec["filename"] = control_path.name
    control_spec["candidates"] = [str(tmp_path)]
    control_spec["patterns"] = []
    control_spec["expected_raw_sha256"] = control_artifact["raw_sha256"]
    control_spec["expected_decompressed_sha256"] = control_artifact[
        "decompressed_sha256"
    ]
    roles_path = tmp_path / "fixed32.csv"
    pd.DataFrame(
        {"well": [well], "role": ["diagnostic"], "fold": [0]}
    ).to_csv(roles_path, index=False)
    manifest_spec = local_config["data"]["fixed32_manifest"]
    manifest_spec["filename"] = roles_path.name
    manifest_spec["local"] = str(roles_path)
    manifest_spec["expected_sha256"] = hashlib.sha256(roles_path.read_bytes()).hexdigest()
    truth_late = train.attach_truth_late(
        predictions,
        causal_schedule,
        rts_schedule,
        frozen,
        stage="stage0",
        config=local_config,
        raw_dir=tmp_path,
        fold_map={},
        ledger=ledger,
    )
    assert len(truth_late) == len(predictions)
    assert np.isfinite(
        truth_late[
            [
                "true_tvt",
                train.PRIMARY_CONTROL,
                train.CAUSAL_PREDICTION,
                train.RTS_PREDICTION,
            ]
        ].to_numpy(np.float64)
    ).all()
    assert ledger.report()["forbidden_before_freeze"] == {
        "truth_rows": 0,
        "error_rows": 0,
        "outcome_fold_rows": 0,
        "hidden_role_rows": 0,
    }


def test_invalid_prefix_keeps_exact_identity_schedule(
    train: ModuleType,
    config: dict,
) -> None:
    horizontal = synthetic_horizontal(rows=80, prefix_rows=30)
    typewell = synthetic_typewell()
    prepared = train.prepare_likelihood_pf_inputs(horizontal, typewell)
    base_mean = 100.0 + 0.1 * prepared["eval_indices"]
    base_std = np.ones(len(base_mean), dtype=np.float64)
    forward, audit, context = train.forward_affine_schedule(
        horizontal,
        typewell,
        prepared,
        base_mean,
        base_std,
        {"q_intercept": 1.0e-6, "q_log_scale": 1.0e-10},
        config,
    )
    assert audit["fallback"] is True
    np.testing.assert_array_equal(forward["affine_scale_a"], 1.0)
    np.testing.assert_array_equal(forward["affine_intercept_b"], 0.0)
    rts, _ = train.bidirectional_rts_schedule(forward, audit, context, config)
    np.testing.assert_array_equal(rts["affine_scale_a"], 1.0)
    np.testing.assert_array_equal(rts["affine_intercept_b"], 0.0)


def test_common_seed_label_excludes_variant(train: ModuleType) -> None:
    key = "likpf::train::well-a"
    expected = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    expected = expected % 2_147_483_647 + 1
    causal = train.stable_seed("likpf", "train", "well-a")
    rts = train.stable_seed("likpf", "train", "well-a")
    assert causal == rts == expected
    assert train.stable_seed("likpf", "train", "well-b") != expected
    assert "causal" not in key and "rts" not in key


def test_fixed32_manifest_and_leakage_ledger(
    train: ModuleType,
    config: dict,
) -> None:
    observed_sha = hashlib.sha256(FIXED32_MANIFEST.read_bytes()).hexdigest()
    assert observed_sha == config["data"]["fixed32_manifest"]["expected_sha256"]
    wells, report = train.load_fixed32_scope(config)
    assert len(wells) == 32
    assert len(set(wells)) == 32
    assert report["columns_read_before_freeze"] == ["well"]

    ledger = train.LeakageLedger(expected_variant_wells=4)
    ledger.record_base(10)
    ledger.record_process_fold(2)
    ledger.freeze(train.CAUSAL_VARIANT, "a")
    ledger.freeze(train.RTS_VARIANT, "a")
    ledger.record_truth(2)
    ledger.freeze(train.CAUSAL_VARIANT, "b")
    ledger.freeze(train.RTS_VARIANT, "b")
    ledger.record_control(4)
    ledger.record_outcome_fold(2)
    report = ledger.report()
    assert report["all_frozen"] is True
    assert report["forbidden_before_freeze"]["truth_rows"] == 2
    assert report["after_freeze"]["control_rows"] == 4


def test_raw_identity_uses_typed_exp404_contract(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    rows: list[dict[str, str]] = []
    for well, horizontal_text, typewell_text in (
        ("well-a", "MD,TVT\n1,2\n", "TVT,GR\n2,30\n"),
        ("well-b", "MD,TVT\n3,4\n", "TVT,GR\n4,50\n"),
    ):
        horizontal_path = tmp_path / f"{well}__horizontal_well.csv"
        typewell_path = tmp_path / f"{well}__typewell.csv"
        horizontal_path.write_text(horizontal_text)
        typewell_path.write_text(typewell_text)
        rows.append(
            {
                "well_id": well,
                "horizontal_raw_sha256": train.sha256_path(horizontal_path),
                "typewell_raw_sha256": train.sha256_path(typewell_path),
            }
        )
    frame = (
        pd.DataFrame(rows)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    columns = [
        "well_id",
        "horizontal_raw_sha256",
        "typewell_raw_sha256",
    ]
    expected = train.typed_dataframe_content_sha(frame, columns)
    assert expected != train.dataframe_content_sha(frame, columns)
    local_config = copy.deepcopy(config)
    local_config["validation"]["expected_wells"] = 2
    local_config["data"]["expected_raw_well_identity_sha256"] = expected
    report = train.validate_raw_well_identity(local_config, tmp_path)
    assert report["wells"] == 2
    assert report["content_sha256"] == expected


def test_saved_control_uses_artifact_sha_and_records_logical_provenance(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "id": ["well-a_1", "well-a_2"],
            "well_id": ["well-a", "well-a"],
            "row_idx": [1, 2],
            "likpf_scale_5_x1p0": np.asarray([10.25, 11.5], dtype=np.float32),
            "likpf_scale_5_x1p3": np.asarray([10.5, 11.75], dtype=np.float32),
            "likpf_mean_x1p0": np.asarray([10.0, 11.0], dtype=np.float32),
            "likpf_mean_x1p3": np.asarray([10.1, 11.1], dtype=np.float32),
        }
    )
    control_path = tmp_path / "saved_control.csv.gz"
    artifact = train.write_deterministic_gzip_csv(source, control_path)
    local_config = copy.deepcopy(config)
    spec = local_config["data"]["saved_control"]
    spec["filename"] = control_path.name
    spec["candidates"] = [str(tmp_path)]
    spec["patterns"] = []
    spec["expected_raw_sha256"] = artifact["raw_sha256"]
    spec["expected_decompressed_sha256"] = artifact["decompressed_sha256"]
    spec["expected_logical_sha256"] = "f" * 64
    ledger = train.LeakageLedger(expected_variant_wells=2)
    ledger.freeze(train.CAUSAL_VARIANT, "well-a")
    ledger.freeze(train.RTS_VARIANT, "well-a")
    control = train.load_saved_control_after_freeze(
        local_config,
        {"well-a_1", "well-a_2"},
        ledger,
    )
    assert control.columns.tolist() == ["id", train.PRIMARY_CONTROL]
    np.testing.assert_allclose(
        control[train.PRIMARY_CONTROL],
        source["likpf_scale_5_x1p0"],
    )
    assert ledger.report()["after_freeze"]["control_rows"] == 2


def test_stage1_metrics_report_variants_independently(train: ModuleType) -> None:
    rows = 20
    truth = np.linspace(100.0, 119.0, rows)
    frame = pd.DataFrame(
        {
            "well_id": ["w0"] * 10 + ["w1"] * 10,
            "fold": [0] * 10 + [1] * 10,
            "true_tvt": truth,
            "raw_gr_observed": [True, False] * 10,
            "well_missing_fraction": [0.6] * 10 + [0.1] * 10,
            "md_since": [1200.0] * 4 + [100.0] * 16,
            "hidden_like_spatial": [True] * 10 + [False] * 10,
            "hidden_like_typewell_purged": [False] * 10 + [True] * 10,
            "base_exp209_mean": truth + 0.5,
            "likpf_scale_5_x1p0": truth + 1.0,
            "likpf_scale5_causal_affine": truth + 0.5,
            "likpf_scale5_bidirectional_rts_affine": truth + 0.7,
        }
    )
    primary, by_well, blend = train.build_stage1_metric_outputs(frame)
    assert set(primary["variant"]) == {
        train.CAUSAL_VARIANT,
        train.RTS_VARIANT,
    }
    assert set(primary["scope"]) == {
        "overall",
        "fold_0",
        "fold_1",
        "raw_gr_observed",
        "raw_gr_missing",
        "missing_fraction_high",
        "md_since_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    }
    assert len(by_well) == 4
    assert len(blend) == 2
    causal_overall = primary.loc[
        primary["variant"].eq(train.CAUSAL_VARIANT)
        & primary["scope"].eq("overall")
    ].iloc[0]
    assert causal_overall["improvement_ft"] == pytest.approx(0.5)


def test_inference_remains_fail_closed(
    inference: ModuleType,
    config: dict,
) -> None:
    status = inference.validate_inference_is_disabled(config)
    assert status == {
        "implementation_scope": "train_side_stage0_and_stage1_only",
        "canonical_inference_notebook_adopted": False,
        "inference_enabled": False,
        "run_inference": False,
        "create_submission": False,
        "submit_to_kaggle": False,
        "test_exp209_path_regeneration_implemented": False,
        "test_deployment_process_noise_implemented": False,
    }
    with pytest.raises(RuntimeError, match="not implemented or approved"):
        inference.run_inference(config)


def test_compact_sources_are_self_contained_and_canonical_is_adopted(
    config: dict,
) -> None:
    train_source = TRAIN_SOURCE.read_text()
    for heading in (
        "Robust prefix affine and outer-fold process-noise helpers",
        "Causal EKF and bidirectional extended-RTS schedules",
        "Dynamic-affine likelihood-PF kernel",
        "Target-free two-variant generation",
        "Truth-late readout",
        "All-well Stage 1 independent scientific gates",
    ):
        assert heading in train_source
    assert "Path(__file__)" not in train_source
    assert "from settings import" not in train_source
    assert "def _pf_affine_allseeds(" in train_source
    assert "@njit(cache=True)\ndef _interp1(" in train_source
    assert "def forward_affine_schedule(" in train_source
    assert "def bidirectional_rts_schedule(" in train_source
    assert "def run_selected_stage(" in train_source
    assert COMPACT_TRAIN.exists()
    assert COMPACT_INFERENCE.exists()
    compact = json.loads(COMPACT_TRAIN.read_text())
    compact_text = "\n".join(
        "".join(cell.get("source", [])) for cell in compact["cells"]
    )
    assert "def _pf_affine_allseeds(" in compact_text
    assert "def bidirectional_rts_schedule(" in compact_text
    assert "def run_selected_stage(" in compact_text
    assert len(compact["cells"]) >= 20
    canonical = json.loads(CANONICAL_TRAIN.read_text())
    canonical_text = "\n".join(
        "".join(cell.get("source", [])) for cell in canonical["cells"]
    )
    assert "Design-only placeholder" not in canonical_text
    assert "def _pf_affine_allseeds(" in canonical_text
    assert "def run_selected_stage(" in canonical_text
    assert config["implementation"]["canonical_train_notebook_adopted"] is True

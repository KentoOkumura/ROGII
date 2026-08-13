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

from tests.test_support import require_saved_files

ROOT = Path(__file__).resolve().parents[3]
EXP = "exp484_student_t_gr_filtering_likelihood_pf"
EXP_DIR = ROOT / "experiments" / EXP
TRAIN_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
INFERENCE_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.py"
CONFIG_PATH = EXP_DIR / "config.yaml"
MANIFEST_PATH = EXP_DIR / "assets" / "stage0_fixed32_manifest.csv"
CANONICAL_TRAIN = EXP_DIR / f"{EXP}_train.ipynb"
COMPACT_TRAIN = EXP_DIR / f"{EXP}_compact_selfcontained_train.ipynb"
KAGGLE_TRAIN_DIR = EXP_DIR / "kaggle" / "train"
KAGGLE_METADATA = KAGGLE_TRAIN_DIR / "kernel-metadata.json"
EXP404_SOURCE = (
    ROOT
    / "experiments"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_train.py"
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
    previous = os.environ.get("EXP484_IMPORT_ONLY")
    os.environ["EXP484_IMPORT_ONLY"] = "1"
    try:
        return load_module(TRAIN_SOURCE, "exp484_train_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP484_IMPORT_ONLY", None)
        else:
            os.environ["EXP484_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def inference() -> ModuleType:
    return load_module(INFERENCE_SOURCE, "exp484_inference_guard")


@pytest.fixture(scope="module")
def exp404() -> ModuleType:
    previous = os.environ.get("EXP404_IMPORT_ONLY")
    os.environ["EXP404_IMPORT_ONLY"] = "1"
    try:
        return load_module(EXP404_SOURCE, "exp404_parent_for_exp484")
    finally:
        if previous is None:
            os.environ.pop("EXP404_IMPORT_ONLY", None)
        else:
            os.environ["EXP404_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_horizontal() -> pd.DataFrame:
    rows = 14
    return pd.DataFrame(
        {
            "MD": 1000.0 + np.arange(rows, dtype=np.float64),
            "Z": np.linspace(-8000.0, -7998.7, rows),
            "GR": [
                50.0,
                np.nan,
                52.0,
                54.0,
                51.0,
                49.0,
                48.0,
                np.nan,
                53.0,
                90.0,
                54.0,
                52.0,
                49.0,
                51.0,
            ],
            "TVT_input": [
                100.0,
                100.5,
                101.0,
                101.5,
                102.0,
                102.5,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )


def synthetic_typewell() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TVT": np.linspace(95.0, 110.0, 76),
            "GR": np.linspace(45.0, 75.0, 76),
        }
    )


def test_stage1_is_complete_and_other_execution_boundaries_are_locked(
    train: ModuleType,
    config: dict,
) -> None:
    counts = train.validate_execution_contract(
        config,
        require_run_approval=False,
    )
    assert (
        config["experiment"]["status"]
        == "stage1_gate_failed_terminal_close"
    )
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["implementation_approval_received"] is True
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert config["implementation"]["canonical_inference_notebook_adopted"] is False
    assert config["implementation"]["kaggle_package_created"] is True
    assert config["implementation"]["train_execution_enabled"] is True
    assert config["implementation"]["inference_enabled"] is False
    assert config["implementation"]["submission_enabled"] is False
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["stage_0_execution_approved"] is True
    assert config["execution"]["run_stage_0"] is False
    assert config["execution"]["run_stage_1"] is False
    assert config["execution"]["run_inference"] is False
    assert config["execution"]["create_submission"] is False
    assert counts["stage_0_candidate_pf_well_runs"] == 32
    assert counts["stage_0_seed_well_trajectories"] == 4096
    assert counts["stage_0_particle_starts"] == 2_048_000
    assert counts["stage_1_candidate_pf_well_runs"] == 773
    assert counts["stage_1_seed_well_trajectories"] == 98_944
    assert counts["stage_1_particle_starts"] == 49_472_000
    assert counts["control_pf_well_runs"] == 0
    assert counts["hmm_well_runs"] == 0
    assert counts["beam_well_runs"] == 0
    assert counts["lightgbm_configs"] == 0
    assert counts["boosters"] == 0
    assert counts["gpu_runs"] == 0
    assert config["execution"]["stage_0_kernel_version"] == 2
    assert config["execution"]["stage_0_kernel_status"] == "COMPLETE"
    assert config["execution"]["stage_1_execution_approved"] is True
    assert config["execution"]["stage_1_kernel_status"] == "COMPLETE"
    assert config["stage_1_result"]["technical_gate"]["passed"] is True
    assert config["stage_1_result"]["scientific_gate"]["passed"] is False
    assert (
        config["stage_1_result"]["scientific_gate"]["decision"]
        == "terminal_close_without_student_t_or_pf_rescue"
    )
    assert config["stage_0_result"]["all_technical_gates_passed"] is True
    assert config["stage_0_result"]["technical_gates_passed"] == 16
    assert config["stage_0_result"]["technical_gates_total"] == 16
    assert train.validate_execution_contract(
        config,
        require_run_approval=True,
    ) == counts

    denied = copy.deepcopy(config)
    denied["execution"]["run_stage_1"] = True
    denied["execution"]["stage_1_execution_approved"] = False
    with pytest.raises(RuntimeError, match="Stage 1 execution is not approved"):
        train.validate_execution_contract(
            denied,
            require_run_approval=True,
        )

    simultaneous = copy.deepcopy(config)
    simultaneous["execution"]["run_stage_0"] = True
    simultaneous["execution"]["run_stage_1"] = True
    with pytest.raises(ValueError, match="exactly one active execution stage"):
        train.validate_execution_contract(
            simultaneous,
            require_run_approval=True,
        )


def test_scientific_contract_pins_only_student_t_emission(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)
    assert contract["primary_candidate"] == "likpf_scale5_student_t_df4"
    assert contract["primary_control"] == "likpf_scale_5_x1p0"
    assert contract["changed_factor"] == {
        "family": "student_t",
        "df": 4.0,
        "formula": "-0.5*(df+1)*log1p(z^2/df)",
        "normalization_constant": "omitted_as_particle_state_independent",
        "additional_clip": None,
        "application_scope": "every_finite_gr_particle_update",
    }
    assert contract["fixed_from_exp404"]["particles"] == 500
    assert contract["fixed_from_exp404"]["seeds"] == 128
    assert contract["fixed_from_exp404"]["primary_seed_weighting_temperature"] == 5.0
    assert contract["fixed_from_exp404"]["gr_scale_multiplier"] == 1.0
    assert contract["saved_control_rerun"] is False
    assert contract["seed_policy"]["variant_name_in_seed"] is False
    assert len(contract["scientific_contract_sha256"]) == 64

    broken = copy.deepcopy(config)
    broken["model"]["changed_factor"]["df"] = 6.0
    with pytest.raises(ValueError, match="model.changed_factor.df"):
        train.validate_scientific_contract(broken)


def test_content_sha_is_exact_exp404_parent_algorithm(
    train: ModuleType,
    exp404: ModuleType,
) -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["alpha", "beta"],
            "horizontal_raw_sha256": ["0" * 64, "1" * 64],
            "typewell_raw_sha256": ["2" * 64, "3" * 64],
            "numeric": np.asarray([7, 11], dtype=np.int64),
        }
    )
    columns = [
        "well_id",
        "horizontal_raw_sha256",
        "typewell_raw_sha256",
        "numeric",
    ]
    assert train.dataframe_content_sha(
        frame,
        columns,
    ) == exp404.dataframe_content_sha(frame, columns)


def test_student_t_formula_center_limit_and_extreme_weights(
    train: ModuleType,
) -> None:
    z = np.asarray([-8.0, -1.0, 0.0, 1.0, 8.0], dtype=np.float64)
    observed = train.student_t_log_emission(z, df=4.0)
    expected = -2.5 * np.log1p((z * z) / 4.0)
    np.testing.assert_array_equal(observed, expected)
    assert observed[0] == observed[-1]
    assert observed[1] == observed[-2]
    assert observed[2] == 0.0

    report = train.student_t_formula_contract(4.0)
    assert report["passed"] is True
    assert report["checks"]["gaussian_center_quadratic"] is True
    assert report["checks"]["extreme_score_finite"] is True
    assert report["checks"]["finite_positive_weights"] is True
    assert report["additional_clip"] is None


def test_pf_input_preparation_is_exact_exp404_x1p0(
    train: ModuleType,
    exp404: ModuleType,
) -> None:
    horizontal = synthetic_horizontal()
    typewell = synthetic_typewell()
    observed = train.prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        grid_step=0.2,
    )
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
        "md_since",
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
    assert observed["scale_audit"] == expected["scale_audit"]


def test_stable_hash_fixed32_manifest_is_target_free_and_sha_pinned(
    train: ModuleType,
    config: dict,
) -> None:
    expected_sha = config["data"]["fixed32_manifest"]["expected_sha256"]
    assert hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest() == expected_sha
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"well_id": str})
    assert list(manifest.columns) == [
        "well_id",
        "selection_sha256",
        "total_rows",
        "suffix_rows",
    ]
    assert len(manifest) == 32
    assert manifest["well_id"].nunique() == 32
    assert int(manifest["suffix_rows"].sum()) == 165_010
    assert not {
        "TVT",
        "true_tvt",
        "error",
        "fold",
        "role",
        "hidden_like",
    } & set(manifest.columns)

    raw_wells = sorted(
        path.name.replace("__horizontal_well.csv", "")
        for path in (ROOT / "data" / "raw" / "train").glob(
            "*__horizontal_well.csv"
        )
    )
    raw_identity = pd.DataFrame(
        {
            "well_id": raw_wells,
            "horizontal_raw_sha256": ["unused"] * len(raw_wells),
            "typewell_raw_sha256": ["unused"] * len(raw_wells),
        }
    )
    loaded, report = train.load_fixed32_manifest(config, raw_identity)
    pd.testing.assert_frame_equal(loaded, manifest)
    assert report["wells"] == 32
    assert report["suffix_rows"] == 165_010


def test_student_t_kernel_is_stable_finite_and_changes_shock_weights(
    train: ModuleType,
    exp404: ModuleType,
) -> None:
    md = np.arange(1.0, 10.0, dtype=np.float64)
    z = np.linspace(0.0, 0.8, len(md), dtype=np.float64)
    gr = np.asarray(
        [50.0, 52.0, 54.0, 120.0, 51.0, 50.0, 49.0, 51.0, 52.0],
        dtype=np.float64,
    )
    grid_gr = np.linspace(40.0, 70.0, 151, dtype=np.float64)
    parent_args = (
        md,
        z,
        gr,
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        24,
        4,
        12345,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    gaussian = exp404._pf_lik_allseeds(*parent_args)
    student_args = (
        md,
        z,
        gr,
        grid_gr,
        90.0,
        0.2,
        20.0,
        4.0,
        100.0,
        0.01,
        24,
        4,
        12345,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    first = train._pf_student_t_allseeds(*student_args)
    second = train._pf_student_t_allseeds(*student_args)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
        assert np.isfinite(left).all()
    assert not np.array_equal(first[0], gaussian[0])
    assert not np.array_equal(first[1], gaussian[1])
    assert np.all(first[3] > 0.0)
    assert np.all(first[3] <= 24.0)


def test_temperature5_seed_aggregation_is_fixed_and_finite(
    train: ModuleType,
) -> None:
    predictions = np.asarray(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=np.float64,
    )
    log_likelihoods = np.asarray([-10.0, -5.0, -1.0], dtype=np.float64)
    observed = train.aggregate_temperature5(
        predictions,
        log_likelihoods,
        temperature=5.0,
    )
    weights = np.exp((log_likelihoods - log_likelihoods.max()) / 5.0)
    weights /= weights.sum()
    np.testing.assert_allclose(
        observed,
        (weights[:, None] * predictions).sum(axis=0),
    )
    with pytest.raises(ValueError, match="temperature"):
        train.aggregate_temperature5(
            predictions,
            log_likelihoods,
            temperature=8.0,
        )


def test_truth_and_saved_control_are_fail_closed_before_freeze(
    train: ModuleType,
) -> None:
    ledger = train.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="before candidate"):
        ledger.record_truth(1)
    ledger = train.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="before candidate"):
        ledger.record_control(1)
    ledger = train.TruthAccessLedger()
    ledger.mark_frozen()
    ledger.record_truth(3)
    ledger.record_control(5)
    ledger.record_fold(7)
    ledger.record_hidden_like(11)
    assert ledger.report()["before_freeze"] == {
        "truth_rows": 0,
        "control_rows": 0,
        "fold_rows": 0,
        "hidden_like_rows": 0,
    }
    assert ledger.report()["after_freeze"]["truth_rows"] == 3
    assert ledger.report()["after_freeze"]["control_rows"] == 5
    assert ledger.report()["after_freeze"]["fold_rows"] == 7
    assert ledger.report()["after_freeze"]["hidden_like_rows"] == 11


def test_stage1_all_and_gate_accepts_a_clean_synthetic_candidate(
    train: ModuleType,
    config: dict,
) -> None:
    rows = 773
    wells = [f"well_{index:03d}" for index in range(rows)]
    frame = pd.DataFrame(
        {
            "id": [f"{well}:0" for well in wells],
            "well_id": wells,
            "row_idx": np.zeros(rows, dtype=np.int64),
            "suffix_offset": np.zeros(rows, dtype=np.int64),
            "last_known_tvt": np.zeros(rows, dtype=np.float64),
            "md_since": np.full(rows, 1200.0, dtype=np.float64),
            "raw_gr_observed": np.arange(rows) % 2 == 0,
            "well_missing_fraction": np.full(rows, 0.5, dtype=np.float64),
            train.PRIMARY_CANDIDATE: np.zeros(rows, dtype=np.float32),
            train.PRIMARY_CONTROL: np.ones(rows, dtype=np.float32),
            "saved_exp209_hmm": np.zeros(rows, dtype=np.float64),
            "candidate_hmm_50_50": np.zeros(rows, dtype=np.float64),
            "control_hmm_50_50": np.full(rows, 0.5, dtype=np.float64),
            "true_tvt": np.zeros(rows, dtype=np.float64),
            "fold": np.arange(rows, dtype=np.int64) % 5,
            "hidden_like_spatial": np.ones(rows, dtype=bool),
            "hidden_like_typewell_purged": np.ones(rows, dtype=bool),
        }
    )
    primary, by_well, blend = train.build_stage1_metric_outputs(frame)
    audit = pd.DataFrame(
        {
            "well_id": wells,
            "status": ["ok"] * rows,
            "seed_base": [
                train.stable_seed("likpf", "train", well) for well in wells
            ],
            "pf_well_runs": np.ones(rows, dtype=np.int64),
            "seed_well_trajectories": np.full(
                rows,
                128,
                dtype=np.int64,
            ),
            "particle_starts": np.full(
                rows,
                64_000,
                dtype=np.int64,
            ),
            "minimum_ess_min": np.full(rows, 1.0, dtype=np.float64),
            "minimum_ess_max": np.full(rows, 500.0, dtype=np.float64),
            "resampling_count_total": np.zeros(rows, dtype=np.int64),
            "resampling_count_min": np.zeros(rows, dtype=np.int64),
        }
    )
    synthetic_config = copy.deepcopy(config)
    synthetic_config["validation"]["expected_rows"] = rows
    synthetic_config["validation"]["expected_wells"] = rows
    synthetic_config["validation"]["primary_control_rmse_ft"] = 1.0
    synthetic_config["validation"][
        "fixed_hmm_pf_50_50_control_rmse_ft"
    ] = 0.5
    synthetic_config["data"][
        "expected_raw_well_identity_sha256"
    ] = "a" * 64
    frozen = {"sha_readback": {"passed": True}}
    ledger_at_freeze = {
        "prediction_frozen": True,
        "before_freeze": {
            "truth_rows": 0,
            "control_rows": 0,
            "fold_rows": 0,
            "hidden_like_rows": 0,
        },
    }
    gate = train.evaluate_stage1_gate(
        synthetic_config,
        frame,
        audit,
        frozen,
        primary,
        by_well,
        blend,
        ledger_at_freeze,
        {"logical_sha256": "a" * 64},
        runtime_seconds=100.0,
        rss_gb=1.0,
    )
    assert gate["passed"] is True
    assert gate["technical_gate"]["passed"] is True
    assert gate["primary_scientific_gate"]["passed"] is True
    assert gate["primary_scientific_gate"]["improved_folds"] == 5
    assert gate["fixed_exp209_hmm_pf_50_50_guard"]["passed"] is True


def test_saved_float32_restoration_preserves_artifact_semantics(
    train: ModuleType,
) -> None:
    serialized = pd.Series(["11183.766", "11022.869", "12161.080"])
    restored = train.restore_frozen_float32_column(
        serialized,
        label="synthetic saved exp404",
    )
    expected = np.asarray(
        [11183.766, 11022.869, 12161.080],
        dtype=np.float32,
    )
    assert restored.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(restored.to_numpy(), expected)
    assert np.array_equal(
        restored.to_numpy().view(np.uint32),
        expected.view(np.uint32),
    )


def test_inference_guard_and_notebook_sources_are_fail_closed(
    inference: ModuleType,
) -> None:
    guard = inference.validate_inference_is_disabled(inference.CONFIG)
    assert guard["status"] == "inference_disabled_pending_separate_approval"
    assert guard["submission_created"] is False
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "sample_submission" not in inference_source
    assert '.to_csv("submission.csv"' not in inference_source
    assert ".to_csv('submission.csv'" not in inference_source
    assert "def run_stage1(" in train_source
    assert "All-well Stage 1 truth-late CV and promotion gate" in train_source
    assert COMPACT_TRAIN.exists()
    assert CANONICAL_TRAIN.exists()
    assert COMPACT_TRAIN.read_bytes() == CANONICAL_TRAIN.read_bytes()


def test_stage1_kaggle_package_is_cpu_private_and_truth_late(
    config: dict,
) -> None:
    packaged_config_path = KAGGLE_TRAIN_DIR / "config.yaml"
    packaged_source_path = KAGGLE_TRAIN_DIR / f"{EXP}_compact_selfcontained_train.py"
    require_saved_files(KAGGLE_METADATA, packaged_config_path, packaged_source_path)
    metadata = json.loads(KAGGLE_METADATA.read_text())
    packaged_config = yaml.safe_load(packaged_config_path.read_text())
    assert metadata["id"] == (
        "kentookumura/exp484-student-t-gr-filtering-likelihood-pf-train"
    )
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["run_on_push"] is True
    assert metadata["dataset_sources"] == [
        "kentookumura/exp404-v1-frozen-predictions"
    ]
    assert metadata["kernel_sources"] == [
        "kentookumura/exp209-joint-exact-parity-train",
        "kentookumura/exp226-k16-kappa-repro-train",
        "kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train",
    ]
    assert packaged_config["model"] == config["model"]
    assert packaged_config["guards"] == config["guards"]
    assert packaged_config["execution"]["run_stage_0"] is False
    assert packaged_config["execution"]["run_stage_1"] is True
    assert packaged_config["execution"]["run_inference"] is False
    assert packaged_config["execution"]["create_submission"] is False
    packaged_source = packaged_source_path.read_bytes()
    assert hashlib.sha256(packaged_source).hexdigest() == (
        "51ec23a7508932e6bc2350efa4ab87f17752ae4c5097dff635dd3f39899212c7"
    )

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
EXP = "exp486_exp226_geometry_residual_likelihood_pf"
EXP_DIR = ROOT / "experiments" / EXP
TRAIN_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
INFERENCE_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.py"
COMPACT_TRAIN = EXP_DIR / f"{EXP}_compact_selfcontained_train.ipynb"
COMPACT_INFERENCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.ipynb"
CANONICAL_TRAIN = EXP_DIR / f"{EXP}_train.ipynb"
CONFIG_PATH = EXP_DIR / "config.yaml"
METRICS_PATH = EXP_DIR / "metrics.json"
FIXED32_MANIFEST = (
    ROOT
    / "experiments"
    / "exp411_predictive_filtered_rate_innovation_destick"
    / "assets"
    / "stage0_fixed32_manifest.csv"
)
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
    previous = os.environ.get("EXP486_IMPORT_ONLY")
    os.environ["EXP486_IMPORT_ONLY"] = "1"
    try:
        return load_module(TRAIN_SOURCE, "exp486_train_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP486_IMPORT_ONLY", None)
        else:
            os.environ["EXP486_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def inference() -> ModuleType:
    previous = os.environ.get("EXP486_IMPORT_ONLY")
    os.environ["EXP486_IMPORT_ONLY"] = "1"
    try:
        return load_module(INFERENCE_SOURCE, "exp486_inference_guard")
    finally:
        if previous is None:
            os.environ.pop("EXP486_IMPORT_ONLY", None)
        else:
            os.environ["EXP486_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def exp404() -> ModuleType:
    previous = os.environ.get("EXP404_IMPORT_ONLY")
    os.environ["EXP404_IMPORT_ONLY"] = "1"
    try:
        return load_module(EXP404_SOURCE, "exp404_parent_for_exp486")
    finally:
        if previous is None:
            os.environ.pop("EXP404_IMPORT_ONLY", None)
        else:
            os.environ["EXP404_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_horizontal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": np.arange(12, dtype=np.float64) + 1000.0,
            "Z": np.linspace(-8000.0, -7998.9, 12),
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
                55.0,
                54.0,
                52.0,
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
            ],
        }
    )


def synthetic_typewell() -> pd.DataFrame:
    tvt = np.linspace(90.0, 120.0, 151)
    return pd.DataFrame({"TVT": tvt, "GR": np.linspace(40.0, 70.0, 151)})


def synthetic_kernel_inputs() -> tuple[np.ndarray, ...]:
    md = np.arange(1.0, 8.0, dtype=np.float64)
    z = np.linspace(0.0, 0.6, len(md), dtype=np.float64)
    gr = np.asarray([50.0, 52.0, 54.0, 53.0, 51.0, 50.0, 49.0])
    geometry = np.linspace(100.0, 103.0, len(md))
    grid_gr = np.linspace(40.0, 70.0, 151)
    return md, z, gr, geometry, grid_gr


def test_stage1_is_complete_without_reclassifying_stage0(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)
    assert (
        config["experiment"]["status"]
        == "stage1_all_variants_gate_failed_terminal_close"
    )
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["implementation_approval_received"] is True
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert config["implementation"]["canonical_inference_notebook_adopted"] is False
    assert config["implementation"]["kaggle_package_created"] is True
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["stage_0_execution_approved"] is True
    assert config["execution"]["stage_1_execution_approved"] is True
    assert config["execution"]["selected_stage"] is None
    assert config["execution"]["run_stage_0"] is False
    assert config["execution"]["run_stage_1"] is False
    assert config["execution"]["run_inference"] is False
    assert config["execution"]["create_submission"] is False
    assert config["execution"]["stage_0_completed"] is True
    assert config["execution"]["stage_0_all_gates_pass"] is False
    assert config["stage_0_result"]["all_pass"] is False
    assert config["stage_0_result"]["runtime_exception"]["approved"] is True
    assert config["stage_0_result"]["support_bound_numerical_exception"]["accepted"] is True
    assert config["execution"]["stage_1_eligible_pending_separate_user_approval"] is False
    assert config["execution"]["stage_1_completed"] is True
    assert config["execution"]["stage_1_gate_passed"] is False
    assert config["execution"]["stage_1_eligible_variants"] == []
    assert config["execution"]["stage_1_eligible_under_approved_exceptions"] is False
    assert config["execution"]["stage_1_resume_from_frozen_v2"] is True
    assert (
        config["data"]["exp209_hmm_control"]["expected_decompressed_sha256"]
        == "8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5"
    )
    assert (
        config["data"]["stage1_frozen_resume"]["dataset_source"]
        == "kentookumura/exp486-v2-stage1-frozen-targetfree"
    )
    assert train.selected_stage(config) is None
    assert contract["active_variants"] == [
        "absolute_geometry_unary_sigma20_lambda050",
        "slow_residual_offset_state",
    ]
    assert contract["geometry_allowlist"] == [
        "well_id",
        "row_idx",
        "suffix_offset",
        "tvt_geop",
    ]
    assert contract["absolute_geometry_unary"]["sigma_ft"] == 20.0
    assert contract["absolute_geometry_unary"]["lambda"] == 0.5
    assert contract["residual_offset_state"]["initial_offset_rate_center"] == 0.0
    assert contract["pf"]["particles"] == 500
    assert contract["pf"]["seeds"] == 128
    assert contract["pf"]["temperature"] == 5.0
    assert contract["saved_control_rerun"] is False
    assert len(contract["scientific_contract_sha256"]) == 64
    train.validate_scientific_contract(config, require_run_approval=True)

    broken = copy.deepcopy(config)
    broken["experiment"]["status"] = "stage1_resume_approved_pending_kaggle"
    broken["execution"]["selected_stage"] = "stage_1"
    broken["execution"]["run_stage_1"] = True
    broken["execution"]["stage_1_completed"] = False
    broken["execution"]["stage_1_execution_approved"] = False
    with pytest.raises(RuntimeError, match="Stage 1 Kaggle execution is not approved"):
        train.validate_scientific_contract(broken, require_run_approval=True)


def test_stage1_recorded_metrics_match_terminal_gate(config: dict) -> None:
    metrics = json.loads(METRICS_PATH.read_text())
    assert metrics["status"] == "stage1_all_variants_gate_failed_terminal_close"
    assert metrics["cv"] is None
    assert metrics["eligible_variants"] == []
    assert metrics["stage1_result"]["technical_gate_passed"] is True
    assert metrics["stage1_result"]["scientific_gate_passed"] is False
    assert metrics["variant_cv"] == {
        "absolute_geometry_unary_sigma20_lambda050": 9.7269380294375,
        "slow_residual_offset_state": 11.139812021086678,
    }
    assert metrics["prediction_sha256"] == (
        config["stage_1_result"]["reproducibility"]["prediction_logical_sha256"]
    )


def test_execution_counts_are_exact_and_zero_control(
    train: ModuleType,
    config: dict,
) -> None:
    counts = train.validate_execution_contract(config)
    assert counts == {
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


def test_contract_rejects_parameter_rescue_and_unsafe_columns(
    train: ModuleType,
    config: dict,
) -> None:
    broken = copy.deepcopy(config)
    broken["model"]["absolute_geometry_unary"]["lambda"] = 0.6
    with pytest.raises(ValueError, match="lambda"):
        train.validate_scientific_contract(broken)

    broken = copy.deepcopy(config)
    broken["model"]["fixed_from_exp404_for_both"]["particles"] = 1000
    with pytest.raises(ValueError, match="particles"):
        train.validate_scientific_contract(broken)

    broken = copy.deepcopy(config)
    broken["data"]["exp226_oof_geometry"]["prediction_time_usecols"].append("tvt_pred")
    with pytest.raises(ValueError, match="allowlist"):
        train.validate_scientific_contract(broken)


def test_geometry_loader_reads_only_prediction_time_allowlist(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "well_id": ["w", "w"],
            "row_idx": [6, 7],
            "suffix_offset": [0, 1],
            "tvt_geop": [103.0, 103.5],
            "fold": [4, 4],
            "tvt_pred": [999.0, 999.0],
            "gr_delta": [2.0, 2.0],
            "tvt_true": [500.0, 501.0],
            "error": [400.0, 400.0],
            "abs_error": [400.0, 400.0],
        }
    )
    path = tmp_path / "geometry.csv.gz"
    source.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    ledger = train.LeakageLedger(expected_variant_wells=2)
    observed = train.load_fold_safe_geometry(
        path,
        config,
        wells={"w"},
        ledger=ledger,
    )
    assert observed.columns.tolist() == list(train.GEOMETRY_ALLOWLIST)
    assert not {
        "fold",
        "tvt_pred",
        "gr_delta",
        "tvt_true",
        "error",
        "abs_error",
    }.intersection(observed.columns)
    assert ledger.geometry_safe_rows_before_freeze == 2
    assert ledger.forbidden_geometry_columns_read_before_freeze == 0


def test_geometry_alignment_requires_exact_suffix_identity(train: ModuleType) -> None:
    prepared = {
        "eval_indices": np.asarray([6, 7, 8], dtype=np.int64),
    }
    geometry = pd.DataFrame(
        {
            "well_id": ["w"] * 3,
            "row_idx": [6, 7, 8],
            "suffix_offset": [0, 1, 2],
            "tvt_geop": [103.0, 103.5, 104.0],
        }
    )
    np.testing.assert_array_equal(
        train.align_geometry_to_prepared("w", geometry, prepared),
        [103.0, 103.5, 104.0],
    )
    broken = geometry.copy()
    broken.loc[2, "suffix_offset"] = 3
    with pytest.raises(ValueError, match="identity"):
        train.align_geometry_to_prepared("w", broken, prepared)


def test_pf_input_preparation_matches_exp404(
    train: ModuleType,
    exp404: ModuleType,
) -> None:
    horizontal = synthetic_horizontal()
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
    assert observed["scale_audit"]["candidate_scale"] == expected["scale_audit"]["candidate_scale"]


def test_absolute_lambda_zero_is_bitwise_exp404_parity(
    train: ModuleType,
    exp404: ModuleType,
) -> None:
    md, z, gr, geometry, grid_gr = synthetic_kernel_inputs()
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
    observed = train._pf_absolute_geometry_allseeds(
        md,
        z,
        gr,
        geometry,
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
        20.0,
        0.0,
    )
    for index in range(5):
        assert np.array_equal(observed[index], expected[index])


def test_absolute_unary_is_active_and_finite(train: ModuleType) -> None:
    md, z, gr, geometry, grid_gr = synthetic_kernel_inputs()
    baseline = train._pf_absolute_geometry_allseeds(
        md,
        z,
        gr,
        geometry,
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
        20.0,
        0.0,
    )
    candidate = train._pf_absolute_geometry_allseeds(
        md,
        z,
        gr,
        geometry + 20.0,
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
        20.0,
        0.5,
    )
    assert np.isfinite(candidate[0]).all()
    assert np.isfinite(candidate[1]).all()
    assert (candidate[7] <= 0.0).all()
    assert (candidate[7] < 0.0).any()
    assert not np.array_equal(candidate[0], baseline[0])


def test_residual_offset_transition_and_output_formula(train: ModuleType) -> None:
    contract = train.residual_transition_contract()
    assert contract["pass"] is True
    md = np.asarray([1.0, 3.0, 4.0])
    gr = np.asarray([50.0, 50.0, 50.0])
    geometry = np.asarray([100.0, 101.0, 103.0])
    grid_gr = np.full(201, 50.0)
    output = train._pf_residual_offset_allseeds(
        md,
        gr,
        geometry,
        grid_gr,
        0.0,
        1.0,
        20.0,
        2.0,
        0.1,
        1,
        1,
        123,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    expected_offset = np.asarray([2.1, 2.3, 2.4])
    np.testing.assert_allclose(output[5][0], expected_offset, atol=1e-12)
    np.testing.assert_allclose(
        output[0][0],
        geometry + expected_offset,
        atol=1e-12,
    )
    assert np.array_equal(output[2], [0])
    assert np.array_equal(output[4], [0])


def test_common_seed_label_excludes_variant(train: ModuleType) -> None:
    key = "likpf::train::well-a"
    expected = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    expected = expected % 2_147_483_647 + 1
    absolute = train.stable_seed("likpf", "train", "well-a")
    residual = train.stable_seed("likpf", "train", "well-a")
    assert absolute == residual == expected
    assert train.stable_seed("likpf", "train", "well-b") != expected
    assert "absolute" not in key and "residual" not in key


def test_fixed32_manifest_is_pinned_and_prefreeze_well_only(
    train: ModuleType,
    config: dict,
) -> None:
    observed_sha = hashlib.sha256(FIXED32_MANIFEST.read_bytes()).hexdigest()
    assert observed_sha == config["data"]["fixed32_manifest"]["expected_sha256"]
    wells, report = train.load_fixed32_scope(config)
    assert len(wells) == 32
    assert len(set(wells)) == 32
    assert report["columns_read_before_freeze"] == ["well"]
    manifest = pd.read_csv(FIXED32_MANIFEST)
    assert int(manifest["suffix_rows"].sum()) == 156088


def test_leakage_ledger_requires_both_variants_for_every_well(
    train: ModuleType,
) -> None:
    ledger = train.LeakageLedger(expected_variant_wells=4)
    ledger.record_geometry_safe(10)
    ledger.freeze(train.ABSOLUTE_VARIANT, "a")
    ledger.freeze(train.RESIDUAL_VARIANT, "a")
    ledger.freeze(train.ABSOLUTE_VARIANT, "b")
    with pytest.raises(RuntimeError, match="before both"):
        ledger.record_truth(2)
    ledger.freeze(train.RESIDUAL_VARIANT, "b")
    ledger.record_control(4)
    ledger.record_role_fold(2)
    report = ledger.report()
    assert report["all_frozen"] is True
    assert report["before_freeze"]["truth_rows"] == 2
    assert report["before_freeze"]["forbidden_geometry_columns"] == 0
    assert report["after_freeze"]["control_rows"] == 4


def test_stage1_support_tolerance_only_accepts_numerical_overshoot(
    train: ModuleType,
) -> None:
    numerical = pd.DataFrame(
        {
            "typewell_support_fraction": [
                -5.0e-13,
                0.5,
                1.0 + 5.0e-13,
            ]
        }
    )
    assert train.residual_support_within_tolerance(numerical, 1.0e-12) is True
    scientific_failure = numerical.copy()
    scientific_failure.loc[2, "typewell_support_fraction"] = 1.0 + 2.0e-12
    assert (
        train.residual_support_within_tolerance(
            scientific_failure,
            1.0e-12,
        )
        is False
    )


def test_streaming_dataframe_sha_matches_single_payload(train: ModuleType) -> None:
    frame = pd.DataFrame(
        {
            "id": [f"id-{index}" for index in range(17)],
            "value": np.linspace(-1.5, 2.5, 17),
            "label": ["a", "b"] * 8 + ["c"],
        }
    )
    columns = ["id", "value", "label"]
    expected = hashlib.sha256(
        frame[columns].to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()
    assert train.dataframe_content_sha(frame, columns) == expected


def test_stage1_reports_both_variants_independently(train: ModuleType) -> None:
    rows = 10
    truth = np.linspace(100.0, 109.0, rows)
    frame = pd.DataFrame(
        {
            "well_id": ["a"] * 5 + ["b"] * 5,
            "true_tvt": truth,
            train.PRIMARY_CONTROL: truth + 1.0,
            train.ABSOLUTE_PREDICTION: truth + 0.5,
            train.RESIDUAL_PREDICTION: truth + 1.5,
            "saved_exp209_hmm": truth + 0.25,
            f"{train.ABSOLUTE_PREDICTION}__hmm_50_50": truth + 0.375,
            f"{train.RESIDUAL_PREDICTION}__hmm_50_50": truth + 0.875,
            "saved_control_hmm_50_50": truth + 0.625,
            "fold": np.arange(rows) % 5,
            "raw_gr_observed": [True, False] * 5,
            "well_missing_fraction": [0.4] * rows,
            "md_since": np.linspace(1000.0, 2000.0, rows),
            "hidden_like_spatial": [True] * rows,
            "hidden_like_typewell_purged": [True] * rows,
        }
    )
    primary, by_well, blend, reference = train.build_stage1_metric_outputs(
        frame.assign(tvt_geop=truth + 0.75)
    )
    overall = primary.loc[primary["scope"].eq("overall")]
    assert overall["variant"].tolist() == list(train.ACTIVE_VARIANTS)
    assert dict(zip(overall["variant"], overall["candidate_rmse"], strict=True)) == {
        train.ABSOLUTE_VARIANT: 0.5,
        train.RESIDUAL_VARIANT: 1.5,
    }
    assert len(by_well) == 4
    assert set(blend["variant"]) == set(train.ACTIVE_VARIANTS)
    assert set(reference["variant"]) == {"exp226_tvt_geop_reference"}


def test_inference_is_fail_closed(
    inference: ModuleType,
    config: dict,
) -> None:
    status = inference.validate_inference_is_disabled(config)
    assert status == {
        "implementation_scope": "train_side_stage0_and_stage1_two_variant_only",
        "canonical_inference_notebook_adopted": False,
        "inference_enabled": False,
        "run_inference": False,
        "create_submission": False,
        "submit_to_kaggle": False,
        "test_geometry_regeneration_implemented": False,
    }
    with pytest.raises(RuntimeError, match="not implemented or approved"):
        inference.run_inference(config)


def test_compact_sources_are_self_contained_and_canonical_train_is_adopted(
    config: dict,
) -> None:
    train_source = TRAIN_SOURCE.read_text()
    for heading in (
        "Exp404 likelihood-PF input preparation",
        "Absolute-unary and residual-offset PF kernels",
        "Target-free two-variant generation and freeze",
        "Truth-late fixed32 readout",
        "Generated artifacts",
    ):
        assert heading in train_source
    assert "Path(__file__)" not in train_source
    assert "from settings import" not in train_source
    assert "def _pf_absolute_geometry_allseeds(" in train_source
    assert "def _pf_residual_offset_allseeds(" in train_source
    assert "def run_stage0(" in train_source
    assert "def run_stage1(" in train_source
    assert COMPACT_TRAIN.exists()
    assert COMPACT_INFERENCE.exists()
    compact = json.loads(COMPACT_TRAIN.read_text())
    compact_text = "\n".join("".join(cell.get("source", [])) for cell in compact["cells"])
    assert "def _pf_absolute_geometry_allseeds(" in compact_text
    assert "def _pf_residual_offset_allseeds(" in compact_text
    assert "def run_stage0(" in compact_text
    assert "def run_stage1(" in compact_text
    assert len(compact["cells"]) >= 20
    canonical = json.loads(CANONICAL_TRAIN.read_text())
    canonical_text = "\n".join("".join(cell.get("source", [])) for cell in canonical["cells"])
    assert "def _pf_absolute_geometry_allseeds(" in canonical_text
    assert "def _pf_residual_offset_allseeds(" in canonical_text
    assert "def run_stage0(" in canonical_text
    assert "def run_stage1(" in canonical_text
    assert len(canonical["cells"]) >= 20
    assert config["implementation"]["canonical_train_notebook_adopted"] is True

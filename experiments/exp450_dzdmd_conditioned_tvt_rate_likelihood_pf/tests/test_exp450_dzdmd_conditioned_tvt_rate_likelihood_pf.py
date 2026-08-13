from __future__ import annotations

import copy
import hashlib
import importlib.util
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
EXP = "exp450_dzdmd_conditioned_tvt_rate_likelihood_pf"
EXP_DIR = ROOT / "experiments" / EXP
TRAIN_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
INFERENCE_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.py"
CONFIG_PATH = EXP_DIR / "config.yaml"
EXP404_SOURCE = (
    ROOT
    / "experiments"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_train.py"
)
SENTINEL_ASSET = (
    ROOT
    / "experiments"
    / "exp410_likpf_particle_resampling_basin_audit"
    / "assets"
    / "pf_counterfactual_sentinel_wells.csv"
)
FIXED32_ASSET = (
    ROOT
    / "experiments"
    / "exp411_predictive_filtered_rate_innovation_destick"
    / "assets"
    / "stage0_fixed32_manifest.csv"
)
EPISODE_ASSET = (
    ROOT
    / "experiments"
    / "exp408_hmm_message_rate_basin_audit"
    / "assets"
    / "persistent_offset_episodes.csv"
)
CAUSE_ASSET = (
    ROOT
    / "experiments"
    / "exp408_hmm_message_rate_basin_audit"
    / "artifacts"
    / "kaggle_v3"
    / "exp408_hmm_message_rate_basin_audit_episode_summary.csv"
)
V1_PARITY_REPORT = (
    EXP_DIR / "artifacts" / "kaggle_v1" / "artifacts" / f"{EXP}_stage0a_parity_report.csv"
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
    previous = os.environ.get("EXP450_IMPORT_ONLY")
    os.environ["EXP450_IMPORT_ONLY"] = "1"
    try:
        return load_module(TRAIN_SOURCE, "exp450_train_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP450_IMPORT_ONLY", None)
        else:
            os.environ["EXP450_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def inference() -> ModuleType:
    previous = os.environ.get("EXP450_IMPORT_ONLY")
    os.environ["EXP450_IMPORT_ONLY"] = "1"
    try:
        return load_module(INFERENCE_SOURCE, "exp450_inference_guard")
    finally:
        if previous is None:
            os.environ.pop("EXP450_IMPORT_ONLY", None)
        else:
            os.environ["EXP450_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def exp404() -> ModuleType:
    previous = os.environ.get("EXP404_IMPORT_ONLY")
    os.environ["EXP404_IMPORT_ONLY"] = "1"
    try:
        return load_module(EXP404_SOURCE, "exp404_parent_for_exp450")
    finally:
        if previous is None:
            os.environ.pop("EXP404_IMPORT_ONLY", None)
        else:
            os.environ["EXP404_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_horizontal(
    *,
    known_rows: int = 40,
    suffix_rows: int = 8,
    beta: float = -0.8,
    intercept: float = 0.03,
) -> pd.DataFrame:
    rows = known_rows + suffix_rows
    md_steps = 1.0 + 0.2 * (np.arange(rows) % 3)
    md = 1000.0 + np.cumsum(md_steps)
    g = 0.15 * np.sin(np.arange(rows - 1) / 4.0) - 0.04
    z = np.empty(rows, dtype=np.float64)
    z[0] = -8000.0
    z[1:] = z[0] + np.cumsum(g * np.diff(md))
    q = beta * g + intercept
    tvt = np.empty(rows, dtype=np.float64)
    tvt[0] = 10000.0
    tvt[1:] = tvt[0] + np.cumsum(q * np.diff(md))
    tvt_input = tvt.copy()
    tvt_input[known_rows:] = np.nan
    gr = 65.0 + 8.0 * np.sin(np.arange(rows) / 5.0)
    return pd.DataFrame(
        {
            "MD": md,
            "Z": z,
            "GR": gr,
            "TVT_input": tvt_input,
        }
    )


def synthetic_typewell() -> pd.DataFrame:
    tvt = np.linspace(9900.0, 10100.0, 1001)
    return pd.DataFrame(
        {
            "TVT": tvt,
            "GR": 65.0 + 8.0 * np.sin((tvt - 10000.0) / 5.0),
        }
    )


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_contract_and_execution_counts(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)
    assert contract["primary_control"] == "likpf_scale_5_x1p0"
    assert contract["primary_candidate"] == ("likpf_scale5_dzdmd_conditioned_tvt_rate")
    assert contract["active_scientific_variants"] == ["learned_prefix_affine_residual_ar"]
    assert contract["stages"]["stage_0a"]["total_pf_well_runs"] == 24
    assert contract["stages"]["stage_0a"]["seed_well_trajectories"] == 3072
    assert contract["stages"]["stage_0b"]["candidate_pf_well_runs"] == 32
    assert contract["stages"]["stage_0b"]["control_pf_well_runs"] == 0
    assert contract["stages"]["stage_1"]["candidate_pf_well_runs"] == 773
    assert contract["model_count"] == 0
    assert contract["booster_count"] == 0
    assert contract["gpu_count"] == 0
    assert len(contract["scientific_contract_sha256"]) == 64
    with pytest.raises(
        RuntimeError,
        match="no approved Stage 0A/0B selection",
    ):
        train.validate_scientific_contract(
            config,
            require_run_approval=True,
        )


def test_contract_rejects_grid_rescue_and_changed_pf_setting(
    train: ModuleType,
    config: dict,
) -> None:
    broken = copy.deepcopy(config)
    broken["model"]["prefix_affine_fit"]["coefficient_clip"] = [-2.0, 0.0]
    with pytest.raises(ValueError, match="coefficient_clip"):
        train.validate_scientific_contract(broken)

    broken = copy.deepcopy(config)
    broken["model"]["fixed_from_exp404"]["particles"] = 1000
    with pytest.raises(ValueError, match="particles"):
        train.validate_scientific_contract(broken)

    broken = copy.deepcopy(config)
    broken["model"]["active_scientific_variants"].append("exact_transform_rescue")
    with pytest.raises(ValueError, match="active_scientific_variants"):
        train.validate_scientific_contract(broken)


def test_prefix_affine_recovers_visible_prefix_relation(
    train: ModuleType,
) -> None:
    horizontal = synthetic_horizontal(beta=-0.73, intercept=0.017)
    fit = train.fit_prefix_affine(horizontal)
    assert fit.valid_step_count == 39
    assert not fit.fallback_used
    assert fit.beta == pytest.approx(-0.73, abs=1e-10)
    assert fit.intercept == pytest.approx(0.017, abs=1e-10)
    assert fit.fitted_residual_sse < 1e-18

    changed_suffix = horizontal.copy()
    changed_suffix.loc[changed_suffix["TVT_input"].isna(), "TVT_input"] = np.nan
    changed_suffix["suffix_truth_decoy"] = np.linspace(-1e9, 1e9, len(changed_suffix))
    changed_fit = train.fit_prefix_affine(changed_suffix)
    assert changed_fit == fit


def test_prefix_affine_filters_invalid_steps_and_falls_back(
    train: ModuleType,
) -> None:
    horizontal = synthetic_horizontal(known_rows=9)
    horizontal.loc[2, "MD"] = horizontal.loc[1, "MD"]
    horizontal.loc[4, "Z"] = np.nan
    fit = train.fit_prefix_affine(horizontal)
    assert fit.valid_step_count < 10
    assert fit.fallback_used
    assert fit.beta == -1.0
    assert fit.intercept == 0.0


def test_prefix_tail_backtest_is_target_free_and_fixed(
    train: ModuleType,
) -> None:
    horizontal = synthetic_horizontal(
        known_rows=45,
        beta=-0.61,
        intercept=0.024,
    )
    report = train.prefix_tail_backtest(horizontal)
    assert report["eligible"]
    assert report["fit_steps"] == 24
    assert report["holdout_steps"] == 20
    assert report["candidate_sse"] < 1e-18
    assert report["candidate_sse"] < report["exact_sse"]

    ineligible = train.prefix_tail_backtest(synthetic_horizontal(known_rows=25))
    assert not ineligible["eligible"]


def test_first_suffix_delta_g_mu_and_initial_rate_contract(
    train: ModuleType,
) -> None:
    horizontal = synthetic_horizontal(
        known_rows=40,
        beta=-0.8,
        intercept=0.03,
    )
    fit = train.fit_prefix_affine(horizontal)
    prepared = train.prepare_pf_inputs(
        horizontal,
        synthetic_typewell(),
        fit,
    )
    first_eval = int(prepared["eval_indices"][0])
    expected_delta_md = max(
        float(horizontal.loc[first_eval, "MD"]) - float(horizontal.loc[first_eval - 1, "MD"]),
        1.0,
    )
    expected_g = (
        float(horizontal.loc[first_eval, "Z"]) - float(horizontal.loc[first_eval - 1, "Z"])
    ) / expected_delta_md
    assert prepared["eval_delta_md"][0] == pytest.approx(expected_delta_md)
    assert prepared["eval_g"][0] == pytest.approx(expected_g)
    assert prepared["eval_mu"][0] == pytest.approx(fit.beta * expected_g + fit.intercept)
    assert prepared["previous_mu"] == pytest.approx(fit.beta * fit.previous_g + fit.intercept)
    assert prepared["initial_q_rate"] == pytest.approx(prepared["initial_u_rate"] - fit.previous_g)


def test_residual_ar_transition_formula_without_noise(
    train: ModuleType,
) -> None:
    delta_md = np.asarray([1.0, 2.0, 1.5], dtype=np.float64)
    mu = np.asarray([-0.1, 0.2, 0.05], dtype=np.float64)
    gr = np.asarray([50.0, 50.0, 50.0], dtype=np.float64)
    grid_gr = np.full(100, 50.0, dtype=np.float64)
    initial_q = 0.3
    previous_mu = -0.2
    predictions, loglik, resamples, _, clips = train._pf_residual_ar_allseeds(
        delta_md,
        mu,
        gr,
        grid_gr,
        0.0,
        1.0,
        20.0,
        10.0,
        initial_q,
        0.0,
        previous_mu,
        1,
        1,
        123,
        0.998,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    expected = []
    tvt = 10.0
    q = initial_q
    previous = previous_mu
    for step, center in zip(delta_md, mu, strict=True):
        q = center + 0.998 * (q - previous)
        tvt += q * step
        expected.append(tvt)
        previous = center
    np.testing.assert_allclose(predictions[0], expected, rtol=0.0, atol=1e-12)
    assert loglik[0] == pytest.approx(0.0)
    assert int(resamples.sum()) == 0
    assert int(clips.sum()) == 0


def test_exact_coordinate_kernel_has_paired_rng_parity(
    train: ModuleType,
) -> None:
    horizontal = synthetic_horizontal(known_rows=40, suffix_rows=10)
    exact_fit = train.PrefixAffineFit(
        valid_step_count=0,
        fallback_used=True,
        beta=-1.0,
        intercept=0.0,
        previous_g=train.fit_prefix_affine(horizontal).previous_g,
        fitted_residual_sse=np.nan,
        g_min=np.nan,
        g_max=np.nan,
        g_mean=np.nan,
        g_std=np.nan,
        q_min=np.nan,
        q_max=np.nan,
        q_mean=np.nan,
        q_std=np.nan,
    )
    prepared = train.prepare_pf_inputs(
        horizontal,
        synthetic_typewell(),
        exact_fit,
    )
    report, parent, exact = train.paired_coordinate_parity(
        prepared,
        particles=32,
        seeds=4,
        seed_base=123456,
    )
    assert report["finite"]
    assert report["resampling_decision_mismatches"] == 0
    assert report["clip_decision_mismatches"] == 0
    assert report["maximum_seed_prediction_abs_diff"] <= 1e-10
    assert report["maximum_particle_weight_abs_diff"] <= 1e-10
    assert report["maximum_log_likelihood_abs_diff"] <= 1e-10
    assert report["maximum_temperature5_prediction_abs_diff"] <= 1e-10
    np.testing.assert_allclose(parent, exact, rtol=0.0, atol=1e-10)


def test_paired_parent_path_matches_exp404_kernel(
    train: ModuleType,
    exp404: ModuleType,
) -> None:
    horizontal = synthetic_horizontal(known_rows=40, suffix_rows=10)
    first_suffix = int(horizontal["TVT_input"].isna().to_numpy().nonzero()[0][0])
    gap = float(horizontal.loc[first_suffix, "MD"]) - float(horizontal.loc[first_suffix - 1, "MD"])
    horizontal.loc[first_suffix:, "MD"] -= gap - 1.0
    prefix = train.fit_prefix_affine(horizontal)
    exact_fit = train.PrefixAffineFit(
        valid_step_count=0,
        fallback_used=True,
        beta=-1.0,
        intercept=0.0,
        previous_g=prefix.previous_g,
        fitted_residual_sse=np.nan,
        g_min=np.nan,
        g_max=np.nan,
        g_mean=np.nan,
        g_std=np.nan,
        q_min=np.nan,
        q_max=np.nan,
        q_mean=np.nan,
        q_std=np.nan,
    )
    prepared = train.prepare_pf_inputs(
        horizontal,
        synthetic_typewell(),
        exact_fit,
    )
    seed_base = 246810
    paired = train._paired_parent_exact_allseeds(
        prepared["eval_delta_md"],
        prepared["eval_z"],
        prepared["eval_g"],
        prepared["eval_gr"],
        prepared["grid_gr"],
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["gr_scale"]["base_scale"]),
        float(prepared["last_known_u"]),
        float(prepared["last_known_tvt"]),
        float(prepared["initial_u_rate"]),
        float(prepared["previous_g"]),
        32,
        4,
        seed_base,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    parent = exp404._pf_lik_allseeds(
        prepared["eval_md"],
        prepared["eval_z"],
        prepared["eval_gr"],
        prepared["grid_gr"],
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["gr_scale"]["base_scale"]),
        float(prepared["last_known_u"]),
        float(prepared["initial_u_rate"]),
        32,
        4,
        seed_base,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    np.testing.assert_allclose(paired[0], parent[0], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(paired[2], parent[1], rtol=0.0, atol=1e-10)


def test_stable_seed_matches_exp404_policy(train: ModuleType) -> None:
    well = "abc123"
    key = f"likpf::train::{well}"
    expected = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % 2_147_483_647 + 1
    assert train.stable_seed("likpf", "train", well) == expected
    assert train.stable_seed("likpf", "train", well) == train.stable_seed("likpf", "train", well)
    assert train.stable_seed("likpf", "train", well) != train.stable_seed(
        "likpf", "train", "different"
    )


def test_truth_late_ledger_is_fail_closed(train: ModuleType) -> None:
    ledger = train.LeakageLedger()
    with pytest.raises(RuntimeError, match="truth-late input"):
        ledger.require_all_frozen()
    ledger.mark_prefix_fit_frozen()
    ledger.mark_candidate_frozen()
    with pytest.raises(RuntimeError, match="truth-late input"):
        ledger.require_all_frozen()
    ledger.mark_saved_control_frozen()
    ledger.require_all_frozen()
    assert not any(ledger.report()["before_freeze"].values())

    contaminated = train.LeakageLedger(suffix_truth_rows_before_freeze=1)
    with pytest.raises(RuntimeError, match="suffix truth"):
        contaminated.mark_prefix_fit_frozen()


def test_fixed_assets_have_preregistered_identity(config: dict) -> None:
    require_saved_files(SENTINEL_ASSET, FIXED32_ASSET, EPISODE_ASSET, CAUSE_ASSET)
    assert file_sha(SENTINEL_ASSET) == config["data"]["stage_0a_sentinel12"]["expected_sha256"]
    assert file_sha(FIXED32_ASSET) == config["data"]["stage_0b_fixed32"]["expected_sha256"]
    assert file_sha(EPISODE_ASSET) == config["data"]["persistent_episodes"]["expected_sha256"]
    assert file_sha(CAUSE_ASSET) == config["data"]["exp408_episode_causes"]["expected_sha256"]
    fixed32 = pd.read_csv(FIXED32_ASSET)
    assert fixed32["role"].value_counts().to_dict() == {
        "control": 16,
        "persistent": 16,
    }
    assert fixed32["fold"].value_counts().sort_index().to_dict() == {
        0: 8,
        1: 6,
        2: 6,
        3: 6,
        4: 6,
    }


def test_stage0a_gate_checks_exact_execution_counts(
    train: ModuleType,
    config: dict,
) -> None:
    rows = []
    predictions = []
    for index in range(12):
        well = f"w{index:02d}"
        rows.append(
            {
                "well": well,
                "parent_pf_well_runs": 1,
                "exact_transform_pf_well_runs": 1,
                "seed_well_trajectories": 256,
                "particle_starts": 128000,
                "finite": True,
                "maximum_seed_prediction_abs_diff": 0.0,
                "maximum_particle_weight_abs_diff": 0.0,
                "maximum_log_likelihood_abs_diff": 0.0,
                "maximum_temperature5_prediction_abs_diff": 0.0,
                "maximum_position_coordinate_abs_diff": 0.0,
                "maximum_rate_coordinate_abs_diff": 0.0,
                "resampling_decision_mismatches": 0,
                "clip_decision_mismatches": 0,
            }
        )
        predictions.append(
            {
                "well": well,
                "row_idx": 1,
                "parent_temperature5": 10.0,
                "exact_temperature5": 10.0,
            }
        )
    gate = train.evaluate_stage0a(
        pd.DataFrame(rows),
        pd.DataFrame(predictions),
        config,
    )
    assert gate["passed"]
    broken = pd.DataFrame(rows)
    broken.loc[0, "particle_starts"] -= 1
    assert not train.evaluate_stage0a(
        broken,
        pd.DataFrame(predictions),
        config,
    )["passed"]


def test_stage0a_v1_roundoff_is_diagnostic_when_aggregate_output_is_equivalent(
    train: ModuleType,
    config: dict,
) -> None:
    require_saved_files(V1_PARITY_REPORT)
    reports = pd.read_csv(V1_PARITY_REPORT, dtype={"well": str})
    predictions = pd.DataFrame(
        {
            "well": reports["well"],
            "row_idx": 0,
            "parent_temperature5": 10.0,
            "exact_temperature5": 10.0,
        }
    )
    gate = train.evaluate_stage0a(reports, predictions, config)
    assert gate["passed"]
    assert gate["checks"]["temperature5_prediction_parity"]
    assert not gate["diagnostic_checks_not_used_for_gate"]["seed_prediction_parity"]
    assert not gate["diagnostic_checks_not_used_for_gate"]["resampling_decision_parity"]
    assert gate["temperature5_tolerance_ft"] == 1.0e-6

    outside_tolerance = reports.copy()
    outside_tolerance.loc[0, "maximum_temperature5_prediction_abs_diff"] = 1.1e-6
    assert not train.evaluate_stage0a(
        outside_tolerance,
        predictions,
        config,
    )["passed"]


def test_deterministic_csv_roundtrip_preserves_logical_content(
    train: ModuleType,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "well": ["5f4d2a52", "206b6193"],
            "value": [4.836692824028432e-09, 21.17679085005875],
            "count": [57, 0],
            "finite": [True, True],
        }
    )
    plain_path = tmp_path / "roundtrip.csv"
    gzip_path = tmp_path / "roundtrip.csv.gz"
    train.write_deterministic_csv(frame, plain_path)
    train.write_deterministic_gzip_csv(frame, gzip_path)
    plain = pd.read_csv(plain_path, dtype={"well": str}, float_precision="round_trip")
    compressed = pd.read_csv(
        gzip_path,
        compression="gzip",
        dtype={"well": str},
        float_precision="round_trip",
    )
    expected_sha = train.dataframe_content_sha(frame)
    assert train.dataframe_content_sha(plain) == expected_sha
    assert train.dataframe_content_sha(compressed) == expected_sha


def test_exp404_source_logical_sha_uses_parent_typed_contract(
    train: ModuleType,
    exp404: ModuleType,
) -> None:
    frame = pd.DataFrame(
        {
            "id": pd.Series(["a", "b"], dtype=object),
            "well_id": pd.Series(["w1", "w2"], dtype=object),
            "row_idx": pd.Series([1, 2], dtype=np.int64),
            "likpf_scale_5_x1p0": pd.Series([1.25, 2.5], dtype=np.float64),
            "likpf_scale_5_x1p3": pd.Series([1.5, 2.75], dtype=np.float64),
            "likpf_mean_x1p0": pd.Series([1.75, 3.0], dtype=np.float64),
            "likpf_mean_x1p3": pd.Series([2.0, 3.25], dtype=np.float64),
        }
    )
    columns = frame.columns.tolist()
    assert train.dataframe_typed_content_sha(
        frame,
        columns,
    ) == exp404.dataframe_content_sha(frame, columns)


def test_inference_is_explicitly_unimplemented(
    inference: ModuleType,
) -> None:
    with pytest.raises(RuntimeError, match="not implemented or approved"):
        inference.run_inference()


def test_notebook_source_is_self_contained_and_notebook_safe() -> None:
    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert "## Contents" in train_text
    assert "## 5. Visible-prefix affine fit" in train_text
    assert "## 7. Parent/exact-transform paired parity kernel" in train_text
    assert "## 8. Learned residual-AR likelihood-PF kernel" in train_text
    assert "## 10. Truth-late mechanism readout" in train_text
    assert "def fit_prefix_affine" in train_text
    assert "def _paired_parent_exact_allseeds" in train_text
    assert "def _pf_residual_ar_allseeds" in train_text
    assert "def run_stage0ab" in train_text
    assert "__file__" not in train_text
    assert "from exp450" not in train_text
    assert "run_inference" in inference_text
    assert "__file__" not in inference_text


def test_canonical_train_is_adopted_and_inference_remains_guarded_scaffold(
    config: dict,
) -> None:
    assert not config["implementation"]["canonical_notebooks_are_markdown_placeholders"]
    assert config["implementation"]["canonical_train_notebook_adopted"]
    assert not config["implementation"]["canonical_inference_notebook_adopted"]
    canonical_train = EXP_DIR / f"{EXP}_train.ipynb"
    canonical_inference = EXP_DIR / f"{EXP}_inference.ipynb"
    assert canonical_train.exists()
    assert canonical_inference.exists()
    assert "fit_prefix_affine" in canonical_train.read_text()
    assert "compact_selfcontained" not in canonical_inference.read_text()

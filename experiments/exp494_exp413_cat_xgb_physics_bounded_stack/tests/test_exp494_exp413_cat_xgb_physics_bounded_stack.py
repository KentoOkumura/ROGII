from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP = "exp494_exp413_cat_xgb_physics_bounded_stack"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
INFERENCE_SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.py"


def load_source() -> ModuleType:
    previous = os.environ.get("EXP494_IMPORT_ONLY")
    os.environ["EXP494_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("exp494_train", SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP494_IMPORT_ONLY", None)
        else:
            os.environ["EXP494_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def train() -> ModuleType:
    return load_source()


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_static_contract_fixes_cost_family_and_parameters(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_static_contract(config)
    assert contract["families"] == ["lgb", "cat", "xgb", "physics"]
    assert contract["cost"] == {
        "active_variants": 2,
        "model_configs": 2,
        "outer_folds": 5,
        "new_gpu_models": 10,
        "parent_retraining": 0,
        "selector_retraining": 0,
        "new_physics_runs": 0,
    }
    assert config["model"]["catboost"]["params"]["random_seed"] == 7
    assert config["model"]["xgboost"]["params"]["random_state"] == 42
    assert config["physics"]["candidate_id"] == "exp226_w500_50_50"
    assert config["physics"]["source_experiment"] == (
        "exp413_scale5_likpf_full_replacement_on_exp335"
    )
    assert config["physics"]["semantic_slot_id"] == "likpf_mean"
    assert config["physics"]["semantic_value_source"] == "likpf_scale_5_x1p0"
    assert config["physics"]["saved_oof_rmse"] == pytest.approx(
        8.070218793924594,
        abs=1.0e-12,
    )
    assert config["physics"]["public_lb"] is None
    assert (
        config["physics"]["original_exp263_same_id_context"]["transferable_to_scale5_overlay"]
        is False
    )


def test_frozen_parent_config_pin_is_preserved(config: dict) -> None:
    parent = ROOT / "experiments/exp413_scale5_likpf_full_replacement_on_exp335/config.yaml"
    execution_config_sha = config["data"]["exp413"]["config"]["sha256"]
    assert len(execution_config_sha) == 64
    int(execution_config_sha, 16)
    current_parent_config = yaml.safe_load(parent.read_text())
    assert current_parent_config["lineage"]["steering"].startswith("../../docs/legacy/steering/")


def test_bootstrap_sources_exist_when_saved_artifacts_are_available(config: dict) -> None:
    dependencies = config["runtime"]["kaggle"]["bootstrap_dependency_files"]
    assert len(dependencies) == 21
    missing = [
        ROOT / item["source"] for item in dependencies if not (ROOT / item["source"]).is_file()
    ]
    if missing:
        pytest.skip(
            "requires Git-ignored bootstrap artifacts: "
            + ", ".join(str(path.relative_to(ROOT)) for path in missing)
        )
    for item in dependencies:
        assert (ROOT / item["source"]).is_file(), item


def test_reference_submission_override_opens_only_stage_6(
    train: ModuleType,
    config: dict,
) -> None:
    assert config["authorization"]["implementation_approved"] is True
    assert config["authorization"]["kaggle_train_run_approved"] is True
    train.require_train_run_authorization(config)
    assert all(
        config["execution"]["run_flags"][stage]
        for stage in (
            "stage_0_freeze",
            "stage_1_family_train",
            "stage_2_family_audit",
            "stage_3_physics_lock",
            "stage_4_bounded_stack",
            "stage_5_conditional_gate",
        )
    )
    assert config["execution"]["run_flags"]["stage_6_hidden_inference"] is True
    assert config["authorization"]["inference_run_approved"] is True
    assert config["authorization"]["submission_approved"] is True
    assert config["result"]["guard_passed"] is False
    assert config["inference"]["reference_submission_override"] is True
    assert config["inference"]["train_tail_guard_failed_acknowledged"] is True
    assert config["inference"]["selected_prediction"] == "constant_stack"
    assert config["inference"]["forbidden_additions"] == [
        "conditional_confidence_gate",
        "well_level_routing",
        "trajectory_postprocess",
        "weight_refit",
        "threshold_refit",
        "public_lb_adjustment",
    ]


def test_bounded_simplex_projection_is_feasible(train: ModuleType) -> None:
    lower = np.array([0.60, 0.00, 0.00, 0.00])
    upper = np.array([1.00, 0.25, 0.20, 0.20])
    projected = train.project_bounded_simplex(
        [0.30, 0.40, -0.10, 0.40],
        lower,
        upper,
    )
    assert projected.sum() == pytest.approx(1.0, abs=1.0e-10)
    assert np.all(projected >= lower)
    assert np.all(projected <= upper)
    assert projected[0] >= 0.60


def test_slsqp_stack_recovers_feasible_convex_signal(train: ModuleType) -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(500, 4))
    expected = np.array([0.65, 0.15, 0.10, 0.10])
    truth = matrix @ expected
    weights, report = train.solve_bounded_stack(
        matrix,
        truth,
        initial=[0.70, 0.10, 0.05, 0.15],
        lower=[0.60, 0.00, 0.00, 0.00],
        upper=[1.00, 0.25, 0.20, 0.20],
    )
    np.testing.assert_allclose(weights, expected, atol=1.0e-6, rtol=0)
    assert report["success"] is True
    assert report["constraint_residual"] <= 1.0e-8
    assert report["bound_residual"] <= 1.0e-8


def test_crossfit_stack_uses_five_meta_holdouts(
    train: ModuleType,
    config: dict,
) -> None:
    rng = np.random.default_rng(7)
    rows = 250
    folds = np.arange(rows) % 5
    predictions = {name: rng.normal(loc=100.0, scale=5.0, size=rows) for name in train.FAMILY_ORDER}
    truth = (
        0.65 * predictions["lgb"]
        + 0.15 * predictions["cat"]
        + 0.10 * predictions["xgb"]
        + 0.10 * predictions["physics"]
    )
    result = train.crossfit_bounded_stack(predictions, truth, folds, config)
    assert result["fold_weights"].shape == (5, 4)
    assert len(result["weight_rows"]) == 5
    assert np.isfinite(result["crossfit_prediction"]).all()
    np.testing.assert_allclose(result["fold_weights"].sum(axis=1), np.ones(5), atol=1.0e-8)
    assert np.all(result["fold_weights"][:, 0] >= 0.60)
    assert result["deployment_weights"].sum() == pytest.approx(1.0)


def test_streamed_and_materialized_matrix_sha_match(train: ModuleType) -> None:
    base = pd.DataFrame(
        {
            "id": [f"id_{index}" for index in range(6)],
            "well": ["a", "a", "b", "b", "c", "c"],
            "base_0": np.linspace(0.0, 1.0, 6).astype(np.float32),
            "base_1": np.linspace(2.0, 3.0, 6).astype(np.float32),
        }
    )
    positions = np.array([4, 1, 5, 0], dtype=np.int64)
    compact = pd.DataFrame({"compact_0": np.arange(4, dtype=np.float32)})
    signed = pd.DataFrame({"signed_0": -np.arange(4, dtype=np.float32)})
    matrix = train.assemble_matrix(
        base=base,
        positions=positions,
        compact=compact,
        signed=signed,
        base_features=["base_0", "base_1"],
        compact_features=["compact_0"],
        signed_features=["signed_0"],
    )
    materialized = train.feature_matrix_sha256(
        matrix,
        ["base_0", "base_1", "compact_0", "signed_0"],
    )
    streamed = train.stream_matrix_sha256(
        base=base,
        positions=positions,
        compact=compact,
        signed=signed,
        base_features=["base_0", "base_1"],
        compact_features=["compact_0"],
        signed_features=["signed_0"],
        row_chunk=2,
    )
    assert streamed == materialized


def test_disk_backed_base_cache_preserves_exact_matrix(
    train: ModuleType,
    tmp_path: Path,
) -> None:
    base = pd.DataFrame(
        {
            "base_0": np.linspace(0.0, 1.0, 6).astype(np.float32),
            "base_1": np.linspace(2.0, 3.0, 6).astype(np.float32),
        }
    )
    positions = np.array([4, 1, 5, 0], dtype=np.int64)
    compact = pd.DataFrame({"compact_0": np.arange(4, dtype=np.float32)})
    signed = pd.DataFrame({"signed_0": -np.arange(4, dtype=np.float32)})
    cache_path = tmp_path / "base.npy"
    evidence = train.materialize_base_feature_cache(
        base=base,
        base_features=["base_0", "base_1"],
        output_path=cache_path,
        chunk_columns=1,
    )
    assert evidence["rows"] == 6
    assert evidence["features"] == 2
    cache = np.load(cache_path, mmap_mode="r")
    try:
        observed = train.assemble_matrix_from_base_cache(
            base_cache=cache,
            positions=positions,
            compact=compact,
            signed=signed,
            compact_features=["compact_0"],
            signed_features=["signed_0"],
            chunk_columns=1,
        )
    finally:
        train.close_memmap(cache)
    expected = np.column_stack(
        [
            base["base_0"].to_numpy()[positions],
            base["base_1"].to_numpy()[positions],
            compact["compact_0"].to_numpy(),
            signed["signed_0"].to_numpy(),
        ]
    ).astype(np.float32)
    np.testing.assert_array_equal(observed, expected)
    assert train.feature_matrix_sha256(
        observed,
        ["base_0", "base_1", "compact_0", "signed_0"],
    ) == train.feature_matrix_sha256(
        expected,
        ["base_0", "base_1", "compact_0", "signed_0"],
    )


def test_physical_oof_writer_is_chunked_and_preserves_rows(
    train: ModuleType,
    tmp_path: Path,
) -> None:
    parent = pd.DataFrame(
        {
            "id": [f"id_{index}" for index in range(7)],
            "well": ["a", "a", "b", "b", "c", "c", "d"],
            "outer_fold": np.arange(7) % 5,
            "actual_tvt": np.linspace(1.0, 2.0, 7).astype(np.float32),
            "last_known_tvt": np.linspace(0.0, 1.0, 7).astype(np.float32),
            "md_since": np.arange(7, dtype=np.float32),
        }
    )
    prediction = np.linspace(3.0, 4.0, 7).astype(np.float32)
    path = tmp_path / "physical.parquet"
    train.write_physical_candidate_oof(
        parent=parent,
        prediction=prediction,
        output_path=path,
        row_chunk=3,
    )
    observed = pd.read_parquet(path)
    assert observed["id"].tolist() == parent["id"].tolist()
    assert observed["candidate_id"].eq("exp226_w500_50_50").all()
    np.testing.assert_array_equal(
        observed["pred_tvt"].to_numpy(np.float32),
        prediction,
    )


def passing_audit() -> dict:
    fold = pd.DataFrame(
        {
            "outer_fold": np.arange(5),
            "delta_rmse_candidate_minus_parent": [-0.05] * 5,
        }
    )
    scope = pd.DataFrame(
        {
            "scope": ["near_0_250", "mid_250_1000", "far_1000_plus"],
            "delta_rmse_candidate_minus_parent": [-0.01, -0.01, -0.01],
        }
    )
    hidden = pd.DataFrame(
        {
            "scope": [
                "verification_like_spatial",
                "verification_like_typewell_purged",
            ],
            "delta_rmse_candidate_minus_parent": [-0.01, -0.01],
        }
    )
    return {
        "pooled": {"parent_rmse": 8.0, "candidate_rmse": 7.95},
        "fold": fold,
        "scope": scope,
        "hidden": hidden,
        "tail": {"delta_p95": -0.01, "worst_delta": 0.20},
    }


def test_constant_guard_is_all_and_and_fails_tail_regression(
    train: ModuleType,
    config: dict,
) -> None:
    audit = passing_audit()
    passed = train.evaluate_constant_guard(audit, {"coverage": True}, config)
    assert passed["passed"] is True
    audit["tail"]["delta_p95"] = 0.001
    failed = train.evaluate_constant_guard(audit, {"coverage": True}, config)
    assert failed["passed"] is False
    assert failed["checks"]["by_well_p95"] is False


def test_conditional_gate_is_fold_fit_target_free_and_capped(
    train: ModuleType,
    config: dict,
) -> None:
    rows = 500
    folds = np.arange(rows) % 5
    base = np.linspace(0.0, 10.0, rows)
    predictions = {
        "lgb": base,
        "cat": base + 0.1,
        "xgb": base - 0.1,
        "physics": base + np.linspace(-2.0, 2.0, rows),
    }
    constant = base + 0.02
    gated, quantiles = train.conditional_disagreement_prediction(
        predictions=predictions,
        constant_prediction=constant,
        deployment_weights=np.array([0.65, 0.15, 0.10, 0.10]),
        folds=folds,
        config=config,
    )
    assert len(quantiles) == 5
    assert all(item["meta_train_rows"] == 400 for item in quantiles)
    assert np.max(np.abs(gated - constant)) <= 0.25 + 1.0e-7
    assert np.isfinite(gated).all()


def test_compact_source_is_notebook_safe_and_canonical_is_adopted() -> None:
    source = SOURCE.read_text()
    assert "__file__" not in source
    assert "# ## Contents" in source
    assert "# ## 8. Execute Stage 0--5" in source
    assert "strict nested stacking" in source
    assert "model_count_10" in source
    canonical = json.loads((EXP_DIR / f"{EXP}_train.ipynb").read_text())
    candidate = json.loads((EXP_DIR / f"{EXP}_compact_selfcontained_train.ipynb").read_text())
    assert [(cell["cell_type"], cell["source"]) for cell in canonical["cells"]] == [
        (cell["cell_type"], cell["source"]) for cell in candidate["cells"]
    ]
    assert (EXP_DIR / f"{EXP}_inference.ipynb").exists()


def test_catboost_pool_releases_raw_fold_surface_before_fit() -> None:
    source = SOURCE.read_text()
    pool_import = source.index("from catboost import CatBoostRegressor, Pool")
    pool_build = source.index("cat_train_pool = Pool(")
    train_release = source.index("del x_train, y_train")
    valid_pool_build = source.index("cat_valid_pool = Pool(")
    valid_release = source.index("del x_valid, y_valid")
    cat_fit = source.index("cat_model.fit(")
    xgb_reload = source.index("xgb_train_positions,\n            xgb_valid_positions,")
    assert (
        pool_import
        < pool_build
        < train_release
        < valid_pool_build
        < valid_release
        < cat_fit
        < xgb_reload
    )
    assert source.count("load_compact_fold(") >= 4
    assert 'digest.update(memoryview(array).cast("B"))' in source
    assert "base.iloc[positions][columns]" not in source
    assert "base.loc[:, columns].iloc[positions].to_numpy(" in source
    assert "assemble_matrix_from_base_cache(" in source
    assert "_runtime_clean273_float32.npy" in source
    assert "release_process_memory()" in source
    assert '"phase": "train_matrix_ready"' in source
    assert "write_physical_candidate_oof(" in source


def test_hidden_inference_is_dynamic_saved_model_only_constant_stack(
    config: dict,
) -> None:
    source = INFERENCE_SOURCE.read_text()
    assert "__file__" not in source
    assert ".fit(" not in source
    assert "14151" not in source
    assert "reference_submission_override" in source
    assert "exp226_w500_50_50" in source
    assert "deployment_weights[name] * family_predictions[name]" in source
    assert "CatBoostRegressor()" in source
    assert "XGBRegressor()" in source
    assert source.count("load_model(") >= 2
    assert config["inference"]["deployment_weights"] == {
        "lgb": 0.681702678534061,
        "cat": 0.10372958993775055,
        "xgb": 0.01456773152818835,
        "physics": 0.2,
    }
    assert config["runtime"]["kaggle"]["inference_kernel_sources"][-1] == (
        "kentookumura/exp494-exp413-cat-xgb-physics-bounded-stack-train"
    )
    canonical = json.loads((EXP_DIR / f"{EXP}_inference.ipynb").read_text())
    candidate = json.loads((EXP_DIR / f"{EXP}_compact_selfcontained_inference.ipynb").read_text())
    assert [(cell["cell_type"], cell["source"]) for cell in canonical["cells"]] == [
        (cell["cell_type"], cell["source"]) for cell in candidate["cells"]
    ]

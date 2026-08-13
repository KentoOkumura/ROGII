from __future__ import annotations

import copy
import inspect
import json
import os
import runpy
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.test_support import require_saved_files

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / (
    "exp393_exp347_practical_numerical_equivalence_audit"
)
TRAIN = EXP_DIR / (
    "exp393_exp347_practical_numerical_equivalence_audit_"
    "compact_selfcontained_train.py"
)
INFERENCE = EXP_DIR / (
    "exp393_exp347_practical_numerical_equivalence_audit_"
    "compact_selfcontained_inference.py"
)
CONFIG = EXP_DIR / "config.yaml"
STAGE_A_ARTIFACTS = EXP_DIR / "kaggle" / "output" / "stage_a_v4" / "artifacts"
PARENT_TRAIN = ROOT / "experiments" / (
    "exp347_prefix_gr_unary_batched_window_exact_ssm"
) / "exp347_prefix_gr_unary_batched_window_exact_ssm_compact_selfcontained_train.py"


def load_namespace(path: Path, env_name: str) -> dict[str, object]:
    previous = os.environ.get(env_name)
    os.environ[env_name] = "1"
    try:
        return runpy.run_path(str(path))
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


@pytest.fixture(scope="module")
def module() -> dict[str, object]:
    return load_namespace(TRAIN, "EXP393_IMPORT_ONLY")


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load(CONFIG.read_text())
    assert isinstance(value, dict)
    return value


def test_stage_a_user_override_and_execution_contract_are_fixed(
    module: dict[str, object],
    config: dict,
) -> None:
    scientific = module["validate_scientific_contract"](config)
    stage0_execution = module["validate_execution_contract"](config)
    authorized = copy.deepcopy(config)
    authorized["execution"]["kaggle_push_approved"] = True
    authorized["execution"]["stage_a_gpu_approved"] = True
    authorized["execution"]["selected_stage"] = "stage_a_fold0"
    authorized["execution"]["run_stage_a"] = True
    stage_a_execution = module["validate_stage_a_cost_contract"](authorized)
    assert scientific == {
        "locked_fields": 71,
        "preregistered_stage0_gates": 13,
        "stage_a_controls": 3,
        "stage_a_gates": 10,
    }
    assert stage0_execution == {
        "active_audits": 1,
        "fixed_windows": 16,
        "temporary_neural_models": 1,
        "persisted_models": 0,
        "trained_folds": 0,
        "lightgbm_configs": 0,
        "boosters": 0,
        "pf_beam_runs": 0,
        "parent_or_control_retraining": 0,
    }
    assert stage_a_execution == {
        "active_variants": 1,
        "fold_indices": [0],
        "active_architectures": 1,
        "seeds": [42],
        "neural_model_count": 1,
        "persisted_model_count": 1,
        "lightgbm_config_count": 0,
        "total_boosters": 0,
        "control_model_training": 0,
        "pf_beam_well_runs": 0,
        "parent_control_retraining": False,
    }
    assert config["experiment"]["status"] == "stage_a_failed_branch_closed"
    assert config["implementation"]["approved"] is True
    assert config["implementation"]["approval_source"] == (
        "user_message_implement_exp393_2026_07_25"
    )
    assert config["execution"]["selected_stage"] == "implementation_only"
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["execution"]["run_stage0"] is False
    assert config["execution"]["run_stage_a"] is False
    assert config["execution"]["stage_a_gpu_approved"] is False
    assert config["execution"]["current_trained_fold_count"] == 1
    assert config["execution"]["inference_approved"] is False
    assert config["execution"]["submission_approved"] is False
    assert config["execution"]["stage0_gate"]["passed"] is False
    assert module["validate_selected_stage"](config) == "implementation_only"
    assert config["execution"]["stage_a_result"]["passed"] is False
    assert config["execution"]["stage_a_result"]["failed_checks"] == [
        "real_rmse_vs_exp209",
        "well_p95_non_regression",
        "worst_well_regression",
    ]

    unauthorized = copy.deepcopy(authorized)
    unauthorized["execution"]["stage_a_user_override"]["approved"] = False
    with pytest.raises(ValueError, match="explicit post-FAIL user override"):
        module["validate_selected_stage"](unauthorized)

    reclassified = copy.deepcopy(authorized)
    reclassified["execution"]["stage0_gate"]["passed"] = True
    with pytest.raises(ValueError, match="must not be reclassified"):
        module["validate_selected_stage"](reclassified)


def test_gpu_guard_accepts_the_selected_stage_instead_of_requiring_stage0(
    module: dict[str, object],
) -> None:
    source = inspect.getsource(module["require_kaggle_gpu"])
    assert "selected_stage = validate_selected_stage(config)" in source
    assert '"stage0_practical_audit", "stage_a_fold0"' in source
    assert "execution.run_stage0" not in source


def test_parent_fixed16_and_terminal_exp347_evidence_is_sha_locked(
    module: dict[str, object],
    config: dict,
) -> None:
    data = config["data"]
    require_saved_files(
        *(ROOT / data[name]["candidates"][0] for name in (
            "parent_config",
            "parent_stage0_report",
            "parent_window_manifest",
            "parent_boundary_manifest",
        ))
    )
    windows, boundaries, evidence = module["load_parent_fixed16_contract"](
        config
    )
    assert len(windows) == 16
    assert windows["well"].nunique() == 16
    assert windows["benchmark_order"].tolist() == list(range(16))
    assert len(boundaries) == 16
    assert set(boundaries["boundary_source"]) == {"official_prefix"}
    assert set(evidence) == {
        "parent_source",
        "parent_config",
        "parent_stage0_report",
        "parent_window_manifest",
        "parent_boundary_manifest",
    }
    assert evidence["parent_stage0_report"]["sha256"] == (
        "e8a706ba9a75dff54b30b97f289255b002333cb76d2b2dfcac000cfdf56fe454"
    )


def test_exp347_scalar_and_production_batched_dp_sources_are_unchanged(
    module: dict[str, object],
) -> None:
    def source_block(path: Path, start: str, stop: str) -> str:
        source = path.read_text()
        return source[source.index(start) : source.index(stop, source.index(start))]

    assert source_block(
        TRAIN,
        "    def exact_forward_backward(",
        "    def gaussian_label_log_emission(",
    ) == source_block(
        PARENT_TRAIN,
        "    def exact_forward_backward(",
        "    def gaussian_label_log_emission(",
    )
    assert source_block(
        TRAIN,
        "    def batched_exact_forward_backward(",
        "    def batched_gaussian_label_log_emission(",
    ) == source_block(
        PARENT_TRAIN,
        "    def batched_exact_forward_backward(",
        "    def batched_gaussian_label_log_emission(",
    )
    stage0_source = inspect.getsource(module["run_stage0_practical_audit"])
    assert stage0_source.index("freeze_unaries_once(") < stage0_source.index(
        "load_audit_truths("
    )
    assert "run_stage_a(" not in stage0_source
    assert '"exp347_status_changed": False' in stage0_source


def test_stage_a_training_and_readout_sources_match_exp347(
    module: dict[str, object],
) -> None:
    parent = load_namespace(PARENT_TRAIN, "EXP347_IMPORT_ONLY")
    for name in (
        "stable_window_order",
        "batched_window_training_loss",
        "train_fold0_model",
        "load_exp209_baseline",
        "save_model_checkpoint",
        "freeze_outer_valid_predictions",
        "post_freeze_readout",
        "run_stage_a",
    ):
        assert inspect.getsource(module[name]) == inspect.getsource(parent[name])


def test_dtype_generalized_scalar_matches_parent_fp32_and_supports_fp64(
    module: dict[str, object],
    config: dict,
) -> None:
    torch = pytest.importorskip("torch")
    state = module["StateSpec"](
        grid=100.0 + np.arange(12, dtype=np.float64) * 0.35,
        rates=np.linspace(-0.1, 0.1, 41, dtype=np.float64),
        suffix_index=np.arange(6, dtype=np.int64),
        dm=np.ones(6, dtype=np.float64),
        dz=np.zeros(6, dtype=np.float64),
        start_p=4.0,
        init_rate=0.0,
        prefix_end=-1,
        last_known_tvt=101.4,
    )
    generator = torch.Generator().manual_seed(42)
    unary = torch.randn((6, 12), generator=generator, dtype=torch.float32)
    parent_posterior, parent_partition = module["exact_forward_backward"](
        unary, state, config
    )
    generalized_fp32, generalized_partition = module[
        "scalar_exact_forward_backward_dtype"
    ](unary, state, config, torch.float32)
    torch.testing.assert_close(
        generalized_fp32, parent_posterior, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        generalized_partition, parent_partition, rtol=0.0, atol=0.0
    )
    fp64_posterior, fp64_partition = module[
        "scalar_exact_forward_backward_dtype"
    ](unary, state, config, torch.float64)
    assert fp64_posterior.dtype == torch.float64
    assert fp64_partition.dtype == torch.float64
    assert torch.isfinite(fp64_posterior).all()
    assert torch.max(
        torch.abs(fp64_posterior.float() - parent_posterior)
    ).item() < 1e-3


def test_practical_gate_excludes_legacy_posterior_cell_threshold(
    module: dict[str, object],
    config: dict,
) -> None:
    posterior = np.asarray([[0.25, 0.75]], dtype=np.float64)
    diagnostics = {
        "posterior_mean_tvt": {
            "rmse_ft": 0.0005,
            "p99_abs_ft": 0.002,
            "max_abs_ft": 0.01,
        },
        "marginal_map": {"agreement_rate": 1.0},
        "posterior_cell_diagnostic_only": {
            "max_abs_error": 0.5,
            "legacy_exp347_1e6_check": False,
            "promotion_gate": False,
        },
    }
    objective = {
        "loss_max_abs_error": 0.0,
        "partition_max_abs_error": 0.0,
        "gradient_max_abs_error": 0.0,
        "optimizer_step_max_abs_error": 0.0,
        "invalid_gradient_max_abs": 0.0,
        "finite": True,
    }
    mode = {
        "finite": True,
        "invalid_posterior_max_abs": 0.0,
        "posteriors": [posterior],
        "measurements": [{"seconds": 0.1}],
    }
    gate = module["evaluate_practical_gate"](
        diagnostics,
        objective,
        [mode],
        outer_valid_truth_access_count=0,
        stage_a_model_count=0,
        peak_gpu_memory_gb=1.0,
        audit_runtime_hours=0.1,
        config=config,
    )
    assert gate["passed"] is True
    assert gate["legacy_posterior_cell_1e6_is_gate"] is False
    assert "posterior_cell" not in gate["checks"]


def test_completed_stage_a_evidence_matches_the_closed_config(config: dict) -> None:
    prefix = "exp393_exp347_practical_numerical_equivalence_audit"
    metrics_path = STAGE_A_ARTIFACTS / f"{prefix}_stage_a_metrics.json"
    manifest_path = STAGE_A_ARTIFACTS / f"{prefix}_model_manifest.json"
    require_saved_files(metrics_path, manifest_path)
    metrics = json.loads(
        metrics_path.read_text()
    )
    manifest = json.loads(manifest_path.read_text())
    result = config["execution"]["stage_a_result"]
    assert metrics["guard"]["passed"] is False
    assert {
        name for name, passed in metrics["guard"]["checks"].items() if not passed
    } == {
        "real_rmse_vs_exp209",
        "well_p95_non_regression",
        "worst_well_regression",
    }
    assert metrics["metrics"]["real_rmse"] == pytest.approx(result["real_rmse_ft"])
    assert metrics["metrics"]["exp209_rmse"] == pytest.approx(
        result["exp209_rmse_ft"]
    )
    assert manifest["model_sha256"] == result["model_sha256"]
    assert manifest["neural_model_count"] == 1
    assert manifest["parent_control_retraining"] is False


def test_inference_and_submission_remain_disabled(config: dict) -> None:
    inference = load_namespace(INFERENCE, "EXP393_IMPORT_ONLY")
    report = inference["validate_disabled_inference"](config)
    assert report["inference_enabled"] is False
    assert report["create_submission"] is False
    assert report["persistent_model_count"] == 1
    assert report["stage_a_authorized"] is False
    unsafe = copy.deepcopy(config)
    unsafe["execution"]["inference_approved"] = True
    with pytest.raises(ValueError, match="must remain false"):
        inference["validate_disabled_inference"](unsafe)


def test_canonical_notebooks_are_readable_self_contained_exports() -> None:
    train_notebook = EXP_DIR / (
        "exp393_exp347_practical_numerical_equivalence_audit_train.ipynb"
    )
    inference_notebook = EXP_DIR / (
        "exp393_exp347_practical_numerical_equivalence_audit_inference.ipynb"
    )
    for path, required_heading in (
        (train_notebook, "Stage A orchestration and generated artifacts"),
        (inference_notebook, "Fail-closed inference contract"),
    ):
        notebook = json.loads(path.read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert required_heading in source
        assert "from exp393_" not in source

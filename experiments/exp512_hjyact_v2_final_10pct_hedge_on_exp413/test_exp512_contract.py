from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
CANDIDATE = (
    HERE
    / "exp512_hjyact_v2_final_10pct_hedge_on_exp413_"
    "compact_selfcontained_inference.py"
)


def read_yaml(name: str) -> dict:
    value = yaml.safe_load((HERE / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def candidate_text() -> str:
    return CANDIDATE.read_text(encoding="utf-8")


def test_candidate_is_parseable_and_exact_equal_blend_is_frozen() -> None:
    source = candidate_text()
    ast.parse(source)
    config = read_yaml("config.yaml")
    contract = read_yaml("ensemble_contract.yaml")
    assert config["experiment"]["route"] == "ensemble"
    assert config["experiment"]["legacy_name_suffix"] == "10pct_hedge"
    assert config["ensemble"]["exp413_weight"] == 0.50
    assert config["ensemble"]["hjyact_v2_final_weight"] == 0.50
    assert contract["final_formula"]["exp413_weight"] == 0.50
    assert contract["final_formula"]["hjyact_v2_final_weight"] == 0.50
    assert "EXP413_WEIGHT = 0.50" in source
    assert "HJYACT_WEIGHT = 0.50" in source
    assert "0.90 * exp413" not in source
    left = np.asarray([1.0, -2.0, 7.5], dtype=np.float64)
    right = np.asarray([3.0, 6.0, -1.5], dtype=np.float64)
    np.testing.assert_array_equal(0.50 * left + 0.50 * right, (left + right) / 2.0)


def test_source_identity_and_active_path_are_pinned() -> None:
    source = candidate_text()
    assert (
        'SOURCE_PULL_NOTEBOOK_SHA256 = "4b4879a6d427422c127a300e09dc763b71ea5e7878eb3639941c75753a23933c"'
        in source
    )
    assert (
        'SOURCE_CODE_CELL_SHA256 = "ee93ce4c80c6490cbf2f9cfe518e8e3b54516c212aa813c4a045a64b4c126088"'
        in source
    )
    assert "SHA-pinned learned trajectory models are required" in source
    assert "inference-time training is forbidden" in source
    assert "_find_precomputed_learned_submission" not in source


def test_shared_dag_and_route_specific_pf_boundary_are_fail_closed() -> None:
    source = candidate_text()
    assert "CANDIDATE_REUSE_TRACKER = CandidateReuseTracker" in source
    assert '"generation_count": 1' in source
    assert 'record["cache_hit_count"] += 1' in source
    assert '"fallback_to_duplicate_generation": False' in source
    assert "build_replay_test_frame()" not in source
    assert "shared_deterministic_frame.copy(deep=True)" in source
    assert "replay_source.run_pf_ancc" in source
    assert "replay_source.run_pf_z" in source
    assert "replay_source.build_likpf(test_wells, \"test\")" in source
    assert 'reuse_tracker.mark_exp413_hit(route_pf_result["well"])' in source


def test_dynamic_sample_and_posthoc_visible_reference_contract() -> None:
    source = candidate_text()
    config = read_yaml("config.yaml")
    contract = read_yaml("ensemble_contract.yaml")
    assert "def resolve_competition_data_root()" in source
    assert 'Path("/kaggle/input/rogii-wellbore-geology-prediction")' in source
    assert (
        "COMPETITION_DATA_ROOT = '/kaggle/input/competitions/rogii-wellbore-geology-prediction'"
        not in source
    )
    assert (
        "RIDGE_ARTIFACT_ROOT = '/kaggle/input/datasets/ravaghi/"
        "wellbore-geology-prediction-artifacts'"
        not in source
    )
    assert 'RIDGE_ARTIFACT_ROOT = str(HJYACT_INPUT_AUDIT["roots"]["ridge"])' in source
    dynamic_read = 'sample = pd.read_csv(CFG.DATA / "sample_submission.csv", dtype={"id": str})'
    identity_check = 'visible_reference_checks = {"sample_id_order_match": sample_id_sha == VISIBLE_SAMPLE_ID_ORDER_SHA256}'
    source_sha_check = "if visible_reference_checks[\"sample_id_order_match\"]:"
    assert dynamic_read in source
    assert identity_check in source
    assert source.index(dynamic_read) < source.index(identity_check) < source.index(source_sha_check)
    assert "static_exp413_prediction" not in source
    assert "public visible output CSV" not in source
    parity = config["validation"]["exp413_component_parity"]
    assert parity["require_decompressed_content_sha_match"] is False
    assert parity["require_preaudited_witness_sha_for_nonexact_visible_output"] is True
    assert parity["numerical_max_abs_tolerance_ft"] == 0.02
    assert parity["numerical_rmse_tolerance_ft"] == 0.001
    assert contract["components"]["exp413"]["numerical_max_abs_tolerance_ft"] == 0.02
    assert contract["components"]["exp413"]["numerical_rmse_tolerance_ft"] == 0.001
    assert "EXP413_VISIBLE_NUMERICAL_WITNESS_CONTENT_SHA256" in source
    assert "EXP413_VISIBLE_NUMERICAL_MAX_ABS_TOLERANCE_FT = 0.02" in source
    assert "EXP413_VISIBLE_NUMERICAL_RMSE_TOLERANCE_FT = 0.001" in source
    assert 'exp413_parity_mode = "preaudited_platform_numerical_tolerance_witness"' in source
    assert "neither the exact reference nor the " in source
    assert '"pre-audited numerical tolerance witness: "' in source


def test_model_inventory_and_authorization_boundary() -> None:
    config = read_yaml("config.yaml")
    manifest = read_yaml("model_manifest.yaml")
    assert manifest["new_booster_training_count"] == 0
    assert manifest["runtime_fit"]["total_fits"] == 5
    assert manifest["saved_model_files"]["total"] == 83
    assert manifest["saved_model_files"]["contained_estimators_total"] == 103
    assert len(manifest["saved_model_files"]["hjyact"]["trainer_wrappers"]) == 5
    assert len(manifest["saved_model_files"]["hjyact"]["learned_trajectory"]) == 3
    assert manifest["saved_model_files"]["hjyact"]["model_package"] == []
    assert manifest["saved_model_files"]["hjyact"]["model_package_correction_enabled"] is False
    assert config["authorization"]["implementation_approved"] is True
    assert config["authorization"]["canonical_notebook_adoption_approved"] is False
    assert config["authorization"]["kaggle_run_approved"] is True
    assert config["authorization"]["competition_submission_approved"] is False


def test_four_way_well_parallelism_and_model_package_disable_are_frozen() -> None:
    source = candidate_text()
    config = read_yaml("config.yaml")
    assert "SP45_WELL_N_JOBS = 4" in source
    assert "EXP413_WELL_N_JOBS = 4" in source
    assert "MODEL_PACKAGE_CORRECTION_ENABLED = False" in source
    assert "RUN_MODEL_PACKAGE_CORRECTION = MODEL_PACKAGE_CORRECTION_ENABLED" in source
    assert "HJYACT_REQUIRED_INPUTS[\"model_package\"]" not in source
    assert "n_jobs=_sp45_effective_n_jobs" in source
    assert "n_jobs=route_pf_effective_n_jobs" in source
    assert "n_jobs=effective_n_jobs" in source
    assert "well_n_jobs = min(4, len(test_wells))" in source
    assert 'numba_njit(cache=False, nogil=True)' in source
    assert config["runtime"]["well_parallelism"]["sp45"]["n_jobs"] == 4
    assert config["runtime"]["well_parallelism"]["exp413_hmm"]["n_jobs"] == 4
    assert config["runtime"]["well_parallelism"]["exp413_route_pf"]["n_jobs"] == 4
    assert config["runtime"]["well_parallelism"]["exp413_k16"]["n_jobs"] == 4
    assert config["runtime"]["model_package_correction_enabled"] is False
    assert config["public_component"]["model_package"]["enabled"] is False
    assert "pilkwang/rogii-model-package" not in config["runtime"]["kaggle"]["dataset_sources"]


def test_exp413_train_only_kappa_is_pinned_with_runtime_audit() -> None:
    source = candidate_text()
    assert "EXP413_K16_PINNED_KAPPA = (" in source
    assert "EXP413_K16_RUNTIME_KAPPA_AUDIT_MAX_ABS = 1.0e-7" in source
    assert "runtime_fit_kappa = np.asarray(" in source
    assert 'environment["OPENBLAS_CORETYPE"] = "Haswell"' in source
    assert 'architectures != ["Haswell"]' in source
    assert "run_exp413_k16_haswell_subprocess(" in source
    assert '"execution_mode"] = "isolated_haswell_openblas_subprocess"' in source
    assert '"kappa_source": "pinned_train_only_exp413_visible_reference"' in source
    assert '"runtime_fit_vs_pinned_max_abs": runtime_fit_vs_pinned_max_abs' in source


def test_candidate_hash_matches_config_after_finalization() -> None:
    config = read_yaml("config.yaml")
    observed = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    assert config["implementation"]["candidate_source_sha256"] == observed
    assert config["implementation"]["candidate_source_lines"] == len(
        CANDIDATE.read_text(encoding="utf-8").splitlines()
    )

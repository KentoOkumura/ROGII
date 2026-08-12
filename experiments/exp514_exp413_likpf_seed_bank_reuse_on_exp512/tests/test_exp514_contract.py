from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
CANDIDATE = (
    HERE
    / "exp514_exp413_likpf_seed_bank_reuse_on_exp512_"
    "compact_selfcontained_inference.py"
)
STAGE_A_SOURCE = (
    HERE / "exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_a_fixed32.py"
)
STAGE_B_SOURCE = (
    HERE / "exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_b_fixed32.py"
)
GENERATOR = HERE / "prepare_exp514_shared_likpf_candidate.py"
STAGE_B_GENERATOR = HERE / "prepare_exp514_stage_b_fixed32.py"
STAGE_D_SOURCE = (
    HERE / "exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_d_visible.py"
)
STAGE_D_GENERATOR = HERE / "prepare_exp514_stage_d_visible.py"
PARENT = (
    ROOT
    / "experiments/exp512_hjyact_v2_final_10pct_hedge_on_exp413"
    / "exp512_hjyact_v2_final_10pct_hedge_on_exp413_compact_selfcontained_inference.py"
)
REPLAY_SOURCE = (
    ROOT
    / "experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay"
    / "public_notebook_replay_audit.py"
)

EXPECTED_PARENT_SHA256 = (
    "16982879716918811dfa9915c4862d45836bd9360efafbaee41046c3e1b6240f"
)
EXPECTED_REPLAY_SHA256 = (
    "4af212a8a1c83e36cdcc0bc912942a62df1fbc94ca67fd75789171afaa1a647e"
)


def read_yaml(name: str) -> dict:
    value = yaml.safe_load((HERE / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def candidate_text() -> str:
    return CANDIDATE.read_text(encoding="utf-8")


def function_node(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return node


def normalized_function_dump(
    path: Path,
    name: str,
    *,
    canonical_name: str,
    identifier_renames: dict[str, str] | None = None,
) -> str:
    node = copy.deepcopy(function_node(path.read_text(encoding="utf-8"), name))
    node.name = canonical_name
    renames = identifier_renames or {}

    class Rename(ast.NodeTransformer):
        def visit_Name(self, item: ast.Name) -> ast.AST:
            item.id = renames.get(item.id, item.id)
            return item

    node = Rename().visit(node)
    ast.fix_missing_locations(node)
    return ast.dump(node, include_attributes=False)


def load_shared_runtime_module(source_path: Path = CANDIDATE) -> SimpleNamespace:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        "shared_likpf_stable_seed",
        "_shared_likpf_interp1",
        "_shared_pf_lik_allseeds",
        "_shared_likpf_grid",
        "_shared_array_sha",
        "_shared_json_sha",
        "_shared_branch_summary",
        "_shared_likpf_one_well",
        "materialize_shared_likpf_bank",
        "shared_likpf_sp45_adapter",
        "release_shared_likpf_sp45_payload",
        "shared_likpf_exp413_adapter",
        "finalize_shared_likpf_manifest",
    }
    constants = {
        "EXPERIMENT_NAME",
        "EXP073_REPLAY_SOURCE_SHA256",
        "SHARED_LIKPF_SCALES",
        "SHARED_LIKPF_PARTICLES",
        "SHARED_LIKPF_SEEDS",
        "SHARED_LIKPF_BRANCH_SCALE",
        "SHARED_LIKPF_N_JOBS",
        "SHARED_LIKPF_SEED_NAMESPACE",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(ast.get_source_segment(source, node))
        elif isinstance(node, ast.Assign):
            assigned = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if assigned & constants:
                selected.append(ast.get_source_segment(source, node))
    runtime_source = "\n".join(
        [
            "import hashlib, json, time",
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from joblib import Parallel, delayed",
            "def njit(*args, **kwargs):",
            "    def decorate(function):",
            "        return function",
            "    return decorate",
            *selected,
        ]
    )
    namespace: dict[str, object] = {}
    exec(compile(runtime_source, "<exp514-shared-runtime>", "exec"), namespace)

    def deterministic_test_core(
        md_v,
        z_v,
        gr_v,
        gg,
        vmin,
        step,
        gs,
        ls,
        ir,
        n_particles,
        n_seeds,
        seed_base,
        momentum,
        velocity_noise,
        position_noise,
        rough_position,
        rough_rate,
        resample_fraction,
        initial_spread,
    ):
        del (
            gg,
            vmin,
            step,
            gs,
            n_particles,
            momentum,
            velocity_noise,
            position_noise,
            rough_position,
            rough_rate,
            resample_fraction,
            initial_spread,
        )
        predictions = []
        likelihoods = []
        base = ls - np.asarray(z_v, dtype=np.float64) + ir * (
            np.asarray(md_v, dtype=np.float64) - float(md_v[0])
        )
        for seed_index in range(int(n_seeds)):
            rng = np.random.default_rng(int(seed_base) + seed_index)
            noise = rng.normal(0.0, 0.25, len(base)).cumsum()
            prediction = base + noise
            predictions.append(prediction)
            likelihoods.append(-0.1 * seed_index)
        return np.stack(predictions), np.asarray(likelihoods, dtype=np.float64)

    namespace["_shared_pf_lik_allseeds"] = deterministic_test_core
    return SimpleNamespace(**namespace)


def synthetic_loader(well: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert split == "test"
    offset = 0.0 if well == "well_a" else 2.5
    rows = 32
    known_rows = 16
    md = np.arange(rows, dtype=np.float64) * 10.0 + 1000.0
    z = -1000.0 - np.arange(rows, dtype=np.float64) * 0.25
    tvt_input = np.full(rows, np.nan, dtype=np.float64)
    tvt_input[:known_rows] = 800.0 + offset + np.arange(known_rows) * 0.7
    gr = 60.0 + 8.0 * np.sin(np.arange(rows) / 4.0 + offset)
    horizontal = pd.DataFrame(
        {"MD": md, "Z": z, "GR": gr, "TVT_input": tvt_input}
    )
    typewell_tvt = np.arange(650.0, 1050.1, 2.0)
    typewell = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": 60.0 + 8.0 * np.sin(typewell_tvt / 28.0),
        }
    )
    return horizontal, typewell


def test_source_identity_and_generator_are_pinned() -> None:
    source = candidate_text()
    ast.parse(source)
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == EXPECTED_PARENT_SHA256
    assert hashlib.sha256(REPLAY_SOURCE.read_bytes()).hexdigest() == EXPECTED_REPLAY_SHA256
    assert f'EXP512_PARENT_SOURCE_SHA256 = "{EXPECTED_PARENT_SHA256}"' in source
    assert f'EXP073_REPLAY_SOURCE_SHA256 = "{EXPECTED_REPLAY_SHA256}"' in source
    generator_sha = hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
    assert f'EXP514_GENERATOR_SHA256 = "{generator_sha}"' in source
    assert "__file__" not in source


def test_shared_numba_kernel_is_source_identical_to_exp073_contract() -> None:
    candidate_kernel = normalized_function_dump(
        CANDIDATE,
        "_shared_pf_lik_allseeds",
        canonical_name="_pf_lik_allseeds",
        identifier_renames={"_shared_likpf_interp1": "_interp1"},
    )
    replay_kernel = normalized_function_dump(
        REPLAY_SOURCE,
        "_pf_lik_allseeds",
        canonical_name="_pf_lik_allseeds",
    )
    assert candidate_kernel == replay_kernel
    candidate_interp = normalized_function_dump(
        CANDIDATE,
        "_shared_likpf_interp1",
        canonical_name="_interp1",
    )
    replay_interp = normalized_function_dump(
        REPLAY_SOURCE,
        "_interp1",
        canonical_name="_interp1",
    )
    assert candidate_interp == replay_interp
    candidate_seed = normalized_function_dump(
        CANDIDATE,
        "shared_likpf_stable_seed",
        canonical_name="stable_seed",
    )
    replay_seed = normalized_function_dump(
        REPLAY_SOURCE,
        "stable_seed",
        canonical_name="stable_seed",
    )
    assert candidate_seed == replay_seed


def test_shared_bank_is_thread_stable_and_consumed_exactly_once() -> None:
    runtime = load_shared_runtime_module()
    wells = ["well_b", "well_a"]
    bank_one, _ = runtime.materialize_shared_likpf_bank(
        wells,
        "test",
        synthetic_loader,
        n_jobs=1,
        particles=32,
        seeds=4,
    )
    bank_two, _ = runtime.materialize_shared_likpf_bank(
        wells,
        "test",
        synthetic_loader,
        n_jobs=2,
        particles=32,
        seeds=4,
    )
    assert list(bank_one) == ["well_a", "well_b"]
    assert list(bank_two) == ["well_a", "well_b"]
    assert bank_one["well_a"]["seed_base"] != bank_one["well_b"]["seed_base"]
    for well in bank_one:
        for name in ("pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12", "pf_mean"):
            np.testing.assert_array_equal(
                bank_one[well]["sp45_full"][name],
                bank_two[well]["sp45_full"][name],
            )
        assert bank_one[well]["branch_summary"] == bank_two[well]["branch_summary"]
        assert bank_one[well]["audit"]["raw_seed_bank_retained"] is False
        assert "predictions" not in bank_one[well]
        assert "log_likelihoods" not in bank_one[well]
        sp45_frame, branch = runtime.shared_likpf_sp45_adapter(bank_one[well])
        assert set(sp45_frame) == {
            "pf_scale_3",
            "pf_scale_5",
            "pf_scale_8",
            "pf_scale_12",
            "pf_mean",
        }
        assert branch["seed_count"] == 4
    exp413_frame = runtime.shared_likpf_exp413_adapter(
        bank_one, ["well_a", "well_b"]
    )
    assert list(exp413_frame.columns) == ["id", "likpf_scale_5", "likpf_mean"]
    assert exp413_frame["likpf_scale_5"].dtype == np.float32
    manifest = runtime.finalize_shared_likpf_manifest(
        bank_one, ["well_a", "well_b"]
    )
    assert manifest["all_contracts_passed"] is True
    assert manifest["raw_seed_bank_retained"] is False
    assert all(
        record["ledger"]["core_calls"] == 1
        for record in bank_one.values()
    )


def test_candidate_orchestration_removes_both_duplicate_pf_calls() -> None:
    source = candidate_text()
    sp45 = ast.unparse(function_node(source, "_run_sp45_test_well"))
    exp413 = ast.unparse(function_node(source, "generate_dynamic_exp413_prediction"))
    assert "shared_likpf_sp45_adapter" in sp45
    assert "run_pf_lik_ensemble_scales" not in sp45
    assert "last_val" not in sp45
    assert "shared_likpf_exp413_adapter" in exp413
    assert "replay_source.build_likpf" not in exp413
    assert "replay_source.run_pf_ancc" in exp413
    assert "replay_source.run_pf_z" in exp413
    assert source.index("SHARED_LIKPF_BANK, SHARED_LIKPF_PARALLEL_REPORT") < source.index(
        "def _run_sp45_test_well"
    )
    assert "_gold_pf_candidates" in source
    assert "run_beam_ensemble" in source
    assert "gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10., 60.)) * 1.3" in source


def test_runtime_v3_reuses_deterministic_features_and_parallelizes_gold() -> None:
    source = candidate_text()
    main_source = ast.unparse(function_node(source, "main"))
    reuse_source = ast.unparse(
        function_node(source, "build_hjyact_features_from_sp45")
    )
    pf_refresh_source = ast.unparse(function_node(source, "_hjyact_pf_only_well"))

    assert "SP45_SHARED_TEST_FEATURE_FRAME = test_df" in source
    assert "SP45_SHARED_IMPUTERS = (FI, DI)" in source
    assert "build_hjyact_features_from_sp45(test_wids, 'test')" in main_source
    assert "build_features(test_wids" not in main_source
    assert "init_imputers(train_wids)" not in main_source
    assert "run_pf_ancc" in pf_refresh_source
    assert "run_pf_z" in pf_refresh_source
    assert "beam_search" not in reuse_source
    assert "multi_scale_ncc" not in reuse_source
    assert "FormationPlaneKNN" not in reuse_source
    assert '"hjyact_full_build_features_calls": 0' in source
    assert "__exp514_shared_tvt_dense_abs" in source

    assert "backend='multiprocessing'" in source
    assert "_gold_requested_processes = 4" in source
    assert "GOLD_WELL_PARALLEL_REPORT" in source
    assert "for _wi, _wid in enumerate(_gold_wells, 1):" not in source
    assert "inner_kdtree_workers" in source
    assert "well_parallel=GOLD_WELL_PARALLEL_REPORT" in source

    stage_d = STAGE_D_SOURCE.read_text(encoding="utf-8")
    assert '"name": "gold_visible_prefix"' in stage_d
    assert '"scaling": "parallel_4_process"' in stage_d
    assert '"gold_well_parallel": GOLD_WELL_PARALLEL_REPORT' in stage_d


def test_stage_d_v4_bounds_pf_payload_and_releases_consumed_arrays() -> None:
    source = STAGE_D_SOURCE.read_text(encoding="utf-8")
    assert "STAGE_D_RUNTIME_REVISION = 4" in source
    assert "def _run_shared_likpf_sp45_well(order, wid):" in source
    assert "del _shared_sp45_pipeline_results" in source
    assert "'all_well_full_payload_retained': False" in source
    assert "'max_concurrent_full_payload_wells': _sp45_effective_n_jobs" in source
    assert "SHARED_LIKPF_BANK, SHARED_LIKPF_PARALLEL_REPORT = materialize_shared_likpf_bank(" not in source

    runtime = load_shared_runtime_module(STAGE_D_SOURCE)
    record = runtime._shared_likpf_one_well(
        "well_a",
        "test",
        synthetic_loader,
        particles=32,
        seeds=4,
    )
    sp45_frame, _ = runtime.shared_likpf_sp45_adapter(record)
    assert set(sp45_frame) == {
        "pf_scale_3",
        "pf_scale_5",
        "pf_scale_8",
        "pf_scale_12",
        "pf_mean",
    }
    runtime.release_shared_likpf_sp45_payload(record)
    for key in ("sp45_full", "row_index", "evaluation_index", "known_mask"):
        assert key not in record
    exp413_frame = runtime.shared_likpf_exp413_adapter(
        {"well_a": record}, ["well_a"]
    )
    assert list(exp413_frame.columns) == ["id", "likpf_scale_5", "likpf_mean"]
    assert "exp413_frame" not in record
    manifest = runtime.finalize_shared_likpf_manifest(
        {"well_a": record}, ["well_a"]
    )
    assert manifest["all_contracts_passed"] is True


def test_stage_d_v4_releases_ridge_and_transfers_dataframe_ownership() -> None:
    source = STAGE_D_SOURCE.read_text(encoding="utf-8")
    ridge_release = source.index("RIDGE_MEMORY_RELEASE_REPORT = _exp514_release_globals(")
    streaming_start = source.index("def _run_shared_likpf_sp45_well(order, wid):")
    assert ridge_release < streaming_start
    assert "test_df2 = test_df[['id', 'well', 'md_since']].copy()" in source
    assert "test_df2 = test_df.copy()" not in source
    assert "result = base.copy(deep=True)" not in source
    assert 'globals()["SP45_SHARED_TEST_FEATURE_FRAME"] = None' in source
    assert "pf_frame = shared_deterministic_frame.copy(deep=True)" not in source
    assert 'shared_deterministic_frame=globals().pop("HJYACT_SHARED_FEATURE_FRAME")' in source
    assert "pf_frame.copy(), exp145_candidates" not in source
    assert "pf_frame.copy(), paths.test_data_dir" not in source
    assert "return predictions.copy()" not in source


def test_stage_d_v4_visible_sha_witness_is_hidden_safe() -> None:
    source = STAGE_D_SOURCE.read_text(encoding="utf-8")
    guard = 'if visible_reference_checks["sample_id_order_match"]:'
    runtime_section = source.index("# ## 10. Stage D visible runtime")
    guarded_sha = source.index(guard, runtime_section)
    targets = source.index("_stage_d_v2_equivalence_targets = {", runtime_section)
    assert guarded_sha < targets
    assert '"status": "SKIPPED_HIDDEN_DYNAMIC"' in source
    assert '"targets": {}' in source


def test_stage_a_selection_and_gates_are_truth_free_and_frozen() -> None:
    source = candidate_text()
    selector = function_node(source, "select_shared_likpf_stage_a_wells")
    string_literals = {
        node.value
        for node in ast.walk(selector)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "TVT" not in string_literals
    assert {"MD", "Z", "GR", "TVT_input"}.issubset(string_literals)
    assert "suffix_tvt" not in ast.unparse(selector)
    stage_a = ast.unparse(function_node(source, "run_shared_likpf_stage_a"))
    assert "thread_counts=(1, 4)" in stage_a
    assert "reruns=2" in stage_a
    assert '"truth_read": False' in source
    config = read_yaml("config.yaml")
    gate = config["validation"]["stage_a_fixed32"]
    assert gate["wells"] == 32
    assert gate["truth_read"] is False
    assert gate["thread_counts"] == [1, 4]
    assert gate["reruns"] == 2
    assert "stage_b_fixed200" not in config["validation"]
    stage_b = config["validation"]["stage_b_fixed32"]
    assert stage_b["wells"] == 32
    assert stage_b["well_set_source"] == "reuse_stage_a_fixed32_selection_exact"
    assert stage_b["well_selection_sha256"] == config["stage_a_result"][
        "selection_sha256"
    ]
    assert stage_b["allow_well_reselection"] is False
    assert config["authorization"]["stage_b_execution_approved"] is True
    assert config["authorization"]["stage_b_package_scope"] == "stage_b_fixed32_only"
    assert config["authorization"]["stage_b_run_scope"] == "stage_b_fixed32_only"
    assert (
        config["authorization"]["stage_b_output_download_scope"]
        == "stage_b_report_metrics_and_execution_log_only"
    )
    assert config["stage_b_execution_plan"]["total_well_bank_generations"] == 64
    assert config["stage_b_execution_plan"]["new_boosters"] == 0
    assert config["execution_count"]["active_scientific_variants"] == 1
    assert config["execution_count"]["new_boosters"] == 0
    assert config["execution_count"]["parent_control_retraining"] == 0


def test_stage_a_has_a_standalone_non_submission_entrypoint() -> None:
    source = STAGE_A_SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    assert "run_shared_likpf_stage_a(" in source
    assert 'count=32' in source
    assert 'thread_counts=(1, 4)' in source
    assert 'reruns=2' in source
    assert 'split="train"' in source
    assert "generate_dynamic_exp413_prediction" not in source
    assert "run_pf_lik_ensemble_scales" not in source
    assert ".to_csv(" not in source
    assert '"submission_file_generated"] = False' in source
    assert "__file__" not in source


def test_stage_b_is_fixed32_truth_frozen_and_non_submission() -> None:
    source = STAGE_B_SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    assert "__file__" not in source
    assert "STAGE_B_WELLS = 32" in source
    assert "STAGE_B_PARTICLES = 500" in source
    assert "STAGE_B_SEEDS = 128" in source
    assert f'EXPECTED_SELECTION_SHA256 = "{read_yaml("config.yaml")["stage_a_result"]["selection_sha256"]}"' in source
    freeze_call = source.index(
        "prediction_content_sha, prediction_gzip_sha = freeze_prediction_frame("
    )
    fold_call = source.index(
        "fold_map, fold_records, fold_assignment_sha = stage_b_reporting_fold_map(wells)"
    )
    truth_call = source.index("truth_by_id = read_stage_b_truth_after_freeze(wells)")
    assert freeze_call < fold_call < truth_call
    truth_reader = ast.unparse(function_node(source, "read_stage_b_truth_after_freeze"))
    assert "if not PREDICTIONS_FROZEN" in truth_reader
    raw_loader = ast.unparse(function_node(source, "load_stage_b_raw_well"))
    assert "usecols=['MD', 'Z', 'GR', 'TVT_input']" in raw_loader
    assert 'usecols=["TVT"]' not in raw_loader
    assert '"submission.csv"' not in source
    assert "kaggle competitions submit" not in source
    assert '"submission_file_generated": False' in source
    assert '"external_submission_performed": False' in source


def test_stage_b_reuses_exact_parent_selector_beam_and_branch_contract() -> None:
    source = STAGE_B_SOURCE.read_text(encoding="utf-8")
    assert source.count("run_pf_lik_ensemble_scales(") == 2
    assert source.count("materialize_shared_likpf_bank(") == 2
    assert source.count("run_beam_ensemble(") == 2
    assert source.count('legacy["beam"]') == 2
    assert "RUN_BIMODAL_DETECTOR = False" in source
    assert "BRANCH_HEDGE_STRENGTH = 0.60" in source
    assert "BRANCH_HEDGE_MIN_MASS = 0.25" in source
    assert "BRANCH_HEDGE_SEPARATION_LOW = 4.0" in source
    assert "BRANCH_HEDGE_SEPARATION_HIGH = 40.0" in source
    assert "BRANCH_HEDGE_CAP_FT = 2.0" in source
    assert "control_after_branch, control_branch_info = apply_branch_hedge(" in source
    assert "candidate_after_branch, candidate_branch_info = apply_branch_hedge(" in source
    assert '"total_well_bank_generations": 2 * len(wells)' in source
    assert '"lightgbm_configs": 0' in source
    assert '"trained_folds": 0' in source
    assert '"new_boosters": 0' in source
    assert '"parent_control_retraining": 0' in source


def test_stage_b_gate_and_source_identity_match_config() -> None:
    config = read_yaml("config.yaml")
    implementation = config["implementation"]
    stage_b = config["validation"]["stage_b_fixed32"]
    source = STAGE_B_SOURCE.read_text(encoding="utf-8")
    assert implementation["stage_b_generator_sha256"] == hashlib.sha256(
        STAGE_B_GENERATOR.read_bytes()
    ).hexdigest()
    assert implementation["stage_b_source_sha256"] == hashlib.sha256(
        STAGE_B_SOURCE.read_bytes()
    ).hexdigest()
    assert implementation["stage_b_source_lines"] == len(source.splitlines())
    notebook = HERE / implementation["stage_b_notebook"]
    assert implementation["stage_b_notebook_sha256"] == hashlib.sha256(
        notebook.read_bytes()
    ).hexdigest()
    assert implementation["stage_b_notebook_cells"] == len(
        json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    )
    assert f"POOLED_MAX_REGRESSION_FT = {stage_b['pooled_max_regression_ft']}" in source
    assert f"FOLD_MAX_REGRESSION_FT = {stage_b['fold_max_regression_ft_for_nonworse']}" in source
    assert f"REQUIRED_NONWORSE_FOLDS = {stage_b['required_nonworse_folds']}" in source
    assert f"FIXED_SCOPE_MAX_REGRESSION_FT = {stage_b['fixed_scope_max_regression_ft']}" in source
    assert f"BY_WELL_DELTA_P95_MAX_FT = {stage_b['by_well_delta_p95_max_ft']}" in source
    assert f"WORST_WELL_DELTA_MAX_FT = {stage_b['worst_well_delta_max_ft']}" in source


def test_stage_b_metric_bundle_does_not_create_duplicate_metric_columns() -> None:
    source = STAGE_B_SOURCE.read_text(encoding="utf-8")
    metric_source = ast.unparse(function_node(source, "metric_bundle"))
    assert "renamed = scored.copy()" in metric_source
    assert "renamed['control_tvt'] = scored[control_column].to_numpy()" in metric_source
    assert "renamed['candidate_tvt'] = scored[candidate_column].to_numpy()" in metric_source
    assert ".rename(" not in metric_source


def test_stage_c_is_removed_and_stage_d_runtime_estimate_is_explicit() -> None:
    config = read_yaml("config.yaml")
    authorization = config["authorization"]
    implementation = config["implementation"]
    runtime_guard = config["runtime_guard"]
    estimate = runtime_guard["visible_runtime_estimation"]

    assert authorization["stage_c_required"] is False
    assert authorization["stage_c_resolution_source"] == (
        "user_request_2026_08_05_use_stage_d_visible_runtime_estimate"
    )
    assert authorization["stage_d_runtime_estimate_source_approved"] is True
    assert authorization["stage_d_package_approved"] is True
    assert authorization["stage_d_package_scope"] == (
        "stage_d_submission_ready_visible_test_only"
    )
    assert authorization["stage_d_run_approved"] is True
    assert authorization["stage_d_run_scope"] == (
        "stage_d_submission_ready_visible_test_only"
    )
    assert implementation["stage_c_implemented"] is False
    assert implementation["stage_c_executed"] is False
    assert implementation["stage_c_status"] == "not_required_by_user_override"
    assert not list(HERE.glob("*stage_c*.py"))
    assert not list(HERE.glob("*stage_c*.ipynb"))

    assert runtime_guard["source"] == "stage_d_visible_test_stagewise_runtime"
    assert runtime_guard["stage_c_required"] is False
    assert runtime_guard["estimate_only"] is True
    assert runtime_guard["hidden_runtime_observed"] is False
    assert estimate["parallel_worker_reference"] == 4
    assert estimate["parallel_lower_formula"] == "stage_seconds_times_200_div_4"
    assert estimate["parallel_upper_formula"] == (
        "stage_seconds_times_200_div_visible_wells"
    )
    assert estimate["sequential_formula"] == (
        "stage_seconds_times_200_div_visible_wells"
    )
    assert estimate["fixed_overhead_policy"] == "add_once"
    assert estimate["estimated_upper_seconds_max"] == 32400
    assert runtime_guard["visible_three_well_total_runtime_is_submission_evidence"] is False
    assert runtime_guard["visible_three_well_stage_runtime_is_estimation_input"] is True


def test_stage_b_v1_error_is_repaired_and_v2_scientific_gate_is_terminal() -> None:
    config = read_yaml("config.yaml")
    result = config["stage_b_result"]
    assert result["status"] == "FAIL"
    assert result["version_1_status"] == "ERROR"
    assert result["version_2_final_status"] == "KernelWorkerStatus.COMPLETE"
    assert result["scientific_gate_evaluated"] is True
    assert result["predictions_frozen_before_truth"] is True
    assert result["prediction_rows"] == 129906
    assert result["version_1_failure_stage"] == (
        "pre_branch_metric_scoring_after_truth_join"
    )
    assert result["version_1_failure_type"] == (
        "duplicate_metric_column_names_after_rename"
    )
    assert result["rerun_performed"] is True
    assert result["stage_b_source_modified_after_failure"] is True
    assert result["scientific_configuration_changed"] is False
    assert result["repair_scope"] == "metric_bundle_duplicate_column_prevention_only"
    assert result["version_2_prediction_matches_v1"] is True
    assert result["pooled_delta_candidate_minus_control"] > 0.02
    assert result["nonworse_folds"] == 2
    assert result["required_nonworse_folds"] == 4
    assert result["raw_gr_observed_delta"] > 0.05
    assert result["by_well_delta_p95"] > 0.25
    assert result["worst_well_delta"] <= 5.0
    assert result["all_and_gate_passed"] is False
    assert result["terminal_decision"] == "reject_without_rescue_or_submission"
    assert config["authorization"]["stage_b_repair_approved"] is True
    assert config["authorization"]["stage_b_rerun_approved"] is True


def test_stage_d_visible_source_and_runtime_contract_match_config() -> None:
    config = read_yaml("config.yaml")
    implementation = config["implementation"]
    source = STAGE_D_SOURCE.read_text(encoding="utf-8")
    notebook = HERE / implementation["stage_d_notebook"]
    runtime = config["runtime"]["kaggle"]

    ast.parse(source)
    assert implementation["stage_d_generator_sha256"] == hashlib.sha256(
        STAGE_D_GENERATOR.read_bytes()
    ).hexdigest()
    assert implementation["stage_d_source_sha256"] == hashlib.sha256(
        STAGE_D_SOURCE.read_bytes()
    ).hexdigest()
    assert implementation["stage_d_source_lines"] == len(source.splitlines())
    assert implementation["stage_d_notebook_sha256"] == hashlib.sha256(
        notebook.read_bytes()
    ).hexdigest()
    assert implementation["stage_d_notebook_cells"] == len(
        json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    )
    assert "STAGE_D_VISIBLE_STARTED = time.time()" in source
    assert '"stage": "stage_d_submission_ready_visible_test"' in source
    assert '"hidden_runtime_observed": False' in source
    assert '"estimated_pass_not_hidden_runtime_guarantee"' in source
    assert '"external_submission_performed": False' in source
    assert 'FINAL_SUBMISSION_PATH = WORKING_DIR / "submission.csv"' in source
    assert (
        'STAGE_D_VISIBLE_HJYACT_CANDIDATE_SHA256 = '
        '"6b3e1c576afc47f065bdcce12a09f4361a6bb97c63667630f4f5ab1e64fa37b3"'
        in source
    )
    assert 'visible_reference_checks["hjyact_parent_exact_match_required"] = False' in source
    assert 'visible_reference_checks["hjyact_candidate_submission_match"]' in source
    assert "visible hjyact-v2 parity failed" not in source
    assert runtime["stage_d_visible"]["include_experiment_sources"] is False
    assert len(runtime["stage_d_visible"]["bootstrap_dependency_files"]) == 25
    assert len(runtime["stage_d_visible_dataset_sources"]) == 6
    assert len(runtime["stage_d_visible_kernel_sources"]) == 11


def test_stage_d_kaggle_package_is_private_offline_t4_and_pinned() -> None:
    config = read_yaml("config.yaml")
    implementation = config["implementation"]
    package = HERE / "kaggle" / "stage_d_visible"
    notebook = package / implementation["stage_d_notebook"]
    metadata_path = package / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    packaged = json.loads(notebook.read_text(encoding="utf-8"))

    assert implementation["stage_d_kaggle_package_notebook_sha256"] == (
        hashlib.sha256(notebook.read_bytes()).hexdigest()
    )
    assert implementation["stage_d_kaggle_metadata_sha256"] == (
        hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    )
    assert implementation["stage_d_kaggle_package_notebook_bytes"] == notebook.stat().st_size
    assert implementation["stage_d_kaggle_package_notebook_cells"] == len(
        packaged["cells"]
    )
    assert notebook.stat().st_size < 1_000_000
    assert metadata["id"] == "kentookumura/exp514-shared-likpf-stage-d-visible"
    assert metadata["title"] == "exp514 shared likpf stage d visible"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_internet"] is False
    assert metadata["run_on_push"] is True
    assert len(metadata["dataset_sources"]) == 6
    assert len(metadata["kernel_sources"]) == 11
    bootstrap = "".join(packaged["cells"][0]["source"])
    assert "_KAGGLE_BOOTSTRAP_STARTED" in bootstrap
    assert "exp413_runtime/config.yaml" in bootstrap
    assert "stage_d_visible.py" not in bootstrap


def test_candidate_hash_and_authorization_boundary_match_config() -> None:
    config = read_yaml("config.yaml")
    implementation = config["implementation"]
    observed = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    assert config["experiment"]["route"] == "ensemble"
    assert config["authorization"]["implementation_approved"] is True
    assert config["authorization"]["canonical_notebook_adoption_approved"] is False
    assert config["authorization"]["kaggle_package_approved"] is True
    assert config["authorization"]["kaggle_package_scope"] == "stage_a_fixed32_only"
    assert config["authorization"]["kaggle_run_approved"] is True
    assert config["authorization"]["kaggle_run_scope"] == "stage_a_fixed32_only"
    assert config["authorization"]["output_download_approved"] is True
    assert (
        config["authorization"]["output_download_scope"]
        == "stage_a_report_metrics_and_execution_log_only"
    )
    assert config["authorization"]["competition_submission_approved"] is False
    assert implementation["candidate_source_sha256"] == observed
    assert implementation["candidate_source_lines"] == len(
        CANDIDATE.read_text(encoding="utf-8").splitlines()
    )
    assert implementation["stage_a_source_sha256"] == hashlib.sha256(
        STAGE_A_SOURCE.read_bytes()
    ).hexdigest()
    candidate_notebook = HERE / implementation["candidate_notebook"]
    stage_a_notebook = HERE / implementation["stage_a_notebook"]
    assert implementation["candidate_notebook_sha256"] == hashlib.sha256(
        candidate_notebook.read_bytes()
    ).hexdigest()
    assert implementation["stage_a_notebook_sha256"] == hashlib.sha256(
        stage_a_notebook.read_bytes()
    ).hexdigest()
    assert implementation["candidate_notebook_cells"] == len(
        json.loads(candidate_notebook.read_text(encoding="utf-8"))["cells"]
    )
    assert implementation["stage_a_notebook_cells"] == len(
        json.loads(stage_a_notebook.read_text(encoding="utf-8"))["cells"]
    )

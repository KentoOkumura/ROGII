from __future__ import annotations

import inspect
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp270_exact_hmm_posterior_mode_candidate_audit"
TRAIN = EXP_DIR / "exp270_exact_hmm_posterior_mode_candidate_audit_train.py"
PARENT_HMM = (
    ROOT
    / "experiments"
    / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    / "exact_hmm_smoother.py"
)


def load_namespace() -> dict[str, object]:
    previous = os.environ.get("EXP270_IMPORT_ONLY")
    os.environ["EXP270_IMPORT_ONLY"] = "1"
    try:
        return runpy.run_path(str(TRAIN))
    finally:
        if previous is None:
            os.environ.pop("EXP270_IMPORT_ONLY", None)
        else:
            os.environ["EXP270_IMPORT_ONLY"] = previous


def transition_tables(
    dm: np.ndarray,
    dz: np.ndarray,
    step: float,
    rates: np.ndarray,
    sig_r: float,
    sig_p: float,
    mom: float,
) -> tuple[list[np.ndarray], list[tuple[int, np.ndarray]]]:
    rate_tables: list[np.ndarray] = []
    position_tables: list[tuple[int, np.ndarray]] = []
    rate_step = rates[1] - rates[0]
    for time_index in range(len(dm)):
        rate = np.full((len(rates), len(rates)), -np.inf)
        rate_var_cells = (sig_r * np.sqrt(dm[time_index]) / rate_step) ** 2
        for previous_rate in range(len(rates)):
            mean_move = -(1.0 - mom) * rates[previous_rate] * dm[time_index] / rate_step
            plus = max(0.5 * (rate_var_cells + mean_move), 1e-12)
            minus = max(0.5 * (rate_var_cells - mean_move), 1e-12)
            if plus + minus > 0.9:
                scale = 0.9 / (plus + minus)
                plus *= scale
                minus *= scale
            values = np.log([minus, 1.0 - plus - minus, plus])
            lower = max(0, previous_rate - 1)
            upper = min(len(rates), previous_rate + 2)
            for current_rate in range(lower, upper):
                rate[previous_rate, current_rate] = values[current_rate - previous_rate + 1]
        rate_tables.append(rate)
        # The position table depends on current rate, so store all five kernels.
        packed = []
        for current_rate in range(len(rates)):
            mu = rates[current_rate] * dm[time_index] - dz[time_index]
            b0 = int(np.floor(mu / step + 0.5))
            offsets = b0 - 2 + np.arange(5)
            sigma_position = max(sig_p, 0.35 * step)
            log_probability = -0.5 * ((offsets * step - mu) / sigma_position) ** 2
            log_probability -= np.log(np.exp(log_probability).sum())
            packed.append(np.column_stack([offsets, log_probability]))
        position_tables.append((time_index, np.asarray(packed)))
    return rate_tables, position_tables


def exhaustive_joint_paths(
    emission: np.ndarray,
    dm: np.ndarray,
    dz: np.ndarray,
    step: float,
    rates: np.ndarray,
    sig_r: float,
    sig_p: float,
    start_p: float,
    start_sig: float,
    r0: float,
    r0_sig: float,
    lam: float,
    mom: float,
) -> list[tuple[float, tuple[int, ...], tuple[int, ...]]]:
    rate_tables, position_tables = transition_tables(dm, dz, step, rates, sig_r, sig_p, mom)
    paths: list[tuple[float, int, int, tuple[int, ...], tuple[int, ...]]] = []
    for position in range(emission.shape[1]):
        position_prior = -0.5 * (((position - start_p) * step) / start_sig) ** 2
        if position_prior < -60.0:
            continue
        for rate in range(len(rates)):
            rate_prior = -0.5 * ((rates[rate] - r0) / r0_sig) ** 2
            paths.append((position_prior + rate_prior, position, rate, (), ()))
    for time_index in range(len(emission)):
        next_paths = []
        packed = position_tables[time_index][1]
        for score, previous_position, previous_rate, position_path, rate_path in paths:
            for current_rate in range(len(rates)):
                rate_score = rate_tables[time_index][previous_rate, current_rate]
                if not np.isfinite(rate_score):
                    continue
                for offset_float, position_score in packed[current_rate]:
                    current_position = previous_position + int(offset_float)
                    if 0 <= current_position < emission.shape[1]:
                        next_paths.append(
                            (
                                score
                                + rate_score
                                + float(position_score)
                                + lam * float(emission[time_index, current_position]),
                                current_position,
                                current_rate,
                                position_path + (current_position,),
                                rate_path + (current_rate,),
                            )
                        )
        paths = next_paths
    return sorted(
        [(score, positions, rate_path) for score, _, _, positions, rate_path in paths],
        key=lambda item: item[0],
        reverse=True,
    )


def test_exact_joint_top5_matches_exhaustive_small_trellis() -> None:
    namespace = load_namespace()
    decoder = namespace["_hmm2_topk"]
    emission = np.asarray([[-1.2, -0.1, -2.7], [-2.0, -0.4, -1.1]], dtype=np.float32)
    dm = np.asarray([1.0, 1.3], dtype=np.float64)
    dz = np.asarray([0.08, -0.03], dtype=np.float64)
    step = 0.35
    rates = np.asarray([-0.08, 0.0, 0.08], dtype=np.float64)
    args = (
        emission,
        dm,
        dz,
        step,
        rates,
        0.02,
        0.12,
        1.1,
        0.75,
        0.01,
        0.08,
        1.0,
        0.998,
    )
    scores, position_paths, rate_paths = decoder(*args, 5)
    expected = exhaustive_joint_paths(*args)
    np.testing.assert_allclose(scores, [row[0] for row in expected[:5]], atol=2e-5)
    assert [tuple(row) for row in position_paths] == [row[1] for row in expected[:5]]
    assert [tuple(row) for row in rate_paths] == [row[2] for row in expected[:5]]


def test_forward_backward_kernel_matches_exp209_exactly() -> None:
    namespace = load_namespace()
    parent = runpy.run_path(str(PARENT_HMM))
    emission = np.asarray([[-1.2, -0.1, -2.7], [-2.0, -0.4, -1.1]], dtype=np.float32)
    args = (
        emission,
        np.asarray([1.0, 1.3], dtype=np.float64),
        np.asarray([0.08, -0.03], dtype=np.float64),
        0.35,
        np.asarray([-0.08, 0.0, 0.08], dtype=np.float64),
        0.02,
        0.12,
        1.1,
        0.75,
        0.01,
        0.08,
        1.0,
        0.998,
    )
    actual_posterior, actual_log_likelihood = namespace["_hmm2_fb"](*args)
    expected_posterior, expected_log_likelihood = parent["_hmm2_fb"](*args)
    np.testing.assert_array_equal(actual_posterior, expected_posterior)
    assert actual_log_likelihood == expected_log_likelihood


def test_posterior_mean_storage_matches_exp209_float32_contract() -> None:
    namespace = load_namespace()
    storage_values = namespace["candidate_storage_values"]
    values = np.asarray([10000.123456789, 16000.987654321], dtype=np.float64)

    posterior_mean = storage_values("posterior_mean", values)
    marginal_map = storage_values("marginal_map", values)

    assert posterior_mean.dtype == np.float32
    np.testing.assert_array_equal(posterior_mean, values.astype(np.float32))
    assert marginal_map.dtype == np.float64
    np.testing.assert_array_equal(marginal_map, values)


def test_tvt_sequence_dedup_collapses_rate_only_variants_without_backfill() -> None:
    namespace = load_namespace()
    deduplicate = namespace["deduplicate_tvt_paths"]
    scores = np.asarray([10.0, 9.0, 8.0, 7.0, 6.0])
    position = np.asarray(
        [[0, 1, 2], [0, 1, 2], [0, 1, 1], [0, 1, 1], [1, 1, 2]],
        dtype=np.int32,
    )
    rate = np.asarray(
        [[0, 0, 0], [1, 1, 1], [0, 1, 1], [2, 1, 1], [1, 1, 2]],
        dtype=np.int16,
    )
    unique, audit = deduplicate(scores, position, rate)
    assert len(unique) == 3
    assert [item["joint_rank"] for item in unique] == [1, 3, 5]
    assert audit[1]["status"] == "duplicate_tvt_path"
    assert audit[3]["status"] == "duplicate_tvt_path"


def test_block_oracle_uses_fixed_contiguous_rows_and_is_transient() -> None:
    namespace = load_namespace()
    oracle = namespace["oracle_prediction"]
    frame = pd.DataFrame(
        {
            "well": ["w"] * 4,
            "true_tvt_readout_only": [0.0, 0.0, 10.0, 10.0],
            "a": [0.0, 0.0, 0.0, 0.0],
            "b": [10.0, 10.0, 10.0, 10.0],
        }
    )
    row_prediction, _ = oracle(frame, ("a", "b"), "row")
    block_prediction, _ = oracle(frame, ("a", "b"), "block", 2)
    well_prediction, _ = oracle(frame, ("a", "b"), "well")
    np.testing.assert_array_equal(row_prediction, [0.0, 0.0, 10.0, 10.0])
    np.testing.assert_array_equal(block_prediction, row_prediction)
    np.testing.assert_array_equal(well_prediction, [0.0, 0.0, 0.0, 0.0])
    assert not any("oracle" in column for column in frame.columns)


def test_config_and_target_free_generation_contract() -> None:
    namespace = load_namespace()
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    namespace["validate_scientific_contract"](config)
    assert config["lineage"]["parent"].startswith("exp209_")
    assert config["experiment"]["route"] == "pf_beam"
    assert config["model"]["decoder"]["joint_top_k"] == 5
    assert config["model"]["decoder"]["deduplicate_by"] == "tvt_grid_index_sequence"
    assert config["model"]["decoder"]["backfill_after_dedup"] is False
    assert config["model"]["decoder"]["persist_full_posterior"] is False
    assert config["model"]["decoder"]["persist_rate_paths"] is False
    assert config["audit"]["oracle_block_rows"] == [128, 256, 512]
    assert config["execution"]["active_hmm_variants"] == 1
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["outer_workers"] == 1
    assert config["execution"]["shard_count"] == 2
    assert config["execution"]["total_hmm_well_runs"] == 773
    assert config["execution"]["streaming_candidate_write"] is True
    assert config["execution"]["stream_flush_every_wells"] > 0
    assert config["execution"]["parity_chunksize_rows"] > 0
    assert config["execution"]["frame_write_chunksize_rows"] > 0
    assert config["execution"]["binary_hash_chunk_bytes"] > 0
    assert config["execution"]["gzip_compresslevel"] == 1
    assert config["execution"]["gzip_mtime"] == 0
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["control_or_parent_retraining"] is False
    assert sum(spec["expected_rows"] for spec in config["data"]["shard_outputs"]) == 3783989
    assert sum(spec["expected_wells"] for spec in config["data"]["shard_outputs"]) == 773
    assert config["inference"]["enabled"] is False
    signature = inspect.signature(namespace["run_hmm_posterior_modes"])
    assert "truth" not in signature.parameters
    assert "target" not in signature.parameters
    source = inspect.getsource(namespace["build_candidate_rows_for_well"])
    assert source.index('horizontal.drop(columns=["TVT"])') < source.index("true_tvt =")


def test_compact_backpointer_budget_is_one_byte_per_state_rank_transition() -> None:
    namespace = load_namespace()
    estimate = namespace["estimate_backpointer_bytes"]
    assert estimate(10, 7, 3, 5) == 9 * 7 * 3 * 5
    source = TRAIN.read_text()
    assert "np.uint8" in source
    assert "from settings import" not in source
    assert "__file__" not in source
    assert "# ## Contents" in source


def test_train_data_dir_supports_competition_mount_layout(tmp_path: Path) -> None:
    namespace = load_namespace()
    resolver = namespace["train_data_dir"]
    input_root = tmp_path / "input"
    train = input_root / "competitions" / "rogii-wellbore-geology-prediction" / "train"
    train.mkdir(parents=True)
    (train / "well__horizontal_well.csv").write_text("MD\n1\n")
    globals_dict = resolver.__globals__
    original = globals_dict["KAGGLE_INPUT_ROOT"]
    globals_dict["KAGGLE_INPUT_ROOT"] = input_root
    try:
        assert resolver({"data": {"train_dir": "data/raw/train"}}) == train
    finally:
        globals_dict["KAGGLE_INPUT_ROOT"] = original


def test_target_free_stable_two_shard_partition() -> None:
    namespace = load_namespace()
    stable_well_shard = namespace["stable_well_shard"]
    assignments = [stable_well_shard(f"well_{index:04d}", 2) for index in range(200)]
    assert set(assignments) == {0, 1}
    assert assignments == [
        stable_well_shard(f"well_{index:04d}", 2) for index in range(200)
    ]


def test_all_train_sources_are_self_contained_and_mode_pinned() -> None:
    expected_modes = {
        "exp270_exact_hmm_posterior_mode_candidate_audit_train.py": "aggregate",
        "exp270_exact_hmm_posterior_mode_candidate_audit_train_variant0.py": "shard0",
        "exp270_exact_hmm_posterior_mode_candidate_audit_train_variant1.py": "shard1",
    }
    for filename, mode in expected_modes.items():
        source = (EXP_DIR / filename).read_text()
        assert f'RUN_KIND_OVERRIDE = "{mode}"' in source
        assert "from settings import" not in source
        assert "__file__" not in source
        assert "# ## Contents" in source
        assert '"oracle_prediction_persisted": False' in source
        assert '"selector_persisted": False' in source


def test_aggregate_requires_fixed_shard_sha_before_reading() -> None:
    namespace = load_namespace()
    required_sha = namespace["required_sha"]
    with np.testing.assert_raises_regex(ValueError, "fixed 64-character"):
        required_sha({"expected_raw_sha256": None}, "expected_raw_sha256", "shard0")
    digest = "a" * 64
    assert (
        required_sha({"expected_raw_sha256": digest}, "expected_raw_sha256", "shard0")
        == digest
    )


def test_chunked_array_bundle_sha_matches_in_memory_contract(tmp_path: Path) -> None:
    namespace = load_namespace()
    candidates = tuple(namespace["ALL_CANDIDATES"])
    rows = 13
    frame = pd.DataFrame(
        {
            "row_idx": np.arange(rows, dtype=np.int64) + 7,
            **{
                candidate: np.linspace(index, index + 1.0, rows, dtype=np.float64)
                for index, candidate in enumerate(candidates)
            },
        }
    )
    candidate_matrix = frame[list(candidates)].to_numpy(np.float32)
    row_idx = frame["row_idx"].to_numpy(np.int64)
    expected = namespace["array_bundle_sha256"](
        row_idx=row_idx,
        candidates=candidate_matrix,
    )
    assert (
        namespace["array_bundle_sha256_from_frame"](frame, chunk_rows=4) == expected
    )

    candidate_path = tmp_path / "candidates.float32.part"
    row_idx_path = tmp_path / "row_idx.int64.part"
    candidate_matrix.tofile(candidate_path)
    row_idx.tofile(row_idx_path)
    assert (
        namespace["array_bundle_sha256_from_binary_parts"](
            candidate_path=candidate_path,
            row_idx_path=row_idx_path,
            rows=rows,
            candidate_count=len(candidates),
            chunk_bytes=37,
        )
        == expected
    )


def test_deterministic_gzip_and_linear_chunked_parity(tmp_path: Path) -> None:
    namespace = load_namespace()
    candidate = pd.DataFrame(
        {
            "id": ["w1_2", "w1_3", "w1_4"],
            "well": ["w1", "w1", "w1"],
            "posterior_mean": [10.25, 10.5, 10.75],
        }
    )
    candidate_path = tmp_path / "candidate.csv.gz"
    candidate_copy_path = tmp_path / "candidate_copy.csv.gz"
    for path in (candidate_path, candidate_copy_path):
        namespace["write_dataframe_gzip_deterministic"](
            path,
            candidate,
            chunk_rows=2,
            compresslevel=1,
            mtime=0,
        )
    assert namespace["sha256_path"](candidate_path) == namespace["sha256_path"](
        candidate_copy_path
    )

    control = pd.DataFrame(
        {
            "id": ["w0_1", "w1_2", "w1_3", "w1_4", "w2_1"],
            "well": ["w0", "w1", "w1", "w1", "w2"],
            "hmm_mean_tvt": [1.0, 10.25, 10.5, 10.75, 20.0],
        }
    )
    control_path = tmp_path / "control.csv.gz"
    namespace["write_dataframe_gzip_deterministic"](
        control_path,
        control,
        chunk_rows=2,
        compresslevel=1,
        mtime=0,
    )
    config = {
        "data": {
            "exp209_hmm_control": {
                "filename": control_path.name,
                "candidates": [str(control_path)],
                "expected_decompressed_sha256": namespace["sha256_gzip_decompressed"](
                    control_path
                ),
                "prediction_column": "hmm_mean_tvt",
                "parity_atol_ft": 1e-5,
            }
        },
        "execution": {"parity_chunksize_rows": 2},
    }
    parity, resolved, _ = namespace["validate_posterior_mean_parity_batches"](
        namespace["iter_candidate_csv_batches"](candidate_path, 2),
        config,
        expected_rows=3,
        selected_wells={"w1"},
    )
    assert resolved == control_path
    assert parity["passed"] is True
    assert parity["rows"] == 3
    assert parity["max_abs_diff_ft"] == 0.0
    assert parity["alignment"] == "linear_ordered_id_well_chunks"


def test_shard_generation_source_is_memory_bounded() -> None:
    namespace = load_namespace()
    source = inspect.getsource(namespace["run_shard_generation"])
    assert "frames.append" not in source
    assert "pd.concat" not in source
    assert "candidate_text_file" in source
    assert "array_bundle_sha256_from_binary_parts" in source
    assert "validate_posterior_mean_parity_batches" in source
    assert "log_stage" in source
    assert "np.setdiff1d" not in TRAIN.read_text()

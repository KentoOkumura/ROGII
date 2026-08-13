from __future__ import annotations

import ast
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

ROOT = Path(__file__).resolve().parents[3]
EXP = "exp452_scale5_likpf_direct_public_lb_audit"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_inference.py"
EXP413_DIR = ROOT / "experiments" / "exp413_scale5_likpf_full_replacement_on_exp335"
EXP073_SOURCE = (
    ROOT
    / "experiments"
    / "exp073_gpu_reproducibility_guard_for_exp063_full_replay"
    / "public_notebook_replay_audit.py"
)
PUBLIC_REFERENCE = (
    EXP413_DIR
    / "kaggle"
    / "output"
    / "inference_v4_hidden_compatible"
    / "artifacts"
    / "exp263_replay_for_exp145.csv.gz"
)


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inference() -> ModuleType:
    previous = os.environ.get("EXP452_IMPORT_ONLY")
    os.environ["EXP452_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp452_inference_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP452_IMPORT_ONLY", None)
        else:
            os.environ["EXP452_IMPORT_ONLY"] = previous


@pytest.fixture()
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    return ast.dump(node, include_attributes=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_contract_is_one_candidate_zero_model_and_fail_closed(
    inference: ModuleType,
    config: dict,
) -> None:
    contract = inference.validate_frozen_contract(config)
    assert contract["candidate"] == "likpf_scale_5_x1p0"
    assert contract["particles"] == 500
    assert contract["seeds"] == 128
    assert contract["temperature"] == 5.0
    assert contract["alternative_aggregations_generated"] == 0
    assert contract["model_configs"] == 0
    assert contract["trained_folds"] == 0
    assert contract["boosters"] == 0
    assert contract["parent_control_reruns"] == 0

    assert (
        inference.validate_frozen_contract(
            config,
            require_execution_approval=True,
        )["candidate"]
        == "likpf_scale_5_x1p0"
    )
    assert config["execution"]["competition_submission_approved"] is False
    assert config["submission_plan"]["competition_submission_approved"] is False

    unapproved = copy.deepcopy(config)
    unapproved["execution"]["kaggle_run_approved"] = False
    with pytest.raises(RuntimeError, match="not approved"):
        inference.validate_frozen_contract(
            unapproved,
            require_execution_approval=True,
        )


def test_parent_sources_and_extracted_pf_kernel_are_sha_pinned(
    inference: ModuleType,
    config: dict,
) -> None:
    exp413_source = (
        EXP413_DIR / "exp413_scale5_likpf_full_replacement_on_exp335_current_test_inference.py"
    )
    assert sha256(exp413_source) == config["data"]["pinned_source"]["sha256"]
    execution_config_sha = config["data"]["pinned_source"]["config_sha256"]
    assert len(execution_config_sha) == 64
    int(execution_config_sha, 16)
    current_parent_config = yaml.safe_load((EXP413_DIR / "config.yaml").read_text())
    assert current_parent_config["lineage"]["steering"].startswith("../../docs/legacy/steering/")
    assert sha256(EXP073_SOURCE) == config["data"]["pinned_source"]["pf_source_sha256"]
    assert config["data"]["pinned_source"]["pf_source_sha256"] == (
        inference.EXPECTED_EXP073_PF_SOURCE_SHA256
    )

    for function_name in ("stable_seed", "_interp1", "_grid", "_pf_lik_allseeds"):
        assert function_ast(SOURCE, function_name) == function_ast(
            EXP073_SOURCE,
            function_name,
        )


def test_stable_seed_namespace_matches_exp413_v4_records(
    inference: ModuleType,
) -> None:
    expected = {
        "000d7d20": 805188988,
        "00bbac68": 829597097,
        "00e12e8b": 1365511604,
    }
    actual = {well: inference.stable_seed("likpf", "test", well) for well in expected}
    assert actual == expected
    assert len(set(actual.values())) == len(actual)


def test_numba_seeded_kernel_is_repeatable(inference: ModuleType) -> None:
    md = np.arange(1.0, 7.0, dtype=np.float64)
    z = np.zeros_like(md)
    gr = np.linspace(45.0, 55.0, len(md), dtype=np.float64)
    grid = np.linspace(40.0, 60.0, 101, dtype=np.float64)
    args = (
        md,
        z,
        gr,
        grid,
        0.0,
        0.2,
        20.0,
        10_000.0,
        0.0,
        32,
        4,
        123456,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )
    pred_a, ll_a = inference._pf_lik_allseeds(*args)
    pred_b, ll_b = inference._pf_lik_allseeds(*args)
    np.testing.assert_array_equal(pred_a, pred_b)
    np.testing.assert_array_equal(ll_a, ll_b)


def test_submission_restores_sample_order_and_forbids_fallback(
    inference: ModuleType,
) -> None:
    sample = pd.DataFrame(
        {
            "id": ["well_a_2", "well_a_0", "well_a_1"],
            "tvt": [0.0, 0.0, 0.0],
        }
    )
    candidate = pd.DataFrame(
        {
            "id": ["well_a_0", "well_a_1", "well_a_2"],
            "well": ["well_a"] * 3,
            "well_row_idx": np.arange(3, dtype=np.int32),
            "candidate_tvt": np.asarray([10.0, 11.0, 12.0], dtype=np.float32),
        }
    )
    submission = inference.build_submission(sample, candidate)
    assert submission["id"].tolist() == sample["id"].tolist()
    np.testing.assert_array_equal(
        submission["tvt"].to_numpy(np.float32),
        np.asarray([12.0, 10.0, 11.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="do not match"):
        inference.build_submission(sample, candidate.iloc[:-1])


@pytest.mark.skipif(
    not PUBLIC_REFERENCE.is_file(),
    reason="requires the Git-ignored exp413 public-reference artifact",
)
def test_public_reference_sha_and_logical_candidate_are_frozen(
    inference: ModuleType,
    config: dict,
) -> None:
    reference_config = config["data"]["public_reference"]
    assert sha256(PUBLIC_REFERENCE) == reference_config["expected_file_sha256"]
    assert (
        inference.sha256_gzip_content(PUBLIC_REFERENCE)
        == reference_config["expected_decompressed_sha256"]
    )
    reference = inference.load_public_reference(PUBLIC_REFERENCE)
    assert (
        inference.frame_content_sha256(reference)
        == reference_config["expected_candidate_content_sha256"]
    )
    assert reference["candidate_tvt"].dtype == np.float32


@pytest.mark.skipif(
    os.environ.get("EXP452_RUN_FULL_PARITY") != "1",
    reason="set EXP452_RUN_FULL_PARITY=1 for the 500-particle x 128-seed public parity run",
)
def test_full_public_scale5_surface_has_exact_exp413_v4_parity(
    inference: ModuleType,
    config: dict,
) -> None:
    data_root = ROOT / "data" / "raw"
    sample = pd.read_csv(data_root / "sample_submission.csv", dtype={"id": str})
    identity = inference.parse_identity(sample)
    wells = sorted(identity["well"].unique().tolist())
    inference.configure_public_runtime(
        data_dir=data_root,
        output_dir=EXP_DIR / "artifacts",
        n_jobs=int(config["runtime"]["num_workers"]),
        pf_seeds=128,
        pf_particles=500,
    )
    inference.warm_up_likpf_kernel()
    candidate, audits = inference.build_scale5_surface(
        wells,
        n_jobs=int(config["runtime"]["num_workers"]),
    )
    assert {item["well"] for item in audits} == set(wells)
    assert all(item["materialized_aggregations"] == [inference.ACTIVE_VARIANT] for item in audits)
    assert len(candidate) == len(sample)
    assert set(candidate["id"].astype(str)) == set(sample["id"].astype(str))
    assert inference.build_submission(sample, candidate)["tvt"].notna().all()
    parity = inference.validate_public_reference_parity(candidate, PUBLIC_REFERENCE)
    assert parity["status"] == "passed_public_float32_exact_parity"
    assert parity["float32_max_abs_ft"] == 0.0
    assert parity["candidate_content_sha256"] == (
        inference.EXPECTED_PUBLIC_CANDIDATE_CONTENT_SHA256
    )

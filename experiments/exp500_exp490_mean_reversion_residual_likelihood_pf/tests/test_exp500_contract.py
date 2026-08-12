from __future__ import annotations

import gzip
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

EXP = "exp500_exp490_mean_reversion_residual_likelihood_pf"
EXP_DIR = Path(__file__).resolve().parents[1]
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
CONFIG_PATH = EXP_DIR / "config.yaml"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train() -> ModuleType:
    previous = os.environ.get("EXP500_IMPORT_ONLY")
    os.environ["EXP500_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp500_stage0_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP500_IMPORT_ONLY", None)
        else:
            os.environ["EXP500_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_stage1_override_preserves_stage0_fail_and_execution_counts(
    train: ModuleType, config: dict
) -> None:
    contract = train.validate_scientific_contract(config)
    assert config["experiment"]["status"] == "stage1_fail_closed_under_override"
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["scope"] == "stage0_and_stage1_single_variant"
    assert config["implementation"]["implementation_approval_received"] is True
    assert config["implementation"]["canonical_notebook_adopted"] is True
    assert config["implementation"]["kaggle_package_created"] is True
    assert config["implementation"]["stage_0_implemented"] is True
    assert config["implementation"]["stage_1_implemented"] is True
    assert config["implementation"]["stage_1_override_received"] is True
    assert config["implementation"]["stage_1_override_preserves_stage_0_fail"] is True
    assert config["implementation"]["stage_0_completed"] is True
    assert config["implementation"]["stage_0_all_pass"] is False
    assert config["implementation"]["stage_0_technical_all_pass"] is True
    assert config["implementation"]["stage_0_mechanism_all_pass"] is False
    assert config["execution"]["run_stage_0"] is False
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["stage_0_run_approved"] is True
    assert config["execution"]["stage_1_implementation_approved"] is True
    assert config["execution"]["stage_1_run_approved"] is True
    assert config["execution"]["stage_1_stage0_gate_override_approved"] is True
    assert config["execution"]["run_stage_1_shard"] is False
    assert config["execution"]["run_stage_1_merge"] is False
    assert config["execution"]["selected_stage_1_shard_index"] is None
    assert config["execution"]["stage_0"]["status"] == "stage0_fail_closed"
    assert (
        config["execution"]["stage_0"]["next_action"]
        == "terminal_close_without_same_fixed44_rescue"
    )
    assert config["inference"]["enabled"] is False
    assert contract["active_variants"] == [
        "k16_half_life_mean_reverting_residual_likpf"
    ]
    assert contract["execution_counts"]["stage_0.candidate_pf_well_runs"] == 44
    assert contract["execution_counts"]["stage_0.seed_well_trajectories"] == 5632
    assert contract["execution_counts"]["stage_0.particle_starts"] == 2816000
    assert contract["execution_counts"]["stage_1.candidate_pf_well_runs"] == 773
    assert contract["execution_counts"]["stage_1.seed_well_trajectories"] == 98944
    assert contract["execution_counts"]["stage_1.particle_starts"] == 49472000
    assert contract["execution_counts"]["gpu_runs"] == 0
    assert contract["stage_1_user_override"]["stage_0_fail_preserved"] is True
    assert config["runtime"]["kaggle"]["stage_1_shard_kernel_ids"] == [
        f"kentookumura/exp500-mean-revert-resid-likpf-full-shard{index}"
        for index in range(4)
    ]
    assert config["runtime"]["kaggle"]["train_kernel_sources"] == [
        "kentookumura/exp226-k16-kappa-repro-train",
        "kentookumura/exp209-joint-exact-parity-train",
        "kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train",
    ]
    assert config["data"]["exp226_oof_geometry"]["postfreeze_columns"] == [
        "fold",
        "tvt_pred",
    ]

    train.validate_execution_contract(config, require_run_approval=True)
    broken = yaml.safe_load(yaml.safe_dump(config))
    broken["execution"]["kaggle_push_approved"] = False
    with pytest.raises(RuntimeError, match="Kaggle push is not approved"):
        train.validate_execution_contract(broken, require_run_approval=True)
    broken = yaml.safe_load(yaml.safe_dump(config))
    broken["implementation"]["stage_0_all_pass"] = True
    with pytest.raises(ValueError, match="Stage 0 fail must remain preserved"):
        train.validate_execution_contract(broken)


def test_stage1_selection_is_mutually_exclusive_and_bounded(
    train: ModuleType, config: dict
) -> None:
    assert train.selected_stage1(config) is None
    shard = yaml.safe_load(yaml.safe_dump(config))
    shard["execution"]["run_stage_1_shard"] = True
    shard["execution"]["selected_stage_1_shard_index"] = 2
    assert train.selected_stage1(shard) == "stage1_shard"
    shard["execution"]["selected_stage_1_shard_index"] = 4
    with pytest.raises(ValueError, match=r"must be in \[0, 3\]"):
        train.selected_stage1(shard)
    both = yaml.safe_load(yaml.safe_dump(config))
    both["execution"]["run_stage_1_shard"] = True
    both["execution"]["run_stage_1_merge"] = True
    both["execution"]["selected_stage_1_shard_index"] = 0
    with pytest.raises(ValueError, match="exactly one Stage 1"):
        train.selected_stage1(both)
    merge = yaml.safe_load(yaml.safe_dump(config))
    merge["execution"]["run_stage_1_merge"] = True
    assert train.selected_stage1(merge) == "stage1_merge"


def test_stage1_shard_hash_is_stable(train: ModuleType, config: dict) -> None:
    wells = ["alpha", "beta", "gamma", "delta"]
    observed = [train.shard_index(well) for well in wells]
    expected = [
        int.from_bytes(
            hashlib.sha256(f"exp500::full_pf_shard::{well}".encode()).digest()[:8],
            "little",
        )
        % 4
        for well in wells
    ]
    assert observed == expected
    assert config["execution"]["stage_1"]["expected_shard_wells"] == [200, 182, 181, 210]
    assert config["execution"]["stage_1"]["expected_shard_suffix_rows"] == [
        983418,
        906216,
        898293,
        996062,
    ]


def test_k16_destination_ownership_and_exact_half_life(train: ModuleType) -> None:
    unknown_md = 1000.0 + np.cumsum(np.linspace(0.5, 2.0, 37))
    result = train.k16_segment_half_life(
        unknown_md, last_known_md=999.0, segment_count=16
    )
    assert np.all(result["dmd"] > 0.0)
    assert np.array_equal(np.unique(result["segment_id"]), np.arange(16))
    assert result["segment_id"][0] == 0
    assert result["segment_id"][-1] == 15
    assert np.allclose(result["segment_cumulative_rho"], 0.5, rtol=0.0, atol=1e-12)
    assert np.all((result["rho"] > 0.0) & (result["rho"] <= 1.0))


def test_zero_state_and_rho_one_exp486_parity(train: ModuleType) -> None:
    dmd = np.asarray([1.0, 2.0, 3.0])
    rho = np.asarray([0.9, 0.8, 0.7])
    assert train.zero_state_geometry_identity(dmd, rho)["pass"] is True
    parity = train.rho_one_exp486_transition_parity()
    assert parity["pass"] is True
    assert parity["rate_float32_bitwise_equal"] is True
    assert parity["offset_float32_bitwise_equal"] is True


def test_stable_rng_is_independent_of_intervening_well_order(train: ModuleType) -> None:
    args = (
        np.asarray([1.0, 1.5, 0.75]),
        np.asarray([0.9, 0.8, 0.7]),
        np.asarray([50.0, 51.0, 49.0]),
        np.asarray([100.0, 100.2, 100.4]),
        np.linspace(40.0, 70.0, 151),
        90.0,
        0.2,
        20.0,
        0.0,
        0.0,
        24,
        3,
    )
    tail = (0.998, 0.002, 0.005, 0.1, 0.001, 0.5, 4.5, 0.01)
    seed_a = train.stable_seed("likpf", "train", "well_a")
    seed_b = train.stable_seed("likpf", "train", "well_b")
    expected_a = (
        int(hashlib.sha256(b"likpf::train::well_a").hexdigest()[:16], 16)
        % 2_147_483_647
        + 1
    )
    assert seed_a == expected_a
    first = train._pf_residual_offset_allseeds(*args, seed_a, *tail)
    _ = train._pf_residual_offset_allseeds(*args, seed_b, *tail)
    repeated = train._pf_residual_offset_allseeds(*args, seed_a, *tail)
    for left, right in zip(first, repeated, strict=True):
        assert np.array_equal(left, right)


def test_truth_role_and_control_access_fail_before_freeze(train: ModuleType) -> None:
    ledger = train.LeakageLedger(expected_variant_wells=2)
    with pytest.raises(RuntimeError, match="before the exp500 candidate froze"):
        ledger.record_truth(1)
    ledger = train.LeakageLedger(expected_variant_wells=2)
    with pytest.raises(RuntimeError, match="before the exp500 candidate froze"):
        ledger.record_control(1)
    ledger = train.LeakageLedger(expected_variant_wells=2)
    with pytest.raises(RuntimeError, match="before the exp500 candidate froze"):
        ledger.record_role_fold_episode(1)


def test_fixed44_assets_and_notebook_safety(train: ModuleType, config: dict) -> None:
    wells, report = train.load_fixed44_identity(config)
    assert len(wells) == len(set(wells)) == 44
    assert report["fixed32_wells"] == 32
    assert report["sentinel_wells"] == 12
    assert report["overlap_wells"] == 0
    assert config["data"]["stage_0_expected_suffix_rows"] == 224400
    source = SOURCE.read_text()
    assert "Path(__file__)" not in source
    assert "def run_stage1_shard(" in source
    assert "def run_stage1_merge(" in source
    assert "# ## 12. Setup, execution selection, and configuration preview" in source
    assert "huber_log_likelihood" not in source


def test_saved_csv_content_sha_accepts_kaggle_expanded_file(
    train: ModuleType, tmp_path: Path
) -> None:
    payload = b"id,prediction\na,1.0\nb,2.0\n"
    plain = tmp_path / "saved.csv"
    compressed = tmp_path / "saved.csv.gz"
    disguised = tmp_path / "saved.csv.gz.bin"
    plain.write_bytes(payload)
    with gzip.GzipFile(filename=compressed, mode="wb", mtime=0) as file_pointer:
        file_pointer.write(payload)
    with gzip.GzipFile(filename=disguised, mode="wb", mtime=0) as file_pointer:
        file_pointer.write(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert train.sha256_decompressed_csv(plain) == expected
    assert train.sha256_decompressed_csv(compressed) == expected
    assert train.sha256_decompressed_csv(disguised) == expected


def test_stage1_merge_accepts_platform_expanded_gzip_by_content_sha(
    train: ModuleType, tmp_path: Path
) -> None:
    frame = pd.DataFrame({"id": ["a", "b"], "prediction": [1.0, 2.0]})
    compressed_root = tmp_path / "compressed"
    expanded_root = tmp_path / "expanded"
    compressed_root.mkdir()
    expanded_root.mkdir()
    compressed = compressed_root / "candidate.csv.gz"
    report = train.write_deterministic_gzip_csv(frame, compressed)
    expanded = expanded_root / "candidate.csv"
    with gzip.open(compressed, "rb") as source:
        expanded.write_bytes(source.read())
    assert train._artifact_file(expanded_root, "candidate.csv.gz") == expanded
    train._verify_artifact_report(expanded, report, "expanded candidate")

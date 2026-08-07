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

ROOT = Path(__file__).resolve().parents[1]
EXP = "exp416_roughening_x10_likpf_full_oof_ablation"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp072_exp063_full_replay_feature_cache"
    / "public_notebook_replay_audit.py"
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
    previous = os.environ.get("EXP416_IMPORT_ONLY")
    os.environ["EXP416_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp416_train_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP416_IMPORT_ONLY", None)
        else:
            os.environ["EXP416_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def test_frozen_scientific_contract_and_zero_control_rerun(train: ModuleType, config: dict) -> None:
    contract = train.validate_scientific_contract(config)
    assert contract["primary_candidate"] == "likpf_roughening_x10_mean"
    assert contract["control_pf"] == "saved_exp072_load_only_zero_reruns"
    assert contract["execution_counts"] == {
        "scientific_variants": 1,
        "candidate_pf_well_runs": 773,
        "parent_pf_control_reruns": 0,
        "seeds_per_well": 128,
        "seed_well_trajectories": 98_944,
        "particles_per_seed": 500,
        "particle_starts": 49_472_000,
        "reporting_folds": 5,
        "well_shard_count": 4,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    assert (
        contract["scientific_contract_sha256"]
        == "9c9bdaa93f0e64aa2ea54a46ae8fbb2a4f1f4f05a34b0d98e734e2b3c8ac398a"
    )
    approved = train.validate_scientific_contract(config, require_run_approval=True)
    assert approved["scientific_contract_sha256"] == contract["scientific_contract_sha256"]
    broken = copy.deepcopy(config)
    broken["execution"]["train_run_approved"] = False
    with pytest.raises(RuntimeError, match="package/push/train run is not approved"):
        train.validate_scientific_contract(broken, require_run_approval=True)


def test_only_two_roughening_values_change_by_exact_x10(
    train: ModuleType,
    config: dict,
) -> None:
    diff = train.roughening_only_parameter_diff(config)
    assert diff == {
        "rough_position": (0.1, 1.0),
        "rough_rate": (0.001, 0.01),
    }
    broken = copy.deepcopy(config)
    broken["model"]["pf"]["position_noise"] = 0.006
    control, candidate = train.pf_parameter_sets(broken)
    assert control["position_noise"] == candidate["position_noise"] == 0.006
    broken["model"]["pf"]["candidate_rough_rate"] = 0.02
    with pytest.raises(ValueError, match="fixed roughening multiplier"):
        train.roughening_only_parameter_diff(broken)


def test_exp209_control_is_reconstructed_from_actual_cache_columns(
    train: ModuleType,
    config: dict,
) -> None:
    assert train.exp209_reconstruction_columns(config) == [
        "hmm_mean_tvt",
        "hmm_minus_likpf_mean",
    ]
    source = pd.DataFrame(
        {
            "hmm_mean_tvt": np.asarray([100.0, 102.5], dtype=np.float64),
            "hmm_minus_likpf_mean": np.asarray([1.25, -0.5], dtype=np.float64),
        }
    )
    reconstructed = train.reconstruct_exp209_likpf(source, config)
    assert reconstructed.dtype == np.float32
    np.testing.assert_array_equal(
        reconstructed,
        np.asarray([98.75, 103.0], dtype=np.float32),
    )

    broken = copy.deepcopy(config)
    broken["data"]["exp209_reconstructed_control"]["reconstruction_columns"] = [
        "likpf_mean_exp209_reconstructed"
    ]
    with pytest.raises(ValueError, match="exp209 reconstruction columns"):
        train.exp209_reconstruction_columns(broken)


def test_stable_seed_matches_exp072_policy(train: ModuleType) -> None:
    key = "likpf::train::well-a"
    expected = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % 2_147_483_647 + 1
    assert train.stable_seed("likpf", "train", "well-a") == expected
    assert train.stable_seed("likpf", "train", "well-b") != expected


def test_deterministic_lpt_sharding_is_order_independent(train: ModuleType) -> None:
    manifest = pd.DataFrame(
        {
            "well_id": ["a", "b", "c", "d", "e", "f", "g"],
            "suffix_rows": [10, 9, 8, 7, 6, 5, 4],
        }
    )
    first = train.assign_lpt_shards(manifest, shard_count=4)
    second = train.assign_lpt_shards(
        manifest.sample(frac=1.0, random_state=42),
        shard_count=4,
    )
    pd.testing.assert_frame_equal(first, second)
    assert sorted(first["shard_index"].unique().tolist()) == [0, 1, 2, 3]
    loads = first.groupby("shard_index")["suffix_rows"].sum()
    assert int(loads.max() - loads.min()) <= 4


def test_strict_four_shard_merge_preserves_float32_logical_sha(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = copy.deepcopy(config)
    local["validation"]["expected_rows"] = 8
    local["validation"]["expected_wells"] = 4
    local["execution"]["candidate_pf_well_runs"] = 4
    local["execution"]["seed_well_trajectories"] = 512
    local["execution"]["particle_starts"] = 256_000
    contract = {"scientific_contract_sha256": "c" * 64}
    monkeypatch.setattr(train, "validate_scientific_contract", lambda _: contract)
    roots: list[Path] = []
    for shard_index in range(4):
        root = tmp_path / f"shard{shard_index}"
        root.mkdir()
        well = f"w{shard_index}"
        candidate = pd.DataFrame(
            {
                "id": [f"{well}_0", f"{well}_1"],
                "well_id": [well, well],
                "row_idx": np.asarray([0, 1], dtype=np.int64),
                "suffix_offset": np.asarray([0, 1], dtype=np.int64),
                "last_known_tvt": np.asarray([100.0, 100.0], dtype=np.float64),
                "md_since": np.asarray([1.0, 2.0], dtype=np.float64),
                "raw_gr_observed": [True, False],
                "likpf_roughening_x10_mean": np.asarray(
                    [100.25 + shard_index, 100.5 + shard_index],
                    dtype=np.float32,
                ),
            }
        )
        prediction_path = (
            root / f"{EXP}_shard{shard_index}_candidate_predictions.csv.gz"
        )
        frozen = train.freeze_prediction_frame(candidate, prediction_path)
        pd.DataFrame(
            {
                "well_id": [well],
                "status": ["ok"],
                "seed_well_trajectories": [128],
                "particle_starts": [64_000],
            }
        ).to_csv(root / f"{EXP}_shard{shard_index}_well_audit.csv", index=False)
        pd.DataFrame(
            {
                "well_id": [well],
                "suffix_rows": [2],
                "shard_index": [shard_index],
            }
        ).to_csv(root / f"{EXP}_shard{shard_index}_well_manifest.csv", index=False)
        (root / f"{EXP}_shard{shard_index}_summary.json").write_text(
            json.dumps(
                {
                    "stage": "candidate_shard",
                    "shard_index": shard_index,
                    "scientific_contract_sha256": "c" * 64,
                    "frozen_prediction": frozen,
                    "counts": {
                        "candidate_pf_well_runs": 1,
                        "seed_well_trajectories": 128,
                        "particle_starts": 64_000,
                    },
                }
            )
        )
        roots.append(root)
    ledger = train.TruthAccessLedger()
    merged, audit, frozen, paths = train.merge_shard_outputs(
        roots,
        tmp_path / "merged",
        local,
        ledger=ledger,
    )
    assert len(merged) == 8
    assert merged["likpf_roughening_x10_mean"].dtype == np.float32
    assert len(audit) == 4
    assert frozen["execution_counts"]["candidate_pf_well_runs"] == 4
    assert paths["merged_prediction"].exists()
    assert ledger.prediction_frozen is True


def test_horizontal_loader_excludes_truth_and_freeze_ledger(
    train: ModuleType,
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0],
            "Z": [0.0, 0.1, 0.2],
            "GR": [50.0, np.nan, 55.0],
            "TVT_input": [100.0, np.nan, np.nan],
            "TVT": [100.0, 101.0, 102.0],
            "error": [0.0, 1.0, 2.0],
        }
    ).to_csv(tmp_path / "w__horizontal_well.csv", index=False)
    horizontal = train.load_horizontal_without_truth("w", tmp_path)
    assert list(horizontal.columns) == ["MD", "Z", "GR", "TVT_input"]

    ledger = train.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="requires a frozen prediction"):
        ledger.require_frozen()
    candidate = pd.DataFrame(
        {
            "id": ["w_1", "w_2"],
            "well_id": ["w", "w"],
            "row_idx": np.asarray([1, 2], dtype=np.int64),
            "suffix_offset": np.asarray([0, 1], dtype=np.int64),
            "last_known_tvt": np.asarray([100.0, 100.0], dtype=np.float64),
            "md_since": np.asarray([1.0, 2.0], dtype=np.float64),
            "raw_gr_observed": [False, True],
            "likpf_roughening_x10_mean": np.asarray([101.0, 102.0], dtype=np.float32),
        }
    )
    frozen = train.freeze_prediction_frame(
        candidate,
        tmp_path / "prediction.csv.gz",
        ledger=ledger,
    )
    assert frozen["frozen_before_truth_attachment"] is True
    assert ledger.report()["prediction_frozen"] is True
    assert all(value == 0 for value in ledger.report()["before_freeze"].values())


def test_control_kernel_is_exact_parent_fixture_parity(train: ModuleType) -> None:
    if "numba" not in sys.modules:
        numba_stub = ModuleType("numba")

        def identity_njit(*args, **kwargs):
            del kwargs
            if args and callable(args[0]):
                return args[0]

            def decorator(function):
                return function

            return decorator

        numba_stub.njit = identity_njit
        sys.modules["numba"] = numba_stub
    parent = load_module(PARENT_SOURCE, "exp072_parent_for_exp416")
    md = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    z = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    gr = np.array([50.0, 52.0, 54.0, 53.0, 51.0], dtype=np.float64)
    grid_gr = np.linspace(40.0, 70.0, 151, dtype=np.float64)
    args = (
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
    observed = train._pf_lik_allseeds(*args)
    expected_predictions, expected_likelihoods = parent._pf_lik_allseeds(*args)
    assert np.array_equal(observed[0], expected_predictions)
    assert np.array_equal(observed[1], expected_likelihoods)
    assert observed[2].shape == (4,)
    assert observed[3].shape == (4,)
    assert observed[4].shape == (4,)


def test_exp072_input_preparation_keeps_gr_scale_and_missing_policy(train: ModuleType) -> None:
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(12, dtype=float) + 1.0,
            "Z": np.linspace(0.0, 1.1, 12),
            "GR": [50.0, np.nan, 70.0, 80.0, 65.0, 60.0, np.nan, 55.0, 58.0, np.nan, 62.0, 64.0],
            "TVT_input": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, *([np.nan] * 6)],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(90.0, 130.0, 81),
            "GR": np.linspace(45.0, 85.0, 81),
        }
    )
    prepared = train.prepare_likelihood_pf_inputs(horizontal, typewell, grid_step=0.2)
    known = horizontal["TVT_input"].notna().to_numpy()
    expected_gr = np.interp(
        horizontal.loc[known, "TVT_input"].to_numpy(np.float64),
        typewell["TVT"].to_numpy(np.float64),
        typewell["GR"].to_numpy(np.float64),
    )
    residual = horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64) - expected_gr
    assert prepared["scale_audit"]["base_scale"] == pytest.approx(
        float(np.clip(np.std(residual), 10.0, 60.0))
    )
    assert prepared["scale_audit"]["known_gr_missing_rows"] == 1
    assert np.isfinite(prepared["eval_gr"]).all()


def test_episode_gate_requires_all_preregistered_conditions(
    train: ModuleType,
    config: dict,
) -> None:
    local = copy.deepcopy(config)
    rows = 10
    truth = np.arange(rows, dtype=np.float64)
    control = truth + 1.0
    candidate = truth + 0.5
    frame = pd.DataFrame(
        {
            "id": [f"w{i}_{i}" for i in range(rows)],
            "well_id": [f"w{i}" for i in range(rows)],
            "row_idx": np.zeros(rows, dtype=np.int64),
            "true_tvt": truth,
            "saved_exp072_likpf_mean": control,
            "exp209_reconstructed_likpf_mean": control,
            "likpf_roughening_x10_mean": candidate.astype(np.float32),
            "fold": np.asarray([0, 1, 2, 3, 4] * 2, dtype=np.int64),
            "raw_gr_observed": [True, False] * 5,
            "md_since": np.asarray([1100.0] * rows),
            "hidden_like_spatial": [True] * rows,
            "hidden_like_typewell_purged": [True] * rows,
        }
    )
    metrics = pd.DataFrame(
        [
            train.metric_record(frame, mask, scope=scope)
            for scope, mask in train.metric_scopes(frame)
        ]
    )
    by_well = pd.DataFrame(
        {
            "well_id": frame["well_id"],
            "delta_rmse_candidate_minus_control": [-0.5] * rows,
        }
    )
    episode_metrics = pd.DataFrame(
        {
            "episode_id": [f"e{i}" for i in range(16)],
            "rows": [1] * 16,
            "candidate_sse": [0.25] * 16,
            "control_sse": [1.0] * 16,
        }
    )
    audit = pd.DataFrame(
        {
            "well_id": frame["well_id"],
            "status": ["ok"] * rows,
            "seed_well_trajectories": [128] * rows,
            "particle_starts": [64_000] * rows,
        }
    )
    local["validation"].update(
        {
            "expected_rows": rows,
            "expected_wells": rows,
            "saved_control_rmse_ft": 1.0,
        }
    )
    local["execution"].update(
        {
            "candidate_pf_well_runs": rows,
            "seed_well_trajectories": rows * 128,
            "particle_starts": rows * 64_000,
        }
    )
    ledger = train.TruthAccessLedger(prediction_frozen=True)
    frozen = {
        "logical_content_sha256": "a" * 64,
    }
    shard_summaries = [
        {"runtime": {"elapsed_seconds": 10.0, "peak_rss_gb": 1.0}} for _ in range(4)
    ]
    gate = train.evaluate_gate(
        frame,
        metrics,
        by_well,
        episode_metrics,
        audit,
        frozen,
        ledger,
        shard_summaries,
        local,
        probe_report={"byte_identical_float32": True},
    )
    assert gate["technical_gate"]["passed"] is True
    assert gate["scientific_gate"]["passed"] is True
    assert gate["deterministic_anchor_eligible"] is True

    failed_metrics = metrics.copy()
    failed_metrics.loc[
        failed_metrics["scope"].eq("hidden_like_spatial"),
        "delta_rmse_candidate_minus_control",
    ] = 0.01
    failed = train.evaluate_gate(
        frame,
        failed_metrics,
        by_well,
        episode_metrics,
        audit,
        frozen,
        ledger,
        shard_summaries,
        local,
        probe_report={"byte_identical_float32": True},
    )
    assert failed["scientific_gate"]["passed"] is False
    assert failed["decision"] == "roughening_x10_rejected_close_without_rescue"


def test_notebook_source_is_self_contained_and_execution_is_approved(config: dict) -> None:
    source = SOURCE.read_text()
    assert "__file__" not in source
    assert "from settings import" not in source
    assert "from exp400" not in source
    assert config["implementation"]["canonical_notebooks_remain_placeholders"] is False
    assert config["implementation"]["canonical_inference_notebook_remains_placeholder"] is True
    assert config["execution"]["canonical_notebook_adoption_approved"] is True
    assert config["execution"]["kaggle_package_approved"] is True
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["train_run_approved"] is True
    assert config["execution"]["run_train"] is True
    assert config["inference"]["enabled"] is False
    assert config["execution"]["submission_approved"] is False
    assert config["runtime"]["kaggle"]["kernel_sources"] == [
        "kentookumura/exp072-exp063-full-replay-feature-cache-train",
        "kentookumura/exp209-joint-exact-parity-train",
        "kentookumura/exp226-k16-kappa-repro-train",
    ]
    assert {
        item["destination"]
        for item in config["runtime"]["kaggle"]["bootstrap_dependency_files"]
    } == {
        "assets/exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv",
        "assets/pf_persistent_offset_episodes.csv",
        "assets/pf_counterfactual_sentinel_wells.csv",
    }


def test_metrics_json_records_no_inference_or_submission_result() -> None:
    metrics = json.loads((EXP_DIR / "metrics.json").read_text())
    assert metrics["implementation_started"] is True
    assert metrics["public_lb"] is None
    assert metrics["private_lb"] is None

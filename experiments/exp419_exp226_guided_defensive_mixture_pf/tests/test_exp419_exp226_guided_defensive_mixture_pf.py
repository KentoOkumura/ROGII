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

ROOT = Path(__file__).resolve().parents[3]
EXP = "exp419_exp226_guided_defensive_mixture_pf"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
PARENT_SOURCE = (
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
    previous = os.environ.get("EXP419_IMPORT_ONLY")
    os.environ["EXP419_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp419_train_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP419_IMPORT_ONLY", None)
        else:
            os.environ["EXP419_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def parent() -> ModuleType:
    return load_module(PARENT_SOURCE, "exp404_parent_for_exp419")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def synthetic_pf_inputs() -> tuple[np.ndarray, ...]:
    md = np.arange(1.0, 7.0, dtype=np.float64)
    z = np.linspace(0.0, 0.5, len(md), dtype=np.float64)
    gr = np.array([50.0, 52.0, 54.0, 53.0, 51.0, 50.0], dtype=np.float64)
    grid_gr = np.linspace(40.0, 70.0, 151, dtype=np.float64)
    return md, z, gr, grid_gr


def test_frozen_contract_records_one_variant_and_zero_control_rerun(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)

    assert contract["primary_candidate"] == "exp226_guided_defensive_mixture_scale5"
    assert contract["primary_control"] == "likpf_scale_5_x1p0"
    assert contract["control_pf"] == "saved_exp404_scale5_x1p0_load_only_zero_reruns"
    assert contract["execution_counts"] == {
        "scientific_variants": 1,
        "candidate_pf_well_runs": 773,
        "parent_pf_control_reruns": 0,
        "exp226_reruns": 0,
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
    assert len(contract["scientific_contract_sha256"]) == 64
    approved = train.validate_scientific_contract(config, require_run_approval=True)
    assert approved["scientific_contract_sha256"] == contract["scientific_contract_sha256"]


def test_defensive_mixture_contract_is_exact_and_fail_closed(
    train: ModuleType,
    config: dict,
) -> None:
    proposal = train.proposal_contract(config)

    assert proposal["target_weight"] == 0.5
    assert proposal["geometry_weights"] == [1.0 / 6.0] * 3
    assert proposal["geometry_sigma_multipliers"] == [1.0, 4.0, 16.0]
    assert proposal["weight_sum"] == 1.0
    assert proposal["importance_clipping"] is False
    assert proposal["target_posterior_changed"] is False
    assert len(proposal["proposal_contract_sha256"]) == 64

    broken = copy.deepcopy(config)
    broken["model"]["rate_proposal"]["target_component"]["weight"] = 0.4
    with pytest.raises(ValueError, match="weights do not sum to one"):
        train.proposal_contract(broken)

    broken = copy.deepcopy(config)
    broken["model"]["rate_proposal"]["importance_clipping"] = True
    with pytest.raises(ValueError, match="proposal contract changed"):
        train.proposal_contract(broken)


def test_importance_ratio_is_positive_finite_and_bounded_by_two(
    train: ModuleType,
) -> None:
    geometry_weights = np.asarray([1.0 / 6.0] * 3, dtype=np.float64)
    sigma_multipliers = np.asarray([1.0, 4.0, 16.0], dtype=np.float64)
    samples = np.linspace(-0.15, 0.15, 1201)
    ratios = np.asarray(
        [
            train.defensive_mixture_importance_ratio(
                float(value),
                0.01,
                -0.02,
                0.002,
                0.5,
                geometry_weights,
                sigma_multipliers,
            )
            for value in samples
        ]
    )

    assert np.isfinite(ratios).all()
    assert (ratios >= 0.0).all()
    assert float(ratios[len(ratios) // 2]) > 0.0
    assert float(ratios.max()) <= 2.0 + 1.0e-12


def test_geometry_weight_zero_preserves_exact_exp404_rng_and_predictions(
    train: ModuleType,
    parent: ModuleType,
) -> None:
    md, z, gr, grid_gr = synthetic_pf_inputs()
    parent_args = (
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
    expected = parent._pf_lik_allseeds(*parent_args)
    observed = train._pf_guided_allseeds(
        md,
        z,
        gr,
        np.zeros(len(md), dtype=np.float64),
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
        0.01,
        1.0,
        np.asarray([1.0 / 6.0] * 3, dtype=np.float64),
        np.asarray([1.0, 4.0, 16.0], dtype=np.float64),
    )

    for index in range(5):
        assert np.array_equal(observed[index], expected[index])
    assert np.array_equal(observed[7], np.full(4, 24 * len(md), dtype=float))
    assert np.array_equal(observed[8][:, 0], np.full(4, 24 * len(md)))
    assert not observed[8][:, 1:].any()


def test_guided_kernel_uses_all_components_and_respects_importance_bound(
    train: ModuleType,
) -> None:
    md, z, gr, grid_gr = synthetic_pf_inputs()
    outputs = train._pf_guided_allseeds(
        md,
        z,
        gr,
        np.linspace(-0.01, 0.02, len(md), dtype=np.float64),
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        128,
        8,
        12345,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
        0.01,
        0.5,
        np.asarray([1.0 / 6.0] * 3, dtype=np.float64),
        np.asarray([1.0, 4.0, 16.0], dtype=np.float64),
    )

    assert np.isfinite(outputs[0]).all()
    assert np.isfinite(outputs[1]).all()
    assert float(outputs[6].max()) <= 2.0 + 1.0e-12
    assert (outputs[8].sum(axis=0) > 0).all()
    assert outputs[9].shape == (8, len(md))
    assert outputs[10].shape == (8, len(md))
    assert np.less_equal(outputs[9], outputs[10]).all()


def test_temperature_five_seed_aggregation(train: ModuleType) -> None:
    predictions = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    log_likelihoods = np.asarray([-3.0, -2.0, -1.0])

    observed, weights = train.aggregate_seed_predictions(
        predictions,
        log_likelihoods,
        temperature=5.0,
    )
    expected_weights = np.exp((log_likelihoods - log_likelihoods.max()) / 5.0)
    expected_weights /= expected_weights.sum()

    np.testing.assert_allclose(weights, expected_weights)
    np.testing.assert_allclose(
        observed,
        (expected_weights[:, None] * predictions).sum(axis=0),
    )


def test_geometry_rate_uses_last_known_surface_for_first_suffix(
    train: ModuleType,
) -> None:
    prepared = {
        "eval_indices": np.asarray([3, 4, 5]),
        "eval_md": np.asarray([11.0, 13.0, 16.0]),
        "eval_z": np.asarray([2.0, 2.5, 3.0]),
        "last_known_position": 102.0,
    }
    geometry = pd.DataFrame(
        {
            "well_id": ["w"] * 3,
            "row_idx": [3, 4, 5],
            "suffix_offset": [0, 1, 2],
            "tvt_geop": [101.0, 102.5, 105.0],
        }
    )

    observed = train.geometry_surface_rate(prepared, geometry)

    np.testing.assert_allclose(observed, [1.0, 1.0, 1.0])


def test_geometry_loader_reads_only_proposal_allowlist(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["w", "w"],
            "row_idx": [3, 4],
            "suffix_offset": [0, 1],
            "tvt_geop": [100.0, 101.0],
            "tvt_pred": [999.0, 999.0],
            "gr_delta": [1.0, 1.0],
            "tvt_true": [500.0, 501.0],
            "error": [400.0, 400.0],
            "abs_error": [400.0, 400.0],
            "fold": [0, 0],
        }
    )
    path = tmp_path / "geometry.csv.gz"
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )

    observed = train.load_fold_safe_geometry(path, config)

    assert observed.columns.tolist() == [
        "well_id",
        "row_idx",
        "suffix_offset",
        "tvt_geop",
    ]
    assert not {
        "tvt_pred",
        "gr_delta",
        "tvt_true",
        "error",
        "fold",
    }.intersection(observed.columns)


def test_target_free_support_bounds_are_evaluated_only_after_truth_freeze(
    train: ModuleType,
    tmp_path: Path,
) -> None:
    prediction = pd.DataFrame(
        {
            "id": ["w_3", "w_4"],
            "well_id": ["w", "w"],
            "row_idx": [3, 4],
            "suffix_offset": [0, 1],
            "geometry_surface_rate": [1.0, 1.0],
            train.PRIMARY_CANDIDATE: [101.0, 102.0],
        }
    )
    prediction_path = tmp_path / "prediction.csv.gz"
    train.write_deterministic_gzip_csv(prediction, prediction_path)
    minimum = np.asarray([[99.0, 100.0], [101.0, 103.0]], dtype=np.float32)
    maximum = np.asarray([[101.0, 102.0], [102.0, 104.0]], dtype=np.float32)
    minimum_path = tmp_path / "minimum.npy"
    maximum_path = tmp_path / "maximum.npy"
    np.save(minimum_path, minimum)
    np.save(maximum_path, maximum)
    frame = prediction.copy()
    frame["true_tvt"] = [100.5, 102.5]

    observed = train.attach_candidate_predictive_support(
        frame,
        [
            {
                "prediction_path": prediction_path,
                "minimum_path": minimum_path,
                "maximum_path": maximum_path,
            }
        ],
    )

    np.testing.assert_allclose(
        observed["candidate_predictive_truth_support_fraction"],
        [1.0, 0.0],
    )


def test_stable_seed_and_truth_ledger_contract(train: ModuleType) -> None:
    key = "likpf::train::well-a"
    expected = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % 2_147_483_647 + 1
    assert train.stable_seed("likpf", "train", "well-a") == expected
    ledger = train.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="requires a frozen prediction"):
        ledger.require_frozen()
    ledger.mark_frozen()
    ledger.require_frozen()
    ledger.control_prediction_rows_after_freeze = 3
    assert all(value == 0 for value in ledger.report()["before_freeze"].values())


def test_preflight_probe_is_compared_with_full_shard_candidate(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    probe_well = str(config["reproducibility"]["probe_well"])
    expected = pd.DataFrame(
        {
            "id": [f"{probe_well}_3", f"{probe_well}_4"],
            "well_id": [probe_well, probe_well],
            "row_idx": [3, 4],
            "suffix_offset": [0, 1],
            "geometry_surface_rate": [1.0, 1.0],
            train.PRIMARY_CANDIDATE: np.asarray([101.0, 102.0], dtype=np.float32),
        }
    )
    prediction_path = tmp_path / "preflight.csv.gz"
    train.write_deterministic_gzip_csv(expected, prediction_path)
    report_path = tmp_path / "preflight.json"
    train.write_json(
        report_path,
        {
            "stage": "preflight_probe",
            "passed": True,
            "geometry_weight_zero_parity_max_abs_ft": 0.0,
        },
    )
    local = copy.deepcopy(config)
    local["reproducibility"]["probe_source"] = {
        "filename": prediction_path.name,
        "candidates": [str(tmp_path)],
    }
    local["reproducibility"]["probe_report"] = {
        "filename": report_path.name,
        "candidates": [str(tmp_path)],
    }

    observed = train.load_optional_probe_report(local, expected.copy())

    assert observed is not None
    assert observed["full_shard_comparison_recorded"] is True
    assert observed["byte_identical_float32"] is True
    assert observed["maximum_absolute_difference_ft"] == 0.0


def test_mechanism_and_standalone_gates_are_both_required(
    train: ModuleType,
    config: dict,
) -> None:
    rows = 10
    truth = np.arange(rows, dtype=np.float64) + 100.0
    frame = pd.DataFrame(
        {
            "id": [f"w{i % 2}_{i}" for i in range(rows)],
            "well_id": [f"w{i % 2}" for i in range(rows)],
            "row_idx": np.arange(rows),
            "suffix_offset": np.arange(rows),
            "true_tvt": truth,
            train.PRIMARY_CANDIDATE: truth,
            "likpf_scale_5_x1p0": truth + 1.0,
            "exp226_final_oof": truth + 0.5,
            "fold": np.arange(rows) % 5,
            "raw_gr_observed": np.arange(rows) % 2 == 0,
            "well_missing_fraction": 0.5,
            "md_since": np.arange(rows) * 1100.0,
            "hidden_like_spatial": True,
            "hidden_like_typewell_purged": True,
            "episode_id": ["e0" if i % 2 == 0 else None for i in range(rows)],
            "exp410_fixed_episode": np.arange(rows) % 2 == 0,
            "candidate_predictive_truth_support_fraction": 1.0,
            "exp410_baseline_predictive_truth_support_fraction": 0.0,
        }
    )
    episodes = pd.DataFrame(
        {
            "episode_id": ["e0"],
            "well": ["w0"],
            "start_row_idx": [0],
            "end_row_idx_exclusive": [10],
            "rows": [5],
        }
    )
    metrics, by_well, episode_metrics = train.build_metric_outputs(frame, episodes)
    local = copy.deepcopy(config)
    local["validation"].update(
        {
            "expected_rows": rows,
            "expected_wells": 2,
            "saved_control_rmse_ft": 1.0,
            "saved_exp226_final_rmse_ft": 0.5,
        }
    )
    local["execution"].update(
        {
            "candidate_pf_well_runs": 2,
            "seed_well_trajectories": 256,
            "particle_starts": 128_000,
        }
    )
    audit = pd.DataFrame(
        {
            "well_id": ["w0", "w1"],
            "status": ["ok", "ok"],
            "seed_well_trajectories": [128, 128],
            "particle_starts": [64_000, 64_000],
            "importance_ratio_minimum": [0.0, 0.0],
            "importance_ratio_maximum": [2.0, 2.0],
        }
    )
    shard_summaries = [
        {
            "runtime": {"elapsed_seconds": 1.0},
            "proposal_input": {
                "safe_columns": ["well_id", "row_idx", "suffix_offset", "tvt_geop"],
                "forbidden_exp226_columns_parsed": [],
            },
        }
        for _ in range(4)
    ]
    ledger = train.TruthAccessLedger()
    ledger.mark_frozen()
    gate = train.evaluate_gate(
        frame,
        metrics,
        by_well,
        episode_metrics,
        audit,
        {"logical_content_sha256": "a" * 64},
        ledger,
        shard_summaries,
        local,
        probe_report={
            "byte_identical_float32": True,
            "geometry_weight_zero_parity_max_abs_ft": 0.0,
        },
    )

    assert gate["technical_gate"]["passed"] is True
    assert gate["mechanism_gate"]["passed"] is True
    assert gate["standalone_adoption_gate"]["passed"] is True
    assert gate["passed"] is True


def test_compact_notebook_contract_and_canonical_placeholders_remain() -> None:
    source = SOURCE.read_text()
    required_headings = {
        "# ## 2. Notebook-safe configuration, path, and SHA helpers",
        "# ## 3. Frozen proposal-only scientific contract",
        "# ## 4. Truth-free raw input checks and deterministic LPT sharding",
        "# ## 5. Exact exp072 input preparation and fold-safe geometry rate",
        "# ## 6. Defensive-mixture proposal and exact importance correction",
        "# ## 7. Shard candidate generation and prediction freeze",
        "# ## 8. Strict shard merge and optional rerun probe",
        "# ## 9. Late truth, saved-control, fold, hidden-like, and episode attachment",
        "# ## 10. Metrics and fail-closed scientific gate",
        "# ## 11. Generated artifacts and stage orchestration",
        "# ## 12. Setup and configuration preview",
    }

    assert required_headings.issubset(set(source.splitlines()))
    assert "__file__" not in source
    assert "from settings import" not in source
    assert (EXP_DIR / f"{EXP}_train.ipynb").exists()
    assert (EXP_DIR / f"{EXP}_inference.ipynb").exists()

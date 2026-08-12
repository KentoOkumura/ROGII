from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp254_numba_allseed_pf_speed_reproduction"
TRAIN_SOURCE = EXP_DIR / "exp254_numba_allseed_pf_speed_reproduction_train.py"
CONFIG_PATH = EXP_DIR / "config.yaml"


def load_config() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(payload, dict)
    return payload


def load_definition_namespace() -> dict[str, Any]:
    source = TRAIN_SOURCE.read_text()
    definitions = source.split("# ## 7. Setup and fixed benchmark contract", maxsplit=1)[0]
    definitions = definitions.replace("import numba\n", "")
    definitions = definitions.replace("from numba import njit\n", "")

    def njit(*_args: Any, **_kwargs: Any) -> Any:
        def decorate(function: Any) -> Any:
            return function

        return decorate

    namespace: dict[str, Any] = {"njit": njit}
    exec(compile(definitions, str(TRAIN_SOURCE), "exec"), namespace)
    return namespace


def test_exp254_fixed_runtime_contract() -> None:
    config = load_config()
    assert config["experiment"]["route"] == "pf_beam"
    assert config["lineage"]["parent"] == "exp243_pf_seed_medoids"
    assert config["model"]["runtime"]["particles"] == 500
    assert config["model"]["runtime"]["seed_count"] == 128
    benchmark = config["model"]["benchmark"]
    assert benchmark["seed_counts"] == [1, 4, 16, 32, 64, 128]
    assert benchmark["candidate_spec_counts"] == [1, 10, 100, 300]
    assert benchmark["temperatures"] == [3.0, 5.0, 8.0, 12.0]
    assert benchmark["python_processes"] == 1
    assert benchmark["numba_threads"] == 1
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["control_or_parent_retraining"] is False
    assert config["inference"]["enabled"] is False
    assert config["inference"]["create_submission"] is False


def test_exp254_candidate_specs_are_fixed_and_unique() -> None:
    config = load_config()
    namespace = load_definition_namespace()
    specs = namespace["make_candidate_specs"](300, 128, config)
    assert len(specs) == 300
    assert specs[0].aggregation == "mean"
    assert specs[0].temperature is None
    assert specs[0].seed_indices == tuple(range(128))
    keys = {
        (spec.aggregation, spec.temperature, spec.seed_indices) for spec in specs
    }
    assert len(keys) == 300
    assert {spec.aggregation for spec in specs} == {"mean", "likelihood_weighted"}
    assert {
        spec.temperature for spec in specs if spec.aggregation == "likelihood_weighted"
    } == {3.0, 5.0, 8.0, 12.0}
    assert {spec.subset_count for spec in specs}.issubset({1, 4, 16, 32, 64, 128})
    assert all(
        len(spec.seed_indices) == len(set(spec.seed_indices)) for spec in specs
    )


def test_exp254_legacy_and_allseed_kernels_have_exact_synthetic_parity() -> None:
    namespace = load_definition_namespace()
    legacy = namespace["_legacy_single_seed_pf"]
    allseed = namespace["_exp243_numba_allseed_pf"]
    md = np.asarray([1.0, 2.5, 4.0, 6.0], dtype=np.float64)
    z = np.asarray([10.0, 10.2, 10.4, 10.7], dtype=np.float64)
    gr = np.asarray([95.0, 101.0, 98.0, 104.0], dtype=np.float64)
    grid = np.linspace(80.0, 120.0, 64, dtype=np.float64)
    common = (md, z, gr, grid, -20.0, 1.0, 30.0, 55.0, 0.01, 16)
    tail = (0.998, 0.002, 0.005, 0.10, 0.001, 0.5, 4.5)
    seed_base = 123456
    n_seeds = 4
    legacy_predictions = np.empty((n_seeds, len(md)), dtype=np.float64)
    legacy_likelihoods = np.empty(n_seeds, dtype=np.float64)
    legacy_ess = np.zeros(len(md), dtype=np.float64)
    legacy_resampled = np.zeros(len(md), dtype=np.float64)
    for seed_index in range(n_seeds):
        prediction, likelihood, ess, resampled = legacy(
            *common, seed_base + seed_index, *tail
        )
        legacy_predictions[seed_index] = prediction
        legacy_likelihoods[seed_index] = likelihood
        legacy_ess += ess
        legacy_resampled += resampled
    legacy_ess /= n_seeds
    legacy_resampled /= n_seeds
    all_predictions, all_likelihoods, all_ess, all_resampled = allseed(
        *common, n_seeds, seed_base, *tail
    )
    assert np.array_equal(legacy_predictions, all_predictions)
    assert np.array_equal(legacy_likelihoods, all_likelihoods)
    assert np.array_equal(legacy_predictions.mean(axis=0), all_predictions.mean(axis=0))
    assert np.array_equal(legacy_ess, all_ess)
    assert np.array_equal(legacy_resampled, all_resampled)


def test_exp254_closed_branch_keeps_full_workload_fail_closed() -> None:
    config = load_config()
    full = config["execution"]["full_workload"]
    assert full["require_probe_passed"] is True
    assert full["probe_summary_expected_sha256"] == ""
    source = TRAIN_SOURCE.read_text()
    assert "full_workload is fail-closed" in source
    assert "Pinned probe summary scientific contract SHA mismatch" in source


def test_exp254_does_not_load_horizontal_target_columns() -> None:
    source = TRAIN_SOURCE.read_text()
    assert 'required_horizontal = ["MD", "Z", "GR", "TVT_input"]' in source
    assert "horizontal_path, usecols=required_horizontal" in source
    assert 'required = ["well", "row_idx", "pf_replay_likpf_mean"]' in source

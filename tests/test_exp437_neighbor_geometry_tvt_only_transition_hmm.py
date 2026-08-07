from __future__ import annotations

import importlib.machinery
import importlib.util
import inspect
import sys
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

if importlib.util.find_spec("numba") is None:
    numba_stub = types.ModuleType("numba")
    numba_stub.__spec__ = importlib.machinery.ModuleSpec("numba", loader=None)

    def _njit(*args, **kwargs):
        del kwargs
        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator

    numba_stub.njit = _njit
    numba_stub.set_num_threads = lambda threads: None
    numba_stub.__version__ = "test-stub"
    sys.modules["numba"] = numba_stub

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp437_neighbor_geometry_tvt_only_transition_hmm"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp437_neighbor_geometry_tvt_only_transition_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp437_neighbor_geometry_tvt_only_transition_hmm_compact_selfcontained_inference.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"
MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "exp411_predictive_filtered_rate_innovation_destick"
    / "assets"
    / "stage0_fixed32_manifest.csv"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp437_train_test")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp437_inference_test")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def synthetic_prepared(rows: int = 15, positions: int = 25) -> dict:
    grid = 12_000.0 + np.arange(positions, dtype=np.float64) * 0.35
    x = np.linspace(-1.0, 1.0, positions)
    emission = np.vstack(
        [
            -0.5 * ((x - 0.25 * np.sin(row / 3.0)) / 0.45) ** 2
            for row in range(rows)
        ]
    ).astype(np.float32)
    return {
        "emission_ll": emission,
        "grid": grid,
        "start_p": 12.0,
        "eval_index": np.arange(100, 100 + rows, dtype=np.int64),
    }


def test_stage0_is_completed_and_execution_is_relocked(train, config):
    counts = train.validate_execution_contract(
        config,
        require_run_authorization=False,
    )
    assert counts == {
        "scientific_variants": 1,
        "stage_0_candidate_hmm_well_runs": 32,
        "stage_1_max_candidate_hmm_well_runs": 773,
        "parent_control_hmm_reruns": 0,
        "fitted_ml_models": 0,
        "lightgbm_configs": 0,
        "trained_ml_folds": 0,
        "boosters": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    assert config["implementation"]["enabled"] is True
    assert config["design"]["implementation_authorized"] is True
    assert config["design"]["kaggle_stage_0_completed"] is True
    assert config["design"]["kaggle_stage_0_all_gates_pass"] is False
    assert config["design"]["kaggle_stage_0_authorized"] is False
    assert config["runtime"]["run_approved"] is False
    assert config["execution"]["run_hmm"] is False
    assert config["execution"]["create_prediction"] is False
    with pytest.raises(RuntimeError, match="kaggle_stage_0_authorized"):
        train.validate_execution_contract(
            config,
            require_run_authorization=True,
        )


def test_scientific_contract_pins_single_direct_transition(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["variant"] == "neighbor_geometry_direct_transition"
    assert contract["persistent_state"] == "tvt_probability_distribution_only"
    assert contract["rate_state_present"] is False
    assert contract["branch_state_present"] is False
    assert contract["parent_control_hmm_reruns"] == 0
    assert contract["geometry_allowlist"] == [
        "well_id",
        "row_idx",
        "suffix_offset",
        "tvt_geop",
        "fold",
    ]

    broken = deepcopy(config)
    broken["model"]["fixed_exp435_hmm"]["sig_p"] = 0.03
    with pytest.raises(ValueError, match="fixed exp435 HMM"):
        train.validate_scientific_contract(broken)


def test_geometry_schedule_is_exact_first_difference(train):
    geometry = pd.DataFrame(
        {
            "well_id": ["well_a"] * 4,
            "row_idx": [11, 12, 13, 14],
            "suffix_offset": [0, 1, 2, 3],
            "tvt_geop": [100.2, 100.1, 100.45, 100.6],
            "fold": [2, 2, 2, 2],
        }
    )
    schedule = train.build_geometry_schedule(
        geometry,
        expected_row_idx=np.array([11, 12, 13, 14]),
        last_known_tvt=100.0,
    )
    np.testing.assert_allclose(
        schedule["transition_delta"],
        np.array([0.2, -0.1, 0.35, 0.15]),
        atol=1.0e-14,
    )
    assert schedule["first_difference_parity_max_abs_ft"] == 0.0

    forbidden = geometry.assign(tvt_true=100.0)
    with pytest.raises(ValueError, match="forbidden"):
        train.build_geometry_schedule(
            forbidden,
            expected_row_idx=np.array([11, 12, 13, 14]),
            last_known_tvt=100.0,
        )


def test_direct_kernel_and_hmm_are_normalized(train, config):
    hmm = config["model"]["fixed_exp435_hmm"]
    offsets, kernel, error = train.direct_position_kernel(
        0.43,
        hmm["step"],
        hmm["sig_p"],
    )
    assert offsets.shape == (5,)
    assert kernel.sum() == pytest.approx(1.0, abs=1.0e-14)
    assert error <= 1.0e-14

    prepared = synthetic_prepared()
    transition_delta = np.linspace(-0.12, 0.18, len(prepared["eval_index"]))
    result = train.run_direct_transition_hmm(
        prepared,
        transition_delta,
        hmm,
    )
    assert result["persistent_state_shape"] == (15, 25)
    assert np.isfinite(result["posterior_mean"]).all()
    assert np.isfinite(result["posterior_std"]).all()
    assert result["transition_row_sum_max_error"] <= 1.0e-14
    assert result["posterior_normalization_max_error"] <= 1.0e-6
    source = inspect.getsource(train._direct_transition_forward_backward)
    assert "alpha = np.empty((time_count, position_count)" in source
    assert "rate_count" not in source
    assert "branch" not in source


def test_geometry_reader_uses_read_time_allowlist(train):
    source = inspect.getsource(train.load_geometry_oof)
    assert "usecols=GEOMETRY_ALLOWLIST" in source
    assert "tvt_true" not in source
    assert "tvt_pred" not in source
    assert "gr_delta" not in source


def test_stage0_mechanism_gates_are_and_combined(train, config):
    scope = pd.read_csv(MANIFEST_PATH, dtype={"well": str}).sort_values(
        "well",
        kind="mergesort",
    )
    frozen = []
    metric_rows = []
    for row in scope.itertuples(index=False):
        candidate_rmse = 9.0 if row.role == "persistent" else 10.0
        geometry_rmse = 10.0
        dz_rmse = 12.0
        item = train.FrozenWell(
            well=row.well,
            row_idx=np.array([row.prefix_rows], dtype=np.int64),
            suffix_offset=np.array([0], dtype=np.int64),
            source_fold=np.array([row.fold], dtype=np.int64),
            geometry_prediction=np.array([10.0]),
            geometry_delta=np.array([0.1]),
            exp435_dz_only=np.array([12.0]),
            candidate_prediction=np.array([candidate_rmse]),
            posterior_std=np.array([1.0]),
            raw_gr_missing=np.array([False]),
            last_known_tvt=10.0,
            prefix_rows=row.prefix_rows,
            schedule_sha256=f"schedule-{row.well}",
            prediction_sha256=f"prediction-{row.well}",
            diagnostic_sha256=f"diagnostic-{row.well}",
            first_difference_parity_max_abs_ft=0.0,
            transition_row_sum_max_error=0.0,
            posterior_normalization_max_error=0.0,
            log_likelihood=0.0,
            hmm_seconds=0.0,
            role=row.role,
            fold=row.fold,
        )
        frozen.append(item)
        metric_rows.append(
            {
                "well": row.well,
                "role": row.role,
                "fold": row.fold,
                "rows": 1,
                "candidate_sse": candidate_rmse**2,
                "exp226_geometry_sse": geometry_rmse**2,
                "exp435_dz_only_sse": dz_rmse**2,
                "candidate_rmse_ft": candidate_rmse,
                "exp226_geometry_rmse_ft": geometry_rmse,
                "candidate_delta_vs_exp226_geometry_ft": (
                    candidate_rmse - geometry_rmse
                ),
            }
        )
    adjusted = deepcopy(config)
    adjusted["gates"]["stage_0_technical"]["expected_rows"] = 32
    result = train.evaluate_stage0_gates(
        config=adjusted,
        scope_manifest=scope,
        frozen_wells=frozen,
        well_metrics=pd.DataFrame(metric_rows),
        prediction_artifact={
            "logical_sha256": "same",
            "readback_logical_sha256": "same",
        },
        schedule_artifact={"logical_sha256": "present"},
        ledger=train.LeakageLedger(expected_wells=32),
        elapsed_seconds=1.0,
    )
    assert all(result["technical"].values())
    assert all(result["mechanism"].values())
    assert result["stage0_all_gates_pass"] is True
    assert result["stage1_eligible_for_separate_approval"] is True
    assert result["fixed32_is_cv"] is False
    assert result["fixed32_is_promotion_evidence"] is False


def test_inference_remains_fail_closed(inference, config):
    contract = inference.validate_inference_disabled(config)
    assert contract["implementation_complete"] is True
    assert contract["stage0_completed"] is True
    assert contract["raw_test_geometry_regeneration_implemented"] is False
    with pytest.raises(RuntimeError, match="raw-test exp226 geometry"):
        inference.run_inference(config)


def test_notebook_sources_are_self_contained_and_notebook_safe():
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()
    assert "from settings import" not in train_source
    assert "from settings import" not in inference_source
    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert train_source.count("# %% [markdown]") >= 11
    assert inference_source.count("# %% [markdown]") >= 5

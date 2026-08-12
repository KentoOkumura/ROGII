from __future__ import annotations

import os
import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

EXPERIMENT_DIR = Path(
    "experiments/exp381_formation_contact_order_semimarkov_hmm"
)
TRAIN_SOURCE = EXPERIMENT_DIR / (
    "exp381_formation_contact_order_semimarkov_hmm_"
    "compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXPERIMENT_DIR / (
    "exp381_formation_contact_order_semimarkov_hmm_"
    "compact_selfcontained_inference.py"
)
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"


def load_namespace(path: Path) -> dict[str, object]:
    previous = os.environ.get("EXP381_IMPORT_ONLY")
    os.environ["EXP381_IMPORT_ONLY"] = "1"
    try:
        return runpy.run_path(str(path))
    finally:
        if previous is None:
            os.environ.pop("EXP381_IMPORT_ONLY", None)
        else:
            os.environ["EXP381_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def module() -> dict[str, object]:
    return load_namespace(TRAIN_SOURCE)


@pytest.fixture(scope="module")
def inference_module() -> dict[str, object]:
    return load_namespace(INFERENCE_SOURCE)


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(value, dict)
    return value


def test_execution_contract_is_zero_hmm_and_one_cpu_stage0_run_is_authorized(
    module: dict[str, object],
    config: dict,
) -> None:
    module["validate_execution_contract"](
        config,
        require_run_authorization=False,
    )
    assert config["experiment"]["route"] == "pf_beam"
    assert config["experiment"]["status"] == "stage0_failed_closed"
    assert config["execution"]["implementation_authorized"] is True
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["canonical_inference_notebook_adopted"] is False
    assert config["execution"]["kaggle_execution_authorized"] is True
    assert config["execution"]["stage0_run_authorized"] is True
    assert config["execution"]["one_run_authorization_consumed"] is True
    assert config["execution"]["stage1_implementation_authorized"] is False
    assert config["runtime"]["hmm_runs"] == 0
    assert config["runtime"]["pf_runs"] == 0
    assert config["runtime"]["beam_runs"] == 0
    assert config["runtime"]["lightgbm_boosters"] == 0
    assert config["runtime"]["parent_control_regeneration"] is False
    assert config["runtime"]["kaggle"]["train_run_on_push"] is True
    assert config["runtime"]["kaggle"]["enable_gpu"] is False
    assert config["runtime"]["kaggle"]["enable_internet"] is False
    module["validate_execution_contract"](
        config,
        require_run_authorization=True,
    )


def test_role_read_guard_blocks_target_truth_before_freeze(
    module: dict[str, object],
) -> None:
    ledger = module["RoleReadLedger"]()
    ledger.record_target_safe(
        0,
        "target",
        ["MD", "X", "Y", "Z", "TVT_input"],
        10,
    )
    with pytest.raises(ValueError, match="forbidden columns"):
        ledger.record_target_safe(
            0,
            "target",
            ["MD", "TVT", "ANCC"],
            10,
        )
    with pytest.raises(RuntimeError, match="before freeze"):
        ledger.record_target_truth(
            0,
            "target",
            ["MD", "Z", "TVT", *module["FORMATION_NAMES"]],
            10,
        )
    ledger.freeze()
    ledger.record_target_truth(
        0,
        "target",
        ["MD", "Z", "TVT", *module["FORMATION_NAMES"]],
        10,
    )
    assert len(ledger.pre_freeze_frame()) == 1
    assert len(ledger.late_truth_frame()) == 1


def test_formation_plane_recovers_six_planar_surfaces(
    module: dict[str, object],
) -> None:
    grid_x, grid_y = np.meshgrid(np.arange(5.0), np.arange(4.0))
    xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    surfaces = np.column_stack(
        [
            100.0 * formation_index + 2.0 * xy[:, 0] - 3.0 * xy[:, 1]
            for formation_index in range(6)
        ]
    )
    plane = module["FormationPlaneKNN"](
        wells=np.asarray([f"w{index:02d}" for index in range(len(xy))]),
        xy=xy,
        formation_medians=surfaces,
        k=10,
        chunk_rows=3,
    )
    query = np.asarray([[1.25, 0.75], [2.4, 1.6]])
    prediction, support = plane.predict(query)
    expected = np.column_stack(
        [
            100.0 * formation_index
            + 2.0 * query[:, 0]
            - 3.0 * query[:, 1]
            for formation_index in range(6)
        ]
    )
    assert np.allclose(prediction, expected, atol=1e-7)
    assert np.all(support["effective_donors"] > 1.0)
    assert not support["fallback"].any()


def test_formation_plane_uses_formation_specific_finite_donors(
    module: dict[str, object],
) -> None:
    grid_x, grid_y = np.meshgrid(np.arange(6.0), np.arange(4.0))
    xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    surfaces = np.column_stack(
        [
            100.0 * formation_index + 2.0 * xy[:, 0] - 3.0 * xy[:, 1]
            for formation_index in range(6)
        ]
    )
    surfaces[:2, 0] = np.nan
    surfaces[-1, 4] = np.nan
    plane = module["FormationPlaneKNN"](
        wells=np.asarray([f"w{index:02d}" for index in range(len(xy))]),
        xy=xy,
        formation_medians=surfaces,
        k=10,
        chunk_rows=3,
    )
    query = np.asarray([[1.25, 0.75], [2.4, 1.6]])
    prediction, support = plane.predict(query)
    expected = np.column_stack(
        [
            100.0 * formation_index
            + 2.0 * query[:, 0]
            - 3.0 * query[:, 1]
            for formation_index in range(6)
        ]
    )
    assert np.allclose(prediction, expected, atol=1e-7)
    assert support["formation_reference_counts"].tolist() == [22, 24, 24, 24, 23, 24]


def test_first_crossing_is_deterministic_linear_and_first(
    module: dict[str, object],
) -> None:
    md = np.asarray([10.0, 11.0, 12.0, 13.0, 14.0])
    residual = np.asarray([2.0, 1.0, -1.0, 1.0, -1.0])
    tvt = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])
    crossing = module["first_crossing"](md, residual, tvt=tvt)
    assert crossing is not None
    assert crossing["md"] == pytest.approx(11.5)
    assert crossing["tvt"] == pytest.approx(101.5)
    assert crossing["left_row_idx"] == 1.0

    exact = module["first_crossing"](
        md,
        np.asarray([1.0, 0.0, -1.0, 1.0, -1.0]),
    )
    assert exact is not None
    assert exact["md"] == pytest.approx(11.0)
    assert exact["fraction"] == 0.0


def test_prefix_offset_recovers_one_shared_shift(
    module: dict[str, object],
) -> None:
    centers = np.asarray([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    shift = 7.5
    tvt_input = np.asarray([10.0, 11.0, np.nan])
    z = np.asarray([-1000.0, -1001.0, -1002.0])
    surfaces = np.empty((3, 6), dtype=float)
    for row in range(3):
        surfaces[row] = (
            (tvt_input[row] if np.isfinite(tvt_input[row]) else 12.0)
            + z[row]
            - centers
            - shift
        )
    offset, value_count = module["prefix_additive_offset"](
        tvt_input,
        z,
        surfaces,
        centers,
    )
    assert offset == pytest.approx(shift)
    assert value_count == 12


def _synthetic_contact_tables(module: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth_rows = []
    prediction_rows = []
    for formation_index, formation in enumerate(module["FORMATION_NAMES"][:3]):
        true_md = 100.0 + 100.0 * formation_index
        truth_rows.append(
            {
                "fold": 0,
                "well_id": "w0",
                "formation": formation,
                "formation_index": formation_index,
                "true_md": true_md,
                "true_tvt": 1000.0 + 20.0 * formation_index,
                "true_z": -5000.0,
            }
        )
        prediction_rows.extend(
            [
                {
                    "fold": 0,
                    "well_id": "w0",
                    "method": module["PRIMARY_METHOD"],
                    "formation": formation,
                    "formation_index": formation_index,
                    "predicted_md": true_md + 1.0,
                    "predicted_tvt": 1001.0 + 20.0 * formation_index,
                    "prefix_offset": 5.0,
                    "surface_fallback_fraction": 0.0,
                    "surface_effective_donors_p05": 8.0,
                },
                {
                    "fold": 0,
                    "well_id": "w0",
                    "method": module["CONTROL_METHOD"],
                    "formation": formation,
                    "formation_index": formation_index,
                    "predicted_md": true_md + 10.0,
                    "predicted_tvt": 1002.0 + 20.0 * formation_index,
                    "prefix_offset": 4.0,
                    "surface_fallback_fraction": 0.0,
                    "surface_effective_donors_p05": 100.0,
                },
            ]
        )
    return pd.DataFrame(truth_rows), pd.DataFrame(prediction_rows)


def test_triple_match_metrics_use_fixed_order_and_paired_control(
    module: dict[str, object],
) -> None:
    truth, predictions = _synthetic_contact_tables(module)
    events = module["paired_contact_events"](
        truth,
        predictions,
        minimum_formations=2,
    )
    assert events["eligible"].all()
    assert np.allclose(events["plane_md_abs_error"], 1.0)
    assert np.allclose(events["constant_md_abs_error"], 10.0)
    order = module["order_readout"](events)
    assert order["correct_order"].tolist() == [True]
    target_manifest = pd.DataFrame(
        [{"fold": 0, "well_id": "w0", "target_safe_rows": 100}]
    )
    fold_metrics, formation_metrics, pooled = module["build_contact_metrics"](
        events=events,
        target_manifest=target_manifest,
    )
    assert pooled["crossing_md_mae_ft"] == pytest.approx(1.0)
    assert pooled["gain_vs_constant_surface_ft"] == pytest.approx(9.0)
    assert pooled["contact_tvt_rmse_ft"] == pytest.approx(1.0)
    assert pooled["correct_order_fraction"] == pytest.approx(1.0)
    assert len(fold_metrics) == 1
    assert len(formation_metrics) == 6


def test_stage0_gate_is_fixed_and_requires_all_checks(
    module: dict[str, object],
    config: dict,
) -> None:
    pooled = {
        "eligible_well_fraction": 0.50,
        "contact_event_count": 1200,
        "crossing_md_mae_ft": 100.0,
        "crossing_md_p90_ft": 300.0,
        "contact_tvt_rmse_ft": 10.0,
        "correct_order_fraction": 0.98,
        "gain_vs_constant_surface_ft": 1.0,
    }
    fold_metrics = pd.DataFrame(
        {"gain_vs_constant_surface_ft": [1.0, 1.0, 1.0, 1.0, -0.1]}
    )
    passing = module["evaluate_stage0_gate"](pooled, fold_metrics, config)
    assert passing["passed"] is True
    assert passing["positive_fold_count"] == 4

    failing_pooled = deepcopy(pooled)
    failing_pooled["contact_tvt_rmse_ft"] = 15.01
    failing = module["evaluate_stage0_gate"](
        failing_pooled,
        fold_metrics,
        config,
    )
    assert failing["passed"] is False
    assert failing["checks"]["contact_tvt_rmse_ft"]["passed"] is False


def test_inference_candidate_fails_closed(
    inference_module: dict[str, object],
    config: dict,
) -> None:
    inference_module["validate_inference_is_disabled"](config)
    with pytest.raises(RuntimeError, match="no inference candidate"):
        inference_module["fail_closed"](config)


def test_compact_sources_are_self_contained_and_canonical_notebooks_exist() -> None:
    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert "Path(__file__)" not in train_text
    assert "Path(__file__)" not in inference_text
    assert "from settings import" not in train_text
    assert "from settings import" not in inference_text
    assert train_text.count("# %% [markdown]") >= 11
    assert "## 7. Target-free freeze and 16-well resource readout" in train_text
    assert "## 8. Validation-truth late join and contact metrics" in train_text
    assert "## 9. Fixed Stage 0 AND gate and generated artifacts" in train_text
    assert (
        EXPERIMENT_DIR
        / "exp381_formation_contact_order_semimarkov_hmm_train.ipynb"
    ).exists()
    assert (
        EXPERIMENT_DIR
        / "exp381_formation_contact_order_semimarkov_hmm_inference.ipynb"
    ).exists()

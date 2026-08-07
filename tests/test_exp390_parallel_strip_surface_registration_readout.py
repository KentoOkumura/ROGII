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
    "experiments/exp390_parallel_strip_surface_registration_readout"
)
TRAIN_SOURCE = EXPERIMENT_DIR / (
    "exp390_parallel_strip_surface_registration_readout_"
    "compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXPERIMENT_DIR / (
    "exp390_parallel_strip_surface_registration_readout_"
    "compact_selfcontained_inference.py"
)
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"


def load_namespace(path: Path = TRAIN_SOURCE) -> dict[str, object]:
    previous = os.environ.get("EXP390_IMPORT_ONLY")
    os.environ["EXP390_IMPORT_ONLY"] = "1"
    try:
        return runpy.run_path(str(path))
    finally:
        if previous is None:
            os.environ.pop("EXP390_IMPORT_ONLY", None)
        else:
            os.environ["EXP390_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def module() -> dict[str, object]:
    return load_namespace()


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(value, dict)
    return value


def test_execution_contract_records_authorized_cpu_preflight(
    module: dict[str, object],
    config: dict,
) -> None:
    module["validate_execution_contract"](
        config,
        require_kaggle_authorization=False,
    )
    assert config["experiment"]["route"] == "pf_beam"
    assert config["execution"]["implementation_authorized"] is True
    assert (
        config["execution"]["implementation_approval_source"]
        == "user_message_implement_exp390_2026_07_24"
    )
    assert config["execution"]["canonical_notebook_adoption_authorized"] is True
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["canonical_inference_notebook_adopted"] is False
    assert config["execution"]["kaggle_package_authorized"] is True
    assert config["execution"]["kaggle_push_authorized"] is True
    assert config["execution"]["kaggle_execution_authorized"] is True
    assert (
        config["execution"]["kaggle_execution_approval_source"]
        == "user_message_run_exp390_2026_07_24"
    )
    assert config["execution"]["full_run_authorized"] is False
    assert config["execution"]["current_mode"] == "stage0_resource_preflight"
    assert config["execution"]["inference_enabled"] is False
    assert config["execution"]["submission_enabled"] is False
    module["validate_execution_contract"](
        config,
        require_kaggle_authorization=True,
    )


def test_train_directory_resolution_rejects_three_well_test_dir(
    module: dict[str, object],
    tmp_path: Path,
) -> None:
    test_dir = tmp_path / "test"
    train_dir = tmp_path / "train"
    test_dir.mkdir()
    train_dir.mkdir()
    test_files = [
        test_dir / f"test{index}__horizontal_well.csv"
        for index in range(3)
    ]
    train_files = [
        train_dir / f"train{index}__horizontal_well.csv"
        for index in range(7)
    ]
    assert module["select_train_dir"](
        sorted(test_files + train_files),
        7,
    ) == train_dir
    with pytest.raises(FileNotFoundError, match="exactly 8 files"):
        module["select_train_dir"](
            sorted(test_files + train_files),
            8,
        )


def test_fold_assignment_is_stable_and_parent_parity_is_checked(
    module: dict[str, object],
) -> None:
    wells = [f"w{index:03d}" for index in range(31)]
    first = module["assign_group_folds"](wells, 5, 42)
    second = module["assign_group_folds"](list(reversed(wells)), 5, 42)
    assert first == second
    parent = pd.DataFrame(
        [
            {
                "well_id": well,
                "fold": fold,
                "row_idx": 0,
                "suffix_offset": 0,
                "tvt_pred": 1.0,
            }
            for well, fold in first.items()
        ]
    )
    module["validate_fold_identity"](first, parent)
    broken = parent.copy()
    broken.loc[0, "fold"] = (int(broken.loc[0, "fold"]) + 1) % 5
    with pytest.raises(ValueError, match="fold identity mismatch"):
        module["validate_fold_identity"](first, broken)


def test_role_read_guard_and_truth_late_boundary(
    module: dict[str, object],
) -> None:
    ledger = module["RoleReadLedger"]()
    ledger.record_target(["MD", "X", "Y", "Z", "TVT_input"], 10, 0)
    with pytest.raises(ValueError, match="forbidden columns"):
        ledger.record_target(["MD", "TVT", "GR"], 10, 0)
    fresh = module["RoleReadLedger"]()
    with pytest.raises(RuntimeError, match="before prediction freeze"):
        fresh.record_truth_late(10)
    fresh.freeze()
    fresh.record_truth_late(10)
    assert fresh.truth_joined_after_freeze is True


def test_pca_axis_is_canonical_and_angle_is_modulo_pi(
    module: dict[str, object],
) -> None:
    x = np.linspace(100.0, -100.0, 101)
    y = 2.0 * x + 5.0
    axis = module["canonical_pca_axis"](x, y)
    vector = np.asarray([axis.es_x, axis.es_y])
    assert vector[np.argmax(np.abs(vector))] > 0.0
    reverse = module["Axis2D"](
        es_x=-axis.es_x,
        es_y=-axis.es_y,
        en_x=-axis.en_x,
        en_y=-axis.en_y,
        centroid_x=0.0,
        centroid_y=0.0,
    )
    assert module["axial_angle_mismatch_deg"](axis, reverse) == pytest.approx(0.0)


def test_overlap_and_pair_eligibility_use_same_query_axis(
    module: dict[str, object],
    config: dict,
) -> None:
    assert module["projected_overlap_fraction"](0.0, 100.0, 20.0, 80.0) == 0.6
    target = pd.DataFrame(
        {
            "well_id": "query",
            "row_idx": np.arange(101),
            "MD": np.arange(101, dtype=float),
            "X": np.arange(101, dtype=float),
            "Y": np.zeros(101),
            "Z": -1000.0,
            "TVT_input": np.nan,
            "fold": 0,
            "role": "outer_valid",
        }
    )
    parent = pd.DataFrame({"row_idx": np.arange(101)})
    geometry = module["build_query_geometry"](target, parent)
    donor_frame = pd.DataFrame(
        {
            "well_id": "donor",
            "row_idx": np.arange(121),
            "MD": np.arange(121, dtype=float),
            "X": np.arange(-10.0, 111.0),
            "Y": 100.0,
            "Z": -1000.0,
            "TVT": 2000.0,
        }
    )
    donor = module["build_donor_track"](donor_frame)
    result = module["evaluate_pair"]("query", geometry, donor, config)
    assert result["eligible"] is True
    assert result["projected_overlap"] == pytest.approx(1.0)
    assert result["median_abs_cross_track_distance_ft"] == pytest.approx(100.0)
    assert result["projected_s_monotone_step_fraction"] == pytest.approx(1.0)


def _node_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(
        {
            "fold": 0,
            "well_id": "query",
            "node_idx": [0, 1, 2],
            "node_md": [0.0, 64.0, 128.0],
            "query_x": [0.0, 64.0, 128.0],
            "query_y": [0.0, 0.0, 0.0],
        }
    )
    records: list[dict[str, object]] = []
    for node in nodes.itertuples(index=False):
        for index, donor_n in enumerate((-300.0, -100.0, 100.0, 300.0)):
            intercept = 1000.0 + 0.25 * float(node.node_md)
            records.append(
                {
                    "fold": 0,
                    "well_id": "query",
                    "node_idx": int(node.node_idx),
                    "node_md": float(node.node_md),
                    "node_s": float(node.node_md),
                    "donor_well_id": f"d{index}",
                    "donor_n": donor_n,
                    "donor_surface": intercept + 0.1 * donor_n,
                    "donor_surface_circular": intercept + 10.0 + 0.1 * donor_n,
                }
            )
    return nodes, pd.DataFrame(records)


def test_two_sided_huber_fit_recovers_intercept_and_rejects_one_side(
    module: dict[str, object],
    config: dict,
) -> None:
    nodes, samples = _node_samples()
    fitted = module["fit_strip_nodes"](nodes, samples, config)
    assert fitted["fit_valid"].all()
    assert fitted["intercept_raw"].to_numpy() == pytest.approx(
        [1000.0, 1016.0, 1032.0],
        abs=1.0e-6,
    )
    assert fitted.loc[fitted["node_idx"].eq(1), "intercept_smooth"].iloc[0] == pytest.approx(
        1016.0
    )
    one_side = samples.loc[samples["donor_n"] > 0.0]
    rejected = module["fit_strip_nodes"](nodes, one_side, config)
    assert not rejected["fit_valid"].any()
    assert set(rejected["status"]) == {"low_support"}


def test_same_s_interpolation_preserves_surface_and_cross_track_sign(
    module: dict[str, object],
) -> None:
    nodes = pd.DataFrame(
        {
            "fold": 0,
            "well_id": "query",
            "node_idx": [0, 1, 2],
            "node_md": [0.0, 50.0, 100.0],
            "query_x": [0.0, 50.0, 100.0],
            "query_y": [0.0, 0.0, 0.0],
        }
    )
    donor = module["DonorTrack"](
        well_id="donor",
        md=np.arange(101, dtype=float),
        x=np.arange(101, dtype=float),
        y=np.full(101, 25.0),
        surface=1000.0 + 2.0 * np.arange(101, dtype=float),
        axis=module["canonical_pca_axis"](
            np.arange(101, dtype=float),
            np.full(101, 25.0),
        ),
    )
    axis = module["canonical_pca_axis"](
        nodes["query_x"],
        nodes["query_y"],
    )
    result = module["interpolate_donor_at_nodes"](nodes, donor, axis, 10)
    assert result["donor_n"].to_numpy() == pytest.approx([25.0, 25.0, 25.0])
    assert result["donor_surface"].to_numpy() == pytest.approx(
        [1000.0, 1100.0, 1200.0]
    )


def test_prefix_calibration_uses_only_requested_rows(
    module: dict[str, object],
    config: dict,
) -> None:
    local = deepcopy(config)
    local["strip_coordinate"]["prefix_calibration"]["minimum_finite_prefix_rows"] = 4
    rows = pd.DataFrame(
        {
            "fold": 0,
            "well_id": "query",
            "row_idx": np.arange(8),
            "MD": np.arange(8, dtype=float),
            "Z": -1000.0,
            "TVT_input": 1103.0 + np.arange(8, dtype=float),
            "strip_surface_raw": 100.0 + np.arange(8, dtype=float),
            "strip_valid": True,
        }
    )
    mask = np.asarray([True, True, True, True, False, False, False, False])
    calibrated, record = module["calibrate_prefix"](
        rows,
        "strip_surface_raw",
        local,
        mask,
    )
    assert record["finite_prefix_rows"] == 4
    assert record["vertical_gauge_offset"] == pytest.approx(3.0)
    assert calibrated.to_numpy() == pytest.approx(
        103.0 + np.arange(8, dtype=float)
    )


def test_final_candidate_uses_exact_exp226_fallback(
    module: dict[str, object],
    config: dict,
) -> None:
    local = deepcopy(config)
    local["strip_coordinate"]["prefix_calibration"]["minimum_finite_prefix_rows"] = 2
    rows = pd.DataFrame(
        {
            "fold": 0,
            "well_id": "query",
            "row_idx": [0, 1, 2, 3],
            "MD": [0.0, 1.0, 2.0, 3.0],
            "X": 0.0,
            "Y": 0.0,
            "Z": -1000.0,
            "TVT_input": [1101.0, 1102.0, np.nan, np.nan],
            "strip_surface_raw": [2100.0, 2101.0, 2102.0, np.nan],
            "strip_surface_circular_raw": [2100.0, 2101.0, 2102.0, np.nan],
            "strip_valid": [True, True, True, False],
            "strip_support_reason": ["eligible", "eligible", "eligible", "one_sided"],
        }
    )
    parent = pd.DataFrame(
        {
            "well_id": "query",
            "row_idx": [2, 3],
            "suffix_offset": [0, 1],
            "tvt_pred": [9999.0, 8888.0],
            "fold": 0,
        }
    )
    _, prediction, _ = module["attach_final_candidate"](rows, parent, local)
    strip_value = prediction.loc[
        prediction["row_idx"].eq(2),
        "exp390_prediction",
    ].iloc[0]
    fallback_value = prediction.loc[
        prediction["row_idx"].eq(3),
        "exp390_prediction",
    ].iloc[0]
    fallback_status = prediction.loc[
        prediction["row_idx"].eq(3),
        "candidate_status",
    ].iloc[0]
    assert strip_value == pytest.approx(1103.0)
    assert fallback_value == 8888.0
    assert fallback_status == "exp226_fallback"


def test_inference_candidate_is_fail_closed(config: dict) -> None:
    module = load_namespace(INFERENCE_SOURCE)
    with pytest.raises(RuntimeError, match="inference is disabled"):
        module["assert_inference_authorized"](config)

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
EXP = "exp420_exp226_hmm_guided_defensive_mixture_pf"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"
EXP419_SOURCE = (
    ROOT
    / "experiments"
    / "exp419_exp226_guided_defensive_mixture_pf"
    / "exp419_exp226_guided_defensive_mixture_pf_compact_selfcontained_train.py"
)
EXP404_SOURCE = (
    ROOT
    / "experiments"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation"
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_train.py"
)
FIXED32 = (
    ROOT
    / "experiments"
    / "exp411_predictive_filtered_rate_innovation_destick"
    / "assets"
    / "stage0_fixed32_manifest.csv"
)
SENTINEL12 = (
    ROOT
    / "experiments"
    / "exp410_likpf_particle_resampling_basin_audit"
    / "assets"
    / "pf_counterfactual_sentinel_wells.csv"
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
    previous = os.environ.get("EXP420_IMPORT_ONLY")
    os.environ["EXP420_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp420_train_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP420_IMPORT_ONLY", None)
        else:
            os.environ["EXP420_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def exp419() -> ModuleType:
    previous = os.environ.get("EXP419_IMPORT_ONLY")
    os.environ["EXP419_IMPORT_ONLY"] = "1"
    try:
        return load_module(EXP419_SOURCE, "exp419_parent_for_exp420")
    finally:
        if previous is None:
            os.environ.pop("EXP419_IMPORT_ONLY", None)
        else:
            os.environ["EXP419_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def exp404() -> ModuleType:
    return load_module(EXP404_SOURCE, "exp404_parent_for_exp420")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def synthetic_pf_inputs() -> tuple[np.ndarray, ...]:
    md = np.arange(1.0, 9.0, dtype=np.float64)
    z = np.linspace(0.0, 0.7, len(md), dtype=np.float64)
    gr = np.asarray([50.0, 52.0, 54.0, 53.0, 51.0, 50.0, 49.0, 51.0])
    geometry = np.linspace(-0.01, 0.02, len(md))
    direction = np.asarray([0, 0, 1, 1, 1, -1, -1, 0], dtype=np.int8)
    grid_gr = np.linspace(40.0, 70.0, 151, dtype=np.float64)
    return md, z, gr, geometry, direction, grid_gr


def test_scientific_contract_records_fixed_stage0_and_full_costs(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)
    assert contract["primary_candidate"] == (
        "exp226_hmm_guided_defensive_mixture_scale5"
    )
    assert contract["execution_counts"]["stage_0"][
        "hmm_signal_well_runs"
    ] == 44
    assert contract["execution_counts"]["stage_0"][
        "candidate_pf_well_runs"
    ] == 44
    assert contract["execution_counts"]["full"][
        "hmm_signal_well_runs"
    ] == 773
    assert contract["execution_counts"]["full"][
        "candidate_pf_well_runs"
    ] == 773
    assert contract["execution_counts"]["lightgbm_configs"] == 0
    assert contract["execution_counts"]["boosters"] == 0
    assert contract["execution_counts"]["gpu_runs"] == 0
    assert len(contract["scientific_contract_sha256"]) == 64
    with pytest.raises(RuntimeError, match="package/push/train run is not approved"):
        train.validate_scientific_contract(config, require_run_approval=True)


def test_schedule_and_proposal_contracts_are_exact_and_fail_closed(
    train: ModuleType,
    config: dict,
) -> None:
    schedule = train.hmm_schedule_contract(config)
    proposal = train.proposal_contract(config)
    assert schedule["uses_backward_pass"] is False
    assert schedule["activation_transitions"] == 32
    assert schedule["refractory_rows"] == 128
    assert schedule["first_affected_transition"] == "row_after_trigger"
    assert proposal["target_weight"] == 0.5
    assert proposal["inactive_geometry_weights"] == [1.0 / 6.0] * 3
    assert proposal["active_geometry_weights"] == [1.0 / 12.0] * 3
    assert proposal["active_hmm_weights"] == [1.0 / 12.0] * 3
    assert proposal["inactive_weight_sum"] == 1.0
    assert proposal["active_weight_sum"] == 1.0
    assert proposal["importance_clipping"] is False

    broken = copy.deepcopy(config)
    broken["model"]["hmm_signal"]["cusum"][
        "activation_transitions"
    ] = 31
    with pytest.raises(ValueError, match="schedule contract changed"):
        train.hmm_schedule_contract(broken)

    broken = copy.deepcopy(config)
    broken["model"]["rate_proposal"]["active"][
        "hmm_component_weights"
    ][0] = 0.1
    with pytest.raises(ValueError, match="do not sum to one"):
        train.proposal_contract(broken)


def test_fixed44_manifest_is_sha_pinned_disjoint_and_unique(config: dict) -> None:
    assert hashlib.sha256(FIXED32.read_bytes()).hexdigest() == config["data"][
        "stage_0_fixed32"
    ]["expected_sha256"]
    assert hashlib.sha256(SENTINEL12.read_bytes()).hexdigest() == config["data"][
        "stage_0_pf_sentinel"
    ]["expected_sha256"]
    fixed = set(pd.read_csv(FIXED32, dtype={"well": str})["well"])
    sentinel = set(pd.read_csv(SENTINEL12, dtype={"well": str})["well"])
    assert len(fixed) == 32
    assert len(sentinel) == 12
    assert not fixed & sentinel
    assert len(fixed | sentinel) == 44


def test_exp263_physical_anchor_inputs_keep_parent_value_semantics(
    config: dict,
) -> None:
    exp072 = config["data"]["exp072_saved_likpf"]
    exp209 = config["data"]["exp209_saved_hmm"]
    assert exp072["residual_column"] == "likpf_mean_d"
    assert exp072["anchor_column"] == "last_known_tvt"
    assert exp072["transform"] == "anchor_plus"
    assert exp209["prediction_column"] == "hmm_mean_tvt"
    assert config["guards"]["physical_anchor"][
        "reference"
    ] == "exp263_fixed_physical_blend"


def test_inactive_and_active_importance_ratios_are_positive_and_bounded(
    train: ModuleType,
) -> None:
    sigma_multipliers = np.asarray([1.0, 4.0, 16.0])
    inactive_geometry = np.asarray([1.0 / 6.0] * 3)
    active_geometry = np.asarray([1.0 / 12.0] * 3)
    active_hmm = np.asarray([1.0 / 12.0] * 3)
    samples = np.linspace(-0.15, 0.15, 1201)
    inactive = np.asarray(
        [
            train.scheduled_mixture_importance_ratio(
                float(value),
                0.01,
                -0.02,
                0,
                0.005,
                0.002,
                0.5,
                inactive_geometry,
                np.zeros(3),
                sigma_multipliers,
            )
            for value in samples
        ]
    )
    active = np.asarray(
        [
            train.scheduled_mixture_importance_ratio(
                float(value),
                0.01,
                -0.02,
                1,
                0.005,
                0.002,
                0.5,
                active_geometry,
                active_hmm,
                sigma_multipliers,
            )
            for value in samples
        ]
    )
    assert np.isfinite(inactive).all() and np.isfinite(active).all()
    assert (inactive >= 0.0).all() and (active >= 0.0).all()
    assert float(inactive.max()) <= 2.0 + 1.0e-12
    assert float(active.max()) <= 2.0 + 1.0e-12


def test_all_guidance_zero_is_exp404_exact_rng_parity(
    train: ModuleType,
    exp404: ModuleType,
) -> None:
    md, z, gr, geometry, _, grid_gr = synthetic_pf_inputs()
    expected = exp404._pf_lik_allseeds(
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
    observed = train._pf_guided_allseeds(
        md,
        z,
        gr,
        geometry,
        np.ones(len(md), dtype=np.int8),
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
        np.asarray([1.0 / 6.0] * 3),
        np.asarray([1.0 / 12.0] * 3),
        np.asarray([1.0 / 12.0] * 3),
        np.asarray([1.0, 4.0, 16.0]),
        0.005,
    )
    for index in range(5):
        assert np.array_equal(observed[index], expected[index])


def test_hmm_weight_zero_is_exp419_exact_rng_parity(
    train: ModuleType,
    exp419: ModuleType,
) -> None:
    md, z, gr, geometry, direction, grid_gr = synthetic_pf_inputs()
    expected = exp419._pf_guided_allseeds(
        md,
        z,
        gr,
        geometry,
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        48,
        6,
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
        np.asarray([1.0 / 6.0] * 3),
        np.asarray([1.0, 4.0, 16.0]),
    )
    observed = train._pf_guided_allseeds(
        md,
        z,
        gr,
        geometry,
        direction,
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        48,
        6,
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
        np.asarray([1.0 / 6.0] * 3),
        np.asarray([1.0 / 6.0] * 3),
        np.zeros(3),
        np.asarray([1.0, 4.0, 16.0]),
        0.005,
    )
    for index in range(8):
        assert np.array_equal(observed[index], expected[index])
    assert np.array_equal(observed[8][:, :4], expected[8])
    assert not observed[8][:, 4:].any()
    for index in (9, 10):
        assert np.array_equal(observed[index], expected[index])


def test_active_schedule_uses_all_seven_components_and_respects_bound(
    train: ModuleType,
) -> None:
    md, z, gr, geometry, direction, grid_gr = synthetic_pf_inputs()
    outputs = train._pf_guided_allseeds(
        md,
        z,
        gr,
        geometry,
        direction,
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        256,
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
        np.asarray([1.0 / 6.0] * 3),
        np.asarray([1.0 / 12.0] * 3),
        np.asarray([1.0 / 12.0] * 3),
        np.asarray([1.0, 4.0, 16.0]),
        0.005,
    )
    assert np.isfinite(outputs[0]).all()
    assert float(outputs[6].max()) <= 2.0 + 1.0e-12
    assert outputs[8].shape == (8, 7)
    assert (outputs[8].sum(axis=0) > 0).all()
    assert outputs[9].shape == (8, len(md))
    assert outputs[10].shape == (8, len(md))


def test_synthetic_kernel_parity_report_is_bitwise(train: ModuleType) -> None:
    report = train.synthetic_kernel_parity_report()
    assert report["all_guidance_zero_exp404_rng_parity"] is True
    assert report["hmm_weight_zero_exp419_rng_parity"] is True
    assert report["all_guidance_zero_prediction_max_abs_ft"] == 0.0
    assert report["hmm_weight_zero_prediction_max_abs_ft"] == 0.0


def test_untreated_hmm_trigger_first_affects_the_next_transition(
    train: ModuleType,
) -> None:
    time_count = 24
    position_count = 31
    rates = np.linspace(-0.10, 0.10, 41)
    emission = np.full((time_count, position_count), -20.0, dtype=np.float32)
    for row in range(time_count):
        emission[row, min(position_count - 1, 8 + row)] = 0.0
    outputs = train._untreated_hmm_forward_schedule(
        emission,
        np.ones(time_count),
        np.zeros(time_count),
        0.35,
        rates,
        0.002,
        0.02,
        8.0,
        0.75,
        0.0,
        0.01,
        1.0,
        0.998,
        0.005,
        0.0,
        1.0e-6,
        1.0e-6,
        1.0e-12,
        3,
        4,
    )
    trigger = outputs[5]
    active = outputs[6]
    trigger_rows = np.flatnonzero(trigger)
    assert len(trigger_rows) >= 1
    first = int(trigger_rows[0])
    assert active[first] == 0
    expected_stop = min(first + 4, time_count)
    np.testing.assert_array_equal(
        active[first + 1 : expected_stop],
        np.full(expected_stop - first - 1, trigger[first], dtype=np.int8),
    )
    assert set(np.unique(active)) <= {-1, 0, 1}
    assert np.isfinite(outputs[0]).all()
    assert np.isfinite(outputs[1]).all()
    assert float(outputs[7]) <= 1.0e-5


def test_geometry_loader_reads_only_prefreeze_allowlist(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "geometry.csv.gz"
    pd.DataFrame(
        {
            "well_id": ["w", "w"],
            "row_idx": [3, 4],
            "suffix_offset": [0, 1],
            "tvt_geop": [100.0, 101.0],
            "tvt_pred": [999.0, 999.0],
            "tvt_true": [100.0, 101.0],
            "fold": [0, 0],
        }
    ).to_csv(path, index=False, compression="gzip")
    observed_usecols: list[str] = []
    original = pd.read_csv

    def recording_read_csv(*args, **kwargs):
        usecols = kwargs.get("usecols")
        if usecols is not None:
            observed_usecols.extend(list(usecols))
        return original(*args, **kwargs)

    monkeypatch.setattr(train.pd, "read_csv", recording_read_csv)
    loaded = train.load_fold_safe_geometry(path, config)
    assert list(loaded.columns) == [
        "well_id",
        "row_idx",
        "suffix_offset",
        "tvt_geop",
    ]
    assert observed_usecols == [
        "well_id",
        "row_idx",
        "suffix_offset",
        "tvt_geop",
    ]


def test_schedule_and_candidate_freeze_records_separate_shas(
    train: ModuleType,
    tmp_path: Path,
) -> None:
    candidate = pd.DataFrame(
        {
            "id": ["w_3", "w_4"],
            "well_id": ["w", "w"],
            "row_idx": np.asarray([3, 4], dtype=np.int64),
            "suffix_offset": np.asarray([0, 1], dtype=np.int64),
            "hmm_active": [False, True],
            "hmm_direction": np.asarray([0, 1], dtype=np.int8),
            train.PRIMARY_CANDIDATE: np.asarray(
                [100.0, 101.0],
                dtype=np.float32,
            ),
        }
    )
    report = train.freeze_prediction_frame(
        candidate,
        tmp_path / "candidate.csv.gz",
    )
    assert len(report["logical_content_sha256"]) == 64
    assert len(report["schedule_logical_sha256"]) == 64
    assert report["logical_content_sha256"] != report[
        "schedule_logical_sha256"
    ]


def test_no_notebook_unsafe_file_reference_or_parent_helper_import() -> None:
    source = SOURCE.read_text()
    assert "__file__" not in source
    assert "import exp419" not in source
    assert "import exact_hmm_smoother" not in source

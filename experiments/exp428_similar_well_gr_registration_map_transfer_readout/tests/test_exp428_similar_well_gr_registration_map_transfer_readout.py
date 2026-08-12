from __future__ import annotations

import copy
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
EXP = "exp428_similar_well_gr_registration_map_transfer_readout"
EXP_DIR = ROOT / "experiments" / EXP
SOURCE = EXP_DIR / f"{EXP}_compact_selfcontained_train.py"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train() -> ModuleType:
    previous = os.environ.get("EXP428_IMPORT_ONLY")
    os.environ["EXP428_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp428_train_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP428_IMPORT_ONLY", None)
        else:
            os.environ["EXP428_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def make_profile(
    train: ModuleType,
    well: str,
    gr: np.ndarray,
) -> object:
    rows = len(gr)
    md = np.arange(rows, dtype=np.float64)
    prepared = {
        "progress": np.linspace(0.0, 1.0, rows),
        "normalized": np.interp(
            np.linspace(0.0, 1.0, 256),
            np.linspace(0.0, 1.0, rows),
            (gr - np.nanmedian(gr)) / np.nanstd(gr),
        ),
        "support": np.ones(256, dtype=bool),
    }
    return train.SuffixProfile(
        well=well,
        row_idx=np.arange(rows, dtype=np.int64),
        md=md,
        progress=prepared["progress"],
        gr_raw=np.asarray(gr, dtype=np.float64),
        gr_normalized=prepared["normalized"],
        support_mask=prepared["support"],
        finite_fraction=float(np.isfinite(gr).mean()),
        robust_scale=float(np.nanstd(gr)),
    )


def synthetic_registration(
    train: ModuleType,
    *,
    shift: float = 10.0,
    rows: int = 1536,
) -> tuple[object, np.ndarray, pd.DataFrame]:
    type_tvt = np.arange(800.0, 1400.01, 0.1)

    def curve(values: np.ndarray) -> np.ndarray:
        return (
            80.0
            + 18.0 * np.sin(values / 2.7)
            + 9.0 * np.sin(values / 0.83)
            + 3.0 * np.cos(values / 0.31)
        )

    true_tvt = np.linspace(950.0, 1200.0, rows)
    horizontal_gr = curve(true_tvt + shift)
    profile = make_profile(train, "well", horizontal_gr)
    typewell = pd.DataFrame({"TVT": type_tvt, "GR": curve(type_tvt)})
    return profile, true_tvt, typewell


def test_contract_records_authorized_zero_compute_counts(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)
    assert config["implementation"]["implementation_approved"] is True
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["kaggle_package_approved"] is True
    assert config["execution"]["kaggle_run_approved"] is True
    assert contract["primary"] == train.PRIMARY
    assert train.execution_counts(config) == {
        "audit_variants": 1,
        "reporting_folds": 5,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "pf_well_runs": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "parent_control_reruns": 0,
    }
    assert len(contract["scientific_contract_sha256"]) == 64
    unapproved = copy.deepcopy(config)
    unapproved["execution"]["kaggle_run_approved"] = False
    with pytest.raises(PermissionError, match="not approved"):
        train.validate_scientific_contract(
            unapproved,
            require_run_approval=True,
        )


def test_contract_rejects_shift_grid_and_primary_changes(
    train: ModuleType,
    config: dict,
) -> None:
    changed = copy.deepcopy(config)
    changed["registration_map"]["shift_grid_ft"][-1] = 81.0
    with pytest.raises(ValueError, match="shift grid"):
        train.validate_scientific_contract(changed)
    changed = copy.deepcopy(config)
    changed["candidates"]["primary"]["name"] = "selected_after_readout"
    with pytest.raises(ValueError, match="primary"):
        train.validate_scientific_contract(changed)


def test_fold_assignment_matches_parent_deterministic_contract(
    train: ModuleType,
) -> None:
    wells = [f"well{index:02d}" for index in range(20)]
    first = train.deterministic_well_folds(wells, n_folds=5, seed=42)
    second = train.deterministic_well_folds(
        list(reversed(wells)), n_folds=5, seed=42
    )
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("fold").size().tolist() == [4, 4, 4, 4, 4]


def test_axis_graph_uses_tvt_delta_and_ignores_row_lag(
    train: ModuleType,
    config: dict,
) -> None:
    assignments = pd.DataFrame(
        {
            "well": ["a", "b", "c"],
            "typewell_group_id": ["g", "g", "g"],
            "representative_well": ["a", "a", "a"],
        }
    )
    pairs = pd.DataFrame(
        {
            "well_id_a": ["a", "b", "a"],
            "well_id_b": ["b", "c", "c"],
            "exact_match_rate": [1.0, 1.0, 1.0],
            "overlap_fraction_shorter": [1.0, 1.0, 1.0],
            "tvt_delta_b_minus_a_median": [2.0, 3.0, 5.0],
            "tvt_delta_b_minus_a_min": [2.0, 3.0, 5.0],
            "tvt_delta_b_minus_a_max": [2.0, 3.0, 5.0],
            "row_lag_b_minus_a": [100, -200, 999],
        }
    )
    graph, audit = train.build_typewell_axis_graph(assignments, pairs, config)
    offsets = graph.set_index("well")["axis_offset_ft"].to_dict()
    assert offsets == {"a": 0.0, "b": 2.0, "c": 5.0}
    assert audit["typewell_axis_graph_conflicts"] == 0


def test_axis_graph_detects_cycle_conflict(
    train: ModuleType,
    config: dict,
) -> None:
    assignments = pd.DataFrame(
        {
            "well": ["a", "b", "c"],
            "typewell_group_id": ["g", "g", "g"],
            "representative_well": ["a", "a", "a"],
        }
    )
    pairs = pd.DataFrame(
        {
            "well_id_a": ["a", "b", "a"],
            "well_id_b": ["b", "c", "c"],
            "exact_match_rate": 1.0,
            "overlap_fraction_shorter": 1.0,
            "tvt_delta_b_minus_a_median": [2.0, 3.0, 6.0],
            "tvt_delta_b_minus_a_min": [2.0, 3.0, 6.0],
            "tvt_delta_b_minus_a_max": [2.0, 3.0, 6.0],
        }
    )
    graph, audit = train.build_typewell_axis_graph(assignments, pairs, config)
    assert audit["cycle_conflicts"] > 0
    assert not graph["axis_group_valid"].any()


def test_registration_map_recovers_positive_shift(
    train: ModuleType,
    config: dict,
) -> None:
    profile, true_tvt, typewell = synthetic_registration(train, shift=10.0)
    observed = train.estimate_registration_map(
        "well", profile, true_tvt, typewell, config
    )
    assert observed.supported
    assert observed.identifiable_blocks == 3
    assert observed.global_shift_ft == pytest.approx(10.0)
    assert observed.block_summary["best_shift_ft"].eq(10.0).all()


def test_registration_shift_sign_and_axis_conversion(train: ModuleType) -> None:
    assert train.transfer_shift(
        10.0, query_axis_offset=3.0, donor_axis_offset=1.0
    ) == pytest.approx(12.0)


def test_raw_finite_zncc_respects_minimum_pairs(train: ModuleType) -> None:
    x = np.asarray([1.0, 2.0, np.nan, 4.0])
    y = np.asarray([2.0, 4.0, 8.0, 8.0])
    score, pairs = train.raw_finite_zncc(x, y, minimum_pairs=3)
    assert pairs == 3
    assert score == pytest.approx(1.0)
    score, pairs = train.raw_finite_zncc(x, y, minimum_pairs=4)
    assert pairs == 3
    assert np.isnan(score)


def test_dtw_identity_and_local_progress_mapping(train: ModuleType) -> None:
    values = np.sin(np.linspace(0.0, 8.0, 256))
    dtw = train.constrained_dtw(values, values, band=32, max_run=4)
    assert dtw["normalized_cost"] == pytest.approx(0.0)
    query_progress = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    mapped = train.query_to_donor_progress(
        dtw["query_path"],
        dtw["donor_path"],
        query_progress,
        n_points=256,
    )
    np.testing.assert_allclose(mapped, query_progress, atol=1.0 / 255.0)


def test_preprocessing_keeps_internal_interpolation_available_to_dtw(
    train: ModuleType,
    config: dict,
) -> None:
    md = np.arange(400, dtype=np.float64)
    gr = 80.0 + 10.0 * np.sin(md / 13.0)
    gr[20:-20:5] = np.nan
    prepared = train.preprocess_suffix_gr(md, gr, config)
    assert prepared["support"].mean() >= 0.70
    assert (~prepared["support"]).any()
    assert np.isfinite(prepared["normalized"]).all()
    dtw = train.constrained_dtw(
        prepared["normalized"],
        prepared["normalized"],
        band=32,
        max_run=4,
    )
    assert dtw["normalized_cost"] == pytest.approx(0.0)


def test_stable_random_control_is_order_independent(train: ModuleType) -> None:
    donors = ["d3", "d1", "d2"]
    first = train.stable_random_donor("query", donors)
    second = train.stable_random_donor("query", list(reversed(donors)))
    assert first == second
    assert first in donors


def test_target_free_generation_uses_only_outer_train_donor_truth(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    base_profile, true_tvt, typewell = synthetic_registration(train, shift=10.0)
    profiles = {
        well: make_profile(train, well, base_profile.gr_raw.copy())
        for well in ("q", "d1", "d2")
    }
    for donor in ("d1", "d2"):
        pd.DataFrame({"TVT": true_tvt}).to_csv(
            tmp_path / f"{donor}__horizontal_well.csv", index=False
        )
        typewell.to_csv(tmp_path / f"{donor}__typewell.csv", index=False)
    inventory = pd.DataFrame(
        {
            "well": ["q", "d1", "d2"],
            "typewell_group_id": ["g", "g", "g"],
        }
    )
    axis_graph = pd.DataFrame(
        {
            "well": ["q", "d1", "d2"],
            "typewell_group_id": ["g", "g", "g"],
            "axis_offset_ft": [0.0, 0.0, 0.0],
            "axis_connected": [True, True, True],
            "axis_group_valid": [True, True, True],
        }
    )
    ledger = train.TruthAccessLedger()
    bundle = train.generate_target_free_candidates(
        inventory,
        {0: ({"d1", "d2"}, {"q"})},
        profiles,
        axis_graph,
        tmp_path,
        config,
        ledger,
    )
    query = bundle.wells.iloc[0]
    assert query["well"] == "q"
    assert bool(query["supported"])
    assert query[train.PRIMARY] == pytest.approx(10.0)
    assert query[train.RANDOM] == pytest.approx(10.0)
    assert query[train.GROUP_MEDIAN] == pytest.approx(10.0)
    assert bundle.blocks["top1_local_shift_ft"].eq(10.0).all()
    assert ledger.query_truth_rows_before_freeze == 0
    assert ledger.query_truth_rows_after_freeze == 0
    assert ledger.donor_truth_rows_by_fold == {0: 2 * len(true_tvt)}


def test_safe_reader_does_not_expose_query_truth(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "MD": np.arange(8, dtype=float),
            "GR": np.arange(8, dtype=float) + 10.0,
            "TVT": np.arange(8, dtype=float) + 100.0,
            "TVT_input": [100.0, 101.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    frame.to_csv(tmp_path / "q__horizontal_well.csv", index=False)
    rows = pd.DataFrame({"row_idx": np.arange(2, 8)})
    profile = train.load_safe_suffix_profile("q", rows, tmp_path, config)
    assert profile.row_idx.tolist() == [2, 3, 4, 5, 6, 7]
    assert profile.gr_raw.tolist() == [12.0, 13.0, 14.0, 15.0, 16.0, 17.0]


def test_truth_reader_enforces_fold_role_and_freeze(
    train: ModuleType,
    tmp_path: Path,
) -> None:
    gr = np.linspace(10.0, 20.0, 8)
    profile = make_profile(train, "q", gr)
    pd.DataFrame({"TVT": np.arange(8, dtype=float) + 100.0}).to_csv(
        tmp_path / "q__horizontal_well.csv", index=False
    )
    pd.DataFrame({"TVT": np.arange(20, dtype=float), "GR": np.arange(20)}).to_csv(
        tmp_path / "q__typewell.csv", index=False
    )
    ledger = train.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="outer-train"):
        train.read_registration_inputs(
            "q",
            profile,
            tmp_path,
            fold=0,
            outer_train={"d"},
            outer_valid={"q"},
            ledger=ledger,
            query_after_freeze=False,
        )
    with pytest.raises(RuntimeError, match="frozen"):
        train.read_registration_inputs(
            "q",
            profile,
            tmp_path,
            fold=0,
            outer_train={"d"},
            outer_valid={"q"},
            ledger=ledger,
            query_after_freeze=True,
        )
    ledger.mark_frozen("a" * 64)
    truth, _ = train.read_registration_inputs(
        "q",
        profile,
        tmp_path,
        fold=0,
        outer_train={"d"},
        outer_valid={"q"},
        ledger=ledger,
        query_after_freeze=True,
    )
    assert len(truth) == 8
    assert ledger.query_truth_rows_before_freeze == 0
    assert ledger.query_truth_rows_after_freeze == 8


def test_local_gate_never_rescues_failed_global_gate(train: ModuleType) -> None:
    technical = {"passed": True}
    scientific = {
        "passed": False,
        "checks": {"oracle_gain": True, "oracle_fold_consistency": True},
    }
    local = {"passed": True}
    assert (
        train.decide_result(technical, scientific, local)
        == "registration_map_headroom_but_gr_similarity_selection_failed"
    )

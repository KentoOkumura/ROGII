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
EXP = "exp423_same_typewell_gr_dtw_truth_warp_transfer_readout"
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
    previous = os.environ.get("EXP423_IMPORT_ONLY")
    os.environ["EXP423_IMPORT_ONLY"] = "1"
    try:
        return load_module(SOURCE, "exp423_train_contract")
    finally:
        if previous is None:
            os.environ.pop("EXP423_IMPORT_ONLY", None)
        else:
            os.environ["EXP423_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def make_profile(train: ModuleType, well: str, anchor: float) -> object:
    progress = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    normalized = np.linspace(-1.0, 1.0, 256)
    return train.SuffixProfile(
        well=well,
        row_idx=np.asarray([2, 3, 4], dtype=np.int64),
        md=np.asarray([10.0, 11.0, 12.0], dtype=np.float64),
        progress=progress,
        anchor_tvt=anchor,
        gr_resampled=normalized.copy(),
        gr_normalized=normalized.copy(),
        support_mask=np.ones(256, dtype=bool),
        finite_fraction=1.0,
        robust_center=0.0,
        robust_scale=1.0,
    )


def test_frozen_contract_is_zero_model_and_run_is_approved(
    train: ModuleType,
    config: dict,
) -> None:
    contract = train.validate_scientific_contract(config)

    assert contract["primary_candidate"] == "analog_top5_median"
    assert contract["top_k"] == 5
    assert contract["gr_points"] == 256
    assert contract["dtw_band_points"] == 32
    assert contract["dtw_max_axis_run"] == 4
    assert contract["execution_counts"] == {
        "audit_variants": 1,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "pf_well_runs": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "reporting_folds": 5,
    }
    assert len(contract["scientific_contract_sha256"]) == 64
    observed = train.validate_scientific_contract(
        config, require_run_approval=True
    )
    assert observed["scientific_contract_sha256"] == contract[
        "scientific_contract_sha256"
    ]
    unapproved = copy.deepcopy(config)
    unapproved["execution"]["audit_run_approved"] = False
    with pytest.raises(RuntimeError, match="not approved"):
        train.validate_scientific_contract(
            unapproved, require_run_approval=True
        )


def test_parent_compatible_fold_assignment_is_deterministic(
    train: ModuleType,
) -> None:
    wells = [f"w{index:02d}" for index in range(20)]
    first = train.deterministic_well_folds(wells, n_folds=5, seed=42)
    second = train.deterministic_well_folds(list(reversed(wells)), n_folds=5, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first["well"].is_unique
    assert sorted(first["fold"].unique().tolist()) == [0, 1, 2, 3, 4]
    assert first.groupby("fold").size().tolist() == [4, 4, 4, 4, 4]


def test_gr_preprocessing_is_fixed_robust_and_no_extrapolation(
    train: ModuleType,
    config: dict,
) -> None:
    md = np.arange(8, dtype=np.float64)
    gr = np.asarray([np.nan, 1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0])
    observed = train.preprocess_suffix_gr(md, gr, config)

    assert len(observed["resampled"]) == 256
    assert np.isnan(observed["resampled"][0])
    assert observed["support_mask"][0] == np.bool_(False)
    assert observed["finite_fraction"] == pytest.approx(7 / 8)
    assert observed["scale"] > 0.0
    assert np.isfinite(
        observed["normalized"][observed["support_mask"]]
    ).all()


def test_constrained_dtw_identity_and_axis_run_contract(
    train: ModuleType,
) -> None:
    values = np.sin(np.linspace(0.0, 4.0, 64))
    observed = train.constrained_dtw(values, values, band=8, max_run=4)

    assert observed["normalized_cost"] == pytest.approx(0.0)
    np.testing.assert_array_equal(observed["query_path"], np.arange(64))
    np.testing.assert_array_equal(observed["donor_path"], np.arange(64))
    assert observed["max_vertical_run"] == 0
    assert observed["max_horizontal_run"] == 0


def test_truth_warp_transfer_reanchors_donor_delta(
    train: ModuleType,
) -> None:
    query = make_profile(train, "query", anchor=100.0)
    donor = train.DonorTruth(
        well="donor",
        progress=np.asarray([0.0, 0.5, 1.0]),
        tvt=np.asarray([200.0, 210.0, 230.0]),
    )
    diagonal = np.arange(256, dtype=np.int32)
    prediction = train.transfer_donor_truth_warp(
        query,
        donor,
        {"query_path": diagonal, "donor_path": diagonal},
    )

    np.testing.assert_allclose(prediction, [100.0, 110.0, 130.0])
    assert prediction[0] == pytest.approx(query.anchor_tvt)


def test_target_free_horizontal_reader_never_exposes_query_tvt(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    well = "query"
    frame = pd.DataFrame(
        {
            "MD": np.arange(6, dtype=float),
            "GR": [1.0, 2.0, 3.0, 5.0, 8.0, 13.0],
            "TVT": [100.0, 101.0, 102.0, 104.0, 107.0, 111.0],
            "TVT_input": [100.0, 101.0, np.nan, np.nan, np.nan, np.nan],
        }
    )
    frame.to_csv(tmp_path / f"{well}__horizontal_well.csv", index=False)
    rows = pd.DataFrame(
        {
            "row_idx": [2, 3, 4, 5],
            "last_known_tvt": [101.0] * 4,
        }
    )
    observed = train.load_safe_suffix_profile(well, rows, tmp_path, config)

    assert observed.anchor_tvt == pytest.approx(101.0)
    assert observed.row_idx.tolist() == [2, 3, 4, 5]
    assert len(observed.gr_normalized) == 256


def test_donor_truth_requires_strict_outer_train_and_ledger(
    train: ModuleType,
    tmp_path: Path,
) -> None:
    profile = make_profile(train, "donor", anchor=101.0)
    pd.DataFrame({"TVT": [100.0, 101.0, 102.0, 104.0, 107.0]}).to_csv(
        tmp_path / "donor__horizontal_well.csv", index=False
    )
    ledger = train.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="not strictly outer-train"):
        train.load_outer_train_donor_truth(
            "donor",
            profile,
            tmp_path,
            fold=0,
            outer_train_wells={"other"},
            outer_valid_wells={"donor"},
            ledger=ledger,
        )

    observed = train.load_outer_train_donor_truth(
        "donor",
        profile,
        tmp_path,
        fold=1,
        outer_train_wells={"donor"},
        outer_valid_wells={"query"},
        ledger=ledger,
    )
    assert observed.tvt.tolist() == [102.0, 104.0, 107.0]
    assert ledger.donor_truth_rows_by_fold == {1: 3}
    assert ledger.query_truth_rows_before_freeze == 0


def test_stable_random_control_and_late_truth_ledger(
    train: ModuleType,
) -> None:
    donors = ["d3", "d1", "d2"]
    first = train.stable_random_donor("query", donors)
    second = train.stable_random_donor("query", list(reversed(donors)))
    assert first == second
    assert first in donors

    ledger = train.TruthAccessLedger()
    with pytest.raises(RuntimeError, match="requires a frozen"):
        ledger.require_frozen()
    ledger.mark_frozen("a" * 64)
    ledger.record_query_truth_after_freeze(12)
    assert ledger.query_truth_rows_before_freeze == 0
    assert ledger.query_truth_rows_after_freeze == 12


def test_well_oracle_is_post_freeze_and_whole_well_only(
    train: ModuleType,
) -> None:
    readout = pd.DataFrame(
        {
            "well": ["q1", "q1", "q2", "q2"],
            "fold": [0, 0, 1, 1],
            "true_tvt": [0.0, 10.0, 0.0, 10.0],
            train.PARENT_REFERENCE: [5.0, 5.0, 5.0, 5.0],
        }
    )
    donor_predictions = np.full((4, 5), np.nan, dtype=np.float32)
    donor_predictions[:, 0] = [0.0, 20.0, 5.0, 5.0]
    donor_predictions[:, 1] = [1.0, 11.0, 0.0, 10.0]

    observed, donor_metrics = train.attach_well_oracle(
        readout, donor_predictions
    )

    assert observed["oracle_selected_rank"].tolist() == [2, 2, 2, 2]
    np.testing.assert_allclose(
        observed[train.ORACLE_CANDIDATE],
        donor_predictions[:, 1],
    )
    assert len(donor_metrics) == 4


def test_target_free_generation_uses_only_outer_train_donors(
    train: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    wells = ["query", "d1", "d2", "d3"]
    rows: list[dict[str, object]] = []
    for fold, well in enumerate(wells):
        for offset, row_idx in enumerate([2, 3, 4]):
            rows.append(
                {
                    "id": f"{well}_{row_idx}",
                    "well": well,
                    "row_idx": row_idx,
                    "fold": fold,
                    "md_since": float(offset + 1),
                    "eval_len": 3,
                    "last_known_tvt": 101.0,
                    "typewell_group_id": "group",
                    train.PARENT_REFERENCE: 90.0,
                    train.LIKPF_REFERENCE: 91.0,
                }
            )
    inventory = pd.DataFrame(rows).reset_index(drop=True)
    profiles = {well: make_profile(train, well, anchor=101.0) for well in wells}
    for donor in ["d1", "d2", "d3"]:
        pd.DataFrame({"TVT": [100.0, 101.0, 102.0, 104.0, 107.0]}).to_csv(
            tmp_path / f"{donor}__horizontal_well.csv", index=False
        )
    ledger = train.TruthAccessLedger()
    bundle = train.generate_target_free_candidates(
        inventory,
        {0: ({"d1", "d2", "d3"}, {"query"})},
        profiles,
        tmp_path,
        config,
        ledger,
    )
    query = bundle.target_free_rows.loc[
        bundle.target_free_rows["well"].eq("query")
    ]

    assert query["supported"].all()
    assert query["eligible_donor_count"].eq(3).all()
    assert query["used_donor_count"].eq(3).all()
    np.testing.assert_allclose(
        query[train.PRIMARY_CANDIDATE],
        [101.0, 103.0, 106.0],
    )
    assert set(query["top1_donor_well"]) == {"d1"}
    assert ledger.query_truth_rows_before_freeze == 0
    assert ledger.donor_truth_rows_by_fold == {0: 9}
    assert bundle.fold_audit["donor_query_intersection"].eq(0).all()


def test_execution_adopts_canonical_train_and_keeps_inference_placeholder(
    config: dict,
) -> None:
    source = SOURCE.read_text()
    canonical_train = (
        EXP_DIR / "exp423_same_typewell_gr_dtw_truth_warp_transfer_readout_train.ipynb"
    ).read_text()
    canonical_inference = (
        EXP_DIR
        / "exp423_same_typewell_gr_dtw_truth_warp_transfer_readout_inference.ipynb"
    ).read_text()

    assert config["experiment"]["status"] in {
        "independent_rerun_ready",
        "completed_closed",
    }
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["canonical_notebooks_are_placeholders"] is False
    assert config["implementation"]["canonical_train_notebook_adopted"] is True
    assert (
        config["implementation"]["canonical_inference_notebook_is_placeholder"]
        is True
    )
    assert config["execution"]["audit_run_approved"] is True
    assert config["execution"]["inference_approved"] is False
    assert config["execution"]["submission_approved"] is False
    assert "__file__" not in source
    assert "Run the approved Kaggle CPU readout" in canonical_train
    assert "placeholder" not in canonical_train
    assert "placeholder" in canonical_inference

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp370_triggered_reset_rejuvenation_pf"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp370_triggered_reset_rejuvenation_pf_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp370_triggered_reset_rejuvenation_pf_compact_selfcontained_inference.py"
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
    return load_module(TRAIN_SOURCE, "exp370_train_test")


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_completed_stage0_contract_is_fail_closed(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract["particles"] == 500
    assert contract["diagnostic_seed_count"] == 1
    assert contract["diagnostic_pf_seed_well_runs"] == 773
    assert contract["bad_event_horizon_rows"] == 128
    assert contract["atlas"]["query_window_rows"] == 256
    assert contract["atlas"]["top_k"] == 3
    assert contract["run_stage_0"] is False
    assert contract["run_stage_1"] is False
    assert len(contract["contract_sha256"]) == 64
    with pytest.raises(PermissionError, match="execution.run_stage_0 is false"):
        train.validate_scientific_contract(
            config,
            require_run_approval=True,
        )


def test_inference_is_fail_closed(config):
    inference = load_module(INFERENCE_SOURCE, "exp370_inference_test")
    status = inference.validate_disabled_inference(config)
    assert status["implementation_scope"] == "stage0_only"
    assert status["run_stage_1"] is False
    assert status["run_inference"] is False
    assert status["create_submission"] is False
    assert status["stage1_seed_well_runs"] == 98_944


def test_resolve_train_dir_prefers_competition_train_and_rejects_test(
    train,
    config,
    tmp_path,
    monkeypatch,
):
    kaggle_root = tmp_path / "kaggle_input"
    competition = (
        kaggle_root
        / "competitions"
        / "rogii-wellbore-geology-prediction"
    )
    train_dir = competition / "train"
    test_dir = competition / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    for well in ("well_a", "well_b"):
        (train_dir / f"{well}__horizontal_well.csv").write_text("MD\n1\n")
        (train_dir / f"{well}__typewell.csv").write_text("TVT\n1\n")
    (test_dir / "wrong__horizontal_well.csv").write_text("MD\n1\n")
    (test_dir / "wrong__typewell.csv").write_text("TVT\n1\n")
    local_config = deepcopy(config)
    local_config["data"]["train_dir"] = "does/not/exist"
    local_config["data"]["parent_control"]["expected_wells"] = 2
    monkeypatch.setattr(train, "KAGGLE_INPUT_ROOT", kaggle_root)
    monkeypatch.setattr(train, "PACKAGE_DIR", tmp_path / "package")
    monkeypatch.chdir(tmp_path)
    assert train.resolve_train_dir(local_config) == train_dir


def test_stable_seed_is_fold_well_family_and_seed_index_keyed(train):
    first = train.stable_seed(train.EXPERIMENT_NAME, 0, "well-a", "pf", 0)
    repeated = train.stable_seed(train.EXPERIMENT_NAME, 0, "well-a", "pf", 0)
    next_fold = train.stable_seed(train.EXPERIMENT_NAME, 1, "well-a", "pf", 0)
    next_seed = train.stable_seed(train.EXPERIMENT_NAME, 0, "well-a", "pf", 1)
    assert first == repeated
    assert first != next_fold
    assert first != next_seed
    assert 1 <= first <= 2_147_483_647


def test_diagnostic_pf_is_deterministic_and_records_pre_resample_ess(train):
    rows = 24
    md = np.arange(1.0, rows + 1.0)
    z = np.zeros(rows)
    observed = np.linspace(0.0, 12.0, rows)
    grid = np.linspace(0.0, 12.0, 121)
    args = (
        md,
        z,
        observed,
        grid,
        0.0,
        0.1,
        1.5,
        0.0,
        0.0,
        0.05,
        64,
        1234,
        0.998,
        0.002,
        0.005,
        0.10,
        0.001,
        0.5,
        4.5,
        100.0,
    )
    first_pred, first_ess = train.diagnostic_likelihood_pf(*args)
    second_pred, second_ess = train.diagnostic_likelihood_pf(*args)
    np.testing.assert_array_equal(first_pred, second_pred)
    np.testing.assert_array_equal(first_ess, second_ess)
    assert np.isfinite(first_pred).all()
    assert np.all((first_ess > 0.0) & (first_ess <= 1.0 + 1.0e-12))
    assert np.any(first_ess < 1.0)


def test_refractory_and_circular_control_preserve_event_count(train):
    candidate = np.zeros(20, dtype=bool)
    candidate[[1, 3, 6, 7, 13]] = True
    accepted = train.apply_refractory(candidate, refractory_rows=5)
    assert np.flatnonzero(accepted).tolist() == [1, 6, 13]
    score = np.where(accepted, np.arange(20, dtype=float) + 1.0, 0.0)
    circular = train.circular_shift_trigger_score(score, shift_rows=512)
    assert np.count_nonzero(circular) == np.count_nonzero(score)
    assert not np.array_equal(circular, score)


def test_query_patch_is_znorm_and_top3_enforces_ten_foot_separation(train):
    values = np.sin(np.arange(300, dtype=float) / 11.0)
    patch = train.resampled_znorm_patch(values, 150, 256, 32)
    assert patch is not None
    assert float(np.mean(patch)) == pytest.approx(0.0, abs=1.0e-6)
    assert float(np.sqrt(np.mean(patch**2))) == pytest.approx(1.0, abs=1.0e-6)
    selected = train.select_topk_separated(
        proposal_tvt=np.asarray([100.0, 104.0, 111.0, 121.0, 140.0]),
        score=np.asarray([0.99, 0.98, 0.97, 0.96, 0.95]),
        top_k=3,
        minimum_separation_ft=10.0,
    )
    assert selected.tolist() == [0, 2, 3]


def test_fold_safe_atlas_excludes_each_validation_fold(train, config, tmp_path):
    wells = [f"well-{index}" for index in range(5)]
    fold_by_well = {well: index for index, well in enumerate(wells)}
    rows = 320
    for index, well in enumerate(wells):
        coordinate = np.arange(rows, dtype=float)
        horizontal = pd.DataFrame(
            {
                "GR": 60.0 + 8.0 * np.sin((coordinate + index) / 13.0),
                "TVT": 1000.0 + 0.2 * coordinate,
            }
        )
        horizontal.to_csv(
            tmp_path / f"{well}__horizontal_well.csv",
            index=False,
        )
    ledger = train.TruthAccessLedger()
    atlas, report = train.build_fold_safe_atlases(
        wells,
        fold_by_well,
        tmp_path,
        config,
        ledger,
    )
    assert not atlas.empty
    assert set(atlas["fold"].unique()) == {0, 1, 2, 3, 4}
    assert all(
        row["validation_source_intersection_count"] == 0
        for row in report["folds"]
    )
    assert all(row["source_well_count"] == 4 for row in report["folds"])
    assert ledger.donor_fold_leakage_violations == 0
    assert ledger.outer_train_donor_truth_rows_before_freeze == rows * 5 * 4


def test_target_truth_ledger_rejects_prefreeze_and_allows_outer_train_donor(train):
    ledger = train.TruthAccessLedger()
    ledger.record_outer_train_donor_truth(
        donor_fold=0,
        atlas_fold=1,
        rows=100,
    )
    with pytest.raises(ValueError, match="forbidden target pre-freeze"):
        ledger.guard_target_prefreeze_columns(["MD", "TVT"], 12, "synthetic")
    assert ledger.outer_train_donor_truth_rows_before_freeze == 100
    assert ledger.target_truth_rows_before_freeze == 12
    with pytest.raises(RuntimeError, match="before SHA freeze"):
        ledger.freeze()


def test_saved_parent_loader_materializes_absolute_likpf(train, config, tmp_path):
    frame = pd.DataFrame(
        {
            "id": ["well-a_2", "well-a_3"],
            "well": ["well-a", "well-a"],
            "last_known_tvt": [100.0, 100.0],
            "likpf_mean_d": [2.5, 3.5],
        }
    )
    path = tmp_path / "saved.csv.gz"
    train.write_deterministic_csv_gzip(path, frame)
    synthetic = deepcopy(config)
    spec = synthetic["data"]["parent_control"]
    spec["filename"] = path.name
    spec["candidates"] = [str(tmp_path)]
    spec["expected_rows"] = 2
    spec["expected_wells"] = 1
    spec["expected_decompressed_sha256"] = train.sha256_decompressed_gzip(path)
    loaded, report = train.load_saved_parent_likpf(synthetic)
    np.testing.assert_allclose(loaded["saved_likpf_tvt"], [102.5, 103.5])
    assert report["rows"] == 2
    assert report["wells"] == 1


def test_prefreeze_builder_uses_saved_base_and_never_reads_target_truth(
    train, config, tmp_path
):
    synthetic = deepcopy(config)
    synthetic["validation"]["stage_0"]["particles"] = 32
    synthetic["validation"]["stage_0"]["trigger"][
        "gr_change_robust_z_quantile_from_known_prefix"
    ] = 0.5
    synthetic["validation"]["stage_0"]["trigger"]["maximum_ess_fraction"] = 1.0
    well = "synthetic"
    prefix_rows = 128
    suffix_rows = 260
    total_rows = prefix_rows + suffix_rows
    coordinate = np.arange(total_rows, dtype=float)
    tvt_input = np.full(total_rows, np.nan)
    tvt_input[:prefix_rows] = 100.0 + 0.1 * coordinate[:prefix_rows]
    gr = 60.0 + 8.0 * np.sin(coordinate / 13.0)
    gr[prefix_rows] += 100.0
    horizontal = pd.DataFrame(
        {
            "MD": coordinate,
            "Z": np.zeros(total_rows),
            "GR": gr,
            "TVT_input": tvt_input,
            "TVT": 100.0 + 0.1 * coordinate,
        }
    )
    typewell_tvt = np.linspace(80.0, 180.0, 501)
    typewell = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": 60.0 + 8.0 * np.sin((typewell_tvt - 100.0) / 1.3),
        }
    )
    horizontal.to_csv(tmp_path / f"{well}__horizontal_well.csv", index=False)
    typewell.to_csv(tmp_path / f"{well}__typewell.csv", index=False)
    suffix_index = np.arange(prefix_rows, total_rows)
    saved = pd.DataFrame(
        {
            "id": [f"{well}_{row}" for row in suffix_index],
            "well_id": well,
            "row_idx": suffix_index,
            "saved_likpf_tvt": np.full(suffix_rows, 1234.0),
        }
    )
    query = train.resampled_znorm_patch(gr, prefix_rows, 256, 32)
    assert query is not None
    patch_columns = train.atlas_patch_columns(32)
    atlas_rows = []
    for bin_key, proposal_tvt in ((60, 121.0), (70, 141.0), (80, 161.0)):
        row = {
            "fold": 0,
            "tvt_bin": bin_key,
            "proposal_tvt": proposal_tvt,
            "patch_count": 10,
            "source_well_count": 3,
            "source_wells_sha256": "a" * 64,
        }
        row.update(
            {
                column: float(query[index])
                for index, column in enumerate(patch_columns)
            }
        )
        atlas_rows.append(row)
    ledger = train.TruthAccessLedger()
    trigger, proposal, manifest = train.build_prefreeze_rows_for_well(
        well,
        0,
        saved,
        tmp_path,
        pd.DataFrame(atlas_rows),
        synthetic,
        ledger,
    )
    assert trigger["accepted_trigger"].sum() >= 1
    assert not proposal.empty
    assert trigger["saved_likpf_tvt"].eq(1234.0).all()
    assert "TVT" not in trigger.columns
    assert ledger.target_truth_rows_before_freeze == 0
    assert manifest["pf_seed_well_runs"] == 1


def test_trigger_and_event_metrics_use_saved_base_coverage(train):
    trigger = pd.DataFrame(
        {
            "eligible": [True, True, True, True],
            "accepted_trigger": [False, False, True, True],
            "bad_event": [False, False, True, True],
            "trigger_score": [0.0, 0.0, 0.9, 1.0],
            "circular_trigger_score": [0.9, 1.0, 0.0, 0.0],
        }
    )
    trigger_metrics = train.trigger_metrics(trigger)
    assert trigger_metrics["trigger_bad_event_auc"] == pytest.approx(1.0)
    assert trigger_metrics["circular_bad_event_auc"] == pytest.approx(0.0)
    events = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "atlas_top3_within10": [True, True],
            "base_likpf_within10": [False, True],
            "best_atlas_abs_error": [1.0, 2.0],
            "base_abs_error": [20.0, 3.0],
            "top1_zncc": [0.9, 0.8],
        }
    )
    event_metrics = train.event_metrics(events)
    assert event_metrics["atlas_top3_within10_coverage"] == pytest.approx(1.0)
    assert event_metrics["base_likpf_within10_coverage"] == pytest.approx(0.5)
    assert event_metrics["coverage_gain_over_base_likpf"] == pytest.approx(0.5)


def test_source_is_not_thin_and_is_not_file_dependent():
    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert train_text.count("# %% [markdown]") >= 12
    assert len(train_text.splitlines()) >= 1_500
    assert "__file__" not in train_text
    assert "__file__" not in inference_text
    assert "from settings import" not in train_text
    assert "sample_submission" not in inference_text

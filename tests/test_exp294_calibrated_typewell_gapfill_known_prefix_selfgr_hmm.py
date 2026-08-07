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
EXP_DIR = ROOT / "experiments" / "exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm"
MODULE_PATH = EXP_DIR / (
    "exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm_compact_selfcontained_train.py"
)
SPEC = importlib.util.spec_from_file_location("exp294_stage0_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EXP294 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXP294
SPEC.loader.exec_module(EXP294)


def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def synthetic_typewell() -> pd.DataFrame:
    tvt = np.linspace(80.0, 280.0, 801)
    gr = 70.0 + 18.0 * np.sin(tvt / 8.0) + 7.0 * np.cos(tvt / 17.0)
    return pd.DataFrame({"TVT": tvt, "GR": gr})


def synthetic_horizontal(rows: int = 180, known_rows: int = 160) -> pd.DataFrame:
    typewell = synthetic_typewell()
    tvt_input = 100.0 + np.arange(rows, dtype=float)
    reference = np.interp(tvt_input, typewell["TVT"], typewell["GR"])
    gr = 1.35 * reference + 4.25
    tvt_input[known_rows:] = np.nan
    return pd.DataFrame({"TVT_input": tvt_input, "GR": gr})


def make_scan(well: str, fold: int, runs: tuple[int, ...]) -> object:
    return EXP294.WellScan(
        well=well,
        reporting_fold=fold,
        rows=180,
        known_rows=160,
        finite_known_gr_rows=150,
        natural_missing_run_lengths=runs,
        horizontal_name=f"{well}__horizontal_well.csv",
        typewell_name=f"{well}__typewell.csv",
        horizontal_sha256="h",
        typewell_sha256="t",
    )


def test_config_is_closed_after_stage0_guard_fail() -> None:
    config = load_config()
    EXP294.validate_implementation_contract(config)
    assert config["experiment"]["route"] == "ensemble"
    assert config["stage0"]["audit_variants"] == 1
    assert config["model"]["lightgbm_configs"] == 0
    assert config["model"]["trained_folds"] == 0
    assert config["model"]["boosters"] == 0
    assert config["experiment"]["status"] == "stage0_completed_fail_branch_closed"
    assert config["stage0"]["implementation_status"] == "completed_kaggle_v1_guard_fail"
    assert config["stage1"]["implementation_status"] == "closed_stage0_guard_fail"
    assert config["execution"]["run_stage0"] is False
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["stage0_approved_at"] == "2026-07-19"
    assert config["execution"]["stage0_result"] == "FAIL"
    assert config["execution"]["stage1_authorized"] is False
    assert config["execution"]["run_stage1"] is False
    assert config["execution"]["regenerate_parent_control"] is False
    assert config["execution"]["run_inference"] is False
    assert config["execution"]["write_submission"] is False

    unsafe = deepcopy(config)
    unsafe["gapfill"]["preserve_raw_missing_mask"] = False
    with pytest.raises(AssertionError, match="raw_mask_preserved"):
        EXP294.validate_implementation_contract(unsafe)


def test_stage0_runner_requires_explicit_execution_switch(tmp_path: Path) -> None:
    config = load_config()
    with pytest.raises(RuntimeError, match="obtain user approval"):
        EXP294.run_stage0(config, train_dir=tmp_path, output_dir=tmp_path)


def test_stable_reporting_fold_uses_utf8_sha256_first8_big_endian() -> None:
    assert EXP294.stable_reporting_fold("well-a") == 0
    assert EXP294.stable_reporting_fold("alpha") == 4
    assert EXP294.stable_reporting_fold("well-a") == EXP294.stable_reporting_fold("well-a")
    with pytest.raises(ValueError, match="positive"):
        EXP294.stable_reporting_fold("well-a", 0)


def test_external_run_lengths_exclude_reporting_fold_and_clip() -> None:
    scans = [make_scan("fold0", 0, (64,)), make_scan("fold1", 1, (1,))]
    lengths = EXP294.fold_external_run_lengths(
        scans,
        n_folds=2,
        quantiles=[0.25, 0.50, 0.90],
        clip_rows=[1, 64],
        fallback_rows=[1, 4, 16],
    )
    assert lengths[0] == {"q25": 1, "q50": 1, "q90": 1}
    assert lengths[1] == {"q25": 64, "q50": 64, "q90": 64}
    assert EXP294.round_half_up(2.5) == 3


def test_typewell_duplicate_median_and_no_extrapolation() -> None:
    frame = pd.DataFrame({"TVT": [2.0, 1.0, 1.0, np.nan, 3.0], "GR": [20.0, 8.0, 12.0, 99.0, 30.0]})
    tvt, gr = EXP294.prepare_typewell_curve(frame)
    np.testing.assert_allclose(tvt, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(gr, [10.0, 20.0, 30.0])
    sampled = EXP294.interpolate_no_extrapolation(np.asarray([0.0, 1.5, 4.0]), tvt, gr)
    assert np.isnan(sampled[0]) and np.isnan(sampled[2])
    assert sampled[1] == pytest.approx(15.0)


def test_huber_affine_recovers_map_with_outlier_and_rejects_low_iqr() -> None:
    x = np.linspace(0.0, 100.0, 200)
    y = 1.7 * x + 4.5
    y[75] += 500.0
    fit = EXP294.fit_huber_affine(
        x,
        y,
        minimum_pairs=32,
        minimum_iqr=5.0,
        huber_k=1.345,
        max_iterations=20,
        relative_tolerance=1.0e-8,
        scale_floor=1.0,
    )
    assert fit.valid
    assert fit.slope == pytest.approx(1.7, abs=0.01)
    assert fit.intercept == pytest.approx(4.5, abs=0.2)
    assert fit.converged

    invalid = EXP294.fit_huber_affine(
        np.ones(40),
        np.arange(40, dtype=float),
        minimum_pairs=32,
        minimum_iqr=5.0,
        huber_k=1.345,
        max_iterations=20,
        relative_tolerance=1.0e-8,
        scale_floor=1.0,
    )
    assert not invalid.valid
    assert invalid.fallback_reason == "typewell_gr_iqr_below_minimum"


def test_pseudo_manifest_is_truth_free_stable_and_nonoverlapping(tmp_path: Path) -> None:
    config = load_config()
    well = "well-a"
    horizontal = synthetic_horizontal()
    horizontal.assign(TVT=np.arange(len(horizontal), dtype=float)).to_csv(
        tmp_path / f"{well}__horizontal_well.csv", index=False
    )
    synthetic_typewell().to_csv(tmp_path / f"{well}__typewell.csv", index=False)
    fold = EXP294.stable_reporting_fold(well)
    scan = make_scan(well, fold, ())
    fold_lengths = {index: {"q25": 4, "q50": 8, "q90": 16} for index in range(5)}
    first, first_selection = EXP294.build_pseudo_missing_manifest(
        tmp_path, [scan], fold_lengths, config
    )
    second, second_selection = EXP294.build_pseudo_missing_manifest(
        tmp_path, [scan], fold_lengths, config
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_selection, second_selection)
    assert not EXP294.FORBIDDEN_STAGE0_COLUMNS.intersection(first.columns)
    assert not first.duplicated(["well", "row_position"]).any()
    assert set(first_selection["status"]) == {"selected"}
    assert sorted(first.groupby("block_id").size()) == [4, 8, 16]

    loaded = EXP294.safe_horizontal(tmp_path / f"{well}__horizontal_well.csv")
    assert loaded.columns.tolist() == ["TVT_input", "GR"]


def test_hybrid_reconstruction_preserves_observed_and_excludes_pseudo_fit_rows() -> None:
    config = load_config()
    horizontal = synthetic_horizontal()
    horizontal.loc[25:27, "GR"] = np.nan
    pseudo_rows = np.arange(80, 88, dtype=np.int64)
    predictions, audit = EXP294.reconstruct_well(
        well="well-a",
        horizontal=horizontal,
        typewell=synthetic_typewell(),
        pseudo_rows=pseudo_rows,
        config=config,
    )
    assert audit["valid"] is True
    assert audit["observed_known_gr_exact_parity"] is True
    assert audit["raw_missing_mask_exact_parity"] is True
    assert audit["pseudo_mask_fit_overlap_rows"] == 0
    assert audit["target_side_typewell_fill_count"] == 0
    assert audit["gapfilled_natural_rows"] == 3
    assert audit["gapfilled_pseudo_rows"] == len(pseudo_rows)
    assert audit["pseudo_prediction_finite_coverage"] == 1.0
    np.testing.assert_allclose(predictions["variant_gr"], predictions["true_gr"], atol=1.0e-8)
    assert EXP294.rmse(predictions["true_gr"], predictions["variant_gr"]) < EXP294.rmse(
        predictions["true_gr"], predictions["control_gr"]
    )


def test_metrics_and_hard_gates_have_correct_direction() -> None:
    config = load_config()
    rows: list[dict] = []
    truth = np.asarray([-1.0, 0.0, 1.0, 0.0])
    for fold in range(5):
        well = f"well-{fold}"
        for row_position, true_gr in enumerate(truth):
            rows.append(
                {
                    "well": well,
                    "reporting_fold": fold,
                    "quantile_label": "q25",
                    "block_id": f"block-{fold}",
                    "row_position": row_position,
                    "true_gr": true_gr,
                    "control_gr": 0.0,
                    "variant_gr": true_gr,
                }
            )
    predictions = pd.DataFrame(rows)
    blocks = EXP294.build_block_metrics(predictions)
    by_well = EXP294.aggregate_metrics(predictions, blocks, group_column="well")
    by_fold = EXP294.aggregate_metrics(predictions, blocks, group_column="reporting_fold")
    audits = pd.DataFrame(
        {
            "observed_known_gr_exact_parity": [True] * 5,
            "raw_missing_mask_exact_parity": [True] * 5,
            "pseudo_mask_fit_overlap_rows": [0] * 5,
            "target_side_typewell_fill_count": [0] * 5,
        }
    )
    pooled, gates = EXP294.evaluate_hard_gates(
        predictions, blocks, by_well, by_fold, audits, config
    )
    assert pooled["variant_rmse"] == 0.0
    assert pooled["rmse_improving_reporting_folds"] == 5
    assert pooled["zncc_positive_reporting_folds"] == 5
    assert gates["passed"] is True


def test_stage0_synthetic_end_to_end_freezes_manifest_before_truth_outputs(tmp_path: Path) -> None:
    config = load_config()
    config["execution"]["run_stage0"] = True
    wells_by_fold: dict[int, str] = {}
    for index in range(10_000):
        well = f"synthetic-{index:04d}"
        wells_by_fold.setdefault(EXP294.stable_reporting_fold(well), well)
        if len(wells_by_fold) == 5:
            break
    assert set(wells_by_fold) == set(range(5))

    input_dir = tmp_path / "train"
    output_dir = tmp_path / "artifacts"
    input_dir.mkdir()
    typewell = synthetic_typewell()
    for well in wells_by_fold.values():
        horizontal = synthetic_horizontal()
        horizontal.assign(TVT=200.0 + np.arange(len(horizontal))).to_csv(
            input_dir / f"{well}__horizontal_well.csv", index=False
        )
        typewell.to_csv(input_dir / f"{well}__typewell.csv", index=False)

    summary = EXP294.run_stage0(
        config,
        train_dir=input_dir,
        output_dir=output_dir,
    )
    assert summary["input_wells"] == 5
    assert summary["leakage_contract"]["suffix_tvt_read"] is False
    assert summary["leakage_contract"]["pseudo_mask_frozen_before_truth_join"] is True
    assert summary["leakage_contract"]["target_side_typewell_fill_count"] == 0
    assert summary["stage1_authorized"] is False

    frozen = pd.read_csv(output_dir / "stage0_pseudo_missing_manifest.csv.gz")
    heldout = pd.read_csv(output_dir / "stage0_heldout_predictions.csv.gz")
    assert "true_gr" not in frozen.columns
    assert "true_gr" in heldout.columns
    assert not EXP294.FORBIDDEN_STAGE0_COLUMNS.intersection(frozen.columns)
    assert (
        EXP294.sha256_gzip_decompressed(output_dir / "stage0_pseudo_missing_manifest.csv.gz")
        == summary["leakage_contract"]["pseudo_mask_decompressed_content_sha256"]
    )
    assert (output_dir / "artifact_manifest.json").exists()


def test_compact_notebook_is_self_contained_and_stage1_is_not_implemented() -> None:
    source = MODULE_PATH.read_text()
    assert "from settings import" not in source
    assert "from exact_hmm_smoother import" not in source
    assert "Path(__file__)" not in source
    assert "def run_stage0(" in source
    assert "def run_stage1(" not in source
    assert "submission.csv" not in source

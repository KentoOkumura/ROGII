from __future__ import annotations

import importlib.util
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp285_exp226_prefix_masked_offset_predictability_readout"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp285_exp226_prefix_masked_offset_predictability_readout_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXP_DIR
    / "exp285_exp226_prefix_masked_offset_predictability_readout_compact_selfcontained_inference.py"
)


def load_module(path: Path = TRAIN_SOURCE, name: str = "exp285_train"):
    previous = os.environ.get("EXP285_IMPORT_ONLY")
    os.environ["EXP285_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP285_IMPORT_ONLY", None)
        else:
            os.environ["EXP285_IMPORT_ONLY"] = previous


def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def synthetic_frame(rows: int = 1800, known_rows: int = 1200) -> pd.DataFrame:
    index = np.arange(rows)
    tvt_input = 1000.0 + 0.2 * index
    tvt_input[known_rows:] = np.nan
    return pd.DataFrame(
        {
            "id": [f"id-{value}" for value in index],
            "X": 10.0 + 0.7 * index,
            "Y": 20.0 + 0.3 * index,
            "Z": -1000.0 - 0.1 * index,
            "MD": index.astype(float),
            "TVT_input": tvt_input,
        }
    )


def synthetic_readout_frames(wells: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = np.arange(wells)
    signal = np.linspace(-8.0, 8.0, wells) + 0.05 * np.sin(index)
    prefix = pd.DataFrame(
        {
            "well_id": [f"well-{value:03d}" for value in index],
            "fold": index % 5,
            "pseudo_offset_median": signal,
            "pseudo_offset_slope": signal / 100.0,
            "pseudo_block_drift_rate": signal / 80.0,
        }
    )
    official = pd.DataFrame(
        {
            "well_id": prefix["well_id"],
            "fold": prefix["fold"],
            "official_offset_median": signal * 1.1,
            "official_offset_slope": signal / 90.0,
            "official_block_drift_rate": signal / 70.0,
            "official_h256_offset_median": signal,
            "official_h512_offset_median": signal,
            "official_h640_offset_median": signal,
            "official_near_0_250_offset_median": signal,
            "official_long_tail_1000_plus_offset_median": signal,
        }
    )
    hidden = pd.DataFrame(
        {
            "well_id": prefix["well_id"],
            "verification_like_spatial_role": ["valid"] * wells,
            "verification_like_typewell_purged_role": ["valid"] * wells,
        }
    )
    return prefix, official, hidden


def test_config_and_zero_booster_contract() -> None:
    module = load_module(name="exp285_contract")
    config = load_config()
    module.validate_scientific_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["pseudo_mask"]["masked_rows"] == 640
    assert config["pseudo_mask"]["minimum_visible_rows_before_cut"] == 512
    assert config["prefix_summary"]["block_count"] == 5
    assert config["prefix_summary"]["block_rows"] == 128
    assert config["negative_control"]["permutations"] == 256
    assert config["execution"]["active_readout_variants"] == 1
    assert config["execution"]["lightgbm_config_count"] == 0
    assert config["execution"]["trained_fold_count"] == 0
    assert config["execution"]["total_boosters"] == 0
    assert config["execution"]["hmm_well_runs"] == 0
    assert config["execution"]["pf_well_runs"] == 0
    assert config["execution"]["control_or_parent_retraining"] is False
    assert config["execution"]["implementation_approved"] is True
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["kaggle_push_approval_source"] == "user_message_run_2026_07_19"


def test_mask_and_geometry_target_extend_through_well_end() -> None:
    module = load_module(name="exp285_mask")
    config = load_config()
    frame = synthetic_frame()
    masked, heldout, manifest = module.build_pseudo_mask("well-a", frame, config)
    assert manifest["official_last_known_row"] == 1199
    assert manifest["cut_row"] == 559
    assert manifest["visible_rows"] == 560
    assert manifest["masked_rows"] == 640
    assert manifest["full_replay_rows"] == 1240
    assert len(heldout) == 640
    assert masked.loc[:559, "TVT_input"].notna().all()
    assert masked.loc[560:, "TVT_input"].isna().all()
    target = module.build_target_geometry_well(
        "well-a", masked, cut=int(manifest["cut_row"]), params=module.params_from_config(config)
    )
    assert target.n == len(frame) - 560
    assert target.n > 640
    with pytest.raises(ValueError, match="forbidden columns"):
        module.validate_target_safe_frame(masked.assign(TVT=np.arange(len(masked))))


def test_target_safe_loader_synthesizes_row_id(tmp_path: Path) -> None:
    module = load_module(name="exp285_target_loader")
    path = tmp_path / "well-a__horizontal_well.csv"
    synthetic_frame(rows=8, known_rows=6).drop(columns="id").to_csv(path, index=False)
    loaded = module.load_target_safe_horizontal(path)
    assert tuple(loaded.columns) == module.TARGET_SAFE_COLUMNS
    assert loaded["id"].tolist() == [f"well-a:{row_idx}" for row_idx in range(8)]
    module.validate_target_safe_frame(loaded)


def test_five_block_summary_has_fixed_slope_and_drift() -> None:
    module = load_module(name="exp285_summary")
    md = np.arange(640, dtype=float)
    residual = 2.0 + 0.01 * md
    blocks = [np.arange(start, start + 128) for start in range(0, 640, 128)]
    summary = module.summarize_residual_blocks(md, residual, blocks)
    assert summary["offset_median"] == pytest.approx(5.195)
    assert summary["offset_slope"] == pytest.approx(0.01)
    assert summary["block_drift_rate"] == pytest.approx(0.01)
    assert summary["residual_finite_fraction"] == 1.0


def test_prefix_truth_attachment_requires_frozen_pseudo_path() -> None:
    module = load_module(name="exp285_freeze")
    config = load_config()
    offsets = np.arange(1, 641)
    pseudo = pd.DataFrame(
        {
            "well_id": ["well-a"] * 640,
            "fold": [0] * 640,
            "row_idx": 559 + offsets,
            "pseudo_cut_row": [559] * 640,
            "official_last_known_row": [1199] * 640,
            "masked_offset": offsets,
            "MD": offsets.astype(float),
            "pseudo_tvt_geop": 1000.0 + 0.1 * offsets,
            "donor_distance": [100.0] * 640,
            "truth_attached": [False] * 640,
        }
    )
    heldout = pd.DataFrame(
        {
            "well_id": ["well-a"] * 640,
            "row_idx": 559 + offsets,
            "masked_offset": offsets,
            "id": [f"id-{value}" for value in offsets],
            "MD": offsets.astype(float),
            "masked_tvt_input": 1002.0 + 0.11 * offsets,
        }
    )
    mask_manifest = pd.DataFrame(
        {
            "well_id": ["well-a"],
            "post_cut_tvt_input_finite_rows_after_mask": [0],
        }
    )
    frozen = module.assert_target_free_pseudo_paths(pseudo, mask_manifest, config)
    with pytest.raises(ValueError, match="frozen content SHA"):
        module.build_prefix_offset_summary(pseudo, heldout, frozen_hashes={}, config=config)
    summary = module.build_prefix_offset_summary(
        pseudo, heldout, frozen_hashes=frozen, config=config
    )
    assert summary.loc[0, "pseudo_offset_median"] == pytest.approx(5.205)
    assert summary.loc[0, "pseudo_offset_slope"] == pytest.approx(0.01)
    frozen = module.freeze_prefix_summary(summary, frozen)
    assert set(frozen) == {"mask_manifest", "pseudo_geop", "prefix_summary"}


def test_official_truth_reader_is_post_freeze_only(tmp_path: Path) -> None:
    module = load_module(name="exp285_official")
    config = load_config()
    row_idx = np.arange(11, 111)
    oof = pd.DataFrame(
        {
            "well_id": ["well-a"] * len(row_idx),
            "row_idx": row_idx,
            "suffix_offset": np.arange(len(row_idx)),
            "fold": [0] * len(row_idx),
            "tvt_geop": 1000.0 + 0.1 * row_idx,
            "tvt_true": 1002.0 + 0.11 * row_idx,
        }
    )
    oof_path = tmp_path / "oof.csv.gz"
    oof.to_csv(oof_path, index=False, compression="gzip")
    prefix = pd.DataFrame(
        {"well_id": ["well-a"], "fold": [0], "official_last_known_row": [10]}
    )
    with pytest.raises(ValueError, match="frozen content SHA"):
        module.load_official_target_rows(oof_path, prefix, frozen_hashes={})
    frozen = {"pseudo_geop": "a", "prefix_summary": "b"}
    official = module.load_official_target_rows(oof_path, prefix, frozen_hashes=frozen)
    raw_dir = tmp_path / "train"
    raw_dir.mkdir()
    pd.DataFrame({"MD": np.arange(111, dtype=float)}).to_csv(
        raw_dir / "well-a__horizontal_well.csv", index=False
    )
    summary = module.build_official_target_summary(official, raw_dir, prefix, config)
    assert summary.loc[0, "official_offset_median"] == pytest.approx(2.605)
    assert summary.loc[0, "official_offset_slope"] == pytest.approx(0.01)


def test_readout_and_permutation_are_deterministic() -> None:
    module = load_module(name="exp285_metrics")
    config = load_config()
    prefix, official, hidden = synthetic_readout_frames()
    by_well, metrics = module.build_predictability_readout(prefix, official, hidden)
    primary = module.metric_lookup(metrics, "overall", "offset_median")
    assert primary["spearman"] == pytest.approx(1.0)
    assert primary["sign_balanced_accuracy"] == pytest.approx(1.0)
    first_frame, first_summary = module.build_permutation_metrics(by_well, config)
    second_frame, second_summary = module.build_permutation_metrics(by_well, config)
    pd.testing.assert_frame_equal(first_frame, second_frame)
    assert first_summary == second_summary
    assert first_summary["pvalue"] <= 0.01


def test_scientific_guard_and_inference_fail_closed() -> None:
    module = load_module(name="exp285_guard")
    config = deepcopy(load_config())
    config["validation"]["guards"]["required_eligible_wells"] = 50
    prefix, official, hidden = synthetic_readout_frames()
    by_well, metrics = module.build_predictability_readout(prefix, official, hidden)
    _, permutation = module.build_permutation_metrics(by_well, config)
    mask_manifest = pd.DataFrame(
        {
            "well_id": prefix["well_id"],
            "target_well_in_donor_field": [False] * len(prefix),
            "post_cut_tvt_input_finite_rows_after_mask": [0] * len(prefix),
        }
    )
    pseudo_paths = pd.DataFrame(
        {
            "MD": np.arange(len(prefix), dtype=float),
            "pseudo_tvt_geop": np.arange(len(prefix), dtype=float),
            "donor_distance": np.ones(len(prefix)),
        }
    )
    guard = module.evaluate_scientific_guard(
        mask_manifest, pseudo_paths, prefix, metrics, permutation, config
    )
    assert guard["passed"] is True
    failed_metrics = metrics.copy()
    failed_metrics.loc[
        (failed_metrics["scope"] == "overall")
        & (failed_metrics["family"] == "offset_median"),
        "spearman",
    ] = 0.0
    failed = module.evaluate_scientific_guard(
        mask_manifest, pseudo_paths, prefix, failed_metrics, permutation, config
    )
    assert failed["passed"] is False

    inference = load_module(INFERENCE_SOURCE, "exp285_inference")
    inference.assert_inference_disabled(load_config())
    with pytest.raises(RuntimeError, match="train-side prefix-masked offset predictability"):
        inference.fail_closed()

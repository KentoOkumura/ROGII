from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp311_typewell_group_prefix_suffix_gr_calibration_readout"
TRAIN_SOURCE = (
    EXP_DIR
    / "exp311_typewell_group_prefix_suffix_gr_calibration_readout_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp311_typewell_group_prefix_suffix_gr_calibration_readout_compact_selfcontained_inference.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_contract_fixes_zero_booster_truth_boundary_and_gr_score_unit() -> None:
    module = load_module(TRAIN_SOURCE, "exp311_contract")
    config = load_config()
    module.validate_scientific_contract(config)
    config["execution_contract"]["boosters"] = 1
    with pytest.raises(ValueError, match="boosters"):
        module.validate_scientific_contract(config)


def test_sha_fold_manifest_is_deterministic_balanced_and_well_disjoint() -> None:
    module = load_module(TRAIN_SOURCE, "exp311_folds")
    wells = [f"well_{index:03d}" for index in range(23)]
    first = module.build_fold_manifest(wells, 5, 42)
    second = module.build_fold_manifest(list(reversed(wells)), 5, 42)
    pd.testing.assert_frame_equal(first, second)
    assert first["well_id"].nunique() == len(wells)
    assert first.groupby("fold").size().max() - first.groupby("fold").size().min() <= 1
    assert first["sha256_order_key"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_typewell_median_duplicate_reduction_and_no_extrapolation() -> None:
    module = load_module(TRAIN_SOURCE, "exp311_interpolation")
    typewell = pd.DataFrame(
        {
            "TVT": [0.0, 1.0, 1.0, 2.0],
            "GR": [10.0, 20.0, 24.0, 30.0],
        }
    )
    tvt, gr = module.collapse_typewell(typewell)
    np.testing.assert_allclose(tvt, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(gr, [10.0, 22.0, 30.0])
    result = module.interpolate_no_extrapolation(np.array([-1.0, 0.5, 2.0, 3.0]), tvt, gr)
    assert np.isnan(result[0]) and np.isnan(result[-1])
    np.testing.assert_allclose(result[1:3], [16.0, 30.0])


def test_huber_affine_is_deterministic_robust_and_identity_shrunk() -> None:
    module = load_module(TRAIN_SOURCE, "exp311_huber")
    config = load_config()["calibration"]
    x = np.linspace(0.0, 100.0, 300)
    y = 1.8 * x + 7.0
    y[150] += 5000.0
    first = module.fit_huber_affine_with_identity_shrinkage(x, y, config)
    second = module.fit_huber_affine_with_identity_shrinkage(x, y, config)
    assert first == second
    assert first["fit_available"]
    assert first["raw_slope"] == pytest.approx(1.8, abs=1.0e-3)
    assert first["raw_intercept"] == pytest.approx(7.0, abs=0.1)
    assert 1.0 < first["slope"] < first["raw_slope"]
    assert 0.0 < first["intercept"] < first["raw_intercept"]
    assert first["shrinkage_alpha"] == pytest.approx(0.6)


def test_outer_valid_truth_requires_complete_freeze_hash(tmp_path: Path) -> None:
    module = load_module(TRAIN_SOURCE, "exp311_truth_freeze")
    horizontal_path = tmp_path / "well_a__horizontal_well.csv"
    pd.DataFrame({"TVT": [0.0, 1.0, 2.0]}).to_csv(horizontal_path, index=False)
    target_free = module.TargetFreeWell(
        well_id="well_a",
        horizontal_path=horizontal_path,
        typewell_path=tmp_path / "well_a__typewell.csv",
        horizontal_gr=np.array([10.0, 20.0, 30.0]),
        tvt_input=np.array([0.0, np.nan, np.nan]),
        typewell_tvt=np.array([0.0, 1.0, 2.0]),
        typewell_gr=np.array([10.0, 20.0, 30.0]),
        prefix_row_idx=np.array([0], dtype=np.int32),
        prefix_typewell_gr=np.array([10.0]),
        prefix_horizontal_gr=np.array([10.0]),
        suffix_row_idx=np.array([1, 2], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="complete frozen"):
        module.attach_suffix_truth_after_freeze(target_free, freeze_sha256="short")
    pairs = module.attach_suffix_truth_after_freeze(target_free, freeze_sha256="a" * 64)
    assert pairs["row_idx"].tolist() == [1, 2]
    np.testing.assert_allclose(pairs["typewell_gr"], [20.0, 30.0])


def test_group_prior_is_equal_well_median_and_controls_are_deterministic() -> None:
    module = load_module(TRAIN_SOURCE, "exp311_groups")
    config = load_config()
    stats = pd.DataFrame(
        {
            "well_id": ["a", "b", "c"],
            "fit_available": [True, True, True],
            "support_rows": [32, 3200, 64],
            "slope": [1.0, 3.0, 2.0],
            "intercept": [0.0, 20.0, 10.0],
            "bias_at_gr50": [0.0, 120.0, 60.0],
            "residual_sigma_mad": [1.0, 9.0, 5.0],
            "fit_rmse": [2.0, 10.0, 6.0],
        }
    )
    lookup = {"a": "g", "b": "g", "c": "g"}
    priors = module.aggregate_group_priors(
        stats,
        lookup,
        fold=0,
        group_scheme="native_overlap_1",
        surface="same_typewell_heldout_well",
        control="real",
        config=config,
    )
    row = priors.iloc[0]
    assert row["slope"] == 2.0
    assert row["residual_sigma_mad"] == 5.0
    assert row["source_wells"] == 3
    assert row["support_rows"] == 3296
    first = module.shuffled_group_lookup(list(lookup), lookup, 0)
    second = module.shuffled_group_lookup(list(reversed(lookup)), lookup, 0)
    assert first == second
    assert sorted(first.values()) == sorted(lookup.values())


def test_inference_contract_is_fail_closed() -> None:
    module = load_module(INFERENCE_SOURCE, "exp311_inference")
    config = load_config()
    contract = module.validate_disabled_inference(config)
    assert not contract["inference_enabled"]
    assert not contract["create_submission"]
    config["inference"]["enabled"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        module.validate_disabled_inference(config)

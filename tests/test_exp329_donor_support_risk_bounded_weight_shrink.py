from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp329_donor_support_risk_bounded_weight_shrink"
TRAIN_PATH = (
    EXP_DIR / "exp329_donor_support_risk_bounded_weight_shrink_compact_selfcontained_train.py"
)


def load_module():
    os.environ["EXP329_IMPORT_ONLY"] = "1"
    name = "exp329_train"
    spec = importlib.util.spec_from_file_location(name, TRAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train = load_module()


@pytest.fixture
def config() -> dict:
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def synthetic_support() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for segment in range(4):
            value = float(10 * fold + segment)
            rows.append(
                {
                    "well_id": f"well_{fold}",
                    "readout_fold": fold,
                    "exp226_source_fold": (fold + 1) % 5,
                    "segment_id": segment,
                    "donor_count": 50,
                    "donor_distance": value + 1.0,
                    "effective_sample_size": 50.0 - value / 10.0,
                    "local_linear_log1p_condition": value + 2.0,
                    "raw_donor_weighted_mad": value + 3.0,
                    "smoothed_donor_weighted_mad": value + 4.0,
                    "raw_smoothed_local_linear_abs_disagreement": value + 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_contract_authorizes_stage0_only_and_keeps_stage1_closed(config: dict) -> None:
    train.validate_scientific_contract(config, require_kaggle_approval=False)
    train.validate_scientific_contract(config, require_kaggle_approval=True)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["implementation"]["enabled"] is True
    assert config["stage_0_readout"]["enabled_after_implementation"] is True
    assert config["stage_1_bounded_shrink"]["enabled_after_implementation"] is False
    assert config["execution_contract"]["stage_0"]["model_configs"] == 0
    assert config["execution_contract"]["stage_0"]["boosters"] == 0
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["run_stage_0"] is True
    assert config["execution"]["run_stage_1"] is False
    changed = copy.deepcopy(config)
    changed["exp226_geometry"]["local_linear_k"] = 24
    with pytest.raises(ValueError, match="frozen contract changed"):
        train.validate_scientific_contract(changed, require_kaggle_approval=False)


def test_exp263_readout_and_exp226_source_folds_are_separate() -> None:
    audit = train.audit_saved_fold_relationship(
        ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        [4, 4, 3, 3, 2, 2, 1, 1, 0, 0],
        [0, 1, 2, 3, 4],
    )
    assert audit["policy"].startswith("exp226_source_fold_for_donors")
    assert audit["same_fold_row_fraction"] == pytest.approx(0.2)
    assert sum(row["rows"] for row in audit["row_contingency"]) == 10


def test_identity_array_comparison_supports_strings_and_aligned_nan() -> None:
    assert train.arrays_equal_with_missing(
        np.array(["well_a", "well_b"], dtype=object),
        np.array(["well_a", "well_b"], dtype=object),
    )
    assert train.arrays_equal_with_missing(
        np.array([1.0, np.nan]), np.array([1.0, np.nan])
    )
    assert not train.arrays_equal_with_missing(
        np.array(["well_a", "well_b"], dtype=object),
        np.array(["well_a", "well_c"], dtype=object),
    )


def test_weighted_median_and_parent_local_linear_normal_matrix() -> None:
    assert train.weighted_median(np.array([1.0, 2.0, 100.0]), np.array([1.0, 4.0, 1.0])) == 2.0
    design = np.column_stack([np.ones(5), np.linspace(-0.2, 0.2, 5), np.linspace(0.3, -0.1, 5)])
    weights = np.exp(-np.linspace(0.0, 1.0, 5))
    values = np.array([2.0, 2.2, 2.4, 2.7, 3.0])
    actual, condition = train.solve_local_linear(design, weights, values, 1.0)
    normal = (design * weights[:, None]).T @ design
    normal += weights.sum() * np.diag([0.0, 1.0, 1.0])
    expected = np.linalg.solve(normal, (design * weights[:, None]).T @ values)[0]
    assert actual == pytest.approx(expected)
    assert condition == pytest.approx(np.linalg.cond(normal))


def test_reconstructed_support_uses_k50_and_excludes_target_source_fold() -> None:
    params = train.K16Params()
    count = 60
    x = np.linspace(-1000.0, 1000.0, count)
    y = np.linspace(500.0, -500.0, count)
    wi = np.arange(count, dtype=np.float64)
    segment = np.arange(count, dtype=np.float64) % 16
    raw = np.column_stack([x, y, 0.01 * x + 0.02 * y, wi, segment])
    smoothed = raw.copy()
    smoothed[:, 2] += np.linspace(-1.0, 1.0, count)
    target = train.WellGeometry(
        wid="target",
        wi=999,
        s=0,
        n=16,
        ndz=np.ones(16),
        anchor=100.0,
        segid=np.arange(16, dtype=np.int16),
        mid=np.column_stack([np.linspace(-100.0, 100.0, 16), np.zeros(16)]),
        proj=np.ones(16),
        x=np.arange(17, dtype=float),
        y=np.zeros(17),
        z=-np.arange(17, dtype=float),
    )
    wi_to_well = {index: f"donor_{index}" for index in range(count)}
    source_fold = {f"donor_{index}": index % 4 for index in range(count)}
    features, ledger = train.reconstruct_well_support(
        target,
        raw,
        smoothed,
        params,
        wi_to_well,
        source_fold,
        readout_fold=2,
        target_source_fold=4,
    )
    assert len(features) == 16
    assert len(ledger) == 16 * 50
    assert (features["donor_count"] == 50).all()
    assert not (ledger["donor_source_fold"] == 4).any()
    assert np.isfinite(features[[item[0] for item in train.RISK_SPECS]].to_numpy()).all()


def test_outer_train_ecdf_excludes_evaluation_fold_and_is_finite() -> None:
    support = synthetic_support()
    risk, cdf = train.fit_outer_train_risk(support, [0, 1, 2, 3, 4])
    assert len(risk) == len(support)
    assert risk["risk_score"].between(0.0, 1.0).all()
    fold0_reference = cdf.loc[cdf["evaluation_fold"] == 0, "reference_well_id"]
    assert "well_0" not in set(fold0_reference)
    assert set(risk.columns).issuperset(item[2] for item in train.RISK_SPECS)


def test_stable_tie_order_and_circular_control() -> None:
    percentile = train.stable_empirical_percentile(
        np.array([1.0, 1.0, 2.0]),
        ["b", "a", "c"],
        np.array([0, 0, 0]),
        np.array([1.0, 1.0]),
        ["a", "z"],
        np.array([0, 0]),
    )
    np.testing.assert_allclose(percentile, [1.0 / 3.0, 2.0 / 3.0])
    risk = pd.DataFrame(
        {
            "well_id": ["well_a"] * 16,
            "readout_fold": [0] * 16,
            "exp226_source_fold": [1] * 16,
            "segment_id": np.arange(16),
            "risk_score": np.linspace(0.0, 1.0, 16),
        }
    )
    config = {
        "stage_0_readout": {"negative_control": {"key_prefix": "exp329"}},
        "exp226_geometry": {"k_segments": 16},
    }
    controlled = train.add_circular_control(risk, config)
    offset = train.stable_circular_offset("well_a", 16, "exp329")
    assert 1 <= offset <= 15
    assert controlled["control_offset_segments"].iloc[0] == offset
    np.testing.assert_array_equal(
        np.sort(controlled["risk_score"]), np.sort(controlled["control_risk_score"])
    )
    assert not np.array_equal(
        controlled["risk_score"].to_numpy(), controlled["control_risk_score"].to_numpy()
    )


def test_auc_ties_and_segment_benefit_direction() -> None:
    labels = np.array([False, False, True, True])
    assert train.binary_auc(labels, np.array([0.0, 0.0, 1.0, 1.0])) == 1.0
    assert train.binary_auc(labels, np.ones(4)) == 0.5
    joined = pd.DataFrame(
        {
            "well_id": ["a"] * 4,
            "readout_fold": [0] * 4,
            "exp226_source_fold": [1] * 4,
            "segment_id": [0, 0, 1, 1],
            "md_since_ft": [10.0, 20.0, 30.0, 40.0],
            "risk_score": [0.1, 0.1, 0.9, 0.9],
            "control_risk_score": [0.9, 0.9, 0.1, 0.1],
            "tvt_true": [0.0, 0.0, 0.0, 0.0],
            "p_base": [2.0, 2.0, 4.0, 4.0],
            "p_other": [3.0, 3.0, 1.0, 1.0],
        }
    )
    benefit = train.build_segment_benefit(joined)
    assert benefit.loc[benefit["segment_id"] == 0, "positive_benefit"].item() is False
    assert benefit.loc[benefit["segment_id"] == 1, "positive_benefit"].item() is True


def test_truth_join_requires_frozen_contract() -> None:
    target_free = pd.DataFrame({"well_id": ["a"], "row_idx": [1], "p_base": [10.0]})
    truth = pd.DataFrame({"well_id": ["a"], "row_idx": [1], "tvt_true": [11.0]})
    with pytest.raises(ValueError, match="frozen target-free contract"):
        train.attach_truth_after_freeze(target_free, truth, target_free_contract_sha256="")
    joined = train.attach_truth_after_freeze(
        target_free, truth, target_free_contract_sha256="frozen"
    )
    assert joined["tvt_true"].item() == 11.0


def test_compact_source_is_notebook_safe_and_stage1_code_is_absent() -> None:
    source = TRAIN_PATH.read_text()
    assert "__file__" not in source
    assert "# ## Contents" in source
    assert "run_stage0_experiment(CONFIG)" in source
    assert "build_bounded_shrink" not in source
    assert (EXP_DIR / "exp329_donor_support_risk_bounded_weight_shrink_train.ipynb").exists()

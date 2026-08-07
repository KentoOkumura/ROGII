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

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm"
)
TRAIN_SOURCE = EXP_DIR / (
    "exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm_"
    "compact_selfcontained_train.py"
)
TRAIN_NOTEBOOK = EXP_DIR / (
    "exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm_"
    "compact_selfcontained_train.ipynb"
)
CANONICAL_NOTEBOOK = EXP_DIR / (
    "exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm_train.ipynb"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP397_IMPORT_ONLY")
    os.environ["EXP397_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP397_IMPORT_ONLY", None)
        else:
            os.environ["EXP397_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp397_train_test")


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_stage0_completed_fail_closed_and_rerun_is_disabled(train, config):
    contract = train.validate_scientific_contract(config)
    assert contract == {
        "diagnostic_variants": 1,
        "reporting_folds": 5,
        "hmm_well_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "boosters": 0,
    }
    assert config["experiment"]["status"] == (
        "stage_0_completed_guard_failed_closed"
    )
    assert config["implementation"]["stage_0_implemented"] is True
    assert config["implementation"]["stage_1_implemented"] is False
    assert config["implementation"]["canonical_notebook_adopted"] is True
    assert config["execution"]["kaggle_push_approved"] is False
    assert config["execution"]["train_run_approved"] is False
    assert config["execution"]["run_stage_0"] is False
    assert config["execution"]["run_stage_1"] is False
    assert config["execution"]["run_hmm"] is False
    assert config["execution"]["stage_0_gate_passed"] is False
    assert config["execution"]["stage_1_eligible"] is False
    assert config["runtime"]["kaggle"]["train_run_on_push"] is False
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        train.validate_scientific_contract(config, require_run_approval=True)

    approved = deepcopy(config)
    approved["execution"]["kaggle_push_approved"] = True
    approved["execution"]["train_run_approved"] = True
    approved["execution"]["run_stage_0"] = True
    approved["runtime"]["kaggle"]["train_run_on_push"] = True
    with pytest.raises(RuntimeError, match="must execute on Kaggle CPU"):
        train.validate_scientific_contract(approved, require_run_approval=True)


def test_contract_rejects_selector_or_gate_drift(train, config):
    changed_threshold = deepcopy(config)
    changed_threshold["model"]["agreement"]["threshold"] = 0.49
    with pytest.raises(ValueError, match="model.agreement.threshold"):
        train.validate_scientific_contract(changed_threshold)

    changed_tail = deepcopy(config)
    changed_tail["model"]["stage_0"]["tail_window_raw_rows"] = 256
    with pytest.raises(ValueError, match="tail_window_raw_rows"):
        train.validate_scientific_contract(changed_tail)

    changed_gate = deepcopy(config)
    changed_gate["model"]["stage_0"]["pass_requires_all"][
        "minimum_full_tail_multiplier_agreement"
    ] = 0.79
    with pytest.raises(ValueError, match="Stage 0 gate contract changed"):
        train.validate_scientific_contract(changed_gate)


def test_raw_finite_pairs_and_tail_window_are_fixed_before_missing_filter(
    train,
    config,
):
    rows = 700
    tvt = 100.0 + 0.25 * np.arange(rows)
    gr = 2.0 * tvt + np.sin(np.arange(rows) / 7.0)
    gr[200] = np.nan
    tvt_input = tvt.copy()
    tvt_input[600:] = np.nan
    horizontal = pd.DataFrame({"GR": gr, "TVT_input": tvt_input})

    typewell_tvt = np.linspace(0.0, 400.0, 1601)
    typewell = pd.DataFrame(
        {"TVT": typewell_tvt, "GR": 2.0 * typewell_tvt}
    )
    agreement = train.build_well_agreement(
        horizontal,
        typewell,
        well_id="well-a",
        fold=2,
        config=config,
    )

    assert agreement["full_known_raw_row_count"] == 600
    assert agreement["full_pair_count"] == 599
    assert agreement["full_first_pair_row_idx"] == 0
    assert agreement["full_last_pair_row_idx"] == 599
    assert agreement["tail_known_raw_row_count"] == 512
    assert agreement["tail_pair_count"] == 511
    assert agreement["tail_first_pair_row_idx"] == 88
    assert agreement["tail_last_pair_row_idx"] == 599
    assert agreement["full_evaluable"] is True
    assert agreement["tail_evaluable"] is True
    assert agreement["sigma_multiplier"] == pytest.approx(1.0)
    assert agreement["coefficient_group"] == "good_agreement_parent_noop"


def test_typewell_stable_sort_fill_and_endpoint_hold_are_deterministic(train):
    typewell = pd.DataFrame(
        {
            "TVT": [3.0, 1.0, 2.0, 4.0],
            "GR": [30.0, np.nan, 20.0, np.nan],
        }
    )
    tvt, gr = train.prepare_typewell(typewell)
    assert tvt.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert gr.tolist() == [20.0, 20.0, 30.0, 30.0]

    horizontal = pd.DataFrame(
        {
            "GR": [20.0, 20.0, 25.0, 30.0, 30.0],
            "TVT_input": [0.0, 1.0, 2.5, 4.0, 5.0],
        }
    )
    result = train.estimate_agreement_window(
        horizontal,
        tvt,
        gr,
        window="full_known_prefix",
        tail_rows=512,
        minimum_pairs=2,
        minimum_standard_deviation=1.0e-6,
        threshold=0.50,
        poor_multiplier=1.3,
        fallback_multiplier=1.0,
    )
    assert result["pair_count"] == 5
    assert result["mean_bias"] == pytest.approx(0.0)
    assert result["rho_gr"] == pytest.approx(1.0)


def test_boundary_support_low_variance_and_nonfinite_rules_are_exact(train):
    evaluable, multiplier, reason = train.select_sigma_multiplier(
        pair_count=64,
        horizontal_std=2.0,
        typewell_std=3.0,
        rho_gr=0.50,
        minimum_pairs=64,
        minimum_standard_deviation=1.0e-6,
        threshold=0.50,
        poor_multiplier=1.3,
        fallback_multiplier=1.0,
    )
    assert evaluable is True
    assert multiplier == pytest.approx(1.0)
    assert reason == ""

    evaluable, multiplier, reason = train.select_sigma_multiplier(
        pair_count=64,
        horizontal_std=2.0,
        typewell_std=3.0,
        rho_gr=np.nextafter(0.50, 0.0),
        minimum_pairs=64,
        minimum_standard_deviation=1.0e-6,
        threshold=0.50,
        poor_multiplier=1.3,
        fallback_multiplier=1.0,
    )
    assert evaluable is True
    assert multiplier == pytest.approx(1.3)
    assert reason == ""

    evaluable, multiplier, reason = train.select_sigma_multiplier(
        pair_count=63,
        horizontal_std=1.0e-6,
        typewell_std=2.0,
        rho_gr=np.nan,
        minimum_pairs=64,
        minimum_standard_deviation=1.0e-6,
        threshold=0.50,
        poor_multiplier=1.3,
        fallback_multiplier=1.0,
    )
    assert evaluable is False
    assert multiplier == pytest.approx(1.0)
    assert "insufficient_pair_count" in reason
    assert "horizontal_std_at_or_below_minimum" in reason
    assert "nonfinite_rho" in reason


def test_parent_clip_is_applied_before_single_multiplier_without_reclip(train):
    effective = train.apply_multiplier_to_clipped_parent_sigma(
        np.array([10.0, 30.0, 60.0]),
        np.array([1.0, 1.3, 1.3]),
    )
    assert effective.tolist() == pytest.approx([10.0, 39.0, 78.0])
    with pytest.raises(ValueError, match="already be clipped"):
        train.apply_multiplier_to_clipped_parent_sigma(61.0, 1.3)
    with pytest.raises(ValueError, match="exactly 1.0 or 1.3"):
        train.apply_multiplier_to_clipped_parent_sigma(60.0, 1.2)


def synthetic_agreement_schedule(train, config, wells: int = 20) -> pd.DataFrame:
    rows = 160
    typewell = pd.DataFrame(
        {
            "TVT": np.arange(rows, dtype=float),
            "GR": np.linspace(10.0, 100.0, rows),
        }
    )
    records = []
    for index in range(wells):
        base = typewell["GR"].to_numpy(copy=True)
        horizontal_gr = (
            base + 0.01 * np.sin(np.arange(rows) + index)
            if index % 2 == 0
            else base[::-1] + 0.01 * np.sin(np.arange(rows) + index)
        )
        horizontal = pd.DataFrame(
            {
                "GR": horizontal_gr,
                "TVT_input": np.arange(rows, dtype=float),
            }
        )
        records.append(
            train.build_well_agreement(
                horizontal,
                typewell,
                well_id=f"well-{index:03d}",
                fold=index % 5,
                config=config,
            )
        )
    return pd.DataFrame(records).sort_values("well_id").reset_index(drop=True)


def test_target_free_freeze_is_order_sensitive_and_detects_mutation(train, config):
    schedule = synthetic_agreement_schedule(train, config)
    ledger = train.TargetFreeLedger()
    freeze = train.freeze_target_free_agreement(schedule, ledger, config)
    assert ledger.frozen is True
    assert freeze["truth_rows_accessed_before_freeze"] == 0
    assert freeze["hmm_well_runs"] == 0
    assert freeze["parent_control_loaded"] is False
    train.verify_target_free_freeze(schedule, freeze)

    changed = schedule.copy()
    changed.loc[0, "full_rho_gr"] += 0.01
    with pytest.raises(RuntimeError, match="changed after target-free SHA freeze"):
        train.verify_target_free_freeze(changed, freeze)

    reversed_schedule = schedule.iloc[::-1].reset_index(drop=True)
    reversed_freeze = train.freeze_target_free_agreement(
        reversed_schedule,
        train.TargetFreeLedger(),
        config,
    )
    assert (
        reversed_freeze["agreement_logical_content_sha256"]
        == freeze["agreement_logical_content_sha256"]
    )


def test_stability_readout_uses_joint_evaluable_wells_only(train, config):
    schedule = synthetic_agreement_schedule(train, config)
    schedule.loc[0, "tail_evaluable"] = False
    schedule.loc[0, "tail_multiplier"] = 1.0
    schedule.loc[0, "tail_rho_gr"] = np.nan
    stability, fold_metrics, pooled = train.build_stability_readout(
        schedule,
        config,
    )
    assert len(stability) == 20
    assert len(fold_metrics) == 5
    assert pooled["joint_evaluable_wells"] == 19
    assert stability.loc[0, "joint_evaluable"] == np.False_
    expected = stability.loc[
        stability["joint_evaluable"],
        "full_tail_multiplier_match",
    ].mean()
    assert pooled["full_tail_multiplier_agreement"] == pytest.approx(expected)


def test_stage0_gate_requires_all_frozen_coverage_and_stability_checks(
    train,
    config,
):
    pooled = {
        "expected_wells": 773,
        "actual_wells": 773,
        "expected_folds": [0, 1, 2, 3, 4],
        "actual_folds": [0, 1, 2, 3, 4],
        "full_evaluable_fraction": 0.95,
        "fallback_fraction": 0.05,
        "poor_multiplier_fraction": 0.40,
        "tail_evaluable_fraction": 0.85,
        "full_tail_multiplier_agreement": 0.90,
        "full_tail_spearman_correlation": 0.80,
        "minimum_fold_full_evaluable_fraction": 0.90,
    }
    fold_metrics = pd.DataFrame({"fold": np.arange(5)})
    passed = train.evaluate_stage_0_gate(pooled, fold_metrics, config)
    assert passed["passed"] is True
    assert passed["stage_1_eligible"] is True
    assert passed["decision"] == (
        "stage_0_passed_stage_1_requires_separate_user_approval"
    )
    assert passed["rescue_grid_allowed"] is False

    for key, bad_value, check_name in (
        (
            "full_evaluable_fraction",
            0.89,
            "minimum_primary_evaluable_well_fraction",
        ),
        ("fallback_fraction", 0.11, "maximum_fallback_well_fraction"),
        (
            "poor_multiplier_fraction",
            0.09,
            "poor_multiplier_well_fraction_range",
        ),
        (
            "minimum_fold_full_evaluable_fraction",
            0.79,
            "minimum_each_fold_primary_evaluable_fraction",
        ),
        (
            "tail_evaluable_fraction",
            0.74,
            "minimum_tail_evaluable_well_fraction",
        ),
        (
            "full_tail_multiplier_agreement",
            0.79,
            "minimum_full_tail_multiplier_agreement",
        ),
        (
            "full_tail_spearman_correlation",
            0.69,
            "minimum_full_tail_spearman_correlation",
        ),
    ):
        failed_pooled = deepcopy(pooled)
        failed_pooled[key] = bad_value
        failed = train.evaluate_stage_0_gate(
            failed_pooled,
            fold_metrics,
            config,
        )
        assert failed["passed"] is False
        assert failed["checks"][check_name] is False
        assert failed["decision"] == "stage_0_failed_close_without_rescue"


def test_target_free_reader_never_loads_horizontal_truth_or_formation(
    train,
    tmp_path,
):
    horizontal = pd.DataFrame(
        {
            "GR": [10.0, 11.0, 12.0],
            "TVT_input": [100.0, 101.0, np.nan],
            "TVT": [100.0, 101.0, 102.0],
            "Formation": ["A", "A", "B"],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": [99.0, 100.0, 101.0],
            "GR": [9.0, 10.0, 11.0],
        }
    )
    horizontal.to_csv(tmp_path / "well-a__horizontal_well.csv", index=False)
    typewell.to_csv(tmp_path / "well-a__typewell.csv", index=False)
    ledger = train.TargetFreeLedger()
    loaded_horizontal, loaded_typewell, _, _ = train.load_target_free_well(
        tmp_path,
        "well-a",
        ledger,
    )
    assert loaded_horizontal.columns.tolist() == ["GR", "TVT_input"]
    assert loaded_typewell.columns.tolist() == ["TVT", "GR"]
    assert ledger.forbidden_rows_before_freeze == 0

    with pytest.raises(ValueError, match="forbidden pre-freeze"):
        ledger.guard_horizontal_columns(["GR", "TVT"], 3, "unsafe")


def test_candidate_notebook_is_self_contained_and_canonical_is_adopted(
    config,
):
    source = TRAIN_SOURCE.read_text()
    assert "## 1. Imports and notebook-safe configuration" in source
    assert "## 4. Target-free raw input and reporting-fold preflight" in source
    assert "## 5. Full-prefix and last-512 GR agreement helpers" in source
    assert "## 6. Coefficient freeze, stability readout, and Stage 0 gate" in source
    assert "## 7. Stage 0 orchestration and generated artifacts" in source
    assert "def build_well_agreement" in source
    assert "def freeze_target_free_agreement" in source
    assert "def evaluate_stage_0_gate" in source
    assert "def run_stage_0_experiment" in source
    assert "from settings import" not in source
    assert "from src" not in source
    assert "Path(__" + "file__)" not in source
    assert TRAIN_NOTEBOOK.is_file()
    assert CANONICAL_NOTEBOOK.is_file()
    assert config["implementation"]["train_notebook_state"] == (
        "compact_selfcontained_canonical_adopted"
    )

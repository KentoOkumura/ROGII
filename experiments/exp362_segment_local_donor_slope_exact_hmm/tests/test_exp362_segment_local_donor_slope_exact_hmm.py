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

from tests.test_support import ensure_numba_test_stub, require_saved_files

ensure_numba_test_stub()

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "exp362_segment_local_donor_slope_exact_hmm"
TRAIN_SOURCE = EXP_DIR / (
    "exp362_segment_local_donor_slope_exact_hmm_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp362_segment_local_donor_slope_exact_hmm_compact_selfcontained_inference.py"
)
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    / "exact_hmm_smoother.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP362_IMPORT_ONLY")
    os.environ["EXP362_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP362_IMPORT_ONLY", None)
        else:
            os.environ["EXP362_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp362_train_test")


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_contract_is_one_773_well_hmm_candidate_and_completed_run_is_disabled(
    train, config
) -> None:
    counts = train.validate_scientific_contract(config)
    assert counts == {
        "scientific_variants": 1,
        "reporting_folds": 5,
        "hmm_well_runs": 773,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "gpu": False,
        "parent_control_retraining": False,
    }
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["canonical_notebook_adopted"] is True
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["active_stage"] == "completed_postrun_support_audit_failed_closed"
    assert config["execution"]["run_hmm"] is False
    with pytest.raises(RuntimeError, match="run_hmm must be explicitly enabled"):
        train.validate_scientific_contract(config, require_run_approval=True)

    unapproved = deepcopy(config)
    unapproved["execution"]["kaggle_push_approved"] = False
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        train.validate_scientific_contract(unapproved, require_run_approval=True)

    approved = deepcopy(config)
    approved["execution"]["run_hmm"] = True
    train.validate_scientific_contract(approved, require_run_approval=True)


def test_stable_fold_assignment_is_hash_ordered_and_group_safe(train) -> None:
    wells = [f"well_{index:02d}" for index in range(17)]
    first = train.stable_fold_assignment(wells, seed=42, n_folds=5)
    second = train.stable_fold_assignment(list(reversed(wells)), seed=42, n_folds=5)
    pd.testing.assert_frame_equal(first, second)
    assert first["well_id"].nunique() == len(wells)
    assert set(first["fold"]) == {0, 1, 2, 3, 4}
    ordered = first.sort_values(["fold_order_sha256", "well_id"], kind="mergesort").reset_index(
        drop=True
    )
    np.testing.assert_array_equal(ordered["fold"], np.arange(len(wells)) % 5)


def test_k16_donor_segment_fit_recovers_u_rate_and_heading(train) -> None:
    md = np.arange(160, dtype=float)
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": 10.0 + 0.6 * md,
            "Y": 20.0 - 0.8 * md,
            "Z": -0.2 * md,
            "TVT": 100.0 + 0.25 * md,
        }
    )
    segments = train.build_donor_segments_for_well("donor_a", horizontal, k_segments=16)
    assert len(segments) == 16
    np.testing.assert_allclose(segments["heading_x"], 0.6, atol=1.0e-12)
    np.testing.assert_allclose(segments["heading_y"], -0.8, atol=1.0e-12)
    np.testing.assert_allclose(segments["u_rate"], 0.05, atol=1.0e-12)
    assert (segments["finite_rows"] == 10).all()


def local_gradient_donors() -> pd.DataFrame:
    rows = []
    gradient = np.array([0.03, -0.01])
    for well_index in range(12):
        angle = 2.0 * np.pi * well_index / 12.0
        heading = np.array([np.cos(angle), np.sin(angle)])
        for segment_id, distance in [(0, 25.0 + well_index), (1, 400.0 + well_index)]:
            rows.append(
                {
                    "outer_fold": 0,
                    "donor_well_id": f"d{well_index:02d}",
                    "segment_id": segment_id,
                    "center_x": distance,
                    "center_y": 0.0,
                    "u_rate": float(heading @ gradient),
                    "heading_x": heading[0],
                    "heading_y": heading[1],
                }
            )
    return pd.DataFrame(rows)


def test_local_gradient_uses_one_nearest_segment_per_donor_and_recovers_plane(
    train, config
) -> None:
    target = {
        "center_x": 0.0,
        "center_y": 0.0,
        "heading_x": 1.0,
        "heading_y": 0.0,
        "target_geometry_valid": True,
    }
    estimate = train.estimate_local_gradient_prior(
        target,
        local_gradient_donors(),
        prefix_rate=0.03,
        config=config,
    )
    assert estimate["candidate_donor_wells"] == 12
    assert estimate["selected_donor_wells"] == 12
    assert estimate["fallback"] is False
    assert estimate["effective_donors"] >= 10.0
    assert estimate["directional_information"] >= 0.30
    assert estimate["mu_rate"] == pytest.approx(0.03, abs=5.0e-5)


def test_local_gradient_falls_back_on_direction_and_rate_guards(train, config) -> None:
    target = {
        "center_x": 0.0,
        "center_y": 0.0,
        "heading_x": 1.0,
        "heading_y": 0.0,
        "target_geometry_valid": True,
    }
    donors = local_gradient_donors()
    donors["heading_x"] = 0.0
    donors["heading_y"] = 1.0
    donors["u_rate"] = 0.02
    direction = train.estimate_local_gradient_prior(target, donors, prefix_rate=0.03, config=config)
    assert direction["fallback"] is True
    assert direction["fallback_reason"] == "directional_information"
    assert direction["mu_rate"] == 0.03

    abnormal = local_gradient_donors()
    abnormal["u_rate"] = abnormal["heading_x"] * 0.20
    rate = train.estimate_local_gradient_prior(target, abnormal, prefix_rate=0.03, config=config)
    assert rate["fallback"] is True
    assert rate["fallback_reason"] == "mu_prefix_delta"
    assert rate["mu_rate"] == 0.03


def test_target_schedule_and_hmm_preparation_are_truth_free(train, config) -> None:
    known_rows = 40
    suffix_rows = 32
    rows = known_rows + suffix_rows
    md = np.arange(rows, dtype=float) * 5.0
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": 0.8 * md,
            "Y": 0.6 * md,
            "Z": -0.2 * md,
            "GR": np.linspace(40.0, 80.0, rows),
            "TVT_input": np.r_[
                100.0 + 0.4 * md[:known_rows],
                [np.nan] * suffix_rows,
            ],
        }
    )
    target, eval_index = train.build_target_segments("well_a", horizontal, 0, k_segments=16)
    assert len(target) == 16
    assert target["target_geometry_valid"].all()
    schedule = pd.DataFrame(
        {
            "id": [f"well_a_{row}" for row in eval_index],
            "well_id": "well_a",
            "row_idx": eval_index,
            "suffix_offset": np.arange(suffix_rows),
            "fold": 0,
            "segment_id": train.md_segment_ids(md[eval_index], 16),
            "md": md[eval_index],
            "md_since": md[eval_index] - md[known_rows - 1],
            "mu_rate": np.linspace(0.20, 0.24, suffix_rows),
            "prefix_rate": 0.20,
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(80.0, 250.0, 200),
            "GR": np.linspace(35.0, 85.0, 200),
        }
    )
    prepared = train.prepare_hmm_inputs(horizontal, typewell, schedule, config)
    np.testing.assert_allclose(
        prepared["effective_dz"],
        prepared["dz"] - prepared["prior_mu"] * prepared["dm"],
    )
    assert prepared["rates"][0] == pytest.approx(-0.10)
    assert prepared["rates"][-1] == pytest.approx(0.10)
    assert len(prepared["rates"]) == 41
    assert not any(
        forbidden in schedule.columns
        for forbidden in ("TVT", "tvt_true", "tvt_pred", "tvt_geop", "gr_delta")
    )


def test_exact_kernel_is_bitwise_equal_to_the_exp209_parent_kernel(train) -> None:
    parent = load_module(PARENT_SOURCE, "exp209_parent_for_exp362")
    rng = np.random.default_rng(7)
    emission = rng.normal(size=(4, 31)).astype(np.float32)
    dm = np.full(4, 5.0, dtype=np.float64)
    dz = np.linspace(-0.2, 0.2, 4)
    rates = np.linspace(-0.1, 0.1, 5)
    arguments = (
        emission,
        dm,
        dz,
        0.35,
        rates,
        0.002,
        0.02,
        15.0,
        0.75,
        0.01,
        0.01,
        1.0,
        0.998,
    )
    candidate_post, candidate_loglik = train._hmm2_fb(*arguments)
    parent_post, parent_loglik = parent._hmm2_fb(*arguments)
    np.testing.assert_array_equal(candidate_post, parent_post)
    assert candidate_loglik == parent_loglik


def test_truth_free_loader_excludes_tvt_and_inference_is_fail_closed(train, tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "X": [0.0, 1.0],
            "Y": [0.0, 0.0],
            "Z": [0.0, -0.1],
            "GR": [50.0, np.nan],
            "TVT_input": [100.0, np.nan],
            "TVT": [100.0, 101.0],
        }
    ).to_csv(tmp_path / "a__horizontal_well.csv", index=False)
    frame = train.load_horizontal_without_truth("a", tmp_path)
    assert list(frame.columns) == ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    assert "TVT" not in frame

    inference = load_module(INFERENCE_SOURCE, "exp362_inference_test")
    with pytest.raises(RuntimeError, match="inference and submission are disabled"):
        inference.assert_inference_disabled()


def test_hidden_like_dependency_is_preflighted_without_parsing_roles_early(train, config) -> None:
    require_saved_files(
        ROOT
        / "experiments/exp115_hidden_like_spatial_holdout_from_ppt"
        / "artifacts/exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv"
    )
    report = train.preflight_hidden_like_dependency(config)
    assert report["raw_sha256"] == config["data"]["hidden_like"]["expected_sha256"]
    assert report["roles_parsed_before_prediction_freeze"] is False
    assert {
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    }.issubset(report["columns"])


def test_scientific_gate_enforces_parent_parity_and_all_longtail_guards(train, config) -> None:
    local = deepcopy(config)
    local["validation"]["expected_rows"] = 10
    local["validation"]["expected_wells"] = 5
    local["execution_contract"]["hmm_well_runs"] = 5
    parent_rmse = float(local["data"]["exp209_control"]["direct_rmse_ft"])
    frame = pd.DataFrame(
        {
            "well_id": np.repeat([f"w{fold}" for fold in range(5)], 2),
            "row_idx": np.tile([0, 1], 5),
            "fold": np.repeat(np.arange(5), 2),
            "md_since": np.full(10, 1200.0),
            "true_tvt": np.zeros(10),
            "parent_tvt": np.full(10, parent_rmse),
            "candidate_tvt": np.full(10, parent_rmse - 0.10),
            "hidden_like_spatial": np.ones(10, dtype=bool),
            "hidden_like_typewell_purged": np.ones(10, dtype=bool),
        }
    )
    fold_metrics, distance_metrics, hidden_metrics = train.build_scope_metrics(frame, local)
    by_well = train.build_by_well_metrics(frame)
    donor = pd.DataFrame(
        {
            "outer_fold": np.arange(5),
            "well_fold": (np.arange(5) + 1) % 5,
        }
    )
    schedule = train.FrozenPriorSchedule(
        donor_ledger=donor,
        target_segments=pd.DataFrame(),
        rowwise_schedule=pd.DataFrame(),
        fold_assignment=pd.DataFrame(),
        donor_ledger_content_sha256="a" * 64,
        target_segments_content_sha256="b" * 64,
        rowwise_schedule_content_sha256="c" * 64,
    )
    runtime = pd.DataFrame(
        {
            "well_id": [f"w{fold}" for fold in range(5)],
            "posterior_row_sum_max_abs_error": np.zeros(5),
        }
    )
    frozen = train.FrozenPrediction(
        frame=pd.DataFrame(),
        runtime=runtime,
        prediction_content_sha256="d" * 64,
        runtime_content_sha256="e" * 64,
    )
    parent_report = {
        "decompressed_sha256": local["data"]["exp209_control"][
            "expected_hmm_cache_decompressed_sha256"
        ]
    }
    gate = train.evaluate_gate(
        frame,
        fold_metrics,
        distance_metrics,
        hidden_metrics,
        by_well,
        schedule,
        frozen,
        parent_report,
        runtime_seconds=10.0,
        config=local,
    )
    assert gate["technical_gate"]["passed"] is True
    assert gate["scientific_gate"]["passed"] is True
    assert gate["passed"] is True

    failed = frame.copy()
    failed.loc[failed["well_id"] == "w4", "candidate_tvt"] = parent_rmse + 1.0
    fold_metrics, distance_metrics, hidden_metrics = train.build_scope_metrics(failed, local)
    by_well = train.build_by_well_metrics(failed)
    gate = train.evaluate_gate(
        failed,
        fold_metrics,
        distance_metrics,
        hidden_metrics,
        by_well,
        schedule,
        frozen,
        parent_report,
        runtime_seconds=10.0,
        config=local,
    )
    assert gate["scientific_gate"]["passed"] is False
    assert gate["passed"] is False

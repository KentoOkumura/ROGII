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
    / "exp406_loop_closed_multiwell_rgt_fixed16_stage0"
)
SOURCE = EXP_DIR / (
    "exp406_loop_closed_multiwell_rgt_fixed16_stage0_"
    "compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp406_loop_closed_multiwell_rgt_fixed16_stage0_"
    "compact_selfcontained_inference.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"
EXP226_SOURCE = (
    ROOT
    / "experiments"
    / "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
    / "connortynan_k16_reproduction.py"
)
EXP386_SOURCE = (
    ROOT
    / "experiments"
    / "exp386_cycle_consistent_rgt_scenario_bank"
    / "exp386_cycle_consistent_rgt_scenario_bank_compact_selfcontained_train.py"
)


def load_source():
    previous = os.environ.get("EXP406_IMPORT_ONLY")
    os.environ["EXP406_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("exp406_train_test", SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP406_IMPORT_ONLY", None)
        else:
            os.environ["EXP406_IMPORT_ONLY"] = previous


def load_exp226_source():
    spec = importlib.util.spec_from_file_location("exp226_source_parity", EXP226_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_inference_source():
    previous = os.environ.get("EXP406_IMPORT_ONLY")
    os.environ["EXP406_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(
            "exp406_inference_test",
            INFERENCE_SOURCE,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP406_IMPORT_ONLY", None)
        else:
            os.environ["EXP406_IMPORT_ONLY"] = previous


def load_exp386_source():
    previous = os.environ.get("EXP386_IMPORT_ONLY")
    os.environ["EXP386_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(
            "exp386_selector_parity",
            EXP386_SOURCE,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP386_IMPORT_ONLY", None)
        else:
            os.environ["EXP386_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def candidate():
    return load_source()


@pytest.fixture(scope="module")
def exp226_source():
    return load_exp226_source()


@pytest.fixture(scope="module")
def inference_candidate():
    return load_inference_source()


@pytest.fixture(scope="module")
def exp386_source():
    return load_exp386_source()


@pytest.fixture()
def config():
    value = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(value, dict)
    return value


def test_contract_disables_execution_after_fixed16_stage0_close(candidate, config):
    counts = candidate.validate_execution_contract(
        config,
        require_kaggle_authorization=False,
    )
    assert counts["target_wells"] == 16
    assert counts["reporting_folds"] == 5
    assert counts["model_configs"] == 0
    assert counts["lightgbm_boosters"] == 0
    assert config["implementation"]["enabled"] is True
    assert config["execution"]["implementation_authorized"] is True
    assert config["execution"]["kaggle_execution_authorized"] is False
    assert config["execution"]["full_oof_stage1_authorized"] is False
    assert config["execution"]["inference_authorized"] is False
    assert config["execution"]["submission_authorized"] is False
    with pytest.raises(RuntimeError, match="Kaggle execution is not authorized"):
        candidate.validate_execution_contract(
            config,
            require_kaggle_authorization=True,
        )
    broken = deepcopy(config)
    broken["pairwise_gr"]["shift_grid_ft"]["step"] = 2.5
    with pytest.raises(ValueError, match="shift grid"):
        candidate.validate_execution_contract(
            broken,
            require_kaggle_authorization=False,
        )


def test_inference_is_fail_closed(inference_candidate, config):
    contract = inference_candidate.validate_disabled_inference(config)
    assert not any(contract.values())
    with pytest.raises(RuntimeError, match="diagnostics only"):
        inference_candidate.run_inference()
    broken = deepcopy(config)
    broken["execution"]["inference_authorized"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        inference_candidate.validate_disabled_inference(broken)


def test_fixed16_selector_is_round_robin_fold_sorted_offset(
    candidate,
    exp386_source,
):
    fold_by_well = {
        f"f{fold}_{offset:02d}": fold
        for fold in range(5)
        for offset in range(5)
    }
    selected = candidate.select_fixed16_wells(
        fold_by_well,
        maximum_wells=16,
        folds=[0, 1, 2, 3, 4],
    )
    expected = [
        *(f"f{fold}_00" for fold in range(5)),
        *(f"f{fold}_01" for fold in range(5)),
        *(f"f{fold}_02" for fold in range(5)),
        "f0_03",
    ]
    assert selected == expected
    exp386_selected = exp386_source.select_preflight_wells(
        fold_by_well,
        maximum_wells=16,
        folds=[0, 1, 2, 3, 4],
    )
    assert set(selected) == exp386_selected


def test_role_ledger_allows_target_gr_and_blocks_truth_or_formation(candidate):
    ledger = candidate.RoleReadLedger()
    ledger.record_target_safe(
        0,
        "target",
        ["MD", "X", "Y", "Z", "GR", "TVT_input"],
        10,
    )
    with pytest.raises(ValueError, match="forbidden columns"):
        ledger.record_target_safe(
            0,
            "target",
            ["MD", "GR", "TVT", "ANCC"],
            10,
        )
    with pytest.raises(ValueError, match="before graph SHA freeze"):
        ledger.record_prefix_truth_late(0, "target", 10)
    ledger.mark_frozen(
        {
            "pairwise_edges": "a",
            "cycle_basis": "b",
            "loop_closed_gauge": "c",
        }
    )
    ledger.record_prefix_truth_late(0, "target", 10)
    assert ledger.summary()["prefix_truth_joined_after_freeze"] is True
    assert ledger.summary()["suffix_truth_reads_before_freeze"] == 0
    assert ledger.summary()["target_formation_reads_before_freeze"] == 0


def test_morphology_prefers_matching_shape(candidate, config):
    rows = 256
    x = np.arange(rows, dtype=float)
    query = 80.0 + 14.0 * np.sin(x / 11.0) + 7.0 * np.cos(x / 27.0)
    matching = query * 1.7 + 12.0
    mismatch = matching[::-1]
    real = candidate.morphology_score(query, matching, config)
    wrong = candidate.morphology_score(query, mismatch, config)
    assert real["eligible"] is True
    assert real["finite_pairs"] == rows
    assert real["score"] > 0.99
    assert real["score"] > wrong["score"] + 0.5


def test_circular_control_is_stable_and_preserves_nan_mask(candidate, config):
    values = np.arange(256, dtype=float)
    values[[3, 91, 200]] = np.nan
    first, first_offset = candidate.stable_circular_control(
        values,
        "target-a",
        7,
        config,
    )
    second, second_offset = candidate.stable_circular_control(
        values,
        "target-a",
        7,
        config,
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.isnan(first), np.isnan(values))
    assert first_offset == second_offset
    assert first_offset >= int(np.ceil(0.25 * np.isfinite(values).sum()))


def make_context(well: str, shift: float = 0.0, rows: int = 384):
    row = np.arange(rows, dtype=np.int32)
    base = 1000.0 + 0.5 * row
    gr = (
        80.0
        + 15.0 * np.sin((base + shift) / 8.0)
        + 5.0 * np.cos((base + shift) / 21.0)
    )
    return pd.DataFrame(
        {
            "MD": row.astype(float),
            "X": np.full(rows, float(ord(well[-1]))),
            "Y": np.zeros(rows),
            "Z": -row.astype(float),
            "GR": gr,
            "TVT_input": base,
            "base_tvt": base,
            "well_id": well,
            "row_idx": row,
            "fold": 0,
            "role": "outer_valid" if well == "target" else "outer_train",
        }
    )


def test_donor_priority_and_pairwise_edges_are_capped(candidate, config):
    contexts = {
        "target": make_context("target", shift=10.0),
        "donor_a": make_context("donor_a"),
        "donor_b": make_context("donor_b"),
        "donor_c": make_context("donor_c"),
    }
    groups = {
        "target": "same",
        "donor_a": "other",
        "donor_b": "same",
        "donor_c": "other",
    }
    selected = candidate.select_donor_wells(
        "target",
        ["donor_a", "donor_b", "donor_c"],
        contexts,
        groups,
        maximum=3,
    )
    assert selected[0] == "donor_b"
    edges, funnel, blocks = candidate.build_pairwise_edges(
        fold=0,
        contexts=contexts,
        target_wells=["target"],
        donor_wells=["donor_a", "donor_b", "donor_c"],
        typewell_by_well=groups,
        config=config,
    )
    assert not edges.empty
    assert not funnel.empty
    assert not blocks.empty
    assert not edges["target_well_id"].eq("target").any()
    per_block = edges.groupby(
        ["source_well_id", "source_block_id"],
        sort=False,
    ).size()
    assert int(per_block.max()) <= 4
    target_edges = edges.loc[edges["source_well_id"].eq("target")]
    assert np.isfinite(target_edges["real_ncc"]).all()


def test_loop_solver_closes_inconsistent_triangle(candidate, config):
    edge_rows = []
    for rank, (source, target, measurement) in enumerate(
        [("a", "b", 1.0), ("b", "c", 2.0), ("a", "c", 4.0)],
        start=1,
    ):
        edge_rows.append(
            {
                "fold": 0,
                "source_well_id": source,
                "target_well_id": target,
                "source_role": "outer_train",
                "target_role": "outer_train",
                "source_block_id": 0,
                "target_block_id": 0,
                "edge_rank": rank,
                "edge_id": f"{source}->{target}",
                "relative_offset_ft": measurement,
                "real_ncc": 0.8,
                "pair_fraction": 1.0,
            }
        )
    edges = pd.DataFrame(edge_rows)
    blocks = pd.DataFrame(
        [
            {
                "fold": 0,
                "well_id": well,
                "role": "outer_train",
                "block_id": 0,
                "center_row_idx": 128,
                "center_tvt": 1000.0,
                "last_finite_prefix_row": -1,
            }
            for well in ("a", "b", "c")
        ]
    )
    solved, gauge, manifest = candidate.solve_loop_closed_graph(
        edges,
        blocks,
        target_wells=[],
        config=config,
    )
    assert manifest["fundamental_cycles"] == 1
    cycles = manifest["cycles"]
    assert cycles.iloc[0]["raw_cycle_residual_abs_ft"] == pytest.approx(1.0)
    assert cycles.iloc[0]["solved_cycle_residual_abs_ft"] < 1.0e-8
    assert np.isfinite(solved["edge_residual_ft"]).all()
    assert gauge["finite_gauge"].all()


def test_k16_pseudo_target_uses_exact_heldout_geometry_rows(candidate, config):
    rows = 1700
    row = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "well_id": "target",
            "X": row,
            "Y": 0.25 * row,
            "Z": -0.5 * row,
            "TVT_input": np.where(row < 1600, 1000.0 + 0.6 * row, np.nan),
        }
    )
    params = candidate.k16_params(config)
    target = candidate.make_k16_pseudo_target(
        frame,
        pseudo_cut_row=1087,
        heldout_rows=512,
        params=params,
    )
    assert target.n == 512
    assert len(target.ndz) == 512
    assert target.ndz == pytest.approx(np.full(512, 0.5))
    assert target.anchor == pytest.approx(1000.0 + 0.6 * 1087)


def test_k16_geometry_primitives_match_exp226_source(candidate, exp226_source, config):
    actual_params = candidate.k16_params(config)
    expected_params = exp226_source.K16Params()
    for name in (
        "theta0",
        "k_segments",
        "local_linear_k",
        "local_linear_bandwidth",
        "local_linear_ridge",
        "smooth_rho",
        "gate",
        "field_min_proj",
        "kbins",
        "kappa_regimes",
        "rot_max_deg",
        "ancc_theta_bandwidth",
    ):
        assert getattr(actual_params, name) == getattr(expected_params, name)
    rows = 900
    row = np.arange(rows, dtype=float)
    x = 100.0 + 0.7 * row
    y = 200.0 + 0.2 * row + 3.0 * np.sin(row / 40.0)
    actual_geometry = candidate.k16_segment_geometry(
        x,
        y,
        199,
        512,
        actual_params,
    )
    expected_geometry = exp226_source.segment_geometry(
        x,
        y,
        199,
        512,
        expected_params,
    )
    for actual, expected in zip(actual_geometry, expected_geometry, strict=True):
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)
    ndz = 0.4 + 0.02 * np.sin(np.arange(512) / 17.0)
    u = np.cumsum(ndz)
    r0 = u + 0.1 * np.sin(np.arange(512) / 80.0)
    actual_coefficients = candidate.k16_fit_coeffs(
        r0,
        u,
        512,
        actual_params,
        rho=actual_params.smooth_rho,
    )
    expected_coefficients = exp226_source.fit_coeffs(
        r0,
        u,
        512,
        expected_params,
        rho=expected_params.smooth_rho,
    )
    np.testing.assert_allclose(
        actual_coefficients,
        expected_coefficients,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_prefix_gate_requires_gain_and_four_folds(candidate, config):
    rows = [
        {"scope": "pooled", "fold": -1, "gain_vs_exp226_ft": 0.30},
        *[
            {
                "scope": f"fold_{fold}",
                "fold": fold,
                "gain_vs_exp226_ft": 0.1 if fold < 4 else -0.1,
            }
            for fold in range(5)
        ],
    ]
    gate = candidate.evaluate_prefix_gate(pd.DataFrame(rows), config)
    assert gate["passed"] is True
    rows[0]["gain_vs_exp226_ft"] = 0.24
    assert (
        candidate.evaluate_prefix_gate(pd.DataFrame(rows), config)["passed"]
        is False
    )


def test_resource_projection_excludes_prefix_control_time(candidate, config):
    resources = candidate.resource_projection(
        target_free_elapsed_seconds=16.0,
        total_elapsed_seconds=116.0,
        fixed16_wells=16,
        config=config,
    )
    expected = 16.0 * int(config["validation"]["expected_wells"]) / 16
    assert resources["projected_full_runtime_seconds"] == expected
    assert resources["fixed16_prefix_diagnostic_elapsed_seconds"] == 100.0
    assert (
        resources["projection_method"]
        == "fixed16_target_free_graph_elapsed_linear_well_scaling;"
        "prefix_pseudocut_control_is_stage0_diagnostic_only"
    )


def test_candidate_is_not_a_thin_helper_notebook():
    source = SOURCE.read_text()
    assert "Path(__file__)" not in source
    assert "__file__" not in source
    for section in (
        "## 2. Runtime, path, SHA, and serialization helpers",
        "## 4. Block construction, donor selection, and GR morphology",
        "## 6. Fundamental cycles and Huber IRLS loop closure",
        "## 8. Fixed16-only exp226 K16 geometry control and prefix readout",
        "## 9. Resource projection, artifact manifest, and orchestration",
    ):
        assert section in source
    assert "from settings import" not in source
    assert len(source.splitlines()) > 1500
    inference_source = INFERENCE_SOURCE.read_text()
    assert "Path(__file__)" not in inference_source
    assert "submission" in inference_source
    assert "fail-closed" in inference_source

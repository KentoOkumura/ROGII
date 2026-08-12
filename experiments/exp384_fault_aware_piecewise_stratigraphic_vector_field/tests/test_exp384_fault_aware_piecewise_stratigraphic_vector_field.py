from __future__ import annotations

import os
import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

EXPERIMENT_DIR = Path(
    "experiments/exp384_fault_aware_piecewise_stratigraphic_vector_field"
)
TRAIN_SOURCE = EXPERIMENT_DIR / (
    "exp384_fault_aware_piecewise_stratigraphic_vector_field_"
    "compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXPERIMENT_DIR / (
    "exp384_fault_aware_piecewise_stratigraphic_vector_field_"
    "compact_selfcontained_inference.py"
)
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"


def load_namespace(path: Path = TRAIN_SOURCE) -> dict[str, object]:
    previous = os.environ.get("EXP384_IMPORT_ONLY")
    os.environ["EXP384_IMPORT_ONLY"] = "1"
    try:
        return runpy.run_path(str(path))
    finally:
        if previous is None:
            os.environ.pop("EXP384_IMPORT_ONLY", None)
        else:
            os.environ["EXP384_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def module() -> dict[str, object]:
    return load_namespace()


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(value, dict)
    return value


def test_execution_contract_records_implementation_but_blocks_run(
    module: dict[str, object],
    config: dict,
) -> None:
    module["validate_execution_contract"](config, require_kaggle_authorization=False)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["execution"]["implementation_authorized"] is True
    assert (
        config["execution"]["implementation_approval_source"]
        == "user_message_implement_exp384_2026_07_24"
    )
    assert config["execution"]["kaggle_execution_authorized"] is False
    assert config["runtime"]["scientific_candidates"] == 1
    assert config["runtime"]["reporting_folds"] == 5
    assert config["runtime"]["fitted_models"] == 0
    assert config["runtime"]["hmm_runs"] == 0
    assert config["runtime"]["pf_runs"] == 0
    assert config["runtime"]["beam_runs"] == 0
    assert config["runtime"]["lightgbm_boosters"] == 0
    assert config["runtime"]["replay_parent_control"] is False
    with pytest.raises(RuntimeError, match="not authorized"):
        module["validate_execution_contract"](
            config, require_kaggle_authorization=True
        )


def _base_node(module: dict[str, object], index: int, well: str) -> dict[str, object]:
    row: dict[str, object] = {
        "fold": 0,
        "role": "outer_train",
        "well_id": well,
        "MD": float(index),
        "X": float(index * 100),
        "Y": 0.0,
        "Z": -1000.0,
        "window_scale_ft": 256.0,
        "S_true": 100.0 + index,
        "tangent_x": 1.0,
        "tangent_y": 0.0,
        "rate_true": 1.0,
        "window_residual_variance": 1.0,
        "smooth_absolute_residual": 0.0,
        "smooth_rate_residual": 0.0,
    }
    for column in module["SIGNATURE_COLUMNS"]:
        row[column] = 0.0
    for formation_index, column in enumerate(module["SURFACE_COLUMNS"]):
        row[column] = float(formation_index * 10)
    for column in module["SURFACE_GRAD_X_COLUMNS"]:
        row[column] = 0.0
    for column in module["SURFACE_GRAD_Y_COLUMNS"]:
        row[column] = 0.0
    for column in module["SURFACE_VARIANCE_COLUMNS"]:
        row[column] = 1.0
    for column in module["FAULT_SURFACE_RESIDUAL_COLUMNS"]:
        row[column] = 0.0
    for column in module["THICKNESS_COLUMNS"]:
        row[column] = 10.0
    return row


def test_role_guard_rejects_outer_valid_graph_rows(
    module: dict[str, object],
    config: dict,
) -> None:
    rows = [_base_node(module, index, f"w{index:02d}") for index in range(16)]
    frame = pd.DataFrame(rows)
    ledger = module["RoleReadLedger"]()
    canonical = module["canonicalize_graph_nodes"](frame, config, ledger)
    assert len(canonical) == 16
    contaminated = frame.copy()
    contaminated.loc[0, "role"] = "outer_valid"
    with pytest.raises(ValueError, match="non-outer-train"):
        module["canonicalize_graph_nodes"](
            contaminated, config, module["RoleReadLedger"]()
        )


def test_train_query_contract_rejects_visible_test_role(
    module: dict[str, object],
) -> None:
    outer_valid = pd.DataFrame([_query_row(module)])
    module["validate_target_safe_query"](
        outer_valid, module["RoleReadLedger"]()
    )
    visible_test = outer_valid.assign(role="test")
    with pytest.raises(ValueError, match="unsupported roles"):
        module["validate_target_safe_query"](
            visible_test, module["RoleReadLedger"]()
        )


def test_fault_edges_use_fixed_and_cut_rule(
    module: dict[str, object],
    config: dict,
) -> None:
    config = deepcopy(config)
    config["method"]["graph"]["nearest_unique_wells"] = 3
    config["method"]["graph"]["initial_neighbor_query"] = 16
    rows = [_base_node(module, index, f"w{index:02d}") for index in range(16)]
    # One deterministic outlier creates both formation and structural jumps.
    for column in module["FAULT_FORMATION_COLUMNS"]:
        rows[-1][column] = 100.0
    rows[-1]["smooth_absolute_residual"] = 100.0
    rows[-1]["smooth_rate_residual"] = 100.0
    frame = pd.DataFrame(rows)
    nodes = module["canonicalize_graph_nodes"](
        frame, config, module["RoleReadLedger"]()
    )
    edges = module["build_fault_graph"](nodes, config)
    assert edges["cut"].any()
    assert np.array_equal(
        edges["cut"].to_numpy(),
        (
            edges["formation_jump"].to_numpy()
            & edges["structural_jump"].to_numpy()
        ),
    )
    uncut = edges.loc[~edges["cut"]]
    assert (uncut["node_u"] < uncut["node_v"]).all()


def test_component_ids_are_stable_and_small_components_are_ineligible(
    module: dict[str, object],
    config: dict,
) -> None:
    rows = [_base_node(module, index, f"w{index:02d}") for index in range(12)]
    nodes = pd.DataFrame(rows).sort_values(["well_id", "MD"]).reset_index(drop=True)
    nodes["node_id"] = np.arange(len(nodes))
    edge_rows = []
    for index in range(9):
        edge_rows.append(
            {
                "fold": 0,
                "node_u": index,
                "node_v": index + 1,
                "distance_ft": 1.0,
                "cut": False,
            }
        )
    edge_rows.append(
        {
            "fold": 0,
            "node_u": 9,
            "node_v": 10,
            "distance_ft": 1.0,
            "cut": True,
        }
    )
    edge_rows.append(
        {
            "fold": 0,
            "node_u": 10,
            "node_v": 11,
            "distance_ft": 1.0,
            "cut": False,
        }
    )
    edges = pd.DataFrame(edge_rows)
    first, summary = module["assign_fault_components"](nodes, edges, config)
    second, _ = module["assign_fault_components"](
        nodes.sample(frac=1.0, random_state=11), edges.sample(frac=1.0, random_state=7), config
    )
    first = first.sort_values("node_id").reset_index(drop=True)
    second = second.sort_values("node_id").reset_index(drop=True)
    assert first["component_id"].tolist() == second["component_id"].tolist()
    assert summary["eligible"].sum() == 1
    assert sorted(summary["unique_wells"].tolist()) == [2, 10]


def test_no_cut_graph_has_no_piecewise_eligible_component(
    module: dict[str, object],
    config: dict,
) -> None:
    rows = [_base_node(module, index, f"w{index:02d}") for index in range(12)]
    nodes = pd.DataFrame(rows)
    nodes["node_id"] = np.arange(len(nodes))
    edges = pd.DataFrame(
        [
            {
                "fold": 0,
                "node_u": index,
                "node_v": index + 1,
                "distance_ft": 1.0,
                "cut": False,
            }
            for index in range(11)
        ]
    )
    assignments, summary = module["assign_fault_components"](nodes, edges, config)
    assert not assignments["component_eligible"].any()
    assert summary["fault_boundary_edges"].eq(0).all()


def _planar_donors(module: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(20):
        x = float((index % 5) * 100)
        y = float((index // 5) * 100)
        s_true = 100.0 + 2.0 * x - 3.0 * y
        row = _base_node(module, index, f"w{index:02d}")
        row.update(
            {
                "node_id": index,
                "X": x,
                "Y": y,
                "S_true": s_true,
                "rate_true": 2.0,
                "tangent_x": 1.0,
                "tangent_y": 0.0,
            }
        )
        for formation_index, name in enumerate(module["FORMATION_NAMES"]):
            row[f"surface_{name}"] = (
                float(formation_index * 10) + 0.5 * x + 0.25 * y
            )
            row[f"surface_grad_x_{name}"] = 0.5
            row[f"surface_grad_y_{name}"] = 0.25
        rows.append(row)
    return pd.DataFrame(rows)


def _query_row(module: dict[str, object], *, query_id: str = "q0") -> pd.Series:
    x = 175.0
    y = 125.0
    row: dict[str, object] = {
        "fold": 0,
        "role": "outer_valid",
        "query_id": query_id,
        "well_id": "target",
        "MD": 0.0,
        "X": x,
        "Y": y,
        "Z": -1000.0,
        "TVT_input": 1075.0,
        "tangent_x": 1.0,
        "tangent_y": 0.0,
        "base_absolute_s": 100.0,
        "base_rate": 1.0,
        "base_absolute_variance": 4.0,
        "base_rate_variance": 1.0,
        "base_support_ess": 20.0,
        "base_unique_wells": 20.0,
        "base_condition_number": 100.0,
        "base_surface_variance": 1.0,
        "surface_variance_reference": 10.0,
        "fallback_rate": 0.5,
        "base_path_s": 100.0,
    }
    for column in module["SIGNATURE_COLUMNS"]:
        row[column] = 0.0
    for formation_index, name in enumerate(module["FORMATION_NAMES"]):
        row[f"surface_{name}"] = (
            float(formation_index * 10) + 0.5 * x + 0.25 * y
        )
        row[f"surface_grad_x_{name}"] = 0.5
        row[f"surface_grad_y_{name}"] = 0.25
        row[f"surface_variance_{name}"] = 1.0
    return pd.Series(row)


def test_component_field_recovers_planar_absolute_and_vector_rate(
    module: dict[str, object],
    config: dict,
) -> None:
    query = _query_row(module)
    fitted = module["fit_component_field"](_planar_donors(module), query, config)
    assert fitted is not None
    expected_s = 100.0 + 2.0 * float(query["X"]) - 3.0 * float(query["Y"])
    # The preregistered trace-scaled ridge introduces a small, deterministic bias.
    assert float(fitted["field_absolute_s"]) == pytest.approx(expected_s, abs=2.0e-2)
    assert float(fitted["field_rate"]) == pytest.approx(2.0, abs=2.0e-3)
    assert int(fitted["unique_wells"]) >= 12
    assert float(fitted["support_ess"]) > 0


def test_posterior_has_fixed_base_floor_and_exact_no_component_fallback(
    module: dict[str, object],
    config: dict,
) -> None:
    first = _query_row(module, query_id="q0")
    second = _query_row(module, query_id="q1")
    second["MD"] = 64.0
    second["TVT_input"] = np.nan
    query = pd.DataFrame([first, second])
    fields = pd.DataFrame(
        [
            {
                "fold": 0,
                "query_id": "q0",
                "well_id": "target",
                "MD": 0.0,
                "component_id": "f00_c000000",
                "target_free_log_weight": -1.0,
                "field_absolute_s": 100.0,
                "field_rate": 1.5,
                "absolute_variance": 2.0,
                "rate_variance": 0.5,
                "support_ess": 16.0,
                "unique_wells": 16,
                "condition_number": 100.0,
                "surface_variance": 1.0,
            }
        ]
    )
    prefix = module["build_prefix_likelihood"](
        query, fields, prefix_scale_ft=2.0
    )
    posterior = module["build_domain_posterior"](query, fields, prefix, config)
    q0 = posterior.loc[posterior["query_id"].eq("q0")]
    q1 = posterior.loc[posterior["query_id"].eq("q1")]
    assert q0.loc[q0["domain"].eq("base"), "posterior_mass"].item() == 0.25
    assert q0["posterior_mass"].sum() == pytest.approx(1.0, abs=1.0e-12)
    assert q1["posterior_mass"].item() == 1.0
    mixed = module["marginalize_fields"](query, fields, posterior)
    q1_mixed = mixed.loc[mixed["query_id"].eq("q1")].iloc[0]
    assert not bool(q1_mixed["eligible"])
    assert q1_mixed["mixed_absolute_s"] == second["base_absolute_s"]
    assert q1_mixed["mixed_rate"] == second["base_rate"]


def test_path_solver_hard_constrains_all_known_prefix_rows(
    module: dict[str, object],
    config: dict,
) -> None:
    rows = []
    for index in range(5):
        md = float(index * 64)
        z = -1000.0
        true_s = 200.0 + 0.5 * md
        rows.append(
            {
                "query_id": f"q{index}",
                "MD": md,
                "Z": z,
                "TVT_input": true_s - z if index < 2 else np.nan,
                "calibrated_absolute_s": true_s,
                "mixed_absolute_variance": 1.0,
                "final_rate": 0.5,
                "mixed_rate_variance": 1.0,
                "base_path_s": true_s + 20.0,
                "eligible": True,
            }
        )
    block = pd.DataFrame(rows)
    solution, status = module["solve_path_for_well"](block, config)
    assert status == "piecewise"
    assert solution[0] == pytest.approx(200.0)
    assert solution[1] == pytest.approx(232.0)
    assert np.allclose(solution, 200.0 + 0.5 * block["MD"], atol=1.0e-7)


def test_path_solver_hard_constrains_ineligible_node_to_exp383(
    module: dict[str, object],
    config: dict,
) -> None:
    rows = []
    for index in range(5):
        md = float(index * 64)
        rows.append(
            {
                "query_id": f"q{index}",
                "MD": md,
                "Z": -1000.0,
                "TVT_input": 1200.0 if index == 0 else np.nan,
                "calibrated_absolute_s": 200.0 + 0.5 * md,
                "mixed_absolute_variance": 1.0,
                "final_rate": 0.5,
                "mixed_rate_variance": 1.0,
                "base_path_s": 200.0 + 0.25 * md,
                "eligible": index != 2,
            }
        )
    block = pd.DataFrame(rows)
    solution, status = module["solve_path_for_well"](block, config)
    assert status == "piecewise"
    assert solution[2] == pytest.approx(block.loc[2, "base_path_s"], abs=1.0e-12)


def test_truth_join_requires_target_free_freeze(
    module: dict[str, object],
) -> None:
    prediction = pd.DataFrame(
        {
            "fold": [0],
            "well_id": ["w0"],
            "row_idx": [1],
            "exp383_prediction": [10.0],
            "exp384_prediction": [9.0],
            "exp384_path_status": ["piecewise"],
        }
    )
    truth = pd.DataFrame(
        {
            "fold": [0],
            "well_id": ["w0"],
            "row_idx": [1],
            "tvt_true": [8.0],
            "exp383_prediction": [10.0],
            "distance_from_anchor": [1000.0],
            "hidden_like_spatial": [True],
            "hidden_like_typewell_purged": [True],
        }
    )
    with pytest.raises(ValueError, match="before target-free"):
        module["late_join_truth"](
            prediction, truth, {}, module["RoleReadLedger"]()
        )
    ledger = module["RoleReadLedger"]()
    joined = module["late_join_truth"](
        prediction, truth, {"prediction": "a" * 64}, ledger
    )
    assert len(joined) == 1
    assert ledger.truth_joined_after_freeze


def test_pre_freeze_oof_keys_reject_truth_columns(
    module: dict[str, object],
) -> None:
    safe = pd.DataFrame(
        {
            "fold": [0],
            "well_id": ["w0"],
            "row_idx": [1],
            "MD": [64.0],
            "Z": [-1000.0],
            "exp383_prediction": [1200.0],
        }
    )
    module["validate_truth_free_oof_keys"](safe, module["RoleReadLedger"]())
    with pytest.raises(ValueError, match="forbidden columns"):
        module["validate_truth_free_oof_keys"](
            safe.assign(tvt_true=1201.0), module["RoleReadLedger"]()
        )


def test_parent_manifest_is_fail_closed_until_real_exp383_sha_is_pinned(
    module: dict[str, object],
    config: dict,
) -> None:
    manifest = {
        "experiment": module["PARENT_EXPERIMENT"],
        "stage0": {"passed": True},
        "stage1": {"passed": True},
        "validation": {
            "score_rows": config["validation"]["expected_rows"],
            "wells": config["validation"]["expected_wells"],
            "folds": config["validation"]["expected_folds"],
        },
        "artifacts": {
            name: {
                "logical_content_sha256": "a" * 64,
                "schema_sha256": "b" * 64,
            }
            for name in set(config["data"]["parent_artifacts"]["files"]).difference(
                {"manifest"}
            )
        },
    }
    with pytest.raises(ValueError, match="not pinned"):
        module["validate_parent_manifest"](manifest, config)
    pinned = deepcopy(config)
    pinned["data"]["parent_artifacts"]["expected_manifest_logical_sha256"] = module[
        "parent_manifest_sha256"
    ](manifest)
    module["validate_parent_manifest"](manifest, pinned)
    failed = deepcopy(manifest)
    failed["stage1"]["passed"] = False
    with pytest.raises(ValueError, match="Stage 1 PASS"):
        module["validate_parent_manifest"](failed, pinned)


def test_inference_remains_fail_closed(config: dict) -> None:
    inference = load_namespace(INFERENCE_SOURCE)
    contract = inference["validate_disabled_inference"](config)
    assert contract["inference_enabled"] is False
    assert contract["submission_enabled"] is False
    with pytest.raises(RuntimeError, match="fail-closed"):
        inference["run_inference"]()

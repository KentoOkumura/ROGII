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
    "experiments/exp386_cycle_consistent_rgt_scenario_bank"
)
TRAIN_SOURCE = EXPERIMENT_DIR / (
    "exp386_cycle_consistent_rgt_scenario_bank_"
    "compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXPERIMENT_DIR / (
    "exp386_cycle_consistent_rgt_scenario_bank_"
    "compact_selfcontained_inference.py"
)
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"


def load_namespace(path: Path = TRAIN_SOURCE) -> dict[str, object]:
    previous = os.environ.get("EXP386_IMPORT_ONLY")
    os.environ["EXP386_IMPORT_ONLY"] = "1"
    try:
        return runpy.run_path(str(path))
    finally:
        if previous is None:
            os.environ.pop("EXP386_IMPORT_ONLY", None)
        else:
            os.environ["EXP386_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def module() -> dict[str, object]:
    return load_namespace()


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(value, dict)
    return value


def test_execution_contract_records_authorized_preflight(
    module: dict[str, object],
    config: dict,
) -> None:
    module["validate_execution_contract"](
        config,
        require_kaggle_authorization=False,
    )
    assert config["experiment"]["route"] == "pf_beam"
    assert config["experiment"]["status"] == (
        "kaggle_cpu_v1_completed_stage0_fail_closed"
    )
    assert config["lineage"]["parent"] == (
        "independent_topology_first_rgt_physics_family"
    )
    assert config["execution"]["implementation_authorized"] is True
    assert config["execution"]["implementation_approval_source"] == (
        "user_message_implement_exp386_2026_07_24"
    )
    assert config["execution"]["canonical_notebook_adoption_authorized"] is True
    assert config["execution"]["canonical_train_notebook_adopted"] is True
    assert config["execution"]["canonical_inference_notebook_adopted"] is True
    assert config["execution"]["kaggle_execution_authorized"] is True
    assert config["execution"]["full_run_authorized"] is True
    assert config["execution"]["full_run_effective_after"] == (
        "stage0_resource_preflight_pass"
    )
    assert config["execution"]["stage0_passed"] is False
    assert config["execution"]["full_run_gate_satisfied"] is False
    assert config["execution"]["full_run_executed"] is False
    assert config["execution"]["inference_enabled"] is False
    assert config["execution"]["submission_enabled"] is False
    assert config["runtime"]["scientific_variants"] == 1
    assert config["runtime"]["graph_fold_solves"] == 5
    assert config["runtime"]["target_well_path_solves"] == 773
    assert config["runtime"]["fitted_models"] == 0
    assert config["runtime"]["lightgbm_boosters"] == 0
    assert config["runtime"]["hmm_runs"] == 0
    assert config["runtime"]["pf_runs"] == 0
    assert config["runtime"]["beam_runs"] == 0
    module["validate_execution_contract"](
        config,
        require_kaggle_authorization=True,
    )
    broken = deepcopy(config)
    broken["runtime"]["graph_fold_solves"] = 6
    with pytest.raises(ValueError, match="graph_fold_solves"):
        module["validate_execution_contract"](
            broken,
            require_kaggle_authorization=False,
        )


def _rgt_source(module: dict[str, object], rows: int = 161) -> pd.DataFrame:
    md = np.arange(rows, dtype=float)
    z = -90.0 - 0.5 * md
    frame = pd.DataFrame(
        {
            "MD": md,
            "X": 100.0 + 0.2 * md,
            "Y": 200.0 + 0.1 * md,
            "Z": z,
            "TVT": 1000.0 + 0.55 * md,
            "GR": 80.0 + np.sin(md / 10.0),
            "well_id": "source",
            "row_idx": np.arange(rows, dtype=int),
            "fold": 0,
            "role": "outer_train",
        }
    )
    for index, name in enumerate(module["FORMATION_NAMES"]):
        frame[name] = -100.0 - 20.0 * index
    return frame


def test_ordered_rgt_uses_fixed_formation_order_and_extrapolates(
    module: dict[str, object],
) -> None:
    source = _rgt_source(module, rows=5)
    source["Z"] = [-90.0, -100.0, -110.0, -120.0, -210.0]
    converted = module["ordered_formation_rgt"](source)
    assert converted["rgt"].to_numpy() == pytest.approx(
        [-0.5, 0.0, 0.5, 1.0, 5.5]
    )
    assert converted["rgt_available"].all()
    broken = source.copy()
    broken.loc[0, "ASTNU"] = -80.0
    invalid = module["ordered_formation_rgt"](broken)
    assert not bool(invalid.loc[0, "rgt_available"])
    assert np.isnan(float(invalid.loc[0, "rgt"]))


def test_rgt_nodes_follow_fixed_64_32_contract(
    module: dict[str, object],
    config: dict,
) -> None:
    nodes, diagnostics = module["build_rgt_nodes_for_well"](
        _rgt_source(module),
        config,
    )
    assert len(nodes) >= 4
    assert diagnostics["rgt_source_coverage"] == pytest.approx(1.0)
    assert nodes["rgt_slope_per_md"].gt(0.0).all()
    assert nodes["structural_slope_per_md"].to_numpy() == pytest.approx(
        0.05,
        abs=1.0e-10,
    )
    assert nodes["rgt_stretch_ft_per_interval"].gt(0.0).all()
    assert nodes["node_id"].is_monotonic_increasing


def test_target_role_ledger_rejects_gr_formation_and_truth(
    module: dict[str, object],
) -> None:
    ledger = module["RoleReadLedger"]()
    ledger.record_target_safe(
        fold=0,
        well_id="target",
        columns=["MD", "X", "Y", "Z", "TVT_input"],
        rows=10,
    )
    with pytest.raises(ValueError, match="forbidden columns"):
        ledger.record_target_safe(
            fold=0,
            well_id="target",
            columns=["MD", "GR", "ANCC", "TVT"],
            rows=10,
        )
    with pytest.raises(ValueError, match="before target-free"):
        ledger.record_truth_late(fold=0, well_id="target", rows=10)
    with pytest.raises(ValueError, match="leaked"):
        ledger.validate_disjoint(["source", "target"], ["target"])


def test_cycle_solver_closes_consistent_triangle(
    module: dict[str, object],
    config: dict,
) -> None:
    rows = [
        ("a", "b", 1.0),
        ("b", "c", 2.0),
        ("a", "c", 3.0),
    ]
    edges = pd.DataFrame(
        [
            {
                "fold": 0,
                "source_well_id": source,
                "target_well_id": target,
                "edge_id": f"{source}_{target}",
                "raw_offset_interval": offset,
                "graph_edge_cost": 0.1,
            }
            for source, target, offset in rows
        ]
    )
    solved, potentials, manifest = module["solve_cycle_consistent_graph"](
        edges,
        ["a", "b", "c"],
        config,
    )
    assert manifest["fundamental_cycles"] == 1
    assert manifest["cycle_residual_p95"] == pytest.approx(0.0, abs=1.0e-8)
    assert solved["cycle_residual_abs_interval"].max() < 1.0e-8
    assert potentials["component_id"].nunique() == 1


def test_correspondence_edges_are_cross_well_stable_and_capped(
    module: dict[str, object],
    config: dict,
) -> None:
    parts: list[pd.DataFrame] = []
    for well_index in range(8):
        source = _rgt_source(module)
        source["well_id"] = f"w{well_index:02d}"
        source["X"] = source["X"] + 100.0 * well_index
        source["TVT"] = source["TVT"] + 0.001 * well_index * source["MD"]
        nodes, _ = module["build_rgt_nodes_for_well"](source, config)
        parts.append(nodes)
    nodes = pd.concat(parts, ignore_index=True)
    edges, profiles = module["build_correspondence_edges"](nodes, config)
    assert len(profiles) == 8
    assert not edges.empty
    assert edges["source_well_id"].ne(edges["target_well_id"]).all()
    assert edges["match_count"].le(
        config["rgt"]["graph_edges"]["maximum_nodes_per_source_well"]
    ).all()
    assert edges["edge_id"].is_unique
    assert np.isfinite(
        edges[
            [
                "raw_offset_interval",
                "graph_edge_cost",
                "mean_rgt_mismatch",
            ]
        ].to_numpy(dtype=float)
    ).all()


def test_k_shortest_paths_are_simple_stable_and_cost_ordered(
    module: dict[str, object],
) -> None:
    adjacency = {
        "s": [("a", 1.0), ("b", 1.0)],
        "a": [("g", 1.0), ("b", 0.25)],
        "b": [("g", 1.0), ("a", 0.25)],
        "g": [],
    }
    first = module["deterministic_k_shortest_paths"](adjacency, "s", "g", 4)
    second = module["deterministic_k_shortest_paths"](adjacency, "s", "g", 4)
    assert first == second
    assert [cost for cost, _ in first] == sorted(cost for cost, _ in first)
    assert first[0][1] == ("s", "a", "g")
    assert all(len(path) == len(set(path)) for _, path in first)


def _scenario_graph(module: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    for well_index in range(10):
        well_id = f"d{well_index:02d}"
        profile_rows.append(
            {
                "fold": 0,
                "well_id": well_id,
                "centroid_x": float(well_index * 10),
                "centroid_y": 0.0,
                "direction_x": 1.0,
                "direction_y": 0.0,
                "direction_z": 0.0,
                "node_count": 5,
            }
        )
        for node_index in range(5):
            node_rows.append(
                {
                    "fold": 0,
                    "well_id": well_id,
                    "node_id": f"{well_id}_n{node_index}",
                    "MD": float(node_index * 50),
                    "X": float(well_index * 10 + node_index),
                    "Y": 0.0,
                    "rgt_median": float(node_index),
                    "rgt_slope_per_md": 0.01 + 0.0005 * well_index,
                    "structural_slope_per_md": 0.03 + 0.006 * well_index,
                    "rgt_stretch_ft_per_interval": 3.0 + 0.5 * well_index,
                    "reference_gr_median": 70.0 + well_index + node_index,
                    "reference_gr_diff_median": 0.1 * well_index,
                }
            )
    return pd.DataFrame(node_rows), pd.DataFrame(profile_rows)


def _target_safe(rows: int = 241) -> pd.DataFrame:
    md = np.arange(rows, dtype=float)
    z = -1000.0 - 0.8 * md
    tvt = 1100.0 + 0.85 * md
    tvt_input = np.where(md <= 120.0, tvt, np.nan)
    return pd.DataFrame(
        {
            "MD": md,
            "X": 5.0 + 0.1 * md,
            "Y": np.zeros(rows),
            "Z": z,
            "TVT_input": tvt_input,
            "well_id": "target",
            "row_idx": np.arange(rows, dtype=int),
            "fold": 0,
            "role": "outer_valid",
        }
    )


def test_scenario_bank_is_target_safe_anchored_and_diverse(
    module: dict[str, object],
    config: dict,
) -> None:
    nodes, profiles = _scenario_graph(module)
    edges = pd.DataFrame(
        [
            {
                "fold": 0,
                "source_well_id": f"d{left:02d}",
                "target_well_id": f"d{right:02d}",
                "edge_id": f"d{left:02d}_d{right:02d}",
                "graph_edge_cost": 0.1 + 0.01 * abs(right - left),
                "cycle_residual_interval": 0.01,
                "cycle_residual_abs_interval": 0.01,
            }
            for left in range(10)
            for right in range(left + 1, 10)
        ]
    )
    target = _target_safe()
    paths, references, diagnostics = module["generate_scenario_bank_for_well"](
        target,
        nodes,
        edges,
        profiles,
        config,
    )
    assert diagnostics["scenario_count"] >= 8
    assert diagnostics["scenario_count"] <= 32
    assert diagnostics["finite_paths"] is True
    assert "GR" not in paths.columns
    assert "TVT" not in paths.columns
    assert references["reference_gr"].notna().all()
    anchor = int(np.flatnonzero(np.isfinite(target["TVT_input"]))[-1])
    anchor_md = float(target.loc[anchor, "MD"])
    anchored = paths.loc[paths["MD"].eq(anchor_md)]
    expected = float(target.loc[anchor, "TVT_input"])
    assert anchored["tvt_path"].to_numpy() == pytest.approx(expected, abs=1.0e-10)
    vectors = [
        group.sort_values("control_index")["tvt_path"].to_numpy(dtype=float)
        for _, group in paths.groupby("scenario_rank", sort=True)
    ]
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            assert module["rmse"](vectors[left], vectors[right]) >= 0.5


def test_target_safe_guard_fails_if_gr_is_attached(
    module: dict[str, object],
    config: dict,
) -> None:
    target = _target_safe()
    target["GR"] = 80.0
    with pytest.raises(ValueError, match="forbidden"):
        module["build_target_control_grid"](target, config)


def test_freeze_must_precede_truth_join(module: dict[str, object]) -> None:
    ledger = module["RoleReadLedger"]()
    hashes = module["freeze_target_free_outputs"](
        {
            "fold_manifest": pd.DataFrame(
                [{"fold": 0, "role": "outer_valid", "well_id": "w"}]
            ),
            "role_read_ledger": ledger.as_frame(),
            "rgt_nodes": pd.DataFrame(),
            "graph_edges": pd.DataFrame(),
            "target_paths": pd.DataFrame(),
            "reference_gr_templates": pd.DataFrame(),
        },
        [{"fold": 0, "cycle_residual_p95": 0.0}],
    )
    assert hashes
    ledger.mark_frozen(hashes)
    ledger.record_truth_late(fold=0, well_id="target", rows=10)
    assert ledger.summary()["truth_joined_after_freeze"] is True


def test_fail_closed_inference_contract(config: dict) -> None:
    inference = load_namespace(INFERENCE_SOURCE)
    contract = inference["validate_disabled_inference"](config)
    assert contract["inference_enabled"] is False
    assert contract["submission_enabled"] is False
    with pytest.raises(RuntimeError, match="fail-closed"):
        inference["run_inference"]()

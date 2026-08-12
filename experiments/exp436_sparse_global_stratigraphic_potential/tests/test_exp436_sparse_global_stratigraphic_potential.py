from __future__ import annotations

import inspect
import os
import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = (
    ROOT
    / "experiments"
    / "exp436_sparse_global_stratigraphic_potential"
)
TRAIN_SOURCE = (
    EXP_DIR
    / "exp436_sparse_global_stratigraphic_potential_compact_selfcontained_train.py"
)
CONFIG_PATH = EXP_DIR / "config.yaml"


def load_namespace() -> dict[str, object]:
    previous = os.environ.get("EXP436_IMPORT_ONLY")
    os.environ["EXP436_IMPORT_ONLY"] = "1"
    try:
        return runpy.run_path(str(TRAIN_SOURCE))
    finally:
        if previous is None:
            os.environ.pop("EXP436_IMPORT_ONLY", None)
        else:
            os.environ["EXP436_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def module() -> dict[str, object]:
    return load_namespace()


@pytest.fixture()
def config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    assert isinstance(value, dict)
    return value


def synthetic_nodes(
    module: dict[str, object],
    *,
    formation: str = "ANCC",
    fold: int = 0,
    constant_u: float = 12345.0,
) -> pd.DataFrame:
    x, y = np.meshgrid(
        np.linspace(-750.0, 750.0, 8),
        np.linspace(-600.0, 600.0, 5),
    )
    rows = len(x.ravel())
    return pd.DataFrame(
        {
            "fold": fold,
            "formation": formation,
            "formation_index": module["FORMATION_INDEX"][formation],
            "well_id": [f"source_{index:03d}" for index in range(rows)],
            "contact_md": 1000.0 + np.arange(rows),
            "contact_x": x.ravel(),
            "contact_y": y.ravel(),
            "contact_z": 8000.0,
            "contact_tvt": constant_u - 8000.0,
            "left_row_idx": 10.0,
            "fraction": 0.5,
            "u_contact": constant_u,
        }
    )


def test_stage0_is_complete_and_all_future_run_stages_are_locked(
    module: dict[str, object],
    config: dict,
) -> None:
    observed = module["validate_execution_contract"](
        config,
        require_stage0_authorization=False,
    )
    assert observed["scientific_candidates"] == 1
    assert observed["report_only_single_formation_paths"] == 6
    assert observed["global_surface_fits"] == 30
    assert observed["maximum_sparse_solves_including_irls"] == 180
    assert observed["boosters"] == 0
    assert observed["hmm_runs"] == 0
    assert observed["pf_runs"] == 0
    assert observed["beam_runs"] == 0
    assert config["execution"]["selected_stage"] == "stage0"
    assert config["implementation"]["enabled"] is True
    assert config["experiment"]["status"] == "stage0_fail_closed"
    assert config["implementation"]["stage0_completed"] is True
    assert config["authorization"]["canonical_train_notebook_adopted"] is True
    assert config["authorization"]["stage0_run_authorized"] is False
    assert config["authorization"]["stage1_run_authorized"] is False
    assert config["authorization"]["stage2_run_authorized"] is False
    assert (
        config["gates"]["stage1_prefix_rolling_origin"]["null"]
        == "constant_U_from_pseudo_anchor"
    )
    with pytest.raises(RuntimeError, match="Stage 0 execution remains locked"):
        module["validate_execution_contract"](
            config,
            require_stage0_authorization=True,
        )


def test_stage0_surface_fit_records_insufficient_support_as_fail_closed(
    module: dict[str, object],
    config: dict,
) -> None:
    nodes = synthetic_nodes(module).iloc[:5].copy()
    ancc_edges = module["build_graph_edges"](
        nodes,
        nearest_wells=8,
        maximum_distance_ft=4000.0,
        bandwidth_ft=1000.0,
    )
    edges_by_key = {
        (0, formation): (
            ancc_edges
            if formation == "ANCC"
            else ancc_edges.iloc[0:0].copy()
        )
        for formation in module["FORMATION_NAMES"]
    }
    fields, manifest = module["fit_fold_fields_fail_closed"](
        fold=0,
        nodes=nodes,
        edges_by_key=edges_by_key,
        config=config,
    )
    assert fields == {}
    assert len(manifest) == 6
    assert not manifest["accepted"].any()
    assert manifest["sparse_solves"].eq(0).all()
    assert manifest["failure_reason"].str.contains("below 32").all()


def test_first_contact_is_first_exact_or_linearly_interpolated(
    module: dict[str, object],
) -> None:
    md = np.array([0.0, 10.0, 20.0, 30.0])
    residual = np.array([2.0, 1.0, -1.0, 1.0])
    tvt = np.array([100.0, 110.0, 120.0, 130.0])
    crossing = module["first_crossing"](md, residual, tvt=tvt)
    assert crossing is not None
    assert crossing["contact_md"] == pytest.approx(15.0)
    assert crossing["contact_tvt"] == pytest.approx(115.0)
    assert crossing["left_row_idx"] == 1.0

    exact = module["first_crossing"](
        md,
        np.array([1.0, 0.0, -1.0, 1.0]),
    )
    assert exact is not None
    assert exact["contact_md"] == pytest.approx(10.0)
    assert exact["fraction"] == 0.0


def test_contact_extraction_uses_u_equals_tvt_plus_z(
    module: dict[str, object],
) -> None:
    md = np.array([0.0, 10.0, 20.0])
    z = np.array([1000.0, 990.0, 980.0])
    frame = pd.DataFrame(
        {
            "MD": md,
            "X": [0.0, 1.0, 2.0],
            "Y": [2.0, 3.0, 4.0],
            "Z": z,
            "TVT": [5000.0, 5010.0, 5020.0],
        }
    )
    for index, formation in enumerate(module["FORMATION_NAMES"]):
        residual = np.array([1.0, -1.0, -2.0])
        frame[formation] = z - residual - index
    nodes = module["extract_first_contact_nodes"](
        frame,
        fold=2,
        well_id="well_a",
    )
    ancc = nodes.loc[nodes["formation"].eq("ANCC")].iloc[0]
    assert ancc["contact_md"] == pytest.approx(5.0)
    assert ancc["contact_z"] == pytest.approx(995.0)
    assert ancc["contact_tvt"] == pytest.approx(5005.0)
    assert ancc["u_contact"] == pytest.approx(6000.0)


def test_graph_is_same_formation_stable_unique_and_distance_bounded(
    module: dict[str, object],
) -> None:
    nodes = synthetic_nodes(module)
    edges = module["build_graph_edges"](
        nodes,
        nearest_wells=8,
        maximum_distance_ft=4000.0,
        bandwidth_ft=1000.0,
    )
    assert len(edges) > 0
    assert edges["formation"].eq("ANCC").all()
    assert edges["distance_ft"].le(4000.0).all()
    assert edges["left_node"].lt(edges["right_node"]).all()
    assert not edges.duplicated(
        ["fold", "formation_index", "left_node", "right_node"]
    ).any()
    repeated = module["build_graph_edges"](
        nodes.sample(frac=1.0, random_state=7),
        nearest_wells=8,
        maximum_distance_ft=4000.0,
        bandwidth_ft=1000.0,
    )
    pd.testing.assert_frame_equal(edges, repeated)


def test_sparse_huber_solver_and_query_recover_a_smooth_field(
    module: dict[str, object],
    config: dict,
) -> None:
    nodes = synthetic_nodes(module)
    expected_node_u = (
        nodes["u_contact"].to_numpy(np.float64)
        + 0.01 * nodes["contact_x"].to_numpy(np.float64)
        + 0.005 * nodes["contact_y"].to_numpy(np.float64)
    )
    nodes["u_contact"] = expected_node_u
    nodes["contact_tvt"] = expected_node_u - nodes["contact_z"]
    edges = module["build_graph_edges"](
        nodes,
        nearest_wells=8,
        maximum_distance_ft=4000.0,
        bandwidth_ft=1000.0,
    )
    surface = module["solve_sparse_potential"](nodes, edges, config)
    assert surface.diagnostics["accepted"] is True
    assert surface.diagnostics["sparse_solves"] == 6
    assert surface.diagnostics["components"] == 1
    assert np.isfinite(surface.solved_u).all()
    assert np.sqrt(np.mean((surface.solved_u - expected_node_u) ** 2)) < 4.0
    prediction, support = module["query_sparse_potential"](
        surface,
        np.array([[0.0, 0.0], [100.0, 100.0]]),
        config,
    )
    np.testing.assert_allclose(
        prediction,
        np.array([12345.0, 12346.5]),
        rtol=0.0,
        atol=1.0,
    )
    assert support["supported"].all()
    assert support["unique_source_wells"].ge(8).all()


def test_fixed_support_equal_weight_path_uses_only_anchor_difference(
    module: dict[str, object],
    config: dict,
) -> None:
    fields = {}
    for index, formation in enumerate(module["FORMATION_NAMES"]):
        nodes = synthetic_nodes(
            module,
            formation=formation,
            constant_u=12000.0 + index * 100.0,
        )
        edges = module["build_graph_edges"](
            nodes,
            nearest_wells=8,
            maximum_distance_ft=4000.0,
            bandwidth_ft=1000.0,
        )
        fields[formation] = module["SparsePotentialSurface"](
            fold=0,
            formation=formation,
            formation_index=index,
            nodes=nodes.sort_values(
                ["well_id", "contact_md"],
                kind="mergesort",
            ).reset_index(drop=True),
            edges=edges,
            solved_u=np.full(len(nodes), 12000.0 + index * 100.0),
            scale=1.0,
            diagnostics={"accepted": True},
        )
    well = module["TargetWell"](
        well_id="target",
        fold=0,
        md=np.array([0.0, 64.0, 128.0]),
        x=np.array([0.0, 50.0, 100.0]),
        y=np.array([0.0, 0.0, 0.0]),
        z=np.array([1000.0, 1001.5, 1004.0]),
        tvt_input=np.array([5000.0, np.nan, np.nan]),
        suffix_row_idx=np.array([1, 2], dtype=np.int32),
        suffix_offset=np.array([0, 1], dtype=np.int32),
    )
    prediction, support = module["predict_interval"](
        well,
        fields,
        start_row=0,
        row_indices=np.array([1, 2]),
        config=config,
        purpose="test",
    )
    np.testing.assert_allclose(
        prediction["tvt_pred_exp436"],
        np.array([4998.5, 4996.0]),
        rtol=0.0,
        atol=1.0e-10,
    )
    assert prediction["supported_formation_count"].eq(6).all()
    assert support["fixed_supported_formation"].all()


def test_target_role_guard_blocks_formation_and_truth_before_freeze(
    module: dict[str, object],
) -> None:
    ledger = module["RoleReadLedger"]()
    ledger.record_target_safe(
        fold=0,
        well_id="target",
        columns=module["TARGET_SAFE_COLUMNS"],
        rows=10,
    )
    with pytest.raises(ValueError, match="forbidden columns"):
        ledger.record_target_safe(
            fold=0,
            well_id="target",
            columns=("MD", "X", "Y", "Z", "TVT_input", "ANCC"),
            rows=10,
        )
    with pytest.raises(RuntimeError, match="before prediction freeze"):
        ledger.record_truth_late(
            source="exp226",
            columns=module["EXP226_LATE_COLUMNS"],
            rows=10,
        )
    ledger.freeze()
    ledger.record_truth_late(
        source="exp226",
        columns=module["EXP226_LATE_COLUMNS"],
        rows=10,
    )
    assert ledger.frame()["after_freeze"].sum() == 1


def test_stage1_gate_is_an_and_gate(
    module: dict[str, object],
    config: dict,
) -> None:
    adjusted = deepcopy(config)
    adjusted["validation"]["expected_wells"] = 5
    rows = []
    for fold in range(5):
        for offset in range(2):
            truth = 100.0 + fold + offset
            rows.append(
                {
                    "well_id": f"well_{fold}",
                    "fold": fold,
                    "tvt_visible": truth,
                    "tvt_pred_exp436": truth,
                    "tvt_null_constant_u": truth + 1.0,
                    "endpoint": offset == 1,
                }
            )
    decision = module["evaluate_stage1_rolling_origin"](
        pd.DataFrame(rows),
        adjusted,
    )
    assert decision["passed"] is True
    assert all(decision["checks"].values())
    broken = pd.DataFrame(rows)
    broken["tvt_pred_exp436"] = broken["tvt_null_constant_u"]
    failed = module["evaluate_stage1_rolling_origin"](broken, adjusted)
    assert failed["passed"] is False


def test_stage2_gate_requires_all_preregistered_scopes(
    module: dict[str, object],
    config: dict,
) -> None:
    rows = []
    hidden = []
    for well_index in range(10):
        well = f"well_{well_index}"
        hidden.append(
            {
                "well_id": well,
                "verification_like_spatial_role": "valid",
                "verification_like_typewell_purged_role": "valid",
            }
        )
        for offset, distance in enumerate((100.0, 500.0, 1200.0)):
            angle = 2.0 * np.pi * (well_index * 3 + offset) / 30.0
            rows.append(
                {
                    "well_id": well,
                    "fold": well_index % 5,
                    "row_idx": offset,
                    "distance_from_anchor": distance,
                    "tvt_true": 0.0,
                    "tvt_pred_exp436": 0.1 * np.sin(angle),
                    "tvt_pred": 1.0 * np.cos(angle),
                }
            )
    _, by_well, decision = module["build_stage2_readout"](
        pd.DataFrame(rows),
        pd.DataFrame(hidden),
        config,
    )
    assert decision["passed"] is True
    assert len(by_well) == 10
    broken = deepcopy(config)
    broken["gates"]["stage2_truth_late_oof"][
        "prediction_correlation_vs_control_max"
    ] = -1.0
    _, _, failed = module["build_stage2_readout"](
        pd.DataFrame(rows),
        pd.DataFrame(hidden),
        broken,
    )
    assert failed["passed"] is False
    assert failed["checks"]["prediction_correlation"] is False


def test_compact_source_is_notebook_safe_and_target_read_is_allowlisted(
    module: dict[str, object],
) -> None:
    source = TRAIN_SOURCE.read_text()
    assert "__file__" not in source
    loader_source = inspect.getsource(module["load_target_safe"])
    assert "usecols=list(TARGET_SAFE_COLUMNS)" in loader_source
    assert "TVT_suffix" not in loader_source
    assert "GR" not in loader_source

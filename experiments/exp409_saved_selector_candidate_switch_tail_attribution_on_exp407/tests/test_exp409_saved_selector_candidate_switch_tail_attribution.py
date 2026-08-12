from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP_NAME = "exp409_saved_selector_candidate_switch_tail_attribution_on_exp407"
EXP_DIR = ROOT / "experiments" / EXP_NAME
SOURCE_PATH = EXP_DIR / f"{EXP_NAME}_compact_selfcontained_train.py"
sys.path.insert(0, str(EXP_DIR))

from exp409_saved_selector_candidate_switch_tail_attribution_on_exp407_compact_selfcontained_train import (  # noqa: E402,E501
    aggregate_attribution,
    build_selection_freeze_batch,
    build_truth_attribution_batch,
    combine_partial_aggregates,
    evaluate_tail_consistency_gate,
    select_truth_free_batch,
    write_selection_freeze,
)

CANDIDATES = ["a", "b", "fixed"]
SELECTABLE = ["a", "b"]


def make_long(
    *,
    selected: list[str],
    candidate_values: dict[str, list[float]] | None = None,
    include_truth: bool = False,
    actual_errors: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = candidate_values or {
        "a": [100.0, 101.0, 102.0, 103.0, 104.0],
        "b": [110.0, 111.0, 112.0, 113.0, 114.0],
        "fixed": [105.0, 106.0, 107.0, 108.0, 109.0],
    }
    errors = actual_errors or {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [4.0, 3.0, 2.0, 1.0, 6.0],
        "fixed": [2.0, 2.0, 2.0, 2.0, 2.0],
    }
    for base_index in range(5):
        for candidate in CANDIDATES:
            row: dict[str, object] = {
                "id": f"id_{base_index}",
                "well": f"well_{base_index}",
                "well_row_idx": base_index,
                "outer_fold": base_index,
                "md_since": [100.0, 300.0, 700.0, 1200.0, 1500.0][base_index],
                "candidate_id": candidate,
                "candidate_tvt": values[candidate][base_index],
                "candidate_available": True,
                "pred_abs_error": (
                    0.1
                    if candidate == selected[base_index]
                    else (0.2 if candidate in SELECTABLE else 0.05)
                ),
                "feature_schema_sha": "schema",
                "candidate_contract_sha": "contract",
                "model_fold": base_index,
            }
            if include_truth:
                row["actual_abs_error"] = errors[candidate][base_index]
            rows.append(row)
    return pd.DataFrame(rows)


def hidden_assignment() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": [f"well_{index}" for index in range(5)],
            "verification_like_spatial_role": ["valid"] * 5,
            "verification_like_typewell_purged_role": ["valid"] * 5,
        }
    )


def test_static_contract_is_approved_kaggle_run_and_zero_generation() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    assert config["experiment"]["route"] == "ml_model"
    assert config["implementation"]["self_contained_notebook"] is True
    assert config["implementation"]["canonical_notebook_adopted"] is True
    assert config["execution"]["run_approved"] is True
    assert config["execution"]["kaggle_run_approved"] is True
    assert config["execution"]["local_notebook_execution_approved"] is False
    assert config["execution"]["private_parent_oof_input_created"] is True
    for key in [
        "variants",
        "models",
        "folds_for_fitting",
        "boosters",
        "prediction_rows_generated",
        "control_retraining",
        "pf_runs",
        "hmm_runs",
        "beam_runs",
        "inference_runs",
        "submissions",
    ]:
        assert config["execution"][key] == 0
    assert config["validation"]["preregistered_worst_well"] == "52f1e77a"
    assert (
        config["data"]["parent_candidate_score_oof"]["sha256"]
        == "9a91b62599278d4e56d57074df4725d4a09391460458b8eccc02dd50af34d48a"
    )
    assert (
        config["data"]["exp407_candidate_score_oof"]["sha256"]
        == "d993b806d92c2462c1509f110669b272b27d48806c0280a2cf54e87c7f32f1e8"
    )


def test_truth_free_selection_rejects_actual_error_column() -> None:
    frame = make_long(selected=["a"] * 5, include_truth=True)
    with pytest.raises(ValueError, match="truth columns are forbidden"):
        select_truth_free_batch(frame, CANDIDATES, SELECTABLE, source="synthetic")


def test_selection_uses_frozen_candidate_order_for_ties() -> None:
    frame = make_long(selected=["a"] * 5)
    for base_index in range(5):
        mask = frame["id"].eq(f"id_{base_index}") & frame["candidate_id"].isin(["a", "b"])
        frame.loc[mask, "pred_abs_error"] = 0.25
    selected, _ = select_truth_free_batch(
        frame, CANDIDATES, SELECTABLE, source="synthetic"
    )
    assert selected["selected_candidate"].tolist() == ["a"] * 5


def test_freeze_checks_keys_values_and_target_free_scopes() -> None:
    parent = make_long(selected=["a", "a", "a", "a", "a"])
    exp407 = make_long(selected=["b", "a", "b", "a", "b"])
    freeze = build_selection_freeze_batch(
        parent,
        exp407,
        hidden_assignment(),
        CANDIDATES,
        SELECTABLE,
        expected_feature_schema_sha256="schema",
        expected_candidate_contract_sha256="contract",
    )
    assert freeze["switched"].tolist() == [True, False, True, False, True]
    assert freeze["distance_bucket"].tolist() == [
        "near_0_250",
        "250_500",
        "500_1000",
        "1000_plus",
        "1000_plus",
    ]
    assert freeze["transition_id"].iloc[0] == "a -> b"
    changed = exp407.copy()
    changed.loc[
        changed["id"].eq("id_2") & changed["candidate_id"].eq("b"),
        "candidate_tvt",
    ] += 0.1
    with pytest.raises(ValueError, match="candidate values differ"):
        build_selection_freeze_batch(
            parent,
            changed,
            hidden_assignment(),
            CANDIDATES,
            SELECTABLE,
        )


def test_truth_join_uses_only_frozen_selected_candidate_errors() -> None:
    parent_no_truth = make_long(selected=["a"] * 5)
    exp407_no_truth = make_long(selected=["b", "a", "b", "a", "b"])
    freeze = build_selection_freeze_batch(
        parent_no_truth,
        exp407_no_truth,
        hidden_assignment(),
        CANDIDATES,
        SELECTABLE,
    )
    parent_truth = make_long(selected=["a"] * 5, include_truth=True)
    exp407_truth = make_long(
        selected=["b", "a", "b", "a", "b"], include_truth=True
    )
    attribution = build_truth_attribution_batch(
        freeze, parent_truth, exp407_truth, CANDIDATES
    )
    expected_parent = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_exp407 = np.asarray([4.0, 2.0, 2.0, 4.0, 6.0])
    np.testing.assert_allclose(attribution["parent_abs_error"], expected_parent)
    np.testing.assert_allclose(attribution["exp407_abs_error"], expected_exp407)
    np.testing.assert_allclose(
        attribution["delta_sse"],
        np.square(expected_exp407) - np.square(expected_parent),
    )


def test_additive_attribution_reconstructs_total_sse_delta() -> None:
    parent = make_long(selected=["a"] * 5)
    exp407 = make_long(selected=["b", "a", "b", "a", "b"])
    freeze = build_selection_freeze_batch(
        parent, exp407, hidden_assignment(), CANDIDATES, SELECTABLE
    )
    truth = make_long(selected=["a"] * 5, include_truth=True)
    attribution = build_truth_attribution_batch(
        freeze, truth, truth.copy(), CANDIDATES
    )
    partial = aggregate_attribution(
        attribution,
        [
            "parent_selected_candidate",
            "exp407_selected_candidate",
            "transition_id",
        ],
    )
    combined = combine_partial_aggregates(
        [partial],
        [
            "parent_selected_candidate",
            "exp407_selected_candidate",
            "transition_id",
        ],
    )
    assert combined["rows"].sum() == len(attribution)
    assert combined["delta_sse"].sum() == pytest.approx(
        attribution["delta_sse"].sum()
    )
    assert combined["positive_excess_sse_share"].sum() == pytest.approx(1.0)


def test_physical_freeze_is_written_before_truth_read(
    tmp_path: Path,
) -> None:
    parent_path = tmp_path / "parent.parquet"
    exp407_path = tmp_path / "exp407.parquet"
    freeze_path = tmp_path / "selection_freeze.parquet"
    make_long(selected=["a"] * 5).to_parquet(parent_path, index=False)
    make_long(selected=["b", "a", "b", "a", "b"]).to_parquet(
        exp407_path, index=False
    )
    manifest = write_selection_freeze(
        parent_path=parent_path,
        exp407_path=exp407_path,
        hidden_assignment=hidden_assignment(),
        output_path=freeze_path,
        candidate_order=CANDIDATES,
        selectable_candidates=SELECTABLE,
        batch_base_rows=2,
        expected_feature_schema_sha256="schema",
        expected_candidate_contract_sha256="contract",
        expected_base_rows=5,
        expected_long_rows=15,
    )
    assert freeze_path.exists()
    assert len(pd.read_parquet(freeze_path)) == 5
    assert manifest["truth_columns_read"] == []
    assert manifest["forbidden_truth_read_count"] == 0
    assert len(manifest["selection_freeze_sha256"]) == 64


def metric_table(
    *,
    scopes: list[str],
    top_transition_by_fold: list[str],
    scope_column: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scope in scopes:
        for fold, top_transition in enumerate(top_transition_by_fold):
            for transition, delta in [
                (top_transition, 10.0),
                ("b -> a" if top_transition == "a -> b" else "a -> b", 2.0),
            ]:
                parent, exp407 = transition.split(" -> ")
                records.append(
                    {
                        scope_column: scope,
                        "outer_fold": fold,
                        "parent_selected_candidate": parent,
                        "exp407_selected_candidate": exp407,
                        "transition_id": transition,
                        "rows": 10,
                        "delta_sse": delta,
                        "positive_excess_sse": delta,
                        "positive_excess_sse_share": delta / 12.0,
                    }
                )
    return pd.DataFrame(records)


def test_tail_consistency_gate_requires_same_4_of_5_and_worst_well() -> None:
    top = ["a -> b", "a -> b", "a -> b", "a -> b", "b -> a"]
    distance = metric_table(
        scopes=["1000_plus"], top_transition_by_fold=top, scope_column="distance_bucket"
    )
    hidden = metric_table(
        scopes=["hidden_like_spatial", "hidden_like_typewell_purged"],
        top_transition_by_fold=top,
        scope_column="scope",
    )
    by_well = pd.DataFrame(
        [
            {
                "well": "52f1e77a",
                "outer_fold": 2,
                "parent_selected_candidate": "a",
                "exp407_selected_candidate": "b",
                "transition_id": "a -> b",
                "rows": 20,
                "delta_sse": 20.0,
                "positive_excess_sse": 20.0,
            },
            {
                "well": "52f1e77a",
                "outer_fold": 2,
                "parent_selected_candidate": "b",
                "exp407_selected_candidate": "a",
                "transition_id": "b -> a",
                "rows": 20,
                "delta_sse": 5.0,
                "positive_excess_sse": 5.0,
            },
        ]
    )
    gate, rank1 = evaluate_tail_consistency_gate(
        distance,
        hidden,
        by_well,
        expected_folds=[0, 1, 2, 3, 4],
        minimum_rank1_folds=4,
        preregistered_worst_well="52f1e77a",
    )
    assert len(rank1) == 15
    assert gate["passed"] is True
    assert gate["cause_transition"] == "a -> b"
    assert gate["decision"] == "candidate_switch_tail_cause_supported"

    diffuse_hidden = hidden.copy()
    for fold in [2, 3]:
        mask = diffuse_hidden["outer_fold"].eq(fold)
        diffuse_hidden.loc[
            mask & diffuse_hidden["transition_id"].eq("a -> b"), "delta_sse"
        ] = 1.0
        diffuse_hidden.loc[
            mask & diffuse_hidden["transition_id"].eq("b -> a"), "delta_sse"
        ] = 11.0
    diffuse_gate, _ = evaluate_tail_consistency_gate(
        distance,
        diffuse_hidden,
        by_well,
        expected_folds=[0, 1, 2, 3, 4],
        minimum_rank1_folds=4,
        preregistered_worst_well="52f1e77a",
    )
    assert diffuse_gate["passed"] is False
    assert (
        diffuse_gate["decision"]
        == "diffuse_or_nonreproducible_candidate_switch_cause"
    )


def test_jupytext_source_is_self_contained_and_metrics_json_is_valid() -> None:
    source = SOURCE_PATH.read_text()
    assert source.count("# %% [markdown]") >= 8
    assert "from settings import" not in source
    assert f"from {EXP_NAME}" not in source
    assert "__file__" not in source
    metrics = json.loads((EXP_DIR / "metrics.json").read_text())
    assert metrics["models"] == 0
    assert metrics["boosters"] == 0
    assert metrics["kaggle_execution_performed"] is True
    assert metrics["kaggle_kernel"]["status"] == "complete"
    assert metrics["diagnostic_result"]["gate_passed"] is False

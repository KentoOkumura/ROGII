from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SOURCE = (
    HERE
    / "exp506_exp490_mean_reversion_correction_blend_on_exp413_compact_selfcontained_train.py"
)


def load_module():
    root = HERE.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("exp506_train", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    wells = [f"w{index}" for index in range(10)]
    rows: list[dict[str, object]] = []
    correction_rows: list[dict[str, object]] = []
    for well_index, well in enumerate(wells):
        fold = well_index % 5
        for suffix_offset, row_idx in enumerate((10, 11)):
            md_since = float((100.0, 500.0, 1500.0)[(well_index + suffix_offset) % 3])
            anchor_prediction = 100.0 + well_index + suffix_offset
            correction = 2.0 + 0.1 * well_index + 0.2 * suffix_offset
            actual = anchor_prediction + 0.05 * correction
            rows.append(
                {
                    "id": f"{well}_{row_idx}",
                    "well": well,
                    "md_since": md_since,
                    "outer_fold": fold,
                    "actual_tvt": actual,
                    "scale5_x1p0_full_replacement__lgb_mean__pred_tvt": anchor_prediction,
                }
            )
            exp357 = 80.0 + well_index
            correction_rows.append(
                {
                    "well": well,
                    "row_idx": row_idx,
                    "suffix_offset": suffix_offset,
                    "md_since": md_since,
                    "geometry_mean_reverting_hmm": exp357 + correction,
                    "exp357_parent_prediction": exp357,
                    "fold": fold,
                    "true_tvt_readout_only": actual,
                    "candidate_error": exp357 + correction - actual,
                    "episode_id": 99,
                    "scientific_gate": "forbidden_before_freeze",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(correction_rows)


def test_frozen_anchor_resolution_and_zero_compute_contract() -> None:
    module = load_module()
    module.validate_exp506_contract(module.CONFIG, module.CONTRACT)
    resolution = module.resolved_anchor_manifest(module.CONFIG)
    assert resolution["exp497"]["terminal_status"] == "completed_gate_failed_closed"
    assert resolution["exp497"]["promotion_gate_passed"] is False
    assert resolution["exp497"]["selected_prediction"] == "exp413_oof"
    assert resolution["selected_anchor"]["id"] == "exp413_saved_stage_d_oof"
    execution = module.CONFIG["execution_contract"]
    assert execution["scientific_primary_variants"] == 1
    assert execution["report_only_controls"] == 1
    assert all(
        execution[key] == 0
        for key in (
            "trained_models",
            "total_boosters",
            "hmm_runs",
            "pf_runs",
            "beam_runs",
            "parent_or_control_retraining",
            "gpu_runs",
        )
    )
    assert module.CONFIG["implementation"]["kaggle_run_approved"] is True
    assert (
        module.CONFIG["implementation"][
            "canonical_train_notebook_is_template_placeholder"
        ]
        is False
    )


def test_numpy_boolean_metrics_are_json_serializable() -> None:
    module = load_module()
    payload = {
        "primary_gate": {
            "passed": np.bool_(False),
            "checks": {"lambda_positive": np.bool_(True)},
        }
    }
    encoded = json.dumps(module.to_jsonable(payload))
    assert json.loads(encoded) == {
        "primary_gate": {
            "passed": False,
            "checks": {"lambda_positive": True},
        }
    }


def test_truth_free_loaders_use_exact_allowlist_and_freeze_before_truth(
    tmp_path: Path,
) -> None:
    module = load_module()
    anchor_source, correction_source = synthetic_sources()
    anchor_path = tmp_path / "stage_d_oof_predictions.parquet"
    correction_path = tmp_path / "exp490.csv.gz"
    anchor_source.to_parquet(anchor_path, index=False)
    correction_source.to_csv(correction_path, index=False, compression="gzip")

    anchor, anchor_manifest = module.load_anchor_without_truth(
        anchor_path,
        expected_sha256=module.sha256_file(anchor_path),
        expected_rows=len(anchor_source),
        expected_wells=anchor_source["well"].nunique(),
        expected_cv=module.rmse(
            anchor_source["actual_tvt"],
            anchor_source[
                "scale5_x1p0_full_replacement__lgb_mean__pred_tvt"
            ],
        ),
    )
    correction, correction_manifest = module.load_correction_without_truth(
        correction_path,
        expected_raw_gzip_sha256=module.sha256_file(correction_path),
        expected_decompressed_sha256=module.sha256_gzip_payload(correction_path),
        expected_rows=len(correction_source),
        expected_wells=correction_source["well"].nunique(),
    )
    assert anchor_manifest["truth_column_loaded"] is False
    assert correction_manifest["loaded_columns"] == list(
        module.EXP490_INPUT_ALLOWLIST
    )
    assert correction_manifest["loaded_column_count"] == 6
    assert (
        correction_manifest[
            "forbidden_truth_error_episode_role_fold_scope_by_well_gate_columns_loaded"
        ]
        == 0
    )
    assert "fold" not in correction.columns
    assert "true_tvt_readout_only" not in correction.columns
    frozen, frozen_manifest = module.freeze_truth_free_components(anchor, correction)
    assert "actual_tvt" not in frozen.columns
    assert frozen_manifest["truth_attached"] is False
    assert frozen_manifest["missing_or_extra_keys"] == 0
    assert frozen_manifest["suffix_offset_mismatch_rows"] == 0

    expected_anchor_rmse = module.rmse(
        anchor_source["actual_tvt"],
        anchor_source["scale5_x1p0_full_replacement__lgb_mean__pred_tvt"],
    )
    with_truth, truth_manifest = module.attach_anchor_truth(
        anchor_path,
        frozen,
        expected_sha256=module.sha256_file(anchor_path),
        expected_anchor_rmse=expected_anchor_rmse,
    )
    assert truth_manifest["truth_loaded_after_component_freeze"] is True
    assert np.isfinite(with_truth["actual_tvt"]).all()


def test_other_four_fold_closed_form_recovers_fixed_lambda() -> None:
    module = load_module()
    anchor_source, correction_source = synthetic_sources()
    frame = pd.DataFrame(
        {
            "well": correction_source["well"],
            "row_idx": correction_source["row_idx"],
            "outer_fold": anchor_source["outer_fold"],
            "actual_tvt": anchor_source["actual_tvt"],
            "anchor_prediction": anchor_source[
                "scale5_x1p0_full_replacement__lgb_mean__pred_tvt"
            ],
            "correction": correction_source["geometry_mean_reverting_hmm"]
            - correction_source["exp357_parent_prediction"],
        }
    )
    prediction, weights = module.crossfit_additive_component(
        frame, "correction", weight_name="lambda"
    )
    assert np.allclose(weights["lambda"], 0.05)
    assert weights["strict_interior"].all()
    assert np.allclose(prediction, frame["actual_tvt"])
    assert all(len(value.split(",")) == 4 for value in weights["fit_folds"])


def test_held_fold_truth_cannot_change_its_own_weight() -> None:
    module = load_module()
    anchor_source, correction_source = synthetic_sources()
    frame = pd.DataFrame(
        {
            "well": correction_source["well"],
            "row_idx": correction_source["row_idx"],
            "outer_fold": anchor_source["outer_fold"],
            "actual_tvt": anchor_source["actual_tvt"],
            "anchor_prediction": anchor_source[
                "scale5_x1p0_full_replacement__lgb_mean__pred_tvt"
            ],
            "correction": correction_source["geometry_mean_reverting_hmm"]
            - correction_source["exp357_parent_prediction"],
        }
    )
    _, original = module.crossfit_additive_component(frame, "correction")
    changed = frame.copy()
    changed.loc[changed["outer_fold"].eq(0), "actual_tvt"] += 1000.0
    _, modified = module.crossfit_additive_component(changed, "correction")
    original_weight = original.set_index("held_fold").loc[0, "lambda"]
    modified_weight = modified.set_index("held_fold").loc[0, "lambda"]
    assert original_weight == modified_weight


def test_primary_gate_is_all_and_and_control_is_nonselectable(tmp_path: Path) -> None:
    module = load_module()
    anchor_source, correction_source = synthetic_sources()
    frame = pd.DataFrame(
        {
            "id": anchor_source["id"],
            "well": anchor_source["well"],
            "row_idx": correction_source["row_idx"],
            "md_since": anchor_source["md_since"],
            "outer_fold": anchor_source["outer_fold"],
            "actual_tvt": anchor_source["actual_tvt"],
            "anchor_prediction": anchor_source[
                "scale5_x1p0_full_replacement__lgb_mean__pred_tvt"
            ],
            "geometry_mean_reverting_hmm": correction_source[
                "geometry_mean_reverting_hmm"
            ],
            "exp357_parent_prediction": correction_source[
                "exp357_parent_prediction"
            ],
        }
    )
    frame["correction"] = (
        frame["geometry_mean_reverting_hmm"] - frame["exp357_parent_prediction"]
    )
    frame["primary_prediction"] = frame["actual_tvt"]
    assignment = pd.DataFrame(
        {
            "well_id": sorted(frame["well"].unique()),
            "verification_like_spatial_role": "valid",
            "verification_like_typewell_purged_role": "valid",
        }
    )
    assignment_path = tmp_path / "assignment.csv"
    assignment.to_csv(assignment_path, index=False)
    pooled, folds, scopes, by_well = module.build_primary_readouts(
        frame,
        hidden_like_assignment_path=assignment_path,
        hidden_like_assignment_sha256=module.sha256_file(assignment_path),
    )
    weights = pd.DataFrame(
        {
            "held_fold": range(5),
            "lambda": [0.05] * 5,
            "strict_interior": [True] * 5,
        }
    )
    gate = module.build_primary_gate(
        config=module.CONFIG,
        pooled=pooled,
        fold_metrics=folds,
        scope_metrics=scopes,
        by_well=by_well,
        weights=weights,
        technical_checks={"synthetic_contract": True},
    )
    assert gate["passed"] is True
    assert all(gate["checks"].values())
    control, _, _ = module.build_report_only_control(frame)
    assert control["selectable"] is False
    assert control["may_rescue_primary"] is False


def test_upper_bound_hit_fails_strict_lambda_gate(tmp_path: Path) -> None:
    module = load_module()
    anchor_source, correction_source = synthetic_sources()
    frame = pd.DataFrame(
        {
            "id": anchor_source["id"],
            "well": anchor_source["well"],
            "row_idx": correction_source["row_idx"],
            "md_since": anchor_source["md_since"],
            "outer_fold": anchor_source["outer_fold"],
            "actual_tvt": anchor_source["actual_tvt"],
            "anchor_prediction": anchor_source[
                "scale5_x1p0_full_replacement__lgb_mean__pred_tvt"
            ],
            "primary_prediction": anchor_source["actual_tvt"],
        }
    )
    assignment = pd.DataFrame(
        {
            "well_id": sorted(frame["well"].unique()),
            "verification_like_spatial_role": "valid",
            "verification_like_typewell_purged_role": "valid",
        }
    )
    assignment_path = tmp_path / "assignment.csv"
    assignment.to_csv(assignment_path, index=False)
    pooled, folds, scopes, by_well = module.build_primary_readouts(
        frame,
        hidden_like_assignment_path=assignment_path,
        hidden_like_assignment_sha256=module.sha256_file(assignment_path),
    )
    weights = pd.DataFrame(
        {
            "held_fold": range(5),
            "lambda": [0.10] * 5,
            "strict_interior": [False] * 5,
        }
    )
    gate = module.build_primary_gate(
        config=module.CONFIG,
        pooled=pooled,
        fold_metrics=folds,
        scope_metrics=scopes,
        by_well=by_well,
        weights=weights,
        technical_checks={"synthetic_contract": True},
    )
    assert gate["passed"] is False
    assert gate["checks"]["lambda_strictly_below_upper_bound_5_of_5"] is False
    assert gate["decision"] == "FAIL_CLOSE_WITHOUT_WEIGHT_SCOPE_COMPONENT_OR_GATE_RESCUE"


def test_source_has_phase_order_and_no_model_or_inference_path() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert source.index("write_json(gate_path, gate)") < source.index(
        "report_only_control, control_weights, control_prediction = build_report_only_control("
    )
    assert "Path(__file__)" not in source
    assert "import lightgbm" not in source
    assert "import catboost" not in source
    assert "run_particle_filter(" not in source
    assert "run_beam(" not in source
    assert "to_csv(\"submission.csv\"" not in source
    for artifact in (
        "anchor_resolution_manifest.json",
        "input_manifest.json",
        "correction_manifest.json",
        "meta_fold_weights.csv",
        "primary_oof_predictions.parquet",
        "primary_fold_metrics.csv",
        "primary_scope_metrics.csv",
        "primary_by_well.csv",
        "primary_gate.json",
        "report_only_control.json",
        "reproducibility_manifest.json",
    ):
        assert artifact in source

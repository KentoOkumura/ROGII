from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.candidate_selector_pipeline import (
    Exp263CandidateCache,
    IncrementalParquetWriter,
    ShapeState,
    add_candidate_labels,
    build_candidate_long_features,
    build_raw_context,
    candidate_contract_sha,
    candidate_ids,
    deterministic_sample_indices,
    load_feature_schema,
    logical_frame_sha256,
    sha256_file,
    write_json,
)


EXPERIMENT_NAME = "exp414_fold_safe_candidate_rmse_offset_selector_on_exp264"
FIT_ROW_ID_COLUMNS = ["id", "well", "well_row_idx", "outer_fold"]
REQUIRED_LABEL_COLUMNS = [
    *FIT_ROW_ID_COLUMNS,
    "candidate_id",
    "candidate_abs_error",
]
HIDDEN_LIKE_SCOPES = {
    "hidden_like_spatial": "verification_like_spatial_role",
    "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
}


def _float64_content_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _pooled_rmse(frame: pd.DataFrame, metric: str) -> float:
    if frame.empty:
        raise ValueError(f"cannot pool empty RMSE metric {metric}")
    rows = pd.to_numeric(frame["rows"], errors="raise").to_numpy(np.float64)
    values = pd.to_numeric(frame[metric], errors="raise").to_numpy(np.float64)
    if np.any(rows <= 0) or not np.isfinite(values).all():
        raise ValueError(f"invalid rows or values for pooled RMSE metric {metric}")
    return float(np.sqrt(np.average(np.square(values), weights=rows)))


def _rmse_from_abs_error(abs_error: np.ndarray) -> float:
    values = np.asarray(abs_error, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("cannot calculate RMSE from empty or non-finite errors")
    return float(np.sqrt(np.mean(np.square(values))))


def validate_candidate_rmse_offset_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    required = {
        "enabled": True,
        "fit_partition": "exact_deterministic_sampled_outer_train_rows",
        "target": "candidate_abs_error_minus_fit_candidate_rmse",
        "reconstruction": "max_zero_of_residual_prediction_plus_fit_candidate_rmse",
        "training_sample_weight": "none",
        "validation_sample_weight": "none",
        "metric_sample_weight": "none",
        "binary_objective_fit": False,
    }
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(
                f"candidate RMSE offset contract changed for {key}: "
                f"{cfg.get(key)!r} != {expected!r}"
            )
    if float(cfg.get("offset_scale", float("nan"))) != 1.0:
        raise ValueError("candidate RMSE offset scale must remain exactly 1.0")
    forbidden = set(str(item) for item in cfg.get("forbidden", []))
    expected_forbidden = {
        "global_oof_rmse",
        "target_division",
        "inverse_rmse_sample_weight",
        "binary_objective_weight",
        "offset_scale_grid",
        "clip_grid",
        "candidate_subset",
        "offset_as_feature",
    }
    if forbidden != expected_forbidden:
        raise ValueError("candidate RMSE offset forbidden-operation contract changed")
    return cfg


@dataclass
class CandidateRmseOffsetResult:
    partition_id: int
    candidate_order: list[str]
    offset: np.ndarray
    residual_target: np.ndarray
    table: pd.DataFrame
    sampling_manifest: dict[str, Any]
    truth_read_ledger: dict[str, Any]
    audit: dict[str, Any]


def build_candidate_rmse_offsets(
    labels: pd.DataFrame,
    candidate_order: Sequence[str],
    *,
    partition_id: int,
) -> CandidateRmseOffsetResult:
    """Build additive candidate-RMSE offsets from one exact fit partition.

    The function deliberately accepts only the already sampled candidate-long
    fit labels. It has no validation/global-OOF input and creates no weights.
    """

    candidates = [str(item) for item in candidate_order]
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("candidate order must be non-empty and unique")
    missing = [column for column in REQUIRED_LABEL_COLUMNS if column not in labels]
    if missing:
        raise ValueError(f"fit labels are missing required columns: {missing}")

    n_candidates = len(candidates)
    if len(labels) == 0 or len(labels) % n_candidates:
        raise ValueError("candidate-long labels do not form complete base-row blocks")
    n_base_rows = len(labels) // n_candidates
    expected_ids = np.tile(np.asarray(candidates, dtype=object), n_base_rows)
    actual_ids = labels["candidate_id"].astype(str).to_numpy()
    if not np.array_equal(actual_ids, expected_ids):
        raise ValueError("candidate-long order differs from the frozen candidate order")

    for column in FIT_ROW_ID_COLUMNS:
        matrix = labels[column].to_numpy().reshape(n_base_rows, n_candidates)
        if not np.all(matrix == matrix[:, :1]):
            raise ValueError(f"{column} changes within a candidate-long base-row block")
    fit_row_ids = labels.iloc[::n_candidates][FIT_ROW_ID_COLUMNS].reset_index(drop=True)
    if fit_row_ids["id"].astype(str).duplicated().any():
        raise ValueError("fit partition contains duplicate base-row IDs")

    errors = pd.to_numeric(labels["candidate_abs_error"], errors="coerce").to_numpy(
        np.float64
    )
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("fit-partition candidate errors must be finite and non-negative")
    error_matrix = errors.reshape(n_base_rows, n_candidates)
    offset = np.sqrt(np.mean(np.square(error_matrix), axis=0, dtype=np.float64))
    if not np.isfinite(offset).all() or np.any(offset < 0):
        raise ValueError("fit-partition candidate RMSE offset is invalid")

    residual_target = (error_matrix - offset[None, :]).reshape(-1)
    if not np.isfinite(residual_target).all():
        raise ValueError("candidate RMSE residual target is non-finite")
    reconstructed = residual_target + np.tile(offset, n_base_rows)
    reconstruction_max_abs_error = float(np.max(np.abs(reconstructed - errors)))
    if reconstruction_max_abs_error > 1.0e-12:
        raise AssertionError("candidate RMSE residual reconstruction is not exact")

    fit_row_id_sha = logical_frame_sha256(fit_row_ids)
    source_fold_counts = (
        fit_row_ids.groupby("outer_fold", sort=True).size().astype(int).to_dict()
    )
    table = pd.DataFrame(
        {
            "fit_partition": int(partition_id),
            "candidate_position": np.arange(n_candidates, dtype=np.int16),
            "candidate_id": candidates,
            "fit_candidate_rmse": offset,
            "offset_scale": 1.0,
            "fit_row_count": n_base_rows,
            "fit_long_row_count": len(labels),
            "fit_row_id_content_sha256": fit_row_id_sha,
        }
    )
    sampling_manifest = {
        "fit_partition": int(partition_id),
        "sampling_contract": "exp264_corrected_stage_b_v5",
        "stage_sample_seed_parts": ["exp264", "stage_b_sample", "<source_fold>"],
        "outer_train_seed_parts": [
            "exp264",
            "outer_train",
            int(partition_id),
            "<source_fold>",
        ],
        "fit_base_rows": n_base_rows,
        "fit_long_rows": len(labels),
        "fit_wells": int(fit_row_ids["well"].astype(str).nunique()),
        "source_outer_fold_base_rows": {
            str(int(key)): int(value) for key, value in source_fold_counts.items()
        },
        "fit_row_id_columns": list(FIT_ROW_ID_COLUMNS),
        "fit_row_id_content_sha256": fit_row_id_sha,
    }
    truth_read_ledger = {
        "fit_partition": int(partition_id),
        "offset_input": "candidate_abs_error_from_exact_fit_partition_labels_only",
        "fit_truth_rows": n_base_rows,
        "outer_valid_truth_reads_for_offset": 0,
        "hidden_like_truth_reads_for_offset": 0,
        "global_oof_truth_reads_for_offset": 0,
        "current_test_truth_reads_for_offset": 0,
    }
    audit = {
        "fit_partition": int(partition_id),
        "candidate_count": n_candidates,
        "fit_candidate_abs_error_content_sha256": _float64_content_sha256(errors),
        "residual_target_content_sha256": _float64_content_sha256(residual_target),
        "offset_content_sha256": _float64_content_sha256(offset),
        "residual_reconstruction_max_abs_error": reconstruction_max_abs_error,
        "training_sample_weight_applied": False,
        "validation_sample_weight_applied": False,
        "metric_sample_weight_applied": False,
        "binary_model_fit_count": 0,
        "fit_valid_well_overlap": None,
    }
    return CandidateRmseOffsetResult(
        partition_id=int(partition_id),
        candidate_order=candidates,
        offset=offset,
        residual_target=residual_target,
        table=table,
        sampling_manifest=sampling_manifest,
        truth_read_ledger=truth_read_ledger,
        audit=audit,
    )


def offsets_for_labels(
    labels: pd.DataFrame,
    candidate_order: Sequence[str],
    offset: Sequence[float],
) -> np.ndarray:
    candidates = [str(item) for item in candidate_order]
    values = np.asarray(offset, dtype=np.float64)
    if values.shape != (len(candidates),) or not np.isfinite(values).all():
        raise ValueError("candidate offset vector is invalid")
    if len(labels) % len(candidates):
        raise ValueError("labels do not form complete candidate blocks")
    expected_ids = np.tile(
        np.asarray(candidates, dtype=object), len(labels) // len(candidates)
    )
    if not np.array_equal(labels["candidate_id"].astype(str).to_numpy(), expected_ids):
        raise ValueError("label candidate order differs from offset candidate order")
    return np.tile(values, len(labels) // len(candidates))


def reconstruct_abs_error_score(
    residual_prediction: np.ndarray,
    row_offset: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(residual_prediction, dtype=np.float64)
    offset = np.asarray(row_offset, dtype=np.float64)
    if residual.shape != offset.shape:
        raise ValueError("residual prediction and RMSE offset shapes differ")
    score = np.maximum(residual + offset, 0.0)
    if not np.isfinite(score).all():
        raise ValueError("reconstructed candidate score is non-finite")
    return score


def write_candidate_rmse_offset_artifacts(
    output_dir: Path,
    results: Sequence[CandidateRmseOffsetResult],
) -> dict[str, Any]:
    if not results:
        raise ValueError("no candidate RMSE offset results were provided")
    ordered = sorted(results, key=lambda item: item.partition_id)
    if len({item.partition_id for item in ordered}) != len(ordered):
        raise ValueError("candidate RMSE offset partitions are duplicated")
    candidate_order = ordered[0].candidate_order
    if any(item.candidate_order != candidate_order for item in ordered):
        raise ValueError("candidate order changed across offset partitions")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    table = pd.concat([item.table for item in ordered], ignore_index=True)
    table_path = root / "candidate_rmse_offset_by_fold.csv"
    table.to_csv(table_path, index=False)

    sampling = pd.DataFrame([item.sampling_manifest for item in ordered])
    for column in (
        "stage_sample_seed_parts",
        "outer_train_seed_parts",
        "fit_row_id_columns",
        "source_outer_fold_base_rows",
    ):
        sampling[column] = sampling[column].map(
            lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
    sampling_path = root / "candidate_rmse_offset_sampling_manifest.csv"
    sampling.to_csv(sampling_path, index=False)

    truth = pd.DataFrame([item.truth_read_ledger for item in ordered])
    truth_path = root / "candidate_rmse_offset_truth_read_ledger.csv"
    truth.to_csv(truth_path, index=False)
    forbidden_truth_columns = [
        "outer_valid_truth_reads_for_offset",
        "hidden_like_truth_reads_for_offset",
        "global_oof_truth_reads_for_offset",
        "current_test_truth_reads_for_offset",
    ]
    audits = [dict(item.audit) for item in ordered]
    manifest = {
        "schema_version": "1.0.0",
        "status": "fit_partition_candidate_rmse_additive_offsets_complete",
        "method": "unweighted_l1_candidate_rmse_additive_base_offset",
        "candidate_order": candidate_order,
        "candidate_count": len(candidate_order),
        "fit_partitions": [item.partition_id for item in ordered],
        "partition_count": len(ordered),
        "offset_scale": 1.0,
        "training_sample_weight_applied": False,
        "validation_sample_weight_applied": False,
        "metric_sample_weight_applied": False,
        "binary_model_fit_count": 0,
        "table": {
            "path": table_path.name,
            "rows": len(table),
            "sha256": sha256_file(table_path),
            "logical_sha256": logical_frame_sha256(table),
        },
        "sampling_manifest": {
            "path": sampling_path.name,
            "rows": len(sampling),
            "sha256": sha256_file(sampling_path),
        },
        "truth_read_ledger": {
            "path": truth_path.name,
            "rows": len(truth),
            "sha256": sha256_file(truth_path),
            "forbidden_truth_reads": int(
                truth[forbidden_truth_columns].to_numpy(np.int64).sum()
            ),
        },
        "audits": audits,
        "all_checks_passed": bool(
            len(ordered) == 5
            and int(truth[forbidden_truth_columns].to_numpy(np.int64).sum()) == 0
            and all(
                int(item["fit_valid_well_overlap"]) == 0
                and float(item["residual_reconstruction_max_abs_error"]) <= 1.0e-12
                and not bool(item["training_sample_weight_applied"])
                and not bool(item["validation_sample_weight_applied"])
                and not bool(item["metric_sample_weight_applied"])
                and int(item["binary_model_fit_count"]) == 0
                for item in audits
            )
        ),
    }
    manifest_path = root / "candidate_rmse_offset_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def validate_static_contract(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = candidate_ids(contract)
    expected_candidates = [str(item) for item in config["candidate_bank"]["order"]]
    if candidates != expected_candidates:
        raise ValueError("candidate order differs from the frozen parent contract")
    primary = [
        str(item)
        for item in contract["legal_domains"]["primitive_pair_bank"]["candidates"]
    ]
    if primary != [str(item) for item in config["candidate_bank"]["primary_domain"]]:
        raise ValueError("primary selection domain differs from the frozen contract")
    if config["experiment"]["route"] != "ml_model":
        raise ValueError("candidate RMSE offset selector must remain ml_model")
    if int(config["features"]["feature_count"]) != 88:
        raise ValueError("candidate RMSE offset selector must retain 88 features")
    validate_candidate_rmse_offset_config(config["candidate_rmse_offset"])

    execution = config["execution"]
    cost = {
        "active_variants": int(execution["active_variants"]),
        "objectives": int(execution["objectives"]),
        "outer_folds": int(execution["outer_folds"]),
        "planned_cpu_boosters": int(execution["planned_cpu_boosters"]),
        "parent_control_retraining": bool(execution["parent_control_retraining"]),
        "classifier_boosters": int(execution["classifier_boosters"]),
        "gpu_boosters": int(execution["gpu_boosters"]),
        "pf_hmm_beam_regeneration": bool(execution["pf_hmm_beam_regeneration"]),
        "inference": bool(execution["inference"]),
        "submission": bool(execution["submission"]),
    }
    expected_cost = {
        "active_variants": 1,
        "objectives": 1,
        "outer_folds": 5,
        "planned_cpu_boosters": 5,
        "parent_control_retraining": False,
        "classifier_boosters": 0,
        "gpu_boosters": 0,
        "pf_hmm_beam_regeneration": False,
        "inference": False,
        "submission": False,
    }
    if cost != expected_cost:
        raise ValueError(f"exp414 compute contract changed: {cost}")
    return {
        "experiment": EXPERIMENT_NAME,
        "candidate_order": candidates,
        "candidate_count": len(candidates),
        "primary_domain": primary,
        "feature_count": 88,
        "cost": cost,
        "run_approved": bool(execution["run_approved"]),
    }


def run_rmse_offset_stage_b(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    cache_root: Path,
    raw_train_dir: Path,
    output_dir: Path,
    cache_factory: Callable[[Path, Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    validate_static_contract(config, contract)
    root = Path(output_dir)
    schema = load_feature_schema(root / "feature_schema.json")
    features = [str(item) for item in schema["features"]]
    if len(features) != 88:
        raise ValueError("Stage A feature count differs from the frozen 88")
    cache = (
        Exp263CandidateCache(cache_root, contract)
        if cache_factory is None
        else cache_factory(cache_root, contract)
    )
    n_folds = int(config["validation"]["outer_folds"])
    n_candidates = len(cache.ids)
    feature_cfg = dict(config["features"])
    feature_cfg["primary_domain"] = contract["legal_domains"]["primitive_pair_bank"][
        "candidates"
    ]
    feature_cfg["fixed_domain"] = contract["legal_domains"]["primitive_fixed_bank"][
        "candidates"
    ]
    train_cfg = dict(config["model"]["training"])
    per_fold_train_limit = max(
        1,
        int(
            math.ceil(
                int(train_cfg["max_train_base_rows_per_outer_fold"])
                / (n_folds - 1)
            )
        ),
    )
    valid_limit = int(train_cfg["max_valid_base_rows_for_early_stopping"])

    sampled: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold in range(n_folds):
        bundle = cache.load_fold(fold)
        context, truth = build_raw_context(
            bundle.base, raw_train_dir, feature_cfg, require_truth=True
        )
        assert truth is not None
        indices = deterministic_sample_indices(
            bundle.base,
            max(per_fold_train_limit, valid_limit),
            "exp264",
            "stage_b_sample",
            fold,
        )
        long, metadata = build_candidate_long_features(
            bundle,
            context,
            indices,
            feature_cfg,
            expected_features=features,
        )
        sampled[fold] = (long, add_candidate_labels(metadata, truth[indices], n_candidates))

    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    common = dict(config["model"]["lightgbm_common"])
    seed = int(config["validation"]["seed"])
    num_round = int(train_cfg["num_boost_round"])

    def model_callbacks() -> list[Any]:
        return [
            early_stopping(int(train_cfg["early_stopping_rounds"]), verbose=False),
            log_evaluation(int(train_cfg["log_evaluation_period"])),
        ]

    score_path = root / "candidate_score_oof.parquet"
    score_writer = IncrementalParquetWriter(score_path)
    offset_results: list[CandidateRmseOffsetResult] = []
    model_records: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    candidate_metric_rows: list[dict[str, Any]] = []
    distance_metric_rows: list[dict[str, Any]] = []
    selection_counts: dict[tuple[int, str], int] = defaultdict(int)
    by_well_parts: dict[str, list[np.ndarray]] = defaultdict(list)

    for outer_fold in range(n_folds):
        train_feature_parts: list[pd.DataFrame] = []
        train_label_parts: list[pd.DataFrame] = []
        for source_fold in range(n_folds):
            if source_fold == outer_fold:
                continue
            feature_part, label_part = sampled[source_fold]
            sampled_base_count = len(feature_part) // n_candidates
            selected_base = deterministic_sample_indices(
                pd.DataFrame(index=np.arange(sampled_base_count)),
                min(per_fold_train_limit, sampled_base_count),
                "exp264",
                "outer_train",
                outer_fold,
                source_fold,
            )
            selected_long = (
                selected_base[:, None] * n_candidates
                + np.arange(n_candidates)[None, :]
            ).reshape(-1)
            train_feature_parts.append(feature_part.iloc[selected_long])
            train_label_parts.append(label_part.iloc[selected_long])

        x_train = pd.concat(train_feature_parts, ignore_index=True).astype(np.float32)
        y_train = pd.concat(train_label_parts, ignore_index=True)
        x_valid_all, y_valid_all = sampled[outer_fold]
        valid_long_limit = min(len(x_valid_all), valid_limit * n_candidates)
        x_valid = x_valid_all.iloc[:valid_long_limit].astype(np.float32)
        y_valid = y_valid_all.iloc[:valid_long_limit]

        offset_result = build_candidate_rmse_offsets(
            y_train,
            cache.ids,
            partition_id=outer_fold,
        )
        train_wells = set(y_train["well"].astype(str))
        valid_wells = set(y_valid["well"].astype(str))
        overlap = len(train_wells.intersection(valid_wells))
        if overlap:
            raise ValueError("candidate RMSE offset fit rows overlap outer-valid wells")
        offset_result.audit.update(
            {
                "fit_valid_well_overlap": overlap,
                "fit_wells": len(train_wells),
                "valid_wells": len(valid_wells),
                "fit_feature_schema_sha256": schema["feature_schema_sha256"],
                "fit_feature_content_sha256": logical_frame_sha256(x_train),
            }
        )
        offset_results.append(offset_result)
        valid_offset = offsets_for_labels(
            y_valid,
            cache.ids,
            offset_result.offset,
        )
        valid_residual = (
            y_valid["candidate_abs_error"].to_numpy(np.float64) - valid_offset
        )

        regressor = LGBMRegressor(
            objective="regression_l1",
            n_estimators=num_round,
            random_state=seed + 100 + outer_fold,
            **common,
        )
        regressor.fit(
            x_train,
            offset_result.residual_target,
            eval_set=[(x_valid, valid_residual)],
            eval_metric="l1",
            callbacks=model_callbacks(),
        )
        model_path = model_dir / f"selector_pred_abs_error_fold{outer_fold}.txt"
        regressor.booster_.save_model(str(model_path))
        model_sha = sha256_file(model_path)
        model_records.append(
            {
                "outer_fold": outer_fold,
                "objective": "pred_abs_error",
                "path": str(model_path.relative_to(root)),
                "sha256": model_sha,
                "best_iteration": int(regressor.best_iteration_),
                "train_long_rows": len(x_train),
                "early_stop_long_rows": len(x_valid),
                "training_label": "candidate_abs_error_minus_fit_candidate_rmse",
                "score_reconstruction": (
                    "max_zero_of_residual_prediction_plus_fit_candidate_rmse"
                ),
                "offset_content_sha256": offset_result.audit[
                    "offset_content_sha256"
                ],
                "training_sample_weight_applied": False,
                "validation_sample_weight_applied": False,
            }
        )
        for importance_type in ("gain", "split"):
            importance_values = regressor.booster_.feature_importance(
                importance_type=importance_type
            )
            for feature, importance in zip(
                features, importance_values, strict=True
            ):
                importance_rows.append(
                    {
                        "feature": feature,
                        "objective": "pred_abs_error",
                        "fold": outer_fold,
                        "importance_type": importance_type,
                        "importance": float(importance),
                    }
                )

        bundle = cache.load_fold(outer_fold)
        context, truth = build_raw_context(
            bundle.base, raw_train_dir, feature_cfg, require_truth=True
        )
        assert truth is not None
        shape_state = ShapeState.from_bundle(bundle.base, bundle.values)
        fold_actual_error: list[np.ndarray] = []
        fold_pred_error: list[np.ndarray] = []
        fold_hard_error: list[np.ndarray] = []
        fold_md_since: list[np.ndarray] = []
        primary_positions = [
            cache.ids.index(str(name))
            for name in contract["legal_domains"]["primitive_pair_bank"]["candidates"]
        ]
        fallback_position = cache.ids.index(str(config["candidate_bank"]["fixed_fallback"]))
        chunk_size = int(train_cfg["predict_base_row_chunk_size"])
        for start in range(0, len(bundle.base), chunk_size):
            stop = min(start + chunk_size, len(bundle.base))
            indices = np.arange(start, stop, dtype=np.int64)
            long, metadata = build_candidate_long_features(
                bundle,
                context,
                indices,
                feature_cfg,
                shape_state=shape_state,
                expected_features=features,
            )
            residual_prediction = regressor.predict(
                long.astype(np.float32),
                num_iteration=regressor.best_iteration_,
            )
            row_offset = np.tile(offset_result.offset, len(indices))
            reconstructed = reconstruct_abs_error_score(
                residual_prediction,
                row_offset,
            )
            score_matrix = reconstructed.reshape(len(indices), n_candidates)
            labels = add_candidate_labels(metadata, truth[indices], n_candidates)
            actual_matrix = (
                labels["candidate_abs_error"]
                .to_numpy(np.float64)
                .reshape(len(indices), n_candidates)
            )

            score = metadata.copy()
            score["actual_abs_error"] = actual_matrix.reshape(-1).astype(np.float32)
            score["actual_within10"] = (
                actual_matrix.reshape(-1) <= 10.0
            ).astype(np.int8)
            score["fit_candidate_rmse"] = row_offset.astype(np.float32)
            score["pred_residual_abs_error"] = np.asarray(
                residual_prediction, dtype=np.float32
            )
            score["pred_abs_error"] = reconstructed.astype(np.float32)
            score["feature_schema_sha"] = schema["feature_schema_sha256"]
            score["candidate_contract_sha"] = candidate_contract_sha(contract)
            score["model_fold"] = outer_fold
            score["pred_abs_error_model_sha"] = model_sha
            score_writer.write(score)

            primary_scores = score_matrix[:, primary_positions]
            selected_local = np.argmin(primary_scores, axis=1)
            selected_position = np.asarray(primary_positions)[selected_local]
            selected_names = np.asarray(cache.ids, dtype=object)[selected_position]
            for name, count in zip(
                *np.unique(selected_names, return_counts=True), strict=True
            ):
                selection_counts[(outer_fold, str(name))] += int(count)
            row_index = np.arange(len(indices))
            hard_error = actual_matrix[row_index, selected_position]
            fallback_error = actual_matrix[:, fallback_position]
            md_since = bundle.base.iloc[indices]["md_since"].to_numpy(np.float64)

            fold_actual_error.append(actual_matrix.reshape(-1))
            fold_pred_error.append(score_matrix.reshape(-1))
            fold_hard_error.append(hard_error)
            fold_md_since.append(md_since)
            wells = bundle.base.iloc[indices]["well"].astype(str).to_numpy()
            for well in np.unique(wells):
                mask = wells == well
                by_well_parts[str(well)].append(
                    np.column_stack(
                        [
                            hard_error[mask],
                            fallback_error[mask],
                        ]
                    )
                )

        actual_vector = np.concatenate(fold_actual_error)
        predicted_vector = np.concatenate(fold_pred_error)
        hard_error_vector = np.concatenate(fold_hard_error)
        md_vector = np.concatenate(fold_md_since)
        actual_matrix = actual_vector.reshape(-1, n_candidates)
        predicted_matrix = predicted_vector.reshape(-1, n_candidates)
        for position, candidate_id in enumerate(cache.ids):
            candidate_metric_rows.append(
                {
                    "outer_fold": outer_fold,
                    "candidate_id": candidate_id,
                    "rows": len(actual_matrix),
                    "expected_error_mae": float(
                        np.mean(
                            np.abs(
                                predicted_matrix[:, position]
                                - actual_matrix[:, position]
                            )
                        )
                    ),
                }
            )
        for bucket, lower, upper in (
            ("near_0_250", -np.inf, 250.0),
            ("250_500", 250.0, 500.0),
            ("500_1000", 500.0, 1000.0),
            ("1000_plus", 1000.0, np.inf),
        ):
            mask = (md_vector >= lower) & (md_vector < upper)
            distance_metric_rows.append(
                {
                    "outer_fold": outer_fold,
                    "distance_bucket": bucket,
                    "rows": int(mask.sum()),
                    "hard_primary_rmse": _rmse_from_abs_error(
                        hard_error_vector[mask]
                    )
                    if mask.any()
                    else np.nan,
                }
            )
        fold_metric_rows.append(
            {
                "scope": "outer_fold",
                "fold": outer_fold,
                "rows": len(actual_vector),
                "base_rows": len(hard_error_vector),
                "expected_error_mae": float(
                    np.mean(np.abs(predicted_vector - actual_vector))
                ),
                "hard_primary_rmse": _rmse_from_abs_error(hard_error_vector),
            }
        )

    score_writer.close()
    offset_manifest = write_candidate_rmse_offset_artifacts(root, offset_results)
    if not bool(offset_manifest["all_checks_passed"]):
        raise RuntimeError("candidate RMSE offset technical audit failed")

    importance = pd.DataFrame(importance_rows)
    pivot = (
        importance.pivot_table(
            index=["feature", "objective", "fold"],
            columns="importance_type",
            values="importance",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"gain": "gain_importance", "split": "split_importance"})
    )
    summary_importance = pivot.groupby(["feature", "objective"], as_index=False)[
        "gain_importance"
    ].agg(importance_mean="mean", importance_std="std")
    summary_importance["importance_rank"] = summary_importance.groupby(
        "objective"
    )["importance_mean"].rank(method="dense", ascending=False)
    pivot = pivot.merge(
        summary_importance,
        on=["feature", "objective"],
        how="left",
        validate="many_to_one",
    )
    pivot.to_csv(root / "feature_importance_by_objective_fold.csv", index=False)
    pd.DataFrame(candidate_metric_rows).to_csv(
        root / "selector_candidate_metrics.csv", index=False
    )
    pd.DataFrame(distance_metric_rows).to_csv(
        root / "selector_distance_bucket_metrics.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "outer_fold": fold,
                "objective": "pred_abs_error",
                "candidate_id": candidate,
                "selected_rows": count,
            }
            for (fold, candidate), count in sorted(selection_counts.items())
        ]
    ).to_csv(root / "selector_selection_rate.csv", index=False)

    by_well_rows: list[dict[str, Any]] = []
    for well, parts in by_well_parts.items():
        values = np.concatenate(parts, axis=0)
        by_well_rows.append(
            {
                "well": well,
                "rows": len(values),
                "hard_primary_rmse": _rmse_from_abs_error(values[:, 0]),
                "fixed_fallback_rmse": _rmse_from_abs_error(values[:, 1]),
            }
        )
    pd.DataFrame(by_well_rows).to_csv(root / "selector_by_well.csv", index=False)

    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics.to_csv(root / "selector_metrics.csv", index=False)
    long_rows = fold_metrics["rows"].to_numpy(np.float64)
    base_rows = fold_metrics["base_rows"].to_numpy(np.float64)
    pooled_expected_error_mae = float(
        np.average(fold_metrics["expected_error_mae"], weights=long_rows)
    )
    hard_primary_oof_rmse = float(
        np.sqrt(
            np.average(
                np.square(fold_metrics["hard_primary_rmse"]), weights=base_rows
            )
        )
    )
    model_manifest = {
        "schema_version": "1.0.0",
        "status": "rmse_offset_selector_outer_oof_completed",
        "candidate_order": cache.ids,
        "feature_schema_sha256": schema["feature_schema_sha256"],
        "feature_count": len(features),
        "models": model_records,
        "model_count": len(model_records),
        "objective_inventory": ["pred_abs_error"],
        "classifier_model_count": 0,
        "candidate_rmse_offset": offset_manifest,
    }
    write_json(root / "selector_model_manifest.json", model_manifest)
    metrics = {
        "status": "rmse_offset_selector_outer_oof_completed",
        "fold_metrics": fold_metric_rows,
        "pooled_score_metrics": {
            "expected_error_mae": pooled_expected_error_mae,
        },
        "hard_primary_oof_rmse": hard_primary_oof_rmse,
        "model_count": len(model_records),
        "classifier_model_count": 0,
        "candidate_score_oof_rows": int(score_writer.rows),
        "candidate_score_oof_sha256": sha256_file(score_path),
        "model_manifest_sha256": sha256_file(
            root / "selector_model_manifest.json"
        ),
        "candidate_rmse_offset": offset_manifest,
    }
    write_json(root / "selector_metrics.json", metrics)
    reproducibility_path = root / "reproducibility_manifest.json"
    reproducibility = json.loads(reproducibility_path.read_text())
    reproducibility.update(
        {
            "status": "rmse_offset_selector_outer_oof_completed",
            "candidate_rmse_offset_manifest_sha256": offset_manifest[
                "manifest_sha256"
            ],
            "model_manifest_sha256": metrics["model_manifest_sha256"],
            "candidate_score_oof_sha256": metrics["candidate_score_oof_sha256"],
        }
    )
    write_json(reproducibility_path, reproducibility)
    return metrics


def _hidden_like_metrics(
    new_by_well: pd.DataFrame,
    parent_by_well: pd.DataFrame,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    if assignment["well_id"].astype(str).duplicated().any():
        raise ValueError("hidden-like assignment contains duplicate well IDs")
    rows: list[dict[str, Any]] = []
    for scope, role_column in HIDDEN_LIKE_SCOPES.items():
        if role_column not in assignment:
            raise ValueError(f"hidden-like assignment is missing {role_column}")
        wells = set(
            assignment.loc[assignment[role_column].eq("valid"), "well_id"].astype(str)
        )
        new_scope = new_by_well[new_by_well["well"].astype(str).isin(wells)]
        parent_scope = parent_by_well[
            parent_by_well["well"].astype(str).isin(wells)
        ]
        if set(new_scope["well"].astype(str)) != wells:
            raise ValueError(f"new by-well metrics do not cover {scope}")
        if set(parent_scope["well"].astype(str)) != wells:
            raise ValueError(f"parent by-well metrics do not cover {scope}")
        new_rmse = _pooled_rmse(new_scope, "hard_primary_rmse")
        parent_rmse = _pooled_rmse(parent_scope, "hard_primary_rmse")
        rows.append(
            {
                "scope": scope,
                "role_column": role_column,
                "wells": len(wells),
                "rows": int(new_scope["rows"].sum()),
                "parent_hard_primary_rmse": parent_rmse,
                "new_hard_primary_rmse": new_rmse,
                "delta_rmse_new_minus_parent": new_rmse - parent_rmse,
            }
        )
    return pd.DataFrame(rows)


def evaluate_rmse_offset_gate(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    output_dir: Path,
    parent_metrics_path: Path,
    parent_bucket_path: Path,
    parent_by_well_path: Path,
    hidden_like_assignment_path: Path,
    root_cause_summary_path: Path,
    source_config_path: Path,
    source_candidate_contract_path: Path,
) -> dict[str, Any]:
    static = validate_static_contract(config, contract)
    root = Path(output_dir)
    new_metrics = json.loads((root / "selector_metrics.json").read_text())
    parent_metrics = json.loads(Path(parent_metrics_path).read_text())
    new_fold = pd.DataFrame(new_metrics["fold_metrics"]).sort_values("fold")
    parent_fold = pd.DataFrame(parent_metrics["fold_metrics"]).sort_values("fold")
    if not np.array_equal(
        new_fold["fold"].to_numpy(), parent_fold["fold"].to_numpy()
    ):
        raise ValueError("new and parent fold inventories differ")
    if not np.array_equal(
        new_fold["base_rows"].to_numpy(),
        (parent_fold["rows"] // len(static["candidate_order"])).to_numpy(),
    ):
        raise ValueError("new and parent base-row counts differ")

    fold_comparison = parent_fold.add_prefix("parent__").join(
        new_fold.reset_index(drop=True).add_prefix("new__")
    )
    fold_comparison["fold"] = new_fold["fold"].to_numpy()
    for metric in ("expected_error_mae", "hard_primary_rmse"):
        fold_comparison[f"delta__{metric}__new_minus_parent"] = (
            new_fold[metric].to_numpy() - parent_fold[metric].to_numpy()
        )
    fold_comparison_path = root / "exp414_parent_fold_comparison.csv"
    fold_comparison.to_csv(fold_comparison_path, index=False)

    new_bucket = pd.read_csv(root / "selector_distance_bucket_metrics.csv")
    parent_bucket = pd.read_csv(parent_bucket_path)
    key = ["outer_fold", "distance_bucket"]
    bucket_comparison = parent_bucket.merge(
        new_bucket,
        on=key,
        suffixes=("__parent", "__new"),
        validate="one_to_one",
    )
    if len(bucket_comparison) != len(parent_bucket) or len(
        bucket_comparison
    ) != len(new_bucket):
        raise ValueError("new and parent distance-bucket inventories differ")
    bucket_comparison["delta__hard_primary_rmse__new_minus_parent"] = (
        bucket_comparison["hard_primary_rmse__new"]
        - bucket_comparison["hard_primary_rmse__parent"]
    )
    bucket_comparison_path = root / "exp414_parent_bucket_comparison.csv"
    bucket_comparison.to_csv(bucket_comparison_path, index=False)
    pooled_bucket: dict[str, float] = {}
    for bucket in ("near_0_250", "1000_plus"):
        new_scope = new_bucket[new_bucket["distance_bucket"].eq(bucket)]
        parent_scope = parent_bucket[parent_bucket["distance_bucket"].eq(bucket)]
        pooled_bucket[bucket] = _pooled_rmse(
            new_scope, "hard_primary_rmse"
        ) - _pooled_rmse(parent_scope, "hard_primary_rmse")

    new_by_well = pd.read_csv(root / "selector_by_well.csv", dtype={"well": str})
    parent_by_well = pd.read_csv(parent_by_well_path, dtype={"well": str})
    by_well = parent_by_well.merge(
        new_by_well,
        on="well",
        suffixes=("__parent", "__new"),
        validate="one_to_one",
    )
    if len(by_well) != int(config["data"]["expected_wells"]):
        raise ValueError("new and parent by-well inventories differ")
    if not np.array_equal(by_well["rows__parent"], by_well["rows__new"]):
        raise ValueError("new and parent by-well row counts differ")
    by_well["delta_rmse_new_minus_parent"] = (
        by_well["hard_primary_rmse__new"]
        - by_well["hard_primary_rmse__parent"]
    )
    by_well_path = root / "exp414_parent_by_well_comparison.csv"
    by_well.to_csv(by_well_path, index=False)

    assignment = pd.read_csv(hidden_like_assignment_path, dtype={"well_id": str})
    hidden = _hidden_like_metrics(new_by_well, parent_by_well, assignment)
    hidden_path = root / "exp414_parent_hidden_like_comparison.csv"
    hidden.to_csv(hidden_path, index=False)

    root_cause = json.loads(Path(root_cause_summary_path).read_text())
    root_gate = root_cause["root_cause_gate"]
    model_manifest = json.loads((root / "selector_model_manifest.json").read_text())
    offset_manifest = json.loads(
        (root / "candidate_rmse_offset_manifest.json").read_text()
    )
    model_sha_matches = sum(
        sha256_file(root / str(item["path"])) == str(item["sha256"])
        for item in model_manifest["models"]
    )
    technical_checks = {
        "root_cause_gate": bool(root_gate["passed"]),
        "feature_count": int(model_manifest["feature_count"]) == 88,
        "feature_schema": str(model_manifest["feature_schema_sha256"])
        == str(config["data"]["expected_feature_schema_logical_sha256"]),
        "candidate_order": model_manifest["candidate_order"]
        == static["candidate_order"],
        "model_count": int(model_manifest["model_count"]) == 5,
        "classifier_model_count": int(model_manifest["classifier_model_count"]) == 0,
        "model_sha_matches": model_sha_matches == 5,
        "objective_inventory": model_manifest["objective_inventory"]
        == ["pred_abs_error"],
        "offset_manifest": bool(offset_manifest["all_checks_passed"]),
        "offset_rows": int(offset_manifest["table"]["rows"]) == 60,
        "sample_weight_absent": not bool(
            offset_manifest["training_sample_weight_applied"]
        )
        and not bool(offset_manifest["validation_sample_weight_applied"])
        and not bool(offset_manifest["metric_sample_weight_applied"]),
        "forbidden_truth_reads": int(
            offset_manifest["truth_read_ledger"]["forbidden_truth_reads"]
        )
        == 0,
        "fit_valid_well_overlap": sum(
            int(item["fit_valid_well_overlap"])
            for item in offset_manifest["audits"]
        )
        == 0,
        "candidate_score_rows": int(new_metrics["candidate_score_oof_rows"])
        == int(config["data"]["expected_candidate_long_oof_rows"]),
        "candidate_score_sha": sha256_file(root / "candidate_score_oof.parquet")
        == str(new_metrics["candidate_score_oof_sha256"]),
        "model_manifest_sha": sha256_file(root / "selector_model_manifest.json")
        == str(new_metrics["model_manifest_sha256"]),
        "parent_control_retraining": not bool(
            config["execution"]["parent_control_retraining"]
        ),
    }

    science_cfg = config["scientific_gate"]
    hidden_delta = hidden.set_index("scope")[
        "delta_rmse_new_minus_parent"
    ].to_dict()
    expected_error_nonworse_folds = int(
        (
            new_fold["expected_error_mae"].to_numpy()
            <= parent_fold["expected_error_mae"].to_numpy()
        ).sum()
    )
    hard_nonworse_folds = int(
        (
            new_fold["hard_primary_rmse"].to_numpy()
            <= parent_fold["hard_primary_rmse"].to_numpy()
        ).sum()
    )
    instability = root_cause["treatment_instability"]
    scientific_checks = {
        "expected_error_mae": float(
            new_metrics["pooled_score_metrics"]["expected_error_mae"]
        )
        <= float(science_cfg["expected_error_mae_max"]),
        "expected_error_mae_nonworse_folds": expected_error_nonworse_folds
        >= int(science_cfg["expected_error_mae_nonworse_folds_min"]),
        "hard_primary_oof_rmse": float(new_metrics["hard_primary_oof_rmse"])
        <= float(science_cfg["hard_primary_oof_rmse_max"]),
        "hard_primary_nonworse_folds": hard_nonworse_folds
        >= int(science_cfg["hard_primary_nonworse_folds_min"]),
        "near_non_regression": float(pooled_bucket["near_0_250"])
        <= float(science_cfg["near_delta_vs_parent_max_ft"]),
        "distance_1000_plus_non_regression": float(pooled_bucket["1000_plus"])
        <= float(science_cfg["distance_1000_plus_delta_vs_parent_max_ft"]),
        "hidden_like_spatial_non_regression": float(
            hidden_delta["hidden_like_spatial"]
        )
        <= float(science_cfg["hidden_like_spatial_delta_vs_parent_max_ft"]),
        "hidden_like_typewell_purged_non_regression": float(
            hidden_delta["hidden_like_typewell_purged"]
        )
        <= float(
            science_cfg["hidden_like_typewell_purged_delta_vs_parent_max_ft"]
        ),
        "worst_well_non_regression": float(
            by_well["delta_rmse_new_minus_parent"].max()
        )
        <= float(science_cfg["worst_well_regression_vs_parent_max_ft"]),
        "row_local_instability_not_amplified": float(
            instability["treatment_mean_centered_delta_std"]
        )
        <= float(instability["exp407_mean_centered_delta_std"])
        * float(
            science_cfg[
                "mean_centered_score_delta_std_vs_parent_max_relative_to_exp407"
            ]
        ),
    }
    technical_passed = bool(all(technical_checks.values()))
    scientific_passed = bool(all(scientific_checks.values()))
    if not technical_passed:
        decision = "technical_fail_fix_same_frozen_contract_only"
    elif scientific_passed:
        decision = "rmse_additive_offset_method_established_at_stage_b"
    else:
        decision = "scientific_fail_close_rmse_additive_offset_without_rescue"

    summary = {
        "status": "exp414_rmse_offset_stage_b_gate_evaluated",
        "experiment": EXPERIMENT_NAME,
        "route": config["experiment"]["route"],
        "source_config_sha256": sha256_file(source_config_path),
        "source_candidate_contract_sha256": sha256_file(
            source_candidate_contract_path
        ),
        "cost_contract": static["cost"],
        "root_cause": root_cause,
        "technical": {
            "checks": technical_checks,
            "passed": technical_passed,
        },
        "scientific": {
            "checks": scientific_checks,
            "metrics": {
                "new_expected_error_mae": float(
                    new_metrics["pooled_score_metrics"]["expected_error_mae"]
                ),
                "parent_expected_error_mae": float(
                    parent_metrics["pooled_score_metrics"]["expected_error_mae"]
                ),
                "expected_error_nonworse_folds": expected_error_nonworse_folds,
                "new_hard_primary_oof_rmse": float(
                    new_metrics["hard_primary_oof_rmse"]
                ),
                "parent_hard_primary_oof_rmse": float(
                    parent_metrics["hard_primary_oof_rmse"]
                ),
                "hard_nonworse_folds": hard_nonworse_folds,
                "distance_delta_rmse": pooled_bucket,
                "hidden_like_delta_rmse": hidden_delta,
                "worst_well_delta_rmse": float(
                    by_well["delta_rmse_new_minus_parent"].max()
                ),
                "treatment_instability": instability,
            },
            "passed": scientific_passed,
        },
        "decision": decision,
        "artifacts": {
            "fold_comparison": fold_comparison_path.name,
            "bucket_comparison": bucket_comparison_path.name,
            "by_well_comparison": by_well_path.name,
            "hidden_like_comparison": hidden_path.name,
            "root_cause_summary": Path(root_cause_summary_path).name,
        },
    }
    gate_path = root / "exp414_stage_b_gate.json"
    write_json(gate_path, _jsonable(summary))
    summary["gate_file_sha256"] = sha256_file(gate_path)
    reproducibility_path = root / "reproducibility_manifest.json"
    reproducibility = json.loads(reproducibility_path.read_text())
    reproducibility.update(
        {
            "status": "exp414_rmse_offset_stage_b_gate_evaluated",
            "source_config_sha256": summary["source_config_sha256"],
            "source_candidate_contract_sha256": summary[
                "source_candidate_contract_sha256"
            ],
            "exp414_stage_b_gate_sha256": summary["gate_file_sha256"],
            "exp414_decision": decision,
        }
    )
    write_json(reproducibility_path, reproducibility)
    return summary


__all__ = [
    "CandidateRmseOffsetResult",
    "EXPERIMENT_NAME",
    "build_candidate_rmse_offsets",
    "evaluate_rmse_offset_gate",
    "offsets_for_labels",
    "reconstruct_abs_error_score",
    "run_rmse_offset_stage_b",
    "validate_candidate_rmse_offset_config",
    "validate_static_contract",
    "write_candidate_rmse_offset_artifacts",
]

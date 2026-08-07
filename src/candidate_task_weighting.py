from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


FIT_ROW_ID_COLUMNS = ["id", "well", "well_row_idx", "outer_fold"]
REQUIRED_LABEL_COLUMNS = [
    *FIT_ROW_ID_COLUMNS,
    "candidate_id",
    "candidate_abs_error",
]


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


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _logical_frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.select_dtypes(include=["string"]).columns:
        normalized[column] = normalized[column].astype(object)
    digest = hashlib.sha256()
    digest.update("|".join(normalized.columns).encode())
    digest.update("|".join(str(dtype) for dtype in normalized.dtypes).encode())
    hashes = pd.util.hash_pandas_object(normalized, index=False, categorize=True)
    digest.update(hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def _float64_content_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def validate_inverse_rmse_weight_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    if not bool(cfg.get("enabled")):
        raise ValueError("candidate task weighting must be explicitly enabled")
    if str(cfg.get("application")) != "training_rows_only":
        raise ValueError("candidate task weights must be limited to training rows")
    if str(cfg.get("unit")) != "candidate_within_exact_model_fit_partition":
        raise ValueError("candidate task weight unit differs from the frozen contract")
    if not bool(cfg.get("truth_join_after_schema_fold_and_row_freeze")):
        raise ValueError("truth must be joined only after schema/fold/row freeze")

    rmse_cfg = dict(cfg.get("rmse", {}))
    inverse_cfg = dict(cfg.get("inverse", {}))
    normalization_cfg = dict(cfg.get("normalization", {}))
    epsilon = float(rmse_cfg.get("epsilon_ft", float("nan")))
    exponent = float(inverse_cfg.get("exponent", float("nan")))
    clip = normalization_cfg.get("clip")
    if not math.isfinite(epsilon) or epsilon != 1.0e-6:
        raise ValueError("inverse-RMSE epsilon must remain 1e-6 ft")
    if not math.isfinite(exponent) or exponent != 1.0:
        raise ValueError("only inverse-RMSE exponent 1.0 is permitted")
    if list(clip or []) != [0.5, 1.5]:
        raise ValueError("candidate task weight clip must remain [0.5, 1.5]")
    if normalization_cfg.get("before_clip") != "divide_by_candidate_mean":
        raise ValueError("pre-clip normalization must divide by the candidate mean")
    if normalization_cfg.get("after_clip") != "divide_by_candidate_mean":
        raise ValueError("post-clip normalization must divide by the candidate mean")
    if not bool(normalization_cfg.get("fail_if_final_range_violated")):
        raise ValueError("final range violations must fail closed")

    objectives = dict(cfg.get("objectives", {}))
    if objectives != {
        "pred_abs_error": "same_candidate_task_weight",
        "p_within10": "same_candidate_task_weight",
    }:
        raise ValueError("both selector objectives must use the same candidate weight")
    if cfg.get("validation_weight") != "none" or cfg.get("metric_weight") != "none":
        raise ValueError("validation and reported metrics must remain unweighted")
    return cfg


@dataclass
class CandidateTaskWeightResult:
    partition_id: int
    candidate_order: list[str]
    sample_weight: np.ndarray
    table: pd.DataFrame
    sampling_manifest: dict[str, Any]
    truth_read_ledger: dict[str, Any]
    audit: dict[str, Any]


def build_inverse_rmse_candidate_task_weights(
    labels: pd.DataFrame,
    candidate_order: Sequence[str],
    *,
    partition_id: int,
    config: Mapping[str, Any],
) -> CandidateTaskWeightResult:
    """Build mean-one inverse-RMSE weights from one exact model fit partition.

    ``labels`` must already be restricted to the deterministic sampled fit rows.
    The function deliberately accepts no validation labels or global OOF table.
    """

    cfg = validate_inverse_rmse_weight_config(config)
    candidates = [str(item) for item in candidate_order]
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidate order must be non-empty and unique")
    missing = [column for column in REQUIRED_LABEL_COLUMNS if column not in labels]
    if missing:
        raise ValueError(f"fit labels are missing required columns: {missing}")

    n_candidates = len(candidates)
    if len(labels) == 0 or len(labels) % n_candidates:
        raise ValueError("candidate-long fit labels do not form complete base-row blocks")
    n_base_rows = len(labels) // n_candidates
    expected_candidate_ids = np.tile(np.asarray(candidates, dtype=object), n_base_rows)
    actual_candidate_ids = labels["candidate_id"].astype(str).to_numpy()
    if not np.array_equal(actual_candidate_ids, expected_candidate_ids):
        raise ValueError("candidate-long order differs from the frozen candidate order")

    for column in FIT_ROW_ID_COLUMNS:
        matrix = labels[column].to_numpy().reshape(n_base_rows, n_candidates)
        if not np.all(matrix == matrix[:, :1]):
            raise ValueError(f"{column} changes within a candidate-long base-row block")

    fit_row_ids = labels.iloc[::n_candidates][FIT_ROW_ID_COLUMNS].reset_index(drop=True)
    if fit_row_ids["id"].astype(str).duplicated().any():
        raise ValueError("fit partition contains duplicate base-row IDs")
    fit_row_id_sha = _logical_frame_sha256(fit_row_ids)

    errors = pd.to_numeric(labels["candidate_abs_error"], errors="coerce").to_numpy(
        np.float64
    )
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("fit-partition candidate errors must be finite and non-negative")
    error_matrix = errors.reshape(n_base_rows, n_candidates)
    candidate_rmse = np.sqrt(np.mean(np.square(error_matrix), axis=0, dtype=np.float64))
    if not np.isfinite(candidate_rmse).all():
        raise ValueError("fit-partition candidate RMSE is non-finite")

    epsilon = float(cfg["rmse"]["epsilon_ft"])
    raw_inverse = 1.0 / np.maximum(candidate_rmse, epsilon)
    normalized = raw_inverse / np.mean(raw_inverse)
    clip_min, clip_max = (float(value) for value in cfg["normalization"]["clip"])
    clipped = np.clip(normalized, clip_min, clip_max)
    final_weight = clipped / np.mean(clipped)
    mean_tolerance = float(cfg["normalization"]["final_mean_absolute_tolerance"])
    final_mean = float(np.mean(final_weight))
    if not np.isfinite(final_weight).all():
        raise ValueError("final candidate task weights are non-finite")
    if abs(final_mean - 1.0) > mean_tolerance:
        raise ValueError("final candidate task weight mean differs from 1.0")
    if (
        float(np.min(final_weight)) < clip_min
        or float(np.max(final_weight)) > clip_max
    ):
        raise ValueError(
            "post-clip mean normalization moved final candidate weights outside "
            "the frozen [0.5, 1.5] range"
        )

    sample_weight = np.tile(final_weight, n_base_rows).astype(np.float64, copy=False)
    if len(sample_weight) != len(labels) or not np.isfinite(sample_weight).all():
        raise AssertionError("candidate task sample-weight vector is invalid")

    source_fold_counts = (
        fit_row_ids.groupby("outer_fold", sort=True).size().astype(int).to_dict()
    )
    table = pd.DataFrame(
        {
            "fit_partition": int(partition_id),
            "candidate_position": np.arange(n_candidates, dtype=np.int16),
            "candidate_id": candidates,
            "fit_candidate_rmse": candidate_rmse,
            "raw_inverse_rmse": raw_inverse,
            "mean_normalized_weight": normalized,
            "clipped_weight": clipped,
            "final_weight": final_weight,
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
        "candidate_order_sha256": _sha256_json(candidates),
    }
    truth_read_ledger = {
        "fit_partition": int(partition_id),
        "weight_input": "candidate_abs_error_from_exact_fit_partition_labels_only",
        "fit_truth_rows": n_base_rows,
        "outer_valid_truth_reads_for_weight": 0,
        "inner_valid_truth_reads_for_weight": 0,
        "hidden_like_truth_reads_for_weight": 0,
        "global_oof_truth_reads_for_weight": 0,
        "current_test_truth_reads_for_weight": 0,
    }
    audit = {
        "fit_partition": int(partition_id),
        "candidate_count": n_candidates,
        "candidate_coverage_errors": 0,
        "missing_or_nonfinite_weight_rows": 0,
        "final_weight_mean": final_mean,
        "final_weight_min": float(np.min(final_weight)),
        "final_weight_max": float(np.max(final_weight)),
        "sample_weight_mean": float(np.mean(sample_weight)),
        "sample_weight_content_sha256": _float64_content_sha256(sample_weight),
        "fit_candidate_abs_error_content_sha256": _float64_content_sha256(errors),
        "validation_sample_weight_applied": False,
        "metric_sample_weight_applied": False,
        "fit_valid_well_overlap": None,
    }
    return CandidateTaskWeightResult(
        partition_id=int(partition_id),
        candidate_order=candidates,
        sample_weight=sample_weight,
        table=table,
        sampling_manifest=sampling_manifest,
        truth_read_ledger=truth_read_ledger,
        audit=audit,
    )


def write_candidate_task_weight_artifacts(
    output_dir: Path,
    results: Sequence[CandidateTaskWeightResult],
) -> dict[str, Any]:
    if not results:
        raise ValueError("no candidate task weight results were provided")
    ordered = sorted(results, key=lambda item: item.partition_id)
    partitions = [item.partition_id for item in ordered]
    if len(set(partitions)) != len(partitions):
        raise ValueError("candidate task weight partitions are duplicated")
    reference_order = ordered[0].candidate_order
    if any(item.candidate_order != reference_order for item in ordered):
        raise ValueError("candidate order changed across fit partitions")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    table = pd.concat([item.table for item in ordered], ignore_index=True)
    table_path = root / "candidate_task_weight_by_fold.csv"
    table.to_csv(table_path, index=False)

    sampling = pd.DataFrame([item.sampling_manifest for item in ordered])
    sampling["source_outer_fold_base_rows"] = sampling[
        "source_outer_fold_base_rows"
    ].map(lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
    sampling["fit_row_id_columns"] = sampling["fit_row_id_columns"].map(
        lambda value: json.dumps(value, separators=(",", ":"))
    )
    sampling["stage_sample_seed_parts"] = sampling["stage_sample_seed_parts"].map(
        lambda value: json.dumps(value, separators=(",", ":"))
    )
    sampling["outer_train_seed_parts"] = sampling["outer_train_seed_parts"].map(
        lambda value: json.dumps(value, separators=(",", ":"))
    )
    sampling_path = root / "candidate_task_weight_sampling_manifest.csv"
    sampling.to_csv(sampling_path, index=False)

    truth_ledger = pd.DataFrame([item.truth_read_ledger for item in ordered])
    truth_path = root / "candidate_task_weight_truth_read_ledger.csv"
    truth_ledger.to_csv(truth_path, index=False)

    audits = [dict(item.audit) for item in ordered]
    manifest = {
        "schema_version": "1.0.0",
        "status": "fit_partition_inverse_rmse_candidate_task_weights_complete",
        "method": "fit_partition_mean_one_clipped_inverse_rmse",
        "candidate_order": reference_order,
        "candidate_count": len(reference_order),
        "fit_partitions": partitions,
        "partition_count": len(partitions),
        "same_weight_for_objectives": ["pred_abs_error", "p_within10"],
        "validation_sample_weight_applied": False,
        "metric_sample_weight_applied": False,
        "table": {
            "path": table_path.name,
            "rows": len(table),
            "sha256": _sha256_file(table_path),
            "logical_sha256": _logical_frame_sha256(table),
        },
        "sampling_manifest": {
            "path": sampling_path.name,
            "rows": len(sampling),
            "sha256": _sha256_file(sampling_path),
        },
        "truth_read_ledger": {
            "path": truth_path.name,
            "rows": len(truth_ledger),
            "sha256": _sha256_file(truth_path),
            "forbidden_truth_reads": int(
                truth_ledger[
                    [
                        "outer_valid_truth_reads_for_weight",
                        "inner_valid_truth_reads_for_weight",
                        "hidden_like_truth_reads_for_weight",
                        "global_oof_truth_reads_for_weight",
                        "current_test_truth_reads_for_weight",
                    ]
                ]
                .to_numpy(np.int64)
                .sum()
            ),
        },
        "audits": audits,
        "all_checks_passed": bool(
            all(
                item["candidate_coverage_errors"] == 0
                and item["missing_or_nonfinite_weight_rows"] == 0
                and item["fit_valid_well_overlap"] == 0
                and abs(float(item["final_weight_mean"]) - 1.0) <= 1.0e-12
                and float(item["final_weight_min"]) >= 0.5
                and float(item["final_weight_max"]) <= 1.5
                and not bool(item["validation_sample_weight_applied"])
                and not bool(item["metric_sample_weight_applied"])
                for item in audits
            )
        ),
    }
    manifest_path = root / "candidate_task_weight_manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False) + "\n"
    )
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    return manifest


__all__ = [
    "CandidateTaskWeightResult",
    "build_inverse_rmse_candidate_task_weights",
    "validate_inverse_rmse_weight_config",
    "write_candidate_task_weight_artifacts",
]

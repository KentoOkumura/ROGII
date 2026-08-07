from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from src.candidate_selector_pipeline import (
    logical_frame_sha256,
    sha256_file,
    sha256_json,
    to_jsonable,
    write_json,
)
from src.well_segment_candidate_divergence import signature_feature_columns

PRIMARY_METRICS = (
    "bank_range_mean",
    "bank_range_p90",
    "pair_abs_gap_mean",
    "pair_abs_gap_p90",
)
AXIS_COLUMNS = ("fixed_range_gap_axis", "pca1_axis")
OUTCOME_COLUMNS = ("actual_mae", "calibration_bias")


def primary_feature_columns() -> list[str]:
    return [
        column
        for column in signature_feature_columns()
        if column.rsplit("__", 1)[-1] in PRIMARY_METRICS
    ]


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return math.nan
    x_rank = rankdata(x[finite], method="average")
    y_rank = rankdata(y[finite], method="average")
    x_std = float(np.std(x_rank))
    y_std = float(np.std(y_rank))
    if x_std <= 0.0 or y_std <= 0.0:
        return math.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def assert_signature_contract(
    signatures: pd.DataFrame,
    *,
    expected_wells: int,
    expected_folds: int,
    expected_features: int,
) -> None:
    features = signature_feature_columns()
    required = {"well", "outer_fold", "eval_rows", *features}
    missing = sorted(required - set(signatures.columns))
    if missing:
        raise ValueError(f"well signature input is incomplete: missing={missing}")
    if len(signatures) != int(expected_wells):
        raise ValueError(
            f"well signature count mismatch: expected={expected_wells}, actual={len(signatures)}"
        )
    if signatures["well"].astype(str).nunique() != int(expected_wells):
        raise ValueError("well signatures must contain exactly one row per well")
    if signatures["outer_fold"].nunique() != int(expected_folds):
        raise ValueError("well signature outer-fold count mismatch")
    if len(features) != int(expected_features):
        raise ValueError("well signature feature count mismatch")
    values = signatures[features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("well signatures contain non-finite values")
    folds = sorted(int(item) for item in signatures["outer_fold"].unique())
    if folds != list(range(int(expected_folds))):
        raise ValueError(f"unexpected outer-fold labels: {folds}")


def fit_oof_continuous_axes(
    signatures: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fit fixed range/gap and PCA1 axes on each outer-train partition."""

    technical = dict(config.get("guards", {}).get("technical", {}))
    assert_signature_contract(
        signatures,
        expected_wells=int(technical["expected_wells"]),
        expected_folds=int(technical["expected_folds"]),
        expected_features=int(technical["expected_features"]),
    )
    features = signature_feature_columns()
    primary = primary_feature_columns()
    primary_indices = np.asarray([features.index(column) for column in primary], dtype=np.int64)
    if len(primary) != int(config.get("axes", {}).get("primary_feature_count", 12)):
        raise ValueError("fixed range/gap feature count does not match config")

    preprocessing = dict(config.get("preprocessing", {}))
    quantile_range = tuple(float(item) for item in preprocessing["quantile_range"])
    clip_low, clip_high = [float(item) for item in preprocessing["scaled_clip"]]
    folds = sorted(int(item) for item in signatures["outer_fold"].unique())
    outputs: list[pd.DataFrame] = []
    preprocessors: list[dict[str, Any]] = []

    for fold in folds:
        train_mask = signatures["outer_fold"].to_numpy(np.int16) != fold
        valid_mask = ~train_mask
        train_raw = signatures.loc[train_mask, features].apply(pd.to_numeric, errors="coerce")
        valid_raw = signatures.loc[valid_mask, features].apply(pd.to_numeric, errors="coerce")
        medians = train_raw.median(axis=0, skipna=True).fillna(0.0)
        train_imputed = train_raw.fillna(medians).to_numpy(np.float64)
        valid_imputed = valid_raw.fillna(medians).to_numpy(np.float64)

        scaler = RobustScaler(quantile_range=quantile_range)
        train_scaled = np.clip(scaler.fit_transform(train_imputed), clip_low, clip_high)
        valid_scaled = np.clip(scaler.transform(valid_imputed), clip_low, clip_high)
        train_primary = np.mean(train_scaled[:, primary_indices], axis=1)
        valid_primary = np.mean(valid_scaled[:, primary_indices], axis=1)

        pca = PCA(n_components=1, svd_solver="full")
        train_pca = pca.fit_transform(train_scaled)[:, 0]
        valid_pca = pca.transform(valid_scaled)[:, 0]
        orientation_correlation = _spearman(train_pca, train_primary)
        if math.isfinite(orientation_correlation) and abs(orientation_correlation) > 1e-12:
            orientation = 1.0 if orientation_correlation > 0.0 else -1.0
            orientation_rule = "outer_train_spearman_with_fixed_range_gap_axis"
        else:
            loading_sum = float(np.sum(pca.components_[0, primary_indices]))
            orientation = 1.0 if loading_sum >= 0.0 else -1.0
            orientation_rule = "outer_train_primary_loading_sum_fallback"
        train_pca *= orientation
        valid_pca *= orientation

        output = signatures.loc[valid_mask, ["well", "outer_fold", "eval_rows"]].copy()
        output["well"] = output["well"].astype(str)
        output["fixed_range_gap_axis"] = valid_primary
        output["pca1_axis"] = valid_pca
        output["signature_imputed_values"] = valid_raw.isna().sum(axis=1).to_numpy(np.int16)
        outputs.append(output)
        preprocessors.append(
            {
                "outer_fold": fold,
                "outer_train_wells": int(train_mask.sum()),
                "outer_valid_wells": int(valid_mask.sum()),
                "feature_columns": features,
                "primary_feature_columns": primary,
                "outer_train_medians": medians.to_dict(),
                "robust_scaler_center": scaler.center_.tolist(),
                "robust_scaler_scale": scaler.scale_.tolist(),
                "quantile_range": list(quantile_range),
                "scaled_clip": [clip_low, clip_high],
                "pca_mean": pca.mean_.tolist(),
                "pca_component_oriented": (pca.components_[0] * orientation).tolist(),
                "pca_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
                "pca_orientation": orientation,
                "pca_orientation_rule": orientation_rule,
                "pca_outer_train_spearman_before_orientation": orientation_correlation,
                "pca_outer_train_spearman_after_orientation": _spearman(
                    train_pca, train_primary
                ),
            }
        )

    axes = pd.concat(outputs, ignore_index=True).sort_values(
        ["outer_fold", "well"], kind="stable"
    ).reset_index(drop=True)
    if len(axes) != len(signatures) or not axes["well"].is_unique:
        raise ValueError("OOF continuous axes do not cover each well exactly once")
    if not np.isfinite(axes[list(AXIS_COLUMNS)].to_numpy(np.float64)).all():
        raise ValueError("OOF continuous axes contain non-finite values")
    return axes, preprocessors


def stream_candidate_well_metrics(
    candidate_score_path: Path,
    signatures: pd.DataFrame,
    candidate_ids: Sequence[str],
    *,
    batch_size: int,
    expected_rows_per_candidate: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream exp264 candidate-long OOF scores into well×candidate metrics."""

    import pyarrow.parquet as pq

    path = Path(candidate_score_path)
    if not path.exists():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    required = {
        "well",
        "outer_fold",
        "candidate_id",
        "pred_abs_error",
        "actual_abs_error",
    }
    missing = sorted(required - set(parquet.schema.names))
    if missing:
        raise ValueError(f"candidate score input is incomplete: missing={missing}")
    columns = sorted(required)
    if "candidate_available" in parquet.schema.names:
        columns.append("candidate_available")

    candidates = tuple(str(item) for item in candidate_ids)
    candidate_set = set(candidates)
    fold_lookup = signatures.set_index(signatures["well"].astype(str))["outer_fold"]
    if not fold_lookup.index.is_unique:
        raise ValueError("signature well lookup must be unique")
    stats: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    rows_by_candidate: dict[str, int] = defaultdict(int)
    batches = 0

    for batch in parquet.iter_batches(batch_size=int(batch_size), columns=columns):
        batches += 1
        frame = batch.to_pandas()
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame = frame[frame["candidate_id"].isin(candidate_set)].copy()
        if "candidate_available" in frame.columns:
            frame = frame[frame["candidate_available"].astype(bool)].copy()
        if frame.empty:
            continue
        frame["well"] = frame["well"].astype(str)
        mapped_fold = fold_lookup.reindex(frame["well"].to_numpy())
        if mapped_fold.isna().any():
            raise ValueError("candidate score contains wells missing from signatures")
        score_fold = pd.to_numeric(frame["outer_fold"], errors="raise").to_numpy(np.int16)
        if not np.array_equal(score_fold, mapped_fold.to_numpy(np.int16)):
            raise ValueError("candidate score outer fold does not match signatures")
        frame["actual_abs_error"] = pd.to_numeric(
            frame["actual_abs_error"], errors="coerce"
        )
        frame["pred_abs_error"] = pd.to_numeric(frame["pred_abs_error"], errors="coerce")
        if not np.isfinite(
            frame[["actual_abs_error", "pred_abs_error"]].to_numpy(np.float64)
        ).all():
            raise ValueError("candidate score contains non-finite outcomes")
        grouped = frame.groupby(["well", "candidate_id"], sort=False).agg(
            rows=("actual_abs_error", "size"),
            actual_sum=("actual_abs_error", "sum"),
            predicted_sum=("pred_abs_error", "sum"),
        )
        for (well, candidate_id), row in grouped.iterrows():
            values = stats[(str(well), str(candidate_id))]
            values[0] += int(row["rows"])
            values[1] += float(row["actual_sum"])
            values[2] += float(row["predicted_sum"])
        for candidate_id, count in frame["candidate_id"].value_counts().items():
            rows_by_candidate[str(candidate_id)] += int(count)

    if set(rows_by_candidate) != candidate_set:
        raise ValueError(
            f"candidate score coverage mismatch: expected={sorted(candidate_set)}, "
            f"actual={sorted(rows_by_candidate)}"
        )
    if expected_rows_per_candidate is not None:
        incomplete = {
            candidate: rows_by_candidate[candidate]
            for candidate in candidates
            if rows_by_candidate[candidate] != int(expected_rows_per_candidate)
        }
        if incomplete:
            raise ValueError(
                "candidate score row coverage mismatch: "
                f"expected={expected_rows_per_candidate}, actual={incomplete}"
            )

    records: list[dict[str, Any]] = []
    for (well, candidate_id), (rows, actual_sum, predicted_sum) in sorted(stats.items()):
        count = int(rows)
        if count <= 0:
            raise ValueError("candidate score produced an empty well aggregate")
        actual_mae = actual_sum / count
        predicted_mean = predicted_sum / count
        records.append(
            {
                "well": well,
                "outer_fold": int(fold_lookup.loc[well]),
                "candidate_id": candidate_id,
                "rows": count,
                "actual_mae": actual_mae,
                "predicted_abs_error_mean": predicted_mean,
                "calibration_bias": predicted_mean - actual_mae,
            }
        )
    metrics = pd.DataFrame.from_records(records).sort_values(
        ["outer_fold", "well", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    expected_pairs = len(signatures) * len(candidates)
    if len(metrics) != expected_pairs:
        raise ValueError(
            f"well-candidate coverage mismatch: expected={expected_pairs}, actual={len(metrics)}"
        )
    counts = metrics.groupby("well")["candidate_id"].nunique()
    if not counts.eq(len(candidates)).all():
        raise ValueError("not every well contains all fixed candidates")
    evidence = {
        "parquet_batches": batches,
        "parquet_row_groups": int(parquet.num_row_groups),
        "candidate_ids": list(candidates),
        "rows_by_candidate": dict(sorted(rows_by_candidate.items())),
        "well_candidate_rows": int(len(metrics)),
        "wells": int(metrics["well"].nunique()),
    }
    return metrics, evidence


def _stratified_bootstrap_spearman(
    frame: pd.DataFrame,
    axis: str,
    outcome: str,
    *,
    n_resamples: int,
    interval: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    fold_indices = [
        group.index.to_numpy(np.int64)
        for _, group in frame.groupby("outer_fold", sort=True)
    ]
    correlations = np.empty(int(n_resamples), dtype=np.float64)
    for index in range(int(n_resamples)):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in fold_indices]
        )
        correlations[index] = _spearman(
            frame.loc[sampled, axis].to_numpy(np.float64),
            frame.loc[sampled, outcome].to_numpy(np.float64),
        )
    finite = correlations[np.isfinite(correlations)]
    alpha = (1.0 - float(interval)) / 2.0
    return {
        "estimate": _spearman(frame[axis], frame[outcome]),
        "lower": float(np.quantile(finite, alpha)) if len(finite) else math.nan,
        "upper": float(np.quantile(finite, 1.0 - alpha)) if len(finite) else math.nan,
        "interval": float(interval),
        "n_resamples": int(n_resamples),
        "n_valid_resamples": int(len(finite)),
        "seed": int(seed),
    }


def build_readout_tables(
    axes: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    merged = candidate_metrics.merge(
        axes[["well", "outer_fold", *AXIS_COLUMNS]],
        on=["well", "outer_fold"],
        how="left",
        validate="many_to_one",
    )
    if merged[list(AXIS_COLUMNS)].isna().any().any():
        raise ValueError("well-candidate metrics are missing continuous axes")
    bank = (
        merged.groupby(["well", "outer_fold"], sort=True)
        .agg(
            eval_rows_per_candidate=("rows", "first"),
            candidate_count=("candidate_id", "nunique"),
            actual_mae=("actual_mae", "mean"),
            predicted_abs_error_mean=("predicted_abs_error_mean", "mean"),
            calibration_bias=("calibration_bias", "mean"),
            fixed_range_gap_axis=("fixed_range_gap_axis", "first"),
            pca1_axis=("pca1_axis", "first"),
        )
        .reset_index()
    )
    expected_candidates = len(config.get("candidate_bank", {}).get("primitive_ids", []))
    if not bank["candidate_count"].eq(expected_candidates).all():
        raise ValueError("candidate-bank well summary is incomplete")

    correlation_records: list[dict[str, Any]] = []
    for axis in AXIS_COLUMNS:
        for outcome in OUTCOME_COLUMNS:
            for fold, group in bank.groupby("outer_fold", sort=True):
                correlation_records.append(
                    {
                        "scope": "candidate_bank_mean",
                        "candidate_id": "all_six_equal_weight",
                        "outer_fold": str(int(fold)),
                        "axis": axis,
                        "outcome": outcome,
                        "wells": int(len(group)),
                        "spearman": _spearman(group[axis], group[outcome]),
                    }
                )
            correlation_records.append(
                {
                    "scope": "candidate_bank_mean",
                    "candidate_id": "all_six_equal_weight",
                    "outer_fold": "all",
                    "axis": axis,
                    "outcome": outcome,
                    "wells": int(len(bank)),
                    "spearman": _spearman(bank[axis], bank[outcome]),
                }
            )
            for candidate_id, candidate_group in merged.groupby("candidate_id", sort=True):
                for fold, group in candidate_group.groupby("outer_fold", sort=True):
                    correlation_records.append(
                        {
                            "scope": "candidate",
                            "candidate_id": str(candidate_id),
                            "outer_fold": str(int(fold)),
                            "axis": axis,
                            "outcome": outcome,
                            "wells": int(len(group)),
                            "spearman": _spearman(group[axis], group[outcome]),
                        }
                    )
                correlation_records.append(
                    {
                        "scope": "candidate",
                        "candidate_id": str(candidate_id),
                        "outer_fold": "all",
                        "axis": axis,
                        "outcome": outcome,
                        "wells": int(len(candidate_group)),
                        "spearman": _spearman(candidate_group[axis], candidate_group[outcome]),
                    }
                )
    correlations = pd.DataFrame.from_records(correlation_records).sort_values(
        ["scope", "candidate_id", "axis", "outcome", "outer_fold"], kind="stable"
    ).reset_index(drop=True)

    bootstrap_cfg = dict(config.get("bootstrap", {}))
    base_seed = int(bootstrap_cfg["seed"])
    bootstrap_records: list[dict[str, Any]] = []
    for axis in AXIS_COLUMNS:
        for outcome in OUTCOME_COLUMNS:
            seed = stable_seed(base_seed, axis, outcome, "candidate_bank_mean")
            record = _stratified_bootstrap_spearman(
                bank,
                axis,
                outcome,
                n_resamples=int(bootstrap_cfg["n_resamples"]),
                interval=float(bootstrap_cfg["interval"]),
                seed=seed,
            )
            bootstrap_records.append(
                {
                    "scope": "candidate_bank_mean",
                    "candidate_id": "all_six_equal_weight",
                    "axis": axis,
                    "outcome": outcome,
                    "wells": int(len(bank)),
                    **record,
                }
            )
    bootstrap = pd.DataFrame.from_records(bootstrap_records)

    quantile_records: list[dict[str, Any]] = []
    quantiles = int(config.get("readout", {}).get("axis_quantiles", 10))
    for axis in AXIS_COLUMNS:
        ordered_rank = bank[axis].rank(method="first")
        buckets = pd.qcut(ordered_rank, q=quantiles, labels=False)
        bucket_frame = bank.assign(axis_quantile=buckets.astype(int))
        for bucket, group in bucket_frame.groupby("axis_quantile", sort=True):
            quantile_records.append(
                {
                    "axis": axis,
                    "axis_quantile": int(bucket),
                    "wells": int(len(group)),
                    "axis_min": float(group[axis].min()),
                    "axis_max": float(group[axis].max()),
                    "axis_mean": float(group[axis].mean()),
                    "actual_mae": float(group["actual_mae"].mean()),
                    "predicted_abs_error_mean": float(
                        group["predicted_abs_error_mean"].mean()
                    ),
                    "calibration_bias": float(group["calibration_bias"].mean()),
                }
            )
    quantile_metrics = pd.DataFrame.from_records(quantile_records)

    guard_cfg = dict(config.get("guards", {}).get("monotonic_risk", {}))
    primary_fold = correlations[
        (correlations["scope"] == "candidate_bank_mean")
        & (correlations["axis"] == "fixed_range_gap_axis")
        & (correlations["outer_fold"] != "all")
    ]
    actual_fold = primary_fold[primary_fold["outcome"] == "actual_mae"]
    calibration_fold = primary_fold[primary_fold["outcome"] == "calibration_bias"]
    epsilon = float(guard_cfg.get("sign_epsilon", 1e-12))
    actual_direction_count = int((actual_fold["spearman"] > epsilon).sum())
    calibration_direction_count = int((calibration_fold["spearman"] < -epsilon).sum())
    actual_bootstrap = bootstrap[
        (bootstrap["axis"] == "fixed_range_gap_axis")
        & (bootstrap["outcome"] == "actual_mae")
    ].iloc[0]
    calibration_bootstrap = bootstrap[
        (bootstrap["axis"] == "fixed_range_gap_axis")
        & (bootstrap["outcome"] == "calibration_bias")
    ].iloc[0]
    required_folds = int(guard_cfg["required_same_direction_folds"])
    min_actual_lower = float(guard_cfg["min_actual_mae_bootstrap_lower"])
    max_calibration_upper = float(guard_cfg["max_calibration_bias_bootstrap_upper"])
    guard = {
        "primary_axis": "fixed_range_gap_axis",
        "pca1_report_only": True,
        "actual_mae_positive_fold_count": actual_direction_count,
        "calibration_bias_negative_fold_count": calibration_direction_count,
        "required_same_direction_folds": required_folds,
        "actual_mae_bootstrap_lower": float(actual_bootstrap["lower"]),
        "min_actual_mae_bootstrap_lower": min_actual_lower,
        "calibration_bias_bootstrap_upper": float(calibration_bootstrap["upper"]),
        "max_calibration_bias_bootstrap_upper": max_calibration_upper,
    }
    guard["actual_mae_guard_pass"] = bool(
        actual_direction_count == required_folds
        and float(actual_bootstrap["lower"]) >= min_actual_lower
    )
    guard["calibration_bias_guard_pass"] = bool(
        calibration_direction_count == required_folds
        and float(calibration_bootstrap["upper"]) <= max_calibration_upper
    )
    guard["continuous_risk_guard_pass"] = bool(
        guard["actual_mae_guard_pass"] and guard["calibration_bias_guard_pass"]
    )
    guard["separate_add_only_candidate_supported"] = guard[
        "continuous_risk_guard_pass"
    ]

    return {
        "well_candidate": merged,
        "by_well": bank,
        "correlations": correlations,
        "bootstrap": bootstrap,
        "quantile_metrics": quantile_metrics,
        "guard": guard,
    }


def save_readout_artifacts(
    *,
    output_dir: Path,
    axes: pd.DataFrame,
    preprocessors: Sequence[Mapping[str, Any]],
    tables: Mapping[str, Any],
    input_evidence: Mapping[str, Any],
    stream_evidence: Mapping[str, Any],
    config: Mapping[str, Any],
    plot_path: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "axes": output_dir / "well_continuous_divergence_axes.parquet",
        "well_candidate": output_dir / "well_candidate_risk_metrics.csv",
        "by_well": output_dir / "well_divergence_readout_by_well.csv",
        "correlations": output_dir / "well_divergence_spearman.csv",
        "bootstrap": output_dir / "well_divergence_bootstrap_intervals.csv",
        "quantile_metrics": output_dir / "well_divergence_quantile_metrics.csv",
        "preprocessors": output_dir / "continuous_axis_preprocessors.json",
        "plot": Path(plot_path),
    }
    axes.to_parquet(paths["axes"], index=False)
    for key in ("well_candidate", "by_well", "correlations", "bootstrap", "quantile_metrics"):
        tables[key].to_csv(paths[key], index=False)
    preprocessor_payload = {"folds": [to_jsonable(dict(item)) for item in preprocessors]}
    write_json(paths["preprocessors"], preprocessor_payload)

    guard_pass = bool(tables["guard"]["continuous_risk_guard_pass"])
    summary = {
        "status": "complete" if guard_pass else "guard_failed",
        "technical": {
            "wells": int(len(axes)),
            "folds": int(axes["outer_fold"].nunique()),
            "signature_features": len(signature_feature_columns()),
            "primary_features": len(primary_feature_columns()),
            "candidates": int(tables["well_candidate"]["candidate_id"].nunique()),
            "well_candidate_rows": int(len(tables["well_candidate"])),
            "missing_axis_values": int(axes[list(AXIS_COLUMNS)].isna().sum().sum()),
        },
        "compute": {
            "variants": 0,
            "lightgbm_configs": 0,
            "folds_trained": 0,
            "boosters": 0,
            "parent_control_retraining": False,
            "inference": False,
            "submission": False,
        },
        "guard": to_jsonable(dict(tables["guard"])),
        "stream": to_jsonable(dict(stream_evidence)),
        "primary_axis_only_for_decision": True,
        "pca1_report_only": True,
    }
    write_json(output_dir / "readout_summary.json", summary)
    artifact_sha = {
        name: sha256_file(path)
        for name, path in paths.items()
        if Path(path).exists()
    }
    reproducibility = {
        "seed": int(config.get("reproducibility", {}).get("seed", 42)),
        "bootstrap_seed": int(config.get("bootstrap", {}).get("seed", 42)),
        "input_evidence": to_jsonable(dict(input_evidence)),
        "feature_schema_sha256": sha256_json(signature_feature_columns()),
        "primary_feature_schema_sha256": sha256_json(primary_feature_columns()),
        "axes_logical_sha256": logical_frame_sha256(axes),
        "by_well_logical_sha256": logical_frame_sha256(tables["by_well"]),
        "preprocessor_sha256": sha256_json(preprocessor_payload),
        "artifact_sha256": artifact_sha,
        "model_manifest_sha256": None,
        "prediction_sha256": None,
        "submission_sha256": None,
        "deterministic_anchor": False,
    }
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    return summary

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from exact_hmm_smoother import (
    load_lgb_prediction_source,
    resolve_existing_file,
    sha256_gzip_decompressed,
    sha256_path,
    to_jsonable,
)
from settings import ExperimentPaths, get_nested, load_config
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

EXPERIMENT_NAME = "exp240_shrinkage_residual_scale_emission_hmm_on_exp218"
OUTPUT_PREFIX = "exp240_shrinkage_residual_scale_emission_hmm_on_exp218"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _input_sha(path: Path) -> dict[str, str]:
    values = {"raw": sha256_path(path)}
    if path.suffix == ".gz":
        values["decompressed"] = sha256_gzip_decompressed(path)
    return values


def _row_index(ids: pd.Series) -> np.ndarray:
    suffix = ids.astype(str).str.rsplit("_", n=1).str[-1]
    values = pd.to_numeric(suffix, errors="coerce")
    if values.isna().any():
        examples = ids.loc[values.isna()].astype(str).head(5).tolist()
        raise ValueError(f"id must end in a numeric row index, examples={examples}")
    return values.to_numpy(np.int64)


def _load_center_and_context(
    paths: ExperimentPaths,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scale_config = dict(get_nested(config, "residual_scale") or {})
    lgb_config = dict(get_nested(config, "lgb_emission") or {})
    sources = dict(lgb_config.get("sources") or {})
    center_source = str(scale_config.get("center_source") or "")
    if center_source not in sources:
        raise KeyError(
            "residual_scale.center_source is not configured in lgb_emission.sources: "
            f"{center_source}"
        )

    center_payload = load_lgb_prediction_source(
        paths.root, center_source, dict(sources[center_source])
    )
    center = center_payload["predictions"].rename("pred_tvt").reset_index()
    center["id"] = center["id"].astype(str)
    if center["id"].duplicated().any():
        raise ValueError("exp218 lgb_mean OOF center has duplicated ids")

    required = [str(value) for value in (scale_config.get("required_context_columns") or [])]
    required_set = set(required)
    expected = {"id", "well", "target", "last_known_tvt", "md_since"}
    if required_set != expected:
        raise ValueError(f"residual-scale required context columns must be {sorted(expected)}")
    context_path = resolve_existing_file(
        paths.root, list(scale_config.get("context_candidates") or [])
    )
    header = pd.read_csv(context_path, nrows=0)
    missing = sorted(required_set.difference(header.columns))
    if missing:
        raise ValueError(f"row context is missing required columns {missing}: {context_path}")
    context = pd.read_csv(context_path, usecols=required, dtype={"id": str, "well": str})
    context["id"] = context["id"].astype(str)
    context["well"] = context["well"].astype(str)
    if context["id"].duplicated().any():
        raise ValueError("row context has duplicated ids")

    merged = context.merge(center, on="id", how="inner", validate="one_to_one")
    if len(merged) != len(context) or len(merged) != len(center):
        missing_center = sorted(set(context["id"]) - set(center["id"]))[:5]
        extra_center = sorted(set(center["id"]) - set(context["id"]))[:5]
        raise ValueError(
            "exp218 OOF and row context must have identical ID coverage: "
            f"context={len(context)} center={len(center)} merged={len(merged)} "
            f"missing_center_examples={missing_center} extra_center_examples={extra_center}"
        )

    for column in ["target", "last_known_tvt", "md_since", "pred_tvt"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    if merged[["target", "last_known_tvt", "md_since", "pred_tvt"]].isna().any().any():
        raise ValueError("residual-scale source contains non-numeric or missing required values")
    if not np.isfinite(
        merged[["target", "last_known_tvt", "md_since", "pred_tvt"]].to_numpy(np.float64)
    ).all():
        raise ValueError("residual-scale source contains non-finite required values")
    if (merged["md_since"] < 0.0).any():
        raise ValueError("md_since must be non-negative")

    merged["_row_index"] = _row_index(merged["id"])
    merged = merged.sort_values(["well", "_row_index"], kind="stable").reset_index(drop=True)
    source_meta = {
        "center": center_payload["meta"],
        "context_path": str(context_path),
        "context_sha256": _input_sha(context_path),
        "rows": int(len(merged)),
        "wells": int(merged["well"].nunique()),
    }
    return merged, source_meta


def _add_target_free_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    work = frame.copy()
    work["true_tvt"] = work["last_known_tvt"] + work["target"]
    work["residual"] = work["pred_tvt"] - work["true_tvt"]
    work["abs_residual"] = np.abs(work["residual"])
    work["log1p_md_since"] = np.log1p(work["md_since"].to_numpy(np.float64))
    work["center_abs_delta"] = np.abs(work["pred_tvt"] - work["last_known_tvt"])
    work["center_abs_step_delta"] = (
        work.groupby("well", sort=False)["pred_tvt"].diff().abs().fillna(work["center_abs_delta"])
    )
    missing = sorted(set(feature_columns).difference(work.columns))
    if missing:
        raise ValueError(f"configured residual-scale features are unavailable: {missing}")
    values = work[feature_columns].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("target-free residual-scale feature matrix contains non-finite values")
    return work


def _calibration_table(frame: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    ranks = frame["sigma_tvt"].rank(method="first")
    work = frame.assign(scale_bin=pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop"))
    rows: list[dict[str, Any]] = []
    for bin_id, group in work.groupby("scale_bin", sort=True, observed=True):
        residual = group["residual"].to_numpy(np.float64)
        sigma = group["sigma_tvt"].to_numpy(np.float64)
        rows.append(
            {
                "scale_bin": int(bin_id),
                "rows": int(len(group)),
                "sigma_min": float(np.min(sigma)),
                "sigma_mean": float(np.mean(sigma)),
                "sigma_p90": float(np.quantile(sigma, 0.90)),
                "abs_error_mean": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "rmse_over_sigma_mean": float(
                    np.sqrt(np.mean(residual**2)) / max(float(np.mean(sigma)), 1e-12)
                ),
            }
        )
    return pd.DataFrame(rows)


def _guard_summary(
    *,
    calibration: pd.DataFrame,
    sigma_floor_rate: float,
    sigma_cap_rate: float,
    strict_fold_separation: bool,
    scale_config: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    guard_config = dict(scale_config.get("guard") or {})
    bottom_rmse = float(calibration.iloc[0]["rmse"])
    top_rmse = float(calibration.iloc[-1]["rmse"])
    top_to_bottom_ratio = top_rmse / max(bottom_rmse, 1e-12)
    spearman = float(frame["sigma_tvt"].corr(frame["abs_residual"], method="spearman"))
    checks = {
        "strict_fold_separation": (
            not bool(guard_config.get("require_strict_fold_separation", True))
        )
        or strict_fold_separation,
        "spearman_abs_error": spearman >= float(guard_config.get("min_spearman_abs_error", 0.0)),
        "top_to_bottom_rmse_ratio": top_to_bottom_ratio
        >= float(guard_config.get("min_top_to_bottom_rmse_ratio", 1.0)),
        "sigma_floor_rate": sigma_floor_rate
        <= float(guard_config.get("max_sigma_floor_rate", 1.0)),
        "sigma_cap_rate": sigma_cap_rate <= float(guard_config.get("max_sigma_cap_rate", 1.0)),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": guard_config,
        "spearman_sigma_vs_abs_error": spearman,
        "bottom_decile_rmse": bottom_rmse,
        "top_decile_rmse": top_rmse,
        "top_to_bottom_rmse_ratio": top_to_bottom_ratio,
        "sigma_floor_rate": sigma_floor_rate,
        "sigma_cap_rate": sigma_cap_rate,
    }


def run_crossfitted_residual_scale(
    *,
    max_wells: int | None = None,
) -> dict[str, Any]:
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    scale_config = dict(get_nested(config, "residual_scale") or {})
    feature_columns = [str(value) for value in (scale_config.get("feature_columns") or [])]
    if not feature_columns:
        raise ValueError("residual_scale.feature_columns must not be empty")
    n_splits = int(scale_config.get("n_splits", 5))
    if n_splits < 2:
        raise ValueError("residual_scale.n_splits must be at least two")

    frame, source_meta = _load_center_and_context(paths, config)
    if max_wells is not None:
        selected_wells = sorted(frame["well"].unique())[: int(max_wells)]
        frame = frame[frame["well"].isin(selected_wells)].copy().reset_index(drop=True)
    if frame["well"].nunique() < n_splits:
        raise ValueError("residual-scale selected wells are fewer than n_splits")
    frame = _add_target_free_features(frame, feature_columns)

    residual_clip = float(scale_config.get("squared_residual_clip", 100.0))
    sigma_floor = float(scale_config.get("sigma_floor", 2.5))
    sigma_cap = float(scale_config.get("sigma_cap", 40.0))
    if residual_clip <= 0.0 or sigma_floor <= 0.0 or sigma_cap < sigma_floor:
        raise ValueError("invalid residual-scale residual clip or sigma floor/cap")
    target_log_sq = np.log1p(
        np.minimum(frame["residual"].to_numpy(np.float64) ** 2, residual_clip**2)
    )
    x_values = frame[feature_columns].to_numpy(np.float64)
    groups = frame["well"].astype(str).to_numpy()

    estimator_config = dict(scale_config.get("estimator") or {})
    allowed_estimator_keys = {
        "loss",
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "early_stopping",
        "random_state",
    }
    estimator_kwargs = {
        key: value for key, value in estimator_config.items() if key in allowed_estimator_keys
    }
    oof_log_sq = np.full(len(frame), np.nan, dtype=np.float64)
    fold_index = np.full(len(frame), -1, dtype=np.int16)
    fold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(x_values, target_log_sq, groups=groups)
    ):
        train_wells = set(groups[train_idx].tolist())
        valid_wells = set(groups[valid_idx].tolist())
        overlap = sorted(train_wells.intersection(valid_wells))
        if overlap:
            raise RuntimeError(
                f"residual-scale fold {fold} has train/valid well overlap: {overlap[:5]}"
            )
        estimator = HistGradientBoostingRegressor(**estimator_kwargs)
        estimator.fit(x_values[train_idx], target_log_sq[train_idx])
        oof_log_sq[valid_idx] = estimator.predict(x_values[valid_idx])
        fold_index[valid_idx] = fold
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "train_wells": int(len(train_wells)),
                "valid_wells": int(len(valid_wells)),
                "well_overlap_count": int(len(overlap)),
                "valid_log_squared_residual_mean": float(np.mean(target_log_sq[valid_idx])),
                "predicted_log_squared_residual_mean": float(np.mean(oof_log_sq[valid_idx])),
            }
        )
    if np.isnan(oof_log_sq).any() or (fold_index < 0).any():
        raise RuntimeError(
            "residual-scale cross-fit did not produce exactly one held-out prediction per row"
        )

    sigma_raw = np.sqrt(np.maximum(np.expm1(oof_log_sq), 0.0))
    sigma_tvt = np.clip(sigma_raw, sigma_floor, sigma_cap)
    if not np.isfinite(sigma_tvt).all() or float(np.min(sigma_tvt)) <= 0.0:
        raise RuntimeError("residual-scale sigma contains non-finite or non-positive values")
    frame["residual_scale_fold"] = fold_index
    frame["sigma_raw_tvt"] = sigma_raw
    frame["sigma_tvt"] = sigma_tvt
    frame["sigma_floor_clamped"] = sigma_raw <= sigma_floor
    frame["sigma_cap_clamped"] = sigma_raw >= sigma_cap

    calibration = _calibration_table(frame, int(scale_config.get("calibration_bins", 10)))
    sigma_floor_rate = float(frame["sigma_floor_clamped"].mean())
    sigma_cap_rate = float(frame["sigma_cap_clamped"].mean())
    strict_fold_separation = all(row["well_overlap_count"] == 0 for row in fold_rows)
    guard = _guard_summary(
        calibration=calibration,
        sigma_floor_rate=sigma_floor_rate,
        sigma_cap_rate=sigma_cap_rate,
        strict_fold_separation=strict_fold_separation,
        scale_config=scale_config,
        frame=frame,
    )

    prediction_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_residual_scale_predictions.csv.gz"
    calibration_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_residual_scale_calibration.csv"
    folds_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_residual_scale_folds.csv"
    schema_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_residual_scale_feature_schema.csv"
    summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_residual_scale_summary.json"
    output_columns = [
        "id",
        "well",
        "residual_scale_fold",
        "target",
        "last_known_tvt",
        "md_since",
        "pred_tvt",
        "true_tvt",
        "residual",
        "abs_residual",
        "sigma_raw_tvt",
        "sigma_tvt",
        "sigma_floor_clamped",
        "sigma_cap_clamped",
        *feature_columns,
    ]
    frame[output_columns].to_csv(prediction_path, index=False, compression="gzip")
    calibration.to_csv(calibration_path, index=False)
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False)
    pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_columns), dtype=np.int16),
            "feature": feature_columns,
        }
    ).to_csv(schema_path, index=False)

    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "crossfitted_residual_scale_completed",
        "mode": "saved_exp218_oof_center_inner_groupkfold_residual_scale",
        "source": source_meta,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "feature_columns": feature_columns,
        "n_splits": n_splits,
        "estimator": {"name": "HistGradientBoostingRegressor", **estimator_kwargs},
        "residual_target": {
            "transform": "log1p(clipped_squared_residual)",
            "squared_residual_clip": residual_clip,
        },
        "sigma": {
            "floor": sigma_floor,
            "cap": sigma_cap,
            "raw_min": float(np.min(sigma_raw)),
            "raw_mean": float(np.mean(sigma_raw)),
            "raw_p90": float(np.quantile(sigma_raw, 0.90)),
            "raw_max": float(np.max(sigma_raw)),
            "effective_min": float(np.min(sigma_tvt)),
            "effective_mean": float(np.mean(sigma_tvt)),
            "effective_p90": float(np.quantile(sigma_tvt, 0.90)),
            "effective_max": float(np.max(sigma_tvt)),
            "floor_rate": sigma_floor_rate,
            "cap_rate": sigma_cap_rate,
        },
        "guard": guard,
        "folds": fold_rows,
        "outputs": {
            "predictions": prediction_path.name,
            "calibration": calibration_path.name,
            "folds": folds_path.name,
            "feature_schema": schema_path.name,
            "summary": summary_path.name,
        },
        "sha256": {
            "predictions_gzip": sha256_path(prediction_path),
            "predictions_decompressed": sha256_gzip_decompressed(prediction_path),
            "calibration": sha256_path(calibration_path),
            "folds": sha256_path(folds_path),
            "feature_schema": sha256_path(schema_path),
        },
        "notes": [
            "exp218 lgb_mean OOF is an immutable center and is never retrained in this experiment.",
            "Each sigma is produced by a GroupKFold model that excludes the row's entire well.",
            "Unknown-suffix TVT is used only as a residual-scale training target "
            "and diagnostic readout.",
        ],
    }
    write_json(summary_path, summary)
    summary["sha256"]["summary"] = sha256_path(summary_path)
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run_crossfitted_residual_scale()

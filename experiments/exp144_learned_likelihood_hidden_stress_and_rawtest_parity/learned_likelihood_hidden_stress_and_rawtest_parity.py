from __future__ import annotations

import gzip
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, is_kaggle_runtime, load_config

OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")

EXP127_PREDICTIONS = "exp127_learned_likelihood_features_on_exp092_predictions.csv.gz"
EXP127_SCHEMA = "exp127_learned_likelihood_features_on_exp092_feature_schema.csv"
EXP127_SUMMARY = "exp127_learned_likelihood_features_on_exp092_summary.json"
EXP112_ML_FEATURES = "exp112_learned_pf_likelihood_weight_or_feature_followup_ml_features.csv.gz"
EXP112_FEATURE_SCHEMA = "exp112_learned_pf_likelihood_weight_or_feature_followup_feature_schema.csv"
EXP112_SUMMARY = "exp112_learned_pf_likelihood_weight_or_feature_followup_summary.json"
EXP115_FOLD_ASSIGNMENTS = "exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv"
EXP115_WELL_METADATA = "exp115_hidden_like_spatial_holdout_from_ppt_well_metadata.csv"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except TypeError:
        pass
    return value


def is_gzip_path(path: Path) -> bool:
    return path.suffix == ".gz" or "".join(path.suffixes[-2:]) == ".csv.gz"


def sha256_file(path: str | Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_paths(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, str | Path):
        return [Path(value)]
    if isinstance(value, list | tuple):
        return [Path(item) for item in value if item]
    return []


def resolve_candidate(paths: ExperimentPaths, value: str | Path) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    return path if path.is_absolute() else paths.root / path


def find_input_file(
    paths: ExperimentPaths,
    filename: str,
    configured: Any = None,
    *,
    local_roots: list[Path] | None = None,
    required: bool = True,
) -> Path | None:
    candidates: list[Path] = []
    candidates.extend(resolve_candidate(paths, item) for item in _as_paths(configured))
    for root in local_roots or []:
        root_path = resolve_candidate(paths, root)
        candidates.extend([root_path / filename, root_path / "artifacts" / filename])
    candidates.extend([Path.cwd() / filename, Path.cwd() / "artifacts" / filename])
    if is_kaggle_runtime() and KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")))
    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    if required:
        raise FileNotFoundError(
            f"input file not found or empty: {filename}. Checked:\n" + "\n".join(checked[:120])
        )
    return None


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open() as fp:
        value = json.load(fp)
    return value if isinstance(value, dict) else {}


def rmse_from_error(error: pd.Series | np.ndarray) -> float:
    values = np.asarray(error, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values[finite]))))


def parse_row_index(ids: pd.Series) -> pd.Series:
    suffix = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    return pd.to_numeric(suffix, errors="coerce")


def safe_cut(
    values: pd.Series | np.ndarray,
    bins: list[float],
    labels: list[str],
) -> pd.Series:
    return (
        pd.cut(
            pd.to_numeric(values, errors="coerce"),
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )
        .astype("string")
        .fillna("unknown")
    )


def safe_qcut(values: pd.Series, q: int, *, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    out = pd.Series("missing", index=values.index, dtype="object")
    if finite.nunique(dropna=True) <= 1:
        out.loc[finite.index] = f"{prefix}_single"
        return out
    try:
        cut = pd.qcut(finite, q=min(q, int(finite.nunique())), duplicates="drop")
    except ValueError:
        out.loc[finite.index] = f"{prefix}_single"
        return out
    labels = {interval: f"{prefix}_q{idx + 1}" for idx, interval in enumerate(cut.cat.categories)}
    out.loc[finite.index] = cut.map(labels).astype(str)
    return out


def metric_table(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_cols, dropna=False, observed=True)
        .agg(
            rows=("squared_error", "size"),
            wells=("well", "nunique"),
            sse=("squared_error", "sum"),
            error_sum=("error", "sum"),
            abs_error_sum=("abs_error", "sum"),
        )
        .reset_index()
    )
    grouped["rmse_tvt"] = np.sqrt(grouped["sse"] / grouped["rows"].clip(lower=1))
    grouped["error_mean"] = grouped["error_sum"] / grouped["rows"].clip(lower=1)
    grouped["error_abs_mean"] = grouped["abs_error_sum"] / grouped["rows"].clip(lower=1)
    return grouped.drop(columns=["error_sum", "abs_error_sum"])


def sort_output(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_cols = [
        col
        for col in [
            "split_variant",
            "variant",
            "mode",
            "model",
            "bucket_family",
            "bucket",
            "well",
            "check",
        ]
        if col in frame.columns
    ]
    return frame.sort_values(sort_cols).reset_index(drop=True) if sort_cols else frame


def load_exp115_context(
    paths: ExperimentPaths,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_path = find_input_file(
        paths,
        EXP115_FOLD_ASSIGNMENTS,
        get_nested(config, "data.exp115_fold_assignments"),
        local_roots=[
            Path("experiments/exp115_hidden_like_spatial_holdout_from_ppt/artifacts"),
            Path("experiments/exp115_hidden_like_spatial_holdout_from_ppt/kaggle/output/train_v1"),
        ],
    )
    meta_path = find_input_file(
        paths,
        EXP115_WELL_METADATA,
        get_nested(config, "data.exp115_well_metadata"),
        local_roots=[
            Path("experiments/exp115_hidden_like_spatial_holdout_from_ppt/artifacts"),
            Path("experiments/exp115_hidden_like_spatial_holdout_from_ppt/kaggle/output/train_v1"),
        ],
    )
    assert fold_path is not None and meta_path is not None
    folds = pd.read_csv(fold_path, dtype={"well_id": str})
    meta = pd.read_csv(meta_path, dtype={"well_id": str})
    role_cols = [col for col in folds.columns if col.endswith("_role")]
    drop_cols = [col for col in role_cols if col in meta.columns]
    merged = meta.drop(columns=drop_cols, errors="ignore").merge(
        folds[["well_id", *role_cols]],
        on="well_id",
        how="left",
        validate="one_to_one",
    )
    if "typewell_group_size" in merged.columns:
        bins = [0, 1, 2, 3, 5, 10, math.inf]
        labels = ["size_1", "size_2", "size_3", "size_4_5", "size_6_10", "size_11_plus"]
        merged["typewell_group_size_bin"] = safe_cut(merged["typewell_group_size"], bins, labels)
    metadata = {
        "fold_assignments_path": str(fold_path),
        "well_metadata_path": str(meta_path),
        "fold_assignments_sha256": sha256_file(fold_path),
        "well_metadata_sha256": sha256_file(meta_path),
        "rows": int(len(merged)),
        "role_columns": role_cols,
    }
    return merged, metadata


def load_exp112_features(
    paths: ExperimentPaths,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_path = find_input_file(
        paths,
        EXP112_ML_FEATURES,
        get_nested(config, "data.exp112_ml_features"),
        local_roots=[
            Path(
                "experiments/exp112_learned_pf_likelihood_weight_or_feature_followup/kaggle/output/train_v1"
            )
        ],
    )
    schema_path = find_input_file(
        paths,
        EXP112_FEATURE_SCHEMA,
        get_nested(config, "data.exp112_feature_schema"),
        local_roots=[
            Path(
                "experiments/exp112_learned_pf_likelihood_weight_or_feature_followup/kaggle/output/train_v1"
            )
        ],
    )
    summary_path = find_input_file(
        paths,
        EXP112_SUMMARY,
        get_nested(config, "data.exp112_summary"),
        local_roots=[
            Path(
                "experiments/exp112_learned_pf_likelihood_weight_or_feature_followup/kaggle/output/train_v1"
            )
        ],
    )
    assert feature_path is not None and schema_path is not None
    usecols = [
        "id",
        "well",
        "fold",
        "md_since",
        "learned_prob_margin_top1_top2",
        "learned_prob_entropy",
        "learned_error_likpf_rank",
        "candidate_tvt_std",
        "candidate_tvt_range",
        "learned_prob_likpf_mean",
        "learned_pred_abs_error_likpf_mean",
    ]
    header = pd.read_csv(feature_path, nrows=0).columns.tolist()
    missing = sorted(set(usecols) - set(header))
    if missing:
        raise ValueError(f"{feature_path} missing exp112 feature columns: {missing}")
    frame = pd.read_csv(feature_path, usecols=usecols, dtype={"id": str, "well": str})
    for col in frame.columns:
        if col not in {"id", "well"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    metadata = {
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "feature_decompressed_sha256": sha256_file(feature_path, decompressed=True),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "summary_path": str(summary_path) if summary_path else None,
        "summary_sha256": sha256_file(summary_path) if summary_path else None,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": int(len(header)),
    }
    return frame, metadata


def load_exp127_predictions(
    paths: ExperimentPaths,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred_path = find_input_file(
        paths,
        EXP127_PREDICTIONS,
        get_nested(config, "data.exp127_predictions"),
        local_roots=[
            Path("experiments/exp127_learned_likelihood_features_on_exp092/kaggle/output/train_v1")
        ],
    )
    schema_path = find_input_file(
        paths,
        EXP127_SCHEMA,
        get_nested(config, "data.exp127_feature_schema"),
        local_roots=[
            Path("experiments/exp127_learned_likelihood_features_on_exp092/kaggle/output/train_v1")
        ],
    )
    summary_path = find_input_file(
        paths,
        EXP127_SUMMARY,
        get_nested(config, "data.exp127_summary"),
        local_roots=[
            Path("experiments/exp127_learned_likelihood_features_on_exp092/kaggle/output/train_v1")
        ],
    )
    assert pred_path is not None and schema_path is not None
    models = set(get_nested(config, "audit.models") or ["lgb0", "lgb1", "lgb2", "lgb_mean"])
    usecols = [
        "id",
        "well",
        "variant",
        "mode",
        "model",
        "last_known_tvt",
        "target_tvt",
        "pred_tvt",
    ]
    frame = pd.read_csv(pred_path, usecols=usecols, dtype={"id": str, "well": str})
    frame = frame.loc[frame["model"].astype(str).isin(models)].copy()
    for col in ["last_known_tvt", "target_tvt", "pred_tvt"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[np.isfinite(frame["target_tvt"]) & np.isfinite(frame["pred_tvt"])].copy()
    metadata = {
        "prediction_path": str(pred_path),
        "prediction_sha256": sha256_file(pred_path),
        "prediction_decompressed_sha256": sha256_file(pred_path, decompressed=True),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "summary_path": str(summary_path) if summary_path else None,
        "summary_sha256": sha256_file(summary_path) if summary_path else None,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "variants": sorted(frame["variant"].astype(str).unique().tolist()),
        "models": sorted(frame["model"].astype(str).unique().tolist()),
    }
    return frame, metadata


def add_context(
    predictions: pd.DataFrame,
    exp112_features: pd.DataFrame,
    exp115_meta: pd.DataFrame,
) -> pd.DataFrame:
    feature_cols = [col for col in exp112_features.columns if col not in {"well"}]
    merged = predictions.merge(
        exp112_features[feature_cols],
        on="id",
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        exp115_meta,
        left_on="well",
        right_on="well_id",
        how="left",
        validate="many_to_one",
    )
    merged["row_index"] = parse_row_index(merged["id"])
    merged["eval_rank"] = merged["row_index"] - pd.to_numeric(
        merged.get("first_eval_row"), errors="coerce"
    )
    merged["error"] = merged["pred_tvt"] - merged["target_tvt"]
    merged["abs_error"] = merged["error"].abs()
    merged["squared_error"] = merged["error"] ** 2
    merged["eval_rank_bucket"] = safe_cut(
        merged["eval_rank"],
        [-math.inf, 50, 100, 250, 500, 1000, math.inf],
        ["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
    )
    merged["md_since_bucket"] = safe_cut(
        merged["md_since"],
        [-math.inf, 50, 100, 250, 500, 1000, math.inf],
        ["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
    )
    merged["ll_entropy_bucket"] = safe_qcut(
        merged["learned_prob_entropy"], 4, prefix="ll_entropy"
    )
    merged["ll_margin_bucket"] = safe_qcut(
        merged["learned_prob_margin_top1_top2"], 4, prefix="ll_margin"
    )
    merged["candidate_range_bucket"] = safe_qcut(
        merged["candidate_tvt_range"], 4, prefix="candidate_range"
    )
    return merged


def split_frames(frame: pd.DataFrame, config: dict[str, Any]) -> list[tuple[str, pd.DataFrame]]:
    outputs: list[tuple[str, pd.DataFrame]] = [("all_shared_rows", frame)]
    for split in get_nested(config, "audit.hidden_like_split_variants") or []:
        role_col = f"{split}_role"
        if role_col not in frame.columns:
            outputs.append((split, frame.iloc[0:0].copy()))
            continue
        outputs.append((split, frame.loc[frame[role_col].astype(str) == "valid"].copy()))
    return outputs


def build_metrics(
    context: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_parts: list[pd.DataFrame] = []
    bucket_parts: list[pd.DataFrame] = []
    by_well_parts: list[pd.DataFrame] = []
    base_cols = ["split_variant", "variant", "mode", "model"]
    bucket_cols = list(get_nested(config, "audit.bucket_columns") or [])
    for split, split_frame in split_frames(context, config):
        if split_frame.empty:
            continue
        split_frame = split_frame.copy()
        split_frame["split_variant"] = split
        overall_parts.append(metric_table(split_frame, base_cols))
        by_well_parts.append(metric_table(split_frame, base_cols + ["well"]))
        for bucket_col in bucket_cols:
            if bucket_col not in split_frame.columns:
                continue
            bucket_frame = split_frame.copy()
            bucket_frame["bucket_family"] = bucket_col
            bucket_frame["bucket"] = bucket_frame[bucket_col].astype(str)
            bucket_parts.append(metric_table(bucket_frame, base_cols + ["bucket_family", "bucket"]))
    overall = pd.concat(overall_parts, ignore_index=True) if overall_parts else pd.DataFrame()
    bucket = pd.concat(bucket_parts, ignore_index=True) if bucket_parts else pd.DataFrame()
    by_well = pd.concat(by_well_parts, ignore_index=True) if by_well_parts else pd.DataFrame()
    return overall, bucket, by_well


def build_deltas(
    overall: pd.DataFrame,
    bucket: pd.DataFrame,
    by_well: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    control = str(get_nested(config, "audit.control_variant") or "exp092_shared_row_control")
    addonly = str(
        get_nested(config, "audit.addonly_variant") or "learned_likelihood_confidence_addonly"
    )

    def delta_table(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        control_cols = keys + ["rows", "wells", "rmse_tvt", "sse", "error_abs_mean"]
        base = frame.loc[frame["variant"].astype(str) == control, control_cols].copy()
        test = frame.loc[frame["variant"].astype(str) == addonly].copy()
        if base.empty or test.empty:
            return pd.DataFrame()
        base = base.rename(
            columns={
                "rows": "control_rows",
                "wells": "control_wells",
                "rmse_tvt": "control_rmse_tvt",
                "sse": "control_sse",
                "error_abs_mean": "control_error_abs_mean",
            }
        )
        merged = test.merge(base, on=keys, how="inner", validate="many_to_one")
        merged["delta_rmse_addonly_minus_control"] = (
            merged["rmse_tvt"] - merged["control_rmse_tvt"]
        )
        merged["delta_sse_addonly_minus_control"] = merged["sse"] - merged["control_sse"]
        merged["delta_error_abs_mean_addonly_minus_control"] = (
            merged["error_abs_mean"] - merged["control_error_abs_mean"]
        )
        return merged

    overall_delta = delta_table(overall, ["split_variant", "mode", "model"])
    bucket_delta = delta_table(
        bucket,
        ["split_variant", "mode", "model", "bucket_family", "bucket"],
    )
    by_well_delta = delta_table(by_well, ["split_variant", "mode", "model", "well"])
    return overall_delta, bucket_delta, by_well_delta


def build_rawtest_parity_checklist(
    paths: ExperimentPaths,
    config: dict[str, Any],
    input_meta: dict[str, Any],
    exp127_summary: dict[str, Any],
    exp112_summary: dict[str, Any],
    exp127_schema: pd.DataFrame,
    exp112_feature_header: list[str],
    context: pd.DataFrame,
) -> pd.DataFrame:
    sample_path = paths.sample_submission_path
    test_dir = paths.test_data_dir
    test_files = sorted(test_dir.glob("*__horizontal_well.csv")) if test_dir.exists() else []
    base_wells = int(
        exp127_summary.get("shared_row_filter", {}).get(
            "base_wells_before_shared_filter",
            exp127_summary.get("feature_source", {}).get("wells", 0),
        )
        or 0
    )
    exp112_wells = int(input_meta["exp112"]["wells"])
    exp127_variants = exp127_schema["variant"].astype(str).unique().tolist()
    addonly_features = int(
        exp127_schema.loc[
            exp127_schema["variant"].astype(str)
            == str(get_nested(config, "audit.addonly_variant")),
            "feature",
        ].nunique()
    )
    expected_addonly_features = int(
        get_nested(config, "audit.expected_addonly_feature_count") or 294
    )
    spatial_valid_wells = int(
        context.loc[
            context["verification_like_spatial_role"].astype(str) == "valid",
            "well",
        ].nunique()
    )
    rows: list[dict[str, Any]] = [
        {
            "check": "exp127_oof_predictions_available",
            "status": "pass" if input_meta["exp127"]["rows"] > 0 else "fail",
            "observed": input_meta["exp127"]["rows"],
            "expected": ">0",
            "notes": "Uses saved exp127 row-level OOF predictions; no model is retrained.",
        },
        {
            "check": "exp127_addonly_schema_feature_count",
            "status": "pass" if addonly_features == expected_addonly_features else "fail",
            "observed": addonly_features,
            "expected": expected_addonly_features,
            "notes": f"Schema variants: {', '.join(sorted(exp127_variants))}.",
        },
        {
            "check": "exp112_feature_cache_schema",
            "status": "pass" if len(exp112_feature_header) >= 51 else "fail",
            "observed": len(exp112_feature_header),
            "expected": ">=51 including id/well/fold/md_since and learned likelihood fields",
            "notes": "This is the train-side ML feature cache consumed by exp127.",
        },
        {
            "check": "exp112_feature_cache_full_train_coverage",
            "status": "pass" if base_wells and exp112_wells == base_wells else "fail",
            "observed": f"{exp112_wells}/{base_wells or 'unknown'} wells",
            "expected": "all exp072/exp092 train wells",
            "notes": (
                "exp127 was a 155-well shared-row subset; this does not support "
                "full-train anchor updates."
            ),
        },
        {
            "check": "exp115_hidden_like_overlap",
            "status": "pass" if spatial_valid_wells > 0 else "fail",
            "observed": spatial_valid_wells,
            "expected": ">0 valid wells overlapping exp127 shared rows",
            "notes": "Overlap is a stress readout only; it is not a refit on exp115 split.",
        },
        {
            "check": "raw_test_files_available",
            "status": "pass" if sample_path.exists() and test_files else "fail",
            "observed": f"sample_submission={sample_path.exists()}, test_files={len(test_files)}",
            "expected": "sample submission and raw test well files available",
            "notes": "Availability alone is not learned-likelihood feature parity.",
        },
        {
            "check": "exp112_raw_test_feature_regeneration",
            "status": "fail",
            "observed": "no exp112 inference feature generator or raw-test ml_features artifact",
            "expected": (
                "target-free raw-test feature generator with schema parity to exp112/exp127"
            ),
            "notes": (
                "exp112 inference explicitly produces no submission and requires "
                "raw-test parity first."
            ),
        },
        {
            "check": "hidden_submission_candidate",
            "status": "fail",
            "observed": "not selected",
            "expected": "do not submit from this audit",
            "notes": str(
                exp112_summary.get("decision", {}).get("recommendation", "no direct submit")
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_decision(
    overall_delta: pd.DataFrame,
    by_well_delta: pd.DataFrame,
    checklist: pd.DataFrame,
) -> dict[str, Any]:
    hidden_splits = ["verification_like_spatial", "verification_like_typewell_purged"]
    focus = overall_delta[
        (overall_delta["model"].astype(str) == "lgb_mean")
        & (overall_delta["split_variant"].astype(str).isin(hidden_splits))
    ].copy()
    hidden_supported = bool(
        not focus.empty and (focus["delta_rmse_addonly_minus_control"] < 0).all()
    )
    worst = by_well_delta[
        (by_well_delta["model"].astype(str) == "lgb_mean")
        & (by_well_delta["split_variant"].astype(str).isin(hidden_splits))
    ]
    max_regression = (
        float(worst["delta_rmse_addonly_minus_control"].max()) if not worst.empty else None
    )
    parity_pass = bool((checklist["status"].astype(str) == "pass").all())
    if hidden_supported and parity_pass:
        recommendation = "feature_family_reusable_after_review_no_submit"
    elif hidden_supported:
        recommendation = "hidden_like_supported_but_rawtest_parity_missing"
    else:
        recommendation = "diagnostic_only_hidden_like_or_parity_not_supported"
    return {
        "hidden_like_lgb_mean_supported": hidden_supported,
        "rawtest_parity_all_pass": parity_pass,
        "max_hidden_like_well_regression_lgb_mean": max_regression,
        "direct_submission_candidate": "not_selected",
        "recommendation": recommendation,
    }


def write_outputs(
    paths: ExperimentPaths,
    config: dict[str, Any],
    input_meta: dict[str, Any],
    overall: pd.DataFrame,
    bucket: pd.DataFrame,
    by_well: pd.DataFrame,
    overall_delta: pd.DataFrame,
    bucket_delta: pd.DataFrame,
    by_well_delta: pd.DataFrame,
    checklist: pd.DataFrame,
    decision: dict[str, Any],
) -> dict[str, Any]:
    paths.ensure_output_dirs()
    output_prefix = str(get_nested(config, "audit.output_prefix") or OUTPUT_PREFIX)
    files = {
        "overall_metrics": paths.artifacts_dir / f"{output_prefix}_overall_metrics.csv",
        "bucket_metrics": paths.artifacts_dir / f"{output_prefix}_bucket_metrics.csv",
        "by_well": paths.artifacts_dir / f"{output_prefix}_by_well.csv",
        "overall_delta": paths.artifacts_dir / f"{output_prefix}_overall_delta.csv",
        "bucket_delta": paths.artifacts_dir / f"{output_prefix}_bucket_delta.csv",
        "worst_well_delta": paths.artifacts_dir / f"{output_prefix}_worst_well_delta.csv",
        "rawtest_parity_checklist": paths.artifacts_dir
        / f"{output_prefix}_rawtest_parity_checklist.csv",
        "summary": paths.artifacts_dir / f"{output_prefix}_summary.json",
    }
    sort_output(overall).to_csv(files["overall_metrics"], index=False)
    sort_output(bucket).to_csv(files["bucket_metrics"], index=False)
    sort_output(by_well).to_csv(files["by_well"], index=False)
    sort_output(overall_delta).to_csv(files["overall_delta"], index=False)
    sort_output(bucket_delta).to_csv(files["bucket_delta"], index=False)
    sort_output(by_well_delta).to_csv(files["worst_well_delta"], index=False)
    sort_output(checklist).to_csv(files["rawtest_parity_checklist"], index=False)
    artifact_sha256 = {
        key: sha256_file(path, decompressed=is_gzip_path(path))
        for key, path in files.items()
        if key != "summary" and path.exists()
    }
    focus_rows = []
    if not overall_delta.empty:
        focus = overall_delta[
            (overall_delta["model"].astype(str) == "lgb_mean")
            & overall_delta["split_variant"].astype(str).isin(
                [
                    "all_shared_rows",
                    "verification_like_spatial",
                    "verification_like_typewell_purged",
                ]
            )
        ]
        focus_rows = focus[
            [
                "split_variant",
                "mode",
                "model",
                "rows",
                "rmse_tvt",
                "control_rmse_tvt",
                "delta_rmse_addonly_minus_control",
            ]
        ].to_dict("records")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": get_nested(config, "audit.mode"),
        "route": get_nested(config, "experiment.route"),
        "retrain_model": False,
        "input_meta": input_meta,
        "lgb_mean_delta_focus": focus_rows,
        "rawtest_parity_status_counts": checklist["status"].value_counts().to_dict(),
        "decision": decision,
        "artifacts": {key: str(path) for key, path in files.items()},
        "artifact_sha256": artifact_sha256,
    }
    files["summary"].write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    summary["artifact_sha256"]["summary"] = sha256_file(files["summary"])
    files["summary"].write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> dict[str, Any]:
    config = load_config()
    paths = ExperimentPaths()
    exp115_meta, exp115_input = load_exp115_context(paths, config)
    exp112_features, exp112_input = load_exp112_features(paths, config)
    predictions, exp127_input = load_exp127_predictions(paths, config)

    exp127_summary_path = (
        Path(exp127_input["summary_path"]) if exp127_input["summary_path"] else None
    )
    exp112_summary_path = (
        Path(exp112_input["summary_path"]) if exp112_input["summary_path"] else None
    )
    exp127_summary = load_json(exp127_summary_path)
    exp112_summary = load_json(exp112_summary_path)
    exp127_schema = pd.read_csv(exp127_input["schema_path"])
    exp112_header = pd.read_csv(exp112_input["feature_path"], nrows=0).columns.tolist()

    context = add_context(predictions, exp112_features, exp115_meta)
    overall, bucket, by_well = build_metrics(context, config)
    overall_delta, bucket_delta, by_well_delta = build_deltas(overall, bucket, by_well, config)
    input_meta = {
        "exp115": exp115_input,
        "exp112": exp112_input,
        "exp127": exp127_input,
        "joined_context": {
            "rows": int(len(context)),
            "wells": int(context["well"].nunique()),
            "missing_exp112_rows": int(context["md_since"].isna().sum()),
            "missing_exp115_meta_rows": int(context["well_id"].isna().sum()),
        },
    }
    checklist = build_rawtest_parity_checklist(
        paths,
        config,
        input_meta,
        exp127_summary,
        exp112_summary,
        exp127_schema,
        exp112_header,
        context,
    )
    decision = build_decision(overall_delta, by_well_delta, checklist)
    summary = write_outputs(
        paths,
        config,
        input_meta,
        overall,
        bucket,
        by_well,
        overall_delta,
        bucket_delta,
        by_well_delta,
        checklist,
        decision,
    )
    print(json.dumps(to_jsonable(summary["decision"]), ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()

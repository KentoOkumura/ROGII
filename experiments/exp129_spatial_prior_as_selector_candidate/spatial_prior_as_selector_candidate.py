from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config
from sklearn.model_selection import GroupKFold

OUTPUT_PREFIX = "exp129_spatial_prior_as_selector_candidate"
DEFAULT_EXP099_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
DEFAULT_EXP099_SCHEMA = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv"
)
DEFAULT_EXP114_OOF = "exp114_spatial_neighbor_prior_signal_audit_oof_predictions.csv.gz"
DEFAULT_EXP114_SUMMARY = "exp114_spatial_neighbor_prior_signal_audit_summary.json"
PROTECTED_COLUMNS = {
    "id",
    "well",
    "target",
    "true_tvt",
    "oracle_label",
    "oracle_candidate",
}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    column: str
    family: str = "base"
    source_variant: str | None = None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_sha256(frame: pd.DataFrame, *, value_col: str) -> str:
    digest = hashlib.sha256()
    for row in frame[["id", value_col]].itertuples(index=False):
        digest.update(str(row.id).encode("utf-8"))
        digest.update(b",")
        digest.update(np.float64(row[1]).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def find_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    parent_parts: tuple[str, ...] = (),
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend([Path.cwd() / filename, Path.cwd() / "artifacts" / filename])
    if parent_parts:
        candidates.append(Path(*parent_parts) / filename)
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def _row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int32)


def _distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def _tail_rank_bucket(ids: pd.Series) -> pd.Categorical:
    ranks = _row_indices_from_ids(ids)
    return pd.cut(
        ranks,
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )


def _quantile_bucket(values: pd.Series | np.ndarray, prefix: str) -> pd.Categorical:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series[np.isfinite(series)]
    if finite.nunique(dropna=True) < 4:
        return pd.Categorical([f"{prefix}_unknown"] * len(series))
    edges = np.unique(np.nanquantile(finite, [0.0, 0.25, 0.50, 0.75, 1.0]))
    if len(edges) < 3:
        return pd.Categorical([f"{prefix}_unknown"] * len(series))
    labels = [f"{prefix}_q{i + 1}" for i in range(len(edges) - 1)]
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred.astype(np.float64) - y_true.astype(np.float64)))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred.astype(np.float64) - y_true.astype(np.float64))))


def candidate_specs_from_config(config: dict[str, Any]) -> list[CandidateSpec]:
    values = get_nested(config, "selector.candidates") or []
    specs: list[CandidateSpec] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("selector.candidates entries must be mappings")
        specs.append(
            CandidateSpec(
                name=str(item["name"]),
                column=str(item.get("column", item["name"])),
                family=str(item.get("family", "base")),
                source_variant=item.get("source_variant"),
            )
        )
    if not specs:
        raise ValueError("selector.candidates must not be empty")
    return specs


def build_required_columns(
    config: dict[str, Any], candidates: list[CandidateSpec]
) -> dict[str, list[str]]:
    exp099 = {"id", "well", "target", "last_known_tvt"}
    exp114 = {"id", "well"}
    for spec in candidates:
        if spec.family == "spatial":
            exp114.add(spec.column)
        else:
            exp099.add(spec.column)
    for key in [
        "selector.context_columns",
        "selector.multiobs_feature_columns",
        "selector.optional_columns",
    ]:
        exp099.update(str(value) for value in (get_nested(config, key) or []))
    for value in get_nested(config, "selector.spatial_feature_columns") or []:
        exp114.add(str(value))
    return {"exp099": sorted(exp099), "exp114": sorted(exp114)}


def load_exp099_cache(
    *,
    cache_path: str | Path | None,
    schema_path: str | Path | None,
    required_columns: list[str],
    max_rows: int | None,
    record_full_sha: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        DEFAULT_EXP099_CACHE,
        cache_path,
        parent_parts=(
            "experiments",
            "exp099_pf_multi_observation_likelihood_probe",
            "kaggle",
            "output",
            "train_v2",
            "artifacts",
        ),
    )
    schema = find_artifact(
        DEFAULT_EXP099_SCHEMA,
        schema_path,
        parent_parts=(
            "experiments",
            "exp099_pf_multi_observation_likelihood_probe",
            "kaggle",
            "output",
            "train_v2",
            "artifacts",
        ),
    )
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    frame = pd.read_csv(
        source,
        usecols=required_columns,
        nrows=max_rows,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    meta = {
        "path": str(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "source_sha256": sha256_path(source) if record_full_sha else None,
        "source_decompressed_sha256": sha256_path(source, decompressed=True)
        if record_full_sha
        else None,
        "schema_path": str(schema),
        "schema_sha256": sha256_path(schema) if record_full_sha else None,
    }
    return frame, meta


def load_exp114_oof(
    *,
    oof_path: str | Path | None,
    summary_path: str | Path | None,
    required_columns: list[str],
    ids: pd.Series,
    record_full_sha: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        DEFAULT_EXP114_OOF,
        oof_path,
        parent_parts=(
            "experiments",
            "exp114_spatial_neighbor_prior_signal_audit",
            "kaggle",
            "output",
            "train_v1",
            "artifacts",
        ),
    )
    summary = find_artifact(
        DEFAULT_EXP114_SUMMARY,
        summary_path,
        parent_parts=(
            "experiments",
            "exp114_spatial_neighbor_prior_signal_audit",
            "kaggle",
            "output",
            "train_v1",
            "artifacts",
        ),
    )
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    id_filter = set(ids.astype(str).tolist())
    seen_ids: set[str] = set()
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source,
        usecols=required_columns,
        chunksize=500_000,
        dtype={"id": str, "well": str},
        low_memory=False,
    ):
        chunk = chunk[chunk["id"].astype(str).isin(id_filter)]
        if len(chunk):
            seen_ids.update(chunk["id"].astype(str).tolist())
            chunks.append(chunk)
            if seen_ids >= id_filter:
                break
    if not chunks:
        raise ValueError("exp114 OOF artifact has no rows matching exp099 cache ids")
    frame = pd.concat(chunks, ignore_index=True)
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    meta = {
        "path": str(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "source_sha256": sha256_path(source) if record_full_sha else None,
        "source_decompressed_sha256": sha256_path(source, decompressed=True)
        if record_full_sha
        else None,
        "summary_path": str(summary),
        "summary_sha256": sha256_path(summary) if record_full_sha else None,
    }
    return frame, meta


def merge_surfaces(exp099: pd.DataFrame, exp114: pd.DataFrame) -> pd.DataFrame:
    merged = exp099.merge(exp114, on=["id", "well"], how="left", validate="one_to_one")
    if len(merged) != len(exp099):
        raise ValueError(f"merge row mismatch: exp099={len(exp099)} merged={len(merged)}")
    return merged


def add_candidate_labels(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    out = frame.copy()
    out["true_tvt"] = out["last_known_tvt"].astype(np.float32) + out["target"].astype(np.float32)
    candidate_values = np.column_stack(
        [
            pd.to_numeric(out[spec.column], errors="coerce").to_numpy(np.float32)
            for spec in candidates
        ]
    )
    true_tvt = out["true_tvt"].to_numpy(np.float32)
    finite = np.isfinite(candidate_values)
    errors = np.abs(candidate_values - true_tvt[:, None]).astype(np.float32)
    errors[~finite] = np.inf
    if np.isinf(errors).all(axis=1).any():
        bad_ids = out.loc[np.isinf(errors).all(axis=1), "id"].head(5).tolist()
        raise ValueError(f"rows without any finite candidate, examples={bad_ids}")
    labels = np.argmin(errors, axis=1).astype(np.int16)
    out["oracle_label"] = labels
    out["oracle_candidate"] = np.asarray([candidates[i].name for i in labels], dtype=object)
    return out, candidate_values, errors, labels


def select_numeric_feature_columns(
    frame: pd.DataFrame,
    config: dict[str, Any],
    candidates: list[CandidateSpec],
) -> list[str]:
    configured = [
        str(value)
        for value in (
            (get_nested(config, "selector.context_columns") or [])
            + (get_nested(config, "selector.multiobs_feature_columns") or [])
            + (get_nested(config, "selector.spatial_feature_columns") or [])
        )
    ]
    engineered: list[str] = []
    last = frame["last_known_tvt"].astype(np.float32)
    value_cols = [spec.column for spec in candidates]
    for spec in candidates:
        delta_col = f"{spec.name}_minus_last"
        frame[delta_col] = frame[spec.column].astype(np.float32) - last
        engineered.append(delta_col)
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            col = f"{left.name}_vs_{right.name}_abs"
            frame[col] = np.abs(
                frame[left.column].astype(np.float32) - frame[right.column].astype(np.float32)
            )
            engineered.append(col)
    frame["candidate_mean"] = frame[value_cols].mean(axis=1).astype(np.float32)
    frame["candidate_std"] = frame[value_cols].std(axis=1).astype(np.float32)
    frame["candidate_range"] = (
        frame[value_cols].max(axis=1) - frame[value_cols].min(axis=1)
    ).astype(np.float32)
    engineered.extend(["candidate_mean", "candidate_std", "candidate_range"])

    columns: list[str] = []
    for column in configured + engineered:
        if column in frame.columns and column not in PROTECTED_COLUMNS and column not in columns:
            columns.append(column)
    missing = [column for column in configured if column not in frame.columns]
    if missing:
        raise ValueError(f"configured feature columns are missing: {missing}")
    numeric_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(frame[column]) and frame[column].notna().any()
    ]
    if not numeric_columns:
        raise ValueError("no numeric feature columns selected")
    return numeric_columns


def fit_impute(
    train: pd.DataFrame, valid: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    train_values = train[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    valid_values = valid[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    medians = np.nanmedian(train_values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    train_bad = ~np.isfinite(train_values)
    valid_bad = ~np.isfinite(valid_values)
    if train_bad.any():
        train_values[train_bad] = np.take(medians, np.where(train_bad)[1])
    if valid_bad.any():
        valid_values[valid_bad] = np.take(medians, np.where(valid_bad)[1])
    return train_values, valid_values


def candidate_metrics(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    candidate_values: np.ndarray,
    errors: np.ndarray,
    oracle_labels: np.ndarray,
) -> pd.DataFrame:
    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    rows: list[dict[str, Any]] = []
    true_ranks = np.argsort(errors, axis=1)
    for idx, spec in enumerate(candidates):
        finite = np.isfinite(candidate_values[:, idx])
        abs_error = errors[:, idx]
        selected = oracle_labels == idx
        row = {
            "candidate": spec.name,
            "family": spec.family,
            "column": spec.column,
            "valid_rows": int(finite.sum()),
            "coverage": float(finite.mean()),
            "oracle_top1_rows": int(selected.sum()),
            "oracle_top1_rate": float(selected.mean()),
        }
        if finite.any():
            row.update(
                {
                    "rmse_valid": rmse(true_tvt[finite], candidate_values[finite, idx]),
                    "mae_valid": mae(true_tvt[finite], candidate_values[finite, idx]),
                    "within_10ft_valid": float(np.mean(abs_error[finite] <= 10.0)),
                    "abs_error_mean_valid": float(np.mean(abs_error[finite])),
                }
            )
        for k in [2, 3, 5]:
            row[f"true_top{k}_rate"] = float(np.mean((true_ranks[:, :k] == idx).any(axis=1)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["family", "rmse_valid"], na_position="last")


def topk_oracle_readout(
    candidates: list[CandidateSpec],
    errors: np.ndarray,
    oracle_labels: np.ndarray,
) -> pd.DataFrame:
    spatial_indices = np.asarray(
        [idx for idx, spec in enumerate(candidates) if spec.family == "spatial"], dtype=np.int16
    )
    sorted_indices = np.argsort(errors, axis=1)
    rows = []
    for k in [1, 2, 3, 5]:
        topk = sorted_indices[:, :k]
        spatial_in_topk = np.isin(topk, spatial_indices).any(axis=1)
        oracle_in_topk = (topk == oracle_labels[:, None]).any(axis=1)
        rows.append(
            {
                "topk": int(k),
                "spatial_in_true_error_topk_rate": float(spatial_in_topk.mean()),
                "oracle_label_in_true_error_topk_rate": float(oracle_in_topk.mean()),
                "rows": int(len(errors)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_selection(
    *,
    frame: pd.DataFrame,
    selected_indices: np.ndarray,
    candidate_values: np.ndarray,
    oracle_labels: np.ndarray,
    candidates: list[CandidateSpec],
    variant: str,
    mode: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    row_idx = np.arange(len(frame))
    selected_tvt = candidate_values[row_idx, selected_indices].astype(np.float32)
    selected_valid = np.isfinite(selected_tvt)
    if not selected_valid.all():
        fallback_idx = [spec.name for spec in candidates].index("likpf_mean")
        selected_indices = selected_indices.copy()
        selected_indices[~selected_valid] = fallback_idx
        selected_tvt = candidate_values[row_idx, selected_indices].astype(np.float32)
    abs_error = np.abs(selected_tvt - true_tvt)
    candidate_names = [spec.name for spec in candidates]
    pred = pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "well": frame["well"].to_numpy(),
            "variant": variant,
            "mode": mode,
            "selected_candidate": np.asarray([candidate_names[i] for i in selected_indices]),
            "selected_candidate_index": selected_indices.astype(np.int16),
            "selected_tvt": selected_tvt,
            "true_tvt": true_tvt,
            "abs_error": abs_error.astype(np.float32),
            "oracle_candidate": frame["oracle_candidate"].to_numpy(),
            "oracle_label": oracle_labels.astype(np.int16),
        }
    )
    metrics = {
        "variant": variant,
        "mode": mode,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "rmse_tvt": rmse(true_tvt, selected_tvt),
        "mae_tvt": mae(true_tvt, selected_tvt),
        "oracle_label_accuracy": float(np.mean(selected_indices == oracle_labels)),
    }
    for threshold in [1.0, 2.0, 5.0, 10.0]:
        metrics[f"within_{int(threshold)}ft"] = float(np.mean(abs_error <= threshold))
    return metrics, pred


def select_oracle_subset(
    candidates: list[CandidateSpec], errors: np.ndarray, family: str
) -> np.ndarray:
    subset = [idx for idx, spec in enumerate(candidates) if spec.family == family]
    if not subset:
        raise ValueError(f"no candidates for family={family}")
    subset_errors = errors[:, subset]
    return np.asarray(subset, dtype=np.int16)[np.argmin(subset_errors, axis=1)].astype(np.int16)


def selection_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    total_by_variant = (
        predictions.groupby(["variant", "mode"], observed=True).size().rename("total")
    )
    counts = (
        predictions.groupby(["variant", "mode", "selected_candidate"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    rows = []
    for row in counts.itertuples(index=False):
        total = int(total_by_variant.loc[(row.variant, row.mode)])
        rows.append(
            {
                "variant": row.variant,
                "mode": row.mode,
                "selected_candidate": row.selected_candidate,
                "rows": int(row.rows),
                "rate": float(row.rows / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "mode", "selected_candidate"])


def summarize_by_well(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, mode, well), group in predictions.groupby(
        ["variant", "mode", "well"], observed=True
    ):
        ordered = group.assign(row_index=_row_indices_from_ids(group["id"])).sort_values(
            "row_index"
        )
        selected = ordered["selected_candidate"].to_numpy()
        switches = int(np.sum(selected[1:] != selected[:-1])) if len(selected) > 1 else 0
        segment_lengths: list[int] = []
        if len(selected):
            start = 0
            for idx in range(1, len(selected)):
                if selected[idx] != selected[idx - 1]:
                    segment_lengths.append(idx - start)
                    start = idx
            segment_lengths.append(len(selected) - start)
        rows.append(
            {
                "variant": variant,
                "mode": mode,
                "well": well,
                "rows": int(len(group)),
                "rmse_tvt": rmse(group["true_tvt"].to_numpy(), group["selected_tvt"].to_numpy()),
                "mae_tvt": mae(group["true_tvt"].to_numpy(), group["selected_tvt"].to_numpy()),
                "within_10ft": float(np.mean(group["abs_error"].to_numpy() <= 10.0)),
                "path_switch_count": switches,
                "path_switch_per_1000_rows": float(switches / max(len(group), 1) * 1000.0),
                "segment_len_min": int(min(segment_lengths)) if segment_lengths else 0,
                "segment_len_p10": float(np.quantile(segment_lengths, 0.10))
                if segment_lengths
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["variant", "mode", "rmse_tvt"], ascending=[True, True, False]
    )


def bucket_metrics(predictions: pd.DataFrame, source_frame: pd.DataFrame) -> pd.DataFrame:
    context = source_frame[["id"]].copy()
    context["distance_bucket"] = _distance_bucket(source_frame.get("md_since", np.nan))
    context["tail_rank_bucket"] = _tail_rank_bucket(source_frame["id"])
    for source_column, bucket_name in [
        ("eval_len", "eval_len_bucket"),
        ("pf_ancc_std", "pf_seed_std_bucket"),
        ("likpf_mean_d", "likpf_delta_bucket"),
        ("xy_plus_trajectory_shape_k8_prior_std", "spatial_traj_std_bucket"),
        ("xy_plus_trajectory_shape_k8_distance_mean", "spatial_traj_distance_bucket"),
    ]:
        if source_column in source_frame.columns:
            context[bucket_name] = _quantile_bucket(source_frame[source_column], bucket_name)
    merged = predictions.merge(context, on="id", how="left", validate="many_to_one")
    rows = []
    bucket_cols = [col for col in context.columns if col != "id"]
    for bucket_col in bucket_cols:
        for (variant, mode, bucket), group in merged.groupby(
            ["variant", "mode", bucket_col], observed=True
        ):
            rows.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "bucket_family": bucket_col,
                    "bucket": str(bucket),
                    "rows": int(len(group)),
                    "rmse_tvt": rmse(
                        group["true_tvt"].to_numpy(), group["selected_tvt"].to_numpy()
                    ),
                    "mae_tvt": mae(group["true_tvt"].to_numpy(), group["selected_tvt"].to_numpy()),
                    "within_10ft": float(np.mean(group["abs_error"].to_numpy() <= 10.0)),
                    "oracle_label_accuracy": float(
                        np.mean(
                            group["selected_candidate_index"].to_numpy()
                            == group["oracle_label"].to_numpy()
                        )
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["variant", "mode", "bucket_family", "bucket"])


def build_long_frame(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    candidates: list[CandidateSpec],
    *,
    row_feature_columns: list[str],
    candidate_values: np.ndarray,
    errors: np.ndarray,
    sample_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if sample_rows is not None and len(row_indices) > sample_rows:
        rng = np.random.default_rng(seed)
        row_indices = np.sort(rng.choice(row_indices, size=int(sample_rows), replace=False))
    family_codes = {
        family: idx for idx, family in enumerate(sorted({spec.family for spec in candidates}))
    }
    chunks: list[pd.DataFrame] = []
    y_error_chunks: list[np.ndarray] = []
    last = frame["last_known_tvt"].to_numpy(np.float32)
    for cand_idx, spec in enumerate(candidates):
        part = frame.iloc[row_indices][["id", "well", *row_feature_columns]].copy()
        values = candidate_values[row_indices, cand_idx].astype(np.float32)
        valid = np.isfinite(values)
        part["candidate_index"] = np.int16(cand_idx)
        part["candidate_family_code"] = np.int16(family_codes[spec.family])
        part["candidate_is_spatial"] = np.int8(spec.family == "spatial")
        part["candidate_tvt"] = np.where(valid, values, 0.0).astype(np.float32)
        part["candidate_valid"] = valid.astype(np.int8)
        part["candidate_minus_last"] = np.where(valid, values - last[row_indices], 0.0).astype(
            np.float32
        )
        for suffix in [
            "prior_std",
            "prior_count",
            "neighbor_wells",
            "distance_mean",
            "same_typewell_share",
        ]:
            value = np.zeros(len(row_indices), dtype=np.float32)
            if spec.source_variant:
                col = f"{spec.source_variant}_{suffix}"
                if col in frame.columns:
                    value = frame.iloc[row_indices][col].to_numpy(np.float32)
            part[f"candidate_{suffix}"] = value
        y = errors[row_indices, cand_idx].astype(np.float32)
        y[~np.isfinite(y)] = 1_000_000.0
        y_error_chunks.append(y)
        chunks.append(part)
    long_frame = pd.concat(chunks, ignore_index=True)
    y_error = np.concatenate(y_error_chunks).astype(np.float32)
    return long_frame, y_error


def viterbi_select(costs: np.ndarray, penalty: float) -> np.ndarray:
    n_rows, n_candidates = costs.shape
    dp = np.empty((n_rows, n_candidates), dtype=np.float32)
    back = np.zeros((n_rows, n_candidates), dtype=np.int16)
    dp[0] = costs[0]
    for row in range(1, n_rows):
        prev = dp[row - 1]
        stay = prev
        best_prev_idx = int(np.argmin(prev))
        best_prev = float(prev[best_prev_idx])
        for cand in range(n_candidates):
            switch_cost = best_prev + penalty
            if stay[cand] <= switch_cost:
                dp[row, cand] = costs[row, cand] + stay[cand]
                back[row, cand] = cand
            else:
                dp[row, cand] = costs[row, cand] + switch_cost
                back[row, cand] = best_prev_idx
    out = np.zeros(n_rows, dtype=np.int16)
    out[-1] = int(np.argmin(dp[-1]))
    for row in range(n_rows - 2, -1, -1):
        out[row] = back[row + 1, out[row + 1]]
    return out


def apply_viterbi_by_well(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    pred_error: np.ndarray,
    penalty: float,
) -> np.ndarray:
    selected = np.zeros(len(row_indices), dtype=np.int16)
    valid_frame = frame.iloc[row_indices][["id", "well"]].copy()
    valid_frame["_pos"] = np.arange(len(row_indices), dtype=np.int32)
    valid_frame["_row_index"] = _row_indices_from_ids(valid_frame["id"])
    for _, group in valid_frame.sort_values(["well", "_row_index"]).groupby("well", sort=False):
        positions = group["_pos"].to_numpy(np.int32)
        selected[positions] = viterbi_select(pred_error[positions], penalty)
    return selected


def train_error_ranker(
    *,
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    candidate_values: np.ndarray,
    errors: np.ndarray,
    oracle_labels: np.ndarray,
    feature_columns: list[str],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    seed = int(get_nested(config, "validation.seed") or 42)
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    log_period = int(get_nested(config, "selector.log_period") or 100)
    max_train_rows = get_nested(config, "selector.long_model.max_train_rows_per_fold")
    max_train_rows = int(max_train_rows) if max_train_rows is not None else None
    penalties = [
        float(value) for value in (get_nested(config, "selector.viterbi_switch_penalties") or [])
    ]
    params = dict(get_nested(config, "selector.long_model.error_lgbm.params") or {})
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    cv = GroupKFold(n_splits=n_folds)
    folds = list(cv.split(frame, oracle_labels, groups=frame["well"]))
    oof_rowwise = np.zeros(len(frame), dtype=np.int16)
    oof_viterbi = {penalty: np.zeros(len(frame), dtype=np.int16) for penalty in penalties}
    topk_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    spatial_indices = np.asarray(
        [idx for idx, spec in enumerate(candidates) if spec.family == "spatial"], dtype=np.int16
    )

    for fold, (train_idx, valid_idx) in enumerate(folds):
        print(f"[fold {fold}] train={len(train_idx)} valid={len(valid_idx)}", flush=True)
        long_train, train_error = build_long_frame(
            frame,
            train_idx,
            candidates,
            row_feature_columns=feature_columns,
            candidate_values=candidate_values,
            errors=errors,
            sample_rows=max_train_rows,
            seed=seed + 101 * fold,
        )
        long_valid, _ = build_long_frame(
            frame,
            valid_idx,
            candidates,
            row_feature_columns=feature_columns,
            candidate_values=candidate_values,
            errors=errors,
            sample_rows=None,
            seed=seed,
        )
        long_feature_columns = [
            col
            for col in long_train.columns
            if col not in {"id", "well"} and pd.api.types.is_numeric_dtype(long_train[col])
        ]
        x_train, x_valid = fit_impute(long_train, long_valid, long_feature_columns)
        valid_error = np.clip(
            errors[valid_idx].T.reshape(-1).astype(np.float32),
            0.0,
            1_000_000.0,
        )
        valid_error[~np.isfinite(valid_error)] = 1_000_000.0
        model = LGBMRegressor(
            objective="regression_l1",
            random_state=seed + 2000 + fold,
            **params,
        )
        model.fit(
            x_train,
            train_error,
            eval_set=[(x_valid, valid_error)],
            eval_metric="l1",
            callbacks=[early_stopping(50), log_evaluation(log_period)],
        )
        pred_error = model.predict(x_valid).reshape(len(candidates), len(valid_idx)).T
        invalid = ~np.isfinite(candidate_values[valid_idx])
        pred_error[invalid] = 1_000_000.0
        rowwise = np.argmin(pred_error, axis=1).astype(np.int16)
        oof_rowwise[valid_idx] = rowwise
        pred_rank = np.argsort(pred_error, axis=1)
        for k in [1, 2, 3, 5]:
            topk = pred_rank[:, :k]
            topk_rows.append(
                {
                    "variant": "lgb_error_ranker_predicted",
                    "fold": int(fold),
                    "topk": int(k),
                    "oracle_label_coverage": float(
                        np.mean((topk == oracle_labels[valid_idx, None]).any(axis=1))
                    ),
                    "spatial_in_predicted_topk_rate": float(
                        np.isin(topk, spatial_indices).any(axis=1).mean()
                    ),
                    "rows": int(len(valid_idx)),
                }
            )
        for penalty in penalties:
            oof_viterbi[penalty][valid_idx] = apply_viterbi_by_well(
                frame, valid_idx, pred_error, penalty
            )
        model_path = model_dir / f"{OUTPUT_PREFIX}_lgb_error_ranker_fold{fold}.txt"
        model.booster_.save_model(str(model_path))
        manifest.append(
            {
                "variant": "lgb_error_ranker",
                "fold": fold,
                "path": str(model_path.relative_to(output_dir)),
                "sha256": sha256_path(model_path),
                "best_iteration": int(model.best_iteration_ or model.n_estimators),
            }
        )
        for feature, importance in zip(
            long_feature_columns, model.feature_importances_, strict=False
        ):
            importance_rows.append(
                {
                    "variant": "lgb_error_ranker",
                    "fold": fold,
                    "feature": feature,
                    "importance": float(importance),
                }
            )

    metric_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    metrics, pred = evaluate_selection(
        frame=frame,
        selected_indices=oof_rowwise,
        candidate_values=candidate_values,
        oracle_labels=oracle_labels,
        candidates=candidates,
        variant="lgb_error_ranker_rowwise",
        mode="oof",
    )
    metric_rows.append(metrics)
    pred_frames.append(pred)
    for penalty, selected in oof_viterbi.items():
        metrics, pred = evaluate_selection(
            frame=frame,
            selected_indices=selected,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            candidates=candidates,
            variant=f"lgb_error_ranker_viterbi_p{str(penalty).replace('.', 'p')}",
            mode="oof_viterbi",
        )
        metrics["viterbi_switch_penalty"] = penalty
        metric_rows.append(metrics)
        pred_frames.append(pred)
    manifest_path = output_dir / f"{OUTPUT_PREFIX}_model_manifest.json"
    with manifest_path.open("w") as fp:
        json.dump(to_jsonable({"models": manifest}), fp, indent=2, sort_keys=True)
    manifest_meta = [
        {**item, "manifest": manifest_path.name, "manifest_sha256": sha256_path(manifest_path)}
        for item in manifest
    ]
    return (
        pd.DataFrame(metric_rows),
        pd.concat(pred_frames, ignore_index=True),
        pd.DataFrame(topk_rows),
        pd.DataFrame(importance_rows),
        manifest_meta,
    )


def summarize_decision(metrics: pd.DataFrame, distribution: pd.DataFrame) -> dict[str, Any]:
    likpf = metrics[metrics["variant"].eq("likpf_mean_single")].head(1)
    oracle_base = metrics[metrics["variant"].eq("oracle_base_only")].head(1)
    oracle_expanded = metrics[metrics["variant"].eq("oracle_expanded")].head(1)
    oof = metrics[metrics["mode"].astype(str).str.startswith("oof")].sort_values("rmse_tvt")
    decision = {
        "recommendation": "train_side_audit_pending_kaggle_run"
        if oof.empty
        else "selector_not_supported",
        "expanded_oracle_delta_vs_base_oracle": None,
        "best_oof_delta_vs_likpf_mean": None,
        "best_oof_variant": None,
    }
    if not oracle_base.empty and not oracle_expanded.empty:
        decision["expanded_oracle_delta_vs_base_oracle"] = float(
            oracle_expanded.iloc[0]["rmse_tvt"] - oracle_base.iloc[0]["rmse_tvt"]
        )
    if not oof.empty and not likpf.empty:
        best = oof.iloc[0]
        delta = float(best["rmse_tvt"] - likpf.iloc[0]["rmse_tvt"])
        decision["best_oof_variant"] = to_jsonable(best.to_dict())
        decision["best_oof_delta_vs_likpf_mean"] = delta
        spatial_dist = distribution[
            (distribution["variant"] == best["variant"])
            & (distribution["selected_candidate"].astype(str).str.contains("xy_", regex=False))
        ]
        spatial_rate = float(spatial_dist["rate"].sum()) if len(spatial_dist) else 0.0
        decision["best_oof_spatial_selection_rate"] = spatial_rate
        if delta < -0.05 and spatial_rate > 0.005:
            decision["recommendation"] = (
                "selector_supported_needs_rawtest_parity_and_hidden_like_stress"
            )
        elif delta < 0.0:
            decision["recommendation"] = "weak_selector_supported_for_diagnostics_only"
    return decision


def run_spatial_prior_as_selector_candidate(
    *,
    output_dir: str | Path,
    exp099_cache_path: str | Path | None,
    exp099_schema_path: str | Path | None,
    exp114_oof_path: str | Path | None,
    exp114_summary_path: str | Path | None,
    max_rows: int | None,
    skip_models: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    config = load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_specs_from_config(config)
    required = build_required_columns(config, candidates)
    record_full_sha = max_rows is None
    exp099, exp099_meta = load_exp099_cache(
        cache_path=exp099_cache_path,
        schema_path=exp099_schema_path,
        required_columns=required["exp099"],
        max_rows=max_rows,
        record_full_sha=record_full_sha,
    )
    exp114, exp114_meta = load_exp114_oof(
        oof_path=exp114_oof_path,
        summary_path=exp114_summary_path,
        required_columns=required["exp114"],
        ids=exp099["id"],
        record_full_sha=record_full_sha,
    )
    frame = merge_surfaces(exp099, exp114)
    frame, candidate_values, errors, oracle_labels = add_candidate_labels(frame, candidates)
    feature_columns = select_numeric_feature_columns(frame, config, candidates)
    cand_metrics = candidate_metrics(frame, candidates, candidate_values, errors, oracle_labels)
    topk_true = topk_oracle_readout(candidates, errors, oracle_labels)

    metric_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    names = [spec.name for spec in candidates]
    baseline_indices = {
        "likpf_mean_single": np.full(len(frame), names.index("likpf_mean"), dtype=np.int16),
        "oracle_base_only": select_oracle_subset(candidates, errors, "base"),
        "oracle_spatial_only": select_oracle_subset(candidates, errors, "spatial"),
        "oracle_expanded": oracle_labels.astype(np.int16),
    }
    for variant, selected in baseline_indices.items():
        mode = "oracle" if variant.startswith("oracle") else "baseline"
        metrics, pred = evaluate_selection(
            frame=frame,
            selected_indices=selected,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            candidates=candidates,
            variant=variant,
            mode=mode,
        )
        metric_rows.append(metrics)
        pred_frames.append(pred)

    topk_pred = pd.DataFrame()
    importance = pd.DataFrame()
    model_manifest: list[dict[str, Any]] = []
    if not skip_models and bool(get_nested(config, "selector.run_lgbm")):
        model_metrics, model_preds, topk_pred, importance, model_manifest = train_error_ranker(
            frame=frame,
            candidates=candidates,
            candidate_values=candidate_values,
            errors=errors,
            oracle_labels=oracle_labels,
            feature_columns=feature_columns,
            config=config,
            output_dir=output_dir,
        )
        metric_rows.extend(model_metrics.to_dict("records"))
        pred_frames.append(model_preds)
    else:
        model_manifest_path = output_dir / f"{OUTPUT_PREFIX}_model_manifest.json"
        with model_manifest_path.open("w") as fp:
            json.dump({"models": []}, fp, indent=2, sort_keys=True)

    predictions = pd.concat(pred_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values("rmse_tvt")
    distribution = selection_distribution(predictions)
    by_well = summarize_by_well(predictions)
    buckets = bucket_metrics(predictions, frame)
    mean_importance = (
        importance.groupby(["variant", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            folds=("importance", "size"),
        )
        .sort_values(["variant", "mean_importance"], ascending=[True, False])
        if len(importance)
        else pd.DataFrame(
            columns=["variant", "feature", "mean_importance", "std_importance", "folds"]
        )
    )
    decision = summarize_decision(metrics, distribution)

    metrics_path = output_dir / f"{OUTPUT_PREFIX}_metrics.csv"
    predictions_path = output_dir / f"{OUTPUT_PREFIX}_oof_selected_predictions.csv.gz"
    distribution_path = output_dir / f"{OUTPUT_PREFIX}_selection_distribution.csv"
    by_well_path = output_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    buckets_path = output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    cand_metrics_path = output_dir / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    topk_true_path = output_dir / f"{OUTPUT_PREFIX}_true_error_topk_metrics.csv"
    topk_pred_path = output_dir / f"{OUTPUT_PREFIX}_predicted_topk_metrics.csv"
    importance_path = output_dir / f"{OUTPUT_PREFIX}_feature_importance.csv"
    mean_importance_path = output_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv"
    schema_out_path = output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    distribution.to_csv(distribution_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    buckets.to_csv(buckets_path, index=False)
    cand_metrics.to_csv(cand_metrics_path, index=False)
    topk_true.to_csv(topk_true_path, index=False)
    topk_pred.to_csv(topk_pred_path, index=False)
    importance.to_csv(importance_path, index=False)
    mean_importance.to_csv(mean_importance_path, index=False)
    pd.DataFrame(
        [{"feature_index": idx, "feature": feature} for idx, feature in enumerate(feature_columns)]
    ).to_csv(schema_out_path, index=False)

    prediction_hashes = {
        variant: prediction_sha256(group, value_col="selected_tvt")
        for variant, group in predictions.groupby("variant", observed=True)
    }
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_debug_completed"
        if max_rows is not None
        else "completed_train_side_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - t0),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "candidates": [to_jsonable(spec.__dict__) for spec in candidates],
        "source": {"exp099": exp099_meta, "exp114": exp114_meta},
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "best_metric": to_jsonable(metrics.iloc[0].to_dict() if len(metrics) else {}),
        "decision": to_jsonable(decision),
        "sha256": {
            "metrics": sha256_path(metrics_path),
            "predictions": sha256_path(predictions_path),
            "predictions_decompressed": sha256_path(predictions_path, decompressed=True),
            "feature_schema": sha256_path(schema_out_path),
            "candidate_metrics": sha256_path(cand_metrics_path),
            "prediction_by_variant": prediction_hashes,
        },
        "model_manifest": model_manifest,
        "artifacts": {
            "metrics": metrics_path.name,
            "oof_selected_predictions": predictions_path.name,
            "selection_distribution": distribution_path.name,
            "by_well": by_well_path.name,
            "bucket_metrics": buckets_path.name,
            "candidate_metrics": cand_metrics_path.name,
            "true_error_topk_metrics": topk_true_path.name,
            "predicted_topk_metrics": topk_pred_path.name,
            "feature_importance": importance_path.name,
            "feature_importance_mean": mean_importance_path.name,
            "feature_schema": schema_out_path.name,
            "model_manifest": f"{OUTPUT_PREFIX}_model_manifest.json",
        },
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    with summary_path.open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--exp099-cache-path", type=Path, default=None)
    parser.add_argument("--exp099-schema-path", type=Path, default=None)
    parser.add_argument("--exp114-oof-path", type=Path, default=None)
    parser.add_argument("--exp114-summary-path", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args(argv)
    paths = ExperimentPaths()
    config = load_config()
    output_dir = args.output_dir or (
        paths.artifacts_dir
        if not Path("/kaggle/working").exists()
        else Path("/kaggle/working") / "artifacts"
    )
    max_rows = args.max_rows
    configured_max = get_nested(config, "selector.max_rows")
    if max_rows is None and configured_max is not None:
        max_rows = int(configured_max)
    return run_spatial_prior_as_selector_candidate(
        output_dir=output_dir,
        exp099_cache_path=args.exp099_cache_path
        or get_nested(config, "data.exp099_train_feature_cache_local"),
        exp099_schema_path=args.exp099_schema_path
        or get_nested(config, "data.exp099_train_feature_schema_local"),
        exp114_oof_path=args.exp114_oof_path
        or get_nested(config, "data.exp114_oof_predictions_local"),
        exp114_summary_path=args.exp114_summary_path
        or get_nested(config, "data.exp114_summary_local"),
        max_rows=max_rows,
        skip_models=args.skip_models,
    )


if __name__ == "__main__":
    main()

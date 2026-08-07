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

EXP072_ARTIFACTS = Path("experiments") / "exp072_exp063_full_replay_feature_cache" / "artifacts"
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
FULL_REPLAY_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
EXP056_ARTIFACTS = (
    Path("experiments") / "exp056_public_sel15_pf_oof_multicutoff_artifact" / "artifacts"
)
EXP083_ARTIFACTS = Path("experiments") / "exp083_pf_beam_true_tvt_2d_well_eda" / "artifacts"
EXP056_WELL_SUMMARY = "public_sel15_pf_oof_well_summary.csv"
EXP083_WELL_SUMMARY = "pf_beam_true_tvt_2d_well_eda_clean_all_well_summary.csv"
OUTPUT_PREFIX = "exp093_pf_candidate_coverage_then_ranker_audit"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    source_column: str
    transform: str
    role: str
    enabled: bool = True


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    local_artifacts: Path = EXP072_ARTIFACTS,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            local_artifacts / filename,
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
        ]
    )
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def find_optional_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    local_artifacts: Path,
) -> Path | None:
    try:
        return find_artifact(filename, explicit_path, local_artifacts=local_artifacts)
    except FileNotFoundError:
        return None


def numeric_array(frame: pd.DataFrame, column: str, *, default: float | None = None) -> np.ndarray:
    if column not in frame.columns:
        if default is None:
            raise ValueError(f"required column is missing: {column}")
        return np.full(len(frame), float(default), dtype=np.float32)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def read_feature_cache(
    cache_path: str | Path | None,
    *,
    required_columns: list[str],
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(FULL_REPLAY_TRAIN_FEATURES, cache_path)
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
    if frame[["id", "well"]].isna().any().any():
        raise ValueError("feature cache contains missing id/well values")
    schema_path: Path | None = None
    try:
        schema_path = find_artifact(FULL_REPLAY_FEATURE_SCHEMA)
    except FileNotFoundError:
        schema_path = None
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": (
            sha256_path(source, decompressed=True) if source.suffix == ".gz" else None
        ),
        "schema": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_path(schema_path) if schema_path else None,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": list(frame.columns),
    }
    return frame, metadata


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


def read_exp056_multicutoff_context(
    explicit_path: str | Path | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    path = find_optional_artifact(
        EXP056_WELL_SUMMARY,
        explicit_path,
        local_artifacts=EXP056_ARTIFACTS,
    )
    if path is None:
        return None, {"available": False, "reason": "exp056 well summary not found"}
    frame = pd.read_csv(path, dtype={"well_id": str})
    for column in frame.columns:
        if column != "well_id":
            frame[column] = pd.to_numeric(frame[column], errors="ignore")
    grouped = frame.groupby("well_id", sort=False)
    context = grouped.agg(
        exp056_cutoff_count=("cutoff_fraction", "nunique"),
        exp056_rows_max=("rows", "max"),
        exp056_pf_rmse_min=("pf_rmse", "min"),
        exp056_pf_rmse_mean=("pf_rmse", "mean"),
        exp056_anchor_rmse_mean=("last_anchor_rmse", "mean"),
        exp056_beam_rmse_mean=("beam_rmse", "mean"),
        exp056_pf_weight_entropy_mean=("pf_weight_entropy", "mean"),
        exp056_beam_final_cost_min=("beam_final_cost_min", "min"),
    ).reset_index()
    context = context.rename(columns={"well_id": "well"})
    meta = {
        "available": True,
        "source": str(path),
        "source_sha256": sha256_path(path),
        "rows": int(len(frame)),
        "wells": int(context["well"].nunique()),
    }
    return context, meta


def read_exp083_well_context(
    explicit_path: str | Path | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    path = find_optional_artifact(
        EXP083_WELL_SUMMARY,
        explicit_path,
        local_artifacts=EXP083_ARTIFACTS,
    )
    if path is None:
        return None, {"available": False, "reason": "exp083 well summary not found"}
    frame = pd.read_csv(path, dtype={"well_id": str})
    keep = [
        "well_id",
        "pf_beam_abs_diff_mean",
        "pf_beam_abs_diff_p95",
        "pf_ancc_std_mean",
        "pf_ancc_std_p95",
        "beam_std_d_mean",
        "beam_std_d_p95",
        "eval_len_mean",
        "known_len_mean",
        "likpf_mean_d_mean",
    ]
    context = frame[[column for column in keep if column in frame.columns]].copy()
    context = context.rename(columns={"well_id": "well"})
    for column in context.columns:
        if column != "well":
            context[column] = pd.to_numeric(context[column], errors="coerce")
    meta = {
        "available": True,
        "source": str(path),
        "source_sha256": sha256_path(path),
        "rows": int(len(frame)),
        "wells": int(context["well"].nunique()),
        "columns": list(context.columns),
    }
    return context, meta


def merge_well_context(
    source_frame: pd.DataFrame,
    *,
    exp056_context_path: str | Path | None = None,
    exp083_context_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    context = source_frame[["id", "well"]].copy()
    context["distance_bucket"] = _distance_bucket(source_frame.get("md_since", np.nan))
    context["tail_rank_bucket"] = _tail_rank_bucket(source_frame["id"])
    for source_column, bucket_name in [
        ("eval_len", "eval_len_bucket"),
        ("pf_ancc_std", "pf_seed_std_bucket"),
        ("likpf_mean_d", "likpf_delta_bucket"),
    ]:
        if source_column in source_frame.columns:
            context[bucket_name] = _quantile_bucket(source_frame[source_column], bucket_name)
        else:
            context[bucket_name] = pd.Categorical([f"{bucket_name}_unknown"] * len(context))
    if {"pf_ancc", "beam_mean_d", "last_known_tvt"}.issubset(source_frame.columns):
        pf_beam_abs = np.abs(
            numeric_array(source_frame, "pf_ancc")
            - (
                numeric_array(source_frame, "last_known_tvt")
                + numeric_array(source_frame, "beam_mean_d")
            )
        )
        context["pf_beam_disagreement_bucket"] = _quantile_bucket(
            pf_beam_abs,
            "pf_beam_disagreement",
        )
    else:
        context["pf_beam_disagreement_bucket"] = pd.Categorical(
            ["pf_beam_disagreement_unknown"] * len(context)
        )

    exp056_context, exp056_meta = read_exp056_multicutoff_context(exp056_context_path)
    exp083_context, exp083_meta = read_exp083_well_context(exp083_context_path)
    if exp056_context is not None:
        context = context.merge(exp056_context, on="well", how="left", validate="many_to_one")
        context["exp056_pf_rmse_min_bucket"] = _quantile_bucket(
            context["exp056_pf_rmse_min"],
            "exp056_pf_rmse_min",
        )
    else:
        context["exp056_pf_rmse_min_bucket"] = pd.Categorical(
            ["exp056_pf_rmse_min_unknown"] * len(context)
        )
    if exp083_context is not None:
        context = context.merge(exp083_context, on="well", how="left", validate="many_to_one")
        if "pf_beam_abs_diff_mean" in context.columns:
            context["exp083_pf_beam_abs_diff_bucket"] = _quantile_bucket(
                context["pf_beam_abs_diff_mean"],
                "exp083_pf_beam_abs_diff",
            )
    if "exp083_pf_beam_abs_diff_bucket" not in context.columns:
        context["exp083_pf_beam_abs_diff_bucket"] = pd.Categorical(
            ["exp083_pf_beam_abs_diff_unknown"] * len(context)
        )
    return context, {"exp056": exp056_meta, "exp083": exp083_meta}


def _finite_ncc_and_l2(query: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    query_norm = (query - query.mean(axis=1, keepdims=True)) / (
        query.std(axis=1, keepdims=True) + 1e-6
    )
    candidates_norm = (candidates - candidates.mean(axis=1, keepdims=True)) / (
        candidates.std(axis=1, keepdims=True) + 1e-6
    )
    ncc = query_norm @ candidates_norm.T / float(query.shape[1])
    l2 = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * ncc))
    return ncc.astype(np.float32), l2.astype(np.float32)


def _self_gr_candidates_for_well(
    *,
    full_gr: np.ndarray,
    prefix_tvt: np.ndarray,
    eval_indices: np.ndarray,
    half_windows: tuple[int, ...],
    stride: int,
    prefix_tail_rows: int,
) -> dict[str, np.ndarray]:
    n_eval = len(eval_indices)
    prefix_len = len(prefix_tvt)
    smoothed_gr = (
        pd.Series(full_gr, dtype="float32")
        .rolling(5, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )
    outputs: dict[str, np.ndarray] = {}
    score_columns: list[np.ndarray] = []
    tvt_columns: list[np.ndarray] = []
    l2_columns: list[np.ndarray] = []
    for half_window in half_windows:
        key = int(half_window)
        window = 2 * key + 1
        if prefix_len < window + 1 or n_eval == 0:
            score = np.zeros(n_eval, dtype=np.float32)
            second = np.zeros(n_eval, dtype=np.float32)
            l2 = np.full(n_eval, 10.0, dtype=np.float32)
            matched_tvt = np.full(n_eval, float(prefix_tvt[-1]), dtype=np.float32)
            matched_idx = np.full(n_eval, prefix_len - 1, dtype=np.float32)
        else:
            start_min = max(0, prefix_len - int(prefix_tail_rows))
            starts = np.arange(start_min, prefix_len - window + 1, int(stride), dtype=np.int32)
            if len(starts) == 0:
                starts = np.array([max(0, prefix_len - window)], dtype=np.int32)
            offsets = np.arange(window, dtype=np.int32)
            candidates = smoothed_gr[starts[:, None] + offsets[None, :]].astype(np.float32)
            padded = np.pad(smoothed_gr, key, mode="edge")
            query = padded[eval_indices[:, None] + offsets[None, :]].astype(np.float32)
            ncc, l2_matrix = _finite_ncc_and_l2(query, candidates)
            best = ncc.argmax(axis=1)
            score = ncc[np.arange(n_eval), best].astype(np.float32)
            second = (
                np.partition(ncc, -2, axis=1)[:, -2].astype(np.float32)
                if ncc.shape[1] > 1
                else np.zeros(n_eval, dtype=np.float32)
            )
            l2 = l2_matrix[np.arange(n_eval), best].astype(np.float32)
            matched_idx = np.clip(starts[best] + key, 0, prefix_len - 1).astype(np.float32)
            matched_tvt = prefix_tvt[matched_idx.astype(np.int32)].astype(np.float32)
        outputs[f"self_gr_sc{key}"] = matched_tvt
        outputs[f"self_gr_sc{key}_score"] = score
        outputs[f"self_gr_sc{key}_gap"] = (score - second).astype(np.float32)
        outputs[f"self_gr_sc{key}_l2"] = l2
        outputs[f"self_gr_sc{key}_lag_rows"] = (matched_idx - float(prefix_len - 1)).astype(
            np.float32
        )
        score_columns.append(score)
        tvt_columns.append(matched_tvt)
        l2_columns.append(l2)

    score_matrix = np.stack(score_columns, axis=1)
    tvt_matrix = np.stack(tvt_columns, axis=1)
    l2_matrix = np.stack(l2_columns, axis=1)
    weights = np.exp(3.0 * score_matrix)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-9
    best_pos = score_matrix.argmax(axis=1)
    outputs["self_gr_ens"] = (tvt_matrix * weights).sum(axis=1).astype(np.float32)
    outputs["self_gr_best"] = tvt_matrix[np.arange(n_eval), best_pos].astype(np.float32)
    outputs["self_gr_score_max"] = score_matrix.max(axis=1).astype(np.float32)
    outputs["self_gr_score_mean"] = score_matrix.mean(axis=1).astype(np.float32)
    outputs["self_gr_score_gap_max"] = np.max(
        np.stack([outputs[f"self_gr_sc{int(scale)}_gap"] for scale in half_windows], axis=1),
        axis=1,
    ).astype(np.float32)
    outputs["self_gr_l2_min"] = l2_matrix.min(axis=1).astype(np.float32)
    outputs["self_gr_tvt_range"] = (tvt_matrix.max(axis=1) - tvt_matrix.min(axis=1)).astype(
        np.float32
    )
    outputs["self_gr_best_scale"] = np.asarray(half_windows, dtype=np.float32)[best_pos]
    return outputs


def build_self_gr_candidate_frame(
    frame: pd.DataFrame,
    *,
    train_dir: str | Path,
    half_windows: tuple[int, ...],
    stride: int,
    prefix_tail_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dir = Path(train_dir)
    base = pd.DataFrame({"id": frame["id"].astype(str), "well": frame["well"].astype(str)})
    base["_row_idx"] = _row_indices_from_ids(base["id"])
    candidate_frames: list[pd.DataFrame] = []
    well_rows: list[dict[str, Any]] = []
    for well, positions in base.groupby("well", sort=False).groups.items():
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        if not horizontal_path.exists():
            raise FileNotFoundError(f"raw train horizontal well file not found: {horizontal_path}")
        horizontal = pd.read_csv(horizontal_path, usecols=["GR", "TVT_input"])
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
        known_mask = tvt_input.notna().to_numpy()
        if not known_mask.any():
            raise ValueError(f"No finite TVT_input prefix rows for well {well}")
        prefix_len = int(np.flatnonzero(known_mask)[-1] + 1)
        prefix_tvt = tvt_input.iloc[:prefix_len].to_numpy(np.float32)
        gr_series = pd.to_numeric(horizontal["GR"], errors="coerce")
        fallback = float(gr_series.iloc[:prefix_len].mean())
        if not np.isfinite(fallback):
            fallback = float(gr_series.mean()) if np.isfinite(float(gr_series.mean())) else 0.0
        full_gr = (
            gr_series.interpolate(limit_direction="both").fillna(fallback).to_numpy(np.float32)
        )
        row_idx = base.loc[list(positions), "_row_idx"].to_numpy(np.int32)
        if row_idx.min(initial=0) < 0 or row_idx.max(initial=0) >= len(horizontal):
            raise ValueError(f"row index out of range for well {well}")
        outputs = _self_gr_candidates_for_well(
            full_gr=full_gr,
            prefix_tvt=prefix_tvt,
            eval_indices=row_idx,
            half_windows=half_windows,
            stride=stride,
            prefix_tail_rows=prefix_tail_rows,
        )
        rows = pd.DataFrame(
            {
                "id": base.loc[list(positions), "id"].to_numpy(),
                "well": str(well),
                "self_gr_known_prefix_rows": np.full(len(row_idx), prefix_len, dtype=np.float32),
                "self_gr_eval_len": np.full(
                    len(row_idx),
                    max(0, len(horizontal) - prefix_len),
                    dtype=np.float32,
                ),
                "self_gr_prefix_missing_rate": np.full(
                    len(row_idx),
                    float(pd.isna(horizontal["GR"].iloc[:prefix_len]).mean()),
                    dtype=np.float32,
                ),
                "self_gr_eval_missing_rate": np.full(
                    len(row_idx),
                    float(pd.isna(horizontal["GR"].iloc[prefix_len:]).mean()),
                    dtype=np.float32,
                ),
            }
        )
        for column, values in outputs.items():
            rows[column] = np.asarray(values, dtype=np.float32)
        candidate_frames.append(rows)
        well_rows.append(
            {
                "well": str(well),
                "rows": int(len(row_idx)),
                "known_prefix_rows": int(prefix_len),
                "eval_len": int(max(0, len(horizontal) - prefix_len)),
                "prefix_missing_rate": float(pd.isna(horizontal["GR"].iloc[:prefix_len]).mean()),
                "eval_missing_rate": float(pd.isna(horizontal["GR"].iloc[prefix_len:]).mean()),
            }
        )
    out = pd.concat(candidate_frames, ignore_index=True)
    if out.drop(columns=["id", "well"]).isna().any().any():
        raise ValueError("self-GR candidate frame contains missing values")
    return out, pd.DataFrame(well_rows)


def build_required_columns(
    candidate_specs: list[CandidateSpec], extra_columns: list[str]
) -> list[str]:
    columns = {"id", "well", "target", "last_known_tvt"}
    columns.update(extra_columns)
    for spec in candidate_specs:
        columns.add(spec.source_column)
    return sorted(columns)


def materialize_existing_candidates(
    frame: pd.DataFrame,
    candidate_specs: list[CandidateSpec],
) -> pd.DataFrame:
    out = frame[["id", "well", "target", "last_known_tvt"]].copy()
    last_known = numeric_array(frame, "last_known_tvt")
    out["true_tvt"] = last_known + numeric_array(frame, "target")
    for spec in candidate_specs:
        if not spec.enabled:
            continue
        values = numeric_array(frame, spec.source_column)
        if spec.transform == "absolute":
            out[spec.name] = values
        elif spec.transform == "base_plus_delta":
            out[spec.name] = last_known + values
        else:
            raise ValueError(f"Unsupported candidate transform: {spec.transform}")
    return out


def _candidate_score(
    frame: pd.DataFrame,
    candidate_name: str,
    *,
    self_gr_columns: set[str],
) -> np.ndarray:
    n = len(frame)
    if candidate_name in self_gr_columns:
        score_col = (
            "self_gr_score_max"
            if candidate_name in {"self_gr_ens", "self_gr_best"}
            else (f"{candidate_name}_score")
        )
        score = numeric_array(frame, score_col, default=0.0)
        score = np.clip((score + 1.0) / 2.0, 0.0, 1.0)
        gap = numeric_array(frame, "self_gr_score_gap_max", default=0.0)
        return np.clip(0.75 * score + 0.25 * np.clip(gap, 0.0, 1.0), 0.0, 1.0)

    last_known = numeric_array(frame, "last_known_tvt")
    if candidate_name == "last_anchor_tvt":
        return np.full(n, 0.15, dtype=np.float32)
    if candidate_name == "pf_ancc":
        pf_std = np.abs(numeric_array(frame, "pf_ancc_std", default=50.0))
        pf_likpf = np.abs(
            numeric_array(frame, "pf_ancc") - numeric_array(frame, "likpf_mean", default=np.nan)
        )
        pf_likpf = np.nan_to_num(pf_likpf, nan=50.0)
        return (1.0 / (1.0 + pf_std / 50.0 + pf_likpf / 100.0)).astype(np.float32)
    if candidate_name == "beam_mean":
        disagreement = np.abs(
            numeric_array(frame, "pf_ancc", default=np.nan) - numeric_array(frame, "beam_mean")
        )
        disagreement = np.nan_to_num(disagreement, nan=75.0)
        return (1.0 / (1.0 + disagreement / 75.0)).astype(np.float32)
    if candidate_name == "likpf_mean":
        delta = np.abs(numeric_array(frame, "likpf_mean") - last_known)
        return (1.0 / (1.0 + delta / 150.0)).astype(np.float32)
    if candidate_name in {"sc_ens", "hyb"}:
        delta = np.abs(numeric_array(frame, candidate_name) - last_known)
        return (0.5 / (1.0 + delta / 200.0)).astype(np.float32)
    return np.full(n, 0.35, dtype=np.float32)


def build_candidate_long_frame(
    base_frame: pd.DataFrame,
    candidate_columns: list[str],
    *,
    self_gr_columns: set[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    true_tvt = numeric_array(base_frame, "true_tvt")
    for name in candidate_columns:
        if name not in base_frame.columns:
            continue
        pred = numeric_array(base_frame, name)
        score = _candidate_score(base_frame, name, self_gr_columns=self_gr_columns)
        item = pd.DataFrame(
            {
                "id": base_frame["id"].to_numpy(),
                "well": base_frame["well"].to_numpy(),
                "candidate": name,
                "pred_tvt": pred,
                "target_tvt": true_tvt,
                "abs_error": np.abs(pred - true_tvt).astype(np.float32),
                "rank_score": score,
                "candidate_family": (
                    "self_gr" if name in self_gr_columns else "existing_pf_beam_likpf"
                ),
            }
        )
        rows.append(item)
    long_frame = pd.concat(rows, ignore_index=True)
    if not np.isfinite(
        long_frame[["pred_tvt", "target_tvt", "abs_error", "rank_score"]].to_numpy()
    ).all():
        raise ValueError("candidate long frame contains non-finite numeric values")
    return long_frame


def summarize_candidate_metrics(long_frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, group in long_frame.groupby("candidate", sort=False):
        error = group["pred_tvt"].to_numpy(np.float32) - group["target_tvt"].to_numpy(np.float32)
        abs_error = np.abs(error)
        row: dict[str, Any] = {
            "candidate": str(candidate),
            "candidate_family": str(group["candidate_family"].iloc[0]),
            "rows": int(len(group)),
            "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
            "mae_tvt": float(np.mean(abs_error)),
            "bias_tvt": float(np.mean(error)),
            "abs_error_p50": float(np.quantile(abs_error, 0.50)),
            "abs_error_p90": float(np.quantile(abs_error, 0.90)),
            "abs_error_p95": float(np.quantile(abs_error, 0.95)),
            "rank_score_mean": float(group["rank_score"].mean()),
        }
        for threshold in thresholds:
            row[f"within_{threshold:g}ft"] = float(np.mean(abs_error <= float(threshold)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse_tvt").reset_index(drop=True)


def summarize_oracle_and_rank_metrics_for_sets(
    long_frame: pd.DataFrame,
    thresholds: list[float],
    topk_values: list[int],
    candidate_sets: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_set, candidates in candidate_sets.items():
        subset_frame = long_frame[long_frame["candidate"].isin(candidates)].copy()
        candidate_count = int(subset_frame["candidate"].nunique())
        if candidate_count == 0:
            continue
        sorted_frames = {
            "oracle_best_error": subset_frame.sort_values(
                ["id", "abs_error"],
                ascending=[True, True],
            ),
            "candidate_rank_score": subset_frame.sort_values(
                ["id", "rank_score"],
                ascending=[True, False],
            ),
        }
        for rank_family, sorted_frame in sorted_frames.items():
            for topk in topk_values:
                k = min(int(topk), candidate_count)
                topk_subset = sorted_frame.groupby("id", sort=False).head(k).copy()
                best = (
                    topk_subset.sort_values(["id", "abs_error"], ascending=[True, True])
                    .groupby("id", sort=False)
                    .head(1)
                )
                error = best["pred_tvt"].to_numpy(np.float32) - best["target_tvt"].to_numpy(
                    np.float32
                )
                abs_error = np.abs(error)
                row: dict[str, Any] = {
                    "candidate_set": candidate_set,
                    "rank_family": rank_family,
                    "topk": int(k),
                    "candidate_count": int(candidate_count),
                    "rows": int(len(best)),
                    "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
                    "mae_tvt": float(np.mean(abs_error)),
                    "selected_self_gr_rate": float(
                        best["candidate_family"].eq("self_gr").mean()
                    ),
                    "selected_candidate_top": str(best["candidate"].mode().iloc[0]),
                }
                for threshold in thresholds:
                    row[f"within_{threshold:g}ft"] = float(np.mean(abs_error <= float(threshold)))
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_bucket_metrics(
    long_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    frame = long_frame.merge(context_frame, on=["id", "well"], how="left", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    bucket_families = [
        "distance_bucket",
        "tail_rank_bucket",
        "eval_len_bucket",
        "pf_seed_std_bucket",
        "likpf_delta_bucket",
        "pf_beam_disagreement_bucket",
        "exp056_pf_rmse_min_bucket",
        "exp083_pf_beam_abs_diff_bucket",
    ]
    for bucket_family in bucket_families:
        for (candidate, bucket), group in frame.groupby(
            ["candidate", bucket_family], observed=True
        ):
            error = group["pred_tvt"].to_numpy(np.float32) - group["target_tvt"].to_numpy(
                np.float32
            )
            abs_error = np.abs(error)
            row: dict[str, Any] = {
                "candidate": str(candidate),
                "bucket_family": bucket_family,
                "bucket": str(bucket),
                "rows": int(len(group)),
                "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
                "mae_tvt": float(np.mean(abs_error)),
            }
            for threshold in thresholds:
                row[f"miss_gt_{threshold:g}ft"] = float(np.mean(abs_error > float(threshold)))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_candidate_set_bucket_metrics(
    long_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    thresholds: list[float],
    topk_values: list[int],
    candidate_sets: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = long_frame.merge(context_frame, on=["id", "well"], how="left", validate="many_to_one")
    bucket_families = [
        "distance_bucket",
        "tail_rank_bucket",
        "eval_len_bucket",
        "pf_seed_std_bucket",
        "likpf_delta_bucket",
        "pf_beam_disagreement_bucket",
        "exp056_pf_rmse_min_bucket",
        "exp083_pf_beam_abs_diff_bucket",
    ]
    for candidate_set, candidates in candidate_sets.items():
        set_frame = frame[frame["candidate"].isin(candidates)].copy()
        candidate_count = int(set_frame["candidate"].nunique())
        if candidate_count == 0:
            continue
        for bucket_family in bucket_families:
            for bucket, bucket_frame in set_frame.groupby(bucket_family, observed=True):
                for rank_family, sorted_frame in {
                    "oracle_best_error": bucket_frame.sort_values(
                        ["id", "abs_error"],
                        ascending=[True, True],
                    ),
                    "candidate_rank_score": bucket_frame.sort_values(
                        ["id", "rank_score"],
                        ascending=[True, False],
                    ),
                }.items():
                    for topk in topk_values:
                        k = min(int(topk), candidate_count)
                        topk_subset = sorted_frame.groupby("id", sort=False).head(k)
                        best = (
                            topk_subset.sort_values(
                                ["id", "abs_error"],
                                ascending=[True, True],
                            )
                            .groupby("id", sort=False)
                            .head(1)
                        )
                        if best.empty:
                            continue
                        error = (
                            best["pred_tvt"].to_numpy(np.float32)
                            - best["target_tvt"].to_numpy(np.float32)
                        )
                        abs_error = np.abs(error)
                        row: dict[str, Any] = {
                            "candidate_set": candidate_set,
                            "bucket_family": bucket_family,
                            "bucket": str(bucket),
                            "rank_family": rank_family,
                            "topk": int(k),
                            "candidate_count": int(candidate_count),
                            "rows": int(len(best)),
                            "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
                            "mae_tvt": float(np.mean(abs_error)),
                            "selected_self_gr_rate": float(
                                best["candidate_family"].eq("self_gr").mean()
                            ),
                        }
                        for threshold in thresholds:
                            row[f"within_{threshold:g}ft"] = float(
                                np.mean(abs_error <= float(threshold))
                            )
                        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_well(long_frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, well), group in long_frame.groupby(["candidate", "well"], sort=False):
        error = group["pred_tvt"].to_numpy(np.float32) - group["target_tvt"].to_numpy(np.float32)
        abs_error = np.abs(error)
        row: dict[str, Any] = {
            "candidate": str(candidate),
            "well": str(well),
            "rows": int(len(group)),
            "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
            "mae_tvt": float(np.mean(abs_error)),
        }
        for threshold in thresholds:
            row[f"within_{threshold:g}ft"] = float(np.mean(abs_error <= float(threshold)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["candidate", "rmse_tvt"], ascending=[True, False])


def candidate_sets_from_config(
    config: dict[str, Any],
    *,
    existing_candidate_columns: list[str],
    self_gr_candidate_columns: list[str],
) -> dict[str, list[str]]:
    configured = get_nested(config, "audit.candidate_sets") or config.get("candidate_sets") or []
    if configured:
        out: dict[str, list[str]] = {}
        for item in configured:
            name = str(item["name"])
            out[name] = [str(candidate) for candidate in item.get("candidates", [])]
        return out
    baseline = [
        candidate
        for candidate in existing_candidate_columns
        if candidate not in {"last_anchor_tvt", "last_known_tvt"}
    ]
    return {
        "baseline_primary": baseline,
        "baseline_plus_self_gr": baseline + list(self_gr_candidate_columns),
    }


def summarize_ranker_readiness(
    rank_metrics: pd.DataFrame,
    candidate_set_bucket_metrics: pd.DataFrame,
    *,
    primary_threshold_ft: float = 10.0,
    min_topk_coverage: float = 0.90,
    min_oracle_rmse_gain: float = 1.0,
) -> dict[str, Any]:
    within_col = f"within_{primary_threshold_ft:g}ft"
    oracle = rank_metrics[
        (rank_metrics["rank_family"] == "oracle_best_error")
        & (rank_metrics["topk"] == rank_metrics["candidate_count"])
    ].copy()
    rank_top1 = rank_metrics[
        (rank_metrics["rank_family"] == "candidate_rank_score") & (rank_metrics["topk"] == 1)
    ].copy()
    by_set = {str(row["candidate_set"]): row for _, row in oracle.iterrows()}
    baseline = by_set.get("baseline_primary")
    expanded = by_set.get("baseline_plus_self_gr")
    oracle_rmse_gain = None
    coverage_gain = None
    if baseline is not None and expanded is not None:
        oracle_rmse_gain = float(baseline["rmse_tvt"] - expanded["rmse_tvt"])
        coverage_gain = float(expanded[within_col] - baseline[within_col])
    expanded_coverage = float(expanded[within_col]) if expanded is not None else None
    rank_top1_rows = {
        str(row["candidate_set"]): to_jsonable(row.to_dict()) for _, row in rank_top1.iterrows()
    }
    weak_buckets: list[dict[str, Any]] = []
    has_bucket_metrics = (
        not candidate_set_bucket_metrics.empty
        and within_col in candidate_set_bucket_metrics.columns
    )
    if has_bucket_metrics:
        full_topk = candidate_set_bucket_metrics[
            (candidate_set_bucket_metrics["candidate_set"] == "baseline_plus_self_gr")
            & (candidate_set_bucket_metrics["rank_family"] == "oracle_best_error")
            & (
                candidate_set_bucket_metrics["topk"]
                == candidate_set_bucket_metrics["candidate_count"]
            )
            & (candidate_set_bucket_metrics[within_col] < min_topk_coverage)
        ].copy()
        full_topk = full_topk.sort_values([within_col, "rows"], ascending=[True, False]).head(20)
        weak_buckets = [to_jsonable(row.to_dict()) for _, row in full_topk.iterrows()]

    if expanded_coverage is None:
        recommendation = "blocked_missing_oracle_metrics"
    elif expanded_coverage >= min_topk_coverage and (
        oracle_rmse_gain is None or oracle_rmse_gain >= min_oracle_rmse_gain
    ):
        recommendation = "proceed_to_supervised_candidate_ranker"
    elif expanded_coverage >= min_topk_coverage:
        recommendation = "ranking_or_likelihood_scorer_audit_before_ranker"
    else:
        recommendation = "candidate_generation_failure_map_before_ranker"
    return {
        "primary_threshold_ft": float(primary_threshold_ft),
        "min_topk_coverage": float(min_topk_coverage),
        "min_oracle_rmse_gain": float(min_oracle_rmse_gain),
        "baseline_primary_oracle": (
            to_jsonable(baseline.to_dict()) if baseline is not None else None
        ),
        "baseline_plus_self_gr_oracle": (
            to_jsonable(expanded.to_dict()) if expanded is not None else None
        ),
        "oracle_rmse_gain_from_self_gr": oracle_rmse_gain,
        "oracle_coverage_gain_from_self_gr": coverage_gain,
        "candidate_rank_score_top1": rank_top1_rows,
        "weak_oracle_coverage_buckets": weak_buckets,
        "recommendation": recommendation,
    }


def run_pf_candidate_coverage_then_ranker_audit(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None,
    candidate_specs: list[CandidateSpec],
    extra_source_columns: list[str],
    self_gr_config: dict[str, Any],
    candidate_set_config: dict[str, Any],
    exp056_context_path: str | Path | None,
    exp083_context_path: str | Path | None,
    thresholds: list[float],
    topk_values: list[int],
    max_rows: int | None = None,
    save_candidate_long: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    half_windows = tuple(int(value) for value in self_gr_config.get("half_windows", [8, 15, 25]))
    stride = int(self_gr_config.get("stride", 3))
    prefix_tail_rows = int(self_gr_config.get("prefix_tail_rows", 2048))
    if not half_windows:
        raise ValueError("self_gr half_windows must not be empty")

    required_columns = build_required_columns(candidate_specs, extra_source_columns)
    source_frame, source_meta = read_feature_cache(
        cache_path,
        required_columns=required_columns,
        max_rows=max_rows,
    )
    existing = materialize_existing_candidates(source_frame, candidate_specs)
    self_gr_frame, self_gr_well_summary = build_self_gr_candidate_frame(
        source_frame,
        train_dir=train_dir,
        half_windows=half_windows,
        stride=stride,
        prefix_tail_rows=prefix_tail_rows,
    )
    full_frame = existing.merge(self_gr_frame, on=["id", "well"], how="left", validate="one_to_one")
    if full_frame.isna().any().any():
        raise ValueError("candidate merge produced missing values")

    self_gr_columns = {"self_gr_ens", "self_gr_best"}
    for scale in half_windows:
        self_gr_columns.add(f"self_gr_sc{int(scale)}")
    existing_candidate_columns = [spec.name for spec in candidate_specs if spec.enabled]
    self_gr_candidate_columns = [
        "self_gr_ens",
        "self_gr_best",
        *[f"self_gr_sc{int(scale)}" for scale in half_windows],
    ]
    candidate_columns = existing_candidate_columns + self_gr_candidate_columns
    candidate_sets = candidate_sets_from_config(
        candidate_set_config,
        existing_candidate_columns=existing_candidate_columns,
        self_gr_candidate_columns=self_gr_candidate_columns,
    )
    long_frame = build_candidate_long_frame(
        full_frame,
        candidate_columns,
        self_gr_columns=self_gr_columns,
    )
    context_frame, context_meta = merge_well_context(
        source_frame,
        exp056_context_path=exp056_context_path,
        exp083_context_path=exp083_context_path,
    )

    candidate_metrics = summarize_candidate_metrics(long_frame, thresholds)
    rank_metrics = summarize_oracle_and_rank_metrics_for_sets(
        long_frame,
        thresholds,
        topk_values,
        candidate_sets,
    )
    bucket_metrics = summarize_bucket_metrics(long_frame, context_frame, thresholds)
    candidate_set_bucket_metrics = summarize_candidate_set_bucket_metrics(
        long_frame,
        context_frame,
        thresholds,
        topk_values,
        candidate_sets,
    )
    by_well = summarize_by_well(long_frame, thresholds)
    readiness = summarize_ranker_readiness(
        rank_metrics,
        candidate_set_bucket_metrics,
        primary_threshold_ft=float(candidate_set_config.get("primary_threshold_ft", 10.0)),
        min_topk_coverage=float(candidate_set_config.get("min_topk_coverage", 0.90)),
        min_oracle_rmse_gain=float(candidate_set_config.get("min_oracle_rmse_gain", 1.0)),
    )

    candidate_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_candidate_metrics.csv", index=False)
    rank_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_rank_metrics.csv", index=False)
    bucket_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv", index=False)
    candidate_set_bucket_metrics.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_candidate_set_bucket_metrics.csv",
        index=False,
    )
    by_well.to_csv(output_dir / f"{OUTPUT_PREFIX}_by_well.csv", index=False)
    self_gr_well_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_self_gr_well_summary.csv",
        index=False,
    )
    context_frame.to_csv(output_dir / f"{OUTPUT_PREFIX}_row_context.csv.gz", index=False)
    source_schema = pd.DataFrame(
        [{"column": column, "role": "source"} for column in source_frame.columns]
        + [{"column": column, "role": "candidate"} for column in candidate_columns]
    )
    source_schema.to_csv(output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv", index=False)
    if save_candidate_long:
        long_frame.to_csv(
            output_dir / f"{OUTPUT_PREFIX}_candidate_long.csv.gz",
            index=False,
            compression="gzip",
        )

    best_candidate = candidate_metrics.iloc[0].to_dict() if not candidate_metrics.empty else {}
    best_oracle = (
        rank_metrics.sort_values(["candidate_set", "rank_family", "topk"]).iloc[0].to_dict()
        if not rank_metrics.empty
        else {}
    )
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "completed_train_side_audit" if max_rows is None else "debug_completed",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - t0),
        "source": source_meta,
        "well_context": context_meta,
        "self_gr": {
            "half_windows": list(half_windows),
            "stride": int(stride),
            "prefix_tail_rows": int(prefix_tail_rows),
            "candidate_columns": sorted(self_gr_columns),
            "well_summary_rows": int(len(self_gr_well_summary)),
        },
        "candidates": candidate_columns,
        "candidate_sets": candidate_sets,
        "thresholds_ft": thresholds,
        "topk_values": topk_values,
        "best_candidate_by_rmse": to_jsonable(best_candidate),
        "rank_metric_first_row": to_jsonable(best_oracle),
        "ranker_readiness": to_jsonable(readiness),
        "outputs": {
            "candidate_metrics": f"{OUTPUT_PREFIX}_candidate_metrics.csv",
            "rank_metrics": f"{OUTPUT_PREFIX}_rank_metrics.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "candidate_set_bucket_metrics": (
                f"{OUTPUT_PREFIX}_candidate_set_bucket_metrics.csv"
            ),
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "self_gr_well_summary": f"{OUTPUT_PREFIX}_self_gr_well_summary.csv",
            "candidate_long": f"{OUTPUT_PREFIX}_candidate_long.csv.gz"
            if save_candidate_long
            else None,
            "row_context": f"{OUTPUT_PREFIX}_row_context.csv.gz",
            "feature_schema": f"{OUTPUT_PREFIX}_feature_schema.csv",
        },
    }
    with (output_dir / f"{OUTPUT_PREFIX}_summary.json").open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
    return summary


def candidate_specs_from_config(config: dict[str, Any]) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for item in get_nested(config, "audit.candidates") or []:
        specs.append(
            CandidateSpec(
                name=str(item["name"]),
                source_column=str(item.get("source_column") or item["name"]),
                transform=str(item.get("transform", "absolute")),
                role=str(item.get("role", item.get("name", "candidate"))),
                enabled=bool(item.get("enabled", True)),
            )
        )
    if not specs:
        raise ValueError("audit.candidates must configure at least one candidate")
    return specs


def run_from_config(
    config: dict[str, Any] | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    audit_config = get_nested(config, "audit") or {}
    return run_pf_candidate_coverage_then_ranker_audit(
        output_dir=output_dir or paths.artifacts_dir,
        train_dir=paths.train_data_dir,
        cache_path=get_nested(config, "data.exp072_train_feature_cache_local"),
        candidate_specs=candidate_specs_from_config(config),
        extra_source_columns=[
            str(col) for col in get_nested(config, "audit.extra_source_columns") or []
        ],
        self_gr_config=get_nested(config, "model.self_gr_candidate") or {},
        candidate_set_config=audit_config,
        exp056_context_path=get_nested(config, "data.exp056_well_summary_local"),
        exp083_context_path=get_nested(config, "data.exp083_well_summary_local"),
        thresholds=[
            float(value) for value in get_nested(config, "audit.thresholds_ft") or [1, 2, 5, 10]
        ],
        topk_values=[
            int(value) for value in get_nested(config, "audit.topk_values") or [1, 2, 3, 5, 10]
        ],
        max_rows=get_nested(config, "audit.max_rows"),
        save_candidate_long=bool(get_nested(config, "audit.save_candidate_long") is not False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    config = load_config()
    if args.max_rows is not None:
        config.setdefault("audit", {})["max_rows"] = args.max_rows
    summary = run_from_config(config, output_dir=args.output_dir)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import gc
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

OUTPUT_PREFIX = "exp176_typewell_late_range_pfbeam_candidate_prior"
DEFAULT_TRAIN_FEATURE_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
DEFAULT_TRAIN_FEATURE_SCHEMA = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv"
)
DEFAULT_DENSE_FEATURE_CACHE = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
DEFAULT_DENSE_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
PROTECTED_COLUMNS = {"id", "well", "target", "true_tvt", "oracle_label", "oracle_candidate"}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    column: str


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


def find_artifact(filename: str, explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
            Path("experiments")
            / "exp099_pf_multi_observation_likelihood_probe"
            / "kaggle"
            / "output"
            / "train_v2"
            / "artifacts"
            / filename,
            Path("experiments")
            / "exp072_exp063_full_replay_feature_cache"
            / "artifacts"
            / filename,
        ]
    )
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


def load_train_feature_cache(
    *,
    cache_path: str | Path | None,
    schema_path: str | Path | None,
    required_columns: list[str],
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(DEFAULT_TRAIN_FEATURE_CACHE, cache_path)
    schema = find_artifact(DEFAULT_TRAIN_FEATURE_SCHEMA, schema_path)
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
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": (
            sha256_path(source, decompressed=True) if source.suffix == ".gz" else None
        ),
        "schema_path": str(schema),
        "schema_sha256": sha256_path(schema),
    }
    return frame, meta


def candidate_specs_from_config(config: dict[str, Any]) -> list[CandidateSpec]:
    values = get_nested(config, "ranker.candidates") or []
    specs: list[CandidateSpec] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("ranker.candidates entries must be mappings")
        specs.append(
            CandidateSpec(name=str(item["name"]), column=str(item.get("column", item["name"])))
        )
    if not specs:
        raise ValueError("ranker.candidates must not be empty")
    return specs


def build_required_columns(config: dict[str, Any], candidates: list[CandidateSpec]) -> list[str]:
    required = {"id", "well", "target", "last_known_tvt"}
    auxiliary_columns = set(
        get_nested(config, "ranker.feature_enrichment.auxiliary_candidate_columns") or []
    )
    required.update(spec.column for spec in candidates if spec.column not in auxiliary_columns)
    for key in [
        "ranker.context_columns",
        "ranker.multiobs_feature_columns",
        "ranker.optional_columns",
    ]:
        values = get_nested(config, key) or []
        required.update(str(value) for value in values if str(value) not in auxiliary_columns)
    return sorted(required)


def _rank01(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values.astype(np.float32))
    return series.rank(method="average", pct=True).fillna(0.5).to_numpy(np.float32)


def pct_label(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p").replace("-", "m")


def _as_float_list(values: Any, default: list[float]) -> list[float]:
    if values is None:
        return default
    return [float(value) for value in values]


def _candidate_pct(
    candidate_tvt: np.ndarray,
    typewell_min: np.ndarray,
    typewell_span: np.ndarray,
) -> np.ndarray:
    pct = (candidate_tvt.astype(np.float32) - typewell_min.astype(np.float32)) / np.maximum(
        typewell_span.astype(np.float32),
        np.float32(1e-6),
    )
    return pct.astype(np.float32)


def read_typewell_late_range_context(
    *,
    train_dir: str | Path,
    min_typewell_span: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_dir = Path(train_dir)
    rows: list[dict[str, Any]] = []
    for typewell_path in sorted(train_dir.glob("*__typewell.csv")):
        well = typewell_path.name.replace("__typewell.csv", "")
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        if not horizontal_path.exists():
            continue
        typewell = pd.read_csv(typewell_path, usecols=["TVT"])
        horizontal = pd.read_csv(
            horizontal_path,
            usecols=lambda col: col in {"MD", "TVT_input"},
        )
        tvt = pd.to_numeric(typewell["TVT"], errors="coerce").dropna()
        if tvt.empty:
            continue
        typewell_min = float(tvt.min())
        typewell_max = float(tvt.max())
        typewell_span = float(typewell_max - typewell_min)
        known = horizontal[pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna()]
        if known.empty:
            last_known_tvt = np.nan
            last_known_md = np.nan
        else:
            last_known_tvt = float(known["TVT_input"].iloc[-1])
            last_known_md = float(known["MD"].iloc[-1]) if "MD" in known.columns else np.nan
        known_last_pct = (
            (last_known_tvt - typewell_min) / typewell_span
            if typewell_span >= min_typewell_span and np.isfinite(last_known_tvt)
            else np.nan
        )
        rows.append(
            {
                "well": well,
                "typewell_min": typewell_min,
                "typewell_max": typewell_max,
                "typewell_span": typewell_span,
                "known_last_tvt_raw": last_known_tvt,
                "known_last_md_raw": last_known_md,
                "known_last_pct": known_last_pct,
                "valid_typewell_span": bool(typewell_span >= min_typewell_span),
            }
        )
    context = pd.DataFrame(rows)
    if context.empty:
        raise ValueError(f"No typewell context was read from {train_dir}")
    valid_context = context[context["valid_typewell_span"] & context["known_last_pct"].notna()]
    meta = {
        "train_dir": str(train_dir),
        "context_rows": int(len(context)),
        "valid_context_rows": int(len(valid_context)),
        "min_typewell_span": float(min_typewell_span),
    }
    return context, meta


def load_auxiliary_feature_cache(
    *,
    cache_path: str | Path | None,
    schema_path: str | Path | None,
    required_columns: list[str],
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(DEFAULT_DENSE_FEATURE_CACHE, cache_path)
    schema = find_artifact(DEFAULT_DENSE_FEATURE_SCHEMA, schema_path)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"{source} is missing auxiliary columns: {missing}")
    frame = pd.read_csv(
        source,
        usecols=required_columns,
        nrows=max_rows,
        dtype={"id": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    for column in frame.columns:
        if column != "id":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    meta = {
        "path": str(source),
        "rows": int(len(frame)),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": (
            sha256_path(source, decompressed=True) if source.suffix == ".gz" else None
        ),
        "schema_path": str(schema),
        "schema_sha256": sha256_path(schema),
    }
    return frame, meta


def add_feature_enrichment(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    max_rows: int | None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    enrichment = get_nested(config, "ranker.feature_enrichment") or {}
    if not enrichment.get("enabled", False):
        return frame, [], {"enabled": False}

    auxiliary_columns = ["id", *[str(value) for value in enrichment.get("auxiliary_columns", [])]]
    auxiliary, source_meta = load_auxiliary_feature_cache(
        cache_path=get_nested(config, "data.exp072_train_feature_cache_local"),
        schema_path=get_nested(config, "data.exp072_feature_schema_local"),
        required_columns=auxiliary_columns,
        max_rows=max_rows,
    )
    before_rows = len(frame)
    frame = frame.merge(auxiliary, on="id", how="left", validate="one_to_one")
    if len(frame) != before_rows:
        raise ValueError("auxiliary feature join changed row count")

    missing_rate = frame[auxiliary_columns[1:]].isna().mean().max()
    if missing_rate > float(enrichment.get("max_missing_rate", 0.0)):
        raise ValueError(f"auxiliary feature join missing_rate={missing_rate:.6f}")

    generated: dict[str, np.ndarray] = {}
    last_tvt = frame["last_known_tvt"].to_numpy(np.float32)
    dense_delta_columns = enrichment.get("dense_delta_columns", {})
    dense_candidate_names: list[str] = []
    for candidate_name, delta_column in dense_delta_columns.items():
        candidate_name = str(candidate_name)
        delta_column = str(delta_column)
        if delta_column not in frame.columns:
            raise ValueError(f"missing dense delta column: {delta_column}")
        delta = frame[delta_column].to_numpy(np.float32)
        frame[candidate_name] = (last_tvt + delta).astype(np.float32)
        dense_candidate_names.append(candidate_name)
        generated[f"{candidate_name}_abs_delta_from_last"] = np.abs(delta).astype(np.float32)

    if len(dense_candidate_names) >= 2:
        dense_values = frame[dense_candidate_names].to_numpy(np.float32)
        generated["dense_candidate_mean"] = np.mean(dense_values, axis=1).astype(np.float32)
        generated["dense_candidate_std"] = np.std(dense_values, axis=1).astype(np.float32)
        generated["dense_candidate_range"] = (
            np.max(dense_values, axis=1) - np.min(dense_values, axis=1)
        ).astype(np.float32)

    primary_dense = str(enrichment.get("primary_dense_candidate", "tvt_densew"))
    reference_candidates = [str(value) for value in enrichment.get("reference_candidates", [])]
    dense_scale = np.maximum(
        np.abs(frame.get("dense_std", pd.Series(0.0, index=frame.index)).to_numpy(np.float32)),
        float(enrichment.get("min_dense_scale", 10.0)),
    )
    for ref in reference_candidates:
        if ref in frame.columns and primary_dense in frame.columns:
            diff = frame[ref].to_numpy(np.float32) - frame[primary_dense].to_numpy(np.float32)
            generated[f"{ref}_minus_{primary_dense}"] = diff.astype(np.float32)
            generated[f"{ref}_minus_{primary_dense}_abs_norm"] = (
                np.abs(diff) / dense_scale
            ).astype(np.float32)

    if "md_since" in frame.columns:
        md_since = frame["md_since"].to_numpy(np.float32)
    else:
        md_since = _row_indices_from_ids(frame["id"]).astype(np.float32)
    md_scale = np.maximum(np.abs(md_since), float(enrichment.get("min_md_scale", 25.0)))
    row_index = _row_indices_from_ids(frame["id"]).astype(np.float32)
    generated["tail_rank_norm"] = np.minimum(row_index / 1000.0, 5.0).astype(np.float32)
    generated["longtail_1000_flag"] = (row_index >= 1000.0).astype(np.float32)
    generated["near_md_50_flag"] = (
        md_since <= float(enrichment.get("near_md_threshold", 50.0))
    ).astype(np.float32)
    for candidate_name in dense_candidate_names:
        delta_col = str(dense_delta_columns[candidate_name])
        delta = frame[delta_col].to_numpy(np.float32)
        generated[f"{candidate_name}_drift_per_md"] = (delta / md_scale).astype(np.float32)

    pf_vs_dense_abs_norm = (
        np.abs(frame["pf_vs_dense"].to_numpy(np.float32)) / dense_scale
        if "pf_vs_dense" in frame.columns
        else np.zeros(len(frame), dtype=np.float32)
    )
    dense_std_norm = (
        np.abs(frame["dense_std"].to_numpy(np.float32)) / dense_scale
        if "dense_std" in frame.columns
        else np.zeros(len(frame), dtype=np.float32)
    )
    dense_dist_norm = (
        np.abs(frame["dense_dist"].to_numpy(np.float32)) / dense_scale
        if "dense_dist" in frame.columns
        else np.zeros(len(frame), dtype=np.float32)
    )
    high_disagreement = (
        0.45 * _rank01(pf_vs_dense_abs_norm)
        + 0.35 * _rank01(dense_std_norm)
        + 0.20 * _rank01(dense_dist_norm)
    ).astype(np.float32)
    generated["pf_vs_dense_abs_norm"] = pf_vs_dense_abs_norm.astype(np.float32)
    generated["dense_std_norm"] = dense_std_norm.astype(np.float32)
    generated["dense_dist_norm"] = dense_dist_norm.astype(np.float32)
    generated["high_disagreement_proxy"] = high_disagreement
    generated["high_disagreement_x_longtail"] = (
        high_disagreement * generated["longtail_1000_flag"]
    ).astype(np.float32)

    prefix = str(enrichment.get("prefix", "crfe_"))
    generated_columns: list[str] = []
    for name, values in generated.items():
        column = f"{prefix}{name}"
        frame[column] = values.astype(np.float32)
        generated_columns.append(column)
    bad_columns = [
        column
        for column in generated_columns
        if not np.isfinite(frame[column].to_numpy(np.float32)).all()
    ]
    if bad_columns:
        raise ValueError(f"feature enrichment columns contain non-finite values: {bad_columns}")

    meta = {
        "enabled": True,
        "source": source_meta,
        "joined_rows": int(len(frame)),
        "missing_rate_max": float(missing_rate),
        "dense_candidate_names": dense_candidate_names,
        "generated_feature_count": int(len(generated_columns)),
        "generated_feature_columns": generated_columns,
    }
    return frame, generated_columns, meta


def add_typewell_late_range_prior(
    frame: pd.DataFrame,
    config: dict[str, Any],
    candidates: list[CandidateSpec],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    prior = get_nested(config, "ranker.typewell_late_range_prior") or {}
    if not prior.get("enabled", False):
        return frame, [], {"enabled": False}

    paths = ExperimentPaths()
    context, context_meta = read_typewell_late_range_context(
        train_dir=paths.train_data_dir,
        min_typewell_span=float(prior.get("min_typewell_span", 1.0)),
    )
    before_rows = len(frame)
    context_columns = [
        "well",
        "typewell_min",
        "typewell_max",
        "typewell_span",
        "known_last_pct",
        "valid_typewell_span",
    ]
    frame = frame.merge(context[context_columns], on="well", how="left", validate="many_to_one")
    if len(frame) != before_rows:
        raise ValueError("typewell context join changed row count")

    missing_rate = frame[["typewell_min", "typewell_span", "known_last_pct"]].isna().mean().max()
    if missing_rate > float(prior.get("max_context_missing_rate", 0.0)):
        raise ValueError(f"typewell context join missing_rate={missing_rate:.6f}")
    invalid = ~frame["valid_typewell_span"].fillna(False).astype(bool)
    if invalid.any():
        examples = sorted(frame.loc[invalid, "well"].unique())[:10]
        raise ValueError(f"invalid typewell context for wells: {examples}")

    prefix = str(prior.get("row_feature_prefix", "tlp_"))
    lower_bounds = _as_float_list(prior.get("candidate_pct_lower_bounds"), [0.5, 0.6, 0.7])
    known_thresholds = _as_float_list(prior.get("known_last_pct_min"), [0.75, 0.80])
    known_margins = _as_float_list(prior.get("known_last_margins"), [0.05, 0.10])

    generated: dict[str, np.ndarray] = {}
    known_last_pct = frame["known_last_pct"].to_numpy(np.float32)
    typewell_min = frame["typewell_min"].to_numpy(np.float32)
    typewell_span = frame["typewell_span"].to_numpy(np.float32)
    generated["known_last_pct"] = known_last_pct
    generated["typewell_span_log1p"] = np.log1p(np.maximum(typewell_span, 0.0)).astype(np.float32)
    for threshold in known_thresholds:
        generated[f"known_last_ge_{pct_label(threshold)}"] = (
            known_last_pct >= np.float32(threshold)
        ).astype(np.float32)

    pct_values: dict[str, np.ndarray] = {}
    for spec in candidates:
        pct = _candidate_pct(
            frame[spec.column].to_numpy(np.float32),
            typewell_min,
            typewell_span,
        )
        pct_values[spec.name] = pct
        generated[f"{spec.name}_candidate_pct"] = pct
        generated[f"{spec.name}_candidate_pct_minus_known_last_pct"] = (
            pct - known_last_pct
        ).astype(np.float32)
        for lower in lower_bounds:
            below = pct < np.float32(lower)
            generated[f"{spec.name}_candidate_pct_below_{pct_label(lower)}"] = below.astype(
                np.float32
            )
        for margin in known_margins:
            dynamic_lower = np.clip(known_last_pct - np.float32(margin), 0.0, 1.0)
            generated[f"{spec.name}_candidate_pct_below_known_last_m{pct_label(margin)}"] = (
                pct < dynamic_lower
            ).astype(np.float32)

    pct_matrix = np.column_stack([pct_values[spec.name] for spec in candidates]).astype(np.float32)
    generated["candidate_pct_min"] = np.min(pct_matrix, axis=1).astype(np.float32)
    generated["candidate_pct_max"] = np.max(pct_matrix, axis=1).astype(np.float32)
    generated["candidate_pct_mean"] = np.mean(pct_matrix, axis=1).astype(np.float32)
    generated["candidate_pct_std"] = np.std(pct_matrix, axis=1).astype(np.float32)
    generated["candidate_pct_range"] = (
        generated["candidate_pct_max"] - generated["candidate_pct_min"]
    ).astype(np.float32)
    for lower in lower_bounds:
        below = pct_matrix < np.float32(lower)
        generated[f"candidate_pct_below_{pct_label(lower)}_count"] = below.sum(axis=1).astype(
            np.float32
        )
        generated[f"candidate_pct_below_{pct_label(lower)}_rate"] = below.mean(axis=1).astype(
            np.float32
        )
        for threshold in known_thresholds:
            late = known_last_pct >= np.float32(threshold)
            generated[
                f"known_last_ge_{pct_label(threshold)}_x_candidate_below_{pct_label(lower)}_rate"
            ] = (below.mean(axis=1) * late.astype(np.float32)).astype(np.float32)

    generated_columns: list[str] = []
    for name, values in generated.items():
        column = f"{prefix}{name}"
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"typewell late-range feature contains non-finite values: {column}")
        frame[column] = values
        generated_columns.append(column)

    meta = {
        "enabled": True,
        "source": context_meta,
        "joined_rows": int(len(frame)),
        "missing_rate_max": float(missing_rate),
        "known_last_pct_min": known_thresholds,
        "candidate_pct_lower_bounds": lower_bounds,
        "known_last_margins": known_margins,
        "generated_feature_count": int(len(generated_columns)),
        "generated_feature_columns": generated_columns,
    }
    return frame, generated_columns, meta


def add_candidate_late_range_columns(
    part: pd.DataFrame,
    source_rows: pd.DataFrame,
    candidate_tvt: np.ndarray,
    prior: dict[str, Any],
) -> None:
    if not prior.get("enabled", False):
        return
    prefix = str(prior.get("candidate_feature_prefix", "candidate_tlp_"))
    lower_bounds = _as_float_list(prior.get("candidate_pct_lower_bounds"), [0.5, 0.6, 0.7])
    known_thresholds = _as_float_list(prior.get("known_last_pct_min"), [0.75, 0.80])
    known_margins = _as_float_list(prior.get("known_last_margins"), [0.05, 0.10])
    typewell_min = source_rows["typewell_min"].to_numpy(np.float32)
    typewell_span = source_rows["typewell_span"].to_numpy(np.float32)
    known_last_pct = source_rows["known_last_pct"].to_numpy(np.float32)
    candidate_pct = _candidate_pct(candidate_tvt.astype(np.float32), typewell_min, typewell_span)
    part[f"{prefix}candidate_pct"] = candidate_pct.astype(np.float32)
    part[f"{prefix}candidate_pct_minus_known_last_pct"] = (candidate_pct - known_last_pct).astype(
        np.float32
    )
    part[f"{prefix}known_last_pct"] = known_last_pct.astype(np.float32)
    risk_terms: list[np.ndarray] = []
    for lower in lower_bounds:
        below = candidate_pct < np.float32(lower)
        gap = np.maximum(np.float32(lower) - candidate_pct, 0.0).astype(np.float32)
        part[f"{prefix}candidate_pct_below_{pct_label(lower)}"] = below.astype(np.float32)
        part[f"{prefix}candidate_pct_gap_to_{pct_label(lower)}"] = gap
        for threshold in known_thresholds:
            late = known_last_pct >= np.float32(threshold)
            interaction = (late & below).astype(np.float32)
            part[f"{prefix}known_last_ge_{pct_label(threshold)}_x_below_{pct_label(lower)}"] = (
                interaction
            )
            risk_terms.append(interaction * gap)
    for margin in known_margins:
        dynamic_lower = np.clip(known_last_pct - np.float32(margin), 0.0, 1.0)
        below_dynamic = candidate_pct < dynamic_lower
        gap_dynamic = np.maximum(dynamic_lower - candidate_pct, 0.0).astype(np.float32)
        part[f"{prefix}candidate_pct_below_known_last_m{pct_label(margin)}"] = below_dynamic.astype(
            np.float32
        )
        part[f"{prefix}candidate_pct_gap_to_known_last_m{pct_label(margin)}"] = gap_dynamic
        risk_terms.append(below_dynamic.astype(np.float32) * gap_dynamic)
    if risk_terms:
        part[f"{prefix}risk_score"] = np.maximum.reduce(risk_terms).astype(np.float32)


def add_candidate_labels_and_features(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    *,
    include_candidate_values: bool,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    out = frame.copy()
    out["true_tvt"] = out["last_known_tvt"].astype(np.float32) + out["target"].astype(np.float32)
    candidate_values = np.column_stack(
        [
            pd.to_numeric(out[spec.column], errors="coerce").to_numpy(np.float32)
            for spec in candidates
        ]
    )
    if not np.isfinite(candidate_values).all():
        bad = np.argwhere(~np.isfinite(candidate_values))[:5].tolist()
        raise ValueError(f"candidate values contain non-finite values, examples={bad}")
    true_tvt = out["true_tvt"].to_numpy(np.float32)
    errors = np.abs(candidate_values - true_tvt[:, None])
    labels = np.argmin(errors, axis=1).astype(np.int16)
    out["oracle_label"] = labels
    out["oracle_candidate"] = np.asarray([candidates[i].name for i in labels], dtype=object)

    feature_columns: list[str] = []
    for spec in candidates:
        delta_col = f"{spec.name}_minus_last"
        out[delta_col] = out[spec.column].astype(np.float32) - out["last_known_tvt"].astype(
            np.float32
        )
        feature_columns.append(delta_col)
        if include_candidate_values:
            feature_columns.append(spec.column)

    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            col = f"{left.name}_vs_{right.name}_abs"
            out[col] = np.abs(
                out[left.column].astype(np.float32) - out[right.column].astype(np.float32)
            )
            feature_columns.append(col)

    value_cols = [spec.column for spec in candidates]
    out["candidate_mean"] = out[value_cols].mean(axis=1).astype(np.float32)
    out["candidate_std"] = out[value_cols].std(axis=1).astype(np.float32)
    out["candidate_range"] = (out[value_cols].max(axis=1) - out[value_cols].min(axis=1)).astype(
        np.float32
    )
    feature_columns.extend(["candidate_mean", "candidate_std", "candidate_range"])
    return out, feature_columns, candidate_values, labels


def select_numeric_feature_columns(
    frame: pd.DataFrame,
    config: dict[str, Any],
    engineered_columns: list[str],
) -> list[str]:
    configured = [
        str(value)
        for value in (
            (get_nested(config, "ranker.context_columns") or [])
            + (get_nested(config, "ranker.multiobs_feature_columns") or [])
            + (get_nested(config, "ranker.feature_enrichment.base_feature_columns") or [])
        )
    ]
    columns: list[str] = []
    for column in configured + engineered_columns:
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
    train_values = train[columns].to_numpy(dtype=np.float32, copy=True)
    valid_values = valid[columns].to_numpy(dtype=np.float32, copy=True)
    train_values[~np.isfinite(train_values)] = np.nan
    valid_values[~np.isfinite(valid_values)] = np.nan
    medians = np.nanmedian(train_values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    train_bad = ~np.isfinite(train_values)
    valid_bad = ~np.isfinite(valid_values)
    if train_bad.any():
        train_values[train_bad] = np.take(medians, np.where(train_bad)[1])
    if valid_bad.any():
        valid_values[valid_bad] = np.take(medians, np.where(valid_bad)[1])
    return train_values, valid_values


def select_long_row_feature_columns(
    config: dict[str, Any], feature_columns: list[str]
) -> list[str]:
    long_config = get_nested(config, "ranker.long_models") or {}
    exclude_prefixes = [str(value) for value in long_config.get("row_feature_exclude_prefixes", [])]
    keep_columns = {str(value) for value in long_config.get("row_feature_keep_columns", [])}
    selected: list[str] = []
    for column in feature_columns:
        if column in keep_columns:
            selected.append(column)
            continue
        if any(column.startswith(prefix) for prefix in exclude_prefixes):
            continue
        selected.append(column)
    if not selected:
        raise ValueError("long row feature column selection is empty")
    return selected


def build_long_frame(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    candidates: list[CandidateSpec],
    *,
    row_feature_columns: list[str],
    candidate_values: np.ndarray,
    oracle_labels: np.ndarray,
    late_range_prior: dict[str, Any],
    sample_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if sample_rows is not None and len(row_indices) > sample_rows:
        rng = np.random.default_rng(seed)
        row_indices = np.sort(rng.choice(row_indices, size=int(sample_rows), replace=False))
    chunks: list[pd.DataFrame] = []
    y_error_chunks: list[np.ndarray] = []
    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    last_known_tvt = frame["last_known_tvt"].to_numpy(np.float32)
    source_rows = frame.iloc[row_indices]
    for cand_idx, spec in enumerate(candidates):
        part = source_rows[row_feature_columns].copy()
        part["candidate_index"] = np.int16(cand_idx)
        part["candidate_name_code"] = np.int16(cand_idx)
        part["candidate_tvt"] = candidate_values[row_indices, cand_idx].astype(np.float32)
        part["candidate_minus_last"] = (
            candidate_values[row_indices, cand_idx] - last_known_tvt[row_indices]
        ).astype(np.float32)
        add_candidate_late_range_columns(
            part,
            source_rows,
            candidate_values[row_indices, cand_idx],
            late_range_prior,
        )
        score_col = f"multiobs_score_{spec.name}"
        mae_col = f"multiobs_mae_{spec.name}"
        ncc_col = f"multiobs_ncc_{spec.name}"
        part["candidate_multiobs_score"] = (
            source_rows[score_col].to_numpy(np.float32) if score_col in frame.columns else 0.0
        )
        part["candidate_multiobs_mae"] = (
            source_rows[mae_col].to_numpy(np.float32) if mae_col in frame.columns else 0.0
        )
        part["candidate_multiobs_ncc"] = (
            source_rows[ncc_col].to_numpy(np.float32) if ncc_col in frame.columns else 0.0
        )
        part["is_oracle"] = (oracle_labels[row_indices] == cand_idx).astype(np.int8)
        y_error_chunks.append(
            np.abs(candidate_values[row_indices, cand_idx] - true_tvt[row_indices])
        )
        chunks.append(part)
    long_frame = pd.concat(chunks, ignore_index=True)
    y_error = np.concatenate(y_error_chunks).astype(np.float32)
    return long_frame, y_error


def evaluate_selection(
    *,
    frame: pd.DataFrame,
    selected_indices: np.ndarray,
    candidate_values: np.ndarray,
    oracle_labels: np.ndarray,
    candidate_names: list[str],
    variant: str,
    mode: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    selected_tvt = candidate_values[np.arange(len(frame)), selected_indices].astype(np.float32)
    abs_error = np.abs(selected_tvt - true_tvt)
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


def selection_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_by_variant = (
        predictions.groupby(["variant", "mode"], observed=True).size().rename("total")
    )
    counts = (
        predictions.groupby(["variant", "mode", "selected_candidate"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
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
    ]:
        if source_column in source_frame.columns:
            context[bucket_name] = _quantile_bucket(source_frame[source_column], bucket_name)
    merged = predictions.merge(context, on="id", how="left", validate="many_to_one")
    rows = []
    bucket_cols = [col for col in context.columns if col != "id"]
    for bucket_col in bucket_cols:
        for (variant, mode, bucket), group in merged.groupby(
            ["variant", "mode", bucket_col],
            observed=True,
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


def top1_from_multiobs_scores(frame: pd.DataFrame, candidates: list[CandidateSpec]) -> np.ndarray:
    score_cols = [f"multiobs_score_{spec.name}" for spec in candidates]
    if not all(col in frame.columns for col in score_cols):
        return np.full(
            len(frame), [spec.name for spec in candidates].index("likpf_mean"), dtype=np.int16
        )
    scores = frame[score_cols].replace([np.inf, -np.inf], np.nan).fillna(-1e9).to_numpy(np.float32)
    return np.argmax(scores, axis=1).astype(np.int16)


def train_and_score(
    *,
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    candidate_values: np.ndarray,
    oracle_labels: np.ndarray,
    feature_columns: list[str],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

    seed = int(get_nested(config, "validation.seed") or 42)
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    log_period = int(get_nested(config, "ranker.log_period") or 100)
    max_train_rows = get_nested(config, "ranker.long_models.max_train_rows_per_fold")
    max_train_rows = int(max_train_rows) if max_train_rows is not None else None
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    candidate_names = [spec.name for spec in candidates]

    cv = GroupKFold(n_splits=n_folds)
    oof_selected: dict[str, np.ndarray] = {
        "lgb_multiclass": np.zeros(len(frame), dtype=np.int16),
        "lgb_candidate_binary": np.zeros(len(frame), dtype=np.int16),
        "lgb_candidate_error_ranker": np.zeros(len(frame), dtype=np.int16),
    }
    importance_rows: list[dict[str, Any]] = []
    model_manifest: list[dict[str, Any]] = []
    folds = list(cv.split(frame, oracle_labels, groups=frame["well"]))
    multiclass_params = dict(get_nested(config, "ranker.multiclass_lgbm.params") or {})
    binary_params = dict(get_nested(config, "ranker.long_models.binary_lgbm.params") or {})
    error_params = dict(get_nested(config, "ranker.long_models.error_lgbm.params") or {})
    late_range_prior = get_nested(config, "ranker.typewell_late_range_prior") or {}
    row_features = select_long_row_feature_columns(config, feature_columns)
    print(
        "[train] feature_columns="
        f"{len(feature_columns)} long_row_features={len(row_features)} "
        f"max_long_train_rows_per_fold={max_train_rows}",
        flush=True,
    )

    for fold, (train_idx, valid_idx) in enumerate(folds):
        print(f"[fold {fold}] train={len(train_idx)} valid={len(valid_idx)}", flush=True)
        train_frame = frame.iloc[train_idx]
        valid_frame = frame.iloc[valid_idx]
        x_train, x_valid = fit_impute(train_frame, valid_frame, feature_columns)
        y_train = oracle_labels[train_idx]
        y_valid = oracle_labels[valid_idx]

        multiclass = LGBMClassifier(
            objective="multiclass",
            num_class=len(candidates),
            random_state=seed + fold,
            **multiclass_params,
        )
        multiclass.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(50), log_evaluation(log_period)],
        )
        oof_selected["lgb_multiclass"][valid_idx] = multiclass.predict(x_valid).astype(np.int16)
        model_path = model_dir / f"{OUTPUT_PREFIX}_lgb_multiclass_fold{fold}.txt"
        multiclass.booster_.save_model(str(model_path))
        model_manifest.append(
            {
                "variant": "lgb_multiclass",
                "fold": fold,
                "path": str(model_path.relative_to(output_dir)),
                "sha256": sha256_path(model_path),
                "best_iteration": int(multiclass.best_iteration_ or multiclass.n_estimators),
            }
        )
        for feature, importance in zip(
            feature_columns, multiclass.feature_importances_, strict=False
        ):
            importance_rows.append(
                {
                    "variant": "lgb_multiclass",
                    "fold": fold,
                    "feature": feature,
                    "importance": float(importance),
                }
            )
        del x_train, x_valid
        gc.collect()

        long_train, train_error = build_long_frame(
            frame,
            train_idx,
            candidates,
            row_feature_columns=row_features,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            late_range_prior=late_range_prior,
            sample_rows=max_train_rows,
            seed=seed + 101 * fold,
        )
        long_valid, _ = build_long_frame(
            frame,
            valid_idx,
            candidates,
            row_feature_columns=row_features,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            late_range_prior=late_range_prior,
            sample_rows=None,
            seed=seed,
        )
        long_feature_columns = [
            col
            for col in long_train.columns
            if col != "is_oracle" and pd.api.types.is_numeric_dtype(long_train[col])
        ]
        print(
            f"[fold {fold}] long_train={len(long_train)} long_valid={len(long_valid)} "
            f"long_features={len(long_feature_columns)}",
            flush=True,
        )
        x_long_train, x_long_valid = fit_impute(long_train, long_valid, long_feature_columns)
        y_bin_train = long_train["is_oracle"].to_numpy(np.int8)
        y_bin_valid = long_valid["is_oracle"].to_numpy(np.int8)

        binary = LGBMClassifier(
            objective="binary",
            random_state=seed + 1000 + fold,
            **binary_params,
        )
        binary.fit(
            x_long_train,
            y_bin_train,
            eval_set=[(x_long_valid, y_bin_valid)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(50), log_evaluation(log_period)],
        )
        binary_score = (
            binary.predict_proba(x_long_valid)[:, 1].reshape(len(candidates), len(valid_idx)).T
        )
        oof_selected["lgb_candidate_binary"][valid_idx] = np.argmax(binary_score, axis=1).astype(
            np.int16
        )
        model_path = model_dir / f"{OUTPUT_PREFIX}_lgb_candidate_binary_fold{fold}.txt"
        binary.booster_.save_model(str(model_path))
        model_manifest.append(
            {
                "variant": "lgb_candidate_binary",
                "fold": fold,
                "path": str(model_path.relative_to(output_dir)),
                "sha256": sha256_path(model_path),
                "best_iteration": int(binary.best_iteration_ or binary.n_estimators),
            }
        )
        for feature, importance in zip(
            long_feature_columns, binary.feature_importances_, strict=False
        ):
            importance_rows.append(
                {
                    "variant": "lgb_candidate_binary",
                    "fold": fold,
                    "feature": feature,
                    "importance": float(importance),
                }
            )

        error_ranker = LGBMRegressor(
            objective="regression_l1",
            random_state=seed + 2000 + fold,
            **error_params,
        )
        valid_error = np.abs(
            long_valid["candidate_tvt"].to_numpy(np.float32)
            - np.tile(valid_frame["true_tvt"].to_numpy(np.float32), len(candidates))
        )
        error_ranker.fit(
            x_long_train,
            train_error,
            eval_set=[(x_long_valid, valid_error)],
            eval_metric="l1",
            callbacks=[early_stopping(50), log_evaluation(log_period)],
        )
        pred_error = error_ranker.predict(x_long_valid).reshape(len(candidates), len(valid_idx)).T
        oof_selected["lgb_candidate_error_ranker"][valid_idx] = np.argmin(
            pred_error, axis=1
        ).astype(np.int16)
        model_path = model_dir / f"{OUTPUT_PREFIX}_lgb_candidate_error_ranker_fold{fold}.txt"
        error_ranker.booster_.save_model(str(model_path))
        model_manifest.append(
            {
                "variant": "lgb_candidate_error_ranker",
                "fold": fold,
                "path": str(model_path.relative_to(output_dir)),
                "sha256": sha256_path(model_path),
                "best_iteration": int(error_ranker.best_iteration_ or error_ranker.n_estimators),
            }
        )
        for feature, importance in zip(
            long_feature_columns, error_ranker.feature_importances_, strict=False
        ):
            importance_rows.append(
                {
                    "variant": "lgb_candidate_error_ranker",
                    "fold": fold,
                    "feature": feature,
                    "importance": float(importance),
                }
            )
        del (
            long_train,
            long_valid,
            x_long_train,
            x_long_valid,
            train_error,
            y_bin_train,
            y_bin_valid,
            binary_score,
            valid_error,
            pred_error,
        )
        gc.collect()

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    baseline_indices = {
        "likpf_mean_single": np.full(
            len(frame), candidate_names.index("likpf_mean"), dtype=np.int16
        ),
        "multiobs_score_top1": top1_from_multiobs_scores(frame, candidates),
        "oracle": oracle_labels.astype(np.int16),
    }
    for variant, selected in {**baseline_indices, **oof_selected}.items():
        mode = (
            "oracle"
            if variant == "oracle"
            else ("baseline" if variant in baseline_indices else "oof")
        )
        metrics, pred = evaluate_selection(
            frame=frame,
            selected_indices=selected,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            candidate_names=candidate_names,
            variant=variant,
            mode=mode,
        )
        metric_rows.append(metrics)
        prediction_frames.append(pred)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values("rmse_tvt")
    importance = pd.DataFrame(importance_rows)
    manifest_path = output_dir / f"{OUTPUT_PREFIX}_model_manifest.json"
    with manifest_path.open("w") as fp:
        json.dump(to_jsonable({"models": model_manifest}), fp, indent=2, sort_keys=True)
    model_manifest_meta = [
        {
            **item,
            "manifest": manifest_path.name,
            "manifest_sha256": sha256_path(manifest_path),
        }
        for item in model_manifest
    ]
    return metrics, predictions, importance, model_manifest_meta


def summarize_decision(
    metrics: pd.DataFrame, distribution: pd.DataFrame, by_well: pd.DataFrame
) -> dict[str, Any]:
    best_oof = metrics[metrics["mode"].eq("oof")].sort_values("rmse_tvt").head(1)
    likpf = metrics[metrics["variant"].eq("likpf_mean_single")].head(1)
    multiobs = metrics[metrics["variant"].eq("multiobs_score_top1")].head(1)
    decision = "ranker_not_run"
    delta_likpf = None
    delta_multiobs = None
    pf_rate = None
    if not best_oof.empty:
        best = best_oof.iloc[0]
        if not likpf.empty:
            delta_likpf = float(best["rmse_tvt"] - likpf.iloc[0]["rmse_tvt"])
        if not multiobs.empty:
            delta_multiobs = float(best["rmse_tvt"] - multiobs.iloc[0]["rmse_tvt"])
        dist = distribution[
            (distribution["variant"] == best["variant"])
            & (distribution["selected_candidate"] == "pf_ancc")
        ]
        pf_rate = float(dist["rate"].iloc[0]) if not dist.empty else 0.0
        worst_switch = by_well[by_well["variant"].eq(best["variant"])][
            "path_switch_per_1000_rows"
        ].max()
        if delta_likpf is not None and delta_likpf < -0.25 and pf_rate >= 0.05:
            decision = "ranker_supported_for_followup_continuity_audit"
        elif delta_likpf is not None and delta_likpf < 0.0:
            decision = "weak_ranker_supported_needs_feature_or_continuity_audit"
        else:
            decision = "ranker_not_supported"
        return {
            "recommendation": decision,
            "best_oof_variant": to_jsonable(best.to_dict()),
            "delta_rmse_vs_likpf_mean": delta_likpf,
            "delta_rmse_vs_multiobs_score_top1": delta_multiobs,
            "best_oof_pf_ancc_selection_rate": pf_rate,
            "best_oof_max_path_switch_per_1000_rows": (
                float(worst_switch) if pd.notna(worst_switch) else None
            ),
        }
    return {"recommendation": decision}


def run_typewell_late_range_pfbeam_candidate_prior(
    *,
    output_dir: str | Path,
    cache_path: str | Path | None,
    schema_path: str | Path | None,
    max_rows: int | None,
) -> dict[str, Any]:
    t0 = time.time()
    config = load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_specs_from_config(config)
    required_columns = build_required_columns(config, candidates)
    frame, source_meta = load_train_feature_cache(
        cache_path=cache_path,
        schema_path=schema_path,
        required_columns=required_columns,
        max_rows=max_rows,
    )
    frame, enrichment_columns, enrichment_meta = add_feature_enrichment(
        frame,
        config,
        max_rows=max_rows,
    )
    missing_candidate_columns = [
        spec.column for spec in candidates if spec.column not in frame.columns
    ]
    if missing_candidate_columns:
        raise ValueError(
            f"candidate columns are missing after enrichment: {missing_candidate_columns}"
        )
    frame, late_range_columns, late_range_meta = add_typewell_late_range_prior(
        frame,
        config,
        candidates,
    )
    frame, engineered_columns, candidate_values, oracle_labels = add_candidate_labels_and_features(
        frame,
        candidates,
        include_candidate_values=bool(get_nested(config, "ranker.include_candidate_values")),
    )
    feature_columns = select_numeric_feature_columns(
        frame,
        config,
        [*engineered_columns, *enrichment_columns, *late_range_columns],
    )
    metrics, predictions, importance, model_manifest = train_and_score(
        frame=frame,
        candidates=candidates,
        candidate_values=candidate_values,
        oracle_labels=oracle_labels,
        feature_columns=feature_columns,
        config=config,
        output_dir=output_dir,
    )
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
    )
    decision = summarize_decision(metrics, distribution, by_well)

    metrics_path = output_dir / f"{OUTPUT_PREFIX}_metrics.csv"
    predictions_path = output_dir / f"{OUTPUT_PREFIX}_oof_selected_predictions.csv.gz"
    distribution_path = output_dir / f"{OUTPUT_PREFIX}_selection_distribution.csv"
    by_well_path = output_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    buckets_path = output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    importance_path = output_dir / f"{OUTPUT_PREFIX}_feature_importance.csv"
    mean_importance_path = output_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv"
    schema_out_path = output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    distribution.to_csv(distribution_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    buckets.to_csv(buckets_path, index=False)
    importance.to_csv(importance_path, index=False)
    mean_importance.to_csv(mean_importance_path, index=False)
    pd.DataFrame(
        [{"feature_index": idx, "feature": feature} for idx, feature in enumerate(feature_columns)]
    ).to_csv(schema_out_path, index=False)

    best = metrics.iloc[0].to_dict() if not metrics.empty else {}
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
        "candidates": [spec.name for spec in candidates],
        "source": source_meta,
        "feature_enrichment": to_jsonable(enrichment_meta),
        "typewell_late_range_prior": to_jsonable(late_range_meta),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "best_metric": to_jsonable(best),
        "decision": to_jsonable(decision),
        "sha256": {
            "metrics": sha256_path(metrics_path),
            "predictions": sha256_path(predictions_path),
            "predictions_decompressed": sha256_path(predictions_path, decompressed=True),
            "feature_schema": sha256_path(schema_out_path),
            "prediction_by_variant": prediction_hashes,
        },
        "model_manifest": model_manifest,
        "artifacts": {
            "metrics": metrics_path.name,
            "oof_selected_predictions": predictions_path.name,
            "selection_distribution": distribution_path.name,
            "by_well": by_well_path.name,
            "bucket_metrics": buckets_path.name,
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
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--schema-path", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args(argv)
    paths = ExperimentPaths()
    config = load_config()
    output_dir = args.output_dir or (
        paths.artifacts_dir
        if not (Path("/kaggle/working").exists())
        else Path("/kaggle/working") / "artifacts"
    )
    cache_path = args.cache_path or get_nested(config, "data.exp099_train_feature_cache_local")
    schema_path = args.schema_path or get_nested(config, "data.exp099_train_feature_schema_local")
    max_rows = args.max_rows
    configured_max = get_nested(config, "ranker.max_rows")
    if max_rows is None and configured_max is not None:
        max_rows = int(configured_max)
    return run_typewell_late_range_pfbeam_candidate_prior(
        output_dir=output_dir,
        cache_path=cache_path,
        schema_path=schema_path,
        max_rows=max_rows,
    )


run_candidate_ranker_feature_enrichment = run_typewell_late_range_pfbeam_candidate_prior
run_pf_candidate_ranker_or_nway_classifier = run_candidate_ranker_feature_enrichment


if __name__ == "__main__":
    main()

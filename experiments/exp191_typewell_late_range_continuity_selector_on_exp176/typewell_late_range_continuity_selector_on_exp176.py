from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config
from sklearn.model_selection import GroupKFold

OUTPUT_PREFIX = "exp191_typewell_late_range_continuity_selector_on_exp176"
EXP099_FEATURE_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
EXP099_FEATURE_SCHEMA = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv"
)
EXP072_FEATURE_CACHE = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
EXP072_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
EXP176_MANIFEST = "exp176_typewell_late_range_pfbeam_candidate_prior_model_manifest.json"
EXP176_FEATURE_SCHEMA = "exp176_typewell_late_range_pfbeam_candidate_prior_feature_schema.csv"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    column: str


@dataclass(frozen=True)
class ViterbiSpec:
    variant: str
    switch_penalty: float
    nondefault_bias: float
    jump_penalty_weight: float
    jump_free_ft: float
    jump_scale_ft: float
    max_abs_delta_vs_default: float
    max_pf_ancc_std: float
    min_md_since: float
    min_segment_len: int


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
    *,
    explicit_path: str | Path | None = None,
    explicit_dir: str | Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    if explicit_dir is not None:
        candidates.append(Path(explicit_dir) / filename)
    candidates.extend(
        [
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
            Path("experiments")
            / "exp176_typewell_late_range_pfbeam_candidate_prior"
            / "kaggle"
            / "output"
            / "train_v3"
            / "artifacts"
            / filename,
            Path("experiments")
            / "exp157_candidate_ranker_feature_enrichment"
            / "kaggle"
            / "output"
            / "train_v1"
            / "artifacts"
            / filename,
            Path("experiments")
            / "exp099_pf_multi_observation_likelihood_probe"
            / "kaggle"
            / "output"
            / "train_v2"
            / "artifacts"
            / filename,
        ]
    )
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:100])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def resolve_model_path(manifest_path: Path, relative_path: str) -> Path:
    candidates = [
        manifest_path.parent / relative_path,
        manifest_path.parent / Path(relative_path).name,
        Path("experiments")
        / "exp176_typewell_late_range_pfbeam_candidate_prior"
        / "kaggle"
        / "output"
        / "train_v3"
        / "artifacts"
        / relative_path,
        Path("experiments")
        / "exp176_typewell_late_range_pfbeam_candidate_prior"
        / "kaggle"
        / "output"
        / "train_v3"
        / "artifacts"
        / Path(relative_path).name,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{Path(relative_path).name}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(f"model file not found for {relative_path}")


def candidate_specs_from_config(config: dict[str, Any]) -> list[CandidateSpec]:
    values = get_nested(config, "selector.candidates") or []
    specs: list[CandidateSpec] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("selector.candidates entries must be mappings")
        specs.append(
            CandidateSpec(name=str(item["name"]), column=str(item.get("column", item["name"])))
        )
    if not specs:
        raise ValueError("selector.candidates must not be empty")
    return specs


def configured_raw_columns(config: dict[str, Any], candidates: list[CandidateSpec]) -> list[str]:
    required = {"id", "well", "target", "last_known_tvt"}
    auxiliary_columns = set(
        get_nested(config, "selector.feature_enrichment.auxiliary_candidate_columns") or []
    )
    required.update(spec.column for spec in candidates if spec.column not in auxiliary_columns)
    for key in [
        "selector.context_columns",
        "selector.multiobs_feature_columns",
        "selector.optional_columns",
    ]:
        required.update(
            str(value)
            for value in (get_nested(config, key) or [])
            if str(value) not in auxiliary_columns
        )
    return sorted(required)


def load_feature_cache(
    *,
    config: dict[str, Any],
    required_columns: list[str],
    max_rows: int | None,
    cache_path: str | Path | None,
    schema_path: str | Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        EXP099_FEATURE_CACHE,
        explicit_path=cache_path or get_nested(config, "data.exp099_train_feature_cache_local"),
    )
    schema = find_artifact(
        EXP099_FEATURE_SCHEMA,
        explicit_path=schema_path or get_nested(config, "data.exp099_train_feature_schema_local"),
    )
    header = pd.read_csv(source, nrows=0).columns.tolist()
    load_columns = [column for column in required_columns if column in header]
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required raw columns: {missing}")
    frame = pd.read_csv(
        source,
        usecols=load_columns,
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
    config: dict[str, Any],
    required_columns: list[str],
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        EXP072_FEATURE_CACHE,
        explicit_path=get_nested(config, "data.exp072_train_feature_cache_local"),
    )
    schema = find_artifact(
        EXP072_FEATURE_SCHEMA,
        explicit_path=get_nested(config, "data.exp072_feature_schema_local"),
    )
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
    enrichment = get_nested(config, "selector.feature_enrichment") or {}
    if not enrichment.get("enabled", False):
        return frame, [], {"enabled": False}

    auxiliary_columns = ["id", *[str(value) for value in enrichment.get("auxiliary_columns", [])]]
    auxiliary, source_meta = load_auxiliary_feature_cache(
        config=config,
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
    prior = get_nested(config, "selector.typewell_late_range_prior") or {}
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
    frame: pd.DataFrame, candidates: list[CandidateSpec]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
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
    oracle_labels = np.argmin(errors, axis=1).astype(np.int16)
    out["oracle_label"] = oracle_labels
    out["oracle_candidate"] = np.asarray([candidates[i].name for i in oracle_labels], dtype=object)

    for spec in candidates:
        out[f"{spec.name}_minus_last"] = out[spec.column].astype(np.float32) - out[
            "last_known_tvt"
        ].astype(np.float32)
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            out[f"{left.name}_vs_{right.name}_abs"] = np.abs(
                out[left.column].astype(np.float32) - out[right.column].astype(np.float32)
            )
    value_cols = [spec.column for spec in candidates]
    out["candidate_mean"] = out[value_cols].mean(axis=1).astype(np.float32)
    out["candidate_std"] = out[value_cols].std(axis=1).astype(np.float32)
    out["candidate_range"] = (out[value_cols].max(axis=1) - out[value_cols].min(axis=1)).astype(
        np.float32
    )
    return out, candidate_values, oracle_labels


def load_exp176_feature_columns(config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    schema_path = find_artifact(
        str(get_nested(config, "data.exp176_feature_schema") or EXP176_FEATURE_SCHEMA),
        explicit_dir=get_nested(config, "data.exp176_artifact_dir_local"),
    )
    schema = pd.read_csv(schema_path)
    if "feature" not in schema.columns:
        raise ValueError(f"{schema_path} must contain a feature column")
    feature_columns = schema["feature"].astype(str).tolist()
    if not feature_columns:
        raise ValueError(f"{schema_path} contains no features")
    return feature_columns, {"path": str(schema_path), "sha256": sha256_path(schema_path)}


def select_long_row_feature_columns(
    config: dict[str, Any], feature_columns: list[str]
) -> list[str]:
    long_config = get_nested(config, "selector.long_models") or {}
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


def fit_impute_values(train: pd.DataFrame, valid: pd.DataFrame, columns: list[str]) -> np.ndarray:
    train_values = train[columns].to_numpy(dtype=np.float32, copy=True)
    valid_values = valid[columns].to_numpy(dtype=np.float32, copy=True)
    train_values[~np.isfinite(train_values)] = np.nan
    valid_values[~np.isfinite(valid_values)] = np.nan
    medians = np.nanmedian(train_values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    bad = ~np.isfinite(valid_values)
    if bad.any():
        valid_values[bad] = np.take(medians, np.where(bad)[1])
    return valid_values


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
) -> pd.DataFrame:
    if sample_rows is not None and len(row_indices) > sample_rows:
        rng = np.random.default_rng(seed)
        row_indices = np.sort(rng.choice(row_indices, size=int(sample_rows), replace=False))
    chunks: list[pd.DataFrame] = []
    last_known = frame["last_known_tvt"].to_numpy(np.float32)
    source_rows = frame.iloc[row_indices]
    for cand_idx, spec in enumerate(candidates):
        part = source_rows[row_feature_columns].copy()
        part["candidate_index"] = np.int16(cand_idx)
        part["candidate_name_code"] = np.int16(cand_idx)
        part["candidate_tvt"] = candidate_values[row_indices, cand_idx].astype(np.float32)
        part["candidate_minus_last"] = (
            candidate_values[row_indices, cand_idx] - last_known[row_indices]
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
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def long_feature_columns(long_frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in long_frame.columns
        if col not in {"id", "well", "is_oracle"} and pd.api.types.is_numeric_dtype(long_frame[col])
    ]


def fit_impute_long_values(
    train_long: pd.DataFrame, valid_long: pd.DataFrame, columns: list[str]
) -> np.ndarray:
    train_values = train_long[columns].to_numpy(dtype=np.float32, copy=True)
    valid_values = valid_long[columns].to_numpy(dtype=np.float32, copy=True)
    train_values[~np.isfinite(train_values)] = np.nan
    valid_values[~np.isfinite(valid_values)] = np.nan
    medians = np.nanmedian(train_values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    bad = ~np.isfinite(valid_values)
    if bad.any():
        valid_values[bad] = np.take(medians, np.where(bad)[1])
    return valid_values


def load_manifest(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    manifest_path = find_artifact(
        str(get_nested(config, "data.exp176_model_manifest") or EXP176_MANIFEST),
        explicit_dir=get_nested(config, "data.exp176_artifact_dir_local"),
    )
    with manifest_path.open() as fp:
        manifest = json.load(fp)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("models"), list):
        raise ValueError(f"invalid exp176 model manifest: {manifest_path}")
    return manifest_path, manifest


def model_item(manifest: dict[str, Any], variant: str, fold: int) -> dict[str, Any]:
    for item in manifest["models"]:
        if item.get("variant") == variant and int(item.get("fold")) == int(fold):
            return item
    raise KeyError(f"model not found: variant={variant} fold={fold}")


def reconstruct_exp176_scores(
    *,
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    candidate_values: np.ndarray,
    oracle_labels: np.ndarray,
    feature_columns: list[str],
    config: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    from lightgbm import Booster

    seed = int(get_nested(config, "validation.seed") or 42)
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    max_train_rows = get_nested(config, "selector.max_train_rows_per_fold")
    max_train_rows = int(max_train_rows) if max_train_rows is not None else None
    late_range_prior = get_nested(config, "selector.typewell_late_range_prior") or {}
    row_feature_columns = select_long_row_feature_columns(config, feature_columns)
    n_rows = len(frame)
    n_candidates = len(candidates)
    scores = {
        "multiclass_proba": np.zeros((n_rows, n_candidates), dtype=np.float32),
        "binary_proba": np.zeros((n_rows, n_candidates), dtype=np.float32),
        "predicted_error": np.zeros((n_rows, n_candidates), dtype=np.float32),
    }
    manifest_rows: list[dict[str, Any]] = []

    cv = GroupKFold(n_splits=n_folds)
    folds = list(cv.split(frame, oracle_labels, groups=frame["well"]))
    for fold, (train_idx, valid_idx) in enumerate(folds):
        print(f"[fold {fold}] reconstruct scores train={len(train_idx)} valid={len(valid_idx)}")
        train_frame = frame.iloc[train_idx]
        valid_frame = frame.iloc[valid_idx]
        x_valid = fit_impute_values(train_frame, valid_frame, feature_columns)

        multiclass_item = model_item(manifest, "lgb_multiclass", fold)
        multiclass_path = resolve_model_path(manifest_path, str(multiclass_item["path"]))
        multiclass = Booster(model_file=str(multiclass_path))
        multiclass_pred = multiclass.predict(x_valid)
        scores["multiclass_proba"][valid_idx] = np.asarray(multiclass_pred, dtype=np.float32)
        manifest_rows.append(
            {
                **multiclass_item,
                "resolved_path": str(multiclass_path),
                "resolved_sha256": sha256_path(multiclass_path),
            }
        )

        train_long = build_long_frame(
            frame,
            train_idx,
            candidates,
            row_feature_columns=row_feature_columns,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            late_range_prior=late_range_prior,
            sample_rows=max_train_rows,
            seed=seed + 101 * fold,
        )
        valid_long = build_long_frame(
            frame,
            valid_idx,
            candidates,
            row_feature_columns=row_feature_columns,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            late_range_prior=late_range_prior,
            sample_rows=None,
            seed=seed,
        )
        long_columns = long_feature_columns(train_long)
        x_long_valid = fit_impute_long_values(train_long, valid_long, long_columns)

        binary_item = model_item(manifest, "lgb_candidate_binary", fold)
        binary_path = resolve_model_path(manifest_path, str(binary_item["path"]))
        binary = Booster(model_file=str(binary_path))
        binary_pred = binary.predict(x_long_valid).reshape(n_candidates, len(valid_idx)).T
        scores["binary_proba"][valid_idx] = np.asarray(binary_pred, dtype=np.float32)
        manifest_rows.append(
            {
                **binary_item,
                "resolved_path": str(binary_path),
                "resolved_sha256": sha256_path(binary_path),
            }
        )

        error_item = model_item(manifest, "lgb_candidate_error_ranker", fold)
        error_path = resolve_model_path(manifest_path, str(error_item["path"]))
        error_model = Booster(model_file=str(error_path))
        error_pred = error_model.predict(x_long_valid).reshape(n_candidates, len(valid_idx)).T
        scores["predicted_error"][valid_idx] = np.asarray(error_pred, dtype=np.float32)
        manifest_rows.append(
            {
                **error_item,
                "resolved_path": str(error_path),
                "resolved_sha256": sha256_path(error_path),
            }
        )

    return scores, pd.DataFrame(manifest_rows)


def second_margin_high(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values, axis=1)
    top = order[:, -1]
    top1 = values[np.arange(len(values)), top]
    top2 = values[np.arange(len(values)), order[:, -2]]
    return top.astype(np.int16), (top1 - top2).astype(np.float32)


def second_margin_low(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(values, axis=1)
    top = order[:, 0]
    top1 = values[np.arange(len(values)), top]
    top2 = values[np.arange(len(values)), order[:, 1]]
    return top.astype(np.int16), (top2 - top1).astype(np.float32), top1.astype(np.float32)


def _row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int32)


def categorical_codes(values: pd.Series | pd.Categorical) -> np.ndarray:
    if isinstance(values, pd.Series):
        return values.cat.codes.to_numpy(np.int16)
    return values.codes.astype(np.int16)


def distance_bucket_codes(values: pd.Series | np.ndarray) -> tuple[np.ndarray, list[str]]:
    labels = ["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"]
    cats = pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=labels,
        include_lowest=True,
    )
    return categorical_codes(cats), labels


def tail_rank_bucket_codes(ids: pd.Series) -> tuple[np.ndarray, list[str]]:
    labels = ["000_099", "100_249", "250_499", "500_999", "1000_plus"]
    cats = pd.cut(
        _row_indices_from_ids(ids),
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=labels,
        include_lowest=True,
    )
    return categorical_codes(cats), labels


def quantile_bucket_codes(
    values: pd.Series | np.ndarray, prefix: str
) -> tuple[np.ndarray, list[str]]:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series[np.isfinite(series)]
    if finite.nunique(dropna=True) < 4:
        return np.zeros(len(series), dtype=np.int16), [f"{prefix}_unknown"]
    edges = np.unique(np.nanquantile(finite, [0.0, 0.25, 0.50, 0.75, 1.0]))
    if len(edges) < 3:
        return np.zeros(len(series), dtype=np.int16), [f"{prefix}_unknown"]
    labels = [f"{prefix}_q{i + 1}" for i in range(len(edges) - 1)]
    cats = pd.cut(series, bins=edges, labels=labels, include_lowest=True)
    return categorical_codes(cats), labels


def variant_specs_from_config(config: dict[str, Any]) -> list[ViterbiSpec]:
    values = get_nested(config, "selector.viterbi_grid") or {}
    switch_penalties = [float(v) for v in values.get("switch_penalty", [0.0, 2.5, 5.0])]
    nondefault_biases = [float(v) for v in values.get("nondefault_bias", [0.0])]
    jump_weights = [float(v) for v in values.get("jump_penalty_weight", [0.0, 0.5])]
    jump_free_values = [float(v) for v in values.get("jump_free_ft", [25.0])]
    jump_scale = float(values.get("jump_scale_ft", 25.0))
    delta_caps = [float(v) for v in values.get("max_abs_delta_vs_default", [35.0])]
    std_caps = [float(v) for v in values.get("max_pf_ancc_std", [999999.0])]
    min_md_values = [float(v) for v in values.get("min_md_since", [0.0])]
    min_segment_values = [int(v) for v in values.get("min_segment_len", [1, 12])]

    specs: list[ViterbiSpec] = []
    for switch_penalty in switch_penalties:
        for nondefault_bias in nondefault_biases:
            for jump_weight in jump_weights:
                for jump_free in jump_free_values:
                    for delta_cap in delta_caps:
                        for std_cap in std_caps:
                            for min_md_since in min_md_values:
                                for min_segment_len in min_segment_values:
                                    variant = (
                                        "viterbi"
                                        f"_sw{int(round(switch_penalty * 10)):03d}"
                                        f"_bias{int(round(nondefault_bias * 10)):03d}"
                                        f"_jw{int(round(jump_weight * 100)):03d}"
                                        f"_jf{int(round(jump_free)):03d}"
                                        f"_d{int(round(min(delta_cap, 999999.0))):03d}"
                                        f"_std{int(round(min(std_cap, 999999.0))):06d}"
                                        f"_md{int(round(min_md_since)):04d}"
                                        f"_seg{int(min_segment_len):03d}"
                                    )
                                    specs.append(
                                        ViterbiSpec(
                                            variant=variant,
                                            switch_penalty=switch_penalty,
                                            nondefault_bias=nondefault_bias,
                                            jump_penalty_weight=jump_weight,
                                            jump_free_ft=jump_free,
                                            jump_scale_ft=jump_scale,
                                            max_abs_delta_vs_default=delta_cap,
                                            max_pf_ancc_std=std_cap,
                                            min_md_since=min_md_since,
                                            min_segment_len=min_segment_len,
                                        )
                                    )
    return specs


def build_local_cost(
    *,
    predicted_error: np.ndarray,
    candidate_values: np.ndarray,
    candidate_names: list[str],
    frame: pd.DataFrame,
    default_idx: int,
    allowed_switch_idx: np.ndarray,
    spec: ViterbiSpec,
) -> np.ndarray:
    cost = np.asarray(predicted_error, dtype=np.float32).copy()
    cost[~np.isfinite(cost)] = 1.0e6
    cost = np.maximum(cost, 0.0)

    n_rows, n_candidates = cost.shape
    allowed = np.zeros(n_candidates, dtype=bool)
    allowed[default_idx] = True
    allowed[allowed_switch_idx] = True
    cost[:, ~allowed] = 1.0e6

    nondefault = np.arange(n_candidates) != default_idx
    if spec.nondefault_bias:
        cost[:, nondefault] += np.float32(spec.nondefault_bias)

    default_values = candidate_values[:, default_idx]
    delta_vs_default = np.abs(candidate_values - default_values[:, None])
    too_far = delta_vs_default > np.float32(spec.max_abs_delta_vs_default)
    too_far[:, default_idx] = False
    cost[too_far] = 1.0e6

    pf_std = (
        pd.to_numeric(frame["pf_ancc_std"], errors="coerce").fillna(np.inf).to_numpy(np.float32)
        if "pf_ancc_std" in frame.columns
        else np.full(n_rows, np.inf, dtype=np.float32)
    )
    md_since = (
        pd.to_numeric(frame["md_since"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        if "md_since" in frame.columns
        else np.zeros(n_rows, dtype=np.float32)
    )
    nondefault_row_block = (pf_std > spec.max_pf_ancc_std) | (md_since < spec.min_md_since)
    cost[np.ix_(nondefault_row_block, nondefault)] = 1.0e6
    return cost


def run_viterbi_for_well(
    local_cost: np.ndarray,
    candidate_values: np.ndarray,
    *,
    switch_penalty: float,
    jump_penalty_weight: float,
    jump_free_ft: float,
    jump_scale_ft: float,
) -> np.ndarray:
    n_rows, n_candidates = local_cost.shape
    if n_rows == 0:
        return np.empty(0, dtype=np.int16)
    dp = np.empty((n_rows, n_candidates), dtype=np.float64)
    back = np.zeros((n_rows, n_candidates), dtype=np.int16)
    dp[0] = local_cost[0].astype(np.float64)
    candidate_index = np.arange(n_candidates)
    switch_matrix = (candidate_index[:, None] != candidate_index[None, :]).astype(np.float64)
    switch_matrix *= float(switch_penalty)

    for row in range(1, n_rows):
        prev_values = candidate_values[row - 1]
        curr_values = candidate_values[row]
        jump = np.abs(curr_values[None, :] - prev_values[:, None])
        jump_cost = np.maximum(jump - float(jump_free_ft), 0.0) / max(float(jump_scale_ft), 1e-6)
        transition = dp[row - 1][:, None] + switch_matrix + float(jump_penalty_weight) * jump_cost
        back[row] = np.argmin(transition, axis=0).astype(np.int16)
        dp[row] = local_cost[row].astype(np.float64) + transition[back[row], candidate_index]

    selected = np.empty(n_rows, dtype=np.int16)
    selected[-1] = int(np.argmin(dp[-1]))
    for row in range(n_rows - 1, 0, -1):
        selected[row - 1] = back[row, selected[row]]
    return selected


def prune_short_switch_segments(
    selected_idx: np.ndarray,
    *,
    frame: pd.DataFrame,
    default_idx: int,
    min_segment_len: int,
) -> np.ndarray:
    if min_segment_len <= 1:
        return selected_idx
    out = selected_idx.copy()
    row_indices = _row_indices_from_ids(frame["id"])
    well_codes, _well_names = pd.factorize(frame["well"], sort=True)
    order = np.lexsort((row_indices, well_codes.astype(np.int32)))
    ordered_selected = out[order]
    ordered_well = well_codes[order]
    start = 0
    n_rows = len(order)
    while start < n_rows:
        end = start + 1
        while (
            end < n_rows
            and ordered_well[end] == ordered_well[start]
            and ordered_selected[end] == ordered_selected[start]
        ):
            end += 1
        candidate_idx = int(ordered_selected[start])
        if candidate_idx != default_idx and end - start < int(min_segment_len):
            out[order[start:end]] = default_idx
        start = end
    return out


def viterbi_select(
    *,
    frame: pd.DataFrame,
    predicted_error: np.ndarray,
    candidate_values: np.ndarray,
    candidate_names: list[str],
    default_idx: int,
    allowed_switch_idx: np.ndarray,
    spec: ViterbiSpec,
) -> np.ndarray:
    local_cost = build_local_cost(
        predicted_error=predicted_error,
        candidate_values=candidate_values,
        candidate_names=candidate_names,
        frame=frame,
        default_idx=default_idx,
        allowed_switch_idx=allowed_switch_idx,
        spec=spec,
    )
    selected = np.full(len(frame), default_idx, dtype=np.int16)
    row_indices = _row_indices_from_ids(frame["id"])
    well_codes, _well_names = pd.factorize(frame["well"], sort=True)
    order = np.lexsort((row_indices, well_codes.astype(np.int32)))
    ordered_well = well_codes[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and ordered_well[end] == ordered_well[start]:
            end += 1
        positions = order[start:end]
        selected[positions] = run_viterbi_for_well(
            local_cost[positions],
            candidate_values[positions],
            switch_penalty=spec.switch_penalty,
            jump_penalty_weight=spec.jump_penalty_weight,
            jump_free_ft=spec.jump_free_ft,
            jump_scale_ft=spec.jump_scale_ft,
        )
        start = end
    return prune_short_switch_segments(
        selected,
        frame=frame,
        default_idx=default_idx,
        min_segment_len=spec.min_segment_len,
    )


def metrics_for_selection(
    *,
    variant: str,
    mode: str,
    selected_idx: np.ndarray,
    candidate_values: np.ndarray,
    true_tvt: np.ndarray,
    oracle_labels: np.ndarray,
    default_idx: int,
) -> dict[str, Any]:
    pred = candidate_values[np.arange(len(selected_idx)), selected_idx].astype(np.float32)
    err = pred.astype(np.float64) - true_tvt.astype(np.float64)
    abs_err = np.abs(err)
    switched = selected_idx != default_idx
    metrics: dict[str, Any] = {
        "variant": variant,
        "mode": mode,
        "rows": int(len(selected_idx)),
        "rmse_tvt": float(np.sqrt(np.mean(np.square(err)))),
        "mae_tvt": float(np.mean(abs_err)),
        "oracle_label_accuracy": float(np.mean(selected_idx == oracle_labels)),
        "switch_rows": int(np.sum(switched)),
        "switch_rate": float(np.mean(switched)),
    }
    for threshold in [1.0, 2.0, 5.0, 10.0]:
        metrics[f"within_{int(threshold)}ft"] = float(np.mean(abs_err <= threshold))
    return metrics


def selection_distribution_rows(
    *, variant: str, mode: str, selected_idx: np.ndarray, candidate_names: list[str]
) -> list[dict[str, Any]]:
    rows = []
    total = len(selected_idx)
    counts = np.bincount(selected_idx.astype(np.int16), minlength=len(candidate_names))
    for idx, count in enumerate(counts):
        if count:
            rows.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "selected_candidate": candidate_names[idx],
                    "rows": int(count),
                    "rate": float(count / total) if total else 0.0,
                }
            )
    return rows


def by_well_rows(
    *,
    variant: str,
    mode: str,
    selected_idx: np.ndarray,
    selected_tvt: np.ndarray,
    true_tvt: np.ndarray,
    well_codes: np.ndarray,
    well_names: np.ndarray,
    order: np.ndarray,
) -> list[dict[str, Any]]:
    n_wells = len(well_names)
    err = selected_tvt.astype(np.float64) - true_tvt.astype(np.float64)
    abs_err = np.abs(err)
    counts = np.bincount(well_codes, minlength=n_wells)
    se = np.bincount(well_codes, weights=np.square(err), minlength=n_wells)
    ae = np.bincount(well_codes, weights=abs_err, minlength=n_wells)
    within10 = np.bincount(well_codes, weights=(abs_err <= 10.0).astype(float), minlength=n_wells)
    ordered_sel = selected_idx[order]
    ordered_well = well_codes[order]
    same_well = ordered_well[1:] == ordered_well[:-1]
    switch_mask = same_well & (ordered_sel[1:] != ordered_sel[:-1])
    switch_wells = ordered_well[1:][switch_mask]
    switches = np.bincount(switch_wells, minlength=n_wells)
    rows = []
    for code, well in enumerate(well_names):
        if counts[code] == 0:
            continue
        rows.append(
            {
                "variant": variant,
                "mode": mode,
                "well": str(well),
                "rows": int(counts[code]),
                "rmse_tvt": float(np.sqrt(se[code] / counts[code])),
                "mae_tvt": float(ae[code] / counts[code]),
                "within_10ft": float(within10[code] / counts[code]),
                "path_switch_count": int(switches[code]),
                "path_switch_per_1000_rows": float(switches[code] / counts[code] * 1000.0),
            }
        )
    return rows


def bucket_metric_rows(
    *,
    variant: str,
    mode: str,
    selected_tvt: np.ndarray,
    true_tvt: np.ndarray,
    bucket_defs: list[tuple[str, np.ndarray, list[str]]],
) -> list[dict[str, Any]]:
    err = selected_tvt.astype(np.float64) - true_tvt.astype(np.float64)
    abs_err = np.abs(err)
    rows: list[dict[str, Any]] = []
    for family, codes, labels in bucket_defs:
        valid = codes >= 0
        for code, label in enumerate(labels):
            mask = valid & (codes == code)
            count = int(np.sum(mask))
            if not count:
                continue
            rows.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "bucket_family": family,
                    "bucket": label,
                    "rows": count,
                    "rmse_tvt": float(np.sqrt(np.mean(np.square(err[mask])))),
                    "mae_tvt": float(np.mean(abs_err[mask])),
                    "within_10ft": float(np.mean(abs_err[mask] <= 10.0)),
                }
            )
    return rows


def selected_prediction_frame(
    *,
    frame: pd.DataFrame,
    variant: str,
    mode: str,
    selected_idx: np.ndarray,
    candidate_values: np.ndarray,
    true_tvt: np.ndarray,
    oracle_labels: np.ndarray,
    candidate_names: list[str],
) -> pd.DataFrame:
    selected_tvt = candidate_values[np.arange(len(selected_idx)), selected_idx].astype(np.float32)
    return pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "well": frame["well"].to_numpy(),
            "variant": variant,
            "mode": mode,
            "selected_candidate": np.asarray([candidate_names[i] for i in selected_idx]),
            "selected_candidate_index": selected_idx.astype(np.int16),
            "selected_tvt": selected_tvt,
            "true_tvt": true_tvt.astype(np.float32),
            "abs_error": np.abs(selected_tvt - true_tvt).astype(np.float32),
            "oracle_candidate": np.asarray([candidate_names[i] for i in oracle_labels]),
            "oracle_label": oracle_labels.astype(np.int16),
        }
    )


def score_summary_rows(
    *,
    scores: dict[str, np.ndarray],
    candidate_names: list[str],
    error_selected: np.ndarray,
    error_margin: np.ndarray,
    binary_margin: np.ndarray,
    multiclass_margin: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, values in [
        ("error_margin", error_margin),
        ("binary_margin", binary_margin),
        ("multiclass_margin", multiclass_margin),
    ]:
        rows.append(
            {
                "score": name,
                "min": float(np.nanmin(values)),
                "p25": float(np.nanquantile(values, 0.25)),
                "median": float(np.nanmedian(values)),
                "p75": float(np.nanquantile(values, 0.75)),
                "max": float(np.nanmax(values)),
            }
        )
    row_idx = np.arange(len(error_selected))
    selected_error = scores["predicted_error"][row_idx, error_selected]
    rows.append(
        {
            "score": "predicted_error_selected",
            "min": float(np.nanmin(selected_error)),
            "p25": float(np.nanquantile(selected_error, 0.25)),
            "median": float(np.nanmedian(selected_error)),
            "p75": float(np.nanquantile(selected_error, 0.75)),
            "max": float(np.nanmax(selected_error)),
        }
    )
    for cand_idx, candidate in enumerate(candidate_names):
        values = scores["predicted_error"][:, cand_idx]
        rows.append(
            {
                "score": f"predicted_error_{candidate}",
                "min": float(np.nanmin(values)),
                "p25": float(np.nanquantile(values, 0.25)),
                "median": float(np.nanmedian(values)),
                "p75": float(np.nanquantile(values, 0.75)),
                "max": float(np.nanmax(values)),
            }
        )
    return rows


def evaluate_selection(
    *,
    variant: str,
    mode: str,
    selected_idx: np.ndarray,
    params: dict[str, Any],
    candidate_values: np.ndarray,
    true_tvt: np.ndarray,
    oracle_labels: np.ndarray,
    default_idx: int,
    candidate_names: list[str],
    well_codes: np.ndarray,
    well_names: np.ndarray,
    order: np.ndarray,
    bucket_defs: list[tuple[str, np.ndarray, list[str]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_tvt = candidate_values[np.arange(len(selected_idx)), selected_idx].astype(np.float32)
    metric = {
        **metrics_for_selection(
            variant=variant,
            mode=mode,
            selected_idx=selected_idx,
            candidate_values=candidate_values,
            true_tvt=true_tvt,
            oracle_labels=oracle_labels,
            default_idx=default_idx,
        ),
        **{f"param_{key}": value for key, value in params.items()},
    }
    distribution = selection_distribution_rows(
        variant=variant,
        mode=mode,
        selected_idx=selected_idx,
        candidate_names=candidate_names,
    )
    by_well = by_well_rows(
        variant=variant,
        mode=mode,
        selected_idx=selected_idx,
        selected_tvt=selected_tvt,
        true_tvt=true_tvt,
        well_codes=well_codes,
        well_names=well_names,
        order=order,
    )
    buckets = bucket_metric_rows(
        variant=variant,
        mode=mode,
        selected_tvt=selected_tvt,
        true_tvt=true_tvt,
        bucket_defs=bucket_defs,
    )
    return metric, distribution, by_well, buckets


def run_segment_viterbi_candidate_selector(
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
    candidate_names = [spec.name for spec in candidates]
    default_idx = candidate_names.index(
        str(get_nested(config, "selector.default_candidate") or "likpf_mean")
    )
    allowed_names = [
        str(value) for value in (get_nested(config, "selector.allowed_switch_candidates") or [])
    ]
    allowed_switch_idx = np.asarray(
        [candidate_names.index(name) for name in allowed_names],
        dtype=np.int16,
    )

    raw_columns = configured_raw_columns(config, candidates)
    frame, source_meta = load_feature_cache(
        config=config,
        required_columns=raw_columns,
        max_rows=max_rows,
        cache_path=cache_path,
        schema_path=schema_path,
    )
    frame, _enrichment_columns, enrichment_meta = add_feature_enrichment(
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
    frame, candidate_values, oracle_labels = add_candidate_labels_and_features(frame, candidates)
    feature_columns, exp176_schema_meta = load_exp176_feature_columns(config)
    missing_features = [column for column in feature_columns if column not in frame.columns]
    if missing_features:
        raise ValueError(
            f"exp176 feature schema columns missing after engineering: {missing_features}"
        )

    manifest_path, manifest = load_manifest(config)
    scores, model_manifest = reconstruct_exp176_scores(
        frame=frame,
        candidates=candidates,
        candidate_values=candidate_values,
        oracle_labels=oracle_labels,
        feature_columns=feature_columns,
        config=config,
        manifest_path=manifest_path,
        manifest=manifest,
    )

    _binary_selected, binary_margin = second_margin_high(scores["binary_proba"])
    _multiclass_selected, multiclass_margin = second_margin_high(scores["multiclass_proba"])
    error_selected, error_margin, _error_top1 = second_margin_low(scores["predicted_error"])

    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    row_indices = _row_indices_from_ids(frame["id"])
    well_codes, well_names = pd.factorize(frame["well"], sort=True)
    well_codes = well_codes.astype(np.int32)
    order = np.lexsort((row_indices, well_codes))
    bucket_defs: list[tuple[str, np.ndarray, list[str]]] = []
    distance_values = frame["md_since"] if "md_since" in frame.columns else np.nan
    codes, labels = distance_bucket_codes(distance_values)
    bucket_defs.append(("distance_bucket", codes, labels))
    codes, labels = tail_rank_bucket_codes(frame["id"])
    bucket_defs.append(("tail_rank_bucket", codes, labels))
    for source_column, bucket_name in [
        ("eval_len", "eval_len_bucket"),
        ("pf_ancc_std", "pf_seed_std_bucket"),
        ("likpf_mean_d", "likpf_delta_bucket"),
    ]:
        if source_column in frame.columns:
            codes, labels = quantile_bucket_codes(frame[source_column], bucket_name)
            bucket_defs.append((bucket_name, codes, labels))

    metric_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    by_well_all: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    params_rows: list[dict[str, Any]] = []

    baseline_selections: dict[str, tuple[str, np.ndarray, dict[str, Any]]] = {
        "likpf_mean_single": (
            "baseline",
            np.full(len(frame), default_idx, dtype=np.int16),
            {"source": "fixed_default"},
        ),
        "exp176_error_ranker_rowwise": (
            "oof",
            error_selected.astype(np.int16),
            {"source": "exp176_lgb_candidate_error_ranker_argmin"},
        ),
        "oracle": (
            "oracle",
            oracle_labels.astype(np.int16),
            {"source": "oracle_best_candidate"},
        ),
    }

    for variant, (mode, selected, params) in baseline_selections.items():
        metric, distribution, by_well, buckets = evaluate_selection(
            variant=variant,
            mode=mode,
            selected_idx=selected,
            params=params,
            candidate_values=candidate_values,
            true_tvt=true_tvt,
            oracle_labels=oracle_labels,
            default_idx=default_idx,
            candidate_names=candidate_names,
            well_codes=well_codes,
            well_names=well_names,
            order=order,
            bucket_defs=bucket_defs,
        )
        metric_rows.append(metric)
        distribution_rows.extend(distribution)
        by_well_all.extend(by_well)
        bucket_rows.extend(buckets)
        params_rows.append({"variant": variant, "mode": mode, **params})

    specs = variant_specs_from_config(config)
    best_viterbi_variant: str | None = None
    best_viterbi_rmse = math.inf
    best_viterbi_selected: np.ndarray | None = None
    log_period = int(get_nested(config, "selector.log_period") or 10)
    for idx, spec in enumerate(specs, start=1):
        if idx % log_period == 0:
            print(f"[viterbi] evaluated {idx - 1}/{len(specs)} variants", flush=True)
        selected = viterbi_select(
            frame=frame,
            predicted_error=scores["predicted_error"],
            candidate_values=candidate_values,
            candidate_names=candidate_names,
            default_idx=default_idx,
            allowed_switch_idx=allowed_switch_idx,
            spec=spec,
        )
        params = {
            "switch_penalty": spec.switch_penalty,
            "nondefault_bias": spec.nondefault_bias,
            "jump_penalty_weight": spec.jump_penalty_weight,
            "jump_free_ft": spec.jump_free_ft,
            "jump_scale_ft": spec.jump_scale_ft,
            "max_abs_delta_vs_default": spec.max_abs_delta_vs_default,
            "max_pf_ancc_std": spec.max_pf_ancc_std,
            "min_md_since": spec.min_md_since,
            "min_segment_len": spec.min_segment_len,
        }
        metric, distribution, by_well, buckets = evaluate_selection(
            variant=spec.variant,
            mode="viterbi",
            selected_idx=selected,
            params=params,
            candidate_values=candidate_values,
            true_tvt=true_tvt,
            oracle_labels=oracle_labels,
            default_idx=default_idx,
            candidate_names=candidate_names,
            well_codes=well_codes,
            well_names=well_names,
            order=order,
            bucket_defs=bucket_defs,
        )
        metric_rows.append(metric)
        distribution_rows.extend(distribution)
        by_well_all.extend(by_well)
        bucket_rows.extend(buckets)
        params_rows.append({"variant": spec.variant, "mode": "viterbi", **params})
        rmse = float(metric["rmse_tvt"])
        if rmse < best_viterbi_rmse:
            best_viterbi_rmse = rmse
            best_viterbi_variant = spec.variant
            best_viterbi_selected = selected.copy()

    metrics = pd.DataFrame(metric_rows).sort_values("rmse_tvt")
    distribution = pd.DataFrame(distribution_rows)
    by_well = pd.DataFrame(by_well_all).sort_values(
        ["variant", "mode", "rmse_tvt"], ascending=[True, True, False]
    )
    buckets = pd.DataFrame(bucket_rows).sort_values(["variant", "mode", "bucket_family", "bucket"])
    params_frame = pd.DataFrame(params_rows)

    likpf_rmse = float(metrics.loc[metrics["variant"].eq("likpf_mean_single"), "rmse_tvt"].iloc[0])
    best_viterbi_delta = (
        float(best_viterbi_rmse - likpf_rmse) if best_viterbi_variant is not None else None
    )
    recommendation = (
        "segment_viterbi_supported_for_continuity_audit"
        if best_viterbi_delta is not None and best_viterbi_delta < 0.0
        else "segment_viterbi_not_supported"
    )

    prediction_frames = []
    save_variants = {
        "likpf_mean_single": baseline_selections["likpf_mean_single"][1],
        "exp176_error_ranker_rowwise": baseline_selections["exp176_error_ranker_rowwise"][1],
        "oracle": baseline_selections["oracle"][1],
    }
    if best_viterbi_variant is not None and best_viterbi_selected is not None:
        save_variants[best_viterbi_variant] = best_viterbi_selected
    for variant, selected in save_variants.items():
        mode = "viterbi" if variant == best_viterbi_variant else baseline_selections[variant][0]
        prediction_frames.append(
            selected_prediction_frame(
                frame=frame,
                variant=variant,
                mode=mode,
                selected_idx=selected,
                candidate_values=candidate_values,
                true_tvt=true_tvt,
                oracle_labels=oracle_labels,
                candidate_names=candidate_names,
            )
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)

    score_summary = pd.DataFrame(
        score_summary_rows(
            scores=scores,
            candidate_names=candidate_names,
            error_selected=error_selected,
            error_margin=error_margin,
            binary_margin=binary_margin,
            multiclass_margin=multiclass_margin,
        )
    )

    metrics_path = output_dir / f"{OUTPUT_PREFIX}_metrics.csv"
    predictions_path = output_dir / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
    distribution_path = output_dir / f"{OUTPUT_PREFIX}_selection_distribution.csv"
    by_well_path = output_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    buckets_path = output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    params_path = output_dir / f"{OUTPUT_PREFIX}_viterbi_params.csv"
    score_summary_path = output_dir / f"{OUTPUT_PREFIX}_score_summary.csv"
    model_manifest_path = output_dir / f"{OUTPUT_PREFIX}_exp176_model_manifest_resolved.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    distribution.to_csv(distribution_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    buckets.to_csv(buckets_path, index=False)
    params_frame.to_csv(params_path, index=False)
    score_summary.to_csv(score_summary_path, index=False)
    model_manifest.to_csv(model_manifest_path, index=False)

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
        "candidate_names": candidate_names,
        "default_candidate": candidate_names[default_idx],
        "allowed_switch_candidates": allowed_names,
        "viterbi_variant_count": int(len(specs)),
        "best_viterbi_variant": best_viterbi_variant,
        "best_viterbi_rmse_tvt": (
            float(best_viterbi_rmse) if np.isfinite(best_viterbi_rmse) else None
        ),
        "delta_rmse_vs_likpf_mean": best_viterbi_delta,
        "recommendation": recommendation,
        "source_meta": source_meta,
        "feature_enrichment": enrichment_meta,
        "typewell_late_range_prior": late_range_meta,
        "typewell_late_range_feature_count": int(len(late_range_columns)),
        "exp176_feature_schema": exp176_schema_meta,
        "exp176_model_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_path(manifest_path),
            "resolved_models": int(len(model_manifest)),
        },
        "sha256": {
            "metrics": sha256_path(metrics_path),
            "oof_predictions_decompressed": sha256_path(predictions_path, decompressed=True),
            "selection_distribution": sha256_path(distribution_path),
            "by_well": sha256_path(by_well_path),
            "bucket_metrics": sha256_path(buckets_path),
            "viterbi_params": sha256_path(params_path),
            "score_summary": sha256_path(score_summary_path),
            "prediction_by_variant": prediction_hashes,
        },
        "artifacts": {
            "metrics": metrics_path.name,
            "oof_predictions": predictions_path.name,
            "selection_distribution": distribution_path.name,
            "by_well": by_well_path.name,
            "bucket_metrics": buckets_path.name,
            "viterbi_params": params_path.name,
            "score_summary": score_summary_path.name,
            "exp176_model_manifest_resolved": model_manifest_path.name,
        },
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    with summary_path.open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
    summary["artifacts"]["summary"] = summary_path.name
    summary["sha256"]["summary"] = sha256_path(summary_path)

    print("Best rows:")
    print(metrics.head(10).to_string(index=False))
    print(f"Saved artifacts to {output_dir}")
    return summary


run_typewell_late_range_continuity_selector = run_segment_viterbi_candidate_selector
run_segment_continuity_selector = run_segment_viterbi_candidate_selector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--schema-path", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    output_dir = Path(args.output_dir) if args.output_dir else paths.artifacts_dir
    return run_typewell_late_range_continuity_selector(
        output_dir=output_dir,
        cache_path=args.cache_path,
        schema_path=args.schema_path,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    result = main()
    print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))

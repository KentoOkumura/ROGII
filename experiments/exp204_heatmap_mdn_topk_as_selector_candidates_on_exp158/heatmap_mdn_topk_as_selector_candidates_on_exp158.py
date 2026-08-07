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

OUTPUT_PREFIX = "exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158"
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
EXP182_VALIDATION_PREDICTIONS = (
    "exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe_validation_predictions.csv.gz"
)
EXP182_SAMPLE_INDEX = "exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe_sample_index.csv.gz"
EXP182_SUMMARY = "exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe_summary.json"
EXP202_HEATMAP_CANDIDATES = (
    "exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidates.csv.gz"
)
EXP065_CLUSTER_ASSIGNMENTS = "common_typewell_cluster_assignments.csv"
EXP109_OOF = "exp109_typewell_neighbor_prior_features_oof_predictions.csv.gz"
EXP114_OOF = "exp114_spatial_neighbor_prior_signal_audit_oof_predictions.csv.gz"
EXP114_WELL_GEOMETRY = "exp114_spatial_neighbor_prior_signal_audit_well_geometry_summary.csv"
EXP115_FOLD_ASSIGNMENTS = "exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv"
PROTECTED_COLUMNS = {"id", "well", "target", "true_tvt", "oracle_label", "oracle_candidate"}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    column: str


@dataclass(frozen=True)
class PriorSpec:
    name: str
    family: str
    prior_tvt: str
    prior_std: str | None = None
    prior_count: str | None = None
    neighbor_wells: str | None = None
    distance_mean: str | None = None
    azimuth_mismatch: str | None = None


@dataclass(frozen=True)
class ClusterGateSpec:
    name: str
    raw: dict[str, Any]


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
            Path("experiments")
            / "exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe"
            / "kaggle"
            / "output"
            / "train_v1"
            / "artifacts"
            / filename,
            Path("experiments")
            / "exp065_typewell_supertype_cluster_cv_audit"
            / "artifacts"
            / filename,
            Path("experiments")
            / "exp109_typewell_neighbor_prior_features"
            / "kaggle"
            / "output"
            / "train_v2"
            / "artifacts"
            / filename,
            Path("experiments")
            / "exp114_spatial_neighbor_prior_signal_audit"
            / "kaggle"
            / "output"
            / "train_v1"
            / "artifacts"
            / filename,
            Path("experiments")
            / "exp115_hidden_like_spatial_holdout_from_ppt"
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
    generated_candidate_columns = set(
        get_nested(config, "ranker.generated_candidate_columns") or []
    )
    required.update(
        spec.column
        for spec in candidates
        if spec.column not in auxiliary_columns and spec.column not in generated_candidate_columns
    )
    for key in [
        "ranker.context_columns",
        "ranker.multiobs_feature_columns",
        "ranker.optional_columns",
    ]:
        values = get_nested(config, key) or []
        required.update(str(value) for value in values if str(value) not in auxiliary_columns)
    return sorted(required)


def hmdn_candidate_rank(name: str) -> int | None:
    prefix = "hmdn_top"
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def base_candidate_specs(candidates: list[CandidateSpec]) -> list[CandidateSpec]:
    return [spec for spec in candidates if hmdn_candidate_rank(spec.name) is None]


def pairwise_candidate_specs(
    candidates: list[CandidateSpec], config: dict[str, Any]
) -> list[CandidateSpec]:
    scope = str(get_nested(config, "ranker.candidate_pairwise_scope") or "all")
    if scope == "all":
        return candidates
    if scope == "base_only":
        return base_candidate_specs(candidates)
    raise ValueError(f"unknown ranker.candidate_pairwise_scope: {scope}")


def candidate_minus_last_feature_name(spec: CandidateSpec, existing_columns: pd.Index) -> str:
    base_name = f"{spec.name}_minus_last"
    if hmdn_candidate_rank(spec.name) is not None or base_name in existing_columns:
        return f"candidate_{spec.name}_minus_last"
    return base_name


def assert_unique_columns(frame: pd.DataFrame, context: str) -> None:
    duplicate_columns = frame.columns[frame.columns.duplicated()].unique().tolist()
    if duplicate_columns:
        raise ValueError(f"{context} has duplicate columns: {duplicate_columns[:20]}")


def sample_row_indices(
    row_indices: np.ndarray,
    max_rows: int | None,
    *,
    seed: int,
) -> np.ndarray:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if max_rows is not None and len(row_indices) > max_rows:
        rng = np.random.default_rng(seed)
        row_indices = np.sort(rng.choice(row_indices, size=int(max_rows), replace=False))
    return row_indices


def _rank01(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values.astype(np.float32))
    return series.rank(method="average", pct=True).fillna(0.5).to_numpy(np.float32)


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=np.float32)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def robust_scale(values: np.ndarray, floor: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float(floor)
    q75, q25 = np.nanquantile(finite, [0.75, 0.25])
    iqr = float(q75 - q25)
    mad = float(np.nanmedian(np.abs(finite - np.nanmedian(finite))) * 1.4826)
    return max(iqr / 1.349 if iqr > 0.0 else 0.0, mad, float(floor))


def parse_prior_specs(config: dict[str, Any]) -> list[PriorSpec]:
    specs: list[PriorSpec] = []
    for raw in get_nested(config, "ranker.cluster_prior_features.prior_variants") or []:
        specs.append(
            PriorSpec(
                name=str(raw["name"]),
                family=str(raw["family"]),
                prior_tvt=str(raw["prior_tvt"]),
                prior_std=None if raw.get("prior_std") is None else str(raw["prior_std"]),
                prior_count=None if raw.get("prior_count") is None else str(raw["prior_count"]),
                neighbor_wells=(
                    None if raw.get("neighbor_wells") is None else str(raw["neighbor_wells"])
                ),
                distance_mean=(
                    None if raw.get("distance_mean") is None else str(raw["distance_mean"])
                ),
                azimuth_mismatch=(
                    None
                    if raw.get("azimuth_mismatch") is None
                    else str(raw["azimuth_mismatch"])
                ),
            )
        )
    if not specs:
        raise ValueError("ranker.cluster_prior_features.prior_variants must not be empty")
    return specs


def parse_cluster_gates(config: dict[str, Any]) -> list[ClusterGateSpec]:
    gates = [
        ClusterGateSpec(name=str(raw["name"]), raw=dict(raw))
        for raw in (get_nested(config, "ranker.cluster_prior_features.cluster_gates") or [])
    ]
    if not gates:
        gates.append(
            ClusterGateSpec(
                name="any_outlier_signal_k8",
                raw={
                    "name": "any_outlier_signal_k8",
                    "any_of": [
                        {"own_cluster_dist_z_gt": 1.5},
                        {"nearest_other_closer": True},
                        {"nearby_majority_diff": True, "nearby_k": 8},
                    ],
                },
            )
        )
    return gates


def read_prior_columns(
    *,
    source: Path,
    columns: set[str],
    max_rows: int | None,
    family_prefix: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(source, nrows=0).columns.tolist()
    usecols = ["id", *sorted(column for column in columns if column in header)]
    missing = sorted(column for column in columns if column not in header)
    if missing:
        raise ValueError(f"{source} is missing prior columns: {missing}")
    frame = pd.read_csv(
        source,
        usecols=usecols,
        nrows=max_rows,
        dtype={"id": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    rename: dict[str, str] = {}
    for column in frame.columns:
        if column == "id":
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
        rename[column] = f"copcf_{family_prefix}_{column}"
    frame = frame.rename(columns=rename)
    meta = {
        "path": str(source),
        "rows": int(len(frame)),
        "columns": [column for column in frame.columns if column != "id"],
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": (
            sha256_path(source, decompressed=True) if source.suffix == ".gz" else None
        ),
    }
    return frame, meta


def read_prior_feature_frame(
    config: dict[str, Any],
    priors: list[PriorSpec],
    *,
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    typewell_cols: set[str] = set()
    spatial_cols: set[str] = set()
    for prior in priors:
        target = typewell_cols if prior.family == "typewell" else spatial_cols
        for column in [
            prior.prior_tvt,
            prior.prior_std,
            prior.prior_count,
            prior.neighbor_wells,
            prior.distance_mean,
            prior.azimuth_mismatch,
        ]:
            if column:
                target.add(column)

    frames: list[pd.DataFrame] = []
    meta: dict[str, Any] = {}
    if typewell_cols:
        source = find_artifact(
            EXP109_OOF,
            get_nested(config, "data.exp109_oof_predictions_local"),
        )
        typewell, typewell_meta = read_prior_columns(
            source=source,
            columns=typewell_cols,
            max_rows=max_rows,
            family_prefix="typewell",
        )
        frames.append(typewell)
        meta["typewell"] = typewell_meta
    if spatial_cols:
        source = find_artifact(
            EXP114_OOF,
            get_nested(config, "data.exp114_oof_predictions_local"),
        )
        spatial, spatial_meta = read_prior_columns(
            source=source,
            columns=spatial_cols,
            max_rows=max_rows,
            family_prefix="spatial",
        )
        frames.append(spatial)
        meta["spatial"] = spatial_meta
    if not frames:
        raise ValueError("no prior source columns configured")
    prior_frame = frames[0]
    for other in frames[1:]:
        before = len(prior_frame)
        prior_frame = prior_frame.merge(other, on="id", how="outer", validate="one_to_one")
        if len(prior_frame) != before:
            raise ValueError("prior source merge changed row count")
    return prior_frame, meta


def read_cluster_assignments(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        EXP065_CLUSTER_ASSIGNMENTS,
        get_nested(config, "data.exp065_cluster_assignments_local"),
    )
    frame = pd.read_csv(source, dtype=str)
    required = {"method", "threshold", "cluster_id", "well_id", "cluster_size"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    method = str(get_nested(config, "cluster.assignment_method") or "native_overlap")
    threshold = str(get_nested(config, "cluster.assignment_threshold") or "1")
    subset = frame[(frame["method"] == method) & (frame["threshold"].astype(str) == threshold)]
    if subset.empty:
        raise ValueError(f"no cluster assignments for method={method} threshold={threshold}")
    subset = subset.copy()
    subset["cluster_size"] = pd.to_numeric(subset["cluster_size"], errors="coerce").fillna(0)
    subset["cluster_size"] = subset["cluster_size"].astype(int)
    meta = {
        "path": str(source),
        "sha256": sha256_path(source),
        "method": method,
        "threshold": threshold,
        "rows": int(len(subset)),
        "wells": int(subset["well_id"].nunique()),
        "clusters": int(subset["cluster_id"].nunique()),
    }
    return subset, meta


def read_well_geometry(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        EXP114_WELL_GEOMETRY,
        get_nested(config, "data.exp114_well_geometry_local"),
    )
    frame = pd.read_csv(source, dtype={"well": str})
    required = {"well", "centroid_x", "centroid_y"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    for column in frame.columns:
        if column != "well":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    meta = {
        "path": str(source),
        "sha256": sha256_path(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
    }
    return frame, meta


def build_cluster_features(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    assignments, assignment_meta = read_cluster_assignments(config)
    geometry, geometry_meta = read_well_geometry(config)
    min_cluster_size = int(get_nested(config, "cluster.min_cluster_size") or 2)
    scale_floor = float(get_nested(config, "cluster.robust_scale_floor_ft") or 250.0)
    nearby_k_values = [
        int(value) for value in get_nested(config, "cluster.nearby_k_values") or [5, 8, 12]
    ]
    majority_min_share = float(get_nested(config, "cluster.nearby_majority_min_share") or 0.5)

    assignment_cols = ["well_id", "cluster_id", "cluster_size", "representative_well_id"]
    available = [column for column in assignment_cols if column in assignments.columns]
    joined = geometry.merge(
        assignments[available].rename(columns={"well_id": "well"}),
        on="well",
        how="left",
        validate="one_to_one",
    )
    joined["cluster_size"] = pd.to_numeric(
        joined["cluster_size"], errors="coerce"
    ).fillna(0).astype(int)
    joined["cluster_id"] = joined["cluster_id"].astype("string")

    valid_cluster = joined["cluster_id"].notna() & (joined["cluster_size"] >= min_cluster_size)
    cluster_stats_rows: list[dict[str, Any]] = []
    for cluster_id, group in joined[valid_cluster].groupby("cluster_id", sort=False):
        x = numeric_array(group, "centroid_x").astype(np.float64)
        y = numeric_array(group, "centroid_y").astype(np.float64)
        center_x = float(np.nanmedian(x))
        center_y = float(np.nanmedian(y))
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        cluster_stats_rows.append(
            {
                "cluster_id": str(cluster_id),
                "cluster_center_x": center_x,
                "cluster_center_y": center_y,
                "cluster_member_wells": int(len(group)),
                "cluster_dist_median": float(np.nanmedian(dist)),
                "cluster_dist_scale": robust_scale(dist, scale_floor),
                "cluster_dist_p90": float(np.nanquantile(dist, 0.90)) if len(dist) else np.nan,
            }
        )
    cluster_stats = pd.DataFrame(cluster_stats_rows)
    joined = joined.merge(cluster_stats, on="cluster_id", how="left")
    x = numeric_array(joined, "centroid_x").astype(np.float64)
    y = numeric_array(joined, "centroid_y").astype(np.float64)
    cx = numeric_array(joined, "cluster_center_x").astype(np.float64)
    cy = numeric_array(joined, "cluster_center_y").astype(np.float64)
    joined["copcf_own_cluster_dist"] = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float32)
    joined["copcf_own_cluster_dist_z"] = (
        (
            numeric_array(joined, "copcf_own_cluster_dist")
            - numeric_array(joined, "cluster_dist_median")
        )
        / numeric_array(joined, "cluster_dist_scale")
    ).astype(np.float32)

    centers = cluster_stats[["cluster_id", "cluster_center_x", "cluster_center_y"]].copy()
    center_ids = centers["cluster_id"].astype(str).to_numpy()
    center_xy = centers[["cluster_center_x", "cluster_center_y"]].to_numpy(np.float64)
    well_xy = joined[["centroid_x", "centroid_y"]].to_numpy(np.float64)
    nearest_ids: list[str | None] = []
    nearest_dist: list[float] = []
    for own_cluster, point in zip(joined["cluster_id"].astype("string"), well_xy, strict=False):
        if len(center_xy) <= 1 or pd.isna(own_cluster) or not np.isfinite(point).all():
            nearest_ids.append(None)
            nearest_dist.append(np.nan)
            continue
        dist = np.sqrt(np.sum((center_xy - point) ** 2, axis=1))
        dist[center_ids == str(own_cluster)] = np.inf
        idx = int(np.argmin(dist))
        nearest_ids.append(str(center_ids[idx]) if np.isfinite(dist[idx]) else None)
        nearest_dist.append(float(dist[idx]) if np.isfinite(dist[idx]) else np.nan)
    joined["nearest_other_cluster_id"] = nearest_ids
    joined["copcf_nearest_other_cluster_dist"] = np.asarray(nearest_dist, dtype=np.float32)
    joined["copcf_nearest_other_closer"] = (
        numeric_array(joined, "copcf_nearest_other_cluster_dist")
        < numeric_array(joined, "copcf_own_cluster_dist")
    )

    dist_matrix = np.sqrt(np.sum((well_xy[:, None, :] - well_xy[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    cluster_values = joined["cluster_id"].astype("string").to_numpy()
    for k in nearby_k_values:
        majority_counts: list[int] = []
        majority_shares: list[float] = []
        diff_flags: list[bool] = []
        for i in range(len(joined)):
            if not np.isfinite(dist_matrix[i]).any():
                majority_counts.append(0)
                majority_shares.append(0.0)
                diff_flags.append(False)
                continue
            idx = np.argsort(dist_matrix[i])[:k]
            values = [str(cluster_values[j]) for j in idx if not pd.isna(cluster_values[j])]
            if not values:
                majority_counts.append(0)
                majority_shares.append(0.0)
                diff_flags.append(False)
                continue
            counts = pd.Series(values).value_counts()
            majority_cluster = str(counts.index[0])
            majority_count = int(counts.iloc[0])
            share = float(majority_count / max(len(values), 1))
            own_cluster = None if pd.isna(cluster_values[i]) else str(cluster_values[i])
            majority_counts.append(majority_count)
            majority_shares.append(share)
            diff_flags.append(
                bool(
                    own_cluster is not None
                    and majority_cluster != own_cluster
                    and share >= majority_min_share
                )
            )
        joined[f"copcf_nearby_majority_count_k{k}"] = majority_counts
        joined[f"copcf_nearby_majority_share_k{k}"] = majority_shares
        joined[f"copcf_nearby_majority_diff_k{k}"] = diff_flags

    joined["copcf_cluster_feature_valid"] = valid_cluster.to_numpy()
    output_columns = [
        "well",
        "cluster_id",
        "cluster_size",
        "copcf_cluster_feature_valid",
        "copcf_own_cluster_dist",
        "copcf_own_cluster_dist_z",
        "copcf_nearest_other_cluster_dist",
        "copcf_nearest_other_closer",
    ]
    for k in nearby_k_values:
        output_columns.extend(
            [
                f"copcf_nearby_majority_count_k{k}",
                f"copcf_nearby_majority_share_k{k}",
                f"copcf_nearby_majority_diff_k{k}",
            ]
        )
    out = joined[output_columns].copy()
    for column in out.columns:
        if column.startswith("copcf_") and out[column].dtype == bool:
            out[column] = out[column].astype(np.float32)
    meta = {
        "assignments": assignment_meta,
        "geometry": geometry_meta,
        "min_cluster_size": min_cluster_size,
        "robust_scale_floor_ft": scale_floor,
        "nearby_k_values": nearby_k_values,
        "nearby_majority_min_share": majority_min_share,
        "cluster_feature_wells": int(out["copcf_cluster_feature_valid"].sum()),
        "clusters_used": int(cluster_stats["cluster_id"].nunique()) if len(cluster_stats) else 0,
    }
    return out, meta


def condition_mask(frame: pd.DataFrame, raw: dict[str, Any]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    if raw.get("own_cluster_dist_z_gt") is not None:
        mask &= numeric_array(frame, "copcf_own_cluster_dist_z") > float(
            raw["own_cluster_dist_z_gt"]
        )
    if bool(raw.get("nearest_other_closer", False)):
        mask &= frame["copcf_nearest_other_closer"].fillna(False).to_numpy(bool)
    if bool(raw.get("nearby_majority_diff", False)):
        k = int(raw.get("nearby_k", 8))
        column = f"copcf_nearby_majority_diff_k{k}"
        if column not in frame.columns:
            return np.zeros(len(frame), dtype=bool)
        mask &= frame[column].fillna(False).to_numpy(bool)
    return mask


def cluster_gate_mask(frame: pd.DataFrame, gate: ClusterGateSpec) -> np.ndarray:
    base = frame["copcf_cluster_feature_valid"].fillna(False).to_numpy(bool)
    raw = dict(gate.raw)
    if raw.get("all_rows"):
        return np.ones(len(frame), dtype=bool)
    if raw.get("any_of"):
        options = [condition_mask(frame, dict(item)) for item in raw["any_of"]]
        return base & np.logical_or.reduce(options) if options else np.zeros(len(frame), dtype=bool)
    return base & condition_mask(frame, raw)


def prefixed_prior_column(prior: PriorSpec, source_column: str | None) -> str | None:
    if source_column is None:
        return None
    return f"copcf_{prior.family}_{source_column}"


def add_cluster_prior_confidence_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    max_rows: int | None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    settings = get_nested(config, "ranker.cluster_prior_features") or {}
    if not settings.get("enabled", False):
        return frame, [], {"enabled": False}

    priors = parse_prior_specs(config)
    gates = parse_cluster_gates(config)
    prior_frame, prior_meta = read_prior_feature_frame(config, priors, max_rows=max_rows)
    cluster_features, cluster_meta = build_cluster_features(config)

    before_rows = len(frame)
    frame = frame.merge(prior_frame, on="id", how="left", validate="one_to_one")
    if len(frame) != before_rows:
        raise ValueError("prior feature join changed row count")
    frame = frame.merge(cluster_features, on="well", how="left", validate="many_to_one")
    if len(frame) != before_rows:
        raise ValueError("cluster feature join changed row count")

    generated: dict[str, np.ndarray] = {}
    row_cluster_cols = [
        "copcf_cluster_feature_valid",
        "copcf_own_cluster_dist",
        "copcf_own_cluster_dist_z",
        "copcf_nearest_other_cluster_dist",
        "copcf_nearest_other_closer",
    ]
    for column in row_cluster_cols:
        if column in frame.columns:
            generated[column] = numeric_array(frame, column)
    for k in get_nested(config, "cluster.nearby_k_values") or [5, 8, 12]:
        for suffix in ["count", "share", "diff"]:
            column = f"copcf_nearby_majority_{suffix}_k{int(k)}"
            if column in frame.columns:
                generated[column] = numeric_array(frame, column)

    for gate in gates:
        gate_col = f"copcf_gate_{gate.name}"
        generated[gate_col] = cluster_gate_mask(frame, gate).astype(np.float32)
        frame[gate_col] = generated[gate_col]
        ratio_col = f"copcf_well_gate_ratio_{gate.name}"
        generated[ratio_col] = (
            frame.groupby("well", observed=True)[gate_col].transform("mean").to_numpy(np.float32)
        )
        frame[ratio_col] = generated[ratio_col]

    any_gate_columns = [f"copcf_gate_{gate.name}" for gate in gates]
    if any_gate_columns:
        generated["copcf_any_configured_gate"] = (
            np.column_stack([frame[col].to_numpy(np.float32) for col in any_gate_columns]).max(
                axis=1
            )
        ).astype(np.float32)

    primary_typewell = str(settings.get("primary_typewell_prior", "typewell_native_overlap_0p999"))
    primary_spatial = str(
        settings.get("primary_spatial_prior", "spatial_xy_plus_trajectory_shape_k8")
    )
    prior_lookup = {prior.name: prior for prior in priors}
    if primary_typewell in prior_lookup and primary_spatial in prior_lookup:
        typewell_prior = prior_lookup[primary_typewell]
        spatial_prior = prior_lookup[primary_spatial]
        typewell_col = prefixed_prior_column(typewell_prior, typewell_prior.prior_tvt)
        spatial_col = prefixed_prior_column(spatial_prior, spatial_prior.prior_tvt)
        if typewell_col in frame.columns and spatial_col in frame.columns:
            agreement = numeric_array(frame, typewell_col) - numeric_array(frame, spatial_col)
            generated["copcf_typewell_spatial_prior_delta"] = agreement.astype(np.float32)
            generated["copcf_typewell_spatial_prior_abs_delta"] = np.abs(agreement).astype(
                np.float32
            )

    for prior in priors:
        prior_tvt_col = prefixed_prior_column(prior, prior.prior_tvt)
        prior_std_col = prefixed_prior_column(prior, prior.prior_std)
        prior_count_col = prefixed_prior_column(prior, prior.prior_count)
        prior_neighbors_col = prefixed_prior_column(prior, prior.neighbor_wells)
        if prior_tvt_col is None or prior_tvt_col not in frame.columns:
            continue
        prior_tvt = numeric_array(frame, prior_tvt_col)
        valid = np.isfinite(prior_tvt)
        generated[f"copcf_{prior.name}_valid_prior"] = valid.astype(np.float32)
        if prior_std_col and prior_std_col in frame.columns:
            generated[f"copcf_{prior.name}_prior_std"] = numeric_array(frame, prior_std_col)
        if prior_count_col and prior_count_col in frame.columns:
            generated[f"copcf_{prior.name}_prior_count"] = numeric_array(frame, prior_count_col)
        if prior_neighbors_col and prior_neighbors_col in frame.columns:
            generated[f"copcf_{prior.name}_neighbor_wells"] = numeric_array(
                frame, prior_neighbors_col
            )

    generated_columns: list[str] = []
    for column, values in generated.items():
        values = np.asarray(values, dtype=np.float32)
        values[~np.isfinite(values)] = np.nan
        frame[column] = values
        generated_columns.append(column)

    max_missing_rate = float(settings.get("max_missing_rate", 0.0))
    if max_missing_rate < 1.0 and generated_columns:
        missing_rate = float(frame[generated_columns].isna().mean().max())
        if missing_rate > max_missing_rate:
            raise ValueError(
                f"cluster/prior feature join missing_rate={missing_rate:.6f} "
                f"> max_missing_rate={max_missing_rate:.6f}"
            )
    else:
        missing_rate = (
            float(frame[generated_columns].isna().mean().max()) if generated_columns else 0.0
        )

    meta = {
        "enabled": True,
        "priors": [prior.__dict__ for prior in priors],
        "cluster_gates": [gate.raw for gate in gates],
        "prior_sources": prior_meta,
        "cluster_features": cluster_meta,
        "generated_feature_count": int(len(generated_columns)),
        "generated_feature_columns": generated_columns,
        "missing_rate_max": missing_rate,
    }
    return frame, generated_columns, meta


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


def read_heatmap_predictions(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        EXP182_VALIDATION_PREDICTIONS,
        get_nested(config, "data.exp182_validation_predictions_local"),
    )
    sample_index = find_artifact(
        EXP182_SAMPLE_INDEX,
        get_nested(config, "data.exp182_sample_index_local"),
    )
    summary = find_artifact(
        EXP182_SUMMARY,
        get_nested(config, "data.exp182_summary_local"),
    )
    heatmap_config = get_nested(config, "ranker.heatmap_path_features") or {}
    topk_ranks = [int(value) for value in heatmap_config.get("topk_ranks", [1, 2, 3, 5, 10])]
    max_rank = max(max(topk_ranks), 10)
    allowed_specs = {
        str(heatmap_config.get("primary_run_spec", "base_real_w128_b64_fullfold")),
        str(heatmap_config.get("shuffled_run_spec", "base_shuffled_w128_b64_fullfold")),
        str(heatmap_config.get("no_gr_run_spec", "base_no_gr_w128_b64_fullfold")),
    }
    header = pd.read_csv(source, nrows=0).columns.tolist()
    required = {
        "run_spec",
        "split",
        "well",
        "row_center",
        "last_known_tvt",
        "prior_center_tvt",
        "md_since_prefix",
        "z_since_prefix",
        "label_fraction",
        "path_step_abs_mean_ft",
        "path_step_abs_max_ft",
    }
    for rank in range(1, max_rank + 1):
        required.add(f"pred_top{rank}_tvt")
        required.add(f"pred_top{rank}_score")
    missing = sorted(column for column in required if column not in header)
    if missing:
        raise ValueError(f"{source} is missing heatmap prediction columns: {missing}")
    forbidden = {
        column
        for column in header
        if column.endswith("_abs_error")
        or column.endswith("_within10")
        or column.startswith("top")
        and ("best_abs_error" in column or "within10" in column)
        or column in {"true_center_tvt", "true_center_tvt_sample", "target_in_grid"}
    }
    usecols = sorted(required)
    leaked = sorted(set(usecols).intersection(forbidden))
    if leaked:
        raise ValueError(
            f"heatmap feature usecols include forbidden target-derived columns: {leaked}"
        )
    frame = pd.read_csv(source, usecols=usecols, dtype={"well": str}, low_memory=False)
    frame["well"] = frame["well"].astype(str)
    frame["run_spec"] = frame["run_spec"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame = frame[frame["split"].eq("valid") & frame["run_spec"].isin(allowed_specs)].copy()
    if frame.empty:
        raise ValueError(
            f"no exp182 valid heatmap predictions for run specs={sorted(allowed_specs)}"
        )
    for column in frame.columns:
        if column not in {"run_spec", "split", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    run_spec_counts = frame.groupby("run_spec", observed=True).size().to_dict()
    missing_specs = sorted(spec for spec in allowed_specs if spec not in run_spec_counts)
    if missing_specs:
        raise ValueError(f"missing heatmap run specs in {source}: {missing_specs}")
    meta = {
        "path": str(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "run_spec_counts": {str(key): int(value) for key, value in run_spec_counts.items()},
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": (
            sha256_path(source, decompressed=True) if source.suffix == ".gz" else None
        ),
        "sample_index_path": str(sample_index),
        "sample_index_sha256": sha256_path(sample_index),
        "sample_index_decompressed_sha256": (
            sha256_path(sample_index, decompressed=True) if sample_index.suffix == ".gz" else None
        ),
        "summary_path": str(summary),
        "summary_sha256": sha256_path(summary),
    }
    return frame, meta


def _positive_score_matrix(source: pd.DataFrame, max_rank: int = 10) -> np.ndarray:
    score_cols = [f"pred_top{rank}_score" for rank in range(1, max_rank + 1)]
    values = source[score_cols].to_numpy(np.float32)
    return np.where(np.isfinite(values) & (values > 0.0), values, 0.0).astype(np.float32)


def _tvt_matrix(source: pd.DataFrame, max_rank: int = 10) -> np.ndarray:
    tvt_cols = [f"pred_top{rank}_tvt" for rank in range(1, max_rank + 1)]
    return source[tvt_cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)


def build_heatmap_source_features(
    source: pd.DataFrame,
    *,
    tag: str,
    config: dict[str, Any],
    keep_all_tvt: bool,
) -> tuple[pd.DataFrame, list[str]]:
    heatmap_config = get_nested(config, "ranker.heatmap_path_features") or {}
    eps = float(heatmap_config.get("confidence_eps", 1e-6))
    topk_ranks = [int(value) for value in heatmap_config.get("topk_ranks", [1, 2, 3, 5, 10])]
    max_rank = max(max(topk_ranks), 10)
    scores = _positive_score_matrix(source, max_rank=max_rank)
    tvts = _tvt_matrix(source, max_rank=max_rank)
    score_sum = np.maximum(scores.sum(axis=1), eps)
    probs = scores / score_sum[:, None]
    entropy = -np.sum(probs * np.log(np.maximum(probs, eps)), axis=1) / np.log(float(max_rank))
    generated: dict[str, np.ndarray] = {
        f"hmpf_{tag}_top1_score": scores[:, 0],
        f"hmpf_{tag}_top1_top2_score_margin": scores[:, 0] - scores[:, 1],
        f"hmpf_{tag}_top1_top3_score_margin": scores[:, 0] - scores[:, 2],
        f"hmpf_{tag}_score_entropy": entropy.astype(np.float32),
        f"hmpf_{tag}_score_top3_mass": (scores[:, :3].sum(axis=1) / score_sum).astype(
            np.float32
        ),
        f"hmpf_{tag}_score_top5_mass": (scores[:, :5].sum(axis=1) / score_sum).astype(
            np.float32
        ),
        f"hmpf_{tag}_path_step_abs_mean_ft": numeric_array(source, "path_step_abs_mean_ft"),
        f"hmpf_{tag}_path_step_abs_max_ft": numeric_array(source, "path_step_abs_max_ft"),
        f"hmpf_{tag}_prior_center_minus_last": (
            numeric_array(source, "prior_center_tvt") - numeric_array(source, "last_known_tvt")
        ).astype(np.float32),
        f"hmpf_{tag}_md_since_prefix": numeric_array(source, "md_since_prefix"),
        f"hmpf_{tag}_z_since_prefix": numeric_array(source, "z_since_prefix"),
        f"hmpf_{tag}_label_fraction_proxy": numeric_array(source, "label_fraction"),
    }
    ranks_to_store = range(1, max_rank + 1) if keep_all_tvt else [1]
    for rank in ranks_to_store:
        generated[f"hmpf_{tag}_top{rank}_tvt"] = tvts[:, rank - 1]
    for topk in topk_ranks:
        subset = tvts[:, :topk]
        generated[f"hmpf_{tag}_top{topk}_tvt_mean"] = np.nanmean(subset, axis=1).astype(
            np.float32
        )
        generated[f"hmpf_{tag}_top{topk}_tvt_std"] = np.nanstd(subset, axis=1).astype(
            np.float32
        )
        generated[f"hmpf_{tag}_top{topk}_tvt_range"] = (
            np.nanmax(subset, axis=1) - np.nanmin(subset, axis=1)
        ).astype(np.float32)
    generated[f"hmpf_{tag}_top1_minus_last"] = (
        generated[f"hmpf_{tag}_top1_tvt"] - numeric_array(source, "last_known_tvt")
    ).astype(np.float32)
    generated[f"hmpf_{tag}_top1_minus_prior_center"] = (
        generated[f"hmpf_{tag}_top1_tvt"] - numeric_array(source, "prior_center_tvt")
    ).astype(np.float32)
    generated[f"hmpf_{tag}_top3_mean_minus_prior_center"] = (
        generated[f"hmpf_{tag}_top3_tvt_mean"] - numeric_array(source, "prior_center_tvt")
    ).astype(np.float32)
    out = source[["well", "row_center"]].copy()
    out["row_center"] = pd.to_numeric(out["row_center"], errors="coerce").astype(np.float32)
    generated_columns: list[str] = []
    for column, values in generated.items():
        safe = np.asarray(values, dtype=np.float32)
        if not np.isfinite(safe).all():
            finite = safe[np.isfinite(safe)]
            fill_value = float(np.nanmedian(finite)) if len(finite) else 0.0
            safe = np.where(np.isfinite(safe), safe, fill_value).astype(np.float32)
        out[column] = safe
        generated_columns.append(column)
    out = out.groupby(["well", "row_center"], as_index=False, observed=True).mean(numeric_only=True)
    return out, generated_columns


def interpolate_heatmap_features_for_frame(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    feature_columns: list[str],
    *,
    prefix: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    row_index = _row_indices_from_ids(frame["id"]).astype(np.float32)
    aligned = pd.DataFrame({"id": frame["id"].to_numpy()})
    for column in feature_columns:
        aligned[column] = np.zeros(len(frame), dtype=np.float32)
    source_valid = np.zeros(len(frame), dtype=np.float32)
    coverage_left = np.zeros(len(frame), dtype=np.float32)
    coverage_right = np.zeros(len(frame), dtype=np.float32)
    nearest_distance = np.full(len(frame), np.nan, dtype=np.float32)
    source_by_well = {
        str(well): group.sort_values("row_center")
        for well, group in source.groupby("well", observed=True)
    }
    missing_wells: list[str] = []
    for well, positions in frame.groupby("well", sort=False, observed=True).indices.items():
        well_key = str(well)
        target_rows = row_index[positions]
        source_group = source_by_well.get(well_key)
        if source_group is None or source_group.empty:
            missing_wells.append(well_key)
            continue
        x = source_group["row_center"].to_numpy(np.float32)
        order = np.argsort(x)
        x = x[order]
        unique_x, unique_idx = np.unique(x, return_index=True)
        source_valid[positions] = 1.0
        coverage_left[positions] = (target_rows >= float(unique_x.min())).astype(np.float32)
        coverage_right[positions] = (target_rows <= float(unique_x.max())).astype(np.float32)
        nearest_pos = np.searchsorted(unique_x, target_rows)
        left = np.clip(nearest_pos - 1, 0, len(unique_x) - 1)
        right = np.clip(nearest_pos, 0, len(unique_x) - 1)
        nearest_distance[positions] = np.minimum(
            np.abs(target_rows - unique_x[left]),
            np.abs(target_rows - unique_x[right]),
        ).astype(np.float32)
        for column in feature_columns:
            y = source_group[column].to_numpy(np.float32)[order][unique_idx]
            finite = np.isfinite(y)
            if not finite.any():
                values = np.zeros(len(target_rows), dtype=np.float32)
            elif finite.sum() == 1:
                values = np.full(len(target_rows), float(y[finite][0]), dtype=np.float32)
            else:
                values = np.interp(
                    target_rows.astype(np.float64),
                    unique_x[finite].astype(np.float64),
                    y[finite].astype(np.float64),
                    left=float(y[finite][0]),
                    right=float(y[finite][-1]),
                ).astype(np.float32)
            aligned.loc[positions, column] = values
    aligned[f"{prefix}source_valid"] = source_valid
    aligned[f"{prefix}row_coverage_left"] = coverage_left
    aligned[f"{prefix}row_coverage_right"] = coverage_right
    aligned[f"{prefix}nearest_sample_distance_rows"] = np.where(
        np.isfinite(nearest_distance), nearest_distance, 1_000_000.0
    ).astype(np.float32)
    meta = {
        "missing_wells": missing_wells[:20],
        "missing_well_count": int(len(missing_wells)),
        "source_valid_rate": float(source_valid.mean()) if len(source_valid) else 0.0,
        "nearest_sample_distance_rows_p95": float(
            np.nanquantile(nearest_distance[np.isfinite(nearest_distance)], 0.95)
        )
        if np.isfinite(nearest_distance).any()
        else None,
    }
    return aligned, meta


def add_heatmap_path_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
    candidates: list[CandidateSpec],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    heatmap_config = get_nested(config, "ranker.heatmap_path_features") or {}
    if not heatmap_config.get("enabled", False):
        return frame, [], {"enabled": False}
    prefix = str(heatmap_config.get("prefix", "hmpf_"))
    predictions, source_meta = read_heatmap_predictions(config)
    run_specs = {
        "real": str(heatmap_config.get("primary_run_spec", "base_real_w128_b64_fullfold")),
        "shuffled": str(
            heatmap_config.get("shuffled_run_spec", "base_shuffled_w128_b64_fullfold")
        ),
        "no_gr": str(heatmap_config.get("no_gr_run_spec", "base_no_gr_w128_b64_fullfold")),
    }
    all_columns: list[str] = []
    interpolation_meta: dict[str, Any] = {}
    before_rows = len(frame)
    for tag, run_spec in run_specs.items():
        source = predictions[predictions["run_spec"].eq(run_spec)].copy()
        source_features, source_columns = build_heatmap_source_features(
            source,
            tag=tag,
            config=config,
            keep_all_tvt=(tag == "real"),
        )
        aligned, meta = interpolate_heatmap_features_for_frame(
            frame,
            source_features,
            source_columns,
            prefix=prefix,
        )
        interpolation_meta[tag] = {"run_spec": run_spec, **meta}
        join_columns = ["id", *source_columns]
        if tag == "real":
            join_columns.extend(
                [
                    f"{prefix}source_valid",
                    f"{prefix}row_coverage_left",
                    f"{prefix}row_coverage_right",
                    f"{prefix}nearest_sample_distance_rows",
                ]
            )
        frame = frame.merge(aligned[join_columns], on="id", how="left", validate="one_to_one")
        if len(frame) != before_rows:
            raise ValueError("heatmap feature join changed row count")
        all_columns.extend([column for column in join_columns if column != "id"])

    distance_candidates = base_candidate_specs(candidates)
    if not distance_candidates:
        distance_candidates = candidates
    candidate_matrix = np.column_stack(
        [
            pd.to_numeric(frame[spec.column], errors="coerce").to_numpy(np.float32)
            for spec in distance_candidates
            if spec.column in frame.columns
        ]
    )
    candidate_mean = np.nanmean(candidate_matrix, axis=1).astype(np.float32)
    candidate_min_abs = np.nanmin(
        np.abs(candidate_matrix - frame[f"{prefix}real_top1_tvt"].to_numpy(np.float32)[:, None]),
        axis=1,
    ).astype(np.float32)
    top3_mean = frame[f"{prefix}real_top3_tvt_mean"].to_numpy(np.float32)
    top1 = frame[f"{prefix}real_top1_tvt"].to_numpy(np.float32)
    extra_features: dict[str, np.ndarray] = {}
    for ref in ["likpf_mean", "beam_mean", "pf_ancc", "tvt_densew"]:
        if ref in frame.columns:
            extra_features[f"{prefix}real_top1_minus_{ref}"] = (
                top1 - frame[ref].to_numpy(np.float32)
            ).astype(np.float32)
            extra_features[f"{prefix}real_top3_mean_minus_{ref}"] = (
                top3_mean - frame[ref].to_numpy(np.float32)
            ).astype(np.float32)
    extra_features[f"{prefix}real_top1_vs_candidate_mean_abs"] = np.abs(
        top1 - candidate_mean
    ).astype(np.float32)
    extra_features[f"{prefix}real_top1_vs_candidate_min_abs"] = candidate_min_abs
    extra_features[f"{prefix}real_top3_mean_vs_candidate_mean_abs"] = np.abs(
        top3_mean - candidate_mean
    ).astype(np.float32)
    for gap_name, left, right in [
        (
            "real_vs_shuffled_top1_score_gap",
            f"{prefix}real_top1_score",
            f"{prefix}shuffled_top1_score",
        ),
        ("real_vs_no_gr_top1_score_gap", f"{prefix}real_top1_score", f"{prefix}no_gr_top1_score"),
        (
            "real_vs_shuffled_margin_gap",
            f"{prefix}real_top1_top2_score_margin",
            f"{prefix}shuffled_top1_top2_score_margin",
        ),
        (
            "real_vs_no_gr_margin_gap",
            f"{prefix}real_top1_top2_score_margin",
            f"{prefix}no_gr_top1_top2_score_margin",
        ),
    ]:
        extra_features[f"{prefix}{gap_name}"] = (
            frame[left].to_numpy(np.float32) - frame[right].to_numpy(np.float32)
        ).astype(np.float32)
    for gap_name, other in [
        ("real_vs_shuffled_top1_tvt_abs_delta", f"{prefix}shuffled_top1_tvt"),
        ("real_vs_no_gr_top1_tvt_abs_delta", f"{prefix}no_gr_top1_tvt"),
    ]:
        extra_features[f"{prefix}{gap_name}"] = np.abs(
            top1 - frame[other].to_numpy(np.float32)
        ).astype(np.float32)
    score_rank = _rank01(frame[f"{prefix}real_top1_score"].to_numpy(np.float32))
    margin_rank = _rank01(frame[f"{prefix}real_top1_top2_score_margin"].to_numpy(np.float32))
    entropy_conf = np.clip(
        1.0 - frame[f"{prefix}real_score_entropy"].to_numpy(np.float32),
        0.0,
        1.0,
    )
    spread_conf = 1.0 / (
        1.0 + np.maximum(frame[f"{prefix}real_top3_tvt_std"].to_numpy(np.float32), 0.0) / 25.0
    )
    confidence = (
        0.35 * score_rank + 0.25 * margin_rank + 0.25 * entropy_conf + 0.15 * spread_conf
    ).astype(np.float32)
    row_index = _row_indices_from_ids(frame["id"]).astype(np.float32)
    extra_features[f"{prefix}real_score_top1_rank01"] = score_rank.astype(np.float32)
    extra_features[f"{prefix}real_confidence_proxy"] = confidence
    extra_features[f"{prefix}real_confidence_x_longtail"] = (
        confidence * (row_index >= 1000.0).astype(np.float32)
    ).astype(np.float32)
    frame = pd.concat([frame, pd.DataFrame(extra_features, index=frame.index)], axis=1)
    all_columns.extend(extra_features)
    generated_columns = list(dict.fromkeys(all_columns))
    bad_columns = [
        column
        for column in generated_columns
        if column in frame.columns and not np.isfinite(frame[column].to_numpy(np.float32)).all()
    ]
    if bad_columns:
        raise ValueError(f"heatmap feature columns contain non-finite values: {bad_columns}")
    missing_rate = frame[generated_columns].isna().mean().max()
    if missing_rate > float(heatmap_config.get("max_missing_rate", 0.0)):
        raise ValueError(f"heatmap feature missing_rate={missing_rate:.6f}")
    meta = {
        "enabled": True,
        "source": source_meta,
        "interpolation": interpolation_meta,
        "joined_rows": int(len(frame)),
        "missing_rate_max": float(missing_rate),
        "generated_feature_count": int(len(generated_columns)),
        "generated_feature_columns": generated_columns,
    }
    return frame, generated_columns, meta


def read_heatmap_mdn_candidates(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        EXP202_HEATMAP_CANDIDATES,
        get_nested(config, "data.exp202_heatmap_candidates_local"),
    )
    settings = get_nested(config, "ranker.heatmap_mdn_candidate_features") or {}
    topk_ranks = [int(value) for value in settings.get("topk_ranks", [1, 2, 3, 5, 10])]
    max_rank = max(max(topk_ranks), 10)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    required = {
        "id",
        "well",
        "row_center",
        "last_known_tvt",
        "prior_center_tvt",
        "md_since_prefix",
        "path_step_abs_mean_ft",
        "path_step_abs_max_ft",
        "score_entropy",
        "score_top3_mass",
        "score_top5_mass",
        "top1_top2_score_margin",
        "top1_top3_score_margin",
    }
    for rank in range(1, max_rank + 1):
        required.add(f"pred_top{rank}_tvt")
        required.add(f"pred_top{rank}_score")
    missing = sorted(column for column in required if column not in header)
    if missing:
        raise ValueError(f"{source} is missing heatmap MDN candidate columns: {missing}")
    forbidden = {
        column
        for column in header
        if column.endswith("_abs_error")
        or column.endswith("_within10")
        or column in {"true_center_tvt", "target_in_grid", "best_mode"}
    }
    usecols = sorted(required)
    leaked = sorted(set(usecols).intersection(forbidden))
    if leaked:
        raise ValueError(f"heatmap MDN feature usecols include target-derived columns: {leaked}")
    frame = pd.read_csv(source, usecols=usecols, dtype={"id": str, "well": str}, low_memory=False)
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    if frame.empty:
        raise ValueError(f"heatmap MDN candidate source is empty: {source}")
    meta = {
        "path": str(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": (
            sha256_path(source, decompressed=True) if source.suffix == ".gz" else None
        ),
        "topk_ranks": topk_ranks,
    }
    return frame, meta


def build_heatmap_mdn_source_features(
    source: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    settings = get_nested(config, "ranker.heatmap_mdn_candidate_features") or {}
    prefix = str(settings.get("prefix", "hmdn_"))
    eps = float(settings.get("confidence_eps", 1e-6))
    topk_ranks = [int(value) for value in settings.get("topk_ranks", [1, 2, 3, 5, 10])]
    max_rank = max(max(topk_ranks), 10)
    scores = _positive_score_matrix(source, max_rank=max_rank)
    tvts = _tvt_matrix(source, max_rank=max_rank)
    score_sum = np.maximum(scores.sum(axis=1), eps)
    probs = scores / score_sum[:, None]
    entropy = -np.sum(probs * np.log(np.maximum(probs, eps)), axis=1) / np.log(float(max_rank))
    generated: dict[str, np.ndarray] = {
        f"{prefix}top1_score": scores[:, 0],
        f"{prefix}top1_top2_score_margin": (
            numeric_array(source, "top1_top2_score_margin")
        ).astype(np.float32),
        f"{prefix}top1_top3_score_margin": (
            numeric_array(source, "top1_top3_score_margin")
        ).astype(np.float32),
        f"{prefix}score_entropy": entropy.astype(np.float32),
        f"{prefix}score_top3_mass": (
            numeric_array(source, "score_top3_mass")
        ).astype(np.float32),
        f"{prefix}score_top5_mass": (
            numeric_array(source, "score_top5_mass")
        ).astype(np.float32),
        f"{prefix}path_step_abs_mean_ft": (
            numeric_array(source, "path_step_abs_mean_ft")
        ).astype(np.float32),
        f"{prefix}path_step_abs_max_ft": (
            numeric_array(source, "path_step_abs_max_ft")
        ).astype(np.float32),
        f"{prefix}prior_center_minus_last": (
            numeric_array(source, "prior_center_tvt") - numeric_array(source, "last_known_tvt")
        ).astype(np.float32),
        f"{prefix}md_since_prefix": numeric_array(source, "md_since_prefix"),
    }
    for rank in range(1, max_rank + 1):
        generated[f"{prefix}top{rank}_tvt"] = tvts[:, rank - 1]
        generated[f"{prefix}top{rank}_score"] = scores[:, rank - 1]
    for topk in topk_ranks:
        subset = tvts[:, :topk]
        generated[f"{prefix}top{topk}_tvt_mean"] = np.nanmean(subset, axis=1).astype(np.float32)
        generated[f"{prefix}top{topk}_tvt_std"] = np.nanstd(subset, axis=1).astype(np.float32)
        generated[f"{prefix}top{topk}_tvt_range"] = (
            np.nanmax(subset, axis=1) - np.nanmin(subset, axis=1)
        ).astype(np.float32)
        generated[f"{prefix}top{topk}_score_mass"] = (
            scores[:, :topk].sum(axis=1) / score_sum
        ).astype(np.float32)
    generated[f"{prefix}top1_minus_last"] = (
        generated[f"{prefix}top1_tvt"] - numeric_array(source, "last_known_tvt")
    ).astype(np.float32)
    generated[f"{prefix}top1_minus_prior_center"] = (
        generated[f"{prefix}top1_tvt"] - numeric_array(source, "prior_center_tvt")
    ).astype(np.float32)
    generated[f"{prefix}top3_mean_minus_prior_center"] = (
        generated[f"{prefix}top3_tvt_mean"] - numeric_array(source, "prior_center_tvt")
    ).astype(np.float32)
    out = source[["well", "row_center"]].copy()
    out["row_center"] = pd.to_numeric(out["row_center"], errors="coerce").astype(np.float32)
    generated_columns: list[str] = []
    for column, values in generated.items():
        safe = np.asarray(values, dtype=np.float32)
        if not np.isfinite(safe).all():
            finite = safe[np.isfinite(safe)]
            fill_value = float(np.nanmedian(finite)) if len(finite) else 0.0
            safe = np.where(np.isfinite(safe), safe, fill_value).astype(np.float32)
        out[column] = safe
        generated_columns.append(column)
    out = out.groupby(["well", "row_center"], as_index=False, observed=True).mean(numeric_only=True)
    return out, generated_columns


def add_heatmap_mdn_candidate_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
    candidates: list[CandidateSpec],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    settings = get_nested(config, "ranker.heatmap_mdn_candidate_features") or {}
    if not settings.get("enabled", False):
        return frame, [], {"enabled": False}
    prefix = str(settings.get("prefix", "hmdn_"))
    source, source_meta = read_heatmap_mdn_candidates(config)
    source_features, source_columns = build_heatmap_mdn_source_features(source, config)
    aligned, interpolation_meta = interpolate_heatmap_features_for_frame(
        frame,
        source_features,
        source_columns,
        prefix=prefix,
    )
    join_columns = [
        "id",
        *source_columns,
        f"{prefix}source_valid",
        f"{prefix}row_coverage_left",
        f"{prefix}row_coverage_right",
        f"{prefix}nearest_sample_distance_rows",
    ]
    before_rows = len(frame)
    frame = frame.merge(aligned[join_columns], on="id", how="left", validate="one_to_one")
    if len(frame) != before_rows:
        raise ValueError("heatmap MDN feature join changed row count")
    candidate_matrix = np.column_stack(
        [
            pd.to_numeric(frame[spec.column], errors="coerce").to_numpy(np.float32)
            for spec in candidates
            if spec.column in frame.columns
        ]
    )
    candidate_mean = np.nanmean(candidate_matrix, axis=1).astype(np.float32)
    top1 = frame[f"{prefix}top1_tvt"].to_numpy(np.float32)
    top3_mean = frame[f"{prefix}top3_tvt_mean"].to_numpy(np.float32)
    extra_features: dict[str, np.ndarray] = {}
    for ref in ["likpf_mean", "beam_mean", "pf_ancc", "tvt_densew"]:
        if ref in frame.columns:
            ref_values = frame[ref].to_numpy(np.float32)
            extra_features[f"{prefix}top1_minus_{ref}"] = (top1 - ref_values).astype(np.float32)
            extra_features[f"{prefix}top3_mean_minus_{ref}"] = (
                top3_mean - ref_values
            ).astype(np.float32)
    extra_features[f"{prefix}top1_vs_candidate_mean_abs"] = np.abs(
        top1 - candidate_mean
    ).astype(np.float32)
    extra_features[f"{prefix}top3_mean_vs_candidate_mean_abs"] = np.abs(
        top3_mean - candidate_mean
    ).astype(np.float32)
    for topk in [3, 5, 10]:
        columns = [
            f"{prefix}top{rank}_tvt"
            for rank in range(1, topk + 1)
            if f"{prefix}top{rank}_tvt" in frame.columns
        ]
        if len(columns) != topk:
            continue
        values = frame[columns].to_numpy(np.float32)
        min_abs = np.nanmin(np.abs(values[:, :, None] - candidate_matrix[:, None, :]), axis=(1, 2))
        min_abs = np.where(np.isfinite(min_abs), min_abs, 1_000_000.0).astype(np.float32)
        extra_features[f"{prefix}top{topk}_vs_candidate_min_abs"] = min_abs
        extra_features[f"{prefix}top{topk}_any_candidate_within10_proxy"] = (
            min_abs <= 10.0
        ).astype(np.float32)
    score_rank = _rank01(frame[f"{prefix}top1_score"].to_numpy(np.float32))
    margin_rank = _rank01(frame[f"{prefix}top1_top2_score_margin"].to_numpy(np.float32))
    entropy_conf = np.clip(1.0 - frame[f"{prefix}score_entropy"].to_numpy(np.float32), 0.0, 1.0)
    spread_conf = 1.0 / (
        1.0 + np.maximum(frame[f"{prefix}top10_tvt_std"].to_numpy(np.float32), 0.0) / 25.0
    )
    confidence = (
        0.35 * score_rank + 0.25 * margin_rank + 0.20 * entropy_conf + 0.20 * spread_conf
    ).astype(np.float32)
    row_index = _row_indices_from_ids(frame["id"]).astype(np.float32)
    extra_features[f"{prefix}score_top1_rank01"] = score_rank.astype(np.float32)
    extra_features[f"{prefix}confidence_proxy"] = confidence
    extra_features[f"{prefix}confidence_x_longtail"] = (
        confidence * (row_index >= 1000.0).astype(np.float32)
    ).astype(np.float32)
    frame = pd.concat([frame, pd.DataFrame(extra_features, index=frame.index)], axis=1)
    generated_columns = list(
        dict.fromkeys(
            [
                column
                for column in join_columns
                if column != "id"
            ]
            + list(extra_features)
        )
    )
    bad_columns = [
        column
        for column in generated_columns
        if column in frame.columns and not np.isfinite(frame[column].to_numpy(np.float32)).all()
    ]
    if bad_columns:
        raise ValueError(f"heatmap MDN feature columns contain non-finite values: {bad_columns}")
    missing_rate = frame[generated_columns].isna().mean().max()
    if missing_rate > float(settings.get("max_missing_rate", 0.0)):
        raise ValueError(f"heatmap MDN feature missing_rate={missing_rate:.6f}")
    meta = {
        "enabled": True,
        "source": source_meta,
        "interpolation": interpolation_meta,
        "joined_rows": int(len(frame)),
        "missing_rate_max": float(missing_rate),
        "generated_feature_count": int(len(generated_columns)),
        "generated_feature_columns": generated_columns,
    }
    return frame, generated_columns, meta


def summarize_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "feature": column,
                "missing_rate": float(np.mean(~np.isfinite(values))) if len(values) else 0.0,
                "mean": float(np.mean(finite)) if len(finite) else np.nan,
                "std": float(np.std(finite)) if len(finite) else np.nan,
                "p05": float(np.quantile(finite, 0.05)) if len(finite) else np.nan,
                "p50": float(np.quantile(finite, 0.50)) if len(finite) else np.nan,
                "p95": float(np.quantile(finite, 0.95)) if len(finite) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_candidate_labels_and_features(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    *,
    config: dict[str, Any],
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
    extra_features: dict[str, np.ndarray] = {}
    last_known = out["last_known_tvt"].to_numpy(np.float32)
    for spec in candidates:
        delta_col = candidate_minus_last_feature_name(spec, out.columns)
        extra_features[delta_col] = (
            out[spec.column].to_numpy(np.float32) - last_known
        ).astype(np.float32)
        feature_columns.append(delta_col)
        if include_candidate_values:
            feature_columns.append(spec.column)

    pairwise_candidates = pairwise_candidate_specs(candidates, config)
    for i, left in enumerate(pairwise_candidates):
        for right in pairwise_candidates[i + 1 :]:
            col = f"{left.name}_vs_{right.name}_abs"
            extra_features[col] = np.abs(
                out[left.column].to_numpy(np.float32) - out[right.column].to_numpy(np.float32)
            )
            feature_columns.append(col)

    extra_features["candidate_mean"] = candidate_values.mean(axis=1).astype(np.float32)
    extra_features["candidate_std"] = candidate_values.std(axis=1).astype(np.float32)
    extra_features["candidate_range"] = (
        candidate_values.max(axis=1) - candidate_values.min(axis=1)
    ).astype(
        np.float32
    )
    feature_columns.extend(["candidate_mean", "candidate_std", "candidate_range"])
    out = pd.concat([out, pd.DataFrame(extra_features, index=out.index)], axis=1)
    assert_unique_columns(out, "candidate label feature frame")
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
            + (get_nested(config, "ranker.heatmap_path_features.base_feature_columns") or [])
            + (
                get_nested(config, "ranker.heatmap_mdn_candidate_features.base_feature_columns")
                or []
            )
            + (get_nested(config, "ranker.cluster_prior_features.base_feature_columns") or [])
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
    train_values = train[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    medians = np.nanmedian(train_values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    train_bad = ~np.isfinite(train_values)
    if train_bad.any():
        train_values[train_bad] = np.take(medians, np.where(train_bad)[1])
    valid_values = transform_impute_values(valid, columns, medians)
    return train_values, valid_values


def fit_impute_train_values(
    train: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    train_values = train[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    medians = np.nanmedian(train_values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    train_bad = ~np.isfinite(train_values)
    if train_bad.any():
        train_values[train_bad] = np.take(medians, np.where(train_bad)[1])
    return train_values, medians


def transform_impute_values(
    frame: pd.DataFrame, columns: list[str], medians: np.ndarray
) -> np.ndarray:
    values = frame[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    valid_bad = ~np.isfinite(values)
    if valid_bad.any():
        values[valid_bad] = np.take(medians, np.where(valid_bad)[1])
    return values


def build_long_frame(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    candidates: list[CandidateSpec],
    *,
    row_feature_columns: list[str],
    candidate_values: np.ndarray,
    oracle_labels: np.ndarray,
    sample_rows: int | None,
    seed: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    row_indices = sample_row_indices(row_indices, sample_rows, seed=seed)
    chunks: list[pd.DataFrame] = []
    y_error_chunks: list[np.ndarray] = []
    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    cluster_settings = get_nested(config, "ranker.cluster_prior_features") or {}
    cluster_enabled = bool(cluster_settings.get("enabled", False))
    priors = parse_prior_specs(config) if cluster_enabled else []
    gates = parse_cluster_gates(config) if cluster_enabled else []
    alpha = float(cluster_settings.get("feature_correction_alpha", 0.2))
    clips = [
        float(value)
        for value in cluster_settings.get("feature_correction_clips", [20.0, 40.0])
    ]
    quality_std = float(cluster_settings.get("feature_quality_max_prior_std", 20.0))
    family_groups = {
        "pfbeam": {"pf_ancc", "beam_mean", "likpf_mean", "sc_ens", "hyb"},
        "dense": {"tvt_dense", "tvt_densew", "tvt_dense50"},
    }
    heatmap_settings = get_nested(config, "ranker.heatmap_path_features") or {}
    heatmap_enabled = bool(heatmap_settings.get("enabled", False))
    heatmap_prefix = str(heatmap_settings.get("prefix", "hmpf_"))
    heatmap_scale_floor = float(heatmap_settings.get("min_candidate_scale_ft", 10.0))
    for cand_idx, spec in enumerate(candidates):
        part = frame.iloc[row_indices][["id", "well", *row_feature_columns]].copy()
        source_rows = frame.iloc[row_indices]
        candidate_tvt = candidate_values[row_indices, cand_idx].astype(np.float32)
        part["candidate_index"] = np.int16(cand_idx)
        part["candidate_name_code"] = np.int16(cand_idx)
        part["candidate_tvt"] = candidate_tvt
        part["candidate_minus_last"] = (
            candidate_tvt
            - frame["last_known_tvt"].to_numpy(np.float32)[row_indices]
        ).astype(np.float32)
        hmdn_rank = hmdn_candidate_rank(spec.name)
        part["candidate_is_default_likpf"] = np.float32(spec.name == "likpf_mean")
        part["candidate_is_pfbeam_family"] = np.float32(spec.name in family_groups["pfbeam"])
        part["candidate_is_dense_family"] = np.float32(spec.name in family_groups["dense"])
        part["candidate_is_hmdn_family"] = np.float32(hmdn_rank is not None)
        part["candidate_hmdn_rank"] = np.float32(hmdn_rank if hmdn_rank is not None else 0)
        score_col = f"multiobs_score_{spec.name}"
        mae_col = f"multiobs_mae_{spec.name}"
        ncc_col = f"multiobs_ncc_{spec.name}"
        part["candidate_multiobs_score"] = (
            frame.iloc[row_indices][score_col].to_numpy(np.float32)
            if score_col in frame.columns
            else 0.0
        )
        part["candidate_multiobs_mae"] = (
            frame.iloc[row_indices][mae_col].to_numpy(np.float32)
            if mae_col in frame.columns
            else 0.0
        )
        part["candidate_multiobs_ncc"] = (
            frame.iloc[row_indices][ncc_col].to_numpy(np.float32)
            if ncc_col in frame.columns
            else 0.0
        )
        part["is_oracle"] = (oracle_labels[row_indices] == cand_idx).astype(np.int8)
        if cluster_enabled:
            for prior in priors:
                prior_col = prefixed_prior_column(prior, prior.prior_tvt)
                if prior_col is None or prior_col not in source_rows.columns:
                    continue
                prior_tvt = pd.to_numeric(source_rows[prior_col], errors="coerce").to_numpy(
                    np.float32
                )
                delta = (prior_tvt - candidate_tvt).astype(np.float32)
                valid_prior = np.isfinite(prior_tvt)
                std_col = prefixed_prior_column(prior, prior.prior_std)
                std = (
                    pd.to_numeric(source_rows[std_col], errors="coerce").to_numpy(np.float32)
                    if std_col is not None and std_col in source_rows.columns
                    else np.full(len(source_rows), np.nan, dtype=np.float32)
                )
                count_col = prefixed_prior_column(prior, prior.prior_count)
                count = (
                    pd.to_numeric(source_rows[count_col], errors="coerce").to_numpy(np.float32)
                    if count_col is not None and count_col in source_rows.columns
                    else np.full(len(source_rows), np.nan, dtype=np.float32)
                )
                neighbor_col = prefixed_prior_column(prior, prior.neighbor_wells)
                neighbor_wells = (
                    pd.to_numeric(source_rows[neighbor_col], errors="coerce").to_numpy(np.float32)
                    if neighbor_col is not None and neighbor_col in source_rows.columns
                    else np.full(len(source_rows), np.nan, dtype=np.float32)
                )
                std_scale = np.maximum(
                    np.where(np.isfinite(std), np.abs(std), quality_std),
                    float(cluster_settings.get("min_prior_scale_ft", 10.0)),
                ).astype(np.float32)
                safe_delta = np.where(np.isfinite(delta), delta, 0.0).astype(np.float32)
                part[f"copcf_{prior.name}_minus_candidate"] = safe_delta
                part[f"copcf_{prior.name}_minus_candidate_abs"] = np.abs(safe_delta)
                part[f"copcf_{prior.name}_minus_candidate_abs_norm"] = (
                    np.abs(safe_delta) / std_scale
                ).astype(np.float32)
                part[f"copcf_{prior.name}_valid_x_candidate"] = valid_prior.astype(np.float32)
                part[f"copcf_{prior.name}_std_x_candidate"] = np.where(
                    np.isfinite(std), std, np.nan
                ).astype(np.float32)
                part[f"copcf_{prior.name}_count_x_candidate"] = np.where(
                    np.isfinite(count), count, np.nan
                ).astype(np.float32)
                part[f"copcf_{prior.name}_neighbor_wells_x_candidate"] = np.where(
                    np.isfinite(neighbor_wells), neighbor_wells, np.nan
                ).astype(np.float32)
                quality_mask = valid_prior & np.isfinite(std) & (std <= quality_std)
                for gate in gates:
                    gate_col = f"copcf_gate_{gate.name}"
                    if gate_col not in source_rows.columns:
                        continue
                    gate_mask = source_rows[gate_col].fillna(0.0).to_numpy(np.float32) > 0.5
                    active = quality_mask & gate_mask
                    part[f"copcf_{prior.name}_{gate.name}_gate_x_candidate"] = active.astype(
                        np.float32
                    )
                    family_value = 1.0 if spec.name in family_groups["dense"] else 0.0
                    part[f"copcf_{prior.name}_{gate.name}_gate_x_dense_family"] = (
                        active.astype(np.float32) * np.float32(family_value)
                    )
                    for clip in clips:
                        clip_tag = f"c{str(clip).replace('.', 'p').rstrip('0').rstrip('p')}"
                        correction = np.zeros(len(source_rows), dtype=np.float32)
                        correction[active] = (
                            alpha * np.clip(safe_delta[active], -clip, clip)
                        ).astype(np.float32)
                        part[f"copcf_{prior.name}_{gate.name}_corr_abs_{clip_tag}"] = np.abs(
                            correction
                        ).astype(np.float32)
                        part[f"copcf_{prior.name}_{gate.name}_clip_hit_{clip_tag}"] = (
                            active & (np.abs(safe_delta) > clip)
                        ).astype(np.float32)
        if heatmap_enabled and f"{heatmap_prefix}real_top1_tvt" in source_rows.columns:
            candidate_std = np.maximum(
                pd.to_numeric(source_rows.get("candidate_std", 0.0), errors="coerce")
                .fillna(0.0)
                .to_numpy(np.float32),
                heatmap_scale_floor,
            )
            for source_name, column in [
                ("top1", f"{heatmap_prefix}real_top1_tvt"),
                ("top3_mean", f"{heatmap_prefix}real_top3_tvt_mean"),
                ("top5_mean", f"{heatmap_prefix}real_top5_tvt_mean"),
                ("top10_mean", f"{heatmap_prefix}real_top10_tvt_mean"),
            ]:
                if column not in source_rows.columns:
                    continue
                delta = source_rows[column].to_numpy(np.float32) - candidate_tvt
                part[f"{heatmap_prefix}real_{source_name}_minus_candidate"] = delta.astype(
                    np.float32
                )
                part[f"{heatmap_prefix}real_{source_name}_minus_candidate_abs"] = np.abs(
                    delta
                ).astype(np.float32)
                part[f"{heatmap_prefix}real_{source_name}_minus_candidate_abs_norm"] = (
                    np.abs(delta) / candidate_std
                ).astype(np.float32)
            for topk in [3, 5, 10]:
                columns = [
                    f"{heatmap_prefix}real_top{rank}_tvt"
                    for rank in range(1, topk + 1)
                    if f"{heatmap_prefix}real_top{rank}_tvt" in source_rows.columns
                ]
                if len(columns) != topk:
                    continue
                values = source_rows[columns].to_numpy(np.float32)
                min_abs = np.nanmin(np.abs(values - candidate_tvt[:, None]), axis=1)
                min_abs = np.where(np.isfinite(min_abs), min_abs, 1_000_000.0).astype(
                    np.float32
                )
                part[f"{heatmap_prefix}real_top{topk}_candidate_min_abs"] = min_abs
                part[f"{heatmap_prefix}real_top{topk}_candidate_min_abs_norm"] = (
                    min_abs / candidate_std
                ).astype(np.float32)
                part[f"{heatmap_prefix}real_top{topk}_candidate_within10_proxy"] = (
                    min_abs <= 10.0
                ).astype(np.float32)
            if f"{heatmap_prefix}real_confidence_proxy" in source_rows.columns:
                confidence = source_rows[f"{heatmap_prefix}real_confidence_proxy"].to_numpy(
                    np.float32
                )
                family_value = np.float32(1.0 if spec.name in family_groups["pfbeam"] else 0.0)
                dense_value = np.float32(1.0 if spec.name in family_groups["dense"] else 0.0)
                part[f"{heatmap_prefix}real_confidence_x_candidate_pfbeam_family"] = (
                    confidence * family_value
                ).astype(np.float32)
                part[f"{heatmap_prefix}real_confidence_x_candidate_dense_family"] = (
                    confidence * dense_value
                ).astype(np.float32)
        mdn_settings = get_nested(config, "ranker.heatmap_mdn_candidate_features") or {}
        mdn_enabled = bool(mdn_settings.get("enabled", False))
        mdn_prefix = str(mdn_settings.get("prefix", "hmdn_"))
        mdn_scale_floor = float(mdn_settings.get("min_candidate_scale_ft", 10.0))
        if mdn_enabled and f"{mdn_prefix}top1_tvt" in source_rows.columns:
            candidate_std = np.maximum(
                pd.to_numeric(source_rows.get("candidate_std", 0.0), errors="coerce")
                .fillna(0.0)
                .to_numpy(np.float32),
                mdn_scale_floor,
            )
            for source_name, column in [
                ("top1", f"{mdn_prefix}top1_tvt"),
                ("top3_mean", f"{mdn_prefix}top3_tvt_mean"),
                ("top5_mean", f"{mdn_prefix}top5_tvt_mean"),
                ("top10_mean", f"{mdn_prefix}top10_tvt_mean"),
            ]:
                if column not in source_rows.columns:
                    continue
                delta = source_rows[column].to_numpy(np.float32) - candidate_tvt
                part[f"{mdn_prefix}{source_name}_minus_candidate"] = delta.astype(np.float32)
                part[f"{mdn_prefix}{source_name}_minus_candidate_abs"] = np.abs(delta).astype(
                    np.float32
                )
                part[f"{mdn_prefix}{source_name}_minus_candidate_abs_norm"] = (
                    np.abs(delta) / candidate_std
                ).astype(np.float32)
            for topk in [3, 5, 10]:
                columns = [
                    f"{mdn_prefix}top{rank}_tvt"
                    for rank in range(1, topk + 1)
                    if f"{mdn_prefix}top{rank}_tvt" in source_rows.columns
                ]
                score_columns = [
                    f"{mdn_prefix}top{rank}_score"
                    for rank in range(1, topk + 1)
                    if f"{mdn_prefix}top{rank}_score" in source_rows.columns
                ]
                if len(columns) != topk:
                    continue
                values = source_rows[columns].to_numpy(np.float32)
                abs_delta = np.abs(values - candidate_tvt[:, None])
                min_abs = np.nanmin(abs_delta, axis=1)
                min_abs = np.where(np.isfinite(min_abs), min_abs, 1_000_000.0).astype(
                    np.float32
                )
                part[f"{mdn_prefix}top{topk}_candidate_min_abs"] = min_abs
                part[f"{mdn_prefix}top{topk}_candidate_min_abs_norm"] = (
                    min_abs / candidate_std
                ).astype(np.float32)
                part[f"{mdn_prefix}top{topk}_candidate_within10_proxy"] = (
                    min_abs <= 10.0
                ).astype(np.float32)
                if len(score_columns) == topk:
                    scores = source_rows[score_columns].to_numpy(np.float32)
                    best_rank = np.nanargmin(abs_delta, axis=1)
                    part[f"{mdn_prefix}top{topk}_nearest_candidate_score"] = scores[
                        np.arange(len(scores)), best_rank
                    ].astype(np.float32)
            if f"{mdn_prefix}confidence_proxy" in source_rows.columns:
                confidence = source_rows[f"{mdn_prefix}confidence_proxy"].to_numpy(np.float32)
                family_value = np.float32(1.0 if spec.name in family_groups["pfbeam"] else 0.0)
                dense_value = np.float32(1.0 if spec.name in family_groups["dense"] else 0.0)
                hmdn_value = np.float32(1.0 if hmdn_rank is not None else 0.0)
                part[f"{mdn_prefix}confidence_x_candidate_pfbeam_family"] = (
                    confidence * family_value
                ).astype(np.float32)
                part[f"{mdn_prefix}confidence_x_candidate_dense_family"] = (
                    confidence * dense_value
                ).astype(np.float32)
                part[f"{mdn_prefix}confidence_x_candidate_hmdn_family"] = (
                    confidence * hmdn_value
                ).astype(np.float32)
            if hmdn_rank is not None:
                score_col = f"{mdn_prefix}top{hmdn_rank}_score"
                part[f"{mdn_prefix}candidate_rank01"] = np.float32(
                    min(max(hmdn_rank, 1), 10) / 10.0
                )
                part[f"{mdn_prefix}candidate_score"] = (
                    source_rows[score_col].to_numpy(np.float32)
                    if score_col in source_rows.columns
                    else np.zeros(len(source_rows), dtype=np.float32)
                )
            else:
                part[f"{mdn_prefix}candidate_rank01"] = np.float32(0.0)
                part[f"{mdn_prefix}candidate_score"] = np.float32(0.0)
        y_error_chunks.append(
            np.abs(candidate_tvt - true_tvt[row_indices])
        )
        chunks.append(part)
    long_frame = pd.concat(chunks, ignore_index=True)
    y_error = np.concatenate(y_error_chunks).astype(np.float32)
    return long_frame, y_error


def predict_long_model_in_chunks(
    *,
    model: Any,
    prediction_kind: str,
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    candidates: list[CandidateSpec],
    row_feature_columns: list[str],
    candidate_values: np.ndarray,
    oracle_labels: np.ndarray,
    config: dict[str, Any],
    long_feature_columns: list[str],
    medians: np.ndarray,
    chunk_rows: int,
) -> np.ndarray:
    scores = np.zeros((len(row_indices), len(candidates)), dtype=np.float32)
    for start in range(0, len(row_indices), chunk_rows):
        end = min(start + chunk_rows, len(row_indices))
        chunk_indices = row_indices[start:end]
        long_chunk, _ = build_long_frame(
            frame,
            chunk_indices,
            candidates,
            row_feature_columns=row_feature_columns,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            sample_rows=None,
            seed=0,
            config=config,
        )
        x_chunk = transform_impute_values(long_chunk, long_feature_columns, medians)
        if prediction_kind == "binary":
            chunk_pred = model.predict_proba(x_chunk)[:, 1]
        elif prediction_kind == "error":
            chunk_pred = model.predict(x_chunk)
        else:
            raise ValueError(f"unknown prediction_kind: {prediction_kind}")
        scores[start:end] = np.asarray(chunk_pred, dtype=np.float32).reshape(
            len(candidates), len(chunk_indices)
        ).T
        del long_chunk, x_chunk, chunk_pred
        gc.collect()
    return scores


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
        ("hmdn_confidence_proxy", "hmdn_confidence_bucket"),
        ("hmdn_nearest_sample_distance_rows", "hmdn_sparse_distance_bucket"),
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


def read_exp115_roles(config: dict[str, Any]) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    try:
        source = find_artifact(
            EXP115_FOLD_ASSIGNMENTS,
            get_nested(config, "data.exp115_fold_assignments_local"),
        )
    except FileNotFoundError:
        return None, {"loaded": False, "path": None}
    frame = pd.read_csv(source, dtype=str)
    required = {
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    frame = frame[list(required)].rename(columns={"well_id": "well"})
    meta = {
        "loaded": True,
        "path": str(source),
        "sha256": sha256_path(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
    }
    return frame, meta


def subgroup_context(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    context = frame[["id", "well"]].copy()
    meta: dict[str, Any] = {}
    roles, roles_meta = read_exp115_roles(config)
    meta["exp115_roles"] = roles_meta
    if roles is not None:
        context = context.merge(roles, on="well", how="left", validate="many_to_one")
    for column in [
        "copcf_gate_any_outlier_signal_k8",
        "copcf_gate_own_z_gt2p0",
        "copcf_nearest_other_closer",
        "copcf_nearby_majority_diff_k8",
        "hmpf_source_valid",
        "hmpf_real_confidence_proxy",
        "hmpf_nearest_sample_distance_rows",
        "hmdn_source_valid",
        "hmdn_confidence_proxy",
        "hmdn_nearest_sample_distance_rows",
    ]:
        if column in frame.columns:
            context[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return context, meta


def subgroup_metrics(
    predictions: pd.DataFrame, source_frame: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    context, _meta = subgroup_context(source_frame, config)
    merged = predictions.merge(context, on=["id", "well"], how="left", validate="many_to_one")
    masks: dict[str, np.ndarray] = {}
    if "verification_like_spatial_role" in merged.columns:
        masks["exp115_spatial_valid"] = (
            merged["verification_like_spatial_role"].eq("valid").to_numpy()
        )
    if "verification_like_typewell_purged_role" in merged.columns:
        masks["exp115_typewell_purged_valid"] = merged[
            "verification_like_typewell_purged_role"
        ].eq("valid").to_numpy()
    for column in [
        "copcf_gate_any_outlier_signal_k8",
        "copcf_gate_own_z_gt2p0",
        "copcf_nearest_other_closer",
        "copcf_nearby_majority_diff_k8",
    ]:
        if column in merged.columns:
            masks[column] = (
                pd.to_numeric(merged[column], errors="coerce").fillna(0.0).to_numpy() > 0.5
            )
    if "hmpf_real_confidence_proxy" in merged.columns:
        confidence = pd.to_numeric(
            merged["hmpf_real_confidence_proxy"], errors="coerce"
        ).fillna(0.0)
        masks["hmpf_confidence_bottom_quartile"] = (
            confidence.to_numpy() <= float(confidence.quantile(0.25))
        )
    if "hmpf_nearest_sample_distance_rows" in merged.columns:
        distance = pd.to_numeric(
            merged["hmpf_nearest_sample_distance_rows"], errors="coerce"
        ).fillna(1_000_000.0)
        masks["hmpf_far_from_sparse_sample_gt512"] = distance.to_numpy() > 512.0
    if "hmdn_confidence_proxy" in merged.columns:
        confidence = pd.to_numeric(merged["hmdn_confidence_proxy"], errors="coerce").fillna(0.0)
        masks["hmdn_confidence_bottom_quartile"] = (
            confidence.to_numpy() <= float(confidence.quantile(0.25))
        )
    if "hmdn_nearest_sample_distance_rows" in merged.columns:
        distance = pd.to_numeric(
            merged["hmdn_nearest_sample_distance_rows"], errors="coerce"
        ).fillna(1_000_000.0)
        masks["hmdn_far_from_sparse_sample_gt512"] = distance.to_numpy() > 512.0
    rows: list[dict[str, Any]] = []
    grouped = merged.groupby(["variant", "mode"], observed=True).groups
    for (variant, mode), group_idx in grouped.items():
        group_positions = np.asarray(group_idx, dtype=np.int64)
        for subgroup, mask in masks.items():
            positions = group_positions[mask[group_positions]]
            if len(positions) == 0:
                continue
            group = merged.iloc[positions]
            rows.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "subgroup": subgroup,
                    "rows": int(len(group)),
                    "wells": int(group["well"].nunique()),
                    "rmse_tvt": rmse(
                        group["true_tvt"].to_numpy(), group["selected_tvt"].to_numpy()
                    ),
                    "mae_tvt": mae(group["true_tvt"].to_numpy(), group["selected_tvt"].to_numpy()),
                    "within_10ft": float(np.mean(group["abs_error"].to_numpy() <= 10.0)),
                }
            )
    return pd.DataFrame(rows).sort_values(["variant", "mode", "subgroup"])


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, np.ndarray]]:
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

    seed = int(get_nested(config, "validation.seed") or 42)
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    log_period = int(get_nested(config, "ranker.log_period") or 100)
    max_train_rows = get_nested(config, "ranker.long_models.max_train_rows_per_fold")
    max_train_rows = int(max_train_rows) if max_train_rows is not None else None
    max_valid_rows = get_nested(config, "ranker.long_models.max_valid_rows_per_fold")
    max_valid_rows = int(max_valid_rows) if max_valid_rows is not None else None
    predict_chunk_rows = int(
        get_nested(config, "ranker.long_models.predict_chunk_rows") or 50_000
    )
    max_mc_train_rows = get_nested(config, "ranker.multiclass_lgbm.max_train_rows_per_fold")
    max_mc_train_rows = int(max_mc_train_rows) if max_mc_train_rows is not None else None
    max_mc_valid_rows = get_nested(config, "ranker.multiclass_lgbm.max_valid_rows_per_fold")
    max_mc_valid_rows = int(max_mc_valid_rows) if max_mc_valid_rows is not None else None
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    candidate_names = [spec.name for spec in candidates]

    cv = GroupKFold(n_splits=n_folds)
    oof_selected: dict[str, np.ndarray] = {
        "lgb_multiclass": np.zeros(len(frame), dtype=np.int16),
        "lgb_candidate_binary": np.zeros(len(frame), dtype=np.int16),
        "lgb_candidate_error_ranker": np.zeros(len(frame), dtype=np.int16),
    }
    oof_scores = {
        "multiclass_proba": np.zeros((len(frame), len(candidates)), dtype=np.float32),
        "binary_proba": np.zeros((len(frame), len(candidates)), dtype=np.float32),
        "predicted_error": np.zeros((len(frame), len(candidates)), dtype=np.float32),
    }
    importance_rows: list[dict[str, Any]] = []
    model_manifest: list[dict[str, Any]] = []
    folds = list(cv.split(frame, oracle_labels, groups=frame["well"]))
    multiclass_params = dict(get_nested(config, "ranker.multiclass_lgbm.params") or {})
    binary_params = dict(get_nested(config, "ranker.long_models.binary_lgbm.params") or {})
    error_params = dict(get_nested(config, "ranker.long_models.error_lgbm.params") or {})

    for fold, (train_idx, valid_idx) in enumerate(folds):
        print(f"[fold {fold}] train={len(train_idx)} valid={len(valid_idx)}", flush=True)
        mc_train_idx = sample_row_indices(
            train_idx,
            max_mc_train_rows,
            seed=seed + 17 * fold,
        )
        mc_eval_idx = sample_row_indices(
            valid_idx,
            max_mc_valid_rows,
            seed=seed + 19 * fold,
        )
        if len(mc_train_idx) != len(train_idx) or len(mc_eval_idx) != len(valid_idx):
            print(
                "[fold "
                f"{fold}] multiclass sampled train={len(mc_train_idx)}/{len(train_idx)} "
                f"eval={len(mc_eval_idx)}/{len(valid_idx)}",
                flush=True,
            )
        train_frame = frame.iloc[mc_train_idx]
        eval_frame = frame.iloc[mc_eval_idx]
        x_train, multiclass_medians = fit_impute_train_values(train_frame, feature_columns)
        x_eval = transform_impute_values(eval_frame, feature_columns, multiclass_medians)
        y_train = oracle_labels[mc_train_idx]
        y_eval = oracle_labels[mc_eval_idx]

        multiclass = LGBMClassifier(
            objective="multiclass",
            num_class=len(candidates),
            random_state=seed + fold,
            **multiclass_params,
        )
        multiclass.fit(
            x_train,
            y_train,
            eval_set=[(x_eval, y_eval)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(50), log_evaluation(log_period)],
        )
        x_valid = transform_impute_values(
            frame.iloc[valid_idx],
            feature_columns,
            multiclass_medians,
        )
        multiclass_proba = multiclass.predict_proba(x_valid)
        oof_scores["multiclass_proba"][valid_idx] = np.asarray(
            multiclass_proba, dtype=np.float32
        )
        oof_selected["lgb_multiclass"][valid_idx] = np.argmax(multiclass_proba, axis=1).astype(
            np.int16
        )
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
        del x_train, x_eval, x_valid, multiclass_proba, train_frame, eval_frame
        gc.collect()

        row_features = feature_columns
        long_train, train_error = build_long_frame(
            frame,
            train_idx,
            candidates,
            row_feature_columns=row_features,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            sample_rows=max_train_rows,
            seed=seed + 101 * fold,
            config=config,
        )
        long_eval, eval_error = build_long_frame(
            frame,
            valid_idx,
            candidates,
            row_feature_columns=row_features,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            sample_rows=max_valid_rows,
            seed=seed + 1001 * fold,
            config=config,
        )
        long_feature_columns = [
            col
            for col in long_train.columns
            if col not in {"id", "well", "is_oracle"}
            and pd.api.types.is_numeric_dtype(long_train[col])
        ]
        x_long_train, long_medians = fit_impute_train_values(long_train, long_feature_columns)
        x_long_eval = transform_impute_values(long_eval, long_feature_columns, long_medians)
        y_bin_train = long_train["is_oracle"].to_numpy(np.int8)
        y_bin_eval = long_eval["is_oracle"].to_numpy(np.int8)

        binary = LGBMClassifier(
            objective="binary",
            random_state=seed + 1000 + fold,
            **binary_params,
        )
        binary.fit(
            x_long_train,
            y_bin_train,
            eval_set=[(x_long_eval, y_bin_eval)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(50), log_evaluation(log_period)],
        )
        binary_score = predict_long_model_in_chunks(
            model=binary,
            prediction_kind="binary",
            frame=frame,
            row_indices=valid_idx,
            candidates=candidates,
            row_feature_columns=row_features,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            config=config,
            long_feature_columns=long_feature_columns,
            medians=long_medians,
            chunk_rows=predict_chunk_rows,
        )
        oof_scores["binary_proba"][valid_idx] = np.asarray(binary_score, dtype=np.float32)
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
        error_ranker.fit(
            x_long_train,
            train_error,
            eval_set=[(x_long_eval, eval_error)],
            eval_metric="l1",
            callbacks=[early_stopping(50), log_evaluation(log_period)],
        )
        pred_error = predict_long_model_in_chunks(
            model=error_ranker,
            prediction_kind="error",
            frame=frame,
            row_indices=valid_idx,
            candidates=candidates,
            row_feature_columns=row_features,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            config=config,
            long_feature_columns=long_feature_columns,
            medians=long_medians,
            chunk_rows=predict_chunk_rows,
        )
        oof_scores["predicted_error"][valid_idx] = np.asarray(pred_error, dtype=np.float32)
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
            train_error,
            long_eval,
            eval_error,
            x_long_train,
            x_long_eval,
            y_bin_train,
            y_bin_eval,
            binary_score,
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
    return metrics, predictions, importance, model_manifest_meta, oof_scores


def second_margin_low(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(values, axis=1)
    top1 = order[:, 0].astype(np.int16)
    top2 = order[:, 1].astype(np.int16) if values.shape[1] > 1 else top1.copy()
    margin = (values[np.arange(len(values)), top2] - values[np.arange(len(values)), top1]).astype(
        np.float32
    )
    return top1, margin, values[np.arange(len(values)), top1].astype(np.float32)


def categorical_codes(values: pd.Series | pd.Categorical) -> tuple[np.ndarray, list[str]]:
    if isinstance(values, pd.Series):
        categorical = values.astype("category")
        return (
            categorical.cat.codes.to_numpy(np.int16),
            [str(label) for label in categorical.cat.categories],
        )
    return values.codes.astype(np.int16), [str(label) for label in values.categories]


def distance_bucket_codes(values: pd.Series | np.ndarray) -> tuple[np.ndarray, list[str]]:
    return categorical_codes(_distance_bucket(values))


def tail_rank_bucket_codes(ids: pd.Series) -> tuple[np.ndarray, list[str]]:
    return categorical_codes(_tail_rank_bucket(ids))


def quantile_bucket_codes(
    values: pd.Series | np.ndarray, prefix: str
) -> tuple[np.ndarray, list[str]]:
    return categorical_codes(_quantile_bucket(values, prefix))


def variant_specs_from_config(config: dict[str, Any]) -> list[ViterbiSpec]:
    values = get_nested(config, "selector.viterbi_grid") or {}
    specs: list[ViterbiSpec] = []
    for switch_penalty in values.get("switch_penalty", [0.0]):
        for nondefault_bias in values.get("nondefault_bias", [0.0]):
            for jump_penalty_weight in values.get("jump_penalty_weight", [0.0]):
                for jump_free_ft in values.get("jump_free_ft", [25.0]):
                    for max_abs_delta in values.get("max_abs_delta_vs_default", [999999.0]):
                        for max_pf_std in values.get("max_pf_ancc_std", [999999.0]):
                            for min_md_since in values.get("min_md_since", [0.0]):
                                for min_segment_len in values.get("min_segment_len", [1]):
                                    variant = (
                                        f"viterbi_sw{int(float(switch_penalty) * 10):03d}"
                                        f"_bias{int(float(nondefault_bias) * 10):03d}"
                                        f"_jw{int(float(jump_penalty_weight) * 100):03d}"
                                        f"_jf{int(float(jump_free_ft)):03d}"
                                        f"_d{int(float(max_abs_delta)):04d}"
                                        f"_std{int(float(max_pf_std)):06d}"
                                        f"_md{int(float(min_md_since)):04d}"
                                        f"_seg{int(min_segment_len):03d}"
                                    )
                                    specs.append(
                                        ViterbiSpec(
                                            variant=variant,
                                            switch_penalty=float(switch_penalty),
                                            nondefault_bias=float(nondefault_bias),
                                            jump_penalty_weight=float(jump_penalty_weight),
                                            jump_free_ft=float(jump_free_ft),
                                            jump_scale_ft=float(
                                                values.get("jump_scale_ft", 25.0)
                                            ),
                                            max_abs_delta_vs_default=float(max_abs_delta),
                                            max_pf_ancc_std=float(max_pf_std),
                                            min_md_since=float(min_md_since),
                                            min_segment_len=int(min_segment_len),
                                        )
                                    )
    return specs


def build_local_cost(
    *,
    frame: pd.DataFrame,
    predicted_error: np.ndarray,
    candidate_values: np.ndarray,
    candidate_names: list[str],
    default_idx: int,
    allowed_switch_idx: np.ndarray,
    spec: ViterbiSpec,
) -> np.ndarray:
    cost = predicted_error.astype(np.float32).copy()
    n_rows, n_candidates = cost.shape
    allowed = np.zeros(n_candidates, dtype=bool)
    allowed[default_idx] = True
    allowed[allowed_switch_idx] = True
    cost[:, ~allowed] = np.float32(1e9)
    nondefault = np.arange(n_candidates) != default_idx
    if spec.nondefault_bias != 0.0:
        cost[:, nondefault] += np.float32(spec.nondefault_bias)

    default_values = candidate_values[:, default_idx]
    delta_vs_default = np.abs(candidate_values - default_values[:, None])
    cost[(delta_vs_default > spec.max_abs_delta_vs_default) & nondefault[None, :]] = np.float32(1e9)

    if "pf_ancc_std" in frame.columns and "pf_ancc" in candidate_names:
        pf_idx = candidate_names.index("pf_ancc")
        pf_std = pd.to_numeric(frame["pf_ancc_std"], errors="coerce").to_numpy(np.float32)
        cost[(pf_std > spec.max_pf_ancc_std), pf_idx] = np.float32(1e9)

    if "md_since" in frame.columns and spec.min_md_since > 0.0:
        md_since = pd.to_numeric(frame["md_since"], errors="coerce").to_numpy(np.float32)
        cost[(md_since < spec.min_md_since)[:, None] & nondefault[None, :]] = np.float32(1e9)

    if not np.isfinite(cost).all():
        bad = np.argwhere(~np.isfinite(cost))[:5].tolist()
        raise ValueError(f"local cost has non-finite values: {bad}")
    if len(cost) != n_rows:
        raise ValueError("local cost row mismatch")
    return cost


def run_viterbi_for_well(
    *,
    local_cost: np.ndarray,
    candidate_values: np.ndarray,
    spec: ViterbiSpec,
) -> np.ndarray:
    n_rows, n_candidates = local_cost.shape
    if n_rows == 0:
        return np.empty(0, dtype=np.int16)
    dp = np.empty((n_rows, n_candidates), dtype=np.float64)
    back = np.zeros((n_rows, n_candidates), dtype=np.int16)
    dp[0] = local_cost[0].astype(np.float64)
    candidate_index = np.arange(n_candidates)
    switch_matrix = (candidate_index[:, None] != candidate_index[None, :]).astype(np.float64)
    for row in range(1, n_rows):
        prev_values = candidate_values[row - 1]
        curr_values = candidate_values[row]
        jump = np.maximum(
            np.abs(prev_values[:, None] - curr_values[None, :]) - spec.jump_free_ft,
            0.0,
        ) / max(spec.jump_scale_ft, 1e-6)
        transition = spec.switch_penalty * switch_matrix + spec.jump_penalty_weight * jump
        prev = dp[row - 1][:, None] + transition
        back[row] = np.argmin(prev, axis=0).astype(np.int16)
        dp[row] = local_cost[row].astype(np.float64) + prev[back[row], candidate_index]
    selected = np.empty(n_rows, dtype=np.int16)
    selected[-1] = int(np.argmin(dp[-1]))
    for row in range(n_rows - 2, -1, -1):
        selected[row] = back[row + 1, selected[row + 1]]
    return selected


def prune_short_switch_segments(
    selected: np.ndarray,
    *,
    default_idx: int,
    min_segment_len: int,
) -> np.ndarray:
    if min_segment_len <= 1 or len(selected) == 0:
        return selected
    out = selected.copy()
    start = 0
    while start < len(out):
        end = start + 1
        while end < len(out) and out[end] == out[start]:
            end += 1
        if out[start] != default_idx and end - start < min_segment_len:
            out[start:end] = default_idx
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
        frame=frame,
        predicted_error=predicted_error,
        candidate_values=candidate_values,
        candidate_names=candidate_names,
        default_idx=default_idx,
        allowed_switch_idx=allowed_switch_idx,
        spec=spec,
    )
    row_indices = _row_indices_from_ids(frame["id"])
    well_codes, _ = pd.factorize(frame["well"], sort=True)
    order = np.lexsort((row_indices, well_codes))
    selected = np.full(len(frame), default_idx, dtype=np.int16)
    for _, positions in pd.Series(order).groupby(well_codes[order], sort=False):
        pos = positions.to_numpy(np.int64)
        path = run_viterbi_for_well(
            local_cost=local_cost[pos],
            candidate_values=candidate_values[pos],
            spec=spec,
        )
        selected[pos] = prune_short_switch_segments(
            path,
            default_idx=default_idx,
            min_segment_len=spec.min_segment_len,
        )
    return selected


def metrics_for_indices(
    *,
    frame: pd.DataFrame,
    selected_idx: np.ndarray,
    candidate_values: np.ndarray,
    oracle_labels: np.ndarray,
    candidate_names: list[str],
    variant: str,
    mode: str,
    default_idx: int,
    well_codes: np.ndarray,
    order: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    pred = candidate_values[np.arange(len(selected_idx)), selected_idx].astype(np.float32)
    abs_error = np.abs(pred - true_tvt)
    ordered_selected = selected_idx[order]
    ordered_wells = well_codes[order]
    switch_mask = (ordered_selected[1:] != ordered_selected[:-1]) & (
        ordered_wells[1:] == ordered_wells[:-1]
    )
    switch_count = int(np.sum(switch_mask)) if len(ordered_selected) > 1 else 0
    return {
        "variant": variant,
        "mode": mode,
        **params,
        "rows": int(len(selected_idx)),
        "wells": int(frame["well"].nunique()),
        "rmse_tvt": rmse(true_tvt, pred),
        "mae_tvt": mae(true_tvt, pred),
        "within_10ft": float(np.mean(abs_error <= 10.0)),
        "oracle_label_accuracy": float(np.mean(selected_idx == oracle_labels)),
        "default_candidate_rate": float(np.mean(selected_idx == default_idx)),
        "path_switch_count": switch_count,
        "path_switch_per_1000_rows": float(switch_count / max(len(selected_idx), 1) * 1000.0),
    }


def selection_distribution_rows(
    *, variant: str, mode: str, selected_idx: np.ndarray, candidate_names: list[str]
) -> list[dict[str, Any]]:
    counts = np.bincount(selected_idx.astype(np.int16), minlength=len(candidate_names))
    total = int(np.sum(counts))
    return [
        {
            "variant": variant,
            "mode": mode,
            "selected_candidate": candidate_names[idx],
            "rows": int(count),
            "rate": float(count / total) if total else 0.0,
        }
        for idx, count in enumerate(counts)
    ]


def by_well_rows(
    *,
    frame: pd.DataFrame,
    selected_idx: np.ndarray,
    candidate_values: np.ndarray,
    variant: str,
    mode: str,
    well_codes: np.ndarray,
    well_names: pd.Index,
    order: np.ndarray,
) -> list[dict[str, Any]]:
    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    pred = candidate_values[np.arange(len(selected_idx)), selected_idx].astype(np.float32)
    rows: list[dict[str, Any]] = []
    for code, well in enumerate(well_names):
        positions = order[well_codes[order] == code]
        if len(positions) == 0:
            continue
        sel = selected_idx[positions]
        switches = int(np.sum(sel[1:] != sel[:-1])) if len(sel) > 1 else 0
        abs_error = np.abs(pred[positions] - true_tvt[positions])
        rows.append(
            {
                "variant": variant,
                "mode": mode,
                "well": str(well),
                "rows": int(len(positions)),
                "rmse_tvt": rmse(true_tvt[positions], pred[positions]),
                "mae_tvt": mae(true_tvt[positions], pred[positions]),
                "within_10ft": float(np.mean(abs_error <= 10.0)),
                "path_switch_count": switches,
                "path_switch_per_1000_rows": float(switches / max(len(positions), 1) * 1000.0),
            }
        )
    return rows


def bucket_metric_rows(
    *,
    frame: pd.DataFrame,
    selected_idx: np.ndarray,
    candidate_values: np.ndarray,
    oracle_labels: np.ndarray,
    variant: str,
    mode: str,
    bucket_defs: list[tuple[str, np.ndarray, list[str]]],
) -> list[dict[str, Any]]:
    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    pred = candidate_values[np.arange(len(selected_idx)), selected_idx].astype(np.float32)
    rows: list[dict[str, Any]] = []
    for bucket_family, codes, labels in bucket_defs:
        for bucket_code, label in enumerate(labels):
            mask = codes == bucket_code
            if not mask.any():
                continue
            abs_error = np.abs(pred[mask] - true_tvt[mask])
            rows.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "bucket_family": bucket_family,
                    "bucket": str(label),
                    "rows": int(mask.sum()),
                    "rmse_tvt": rmse(true_tvt[mask], pred[mask]),
                    "mae_tvt": mae(true_tvt[mask], pred[mask]),
                    "within_10ft": float(np.mean(abs_error <= 10.0)),
                    "oracle_label_accuracy": float(
                        np.mean(selected_idx[mask] == oracle_labels[mask])
                    ),
                }
            )
    return rows


def subgroup_defs_for_frame(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[list[tuple[str, np.ndarray]], dict[str, Any]]:
    context, meta = subgroup_context(frame, config)
    masks: list[tuple[str, np.ndarray]] = []
    if "verification_like_spatial_role" in context.columns:
        masks.append(
            (
                "exp115_spatial_valid",
                context["verification_like_spatial_role"].eq("valid").to_numpy(),
            )
        )
    if "verification_like_typewell_purged_role" in context.columns:
        masks.append(
            (
                "exp115_typewell_purged_valid",
                context["verification_like_typewell_purged_role"].eq("valid").to_numpy(),
            )
        )
    for column in [
        "copcf_gate_any_outlier_signal_k8",
        "copcf_gate_own_z_gt2p0",
        "copcf_nearest_other_closer",
        "copcf_nearby_majority_diff_k8",
    ]:
        if column in context.columns:
            masks.append(
                (
                    column,
                    pd.to_numeric(context[column], errors="coerce").fillna(0.0).to_numpy()
                    > 0.5,
                )
            )
    if "hmpf_real_confidence_proxy" in context.columns:
        confidence = pd.to_numeric(
            context["hmpf_real_confidence_proxy"], errors="coerce"
        ).fillna(0.0)
        masks.append(
            (
                "hmpf_confidence_bottom_quartile",
                confidence.to_numpy() <= float(confidence.quantile(0.25)),
            )
        )
    if "hmpf_nearest_sample_distance_rows" in context.columns:
        distance = pd.to_numeric(
            context["hmpf_nearest_sample_distance_rows"], errors="coerce"
        ).fillna(1_000_000.0)
        masks.append(("hmpf_far_from_sparse_sample_gt512", distance.to_numpy() > 512.0))
    if "hmdn_confidence_proxy" in context.columns:
        confidence = pd.to_numeric(context["hmdn_confidence_proxy"], errors="coerce").fillna(0.0)
        masks.append(
            (
                "hmdn_confidence_bottom_quartile",
                confidence.to_numpy() <= float(confidence.quantile(0.25)),
            )
        )
    if "hmdn_nearest_sample_distance_rows" in context.columns:
        distance = pd.to_numeric(
            context["hmdn_nearest_sample_distance_rows"], errors="coerce"
        ).fillna(1_000_000.0)
        masks.append(("hmdn_far_from_sparse_sample_gt512", distance.to_numpy() > 512.0))
    return masks, meta


def subgroup_metric_rows(
    *,
    frame: pd.DataFrame,
    selected_idx: np.ndarray,
    candidate_values: np.ndarray,
    variant: str,
    mode: str,
    subgroup_defs: list[tuple[str, np.ndarray]],
) -> list[dict[str, Any]]:
    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    pred = candidate_values[np.arange(len(selected_idx)), selected_idx].astype(np.float32)
    wells = frame["well"].to_numpy()
    rows: list[dict[str, Any]] = []
    for subgroup, mask in subgroup_defs:
        if not mask.any():
            continue
        abs_error = np.abs(pred[mask] - true_tvt[mask])
        rows.append(
            {
                "variant": variant,
                "mode": mode,
                "subgroup": subgroup,
                "rows": int(mask.sum()),
                "wells": int(pd.Series(wells[mask]).nunique()),
                "rmse_tvt": rmse(true_tvt[mask], pred[mask]),
                "mae_tvt": mae(true_tvt[mask], pred[mask]),
                "within_10ft": float(np.mean(abs_error <= 10.0)),
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
    oracle_labels: np.ndarray,
    candidate_names: list[str],
) -> pd.DataFrame:
    selected_tvt = candidate_values[np.arange(len(selected_idx)), selected_idx].astype(np.float32)
    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    return pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "well": frame["well"].to_numpy(),
            "variant": variant,
            "mode": mode,
            "selected_candidate": np.asarray([candidate_names[i] for i in selected_idx]),
            "selected_candidate_index": selected_idx.astype(np.int16),
            "selected_tvt": selected_tvt,
            "true_tvt": true_tvt,
            "abs_error": np.abs(selected_tvt - true_tvt).astype(np.float32),
            "oracle_candidate": np.asarray([candidate_names[i] for i in oracle_labels]),
            "oracle_label": oracle_labels.astype(np.int16),
        }
    )


def evaluate_viterbi_grid(
    *,
    frame: pd.DataFrame,
    scores: dict[str, np.ndarray],
    candidate_values: np.ndarray,
    oracle_labels: np.ndarray,
    candidate_names: list[str],
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str | None,
    np.ndarray | None,
]:
    default_idx = candidate_names.index(
        str(get_nested(config, "selector.default_candidate") or "likpf_mean")
    )
    allowed_names = [
        str(value) for value in (get_nested(config, "selector.allowed_switch_candidates") or [])
    ]
    allowed_switch_idx = np.asarray(
        [candidate_names.index(name) for name in allowed_names], dtype=np.int16
    )
    row_indices = _row_indices_from_ids(frame["id"])
    well_codes, well_names = pd.factorize(frame["well"], sort=True)
    well_codes = well_codes.astype(np.int32)
    order = np.lexsort((row_indices, well_codes))
    bucket_defs: list[tuple[str, np.ndarray, list[str]]] = []
    codes, labels = distance_bucket_codes(
        frame["md_since"] if "md_since" in frame.columns else np.nan
    )
    bucket_defs.append(("distance_bucket", codes, labels))
    codes, labels = tail_rank_bucket_codes(frame["id"])
    bucket_defs.append(("tail_rank_bucket", codes, labels))
    for source_column, bucket_name in [
        ("eval_len", "eval_len_bucket"),
        ("pf_ancc_std", "pf_seed_std_bucket"),
        ("likpf_mean_d", "likpf_delta_bucket"),
        ("copcf_own_cluster_dist_z", "cluster_outlier_z_bucket"),
        ("hmpf_real_confidence_proxy", "heatmap_confidence_bucket"),
        ("hmpf_nearest_sample_distance_rows", "heatmap_sparse_distance_bucket"),
        ("hmpf_real_vs_shuffled_top1_score_gap", "heatmap_real_shuffled_gap_bucket"),
        ("hmdn_confidence_proxy", "hmdn_confidence_bucket"),
        ("hmdn_nearest_sample_distance_rows", "hmdn_sparse_distance_bucket"),
        ("hmdn_top10_vs_candidate_min_abs", "hmdn_top10_candidate_gap_bucket"),
    ]:
        if source_column in frame.columns:
            codes, labels = quantile_bucket_codes(frame[source_column], bucket_name)
            bucket_defs.append((bucket_name, codes, labels))
    subgroup_defs, _subgroup_meta = subgroup_defs_for_frame(frame, config)

    metric_rows: list[dict[str, Any]] = []
    distribution_rows_all: list[dict[str, Any]] = []
    by_well_all: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    params_rows: list[dict[str, Any]] = []
    specs = variant_specs_from_config(config)
    best_variant: str | None = None
    best_rmse = np.inf
    best_selected: np.ndarray | None = None
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
        metric = metrics_for_indices(
            frame=frame,
            selected_idx=selected,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            candidate_names=candidate_names,
            variant=spec.variant,
            mode="viterbi",
            default_idx=default_idx,
            well_codes=well_codes,
            order=order,
            params=params,
        )
        metric_rows.append(metric)
        distribution_rows_all.extend(
            selection_distribution_rows(
                variant=spec.variant,
                mode="viterbi",
                selected_idx=selected,
                candidate_names=candidate_names,
            )
        )
        by_well_all.extend(
            by_well_rows(
                frame=frame,
                selected_idx=selected,
                candidate_values=candidate_values,
                variant=spec.variant,
                mode="viterbi",
                well_codes=well_codes,
                well_names=well_names,
                order=order,
            )
        )
        bucket_rows.extend(
            bucket_metric_rows(
                frame=frame,
                selected_idx=selected,
                candidate_values=candidate_values,
                oracle_labels=oracle_labels,
                variant=spec.variant,
                mode="viterbi",
                bucket_defs=bucket_defs,
            )
        )
        subgroup_rows.extend(
            subgroup_metric_rows(
                frame=frame,
                selected_idx=selected,
                candidate_values=candidate_values,
                variant=spec.variant,
                mode="viterbi",
                subgroup_defs=subgroup_defs,
            )
        )
        params_rows.append({"variant": spec.variant, "mode": "viterbi", **params})
        rmse_value = float(metric["rmse_tvt"])
        if rmse_value < best_rmse:
            best_rmse = rmse_value
            best_variant = spec.variant
            best_selected = selected.copy()

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(distribution_rows_all),
        pd.DataFrame(by_well_all),
        pd.DataFrame(bucket_rows),
        pd.DataFrame(subgroup_rows),
        pd.DataFrame(params_rows),
        best_variant,
        best_selected,
    )


def summarize_decision(
    metrics: pd.DataFrame, distribution: pd.DataFrame, by_well: pd.DataFrame
) -> dict[str, Any]:
    best_oof = metrics[metrics["mode"].isin(["oof", "viterbi"])].sort_values("rmse_tvt").head(1)
    likpf = metrics[metrics["variant"].eq("likpf_mean_single")].head(1)
    multiobs = metrics[metrics["variant"].eq("multiobs_score_top1")].head(1)
    exp157_rowwise_rmse = 10.79579983712686
    exp158_continuity_rmse = 10.789163253
    decision = "selector_not_run"
    delta_likpf = None
    delta_multiobs = None
    delta_exp157 = None
    delta_exp158 = None
    pf_rate = None
    hmdn_rate = None
    if not best_oof.empty:
        best = best_oof.iloc[0]
        if not likpf.empty:
            delta_likpf = float(best["rmse_tvt"] - likpf.iloc[0]["rmse_tvt"])
        if not multiobs.empty:
            delta_multiobs = float(best["rmse_tvt"] - multiobs.iloc[0]["rmse_tvt"])
        delta_exp157 = float(best["rmse_tvt"] - exp157_rowwise_rmse)
        delta_exp158 = float(best["rmse_tvt"] - exp158_continuity_rmse)
        dist = distribution[
            (distribution["variant"] == best["variant"])
            & (distribution["selected_candidate"] == "pf_ancc")
        ]
        pf_rate = float(dist["rate"].iloc[0]) if not dist.empty else 0.0
        hmdn_dist = distribution[
            (distribution["variant"] == best["variant"])
            & (distribution["selected_candidate"].astype(str).str.startswith("hmdn_top"))
        ]
        hmdn_rate = float(hmdn_dist["rate"].sum()) if not hmdn_dist.empty else 0.0
        worst_switch = by_well[by_well["variant"].eq(best["variant"])][
            "path_switch_per_1000_rows"
        ].max()
        if delta_exp158 is not None and delta_exp158 < 0.0:
            decision = "heatmap_mdn_topk_candidate_selector_supported_for_review"
        elif delta_likpf is not None and delta_likpf < -0.25 and pf_rate >= 0.05:
            decision = "ranker_supported_but_not_better_than_exp158_continuity"
        elif delta_likpf is not None and delta_likpf < 0.0:
            decision = "weak_selector_supported_against_likpf_only"
        else:
            decision = "selector_not_supported"
        return {
            "recommendation": decision,
            "best_variant": to_jsonable(best.to_dict()),
            "delta_rmse_vs_likpf_mean": delta_likpf,
            "delta_rmse_vs_multiobs_score_top1": delta_multiobs,
            "delta_rmse_vs_exp157_rowwise": delta_exp157,
            "delta_rmse_vs_exp158_continuity": delta_exp158,
            "best_pf_ancc_selection_rate": pf_rate,
            "best_hmdn_selection_rate": hmdn_rate,
            "best_max_path_switch_per_1000_rows": (
                float(worst_switch) if pd.notna(worst_switch) else None
            ),
        }
    return {"recommendation": decision}


def run_heatmap_mdn_topk_as_selector_candidates_on_exp158(
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
    frame, heatmap_columns, heatmap_meta = add_heatmap_path_features(
        frame,
        config,
        candidates,
    )
    frame, heatmap_mdn_columns, heatmap_mdn_meta = add_heatmap_mdn_candidate_features(
        frame,
        config,
        candidates,
    )
    missing_candidate_columns = [
        spec.column for spec in candidates if spec.column not in frame.columns
    ]
    if missing_candidate_columns:
        raise ValueError(
            f"candidate columns are missing after enrichment: {missing_candidate_columns}"
        )
    frame, engineered_columns, candidate_values, oracle_labels = add_candidate_labels_and_features(
        frame,
        candidates,
        config=config,
        include_candidate_values=bool(get_nested(config, "ranker.include_candidate_values")),
    )
    assert_unique_columns(frame, "assembled train feature frame")
    feature_columns = select_numeric_feature_columns(
        frame,
        config,
        [*engineered_columns, *enrichment_columns, *heatmap_columns, *heatmap_mdn_columns],
    )
    metrics, predictions, importance, model_manifest, scores = train_and_score(
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
    subgroups = subgroup_metrics(predictions, frame, config)
    (
        viterbi_metrics,
        viterbi_distribution,
        viterbi_by_well,
        viterbi_buckets,
        viterbi_subgroups,
        viterbi_params,
        best_viterbi_variant,
        best_viterbi_selected,
    ) = evaluate_viterbi_grid(
        frame=frame,
        scores=scores,
        candidate_values=candidate_values,
        oracle_labels=oracle_labels,
        candidate_names=[spec.name for spec in candidates],
        config=config,
    )
    if len(viterbi_metrics):
        metrics = pd.concat([metrics, viterbi_metrics], ignore_index=True, sort=False)
        distribution = pd.concat(
            [distribution, viterbi_distribution], ignore_index=True, sort=False
        )
        by_well = pd.concat([by_well, viterbi_by_well], ignore_index=True, sort=False)
        buckets = pd.concat([buckets, viterbi_buckets], ignore_index=True, sort=False)
        subgroups = pd.concat([subgroups, viterbi_subgroups], ignore_index=True, sort=False)
    if best_viterbi_variant is not None and best_viterbi_selected is not None:
        predictions = pd.concat(
            [
                predictions,
                selected_prediction_frame(
                    frame=frame,
                    variant=best_viterbi_variant,
                    mode="viterbi",
                    selected_idx=best_viterbi_selected,
                    candidate_values=candidate_values,
                    oracle_labels=oracle_labels,
                    candidate_names=[spec.name for spec in candidates],
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    metrics = metrics.sort_values("rmse_tvt")
    distribution = distribution.sort_values(["variant", "mode", "selected_candidate"])
    by_well = by_well.sort_values(["variant", "mode", "rmse_tvt"], ascending=[True, True, False])
    buckets = buckets.sort_values(["variant", "mode", "bucket_family", "bucket"])
    subgroups = subgroups.sort_values(["variant", "mode", "subgroup"])
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
    error_selected, error_margin, error_top1 = second_margin_low(scores["predicted_error"])
    score_summary = pd.DataFrame(
        [
            {
                "score": "predicted_error_selected",
                "mean": float(np.mean(error_top1)),
                "std": float(np.std(error_top1)),
                "p05": float(np.quantile(error_top1, 0.05)),
                "p50": float(np.quantile(error_top1, 0.50)),
                "p95": float(np.quantile(error_top1, 0.95)),
            },
            {
                "score": "predicted_error_margin",
                "mean": float(np.mean(error_margin)),
                "std": float(np.std(error_margin)),
                "p05": float(np.quantile(error_margin, 0.05)),
                "p50": float(np.quantile(error_margin, 0.50)),
                "p95": float(np.quantile(error_margin, 0.95)),
            },
            *[
                {
                    "score": f"predicted_error_{spec.name}",
                    "mean": float(np.mean(scores["predicted_error"][:, idx])),
                    "std": float(np.std(scores["predicted_error"][:, idx])),
                    "p05": float(np.quantile(scores["predicted_error"][:, idx], 0.05)),
                    "p50": float(np.quantile(scores["predicted_error"][:, idx], 0.50)),
                    "p95": float(np.quantile(scores["predicted_error"][:, idx], 0.95)),
                    "selected_rate": float(np.mean(error_selected == idx)),
                }
                for idx, spec in enumerate(candidates)
            ],
        ]
    )

    metrics_path = output_dir / f"{OUTPUT_PREFIX}_metrics.csv"
    predictions_path = output_dir / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
    distribution_path = output_dir / f"{OUTPUT_PREFIX}_selection_distribution.csv"
    by_well_path = output_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    buckets_path = output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    subgroups_path = output_dir / f"{OUTPUT_PREFIX}_subgroup_metrics.csv"
    viterbi_params_path = output_dir / f"{OUTPUT_PREFIX}_viterbi_params.csv"
    score_summary_path = output_dir / f"{OUTPUT_PREFIX}_score_summary.csv"
    importance_path = output_dir / f"{OUTPUT_PREFIX}_feature_importance.csv"
    mean_importance_path = output_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv"
    schema_out_path = output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
    heatmap_schema_path = output_dir / f"{OUTPUT_PREFIX}_heatmap_feature_schema.csv"
    heatmap_summary_path = output_dir / f"{OUTPUT_PREFIX}_heatmap_feature_summary.csv"
    heatmap_mdn_schema_path = output_dir / f"{OUTPUT_PREFIX}_heatmap_mdn_feature_schema.csv"
    heatmap_mdn_summary_path = output_dir / f"{OUTPUT_PREFIX}_heatmap_mdn_feature_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    distribution.to_csv(distribution_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    buckets.to_csv(buckets_path, index=False)
    subgroups.to_csv(subgroups_path, index=False)
    viterbi_params.to_csv(viterbi_params_path, index=False)
    score_summary.to_csv(score_summary_path, index=False)
    importance.to_csv(importance_path, index=False)
    mean_importance.to_csv(mean_importance_path, index=False)
    pd.DataFrame(
        [{"feature_index": idx, "feature": feature} for idx, feature in enumerate(feature_columns)]
    ).to_csv(schema_out_path, index=False)
    pd.DataFrame(
        [{"feature_index": idx, "feature": feature} for idx, feature in enumerate(heatmap_columns)]
    ).to_csv(heatmap_schema_path, index=False)
    summarize_feature_frame(frame, heatmap_columns).to_csv(heatmap_summary_path, index=False)
    pd.DataFrame(
        [
            {"feature_index": idx, "feature": feature}
            for idx, feature in enumerate(heatmap_mdn_columns)
        ]
    ).to_csv(heatmap_mdn_schema_path, index=False)
    summarize_feature_frame(frame, heatmap_mdn_columns).to_csv(
        heatmap_mdn_summary_path,
        index=False,
    )

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
        "heatmap_path_features": to_jsonable(heatmap_meta),
        "heatmap_mdn_candidate_features": to_jsonable(heatmap_mdn_meta),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "best_metric": to_jsonable(best),
        "best_viterbi_variant": best_viterbi_variant,
        "decision": to_jsonable(decision),
        "sha256": {
            "metrics": sha256_path(metrics_path),
            "predictions": sha256_path(predictions_path),
            "predictions_decompressed": sha256_path(predictions_path, decompressed=True),
            "feature_schema": sha256_path(schema_out_path),
            "heatmap_feature_schema": sha256_path(heatmap_schema_path),
            "heatmap_feature_summary": sha256_path(heatmap_summary_path),
            "heatmap_mdn_feature_schema": sha256_path(heatmap_mdn_schema_path),
            "heatmap_mdn_feature_summary": sha256_path(heatmap_mdn_summary_path),
            "subgroup_metrics": sha256_path(subgroups_path),
            "viterbi_params": sha256_path(viterbi_params_path),
            "score_summary": sha256_path(score_summary_path),
            "prediction_by_variant": prediction_hashes,
        },
        "model_manifest": model_manifest,
        "artifacts": {
            "metrics": metrics_path.name,
            "oof_predictions": predictions_path.name,
            "selection_distribution": distribution_path.name,
            "by_well": by_well_path.name,
            "bucket_metrics": buckets_path.name,
            "subgroup_metrics": subgroups_path.name,
            "viterbi_params": viterbi_params_path.name,
            "score_summary": score_summary_path.name,
            "feature_importance": importance_path.name,
            "feature_importance_mean": mean_importance_path.name,
            "feature_schema": schema_out_path.name,
            "heatmap_feature_schema": heatmap_schema_path.name,
            "heatmap_feature_summary": heatmap_summary_path.name,
            "heatmap_mdn_feature_schema": heatmap_mdn_schema_path.name,
            "heatmap_mdn_feature_summary": heatmap_mdn_summary_path.name,
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
    return run_heatmap_mdn_topk_as_selector_candidates_on_exp158(
        output_dir=output_dir,
        cache_path=cache_path,
        schema_path=schema_path,
        max_rows=max_rows,
    )


run_cnn_sdf_mtp_heatmap_path_features_on_exp158 = (
    run_heatmap_mdn_topk_as_selector_candidates_on_exp158
)
run_candidate_ranker_feature_enrichment = run_heatmap_mdn_topk_as_selector_candidates_on_exp158
run_pf_candidate_ranker_or_nway_classifier = run_heatmap_mdn_topk_as_selector_candidates_on_exp158


if __name__ == "__main__":
    main()

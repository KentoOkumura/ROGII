from __future__ import annotations

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

OUTPUT_PREFIX = "exp187_cluster_outlier_alt_typewell_pfbeam_audit"
EXPERIMENT_NAME = "exp187_cluster_outlier_alt_typewell_pfbeam_audit"
EXP065_CLUSTER_ASSIGNMENTS = "common_typewell_cluster_assignments.csv"
EXP114_WELL_GEOMETRY = "exp114_spatial_neighbor_prior_signal_audit_well_geometry_summary.csv"
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
FULL_REPLAY_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"


@dataclass(frozen=True)
class TypewellStrategy:
    name: str
    kind: str
    source_cluster_id: str | None
    source_well: str
    source_path: Path | None
    source_wells: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class PrefixHoldout:
    well: str
    masked: pd.DataFrame
    eval_index: np.ndarray
    eval_ids: np.ndarray
    true_tvt: np.ndarray
    target_delta: np.ndarray
    last_known_tvt: float
    last_known_md: float
    cache_md_since: np.ndarray | None
    status: dict[str, Any]


@dataclass(frozen=True)
class PfRun:
    preds: np.ndarray
    log_likelihoods: np.ndarray
    ess_mean_by_row: np.ndarray
    resampled_by_row: np.ndarray
    seed_weights: np.ndarray


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
        return float(value) if np.isfinite(float(value)) else None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    key = "::".join(str(part) for part in parts).encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def find_existing_path(
    *,
    filename: str,
    explicit_path: str | Path | None = None,
    candidates: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Path | None:
    paths: list[Path] = []
    if explicit_path is not None:
        paths.append(Path(explicit_path))
    if candidates:
        paths.extend(Path(item) for item in candidates)
    paths.extend(
        [
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
            Path("artifacts") / filename,
            Path("experiments")
            / "exp065_typewell_supertype_cluster_cv_audit"
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
            / "exp072_exp063_full_replay_feature_cache"
            / "artifacts"
            / filename,
        ]
    )
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in KAGGLE_INPUT_ROOT.glob(f"**/{filename}"):
            if path.exists() and path.stat().st_size > 0:
                return path
    return None


def require_path(
    *,
    filename: str,
    explicit_path: str | Path | None = None,
    candidates: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Path:
    path = find_existing_path(filename=filename, explicit_path=explicit_path, candidates=candidates)
    if path is None:
        checked = [str(explicit_path)] if explicit_path is not None else []
        checked.extend(str(item) for item in candidates or [])
        raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked: {checked}")
    return path


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"required column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int64)


def read_exp072_eval_cache(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = require_path(
        filename=FULL_REPLAY_TRAIN_FEATURES,
        explicit_path=get_nested(config, "data.exp072_train_feature_cache_local"),
    )
    required = ["id", "well", "target", "last_known_tvt"]
    optional = ["md_since", "eval_len", "known_len"]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    usecols = required + [column for column in optional if column in header]
    max_rows = get_nested(config, "audit.max_rows")
    frame = pd.read_csv(
        source,
        usecols=usecols,
        nrows=None if max_rows is None else int(max_rows),
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    frame["row_idx"] = row_indices_from_ids(frame["id"]).astype(np.int32)
    frame["true_tvt"] = (
        numeric_array(frame, "last_known_tvt") + numeric_array(frame, "target")
    ).astype(np.float32)

    schema_path = find_existing_path(
        filename=FULL_REPLAY_FEATURE_SCHEMA,
        explicit_path=get_nested(config, "data.exp072_feature_schema_local"),
    )
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
        "max_rows": None if max_rows is None else int(max_rows),
    }
    return frame, metadata


def fill_numeric(values: pd.Series | np.ndarray, fallback: float) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    if series.notna().any():
        fallback = float(series.mean())
    filled = series.interpolate(limit_direction="both").ffill().bfill().fillna(fallback)
    return filled.to_numpy(np.float32)


def smooth(values: np.ndarray, window: int, fallback: float) -> np.ndarray:
    if window <= 1:
        return fill_numeric(values, fallback)
    return (
        pd.Series(values, dtype="float64")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .fillna(fallback)
        .rolling(int(window), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )


def nearest_index(values: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(values, target, side="left"))
    if idx >= len(values):
        return len(values) - 1
    if idx > 0 and abs(float(values[idx - 1]) - target) <= abs(float(values[idx]) - target):
        return idx - 1
    return idx


def read_cluster_assignments(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = require_path(
        filename=EXP065_CLUSTER_ASSIGNMENTS,
        explicit_path=get_nested(config, "data.exp065_cluster_assignments_local"),
    )
    frame = pd.read_csv(source, dtype=str)
    required = {"method", "threshold", "cluster_id", "well_id", "cluster_size"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    frame["well_id"] = frame["well_id"].astype(str)
    frame["cluster_id"] = frame["cluster_id"].astype(str)
    if "representative_well_id" not in frame.columns:
        frame["representative_well_id"] = frame["well_id"]
    frame["representative_well_id"] = frame["representative_well_id"].astype(str)
    frame["cluster_size"] = (
        pd.to_numeric(frame["cluster_size"], errors="coerce").fillna(0).astype(int)
    )
    method = str(get_nested(config, "cluster.assignment_method") or "native_overlap")
    threshold = str(get_nested(config, "cluster.assignment_threshold") or "1")
    subset = frame[
        (frame["method"].astype(str) == method) & (frame["threshold"].astype(str) == threshold)
    ].copy()
    if subset.empty:
        raise ValueError(f"no cluster assignments for method={method} threshold={threshold}")
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "selected_method": method,
        "selected_threshold": threshold,
        "selected_rows": int(len(subset)),
        "selected_wells": int(subset["well_id"].nunique()),
        "selected_clusters": int(subset["cluster_id"].nunique()),
    }
    return subset, metadata


def read_well_geometry(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = require_path(
        filename=EXP114_WELL_GEOMETRY,
        explicit_path=get_nested(config, "data.exp114_well_geometry_local"),
    )
    required = ["well", "centroid_x", "centroid_y"]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    optional = [
        "rows",
        "eval_start_row_idx",
        "eval_end_row_idx",
        "centroid_z",
        "azimuth",
        "tortuosity",
        "prefix_tvt_range",
        "last_md",
    ]
    usecols = required + [column for column in optional if column in header]
    frame = pd.read_csv(source, usecols=usecols, dtype={"well": str}, low_memory=False)
    for column in frame.columns:
        if column != "well":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": list(frame.columns),
    }
    return frame, metadata


def robust_scale(values: np.ndarray, floor: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float(floor)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(floor)
    return max(scale, float(floor))


def representative_by_cluster(assignments: pd.DataFrame) -> dict[str, str]:
    representatives: dict[str, str] = {}
    for cluster_id, group in assignments.groupby("cluster_id", sort=False):
        reps = group["representative_well_id"].dropna().astype(str)
        if len(reps):
            representatives[str(cluster_id)] = str(reps.iloc[0])
        else:
            representatives[str(cluster_id)] = str(group["well_id"].iloc[0])
    return representatives


def cluster_members_by_cluster(
    config: dict[str, Any],
    train_dir: Path | None = None,
) -> dict[str, tuple[str, ...]]:
    assignments, _ = read_cluster_assignments(config)
    members: dict[str, tuple[str, ...]] = {}
    for cluster_id, group in assignments.groupby("cluster_id", sort=False):
        wells = sorted(group["well_id"].dropna().astype(str).unique().tolist())
        if train_dir is not None:
            wells = [well for well in wells if (train_dir / f"{well}__typewell.csv").exists()]
        members[str(cluster_id)] = tuple(wells)
    return members


def cluster_typewell_available(
    members_by_cluster: dict[str, tuple[str, ...]],
    cluster_id: Any,
) -> bool:
    if not is_truthy(cluster_id):
        return False
    return len(members_by_cluster.get(str(cluster_id), ())) > 0


def build_cluster_features(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    assignments, assignment_meta = read_cluster_assignments(config)
    geometry, geometry_meta = read_well_geometry(config)
    min_cluster_size = int(get_nested(config, "cluster.min_cluster_size") or 2)
    scale_floor = float(get_nested(config, "cluster.robust_scale_floor_ft") or 250.0)
    nearby_k_values = [int(value) for value in get_nested(config, "cluster.nearby_k_values") or [8]]
    majority_min_share = float(
        get_nested(config, "cluster.nearby_weighted_majority_min_share") or 0.42
    )
    weight_epsilon = float(get_nested(config, "cluster.nearby_weight_epsilon_ft") or 250.0)
    representatives = representative_by_cluster(assignments)

    assignment_cols = [
        "well_id",
        "cluster_id",
        "cluster_size",
        "representative_well_id",
    ]
    joined = geometry.merge(
        assignments[assignment_cols].rename(columns={"well_id": "well"}),
        on="well",
        how="left",
        validate="one_to_one",
    )
    joined["cluster_size"] = (
        pd.to_numeric(joined["cluster_size"], errors="coerce").fillna(0).astype(int)
    )
    joined["cluster_id"] = joined["cluster_id"].astype("string")
    joined["representative_well_id"] = joined["representative_well_id"].astype("string")
    valid_cluster = joined["cluster_id"].notna() & (joined["cluster_size"] >= min_cluster_size)

    cluster_stats_rows: list[dict[str, Any]] = []
    for cluster_id, group in joined[valid_cluster].groupby("cluster_id", sort=False):
        x = numeric_array(group, "centroid_x").astype(np.float64)
        y = numeric_array(group, "centroid_y").astype(np.float64)
        center_x = float(np.nanmedian(x))
        center_y = float(np.nanmedian(y))
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        median_dist = float(np.nanmedian(dist))
        scale = robust_scale(dist, scale_floor)
        cluster_stats_rows.append(
            {
                "cluster_id": str(cluster_id),
                "cluster_center_x": center_x,
                "cluster_center_y": center_y,
                "cluster_member_wells": int(len(group)),
                "cluster_dist_median": median_dist,
                "cluster_dist_scale": scale,
                "cluster_dist_p90": float(np.nanquantile(dist, 0.90)) if len(dist) else np.nan,
            }
        )
    cluster_stats = pd.DataFrame(cluster_stats_rows)
    joined = joined.merge(cluster_stats, on="cluster_id", how="left")

    x = numeric_array(joined, "centroid_x").astype(np.float64)
    y = numeric_array(joined, "centroid_y").astype(np.float64)
    cx = numeric_array(joined, "cluster_center_x").astype(np.float64)
    cy = numeric_array(joined, "cluster_center_y").astype(np.float64)
    joined["own_cluster_dist"] = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float32)
    joined["own_cluster_dist_z"] = (
        (numeric_array(joined, "own_cluster_dist") - numeric_array(joined, "cluster_dist_median"))
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
    joined["nearest_other_cluster_dist"] = np.asarray(nearest_dist, dtype=np.float32)
    joined["nearest_other_closer"] = numeric_array(
        joined,
        "nearest_other_cluster_dist",
    ) < numeric_array(joined, "own_cluster_dist")
    joined["nearest_other_cluster_rep_well"] = [
        representatives.get(str(cluster_id)) if cluster_id is not None else None
        for cluster_id in nearest_ids
    ]

    dist_matrix = np.sqrt(np.sum((well_xy[:, None, :] - well_xy[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    cluster_values = joined["cluster_id"].astype("string").to_numpy()
    for k in nearby_k_values:
        majority_clusters: list[str | None] = []
        majority_reps: list[str | None] = []
        majority_counts: list[int] = []
        majority_weight_shares: list[float] = []
        diff_flags: list[bool] = []
        for i in range(len(joined)):
            finite_neighbor = np.isfinite(dist_matrix[i])
            if not finite_neighbor.any():
                majority_clusters.append(None)
                majority_reps.append(None)
                majority_counts.append(0)
                majority_weight_shares.append(0.0)
                diff_flags.append(False)
                continue
            idx = np.argsort(dist_matrix[i])[:k]
            weighted: dict[str, float] = {}
            counts: dict[str, int] = {}
            total_weight = 0.0
            for j in idx:
                cluster_id = cluster_values[j]
                if pd.isna(cluster_id):
                    continue
                cluster_key = str(cluster_id)
                weight = 1.0 / (float(dist_matrix[i, j]) + weight_epsilon)
                weighted[cluster_key] = weighted.get(cluster_key, 0.0) + weight
                counts[cluster_key] = counts.get(cluster_key, 0) + 1
                total_weight += weight
            if not weighted or total_weight <= 0.0:
                majority_clusters.append(None)
                majority_reps.append(None)
                majority_counts.append(0)
                majority_weight_shares.append(0.0)
                diff_flags.append(False)
                continue
            majority_cluster = max(weighted.items(), key=lambda item: item[1])[0]
            share = float(weighted[majority_cluster] / total_weight)
            own_cluster = None if pd.isna(cluster_values[i]) else str(cluster_values[i])
            majority_clusters.append(majority_cluster)
            majority_reps.append(representatives.get(majority_cluster))
            majority_counts.append(int(counts[majority_cluster]))
            majority_weight_shares.append(share)
            diff_flags.append(
                bool(
                    own_cluster is not None
                    and majority_cluster != own_cluster
                    and share >= majority_min_share
                )
            )
        joined[f"nearby_weighted_majority_cluster_k{k}"] = majority_clusters
        joined[f"nearby_weighted_majority_rep_well_k{k}"] = majority_reps
        joined[f"nearby_weighted_majority_count_k{k}"] = majority_counts
        joined[f"nearby_weighted_majority_share_k{k}"] = majority_weight_shares
        joined[f"nearby_weighted_majority_diff_k{k}"] = diff_flags

    joined["cluster_feature_valid"] = valid_cluster.to_numpy()
    metadata = {
        "assignments": assignment_meta,
        "geometry": geometry_meta,
        "min_cluster_size": min_cluster_size,
        "robust_scale_floor_ft": scale_floor,
        "nearby_k_values": nearby_k_values,
        "nearby_weight_epsilon_ft": weight_epsilon,
        "nearby_weighted_majority_min_share": majority_min_share,
        "cluster_feature_wells": int(joined["cluster_feature_valid"].sum()),
        "clusters_used": int(cluster_stats["cluster_id"].nunique()) if len(cluster_stats) else 0,
    }
    return joined, metadata


def target_gate_mask(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    gate = get_nested(config, "cluster.target_gate") or {}
    base = frame["cluster_feature_valid"].fillna(False).to_numpy(bool)
    components: list[np.ndarray] = []
    if gate.get("own_cluster_dist_z_gt") is not None:
        components.append(
            numeric_array(frame, "own_cluster_dist_z") > float(gate["own_cluster_dist_z_gt"])
        )
    if bool(gate.get("nearest_other_closer", False)):
        components.append(frame["nearest_other_closer"].fillna(False).to_numpy(bool))
    if bool(gate.get("nearby_weighted_majority_diff", False)):
        k = int(gate.get("nearby_k", 8))
        column = f"nearby_weighted_majority_diff_k{k}"
        if column in frame.columns:
            components.append(frame[column].fillna(False).to_numpy(bool))
    if not components:
        return base
    stacked = np.vstack(components)
    mode = str(gate.get("mode", "any"))
    selected = np.all(stacked, axis=0) if mode == "all" else np.any(stacked, axis=0)
    return base & selected


def is_truthy(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def typewell_exists(train_dir: Path, well: Any) -> bool:
    if not is_truthy(well):
        return False
    return (train_dir / f"{str(well)}__typewell.csv").exists()


def select_target_wells(
    cluster_features: pd.DataFrame,
    train_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    include = [
        str(value)
        for value in (get_nested(config, "model.validation_surface.well_include") or [])
        if value
    ]
    members_by_cluster = cluster_members_by_cluster(config, train_dir)
    frame = cluster_features.copy()
    frame["target_gate"] = target_gate_mask(frame, config)
    k = int(get_nested(config, "cluster.target_gate.nearby_k") or 8)
    nearby_diff = f"nearby_weighted_majority_diff_k{k}"
    nearby_diff_signal = frame.get(nearby_diff, pd.Series(False, index=frame.index))
    frame["target_outlier_score"] = (
        pd.to_numeric(frame["own_cluster_dist_z"], errors="coerce").fillna(-999.0)
        + 1.0 * frame["nearest_other_closer"].fillna(False).astype(float)
        + 1.0 * nearby_diff_signal.fillna(False).astype(float)
    )
    frame["own_typewell_exists"] = frame["well"].map(lambda well: typewell_exists(train_dir, well))
    frame["nearest_other_typewell_exists"] = frame["nearest_other_cluster_id"].map(
        lambda cluster_id: cluster_typewell_available(members_by_cluster, cluster_id)
    )
    nearby_rep = f"nearby_weighted_majority_rep_well_k{k}"
    nearby_cluster = f"nearby_weighted_majority_cluster_k{k}"
    frame["nearby_majority_typewell_exists"] = frame[nearby_cluster].map(
        lambda cluster_id: cluster_typewell_available(members_by_cluster, cluster_id)
    )
    frame["nearby_majority_rep_well"] = frame.get(nearby_rep)
    if include:
        selected = frame[frame["well"].isin(include)].copy()
    else:
        selected = frame[
            frame["target_gate"]
            & frame["own_typewell_exists"]
            & (frame["nearest_other_typewell_exists"] | frame["nearby_majority_typewell_exists"])
        ].copy()
        selected = selected.sort_values(
            ["target_outlier_score", "own_cluster_dist_z"],
            ascending=[False, False],
            na_position="last",
        )
        max_target_wells = get_nested(config, "model.validation_surface.max_target_wells")
        if max_target_wells is not None:
            selected = selected.head(int(max_target_wells))
    return selected.reset_index(drop=True)


def load_typewell(path: Path) -> pd.DataFrame:
    typewell = pd.read_csv(path, low_memory=False).sort_values("TVT").reset_index(drop=True)
    if len(typewell) < 3:
        raise ValueError(f"typewell must contain at least 3 rows: {path}")
    return typewell


def build_composite_typewell(
    source_wells: tuple[str, ...],
    train_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    composite_cfg = get_nested(config, "model.composite_typewell") or {}
    tvt_bin_ft = float(composite_cfg.get("tvt_bin_ft", 1.0))
    min_rows = int(composite_cfg.get("min_rows", 3))
    frames: list[pd.DataFrame] = []
    for well in source_wells:
        path = train_dir / f"{well}__typewell.csv"
        if not path.exists():
            continue
        typewell = pd.read_csv(path, usecols=["TVT", "GR"], low_memory=False)
        typewell["TVT"] = pd.to_numeric(typewell["TVT"], errors="coerce")
        typewell["GR"] = pd.to_numeric(typewell["GR"], errors="coerce")
        typewell = typewell.dropna(subset=["TVT", "GR"]).copy()
        if typewell.empty:
            continue
        typewell["source_well"] = well
        frames.append(typewell)
    if not frames:
        raise ValueError("no source typewell rows available for composite")

    combined = pd.concat(frames, ignore_index=True)
    if tvt_bin_ft > 0.0:
        combined["tvt_bin"] = np.round(combined["TVT"] / tvt_bin_ft) * tvt_bin_ft
    else:
        combined["tvt_bin"] = combined["TVT"].round(6)
    composite = (
        combined.groupby("tvt_bin", as_index=False)
        .agg(
            TVT=("TVT", "median"),
            GR=("GR", "median"),
            source_row_count=("GR", "size"),
            source_well_count=("source_well", "nunique"),
        )
        .sort_values("TVT")
        .reset_index(drop=True)
    )
    composite = composite[np.isfinite(composite["TVT"]) & np.isfinite(composite["GR"])].copy()
    composite = composite.drop_duplicates(subset=["TVT"], keep="first").reset_index(drop=True)
    if len(composite) < min_rows:
        raise ValueError(
            f"composite typewell has too few rows: rows={len(composite)} wells={len(source_wells)}"
        )
    return composite


def strategy_cache_key(strategy: TypewellStrategy) -> str:
    if strategy.source_path is not None:
        return f"path:{strategy.source_path}"
    wells_key = ",".join(strategy.source_wells)
    wells_digest = hashlib.sha256(wells_key.encode()).hexdigest()
    return f"composite:{strategy.source_cluster_id}:{wells_digest}"


def load_strategy_typewell(
    strategy: TypewellStrategy,
    train_dir: Path,
    config: dict[str, Any],
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    key = strategy_cache_key(strategy)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if strategy.source_path is not None:
        typewell = load_typewell(strategy.source_path)
    else:
        typewell = build_composite_typewell(strategy.source_wells, train_dir, config)
    cache[key] = typewell
    return typewell


def build_typewell_strategies(
    row: pd.Series,
    train_dir: Path,
    config: dict[str, Any],
) -> list[TypewellStrategy]:
    requested = set(get_nested(config, "model.typewell_strategies") or [])
    members_by_cluster = cluster_members_by_cluster(config, train_dir)
    strategies: list[TypewellStrategy] = []
    well = str(row["well"])
    own_cluster = None if pd.isna(row.get("cluster_id")) else str(row.get("cluster_id"))
    own_path = train_dir / f"{well}__typewell.csv"
    if own_path.exists() and (not requested or "own_typewell" in requested):
        strategies.append(
            TypewellStrategy(
                name="own_typewell",
                kind="own",
                source_cluster_id=own_cluster,
                source_well=well,
                source_path=own_path,
                source_wells=(well,),
                description="query well own typewell",
            )
        )

    if not requested or "nearest_other_cluster_composite" in requested:
        source_cluster = row.get("nearest_other_cluster_id")
        source_cluster_id = None if pd.isna(source_cluster) else str(source_cluster)
        source_wells = members_by_cluster.get(source_cluster_id or "", ())
        if source_cluster_id is not None and source_wells:
            representative = str(row.get("nearest_other_cluster_rep_well") or source_wells[0])
            strategies.append(
                TypewellStrategy(
                    name="nearest_other_cluster_composite",
                    kind="nearest_other_cluster_composite",
                    source_cluster_id=source_cluster_id,
                    source_well=representative,
                    source_path=None,
                    source_wells=source_wells,
                    description=(
                        "TVT-binned composite typewell built from all available members of "
                        "the nearest other cluster"
                    ),
                )
            )

    k = int(get_nested(config, "cluster.target_gate.nearby_k") or 8)
    nearby_name = f"nearby_majority_cluster_composite_k{k}"
    if not requested or nearby_name in requested:
        source_cluster = row.get(f"nearby_weighted_majority_cluster_k{k}")
        diff = is_truthy(row.get(f"nearby_weighted_majority_diff_k{k}", False))
        source_cluster_id = None if pd.isna(source_cluster) else str(source_cluster)
        source_wells = members_by_cluster.get(source_cluster_id or "", ())
        if diff and source_cluster_id is not None and source_wells:
            representative = str(
                row.get(f"nearby_weighted_majority_rep_well_k{k}") or source_wells[0]
            )
            strategies.append(
                TypewellStrategy(
                    name=nearby_name,
                    kind="nearby_weighted_majority_cluster_composite",
                    source_cluster_id=source_cluster_id,
                    source_well=representative,
                    source_path=None,
                    source_wells=source_wells,
                    description=(
                        "TVT-binned composite typewell built from all available members of "
                        f"the nearby weighted-majority cluster k={k}"
                    ),
                )
            )
    return strategies


def build_eval_zone_for_well(
    well: str,
    eval_cache_rows: pd.DataFrame,
    train_dir: Path,
    config: dict[str, Any],
) -> PrefixHoldout | None:
    validation_cfg = get_nested(config, "model.validation_surface") or {}
    min_known = int(validation_cfg.get("min_known_prefix_rows", 160))
    min_eval = int(validation_cfg.get("min_eval_rows", 64))

    hw_path = train_dir / f"{well}__horizontal_well.csv"
    if not hw_path.exists() or eval_cache_rows.empty:
        return None
    horizontal = pd.read_csv(hw_path, low_memory=False)

    cache_rows = eval_cache_rows.copy()
    if "row_idx" not in cache_rows.columns:
        cache_rows["row_idx"] = row_indices_from_ids(cache_rows["id"]).astype(np.int32)
    cache_rows = cache_rows.sort_values("row_idx").reset_index(drop=True)
    eval_index = pd.to_numeric(cache_rows["row_idx"], errors="coerce").to_numpy(np.int64)
    valid_index = (eval_index >= 0) & (eval_index < len(horizontal))
    if not valid_index.all():
        cache_rows = cache_rows.loc[valid_index].reset_index(drop=True)
        eval_index = eval_index[valid_index]
    if len(eval_index) < min_eval:
        return None

    known_mask = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
    known_idx = np.flatnonzero(known_mask)
    if len(known_idx) < min_known:
        return None
    prefix_end = int(eval_index[0])
    known_before = known_idx[known_idx < prefix_end]
    if len(known_before) < min_known:
        return None
    last_known_idx = int(known_before[-1])
    masked = horizontal.iloc[: int(eval_index[-1]) + 1].copy()

    target_delta = numeric_array(cache_rows, "target")
    last_known_values = numeric_array(cache_rows, "last_known_tvt")
    last_known_tvt = float(last_known_values[0])
    truth = (last_known_values + target_delta).astype(np.float32)
    last_known_md = float(horizontal.loc[last_known_idx, "MD"])
    md_since = numeric_array(cache_rows, "md_since") if "md_since" in cache_rows.columns else None
    status = {
        "well": well,
        "status": "ok",
        "validation_surface": "exp072_TVT_input_missing_equivalent_rows",
        "known_rows": int(len(known_idx)),
        "eval_rows": int(len(eval_index)),
        "last_known_idx": int(last_known_idx),
        "last_known_tvt": last_known_tvt,
        "last_known_md": last_known_md,
        "raw_eval_tvt_input_missing_rate": float(
            pd.to_numeric(horizontal.loc[eval_index, "TVT_input"], errors="coerce")
            .isna()
            .mean()
        ),
    }
    return PrefixHoldout(
        well=well,
        masked=masked,
        eval_index=eval_index,
        eval_ids=cache_rows["id"].astype(str).to_numpy(),
        true_tvt=truth,
        target_delta=target_delta,
        last_known_tvt=last_known_tvt,
        last_known_md=last_known_md,
        cache_md_since=md_since,
        status=status,
    )


def gr_sigma(
    prefix: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    config: dict[str, Any],
) -> float:
    runtime = get_nested(config, "model.runtime") or {}
    finite = prefix["TVT_input"].notna() & prefix["GR"].notna()
    if int(finite.sum()) < 20:
        return float(runtime.get("gr_sigma_default", 30.0))
    residual = pd.to_numeric(prefix.loc[finite, "GR"], errors="coerce").to_numpy(
        np.float64
    ) - np.interp(
        pd.to_numeric(prefix.loc[finite, "TVT_input"], errors="coerce").to_numpy(np.float64),
        tw_tvt,
        tw_gr,
    )
    return float(
        np.clip(
            np.nanstd(residual),
            float(runtime.get("gr_sigma_min", 10.0)),
            float(runtime.get("gr_sigma_max", 60.0)),
        )
    )


def initial_velocity(prefix: pd.DataFrame) -> float:
    tail = prefix.tail(30)
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    dm = np.diff(md)
    dt = np.diff(tvt)
    finite = np.isfinite(dm) & np.isfinite(dt) & (dm > 0.0)
    if int(finite.sum()) < 3:
        return 0.0
    return float(np.median(dt[finite] / dm[finite]))


def systematic_resample(
    rng: np.random.Generator,
    pos: np.ndarray,
    vel: np.ndarray,
    weights: np.ndarray,
    pos_noise: float,
    vel_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(pos)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    positions = rng.uniform(0.0, 1.0 / n) + np.arange(n, dtype=np.float64) / n
    idx = np.searchsorted(cdf, positions, side="left")
    return (
        pos[idx] + pos_noise * rng.standard_normal(n),
        vel[idx] + vel_noise * rng.standard_normal(n),
    )


def run_pf_for_holdout(
    holdout: PrefixHoldout,
    strategy: TypewellStrategy,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> PfRun:
    runtime = get_nested(config, "model.runtime") or {}
    n_particles = int(runtime.get("particles", 260))
    seed_count = int(runtime.get("seed_count", 8))
    temperature = float(runtime.get("likelihood_temperature", 6.0))
    resample_threshold = float(runtime.get("resample_threshold", 0.5))
    init_spread = float(runtime.get("init_spread", 4.5))
    velocity_noise = float(runtime.get("velocity_noise", 0.002))
    position_noise = float(runtime.get("position_noise", 0.005))
    resample_pos_noise = float(runtime.get("resample_pos_noise", 0.10))
    resample_velocity_noise = float(runtime.get("resample_velocity_noise", 0.001))

    hw = holdout.masked
    tw = typewell
    tw_tvt = numeric_array(tw, "TVT").astype(np.float64)
    tw_gr = fill_numeric(tw["GR"], float(np.nanmean(numeric_array(tw, "GR")))).astype(np.float64)
    eval_rows = hw.loc[holdout.eval_index].copy()
    md = numeric_array(eval_rows, "MD").astype(np.float64)
    gr = fill_numeric(eval_rows["GR"], float(np.nanmean(tw_gr))).astype(np.float64)
    prefix = hw.loc[: int(holdout.eval_index[0]) - 1]
    sigma = gr_sigma(prefix, tw_tvt, tw_gr, config)
    init_vel = initial_velocity(prefix)
    tmin = float(tw_tvt.min())
    tmax = float(tw_tvt.max())

    preds = np.empty((seed_count, len(eval_rows)), dtype=np.float32)
    log_likelihoods = np.empty(seed_count, dtype=np.float64)
    ess_accum = np.zeros(len(eval_rows), dtype=np.float64)
    resampled_accum = np.zeros(len(eval_rows), dtype=np.float64)

    for seed_index in range(seed_count):
        rng = np.random.default_rng(stable_seed(EXPERIMENT_NAME, holdout.well, "pf", seed_index))
        pos = holdout.last_known_tvt + init_spread * rng.standard_normal(n_particles)
        vel = init_vel + 0.01 * rng.standard_normal(n_particles)
        weights = np.full(n_particles, 1.0 / n_particles, dtype=np.float64)
        prev_md = holdout.last_known_md
        log_lik = 0.0
        for row_pos, (row_md, row_gr) in enumerate(zip(md, gr, strict=True)):
            delta_md = max(float(row_md - prev_md), 1.0)
            vel = 0.998 * vel + velocity_noise * rng.standard_normal(n_particles)
            pos = pos + vel * delta_md + position_noise * rng.standard_normal(n_particles)
            pos = np.clip(pos, tmin - 100.0, tmax + 100.0)

            expected_gr = np.interp(pos, tw_tvt, tw_gr)
            residual = (float(row_gr) - expected_gr) / max(sigma, 1e-6)
            residual2 = np.minimum(residual * residual, 600.0)
            likelihood = np.exp(-0.5 * residual2)
            likelihood = np.maximum(likelihood, 1e-300)
            avg_likelihood = float(np.dot(weights, likelihood))
            log_lik += float(np.log(max(avg_likelihood, 1e-300)))

            weights = weights * likelihood
            weight_sum = float(weights.sum())
            if weight_sum > 0.0 and np.isfinite(weight_sum):
                weights = weights / weight_sum
            else:
                weights.fill(1.0 / n_particles)

            ess = 1.0 / max(float(np.dot(weights, weights)), 1e-300)
            ess_accum[row_pos] += ess
            if ess < resample_threshold * n_particles:
                pos, vel = systematic_resample(
                    rng,
                    pos,
                    vel,
                    weights,
                    resample_pos_noise,
                    resample_velocity_noise,
                )
                weights.fill(1.0 / n_particles)
                resampled_accum[row_pos] += 1.0

            preds[seed_index, row_pos] = np.float32(np.dot(weights, pos))
            prev_md = float(row_md)
        log_likelihoods[seed_index] = log_lik

    centered = log_likelihoods - float(np.max(log_likelihoods))
    seed_weights = np.exp(centered / max(temperature, 1e-6))
    seed_weights = seed_weights / max(float(seed_weights.sum()), 1e-300)
    return PfRun(
        preds=preds,
        log_likelihoods=log_likelihoods,
        ess_mean_by_row=(ess_accum / seed_count).astype(np.float32),
        resampled_by_row=(resampled_accum / seed_count).astype(np.float32),
        seed_weights=seed_weights.astype(np.float32),
    )


def beam_search_for_holdout(
    holdout: PrefixHoldout,
    strategy: TypewellStrategy,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> np.ndarray:
    beam_cfg = get_nested(config, "model.beam") or {}
    beam_size = int(beam_cfg.get("beam_size", 14))
    move_radius = int(beam_cfg.get("move_radius", 2))
    move_cost = float(beam_cfg.get("move_cost", 16.0))
    error_scale = float(beam_cfg.get("error_scale", 120.0))
    smooth_window = int(beam_cfg.get("smooth_window", 5))

    hw = holdout.masked
    tw = typewell
    tw_tvt = numeric_array(tw, "TVT").astype(np.float64)
    tw_gr = fill_numeric(tw["GR"], float(np.nanmean(numeric_array(tw, "GR")))).astype(np.float64)
    tw_gr = smooth(tw_gr, smooth_window, float(np.nanmean(tw_gr))).astype(np.float64)
    eval_rows = hw.loc[holdout.eval_index].copy()
    gr = smooth(numeric_array(eval_rows, "GR"), smooth_window, float(np.nanmean(tw_gr))).astype(
        np.float64
    )

    start_idx = nearest_index(tw_tvt, holdout.last_known_tvt)
    active: dict[int, tuple[float, list[int]]] = {start_idx: (0.0, [])}
    for row_gr in gr:
        candidates: dict[int, tuple[float, list[int]]] = {}
        for idx, (cost, path) in active.items():
            for delta in range(-move_radius, move_radius + 1):
                next_idx = int(np.clip(idx + delta, 0, len(tw_tvt) - 1))
                gr_cost = ((float(row_gr) - float(tw_gr[next_idx])) ** 2) / max(
                    error_scale,
                    1e-6,
                )
                total = cost + gr_cost + move_cost * abs(delta)
                previous = candidates.get(next_idx)
                if previous is None or total < previous[0]:
                    candidates[next_idx] = (total, [*path, next_idx])
        kept = sorted(candidates.items(), key=lambda item: item[1][0])[:beam_size]
        active = {idx: value for idx, value in kept}
    if not active:
        return np.full(len(eval_rows), holdout.last_known_tvt, dtype=np.float32)
    _, (_, best_path) = min(active.items(), key=lambda item: item[1][0])
    return tw_tvt[np.asarray(best_path, dtype=np.int64)].astype(np.float32)


def score_prediction(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(pred) & np.isfinite(true)
    if not finite.any():
        return {
            "rows": 0,
            "coverage": 0.0,
            "rmse": None,
            "mae": None,
            "within10": None,
            "bias": None,
        }
    err = pred[finite].astype(np.float64) - true[finite].astype(np.float64)
    return {
        "rows": int(finite.sum()),
        "coverage": float(finite.mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "within10": float(np.mean(np.abs(err) <= 10.0)),
        "bias": float(np.mean(err)),
    }


def distance_bucket(md_since: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(pd.Series(md_since), errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def candidate_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if (column.startswith("pf_") or column.startswith("beam_")) and not column.endswith("_diag")
    ]


def path_jump_rate_for_candidate(
    frame: pd.DataFrame,
    column: str,
    threshold_ft: float,
) -> float | None:
    jumps = 0
    total = 0
    for _, group in frame.sort_values(["well", "row_idx"]).groupby("well", sort=False):
        values = pd.to_numeric(group[column], errors="coerce").to_numpy(np.float64)
        finite = np.isfinite(values)
        if finite.sum() < 2:
            continue
        diffs = np.abs(np.diff(values[finite]))
        jumps += int(np.sum(diffs > threshold_ft))
        total += int(len(diffs))
    if total == 0:
        return None
    return float(jumps / total)


def compute_candidate_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    primary = str(get_nested(config, "audit.primary_baseline") or "pf_own_typewell_lik_mean")
    baseline = score_prediction(numeric_array(frame, primary), true) if primary in frame else None
    threshold = float(get_nested(config, "audit.path_jump_threshold_ft") or 8.0)
    rows: list[dict[str, Any]] = []
    for column in candidate_columns(frame):
        pred = numeric_array(frame, column)
        score = score_prediction(pred, true)
        delta = None
        if baseline and score["rmse"] is not None and baseline["rmse"] is not None:
            delta = float(score["rmse"] - baseline["rmse"])
        rows.append(
            {
                "candidate": column,
                "is_oracle_diagnostic": bool("_oracle" in column),
                **score,
                "delta_rmse_vs_primary_baseline": delta,
                "path_jump_rate": path_jump_rate_for_candidate(frame, column, threshold),
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "candidate"], na_position="last")


def compute_strategy_delta_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    specs = [
        ("pf_lik_mean", "pf_own_typewell_lik_mean", "_lik_mean"),
        ("pf_best_seed", "pf_own_typewell_best_seed", "_best_seed"),
        ("pf_top3_oracle", "pf_own_typewell_top3_oracle", "_top3_oracle"),
        ("beam_top1", "beam_own_typewell_top1", "_top1"),
    ]
    strategies = sorted(
        {
            column.removeprefix("pf_").removesuffix("_lik_mean")
            for column in frame.columns
            if column.startswith("pf_") and column.endswith("_lik_mean")
        }
    )
    rows: list[dict[str, Any]] = []
    for family, own_column, suffix in specs:
        if own_column not in frame:
            continue
        own_score = score_prediction(numeric_array(frame, own_column), true)
        for strategy in strategies:
            if strategy == "own_typewell":
                continue
            column = (
                f"pf_{strategy}{suffix}" if family.startswith("pf_") else f"beam_{strategy}{suffix}"
            )
            if column not in frame:
                continue
            score = score_prediction(numeric_array(frame, column), true)
            own_values = numeric_array(frame, own_column)
            alt_values = numeric_array(frame, column)
            diff = alt_values.astype(np.float64) - own_values.astype(np.float64)
            finite = np.isfinite(diff)
            delta = None
            if score["rmse"] is not None and own_score["rmse"] is not None:
                delta = float(score["rmse"] - own_score["rmse"])
            rows.append(
                {
                    "family": family,
                    "strategy": strategy,
                    "candidate": column,
                    "own_candidate": own_column,
                    **score,
                    "own_rmse": own_score["rmse"],
                    "delta_rmse_vs_own_family": delta,
                    "row_abs_diff_mean_vs_own": (
                        float(np.mean(np.abs(diff[finite]))) if finite.any() else None
                    ),
                    "row_diff_rmse_vs_own": (
                        float(np.sqrt(np.mean(diff[finite] * diff[finite])))
                        if finite.any()
                        else None
                    ),
                    "changed_rows_vs_own": int(np.sum(np.abs(diff[finite]) > 1.0e-6))
                    if finite.any()
                    else 0,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["delta_rmse_vs_own_family", "candidate"],
        na_position="last",
    )


def compute_bucket_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    buckets = distance_bucket(frame["md_since"])
    rows: list[dict[str, Any]] = []
    for column in candidate_columns(frame):
        pred = numeric_array(frame, column)
        for bucket in pd.Series(buckets).cat.categories:
            mask = np.asarray(buckets == bucket, dtype=bool)
            if not mask.any():
                continue
            rows.append(
                {
                    "candidate": column,
                    "distance_bucket": str(bucket),
                    **score_prediction(pred[mask], true[mask]),
                }
            )
    return pd.DataFrame(rows)


def compute_by_well(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    primary = str(get_nested(config, "audit.primary_baseline") or "pf_own_typewell_lik_mean")
    rows: list[dict[str, Any]] = []
    well_meta = (
        frame.sort_values(["well", "row_idx"]).groupby("well", sort=False).head(1).set_index("well")
    )
    for well, group in frame.groupby("well", sort=False):
        true = numeric_array(group, "true_tvt")
        base_rmse = None
        if primary in group:
            base_rmse = score_prediction(numeric_array(group, primary), true)["rmse"]
        for column in candidate_columns(group):
            score = score_prediction(numeric_array(group, column), true)
            delta = None
            if base_rmse is not None and score["rmse"] is not None:
                delta = float(score["rmse"] - base_rmse)
            meta = well_meta.loc[well]
            rows.append(
                {
                    "well": str(well),
                    "candidate": column,
                    "own_cluster_id": meta.get("own_cluster_id"),
                    "own_cluster_dist_z": meta.get("own_cluster_dist_z"),
                    "nearest_other_cluster_id": meta.get("nearest_other_cluster_id"),
                    "nearby_weighted_majority_cluster_k8": meta.get(
                        "nearby_weighted_majority_cluster_k8"
                    ),
                    **score,
                    "delta_rmse_vs_primary_baseline": delta,
                }
            )
    return pd.DataFrame(rows)


def compute_group_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    k8_diff_col = "nearby_weighted_majority_diff_k8"
    groups: dict[str, np.ndarray] = {
        "all": np.ones(len(frame), dtype=bool),
        "near_000_050": numeric_array(frame, "md_since") <= 50.0,
        "longtail_1000_plus": numeric_array(frame, "md_since") >= 1000.0,
        "own_cluster_z_gt1p5": numeric_array(frame, "own_cluster_dist_z") > 1.5,
        "nearest_other_closer": frame["nearest_other_closer"].fillna(False).to_numpy(bool),
        "nearby_weighted_majority_diff_k8": frame[k8_diff_col].fillna(False).to_numpy(bool)
        if k8_diff_col in frame.columns
        else np.zeros(len(frame), dtype=bool),
    }
    rows: list[dict[str, Any]] = []
    true = numeric_array(frame, "true_tvt")
    for group_name, mask in groups.items():
        if not mask.any():
            continue
        for column in candidate_columns(frame):
            rows.append(
                {
                    "group": group_name,
                    "candidate": column,
                    **score_prediction(numeric_array(frame, column)[mask], true[mask]),
                }
            )
    return pd.DataFrame(rows)


def add_worst_well_regression(candidate_metrics: pd.DataFrame, by_well: pd.DataFrame) -> None:
    if by_well.empty:
        candidate_metrics["max_well_regression_vs_primary"] = np.nan
        return
    max_regression = by_well.groupby("candidate", observed=True)[
        "delta_rmse_vs_primary_baseline"
    ].max()
    candidate_metrics["max_well_regression_vs_primary"] = candidate_metrics["candidate"].map(
        max_regression
    )


def build_row_frame_for_holdout(
    holdout: PrefixHoldout,
    cluster_row: pd.Series,
    strategies: list[TypewellStrategy],
    pf_outputs: dict[str, PfRun],
    beam_outputs: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    eval_rows = holdout.masked.loc[holdout.eval_index]
    md_since = (
        holdout.cache_md_since
        if holdout.cache_md_since is not None
        else numeric_array(eval_rows, "MD") - np.float32(holdout.last_known_md)
    )
    row_frame = pd.DataFrame(
        {
            "well": holdout.well,
            "row_idx": holdout.eval_index.astype(np.int32),
            "id": holdout.eval_ids.astype(str),
            "target": holdout.target_delta.astype(np.float32),
            "true_tvt": holdout.true_tvt.astype(np.float32),
            "last_known_tvt": np.float32(holdout.last_known_tvt),
            "last_known_md": np.float32(holdout.last_known_md),
            "md_since": md_since.astype(np.float32),
            "own_cluster_id": cluster_row.get("cluster_id"),
            "own_cluster_dist": np.float32(cluster_row.get("own_cluster_dist", np.nan)),
            "own_cluster_dist_z": np.float32(cluster_row.get("own_cluster_dist_z", np.nan)),
            "nearest_other_cluster_id": cluster_row.get("nearest_other_cluster_id"),
            "nearest_other_closer": bool(cluster_row.get("nearest_other_closer", False)),
            "nearest_other_cluster_rep_well": cluster_row.get("nearest_other_cluster_rep_well"),
            "nearby_weighted_majority_cluster_k8": cluster_row.get(
                "nearby_weighted_majority_cluster_k8"
            ),
            "nearby_weighted_majority_rep_well_k8": cluster_row.get(
                "nearby_weighted_majority_rep_well_k8"
            ),
            "nearby_weighted_majority_share_k8": np.float32(
                cluster_row.get("nearby_weighted_majority_share_k8", np.nan)
            ),
            "nearby_weighted_majority_diff_k8": bool(
                cluster_row.get("nearby_weighted_majority_diff_k8", False)
            ),
        }
    )
    top_k = int(get_nested(config, "model.runtime.topk_oracle") or 3)
    diagnostics: list[dict[str, Any]] = []
    for strategy in strategies:
        run = pf_outputs[strategy.name]
        weighted = (run.seed_weights[:, None] * run.preds).sum(axis=0)
        best_idx = int(np.argmax(run.log_likelihoods))
        top_idx = np.argsort(run.log_likelihoods)[::-1][:top_k]
        oracle = np.empty(len(row_frame), dtype=np.float32)
        truth = row_frame["true_tvt"].to_numpy(np.float32)
        for i in range(len(row_frame)):
            seed_values = run.preds[top_idx, i]
            oracle[i] = seed_values[np.argmin(np.abs(seed_values - truth[i]))]
        row_frame[f"pf_{strategy.name}_lik_mean"] = weighted.astype(np.float32)
        row_frame[f"pf_{strategy.name}_best_seed"] = run.preds[best_idx].astype(np.float32)
        row_frame[f"pf_{strategy.name}_top{top_k}_oracle"] = oracle
        row_frame[f"pf_{strategy.name}_ess_mean_diag"] = run.ess_mean_by_row.astype(np.float32)
        row_frame[f"pf_{strategy.name}_resampled_rate_diag"] = run.resampled_by_row.astype(
            np.float32
        )
        diagnostics.append(
            {
                "well": holdout.well,
                "strategy": strategy.name,
                "strategy_kind": strategy.kind,
                "source_well": strategy.source_well,
                "source_cluster_id": strategy.source_cluster_id,
                "seed_count": int(run.preds.shape[0]),
                "rows": int(run.preds.shape[1]),
                "log_likelihood_mean": float(np.mean(run.log_likelihoods)),
                "log_likelihood_std": float(np.std(run.log_likelihoods)),
                "ess_mean": float(np.mean(run.ess_mean_by_row)),
                "resampling_rate": float(np.mean(run.resampled_by_row)),
                "seed_weight_max": float(np.max(run.seed_weights)),
            }
        )
    for strategy in strategies:
        row_frame[f"beam_{strategy.name}_top1"] = beam_outputs[strategy.name].astype(np.float32)
    return row_frame, diagnostics


def summarize_strategy_sources(
    target_features: pd.DataFrame,
    train_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in target_features.iterrows():
        strategies = build_typewell_strategies(row, train_dir, config)
        for strategy in strategies:
            rows.append(
                {
                    "well": row["well"],
                    "strategy": strategy.name,
                    "strategy_kind": strategy.kind,
                    "source_well": strategy.source_well,
                    "source_cluster_id": strategy.source_cluster_id,
                    "source_path": str(strategy.source_path) if strategy.source_path else None,
                    "source_member_count": int(len(strategy.source_wells)),
                    "source_wells_preview": ",".join(strategy.source_wells[:12]),
                    "source_wells_truncated": len(strategy.source_wells) > 12,
                    "description": strategy.description,
                }
            )
    return pd.DataFrame(rows)


def run_alt_typewell_pfbeam_audit(
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    started = time.time()
    config = load_config() if config is None else config
    paths = ExperimentPaths() if paths is None else paths
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()

    train_dir = paths.train_data_dir
    cluster_features, cluster_meta = build_cluster_features(config)
    target_features = select_target_wells(cluster_features, train_dir, config)
    strategy_sources = summarize_strategy_sources(target_features, train_dir, config)
    validation_frame, validation_meta = read_exp072_eval_cache(config)
    target_well_set = set(target_features["well"].astype(str).tolist())
    validation_frame = validation_frame[validation_frame["well"].isin(target_well_set)].copy()
    validation_rows_by_well = {
        str(well): group.copy() for well, group in validation_frame.groupby("well", sort=False)
    }

    row_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    typewell_cache: dict[str, pd.DataFrame] = {}

    for _, cluster_row in target_features.iterrows():
        well = str(cluster_row["well"])
        holdout = build_eval_zone_for_well(
            well,
            validation_rows_by_well.get(well, pd.DataFrame()),
            train_dir,
            config,
        )
        if holdout is None:
            status_rows.append({"well": well, "status": "skipped_no_valid_exp072_eval_zone"})
            continue
        strategies = build_typewell_strategies(cluster_row, train_dir, config)
        if not any(strategy.name == "own_typewell" for strategy in strategies):
            status_rows.append({"well": well, "status": "skipped_missing_own_typewell"})
            continue
        if len(strategies) < 2:
            status_rows.append({"well": well, "status": "skipped_no_alt_typewell_strategy"})
            continue
        pf_outputs: dict[str, PfRun] = {}
        beam_outputs: dict[str, np.ndarray] = {}
        for strategy in strategies:
            typewell = load_strategy_typewell(strategy, train_dir, config, typewell_cache)
            pf_outputs[strategy.name] = run_pf_for_holdout(holdout, strategy, typewell, config)
            beam_outputs[strategy.name] = beam_search_for_holdout(
                holdout,
                strategy,
                typewell,
                config,
            )
        frame, diag = build_row_frame_for_holdout(
            holdout,
            cluster_row,
            strategies,
            pf_outputs,
            beam_outputs,
            config,
        )
        row_frames.append(frame)
        diagnostics.extend(diag)
        status = dict(holdout.status)
        status.update(
            {
                "strategies": ",".join(strategy.name for strategy in strategies),
                "alt_strategy_count": int(len(strategies) - 1),
            }
        )
        status_rows.append(status)

    if not row_frames:
        raise RuntimeError("No alt-typewell PF/Beam audit rows were generated.")

    row_frame = pd.concat(row_frames, ignore_index=True)
    pf_diagnostics = pd.DataFrame(diagnostics)
    well_status = pd.DataFrame(status_rows)
    candidate_metrics = compute_candidate_metrics(row_frame, config)
    strategy_delta_metrics = compute_strategy_delta_metrics(row_frame)
    bucket_metrics = compute_bucket_metrics(row_frame)
    by_well = compute_by_well(row_frame, config)
    group_metrics = compute_group_metrics(row_frame)
    add_worst_well_regression(candidate_metrics, by_well)

    artifacts = paths.artifacts_dir
    candidate_metrics_path = artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    strategy_delta_path = artifacts / f"{OUTPUT_PREFIX}_strategy_delta_metrics.csv"
    bucket_metrics_path = artifacts / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    group_metrics_path = artifacts / f"{OUTPUT_PREFIX}_group_metrics.csv"
    pf_diagnostics_path = artifacts / f"{OUTPUT_PREFIX}_pf_diagnostics.csv"
    target_features_path = artifacts / f"{OUTPUT_PREFIX}_target_well_features.csv"
    strategy_sources_path = artifacts / f"{OUTPUT_PREFIX}_strategy_sources.csv"
    well_status_path = artifacts / f"{OUTPUT_PREFIX}_well_status.csv"
    row_candidates_path = artifacts / f"{OUTPUT_PREFIX}_row_candidates.csv.gz"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    candidate_metrics.to_csv(candidate_metrics_path, index=False)
    strategy_delta_metrics.to_csv(strategy_delta_path, index=False)
    bucket_metrics.to_csv(bucket_metrics_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    group_metrics.to_csv(group_metrics_path, index=False)
    pf_diagnostics.to_csv(pf_diagnostics_path, index=False)
    target_features.to_csv(target_features_path, index=False)
    strategy_sources.to_csv(strategy_sources_path, index=False)
    well_status.to_csv(well_status_path, index=False)
    row_frame.to_csv(row_candidates_path, index=False, compression="gzip")

    best_row = (
        candidate_metrics.sort_values(["rmse", "candidate"], na_position="last").iloc[0].to_dict()
    )
    deployable_metrics = candidate_metrics[~candidate_metrics["is_oracle_diagnostic"]]
    best_non_oracle_row = (
        deployable_metrics.sort_values(["rmse", "candidate"], na_position="last").iloc[0].to_dict()
    )
    best_alt_delta = (
        strategy_delta_metrics[~strategy_delta_metrics["candidate"].str.contains("_oracle")]
        .sort_values(["delta_rmse_vs_own_family", "candidate"], na_position="last")
        .head(1)
        .to_dict("records")
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_audit",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - started),
        "rows": int(len(row_frame)),
        "wells": int(row_frame["well"].nunique()),
        "target_wells_selected": int(len(target_features)),
        "strategy_rows": int(len(strategy_sources)),
        "primary_baseline": get_nested(config, "audit.primary_baseline"),
        "best_candidate_by_rmse": best_row,
        "best_non_oracle_candidate_by_rmse": best_non_oracle_row,
        "best_alt_delta_vs_own_family": best_alt_delta[0] if best_alt_delta else None,
        "cluster_features": cluster_meta,
        "validation_surface": validation_meta,
        "typewell_cache_entries": int(len(typewell_cache)),
        "pf_diagnostics_summary": (
            pf_diagnostics.groupby("strategy", observed=True)
            .agg(
                wells=("well", "nunique"),
                rows=("rows", "sum"),
                ess_mean=("ess_mean", "mean"),
                resampling_rate=("resampling_rate", "mean"),
                log_likelihood_mean=("log_likelihood_mean", "mean"),
            )
            .reset_index()
            .to_dict("records")
        ),
        "artifacts": {
            "candidate_metrics": str(candidate_metrics_path),
            "strategy_delta_metrics": str(strategy_delta_path),
            "bucket_metrics": str(bucket_metrics_path),
            "by_well": str(by_well_path),
            "group_metrics": str(group_metrics_path),
            "pf_diagnostics": str(pf_diagnostics_path),
            "target_well_features": str(target_features_path),
            "strategy_sources": str(strategy_sources_path),
            "well_status": str(well_status_path),
            "row_candidates": str(row_candidates_path),
            "summary": str(summary_path),
        },
    }
    write_json(summary_path, summary)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_audit",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "rows": int(len(row_frame)),
        "wells": int(row_frame["well"].nunique()),
        "validation_surface": validation_meta,
        "best_candidate_by_rmse": best_row,
        "best_non_oracle_candidate_by_rmse": best_non_oracle_row,
        "best_alt_delta_vs_own_family": summary["best_alt_delta_vs_own_family"],
        "artifacts": summary["artifacts"],
        "sha256": {
            "candidate_metrics": sha256_path(candidate_metrics_path),
            "strategy_delta_metrics": sha256_path(strategy_delta_path),
            "bucket_metrics": sha256_path(bucket_metrics_path),
            "by_well": sha256_path(by_well_path),
            "group_metrics": sha256_path(group_metrics_path),
            "pf_diagnostics": sha256_path(pf_diagnostics_path),
            "target_well_features": sha256_path(target_features_path),
            "strategy_sources": sha256_path(strategy_sources_path),
            "well_status": sha256_path(well_status_path),
            "row_candidates_raw_gzip": sha256_path(row_candidates_path),
            "row_candidates_decompressed": sha256_path(row_candidates_path, decompressed=True),
            "summary": sha256_path(summary_path),
        },
        "notes": (
            "Train-side exp072-aligned alt-typewell PF/Beam audit only; "
            "no model, inference, or submission."
        ),
    }
    write_json(paths.metrics_path, metrics)
    return {
        "summary": summary,
        "candidate_metrics": candidate_metrics,
        "strategy_delta_metrics": strategy_delta_metrics,
        "bucket_metrics": bucket_metrics,
        "by_well": by_well,
        "group_metrics": group_metrics,
        "pf_diagnostics": pf_diagnostics,
        "target_features": target_features,
        "strategy_sources": strategy_sources,
        "well_status": well_status,
        "row_frame": row_frame,
    }


def main() -> dict[str, Any]:
    return run_alt_typewell_pfbeam_audit()


if __name__ == "__main__":
    result = main()
    print(json.dumps(to_jsonable(result["summary"]), indent=2, sort_keys=True))

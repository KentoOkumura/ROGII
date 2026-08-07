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

OUTPUT_PREFIX = "exp109_typewell_neighbor_prior_features"
EXP099_FEATURE_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
EXP099_FEATURE_SCHEMA = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv"
)
EXP065_CLUSTER_ASSIGNMENTS = "common_typewell_cluster_assignments.csv"


@dataclass(frozen=True)
class GroupMethod:
    name: str
    method: str
    threshold: str


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


def find_artifact(filename: str, explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
            Path("artifacts") / filename,
            Path("experiments")
            / "exp099_pf_multi_observation_likelihood_probe"
            / "kaggle"
            / "output"
            / "train_v2"
            / "artifacts"
            / filename,
            Path("experiments")
            / "exp065_typewell_supertype_cluster_cv_audit"
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


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"required column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def read_feature_cache(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    explicit = get_nested(config, "data.exp099_train_feature_cache_local")
    source = find_artifact(EXP099_FEATURE_CACHE, explicit)
    required = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "pf_ancc",
        "beam_mean",
        "likpf_mean",
        "sc_ens",
        "hyb",
        "last_anchor_tvt",
        "eval_len",
        "md_since",
    ]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    max_rows = get_nested(config, "audit.max_rows")
    frame = pd.read_csv(
        source,
        usecols=required,
        nrows=None if max_rows in {None, "null"} else int(max_rows),
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    frame["true_tvt"] = frame["last_known_tvt"] + frame["target"]
    frame["true_delta_from_anchor"] = frame["true_tvt"] - frame["last_known_tvt"]
    schema_path: Path | None = None
    try:
        schema_path = find_artifact(
            EXP099_FEATURE_SCHEMA,
            get_nested(config, "data.exp099_train_feature_schema_local"),
        )
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


def read_cluster_assignments(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    explicit = get_nested(config, "data.exp065_cluster_assignments_local")
    source = find_artifact(EXP065_CLUSTER_ASSIGNMENTS, explicit)
    frame = pd.read_csv(source, dtype=str)
    required = {"method", "threshold", "cluster_id", "well_id", "cluster_size"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    frame["well_id"] = frame["well_id"].astype(str)
    frame["cluster_id"] = frame["cluster_id"].astype(str)
    frame["cluster_size"] = (
        pd.to_numeric(frame["cluster_size"], errors="coerce").fillna(0).astype(int)
    )
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
    }
    return frame, metadata


def parse_group_methods(config: dict[str, Any]) -> list[GroupMethod]:
    raw_methods = get_nested(config, "model.neighbor_prior.group_methods") or []
    methods: list[GroupMethod] = []
    for raw in raw_methods:
        methods.append(
            GroupMethod(
                name=str(raw["name"]),
                method=str(raw["method"]),
                threshold=str(raw["threshold"]),
            )
        )
    if not methods:
        raise ValueError("model.neighbor_prior.group_methods must not be empty")
    return methods


def make_group_lookup(
    assignments: pd.DataFrame,
    method: GroupMethod,
    *,
    min_cluster_size: int,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int]]:
    subset = assignments[
        (assignments["method"].astype(str) == method.method)
        & (assignments["threshold"].astype(str) == method.threshold)
        & (assignments["cluster_size"] >= min_cluster_size)
    ].copy()
    well_to_cluster = dict(zip(subset["well_id"], subset["cluster_id"], strict=False))
    cluster_to_wells = {
        cluster: sorted(group["well_id"].astype(str).tolist())
        for cluster, group in subset.groupby("cluster_id", sort=False)
    }
    cluster_sizes = {cluster: len(wells) for cluster, wells in cluster_to_wells.items()}
    return well_to_cluster, cluster_to_wells, cluster_sizes


def groupkfold_wells(wells: np.ndarray, n_folds: int, seed: int) -> list[tuple[set[str], set[str]]]:
    wells = np.array(sorted(map(str, wells)))
    rng = np.random.default_rng(seed)
    shuffled = wells.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, n_folds)
    splits: list[tuple[set[str], set[str]]] = []
    all_wells = set(wells.tolist())
    for valid in folds:
        valid_set = set(valid.tolist())
        splits.append((all_wells.difference(valid_set), valid_set))
    return splits


def build_well_arrays(frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for well, group in frame.groupby("well", sort=False):
        order = np.argsort(pd.to_numeric(group["md_since"], errors="coerce").to_numpy(np.float32))
        arrays[str(well)] = {
            "index": group.index.to_numpy(np.int64)[order],
            "md_since": numeric_array(group, "md_since")[order],
            "true_delta": numeric_array(group, "true_delta_from_anchor")[order],
        }
    return arrays


def interp_neighbor_delta(
    query_md: np.ndarray,
    neighbor_md: np.ndarray,
    neighbor_delta: np.ndarray,
    *,
    require_in_range: bool,
) -> np.ndarray:
    finite = np.isfinite(neighbor_md) & np.isfinite(neighbor_delta)
    if finite.sum() < 2:
        return np.full(len(query_md), np.nan, dtype=np.float32)
    x = neighbor_md[finite].astype(np.float64)
    y = neighbor_delta[finite].astype(np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]
    if len(x) < 2:
        return np.full(len(query_md), np.nan, dtype=np.float32)
    left = np.nan if require_in_range else float(y[0])
    right = np.nan if require_in_range else float(y[-1])
    return np.interp(query_md.astype(np.float64), x, y, left=left, right=right).astype(np.float32)


def generate_prior_for_method(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    method: GroupMethod,
    config: dict[str, Any],
) -> pd.DataFrame:
    prior_cfg = get_nested(config, "model.neighbor_prior") or {}
    min_cluster_size = int(prior_cfg.get("min_cluster_size", 2))
    min_neighbor_wells = int(prior_cfg.get("min_neighbor_wells", 1))
    min_row_neighbor_values = int(prior_cfg.get("min_row_neighbor_values", 1))
    require_in_range = bool((prior_cfg.get("interpolation") or {}).get("require_in_range", True))
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    seed = int(get_nested(config, "validation.seed") or 42)

    well_to_cluster, cluster_to_wells, cluster_sizes = make_group_lookup(
        assignments,
        method,
        min_cluster_size=min_cluster_size,
    )
    well_arrays = build_well_arrays(frame)
    prior_delta = np.full(len(frame), np.nan, dtype=np.float32)
    prior_std = np.full(len(frame), np.nan, dtype=np.float32)
    prior_count = np.zeros(len(frame), dtype=np.int16)
    prior_neighbor_wells = np.zeros(len(frame), dtype=np.int16)
    prior_cluster_size = np.zeros(len(frame), dtype=np.int16)

    splits = groupkfold_wells(frame["well"].unique(), n_folds, seed)
    for fold, (train_wells, valid_wells) in enumerate(splits):
        del fold
        for well in sorted(valid_wells):
            if well not in well_arrays:
                continue
            cluster = well_to_cluster.get(well)
            if cluster is None:
                continue
            candidate_neighbors = [
                neighbor
                for neighbor in cluster_to_wells.get(cluster, [])
                if neighbor in train_wells and neighbor in well_arrays and neighbor != well
            ]
            if len(candidate_neighbors) < min_neighbor_wells:
                continue
            query = well_arrays[well]
            row_idx = query["index"]
            query_md = query["md_since"]
            neighbor_values: list[np.ndarray] = []
            for neighbor in candidate_neighbors:
                neighbor_data = well_arrays[neighbor]
                values = interp_neighbor_delta(
                    query_md,
                    neighbor_data["md_since"],
                    neighbor_data["true_delta"],
                    require_in_range=require_in_range,
                )
                if np.isfinite(values).any():
                    neighbor_values.append(values)
            if not neighbor_values:
                continue
            stacked = np.vstack(neighbor_values)
            counts = np.isfinite(stacked).sum(axis=0)
            valid_rows = counts >= min_row_neighbor_values
            if not valid_rows.any():
                continue
            prior_delta[row_idx[valid_rows]] = np.nanmedian(stacked[:, valid_rows], axis=0).astype(
                np.float32
            )
            prior_std[row_idx[valid_rows]] = np.nanstd(stacked[:, valid_rows], axis=0).astype(
                np.float32
            )
            prior_count[row_idx] = counts.astype(np.int16)
            prior_neighbor_wells[row_idx] = len(candidate_neighbors)
            prior_cluster_size[row_idx] = cluster_sizes.get(cluster, 0)

    out = pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "well": frame["well"].to_numpy(),
            f"{method.name}_prior_delta": prior_delta,
            f"{method.name}_prior_tvt": numeric_array(frame, "last_known_tvt") + prior_delta,
            f"{method.name}_prior_std": prior_std,
            f"{method.name}_prior_count": prior_count,
            f"{method.name}_neighbor_wells": prior_neighbor_wells,
            f"{method.name}_cluster_size": prior_cluster_size,
        }
    )
    return out


def _distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def compute_metrics(
    frame: pd.DataFrame,
    candidate_columns: list[str],
    *,
    group_method_by_candidate: dict[str, str],
) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt").astype(np.float64)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_columns:
        pred = numeric_array(frame, candidate).astype(np.float64)
        mask = np.isfinite(true) & np.isfinite(pred)
        if not mask.any():
            continue
        error = pred[mask] - true[mask]
        rows.append(
            {
                "candidate": candidate,
                "group_method": group_method_by_candidate.get(candidate, "baseline"),
                "rows": int(mask.sum()),
                "coverage": float(mask.mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "within10": float(np.mean(np.abs(error) <= 10.0)),
                "bias": float(np.mean(error)),
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "candidate"]).reset_index(drop=True)


def compute_bucket_metrics(frame: pd.DataFrame, candidate_columns: list[str]) -> pd.DataFrame:
    work = frame[["true_tvt", "md_since"] + candidate_columns].copy()
    work["distance_bucket"] = _distance_bucket(work["md_since"])
    rows: list[dict[str, Any]] = []
    true = numeric_array(work, "true_tvt").astype(np.float64)
    for candidate in candidate_columns:
        pred = numeric_array(work, candidate).astype(np.float64)
        for bucket, idx in work.groupby("distance_bucket", observed=False).groups.items():
            positions = np.array(list(idx), dtype=np.int64)
            mask = np.isfinite(true[positions]) & np.isfinite(pred[positions])
            if not mask.any():
                continue
            error = pred[positions][mask] - true[positions][mask]
            rows.append(
                {
                    "candidate": candidate,
                    "distance_bucket": str(bucket),
                    "rows": int(mask.sum()),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                    "within10": float(np.mean(np.abs(error) <= 10.0)),
                    "bias": float(np.mean(error)),
                }
            )
    return pd.DataFrame(rows)


def compute_by_well(frame: pd.DataFrame, candidate_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=False):
        true = numeric_array(group, "true_tvt").astype(np.float64)
        for candidate in candidate_columns:
            pred = numeric_array(group, candidate).astype(np.float64)
            mask = np.isfinite(true) & np.isfinite(pred)
            if not mask.any():
                continue
            error = pred[mask] - true[mask]
            rows.append(
                {
                    "well": str(well),
                    "candidate": candidate,
                    "rows": int(mask.sum()),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                    "within10": float(np.mean(np.abs(error) <= 10.0)),
                    "bias": float(np.mean(error)),
                }
            )
    return pd.DataFrame(rows)


def add_corrected_candidates(
    frame: pd.DataFrame,
    methods: list[GroupMethod],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    prior_cfg = get_nested(config, "model.neighbor_prior") or {}
    base_candidates = list(prior_cfg.get("base_candidates", ["likpf_mean"]))
    alphas = [float(value) for value in prior_cfg.get("correction_alphas", [0.1, 0.2])]
    clips = [float(value) for value in prior_cfg.get("correction_clip_ft", [20.0])]
    std_thresholds = [float(value) for value in prior_cfg.get("gated_std_thresholds", [10.0])]
    min_neighbors_values = [int(value) for value in prior_cfg.get("gated_min_neighbors", [3])]
    candidate_columns: list[str] = list(prior_cfg.get("score_baselines", []))
    group_method_by_candidate = {candidate: "baseline" for candidate in candidate_columns}

    for method in methods:
        prior_col = f"{method.name}_prior_tvt"
        std_col = f"{method.name}_prior_std"
        count_col = f"{method.name}_prior_count"
        if prior_col not in frame.columns:
            continue
        prior_values = numeric_array(frame, prior_col)
        candidate_columns.append(prior_col)
        group_method_by_candidate[prior_col] = method.name
        for base in base_candidates:
            if base not in frame.columns:
                continue
            base_values = numeric_array(frame, base)
            diff = prior_values - base_values
            for alpha in alphas:
                alpha_tag = str(alpha).replace(".", "p")
                for clip in clips:
                    clip_tag = str(int(clip)) if clip.is_integer() else str(clip).replace(".", "p")
                    name = f"{method.name}_{base}_corr_a{alpha_tag}_c{clip_tag}"
                    corrected = base_values.copy()
                    valid = np.isfinite(diff)
                    corrected[valid] = base_values[valid] + alpha * np.clip(
                        diff[valid],
                        -clip,
                        clip,
                    )
                    frame[name] = corrected.astype(np.float32)
                    candidate_columns.append(name)
                    group_method_by_candidate[name] = method.name
            for std_threshold in std_thresholds:
                std_tag = (
                    str(int(std_threshold))
                    if std_threshold.is_integer()
                    else str(std_threshold).replace(".", "p")
                )
                for min_neighbors in min_neighbors_values:
                    name = f"{method.name}_{base}_gate_std{std_tag}_n{min_neighbors}"
                    corrected = base_values.copy()
                    gate = (
                        np.isfinite(diff)
                        & np.isfinite(numeric_array(frame, std_col))
                        & (numeric_array(frame, std_col) <= std_threshold)
                        & (numeric_array(frame, count_col) >= min_neighbors)
                    )
                    corrected[gate] = base_values[gate] + 0.2 * np.clip(diff[gate], -20.0, 20.0)
                    frame[name] = corrected.astype(np.float32)
                    candidate_columns.append(name)
                    group_method_by_candidate[name] = method.name

    seen: set[str] = set()
    deduped = []
    for candidate in candidate_columns:
        if candidate in frame.columns and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return frame, deduped, group_method_by_candidate


def write_feature_schema(path: Path, columns: list[str]) -> None:
    schema = pd.DataFrame(
        {
            "variant": "typewell_neighbor_prior_features",
            "feature_index": np.arange(len(columns), dtype=int),
            "feature": columns,
        }
    )
    schema.to_csv(path, index=False)


def run_audit(
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    start = time.time()
    config = load_config() if config is None else config
    paths = ExperimentPaths() if paths is None else paths
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()

    frame, feature_meta = read_feature_cache(config)
    assignments, cluster_meta = read_cluster_assignments(config)
    methods = parse_group_methods(config)

    prior_frames = []
    for method in methods:
        prior_frames.append(generate_prior_for_method(frame, assignments, method, config))
    work = frame.copy()
    for prior in prior_frames:
        extra_cols = [column for column in prior.columns if column not in {"id", "well"}]
        work = work.merge(prior[["id", "well", *extra_cols]], on=["id", "well"], how="left")

    work, candidate_columns, group_method_by_candidate = add_corrected_candidates(
        work,
        methods,
        config,
    )
    candidate_metrics = compute_metrics(
        work,
        candidate_columns,
        group_method_by_candidate=group_method_by_candidate,
    )
    bucket_metrics = compute_bucket_metrics(work, candidate_columns)
    by_well = compute_by_well(work, candidate_columns)

    artifacts = paths.artifacts_dir
    metrics_path = artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    bucket_path = artifacts / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    oof_path = artifacts / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
    schema_path = artifacts / f"{OUTPUT_PREFIX}_feature_schema.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    candidate_metrics.to_csv(metrics_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    keep_columns = [
        "id",
        "well",
        "target",
        "true_tvt",
        "last_known_tvt",
        "md_since",
        "eval_len",
        *[
            column
            for column in work.columns
            if column.endswith("_prior_tvt")
            or column.endswith("_prior_delta")
            or column.endswith("_prior_std")
            or column.endswith("_prior_count")
            or column.endswith("_neighbor_wells")
            or column.endswith("_cluster_size")
        ],
        *candidate_columns,
    ]
    keep_columns = list(
        dict.fromkeys([column for column in keep_columns if column in work.columns])
    )
    work[keep_columns].to_csv(oof_path, index=False, compression="gzip")
    write_feature_schema(schema_path, keep_columns)

    best = candidate_metrics.iloc[0].to_dict() if len(candidate_metrics) else {}
    baseline = candidate_metrics[candidate_metrics["candidate"] == "likpf_mean"]
    baseline_row = baseline.iloc[0].to_dict() if len(baseline) else {}
    prior_coverage = {
        method.name: {
            "prior_valid_rate": float(
                np.isfinite(numeric_array(work, f"{method.name}_prior_tvt")).mean()
            ),
            "mean_row_neighbor_count": float(
                np.mean(numeric_array(work, f"{method.name}_prior_count"))
            ),
            "max_cluster_size": int(np.nanmax(numeric_array(work, f"{method.name}_cluster_size"))),
        }
        for method in methods
        if f"{method.name}_prior_tvt" in work.columns
    }
    summary = {
        "experiment": OUTPUT_PREFIX,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.time() - start,
        "rows": int(len(work)),
        "wells": int(work["well"].nunique()),
        "feature_cache": feature_meta,
        "cluster_assignments": cluster_meta,
        "group_methods": [method.__dict__ for method in methods],
        "prior_coverage": prior_coverage,
        "best_candidate": to_jsonable(best),
        "likpf_baseline": to_jsonable(baseline_row),
        "delta_best_minus_likpf_rmse": (
            float(best["rmse"] - baseline_row["rmse"]) if best and baseline_row else None
        ),
        "artifacts": {
            "candidate_metrics": str(metrics_path),
            "bucket_metrics": str(bucket_path),
            "by_well": str(by_well_path),
            "oof_predictions": str(oof_path),
            "feature_schema": str(schema_path),
            "summary": str(summary_path),
        },
        "artifact_sha256": {
            "candidate_metrics": sha256_path(metrics_path),
            "bucket_metrics": sha256_path(bucket_path),
            "by_well": sha256_path(by_well_path),
            "oof_predictions_raw": sha256_path(oof_path),
            "oof_predictions_decompressed": sha256_path(oof_path, decompressed=True),
            "feature_schema": sha256_path(schema_path),
        },
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    metrics_json = {
        "status": "completed_train_side_audit",
        "best_candidate": to_jsonable(best),
        "likpf_baseline": to_jsonable(baseline_row),
        "delta_best_minus_likpf_rmse": summary["delta_best_minus_likpf_rmse"],
        "rows": int(len(work)),
        "wells": int(work["well"].nunique()),
        "summary_path": str(summary_path),
    }
    paths.metrics_path.write_text(
        json.dumps(to_jsonable(metrics_json), indent=2, sort_keys=True) + "\n"
    )
    return summary


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(to_jsonable(result["best_candidate"]), indent=2, sort_keys=True))

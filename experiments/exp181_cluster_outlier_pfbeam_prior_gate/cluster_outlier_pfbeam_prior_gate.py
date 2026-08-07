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

OUTPUT_PREFIX = "exp181_cluster_outlier_pfbeam_prior_gate"
EXP065_CLUSTER_ASSIGNMENTS = "common_typewell_cluster_assignments.csv"
EXP109_OOF = "exp109_typewell_neighbor_prior_features_oof_predictions.csv.gz"
EXP109_SUMMARY = "exp109_typewell_neighbor_prior_features_summary.json"
EXP114_OOF = "exp114_spatial_neighbor_prior_signal_audit_oof_predictions.csv.gz"
EXP114_WELL_GEOMETRY = "exp114_spatial_neighbor_prior_signal_audit_well_geometry_summary.csv"
EXP114_SUMMARY = "exp114_spatial_neighbor_prior_signal_audit_summary.json"
EXP115_FOLD_ASSIGNMENTS = "exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv"


@dataclass(frozen=True)
class BaseCandidateSpec:
    name: str
    display_name: str
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
class QualityGateSpec:
    name: str
    max_prior_std: float | None = None
    min_neighbor_wells: int | None = None


@dataclass(frozen=True)
class ClusterGateSpec:
    name: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class PolicySpec:
    policy: str
    source_name: str
    model: str
    prior: PriorSpec | None
    cluster_gate: ClusterGateSpec | None
    quality_gate: QualityGateSpec | None
    alpha: float
    clip: float


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


def prediction_sha256(ids: pd.Series, values: np.ndarray, *, label: str) -> str:
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(b"\n")
    for row_id, value in zip(ids.astype(str), values.astype(np.float64), strict=False):
        digest.update(row_id.encode("utf-8"))
        digest.update(b",")
        digest.update(np.float64(value).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


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


def float_tag(value: float) -> str:
    text = f"{float(value):.5g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def distance_bucket(values: pd.Series | np.ndarray) -> pd.Series | pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def distance_bucket_categories(buckets: pd.Series | pd.Categorical) -> pd.Index:
    return buckets.cat.categories if isinstance(buckets, pd.Series) else buckets.categories


def score_prediction(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    pred_values = pred.astype(np.float64)
    true_values = true.astype(np.float64)
    mask = np.isfinite(pred_values) & np.isfinite(true_values)
    if not mask.any():
        return {
            "rows": 0,
            "coverage": 0.0,
            "rmse": None,
            "mae": None,
            "within10": None,
            "bias": None,
        }
    error = pred_values[mask] - true_values[mask]
    return {
        "rows": int(mask.sum()),
        "coverage": float(mask.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "within10": float(np.mean(np.abs(error) <= 10.0)),
        "bias": float(np.mean(error)),
    }


def base_score_state(base: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    error = base.astype(np.float64) - true.astype(np.float64)
    valid = np.isfinite(error)
    valid_error = error[valid]
    return {
        "valid": valid,
        "error": error,
        "n": int(valid.sum()),
        "denom": int(len(error)),
        "sse": float(np.sum(valid_error**2)),
        "abs_sum": float(np.sum(np.abs(valid_error))),
        "within": int(np.sum(np.abs(valid_error) <= 10.0)),
        "bias_sum": float(np.sum(valid_error)),
    }


def score_from_sparse_correction(
    state: dict[str, Any],
    correction: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    valid = state["valid"]
    change_mask = active & valid & np.isfinite(correction)
    n = int(state["n"])
    if n == 0:
        return {
            "rows": 0,
            "coverage": 0.0,
            "rmse": None,
            "mae": None,
            "within10": None,
            "bias": None,
        }
    sse = float(state["sse"])
    abs_sum = float(state["abs_sum"])
    within = int(state["within"])
    bias_sum = float(state["bias_sum"])
    if change_mask.any():
        old_error = state["error"][change_mask]
        new_error = old_error + correction[change_mask].astype(np.float64)
        sse += float(np.sum(new_error**2 - old_error**2))
        abs_sum += float(np.sum(np.abs(new_error) - np.abs(old_error)))
        within += int(np.sum(np.abs(new_error) <= 10.0) - np.sum(np.abs(old_error) <= 10.0))
        bias_sum += float(np.sum(correction[change_mask].astype(np.float64)))
    return {
        "rows": n,
        "coverage": float(n / max(int(state["denom"]), 1)),
        "rmse": float(np.sqrt(max(sse, 0.0) / n)),
        "mae": float(abs_sum / n),
        "within10": float(within / n),
        "bias": float(bias_sum / n),
    }


def score_from_indexed_correction(
    state: dict[str, Any],
    active_idx: np.ndarray,
    correction_active: np.ndarray,
) -> dict[str, Any]:
    n = int(state["n"])
    if n == 0:
        return {
            "rows": 0,
            "coverage": 0.0,
            "rmse": None,
            "mae": None,
            "within10": None,
            "bias": None,
        }
    sse = float(state["sse"])
    abs_sum = float(state["abs_sum"])
    within = int(state["within"])
    bias_sum = float(state["bias_sum"])
    if len(active_idx):
        old_error = state["error"][active_idx]
        new_error = old_error + correction_active.astype(np.float64)
        sse += float(np.sum(new_error**2 - old_error**2))
        abs_sum += float(np.sum(np.abs(new_error) - np.abs(old_error)))
        within += int(np.sum(np.abs(new_error) <= 10.0) - np.sum(np.abs(old_error) <= 10.0))
        bias_sum += float(np.sum(correction_active.astype(np.float64)))
    return {
        "rows": n,
        "coverage": float(n / max(int(state["denom"]), 1)),
        "rmse": float(np.sqrt(max(sse, 0.0) / n)),
        "mae": float(abs_sum / n),
        "within10": float(within / n),
        "bias": float(bias_sum / n),
    }


def parse_base_candidates(config: dict[str, Any]) -> list[BaseCandidateSpec]:
    specs: list[BaseCandidateSpec] = []
    for raw in get_nested(config, "base_candidates") or []:
        name = str(raw["name"])
        specs.append(
            BaseCandidateSpec(
                name=name,
                display_name=str(raw.get("display_name", name)),
                column=str(raw.get("column", name)),
            )
        )
    if not specs:
        specs = [
            BaseCandidateSpec(
                name="likpf_mean",
                display_name="likelihood PF mean",
                column="likpf_mean",
            ),
            BaseCandidateSpec(name="pf_ancc", display_name="PF ANCC", column="pf_ancc"),
            BaseCandidateSpec(name="beam_mean", display_name="Beam mean", column="beam_mean"),
        ]
    return specs


def parse_prior_specs(config: dict[str, Any]) -> list[PriorSpec]:
    specs: list[PriorSpec] = []
    for raw in get_nested(config, "gate.prior_variants") or []:
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
                    None if raw.get("azimuth_mismatch") is None else str(raw["azimuth_mismatch"])
                ),
            )
        )
    if not specs:
        raise ValueError("gate.prior_variants must not be empty")
    return specs


def parse_quality_gates(config: dict[str, Any]) -> list[QualityGateSpec]:
    gates: list[QualityGateSpec] = []
    for raw in get_nested(config, "gate.prior_quality") or []:
        gates.append(
            QualityGateSpec(
                name=str(raw["name"]),
                max_prior_std=(
                    None if raw.get("max_prior_std") is None else float(raw["max_prior_std"])
                ),
                min_neighbor_wells=(
                    None
                    if raw.get("min_neighbor_wells") is None
                    else int(raw["min_neighbor_wells"])
                ),
            )
        )
    if not gates:
        gates.append(QualityGateSpec(name="valid_prior"))
    return gates


def parse_cluster_gates(config: dict[str, Any]) -> list[ClusterGateSpec]:
    gates: list[ClusterGateSpec] = []
    for raw in get_nested(config, "cluster.gates") or []:
        gates.append(ClusterGateSpec(name=str(raw["name"]), raw=dict(raw)))
    if not gates:
        gates.append(ClusterGateSpec(name="own_z_gt1p5", raw={"own_cluster_dist_z_gt": 1.5}))
    return gates


def parse_reference_policy_specs(
    config: dict[str, Any],
    *,
    source_name: str,
    base_candidate: BaseCandidateSpec,
    priors: list[PriorSpec],
    quality_gates: list[QualityGateSpec],
) -> list[PolicySpec]:
    prior_lookup = {prior.name: prior for prior in priors}
    quality_lookup = {gate.name: gate for gate in quality_gates}
    fallback_quality = quality_lookup.get("valid_prior") or quality_gates[0]
    specs: list[PolicySpec] = []
    for raw in get_nested(config, "audit.reference_policies") or []:
        base_name = str(raw.get("base_candidate", ""))
        if base_name not in {base_candidate.name, base_candidate.column}:
            continue
        prior_name = str(raw["prior"])
        if prior_name not in prior_lookup:
            raise ValueError(f"unknown reference prior: {prior_name}")
        name = str(raw["name"])
        quality_name = str(raw.get("quality_gate", fallback_quality.name))
        quality_gate = quality_lookup.get(quality_name)
        if quality_gate is None:
            raise ValueError(f"unknown reference quality gate: {quality_name}")
        cluster_gate = ClusterGateSpec(
            name=str(raw.get("cluster_gate", "global_all_rows")),
            raw={"all_rows": True},
        )
        policy = f"{source_name}__{base_candidate.name}__reference__{name}"
        specs.append(
            PolicySpec(
                policy=policy,
                source_name=source_name,
                model=base_candidate.name,
                prior=prior_lookup[prior_name],
                cluster_gate=cluster_gate,
                quality_gate=quality_gate,
                alpha=float(raw["alpha"]),
                clip=float(raw["clip"]),
            )
        )
    return specs


def read_typewell_prior(
    config: dict[str, Any],
    priors: list[PriorSpec],
    base_candidates: list[BaseCandidateSpec],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = require_path(
        filename=EXP109_OOF,
        explicit_path=get_nested(config, "data.exp109_oof_predictions_local"),
    )
    required = {"id", "well", "true_tvt", "last_known_tvt", "md_since", "eval_len"}
    for candidate in base_candidates:
        required.add(candidate.column)
    for prior in priors:
        if prior.family != "typewell":
            continue
        required.add(prior.prior_tvt)
        for column in [prior.prior_std, prior.prior_count, prior.neighbor_wells]:
            if column:
                required.add(column)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = sorted(required.difference(header))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    max_rows = get_nested(config, "audit.max_rows")
    frame = pd.read_csv(
        source,
        usecols=sorted(required),
        nrows=None if max_rows in {None, "null"} else int(max_rows),
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": sha256_path(source, decompressed=source.suffix == ".gz"),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": list(frame.columns),
    }
    summary_path = find_existing_path(
        filename=EXP109_SUMMARY,
        explicit_path=get_nested(config, "data.exp109_summary_local"),
    )
    metadata["summary"] = str(summary_path) if summary_path else None
    metadata["summary_sha256"] = sha256_path(summary_path) if summary_path else None
    return frame, metadata


def read_spatial_prior(
    config: dict[str, Any],
    priors: list[PriorSpec],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = require_path(
        filename=EXP114_OOF,
        explicit_path=get_nested(config, "data.exp114_oof_predictions_local"),
    )
    required = {"id", "well", "true_tvt"}
    for prior in priors:
        if prior.family != "spatial":
            continue
        required.add(prior.prior_tvt)
        for column in [
            prior.prior_std,
            prior.prior_count,
            prior.neighbor_wells,
            prior.distance_mean,
            prior.azimuth_mismatch,
        ]:
            if column:
                required.add(column)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = sorted(required.difference(header))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    max_rows = get_nested(config, "audit.max_rows")
    frame = pd.read_csv(
        source,
        usecols=sorted(required),
        nrows=None if max_rows in {None, "null"} else int(max_rows),
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame = frame.rename(columns={"true_tvt": "exp114_true_tvt"})
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": sha256_path(source, decompressed=source.suffix == ".gz"),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": list(frame.columns),
    }
    summary_path = find_existing_path(
        filename=EXP114_SUMMARY,
        explicit_path=get_nested(config, "data.exp114_summary_local"),
    )
    metadata["summary"] = str(summary_path) if summary_path else None
    metadata["summary_sha256"] = sha256_path(summary_path) if summary_path else None
    return frame, metadata


def read_prior_frame(
    config: dict[str, Any],
    priors: list[PriorSpec],
    base_candidates: list[BaseCandidateSpec],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    typewell, typewell_meta = read_typewell_prior(config, priors, base_candidates)
    spatial, spatial_meta = read_spatial_prior(config, priors)
    frame = typewell.merge(spatial, on=["id", "well"], how="inner", validate="one_to_one")
    target_diff = np.abs(numeric_array(frame, "true_tvt") - numeric_array(frame, "exp114_true_tvt"))
    target_diff_max = float(np.nanmax(target_diff)) if len(target_diff) else None
    if target_diff_max is not None and target_diff_max > 1.0e-3:
        raise ValueError(f"exp109 true_tvt and exp114 true_tvt differ: max={target_diff_max}")
    frame = frame.drop(columns=["exp114_true_tvt"])
    metadata = {
        "typewell": typewell_meta,
        "spatial": spatial_meta,
        "joined_rows": int(len(frame)),
        "joined_wells": int(frame["well"].nunique()),
        "target_tvt_max_abs_diff": target_diff_max,
    }
    return frame, metadata


def read_prediction_source(
    config: dict[str, Any],
    spec: Any,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    filename = Path(spec.path_candidates[0]).name if spec.path_candidates else ""
    source = find_existing_path(filename=filename, candidates=spec.path_candidates)
    metadata: dict[str, Any] = {
        "name": spec.name,
        "display_name": spec.display_name,
        "required": spec.required,
        "source": str(source) if source else None,
        "loaded": False,
    }
    if source is None:
        if spec.required:
            raise FileNotFoundError(f"required prediction source not found: {spec.name}")
        return None, metadata
    usecols = ["id", "well", "variant", "mode", "model", "target_tvt", "pred_tvt"]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in usecols if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source,
        usecols=usecols,
        dtype={"id": str, "well": str, "variant": str, "mode": str, "model": str},
        chunksize=int(get_nested(config, "runtime.read_chunksize") or 1_000_000),
        low_memory=False,
    ):
        mask = chunk["model"].isin(spec.models)
        if spec.variant:
            mask &= chunk["variant"].astype(str) == spec.variant
        if spec.mode:
            mask &= chunk["mode"].astype(str) == spec.mode
        part = chunk.loc[mask].copy()
        if part.empty:
            continue
        part["target_tvt"] = pd.to_numeric(part["target_tvt"], errors="coerce").astype(np.float32)
        part["pred_tvt"] = pd.to_numeric(part["pred_tvt"], errors="coerce").astype(np.float32)
        part["source_name"] = spec.name
        part["source_display_name"] = spec.display_name
        chunks.append(part)
    if not chunks:
        if spec.required:
            raise ValueError(f"no rows matched required prediction source: {spec.name}")
        metadata["source_sha256"] = sha256_path(source)
        metadata["source_decompressed_sha256"] = sha256_path(
            source,
            decompressed=source.suffix == ".gz",
        )
        return None, metadata
    frame = pd.concat(chunks, ignore_index=True, sort=False)
    metadata.update(
        {
            "loaded": True,
            "source_sha256": sha256_path(source),
            "source_decompressed_sha256": sha256_path(source, decompressed=source.suffix == ".gz"),
            "rows": int(len(frame)),
            "wells": int(frame["well"].nunique()),
            "models": sorted(frame["model"].unique().tolist()),
            "variant": spec.variant,
            "mode": spec.mode,
        }
    )
    return frame, metadata


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


def build_cluster_features(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    assignments, assignment_meta = read_cluster_assignments(config)
    geometry, geometry_meta = read_well_geometry(config)
    min_cluster_size = int(get_nested(config, "cluster.min_cluster_size") or 2)
    scale_floor = float(get_nested(config, "cluster.robust_scale_floor_ft") or 250.0)
    nearby_k_values = [int(value) for value in get_nested(config, "cluster.nearby_k_values") or [8]]
    majority_min_share = float(get_nested(config, "cluster.nearby_majority_min_share") or 0.5)

    assignment_cols = ["well_id", "cluster_id", "cluster_size", "representative_well_id"]
    available = [column for column in assignment_cols if column in assignments.columns]
    joined = geometry.merge(
        assignments[available].rename(columns={"well_id": "well"}),
        on="well",
        how="left",
        validate="one_to_one",
    )
    joined["cluster_size"] = (
        pd.to_numeric(joined["cluster_size"], errors="coerce").fillna(0).astype(int)
    )
    joined["cluster_id"] = joined["cluster_id"].astype("string")

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
        joined, "nearest_other_cluster_dist"
    ) < numeric_array(joined, "own_cluster_dist")

    dist_matrix = np.sqrt(np.sum((well_xy[:, None, :] - well_xy[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    cluster_values = joined["cluster_id"].astype("string").to_numpy()
    for k in nearby_k_values:
        majority_clusters: list[str | None] = []
        majority_counts: list[int] = []
        majority_shares: list[float] = []
        diff_flags: list[bool] = []
        for i in range(len(joined)):
            if not np.isfinite(dist_matrix[i]).any():
                majority_clusters.append(None)
                majority_counts.append(0)
                majority_shares.append(0.0)
                diff_flags.append(False)
                continue
            idx = np.argsort(dist_matrix[i])[:k]
            values = [str(cluster_values[j]) for j in idx if not pd.isna(cluster_values[j])]
            if not values:
                majority_clusters.append(None)
                majority_counts.append(0)
                majority_shares.append(0.0)
                diff_flags.append(False)
                continue
            counts = pd.Series(values).value_counts()
            majority_cluster = str(counts.index[0])
            majority_count = int(counts.iloc[0])
            share = float(majority_count / max(len(values), 1))
            own_cluster = None if pd.isna(cluster_values[i]) else str(cluster_values[i])
            majority_clusters.append(majority_cluster)
            majority_counts.append(majority_count)
            majority_shares.append(share)
            diff_flags.append(
                bool(
                    own_cluster is not None
                    and majority_cluster != own_cluster
                    and share >= majority_min_share
                )
            )
        joined[f"nearby_majority_cluster_k{k}"] = majority_clusters
        joined[f"nearby_majority_count_k{k}"] = majority_counts
        joined[f"nearby_majority_share_k{k}"] = majority_shares
        joined[f"nearby_majority_diff_k{k}"] = diff_flags

    joined["cluster_feature_valid"] = valid_cluster.to_numpy()
    metadata = {
        "assignments": assignment_meta,
        "geometry": geometry_meta,
        "min_cluster_size": min_cluster_size,
        "robust_scale_floor_ft": scale_floor,
        "nearby_k_values": nearby_k_values,
        "nearby_majority_min_share": majority_min_share,
        "cluster_feature_wells": int(joined["cluster_feature_valid"].sum()),
        "clusters_used": int(cluster_stats["cluster_id"].nunique()) if len(cluster_stats) else 0,
    }
    return joined, metadata


def read_exp115_roles(config: dict[str, Any]) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    source = find_existing_path(
        filename=EXP115_FOLD_ASSIGNMENTS,
        explicit_path=get_nested(config, "data.exp115_fold_assignments_local"),
    )
    metadata: dict[str, Any] = {"source": str(source) if source else None, "loaded": False}
    if source is None:
        return None, metadata
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
    metadata.update(
        {
            "loaded": True,
            "source_sha256": sha256_path(source),
            "rows": int(len(frame)),
            "wells": int(frame["well"].nunique()),
        }
    )
    return frame, metadata


def condition_mask(frame: pd.DataFrame, raw: dict[str, Any]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    if raw.get("own_cluster_dist_z_gt") is not None:
        mask &= numeric_array(frame, "own_cluster_dist_z") > float(raw["own_cluster_dist_z_gt"])
    if bool(raw.get("nearest_other_closer", False)):
        mask &= frame["nearest_other_closer"].fillna(False).to_numpy(bool)
    if bool(raw.get("nearby_majority_diff", False)):
        k = int(raw.get("nearby_k", 8))
        column = f"nearby_majority_diff_k{k}"
        if column not in frame.columns:
            return np.zeros(len(frame), dtype=bool)
        mask &= frame[column].fillna(False).to_numpy(bool)
    return mask


def cluster_gate_mask(frame: pd.DataFrame, gate: ClusterGateSpec) -> np.ndarray:
    base = frame["cluster_feature_valid"].fillna(False).to_numpy(bool)
    raw = dict(gate.raw)
    if raw.get("all_rows"):
        return np.ones(len(frame), dtype=bool)
    if raw.get("any_of"):
        options = [condition_mask(frame, dict(item)) for item in raw["any_of"]]
        if not options:
            return np.zeros(len(frame), dtype=bool)
        return base & np.logical_or.reduce(options)
    return base & condition_mask(frame, raw)


def quality_gate_mask(frame: pd.DataFrame, prior: PriorSpec, gate: QualityGateSpec) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    if gate.max_prior_std is not None:
        if prior.prior_std is None or prior.prior_std not in frame.columns:
            return np.zeros(len(frame), dtype=bool)
        values = numeric_array(frame, prior.prior_std)
        mask &= np.isfinite(values) & (values <= gate.max_prior_std)
    if gate.min_neighbor_wells is not None:
        if prior.neighbor_wells is None or prior.neighbor_wells not in frame.columns:
            return np.zeros(len(frame), dtype=bool)
        values = numeric_array(frame, prior.neighbor_wells)
        mask &= np.isfinite(values) & (values >= gate.min_neighbor_wells)
    return mask


def make_prediction(
    frame: pd.DataFrame,
    spec: PolicySpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = numeric_array(frame, "base_pred_tvt")
    if spec.prior is None:
        correction = np.zeros(len(frame), dtype=np.float32)
        return base.copy(), correction, np.zeros(len(frame), dtype=bool)
    prior = numeric_array(frame, spec.prior.prior_tvt)
    delta = prior - base
    active = (
        np.isfinite(prior)
        & np.isfinite(base)
        & np.isfinite(delta)
        & cluster_gate_mask(frame, spec.cluster_gate)
        & quality_gate_mask(frame, spec.prior, spec.quality_gate)
    )
    correction = np.zeros(len(frame), dtype=np.float32)
    correction[active] = spec.alpha * np.clip(delta[active], -spec.clip, spec.clip)
    pred = base.copy()
    pred[active] = pred[active] + correction[active]
    return pred.astype(np.float32), correction, active


def compute_overall_metrics(
    frame: pd.DataFrame,
    *,
    priors: list[PriorSpec],
    cluster_gates: list[ClusterGateSpec],
    quality_gates: list[QualityGateSpec],
    alphas: list[float],
    clips: list[float],
    reference_policies: list[PolicySpec] | None = None,
) -> tuple[pd.DataFrame, dict[str, PolicySpec]]:
    true = numeric_array(frame, "true_tvt")
    base = numeric_array(frame, "base_pred_tvt")
    state = base_score_state(base, true)
    base_score = score_prediction(base, true)
    source_name = str(frame["source_name"].iloc[0])
    model = str(frame["model"].iloc[0])
    baseline_policy = f"{source_name}__{model}__baseline"
    policy_lookup: dict[str, PolicySpec] = {
        baseline_policy: PolicySpec(
            policy=baseline_policy,
            source_name=source_name,
            model=model,
            prior=None,
            cluster_gate=None,
            quality_gate=None,
            alpha=0.0,
            clip=0.0,
        )
    }
    rows: list[dict[str, Any]] = [
        {
            "policy": baseline_policy,
            "source_name": source_name,
            "model": model,
            "prior": "baseline",
            "prior_family": "baseline",
            "cluster_gate": "baseline",
            "quality_gate": "baseline",
            "alpha": 0.0,
            "clip": 0.0,
            "gate_rows": 0,
            "gate_wells": 0,
            "gate_rate": 0.0,
            "correction_abs_mean": 0.0,
            "correction_abs_p95": 0.0,
            "correction_abs_max": 0.0,
            "prediction_sha256": prediction_sha256(
                frame["id"],
                base,
                label=f"{OUTPUT_PREFIX}/{baseline_policy}",
            ),
            "delta_rmse_vs_baseline": 0.0,
            "delta_mae_vs_baseline": 0.0,
            "delta_within10_vs_baseline": 0.0,
            **base_score,
        }
    ]
    for prior in priors:
        if prior.prior_tvt not in frame.columns:
            continue
        prior_values = numeric_array(frame, prior.prior_tvt)
        base_delta = prior_values - base
        for cluster_gate in cluster_gates:
            cluster_mask = cluster_gate_mask(frame, cluster_gate)
            for quality_gate in quality_gates:
                quality_mask = quality_gate_mask(frame, prior, quality_gate)
                finite = np.isfinite(prior_values) & np.isfinite(base_delta)
                gate_mask = finite & cluster_mask & quality_mask
                gate_rows = int(gate_mask.sum())
                gate_wells = int(frame.loc[gate_mask, "well"].nunique())
                gate_rate = float(gate_mask.mean())
                active_idx = np.flatnonzero(gate_mask & state["valid"])
                delta_active = base_delta[active_idx]
                for alpha in alphas:
                    for clip in clips:
                        correction_active = (alpha * np.clip(delta_active, -clip, clip)).astype(
                            np.float32
                        )
                        score = score_from_indexed_correction(
                            state,
                            active_idx,
                            correction_active,
                        )
                        active_abs = np.abs(correction_active)
                        policy = (
                            f"{source_name}__{model}__{prior.name}__{cluster_gate.name}"
                            f"__{quality_gate.name}__a{float_tag(alpha)}__c{float_tag(clip)}"
                        )
                        policy_lookup[policy] = PolicySpec(
                            policy=policy,
                            source_name=source_name,
                            model=model,
                            prior=prior,
                            cluster_gate=cluster_gate,
                            quality_gate=quality_gate,
                            alpha=alpha,
                            clip=clip,
                        )
                        rows.append(
                            {
                                "policy": policy,
                                "source_name": source_name,
                                "model": model,
                                "prior": prior.name,
                                "prior_family": prior.family,
                                "cluster_gate": cluster_gate.name,
                                "quality_gate": quality_gate.name,
                                "alpha": alpha,
                                "clip": clip,
                                "gate_rows": gate_rows,
                                "gate_wells": gate_wells,
                                "gate_rate": gate_rate,
                                "correction_abs_mean": (
                                    float(np.mean(active_abs)) if len(active_abs) else 0.0
                                ),
                                "correction_abs_p95": (
                                    float(np.quantile(active_abs, 0.95)) if len(active_abs) else 0.0
                                ),
                                "correction_abs_max": (
                                    float(np.max(active_abs)) if len(active_abs) else 0.0
                                ),
                                "prediction_sha256": None,
                                "delta_rmse_vs_baseline": (
                                    None
                                    if score["rmse"] is None
                                    else score["rmse"] - base_score["rmse"]
                                ),
                                "delta_mae_vs_baseline": (
                                    None
                                    if score["mae"] is None
                                    else score["mae"] - base_score["mae"]
                                ),
                                "delta_within10_vs_baseline": (
                                    None
                                    if score["within10"] is None
                                    else score["within10"] - base_score["within10"]
                                ),
                                **score,
                            }
                        )
    for spec in reference_policies or []:
        if spec.prior is None or spec.prior.prior_tvt not in frame.columns:
            continue
        prior_values = numeric_array(frame, spec.prior.prior_tvt)
        base_delta = prior_values - base
        active = (
            np.isfinite(prior_values)
            & np.isfinite(base_delta)
            & cluster_gate_mask(frame, spec.cluster_gate)
            & quality_gate_mask(frame, spec.prior, spec.quality_gate)
        )
        active_idx = np.flatnonzero(active & state["valid"])
        correction_active = (
            spec.alpha * np.clip(base_delta[active_idx], -spec.clip, spec.clip)
        ).astype(np.float32)
        score = score_from_indexed_correction(state, active_idx, correction_active)
        active_abs = np.abs(correction_active)
        policy_lookup[spec.policy] = spec
        rows.append(
            {
                "policy": spec.policy,
                "source_name": source_name,
                "model": model,
                "prior": spec.prior.name,
                "prior_family": spec.prior.family,
                "cluster_gate": spec.cluster_gate.name,
                "quality_gate": spec.quality_gate.name,
                "alpha": spec.alpha,
                "clip": spec.clip,
                "gate_rows": int(active.sum()),
                "gate_wells": int(frame.loc[active, "well"].nunique()),
                "gate_rate": float(active.mean()),
                "correction_abs_mean": float(np.mean(active_abs)) if len(active_abs) else 0.0,
                "correction_abs_p95": float(np.quantile(active_abs, 0.95))
                if len(active_abs)
                else 0.0,
                "correction_abs_max": float(np.max(active_abs)) if len(active_abs) else 0.0,
                "prediction_sha256": None,
                "delta_rmse_vs_baseline": (
                    None if score["rmse"] is None else score["rmse"] - base_score["rmse"]
                ),
                "delta_mae_vs_baseline": (
                    None if score["mae"] is None else score["mae"] - base_score["mae"]
                ),
                "delta_within10_vs_baseline": (
                    None
                    if score["within10"] is None
                    else score["within10"] - base_score["within10"]
                ),
                **score,
            }
        )
    metrics = pd.DataFrame(rows).sort_values(["rmse", "policy"], na_position="last")
    return metrics, policy_lookup


def subgroup_masks(frame: pd.DataFrame, spec: PolicySpec | None = None) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {
        "all": np.ones(len(frame), dtype=bool),
        "cluster_own_z_gt1p5": numeric_array(frame, "own_cluster_dist_z") > 1.5,
        "cluster_nearest_other_closer": frame["nearest_other_closer"].fillna(False).to_numpy(bool),
    }
    for column in frame.columns:
        if column.startswith("nearby_majority_diff_k"):
            masks[f"cluster_{column}"] = frame[column].fillna(False).to_numpy(bool)
    if spec is not None and spec.cluster_gate is not None:
        masks[f"policy_cluster_gate__{spec.cluster_gate.name}"] = cluster_gate_mask(
            frame,
            spec.cluster_gate,
        )
    buckets = distance_bucket(frame["md_since"])
    for bucket in distance_bucket_categories(buckets):
        if str(bucket) in {"000_050", "1000_plus"}:
            masks[f"distance_{bucket}"] = (buckets.astype(str) == str(bucket)).to_numpy()
    for column in [
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ]:
        if column in frame.columns:
            masks[f"exp115_{column.replace('_role', '')}_valid"] = (
                frame[column].astype(str).to_numpy() == "valid"
            )
    return masks


def detailed_metrics(
    frame: pd.DataFrame,
    policies: list[PolicySpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_well_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for spec in policies:
        source_frame = frame[
            (frame["source_name"].astype(str) == spec.source_name)
            & (frame["model"].astype(str) == spec.model)
        ].reset_index(drop=True)
        if source_frame.empty:
            continue
        true = numeric_array(source_frame, "true_tvt")
        pred, correction, _ = make_prediction(source_frame, spec)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "id": source_frame["id"].astype(str),
                    "well": source_frame["well"].astype(str),
                    "source_name": spec.source_name,
                    "model": spec.model,
                    "policy": spec.policy,
                    "true_tvt": true,
                    "md_since": numeric_array(source_frame, "md_since"),
                    "base_pred_tvt": numeric_array(source_frame, "base_pred_tvt"),
                    "pred_tvt": pred,
                    "correction": correction,
                }
            )
        )
        for well, group_idx in source_frame.groupby("well", sort=False).groups.items():
            positions = np.asarray(list(group_idx), dtype=np.int64)
            score = score_prediction(pred[positions], true[positions])
            if score["rows"] == 0:
                continue
            by_well_rows.append(
                {
                    "policy": spec.policy,
                    "source_name": spec.source_name,
                    "model": spec.model,
                    "well": str(well),
                    **score,
                }
            )
            order = np.argsort(numeric_array(source_frame.iloc[positions], "md_since"))
            pred_step = np.abs(np.diff(pred[positions][order].astype(np.float64)))
            corr_step = np.abs(np.diff(correction[positions][order].astype(np.float64)))
            path_rows.append(
                {
                    "policy": spec.policy,
                    "source_name": spec.source_name,
                    "model": spec.model,
                    "well": str(well),
                    "rows": int(len(positions)),
                    "pred_step_abs_p95": (
                        float(np.quantile(pred_step, 0.95)) if len(pred_step) else 0.0
                    ),
                    "pred_step_abs_max": float(np.max(pred_step)) if len(pred_step) else 0.0,
                    "pred_step_abs_ge10": int(np.sum(pred_step >= 10.0)),
                    "pred_step_abs_ge25": int(np.sum(pred_step >= 25.0)),
                    "correction_step_abs_p95": (
                        float(np.quantile(corr_step, 0.95)) if len(corr_step) else 0.0
                    ),
                    "correction_step_abs_max": float(np.max(corr_step)) if len(corr_step) else 0.0,
                    "correction_step_abs_ge5": int(np.sum(corr_step >= 5.0)),
                }
            )
        buckets = distance_bucket(source_frame["md_since"])
        for bucket in distance_bucket_categories(buckets):
            mask = (buckets.astype(str) == str(bucket)).to_numpy()
            score = score_prediction(pred[mask], true[mask])
            if score["rows"] == 0:
                continue
            bucket_rows.append(
                {
                    "policy": spec.policy,
                    "source_name": spec.source_name,
                    "model": spec.model,
                    "distance_bucket": str(bucket),
                    **score,
                }
            )
        for subgroup, mask in subgroup_masks(source_frame, spec).items():
            score = score_prediction(pred[mask], true[mask])
            if score["rows"] == 0:
                continue
            subgroup_rows.append(
                {
                    "policy": spec.policy,
                    "source_name": spec.source_name,
                    "model": spec.model,
                    "subgroup": subgroup,
                    "wells": int(source_frame.loc[mask, "well"].nunique()),
                    **score,
                }
            )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True, sort=False)
        if prediction_frames
        else pd.DataFrame()
    )
    return (
        pd.DataFrame(by_well_rows),
        pd.DataFrame(bucket_rows),
        pd.DataFrame(subgroup_rows),
        pd.DataFrame(path_rows),
        predictions,
    )


def summarize_by_well_delta(by_well: pd.DataFrame) -> pd.DataFrame:
    if by_well.empty:
        return pd.DataFrame()
    baseline = by_well[by_well["policy"].str.endswith("__baseline")][
        ["source_name", "model", "well", "rmse"]
    ].rename(columns={"rmse": "baseline_rmse"})
    merged = by_well.merge(baseline, on=["source_name", "model", "well"], how="left")
    merged["delta_rmse_vs_baseline"] = merged["rmse"] - merged["baseline_rmse"]
    rows: list[dict[str, Any]] = []
    for (source_name, model, policy), group in merged.groupby(
        ["source_name", "model", "policy"],
        sort=False,
    ):
        if str(policy).endswith("__baseline"):
            continue
        delta = pd.to_numeric(group["delta_rmse_vs_baseline"], errors="coerce")
        rows.append(
            {
                "source_name": source_name,
                "model": model,
                "policy": policy,
                "wells": int(delta.notna().sum()),
                "improved_wells": int((delta < 0.0).sum()),
                "worse_wells": int((delta > 0.0).sum()),
                "same_wells": int((delta == 0.0).sum()),
                "max_regression_rmse": float(delta.max()) if delta.notna().any() else np.nan,
                "max_improvement_rmse": float(delta.min()) if delta.notna().any() else np.nan,
                "mean_delta_rmse": float(delta.mean()) if delta.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["max_regression_rmse", "mean_delta_rmse"],
        na_position="last",
    )


def write_feature_schema(path: Path, columns: list[str]) -> None:
    pd.DataFrame(
        {
            "variant": OUTPUT_PREFIX,
            "feature_index": np.arange(len(columns), dtype=int),
            "feature": columns,
        }
    ).to_csv(path, index=False)


def run_cluster_outlier_pfbeam_prior_gate(
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    start = time.time()
    config = load_config() if config is None else config
    paths = ExperimentPaths() if paths is None else paths
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()

    priors = parse_prior_specs(config)
    cluster_gates = parse_cluster_gates(config)
    quality_gates = parse_quality_gates(config)
    base_candidates = parse_base_candidates(config)
    alphas = [float(value) for value in get_nested(config, "gate.correction_alphas") or [0.1]]
    clips = [float(value) for value in get_nested(config, "gate.correction_clip_ft") or [20.0]]
    top_n = int(get_nested(config, "audit.top_n_detailed_policies") or 16)
    top_prediction_n = int(get_nested(config, "audit.top_n_prediction_policies") or 8)

    prior_frame, prior_meta = read_prior_frame(config, priors, base_candidates)
    cluster_features, cluster_meta = build_cluster_features(config)
    exp115_roles, exp115_meta = read_exp115_roles(config)

    base_feature_cols = [
        column
        for column in cluster_features.columns
        if column not in {"well", "representative_well_id"}
    ]
    all_frames: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    policy_lookup: dict[str, PolicySpec] = {}
    source_name = str(get_nested(config, "audit.source_name") or "pfbeam_oof")
    source_metadata: list[dict[str, Any]] = []
    for base_candidate in base_candidates:
        if base_candidate.column not in prior_frame.columns:
            raise ValueError(f"base candidate column is missing: {base_candidate.column}")
        frame = prior_frame.merge(
            cluster_features,
            on="well",
            how="left",
            validate="many_to_one",
        )
        if exp115_roles is not None:
            frame = frame.merge(exp115_roles, on="well", how="left", validate="many_to_one")
        frame["base_pred_tvt"] = numeric_array(frame, base_candidate.column)
        frame["source_name"] = source_name
        frame["source_display_name"] = "exp099 PF/Beam/likPF fixed OOF candidates"
        frame["model"] = base_candidate.name
        source_metadata.append(
            {
                "name": base_candidate.name,
                "display_name": base_candidate.display_name,
                "column": base_candidate.column,
                "rows": int(len(frame)),
                "wells": int(frame["well"].nunique()),
                "source": "exp109_oof_predictions_with_exp099_base_candidates",
            }
        )
        all_frames.append(frame)
        reference_policies = parse_reference_policy_specs(
            config,
            source_name=source_name,
            base_candidate=base_candidate,
            priors=priors,
            quality_gates=quality_gates,
        )
        metrics, lookup = compute_overall_metrics(
            frame.reset_index(drop=True),
            priors=priors,
            cluster_gates=cluster_gates,
            quality_gates=quality_gates,
            alphas=alphas,
            clips=clips,
            reference_policies=reference_policies,
        )
        all_metrics.append(metrics)
        policy_lookup.update(lookup)

    if not all_frames or not all_metrics:
        raise FileNotFoundError(
            "No PF/Beam base candidate was loaded. Ensure exp109 OOF predictions are "
            "available as Kaggle kernel sources or local artifacts."
        )

    joined_frame = pd.concat(all_frames, ignore_index=True, sort=False)
    gate_metrics = pd.concat(all_metrics, ignore_index=True, sort=False).sort_values(
        ["rmse", "policy"],
        na_position="last",
    )
    top_policies: list[PolicySpec] = []
    for _, row in gate_metrics.head(top_n).iterrows():
        policy = str(row["policy"])
        if policy in policy_lookup:
            top_policies.append(policy_lookup[policy])
    baseline_policies = [
        spec for policy, spec in policy_lookup.items() if policy.endswith("__baseline")
    ]
    for baseline in baseline_policies:
        if not any(spec.policy == baseline.policy for spec in top_policies):
            top_policies.insert(0, baseline)
    top_policies = top_policies[: max(top_n + len(baseline_policies), top_prediction_n)]

    by_well, bucket_metrics, subgroup_metrics, path_continuity, predictions = detailed_metrics(
        joined_frame,
        top_policies,
    )
    by_well_delta = summarize_by_well_delta(by_well)
    if not predictions.empty:
        keep_policies = [spec.policy for spec in top_policies[:top_prediction_n]]
        predictions = predictions[predictions["policy"].isin(keep_policies)].copy()

    artifacts = paths.artifacts_dir
    metrics_path = artifacts / f"{OUTPUT_PREFIX}_gate_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    by_well_delta_path = artifacts / f"{OUTPUT_PREFIX}_by_well_delta.csv"
    bucket_path = artifacts / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    subgroup_path = artifacts / f"{OUTPUT_PREFIX}_subgroup_metrics.csv"
    path_path = artifacts / f"{OUTPUT_PREFIX}_path_continuity.csv"
    cluster_path = artifacts / f"{OUTPUT_PREFIX}_cluster_outlier_well_features.csv"
    predictions_path = artifacts / f"{OUTPUT_PREFIX}_top_gated_predictions.csv.gz"
    schema_path = artifacts / f"{OUTPUT_PREFIX}_feature_schema.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    gate_metrics.to_csv(metrics_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    by_well_delta.to_csv(by_well_delta_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    subgroup_metrics.to_csv(subgroup_path, index=False)
    path_continuity.to_csv(path_path, index=False)
    cluster_features.to_csv(cluster_path, index=False)
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    feature_columns = list(
        dict.fromkeys(
            [
                *base_feature_cols,
                *[candidate.column for candidate in base_candidates],
                *[prior.prior_tvt for prior in priors],
            ]
        )
    )
    write_feature_schema(schema_path, feature_columns)

    best = gate_metrics.iloc[0].to_dict() if len(gate_metrics) else {}
    baseline_policy = f"{best.get('source_name')}__{best.get('model')}__baseline" if best else ""
    baseline = gate_metrics[gate_metrics["policy"] == baseline_policy]
    baseline_row = baseline.iloc[0].to_dict() if len(baseline) else {}
    gated_metrics = gate_metrics[
        (gate_metrics["prior"].astype(str) != "baseline")
        & (~gate_metrics["policy"].astype(str).str.contains("__reference__", regex=False))
    ].copy()
    best_gated = gated_metrics.iloc[0].to_dict() if len(gated_metrics) else {}
    gated_baseline_policy = (
        f"{best_gated.get('source_name')}__{best_gated.get('model')}__baseline"
        if best_gated
        else ""
    )
    gated_baseline = gate_metrics[gate_metrics["policy"] == gated_baseline_policy]
    gated_baseline_row = gated_baseline.iloc[0].to_dict() if len(gated_baseline) else {}
    best_by_well = (
        by_well_delta[by_well_delta["policy"] == best.get("policy")].iloc[0].to_dict()
        if len(by_well_delta) and best and (by_well_delta["policy"] == best.get("policy")).any()
        else {}
    )
    best_gated_by_well = (
        by_well_delta[by_well_delta["policy"] == best_gated.get("policy")].iloc[0].to_dict()
        if (
            len(by_well_delta)
            and best_gated
            and (by_well_delta["policy"] == best_gated.get("policy")).any()
        )
        else {}
    )
    decision = "cluster_outlier_pfbeam_prior_gate_not_supported"
    if best_gated and float(best_gated.get("delta_rmse_vs_baseline") or 0.0) < 0.0:
        max_regression = float(best_gated_by_well.get("max_regression_rmse", np.inf))
        decision = (
            "cluster_outlier_pfbeam_prior_gate_supported_for_review"
            if max_regression <= float(get_nested(config, "audit.max_regression_warn_rmse") or 0.25)
            else "global_gain_but_worst_well_warning"
        )

    summary = {
        "experiment": OUTPUT_PREFIX,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.time() - start,
        "rows": int(len(joined_frame)),
        "wells": int(joined_frame["well"].nunique()),
        "base_candidates": [candidate.__dict__ for candidate in base_candidates],
        "candidate_sources": source_metadata,
        "prior_sources": prior_meta,
        "cluster_features": cluster_meta,
        "exp115_roles": exp115_meta,
        "priors": [prior.__dict__ for prior in priors],
        "cluster_gates": [gate.raw for gate in cluster_gates],
        "quality_gates": [gate.__dict__ for gate in quality_gates],
        "alphas": alphas,
        "clips": clips,
        "reference_policies": get_nested(config, "audit.reference_policies") or [],
        "best_policy": to_jsonable(best),
        "baseline_policy": to_jsonable(baseline_row),
        "best_by_well_delta": to_jsonable(best_by_well),
        "best_gated_policy": to_jsonable(best_gated),
        "best_gated_baseline_policy": to_jsonable(gated_baseline_row),
        "best_gated_by_well_delta": to_jsonable(best_gated_by_well),
        "decision": decision,
        "artifacts": {
            "gate_metrics": str(metrics_path),
            "by_well": str(by_well_path),
            "by_well_delta": str(by_well_delta_path),
            "bucket_metrics": str(bucket_path),
            "subgroup_metrics": str(subgroup_path),
            "path_continuity": str(path_path),
            "cluster_outlier_well_features": str(cluster_path),
            "top_gated_predictions": str(predictions_path),
            "feature_schema": str(schema_path),
            "summary": str(summary_path),
        },
        "artifact_sha256": {
            "gate_metrics": sha256_path(metrics_path),
            "by_well": sha256_path(by_well_path),
            "by_well_delta": sha256_path(by_well_delta_path),
            "bucket_metrics": sha256_path(bucket_path),
            "subgroup_metrics": sha256_path(subgroup_path),
            "path_continuity": sha256_path(path_path),
            "cluster_outlier_well_features": sha256_path(cluster_path),
            "top_gated_predictions_raw": sha256_path(predictions_path),
            "top_gated_predictions_decompressed": sha256_path(predictions_path, decompressed=True),
            "feature_schema": sha256_path(schema_path),
        },
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    metrics_json = {
        "experiment": OUTPUT_PREFIX,
        "status": "completed_train_side_rejected_no_submit",
        "route": get_nested(config, "experiment.route"),
        "parent": get_nested(config, "lineage.parent"),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "best_policy": to_jsonable(best),
        "baseline_policy": to_jsonable(baseline_row),
        "best_by_well_delta": to_jsonable(best_by_well),
        "best_gated_policy": to_jsonable(best_gated),
        "best_gated_baseline_policy": to_jsonable(gated_baseline_row),
        "best_gated_by_well_delta": to_jsonable(best_gated_by_well),
        "decision": decision,
        "rows": int(len(joined_frame)),
        "wells": int(joined_frame["well"].nunique()),
        "summary_path": str(summary_path),
        "notes": (
            "Completed train-side cluster-outlier gated prior audit on PF/Beam candidates. "
            "Do not inference-port or submit direct posthoc correction without a stronger "
            "worst-well guard."
        ),
    }
    paths.metrics_path.write_text(
        json.dumps(to_jsonable(metrics_json), indent=2, sort_keys=True) + "\n"
    )
    return summary


if __name__ == "__main__":
    result = run_cluster_outlier_pfbeam_prior_gate()
    print(json.dumps(to_jsonable(result["best_policy"]), indent=2, sort_keys=True))

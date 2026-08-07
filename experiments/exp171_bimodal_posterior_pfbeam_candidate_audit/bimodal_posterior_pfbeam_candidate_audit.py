from __future__ import annotations

import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_PREFIX = "exp171_bimodal_posterior_pfbeam_candidate_audit"
EXP072_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)


@dataclass(frozen=True)
class FilteredSeries:
    name: str
    values: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    source_columns: tuple[str, ...]
    transform: str


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
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
    except TypeError:
        pass
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed(path: Path) -> str | None:
    if path.suffix != ".gz":
        return None
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def resolve_train_dir(path: str | Path) -> Path:
    train_dir = Path(path)
    if train_dir.exists():
        return train_dir
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for candidate in [
            input_root / "rogii-wellbore-geology-prediction" / "train",
            input_root / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        ]:
            if candidate.exists():
                return candidate
        for candidate in sorted(input_root.glob("**/train")):
            if any(candidate.glob("*__horizontal_well.csv")):
                return candidate
    return train_dir


def find_artifact(filename: str, explicit_path: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            Path("experiments")
            / "exp072_exp063_full_replay_feature_cache"
            / "artifacts"
            / filename,
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
        ]
    )
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(input_root.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def list_wells(train_dir: Path, audit_config: dict[str, Any]) -> list[str]:
    include = [str(value) for value in audit_config.get("well_include", []) if value]
    if include:
        wells = include
    else:
        wells = sorted(
            path.name.removesuffix("__horizontal_well.csv")
            for path in train_dir.glob("*__horizontal_well.csv")
        )
    max_wells = audit_config.get("max_wells")
    if max_wells is not None:
        wells = wells[: int(max_wells)]
    if not wells:
        raise FileNotFoundError(f"No train horizontal wells found under {train_dir}")
    return wells


def parse_row_id(well: str, row_idx: np.ndarray) -> np.ndarray:
    return np.asarray([f"{well}_{int(idx)}" for idx in row_idx], dtype=object)


def row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int32)


def fill_numeric(values: pd.Series | np.ndarray, fallback: float = 0.0) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    if series.notna().any():
        fallback = float(series.mean())
    filled = series.interpolate(limit_direction="both").ffill().bfill().fillna(fallback)
    return filled.to_numpy(np.float32)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(int(window), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(int(window), center=True, min_periods=1)
        .median()
        .to_numpy(np.float32)
    )


def savgol_or_rolling_mean(
    values: np.ndarray,
    *,
    window: int,
    polyorder: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    window = int(window)
    if window % 2 == 0:
        window += 1
    if len(values) < window or window <= int(polyorder):
        effective = max(3, min(len(values), window))
        return rolling_mean(values, effective), {
            "effective_kind": "rolling_mean_short_series",
            "window": int(effective),
        }
    try:
        from scipy.signal import savgol_filter

        return (
            savgol_filter(
                values,
                window_length=window,
                polyorder=int(polyorder),
                mode="interp",
            ).astype(np.float32),
            {"effective_kind": "savgol", "window": window, "polyorder": int(polyorder)},
        )
    except Exception as exc:  # pragma: no cover - depends on Kaggle image packages.
        return rolling_mean(values, window), {
            "effective_kind": "rolling_mean_fallback",
            "window": window,
            "polyorder": int(polyorder),
            "fallback_reason": type(exc).__name__,
        }


def build_filters(values: np.ndarray, filters_config: list[dict[str, Any]]) -> list[FilteredSeries]:
    filters: list[FilteredSeries] = []
    for spec in filters_config:
        name = str(spec["name"])
        kind = str(spec.get("kind", "raw"))
        if kind == "raw":
            filtered = values.astype(np.float32)
            metadata = {"effective_kind": "raw"}
        elif kind == "rolling_median":
            window = int(spec.get("window", 11))
            filtered = rolling_median(values, window)
            metadata = {"effective_kind": "rolling_median", "window": window}
        elif kind == "rolling_mean":
            window = int(spec.get("window", 21))
            filtered = rolling_mean(values, window)
            metadata = {"effective_kind": "rolling_mean", "window": window}
        elif kind == "savgol":
            filtered, metadata = savgol_or_rolling_mean(
                values,
                window=int(spec.get("window", 31)),
                polyorder=int(spec.get("polyorder", 2)),
            )
        else:
            raise ValueError(f"Unknown GR filter kind: {kind}")
        filters.append(
            FilteredSeries(name=name, values=filtered.astype(np.float32), metadata=metadata)
        )
    return filters


def deterministic_eval_indices(start: int, stop: int, max_rows: int) -> np.ndarray:
    if stop <= start:
        return np.zeros(0, dtype=np.int32)
    rows = np.arange(start, stop, dtype=np.int32)
    if len(rows) <= int(max_rows):
        return rows
    positions = np.linspace(0, len(rows) - 1, int(max_rows))
    selected = rows[np.unique(np.rint(positions).astype(np.int32))]
    return selected.astype(np.int32)


def prefix_slope_prior(
    *,
    md: np.ndarray,
    tvt_input: np.ndarray,
    known_end: int,
    slope_window_rows: int,
    slope_clip: tuple[float, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    if known_end <= 1:
        raise ValueError("known_end must include at least two prefix rows")
    fit_start = max(0, int(known_end) - int(slope_window_rows))
    fit_md = md[fit_start:known_end].astype(np.float64)
    fit_tvt = tvt_input[fit_start:known_end].astype(np.float64)
    finite = np.isfinite(fit_md) & np.isfinite(fit_tvt)
    if finite.sum() >= 2 and float(np.nanstd(fit_md[finite])) > 1e-6:
        slope, intercept = np.polyfit(fit_md[finite], fit_tvt[finite], deg=1)
    else:
        slope = 1.0
        intercept = float(tvt_input[known_end - 1] - md[known_end - 1])
    unclipped_slope = float(slope)
    lo, hi = slope_clip
    slope = float(np.clip(slope, lo, hi))
    last_md = float(md[known_end - 1])
    last_tvt = float(tvt_input[known_end - 1])
    prior = last_tvt + slope * (md.astype(np.float64) - last_md)
    return prior.astype(np.float32), {
        "known_end": int(known_end),
        "fit_start": int(fit_start),
        "fit_rows": int(finite.sum()),
        "unclipped_slope": unclipped_slope,
        "slope": slope,
        "intercept": float(intercept),
        "last_md": last_md,
        "last_tvt": last_tvt,
    }


def standardize_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    scale = values.std(axis=-1, keepdims=True) + 1e-6
    return centered / scale


def gather_horizontal(series: np.ndarray, centers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    idx = np.clip(centers[:, None] + offsets[None, :], 0, len(series) - 1)
    return series[idx].astype(np.float32)


def interpolate_typewell(
    type_tvt: np.ndarray,
    type_gr: np.ndarray,
    candidate_tvt: np.ndarray,
) -> np.ndarray:
    flat = np.interp(
        candidate_tvt.reshape(-1),
        type_tvt.astype(np.float64),
        type_gr.astype(np.float64),
        left=float(type_gr[0]),
        right=float(type_gr[-1]),
    )
    return flat.reshape(candidate_tvt.shape).astype(np.float32)


def parse_candidate_specs(audit_config: dict[str, Any]) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for raw in audit_config.get("pfbeam_candidates", []):
        source_columns = raw.get("source_columns")
        if source_columns is None:
            source_columns = [raw["source_column"]]
        specs.append(
            CandidateSpec(
                name=str(raw["name"]),
                source_columns=tuple(str(column) for column in source_columns),
                transform=str(raw["transform"]),
            )
        )
    return specs


def resolve_candidate_sources(header: list[str], specs: list[CandidateSpec]) -> dict[str, str]:
    header_set = set(header)
    resolved: dict[str, str] = {}
    missing: dict[str, list[str]] = {}
    for spec in specs:
        match = next((column for column in spec.source_columns if column in header_set), None)
        if match is None:
            missing[spec.name] = list(spec.source_columns)
        else:
            resolved[spec.name] = match
    if missing:
        detail = "; ".join(f"{name}: {columns}" for name, columns in missing.items())
        raise ValueError(f"feature cache is missing candidate source columns: {detail}")
    return resolved


def read_candidate_cache(
    *,
    config: dict[str, Any],
    audit_config: dict[str, Any],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    specs = parse_candidate_specs(audit_config)
    if not specs:
        return None, {"enabled": False, "reason": "no_candidate_specs"}
    source = find_artifact(
        EXP072_TRAIN_FEATURES,
        get_nested(config, "data.exp072_train_feature_cache_local"),
    )
    if source is None:
        return None, {
            "enabled": False,
            "reason": "candidate_cache_not_found",
            "filename": EXP072_TRAIN_FEATURES,
        }

    header = pd.read_csv(source, nrows=0).columns.tolist()
    resolved = resolve_candidate_sources(header, specs)
    required = {"id", "well", "last_known_tvt"}
    optional = {"target", "md_since", "eval_len"}
    usecols = sorted(required | optional.intersection(header) | set(resolved.values()))
    frame = pd.read_csv(source, usecols=usecols, dtype={"id": str, "well": str}, low_memory=False)
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    frame["row_idx"] = row_indices_from_ids(frame["id"])
    last_known = pd.to_numeric(frame["last_known_tvt"], errors="coerce").to_numpy(np.float32)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    for spec in specs:
        source_column = resolved[spec.name]
        values = frame[source_column].to_numpy(np.float32)
        if spec.transform == "absolute":
            pred = values
        elif spec.transform == "base_plus_delta":
            pred = last_known + values if source_column.endswith("_d") else values
        else:
            raise ValueError(f"unsupported candidate transform: {spec.transform}")
        frame[spec.name] = pred.astype(np.float32)
    keep = ["id", "well", "row_idx", *[spec.name for spec in specs]]
    metadata = {
        "enabled": True,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_decompressed_sha256": sha256_decompressed(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "candidate_names": [spec.name for spec in specs],
        "resolved_sources": resolved,
    }
    return frame[keep], metadata


def local_minima_positions(cost: np.ndarray) -> list[int]:
    if len(cost) == 0:
        return []
    positions = [0] if cost[0] <= cost[1] else []
    for i in range(1, len(cost) - 1):
        if cost[i] <= cost[i - 1] and cost[i] <= cost[i + 1]:
            positions.append(i)
    if len(cost) > 1 and cost[-1] <= cost[-2]:
        positions.append(len(cost) - 1)
    return positions


def choose_top2_modes(
    cost: np.ndarray,
    shifts: np.ndarray,
    *,
    min_separation_ft: float,
    max_separation_ft: float,
) -> tuple[int, int, bool]:
    order = np.argsort(cost)
    top1 = int(order[0])
    local_positions = local_minima_positions(cost)
    local_sorted = sorted(local_positions, key=lambda pos: float(cost[pos]))
    for pos in local_sorted:
        separation = abs(float(shifts[pos] - shifts[top1]))
        if min_separation_ft <= separation <= max_separation_ft:
            return top1, int(pos), True
    for pos in order[1:]:
        separation = abs(float(shifts[pos] - shifts[top1]))
        if min_separation_ft <= separation <= max_separation_ft:
            return top1, int(pos), False
    second = int(order[1]) if len(order) > 1 else top1
    return top1, second, False


def scan_bimodal_posterior_for_region(
    *,
    well: str,
    region: str,
    row_idx: np.ndarray,
    md: np.ndarray,
    true_tvt: np.ndarray,
    prior_tvt_all: np.ndarray,
    horizontal_gr: np.ndarray,
    type_tvt: np.ndarray,
    type_gr: np.ndarray,
    shifts: np.ndarray,
    local_offsets: np.ndarray,
    ncc_weight: float,
    posterior_temperatures: list[float],
    min_mode_separation_ft: float,
    max_mode_separation_ft: float,
    candidate_tvt_by_name: dict[str, np.ndarray],
) -> pd.DataFrame:
    if row_idx.size == 0:
        return pd.DataFrame()

    eval_gr = gather_horizontal(horizontal_gr, row_idx, local_offsets)
    local_rows = np.clip(row_idx[:, None] + local_offsets[None, :], 0, len(prior_tvt_all) - 1)
    local_prior = prior_tvt_all[local_rows]
    candidate_tvt = local_prior[:, None, :] + shifts[None, :, None]
    candidate_gr = interpolate_typewell(type_tvt, type_gr, candidate_tvt)

    mae = np.mean(np.abs(candidate_gr - eval_gr[:, None, :]), axis=2)
    eval_norm = standardize_rows(eval_gr)
    cand_norm = standardize_rows(candidate_gr)
    ncc = np.mean(cand_norm * eval_norm[:, None, :], axis=2)
    cost = mae - float(ncc_weight) * ncc

    rows: list[dict[str, Any]] = []
    prior_center = prior_tvt_all[row_idx].astype(np.float32)
    truth = true_tvt[row_idx].astype(np.float32)
    for i, idx in enumerate(row_idx):
        top1_pos, top2_pos, top2_is_local_min = choose_top2_modes(
            cost[i],
            shifts,
            min_separation_ft=float(min_mode_separation_ft),
            max_separation_ft=float(max_mode_separation_ft),
        )
        top1_cost = float(cost[i, top1_pos])
        top2_cost = float(cost[i, top2_pos])
        top1_shift = float(shifts[top1_pos])
        top2_shift = float(shifts[top2_pos])
        top1_tvt = float(prior_center[i] + top1_shift)
        top2_tvt = float(prior_center[i] + top2_shift)
        separation = abs(top2_tvt - top1_tvt)
        bimodal_flag = (
            top2_is_local_min
            and float(min_mode_separation_ft) <= separation <= float(max_mode_separation_ft)
        )
        base = {
            "id": f"{well}_{int(idx)}",
            "well": well,
            "eval_region": region,
            "row_idx": int(idx),
            "md": float(md[idx]),
            "true_tvt": float(truth[i]),
            "prior_center_tvt": float(prior_center[i]),
            "prior_error": float(prior_center[i] - truth[i]),
            "top1_shift_ft": top1_shift,
            "top2_shift_ft": top2_shift,
            "top1_tvt": top1_tvt,
            "top2_tvt": top2_tvt,
            "top1_cost": top1_cost,
            "top2_cost": top2_cost,
            "top2_minus_top1_cost": top2_cost - top1_cost,
            "mode_separation_ft": float(separation),
            "top2_is_local_min": bool(top2_is_local_min),
            "bimodal_flag": bool(bimodal_flag),
            "commit_top1_pred": top1_tvt,
            "commit_top2_pred": top2_tvt,
            "midpoint_pred": 0.5 * (top1_tvt + top2_tvt),
        }
        for temperature in posterior_temperatures:
            logits = -np.asarray([top1_cost, top2_cost], dtype=np.float64) / max(
                float(temperature), 1e-6
            )
            logits = logits - float(np.max(logits))
            probs = np.exp(np.clip(logits, -80.0, 80.0))
            probs = probs / (float(probs.sum()) + 1e-12)
            pred = float(probs[0] * top1_tvt + probs[1] * top2_tvt)
            entropy = -float(np.sum(probs * np.log(probs + 1e-12)) / np.log(2.0))
            suffix = f"t{temperature:g}".replace(".", "p")
            base[f"posterior_mean_{suffix}_pred"] = pred
            base[f"posterior_p_top1_{suffix}"] = float(probs[0])
            base[f"posterior_entropy_{suffix}"] = entropy
        for name, candidate_values in candidate_tvt_by_name.items():
            value = float(candidate_values[i]) if np.isfinite(candidate_values[i]) else np.nan
            base[f"{name}_pred"] = value
        rows.append(base)
    return pd.DataFrame(rows)


def make_distance_bucket(values: pd.Series | np.ndarray) -> pd.Series:
    return (
        pd.cut(
            pd.to_numeric(values, errors="coerce"),
            bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
            labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
            include_lowest=True,
        )
        .astype("string")
        .fillna("unknown")
    )


def make_separation_bucket(values: pd.Series | np.ndarray) -> pd.Series:
    return (
        pd.cut(
            pd.to_numeric(values, errors="coerce"),
            bins=[-np.inf, 6.0, 12.0, 18.0, 24.0, 30.0, np.inf],
            labels=["lt06", "06_12", "12_18", "18_24", "24_30", "gt30"],
            include_lowest=True,
        )
        .astype("string")
        .fillna("unknown")
    )


def prediction_columns(row_context: pd.DataFrame) -> list[str]:
    return sorted(column for column in row_context.columns if column.endswith("_pred"))


def score_prediction(group: pd.DataFrame, pred_col: str) -> dict[str, Any]:
    pred = pd.to_numeric(group[pred_col], errors="coerce").to_numpy(np.float64)
    true = pd.to_numeric(group["true_tvt"], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(pred) & np.isfinite(true)
    if not finite.any():
        return {
            "rows": int(len(group)),
            "scored_rows": 0,
            "rmse_tvt": None,
            "mae_tvt": None,
            "bias_tvt": None,
            "within2": None,
            "within5": None,
            "within10": None,
        }
    error = pred[finite] - true[finite]
    abs_error = np.abs(error)
    return {
        "rows": int(len(group)),
        "scored_rows": int(finite.sum()),
        "rmse_tvt": float(np.sqrt(np.mean(error * error))),
        "mae_tvt": float(np.mean(abs_error)),
        "bias_tvt": float(np.mean(error)),
        "within2": float(np.mean(abs_error <= 2.0)),
        "within5": float(np.mean(abs_error <= 5.0)),
        "within10": float(np.mean(abs_error <= 10.0)),
    }


def summarize_candidate_metrics(row_context: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pred_cols = prediction_columns(row_context)
    group_cols = ["surface", "filter", "eval_region"]
    for keys, group in row_context.groupby(group_cols, sort=False):
        for pred_col in pred_cols:
            rows.append(
                {
                    **dict(zip(group_cols, keys, strict=True)),
                    "candidate": pred_col.removesuffix("_pred"),
                    **score_prediction(group, pred_col),
                }
            )
    for keys, group in row_context.groupby(["surface", "filter"], sort=False):
        for pred_col in pred_cols:
            rows.append(
                {
                    "surface": keys[0],
                    "filter": keys[1],
                    "eval_region": "all",
                    "candidate": pred_col.removesuffix("_pred"),
                    **score_prediction(group, pred_col),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["eval_region", "rmse_tvt", "candidate"], na_position="last"
    )


def summarize_bucket_metrics(row_context: pd.DataFrame) -> pd.DataFrame:
    frame = row_context.copy()
    frame["distance_bucket"] = make_distance_bucket(frame["distance_from_known_prefix"])
    frame["mode_separation_bucket"] = make_separation_bucket(frame["mode_separation_ft"])
    frame["bimodal_bucket"] = np.where(frame["bimodal_flag"].astype(bool), "bimodal", "unimodal")
    bucket_specs = [
        ("distance", "distance_bucket"),
        ("mode_separation", "mode_separation_bucket"),
        ("bimodal_flag", "bimodal_bucket"),
    ]
    rows: list[dict[str, Any]] = []
    pred_cols = prediction_columns(frame)
    for bucket_type, bucket_col in bucket_specs:
        group_cols = ["surface", "filter", "eval_region", bucket_col]
        for keys, group in frame.groupby(group_cols, sort=False):
            for pred_col in pred_cols:
                rows.append(
                    {
                        "surface": keys[0],
                        "filter": keys[1],
                        "eval_region": keys[2],
                        "bucket_type": bucket_type,
                        "bucket": keys[3],
                        "candidate": pred_col.removesuffix("_pred"),
                        **score_prediction(group, pred_col),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["bucket_type", "eval_region", "bucket", "rmse_tvt", "candidate"],
        na_position="last",
    )


def summarize_well_metrics(row_context: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pred_cols = prediction_columns(row_context)
    for keys, group in row_context.groupby(
        ["surface", "filter", "eval_region", "well"],
        sort=False,
    ):
        for pred_col in pred_cols:
            rows.append(
                {
                    "surface": keys[0],
                    "filter": keys[1],
                    "eval_region": keys[2],
                    "well": keys[3],
                    "candidate": pred_col.removesuffix("_pred"),
                    **score_prediction(group, pred_col),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["candidate", "eval_region", "rmse_tvt"], ascending=[True, True, False]
    )


def summarize_posterior_vs_commit(row_context: pd.DataFrame) -> pd.DataFrame:
    commit = row_context[["id", "surface", "eval_region", "commit_top1_pred", "true_tvt"]].copy()
    commit["commit_abs_error"] = np.abs(commit["commit_top1_pred"] - commit["true_tvt"])
    rows: list[dict[str, Any]] = []
    for pred_col in prediction_columns(row_context):
        if pred_col == "commit_top1_pred":
            continue
        work = row_context[["id", "surface", "eval_region", pred_col, "true_tvt"]].merge(
            commit[["id", "surface", "eval_region", "commit_abs_error"]],
            on=["id", "surface", "eval_region"],
            how="left",
            validate="one_to_one",
        )
        work["abs_error"] = np.abs(work[pred_col] - work["true_tvt"])
        work["gain_vs_commit"] = work["commit_abs_error"] - work["abs_error"]
        for keys, group in work.groupby(["surface", "eval_region"], sort=False):
            finite = np.isfinite(group["gain_vs_commit"])
            rows.append(
                {
                    "surface": keys[0],
                    "eval_region": keys[1],
                    "candidate": pred_col.removesuffix("_pred"),
                    "rows": int(finite.sum()),
                    "mean_abs_error_gain_vs_commit": float(
                        group.loc[finite, "gain_vs_commit"].mean()
                    ),
                    "median_abs_error_gain_vs_commit": float(
                        group.loc[finite, "gain_vs_commit"].median()
                    ),
                    "improved_rate_vs_commit": float(
                        (group.loc[finite, "gain_vs_commit"] > 0).mean()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["eval_region", "mean_abs_error_gain_vs_commit"], ascending=[True, False]
    )


def candidate_values_for_rows(
    candidate_cache_by_well: dict[str, pd.DataFrame],
    well: str,
    row_idx: np.ndarray,
    candidate_names: list[str],
) -> dict[str, np.ndarray]:
    if well not in candidate_cache_by_well:
        return {}
    frame = candidate_cache_by_well[well].set_index("row_idx", drop=False)
    result: dict[str, np.ndarray] = {}
    row_index = pd.Index(row_idx)
    for name in candidate_names:
        values = np.full(len(row_idx), np.nan, dtype=np.float32)
        if name not in frame.columns:
            continue
        common = np.intersect1d(row_idx, frame.index.to_numpy(np.int32), assume_unique=False)
        if common.size:
            pos = row_index.get_indexer(common)
            values[pos] = frame.loc[common, name].to_numpy(np.float32)
        result[name] = values
    return result


def build_well_audit(
    *,
    well: str,
    train_dir: Path,
    audit_config: dict[str, Any],
    candidate_cache_by_well: dict[str, pd.DataFrame],
    candidate_names: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    if not horizontal_path.exists() or not typewell_path.exists():
        raise FileNotFoundError(f"missing raw train files for well {well}")

    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    required_horizontal = {"MD", "TVT", "GR", "TVT_input"}
    required_typewell = {"TVT", "GR"}
    missing_h = required_horizontal - set(horizontal.columns)
    missing_t = required_typewell - set(typewell.columns)
    if missing_h or missing_t:
        raise ValueError(f"{well} missing columns: horizontal={missing_h}, typewell={missing_t}")

    md = fill_numeric(horizontal["MD"])
    true_tvt = fill_numeric(horizontal["TVT"])
    tvt_input_raw = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
    known = tvt_input_raw.notna().to_numpy()
    if not known.any():
        raise ValueError(f"No finite TVT_input prefix for {well}")
    known_end = int(np.flatnonzero(known)[-1] + 1)
    min_prefix = int(audit_config.get("min_prefix_rows", 80))
    if known_end < min_prefix:
        return pd.DataFrame(), [
            {
                "well": well,
                "skipped": True,
                "reason": "short_prefix",
                "known_prefix_rows": known_end,
                "horizontal_rows": int(len(horizontal)),
                "typewell_rows": int(len(typewell)),
            }
        ]
    tvt_input = fill_numeric(tvt_input_raw)

    type_sorted = typewell[["TVT", "GR"]].dropna().sort_values("TVT")
    type_tvt = pd.to_numeric(type_sorted["TVT"], errors="coerce").to_numpy(np.float32)
    type_gr = fill_numeric(type_sorted["GR"])
    type_window = int(audit_config.get("typewell_smooth_window", 5))
    type_gr = rolling_mean(type_gr, type_window)
    if len(type_tvt) < 4:
        raise ValueError(f"Typewell GR too short for {well}")

    full_gr = fill_numeric(horizontal["GR"])
    filters = build_filters(
        full_gr,
        list(audit_config.get("filters", [{"name": "raw", "kind": "raw"}])),
    )
    max_eval = int(audit_config.get("max_eval_rows_per_region_per_well", 256))
    prefix_backtest_tail = int(audit_config.get("prefix_backtest_tail_rows", 256))
    slope_window = int(audit_config.get("prefix_slope_window_rows", 80))
    slope_clip_config = audit_config.get("slope_clip", [-3.0, 3.0])
    slope_clip = (float(slope_clip_config[0]), float(slope_clip_config[1]))
    shifts = np.arange(
        float(audit_config.get("shift_min_ft", -80.0)),
        float(audit_config.get("shift_max_ft", 80.0))
        + 0.5 * float(audit_config.get("shift_step_ft", 2.0)),
        float(audit_config.get("shift_step_ft", 2.0)),
        dtype=np.float32,
    )
    local_offsets = np.asarray(
        [int(value) for value in audit_config.get("local_offsets_rows", [-24, -12, 0, 12, 24])],
        dtype=np.int32,
    )

    regions: list[tuple[str, int, np.ndarray]] = []
    hidden_rows = deterministic_eval_indices(known_end, len(horizontal), max_eval)
    if hidden_rows.size:
        regions.append(("hidden_tail", known_end, hidden_rows))
    backtest_start = max(min_prefix, known_end - prefix_backtest_tail)
    backtest_rows = deterministic_eval_indices(backtest_start, known_end, max_eval)
    if backtest_rows.size and backtest_start >= min_prefix:
        regions.append(("prefix_backtest", backtest_start, backtest_rows))

    row_frames: list[pd.DataFrame] = []
    input_rows: list[dict[str, Any]] = []
    for region, region_known_end, row_idx in regions:
        prior, prior_meta = prefix_slope_prior(
            md=md,
            tvt_input=tvt_input,
            known_end=region_known_end,
            slope_window_rows=slope_window,
            slope_clip=slope_clip,
        )
        candidate_tvt_by_name = (
            candidate_values_for_rows(
                candidate_cache_by_well,
                well,
                row_idx,
                candidate_names,
            )
            if region == "hidden_tail"
            else {}
        )
        for filtered in filters:
            frame = scan_bimodal_posterior_for_region(
                well=well,
                region=region,
                row_idx=row_idx,
                md=md,
                true_tvt=true_tvt,
                prior_tvt_all=prior,
                horizontal_gr=filtered.values,
                type_tvt=type_tvt,
                type_gr=type_gr,
                shifts=shifts,
                local_offsets=local_offsets,
                ncc_weight=float(audit_config.get("ncc_weight", 8.0)),
                posterior_temperatures=[
                    float(value) for value in audit_config.get("posterior_temperatures", [4.0])
                ],
                min_mode_separation_ft=float(audit_config.get("min_mode_separation_ft", 6.0)),
                max_mode_separation_ft=float(audit_config.get("max_mode_separation_ft", 30.0)),
                candidate_tvt_by_name=candidate_tvt_by_name,
            )
            if frame.empty:
                continue
            surface = f"{filtered.name}__bimodal_posterior"
            frame.insert(3, "filter", filtered.name)
            frame.insert(4, "surface", surface)
            frame["known_prefix_rows"] = int(region_known_end)
            frame["distance_from_known_prefix"] = (
                frame["row_idx"] - int(region_known_end)
            ).astype(np.float32)
            row_frames.append(frame)
            input_rows.append(
                {
                    "well": well,
                    "filter": filtered.name,
                    "surface": surface,
                    "eval_region": region,
                    "horizontal_rows": int(len(horizontal)),
                    "typewell_rows": int(len(typewell)),
                    "known_prefix_rows": int(region_known_end),
                    "eval_rows": int(len(row_idx)),
                    "gr_missing_rate": float(pd.isna(horizontal["GR"]).mean()),
                    "typewell_gr_missing_rate": float(pd.isna(typewell["GR"]).mean()),
                    "horizontal_sha256": sha256_file(horizontal_path),
                    "typewell_sha256": sha256_file(typewell_path),
                    **{f"prior_{key}": value for key, value in prior_meta.items()},
                    "filter_metadata": json.dumps(to_jsonable(filtered.metadata), sort_keys=True),
                }
            )

    if not row_frames:
        return pd.DataFrame(), input_rows
    return pd.concat(row_frames, ignore_index=True), input_rows


def run_bimodal_posterior_pfbeam_candidate_audit(
    *,
    config: dict[str, Any],
    train_dir: str | Path,
    output_dir: str | Path,
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    start = time.time()
    audit_config = get_nested(config, "audit", {})
    output_prefix = str(audit_config.get("output_prefix", OUTPUT_PREFIX))
    train_dir = resolve_train_dir(train_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_cache, candidate_metadata = read_candidate_cache(
        config=config,
        audit_config=audit_config,
    )
    candidate_names = (
        list(candidate_metadata.get("candidate_names", []))
        if candidate_metadata.get("enabled")
        else []
    )
    candidate_cache_by_well: dict[str, pd.DataFrame] = {}
    if candidate_cache is not None:
        candidate_cache_by_well = {
            str(well): group.reset_index(drop=True)
            for well, group in candidate_cache.groupby("well", sort=False)
        }

    wells = list_wells(train_dir, audit_config)
    row_frames: list[pd.DataFrame] = []
    input_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for i, well in enumerate(wells, start=1):
        if i == 1 or i % 50 == 0 or i == len(wells):
            print(f"[{i}/{len(wells)}] auditing {well}")
        try:
            frame, well_inputs = build_well_audit(
                well=well,
                train_dir=train_dir,
                audit_config=audit_config,
                candidate_cache_by_well=candidate_cache_by_well,
                candidate_names=candidate_names,
            )
        except Exception as exc:
            skipped.append({"well": well, "reason": type(exc).__name__, "message": str(exc)})
            continue
        if not frame.empty:
            row_frames.append(frame)
        input_rows.extend(well_inputs)

    if not row_frames:
        raise RuntimeError("No row contexts were generated for bimodal posterior audit")
    row_context = pd.concat(row_frames, ignore_index=True)
    candidate_metrics = summarize_candidate_metrics(row_context)
    bucket_metrics = summarize_bucket_metrics(row_context)
    well_metrics = summarize_well_metrics(row_context)
    gain_vs_commit = summarize_posterior_vs_commit(row_context)
    input_summary = pd.DataFrame(input_rows)

    row_context_path = output_dir / f"{output_prefix}_row_context.csv.gz"
    candidate_metrics_path = output_dir / f"{output_prefix}_candidate_metrics.csv"
    bucket_metrics_path = output_dir / f"{output_prefix}_bucket_metrics.csv"
    well_metrics_path = output_dir / f"{output_prefix}_well_metrics.csv"
    gain_path = output_dir / f"{output_prefix}_gain_vs_commit.csv"
    input_summary_path = output_dir / f"{output_prefix}_well_input_summary.csv"
    summary_path = output_dir / f"{output_prefix}_summary.json"

    row_context.to_csv(row_context_path, index=False, compression="gzip")
    candidate_metrics.to_csv(candidate_metrics_path, index=False)
    bucket_metrics.to_csv(bucket_metrics_path, index=False)
    well_metrics.to_csv(well_metrics_path, index=False)
    gain_vs_commit.to_csv(gain_path, index=False)
    input_summary.to_csv(input_summary_path, index=False)

    all_metrics = candidate_metrics[candidate_metrics["eval_region"].eq("all")]
    best_all = all_metrics.sort_values("rmse_tvt").head(1).to_dict("records")
    commit_all = all_metrics[all_metrics["candidate"].eq("commit_top1")].to_dict("records")
    posterior_all = all_metrics[
        all_metrics["candidate"].str.startswith("posterior_mean", na=False)
    ].sort_values("rmse_tvt")
    likpf_all = all_metrics[all_metrics["candidate"].eq("likpf_mean")].to_dict("records")
    bimodal_rate = float(row_context["bimodal_flag"].mean())
    summary: dict[str, Any] = {
        "experiment": get_nested(
            config,
            "experiment.name",
            "exp171_bimodal_posterior_pfbeam_candidate_audit",
        ),
        "status": "completed_train_side_rejected_no_submit",
        "route": get_nested(config, "experiment.route", "pf_beam"),
        "train_dir": str(train_dir),
        "wells_requested": int(len(wells)),
        "wells_with_rows": int(row_context["well"].nunique()),
        "skipped_wells": skipped,
        "rows": int(len(row_context)),
        "surfaces": sorted(row_context["surface"].unique().tolist()),
        "filters": sorted(row_context["filter"].unique().tolist()),
        "eval_regions": sorted(row_context["eval_region"].unique().tolist()),
        "candidate_cache": candidate_metadata,
        "bimodal_rate": bimodal_rate,
        "commit_top1_all": commit_all[0] if commit_all else None,
        "best_posterior_all": posterior_all.head(1).to_dict("records")[0]
        if not posterior_all.empty
        else None,
        "best_all": best_all[0] if best_all else None,
        "likpf_mean_all": likpf_all[0] if likpf_all else None,
        "decision": {
            "recommendation": "diagnostic_only_until_kaggle_result_review",
            "reason": (
                "This audit evaluates fixed temperature posterior means from target-free "
                "top2 GR modes. Any candidate replacement, ML feature adoption, inference "
                "port, or submission requires separate parity and stress checks."
            ),
        },
        "artifacts": {
            "row_context": str(row_context_path),
            "candidate_metrics": str(candidate_metrics_path),
            "bucket_metrics": str(bucket_metrics_path),
            "well_metrics": str(well_metrics_path),
            "gain_vs_commit": str(gain_path),
            "well_input_summary": str(input_summary_path),
            "summary": str(summary_path),
        },
        "artifact_sha256": {
            "row_context_gzip": sha256_file(row_context_path),
            "row_context_decompressed": sha256_decompressed(row_context_path),
            "candidate_metrics": sha256_file(candidate_metrics_path),
            "bucket_metrics": sha256_file(bucket_metrics_path),
            "well_metrics": sha256_file(well_metrics_path),
            "gain_vs_commit": sha256_file(gain_path),
            "well_input_summary": sha256_file(input_summary_path),
        },
        "runtime_sec": float(time.time() - start),
        "audit_config": audit_config,
    }
    write_json(summary_path, summary)
    if metrics_path is not None:
        write_json(
            Path(metrics_path),
            {
                "experiment": summary["experiment"],
                "status": summary["status"],
                "metric": "rmse",
                "cv": None,
                "public_lb": None,
                "private_lb": None,
                "commit_top1_all": summary["commit_top1_all"],
                "best_posterior_all": summary["best_posterior_all"],
                "best_all": summary["best_all"],
                "likpf_mean_all": summary["likpf_mean_all"],
                "rows": summary["rows"],
                "wells_with_rows": summary["wells_with_rows"],
                "summary_path": str(summary_path),
                "notes": "Kaggle train-side diagnostic is implemented but not yet executed.",
            },
        )
    return {
        "summary": summary,
        "row_context": row_context,
        "candidate_metrics": candidate_metrics,
        "bucket_metrics": bucket_metrics,
        "well_metrics": well_metrics,
        "gain_vs_commit": gain_vs_commit,
        "input_summary": input_summary,
    }

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

OUTPUT_PREFIX = "exp167_fft_denoised_gr_matching_audit"


@dataclass(frozen=True)
class FilteredSeries:
    name: str
    values: np.ndarray
    metadata: dict[str, Any]


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
        return rolling_mean(values, max(3, min(len(values), window))), {
            "effective_kind": "rolling_mean_short_series",
            "window": int(max(3, min(len(values), window))),
        }
    try:
        from scipy.signal import savgol_filter

        return (
            savgol_filter(
                values, window_length=window, polyorder=int(polyorder), mode="interp"
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


def fft_notch(
    values: np.ndarray,
    *,
    max_peaks: int,
    min_period_ft: float,
    max_period_ft: float,
    notch_width_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = int(len(values))
    if n < 8:
        return values.astype(np.float32), {"effective_kind": "raw_short_series", "peaks": []}

    centered = values.astype(np.float64) - float(np.mean(values))
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(n, d=1.0)
    power = np.abs(spectrum) ** 2

    min_freq = 1.0 / max(float(max_period_ft), 1e-6)
    max_freq = 1.0 / max(float(min_period_ft), 1e-6)
    valid = (freqs >= min_freq) & (freqs <= max_freq)
    valid[0] = False
    candidate_idx = np.flatnonzero(valid)
    if candidate_idx.size == 0:
        return values.astype(np.float32), {
            "effective_kind": "raw_no_valid_fft_band",
            "peaks": [],
        }

    order = candidate_idx[np.argsort(power[candidate_idx])[::-1]]
    selected: list[int] = []
    width = int(max(1, notch_width_bins))
    for idx in order:
        if all(abs(int(idx) - prev) > width for prev in selected):
            selected.append(int(idx))
        if len(selected) >= int(max_peaks):
            break

    filtered = spectrum.copy()
    for idx in selected:
        lo = max(1, idx - width)
        hi = min(len(filtered) - 1, idx + width)
        filtered[lo : hi + 1] = 0.0
    denoised = np.fft.irfft(filtered, n=n) + float(np.mean(values))
    peaks = [
        {
            "bin": int(idx),
            "frequency_cyc_per_ft": float(freqs[idx]),
            "period_ft": float(1.0 / freqs[idx]) if freqs[idx] > 0 else None,
            "power": float(power[idx]),
        }
        for idx in selected
    ]
    return denoised.astype(np.float32), {
        "effective_kind": "fft_notch",
        "max_peaks": int(max_peaks),
        "notch_width_bins": width,
        "min_period_ft": float(min_period_ft),
        "max_period_ft": float(max_period_ft),
        "peaks": peaks,
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
        elif kind == "fft_notch":
            filtered, metadata = fft_notch(
                values,
                max_peaks=int(spec.get("max_peaks", 2)),
                min_period_ft=float(spec.get("min_period_ft", 6.0)),
                max_period_ft=float(spec.get("max_period_ft", 120.0)),
                notch_width_bins=int(spec.get("notch_width_bins", 2)),
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
    type_tvt: np.ndarray, type_gr: np.ndarray, candidate_tvt: np.ndarray
) -> np.ndarray:
    flat = np.interp(
        candidate_tvt.reshape(-1),
        type_tvt.astype(np.float64),
        type_gr.astype(np.float64),
        left=float(type_gr[0]),
        right=float(type_gr[-1]),
    )
    return flat.reshape(candidate_tvt.shape).astype(np.float32)


def scan_filter_for_region(
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
    score_temperature: float,
    ncc_weight: float,
    decoy_offsets_ft: np.ndarray,
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
    best_pos = np.argmin(cost, axis=1)
    best_cost = cost[np.arange(len(row_idx)), best_pos]
    if cost.shape[1] > 1:
        partitioned = np.partition(cost, 1, axis=1)
        second_cost = partitioned[:, 1]
    else:
        second_cost = best_cost

    logits = -(cost - cost.min(axis=1, keepdims=True)) / max(float(score_temperature), 1e-6)
    weights = np.exp(np.clip(logits, -80.0, 80.0))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-12
    entropy = -np.sum(weights * np.log(weights + 1e-12), axis=1) / np.log(cost.shape[1])

    best_shift = shifts[best_pos]
    prior_center = prior_tvt_all[row_idx]
    pred_tvt = prior_center + best_shift
    error = pred_tvt - true_tvt[row_idx]

    decoy_gaps = []
    for offset in decoy_offsets_ft:
        target_shifts = np.concatenate([best_shift - float(offset), best_shift + float(offset)])
        shift_pos = np.abs(shifts[None, :] - target_shifts[:, None]).argmin(axis=1)
        first = cost[np.arange(len(row_idx)), shift_pos[: len(row_idx)]]
        second = cost[np.arange(len(row_idx)), shift_pos[len(row_idx) :]]
        decoy_gaps.append(np.minimum(first, second) - best_cost)
    decoy_gap = (
        np.min(np.stack(decoy_gaps, axis=1), axis=1) if decoy_gaps else second_cost - best_cost
    )

    best_ncc = ncc[np.arange(len(row_idx)), best_pos]
    best_mae = mae[np.arange(len(row_idx)), best_pos]
    return pd.DataFrame(
        {
            "id": parse_row_id(well, row_idx),
            "well": well,
            "eval_region": region,
            "row_idx": row_idx.astype(np.int32),
            "md": md[row_idx].astype(np.float32),
            "distance_from_region_prefix": (row_idx - int(row_idx.min())).astype(np.float32),
            "true_tvt": true_tvt[row_idx].astype(np.float32),
            "prior_center_tvt": prior_center.astype(np.float32),
            "prior_error": (prior_center - true_tvt[row_idx]).astype(np.float32),
            "pred_tvt": pred_tvt.astype(np.float32),
            "best_shift_ft": best_shift.astype(np.float32),
            "error": error.astype(np.float32),
            "abs_error": np.abs(error).astype(np.float32),
            "best_cost": best_cost.astype(np.float32),
            "top1_top2_cost_gap": (second_cost - best_cost).astype(np.float32),
            "entropy": entropy.astype(np.float32),
            "decoy_gap_15_25ft": decoy_gap.astype(np.float32),
            "best_ncc": best_ncc.astype(np.float32),
            "best_gr_mae": best_mae.astype(np.float32),
        }
    )


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


def metric_row(group: pd.DataFrame, *, keys: dict[str, Any]) -> dict[str, Any]:
    error = pd.to_numeric(group["error"], errors="coerce").to_numpy(np.float64)
    abs_error = np.abs(error)
    finite = np.isfinite(error)
    if not finite.any():
        metrics = {
            "rows": int(len(group)),
            "rmse_tvt": None,
            "mae_tvt": None,
            "bias_tvt": None,
            "within2": None,
            "within5": None,
            "within10": None,
            "gap_mean": None,
            "entropy_mean": None,
            "decoy_gap_mean": None,
        }
    else:
        finite_error = error[finite]
        finite_abs = abs_error[finite]
        metrics = {
            "rows": int(finite.sum()),
            "rmse_tvt": float(np.sqrt(np.mean(np.square(finite_error)))),
            "mae_tvt": float(np.mean(finite_abs)),
            "bias_tvt": float(np.mean(finite_error)),
            "within2": float(np.mean(finite_abs <= 2.0)),
            "within5": float(np.mean(finite_abs <= 5.0)),
            "within10": float(np.mean(finite_abs <= 10.0)),
            "gap_mean": float(group["top1_top2_cost_gap"].mean()),
            "entropy_mean": float(group["entropy"].mean()),
            "decoy_gap_mean": float(group["decoy_gap_15_25ft"].mean()),
        }
    return {**keys, **metrics}


def summarize_metrics(row_context: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filter_rows = []
    for (filter_name, region), group in row_context.groupby(["filter", "eval_region"], sort=False):
        filter_rows.append(metric_row(group, keys={"filter": filter_name, "eval_region": region}))
    for filter_name, group in row_context.groupby("filter", sort=False):
        filter_rows.append(metric_row(group, keys={"filter": filter_name, "eval_region": "all"}))
    filter_metrics = pd.DataFrame(filter_rows).sort_values(["eval_region", "rmse_tvt", "filter"])

    bucket_rows = []
    row_context = row_context.copy()
    row_context["distance_bucket"] = make_distance_bucket(row_context["distance_from_known_prefix"])
    row_context["prior_error_bucket"] = make_distance_bucket(np.abs(row_context["prior_error"]))
    for (filter_name, region, bucket), group in row_context.groupby(
        ["filter", "eval_region", "distance_bucket"],
        sort=False,
    ):
        bucket_rows.append(
            metric_row(
                group,
                keys={
                    "filter": filter_name,
                    "eval_region": region,
                    "bucket_type": "distance",
                    "bucket": bucket,
                },
            )
        )
    for (filter_name, region, bucket), group in row_context.groupby(
        ["filter", "eval_region", "prior_error_bucket"],
        sort=False,
    ):
        bucket_rows.append(
            metric_row(
                group,
                keys={
                    "filter": filter_name,
                    "eval_region": region,
                    "bucket_type": "prior_error_abs",
                    "bucket": bucket,
                },
            )
        )
    bucket_metrics = pd.DataFrame(bucket_rows)

    well_rows = []
    for (filter_name, region, well), group in row_context.groupby(
        ["filter", "eval_region", "well"],
        sort=False,
    ):
        well_rows.append(
            metric_row(group, keys={"filter": filter_name, "eval_region": region, "well": well})
        )
    well_metrics = pd.DataFrame(well_rows).sort_values(
        ["filter", "eval_region", "rmse_tvt"], ascending=[True, True, False]
    )
    return filter_metrics, bucket_metrics, well_metrics


def summarize_gain_vs_raw(row_context: pd.DataFrame) -> pd.DataFrame:
    raw = row_context[row_context["filter"].eq("raw")][
        [
            "id",
            "well",
            "eval_region",
            "row_idx",
            "abs_error",
            "top1_top2_cost_gap",
            "entropy",
            "decoy_gap_15_25ft",
        ]
    ].rename(
        columns={
            "abs_error": "raw_abs_error",
            "top1_top2_cost_gap": "raw_gap",
            "entropy": "raw_entropy",
            "decoy_gap_15_25ft": "raw_decoy_gap",
        }
    )
    merged = row_context.merge(
        raw,
        on=["id", "well", "eval_region", "row_idx"],
        how="left",
        validate="many_to_one",
    )
    merged = merged[~merged["filter"].eq("raw")].copy()
    merged["abs_error_gain_vs_raw"] = merged["raw_abs_error"] - merged["abs_error"]
    merged["gap_gain_vs_raw"] = merged["top1_top2_cost_gap"] - merged["raw_gap"]
    merged["entropy_reduction_vs_raw"] = merged["raw_entropy"] - merged["entropy"]
    merged["decoy_gap_gain_vs_raw"] = merged["decoy_gap_15_25ft"] - merged["raw_decoy_gap"]
    rows = []
    for (filter_name, region), group in merged.groupby(["filter", "eval_region"], sort=False):
        rows.append(
            {
                "filter": filter_name,
                "eval_region": region,
                "rows": int(len(group)),
                "mean_abs_error_gain_vs_raw": float(group["abs_error_gain_vs_raw"].mean()),
                "median_abs_error_gain_vs_raw": float(group["abs_error_gain_vs_raw"].median()),
                "improved_rate_vs_raw": float((group["abs_error_gain_vs_raw"] > 0).mean()),
                "mean_gap_gain_vs_raw": float(group["gap_gain_vs_raw"].mean()),
                "mean_entropy_reduction_vs_raw": float(group["entropy_reduction_vs_raw"].mean()),
                "mean_decoy_gap_gain_vs_raw": float(group["decoy_gap_gain_vs_raw"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["eval_region", "mean_abs_error_gain_vs_raw"], ascending=[True, False]
    )


def build_well_audit(
    *,
    well: str,
    train_dir: Path,
    audit_config: dict[str, Any],
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
        full_gr, list(audit_config.get("filters", [{"name": "raw", "kind": "raw"}]))
    )
    max_eval = int(audit_config.get("max_eval_rows_per_region_per_well", 256))
    prefix_backtest_tail = int(audit_config.get("prefix_backtest_tail_rows", 256))
    slope_window = int(audit_config.get("prefix_slope_window_rows", 80))
    slope_clip_config = audit_config.get("slope_clip", [-3.0, 3.0])
    slope_clip = (float(slope_clip_config[0]), float(slope_clip_config[1]))
    shifts = np.arange(
        float(audit_config.get("shift_min_ft", -220.0)),
        float(audit_config.get("shift_max_ft", 220.0))
        + 0.5 * float(audit_config.get("shift_step_ft", 5.0)),
        float(audit_config.get("shift_step_ft", 5.0)),
        dtype=np.float32,
    )
    local_offsets = np.asarray(
        [int(value) for value in audit_config.get("local_offsets_rows", [-24, -12, 0, 12, 24])],
        dtype=np.int32,
    )
    decoy_offsets = np.asarray(
        [float(value) for value in audit_config.get("decoy_offsets_ft", [15.0, 20.0, 25.0])],
        dtype=np.float32,
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
        for filtered in filters:
            frame = scan_filter_for_region(
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
                score_temperature=float(audit_config.get("score_temperature", 6.0)),
                ncc_weight=float(audit_config.get("ncc_weight", 8.0)),
                decoy_offsets_ft=decoy_offsets,
            )
            if frame.empty:
                continue
            frame.insert(3, "filter", filtered.name)
            frame["known_prefix_rows"] = int(region_known_end)
            frame["distance_from_known_prefix"] = (frame["row_idx"] - int(region_known_end)).astype(
                np.float32
            )
            peak_periods = [
                peak.get("period_ft")
                for peak in filtered.metadata.get("peaks", [])
                if peak.get("period_ft") is not None
            ]
            frame["fft_primary_period_ft"] = float(peak_periods[0]) if peak_periods else np.nan
            row_frames.append(frame)
            input_rows.append(
                {
                    "well": well,
                    "filter": filtered.name,
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


def run_fft_denoised_gr_matching_audit(
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
            )
        except Exception as exc:
            skipped.append({"well": well, "reason": type(exc).__name__, "message": str(exc)})
            continue
        if not frame.empty:
            row_frames.append(frame)
        input_rows.extend(well_inputs)

    if not row_frames:
        raise RuntimeError("No row contexts were generated for FFT denoised GR matching audit")
    row_context = pd.concat(row_frames, ignore_index=True)
    filter_metrics, bucket_metrics, well_metrics = summarize_metrics(row_context)
    gain_vs_raw = summarize_gain_vs_raw(row_context)
    input_summary = pd.DataFrame(input_rows)

    row_context_path = output_dir / f"{output_prefix}_row_context.csv.gz"
    filter_metrics_path = output_dir / f"{output_prefix}_filter_metrics.csv"
    bucket_metrics_path = output_dir / f"{output_prefix}_bucket_metrics.csv"
    well_metrics_path = output_dir / f"{output_prefix}_well_metrics.csv"
    gain_path = output_dir / f"{output_prefix}_filter_gain_vs_raw.csv"
    input_summary_path = output_dir / f"{output_prefix}_well_input_summary.csv"
    summary_path = output_dir / f"{output_prefix}_summary.json"

    row_context.to_csv(row_context_path, index=False)
    filter_metrics.to_csv(filter_metrics_path, index=False)
    bucket_metrics.to_csv(bucket_metrics_path, index=False)
    well_metrics.to_csv(well_metrics_path, index=False)
    gain_vs_raw.to_csv(gain_path, index=False)
    input_summary.to_csv(input_summary_path, index=False)

    best_all = (
        filter_metrics[filter_metrics["eval_region"].eq("all")]
        .sort_values("rmse_tvt")
        .head(1)
        .to_dict("records")
    )
    raw_all = filter_metrics[
        filter_metrics["eval_region"].eq("all") & filter_metrics["filter"].eq("raw")
    ].to_dict("records")
    summary: dict[str, Any] = {
        "experiment": get_nested(
            config, "experiment.name", "exp167_fft_denoised_gr_matching_audit"
        ),
        "status": "completed_train_side_audit",
        "route": get_nested(config, "experiment.route", "pf_beam"),
        "train_dir": str(train_dir),
        "wells_requested": int(len(wells)),
        "wells_with_rows": int(row_context["well"].nunique()),
        "skipped_wells": skipped,
        "rows": int(len(row_context)),
        "filters": sorted(row_context["filter"].unique().tolist()),
        "eval_regions": sorted(row_context["eval_region"].unique().tolist()),
        "raw_all": raw_all[0] if raw_all else None,
        "best_all": best_all[0] if best_all else None,
        "artifacts": {
            "row_context": str(row_context_path),
            "filter_metrics": str(filter_metrics_path),
            "bucket_metrics": str(bucket_metrics_path),
            "well_metrics": str(well_metrics_path),
            "filter_gain_vs_raw": str(gain_path),
            "well_input_summary": str(input_summary_path),
            "summary": str(summary_path),
        },
        "artifact_sha256": {
            "row_context_gzip": sha256_file(row_context_path),
            "row_context_decompressed": sha256_decompressed(row_context_path),
            "filter_metrics": sha256_file(filter_metrics_path),
            "bucket_metrics": sha256_file(bucket_metrics_path),
            "well_metrics": sha256_file(well_metrics_path),
            "filter_gain_vs_raw": sha256_file(gain_path),
            "well_input_summary": sha256_file(input_summary_path),
        },
        "runtime_sec": float(time.time() - start),
        "audit_config": audit_config,
    }
    write_json(summary_path, summary)
    if metrics_path is not None:
        write_json(Path(metrics_path), summary)
    return {
        "summary": summary,
        "row_context": row_context,
        "filter_metrics": filter_metrics,
        "bucket_metrics": bucket_metrics,
        "well_metrics": well_metrics,
        "gain_vs_raw": gain_vs_raw,
        "input_summary": input_summary,
    }

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResidualProfile:
    well: str
    md: np.ndarray
    clean_gr: np.ndarray
    residual: np.ndarray
    missing_mask: np.ndarray
    gain: float
    bias: float
    fit_rmse: float
    fit_mae: float
    fit_points: int
    fit_scope: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        length = len(self.md)
        if any(len(values) != length for values in (self.clean_gr, self.residual, self.missing_mask)):
            raise ValueError(f"profile arrays have inconsistent length for {self.well}")
        if self.missing_mask.dtype != np.bool_:
            raise ValueError("missing_mask must be boolean")


@dataclass(frozen=True)
class ResidualView:
    well: str
    variant: str
    raw_gr: np.ndarray
    imputed_gr: np.ndarray
    missing_mask: np.ndarray
    transplanted_residual: np.ndarray
    inventory: tuple[Mapping[str, Any], ...]


def stable_uint64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def content_sha256(*arrays: np.ndarray, metadata: Mapping[str, Any] | None = None) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    if metadata is not None:
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=_json_default).encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def robust_affine_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    trim_quantile: float = 0.90,
    iterations: int = 3,
    min_points: int = 40,
    slope_bounds: tuple[float, float] = (0.25, 4.0),
) -> tuple[float, float, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < int(min_points):
        return 1.0, 0.0, finite
    mask = finite.copy()
    for _ in range(max(int(iterations), 1)):
        slope, bias = _least_squares_affine(x[mask], y[mask])
        residual = np.abs(y - (slope * x + bias))
        values = residual[finite & np.isfinite(residual)]
        if len(values) < int(min_points):
            break
        cutoff = float(np.quantile(values, float(trim_quantile)))
        updated = finite & (residual <= cutoff)
        if int(updated.sum()) < int(min_points) or np.array_equal(updated, mask):
            break
        mask = updated
    slope, bias = _least_squares_affine(x[mask], y[mask])
    if not np.isfinite(slope) or not np.isfinite(bias) or not slope_bounds[0] <= slope <= slope_bounds[1]:
        return 1.0, 0.0, finite
    return float(slope), float(bias), mask


def _least_squares_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return 1.0, 0.0
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum(np.square(x - x_mean)))
    if denom <= 1e-12:
        return 1.0, 0.0
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    return slope, float(y_mean - slope * x_mean)


def read_residual_profile(
    well: str,
    horizontal_path: str | Path,
    typewell_path: str | Path,
    *,
    fit_scope: str = "full_true_tvt",
    trim_quantile: float = 0.90,
    iterations: int = 3,
    min_points: int = 40,
    slope_bounds: tuple[float, float] = (0.25, 4.0),
) -> ResidualProfile:
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    required_horizontal = {"MD", "GR", "TVT", "TVT_input"}
    required_typewell = {"TVT", "GR"}
    missing_horizontal = sorted(required_horizontal - set(horizontal.columns))
    missing_typewell = sorted(required_typewell - set(typewell.columns))
    if missing_horizontal or missing_typewell:
        raise ValueError(
            f"missing residual source columns for {well}: "
            f"horizontal={missing_horizontal}, typewell={missing_typewell}"
        )

    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    observed = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    true_tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
    known_tvt = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(md).all():
        raise ValueError(f"non-finite MD in residual source {well}")

    type_frame = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna()
    type_frame = type_frame.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    if len(type_frame) < 4:
        raise ValueError(f"typewell is too short for residual extraction: {well}")
    type_tvt = type_frame["TVT"].to_numpy(np.float64)
    type_gr = type_frame["GR"].to_numpy(np.float64)

    if fit_scope == "known_prefix":
        fit_tvt = known_tvt
    elif fit_scope == "full_true_tvt":
        fit_tvt = true_tvt
    else:
        raise ValueError(f"unsupported fit_scope: {fit_scope}")
    fit_type_gr = _interp_with_nan(type_tvt, type_gr, fit_tvt)
    gain, bias, fit_mask = robust_affine_fit(
        fit_type_gr,
        observed,
        trim_quantile=trim_quantile,
        iterations=iterations,
        min_points=min_points,
        slope_bounds=slope_bounds,
    )

    full_type_gr = _interp_with_nan(type_tvt, type_gr, true_tvt)
    clean = gain * full_type_gr + bias
    residual = observed - clean
    missing_mask = ~np.isfinite(observed)
    residual[missing_mask | ~np.isfinite(clean)] = np.nan
    fitted_error = observed - (gain * fit_type_gr + bias)
    used = fit_mask & np.isfinite(fitted_error)
    fit_rmse = float(np.sqrt(np.mean(np.square(fitted_error[used])))) if used.any() else np.inf
    fit_mae = float(np.mean(np.abs(fitted_error[used]))) if used.any() else np.inf
    stats = residual_statistics(residual, missing_mask, md)
    metadata = {
        **stats,
        "horizontal_rows": int(len(horizontal)),
        "typewell_rows": int(len(type_frame)),
        "profile_sha256": content_sha256(
            md.astype(np.float32),
            np.nan_to_num(clean, nan=np.float32(-9999.0)).astype(np.float32),
            np.nan_to_num(residual, nan=np.float32(-9999.0)).astype(np.float32),
            missing_mask,
            metadata={"well": str(well), "fit_scope": fit_scope, "gain": gain, "bias": bias},
        ),
    }
    return ResidualProfile(
        well=str(well),
        md=md.astype(np.float32),
        clean_gr=clean.astype(np.float32),
        residual=residual.astype(np.float32),
        missing_mask=missing_mask.astype(bool),
        gain=float(gain),
        bias=float(bias),
        fit_rmse=fit_rmse,
        fit_mae=fit_mae,
        fit_points=int(used.sum()),
        fit_scope=str(fit_scope),
        metadata=metadata,
    )


def _interp_with_nan(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    result = np.full(len(query), np.nan, dtype=np.float64)
    finite = np.isfinite(query)
    if finite.any():
        result[finite] = np.interp(query[finite], x, y, left=np.nan, right=np.nan)
    return result


def residual_statistics(
    residual: np.ndarray,
    missing_mask: np.ndarray,
    md: np.ndarray,
) -> dict[str, float | int]:
    residual = np.asarray(residual, dtype=np.float64)
    missing_mask = np.asarray(missing_mask, dtype=bool)
    filled = interpolate_missing(residual, fallback=0.0)
    center = float(np.median(filled)) if len(filled) else 0.0
    mad = float(np.median(np.abs(filled - center))) if len(filled) else 0.0
    robust_scale = max(1.4826 * mad, float(np.std(filled)), 1e-6)
    smooth = (
        pd.Series(filled)
        .rolling(min(65, max(len(filled), 1)), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float64)
    )
    detail = filled - smooth
    pair_count = len(filled) // 2
    haar_detail = (
        (filled[: 2 * pair_count : 2] - filled[1 : 2 * pair_count : 2]) / np.sqrt(2.0)
        if pair_count
        else np.asarray([], dtype=np.float64)
    )
    frequency = _fft_summary(filled, np.asarray(md, dtype=np.float64))
    unique = np.unique(np.round(filled[np.isfinite(filled)], decimals=6))
    steps = np.diff(unique)
    positive_steps = steps[steps > 1e-6]
    return {
        "residual_mean": float(np.mean(filled)) if len(filled) else 0.0,
        "residual_std": float(np.std(filled)) if len(filled) else 0.0,
        "residual_robust_scale": robust_scale,
        "detail_energy": float(np.mean(np.square(detail))) if len(detail) else 0.0,
        "haar_dwt_detail_energy": (
            float(np.mean(np.square(haar_detail))) if len(haar_detail) else 0.0
        ),
        "spike_rate": float(np.mean(np.abs(filled - center) > 6.0 * robust_scale)),
        "quantization_step_median": float(np.median(positive_steps)) if len(positive_steps) else 0.0,
        "unique_ratio": float(len(unique) / max(len(filled), 1)),
        "missing_rate": float(np.mean(missing_mask)) if len(missing_mask) else 0.0,
        "missing_run_max": int(_max_true_run(missing_mask)),
        **frequency,
    }


def _fft_summary(values: np.ndarray, md: np.ndarray) -> dict[str, float]:
    if len(values) < 8:
        return {
            "fft_dominant_frequency_norm": 0.0,
            "fft_dominant_energy_ratio": 0.0,
            "fft_rotation_energy_ratio": 0.0,
            "fft_high_frequency_ratio": 0.0,
        }
    x = np.arange(len(values), dtype=np.float64)
    slope, bias = _least_squares_affine(x, values)
    centered = values - (slope * x + bias)
    spacing = float(np.median(np.diff(md))) if len(md) > 1 else 1.0
    if not np.isfinite(spacing) or abs(spacing) <= 1e-9:
        spacing = 1.0
    power = np.square(np.abs(np.fft.rfft(centered)))[1:]
    freqs = np.fft.rfftfreq(len(centered), d=abs(spacing))[1:]
    total = max(float(np.sum(power)), 1e-12)
    nyquist = max(float(np.max(freqs)), 1e-12)
    normalized = freqs / nyquist
    dominant = int(np.argmax(power)) if len(power) else 0
    return {
        "fft_dominant_frequency_norm": float(normalized[dominant]) if len(power) else 0.0,
        "fft_dominant_energy_ratio": float(power[dominant] / total) if len(power) else 0.0,
        "fft_rotation_energy_ratio": float(np.sum(power[(normalized >= 0.06) & (normalized <= 0.35)]) / total),
        "fft_high_frequency_ratio": float(np.sum(power[normalized >= 0.35]) / total),
    }


def _max_true_run(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    padded = np.pad(mask.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return int(np.max(stops - starts))


def interpolate_missing(values: np.ndarray, *, fallback: float = 0.0) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    finite = series[np.isfinite(series)]
    fill = float(finite.median()) if len(finite) else float(fallback)
    return (
        series.interpolate(limit_direction="both").ffill().bfill().fillna(fill).to_numpy(np.float32)
    )


def impute_and_smooth_gr(raw_gr: np.ndarray, *, rolling_window: int = 5) -> np.ndarray:
    filled = interpolate_missing(raw_gr, fallback=0.0)
    return (
        pd.Series(filled)
        .rolling(int(rolling_window), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )


def synthesize_residual_view(
    recipient: ResidualProfile,
    donor_profiles: Sequence[ResidualProfile],
    *,
    variant: str,
    seed_parts: Sequence[object],
    block_lengths: Sequence[int] = (64, 128, 256),
    rolling_window: int = 5,
    residual_clip_abs: float | None = 200.0,
) -> ResidualView:
    allowed_variants = {
        "clean_duplicate",
        "real_residual_block",
        "white_noise",
        "shuffled_residual",
    }
    if variant not in allowed_variants:
        raise ValueError(f"unsupported residual augmentation variant: {variant}")
    donors = sorted(
        [profile for profile in donor_profiles if profile.well != recipient.well],
        key=lambda profile: profile.well,
    )
    if not donors:
        donors = sorted(list(donor_profiles), key=lambda profile: profile.well)
    if variant != "clean_duplicate" and not donors:
        raise ValueError("residual augmentation requires at least one donor profile")
    lengths = sorted({max(int(length), 1) for length in block_lengths})
    rng = np.random.default_rng(stable_uint64(*seed_parts, recipient.well, variant))
    n_rows = len(recipient.md)
    transplanted = np.zeros(n_rows, dtype=np.float32)
    missing = np.zeros(n_rows, dtype=bool)
    inventory: list[dict[str, Any]] = []
    cursor = 0
    block_index = 0
    while cursor < n_rows:
        remaining = n_rows - cursor
        requested = int(lengths[int(rng.integers(0, len(lengths)))])
        length = min(requested, remaining)
        if variant == "clean_duplicate":
            donor = recipient
            start = cursor
            raw_values = recipient.residual[cursor : cursor + length]
            values = interpolate_missing(raw_values, fallback=0.0)
            block_missing = recipient.missing_mask[cursor : cursor + length].copy()
        else:
            donor = donors[int(rng.integers(0, len(donors)))]
            length = min(length, len(donor.residual))
            max_start = max(len(donor.residual) - length, 0)
            start = int(rng.integers(0, max_start + 1))
            raw_values = donor.residual[start : start + length]
            block_missing = donor.missing_mask[start : start + length].copy()
            values = interpolate_missing(raw_values, fallback=0.0)
            if variant == "white_noise":
                finite = raw_values[np.isfinite(raw_values)]
                scale = float(np.std(finite)) if len(finite) else float(
                    donor.metadata.get("residual_robust_scale", 1.0)
                )
                values = rng.normal(0.0, max(scale, 1e-6), size=length).astype(np.float32)
            elif variant == "shuffled_residual":
                order = rng.permutation(length)
                values = values[order]
                block_missing = block_missing[order]
        if residual_clip_abs is not None:
            values = np.clip(values, -float(residual_clip_abs), float(residual_clip_abs))
        stop = cursor + length
        transplanted[cursor:stop] = values
        missing[cursor:stop] = block_missing
        inventory.append(
            {
                "recipient_well": recipient.well,
                "variant": variant,
                "block_index": block_index,
                "recipient_start": cursor,
                "recipient_stop": stop,
                "donor_well": donor.well,
                "donor_start": start,
                "donor_stop": start + length,
                "block_length": length,
                "missing_rows": int(block_missing.sum()),
                "residual_mean": float(np.mean(values)),
                "residual_std": float(np.std(values)),
            }
        )
        cursor = stop
        block_index += 1
    clean = interpolate_missing(recipient.clean_gr, fallback=0.0)
    raw = clean + transplanted
    raw = raw.astype(np.float32)
    raw[missing] = np.nan
    imputed = impute_and_smooth_gr(raw, rolling_window=rolling_window)
    return ResidualView(
        well=recipient.well,
        variant=variant,
        raw_gr=raw,
        imputed_gr=imputed,
        missing_mask=missing,
        transplanted_residual=transplanted,
        inventory=tuple(inventory),
    )


def profile_inventory(profile: ResidualProfile) -> dict[str, Any]:
    return {
        "well": profile.well,
        "rows": int(len(profile.md)),
        "gain": float(profile.gain),
        "bias": float(profile.bias),
        "fit_rmse": float(profile.fit_rmse),
        "fit_mae": float(profile.fit_mae),
        "fit_points": int(profile.fit_points),
        "fit_scope": profile.fit_scope,
        **dict(profile.metadata),
    }

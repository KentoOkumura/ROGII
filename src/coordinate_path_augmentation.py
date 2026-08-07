from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HORIZONTAL_REQUIRED = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TYPEWELL_REQUIRED = ("TVT", "GR")
STRICT_TRANSFORMS = (
    "heel_center_translation",
    "lateral_reflection",
    "yaw_rotation",
    "tvt_datum_shift",
)
APPROXIMATE_TRANSFORMS = (
    "md_stretch",
    "tvt_shear",
    "xy_plane_tilt",
    "low_frequency_spline_warp",
    "smooth_xyz_control_perturbation",
)
ALL_TRANSFORMS = (*STRICT_TRANSFORMS, *APPROXIMATE_TRANSFORMS)

ENVELOPE_METRICS = (
    "md_step_q01",
    "md_step_q99",
    "xy_slope_q95",
    "z_slope_abs_q95",
    "xy_curvature_q95",
    "z_curvature_abs_q95",
    "tvt_slope_abs_q95",
)


@dataclass(frozen=True)
class TransformSpec:
    kind: str
    parameters: Mapping[str, float]
    exact: bool
    view_slot: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ALL_TRANSFORMS:
            raise ValueError(f"unknown transform kind: {self.kind}")
        if self.exact != (self.kind in STRICT_TRANSFORMS):
            raise ValueError(f"exact flag does not match transform kind: {self.kind}")


@dataclass(frozen=True)
class TransformResult:
    horizontal: pd.DataFrame
    typewell: pd.DataFrame
    metadata: Mapping[str, Any]


def stable_uint64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def stable_choice(values: Sequence[Any], *key_parts: object) -> Any:
    if not values:
        raise ValueError("stable_choice requires at least one value")
    return values[stable_uint64(*key_parts) % len(values)]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def read_horizontal(path: str | Path, *, require_target: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(HORIZONTAL_REQUIRED)
    if require_target:
        required.add("TVT")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"horizontal file is missing columns {missing}: {path}")
    numeric = sorted(required | ({"TVT"} if "TVT" in frame.columns else set()))
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["MD", "X", "Y", "Z"]].isna().any().any():
        raise ValueError(f"horizontal MD/X/Y/Z contains missing values: {path}")
    return frame


def read_typewell(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(TYPEWELL_REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"typewell file is missing columns {missing}: {path}")
    for column in TYPEWELL_REQUIRED:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["TVT"].notna().sum() < 2 or frame["GR"].notna().sum() < 2:
        raise ValueError(f"typewell has insufficient finite TVT/GR values: {path}")
    return frame


def official_start_index(horizontal: pd.DataFrame) -> int:
    known = np.flatnonzero(pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna())
    if len(known) == 0:
        raise ValueError("horizontal well has no finite TVT_input prefix")
    if not np.array_equal(known, np.arange(known[-1] + 1)):
        raise ValueError("TVT_input known rows are not a contiguous prefix")
    return int(known[-1])


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def _gradient(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if len(values) < 2 or np.unique(coordinates).size < 2:
        return np.zeros(len(values), dtype=np.float64)
    result = np.gradient(values, coordinates, edge_order=1)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _tail_progress(horizontal: pd.DataFrame, anchor_index: int) -> np.ndarray:
    md = _numeric(horizontal, "MD")
    anchor_md = float(md[anchor_index])
    tail_span = max(float(md[-1] - anchor_md), 1e-9)
    progress = np.clip((md - anchor_md) / tail_span, 0.0, 1.0)
    progress[: anchor_index + 1] = 0.0
    return progress


def _heading(horizontal: pd.DataFrame, anchor_index: int) -> tuple[float, float]:
    x = _numeric(horizontal, "X")
    y = _numeric(horizontal, "Y")
    dx = float(x[anchor_index] - x[0])
    dy = float(y[anchor_index] - y[0])
    norm = float(np.hypot(dx, dy))
    if norm <= 1e-9:
        dx = float(x[-1] - x[0])
        dy = float(y[-1] - y[0])
        norm = float(np.hypot(dx, dy))
    if norm <= 1e-9:
        return 1.0, 0.0
    return dx / norm, dy / norm


def _smoothstep_control_curve(
    progress: np.ndarray,
    control_positions: Sequence[float],
    control_values: Sequence[float],
) -> np.ndarray:
    positions = np.asarray(control_positions, dtype=np.float64)
    values = np.asarray(control_values, dtype=np.float64)
    if len(positions) != len(values) or len(positions) < 2:
        raise ValueError("control positions and values must have equal length >= 2")
    if not np.all(np.diff(positions) > 0) or positions[0] != 0.0 or positions[-1] != 1.0:
        raise ValueError("control positions must be strictly increasing from 0 to 1")
    u = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    segment = np.searchsorted(positions, u, side="right") - 1
    segment = np.clip(segment, 0, len(positions) - 2)
    left = positions[segment]
    right = positions[segment + 1]
    local = np.divide(u - left, right - left, out=np.zeros_like(u), where=right > left)
    weight = local * local * (3.0 - 2.0 * local)
    return values[segment] * (1.0 - weight) + values[segment + 1] * weight


def _candidate_columns(horizontal: pd.DataFrame, candidate_columns: Iterable[str]) -> list[str]:
    return [column for column in candidate_columns if column in horizontal.columns]


def choose_transform_spec(
    kind: str,
    *,
    seed: int,
    well: str,
    view_slot: int,
    parameter_grid: Mapping[str, Sequence[Any]],
) -> TransformSpec:
    parameters: dict[str, float] = {}
    for name in sorted(parameter_grid):
        value = stable_choice(list(parameter_grid[name]), seed, well, kind, view_slot, name)
        parameters[name] = float(value)
    return TransformSpec(
        kind=kind,
        parameters=parameters,
        exact=kind in STRICT_TRANSFORMS,
        view_slot=int(view_slot),
    )


def apply_transform(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    spec: TransformSpec,
    *,
    candidate_columns: Iterable[str] = (),
) -> TransformResult:
    source = horizontal.copy(deep=True)
    transformed = horizontal.copy(deep=True)
    transformed_typewell = typewell.copy(deep=True)
    anchor_index = official_start_index(source)
    x0 = float(_numeric(source, "X")[0])
    y0 = float(_numeric(source, "Y")[0])
    anchor_md = float(_numeric(source, "MD")[anchor_index])
    anchor_x = float(_numeric(source, "X")[anchor_index])
    anchor_y = float(_numeric(source, "Y")[anchor_index])
    progress = _tail_progress(source, anchor_index)
    tail_mask = progress > 0.0
    candidates = _candidate_columns(source, candidate_columns)
    metadata: dict[str, Any] = {
        "kind": spec.kind,
        "exact": bool(spec.exact),
        "parameters": dict(spec.parameters),
        "view_slot": int(spec.view_slot),
        "anchor_index": int(anchor_index),
        "anchor_md": anchor_md,
        "anchor_x": anchor_x,
        "anchor_y": anchor_y,
        "heel_x": x0,
        "heel_y": y0,
        "candidate_columns": candidates,
    }

    if spec.kind == "heel_center_translation":
        transformed["X"] = _numeric(source, "X") - x0
        transformed["Y"] = _numeric(source, "Y") - y0
    elif spec.kind == "lateral_reflection":
        ux, uy = _heading(source, anchor_index)
        dx = _numeric(source, "X") - x0
        dy = _numeric(source, "Y") - y0
        along = dx * ux + dy * uy
        cross = -dx * uy + dy * ux
        transformed["X"] = x0 + along * ux + cross * uy
        transformed["Y"] = y0 + along * uy - cross * ux
        metadata.update({"heading_x": ux, "heading_y": uy})
    elif spec.kind == "yaw_rotation":
        angle_degrees = float(spec.parameters.get("angle_degrees", 90.0))
        angle = np.deg2rad(angle_degrees)
        cos_value, sin_value = float(np.cos(angle)), float(np.sin(angle))
        dx = _numeric(source, "X") - x0
        dy = _numeric(source, "Y") - y0
        transformed["X"] = x0 + cos_value * dx - sin_value * dy
        transformed["Y"] = y0 + sin_value * dx + cos_value * dy
        metadata["angle_degrees"] = angle_degrees
    elif spec.kind == "tvt_datum_shift":
        shift = float(spec.parameters.get("shift_ft", 0.0))
        for column in ("TVT", "TVT_input", *candidates):
            if column in transformed.columns:
                transformed[column] = pd.to_numeric(transformed[column], errors="coerce") + shift
        transformed_typewell["TVT"] = (
            pd.to_numeric(transformed_typewell["TVT"], errors="coerce") + shift
        )
        metadata["tvt_delta"] = np.full(len(transformed), shift, dtype=np.float64)
    elif spec.kind == "md_stretch":
        factor = float(spec.parameters.get("factor", 1.0))
        if factor <= 0.0:
            raise ValueError("MD stretch factor must be positive")
        md = _numeric(source, "MD")
        md_new = md.copy()
        md_new[tail_mask] = anchor_md + factor * (md[tail_mask] - anchor_md)
        transformed["MD"] = md_new
        metadata["md_factor"] = factor
    elif spec.kind == "tvt_shear":
        tail_delta = float(spec.parameters.get("tail_delta_ft", 0.0))
        tvt_delta = tail_delta * progress
        _apply_tvt_delta(transformed, tvt_delta, candidates)
        metadata["tvt_delta"] = tvt_delta
    elif spec.kind == "xy_plane_tilt":
        slope_x = float(spec.parameters.get("slope_x", 0.0))
        slope_y = float(spec.parameters.get("slope_y", 0.0))
        tvt_delta = slope_x * (_numeric(source, "X") - anchor_x) + slope_y * (
            _numeric(source, "Y") - anchor_y
        )
        tvt_delta[~tail_mask] = 0.0
        _apply_tvt_delta(transformed, tvt_delta, candidates)
        metadata["tvt_delta"] = tvt_delta
    elif spec.kind == "low_frequency_spline_warp":
        amplitude = float(spec.parameters.get("amplitude_ft", 0.0))
        middle_sign = float(spec.parameters.get("middle_sign", 1.0))
        tvt_delta = _smoothstep_control_curve(
            progress,
            (0.0, 0.33, 0.66, 1.0),
            (0.0, 0.35 * amplitude * middle_sign, 0.70 * amplitude, amplitude),
        )
        tvt_delta[~tail_mask] = 0.0
        _apply_tvt_delta(transformed, tvt_delta, candidates)
        metadata["tvt_delta"] = tvt_delta
    elif spec.kind == "smooth_xyz_control_perturbation":
        amplitude = float(spec.parameters.get("amplitude_ft", 0.0))
        sign_x = float(spec.parameters.get("sign_x", 1.0))
        sign_y = float(spec.parameters.get("sign_y", -1.0))
        sign_z = float(spec.parameters.get("sign_z", 1.0))
        controls = (0.0, 0.33, 0.66, 1.0)
        dx = _smoothstep_control_curve(
            progress,
            controls,
            (0.0, 0.45 * amplitude * sign_x, 0.80 * amplitude * sign_x, amplitude * sign_x),
        )
        dy = _smoothstep_control_curve(
            progress,
            controls,
            (0.0, -0.30 * amplitude * sign_y, 0.55 * amplitude * sign_y, amplitude * sign_y),
        )
        dz = _smoothstep_control_curve(
            progress,
            controls,
            (0.0, 0.10 * amplitude * sign_z, 0.25 * amplitude * sign_z, 0.35 * amplitude * sign_z),
        )
        dx[~tail_mask] = 0.0
        dy[~tail_mask] = 0.0
        dz[~tail_mask] = 0.0
        transformed["X"] = _numeric(source, "X") + dx
        transformed["Y"] = _numeric(source, "Y") + dy
        transformed["Z"] = _numeric(source, "Z") + dz
        metadata.update({"x_delta": dx, "y_delta": dy, "z_delta": dz})
    else:  # pragma: no cover - TransformSpec validates kinds.
        raise AssertionError(spec.kind)

    if spec.kind in APPROXIMATE_TRANSFORMS:
        if "TVT" not in transformed.columns:
            raise ValueError(
                "approximate train augmentation requires finite true TVT for GR resampling"
            )
        resampled, coverage = resample_typewell_gr(
            transformed_typewell, _numeric(transformed, "TVT")
        )
        gr = _numeric(source, "GR").copy()
        gr[tail_mask] = resampled[tail_mask]
        transformed["GR"] = gr
        metadata["typewell_coverage_fraction"] = (
            float(np.mean(coverage[tail_mask])) if tail_mask.any() else 1.0
        )
        metadata["resampled_tail_rows"] = int(tail_mask.sum())
    else:
        metadata["typewell_coverage_fraction"] = 1.0
        metadata["resampled_tail_rows"] = 0

    _validate_transformed_frame(transformed, anchor_index)
    metadata["anchor_max_abs_delta"] = anchor_max_abs_delta(source, transformed, anchor_index)
    return TransformResult(transformed, transformed_typewell, metadata)


def _apply_tvt_delta(
    frame: pd.DataFrame,
    tvt_delta: np.ndarray,
    candidate_columns: Sequence[str],
) -> None:
    for column in ("TVT", *candidate_columns):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce") + tvt_delta


def _validate_transformed_frame(frame: pd.DataFrame, anchor_index: int) -> None:
    required_finite = frame[["MD", "X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(required_finite.to_numpy(np.float64)).all():
        raise ValueError("transformed MD/X/Y/Z contains non-finite values")
    md = required_finite["MD"].to_numpy(np.float64)
    if not np.all(np.diff(md) > 0.0):
        raise ValueError("transformed MD is not strictly increasing")
    if anchor_index < 0 or anchor_index >= len(frame):
        raise ValueError("invalid transformed anchor index")


def inverse_exact_transform(
    result: TransformResult,
    spec: TransformSpec,
) -> TransformResult:
    if not spec.exact:
        raise ValueError(f"approximate transform has no strict inverse: {spec.kind}")
    horizontal = result.horizontal.copy(deep=True)
    typewell = result.typewell.copy(deep=True)
    metadata = dict(result.metadata)
    x0 = float(metadata["heel_x"])
    y0 = float(metadata["heel_y"])
    candidates = list(metadata.get("candidate_columns") or [])

    if spec.kind == "heel_center_translation":
        horizontal["X"] = _numeric(horizontal, "X") + x0
        horizontal["Y"] = _numeric(horizontal, "Y") + y0
    elif spec.kind == "lateral_reflection":
        ux = float(metadata["heading_x"])
        uy = float(metadata["heading_y"])
        dx = _numeric(horizontal, "X") - x0
        dy = _numeric(horizontal, "Y") - y0
        along = dx * ux + dy * uy
        cross = -dx * uy + dy * ux
        horizontal["X"] = x0 + along * ux + cross * uy
        horizontal["Y"] = y0 + along * uy - cross * ux
    elif spec.kind == "yaw_rotation":
        angle = -np.deg2rad(float(metadata["angle_degrees"]))
        cos_value, sin_value = float(np.cos(angle)), float(np.sin(angle))
        dx = _numeric(horizontal, "X") - x0
        dy = _numeric(horizontal, "Y") - y0
        horizontal["X"] = x0 + cos_value * dx - sin_value * dy
        horizontal["Y"] = y0 + sin_value * dx + cos_value * dy
    elif spec.kind == "tvt_datum_shift":
        shift = float(spec.parameters.get("shift_ft", 0.0))
        for column in ("TVT", "TVT_input", *candidates):
            if column in horizontal.columns:
                horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce") - shift
        typewell["TVT"] = pd.to_numeric(typewell["TVT"], errors="coerce") - shift
    else:  # pragma: no cover
        raise AssertionError(spec.kind)
    return TransformResult(horizontal, typewell, {**metadata, "inverted": True})


def exact_inverse_error(
    original_horizontal: pd.DataFrame,
    original_typewell: pd.DataFrame,
    inverted: TransformResult,
    *,
    candidate_columns: Iterable[str] = (),
) -> dict[str, float]:
    horizontal_columns = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    if "TVT" in original_horizontal.columns:
        horizontal_columns.append("TVT")
    horizontal_columns.extend(_candidate_columns(original_horizontal, candidate_columns))
    errors: dict[str, float] = {}
    for column in horizontal_columns:
        left = _numeric(original_horizontal, column)
        right = _numeric(inverted.horizontal, column)
        finite = np.isfinite(left) & np.isfinite(right)
        if not np.array_equal(np.isfinite(left), np.isfinite(right)):
            errors[column] = float("inf")
        elif finite.any():
            errors[column] = float(np.max(np.abs(left[finite] - right[finite])))
        else:
            errors[column] = 0.0
    for column in TYPEWELL_REQUIRED:
        left = _numeric(original_typewell, column)
        right = _numeric(inverted.typewell, column)
        finite = np.isfinite(left) & np.isfinite(right)
        key = f"typewell_{column}"
        if not np.array_equal(np.isfinite(left), np.isfinite(right)):
            errors[key] = float("inf")
        elif finite.any():
            errors[key] = float(np.max(np.abs(left[finite] - right[finite])))
        else:
            errors[key] = 0.0
    errors["max_abs"] = float(max(errors.values(), default=0.0))
    return errors


def anchor_max_abs_delta(
    original: pd.DataFrame,
    transformed: pd.DataFrame,
    anchor_index: int,
) -> float:
    columns = ["MD", "X", "Y", "Z", "TVT_input"]
    values: list[float] = []
    for column in columns:
        left = float(pd.to_numeric(original[column], errors="coerce").iloc[anchor_index])
        right = float(pd.to_numeric(transformed[column], errors="coerce").iloc[anchor_index])
        if np.isfinite(left) and np.isfinite(right):
            values.append(abs(left - right))
    return float(max(values, default=0.0))


def resample_typewell_gr(
    typewell: pd.DataFrame,
    tvt_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tvt = _numeric(typewell, "TVT")
    gr = _numeric(typewell, "GR")
    finite = np.isfinite(tvt) & np.isfinite(gr)
    if finite.sum() < 2:
        raise ValueError("typewell has fewer than two finite TVT/GR pairs")
    pairs = (
        pd.DataFrame({"TVT": tvt[finite], "GR": gr[finite]})
        .groupby("TVT", as_index=False)["GR"]
        .mean()
    )
    xp = pairs["TVT"].to_numpy(np.float64)
    fp = pairs["GR"].to_numpy(np.float64)
    values = np.asarray(tvt_values, dtype=np.float64)
    coverage = np.isfinite(values) & (values >= xp[0]) & (values <= xp[-1])
    result = np.full(len(values), np.nan, dtype=np.float64)
    result[coverage] = np.interp(values[coverage], xp, fp)
    return result, coverage


def regenerate_local_features(horizontal: pd.DataFrame) -> pd.DataFrame:
    anchor_index = official_start_index(horizontal)
    md = _numeric(horizontal, "MD")
    x = _numeric(horizontal, "X")
    y = _numeric(horizontal, "Y")
    z = _numeric(horizontal, "Z")
    dx = _gradient(x, md)
    dy = _gradient(y, md)
    dz = _gradient(z, md)
    d2x = _gradient(dx, md)
    d2y = _gradient(dy, md)
    d2z = _gradient(dz, md)
    ux, uy = _heading(horizontal, anchor_index)
    anchor_x, anchor_y = x[anchor_index], y[anchor_index]
    rel_x, rel_y = x - anchor_x, y - anchor_y
    result = pd.DataFrame(
        {
            "dX_dMD": dx,
            "dY_dMD": dy,
            "dZ_dMD": dz,
            "d2X_dMD2": d2x,
            "d2Y_dMD2": d2y,
            "d2Z_dMD2": d2z,
            "xy_slope": np.hypot(dx, dy),
            "xyz_slope": np.sqrt(dx * dx + dy * dy + dz * dz),
            "xy_curvature": np.hypot(d2x, d2y),
            "z_curvature_abs": np.abs(d2z),
            "along_track": rel_x * ux + rel_y * uy,
            "cross_track": -rel_x * uy + rel_y * ux,
            "tail_progress": _tail_progress(horizontal, anchor_index),
        }
    )
    return result.astype(np.float32)


def _finite_quantile(values: np.ndarray, quantile: float, default: float = 0.0) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if len(finite) else float(default)


def _spectral_summary(values: np.ndarray) -> dict[str, float]:
    series = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(series)
    if finite.sum() < 8:
        return {
            "gr_fft_low_fraction": 0.0,
            "gr_fft_mid_fraction": 0.0,
            "gr_fft_high_fraction": 0.0,
            "gr_haar_level1_energy": 0.0,
            "gr_haar_level2_energy": 0.0,
            "gr_haar_level3_energy": 0.0,
        }
    index = np.arange(len(series), dtype=np.float64)
    filled = np.interp(index, index[finite], series[finite])
    centered = filled - float(np.mean(filled))
    power = np.abs(np.fft.rfft(centered)) ** 2
    power[0] = 0.0
    total = max(float(power.sum()), 1e-12)
    n = len(power)
    low_end = max(2, int(np.ceil(n * 0.10)))
    mid_end = max(low_end + 1, int(np.ceil(n * 0.35)))
    current = centered.copy()
    haar: list[float] = []
    for _ in range(3):
        if len(current) < 2:
            haar.append(0.0)
            continue
        usable = current[: len(current) - (len(current) % 2)]
        detail = (usable[0::2] - usable[1::2]) / np.sqrt(2.0)
        approx = (usable[0::2] + usable[1::2]) / np.sqrt(2.0)
        haar.append(float(np.mean(detail * detail)) if len(detail) else 0.0)
        current = approx
    return {
        "gr_fft_low_fraction": float(power[1:low_end].sum() / total),
        "gr_fft_mid_fraction": float(power[low_end:mid_end].sum() / total),
        "gr_fft_high_fraction": float(power[mid_end:].sum() / total),
        "gr_haar_level1_energy": haar[0],
        "gr_haar_level2_energy": haar[1],
        "gr_haar_level3_energy": haar[2],
    }


def summarize_well(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    anchor_index = official_start_index(horizontal)
    tail = np.arange(len(horizontal)) > anchor_index
    if not tail.any():
        raise ValueError("well has no evaluation tail after the official start")
    md = _numeric(horizontal, "MD")
    local = regenerate_local_features(horizontal)
    summary: dict[str, Any] = {
        "rows": int(len(horizontal)),
        "known_rows": int(anchor_index + 1),
        "tail_rows": int(tail.sum()),
        "md_step_min": float(np.min(np.diff(md))),
        "md_step_q01": _finite_quantile(np.diff(md), 0.01),
        "md_step_q99": _finite_quantile(np.diff(md), 0.99),
        "xy_slope_q95": _finite_quantile(local.loc[tail, "xy_slope"].to_numpy(), 0.95),
        "z_slope_abs_q95": _finite_quantile(np.abs(local.loc[tail, "dZ_dMD"].to_numpy()), 0.95),
        "xy_curvature_q95": _finite_quantile(local.loc[tail, "xy_curvature"].to_numpy(), 0.95),
        "z_curvature_abs_q95": _finite_quantile(
            local.loc[tail, "z_curvature_abs"].to_numpy(), 0.95
        ),
        "gr_missing_fraction": float(
            pd.to_numeric(horizontal["GR"], errors="coerce").isna().mean()
        ),
        "prefix_gr_missing_fraction": float(
            pd.to_numeric(horizontal["GR"], errors="coerce").iloc[: anchor_index + 1].isna().mean()
        ),
        "prefix_md_span": float(md[anchor_index] - md[0]),
    }
    if "TVT" in horizontal.columns:
        tvt = _numeric(horizontal, "TVT")
        tvt_slope = _gradient(tvt, md)
        summary["tvt_slope_abs_q95"] = _finite_quantile(np.abs(tvt_slope[tail]), 0.95)
        _, coverage = resample_typewell_gr(typewell, tvt)
        summary["typewell_coverage_fraction"] = float(np.mean(coverage[tail]))
    else:
        summary["tvt_slope_abs_q95"] = float("nan")
        summary["typewell_coverage_fraction"] = (
            float(metadata.get("typewell_coverage_fraction", 0.0)) if metadata else 0.0
        )
    summary.update(_spectral_summary(_numeric(horizontal, "GR")[tail]))
    if metadata:
        summary["anchor_max_abs_delta"] = float(metadata.get("anchor_max_abs_delta", 0.0))
        summary["resampled_tail_rows"] = int(metadata.get("resampled_tail_rows", 0))
    return summary


def fit_distribution_envelope(
    real_summaries: pd.DataFrame,
    *,
    lower_quantile: float,
    upper_quantile: float,
    relative_margin: float,
    min_typewell_coverage: float,
) -> dict[str, Any]:
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("invalid distribution envelope quantiles")
    if relative_margin < 0.0:
        raise ValueError("relative_margin must be non-negative")
    metrics: dict[str, dict[str, float]] = {}
    for metric in ENVELOPE_METRICS:
        values = (
            pd.to_numeric(real_summaries[metric], errors="coerce").dropna().to_numpy(np.float64)
        )
        if len(values) == 0:
            raise ValueError(f"no finite real values for envelope metric: {metric}")
        lower = float(np.quantile(values, lower_quantile))
        upper = float(np.quantile(values, upper_quantile))
        span = max(upper - lower, abs(lower) * 0.05, abs(upper) * 0.05, 1e-9)
        metrics[metric] = {
            "lower": lower - relative_margin * span,
            "upper": upper + relative_margin * span,
            "real_min": float(np.min(values)),
            "real_max": float(np.max(values)),
        }
    envelope = {
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "relative_margin": float(relative_margin),
        "min_typewell_coverage": float(min_typewell_coverage),
        "metrics": metrics,
    }
    envelope["content_sha256"] = json_sha256(envelope)
    return envelope


def evaluate_distribution_guard(
    summary: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    exact: bool,
    inverse_max_abs: float | None,
    inverse_tolerance: float,
    anchor_tolerance: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(summary.get("md_step_min", 0.0)) <= 0.0:
        reasons.append("non_monotone_md")
    if float(summary.get("anchor_max_abs_delta", 0.0)) > anchor_tolerance and not exact:
        reasons.append("anchor_discontinuity")
    if not exact:
        if float(summary.get("typewell_coverage_fraction", 0.0)) < float(
            envelope["min_typewell_coverage"]
        ):
            reasons.append("typewell_coverage")
        for metric, bounds in envelope["metrics"].items():
            value = float(summary.get(metric, float("nan")))
            if not np.isfinite(value):
                reasons.append(f"nonfinite:{metric}")
            elif value < float(bounds["lower"]) or value > float(bounds["upper"]):
                reasons.append(f"out_of_envelope:{metric}")
    if exact and (inverse_max_abs is None or inverse_max_abs > inverse_tolerance):
        reasons.append("inverse_consistency")
    return not reasons, reasons


def parameter_manifest(spec: TransformSpec) -> dict[str, Any]:
    return {
        "transform_kind": spec.kind,
        "transform_class": "exact" if spec.exact else "approximate",
        "view_slot": int(spec.view_slot),
        "parameter_json": json.dumps(dict(spec.parameters), sort_keys=True, separators=(",", ":")),
        **{f"parameter_{key}": float(value) for key, value in sorted(spec.parameters.items())},
    }


__all__ = [
    "ALL_TRANSFORMS",
    "APPROXIMATE_TRANSFORMS",
    "ENVELOPE_METRICS",
    "STRICT_TRANSFORMS",
    "TransformResult",
    "TransformSpec",
    "apply_transform",
    "choose_transform_spec",
    "evaluate_distribution_guard",
    "exact_inverse_error",
    "fit_distribution_envelope",
    "inverse_exact_transform",
    "json_sha256",
    "official_start_index",
    "parameter_manifest",
    "read_horizontal",
    "read_typewell",
    "regenerate_local_features",
    "resample_typewell_gr",
    "sha256_file",
    "sha256_gzip_decompressed",
    "stable_choice",
    "stable_uint64",
    "summarize_well",
]

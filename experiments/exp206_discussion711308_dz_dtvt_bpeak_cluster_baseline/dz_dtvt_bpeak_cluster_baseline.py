from __future__ import annotations

import gzip
import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_NAME = "exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline"
OUTPUT_PREFIX = "exp206"
HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    mode: str
    k: int | None = None
    feature_set: str | None = None
    holdout_rows: int | None = None


TAIL_SHAPE_INDICES = tuple(range(6))
TAIL_SHAPE_COLUMNS = [
    f"tail300_{curve}_rel_q{idx}" for curve in ("x", "y", "z", "tvt") for idx in TAIL_SHAPE_INDICES
]
WELL_GEOMETRY_COLUMNS = [
    f"well_{curve}_rel_q{idx}" for curve in ("x", "y", "z") for idx in TAIL_SHAPE_INDICES
]


VARIANT_SPECS: dict[str, VariantSpec] = {
    "global_median": VariantSpec("global_median", "global"),
    "prefix_fit": VariantSpec("prefix_fit", "prefix_fit"),
    "known_tvt_fit_full": VariantSpec("known_tvt_fit_full", "query_known_tvt_fit_full"),
    "peak_cluster_median": VariantSpec("peak_cluster_median", "cluster"),
    "nearest_xy_k8": VariantSpec("nearest_xy_k8", "nearest_xy", k=8),
    "hybrid_peak_xy_k8": VariantSpec("hybrid_peak_xy_k8", "hybrid_peak_xy", k=8),
    "exact_typewell_peak_xy_k8": VariantSpec(
        "exact_typewell_peak_xy_k8", "exact_typewell_peak_xy", k=8
    ),
    "step_xyz_tvt_tail_k8": VariantSpec(
        "step_xyz_tvt_tail_k8", "feature_nearest", k=8, feature_set="xyz_tvt_tail"
    ),
    "step_xyz_tvt_tail_prefixab_k5": VariantSpec(
        "step_xyz_tvt_tail_prefixab_k5",
        "feature_nearest",
        k=5,
        feature_set="xyz_tvt_tail_prefixab",
    ),
    "prefix_holdout_source_b_fixeda_h600": VariantSpec(
        "prefix_holdout_source_b_fixeda_h600",
        "prefix_holdout_source_b_fixeda",
        k=16,
        feature_set="xyz_tvt_tail",
        holdout_rows=600,
    ),
    "prefix_holdout_source_ab_h300": VariantSpec(
        "prefix_holdout_source_ab_h300",
        "prefix_holdout_source_ab",
        k=16,
        feature_set="xyz_tvt_tail",
        holdout_rows=300,
    ),
    "discussion_cluster_ab_k12": VariantSpec(
        "discussion_cluster_ab_k12",
        "feature_cluster",
        k=12,
        feature_set="xyz_tvt_tail_shape",
    ),
    "discussion_cluster_ab_k24": VariantSpec(
        "discussion_cluster_ab_k24",
        "feature_cluster",
        k=24,
        feature_set="xyz_tvt_tail_shape",
    ),
    "discussion_cluster_holdout_ab_k24_h300": VariantSpec(
        "discussion_cluster_holdout_ab_k24_h300",
        "feature_cluster_holdout_ab",
        k=24,
        feature_set="xyz_tvt_tail_shape",
        holdout_rows=300,
    ),
    "discussion_fullxyz_cluster_ab_k24": VariantSpec(
        "discussion_fullxyz_cluster_ab_k24",
        "feature_cluster",
        k=24,
        feature_set="full_xyz_tvt_tail_shape",
    ),
    "discussion_fullxyz_cluster_holdout_ab_k24_h300": VariantSpec(
        "discussion_fullxyz_cluster_holdout_ab_k24_h300",
        "feature_cluster_holdout_ab",
        k=24,
        feature_set="full_xyz_tvt_tail_shape",
        holdout_rows=300,
    ),
}


FEATURE_SETS: dict[str, list[str]] = {
    "xyz_tvt_tail": [
        "last_known_x",
        "last_known_y",
        "last_known_z",
        "last_known_tvt",
        "tail300_tvt_slope",
        "tail300_z_slope",
        "tail300_tvt_delta",
        "tail300_z_delta",
    ],
    "xyz_tvt_tail_prefixab": [
        "last_known_x",
        "last_known_y",
        "last_known_z",
        "last_known_tvt",
        "tail300_tvt_slope",
        "tail300_z_slope",
        "tail300_tvt_delta",
        "tail300_z_delta",
        "prefix_a",
        "prefix_b",
    ],
    "xyz_tvt_tail_shape": [
        "last_known_x",
        "last_known_y",
        "last_known_z",
        "last_known_tvt",
        "tail300_tvt_slope",
        "tail300_z_slope",
        "tail300_tvt_delta",
        "tail300_z_delta",
        "tail300_x_delta",
        "tail300_y_delta",
        "tail300_tvt_step_std",
        "tail300_z_step_std",
        *TAIL_SHAPE_COLUMNS,
    ],
    "full_xyz_tvt_tail_shape": [
        "last_known_x",
        "last_known_y",
        "last_known_z",
        "last_known_tvt",
        "x_mean",
        "y_mean",
        "z_mean",
        "x_span",
        "y_span",
        "z_span",
        "md_span",
        "eval_md_span",
        "tail300_tvt_slope",
        "tail300_z_slope",
        "tail300_tvt_delta",
        "tail300_z_delta",
        "tail300_x_delta",
        "tail300_y_delta",
        "tail300_tvt_step_std",
        "tail300_z_step_std",
        *WELL_GEOMETRY_COLUMNS,
        *TAIL_SHAPE_COLUMNS,
    ],
}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256_frame(frame: pd.DataFrame) -> str:
    csv = frame.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(csv.encode("utf-8")).hexdigest()


def well_from_path(path: Path, suffix: str) -> str:
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected well path name: {path}")
    return name[: -len(suffix)]


def list_wells(data_dir: str | Path) -> list[str]:
    data_dir = Path(data_dir)
    wells: list[str] = []
    for path in sorted(data_dir.glob(f"*{HORIZONTAL_SUFFIX}")):
        well = well_from_path(path, HORIZONTAL_SUFFIX)
        if (data_dir / f"{well}{TYPEWELL_SUFFIX}").exists():
            wells.append(well)
    return wells


def load_horizontal(data_dir: str | Path, well: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{well}{HORIZONTAL_SUFFIX}"
    frame = pd.read_csv(path)
    frame["row_idx"] = np.arange(len(frame), dtype=np.int32)
    frame["id"] = [f"{well}_{idx}" for idx in frame["row_idx"]]
    return frame.sort_values("MD").reset_index(drop=True)


def typewell_sha16(data_dir: str | Path, well: str) -> str:
    path = Path(data_dir) / f"{well}{TYPEWELL_SUFFIX}"
    return sha256_path(path)[:16]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def within_abs(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) <= threshold))


def fit_ab_from_arrays(
    md: np.ndarray,
    z: np.ndarray,
    tvt: np.ndarray,
    *,
    min_steps: int,
) -> dict[str, float]:
    md = np.asarray(md, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    tvt = np.asarray(tvt, dtype=np.float64)
    finite = np.isfinite(md) & np.isfinite(z) & np.isfinite(tvt)
    step_mask = finite[1:] & finite[:-1]
    dmd = np.diff(md)
    dz = np.diff(z)
    dtvt = np.diff(tvt)
    step_mask &= np.isfinite(dmd) & np.isfinite(dz) & np.isfinite(dtvt) & (dmd > 0)
    if int(step_mask.sum()) < min_steps:
        return {
            "a": float("nan"),
            "b": float("nan"),
            "fit_steps": int(step_mask.sum()),
            "fit_rmse_rate": float("nan"),
            "fit_corr": float("nan"),
        }

    x = dz[step_mask]
    y = dtvt[step_mask]
    design = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = a * x + b
    corr = (
        float(np.corrcoef(x, y)[0, 1])
        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0
        else float("nan")
    )
    return {
        "a": float(a),
        "b": float(b),
        "fit_steps": int(step_mask.sum()),
        "fit_rmse_rate": rmse(y, pred),
        "fit_corr": corr,
    }


def fit_ab(
    frame: pd.DataFrame,
    tvt_column: str,
    *,
    min_steps: int,
    tail_rows: int | None = None,
) -> dict[str, float]:
    work = frame[frame[tvt_column].notna()].copy()
    if tail_rows is not None and tail_rows > 0:
        work = work.tail(int(tail_rows))
    if work.empty:
        return {
            "a": float("nan"),
            "b": float("nan"),
            "fit_steps": 0,
            "fit_rmse_rate": float("nan"),
            "fit_corr": float("nan"),
        }
    return fit_ab_from_arrays(
        work["MD"].to_numpy(),
        work["Z"].to_numpy(),
        work[tvt_column].to_numpy(),
        min_steps=min_steps,
    )


def summarize_well_geometry(frame: pd.DataFrame) -> dict[str, float]:
    features: dict[str, float] = {}
    for curve_name, column in {
        "x": "X",
        "y": "Y",
        "z": "Z",
    }.items():
        values = frame[column].to_numpy(dtype=np.float64)
        for idx, value in enumerate(relative_tail_samples(values, sample_count=6)):
            features[f"well_{curve_name}_rel_q{idx}"] = value
    return features


def summarize_horizontal(
    data_dir: str | Path,
    well: str,
    *,
    split: str,
    min_fit_steps: int,
    prefix_tail_rows: int,
) -> dict[str, Any]:
    frame = load_horizontal(data_dir, well)
    known_mask = frame["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    if not known_mask.any():
        last_known_idx = 0
        last_known_tvt = float("nan")
        last_known_md = float(frame["MD"].iloc[0])
        last_known_x = float(frame["X"].iloc[0])
        last_known_y = float(frame["Y"].iloc[0])
        last_known_z = float(frame["Z"].iloc[0])
    else:
        last_known_pos = int(np.flatnonzero(known_mask)[-1])
        last_known_idx = int(frame["row_idx"].iloc[last_known_pos])
        last = frame.iloc[last_known_pos]
        last_known_tvt = float(last["TVT_input"])
        last_known_md = float(last["MD"])
        last_known_x = float(last["X"])
        last_known_y = float(last["Y"])
        last_known_z = float(last["Z"])

    prefix_fit = fit_ab(
        frame,
        "TVT_input",
        min_steps=min_fit_steps,
        tail_rows=prefix_tail_rows,
    )
    full_fit = (
        fit_ab(frame, "TVT", min_steps=min_fit_steps, tail_rows=None)
        if "TVT" in frame.columns
        else {
            "a": float("nan"),
            "b": float("nan"),
            "fit_steps": 0,
            "fit_rmse_rate": float("nan"),
            "fit_corr": float("nan"),
        }
    )
    eval_part = frame.loc[eval_mask]
    prefix_tail = frame.loc[known_mask].tail(int(prefix_tail_rows)).copy()
    tail_features = summarize_prefix_tail(prefix_tail)
    x = frame["X"].to_numpy(dtype=np.float64)
    y = frame["Y"].to_numpy(dtype=np.float64)
    z = frame["Z"].to_numpy(dtype=np.float64)
    md = frame["MD"].to_numpy(dtype=np.float64)
    typewell_hash = typewell_sha16(data_dir, well)
    geometry_features = summarize_well_geometry(frame)
    return {
        "split": split,
        "well": well,
        "rows": int(len(frame)),
        "known_rows": int(known_mask.sum()),
        "eval_rows": int(eval_mask.sum()),
        "last_known_idx": last_known_idx,
        "last_known_tvt": last_known_tvt,
        "last_known_md": last_known_md,
        "last_known_x": last_known_x,
        "last_known_y": last_known_y,
        "last_known_z": last_known_z,
        "eval_md_max": float(eval_part["MD"].max()) if len(eval_part) else float("nan"),
        "eval_md_span": float(eval_part["MD"].max() - last_known_md) if len(eval_part) else 0.0,
        "x_mean": float(np.nanmean(x)),
        "y_mean": float(np.nanmean(y)),
        "z_mean": float(np.nanmean(z)),
        "x_span": float(np.nanmax(x) - np.nanmin(x)),
        "y_span": float(np.nanmax(y) - np.nanmin(y)),
        "z_span": float(np.nanmax(z) - np.nanmin(z)),
        "md_span": float(np.nanmax(md) - np.nanmin(md)),
        "prefix_a": prefix_fit["a"],
        "prefix_b": prefix_fit["b"],
        "prefix_fit_steps": prefix_fit["fit_steps"],
        "prefix_fit_rmse_rate": prefix_fit["fit_rmse_rate"],
        "prefix_fit_corr": prefix_fit["fit_corr"],
        "full_a": full_fit["a"],
        "full_b": full_fit["b"],
        "full_fit_steps": full_fit["fit_steps"],
        "full_fit_rmse_rate": full_fit["fit_rmse_rate"],
        "full_fit_corr": full_fit["fit_corr"],
        **geometry_features,
        **tail_features,
        "typewell_exact_sha16": typewell_hash,
        "exact_typewell_group": f"exact_{typewell_hash}",
    }


def relative_tail_samples(values: np.ndarray, *, sample_count: int = 6) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).any():
        return [float("nan")] * int(sample_count)
    if len(values) == 1:
        return [0.0] * int(sample_count)

    source_positions = np.arange(len(values), dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.all():
        values = np.interp(source_positions, source_positions[finite], values[finite])
    sample_positions = np.linspace(0.0, float(len(values) - 1), int(sample_count))
    sampled = np.interp(sample_positions, source_positions, values)
    sampled = sampled - float(values[-1])
    return [float(value) for value in sampled]


def summarize_prefix_tail(frame: pd.DataFrame) -> dict[str, float]:
    keys = [
        "tail300_md_span",
        "tail300_tvt_delta",
        "tail300_z_delta",
        "tail300_x_delta",
        "tail300_y_delta",
        "tail300_tvt_slope",
        "tail300_z_slope",
        "tail300_tvt_step_std",
        "tail300_z_step_std",
        *TAIL_SHAPE_COLUMNS,
    ]
    if len(frame) < 2:
        return {key: float("nan") for key in keys}

    md = frame["MD"].to_numpy(dtype=np.float64)
    tvt = frame["TVT_input"].to_numpy(dtype=np.float64)
    z = frame["Z"].to_numpy(dtype=np.float64)
    x = frame["X"].to_numpy(dtype=np.float64)
    y = frame["Y"].to_numpy(dtype=np.float64)
    md_span = float(md[-1] - md[0]) if math.isfinite(float(md[-1] - md[0])) else float("nan")
    if not math.isfinite(md_span) or abs(md_span) < 1e-12:
        tvt_slope = float("nan")
        z_slope = float("nan")
    else:
        tvt_slope = float((tvt[-1] - tvt[0]) / md_span)
        z_slope = float((z[-1] - z[0]) / md_span)
    samples: dict[str, float] = {}
    for curve_name, values in {
        "x": x,
        "y": y,
        "z": z,
        "tvt": tvt,
    }.items():
        for idx, value in enumerate(relative_tail_samples(values, sample_count=6)):
            samples[f"tail300_{curve_name}_rel_q{idx}"] = value

    return {
        "tail300_md_span": md_span,
        "tail300_tvt_delta": float(tvt[-1] - tvt[0]),
        "tail300_z_delta": float(z[-1] - z[0]),
        "tail300_x_delta": float(x[-1] - x[0]),
        "tail300_y_delta": float(y[-1] - y[0]),
        "tail300_tvt_slope": tvt_slope,
        "tail300_z_slope": z_slope,
        "tail300_tvt_step_std": float(np.nanstd(np.diff(tvt))) if len(tvt) > 2 else float("nan"),
        "tail300_z_step_std": float(np.nanstd(np.diff(z))) if len(z) > 2 else float("nan"),
        **samples,
    }


def build_well_table(
    data_dir: str | Path,
    *,
    split: str,
    min_fit_steps: int,
    prefix_tail_rows: int,
) -> pd.DataFrame:
    rows = [
        summarize_horizontal(
            data_dir,
            well,
            split=split,
            min_fit_steps=min_fit_steps,
            prefix_tail_rows=prefix_tail_rows,
        )
        for well in list_wells(data_dir)
    ]
    if not rows:
        raise FileNotFoundError(f"No wells found under {data_dir}")
    table = pd.DataFrame(rows).sort_values("well").reset_index(drop=True)
    group_sizes = table.groupby("exact_typewell_group")["well"].transform("size")
    table["exact_typewell_group_size_in_split"] = group_sizes.astype(int)
    return table


def peak_centers_from_b(
    values: pd.Series | np.ndarray,
    *,
    bins: int,
    min_count: int,
    max_peaks: int,
    fallback_quantiles: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.array([0.0], dtype=np.float64)
    if len(arr) <= max(3, min_count):
        return np.array([float(np.median(arr))], dtype=np.float64)

    lo, hi = np.percentile(arr, [1.0, 99.0])
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.array([float(np.median(arr))], dtype=np.float64)
    counts, edges = np.histogram(arr, bins=int(bins), range=(float(lo), float(hi)))
    centers = (edges[:-1] + edges[1:]) / 2.0
    smooth = np.convolve(counts.astype(np.float64), np.array([1.0, 2.0, 1.0]) / 4.0, mode="same")
    peak_idx: list[int] = []
    for idx in range(len(smooth)):
        left = smooth[idx - 1] if idx > 0 else -1.0
        right = smooth[idx + 1] if idx + 1 < len(smooth) else -1.0
        if smooth[idx] >= left and smooth[idx] >= right and counts[idx] >= min_count:
            peak_idx.append(idx)

    if not peak_idx:
        quantiles = np.linspace(0.0, 1.0, int(fallback_quantiles) + 2)[1:-1]
        fallback = np.quantile(arr, quantiles)
        return np.unique(np.round(fallback, 10)).astype(np.float64)

    peak_idx = sorted(peak_idx, key=lambda idx: (-counts[idx], centers[idx]))[: int(max_peaks)]
    return np.asarray(sorted(float(centers[idx]) for idx in peak_idx), dtype=np.float64)


def assign_peak(values: pd.Series | np.ndarray, centers: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    labels = np.full(len(values), -1, dtype=np.int32)
    if len(centers) == 0:
        return labels
    finite = np.isfinite(values)
    if finite.any():
        labels[finite] = np.argmin(np.abs(values[finite, None] - centers[None, :]), axis=1)
    return labels


def add_peak_labels(table: pd.DataFrame, centers: np.ndarray) -> pd.DataFrame:
    out = table.copy()
    out["full_b_peak_label"] = assign_peak(out["full_b"].to_numpy(dtype=np.float64), centers)
    out["prefix_b_peak_label"] = assign_peak(out["prefix_b"].to_numpy(dtype=np.float64), centers)
    out["full_b_peak_center"] = [
        float(centers[label]) if 0 <= int(label) < len(centers) else float("nan")
        for label in out["full_b_peak_label"]
    ]
    out["prefix_b_peak_center"] = [
        float(centers[label]) if 0 <= int(label) < len(centers) else float("nan")
        for label in out["prefix_b_peak_label"]
    ]
    return out


def source_default_ab(source: pd.DataFrame) -> tuple[float, float]:
    finite = source[np.isfinite(source["full_a"]) & np.isfinite(source["full_b"])]
    if finite.empty:
        return 0.0, 0.0
    return float(finite["full_a"].median()), float(finite["full_b"].median())


def xy_distance(source: pd.DataFrame, query: pd.Series) -> np.ndarray:
    dx = source["last_known_x"].to_numpy(dtype=np.float64) - float(query["last_known_x"])
    dy = source["last_known_y"].to_numpy(dtype=np.float64) - float(query["last_known_y"])
    return np.sqrt(dx * dx + dy * dy)


def standardized_feature_values(
    source: pd.DataFrame, query: pd.Series, feature_set: str
) -> tuple[np.ndarray, np.ndarray]:
    columns = FEATURE_SETS.get(feature_set)
    if not columns:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    values = source[columns].to_numpy(dtype=np.float64)
    query_values = query[columns].to_numpy(dtype=np.float64)
    combined = np.vstack([values, query_values[None, :]])
    med = np.nanmedian(combined, axis=0)
    q25 = np.nanquantile(combined, 0.25, axis=0)
    q75 = np.nanquantile(combined, 0.75, axis=0)
    scale = q75 - q25
    std = np.nanstd(combined, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    scaled = (values - med) / scale
    scaled_query = (query_values - med) / scale
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    scaled_query = np.where(np.isfinite(scaled_query), scaled_query, 0.0)
    return scaled, scaled_query


def feature_distance(source: pd.DataFrame, query: pd.Series, feature_set: str) -> np.ndarray:
    scaled, scaled_query = standardized_feature_values(source, query, feature_set)
    diff = scaled - scaled_query[None, :]
    return np.sqrt(np.mean(diff * diff, axis=1))


def deterministic_kmeans(
    values: np.ndarray,
    *,
    n_clusters: int,
    max_iter: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape={values.shape}")
    n_rows = int(values.shape[0])
    if n_rows == 0:
        return np.asarray([], dtype=np.int32), np.empty((0, values.shape[1]), dtype=np.float64)

    k = max(1, min(int(n_clusters), n_rows))
    if k == 1:
        labels = np.zeros(n_rows, dtype=np.int32)
        return labels, values.mean(axis=0, keepdims=True)

    centered = values - values.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projection = centered @ vt[0]
    except np.linalg.LinAlgError:
        projection = centered[:, 0]
    order = np.argsort(projection, kind="mergesort")
    init_positions = np.linspace(0, n_rows - 1, k).round().astype(int)
    centers = values[order[init_positions]].copy()
    labels = np.full(n_rows, -1, dtype=np.int32)

    for _ in range(int(max_iter)):
        distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        next_labels = np.argmin(distances, axis=1).astype(np.int32)
        next_centers = centers.copy()
        for label in range(k):
            mask = next_labels == label
            if mask.any():
                next_centers[label] = values[mask].mean(axis=0)
        if np.array_equal(next_labels, labels) and np.allclose(next_centers, centers):
            labels = next_labels
            centers = next_centers
            break
        labels = next_labels
        centers = next_centers
    return labels, centers


def feature_cluster_sources(
    source: pd.DataFrame,
    query: pd.Series,
    *,
    feature_set: str,
    n_clusters: int,
) -> tuple[pd.DataFrame, str]:
    if source.empty:
        return source.copy(), "empty_feature_cluster"
    scaled, scaled_query = standardized_feature_values(source, query, feature_set)
    labels, centers = deterministic_kmeans(scaled, n_clusters=int(n_clusters))
    if len(centers) == 0:
        return source.copy(), "fallback_empty_feature_cluster"

    query_distances = np.sum((centers - scaled_query[None, :]) ** 2, axis=1)
    query_cluster = int(np.argmin(query_distances))
    source_distances = np.sqrt(np.mean((scaled - scaled_query[None, :]) ** 2, axis=1))
    work = source.copy()
    work["_source_distance"] = source_distances
    work["_feature_cluster_label"] = labels
    work["_query_feature_cluster_label"] = query_cluster
    selected = work.loc[labels == query_cluster].sort_values(["_source_distance", "well"])
    if selected.empty:
        return work.sort_values(["_source_distance", "well"]).head(1), "fallback_nearest_feature"
    return selected, f"feature_cluster_{feature_set}_k{int(n_clusters)}"


def aggregate_source_ab(
    source: pd.DataFrame,
    query: pd.Series,
    *,
    aggregation: str,
    distance_floor_ft: float,
) -> tuple[float, float, float]:
    finite = source[np.isfinite(source["full_a"]) & np.isfinite(source["full_b"])].copy()
    if finite.empty:
        return float("nan"), float("nan"), float("nan")
    if "_source_distance" in finite.columns:
        distances = finite["_source_distance"].to_numpy(dtype=np.float64)
    else:
        distances = xy_distance(finite, query)
    if aggregation == "weighted_mean":
        weights = 1.0 / np.maximum(distances, float(distance_floor_ft))
        weights = weights / np.sum(weights)
        a = float(np.sum(finite["full_a"].to_numpy(dtype=np.float64) * weights))
        b = float(np.sum(finite["full_b"].to_numpy(dtype=np.float64) * weights))
        return a, b, float(np.mean(distances))
    return (
        float(finite["full_a"].median()),
        float(finite["full_b"].median()),
        float(np.mean(distances)),
    )


def choose_sources(
    source: pd.DataFrame,
    query: pd.Series,
    spec: VariantSpec,
    *,
    nearest_k: int,
) -> tuple[pd.DataFrame, str]:
    if spec.mode == "global":
        return source.copy(), "global"
    if spec.mode == "cluster":
        selected = source[source["full_b_peak_label"] == int(query["prefix_b_peak_label"])]
        stage = "same_prefix_b_peak" if not selected.empty else "fallback_global"
        return selected if not selected.empty else source.copy(), stage
    if spec.mode == "nearest_xy":
        work = source.copy()
        work["_source_distance"] = xy_distance(work, query)
        selected = work.sort_values(["_source_distance", "well"]).head(int(spec.k or nearest_k))
        return (
            selected,
            "nearest_xy",
        )
    if spec.mode == "feature_nearest":
        work = source.copy()
        work["_source_distance"] = feature_distance(work, query, spec.feature_set or "xyz_tvt_tail")
        selected = work.sort_values(["_source_distance", "well"]).head(int(spec.k or nearest_k))
        return selected, f"nearest_{spec.feature_set}"
    if spec.mode == "feature_cluster":
        return feature_cluster_sources(
            source,
            query,
            feature_set=spec.feature_set or "xyz_tvt_tail_shape",
            n_clusters=int(spec.k or nearest_k),
        )
    if spec.mode == "hybrid_peak_xy":
        peak_label = int(query["prefix_b_peak_label"])
        selected = source[source["full_b_peak_label"] == peak_label].copy()
        stage = "same_prefix_b_peak_xy"
        if selected.empty:
            selected = source.copy()
            stage = "fallback_nearest_xy"
        selected["_source_distance"] = xy_distance(selected, query)
        selected = selected.sort_values(["_source_distance", "well"]).head(int(spec.k or nearest_k))
        return (
            selected,
            stage,
        )
    if spec.mode == "exact_typewell_peak_xy":
        same_type = source[source["exact_typewell_group"] == str(query["exact_typewell_group"])]
        peak_label = int(query["prefix_b_peak_label"])
        selected = same_type[same_type["full_b_peak_label"] == peak_label].copy()
        stage = "same_typewell_and_peak_xy"
        if selected.empty and not same_type.empty:
            selected = same_type.copy()
            stage = "same_typewell_xy"
        if selected.empty:
            selected = source[source["full_b_peak_label"] == peak_label].copy()
            stage = "same_peak_xy"
        if selected.empty:
            selected = source.copy()
            stage = "fallback_nearest_xy"
        selected["_source_distance"] = xy_distance(selected, query)
        selected = selected.sort_values(["_source_distance", "well"]).head(int(spec.k or nearest_k))
        return (
            selected,
            stage,
        )
    raise ValueError(f"Unsupported variant mode: {spec.mode}")


def select_ab_for_variant(
    source: pd.DataFrame,
    query: pd.Series,
    spec: VariantSpec,
    *,
    nearest_k: int,
    aggregation: str,
    distance_floor_ft: float,
    min_fit_steps: int,
    query_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    default_a, default_b = source_default_ab(source)
    if spec.mode in {"prefix_fit", "query_known_tvt_fit_full"}:
        if spec.mode == "query_known_tvt_fit_full":
            if query_frame is None:
                raise ValueError(f"query_frame is required for {spec.mode}")
            fit = fit_ab(
                query_frame,
                "TVT_input",
                min_steps=int(min_fit_steps),
                tail_rows=None,
            )
            candidate_a = float(fit["a"])
            candidate_b = float(fit["b"])
            fit_steps = int(fit["fit_steps"])
            fit_rmse_rate = float(fit["fit_rmse_rate"])
            fit_corr = float(fit["fit_corr"])
            fit_source = "query_known_tvt_full"
        else:
            candidate_a = float(query["prefix_a"])
            candidate_b = float(query["prefix_b"])
            fit_steps = int(query["prefix_fit_steps"])
            fit_rmse_rate = float(query["prefix_fit_rmse_rate"])
            fit_corr = float(query["prefix_fit_corr"])
            fit_source = "query_known_tvt_tail"
        has_fit = math.isfinite(candidate_a) and math.isfinite(candidate_b)
        a = candidate_a if has_fit else default_a
        b = candidate_b if has_fit else default_b
        stage = f"{fit_source}_fit" if has_fit else "fallback_source_median"
        return {
            "variant": spec.name,
            "source_count": 0,
            "fallback_stage": stage,
            "assigned_a": a,
            "assigned_b": b,
            "query_fit_source": fit_source,
            "query_fit_steps": fit_steps,
            "query_fit_rmse_rate": fit_rmse_rate,
            "query_fit_corr": fit_corr,
            "source_mean_xy_distance": float("nan"),
            "same_typewell_share": float("nan"),
            "same_peak_share": float("nan"),
        }

    if spec.mode in {
        "prefix_holdout_source_b_fixeda",
        "prefix_holdout_source_ab",
        "feature_cluster_holdout_ab",
    }:
        if query_frame is None:
            raise ValueError(f"query_frame is required for {spec.mode}")
        if spec.mode == "feature_cluster_holdout_ab":
            selected, stage = choose_sources(
                source,
                query,
                VariantSpec(
                    spec.name,
                    "feature_cluster",
                    k=spec.k,
                    feature_set=spec.feature_set or "xyz_tvt_tail_shape",
                ),
                nearest_k=nearest_k,
            )
            holdout_mode = "prefix_holdout_source_ab"
        else:
            selected, stage = choose_sources(
                source,
                query,
                VariantSpec(
                    spec.name,
                    "feature_nearest",
                    k=spec.k,
                    feature_set=spec.feature_set or "xyz_tvt_tail",
                ),
                nearest_k=nearest_k,
            )
            holdout_mode = spec.mode
        best_a, best_b, best_score = select_prefix_holdout_ab(
            query_frame,
            selected,
            default_a=default_a,
            default_b=default_b,
            holdout_rows=int(spec.holdout_rows or 300),
            mode=holdout_mode,
        )
        if selected.empty:
            same_typewell_share = float("nan")
            same_peak_share = float("nan")
        else:
            same_type = selected["exact_typewell_group"].astype(str)
            query_type = str(query["exact_typewell_group"])
            same_typewell_share = float(np.mean(same_type == query_type))
            same_peak = selected["full_b_peak_label"].astype(int)
            query_peak = int(query["prefix_b_peak_label"])
            same_peak_share = float(np.mean(same_peak == query_peak))
        return {
            "variant": spec.name,
            "source_count": int(len(selected)),
            "fallback_stage": f"{stage}_prefix_holdout",
            "assigned_a": best_a,
            "assigned_b": best_b,
            "source_mean_xy_distance": float(np.nanmean(xy_distance(selected, query)))
            if not selected.empty
            else float("nan"),
            "same_typewell_share": same_typewell_share,
            "same_peak_share": same_peak_share,
            "prefix_holdout_rmse": best_score,
        }

    selected, stage = choose_sources(source, query, spec, nearest_k=nearest_k)
    a, b, mean_distance = aggregate_source_ab(
        selected, query, aggregation=aggregation, distance_floor_ft=distance_floor_ft
    )
    if not math.isfinite(a) or not math.isfinite(b):
        a, b = default_a, default_b
        stage = f"{stage}_fallback_source_median"
    if selected.empty:
        same_typewell_share = float("nan")
        same_peak_share = float("nan")
    else:
        same_type = selected["exact_typewell_group"].astype(str)
        query_type = str(query["exact_typewell_group"])
        same_typewell_share = float(np.mean(same_type == query_type))
        same_peak = selected["full_b_peak_label"].astype(int)
        query_peak = int(query["prefix_b_peak_label"])
        same_peak_share = float(np.mean(same_peak == query_peak))
    return {
        "variant": spec.name,
        "source_count": int(len(selected)),
        "fallback_stage": stage,
        "assigned_a": a,
        "assigned_b": b,
        "source_mean_xy_distance": mean_distance,
        "same_typewell_share": same_typewell_share,
        "same_peak_share": same_peak_share,
    }


def predict_positions_from_anchor(
    frame: pd.DataFrame,
    *,
    start_pos: int,
    target_positions: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    target_positions = np.asarray(target_positions, dtype=np.int64)
    if len(target_positions) == 0:
        return np.asarray([], dtype=np.float64)
    max_pos = int(np.max(target_positions))
    if max_pos <= start_pos:
        return np.asarray([], dtype=np.float64)
    z = frame["Z"].to_numpy(dtype=np.float64)
    dz = np.diff(z)[start_pos:max_pos]
    dz = np.where(np.isfinite(dz), dz, 0.0)
    steps = np.arange(1, len(dz) + 1, dtype=np.float64)
    anchor = float(frame["TVT_input"].iloc[start_pos])
    all_pred = anchor + float(a) * np.cumsum(dz) + float(b) * steps
    rel = target_positions - int(start_pos) - 1
    return all_pred[rel]


def score_ab_on_prefix_holdout(
    frame: pd.DataFrame,
    *,
    a: float,
    b: float,
    holdout_rows: int,
    min_anchor_rows: int = 50,
) -> float:
    known_positions = np.flatnonzero(frame["TVT_input"].notna().to_numpy())
    if len(known_positions) <= int(holdout_rows) + int(min_anchor_rows):
        return float("inf")
    holdout_positions = known_positions[-int(holdout_rows) :]
    start_pos = int(known_positions[-int(holdout_rows) - 1])
    y = frame["TVT_input"].to_numpy(dtype=np.float64)[holdout_positions]
    pred = predict_positions_from_anchor(
        frame,
        start_pos=start_pos,
        target_positions=holdout_positions,
        a=a,
        b=b,
    )
    if len(pred) != len(y):
        return float("inf")
    return rmse(y, pred)


def select_prefix_holdout_ab(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    default_a: float,
    default_b: float,
    holdout_rows: int,
    mode: str,
) -> tuple[float, float, float]:
    finite = selected[np.isfinite(selected["full_a"]) & np.isfinite(selected["full_b"])]
    candidates: list[tuple[float, float]] = []
    if mode == "prefix_holdout_source_b_fixeda":
        candidates.extend((-1.0, float(value)) for value in finite["full_b"].to_numpy())
    elif mode == "prefix_holdout_source_ab":
        candidates.extend(
            (float(row.full_a), float(row.full_b)) for row in finite.itertuples(index=False)
        )
    else:
        raise ValueError(f"Unsupported prefix holdout mode: {mode}")
    if not candidates:
        candidates.append((default_a, default_b))

    best_score = float("inf")
    best_a = float(default_a)
    best_b = float(default_b)
    seen: set[tuple[float, float]] = set()
    for a, b in candidates:
        if not math.isfinite(a) or not math.isfinite(b):
            continue
        key = (round(float(a), 8), round(float(b), 8))
        if key in seen:
            continue
        seen.add(key)
        score = score_ab_on_prefix_holdout(
            frame,
            a=float(a),
            b=float(b),
            holdout_rows=int(holdout_rows),
        )
        if score < best_score:
            best_score = score
            best_a = float(a)
            best_b = float(b)
    if not math.isfinite(best_score):
        best_a = float(default_a)
        best_b = float(default_b)
    return best_a, best_b, best_score


def integrate_unknown_suffix(frame: pd.DataFrame, *, a: float, b: float) -> np.ndarray:
    known_mask = frame["TVT_input"].notna().to_numpy()
    unknown_mask = ~known_mask
    out = np.full(len(frame), np.nan, dtype=np.float64)
    if not known_mask.any():
        start_pos = 0
        out[start_pos] = 0.0
    else:
        start_pos = int(np.flatnonzero(known_mask)[-1])
        out[start_pos] = float(frame["TVT_input"].iloc[start_pos])

    md = frame["MD"].to_numpy(dtype=np.float64)
    z = frame["Z"].to_numpy(dtype=np.float64)
    for pos in range(start_pos + 1, len(frame)):
        dmd = md[pos] - md[pos - 1]
        dz = z[pos] - z[pos - 1]
        if not math.isfinite(dmd) or dmd <= 0 or not math.isfinite(dz):
            out[pos] = out[pos - 1]
        else:
            out[pos] = out[pos - 1] + float(a) * dz + float(b)
    return out[unknown_mask]


def prediction_base_frame(frame: pd.DataFrame, well: str) -> pd.DataFrame:
    unknown = frame[frame["TVT_input"].isna()].copy()
    known = frame[frame["TVT_input"].notna()]
    last_known_md = float(known["MD"].iloc[-1]) if len(known) else float(frame["MD"].iloc[0])
    last_known_tvt = float(known["TVT_input"].iloc[-1]) if len(known) else float("nan")
    return pd.DataFrame(
        {
            "id": unknown["id"].astype(str).to_numpy(),
            "well": well,
            "row_idx": unknown["row_idx"].to_numpy(dtype=np.int32),
            "md": unknown["MD"].to_numpy(dtype=np.float64),
            "md_since": unknown["MD"].to_numpy(dtype=np.float64) - last_known_md,
            "target_tvt": unknown["TVT"].to_numpy(dtype=np.float64)
            if "TVT" in unknown.columns
            else np.nan,
            "last_known_tvt": last_known_tvt,
        }
    )


def metric_record(frame: pd.DataFrame, pred_column: str, *, variant: str) -> dict[str, Any]:
    y = frame["target_tvt"].to_numpy(dtype=np.float64)
    p = frame[pred_column].to_numpy(dtype=np.float64)
    return {
        "variant": variant,
        "rows": int(np.isfinite(y).sum()),
        "rmse": rmse(y, p),
        "mae": mae(y, p),
        "within10": within_abs(y, p, 10.0),
        "bias_pred_minus_true": float(np.nanmean(p - y)),
    }


def distance_bucket_metrics(
    predictions: pd.DataFrame,
    variants: list[str],
    bucket_edges: list[float],
) -> pd.DataFrame:
    labels = []
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:], strict=True):
        label = f"{int(lo):03d}_{int(hi):03d}" if hi < 1e8 else f"{int(lo):04d}_plus"
        labels.append(label)
    bucket = pd.cut(
        predictions["md_since"],
        bins=bucket_edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    rows: list[dict[str, Any]] = []
    work = predictions.assign(distance_bucket=bucket)
    for variant in variants:
        pred_col = f"pred_{variant}"
        for bucket_name, part in work.groupby("distance_bucket", observed=True, sort=True):
            rec = metric_record(part, pred_col, variant=variant)
            rec["distance_bucket"] = str(bucket_name)
            rows.append(rec)
    return pd.DataFrame(rows)


def by_well_metrics(predictions: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        pred_col = f"pred_{variant}"
        for well, part in predictions.groupby("well", sort=True):
            rec = metric_record(part, pred_col, variant=variant)
            rec["well"] = well
            rec["md_since_max"] = float(part["md_since"].max())
            rows.append(rec)
    return (
        pd.DataFrame(rows)
        .sort_values(["variant", "rmse"], ascending=[True, False])
        .reset_index(drop=True)
    )


def cluster_purity(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, part in table.groupby("exact_typewell_group", sort=True):
        labels = part["full_b_peak_label"].astype(int)
        counts = labels.value_counts().sort_index()
        total = int(counts.sum())
        majority = int(counts.max()) if total else 0
        majority_label = int(counts.idxmax()) if total else -1
        rows.append(
            {
                "exact_typewell_group": group,
                "wells": total,
                "majority_full_b_peak_label": majority_label,
                "majority_share": float(majority / total) if total else float("nan"),
                "unique_full_b_peaks": int(labels.nunique()),
                "full_b_mean": float(part["full_b"].mean()),
                "full_b_std": float(part["full_b"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["wells", "majority_share"], ascending=[False, False])


def peak_summary(table: pd.DataFrame, centers: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, center in enumerate(centers):
        rows.append(
            {
                "b_peak_label": idx,
                "b_peak_center": float(center),
                "full_fit_wells": int((table["full_b_peak_label"] == idx).sum()),
                "prefix_fit_wells": int((table["prefix_b_peak_label"] == idx).sum()),
            }
        )
    return pd.DataFrame(rows)


def _color_for_label(label: int) -> str:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    return "#cccccc" if label < 0 else palette[int(label) % len(palette)]


def write_xy_svg(table: pd.DataFrame, label_col: str, path: Path, *, title: str) -> None:
    width = 920
    height = 720
    pad = 52
    x = table["last_known_x"].to_numpy(dtype=np.float64)
    y = table["last_known_y"].to_numpy(dtype=np.float64)
    labels = table[label_col].to_numpy(dtype=np.int32)
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    viewbox = f"0 0 {width} {height}"
    text_title = html.escape(title)
    plot_width = width - 2 * pad
    plot_height = height - 2 * pad
    svg_open = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="{viewbox}">'
    )
    parts = [
        svg_open,
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad}" y="32" font-family="sans-serif" font-size="20">{text_title}</text>',
        f'<rect x="{pad}" y="{pad}" width="{plot_width}" height="{plot_height}" '
        'fill="#f7f7f7" stroke="#333"/>',
    ]
    for row, x_value, y_value, label in zip(
        table.itertuples(index=False), x, y, labels, strict=True
    ):
        sx = pad + (x_value - x_min) / (x_max - x_min) * (width - 2 * pad)
        sy = height - pad - (y_value - y_min) / (y_max - y_min) * (height - 2 * pad)
        color = _color_for_label(int(label))
        full_b = float(getattr(row, "full_b", float("nan")))
        title_text = html.escape(f"{row.well} {label_col}={int(label)} b={full_b:.5f}")
        parts.append(
            f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="4.0" fill="{color}" fill-opacity="0.78">'
            f"<title>{title_text}</title></circle>"
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def resolve_variants(config: dict[str, Any]) -> list[str]:
    params = model_params(config)
    names = params.get("variants") or list(VARIANT_SPECS)
    unknown = [name for name in names if name not in VARIANT_SPECS]
    if unknown:
        raise ValueError(f"Unknown variant(s): {unknown}")
    return [str(name) for name in names]


def model_params(config: dict[str, Any]) -> dict[str, Any]:
    return dict((config.get("model") or {}).get("params") or {})


def assign_for_query(
    source: pd.DataFrame,
    query: pd.Series,
    centers: np.ndarray,
    variants: list[str],
    params: dict[str, Any],
    query_frame: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    source = add_peak_labels(source, centers)
    query = query.copy()
    query["prefix_b_peak_label"] = int(assign_peak(np.asarray([query["prefix_b"]]), centers)[0])
    assignments: list[dict[str, Any]] = []
    for variant in variants:
        record = select_ab_for_variant(
            source,
            query,
            VARIANT_SPECS[variant],
            nearest_k=int(params.get("nearest_k", 8)),
            aggregation=str(params.get("aggregation", "weighted_mean")),
            distance_floor_ft=float(params.get("distance_floor_ft", 100.0)),
            min_fit_steps=int(params.get("min_fit_steps", 20)),
            query_frame=query_frame,
        )
        record.update(
            {
                "well": str(query["well"]),
                "query_prefix_a": float(query["prefix_a"]),
                "query_prefix_b": float(query["prefix_b"]),
                "query_prefix_b_peak_label": int(query["prefix_b_peak_label"]),
                "query_exact_typewell_group": str(query["exact_typewell_group"]),
                "b_peak_centers": "|".join(f"{float(value):.8g}" for value in centers),
            }
        )
        assignments.append(record)
    return assignments


def build_predictions_for_well(
    data_dir: Path,
    well: str,
    source: pd.DataFrame,
    query: pd.Series,
    centers: np.ndarray,
    variants: list[str],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frame = load_horizontal(data_dir, well)
    base = prediction_base_frame(frame, well)
    assignments = assign_for_query(
        source,
        query,
        centers,
        variants,
        params,
        query_frame=frame,
    )
    for assignment in assignments:
        pred = integrate_unknown_suffix(
            frame,
            a=float(assignment["assigned_a"]),
            b=float(assignment["assigned_b"]),
        )
        base[f"pred_{assignment['variant']}"] = pred
    return base, assignments


def run_train_audit(paths: Any, config: dict[str, Any]) -> dict[str, Any]:
    paths.ensure_output_dirs()
    params = model_params(config)
    variants = resolve_variants(config)
    selected_variant = str(params.get("selected_variant") or variants[-1])
    if selected_variant not in variants:
        raise ValueError(f"selected_variant must be one of variants: {selected_variant}")
    min_fit_steps = int(params.get("min_fit_steps", 20))
    prefix_tail_rows = int(params.get("prefix_tail_rows", 300))

    train_table = build_well_table(
        paths.train_data_dir,
        split="train",
        min_fit_steps=min_fit_steps,
        prefix_tail_rows=prefix_tail_rows,
    )
    all_centers = peak_centers_from_b(
        train_table["full_b"],
        bins=int(params.get("b_peak_bins", 32)),
        min_count=int(params.get("b_peak_min_count", 8)),
        max_peaks=int(params.get("b_peak_max_peaks", 8)),
        fallback_quantiles=int(params.get("b_peak_fallback_quantiles", 5)),
    )
    train_table = add_peak_labels(train_table, all_centers)

    prediction_parts: list[pd.DataFrame] = []
    assignment_records: list[dict[str, Any]] = []
    for query in train_table.sort_values("well").itertuples(index=False):
        query_series = pd.Series(query._asdict())
        source = train_table[train_table["well"] != str(query.well)].copy()
        centers = peak_centers_from_b(
            source["full_b"],
            bins=int(params.get("b_peak_bins", 32)),
            min_count=int(params.get("b_peak_min_count", 8)),
            max_peaks=int(params.get("b_peak_max_peaks", 8)),
            fallback_quantiles=int(params.get("b_peak_fallback_quantiles", 5)),
        )
        pred_part, assignments = build_predictions_for_well(
            paths.train_data_dir,
            str(query.well),
            source,
            query_series,
            centers,
            variants,
            params,
        )
        prediction_parts.append(pred_part)
        assignment_records.extend(assignments)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    assignments = pd.DataFrame(assignment_records)

    metrics = pd.DataFrame(
        [metric_record(predictions, f"pred_{variant}", variant=variant) for variant in variants]
    ).sort_values("rmse")
    diagnostics = (config.get("model") or {}).get("diagnostics") or {}
    bucket_edges = [float(value) for value in diagnostics.get("distance_buckets")]
    bucket_metrics = distance_bucket_metrics(predictions, variants, bucket_edges)
    well_metrics = by_well_metrics(predictions, variants)
    purity = cluster_purity(train_table)
    peaks = peak_summary(train_table, all_centers)
    assignment_summary = (
        assignments.groupby(["variant", "fallback_stage"], sort=True)
        .agg(
            wells=("well", "count"),
            source_count_mean=("source_count", "mean"),
            assigned_a_mean=("assigned_a", "mean"),
            assigned_b_mean=("assigned_b", "mean"),
            same_typewell_share_mean=("same_typewell_share", "mean"),
            same_peak_share_mean=("same_peak_share", "mean"),
        )
        .reset_index()
    )

    out_dir = paths.artifacts_dir
    full_fit_path = out_dir / f"{OUTPUT_PREFIX}_full_fit_by_well.csv"
    peak_path = out_dir / f"{OUTPUT_PREFIX}_b_peak_summary.csv"
    metrics_path = out_dir / f"{OUTPUT_PREFIX}_train_variant_metrics.csv"
    bucket_path = out_dir / f"{OUTPUT_PREFIX}_distance_bucket_metrics.csv"
    by_well_path = out_dir / f"{OUTPUT_PREFIX}_by_well_metrics.csv"
    purity_path = out_dir / f"{OUTPUT_PREFIX}_cluster_purity.csv"
    assignment_path = out_dir / f"{OUTPUT_PREFIX}_reference_assignments.csv"
    assignment_summary_path = out_dir / f"{OUTPUT_PREFIX}_reference_assignment_summary.csv"
    prediction_path = out_dir / f"{OUTPUT_PREFIX}_train_oof_predictions.csv.gz"
    summary_path = out_dir / f"{OUTPUT_PREFIX}_summary.json"
    xy_path = out_dir / f"{OUTPUT_PREFIX}_b_peak_xy_map.svg"
    prefix_xy_path = out_dir / f"{OUTPUT_PREFIX}_prefix_b_peak_xy_map.svg"

    train_table.to_csv(full_fit_path, index=False)
    peaks.to_csv(peak_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    well_metrics.to_csv(by_well_path, index=False)
    purity.to_csv(purity_path, index=False)
    assignments.to_csv(assignment_path, index=False)
    assignment_summary.to_csv(assignment_summary_path, index=False)
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    if bool(diagnostics.get("write_svg_maps", True)):
        write_xy_svg(
            train_table, "full_b_peak_label", xy_path, title="Full true b peak label by XY"
        )
        write_xy_svg(
            train_table, "prefix_b_peak_label", prefix_xy_path, title="Prefix b peak label by XY"
        )

    best = metrics.iloc[0].to_dict()
    selected = metrics[metrics["variant"] == selected_variant].iloc[0].to_dict()
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "train_audit_completed",
        "selected_variant": selected_variant,
        "best_variant": best,
        "selected_variant_metrics": selected,
        "train_wells": int(train_table["well"].nunique()),
        "train_unknown_rows": int(len(predictions)),
        "b_peak_centers": [float(value) for value in all_centers],
        "artifacts": {
            "full_fit_by_well": str(full_fit_path),
            "b_peak_summary": str(peak_path),
            "train_variant_metrics": str(metrics_path),
            "distance_bucket_metrics": str(bucket_path),
            "by_well_metrics": str(by_well_path),
            "cluster_purity": str(purity_path),
            "reference_assignments": str(assignment_path),
            "reference_assignment_summary": str(assignment_summary_path),
            "train_oof_predictions": str(prediction_path),
            "summary": str(summary_path),
            "b_peak_xy_map": str(xy_path),
            "prefix_b_peak_xy_map": str(prefix_xy_path),
        },
        "sha256": {
            "train_oof_predictions_raw_gzip": sha256_path(prediction_path),
            "train_oof_predictions_decompressed": sha256_gzip_decompressed(prediction_path),
            "full_fit_by_well": sha256_path(full_fit_path),
            "variant_metrics": sha256_path(metrics_path),
        },
    }
    write_json(summary_path, summary)
    paths.metrics_path.write_text(
        # Keep this short enough for experiment_summary.md while preserving the source idea.
        json.dumps(
            to_jsonable(
                {
                    "experiment": EXPERIMENT_NAME,
                    "route": "pf_beam",
                    "status": "train_audit_completed",
                    "metric": "rmse_tvt",
                    "cv": selected.get("rmse"),
                    "public_lb": None,
                    "private_lb": None,
                    "key_idea": (
                        "No-ML dZ/dTVT b-peak cluster direct baseline from discussion 711308."
                    ),
                    "selected_variant": selected_variant,
                    "best_variant": best,
                    "selected_variant_metrics": selected,
                    "b_peak_count": int(len(all_centers)),
                    "train_wells": int(train_table["well"].nunique()),
                    "train_unknown_rows": int(len(predictions)),
                    "feature_content_sha256": summary["sha256"][
                        "train_oof_predictions_decompressed"
                    ],
                    "updated_at": "2026-07-06",
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return summary


def run_inference(paths: Any, config: dict[str, Any]) -> dict[str, Any]:
    paths.ensure_output_dirs()
    params = model_params(config)
    variants = resolve_variants(config)
    selected_variant = str(
        ((config.get("inference") or {}).get("selected_variant"))
        or params.get("selected_variant")
        or variants[-1]
    )
    if selected_variant not in variants:
        raise ValueError(f"selected_variant must be one of variants: {selected_variant}")
    min_fit_steps = int(params.get("min_fit_steps", 20))
    prefix_tail_rows = int(params.get("prefix_tail_rows", 300))

    train_table = build_well_table(
        paths.train_data_dir,
        split="train",
        min_fit_steps=min_fit_steps,
        prefix_tail_rows=prefix_tail_rows,
    )
    test_table = build_well_table(
        paths.test_data_dir,
        split="test",
        min_fit_steps=min_fit_steps,
        prefix_tail_rows=prefix_tail_rows,
    )
    train_group_counts = train_table["exact_typewell_group"].value_counts()
    test_table["exact_typewell_group_size_in_train"] = (
        test_table["exact_typewell_group"].map(train_group_counts).fillna(0).astype(int)
    )
    centers = peak_centers_from_b(
        train_table["full_b"],
        bins=int(params.get("b_peak_bins", 32)),
        min_count=int(params.get("b_peak_min_count", 8)),
        max_peaks=int(params.get("b_peak_max_peaks", 8)),
        fallback_quantiles=int(params.get("b_peak_fallback_quantiles", 5)),
    )
    train_table = add_peak_labels(train_table, centers)

    prediction_parts: list[pd.DataFrame] = []
    assignment_records: list[dict[str, Any]] = []
    for query in test_table.sort_values("well").itertuples(index=False):
        query_series = pd.Series(query._asdict())
        pred_part, assignments = build_predictions_for_well(
            paths.test_data_dir,
            str(query.well),
            train_table,
            query_series,
            centers,
            variants,
            params,
        )
        prediction_parts.append(pred_part)
        assignment_records.extend(assignments)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    assignments = pd.DataFrame(assignment_records)
    selected_col = f"pred_{selected_variant}"

    sample = pd.read_csv(paths.sample_submission_path)
    if "id" not in sample.columns:
        raise ValueError(f"sample submission missing id column: {paths.sample_submission_path}")
    pred_map = predictions.set_index("id")[selected_col].to_dict()
    sample_ids = sample["id"].astype(str)
    missing_ids = [str(value) for value in sample_ids if str(value) not in pred_map]
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise ValueError(f"Missing predictions for {len(missing_ids)} sample ids, first: {preview}")
    submission = pd.DataFrame(
        {
            "id": sample_ids,
            "tvt": [float(pred_map[str(value)]) for value in sample_ids],
        }
    )
    submission.to_csv(paths.submission_path, index=False)

    out_dir = paths.artifacts_dir
    assignment_path = out_dir / f"{OUTPUT_PREFIX}_test_assignments.csv"
    prediction_summary_path = out_dir / f"{OUTPUT_PREFIX}_test_prediction_summary.json"
    assignments.to_csv(assignment_path, index=False)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "inference_completed",
        "selected_variant": selected_variant,
        "test_wells": int(test_table["well"].nunique()),
        "submission_rows": int(len(submission)),
        "prediction_min": float(submission["tvt"].min()),
        "prediction_max": float(submission["tvt"].max()),
        "prediction_mean": float(submission["tvt"].mean()),
        "prediction_std": float(submission["tvt"].std(ddof=0)),
        "b_peak_centers": [float(value) for value in centers],
        "artifacts": {
            "test_assignments": str(assignment_path),
            "test_prediction_summary": str(prediction_summary_path),
            "submission": str(paths.submission_path),
        },
        "sha256": {
            "submission_csv": sha256_path(paths.submission_path),
            "submission_content": content_sha256_frame(submission),
            "test_assignments": sha256_path(assignment_path),
        },
    }
    write_json(prediction_summary_path, summary)
    return summary

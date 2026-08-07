from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config
from sklearn.metrics import log_loss, roc_auc_score

EXP072_ARTIFACTS = Path("experiments") / "exp072_exp063_full_replay_feature_cache" / "artifacts"
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
FULL_REPLAY_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
OUTPUT_PREFIX = "exp131_gr_shape_descriptor_matching_ablation"
TRAIN_FEATURE_CACHE_VARIANT = "descriptor_scores"
TRAIN_FEATURE_CACHE_FILENAME = (
    f"{OUTPUT_PREFIX}_{TRAIN_FEATURE_CACHE_VARIANT}_train_features.csv.gz"
)
TRAIN_FEATURE_SCHEMA_FILENAME = f"{OUTPUT_PREFIX}_{TRAIN_FEATURE_CACHE_VARIANT}_feature_schema.csv"
REAL_SCORE_VARIANTS = [
    "raw_point_real",
    "ncc_window_real",
    "banded_shift_real",
    "shape_descriptor_real",
    "combo_descriptor_real",
]
NEGATIVE_CONTROL_VARIANTS = ["combo_descriptor_shuffled", "no_gr_constant"]
ALL_SCORE_VARIANTS = REAL_SCORE_VARIANTS + NEGATIVE_CONTROL_VARIANTS


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    source_column: str
    transform: str
    role: str
    enabled: bool = True


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
    for raw_id in ids.astype(str).to_numpy():
        digest.update(raw_id.encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.asarray(values, dtype=np.float32).tobytes())
    return digest.hexdigest()


def find_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    local_artifacts: Path = EXP072_ARTIFACTS,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            local_artifacts / filename,
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
        ]
    )
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def numeric_array(frame: pd.DataFrame, column: str, *, default: float | None = None) -> np.ndarray:
    if column not in frame.columns:
        if default is None:
            raise ValueError(f"required column is missing: {column}")
        return np.full(len(frame), float(default), dtype=np.float32)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


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


def read_feature_cache(
    cache_path: str | Path | None,
    *,
    required_columns: list[str],
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(FULL_REPLAY_TRAIN_FEATURES, cache_path)
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
    schema_path: Path | None = None
    try:
        schema_path = find_artifact(FULL_REPLAY_FEATURE_SCHEMA)
    except FileNotFoundError:
        schema_path = None
    return frame, {
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


def build_required_columns(
    candidate_specs: list[CandidateSpec],
    extra_columns: list[str],
) -> list[str]:
    columns = {"id", "well", "target", "last_known_tvt"}
    columns.update(extra_columns)
    for spec in candidate_specs:
        columns.add(spec.source_column)
    return sorted(columns)


def materialize_existing_candidates(
    frame: pd.DataFrame,
    candidate_specs: list[CandidateSpec],
) -> pd.DataFrame:
    out = frame[["id", "well", "target", "last_known_tvt"]].copy()
    last_known = numeric_array(frame, "last_known_tvt")
    out["true_tvt"] = last_known + numeric_array(frame, "target")
    for spec in candidate_specs:
        if not spec.enabled:
            continue
        values = numeric_array(frame, spec.source_column)
        if spec.transform == "absolute":
            out[spec.name] = values
        elif spec.transform == "base_plus_delta":
            out[spec.name] = last_known + values
        else:
            raise ValueError(f"Unsupported candidate transform: {spec.transform}")
    return out


def _nearest_prefix_indices(prefix_tvt: np.ndarray, candidate_tvt: np.ndarray) -> np.ndarray:
    order = np.argsort(prefix_tvt)
    sorted_tvt = prefix_tvt[order]
    positions = np.searchsorted(sorted_tvt, candidate_tvt, side="left")
    left = np.clip(positions - 1, 0, len(sorted_tvt) - 1)
    right = np.clip(positions, 0, len(sorted_tvt) - 1)
    choose_right = np.abs(sorted_tvt[right] - candidate_tvt) < np.abs(
        sorted_tvt[left] - candidate_tvt
    )
    nearest_sorted = np.where(choose_right, right, left)
    return order[nearest_sorted].astype(np.int32)


def _standardize_last_axis(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    scale = values.std(axis=-1, keepdims=True) + 1e-6
    return centered / scale


def _local_vectors(series: np.ndarray, centers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    indices = np.clip(centers[..., None] + offsets.astype(np.int32), 0, len(series) - 1)
    return series[indices]


def _local_derivatives(
    series: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray,
    *,
    step: int,
) -> np.ndarray:
    left = np.clip(centers[..., None] + offsets.astype(np.int32) - int(step), 0, len(series) - 1)
    right = np.clip(centers[..., None] + offsets.astype(np.int32) + int(step), 0, len(series) - 1)
    return (series[right] - series[left]) / max(2 * int(step), 1)


def _peak_count_proxy(vectors: np.ndarray) -> np.ndarray:
    diffs = np.diff(vectors, axis=-1)
    signs = np.sign(diffs)
    turns = signs[..., 1:] * signs[..., :-1] < 0
    return turns.sum(axis=-1).astype(np.float32)


def _shifted_window_mae(
    series: np.ndarray,
    eval_centers: np.ndarray,
    candidate_centers: np.ndarray,
    window_offsets: np.ndarray,
    shift_offsets: np.ndarray,
) -> np.ndarray:
    eval_vectors = _local_vectors(series, eval_centers, window_offsets)
    shifted_costs = []
    for shift in shift_offsets:
        candidate_vectors = _local_vectors(series, candidate_centers + int(shift), window_offsets)
        shifted_costs.append(np.mean(np.abs(candidate_vectors - eval_vectors[:, None, :]), axis=2))
    return np.min(np.stack(shifted_costs, axis=2), axis=2).astype(np.float32)


def _descriptor_scores_for_gr(
    *,
    full_gr: np.ndarray,
    missing_mask: np.ndarray,
    prefix_tvt: np.ndarray,
    row_idx: np.ndarray,
    candidate_values: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    window_offsets = np.asarray(config.get("window_offsets", [-24, -12, 0, 12, 24]), np.int32)
    derivative_offsets = np.asarray(config.get("derivative_offsets", [-12, 0, 12]), np.int32)
    shift_offsets = np.asarray(config.get("shift_offsets", [-6, 0, 6]), np.int32)
    derivative_step = int(config.get("derivative_step", 3))
    gr_scale = float(config.get("gr_scale", 18.0))
    derivative_scale = float(config.get("derivative_scale", 9.0))
    energy_scale = float(config.get("energy_scale", 18.0))
    shape_scale = float(config.get("shape_scale", 1.0))
    band_scale = float(config.get("band_scale", 16.0))

    candidate_values = np.nan_to_num(candidate_values, nan=float(prefix_tvt[-1]))
    n_rows, n_candidates = candidate_values.shape
    candidate_centers = _nearest_prefix_indices(
        prefix_tvt,
        candidate_values.reshape(-1),
    ).reshape(n_rows, n_candidates)

    eval_point = full_gr[row_idx]
    candidate_point = full_gr[candidate_centers]
    raw_point = np.abs(candidate_point - eval_point[:, None]).astype(np.float32)

    eval_window = _local_vectors(full_gr, row_idx, window_offsets)
    candidate_window = _local_vectors(full_gr, candidate_centers, window_offsets)
    eval_norm = _standardize_last_axis(eval_window)
    candidate_norm = _standardize_last_axis(candidate_window)
    ncc = np.mean(candidate_norm * eval_norm[:, None, :], axis=2)
    window_mae = np.mean(np.abs(candidate_window - eval_window[:, None, :]), axis=2)

    eval_derivative = _local_derivatives(
        full_gr,
        row_idx,
        derivative_offsets,
        step=derivative_step,
    )
    candidate_derivative = _local_derivatives(
        full_gr,
        candidate_centers,
        derivative_offsets,
        step=derivative_step,
    )
    derivative_mae = np.mean(
        np.abs(candidate_derivative - eval_derivative[:, None, :]),
        axis=2,
    )

    eval_curvature = np.diff(eval_derivative, n=2, axis=-1)
    candidate_curvature = np.diff(candidate_derivative, n=2, axis=-1)
    if eval_curvature.shape[-1] == 0:
        curvature_mae = np.zeros((n_rows, n_candidates), dtype=np.float32)
    else:
        curvature_mae = np.mean(
            np.abs(candidate_curvature - eval_curvature[:, None, :]),
            axis=2,
        )

    eval_energy = np.sqrt(np.mean(np.square(eval_derivative), axis=1))
    candidate_energy = np.sqrt(np.mean(np.square(candidate_derivative), axis=2))
    energy_abs = np.abs(candidate_energy - eval_energy[:, None])

    eval_peak_count = _peak_count_proxy(eval_window)
    candidate_peak_count = _peak_count_proxy(candidate_window)
    peak_count_abs = np.abs(candidate_peak_count - eval_peak_count[:, None])

    eval_missing = _local_vectors(missing_mask.astype(np.float32), row_idx, window_offsets).mean(
        axis=1,
    )
    candidate_missing = _local_vectors(
        missing_mask.astype(np.float32),
        candidate_centers,
        window_offsets,
    ).mean(axis=2)
    missing_gap_abs = np.abs(candidate_missing - eval_missing[:, None])

    banded_shift = _shifted_window_mae(
        full_gr,
        row_idx,
        candidate_centers,
        window_offsets,
        shift_offsets,
    )
    local_shape = np.mean(np.abs(candidate_norm - eval_norm[:, None, :]), axis=2)

    shape_distance = (
        0.40 * local_shape
        + 0.20 * np.clip(derivative_mae / max(derivative_scale, 1e-6), 0.0, 5.0)
        + 0.15 * np.clip(curvature_mae / max(derivative_scale, 1e-6), 0.0, 5.0)
        + 0.15 * np.clip(energy_abs / max(energy_scale, 1e-6), 0.0, 5.0)
        + 0.05 * np.clip(peak_count_abs / 4.0, 0.0, 2.0)
        + 0.05 * np.clip(missing_gap_abs, 0.0, 1.0)
    ).astype(np.float32)
    combo_cost = (
        0.20 * np.clip(raw_point / max(gr_scale, 1e-6), 0.0, 5.0)
        + 0.25 * np.clip(window_mae / max(gr_scale, 1e-6), 0.0, 5.0)
        + 0.25 * np.clip(banded_shift / max(band_scale, 1e-6), 0.0, 5.0)
        + 0.30 * np.clip(shape_distance / max(shape_scale, 1e-6), 0.0, 5.0)
    ).astype(np.float32)

    return {
        "raw_point_abs": raw_point,
        "window_mae": window_mae.astype(np.float32),
        "window_ncc": ncc.astype(np.float32),
        "banded_shift_mae": banded_shift,
        "derivative_mae": derivative_mae.astype(np.float32),
        "curvature_mae": curvature_mae.astype(np.float32),
        "energy_abs": energy_abs.astype(np.float32),
        "peak_count_abs": peak_count_abs.astype(np.float32),
        "missing_gap_abs": missing_gap_abs.astype(np.float32),
        "shape_distance": shape_distance,
        "combo_cost": combo_cost,
        "raw_point_real": np.exp(-raw_point / max(gr_scale, 1e-6)).astype(np.float32),
        "ncc_window_real": np.clip((ncc + 1.0) / 2.0, 0.0, 1.0).astype(np.float32),
        "banded_shift_real": np.exp(-banded_shift / max(band_scale, 1e-6)).astype(np.float32),
        "shape_descriptor_real": np.exp(
            -shape_distance / max(shape_scale, 1e-6),
        ).astype(np.float32),
        "combo_descriptor_real": np.exp(-combo_cost).astype(np.float32),
    }


def _constant_score(shape: tuple[int, int], value: float = 0.2) -> np.ndarray:
    return np.full(shape, float(value), dtype=np.float32)


def build_descriptor_score_frame(
    frame: pd.DataFrame,
    existing_candidates: pd.DataFrame,
    *,
    train_dir: str | Path,
    candidate_names: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dir = Path(train_dir)
    gr_rolling_window = int(config.get("gr_rolling_window", 5))
    base = pd.DataFrame({"id": frame["id"].astype(str), "well": frame["well"].astype(str)})
    base["_row_idx"] = _row_indices_from_ids(base["id"])
    score_frames: list[pd.DataFrame] = []
    well_rows: list[dict[str, Any]] = []

    for well, positions in base.groupby("well", sort=False).groups.items():
        position_list = list(positions)
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        if not horizontal_path.exists():
            raise FileNotFoundError(f"raw train horizontal well file not found: {horizontal_path}")
        horizontal = pd.read_csv(horizontal_path, usecols=["GR", "TVT_input"])
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
        known_mask = tvt_input.notna().to_numpy()
        if not known_mask.any():
            raise ValueError(f"No finite TVT_input prefix rows for well {well}")
        prefix_len = int(np.flatnonzero(known_mask)[-1] + 1)
        prefix_tvt = (
            tvt_input.iloc[:prefix_len]
            .interpolate(limit_direction="both")
            .ffill()
            .bfill()
            .to_numpy(np.float32)
        )
        if not np.isfinite(prefix_tvt).all():
            raise ValueError(f"Non-finite prefix TVT after interpolation for well {well}")

        gr_series = pd.to_numeric(horizontal["GR"], errors="coerce")
        missing_mask = gr_series.isna().to_numpy()
        fallback = float(gr_series.iloc[:prefix_len].mean())
        if not np.isfinite(fallback):
            fallback = float(gr_series.mean()) if np.isfinite(float(gr_series.mean())) else 0.0
        full_gr = (
            gr_series.interpolate(limit_direction="both")
            .fillna(fallback)
            .rolling(gr_rolling_window, center=True, min_periods=1)
            .mean()
            .to_numpy(np.float32)
        )
        row_idx = base.loc[position_list, "_row_idx"].to_numpy(np.int32)
        if row_idx.min(initial=0) < 0 or row_idx.max(initial=0) >= len(horizontal):
            raise ValueError(f"row index out of range for well {well}")
        candidate_values = existing_candidates.loc[position_list, candidate_names].to_numpy(
            np.float32,
        )
        real = _descriptor_scores_for_gr(
            full_gr=full_gr,
            missing_mask=missing_mask,
            prefix_tvt=prefix_tvt,
            row_idx=row_idx,
            candidate_values=candidate_values,
            config=config,
        )
        shift = max(37, min(len(full_gr) - 1, 251))
        shuffled = _descriptor_scores_for_gr(
            full_gr=np.roll(full_gr, shift),
            missing_mask=np.roll(missing_mask, shift),
            prefix_tvt=prefix_tvt,
            row_idx=row_idx,
            candidate_values=candidate_values,
            config=config,
        )
        rows = pd.DataFrame(
            {
                "id": base.loc[position_list, "id"].to_numpy(),
                "well": str(well),
            },
        )
        score_shape = candidate_values.shape
        for i, candidate in enumerate(candidate_names):
            for metric_name in [
                "raw_point_abs",
                "window_mae",
                "window_ncc",
                "banded_shift_mae",
                "derivative_mae",
                "curvature_mae",
                "energy_abs",
                "peak_count_abs",
                "missing_gap_abs",
                "shape_distance",
                "combo_cost",
            ]:
                rows[f"{metric_name}_{candidate}"] = real[metric_name][:, i].astype(np.float32)
            for variant in REAL_SCORE_VARIANTS:
                rows[f"score_{variant}_{candidate}"] = real[variant][:, i].astype(np.float32)
            rows[f"score_combo_descriptor_shuffled_{candidate}"] = shuffled[
                "combo_descriptor_real"
            ][:, i].astype(np.float32)
            rows[f"score_no_gr_constant_{candidate}"] = _constant_score(score_shape)[:, i]

        for variant in ALL_SCORE_VARIANTS:
            score_matrix = np.column_stack(
                [
                    rows[f"score_{variant}_{candidate}"].to_numpy(np.float32)
                    for candidate in candidate_names
                ],
            )
            top_pos = score_matrix.argmax(axis=1)
            rows[f"{variant}_score_max"] = score_matrix.max(axis=1).astype(np.float32)
            rows[f"{variant}_score_mean"] = score_matrix.mean(axis=1).astype(np.float32)
            sorted_scores = np.sort(score_matrix, axis=1)
            rows[f"{variant}_score_gap"] = (
                sorted_scores[:, -1] - sorted_scores[:, -2]
                if score_matrix.shape[1] > 1
                else sorted_scores[:, -1]
            ).astype(np.float32)
            rows[f"{variant}_top1_source_id"] = top_pos.astype(np.float32)
            rows[f"{variant}_top1_tvt"] = candidate_values[
                np.arange(len(row_idx)),
                top_pos,
            ].astype(np.float32)
        score_frames.append(rows)
        well_rows.append(
            {
                "well": str(well),
                "rows": int(len(row_idx)),
                "known_prefix_rows": int(prefix_len),
                "eval_len": int(max(0, len(horizontal) - prefix_len)),
                "gr_missing_rate": float(missing_mask.mean()),
                "combo_real_mean": float(
                    np.mean(real["combo_descriptor_real"]),
                ),
                "combo_shuffled_mean": float(
                    np.mean(shuffled["combo_descriptor_real"]),
                ),
                "combo_real_minus_shuffled": float(
                    np.mean(real["combo_descriptor_real"])
                    - np.mean(shuffled["combo_descriptor_real"]),
                ),
            },
        )
    out = pd.concat(score_frames, ignore_index=True)
    if out.drop(columns=["id", "well"]).isna().any().any():
        bad = out.columns[out.isna().any()].tolist()[:20]
        raise ValueError(f"descriptor score frame contains missing values: {bad}")
    return out, pd.DataFrame(well_rows)


def _candidate_score(long_frame: pd.DataFrame, variant: str) -> np.ndarray:
    column = f"score_{variant}"
    if column not in long_frame.columns:
        raise ValueError(f"candidate long frame missing score column: {column}")
    return np.clip(numeric_array(long_frame, column, default=0.0), 1e-6, 1.0 - 1e-6)


def build_candidate_long_frame(
    base_frame: pd.DataFrame,
    candidate_columns: list[str],
    score_variants: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    true_tvt = numeric_array(base_frame, "true_tvt")
    for candidate in candidate_columns:
        if candidate not in base_frame.columns:
            continue
        pred = numeric_array(base_frame, candidate)
        item = pd.DataFrame(
            {
                "id": base_frame["id"].to_numpy(),
                "well": base_frame["well"].to_numpy(),
                "candidate": candidate,
                "pred_tvt": pred,
                "target_tvt": true_tvt,
                "abs_error": np.abs(pred - true_tvt).astype(np.float32),
                "candidate_family": "existing",
            },
        )
        for variant in score_variants:
            score_col = f"score_{variant}_{candidate}"
            if score_col in base_frame.columns:
                item[f"score_{variant}"] = np.clip(
                    numeric_array(base_frame, score_col, default=0.0),
                    1e-6,
                    1.0 - 1e-6,
                )
            else:
                item[f"score_{variant}"] = np.full(len(base_frame), 1e-6, dtype=np.float32)
        rows.append(item)
    long_frame = pd.concat(rows, ignore_index=True)
    numeric_columns = [
        "pred_tvt",
        "target_tvt",
        "abs_error",
        *[f"score_{variant}" for variant in score_variants],
    ]
    if not np.isfinite(long_frame[numeric_columns].to_numpy()).all():
        raise ValueError("candidate long frame contains non-finite numeric values")
    return long_frame


def summarize_candidate_metrics(long_frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, group in long_frame.groupby("candidate", sort=False):
        error = group["pred_tvt"].to_numpy(np.float32) - group["target_tvt"].to_numpy(np.float32)
        abs_error = np.abs(error)
        row: dict[str, Any] = {
            "candidate": str(candidate),
            "rows": int(len(group)),
            "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
            "mae_tvt": float(np.mean(abs_error)),
            "bias_tvt": float(np.mean(error)),
            "abs_error_p50": float(np.quantile(abs_error, 0.50)),
            "abs_error_p90": float(np.quantile(abs_error, 0.90)),
            "abs_error_p95": float(np.quantile(abs_error, 0.95)),
        }
        for threshold in thresholds:
            row[f"within_{threshold:g}ft"] = float(np.mean(abs_error <= float(threshold)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse_tvt").reset_index(drop=True)


def summarize_score_variant_metrics(
    long_frame: pd.DataFrame,
    score_variants: list[str],
    *,
    positive_threshold_ft: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    label = (long_frame["abs_error"].to_numpy(np.float32) <= positive_threshold_ft).astype(
        np.int32,
    )
    for variant in score_variants:
        score = _candidate_score(long_frame, variant)
        if len(np.unique(label)) == 2:
            auc = float(roc_auc_score(label, score))
            loss = float(log_loss(label, score, labels=[0, 1]))
        else:
            auc = np.nan
            loss = np.nan
        rows.append(
            {
                "score_variant": variant,
                "rows": int(len(long_frame)),
                "positive_threshold_ft": float(positive_threshold_ft),
                "positive_rate": float(label.mean()),
                "auc_within_threshold": auc,
                "logloss_within_threshold": loss,
                "score_mean": float(np.mean(score)),
                "score_p10": float(np.quantile(score, 0.10)),
                "score_p50": float(np.quantile(score, 0.50)),
                "score_p90": float(np.quantile(score, 0.90)),
            },
        )
    return pd.DataFrame(rows).sort_values(
        ["auc_within_threshold", "logloss_within_threshold"],
        ascending=[False, True],
    )


def summarize_rank_metrics(
    long_frame: pd.DataFrame,
    thresholds: list[float],
    topk_values: list[int],
    candidate_sets: dict[str, list[str]],
    score_variants: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_set, candidates in candidate_sets.items():
        subset_frame = long_frame[long_frame["candidate"].isin(candidates)].copy()
        candidate_count = int(subset_frame["candidate"].nunique())
        if candidate_count == 0:
            continue
        sorted_oracle = subset_frame.sort_values(["id", "abs_error"])
        for topk in topk_values:
            k = min(int(topk), candidate_count)
            subset = sorted_oracle.groupby("id", sort=False).head(k)
            best = subset.sort_values(["id", "abs_error"]).groupby("id", sort=False).head(1)
            rows.append(
                _rank_row(
                    best,
                    candidate_set=candidate_set,
                    rank_family="oracle_best_error",
                    score_variant="oracle",
                    topk=k,
                    candidate_count=candidate_count,
                    thresholds=thresholds,
                ),
            )
        for variant in score_variants:
            sorted_score = subset_frame.sort_values(
                ["id", f"score_{variant}"],
                ascending=[True, False],
            )
            for topk in topk_values:
                k = min(int(topk), candidate_count)
                subset = sorted_score.groupby("id", sort=False).head(k)
                best = subset.sort_values(["id", "abs_error"]).groupby("id", sort=False).head(1)
                rows.append(
                    _rank_row(
                        best,
                        candidate_set=candidate_set,
                        rank_family="candidate_rank_score",
                        score_variant=variant,
                        topk=k,
                        candidate_count=candidate_count,
                        thresholds=thresholds,
                    ),
                )
    return pd.DataFrame(rows)


def _rank_row(
    frame: pd.DataFrame,
    *,
    candidate_set: str,
    rank_family: str,
    score_variant: str,
    topk: int,
    candidate_count: int,
    thresholds: list[float],
) -> dict[str, Any]:
    error = frame["pred_tvt"].to_numpy(np.float32) - frame["target_tvt"].to_numpy(np.float32)
    abs_error = np.abs(error)
    row: dict[str, Any] = {
        "candidate_set": candidate_set,
        "rank_family": rank_family,
        "score_variant": score_variant,
        "topk": int(topk),
        "candidate_count": int(candidate_count),
        "rows": int(len(frame)),
        "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
        "mae_tvt": float(np.mean(abs_error)),
        "selected_candidate_top": str(frame["candidate"].mode().iloc[0]),
    }
    for threshold in thresholds:
        row[f"within_{threshold:g}ft"] = float(np.mean(abs_error <= float(threshold)))
    return row


def build_row_context(source_frame: pd.DataFrame, full_frame: pd.DataFrame) -> pd.DataFrame:
    context = source_frame[["id", "well"]].copy()
    context["distance_bucket"] = _distance_bucket(source_frame.get("md_since", np.nan))
    context["tail_rank_bucket"] = _tail_rank_bucket(source_frame["id"])
    for source_column, bucket_name in [
        ("eval_len", "eval_len_bucket"),
        ("pf_ancc_std", "pf_seed_std_bucket"),
        ("likpf_mean_d", "likpf_delta_bucket"),
    ]:
        if source_column in source_frame.columns:
            context[bucket_name] = _quantile_bucket(source_frame[source_column], bucket_name)
        else:
            context[bucket_name] = pd.Categorical([f"{bucket_name}_unknown"] * len(context))
    context["combo_real_score_bucket"] = _quantile_bucket(
        full_frame["combo_descriptor_real_score_max"],
        "combo_real_score",
    )
    return context


def summarize_bucket_metrics(
    long_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    thresholds: list[float],
    score_variants: list[str],
) -> pd.DataFrame:
    frame = long_frame.merge(context_frame, on=["id", "well"], how="left", validate="many_to_one")
    bucket_families = [
        "distance_bucket",
        "tail_rank_bucket",
        "eval_len_bucket",
        "pf_seed_std_bucket",
        "likpf_delta_bucket",
        "combo_real_score_bucket",
    ]
    rows: list[dict[str, Any]] = []
    for bucket_family in bucket_families:
        for variant in score_variants:
            selected = (
                frame.sort_values(["id", f"score_{variant}"], ascending=[True, False])
                .groupby("id", sort=False)
                .head(1)
            )
            for bucket, group in selected.groupby(bucket_family, observed=True):
                error = group["pred_tvt"].to_numpy(np.float32) - group["target_tvt"].to_numpy(
                    np.float32,
                )
                abs_error = np.abs(error)
                row: dict[str, Any] = {
                    "score_variant": variant,
                    "bucket_family": bucket_family,
                    "bucket": str(bucket),
                    "rows": int(len(group)),
                    "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
                    "mae_tvt": float(np.mean(abs_error)),
                }
                for threshold in thresholds:
                    row[f"miss_gt_{threshold:g}ft"] = float(np.mean(abs_error > threshold))
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_well(
    long_frame: pd.DataFrame,
    thresholds: list[float],
    score_variants: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in score_variants:
        selected = (
            long_frame.sort_values(["id", f"score_{variant}"], ascending=[True, False])
            .groupby("id", sort=False)
            .head(1)
        )
        for well, group in selected.groupby("well", sort=False):
            error = group["pred_tvt"].to_numpy(np.float32) - group["target_tvt"].to_numpy(
                np.float32,
            )
            abs_error = np.abs(error)
            row: dict[str, Any] = {
                "score_variant": variant,
                "well": str(well),
                "rows": int(len(group)),
                "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
                "mae_tvt": float(np.mean(abs_error)),
                "selected_candidate_top": str(group["candidate"].mode().iloc[0]),
            }
            for threshold in thresholds:
                row[f"within_{threshold:g}ft"] = float(np.mean(abs_error <= threshold))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["score_variant", "rmse_tvt"], ascending=[True, False])


def candidate_sets_from_config(config: dict[str, Any]) -> dict[str, list[str]]:
    configured = get_nested(config, "audit.candidate_sets") or []
    out: dict[str, list[str]] = {}
    for item in configured:
        out[str(item["name"])] = [str(candidate) for candidate in item.get("candidates", [])]
    if not out:
        raise ValueError("audit.candidate_sets must configure at least one candidate set")
    return out


def score_variants_from_config(config: dict[str, Any]) -> list[str]:
    variants = [str(value) for value in get_nested(config, "audit.score_variants") or []]
    if not variants:
        return list(ALL_SCORE_VARIANTS)
    missing = sorted(set(variants) - set(ALL_SCORE_VARIANTS))
    if missing:
        raise ValueError(f"Unsupported score variants: {missing}")
    return variants


def summarize_probe_decision(
    score_metrics: pd.DataFrame,
    rank_metrics: pd.DataFrame,
    *,
    primary_threshold_ft: float,
) -> dict[str, Any]:
    score_by_variant = {
        str(row["score_variant"]): to_jsonable(row.to_dict()) for _, row in score_metrics.iterrows()
    }
    top1 = rank_metrics[
        (rank_metrics["rank_family"] == "candidate_rank_score") & (rank_metrics["topk"] == 1)
    ]
    top1_by_variant = {
        str(row["score_variant"]): to_jsonable(row.to_dict()) for _, row in top1.iterrows()
    }
    oracle = rank_metrics[
        (rank_metrics["rank_family"] == "oracle_best_error")
        & (rank_metrics["topk"] == rank_metrics["candidate_count"])
    ]
    oracle_row = oracle.iloc[0].to_dict() if len(oracle) else None
    real_auc = float(
        score_by_variant.get("combo_descriptor_real", {}).get("auc_within_threshold") or 0.0,
    )
    shuffled_auc = float(
        score_by_variant.get("combo_descriptor_shuffled", {}).get("auc_within_threshold") or 0.0,
    )
    no_gr_auc = float(score_by_variant.get("no_gr_constant", {}).get("auc_within_threshold") or 0.0)
    real_top1 = top1_by_variant.get("combo_descriptor_real")
    shuffled_top1 = top1_by_variant.get("combo_descriptor_shuffled")
    real_beats_controls = real_auc > max(shuffled_auc, no_gr_auc) + 0.01
    top1_supported = False
    if real_top1 and shuffled_top1:
        top1_supported = float(real_top1["rmse_tvt"]) < float(shuffled_top1["rmse_tvt"])
    recommendation = (
        "shape_descriptor_supported_for_likelihood_features"
        if real_beats_controls and top1_supported
        else "do_not_use_shape_descriptor_without_stronger_verifier"
    )
    return {
        "primary_threshold_ft": float(primary_threshold_ft),
        "baseline_oracle": to_jsonable(oracle_row) if oracle_row else None,
        "score_metrics_by_variant": score_by_variant,
        "rank_score_top1_by_variant": top1_by_variant,
        "combo_real_auc_minus_shuffled": real_auc - shuffled_auc,
        "combo_real_auc_minus_no_gr": real_auc - no_gr_auc,
        "combo_real_beats_negative_controls": bool(real_beats_controls),
        "combo_real_top1_beats_shuffled": bool(top1_supported),
        "recommendation": recommendation,
    }


def write_train_feature_cache(
    *,
    output_dir: Path,
    source_frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    candidate_columns: list[str],
    score_variants: list[str],
) -> dict[str, Any]:
    meta_columns = ["id", "well", "target"]
    train_frame = source_frame[meta_columns].copy()
    feature_columns: list[str] = []
    for column in source_frame.columns:
        if column not in {"id", "well", "target"}:
            feature_columns.append(column)
    for column in candidate_columns:
        if column not in {"id", "well", "target", "true_tvt"}:
            feature_columns.append(column)
    for column in full_frame.columns:
        if (
            column.endswith(tuple(candidate_columns))
            or any(column.startswith(f"{variant}_") for variant in score_variants)
            or column.startswith("score_")
        ) and column not in {"id", "well", "target", "true_tvt"}:
            feature_columns.append(column)
    feature_columns = list(dict.fromkeys(feature_columns))
    for column in feature_columns:
        values = full_frame[column] if column in full_frame.columns else source_frame[column]
        train_frame[column] = pd.to_numeric(values, errors="coerce").astype(np.float32)
    if not np.isfinite(train_frame[["target", *feature_columns]].to_numpy(np.float32)).all():
        bad = [
            column
            for column in ["target", *feature_columns]
            if not np.isfinite(train_frame[column].to_numpy(np.float32)).all()
        ]
        raise ValueError(f"train feature cache contains non-finite values: {bad[:20]}")
    train_path = output_dir / TRAIN_FEATURE_CACHE_FILENAME
    schema_path = output_dir / TRAIN_FEATURE_SCHEMA_FILENAME
    train_frame.to_csv(train_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "variant": TRAIN_FEATURE_CACHE_VARIANT,
            "feature_index": np.arange(len(feature_columns), dtype=np.int32),
            "feature": feature_columns,
        },
    ).to_csv(schema_path, index=False)
    return {
        "variant": TRAIN_FEATURE_CACHE_VARIANT,
        "rows": int(len(train_frame)),
        "wells": int(train_frame["well"].nunique()),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "outputs": {
            "train_features": train_path.name,
            "train_feature_schema": schema_path.name,
        },
        "sha256": {
            "train_features": sha256_path(train_path),
            "train_features_decompressed": sha256_path(train_path, decompressed=True),
            "train_feature_schema": sha256_path(schema_path),
        },
    }


def run_gr_shape_descriptor_matching_ablation(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None,
    candidate_specs: list[CandidateSpec],
    extra_source_columns: list[str],
    descriptor_config: dict[str, Any],
    candidate_sets: dict[str, list[str]],
    score_variants: list[str],
    thresholds: list[float],
    topk_values: list[int],
    max_rows: int | None = None,
    save_candidate_long: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required_columns = build_required_columns(candidate_specs, extra_source_columns)
    source_frame, source_meta = read_feature_cache(
        cache_path,
        required_columns=required_columns,
        max_rows=max_rows,
    )
    existing = materialize_existing_candidates(source_frame, candidate_specs)
    descriptor_candidate_names = [
        str(name)
        for name in descriptor_config.get(
            "score_candidates",
            ["pf_ancc", "beam_mean", "likpf_mean", "sc_ens", "hyb"],
        )
    ]
    missing = [name for name in descriptor_candidate_names if name not in existing.columns]
    if missing:
        raise ValueError(f"descriptor score candidates missing from frame: {missing}")
    descriptor_frame, descriptor_well_summary = build_descriptor_score_frame(
        source_frame,
        existing,
        train_dir=train_dir,
        candidate_names=descriptor_candidate_names,
        config=descriptor_config,
    )
    full_frame = existing.merge(
        descriptor_frame,
        on=["id", "well"],
        how="left",
        validate="one_to_one",
    )
    if full_frame.isna().any().any():
        bad = full_frame.columns[full_frame.isna().any()].tolist()[:20]
        raise ValueError(f"candidate merge produced missing values: {bad}")

    existing_candidate_columns = [spec.name for spec in candidate_specs if spec.enabled]
    candidate_columns = [
        candidate
        for candidate in existing_candidate_columns
        if candidate in descriptor_candidate_names
    ]
    long_frame = build_candidate_long_frame(full_frame, candidate_columns, score_variants)
    context_frame = build_row_context(source_frame, full_frame)
    train_feature_cache = write_train_feature_cache(
        output_dir=output_dir,
        source_frame=source_frame,
        full_frame=full_frame,
        candidate_columns=candidate_columns,
        score_variants=score_variants,
    )

    primary_threshold = float(descriptor_config.get("primary_threshold_ft", 10.0))
    candidate_metrics = summarize_candidate_metrics(long_frame, thresholds)
    score_metrics = summarize_score_variant_metrics(
        long_frame,
        score_variants,
        positive_threshold_ft=primary_threshold,
    )
    rank_metrics = summarize_rank_metrics(
        long_frame,
        thresholds,
        topk_values,
        candidate_sets,
        score_variants,
    )
    bucket_metrics = summarize_bucket_metrics(long_frame, context_frame, thresholds, score_variants)
    by_well = summarize_by_well(long_frame, thresholds, score_variants)
    decision = summarize_probe_decision(
        score_metrics,
        rank_metrics,
        primary_threshold_ft=primary_threshold,
    )

    candidate_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_candidate_metrics.csv", index=False)
    score_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_score_variant_metrics.csv", index=False)
    rank_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_rank_metrics.csv", index=False)
    bucket_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv", index=False)
    by_well.to_csv(output_dir / f"{OUTPUT_PREFIX}_by_well.csv", index=False)
    descriptor_well_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_descriptor_well_summary.csv",
        index=False,
    )
    context_frame.to_csv(output_dir / f"{OUTPUT_PREFIX}_row_context.csv.gz", index=False)
    source_schema = pd.DataFrame(
        [{"column": column, "role": "source"} for column in source_frame.columns]
        + [{"column": column, "role": "candidate"} for column in candidate_columns]
        + [
            {"column": column, "role": "descriptor_score"}
            for column in descriptor_frame.columns
            if column not in {"id", "well"}
        ],
    )
    source_schema.to_csv(output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv", index=False)
    if save_candidate_long:
        long_frame.to_csv(
            output_dir / f"{OUTPUT_PREFIX}_candidate_long.csv.gz",
            index=False,
            compression="gzip",
        )

    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "completed_train_side_audit" if max_rows is None else "debug_completed",
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "runtime_seconds": float(time.time() - t0),
        "source": source_meta,
        "descriptor_matching": {
            "config": descriptor_config,
            "score_candidates": descriptor_candidate_names,
            "score_variants": score_variants,
            "well_summary_rows": int(len(descriptor_well_summary)),
        },
        "train_feature_cache": train_feature_cache,
        "candidate_sets": candidate_sets,
        "thresholds_ft": thresholds,
        "topk_values": topk_values,
        "best_candidate_by_rmse": to_jsonable(candidate_metrics.iloc[0].to_dict()),
        "best_score_variant_by_auc": to_jsonable(score_metrics.iloc[0].to_dict()),
        "probe_decision": to_jsonable(decision),
        "prediction_sha": {
            f"{variant}_top1_tvt": prediction_sha256(
                full_frame["id"],
                full_frame[f"{variant}_top1_tvt"].to_numpy(np.float32),
                label=f"{OUTPUT_PREFIX}:{variant}_top1_tvt",
            )
            for variant in score_variants
        },
        "outputs": {
            "candidate_metrics": f"{OUTPUT_PREFIX}_candidate_metrics.csv",
            "score_variant_metrics": f"{OUTPUT_PREFIX}_score_variant_metrics.csv",
            "rank_metrics": f"{OUTPUT_PREFIX}_rank_metrics.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "descriptor_well_summary": f"{OUTPUT_PREFIX}_descriptor_well_summary.csv",
            "train_features": TRAIN_FEATURE_CACHE_FILENAME,
            "train_feature_schema": TRAIN_FEATURE_SCHEMA_FILENAME,
            "candidate_long": f"{OUTPUT_PREFIX}_candidate_long.csv.gz"
            if save_candidate_long
            else None,
            "row_context": f"{OUTPUT_PREFIX}_row_context.csv.gz",
            "feature_schema": f"{OUTPUT_PREFIX}_feature_schema.csv",
            "summary": f"{OUTPUT_PREFIX}_summary.json",
        },
    }
    with (output_dir / f"{OUTPUT_PREFIX}_summary.json").open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
    return summary


def candidate_specs_from_config(config: dict[str, Any]) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for item in get_nested(config, "audit.candidates") or []:
        specs.append(
            CandidateSpec(
                name=str(item["name"]),
                source_column=str(item.get("source_column") or item["name"]),
                transform=str(item.get("transform", "absolute")),
                role=str(item.get("role", item.get("name", "candidate"))),
                enabled=bool(item.get("enabled", True)),
            ),
        )
    if not specs:
        raise ValueError("audit.candidates must configure at least one candidate")
    return specs


def run_from_config(
    config: dict[str, Any] | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    return run_gr_shape_descriptor_matching_ablation(
        output_dir=output_dir or paths.artifacts_dir,
        train_dir=paths.train_data_dir,
        cache_path=get_nested(config, "data.exp072_train_feature_cache_local"),
        candidate_specs=candidate_specs_from_config(config),
        extra_source_columns=[
            str(col) for col in get_nested(config, "audit.extra_source_columns") or []
        ],
        descriptor_config=get_nested(config, "model.descriptor_matching") or {},
        candidate_sets=candidate_sets_from_config(config),
        score_variants=score_variants_from_config(config),
        thresholds=[
            float(value) for value in get_nested(config, "audit.thresholds_ft") or [1, 2, 5, 10]
        ],
        topk_values=[
            int(value) for value in get_nested(config, "audit.topk_values") or [1, 2, 3, 5]
        ],
        max_rows=get_nested(config, "audit.max_rows"),
        save_candidate_long=bool(get_nested(config, "audit.save_candidate_long") is not False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    config = load_config()
    if args.max_rows is not None:
        config.setdefault("audit", {})["max_rows"] = args.max_rows
    summary = run_from_config(config, output_dir=args.output_dir)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

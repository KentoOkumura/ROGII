from __future__ import annotations

import argparse
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

EXP072_ARTIFACTS = Path("experiments") / "exp072_exp063_full_replay_feature_cache" / "artifacts"
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
FULL_REPLAY_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
OUTPUT_PREFIX = "exp132_multi_scale_gr_observation_likelihood"
TRAIN_FEATURE_CACHE_VARIANT = "multi_scale_gr_observation_likelihood"
TRAIN_FEATURE_CACHE_FILENAME = (
    f"{OUTPUT_PREFIX}_{TRAIN_FEATURE_CACHE_VARIANT}_train_features.csv.gz"
)
TRAIN_FEATURE_SCHEMA_FILENAME = (
    f"{OUTPUT_PREFIX}_{TRAIN_FEATURE_CACHE_VARIANT}_feature_schema.csv"
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    source_column: str
    transform: str
    role: str
    enabled: bool = True


@dataclass(frozen=True)
class ScaleSeries:
    window: int
    smooth: np.ndarray
    zscore: np.ndarray
    derivative: np.ndarray
    energy: np.ndarray


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
    if frame[["id", "well"]].isna().any().any():
        raise ValueError("feature cache contains missing id/well values")
    schema_path: Path | None = None
    try:
        schema_path = find_artifact(FULL_REPLAY_FEATURE_SCHEMA)
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


def _standardize_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True) + 1e-6
    return centered / scale


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(int(window), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(int(window), center=True, min_periods=1)
        .std()
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _build_scale_series(full_gr: np.ndarray, windows: list[int]) -> list[ScaleSeries]:
    out: list[ScaleSeries] = []
    for window in windows:
        smooth = _rolling_mean(full_gr, int(window))
        std = _rolling_std(full_gr, int(window))
        zscore = ((full_gr - smooth) / (std + 1e-6)).astype(np.float32)
        derivative = np.gradient(smooth).astype(np.float32)
        energy = _rolling_mean(np.abs(derivative), max(3, int(window))).astype(np.float32)
        out.append(
            ScaleSeries(
                window=int(window),
                smooth=smooth.astype(np.float32),
                zscore=zscore,
                derivative=derivative,
                energy=energy,
            )
        )
    return out


def _gather_2d(series: np.ndarray, centers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    gathered = []
    for offset in offsets:
        indices = np.clip(centers + int(offset), 0, len(series) - 1)
        gathered.append(series[indices])
    return np.stack(gathered, axis=-1).astype(np.float32)


def _combined_scale_score(
    *,
    scale_series: list[ScaleSeries],
    row_idx: np.ndarray,
    candidate_idx: np.ndarray,
    observation_offsets: np.ndarray,
    gr_scale: float,
    z_scale: float,
    derivative_scale: float,
    energy_scale: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    score_parts: list[np.ndarray] = []
    raw_mae_parts: list[np.ndarray] = []
    z_mae_parts: list[np.ndarray] = []
    derivative_mae_parts: list[np.ndarray] = []
    energy_mae_parts: list[np.ndarray] = []
    ncc_parts: list[np.ndarray] = []
    for series in scale_series:
        eval_smooth = _gather_2d(series.smooth, row_idx, observation_offsets)
        cand_smooth = _gather_2d(series.smooth, candidate_idx, observation_offsets)
        eval_z = _gather_2d(series.zscore, row_idx, observation_offsets)
        cand_z = _gather_2d(series.zscore, candidate_idx, observation_offsets)
        eval_derivative = _gather_2d(series.derivative, row_idx, observation_offsets)
        cand_derivative = _gather_2d(series.derivative, candidate_idx, observation_offsets)
        eval_energy = _gather_2d(series.energy, row_idx, observation_offsets)
        cand_energy = _gather_2d(series.energy, candidate_idx, observation_offsets)

        raw_mae = np.mean(np.abs(cand_smooth - eval_smooth[:, None, :]), axis=2)
        z_mae = np.mean(np.abs(cand_z - eval_z[:, None, :]), axis=2)
        derivative_mae = np.mean(
            np.abs(cand_derivative - eval_derivative[:, None, :]),
            axis=2,
        )
        energy_mae = np.mean(np.abs(cand_energy - eval_energy[:, None, :]), axis=2)
        eval_norm = _standardize_rows(eval_smooth)
        flat_candidate = cand_smooth.reshape(
            cand_smooth.shape[0] * cand_smooth.shape[1],
            cand_smooth.shape[2],
        )
        candidate_norm = _standardize_rows(flat_candidate).reshape(cand_smooth.shape)
        ncc = np.mean(candidate_norm * eval_norm[:, None, :], axis=2)
        ncc_score = np.clip((ncc + 1.0) / 2.0, 0.0, 1.0)
        score = (
            np.exp(-(raw_mae / max(gr_scale, 1e-6)))
            * (0.25 + 0.75 * ncc_score)
            * np.exp(-(z_mae / max(z_scale, 1e-6)))
            * np.exp(-(derivative_mae / max(derivative_scale, 1e-6)))
            * np.exp(-(energy_mae / max(energy_scale, 1e-6)))
        )
        score_parts.append(np.clip(score, 0.0, 1.0).astype(np.float32))
        raw_mae_parts.append(raw_mae.astype(np.float32))
        z_mae_parts.append(z_mae.astype(np.float32))
        derivative_mae_parts.append(derivative_mae.astype(np.float32))
        energy_mae_parts.append(energy_mae.astype(np.float32))
        ncc_parts.append(ncc.astype(np.float32))

    score_stack = np.stack(score_parts, axis=2)
    details = {
        "score_mean": score_stack.mean(axis=2).astype(np.float32),
        "score_max": score_stack.max(axis=2).astype(np.float32),
        "raw_mae": np.stack(raw_mae_parts, axis=2).mean(axis=2).astype(np.float32),
        "z_mae": np.stack(z_mae_parts, axis=2).mean(axis=2).astype(np.float32),
        "derivative_mae": np.stack(derivative_mae_parts, axis=2).mean(axis=2).astype(np.float32),
        "energy_mae": np.stack(energy_mae_parts, axis=2).mean(axis=2).astype(np.float32),
        "ncc": np.stack(ncc_parts, axis=2).mean(axis=2).astype(np.float32),
    }
    combined = (0.60 * details["score_mean"] + 0.40 * details["score_max"]).astype(np.float32)
    return np.clip(combined, 0.0, 1.0), details


def _candidate_multi_scale_scores_for_well(
    *,
    full_gr: np.ndarray,
    prefix_tvt: np.ndarray,
    row_idx: np.ndarray,
    candidate_values: np.ndarray,
    observation_offsets: np.ndarray,
    scale_windows: list[int],
    gr_scale: float,
    z_scale: float,
    derivative_scale: float,
    energy_scale: float,
    out_of_range_scale: float,
    decoy_shifts: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n_rows, n_candidates = candidate_values.shape
    candidate_values = np.nan_to_num(candidate_values, nan=float(prefix_tvt[-1]))
    nearest_prefix = _nearest_prefix_indices(prefix_tvt, candidate_values.reshape(-1)).reshape(
        n_rows,
        n_candidates,
    )
    scale_series = _build_scale_series(full_gr, scale_windows)
    score, details = _combined_scale_score(
        scale_series=scale_series,
        row_idx=row_idx,
        candidate_idx=nearest_prefix,
        observation_offsets=observation_offsets,
        gr_scale=gr_scale,
        z_scale=z_scale,
        derivative_scale=derivative_scale,
        energy_scale=energy_scale,
    )

    low = float(np.nanmin(prefix_tvt))
    high = float(np.nanmax(prefix_tvt))
    below = np.maximum(0.0, low - candidate_values)
    above = np.maximum(0.0, candidate_values - high)
    range_penalty = np.exp(-((below + above) / max(out_of_range_scale, 1e-6))).astype(np.float32)
    score = np.clip(score * range_penalty, 0.0, 1.0).astype(np.float32)

    decoy_scores = []
    for shift in decoy_shifts:
        shifted_idx = np.clip(nearest_prefix + int(shift), 0, len(full_gr) - 1)
        shifted_score, _ = _combined_scale_score(
            scale_series=scale_series,
            row_idx=row_idx,
            candidate_idx=shifted_idx,
            observation_offsets=observation_offsets,
            gr_scale=gr_scale,
            z_scale=z_scale,
            derivative_scale=derivative_scale,
            energy_scale=energy_scale,
        )
        decoy_scores.append((shifted_score * range_penalty).astype(np.float32))
    if decoy_scores:
        decoy_max = np.stack(decoy_scores, axis=2).max(axis=2)
    else:
        decoy_max = np.zeros_like(score, dtype=np.float32)
    details["decoy_max_score"] = decoy_max.astype(np.float32)
    details["decoy_gap"] = (score - decoy_max).astype(np.float32)
    details["range_penalty"] = range_penalty.astype(np.float32)
    return score, details


def build_multi_scale_candidate_frame(
    frame: pd.DataFrame,
    existing_candidates: pd.DataFrame,
    *,
    train_dir: str | Path,
    candidate_names: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dir = Path(train_dir)
    observation_offsets = np.asarray(
        [int(value) for value in config.get("observation_offsets", [-48, -24, -12, 0, 12, 24, 48])],
        dtype=np.int32,
    )
    if observation_offsets.size == 0:
        raise ValueError("multi_scale_gr_likelihood.observation_offsets must not be empty")
    scale_windows = [int(value) for value in config.get("scale_windows", [5, 11, 21])]
    gr_rolling_window = int(config.get("gr_rolling_window", 5))
    gr_scale = float(config.get("gr_scale", 18.0))
    z_scale = float(config.get("z_scale", 2.0))
    derivative_scale = float(config.get("derivative_scale", 5.0))
    energy_scale = float(config.get("energy_scale", 5.0))
    out_of_range_scale = float(config.get("out_of_range_scale", 80.0))
    softmax_temperatures = [
        float(value) for value in config.get("softmax_temperatures", [0.20, 0.35])
    ]
    blend_weights = [float(value) for value in config.get("likpf_blend_weights", [0.10, 0.25])]
    gate_configs = list(config.get("gate_configs", []))
    decoy_shifts = np.asarray(
        [int(value) for value in config.get("decoy_shifts", [-24, -18, 18, 24])],
        dtype=np.int32,
    )

    base = pd.DataFrame({"id": frame["id"].astype(str), "well": frame["well"].astype(str)})
    base["_row_idx"] = _row_indices_from_ids(base["id"])
    score_frames: list[pd.DataFrame] = []
    well_rows: list[dict[str, Any]] = []
    for well, positions in base.groupby("well", sort=False).groups.items():
        positions_list = list(positions)
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
        fallback = float(gr_series.iloc[:prefix_len].mean())
        if not np.isfinite(fallback):
            full_mean = float(gr_series.mean())
            fallback = full_mean if np.isfinite(full_mean) else 0.0
        full_gr = (
            gr_series.interpolate(limit_direction="both")
            .fillna(fallback)
            .rolling(gr_rolling_window, center=True, min_periods=1)
            .mean()
            .to_numpy(np.float32)
        )
        row_idx = base.loc[positions_list, "_row_idx"].to_numpy(np.int32)
        if row_idx.min(initial=0) < 0 or row_idx.max(initial=0) >= len(horizontal):
            raise ValueError(f"row index out of range for well {well}")
        candidate_values = (
            existing_candidates.loc[positions_list, candidate_names].to_numpy(np.float32)
        )
        score, details = _candidate_multi_scale_scores_for_well(
            full_gr=full_gr,
            prefix_tvt=prefix_tvt,
            row_idx=row_idx,
            candidate_values=candidate_values,
            observation_offsets=observation_offsets,
            scale_windows=scale_windows,
            gr_scale=gr_scale,
            z_scale=z_scale,
            derivative_scale=derivative_scale,
            energy_scale=energy_scale,
            out_of_range_scale=out_of_range_scale,
            decoy_shifts=decoy_shifts,
        )
        order = np.argsort(score, axis=1)
        best_pos = order[:, -1]
        second_pos = order[:, -2] if score.shape[1] > 1 else best_pos
        top1 = candidate_values[np.arange(len(row_idx)), best_pos]
        top2 = candidate_values[np.arange(len(row_idx)), second_pos]
        max_score = score[np.arange(len(row_idx)), best_pos]
        second_score = score[np.arange(len(row_idx)), second_pos]
        score_gap = (max_score - second_score).astype(np.float32)
        rows = pd.DataFrame(
            {
                "id": base.loc[positions_list, "id"].to_numpy(),
                "well": str(well),
                "msgr_top1": top1.astype(np.float32),
                "msgr_top2": top2.astype(np.float32),
                "msgr_score_max": max_score.astype(np.float32),
                "msgr_score_mean": score.mean(axis=1).astype(np.float32),
                "msgr_score_gap": score_gap,
                "msgr_top1_source_id": best_pos.astype(np.float32),
                "msgr_top2_source_id": second_pos.astype(np.float32),
                "msgr_top1_raw_mae": details["raw_mae"][
                    np.arange(len(row_idx)), best_pos
                ].astype(np.float32),
                "msgr_top1_z_mae": details["z_mae"][
                    np.arange(len(row_idx)), best_pos
                ].astype(np.float32),
                "msgr_top1_derivative_mae": details["derivative_mae"][
                    np.arange(len(row_idx)), best_pos
                ].astype(np.float32),
                "msgr_top1_energy_mae": details["energy_mae"][
                    np.arange(len(row_idx)), best_pos
                ].astype(np.float32),
                "msgr_top1_ncc": details["ncc"][np.arange(len(row_idx)), best_pos].astype(
                    np.float32
                ),
                "msgr_top1_decoy_gap": details["decoy_gap"][
                    np.arange(len(row_idx)), best_pos
                ].astype(np.float32),
                "msgr_expected_error_proxy": (
                    -10.0 * np.log(np.clip(max_score, 1e-6, 1.0))
                ).astype(np.float32),
                "msgr_within10_prob_proxy": np.clip(max_score, 0.0, 1.0).astype(np.float32),
                "msgr_ambiguity_proxy": np.clip(
                    1.0 - score_gap - np.maximum(0.0, details["decoy_max_score"].max(axis=1)),
                    0.0,
                    1.0,
                ).astype(np.float32),
            }
        )
        for i, name in enumerate(candidate_names):
            rows[f"msgr_score_{name}"] = score[:, i].astype(np.float32)
            rows[f"msgr_raw_mae_{name}"] = details["raw_mae"][:, i].astype(np.float32)
            rows[f"msgr_z_mae_{name}"] = details["z_mae"][:, i].astype(np.float32)
            rows[f"msgr_derivative_mae_{name}"] = details["derivative_mae"][:, i].astype(
                np.float32
            )
            rows[f"msgr_energy_mae_{name}"] = details["energy_mae"][:, i].astype(np.float32)
            rows[f"msgr_ncc_{name}"] = details["ncc"][:, i].astype(np.float32)
            rows[f"msgr_decoy_gap_{name}"] = details["decoy_gap"][:, i].astype(np.float32)
        for temp in softmax_temperatures:
            logits = score / max(temp, 1e-6)
            logits = logits - logits.max(axis=1, keepdims=True)
            weights = np.exp(logits)
            weights /= weights.sum(axis=1, keepdims=True) + 1e-9
            key = str(temp).replace(".", "p")
            rows[f"msgr_softmax_t{key}"] = (candidate_values * weights).sum(axis=1)
        if "likpf_mean" in candidate_names:
            likpf = existing_candidates.loc[positions_list, "likpf_mean"].to_numpy(np.float32)
            top1_abs_delta = np.abs(top1 - likpf).astype(np.float32)
            for weight in blend_weights:
                key = str(weight).replace(".", "p")
                rows[f"likpf_msgr_blend_w{key}"] = (
                    (1.0 - weight) * likpf + weight * top1
                ).astype(np.float32)
            for gate in gate_configs:
                margin = float(gate.get("min_score_gap", 0.05))
                min_score = float(gate.get("min_score", 0.35))
                max_delta = float(gate.get("max_abs_delta", 80.0))
                name = str(gate.get("name", f"m{margin:g}_s{min_score:g}_d{max_delta:g}"))
                fire = (
                    (score_gap >= margin)
                    & (max_score >= min_score)
                    & (top1_abs_delta <= max_delta)
                )
                rows[f"msgr_gate_{name}"] = np.where(fire, top1, likpf).astype(np.float32)
                rows[f"msgr_gate_{name}_flag"] = fire.astype(np.float32)
        score_frames.append(rows)
        well_rows.append(
            {
                "well": str(well),
                "rows": int(len(row_idx)),
                "known_prefix_rows": int(prefix_len),
                "eval_len": int(max(0, len(horizontal) - prefix_len)),
                "gr_missing_rate": float(pd.isna(horizontal["GR"]).mean()),
                "msgr_score_mean": float(np.mean(score)),
                "msgr_score_p10": float(np.quantile(score, 0.10)),
                "msgr_score_p90": float(np.quantile(score, 0.90)),
                "msgr_score_gap_mean": float(np.mean(score_gap)),
            }
        )
    out = pd.concat(score_frames, ignore_index=True)
    if out.drop(columns=["id", "well"]).isna().any().any():
        raise ValueError("multi-scale GR candidate frame contains missing values")
    return out, pd.DataFrame(well_rows)


def _candidate_score(
    frame: pd.DataFrame,
    candidate_name: str,
    *,
    msgr_columns: set[str],
) -> np.ndarray:
    n = len(frame)
    if candidate_name in msgr_columns:
        return np.clip(numeric_array(frame, "msgr_score_max", default=0.0), 0.0, 1.0)
    score_col = f"msgr_score_{candidate_name}"
    if score_col in frame.columns:
        return np.clip(numeric_array(frame, score_col, default=0.0), 0.0, 1.0)
    if candidate_name == "last_anchor_tvt":
        return np.full(n, 0.10, dtype=np.float32)
    return np.full(n, 0.25, dtype=np.float32)


def build_candidate_long_frame(
    base_frame: pd.DataFrame,
    candidate_columns: list[str],
    *,
    msgr_columns: set[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    true_tvt = numeric_array(base_frame, "true_tvt")
    for name in candidate_columns:
        if name not in base_frame.columns:
            continue
        pred = numeric_array(base_frame, name)
        score = _candidate_score(base_frame, name, msgr_columns=msgr_columns)
        family = "multi_scale_gr_likelihood" if name in msgr_columns else "existing"
        item = pd.DataFrame(
            {
                "id": base_frame["id"].to_numpy(),
                "well": base_frame["well"].to_numpy(),
                "candidate": name,
                "pred_tvt": pred,
                "target_tvt": true_tvt,
                "abs_error": np.abs(pred - true_tvt).astype(np.float32),
                "rank_score": score,
                "candidate_family": family,
            }
        )
        rows.append(item)
    long_frame = pd.concat(rows, ignore_index=True)
    if not np.isfinite(
        long_frame[["pred_tvt", "target_tvt", "abs_error", "rank_score"]].to_numpy()
    ).all():
        raise ValueError("candidate long frame contains non-finite numeric values")
    return long_frame


def write_train_feature_cache(
    *,
    output_dir: Path,
    source_frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    candidate_columns: list[str],
) -> dict[str, Any]:
    meta_columns = ["id", "well", "target"]
    missing_meta = [column for column in meta_columns if column not in source_frame.columns]
    if missing_meta:
        raise ValueError(
            f"source frame is missing train feature cache meta columns: {missing_meta}"
        )

    feature_columns: list[str] = []
    for column in source_frame.columns:
        if column not in {"id", "well", "target"}:
            feature_columns.append(column)
    for column in candidate_columns:
        if column not in {"id", "well", "target", "true_tvt"}:
            feature_columns.append(column)
    for column in full_frame.columns:
        if (
            column.startswith("msgr_")
            or column.startswith("likpf_msgr_blend_")
        ) and column not in {"id", "well", "target", "true_tvt"}:
            feature_columns.append(column)

    feature_columns = list(dict.fromkeys(feature_columns))
    if not feature_columns:
        raise ValueError("train feature cache feature column list is empty")

    train_frame = source_frame[meta_columns].copy()
    for column in feature_columns:
        if column in full_frame.columns:
            values = full_frame[column]
        elif column in source_frame.columns:
            values = source_frame[column]
        else:
            raise ValueError(f"train feature cache column is missing: {column}")
        train_frame[column] = pd.to_numeric(values, errors="coerce").astype(np.float32)

    numeric_values = train_frame[["target", *feature_columns]].to_numpy(np.float32)
    if not np.isfinite(numeric_values).all():
        bad_columns = [
            column
            for column in ["target", *feature_columns]
            if not np.isfinite(train_frame[column].to_numpy(np.float32)).all()
        ]
        raise ValueError(f"train feature cache contains non-finite values: {bad_columns[:20]}")

    train_path = output_dir / TRAIN_FEATURE_CACHE_FILENAME
    schema_path = output_dir / TRAIN_FEATURE_SCHEMA_FILENAME
    train_frame.to_csv(train_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "variant": TRAIN_FEATURE_CACHE_VARIANT,
            "feature_index": np.arange(len(feature_columns), dtype=np.int32),
            "feature": feature_columns,
        }
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


def summarize_candidate_metrics(long_frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, group in long_frame.groupby("candidate", sort=False):
        error = group["pred_tvt"].to_numpy(np.float32) - group["target_tvt"].to_numpy(np.float32)
        abs_error = np.abs(error)
        row: dict[str, Any] = {
            "candidate": str(candidate),
            "candidate_family": str(group["candidate_family"].iloc[0]),
            "rows": int(len(group)),
            "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
            "mae_tvt": float(np.mean(abs_error)),
            "bias_tvt": float(np.mean(error)),
            "abs_error_p50": float(np.quantile(abs_error, 0.50)),
            "abs_error_p90": float(np.quantile(abs_error, 0.90)),
            "abs_error_p95": float(np.quantile(abs_error, 0.95)),
            "rank_score_mean": float(group["rank_score"].mean()),
        }
        for threshold in thresholds:
            row[f"within_{threshold:g}ft"] = float(np.mean(abs_error <= float(threshold)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse_tvt").reset_index(drop=True)


def summarize_rank_metrics(
    long_frame: pd.DataFrame,
    thresholds: list[float],
    topk_values: list[int],
    candidate_sets: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_set, candidates in candidate_sets.items():
        subset_frame = long_frame[long_frame["candidate"].isin(candidates)].copy()
        candidate_count = int(subset_frame["candidate"].nunique())
        if candidate_count == 0:
            continue
        for rank_family, sorted_frame in {
            "oracle_best_error": subset_frame.sort_values(["id", "abs_error"]),
            "candidate_rank_score": subset_frame.sort_values(
                ["id", "rank_score"],
                ascending=[True, False],
            ),
        }.items():
            for topk in topk_values:
                k = min(int(topk), candidate_count)
                subset = sorted_frame.groupby("id", sort=False).head(k)
                best = subset.sort_values(["id", "abs_error"]).groupby("id", sort=False).head(1)
                error = best["pred_tvt"].to_numpy(np.float32) - best["target_tvt"].to_numpy(
                    np.float32
                )
                abs_error = np.abs(error)
                row: dict[str, Any] = {
                    "candidate_set": candidate_set,
                    "rank_family": rank_family,
                    "topk": int(k),
                    "candidate_count": int(candidate_count),
                    "rows": int(len(best)),
                    "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
                    "mae_tvt": float(np.mean(abs_error)),
                    "selected_msgr_rate": float(
                        best["candidate_family"].eq("multi_scale_gr_likelihood").mean()
                    ),
                    "selected_candidate_top": str(best["candidate"].mode().iloc[0]),
                }
                for threshold in thresholds:
                    row[f"within_{threshold:g}ft"] = float(np.mean(abs_error <= float(threshold)))
                rows.append(row)
    return pd.DataFrame(rows)


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
    context["msgr_score_bucket"] = _quantile_bucket(
        full_frame["msgr_score_max"],
        "msgr_score",
    )
    context["msgr_gap_bucket"] = _quantile_bucket(
        full_frame["msgr_score_gap"],
        "msgr_gap",
    )
    return context


def summarize_bucket_metrics(
    long_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    frame = long_frame.merge(context_frame, on=["id", "well"], how="left", validate="many_to_one")
    bucket_families = [
        "distance_bucket",
        "tail_rank_bucket",
        "eval_len_bucket",
        "pf_seed_std_bucket",
        "likpf_delta_bucket",
        "msgr_score_bucket",
        "msgr_gap_bucket",
    ]
    rows: list[dict[str, Any]] = []
    for bucket_family in bucket_families:
        for (candidate, bucket), group in frame.groupby(
            ["candidate", bucket_family],
            observed=True,
        ):
            error = group["pred_tvt"].to_numpy(np.float32) - group["target_tvt"].to_numpy(
                np.float32
            )
            abs_error = np.abs(error)
            row: dict[str, Any] = {
                "candidate": str(candidate),
                "bucket_family": bucket_family,
                "bucket": str(bucket),
                "rows": int(len(group)),
                "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
                "mae_tvt": float(np.mean(abs_error)),
            }
            for threshold in thresholds:
                row[f"miss_gt_{threshold:g}ft"] = float(np.mean(abs_error > float(threshold)))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_well(long_frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, well), group in long_frame.groupby(["candidate", "well"], sort=False):
        error = group["pred_tvt"].to_numpy(np.float32) - group["target_tvt"].to_numpy(np.float32)
        abs_error = np.abs(error)
        row: dict[str, Any] = {
            "candidate": str(candidate),
            "well": str(well),
            "rows": int(len(group)),
            "rmse_tvt": float(np.sqrt(np.mean(np.square(error)))),
            "mae_tvt": float(np.mean(abs_error)),
        }
        for threshold in thresholds:
            row[f"within_{threshold:g}ft"] = float(np.mean(abs_error <= float(threshold)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["candidate", "rmse_tvt"], ascending=[True, False])


def candidate_sets_from_config(config: dict[str, Any]) -> dict[str, list[str]]:
    configured = get_nested(config, "audit.candidate_sets") or []
    out: dict[str, list[str]] = {}
    for item in configured:
        out[str(item["name"])] = [str(candidate) for candidate in item.get("candidates", [])]
    if not out:
        raise ValueError("audit.candidate_sets must configure at least one candidate set")
    return out


def summarize_probe_decision(
    rank_metrics: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    *,
    primary_threshold_ft: float,
) -> dict:
    within_col = f"within_{primary_threshold_ft:g}ft"
    oracle = rank_metrics[
        (rank_metrics["rank_family"] == "oracle_best_error")
        & (rank_metrics["topk"] == rank_metrics["candidate_count"])
    ]
    rank_top1 = rank_metrics[
        (rank_metrics["rank_family"] == "candidate_rank_score") & (rank_metrics["topk"] == 1)
    ]
    by_set = {str(row["candidate_set"]): row for _, row in oracle.iterrows()}
    baseline = by_set.get("baseline_primary")
    expanded = by_set.get("baseline_plus_msgr")
    rmse_gain = None
    coverage_gain = None
    if baseline is not None and expanded is not None:
        rmse_gain = float(baseline["rmse_tvt"] - expanded["rmse_tvt"])
        coverage_gain = float(expanded[within_col] - baseline[within_col])
    candidate_by_name = {
        str(row["candidate"]): to_jsonable(row.to_dict())
        for _, row in candidate_metrics.iterrows()
    }
    likpf = candidate_by_name.get("likpf_mean")
    gate_rows = {
        name: row
        for name, row in candidate_by_name.items()
        if name.startswith("msgr_gate_") and not name.endswith("_flag")
    }
    best_gate = None
    if gate_rows:
        best_gate = min(gate_rows.values(), key=lambda row: float(row["rmse_tvt"]))
    best_gate_delta = None
    if likpf is not None and best_gate is not None:
        best_gate_delta = float(best_gate["rmse_tvt"]) - float(likpf["rmse_tvt"])
    return {
        "primary_threshold_ft": float(primary_threshold_ft),
        "baseline_primary_oracle": (
            to_jsonable(baseline.to_dict()) if baseline is not None else None
        ),
        "baseline_plus_msgr_oracle": (
            to_jsonable(expanded.to_dict()) if expanded is not None else None
        ),
        "oracle_rmse_gain_from_msgr": rmse_gain,
        "oracle_coverage_gain_from_msgr": coverage_gain,
        "candidate_rank_score_top1": {
            str(row["candidate_set"]): to_jsonable(row.to_dict())
            for _, row in rank_top1.iterrows()
        },
        "likpf_mean": likpf,
        "best_low_switch_gate": best_gate,
        "best_low_switch_gate_delta_vs_likpf_rmse": best_gate_delta,
        "recommendation": (
            "multi_scale_gr_likelihood_supported_for_feature_or_verifier_followup"
            if (
                (rmse_gain is not None and rmse_gain > 0.0)
                or (best_gate_delta is not None and best_gate_delta < 0.0)
            )
            else "do_not_use_multi_scale_gr_likelihood_without_better_verifier"
        ),
    }


def run_multi_scale_gr_observation_likelihood(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None,
    candidate_specs: list[CandidateSpec],
    extra_source_columns: list[str],
    likelihood_config: dict[str, Any],
    candidate_sets: dict[str, list[str]],
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
    likelihood_candidate_names = [
        str(name)
        for name in likelihood_config.get(
            "score_candidates",
            ["pf_ancc", "beam_mean", "likpf_mean", "sc_ens", "hyb"],
        )
    ]
    missing = [name for name in likelihood_candidate_names if name not in existing.columns]
    if missing:
        raise ValueError(f"multi-scale GR score candidates missing from frame: {missing}")
    msgr_frame, msgr_well_summary = build_multi_scale_candidate_frame(
        source_frame,
        existing,
        train_dir=train_dir,
        candidate_names=likelihood_candidate_names,
        config=likelihood_config,
    )
    full_frame = existing.merge(
        msgr_frame,
        on=["id", "well"],
        how="left",
        validate="one_to_one",
    )
    if full_frame.isna().any().any():
        raise ValueError("candidate merge produced missing values")

    existing_candidate_columns = [spec.name for spec in candidate_specs if spec.enabled]
    msgr_candidate_columns = [
        column
        for column in full_frame.columns
        if column.startswith("msgr_softmax_")
        or column.startswith("likpf_msgr_blend_")
        or column.startswith("msgr_gate_")
        or column in {"msgr_top1", "msgr_top2"}
    ]
    msgr_candidate_columns = [
        column for column in msgr_candidate_columns if not column.endswith("_flag")
    ]
    candidate_columns = existing_candidate_columns + msgr_candidate_columns
    long_frame = build_candidate_long_frame(
        full_frame,
        candidate_columns,
        msgr_columns=set(msgr_candidate_columns),
    )
    context_frame = build_row_context(source_frame, full_frame)
    train_feature_cache = write_train_feature_cache(
        output_dir=output_dir,
        source_frame=source_frame,
        full_frame=full_frame,
        candidate_columns=candidate_columns,
    )

    candidate_metrics = summarize_candidate_metrics(long_frame, thresholds)
    rank_metrics = summarize_rank_metrics(long_frame, thresholds, topk_values, candidate_sets)
    bucket_metrics = summarize_bucket_metrics(long_frame, context_frame, thresholds)
    by_well = summarize_by_well(long_frame, thresholds)
    decision = summarize_probe_decision(
        rank_metrics,
        candidate_metrics,
        primary_threshold_ft=float(likelihood_config.get("primary_threshold_ft", 10.0)),
    )

    candidate_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_candidate_metrics.csv", index=False)
    rank_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_rank_metrics.csv", index=False)
    bucket_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv", index=False)
    by_well.to_csv(output_dir / f"{OUTPUT_PREFIX}_by_well.csv", index=False)
    msgr_well_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_msgr_well_summary.csv",
        index=False,
    )
    context_frame.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_row_context.csv.gz",
        index=False,
        compression="gzip",
    )
    source_schema = pd.DataFrame(
        [{"column": column, "role": "source"} for column in source_frame.columns]
        + [{"column": column, "role": "candidate"} for column in candidate_columns]
        + [
            {"column": column, "role": "multi_scale_gr_likelihood"}
            for column in msgr_frame.columns
            if column not in {"id", "well"}
        ]
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
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - t0),
        "source": source_meta,
        "multi_scale_gr_likelihood": {
            "config": likelihood_config,
            "score_candidates": likelihood_candidate_names,
            "generated_candidates": msgr_candidate_columns,
            "well_summary_rows": int(len(msgr_well_summary)),
        },
        "train_feature_cache": train_feature_cache,
        "candidate_sets": candidate_sets,
        "thresholds_ft": thresholds,
        "topk_values": topk_values,
        "best_candidate_by_rmse": to_jsonable(candidate_metrics.iloc[0].to_dict()),
        "probe_decision": to_jsonable(decision),
        "outputs": {
            "candidate_metrics": f"{OUTPUT_PREFIX}_candidate_metrics.csv",
            "rank_metrics": f"{OUTPUT_PREFIX}_rank_metrics.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "msgr_well_summary": f"{OUTPUT_PREFIX}_msgr_well_summary.csv",
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
            )
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
    return run_multi_scale_gr_observation_likelihood(
        output_dir=output_dir or paths.artifacts_dir,
        train_dir=paths.train_data_dir,
        cache_path=get_nested(config, "data.exp072_train_feature_cache_local"),
        candidate_specs=candidate_specs_from_config(config),
        extra_source_columns=[
            str(col) for col in get_nested(config, "audit.extra_source_columns") or []
        ],
        likelihood_config=get_nested(config, "model.multi_scale_gr_likelihood") or {},
        candidate_sets=candidate_sets_from_config(config),
        thresholds=[
            float(value) for value in get_nested(config, "audit.thresholds_ft") or [1, 2, 5, 10]
        ],
        topk_values=[
            int(value) for value in get_nested(config, "audit.topk_values") or [1, 2, 3, 5, 10]
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

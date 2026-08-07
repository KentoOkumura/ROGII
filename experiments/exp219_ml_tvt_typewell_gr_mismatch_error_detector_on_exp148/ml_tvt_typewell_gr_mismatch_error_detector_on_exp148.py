from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR if (EXP_DIR / "project.yml").exists() else EXP_DIR.parents[1]
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PREFIX = "exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148"


def load_config() -> dict[str, Any]:
    with (EXP_DIR / "config.yaml").open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def resolve_first_existing(candidates: list[str] | tuple[str, ...]) -> Path:
    checked: list[str] = []
    for raw in candidates:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        checked.append(str(path))
        if path.exists() and path.stat().st_size > 0:
            return path
        if KAGGLE_INPUT_ROOT.exists():
            matches = sorted(
                candidate
                for candidate in KAGGLE_INPUT_ROOT.rglob(Path(raw).name)
                if candidate.is_file() and candidate.stat().st_size > 0
            )
            if matches:
                return matches[0]
    raise FileNotFoundError("No existing non-empty path among: " + ", ".join(checked))


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_gzip_decompressed(path: Path) -> str:
    hasher = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_dataframe_content(df: pd.DataFrame, columns: list[str]) -> str:
    hasher = hashlib.sha256()
    for column in columns:
        hasher.update(str(column).encode("utf-8"))
        hasher.update(b"\0")
    csv_bytes = df[columns].to_csv(index=False, lineterminator="\n").encode("utf-8")
    hasher.update(csv_bytes)
    return hasher.hexdigest()


def write_csv(df: pd.DataFrame, path: Path, *, gzip_output: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_output:
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _stable_int(*parts: str, modulo: int | None = None) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    value = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    return value % int(modulo) if modulo else value


def _fill_numeric_array(series: pd.Series, fallback: float) -> np.ndarray:
    return (
        pd.to_numeric(series, errors="coerce")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .fillna(float(fallback))
        .to_numpy(np.float32)
    )


def _rolling_mean_array(values: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    if window <= 1:
        return np.asarray(values, dtype=np.float32)
    return (
        pd.Series(np.asarray(values, dtype=np.float32))
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )


def _nearest_indices(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_values, values, side="left")
    left = np.clip(positions - 1, 0, len(sorted_values) - 1)
    right = np.clip(positions, 0, len(sorted_values) - 1)
    choose_right = np.abs(sorted_values[right] - values) < np.abs(sorted_values[left] - values)
    return np.where(choose_right, right, left).astype(np.int32)


def _gather_windows(values: np.ndarray, centers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    idx = np.clip(centers[:, None] + offsets.astype(np.int32), 0, len(values) - 1)
    return values[idx].astype(np.float32)


def _standardize_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True) + 1e-6
    return centered / scale


def _safe_div(num: float, den: float) -> float:
    if not np.isfinite(den) or den == 0:
        return float("nan")
    return float(num / den)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(diff))))


def _binary_auc(y_true: np.ndarray, signal: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    x = np.asarray(signal, dtype=np.float64)
    mask = np.isfinite(x)
    y = y[mask]
    x = x[mask]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(x).rank(method="average").to_numpy(np.float64)
    pos_rank_sum = float(ranks[y].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    if not np.isfinite(std) or std < 1e-9:
        return np.zeros(len(arr), dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32)


def _softmax_entropy(scores: np.ndarray) -> np.ndarray:
    score = np.asarray(scores, dtype=np.float64)
    score = np.clip(score, 1e-12, None)
    prob = score / np.sum(score, axis=1, keepdims=True)
    entropy = -np.sum(prob * np.log(prob + 1e-12), axis=1)
    norm = math.log(score.shape[1]) if score.shape[1] > 1 else 1.0
    return (entropy / norm).astype(np.float32)


def load_exp148_predictions(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    validation = config["validation"]
    usecols = [
        "id",
        "well",
        "variant",
        "mode",
        "model",
        "last_known_tvt",
        "target_tvt",
        "pred_tvt",
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=750_000):
        mask = (
            chunk["model"].eq(validation["source_prediction_model"])
            & chunk["mode"].eq(validation["source_prediction_mode"])
            & chunk["variant"].eq(validation["source_prediction_variant"])
        )
        part = chunk.loc[mask].copy()
        if not part.empty:
            chunks.append(part)
    if not chunks:
        raise ValueError(f"No exp148 prediction rows matched configured filters in {path}")
    pred = pd.concat(chunks, ignore_index=True)
    pred["id"] = pred["id"].astype(str)
    pred["well"] = pred["well"].astype(str)
    pred["row_idx"] = pred["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
    pred["target_tvt"] = pd.to_numeric(pred["target_tvt"], errors="raise").astype(np.float32)
    pred["pred_tvt"] = pd.to_numeric(pred["pred_tvt"], errors="raise").astype(np.float32)
    pred["last_known_tvt"] = pd.to_numeric(
        pred["last_known_tvt"],
        errors="raise",
    ).astype(np.float32)
    pred["residual"] = (pred["pred_tvt"] - pred["target_tvt"]).astype(np.float32)
    pred["abs_error"] = np.abs(pred["residual"]).astype(np.float32)
    for threshold in get_nested(config, "readout.error_thresholds_ft", [5.0, 10.0, 20.0]):
        pred[f"abs_error_gt{int(float(threshold))}"] = pred["abs_error"] > float(threshold)
    pred = pred.drop(columns=["variant", "mode", "model"])
    if pred.duplicated(["id", "well"]).any():
        raise ValueError("exp148 prediction rows contain duplicated id/well keys")
    return pred.sort_values(["well", "row_idx"]).reset_index(drop=True)


def load_optional_candidate_disagreement(config: dict[str, Any]) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    candidates = get_nested(config, "data.learned_likelihood_feature_candidates", [])
    if not candidates:
        return None, {"available": False, "reason": "no candidates configured"}
    try:
        path = resolve_first_existing(candidates)
    except FileNotFoundError as exc:
        return None, {"available": False, "reason": str(exc)}
    header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    preferred = get_nested(
        config,
        "readout.candidate_disagreement_columns",
        ["candidate_tvt_range", "candidate_tvt_std", "learned_prob_entropy"],
    )
    available = [col for col in preferred if col in header]
    if not available:
        return None, {
            "available": False,
            "path": str(path),
            "reason": "candidate disagreement columns missing",
        }
    frame = pd.read_csv(path, usecols=["id", "well", *available])
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    rename: dict[str, str] = {}
    if "candidate_tvt_range" in frame.columns:
        rename["candidate_tvt_range"] = "candidate_disagreement"
    elif "candidate_tvt_std" in frame.columns:
        rename["candidate_tvt_std"] = "candidate_disagreement"
    frame = frame.rename(columns=rename)
    if "candidate_disagreement" not in frame.columns:
        frame["candidate_disagreement"] = 0.0
    frame["candidate_disagreement"] = pd.to_numeric(
        frame["candidate_disagreement"],
        errors="coerce",
    ).fillna(0.0)
    if "learned_prob_entropy" in frame.columns:
        frame["learned_prob_entropy"] = pd.to_numeric(
            frame["learned_prob_entropy"],
            errors="coerce",
        ).fillna(0.0)
    return frame, {
        "available": True,
        "path": str(path),
        "sha256_decompressed": sha256_gzip_decompressed(path) if path.suffix == ".gz" else sha256_file(path),
        "columns": available,
    }


def _read_well_arrays(well: str, raw_dir: Path, *, smoothing_window: int) -> dict[str, Any]:
    horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
    typewell_path = raw_dir / f"{well}__typewell.csv"
    if not horizontal_path.exists():
        raise FileNotFoundError(f"raw horizontal well file not found: {horizontal_path}")
    if not typewell_path.exists():
        raise FileNotFoundError(f"raw typewell file not found: {typewell_path}")
    horizontal = pd.read_csv(horizontal_path, usecols=["MD", "Z", "GR", "TVT_input"])
    typewell = (
        pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        .sort_values("TVT")
        .reset_index(drop=True)
    )

    horizontal_gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce")
    typewell_gr_raw = pd.to_numeric(typewell["GR"], errors="coerce")
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float32)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if len(known) == 0:
        raise ValueError(f"No finite TVT_input prefix rows for well={well}")
    last_known_idx = int(known[-1])
    prefix_end = last_known_idx + 1
    horizontal_fallback = float(horizontal_gr_raw.iloc[:prefix_end].mean())
    if not np.isfinite(horizontal_fallback):
        horizontal_fallback = float(horizontal_gr_raw.mean())
    if not np.isfinite(horizontal_fallback):
        horizontal_fallback = float(typewell_gr_raw.mean())
    if not np.isfinite(horizontal_fallback):
        horizontal_fallback = 0.0
    typewell_fallback = float(typewell_gr_raw.mean())
    if not np.isfinite(typewell_fallback):
        typewell_fallback = horizontal_fallback

    horizontal_gr_filled = _fill_numeric_array(horizontal_gr_raw, horizontal_fallback)
    typewell_gr_filled = _fill_numeric_array(typewell_gr_raw, typewell_fallback)
    typewell_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float32)
    finite = np.isfinite(typewell_tvt)
    if finite.sum() < 4:
        raise ValueError(f"Too few finite typewell TVT rows for well={well}")
    if not finite.all():
        keep = np.flatnonzero(finite)
        typewell_tvt = typewell_tvt[keep]
        typewell_gr_filled = typewell_gr_filled[keep]
        typewell_gr_raw = typewell_gr_raw.iloc[keep].reset_index(drop=True)

    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float32)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float32)
    last_known_md = float(md[last_known_idx])
    return {
        "well": str(well),
        "horizontal_md": md,
        "horizontal_z": z,
        "horizontal_gr_raw": horizontal_gr_filled.astype(np.float32),
        "horizontal_gr_denoised": _rolling_mean_array(horizontal_gr_filled, smoothing_window),
        "horizontal_missing": horizontal_gr_raw.isna().to_numpy(),
        "typewell_tvt": typewell_tvt.astype(np.float32),
        "typewell_gr_raw": typewell_gr_filled.astype(np.float32),
        "typewell_gr_denoised": _rolling_mean_array(typewell_gr_filled, smoothing_window),
        "typewell_missing": typewell_gr_raw.isna().to_numpy(),
        "last_known_idx": last_known_idx,
        "last_known_md": last_known_md,
        "last_known_tvt": float(tvt_input[last_known_idx]),
        "prefix_end": prefix_end,
    }


def _score_candidate_windows(
    *,
    arrays: dict[str, Any],
    row_idx: np.ndarray,
    candidate_tvt: np.ndarray,
    window_offsets: np.ndarray,
    derivative_scale: float,
    denoised: bool,
) -> dict[str, np.ndarray]:
    h_key = "horizontal_gr_denoised" if denoised else "horizontal_gr_raw"
    t_key = "typewell_gr_denoised" if denoised else "typewell_gr_raw"
    horizontal_gr = arrays[h_key]
    typewell_gr = arrays[t_key]
    candidate_idx = _nearest_indices(arrays["typewell_tvt"], candidate_tvt.astype(np.float32))
    h_window = _gather_windows(horizontal_gr, row_idx, window_offsets)
    t_window = _gather_windows(typewell_gr, candidate_idx, window_offsets)
    h_norm = _standardize_rows(h_window)
    t_norm = _standardize_rows(t_window)
    ncc = np.mean(h_norm * t_norm, axis=1)

    h_derivative = np.gradient(horizontal_gr).astype(np.float32)
    t_derivative = np.gradient(typewell_gr).astype(np.float32)
    h_d = _gather_windows(h_derivative, row_idx, window_offsets)
    t_d = _gather_windows(t_derivative, candidate_idx, window_offsets)
    h_d_norm = _standardize_rows(h_d)
    t_d_norm = _standardize_rows(t_d)
    derivative_ncc = np.mean(h_d_norm * t_d_norm, axis=1)
    h_missing = _gather_windows(arrays["horizontal_missing"].astype(np.float32), row_idx, window_offsets)
    t_missing = _gather_windows(arrays["typewell_missing"].astype(np.float32), candidate_idx, window_offsets)

    h_center = horizontal_gr[np.clip(row_idx, 0, len(horizontal_gr) - 1)]
    t_center = typewell_gr[np.clip(candidate_idx, 0, len(typewell_gr) - 1)]
    raw_abs = np.abs(t_center - h_center)
    window_rmse = np.sqrt(np.mean(np.square(t_window - h_window), axis=1))
    z_mae = np.mean(np.abs(t_norm - h_norm), axis=1)
    derivative_mae = np.mean(np.abs(t_d - h_d), axis=1)
    missing_mean = 0.5 * h_missing.mean(axis=1) + 0.5 * t_missing.mean(axis=1)
    combo_cost = (
        0.18 * np.clip(raw_abs / 18.0, 0.0, 5.0)
        + 0.24 * np.clip(window_rmse / 18.0, 0.0, 5.0)
        + 0.22 * np.clip((1.0 - ncc) / 2.0, 0.0, 2.0)
        + 0.18 * np.clip((1.0 - derivative_ncc) / 2.0, 0.0, 2.0)
        + 0.13 * np.clip(derivative_mae / max(float(derivative_scale), 1.0), 0.0, 5.0)
        + 0.05 * np.clip(missing_mean, 0.0, 1.0)
    )
    return {
        "score": np.exp(-combo_cost).astype(np.float32),
        "window_ncc": ncc.astype(np.float32),
        "derivative_ncc": derivative_ncc.astype(np.float32),
        "window_rmse": window_rmse.astype(np.float32),
        "raw_abs": raw_abs.astype(np.float32),
        "missing_mean": missing_mean.astype(np.float32),
    }


def _score_decoy_at_ml(
    *,
    arrays: dict[str, Any],
    row_idx: np.ndarray,
    candidate_tvt: np.ndarray,
    window_offsets: np.ndarray,
    derivative_scale: float,
) -> np.ndarray:
    roll = _stable_int(
        "exp219_ml_tvt_typewell_gr_mismatch_decoy",
        str(arrays["well"]),
        modulo=max(len(arrays["typewell_gr_raw"]) - 1, 1),
    ) + 1
    decoy_arrays = dict(arrays)
    decoy_arrays["typewell_gr_raw"] = np.roll(arrays["typewell_gr_raw"], int(roll))
    decoy_arrays["typewell_missing"] = np.roll(arrays["typewell_missing"], int(roll))
    return _score_candidate_windows(
        arrays=decoy_arrays,
        row_idx=row_idx,
        candidate_tvt=candidate_tvt,
        window_offsets=window_offsets,
        derivative_scale=derivative_scale,
        denoised=False,
    )["score"]


def _local_u_mse(arrays: dict[str, Any], row_idx: np.ndarray, pred_tvt: np.ndarray, window: int) -> np.ndarray:
    row_z = arrays["horizontal_z"][np.clip(row_idx, 0, len(arrays["horizontal_z"]) - 1)]
    u_pred = pd.Series((pred_tvt.astype(np.float32) + row_z.astype(np.float32)).astype(np.float32))
    trend = u_pred.rolling(max(int(window), 3), center=True, min_periods=1).mean()
    mse = (u_pred - trend).pow(2).rolling(max(int(window), 3), center=True, min_periods=1).mean()
    return mse.fillna(0.0).to_numpy(np.float32)


def build_mismatch_features(
    pred: pd.DataFrame,
    *,
    raw_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cfg = get_nested(config, "model.ml_tvt_typewell_gr_mismatch_features", {})
    offsets_ft = np.asarray(feature_cfg.get("offsets_ft", [-50, -25, -10, 0, 10, 25, 50]), dtype=np.float32)
    if len(offsets_ft) == 0:
        raise ValueError("offsets_ft must contain at least one value")
    zero_pos = int(np.argmin(np.abs(offsets_ft)))
    half_window = int(feature_cfg.get("window_half_rows", 32))
    window_offsets = np.arange(-half_window, half_window + 1, dtype=np.int32)
    smoothing_window = int(feature_cfg.get("typewell_smooth_window", 5))
    derivative_scale = float(feature_cfg.get("derivative_scale", 3.0))
    local_u_window = int(feature_cfg.get("local_u_window_rows", 129))

    parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for well, well_pred in pred.groupby("well", sort=True):
        arrays = _read_well_arrays(str(well), raw_dir, smoothing_window=smoothing_window)
        work = well_pred.sort_values("row_idx").copy()
        row_idx = work["row_idx"].to_numpy(np.int32)
        pred_tvt = work["pred_tvt"].to_numpy(np.float32)
        candidate = pred_tvt[:, None] + offsets_ft[None, :]
        flat_rows = np.repeat(row_idx, len(offsets_ft)).astype(np.int32)
        flat_candidate = candidate.reshape(-1).astype(np.float32)

        raw_score = _score_candidate_windows(
            arrays=arrays,
            row_idx=flat_rows,
            candidate_tvt=flat_candidate,
            window_offsets=window_offsets,
            derivative_scale=derivative_scale,
            denoised=False,
        )
        den_score = _score_candidate_windows(
            arrays=arrays,
            row_idx=flat_rows,
            candidate_tvt=flat_candidate,
            window_offsets=window_offsets,
            derivative_scale=derivative_scale,
            denoised=True,
        )
        score = raw_score["score"].reshape(len(work), len(offsets_ft))
        denoised_score = den_score["score"].reshape(len(work), len(offsets_ft))
        derivative_ncc = raw_score["derivative_ncc"].reshape(len(work), len(offsets_ft))
        window_ncc = raw_score["window_ncc"].reshape(len(work), len(offsets_ft))
        window_rmse = raw_score["window_rmse"].reshape(len(work), len(offsets_ft))
        raw_abs = raw_score["raw_abs"].reshape(len(work), len(offsets_ft))
        missing_mean = raw_score["missing_mean"].reshape(len(work), len(offsets_ft))

        best_pos = score.argmax(axis=1)
        best_score = score[np.arange(len(work)), best_pos]
        score_at_ml = score[:, zero_pos]
        denoised_at_ml = denoised_score[:, zero_pos]
        best_offset = offsets_ft[best_pos].astype(np.float32)
        sorted_score = np.sort(score, axis=1)
        second_best = sorted_score[:, -2] if len(offsets_ft) > 1 else best_score
        decoy_score = _score_decoy_at_ml(
            arrays=arrays,
            row_idx=row_idx,
            candidate_tvt=pred_tvt + float(offsets_ft[zero_pos]),
            window_offsets=window_offsets,
            derivative_scale=derivative_scale,
        )
        md = arrays["horizontal_md"][np.clip(row_idx, 0, len(arrays["horizontal_md"]) - 1)]
        md_since = np.maximum(0.0, md - float(arrays["last_known_md"])).astype(np.float32)

        out = work[["id", "well", "row_idx", "target_tvt", "pred_tvt", "last_known_tvt", "residual", "abs_error"]].copy()
        out["mlgr_score_at_ml"] = score_at_ml.astype(np.float32)
        out["mlgr_best_offset_ft"] = best_offset.astype(np.float32)
        out["mlgr_abs_best_offset_ft"] = np.abs(best_offset).astype(np.float32)
        out["mlgr_best_score"] = best_score.astype(np.float32)
        out["mlgr_second_best_score"] = second_best.astype(np.float32)
        out["mlgr_score_gap"] = (best_score - score_at_ml).astype(np.float32)
        out["mlgr_best_second_gap"] = (best_score - second_best).astype(np.float32)
        out["mlgr_entropy"] = _softmax_entropy(score)
        out["mlgr_decoy_score_at_ml"] = decoy_score.astype(np.float32)
        out["mlgr_decoy_gap"] = (score_at_ml - decoy_score).astype(np.float32)
        out["mlgr_window_ncc_at_ml"] = window_ncc[:, zero_pos].astype(np.float32)
        out["mlgr_window_rmse_at_ml"] = window_rmse[:, zero_pos].astype(np.float32)
        out["mlgr_raw_abs_at_ml"] = raw_abs[:, zero_pos].astype(np.float32)
        out["mlgr_derivative_ncc_at_ml"] = derivative_ncc[:, zero_pos].astype(np.float32)
        out["mlgr_derivative_ncc_at_best"] = derivative_ncc[np.arange(len(work)), best_pos].astype(np.float32)
        out["mlgr_missing_mean_at_ml"] = missing_mean[:, zero_pos].astype(np.float32)
        out["mlgr_denoised_score_at_ml"] = denoised_at_ml.astype(np.float32)
        out["mlgr_raw_vs_denoised_score_gap"] = (score_at_ml - denoised_at_ml).astype(np.float32)
        out["mlgr_abs_raw_vs_denoised_score_gap"] = np.abs(score_at_ml - denoised_at_ml).astype(np.float32)
        out["mlgr_local_z_mse"] = _local_u_mse(arrays, row_idx, pred_tvt, local_u_window)
        out["mlgr_md_since"] = md_since.astype(np.float32)
        out["mlgr_abs_best_offset_x_md_since"] = (
            np.abs(best_offset) * np.log1p(md_since)
        ).astype(np.float32)
        for i, offset in enumerate(offsets_ft):
            token = f"{int(offset):+d}".replace("+", "p").replace("-", "m")
            out[f"mlgr_score_offset_{token}"] = score[:, i].astype(np.float32)
        summaries.append(
            {
                "well": str(well),
                "rows": int(len(out)),
                "typewell_rows": int(len(arrays["typewell_tvt"])),
                "prefix_end": int(arrays["prefix_end"]),
                "score_at_ml_mean": float(out["mlgr_score_at_ml"].mean()),
                "score_gap_mean": float(out["mlgr_score_gap"].mean()),
                "abs_best_offset_mean": float(out["mlgr_abs_best_offset_ft"].mean()),
            }
        )
        parts.append(out)
    features = pd.concat(parts, ignore_index=True)
    cand, cand_meta = load_optional_candidate_disagreement(config)
    if cand is not None:
        features = features.merge(cand, on=["id", "well"], how="left", validate="one_to_one")
        features["candidate_disagreement"] = features["candidate_disagreement"].fillna(0.0)
        if "learned_prob_entropy" in features.columns:
            features["learned_prob_entropy"] = features["learned_prob_entropy"].fillna(0.0)
    else:
        features["candidate_disagreement"] = 0.0
        features["learned_prob_entropy"] = 0.0
    features["candidate_disagreement_available"] = bool(cand_meta.get("available", False))
    features["mlgr_score_gap_x_candidate_disagreement"] = (
        features["mlgr_score_gap"].to_numpy(np.float32)
        * features["candidate_disagreement"].to_numpy(np.float32)
    ).astype(np.float32)
    features["mlgr_abs_best_offset_x_candidate_disagreement"] = (
        features["mlgr_abs_best_offset_ft"].to_numpy(np.float32)
        * features["candidate_disagreement"].to_numpy(np.float32)
    ).astype(np.float32)
    features["mlgr_mismatch_signal"] = (
        _zscore(features["mlgr_score_gap"])
        + _zscore(features["mlgr_abs_best_offset_ft"])
        + _zscore(features["mlgr_entropy"])
        + _zscore(features["mlgr_abs_raw_vs_denoised_score_gap"])
        + _zscore(features["mlgr_local_z_mse"])
        - _zscore(features["mlgr_score_at_ml"])
        - _zscore(features["mlgr_decoy_gap"])
    ).astype(np.float32)
    for threshold in get_nested(config, "readout.error_thresholds_ft", [5.0, 10.0, 20.0]):
        features[f"abs_error_gt{int(float(threshold))}"] = features["abs_error"] > float(threshold)
    summary = pd.DataFrame(summaries)
    summary["candidate_disagreement_available"] = bool(cand_meta.get("available", False))
    return features, summary


def _feature_schema(features: pd.DataFrame) -> pd.DataFrame:
    roles: list[dict[str, str]] = []
    for column in features.columns:
        if column in {"id", "well", "row_idx"}:
            role = "key"
        elif column in {"target_tvt", "pred_tvt", "last_known_tvt", "residual", "abs_error"} or column.startswith("abs_error_gt"):
            role = "readout_label"
        elif column.startswith("mlgr_") or column in {
            "candidate_disagreement",
            "candidate_disagreement_available",
            "learned_prob_entropy",
        }:
            role = "feature"
        else:
            role = "metadata"
        roles.append({"column": column, "role": role, "dtype": str(features[column].dtype)})
    return pd.DataFrame(roles)


def signal_readout(features: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = float(get_nested(config, "readout.primary_error_threshold_ft", 10.0))
    label_col = f"abs_error_gt{int(threshold)}"
    if label_col not in features.columns:
        label_col = "abs_error_gt10"
    signal_columns = [
        col
        for col in get_nested(config, "readout.signal_columns", [])
        if col in features.columns
    ]
    if not signal_columns:
        signal_columns = [
            "mlgr_mismatch_signal",
            "mlgr_score_gap",
            "mlgr_abs_best_offset_ft",
            "mlgr_entropy",
            "mlgr_score_at_ml",
            "mlgr_decoy_gap",
            "mlgr_local_z_mse",
            "mlgr_abs_raw_vs_denoised_score_gap",
            "mlgr_score_gap_x_candidate_disagreement",
        ]
    quantiles = [float(q) for q in get_nested(config, "readout.high_signal_quantiles", [0.80, 0.90, 0.95])]
    base_rate = float(features[label_col].mean())
    base_abs_error = float(features["abs_error"].mean())
    signal_rows: list[dict[str, Any]] = []
    lift_rows: list[dict[str, Any]] = []
    for column in signal_columns:
        raw_auc = _binary_auc(features[label_col].to_numpy(bool), features[column].to_numpy(np.float64))
        direction = 1.0
        best_auc = raw_auc
        if np.isfinite(raw_auc) and raw_auc < 0.5:
            direction = -1.0
            best_auc = 1.0 - raw_auc
        oriented = features[column].to_numpy(np.float64) * direction
        corr = float(pd.Series(features[column]).corr(features["abs_error"], method="spearman"))
        signal_rows.append(
            {
                "signal": column,
                "raw_auc": raw_auc,
                "best_oriented_auc": best_auc,
                "direction": direction,
                "spearman_abs_error": corr,
                "mean": float(np.nanmean(features[column].to_numpy(np.float64))),
                "std": float(np.nanstd(features[column].to_numpy(np.float64))),
            }
        )
        for quantile in quantiles:
            cutoff = float(np.nanquantile(oriented, quantile))
            mask = oriented >= cutoff
            subset = features.loc[mask]
            lift_rows.append(
                {
                    "signal": column,
                    "quantile": quantile,
                    "cutoff_oriented": cutoff,
                    "rows": int(mask.sum()),
                    "wells": int(subset["well"].nunique()),
                    "abs_error_mean": float(subset["abs_error"].mean()),
                    "abs_error_lift": _safe_div(float(subset["abs_error"].mean()), base_abs_error),
                    "error_gt_rate": float(subset[label_col].mean()),
                    "error_gt_lift": _safe_div(float(subset[label_col].mean()), base_rate),
                    "rmse_tvt": _rmse(subset["target_tvt"].to_numpy(), subset["pred_tvt"].to_numpy()),
                }
            )
    return pd.DataFrame(signal_rows), pd.DataFrame(lift_rows)


def distance_bucket_readout(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    bins = [float(x) for x in get_nested(config, "readout.distance_bucket_edges_ft", [0, 50, 250, 500, 1000, 1e12])]
    labels = ["000_050", "050_250", "250_500", "500_1000", "1000_plus"]
    out = features.copy()
    out["distance_bucket"] = pd.cut(
        out["mlgr_md_since"],
        bins=bins,
        labels=labels[: max(len(bins) - 1, 0)],
        include_lowest=True,
        right=False,
    )
    signal = out["mlgr_mismatch_signal"].to_numpy(np.float64)
    cutoff = float(np.nanquantile(signal, float(get_nested(config, "readout.primary_high_signal_quantile", 0.90))))
    out["high_mismatch"] = signal >= cutoff
    rows: list[dict[str, Any]] = []
    for bucket, group in out.groupby("distance_bucket", observed=False):
        if group.empty:
            continue
        high = group[group["high_mismatch"]]
        rows.append(
            {
                "distance_bucket": str(bucket),
                "rows": int(len(group)),
                "wells": int(group["well"].nunique()),
                "rmse_tvt": _rmse(group["target_tvt"].to_numpy(), group["pred_tvt"].to_numpy()),
                "abs_error_mean": float(group["abs_error"].mean()),
                "error_gt10_rate": float(group["abs_error_gt10"].mean()),
                "high_mismatch_rate": float(group["high_mismatch"].mean()),
                "high_rows": int(len(high)),
                "high_abs_error_mean": float(high["abs_error"].mean()) if len(high) else float("nan"),
                "high_error_gt10_rate": float(high["abs_error_gt10"].mean()) if len(high) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def by_well_readout(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cutoff = float(
        np.nanquantile(
            features["mlgr_mismatch_signal"].to_numpy(np.float64),
            float(get_nested(config, "readout.primary_high_signal_quantile", 0.90)),
        )
    )
    work = features.copy()
    work["high_mismatch"] = work["mlgr_mismatch_signal"] >= cutoff
    rows: list[dict[str, Any]] = []
    for well, group in work.groupby("well", sort=True):
        rows.append(
            {
                "well": str(well),
                "rows": int(len(group)),
                "rmse_tvt": _rmse(group["target_tvt"].to_numpy(), group["pred_tvt"].to_numpy()),
                "mae_tvt": float(group["abs_error"].mean()),
                "bias_mean": float(group["residual"].mean()),
                "abs_error_gt10_rate": float(group["abs_error_gt10"].mean()),
                "mismatch_signal_mean": float(group["mlgr_mismatch_signal"].mean()),
                "high_mismatch_rate": float(group["high_mismatch"].mean()),
                "score_gap_mean": float(group["mlgr_score_gap"].mean()),
                "abs_best_offset_mean": float(group["mlgr_abs_best_offset_ft"].mean()),
                "score_at_ml_mean": float(group["mlgr_score_at_ml"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_tvt", ascending=False).reset_index(drop=True)


def correction_diagnostics(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    correction_cfg = get_nested(config, "readout.correction_diagnostic", {})
    alphas = [float(x) for x in correction_cfg.get("alphas", [0.1, 0.25])]
    clips = [float(x) for x in correction_cfg.get("clips_ft", [2.5, 5.0, 10.0])]
    base_rmse = _rmse(features["target_tvt"].to_numpy(), features["pred_tvt"].to_numpy())
    rows: list[dict[str, Any]] = [
        {
            "candidate": "base_exp148_lgb_mean",
            "alpha": 0.0,
            "clip_ft": 0.0,
            "rmse_tvt": base_rmse,
            "delta_vs_base": 0.0,
            "mae_tvt": float(features["abs_error"].mean()),
            "within10": float((features["abs_error"] <= 10.0).mean()),
        }
    ]
    best_offset = features["mlgr_best_offset_ft"].to_numpy(np.float32)
    pred = features["pred_tvt"].to_numpy(np.float32)
    target = features["target_tvt"].to_numpy(np.float32)
    for alpha in alphas:
        for clip in clips:
            correction = np.clip(alpha * best_offset, -clip, clip)
            diag_pred = pred + correction
            abs_error = np.abs(diag_pred - target)
            rmse = _rmse(target, diag_pred)
            rows.append(
                {
                    "candidate": f"diagnostic_alpha{alpha:g}_clip{clip:g}",
                    "alpha": alpha,
                    "clip_ft": clip,
                    "rmse_tvt": rmse,
                    "delta_vs_base": rmse - base_rmse,
                    "mae_tvt": float(abs_error.mean()),
                    "within10": float((abs_error <= 10.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def subgroup_readout(features: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = get_nested(config, "data.hidden_like_fold_assignment_candidates", [])
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"available": False}
    if not candidates:
        return pd.DataFrame(rows), meta
    try:
        path = resolve_first_existing(candidates)
    except FileNotFoundError as exc:
        meta["reason"] = str(exc)
        return pd.DataFrame(rows), meta
    folds = pd.read_csv(path)
    well_col = "well_id" if "well_id" in folds.columns else "well"
    folds = folds.rename(columns={well_col: "well"})
    folds["well"] = folds["well"].astype(str)
    work = features.merge(folds, on="well", how="left")
    meta = {"available": True, "path": str(path), "sha256": sha256_file(path)}
    for role_col in [
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ]:
        if role_col not in work.columns:
            continue
        for role_name, group in work.groupby(role_col, dropna=True):
            if group.empty:
                continue
            auc = _binary_auc(group["abs_error_gt10"].to_numpy(bool), group["mlgr_mismatch_signal"].to_numpy(np.float64))
            rows.append(
                {
                    "split": role_col,
                    "role": str(role_name),
                    "rows": int(len(group)),
                    "wells": int(group["well"].nunique()),
                    "rmse_tvt": _rmse(group["target_tvt"].to_numpy(), group["pred_tvt"].to_numpy()),
                    "abs_error_mean": float(group["abs_error"].mean()),
                    "abs_error_gt10_rate": float(group["abs_error_gt10"].mean()),
                    "mismatch_signal_auc_abs_error_gt10": auc,
                    "mismatch_signal_mean": float(group["mlgr_mismatch_signal"].mean()),
                }
            )
    return pd.DataFrame(rows), meta


def run_readout(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_config() if config is None else config
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = resolve_first_existing(cfg["data"]["exp148_prediction_candidates"])
    pred = load_exp148_predictions(prediction_path, cfg)
    features, well_feature_summary = build_mismatch_features(
        pred,
        raw_dir=Path(train_dir),
        config=cfg,
    )
    signal_auc, signal_lift = signal_readout(features, cfg)
    bucket_metrics = distance_bucket_readout(features, cfg)
    by_well = by_well_readout(features, cfg)
    correction = correction_diagnostics(features, cfg)
    subgroup, subgroup_meta = subgroup_readout(features, cfg)
    schema = _feature_schema(features)

    feature_columns = schema.loc[schema["role"].eq("feature"), "column"].tolist()
    label_columns = schema.loc[schema["role"].eq("readout_label"), "column"].tolist()
    feature_path = output_dir / f"{OUTPUT_PREFIX}_features.csv.gz"
    schema_path = output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
    signal_auc_path = output_dir / f"{OUTPUT_PREFIX}_signal_auc.csv"
    signal_lift_path = output_dir / f"{OUTPUT_PREFIX}_signal_lift.csv"
    bucket_path = output_dir / f"{OUTPUT_PREFIX}_distance_bucket_metrics.csv"
    by_well_path = output_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    correction_path = output_dir / f"{OUTPUT_PREFIX}_correction_diagnostics.csv"
    subgroup_path = output_dir / f"{OUTPUT_PREFIX}_subgroup_metrics.csv"
    well_feature_summary_path = output_dir / f"{OUTPUT_PREFIX}_well_feature_summary.csv"
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"

    write_csv(features, feature_path, gzip_output=True)
    write_csv(schema, schema_path)
    write_csv(signal_auc, signal_auc_path)
    write_csv(signal_lift, signal_lift_path)
    write_csv(bucket_metrics, bucket_path)
    write_csv(by_well, by_well_path)
    write_csv(correction, correction_path)
    write_csv(subgroup, subgroup_path)
    write_csv(well_feature_summary, well_feature_summary_path)

    base_rmse = _rmse(features["target_tvt"].to_numpy(), features["pred_tvt"].to_numpy())
    primary_auc_row = signal_auc.loc[signal_auc["signal"].eq("mlgr_mismatch_signal")]
    primary_auc = (
        float(primary_auc_row["best_oriented_auc"].iloc[0])
        if len(primary_auc_row)
        else float("nan")
    )
    primary_lift = signal_lift[
        signal_lift["signal"].eq("mlgr_mismatch_signal")
        & signal_lift["quantile"].eq(float(get_nested(cfg, "readout.primary_high_signal_quantile", 0.90)))
    ]
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_readout_not_run_as_model",
        "rows": int(len(features)),
        "wells": int(features["well"].nunique()),
        "feature_columns": int(len(feature_columns)),
        "label_columns": int(len(label_columns)),
        "prediction_path": str(prediction_path),
        "prediction_decompressed_sha256": sha256_gzip_decompressed(prediction_path)
        if prediction_path.suffix == ".gz"
        else sha256_file(prediction_path),
        "feature_cache": str(feature_path),
        "feature_cache_decompressed_sha256": sha256_gzip_decompressed(feature_path),
        "feature_schema_sha256": sha256_file(schema_path),
        "base_exp148_rmse_tvt": base_rmse,
        "base_exp148_mae_tvt": float(features["abs_error"].mean()),
        "base_exp148_abs_error_gt10_rate": float(features["abs_error_gt10"].mean()),
        "primary_signal": "mlgr_mismatch_signal",
        "primary_signal_auc_abs_error_gt10": primary_auc,
        "primary_high_signal_q90": primary_lift.iloc[0].to_dict() if len(primary_lift) else {},
        "best_signal_by_auc": signal_auc.sort_values("best_oriented_auc", ascending=False)
        .head(1)
        .to_dict(orient="records"),
        "correction_best_by_rmse": correction.sort_values("rmse_tvt").head(1).to_dict(orient="records"),
        "candidate_disagreement_available": bool(features["candidate_disagreement_available"].any()),
        "subgroup_meta": subgroup_meta,
        "artifacts": {
            "features": feature_path.name,
            "feature_schema": schema_path.name,
            "signal_auc": signal_auc_path.name,
            "signal_lift": signal_lift_path.name,
            "distance_bucket_metrics": bucket_path.name,
            "by_well": by_well_path.name,
            "correction_diagnostics": correction_path.name,
            "subgroup_metrics": subgroup_path.name,
            "well_feature_summary": well_feature_summary_path.name,
            "summary": summary_path.name,
        },
        "notes": [
            "Feature source uses exp148 OOF pred_tvt, raw horizontal GR, raw typewell GR, known-prefix TVT_input/MD/Z, and optional target-free learned-likelihood candidate disagreement only.",
            "target_tvt, abs_error, and abs_error_gt* columns are written only for readout labels and must not be used as downstream features.",
            "correction diagnostics are diagnostic only; best_offset is not approved for hard correction or replacement.",
        ],
    }
    write_json(summary, summary_path)

    proceed_to_lgb_addonly = bool(
        np.isfinite(primary_auc)
        and primary_auc >= float(get_nested(cfg, "readout.addonly_gate.auc_abs_error_gt10", 0.65))
        and (
            len(primary_lift) == 0
            or float(primary_lift["error_gt_lift"].iloc[0])
            >= float(get_nested(cfg, "readout.addonly_gate.high_mismatch_error_lift", 1.5))
        )
    )
    metrics = {
        "experiment": OUTPUT_PREFIX,
        "status": "completed_readout_supported_for_addonly"
        if proceed_to_lgb_addonly
        else "completed_readout_rejected_no_addonly_no_submit",
        "route": "ml_model",
        "parent": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "kind": "no_training_oof_error_detector_readout",
        "base_exp148_rmse_tvt": base_rmse,
        "primary_signal_auc_abs_error_gt10": primary_auc,
        "primary_high_signal_q90_error_gt_lift": float(primary_lift["error_gt_lift"].iloc[0])
        if len(primary_lift)
        else float("nan"),
        "feature_cache_decompressed_sha256": summary["feature_cache_decompressed_sha256"],
        "generated_artifacts": summary["artifacts"],
        "next_gate": {
            "auc_threshold": float(get_nested(cfg, "readout.addonly_gate.auc_abs_error_gt10", 0.65)),
            "error_lift_threshold": float(get_nested(cfg, "readout.addonly_gate.high_mismatch_error_lift", 1.5)),
            "proceed_to_lgb_addonly": proceed_to_lgb_addonly,
        },
    }
    write_json(metrics, EXP_DIR / "metrics.json")
    return summary


def main() -> dict[str, Any]:
    cfg = load_config()
    output_dir = ROOT / get_nested(cfg, "data.output_dir", f"experiments/{OUTPUT_PREFIX}/artifacts")
    train_dir = ROOT / get_nested(cfg, "data.train_dir", "data/raw/train")
    return run_readout(output_dir=output_dir, train_dir=train_dir, config=cfg)


if __name__ == "__main__":
    result = main()
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))

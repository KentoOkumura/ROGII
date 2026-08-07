from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_PREFIX = "exp133_gr_bimodal_match_ambiguity_detector"
TRAIN_FEATURE_VARIANT = "gr_bimodal_ambiguity"
TRAIN_FEATURE_FILENAME = f"{OUTPUT_PREFIX}_{TRAIN_FEATURE_VARIANT}_train_features.csv.gz"
TRAIN_SCHEMA_FILENAME = f"{OUTPUT_PREFIX}_{TRAIN_FEATURE_VARIANT}_feature_schema.csv"

EXP072_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
EXP072_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
EXP073_PREDICTIONS = "exp063_full_replay_repro_guard_predictions.csv.gz"
EXP092_PREDICTIONS = "exp092_u_projection_correction_disagreement_fullrun_predictions.csv.gz"


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


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _as_paths(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, str | Path):
        return [Path(value)]
    if isinstance(value, list | tuple):
        return [Path(item) for item in value if item]
    return []


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed(path: str | Path) -> str | None:
    path = Path(path)
    if path.suffix != ".gz":
        return None
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_input_file(
    filename: str,
    configured: Any = None,
    *,
    local_roots: list[Path] | None = None,
) -> Path:
    candidates: list[Path] = []
    candidates.extend(_as_paths(configured))
    for root in local_roots or []:
        candidates.extend([root / filename, root / "artifacts" / filename])
    candidates.extend([Path.cwd() / filename, Path.cwd() / "artifacts" / filename])

    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(sorted(input_root.glob(f"**/{filename}")))

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(
        f"input file not found or empty: {filename}. Checked:\n" + "\n".join(checked[:160])
    )


def resolve_train_dir(train_dir: str | Path) -> Path:
    path = Path(train_dir)
    if path.exists():
        return path
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for candidate in [
            input_root / "competitions" / "rogii-wellbore-geology-prediction" / "train",
            input_root / "rogii-wellbore-geology-prediction" / "train",
        ]:
            if candidate.exists():
                return candidate
        for candidate in sorted(input_root.glob("**/train")):
            if any(candidate.glob("*__horizontal_well.csv")):
                return candidate
    return path


def parse_tail_rank(ids: pd.Series) -> pd.Series:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    return pd.to_numeric(extracted, errors="raise").astype("int32")


def rmse(error: pd.Series | np.ndarray) -> float:
    values = np.asarray(error, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values[finite]))))


def safe_qcut(values: pd.Series | np.ndarray, q: int, *, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    result = pd.Series("missing", index=numeric.index, dtype="object")
    if finite.nunique(dropna=True) <= 1:
        result.loc[finite.index] = f"{prefix}_single"
        return result
    try:
        cut = pd.qcut(finite, q=min(q, int(finite.nunique())), duplicates="drop")
    except ValueError:
        result.loc[finite.index] = f"{prefix}_single"
        return result
    labels = {interval: f"{prefix}_q{i + 1}" for i, interval in enumerate(cut.cat.categories)}
    result.loc[finite.index] = cut.map(labels).astype(str)
    return result


def distance_bucket(values: pd.Series | np.ndarray) -> pd.Series:
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


def read_selected_prediction(
    path: Path,
    *,
    selector_col: str,
    selector_value: str,
    pred_col: str,
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(
        path,
        usecols=["id", "well", selector_col, "target_tvt", "last_known_tvt", "pred_tvt"],
        dtype={"id": "string", "well": "string"},
        nrows=max_rows,
        low_memory=False,
    )
    selected = frame[frame[selector_col].astype(str).eq(selector_value)].copy()
    if selected.empty:
        raise ValueError(f"No rows for {selector_col}={selector_value} in {path}")
    selected["id"] = selected["id"].astype(str)
    selected["well"] = selected["well"].astype(str)
    selected[pred_col] = pd.to_numeric(selected["pred_tvt"], errors="raise").astype("float32")
    selected["target_tvt"] = pd.to_numeric(selected["target_tvt"], errors="raise").astype(
        "float32"
    )
    selected["last_known_tvt"] = pd.to_numeric(
        selected["last_known_tvt"], errors="raise"
    ).astype("float32")
    out = selected[["id", "well", "target_tvt", "last_known_tvt", pred_col]].reset_index(
        drop=True
    )
    return out, {
        "path": str(path),
        "raw_file_sha256": sha256_file(path),
        "decompressed_content_sha256": sha256_decompressed(path),
        "selector_col": selector_col,
        "selector_value": selector_value,
        "rows": int(len(out)),
        "wells": int(out["well"].nunique()),
    }


def load_prediction_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    max_rows = get_nested(config, "audit.max_prediction_rows")
    max_rows = None if max_rows is None else int(max_rows)
    exp073_path = find_input_file(
        EXP073_PREDICTIONS,
        get_nested(config, "data.exp073_predictions"),
        local_roots=[
            Path("/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_v2"),
            Path("experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/output/train_v2"),
        ],
    )
    exp092_path = find_input_file(
        EXP092_PREDICTIONS,
        get_nested(config, "data.exp092_predictions"),
        local_roots=[
            Path("/tmp/exp092_train_output_check"),
            Path("experiments/exp092_u_projection_correction_disagreement_fullrun/kaggle/output/train"),
        ],
    )
    exp073, exp073_meta = read_selected_prediction(
        exp073_path,
        selector_col="model",
        selector_value=str(get_nested(config, "model.exp073_model", "lgb_mean")),
        pred_col="pred_exp073_lgb_mean",
        max_rows=max_rows,
    )
    exp092_lgb1, exp092_lgb1_meta = read_selected_prediction(
        exp092_path,
        selector_col="model",
        selector_value=str(get_nested(config, "model.exp092_model", "lgb1")),
        pred_col="pred_exp092_lgb1",
        max_rows=max_rows,
    )
    exp092_mean, exp092_mean_meta = read_selected_prediction(
        exp092_path,
        selector_col="model",
        selector_value=str(get_nested(config, "model.exp092_reference_model", "lgb_mean")),
        pred_col="pred_exp092_lgb_mean",
        max_rows=max_rows,
    )
    merged = (
        exp073.merge(
            exp092_lgb1[["id", "well", "pred_exp092_lgb1"]],
            on=["id", "well"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            exp092_mean[["id", "well", "pred_exp092_lgb_mean"]],
            on=["id", "well"],
            how="inner",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )
    if merged.empty:
        raise ValueError("Prediction inputs have no overlapping rows")
    merged["tail_rank"] = parse_tail_rank(merged["id"])
    return merged, {
        "exp073": exp073_meta,
        "exp092_lgb1": exp092_lgb1_meta,
        "exp092_lgb_mean": exp092_mean_meta,
        "joined_rows": int(len(merged)),
        "joined_wells": int(merged["well"].nunique()),
    }


def load_feature_cache(
    config: dict[str, Any],
    ids: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_input_file(
        EXP072_FEATURES,
        get_nested(config, "data.exp072_feature_cache"),
        local_roots=[
            Path("/tmp/kaggle-output/exp072_exp063_full_replay_feature_cache/train_v1"),
            Path("experiments/exp072_exp063_full_replay_feature_cache/kaggle/output/train_v1"),
            Path("experiments/exp072_exp063_full_replay_feature_cache/artifacts"),
        ],
    )
    columns = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "pf_ancc",
        "pf_ancc_std",
        "beam_mean_d",
        "beam_std_d",
        "sc_ens_d",
        "hyb_d",
        "likpf_mean_d",
        "md_since",
        "eval_len",
    ]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [col for col in columns if col not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    max_rows = get_nested(config, "audit.max_feature_rows")
    max_rows = None if max_rows is None else int(max_rows)
    frame = pd.read_csv(
        source,
        usecols=columns,
        dtype={"id": "string", "well": "string"},
        nrows=max_rows,
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    frame = frame[frame["id"].isin(set(ids.astype(str)))].copy()
    if frame.empty:
        raise ValueError("Feature cache has no rows overlapping prediction ids")
    for col in frame.columns:
        if col not in {"id", "well"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")

    schema_path: Path | None = None
    try:
        schema_path = find_input_file(
            EXP072_SCHEMA,
            get_nested(config, "data.exp072_feature_schema"),
            local_roots=[Path("experiments/exp072_exp063_full_replay_feature_cache/artifacts")],
        )
    except FileNotFoundError:
        schema_path = None
    return frame, {
        "path": str(source),
        "raw_file_sha256": sha256_file(source),
        "decompressed_content_sha256": sha256_decompressed(source),
        "schema_path": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_file(schema_path) if schema_path else None,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": columns,
    }


def build_candidate_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    out["target_tvt_from_cache"] = out["last_known_tvt_cache"] + out["target"]
    target_gap = np.abs(out["target_tvt"] - out["target_tvt_from_cache"])
    if float(target_gap.max()) > 0.25:
        raise ValueError(f"Prediction target and cache target differ: max={target_gap.max()}")
    out["pred_last_anchor"] = out["last_known_tvt_cache"]
    out["pred_pf_ancc"] = out["pf_ancc"]
    out["pred_beam_mean"] = out["last_known_tvt_cache"] + out["beam_mean_d"]
    out["pred_likpf_mean"] = out["last_known_tvt_cache"] + out["likpf_mean_d"]
    out["pred_sc_ens"] = out["last_known_tvt_cache"] + out["sc_ens_d"]
    out["pred_hyb"] = out["last_known_tvt_cache"] + out["hyb_d"]
    candidate_cols = [
        "pred_last_anchor",
        "pred_pf_ancc",
        "pred_beam_mean",
        "pred_likpf_mean",
        "pred_sc_ens",
        "pred_hyb",
        "pred_exp073_lgb_mean",
        "pred_exp092_lgb1",
        "pred_exp092_lgb_mean",
    ]
    return out, candidate_cols


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


def _gather_2d(series: np.ndarray, centers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    gathered = []
    for offset in offsets:
        idx = np.clip(centers + int(offset), 0, len(series) - 1)
        gathered.append(series[idx])
    return np.stack(gathered, axis=-1).astype(np.float32)


def _score_centers(
    *,
    full_gr: np.ndarray,
    smooth_gr: np.ndarray,
    z_gr: np.ndarray,
    derivative: np.ndarray,
    energy: np.ndarray,
    row_idx: np.ndarray,
    candidate_idx: np.ndarray,
    window_offsets: np.ndarray,
    gr_scale: float,
    z_scale: float,
    derivative_scale: float,
    energy_scale: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    eval_raw = _gather_2d(full_gr, row_idx, window_offsets)
    cand_raw = _gather_2d(full_gr, candidate_idx, window_offsets)
    eval_smooth = _gather_2d(smooth_gr, row_idx, window_offsets)
    cand_smooth = _gather_2d(smooth_gr, candidate_idx, window_offsets)
    eval_z = _gather_2d(z_gr, row_idx, window_offsets)
    cand_z = _gather_2d(z_gr, candidate_idx, window_offsets)
    eval_deriv = _gather_2d(derivative, row_idx, window_offsets)
    cand_deriv = _gather_2d(derivative, candidate_idx, window_offsets)
    eval_energy = _gather_2d(energy, row_idx, window_offsets)
    cand_energy = _gather_2d(energy, candidate_idx, window_offsets)

    raw_mae = np.mean(np.abs(cand_raw - eval_raw[:, None, :]), axis=2)
    smooth_mae = np.mean(np.abs(cand_smooth - eval_smooth[:, None, :]), axis=2)
    z_mae = np.mean(np.abs(cand_z - eval_z[:, None, :]), axis=2)
    derivative_mae = np.mean(np.abs(cand_deriv - eval_deriv[:, None, :]), axis=2)
    energy_mae = np.mean(np.abs(cand_energy - eval_energy[:, None, :]), axis=2)
    eval_norm = _standardize_rows(eval_smooth)
    flat_cand = cand_smooth.reshape(
        cand_smooth.shape[0] * cand_smooth.shape[1],
        cand_smooth.shape[2],
    )
    cand_norm = _standardize_rows(flat_cand).reshape(cand_smooth.shape)
    ncc = np.mean(cand_norm * eval_norm[:, None, :], axis=2)
    ncc_score = np.clip((ncc + 1.0) / 2.0, 0.0, 1.0)
    score = (
        np.exp(-smooth_mae / max(gr_scale, 1e-6))
        * (0.25 + 0.75 * ncc_score)
        * np.exp(-z_mae / max(z_scale, 1e-6))
        * np.exp(-derivative_mae / max(derivative_scale, 1e-6))
        * np.exp(-energy_mae / max(energy_scale, 1e-6))
    )
    return np.clip(score, 0.0, 1.0).astype(np.float32), {
        "raw_mae": raw_mae.astype(np.float32),
        "smooth_mae": smooth_mae.astype(np.float32),
        "z_mae": z_mae.astype(np.float32),
        "derivative_mae": derivative_mae.astype(np.float32),
        "energy_mae": energy_mae.astype(np.float32),
        "ncc": ncc.astype(np.float32),
    }


def _build_well_ambiguity(
    *,
    raw: pd.DataFrame,
    group: pd.DataFrame,
    candidate_cols: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    shifts_ft = np.asarray(
        [float(v) for v in config.get("score_shifts_ft", [-25, -20, -15, 0, 15, 20, 25])],
        dtype=np.float32,
    )
    if not np.any(np.isclose(shifts_ft, 0.0)):
        shifts_ft = np.sort(np.append(shifts_ft, 0.0)).astype(np.float32)
    zero_shift_index = int(np.argmin(np.abs(shifts_ft)))
    window_offsets = np.asarray(
        [int(v) for v in config.get("window_offsets_rows", [-24, -12, -6, 0, 6, 12, 24])],
        dtype=np.int32,
    )
    gr_rolling_window = int(config.get("gr_rolling_window", 5))
    gr_scale = float(config.get("gr_scale", 18.0))
    z_scale = float(config.get("z_scale", 2.0))
    derivative_scale = float(config.get("derivative_scale", 5.0))
    energy_scale = float(config.get("energy_scale", 5.0))
    min_peak_score = float(config.get("min_peak_score", 0.25))
    peak_delta_score = float(config.get("peak_delta_score", 0.05))
    margin_threshold = float(config.get("margin_threshold", 0.06))
    entropy_threshold = float(config.get("entropy_threshold", 0.82))
    top2_ratio_threshold = float(config.get("top2_ratio_threshold", 0.88))
    flat_score_range = float(config.get("flat_score_range", 0.04))
    decoy_min = float(config.get("decoy_spacing_min_ft", 12.0))
    decoy_max = float(config.get("decoy_spacing_max_ft", 28.0))

    tvt_input = pd.to_numeric(raw["TVT_input"], errors="coerce")
    known_mask = tvt_input.notna().to_numpy()
    if not known_mask.any():
        raise ValueError(f"No finite TVT_input prefix rows for well {group['well'].iloc[0]}")
    prefix_len = int(np.flatnonzero(known_mask)[-1] + 1)
    prefix_tvt = (
        tvt_input.iloc[:prefix_len]
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(np.float32)
    )
    gr_series = pd.to_numeric(raw["GR"], errors="coerce")
    fallback = float(gr_series.iloc[:prefix_len].mean())
    if not np.isfinite(fallback):
        fallback = float(gr_series.mean()) if np.isfinite(float(gr_series.mean())) else 0.0
    full_gr = (
        gr_series.interpolate(limit_direction="both")
        .fillna(fallback)
        .to_numpy(np.float32)
    )
    smooth_gr = _rolling_mean(full_gr, gr_rolling_window)
    std_gr = _rolling_std(full_gr, gr_rolling_window)
    z_gr = ((full_gr - smooth_gr) / (std_gr + 1e-6)).astype(np.float32)
    derivative = np.gradient(smooth_gr).astype(np.float32)
    energy = _rolling_mean(np.abs(derivative), max(3, gr_rolling_window)).astype(np.float32)

    row_idx = group["tail_rank"].to_numpy(np.int32)
    candidate_values = group[candidate_cols].to_numpy(np.float32)
    candidate_values = np.nan_to_num(candidate_values, nan=float(prefix_tvt[-1]))
    n_rows, n_candidates = candidate_values.shape
    shifted_values = candidate_values[:, :, None] + shifts_ft[None, None, :]
    candidate_idx = _nearest_prefix_indices(prefix_tvt, shifted_values.reshape(-1)).reshape(
        n_rows, n_candidates * len(shifts_ft)
    )
    score_flat, details = _score_centers(
        full_gr=full_gr,
        smooth_gr=smooth_gr,
        z_gr=z_gr,
        derivative=derivative,
        energy=energy,
        row_idx=row_idx,
        candidate_idx=candidate_idx,
        window_offsets=window_offsets,
        gr_scale=gr_scale,
        z_scale=z_scale,
        derivative_scale=derivative_scale,
        energy_scale=energy_scale,
    )
    score_cube = score_flat.reshape(n_rows, n_candidates, len(shifts_ft))
    value_flat = shifted_values.reshape(n_rows, n_candidates * len(shifts_ft))
    order = np.argsort(score_flat, axis=1)
    best = order[:, -1]
    second = order[:, -2] if score_flat.shape[1] > 1 else best
    rows = np.arange(n_rows)
    top1_score = score_flat[rows, best]
    top2_score = score_flat[rows, second]
    margin = (top1_score - top2_score).astype(np.float32)
    top1_tvt = value_flat[rows, best].astype(np.float32)
    top2_tvt = value_flat[rows, second].astype(np.float32)
    spacing = np.abs(top1_tvt - top2_tvt).astype(np.float32)
    probs = score_flat / (score_flat.sum(axis=1, keepdims=True) + 1e-9)
    entropy = (
        -np.sum(probs * np.log(probs + 1e-9), axis=1) / np.log(score_flat.shape[1])
    ).astype(np.float32)
    peak_count = (
        score_flat >= np.maximum(min_peak_score, top1_score[:, None] - peak_delta_score)
    ).sum(axis=1)
    score_range = (score_flat.max(axis=1) - score_flat.min(axis=1)).astype(np.float32)
    flat_flag = (top1_score < min_peak_score) | (score_range < flat_score_range)
    top2_ratio = (top2_score / np.maximum(top1_score, 1e-6)).astype(np.float32)
    decoy_spacing_flag = (spacing >= decoy_min) & (spacing <= decoy_max)
    bimodality = np.where(decoy_spacing_flag, top2_ratio, 0.0).astype(np.float32)
    ambiguous_flag = (
        ~flat_flag
        & (
            (margin <= margin_threshold)
            | (entropy >= entropy_threshold)
            | ((top2_ratio >= top2_ratio_threshold) & decoy_spacing_flag)
            | (peak_count >= 3)
        )
    )
    ambiguity_score = np.clip(
        0.35 * (1.0 - np.clip(margin / max(margin_threshold, 1e-6), 0.0, 1.0))
        + 0.25 * entropy
        + 0.25 * bimodality
        + 0.15 * np.clip((peak_count - 1) / 4.0, 0.0, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)
    zero_scores = score_cube[:, :, zero_shift_index]
    best_zero = np.argmax(zero_scores, axis=1)
    topzero_score = zero_scores[rows, best_zero]
    topzero_tvt = candidate_values[rows, best_zero]
    abs_shift = np.abs(shifts_ft)
    decoy_gap_cols: dict[str, np.ndarray] = {}
    for shift in [15.0, 20.0, 25.0]:
        idx = np.where(np.isclose(abs_shift, shift))[0]
        if len(idx):
            shifted_best = score_cube[:, :, idx].max(axis=(1, 2))
            shifted_topzero = score_cube[
                rows[:, None],
                best_zero[:, None],
                idx[None, :],
            ].max(axis=1)
            decoy_gap_cols[f"grbm_gap_to_shift_{int(shift)}ft_any"] = (
                top1_score - shifted_best
            ).astype(np.float32)
            decoy_gap_cols[f"grbm_gap_to_shift_{int(shift)}ft_topzero"] = (
                topzero_score - shifted_topzero
            ).astype(np.float32)

    candidate_std = np.nanstd(candidate_values, axis=1).astype(np.float32)
    candidate_range = (
        np.nanmax(candidate_values, axis=1) - np.nanmin(candidate_values, axis=1)
    ).astype(np.float32)
    midpoint = ((top1_tvt + top2_tvt) / 2.0).astype(np.float32)
    likpf = group["pred_likpf_mean"].to_numpy(np.float32)
    avg_weight = np.where(ambiguous_flag, np.maximum(0.25, ambiguity_score), 0.0).astype(np.float32)
    likpf_midpoint_blend = ((1.0 - avg_weight) * likpf + avg_weight * midpoint).astype(np.float32)

    out = pd.DataFrame(
        {
            "id": group["id"].to_numpy(),
            "well": group["well"].to_numpy(),
            "grbm_top1_score": top1_score.astype(np.float32),
            "grbm_top2_score": top2_score.astype(np.float32),
            "grbm_top1_top2_margin": margin,
            "grbm_top2_ratio": top2_ratio,
            "grbm_score_entropy": entropy,
            "grbm_peak_count": peak_count.astype(np.float32),
            "grbm_peak_spacing_ft": spacing,
            "grbm_candidate_std": candidate_std,
            "grbm_candidate_range": candidate_range,
            "grbm_bimodality_score": bimodality,
            "grbm_ambiguity_score": ambiguity_score,
            "grbm_ambiguous_flag": ambiguous_flag.astype(np.float32),
            "grbm_flat_score_flag": flat_flag.astype(np.float32),
            "grbm_top1_tvt": top1_tvt,
            "grbm_top2_tvt": top2_tvt,
            "grbm_midpoint_proxy": midpoint,
            "grbm_mode_commit_proxy": top1_tvt,
            "grbm_topzero_tvt": topzero_tvt.astype(np.float32),
            "grbm_likpf_midpoint_blend": likpf_midpoint_blend,
            "grbm_averaging_weight": avg_weight,
            "grbm_top1_source_id": best.astype(np.float32),
            "grbm_top2_source_id": second.astype(np.float32),
            "grbm_top1_smooth_mae": details["smooth_mae"][rows, best].astype(np.float32),
            "grbm_top1_ncc": details["ncc"][rows, best].astype(np.float32),
            "grbm_local_spectral_energy": _gather_2d(
                energy,
                row_idx,
                np.asarray([0], dtype=np.int32),
            )
            .reshape(-1)
            .astype(np.float32),
            **decoy_gap_cols,
        }
    )
    for i, name in enumerate(candidate_cols):
        out[f"grbm_zero_score_{name}"] = zero_scores[:, i].astype(np.float32)
    meta = {
        "rows": int(n_rows),
        "prefix_len": int(prefix_len),
        "eval_len": int(max(0, len(raw) - prefix_len)),
        "gr_missing_rate": float(gr_series.isna().mean()),
        "ambiguous_rate": float(ambiguous_flag.mean()),
        "flat_rate": float(flat_flag.mean()),
        "score_mean": float(score_flat.mean()),
    }
    return out, meta


def build_ambiguity_features(
    frame: pd.DataFrame,
    candidate_cols: list[str],
    *,
    train_dir: str | Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_dir = resolve_train_dir(train_dir)
    rows: list[pd.DataFrame] = []
    well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=False):
        path = train_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw train horizontal well not found: {path}")
        raw = pd.read_csv(path, usecols=["GR", "TVT_input"])
        feature_rows, meta = _build_well_ambiguity(
            raw=raw,
            group=group.sort_values("tail_rank"),
            candidate_cols=candidate_cols,
            config=config,
        )
        rows.append(feature_rows)
        well_rows.append({"well": str(well), **meta})
    feature_frame = pd.concat(rows, ignore_index=True)
    feature_frame = frame[["id", "well"]].merge(
        feature_frame,
        on=["id", "well"],
        how="left",
        validate="one_to_one",
    )
    numeric = feature_frame.drop(columns=["id", "well"]).to_numpy(np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError("GR bimodal ambiguity feature frame contains non-finite values")
    well_summary = pd.DataFrame(well_rows)
    return feature_frame, well_summary, {
        "train_dir": str(train_dir),
        "rows": int(len(feature_frame)),
        "wells": int(feature_frame["well"].nunique()),
        "feature_columns": [c for c in feature_frame.columns if c not in {"id", "well"}],
    }


def add_error_columns(frame: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in candidate_cols:
        out[f"{col}_error"] = out[col] - out["target_tvt"]
        out[f"{col}_abs_error"] = out[f"{col}_error"].abs()
    return out


def candidate_metrics(frame: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in candidate_cols:
        error = frame[f"{col}_error"]
        abs_error = frame[f"{col}_abs_error"]
        rows.append(
            {
                "candidate": col,
                "rows": int(len(frame)),
                "wells": int(frame["well"].nunique()),
                "rmse": rmse(error),
                "mae": float(abs_error.mean()),
                "bias": float(error.mean()),
                "within_5ft": float((abs_error <= 5.0).mean()),
                "within_10ft": float((abs_error <= 10.0).mean()),
                "abs_error_p90": float(abs_error.quantile(0.90)),
                "abs_error_p95": float(abs_error.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def summarize_buckets(frame: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_specs = [
        ("ambiguous_flag", ["grbm_ambiguous_flag"]),
        ("flat_score_flag", ["grbm_flat_score_flag"]),
        ("ambiguity_bucket", ["grbm_ambiguity_bucket"]),
        ("margin_bucket", ["grbm_margin_bucket"]),
        ("entropy_bucket", ["grbm_entropy_bucket"]),
        ("distance_bucket", ["distance_bucket"]),
        ("distance_x_ambiguous", ["distance_bucket", "grbm_ambiguous_flag"]),
    ]
    for family, cols in group_specs:
        for keys, group in frame.groupby(cols, dropna=False, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            record: dict[str, Any] = {
                "group_family": family,
                "rows": int(len(group)),
                "wells": int(group["well"].nunique()),
            }
            for col, key in zip(cols, keys, strict=False):
                record[col] = key
            for candidate in candidate_cols:
                record[f"{candidate}_rmse"] = rmse(group[f"{candidate}_error"])
                record[f"{candidate}_mae"] = float(group[f"{candidate}_abs_error"].mean())
            rows.append(record)
    return pd.DataFrame(rows)


def summarize_wells(frame: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=False):
        row: dict[str, Any] = {
            "well": str(well),
            "rows": int(len(group)),
            "ambiguous_rate": float(group["grbm_ambiguous_flag"].mean()),
            "flat_rate": float(group["grbm_flat_score_flag"].mean()),
            "ambiguity_score_mean": float(group["grbm_ambiguity_score"].mean()),
            "top1_top2_margin_mean": float(group["grbm_top1_top2_margin"].mean()),
        }
        for candidate in candidate_cols:
            row[f"{candidate}_rmse"] = rmse(group[f"{candidate}_error"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pred_exp092_lgb1_rmse", ascending=False)


def write_feature_cache(
    *,
    output_dir: Path,
    source_frame: pd.DataFrame,
    ambiguity_frame: pd.DataFrame,
) -> dict[str, Any]:
    meta_cols = ["id", "well", "target"]
    feature_cols = [c for c in ambiguity_frame.columns if c not in {"id", "well"}]
    out = source_frame[meta_cols].copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(ambiguity_frame[col], errors="coerce").astype("float32")
    numeric = out[["target", *feature_cols]].to_numpy(np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError("feature cache contains non-finite values")
    train_path = output_dir / TRAIN_FEATURE_FILENAME
    schema_path = output_dir / TRAIN_SCHEMA_FILENAME
    out.to_csv(train_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "variant": TRAIN_FEATURE_VARIANT,
            "feature_index": np.arange(len(feature_cols), dtype=np.int32),
            "feature": feature_cols,
        }
    ).to_csv(schema_path, index=False)
    return {
        "variant": TRAIN_FEATURE_VARIANT,
        "rows": int(len(out)),
        "wells": int(out["well"].nunique()),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "outputs": {
            "train_features": train_path.name,
            "train_feature_schema": schema_path.name,
        },
        "sha256": {
            "train_features": sha256_file(train_path),
            "train_features_decompressed": sha256_decompressed(train_path),
            "train_feature_schema": sha256_file(schema_path),
        },
    }


def dataframe_to_markdown(frame: pd.DataFrame, *, max_rows: int = 12, max_cols: int = 12) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).iloc[:, :max_cols].copy()

    def fmt(value: Any) -> str:
        if isinstance(value, float | np.floating):
            return "" if not np.isfinite(float(value)) else f"{float(value):.6g}"
        if pd.isna(value):
            return ""
        return str(value)

    cols = [str(col) for col in display.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in display.columns) + " |")
    if frame.shape[1] > max_cols:
        lines.append(f"\n_Only first {max_cols} columns shown._")
    return "\n".join(lines)


def render_readme(summary: dict[str, Any], metrics: pd.DataFrame, buckets: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# exp133_gr_bimodal_match_ambiguity_detector",
            "",
            "GR score curve の二峰性、+/-15-25ft decoy、flat match を "
            "target-free に検出し、既存候補 error と突き合わせる train-side 診断生成物。",
            "",
            "## Overall",
            "",
            f"- rows: {summary['rows']}",
            f"- wells: {summary['wells']}",
            f"- ambiguous rate: {summary['ambiguity']['ambiguous_rate']:.6f}",
            f"- flat score rate: {summary['ambiguity']['flat_rate']:.6f}",
            "",
            "## Candidate Metrics",
            "",
            dataframe_to_markdown(metrics),
            "",
            "## Ambiguity Buckets",
            "",
            dataframe_to_markdown(buckets[buckets['group_family'].eq('ambiguous_flag')]),
            "",
            "## Files",
            "",
            f"- `{OUTPUT_PREFIX}_row_context.csv.gz`",
            f"- `{OUTPUT_PREFIX}_candidate_metrics.csv`",
            f"- `{OUTPUT_PREFIX}_bucket_metrics.csv`",
            f"- `{OUTPUT_PREFIX}_well_metrics.csv`",
            f"- `{TRAIN_FEATURE_FILENAME}`",
            f"- `{TRAIN_SCHEMA_FILENAME}`",
            f"- `{OUTPUT_PREFIX}_summary.json`",
        ]
    ) + "\n"


def run_train_from_config(config: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, prediction_meta = load_prediction_inputs(config)
    features, feature_meta = load_feature_cache(config, predictions["id"])
    features = features.rename(columns={"last_known_tvt": "last_known_tvt_cache"})
    frame = predictions.merge(features, on=["id", "well"], how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("No rows after joining prediction inputs and exp072 cache")
    frame, base_candidate_cols = build_candidate_columns(frame)

    ambiguity_frame, well_input_summary, ambiguity_meta = build_ambiguity_features(
        frame,
        base_candidate_cols,
        train_dir=get_nested(config, "data.train_dir", "data/raw/train"),
        config=get_nested(config, "model.gr_bimodal_ambiguity", {}),
    )
    frame = frame.merge(ambiguity_frame, on=["id", "well"], how="inner", validate="one_to_one")
    frame["distance_bucket"] = distance_bucket(frame["md_since"])
    frame["grbm_ambiguity_bucket"] = safe_qcut(frame["grbm_ambiguity_score"], 4, prefix="ambiguity")
    frame["grbm_margin_bucket"] = safe_qcut(frame["grbm_top1_top2_margin"], 4, prefix="margin")
    frame["grbm_entropy_bucket"] = safe_qcut(frame["grbm_score_entropy"], 4, prefix="entropy")

    proxy_cols = ["grbm_mode_commit_proxy", "grbm_midpoint_proxy", "grbm_likpf_midpoint_blend"]
    candidate_cols = [*base_candidate_cols, *proxy_cols]
    frame = add_error_columns(frame, candidate_cols)
    metrics = candidate_metrics(frame, candidate_cols)
    buckets = summarize_buckets(frame, candidate_cols)
    wells = summarize_wells(frame, candidate_cols)
    feature_cache = write_feature_cache(
        output_dir=output_dir,
        source_frame=frame[["id", "well", "target"]],
        ambiguity_frame=ambiguity_frame,
    )

    row_cols = [
        "id",
        "well",
        "target_tvt",
        "tail_rank",
        "md_since",
        "distance_bucket",
        "grbm_ambiguous_flag",
        "grbm_flat_score_flag",
        "grbm_ambiguity_score",
        "grbm_top1_top2_margin",
        "grbm_score_entropy",
        "grbm_peak_count",
        "grbm_peak_spacing_ft",
        *candidate_cols,
        *[f"{c}_abs_error" for c in candidate_cols],
    ]
    row_context = frame[row_cols].copy()

    row_path = output_dir / f"{OUTPUT_PREFIX}_row_context.csv.gz"
    metrics_path = output_dir / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    bucket_path = output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    well_path = output_dir / f"{OUTPUT_PREFIX}_well_metrics.csv"
    well_input_path = output_dir / f"{OUTPUT_PREFIX}_grbm_well_input_summary.csv"
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    readme_path = output_dir / "README.md"
    row_context.to_csv(row_path, index=False, compression="gzip")
    metrics.to_csv(metrics_path, index=False)
    buckets.to_csv(bucket_path, index=False)
    wells.to_csv(well_path, index=False)
    well_input_summary.to_csv(well_input_path, index=False)

    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_train_side_diagnostic",
        "runtime_seconds": round(time.time() - started, 3),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "ambiguity": {
            "ambiguous_rate": float(frame["grbm_ambiguous_flag"].mean()),
            "flat_rate": float(frame["grbm_flat_score_flag"].mean()),
            "ambiguity_score_mean": float(frame["grbm_ambiguity_score"].mean()),
            "top1_top2_margin_mean": float(frame["grbm_top1_top2_margin"].mean()),
        },
        "best_candidate_by_rmse": to_jsonable(metrics.iloc[0].to_dict()),
        "feature_cache": feature_cache,
        "ambiguity_features": ambiguity_meta,
        "source": {
            "predictions": prediction_meta,
            "feature_cache": feature_meta,
        },
        "outputs": {
            "row_context": row_path.name,
            "candidate_metrics": metrics_path.name,
            "bucket_metrics": bucket_path.name,
            "well_metrics": well_path.name,
            "well_input_summary": well_input_path.name,
            "train_features": TRAIN_FEATURE_FILENAME,
            "train_feature_schema": TRAIN_SCHEMA_FILENAME,
            "summary": summary_path.name,
            "readme": readme_path.name,
        },
        "recommendation": "diagnostic_only_review_ambiguity_buckets_before_ml_feature_followup",
    }
    with summary_path.open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
        fp.write("\n")
    readme_path.write_text(render_readme(summary, metrics, buckets), encoding="utf-8")
    return summary


def run_inference_from_config(config: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "not_selected_no_submission",
        "reason": (
            "GR bimodal match ambiguity detector is a train-side diagnostic "
            "and feature-cache experiment."
        ),
        "inference_mode": get_nested(config, "inference.mode", "disabled_diagnostic_only"),
        "outputs": {},
    }
    with (output_dir / f"{OUTPUT_PREFIX}_inference_summary.json").open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
        fp.write("\n")
    return summary

from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

OUTPUT_PREFIX = "exp096_projection_fadein_after_prefix"
EXP073_OOF_FILENAME = "exp063_full_replay_repro_guard_predictions.csv.gz"
EXP073_INFERENCE_FILENAME = "exp063_full_replay_repro_guard_inference_test_predictions.csv.gz"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if value is pd.NA:
        return None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except TypeError:
        pass
    return value


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(np.asarray(y_true, float), np.asarray(y_pred, float))))


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_decompressed_csv(path: str | Path) -> str:
    path = Path(path)
    hasher = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prediction_sha256(ids: pd.Series, values: np.ndarray, *, label: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(label.encode("utf-8"))
    for raw_id in ids.astype(str).to_numpy():
        hasher.update(raw_id.encode("utf-8"))
        hasher.update(b"\0")
    hasher.update(np.asarray(values, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current: dict[str, Any] = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _as_path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, str | Path):
        return [Path(value)]
    if isinstance(value, list | tuple):
        return [Path(item) for item in value if item]
    return []


def find_input_file(
    filename: str,
    configured: Any = None,
    *,
    local_roots: list[Path] | None = None,
) -> Path:
    candidates: list[Path] = []
    candidates.extend(_as_path_list(configured))
    for root in local_roots or []:
        candidates.append(root / filename)
        candidates.append(root / "artifacts" / filename)
    candidates.extend(
        [
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for candidate in sorted(input_root.glob(f"**/{filename}")):
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate

    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"input file not found or empty: {filename}. Checked:\n{checked}")


def _tail_index(ids: pd.Series) -> np.ndarray:
    suffix = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    row_idx = pd.to_numeric(suffix, errors="coerce")
    if row_idx.isna().any():
        bad = ids[row_idx.isna()].head(5).tolist()
        raise ValueError(f"Cannot parse row index suffix from ids: {bad}")
    return row_idx.to_numpy(np.int64)


def _distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def _tail_rank_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )


def _tail_length_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 499, 999, 1499, 1999, np.inf],
        labels=["000_499", "500_999", "1000_1499", "1500_1999", "2000_plus"],
        include_lowest=True,
    )


def _well_hash_fold(well: str, n_folds: int) -> int:
    digest = hashlib.sha256(str(well).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % int(n_folds)


def assign_reporting_folds(frame: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    result = frame.copy()
    result["original_fold"] = -1
    groups = result["well"].to_numpy()
    group_count = int(pd.Series(groups).nunique())
    effective_folds = min(int(n_folds), group_count)
    if effective_folds >= 2:
        cv = GroupKFold(n_splits=effective_folds)
        for fold, (_, valid_idx) in enumerate(cv.split(np.zeros(len(result)), groups=groups)):
            result.loc[result.index[valid_idx], "original_fold"] = int(fold)
    else:
        result["original_fold"] = 0
    hash_folds = max(effective_folds, 1)
    result["well_hash_fold"] = result["well"].map(lambda well: _well_hash_fold(str(well), hash_folds))
    return result


def load_exp073_predictions(
    config: dict[str, Any],
    *,
    inference: bool = False,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit = config.get("audit", {})
    source_key = "data.exp073_inference_predictions" if inference else "data.exp073_oof_predictions"
    filename = EXP073_INFERENCE_FILENAME if inference else EXP073_OOF_FILENAME
    local_roots = [
        Path("/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_v2"),
        Path("/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_v2"),
        Path("/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_v1"),
    ]
    source = find_input_file(filename, get_nested(config, source_key), local_roots=local_roots)
    mode = str(audit.get("selected_mode", "gpu_repro_guard_dp_threads8"))
    model = str(audit.get("selected_model", "lgb_mean"))
    usecols = ["id", "well", "mode", "model", "last_known_tvt", "pred_delta", "pred_tvt"]
    if not inference:
        usecols.extend(["target_tvt", "target_delta"])
    dtypes = {
        "id": "string",
        "well": "string",
        "mode": "string",
        "model": "string",
        "last_known_tvt": "float32",
        "pred_delta": "float32",
        "pred_tvt": "float32",
        "target_tvt": "float32",
        "target_delta": "float32",
    }
    chunks: list[pd.DataFrame] = []
    rows = 0
    chunksize = int(audit.get("prediction_read_chunksize", 500_000))
    for chunk in pd.read_csv(source, usecols=usecols, dtype=dtypes, chunksize=chunksize):
        filtered = chunk[(chunk["mode"] == mode) & (chunk["model"] == model)].copy()
        if filtered.empty:
            continue
        if max_rows is not None and rows + len(filtered) > max_rows:
            filtered = filtered.iloc[: max(0, int(max_rows) - rows)].copy()
        chunks.append(filtered)
        rows += len(filtered)
        if max_rows is not None and rows >= int(max_rows):
            break
    if not chunks:
        raise ValueError(f"No rows found in {source} for mode={mode} model={model}")
    frame = pd.concat(chunks, ignore_index=True)
    for col in ["id", "well", "mode", "model"]:
        frame[col] = frame[col].astype(str)
    numeric_cols = [col for col in frame.columns if col not in {"id", "well", "mode", "model"}]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    if not np.isfinite(frame[numeric_cols].to_numpy(np.float32)).all():
        raise ValueError("exp073 prediction frame contains non-finite numeric values")
    metadata = {
        "path": str(source),
        "raw_file_sha256": sha256_file(source),
        "decompressed_content_sha256": sha256_decompressed_csv(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "mode": mode,
        "model": model,
        "prediction_sha256": prediction_sha256(
            frame["id"], frame["pred_tvt"].to_numpy(np.float32), label=f"exp073/{mode}/{model}"
        ),
    }
    return frame, metadata


def add_raw_well_context(
    predictions: pd.DataFrame,
    raw_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = predictions.reset_index(drop=True).copy()
    row_index = _tail_index(frame["id"])
    md = np.full(len(frame), np.nan, dtype=np.float32)
    z = np.full(len(frame), np.nan, dtype=np.float32)
    raw_tvt_input = np.full(len(frame), np.nan, dtype=np.float32)
    anchor_idx = np.full(len(frame), -1, dtype=np.int32)
    anchor_md = np.full(len(frame), np.nan, dtype=np.float32)
    anchor_z0 = np.full(len(frame), np.nan, dtype=np.float32)
    anchor_t0 = np.full(len(frame), np.nan, dtype=np.float32)
    known_prefix_rows = np.full(len(frame), -1, dtype=np.int32)
    raw_dir = Path(raw_dir)
    well_rows: list[dict[str, Any]] = []

    for well, idx in frame.groupby("well", sort=False).indices.items():
        idx_array = np.asarray(idx, dtype=np.int64)
        well_name = str(well)
        path = raw_dir / f"{well_name}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw horizontal well file not found: {path}")
        raw = pd.read_csv(path, usecols=["MD", "Z", "TVT_input"])
        take = row_index[idx_array]
        if int(take.max()) >= len(raw) or int(take.min()) < 0:
            raise ValueError(f"row index out of range for {path}: min={take.min()} max={take.max()}")
        selected = raw.iloc[take]
        md[idx_array] = pd.to_numeric(selected["MD"], errors="coerce").to_numpy(np.float32)
        z[idx_array] = pd.to_numeric(selected["Z"], errors="coerce").to_numpy(np.float32)
        raw_tvt_input[idx_array] = pd.to_numeric(
            selected["TVT_input"], errors="coerce"
        ).to_numpy(np.float32)

        known = raw[pd.to_numeric(raw["TVT_input"], errors="coerce").notna()]
        if known.empty:
            raise ValueError(f"No finite TVT_input prefix rows for {well_name}")
        anchor = known.iloc[-1]
        anchor_row_idx = int(known.index[-1])
        anchor_idx[idx_array] = anchor_row_idx
        anchor_md[idx_array] = float(anchor["MD"])
        anchor_z0[idx_array] = float(anchor["Z"])
        anchor_t0[idx_array] = float(anchor["TVT_input"])
        known_prefix_rows[idx_array] = int(len(known))
        well_rows.append(
            {
                "well": well_name,
                "rows": int(len(idx_array)),
                "raw_rows": int(len(raw)),
                "min_row_index": int(take.min()),
                "max_row_index": int(take.max()),
                "anchor_row_index": anchor_row_idx,
                "known_prefix_rows": int(len(known)),
                "anchor_md": float(anchor["MD"]),
                "anchor_z0": float(anchor["Z"]),
                "anchor_t0": float(anchor["TVT_input"]),
            }
        )

    frame["raw_row_index"] = row_index.astype(np.int32)
    frame["md"] = md
    frame["z"] = z
    frame["raw_tvt_input"] = raw_tvt_input
    frame["anchor_row_index"] = anchor_idx
    frame["anchor_md"] = anchor_md
    frame["anchor_z0"] = anchor_z0
    frame["anchor_t0"] = anchor_t0
    frame["known_prefix_rows"] = known_prefix_rows
    frame["md_since"] = (frame["md"] - frame["anchor_md"]).astype(np.float32)
    frame["tail_rank"] = (frame["raw_row_index"] - frame["anchor_row_index"]).astype(np.int32)
    frame["tail_length"] = frame.groupby("well")["id"].transform("size").astype(np.int32)
    frame["u0"] = (frame["anchor_t0"] + frame["anchor_z0"]).astype(np.float32)
    numeric = [
        "md",
        "z",
        "anchor_md",
        "anchor_z0",
        "anchor_t0",
        "md_since",
        "tail_rank",
        "u0",
    ]
    if not np.isfinite(frame[numeric].to_numpy(np.float32)).all():
        raise ValueError("raw well context contains non-finite numeric values")
    t0_diff = np.abs(
        frame["last_known_tvt"].to_numpy(np.float32) - frame["anchor_t0"].to_numpy(np.float32)
    )
    meta = {
        "raw_dir": str(raw_dir),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "anchor_t0_vs_exp073_last_known_abs_max": float(t0_diff.max()),
        "anchor_t0_vs_exp073_last_known_abs_mean": float(t0_diff.mean()),
        "known_prefix_rows_min": int(frame["known_prefix_rows"].min()),
        "known_prefix_rows_max": int(frame["known_prefix_rows"].max()),
        "well_summary": well_rows[:10],
    }
    if meta["anchor_t0_vs_exp073_last_known_abs_max"] > 0.05:
        raise ValueError(
            "raw prefix anchor does not match exp073 last_known_tvt; "
            f"max abs diff={meta['anchor_t0_vs_exp073_last_known_abs_max']}"
        )
    return frame, meta


def generate_exp073_base_predictions_for_current_test(
    config: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, dict[str, Any]]:
    from exp063_full_replay_reproducibility_guard import run_saved_model_inference

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation = get_nested(config, "inference.exp073_base_generation", {}) or {}
    feature_generation = generation.get("feature_generation", {}) or {}
    sample_path = Path(get_nested(config, "data.sample_submission", "data/raw/sample_submission.csv"))
    raw_dir = Path(get_nested(config, "data.raw_dir", "data/raw"))
    temp_submission_path = output_dir / "_exp073_base_submission.csv"
    summary = run_saved_model_inference(
        output_dir=output_dir,
        submission_path=temp_submission_path,
        sample_submission_path=sample_path,
        data_dir=raw_dir,
        tracker_test_path=generation.get("tracker_test_path"),
        model_manifest_path=generation.get("model_manifest_path"),
        mode_name=str(generation.get("selected_mode", get_nested(config, "audit.selected_mode", "gpu_repro_guard_dp_threads8"))),
        model_name=str(generation.get("selected_model", get_nested(config, "audit.selected_model", "lgb_mean"))),
        submission_target_column=str(get_nested(config, "data.submission_target_column", "tvt")),
        regenerate_test_features=bool(generation.get("regenerate_test_features", True)),
        n_jobs=int(feature_generation.get("n_jobs", 8)),
        pf_seeds=int(feature_generation.get("pf_seeds", 128)),
        pf_particles=int(feature_generation.get("pf_particles", 500)),
        fast=bool(feature_generation.get("fast", False)),
        use_gpu=str(feature_generation.get("use_gpu", "auto")),
    )
    prediction_path = output_dir / EXP073_INFERENCE_FILENAME
    if not prediction_path.exists() or prediction_path.stat().st_size <= 0:
        raise FileNotFoundError(f"generated exp073 base prediction missing: {prediction_path}")
    return prediction_path, summary


def weighted_polyfit_predict(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int,
    robust_c: float,
    robust_iters: int,
) -> tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        fill = float(np.nanmedian(y[finite])) if finite.any() else 0.0
        return np.full(len(y), fill, dtype=np.float32), 0

    x_fit = x[finite]
    y_fit = y[finite]
    x_center = float(np.median(x_fit))
    x_scale = float(np.nanpercentile(x_fit, 95) - np.nanpercentile(x_fit, 5))
    if not np.isfinite(x_scale) or x_scale <= 1e-6:
        x_scale = max(float(np.max(x_fit) - np.min(x_fit)), 1.0)
    x_norm = (x - x_center) / x_scale
    x_fit_norm = x_norm[finite]
    unique_count = int(np.unique(np.round(x_fit_norm, 8)).size)
    fit_degree = int(min(max(int(degree), 0), max(unique_count - 1, 0)))
    weights = np.ones(len(y_fit), dtype=np.float64)
    coef = np.array([float(np.mean(y_fit))])
    for _ in range(max(int(robust_iters), 1)):
        coef = np.polyfit(x_fit_norm, y_fit, deg=fit_degree, w=weights)
        residual = y_fit - np.polyval(coef, x_fit_norm)
        scale = float(np.median(np.abs(residual - np.median(residual)))) * 1.4826
        if not np.isfinite(scale) or scale <= 1e-6:
            break
        weights = np.minimum(1.0, (float(robust_c) * scale) / (np.abs(residual) + 1e-6))
    pred = np.polyval(coef, x_norm)
    return pred.astype(np.float32), fit_degree


def apply_projection_variant(
    frame: pd.DataFrame,
    *,
    degree: int,
    beta: float,
    fade_start: float,
    fade_end: float,
    robust_c: float,
    robust_iters: int,
    x_column: str,
    max_abs_correction: float | None,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "well": frame["well"].to_numpy(),
            "target_tvt": frame["target_tvt"].to_numpy(np.float32)
            if "target_tvt" in frame.columns
            else np.full(len(frame), np.nan, dtype=np.float32),
            "base_pred_tvt": frame["pred_tvt"].to_numpy(np.float32),
            "last_known_tvt": frame["last_known_tvt"].to_numpy(np.float32),
            "md_since": frame["md_since"].to_numpy(np.float32),
            "tail_rank": frame["tail_rank"].to_numpy(np.int32),
            "tail_length": frame["tail_length"].to_numpy(np.int32),
        }
    )
    pred_u = (
        frame["pred_tvt"].to_numpy(np.float32)
        + frame["z"].to_numpy(np.float32)
        - frame["u0"].to_numpy(np.float32)
    )
    projected_u = np.zeros(len(frame), dtype=np.float32)
    used_degree = np.zeros(len(frame), dtype=np.int16)
    x_all = frame[x_column].to_numpy(np.float32)
    for _, idx in frame.groupby("well", sort=False).indices.items():
        idx_array = np.asarray(idx, dtype=np.int64)
        values, fit_degree = weighted_polyfit_predict(
            x_all[idx_array],
            pred_u[idx_array],
            degree=int(degree),
            robust_c=float(robust_c),
            robust_iters=int(robust_iters),
        )
        projected_u[idx_array] = values
        used_degree[idx_array] = int(fit_degree)

    direct_projected_tvt = projected_u - frame["z"].to_numpy(np.float32) + frame["u0"].to_numpy(
        np.float32
    )
    correction = (direct_projected_tvt - frame["pred_tvt"].to_numpy(np.float32)).astype(np.float32)
    if max_abs_correction is not None:
        correction = np.clip(correction, -float(max_abs_correction), float(max_abs_correction))
    md_since = frame["md_since"].to_numpy(np.float32)
    fade_start = float(fade_start)
    fade_end = float(fade_end)
    if fade_end <= fade_start:
        fade_weight = (md_since > fade_start).astype(np.float32)
    else:
        fade_weight = np.clip((md_since - fade_start) / (fade_end - fade_start), 0.0, 1.0).astype(
            np.float32
        )
    effective_beta = (float(beta) * fade_weight).astype(np.float32)
    applied_correction = (effective_beta * correction).astype(np.float32)
    result["pred_tvt"] = (frame["pred_tvt"].to_numpy(np.float32) + applied_correction).astype(
        np.float32
    )
    result["direct_projected_tvt"] = direct_projected_tvt.astype(np.float32)
    result["projection_correction"] = correction.astype(np.float32)
    result["projection_correction_applied"] = applied_correction
    result["projection_beta_effective"] = effective_beta
    result["fade_start"] = np.full(len(result), fade_start, dtype=np.float32)
    result["fade_end"] = np.full(len(result), fade_end, dtype=np.float32)
    result["pred_u"] = pred_u.astype(np.float32)
    result["projected_u"] = projected_u.astype(np.float32)
    result["fit_degree"] = used_degree.astype(np.int16)
    return result


def _metric_frame(
    prediction_frame: pd.DataFrame,
    *,
    variant: str,
    degree: int | None,
    beta: float | None,
    fade_start: float | None,
    fade_end: float | None,
    robust_c: float | None,
    baseline_rmse: float,
) -> dict[str, Any]:
    error = prediction_frame["pred_tvt"].to_numpy(np.float32) - prediction_frame[
        "target_tvt"
    ].to_numpy(np.float32)
    base_error = prediction_frame["base_pred_tvt"].to_numpy(np.float32) - prediction_frame[
        "target_tvt"
    ].to_numpy(np.float32)
    pred_rmse = float(np.sqrt(np.mean(np.square(error))))
    correction = prediction_frame["projection_correction_applied"].to_numpy(np.float32)
    return {
        "variant": variant,
        "degree": degree,
        "beta": beta,
        "fade_start": fade_start,
        "fade_end": fade_end,
        "robust_c": robust_c,
        "rows": int(len(prediction_frame)),
        "wells": int(prediction_frame["well"].nunique()),
        "rmse_tvt": pred_rmse,
        "baseline_rmse_tvt": baseline_rmse,
        "delta_vs_baseline": pred_rmse - baseline_rmse,
        "base_error_abs_mean": float(np.mean(np.abs(base_error))),
        "error_abs_mean": float(np.mean(np.abs(error))),
        "prediction_min": float(prediction_frame["pred_tvt"].min()),
        "prediction_max": float(prediction_frame["pred_tvt"].max()),
        "prediction_mean": float(prediction_frame["pred_tvt"].mean()),
        "prediction_std": float(prediction_frame["pred_tvt"].std()),
        "correction_abs_mean": float(np.mean(np.abs(correction))),
        "correction_abs_p95": float(np.quantile(np.abs(correction), 0.95)),
        "correction_abs_max": float(np.max(np.abs(correction))),
        "effective_beta_mean": float(
            np.mean(prediction_frame.get("projection_beta_effective", pd.Series([0.0])).to_numpy())
        ),
        "effective_beta_max": float(
            np.max(prediction_frame.get("projection_beta_effective", pd.Series([0.0])).to_numpy())
        ),
        "prediction_sha256": prediction_sha256(
            prediction_frame["id"],
            prediction_frame["pred_tvt"].to_numpy(np.float32),
            label=variant,
        ),
    }


def _group_metrics(
    prediction_frame: pd.DataFrame,
    base_frame: pd.DataFrame,
    *,
    variant: str,
    group_col: str,
    baseline_by_group: pd.DataFrame,
) -> pd.DataFrame:
    frame = prediction_frame[["id", "target_tvt", "pred_tvt"]].merge(
        base_frame[["id", group_col]], on="id", how="left", validate="one_to_one"
    )
    frame["error"] = frame["pred_tvt"] - frame["target_tvt"]
    grouped = (
        frame.groupby(group_col, observed=True)
        .agg(
            rows=("id", "size"),
            rmse_tvt=("error", lambda value: float(np.sqrt(np.mean(np.square(value))))),
            error_abs_mean=("error", lambda value: float(np.mean(np.abs(value)))),
        )
        .reset_index()
        .rename(columns={group_col: "group"})
    )
    grouped["variant"] = variant
    grouped["group_family"] = group_col
    grouped["group"] = grouped["group"].astype(str)
    grouped = grouped.merge(
        baseline_by_group[["group", "baseline_rmse_tvt"]],
        on="group",
        how="left",
        validate="one_to_one",
    )
    grouped["delta_vs_baseline"] = grouped["rmse_tvt"] - grouped["baseline_rmse_tvt"]
    return grouped[
        [
            "variant",
            "group_family",
            "group",
            "rows",
            "rmse_tvt",
            "baseline_rmse_tvt",
            "delta_vs_baseline",
            "error_abs_mean",
        ]
    ]


def _baseline_group_metrics(base_frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    frame = base_frame[["id", "target_tvt", "pred_tvt", group_col]].copy()
    frame["error"] = frame["pred_tvt"] - frame["target_tvt"]
    grouped = (
        frame.groupby(group_col, observed=True)
        .agg(
            rows=("id", "size"),
            baseline_rmse_tvt=("error", lambda value: float(np.sqrt(np.mean(np.square(value))))),
        )
        .reset_index()
        .rename(columns={group_col: "group"})
    )
    grouped["group"] = grouped["group"].astype(str)
    return grouped


def evaluate_projection_grid(
    frame: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = config.get("audit", {})
    grid = audit.get("grid", {})
    betas = [float(value) for value in grid.get("betas", [0.25, 0.5, 0.75])]
    fade_windows = [
        (float(value["start"]), float(value["end"]))
        if isinstance(value, dict)
        else (float(value[0]), float(value[1]))
        for value in grid.get("fade_windows", [{"start": 250.0, "end": 750.0}])
    ]
    if grid.get("variants"):
        projection_variants = [
            {"degree": int(value["degree"]), "robust_c": float(value["robust_c"])}
            for value in grid["variants"]
        ]
    else:
        degrees = [int(value) for value in grid.get("degrees", [3, 4, 5])]
        robust_cs = [float(value) for value in grid.get("robust_c", [1.5, 2.0, 3.0])]
        projection_variants = [
            {"degree": degree, "robust_c": robust_c}
            for degree in degrees
            for robust_c in robust_cs
        ]
    robust_iters = int(grid.get("robust_iters", 4))
    x_column = str(grid.get("x_column", "md_since"))
    max_abs = grid.get("max_abs_correction_ft")
    max_abs_correction = None if max_abs is None else float(max_abs)

    n_folds = int(get_nested(config, "validation.n_folds", 5))
    base = assign_reporting_folds(frame, n_folds)
    base["distance_bucket"] = _distance_bucket(base["md_since"]).astype(str)
    base["tail_rank_bucket"] = _tail_rank_bucket(base["tail_rank"]).astype(str)
    base["tail_length_bucket"] = _tail_length_bucket(base["tail_length"]).astype(str)
    base_prediction = pd.DataFrame(
        {
            "id": base["id"],
            "well": base["well"],
            "target_tvt": base["target_tvt"],
            "base_pred_tvt": base["pred_tvt"],
            "pred_tvt": base["pred_tvt"],
            "projection_correction_applied": np.zeros(len(base), dtype=np.float32),
            "projection_beta_effective": np.zeros(len(base), dtype=np.float32),
            "fade_start": np.full(len(base), np.nan, dtype=np.float32),
            "fade_end": np.full(len(base), np.nan, dtype=np.float32),
        }
    )
    baseline_rmse = rmse(base["target_tvt"].to_numpy(np.float32), base["pred_tvt"].to_numpy(np.float32))
    variant_rows = [
        _metric_frame(
            base_prediction,
            variant="baseline_exp073",
            degree=None,
            beta=None,
            fade_start=None,
            fade_end=None,
            robust_c=None,
            baseline_rmse=baseline_rmse,
        )
    ]
    baseline_groups = {
        group: _baseline_group_metrics(base, group)
        for group in [
            "original_fold",
            "well_hash_fold",
            "distance_bucket",
            "tail_rank_bucket",
            "tail_length_bucket",
        ]
    }
    fold_frames: list[pd.DataFrame] = []
    bucket_frames: list[pd.DataFrame] = []
    by_well_frames: list[pd.DataFrame] = []
    best_variant: dict[str, Any] | None = None

    for spec in projection_variants:
        degree = int(spec["degree"])
        robust_c = float(spec["robust_c"])
        for beta in betas:
            for fade_start, fade_end in fade_windows:
                variant = (
                    f"degree{degree}_beta{beta:g}_c{robust_c:g}"
                    f"_fade{fade_start:g}_{fade_end:g}"
                )
                pred = apply_projection_variant(
                    base,
                    degree=degree,
                    beta=beta,
                    fade_start=fade_start,
                    fade_end=fade_end,
                    robust_c=robust_c,
                    robust_iters=robust_iters,
                    x_column=x_column,
                    max_abs_correction=max_abs_correction,
                )
                metric = _metric_frame(
                    pred,
                    variant=variant,
                    degree=degree,
                    beta=beta,
                    fade_start=fade_start,
                    fade_end=fade_end,
                    robust_c=robust_c,
                    baseline_rmse=baseline_rmse,
                )
                variant_rows.append(metric)
                if best_variant is None or metric["rmse_tvt"] < best_variant["rmse_tvt"]:
                    best_variant = dict(metric)

                for group in ["original_fold", "well_hash_fold"]:
                    fold_frames.append(
                        _group_metrics(
                            pred,
                            base,
                            variant=variant,
                            group_col=group,
                            baseline_by_group=baseline_groups[group],
                        )
                    )
                for group in ["distance_bucket", "tail_rank_bucket", "tail_length_bucket"]:
                    bucket_frames.append(
                        _group_metrics(
                            pred,
                            base,
                            variant=variant,
                            group_col=group,
                            baseline_by_group=baseline_groups[group],
                        )
                    )
                by_well = _group_metrics(
                    pred,
                    base,
                    variant=variant,
                    group_col="well",
                    baseline_by_group=_baseline_group_metrics(base, "well"),
                ).sort_values("delta_vs_baseline", ascending=False)
                by_well_frames.append(by_well)

    variant_metrics = pd.DataFrame(variant_rows).sort_values("rmse_tvt")
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    bucket_metrics = pd.concat(bucket_frames, ignore_index=True)
    by_well = pd.concat(by_well_frames, ignore_index=True)

    best_variant_name = str(variant_metrics.iloc[0]["variant"])
    if best_variant_name == "baseline_exp073":
        best_prediction = base_prediction.copy()
    else:
        best_row = variant_metrics.iloc[0]
        best_prediction = apply_projection_variant(
            base,
            degree=int(best_row["degree"]),
            beta=float(best_row["beta"]),
            fade_start=float(best_row["fade_start"]),
            fade_end=float(best_row["fade_end"]),
            robust_c=float(best_row["robust_c"]),
            robust_iters=robust_iters,
            x_column=x_column,
            max_abs_correction=max_abs_correction,
        )
    best_prediction.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_best_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    variant_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_variant_metrics.csv", index=False)
    fold_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_fold_metrics.csv", index=False)
    bucket_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv", index=False)
    by_well.to_csv(output_dir / f"{OUTPUT_PREFIX}_by_well.csv", index=False)
    context_schema = pd.DataFrame(
        {
            "column": [
                "md",
                "z",
                "anchor_md",
                "anchor_z0",
                "anchor_t0",
                "md_since",
                "tail_rank",
                "tail_length",
                "u0",
                "projection_beta_effective",
                "fade_start",
                "fade_end",
            ],
            "source": [
                "raw horizontal well",
                "raw horizontal well",
                "last finite TVT_input prefix row",
                "last finite TVT_input prefix row",
                "last finite TVT_input prefix row",
                "md - anchor_md",
                "raw_row_index - anchor_row_index",
                "rows per scored well",
                "anchor_t0 + anchor_z0",
                "row-wise beta after md_since fade-in",
                "projection beta is zero at and below this md_since",
                "projection beta reaches selected beta at and above this md_since",
            ],
        }
    )
    context_schema.to_csv(output_dir / f"{OUTPUT_PREFIX}_context_schema.csv", index=False)

    best_fold = fold_metrics[fold_metrics["variant"] == best_variant_name]
    best_bucket = bucket_metrics[bucket_metrics["variant"] == best_variant_name]
    guard = audit.get("selection_guard", {})
    max_fold_regression = float(best_fold["delta_vs_baseline"].max()) if len(best_fold) else 0.0
    near_rows = best_bucket[
        (best_bucket["group_family"] == "distance_bucket") & (best_bucket["group"].eq("000_050"))
    ]
    short_tail = best_bucket[
        (best_bucket["group_family"] == "tail_length_bucket")
        & (best_bucket["group"].isin(["000_499", "500_999"]))
    ]
    near_regression = float(near_rows["delta_vs_baseline"].max()) if len(near_rows) else 0.0
    short_tail_regression = (
        float(short_tail["delta_vs_baseline"].max()) if len(short_tail) else 0.0
    )
    best_metric = variant_metrics.iloc[0].to_dict()
    passes_guard = (
        best_variant_name != "baseline_exp073"
        and float(best_metric["delta_vs_baseline"]) <= -float(guard.get("min_rmse_gain", 0.0))
        and max_fold_regression <= float(guard.get("max_fold_regression", 0.0))
        and near_regression <= float(guard.get("max_near_row_regression", 0.02))
        and short_tail_regression <= float(guard.get("max_short_tail_regression", 0.02))
        and float(best_metric["correction_abs_p95"]) <= float(guard.get("max_correction_abs_p95", 10.0))
    )
    recommendation = "port_to_inference_candidate" if passes_guard else "do_not_port_without_review"
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "completed_projection_audit",
        "runtime_seconds": round(time.time() - started, 3),
        "baseline": variant_rows[0],
        "best_variant": to_jsonable(best_metric),
        "guard": {
            "passes_guard": bool(passes_guard),
            "recommendation": recommendation,
            "max_fold_regression": max_fold_regression,
            "near_row_regression": near_regression,
            "short_tail_regression": short_tail_regression,
            "thresholds": guard,
        },
        "grid": {
            "projection_variants": projection_variants,
            "betas": betas,
            "fade_windows": [{"start": start, "end": end} for start, end in fade_windows],
            "robust_iters": robust_iters,
            "x_column": x_column,
            "max_abs_correction_ft": max_abs_correction,
        },
        "outputs": {
            "variant_metrics": f"{OUTPUT_PREFIX}_variant_metrics.csv",
            "fold_metrics": f"{OUTPUT_PREFIX}_fold_metrics.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "best_predictions": f"{OUTPUT_PREFIX}_best_predictions.csv.gz",
            "context_schema": f"{OUTPUT_PREFIX}_context_schema.csv",
            "summary": f"{OUTPUT_PREFIX}_summary.json",
        },
    }
    with (output_dir / f"{OUTPUT_PREFIX}_summary.json").open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
    return summary


def run_train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    max_rows = get_nested(config, "audit.max_rows")
    max_rows = None if max_rows is None else int(max_rows)
    prediction_frame, prediction_meta = load_exp073_predictions(
        config,
        inference=False,
        max_rows=max_rows,
    )
    raw_dir = get_nested(config, "data.train_dir", "data/raw/train")
    frame, raw_meta = add_raw_well_context(prediction_frame, raw_dir)
    output_dir = get_nested(config, "runtime.output_dir", None)
    if output_dir is None:
        output_dir = Path("experiments") / OUTPUT_PREFIX / "artifacts"
    summary = evaluate_projection_grid(frame, config, output_dir)
    summary["source"] = {
        "exp073_oof_predictions": prediction_meta,
        "raw_context": raw_meta,
    }
    with (Path(output_dir) / f"{OUTPUT_PREFIX}_summary.json").open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
    return summary


def run_inference_from_config(config: dict[str, Any]) -> dict[str, Any]:
    selected = get_nested(config, "inference.selected_variant")
    output_dir = Path(get_nested(config, "runtime.output_dir", Path("experiments") / OUTPUT_PREFIX / "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if not selected:
        summary = {
            "experiment": OUTPUT_PREFIX,
            "status": "not_selected_no_submission",
            "reason": "inference.selected_variant is null; train-side projection guard must pass first.",
            "outputs": {},
        }
        with (output_dir / f"{OUTPUT_PREFIX}_inference_summary.json").open("w") as fp:
            json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
        return summary

    base_generation_meta: dict[str, Any] | None = None
    if bool(get_nested(config, "inference.generate_exp073_base_on_current_test", False)):
        generated_prediction_path, base_generation_meta = generate_exp073_base_predictions_for_current_test(
            config,
            output_dir,
        )
        configured_sources = _as_path_list(get_nested(config, "data.exp073_inference_predictions"))
        set_nested(
            config,
            "data.exp073_inference_predictions",
            [str(generated_prediction_path), *[str(path) for path in configured_sources]],
        )

    prediction_frame, prediction_meta = load_exp073_predictions(config, inference=True)
    raw_dir = get_nested(config, "data.test_dir", "data/raw/test")
    frame, raw_meta = add_raw_well_context(prediction_frame, raw_dir)
    pred = apply_projection_variant(
        frame,
        degree=int(selected["degree"]),
        beta=float(selected["beta"]),
        fade_start=float(selected.get("fade_start", get_nested(config, "audit.grid.default_fade_start", 250.0))),
        fade_end=float(selected.get("fade_end", get_nested(config, "audit.grid.default_fade_end", 750.0))),
        robust_c=float(selected["robust_c"]),
        robust_iters=int(selected.get("robust_iters", get_nested(config, "audit.grid.robust_iters", 4))),
        x_column=str(selected.get("x_column", get_nested(config, "audit.grid.x_column", "md_since"))),
        max_abs_correction=selected.get("max_abs_correction_ft"),
    )
    pred_path = output_dir / f"{OUTPUT_PREFIX}_inference_test_predictions.csv.gz"
    pred.to_csv(pred_path, index=False, compression="gzip")

    sample_path = Path(get_nested(config, "data.sample_submission", "data/raw/sample_submission.csv"))
    sample = pd.read_csv(sample_path, dtype={"id": str})
    target_col = str(get_nested(config, "data.submission_target_column", "tvt"))
    pred_map = dict(zip(pred["id"].astype(str), pred["pred_tvt"], strict=False))
    mapped = sample["id"].astype(str).map(pred_map)
    fallback_rows = int(mapped.isna().sum())
    if fallback_rows:
        allow_fallback = bool(get_nested(config, "inference.allow_submission_fallback", False))
        if not allow_fallback:
            missing = sample.loc[mapped.isna(), "id"].astype(str).head(10).tolist()
            raise ValueError(
                "projection inference did not produce predictions for all submission ids; "
                f"missing_rows={fallback_rows} first_missing_ids={missing}"
            )
        mapped = mapped.fillna(float(pred["pred_tvt"].mean()))
    sample[target_col] = mapped.astype("float64")
    submission_path = output_dir.parent / "submission.csv"
    sample.to_csv(submission_path, index=False)
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "inference_projection_written",
        "selected_variant": selected,
        "source": {
            "exp073_inference_predictions": prediction_meta,
            "exp073_base_generation": base_generation_meta,
            "raw_context": raw_meta,
        },
        "submission": {
            "path": str(submission_path),
            "rows": int(len(sample)),
            "fallback_rows": fallback_rows,
            "prediction_min": float(sample[target_col].min()),
            "prediction_max": float(sample[target_col].max()),
            "prediction_mean": float(sample[target_col].mean()),
            "prediction_std": float(sample[target_col].std()),
            "submission_sha256": sha256_file(submission_path),
        },
        "outputs": {
            "test_predictions": pred_path.name,
            "summary": f"{OUTPUT_PREFIX}_inference_summary.json",
        },
    }
    with (output_dir / f"{OUTPUT_PREFIX}_inference_summary.json").open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
    return summary

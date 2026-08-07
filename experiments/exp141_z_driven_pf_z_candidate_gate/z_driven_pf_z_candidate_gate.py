from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import KAGGLE_INPUT_ROOT, get_nested

OUTPUT_PREFIX = "exp141_z_driven_pf_z_candidate_gate"
EXP072_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
EXP072_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"


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
        return None if not np.isfinite(float(value)) else float(value)
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except TypeError:
        pass
    return value


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
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")))

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(
        f"input file not found or empty: {filename}. Checked:\n" + "\n".join(checked[:160])
    )


def parse_tail_rank(ids: pd.Series) -> pd.Series:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    return pd.to_numeric(extracted, errors="raise").astype("int32")


def rmse(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(array[finite]))))


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


def tail_rank_bucket(values: pd.Series | np.ndarray) -> pd.Series:
    return (
        pd.cut(
            pd.to_numeric(values, errors="coerce"),
            bins=[-np.inf, 99, 249, 499, 999, 1499, np.inf],
            labels=["000_099", "100_249", "250_499", "500_999", "1000_1499", "1500_plus"],
            include_lowest=True,
        )
        .astype("string")
        .fillna("unknown")
    )


def safe_qcut(values: pd.Series, q: int, *, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    result = pd.Series("missing", index=values.index, dtype="object")
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


def finite_quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return float("inf")
    return float(finite.quantile(q))


def percentile_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    ranked = numeric.rank(pct=True, method="average")
    return ranked.fillna(0.0).astype("float32")


def keep_min_true_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    keep = np.zeros(len(mask), dtype=bool)
    if len(mask) == 0:
        return keep
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        if (not value or idx == len(mask) - 1) and start is not None:
            end = idx + 1 if value and idx == len(mask) - 1 else idx
            if end - start >= min_len:
                keep[start:end] = True
            start = None
    return keep


def contiguous_true_segments(indices: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    start: int | None = None
    for pos, value in enumerate(mask):
        if value and start is None:
            start = pos
        if (not value or pos == len(mask) - 1) and start is not None:
            end = pos + 1 if value and pos == len(mask) - 1 else pos
            segments.append(indices[start:end])
            start = None
    return segments


def cap_mask_by_rate(
    frame: pd.DataFrame,
    mask: pd.Series,
    score: pd.Series,
    *,
    cap: float | None,
    scope: str,
) -> tuple[pd.Series, dict[str, Any]]:
    mask = mask.astype(bool)
    pre_rows = int(mask.sum())
    if cap is None or cap <= 0 or pre_rows == 0:
        return mask, {"cap_applied": False, "pre_cap_rows": pre_rows, "post_cap_rows": pre_rows}

    max_rows = max(1, int(np.floor(len(frame) * float(cap))))
    if pre_rows <= max_rows:
        return mask, {
            "cap_applied": False,
            "switch_rate_cap": float(cap),
            "pre_cap_rows": pre_rows,
            "post_cap_rows": pre_rows,
            "cap_max_rows": max_rows,
        }

    keep = pd.Series(False, index=frame.index)
    if scope == "segment":
        segment_rows: list[dict[str, Any]] = []
        ordered_frame = frame.sort_values(["well", "tail_rank"])
        for _, group in ordered_frame.groupby("well", sort=False):
            group_idx = group.index.to_numpy()
            group_mask = mask.loc[group_idx].to_numpy(bool)
            for segment_idx in contiguous_true_segments(group_idx, group_mask):
                segment_rows.append(
                    {
                        "indices": segment_idx,
                        "rows": int(len(segment_idx)),
                        "score": float(score.loc[segment_idx].mean()),
                    }
                )
        used = 0
        for segment in sorted(segment_rows, key=lambda item: item["score"], reverse=True):
            if used >= max_rows:
                break
            if used + int(segment["rows"]) > max_rows and used > 0:
                continue
            keep.loc[segment["indices"]] = True
            used += int(segment["rows"])
    elif scope == "well":
        well_score = (
            score[mask].groupby(frame.loc[mask, "well"]).mean().sort_values(ascending=False)
        )
        used = 0
        for well in well_score.index:
            indices = frame.index[mask & frame["well"].eq(well)]
            if used >= max_rows:
                break
            if used + len(indices) > max_rows and used > 0:
                continue
            keep.loc[indices] = True
            used += len(indices)
    else:
        selected = score[mask].sort_values(ascending=False).head(max_rows).index
        keep.loc[selected] = True

    return keep, {
        "cap_applied": True,
        "switch_rate_cap": float(cap),
        "pre_cap_rows": pre_rows,
        "post_cap_rows": int(keep.sum()),
        "cap_max_rows": max_rows,
    }


def load_feature_cache(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_input_file(
        EXP072_FEATURES,
        get_nested(config, "data.exp072_feature_cache"),
        local_roots=[
            Path("/tmp/exp072_cache_redownload"),
            Path("/tmp/kaggle-output/exp072_exp063_full_replay_feature_cache/train_v1"),
            Path("experiments/exp072_exp063_full_replay_feature_cache/kaggle/output/train_v1"),
        ],
    )
    required_columns = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "pf_ancc",
        "pf_ancc_std",
        "pf_z",
        "pf_z_delta",
        "pf_vs_z",
        "beam_mean_d",
        "beam_std_d",
        "likpf_mean_d",
        "md_since",
        "eval_len",
        "dzdmd",
        "z",
    ]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")

    max_rows = get_nested(config, "audit.max_feature_rows")
    max_rows = None if max_rows is None else int(max_rows)
    frame = pd.read_csv(
        source,
        usecols=required_columns,
        dtype={"id": "string", "well": "string"},
        nrows=max_rows,
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    if frame.empty:
        raise ValueError("exp072 feature cache is empty after loading")

    schema_path: Path | None = None
    try:
        schema_path = find_input_file(
            EXP072_SCHEMA,
            get_nested(config, "data.exp072_feature_schema"),
            local_roots=[
                Path("experiments/exp072_exp063_full_replay_feature_cache/artifacts"),
                Path("/tmp/exp072_cache_redownload"),
                Path("/tmp/kaggle-output/exp072_exp063_full_replay_feature_cache/train_v1"),
            ],
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
        "columns": required_columns,
    }


def add_path_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["well", "tail_rank"]).copy()
    for column in ["pred_likpf_mean", "pred_pf_z", "pred_pf_ancc", "pred_beam_mean"]:
        slope_values = np.zeros(len(out), dtype=np.float32)
        curvature_values = np.zeros(len(out), dtype=np.float32)
        for _, group in out.groupby("well", sort=False):
            idx = group.index.to_numpy()
            pred = group[column].to_numpy(np.float64)
            md = group["md_since"].to_numpy(np.float64)
            if len(group) <= 1:
                continue
            delta_md = np.diff(md)
            fallback = np.nanmedian(delta_md[delta_md > 0]) if np.any(delta_md > 0) else 1.0
            delta_md = np.where(delta_md > 0, delta_md, fallback)
            slope = np.concatenate([[0.0], np.diff(pred) / delta_md])
            curvature = np.concatenate([[0.0], np.diff(slope)])
            slope_values[out.index.get_indexer(idx)] = slope.astype(np.float32)
            curvature_values[out.index.get_indexer(idx)] = curvature.astype(np.float32)
        out[f"{column}_slope"] = slope_values
        out[f"{column}_curvature"] = curvature_values
    return out.sort_index()


def build_surface(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["tail_rank"] = parse_tail_rank(out["id"])
    out["target_tvt"] = out["last_known_tvt"] + out["target"]
    out["pred_likpf_mean"] = out["last_known_tvt"] + out["likpf_mean_d"]
    out["pred_pf_z"] = out["pf_z"]
    out["pred_pf_ancc"] = out["pf_ancc"]
    out["pred_beam_mean"] = out["last_known_tvt"] + out["beam_mean_d"]
    out = add_path_features(out)

    out["minus_z_slope"] = -pd.to_numeric(out["dzdmd"], errors="coerce")
    out["dzdmd_abs"] = out["dzdmd"].abs()
    out["pf_z_likpf_abs_diff"] = (out["pred_pf_z"] - out["pred_likpf_mean"]).abs()
    out["pf_ancc_z_abs_diff"] = (out["pred_pf_ancc"] - out["pred_pf_z"]).abs()
    out["pf_beam_abs_diff"] = (out["pred_pf_ancc"] - out["pred_beam_mean"]).abs()
    out["likpf_beam_abs_diff"] = (out["pred_likpf_mean"] - out["pred_beam_mean"]).abs()
    out["pf_z_alignment_abs"] = (out["pred_pf_z_slope"] - out["minus_z_slope"]).abs()
    out["likpf_alignment_abs"] = (out["pred_likpf_mean_slope"] - out["minus_z_slope"]).abs()
    out["z_alignment_margin"] = out["likpf_alignment_abs"] - out["pf_z_alignment_abs"]
    out["pf_z_roughness_abs"] = out["pred_pf_z_curvature"].abs()
    out["likpf_roughness_abs"] = out["pred_likpf_mean_curvature"].abs()
    out["pf_z_step_abs"] = out["pred_pf_z_slope"].abs()
    out["likpf_step_abs"] = out["pred_likpf_mean_slope"].abs()

    out["distance_bucket"] = distance_bucket(out["md_since"])
    out["tail_rank_bucket"] = tail_rank_bucket(out["tail_rank"])
    out["z_slope_bucket"] = safe_qcut(out["dzdmd_abs"], 4, prefix="z_slope")
    out["z_alignment_margin_bucket"] = safe_qcut(
        out["z_alignment_margin"], 4, prefix="z_align_margin"
    )
    out["pf_z_likpf_diff_bucket"] = safe_qcut(
        out["pf_z_likpf_abs_diff"], 4, prefix="pfz_likpf_diff"
    )
    out["pf_beam_disagreement_bucket"] = safe_qcut(
        out["pf_beam_abs_diff"], 4, prefix="pf_beam_diff"
    )
    out["pf_z_roughness_bucket"] = safe_qcut(out["pf_z_roughness_abs"], 4, prefix="pfz_roughness")

    candidate_cols = ["pred_likpf_mean", "pred_pf_z", "pred_pf_ancc", "pred_beam_mean"]
    for column in candidate_cols:
        out[f"{column}_error"] = out[column] - out["target_tvt"]
        out[f"{column}_abs_error"] = out[f"{column}_error"].abs()
    abs_frame = out[[f"{column}_abs_error" for column in candidate_cols]]
    out["oracle_candidate"] = abs_frame.idxmin(axis=1).str.replace("_abs_error", "", regex=False)
    out["oracle_abs_error"] = abs_frame.min(axis=1)
    pfz_abs = out[["pred_likpf_mean_abs_error", "pred_pf_z_abs_error"]]
    out["oracle_likpf_pfz_candidate"] = pfz_abs.idxmin(axis=1).str.replace(
        "_abs_error", "", regex=False
    )
    out["oracle_likpf_pfz_abs_error"] = pfz_abs.min(axis=1)
    return out


def build_gate_mask(
    frame: pd.DataFrame,
    variant: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    thresholds = {
        "dzdmd_abs": finite_quantile(frame["dzdmd_abs"], float(variant.get("dzdmd_q", 0.85))),
        "z_alignment_margin": finite_quantile(
            frame["z_alignment_margin"], float(variant.get("alignment_margin_q", 0.70))
        ),
        "pf_z_likpf_abs_diff": finite_quantile(
            frame["pf_z_likpf_abs_diff"], float(variant.get("pf_z_diff_q", 0.70))
        ),
        "pf_ancc_z_abs_diff": finite_quantile(
            frame["pf_ancc_z_abs_diff"], float(variant.get("pf_z_diff_q", 0.70))
        ),
        "pf_beam_abs_diff": finite_quantile(
            frame["pf_beam_abs_diff"], float(variant.get("pf_beam_q", 0.70))
        ),
        "beam_std_d": finite_quantile(
            frame["beam_std_d"].abs(), float(variant.get("pf_beam_q", 0.70))
        ),
        "pf_z_roughness_abs": finite_quantile(
            frame["pf_z_roughness_abs"], float(variant.get("roughness_max_q", 0.90))
        ),
    }
    condition_frame = pd.DataFrame(
        {
            "z_slope_high": frame["dzdmd_abs"] >= thresholds["dzdmd_abs"],
            "pf_z_more_z_aligned": frame["z_alignment_margin"] >= thresholds["z_alignment_margin"],
            "pf_z_likpf_diff_high": frame["pf_z_likpf_abs_diff"]
            >= thresholds["pf_z_likpf_abs_diff"],
            "pf_ancc_z_diff_high": frame["pf_ancc_z_abs_diff"] >= thresholds["pf_ancc_z_abs_diff"],
            "pf_beam_diff_high": frame["pf_beam_abs_diff"] >= thresholds["pf_beam_abs_diff"],
            "beam_spread_high": frame["beam_std_d"].abs() >= thresholds["beam_std_d"],
        },
        index=frame.index,
    )
    condition_score = condition_frame.sum(axis=1)
    max_abs_delta = variant.get("max_abs_delta")
    delta_ok = pd.Series(True, index=frame.index)
    if max_abs_delta is not None:
        delta_ok = frame["pf_z_likpf_abs_diff"] <= float(max_abs_delta)
    roughness_ok = frame["pf_z_roughness_abs"] <= thresholds["pf_z_roughness_abs"]
    guard = (
        (frame["tail_rank"] >= int(variant.get("min_tail_rank", 100)))
        & (frame["md_since"] >= float(variant.get("min_md_since", 50.0)))
        & np.isfinite(frame["pred_pf_z"])
        & np.isfinite(frame["pred_likpf_mean"])
    )
    high = (
        (condition_score >= int(variant.get("min_conditions", 3))) & roughness_ok & delta_ok & guard
    )
    scope = str(variant.get("scope", "segment"))
    if scope == "row":
        mask = high.astype(bool)
    elif scope == "segment":
        values = np.zeros(len(frame), dtype=bool)
        ordered_frame = frame.sort_values(["well", "tail_rank"])
        min_segment_rows = int(variant.get("min_segment_rows", 16))
        for _, group in ordered_frame.groupby("well", sort=False):
            group_mask = high.loc[group.index].to_numpy(bool)
            values[frame.index.get_indexer(group.index)] = keep_min_true_runs(
                group_mask, min_segment_rows
            )
        mask = pd.Series(values, index=frame.index)
    elif scope == "well":
        high_rate = high.groupby(frame["well"]).mean()
        eligible = set(
            high_rate[high_rate >= float(variant.get("min_well_high_rate", 0.02))]
            .index.astype(str)
            .tolist()
        )
        mask = frame["well"].astype(str).isin(eligible) & guard
    else:
        raise ValueError(f"unknown gate scope: {scope}")

    gate_score = (
        percentile_rank(frame["dzdmd_abs"])
        + percentile_rank(frame["z_alignment_margin"])
        + percentile_rank(frame["pf_z_likpf_abs_diff"])
        + percentile_rank(frame["pf_beam_abs_diff"])
        + percentile_rank(frame["beam_std_d"].abs())
        - percentile_rank(frame["pf_z_roughness_abs"])
    )
    capped_mask, cap_meta = cap_mask_by_rate(
        frame,
        mask,
        gate_score,
        cap=variant.get("switch_rate_cap"),
        scope=scope,
    )
    meta = {
        "thresholds": thresholds,
        "scope": scope,
        "min_conditions": int(variant.get("min_conditions", 3)),
        "min_tail_rank": int(variant.get("min_tail_rank", 100)),
        "min_md_since": float(variant.get("min_md_since", 50.0)),
        "max_abs_delta": None if max_abs_delta is None else float(max_abs_delta),
        "pre_scope_high_rows": int(high.sum()),
        "pre_scope_high_rate": float(high.mean()),
        "pre_cap_gate_rows": int(mask.sum()),
        "pre_cap_gate_rate": float(mask.mean()),
        "gate_rows": int(capped_mask.sum()),
        "gate_rate": float(capped_mask.mean()),
        "gate_wells": int(frame.loc[capped_mask, "well"].nunique()),
        "condition_true_rates": {
            column: float(condition_frame[column].mean()) for column in condition_frame.columns
        },
        **cap_meta,
    }
    return capped_mask, meta


def apply_gate_variant(
    frame: pd.DataFrame, variant: dict[str, Any]
) -> tuple[pd.Series, dict[str, Any]]:
    mask, mask_meta = build_gate_mask(frame, variant)
    base = frame["pred_likpf_mean"].astype("float32")
    correction = frame["pred_pf_z"].astype("float32") - base
    clip = variant.get("clip_abs")
    if clip is not None:
        correction = correction.clip(lower=-float(clip), upper=float(clip))
    alpha = float(variant.get("alpha", 1.0))
    pred = base.astype("float64").copy()
    pred.loc[mask] = base.loc[mask].astype("float64") + alpha * correction.loc[mask].astype(
        "float64"
    )
    return pred.astype("float32"), {
        **mask_meta,
        "candidate": "pred_pf_z",
        "alpha": alpha,
        "clip_abs": None if clip is None else float(clip),
    }


def summarize_prediction(
    frame: pd.DataFrame,
    *,
    name: str,
    pred: pd.Series,
    gate_meta: dict[str, Any] | None,
    base_by_well: pd.Series | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    error = pred - frame["target_tvt"]
    abs_error = error.abs()
    rows: list[dict[str, Any]] = []
    step_ge10 = 0
    step_ge25 = 0
    step_p95_values: list[float] = []
    for well, group in frame.assign(_pred=pred, _error=error, _abs_error=abs_error).groupby(
        "well", sort=False
    ):
        ordered = group.sort_values("tail_rank")
        step = ordered["_pred"].diff().abs().dropna()
        step_ge10 += int((step >= 10.0).sum()) if len(step) else 0
        step_ge25 += int((step >= 25.0).sum()) if len(step) else 0
        if len(step):
            step_p95_values.append(float(step.quantile(0.95)))
        rows.append(
            {
                "variant": name,
                "well": str(well),
                "rows": int(len(group)),
                "rmse": rmse(group["_error"]),
                "mae": float(group["_abs_error"].mean()),
                "within10": float((group["_abs_error"] <= 10.0).mean()),
                "step_abs_p95": float(step.quantile(0.95)) if len(step) else np.nan,
                "step_abs_max": float(step.max()) if len(step) else np.nan,
                "step_ge10": int((step >= 10.0).sum()) if len(step) else 0,
                "step_ge25": int((step >= 25.0).sum()) if len(step) else 0,
            }
        )
    by_well = pd.DataFrame(rows)
    if base_by_well is not None:
        by_well = by_well.merge(
            base_by_well.rename("base_likpf_rmse"),
            left_on="well",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        by_well["delta_rmse_vs_likpf"] = by_well["rmse"] - by_well["base_likpf_rmse"]
        max_well_regression = float(by_well["delta_rmse_vs_likpf"].max())
        improved_wells = int((by_well["delta_rmse_vs_likpf"] < -1e-12).sum())
        worsened_wells = int((by_well["delta_rmse_vs_likpf"] > 1e-12).sum())
    else:
        by_well["delta_rmse_vs_likpf"] = 0.0
        max_well_regression = 0.0
        improved_wells = 0
        worsened_wells = 0

    summary = {
        "variant": name,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "rmse": rmse(error),
        "mae": float(abs_error.mean()),
        "bias": float(error.mean()),
        "within10": float((abs_error <= 10.0).mean()),
        "abs_error_p90": float(abs_error.quantile(0.90)),
        "abs_error_p95": float(abs_error.quantile(0.95)),
        "step_ge10": int(step_ge10),
        "step_ge25": int(step_ge25),
        "step_abs_p95_mean_by_well": float(np.nanmean(step_p95_values))
        if step_p95_values
        else np.nan,
        "max_well_regression_vs_likpf": max_well_regression,
        "improved_wells_vs_likpf": improved_wells,
        "worsened_wells_vs_likpf": worsened_wells,
    }
    if gate_meta:
        summary.update(gate_meta)
    return summary, by_well


def summarize_buckets(
    frame: pd.DataFrame,
    predictions: dict[str, pd.Series],
    group_specs: list[tuple[str, list[str]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = frame.copy()
    for name, pred in predictions.items():
        work[f"__{name}_error"] = pred - work["target_tvt"]
        work[f"__{name}_abs_error"] = work[f"__{name}_error"].abs()
    for family, cols in group_specs:
        for keys, group in work.groupby(cols, dropna=False, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            record: dict[str, Any] = {
                "group_family": family,
                "rows": int(len(group)),
                "wells": int(group["well"].nunique()),
            }
            for col, key in zip(cols, keys, strict=False):
                record[col] = key
            for name in predictions:
                record[f"{name}_rmse"] = rmse(group[f"__{name}_error"])
                record[f"{name}_mae"] = float(group[f"__{name}_abs_error"].mean())
                record[f"{name}_within10"] = float((group[f"__{name}_abs_error"] <= 10.0).mean())
            if "base_likpf_mean" in predictions:
                base_rmse = record["base_likpf_mean_rmse"]
                for name in predictions:
                    record[f"{name}_delta_rmse_vs_likpf"] = record[f"{name}_rmse"] - base_rmse
            rows.append(record)
    return pd.DataFrame(rows)


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


def rawtest_parity_checklist(config: dict[str, Any], feature_meta: dict[str, Any]) -> pd.DataFrame:
    required_columns = set(get_nested(config, "audit.required_rawtest_compatible_columns") or [])
    available = set(feature_meta.get("columns", []))
    rows = [
        {
            "check": "feature_cache_required_columns_present",
            "status": "pass" if required_columns.issubset(available) else "fail",
            "detail": ",".join(sorted(required_columns - available)),
        },
        {
            "check": "gate_conditions_target_free",
            "status": "pass",
            "detail": (
                "Gate masks use trajectory, candidate disagreement, path-shape, "
                "and distance columns only."
            ),
        },
        {
            "check": "new_model_training",
            "status": "pass",
            "detail": "No model is trained in exp141.",
        },
        {
            "check": "pf_particle_regeneration",
            "status": "pass",
            "detail": "No PF path is regenerated; exp072 saved cache is read as input.",
        },
        {
            "check": "inference_port",
            "status": "not_applicable",
            "detail": (
                "Diagnostic-only. A raw-test inference port is required before any submission."
            ),
        },
    ]
    return pd.DataFrame(rows)


def render_readme(
    summary: dict[str, Any],
    metrics: pd.DataFrame,
    bucket_metrics: pd.DataFrame,
    representative: pd.DataFrame,
) -> str:
    best = metrics.sort_values("rmse").head(12)
    risky = bucket_metrics.sort_values("base_likpf_mean_rmse", ascending=False).head(12)
    lines = [
        "# exp141_z_driven_pf_z_candidate_gate",
        "",
        (
            "`likpf_mean` を default に固定し、Z-driven と見なせる区間だけ "
            "`pf_z` を低頻度に選ぶ posthoc gate 診断。"
        ),
        "新規学習、PF 再生成、提出ファイル生成は行わない。",
        "",
        "## Overall",
        "",
        f"- rows: {summary['rows']}",
        f"- wells: {summary['wells']}",
        f"- base likpf_mean RMSE: {summary['base']['rmse']:.9f}",
        f"- best RMSE variant: {summary['best']['variant']} / {summary['best']['rmse']:.9f}",
        f"- best delta vs likpf: {summary['best']['delta_rmse_vs_likpf']:.9f}",
        "",
        "## Best Variants",
        "",
        dataframe_to_markdown(best),
        "",
        "## Representative Z-driven Wells",
        "",
        dataframe_to_markdown(representative),
        "",
        "## Highest Base RMSE Buckets",
        "",
        dataframe_to_markdown(risky),
        "",
        "## Files",
        "",
        f"- `{OUTPUT_PREFIX}_metrics.csv`",
        f"- `{OUTPUT_PREFIX}_gate_variants.csv`",
        f"- `{OUTPUT_PREFIX}_by_well.csv`",
        f"- `{OUTPUT_PREFIX}_bucket_metrics.csv`",
        f"- `{OUTPUT_PREFIX}_representative_wells.csv`",
        f"- `{OUTPUT_PREFIX}_rawtest_parity_checklist.csv`",
        f"- `{OUTPUT_PREFIX}_prediction_sample.csv.gz`",
        f"- `{OUTPUT_PREFIX}_summary.json`",
    ]
    return "\n".join(lines) + "\n"


def summarize_representative_wells(
    frame: pd.DataFrame,
    predictions: dict[str, pd.Series],
    wells: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well in wells:
        subset = frame[frame["well"].astype(str).eq(str(well))]
        if subset.empty:
            rows.append({"well": str(well), "variant": "missing", "rows": 0})
            continue
        for name, pred in predictions.items():
            local_pred = pred.loc[subset.index]
            error = local_pred - subset["target_tvt"]
            abs_error = error.abs()
            rows.append(
                {
                    "well": str(well),
                    "variant": name,
                    "rows": int(len(subset)),
                    "rmse": rmse(error),
                    "mae": float(abs_error.mean()),
                    "within10": float((abs_error <= 10.0).mean()),
                    "z_slope_abs_mean": float(subset["dzdmd_abs"].mean()),
                    "pf_z_likpf_abs_diff_mean": float(subset["pf_z_likpf_abs_diff"].mean()),
                }
            )
    result = pd.DataFrame(rows)
    if "rmse" not in result.columns:
        result["base_likpf_rmse"] = np.nan
        result["delta_rmse_vs_likpf"] = np.nan
        return result
    base = result[result["variant"].eq("base_likpf_mean")][["well", "rmse"]].rename(
        columns={"rmse": "base_likpf_rmse"}
    )
    result = result.merge(base, on="well", how="left")
    if "rmse" in result:
        result["delta_rmse_vs_likpf"] = result["rmse"] - result["base_likpf_rmse"]
    return result


def run_gate_audit(
    frame: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
    source_meta: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, pd.Series] = {
        "base_likpf_mean": frame["pred_likpf_mean"].astype("float32"),
        "single_pf_z": frame["pred_pf_z"].astype("float32"),
        "single_pf_ancc": frame["pred_pf_ancc"].astype("float32"),
        "single_beam_mean": frame["pred_beam_mean"].astype("float32"),
        "oracle_likpf_pfz": (frame["target_tvt"] + frame["oracle_likpf_pfz_abs_error"]).astype(
            "float32"
        ),
        "oracle_core_pfbeam": (frame["target_tvt"] + frame["oracle_abs_error"]).astype("float32"),
    }
    gate_rows: list[dict[str, Any]] = []
    for variant in get_nested(config, "audit.gate_variants") or []:
        name = str(variant["name"])
        pred, gate_meta = apply_gate_variant(frame, variant)
        predictions[name] = pred
        gate_rows.append({"variant": name, **gate_meta})

    base_error_by_well = frame.groupby("well", sort=False)["pred_likpf_mean_error"].apply(rmse)
    metric_rows: list[dict[str, Any]] = []
    by_well_frames: list[pd.DataFrame] = []
    for name, pred in predictions.items():
        gate_meta = next((row for row in gate_rows if row["variant"] == name), None)
        summary, by_well = summarize_prediction(
            frame,
            name=name,
            pred=pred,
            gate_meta=gate_meta,
            base_by_well=base_error_by_well,
        )
        metric_rows.append(summary)
        by_well_frames.append(by_well)
    metrics = pd.DataFrame(metric_rows)
    base_rmse = float(metrics.loc[metrics["variant"].eq("base_likpf_mean"), "rmse"].iloc[0])
    metrics["delta_rmse_vs_likpf"] = metrics["rmse"] - base_rmse
    by_well_all = pd.concat(by_well_frames, ignore_index=True)

    group_specs = [
        ("distance_bucket", ["distance_bucket"]),
        ("tail_rank_bucket", ["tail_rank_bucket"]),
        ("z_slope_bucket", ["z_slope_bucket"]),
        ("z_alignment_margin_bucket", ["z_alignment_margin_bucket"]),
        ("pf_z_likpf_diff_bucket", ["pf_z_likpf_diff_bucket"]),
        ("pf_beam_disagreement_bucket", ["pf_beam_disagreement_bucket"]),
        ("pf_z_roughness_bucket", ["pf_z_roughness_bucket"]),
        ("distance_x_z_slope", ["distance_bucket", "z_slope_bucket"]),
        ("distance_x_pfz_likpf_diff", ["distance_bucket", "pf_z_likpf_diff_bucket"]),
    ]
    bucket_metrics = summarize_buckets(frame, predictions, group_specs)
    representative_wells = [
        str(well) for well in (get_nested(config, "audit.representative_z_driven_wells") or [])
    ]
    representative = summarize_representative_wells(frame, predictions, representative_wells)
    gate_variants = pd.DataFrame(gate_rows)
    parity = rawtest_parity_checklist(config, source_meta["feature_cache"])

    sample_n = int(get_nested(config, "audit.prediction_sample_rows") or 200000)
    sample_cols = [
        "id",
        "well",
        "target_tvt",
        "tail_rank",
        "md_since",
        "distance_bucket",
        "z_slope_bucket",
        "z_alignment_margin_bucket",
        "oracle_candidate",
        "oracle_likpf_pfz_candidate",
        "pred_likpf_mean",
        "pred_pf_z",
        "pred_pf_ancc",
        "pred_beam_mean",
        "dzdmd_abs",
        "z_alignment_margin",
        "pf_z_likpf_abs_diff",
        "pf_beam_abs_diff",
        "pf_z_roughness_abs",
    ]
    prediction_sample = frame[sample_cols].head(sample_n).copy()

    paths = {
        "metrics": output_dir / f"{OUTPUT_PREFIX}_metrics.csv",
        "gate_variants": output_dir / f"{OUTPUT_PREFIX}_gate_variants.csv",
        "by_well": output_dir / f"{OUTPUT_PREFIX}_by_well.csv",
        "bucket_metrics": output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv",
        "representative": output_dir / f"{OUTPUT_PREFIX}_representative_wells.csv",
        "rawtest_parity": output_dir / f"{OUTPUT_PREFIX}_rawtest_parity_checklist.csv",
        "prediction_sample": output_dir / f"{OUTPUT_PREFIX}_prediction_sample.csv.gz",
        "summary": output_dir / f"{OUTPUT_PREFIX}_summary.json",
        "readme": output_dir / "README.md",
    }
    metrics.to_csv(paths["metrics"], index=False)
    gate_variants.to_csv(paths["gate_variants"], index=False)
    by_well_all.to_csv(paths["by_well"], index=False)
    bucket_metrics.to_csv(paths["bucket_metrics"], index=False)
    representative.to_csv(paths["representative"], index=False)
    parity.to_csv(paths["rawtest_parity"], index=False)
    prediction_sample.to_csv(paths["prediction_sample"], index=False, compression="gzip")

    best_row = metrics[~metrics["variant"].str.startswith("oracle_")].sort_values("rmse").iloc[0]
    base_row = metrics[metrics["variant"].eq("base_likpf_mean")].iloc[0]
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_train_side_posthoc_audit",
        "runtime_seconds": round(time.time() - started, 3),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "base": {
            "variant": "base_likpf_mean",
            "rmse": float(base_row["rmse"]),
            "mae": float(base_row["mae"]),
            "within10": float(base_row["within10"]),
        },
        "best": {
            "variant": str(best_row["variant"]),
            "rmse": float(best_row["rmse"]),
            "delta_rmse_vs_likpf": float(best_row["delta_rmse_vs_likpf"]),
            "max_well_regression_vs_likpf": float(best_row["max_well_regression_vs_likpf"]),
        },
        "oracle": {
            "likpf_pfz_rmse": float(
                metrics.loc[metrics["variant"].eq("oracle_likpf_pfz"), "rmse"].iloc[0]
            ),
            "core_pfbeam_rmse": float(
                metrics.loc[metrics["variant"].eq("oracle_core_pfbeam"), "rmse"].iloc[0]
            ),
            "likpf_pfz_oracle_distribution": frame["oracle_likpf_pfz_candidate"]
            .value_counts()
            .to_dict(),
            "core_oracle_distribution": frame["oracle_candidate"].value_counts().to_dict(),
        },
        "source": source_meta,
        "outputs": {key: path.name for key, path in paths.items()},
    }
    with paths["summary"].open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
        fp.write("\n")
    paths["readme"].write_text(
        render_readme(summary, metrics, bucket_metrics, representative),
        encoding="utf-8",
    )
    return summary


def run_train_from_config(config: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
    features, feature_meta = load_feature_cache(config)
    frame = build_surface(features)
    source_meta = {"feature_cache": feature_meta}
    return run_gate_audit(frame, config, output_dir, source_meta)


def run_inference_from_config(config: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "not_selected_no_submission",
        "reason": "exp141 is a train-side posthoc gate audit only.",
        "inference_mode": get_nested(config, "inference.mode") or "disabled_diagnostic_only",
        "outputs": {},
    }
    with (output_dir / f"{OUTPUT_PREFIX}_inference_summary.json").open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
        fp.write("\n")
    return summary

from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_PREFIX = "exp135_tvt_dense_high_drift_confidence_gate_on_exp092"

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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
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
        f"input file not found or empty: {filename}. Checked:\n" + "\n".join(checked[:120])
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


def resolve_train_dir(train_dir: str | Path) -> Path:
    path = Path(train_dir)
    if path.exists():
        return path
    input_root = Path("/kaggle/input")
    if input_root.exists():
        direct_candidates = [
            input_root / "competitions" / "rogii-wellbore-geology-prediction" / "train",
            input_root / "rogii-wellbore-geology-prediction" / "train",
        ]
        for candidate in direct_candidates:
            if candidate.exists():
                return candidate
        for candidate in sorted(input_root.glob("**/train")):
            if any(candidate.glob("*__horizontal_well.csv")):
                return candidate
    return path


def read_selected_prediction(
    path: Path,
    *,
    selector_col: str,
    selector_value: str,
    pred_col: str,
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = ["id", "well", selector_col, "target_tvt", "last_known_tvt", "pred_tvt"]
    frame = pd.read_csv(
        path,
        usecols=usecols,
        nrows=max_rows,
        dtype={"id": "string", "well": "string"},
        low_memory=False,
    )
    selected = frame[frame[selector_col].astype(str).eq(selector_value)].copy()
    if selected.empty:
        raise ValueError(f"No rows for {selector_col}={selector_value} in {path}")
    selected["id"] = selected["id"].astype(str)
    selected["well"] = selected["well"].astype(str)
    selected[pred_col] = pd.to_numeric(selected["pred_tvt"], errors="raise").astype("float32")
    selected["target_tvt"] = pd.to_numeric(selected["target_tvt"], errors="raise").astype("float32")
    selected["last_known_tvt"] = pd.to_numeric(selected["last_known_tvt"], errors="raise").astype(
        "float32"
    )
    out = selected[["id", "well", "target_tvt", "last_known_tvt", pred_col]].reset_index(drop=True)
    meta = {
        "path": str(path),
        "raw_file_sha256": sha256_file(path),
        "decompressed_content_sha256": sha256_decompressed(path),
        "selector_col": selector_col,
        "selector_value": selector_value,
        "rows": int(len(out)),
        "wells": int(out["well"].nunique()),
    }
    return out, meta


def load_prediction_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    max_rows = get_nested(config, "audit.max_prediction_rows")
    max_rows = None if max_rows is None else int(max_rows)
    exp073_path = find_input_file(
        EXP073_PREDICTIONS,
        get_nested(config, "data.exp073_predictions"),
        local_roots=[
            Path(
                "/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_v2"
            ),
            Path(
                "experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/output/train_v2"
            ),
        ],
    )
    exp092_path = find_input_file(
        EXP092_PREDICTIONS,
        get_nested(config, "data.exp092_predictions"),
        local_roots=[
            Path("/tmp/exp092_train_output_check"),
            Path(
                "experiments/exp092_u_projection_correction_disagreement_fullrun/kaggle/output/train"
            ),
        ],
    )
    exp073, exp073_meta = read_selected_prediction(
        exp073_path,
        selector_col="model",
        selector_value=str(get_nested(config, "model.exp073_model", "lgb_mean")),
        pred_col="pred_exp073_lgb_mean",
        max_rows=max_rows,
    )
    exp092, exp092_meta = read_selected_prediction(
        exp092_path,
        selector_col="model",
        selector_value=str(get_nested(config, "model.exp092_model", "lgb1")),
        pred_col="pred_exp092_lgb1",
        max_rows=max_rows,
    )
    merged = exp092.merge(
        exp073[["id", "well", "pred_exp073_lgb_mean"]],
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Prediction inputs have no overlapping rows")
    merged["tail_rank"] = parse_tail_rank(merged["id"])
    return merged, {
        "exp092_lgb1": exp092_meta,
        "exp073_lgb_mean": exp073_meta,
        "joined_rows": int(len(merged)),
        "joined_wells": int(merged["well"].nunique()),
    }


def load_feature_cache(
    config: dict[str, Any], ids: pd.Series
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_input_file(
        EXP072_FEATURES,
        get_nested(config, "data.exp072_feature_cache"),
        local_roots=[
            Path("/tmp/kaggle-output/exp072_exp063_full_replay_feature_cache/train_v1"),
            Path("experiments/exp072_exp063_full_replay_feature_cache/kaggle/output/train_v1"),
        ],
    )
    columns = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "pf_ancc",
        "pf_ancc_std",
        "pf_z",
        "pf_z_delta",
        "beam_mean_d",
        "beam_std_d",
        "likpf_mean_d",
        "tvt_dense_d",
        "tvt_densew_d",
        "tvt_dense50_d",
        "tvtF_ANCC",
        "dense_std",
        "dense_dist",
        "dense_nb_std",
        "pf_vs_dense",
        "spatial_vs_dense",
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
    id_set = set(ids.astype(str))
    frame = frame[frame["id"].isin(id_set)].copy()
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


def add_raw_context(
    frame: pd.DataFrame, train_dir: str | Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_dir = resolve_train_dir(train_dir)
    rows: list[pd.DataFrame] = []
    well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=False):
        path = train_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw train well not found: {path}")
        raw = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
        raw["MD"] = pd.to_numeric(raw["MD"], errors="coerce")
        raw["Z"] = pd.to_numeric(raw["Z"], errors="coerce")
        raw["GR"] = pd.to_numeric(raw["GR"], errors="coerce")
        raw["TVT_input"] = pd.to_numeric(raw["TVT_input"], errors="coerce")
        known = raw[raw["TVT_input"].notna()]
        if known.empty:
            raise ValueError(f"No TVT_input prefix for well {well}")
        prefix_len = int(known.index[-1] + 1)
        anchor_md = float(raw.loc[prefix_len - 1, "MD"])
        idx = group["tail_rank"].to_numpy(np.int64)
        selected = raw.iloc[idx].copy()
        rows.append(
            pd.DataFrame(
                {
                    "id": group["id"].to_numpy(),
                    "raw_md": selected["MD"].to_numpy(np.float32),
                    "raw_z": selected["Z"].to_numpy(np.float32),
                    "raw_gr": selected["GR"].to_numpy(np.float32),
                    "md_since": selected["MD"].to_numpy(np.float32) - np.float32(anchor_md),
                }
            )
        )
        eval_part = raw.iloc[prefix_len:]
        well_rows.append(
            {
                "well": str(well),
                "prefix_length": prefix_len,
                "tail_length": int(len(raw) - prefix_len),
                "eval_z_span": float(eval_part["Z"].max() - eval_part["Z"].min()),
                "eval_gr_missing_rate": float(eval_part["GR"].isna().mean()),
            }
        )
    context = pd.concat(rows, ignore_index=True)
    well_context = pd.DataFrame(well_rows)
    out = frame.merge(context, on="id", how="left", validate="one_to_one")
    out = out.merge(well_context, on="well", how="left", validate="many_to_one")
    return out, {
        "train_dir": str(train_dir),
        "rows": int(len(out)),
        "wells": int(out["well"].nunique()),
        "prefix_length_min": int(out["prefix_length"].min()),
        "prefix_length_max": int(out["prefix_length"].max()),
    }


def build_surface(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    out["target_tvt_from_cache"] = out["last_known_tvt_cache"] + out["target"]
    target_gap = np.abs(out["target_tvt"] - out["target_tvt_from_cache"])
    if float(target_gap.max()) > 0.25:
        raise ValueError(
            f"Prediction target and feature-cache target differ: max={target_gap.max()}"
        )

    out["pred_likpf_mean"] = out["last_known_tvt_cache"] + out["likpf_mean_d"]
    out["pred_pf_ancc"] = out["pf_ancc"]
    out["pred_pf_z"] = out["pf_z"]
    out["pred_beam_mean"] = out["last_known_tvt_cache"] + out["beam_mean_d"]
    out["pred_tvt_dense"] = out["last_known_tvt_cache"] + out["tvt_dense_d"]
    out["pred_tvt_densew"] = out["last_known_tvt_cache"] + out["tvt_densew_d"]
    out["pred_tvt_dense50"] = out["last_known_tvt_cache"] + out["tvt_dense50_d"]
    out["pred_tvtF_ANCC"] = out["tvtF_ANCC"]

    out["pf_beam_abs_diff"] = (out["pred_pf_ancc"] - out["pred_beam_mean"]).abs()
    out["pf_dense_abs_diff"] = (out["pred_likpf_mean"] - out["pred_tvt_dense"]).abs()
    out["exp092_dense_abs_diff"] = (out["pred_exp092_lgb1"] - out["pred_tvt_dense"]).abs()
    out["tvt_dense_d_abs"] = out["tvt_dense_d"].abs()
    out["likpf_mean_d_abs"] = out["likpf_mean_d"].abs()
    out["dense_std_abs"] = out["dense_std"].abs()
    out["pf_vs_dense_abs"] = out["pf_vs_dense"].abs()

    out["distance_bucket"] = distance_bucket(out["md_since"])
    out["tail_rank_bucket"] = tail_rank_bucket(out["tail_rank"])
    out["pf_dense_disagreement_bucket"] = safe_qcut(
        out["pf_dense_abs_diff"], 4, prefix="pf_dense_diff"
    )
    out["exp092_dense_disagreement_bucket"] = safe_qcut(
        out["exp092_dense_abs_diff"], 4, prefix="exp092_dense_diff"
    )
    out["dense_std_bucket"] = safe_qcut(out["dense_std_abs"], 4, prefix="dense_std")
    out["dense_drift_bucket"] = safe_qcut(out["tvt_dense_d_abs"], 4, prefix="dense_drift")
    out["pf_beam_disagreement_bucket"] = safe_qcut(
        out["pf_beam_abs_diff"], 4, prefix="pf_beam_diff"
    )

    candidate_cols = [
        "pred_exp092_lgb1",
        "pred_exp073_lgb_mean",
        "pred_likpf_mean",
        "pred_pf_ancc",
        "pred_pf_z",
        "pred_beam_mean",
        "pred_tvt_dense",
        "pred_tvt_densew",
        "pred_tvt_dense50",
        "pred_tvtF_ANCC",
    ]
    for col in candidate_cols:
        out[f"{col}_error"] = out[col] - out["target_tvt"]
        out[f"{col}_abs_error"] = out[f"{col}_error"].abs()
    dense_candidate_cols = [
        "pred_tvt_dense",
        "pred_tvt_densew",
        "pred_tvt_dense50",
        "pred_tvtF_ANCC",
    ]
    all_abs = out[[f"{col}_abs_error" for col in candidate_cols]]
    dense_abs = out[[f"{col}_abs_error" for col in dense_candidate_cols]]
    out["oracle_candidate"] = all_abs.idxmin(axis=1).str.replace("_abs_error", "", regex=False)
    out["oracle_abs_error"] = all_abs.min(axis=1)
    out["dense_oracle_candidate"] = dense_abs.idxmin(axis=1).str.replace(
        "_abs_error", "", regex=False
    )
    out["dense_oracle_abs_error"] = dense_abs.min(axis=1)
    return out, candidate_cols


def finite_quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return float("inf")
    return float(finite.quantile(q))


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


def build_target_free_gate_mask(
    frame: pd.DataFrame,
    *,
    dense_q: float,
    pf_dense_q: float,
    pf_beam_q: float,
    min_tail_rank: int,
    min_conditions: int,
    scope: str,
    min_segment_rows: int,
    min_well_high_rate: float,
) -> tuple[pd.Series, dict[str, Any]]:
    thresholds = {
        "dense_std_abs": finite_quantile(frame["dense_std_abs"], dense_q),
        "tvt_dense_d_abs": finite_quantile(frame["tvt_dense_d_abs"], dense_q),
        "pf_dense_abs_diff": finite_quantile(frame["pf_dense_abs_diff"], pf_dense_q),
        "exp092_dense_abs_diff": finite_quantile(frame["exp092_dense_abs_diff"], pf_dense_q),
        "pf_beam_abs_diff": finite_quantile(frame["pf_beam_abs_diff"], pf_beam_q),
    }
    condition_frame = pd.DataFrame(
        {
            "dense_std_high": frame["dense_std_abs"] >= thresholds["dense_std_abs"],
            "dense_drift_high": frame["tvt_dense_d_abs"] >= thresholds["tvt_dense_d_abs"],
            "pf_dense_diff_high": frame["pf_dense_abs_diff"] >= thresholds["pf_dense_abs_diff"],
            "exp092_dense_diff_high": frame["exp092_dense_abs_diff"]
            >= thresholds["exp092_dense_abs_diff"],
            "pf_beam_diff_high": frame["pf_beam_abs_diff"] >= thresholds["pf_beam_abs_diff"],
        },
        index=frame.index,
    )
    high_score = condition_frame.sum(axis=1)
    high = (frame["tail_rank"] >= int(min_tail_rank)) & (high_score >= int(min_conditions))

    if scope == "row":
        mask = high.to_numpy(bool)
    elif scope == "segment":
        mask = np.zeros(len(frame), dtype=bool)
        for _, group in frame.sort_values(["well", "tail_rank"]).groupby("well", sort=False):
            group_mask = high.loc[group.index].to_numpy(bool)
            mask[group.index.to_numpy()] = keep_min_true_runs(group_mask, int(min_segment_rows))
    elif scope == "well":
        high_rate = high.groupby(frame["well"]).mean()
        eligible_wells = set(high_rate[high_rate >= float(min_well_high_rate)].index.astype(str))
        mask = frame["well"].astype(str).isin(eligible_wells).to_numpy(bool) & (
            frame["tail_rank"].to_numpy() >= int(min_tail_rank)
        )
    else:
        raise ValueError(f"unknown gate scope: {scope}")

    mask_series = pd.Series(mask, index=frame.index)
    meta = {
        "thresholds": thresholds,
        "scope": scope,
        "min_tail_rank": int(min_tail_rank),
        "min_conditions": int(min_conditions),
        "min_segment_rows": int(min_segment_rows),
        "min_well_high_rate": float(min_well_high_rate),
        "pre_scope_high_rows": int(high.sum()),
        "pre_scope_high_rate": float(high.mean()),
        "gate_rows": int(mask_series.sum()),
        "gate_rate": float(mask_series.mean()),
        "gate_wells": int(frame.loc[mask_series, "well"].nunique()),
    }
    return mask_series, meta


def apply_gate_variant(
    frame: pd.DataFrame, variant: dict[str, Any]
) -> tuple[pd.Series, dict[str, Any]]:
    candidate = str(variant["candidate"])
    if candidate not in frame.columns:
        raise ValueError(f"gate candidate not found: {candidate}")
    mask, mask_meta = build_target_free_gate_mask(
        frame,
        dense_q=float(variant.get("dense_q", 0.75)),
        pf_dense_q=float(variant.get("pf_dense_q", 0.75)),
        pf_beam_q=float(variant.get("pf_beam_q", 0.75)),
        min_tail_rank=int(variant.get("min_tail_rank", 500)),
        min_conditions=int(variant.get("min_conditions", 2)),
        scope=str(variant.get("scope", "segment")),
        min_segment_rows=int(variant.get("min_segment_rows", 25)),
        min_well_high_rate=float(variant.get("min_well_high_rate", 0.05)),
    )
    base = frame["pred_exp092_lgb1"].astype("float32")
    correction = frame[candidate].astype("float32") - base
    clip = variant.get("clip_abs")
    if clip is not None:
        correction = correction.clip(lower=-float(clip), upper=float(clip))
    alpha = float(variant.get("alpha", 1.0))
    pred = base.astype("float64").copy()
    pred.loc[mask] = base.loc[mask].astype("float64") + alpha * correction.loc[mask].astype(
        "float64"
    )
    meta = {
        **mask_meta,
        "candidate": candidate,
        "alpha": alpha,
        "clip_abs": None if clip is None else float(clip),
    }
    return pred.astype("float32"), meta


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
            base_by_well.rename("base_rmse"),
            left_on="well",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        by_well["delta_rmse_vs_exp092"] = by_well["rmse"] - by_well["base_rmse"]
        max_well_regression = float(by_well["delta_rmse_vs_exp092"].max())
        improved_wells = int((by_well["delta_rmse_vs_exp092"] < -1e-12).sum())
        worsened_wells = int((by_well["delta_rmse_vs_exp092"] > 1e-12).sum())
    else:
        by_well["delta_rmse_vs_exp092"] = 0.0
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
        "max_well_regression_vs_exp092": max_well_regression,
        "improved_wells_vs_exp092": improved_wells,
        "worsened_wells_vs_exp092": worsened_wells,
    }
    if gate_meta:
        summary.update(gate_meta)
    return summary, by_well


def build_common_worst_sets(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=False):
        well_rows.append(
            {
                "well": str(well),
                "exp092_rmse": rmse(group["pred_exp092_lgb1_error"]),
                "likpf_rmse": rmse(group["pred_likpf_mean_error"]),
            }
        )
    wells = pd.DataFrame(well_rows)
    wells["exp092_rank"] = wells["exp092_rmse"].rank(method="first", ascending=False)
    wells["likpf_rank"] = wells["likpf_rmse"].rank(method="first", ascending=False)
    wells["rank_sum"] = wells["exp092_rank"] + wells["likpf_rank"]
    pf_worst50 = wells.sort_values("likpf_rmse", ascending=False).head(50)["well"].tolist()
    exp092_worst50 = wells.sort_values("exp092_rmse", ascending=False).head(50)["well"].tolist()
    common = sorted(set(pf_worst50) & set(exp092_worst50))
    if len(common) < 26:
        filler = (
            wells[~wells["well"].isin(common)]
            .sort_values("rank_sum")
            .head(26 - len(common))["well"]
            .tolist()
        )
        common = common + filler
    common26 = (
        wells[wells["well"].isin(common)]
        .sort_values("rank_sum")
        .head(26)["well"]
        .astype(str)
        .tolist()
    )
    return wells, {
        "pf_likpf_worst50": [str(item) for item in pf_worst50],
        "exp092_worst50": [str(item) for item in exp092_worst50],
        "common_pf_ml_worst26": common26,
    }


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
            if "base_exp092_lgb1" in predictions:
                base_rmse = record["base_exp092_lgb1_rmse"]
                for name in predictions:
                    record[f"{name}_delta_rmse_vs_exp092"] = record[f"{name}_rmse"] - base_rmse
            rows.append(record)
    return pd.DataFrame(rows)


def summarize_common_worst(
    frame: pd.DataFrame,
    predictions: dict[str, pd.Series],
    worst_sets: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for set_name, wells in worst_sets.items():
        subset = frame[frame["well"].astype(str).isin(set(wells))]
        for name, pred in predictions.items():
            local_pred = pred.loc[subset.index]
            error = local_pred - subset["target_tvt"]
            abs_error = error.abs()
            rows.append(
                {
                    "set_name": set_name,
                    "variant": name,
                    "rows": int(len(subset)),
                    "wells": int(subset["well"].nunique()),
                    "rmse": rmse(error),
                    "mae": float(abs_error.mean()),
                    "within10": float((abs_error <= 10.0).mean()),
                }
            )
    result = pd.DataFrame(rows)
    base = result[result["variant"].eq("base_exp092_lgb1")][["set_name", "rmse"]].rename(
        columns={"rmse": "base_rmse"}
    )
    result = result.merge(base, on="set_name", how="left", validate="many_to_one")
    result["delta_rmse_vs_exp092"] = result["rmse"] - result["base_rmse"]
    return result


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


def render_readme(
    summary: dict[str, Any],
    metrics: pd.DataFrame,
    common_worst: pd.DataFrame,
    bucket_metrics: pd.DataFrame,
) -> str:
    best = metrics.sort_values("rmse").head(12)
    worst_sets = common_worst.sort_values(["set_name", "delta_rmse_vs_exp092"]).head(30)
    risky = bucket_metrics.sort_values("base_exp092_lgb1_rmse", ascending=False).head(12)
    lines = [
        "# exp135_tvt_dense_high_drift_confidence_gate_on_exp092",
        "",
        (
            "exp092 lgb1 を基準に、tvt_dense 系候補を high-drift / "
            "high-disagreement regime だけで低頻度に使う posthoc gate 診断。"
        ),
        "LightGBM の新規学習は行わない。",
        "",
        "## Overall",
        "",
        f"- rows: {summary['rows']}",
        f"- wells: {summary['wells']}",
        f"- base exp092 lgb1 RMSE: {summary['base']['rmse']:.9f}",
        f"- best RMSE variant: {summary['best']['variant']} / {summary['best']['rmse']:.9f}",
        f"- best delta vs exp092: {summary['best']['delta_rmse_vs_exp092']:.9f}",
        "",
        "## Best Variants",
        "",
        dataframe_to_markdown(best),
        "",
        "## Common Worst Readout",
        "",
        dataframe_to_markdown(worst_sets),
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
        f"- `{OUTPUT_PREFIX}_common_worst_metrics.csv`",
        f"- `{OUTPUT_PREFIX}_rawtest_parity_checklist.csv`",
        f"- `{OUTPUT_PREFIX}_prediction_sample.csv.gz`",
        f"- `{OUTPUT_PREFIX}_summary.json`",
    ]
    return "\n".join(lines) + "\n"


def rawtest_parity_checklist(config: dict[str, Any], feature_meta: dict[str, Any]) -> pd.DataFrame:
    required_columns = set(get_nested(config, "audit.required_rawtest_compatible_columns", []))
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
                "Gate masks use tail_rank/md_since and exp072 target-free "
                "candidate/disagreement columns only."
            ),
        },
        {
            "check": "new_lightgbm_training",
            "status": "pass",
            "detail": "No LightGBM training is performed in exp135.",
        },
        {
            "check": "inference_port",
            "status": "not_applicable",
            "detail": (
                "Diagnostic-only. A raw-test inference port must be implemented "
                "later if a gate passes."
            ),
        },
    ]
    return pd.DataFrame(rows)


def run_gate_audit(
    frame: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
    source_meta: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    well_reference, worst_sets = build_common_worst_sets(frame)
    frame["is_pf_likpf_worst50"] = frame["well"].astype(str).isin(worst_sets["pf_likpf_worst50"])
    frame["is_common_pf_ml_worst26"] = (
        frame["well"].astype(str).isin(worst_sets["common_pf_ml_worst26"])
    )

    predictions: dict[str, pd.Series] = {
        "base_exp092_lgb1": frame["pred_exp092_lgb1"].astype("float32"),
        "ref_exp073_lgb_mean": frame["pred_exp073_lgb_mean"].astype("float32"),
        "single_likpf_mean": frame["pred_likpf_mean"].astype("float32"),
        "single_tvt_dense": frame["pred_tvt_dense"].astype("float32"),
        "single_tvt_densew": frame["pred_tvt_densew"].astype("float32"),
        "single_tvt_dense50": frame["pred_tvt_dense50"].astype("float32"),
        "single_tvtF_ANCC": frame["pred_tvtF_ANCC"].astype("float32"),
        "oracle_all_candidates": (frame["target_tvt"] + frame["oracle_abs_error"]).astype(
            "float32"
        ),
        "oracle_dense_candidates": (frame["target_tvt"] + frame["dense_oracle_abs_error"]).astype(
            "float32"
        ),
    }
    gate_rows: list[dict[str, Any]] = []
    for variant in get_nested(config, "audit.gate_variants", []):
        name = str(variant["name"])
        pred, gate_meta = apply_gate_variant(frame, variant)
        predictions[name] = pred
        gate_rows.append({"variant": name, **gate_meta})

    base_error_by_well = frame.groupby("well", sort=False)["pred_exp092_lgb1_error"].apply(rmse)
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
    base_rmse = float(metrics.loc[metrics["variant"].eq("base_exp092_lgb1"), "rmse"].iloc[0])
    metrics["delta_rmse_vs_exp092"] = metrics["rmse"] - base_rmse
    by_well_all = pd.concat(by_well_frames, ignore_index=True)

    group_specs = [
        ("distance_bucket", ["distance_bucket"]),
        ("tail_rank_bucket", ["tail_rank_bucket"]),
        ("pf_dense_disagreement_bucket", ["pf_dense_disagreement_bucket"]),
        ("exp092_dense_disagreement_bucket", ["exp092_dense_disagreement_bucket"]),
        ("dense_std_bucket", ["dense_std_bucket"]),
        ("dense_drift_bucket", ["dense_drift_bucket"]),
        ("pf_beam_disagreement_bucket", ["pf_beam_disagreement_bucket"]),
        ("distance_x_pf_dense", ["distance_bucket", "pf_dense_disagreement_bucket"]),
        ("distance_x_dense_std", ["distance_bucket", "dense_std_bucket"]),
        ("common_pf_ml_worst26", ["is_common_pf_ml_worst26"]),
        ("pf_likpf_worst50", ["is_pf_likpf_worst50"]),
    ]
    bucket_metrics = summarize_buckets(frame, predictions, group_specs)
    common_worst = summarize_common_worst(frame, predictions, worst_sets)
    gate_variants = pd.DataFrame(gate_rows)
    parity = rawtest_parity_checklist(config, source_meta["feature_cache"])

    sample_n = int(get_nested(config, "audit.prediction_sample_rows", 200000))
    sample_cols = [
        "id",
        "well",
        "target_tvt",
        "tail_rank",
        "md_since",
        "distance_bucket",
        "pf_dense_disagreement_bucket",
        "dense_std_bucket",
        "dense_drift_bucket",
        "oracle_candidate",
        "dense_oracle_candidate",
        "pred_exp092_lgb1",
        "pred_likpf_mean",
        "pred_tvt_dense",
        "pred_tvt_densew",
        "pred_tvt_dense50",
        "pf_dense_abs_diff",
        "exp092_dense_abs_diff",
        "dense_std_abs",
        "tvt_dense_d_abs",
    ]
    prediction_sample = frame[sample_cols].head(sample_n).copy()

    paths = {
        "metrics": output_dir / f"{OUTPUT_PREFIX}_metrics.csv",
        "gate_variants": output_dir / f"{OUTPUT_PREFIX}_gate_variants.csv",
        "by_well": output_dir / f"{OUTPUT_PREFIX}_by_well.csv",
        "bucket_metrics": output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv",
        "common_worst": output_dir / f"{OUTPUT_PREFIX}_common_worst_metrics.csv",
        "well_reference": output_dir / f"{OUTPUT_PREFIX}_well_reference.csv",
        "rawtest_parity": output_dir / f"{OUTPUT_PREFIX}_rawtest_parity_checklist.csv",
        "prediction_sample": output_dir / f"{OUTPUT_PREFIX}_prediction_sample.csv.gz",
        "summary": output_dir / f"{OUTPUT_PREFIX}_summary.json",
        "readme": output_dir / "README.md",
    }
    metrics.to_csv(paths["metrics"], index=False)
    gate_variants.to_csv(paths["gate_variants"], index=False)
    by_well_all.to_csv(paths["by_well"], index=False)
    bucket_metrics.to_csv(paths["bucket_metrics"], index=False)
    common_worst.to_csv(paths["common_worst"], index=False)
    well_reference.to_csv(paths["well_reference"], index=False)
    parity.to_csv(paths["rawtest_parity"], index=False)
    prediction_sample.to_csv(paths["prediction_sample"], index=False, compression="gzip")

    best_row = metrics[~metrics["variant"].str.startswith("oracle_")].sort_values("rmse").iloc[0]
    base_row = metrics[metrics["variant"].eq("base_exp092_lgb1")].iloc[0]
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_train_side_posthoc_audit",
        "runtime_seconds": round(time.time() - started, 3),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "base": {
            "variant": "base_exp092_lgb1",
            "rmse": float(base_row["rmse"]),
            "mae": float(base_row["mae"]),
            "within10": float(base_row["within10"]),
        },
        "best": {
            "variant": str(best_row["variant"]),
            "rmse": float(best_row["rmse"]),
            "delta_rmse_vs_exp092": float(best_row["delta_rmse_vs_exp092"]),
            "max_well_regression_vs_exp092": float(best_row["max_well_regression_vs_exp092"]),
        },
        "oracle": {
            "all_candidates_rmse": float(
                metrics.loc[metrics["variant"].eq("oracle_all_candidates"), "rmse"].iloc[0]
            ),
            "dense_candidates_rmse": float(
                metrics.loc[metrics["variant"].eq("oracle_dense_candidates"), "rmse"].iloc[0]
            ),
            "dense_oracle_distribution": frame["dense_oracle_candidate"].value_counts().to_dict(),
        },
        "worst_sets": worst_sets,
        "source": source_meta,
        "outputs": {key: path.name for key, path in paths.items()},
    }
    with paths["summary"].open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
        fp.write("\n")
    paths["readme"].write_text(
        render_readme(summary, metrics, common_worst, bucket_metrics),
        encoding="utf-8",
    )
    return summary


def run_train_from_config(config: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
    predictions, prediction_meta = load_prediction_inputs(config)
    features, feature_meta = load_feature_cache(config, predictions["id"])
    features = features.rename(columns={"last_known_tvt": "last_known_tvt_cache"})
    frame = predictions.merge(features, on=["id", "well"], how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("No rows after joining predictions and exp072 feature cache")
    train_dir = get_nested(config, "data.train_dir", "data/raw/train")
    frame, raw_meta = add_raw_context(frame, train_dir)
    frame, _ = build_surface(frame)
    source_meta = {
        "predictions": prediction_meta,
        "feature_cache": feature_meta,
        "raw_context": raw_meta,
    }
    return run_gate_audit(frame, config, output_dir, source_meta)


def run_inference_from_config(config: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "not_selected_no_submission",
        "reason": "exp135 is a train-side posthoc gate audit only.",
        "inference_mode": get_nested(config, "inference.mode", "disabled_diagnostic_only"),
        "outputs": {},
    }
    with (output_dir / f"{OUTPUT_PREFIX}_inference_summary.json").open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
        fp.write("\n")
    return summary

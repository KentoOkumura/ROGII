from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_PREFIX = "exp154_segment_level_dense_candidate_verifier_on_exp148"

EXP072_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
EXP072_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
EXP073_PREDICTIONS = "exp063_full_replay_repro_guard_predictions.csv.gz"
EXP148_PREDICTIONS = "exp148_learned_likelihood_fulltrain_addonly_on_exp092_predictions.csv.gz"


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
    exp148_path = find_input_file(
        EXP148_PREDICTIONS,
        get_nested(config, "data.exp148_predictions"),
        local_roots=[
            Path(
                "/tmp/kaggle-output/exp148_learned_likelihood_fulltrain_addonly_on_exp092/train_v1"
            ),
            Path(
                "experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/kaggle/output/train_v1"
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
    exp148, exp148_meta = read_selected_prediction(
        exp148_path,
        selector_col="model",
        selector_value=str(get_nested(config, "model.exp148_model", "lgb_mean")),
        pred_col="pred_exp148_lgb_mean",
        max_rows=max_rows,
    )
    merged = exp148.merge(
        exp073[["id", "well", "pred_exp073_lgb_mean"]],
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Prediction inputs have no overlapping rows")
    merged["tail_rank"] = parse_tail_rank(merged["id"])
    return merged, {
        "exp148_lgb_mean": exp148_meta,
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
    out["base_dense_abs_diff"] = (out["pred_exp148_lgb_mean"] - out["pred_tvt_dense"]).abs()
    out["tvt_dense_d_abs"] = out["tvt_dense_d"].abs()
    out["likpf_mean_d_abs"] = out["likpf_mean_d"].abs()
    out["dense_std_abs"] = out["dense_std"].abs()
    out["pf_vs_dense_abs"] = out["pf_vs_dense"].abs()

    out["distance_bucket"] = distance_bucket(out["md_since"])
    out["tail_rank_bucket"] = tail_rank_bucket(out["tail_rank"])
    out["pf_dense_disagreement_bucket"] = safe_qcut(
        out["pf_dense_abs_diff"], 4, prefix="pf_dense_diff"
    )
    out["base_dense_disagreement_bucket"] = safe_qcut(
        out["base_dense_abs_diff"], 4, prefix="base_dense_diff"
    )
    out["dense_std_bucket"] = safe_qcut(out["dense_std_abs"], 4, prefix="dense_std")
    out["dense_drift_bucket"] = safe_qcut(out["tvt_dense_d_abs"], 4, prefix="dense_drift")
    out["pf_beam_disagreement_bucket"] = safe_qcut(
        out["pf_beam_abs_diff"], 4, prefix="pf_beam_diff"
    )

    candidate_cols = [
        "pred_exp148_lgb_mean",
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
        out[f"{col}_step_abs"] = (
            out.sort_values(["well", "tail_rank"])
            .groupby("well", sort=False)[col]
            .diff()
            .abs()
            .reindex(out.index)
            .fillna(0.0)
            .astype("float32")
        )
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


def keep_verified_segments(
    group: pd.DataFrame,
    high_mask: np.ndarray,
    *,
    candidate: str,
    min_len: int,
    max_near_md_since: float,
    max_candidate_step_p95: float | None,
    max_candidate_step_max: float | None,
) -> np.ndarray:
    high_mask = np.asarray(high_mask, dtype=bool)
    keep = np.zeros(len(group), dtype=bool)
    if len(group) == 0:
        return keep

    candidate_step = pd.to_numeric(
        group.get(f"{candidate}_step_abs", pd.Series(0.0, index=group.index)),
        errors="coerce",
    ).fillna(0.0)
    md_since = pd.to_numeric(group["md_since"], errors="coerce").fillna(0.0)

    start: int | None = None
    for idx, value in enumerate(high_mask):
        if value and start is None:
            start = idx
        if (not value or idx == len(high_mask) - 1) and start is not None:
            end = idx + 1 if value and idx == len(high_mask) - 1 else idx
            if end - start >= min_len:
                local_step = candidate_step.iloc[start:end]
                local_md = md_since.iloc[start:end]
                near_ok = bool((local_md > max_near_md_since).all())
                p95_ok = (
                    max_candidate_step_p95 is None
                    or float(local_step.quantile(0.95)) <= max_candidate_step_p95
                )
                max_ok = (
                    max_candidate_step_max is None
                    or float(local_step.max()) <= max_candidate_step_max
                )
                if near_ok and p95_ok and max_ok:
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
    candidate: str,
    max_near_md_since: float,
    max_candidate_step_p95: float | None,
    max_candidate_step_max: float | None,
) -> tuple[pd.Series, dict[str, Any]]:
    thresholds = {
        "dense_std_abs": finite_quantile(frame["dense_std_abs"], dense_q),
        "tvt_dense_d_abs": finite_quantile(frame["tvt_dense_d_abs"], dense_q),
        "pf_dense_abs_diff": finite_quantile(frame["pf_dense_abs_diff"], pf_dense_q),
        "base_dense_abs_diff": finite_quantile(frame["base_dense_abs_diff"], pf_dense_q),
        "pf_beam_abs_diff": finite_quantile(frame["pf_beam_abs_diff"], pf_beam_q),
    }
    condition_frame = pd.DataFrame(
        {
            "dense_std_high": frame["dense_std_abs"] >= thresholds["dense_std_abs"],
            "dense_drift_high": frame["tvt_dense_d_abs"] >= thresholds["tvt_dense_d_abs"],
            "pf_dense_diff_high": frame["pf_dense_abs_diff"] >= thresholds["pf_dense_abs_diff"],
            "base_dense_diff_high": frame["base_dense_abs_diff"]
            >= thresholds["base_dense_abs_diff"],
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
            verified = keep_verified_segments(
                group,
                group_mask,
                candidate=candidate,
                min_len=int(min_segment_rows),
                max_near_md_since=float(max_near_md_since),
                max_candidate_step_p95=max_candidate_step_p95,
                max_candidate_step_max=max_candidate_step_max,
            )
            mask[group.index.to_numpy()] = verified
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
        "max_near_md_since": float(max_near_md_since),
        "max_candidate_step_p95": max_candidate_step_p95,
        "max_candidate_step_max": max_candidate_step_max,
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
        candidate=candidate,
        max_near_md_since=float(variant.get("max_near_md_since", 50.0)),
        max_candidate_step_p95=None
        if variant.get("max_candidate_step_p95") is None
        else float(variant["max_candidate_step_p95"]),
        max_candidate_step_max=None
        if variant.get("max_candidate_step_max") is None
        else float(variant["max_candidate_step_max"]),
    )
    base = frame["pred_exp148_lgb_mean"].astype("float32")
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
        by_well["delta_rmse_vs_exp148"] = by_well["rmse"] - by_well["base_rmse"]
        max_well_regression = float(by_well["delta_rmse_vs_exp148"].max())
        improved_wells = int((by_well["delta_rmse_vs_exp148"] < -1e-12).sum())
        worsened_wells = int((by_well["delta_rmse_vs_exp148"] > 1e-12).sum())
    else:
        by_well["delta_rmse_vs_exp148"] = 0.0
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
        "max_well_regression_vs_exp148": max_well_regression,
        "improved_wells_vs_exp148": improved_wells,
        "worsened_wells_vs_exp148": worsened_wells,
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
                "exp148_rmse": rmse(group["pred_exp148_lgb_mean_error"]),
                "likpf_rmse": rmse(group["pred_likpf_mean_error"]),
            }
        )
    wells = pd.DataFrame(well_rows)
    wells["exp148_rank"] = wells["exp148_rmse"].rank(method="first", ascending=False)
    wells["likpf_rank"] = wells["likpf_rmse"].rank(method="first", ascending=False)
    wells["rank_sum"] = wells["exp148_rank"] + wells["likpf_rank"]
    pf_worst50 = wells.sort_values("likpf_rmse", ascending=False).head(50)["well"].tolist()
    exp148_worst50 = wells.sort_values("exp148_rmse", ascending=False).head(50)["well"].tolist()
    common = sorted(set(pf_worst50) & set(exp148_worst50))
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
        "exp148_worst50": [str(item) for item in exp148_worst50],
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
            if "base_exp148_lgb_mean" in predictions:
                base_rmse = record["base_exp148_lgb_mean_rmse"]
                for name in predictions:
                    record[f"{name}_delta_rmse_vs_exp148"] = record[f"{name}_rmse"] - base_rmse
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
    base = result[result["variant"].eq("base_exp148_lgb_mean")][["set_name", "rmse"]].rename(
        columns={"rmse": "base_rmse"}
    )
    result = result.merge(base, on="set_name", how="left", validate="many_to_one")
    result["delta_rmse_vs_exp148"] = result["rmse"] - result["base_rmse"]
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
    worst_sets = common_worst.sort_values(["set_name", "delta_rmse_vs_exp148"]).head(30)
    risky = bucket_metrics.sort_values("base_exp148_lgb_mean_rmse", ascending=False).head(12)
    lines = [
        "# exp154_segment_level_dense_candidate_verifier_on_exp148",
        "",
        (
            "exp148 lgb_mean を基準に、tvt_dense 系候補を high-drift / "
            "high-disagreement regime だけで低頻度に使う posthoc gate 診断。"
        ),
        "LightGBM の新規学習は行わない。",
        "",
        "## Overall",
        "",
        f"- rows: {summary['rows']}",
        f"- wells: {summary['wells']}",
        f"- base exp148 lgb_mean RMSE: {summary['base']['rmse']:.9f}",
        f"- best RMSE variant: {summary['best']['variant']} / {summary['best']['rmse']:.9f}",
        f"- best delta vs exp148: {summary['best']['delta_rmse_vs_exp148']:.9f}",
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
            "detail": "No LightGBM training is performed in exp154.",
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
        "base_exp148_lgb_mean": frame["pred_exp148_lgb_mean"].astype("float32"),
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

    base_error_by_well = frame.groupby("well", sort=False)["pred_exp148_lgb_mean_error"].apply(rmse)
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
    base_rmse = float(metrics.loc[metrics["variant"].eq("base_exp148_lgb_mean"), "rmse"].iloc[0])
    metrics["delta_rmse_vs_exp148"] = metrics["rmse"] - base_rmse
    by_well_all = pd.concat(by_well_frames, ignore_index=True)

    group_specs = [
        ("distance_bucket", ["distance_bucket"]),
        ("tail_rank_bucket", ["tail_rank_bucket"]),
        ("pf_dense_disagreement_bucket", ["pf_dense_disagreement_bucket"]),
        ("base_dense_disagreement_bucket", ["base_dense_disagreement_bucket"]),
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
        "pred_exp148_lgb_mean",
        "pred_likpf_mean",
        "pred_tvt_dense",
        "pred_tvt_densew",
        "pred_tvt_dense50",
        "pf_dense_abs_diff",
        "base_dense_abs_diff",
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
    base_row = metrics[metrics["variant"].eq("base_exp148_lgb_mean")].iloc[0]
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_train_side_posthoc_audit",
        "runtime_seconds": round(time.time() - started, 3),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "base": {
            "variant": "base_exp148_lgb_mean",
            "rmse": float(base_row["rmse"]),
            "mae": float(base_row["mae"]),
            "within10": float(base_row["within10"]),
        },
        "best": {
            "variant": str(best_row["variant"]),
            "rmse": float(best_row["rmse"]),
            "delta_rmse_vs_exp148": float(best_row["delta_rmse_vs_exp148"]),
            "max_well_regression_vs_exp148": float(best_row["max_well_regression_vs_exp148"]),
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


def build_inference_surface(frame: pd.DataFrame, pred_exp148: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out["id"] = out["id"].astype(str)
    out["well"] = out["well"].astype(str)
    out["tail_rank"] = parse_tail_rank(out["id"])
    out["pred_exp148_lgb_mean"] = np.asarray(pred_exp148, dtype=np.float32)

    numeric_columns = [
        "last_known_tvt",
        "md_since",
        "pf_ancc",
        "pf_z",
        "beam_mean_d",
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
    missing = [col for col in numeric_columns if col not in out.columns]
    if missing:
        raise ValueError(f"current-test replay frame missing verifier columns: {missing}")
    for col in numeric_columns:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float32")

    base = out["last_known_tvt"].astype("float32")
    out["pred_likpf_mean"] = base + out["likpf_mean_d"]
    out["pred_pf_ancc"] = out["pf_ancc"]
    out["pred_pf_z"] = out["pf_z"]
    out["pred_beam_mean"] = base + out["beam_mean_d"]
    out["pred_tvt_dense"] = base + out["tvt_dense_d"]
    out["pred_tvt_densew"] = base + out["tvt_densew_d"]
    out["pred_tvt_dense50"] = base + out["tvt_dense50_d"]
    out["pred_tvtF_ANCC"] = out["tvtF_ANCC"]

    out["pf_beam_abs_diff"] = (out["pred_pf_ancc"] - out["pred_beam_mean"]).abs()
    out["pf_dense_abs_diff"] = (out["pred_likpf_mean"] - out["pred_tvt_dense"]).abs()
    out["base_dense_abs_diff"] = (out["pred_exp148_lgb_mean"] - out["pred_tvt_dense"]).abs()
    out["tvt_dense_d_abs"] = out["tvt_dense_d"].abs()
    out["likpf_mean_d_abs"] = out["likpf_mean_d"].abs()
    out["dense_std_abs"] = out["dense_std"].abs()
    out["pf_vs_dense_abs"] = out["pf_vs_dense"].abs()

    out["distance_bucket"] = distance_bucket(out["md_since"])
    out["tail_rank_bucket"] = tail_rank_bucket(out["tail_rank"])
    out["pf_dense_disagreement_bucket"] = safe_qcut(
        out["pf_dense_abs_diff"], 4, prefix="pf_dense_diff"
    )
    out["base_dense_disagreement_bucket"] = safe_qcut(
        out["base_dense_abs_diff"], 4, prefix="base_dense_diff"
    )
    out["dense_std_bucket"] = safe_qcut(out["dense_std_abs"], 4, prefix="dense_std")
    out["dense_drift_bucket"] = safe_qcut(out["tvt_dense_d_abs"], 4, prefix="dense_drift")
    out["pf_beam_disagreement_bucket"] = safe_qcut(
        out["pf_beam_abs_diff"], 4, prefix="pf_beam_diff"
    )

    candidate_cols = [
        "pred_exp148_lgb_mean",
        "pred_likpf_mean",
        "pred_pf_ancc",
        "pred_pf_z",
        "pred_beam_mean",
        "pred_tvt_dense",
        "pred_tvt_densew",
        "pred_tvt_dense50",
        "pred_tvtF_ANCC",
    ]
    ordered = out.sort_values(["well", "tail_rank"])
    for col in candidate_cols:
        out[f"{col}_step_abs"] = (
            ordered.groupby("well", sort=False)[col]
            .diff()
            .abs()
            .reindex(out.index)
            .fillna(0.0)
            .astype("float32")
        )
    return out


def run_exp148_base_prediction_for_verifier(
    config: dict[str, Any],
    *,
    output_dir: Path,
    sample_submission_path: str | Path,
    raw_data_dir: str | Path,
    test_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    import lightgbm as lgb
    from learned_likelihood_fulltrain_addonly_on_exp092 import (
        add_inference_anchor_columns,
        build_learned_likelihood_features,
        build_u_projection_features,
        feature_columns_for_variant,
        find_model_manifest,
        generate_current_test_learned_likelihood_ml_features,
        learned_feature_keys_match,
        load_learned_likelihood_ml_features,
        prediction_sha256,
    )
    from public_notebook_replay_audit import build_replay_test_frame, configure_public_runtime

    started = time.time()
    exp148_cfg = get_nested(config, "inference.exp148", {}) or {}
    variant_name = str(
        exp148_cfg.get("selected_variant") or "learned_likelihood_confidence_addonly"
    )
    mode_name = str(exp148_cfg.get("selected_mode") or "gpu_repro_guard_dp_threads8")
    model_name = str(exp148_cfg.get("selected_model") or "lgb_mean")
    manifest_path = find_model_manifest(exp148_cfg.get("model_manifest_path"))
    manifest = json.loads(manifest_path.read_text())
    model_root = manifest_path.parent

    configure_public_runtime(
        data_dir=raw_data_dir,
        output_dir=output_dir,
        n_jobs=int(get_nested(config, "generator.rawtest_replay.n_jobs", 8)),
        pf_seeds=int(get_nested(config, "generator.rawtest_replay.pf_seeds", 128)),
        pf_particles=int(get_nested(config, "generator.rawtest_replay.pf_particles", 500)),
        fast=bool(get_nested(config, "generator.rawtest_replay.fast", False)),
        use_gpu=str(get_nested(config, "generator.rawtest_replay.use_gpu", "auto")),
    )
    base_test_frame, test_meta = build_replay_test_frame()
    base_test_frame["id"] = base_test_frame["id"].astype(str)
    base_test_frame["well"] = base_test_frame["well"].astype(str)

    base_feature_columns = [str(col) for col in manifest["feature_source"]["feature_columns"]]
    missing_base = sorted(set(base_feature_columns) - set(base_test_frame.columns))
    if missing_base:
        raise ValueError(f"raw-test replay frame is missing base features: {missing_base[:40]}")

    anchored_frame, anchor_meta = add_inference_anchor_columns(base_test_frame, test_dir)
    projection_config = dict(manifest.get("projection_config") or {})
    projection_features, projection_group_columns, projection_summary = build_u_projection_features(
        anchored_frame,
        source_specs=dict(projection_config.get("sources") or {}),
        degree=int(projection_config.get("degree", 3)),
        robust_iters=int(projection_config.get("robust_iters", 3)),
        clip_sigma=float(projection_config.get("clip_sigma", 4.0)),
    )
    configured_projection_groups = manifest.get("projection_feature_groups") or {}
    if configured_projection_groups and {
        key: list(value) for key, value in projection_group_columns.items()
    } != {key: list(value) for key, value in configured_projection_groups.items()}:
        raise ValueError("Projection feature groups differ from exp148 train manifest")

    projection_feature_columns = [
        col for col in projection_features.columns if col not in {"id", "well"}
    ]
    test_frame = pd.concat(
        [
            anchored_frame.reset_index(drop=True),
            projection_features[projection_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )

    try:
        rawtest_learned_features, rawtest_learned_meta = load_learned_likelihood_ml_features(
            None,
            schema_path=get_nested(config, "data.learned_likelihood_rawtest_feature_schema_local"),
            summary_path=get_nested(config, "data.learned_likelihood_rawtest_summary_local"),
            feature_filename=str(
                exp148_cfg.get("rawtest_feature_filename")
                or (
                    "exp145_learned_likelihood_rawtest_feature_generator_parity_"
                    "rawtest_ml_features.csv.gz"
                )
            ),
            source_kind="target_free_rawtest_learned_likelihood_ml_features",
        )
    except FileNotFoundError:
        rawtest_learned_features, rawtest_learned_meta = (
            generate_current_test_learned_likelihood_ml_features(
                test_frame=anchored_frame,
                output_dir=output_dir,
            )
        )
    else:
        if not learned_feature_keys_match(rawtest_learned_features, anchored_frame):
            rawtest_learned_features, rawtest_learned_meta = (
                generate_current_test_learned_likelihood_ml_features(
                    test_frame=anchored_frame,
                    output_dir=output_dir,
                )
            )

    learned_features, learned_group_columns, learned_summary = build_learned_likelihood_features(
        rawtest_learned_features,
        test_frame,
        dict(manifest.get("learned_feature_config") or {}),
    )
    before_join_rows = len(test_frame)
    test_frame = test_frame.merge(
        learned_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if len(test_frame) != before_join_rows:
        raise ValueError(
            "Raw-test learned likelihood features do not cover every replay test row: "
            f"{len(test_frame)} of {before_join_rows}"
        )

    configured_learned_groups = manifest.get("learned_feature_groups") or {}
    if configured_learned_groups and {
        key: list(value) for key, value in learned_group_columns.items()
    } != {key: list(value) for key, value in configured_learned_groups.items()}:
        raise ValueError("Learned likelihood feature groups differ from exp148 train manifest")

    variant_configs = {
        str(item["name"]): dict(item)
        for item in manifest.get("variants", [])
        if item.get("enabled", True)
    }
    if variant_name not in variant_configs:
        raise ValueError(f"variant={variant_name} not found in exp148 train manifest")
    feature_group_columns = {**projection_group_columns, **learned_group_columns}
    feature_columns = feature_columns_for_variant(
        base_feature_columns,
        feature_group_columns,
        variant_configs[variant_name],
    )
    missing_model = sorted(set(feature_columns) - set(test_frame.columns))
    if missing_model:
        raise ValueError(f"test frame is missing exp148 model features: {missing_model[:40]}")
    for col in feature_columns:
        test_frame[col] = pd.to_numeric(test_frame[col], errors="raise").astype(np.float32)
    if not np.isfinite(test_frame[feature_columns].to_numpy(np.float32)).all():
        raise ValueError("exp148 test feature matrix contains non-finite values")

    model_rows = [
        item
        for item in manifest.get("models", [])
        if str(item.get("variant")) == variant_name
        and str(item.get("mode")) == mode_name
        and (model_name == "lgb_mean" or str(item.get("model")) == model_name)
    ]
    if not model_rows:
        raise ValueError(
            f"No exp148 saved models for variant={variant_name} mode={mode_name} model={model_name}"
        )

    x_matrix = test_frame[feature_columns].to_numpy(np.float32)
    pred_delta = np.zeros(len(test_frame), dtype=np.float32)
    loaded_rows: list[dict[str, Any]] = []
    for item in model_rows:
        model_file = model_root / str(item["file"])
        booster = lgb.Booster(model_file=str(model_file))
        pred_delta += booster.predict(x_matrix).astype(np.float32) / float(len(model_rows))
        loaded_rows.append(
            {
                "variant": item.get("variant"),
                "mode": item.get("mode"),
                "model": item.get("model"),
                "fold": item.get("fold"),
                "file": str(item.get("file")),
                "sha256": item.get("sha256"),
            }
        )

    pred_exp148 = (test_frame["last_known_tvt"].to_numpy(np.float32) + pred_delta).astype(
        np.float32
    )
    predictions = pd.DataFrame(
        {
            "id": test_frame["id"].to_numpy(),
            "well": test_frame["well"].to_numpy(),
            "pred_delta_exp148": pred_delta,
            "pred_exp148_lgb_mean": pred_exp148,
        }
    )
    sample = pd.read_csv(sample_submission_path, dtype={"id": str})
    pred_map = dict(
        zip(
            predictions["id"].astype(str),
            predictions["pred_exp148_lgb_mean"],
            strict=False,
        )
    )
    missing_rows = int(sample["id"].astype(str).map(pred_map).isna().sum())

    learned_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_learned_feature_summary.csv",
        index=False,
    )
    projection_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_projection_feature_summary.csv",
        index=False,
    )
    learned_feature_columns = [col for col in learned_features.columns if col not in {"id", "well"}]
    pd.DataFrame(
        [
            {
                "feature_index": int(index),
                "feature": feature,
                "is_projection_feature": bool(feature in projection_feature_columns),
                "is_learned_likelihood_feature": bool(feature in learned_feature_columns),
            }
            for index, feature in enumerate(feature_columns)
        ]
    ).to_csv(output_dir / f"{OUTPUT_PREFIX}_inference_exp148_feature_schema.csv", index=False)

    meta = {
        "variant": variant_name,
        "mode": mode_name,
        "model": model_name,
        "model_count": int(len(model_rows)),
        "model_manifest": str(manifest_path),
        "test_feature_source": test_meta,
        "anchor_source": anchor_meta,
        "rawtest_learned_likelihood_feature_source": rawtest_learned_meta,
        "loaded_models": loaded_rows,
        "feature_count": int(len(feature_columns)),
        "test_rows": int(len(test_frame)),
        "sample_missing_rows_before_verifier": missing_rows,
        "prediction_sha256": prediction_sha256(
            predictions["id"],
            pred_delta,
            label=f"{variant_name}/{mode_name}/{model_name}/test",
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return test_frame, predictions, meta


def run_inference_from_config(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    submission_path: str | Path | None = None,
    sample_submission_path: str | Path | None = None,
    raw_data_dir: str | Path | None = None,
    test_dir: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if submission_path is None:
        submission_path = Path("submission.csv")
    submission_path = Path(submission_path)
    if sample_submission_path is None:
        sample_submission_path = get_nested(
            config,
            "data.sample_submission",
            "data/raw/sample_submission.csv",
        )
    if raw_data_dir is None:
        raw_data_dir = get_nested(config, "data.raw_dir", "data/raw")
    if test_dir is None:
        test_dir = get_nested(config, "data.test_dir", "data/raw/test")

    selected_variant_name = str(
        get_nested(
            config,
            "inference.selected_variant",
            "verifier_dense50_tail1500_q90_min80_clip10_a025",
        )
    )
    variants = {
        str(variant["name"]): dict(variant)
        for variant in get_nested(config, "audit.gate_variants", [])
    }
    if selected_variant_name not in variants:
        raise ValueError(f"selected inference variant not found: {selected_variant_name}")

    test_frame, base_predictions, exp148_meta = run_exp148_base_prediction_for_verifier(
        config,
        output_dir=output_dir,
        sample_submission_path=sample_submission_path,
        raw_data_dir=raw_data_dir,
        test_dir=test_dir,
    )
    surface = build_inference_surface(
        test_frame,
        base_predictions["pred_exp148_lgb_mean"].to_numpy(np.float32),
    )
    verifier_pred, gate_meta = apply_gate_variant(surface, variants[selected_variant_name])
    surface["gate_selected"] = build_target_free_gate_mask(
        surface,
        dense_q=float(variants[selected_variant_name].get("dense_q", 0.75)),
        pf_dense_q=float(variants[selected_variant_name].get("pf_dense_q", 0.75)),
        pf_beam_q=float(variants[selected_variant_name].get("pf_beam_q", 0.75)),
        min_tail_rank=int(variants[selected_variant_name].get("min_tail_rank", 500)),
        min_conditions=int(variants[selected_variant_name].get("min_conditions", 2)),
        scope=str(variants[selected_variant_name].get("scope", "segment")),
        min_segment_rows=int(variants[selected_variant_name].get("min_segment_rows", 25)),
        min_well_high_rate=float(variants[selected_variant_name].get("min_well_high_rate", 0.05)),
        candidate=str(variants[selected_variant_name]["candidate"]),
        max_near_md_since=float(variants[selected_variant_name].get("max_near_md_since", 50.0)),
        max_candidate_step_p95=None
        if variants[selected_variant_name].get("max_candidate_step_p95") is None
        else float(variants[selected_variant_name]["max_candidate_step_p95"]),
        max_candidate_step_max=None
        if variants[selected_variant_name].get("max_candidate_step_max") is None
        else float(variants[selected_variant_name]["max_candidate_step_max"]),
    )[0]
    surface["pred_exp154_verified"] = verifier_pred.to_numpy(np.float32)
    surface["delta_vs_exp148"] = (
        surface["pred_exp154_verified"] - surface["pred_exp148_lgb_mean"]
    ).astype("float32")

    sample = pd.read_csv(sample_submission_path, dtype={"id": str})
    target_column = str(get_nested(config, "data.submission_target_column", "tvt"))
    if target_column not in sample.columns:
        target_column = str(sample.columns[1])
    pred_map = dict(
        zip(
            surface["id"].astype(str),
            surface["pred_exp154_verified"],
            strict=False,
        )
    )
    mapped = sample["id"].astype(str).map(pred_map)
    fallback = float(surface["pred_exp148_lgb_mean"].mean())
    missing_mask = mapped.isna()
    sample[target_column] = mapped.fillna(fallback).astype("float64")
    sample.to_csv(submission_path, index=False)

    prediction_path = output_dir / f"{OUTPUT_PREFIX}_inference_test_predictions.csv.gz"
    surface[
        [
            "id",
            "well",
            "tail_rank",
            "md_since",
            "pred_exp148_lgb_mean",
            "pred_tvt_dense50",
            "pred_tvt_densew",
            "pred_exp154_verified",
            "delta_vs_exp148",
            "gate_selected",
            "pf_dense_abs_diff",
            "base_dense_abs_diff",
            "dense_std_abs",
            "tvt_dense_d_abs",
        ]
    ].to_csv(prediction_path, index=False, compression="gzip")

    changed = surface["delta_vs_exp148"].abs() > 1e-7
    metrics = {
        "selected_variant": selected_variant_name,
        "rows": int(len(surface)),
        "wells": int(surface["well"].nunique()),
        "submission_rows": int(len(sample)),
        "fallback_rows": int(missing_mask.sum()),
        "changed_rows_vs_exp148": int(changed.sum()),
        "changed_rate_vs_exp148": float(changed.mean()),
        "changed_wells_vs_exp148": int(surface.loc[changed, "well"].nunique()),
        "delta_vs_exp148_abs_mean": float(surface["delta_vs_exp148"].abs().mean()),
        "delta_vs_exp148_abs_p95": float(surface["delta_vs_exp148"].abs().quantile(0.95)),
        "delta_vs_exp148_abs_max": float(surface["delta_vs_exp148"].abs().max()),
        "prediction_min": float(sample[target_column].min()),
        "prediction_max": float(sample[target_column].max()),
        "prediction_mean": float(sample[target_column].mean()),
        "prediction_std": float(sample[target_column].std()),
        "submission_sha256": sha256_file(submission_path),
    }
    pd.DataFrame([metrics]).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_metrics.csv",
        index=False,
    )
    pd.DataFrame([gate_meta]).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_gate_meta.csv",
        index=False,
    )
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "inference_completed_submission_created",
        "inference_mode": get_nested(
            config,
            "inference.mode",
            "exp148_saved_booster_plus_dense_verifier",
        ),
        "selected_variant": selected_variant_name,
        "exp148_base": exp148_meta,
        "gate": gate_meta,
        "metrics": metrics,
        "outputs": {
            "submission": str(submission_path),
            "predictions": prediction_path.name,
            "metrics": f"{OUTPUT_PREFIX}_inference_metrics.csv",
            "gate_meta": f"{OUTPUT_PREFIX}_inference_gate_meta.csv",
            "summary": f"{OUTPUT_PREFIX}_inference_summary.json",
        },
    }
    with (output_dir / f"{OUTPUT_PREFIX}_inference_summary.json").open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
        fp.write("\n")
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary

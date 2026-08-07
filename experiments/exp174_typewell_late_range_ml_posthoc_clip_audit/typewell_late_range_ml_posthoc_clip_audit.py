from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_PREFIX = "exp174_typewell_late_range_ml_posthoc_clip_audit"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")

PREDICTION_USECOLS = {
    "id",
    "well",
    "variant",
    "mode",
    "model",
    "target",
    "target_tvt",
    "last_known_tvt",
    "pred_target",
    "pred_delta",
    "pred_tvt",
}


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_gzip_decompressed(path: Path) -> str:
    hasher = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def artifact_sha(path: Path) -> dict[str, Any]:
    meta = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    if path.suffix == ".gz":
        meta["decompressed_sha256"] = sha256_gzip_decompressed(path)
    return meta


def find_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    required: bool = True,
) -> Path | None:
    candidates: list[Path] = []
    if explicit_path:
        path = Path(explicit_path)
        candidates.extend([path, Path.cwd() / path])
    candidates.extend([Path(filename), Path.cwd() / filename])
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    if required:
        checked = "\n".join(str(path) for path in candidates[:100])
        raise FileNotFoundError(f"Artifact not found: {filename}\nChecked:\n{checked}")
    return None


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def within10(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true) <= 10.0))


def pct_label(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def tail_rank(ids: pd.Series) -> pd.Series:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return pd.cut(
        series,
        bins=[-np.inf, 50, 100, 250, 500, 1000, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
    )


def pct_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return pd.cut(
        series,
        bins=[-np.inf, 0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, np.inf],
        labels=[
            "lt_0",
            "0_0p5",
            "0p5_0p6",
            "0p6_0p7",
            "0p7_0p75",
            "0p75_0p8",
            "0p8_0p9",
            "0p9_1p0",
            "gt_1",
        ],
    )


def read_typewell_context(
    train_dir: str | Path,
    test_dir: str | Path | None,
    *,
    min_typewell_span: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    split_dirs = [("train", Path(train_dir))]
    if test_dir is not None:
        split_dirs.append(("test", Path(test_dir)))
    for split, directory in split_dirs:
        for typewell_path in sorted(directory.glob("*__typewell.csv")):
            well = typewell_path.name.replace("__typewell.csv", "")
            horizontal_path = directory / f"{well}__horizontal_well.csv"
            if not horizontal_path.exists():
                continue
            typewell = pd.read_csv(typewell_path, usecols=["TVT"])
            horizontal = pd.read_csv(
                horizontal_path,
                usecols=lambda col: col in {"MD", "TVT", "TVT_input"},
            )
            tvt = pd.to_numeric(typewell["TVT"], errors="coerce").dropna()
            if tvt.empty:
                continue
            typewell_min = float(tvt.min())
            typewell_max = float(tvt.max())
            typewell_span = float(typewell_max - typewell_min)
            known = horizontal[pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna()]
            eval_rows = horizontal[pd.to_numeric(horizontal["TVT_input"], errors="coerce").isna()]
            if known.empty:
                last_known_tvt = np.nan
                last_known_md = np.nan
            else:
                last_known_tvt = float(known["TVT_input"].iloc[-1])
                last_known_md = float(known["MD"].iloc[-1]) if "MD" in known else np.nan
            known_last_pct = (
                (last_known_tvt - typewell_min) / typewell_span
                if typewell_span >= min_typewell_span and np.isfinite(last_known_tvt)
                else np.nan
            )
            rows.append(
                {
                    "split": split,
                    "well": well,
                    "typewell_min": typewell_min,
                    "typewell_max": typewell_max,
                    "typewell_span": typewell_span,
                    "last_known_tvt_raw": last_known_tvt,
                    "last_known_md_raw": last_known_md,
                    "known_last_pct": known_last_pct,
                    "known_rows": int(len(known)),
                    "eval_rows": int(len(eval_rows)),
                    "valid_typewell_span": bool(typewell_span >= min_typewell_span),
                }
            )
    context = pd.DataFrame(rows)
    if context.empty:
        raise FileNotFoundError(f"No typewell context rows found under {train_dir}")
    summary = (
        context.groupby("split", as_index=False)
        .agg(
            wells=("well", "nunique"),
            eval_rows=("eval_rows", "sum"),
            known_last_pct_min=("known_last_pct", "min"),
            known_last_pct_median=("known_last_pct", "median"),
            known_last_pct_max=("known_last_pct", "max"),
            known_last_pct_ge_0p75=("known_last_pct", lambda s: int((s >= 0.75).sum())),
            known_last_pct_lt_0p5=("known_last_pct", lambda s: int((s < 0.5).sum())),
        )
        .sort_values("split")
    )
    return context, summary


def normalize_prediction_frame(frame: pd.DataFrame, source: dict[str, Any]) -> pd.DataFrame:
    frame = frame.copy()
    for col in ["id", "well"]:
        if col not in frame.columns:
            raise ValueError(f"Prediction source {source.get('name')} missing required column: {col}")
        frame[col] = frame[col].astype(str)
    if source.get("variant") is not None and "variant" in frame.columns:
        frame = frame[frame["variant"].astype(str).eq(str(source["variant"]))].copy()
    if source.get("model") is not None and "model" in frame.columns:
        frame = frame[frame["model"].astype(str).eq(str(source["model"]))].copy()
    if frame.empty:
        raise ValueError(f"Prediction source {source.get('name')} is empty after filters")

    if "last_known_tvt" not in frame.columns:
        raise ValueError(f"Prediction source {source.get('name')} missing last_known_tvt")
    frame["last_known_tvt"] = numeric_series(frame, "last_known_tvt")

    if "target_tvt" not in frame.columns:
        if "target" not in frame.columns:
            raise ValueError(f"Prediction source {source.get('name')} missing target_tvt/target")
        frame["target_tvt"] = frame["last_known_tvt"] + numeric_series(frame, "target")
    frame["target_tvt"] = numeric_series(frame, "target_tvt")

    if "pred_tvt" not in frame.columns:
        if "pred_delta" in frame.columns:
            frame["pred_tvt"] = frame["last_known_tvt"] + numeric_series(frame, "pred_delta")
        elif "pred_target" in frame.columns:
            frame["pred_tvt"] = frame["last_known_tvt"] + numeric_series(frame, "pred_target")
        else:
            raise ValueError(f"Prediction source {source.get('name')} missing pred_tvt/pred_delta")
    frame["pred_tvt"] = numeric_series(frame, "pred_tvt")
    if "variant" not in frame.columns:
        frame["variant"] = str(source.get("variant") or "unknown_variant")
    if "mode" not in frame.columns:
        frame["mode"] = "unknown_mode"
    if "model" not in frame.columns:
        frame["model"] = str(source.get("model") or "unknown_model")
    frame["source"] = str(source["name"])
    return frame[
        [
            "source",
            "id",
            "well",
            "variant",
            "mode",
            "model",
            "last_known_tvt",
            "target_tvt",
            "pred_tvt",
        ]
    ].reset_index(drop=True)


def load_prediction_source(source: dict[str, Any]) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    path = find_artifact(
        str(source["filename"]),
        source.get("local_path"),
        required=bool(source.get("required", False)),
    )
    if path is None:
        return None, {
            "name": source.get("name"),
            "status": "missing_optional",
            "filename": source.get("filename"),
        }
    frame = pd.read_csv(
        path,
        usecols=lambda col: col in PREDICTION_USECOLS,
        dtype={"id": str, "well": str},
    )
    frame = normalize_prediction_frame(frame, source)
    meta = {
        "name": source.get("name"),
        "status": "loaded",
        "path": str(path),
        "rows_after_filter": int(len(frame)),
        "wells_after_filter": int(frame["well"].nunique()),
        "variant_filter": source.get("variant"),
        "model_filter": source.get("model"),
        "sha": artifact_sha(path),
    }
    return frame, meta


def enrich_predictions(predictions: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    train_context = context[context["split"].eq("train")].drop(columns=["split"])
    frame = predictions.merge(train_context, on="well", how="left", validate="many_to_one")
    missing = frame["typewell_span"].isna()
    if missing.any():
        missing_wells = sorted(frame.loc[missing, "well"].unique())[:20]
        raise ValueError(f"Missing typewell context for prediction wells: {missing_wells}")
    frame["tail_rank"] = tail_rank(frame["id"])
    frame["distance_bucket"] = distance_bucket(frame["tail_rank"])
    span = frame["typewell_span"].to_numpy(float)
    typewell_min = frame["typewell_min"].to_numpy(float)
    frame["pred_pct"] = (frame["pred_tvt"].to_numpy(float) - typewell_min) / span
    frame["target_pct"] = (frame["target_tvt"].to_numpy(float) - typewell_min) / span
    frame["last_known_pct_from_prediction"] = (
        frame["last_known_tvt"].to_numpy(float) - typewell_min
    ) / span
    frame["known_last_pct_bucket"] = pct_bucket(frame["known_last_pct"])
    frame["pred_pct_bucket"] = pct_bucket(frame["pred_pct"])
    frame["target_pct_bucket"] = pct_bucket(frame["target_pct"])
    frame["test_like_known_last"] = frame["known_last_pct"] >= 0.75
    frame["baseline_front_half_pred"] = frame["pred_pct"] < 0.5
    frame["front_half_target"] = frame["target_pct"] < 0.5
    return frame


def prediction_metrics(
    frame: pd.DataFrame,
    pred: np.ndarray,
    *,
    source: str,
    policy: str,
    baseline_rmse: float,
    spec: dict[str, Any],
    changed_mask: np.ndarray | None = None,
    lower_bound_pct: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = frame["target_tvt"].to_numpy(float)
    rows = int(len(frame))
    changed_mask = np.zeros(rows, dtype=bool) if changed_mask is None else changed_mask
    if changed_mask.any():
        changed_delta = np.abs(pred[changed_mask] - frame["pred_tvt"].to_numpy(float)[changed_mask])
        changed_mean = float(changed_delta.mean())
        changed_p95 = float(np.quantile(changed_delta, 0.95))
    else:
        changed_mean = 0.0
        changed_p95 = 0.0
    if lower_bound_pct is None:
        lower_mean = np.nan
        lower_min = np.nan
        lower_max = np.nan
    else:
        lower_mean = float(np.nanmean(lower_bound_pct))
        lower_min = float(np.nanmin(lower_bound_pct))
        lower_max = float(np.nanmax(lower_bound_pct))
    score = rmse(y_true, pred)
    return {
        "source": source,
        "policy": policy,
        "lower_bound_kind": spec.get("lower_bound_kind"),
        "lower_bound_value": spec.get("lower_bound_value"),
        "known_last_margin": spec.get("known_last_margin"),
        "known_last_pct_min": spec.get("known_last_pct_min"),
        "alpha": spec.get("alpha"),
        "rows": rows,
        "wells": int(frame["well"].nunique()),
        "changed_rows": int(changed_mask.sum()),
        "changed_wells": int(frame.loc[changed_mask, "well"].nunique()) if changed_mask.any() else 0,
        "changed_row_rate": float(changed_mask.mean()),
        "changed_delta_abs_mean": changed_mean,
        "changed_delta_abs_p95": changed_p95,
        "lower_bound_pct_mean": lower_mean,
        "lower_bound_pct_min": lower_min,
        "lower_bound_pct_max": lower_max,
        "rmse_tvt": score,
        "rmse_delta_vs_baseline": float(score - baseline_rmse),
        "mae_tvt": mae(y_true, pred),
        "within10": within10(y_true, pred),
    }


def make_policy_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    posthoc = get_nested(config, "model.posthoc", {})
    thresholds = [float(v) for v in posthoc.get("known_last_pct_min", [0.75])]
    fixed_bounds = [float(v) for v in posthoc.get("fixed_lower_bounds", [0.55, 0.6, 0.65, 0.7])]
    margins = [float(v) for v in posthoc.get("known_last_margins", [0.05, 0.1, 0.15, 0.2])]
    alphas = [float(v) for v in posthoc.get("alphas", [0.25, 0.5, 1.0])]
    specs: list[dict[str, Any]] = []
    for threshold in thresholds:
        for lower in fixed_bounds:
            for alpha in alphas:
                specs.append(
                    {
                        "policy": (
                            f"fixed_lb{pct_label(lower)}_"
                            f"klp{pct_label(threshold)}_a{pct_label(alpha)}"
                        ),
                        "lower_bound_kind": "fixed_pct",
                        "lower_bound_value": lower,
                        "known_last_margin": None,
                        "known_last_pct_min": threshold,
                        "alpha": alpha,
                    }
                )
        for margin in margins:
            for alpha in alphas:
                specs.append(
                    {
                        "policy": (
                            f"known_last_m{pct_label(margin)}_"
                            f"klp{pct_label(threshold)}_a{pct_label(alpha)}"
                        ),
                        "lower_bound_kind": "known_last_minus_margin",
                        "lower_bound_value": None,
                        "known_last_margin": margin,
                        "known_last_pct_min": threshold,
                        "alpha": alpha,
                    }
                )
    return specs


def apply_policy(
    frame: pd.DataFrame,
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = frame["pred_tvt"].to_numpy(float)
    typewell_min = frame["typewell_min"].to_numpy(float)
    span = frame["typewell_span"].to_numpy(float)
    known_last_pct = frame["known_last_pct"].to_numpy(float)
    pred_pct = frame["pred_pct"].to_numpy(float)
    if spec["lower_bound_kind"] == "fixed_pct":
        lower_pct = np.full(len(frame), float(spec["lower_bound_value"]), dtype=float)
    elif spec["lower_bound_kind"] == "known_last_minus_margin":
        lower_pct = known_last_pct - float(spec["known_last_margin"])
        lower_pct = np.clip(lower_pct, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown lower bound kind: {spec['lower_bound_kind']}")
    gate = np.isfinite(known_last_pct) & (known_last_pct >= float(spec["known_last_pct_min"]))
    changed = gate & np.isfinite(pred_pct) & (pred_pct < lower_pct)
    lower_tvt = typewell_min + lower_pct * span
    adjusted = pred.copy()
    adjusted[changed] = pred[changed] + float(spec["alpha"]) * (lower_tvt[changed] - pred[changed])
    return adjusted.astype(np.float32), changed, lower_pct


def make_by_well_metrics(frame: pd.DataFrame, policy_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base_pred = policy_preds["baseline"]
    base_error_by_well = (
        pd.DataFrame(
            {
                "well": frame["well"],
                "base_error": base_pred - frame["target_tvt"].to_numpy(float),
            }
        )
        .groupby("well")["base_error"]
        .agg(lambda value: float(np.sqrt(np.mean(np.square(value)))))
        .rename("baseline_rmse_tvt")
    )
    for policy, pred in policy_preds.items():
        temp = frame[["source", "well", "id"]].copy()
        temp["error_tvt"] = pred - frame["target_tvt"].to_numpy(float)
        grouped = (
            temp.groupby(["source", "well"], as_index=False)
            .agg(
                rows=("id", "size"),
                rmse_tvt=("error_tvt", lambda s: float(np.sqrt(np.mean(np.square(s))))),
                error_abs_mean=("error_tvt", lambda s: float(np.mean(np.abs(s)))),
                error_mean=("error_tvt", "mean"),
            )
            .merge(base_error_by_well, on="well", how="left")
        )
        grouped["policy"] = policy
        grouped["rmse_delta_vs_baseline"] = grouped["rmse_tvt"] - grouped["baseline_rmse_tvt"]
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["policy", "rmse_delta_vs_baseline", "rmse_tvt"],
        ascending=[True, False, False],
    )


def make_bucket_metrics(
    frame: pd.DataFrame,
    policy_preds: dict[str, np.ndarray],
    changed_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for policy, pred in policy_preds.items():
        temp = frame[["source", "id"]].copy()
        temp["error_tvt"] = pred - frame["target_tvt"].to_numpy(float)
        temp["distance_bucket"] = frame["distance_bucket"].astype("object").fillna("unknown")
        temp["known_last_pct_bucket"] = frame["known_last_pct_bucket"].astype("object").fillna(
            "unknown"
        )
        temp["pred_pct_bucket"] = frame["pred_pct_bucket"].astype("object").fillna("unknown")
        temp["target_pct_bucket"] = frame["target_pct_bucket"].astype("object").fillna("unknown")
        temp["changed_flag"] = np.where(changed_masks.get(policy, np.zeros(len(frame), bool)), "changed", "unchanged")
        temp["test_like"] = np.where(frame["test_like_known_last"], "known_last_ge_0p75", "other")
        for bucket_col in [
            "distance_bucket",
            "known_last_pct_bucket",
            "pred_pct_bucket",
            "target_pct_bucket",
            "changed_flag",
            "test_like",
        ]:
            grouped = (
                temp.groupby(["source", bucket_col], observed=True)
                .agg(
                    rows=("id", "size"),
                    rmse_tvt=("error_tvt", lambda s: float(np.sqrt(np.mean(np.square(s))))),
                    error_abs_mean=("error_tvt", lambda s: float(np.mean(np.abs(s)))),
                )
                .reset_index()
                .rename(columns={bucket_col: "bucket"})
            )
            grouped.insert(1, "policy", policy)
            grouped.insert(2, "bucket_family", bucket_col)
            rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def make_group_metrics(
    frame: pd.DataFrame,
    policy_preds: dict[str, np.ndarray],
    changed_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    group_masks = {
        "all": np.ones(len(frame), dtype=bool),
        "test_like_known_last_pct_ge_0p75": frame["test_like_known_last"].to_numpy(bool),
        "baseline_pred_pct_lt_0p5": frame["baseline_front_half_pred"].to_numpy(bool),
        "target_pct_lt_0p5_exception": frame["front_half_target"].to_numpy(bool),
        "near_000_050": frame["distance_bucket"].astype("object").eq("000_050").to_numpy(bool),
        "tail_1000_plus": frame["distance_bucket"].astype("object").eq("1000_plus").to_numpy(bool),
    }
    rows: list[dict[str, Any]] = []
    y_true = frame["target_tvt"].to_numpy(float)
    for policy, pred in policy_preds.items():
        group_masks["changed_rows"] = changed_masks.get(policy, np.zeros(len(frame), bool))
        for group, mask in group_masks.items():
            if not mask.any():
                continue
            rows.append(
                {
                    "source": str(frame["source"].iloc[0]),
                    "policy": policy,
                    "group": group,
                    "rows": int(mask.sum()),
                    "wells": int(frame.loc[mask, "well"].nunique()),
                    "rmse_tvt": rmse(y_true[mask], pred[mask]),
                    "mae_tvt": mae(y_true[mask], pred[mask]),
                    "within10": within10(y_true[mask], pred[mask]),
                }
            )
    return pd.DataFrame(rows)


def write_selected_oof(
    frame: pd.DataFrame,
    policy_preds: dict[str, np.ndarray],
    changed_masks: dict[str, np.ndarray],
    output_path: Path,
) -> dict[str, Any]:
    rows: list[pd.DataFrame] = []
    base_pred = frame["pred_tvt"].to_numpy(float)
    typewell_min = frame["typewell_min"].to_numpy(float)
    span = frame["typewell_span"].to_numpy(float)
    for policy, pred in policy_preds.items():
        out = frame[
            [
                "source",
                "id",
                "well",
                "variant",
                "mode",
                "model",
                "target_tvt",
                "last_known_tvt",
                "typewell_min",
                "typewell_max",
                "known_last_pct",
                "pred_pct",
                "target_pct",
                "distance_bucket",
            ]
        ].copy()
        out["policy"] = policy
        out["baseline_pred_tvt"] = base_pred
        out["pred_tvt"] = pred
        out["adjusted_pct"] = (pred - typewell_min) / span
        out["changed"] = changed_masks.get(policy, np.zeros(len(frame), bool))
        rows.append(out)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(output_path, index=False, compression="gzip")
    return artifact_sha(output_path)


def run_typewell_late_range_ml_posthoc_clip_audit(
    *,
    config: dict[str, Any],
    train_dir: str | Path,
    test_dir: str | Path | None,
    output_dir: str | Path,
    metrics_path: str | Path,
) -> dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    metrics_path = Path(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    min_span = float(get_nested(config, "model.posthoc.min_typewell_span", 1.0))
    save_oof_top_k = int(get_nested(config, "model.posthoc.save_oof_top_k", 12))

    context, typewell_summary = read_typewell_context(
        train_dir,
        test_dir,
        min_typewell_span=min_span,
    )
    prediction_sources = list(get_nested(config, "data.prediction_sources", []))
    if not prediction_sources:
        raise ValueError("No data.prediction_sources configured")

    source_meta_rows: list[dict[str, Any]] = []
    loaded_frames: list[pd.DataFrame] = []
    source_sha: list[dict[str, Any]] = []
    for source in prediction_sources:
        frame, meta = load_prediction_source(source)
        source_meta_rows.append({k: v for k, v in meta.items() if k != "sha"})
        if "sha" in meta:
            source_sha.append({"name": meta["name"], **meta["sha"]})
        if frame is not None:
            loaded_frames.append(frame)
    if not loaded_frames:
        raise FileNotFoundError("No prediction sources were loaded")

    all_candidate_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    selected_policy_preds: list[dict[str, np.ndarray]] = []
    selected_changed_masks: list[dict[str, np.ndarray]] = []
    selected_policy_names: list[list[str]] = []
    specs = make_policy_specs(config)

    for raw_predictions in loaded_frames:
        source_name = str(raw_predictions["source"].iloc[0])
        frame = enrich_predictions(raw_predictions, context)
        frame = frame[np.isfinite(frame["pred_pct"]) & np.isfinite(frame["target_pct"])].reset_index(
            drop=True
        )
        if frame.empty:
            raise ValueError(f"No valid rows after typewell pct enrichment for {source_name}")
        y_true = frame["target_tvt"].to_numpy(float)
        baseline_pred = frame["pred_tvt"].to_numpy(float)
        baseline_rmse = rmse(y_true, baseline_pred)
        baseline_spec = {
            "lower_bound_kind": "baseline",
            "lower_bound_value": None,
            "known_last_margin": None,
            "known_last_pct_min": None,
            "alpha": None,
        }
        all_candidate_rows.append(
            prediction_metrics(
                frame,
                baseline_pred,
                source=source_name,
                policy="baseline",
                baseline_rmse=baseline_rmse,
                spec=baseline_spec,
            )
        )
        for spec in specs:
            adjusted, changed, lower_pct = apply_policy(frame, spec)
            all_candidate_rows.append(
                prediction_metrics(
                    frame,
                    adjusted,
                    source=source_name,
                    policy=str(spec["policy"]),
                    baseline_rmse=baseline_rmse,
                    spec=spec,
                    changed_mask=changed,
                    lower_bound_pct=lower_pct,
                )
            )

        source_metrics = pd.DataFrame(
            [row for row in all_candidate_rows if row["source"] == source_name]
        )
        top_policies = (
            source_metrics.sort_values(
                ["rmse_tvt", "changed_rows", "policy"],
                ascending=[True, True, True],
            )["policy"]
            .head(save_oof_top_k)
            .tolist()
        )
        selected = ["baseline", *[p for p in top_policies if p != "baseline"]]
        policy_preds: dict[str, np.ndarray] = {"baseline": baseline_pred.astype(np.float32)}
        changed_masks: dict[str, np.ndarray] = {"baseline": np.zeros(len(frame), dtype=bool)}
        for spec in specs:
            policy = str(spec["policy"])
            if policy not in selected:
                continue
            adjusted, changed, _ = apply_policy(frame, spec)
            policy_preds[policy] = adjusted
            changed_masks[policy] = changed
        selected_frames.append(frame)
        selected_policy_preds.append(policy_preds)
        selected_changed_masks.append(changed_masks)
        selected_policy_names.append(selected)

    candidate_metrics = pd.DataFrame(all_candidate_rows).sort_values(
        ["source", "rmse_tvt", "changed_rows", "policy"],
        ascending=[True, True, True, True],
    )

    by_well_frames: list[pd.DataFrame] = []
    bucket_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    changed_frames: list[pd.DataFrame] = []
    oof_sha: list[dict[str, Any]] = []
    for frame, policy_preds, changed_masks in zip(
        selected_frames,
        selected_policy_preds,
        selected_changed_masks,
        strict=False,
    ):
        source_name = str(frame["source"].iloc[0])
        by_well = make_by_well_metrics(frame, policy_preds)
        by_well_frames.append(by_well)
        bucket_frames.append(make_bucket_metrics(frame, policy_preds, changed_masks))
        group_frames.append(make_group_metrics(frame, policy_preds, changed_masks))
        changed_rows = []
        for policy, mask in changed_masks.items():
            changed_rows.append(
                {
                    "source": source_name,
                    "policy": policy,
                    "changed_rows": int(mask.sum()),
                    "changed_wells": int(frame.loc[mask, "well"].nunique()) if mask.any() else 0,
                    "changed_pred_pct_min": float(frame.loc[mask, "pred_pct"].min())
                    if mask.any()
                    else np.nan,
                    "changed_pred_pct_median": float(frame.loc[mask, "pred_pct"].median())
                    if mask.any()
                    else np.nan,
                    "changed_target_pct_median": float(frame.loc[mask, "target_pct"].median())
                    if mask.any()
                    else np.nan,
                }
            )
        changed_frames.append(pd.DataFrame(changed_rows))
        oof_path = output_dir / f"{OUTPUT_PREFIX}_{source_name}_selected_oof_predictions.csv.gz"
        oof_sha.append(
            {
                "source": source_name,
                **write_selected_oof(frame, policy_preds, changed_masks, oof_path),
            }
        )

    by_well_metrics = pd.concat(by_well_frames, ignore_index=True)
    bucket_metrics = pd.concat(bucket_frames, ignore_index=True)
    group_metrics = pd.concat(group_frames, ignore_index=True)
    changed_summary = pd.concat(changed_frames, ignore_index=True)
    source_summary = pd.DataFrame(source_meta_rows)

    max_regression = (
        by_well_metrics.groupby(["source", "policy"], as_index=False)["rmse_delta_vs_baseline"]
        .max()
        .rename(columns={"rmse_delta_vs_baseline": "max_well_regression_vs_baseline"})
    )
    candidate_metrics = candidate_metrics.merge(
        max_regression,
        on=["source", "policy"],
        how="left",
    )

    candidate_path = output_dir / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    bucket_path = output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = output_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    group_path = output_dir / f"{OUTPUT_PREFIX}_group_metrics.csv"
    changed_path = output_dir / f"{OUTPUT_PREFIX}_changed_summary.csv"
    source_path = output_dir / f"{OUTPUT_PREFIX}_source_summary.csv"
    typewell_path = output_dir / f"{OUTPUT_PREFIX}_typewell_summary.csv"

    candidate_metrics.to_csv(candidate_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    by_well_metrics.to_csv(by_well_path, index=False)
    group_metrics.to_csv(group_path, index=False)
    changed_summary.to_csv(changed_path, index=False)
    source_summary.to_csv(source_path, index=False)
    typewell_summary.to_csv(typewell_path, index=False)

    best_by_source = (
        candidate_metrics.sort_values(["source", "rmse_tvt", "changed_rows"])
        .groupby("source", as_index=False)
        .head(1)
        .to_dict(orient="records")
    )
    baseline_by_source = (
        candidate_metrics[candidate_metrics["policy"].eq("baseline")]
        .sort_values("source")
        .to_dict(orient="records")
    )
    generated_sha = {
        "candidate_metrics": artifact_sha(candidate_path),
        "bucket_metrics": artifact_sha(bucket_path),
        "by_well": artifact_sha(by_well_path),
        "group_metrics": artifact_sha(group_path),
        "changed_summary": artifact_sha(changed_path),
        "source_summary": artifact_sha(source_path),
        "typewell_summary": artifact_sha(typewell_path),
        "selected_oof_predictions": oof_sha,
    }
    summary = {
        "experiment": get_nested(config, "experiment.name"),
        "status": "completed",
        "mode": get_nested(config, "audit.mode"),
        "route": get_nested(config, "experiment.route"),
        "runtime_sec": float(time.time() - t0),
        "policy_grid_count": int(len(specs)),
        "source_count_loaded": int(len(loaded_frames)),
        "baseline_by_source": baseline_by_source,
        "best_by_source": best_by_source,
        "typewell_summary": typewell_summary.to_dict(orient="records"),
        "source_summary": source_summary.to_dict(orient="records"),
        "source_sha": source_sha,
        "generated_sha": generated_sha,
        "artifacts": {
            "candidate_metrics": candidate_path.name,
            "bucket_metrics": bucket_path.name,
            "by_well": by_well_path.name,
            "group_metrics": group_path.name,
            "changed_summary": changed_path.name,
            "source_summary": source_path.name,
            "typewell_summary": typewell_path.name,
            "selected_oof_predictions_pattern": f"{OUTPUT_PREFIX}_<source>_selected_oof_predictions.csv.gz",
            "summary": f"{OUTPUT_PREFIX}_summary.json",
        },
        "notes": [
            "No model training, PF/Beam regeneration, inference, or submission is performed.",
            "True TVT is used only for scoring and diagnostics.",
        ],
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary["generated_sha"]["summary"] = artifact_sha(summary_path)

    metrics = {
        "experiment": get_nested(config, "experiment.name"),
        "status": "completed",
        "cv": best_by_source,
        "public_lb": None,
        "private_lb": None,
        "metric": get_nested(config, "validation.metric"),
        "seed": get_nested(config, "validation.seed"),
        "summary_path": str(summary_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return {
        "summary": summary,
        "candidate_metrics": candidate_metrics,
        "bucket_metrics": bucket_metrics,
        "by_well_metrics": by_well_metrics,
        "group_metrics": group_metrics,
        "changed_summary": changed_summary,
        "source_summary": source_summary,
        "typewell_summary": typewell_summary,
    }

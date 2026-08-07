from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import ExperimentPaths, get_nested, load_config


EXPERIMENT_NAME = "exp205_exact_hmm_smoother_exp072_compatible_cache_audit"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_existing(paths: ExperimentPaths, candidates: list[str]) -> Path:
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = paths.root / candidate
        checked.append(str(candidate))
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    if KAGGLE_INPUT_ROOT.exists():
        for raw in candidates:
            basename = Path(raw).name
            if not basename:
                continue
            matches = [path for path in sorted(KAGGLE_INPUT_ROOT.rglob(basename)) if path.stat().st_size > 0]
            checked.extend(str(path) for path in matches)
            if matches:
                return matches[0]
    raise FileNotFoundError("No non-empty candidate path exists: " + json.dumps(checked, indent=2))


def exp072_column(candidate: str) -> str:
    if candidate in {"pf_ancc", "pf_z"}:
        return candidate
    return f"{candidate}_d"


def exp072_prediction(frame: pd.DataFrame, candidate: str) -> np.ndarray:
    column = exp072_column(candidate)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    if column.endswith("_d"):
        values = values + pd.to_numeric(frame["last_known_tvt"], errors="coerce").to_numpy(np.float64)
    return values


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def within(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= threshold))


def distance_bucket(md_since: pd.Series) -> pd.Series:
    values = pd.to_numeric(md_since, errors="coerce").to_numpy(np.float64)
    labels = np.full(len(values), "1000_plus", dtype=object)
    labels[values < 1000.0] = "500_1000"
    labels[values < 500.0] = "250_500"
    labels[values < 250.0] = "100_250"
    labels[values < 100.0] = "050_100"
    labels[values < 50.0] = "000_050"
    return pd.Series(labels, index=md_since.index)


def metric_row(
    candidate: str,
    true_tvt: np.ndarray,
    pred: np.ndarray,
    baseline_pred: np.ndarray,
    rows: int,
) -> dict[str, Any]:
    pred_rmse = rmse(true_tvt, pred)
    baseline_rmse = rmse(true_tvt, baseline_pred)
    return {
        "candidate": candidate,
        "rows": int(rows),
        "rmse": pred_rmse,
        "mae": mae(true_tvt, pred),
        "bias": bias(true_tvt, pred),
        "within10": within(true_tvt, pred, 10.0),
        "delta_rmse_vs_exp072_likpf_mean": pred_rmse - baseline_rmse,
        "delta_mae_vs_exp072_likpf_mean": mae(true_tvt, pred) - mae(true_tvt, baseline_pred),
        "delta_within10_vs_exp072_likpf_mean": within(true_tvt, pred, 10.0)
        - within(true_tvt, baseline_pred, 10.0),
    }


def align_frames(baseline: pd.DataFrame, hmm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if len(baseline) == len(hmm):
        mismatch = int((baseline["id"].astype(str).to_numpy() != hmm["id"].astype(str).to_numpy()).sum())
        if mismatch == 0:
            return baseline.reset_index(drop=True), hmm.reset_index(drop=True), 0
    merged = baseline.merge(hmm, on="id", how="inner", suffixes=("_exp072", "_hmm"), validate="one_to_one")
    if len(merged) != len(baseline) or len(merged) != len(hmm):
        raise ValueError(
            f"id set mismatch after merge: baseline={len(baseline)} hmm={len(hmm)} merged={len(merged)}"
        )
    baseline_cols = [col for col in baseline.columns if col != "id"]
    hmm_cols = [col for col in hmm.columns if col != "id"]
    baseline_aligned = pd.DataFrame({"id": merged["id"]})
    hmm_aligned = pd.DataFrame({"id": merged["id"]})
    for col in baseline_cols:
        baseline_aligned[col] = merged[f"{col}_exp072"] if f"{col}_exp072" in merged else merged[col]
    for col in hmm_cols:
        hmm_aligned[col] = merged[f"{col}_hmm"] if f"{col}_hmm" in merged else merged[col]
    return baseline_aligned, hmm_aligned, -1


def compute_uncertainty_bins(
    hmm: pd.DataFrame,
    true_tvt: np.ndarray,
    pred: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "hmm_std": pd.to_numeric(hmm["hmm_std"], errors="coerce").to_numpy(np.float64),
            "abs_error": np.abs(pred - true_tvt),
            "sq_error": (pred - true_tvt) ** 2,
        }
    )
    work["bin"] = pd.qcut(work["hmm_std"].rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for bin_id, group in work.groupby("bin", sort=True):
        rows.append(
            {
                "hmm_std_bin": int(bin_id),
                "rows": int(len(group)),
                "hmm_std_min": float(group["hmm_std"].min()),
                "hmm_std_mean": float(group["hmm_std"].mean()),
                "hmm_std_max": float(group["hmm_std"].max()),
                "hmm_abs_error_mean": float(group["abs_error"].mean()),
                "hmm_rmse": float(np.sqrt(group["sq_error"].mean())),
            }
        )
    return pd.DataFrame(rows)


def compute_step_delta_rates(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    thresholds: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = frame[["well", "last_known_tvt"]].copy()
    for name, pred in predictions.items():
        base[name] = pred
    for name in predictions:
        deltas: list[np.ndarray] = []
        for _, group in base.groupby("well", sort=False):
            pred = group[name].to_numpy(np.float64)
            prev = np.empty(len(group), dtype=np.float64)
            prev[0] = float(group["last_known_tvt"].iloc[0])
            if len(group) > 1:
                prev[1:] = pred[:-1]
            deltas.append(np.abs(pred - prev))
        abs_delta = np.concatenate(deltas) if deltas else np.array([], dtype=np.float64)
        row: dict[str, Any] = {
            "candidate": name,
            "rows": int(len(abs_delta)),
            "abs_step_delta_mean": float(np.mean(abs_delta)) if len(abs_delta) else None,
            "abs_step_delta_p95": float(np.quantile(abs_delta, 0.95)) if len(abs_delta) else None,
            "abs_step_delta_p99": float(np.quantile(abs_delta, 0.99)) if len(abs_delta) else None,
        }
        for threshold in thresholds:
            key = f"rate_abs_step_delta_gt_{str(threshold).replace('.', 'p')}"
            row[key] = float(np.mean(abs_delta > threshold)) if len(abs_delta) else None
        rows.append(row)
    return pd.DataFrame(rows)


def run_direct_comparison() -> dict[str, Any]:
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    comparison = get_nested(config, "comparison") or {}
    output_prefix = str(comparison.get("output_prefix") or "exp205_vs_exp072")
    baseline_candidates = list(comparison.get("baseline_candidate_columns") or ["likpf_mean"])
    blend_weights = [float(v) for v in (comparison.get("blend_hmm_weights") or [0.25, 0.5, 0.75])]
    thresholds = [float(v) for v in (comparison.get("step_delta_thresholds") or [0.08, 0.10, 0.20])]

    baseline_path = resolve_existing(paths, list(comparison.get("baseline_feature_cache") or []))
    hmm_path = resolve_existing(paths, list(comparison.get("hmm_feature_cache") or []))

    baseline_needed = {"id", "well", "target", "last_known_tvt", "md_since"}
    for candidate in baseline_candidates:
        baseline_needed.add(exp072_column(candidate))
    baseline = pd.read_csv(
        baseline_path,
        usecols=sorted(baseline_needed),
        dtype={"id": str, "well": str},
    )
    hmm_needed = {
        "id",
        "well",
        "target",
        "last_known_tvt",
        "md_since",
        "hmm_mean_tvt",
        "hmm_mean_d",
        "hmm_std",
        "hmm_loglik",
    }
    hmm = pd.read_csv(hmm_path, usecols=sorted(hmm_needed), dtype={"id": str, "well": str})
    baseline, hmm, id_mismatches = align_frames(baseline, hmm)

    true_tvt = (
        pd.to_numeric(baseline["last_known_tvt"], errors="coerce").to_numpy(np.float64)
        + pd.to_numeric(baseline["target"], errors="coerce").to_numpy(np.float64)
    )
    likpf = exp072_prediction(baseline, "likpf_mean")
    hmm_pred = pd.to_numeric(hmm["hmm_mean_tvt"], errors="coerce").to_numpy(np.float64)

    predictions: dict[str, np.ndarray] = {}
    for candidate in baseline_candidates:
        predictions[f"exp072_{candidate}"] = exp072_prediction(baseline, candidate)
    predictions["hmm_mean_tvt"] = hmm_pred
    for weight in blend_weights:
        name = f"blend_likpf_hmm_w{int(round(weight * 1000)):03d}"
        predictions[name] = (1.0 - weight) * likpf + weight * hmm_pred

    bucket = distance_bucket(baseline["md_since"])
    overall_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    by_well_rows: list[dict[str, Any]] = []
    for name, pred in predictions.items():
        overall_rows.append(metric_row(name, true_tvt, pred, likpf, len(baseline)))
        for bucket_name in ["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"]:
            mask = bucket.to_numpy() == bucket_name
            if not np.any(mask):
                continue
            row = metric_row(name, true_tvt[mask], pred[mask], likpf[mask], int(mask.sum()))
            row["bucket"] = bucket_name
            bucket_rows.append(row)
        work = pd.DataFrame(
            {
                "well": baseline["well"].astype(str),
                "true_tvt": true_tvt,
                "pred": pred,
                "likpf": likpf,
            }
        )
        for well, group in work.groupby("well", sort=False):
            candidate_rmse = rmse(group["true_tvt"].to_numpy(), group["pred"].to_numpy())
            likpf_rmse = rmse(group["true_tvt"].to_numpy(), group["likpf"].to_numpy())
            by_well_rows.append(
                {
                    "candidate": name,
                    "well": well,
                    "rows": int(len(group)),
                    "rmse": candidate_rmse,
                    "exp072_likpf_mean_rmse": likpf_rmse,
                    "delta_rmse_vs_exp072_likpf_mean": candidate_rmse - likpf_rmse,
                }
            )

    overall = pd.DataFrame(overall_rows).sort_values("rmse")
    bucket_metrics = pd.DataFrame(bucket_rows)
    by_well = pd.DataFrame(by_well_rows)
    uncertainty_bins = compute_uncertainty_bins(hmm, true_tvt, hmm_pred, int(comparison.get("uncertainty_bins", 10)))
    step_delta_rates = compute_step_delta_rates(baseline, predictions, thresholds)

    enriched_path = paths.artifacts_dir / f"{output_prefix}_enriched_hmm_exp072_train_features.csv.gz"
    enriched = pd.DataFrame(
        {
            "id": baseline["id"].astype(str),
            "well": baseline["well"].astype(str),
            "target": baseline["target"],
            "last_known_tvt": baseline["last_known_tvt"],
            "md_since": baseline["md_since"],
            "hmm_mean_tvt": hmm_pred,
            "hmm_mean_d": pd.to_numeric(hmm["hmm_mean_d"], errors="coerce"),
            "hmm_std": pd.to_numeric(hmm["hmm_std"], errors="coerce"),
            "hmm_loglik": pd.to_numeric(hmm["hmm_loglik"], errors="coerce"),
            "hmm_minus_likpf_mean": hmm_pred - likpf,
        }
    )
    for name, pred in predictions.items():
        if name.startswith("blend_likpf_hmm"):
            enriched[name] = pred
    if bool(comparison.get("write_enriched_cache", True)):
        enriched.to_csv(enriched_path, index=False, compression="gzip")

    overall_path = paths.artifacts_dir / f"{output_prefix}_overall_metrics.csv"
    bucket_path = paths.artifacts_dir / f"{output_prefix}_distance_bucket_metrics.csv"
    by_well_path = paths.artifacts_dir / f"{output_prefix}_by_well_delta.csv"
    uncertainty_path = paths.artifacts_dir / f"{output_prefix}_hmm_std_calibration.csv"
    step_delta_path = paths.artifacts_dir / f"{output_prefix}_step_delta_rates.csv"
    summary_path = paths.artifacts_dir / f"{output_prefix}_summary.json"
    overall.to_csv(overall_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    by_well.sort_values(["candidate", "delta_rmse_vs_exp072_likpf_mean"], ascending=[True, False]).to_csv(
        by_well_path,
        index=False,
    )
    uncertainty_bins.to_csv(uncertainty_path, index=False)
    step_delta_rates.to_csv(step_delta_path, index=False)

    hmm_by_well = by_well[by_well["candidate"] == "hmm_mean_tvt"]
    best = overall.iloc[0].to_dict() if len(overall) else {}
    summary = {
        "experiment": EXPERIMENT_NAME,
        "baseline": str(baseline_path),
        "hmm_feature_cache": str(hmm_path),
        "rows_checked": int(len(baseline)),
        "unique_wells": int(baseline["well"].nunique()),
        "id_mismatches": id_mismatches,
        "best_candidate": best,
        "overall_metrics": overall.to_dict(orient="records"),
        "hmm_std_abs_error_corr": float(
            np.corrcoef(pd.to_numeric(hmm["hmm_std"], errors="coerce").to_numpy(np.float64), np.abs(hmm_pred - true_tvt))[
                0,
                1,
            ]
        ),
        "hmm_by_well_delta_summary": {
            "improved_wells": int((hmm_by_well["delta_rmse_vs_exp072_likpf_mean"] < 0).sum()),
            "worsened_wells": int((hmm_by_well["delta_rmse_vs_exp072_likpf_mean"] > 0).sum()),
            "same_wells": int((hmm_by_well["delta_rmse_vs_exp072_likpf_mean"] == 0).sum()),
            "max_regression_rmse": (
                float(hmm_by_well["delta_rmse_vs_exp072_likpf_mean"].max()) if len(hmm_by_well) else None
            ),
            "max_regression_well": (
                str(hmm_by_well.sort_values("delta_rmse_vs_exp072_likpf_mean", ascending=False).iloc[0]["well"])
                if len(hmm_by_well)
                else None
            ),
        },
        "artifacts": {
            "overall_metrics": str(overall_path),
            "distance_bucket_metrics": str(bucket_path),
            "by_well_delta": str(by_well_path),
            "hmm_std_calibration": str(uncertainty_path),
            "step_delta_rates": str(step_delta_path),
            "summary": str(summary_path),
            "enriched_cache": str(enriched_path) if bool(comparison.get("write_enriched_cache", True)) else None,
        },
        "sha256": {
            "overall_metrics": sha256_path(overall_path),
            "distance_bucket_metrics": sha256_path(bucket_path),
            "by_well_delta": sha256_path(by_well_path),
            "hmm_std_calibration": sha256_path(uncertainty_path),
            "step_delta_rates": sha256_path(step_delta_path),
        },
    }
    if enriched_path.exists():
        summary["sha256"]["enriched_cache_gzip"] = sha256_path(enriched_path)
        summary["sha256"]["enriched_cache_decompressed"] = sha256_gzip_decompressed(enriched_path)
    write_json(summary_path, summary)
    summary["sha256"]["summary"] = sha256_path(summary_path)
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    run_direct_comparison()

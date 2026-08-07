# %% [markdown]
# # exp200_pf_step_delta_soft_prior_full_replay_replacement comparison
#
# Self-contained Kaggle comparison for exp200 vs exp072 full replay train feature caches.

# %% [markdown]
# ## Contents
# 1. Imports and constants
# 2. File discovery
# 3. Metric helpers
# 4. Direct comparison
# 5. Guard summary

# %%
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# %% [markdown]
# ## 1. Imports and constants

# %%
EXPERIMENT_NAME = "exp200_pf_step_delta_soft_prior_full_replay_replacement"
BASELINE_NAME = "exp072"
CANDIDATE_NAME = "exp200"
BASELINE_FEATURE_FILE = "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
CANDIDATE_FEATURE_FILE = (
    "exp200_pf_step_delta_soft_prior_full_replay_replacement_full_replay_cache_"
    "pixiux_likpf_step_delta_prior_public_replay_train_features.csv.gz"
)
CANDIDATE_SUMMARY_FILE = "exp200_pf_step_delta_soft_prior_full_replay_replacement_full_replay_cache_summary.json"
CANDIDATES = ["pf_ancc", "pf_z", "beam_mean", "beam_cons", "beam_sm5", "beam_med", "likpf_mean"]
STEP_DELTA_THRESHOLDS = [0.08, 0.10, 0.20]
LIKPF_MEAN_MAX_ALLOWED_DELTA_RMSE = 0.02

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
ARTIFACTS_DIR = WORKING_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

print("input_root:", INPUT_ROOT, "exists=", INPUT_ROOT.exists())
print("working_root:", WORKING_ROOT, "exists=", WORKING_ROOT.exists())
print("artifacts_dir:", ARTIFACTS_DIR)

# %% [markdown]
# ## 2. File discovery

# %%
def find_one(filename: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(filename))
    print(filename, "matches=", len(matches))
    for match in matches[:5]:
        print("  ", match)
    if not matches:
        raise FileNotFoundError(filename)
    return matches[0]


baseline_path = find_one(BASELINE_FEATURE_FILE)
candidate_path = find_one(CANDIDATE_FEATURE_FILE)
summary_matches = sorted(INPUT_ROOT.rglob(CANDIDATE_SUMMARY_FILE))
candidate_summary_path = summary_matches[0] if summary_matches else None
print("candidate_summary_path:", candidate_summary_path)

# %% [markdown]
# ## 3. Metric helpers

# %%
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
        return float(value) if np.isfinite(float(value)) else None
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


def candidate_column(candidate: str) -> str:
    if candidate in {"pf_ancc", "pf_z"}:
        return candidate
    return f"{candidate}_d"


def prediction_tvt(frame: pd.DataFrame, candidate: str) -> np.ndarray:
    column = candidate_column(candidate)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    if column.endswith("_d"):
        values = values + pd.to_numeric(frame["last_known_tvt"], errors="coerce").to_numpy(np.float64)
    return values


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


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
    baseline_pred: np.ndarray,
    candidate_pred: np.ndarray,
    rows: int,
) -> dict[str, Any]:
    baseline_rmse = rmse(true_tvt, baseline_pred)
    candidate_rmse = rmse(true_tvt, candidate_pred)
    baseline_mae = mae(true_tvt, baseline_pred)
    candidate_mae = mae(true_tvt, candidate_pred)
    baseline_within10 = within(true_tvt, baseline_pred, 10.0)
    candidate_within10 = within(true_tvt, candidate_pred, 10.0)
    return {
        "candidate": candidate,
        "rows": int(rows),
        f"{BASELINE_NAME}_rmse": baseline_rmse,
        f"{CANDIDATE_NAME}_rmse": candidate_rmse,
        "delta_rmse": candidate_rmse - baseline_rmse,
        f"{BASELINE_NAME}_mae": baseline_mae,
        f"{CANDIDATE_NAME}_mae": candidate_mae,
        "delta_mae": candidate_mae - baseline_mae,
        f"{BASELINE_NAME}_within10": baseline_within10,
        f"{CANDIDATE_NAME}_within10": candidate_within10,
        "delta_within10": candidate_within10 - baseline_within10,
    }


def compute_step_delta_rates(frame: pd.DataFrame, candidates: list[str], thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = frame[["well", "last_known_tvt"]].copy()
    for candidate in candidates:
        work[candidate] = prediction_tvt(frame, candidate)
    for candidate in candidates:
        deltas: list[np.ndarray] = []
        for _, group in work.groupby("well", sort=False):
            pred = group[candidate].to_numpy(np.float64)
            prev = np.empty(len(group), dtype=np.float64)
            prev[0] = float(group["last_known_tvt"].iloc[0])
            if len(group) > 1:
                prev[1:] = pred[:-1]
            deltas.append(np.abs(pred - prev))
        abs_delta = np.concatenate(deltas) if deltas else np.array([], dtype=np.float64)
        row: dict[str, Any] = {
            "candidate": candidate,
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


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}

# %% [markdown]
# ## 4. Direct comparison

# %%
needed = {"id", "well", "target", "last_known_tvt", "md_since"}
for candidate in CANDIDATES:
    needed.add(candidate_column(candidate))
usecols = sorted(needed)
print("usecols:", usecols)

baseline = pd.read_csv(baseline_path, usecols=usecols, dtype={"id": str, "well": str})
candidate = pd.read_csv(candidate_path, usecols=usecols, dtype={"id": str, "well": str})
print("baseline shape:", baseline.shape)
print("candidate shape:", candidate.shape)

if len(baseline) != len(candidate):
    raise ValueError(f"row count mismatch: baseline={len(baseline)} candidate={len(candidate)}")
id_mismatches = int((baseline["id"].astype(str).to_numpy() != candidate["id"].astype(str).to_numpy()).sum())
if id_mismatches:
    raise ValueError(f"id mismatch count: {id_mismatches}")

true_tvt = (
    pd.to_numeric(baseline["last_known_tvt"], errors="coerce").to_numpy(np.float64)
    + pd.to_numeric(baseline["target"], errors="coerce").to_numpy(np.float64)
)
bucket = distance_bucket(baseline["md_since"])

overall_rows: list[dict[str, Any]] = []
bucket_rows: list[dict[str, Any]] = []
by_well_rows: list[dict[str, Any]] = []

for candidate_name_col in CANDIDATES:
    base_pred = prediction_tvt(baseline, candidate_name_col)
    cand_pred = prediction_tvt(candidate, candidate_name_col)
    overall_rows.append(metric_row(candidate_name_col, true_tvt, base_pred, cand_pred, len(baseline)))
    for bucket_name in ["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"]:
        mask = bucket.to_numpy() == bucket_name
        if not np.any(mask):
            continue
        row = metric_row(candidate_name_col, true_tvt[mask], base_pred[mask], cand_pred[mask], int(mask.sum()))
        row["bucket"] = bucket_name
        bucket_rows.append(row)
    work = pd.DataFrame(
        {
            "well": baseline["well"].astype(str),
            "true_tvt": true_tvt,
            "baseline_pred": base_pred,
            "candidate_pred": cand_pred,
        }
    )
    for well, group in work.groupby("well", sort=False):
        base_rmse = rmse(group["true_tvt"].to_numpy(), group["baseline_pred"].to_numpy())
        cand_rmse = rmse(group["true_tvt"].to_numpy(), group["candidate_pred"].to_numpy())
        by_well_rows.append(
            {
                "candidate": candidate_name_col,
                "well": well,
                "rows": int(len(group)),
                f"{BASELINE_NAME}_rmse": base_rmse,
                f"{CANDIDATE_NAME}_rmse": cand_rmse,
                "delta_rmse": cand_rmse - base_rmse,
            }
        )

overall = pd.DataFrame(overall_rows)
bucket_metrics = pd.DataFrame(bucket_rows)
by_well = pd.DataFrame(by_well_rows)
step_delta_rates = compute_step_delta_rates(candidate, CANDIDATES, STEP_DELTA_THRESHOLDS)

output_prefix = "exp200_vs_exp072"
overall_path = ARTIFACTS_DIR / f"{output_prefix}_overall_metrics.csv"
bucket_path = ARTIFACTS_DIR / f"{output_prefix}_distance_bucket_metrics.csv"
by_well_path = ARTIFACTS_DIR / f"{output_prefix}_by_well_delta.csv"
step_delta_path = ARTIFACTS_DIR / f"{output_prefix}_step_delta_rates.csv"
summary_path = ARTIFACTS_DIR / f"{output_prefix}_summary.json"

overall.to_csv(overall_path, index=False)
bucket_metrics.to_csv(bucket_path, index=False)
by_well.sort_values(["candidate", "delta_rmse"], ascending=[True, False]).to_csv(by_well_path, index=False)
step_delta_rates.to_csv(step_delta_path, index=False)

feature_summary = load_json(candidate_summary_path)
by_well_likpf = by_well[by_well["candidate"] == "likpf_mean"]
summary = {
    "experiment": EXPERIMENT_NAME,
    "baseline": str(baseline_path),
    "candidate": str(candidate_path),
    "rows_checked": int(len(baseline)),
    "unique_wells": int(baseline["well"].nunique()),
    "id_mismatches": id_mismatches,
    "overall_metrics": overall.to_dict(orient="records"),
    "likpf_mean_by_well_delta_summary": {
        "improved_wells": int((by_well_likpf["delta_rmse"] < 0).sum()) if len(by_well_likpf) else 0,
        "worsened_wells": int((by_well_likpf["delta_rmse"] > 0).sum()) if len(by_well_likpf) else 0,
        "same_wells": int((by_well_likpf["delta_rmse"] == 0).sum()) if len(by_well_likpf) else 0,
        "max_regression_rmse": float(by_well_likpf["delta_rmse"].max()) if len(by_well_likpf) else None,
        "max_regression_well": (
            str(by_well_likpf.sort_values("delta_rmse", ascending=False).iloc[0]["well"])
            if len(by_well_likpf)
            else None
        ),
    },
    "step_delta_rates": step_delta_rates.to_dict(orient="records"),
    "feature_generation_likpf_diagnostics": feature_summary.get("feature_meta", {}).get("likpf_diagnostics"),
    "artifacts": {
        "overall_metrics": str(overall_path),
        "distance_bucket_metrics": str(bucket_path),
        "by_well_delta": str(by_well_path),
        "step_delta_rates": str(step_delta_path),
        "summary": str(summary_path),
    },
    "sha256": {
        "overall_metrics": sha256_path(overall_path),
        "distance_bucket_metrics": sha256_path(bucket_path),
        "by_well_delta": sha256_path(by_well_path),
        "step_delta_rates": sha256_path(step_delta_path),
    },
}
write_json(summary_path, summary)
summary["sha256"]["summary"] = sha256_path(summary_path)
write_json(summary_path, summary)
print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))

# %% [markdown]
# ## 5. Guard summary

# %%
display_cols = ["candidate", "exp072_rmse", "exp200_rmse", "delta_rmse", "exp072_mae", "exp200_mae", "delta_mae"]
print(overall[display_cols])
likpf = overall[overall["candidate"] == "likpf_mean"].iloc[0]
guard_pass = bool(float(likpf["delta_rmse"]) <= LIKPF_MEAN_MAX_ALLOWED_DELTA_RMSE)
print("likpf_mean_delta_rmse:", float(likpf["delta_rmse"]))
print("guard_likpf_mean_max_allowed_delta:", LIKPF_MEAN_MAX_ALLOWED_DELTA_RMSE)
print("guard_pass:", guard_pass)

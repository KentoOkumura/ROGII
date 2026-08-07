# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp083 v12 ML OOF and PF/Beam RMSE title plots
#
# Diagnostic visualization notebook. It extends the exp083 v12 prediction plot by
# overlaying exp148 OOF ML predictions and exp226 K16 OOF predictions on the
# same exp072 feature-cache rows, and annotating each plot with per-well exp148
# CV(OOF), exp226 CV(OOF), PF/Beam oracle, and PF/Beam best1 RMSE. The TVT panel
# uses a depth-down y-axis to match official images and discussion plots. It
# visualizes a -Z guide min-max scaled to the generated Likelihood PF mean range
# on the plotted feature-cache rows only, plus exp209 HMM mean output.

# %% [markdown]
# ## Contents
#
# 1. Imports and configuration
# 2. Path resolution
# 3. Input loading and joins
# 4. Plot helpers
# 5. Generate all-well plots
# 6. Summary and generated outputs

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT_NAME = "exp083_pf_beam_true_tvt_2d_well_eda"
OUTPUT_PREFIX = "pf_beam_true_tvt_2d_well_eda_v12_exp148_exp226_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_2sigma_noformation_all"

PFBEAM_FILENAME = "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
EXP148_OOF_FILENAME = "exp148_learned_likelihood_fulltrain_addonly_on_exp092_predictions.csv.gz"
EXP226_OOF_FILENAME = "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train_oof_predictions.csv.gz"
EXP209_ENRICHED_HMM_FILENAME = "exp209_vs_exp072_exp205_enriched_hmm_exp072_train_features.csv.gz"
EXP209_BY_WELL_DELTA_FILENAME = "exp209_vs_exp072_exp205_by_well_delta.csv"
EXP209_OVERALL_METRICS_FILENAME = "exp209_vs_exp072_exp205_overall_metrics.csv"
EXP209_SUMMARY_FILENAME = "exp209_vs_exp072_exp205_summary.json"
EXP209_HMM_MEAN_COLUMN = "hmm_mean_tvt"
EXP226_EXPERIMENT_NAME = "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"

EXP148_VARIANT = "learned_likelihood_confidence_addonly"
EXP148_MODE = "gpu_repro_guard_dp_threads8"
EXP148_MODEL = "lgb_mean"
EXP148_RECORDED_CV_RMSE = 8.50128118189582
EXP226_RECORDED_CV_RMSE = 9.427109596582213

MAX_POINTS_PER_PLOT = 6000
OOF_CHUNKSIZE = 500_000
ZIP_PLOTS = True
TVT_AXIS_INVERTED = True

MAX_PLOTS_ENV = os.environ.get("EXPERIMENT_MAX_PLOTS")
MAX_PLOTS = int(MAX_PLOTS_ENV) if MAX_PLOTS_ENV else None

FORMATION_COLUMNS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
RAW_COLUMNS = ["TVT", "TVT_input", "Z", *FORMATION_COLUMNS]

CANDIDATE_SPECS = [
    {
        "name": "last_anchor_tvt",
        "label": "last anchor",
        "source_column": "last_known_tvt",
        "transform": "absolute",
    },
    {"name": "pf_ancc", "label": "PF ANCC", "source_column": "pf_ancc", "transform": "absolute"},
    {"name": "pf_z", "label": "PF Z", "source_column": "pf_z", "transform": "absolute"},
    {
        "name": "beam_mean",
        "label": "Beam mean",
        "source_column": "beam_mean_d",
        "transform": "base_plus_delta",
    },
    {
        "name": "likpf_mean",
        "label": "Likelihood PF mean",
        "source_column": "likpf_mean_d",
        "transform": "base_plus_delta",
    },
]

PFBEAM_RMSE_SPECS = [
    {"name": "pf_ancc", "label": "PF ANCC"},
    {"name": "pf_z", "label": "PF Z"},
    {"name": "beam_mean", "label": "Beam mean"},
    {"name": "likpf_mean", "label": "Likelihood PF mean"},
]
PFBEAM_RMSE_LABELS = {str(spec["name"]): str(spec["label"]) for spec in PFBEAM_RMSE_SPECS}

BACKGROUND_COLUMNS = [
    {
        "name": "z_likpf_minmax",
        "label": "-Z likPF minmax",
        "plot_column": "z_likpf_minmax_tvt",
        "transform": "raw",
        "use_common_scale": False,
        "already_tvt_scale": True,
        "color": "#db2777",
        "alpha": 0.62,
        "linewidth": 1.35,
        "linestyle": ":",
    },
]

BACKGROUND_BANDS: list[dict[str, Any]] = []

print("Experiment:", EXPERIMENT_NAME)
print("Output prefix:", OUTPUT_PREFIX)
print("Exp148 OOF filter:", EXP148_VARIANT, EXP148_MODE, EXP148_MODEL)
print("Exp148 recorded CV RMSE:", EXP148_RECORDED_CV_RMSE)
print("Exp226 recorded CV RMSE:", EXP226_RECORDED_CV_RMSE)
print("Plot scope: all wells")
print("TVT_input prefix interval plotted: no")
print("Prediction-start vertical line plotted: no")
print("Known TVT probe plotted: no")
print("TVT axis inverted depth-down:", TVT_AXIS_INVERTED)
print("Only Z-to-Likelihood-PF min-max guide plotted: yes, direct -Z to likPF range")
print("Z guide anchor/known-tail direction/clip: no")
print("Formation background plotted: no")
print("exp209 HMM outputs plotted: hmm_mean_tvt with a translucent +/-2sigma band")
print("exp226 OOF plotted: yes, K16 spline kernel kNN adaptive kappa reproduction")
print("Debug max plots override:", MAX_PLOTS)

# %% [markdown]
# ## 2. Path resolution

# %%
def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "experiment_summary.md").exists() and (candidate / "experiments").exists():
            return candidate
    return current


REPO_ROOT = find_repo_root(Path.cwd())
EXP_DIR = REPO_ROOT / "experiments" / EXPERIMENT_NAME
if not EXP_DIR.exists() and Path.cwd().name == EXPERIMENT_NAME:
    EXP_DIR = Path.cwd()

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
ARTIFACTS_DIR = (KAGGLE_WORKING_ROOT if KAGGLE_WORKING_ROOT.exists() else EXP_DIR) / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_plots"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def resolve_from_local_and_kaggle(
    *,
    filename: str,
    local_candidates: list[Path],
    preferred_slugs: list[str],
) -> Path:
    local = _existing(local_candidates)
    if local is not None:
        return local

    if KAGGLE_INPUT_ROOT.exists():
        roots: list[Path] = []
        roots.extend(KAGGLE_INPUT_ROOT / slug for slug in preferred_slugs)
        roots.extend(path for path in sorted(KAGGLE_INPUT_ROOT.iterdir()) if path.is_dir())
        seen: set[Path] = set()
        for root in roots:
            if root in seen or not root.exists():
                continue
            seen.add(root)
            matches = sorted(root.rglob(filename))
            if matches:
                return matches[0]

    checked = "\n".join(str(path) for path in local_candidates)
    raise FileNotFoundError(f"{filename} not found. Checked:\n{checked}\n{kaggle_hint()}")


def kaggle_hint() -> str:
    return (
        "On Kaggle, add these input sources: "
        "kentookumura/exp072-exp063-full-replay-feature-cache-train, "
        "kentookumura/exp148-train, "
        "kentookumura/exp226-k16-kappa-repro-train, and "
        "kentookumura/exp209-joint-exact-parity-train."
    )


def resolve_raw_train_dir() -> Path:
    candidates = [
        REPO_ROOT / "data" / "raw" / "train",
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction/train"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction/train"),
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*__horizontal_well.csv")):
            return candidate
    if KAGGLE_INPUT_ROOT.exists():
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if candidate.is_dir() and any(candidate.glob("*__horizontal_well.csv")):
                return candidate
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Raw train directory not found. Checked:\n{checked}")


pfbeam_path = resolve_from_local_and_kaggle(
    filename=PFBEAM_FILENAME,
    local_candidates=[
        REPO_ROOT
        / "experiments"
        / "exp072_exp063_full_replay_feature_cache"
        / "artifacts"
        / PFBEAM_FILENAME,
        EXP_DIR / "artifacts" / PFBEAM_FILENAME,
    ],
    preferred_slugs=["exp072-exp063-full-replay-feature-cache-train"],
)
exp148_oof_path = resolve_from_local_and_kaggle(
    filename=EXP148_OOF_FILENAME,
    local_candidates=[
        REPO_ROOT
        / "experiments"
        / "exp148_learned_likelihood_fulltrain_addonly_on_exp092"
        / "artifacts"
        / EXP148_OOF_FILENAME,
    ],
    preferred_slugs=["exp148-train"],
)
exp226_output_root = REPO_ROOT / "experiments" / EXP226_EXPERIMENT_NAME / "kaggle" / "output"
exp226_output_artifact_dirs = [
    exp226_output_root / "train_v1" / "artifacts",
    exp226_output_root / "train" / "artifacts",
    Path("/tmp/kaggle-output") / EXP226_EXPERIMENT_NAME / "train_v1" / "artifacts",
    Path("/tmp/kaggle-output") / EXP226_EXPERIMENT_NAME / "train" / "artifacts",
]
exp226_oof_path = resolve_from_local_and_kaggle(
    filename=EXP226_OOF_FILENAME,
    local_candidates=[
        REPO_ROOT / "experiments" / EXP226_EXPERIMENT_NAME / "artifacts" / EXP226_OOF_FILENAME,
        *[artifact_dir / EXP226_OOF_FILENAME for artifact_dir in exp226_output_artifact_dirs],
        EXP_DIR / "artifacts" / EXP226_OOF_FILENAME,
    ],
    preferred_slugs=[
        "exp226-k16-kappa-repro-train",
        "exp226-connortynan-k16-spline-kernel-knn-adaptive-kappa-reproduction-train",
    ],
)
exp209_output_root = (
    REPO_ROOT
    / "experiments"
    / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    / "kaggle"
    / "output"
)
exp209_output_artifact_dirs = [
    exp209_output_root / "train_v3_small" / "artifacts",
    Path("/tmp/kaggle-output/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/train_v3_small/artifacts"),
]
exp209_enriched_hmm_path = resolve_from_local_and_kaggle(
    filename=EXP209_ENRICHED_HMM_FILENAME,
    local_candidates=[
        *[artifact_dir / EXP209_ENRICHED_HMM_FILENAME for artifact_dir in exp209_output_artifact_dirs],
        EXP_DIR / "artifacts" / EXP209_ENRICHED_HMM_FILENAME,
    ],
    preferred_slugs=["exp209-joint-exact-parity-train"],
)
exp209_by_well_delta_path = resolve_from_local_and_kaggle(
    filename=EXP209_BY_WELL_DELTA_FILENAME,
    local_candidates=[
        *[artifact_dir / EXP209_BY_WELL_DELTA_FILENAME for artifact_dir in exp209_output_artifact_dirs],
        EXP_DIR / "artifacts" / EXP209_BY_WELL_DELTA_FILENAME,
    ],
    preferred_slugs=["exp209-joint-exact-parity-train"],
)
exp209_overall_metrics_path = resolve_from_local_and_kaggle(
    filename=EXP209_OVERALL_METRICS_FILENAME,
    local_candidates=[
        *[artifact_dir / EXP209_OVERALL_METRICS_FILENAME for artifact_dir in exp209_output_artifact_dirs],
        EXP_DIR / "artifacts" / EXP209_OVERALL_METRICS_FILENAME,
    ],
    preferred_slugs=["exp209-joint-exact-parity-train"],
)
exp209_summary_path = resolve_from_local_and_kaggle(
    filename=EXP209_SUMMARY_FILENAME,
    local_candidates=[
        *[artifact_dir / EXP209_SUMMARY_FILENAME for artifact_dir in exp209_output_artifact_dirs],
        EXP_DIR / "artifacts" / EXP209_SUMMARY_FILENAME,
    ],
    preferred_slugs=["exp209-joint-exact-parity-train"],
)
raw_train_dir = resolve_raw_train_dir()

print("Repo root:", REPO_ROOT)
print("Experiment dir:", EXP_DIR)
print("Artifacts dir:", ARTIFACTS_DIR)
print("PF/Beam source:", pfbeam_path)
print("exp148 OOF source:", exp148_oof_path)
print("exp226 OOF source:", exp226_oof_path)
print("exp209 enriched HMM source:", exp209_enriched_hmm_path)
print("exp209 by-well delta source:", exp209_by_well_delta_path)
print("exp209 overall metrics source:", exp209_overall_metrics_path)
print("exp209 summary source:", exp209_summary_path)
print("Raw train dir:", raw_train_dir)

# %% [markdown]
# ## 3. Input loading and joins

# %%
def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_raw_row_index(ids: pd.Series) -> pd.Series:
    suffix = ids.astype(str).str.rsplit("_", n=1).str[-1]
    row_idx = pd.to_numeric(suffix, errors="coerce")
    if bool(row_idx.isna().any()):
        bad = ids[row_idx.isna()].astype(str).head(5).tolist()
        raise ValueError(f"Could not parse raw row index from id examples: {bad}")
    return row_idx.astype(np.int64)


def rmse_between(truth: pd.Series, prediction: pd.Series) -> float:
    truth_values = pd.to_numeric(truth, errors="coerce").to_numpy(dtype=float)
    pred_values = pd.to_numeric(prediction, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(truth_values) & np.isfinite(pred_values)
    if not bool(valid.any()):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(pred_values[valid] - truth_values[valid]))))


def rowwise_oracle_rmse(frame: pd.DataFrame, columns: list[str]) -> float:
    truth_values = pd.to_numeric(frame["true_tvt"], errors="coerce").to_numpy(dtype=float)
    squared_errors: list[np.ndarray] = []
    for column in columns:
        if column not in frame:
            continue
        pred_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(truth_values) & np.isfinite(pred_values)
        errors = np.full(len(frame), np.nan, dtype=float)
        errors[valid] = np.square(pred_values[valid] - truth_values[valid])
        squared_errors.append(errors)
    if not squared_errors:
        return float("nan")
    error_matrix = np.vstack(squared_errors).T
    row_valid = np.isfinite(error_matrix).any(axis=1)
    if not bool(row_valid.any()):
        return float("nan")
    best_squared_error = np.nanmin(error_matrix[row_valid], axis=1)
    return float(np.sqrt(np.mean(best_squared_error)))


def prediction_rmse_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "exp148_oof_rmse": float("nan"),
        "exp226_oof_rmse": float("nan"),
        "pfbeam_oracle_rmse": float("nan"),
        "pfbeam_best1_column": None,
        "pfbeam_best1_label": None,
        "pfbeam_best1_rmse": float("nan"),
    }
    truth = pd.to_numeric(frame["true_tvt"], errors="coerce")
    if "exp148_lgb_mean_oof_tvt" in frame:
        metrics["exp148_oof_rmse"] = rmse_between(truth, frame["exp148_lgb_mean_oof_tvt"])
    if "exp226_k16_oof_tvt" in frame:
        metrics["exp226_oof_rmse"] = rmse_between(truth, frame["exp226_k16_oof_tvt"])

    candidate_rmses: dict[str, float] = {}
    candidate_columns = [str(spec["name"]) for spec in PFBEAM_RMSE_SPECS]
    for column in candidate_columns:
        if column not in frame:
            continue
        rmse_value = rmse_between(truth, frame[column])
        metrics[f"{column}_rmse"] = rmse_value
        if np.isfinite(rmse_value):
            candidate_rmses[column] = rmse_value

    metrics["pfbeam_oracle_rmse"] = rowwise_oracle_rmse(frame, candidate_columns)
    if candidate_rmses:
        best_column, best_rmse = min(candidate_rmses.items(), key=lambda item: item[1])
        metrics["pfbeam_best1_column"] = best_column
        metrics["pfbeam_best1_label"] = PFBEAM_RMSE_LABELS.get(best_column, best_column)
        metrics["pfbeam_best1_rmse"] = best_rmse
    return metrics


def fmt_rmse(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.2f}"


def fmt_rate(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{100.0 * numeric:.1f}%"


def jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def jsonable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {str(key): jsonable(value) for key, value in metrics.items()}


def read_pfbeam_frame(path: Path) -> pd.DataFrame:
    needed = {"id", "well", "target", "last_known_tvt", "md_since"}
    needed.update(str(spec["source_column"]) for spec in CANDIDATE_SPECS)
    frame = pd.read_csv(
        path,
        dtype={"id": str, "well": str},
        usecols=lambda column: column in needed,
        low_memory=False,
    )
    frame["well"] = frame["well"].astype(str)
    frame["id"] = frame["id"].astype(str)
    base = pd.to_numeric(frame["last_known_tvt"], errors="coerce")
    frame["true_tvt"] = base + pd.to_numeric(frame["target"], errors="coerce")
    for spec in CANDIDATE_SPECS:
        name = str(spec["name"])
        source_column = str(spec["source_column"])
        if source_column not in frame:
            continue
        values = pd.to_numeric(frame[source_column], errors="coerce")
        if spec["transform"] == "base_plus_delta":
            frame[name] = base + values
        else:
            frame[name] = values
    frame["raw_row_idx"] = parse_raw_row_index(frame["id"])
    return frame


def read_exp148_oof(path: Path) -> pd.DataFrame:
    usecols = {"id", "well", "variant", "mode", "model", "pred_tvt"}
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        dtype={"id": str, "well": str, "variant": str, "mode": str, "model": str},
        usecols=lambda column: column in usecols,
        chunksize=OOF_CHUNKSIZE,
        low_memory=False,
    ):
        mask = chunk["model"].eq(EXP148_MODEL)
        if "variant" in chunk:
            mask &= chunk["variant"].eq(EXP148_VARIANT)
        if "mode" in chunk:
            mask &= chunk["mode"].eq(EXP148_MODE)
        filtered = chunk.loc[mask, ["id", "well", "pred_tvt"]].copy()
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        raise ValueError(
            f"No exp148 OOF rows matched variant={EXP148_VARIANT}, mode={EXP148_MODE}, "
            f"model={EXP148_MODEL}"
        )
    oof = pd.concat(chunks, ignore_index=True)
    oof["id"] = oof["id"].astype(str)
    oof["well"] = oof["well"].astype(str)
    oof["exp148_lgb_mean_oof_tvt"] = pd.to_numeric(oof["pred_tvt"], errors="coerce")
    oof = oof.drop(columns=["pred_tvt"])
    duplicated = int(oof.duplicated(["id", "well"]).sum())
    if duplicated:
        print("Warning: duplicated exp148 OOF id/well rows, keeping the last:", duplicated)
        oof = oof.drop_duplicates(["id", "well"], keep="last")
    return oof


def read_exp226_oof(path: Path) -> pd.DataFrame:
    usecols = {"well_id", "row_idx", "tvt_pred"}
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        dtype={"well_id": str},
        usecols=lambda column: column in usecols,
        chunksize=OOF_CHUNKSIZE,
        low_memory=False,
    ):
        missing = usecols.difference(chunk.columns)
        if missing:
            raise ValueError(f"exp226 OOF missing columns: {sorted(missing)}")
        row_idx = pd.to_numeric(chunk["row_idx"], errors="coerce")
        if bool(row_idx.isna().any()):
            bad = chunk.loc[row_idx.isna(), ["well_id", "row_idx"]].head(5).to_dict(orient="records")
            raise ValueError(f"Could not parse exp226 row_idx examples: {bad}")
        well = chunk["well_id"].astype(str)
        frame = pd.DataFrame(
            {
                "id": well + "_" + row_idx.astype(np.int64).astype(str),
                "well": well,
                "exp226_k16_oof_tvt": pd.to_numeric(chunk["tvt_pred"], errors="coerce"),
            }
        )
        chunks.append(frame)
    if not chunks:
        raise ValueError(f"No exp226 OOF rows loaded from {path}")
    oof = pd.concat(chunks, ignore_index=True)
    duplicated = int(oof.duplicated(["id", "well"]).sum())
    if duplicated:
        print("Warning: duplicated exp226 OOF id/well rows, keeping the last:", duplicated)
        oof = oof.drop_duplicates(["id", "well"], keep="last")
    return oof


def read_exp209_enriched_hmm(path: Path) -> pd.DataFrame:
    keep_cols = [
        "id",
        "well",
        "md_since",
        EXP209_HMM_MEAN_COLUMN,
        "hmm_mean_d",
        "hmm_std",
        "hmm_loglik",
        "hmm_minus_likpf_mean",
    ]
    frame = pd.read_csv(
        path,
        dtype={"id": str, "well": str},
        usecols=lambda column: column in keep_cols,
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    frame["md_since"] = pd.to_numeric(frame["md_since"], errors="coerce")
    for column in [
        EXP209_HMM_MEAN_COLUMN,
        "hmm_mean_d",
        "hmm_std",
        "hmm_loglik",
        "hmm_minus_likpf_mean",
    ]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = np.nan
    duplicated = int(frame.duplicated(["id", "well"]).sum())
    if duplicated:
        print("Warning: duplicated exp209 HMM id/well rows, keeping the last:", duplicated)
        frame = frame.drop_duplicates(["id", "well"], keep="last")
    return frame


def read_exp209_by_well_delta(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"well": str})
    keep_cols = [
        "candidate",
        "well",
        "rows",
        "rmse",
        "exp072_likpf_mean_rmse",
        "delta_rmse_vs_exp072_likpf_mean",
    ]
    frame = frame[[column for column in keep_cols if column in frame]].copy()
    for column in keep_cols:
        if column not in {"candidate", "well"} and column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    duplicated = int(frame.duplicated(["candidate", "well"]).sum())
    if duplicated:
        print("Warning: duplicated exp209 candidate/well rows, keeping the last:", duplicated)
        frame = frame.drop_duplicates(["candidate", "well"], keep="last")
    return frame


def read_exp209_overall_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    numeric_cols = [
        "rows",
        "rmse",
        "mae",
        "bias",
        "within10",
        "delta_rmse_vs_exp072_likpf_mean",
        "delta_mae_vs_exp072_likpf_mean",
        "delta_within10_vs_exp072_likpf_mean",
    ]
    for column in numeric_cols:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def read_exp209_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def exp209_global_metric(candidate: str) -> dict[str, Any]:
    if exp209_overall_metrics.empty or "candidate" not in exp209_overall_metrics:
        return {}
    rows = exp209_overall_metrics.loc[exp209_overall_metrics["candidate"].eq(candidate)]
    if rows.empty:
        return {}
    return {str(key): jsonable(value) for key, value in rows.iloc[0].to_dict().items()}


def build_exp209_well_metrics(frame: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    if frame.empty:
        return metrics
    for well, well_group in frame.groupby("well", sort=False):
        candidate_metrics: dict[str, dict[str, Any]] = {}
        for _, row in well_group.iterrows():
            candidate = str(row.get("candidate", ""))
            if not candidate:
                continue
            candidate_metrics[candidate] = {
                str(key): jsonable(value)
                for key, value in row.to_dict().items()
                if key not in {"candidate", "well"}
            }
        metrics[str(well)] = candidate_metrics
    return metrics


pfbeam = read_pfbeam_frame(pfbeam_path)
exp148_oof = read_exp148_oof(exp148_oof_path)
exp226_oof = read_exp226_oof(exp226_oof_path)
exp209_enriched_hmm = read_exp209_enriched_hmm(exp209_enriched_hmm_path)
exp209_by_well_delta = read_exp209_by_well_delta(exp209_by_well_delta_path)
exp209_overall_metrics = read_exp209_overall_metrics(exp209_overall_metrics_path)
exp209_summary = read_exp209_summary(exp209_summary_path)

exp209_rows_by_well = exp209_enriched_hmm.groupby("well", dropna=False).size()
exp209_well_metrics = build_exp209_well_metrics(exp209_by_well_delta)

plot_frame = (
    pfbeam.merge(exp148_oof, on=["id", "well"], how="left", validate="one_to_one")
    .merge(exp226_oof, on=["id", "well"], how="left", validate="one_to_one")
)
exp209_enriched_hmm = exp209_enriched_hmm.drop(columns=["md_since"]).merge(
    plot_frame[["id", "well", "md_since"]],
    on=["id", "well"],
    how="left",
    validate="one_to_one",
)
all_wells = sorted(plot_frame["well"].dropna().astype(str).unique().tolist())
if MAX_PLOTS is not None:
    plot_wells = all_wells[:MAX_PLOTS]
else:
    plot_wells = all_wells
plot_indices_by_well = plot_frame.groupby("well", sort=False).indices
exp209_hmm_indices_by_well = exp209_enriched_hmm.groupby("well", sort=False).indices

print("PF/Beam rows:", len(pfbeam))
print("PF/Beam wells:", len(all_wells))
print("exp148 lgb_mean OOF rows:", len(exp148_oof))
print("exp226 K16 OOF rows:", len(exp226_oof))
print("exp209 enriched HMM rows:", len(exp209_enriched_hmm))
print("exp209 enriched HMM wells:", int(exp209_enriched_hmm["well"].nunique()))
print("exp209 enriched HMM rows per well min:", int(exp209_rows_by_well.min()))
print("exp209 enriched HMM rows per well max:", int(exp209_rows_by_well.max()))
print("exp209 HMM by-well delta summary:", exp209_summary.get("hmm_by_well_delta_summary", {}))
print("Joined rows:", len(plot_frame))
print("ML OOF coverage:", float(plot_frame["exp148_lgb_mean_oof_tvt"].notna().mean()))
print("exp226 OOF coverage:", float(plot_frame["exp226_k16_oof_tvt"].notna().mean()))
print("Plot wells:", len(plot_wells))
print("First plot wells:", plot_wells[:10])
global_rmse_metrics = prediction_rmse_metrics(plot_frame)
print("Global exp148 OOF RMSE:", fmt_rmse(global_rmse_metrics["exp148_oof_rmse"]))
print("Global exp226 OOF RMSE:", fmt_rmse(global_rmse_metrics["exp226_oof_rmse"]))
print("Global PF/Beam oracle RMSE:", fmt_rmse(global_rmse_metrics["pfbeam_oracle_rmse"]))
print(
    "Global PF/Beam best1 RMSE:",
    fmt_rmse(global_rmse_metrics["pfbeam_best1_rmse"]),
    global_rmse_metrics["pfbeam_best1_label"],
)
print("exp209 likPF metrics:", exp209_global_metric("exp072_likpf_mean"))
print("exp209 HMM mean metrics:", exp209_global_metric(EXP209_HMM_MEAN_COLUMN))

# %% [markdown]
# ## 4. Plot helpers

# %%
def downsample_for_plot(group: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(group) <= max_points:
        return group
    indices = np.linspace(0, len(group) - 1, int(max_points)).round().astype(int)
    return group.iloc[np.unique(indices)].copy()


def read_raw_well(well_id: str) -> pd.DataFrame:
    path = raw_train_dir / f"{well_id}__horizontal_well.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, usecols=lambda column: column in RAW_COLUMNS)
    raw["raw_row_idx"] = np.arange(len(raw), dtype=np.int64)
    return raw.set_index("raw_row_idx", drop=False)


def attach_raw_context(group: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    raw_context = raw.reindex(group["raw_row_idx"].to_numpy())
    for column in RAW_COLUMNS:
        if column in raw_context:
            group[f"raw_{column}"] = raw_context[column].to_numpy()
        else:
            group[f"raw_{column}"] = np.nan
    return group


def simple_likpf_minmax_neg_z_to_tvt(
    target_z: pd.Series,
    likpf_mean: pd.Series,
) -> tuple[pd.Series, dict[str, Any]]:
    target_neg_z = -pd.to_numeric(target_z, errors="coerce")
    target_neg_z_finite = target_neg_z[np.isfinite(target_neg_z)]
    likpf = pd.to_numeric(likpf_mean, errors="coerce")
    likpf_finite = likpf[np.isfinite(likpf)]
    meta: dict[str, Any] = {
        "z_likpf_status": "ok",
        "z_likpf_hidden_neg_z_min": None,
        "z_likpf_hidden_neg_z_max": None,
        "z_likpf_target_min": None,
        "z_likpf_target_max": None,
        "z_likpf_method": "plot_neg_z_minmax_to_likpf_mean_range",
        "z_likpf_anchor_used": False,
        "z_likpf_direction_used": False,
        "z_likpf_clip_to_range": False,
    }
    if target_neg_z_finite.empty:
        meta["z_likpf_status"] = "empty_target_neg_z"
        return pd.Series(np.nan, index=target_z.index, dtype=float), meta
    if likpf_finite.empty:
        meta["z_likpf_status"] = "empty_likpf_mean"
        return pd.Series(np.nan, index=target_z.index, dtype=float), meta

    source_min = float(target_neg_z_finite.min())
    source_max = float(target_neg_z_finite.max())
    target_min = float(likpf_finite.min())
    target_max = float(likpf_finite.max())
    if source_max == source_min:
        meta["z_likpf_status"] = "constant_hidden_neg_z"
        meta["z_likpf_hidden_neg_z_min"] = source_min
        meta["z_likpf_hidden_neg_z_max"] = source_max
        meta["z_likpf_target_min"] = target_min
        meta["z_likpf_target_max"] = target_max
        return pd.Series(np.nan, index=target_z.index, dtype=float), meta
    if target_max == target_min:
        meta["z_likpf_status"] = "constant_likpf_mean"
        meta["z_likpf_hidden_neg_z_min"] = source_min
        meta["z_likpf_hidden_neg_z_max"] = source_max
        meta["z_likpf_target_min"] = target_min
        meta["z_likpf_target_max"] = target_max
        return pd.Series(np.nan, index=target_z.index, dtype=float), meta

    meta["z_likpf_hidden_neg_z_min"] = source_min
    meta["z_likpf_hidden_neg_z_max"] = source_max
    meta["z_likpf_target_min"] = target_min
    meta["z_likpf_target_max"] = target_max

    target_values = target_neg_z.to_numpy(dtype=float)
    progress = (target_values - source_min) / (source_max - source_min)
    values = target_min + progress * (target_max - target_min)
    finite_value_mask = np.isfinite(values)
    if not bool(finite_value_mask.any()):
        meta["z_likpf_status"] = "empty_scaled_values"
        return pd.Series(np.nan, index=target_z.index, dtype=float), meta
    values[~np.isfinite(target_values)] = np.nan
    return pd.Series(values, index=target_z.index, dtype=float), meta


def finite_quantile(values: pd.Series, q: float) -> float:
    values = pd.to_numeric(values, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return float("nan")
    return float(finite.quantile(q))


def build_background_series(group: pd.DataFrame) -> dict[str, pd.Series]:
    target = pd.to_numeric(group["true_tvt"], errors="coerce")
    raw_tvt = pd.to_numeric(group["raw_TVT"], errors="coerce")
    scale_target = pd.concat([target, raw_tvt], ignore_index=True)
    scale_target = scale_target[np.isfinite(scale_target)]
    if scale_target.empty:
        return {}
    y_low = float(scale_target.quantile(0.02))
    y_high = float(scale_target.quantile(0.98))
    if not np.isfinite(y_low) or not np.isfinite(y_high) or y_low == y_high:
        y_low = float(scale_target.min())
        y_high = float(scale_target.max())

    common_values: list[pd.Series] = []
    for item in BACKGROUND_COLUMNS:
        if not bool(item.get("use_common_scale", True)):
            continue
        source_column = str(item.get("plot_column", f"raw_{item['name']}"))
        if source_column not in group:
            continue
        if bool(item.get("already_tvt_scale", False)):
            continue
        values = pd.to_numeric(group[source_column], errors="coerce")
        if item.get("transform") == "negate":
            values = -values
        values = values[np.isfinite(values)]
        if not values.empty:
            common_values.append(values)
    common_finite = pd.concat(common_values, ignore_index=True) if common_values else pd.Series(dtype=float)
    common_low = float(common_finite.quantile(0.02)) if not common_finite.empty else float("nan")
    common_high = float(common_finite.quantile(0.98)) if not common_finite.empty else float("nan")

    background_series: dict[str, pd.Series] = {}
    for item in BACKGROUND_COLUMNS:
        name = str(item["name"])
        source_column = str(item.get("plot_column", f"raw_{name}"))
        if source_column not in group:
            continue
        values = pd.to_numeric(group[source_column], errors="coerce")
        if item.get("transform") == "negate":
            values = -values
        if bool(item.get("already_tvt_scale", False)):
            background_series[name] = values
            continue
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        if bool(item.get("use_common_scale", True)):
            v_low = common_low
            v_high = common_high
        else:
            v_low = float(finite.quantile(0.02))
            v_high = float(finite.quantile(0.98))
        if not np.isfinite(v_low) or not np.isfinite(v_high) or v_low == v_high:
            continue
        background_series[name] = y_low + (values - v_low) * (y_high - y_low) / (v_high - v_low)
    return background_series


def add_background(
    ax: Any,
    x: pd.Series,
    group: pd.DataFrame,
    background_series: dict[str, pd.Series] | None = None,
) -> None:
    if background_series is None:
        background_series = build_background_series(group)
    for band in BACKGROUND_BANDS:
        upper = str(band["upper"])
        lower = str(band["lower"])
        if upper not in background_series or lower not in background_series:
            continue
        ax.fill_between(
            x,
            background_series[upper],
            background_series[lower],
            color=band["color"],
            alpha=float(band["alpha"]),
            linewidth=0,
            label=str(band["label"]),
            zorder=0,
        )
    for item in BACKGROUND_COLUMNS:
        name = str(item["name"])
        if name not in background_series:
            continue
        ax.plot(
            x,
            background_series[name],
            linewidth=float(item.get("linewidth", 1.0)),
            alpha=float(item.get("alpha", 0.2)),
            linestyle=str(item.get("linestyle", "-")),
            color=item.get("color"),
            label=str(item.get("label", name)),
            zorder=1,
        )


def trusted_tvt_axis_limits(group: pd.DataFrame) -> tuple[float, float]:
    columns = [
        "true_tvt",
        "exp148_lgb_mean_oof_tvt",
        "exp226_k16_oof_tvt",
        "last_anchor_tvt",
        "pf_ancc",
        "pf_z",
        "beam_mean",
        "likpf_mean",
        "z_likpf_minmax_tvt",
    ]
    values: list[np.ndarray] = []
    for column in columns:
        if column not in group:
            continue
        arr = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            values.append(arr)
    if not values:
        return float("nan"), float("nan")
    all_values = np.concatenate(values)
    y_min = float(np.nanmin(all_values))
    y_max = float(np.nanmax(all_values))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return float("nan"), float("nan")
    if y_min == y_max:
        margin = 5.0
    else:
        margin = max(8.0, 0.08 * float(y_max - y_min))
    return y_min - margin, y_max + margin


def add_exp209_hmm_outputs(
    ax: Any,
    hmm_group: pd.DataFrame,
    *,
    x_min: float,
    x_max: float,
) -> dict[str, Any]:
    empty_meta = {
        "rows": 0,
        "hmm_mean_segments": 0,
        "hmm_mean_points": 0,
        "hmm_mean_min": np.nan,
        "hmm_mean_max": np.nan,
        "hmm_2sigma_segments": 0,
        "hmm_2sigma_points": 0,
        "hmm_2sigma_min": np.nan,
        "hmm_2sigma_max": np.nan,
        "hmm_std_mean": np.nan,
        "hmm_loglik_mean": np.nan,
        "hmm_minus_likpf_mean_mean": np.nan,
    }
    if hmm_group.empty:
        return empty_meta

    meta = dict(empty_meta)
    meta["rows"] = int(len(hmm_group))
    x_values = pd.to_numeric(hmm_group["md_since"], errors="coerce").to_numpy(dtype=float)
    range_valid = np.isfinite(x_values) & (x_values >= x_min) & (x_values <= x_max)
    if not bool(range_valid.any()):
        return meta

    for source_column, prefix, label, style in [
        (
            EXP209_HMM_MEAN_COLUMN,
            "hmm_mean",
            "exp209 HMM mean",
            {"color": "#7c3aed", "linewidth": 1.9, "alpha": 0.90, "linestyle": "-", "zorder": 5.8},
        )
    ]:
        if not source_column or source_column not in hmm_group:
            continue
        y_values = pd.to_numeric(hmm_group[source_column], errors="coerce").to_numpy(dtype=float)
        valid = range_valid & np.isfinite(y_values)
        if source_column == EXP209_HMM_MEAN_COLUMN and "hmm_std" in hmm_group:
            std_values = pd.to_numeric(hmm_group["hmm_std"], errors="coerce").to_numpy(dtype=float)
            band_valid = valid & np.isfinite(std_values) & (std_values >= 0)
            if int(band_valid.sum()) >= 2:
                band_path = (
                    pd.DataFrame(
                        {
                            "x": x_values[band_valid],
                            "lower": y_values[band_valid] - 2.0 * std_values[band_valid],
                            "upper": y_values[band_valid] + 2.0 * std_values[band_valid],
                        }
                    )
                    .groupby("x", as_index=False)[["lower", "upper"]]
                    .median()
                    .sort_values("x")
                )
                ax.fill_between(
                    band_path["x"].to_numpy(dtype=float),
                    band_path["lower"].to_numpy(dtype=float),
                    band_path["upper"].to_numpy(dtype=float),
                    color="#8b5cf6",
                    alpha=0.13,
                    linewidth=0,
                    label="exp209 HMM +/-2sigma",
                    zorder=5.15,
                )
                meta["hmm_2sigma_segments"] = 1
                meta["hmm_2sigma_points"] = int(len(band_path))
                meta["hmm_2sigma_min"] = float(band_path["lower"].min())
                meta["hmm_2sigma_max"] = float(band_path["upper"].max())
            else:
                meta["hmm_2sigma_points"] = int(band_valid.sum())
        if int(valid.sum()) < 2:
            valid_count = int(valid.sum())
            meta[f"{prefix}_points"] = valid_count
            if valid_count:
                meta[f"{prefix}_min"] = float(np.nanmin(y_values[valid]))
                meta[f"{prefix}_max"] = float(np.nanmax(y_values[valid]))
            continue
        path = (
            pd.DataFrame({"x": x_values[valid], "y": y_values[valid]})
            .groupby("x", as_index=False)["y"]
            .median()
            .sort_values("x")
        )
        ax.plot(
            path["x"].to_numpy(dtype=float),
            path["y"].to_numpy(dtype=float),
            label=label,
            **style,
        )
        meta[f"{prefix}_segments"] = 1
        meta[f"{prefix}_points"] = int(len(path))
        meta[f"{prefix}_min"] = float(path["y"].min())
        meta[f"{prefix}_max"] = float(path["y"].max())

    valid_hmm = range_valid & np.isfinite(pd.to_numeric(hmm_group[EXP209_HMM_MEAN_COLUMN], errors="coerce").to_numpy(dtype=float))
    for source_column, key in [
        ("hmm_std", "hmm_std_mean"),
        ("hmm_loglik", "hmm_loglik_mean"),
        ("hmm_minus_likpf_mean", "hmm_minus_likpf_mean_mean"),
    ]:
        if source_column in hmm_group and bool(valid_hmm.any()):
            values = pd.to_numeric(hmm_group[source_column], errors="coerce").to_numpy(dtype=float)
            finite = values[valid_hmm]
            finite = finite[np.isfinite(finite)]
            if len(finite):
                meta[key] = float(np.nanmean(finite))
    return meta


def exp209_metric_value(
    exp209_metric: dict[str, dict[str, Any]],
    candidate: str,
    field: str,
) -> Any:
    return (exp209_metric.get(candidate) or {}).get(field)


def exp209_title_line(exp209_metric: dict[str, dict[str, Any]]) -> str:
    if not exp209_metric:
        return ""
    return (
        "\n"
        f"exp209 HMM RMSE {fmt_rmse(exp209_metric_value(exp209_metric, EXP209_HMM_MEAN_COLUMN, 'rmse'))} | "
        f"vs likPF {fmt_rmse(exp209_metric_value(exp209_metric, 'exp072_likpf_mean', 'rmse'))}"
    )


def exp209_overlay_range(meta: dict[str, Any]) -> list[float]:
    overlay_values: list[float] = []
    for min_key, max_key in [
        ("hmm_mean_min", "hmm_mean_max"),
        ("hmm_2sigma_min", "hmm_2sigma_max"),
    ]:
        min_value = meta.get(min_key)
        max_value = meta.get(max_key)
        if min_value is None or max_value is None:
            continue
        min_float = float(min_value)
        max_float = float(max_value)
        if np.isfinite(min_float) and np.isfinite(max_float):
            overlay_values.extend([min_float, max_float])
    return overlay_values


def add_derivative_panel(ax: Any, x: pd.Series, group: pd.DataFrame) -> None:
    z = pd.to_numeric(group["raw_Z"], errors="coerce")
    dx = pd.to_numeric(x, errors="coerce").diff()
    dzdmd = z.diff() / dx.where(dx.abs() > 1e-12)
    finite = dzdmd[np.isfinite(dzdmd)]
    if not finite.empty:
        clip = float(finite.abs().quantile(0.995))
        if np.isfinite(clip) and clip > 0:
            dzdmd = dzdmd.clip(-clip, clip)
    ax.axhline(0.0, color="#94a3b8", linewidth=0.8, alpha=0.8)
    ax.plot(x, dzdmd, color="#0f172a", linewidth=1.1, alpha=0.85, label="dZ/dMD")
    ax.set_ylabel("dZ/dMD")
    ax.grid(True, color="#e2e8f0", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper right", fontsize=8)


def plot_one_well(
    well_id: str,
    group: pd.DataFrame,
    exp209_hmm_group: pd.DataFrame,
    exp209_metric: dict[str, dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group = group.sort_values("md_since").reset_index(drop=True)
    source_rows = int(len(group))
    well_rmse_metrics = prediction_rmse_metrics(group)
    group = downsample_for_plot(group, MAX_POINTS_PER_PLOT)
    raw = read_raw_well(well_id)
    group = attach_raw_context(group, raw)
    z_likpf_minmax_tvt, z_likpf_meta = simple_likpf_minmax_neg_z_to_tvt(
        group["raw_Z"],
        group["likpf_mean"],
    )
    group["z_likpf_minmax_tvt"] = z_likpf_minmax_tvt

    x = pd.to_numeric(group["md_since"], errors="coerce")
    true_tvt = pd.to_numeric(group["true_tvt"], errors="coerce")
    ml_oof = pd.to_numeric(group["exp148_lgb_mean_oof_tvt"], errors="coerce")
    exp226_oof = pd.to_numeric(group["exp226_k16_oof_tvt"], errors="coerce")

    fig, (ax, ax_deriv) = plt.subplots(
        2,
        1,
        figsize=(13.0, 8.9),
        dpi=140,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )

    background_series = build_background_series(group)
    add_background(ax, x, group, background_series=background_series)
    ax.plot(x, true_tvt, color="black", linewidth=2.2, label="true TVT", zorder=5)

    line_styles = {
        "last_anchor_tvt": {"color": "#64748b", "linewidth": 1.4, "linestyle": "--", "alpha": 0.82},
        "pf_ancc": {"color": "#1f77b4", "linewidth": 1.45, "linestyle": "-", "alpha": 0.95},
        "pf_z": {"color": "#0891b2", "linewidth": 1.25, "linestyle": "-", "alpha": 0.86},
        "beam_mean": {"color": "#ff7f0e", "linewidth": 1.45, "linestyle": "-", "alpha": 0.95},
        "likpf_mean": {"color": "#2ca02c", "linewidth": 1.6, "linestyle": "-", "alpha": 0.95},
    }
    labels = {str(spec["name"]): str(spec["label"]) for spec in CANDIDATE_SPECS}
    for column, style in line_styles.items():
        if column not in group:
            continue
        ax.plot(x, pd.to_numeric(group[column], errors="coerce"), label=labels[column], zorder=3, **style)

    ax.plot(
        x,
        ml_oof,
        color="#e11d48",
        linewidth=1.9,
        linestyle="-",
        alpha=0.95,
        label="exp148 ML OOF lgb_mean",
        zorder=4,
    )
    ax.plot(
        x,
        exp226_oof,
        color="#a16207",
        linewidth=1.65,
        linestyle="-.",
        alpha=0.92,
        label="exp226 K16 OOF",
        zorder=4.2,
    )
    x_values_for_range = x.to_numpy(dtype=float)
    finite_x = x_values_for_range[np.isfinite(x_values_for_range)]
    if len(finite_x):
        x_min = float(np.nanmin(finite_x))
        x_max = float(np.nanmax(finite_x))
    else:
        x_min = float("-inf")
        x_max = float("inf")
    y_min, y_max = trusted_tvt_axis_limits(group)
    exp209_hmm_meta = add_exp209_hmm_outputs(
        ax,
        exp209_hmm_group,
        x_min=x_min,
        x_max=x_max,
    )
    display_y_min = y_min
    display_y_max = y_max
    overlay_values = exp209_overlay_range(exp209_hmm_meta)
    if overlay_values:
        if np.isfinite(display_y_min) and np.isfinite(display_y_max):
            axis_min = min(float(display_y_min), *overlay_values)
            axis_max = max(float(display_y_max), *overlay_values)
        else:
            axis_min = min(overlay_values)
            axis_max = max(overlay_values)
        margin = 5.0 if axis_min == axis_max else max(8.0, 0.04 * float(axis_max - axis_min))
        display_y_min = axis_min - margin
        display_y_max = axis_max + margin

    add_derivative_panel(ax_deriv, x, group)
    best1_label = well_rmse_metrics.get("pfbeam_best1_label") or "n/a"
    exp209_title = exp209_title_line(exp209_metric)
    ax.set_title(
        (
            f"{well_id} | exp148 CV(OOF) RMSE {fmt_rmse(well_rmse_metrics['exp148_oof_rmse'])} | "
            f"exp226 CV(OOF) RMSE {fmt_rmse(well_rmse_metrics['exp226_oof_rmse'])}\n"
            f"PF/Beam oracle RMSE {fmt_rmse(well_rmse_metrics['pfbeam_oracle_rmse'])} | "
            f"PF/Beam best1 RMSE {fmt_rmse(well_rmse_metrics['pfbeam_best1_rmse'])} ({best1_label})"
            f"{exp209_title}"
        ),
        fontsize=11,
    )
    ax.set_ylabel("TVT")
    if np.isfinite(display_y_min) and np.isfinite(display_y_max) and display_y_min < display_y_max:
        if TVT_AXIS_INVERTED:
            ax.set_ylim(display_y_max, display_y_min)
        else:
            ax.set_ylim(display_y_min, display_y_max)
    elif TVT_AXIS_INVERTED:
        ax.invert_yaxis()
    ax.grid(True, color="#e2e8f0", linewidth=0.7, alpha=0.8)
    ax.legend(loc="best", fontsize=7.8, ncol=3)
    ax_deriv.set_xlabel("md_since")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return {
        "well": well_id,
        "source_rows": source_rows,
        "rows": int(len(group)),
        "x_min": float(np.nanmin(x.to_numpy(dtype=float))) if len(group) else None,
        "x_max": float(np.nanmax(x.to_numpy(dtype=float))) if len(group) else None,
        "ml_oof_coverage": float(ml_oof.notna().mean()),
        "exp226_oof_coverage": float(exp226_oof.notna().mean()),
        "z_likpf_minmax_coverage": float(z_likpf_minmax_tvt.notna().mean()),
        "exp209_hmm_rows": exp209_hmm_meta["rows"],
        "exp209_hmm_mean_segments": exp209_hmm_meta["hmm_mean_segments"],
        "exp209_hmm_mean_points": exp209_hmm_meta["hmm_mean_points"],
        "exp209_hmm_mean_min": exp209_hmm_meta["hmm_mean_min"],
        "exp209_hmm_mean_max": exp209_hmm_meta["hmm_mean_max"],
        "exp209_hmm_2sigma_segments": exp209_hmm_meta["hmm_2sigma_segments"],
        "exp209_hmm_2sigma_points": exp209_hmm_meta["hmm_2sigma_points"],
        "exp209_hmm_2sigma_min": exp209_hmm_meta["hmm_2sigma_min"],
        "exp209_hmm_2sigma_max": exp209_hmm_meta["hmm_2sigma_max"],
        "exp209_hmm_std_mean": exp209_hmm_meta["hmm_std_mean"],
        "exp209_hmm_loglik_mean": exp209_hmm_meta["hmm_loglik_mean"],
        "exp209_hmm_minus_likpf_mean_mean": exp209_hmm_meta["hmm_minus_likpf_mean_mean"],
        "trusted_y_min": y_min,
        "trusted_y_max": y_max,
        "display_y_min": display_y_min,
        "display_y_max": display_y_max,
        "exp209_hmm_rmse": exp209_metric_value(exp209_metric, EXP209_HMM_MEAN_COLUMN, "rmse"),
        "exp209_hmm_delta_rmse_vs_likpf": exp209_metric_value(
            exp209_metric,
            EXP209_HMM_MEAN_COLUMN,
            "delta_rmse_vs_exp072_likpf_mean",
        ),
        "exp209_likpf_rmse": exp209_metric_value(exp209_metric, "exp072_likpf_mean", "rmse"),
        "plot_path": str(output_path),
        **well_rmse_metrics,
        **z_likpf_meta,
    }

# %% [markdown]
# ## 5. Generate all-well plots

# %%
plot_rows: list[dict[str, Any]] = []
for index, well_id in enumerate(plot_wells, start=1):
    group_index = plot_indices_by_well.get(well_id, [])
    group = plot_frame.iloc[group_index].copy()
    exp209_index = exp209_hmm_indices_by_well.get(well_id)
    if exp209_index is None:
        exp209_hmm_group = exp209_enriched_hmm.iloc[0:0].copy()
    else:
        exp209_hmm_group = exp209_enriched_hmm.iloc[exp209_index].copy()
    exp209_metric = exp209_well_metrics.get(well_id, {})
    output_path = PLOTS_DIR / f"{well_id}.png"
    plot_rows.append(
        plot_one_well(
            well_id,
            group,
            exp209_hmm_group,
            exp209_metric,
            output_path,
        )
    )
    if index % 50 == 0 or index == len(plot_wells):
        print(f"wrote {index}/{len(plot_wells)} plots")

manifest = pd.DataFrame(plot_rows)
manifest_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_plot_manifest.csv"
manifest.to_csv(manifest_path, index=False)

zip_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_plots.zip"
if ZIP_PLOTS:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for png_path in sorted(PLOTS_DIR.glob("*.png")):
            zf.write(png_path, arcname=png_path.name)

print("Manifest:", manifest_path)
print("Plots dir:", PLOTS_DIR)
print("Plots zip:", zip_path if zip_path.exists() else None)
print("Manifest rows:", len(manifest))
if not manifest.empty and "exp226_oof_coverage" in manifest:
    print("exp226 OOF coverage min:", float(manifest["exp226_oof_coverage"].min()))
    print("exp226 OOF coverage max:", float(manifest["exp226_oof_coverage"].max()))
if not manifest.empty and "z_likpf_status" in manifest:
    print("Z-to-likPF simple minmax status counts:", manifest["z_likpf_status"].value_counts(dropna=False).to_dict())
    print("Z-to-likPF simple minmax coverage min:", float(manifest["z_likpf_minmax_coverage"].min()))
    print(
        "Z-to-likPF target min range:",
        float(manifest["z_likpf_target_min"].min()),
        float(manifest["z_likpf_target_min"].max()),
    )
    print(
        "Z-to-likPF target max range:",
        float(manifest["z_likpf_target_max"].min()),
        float(manifest["z_likpf_target_max"].max()),
    )
if not manifest.empty and "exp209_hmm_rows" in manifest:
    print("exp209 HMM rows per well min:", int(manifest["exp209_hmm_rows"].min()))
    print("exp209 HMM rows per well max:", int(manifest["exp209_hmm_rows"].max()))
    print(
        "exp209 HMM mean points per well min:",
        int(manifest["exp209_hmm_mean_points"].min()),
    )
    print(
        "exp209 HMM mean points per well max:",
        int(manifest["exp209_hmm_mean_points"].max()),
    )
    print(
        "exp209 HMM mean TVT range:",
        float(manifest["exp209_hmm_mean_min"].min()),
        float(manifest["exp209_hmm_mean_max"].max()),
    )
    print(
        "exp209 HMM +/-2sigma points per well min:",
        int(manifest["exp209_hmm_2sigma_points"].min()),
    )
    print(
        "exp209 HMM +/-2sigma points per well max:",
        int(manifest["exp209_hmm_2sigma_points"].max()),
    )
    print(
        "exp209 HMM +/-2sigma TVT range:",
        float(manifest["exp209_hmm_2sigma_min"].min()),
        float(manifest["exp209_hmm_2sigma_max"].max()),
    )
print(manifest.head().to_string(index=False))

# %% [markdown]
# ## 6. Summary and generated outputs

# %%
summary = {
    "experiment": EXPERIMENT_NAME,
    "notebook": "exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb",
    "created_at_utc": datetime.now(UTC).isoformat(),
    "diagnostic_only": True,
    "plot_scope": "all_wells",
    "tvt_input_prefix_plotted": False,
    "prediction_start_line_plotted": False,
    "known_tvt_probe_plotted": False,
    "tvt_axis_inverted": bool(TVT_AXIS_INVERTED),
    "visual_guides": {
        "z_likpf_minmax_scaling": {
            "method": "plot_neg_z_minmax_to_likelihood_pf_mean_range",
            "anchor_used": False,
            "known_tail_direction_used": False,
            "clip_to_range": False,
            "plotted_on": "exp072_feature_cache_rows_only",
            "known_tvt_input_prefix_plotted": False,
            "known_tvt_input_used_for_scaling": False,
            "hidden_true_tvt_used_for_scaling": False,
            "target_range_source": "generated_likelihood_pf_mean",
            "manifest_status_counts": (
                {str(key): int(value) for key, value in manifest["z_likpf_status"].value_counts(dropna=False).items()}
                if not manifest.empty and "z_likpf_status" in manifest
                else {}
            ),
            "manifest_coverage_min": (
                float(manifest["z_likpf_minmax_coverage"].min())
                if not manifest.empty and "z_likpf_minmax_coverage" in manifest
                else None
            ),
        },
        "formation_background": {
            "plotted": False,
            "columns": FORMATION_COLUMNS,
        },
    },
    "exp148_recorded_cv_rmse": float(EXP148_RECORDED_CV_RMSE),
    "exp226_recorded_cv_rmse": float(EXP226_RECORDED_CV_RMSE),
    "rmse_metrics": {
        "plot_title_scope": "per_well",
        "plot_title_fields": [
            "exp148_oof_rmse",
            "exp226_oof_rmse",
            "pfbeam_oracle_rmse",
            "pfbeam_best1_rmse",
            "pfbeam_best1_label",
            "exp209_hmm_rmse",
            "exp209_likpf_rmse",
        ],
        "global_reference": jsonable_metrics(global_rmse_metrics),
        "exp209_overall_metrics": {
            "exp072_likpf_mean": exp209_global_metric("exp072_likpf_mean"),
            "hmm_mean_tvt": exp209_global_metric(EXP209_HMM_MEAN_COLUMN),
        },
    },
    "source_files": {
        "pfbeam": str(pfbeam_path),
        "exp148_oof": str(exp148_oof_path),
        "exp226_oof": str(exp226_oof_path),
        "exp209_enriched_hmm": str(exp209_enriched_hmm_path),
        "exp209_by_well_delta": str(exp209_by_well_delta_path),
        "exp209_overall_metrics": str(exp209_overall_metrics_path),
        "exp209_summary": str(exp209_summary_path),
        "raw_train_dir": str(raw_train_dir),
    },
    "source_sha256": {
        "pfbeam_gzip": sha256_path(pfbeam_path, decompressed=False) if pfbeam_path.suffix == ".gz" else None,
        "pfbeam_decompressed": sha256_path(pfbeam_path, decompressed=True) if pfbeam_path.suffix == ".gz" else sha256_path(pfbeam_path),
        "exp148_oof_gzip": sha256_path(exp148_oof_path, decompressed=False) if exp148_oof_path.suffix == ".gz" else None,
        "exp148_oof_decompressed": sha256_path(exp148_oof_path, decompressed=True) if exp148_oof_path.suffix == ".gz" else sha256_path(exp148_oof_path),
        "exp226_oof_gzip": sha256_path(exp226_oof_path, decompressed=False) if exp226_oof_path.suffix == ".gz" else None,
        "exp226_oof_decompressed": sha256_path(exp226_oof_path, decompressed=True) if exp226_oof_path.suffix == ".gz" else sha256_path(exp226_oof_path),
        "exp209_enriched_hmm_gzip": (
            sha256_path(exp209_enriched_hmm_path, decompressed=False)
            if exp209_enriched_hmm_path.suffix == ".gz"
            else None
        ),
        "exp209_enriched_hmm_decompressed": (
            sha256_path(exp209_enriched_hmm_path, decompressed=True)
            if exp209_enriched_hmm_path.suffix == ".gz"
            else sha256_path(exp209_enriched_hmm_path)
        ),
        "exp209_by_well_delta": sha256_path(exp209_by_well_delta_path),
        "exp209_overall_metrics": sha256_path(exp209_overall_metrics_path),
        "exp209_summary": sha256_path(exp209_summary_path),
    },
    "exp148_oof_filter": {
        "variant": EXP148_VARIANT,
        "mode": EXP148_MODE,
        "model": EXP148_MODEL,
    },
    "rows": {
        "pfbeam": int(len(pfbeam)),
        "exp148_oof_filtered": int(len(exp148_oof)),
        "exp226_oof": int(len(exp226_oof)),
        "exp209_enriched_hmm": int(len(exp209_enriched_hmm)),
        "exp209_by_well_delta": int(len(exp209_by_well_delta)),
        "joined": int(len(plot_frame)),
    },
    "wells": {
        "source": int(len(all_wells)),
        "plotted": int(len(plot_wells)),
    },
    "coverage": {
        "ml_oof": float(plot_frame["exp148_lgb_mean_oof_tvt"].notna().mean()),
        "manifest_ml_oof_min": float(manifest["ml_oof_coverage"].min()) if not manifest.empty else None,
        "exp226_oof": float(plot_frame["exp226_k16_oof_tvt"].notna().mean()),
        "manifest_exp226_oof_min": float(manifest["exp226_oof_coverage"].min()) if not manifest.empty else None,
        "manifest_z_likpf_minmax_min": (
            float(manifest["z_likpf_minmax_coverage"].min())
            if not manifest.empty and "z_likpf_minmax_coverage" in manifest
            else None
        ),
        "exp209_hmm_md_since": float(exp209_enriched_hmm["md_since"].notna().mean()),
        "manifest_exp209_hmm_rows_min": (
            int(manifest["exp209_hmm_rows"].min())
            if not manifest.empty and "exp209_hmm_rows" in manifest
            else None
        ),
        "manifest_exp209_hmm_mean_points_min": (
            int(manifest["exp209_hmm_mean_points"].min())
            if not manifest.empty and "exp209_hmm_mean_points" in manifest
            else None
        ),
        "manifest_exp209_hmm_2sigma_points_min": (
            int(manifest["exp209_hmm_2sigma_points"].min())
            if not manifest.empty and "exp209_hmm_2sigma_points" in manifest
            else None
        ),
    },
    "exp209_overlay": {
        "plotted": True,
        "hmm_mean_column": EXP209_HMM_MEAN_COLUMN,
        "hmm_std_column": "hmm_std",
        "path_source": "exp209 enriched_hmm_exp072_train_features.csv.gz",
        "plot_method": "plot hmm_mean_tvt by md_since with a translucent hmm_mean_tvt +/- 2*hmm_std band; both contribute to the TVT axis range",
        "hmm_mean_style": "purple HMM mean line",
        "hmm_2sigma_band_style": "translucent purple fill_between band",
        "hmm_2sigma_formula": "lower=hmm_mean_tvt-2*hmm_std, upper=hmm_mean_tvt+2*hmm_std",
        "overall_metrics": [
            {str(key): jsonable(value) for key, value in row.items()}
            for row in exp209_overall_metrics.to_dict(orient="records")
            if str(row.get("candidate")) in {"exp072_likpf_mean", EXP209_HMM_MEAN_COLUMN}
        ],
        "parent_hmm_by_well_delta_summary": jsonable(exp209_summary.get("hmm_by_well_delta_summary", {})),
    },
    "outputs": {
        "manifest": str(manifest_path),
        "plots_dir": str(PLOTS_DIR),
        "plots_zip": str(zip_path) if zip_path.exists() else None,
    },
    "notes": [
        "This is a train-side visualization notebook only.",
        "Known TVT probe is not plotted.",
        "The -Z guide directly min-max scales plotted -Z to the generated Likelihood PF mean range.",
        "The -Z guide does not use known-tail direction, last-known-TVT anchor shifting, or final clipping.",
        "Plot titles show per-well exp148 OOF RMSE, exp226 OOF RMSE, and per-well PF/Beam oracle/best1 RMSE.",
        "The upper TVT panel uses an inverted depth-down y-axis; RMSE metrics are unchanged.",
        "exp226 K16 spline/kernel-kNN OOF predictions are plotted from exp226 train output after well_id,row_idx to id,well conversion.",
        "Previous learned-MTP overlay results are not plotted.",
        "Formation boundaries and formation filled bands are not plotted.",
        "PNG filenames use the well id only, without an all_wells prefix.",
        "exp209 HMM mean and a translucent +/-2sigma band are plotted from enriched_hmm_exp072_train_features.",
        "The exp209 HMM mean line and +/-2sigma band are included in the TVT axis range so scale mismatch remains visible instead of being hidden by y-filtering.",
        "The HMM +/-2sigma band is a posterior-uncertainty visualization guide, not a guaranteed calibrated 95% interval.",
        "exp209 x uses md_since joined from the exp072 plot frame by id/well.",
        "exp209 title metrics are train-side by-well direct comparison readouts and are not hidden-test prediction scores.",
        "TVT_input known prefix rows are not added to the plot frame.",
        "Hidden true TVT is not used for the -Z guide.",
        "Prediction-start vertical line is not plotted.",
        "No model training, PF/Beam regeneration, inference, or submission is performed.",
    ],
}
summary_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

print(json.dumps(summary, indent=2, sort_keys=True))
print("Summary:", summary_path)

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp197 cnn pf likelihood probe train
#
# Train-side diagnostic for backlog `cnn_pf_likelihood_probe`.
# This notebook learns a candidate-level local CNN/SDF likelihood over fixed
# exp099 PF/Beam/likPF candidates. It does not rerun PF, replace live PF
# particle weights, build raw-test features, or create a submission.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and reproducibility helpers
# 3. Candidate cache and baseline helpers
# 4. Raw well window helpers
# 5. Candidate index and CNN dataset
# 6. CNN likelihood model and training helpers
# 7. Diagnostic metrics
# 8. Setup and input checks
# 9. Train CNN likelihood variants
# 10. Metrics, SHA, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from settings import EXPERIMENT_NAME, KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# %% [markdown]
# ## 2. Runtime and reproducibility helpers

# %%
OUTPUT_PREFIX = EXPERIMENT_NAME
EXP099_FEATURE_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
EXP099_FEATURE_SCHEMA = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv"
)
EXP111_OOF_LONG = "exp111_learned_pf_observation_likelihood_probe_oof_likelihood_long.csv.gz"
IMAGE_CHANNEL_SCHEMA = [
    ("typewell_gr_window", "Typewell GR sampled around the fixed candidate TVT."),
    ("horizontal_gr_window", "Horizontal GR sampled around the scored row."),
    ("typewell_minus_horizontal_gr", "Pairwise local GR difference."),
    (
        "candidate_sdf_to_observed_tvt_input",
        "Candidate TVT grid minus observed TVT_input prefix where available.",
    ),
    ("observed_tvt_input_mask", "1 where the horizontal window has observed TVT_input."),
]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy())
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: str, modulo: int | None = None) -> int:
    payload = "::".join(parts).encode("utf-8")
    value = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    return value % int(modulo) if modulo else value


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def require_cuda_device(config: dict[str, Any]) -> torch.device:
    require_cuda = bool(get_nested(config, "model.training.require_cuda"))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "This experiment requires Kaggle GPU. CPU fallback is disabled by config."
        )
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        min_major = int(get_nested(config, "model.training.min_cuda_capability_major") or 0)
        if capability[0] < min_major:
            raise RuntimeError(
                "Allocated GPU is below the configured capability: "
                f"device={torch.cuda.get_device_name(0)!r}, capability={capability}, "
                f"required_major>={min_major}."
            )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def prediction_sha256(frame: pd.DataFrame, *, value_col: str) -> str:
    digest = hashlib.sha256()
    ordered = frame[["variant", "id", "candidate_name", value_col]].sort_values(
        ["variant", "id", "candidate_name"]
    )
    for row in ordered.itertuples(index=False):
        digest.update(str(row.variant).encode("utf-8"))
        digest.update(b",")
        digest.update(str(row.id).encode("utf-8"))
        digest.update(b",")
        digest.update(str(row.candidate_name).encode("utf-8"))
        digest.update(b",")
        digest.update(np.float64(row[3]).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def find_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    search_dirs: list[str | Path] | None = None,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        path = Path(explicit_path)
        candidates.append(path / filename if path.is_dir() else path)
    for directory in search_dirs or []:
        path = Path(directory)
        candidates.append(path / filename if path.is_dir() else path)
    candidates.extend([Path.cwd() / filename, Path.cwd() / "artifacts" / filename])
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:120])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def finite_float_array(
    series: pd.Series | None,
    fallback: float = 0.0,
    length: int | None = None,
) -> np.ndarray:
    if series is None:
        if length is None:
            raise ValueError("length is required when series is None")
        return np.full(length, fallback, dtype=np.float32)
    values = pd.to_numeric(series, errors="coerce")
    values = values.interpolate(limit_direction="both").ffill().bfill()
    fill_value = float(values.dropna().median()) if values.notna().any() else fallback
    return values.fillna(fill_value).to_numpy(np.float32)


def robust_zscore(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros_like(values, dtype=np.float32)
    median = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-6:
        std = float(np.std(finite))
        scale = std if std > 1e-6 else 1.0
    return np.clip((values - median) / scale, -8.0, 8.0).astype(np.float32)


def fill_and_scale_gr(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.to_numeric(series, errors="coerce")
    missing = raw.isna().to_numpy(np.float32)
    filled = raw.interpolate(limit_direction="both").ffill().bfill()
    fallback = float(filled.dropna().median()) if filled.notna().any() else 0.0
    return robust_zscore(filled.fillna(fallback).to_numpy(np.float32)), missing


def parse_row_index(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int32)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred.astype(np.float64) - y_true.astype(np.float64)))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred.astype(np.float64) - y_true.astype(np.float64))))


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    mask = np.isfinite(score)
    if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
        return None
    return float(roc_auc_score(y_true[mask], score[mask]))


def safe_logloss(y_true: np.ndarray, prob: np.ndarray) -> float | None:
    mask = np.isfinite(prob)
    if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
        return None
    return float(log_loss(y_true[mask], np.clip(prob[mask], 1e-6, 1.0 - 1e-6)))


def softmax_np(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values.astype(np.float64) / max(float(temperature), 1e-6)
    scaled = scaled - np.nanmax(scaled)
    exp_values = np.exp(np.nan_to_num(scaled, nan=-1e9))
    total = exp_values.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(values), 1.0 / max(len(values), 1), dtype=np.float64)
    return exp_values / total


# %% [markdown]
# ## 3. Candidate cache and baseline helpers

# %%
@dataclass(frozen=True)
class CandidateSpec:
    name: str
    column: str


def candidate_specs_from_config(config: dict[str, Any]) -> list[CandidateSpec]:
    raw_specs = get_nested(config, "model.candidates") or []
    specs: list[CandidateSpec] = []
    for item in raw_specs:
        if not isinstance(item, dict):
            raise ValueError("model.candidates entries must be mappings")
        specs.append(CandidateSpec(name=str(item["name"]), column=str(item["column"])))
    if not specs:
        raise ValueError("model.candidates must not be empty")
    return specs


def read_exp099_cache(
    *,
    config: dict[str, Any],
    candidates: list[CandidateSpec],
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_path = find_artifact(
        EXP099_FEATURE_CACHE,
        get_nested(config, "data.exp099_train_feature_cache_local"),
        search_dirs=[
            Path("experiments")
            / "exp099_pf_multi_observation_likelihood_probe"
            / "kaggle"
            / "output"
            / "train_v2"
            / "artifacts"
        ],
    )
    schema_path = find_artifact(
        EXP099_FEATURE_SCHEMA,
        get_nested(config, "data.exp099_train_feature_schema_local"),
        search_dirs=[
            Path("experiments")
            / "exp099_pf_multi_observation_likelihood_probe"
            / "kaggle"
            / "output"
            / "train_v2"
            / "artifacts"
        ],
    )
    header = pd.read_csv(cache_path, nrows=0).columns.tolist()
    required = {"id", "well", "target", "last_known_tvt"}
    required.update(spec.column for spec in candidates)
    for spec in candidates:
        required.update(
            {
                f"multiobs_score_{spec.name}",
                f"multiobs_mae_{spec.name}",
                f"multiobs_ncc_{spec.name}",
            }
        )
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"{cache_path} is missing required columns: {missing}")

    optional = [
        str(column)
        for column in (get_nested(config, "model.scalar_context_columns") or [])
        if str(column) in header
    ]
    usecols = sorted(required.union(optional))
    frame = pd.read_csv(
        cache_path,
        usecols=usecols,
        nrows=max_rows,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    frame["row_index"] = parse_row_index(frame["id"])
    frame["true_tvt"] = frame["last_known_tvt"].astype(np.float32) + frame["target"].astype(
        np.float32
    )
    meta = {
        "path": str(cache_path),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns_loaded": usecols,
        "source_sha256": sha256_path(cache_path),
        "source_decompressed_sha256": sha256_path(cache_path, decompressed=True),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_path(schema_path),
    }
    return frame, meta


def load_exp111_oof(config: dict[str, Any]) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    explicit_dir = get_nested(config, "data.exp111_artifact_dir_local")
    try:
        path = find_artifact(EXP111_OOF_LONG, explicit_dir)
    except FileNotFoundError as exc:
        return None, {"loaded": False, "reason": str(exc)}
    usecols = [
        "id",
        "candidate_name",
        "pred_within10_prob",
        "pred_abs_error",
        "baseline_multiobs_score",
        "baseline_multiobs_mae",
        "baseline_multiobs_ncc",
    ]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [column for column in usecols if column not in header]
    if missing:
        return None, {"loaded": False, "path": str(path), "missing_columns": missing}
    frame = pd.read_csv(path, usecols=usecols, dtype={"id": str, "candidate_name": str})
    frame = frame.rename(
        columns={
            "pred_within10_prob": "exp111_pred_within10_prob",
            "pred_abs_error": "exp111_pred_abs_error",
            "baseline_multiobs_score": "exp111_baseline_multiobs_score",
            "baseline_multiobs_mae": "exp111_baseline_multiobs_mae",
            "baseline_multiobs_ncc": "exp111_baseline_multiobs_ncc",
        }
    )
    meta = {
        "loaded": True,
        "path": str(path),
        "rows": int(len(frame)),
        "ids": int(frame["id"].nunique()),
        "sha256": sha256_path(path),
        "decompressed_sha256": sha256_path(path, decompressed=True),
    }
    return frame, meta


def make_fold_indices(
    frame: pd.DataFrame,
    *,
    n_folds: int,
    run_folds: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    cv = GroupKFold(n_splits=int(n_folds))
    splits = list(cv.split(frame, np.zeros(len(frame), dtype=np.float32), groups=frame["well"]))
    return [
        (fold, train_idx.astype(np.int64), valid_idx.astype(np.int64))
        for fold, (train_idx, valid_idx) in enumerate(splits[: int(run_folds)])
    ]


def sample_indices(indices: np.ndarray, *, max_rows: int | None, seed: int) -> np.ndarray:
    if max_rows is None or len(indices) <= int(max_rows):
        return np.sort(indices)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(indices, size=int(max_rows), replace=False))


# %% [markdown]
# ## 4. Raw well window helpers

# %%
@dataclass(frozen=True)
class WellArrays:
    well: str
    horizontal_rows: int
    typewell_rows: int
    md: np.ndarray
    z: np.ndarray
    tvt: np.ndarray
    tvt_input: np.ndarray
    horizontal_gr: np.ndarray
    horizontal_gr_missing: np.ndarray
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray
    typewell_gr_shuffled: np.ndarray
    prefix_end: int
    last_known_tvt: float
    last_known_z: float
    last_known_md: float


def read_well_arrays(
    well: str,
    train_dir: Path,
    *,
    seed: int,
    min_prefix_observed_rows: int,
) -> WellArrays | None:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    if not horizontal_path.exists() or not typewell_path.exists():
        return None
    h = pd.read_csv(horizontal_path)
    t = pd.read_csv(typewell_path)
    required_h = {"MD", "Z", "TVT", "TVT_input", "GR"}
    required_t = {"TVT", "GR"}
    if not required_h.issubset(h.columns) or not required_t.issubset(t.columns):
        return None

    tvt_input_raw = pd.to_numeric(h["TVT_input"], errors="coerce").to_numpy(np.float32)
    known = np.flatnonzero(np.isfinite(tvt_input_raw))
    if len(known) < int(min_prefix_observed_rows):
        return None
    prefix_end = int(known[-1])
    if prefix_end >= len(h) - 2:
        return None

    t = t.sort_values("TVT").reset_index(drop=True)
    typewell_tvt = finite_float_array(t["TVT"])
    typewell_gr, _ = fill_and_scale_gr(t["GR"])
    if len(typewell_tvt) < 32:
        return None

    roll = stable_int(EXPERIMENT_NAME, "shuffle-gr", well, str(seed), modulo=len(typewell_gr))
    typewell_gr_shuffled = np.roll(typewell_gr, int(roll)).astype(np.float32)
    horizontal_gr, horizontal_gr_missing = fill_and_scale_gr(h["GR"])
    md = finite_float_array(h["MD"])
    z = finite_float_array(h["Z"])
    tvt = finite_float_array(h["TVT"])
    tvt_input = pd.to_numeric(h["TVT_input"], errors="coerce").to_numpy(np.float32)

    return WellArrays(
        well=well,
        horizontal_rows=len(h),
        typewell_rows=len(t),
        md=md,
        z=z,
        tvt=tvt,
        tvt_input=tvt_input,
        horizontal_gr=horizontal_gr,
        horizontal_gr_missing=horizontal_gr_missing,
        typewell_tvt=typewell_tvt,
        typewell_gr=typewell_gr,
        typewell_gr_shuffled=typewell_gr_shuffled,
        prefix_end=prefix_end,
        last_known_tvt=float(tvt_input[prefix_end]),
        last_known_z=float(z[prefix_end]),
        last_known_md=float(md[prefix_end]),
    )


def load_well_arrays(
    wells: list[str],
    train_dir: Path,
    *,
    seed: int,
    min_prefix_observed_rows: int,
) -> dict[str, WellArrays]:
    arrays_by_well: dict[str, WellArrays] = {}
    for well in sorted(set(wells)):
        arrays = read_well_arrays(
            well,
            train_dir,
            seed=seed,
            min_prefix_observed_rows=min_prefix_observed_rows,
        )
        if arrays is not None:
            arrays_by_well[well] = arrays
    return arrays_by_well


def point_gr_scores(
    *,
    arrays: WellArrays,
    row_indices: np.ndarray,
    candidate_tvt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_gr = arrays.horizontal_gr[np.clip(row_indices, 0, arrays.horizontal_rows - 1)]
    real = np.interp(candidate_tvt, arrays.typewell_tvt, arrays.typewell_gr).astype(np.float32)
    shuffled = np.interp(candidate_tvt, arrays.typewell_tvt, arrays.typewell_gr_shuffled).astype(
        np.float32
    )
    real_score = -np.abs(real - h_gr).astype(np.float32)
    shuffled_score = -np.abs(shuffled - h_gr).astype(np.float32)
    no_gr_score = np.zeros_like(real_score, dtype=np.float32)
    return real_score, shuffled_score, no_gr_score


# %% [markdown]
# ## 5. Candidate index and CNN dataset

# %%
def build_candidate_index(
    *,
    frame: pd.DataFrame,
    fold: int,
    split: str,
    row_indices: np.ndarray,
    candidates: list[CandidateSpec],
    arrays_by_well: dict[str, WellArrays],
    config: dict[str, Any],
) -> pd.DataFrame:
    base = frame.iloc[row_indices].copy()
    base = base[base["well"].isin(arrays_by_well)].copy()
    if base.empty:
        raise RuntimeError(f"No rows remain for split={split} after raw well filtering")

    candidate_scale = float(get_nested(config, "model.training.candidate_scale_ft") or 200.0)
    rows: list[pd.DataFrame] = []
    for well, group in base.groupby("well", sort=True):
        arrays = arrays_by_well[str(well)]
        group = group[group["row_index"].between(0, arrays.horizontal_rows - 1)].copy()
        if group.empty:
            continue
        row_idx = group["row_index"].to_numpy(np.int32)
        candidate_matrix = np.column_stack(
            [pd.to_numeric(group[spec.column], errors="coerce").to_numpy(np.float32) for spec in candidates]
        )
        row_mean = np.nanmean(candidate_matrix, axis=1).astype(np.float32)
        row_std = np.nanstd(candidate_matrix, axis=1).astype(np.float32)
        row_std_safe = np.where(row_std > 1e-6, row_std, 1.0).astype(np.float32)
        row_range = (np.nanmax(candidate_matrix, axis=1) - np.nanmin(candidate_matrix, axis=1)).astype(
            np.float32
        )
        likpf_values = group["likpf_mean"].to_numpy(np.float32)

        for cand_idx, spec in enumerate(candidates):
            candidate_tvt = candidate_matrix[:, cand_idx].astype(np.float32)
            finite = np.isfinite(candidate_tvt)
            if not finite.any():
                continue
            part = pd.DataFrame(
                {
                    "fold": np.int16(fold),
                    "split": split,
                    "id": group["id"].to_numpy(dtype=object),
                    "well": group["well"].to_numpy(dtype=object),
                    "row_index": row_idx,
                    "candidate_index": np.int16(cand_idx),
                    "candidate_name": spec.name,
                    "candidate_tvt": candidate_tvt,
                    "true_tvt": group["true_tvt"].to_numpy(np.float32),
                    "last_known_tvt": group["last_known_tvt"].to_numpy(np.float32),
                    "target": group["target"].to_numpy(np.float32),
                    "candidate_minus_last": (
                        candidate_tvt - group["last_known_tvt"].to_numpy(np.float32)
                    ).astype(np.float32),
                    "candidate_abs_minus_likpf": np.abs(candidate_tvt - likpf_values).astype(
                        np.float32
                    ),
                    "candidate_abs_minus_row_mean": np.abs(candidate_tvt - row_mean).astype(
                        np.float32
                    ),
                    "candidate_z_within_row": ((candidate_tvt - row_mean) / row_std_safe).astype(
                        np.float32
                    ),
                    "row_candidate_std": row_std,
                    "row_candidate_range": row_range,
                    "candidate_multiobs_score": group[
                        f"multiobs_score_{spec.name}"
                    ].to_numpy(np.float32),
                    "candidate_multiobs_mae": group[f"multiobs_mae_{spec.name}"].to_numpy(
                        np.float32
                    ),
                    "candidate_multiobs_ncc": group[f"multiobs_ncc_{spec.name}"].to_numpy(
                        np.float32
                    ),
                }
            )
            real_score, shuffled_score, no_gr_score = point_gr_scores(
                arrays=arrays,
                row_indices=row_idx,
                candidate_tvt=candidate_tvt,
            )
            part["point_gr_score_real"] = real_score
            part["point_gr_score_shuffled"] = shuffled_score
            part["point_gr_score_no_gr"] = no_gr_score

            for column in get_nested(config, "model.scalar_context_columns") or []:
                column = str(column)
                if column in group.columns:
                    values = pd.to_numeric(group[column], errors="coerce").to_numpy(np.float32)
                    scale = 1000.0 if column in {"md_since", "eval_len"} else candidate_scale
                    part[f"context_{column}_scaled"] = (values / float(scale)).astype(np.float32)

            part = part.loc[finite].copy()
            rows.append(part)

    out = pd.concat(rows, ignore_index=True)
    out["abs_error"] = np.abs(out["candidate_tvt"] - out["true_tvt"]).astype(np.float32)
    out["within_5ft"] = (out["abs_error"] <= 5.0).astype(np.int8)
    out["within_10ft"] = (out["abs_error"] <= 10.0).astype(np.int8)
    out["candidate_index_norm"] = (
        out["candidate_index"].astype(np.float32) / max(len(candidates) - 1, 1)
    ).astype(np.float32)
    for column in [
        "candidate_minus_last",
        "candidate_abs_minus_likpf",
        "candidate_abs_minus_row_mean",
        "row_candidate_std",
        "row_candidate_range",
    ]:
        out[f"{column}_scaled"] = (out[column].astype(np.float32) / candidate_scale).astype(
            np.float32
        )
    out = out.sort_values(["fold", "split", "well", "row_index", "candidate_index"]).reset_index(
        drop=True
    )
    return out


def attach_exp111_oof(candidate_index: pd.DataFrame, exp111: pd.DataFrame | None) -> pd.DataFrame:
    if exp111 is None:
        candidate_index["exp111_pred_within10_prob"] = np.nan
        candidate_index["exp111_pred_abs_error"] = np.nan
        return candidate_index
    return candidate_index.merge(exp111, on=["id", "candidate_name"], how="left")


def scalar_feature_columns(candidate_index: pd.DataFrame) -> list[str]:
    preferred = [
        "candidate_index_norm",
        "candidate_minus_last_scaled",
        "candidate_abs_minus_likpf_scaled",
        "candidate_abs_minus_row_mean_scaled",
        "candidate_z_within_row",
        "row_candidate_std_scaled",
        "row_candidate_range_scaled",
    ]
    context = sorted(column for column in candidate_index.columns if column.startswith("context_"))
    columns = [column for column in preferred + context if column in candidate_index.columns]
    numeric = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(candidate_index[column])
        and candidate_index[column].notna().any()
    ]
    if not numeric:
        raise ValueError("No scalar feature columns were selected")
    return numeric


def fit_scalar_medians(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        value = float(values.median()) if values.notna().any() else 0.0
        medians[column] = value if np.isfinite(value) else 0.0
    return medians


def apply_scalar_medians(
    frame: pd.DataFrame,
    columns: list[str],
    medians: dict[str, float],
) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        values = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[column] = values.fillna(medians[column]).astype(np.float32)
    return out


class CandidateWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        candidate_index: pd.DataFrame,
        arrays_by_well: dict[str, WellArrays],
        scalar_columns: list[str],
        config: dict[str, Any],
        variant: str,
    ) -> None:
        self.candidate_index = candidate_index.reset_index(drop=True)
        self.arrays_by_well = arrays_by_well
        self.scalar_columns = list(scalar_columns)
        self.variant = str(variant)
        training = get_nested(config, "model.training") or {}
        horizontal_window = int(training.get("horizontal_window_rows", 96))
        self.horizontal_offsets = np.arange(
            -(horizontal_window // 2),
            horizontal_window // 2,
            dtype=np.int32,
        )
        grid_bins = int(training.get("typewell_window_bins", 64))
        half_width = float(training.get("tvt_window_half_width_ft", 128.0))
        self.grid_offsets_tvt = np.linspace(-half_width, half_width, grid_bins).astype(np.float32)
        self.history_scale_ft = float(training.get("history_scale_ft", 200.0))
        self.error_scale_ft = float(training.get("error_scale_ft", 40.0))

    def __len__(self) -> int:
        return len(self.candidate_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.candidate_index.iloc[index]
        arrays = self.arrays_by_well[str(row["well"])]
        row_center = int(row["row_index"])
        h_idx = np.clip(row_center + self.horizontal_offsets, 0, arrays.horizontal_rows - 1)
        grid_tvt = float(row["candidate_tvt"]) + self.grid_offsets_tvt

        if self.variant == "shuffled_gr":
            t_gr_source = arrays.typewell_gr_shuffled
        else:
            t_gr_source = arrays.typewell_gr
        t_gr = np.interp(grid_tvt, arrays.typewell_tvt, t_gr_source).astype(np.float32)
        h_gr = arrays.horizontal_gr[h_idx].astype(np.float32)
        if self.variant == "no_gr":
            t_gr = np.zeros_like(t_gr, dtype=np.float32)
            h_gr = np.zeros_like(h_gr, dtype=np.float32)

        t_heatmap = np.broadcast_to(t_gr.reshape(1, -1), (len(h_idx), len(grid_tvt)))
        h_heatmap = np.broadcast_to(h_gr.reshape(-1, 1), (len(h_idx), len(grid_tvt)))
        diff = t_heatmap - h_heatmap

        observed_tvt = arrays.tvt_input[h_idx]
        mask = np.isfinite(observed_tvt).astype(np.float32)
        observed_safe = np.where(np.isfinite(observed_tvt), observed_tvt, 0.0).astype(np.float32)
        sdf = (grid_tvt.reshape(1, -1) - observed_safe.reshape(-1, 1)) / self.history_scale_ft
        sdf = sdf * mask.reshape(-1, 1)
        mask_heatmap = np.broadcast_to(mask.reshape(-1, 1), sdf.shape)
        image = np.stack([t_heatmap, h_heatmap, diff, sdf, mask_heatmap], axis=0).astype(
            np.float32
        )

        scalar = row[self.scalar_columns].to_numpy(np.float32)
        abs_error = float(row["abs_error"])
        return {
            "sample_id": torch.tensor(int(row["sample_id"]), dtype=torch.long),
            "image": torch.from_numpy(image),
            "scalar": torch.from_numpy(scalar),
            "within_10ft": torch.tensor(float(row["within_10ft"]), dtype=torch.float32),
            "abs_error_scaled": torch.tensor(abs_error / self.error_scale_ft, dtype=torch.float32),
            "abs_error": torch.tensor(abs_error, dtype=torch.float32),
        }


# %% [markdown]
# ## 6. CNN likelihood model and training helpers

# %%
class CandidateLikelihoodNet(nn.Module):
    def __init__(
        self,
        *,
        image_channels: int,
        scalar_dim: int,
        channels: list[int],
        kernel_size: int,
        scalar_hidden: int,
        fusion_hidden: int,
        dropout: float,
        use_group_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = int(image_channels)
        padding = int(kernel_size) // 2
        for width in channels:
            width = int(width)
            layers.append(nn.Conv2d(current, width, kernel_size=int(kernel_size), padding=padding))
            if use_group_norm:
                groups = 8 if width % 8 == 0 else 1
                layers.append(nn.GroupNorm(groups, width))
            else:
                layers.append(nn.BatchNorm2d(width))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            current = width
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.scalar = nn.Sequential(
            nn.Linear(int(scalar_dim), int(scalar_hidden)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.head = nn.Sequential(
            nn.Linear(current + int(scalar_hidden), int(fusion_hidden)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.prob_head = nn.Linear(int(fusion_hidden), 1)
        self.error_head = nn.Linear(int(fusion_hidden), 1)

    def forward(self, image: torch.Tensor, scalar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_feature = self.pool(self.backbone(image)).flatten(1)
        scalar_feature = self.scalar(scalar)
        fused = self.head(torch.cat([image_feature, scalar_feature], dim=1))
        logit = self.prob_head(fused).squeeze(1)
        error_scaled = F.softplus(self.error_head(fused).squeeze(1))
        return logit, error_scaled


def make_model(config: dict[str, Any], *, scalar_dim: int) -> CandidateLikelihoodNet:
    arch = get_nested(config, "model.architecture") or {}
    return CandidateLikelihoodNet(
        image_channels=len(IMAGE_CHANNEL_SCHEMA),
        scalar_dim=int(scalar_dim),
        channels=[int(value) for value in arch.get("channels", [32, 64, 64])],
        kernel_size=int(arch.get("kernel_size", 3)),
        scalar_hidden=int(arch.get("scalar_hidden", 32)),
        fusion_hidden=int(arch.get("fusion_hidden", 96)),
        dropout=float(arch.get("dropout", 0.05)),
        use_group_norm=bool(arch.get("use_group_norm", True)),
    )


def candidate_loss(
    *,
    logit: torch.Tensor,
    error_scaled: torch.Tensor,
    target_within10: torch.Tensor,
    target_error_scaled: torch.Tensor,
    error_loss_weight: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logit, target_within10)
    error_loss = F.smooth_l1_loss(error_scaled, target_error_scaled)
    return bce + float(error_loss_weight) * error_loss


def make_loader(
    dataset: CandidateWindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


@torch.no_grad()
def evaluate_model(
    *,
    model: CandidateLikelihoodNet,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    error_scale_ft: float,
    error_loss_weight: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    rows: list[pd.DataFrame] = []
    losses: list[float] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        scalar = batch["scalar"].to(device, non_blocking=True)
        target = batch["within_10ft"].to(device, non_blocking=True)
        target_error = batch["abs_error_scaled"].to(device, non_blocking=True)
        logit, pred_error_scaled = model(image, scalar)
        loss = candidate_loss(
            logit=logit,
            error_scaled=pred_error_scaled,
            target_within10=target,
            target_error_scaled=target_error,
            error_loss_weight=error_loss_weight,
        )
        losses.append(float(loss.detach().cpu()))
        rows.append(
            pd.DataFrame(
                {
                    "sample_id": batch["sample_id"].cpu().numpy().astype(np.int64),
                    "within_10ft": batch["within_10ft"].cpu().numpy().astype(np.int8),
                    "abs_error": batch["abs_error"].cpu().numpy().astype(np.float32),
                    "pred_logit": logit.detach().cpu().numpy().astype(np.float32),
                    "pred_within10_prob": torch.sigmoid(logit).detach().cpu().numpy().astype(
                        np.float32
                    ),
                    "pred_abs_error": (
                        pred_error_scaled.detach().cpu().numpy().astype(np.float32)
                        * float(error_scale_ft)
                    ),
                }
            )
        )
    pred = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    y = pred["within_10ft"].to_numpy(np.int8) if len(pred) else np.array([], dtype=np.int8)
    prob = (
        pred["pred_within10_prob"].to_numpy(np.float32)
        if len(pred)
        else np.array([], dtype=np.float32)
    )
    metrics = {
        "loss": float(np.mean(losses)) if losses else None,
        "rows": int(len(pred)),
        "candidate_auc": safe_auc(y, prob) if len(pred) else None,
        "candidate_logloss": safe_logloss(y, prob) if len(pred) else None,
        "candidate_brier": float(brier_score_loss(y, np.clip(prob, 1e-6, 1.0 - 1e-6)))
        if len(pred) and len(np.unique(y)) >= 2
        else None,
        "observed_within10": float(np.mean(y)) if len(pred) else None,
        "pred_within10_mean": float(np.mean(prob)) if len(pred) else None,
        "pred_abs_error_mae": mae(
            pred["abs_error"].to_numpy(np.float32),
            pred["pred_abs_error"].to_numpy(np.float32),
        )
        if len(pred)
        else None,
    }
    return metrics, pred


def train_variant(
    *,
    variant: str,
    candidate_index: pd.DataFrame,
    arrays_by_well: dict[str, WellArrays],
    scalar_columns: list[str],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, Path]:
    seed = int(get_nested(config, "reproducibility.seed") or 42)
    training = get_nested(config, "model.training") or {}
    batch_size = int(get_nested(config, "runtime.batch_size") or 128)
    epochs = int(training.get("epochs", 3))
    error_scale_ft = float(training.get("error_scale_ft", 40.0))
    error_loss_weight = float(training.get("error_loss_weight", 0.35))

    train_base = candidate_index.loc[candidate_index["split"].eq("train")].copy()
    valid_base = candidate_index.loc[candidate_index["split"].eq("valid")].copy()
    medians = fit_scalar_medians(train_base, scalar_columns)
    train_frame = apply_scalar_medians(train_base, scalar_columns, medians)
    valid_frame = apply_scalar_medians(valid_base, scalar_columns, medians)

    train_dataset = CandidateWindowDataset(
        candidate_index=train_frame,
        arrays_by_well=arrays_by_well,
        scalar_columns=scalar_columns,
        config=config,
        variant=variant,
    )
    valid_dataset = CandidateWindowDataset(
        candidate_index=valid_frame,
        arrays_by_well=arrays_by_well,
        scalar_columns=scalar_columns,
        config=config,
        variant=variant,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool(training.get("dataloader_shuffle", True)),
        seed=stable_int(EXPERIMENT_NAME, variant, "train-loader", str(seed), modulo=2**31 - 1),
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=stable_int(EXPERIMENT_NAME, variant, "valid-loader", str(seed), modulo=2**31 - 1),
    )
    model = make_model(config, scalar_dim=len(scalar_columns)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    grad_clip = float(training.get("gradient_clip_norm", 1.0))

    history_rows: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_auc = -np.inf
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            scalar = batch["scalar"].to(device, non_blocking=True)
            target = batch["within_10ft"].to(device, non_blocking=True)
            target_error = batch["abs_error_scaled"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logit, pred_error_scaled = model(image, scalar)
            loss = candidate_loss(
                logit=logit,
                error_scaled=pred_error_scaled,
                target_within10=target,
                target_error_scaled=target_error,
                error_loss_weight=error_loss_weight,
            )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        valid_metrics, _ = evaluate_model(
            model=model,
            loader=valid_loader,
            device=device,
            error_scale_ft=error_scale_ft,
            error_loss_weight=error_loss_weight,
        )
        valid_auc = valid_metrics["candidate_auc"]
        auc_for_selection = float(valid_auc) if valid_auc is not None else -np.inf
        if auc_for_selection > best_auc:
            best_auc = auc_for_selection
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history_rows.append(
            {
                "variant": variant,
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else None,
                "valid_loss": valid_metrics["loss"],
                "valid_candidate_auc": valid_metrics["candidate_auc"],
                "valid_candidate_logloss": valid_metrics["candidate_logloss"],
                "valid_candidate_brier": valid_metrics["candidate_brier"],
                "valid_observed_within10": valid_metrics["observed_within10"],
                "elapsed_sec": float(time.time() - start),
            }
        )
        print(
            f"{variant} epoch {epoch}/{epochs}: "
            f"train_loss={history_rows[-1]['train_loss']:.5f} "
            f"valid_auc={valid_metrics['candidate_auc']}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics, valid_pred = evaluate_model(
        model=model,
        loader=valid_loader,
        device=device,
        error_scale_ft=error_scale_ft,
        error_loss_weight=error_loss_weight,
    )
    final_metrics.update(
        {
            "variant": variant,
            "train_candidate_rows": int(len(train_dataset)),
            "valid_candidate_rows": int(len(valid_dataset)),
            "epochs": int(epochs),
            "elapsed_sec": float(time.time() - start),
            "best_valid_candidate_auc": float(best_auc) if np.isfinite(best_auc) else None,
            "scalar_medians": medians,
        }
    )
    model_path = output_dir / f"{OUTPUT_PREFIX}_{variant}_candidate_likelihood_model.pt"
    torch.save(
        {
            "experiment": EXPERIMENT_NAME,
            "variant": variant,
            "state_dict": model.state_dict(),
            "scalar_columns": scalar_columns,
            "scalar_medians": medians,
            "config_model": get_nested(config, "model"),
            "metrics": to_jsonable(final_metrics),
        },
        model_path,
    )
    valid_pred["variant"] = variant
    return final_metrics, valid_pred, pd.DataFrame(history_rows), model_path


# %% [markdown]
# ## 7. Diagnostic metrics

# %%
def with_variant_point_score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    values = []
    for row in out.itertuples(index=False):
        if row.variant == "shuffled_gr":
            values.append(row.point_gr_score_shuffled)
        elif row.variant == "no_gr":
            values.append(row.point_gr_score_no_gr)
        else:
            values.append(row.point_gr_score_real)
    out["point_gr_score"] = np.asarray(values, dtype=np.float32)
    return out


def candidate_likelihood_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    score_specs = [
        ("learned_prob", "pred_within10_prob", "probability"),
        ("learned_negative_error", "pred_abs_error", "negative_error"),
        ("point_gr_likelihood", "point_gr_score", "score"),
        ("exp099_multiobs_score", "candidate_multiobs_score", "score"),
        ("exp099_negative_multiobs_mae", "candidate_multiobs_mae", "negative_error"),
        ("exp111_learned_probability", "exp111_pred_within10_prob", "probability"),
        ("exp111_negative_error", "exp111_pred_abs_error", "negative_error"),
    ]
    for variant, variant_frame in oof.groupby("variant", sort=True):
        y = variant_frame["within_10ft"].to_numpy(np.int8)
        for name, column, role in score_specs:
            if column not in variant_frame.columns:
                continue
            raw = pd.to_numeric(variant_frame[column], errors="coerce").to_numpy(np.float32)
            if role == "negative_error":
                score = -raw
            else:
                score = raw
            prob = raw if role == "probability" else None
            finite = np.isfinite(score)
            if finite.sum() == 0:
                continue
            rows.append(
                {
                    "variant": variant,
                    "score": name,
                    "mode": "candidate_likelihood",
                    "candidate_rows": int(finite.sum()),
                    "observed_within10": float(np.mean(y[finite])),
                    "auc": safe_auc(y, score),
                    "logloss": safe_logloss(y, prob) if prob is not None else None,
                    "brier": float(
                        brier_score_loss(y[finite], np.clip(prob[finite], 1e-6, 1.0 - 1e-6))
                    )
                    if prob is not None and len(np.unique(y[finite])) >= 2
                    else None,
                    "mean_abs_error": float(variant_frame.loc[finite, "abs_error"].mean()),
                    "score_mean": float(np.nanmean(score)),
                }
            )
    return pd.DataFrame(rows)


def selected_rows_for_policy(group: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "likpf_mean_single":
        matches = group.index[group["candidate_name"].eq("likpf_mean")].to_numpy()
        return group.loc[matches[0]] if len(matches) else group.iloc[0]
    if policy == "learned_prob_top1":
        return group.loc[group["pred_within10_prob"].astype(float).idxmax()]
    if policy == "learned_error_top1":
        return group.loc[group["pred_abs_error"].astype(float).idxmin()]
    if policy == "point_gr_top1":
        return group.loc[group["point_gr_score"].astype(float).idxmax()]
    if policy == "exp099_multiobs_score_top1":
        return group.loc[group["candidate_multiobs_score"].astype(float).idxmax()]
    if policy == "exp111_prob_top1" and group["exp111_pred_within10_prob"].notna().any():
        return group.loc[group["exp111_pred_within10_prob"].astype(float).idxmax()]
    raise ValueError(f"Unsupported policy or missing values: {policy}")


def collect_policy_predictions(oof: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, variant_frame in oof.groupby("variant", sort=True):
        for policy in policies:
            for _id, group in variant_frame.groupby("id", sort=False):
                try:
                    selected = selected_rows_for_policy(group, policy)
                except ValueError:
                    continue
                rows.append(
                    {
                        "variant": variant,
                        "policy": policy,
                        "id": _id,
                        "well": selected["well"],
                        "row_index": int(selected["row_index"]),
                        "candidate_name": selected["candidate_name"],
                        "pred_tvt": float(selected["candidate_tvt"]),
                        "true_tvt": float(selected["true_tvt"]),
                        "abs_error": float(abs(selected["candidate_tvt"] - selected["true_tvt"])),
                    }
                )
    return pd.DataFrame(rows)


def topk_metrics(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = [
        "likpf_mean_single",
        "learned_prob_top1",
        "learned_error_top1",
        "point_gr_top1",
        "exp099_multiobs_score_top1",
        "exp111_prob_top1",
    ]
    selected = collect_policy_predictions(oof, policies)
    metric_rows: list[dict[str, Any]] = []
    if not selected.empty:
        for (variant, policy), group in selected.groupby(["variant", "policy"], sort=True):
            error = group["abs_error"].to_numpy(np.float32)
            metric_rows.append(
                {
                    "variant": variant,
                    "policy": policy,
                    "mode": "top1",
                    "rows": int(len(group)),
                    "rmse_tvt": float(math.sqrt(float(np.mean(np.square(error))))),
                    "mae_tvt": float(np.mean(error)),
                    "within_5ft": float(np.mean(error <= 5.0)),
                    "within_10ft": float(np.mean(error <= 10.0)),
                }
            )
    score_policies = [
        ("learned_prob", "pred_within10_prob", False),
        ("learned_error", "pred_abs_error", True),
        ("point_gr", "point_gr_score", False),
        ("exp099_multiobs_score", "candidate_multiobs_score", False),
        ("exp111_prob", "exp111_pred_within10_prob", False),
    ]
    for variant, variant_frame in oof.groupby("variant", sort=True):
        for policy, column, ascending in score_policies:
            if column not in variant_frame.columns or variant_frame[column].notna().sum() == 0:
                continue
            for top_k in [1, 2, 3, 5]:
                best_errors: list[float] = []
                for _id, group in variant_frame.groupby("id", sort=False):
                    ranked = group.dropna(subset=[column]).sort_values(column, ascending=ascending)
                    if ranked.empty:
                        continue
                    best_errors.append(float(ranked.head(top_k)["abs_error"].min()))
                if not best_errors:
                    continue
                values = np.asarray(best_errors, dtype=np.float32)
                metric_rows.append(
                    {
                        "variant": variant,
                        "policy": f"{policy}_top{top_k}_coverage",
                        "mode": "topk_coverage",
                        "rows": int(len(values)),
                        "top_k": int(top_k),
                        "oracle_topk_rmse": float(math.sqrt(float(np.mean(np.square(values))))),
                        "oracle_topk_mae": float(np.mean(values)),
                        "topk_within_5ft": float(np.mean(values <= 5.0)),
                        "topk_within_10ft": float(np.mean(values <= 10.0)),
                    }
                )
    return pd.DataFrame(metric_rows), selected


def weighted_metrics(oof: pd.DataFrame, temperatures: list[float]) -> pd.DataFrame:
    score_specs = [
        ("learned_logit_weight", "pred_logit", False),
        ("learned_negative_error_weight", "pred_abs_error", True),
        ("point_gr_weight", "point_gr_score", False),
        ("exp099_multiobs_score_weight", "candidate_multiobs_score", False),
        ("exp111_prob_weight", "exp111_pred_within10_prob", False),
    ]
    rows: list[dict[str, Any]] = []
    for variant, variant_frame in oof.groupby("variant", sort=True):
        for policy, column, ascending in score_specs:
            if column not in variant_frame.columns or variant_frame[column].notna().sum() == 0:
                continue
            for temperature in temperatures:
                pred_rows: list[dict[str, Any]] = []
                for _id, group in variant_frame.groupby("id", sort=False):
                    work = group.dropna(subset=[column]).copy()
                    if work.empty:
                        continue
                    scores = work[column].to_numpy(np.float32)
                    if ascending:
                        scores = -scores
                    weights = softmax_np(scores, float(temperature))
                    candidate_tvt = work["candidate_tvt"].to_numpy(np.float64)
                    pred_tvt = float(np.sum(weights * candidate_tvt))
                    ess = float(1.0 / np.sum(np.square(weights)))
                    pred_rows.append(
                        {
                            "id": _id,
                            "well": work["well"].iloc[0],
                            "row_index": int(work["row_index"].iloc[0]),
                            "pred_tvt": pred_tvt,
                            "true_tvt": float(work["true_tvt"].iloc[0]),
                            "effective_sample_size": ess,
                        }
                    )
                if not pred_rows:
                    continue
                pred = pd.DataFrame(pred_rows)
                error = np.abs(pred["pred_tvt"].to_numpy() - pred["true_tvt"].to_numpy())
                rows.append(
                    {
                        "variant": variant,
                        "policy": policy,
                        "temperature": float(temperature),
                        "rows": int(len(pred)),
                        "rmse_tvt": float(math.sqrt(float(np.mean(np.square(error))))),
                        "mae_tvt": float(np.mean(error)),
                        "within_10ft": float(np.mean(error <= 10.0)),
                        "effective_sample_size_mean": float(pred["effective_sample_size"].mean()),
                        "effective_sample_size_p10": float(
                            pred["effective_sample_size"].quantile(0.10)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def continuity_metrics(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if selected.empty:
        return pd.DataFrame()
    for (variant, policy), group in selected.groupby(["variant", "policy"], sort=True):
        ordered = group.sort_values(["well", "row_index"]).copy()
        same_well = ordered["well"].eq(ordered["well"].shift())
        pred_step = ordered["pred_tvt"].diff().where(same_well)
        true_step = ordered["true_tvt"].diff().where(same_well)
        cand_switch = ordered["candidate_name"].ne(ordered["candidate_name"].shift()).where(
            same_well
        )
        rows.append(
            {
                "variant": variant,
                "policy": policy,
                "rows": int(len(ordered)),
                "pred_step_abs_mean": float(pred_step.abs().mean()),
                "true_step_abs_mean": float(true_step.abs().mean()),
                "step_error_abs_mean": float((pred_step - true_step).abs().mean()),
                "candidate_switch_rate": float(cand_switch.dropna().astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def by_well_metrics(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if selected.empty:
        return pd.DataFrame()
    for (variant, policy, well), group in selected.groupby(["variant", "policy", "well"], sort=True):
        error = np.abs(group["pred_tvt"].to_numpy() - group["true_tvt"].to_numpy())
        rows.append(
            {
                "variant": variant,
                "policy": policy,
                "well": well,
                "rows": int(len(group)),
                "rmse_tvt": float(math.sqrt(float(np.mean(np.square(error))))),
                "mae_tvt": float(np.mean(error)),
                "within_10ft": float(np.mean(error <= 10.0)),
            }
        )
    return pd.DataFrame(rows)


def bucket_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = oof.copy()
    work["tail_rank_bucket"] = pd.cut(
        work["row_index"].astype(float),
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )
    for (variant, bucket), group in work.groupby(["variant", "tail_rank_bucket"], observed=True):
        y = group["within_10ft"].to_numpy(np.int8)
        p = group["pred_within10_prob"].to_numpy(np.float32)
        rows.append(
            {
                "variant": variant,
                "bucket_family": "row_index",
                "bucket": str(bucket),
                "candidate_rows": int(len(group)),
                "observed_within10": float(np.mean(y)),
                "pred_within10_mean": float(np.mean(p)),
                "auc": safe_auc(y, p),
                "mean_abs_error": float(group["abs_error"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_decision(candidate_metrics: pd.DataFrame, topk: pd.DataFrame) -> dict[str, Any]:
    real = candidate_metrics[
        candidate_metrics["variant"].eq("real_gr")
        & candidate_metrics["score"].eq("learned_prob")
    ].head(1)
    shuffled = candidate_metrics[
        candidate_metrics["variant"].eq("shuffled_gr")
        & candidate_metrics["score"].eq("learned_prob")
    ].head(1)
    no_gr = candidate_metrics[
        candidate_metrics["variant"].eq("no_gr")
        & candidate_metrics["score"].eq("learned_prob")
    ].head(1)
    multiobs = candidate_metrics[
        candidate_metrics["variant"].eq("real_gr")
        & candidate_metrics["score"].eq("exp099_multiobs_score")
    ].head(1)
    learned_top1 = topk[
        topk["variant"].eq("real_gr") & topk["policy"].eq("learned_prob_top1")
    ].head(1)
    likpf_top1 = topk[
        topk["variant"].eq("real_gr") & topk["policy"].eq("likpf_mean_single")
    ].head(1)
    decision = "not_run"
    delta_auc_multiobs = None
    real_minus_shuffled_auc = None
    real_minus_no_gr_auc = None
    top1_delta_vs_likpf = None
    if not real.empty:
        real_auc = real.iloc[0]["auc"]
        if not multiobs.empty and pd.notna(real_auc) and pd.notna(multiobs.iloc[0]["auc"]):
            delta_auc_multiobs = float(real_auc - multiobs.iloc[0]["auc"])
        if not shuffled.empty and pd.notna(real_auc) and pd.notna(shuffled.iloc[0]["auc"]):
            real_minus_shuffled_auc = float(real_auc - shuffled.iloc[0]["auc"])
        if not no_gr.empty and pd.notna(real_auc) and pd.notna(no_gr.iloc[0]["auc"]):
            real_minus_no_gr_auc = float(real_auc - no_gr.iloc[0]["auc"])
        if not learned_top1.empty and not likpf_top1.empty:
            top1_delta_vs_likpf = float(
                learned_top1.iloc[0]["rmse_tvt"] - likpf_top1.iloc[0]["rmse_tvt"]
            )
        if (
            delta_auc_multiobs is not None
            and delta_auc_multiobs > 0.01
            and real_minus_shuffled_auc is not None
            and real_minus_shuffled_auc > 0.01
            and real_minus_no_gr_auc is not None
            and real_minus_no_gr_auc > 0.01
        ):
            decision = "cnn_likelihood_supported_for_rawtest_parity_or_feature_followup"
        elif real_minus_shuffled_auc is not None and real_minus_shuffled_auc > 0.0:
            decision = "weak_real_gr_signal_needs_guarded_followup"
        else:
            decision = "cnn_likelihood_not_supported"
    return {
        "recommendation": decision,
        "delta_auc_vs_exp099_multiobs_score": delta_auc_multiobs,
        "real_minus_shuffled_auc": real_minus_shuffled_auc,
        "real_minus_no_gr_auc": real_minus_no_gr_auc,
        "learned_top1_delta_rmse_vs_likpf": top1_delta_vs_likpf,
        "real_learned_metric": to_jsonable(real.iloc[0].to_dict()) if not real.empty else None,
        "real_multiobs_metric": to_jsonable(multiobs.iloc[0].to_dict())
        if not multiobs.empty
        else None,
    }


# %% [markdown]
# ## 8. Setup and input checks

# %%
paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()
seed = int(get_nested(config, "reproducibility.seed") or 42)
set_reproducibility(seed)
device = require_cuda_device(config)
candidates = candidate_specs_from_config(config)
training_config = get_nested(config, "model.training") or {}
active_variants = list(get_nested(config, "model.active_variants") or [])
temperatures = [float(value) for value in get_nested(config, "model.weighting.temperatures") or [0.2]]

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Device:", device)
print("Torch:", torch.__version__)
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
print("Train dir:", paths.train_data_dir)
print("Artifacts dir:", paths.artifacts_dir)
print("Candidates:", [asdict(spec) for spec in candidates])
print("Active variants:", active_variants)
print("Training config:", json.dumps(to_jsonable(training_config), indent=2, sort_keys=True))

# %%
frame, exp099_meta = read_exp099_cache(config=config, candidates=candidates)
exp111_oof, exp111_meta = load_exp111_oof(config)

n_folds = int(get_nested(config, "validation.n_folds") or 5)
run_folds = int(get_nested(config, "validation.run_folds") or 1)
folds = make_fold_indices(frame, n_folds=n_folds, run_folds=run_folds)
if len(folds) != 1:
    raise ValueError("exp197 implementation currently expects one active fold.")
fold, train_idx, valid_idx = folds[0]
train_rows = sample_indices(
    train_idx,
    max_rows=training_config.get("max_train_rows_per_fold"),
    seed=stable_int(EXPERIMENT_NAME, "train-row-sample", str(seed), str(fold), modulo=2**31 - 1),
)
valid_rows = sample_indices(
    valid_idx,
    max_rows=training_config.get("max_valid_rows_per_fold"),
    seed=stable_int(EXPERIMENT_NAME, "valid-row-sample", str(seed), str(fold), modulo=2**31 - 1),
)
needed_wells = sorted(set(frame.iloc[np.concatenate([train_rows, valid_rows])]["well"].astype(str)))
arrays_by_well = load_well_arrays(
    needed_wells,
    paths.train_data_dir,
    seed=seed,
    min_prefix_observed_rows=int(training_config.get("min_prefix_observed_rows", 16)),
)
if not arrays_by_well:
    raise RuntimeError("No usable raw train wells were loaded.")

print("exp099 source:", json.dumps(to_jsonable(exp099_meta), indent=2, sort_keys=True))
print("exp111 source:", json.dumps(to_jsonable(exp111_meta), indent=2, sort_keys=True))
print(
    "Fold sample:",
    json.dumps(
        {
            "fold": fold,
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "needed_wells": len(needed_wells),
            "loaded_wells": len(arrays_by_well),
        },
        indent=2,
    ),
)

# %%
candidate_index = pd.concat(
    [
        build_candidate_index(
            frame=frame,
            fold=fold,
            split="train",
            row_indices=train_rows,
            candidates=candidates,
            arrays_by_well=arrays_by_well,
            config=config,
        ),
        build_candidate_index(
            frame=frame,
            fold=fold,
            split="valid",
            row_indices=valid_rows,
            candidates=candidates,
            arrays_by_well=arrays_by_well,
            config=config,
        ),
    ],
    ignore_index=True,
)
candidate_index = attach_exp111_oof(candidate_index, exp111_oof)
candidate_index.insert(0, "sample_id", np.arange(len(candidate_index), dtype=np.int64))
scalar_columns = scalar_feature_columns(candidate_index)

candidate_index_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_candidate_index.csv.gz"
gzip_csv(candidate_index, candidate_index_path)
sample_overview = (
    candidate_index.groupby(["split", "fold"])
    .agg(
        candidate_rows=("sample_id", "count"),
        ids=("id", "nunique"),
        wells=("well", "nunique"),
        observed_within10=("within_10ft", "mean"),
        exp111_coverage=("exp111_pred_within10_prob", lambda s: float(s.notna().mean())),
    )
    .reset_index()
)
display(sample_overview)
display(candidate_index.head())
print("Scalar feature columns:", scalar_columns)

# %% [markdown]
# ## 9. Train CNN likelihood variants

# %%
metrics_rows: list[dict[str, Any]] = []
prediction_frames: list[pd.DataFrame] = []
history_frames: list[pd.DataFrame] = []
model_manifest: dict[str, Any] = {
    "experiment": EXPERIMENT_NAME,
    "created_at": datetime.now(UTC).isoformat(),
    "device": str(device),
    "torch_version": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "models": {},
}

for variant in active_variants:
    if variant not in {"real_gr", "shuffled_gr", "no_gr"}:
        raise ValueError(f"Unexpected variant: {variant}")
    print(f"=== Training variant: {variant} ===")
    metrics, valid_pred, history, model_path = train_variant(
        variant=variant,
        candidate_index=candidate_index,
        arrays_by_well=arrays_by_well,
        scalar_columns=scalar_columns,
        config=config,
        device=device,
        output_dir=paths.artifacts_dir,
    )
    metrics_rows.append(metrics)
    prediction_frames.append(valid_pred)
    history_frames.append(history)
    model_manifest["models"][variant] = {
        "path": str(model_path),
        "sha256": sha256_path(model_path),
        "bytes": model_path.stat().st_size,
        "metrics": to_jsonable(metrics),
    }
    print(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True))

# %% [markdown]
# ## 10. Metrics, SHA, and generated artifacts

# %%
variant_metrics = pd.DataFrame(metrics_rows)
predictions = pd.concat(prediction_frames, ignore_index=True)
history_df = pd.concat(history_frames, ignore_index=True)

meta_columns = [
    "sample_id",
    "fold",
    "split",
    "id",
    "well",
    "row_index",
    "candidate_index",
    "candidate_name",
    "candidate_tvt",
    "true_tvt",
    "last_known_tvt",
    "target",
    "candidate_minus_last",
    "candidate_abs_minus_likpf",
    "candidate_abs_minus_row_mean",
    "candidate_z_within_row",
    "row_candidate_std",
    "row_candidate_range",
    "candidate_multiobs_score",
    "candidate_multiobs_mae",
    "candidate_multiobs_ncc",
    "point_gr_score_real",
    "point_gr_score_shuffled",
    "point_gr_score_no_gr",
    "exp111_pred_within10_prob",
    "exp111_pred_abs_error",
]
oof = predictions.merge(
    candidate_index[meta_columns],
    on="sample_id",
    how="left",
    suffixes=("", "_candidate"),
)
oof = with_variant_point_score(oof)

candidate_metrics = candidate_likelihood_metrics(oof)
topk_df, selected_df = topk_metrics(oof)
weighted_df = weighted_metrics(oof, temperatures)
continuity_df = continuity_metrics(selected_df)
by_well_df = by_well_metrics(selected_df)
bucket_df = bucket_metrics(oof)
decision = summarize_decision(candidate_metrics, topk_df)

oof_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_oof_candidate_likelihood.csv.gz"
candidate_metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
topk_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_topk_metrics.csv"
weighted_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_weighted_metrics.csv"
continuity_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_continuity_metrics.csv"
by_well_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv"
bucket_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
history_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_training_history.csv"
schema_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_model_manifest.json"
summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"

gzip_csv(oof, oof_path)
candidate_metrics.to_csv(candidate_metrics_path, index=False)
topk_df.to_csv(topk_path, index=False)
weighted_df.to_csv(weighted_path, index=False)
continuity_df.to_csv(continuity_path, index=False)
by_well_df.to_csv(by_well_path, index=False)
bucket_df.to_csv(bucket_path, index=False)
history_df.to_csv(history_path, index=False)

schema_rows = [
    {"feature_type": "image_channel", "feature_index": index, "feature": channel, "description": desc}
    for index, (channel, desc) in enumerate(IMAGE_CHANNEL_SCHEMA)
]
schema_rows.extend(
    {
        "feature_type": "scalar_context",
        "feature_index": index,
        "feature": feature,
        "description": "Candidate geometry or target-free row context scalar.",
    }
    for index, feature in enumerate(scalar_columns)
)
pd.DataFrame(schema_rows).to_csv(schema_path, index=False)

manifest_path.write_text(json.dumps(to_jsonable(model_manifest), indent=2, sort_keys=True) + "\n")

artifact_sha = {
    "candidate_index_csv_gz_sha256": sha256_path(candidate_index_path),
    "candidate_index_csv_decompressed_sha256": sha256_path(
        candidate_index_path,
        decompressed=True,
    ),
    "oof_candidate_likelihood_csv_gz_sha256": sha256_path(oof_path),
    "oof_candidate_likelihood_csv_decompressed_sha256": sha256_path(
        oof_path,
        decompressed=True,
    ),
    "candidate_metrics_csv_sha256": sha256_path(candidate_metrics_path),
    "topk_metrics_csv_sha256": sha256_path(topk_path),
    "weighted_metrics_csv_sha256": sha256_path(weighted_path),
    "continuity_metrics_csv_sha256": sha256_path(continuity_path),
    "by_well_csv_sha256": sha256_path(by_well_path),
    "bucket_metrics_csv_sha256": sha256_path(bucket_path),
    "training_history_csv_sha256": sha256_path(history_path),
    "feature_schema_csv_sha256": sha256_path(schema_path),
    "model_manifest_json_sha256": sha256_path(manifest_path),
    "oof_probability_content_sha256": prediction_sha256(oof, value_col="pred_within10_prob"),
    "oof_expected_error_content_sha256": prediction_sha256(oof, value_col="pred_abs_error"),
}

summary = {
    "experiment": EXPERIMENT_NAME,
    "status": "completed_train_side_gpu_probe",
    "created_at": datetime.now(UTC).isoformat(),
    "seed": seed,
    "fold": int(fold),
    "device": str(device),
    "torch_version": torch.__version__,
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "exp099_source": exp099_meta,
    "exp111_source": exp111_meta,
    "candidate_index": {
        "candidate_rows": int(len(candidate_index)),
        "ids": int(candidate_index["id"].nunique()),
        "wells": int(candidate_index["well"].nunique()),
        "sample_overview": sample_overview.to_dict(orient="records"),
    },
    "active_variants": active_variants,
    "candidate_metrics": candidate_metrics.to_dict(orient="records"),
    "topk_metrics": topk_df.to_dict(orient="records"),
    "weighted_metrics": weighted_df.to_dict(orient="records"),
    "decision": decision,
    "artifact_sha": artifact_sha,
    "model_manifest": model_manifest,
    "reproducibility": {
        "deterministic_anchor": False,
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "num_workers": int(get_nested(config, "runtime.num_workers") or 0),
        "cpu_fallback": False,
        "submission_created": False,
    },
}
summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
artifact_sha["summary_json_sha256"] = sha256_path(summary_path)
summary["artifact_sha"] = artifact_sha
summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")

metrics_json = {
    "experiment": EXPERIMENT_NAME,
    "status": "completed_train_side_gpu_probe",
    "cv": None,
    "public_lb": None,
    "private_lb": None,
    "metric": "candidate_auc",
    "summary": {
        "decision": decision,
        "candidate_metrics": candidate_metrics.to_dict(orient="records"),
        "topk_metrics": topk_df.to_dict(orient="records"),
        "weighted_metrics": weighted_df.to_dict(orient="records"),
        "artifact_sha": artifact_sha,
        "cuda_device_name": summary["cuda_device_name"],
    },
    "notes": "Train-side GPU diagnostic only; no inference and no submission.",
}
paths.metrics_path.write_text(json.dumps(to_jsonable(metrics_json), indent=2, sort_keys=True) + "\n")

display(candidate_metrics)
display(topk_df)
display(weighted_df)
print("Decision:", json.dumps(to_jsonable(decision), indent=2, sort_keys=True))
print("Saved summary:", summary_path)
print(json.dumps(to_jsonable(artifact_sha), indent=2, sort_keys=True))

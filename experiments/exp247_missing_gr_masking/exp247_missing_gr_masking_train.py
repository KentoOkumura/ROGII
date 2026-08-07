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
# # exp247 missing GR masking — train
#
# This notebook performs one exact-HMM ablation. The saved exp221 interpolation
# control is fixed. In the new `mask_only` path, a row whose raw horizontal GR
# is missing contributes zero GR log-likelihood to every TVT state. The unchanged
# exp148 OOF LightGBM unary and exact-HMM transition still apply on that row.
# No model, control HMM, raw-test prediction, or submission is generated.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and input resolution
# 3. Raw GR missing-run annotation and inventory
# 4. Fixed exp221 control and OOF source readout
# 5. Mask-only exact-HMM orchestration
# 6. Paired metrics and divergence diagnostics
# 7. Synthetic emission contract
# 8. Setup and cost guard
# 9. Raw train/test distribution and fixed-control readout
# 10. Full mask-only generation
# 11. Metrics, SHA, and generated files

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from exact_hmm_smoother import (
    NUMBA_AVAILABLE,
    build_gr_emission_loglik,
    get_num_threads,
    list_well_ids,
    load_well,
    prepare_lgb_emission_variants,
    run_hmm2,
    set_num_threads,
)
from IPython.display import display

EXPERIMENT_NAME = "exp247_missing_gr_masking"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        lineterminator="\n",
    )


# %% [markdown]
# ## 2. Configuration and input resolution


# %%
def find_repo_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


ROOT = find_repo_root()


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if value.get("experiment", {}).get("name") == EXPERIMENT_NAME:
            return path
    raise FileNotFoundError(f"Could not resolve config.yaml for {EXPERIMENT_NAME}")


CONFIG_PATH = find_config_path()
config: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text()) or {}


def nested(dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def require_authoritative_runtime() -> None:
    if is_kaggle_runtime():
        return
    if os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") != "1":
        raise RuntimeError(
            "Kaggle Notebook is authoritative. Local execution requires "
            "EXPERIMENT_ALLOW_LOCAL=1 and is debug-only."
        )


def output_dir() -> Path:
    if is_kaggle_runtime():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        path = ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    if is_kaggle_runtime():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return ROOT / "experiments" / EXPERIMENT_NAME / "metrics.json"


def resolve_split_dir(split: str) -> Path:
    configured = Path(str(nested(f"data.{split}_dir", f"data/raw/{split}")))
    local = configured if configured.is_absolute() else ROOT / configured
    suffix = str(nested("data.horizontal_suffix", "__horizontal_well.csv"))
    if local.is_dir() and any(local.glob(f"*{suffix}")):
        return local
    if KAGGLE_INPUT_ROOT.exists():
        matches = sorted(KAGGLE_INPUT_ROOT.rglob(f"{split}/*{suffix}"))
        if matches:
            return matches[0].parent
    raise FileNotFoundError(f"Could not resolve {split} directory containing *{suffix}")


def resolve_existing_file(candidates: list[str]) -> Path:
    checked: list[str] = []
    for raw in candidates:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        checked.append(str(path))
        if path.exists() and path.stat().st_size > 0:
            return path
    if KAGGLE_INPUT_ROOT.exists():
        for raw in candidates:
            basename = Path(raw).name
            for path in sorted(KAGGLE_INPUT_ROOT.rglob(basename)):
                checked.append(str(path))
                if path.stat().st_size > 0:
                    return path
    raise FileNotFoundError("No configured input exists: " + json.dumps(checked, indent=2))


# %% [markdown]
# ## 3. Raw GR missing-run annotation and inventory


# %%
def contiguous_missing_run_lengths(missing: np.ndarray) -> np.ndarray:
    missing = np.asarray(missing, dtype=bool)
    lengths = np.zeros(len(missing), dtype=np.int32)
    start: int | None = None
    for index, value in enumerate(missing):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(missing) - 1):
            stop = index if not value else index + 1
            lengths[start:stop] = stop - start
            start = None
    return lengths


def rows_since_missing_run_end(missing: np.ndarray) -> np.ndarray:
    missing = np.asarray(missing, dtype=bool)
    result = np.full(len(missing), -1, dtype=np.int32)
    seen_missing = False
    offset = 0
    for index, value in enumerate(missing):
        if value:
            seen_missing = True
            offset = 0
            result[index] = 0
        elif seen_missing:
            offset += 1
            result[index] = offset
    return result


def bucket_missing_runs(missing: np.ndarray, run_length: np.ndarray) -> np.ndarray:
    edges = np.asarray(nested("audit.missing_run_buckets.edges"), dtype=np.int64)
    labels = list(nested("audit.missing_run_buckets.labels"))
    result = np.full(len(missing), "observed", dtype=object)
    values = np.asarray(run_length, dtype=np.int64)
    selected = np.asarray(missing, dtype=bool)
    if selected.any():
        indices = np.digitize(values[selected], edges, right=True) - 1
        indices = np.clip(indices, 0, len(labels) - 1)
        result[selected] = np.asarray(labels, dtype=object)[indices]
    return result


def bucket_post_gap(missing: np.ndarray, post_gap_rows: np.ndarray) -> np.ndarray:
    edges = np.asarray(nested("audit.post_gap_buckets.edges"), dtype=np.int64)
    labels = list(nested("audit.post_gap_buckets.labels"))
    result = np.full(len(missing), "no_prior_missing", dtype=object)
    selected_missing = np.asarray(missing, dtype=bool)
    result[selected_missing] = "missing_run"
    values = np.asarray(post_gap_rows, dtype=np.int64)
    selected = (~selected_missing) & (values > 0)
    if selected.any():
        indices = np.digitize(values[selected], edges, right=True) - 1
        indices = np.clip(indices, 0, len(labels) - 1)
        result[selected] = np.asarray(labels, dtype=object)[indices]
    return result


def missing_run_records(
    split: str,
    well: str,
    missing: np.ndarray,
    eval_mask: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    start: int | None = None
    for index, value in enumerate(missing):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(missing) - 1):
            stop = index if not value else index + 1
            overlap = int(eval_mask[start:stop].sum())
            records.append(
                {
                    "split": split,
                    "well": well,
                    "start_row": start,
                    "end_row": stop - 1,
                    "run_length": stop - start,
                    "evaluation_rows": overlap,
                    "known_prefix_rows": int((~eval_mask[start:stop]).sum()),
                    "overlaps_evaluation": bool(overlap),
                }
            )
            start = None
    return records


def scan_raw_split(
    split: str,
    directory: Path,
    *,
    include_eval_context: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    suffix = str(nested("data.horizontal_suffix", "__horizontal_well.csv"))
    inventory_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    context_frames: list[pd.DataFrame] = []
    sha_rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"*{suffix}")):
        well = path.name.removesuffix(suffix)
        horizontal = pd.read_csv(path, usecols=["GR", "TVT_input"])
        gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
        missing = ~np.isfinite(gr)
        eval_mask = horizontal["TVT_input"].isna().to_numpy(bool)
        run_length = contiguous_missing_run_lengths(missing)
        post_gap = rows_since_missing_run_end(missing)
        runs = missing_run_records(split, well, missing, eval_mask)
        run_rows.extend(runs)
        eval_missing = missing & eval_mask
        inventory_rows.append(
            {
                "split": split,
                "well": well,
                "rows": int(len(horizontal)),
                "evaluation_rows": int(eval_mask.sum()),
                "missing_rows": int(missing.sum()),
                "missing_fraction": float(missing.mean()) if len(missing) else 0.0,
                "evaluation_missing_rows": int(eval_missing.sum()),
                "evaluation_missing_fraction": (
                    float(eval_missing.sum() / eval_mask.sum()) if eval_mask.any() else 0.0
                ),
                "missing_run_count": int(len(runs)),
                "longest_missing_run": int(run_length.max()) if len(run_length) else 0,
                "longest_evaluation_missing_run": (
                    int(run_length[eval_missing].max()) if eval_missing.any() else 0
                ),
            }
        )
        sha_rows.append(
            {
                "split": split,
                "well": well,
                "kind": "horizontal",
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_path(path),
            }
        )
        typewell = directory / f"{well}{nested('data.typewell_suffix', '__typewell.csv')}"
        if typewell.exists():
            sha_rows.append(
                {
                    "split": split,
                    "well": well,
                    "kind": "typewell",
                    "path": str(typewell),
                    "bytes": int(typewell.stat().st_size),
                    "sha256": sha256_path(typewell),
                }
            )
        if include_eval_context and eval_mask.any():
            eval_index = np.flatnonzero(eval_mask)
            context_frames.append(
                pd.DataFrame(
                    {
                        "id": [f"{well}_{int(index)}" for index in eval_index],
                        "well": well,
                        "row_index": eval_index,
                        "raw_gr_missing": missing[eval_index],
                        "missing_run_length": run_length[eval_index],
                        "post_gap_rows": post_gap[eval_index],
                        "missing_run_bucket": bucket_missing_runs(
                            missing[eval_index], run_length[eval_index]
                        ),
                        "post_gap_bucket": bucket_post_gap(
                            missing[eval_index], post_gap[eval_index]
                        ),
                    }
                )
            )
    context = pd.concat(context_frames, ignore_index=True) if context_frames else pd.DataFrame()
    return (
        pd.DataFrame(inventory_rows),
        pd.DataFrame(run_rows),
        context,
        pd.DataFrame(sha_rows),
    )


# %% [markdown]
# ## 4. Fixed exp221 control and OOF source readout


# %%
def load_control_frame(path: Path) -> pd.DataFrame:
    column_map = {
        "control_tvt": str(nested("data.control_prediction_column")),
        "control_std": str(nested("data.control_std_column")),
        "control_finite_parent": str(nested("data.control_finite_column")),
        "control_loglik": str(nested("data.control_loglik_column")),
    }
    required = ["id", "well", "target", "last_known_tvt", "md_since", *column_map.values()]
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(required).difference(header.columns))
    if missing:
        raise ValueError(f"Fixed exp221 control cache is missing columns: {missing}")
    frame = pd.read_csv(path, usecols=required, dtype={"id": str, "well": str})
    if frame["id"].duplicated().any():
        raise ValueError("Fixed exp221 control cache has duplicate ids")
    frame = frame.rename(columns={source: target for target, source in column_map.items()})
    frame["true_tvt"] = pd.to_numeric(frame["last_known_tvt"], errors="coerce") + pd.to_numeric(
        frame["target"], errors="coerce"
    )
    numeric = [
        "true_tvt",
        "control_tvt",
        "control_std",
        "control_finite_parent",
        "control_loglik",
        "md_since",
    ]
    if not np.isfinite(frame[numeric].to_numpy(np.float64)).all():
        raise ValueError("Fixed exp221 control cache contains non-finite required values")
    return frame


def distance_bucket(md_since: pd.Series) -> np.ndarray:
    edges = np.asarray(nested("audit.distance_buckets.edges"), dtype=np.float64)
    labels = np.asarray(nested("audit.distance_buckets.labels"), dtype=object)
    values = pd.to_numeric(md_since, errors="coerce").to_numpy(np.float64)
    indices = np.digitize(values, edges, right=True) - 1
    indices = np.clip(indices, 0, len(labels) - 1)
    return labels[indices]


def metric_values(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(target) & np.isfinite(prediction)
    if not finite.any():
        return {
            "rows": int(len(target)),
            "finite_rows": 0,
            "finite_coverage": 0.0,
            "rmse": None,
            "mae": None,
            "within10": None,
        }
    error = prediction[finite] - target[finite]
    return {
        "rows": int(len(target)),
        "finite_rows": int(finite.sum()),
        "finite_coverage": float(finite.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "within10": float(np.mean(np.abs(error) <= 10.0)),
    }


def single_control_group_metrics(
    frame: pd.DataFrame,
    group_type: str,
    groups: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in pd.unique(groups):
        selected = np.asarray(groups, dtype=object) == group
        values = metric_values(
            frame.loc[selected, "true_tvt"].to_numpy(),
            frame.loc[selected, "control_tvt"].to_numpy(),
        )
        rows.append(
            {
                "group_type": group_type,
                "group": str(group),
                "wells": int(frame.loc[selected, "well"].nunique()),
                **{f"control_{key}": value for key, value in values.items()},
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 5. Mask-only exact-HMM orchestration


# %%
def build_mask_only_rows_for_well(
    well: str,
    train_dir: Path,
    hmm_config: dict[str, Any],
    lgb_variant: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    horizontal, typewell = load_well(well, train_dir)
    known_mask = horizontal["TVT_input"].notna().to_numpy(bool)
    eval_mask = ~known_mask
    if not known_mask.any() or not eval_mask.any():
        return pd.DataFrame(), {
            "well": well,
            "status": "skipped_missing_prefix_or_evaluation",
            "rows": 0,
        }
    eval_index = np.flatnonzero(eval_mask)
    eval_ids = [f"{well}_{int(index)}" for index in eval_index]
    lgb_values = lgb_variant["predictions"].reindex(eval_ids)
    if lgb_values.isna().any():
        examples = lgb_values[lgb_values.isna()].index[:5].tolist()
        raise ValueError(f"{well} missing exp148 OOF values, examples={examples}")
    lgb_tvt = lgb_values.to_numpy(np.float64)
    result = run_hmm2(
        horizontal,
        typewell,
        **hmm_config,
        lgb_tvt=lgb_tvt,
        lgb_sigma=float(lgb_variant["sigma"]),
        lgb_lambda=float(lgb_variant["lambda"]),
        lgb_emission_clip=float(lgb_variant["emission_clip"]),
        mask_missing_gr=True,
    )
    raw_missing_full = ~np.isfinite(
        pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    )
    run_length_full = contiguous_missing_run_lengths(raw_missing_full)
    post_gap_full = rows_since_missing_run_end(raw_missing_full)
    raw_missing = raw_missing_full[eval_index]
    run_length = run_length_full[eval_index]
    post_gap = post_gap_full[eval_index]
    last = horizontal.loc[known_mask].iloc[-1]
    true_tvt = pd.to_numeric(horizontal.loc[eval_index, "TVT"], errors="coerce").to_numpy(
        np.float64
    )
    # exp221 serialized its HMM cache through float32. Preserve that storage
    # contract so no-missing wells can be compared bit-for-bit up to a tiny
    # runtime tolerance instead of reporting float64-vs-float32 noise.
    mask_tvt = np.asarray(result["mean_eval"], dtype=np.float32)
    mask_std = np.asarray(result["std_eval"], dtype=np.float32)
    if len(mask_tvt) != len(eval_index):
        raise ValueError(f"{well} HMM output length mismatch")
    frame = pd.DataFrame(
        {
            "id": eval_ids,
            "well": well,
            "row_index": eval_index,
            "true_tvt_raw": true_tvt,
            "mask_tvt": mask_tvt,
            "mask_std": mask_std,
            "mask_finite": np.isfinite(mask_tvt) & np.isfinite(mask_std),
            "raw_gr_missing": raw_missing,
            "missing_run_length": run_length,
            "post_gap_rows": post_gap,
            "missing_run_bucket": bucket_missing_runs(raw_missing, run_length),
            "post_gap_bucket": bucket_post_gap(raw_missing, post_gap),
            "last_known_tvt_raw": float(last["TVT_input"]),
            "md_since_raw": (
                pd.to_numeric(horizontal.loc[eval_index, "MD"], errors="coerce").to_numpy(
                    np.float64
                )
                - float(last["MD"])
            ),
        }
    )
    finite = bool(frame["mask_finite"].all() and np.isfinite(float(result["loglik"])))
    meta = {
        "well": well,
        "status": "ok" if finite else "non_finite",
        "rows": int(len(frame)),
        "raw_gr_missing_rows": int(raw_missing.sum()),
        "longest_evaluation_missing_run": int(run_length.max()) if len(run_length) else 0,
        "masked_gr_rows_reported": int(result["masked_gr_rows"]),
        "grid_size": int(len(result["grid"])),
        "mask_loglik": float(result["loglik"]),
        "prefix_sigma": float(result["prefix_sigma"]),
        "prefix_ir": float(result["prefix_ir"]),
        "mask_rmse": metric_values(true_tvt, mask_tvt)["rmse"],
        "finite": finite,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if meta["raw_gr_missing_rows"] != meta["masked_gr_rows_reported"]:
        raise AssertionError(f"{well} raw missing count differs from HMM mask count")
    return frame, meta


# %% [markdown]
# ## 6. Paired metrics and divergence diagnostics


# %%
def paired_metric_row(frame: pd.DataFrame, group_type: str, group: str) -> dict[str, Any]:
    target = frame["true_tvt"].to_numpy(np.float64)
    control = metric_values(target, frame["control_tvt"].to_numpy(np.float64))
    masked = metric_values(target, frame["mask_tvt"].to_numpy(np.float64))
    return {
        "group_type": group_type,
        "group": group,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "raw_gr_missing_rows": int(frame["raw_gr_missing"].sum()),
        "control_rmse": control["rmse"],
        "mask_rmse": masked["rmse"],
        "delta_rmse_mask_minus_control": (
            float(masked["rmse"] - control["rmse"])
            if masked["rmse"] is not None and control["rmse"] is not None
            else None
        ),
        "control_mae": control["mae"],
        "mask_mae": masked["mae"],
        "delta_mae_mask_minus_control": (
            float(masked["mae"] - control["mae"])
            if masked["mae"] is not None and control["mae"] is not None
            else None
        ),
        "control_within10": control["within10"],
        "mask_within10": masked["within10"],
        "delta_within10_mask_minus_control": (
            float(masked["within10"] - control["within10"])
            if masked["within10"] is not None and control["within10"] is not None
            else None
        ),
        "control_finite_coverage": control["finite_coverage"],
        "mask_finite_coverage": masked["finite_coverage"],
        "changed_rows": int(frame["prediction_changed"].sum()),
        "mean_abs_mask_minus_control": float(frame["abs_mask_minus_control"].mean()),
        "max_abs_mask_minus_control": float(frame["abs_mask_minus_control"].max()),
    }


def grouped_paired_metrics(
    frame: pd.DataFrame,
    group_type: str,
    groups: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = np.asarray(groups, dtype=object)
    for group in pd.unique(groups):
        selected = groups == group
        if selected.any():
            rows.append(paired_metric_row(frame.loc[selected], group_type, str(group)))
    return pd.DataFrame(rows)


def build_by_well_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=True):
        row = paired_metric_row(group, "well", str(well))
        row.update(
            {
                "well": str(well),
                "missing_fraction": float(group["raw_gr_missing"].mean()),
                "longest_missing_run": int(group["missing_run_length"].max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("delta_rmse_mask_minus_control", ascending=False)


def build_divergence_segments(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=True):
        group = group.sort_values("row_index").reset_index(drop=True)
        changed = group["prediction_changed"].to_numpy(bool)
        row_index = group["row_index"].to_numpy(np.int64)
        start: int | None = None
        for position in range(len(group) + 1):
            active = position < len(group) and changed[position]
            contiguous = (
                active
                and start is not None
                and position > 0
                and row_index[position] == row_index[position - 1] + 1
            )
            if active and start is None:
                start = position
            elif active and not contiguous:
                stop = position
                segment = group.iloc[start:stop]
                rows.append(
                    {
                        "well": well,
                        "start_row": int(segment["row_index"].iloc[0]),
                        "end_row": int(segment["row_index"].iloc[-1]),
                        "length_rows": int(len(segment)),
                        "raw_missing_rows": int(segment["raw_gr_missing"].sum()),
                        "post_gap_rows_in_segment": int((segment["post_gap_rows"] > 0).sum()),
                        "mean_abs_mask_minus_control": float(
                            segment["abs_mask_minus_control"].mean()
                        ),
                        "max_abs_mask_minus_control": float(
                            segment["abs_mask_minus_control"].max()
                        ),
                    }
                )
                start = position
            elif not active and start is not None:
                segment = group.iloc[start:position]
                rows.append(
                    {
                        "well": well,
                        "start_row": int(segment["row_index"].iloc[0]),
                        "end_row": int(segment["row_index"].iloc[-1]),
                        "length_rows": int(len(segment)),
                        "raw_missing_rows": int(segment["raw_gr_missing"].sum()),
                        "post_gap_rows_in_segment": int((segment["post_gap_rows"] > 0).sum()),
                        "mean_abs_mask_minus_control": float(
                            segment["abs_mask_minus_control"].mean()
                        ),
                        "max_abs_mask_minus_control": float(
                            segment["abs_mask_minus_control"].max()
                        ),
                    }
                )
                start = None
    return pd.DataFrame(rows)


# %% [markdown]
# ## 7. Synthetic emission contract

# %%
synthetic_gr = np.array([10.0, 20.0, 30.0], dtype=np.float64)
synthetic_grid = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float64)
synthetic_missing = np.array([False, True, False])
synthetic_control = build_gr_emission_loglik(
    synthetic_gr,
    synthetic_grid,
    10.0,
    emission="gauss",
    df=4.0,
    raw_gr_missing=synthetic_missing,
    mask_missing_gr=False,
)
synthetic_masked = build_gr_emission_loglik(
    synthetic_gr,
    synthetic_grid,
    10.0,
    emission="gauss",
    df=4.0,
    raw_gr_missing=synthetic_missing,
    mask_missing_gr=True,
)
assert np.array_equal(synthetic_masked[~synthetic_missing], synthetic_control[~synthetic_missing])
assert np.array_equal(
    synthetic_masked[synthetic_missing], np.zeros((1, len(synthetic_grid)), np.float32)
)
assert not np.array_equal(synthetic_control[synthetic_missing], synthetic_masked[synthetic_missing])
synthetic_lgb_ll = np.full_like(synthetic_masked, -2.0, dtype=np.float32)
synthetic_combined = synthetic_masked + np.float32(0.5) * synthetic_lgb_ll
assert np.all(synthetic_combined[synthetic_missing] == -1.0)
print(
    "Synthetic contract PASS: raw-missing GR unary=0; observed unary unchanged; LGB unary retained."
)


# %% [markdown]
# ## 8. Setup and cost guard

# %%
require_authoritative_runtime()
if not NUMBA_AVAILABLE:
    raise RuntimeError("numba is required for the exact HMM full run")

TRAIN_DIR = resolve_split_dir("train")
TEST_DIR = resolve_split_dir("test")
OUTPUT_DIR = output_dir()
OUTPUT_FILENAMES = dict(nested("audit.outputs"))
OUTPUT_PATHS = {key: OUTPUT_DIR / value for key, value in OUTPUT_FILENAMES.items()}
CONTROL_PATH = resolve_existing_file(list(nested("data.control_cache_candidates")))
HIDDEN_LIKE_PATH = resolve_existing_file(list(nested("data.hidden_like_candidates")))
HMM_CONFIG = dict(nested("model.hmm"))
MAX_WELLS = nested("audit.max_wells")
OUTER_WORKERS = max(1, int(nested("runtime.outer_workers", 1)))
NUMBA_THREADS = max(1, int(nested("runtime.numba_num_threads", 1)))
set_num_threads(NUMBA_THREADS)

cost_guard = {
    "experiment": EXPERIMENT_NAME,
    "route": nested("experiment.route"),
    "active_variants": list(nested("model.active_variants")),
    "active_variant_count": len(list(nested("model.active_variants"))),
    "lightgbm_config_count": int(nested("model.lightgbm_config_count")),
    "fold_training_count": int(nested("model.fold_training_count")),
    "booster_count": int(nested("model.booster_count")),
    "parent_control_retraining": bool(nested("model.parent_control_retraining")),
    "control_source": str(CONTROL_PATH),
    "gpu": bool(nested("runtime.kaggle.enable_gpu")),
    "inference_enabled": bool(nested("inference.enabled")),
    "outer_workers": OUTER_WORKERS,
    "numba_threads_requested": NUMBA_THREADS,
    "numba_threads_effective": get_num_threads(),
}
if cost_guard["active_variant_count"] != 1:
    raise ValueError("exp247 must run exactly one mask_only variant")
if any(
    cost_guard[key] != 0
    for key in ("lightgbm_config_count", "fold_training_count", "booster_count")
):
    raise ValueError("exp247 must not train LightGBM models or boosters")
if cost_guard["parent_control_retraining"] or cost_guard["inference_enabled"]:
    raise ValueError("exp247 must reuse the fixed control and keep inference disabled")
print(json.dumps(to_jsonable(cost_guard), indent=2, sort_keys=True))
print("HMM config:", json.dumps(to_jsonable(HMM_CONFIG), indent=2, sort_keys=True))


# %% [markdown]
# ## 9. Raw train/test distribution and fixed-control readout

# %%
train_inventory, train_runs, train_context, train_sha = scan_raw_split(
    "train", TRAIN_DIR, include_eval_context=True
)
test_inventory, test_runs, _, test_sha = scan_raw_split(
    "test", TEST_DIR, include_eval_context=False
)
missing_inventory = pd.concat([train_inventory, test_inventory], ignore_index=True)
missing_runs = pd.concat([train_runs, test_runs], ignore_index=True)
raw_file_sha = pd.concat([train_sha, test_sha], ignore_index=True)

control = load_control_frame(CONTROL_PATH)
wells = list_well_ids(TRAIN_DIR)
if MAX_WELLS is not None:
    wells = wells[: int(MAX_WELLS)]
    selected_wells = set(wells)
    control = control[control["well"].isin(selected_wells)].copy()
    train_context = train_context[train_context["well"].isin(selected_wells)].copy()
control_context = control.merge(
    train_context,
    on=["id", "well"],
    how="inner",
    validate="one_to_one",
)
if len(control_context) != len(control) or len(control_context) != len(train_context):
    raise ValueError(
        "Control/context id coverage mismatch: "
        f"control={len(control)}, context={len(train_context)}, joined={len(control_context)}"
    )
control_readout_parts = [
    single_control_group_metrics(
        control_context,
        "gr_availability",
        np.where(control_context["raw_gr_missing"], "missing", "observed"),
    ),
    single_control_group_metrics(
        control_context,
        "missing_run",
        control_context["missing_run_bucket"].to_numpy(object),
    ),
    single_control_group_metrics(
        control_context,
        "post_gap",
        control_context["post_gap_bucket"].to_numpy(object),
    ),
]
control_readout = pd.concat(control_readout_parts, ignore_index=True)

# Persist the prerequisite distribution/control readout before the expensive
# exact-HMM run. If the later CPU job times out, these diagnostics remain
# available and the scientific change has not silently skipped its pre-readout.
missing_inventory.to_csv(OUTPUT_PATHS["missing_well_filename"], index=False)
missing_runs.to_csv(OUTPUT_PATHS["missing_run_filename"], index=False)
raw_file_sha.to_csv(OUTPUT_PATHS["raw_file_sha_filename"], index=False)
control_readout.to_csv(OUTPUT_PATHS["control_readout_filename"], index=False)

display(
    missing_inventory.groupby("split", as_index=False).agg(
        wells=("well", "nunique"),
        rows=("rows", "sum"),
        missing_rows=("missing_rows", "sum"),
        evaluation_rows=("evaluation_rows", "sum"),
        evaluation_missing_rows=("evaluation_missing_rows", "sum"),
        longest_missing_run=("longest_missing_run", "max"),
    )
)
display(
    missing_runs.groupby(["split", "run_length"], as_index=False)
    .agg(runs=("well", "size"), wells=("well", "nunique"))
    .sort_values(["split", "run_length"])
    .tail(30)
)
display(control_readout)
print(
    "Visible test distribution is descriptive only; it is not score evidence and "
    "does not alter the mask policy."
)
del train_context, control_context


# %% [markdown]
# ## 10. Full mask-only generation

# %%
lgb_variants, lgb_variant_summary = prepare_lgb_emission_variants(
    ROOT,
    dict(nested("lgb_emission")),
)
if len(lgb_variants) != 1:
    raise ValueError(f"Expected one fixed exp148 LGB unary variant, got {len(lgb_variants)}")
lgb_variant = lgb_variants[0]


def generate_one(
    index: int,
    well: str,
    variant: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(f"[{index}/{len(wells)}] mask-only exact HMM well={well}", flush=True)
    frame, meta = build_mask_only_rows_for_well(well, TRAIN_DIR, HMM_CONFIG, variant)
    print(json.dumps(to_jsonable(meta), sort_keys=True), flush=True)
    return frame, meta


generation_started = time.time()
if OUTER_WORKERS > 1:
    from joblib import Parallel, delayed

    generated = Parallel(n_jobs=OUTER_WORKERS, prefer="threads")(
        delayed(generate_one)(index, well, lgb_variant) for index, well in enumerate(wells, start=1)
    )
else:
    generated = [
        generate_one(index, well, lgb_variant) for index, well in enumerate(wells, start=1)
    ]

mask_frames = [frame for frame, _ in generated if len(frame)]
generation_meta = pd.DataFrame([meta for _, meta in generated])
if not mask_frames:
    raise ValueError("Mask-only HMM generation produced no rows")
mask_frame = pd.concat(mask_frames, ignore_index=True)
if mask_frame["id"].duplicated().any():
    raise ValueError("Mask-only HMM generation produced duplicate ids")
merged = mask_frame.merge(
    control,
    on=["id", "well"],
    how="inner",
    validate="one_to_one",
)
if len(merged) != len(mask_frame) or len(merged) != len(control):
    raise ValueError(
        "Mask-only/fixed-control id coverage mismatch: "
        f"mask={len(mask_frame)}, control={len(control)}, joined={len(merged)}"
    )
target_abs_diff = np.abs(merged["true_tvt_raw"].to_numpy() - merged["true_tvt"].to_numpy())
md_abs_diff = np.abs(merged["md_since_raw"].to_numpy() - merged["md_since"].to_numpy())
if not np.nanmax(target_abs_diff) <= 1e-5:
    raise ValueError(f"Raw TVT and fixed-control target mismatch: max={np.nanmax(target_abs_diff)}")
if not np.nanmax(md_abs_diff) <= 1e-5:
    raise ValueError(f"Raw MD distance and fixed-control mismatch: max={np.nanmax(md_abs_diff)}")

hidden_like = pd.read_csv(HIDDEN_LIKE_PATH, dtype={"well_id": str})
role_columns = dict(nested("audit.hidden_like_roles"))
needed_roles = ["well_id", *role_columns.values()]
missing_role_columns = sorted(set(needed_roles).difference(hidden_like.columns))
if missing_role_columns:
    raise ValueError(f"Hidden-like assignment missing columns: {missing_role_columns}")
hidden_like = hidden_like[needed_roles].rename(columns={"well_id": "well"})
merged = merged.merge(hidden_like, on="well", how="left", validate="many_to_one")
merged["distance_bucket"] = distance_bucket(merged["md_since"])
merged["abs_mask_minus_control"] = np.abs(
    merged["mask_tvt"].to_numpy(np.float64) - merged["control_tvt"].to_numpy(np.float64)
)
change_tolerance = float(nested("audit.prediction_change_tolerance_ft", 1e-6))
merged["prediction_changed"] = merged["abs_mask_minus_control"] > change_tolerance
missing_rows_by_well = merged.groupby("well")["raw_gr_missing"].transform("sum")
no_eval_missing = missing_rows_by_well == 0
control_parity_tolerance = float(nested("audit.fixed_control_parity_tolerance_ft", 1e-5))
no_missing_control_parity_max_abs = (
    float(merged.loc[no_eval_missing, "abs_mask_minus_control"].max())
    if no_eval_missing.any()
    else None
)
if (
    no_missing_control_parity_max_abs is not None
    and no_missing_control_parity_max_abs > control_parity_tolerance
):
    raise ValueError(
        "Fixed-control parity failed on wells with no evaluation GR missing rows: "
        f"max_abs={no_missing_control_parity_max_abs}, "
        f"tolerance={control_parity_tolerance}"
    )
del mask_frame, mask_frames, generated, control, lgb_variants, lgb_variant


# %% [markdown]
# ## 11. Metrics, SHA, and generated files

# %%
metric_parts = [
    pd.DataFrame([paired_metric_row(merged, "overall", "all")]),
    grouped_paired_metrics(
        merged,
        "gr_availability",
        np.where(merged["raw_gr_missing"], "missing", "observed"),
    ),
    grouped_paired_metrics(
        merged,
        "missing_run",
        merged["missing_run_bucket"].to_numpy(object),
    ),
    grouped_paired_metrics(
        merged,
        "post_gap",
        merged["post_gap_bucket"].to_numpy(object),
    ),
    grouped_paired_metrics(
        merged,
        "distance",
        merged["distance_bucket"].to_numpy(object),
    ),
]
short_missing = merged["raw_gr_missing"] & (merged["missing_run_length"] <= 31)
if short_missing.any():
    metric_parts.append(
        pd.DataFrame([paired_metric_row(merged.loc[short_missing], "guard", "missing_run_1_31")])
    )
for subgroup, role_column in role_columns.items():
    selected = merged[role_column].astype(str) == "valid"
    if selected.any():
        metric_parts.append(
            pd.DataFrame([paired_metric_row(merged.loc[selected], "hidden_like", subgroup)])
        )
group_metrics = pd.concat(metric_parts, ignore_index=True)
by_well = build_by_well_metrics(merged)
divergence_segments = build_divergence_segments(merged)

finite_coverage = pd.DataFrame(
    [
        {
            "candidate": "exp221_interpolation_control",
            "rows": int(len(merged)),
            "prediction_finite_rows": int(np.isfinite(merged["control_tvt"]).sum()),
            "std_finite_rows": int(np.isfinite(merged["control_std"]).sum()),
            "parent_finite_flag_rows": int((merged["control_finite_parent"] > 0).sum()),
            "finite_wells": int(
                merged.groupby("well")[["control_tvt", "control_std"]]
                .apply(lambda group: np.isfinite(group.to_numpy(np.float64)).all())
                .sum()
            ),
        },
        {
            "candidate": "mask_only",
            "rows": int(len(merged)),
            "prediction_finite_rows": int(np.isfinite(merged["mask_tvt"]).sum()),
            "std_finite_rows": int(np.isfinite(merged["mask_std"]).sum()),
            "parent_finite_flag_rows": None,
            "finite_wells": int(generation_meta["finite"].fillna(False).sum()),
        },
    ]
)

group_metrics.to_csv(OUTPUT_PATHS["group_metrics_filename"], index=False)
by_well.to_csv(OUTPUT_PATHS["by_well_filename"], index=False)
divergence_segments.to_csv(OUTPUT_PATHS["divergence_segment_filename"], index=False)
finite_coverage.to_csv(OUTPUT_PATHS["finite_coverage_filename"], index=False)
generation_meta.to_csv(OUTPUT_PATHS["generation_well_filename"], index=False)

prediction_columns = ["id", "well", "mask_tvt", "mask_std", "mask_finite"]
write_deterministic_gzip_csv(merged[prediction_columns], OUTPUT_PATHS["prediction_filename"])
row_audit_columns = [
    "id",
    "well",
    "row_index",
    "true_tvt",
    "control_tvt",
    "mask_tvt",
    "control_std",
    "mask_std",
    "raw_gr_missing",
    "missing_run_length",
    "missing_run_bucket",
    "post_gap_rows",
    "post_gap_bucket",
    "md_since",
    "distance_bucket",
    "abs_mask_minus_control",
    "prediction_changed",
    *role_columns.values(),
]
write_deterministic_gzip_csv(merged[row_audit_columns], OUTPUT_PATHS["row_audit_filename"])

input_sha = {
    "config": sha256_path(CONFIG_PATH),
    "control_cache_raw": sha256_path(CONTROL_PATH),
    "control_cache_decompressed": sha256_gzip_decompressed(CONTROL_PATH),
    "lgb_oof_source_raw": sha256_path(Path(lgb_variant_summary[0]["source_meta"]["path"])),
    "hidden_like_assignment": sha256_path(HIDDEN_LIKE_PATH),
    "raw_file_inventory": sha256_path(OUTPUT_PATHS["raw_file_sha_filename"]),
}
output_sha: dict[str, Any] = {}
for key, path in OUTPUT_PATHS.items():
    if key == "summary_filename":
        continue
    output_sha[f"{key}_raw"] = sha256_path(path)
    if path.suffix == ".gz":
        output_sha[f"{key}_decompressed"] = sha256_gzip_decompressed(path)

overall = (
    group_metrics[(group_metrics["group_type"] == "overall") & (group_metrics["group"] == "all")]
    .iloc[0]
    .to_dict()
)
missing_metric_rows = group_metrics[
    (group_metrics["group_type"] == "gr_availability") & (group_metrics["group"] == "missing")
]
missing_metric = missing_metric_rows.iloc[0].to_dict() if len(missing_metric_rows) else None
worst_row = by_well.iloc[0].to_dict() if len(by_well) else None
summary = {
    "experiment": EXPERIMENT_NAME,
    "status": "train_side_mask_ablation_completed",
    "route": nested("experiment.route"),
    "cost_guard": cost_guard,
    "runtime": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "kaggle": is_kaggle_runtime(),
        "kernel_version_env": os.environ.get("KAGGLE_KERNEL_VERSION"),
        "outer_workers": OUTER_WORKERS,
        "numba_threads": get_num_threads(),
        "elapsed_seconds": round(time.time() - generation_started, 3),
    },
    "rows": int(len(merged)),
    "wells": int(merged["well"].nunique()),
    "raw_gr_missing_rows": int(merged["raw_gr_missing"].sum()),
    "raw_gr_missing_wells": int(merged.loc[merged["raw_gr_missing"], "well"].nunique()),
    "prediction_changed_rows": int(merged["prediction_changed"].sum()),
    "prediction_changed_wells": int(merged.loc[merged["prediction_changed"], "well"].nunique()),
    "no_eval_missing_control_parity_max_abs": no_missing_control_parity_max_abs,
    "fixed_control_parity_tolerance_ft": control_parity_tolerance,
    "divergence_segments": int(len(divergence_segments)),
    "longest_divergence_segment_rows": (
        int(divergence_segments["length_rows"].max()) if len(divergence_segments) else 0
    ),
    "overall": overall,
    "raw_missing_rows_metric": missing_metric,
    "worst_well_regression": worst_row,
    "finite_coverage": finite_coverage.to_dict(orient="records"),
    "lgb_unary": lgb_variant_summary,
    "input_sha256": input_sha,
    "output_sha256": output_sha,
    "outputs": {key: path.name for key, path in OUTPUT_PATHS.items()},
    "decision": "pending_user_review_after_full_kaggle_run",
    "notes": [
        "The saved exp221 interpolation control is loaded, not regenerated.",
        "Only raw-missing evaluation rows have their GR emission contribution set to zero.",
        "Transition and the fixed exp148 LGB Gaussian unary remain active on masked rows.",
        "Visible test missingness is descriptive only and is not score evidence.",
        "No run-length gate, hidden-test inference, selector, or submission is produced.",
    ],
}
write_json(OUTPUT_PATHS["summary_filename"], summary)

metrics_payload = {
    "experiment": EXPERIMENT_NAME,
    "status": summary["status"],
    "cv": overall.get("mask_rmse"),
    "control_cv": overall.get("control_rmse"),
    "delta_cv_mask_minus_control": overall.get("delta_rmse_mask_minus_control"),
    "public_lb": None,
    "private_lb": None,
    "metric": "rmse",
    "rows": summary["rows"],
    "wells": summary["wells"],
    "raw_gr_missing_rows": summary["raw_gr_missing_rows"],
    "finite_coverage": summary["finite_coverage"],
    "input_sha256": input_sha,
    "output_sha256": output_sha,
    "summary": OUTPUT_PATHS["summary_filename"].name,
}
write_json(metrics_output_path(), metrics_payload)

display(group_metrics)
display(by_well.head(20))
display(finite_coverage)
print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)

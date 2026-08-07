# %% [markdown]
# # exp244 bidirectional prediction-start pseudo-tail augmentation — train audit
#
# This notebook builds deterministic early/original/late prediction-start views.
# It does not train a model, run inference, or create a submission.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and input resolution
# 3. Official-start metadata and source-well folds
# 4. Bidirectional view manifest
# 5. Prefix reconstruction and feature materialization
# 6. Leakage, distribution, and reproducibility guards
# 7. Audit orchestration
# 8. Metrics and generated files

# %%
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp244_bidirectional_prediction_start_pseudotail_augmentation"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


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


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(find_config_path().read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def resolve_split_dir(config: dict[str, Any], split: str) -> Path:
    configured = Path(str(nested(config, f"data.{split}_dir", f"data/raw/{split}")))
    local = configured if configured.is_absolute() else ROOT / configured
    pattern = str(nested(config, "data.horizontal_glob", "*__horizontal_well.csv"))
    if local.exists() and any(local.glob(pattern)):
        return local
    if KAGGLE_INPUT_ROOT.exists():
        for source in sorted(KAGGLE_INPUT_ROOT.iterdir()):
            candidate = source / split
            if candidate.is_dir() and any(candidate.glob(pattern)):
                return candidate
        matches = sorted(KAGGLE_INPUT_ROOT.rglob(pattern))
        for match in matches:
            if match.parent.name == split:
                return match.parent
    raise FileNotFoundError(f"Could not resolve {split} directory with {pattern}")


def output_dir() -> Path:
    if is_kaggle_runtime():
        path = KAGGLE_WORKING_ROOT
    else:
        path = ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def well_id_from_path(path: Path) -> str:
    suffix = "__horizontal_well.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected horizontal filename: {path.name}")
    return path.name[: -len(suffix)]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts).encode()
    return hashlib.sha256(payload).hexdigest()


class HashWriter:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def write(self, value: str) -> int:
        encoded = value.encode()
        self.digest.update(encoded)
        return len(value)

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def canonical_csv_sha256(frame: pd.DataFrame) -> str:
    writer = HashWriter()
    frame.to_csv(writer, index=False, lineterminator="\n")
    return writer.hexdigest()


# %% [markdown]
# ## 3. Official-start metadata and source-well folds


# %%
def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def read_train_well(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        str(nested(config, "data.md_column", "MD")),
        *list(nested(config, "data.coordinate_columns", ["X", "Y", "Z"])),
        str(nested(config, "data.gr_column", "GR")),
        str(nested(config, "data.target_column", "TVT")),
        str(nested(config, "data.input_target_column", "TVT_input")),
    ]
    frame = pd.read_csv(path, usecols=lambda value: value in set(columns))
    require_columns(frame, columns, path.name)
    return frame


def official_metadata_for_well(
    path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = read_train_well(path, config)
    input_column = str(nested(config, "data.input_target_column", "TVT_input"))
    target_column = str(nested(config, "data.target_column", "TVT"))
    tvt_input = pd.to_numeric(frame[input_column], errors="coerce").to_numpy(float)
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if known.size == 0:
        raise ValueError(f"{path.name} has no known TVT_input prefix")
    official = int(known[-1])
    if not np.array_equal(known, np.arange(official + 1)):
        raise ValueError(f"{path.name} TVT_input is not one contiguous prefix")
    if not np.all(np.isfinite(target)):
        raise ValueError(f"{path.name} train TVT must be finite for train-only augmentation")
    if not np.allclose(tvt_input[: official + 1], target[: official + 1]):
        raise ValueError(f"{path.name} official TVT_input does not match train TVT")
    n_rows = len(frame)
    if official >= n_rows - 1:
        raise ValueError(f"{path.name} has no official evaluation tail")
    return (
        {
            "well_id": well_id_from_path(path),
            "source_path": str(path),
            "source_file_sha256": sha256_file(path),
            "n_rows": int(n_rows),
            "official_start_index": official,
            "official_prefix_rows": official + 1,
            "official_remaining_tail_rows": n_rows - official - 1,
        },
        frame,
    )


def build_well_metadata(
    train_files: list[Path], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(train_files, key=well_id_from_path):
        row, frame = official_metadata_for_well(path, config)
        rows.append(row)
        frames[str(row["well_id"])] = frame
    metadata = pd.DataFrame(rows).sort_values("well_id").reset_index(drop=True)
    if metadata.empty or metadata["well_id"].duplicated().any():
        raise AssertionError("Well metadata is empty or has duplicate well IDs")
    return metadata, frames


def build_fold_manifest(metadata: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    wells = (
        metadata[["well_id", "official_remaining_tail_rows"]]
        .sort_values("well_id")
        .reset_index(drop=True)
    )
    n_folds = int(nested(config, "validation.n_folds", 5))
    if len(wells) < n_folds:
        raise ValueError(f"Need at least {n_folds} source wells")
    assignment = str(
        nested(
            config,
            "validation.fold_assignment",
            "exp218_official_tail_row_weighted_groupkfold",
        )
    )
    if assignment != "exp218_official_tail_row_weighted_groupkfold":
        raise ValueError(f"Unsupported fold assignment: {assignment}")
    weights = wells["official_remaining_tail_rows"].to_numpy(dtype=np.int64)
    if np.any(weights <= 0):
        raise AssertionError("Every source well must have positive official-tail rows")
    order = np.argsort(weights)[::-1]
    fold_loads = np.zeros(n_folds, dtype=np.int64)
    fold_ids = np.full(len(wells), -1, dtype=int)
    for well_index in order:
        fold = int(np.argmin(fold_loads))
        fold_ids[well_index] = fold
        fold_loads[fold] += weights[well_index]
    wells["fold"] = fold_ids
    wells["fold_assignment"] = assignment
    wells["fold_key"] = [
        stable_key(EXPERIMENT_NAME, well, fold)
        for well, fold in zip(wells["well_id"], fold_ids, strict=True)
    ]
    return wells


# %% [markdown]
# ## 4. Bidirectional view manifest


# %%
def start_kind(offset: int) -> str:
    if offset < 0:
        return "early"
    if offset > 0:
        return "late"
    return "original"


def build_view_manifest(
    metadata: pd.DataFrame, fold_manifest: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    offsets = [
        int(value) for value in nested(config, "model.view_generation.start_offsets_rows", [])
    ]
    if sorted(set(offsets)) != sorted(offsets) or 0 not in offsets:
        raise ValueError("start_offsets_rows must be unique and include 0")
    min_prefix = int(nested(config, "model.view_generation.min_prefix_rows", 200))
    min_tail = int(nested(config, "model.view_generation.min_remaining_tail_rows", 50))
    max_views = int(nested(config, "model.view_generation.max_views_per_well", len(offsets)))
    contract = nested(config, "model.replay_contract", {})
    fold_by_well = fold_manifest.set_index("well_id")["fold"]
    rows: list[dict[str, Any]] = []
    for meta in metadata.sort_values("well_id").itertuples(index=False):
        accepted = 0
        for offset in offsets:
            start = int(meta.official_start_index) + offset
            prefix_rows = start + 1
            remaining = int(meta.n_rows) - prefix_rows
            if prefix_rows < min_prefix or remaining < min_tail:
                continue
            kind = start_kind(offset)
            request_id = stable_key(
                EXPERIMENT_NAME, meta.well_id, meta.official_start_index, offset
            )
            rows.append(
                {
                    "request_id": request_id,
                    "source_well": str(meta.well_id),
                    "source_path": str(meta.source_path),
                    "source_fold": int(fold_by_well.loc[str(meta.well_id)]),
                    "start_kind": kind,
                    "start_offset_rows": offset,
                    "start_index": start,
                    "mask_start_index": start + 1,
                    "official_start_index": int(meta.official_start_index),
                    "prefix_rows": prefix_rows,
                    "remaining_tail_rows": remaining,
                    "prefix_fraction": prefix_rows / int(meta.n_rows),
                    "late_train_only": kind == "late",
                    "current_test_compatible": kind != "late",
                    "calibration_backtest_compatible": kind == "early",
                    "calibration_backtest_end_index": int(meta.official_start_index)
                    if kind == "early"
                    else -1,
                    "target_usage": "train_only_augmentation"
                    if kind == "late"
                    else ("official_control" if kind == "original" else "outer_train_augmentation"),
                    "validation_surface": "official_start_only",
                    "outer_train_exclusion_key": str(meta.well_id),
                    "mask_columns_after_start": "|".join(
                        contract.get("mask_columns_after_start", [])
                    ),
                    "target_only_columns_after_start": "|".join(
                        contract.get("target_only_columns_after_start", [])
                    ),
                    "regenerate_feature_groups": "|".join(
                        contract.get("regenerate_feature_groups", [])
                    ),
                    "forbid_full_prefix_cache_slice": bool(
                        contract.get("forbid_full_prefix_cache_slice", True)
                    ),
                    "feature_generation_may_read_tail_tvt": bool(
                        contract.get("feature_generation_may_read_tail_tvt", False)
                    ),
                }
            )
            accepted += 1
        if accepted > max_views:
            raise AssertionError(f"{meta.well_id} exceeds max_views_per_well")
    manifest = (
        pd.DataFrame(rows)
        .sort_values(["source_fold", "source_well", "start_offset_rows"])
        .reset_index(drop=True)
    )
    max_total = int(nested(config, "model.view_generation.max_total_views", 4000))
    if len(manifest) > max_total:
        raise AssertionError(f"View count {len(manifest)} exceeds max_total_views={max_total}")
    return manifest


def assert_manifest_contract(
    manifest: pd.DataFrame,
    metadata: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    if manifest.empty or manifest["request_id"].duplicated().any():
        raise AssertionError("View manifest is empty or has duplicate request IDs")
    if manifest.groupby("source_well")["source_fold"].nunique().max() != 1:
        raise AssertionError("A source well was assigned to multiple folds")
    expected_folds = fold_manifest.set_index("well_id")["fold"]
    aligned = expected_folds.reindex(manifest["source_well"]).to_numpy(dtype=int)
    if not np.array_equal(aligned, manifest["source_fold"].to_numpy(dtype=int)):
        raise AssertionError("Derived view fold does not match source-well fold")
    relation = np.sign(manifest["start_offset_rows"].to_numpy(dtype=int))
    expected_kind = np.where(relation < 0, "early", np.where(relation > 0, "late", "original"))
    if not np.array_equal(expected_kind, manifest["start_kind"].to_numpy(str)):
        raise AssertionError("start_kind and start_offset_rows disagree")
    if not np.array_equal(
        manifest["start_index"].to_numpy(int),
        manifest["official_start_index"].to_numpy(int)
        + manifest["start_offset_rows"].to_numpy(int),
    ):
        raise AssertionError("start index does not match official index plus offset")
    late = manifest["start_kind"].eq("late")
    if (
        not manifest.loc[late, "late_train_only"].all()
        or manifest.loc[late, "current_test_compatible"].any()
    ):
        raise AssertionError("Late views must be train-only and current-test incompatible")
    if manifest["feature_generation_may_read_tail_tvt"].any():
        raise AssertionError("Replay contract permits tail TVT feature leakage")
    if not manifest["forbid_full_prefix_cache_slice"].all():
        raise AssertionError("Replay contract permits full-prefix cache slicing")
    if bool(nested(config, "model.view_generation.require_original_view", True)):
        expected_original = set(metadata["well_id"])
        actual_original = set(manifest.loc[manifest["start_kind"].eq("original"), "source_well"])
        if actual_original != expected_original:
            raise AssertionError("Every source well must retain its original-start control")
    late_share = float(late.mean())
    max_late_share = float(nested(config, "model.view_generation.max_late_view_share", 0.45))
    if late_share > max_late_share:
        raise AssertionError(f"late view share {late_share:.6f} exceeds {max_late_share:.6f}")
    return {
        "request_ids_unique": True,
        "one_fold_per_source_well": True,
        "source_fold_alignment": True,
        "start_relation_valid": True,
        "late_train_only": True,
        "current_test_unknown_tail_forbidden": True,
        "full_prefix_cache_slice_forbidden": True,
        "tail_tvt_feature_read_forbidden": True,
        "all_wells_have_original_control": True,
        "late_view_share": late_share,
        "max_late_view_share": max_late_share,
    }


# %% [markdown]
# ## 5. Prefix reconstruction and feature materialization


# %%
def evenly_spaced_indices(indices: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if count <= 0 or values.size == 0:
        return np.empty(0, dtype=int)
    if values.size <= count:
        return values
    positions = np.linspace(0, values.size - 1, count).round().astype(int)
    return values[np.unique(positions)]


def sample_tail_rows(start: int, n_rows: int, config: dict[str, Any]) -> np.ndarray:
    max_rows = int(nested(config, "model.materialization.max_rows_per_view", 1000))
    tail = np.arange(start + 1, n_rows, dtype=int)
    steps = tail - start - 1
    selected: list[int] = []
    for bucket in nested(config, "model.materialization.distance_buckets", []):
        minimum = int(bucket["min_step"])
        maximum = bucket.get("max_step")
        mask = steps >= minimum
        if maximum is not None:
            mask &= steps <= int(maximum)
        candidates = tail[mask]
        selected.extend(evenly_spaced_indices(candidates, int(bucket["quota"])).tolist())
    unique = np.asarray(sorted(set(selected)), dtype=int)
    if (
        bool(nested(config, "model.materialization.fill_remaining", True))
        and len(unique) < max_rows
    ):
        unused = np.setdiff1d(tail, unique, assume_unique=True)
        fill = evenly_spaced_indices(unused, max_rows - len(unique))
        unique = np.asarray(sorted(set(unique.tolist() + fill.tolist())), dtype=int)
    return unique[:max_rows]


def robust_rate(values: np.ndarray, md: np.ndarray) -> float:
    delta_md = np.diff(md)
    delta_value = np.diff(values)
    valid = np.isfinite(delta_md) & np.isfinite(delta_value) & (np.abs(delta_md) > 1e-12)
    if not np.any(valid):
        return 0.0
    return float(np.median(delta_value[valid] / delta_md[valid]))


def prefix_summary(
    feature_frame: pd.DataFrame, start: int, config: dict[str, Any]
) -> dict[str, float]:
    input_column = str(nested(config, "data.input_target_column", "TVT_input"))
    gr_column = str(nested(config, "data.gr_column", "GR"))
    md_column = str(nested(config, "data.md_column", "MD"))
    prefix = feature_frame.iloc[: start + 1]
    tvt = pd.to_numeric(prefix[input_column], errors="coerce").to_numpy(float)
    gr = pd.to_numeric(prefix[gr_column], errors="coerce").to_numpy(float)
    md = pd.to_numeric(prefix[md_column], errors="coerce").to_numpy(float)
    result = {
        "prefix_gr_mean": float(np.nanmean(gr)) if np.any(np.isfinite(gr)) else 0.0,
        "prefix_gr_std": float(np.nanstd(gr)) if np.any(np.isfinite(gr)) else 0.0,
        "prefix_gr_missing_rate": float(np.mean(~np.isfinite(gr))),
        "prefix_tvt_md_rate": robust_rate(tvt, md),
    }
    for window in nested(config, "model.materialization.recent_windows", [32, 128]):
        size = int(window)
        recent_gr = gr[-size:]
        recent_tvt = tvt[-size:]
        recent_md = md[-size:]
        result[f"prefix_gr_mean_last{size}"] = (
            float(np.nanmean(recent_gr)) if np.any(np.isfinite(recent_gr)) else 0.0
        )
        result[f"prefix_gr_std_last{size}"] = (
            float(np.nanstd(recent_gr)) if np.any(np.isfinite(recent_gr)) else 0.0
        )
        result[f"prefix_tvt_md_rate_last{size}"] = robust_rate(recent_tvt, recent_md)
    return result


def rebuild_tvt_input(
    frame: pd.DataFrame, start: int, config: dict[str, Any]
) -> tuple[pd.DataFrame, np.ndarray]:
    target_column = str(nested(config, "data.target_column", "TVT"))
    input_column = str(nested(config, "data.input_target_column", "TVT_input"))
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float)
    feature_frame = frame.drop(columns=[target_column]).copy()
    feature_frame[input_column] = np.nan
    feature_frame.loc[:start, input_column] = target[: start + 1]
    if feature_frame.loc[start + 1 :, input_column].notna().any():
        raise AssertionError("Rebuilt TVT_input exposes target after selected start")
    return feature_frame, target


def materialize_prefix_features(
    manifest: pd.DataFrame, frames: dict[str, pd.DataFrame], config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    md_column = str(nested(config, "data.md_column", "MD"))
    gr_column = str(nested(config, "data.gr_column", "GR"))
    coordinates = list(nested(config, "data.coordinate_columns", ["X", "Y", "Z"]))
    rows: list[pd.DataFrame] = []
    request_rows: list[dict[str, Any]] = []
    for request in manifest.sort_values("request_id").itertuples(index=False):
        frame = frames[str(request.source_well)]
        start = int(request.start_index)
        feature_frame, target = rebuild_tvt_input(frame, start, config)
        if request.start_kind == "original":
            original = pd.to_numeric(
                frame[str(nested(config, "data.input_target_column", "TVT_input"))], errors="coerce"
            ).to_numpy(float)
            rebuilt = pd.to_numeric(
                feature_frame[str(nested(config, "data.input_target_column", "TVT_input"))],
                errors="coerce",
            ).to_numpy(float)
            if not np.array_equal(np.isfinite(original), np.isfinite(rebuilt)) or not np.allclose(
                original[np.isfinite(original)], rebuilt[np.isfinite(rebuilt)]
            ):
                raise AssertionError("Original-start reconstruction is not identical to TVT_input")
        selected = sample_tail_rows(start, len(frame), config)
        if selected.size == 0:
            raise AssertionError(f"No rows sampled for {request.request_id}")
        md = pd.to_numeric(feature_frame[md_column], errors="coerce").to_numpy(float)
        gr = pd.to_numeric(feature_frame[gr_column], errors="coerce").to_numpy(float)
        xyz = feature_frame[coordinates].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        tvt_input = pd.to_numeric(
            feature_frame[str(nested(config, "data.input_target_column", "TVT_input"))],
            errors="coerce",
        ).to_numpy(float)
        anchor_md = float(md[start])
        anchor_xyz = xyz[start]
        anchor_tvt = float(tvt_input[start])
        summary = prefix_summary(feature_frame, start, config)
        part = pd.DataFrame(
            {
                "request_id": str(request.request_id),
                "source_well": str(request.source_well),
                "source_fold": int(request.source_fold),
                "start_kind": str(request.start_kind),
                "start_offset_rows": int(request.start_offset_rows),
                "start_index": start,
                "official_start_index": int(request.official_start_index),
                "row_index": selected,
                "eval_step": selected - start - 1,
                "prefix_rows": int(request.prefix_rows),
                "remaining_tail_rows": int(request.remaining_tail_rows),
                "prefix_fraction": float(request.prefix_fraction),
                "anchor_tvt_input": anchor_tvt,
                "anchor_md": anchor_md,
                "delta_md": md[selected] - anchor_md,
                "row_gr": gr[selected],
                "anchor_x": float(anchor_xyz[0]),
                "anchor_y": float(anchor_xyz[1]),
                "anchor_z": float(anchor_xyz[2]),
                "delta_x": xyz[selected, 0] - anchor_xyz[0],
                "delta_y": xyz[selected, 1] - anchor_xyz[1],
                "delta_z": xyz[selected, 2] - anchor_xyz[2],
            }
        )
        for name, value in summary.items():
            part[name] = value
        part["target_tvt"] = target[selected]
        if not np.all(np.isfinite(part["target_tvt"])):
            raise AssertionError("Materialized targets must be finite")
        rows.append(part)
        request_rows.append(
            {
                "request_id": str(request.request_id),
                "source_well": str(request.source_well),
                "start_kind": str(request.start_kind),
                "start_offset_rows": int(request.start_offset_rows),
                "available_tail_rows": int(request.remaining_tail_rows),
                "materialized_rows": int(len(part)),
                "first_eval_step": int(part["eval_step"].min()),
                "last_eval_step": int(part["eval_step"].max()),
            }
        )
    features = pd.concat(rows, ignore_index=True)
    features = features.sort_values(["request_id", "row_index"]).reset_index(drop=True)
    request_summary = pd.DataFrame(request_rows).sort_values("request_id").reset_index(drop=True)
    forbidden = set(
        nested(config, "model.materialization.forbidden_feature_columns", ["TVT", "target_tvt"])
    )
    candidate_features = set(features.columns) - {
        "request_id",
        "source_well",
        "source_fold",
        "start_kind",
        "target_tvt",
    }
    if forbidden & candidate_features:
        raise AssertionError(
            f"Forbidden feature columns present: {sorted(forbidden & candidate_features)}"
        )
    return features, request_summary


# %% [markdown]
# ## 6. Leakage, distribution, and reproducibility guards


# %%
def build_distribution_report(
    manifest: pd.DataFrame, request_summary: pd.DataFrame
) -> pd.DataFrame:
    joined = manifest.merge(
        request_summary[["request_id", "materialized_rows"]], on="request_id", validate="1:1"
    )
    report = (
        joined.groupby("start_kind", sort=False)
        .agg(
            views=("request_id", "size"),
            wells=("source_well", "nunique"),
            mean_start_offset_rows=("start_offset_rows", "mean"),
            mean_prefix_rows=("prefix_rows", "mean"),
            mean_remaining_tail_rows=("remaining_tail_rows", "mean"),
            total_materialized_rows=("materialized_rows", "sum"),
            current_test_compatible_views=("current_test_compatible", "sum"),
            calibration_backtest_views=("calibration_backtest_compatible", "sum"),
        )
        .reset_index()
    )
    report["view_share"] = report["views"] / report["views"].sum()
    report["materialized_row_share"] = (
        report["total_materialized_rows"] / report["total_materialized_rows"].sum()
    )
    return report


def write_outputs(
    out: Path,
    metadata: pd.DataFrame,
    folds: pd.DataFrame,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    request_summary: pd.DataFrame,
    distribution: pd.DataFrame,
    guards: dict[str, Any],
    elapsed_seconds: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    files = {
        "well_metadata": out / f"{OUTPUT_PREFIX}_well_metadata.csv",
        "fold_manifest": out / f"{OUTPUT_PREFIX}_fold_manifest.csv",
        "view_manifest": out / f"{OUTPUT_PREFIX}_view_manifest.csv",
        "request_summary": out / f"{OUTPUT_PREFIX}_request_summary.csv",
        "distribution": out / f"{OUTPUT_PREFIX}_distribution_report.csv",
        "features": out / f"{OUTPUT_PREFIX}_prefix_features.csv.gz",
        "schema": out / f"{OUTPUT_PREFIX}_feature_schema.csv",
    }
    metadata.to_csv(files["well_metadata"], index=False)
    folds.to_csv(files["fold_manifest"], index=False)
    manifest.to_csv(files["view_manifest"], index=False)
    request_summary.to_csv(files["request_summary"], index=False)
    distribution.to_csv(files["distribution"], index=False)
    features.to_csv(files["features"], index=False, compression={"method": "gzip", "mtime": 0})
    schema = pd.DataFrame(
        {"column": features.columns, "dtype": [str(features[column].dtype) for column in features]}
    )
    schema.to_csv(files["schema"], index=False)
    content_sha = {
        "well_metadata": canonical_csv_sha256(metadata),
        "fold_manifest": canonical_csv_sha256(folds),
        "view_manifest": canonical_csv_sha256(manifest),
        "request_summary": canonical_csv_sha256(request_summary),
        "distribution": canonical_csv_sha256(distribution),
        "feature_content_decompressed": canonical_csv_sha256(features),
        "feature_schema": canonical_csv_sha256(schema),
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "kaggle_cpu_audit_complete",
        "route": nested(config, "experiment.route"),
        "runtime": "kaggle_cpu" if is_kaggle_runtime() else "local_debug",
        "elapsed_seconds": elapsed_seconds,
        "well_count": int(metadata["well_id"].nunique()),
        "view_count": int(len(manifest)),
        "view_counts": {
            str(key): int(value)
            for key, value in manifest["start_kind"].value_counts().sort_index().items()
        },
        "materialized_rows": int(len(features)),
        "feature_columns_including_identifiers_and_target": int(len(features.columns)),
        "guards": guards,
        "content_sha256": content_sha,
        "raw_gzip_file_sha256_non_primary": sha256_file(files["features"]),
        "execution": nested(config, "model.execution"),
        "generated_files": {key: str(value) for key, value in files.items()},
        "adoption_metric": "official_start_oof_not_run",
        "model_training_performed": False,
        "inference_prediction_performed": False,
        "submission_created": False,
    }
    summary_path = out / f"{OUTPUT_PREFIX}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    metrics_path = (
        KAGGLE_WORKING_ROOT if is_kaggle_runtime() else ROOT / "experiments" / EXPERIMENT_NAME
    ) / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 7. Audit orchestration


# %%
def run_audit() -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config()
    train_dir = resolve_split_dir(config, "train")
    pattern = str(nested(config, "data.horizontal_glob", "*__horizontal_well.csv"))
    train_files = sorted(train_dir.glob(pattern), key=well_id_from_path)
    if not train_files:
        raise FileNotFoundError(f"No train files found in {train_dir}")
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": nested(config, "experiment.route"),
                "stage": nested(config, "model.execution.stage"),
                "train_dir": str(train_dir),
                "train_wells": len(train_files),
                "start_offsets_rows": nested(config, "model.view_generation.start_offsets_rows"),
                "active_audits": nested(config, "model.execution.active_audits"),
                "lightgbm_configs": nested(config, "model.execution.lightgbm_configs"),
                "folds_trained": nested(config, "model.execution.folds_trained"),
                "boosters": nested(config, "model.execution.boosters"),
                "parent_control_retrained": nested(
                    config, "model.execution.parent_control_retrained"
                ),
            },
            indent=2,
        )
    )
    metadata, frames = build_well_metadata(train_files, config)
    folds = build_fold_manifest(metadata, config)
    manifest = build_view_manifest(metadata, folds, config)
    guards = assert_manifest_contract(manifest, metadata, folds, config)
    features, request_summary = materialize_prefix_features(manifest, frames, config)
    distribution = build_distribution_report(manifest, request_summary)
    guards["request_materialization_coverage"] = bool(
        request_summary["request_id"].nunique() == manifest["request_id"].nunique()
    )
    guards["feature_targets_finite"] = bool(np.all(np.isfinite(features["target_tvt"])))
    if not all(value for key, value in guards.items() if isinstance(value, bool)):
        raise AssertionError(f"One or more audit guards failed: {guards}")
    summary = write_outputs(
        output_dir(),
        metadata,
        folds,
        manifest,
        features,
        request_summary,
        distribution,
        guards,
        time.perf_counter() - started,
        config,
    )
    print(distribution.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 8. Metrics and generated files
#
# The first execution must be a Kaggle CPU audit. Local execution is blocked unless
# `EXPERIMENT_ALLOW_LOCAL=1` is explicitly set for an approved smoke debug.


# %%
if not is_kaggle_runtime() and os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") != "1":
    raise RuntimeError(
        "This notebook is Kaggle-first. Set EXPERIMENT_ALLOW_LOCAL=1 only for "
        "an approved local smoke debug."
    )

AUDIT_SUMMARY = run_audit()

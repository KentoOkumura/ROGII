# %% [markdown]
# # exp244 offset-specific full-feature cache
#
# Build one of the four early/late pseudo-start caches used by the direct
# integrated augmentation experiment. The active offset is rendered into a
# standalone notebook source before Jupytext conversion.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and input checks
# 3. Official-start metadata and exp218-compatible folds
# 4. Offset-specific replay requests
# 5. Balanced tail-row sampling
# 6. Full exp218 feature regeneration
# 7. Cache contract and generated artifacts

# %%
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import exp239_exp218_pseudotail_augmentation as augmentation
import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp244_bidirectional_prediction_start_pseudotail_augmentation"
ACTIVE_OFFSET_ROWS = 0  # RENDER_ACTIVE_OFFSET
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Configuration and input checks


# %%
def nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def find_file(filename: str) -> Path:
    candidates = [Path.cwd() / filename, Path.cwd() / "inputs" / filename]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")))
    matches = [path for path in candidates if path.exists() and path.stat().st_size]
    if not matches:
        raise FileNotFoundError(filename)
    return matches[0]


def load_config() -> dict[str, Any]:
    path = find_file("config.yaml")
    value = yaml.safe_load(path.read_text()) or {}
    if value.get("experiment", {}).get("name") != EXPERIMENT_NAME:
        raise AssertionError(f"Unexpected config: {path}")
    return value


def find_train_dir(config: dict[str, Any]) -> Path:
    pattern = str(nested(config, "data.horizontal_glob", "*__horizontal_well.csv"))
    if KAGGLE_INPUT_ROOT.exists():
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if candidate.is_dir() and any(candidate.glob(pattern)):
                return candidate
    local = Path(str(nested(config, "data.train_dir", "data/raw/train")))
    if local.is_dir() and any(local.glob(pattern)):
        return local
    raise FileNotFoundError("Could not resolve raw train directory")


def stable_key(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def cache_spec(config: dict[str, Any], offset: int) -> dict[str, Any]:
    specs = list(nested(config, "model.integrated_augmentation.cache_specs", []))
    matches = [dict(item) for item in specs if int(item["offset_rows"]) == offset]
    if len(matches) != 1:
        raise AssertionError(f"Expected one cache spec for offset {offset}: {matches}")
    return matches[0]


# %% [markdown]
# ## 3. Official-start metadata and exp218-compatible folds


# %%
def well_id_from_path(path: Path) -> str:
    suffix = "__horizontal_well.csv"
    if not path.name.endswith(suffix):
        raise ValueError(path.name)
    return path.name[: -len(suffix)]


def load_surfaces(
    train_dir: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    pattern = str(nested(config, "data.horizontal_glob", "*__horizontal_well.csv"))
    target = str(nested(config, "data.target_column", "TVT"))
    input_target = str(nested(config, "data.input_target_column", "TVT_input"))
    columns = [
        str(nested(config, "data.md_column", "MD")),
        *list(nested(config, "data.coordinate_columns", ["X", "Y", "Z"])),
        str(nested(config, "data.gr_column", "GR")),
        target,
        input_target,
    ]
    rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(train_dir.glob(pattern), key=well_id_from_path):
        frame = pd.read_csv(path, usecols=lambda name: name in set(columns))
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")
        well = well_id_from_path(path)
        known = np.flatnonzero(np.isfinite(pd.to_numeric(frame[input_target]).to_numpy(float)))
        if not len(known) or not np.array_equal(known, np.arange(int(known[-1]) + 1)):
            raise AssertionError(f"Non-contiguous TVT_input prefix: {well}")
        official_start = int(known[-1])
        official_rows = int(len(frame) - official_start - 1)
        if official_rows <= 0:
            raise AssertionError(f"No official tail: {well}")
        rows.append(
            {
                "well_id": well,
                "n_rows": int(len(frame)),
                "official_start_index": official_start,
                "official_tail_rows": official_rows,
            }
        )
        frames[well] = frame
    surface = pd.DataFrame(rows).sort_values("well_id").reset_index(drop=True)
    if len(surface) != int(nested(config, "frozen_anchor_parity.expected_wells", 773)):
        raise AssertionError("Raw well count drift")
    if int(surface["official_tail_rows"].sum()) != int(
        nested(config, "model.integrated_augmentation.expected_official_rows")
    ):
        raise AssertionError("Official row count drift")
    return surface, frames


def weighted_groupkfold(weights: np.ndarray, n_folds: int) -> np.ndarray:
    order = np.argsort(np.asarray(weights, dtype=np.int64))[::-1]
    loads = np.zeros(n_folds, dtype=np.int64)
    assignments = np.full(len(weights), -1, dtype=np.int8)
    for index in order:
        fold = int(np.argmin(loads))
        assignments[index] = fold
        loads[fold] += int(weights[index])
    return assignments


# %% [markdown]
# ## 4-5. Offset-specific replay requests and balanced sampling


# %%
def evenly_spaced(indices: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if count <= 0 or not len(values):
        return np.empty(0, dtype=int)
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, count).round().astype(int)
    return values[np.unique(positions)]


def sample_tail_rows(start: int, n_rows: int, config: dict[str, Any]) -> np.ndarray:
    section = "model.integrated_augmentation.sampling"
    maximum = int(nested(config, f"{section}.max_rows_per_view"))
    tail = np.arange(start + 1, n_rows, dtype=int)
    steps = tail - start - 1
    selected: list[int] = []
    for bucket in nested(config, f"{section}.distance_buckets", []):
        mask = steps >= int(bucket["min_step"])
        if bucket.get("max_step") is not None:
            mask &= steps <= int(bucket["max_step"])
        selected.extend(evenly_spaced(tail[mask], int(bucket["quota"])).tolist())
    result = np.asarray(sorted(set(selected)), dtype=int)
    if bool(nested(config, f"{section}.fill_remaining", True)) and len(result) < maximum:
        unused = np.setdiff1d(tail, result, assume_unique=True)
        fill = evenly_spaced(unused, maximum - len(result))
        result = np.asarray(sorted(set(result.tolist() + fill.tolist())), dtype=int)
    return result[:maximum]


def build_offset_requests(
    surface: pd.DataFrame, config: dict[str, Any], offset: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_prefix = int(nested(config, "model.view_generation.min_prefix_rows", 200))
    min_tail = int(nested(config, "model.view_generation.min_remaining_tail_rows", 50))
    surface = surface.copy()
    surface["fold"] = weighted_groupkfold(
        surface["official_tail_rows"].to_numpy(np.int64),
        int(nested(config, "validation.n_folds", 5)),
    )
    replay_rows: list[dict[str, Any]] = []
    sampled_parts: list[pd.DataFrame] = []
    for row in surface.itertuples(index=False):
        start = int(row.official_start_index) + offset
        remaining = int(row.n_rows) - start - 1
        if start + 1 < min_prefix or remaining < min_tail:
            continue
        request_id = stable_key(EXPERIMENT_NAME, row.well_id, row.official_start_index, offset)
        sampled = sample_tail_rows(start, int(row.n_rows), config)
        if not len(sampled):
            raise AssertionError(f"No sampled rows: {row.well_id}")
        replay_rows.append(
            {
                "request_id": request_id,
                "source_well": str(row.well_id),
                "fold": int(row.fold),
                "cutoff_index": start,
            }
        )
        sampled_parts.append(pd.DataFrame({"request_id": request_id, "row_index": sampled}))
    replay = pd.DataFrame(replay_rows).sort_values("request_id").reset_index(drop=True)
    materialized = (
        pd.concat(sampled_parts, ignore_index=True)
        .sort_values(["request_id", "row_index"])
        .reset_index(drop=True)
    )
    if replay["request_id"].duplicated().any() or materialized.duplicated().any():
        raise AssertionError("Duplicate request or sampled row")
    return replay, materialized


def adapter_config(config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    adapted = copy.deepcopy(config)
    section = dict(nested(config, "model.integrated_augmentation"))
    adapted["model"]["exp218_augmentation"] = {
        "expected_feature_count": int(section["expected_feature_count"]),
        "expected_pseudo_rows": int(spec["expected_rows"]),
        "pseudo_request_count": int(spec["expected_requests"]),
        "feature_cache": {
            "batch_request_count": int(section["cache_batch_request_count"]),
            "expected_shards": int(spec["expected_shards"]),
            "preflight": {"enabled": False, "request_count": 25},
        },
        "feature_generation": dict(section["feature_generation"]),
    }
    return adapted


# %% [markdown]
# ## 6-7. Full feature regeneration and cache contract


# %%
def main() -> dict[str, Any]:
    if not KAGGLE_INPUT_ROOT.exists():
        raise RuntimeError("This full feature cache notebook must run on Kaggle")

    config = load_config()
    spec = cache_spec(config, ACTIVE_OFFSET_ROWS)
    train_dir = find_train_dir(config)
    surface, frames = load_surfaces(train_dir, config)
    replay, materialized = build_offset_requests(surface, config, ACTIVE_OFFSET_ROWS)

    if len(replay) != int(spec["expected_requests"]):
        raise AssertionError(f"request count {len(replay)} != {spec['expected_requests']}")
    if len(materialized) != int(spec["expected_rows"]):
        raise AssertionError(f"sampled rows {len(materialized)} != {spec['expected_rows']}")
    if set(replay["source_well"]) - set(surface["well_id"]):
        raise AssertionError("Replay contains an unknown source well")

    prefix = f"{EXPERIMENT_NAME}_{spec['label']}"
    augmentation.OUTPUT_PREFIX = prefix
    adapted_config = adapter_config(config, spec)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "stage": "offset_full_feature_cache",
                "offset_rows": ACTIVE_OFFSET_ROWS,
                "label": spec["label"],
                "requests": len(replay),
                "rows": len(materialized),
                "features": nested(config, "model.integrated_augmentation.expected_feature_count"),
                "lightgbm_configs": 0,
                "folds": 0,
                "boosters": 0,
                "parent_control_retrained": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    cache_summary = augmentation.run_chunked_feature_cache_generation(
        replay=replay,
        materialized=materialized,
        frames=frames,
        raw_train_dir=train_dir,
        config=adapted_config,
        output_dir=KAGGLE_WORKING_ROOT,
    )
    if int(cache_summary["shards"]) != int(spec["expected_shards"]):
        raise AssertionError("Offset cache shard count drift")

    contract = {
        **cache_summary,
        "experiment": EXPERIMENT_NAME,
        "variant": nested(config, "model.integrated_augmentation.variant"),
        "offset_rows": ACTIVE_OFFSET_ROWS,
        "offset_label": str(spec["label"]),
        "start_kind": "early" if ACTIVE_OFFSET_ROWS < 0 else "late",
        "late_train_only": ACTIVE_OFFSET_ROWS > 0,
        "validation_rows": "official_start_only",
        "feature_generation_may_read_tail_tvt": False,
        "forbid_full_prefix_cache_slice": True,
    }
    contract_path = KAGGLE_WORKING_ROOT / f"{prefix}_offset_cache_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)
    return contract


if os.environ.get("EXP244_IMPORT_ONLY", "0") != "1":
    CACHE_CONTRACT = main()

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
# # exp494 exp413 CatBoost/XGBoost/physics bounded stack — train
#
# exp413 の保存済み Stage D LightGBM OOF と final370 特徴面を固定し、
# CatBoost `cb0` と XGBoost Cdeotte v3 を各 outer 5 folds だけ学習する。
# 保存物理候補 `exp226_w500_50_50` を加え、非負・和1・interceptなしの
# OOF-level cross-fit bounded stack を評価する。
#
# この notebook は selector、signed selector、exp413 LightGBM、
# PF/HMM/Beam を再学習しない。Public LB、candidate grid、parameter grid、
# weight grid も使わない。

# %% [markdown]
# ## Contents
#
# 1. Imports and notebook-safe runtime helpers
# 2. Frozen authorization and cost contract
# 3. Stage 0 input, lineage, and final370 preflight
# 4. CatBoost and XGBoost fold training helpers
# 5. Family-level audit helpers
# 6. Bounded cross-fit stack and conditional gate helpers
# 7. Setup and configuration
# 8. Execute Stage 0--5
# 9. Metrics, feature importance, and reproducibility outputs

# %% [markdown]
# ## 1. Imports and notebook-safe runtime helpers

# %%
from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from scipy.optimize import minimize

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    load_stage_d_compact_fold,
    resolve_exp263_cache_root,
    sha256_file,
    sha256_json,
    write_json,
)
from src.likpf_full_replacement import (
    ReplacementCandidateCache,
    build_replacement_clean273_surface,
    downstream_runtime_config,
    resolve_by_patterns,
    verify_replacement_stage_0_root,
    verify_replacement_stage_c_root,
    verify_replacement_stage_s_root,
)
from src.signed_residual_meta import load_signed_compact_fold

EXPERIMENT_NAME = "exp494_exp413_cat_xgb_physics_bounded_stack"
PARENT_EXPERIMENT = "exp413_scale5_likpf_full_replacement_on_exp335"
FAMILY_ORDER = ("lgb", "cat", "xgb", "physics")
PARENT_PREDICTION_COLUMN = (
    "scale5_x1p0_full_replacement__lgb_mean__pred_tvt"
)
PHYSICS_CANDIDATE_ID = "exp226_w500_50_50"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def nested(mapping: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def find_project_root(start: Path | None = None) -> Path:
    current = Path.cwd() if start is None else Path(start)
    for candidate in (current, *current.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return current


ROOT = find_project_root()


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def require_notebook_runtime() -> None:
    if is_kaggle_runtime() or os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "exp494 is Kaggle-first. Local execution requires an explicitly approved "
        "EXPERIMENT_ALLOW_LOCAL=1 smoke run."
    )


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def resolve_config_path() -> Path:
    candidates = (
        [
            Path.cwd() / "config.yaml",
            KAGGLE_WORKING_ROOT / "config.yaml",
            ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
        ]
        if is_kaggle_runtime()
        else [
            ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("exp494 config.yaml")


def search_roots() -> list[Path]:
    return [KAGGLE_INPUT_ROOT, KAGGLE_WORKING_ROOT, Path("/tmp"), ROOT, Path.cwd()]


def resolve_file(spec: Mapping[str, Any], *, sha_key: str = "sha256") -> Path:
    return resolve_by_patterns(
        [str(item) for item in spec["patterns"]],
        search_roots(),
        marker_sha256=str(spec.get(sha_key) or ""),
    )


def resolve_root(patterns: Sequence[str], marker: str) -> Path:
    candidates: list[Path] = []
    for raw in patterns:
        direct = Path(raw)
        if (direct / marker).exists():
            candidates.append(direct)
        if direct.is_absolute():
            continue
        for root in search_roots():
            if root.exists():
                candidates.extend(
                    item for item in root.glob(raw) if (item / marker).exists()
                )
    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise FileNotFoundError(
            f"expected one configured artifact root containing {marker}, found {unique}"
        )
    for root in search_roots():
        if root.exists():
            candidates.extend(path.parent for path in root.rglob(marker))
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"expected one artifact root containing {marker}, found {unique}"
        )
    return unique[0]


def competition_data_root() -> Path:
    local = ROOT / "data" / "raw"
    if not is_kaggle_runtime():
        return local
    project_path = ROOT / "project.yml"
    project = load_yaml(project_path) if project_path.exists() else {}
    slug = str(nested(project, "competition.slug", ""))
    candidates = [KAGGLE_INPUT_ROOT / slug]
    if slug:
        candidates.append(KAGGLE_INPUT_ROOT / "competitions" / slug)
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    for candidate in sorted(KAGGLE_INPUT_ROOT.iterdir()):
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError("competition train/test root was not found")


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(
        actual, dtype=np.float64
    )
    return float(np.sqrt(np.mean(delta * delta)))


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def string_rows_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for column in columns:
        digest.update(str(column).encode())
        digest.update(b"\0")
    for row in frame[list(columns)].astype(str).itertuples(index=False, name=None):
        for value in row:
            digest.update(value.encode())
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def feature_matrix_sha256(values: np.ndarray, features: Sequence[str]) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(features), separators=(",", ":")).encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def release_process_memory() -> None:
    gc.collect()
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


def process_memory_stats() -> dict[str, float]:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return {}
    values: dict[str, float] = {}
    for line in status_path.read_text().splitlines():
        name, separator, raw = line.partition(":")
        if separator and name in {"VmRSS", "VmHWM"}:
            kib = float(raw.strip().split()[0])
            values[f"{name.lower()}_gib"] = round(kib / (1024**2), 3)
    return values


def absolute_partition_evidence(
    root: Path,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(evidence)
    result["partitions"] = [
        {
            **dict(item),
            "path": str(Path(root) / str(item["path"]))
            if not Path(str(item["path"])).is_absolute()
            else str(item["path"]),
        }
        for item in evidence["partitions"]
    ]
    return result


# %% [markdown]
# ## 2. Frozen authorization and cost contract

# %%
EXPECTED_CATBOOST_PARAMS = {
    "iterations": 8000,
    "depth": 7,
    "learning_rate": 0.02,
    "l2_leaf_reg": 2.0,
    "min_data_in_leaf": 15,
    "border_count": 254,
    "loss_function": "RMSE",
    "task_type": "GPU",
    "od_type": "Iter",
    "od_wait": 300,
    "verbose": 0,
    "random_seed": 7,
    "devices": "0",
    "allow_writing_files": False,
}
EXPECTED_XGBOOST_PARAMS = {
    "n_estimators": 450,
    "learning_rate": 0.035,
    "max_depth": 5,
    "min_child_weight": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 4.0,
    "reg_alpha": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_bin": 256,
    "random_state": 42,
    "n_jobs": -1,
    "device": "cuda",
}
EXPECTED_COST = {
    "active_variants": 2,
    "model_configs": 2,
    "outer_folds": 5,
    "new_gpu_models": 10,
    "parent_retraining": 0,
    "selector_retraining": 0,
    "new_physics_runs": 0,
}
EXPECTED_TRAIN_MEMORY = {
    "base_feature_cache": {
        "format": "numpy_npy_float32_memmap",
        "filename": "_runtime_clean273_float32.npy",
        "column_chunk": 32,
        "delete_after_family_train": True,
    },
    "catboost_pool_build": "train_then_release_then_valid_then_release",
}


def validate_static_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    if nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("exp494 experiment name changed")
    if nested(config, "experiment.route") != "ensemble":
        raise ValueError("exp494 route must remain ensemble")
    if nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp494 parent changed")
    if not bool(nested(config, "authorization.implementation_approved", False)):
        raise RuntimeError("exp494 implementation approval is missing")
    if not bool(nested(config, "implementation.enabled", False)):
        raise RuntimeError("exp494 implementation is disabled")
    variants = [str(item) for item in nested(config, "model.active_variants", [])]
    if variants != ["catboost_pixiux_cb0", "xgboost_cdeotte_v3"]:
        raise ValueError("exp494 active variants changed")
    cost = {
        "active_variants": int(nested(config, "model.active_variant_count")),
        "model_configs": int(nested(config, "model.model_config_count")),
        "outer_folds": int(nested(config, "model.folds")),
        "new_gpu_models": int(nested(config, "model.total_new_models")),
        "parent_retraining": int(
            bool(nested(config, "model.parent_lightgbm_retraining"))
        ),
        "selector_retraining": int(
            bool(nested(config, "model.selector_retraining"))
        )
        + int(bool(nested(config, "model.signed_selector_retraining"))),
        "new_physics_runs": int(nested(config, "physics.new_pf_hmm_beam_runs")),
    }
    if cost != EXPECTED_COST:
        raise ValueError(f"exp494 cost contract changed: {cost}")
    if dict(nested(config, "runtime.train_memory", {})) != EXPECTED_TRAIN_MEMORY:
        raise ValueError("exp494 train memory contract changed")
    if dict(nested(config, "model.catboost.params", {})) != EXPECTED_CATBOOST_PARAMS:
        raise ValueError("CatBoost cb0 parameters changed")
    if dict(nested(config, "model.xgboost.params", {})) != EXPECTED_XGBOOST_PARAMS:
        raise ValueError("XGBoost Cdeotte v3 parameters changed")
    family_order = tuple(str(item) for item in nested(config, "validation.family_order"))
    if family_order != FAMILY_ORDER:
        raise ValueError("family order changed")
    physics_formula = {
        str(key): float(value)
        for key, value in dict(nested(config, "physics.formula", {})).items()
    }
    if physics_formula != {
        "exp226_k16": 0.5,
        "likpf_mean": 0.25,
        "exact_hmm": 0.25,
    }:
        raise ValueError("physical candidate formula changed")
    if nested(config, "physics.candidate_id") != PHYSICS_CANDIDATE_ID:
        raise ValueError("physical candidate ID changed")
    if nested(config, "physics.source_experiment") != PARENT_EXPERIMENT:
        raise ValueError("physical candidate must use the exp413 replacement bank")
    if nested(config, "physics.semantic_slot_id") != "likpf_mean":
        raise ValueError("physical candidate semantic slot changed")
    if nested(config, "physics.semantic_value_source") != "likpf_scale_5_x1p0":
        raise ValueError("physical candidate semantic value source changed")
    lower, upper = stack_bounds(config)
    if not np.array_equal(lower, np.asarray([0.60, 0.0, 0.0, 0.0])):
        raise ValueError("stack lower bounds changed")
    if not np.array_equal(upper, np.asarray([1.0, 0.25, 0.20, 0.20])):
        raise ValueError("stack upper bounds changed")
    forbidden_truth_gate = {
        str(item) for item in nested(config, "confidence_gate.forbidden_features", [])
    }
    if not {"well_id", "X", "Y", "Z", "public_lb", "target", "truth", "error"}.issubset(
        forbidden_truth_gate
    ):
        raise ValueError("confidence gate forbidden-feature contract changed")
    return {"cost": cost, "families": list(family_order)}


def require_train_run_authorization(config: Mapping[str, Any]) -> None:
    if not bool(nested(config, "authorization.kaggle_train_run_approved", False)):
        raise RuntimeError("Kaggle train run requires separate user approval")
    for stage in (
        "stage_0_freeze",
        "stage_1_family_train",
        "stage_2_family_audit",
        "stage_3_physics_lock",
        "stage_4_bounded_stack",
        "stage_5_conditional_gate",
    ):
        if not bool(nested(config, f"execution.run_flags.{stage}", False)):
            raise RuntimeError(f"execution.run_flags.{stage} is false")


# %% [markdown]
# ## 3. Stage 0 input, lineage, and final370 preflight

# %%
def load_exp413_parent_oof(
    *,
    base: pd.DataFrame,
    oof_path: Path,
    metrics_path: Path,
    model_manifest_path: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    exp413 = dict(nested(config, "data.exp413", {}))
    paths = {
        "oof": Path(oof_path),
        "metrics": Path(metrics_path),
        "model_manifest": Path(model_manifest_path),
    }
    expected = {
        "oof": str(exp413["stage_d_oof_sha256"]),
        "metrics": str(exp413["stage_d_metrics_sha256"]),
        "model_manifest": str(exp413["stage_d_model_manifest_sha256"]),
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != expected:
        raise ValueError(f"exp413 Stage D SHA mismatch: {actual}")
    metrics = json.loads(paths["metrics"].read_text())
    manifest = json.loads(paths["model_manifest"].read_text())
    if int(metrics.get("model_count", -1)) != int(exp413["expected_lgb_models"]):
        raise ValueError("exp413 metrics model count mismatch")
    if int(manifest.get("model_count", -1)) != int(exp413["expected_lgb_models"]):
        raise ValueError("exp413 manifest model count mismatch")
    groups = dict(manifest.get("feature_groups") or {})
    features = [
        *[str(item) for item in groups.get("clean_base", [])],
        *[str(item) for item in groups.get("nested_compact", [])],
        *[str(item) for item in groups.get("signed_compact", [])],
    ]
    if len(features) != 370 or len(set(features)) != 370:
        raise ValueError("exp413 model manifest final370 schema mismatch")
    required = [
        "id",
        "well",
        "md_since",
        "last_known_tvt",
        "target",
        "outer_fold",
        "actual_tvt",
        PARENT_PREDICTION_COLUMN,
    ]
    parent = pd.read_parquet(paths["oof"], columns=required)
    if (
        len(parent) != int(nested(config, "validation.expected_rows"))
        or parent["well"].nunique() != int(nested(config, "validation.expected_wells"))
        or parent["id"].astype(str).duplicated().any()
    ):
        raise ValueError("exp413 OOF identity or coverage mismatch")
    parent_index = pd.Index(parent["id"].astype(str))
    positions = parent_index.get_indexer(base["id"].astype(str))
    if np.any(positions < 0) or len(np.unique(positions)) != len(base):
        raise ValueError("exp413 OOF does not align one-to-one with final370 base")
    parent = parent.iloc[positions].reset_index(drop=True)
    if not parent["well"].astype(str).reset_index(drop=True).equals(
        base["well"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("exp413 OOF well alignment mismatch")
    truth = (
        base["last_known_tvt"].to_numpy(np.float32)
        + base["target"].to_numpy(np.float32)
    ).astype(np.float32)
    if (
        float(
            np.abs(parent["actual_tvt"].to_numpy(np.float32) - truth).max(
                initial=0.0
            )
        )
        > 1.0e-4
    ):
        raise ValueError("exp413 OOF truth differs from final370 base")
    observed_rmse = rmse(
        truth, parent[PARENT_PREDICTION_COLUMN].to_numpy(np.float32)
    )
    expected_rmse = float(exp413["expected_oof_rmse"])
    if abs(observed_rmse - expected_rmse) > 1.0e-9:
        raise ValueError(
            f"exp413 OOF RMSE mismatch: observed={observed_rmse}, "
            f"expected={expected_rmse}"
        )
    evidence = {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": actual,
        "rows": len(parent),
        "wells": int(parent["well"].nunique()),
        "feature_count": len(features),
        "model_count": int(manifest["model_count"]),
        "rmse": observed_rmse,
        "models_retrained": 0,
        "prediction_column": PARENT_PREDICTION_COLUMN,
    }
    return parent, evidence, features


def load_physics_candidate(
    *,
    base: pd.DataFrame,
    parent_cache_root: Path,
    replacement_root: Path,
    candidate_contract: Mapping[str, Any],
    parent_fold: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    cache = ReplacementCandidateCache(
        parent_cache_root,
        candidate_contract,
        replacement_root,
    )
    specs = {
        str(item["id"]): dict(item)
        for item in candidate_contract["score_candidates"]
    }
    spec = specs[PHYSICS_CANDIDATE_ID]
    if [str(item) for item in spec["parents"]] != [
        "exp226_k16",
        "likpf_mean",
        "exact_hmm",
    ] or not np.array_equal(
        np.asarray(spec["weights"], dtype=np.float64),
        np.asarray([0.50, 0.25, 0.25], dtype=np.float64),
    ):
        raise ValueError("candidate contract physical formula mismatch")
    base_index = pd.Index(base["id"].astype(str))
    prediction = np.full(len(base), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        bundle = cache.load_fold(fold)
        if bundle.candidate_ids.count(PHYSICS_CANDIDATE_ID) != 1:
            raise ValueError("physical candidate must occur exactly once")
        position = bundle.candidate_ids.index(PHYSICS_CANDIDATE_ID)
        target_positions = base_index.get_indexer(bundle.base["id"].astype(str))
        if np.any(target_positions < 0):
            raise ValueError(f"physical candidate fold {fold} has unknown ids")
        if not np.all(parent_fold[target_positions] == fold):
            raise ValueError(f"physical candidate fold {fold} identity mismatch")
        values = bundle.values[:, position].astype(np.float32, copy=False)
        if not bundle.available[:, position].all() or not np.isfinite(values).all():
            raise ValueError(f"physical candidate fold {fold} is incomplete")
        if np.isfinite(prediction[target_positions]).any():
            raise ValueError("physical OOF row assigned twice")
        prediction[target_positions] = values
        fold_rows.append(
            {
                "outer_fold": fold,
                "rows": len(target_positions),
                "wells": int(bundle.base["well"].nunique()),
                "prediction_sha256": array_sha256(values),
            }
        )
        del bundle, values
        gc.collect()
    if not np.isfinite(prediction).all():
        raise ValueError("physical OOF coverage is incomplete")
    truth = (
        base["last_known_tvt"].to_numpy(np.float32)
        + base["target"].to_numpy(np.float32)
    ).astype(np.float32)
    observed = rmse(truth, prediction)
    if abs(observed - float(nested(config, "physics.saved_oof_rmse"))) > 1.0e-5:
        raise ValueError(f"physical OOF RMSE mismatch: {observed}")
    return prediction, {
        "candidate_id": PHYSICS_CANDIDATE_ID,
        "source_experiment": str(nested(config, "physics.source_experiment")),
        "semantic_slot_id": str(nested(config, "physics.semantic_slot_id")),
        "semantic_value_source": str(
            nested(config, "physics.semantic_value_source")
        ),
        "formula": dict(nested(config, "physics.formula")),
        "rows": len(prediction),
        "wells": int(base["well"].nunique()),
        "rmse": observed,
        "prediction_sha256": array_sha256(prediction),
        "folds": fold_rows,
        "new_pf_hmm_beam_runs": 0,
    }


def load_compact_fold(
    *,
    base_index: pd.Index,
    n_rows: int,
    outer_fold: int,
    stage_c_root: Path,
    stage_c_evidence: Mapping[str, Any],
    stage_s_evidence: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    compact_train, compact_valid = load_stage_d_compact_fold(
        stage_c_root=stage_c_root,
        stage_c_evidence=stage_c_evidence,
        downstream_outer_fold=outer_fold,
    )
    signed_train, signed_valid = load_signed_compact_fold(
        stage_s_evidence=stage_s_evidence,
        downstream_outer_fold=outer_fold,
    )
    for role, compact, signed in (
        ("train", compact_train, signed_train),
        ("valid", compact_valid, signed_valid),
    ):
        if not compact[KEY_COLUMNS].reset_index(drop=True).equals(
            signed[KEY_COLUMNS].reset_index(drop=True)
        ):
            raise ValueError(f"compact/signed key mismatch: {role}")
    train_positions = base_index.get_indexer(compact_train["id"].astype(str))
    valid_positions = base_index.get_indexer(compact_valid["id"].astype(str))
    if np.any(train_positions < 0) or np.any(valid_positions < 0):
        raise ValueError("compact ids are absent from final370 base")
    combined = np.concatenate([train_positions, valid_positions])
    if len(combined) != n_rows or len(np.unique(combined)) != n_rows:
        raise ValueError("compact fold does not cover all rows exactly once")
    if np.intersect1d(train_positions, valid_positions).size:
        raise ValueError("compact train/valid rows overlap")
    return (
        compact_train,
        compact_valid,
        signed_train,
        signed_valid,
        train_positions,
        valid_positions,
    )


def write_physical_candidate_oof(
    *,
    parent: pd.DataFrame,
    prediction: np.ndarray,
    output_path: Path,
    row_chunk: int = 250000,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = [
        "id",
        "well",
        "outer_fold",
        "actual_tvt",
        "last_known_tvt",
        "md_since",
    ]
    writer: pq.ParquetWriter | None = None
    try:
        for start in range(0, len(parent), row_chunk):
            stop = min(start + row_chunk, len(parent))
            chunk = parent.iloc[start:stop].loc[:, columns].copy()
            chunk["candidate_id"] = PHYSICS_CANDIDATE_ID
            chunk["pred_tvt"] = np.asarray(
                prediction[start:stop], dtype=np.float32
            )
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="snappy",
                )
            writer.write_table(table)
            del chunk, table
            release_process_memory()
    finally:
        if writer is not None:
            writer.close()


def assemble_matrix(
    *,
    base: pd.DataFrame,
    positions: np.ndarray,
    compact: pd.DataFrame,
    signed: pd.DataFrame,
    base_features: Sequence[str],
    compact_features: Sequence[str],
    signed_features: Sequence[str],
    chunk_columns: int = 32,
) -> np.ndarray:
    feature_count = len(base_features) + len(compact_features) + len(signed_features)
    matrix = np.empty((len(positions), feature_count), dtype=np.float32)
    for start in range(0, len(base_features), chunk_columns):
        stop = min(start + chunk_columns, len(base_features))
        columns = list(base_features[start:stop])
        matrix[:, start:stop] = base.loc[:, columns].iloc[positions].to_numpy(
            np.float32, copy=True
        )
    compact_start = len(base_features)
    signed_start = compact_start + len(compact_features)
    matrix[:, compact_start:signed_start] = compact[list(compact_features)].to_numpy(
        np.float32, copy=False
    )
    matrix[:, signed_start:] = signed[list(signed_features)].to_numpy(
        np.float32, copy=False
    )
    for start in range(0, len(matrix), 32768):
        if not np.isfinite(matrix[start : start + 32768]).all():
            raise ValueError("final370 matrix contains non-finite values")
    return matrix


def close_memmap(array: np.ndarray) -> None:
    mmap_handle = getattr(array, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()


def materialize_base_feature_cache(
    *,
    base: pd.DataFrame,
    base_features: Sequence[str],
    output_path: Path,
    chunk_columns: int = 32,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(base), len(base_features)),
    )
    try:
        for start in range(0, len(base_features), chunk_columns):
            stop = min(start + chunk_columns, len(base_features))
            columns = list(base_features[start:stop])
            block = base.loc[:, columns].to_numpy(np.float32, copy=True)
            values[:, start:stop] = block
            del block
        values.flush()
    finally:
        close_memmap(values)
        del values
        release_process_memory()
    return {
        "path": str(output_path),
        "rows": len(base),
        "features": len(base_features),
        "dtype": "float32",
        "bytes": output_path.stat().st_size,
    }


def assemble_matrix_from_base_cache(
    *,
    base_cache: np.ndarray,
    positions: np.ndarray,
    compact: pd.DataFrame,
    signed: pd.DataFrame,
    compact_features: Sequence[str],
    signed_features: Sequence[str],
    chunk_columns: int = 32,
) -> np.ndarray:
    base_feature_count = int(base_cache.shape[1])
    feature_count = base_feature_count + len(compact_features) + len(signed_features)
    matrix = np.empty((len(positions), feature_count), dtype=np.float32)
    for start in range(0, base_feature_count, chunk_columns):
        stop = min(start + chunk_columns, base_feature_count)
        matrix[:, start:stop] = base_cache[positions, start:stop]
    compact_start = base_feature_count
    signed_start = compact_start + len(compact_features)
    matrix[:, compact_start:signed_start] = compact[list(compact_features)].to_numpy(
        np.float32, copy=False
    )
    matrix[:, signed_start:] = signed[list(signed_features)].to_numpy(
        np.float32, copy=False
    )
    for start in range(0, len(matrix), 32768):
        if not np.isfinite(matrix[start : start + 32768]).all():
            raise ValueError("final370 matrix contains non-finite values")
    return matrix


def stream_matrix_sha256(
    *,
    base: pd.DataFrame,
    positions: np.ndarray,
    compact: pd.DataFrame,
    signed: pd.DataFrame,
    base_features: Sequence[str],
    compact_features: Sequence[str],
    signed_features: Sequence[str],
    row_chunk: int = 32768,
) -> str:
    features = [*base_features, *compact_features, *signed_features]
    digest = hashlib.sha256()
    digest.update(json.dumps(features, separators=(",", ":")).encode())
    digest.update(b"float32")
    digest.update(
        np.asarray([len(positions), len(features)], dtype=np.int64).tobytes()
    )
    for start in range(0, len(positions), row_chunk):
        stop = min(start + row_chunk, len(positions))
        values = assemble_matrix(
            base=base,
            positions=positions[start:stop],
            compact=compact.iloc[start:stop],
            signed=signed.iloc[start:stop],
            base_features=base_features,
            compact_features=compact_features,
            signed_features=signed_features,
        )
        digest.update(memoryview(np.ascontiguousarray(values)).cast("B"))
        del values
    return digest.hexdigest()


def preflight_all_fold_matrices(
    *,
    base: pd.DataFrame,
    base_features: Sequence[str],
    compact_features: Sequence[str],
    signed_features: Sequence[str],
    parent_fold: np.ndarray,
    stage_c_root: Path,
    stage_c_evidence: Mapping[str, Any],
    stage_s_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    base_index = pd.Index(base["id"].astype(str))
    manifest: list[dict[str, Any]] = []
    for outer_fold in range(5):
        (
            compact_train,
            compact_valid,
            signed_train,
            signed_valid,
            train_positions,
            valid_positions,
        ) = load_compact_fold(
            base_index=base_index,
            n_rows=len(base),
            outer_fold=outer_fold,
            stage_c_root=stage_c_root,
            stage_c_evidence=stage_c_evidence,
            stage_s_evidence=stage_s_evidence,
        )
        if not np.all(parent_fold[valid_positions] == outer_fold):
            raise ValueError(f"exp413 OOF valid fold {outer_fold} mismatch")
        if np.any(parent_fold[train_positions] == outer_fold):
            raise ValueError(f"exp413 OOF train fold {outer_fold} leakage")
        train_matrix_sha = stream_matrix_sha256(
            base=base,
            positions=train_positions,
            compact=compact_train,
            signed=signed_train,
            base_features=base_features,
            compact_features=compact_features,
            signed_features=signed_features,
        )
        valid_matrix_sha = stream_matrix_sha256(
            base=base,
            positions=valid_positions,
            compact=compact_valid,
            signed=signed_valid,
            base_features=base_features,
            compact_features=compact_features,
            signed_features=signed_features,
        )
        manifest.append(
            {
                "outer_fold": outer_fold,
                "train_rows": len(train_positions),
                "valid_rows": len(valid_positions),
                "train_wells": int(compact_train["well"].nunique()),
                "valid_wells": int(compact_valid["well"].nunique()),
                "train_row_key_sha256": string_rows_sha256(
                    compact_train, ["id", "well", "outer_fold"]
                ),
                "valid_row_key_sha256": string_rows_sha256(
                    compact_valid, ["id", "well", "outer_fold"]
                ),
                "train_float32_matrix_content_sha256": train_matrix_sha,
                "valid_float32_matrix_content_sha256": valid_matrix_sha,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "stage_0_matrix_preflight",
                    "outer_fold": outer_fold,
                    "completed_folds": outer_fold + 1,
                    "planned_folds": 5,
                    "train_rows": len(train_positions),
                    "valid_rows": len(valid_positions),
                    **process_memory_stats(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del (
            compact_train,
            compact_valid,
            signed_train,
            signed_valid,
            train_positions,
            valid_positions,
        )
        release_process_memory()
    return manifest


def prepare_stage_0(
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    parent_config_path = resolve_file(nested(config, "data.exp413.config"))
    parent_config = load_yaml(parent_config_path)
    if sha256_file(parent_config_path) != str(
        nested(config, "data.exp413.config.sha256")
    ):
        raise ValueError("exp413 config SHA mismatch")
    stage_0_root = resolve_root(
        parent_config["data"]["replacement_stage_0_root_patterns"],
        "replacement_preflight.json",
    )
    stage_c_root = resolve_root(
        parent_config["data"]["replacement_stage_c_root_patterns"],
        "replacement_stage_c_lineage.json",
    )
    stage_s_root = resolve_root(
        parent_config["data"]["replacement_stage_s_root_patterns"],
        "replacement_stage_s_lineage.json",
    )
    stage_d_root = resolve_root(
        nested(config, "data.exp413.stage_d_root_patterns"),
        str(nested(config, "data.exp413.stage_d_oof_filename")),
    )
    stage_0_evidence = verify_replacement_stage_0_root(stage_0_root, parent_config)
    parent_exp264_config_path = resolve_file(
        parent_config["data"]["parent_configs"]["exp264"]
    )
    parent_exp335_config_path = resolve_file(
        parent_config["data"]["parent_configs"]["exp335"]
    )
    parent_exp264_config = load_yaml(parent_exp264_config_path)
    parent_exp335_config = load_yaml(parent_exp335_config_path)
    candidate_contract_path = resolve_file(parent_config["data"]["candidate_contract"])
    candidate_contract = load_yaml(candidate_contract_path)
    runtime_config = downstream_runtime_config(
        parent_config,
        parent_exp335_config,
        stage_c_root,
        stage_s_root,
    )
    stage_c_evidence = verify_replacement_stage_c_root(stage_c_root, runtime_config)
    stage_s_evidence = verify_replacement_stage_s_root(
        stage_s_root,
        runtime_config,
        stage_c_root=stage_c_root,
    )
    stage_c_evidence = absolute_partition_evidence(stage_c_root, stage_c_evidence)
    stage_s_evidence = absolute_partition_evidence(stage_s_root, stage_s_evidence)
    frozen_prediction_path = resolve_file(
        parent_config["data"]["exp404_scale5_train_prediction"],
        sha_key="expected_raw_sha256",
    )
    exp218_source_path = resolve_file(
        {
            "patterns": parent_config["data"]["exp218_source"]["script_patterns"],
            "sha256": parent_config["data"]["exp218_source"]["script_sha256"],
        }
    )
    exp218_config_path = resolve_file(
        {
            "patterns": parent_config["data"]["exp218_source"]["config_patterns"],
            "sha256": parent_config["data"]["exp218_source"]["config_sha256"],
        }
    )
    exp145_source_path = resolve_file(
        {
            "patterns": parent_config["data"]["exp145_source"]["script_patterns"],
            "sha256": parent_config["data"]["exp145_source"]["script_sha256"],
        }
    )
    exp145_config_path = resolve_file(
        {
            "patterns": parent_config["data"]["exp145_source"]["config_patterns"],
            "sha256": parent_config["data"]["exp145_source"]["config_sha256"],
        }
    )
    multiobs_source_path = resolve_file(
        {
            "patterns": parent_config["data"]["exp145_source"][
                "multiobs_script_patterns"
            ],
            "sha256": parent_config["data"]["exp145_source"][
                "multiobs_script_sha256"
            ],
        }
    )
    exp099_source_path = resolve_file(
        {
            "patterns": parent_config["data"]["exp099_train_feature_cache"][
                "patterns"
            ],
            "expected_raw_sha256": parent_config["data"][
                "exp099_train_feature_cache"
            ]["expected_raw_sha256"],
        },
        sha_key="expected_raw_sha256",
    )
    exp111_schema_path = resolve_file(
        {
            "patterns": parent_config["data"]["exp111_saved_models"][
                "schema_patterns"
            ],
            "sha256": parent_config["data"]["exp111_saved_models"]["schema_sha256"],
        }
    )
    exp111_manifest_path = resolve_file(
        {
            "patterns": parent_config["data"]["exp111_saved_models"][
                "manifest_patterns"
            ],
            "sha256": parent_config["data"]["exp111_saved_models"][
                "manifest_sha256"
            ],
        }
    )
    clean_allowlist_path = resolve_file(
        parent_config["data"]["clean_base_allowlist"]
    )
    hidden_assignment_path = resolve_file(
        parent_config["data"]["hidden_like_assignment"]
    )
    raw_train_dir = competition_data_root() / "train"
    base, base_features, base_evidence, _, _ = build_replacement_clean273_surface(
        config=parent_config,
        frozen_prediction_path=frozen_prediction_path,
        exp218_source_path=exp218_source_path,
        exp218_config_path=exp218_config_path,
        exp099_source_path=exp099_source_path,
        exp145_source_path=exp145_source_path,
        exp145_config_path=exp145_config_path,
        multiobs_source_path=multiobs_source_path,
        exp111_schema_path=exp111_schema_path,
        exp111_manifest_path=exp111_manifest_path,
        clean_allowlist_path=clean_allowlist_path,
        raw_train_dir=raw_train_dir,
    )
    required_base = list(
        dict.fromkeys(
            ["id", "well", "target", "last_known_tvt", "md_since", *base_features]
        )
    )
    base = base.loc[:, ~base.columns.duplicated()].loc[:, required_base].copy()
    print(
        json.dumps(
            {
                "stage": "stage_0_clean273_ready",
                "rows": len(base),
                "wells": int(base["well"].nunique()),
                "base_features": len(base_features),
                **process_memory_stats(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    release_process_memory()
    parent, parent_evidence, parent_features = load_exp413_parent_oof(
        base=base,
        oof_path=stage_d_root / str(nested(config, "data.exp413.stage_d_oof_filename")),
        metrics_path=stage_d_root
        / str(nested(config, "data.exp413.stage_d_metrics_filename")),
        model_manifest_path=stage_d_root
        / str(nested(config, "data.exp413.stage_d_model_manifest_filename")),
        config=config,
    )
    compact_features = [str(item) for item in stage_c_evidence["compact_features"]]
    signed_features = [str(item) for item in stage_s_evidence["features"]]
    final_features = [*base_features, *compact_features, *signed_features]
    if final_features != parent_features:
        raise ValueError("reconstructed final370 order differs from exp413 manifest")
    if [len(base_features), len(compact_features), len(signed_features)] != [
        273,
        74,
        23,
    ]:
        raise ValueError("final370 component feature count changed")
    parent_fold = parent["outer_fold"].to_numpy(np.int8)
    if set(np.unique(parent_fold)) != set(range(5)):
        raise ValueError("exp413 outer fold inventory changed")
    parent_cache_root = resolve_exp263_cache_root(
        parent_exp264_config,
        search_roots(),
    )
    physics_prediction, physics_evidence = load_physics_candidate(
        base=base,
        parent_cache_root=parent_cache_root,
        replacement_root=stage_0_root / "replacement_candidate_cache",
        candidate_contract=candidate_contract,
        parent_fold=parent_fold,
        config=config,
    )
    matrix_manifest = preflight_all_fold_matrices(
        base=base,
        base_features=base_features,
        compact_features=compact_features,
        signed_features=signed_features,
        parent_fold=parent_fold,
        stage_c_root=stage_c_root,
        stage_c_evidence=stage_c_evidence,
        stage_s_evidence=stage_s_evidence,
    )
    release_process_memory()
    print(
        json.dumps(
            {
                "stage": "stage_0_post_preflight",
                "phase": "fold_manifests_ready",
                "completed_folds": len(matrix_manifest),
                **process_memory_stats(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    feature_schema = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_NAME,
        "dtype": "float32",
        "features": final_features,
        "feature_count": len(final_features),
        "feature_schema_sha256": sha256_json(final_features),
        "groups": {
            "clean_base": list(base_features),
            "nested_compact": list(compact_features),
            "signed_compact": list(signed_features),
        },
    }
    write_json(output_dir / "final370_feature_schema.json", feature_schema)
    write_json(
        output_dir / "final370_fold_matrix_manifest.json",
        {
            "schema_version": "1.0.0",
            "status": "stage_0_matrix_preflight_complete",
            "folds": matrix_manifest,
        },
    )
    write_json(output_dir / "frozen_exp413_oof_manifest.json", parent_evidence)
    write_physical_candidate_oof(
        parent=parent,
        prediction=physics_prediction,
        output_path=output_dir / "physical_candidate_oof.parquet",
    )
    print(
        json.dumps(
            {
                "stage": "stage_0_post_preflight",
                "phase": "physical_candidate_oof_written",
                "rows": len(parent),
                **process_memory_stats(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    technical_checks = {
        "rows": len(base) == int(nested(config, "validation.expected_rows")),
        "wells": int(base["well"].nunique())
        == int(nested(config, "validation.expected_wells")),
        "outer_folds": len(np.unique(parent_fold))
        == int(nested(config, "validation.outer_folds")),
        "feature_count": len(final_features)
        == int(nested(config, "validation.expected_feature_count")),
        "unique_feature_names": len(set(final_features)) == len(final_features),
        "unique_row_keys": not parent[["id", "well", "outer_fold"]]
        .astype(str)
        .duplicated()
        .any(),
        "stage_0_semantic_manifest_sha": sha256_file(
            stage_0_root / "replacement_semantic_manifest.json"
        )
        == str(nested(config, "data.exp413.stage_0_semantic_manifest_file_sha256")),
        "stage_c_model_manifest_sha": sha256_file(
            stage_c_root / "nested_selector_model_manifest.json"
        )
        == str(nested(config, "data.exp413.stage_c_model_manifest_sha256")),
        "stage_c_compact_manifest_sha": sha256_file(
            stage_c_root / "nested_compact_manifest.json"
        )
        == str(nested(config, "data.exp413.stage_c_compact_manifest_sha256")),
        "stage_s_model_manifest_sha": sha256_file(
            stage_s_root / "signed_selector_model_manifest.json"
        )
        == str(nested(config, "data.exp413.stage_s_model_manifest_sha256")),
        "stage_s_compact_manifest_sha": sha256_file(
            stage_s_root / "signed_compact_manifest.json"
        )
        == str(nested(config, "data.exp413.stage_s_compact_manifest_sha256")),
        "exp413_oof_sha": parent_evidence["sha256"]["oof"]
        == str(nested(config, "data.exp413.stage_d_oof_sha256")),
        "physics_fixed_one_candidate": physics_evidence["candidate_id"]
        == PHYSICS_CANDIDATE_ID,
        "finite_parent_oof": np.isfinite(
            parent[PARENT_PREDICTION_COLUMN].to_numpy(np.float32)
        ).all(),
        "finite_physics_oof": np.isfinite(physics_prediction).all(),
    }
    if not all(technical_checks.values()):
        failed = [key for key, value in technical_checks.items() if not value]
        raise ValueError(f"Stage 0 failed closed before model training: {failed}")
    preflight = {
        "schema_version": "1.0.0",
        "status": "stage_0_complete_models_trained_zero",
        "passed": True,
        "technical_checks": technical_checks,
        "rows": len(base),
        "wells": int(base["well"].nunique()),
        "feature_counts": {
            "clean_base": len(base_features),
            "nested_compact": len(compact_features),
            "signed_compact": len(signed_features),
            "final": len(final_features),
        },
        "feature_schema_sha256": feature_schema["feature_schema_sha256"],
        "matrix_manifest_sha256": sha256_file(
            output_dir / "final370_fold_matrix_manifest.json"
        ),
        "exp413_parent": parent_evidence,
        "physical_candidate": physics_evidence,
        "replacement_stage_0": stage_0_evidence,
        "clean273": base_evidence,
        "models_trained": 0,
    }
    write_json(output_dir / "stage_0_preflight.json", preflight)
    print(
        json.dumps(
            {
                "stage": "stage_0_complete",
                "models_trained": 0,
                **process_memory_stats(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    base_feature_cache = materialize_base_feature_cache(
        base=base,
        base_features=base_features,
        output_path=output_dir
        / str(nested(config, "runtime.train_memory.base_feature_cache.filename")),
        chunk_columns=int(
            nested(config, "runtime.train_memory.base_feature_cache.column_chunk")
        ),
    )
    base_metadata = base[
        ["id", "well", "target", "last_known_tvt", "md_since"]
    ].copy()
    del base
    release_process_memory()
    print(
        json.dumps(
            {
                "stage": "stage_0_complete",
                "phase": "runtime_base_cache_ready",
                "cache_bytes": base_feature_cache["bytes"],
                **process_memory_stats(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "base": base_metadata,
        "base_feature_cache": base_feature_cache,
        "base_features": base_features,
        "compact_features": compact_features,
        "signed_features": signed_features,
        "final_features": final_features,
        "parent": parent,
        "parent_prediction": parent[PARENT_PREDICTION_COLUMN].to_numpy(np.float32),
        "parent_fold": parent_fold,
        "physics_prediction": physics_prediction,
        "stage_c_root": stage_c_root,
        "stage_c_evidence": stage_c_evidence,
        "stage_s_evidence": stage_s_evidence,
        "matrix_manifest": matrix_manifest,
        "preflight": preflight,
        "hidden_assignment_path": hidden_assignment_path,
    }


# %% [markdown]
# ## 4. CatBoost and XGBoost fold training helpers

# %%
def train_family_models(
    *,
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
    started_at: float,
) -> dict[str, Any]:
    from catboost import CatBoostRegressor, Pool
    from catboost import __version__ as catboost_version
    from xgboost import XGBRegressor
    from xgboost import __version__ as xgboost_version

    base = prepared["base"]
    base_feature_cache = prepared["base_feature_cache"]
    base_features = prepared["base_features"]
    compact_features = prepared["compact_features"]
    signed_features = prepared["signed_features"]
    final_features = prepared["final_features"]
    parent_fold = prepared["parent_fold"]
    stage_c_root = prepared["stage_c_root"]
    stage_c_evidence = prepared["stage_c_evidence"]
    stage_s_evidence = prepared["stage_s_evidence"]
    matrix_manifest = {
        int(item["outer_fold"]): dict(item)
        for item in prepared["matrix_manifest"]
    }
    cache_column_chunk = int(
        nested(config, "runtime.train_memory.base_feature_cache.column_chunk")
    )
    base_index = pd.Index(base["id"].astype(str))
    target = base["target"].to_numpy(np.float32)
    anchor = base["last_known_tvt"].to_numpy(np.float32)
    cat_oof = np.full(len(base), np.nan, dtype=np.float32)
    xgb_oof = np.full(len(base), np.nan, dtype=np.float32)
    model_root = output_dir / "models"
    cat_dir = model_root / "catboost"
    xgb_dir = model_root / "xgboost"
    cat_dir.mkdir(parents=True, exist_ok=True)
    xgb_dir.mkdir(parents=True, exist_ok=True)
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    soft_budget = float(nested(config, "runtime.kaggle.train.soft_budget_seconds"))
    for outer_fold in range(5):
        if time.monotonic() - started_at >= soft_budget:
            raise TimeoutError(
                f"exp494 train soft budget reached before outer fold {outer_fold}"
            )
        print(
            json.dumps(
                {
                    "stage": "family_train",
                    "phase": "fold_start",
                    "outer_fold": outer_fold,
                    "completed_new_models": len(model_rows),
                    "planned_new_models": 10,
                    **process_memory_stats(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        (
            compact_train,
            compact_valid,
            signed_train,
            signed_valid,
            train_positions,
            valid_positions,
        ) = load_compact_fold(
            base_index=base_index,
            n_rows=len(base),
            outer_fold=outer_fold,
            stage_c_root=stage_c_root,
            stage_c_evidence=stage_c_evidence,
            stage_s_evidence=stage_s_evidence,
        )
        if not np.all(parent_fold[valid_positions] == outer_fold):
            raise ValueError(f"training fold {outer_fold} parent identity mismatch")
        print(
            json.dumps(
                {
                    "stage": "family_train",
                    "phase": "fold_surface_loaded",
                    "outer_fold": outer_fold,
                    **process_memory_stats(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        base_cache = np.load(
            str(base_feature_cache["path"]),
            mmap_mode="r",
        )
        try:
            if tuple(base_cache.shape) != (len(base), len(base_features)):
                raise ValueError(
                    f"runtime clean273 cache shape changed: {base_cache.shape}"
                )
            x_train = assemble_matrix_from_base_cache(
                base_cache=base_cache,
                positions=train_positions,
                compact=compact_train,
                signed=signed_train,
                compact_features=compact_features,
                signed_features=signed_features,
                chunk_columns=cache_column_chunk,
            )
            print(
                json.dumps(
                    {
                        "stage": "family_train",
                        "phase": "train_matrix_ready",
                        "outer_fold": outer_fold,
                        "train_float32_gib": round(x_train.nbytes / (1024**3), 3),
                        **process_memory_stats(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            x_valid = assemble_matrix_from_base_cache(
                base_cache=base_cache,
                positions=valid_positions,
                compact=compact_valid,
                signed=signed_valid,
                compact_features=compact_features,
                signed_features=signed_features,
                chunk_columns=cache_column_chunk,
            )
        finally:
            close_memmap(base_cache)
        del base_cache
        release_process_memory()
        observed_train_sha = feature_matrix_sha256(x_train, final_features)
        observed_valid_sha = feature_matrix_sha256(x_valid, final_features)
        expected = matrix_manifest[outer_fold]
        if observed_train_sha != expected["train_float32_matrix_content_sha256"]:
            raise ValueError(f"outer fold {outer_fold} train matrix SHA mismatch")
        if observed_valid_sha != expected["valid_float32_matrix_content_sha256"]:
            raise ValueError(f"outer fold {outer_fold} valid matrix SHA mismatch")
        y_train = target[train_positions]
        y_valid = target[valid_positions]
        del (
            compact_train,
            compact_valid,
            signed_train,
            signed_valid,
        )
        release_process_memory()
        cat_train_pool = Pool(
            x_train,
            label=y_train,
            feature_names=final_features,
        )
        del x_train, y_train
        release_process_memory()
        cat_valid_pool = Pool(
            x_valid,
            label=y_valid,
            feature_names=final_features,
        )
        del x_valid, y_valid
        release_process_memory()
        print(
            json.dumps(
                {
                    "stage": "family_train",
                    "phase": "catboost_pool_ready",
                    "outer_fold": outer_fold,
                    "train_rows": len(train_positions),
                    "valid_rows": len(valid_positions),
                    "train_float32_gib": round(
                        expected["train_rows"] * len(final_features) * 4 / (1024**3),
                        3,
                    ),
                    "valid_float32_gib": round(
                        expected["valid_rows"] * len(final_features) * 4 / (1024**3),
                        3,
                    ),
                    "completed_new_models": len(model_rows),
                    "planned_new_models": 10,
                    **process_memory_stats(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        cat_model = CatBoostRegressor(
            **dict(nested(config, "model.catboost.params"))
        )
        cat_model.fit(
            cat_train_pool,
            eval_set=cat_valid_pool,
            early_stopping_rounds=int(
                nested(config, "model.catboost.fit.early_stopping_rounds")
            ),
            use_best_model=bool(nested(config, "model.catboost.fit.use_best_model")),
        )
        cat_residual = np.asarray(
            cat_model.predict(cat_valid_pool), dtype=np.float32
        )
        if not np.isfinite(cat_residual).all():
            raise ValueError(f"outer fold {outer_fold} CatBoost prediction is non-finite")
        cat_oof[valid_positions] = cat_residual
        cat_path = cat_dir / f"catboost_pixiux_cb0__outer{outer_fold}.cbm"
        cat_model.save_model(str(cat_path))
        cat_tree_count = int(cat_model.tree_count_)
        if cat_tree_count <= 0 or cat_tree_count > 8000:
            raise ValueError(f"outer fold {outer_fold} CatBoost tree count changed")
        model_rows.append(
            {
                "family": "cat",
                "model": "catboost_pixiux_cb0",
                "outer_fold": outer_fold,
                "path": str(cat_path.relative_to(output_dir)),
                "sha256": sha256_file(cat_path),
                "tree_count": cat_tree_count,
                "feature_count": len(final_features),
                "train_rows": len(train_positions),
                "valid_rows": len(valid_positions),
                "train_matrix_sha256": observed_train_sha,
                "valid_matrix_sha256": observed_valid_sha,
            }
        )
        importance_rows.extend(
            {
                "family": "cat",
                "outer_fold": outer_fold,
                "feature": feature,
                "importance": float(value),
            }
            for feature, value in zip(
                final_features,
                cat_model.get_feature_importance(),
                strict=True,
            )
        )
        del cat_model, cat_residual, cat_train_pool, cat_valid_pool
        release_process_memory()

        if time.monotonic() - started_at >= soft_budget:
            raise TimeoutError(
                f"exp494 train soft budget reached before XGBoost outer fold {outer_fold}"
            )
        (
            compact_train,
            compact_valid,
            signed_train,
            signed_valid,
            xgb_train_positions,
            xgb_valid_positions,
        ) = load_compact_fold(
            base_index=base_index,
            n_rows=len(base),
            outer_fold=outer_fold,
            stage_c_root=stage_c_root,
            stage_c_evidence=stage_c_evidence,
            stage_s_evidence=stage_s_evidence,
        )
        if not np.array_equal(
            xgb_train_positions, train_positions
        ) or not np.array_equal(xgb_valid_positions, valid_positions):
            raise ValueError(
                f"outer fold {outer_fold} CatBoost/XGBoost row identity mismatch"
            )
        base_cache = np.load(
            str(base_feature_cache["path"]),
            mmap_mode="r",
        )
        try:
            if tuple(base_cache.shape) != (len(base), len(base_features)):
                raise ValueError(
                    f"runtime clean273 cache shape changed: {base_cache.shape}"
                )
            x_train = assemble_matrix_from_base_cache(
                base_cache=base_cache,
                positions=xgb_train_positions,
                compact=compact_train,
                signed=signed_train,
                compact_features=compact_features,
                signed_features=signed_features,
                chunk_columns=cache_column_chunk,
            )
            x_valid = assemble_matrix_from_base_cache(
                base_cache=base_cache,
                positions=xgb_valid_positions,
                compact=compact_valid,
                signed=signed_valid,
                compact_features=compact_features,
                signed_features=signed_features,
                chunk_columns=cache_column_chunk,
            )
        finally:
            close_memmap(base_cache)
        del base_cache
        release_process_memory()
        if feature_matrix_sha256(x_train, final_features) != observed_train_sha:
            raise ValueError(
                f"outer fold {outer_fold} XGBoost train matrix SHA mismatch"
            )
        if feature_matrix_sha256(x_valid, final_features) != observed_valid_sha:
            raise ValueError(
                f"outer fold {outer_fold} XGBoost valid matrix SHA mismatch"
            )
        y_train = target[xgb_train_positions]
        y_valid = target[xgb_valid_positions]
        del (
            compact_train,
            compact_valid,
            signed_train,
            signed_valid,
            xgb_train_positions,
            xgb_valid_positions,
        )
        release_process_memory()
        print(
            json.dumps(
                {
                    "stage": "family_train",
                    "phase": "xgboost_matrix_ready",
                    "outer_fold": outer_fold,
                    "train_rows": len(train_positions),
                    "valid_rows": len(valid_positions),
                    "train_float32_gib": round(x_train.nbytes / (1024**3), 3),
                    "valid_float32_gib": round(x_valid.nbytes / (1024**3), 3),
                    "completed_new_models": len(model_rows),
                    "planned_new_models": 10,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        xgb_model = XGBRegressor(
            **dict(nested(config, "model.xgboost.params"))
        )
        xgb_model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            verbose=100,
        )
        xgb_residual = np.asarray(xgb_model.predict(x_valid), dtype=np.float32)
        if not np.isfinite(xgb_residual).all():
            raise ValueError(f"outer fold {outer_fold} XGBoost prediction is non-finite")
        xgb_oof[valid_positions] = xgb_residual
        xgb_path = xgb_dir / f"xgboost_cdeotte_v3__outer{outer_fold}.json"
        xgb_model.save_model(xgb_path)
        booster = xgb_model.get_booster()
        tree_count = int(booster.num_boosted_rounds())
        if tree_count != int(nested(config, "model.xgboost.params.n_estimators")):
            raise ValueError(f"outer fold {outer_fold} XGBoost tree count changed")
        model_rows.append(
            {
                "family": "xgb",
                "model": "xgboost_cdeotte_v3",
                "outer_fold": outer_fold,
                "path": str(xgb_path.relative_to(output_dir)),
                "sha256": sha256_file(xgb_path),
                "tree_count": tree_count,
                "feature_count": len(final_features),
                "train_rows": len(train_positions),
                "valid_rows": len(valid_positions),
                "train_matrix_sha256": observed_train_sha,
                "valid_matrix_sha256": observed_valid_sha,
            }
        )
        importance_rows.extend(
            {
                "family": "xgb",
                "outer_fold": outer_fold,
                "feature": feature,
                "importance": float(value),
            }
            for feature, value in zip(
                final_features, xgb_model.feature_importances_, strict=True
            )
        )
        cat_tvt = anchor[valid_positions] + cat_oof[valid_positions]
        xgb_tvt = anchor[valid_positions] + xgb_oof[valid_positions]
        truth = anchor[valid_positions] + target[valid_positions]
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "rows": len(valid_positions),
                "cat_rmse": rmse(truth, cat_tvt),
                "xgb_rmse": rmse(truth, xgb_tvt),
                "completed_new_models": len(model_rows),
                "planned_new_models": 10,
            }
        )
        print(json.dumps(fold_rows[-1], sort_keys=True), flush=True)
        del (
            train_positions,
            valid_positions,
            x_train,
            x_valid,
            y_train,
            y_valid,
            xgb_model,
            booster,
            xgb_residual,
            cat_tvt,
            xgb_tvt,
            truth,
        )
        release_process_memory()
    if len(model_rows) != 10:
        raise ValueError(f"expected exactly 10 new models, found {len(model_rows)}")
    if not np.isfinite(cat_oof).all() or not np.isfinite(xgb_oof).all():
        raise ValueError("family OOF coverage is incomplete")
    manifest = {
        "schema_version": "1.0.0",
        "status": "family_training_complete",
        "model_count": len(model_rows),
        "catboost_model_count": sum(item["family"] == "cat" for item in model_rows),
        "xgboost_model_count": sum(item["family"] == "xgb" for item in model_rows),
        "parent_lightgbm_retraining_models": 0,
        "selector_retraining_models": 0,
        "new_pf_hmm_beam_runs": 0,
        "feature_count": len(final_features),
        "feature_schema_sha256": sha256_json(final_features),
        "catboost_version": catboost_version,
        "xgboost_version": xgboost_version,
        "models": model_rows,
    }
    write_json(output_dir / "family_model_manifest.json", manifest)
    pd.DataFrame(importance_rows).to_csv(
        output_dir / "family_feature_importance.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "family_training_fold_metrics.csv", index=False
    )
    return {
        "cat_prediction": (anchor + cat_oof).astype(np.float32),
        "xgb_prediction": (anchor + xgb_oof).astype(np.float32),
        "manifest": manifest,
        "fold_training_metrics": fold_rows,
    }


# %% [markdown]
# ## 5. Family-level audit helpers

# %%
def load_hidden_masks(
    base: pd.DataFrame,
    assignment_path: Path,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if sha256_file(assignment_path) != str(
        nested(config, "data.hidden_like_assignment.sha256")
    ):
        raise ValueError("hidden-like assignment SHA mismatch")
    assignment = pd.read_csv(assignment_path, dtype={"well_id": str})
    assignment_by_well = assignment.set_index("well_id")
    records = base["well"].astype(str).map(assignment_by_well.to_dict("index"))
    return {
        "verification_like_spatial": records.map(
            lambda value: isinstance(value, dict)
            and value.get("verification_like_spatial_role") == "valid"
        ).to_numpy(bool),
        "verification_like_typewell_purged": records.map(
            lambda value: isinstance(value, dict)
            and value.get("verification_like_typewell_purged_role") == "valid"
        ).to_numpy(bool),
    }


def audit_candidate(
    *,
    base: pd.DataFrame,
    truth: np.ndarray,
    parent: np.ndarray,
    prediction: np.ndarray,
    folds: np.ndarray,
    hidden_masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    pooled = {
        "rows": len(truth),
        "parent_rmse": rmse(truth, parent),
        "candidate_rmse": rmse(truth, prediction),
    }
    pooled["delta_rmse_candidate_minus_parent"] = (
        pooled["candidate_rmse"] - pooled["parent_rmse"]
    )
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        mask = folds == fold
        parent_value = rmse(truth[mask], parent[mask])
        candidate_value = rmse(truth[mask], prediction[mask])
        fold_rows.append(
            {
                "outer_fold": fold,
                "rows": int(mask.sum()),
                "parent_rmse": parent_value,
                "candidate_rmse": candidate_value,
                "delta_rmse_candidate_minus_parent": candidate_value - parent_value,
            }
        )
    md_since = base["md_since"].to_numpy(np.float32)
    scope_masks = {
        "near_0_250": md_since < 250.0,
        "mid_250_1000": (md_since >= 250.0) & (md_since < 1000.0),
        "far_1000_plus": md_since >= 1000.0,
    }
    scope_rows: list[dict[str, Any]] = []
    for scope, mask in scope_masks.items():
        parent_value = rmse(truth[mask], parent[mask])
        candidate_value = rmse(truth[mask], prediction[mask])
        scope_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "parent_rmse": parent_value,
                "candidate_rmse": candidate_value,
                "delta_rmse_candidate_minus_parent": candidate_value - parent_value,
            }
        )
    hidden_rows: list[dict[str, Any]] = []
    for scope, mask in hidden_masks.items():
        if not mask.any():
            raise ValueError(f"hidden-like scope is empty: {scope}")
        parent_value = rmse(truth[mask], parent[mask])
        candidate_value = rmse(truth[mask], prediction[mask])
        hidden_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "parent_rmse": parent_value,
                "candidate_rmse": candidate_value,
                "delta_rmse_candidate_minus_parent": candidate_value - parent_value,
            }
        )
    by_well_source = pd.DataFrame(
        {
            "well": base["well"].astype(str),
            "parent_sqerr": np.square(
                truth.astype(np.float64) - parent.astype(np.float64)
            ),
            "candidate_sqerr": np.square(
                truth.astype(np.float64) - prediction.astype(np.float64)
            ),
        }
    )
    by_well = (
        by_well_source.groupby("well", sort=True)
        .agg(
            rows=("well", "size"),
            parent_mse=("parent_sqerr", "mean"),
            candidate_mse=("candidate_sqerr", "mean"),
        )
        .reset_index()
    )
    by_well["parent_rmse"] = np.sqrt(by_well.pop("parent_mse"))
    by_well["candidate_rmse"] = np.sqrt(by_well.pop("candidate_mse"))
    by_well["delta_rmse_candidate_minus_parent"] = (
        by_well["candidate_rmse"] - by_well["parent_rmse"]
    )
    delta = by_well["delta_rmse_candidate_minus_parent"].to_numpy(np.float64)
    tail = {
        "wells": len(by_well),
        "delta_p50": float(np.quantile(delta, 0.50)),
        "delta_p90": float(np.quantile(delta, 0.90)),
        "delta_p95": float(np.quantile(delta, 0.95)),
        "worst_delta": float(delta.max(initial=-np.inf)),
        "best_delta": float(delta.min(initial=np.inf)),
        "improved_wells": int((delta < 0.0).sum()),
        "worsened_wells": int((delta > 0.0).sum()),
    }
    return {
        "pooled": pooled,
        "fold": pd.DataFrame(fold_rows),
        "scope": pd.DataFrame(scope_rows),
        "hidden": pd.DataFrame(hidden_rows),
        "by_well": by_well,
        "tail": tail,
    }


def family_audit(
    *,
    base: pd.DataFrame,
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    folds: np.ndarray,
    hidden_masks: Mapping[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    matrix = np.column_stack([predictions[name] for name in FAMILY_ORDER]).astype(
        np.float64
    )
    residual = matrix - truth.astype(np.float64)[:, None]
    family_metrics: list[dict[str, Any]] = []
    fold_metrics: list[pd.DataFrame] = []
    scope_metrics: list[pd.DataFrame] = []
    hidden_metrics: list[pd.DataFrame] = []
    by_well_metrics: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        audit = audit_candidate(
            base=base,
            truth=truth,
            parent=predictions["lgb"],
            prediction=predictions[family],
            folds=folds,
            hidden_masks=hidden_masks,
        )
        audits[family] = audit
        family_metrics.append({"family": family, **audit["pooled"], **audit["tail"]})
        for key, target in (
            ("fold", fold_metrics),
            ("scope", scope_metrics),
            ("hidden", hidden_metrics),
            ("by_well", by_well_metrics),
        ):
            frame = audit[key].copy()
            frame.insert(0, "family", family)
            target.append(frame)
    correlation = pd.DataFrame(
        np.corrcoef(matrix, rowvar=False), index=FAMILY_ORDER, columns=FAMILY_ORDER
    )
    residual_correlation = pd.DataFrame(
        np.corrcoef(residual, rowvar=False),
        index=FAMILY_ORDER,
        columns=FAMILY_ORDER,
    )
    error_covariance = pd.DataFrame(
        np.cov(residual, rowvar=False),
        index=FAMILY_ORDER,
        columns=FAMILY_ORDER,
    )
    disagreement_rows: list[dict[str, Any]] = []
    for left_position, left in enumerate(FAMILY_ORDER):
        for right in FAMILY_ORDER[left_position + 1 :]:
            values = np.abs(predictions[left] - predictions[right]).astype(np.float64)
            disagreement_rows.append(
                {
                    "left": left,
                    "right": right,
                    "q50": float(np.quantile(values, 0.50)),
                    "q90": float(np.quantile(values, 0.90)),
                    "q95": float(np.quantile(values, 0.95)),
                    "q99": float(np.quantile(values, 0.99)),
                    "max": float(values.max(initial=0.0)),
                }
            )
    pd.DataFrame(family_metrics).to_csv(
        output_dir / "family_metrics.csv", index=False
    )
    pd.concat(fold_metrics, ignore_index=True).to_csv(
        output_dir / "family_fold_metrics.csv", index=False
    )
    pd.concat(scope_metrics, ignore_index=True).to_csv(
        output_dir / "family_scope_metrics.csv", index=False
    )
    pd.concat(hidden_metrics, ignore_index=True).to_csv(
        output_dir / "family_hidden_like_metrics.csv", index=False
    )
    pd.concat(by_well_metrics, ignore_index=True).to_csv(
        output_dir / "family_by_well.csv", index=False
    )
    correlation.to_csv(output_dir / "family_prediction_correlation.csv")
    residual_correlation.to_csv(
        output_dir / "family_residual_correlation.csv"
    )
    error_covariance.to_csv(output_dir / "family_error_covariance.csv")
    pd.DataFrame(disagreement_rows).to_csv(
        output_dir / "family_disagreement_quantiles.csv", index=False
    )
    return {
        "metrics": family_metrics,
        "prediction_correlation": correlation.to_dict(),
        "residual_correlation": residual_correlation.to_dict(),
        "error_covariance": error_covariance.to_dict(),
        "disagreement_quantiles": disagreement_rows,
        "audits": audits,
    }


# %% [markdown]
# ## 6. Bounded cross-fit stack and conditional gate helpers

# %%
def stack_bounds(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bounds = nested(config, "stacking.bounds")
    lower = np.asarray([float(bounds[name][0]) for name in FAMILY_ORDER])
    upper = np.asarray([float(bounds[name][1]) for name in FAMILY_ORDER])
    if lower.sum() > 1.0 or upper.sum() < 1.0:
        raise ValueError("bounded simplex is infeasible")
    return lower, upper


def project_bounded_simplex(
    values: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    target_sum: float = 1.0,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if vector.shape != low.shape or vector.shape != high.shape:
        raise ValueError("bounded projection shape mismatch")
    if np.any(low > high) or low.sum() > target_sum or high.sum() < target_sum:
        raise ValueError("bounded projection is infeasible")
    left = float(np.min(vector - high)) - 1.0
    right = float(np.max(vector - low)) + 1.0
    for _ in range(200):
        midpoint = 0.5 * (left + right)
        projected = np.clip(vector - midpoint, low, high)
        if projected.sum() > target_sum:
            left = midpoint
        else:
            right = midpoint
    projected = np.clip(vector - 0.5 * (left + right), low, high)
    residual = target_sum - float(projected.sum())
    for position in range(len(projected)):
        if residual > 0:
            change = min(residual, high[position] - projected[position])
        else:
            change = max(residual, low[position] - projected[position])
        projected[position] += change
        residual -= change
    if abs(projected.sum() - target_sum) > 1.0e-10:
        raise ValueError("bounded projection sum parity failed")
    return projected


def solve_bounded_stack(
    prediction_matrix: np.ndarray,
    truth: np.ndarray,
    *,
    initial: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(prediction_matrix, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FAMILY_ORDER):
        raise ValueError("stack prediction matrix shape mismatch")
    if len(matrix) != len(target) or not np.isfinite(matrix).all():
        raise ValueError("stack input coverage mismatch")
    gram = matrix.T @ matrix / len(matrix)
    cross = matrix.T @ target / len(matrix)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ gram @ weights - 2.0 * weights @ cross)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * gram @ weights - 2.0 * cross

    result = minimize(
        objective,
        np.asarray(initial, dtype=np.float64),
        jac=gradient,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: float(np.sum(weights) - 1.0),
                "jac": lambda weights: np.ones_like(weights),
            }
        ],
        options={"ftol": 1.0e-12, "maxiter": 1000, "disp": False},
    )
    weights = np.asarray(result.x, dtype=np.float64)
    constraint_residual = abs(float(weights.sum()) - 1.0)
    bound_residual = max(
        float(np.max(np.asarray(lower) - weights, initial=0.0)),
        float(np.max(weights - np.asarray(upper), initial=0.0)),
    )
    direct_rmse = rmse(target, matrix @ weights)
    if not result.success or constraint_residual > 1.0e-8 or bound_residual > 1.0e-8:
        raise ValueError(
            f"SLSQP bounded stack failed: {result.message}, "
            f"constraint={constraint_residual}, bounds={bound_residual}"
        )
    return weights, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "constraint_residual": constraint_residual,
        "bound_residual": bound_residual,
        "recomputed_rmse": direct_rmse,
    }


def crossfit_bounded_stack(
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    matrix = np.column_stack([predictions[name] for name in FAMILY_ORDER]).astype(
        np.float64
    )
    initial = np.asarray(
        [
            float(nested(config, f"stacking.initial_weights.{name}"))
            for name in FAMILY_ORDER
        ],
        dtype=np.float64,
    )
    lower, upper = stack_bounds(config)
    crossfit_prediction = np.full(len(truth), np.nan, dtype=np.float64)
    weight_rows: list[dict[str, Any]] = []
    fold_weights: list[np.ndarray] = []
    for holdout_fold in range(5):
        meta_train = folds != holdout_fold
        meta_valid = folds == holdout_fold
        weights, solver = solve_bounded_stack(
            matrix[meta_train],
            truth[meta_train],
            initial=initial,
            lower=lower,
            upper=upper,
        )
        crossfit_prediction[meta_valid] = matrix[meta_valid] @ weights
        fold_weights.append(weights)
        weight_rows.append(
            {
                "holdout_outer_fold": holdout_fold,
                **{
                    f"weight_{name}": float(weights[position])
                    for position, name in enumerate(FAMILY_ORDER)
                },
                **{f"solver_{key}": value for key, value in solver.items()},
            }
        )
    if not np.isfinite(crossfit_prediction).all():
        raise ValueError("cross-fit stack OOF coverage is incomplete")
    median_weights = np.median(np.vstack(fold_weights), axis=0)
    deployment_weights = project_bounded_simplex(
        median_weights, lower, upper, target_sum=1.0
    )
    deployment_prediction = matrix @ deployment_weights
    return {
        "crossfit_prediction": crossfit_prediction.astype(np.float32),
        "deployment_prediction": deployment_prediction.astype(np.float32),
        "fold_weights": np.vstack(fold_weights),
        "deployment_weights": deployment_weights,
        "weight_rows": weight_rows,
    }


def evaluate_constant_guard(
    audit: Mapping[str, Any],
    technical_checks: Mapping[str, bool],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    guard = dict(nested(config, "guards.constant_stack"))
    pooled = dict(audit["pooled"])
    fold = audit["fold"]
    scope = audit["scope"].set_index("scope")
    hidden = audit["hidden"].set_index("scope")
    tail = dict(audit["tail"])
    gain = float(pooled["parent_rmse"] - pooled["candidate_rmse"])
    nonworse_folds = int(
        (fold["delta_rmse_candidate_minus_parent"] <= 0.0).sum()
    )
    checks = {
        "technical": bool(all(technical_checks.values())),
        "minimum_gain": gain >= float(guard["minimum_gain_vs_exp413_ft"]),
        "minimum_nonworse_folds": nonworse_folds
        >= int(guard["minimum_nonworse_folds"]),
        "near_0_250": float(
            scope.loc["near_0_250", "delta_rmse_candidate_minus_parent"]
        )
        <= float(guard["maximum_near_0_250_delta_rmse_ft"]),
        "far_1000_plus": float(
            scope.loc["far_1000_plus", "delta_rmse_candidate_minus_parent"]
        )
        <= float(guard["maximum_1000_plus_delta_rmse_ft"]),
        "hidden_like_spatial": float(
            hidden.loc[
                "verification_like_spatial",
                "delta_rmse_candidate_minus_parent",
            ]
        )
        <= float(guard["maximum_hidden_like_delta_rmse_ft"]),
        "hidden_like_typewell_purged": float(
            hidden.loc[
                "verification_like_typewell_purged",
                "delta_rmse_candidate_minus_parent",
            ]
        )
        <= float(guard["maximum_hidden_like_delta_rmse_ft"]),
        "by_well_p95": float(tail["delta_p95"])
        <= float(guard["maximum_by_well_p95_delta_rmse_ft"]),
        "worst_well": float(tail["worst_delta"])
        <= float(guard["maximum_worst_well_delta_rmse_ft"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "gain_vs_exp413_ft": gain,
        "nonworse_folds": nonworse_folds,
        "tail": tail,
        "pass_action": guard["pass_action"],
        "fail_action": guard["fail_action"],
    }


def conditional_disagreement_prediction(
    *,
    predictions: Mapping[str, np.ndarray],
    constant_prediction: np.ndarray,
    deployment_weights: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    physics_weight = float(deployment_weights[FAMILY_ORDER.index("physics")])
    ml_weights = deployment_weights[:3]
    ml_weight_sum = float(ml_weights.sum())
    if ml_weight_sum <= 0.0:
        raise ValueError("deployment ML core has zero weight")
    ml_core = (
        np.column_stack([predictions[name] for name in FAMILY_ORDER[:3]])
        @ (ml_weights / ml_weight_sum)
    )
    physics = np.asarray(predictions["physics"], dtype=np.float64)
    disagreement = np.abs(physics - ml_core)
    gated = np.full(len(physics), np.nan, dtype=np.float64)
    quantile_rows: list[dict[str, Any]] = []
    cap = float(
        nested(config, "confidence_gate.maximum_abs_change_vs_constant_stack_ft")
    )
    for holdout_fold in range(5):
        train = folds != holdout_fold
        valid = folds == holdout_fold
        q50, q90 = np.quantile(disagreement[train], [0.50, 0.90])
        if not q90 > q50:
            raise ValueError("disagreement q90 must be greater than q50")
        d = disagreement[valid]
        scale = np.where(
            d <= q50,
            1.0,
            np.where(
                d >= q90,
                0.5,
                1.0 - 0.5 * (d - q50) / (q90 - q50),
            ),
        )
        raw = ml_core[valid] + physics_weight * scale * (
            physics[valid] - ml_core[valid]
        )
        constant = np.asarray(constant_prediction, dtype=np.float64)[valid]
        gated[valid] = constant + np.clip(raw - constant, -cap, cap)
        quantile_rows.append(
            {
                "holdout_outer_fold": holdout_fold,
                "meta_train_rows": int(train.sum()),
                "holdout_rows": int(valid.sum()),
                "q50": float(q50),
                "q90": float(q90),
                "physics_weight": physics_weight,
                "maximum_abs_change_ft": float(
                    np.abs(gated[valid] - constant).max(initial=0.0)
                ),
            }
        )
    if not np.isfinite(gated).all():
        raise ValueError("conditional disagreement OOF coverage is incomplete")
    if (
        float(
            np.abs(gated - np.asarray(constant_prediction, dtype=np.float64)).max(
                initial=0.0
            )
        )
        > cap + 1.0e-10
    ):
        raise ValueError("conditional disagreement cap failed")
    return gated.astype(np.float32), quantile_rows


def evaluate_conditional_guard(
    *,
    parent_audit: Mapping[str, Any],
    constant_prediction: np.ndarray,
    gate_prediction: np.ndarray,
    truth: np.ndarray,
    folds: np.ndarray,
    constant_guard: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    guard = dict(nested(config, "guards.confidence_gate"))
    constant_rmse = rmse(truth, constant_prediction)
    gate_rmse = rmse(truth, gate_prediction)
    fold_deltas: list[float] = []
    for fold in range(5):
        mask = folds == fold
        fold_deltas.append(
            rmse(truth[mask], gate_prediction[mask])
            - rmse(truth[mask], constant_prediction[mask])
        )
    checks = {
        "constant_guard_retained": bool(constant_guard["passed"]),
        "all_parent_scope_and_tail_guards_retained": bool(
            evaluate_constant_guard(
                parent_audit,
                {"conditional_gate_technical": True},
                config,
            )["passed"]
        ),
        "minimum_gain_vs_constant": constant_rmse - gate_rmse
        >= float(guard["minimum_gain_vs_constant_stack_ft"]),
        "minimum_nonworse_folds": int(
            sum(value <= 0.0 for value in fold_deltas)
        )
        >= int(guard["minimum_nonworse_folds"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "constant_rmse": constant_rmse,
        "gate_rmse": gate_rmse,
        "gain_vs_constant_ft": constant_rmse - gate_rmse,
        "fold_deltas_gate_minus_constant": fold_deltas,
        "selected_prediction": "conditional_gate"
        if all(checks.values())
        else "constant_stack",
    }


# %% [markdown]
# ## 7. Setup and configuration

# %%
# The executable entry point below prints the frozen parent, route, model count,
# and readout limitation before loading large inputs.

# %% [markdown]
# ## 8. Execute Stage 0--5

# %%
def run_experiment(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    started_at = time.monotonic()
    static_contract = validate_static_contract(config)
    require_train_run_authorization(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": nested(config, "experiment.route"),
                "parent": nested(config, "lineage.parent"),
                "cost": static_contract["cost"],
                "strict_nested_base_refit": False,
                "readout": "OOF-level cross-fit",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    prepared = prepare_stage_0(config, output_dir)
    runtime_base_cache_path = Path(prepared["base_feature_cache"]["path"])
    try:
        family_train = train_family_models(
            prepared=prepared,
            config=config,
            output_dir=output_dir,
            started_at=started_at,
        )
    finally:
        runtime_base_cache_path.unlink(missing_ok=True)
        release_process_memory()
    base = prepared["base"]
    parent = prepared["parent"]
    truth = parent["actual_tvt"].to_numpy(np.float32)
    folds = prepared["parent_fold"]
    predictions = {
        "lgb": prepared["parent_prediction"],
        "cat": family_train["cat_prediction"],
        "xgb": family_train["xgb_prediction"],
        "physics": prepared["physics_prediction"],
    }
    if any(not np.isfinite(predictions[name]).all() for name in FAMILY_ORDER):
        raise ValueError("one or more family OOF predictions are incomplete")
    hidden_masks = load_hidden_masks(
        base,
        prepared["hidden_assignment_path"],
        config,
    )
    family_readout = family_audit(
        base=base,
        truth=truth,
        predictions=predictions,
        folds=folds,
        hidden_masks=hidden_masks,
        output_dir=output_dir,
    )
    stack = crossfit_bounded_stack(predictions, truth, folds, config)
    constant_audit = audit_candidate(
        base=base,
        truth=truth,
        parent=predictions["lgb"],
        prediction=stack["crossfit_prediction"],
        folds=folds,
        hidden_masks=hidden_masks,
    )
    technical_checks = {
        "stage_0_passed": bool(prepared["preflight"]["passed"]),
        "new_model_count_10": int(family_train["manifest"]["model_count"]) == 10,
        "parent_retraining_zero": int(
            family_train["manifest"]["parent_lightgbm_retraining_models"]
        )
        == 0,
        "selector_retraining_zero": int(
            family_train["manifest"]["selector_retraining_models"]
        )
        == 0,
        "new_physics_runs_zero": int(
            family_train["manifest"]["new_pf_hmm_beam_runs"]
        )
        == 0,
        "finite_crossfit_coverage": np.isfinite(
            stack["crossfit_prediction"]
        ).all(),
        "row_mismatch_zero": len(stack["crossfit_prediction"]) == len(truth),
        "weight_sum_parity": bool(
            np.allclose(stack["fold_weights"].sum(axis=1), 1.0, atol=1.0e-8)
        ),
    }
    constant_guard = evaluate_constant_guard(
        constant_audit,
        technical_checks,
        config,
    )
    gate_prediction: np.ndarray | None = None
    gate_quantiles: list[dict[str, Any]] = []
    gate_audit: dict[str, Any] | None = None
    gate_guard: dict[str, Any] = {
        "evaluated": False,
        "reason": "constant_stack_guard_failed",
        "selected_prediction": "exp413_lgb",
    }
    selected_prediction = predictions["lgb"]
    selected_name = "exp413_lgb"
    if constant_guard["passed"]:
        gate_prediction, gate_quantiles = conditional_disagreement_prediction(
            predictions=predictions,
            constant_prediction=stack["crossfit_prediction"],
            deployment_weights=stack["deployment_weights"],
            folds=folds,
            config=config,
        )
        gate_audit = audit_candidate(
            base=base,
            truth=truth,
            parent=predictions["lgb"],
            prediction=gate_prediction,
            folds=folds,
            hidden_masks=hidden_masks,
        )
        conditional = evaluate_conditional_guard(
            parent_audit=gate_audit,
            constant_prediction=stack["crossfit_prediction"],
            gate_prediction=gate_prediction,
            truth=truth,
            folds=folds,
            constant_guard=constant_guard,
            config=config,
        )
        gate_guard = {"evaluated": True, **conditional}
        if conditional["passed"]:
            selected_prediction = gate_prediction
            selected_name = "conditional_gate"
        else:
            selected_prediction = stack["crossfit_prediction"]
            selected_name = "constant_stack"
    prediction_frame = parent[
        ["id", "well", "outer_fold", "actual_tvt", "last_known_tvt", "md_since"]
    ].copy()
    for family in FAMILY_ORDER:
        prediction_frame[f"{family}_pred_tvt"] = predictions[family]
    prediction_frame["constant_crossfit_pred_tvt"] = stack["crossfit_prediction"]
    prediction_frame["deployment_weight_pred_tvt"] = stack[
        "deployment_prediction"
    ]
    if gate_prediction is not None:
        prediction_frame["conditional_gate_pred_tvt"] = gate_prediction
    prediction_frame["selected_pred_tvt"] = selected_prediction
    prediction_frame.to_parquet(
        output_dir / "exp494_oof_predictions.parquet", index=False
    )
    pd.DataFrame(stack["weight_rows"]).to_csv(
        output_dir / "crossfit_stack_weights.csv", index=False
    )
    write_json(
        output_dir / "deployment_stack_weights.json",
        {
            "family_order": list(FAMILY_ORDER),
            "weights": {
                name: float(stack["deployment_weights"][position])
                for position, name in enumerate(FAMILY_ORDER)
            },
            "aggregation": nested(config, "stacking.deployment_aggregation"),
            "full_oof_refit_performed": False,
        },
    )
    pd.DataFrame(gate_quantiles).to_csv(
        output_dir / "conditional_gate_quantiles.csv", index=False
    )
    constant_audit["fold"].to_csv(
        output_dir / "constant_stack_fold_metrics.csv", index=False
    )
    constant_audit["scope"].to_csv(
        output_dir / "constant_stack_scope_metrics.csv", index=False
    )
    constant_audit["hidden"].to_csv(
        output_dir / "constant_stack_hidden_like_metrics.csv", index=False
    )
    constant_audit["by_well"].to_csv(
        output_dir / "constant_stack_by_well.csv", index=False
    )
    if gate_audit is not None:
        gate_audit["fold"].to_csv(
            output_dir / "conditional_gate_fold_metrics.csv", index=False
        )
        gate_audit["scope"].to_csv(
            output_dir / "conditional_gate_scope_metrics.csv", index=False
        )
        gate_audit["hidden"].to_csv(
            output_dir / "conditional_gate_hidden_like_metrics.csv", index=False
        )
        gate_audit["by_well"].to_csv(
            output_dir / "conditional_gate_by_well.csv", index=False
        )
    elapsed_seconds = time.monotonic() - started_at
    summary = {
        "schema_version": "1.0.0",
        "status": "train_complete_inference_qualified"
        if constant_guard["passed"]
        else "train_complete_guard_failed_closed",
        "experiment": EXPERIMENT_NAME,
        "route": "ensemble",
        "readout": "OOF-level cross-fit; not strict nested stacking",
        "rows": len(base),
        "wells": int(base["well"].nunique()),
        "elapsed_seconds": elapsed_seconds,
        "cost_contract": static_contract["cost"],
        "feature_count": len(prepared["final_features"]),
        "family_metrics": family_readout["metrics"],
        "constant_stack": {
            "pooled": constant_audit["pooled"],
            "tail": constant_audit["tail"],
            "guard": constant_guard,
        },
        "conditional_gate": gate_guard,
        "selected_prediction": selected_name,
        "selected_rmse": rmse(truth, selected_prediction),
        "deployment_weights": {
            name: float(stack["deployment_weights"][position])
            for position, name in enumerate(FAMILY_ORDER)
        },
        "inference_executed": False,
        "submission_generated": False,
        "deterministic_anchor": False,
    }
    write_json(output_dir / "exp494_train_metrics.json", summary)
    artifact_sha = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    reproducibility = {
        "schema_version": "1.0.0",
        "status": summary["status"],
        "deterministic_anchor": False,
        "gpu_bitwise_deterministic_claimed": False,
        "feature_schema_sha256": sha256_file(
            output_dir / "final370_feature_schema.json"
        ),
        "feature_matrix_manifest_sha256": sha256_file(
            output_dir / "final370_fold_matrix_manifest.json"
        ),
        "model_manifest_sha256": sha256_file(
            output_dir / "family_model_manifest.json"
        ),
        "oof_prediction_sha256": sha256_file(
            output_dir / "exp494_oof_predictions.parquet"
        ),
        "blend_weight_sha256": sha256_file(
            output_dir / "deployment_stack_weights.json"
        ),
        "artifact_sha256": artifact_sha,
        "kernel_version": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "unknown"),
        "submission_generated": False,
    }
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    summary["reproducibility_manifest_sha256"] = sha256_file(
        output_dir / "reproducibility_manifest.json"
    )
    write_json(output_dir / "exp494_train_metrics.json", summary)
    return summary


# %% [markdown]
# ## 9. Metrics, feature importance, and reproducibility outputs

# %%
if os.environ.get("EXP494_IMPORT_ONLY", "0") != "1":
    import matplotlib.pyplot as plt

    require_notebook_runtime()
    CONFIG = load_yaml(resolve_config_path())
    OUTPUT_DIR = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if is_kaggle_runtime()
        else ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
    )
    SUMMARY = run_experiment(CONFIG, OUTPUT_DIR)
    display(pd.DataFrame(SUMMARY["family_metrics"]))
    display(pd.DataFrame([SUMMARY["constant_stack"]["guard"]]))
    display(pd.DataFrame([SUMMARY["conditional_gate"]]))
    importance_path = OUTPUT_DIR / "family_feature_importance.csv"
    importance = pd.read_csv(importance_path)
    mean_importance = (
        importance.groupby(["family", "feature"], as_index=False)["importance"]
        .mean()
        .sort_values(["family", "importance"], ascending=[True, False])
    )
    display(mean_importance.groupby("family", group_keys=False).head(30))
    top = (
        mean_importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .nlargest(30, "importance")
        .sort_values("importance")
    )
    ax = top.plot.barh(
        x="feature",
        y="importance",
        figsize=(11, 11),
        legend=False,
        title="exp494 CatBoost/XGBoost mean feature importance",
    )
    ax.set_xlabel("mean family importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "family_feature_importance_top30.png", dpi=140)
    plt.show()
    print(json.dumps(SUMMARY, indent=2, ensure_ascii=False))
    if SUMMARY["constant_stack"]["guard"]["passed"]:
        print(
            "Train gate PASS. Stop here: hidden-safe inference implementation/run and "
            "external submission require their separately authorized same-exp stages."
        )
    else:
        print(
            "Train gate FAIL. Close without candidate, parameter, bound, weight, "
            "threshold, or same-OOF rescue."
        )

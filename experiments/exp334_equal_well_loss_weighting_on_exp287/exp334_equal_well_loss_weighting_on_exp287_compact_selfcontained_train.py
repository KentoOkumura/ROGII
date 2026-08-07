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
# # exp334 equal-well loss weighting on exp287 — train
#
# exp287 の SHA 固定済み 421 特徴・outer 5 folds・LightGBM 3 configs を維持し、
# outer-train 行だけへ `N / (W * n_w)` の sample weight を付ける。
# validation、early stopping、fold/pooled OOF、scope、by-well 評価はすべて非加重とする。
# 保存済み exp287 / corrected exp264 control は比較専用で、control booster は再学習しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Scientific and GPU cost contract
# 3. Frozen input and parent-artifact helpers
# 4. Equal-well training-weight contract
# 5. Parent feature-surface reconstruction
# 6. Fold matrix and saved-cache helpers
# 7. Weighted LightGBM training
# 8. Unweighted OOF and promotion guards
# 9. Preflight or training orchestration
# 10. Metrics, feature importance, and generated artifacts

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

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

from src.candidate_selector_pipeline import (
    build_stage_d_exp218_surface,
    load_stage_d_compact_fold,
    resolve_existing_path,
    resolve_stage_c_artifact_root,
    sha256_file,
    sha256_json,
    verify_stage_c_artifact_root,
    write_json,
)
from src.fold_safe_formation_pipeline import (
    canonical_formation_feature_names,
    load_saved_exp264_control,
    select_unique_columns,
)

EXPERIMENT_NAME = "exp334_equal_well_loss_weighting_on_exp287"
PARENT_EXPERIMENT = "exp287_fold_safe_formation_74_addonly_on_exp264"
VARIANT_NAME = "equal_well_total_train_weight"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
IMPORT_ONLY = os.environ.get("EXP334_IMPORT_ONLY", "0") == "1"


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted_key)
        value = value[part]
    return value


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    candidates.extend(PACKAGE_DIR.rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    matches = sorted({path.resolve() for path in candidates if path.exists()})
    if len(matches) != 1:
        raise FileNotFoundError(f"exp334 config resolution is ambiguous: {matches}")
    return matches[0]


def find_competition_input_root() -> Path:
    preferred = [
        KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction",
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction",
    ]
    candidates = [
        path.resolve()
        for path in preferred
        if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
    ]
    if not candidates and KAGGLE_INPUT_ROOT.exists():
        candidates = [
            path.resolve()
            for path in KAGGLE_INPUT_ROOT.glob("*/*")
            if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
        ]
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "competition input with train/test directories was not unique: "
            f"{candidates}"
        )
    return candidates[0]


def resolve_parent_artifact_root(
    patterns: Sequence[str], search_roots: Sequence[Path]
) -> Path:
    required = {
        "fold_safe_formation_oof_predictions.parquet",
        "model_manifest.json",
        "metrics.json",
        "fold_metrics.csv",
        "by_well_metrics.csv",
        "formation_fold_manifest.json",
        "raw_train_current_test_schema_audit.csv",
    }
    candidates: list[Path] = []
    for raw in patterns:
        path = Path(raw)
        if path.is_dir():
            candidates.append(path.resolve())
    for root in search_roots:
        if not root.exists():
            continue
        for raw in patterns:
            if not Path(raw).is_absolute():
                candidates.extend(path.resolve() for path in root.glob(raw) if path.is_dir())
    checked: list[str] = []
    for candidate in dict.fromkeys(candidates):
        checked.append(str(candidate))
        if all((candidate / name).is_file() for name in required):
            return candidate
    raise FileNotFoundError(
        "complete saved exp287 artifact root not found; checked="
        + json.dumps(checked[:40])
    )


def rmse(actual: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


# %% [markdown]
# ## 2. Scientific and GPU cost contract
#
# 実装済み stage は `preflight_only` と `equal_well_weight_train` の2つだけである。
# 後者は `kaggle_push_approved=true` と `run_train=true` の両方がなければ停止する。
# 実行量は 1 variant × 3 configs × 5 folds = 15 GPU boosters、control 再学習0。

# %%
def validate_scientific_contract(
    config: Mapping[str, Any], *, require_train_approval: bool
) -> dict[str, Any]:
    if nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("experiment name contract changed")
    if nested(config, "experiment.route") != "ml_model":
        raise ValueError("exp334 route must remain ml_model")
    if not bool(nested(config, "execution.implementation_approved")):
        raise RuntimeError("exp334 implementation is not approved")
    stage = str(nested(config, "execution.stage"))
    allowed = {str(value) for value in nested(config, "execution.allowed_stages")}
    if stage not in allowed or allowed != {"preflight_only", "equal_well_weight_train"}:
        raise ValueError(f"unexpected exp334 execution stage contract: {stage}, {allowed}")
    cost = dict(nested(config, "model.execution_count"))
    expected_cost = {
        "active_variants": 1,
        "lightgbm_configs": 3,
        "folds": 5,
        "planned_gpu_boosters": 15,
        "control_retraining_boosters": 0,
    }
    if cost != expected_cost:
        raise ValueError(f"exp334 GPU cost contract changed: {cost} != {expected_cost}")
    if nested(config, "model.source_surface.lightgbm_config_indices") != [0, 1, 2]:
        raise ValueError("exp334 must use LightGBM configs [0, 1, 2]")
    if int(nested(config, "model.source_surface.final_feature_count")) != 421:
        raise ValueError("exp334 final feature count must remain 421")
    if nested(config, "model.train_weight.validation_weight") is not None:
        raise ValueError("validation sample weight is forbidden")
    if (
        nested(config, "model.train_weight.formula")
        != "N_train_rows/(N_train_wells*n_train_rows_for_well)"
    ):
        raise ValueError("equal-well weight formula changed")
    if bool(nested(config, "execution.control_retraining")):
        raise ValueError("saved exp287/exp264 control retraining is forbidden")
    if bool(nested(config, "execution.run_inference")) or bool(
        nested(config, "execution.create_submission")
    ):
        raise ValueError("inference/submission is outside the exp334 train implementation")
    if bool(nested(config, "execution.submit_to_kaggle")):
        raise ValueError("competition submission is forbidden")
    if bool(nested(config, "runtime.kaggle.enable_internet")):
        raise ValueError("Kaggle internet must remain disabled")
    if not bool(nested(config, "runtime.kaggle.enable_gpu")):
        raise ValueError("exp334 train contract requires Kaggle GPU")
    if require_train_approval:
        if stage != "equal_well_weight_train":
            raise RuntimeError("training requires execution.stage=equal_well_weight_train")
        if not bool(nested(config, "execution.kaggle_push_approved")):
            raise RuntimeError("15-booster Kaggle train requires separate push approval")
        if not bool(nested(config, "execution.run_train")):
            raise RuntimeError("15-booster Kaggle train requires execution.run_train=true")
    return {"stage": stage, **expected_cost}


# %% [markdown]
# ## 3. Frozen input and parent-artifact helpers
#
# exp287 OOF、model/metrics/fold/by-well、formation fold manifest、raw schema audit を
# config の SHA と照合する。formation cache 10 partitions は全file SHAとParquet schema/row数を
# booster fit前に検証し、保存済みcacheだけを再利用する。

# %%
def verify_file_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def verify_parent_artifacts(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    data = dict(config["data"])
    fixed_files = {
        "oof": (
            root / "fold_safe_formation_oof_predictions.parquet",
            data["expected_exp287_oof_sha256"],
        ),
        "model_manifest": (
            root / "model_manifest.json",
            data["expected_exp287_model_manifest_sha256"],
        ),
        "metrics": (root / "metrics.json", data["expected_exp287_metrics_sha256"]),
        "fold_metrics": (
            root / "fold_metrics.csv",
            data["expected_exp287_fold_metrics_sha256"],
        ),
        "by_well": (
            root / "by_well_metrics.csv",
            data["expected_exp287_by_well_sha256"],
        ),
        "formation_manifest": (
            root / "formation_fold_manifest.json",
            data["expected_exp287_formation_fold_manifest_sha256"],
        ),
        "raw_schema_audit": (
            root / "raw_train_current_test_schema_audit.csv",
            data["expected_exp287_raw_schema_audit_sha256"],
        ),
    }
    fixed_sha = {
        name: verify_file_sha(path, expected, f"saved exp287 {name}")
        for name, (path, expected) in fixed_files.items()
    }
    model_manifest = json.loads(fixed_files["model_manifest"][0].read_text())
    if int(model_manifest.get("model_count", -1)) != 15:
        raise ValueError("saved exp287 model manifest must contain 15 models")
    if int(model_manifest.get("feature_count", -1)) != 421:
        raise ValueError("saved exp287 model manifest feature count must be 421")
    expected_schema_sha = str(data["expected_exp287_feature_schema_sha256"])
    if str(model_manifest.get("feature_schema_sha256")) != expected_schema_sha:
        raise ValueError("saved exp287 model feature schema SHA mismatch")
    groups = dict(model_manifest.get("feature_groups") or {})
    expected_group_counts = {
        "clean_base": 273,
        "nested_compact": 74,
        "fold_safe_formation": 74,
    }
    actual_group_counts = {name: len(groups.get(name, [])) for name in expected_group_counts}
    if actual_group_counts != expected_group_counts:
        raise ValueError(
            f"saved exp287 feature-group count mismatch: {actual_group_counts}"
        )

    formation_manifest = json.loads(fixed_files["formation_manifest"][0].read_text())
    partitions = list(formation_manifest.get("partitions") or [])
    if len(partitions) != 10 or int(formation_manifest.get("partition_count", -1)) != 10:
        raise ValueError("saved exp287 formation manifest must contain 10 fold-role caches")
    expected_formation_features = [str(value) for value in groups["fold_safe_formation"]]
    lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for item in partitions:
        fold = int(item["downstream_outer_fold"])
        role = str(item["role"])
        key = (fold, role)
        if key in lookup or fold not in range(5) or role not in {"train", "valid"}:
            raise ValueError(f"invalid saved exp287 formation partition key: {key}")
        path = root / str(item["path"])
        verify_file_sha(path, str(item["file_sha256"]), f"formation cache {key}")
        parquet = pq.ParquetFile(path)
        if int(parquet.metadata.num_rows) != int(item["rows"]):
            raise ValueError(f"formation cache row mismatch: {key}")
        expected_columns = ["id", "well", *expected_formation_features]
        if parquet.schema_arrow.names != expected_columns:
            raise ValueError(f"formation cache schema/order mismatch: {key}")
        if str(item["feature_schema_sha256"]) != sha256_json(expected_formation_features):
            raise ValueError(f"formation cache logical schema mismatch: {key}")
        lookup[key] = {**item, "absolute_path": str(path)}
    if set(lookup) != {(fold, role) for fold in range(5) for role in ["train", "valid"]}:
        raise ValueError("saved exp287 formation fold-role inventory is incomplete")
    evidence = {
        "root": str(root),
        "fixed_file_sha256": fixed_sha,
        "formation_partition_count": len(lookup),
        "formation_partition_file_sha256": {
            f"outer{fold}_{role}": str(item["file_sha256"])
            for (fold, role), item in sorted(lookup.items())
        },
        "formation_partition_logical_sha256": {
            f"outer{fold}_{role}": str(item["logical_content_sha256"])
            for (fold, role), item in sorted(lookup.items())
        },
    }
    return model_manifest, lookup, evidence


def load_exp287_control(
    path: Path,
    *,
    expected_sha256: str,
    base_frame: pd.DataFrame,
    expected_rmse: float,
    tolerance: float = 1.0e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    actual_sha = verify_file_sha(path, expected_sha256, "saved exp287 OOF")
    prediction_column = "fold_safe_formation_74_addonly__lgb_mean__pred_tvt"
    columns = [
        "id",
        "well",
        "outer_fold",
        "actual_tvt",
        prediction_column,
    ]
    frame = pd.read_parquet(path, columns=columns)
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("saved exp287 OOF ids are duplicated")
    indexed = frame.set_index(frame["id"].astype(str), drop=False)
    base_ids = base_frame["id"].astype(str)
    if set(indexed.index) != set(base_ids):
        raise ValueError("saved exp287 OOF ids differ from the reconstructed base surface")
    frame = indexed.loc[base_ids].reset_index(drop=True)
    if not frame["well"].astype(str).equals(base_frame["well"].astype(str).reset_index(drop=True)):
        raise ValueError("saved exp287 OOF well alignment mismatch")
    truth = (
        base_frame["last_known_tvt"].to_numpy(np.float32)
        + base_frame["target"].to_numpy(np.float32)
    ).astype(np.float32)
    if float(np.max(np.abs(frame["actual_tvt"].to_numpy(np.float32) - truth))) > 1.0e-4:
        raise ValueError("saved exp287 OOF truth differs from the reconstructed base surface")
    folds = frame["outer_fold"].to_numpy(np.int8)
    if set(np.unique(folds).tolist()) != set(range(5)):
        raise ValueError("saved exp287 OOF fold assignment is incomplete")
    prediction = frame[prediction_column].to_numpy(np.float32)
    score = rmse(truth, prediction)
    if abs(score - float(expected_rmse)) > tolerance:
        raise ValueError(f"saved exp287 OOF RMSE mismatch: {score} != {expected_rmse}")
    return frame, {
        "path": str(path),
        "sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "rmse": score,
        "prediction_column": prediction_column,
        "fold_assignment_sha256": logical_identity_sha256(
            frame["id"], frame["well"], frame["outer_fold"]
        ),
    }


# %% [markdown]
# ## 4. Equal-well training-weight contract
#
# 重み関数の入力は immutable row identity と well だけである。target、予測、誤差、
# formation値、outer-valid rowは関数シグネチャにも渡さない。各foldで有限・正値・平均1、
# well別総重み `N/W`、同一well同一weightを `1e-10` で検証する。

# %%
def logical_identity_sha256(*columns: pd.Series | np.ndarray) -> str:
    frame = pd.DataFrame(
        {f"column_{index}": np.asarray(column) for index, column in enumerate(columns)}
    )
    hashed = pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(
        dtype="<u8", copy=False
    )
    digest = hashlib.sha256()
    digest.update(b"pandas_hash_pandas_object_categorize_v1\0")
    digest.update(hashed.tobytes(order="C"))
    return digest.hexdigest()


def build_equal_well_weights(
    row_ids: pd.Series | Sequence[str],
    wells: pd.Series | Sequence[str],
    *,
    tolerance: float = 1.0e-10,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    ids = pd.Series(row_ids, copy=False).astype(str).reset_index(drop=True)
    groups = pd.Series(wells, copy=False).astype(str).reset_index(drop=True)
    if len(ids) == 0 or len(ids) != len(groups):
        raise ValueError("equal-well weight identity arrays are empty or misaligned")
    if ids.duplicated().any():
        raise ValueError("equal-well outer-train row ids must be unique")
    if groups.eq("").any() or groups.isna().any():
        raise ValueError("equal-well outer-train wells must be nonempty")
    counts = groups.value_counts(sort=False).sort_index()
    n_rows = len(groups)
    n_wells = len(counts)
    expected_total = float(n_rows) / float(n_wells)
    per_well_weight = float(n_rows) / (float(n_wells) * counts.astype(np.float64))
    weights = groups.map(per_well_weight).to_numpy(np.float64)
    if not np.isfinite(weights).all() or not np.all(weights > 0.0):
        raise ValueError("equal-well weights must be finite and positive")
    if abs(float(weights.mean()) - 1.0) > tolerance:
        raise ValueError("equal-well weights do not have mean one")
    check = pd.DataFrame({"well": groups, "weight": weights}).groupby(
        "well", sort=True
    )["weight"].agg(["count", "sum", "min", "max"])
    if float((check["sum"] - expected_total).abs().max()) > tolerance:
        raise ValueError("equal-well total weight differs across wells")
    if float((check["max"] - check["min"]).abs().max()) > tolerance:
        raise ValueError("rows within one well received different weights")
    summary = pd.DataFrame(
        {
            "well": counts.index.astype(str),
            "rows": counts.to_numpy(np.int64),
            "row_weight": per_well_weight.loc[counts.index].to_numpy(np.float64),
        }
    )
    summary["total_weight"] = summary["rows"] * summary["row_weight"]
    digest = hashlib.sha256()
    digest.update(b"exp334_equal_well_weight_float64_v1\0")
    digest.update(
        pd.util.hash_pandas_object(
            pd.DataFrame({"id": ids, "well": groups}), index=False, categorize=True
        )
        .to_numpy(dtype="<u8", copy=False)
        .tobytes(order="C")
    )
    digest.update(weights.astype("<f8", copy=False).tobytes(order="C"))
    evidence = {
        "rows": n_rows,
        "wells": n_wells,
        "minimum_rows_per_well": int(counts.min()),
        "maximum_rows_per_well": int(counts.max()),
        "minimum_row_weight": float(weights.min()),
        "maximum_row_weight": float(weights.max()),
        "mean_row_weight": float(weights.mean()),
        "expected_total_weight_per_well": expected_total,
        "maximum_total_weight_abs_error": float(
            (check["sum"] - expected_total).abs().max()
        ),
        "maximum_within_well_weight_range": float((check["max"] - check["min"]).max()),
        "row_identity_sha256": logical_identity_sha256(ids, groups),
        "row_weight_logical_sha256": digest.hexdigest(),
        "formula": "N_train_rows/(N_train_wells*n_train_rows_for_well)",
        "validation_weight": None,
        "target_or_error_input_used": False,
    }
    return weights, summary, evidence


def precompute_all_fold_weight_contracts(
    *,
    stage_c_root: Path,
    stage_c_evidence: Mapping[str, Any],
    exp287_control: pd.DataFrame,
    tolerance: float,
    output_dir: Path,
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    exp287_by_id = exp287_control.set_index(exp287_control["id"].astype(str), drop=False)
    evidence_by_fold: dict[int, dict[str, Any]] = {}
    summaries: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        compact_train, compact_valid = load_stage_d_compact_fold(
            stage_c_root=stage_c_root,
            stage_c_evidence=stage_c_evidence,
            downstream_outer_fold=fold,
        )
        train_parent_fold = exp287_by_id.loc[
            compact_train["id"].astype(str), "outer_fold"
        ].to_numpy(np.int8)
        valid_parent_fold = exp287_by_id.loc[
            compact_valid["id"].astype(str), "outer_fold"
        ].to_numpy(np.int8)
        if np.any(train_parent_fold == fold) or not np.all(valid_parent_fold == fold):
            raise ValueError(f"Stage C roles differ from saved exp287 outer fold {fold}")
        weights, summary, evidence = build_equal_well_weights(
            compact_train["id"], compact_train["well"], tolerance=tolerance
        )
        if len(weights) != len(compact_train):
            raise AssertionError("outer-train weight length mismatch")
        evidence_by_fold[fold] = evidence
        summary.insert(0, "outer_fold", fold)
        summaries.append(summary)
        fold_rows.append({"outer_fold": fold, **evidence})
        del compact_train, compact_valid, weights, summary
        gc.collect()
    by_well = pd.concat(summaries, ignore_index=True)
    by_fold = pd.DataFrame(fold_rows)
    by_well.to_csv(output_dir / "train_weight_by_well.csv", index=False)
    by_fold.to_csv(output_dir / "train_weight_summary.csv", index=False)
    return evidence_by_fold, by_well, by_fold


# %% [markdown]
# ## 5. Parent feature-surface reconstruction
#
# exp287 と同じ Stage C 25 partitions、exp218 source/config、clean-273 allowlist、raw trainを
# 使って base 273 + compact 74 を再構成する。formation 74 は保存済みexp287 fold cacheを読む。
# model manifest の3 feature groupsと順序が完全一致しなければ停止する。

# %%
def reconstruct_parent_surface(
    *,
    config: Mapping[str, Any],
    search_roots: Sequence[Path],
    raw_train_dir: Path,
    parent_model_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    stage_c_root = resolve_stage_c_artifact_root(config, search_roots)
    stage_c_evidence = verify_stage_c_artifact_root(stage_c_root, config)
    exp218_source_path = resolve_existing_path(
        [str(value) for value in config["data"]["exp218_source_patterns"]], search_roots
    )
    exp218_config_path = resolve_existing_path(
        [str(value) for value in config["data"]["exp218_config_patterns"]], search_roots
    )
    clean_allowlist_path = resolve_existing_path(
        [str(value) for value in config["data"]["clean_273_allowlist_patterns"]],
        search_roots,
    )
    base_frame, base_features, base_evidence, exp218, exp218_config = (
        build_stage_d_exp218_surface(
            exp218_source_path=exp218_source_path,
            exp218_config_path=exp218_config_path,
            base_feature_allowlist_path=clean_allowlist_path,
            raw_train_dir=raw_train_dir,
            config=config,
        )
    )
    base_frame = select_unique_columns(
        base_frame,
        ["id", "well", "target", "last_known_tvt", "md_since", *base_features],
        context="exp334 reconstructed clean exp287 surface",
    )
    groups = dict(parent_model_manifest["feature_groups"])
    compact_features = [str(value) for value in stage_c_evidence["compact_features"]]
    formation_features = canonical_formation_feature_names()
    if base_features != [str(value) for value in groups["clean_base"]]:
        raise ValueError("reconstructed clean feature order differs from saved exp287")
    if compact_features != [str(value) for value in groups["nested_compact"]]:
        raise ValueError("reconstructed compact feature order differs from saved exp287")
    if formation_features != [str(value) for value in groups["fold_safe_formation"]]:
        raise ValueError("canonical formation feature order differs from saved exp287")
    final_features = [*base_features, *compact_features, *formation_features]
    if len(final_features) != 421 or len(set(final_features)) != 421:
        raise ValueError("exp334 final feature surface must contain 421 unique columns")
    if sha256_json(final_features) != str(parent_model_manifest["feature_schema_sha256"]):
        raise ValueError("reconstructed final feature schema SHA differs from saved exp287")
    if len(base_frame) != int(config["validation"]["expected_rows"]):
        raise ValueError("reconstructed exp334 row count mismatch")
    if int(base_frame["well"].nunique()) != int(config["validation"]["expected_wells"]):
        raise ValueError("reconstructed exp334 well count mismatch")
    return {
        "stage_c_root": stage_c_root,
        "stage_c_evidence": stage_c_evidence,
        "base_frame": base_frame,
        "base_features": base_features,
        "compact_features": compact_features,
        "formation_features": formation_features,
        "final_features": final_features,
        "base_evidence": base_evidence,
        "exp218": exp218,
        "exp218_config": exp218_config,
        "source_paths": {
            "exp218_source": str(exp218_source_path),
            "exp218_config": str(exp218_config_path),
            "clean_allowlist": str(clean_allowlist_path),
        },
    }


# %% [markdown]
# ## 6. Fold matrix and saved-cache helpers

# %%
def load_saved_formation_cache(
    item: Mapping[str, Any],
    *,
    compact: pd.DataFrame,
    feature_names: Sequence[str],
    fold: int,
    role: str,
) -> pd.DataFrame:
    path = Path(str(item["absolute_path"]))
    frame = pd.read_parquet(path, columns=["id", "well", *feature_names])
    if len(frame) != int(item["rows"]):
        raise ValueError(f"saved formation cache row mismatch for fold={fold}, role={role}")
    if not frame[["id", "well"]].reset_index(drop=True).equals(
        compact[["id", "well"]].reset_index(drop=True)
    ):
        raise ValueError(
            f"saved formation cache identity mismatch for fold={fold}, role={role}"
        )
    values = frame[list(feature_names)].to_numpy(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"saved formation cache has nonfinite values: fold={fold}, role={role}")
    return frame


def assemble_fold_matrices(
    *,
    base_frame: pd.DataFrame,
    base_features: Sequence[str],
    compact_features: Sequence[str],
    formation_features: Sequence[str],
    compact_train: pd.DataFrame,
    compact_valid: pd.DataFrame,
    formation_train: pd.DataFrame,
    formation_valid: pd.DataFrame,
    chunk_columns: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    base_index = pd.Index(base_frame["id"].astype(str))
    train_indices = base_index.get_indexer(compact_train["id"].astype(str))
    valid_indices = base_index.get_indexer(compact_valid["id"].astype(str))
    if np.any(train_indices < 0) or np.any(valid_indices < 0):
        raise ValueError("Stage C ids are absent from the reconstructed exp287 base surface")
    if np.intersect1d(train_indices, valid_indices).size:
        raise ValueError("outer-train and outer-valid rows overlap")
    final_features = [*base_features, *compact_features, *formation_features]
    x_train_values = np.empty((len(train_indices), len(final_features)), dtype=np.float32)
    x_valid_values = np.empty((len(valid_indices), len(final_features)), dtype=np.float32)
    for start in range(0, len(base_features), chunk_columns):
        stop = min(start + chunk_columns, len(base_features))
        columns = list(base_features[start:stop])
        source = base_frame[columns]
        x_train_values[:, start:stop] = source.iloc[train_indices].to_numpy(
            np.float32, copy=True
        )
        x_valid_values[:, start:stop] = source.iloc[valid_indices].to_numpy(
            np.float32, copy=True
        )
    compact_start = len(base_features)
    formation_start = compact_start + len(compact_features)
    x_train_values[:, compact_start:formation_start] = compact_train[
        list(compact_features)
    ].to_numpy(np.float32, copy=False)
    x_valid_values[:, compact_start:formation_start] = compact_valid[
        list(compact_features)
    ].to_numpy(np.float32, copy=False)
    x_train_values[:, formation_start:] = formation_train[
        list(formation_features)
    ].to_numpy(np.float32, copy=False)
    x_valid_values[:, formation_start:] = formation_valid[
        list(formation_features)
    ].to_numpy(np.float32, copy=False)
    if not np.isfinite(x_train_values).all() or not np.isfinite(x_valid_values).all():
        raise ValueError("exp334 421-feature matrix contains nonfinite values")
    return (
        train_indices,
        valid_indices,
        pd.DataFrame(x_train_values, columns=final_features, copy=False),
        pd.DataFrame(x_valid_values, columns=final_features, copy=False),
    )


# %% [markdown]
# ## 7. Weighted LightGBM training
#
# `sample_weight=train_weights` だけを追加する。`eval_set` には validation weight を渡さず、
# exp287 と同じ非加重 RMSE で early stopping する。

# %%
def validate_lightgbm_params(params_family: Sequence[Mapping[str, Any]]) -> None:
    if len(params_family) != 3:
        raise ValueError("exp334 must train exactly three LightGBM configs")
    for params in params_family:
        required = {
            "device_type": "gpu",
            "gpu_use_dp": True,
            "deterministic": True,
            "force_col_wise": True,
            "n_jobs": 8,
            "num_threads": 8,
        }
        for key, expected in required.items():
            if params.get(key) != expected:
                raise ValueError(f"LightGBM reproducibility parameter changed: {key}")


def train_weighted_models(
    *,
    config: Mapping[str, Any],
    bundle: Mapping[str, Any],
    parent_artifact_root: Path,
    parent_cache_lookup: Mapping[tuple[int, str], Mapping[str, Any]],
    exp287_control: pd.DataFrame,
    expected_weight_evidence: Mapping[int, Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    base_frame = bundle["base_frame"]
    base_features = bundle["base_features"]
    compact_features = bundle["compact_features"]
    formation_features = bundle["formation_features"]
    final_features = bundle["final_features"]
    exp218 = bundle["exp218"]
    exp218_config = bundle["exp218_config"]
    mode_name = str(config["model"]["source_surface"]["mode"])
    mode_config = dict(exp218_config["model"]["training"]["modes"][mode_name])
    params_all = exp218.apply_mode_overrides(
        exp218.exp063_lgb_config_family(fast=False), mode_config
    )
    config_indices = [int(value) for value in config["model"]["source_surface"][
        "lightgbm_config_indices"
    ]]
    params_family = [params_all[index] for index in config_indices]
    validate_lightgbm_params(params_family)
    target = base_frame["target"].to_numpy(np.float32)
    anchor = base_frame["last_known_tvt"].to_numpy(np.float32)
    truth = (anchor + target).astype(np.float32)
    n_rows = len(base_frame)
    oof_by_config = [np.full(n_rows, np.nan, dtype=np.float32) for _ in params_family]
    oof_fold = np.full(n_rows, -1, dtype=np.int8)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_rows: list[dict[str, Any]] = []
    fold_model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    chunk_columns = int(config["model"]["source_surface"]["matrix_copy_chunk_columns"])
    tolerance = float(config["model"]["train_weight"]["invariant_tolerance"])
    parent_by_id = exp287_control.set_index(exp287_control["id"].astype(str), drop=False)
    for fold in range(5):
        compact_train, compact_valid = load_stage_d_compact_fold(
            stage_c_root=bundle["stage_c_root"],
            stage_c_evidence=bundle["stage_c_evidence"],
            downstream_outer_fold=fold,
        )
        formation_train = load_saved_formation_cache(
            parent_cache_lookup[(fold, "train")],
            compact=compact_train,
            feature_names=formation_features,
            fold=fold,
            role="train",
        )
        formation_valid = load_saved_formation_cache(
            parent_cache_lookup[(fold, "valid")],
            compact=compact_valid,
            feature_names=formation_features,
            fold=fold,
            role="valid",
        )
        train_indices, valid_indices, x_train, x_valid = assemble_fold_matrices(
            base_frame=base_frame,
            base_features=base_features,
            compact_features=compact_features,
            formation_features=formation_features,
            compact_train=compact_train,
            compact_valid=compact_valid,
            formation_train=formation_train,
            formation_valid=formation_valid,
            chunk_columns=chunk_columns,
        )
        train_weights, _, weight_evidence = build_equal_well_weights(
            compact_train["id"], compact_train["well"], tolerance=tolerance
        )
        if weight_evidence != dict(expected_weight_evidence[fold]):
            raise ValueError(f"outer fold {fold} weight evidence changed after preflight")
        parent_valid_folds = parent_by_id.loc[
            compact_valid["id"].astype(str), "outer_fold"
        ].to_numpy(np.int8)
        if not np.all(parent_valid_folds == fold):
            raise ValueError(f"outer fold {fold} differs from saved exp287 OOF assignment")
        if np.any(oof_fold[valid_indices] >= 0):
            raise ValueError("exp334 OOF valid rows were assigned twice")
        oof_fold[valid_indices] = np.int8(fold)
        fold_predictions: list[np.ndarray] = []
        for family_position, (config_index, params) in enumerate(
            zip(config_indices, params_family, strict=True)
        ):
            model = LGBMRegressor(**params)
            model.fit(
                x_train,
                target[train_indices],
                sample_weight=train_weights,
                eval_set=[(x_valid, target[valid_indices])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(
                        int(config["model"]["source_surface"]["early_stopping_rounds"]),
                        verbose=False,
                    ),
                    log_evaluation(
                        int(config["model"]["source_surface"]["log_evaluation_period"])
                    ),
                ],
            )
            best_iteration = int(model.best_iteration_ or params["n_estimators"])
            residual = model.predict(x_valid, num_iteration=best_iteration).astype(np.float32)
            oof_by_config[family_position][valid_indices] = residual
            fold_predictions.append(residual)
            model_path = model_dir / f"lgb{config_index}__outer{fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=best_iteration)
            score = rmse(truth[valid_indices], anchor[valid_indices] + residual)
            model_rows.append(
                {
                    "variant": VARIANT_NAME,
                    "model": f"lgb{config_index}",
                    "config_index": config_index,
                    "outer_fold": fold,
                    "feature_count": len(final_features),
                    "best_iteration": best_iteration,
                    "path": str(model_path.relative_to(output_dir)),
                    "sha256": sha256_file(model_path),
                    "params": params,
                    "train_weight_logical_sha256": weight_evidence[
                        "row_weight_logical_sha256"
                    ],
                    "validation_weight": None,
                }
            )
            fold_model_rows.append(
                {
                    "outer_fold": fold,
                    "model": f"lgb{config_index}",
                    "rows": len(valid_indices),
                    "rmse_tvt_unweighted": score,
                    "best_iteration": best_iteration,
                }
            )
            for importance_type in ["gain", "split"]:
                importance = model.booster_.feature_importance(importance_type=importance_type)
                importance_rows.extend(
                    {
                        "outer_fold": fold,
                        "model": f"lgb{config_index}",
                        "importance_type": importance_type,
                        "feature": feature,
                        "feature_group": (
                            "fold_safe_formation"
                            if feature in formation_features
                            else "nested_compact"
                            if feature in compact_features
                            else "clean_base"
                        ),
                        "importance": float(value),
                    }
                    for feature, value in zip(final_features, importance, strict=True)
                )
            print(
                json.dumps(
                    {
                        "outer_fold": fold,
                        "model": f"lgb{config_index}",
                        "rmse_tvt_unweighted": score,
                        "completed_boosters": len(model_rows),
                        "planned_boosters": 15,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del model, residual
            gc.collect()
        mean_residual = np.mean(np.vstack(fold_predictions), axis=0).astype(np.float32)
        fold_model_rows.append(
            {
                "outer_fold": fold,
                "model": "lgb_mean",
                "rows": len(valid_indices),
                "rmse_tvt_unweighted": rmse(
                    truth[valid_indices], anchor[valid_indices] + mean_residual
                ),
                "best_iteration": None,
            }
        )
        del (
            compact_train,
            compact_valid,
            formation_train,
            formation_valid,
            x_train,
            x_valid,
            train_weights,
            fold_predictions,
            mean_residual,
        )
        gc.collect()
    if len(model_rows) != 15 or np.any(oof_fold < 0):
        raise AssertionError("exp334 15-model OOF contract is incomplete")
    if any(not np.isfinite(prediction).all() for prediction in oof_by_config):
        raise AssertionError("exp334 OOF prediction is incomplete")
    mean_residual = np.mean(np.vstack(oof_by_config), axis=0).astype(np.float32)
    mean_prediction = (anchor + mean_residual).astype(np.float32)
    return {
        "oof_by_config": oof_by_config,
        "oof_fold": oof_fold,
        "mean_prediction": mean_prediction,
        "model_rows": model_rows,
        "fold_model_rows": fold_model_rows,
        "importance_rows": importance_rows,
        "params_family": params_family,
        "parent_artifact_root": str(parent_artifact_root),
    }


# %% [markdown]
# ## 8. Unweighted OOF and promotion guards
#
# pooled/fold/scope/by-well は全て通常のrow RMSEで評価する。tail safetyは、
# exp334-vs-exp287 by-well delta p95、exp334-vs-exp264 worst、clean control比の
# `+1/+3/+5 ft` 悪化well数 `135/39/14` を AND で判定する。

# %%
def evaluate_promotion_guards(
    *,
    config: Mapping[str, Any],
    base_frame: pd.DataFrame,
    exp287_control: pd.DataFrame,
    exp264_control: pd.DataFrame,
    oof_fold: np.ndarray,
    new_prediction: np.ndarray,
    hidden_like_assignment_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = (
        base_frame["last_known_tvt"].to_numpy(np.float32)
        + base_frame["target"].to_numpy(np.float32)
    ).astype(np.float32)
    exp287_column = "fold_safe_formation_74_addonly__lgb_mean__pred_tvt"
    exp287 = exp287_control[exp287_column].to_numpy(np.float32)
    exp264 = exp264_control[
        "selector_compact_addonly__lgb_mean__pred_tvt"
    ].to_numpy(np.float32)
    clean = exp264_control["matched_control__lgb_mean__pred_tvt"].to_numpy(np.float32)
    if not np.array_equal(oof_fold, exp287_control["outer_fold"].to_numpy(np.int8)):
        raise ValueError("exp334 OOF fold assignment differs from saved exp287")
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        mask = oof_fold == fold
        exp287_score = rmse(truth[mask], exp287[mask])
        new_score = rmse(truth[mask], new_prediction[mask])
        fold_rows.append(
            {
                "outer_fold": fold,
                "rows": int(mask.sum()),
                "exp264_rmse": rmse(truth[mask], exp264[mask]),
                "exp287_rmse": exp287_score,
                "equal_well_weight_rmse": new_score,
                "delta_rmse_new_minus_exp287": new_score - exp287_score,
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    md_since = base_frame["md_since"].to_numpy(np.float32)
    bucket_masks = {
        "all": np.ones(len(base_frame), dtype=bool),
        "near_0_250": md_since <= 250.0,
        "mid_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "1000_plus": md_since >= 1000.0,
    }
    bucket_rows: list[dict[str, Any]] = []
    for name, mask in bucket_masks.items():
        exp287_score = rmse(truth[mask], exp287[mask])
        new_score = rmse(truth[mask], new_prediction[mask])
        bucket_rows.append(
            {
                "scope": name,
                "rows": int(mask.sum()),
                "exp287_rmse": exp287_score,
                "equal_well_weight_rmse": new_score,
                "delta_rmse_new_minus_exp287": new_score - exp287_score,
            }
        )
    bucket_metrics = pd.DataFrame(bucket_rows)
    assignment_sha = verify_file_sha(
        hidden_like_assignment_path,
        str(config["data"]["hidden_like_assignment_sha256"]),
        "hidden-like assignment",
    )
    assignment = pd.read_csv(hidden_like_assignment_path, dtype={"well_id": str}).set_index(
        "well_id"
    )
    hidden_rows: list[dict[str, Any]] = []
    for name, column in [
        ("hidden_like_spatial", "verification_like_spatial_role"),
        ("hidden_like_typewell_purged", "verification_like_typewell_purged_role"),
    ]:
        mask = base_frame["well"].astype(str).map(assignment[column]).eq("valid").to_numpy()
        exp287_score = rmse(truth[mask], exp287[mask])
        new_score = rmse(truth[mask], new_prediction[mask])
        hidden_rows.append(
            {
                "scope": name,
                "rows": int(mask.sum()),
                "wells": int(base_frame.loc[mask, "well"].nunique()),
                "exp287_rmse": exp287_score,
                "equal_well_weight_rmse": new_score,
                "delta_rmse_new_minus_exp287": new_score - exp287_score,
                "assignment_sha256": assignment_sha,
            }
        )
    hidden_metrics = pd.DataFrame(hidden_rows)
    well_frame = pd.DataFrame(
        {
            "well": base_frame["well"].astype(str),
            "actual_tvt": truth,
            "clean_273_control": clean,
            "exp264_347": exp264,
            "exp287_421": exp287,
            "exp334_equal_well": new_prediction,
        }
    )
    well_rows: list[dict[str, Any]] = []
    for well, group in well_frame.groupby("well", sort=True):
        clean_score = rmse(group["actual_tvt"], group["clean_273_control"])
        exp264_score = rmse(group["actual_tvt"], group["exp264_347"])
        exp287_score = rmse(group["actual_tvt"], group["exp287_421"])
        new_score = rmse(group["actual_tvt"], group["exp334_equal_well"])
        well_rows.append(
            {
                "well": well,
                "rows": len(group),
                "clean_273_rmse": clean_score,
                "exp264_rmse": exp264_score,
                "exp287_rmse": exp287_score,
                "equal_well_weight_rmse": new_score,
                "exp264_minus_clean_delta": exp264_score - clean_score,
                "exp287_minus_clean_delta": exp287_score - clean_score,
                "new_minus_clean_delta": new_score - clean_score,
                "new_minus_exp264_delta": new_score - exp264_score,
                "new_minus_exp287_delta": new_score - exp287_score,
            }
        )
    by_well = pd.DataFrame(well_rows)
    promotion = dict(config["guards"]["promotion"])
    pooled_exp287 = rmse(truth, exp287)
    pooled_new = rmse(truth, new_prediction)
    nonworse_folds = int((fold_metrics["delta_rmse_new_minus_exp287"] <= 0.0).sum())
    scope_delta = pd.concat(
        [
            bucket_metrics.loc[
                bucket_metrics["scope"].isin(["near_0_250", "mid_250_1000", "1000_plus"]),
                ["scope", "delta_rmse_new_minus_exp287"],
            ],
            hidden_metrics[["scope", "delta_rmse_new_minus_exp287"]],
        ],
        ignore_index=True,
    )
    by_well_delta_p95 = float(by_well["new_minus_exp287_delta"].quantile(0.95))
    worst_vs_exp264 = float(by_well["new_minus_exp264_delta"].max())
    threshold_counts: dict[str, dict[str, int | bool]] = {}
    threshold_checks: list[bool] = []
    maximum_counts = dict(promotion["maximum_worsened_well_counts_vs_exp264"])
    for threshold in [float(value) for value in promotion["worsened_well_thresholds_ft"]]:
        key = f"plus_{threshold:g}ft"
        exp264_count = int((by_well["exp264_minus_clean_delta"] > threshold).sum())
        exp287_count = int((by_well["exp287_minus_clean_delta"] > threshold).sum())
        new_count = int((by_well["new_minus_clean_delta"] > threshold).sum())
        maximum = int(maximum_counts[key])
        passed = new_count <= maximum
        threshold_counts[key] = {
            "exp264_vs_clean_count": exp264_count,
            "exp287_vs_clean_count": exp287_count,
            "exp334_vs_clean_count": new_count,
            "maximum_allowed": maximum,
            "passed": passed,
        }
        threshold_checks.append(passed)
    checks = {
        "pooled_rmse_budget_vs_exp287": (pooled_new - pooled_exp287)
        <= float(promotion["maximum_pooled_delta_rmse_vs_exp287"]),
        "minimum_nonworse_folds_vs_exp287": nonworse_folds
        >= int(promotion["minimum_nonworse_folds_vs_exp287"]),
        "all_scope_rmse_budgets_vs_exp287": bool(
            (
                scope_delta["delta_rmse_new_minus_exp287"]
                <= float(promotion["maximum_scope_delta_rmse_vs_exp287"])
            ).all()
        ),
        "by_well_delta_p95_vs_exp287": by_well_delta_p95
        <= float(promotion["maximum_by_well_delta_p95_vs_exp287"]),
        "worst_well_delta_vs_exp264": worst_vs_exp264
        <= float(promotion["maximum_worst_well_delta_rmse_vs_exp264"]),
        "worsened_well_counts_vs_clean_not_above_exp264": all(threshold_checks),
    }
    guard = {
        "metric_weighting": "unweighted_rows",
        "exp287_rmse": pooled_exp287,
        "equal_well_weight_rmse": pooled_new,
        "delta_rmse_new_minus_exp287": pooled_new - pooled_exp287,
        "nonworse_folds_vs_exp287": nonworse_folds,
        "scope_deltas": dict(
            zip(
                scope_delta["scope"],
                scope_delta["delta_rmse_new_minus_exp287"],
                strict=True,
            )
        ),
        "by_well_delta_p95_vs_exp287": by_well_delta_p95,
        "worst_well_delta_rmse_vs_exp264": worst_vs_exp264,
        "worsened_well_threshold_counts": threshold_counts,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    return guard, fold_metrics, bucket_metrics, hidden_metrics, by_well


# %% [markdown]
# ## 9. Preflight or training orchestration

# %%
def run_experiment() -> dict[str, Any]:
    if not KAGGLE_INPUT_ROOT.exists() or not KAGGLE_WORKING_ROOT.exists():
        raise RuntimeError("Kaggle Notebook execution is authoritative for exp334")
    started = time.perf_counter()
    config = read_yaml(find_config_path())
    stage = str(config["execution"]["stage"])
    cost_contract = validate_scientific_contract(
        config, require_train_approval=stage == "equal_well_weight_train"
    )
    competition_root = find_competition_input_root()
    raw_train_dir = competition_root / "train"
    output_dir = KAGGLE_WORKING_ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    search_roots = [KAGGLE_INPUT_ROOT, Path("/tmp"), PACKAGE_DIR]
    parent_artifact_root = resolve_parent_artifact_root(
        [str(value) for value in config["data"]["saved_exp287_artifact_patterns"]],
        search_roots,
    )
    parent_model_manifest, parent_cache_lookup, parent_evidence = verify_parent_artifacts(
        parent_artifact_root, config
    )
    bundle = reconstruct_parent_surface(
        config=config,
        search_roots=search_roots,
        raw_train_dir=raw_train_dir,
        parent_model_manifest=parent_model_manifest,
    )
    base_frame = bundle["base_frame"]
    exp287_control, exp287_evidence = load_exp287_control(
        parent_artifact_root / "fold_safe_formation_oof_predictions.parquet",
        expected_sha256=str(config["data"]["expected_exp287_oof_sha256"]),
        base_frame=base_frame,
        expected_rmse=float(config["validation"]["primary_control"]["rmse"]),
    )
    exp264_path = resolve_existing_path(
        [str(value) for value in config["data"]["saved_exp264_oof_patterns"]],
        search_roots,
    )
    exp264_control, exp264_evidence = load_saved_exp264_control(
        path=exp264_path,
        expected_sha256=str(config["data"]["expected_exp264_oof_sha256"]),
        base_frame=base_frame,
        expected_rmse=float(config["validation"]["clean_tail_control"]["rmse"]),
        tolerance=1.0e-6,
    )
    hidden_like_path = resolve_existing_path(
        [str(value) for value in config["data"]["hidden_like_assignment_patterns"]],
        search_roots,
    )
    verify_file_sha(
        hidden_like_path,
        str(config["data"]["hidden_like_assignment_sha256"]),
        "hidden-like assignment",
    )
    weight_evidence, weight_by_well, weight_by_fold = precompute_all_fold_weight_contracts(
        stage_c_root=bundle["stage_c_root"],
        stage_c_evidence=bundle["stage_c_evidence"],
        exp287_control=exp287_control,
        tolerance=float(config["model"]["train_weight"]["invariant_tolerance"]),
        output_dir=output_dir,
    )
    preflight = {
        "schema_version": "1.0.0",
        "status": "preflight_passed_zero_boosters",
        "experiment": EXPERIMENT_NAME,
        "route": config["experiment"]["route"],
        "cost_contract": cost_contract,
        "parent_artifacts": parent_evidence,
        "exp287_control": exp287_evidence,
        "exp264_control": exp264_evidence,
        "stage_c": bundle["stage_c_evidence"],
        "clean_base": bundle["base_evidence"],
        "feature_count": len(bundle["final_features"]),
        "feature_schema_sha256": sha256_json(bundle["final_features"]),
        "weight_contracts": weight_evidence,
        "weight_summary_sha256": sha256_file(output_dir / "train_weight_summary.csv"),
        "weight_by_well_sha256": sha256_file(output_dir / "train_weight_by_well.csv"),
        "hidden_like_assignment": {
            "path": str(hidden_like_path),
            "sha256": sha256_file(hidden_like_path),
        },
        "rows": len(base_frame),
        "wells": int(base_frame["well"].nunique()),
        "boosters_trained": 0,
        "prediction_or_submission_generated": False,
    }
    write_json(output_dir / "preflight_manifest.json", preflight)
    if stage == "preflight_only":
        print("exp334 preflight passed with 0 boosters; no prediction/submission generated")
        return preflight

    trained = train_weighted_models(
        config=config,
        bundle=bundle,
        parent_artifact_root=parent_artifact_root,
        parent_cache_lookup=parent_cache_lookup,
        exp287_control=exp287_control,
        expected_weight_evidence=weight_evidence,
        output_dir=output_dir,
    )
    guard, fold_metrics, bucket_metrics, hidden_metrics, by_well = (
        evaluate_promotion_guards(
            config=config,
            base_frame=base_frame,
            exp287_control=exp287_control,
            exp264_control=exp264_control,
            oof_fold=trained["oof_fold"],
            new_prediction=trained["mean_prediction"],
            hidden_like_assignment_path=hidden_like_path,
        )
    )
    anchor = base_frame["last_known_tvt"].to_numpy(np.float32)
    truth = (anchor + base_frame["target"].to_numpy(np.float32)).astype(np.float32)
    prediction_frame = base_frame[
        ["id", "well", "md_since", "last_known_tvt", "target"]
    ].copy()
    prediction_frame["outer_fold"] = trained["oof_fold"]
    prediction_frame["actual_tvt"] = truth
    for config_index, residual in zip(
        config["model"]["source_surface"]["lightgbm_config_indices"],
        trained["oof_by_config"],
        strict=True,
    ):
        prediction_frame[f"{VARIANT_NAME}__lgb{config_index}__pred_tvt"] = (
            anchor + residual
        ).astype(np.float32)
    prediction_frame[f"{VARIANT_NAME}__lgb_mean__pred_tvt"] = trained[
        "mean_prediction"
    ]
    paths = {
        "oof": output_dir / "equal_well_weight_oof_predictions.parquet",
        "fold_metrics": output_dir / "fold_metrics.csv",
        "scope_metrics": output_dir / "scope_metrics.csv",
        "hidden_metrics": output_dir / "hidden_like_metrics.csv",
        "by_well": output_dir / "by_well_metrics.csv",
        "importance": output_dir / "feature_importance.csv",
        "model_manifest": output_dir / "model_manifest.json",
        "metrics": output_dir / "metrics.json",
        "weight_summary": output_dir / "train_weight_summary.csv",
        "weight_by_well": output_dir / "train_weight_by_well.csv",
        "preflight_manifest": output_dir / "preflight_manifest.json",
    }
    prediction_frame.to_parquet(paths["oof"], index=False)
    pd.DataFrame(trained["fold_model_rows"]).merge(
        fold_metrics, on="outer_fold", how="left"
    ).to_csv(paths["fold_metrics"], index=False)
    bucket_metrics.to_csv(paths["scope_metrics"], index=False)
    hidden_metrics.to_csv(paths["hidden_metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    pd.DataFrame(trained["importance_rows"]).to_csv(paths["importance"], index=False)
    model_manifest = {
        "schema_version": "1.0.0",
        "status": "equal_well_weight_15_gpu_boosters_completed",
        "cost_contract": cost_contract,
        "model_count": len(trained["model_rows"]),
        "models": trained["model_rows"],
        "feature_count": len(bundle["final_features"]),
        "feature_schema_sha256": sha256_json(bundle["final_features"]),
        "feature_groups": {
            "clean_base": bundle["base_features"],
            "nested_compact": bundle["compact_features"],
            "fold_safe_formation": bundle["formation_features"],
        },
        "train_weight_formula": config["model"]["train_weight"]["formula"],
        "validation_weight": None,
        "parent_control_retrained": False,
    }
    write_json(paths["model_manifest"], model_manifest)
    metrics = {
        "schema_version": "1.0.0",
        "status": "train_complete_guard_passed"
        if guard["passed"]
        else "train_complete_guard_failed",
        "experiment": EXPERIMENT_NAME,
        "route": "ml_model",
        "cost_contract": cost_contract,
        "rows": len(base_frame),
        "wells": int(base_frame["well"].nunique()),
        "feature_counts": {
            "clean_base": len(bundle["base_features"]),
            "nested_compact": len(bundle["compact_features"]),
            "fold_safe_formation": len(bundle["formation_features"]),
            "final": len(bundle["final_features"]),
        },
        "metric_weighting": "unweighted_rows",
        "train_weight_formula": config["model"]["train_weight"]["formula"],
        "validation_weight": None,
        "guard": guard,
        "model_count": len(trained["model_rows"]),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(paths["metrics"], metrics)
    artifact_sha = {name: sha256_file(path) for name, path in paths.items()}
    reproducibility = {
        "schema_version": "1.0.0",
        "status": metrics["status"],
        "cost_contract": cost_contract,
        "parent_artifacts": parent_evidence,
        "exp287_control": exp287_evidence,
        "exp264_control": exp264_evidence,
        "stage_c_input": bundle["stage_c_evidence"],
        "clean_base_input": bundle["base_evidence"],
        "weight_contracts": weight_evidence,
        "artifact_sha256": artifact_sha,
        "model_manifest_sha256": artifact_sha["model_manifest"],
        "oof_prediction_sha256": artifact_sha["oof"],
        "submission_generated": False,
        "deterministic_anchor": False,
        "guard": guard,
    }
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    return metrics


# %% [markdown]
# ## 10. Metrics, feature importance, and generated artifacts

# %%
if not IMPORT_ONLY:
    import matplotlib.pyplot as plt

    CONFIG = read_yaml(find_config_path())
    EXECUTION_CONTRACT = validate_scientific_contract(
        CONFIG,
        require_train_approval=(
            str(CONFIG["execution"]["stage"]) == "equal_well_weight_train"
        ),
    )
    display(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "parent": CONFIG["lineage"]["parent"],
            "stage": CONFIG["execution"]["stage"],
            "execution_contract": EXECUTION_CONTRACT,
            "weight_formula": CONFIG["model"]["train_weight"]["formula"],
            "validation_weight": CONFIG["model"]["train_weight"]["validation_weight"],
        }
    )
    print("Leakage policy")
    for policy in CONFIG["validation"]["leakage_policy"]:
        print("-", policy)
    print("Forbidden actions")
    for action in CONFIG["guards"]["forbidden"]:
        print("-", action)
    RUN_SUMMARY = run_experiment()
    display(RUN_SUMMARY)
    GENERATED_DIR = KAGGLE_WORKING_ROOT / "artifacts"
    if (GENERATED_DIR / "feature_importance.csv").exists():
        IMPORTANCE = pd.read_csv(GENERATED_DIR / "feature_importance.csv")
        MEAN_GAIN = (
            IMPORTANCE[IMPORTANCE["importance_type"].eq("gain")]
            .groupby(["feature", "feature_group"], as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
        )
        display(MEAN_GAIN.head(60))
        axis = (
            MEAN_GAIN.head(30)
            .sort_values("importance")
            .plot.barh(
                x="feature",
                y="importance",
                figsize=(10, 11),
                legend=False,
                title="exp334 mean gain importance across 15 weighted models",
            )
        )
        axis.set_xlabel("mean gain importance")
        plt.tight_layout()
        plt.savefig(GENERATED_DIR / "feature_importance_top30.png", dpi=140)
        plt.show()
    print("Generated files")
    for generated in sorted(GENERATED_DIR.rglob("*")):
        if generated.is_file():
            print(generated.relative_to(GENERATED_DIR), generated.stat().st_size)

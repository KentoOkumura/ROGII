# %% [markdown]
# # exp244 early / original / late integrated training
#
# Train one new exp218-family variant from the validated official cache plus
# four prediction-start pseudo caches. The saved exp218 OOF is the control and
# is never retrained.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and approval guard
# 3. Official and offset-cache contracts
# 4. Disk-backed cache streaming
# 5. Fold-safe LightGBM training
# 6. Frozen exp218 OOF comparison
# 7. Stress metrics and adoption guards
# 8. Models, predictions, importance, and SHA artifacts

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

EXPERIMENT_NAME = "exp244_bidirectional_prediction_start_pseudotail_augmentation"
OUTPUT_PREFIX = f"{EXPERIMENT_NAME}_integrated"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Configuration and approval guard


# %%
def nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def load_config() -> tuple[Path, dict[str, Any]]:
    candidates = [Path.cwd() / "config.yaml", Path.cwd() / "inputs" / "config.yaml"]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob("**/config.yaml")))
    for path in candidates:
        if not path.exists() or not path.stat().st_size:
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if value.get("experiment", {}).get("name") == EXPERIMENT_NAME:
            return path, value
    raise FileNotFoundError(f"Could not resolve config for {EXPERIMENT_NAME}")


def find_file(filename: str) -> Path:
    candidates = [Path.cwd() / filename, Path.cwd() / "inputs" / filename]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")))
    matches = [path for path in candidates if path.exists() and path.stat().st_size]
    if not matches:
        raise FileNotFoundError(filename)
    return matches[0]


def sha256_file(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as handle:  # type: ignore[arg-type]
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_content_sha256(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(
        dtype=np.uint64, copy=False
    )
    return hashlib.sha256(values.tobytes()).hexdigest()


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# %% [markdown]
# ## 3. Official and offset-cache contracts


# %%
def cache_contract(
    *,
    prefix: str,
    official: bool,
    expected_rows: int,
    expected_shards: int,
    expected_features: int,
    expected_requests: int | None = None,
) -> dict[str, Any]:
    middle = "official_exp218_feature_cache" if official else "exp218_feature_cache"
    manifest_path = find_file(f"{prefix}_{middle}_manifest.csv")
    schema_path = find_file(f"{prefix}_{middle}_schema.csv")
    summary_path = find_file(f"{prefix}_{middle}_summary.json")
    request_manifest_path = None
    if not official:
        request_manifest_path = find_file(f"{prefix}_{middle}_requests.csv")
    manifest = pd.read_csv(manifest_path).sort_values("batch_index").reset_index(drop=True)
    schema = pd.read_csv(schema_path).sort_values("feature_index").reset_index(drop=True)
    summary = json.loads(summary_path.read_text())
    features = schema["feature"].astype(str).tolist()
    if len(manifest) != expected_shards or int(manifest["rows"].sum()) != expected_rows:
        raise AssertionError(f"Cache totals mismatch: {prefix}")
    if len(features) != expected_features:
        raise AssertionError(f"Feature count mismatch: {prefix}")
    if expected_requests is not None and int(manifest["requests"].sum()) != expected_requests:
        raise AssertionError(f"Request count mismatch: {prefix}")
    if bool(summary.get("preflight")):
        raise AssertionError(f"Refusing preflight cache: {prefix}")
    if sha256_file(manifest_path) != str(summary["manifest_sha256"]):
        raise AssertionError(f"Manifest SHA mismatch: {prefix}")
    if sha256_file(schema_path) != str(summary["schema_sha256"]):
        raise AssertionError(f"Schema SHA mismatch: {prefix}")
    if request_manifest_path is not None:
        if sha256_file(request_manifest_path) != str(summary["request_manifest_sha256"]):
            raise AssertionError(f"Request-manifest SHA mismatch: {prefix}")
        requests = pd.read_csv(request_manifest_path)
        if requests["request_id"].nunique() != expected_requests:
            raise AssertionError(f"Request-manifest count mismatch: {prefix}")
    feature_sha = hashlib.sha256("\n".join(features).encode()).hexdigest()
    if feature_sha != str(summary["feature_columns_sha256"]):
        raise AssertionError(f"Feature-column SHA mismatch: {prefix}")
    return {
        "prefix": prefix,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "schema_path": schema_path,
        "features": features,
        "summary_path": summary_path,
        "summary": summary,
        "request_manifest_path": request_manifest_path,
    }


def resolve_shard(manifest_path: Path, relative_path: str) -> Path:
    candidate = manifest_path.parent / relative_path
    if candidate.exists():
        return candidate
    matches = sorted(KAGGLE_INPUT_ROOT.glob(f"**/{Path(relative_path).name}"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not uniquely resolve shard: {relative_path}")


def load_contracts(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    section = "model.integrated_augmentation"
    expected_features = int(nested(config, f"{section}.expected_feature_count"))
    official_config = dict(nested(config, f"{section}.official_cache"))
    official = cache_contract(
        prefix=str(official_config["prefix"]),
        official=True,
        expected_rows=int(nested(config, f"{section}.expected_official_rows")),
        expected_shards=int(official_config["expected_shards"]),
        expected_features=expected_features,
    )
    if str(official["summary"]["manifest_sha256"]) != str(
        official_config["expected_manifest_sha256"]
    ):
        raise AssertionError("Official manifest changed from the approved exp239 cache")
    if str(official["summary"]["schema_sha256"]) != str(official_config["expected_schema_sha256"]):
        raise AssertionError("Official schema changed from the approved exp239 cache")
    if str(official["summary"]["feature_columns_sha256"]) != str(
        official_config["expected_feature_columns_sha256"]
    ):
        raise AssertionError("Official feature-column SHA changed")

    pseudo_contracts: list[dict[str, Any]] = []
    for item in nested(config, f"{section}.cache_specs", []):
        spec = dict(item)
        prefix = f"{EXPERIMENT_NAME}_{spec['label']}"
        contract = cache_contract(
            prefix=prefix,
            official=False,
            expected_rows=int(spec["expected_rows"]),
            expected_shards=int(spec["expected_shards"]),
            expected_features=expected_features,
            expected_requests=int(spec["expected_requests"]),
        )
        expected_sha_keys = {
            "manifest_sha256": "expected_manifest_sha256",
            "schema_sha256": "expected_schema_sha256",
            "request_manifest_sha256": "expected_request_manifest_sha256",
            "feature_columns_sha256": "expected_feature_columns_sha256",
        }
        for summary_key, spec_key in expected_sha_keys.items():
            if str(contract["summary"].get(summary_key)) != str(spec[spec_key]):
                raise AssertionError(
                    f"Pinned cache SHA mismatch for {spec['label']}: {summary_key}"
                )
        if contract["features"] != official["features"]:
            raise AssertionError(f"Official/pseudo feature schema mismatch: {spec['label']}")
        offset_contract_path = find_file(f"{prefix}_offset_cache_contract.json")
        offset_contract = json.loads(offset_contract_path.read_text())
        expected_contract = {
            "experiment": EXPERIMENT_NAME,
            "variant": nested(config, f"{section}.variant"),
            "offset_rows": int(spec["offset_rows"]),
            "offset_label": str(spec["label"]),
            "start_kind": "early" if int(spec["offset_rows"]) < 0 else "late",
            "late_train_only": int(spec["offset_rows"]) > 0,
            "validation_rows": "official_start_only",
            "feature_generation_may_read_tail_tvt": False,
            "forbid_full_prefix_cache_slice": True,
        }
        for key, value in expected_contract.items():
            if offset_contract.get(key) != value:
                raise AssertionError(f"Offset contract mismatch for {spec['label']}: {key}")
        for key in [
            "manifest_sha256",
            "schema_sha256",
            "request_manifest_sha256",
            "feature_columns_sha256",
        ]:
            if offset_contract.get(key) != contract["summary"].get(key):
                raise AssertionError(f"Offset/cache summary mismatch for {spec['label']}: {key}")
        contract["spec"] = spec
        contract["offset_contract_path"] = offset_contract_path
        contract["offset_contract"] = offset_contract
        pseudo_contracts.append(contract)
    if sum(int(item["spec"]["expected_rows"]) for item in pseudo_contracts) != int(
        nested(config, f"{section}.expected_pseudo_rows")
    ):
        raise AssertionError("Configured pseudo row total drift")
    if sum(int(item["spec"]["expected_requests"]) for item in pseudo_contracts) != int(
        nested(config, f"{section}.expected_pseudo_views")
    ):
        raise AssertionError("Configured pseudo view total drift")
    return official, pseudo_contracts


# %% [markdown]
# ## 4. Disk-backed cache streaming


# %%
def stream_caches(
    config: dict[str, Any], official: dict[str, Any], pseudo: list[dict[str, Any]]
) -> dict[str, Any]:
    section = "model.integrated_augmentation"
    official_rows = int(nested(config, f"{section}.expected_official_rows"))
    pseudo_rows = int(nested(config, f"{section}.expected_pseudo_rows"))
    features = list(official["features"])
    feature_count = len(features)
    work_dir = KAGGLE_WORKING_ROOT / "exp244_integrated_memmaps"
    work_dir.mkdir(parents=True, exist_ok=True)

    official_x = np.memmap(
        work_dir / "official_x.float32.mmap",
        dtype=np.float32,
        mode="w+",
        shape=(official_rows, feature_count),
    )
    official_y = np.memmap(
        work_dir / "official_y.float32.mmap", dtype=np.float32, mode="w+", shape=official_rows
    )
    official_ids = np.memmap(
        work_dir / "official_ids.s64.mmap", dtype="S64", mode="w+", shape=official_rows
    )
    official_groups = np.memmap(
        work_dir / "official_groups.int16.mmap", dtype=np.int16, mode="w+", shape=official_rows
    )
    group_labels: list[str] = []
    group_to_code: dict[str, int] = {}
    offset = 0
    for row in official["manifest"].itertuples(index=False):
        path = resolve_shard(official["manifest_path"], str(row.path))
        if sha256_file(path) != str(row.file_sha256):
            raise AssertionError(f"Official shard file SHA mismatch: {path.name}")
        shard = pd.read_parquet(path)
        if len(shard) != int(row.rows) or row_content_sha256(shard) != str(row.row_content_sha256):
            raise AssertionError(f"Official shard content mismatch: {path.name}")
        stop = offset + len(shard)
        official_x[offset:stop] = shard[features].to_numpy(np.float32, copy=False)
        official_y[offset:stop] = shard["target"].to_numpy(np.float32)
        ids = shard["id"].astype(str)
        if int(ids.str.len().max()) > 64:
            raise AssertionError("Official id exceeds S64")
        official_ids[offset:stop] = ids.to_numpy(dtype="S64")
        wells = shard["well"].astype(str)
        for well in pd.unique(wells):
            if well not in group_to_code:
                group_to_code[well] = len(group_labels)
                group_labels.append(well)
        official_groups[offset:stop] = wells.map(group_to_code).to_numpy(np.int16)
        offset = stop
        print(
            f"official shard {int(row.batch_index) + 1}/{len(official['manifest'])} "
            f"rows={len(shard)} peak_rss_mb={peak_rss_mb():.1f}",
            flush=True,
        )
        del shard, ids, wells
        gc.collect()
    if offset != official_rows or len(group_labels) != int(
        nested(config, "frozen_anchor_parity.expected_wells")
    ):
        raise AssertionError("Official cache streaming totals drift")
    if np.any(np.diff(np.asarray(official_groups, dtype=np.int32)) < 0):
        raise AssertionError("Official cache wells are not contiguous in stable order")
    if "last_known_tvt" not in features:
        raise AssertionError("last_known_tvt missing from feature schema")
    official_base = np.memmap(
        work_dir / "official_base.float32.mmap",
        dtype=np.float32,
        mode="w+",
        shape=official_rows,
    )
    official_base[:] = official_x[:, features.index("last_known_tvt")]

    pseudo_x = np.memmap(
        work_dir / "pseudo_x.float32.mmap",
        dtype=np.float32,
        mode="w+",
        shape=(pseudo_rows, feature_count),
    )
    pseudo_y = np.memmap(
        work_dir / "pseudo_y.float32.mmap", dtype=np.float32, mode="w+", shape=pseudo_rows
    )
    pseudo_source = np.memmap(
        work_dir / "pseudo_source.int16.mmap", dtype=np.int16, mode="w+", shape=pseudo_rows
    )
    pseudo_offsets = np.memmap(
        work_dir / "pseudo_offsets.int16.mmap", dtype=np.int16, mode="w+", shape=pseudo_rows
    )
    offset = 0
    for contract in pseudo:
        spec = contract["spec"]
        for row in contract["manifest"].itertuples(index=False):
            path = resolve_shard(contract["manifest_path"], str(row.path))
            if sha256_file(path) != str(row.file_sha256):
                raise AssertionError(f"Pseudo shard file SHA mismatch: {path}")
            shard = pd.read_parquet(path)
            if len(shard) != int(row.rows) or row_content_sha256(shard) != str(
                row.row_content_sha256
            ):
                raise AssertionError(f"Pseudo shard content mismatch: {path}")
            stop = offset + len(shard)
            pseudo_x[offset:stop] = shard[features].to_numpy(np.float32, copy=False)
            pseudo_y[offset:stop] = shard["target"].to_numpy(np.float32)
            source_codes = shard["source_well"].astype(str).map(group_to_code)
            if source_codes.isna().any():
                raise AssertionError("Pseudo cache contains an unknown source well")
            pseudo_source[offset:stop] = source_codes.to_numpy(np.int16)
            pseudo_offsets[offset:stop] = int(spec["offset_rows"])
            offset = stop
            print(
                f"pseudo {spec['label']} shard {int(row.batch_index) + 1}/"
                f"{len(contract['manifest'])} rows={len(shard)} peak_rss_mb={peak_rss_mb():.1f}",
                flush=True,
            )
            del shard, source_codes
            gc.collect()
    if offset != pseudo_rows:
        raise AssertionError("Pseudo cache streaming row total drift")
    for array in [
        official_x,
        official_y,
        official_ids,
        official_groups,
        official_base,
        pseudo_x,
        pseudo_y,
        pseudo_source,
        pseudo_offsets,
    ]:
        array.flush()
    return {
        "work_dir": work_dir,
        "features": features,
        "official_x": official_x,
        "official_y": official_y,
        "official_ids": official_ids,
        "official_groups": official_groups,
        "official_base": official_base,
        "group_labels": group_labels,
        "pseudo_x": pseudo_x,
        "pseudo_y": pseudo_y,
        "pseudo_source": pseudo_source,
        "pseudo_offsets": pseudo_offsets,
    }


# %% [markdown]
# ## 5. Fold-safe LightGBM training


# %%
def train_integrated(
    cache: dict[str, Any], config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    import gr_wavelet_rotation_confidence_features_on_exp148 as exp218
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    exp218_config = yaml.safe_load(find_file("exp218_config.yaml").read_text())
    mode = dict(nested(exp218_config, "model.training.modes.gpu_repro_guard_dp_threads8", {}))
    params_list = exp218.apply_mode_overrides(exp218.exp063_lgb_config_family(fast=False), mode)
    section = "model.integrated_augmentation"
    expected_configs = int(nested(config, f"{section}.training.lightgbm_configs"))
    expected_folds = int(nested(config, f"{section}.training.folds"))
    if len(params_list) != expected_configs or expected_folds != 5:
        raise AssertionError("LightGBM config/fold count drift")

    official_x = cache["official_x"]
    official_y = cache["official_y"]
    official_base = cache["official_base"]
    groups = cache["official_groups"]
    pseudo_x = cache["pseudo_x"]
    pseudo_y = cache["pseudo_y"]
    pseudo_source = cache["pseudo_source"]
    features = cache["features"]
    official_weight = float(nested(config, f"{section}.official_row_weight"))
    pseudo_weight = float(nested(config, f"{section}.pseudo_row_weight"))
    folds = GroupKFold(n_splits=expected_folds)
    fold_assignment = np.full(len(official_y), -1, dtype=np.int8)
    oofs: list[np.ndarray] = []
    training_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    model_dir = KAGGLE_WORKING_ROOT / f"{OUTPUT_PREFIX}_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    work_dir = cache["work_dir"]

    for model_index, params in enumerate(params_list):
        oof = np.zeros(len(official_y), dtype=np.float32)
        for fold, (train_idx, valid_idx) in enumerate(
            folds.split(official_y, official_y, groups=groups)
        ):
            if model_index == 0:
                fold_assignment[valid_idx] = fold
            valid_groups = np.unique(np.asarray(groups[valid_idx]))
            pseudo_idx = np.flatnonzero(~np.isin(pseudo_source, valid_groups))
            train_rows_count = len(train_idx) + len(pseudo_idx)
            train_path = work_dir / f"lgb{model_index}_fold{fold}_train.float32.mmap"
            valid_path = work_dir / f"lgb{model_index}_fold{fold}_valid.float32.mmap"
            x_train = np.memmap(
                train_path,
                dtype=np.float32,
                mode="w+",
                shape=(train_rows_count, len(features)),
            )
            x_valid = np.memmap(
                valid_path,
                dtype=np.float32,
                mode="w+",
                shape=(len(valid_idx), len(features)),
            )
            np.take(official_x, train_idx, axis=0, out=x_train[: len(train_idx)])
            np.take(pseudo_x, pseudo_idx, axis=0, out=x_train[len(train_idx) :])
            np.take(official_x, valid_idx, axis=0, out=x_valid)
            x_train.flush()
            x_valid.flush()
            y_train = np.concatenate([official_y[train_idx], pseudo_y[pseudo_idx]])
            weights = np.concatenate(
                [
                    np.full(len(train_idx), official_weight, np.float32),
                    np.full(len(pseudo_idx), pseudo_weight, np.float32),
                ]
            )
            print(
                f"train lgb{model_index} fold{fold}: official={len(train_idx)} "
                f"pseudo={len(pseudo_idx)} valid={len(valid_idx)} "
                f"peak_rss_mb={peak_rss_mb():.1f}",
                flush=True,
            )
            model = LGBMRegressor(**params)
            model.fit(
                x_train,
                y_train,
                sample_weight=weights,
                eval_set=[(x_valid, official_y[valid_idx])],
                eval_metric="rmse",
                callbacks=[early_stopping(250, verbose=False), log_evaluation(0)],
            )
            best_iteration = int(model.best_iteration_ or params.get("n_estimators", 0))
            prediction = model.predict(x_valid, num_iteration=best_iteration).astype(np.float32)
            oof[valid_idx] = prediction
            model_path = model_dir / f"lgb{model_index}_fold{fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=best_iteration)
            model_record = {
                "model": f"lgb{model_index}",
                "fold": fold,
                "best_iteration": best_iteration,
                "path": str(model_path),
                "sha256": sha256_file(model_path),
            }
            models.append(model_record)
            target_tvt = np.asarray(official_base[valid_idx]) + np.asarray(official_y[valid_idx])
            pred_tvt = np.asarray(official_base[valid_idx]) + prediction
            training_rows.append(
                {
                    **model_record,
                    "official_train_rows": len(train_idx),
                    "pseudo_train_rows": len(pseudo_idx),
                    "valid_rows": len(valid_idx),
                    "rmse_tvt": float(np.sqrt(np.mean(np.square(pred_tvt - target_tvt)))),
                }
            )
            importance_rows.extend(
                {
                    "model": f"lgb{model_index}",
                    "fold": fold,
                    "feature": feature,
                    "gain": float(gain),
                }
                for feature, gain in zip(
                    features,
                    model.booster_.feature_importance(importance_type="gain"),
                    strict=True,
                )
            )
            del x_train, x_valid, y_train, weights, model, prediction, pseudo_idx
            gc.collect()
            train_path.unlink()
            valid_path.unlink()
        oofs.append(oof)
    if np.any(fold_assignment < 0) or len(models) != int(
        nested(config, f"{section}.training.boosters")
    ):
        raise AssertionError("OOF coverage or model count drift")
    model_oofs = np.vstack(oofs)
    mean_residual = np.mean(model_oofs, axis=0).astype(np.float32)
    return (
        mean_residual,
        fold_assignment,
        pd.DataFrame(training_rows),
        pd.DataFrame(importance_rows),
        models,
    )


# %% [markdown]
# ## 6. Frozen exp218 OOF comparison


# %%
def load_frozen_exp218_oof(cache: dict[str, Any], config: dict[str, Any]) -> np.memmap:
    filename = "exp218_gr_wavelet_rotation_confidence_features_on_exp148_predictions.csv.gz"
    path = find_file(filename)
    expected_sha = str(nested(config, "frozen_anchor_parity.expected_oof_decompressed_sha256"))
    if sha256_file(path, decompressed=True) != expected_sha:
        raise AssertionError("Frozen exp218 OOF decompressed SHA mismatch")
    work_dir = cache["work_dir"]
    baseline = np.memmap(
        work_dir / "baseline_tvt.float32.mmap",
        dtype=np.float32,
        mode="w+",
        shape=len(cache["official_y"]),
    )
    offset = 0
    required = ["id", "well", "variant", "mode", "model", "target_tvt", "pred_tvt"]
    for chunk in pd.read_csv(path, usecols=required, chunksize=250_000):
        stop = offset + len(chunk)
        if not np.array_equal(
            chunk["id"].astype(str).to_numpy(dtype="S64"),
            np.asarray(cache["official_ids"][offset:stop]),
        ):
            raise AssertionError("Frozen exp218 OOF id order differs from official cache")
        if set(chunk["variant"].astype(str)) != {
            str(nested(config, "frozen_anchor_parity.variant"))
        }:
            raise AssertionError("Frozen exp218 variant drift")
        if set(chunk["mode"].astype(str)) != {str(nested(config, "frozen_anchor_parity.mode"))}:
            raise AssertionError("Frozen exp218 mode drift")
        if set(chunk["model"].astype(str)) != {str(nested(config, "frozen_anchor_parity.model"))}:
            raise AssertionError("Frozen exp218 model drift")
        expected_target = np.asarray(cache["official_base"][offset:stop]) + np.asarray(
            cache["official_y"][offset:stop]
        )
        if not np.allclose(chunk["target_tvt"].to_numpy(float), expected_target, atol=0.002):
            raise AssertionError("Frozen exp218 target differs from official cache")
        baseline[offset:stop] = chunk["pred_tvt"].to_numpy(np.float32)
        offset = stop
    if offset != len(baseline):
        raise AssertionError("Frozen exp218 OOF row count mismatch")
    baseline.flush()
    return baseline


# %% [markdown]
# ## 7. Stress metrics and adoption guards


# %%
def rmse(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        error = np.asarray(predicted) - np.asarray(actual)
    else:
        error = np.asarray(predicted)[mask] - np.asarray(actual)[mask]
    return float(np.sqrt(np.mean(np.square(error, dtype=np.float64))))


def load_hidden_masks(group_labels: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    filename = "exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv"
    path = find_file(filename)
    roles = pd.read_csv(path, dtype={"well_id": str})
    required = {
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    }
    if not required.issubset(roles.columns):
        raise ValueError(f"Hidden-like assignment missing: {sorted(required - set(roles.columns))}")
    indexed = roles.set_index("well_id")
    spatial = np.asarray(
        [indexed.at[well, "verification_like_spatial_role"] == "valid" for well in group_labels]
    )
    typewell = np.asarray(
        [
            indexed.at[well, "verification_like_typewell_purged_role"] == "valid"
            for well in group_labels
        ]
    )
    return spatial, typewell, {"path": str(path), "sha256": sha256_file(path)}


def evaluate(
    cache: dict[str, Any],
    config: dict[str, Any],
    mean_residual: np.ndarray,
    fold_assignment: np.ndarray,
    baseline_tvt: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray]:
    target_tvt = np.asarray(cache["official_base"]) + np.asarray(cache["official_y"])
    new_tvt = np.asarray(cache["official_base"]) + mean_residual
    groups = np.asarray(cache["official_groups"], dtype=np.int32)
    counts = np.bincount(groups, minlength=len(cache["group_labels"]))
    starts = np.concatenate([[0], np.cumsum(counts[:-1])])
    eval_step = np.arange(len(groups), dtype=np.int32) - starts[groups]
    spatial_groups, typewell_groups, hidden_meta = load_hidden_masks(cache["group_labels"])
    surfaces: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(groups), dtype=bool)),
        ("000_050", eval_step < 50),
        ("050_100", (eval_step >= 50) & (eval_step < 100)),
        ("100_250", (eval_step >= 100) & (eval_step < 250)),
        ("250_500", (eval_step >= 250) & (eval_step < 500)),
        ("500_1000", (eval_step >= 500) & (eval_step < 1000)),
        ("1000_plus", eval_step >= 1000),
        ("hidden_like_spatial", spatial_groups[groups]),
        ("hidden_like_typewell_purged", typewell_groups[groups]),
    ]
    rows: list[dict[str, Any]] = []
    for surface, mask in surfaces:
        raw = rmse(target_tvt, baseline_tvt, mask)
        new = rmse(target_tvt, new_tvt, mask)
        rows.append(
            {
                "surface": surface,
                "rows": int(mask.sum()),
                "raw_exp218_rmse": raw,
                "integrated_rmse": new,
                "delta_rmse": new - raw,
            }
        )
    for fold in range(int(nested(config, "validation.n_folds", 5))):
        mask = fold_assignment == fold
        raw = rmse(target_tvt, baseline_tvt, mask)
        new = rmse(target_tvt, new_tvt, mask)
        rows.append(
            {
                "surface": f"fold_{fold}",
                "rows": int(mask.sum()),
                "raw_exp218_rmse": raw,
                "integrated_rmse": new,
                "delta_rmse": new - raw,
            }
        )
    metrics = pd.DataFrame(rows)

    raw_sq = np.square(np.asarray(baseline_tvt) - target_tvt, dtype=np.float64)
    new_sq = np.square(new_tvt - target_tvt, dtype=np.float64)
    raw_well = np.sqrt(np.bincount(groups, weights=raw_sq) / counts)
    new_well = np.sqrt(np.bincount(groups, weights=new_sq) / counts)
    by_well = pd.DataFrame(
        {
            "well_id": cache["group_labels"],
            "rows": counts,
            "raw_exp218_rmse": raw_well,
            "integrated_rmse": new_well,
            "delta_rmse": new_well - raw_well,
        }
    ).sort_values("well_id")
    lookup = metrics.set_index("surface")
    fold_deltas = [float(lookup.at[f"fold_{fold}", "delta_rmse"]) for fold in range(5)]
    guard_config = dict(nested(config, "model.integrated_augmentation.adoption_guards"))
    guards = {
        "overall_improved": bool(lookup.at["overall", "delta_rmse"] < 0),
        "1000_plus_non_worse": bool(lookup.at["1000_plus", "delta_rmse"] <= 0),
        "hidden_like_spatial_non_worse": bool(lookup.at["hidden_like_spatial", "delta_rmse"] <= 0),
        "hidden_like_typewell_purged_non_worse": bool(
            lookup.at["hidden_like_typewell_purged", "delta_rmse"] <= 0
        ),
        "worst_well_regression_within_limit": bool(
            by_well["delta_rmse"].max() <= float(guard_config["max_worst_well_regression"])
        ),
        "improved_folds": int(sum(value < 0 for value in fold_deltas)),
        "minimum_improved_folds": int(guard_config["min_improved_folds"]),
    }
    guards["adoption_supported"] = bool(
        guards["overall_improved"]
        and guards["1000_plus_non_worse"]
        and guards["hidden_like_spatial_non_worse"]
        and guards["hidden_like_typewell_purged_non_worse"]
        and guards["worst_well_regression_within_limit"]
        and guards["improved_folds"] >= guards["minimum_improved_folds"]
    )
    expected_raw = float(
        nested(config, "model.integrated_augmentation.training.reference_exp218_lgb_mean_rmse")
    )
    if abs(float(lookup.at["overall", "raw_exp218_rmse"]) - expected_raw) > float(
        nested(config, "frozen_anchor_parity.rmse_tolerance")
    ):
        raise AssertionError("Frozen exp218 RMSE changed")
    return (
        metrics,
        by_well,
        {"guards": guards, "hidden_assignment": hidden_meta},
        target_tvt,
        new_tvt,
    )


# %% [markdown]
# ## 8. Models, predictions, importance, and SHA artifacts


# %%
def write_predictions(
    path: Path,
    cache: dict[str, Any],
    fold_assignment: np.ndarray,
    target_tvt: np.ndarray,
    baseline_tvt: np.ndarray,
    new_tvt: np.ndarray,
) -> None:
    groups = np.asarray(cache["official_groups"], dtype=np.int32)
    counts = np.bincount(groups, minlength=len(cache["group_labels"]))
    starts = np.concatenate([[0], np.cumsum(counts[:-1])])
    with gzip.open(path, "wt", newline="") as handle:
        for start in range(0, len(target_tvt), 250_000):
            stop = min(start + 250_000, len(target_tvt))
            group_chunk = groups[start:stop]
            frame = pd.DataFrame(
                {
                    "id": np.asarray(cache["official_ids"][start:stop]).astype(str),
                    "well": [cache["group_labels"][code] for code in group_chunk],
                    "fold": fold_assignment[start:stop],
                    "eval_step": np.arange(start, stop, dtype=np.int64) - starts[group_chunk],
                    "target_tvt": target_tvt[start:stop],
                    "raw_exp218_pred_tvt": np.asarray(baseline_tvt[start:stop]),
                    "pred_tvt": new_tvt[start:stop],
                }
            )
            frame.to_csv(handle, index=False, header=start == 0, lineterminator="\n")


def save_importance_plot(importance: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    top = importance.groupby("feature", as_index=False)["gain"].mean().nlargest(30, "gain")
    fig, axis = plt.subplots(figsize=(9, 10))
    axis.barh(top["feature"][::-1], top["gain"][::-1])
    axis.set_title("exp244 integrated augmentation mean gain")
    axis.set_xlabel("mean gain")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def close_and_cleanup(cache: dict[str, Any], baseline: np.memmap) -> None:
    arrays = [
        cache["official_x"],
        cache["official_y"],
        cache["official_ids"],
        cache["official_groups"],
        cache["official_base"],
        cache["pseudo_x"],
        cache["pseudo_y"],
        cache["pseudo_source"],
        cache["pseudo_offsets"],
        baseline,
    ]
    for array in arrays:
        array.flush()
        array._mmap.close()
    gc.collect()
    for path in cache["work_dir"].glob("*.mmap"):
        path.unlink()


# %% [markdown]
# ## Execution orchestration


# %%
def main() -> dict[str, Any]:
    if not KAGGLE_INPUT_ROOT.exists():
        raise RuntimeError("The integrated training notebook must run on Kaggle")

    STARTED = time.time()
    CONFIG_PATH, CONFIG = load_config()
    SECTION = "model.integrated_augmentation"
    if not bool(nested(CONFIG, f"{SECTION}.run_approved", False)):
        raise RuntimeError(
            "GPU training is implemented but not approved. Set "
            "model.integrated_augmentation.run_approved=true only after explicit user approval."
        )
    print(f"config={CONFIG_PATH}")
    print(f"experiment={nested(CONFIG, 'experiment.name')}")
    print(f"route={nested(CONFIG, 'experiment.route')}")
    print(f"variant={nested(CONFIG, f'{SECTION}.variant')}")
    print(
        "approved_cost="
        f"variants={nested(CONFIG, f'{SECTION}.training.active_variants')} "
        f"configs={nested(CONFIG, f'{SECTION}.training.lightgbm_configs')} "
        f"folds={nested(CONFIG, f'{SECTION}.training.folds')} "
        f"boosters={nested(CONFIG, f'{SECTION}.training.boosters')} "
        "parent_control_retrained="
        f"{nested(CONFIG, f'{SECTION}.training.parent_control_retrained')}",
        flush=True,
    )

    OFFICIAL_CONTRACT, PSEUDO_CONTRACTS = load_contracts(CONFIG)
    CACHE = stream_caches(CONFIG, OFFICIAL_CONTRACT, PSEUDO_CONTRACTS)
    MEAN_RESIDUAL, FOLD_ASSIGNMENT, TRAIN_METRICS, IMPORTANCE, MODELS = train_integrated(
        CACHE, CONFIG
    )
    BASELINE_TVT = load_frozen_exp218_oof(CACHE, CONFIG)
    METRICS, BY_WELL, EVALUATION, TARGET_TVT, NEW_TVT = evaluate(
        CACHE, CONFIG, MEAN_RESIDUAL, FOLD_ASSIGNMENT, BASELINE_TVT
    )

    TRAIN_METRICS_PATH = KAGGLE_WORKING_ROOT / f"{OUTPUT_PREFIX}_training_metrics.csv"
    METRICS_PATH = KAGGLE_WORKING_ROOT / f"{OUTPUT_PREFIX}_metrics.csv"
    BY_WELL_PATH = KAGGLE_WORKING_ROOT / f"{OUTPUT_PREFIX}_by_well.csv"
    IMPORTANCE_PATH = KAGGLE_WORKING_ROOT / f"{OUTPUT_PREFIX}_feature_importance.csv"
    IMPORTANCE_PLOT_PATH = KAGGLE_WORKING_ROOT / f"{OUTPUT_PREFIX}_feature_importance.png"
    PREDICTIONS_PATH = KAGGLE_WORKING_ROOT / f"{OUTPUT_PREFIX}_predictions.csv.gz"
    MODEL_MANIFEST_PATH = KAGGLE_WORKING_ROOT / f"{OUTPUT_PREFIX}_model_manifest.json"
    SCHEMA_PATH = KAGGLE_WORKING_ROOT / f"{OUTPUT_PREFIX}_feature_schema.csv"

    TRAIN_METRICS.to_csv(TRAIN_METRICS_PATH, index=False)
    METRICS.to_csv(METRICS_PATH, index=False)
    BY_WELL.to_csv(BY_WELL_PATH, index=False)
    IMPORTANCE.to_csv(IMPORTANCE_PATH, index=False)
    save_importance_plot(IMPORTANCE, IMPORTANCE_PLOT_PATH)
    pd.DataFrame(
        {"feature_index": range(len(CACHE["features"])), "feature": CACHE["features"]}
    ).to_csv(SCHEMA_PATH, index=False)
    write_predictions(
        PREDICTIONS_PATH,
        CACHE,
        FOLD_ASSIGNMENT,
        TARGET_TVT,
        BASELINE_TVT,
        NEW_TVT,
    )
    MODEL_MANIFEST = {
        "experiment": EXPERIMENT_NAME,
        "variant": nested(CONFIG, f"{SECTION}.variant"),
        "mode": "gpu_repro_guard_dp_threads8",
        "model_count": len(MODELS),
        "models": MODELS,
    }
    MODEL_MANIFEST_PATH.write_text(json.dumps(MODEL_MANIFEST, indent=2, sort_keys=True) + "\n")

    METRIC_LOOKUP = METRICS.set_index("surface")
    SUMMARY = {
        "experiment": EXPERIMENT_NAME,
        "status": "v4_integrated_training_complete",
        "route": nested(CONFIG, "experiment.route"),
        "variant": nested(CONFIG, f"{SECTION}.variant"),
        "official_rows": int(nested(CONFIG, f"{SECTION}.expected_official_rows")),
        "pseudo_rows": int(nested(CONFIG, f"{SECTION}.expected_pseudo_rows")),
        "pseudo_views": int(nested(CONFIG, f"{SECTION}.expected_pseudo_views")),
        "feature_count": len(CACHE["features"]),
        "raw_exp218_oof_rmse": float(METRIC_LOOKUP.at["overall", "raw_exp218_rmse"]),
        "integrated_oof_rmse": float(METRIC_LOOKUP.at["overall", "integrated_rmse"]),
        "delta_rmse": float(METRIC_LOOKUP.at["overall", "delta_rmse"]),
        "evaluation": EVALUATION,
        "execution": {
            "active_variants": 1,
            "lightgbm_configs": 3,
            "folds": 5,
            "boosters": 15,
            "parent_control_retrained": False,
            "elapsed_seconds": time.time() - STARTED,
            "peak_rss_mb": peak_rss_mb(),
        },
        "input_contracts": {
            "official_manifest_sha256": OFFICIAL_CONTRACT["summary"]["manifest_sha256"],
            "pseudo": [
                {
                    "label": item["spec"]["label"],
                    "offset_rows": item["spec"]["offset_rows"],
                    "manifest_sha256": item["summary"]["manifest_sha256"],
                    "schema_sha256": item["summary"]["schema_sha256"],
                    "request_manifest_sha256": item["summary"]["request_manifest_sha256"],
                    "offset_contract_sha256": sha256_file(item["offset_contract_path"]),
                }
                for item in PSEUDO_CONTRACTS
            ],
            "frozen_exp218_oof_decompressed_sha256": nested(
                CONFIG, "frozen_anchor_parity.expected_oof_decompressed_sha256"
            ),
        },
        "artifacts": {
            "training_metrics_sha256": sha256_file(TRAIN_METRICS_PATH),
            "metrics_sha256": sha256_file(METRICS_PATH),
            "by_well_sha256": sha256_file(BY_WELL_PATH),
            "feature_importance_sha256": sha256_file(IMPORTANCE_PATH),
            "feature_schema_sha256": sha256_file(SCHEMA_PATH),
            "model_manifest_sha256": sha256_file(MODEL_MANIFEST_PATH),
            "prediction_decompressed_sha256": sha256_file(PREDICTIONS_PATH, decompressed=True),
        },
        "inference_prediction_performed": False,
        "submission_created": False,
    }
    SUMMARY_PATH = KAGGLE_WORKING_ROOT / f"{OUTPUT_PREFIX}_summary.json"
    SUMMARY_PATH.write_text(json.dumps(SUMMARY, indent=2, sort_keys=True) + "\n")
    print(METRICS.to_string(index=False), flush=True)
    print(json.dumps(SUMMARY, indent=2, sort_keys=True), flush=True)
    close_and_cleanup(CACHE, BASELINE_TVT)
    return SUMMARY


if os.environ.get("EXP244_IMPORT_ONLY", "0") != "1":
    TRAINING_SUMMARY = main()

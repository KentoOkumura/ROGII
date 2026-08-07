# %% [markdown]
# # exp260 matched early / late attribution train
#
# Reuse the exact exp244 official and offset caches. Train only two matched
# variants: official + early views and official + late views. Frozen exp218 and
# exp244 mixed OOF predictions are read-only references.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and approval guard
# 3. Parent cache and frozen-reference contracts
# 4. Direction-masked LightGBM training
# 5. Matched stress metrics and attribution guards
# 6. Models, OOF, feature importance, and SHA artifacts
# 7. Execution orchestration

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

os.environ["EXP244_IMPORT_ONLY"] = "1"
import exp244_integrated_parent as parent  # noqa: E402

EXPERIMENT_NAME = "exp260_matched_early_late_attribution_on_exp244"
OUTPUT_PREFIX = EXPERIMENT_NAME
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
        if nested(value, "experiment.name") == EXPERIMENT_NAME:
            return path, value
    raise FileNotFoundError(f"Could not resolve config for {EXPERIMENT_NAME}")


def sha256_file(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as handle:  # type: ignore[arg-type]
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_approval_and_cost(config: dict[str, Any]) -> list[dict[str, Any]]:
    section = "model.attribution"
    if not bool(nested(config, f"{section}.run_approved", False)):
        raise RuntimeError("Matched 30-booster GPU run is not approved")
    variants = [dict(item) for item in nested(config, f"{section}.active_variants", [])]
    training = dict(nested(config, f"{section}.training", {}))
    if [item.get("name") for item in variants] != ["early_only", "late_only"]:
        raise AssertionError("Expected exactly early_only and late_only variants")
    if (
        int(training.get("active_variants", -1)) != 2
        or int(training.get("lightgbm_configs", -1)) != 3
        or int(training.get("folds", -1)) != 5
        or int(training.get("boosters_per_variant", -1)) != 15
        or int(training.get("total_boosters", -1)) != 30
        or bool(training.get("parent_control_retrained", True))
    ):
        raise AssertionError("Approved compute contract drift")
    expected = {
        "early_only": {"offsets_rows": [-1000, -250], "rows": 384250, "views": 1537},
        "late_only": {"offsets_rows": [250, 1000], "rows": 385907, "views": 1544},
    }
    for item in variants:
        spec = expected[str(item["name"])]
        if [int(value) for value in item["offsets_rows"]] != spec["offsets_rows"]:
            raise AssertionError(f"Offset drift: {item['name']}")
        if int(item["expected_pseudo_rows"]) != spec["rows"]:
            raise AssertionError(f"Pseudo row drift: {item['name']}")
        if int(item["expected_pseudo_views"]) != spec["views"]:
            raise AssertionError(f"Pseudo view drift: {item['name']}")
    return variants


# %% [markdown]
# ## 3. Parent cache and frozen-reference contracts


# %%
def load_mixed_exp244_oof(
    cache: dict[str, Any], config: dict[str, Any], baseline_tvt: np.ndarray
) -> tuple[np.memmap, np.ndarray, dict[str, Any]]:
    filename = (
        "exp244_bidirectional_prediction_start_pseudotail_augmentation_"
        "integrated_predictions.csv.gz"
    )
    path = parent.find_file(filename)
    expected_sha = str(
        nested(config, "model.attribution.references.mixed_prediction_decompressed_sha256")
    )
    actual_sha = sha256_file(path, decompressed=True)
    if actual_sha != expected_sha:
        raise AssertionError("Frozen exp244 mixed OOF decompressed SHA mismatch")
    manifest_path = parent.find_file(
        "exp244_bidirectional_prediction_start_pseudotail_augmentation_"
        "integrated_model_manifest.json"
    )
    if sha256_file(manifest_path) != str(
        nested(config, "model.attribution.references.mixed_model_manifest_sha256")
    ):
        raise AssertionError("Frozen exp244 mixed model manifest SHA mismatch")

    mixed = np.memmap(
        cache["work_dir"] / "mixed_exp244_tvt.float32.mmap",
        dtype=np.float32,
        mode="w+",
        shape=len(cache["official_y"]),
    )
    fold_assignment = np.full(len(mixed), -1, dtype=np.int8)
    offset = 0
    required = [
        "id",
        "fold",
        "target_tvt",
        "raw_exp218_pred_tvt",
        "pred_tvt",
    ]
    for chunk in pd.read_csv(path, usecols=required, chunksize=250_000):
        stop = offset + len(chunk)
        expected_ids = np.asarray(cache["official_ids"][offset:stop])
        if not np.array_equal(chunk["id"].astype(str).to_numpy(dtype="S64"), expected_ids):
            raise AssertionError("Frozen exp244 mixed OOF id order mismatch")
        expected_target = np.asarray(cache["official_base"][offset:stop]) + np.asarray(
            cache["official_y"][offset:stop]
        )
        if not np.allclose(chunk["target_tvt"].to_numpy(float), expected_target, atol=0.002):
            raise AssertionError("Frozen exp244 mixed target mismatch")
        if not np.allclose(
            chunk["raw_exp218_pred_tvt"].to_numpy(float),
            np.asarray(baseline_tvt[offset:stop]),
            atol=0.002,
        ):
            raise AssertionError("Frozen exp244 raw exp218 reference mismatch")
        mixed[offset:stop] = chunk["pred_tvt"].to_numpy(np.float32)
        fold_assignment[offset:stop] = chunk["fold"].to_numpy(np.int8)
        offset = stop
    if offset != len(mixed) or np.any(fold_assignment < 0):
        raise AssertionError("Frozen exp244 mixed OOF coverage mismatch")
    mixed.flush()
    target = np.asarray(cache["official_base"]) + np.asarray(cache["official_y"])
    mixed_rmse = parent.rmse(target, mixed)
    expected_rmse = float(nested(config, "model.attribution.references.mixed_exp244_oof_rmse"))
    if abs(mixed_rmse - expected_rmse) > float(
        nested(config, "frozen_anchor_parity.rmse_tolerance")
    ):
        raise AssertionError("Frozen exp244 mixed OOF RMSE mismatch")
    return mixed, fold_assignment, {
        "path": str(path),
        "prediction_decompressed_sha256": actual_sha,
        "model_manifest_sha256": sha256_file(manifest_path),
        "rmse": mixed_rmse,
    }


def load_lgb_params(config: dict[str, Any]) -> list[dict[str, Any]]:
    import gr_wavelet_rotation_confidence_features_on_exp148 as exp218

    exp218_config = yaml.safe_load(parent.find_file("exp218_config.yaml").read_text())
    mode_name = str(nested(config, "model.attribution.training.mode"))
    mode = dict(nested(exp218_config, f"model.training.modes.{mode_name}", {}))
    params_list = exp218.apply_mode_overrides(exp218.exp063_lgb_config_family(fast=False), mode)
    if len(params_list) != int(
        nested(config, "model.attribution.training.lightgbm_configs")
    ):
        raise AssertionError("LightGBM config count drift")
    for params in params_list:
        expected = {
            "device_type": "gpu",
            "gpu_use_dp": True,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 8,
            "n_jobs": 8,
        }
        for key, value in expected.items():
            if params.get(key) != value:
                raise AssertionError(f"GPU reproducibility mode drift: {key}")
    return params_list


# %% [markdown]
# ## 4. Direction-masked LightGBM training


# %%
def train_variant(
    cache: dict[str, Any],
    config: dict[str, Any],
    variant: dict[str, Any],
    params_list: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    variant_name = str(variant["name"])
    allowed_offsets = np.asarray(variant["offsets_rows"], dtype=np.int16)
    direction_mask = np.isin(np.asarray(cache["pseudo_offsets"]), allowed_offsets)
    if int(direction_mask.sum()) != int(variant["expected_pseudo_rows"]):
        raise AssertionError(f"Direction row count mismatch: {variant_name}")

    official_x = cache["official_x"]
    official_y = cache["official_y"]
    official_base = cache["official_base"]
    groups = cache["official_groups"]
    pseudo_x = cache["pseudo_x"]
    pseudo_y = cache["pseudo_y"]
    pseudo_source = np.asarray(cache["pseudo_source"])
    features = cache["features"]
    official_weight = float(nested(config, "model.attribution.official_row_weight"))
    pseudo_weight = float(nested(config, "model.attribution.pseudo_row_weight"))
    n_folds = int(nested(config, "model.attribution.training.folds"))
    folds = GroupKFold(n_splits=n_folds)
    fold_assignment = np.full(len(official_y), -1, dtype=np.int8)
    oofs: list[np.ndarray] = []
    training_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    model_dir = KAGGLE_WORKING_ROOT / f"{OUTPUT_PREFIX}_{variant_name}_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for model_index, params in enumerate(params_list):
        oof = np.zeros(len(official_y), dtype=np.float32)
        for fold, (train_idx, valid_idx) in enumerate(
            folds.split(official_y, official_y, groups=groups)
        ):
            if model_index == 0:
                fold_assignment[valid_idx] = fold
            valid_groups = np.unique(np.asarray(groups[valid_idx]))
            pseudo_idx = np.flatnonzero(
                direction_mask & ~np.isin(pseudo_source, valid_groups)
            )
            train_rows_count = len(train_idx) + len(pseudo_idx)
            train_path = (
                cache["work_dir"]
                / f"{variant_name}_lgb{model_index}_fold{fold}_train.float32.mmap"
            )
            valid_path = (
                cache["work_dir"]
                / f"{variant_name}_lgb{model_index}_fold{fold}_valid.float32.mmap"
            )
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
                f"train {variant_name} lgb{model_index} fold{fold}: "
                f"official={len(train_idx)} pseudo={len(pseudo_idx)} "
                f"valid={len(valid_idx)} peak_rss_mb={parent.peak_rss_mb():.1f}",
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
                "variant": variant_name,
                "model": f"lgb{model_index}",
                "fold": fold,
                "best_iteration": best_iteration,
                "path": str(model_path),
                "sha256": sha256_file(model_path),
            }
            models.append(model_record)
            target_tvt = np.asarray(official_base[valid_idx]) + np.asarray(
                official_y[valid_idx]
            )
            pred_tvt = np.asarray(official_base[valid_idx]) + prediction
            training_rows.append(
                {
                    **model_record,
                    "official_train_rows": len(train_idx),
                    "pseudo_train_rows": len(pseudo_idx),
                    "valid_rows": len(valid_idx),
                    "rmse_tvt": parent.rmse(target_tvt, pred_tvt),
                }
            )
            importance_rows.extend(
                {
                    "variant": variant_name,
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

    expected_models = int(nested(config, "model.attribution.training.boosters_per_variant"))
    if np.any(fold_assignment < 0) or len(models) != expected_models:
        raise AssertionError(f"OOF coverage/model count drift: {variant_name}")
    mean_residual = np.mean(np.vstack(oofs), axis=0).astype(np.float32)
    return (
        mean_residual,
        fold_assignment,
        pd.DataFrame(training_rows),
        pd.DataFrame(importance_rows),
        models,
    )


# %% [markdown]
# ## 5. Matched stress metrics and attribution guards


# %%
def rename_variant_metrics(
    metrics: pd.DataFrame, by_well: pd.DataFrame, variant_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        metrics.rename(
            columns={
                "integrated_rmse": f"{variant_name}_rmse",
                "delta_rmse": f"{variant_name}_delta_vs_raw",
            }
        ),
        by_well.rename(
            columns={
                "integrated_rmse": f"{variant_name}_rmse",
                "delta_rmse": f"{variant_name}_delta_vs_raw",
            }
        ),
    )


def combine_metrics(
    mixed_metrics: pd.DataFrame,
    early_metrics: pd.DataFrame,
    late_metrics: pd.DataFrame,
) -> pd.DataFrame:
    mixed, _ = rename_variant_metrics(mixed_metrics, pd.DataFrame(), "mixed_exp244")
    early, _ = rename_variant_metrics(early_metrics, pd.DataFrame(), "early_only")
    late, _ = rename_variant_metrics(late_metrics, pd.DataFrame(), "late_only")
    combined = mixed[
        [
            "surface",
            "rows",
            "raw_exp218_rmse",
            "mixed_exp244_rmse",
            "mixed_exp244_delta_vs_raw",
        ]
    ].merge(
        early[["surface", "early_only_rmse", "early_only_delta_vs_raw"]],
        on="surface",
        validate="1:1",
    )
    combined = combined.merge(
        late[["surface", "late_only_rmse", "late_only_delta_vs_raw"]],
        on="surface",
        validate="1:1",
    )
    combined["late_minus_early"] = combined["late_only_rmse"] - combined["early_only_rmse"]
    combined["early_minus_mixed"] = (
        combined["early_only_rmse"] - combined["mixed_exp244_rmse"]
    )
    combined["late_minus_mixed"] = (
        combined["late_only_rmse"] - combined["mixed_exp244_rmse"]
    )
    return combined


def combine_by_well(
    mixed_by_well: pd.DataFrame,
    early_by_well: pd.DataFrame,
    late_by_well: pd.DataFrame,
) -> pd.DataFrame:
    _, mixed = rename_variant_metrics(pd.DataFrame(), mixed_by_well, "mixed_exp244")
    _, early = rename_variant_metrics(pd.DataFrame(), early_by_well, "early_only")
    _, late = rename_variant_metrics(pd.DataFrame(), late_by_well, "late_only")
    combined = mixed[
        [
            "well_id",
            "rows",
            "raw_exp218_rmse",
            "mixed_exp244_rmse",
            "mixed_exp244_delta_vs_raw",
        ]
    ].merge(
        early[["well_id", "early_only_rmse", "early_only_delta_vs_raw"]],
        on="well_id",
        validate="1:1",
    )
    combined = combined.merge(
        late[["well_id", "late_only_rmse", "late_only_delta_vs_raw"]],
        on="well_id",
        validate="1:1",
    )
    combined["late_minus_early"] = combined["late_only_rmse"] - combined["early_only_rmse"]
    combined["early_minus_mixed"] = (
        combined["early_only_rmse"] - combined["mixed_exp244_rmse"]
    )
    combined["late_minus_mixed"] = (
        combined["late_only_rmse"] - combined["mixed_exp244_rmse"]
    )
    return combined.sort_values("well_id").reset_index(drop=True)


def by_well_summary(frame: pd.DataFrame, variant_name: str) -> dict[str, Any]:
    delta_column = f"{variant_name}_delta_vs_raw"
    worst = frame.nlargest(1, delta_column).iloc[0]
    return {
        "improved_wells": int((frame[delta_column] < 0).sum()),
        "worsened_wells": int((frame[delta_column] > 0).sum()),
        "wells_over_2ft_regression": int((frame[delta_column] > 2.0).sum()),
        "worst_well": str(worst["well_id"]),
        "worst_well_regression": float(worst[delta_column]),
        "worst_raw_rmse": float(worst["raw_exp218_rmse"]),
        "worst_variant_rmse": float(worst[f"{variant_name}_rmse"]),
    }


# %% [markdown]
# ## 6. Models, OOF, feature importance, and SHA artifacts


# %%
def write_predictions(
    path: Path,
    cache: dict[str, Any],
    fold_assignment: np.ndarray,
    target_tvt: np.ndarray,
    baseline_tvt: np.ndarray,
    mixed_tvt: np.ndarray,
    early_tvt: np.ndarray,
    late_tvt: np.ndarray,
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
                    "mixed_exp244_pred_tvt": np.asarray(mixed_tvt[start:stop]),
                    "early_only_pred_tvt": early_tvt[start:stop],
                    "late_only_pred_tvt": late_tvt[start:stop],
                }
            )
            frame.to_csv(handle, index=False, header=start == 0, lineterminator="\n")


def save_importance_plot(importance: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    variants = ["early_only", "late_only"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    for axis, variant in zip(axes, variants, strict=True):
        top = (
            importance.loc[importance["variant"] == variant]
            .groupby("feature", as_index=False)["gain"]
            .mean()
            .nlargest(25, "gain")
        )
        axis.barh(top["feature"][::-1], top["gain"][::-1])
        axis.set_title(f"{variant} mean gain")
        axis.set_xlabel("mean gain")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# %% [markdown]
# ## 7. Execution orchestration


# %%
if not KAGGLE_INPUT_ROOT.exists():
    raise RuntimeError("exp260 training must run on Kaggle")

STARTED = time.time()
CONFIG_PATH, CONFIG = load_config()
VARIANTS = validate_approval_and_cost(CONFIG)
print(f"config={CONFIG_PATH}")
print(f"experiment={nested(CONFIG, 'experiment.name')}")
print(f"route={nested(CONFIG, 'experiment.route')}")
print(f"variants={[item['name'] for item in VARIANTS]}")
print(
    "approved_cost=variants=2 configs=3 folds=5 boosters=30 "
    "parent_control_retrained=False",
    flush=True,
)

OFFICIAL_CONTRACT, PSEUDO_CONTRACTS = parent.load_contracts(CONFIG)
CACHE = parent.stream_caches(CONFIG, OFFICIAL_CONTRACT, PSEUDO_CONTRACTS)
BASELINE_TVT = parent.load_frozen_exp218_oof(CACHE, CONFIG)
MIXED_TVT, MIXED_FOLD_ASSIGNMENT, MIXED_META = load_mixed_exp244_oof(
    CACHE, CONFIG, BASELINE_TVT
)
PARAMS_LIST = load_lgb_params(CONFIG)

RESULTS: dict[str, dict[str, Any]] = {}
ALL_TRAIN_METRICS: list[pd.DataFrame] = []
ALL_IMPORTANCE: list[pd.DataFrame] = []
ALL_MODELS: list[dict[str, Any]] = []
COMMON_FOLD_ASSIGNMENT: np.ndarray | None = None

for VARIANT in VARIANTS:
    VARIANT_NAME = str(VARIANT["name"])
    (
        MEAN_RESIDUAL,
        FOLD_ASSIGNMENT,
        TRAIN_METRICS,
        IMPORTANCE,
        MODELS,
    ) = train_variant(CACHE, CONFIG, VARIANT, PARAMS_LIST)
    if COMMON_FOLD_ASSIGNMENT is None:
        COMMON_FOLD_ASSIGNMENT = FOLD_ASSIGNMENT
    elif not np.array_equal(COMMON_FOLD_ASSIGNMENT, FOLD_ASSIGNMENT):
        raise AssertionError("Variant fold assignments differ")
    if not np.array_equal(MIXED_FOLD_ASSIGNMENT, FOLD_ASSIGNMENT):
        raise AssertionError("Matched folds differ from frozen exp244 mixed folds")
    METRICS, BY_WELL, EVALUATION, TARGET_TVT, VARIANT_TVT = parent.evaluate(
        CACHE, CONFIG, MEAN_RESIDUAL, FOLD_ASSIGNMENT, BASELINE_TVT
    )
    if EVALUATION["hidden_assignment"]["sha256"] != str(
        nested(CONFIG, "validation.expected_hidden_assignment_sha256")
    ):
        raise AssertionError("Hidden-like assignment SHA drift")
    RESULTS[VARIANT_NAME] = {
        "metrics": METRICS,
        "by_well": BY_WELL,
        "evaluation": EVALUATION,
        "target_tvt": TARGET_TVT,
        "variant_tvt": VARIANT_TVT,
    }
    ALL_TRAIN_METRICS.append(TRAIN_METRICS)
    ALL_IMPORTANCE.append(IMPORTANCE)
    ALL_MODELS.extend(MODELS)
    del MEAN_RESIDUAL, TRAIN_METRICS, IMPORTANCE, MODELS
    gc.collect()

if COMMON_FOLD_ASSIGNMENT is None or len(ALL_MODELS) != int(
    nested(CONFIG, "model.attribution.training.total_boosters")
):
    raise AssertionError("Matched training did not produce 30 models")

MIXED_RESIDUAL = np.asarray(MIXED_TVT) - np.asarray(CACHE["official_base"])
MIXED_METRICS, MIXED_BY_WELL, MIXED_EVALUATION, _, MIXED_REBUILT_TVT = parent.evaluate(
    CACHE, CONFIG, MIXED_RESIDUAL, COMMON_FOLD_ASSIGNMENT, BASELINE_TVT
)
if MIXED_EVALUATION["hidden_assignment"]["sha256"] != str(
    nested(CONFIG, "validation.expected_hidden_assignment_sha256")
):
    raise AssertionError("Mixed hidden-like assignment SHA drift")
if not np.allclose(MIXED_REBUILT_TVT, np.asarray(MIXED_TVT), atol=0.0001):
    raise AssertionError("Mixed OOF residual reconstruction mismatch")

METRICS = combine_metrics(
    MIXED_METRICS,
    RESULTS["early_only"]["metrics"],
    RESULTS["late_only"]["metrics"],
)
BY_WELL = combine_by_well(
    MIXED_BY_WELL,
    RESULTS["early_only"]["by_well"],
    RESULTS["late_only"]["by_well"],
)
TRAIN_METRICS = pd.concat(ALL_TRAIN_METRICS, ignore_index=True)
IMPORTANCE = pd.concat(ALL_IMPORTANCE, ignore_index=True)

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
    COMMON_FOLD_ASSIGNMENT,
    RESULTS["early_only"]["target_tvt"],
    BASELINE_TVT,
    MIXED_TVT,
    RESULTS["early_only"]["variant_tvt"],
    RESULTS["late_only"]["variant_tvt"],
)

MODEL_MANIFEST = {
    "experiment": EXPERIMENT_NAME,
    "mode": nested(CONFIG, "model.attribution.training.mode"),
    "model_count": len(ALL_MODELS),
    "models": ALL_MODELS,
}
MODEL_MANIFEST_PATH.write_text(json.dumps(MODEL_MANIFEST, indent=2, sort_keys=True) + "\n")

METRIC_LOOKUP = METRICS.set_index("surface")
VARIANT_SUMMARIES: dict[str, Any] = {}
for VARIANT_NAME in ["early_only", "late_only"]:
    VARIANT_SUMMARIES[VARIANT_NAME] = {
        "oof_rmse": float(METRIC_LOOKUP.at["overall", f"{VARIANT_NAME}_rmse"]),
        "delta_vs_raw": float(
            METRIC_LOOKUP.at["overall", f"{VARIANT_NAME}_delta_vs_raw"]
        ),
        "delta_vs_mixed": float(METRIC_LOOKUP.at["overall", f"{VARIANT_NAME}_rmse"])
        - float(METRIC_LOOKUP.at["overall", "mixed_exp244_rmse"]),
        "evaluation": RESULTS[VARIANT_NAME]["evaluation"],
        "by_well": by_well_summary(BY_WELL, VARIANT_NAME),
    }

SUMMARY = {
    "experiment": EXPERIMENT_NAME,
    "status": "matched_early_late_attribution_complete",
    "route": nested(CONFIG, "experiment.route"),
    "raw_exp218_oof_rmse": float(METRIC_LOOKUP.at["overall", "raw_exp218_rmse"]),
    "mixed_exp244_oof_rmse": float(METRIC_LOOKUP.at["overall", "mixed_exp244_rmse"]),
    "variants": VARIANT_SUMMARIES,
    "attribution": {
        "late_minus_early_overall": float(METRIC_LOOKUP.at["overall", "late_minus_early"]),
        "late_independent_compensation_supported": bool(
            RESULTS["late_only"]["evaluation"]["guards"]["adoption_supported"]
        ),
        "mixed_reference_guards": MIXED_EVALUATION["guards"],
    },
    "execution": {
        "active_variants": 2,
        "lightgbm_configs": 3,
        "folds": 5,
        "boosters": 30,
        "parent_control_retrained": False,
        "elapsed_seconds": time.time() - STARTED,
        "peak_rss_mb": parent.peak_rss_mb(),
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
        "frozen_exp244_mixed": MIXED_META,
    },
    "artifacts": {
        "training_metrics_sha256": sha256_file(TRAIN_METRICS_PATH),
        "metrics_sha256": sha256_file(METRICS_PATH),
        "by_well_sha256": sha256_file(BY_WELL_PATH),
        "feature_importance_sha256": sha256_file(IMPORTANCE_PATH),
        "feature_schema_sha256": sha256_file(SCHEMA_PATH),
        "model_manifest_sha256": sha256_file(MODEL_MANIFEST_PATH),
        "prediction_decompressed_sha256": sha256_file(
            PREDICTIONS_PATH, decompressed=True
        ),
    },
    "inference_prediction_performed": False,
    "submission_created": False,
}
SUMMARY_PATH = KAGGLE_WORKING_ROOT / f"{OUTPUT_PREFIX}_summary.json"
SUMMARY_PATH.write_text(json.dumps(SUMMARY, indent=2, sort_keys=True) + "\n")

print(METRICS.to_string(index=False), flush=True)
for VARIANT_NAME in ["early_only", "late_only"]:
    print(f"worst wells: {VARIANT_NAME}", flush=True)
    print(
        BY_WELL.nlargest(10, f"{VARIANT_NAME}_delta_vs_raw")[
            [
                "well_id",
                "raw_exp218_rmse",
                f"{VARIANT_NAME}_rmse",
                f"{VARIANT_NAME}_delta_vs_raw",
            ]
        ].to_string(index=False),
        flush=True,
    )
print(json.dumps(SUMMARY, indent=2, sort_keys=True), flush=True)

MIXED_TVT.flush()
MIXED_TVT._mmap.close()
parent.close_and_cleanup(CACHE, BASELINE_TVT)
TRAINING_SUMMARY = SUMMARY

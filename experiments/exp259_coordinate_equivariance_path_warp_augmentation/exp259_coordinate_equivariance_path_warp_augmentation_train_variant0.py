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
# # exp259 coordinate equivariance path warp augmentation — full-well train
#
# Train one exact TVT-datum augmentation variant on all 773 train wells. The selected
# 295-feature schema is pinned to the completed exp251 version-3 feature audit. Clean
# outer-valid wells are never transformed. `md_stretch` and every other approximate path
# warp stay disabled because their PF/HMM/geometry candidates and absolute spatial priors
# are not regenerated in this compute contract.

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Stage, cost, and transform contract
# 3. Pinned exp251 schema and optional clean control
# 4. Fixed eleven-candidate full-well surface
# 5. Exact datum augmentation and five-fold training
# 6. Clean OOF metrics and control comparison
# 7. Artifacts and SHA evidence

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from sklearn.model_selection import GroupKFold

from src.exact_datum_ranker_augmentation import (
    assign_stable_tvt_shifts,
    build_exact_tvt_datum_long_view,
    select_stable_wells,
)

EXPERIMENT_NAME = "exp259_coordinate_equivariance_path_warp_augmentation"
OUTPUT_PREFIX = f"{EXPERIMENT_NAME}_training"


def find_package_dir() -> Path:
    candidates = [Path.cwd(), Path.cwd() / "experiments" / EXPERIMENT_NAME]
    for candidate in candidates:
        path = candidate / "config.yaml"
        if path.exists() and EXPERIMENT_NAME in path.read_text():
            return candidate.resolve()
    raise FileNotFoundError(f"could not locate package for {EXPERIMENT_NAME}")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    import gzip

    digest = hashlib.sha256()
    opener = gzip.open if decompressed and path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def find_kaggle_input_file(filename: str, *, required: bool) -> Path | None:
    input_root = Path("/kaggle/input")
    matches = sorted(input_root.rglob(filename)) if input_root.exists() else []
    if not matches:
        local_matches = sorted(Path("/tmp").rglob(filename)) if Path("/tmp").exists() else []
        matches = local_matches
    if not matches:
        if required:
            raise FileNotFoundError(f"required Kaggle input artifact not found: {filename}")
        return None
    preferred = [path for path in matches if "exp251" in str(path).lower()]
    return (preferred or matches)[0]


PACKAGE_DIR = find_package_dir()
CONFIG_PATH = PACKAGE_DIR / "config.yaml"
OUTPUT_DIR = (
    Path("/kaggle/working/artifacts")
    if Path("/kaggle/working").exists()
    else PACKAGE_DIR / "artifacts"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config = read_yaml(CONFIG_PATH)

# The bootstrap keeps the parent package in an isolated directory. Import it before any
# module named `settings`, so candidate_ranker_engine resolves exp251's own config.
PARENT_PACKAGE_DIR = PACKAGE_DIR / "parent_exp251"
if not PARENT_PACKAGE_DIR.exists():
    local_parent = PACKAGE_DIR.parent / "exp251_raw_test_safe_dual_objective_candidate_ranker"
    PARENT_PACKAGE_DIR = local_parent if local_parent.exists() else PARENT_PACKAGE_DIR
sys.path.insert(0, str(PARENT_PACKAGE_DIR))
import candidate_ranker_engine as parent_engine  # noqa: E402
from settings import get_nested as parent_get_nested  # noqa: E402
from settings import load_config as load_parent_config  # noqa: E402

parent_config = load_parent_config()

# %% [markdown]
# ## 2. Stage, cost, and transform contract
#
# This is one new variant, two LightGBM objectives, five group folds, and ten CPU
# boosters. All wells enter clean GroupKFold OOF. A stable 25% well subset receives one
# additional exact datum-shift view in outer-train only. The separately running exp251
# 295-feature model remains the clean control and is not retrained here.

# %%
stage = str(get_nested(config, "execution.stage"))
training = dict(get_nested(config, "augmentation.training", {}))
variant = str(training["variant"])
seed = int(get_nested(config, "validation.seed", 42))
n_folds = int(get_nested(config, "validation.n_folds", 5))
cost_contract = {
    "experiment": EXPERIMENT_NAME,
    "route": get_nested(config, "experiment.route"),
    "stage": stage,
    "all_clean_wells": True,
    "active_variants": [variant],
    "objectives": list(get_nested(config, "model.estimator", [])),
    "folds": n_folds,
    "lightgbm_configs": int(get_nested(config, "model.planned_lightgbm_configs", 0)),
    "boosters": int(get_nested(config, "model.planned_boosters", 0)),
    "control_retraining": bool(get_nested(config, "model.control_retraining")),
    "parent_retraining": bool(get_nested(config, "model.parent_retraining")),
    "gpu": bool(get_nested(config, "runtime.kaggle.enable_gpu")),
    "internet": bool(get_nested(config, "runtime.kaggle.enable_internet")),
}
display(cost_contract)

assert stage == "train_exact_datum_after_transform_audit"
assert variant == "exact_tvt_datum_shift"
assert n_folds == 5
assert cost_contract["lightgbm_configs"] == 2
assert cost_contract["boosters"] == 10
assert cost_contract["control_retraining"] is False
assert cost_contract["parent_retraining"] is False
assert cost_contract["gpu"] is False
assert cost_contract["internet"] is False
assert training["eligible_exact_transforms"] == ["tvt_datum_shift"]
assert "md_stretch" in training["disabled_transforms"]
assert set(training["disabled_transforms"]) == set(
    get_nested(config, "augmentation.approximate_transforms")
)

print("Leakage contract")
for rule in get_nested(config, "validation.leakage_policy", []):
    print("-", rule)

# %% [markdown]
# ## 3. Pinned exp251 schema and optional clean control
#
# The exp251 version-3 feature audit is a completed, immutable 295-column contract. Its
# schema SHA is required. The version-4 clean metrics are optional at launch so both
# notebooks can run concurrently; when present they are loaded only for the final
# comparison and never affect folds, rows, shifts, labels, or model fitting.

# %%
schema_name = str(get_nested(config, "data.exp251_selected_feature_schema_name"))
schema_path = find_kaggle_input_file(schema_name, required=True)
assert schema_path is not None
schema_sha = sha256_path(schema_path)
expected_schema_sha = str(training["selected_feature_schema_sha256"])
if schema_sha != expected_schema_sha:
    raise AssertionError(
        f"exp251 selected schema SHA mismatch: {schema_sha} != {expected_schema_sha}"
    )
selected_features = (
    pd.read_csv(schema_path).sort_values("feature_order")["feature"].astype(str).tolist()
)
if len(selected_features) != int(training["expected_selected_feature_count"]):
    raise AssertionError(f"expected 295 selected features, found {len(selected_features)}")
if len(set(selected_features)) != len(selected_features):
    raise AssertionError("selected feature schema contains duplicates")

control_metrics_name = str(get_nested(config, "data.exp251_metrics_name"))
control_metrics_path = find_kaggle_input_file(control_metrics_name, required=False)
control_metrics_sha = sha256_path(control_metrics_path) if control_metrics_path else None
display(
    {
        "selected_schema": str(schema_path),
        "selected_feature_count": len(selected_features),
        "selected_schema_sha256": schema_sha,
        "clean_control_metrics_available": control_metrics_path is not None,
        "clean_control_metrics": str(control_metrics_path) if control_metrics_path else None,
    }
)

# %% [markdown]
# ## 4. Fixed eleven-candidate full-well surface
#
# Rebuild the same exp251 train-side candidate surface from fixed upstream OOF artifacts.
# No PF, Beam, HMM, geometry, exp218, or exp251 model is fit. All 773 wells and all clean
# OOF rows remain in evaluation; parent row caps only bound each fold's LightGBM fitting
# and early-stopping matrices.

# %%
started = time.time()
(
    frame,
    candidates,
    candidate_values,
    _oracle_labels,
    base_feature_columns,
    source_meta,
) = parent_engine.assemble_parent_candidate_surface(
    cache_path=parent_get_nested(parent_config, "data.exp099_train_feature_cache_local"),
    schema_path=parent_get_nested(parent_config, "data.exp099_train_feature_schema_local"),
    max_rows=None,
)
candidate_names = [item.name for item in candidates]
well_ids = sorted(frame["well"].astype(str).unique())
input_contract = {
    "rows": int(len(frame)),
    "wells": int(len(well_ids)),
    "candidate_count": int(len(candidates)),
    "candidates": candidate_names,
    "base_feature_count": int(len(base_feature_columns)),
    "source_meta": source_meta,
}
display(input_contract)
assert len(well_ids) == int(training["expected_well_count"])
assert len(candidates) == 11
assert len(frame) == len(candidate_values)
assert np.isfinite(candidate_values).all()

# %% [markdown]
# ## 5. Exact datum augmentation and five-fold training
#
# The global synthetic subset and each well's shift are SHA256-derived before folding.
# Every fold uses the exp251 row-sampling and LightGBM seed namespaces. Clean rows are
# retained; selected outer-train rows are duplicated and only seven absolute TVT features
# are shifted. Candidate error, within-10 label, and the other 288 selected features must
# remain byte-equivalent before a booster can fit.

# %%
from lightgbm import (  # noqa: E402
    LGBMClassifier,
    LGBMRegressor,
    early_stopping,
    log_evaluation,
)

synthetic_wells = select_stable_wells(
    well_ids,
    fraction=float(training["synthetic_well_fraction"]),
    seed=seed,
    namespace=f"{EXPERIMENT_NAME}:synthetic_wells",
)
shift_grid = [
    float(value)
    for value in get_nested(config, "augmentation.parameter_grid.tvt_datum_shift.shift_ft")
]
shift_by_well = assign_stable_tvt_shifts(
    synthetic_wells,
    shift_grid_ft=shift_grid,
    seed=seed,
    namespace=f"{EXPERIMENT_NAME}:tvt_datum_shift",
)
display(
    {
        "total_wells": len(well_ids),
        "synthetic_wells": len(synthetic_wells),
        "synthetic_fraction": len(synthetic_wells) / len(well_ids),
        "shift_counts": pd.Series(shift_by_well).value_counts().sort_index().to_dict(),
    }
)


def train_exact_datum_outer_oof() -> tuple[
    dict[str, dict[str, np.ndarray]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    pd.DataFrame,
]:
    folds = list(GroupKFold(n_splits=n_folds).split(frame, groups=frame["well"]))
    train_limit = int(
        parent_get_nested(parent_config, "augmentation.max_train_base_rows_per_fold") or 60000
    )
    valid_limit = int(
        parent_get_nested(parent_config, "augmentation.max_valid_base_rows_for_early_stopping")
        or 30000
    )
    log_period = int(parent_get_nested(parent_config, "ranker.log_period") or 100)
    classifier_params = dict(
        parent_get_nested(parent_config, "ranker.long_models.binary_lgbm.params") or {}
    )
    error_params = dict(
        parent_get_nested(parent_config, "ranker.long_models.error_lgbm.params") or {}
    )
    parent_seed_variant = str(get_nested(config, "runtime.deterministic.parent_model_seed_variant"))
    absolute_columns = [str(value) for value in training["absolute_tvt_feature_columns"]]
    tolerance = float(training["exact_delta_tolerance"])

    oof = {
        variant: {
            "probability": np.full((len(frame), len(candidates)), np.nan, dtype=np.float32),
            "predicted_error": np.full((len(frame), len(candidates)), np.nan, dtype=np.float32),
        }
    }
    inventory_parts: list[pd.DataFrame] = []
    guard_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    models_dir = OUTPUT_DIR / f"{OUTPUT_PREFIX}_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    feature_schema: list[str] | None = None

    for fold, (train_idx, valid_idx) in enumerate(folds):
        sampled_train = parent_engine._sample_sorted_rows(
            np.asarray(train_idx, dtype=np.int64),
            train_limit,
            seed=parent_engine.stable_seed(parent_engine.OUTPUT_PREFIX, "train_rows", fold, seed),
        )
        sampled_valid = parent_engine._sample_sorted_rows(
            np.asarray(valid_idx, dtype=np.int64),
            valid_limit,
            seed=parent_engine.stable_seed(parent_engine.OUTPUT_PREFIX, "valid_rows", fold, seed),
        )
        clean_values = candidate_values[sampled_train]
        clean_available = np.ones_like(clean_values, dtype=bool)
        clean_long, clean_error, clean_binary, clean_schema = (
            parent_engine.build_candidate_long_view(
                frame.iloc[sampled_train],
                clean_values,
                clean_available,
                candidates=candidates,
                base_feature_columns=base_feature_columns,
                config=parent_config,
                raw_cache=raw_cache,
            )
        )
        valid_values = candidate_values[sampled_valid]
        valid_long, _valid_error, valid_binary, valid_schema = (
            parent_engine.build_candidate_long_view(
                frame.iloc[sampled_valid],
                valid_values,
                np.ones_like(valid_values, dtype=bool),
                candidates=candidates,
                base_feature_columns=base_feature_columns,
                config=parent_config,
                raw_cache=raw_cache,
            )
        )
        if valid_schema != clean_schema:
            raise AssertionError("clean train and early-stop schemas differ")
        missing = [column for column in selected_features if column not in clean_schema]
        if missing:
            raise AssertionError(f"selected exp251 features are missing: {missing[:20]}")
        if feature_schema is None:
            feature_schema = list(selected_features)
        elif feature_schema != selected_features:
            raise AssertionError("selected schema changed across folds")

        outer_train_wells = set(frame.iloc[train_idx]["well"].astype(str))
        fold_shifts = {
            well: shift for well, shift in shift_by_well.items() if well in outer_train_wells
        }
        augmented_long, augmented_error, augmented_binary, guard = build_exact_tvt_datum_long_view(
            clean_long,
            clean_error,
            clean_binary,
            shift_by_well=fold_shifts,
            absolute_tvt_feature_columns=absolute_columns,
            selected_feature_columns=selected_features,
            tolerance=tolerance,
        )
        guard_rows.append({"fold": fold, **guard})
        train_long = pd.concat([clean_long, augmented_long], ignore_index=True)
        train_error = np.concatenate([clean_error, augmented_error])
        train_binary = np.concatenate([clean_binary, augmented_binary])

        sampled_base = frame.iloc[sampled_train][["id", "well"]].copy()
        sampled_base["well"] = sampled_base["well"].astype(str)
        sampled_base = sampled_base[sampled_base["well"].isin(fold_shifts)].copy()
        sampled_base["fold"] = np.int16(fold)
        sampled_base["variant"] = variant
        sampled_base["transform"] = "tvt_datum_shift"
        sampled_base["shift_ft"] = sampled_base["well"].map(fold_shifts).astype(np.float32)
        sampled_base["outer_valid"] = False
        inventory_parts.append(sampled_base)

        x_train, medians = parent_engine._fit_imputer(train_long, selected_features)
        x_valid = parent_engine._apply_imputer(valid_long, selected_features, medians)
        classifier = LGBMClassifier(
            objective="binary",
            random_state=parent_engine.stable_seed(
                parent_engine.OUTPUT_PREFIX, parent_seed_variant, "classifier", fold, seed
            ),
            **classifier_params,
        )
        classifier.fit(
            x_train,
            train_binary,
            eval_set=[(x_valid, valid_binary)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(50), log_evaluation(log_period)],
        )
        error_model = LGBMRegressor(
            objective="regression_l1",
            random_state=parent_engine.stable_seed(
                parent_engine.OUTPUT_PREFIX, parent_seed_variant, "error", fold, seed
            ),
            **error_params,
        )
        error_model.fit(
            x_train,
            train_error,
            eval_set=[(x_valid, _valid_error)],
            eval_metric="l1",
            callbacks=[early_stopping(50), log_evaluation(log_period)],
        )

        classifier_path = models_dir / f"{variant}_within10_classifier_fold{fold}.txt"
        error_path = models_dir / f"{variant}_expected_error_fold{fold}.txt"
        imputer_path = models_dir / f"{variant}_imputer_medians_fold{fold}.npy"
        classifier.booster_.save_model(str(classifier_path))
        error_model.booster_.save_model(str(error_path))
        np.save(imputer_path, medians)
        for objective, model, path in [
            ("within10_classifier", classifier, classifier_path),
            ("expected_error_regressor", error_model, error_path),
        ]:
            manifest_rows.append(
                {
                    "variant": variant,
                    "objective": objective,
                    "fold": fold,
                    "path": str(path.relative_to(OUTPUT_DIR)),
                    "sha256": sha256_path(path),
                    "imputer_path": str(imputer_path.relative_to(OUTPUT_DIR)),
                    "imputer_sha256": sha256_path(imputer_path),
                    "best_iteration": int(model.best_iteration_ or model.n_estimators),
                    "clean_train_base_rows": int(len(sampled_train)),
                    "clean_train_long_rows": int(len(clean_long)),
                    "augmented_train_long_rows": int(len(augmented_long)),
                    "combined_train_long_rows": int(len(train_long)),
                    "valid_base_rows_for_early_stopping": int(len(sampled_valid)),
                    "valid_long_rows_for_early_stopping": int(len(valid_long)),
                    "seed_namespace_variant": parent_seed_variant,
                }
            )
            for feature, importance in zip(
                selected_features, model.feature_importances_, strict=True
            ):
                importance_rows.append(
                    {
                        "variant": variant,
                        "objective": objective,
                        "fold": fold,
                        "feature": feature,
                        "importance": float(importance),
                    }
                )

        probability, predicted_error = parent_engine._predict_clean_validation(
            frame=frame,
            valid_idx=np.asarray(valid_idx, dtype=np.int64),
            candidate_values=candidate_values,
            candidates=candidates,
            base_feature_columns=base_feature_columns,
            config=parent_config,
            raw_cache=raw_cache,
            feature_columns=selected_features,
            medians=medians,
            classifier=classifier,
            error_model=error_model,
        )
        oof[variant]["probability"][valid_idx] = probability
        oof[variant]["predicted_error"][valid_idx] = predicted_error
        print(
            f"fold={fold} clean={len(clean_long):,} augmented={len(augmented_long):,} "
            f"valid_full={len(valid_idx):,} classifier_iter={classifier.best_iteration_} "
            f"error_iter={error_model.best_iteration_}"
        )
        del x_train, x_valid, train_long, augmented_long, classifier, error_model
        gc.collect()

    assert feature_schema is not None
    if len(manifest_rows) != 10:
        raise AssertionError(f"expected 10 boosters, found {len(manifest_rows)}")
    for scores in oof.values():
        if not np.isfinite(scores["probability"]).all():
            raise AssertionError("OOF probability contains non-finite values")
        if not np.isfinite(scores["predicted_error"]).all():
            raise AssertionError("OOF predicted error contains non-finite values")
    return (
        oof,
        pd.concat(inventory_parts, ignore_index=True),
        pd.DataFrame(guard_rows),
        pd.DataFrame(manifest_rows),
        feature_schema,
        pd.DataFrame(importance_rows),
    )


(
    oof_scores,
    augmentation_inventory,
    equivariance_guards,
    model_manifest,
    feature_schema,
    feature_importance,
) = train_exact_datum_outer_oof()
display(equivariance_guards)
display(model_manifest)

# %% [markdown]
# ## 6. Clean OOF metrics and control comparison
#
# All metrics below score untouched outer-valid wells. If exp251 version-4 clean metrics
# were already complete when this kernel started, an exact control delta is recorded. If
# not, the run still finishes and is marked `pending_clean_control_comparison`; the saved
# exp259 OOF does not need retraining after exp251 completes.

# %%
results = parent_engine.evaluate_oof(
    frame=frame,
    candidates=candidates,
    candidate_values=candidate_values,
    oof_scores=oof_scores,
    config=parent_config,
)
mode = "expected_error_fixed_viterbi"
selected_metric = results["metrics"].query("variant == @variant and mode == @mode").iloc[0]
distance_1000 = (
    results["bucket_metrics"]
    .query(
        "variant == @variant and mode == @mode "
        "and bucket_family == 'distance_bucket' and bucket == '1000_plus'"
    )
    .iloc[0]
)
spatial = (
    results["subgroup_metrics"]
    .query("variant == @variant and mode == @mode and subgroup == 'exp115_spatial_valid'")
    .iloc[0]
)
typewell = (
    results["subgroup_metrics"]
    .query("variant == @variant and mode == @mode and subgroup == 'exp115_typewell_purged_valid'")
    .iloc[0]
)
worst = (
    results["by_well"]
    .query("variant == @variant and mode == @mode")
    .sort_values("rmse_tvt", ascending=False)
    .iloc[0]
)

control_reference: dict[str, Any] = {
    "available": False,
    "variant": "raw_test_regenerated_copcf",
    "mode": mode,
    "metrics_path": str(control_metrics_path) if control_metrics_path else None,
    "metrics_sha256": control_metrics_sha,
}
if control_metrics_path is not None:
    control_table = pd.read_csv(control_metrics_path)
    control_rows = control_table.query("variant == 'raw_test_regenerated_copcf' and mode == @mode")
    if len(control_rows) == 1:
        control_row = control_rows.iloc[0]
        control_reference.update(
            {
                "available": True,
                "rmse_tvt": float(control_row["rmse_tvt"]),
                "delta_rmse_tvt": float(selected_metric["rmse_tvt"] - control_row["rmse_tvt"]),
            }
        )

criteria = dict(parent_get_nested(parent_config, "audit.success_criteria") or {})
checks = {
    "equivariance_guards_pass": bool(equivariance_guards["pass"].all()),
    "model_count_exact": len(model_manifest) == 10,
    "feature_schema_exact": feature_schema == selected_features,
    "all_wells_clean_oof": int(frame["well"].nunique()) == int(training["expected_well_count"]),
    "overall_vs_exp218": float(selected_metric["rmse_tvt"]) <= float(criteria["max_selected_rmse"]),
    "distance_1000_plus": float(distance_1000["rmse_tvt"])
    <= float(criteria["max_distance_1000_plus_rmse"]),
    "hidden_like_spatial": float(spatial["rmse_tvt"])
    <= float(criteria["max_hidden_like_spatial_rmse"]),
    "hidden_like_typewell_purged": float(typewell["rmse_tvt"])
    <= float(criteria["max_hidden_like_typewell_purged_rmse"]),
    "worst_well": float(worst["rmse_tvt"]) <= float(criteria["max_worst_well_rmse"]),
}
if control_reference["available"]:
    checks["nonworse_than_exp251_clean_control"] = float(control_reference["delta_rmse_tvt"]) <= 0.0

decision = {
    "clean_control_comparison_available": bool(control_reference["available"]),
    "adoption_supported": bool(control_reference["available"] and all(checks.values())),
    "checks": checks,
    "selected_rmse": float(selected_metric["rmse_tvt"]),
    "distance_1000_plus_rmse": float(distance_1000["rmse_tvt"]),
    "hidden_like_spatial_rmse": float(spatial["rmse_tvt"]),
    "hidden_like_typewell_purged_rmse": float(typewell["rmse_tvt"]),
    "worst_well": str(worst["well"]),
    "worst_well_rmse": float(worst["rmse_tvt"]),
    "control_reference": control_reference,
}
display(results["metrics"])
display(results["candidate_metrics"])
display(results["bucket_metrics"].query("bucket == '1000_plus'"))
display(results["subgroup_metrics"].query("subgroup.str.startswith('exp115_')", engine="python"))
display(
    results["by_well"]
    .query("variant == @variant and mode == @mode")
    .sort_values("rmse_tvt", ascending=False)
    .head(80)
)
display(decision)

importance_mean = (
    feature_importance.groupby(["variant", "objective", "feature"], as_index=False)["importance"]
    .mean()
    .sort_values(["variant", "objective", "importance"], ascending=[True, True, False])
)
top_importance = importance_mean.query("objective == 'expected_error_regressor'").head(30)
if len(top_importance):
    ax = top_importance.sort_values("importance").plot.barh(
        x="feature",
        y="importance",
        figsize=(9, 9),
        legend=False,
        title="exp259 exact-datum expected-error importance",
    )
    ax.set_xlabel("mean LightGBM split importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_importance_top.png", dpi=140)
    plt.show()

# %% [markdown]
# ## 7. Artifacts and SHA evidence

# %%
artifact_paths: dict[str, Path] = {
    "metrics": OUTPUT_DIR / f"{OUTPUT_PREFIX}_metrics.csv",
    "candidate_metrics": OUTPUT_DIR / f"{OUTPUT_PREFIX}_candidate_metrics.csv",
    "calibration": OUTPUT_DIR / f"{OUTPUT_PREFIX}_calibration.csv",
    "topk_coverage": OUTPUT_DIR / f"{OUTPUT_PREFIX}_topk_coverage.csv",
    "margin_calibration": OUTPUT_DIR / f"{OUTPUT_PREFIX}_margin_calibration.csv",
    "bucket_metrics": OUTPUT_DIR / f"{OUTPUT_PREFIX}_bucket_metrics.csv",
    "subgroup_metrics": OUTPUT_DIR / f"{OUTPUT_PREFIX}_subgroup_metrics.csv",
    "by_well": OUTPUT_DIR / f"{OUTPUT_PREFIX}_by_well.csv",
    "oof_predictions": OUTPUT_DIR / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz",
    "augmentation_inventory": OUTPUT_DIR / f"{OUTPUT_PREFIX}_augmentation_inventory.csv",
    "equivariance_guards": OUTPUT_DIR / f"{OUTPUT_PREFIX}_equivariance_guards.csv",
    "feature_importance_mean": OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_importance_mean.csv",
    "feature_schema": OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_schema.csv",
    "model_manifest": OUTPUT_DIR / f"{OUTPUT_PREFIX}_model_manifest.json",
    "control_reference": OUTPUT_DIR / f"{OUTPUT_PREFIX}_control_reference.json",
    "summary": OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.json",
}
for key in [
    "metrics",
    "candidate_metrics",
    "calibration",
    "topk_coverage",
    "margin_calibration",
    "bucket_metrics",
    "subgroup_metrics",
    "by_well",
]:
    results[key].to_csv(artifact_paths[key], index=False)
results["predictions"].to_csv(artifact_paths["oof_predictions"], index=False, compression="gzip")
augmentation_inventory.to_csv(artifact_paths["augmentation_inventory"], index=False)
equivariance_guards.to_csv(artifact_paths["equivariance_guards"], index=False)
importance_mean.to_csv(artifact_paths["feature_importance_mean"], index=False)
pd.DataFrame({"feature_order": np.arange(len(feature_schema)), "feature": feature_schema}).to_csv(
    artifact_paths["feature_schema"], index=False
)
write_json(artifact_paths["model_manifest"], {"models": model_manifest.to_dict("records")})
write_json(artifact_paths["control_reference"], control_reference)

summary = {
    "experiment": EXPERIMENT_NAME,
    "status": (
        "completed_train_side_adoption_supported"
        if decision["adoption_supported"]
        else "completed_train_side_guard_failed"
        if control_reference["available"]
        else "completed_pending_clean_control_comparison"
    ),
    "stage": stage,
    "route": get_nested(config, "experiment.route"),
    "runtime_seconds": time.time() - started,
    "rows": int(len(frame)),
    "wells": int(frame["well"].nunique()),
    "candidate_count": len(candidates),
    "selected_feature_count": len(feature_schema),
    "synthetic_well_count": len(synthetic_wells),
    "synthetic_well_fraction": len(synthetic_wells) / len(well_ids),
    "active_variants": 1,
    "model_configs": 2,
    "folds": n_folds,
    "boosters": len(model_manifest),
    "control_retraining": False,
    "parent_retraining": False,
    "disabled_approximate_transforms": list(training["disabled_transforms"]),
    "md_stretch_excluded": "md_stretch" in training["disabled_transforms"],
    "selected_schema_source": str(schema_path),
    "selected_schema_sha256": schema_sha,
    "source_meta": source_meta,
    "decision": decision,
    "metrics": results["metrics"].to_dict("records"),
    "candidate_metrics": results["candidate_metrics"].to_dict("records"),
    "artifacts": {key: path.name for key, path in artifact_paths.items()},
}
summary["sha256"] = {
    key: sha256_path(path, decompressed=path.suffix == ".gz")
    for key, path in artifact_paths.items()
    if key != "summary"
}
write_json(artifact_paths["summary"], summary)
summary_sha = sha256_path(artifact_paths["summary"])

print("Generated artifacts")
for key, path in artifact_paths.items():
    print(f"{key}: {path.name}")
print("summary_sha256:", summary_sha)
print("OOF gzip uses decompressed-content SHA:", summary["sha256"]["oof_predictions"])

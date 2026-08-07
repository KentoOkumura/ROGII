# %% [markdown]
# # exp261 LightGBM extra-trees ablation on exp218 train
#
# The frozen exp218 feature surface, folds, seeds, and LightGBM parameters are
# retained. Selected configs add only `extra_trees=True`; saved exp218 boosters
# provide the matched control without retraining it.

# %% [markdown]
# ## Contents
# 1. Imports and source resolution
# 2. Configuration and GPU-cost approval gate
# 3. Frozen exp218 input contracts
# 4. Rebuild the exp218 380-feature surface
# 5. Single-parameter LightGBM contract
# 6. Matched control inference and extra-trees CV
# 7. Stress, blend, and adoption readouts
# 8. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports and source resolution

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from sklearn.model_selection import GroupKFold

PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path(
        "experiments/exp261_lightgbm_extra_trees_ablation_on_exp218"
    )
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_DIR = (
    Path("/kaggle/working/artifacts")
    if Path("/kaggle/working").exists()
    else PACKAGE_DIR / "artifacts"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = str(CONFIG["audit"]["output_prefix"])


def import_file(name: str, candidates: list[Path]):
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Cannot resolve {name}: {candidates}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.NOTEBOOK_SOURCE_PATH = path
    return module


exp218 = import_file(
    "exp218_source",
    [
        PACKAGE_DIR / "exp218_source/gr_wavelet_rotation_confidence_features_on_exp148.py",
        Path(
            "experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/"
            "gr_wavelet_rotation_confidence_features_on_exp148.py"
        ),
        Path(
            "/kaggle/input/exp218-gr-wavelet-rotation-exp148-train/"
            "gr_wavelet_rotation_confidence_features_on_exp148.py"
        ),
    ],
)
exp218_settings = import_file(
    "exp218_settings_source",
    [
        PACKAGE_DIR / "exp218_source/settings.py",
        Path(
            "experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/"
            "settings.py"
        ),
    ],
)


def nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    hasher = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    if decompressed:
        stream = opener(path, "rb")
    else:
        stream = opener(path, "rb")
    with stream as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_artifact(
    filename: str,
    *,
    explicit: str | Path | None = None,
    expected_sha256: str | None = None,
    decompressed: bool = False,
) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([PACKAGE_DIR / filename, PACKAGE_DIR / "inputs" / filename])
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.rglob(filename))
    checked: list[str] = []
    for path in candidates:
        if not path.exists() or path.stat().st_size == 0:
            continue
        if expected_sha256 is not None:
            actual = sha256_path(path, decompressed=decompressed)
            checked.append(f"{path}: {actual}")
            if actual != expected_sha256:
                continue
        return path
    raise FileNotFoundError(
        f"Cannot resolve {filename} with the required SHA. Checked: {checked[:30]}"
    )


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(target64 - prediction64))))


def feature_content_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    hasher = hashlib.sha256()
    chunk_rows = 100_000
    ids = frame["id"].astype(str)
    for start in range(0, len(frame), chunk_rows):
        payload = "\n".join(ids.iloc[start : start + chunk_rows].tolist()).encode()
        hasher.update(payload)
        hasher.update(b"\n")
    for column in columns:
        hasher.update(column.encode())
        hasher.update(b"\0")
        values = frame[column].to_numpy(dtype=np.float32, copy=False)
        for start in range(0, len(values), chunk_rows):
            hasher.update(values[start : start + chunk_rows].tobytes())
    return hasher.hexdigest()


STARTED = time.time()

# %% [markdown]
# ## 2. Configuration and GPU-cost approval gate

# %%
plans = nested(CONFIG, "model.plans", {})
selected_plan_name = nested(CONFIG, "model.selected_plan")
run_approved = bool(nested(CONFIG, "model.run_approved", False))
print(
    json.dumps(
        {
            "experiment": CONFIG["experiment"],
            "route": nested(CONFIG, "experiment.route"),
            "parent": nested(CONFIG, "lineage.parent"),
            "changed_parameter": nested(CONFIG, "model.changed_parameter"),
            "available_plans": plans,
            "selected_plan": selected_plan_name,
            "run_approved": run_approved,
            "parent_control_retraining": nested(
                CONFIG, "model.variant.control_retraining"
            ),
        },
        indent=2,
    )
)

if selected_plan_name not in plans:
    raise RuntimeError(
        "Kaggle train is fail-closed: choose model.selected_plan from "
        f"{sorted(plans)} after user approval."
    )
if not run_approved:
    raise RuntimeError(
        "Kaggle train is fail-closed: user approval for the selected plan and "
        "booster count has not been recorded."
    )
if nested(CONFIG, "model.approval.status") != "approved":
    raise RuntimeError("Approval status must be 'approved' before training.")
if not nested(CONFIG, "model.approval.approved_at"):
    raise RuntimeError("Approval timestamp must be recorded before training.")

selected_plan = plans[selected_plan_name]
if nested(CONFIG, "model.approval.approved_scope") != selected_plan["approval_scope"]:
    raise RuntimeError("Recorded approval scope does not match the selected plan.")
selected_config_indices = [int(value) for value in selected_plan["config_indices"]]
n_folds = int(nested(CONFIG, "validation.n_folds"))
expected_boosters = len(selected_config_indices) * n_folds
if expected_boosters != int(selected_plan["boosters"]):
    raise ValueError("Selected plan booster count does not match config indices x folds")
if bool(nested(CONFIG, "model.variant.control_retraining")):
    raise ValueError("Frozen exp218 control retraining must remain disabled")
print(
    {
        "selected_plan": selected_plan_name,
        "config_indices": selected_config_indices,
        "folds": n_folds,
        "planned_boosters": expected_boosters,
        "control_boosters": 0,
    }
)

# %% [markdown]
# ## 3. Frozen exp218 input contracts

# %%
parent_oof_filename = (
    "exp218_gr_wavelet_rotation_confidence_features_on_exp148_predictions.csv.gz"
)
parent_schema_filename = (
    "exp218_gr_wavelet_rotation_confidence_features_on_exp148_feature_schema.csv"
)
parent_oof_path = find_artifact(
    parent_oof_filename,
    explicit=nested(CONFIG, "data.exp218_oof_predictions_local"),
    expected_sha256=nested(
        CONFIG, "frozen_parent.expected_oof_decompressed_sha256"
    ),
    decompressed=True,
)
parent_schema_path = find_artifact(
    parent_schema_filename,
    explicit=nested(CONFIG, "data.exp218_feature_schema_local"),
    expected_sha256=nested(CONFIG, "frozen_parent.expected_feature_schema_sha256"),
)
parent_manifest_path = find_artifact(
    "manifest.json",
    explicit=nested(CONFIG, "data.exp218_model_manifest_local"),
    expected_sha256=nested(CONFIG, "frozen_parent.expected_model_manifest_sha256"),
)
hidden_assignment_path = find_artifact(
    "exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv",
    explicit=nested(CONFIG, "data.hidden_like_assignment_local"),
    expected_sha256=nested(CONFIG, "validation.expected_hidden_assignment_sha256"),
)

parent_oof = pd.read_csv(
    parent_oof_path,
    dtype={"id": str, "well": str},
)
if len(parent_oof) != int(nested(CONFIG, "frozen_parent.expected_rows")):
    raise ValueError("Frozen exp218 OOF row count changed")
if parent_oof["well"].nunique() != int(nested(CONFIG, "frozen_parent.expected_wells")):
    raise ValueError("Frozen exp218 OOF well count changed")
if set(parent_oof["model"].astype(str).unique()) != {"lgb_mean"}:
    raise ValueError("Frozen exp218 OOF must contain only the lgb_mean surface")
parent_oof_rmse = rmse(parent_oof["target_tvt"], parent_oof["pred_tvt"])
if abs(
    parent_oof_rmse - float(nested(CONFIG, "frozen_parent.expected_oof_runtime_rmse"))
) > float(nested(CONFIG, "frozen_parent.oof_rmse_tolerance")):
    raise ValueError("Frozen exp218 OOF RMSE contract changed")

parent_manifest = json.loads(parent_manifest_path.read_text())
if int(parent_manifest["model_count"]) != 15:
    raise ValueError("Frozen exp218 manifest must contain 15 models")
hidden_assignment = pd.read_csv(hidden_assignment_path, dtype={"well_id": str})
print(
    json.dumps(
        {
            "parent_oof": str(parent_oof_path),
            "parent_oof_rows": len(parent_oof),
            "parent_oof_wells": parent_oof["well"].nunique(),
            "parent_oof_rmse": parent_oof_rmse,
            "parent_oof_decompressed_sha256": sha256_path(
                parent_oof_path, decompressed=True
            ),
            "parent_feature_schema": str(parent_schema_path),
            "parent_model_manifest": str(parent_manifest_path),
            "parent_model_count": parent_manifest["model_count"],
            "hidden_assignment": str(hidden_assignment_path),
        },
        indent=2,
    )
)
display(parent_oof.head())
display(hidden_assignment.head())

# %% [markdown]
# ## 4. Rebuild the exp218 380-feature surface

# %%
exp218_config_path = Path(exp218.NOTEBOOK_SOURCE_PATH).with_name("config.yaml")
if not exp218_config_path.exists():
    raise FileNotFoundError(f"Missing bootstrapped exp218 config: {exp218_config_path}")
exp218_config = yaml.safe_load(exp218_config_path.read_text())

frame, base_feature_columns, base_feature_meta = (
    exp218.load_exp072_full_replay_cache_frame(
        nested(exp218_config, "data.exp072_train_feature_cache_local"),
        max_rows=None,
    )
)
resolved_train_dir = exp218_settings.ExperimentPaths().train_data_dir
if not resolved_train_dir.exists():
    raise FileNotFoundError(
        f"Resolved competition train directory does not exist: {resolved_train_dir}"
    )
frame, anchor_meta = exp218.add_anchor_columns(frame, resolved_train_dir)

projection_config = nested(exp218_config, "model.u_projection", {})
projection, projection_groups, projection_summary = exp218.build_u_projection_features(
    frame,
    source_specs=dict(projection_config.get("sources") or {}),
    degree=int(projection_config.get("degree", 3)),
    robust_iters=int(projection_config.get("robust_iters", 3)),
    clip_sigma=float(projection_config.get("clip_sigma", 4.0)),
)
projection_columns = [column for column in projection if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    frame, projection.reset_index(drop=True), projection_columns
)

learned_source, learned_source_meta = exp218.load_learned_likelihood_ml_features(
    nested(exp218_config, "data.learned_likelihood_train_features_local"),
    schema_path=nested(
        exp218_config, "data.learned_likelihood_train_feature_schema_local"
    ),
    summary_path=nested(exp218_config, "data.learned_likelihood_train_summary_local"),
)
learned, learned_groups, learned_summary = exp218.build_learned_likelihood_features(
    learned_source,
    frame,
    nested(exp218_config, "model.learned_likelihood_features", {}),
)
learned_columns = [column for column in learned if column not in {"id", "well"}]
if not frame["id"].equals(learned["id"]) or not frame["well"].equals(learned["well"]):
    raise ValueError("exp145 learned-likelihood feature row order changed")
exp218._assign_aligned_float32_columns(
    frame, learned.reset_index(drop=True), learned_columns
)

grwr, grwr_groups, grwr_summary, grwr_meta = (
    exp218.build_gr_wavelet_rotation_confidence_features(
        frame,
        train_dir=resolved_train_dir,
        config=nested(
            exp218_config, "model.gr_wavelet_rotation_confidence_features", {}
        ),
    )
)
grwr_columns = [column for column in grwr if column not in {"id", "well"}]
if not frame["id"].equals(grwr["id"]) or not frame["well"].equals(grwr["well"]):
    raise ValueError("GRWR feature row order changed")
exp218._assign_aligned_float32_columns(
    frame, grwr.reset_index(drop=True), grwr_columns
)
del projection, learned_source, learned, grwr
gc.collect()

feature_groups = {**projection_groups, **learned_groups, **grwr_groups}
parent_variant = next(
    variant
    for variant in nested(exp218_config, "model.feature_ablation.active_variants", [])
    if variant.get("name") == "gr_wavelet_rotation_confidence_addonly"
)
feature_columns = exp218.feature_columns_for_variant(
    base_feature_columns,
    feature_groups,
    parent_variant,
)
if len(feature_columns) != int(nested(CONFIG, "frozen_parent.expected_feature_count")):
    raise ValueError(f"Expected 380 exp218 features, got {len(feature_columns)}")
if len(frame) != int(nested(CONFIG, "validation.expected_rows")):
    raise ValueError("Rebuilt exp218 surface row count changed")
if frame["well"].nunique() != int(nested(CONFIG, "validation.expected_wells")):
    raise ValueError("Rebuilt exp218 surface well count changed")
if not frame["id"].astype(str).reset_index(drop=True).equals(
    parent_oof["id"].astype(str).reset_index(drop=True)
):
    raise ValueError("Rebuilt exp218 surface ID order differs from frozen OOF")
if not frame["well"].astype(str).reset_index(drop=True).equals(
    parent_oof["well"].astype(str).reset_index(drop=True)
):
    raise ValueError("Rebuilt exp218 surface well order differs from frozen OOF")
if not np.allclose(
    frame["target"].to_numpy(np.float32),
    parent_oof["target"].to_numpy(np.float32),
    atol=1e-6,
    rtol=0.0,
):
    raise ValueError("Rebuilt exp218 target differs from frozen OOF")
if not np.allclose(
    frame["last_known_tvt"].to_numpy(np.float32),
    parent_oof["last_known_tvt"].to_numpy(np.float32),
    atol=1e-6,
    rtol=0.0,
):
    raise ValueError("Rebuilt exp218 base prediction differs from frozen OOF")
parent_lgb_mean_tvt = parent_oof["pred_tvt"].to_numpy(np.float32, copy=True)

parent_schema = pd.read_csv(parent_schema_path)
parent_schema = parent_schema[
    parent_schema["variant"].eq("gr_wavelet_rotation_confidence_addonly")
].sort_values("feature_index")
if parent_schema["feature"].tolist() != feature_columns:
    raise ValueError("Rebuilt feature schema differs from frozen exp218 schema")
del parent_oof, parent_schema
gc.collect()
feature_schema = pd.DataFrame(
    {
        "feature_index": np.arange(len(feature_columns), dtype=np.int32),
        "feature": feature_columns,
    }
)
feature_schema_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_schema.csv"
feature_schema.to_csv(feature_schema_path, index=False)
feature_content_sha = feature_content_sha256(frame, feature_columns)
print(
    json.dumps(
        {
            "rows": len(frame),
            "wells": frame["well"].nunique(),
            "features": len(feature_columns),
            "feature_schema_sha256": sha256_path(feature_schema_path),
            "feature_content_sha256": feature_content_sha,
            "base_feature_meta": base_feature_meta,
            "anchor_meta": anchor_meta,
            "learned_source_meta": learned_source_meta,
            "grwr_meta": grwr_meta,
        },
        indent=2,
        default=str,
    )
)
display(feature_schema.head(20))
display(projection_summary.head())
display(learned_summary.head())
display(grwr_summary.head())

# %% [markdown]
# ## 5. Single-parameter LightGBM contract

# %%
parent_mode = nested(
    exp218_config, "model.training.modes.gpu_repro_guard_dp_threads8"
)
parent_params = exp218.apply_mode_overrides(
    exp218.exp063_lgb_config_family(fast=False), parent_mode
)
if len(parent_params) != 3:
    raise ValueError("The exp218 LightGBM config family no longer has three configs")
expected_mode_overrides = nested(CONFIG, "model.training.expected_mode_overrides", {})
for key, expected in expected_mode_overrides.items():
    if parent_mode["common_overrides"].get(key) != expected:
        raise ValueError(f"Parent mode override changed for {key}")

selected_params: dict[int, dict[str, Any]] = {}
parameter_audit_rows: list[dict[str, Any]] = []
for config_index in selected_config_indices:
    if config_index < 0 or config_index >= len(parent_params):
        raise ValueError(f"Invalid LightGBM config index: {config_index}")
    control = dict(parent_params[config_index])
    candidate = dict(control)
    candidate["extra_trees"] = True
    changed_keys = sorted(
        key
        for key in set(control) | set(candidate)
        if control.get(key) != candidate.get(key)
    )
    if changed_keys != ["extra_trees"]:
        raise ValueError(
            f"Config {config_index} changes more than extra_trees: {changed_keys}"
        )
    if candidate["extra_trees"] is not True:
        raise ValueError("extra_trees must be exactly True")
    selected_params[config_index] = candidate
    parameter_audit_rows.append(
        {
            "config_index": config_index,
            "changed_keys": ",".join(changed_keys),
            "control_extra_trees": control.get("extra_trees"),
            "candidate_extra_trees": candidate["extra_trees"],
            "control_params_json": json.dumps(control, sort_keys=True),
            "candidate_params_json": json.dumps(candidate, sort_keys=True),
        }
    )
parameter_audit = pd.DataFrame(parameter_audit_rows)
parameter_audit_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_parameter_audit.csv"
parameter_audit.to_csv(parameter_audit_path, index=False)
display(parameter_audit)

# %% [markdown]
# ## 6. Matched control inference and extra-trees CV

# %%
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

y = frame["target"].to_numpy(np.float32)
base = frame["last_known_tvt"].to_numpy(np.float32)
target_tvt = base + y
groups = frame["well"].astype(str).to_numpy()
row_index = np.arange(len(frame), dtype=np.int64)
splits = list(GroupKFold(n_splits=n_folds).split(row_index, y, groups=groups))
fold_assignment = np.full(len(frame), -1, dtype=np.int8)
for fold, (_, valid_idx) in enumerate(splits):
    fold_assignment[valid_idx] = fold
if (fold_assignment < 0).any():
    raise ValueError("GroupKFold did not assign every row exactly once")

manifest_models = parent_manifest["models"]
parent_model_rows = {
    (int(row["model_index"]), int(row["fold"])): row for row in manifest_models
}
if len(parent_model_rows) != 15:
    raise ValueError("Frozen exp218 manifest model index/fold coverage changed")

parent_config_oof = {
    index: np.zeros(len(frame), dtype=np.float32) for index in selected_config_indices
}
extra_config_oof = {
    index: np.zeros(len(frame), dtype=np.float32) for index in selected_config_indices
}
fold_metric_rows: list[dict[str, Any]] = []
importance_rows: list[dict[str, Any]] = []
new_model_rows: list[dict[str, Any]] = []
model_dir = OUTPUT_DIR / f"{OUTPUT_PREFIX}_lgb_models"
model_dir.mkdir(parents=True, exist_ok=True)

for config_index in selected_config_indices:
    params = selected_params[config_index]
    for fold, (train_idx, valid_idx) in enumerate(splits):
        parent_row = parent_model_rows[(config_index, fold)]
        parent_model_path = parent_manifest_path.parent / str(parent_row["file"])
        if not parent_model_path.exists():
            raise FileNotFoundError(f"Missing frozen exp218 model: {parent_model_path}")
        if sha256_path(parent_model_path) != str(parent_row["sha256"]):
            raise ValueError(f"Frozen exp218 model SHA mismatch: {parent_model_path}")

        x_valid = frame.iloc[valid_idx][feature_columns].to_numpy(
            dtype=np.float32, copy=True
        )
        frozen_model = lgb.Booster(model_file=str(parent_model_path))
        frozen_pred = frozen_model.predict(
            x_valid, num_iteration=int(parent_row["best_iteration"])
        ).astype(np.float32)
        parent_config_oof[config_index][valid_idx] = frozen_pred
        del frozen_model

        train_use = train_idx
        max_train_rows = nested(CONFIG, "model.training.max_train_rows")
        if max_train_rows is not None and len(train_use) > int(max_train_rows):
            rng = np.random.default_rng(
                int(nested(CONFIG, "validation.seed")) + config_index * 100 + fold
            )
            train_use = np.sort(
                rng.choice(train_use, size=int(max_train_rows), replace=False)
            )
        x_train = frame.iloc[train_use][feature_columns].to_numpy(
            dtype=np.float32, copy=True
        )
        y_train = y[train_use]
        y_valid = y[valid_idx]
        model = LGBMRegressor(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[
                early_stopping(
                    int(nested(CONFIG, "model.training.early_stopping_rounds")),
                    verbose=False,
                ),
                log_evaluation(0),
            ],
        )
        best_iteration = int(model.best_iteration_ or params["n_estimators"])
        extra_pred = model.predict(
            x_valid, num_iteration=best_iteration
        ).astype(np.float32)
        extra_config_oof[config_index][valid_idx] = extra_pred

        model_filename = f"extra_trees_true__lgb{config_index}__fold{fold}.txt"
        model_path = model_dir / model_filename
        model.booster_.save_model(str(model_path), num_iteration=best_iteration)
        model_sha = sha256_path(model_path)
        new_model_rows.append(
            {
                "plan": selected_plan_name,
                "variant": "extra_trees_true",
                "model": f"lgb{config_index}",
                "model_index": config_index,
                "fold": fold,
                "best_iteration": best_iteration,
                "file": model_filename,
                "sha256": model_sha,
                "extra_trees": True,
            }
        )
        fold_metric_rows.append(
            {
                "model": f"lgb{config_index}",
                "fold": fold,
                "rows": len(valid_idx),
                "train_rows": len(train_use),
                "parent_rmse": rmse(
                    target_tvt[valid_idx], base[valid_idx] + frozen_pred
                ),
                "extra_trees_rmse": rmse(
                    target_tvt[valid_idx], base[valid_idx] + extra_pred
                ),
                "delta_rmse": rmse(
                    target_tvt[valid_idx], base[valid_idx] + extra_pred
                )
                - rmse(target_tvt[valid_idx], base[valid_idx] + frozen_pred),
                "best_iteration": best_iteration,
            }
        )
        for feature, importance in zip(
            feature_columns, model.feature_importances_, strict=False
        ):
            importance_rows.append(
                {
                    "model": f"lgb{config_index}",
                    "model_index": config_index,
                    "fold": fold,
                    "feature": feature,
                    "importance": float(importance),
                }
            )
        print(json.dumps(fold_metric_rows[-1], sort_keys=True), flush=True)
        del model, x_train, x_valid, y_train, y_valid, frozen_pred, extra_pred
        gc.collect()

parent_selected_residual = np.mean(
    np.vstack([parent_config_oof[index] for index in selected_config_indices]), axis=0
).astype(np.float32)
extra_selected_residual = np.mean(
    np.vstack([extra_config_oof[index] for index in selected_config_indices]), axis=0
).astype(np.float32)
parent_selected_tvt = base + parent_selected_residual
extra_selected_tvt = base + extra_selected_residual
parent_full_family_parity: dict[str, Any] | None = None
if selected_config_indices == [0, 1, 2]:
    absolute_difference = np.abs(parent_selected_tvt - parent_lgb_mean_tvt)
    parent_full_family_parity = {
        "mean_absolute_difference": float(absolute_difference.mean()),
        "max_absolute_difference": float(absolute_difference.max()),
        "reconstructed_rmse": rmse(target_tvt, parent_selected_tvt),
        "frozen_oof_rmse": rmse(target_tvt, parent_lgb_mean_tvt),
    }
    if parent_full_family_parity["max_absolute_difference"] > 2e-4:
        raise ValueError(
            "Saved exp218 boosters do not reproduce the frozen lgb_mean OOF"
        )

for fold in range(n_folds):
    mask = fold_assignment == fold
    parent_fold_rmse = rmse(target_tvt[mask], parent_selected_tvt[mask])
    extra_fold_rmse = rmse(target_tvt[mask], extra_selected_tvt[mask])
    fold_metric_rows.append(
        {
            "model": "selected_config_mean",
            "fold": fold,
            "rows": int(mask.sum()),
            "train_rows": None,
            "parent_rmse": parent_fold_rmse,
            "extra_trees_rmse": extra_fold_rmse,
            "delta_rmse": extra_fold_rmse - parent_fold_rmse,
            "best_iteration": None,
        }
    )
fold_metrics = pd.DataFrame(fold_metric_rows)

metric_rows: list[dict[str, Any]] = []
historical_rmse = nested(CONFIG, "frozen_parent.historical_pooled_rmse")
for config_index in selected_config_indices:
    parent_pred_tvt = base + parent_config_oof[config_index]
    extra_pred_tvt = base + extra_config_oof[config_index]
    parent_rmse = rmse(target_tvt, parent_pred_tvt)
    extra_rmse = rmse(target_tvt, extra_pred_tvt)
    if abs(parent_rmse - float(historical_rmse[f"lgb{config_index}"])) > 2e-5:
        raise ValueError(f"Frozen lgb{config_index} inference does not reproduce exp218 CV")
    metric_rows.append(
        {
            "model": f"lgb{config_index}",
            "config_index": config_index,
            "parent_rmse": parent_rmse,
            "extra_trees_rmse": extra_rmse,
            "delta_rmse": extra_rmse - parent_rmse,
            "parent_extra_correlation": float(
                np.corrcoef(parent_pred_tvt, extra_pred_tvt)[0, 1]
            ),
        }
    )
parent_selected_rmse = rmse(target_tvt, parent_selected_tvt)
extra_selected_rmse = rmse(target_tvt, extra_selected_tvt)
metric_rows.append(
    {
        "model": "selected_config_mean",
        "config_index": None,
        "parent_rmse": parent_selected_rmse,
        "extra_trees_rmse": extra_selected_rmse,
        "delta_rmse": extra_selected_rmse - parent_selected_rmse,
        "parent_extra_correlation": float(
            np.corrcoef(parent_selected_tvt, extra_selected_tvt)[0, 1]
        ),
    }
)
metrics = pd.DataFrame(metric_rows)
display(metrics)
display(fold_metrics)

# %% [markdown]
# ## 7. Stress, blend, and adoption readouts

# %%
assignment_by_well = hidden_assignment.set_index("well_id")
roles = frame["well"].astype(str).map(assignment_by_well.to_dict("index"))
spatial_valid = roles.map(
    lambda value: isinstance(value, dict)
    and value.get("verification_like_spatial_role") == "valid"
).to_numpy(bool)
typewell_valid = roles.map(
    lambda value: isinstance(value, dict)
    and value.get("verification_like_typewell_purged_role") == "valid"
).to_numpy(bool)
md_since = frame["md_since"].to_numpy(np.float32)
distance_bucket = pd.cut(
    md_since,
    bins=[-np.inf, 50, 100, 250, 500, 1000, np.inf],
    labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
    right=False,
).astype(str)

surface_masks: dict[str, np.ndarray] = {
    "overall": np.ones(len(frame), dtype=bool),
    "near_000_050": md_since < 50,
    "1000_plus": md_since >= 1000,
    "hidden_like_spatial": spatial_valid,
    "hidden_like_typewell_purged": typewell_valid,
}
for bucket in sorted(pd.unique(distance_bucket)):
    surface_masks[f"distance_{bucket}"] = distance_bucket == bucket

stress_rows: list[dict[str, Any]] = []
for surface, mask in surface_masks.items():
    if not mask.any():
        continue
    parent_value = rmse(target_tvt[mask], parent_selected_tvt[mask])
    extra_value = rmse(target_tvt[mask], extra_selected_tvt[mask])
    stress_rows.append(
        {
            "surface": surface,
            "rows": int(mask.sum()),
            "wells": int(frame.loc[mask, "well"].nunique()),
            "parent_selected_rmse": parent_value,
            "extra_trees_rmse": extra_value,
            "delta_rmse": extra_value - parent_value,
        }
    )
stress_metrics = pd.DataFrame(stress_rows)
bucket_metrics = stress_metrics[stress_metrics["surface"].str.startswith("distance_")].copy()
hidden_like_metrics = stress_metrics[
    stress_metrics["surface"].str.startswith("hidden_like_")
].copy()

blend_rows: list[dict[str, Any]] = []
blend_predictions: dict[str, np.ndarray] = {}
for weight in nested(CONFIG, "model.blend_readout.extra_weights", []):
    weight = float(weight)
    label = f"blend_extra_w{weight:.2f}"
    prediction = (
        (1.0 - weight) * parent_lgb_mean_tvt + weight * extra_selected_tvt
    ).astype(np.float32)
    blend_predictions[label] = prediction
    for surface, mask in surface_masks.items():
        if not mask.any():
            continue
        baseline_rmse = rmse(target_tvt[mask], parent_lgb_mean_tvt[mask])
        blend_rmse = rmse(target_tvt[mask], prediction[mask])
        blend_rows.append(
            {
                "extra_weight": weight,
                "surface": surface,
                "rows": int(mask.sum()),
                "parent_lgb_mean_rmse": baseline_rmse,
                "blend_rmse": blend_rmse,
                "delta_rmse": blend_rmse - baseline_rmse,
            }
        )
blend_readout = pd.DataFrame(blend_rows)

by_well_base = pd.DataFrame(
    {
        "well": frame["well"].astype(str),
        "target_tvt": target_tvt,
        "parent_selected_sqerr": np.square(target_tvt - parent_selected_tvt),
        "extra_selected_sqerr": np.square(target_tvt - extra_selected_tvt),
        "parent_lgb_mean_sqerr": np.square(target_tvt - parent_lgb_mean_tvt),
    }
)
by_well = (
    by_well_base.groupby("well", as_index=False)
    .agg(
        rows=("target_tvt", "size"),
        parent_selected_mse=("parent_selected_sqerr", "mean"),
        extra_trees_mse=("extra_selected_sqerr", "mean"),
        parent_lgb_mean_mse=("parent_lgb_mean_sqerr", "mean"),
    )
)
by_well["parent_selected_rmse"] = np.sqrt(by_well.pop("parent_selected_mse"))
by_well["extra_trees_rmse"] = np.sqrt(by_well.pop("extra_trees_mse"))
by_well["parent_lgb_mean_rmse"] = np.sqrt(by_well.pop("parent_lgb_mean_mse"))
by_well["delta_rmse"] = (
    by_well["extra_trees_rmse"] - by_well["parent_selected_rmse"]
)
by_well = by_well.sort_values("delta_rmse", ascending=False).reset_index(drop=True)

stress_lookup = stress_metrics.set_index("surface")
selected_fold_metrics = fold_metrics[fold_metrics["model"].eq("selected_config_mean")]
guard = {
    "primary_overall_improvement": bool(
        stress_lookup.at["overall", "delta_rmse"] < 0.0
    ),
    "rows_1000_plus_non_worse": bool(
        stress_lookup.at["1000_plus", "delta_rmse"] <= 0.0
    ),
    "hidden_like_spatial_non_worse": bool(
        stress_lookup.at["hidden_like_spatial", "delta_rmse"] <= 0.0
    ),
    "hidden_like_typewell_purged_non_worse": bool(
        stress_lookup.at["hidden_like_typewell_purged", "delta_rmse"] <= 0.0
    ),
    "worst_well_regression_within_limit": bool(
        by_well["delta_rmse"].max()
        <= float(nested(CONFIG, "model.adoption_guards.max_worst_well_regression"))
    ),
    "improved_folds_at_least_minimum": bool(
        int((selected_fold_metrics["delta_rmse"] < 0.0).sum())
        >= int(nested(CONFIG, "model.adoption_guards.min_improved_folds"))
    ),
    "worst_well_regression": float(by_well["delta_rmse"].max()),
    "improved_folds": int((selected_fold_metrics["delta_rmse"] < 0.0).sum()),
}
guard["adoption_supported"] = bool(
    all(value for key, value in guard.items() if isinstance(value, bool))
)
display(stress_metrics)
display(blend_readout)
display(by_well.head(30))
print(json.dumps(guard, indent=2))

# %% [markdown]
# ## 8. Metrics and generated artifacts

# %%
importance = pd.DataFrame(importance_rows)
importance_mean = (
    importance.groupby(["model", "model_index", "feature"], as_index=False)
    .agg(
        mean_importance=("importance", "mean"),
        std_importance=("importance", "std"),
        fold_records=("importance", "size"),
    )
    .sort_values(["model", "mean_importance"], ascending=[True, False])
)

import matplotlib.pyplot as plt

plot_frame = importance_mean.groupby("feature", as_index=False)["mean_importance"].mean()
plot_frame = plot_frame.nlargest(
    int(nested(CONFIG, "model.training.top_n_importance")), "mean_importance"
).sort_values("mean_importance")
fig, ax = plt.subplots(figsize=(12, max(6, 0.24 * len(plot_frame))))
ax.barh(plot_frame["feature"], plot_frame["mean_importance"], color="#2f6f8f")
ax.set_title(f"{OUTPUT_PREFIX}: mean feature importance")
ax.set_xlabel("mean feature_importances_")
fig.tight_layout()
importance_plot_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_importance_mean_top.png"
fig.savefig(importance_plot_path, dpi=160)
plt.close(fig)

oof_output = pd.DataFrame(
    {
        "id": frame["id"].astype(str),
        "well": frame["well"].astype(str),
        "fold": fold_assignment,
        "last_known_tvt": base,
        "target_tvt": target_tvt,
        "parent_lgb_mean_tvt": parent_lgb_mean_tvt,
        "parent_selected_mean_tvt": parent_selected_tvt,
        "extra_trees_selected_mean_tvt": extra_selected_tvt,
    }
)
for config_index in selected_config_indices:
    oof_output[f"parent_lgb{config_index}_tvt"] = (
        base + parent_config_oof[config_index]
    )
    oof_output[f"extra_trees_lgb{config_index}_tvt"] = (
        base + extra_config_oof[config_index]
    )
for label, prediction in blend_predictions.items():
    oof_output[label] = prediction

metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_metrics.csv"
fold_metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_fold_metrics.csv"
bucket_metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
hidden_metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_hidden_like_metrics.csv"
by_well_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_by_well.csv"
blend_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_blend_readout.csv"
importance_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_importance_mean.csv"
oof_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
guard_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_guard.json"
manifest_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_model_manifest.json"
summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.json"

metrics.to_csv(metrics_path, index=False)
fold_metrics.to_csv(fold_metrics_path, index=False)
bucket_metrics.to_csv(bucket_metrics_path, index=False)
hidden_like_metrics.to_csv(hidden_metrics_path, index=False)
by_well.to_csv(by_well_path, index=False)
blend_readout.to_csv(blend_path, index=False)
importance_mean.to_csv(importance_path, index=False)
oof_output.to_csv(oof_path, index=False, compression="gzip")
guard_path.write_text(json.dumps(guard, indent=2))

model_manifest = {
    "experiment": nested(CONFIG, "experiment.name"),
    "parent": nested(CONFIG, "lineage.parent"),
    "plan": selected_plan_name,
    "variant": "extra_trees_true",
    "config_indices": selected_config_indices,
    "folds": n_folds,
    "model_count": len(new_model_rows),
    "control_retraining": False,
    "changed_parameter": {"extra_trees": True},
    "feature_count": len(feature_columns),
    "feature_content_sha256": feature_content_sha,
    "parent_full_family_parity": parent_full_family_parity,
    "models": new_model_rows,
}
manifest_path.write_text(json.dumps(model_manifest, indent=2))

artifact_sha = {
    "parameter_audit": sha256_path(parameter_audit_path),
    "feature_schema": sha256_path(feature_schema_path),
    "metrics": sha256_path(metrics_path),
    "fold_metrics": sha256_path(fold_metrics_path),
    "bucket_metrics": sha256_path(bucket_metrics_path),
    "hidden_like_metrics": sha256_path(hidden_metrics_path),
    "by_well": sha256_path(by_well_path),
    "blend_readout": sha256_path(blend_path),
    "feature_importance_mean": sha256_path(importance_path),
    "oof_decompressed": sha256_path(oof_path, decompressed=True),
    "guard": sha256_path(guard_path),
    "model_manifest": sha256_path(manifest_path),
    "frozen_parent_oof_decompressed": sha256_path(
        parent_oof_path, decompressed=True
    ),
    "frozen_parent_model_manifest": sha256_path(parent_manifest_path),
}
summary = {
    "experiment": nested(CONFIG, "experiment.name"),
    "status": "train_completed_guard_pass" if guard["adoption_supported"] else "train_completed_guard_failed",
    "route": nested(CONFIG, "experiment.route"),
    "parent": nested(CONFIG, "lineage.parent"),
    "selected_plan": selected_plan_name,
    "selected_config_indices": selected_config_indices,
    "active_variants": 1,
    "lightgbm_configs": len(selected_config_indices),
    "folds": n_folds,
    "boosters": len(new_model_rows),
    "control_retraining": False,
    "changed_parameter": {"extra_trees": True},
    "feature_count": len(feature_columns),
    "feature_content_sha256": feature_content_sha,
    "parent_full_family_parity": parent_full_family_parity,
    "primary_parent_rmse": parent_selected_rmse,
    "primary_extra_trees_rmse": extra_selected_rmse,
    "primary_delta_rmse": extra_selected_rmse - parent_selected_rmse,
    "metrics": metrics.to_dict("records"),
    "guard": guard,
    "input_sha256": {
        "frozen_parent_oof_decompressed": sha256_path(
            parent_oof_path, decompressed=True
        ),
        "frozen_parent_feature_schema": sha256_path(parent_schema_path),
        "frozen_parent_model_manifest": sha256_path(parent_manifest_path),
        "hidden_like_assignment": sha256_path(hidden_assignment_path),
    },
    "artifact_sha256": artifact_sha,
    "artifacts": {
        "metrics": metrics_path.name,
        "fold_metrics": fold_metrics_path.name,
        "bucket_metrics": bucket_metrics_path.name,
        "hidden_like_metrics": hidden_metrics_path.name,
        "by_well": by_well_path.name,
        "blend_readout": blend_path.name,
        "feature_importance_mean": importance_path.name,
        "feature_importance_plot": importance_plot_path.name,
        "oof_predictions": oof_path.name,
        "feature_schema": feature_schema_path.name,
        "parameter_audit": parameter_audit_path.name,
        "model_manifest": manifest_path.name,
        "guard": guard_path.name,
    },
    "elapsed_seconds": round(time.time() - STARTED, 3),
}
summary_path.write_text(json.dumps(summary, indent=2))
summary_for_display = {
    **summary,
    "summary_sha256": sha256_path(summary_path),
}
print(json.dumps(summary_for_display, indent=2), flush=True)
display(importance_mean.groupby("model").head(30))

# %% [markdown]
# # exp274 CatBoost final-regressor swap on exp238 train
#
# The exp238 row contract, outer folds, residual target, 380 base features,
# and nested selector rank-slot features stay fixed. Only the final estimator
# changes from the saved LightGBM family to public-notebook CatBoost `cb0`.

# %% [markdown]
# ## Contents
# 1. Imports, paths, and source resolution
# 2. Configuration, public parameter, and GPU-cost contracts
# 3. Frozen selector and parent OOF contracts
# 4. Candidate, fold, and exp218 feature reconstruction
# 5. Fold-specific nested rank-slot feature assembly
# 6. Public CatBoost `cb0` training
# 7. CV, stress-surface, blend, and adoption readouts
# 8. Feature importance, manifests, SHA, and generated artifacts

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor, Pool
from IPython.display import display

STARTED = time.time()
PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path("experiments/exp274_catboost_final_regressor_swap_on_exp238")
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_DIR = (
    Path("/kaggle/working/artifacts")
    if Path("/kaggle/working").exists()
    else PACKAGE_DIR / "artifacts"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = str(CONFIG["audit"]["output_prefix"])


def nested(mapping: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def import_file(
    name: str,
    candidates: list[Path],
    *,
    reset_settings: bool = False,
):
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Cannot resolve {name}: {candidates}")
    if reset_settings:
        sys.modules.pop("settings", None)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else open
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_sha256(values: np.ndarray) -> str:
    if not values.flags.c_contiguous:
        raise ValueError("matrix SHA requires a C-contiguous array")
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(target64 - prediction64))))


def mae(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(target, dtype=np.float64)
                - np.asarray(prediction, dtype=np.float64)
            )
        )
    )


def within10(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(target, dtype=np.float64)
                - np.asarray(prediction, dtype=np.float64)
            )
            <= 10.0
        )
    )


def resolve_file(name: str, candidates: list[Path], recursive_name: str | None = None) -> Path:
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None and recursive_name and Path("/kaggle/input").exists():
        matches = sorted(Path("/kaggle/input").rglob(recursive_name))
        path = matches[0] if matches else None
    if path is None:
        raise FileNotFoundError(f"Cannot resolve {name}: {candidates}")
    return path


engine = import_file(
    "exp238_engine",
    [
        PACKAGE_DIR
        / "exp238_source/nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py",
        Path(
            "experiments/exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218/"
            "nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py"
        ),
    ],
)
exp237 = import_file(
    "exp237_source",
    [
        PACKAGE_DIR / "exp237_source/hmm_exp226_candidate_selector_on_exp183.py",
        Path(
            "experiments/exp237_hmm_exp226_candidate_selector_on_exp183/"
            "hmm_exp226_candidate_selector_on_exp183.py"
        ),
        Path(
            "/kaggle/input/exp237-hmm-exp226-candidate-selector-exp183-train/"
            "hmm_exp226_candidate_selector_on_exp183.py"
        ),
    ],
    reset_settings=True,
)
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
    reset_settings=True,
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

# %% [markdown]
# ## 2. Configuration, public parameter, and GPU-cost contracts

# %%
PUBLIC_CB0 = {
    "iterations": 8000,
    "depth": 7,
    "l2_leaf_reg": 2.0,
    "min_data_in_leaf": 15,
    "border_count": 254,
    "loss_function": "RMSE",
    "task_type": "GPU",
    "od_type": "Iter",
    "od_wait": 300,
    "verbose": 0,
    "learning_rate": 0.02,
    "random_seed": 7,
}
PUBLIC_FIT = {"early_stopping_rounds": 250, "use_best_model": True}
configured_cb0 = dict(nested(CONFIG, "model.catboost.source_exact_params", {}))
configured_fit = dict(nested(CONFIG, "model.catboost.source_exact_fit_params", {}))
if configured_cb0 != PUBLIC_CB0:
    raise ValueError(f"Public CatBoost cb0 drifted: {configured_cb0}")
if configured_fit != PUBLIC_FIT:
    raise ValueError(f"Public CatBoost fit contract drifted: {configured_fit}")

cost_contract = {
    "active_variants": int(nested(CONFIG, "model.active_variant_count")),
    "catboost_configs": int(nested(CONFIG, "model.catboost_config_count")),
    "folds": int(nested(CONFIG, "model.folds")),
    "total_new_models": int(nested(CONFIG, "model.total_new_models")),
    "max_iterations_per_model": int(
        nested(CONFIG, "model.max_iterations_per_model")
    ),
    "maximum_total_iterations": int(
        nested(CONFIG, "model.maximum_total_iterations")
    ),
    "parent_control_retraining": bool(nested(CONFIG, "model.parent_control_retraining")),
    "selector_retraining": bool(nested(CONFIG, "model.selector_retraining")),
}
expected_cost = {
    "active_variants": 1,
    "catboost_configs": 1,
    "folds": 5,
    "total_new_models": 5,
    "max_iterations_per_model": 8000,
    "maximum_total_iterations": 40000,
    "parent_control_retraining": False,
    "selector_retraining": False,
}
if cost_contract != expected_cost:
    raise ValueError(f"GPU cost contract changed: {cost_contract}")

parameter_audit = {
    "public_source": nested(CONFIG, "model.public_source"),
    "source_exact_params": configured_cb0,
    "source_exact_fit_params": configured_fit,
    "runtime_only_params": nested(CONFIG, "model.catboost.runtime_only_params", {}),
    "public_cb0_exact_match": True,
    "public_cb1_trained": False,
    "cost_contract": cost_contract,
}
parameter_audit_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_parameter_audit.json"
parameter_audit_path.write_text(json.dumps(parameter_audit, indent=2))
print(json.dumps(parameter_audit, indent=2), flush=True)

# %% [markdown]
# ## 3. Frozen selector and parent OOF contracts

# %%
selector_summary_name = f"{engine.OUTPUT_PREFIX}_selector_summary.json"
selector_candidates = [
    Path(nested(CONFIG, "data.selector_artifact_dir_local")),
    Path("/kaggle/input/exp238-nested-selector-train/artifacts"),
    Path("/kaggle/input/exp238-nested-selector-train"),
    Path("/kaggle/input/notebooks/kentookumura/exp238-nested-selector-train/artifacts"),
]
SELECTOR_DIR = next(
    (
        candidate
        for candidate in selector_candidates
        if (candidate / selector_summary_name).exists()
    ),
    None,
)
if SELECTOR_DIR is None and Path("/kaggle/input").exists():
    matches = sorted(Path("/kaggle/input").rglob(selector_summary_name))
    SELECTOR_DIR = matches[0].parent if matches else None
if SELECTOR_DIR is None:
    raise FileNotFoundError(f"Selector artifact directory not found: {selector_candidates}")

selector_summary_path = SELECTOR_DIR / selector_summary_name
selector_summary = json.loads(selector_summary_path.read_text())
parent_oof_name = f"{engine.OUTPUT_PREFIX}_final_oof_predictions.csv.gz"
parent_oof_path = resolve_file(
    "exp238 final OOF",
    [
        Path(nested(CONFIG, "data.parent_final_oof_local")),
        Path("/kaggle/input/exp238-nested-rank-slot-exp218-train/artifacts")
        / parent_oof_name,
        Path("/kaggle/input/exp238-nested-rank-slot-exp218-train")
        / parent_oof_name,
        Path(
            "/kaggle/input/notebooks/kentookumura/"
            "exp238-nested-rank-slot-exp218-train/artifacts"
        )
        / parent_oof_name,
    ],
    parent_oof_name,
)
hidden_assignment_path = resolve_file(
    "hidden-like assignment",
    [
        Path(nested(CONFIG, "data.hidden_like_assignment_local")),
        PACKAGE_DIR
        / "exp237_source/inputs/exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv",
    ],
    "exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv",
)

score_input_sha: dict[str, str] = {}
for outer_fold in range(int(nested(CONFIG, "validation.outer_folds"))):
    score_path = (
        SELECTOR_DIR
        / f"{engine.OUTPUT_PREFIX}_nested_scores_outer{outer_fold}.csv.gz"
    )
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    score_input_sha[str(outer_fold)] = sha256_path(score_path, decompressed=True)

input_contract = {
    "selector_summary": str(selector_summary_path),
    "selector_status": selector_summary.get("status"),
    "selector_guard_pass": nested(selector_summary, "decision.guard_pass"),
    "parent_oof": str(parent_oof_path),
    "parent_oof_sha256_decompressed": sha256_path(parent_oof_path, decompressed=True),
    "hidden_assignment": str(hidden_assignment_path),
    "hidden_assignment_sha256": sha256_path(hidden_assignment_path),
    "selector_score_sha256_decompressed": score_input_sha,
    "parent_control_retraining": False,
}
print(json.dumps(input_contract, indent=2), flush=True)

# %% [markdown]
# ## 4. Candidate, fold, and exp218 feature reconstruction

# %%
parent_selector_config = exp237.load_config()
parent_selector_config.setdefault("inference", {})[
    "use_test_base_as_dense_auxiliary"
] = False
candidates = exp237.candidate_specs_from_config(parent_selector_config)
required = exp237.build_required_columns(parent_selector_config, candidates)
selector_frame, _ = exp237.load_train_feature_cache(
    cache_path=exp237.get_nested(
        parent_selector_config, "data.exp099_train_feature_cache_local"
    ),
    schema_path=exp237.get_nested(
        parent_selector_config, "data.exp099_train_feature_schema_local"
    ),
    required_columns=required,
    max_rows=None,
)
selector_frame, _, _ = exp237.add_feature_enrichment(
    selector_frame, parent_selector_config, max_rows=None
)
selector_frame, _, _ = exp237.add_cluster_prior_confidence_features(
    selector_frame, parent_selector_config, max_rows=None
)
selector_frame, _, _ = exp237.add_hmm_exp226_candidate_sources(
    selector_frame, parent_selector_config
)
candidate_columns = [item.column for item in candidates]
runtime_outer, runtime_inner = engine.deterministic_outer_inner_splits(
    selector_frame,
    int(nested(CONFIG, "validation.outer_folds")),
    4,
)
outer = engine.load_nested_fold_contracts(
    SELECTOR_DIR,
    len(selector_frame),
    int(nested(CONFIG, "validation.outer_folds")),
)
fold_contract_readout = []
for outer_fold, ((train_rows, valid_rows), (runtime_train, runtime_valid)) in enumerate(
    zip(outer, runtime_outer, strict=True)
):
    fold_contract_readout.append(
        {
            "outer_fold": outer_fold,
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "train_wells": int(selector_frame.iloc[train_rows].well.nunique()),
            "valid_wells": int(selector_frame.iloc[valid_rows].well.nunique()),
            "runtime_reconstructed_fold_match": bool(
                np.array_equal(train_rows, np.sort(runtime_train))
                and np.array_equal(valid_rows, np.sort(runtime_valid))
            ),
        }
    )
display(pd.DataFrame(fold_contract_readout))
del runtime_outer, runtime_inner
gc.collect()

exp218_source_dir = Path(exp218.__spec__.origin).parent
exp218_config = yaml.safe_load((exp218_source_dir / "config.yaml").read_text())
base_frame, base_feature_columns, _ = exp218.load_exp072_full_replay_cache_frame(
    nested(exp218_config, "data.exp072_train_feature_cache_local"), max_rows=None
)
resolved_train_dir = exp218_settings.ExperimentPaths().train_data_dir
if not resolved_train_dir.exists():
    raise FileNotFoundError(
        f"Resolved competition train directory does not exist: {resolved_train_dir}"
    )
base_frame, _ = exp218.add_anchor_columns(base_frame, resolved_train_dir)

projection_cfg = nested(exp218_config, "model.u_projection", {})
projection, projection_groups, _ = exp218.build_u_projection_features(
    base_frame,
    source_specs=dict(projection_cfg.get("sources") or {}),
    degree=int(projection_cfg.get("degree", 3)),
    robust_iters=int(projection_cfg.get("robust_iters", 3)),
    clip_sigma=float(projection_cfg.get("clip_sigma", 4.0)),
)
projection_columns = [column for column in projection if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    base_frame, projection.reset_index(drop=True), projection_columns
)

learned_source, _ = exp218.load_learned_likelihood_ml_features(
    nested(exp218_config, "data.learned_likelihood_train_features_local"),
    schema_path=nested(
        exp218_config, "data.learned_likelihood_train_feature_schema_local"
    ),
    summary_path=nested(exp218_config, "data.learned_likelihood_train_summary_local"),
)
learned, learned_groups, _ = exp218.build_learned_likelihood_features(
    learned_source,
    base_frame,
    nested(exp218_config, "model.learned_likelihood_features", {}),
)
learned_columns = [column for column in learned if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    base_frame, learned.reset_index(drop=True), learned_columns
)

grwr, grwr_groups, _, _ = exp218.build_gr_wavelet_rotation_confidence_features(
    base_frame,
    train_dir=resolved_train_dir,
    config=nested(exp218_config, "model.gr_wavelet_rotation_confidence_features", {}),
)
grwr_columns = [column for column in grwr if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    base_frame, grwr.reset_index(drop=True), grwr_columns
)
del projection, learned_source, learned, grwr
gc.collect()

feature_groups = {**projection_groups, **learned_groups, **grwr_groups}
parent_variant = next(
    variant
    for variant in nested(exp218_config, "model.feature_ablation.active_variants", [])
    if variant.get("name") == "gr_wavelet_rotation_confidence_addonly"
)
exp218_features = exp218.feature_columns_for_variant(
    base_feature_columns, feature_groups, parent_variant
)
if len(exp218_features) != int(
    nested(CONFIG, "model.expected_base_feature_count")
):
    raise ValueError(f"Unexpected exp218 feature count: {len(exp218_features)}")
if not base_frame[engine.KEYS].reset_index(drop=True).equals(
    selector_frame[engine.KEYS].reset_index(drop=True)
):
    raise ValueError("exp218 and selector frames are not id/well row aligned")

selector_min = selector_frame[
    [*engine.KEYS, "last_known_tvt", *candidate_columns]
].copy()
del selector_frame
selector_frame = selector_min
gc.collect()
print(
    {
        "rows": len(base_frame),
        "wells": int(base_frame.well.nunique()),
        "base_features": len(exp218_features),
        "candidate_columns": len(candidate_columns),
    },
    flush=True,
)

# %% [markdown]
# ## 5. Fold-specific nested rank-slot feature assembly
#
# The large base matrix is materialized one outer fold at a time. The estimate
# below covers the raw float32 matrices only; CatBoost Pool quantization and GPU
# working memory are additional runtime costs and are reported conservatively.

# %%
y = base_frame["target"].to_numpy(np.float32)
anchor = base_frame["last_known_tvt"].to_numpy(np.float32)
truth = anchor + y
catboost_oof_residual = np.full(len(base_frame), np.nan, dtype=np.float32)
fold_assignment = np.full(len(base_frame), -1, dtype=np.int8)
importance_rows: list[dict[str, Any]] = []
model_rows: list[dict[str, Any]] = []
feature_matrix_sha: dict[str, dict[str, str]] = {}
all_feature_columns: list[str] | None = None

runtime_params = dict(nested(CONFIG, "model.catboost.runtime_only_params", {}))
catboost_params = {**configured_cb0, **runtime_params}

# %% [markdown]
# ## 6. Public CatBoost `cb0` training

# %%
model_dir = OUTPUT_DIR / f"{OUTPUT_PREFIX}_catboost_models"
model_dir.mkdir(parents=True, exist_ok=True)
for outer_fold, (train_rows, valid_rows) in enumerate(outer):
    fold_assignment[valid_rows] = outer_fold
    score_item = engine.load_nested_score_artifact(
        SELECTOR_DIR,
        selector_frame,
        outer_fold,
        (train_rows, valid_rows),
        candidate_columns,
    )
    train_extra = engine.rank_slot_features(
        selector_frame,
        train_rows,
        score_item["train_scores"],
        candidate_columns,
    ).reset_index(drop=True)
    valid_extra = engine.rank_slot_features(
        selector_frame,
        valid_rows,
        score_item["valid_scores"],
        candidate_columns,
    ).reset_index(drop=True)
    extra_columns = list(train_extra.columns)
    if len(extra_columns) != int(
        nested(CONFIG, "model.expected_selector_feature_count")
    ):
        raise ValueError(f"Unexpected selector feature count: {len(extra_columns)}")
    all_feature_columns = [*exp218_features, *extra_columns]
    if len(all_feature_columns) != int(
        nested(CONFIG, "model.expected_final_feature_count")
    ):
        raise ValueError(f"Unexpected final feature count: {len(all_feature_columns)}")

    x_train_values = np.empty(
        (len(train_rows), len(all_feature_columns)), dtype=np.float32
    )
    x_valid_values = np.empty(
        (len(valid_rows), len(all_feature_columns)), dtype=np.float32
    )
    for start in range(0, len(exp218_features), 32):
        stop = min(start + 32, len(exp218_features))
        columns = exp218_features[start:stop]
        x_train_values[:, start:stop] = base_frame.iloc[train_rows][
            columns
        ].to_numpy(dtype=np.float32, copy=True)
        x_valid_values[:, start:stop] = base_frame.iloc[valid_rows][
            columns
        ].to_numpy(dtype=np.float32, copy=True)
    x_train_values[:, len(exp218_features) :] = train_extra.to_numpy(
        dtype=np.float32, copy=False
    )
    x_valid_values[:, len(exp218_features) :] = valid_extra.to_numpy(
        dtype=np.float32, copy=False
    )
    matrix_estimate = {
        "outer_fold": outer_fold,
        "train_float32_gib": round(x_train_values.nbytes / (1024**3), 3),
        "valid_float32_gib": round(x_valid_values.nbytes / (1024**3), 3),
        "catboost_pool_and_gpu_working_memory_additional": True,
    }
    print(json.dumps(matrix_estimate, indent=2), flush=True)
    feature_matrix_sha[str(outer_fold)] = {
        "train_float32_content": matrix_sha256(x_train_values),
        "valid_float32_content": matrix_sha256(x_valid_values),
    }

    train_pool = Pool(
        x_train_values,
        label=y[train_rows],
        feature_names=all_feature_columns,
    )
    valid_pool = Pool(
        x_valid_values,
        label=y[valid_rows],
        feature_names=all_feature_columns,
    )
    del x_train_values, x_valid_values, train_extra, valid_extra, score_item
    gc.collect()

    model = CatBoostRegressor(**catboost_params)
    fold_started = time.time()
    model.fit(train_pool, eval_set=valid_pool, **configured_fit)
    prediction = np.asarray(model.predict(valid_pool), dtype=np.float32)
    catboost_oof_residual[valid_rows] = prediction
    model_path = model_dir / f"catboost_cb0__outer{outer_fold}.cbm"
    model.save_model(str(model_path), format="cbm")
    best_iteration = int(model.get_best_iteration())
    feature_importance = np.asarray(
        model.get_feature_importance(type="FeatureImportance"), dtype=np.float64
    )
    importance_rows.extend(
        {
            "model": "catboost_cb0",
            "outer_fold": outer_fold,
            "feature": feature,
            "importance": float(value),
        }
        for feature, value in zip(
            all_feature_columns, feature_importance, strict=True
        )
    )
    model_rows.append(
        {
            "model": "catboost_cb0",
            "outer_fold": outer_fold,
            "file": str(model_path),
            "sha256": sha256_path(model_path),
            "best_iteration": best_iteration,
            "tree_count": int(model.tree_count_),
            "base_features": len(exp218_features),
            "selector_features": len(extra_columns),
            "feature_count": len(all_feature_columns),
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "valid_residual_rmse": rmse(y[valid_rows], prediction),
            "elapsed_seconds": round(time.time() - fold_started, 3),
        }
    )
    print(json.dumps(model_rows[-1], indent=2), flush=True)
    del model, train_pool, valid_pool, prediction, feature_importance
    gc.collect()

if not np.isfinite(catboost_oof_residual).all() or (fold_assignment < 0).any():
    raise AssertionError("CatBoost OOF coverage is incomplete")
if len(model_rows) != int(nested(CONFIG, "model.total_new_models")):
    raise AssertionError(f"Unexpected CatBoost model count: {len(model_rows)}")
assert all_feature_columns is not None

# %% [markdown]
# ## 7. CV, stress-surface, blend, and adoption readouts

# %%
parent_oof = pd.read_csv(
    parent_oof_path,
    usecols=["id", "well", "lgb_mean_pred_tvt"],
    dtype={"id": str, "well": str, "lgb_mean_pred_tvt": np.float32},
)
if not parent_oof[["id", "well"]].reset_index(drop=True).equals(
    base_frame[["id", "well"]].astype(str).reset_index(drop=True)
):
    raise ValueError("Saved exp238 parent OOF row order does not match exp274 frame")
parent_tvt = parent_oof["lgb_mean_pred_tvt"].to_numpy(np.float32)
catboost_tvt = anchor + catboost_oof_residual
blend_weight = float(nested(CONFIG, "model.blend_readout.catboost_weight"))
blend_tvt = ((1.0 - blend_weight) * parent_tvt + blend_weight * catboost_tvt).astype(
    np.float32
)


def metric_record(model_name: str, prediction: np.ndarray) -> dict[str, Any]:
    return {
        "model": model_name,
        "rows": len(prediction),
        "rmse_tvt": rmse(truth, prediction),
        "mae_tvt": mae(truth, prediction),
        "within10": within10(truth, prediction),
    }


metrics = pd.DataFrame(
    [
        metric_record("exp238_lgb_mean_parent", parent_tvt),
        metric_record("catboost_public_cb0", catboost_tvt),
        metric_record(f"fixed_blend_catboost_w{blend_weight:.2f}", blend_tvt),
    ]
)
parent_overall_rmse = float(
    metrics.loc[metrics.model.eq("exp238_lgb_mean_parent"), "rmse_tvt"].iloc[0]
)
metrics["delta_rmse_vs_parent"] = metrics["rmse_tvt"] - parent_overall_rmse

fold_rows: list[dict[str, Any]] = []
for outer_fold in range(int(nested(CONFIG, "validation.outer_folds"))):
    mask = fold_assignment == outer_fold
    parent_value = rmse(truth[mask], parent_tvt[mask])
    catboost_value = rmse(truth[mask], catboost_tvt[mask])
    blend_value = rmse(truth[mask], blend_tvt[mask])
    fold_rows.append(
        {
            "outer_fold": outer_fold,
            "rows": int(mask.sum()),
            "wells": int(base_frame.loc[mask, "well"].nunique()),
            "parent_rmse": parent_value,
            "catboost_rmse": catboost_value,
            "catboost_delta_rmse": catboost_value - parent_value,
            "blend_rmse": blend_value,
            "blend_delta_rmse": blend_value - parent_value,
        }
    )
fold_metrics = pd.DataFrame(fold_rows)

hidden_assignment = pd.read_csv(hidden_assignment_path, dtype={"well_id": str})
assignment_by_well = hidden_assignment.set_index("well_id")
roles = base_frame["well"].astype(str).map(assignment_by_well.to_dict("index"))
spatial_valid = roles.map(
    lambda value: isinstance(value, dict)
    and value.get("verification_like_spatial_role") == "valid"
).to_numpy(bool)
typewell_valid = roles.map(
    lambda value: isinstance(value, dict)
    and value.get("verification_like_typewell_purged_role") == "valid"
).to_numpy(bool)
md_since = base_frame["md_since"].to_numpy(np.float32)
distance_bins = list(nested(CONFIG, "validation.surfaces.distance_bins"))
distance_labels = [
    str(value) for value in nested(CONFIG, "validation.surfaces.distance_labels")
]
distance_bucket = pd.cut(
    md_since,
    bins=distance_bins,
    labels=distance_labels,
    right=False,
).astype(str)
surface_masks: dict[str, np.ndarray] = {
    "overall": np.ones(len(base_frame), dtype=bool),
    "near_000_050": md_since < 50,
    "1000_plus": md_since >= 1000,
    "hidden_like_spatial": spatial_valid,
    "hidden_like_typewell_purged": typewell_valid,
}
for label in distance_labels:
    surface_masks[f"distance_{label}"] = distance_bucket == label

stress_rows: list[dict[str, Any]] = []
blend_rows: list[dict[str, Any]] = []
for surface, mask in surface_masks.items():
    if not mask.any():
        continue
    parent_value = rmse(truth[mask], parent_tvt[mask])
    catboost_value = rmse(truth[mask], catboost_tvt[mask])
    blend_value = rmse(truth[mask], blend_tvt[mask])
    stress_rows.append(
        {
            "surface": surface,
            "rows": int(mask.sum()),
            "wells": int(base_frame.loc[mask, "well"].nunique()),
            "parent_rmse": parent_value,
            "catboost_rmse": catboost_value,
            "delta_rmse": catboost_value - parent_value,
        }
    )
    blend_rows.append(
        {
            "catboost_weight": blend_weight,
            "surface": surface,
            "rows": int(mask.sum()),
            "parent_rmse": parent_value,
            "blend_rmse": blend_value,
            "delta_rmse": blend_value - parent_value,
        }
    )
stress_metrics = pd.DataFrame(stress_rows)
blend_readout = pd.DataFrame(blend_rows)
bucket_metrics = stress_metrics[
    stress_metrics["surface"].str.startswith("distance_")
].copy()
hidden_like_metrics = stress_metrics[
    stress_metrics["surface"].str.startswith("hidden_like_")
].copy()

by_well_source = pd.DataFrame(
    {
        "well": base_frame["well"].astype(str),
        "target_tvt": truth,
        "parent_sqerr": np.square(truth - parent_tvt),
        "catboost_sqerr": np.square(truth - catboost_tvt),
        "blend_sqerr": np.square(truth - blend_tvt),
    }
)
by_well = (
    by_well_source.groupby("well", as_index=False)
    .agg(
        rows=("target_tvt", "size"),
        parent_mse=("parent_sqerr", "mean"),
        catboost_mse=("catboost_sqerr", "mean"),
        blend_mse=("blend_sqerr", "mean"),
    )
)
by_well["parent_rmse"] = np.sqrt(by_well.pop("parent_mse"))
by_well["catboost_rmse"] = np.sqrt(by_well.pop("catboost_mse"))
by_well["blend_rmse"] = np.sqrt(by_well.pop("blend_mse"))
by_well["catboost_delta_rmse"] = by_well["catboost_rmse"] - by_well["parent_rmse"]
by_well["blend_delta_rmse"] = by_well["blend_rmse"] - by_well["parent_rmse"]
by_well = by_well.sort_values("catboost_delta_rmse", ascending=False).reset_index(
    drop=True
)

stress_lookup = stress_metrics.set_index("surface")
guards_cfg = nested(CONFIG, "model.adoption_guards", {})
guard = {
    "raw_overall_improvement": bool(
        stress_lookup.at["overall", "delta_rmse"] < 0.0
    ),
    "raw_1000_plus_non_worse": bool(
        stress_lookup.at["1000_plus", "delta_rmse"] <= 0.0
    ),
    "raw_hidden_like_spatial_non_worse": bool(
        stress_lookup.at["hidden_like_spatial", "delta_rmse"] <= 0.0
    ),
    "raw_hidden_like_typewell_purged_non_worse": bool(
        stress_lookup.at["hidden_like_typewell_purged", "delta_rmse"] <= 0.0
    ),
    "raw_worst_well_regression_within_limit": bool(
        by_well["catboost_delta_rmse"].max()
        <= float(guards_cfg["max_worst_well_regression"])
    ),
    "raw_improved_folds_at_least_minimum": bool(
        int((fold_metrics["catboost_delta_rmse"] < 0.0).sum())
        >= int(guards_cfg["min_improved_folds"])
    ),
    "raw_worst_well_regression": float(by_well["catboost_delta_rmse"].max()),
    "raw_improved_folds": int((fold_metrics["catboost_delta_rmse"] < 0.0).sum()),
    "fixed_blend_overall_delta_rmse": float(
        blend_readout.set_index("surface").at["overall", "delta_rmse"]
    ),
    "inference_allowed": False,
}
raw_guard_keys = [
    key
    for key, value in guard.items()
    if key.startswith("raw_") and isinstance(value, bool)
]
guard["all_raw_guards_pass"] = bool(all(guard[key] for key in raw_guard_keys))
guard["inference_allowed"] = guard["all_raw_guards_pass"]

display(metrics)
display(fold_metrics)
display(stress_metrics)
display(blend_readout)
display(by_well.head(30))
print(json.dumps(guard, indent=2), flush=True)

# %% [markdown]
# ## 8. Feature importance, manifests, SHA, and generated artifacts

# %%
importance = pd.DataFrame(importance_rows)
importance_mean = (
    importance.groupby(["model", "feature"], as_index=False)
    .agg(
        mean_importance=("importance", "mean"),
        std_importance=("importance", "std"),
        fold_records=("importance", "size"),
    )
    .sort_values("mean_importance", ascending=False)
)

plot_frame = importance_mean.nlargest(40, "mean_importance").sort_values(
    "mean_importance"
)
fig, ax = plt.subplots(figsize=(12, max(7, 0.25 * len(plot_frame))))
ax.barh(plot_frame["feature"], plot_frame["mean_importance"], color="#8a4f9e")
ax.set_title(f"{OUTPUT_PREFIX}: mean CatBoost feature importance")
ax.set_xlabel("mean FeatureImportance")
fig.tight_layout()
importance_plot_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_importance_mean_top.png"
fig.savefig(importance_plot_path, dpi=160)
plt.close(fig)

feature_schema = pd.DataFrame(
    {
        "position": np.arange(len(all_feature_columns), dtype=np.int32),
        "feature": all_feature_columns,
        "feature_group": [
            "nested_selector_rank_slot" if feature.startswith("nsel_") else "exp218_base"
            for feature in all_feature_columns
        ],
    }
)
oof_output = pd.DataFrame(
    {
        "id": base_frame["id"].astype(str),
        "well": base_frame["well"].astype(str),
        "outer_fold": fold_assignment,
        "last_known_tvt": anchor,
        "target_tvt": truth,
        "exp238_lgb_mean_tvt": parent_tvt,
        "catboost_public_cb0_tvt": catboost_tvt,
        f"fixed_blend_catboost_w{blend_weight:.2f}": blend_tvt,
    }
)

metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_metrics.csv"
fold_metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_fold_metrics.csv"
bucket_metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
hidden_metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_hidden_like_metrics.csv"
by_well_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_by_well.csv"
blend_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_blend_readout.csv"
schema_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_schema.csv"
importance_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_importance_mean.csv"
oof_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
manifest_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_model_manifest.json"
guard_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_guard.json"
summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.json"

metrics.to_csv(metrics_path, index=False)
fold_metrics.to_csv(fold_metrics_path, index=False)
bucket_metrics.to_csv(bucket_metrics_path, index=False)
hidden_like_metrics.to_csv(hidden_metrics_path, index=False)
by_well.to_csv(by_well_path, index=False)
blend_readout.to_csv(blend_path, index=False)
feature_schema.to_csv(schema_path, index=False)
importance_mean.to_csv(importance_path, index=False)
oof_output.to_csv(oof_path, index=False, compression="gzip")
guard_path.write_text(json.dumps(guard, indent=2))

model_manifest = {
    "experiment": nested(CONFIG, "experiment.name"),
    "parent": nested(CONFIG, "lineage.parent"),
    "variant": "catboost_public_cb0",
    "public_source": nested(CONFIG, "model.public_source"),
    "source_exact_params": configured_cb0,
    "source_exact_fit_params": configured_fit,
    "runtime_only_params": runtime_params,
    "folds": int(nested(CONFIG, "model.folds")),
    "model_count": len(model_rows),
    "parent_control_retraining": False,
    "selector_retraining": False,
    "feature_count": len(all_feature_columns),
    "feature_schema_sha256": sha256_path(schema_path),
    "feature_matrix_sha256": feature_matrix_sha,
    "models": model_rows,
}
manifest_path.write_text(json.dumps(model_manifest, indent=2))

artifact_sha = {
    "parameter_audit": sha256_path(parameter_audit_path),
    "metrics": sha256_path(metrics_path),
    "fold_metrics": sha256_path(fold_metrics_path),
    "bucket_metrics": sha256_path(bucket_metrics_path),
    "hidden_like_metrics": sha256_path(hidden_metrics_path),
    "by_well": sha256_path(by_well_path),
    "blend_readout": sha256_path(blend_path),
    "feature_schema": sha256_path(schema_path),
    "feature_importance_mean": sha256_path(importance_path),
    "oof_decompressed": sha256_path(oof_path, decompressed=True),
    "guard": sha256_path(guard_path),
    "model_manifest": sha256_path(manifest_path),
}
summary = {
    "experiment": nested(CONFIG, "experiment.name"),
    "status": (
        "train_completed_all_raw_guards_pass"
        if guard["all_raw_guards_pass"]
        else "train_completed_raw_guard_failed"
    ),
    "route": nested(CONFIG, "experiment.route"),
    "parent": nested(CONFIG, "lineage.parent"),
    "active_variants": 1,
    "catboost_configs": 1,
    "folds": 5,
    "models": len(model_rows),
    "parent_control_retraining": False,
    "selector_retraining": False,
    "public_source": nested(CONFIG, "model.public_source"),
    "feature_count": len(all_feature_columns),
    "feature_matrix_sha256": feature_matrix_sha,
    "metrics": metrics.to_dict("records"),
    "guard": guard,
    "input_contract": input_contract,
    "artifact_sha256": artifact_sha,
    "artifacts": {
        "parameter_audit": parameter_audit_path.name,
        "metrics": metrics_path.name,
        "fold_metrics": fold_metrics_path.name,
        "bucket_metrics": bucket_metrics_path.name,
        "hidden_like_metrics": hidden_metrics_path.name,
        "by_well": by_well_path.name,
        "blend_readout": blend_path.name,
        "feature_schema": schema_path.name,
        "feature_importance_mean": importance_path.name,
        "feature_importance_plot": importance_plot_path.name,
        "oof_predictions": oof_path.name,
        "model_manifest": manifest_path.name,
        "guard": guard_path.name,
    },
    "elapsed_seconds": round(time.time() - STARTED, 3),
}
summary_path.write_text(json.dumps(summary, indent=2))
print(
    json.dumps(
        {**summary, "summary_sha256": sha256_path(summary_path)},
        indent=2,
    ),
    flush=True,
)

# %% [markdown]
# # exp258 downstream TVT LightGBM train (GPU)

# %% [markdown]
# ## Contents
# 1. Imports and cost contract
# 2. Selector artifact and hard guard
# 3. Candidate and fold reconstruction
# 4. exp218 380-feature surface
# 5. Final 3 x 5 LightGBM training
# 6. Historical exp238 OOF comparison
# 7. Artifacts and final guard

# %%
from __future__ import annotations

import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path("experiments/exp258_gr_residual_noise_transplant_augmentation")
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_PREFIX = str(CONFIG["experiment"]["name"])
OUTPUT_DIR = (
    Path("/kaggle/working/artifacts")
    if Path("/kaggle/working").exists()
    else PACKAGE_DIR / "artifacts"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def import_file(name: str, candidates: list[Path], *, reset_settings: bool = False):
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


engine = import_file(
    "exp238_engine",
    [
        Path("experiments/exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218")
        / "nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py",
        PACKAGE_DIR / "exp238_source/nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py",
        PACKAGE_DIR / "nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py",
    ],
)
engine.OUTPUT_PREFIX = OUTPUT_PREFIX
exp237 = import_file(
    "exp237_source",
    [
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183")
        / "hmm_exp226_candidate_selector_on_exp183.py",
        Path("/kaggle/input/exp237-hmm-exp226-candidate-selector-exp183-train")
        / "hmm_exp226_candidate_selector_on_exp183.py",
        PACKAGE_DIR / "exp237_source/hmm_exp226_candidate_selector_on_exp183.py",
        PACKAGE_DIR / "hmm_exp226_candidate_selector_on_exp183.py",
    ],
    reset_settings=True,
)
exp218 = import_file(
    "exp218_source",
    [
        Path("experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148")
        / "gr_wavelet_rotation_confidence_features_on_exp148.py",
        Path("/kaggle/input/exp218-gr-wavelet-rotation-exp148-train")
        / "gr_wavelet_rotation_confidence_features_on_exp148.py",
        PACKAGE_DIR / "exp218_source/gr_wavelet_rotation_confidence_features_on_exp148.py",
        PACKAGE_DIR / "gr_wavelet_rotation_confidence_features_on_exp148.py",
    ],
    reset_settings=True,
)
exp218_settings = import_file(
    "exp218_settings_source",
    [
        PACKAGE_DIR / "exp218_source/settings.py",
        Path("experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/settings.py"),
    ],
)


def cfg_get(config: dict[str, Any], dotted_key: str):
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


if CONFIG["execution"]["selected_stage"] != "final_train":
    raise RuntimeError(
        "This notebook trains 15 GPU boosters. Set execution.selected_stage=final_train "
        "only after the selector guard passes and the user approves the GPU run."
    )
if CONFIG["runtime"]["kaggle"]["user_approval"]["authorized"] is not True:
    raise RuntimeError("Kaggle GPU final train is not authorized in config.yaml")
print(
    json.dumps(
        {
            "runtime": "GPU",
            "variant": CONFIG["execution"]["final_train_variant"],
            "final_configs": 3,
            "folds": 5,
            "boosters": 15,
            "parent_control_retraining": False,
        },
        indent=2,
    )
)

# %% [markdown]
# ## 2. Selector artifact and hard guard

# %%
selector_dirs = [
    Path(CONFIG["data"]["selector_artifact_dir_local"]),
    Path("/kaggle/input/exp258-gr-residual-noise-transplant-selector-train/artifacts"),
    Path("/kaggle/input/exp258-gr-residual-noise-transplant-selector-train"),
    Path("/kaggle/input/notebooks/kentookumura/exp258-gr-residual-noise-transplant-selector-train/artifacts"),
    Path("/kaggle/input/notebooks/kentookumura/exp258-gr-residual-noise-transplant-selector-train"),
]
summary_name = f"{OUTPUT_PREFIX}_selector_summary.json"
SELECTOR_DIR = next(
    (path for path in selector_dirs if (path / summary_name).exists()), None
)
if SELECTOR_DIR is None and Path("/kaggle/input").exists():
    matches = list(Path("/kaggle/input").rglob(summary_name))
    SELECTOR_DIR = matches[0].parent if matches else None
if SELECTOR_DIR is None:
    raise FileNotFoundError(f"selector artifact directory not found: {selector_dirs}")
selector_summary = json.loads((SELECTOR_DIR / summary_name).read_text())
expected_variant = str(CONFIG["execution"]["final_train_variant"])
if selector_summary.get("variant") != expected_variant:
    raise RuntimeError(
        f"selector variant mismatch: {selector_summary.get('variant')} != {expected_variant}"
    )
if selector_summary.get("decision", {}).get("guard_pass") is not True:
    raise RuntimeError("selector guard failed; final GPU LightGBM training is forbidden")
if int(selector_summary.get("selector_model_count", -1)) != 20:
    raise RuntimeError("selector artifact must contain exactly 20 nested rankers")
print(
    json.dumps(
        {
            "selector_dir": str(SELECTOR_DIR),
            "selector_status": selector_summary["status"],
            "selector_variant": selector_summary["variant"],
            "selector_guard": selector_summary["decision"],
        },
        indent=2,
    )
)

# %% [markdown]
# ## 3. Candidate and fold reconstruction

# %%
parent_config = exp237.load_config()
parent_config.setdefault("inference", {})["use_test_base_as_dense_auxiliary"] = False
candidates = exp237.candidate_specs_from_config(parent_config)
required = exp237.build_required_columns(parent_config, candidates)
selector_frame, _ = exp237.load_train_feature_cache(
    cache_path=exp237.get_nested(parent_config, "data.exp099_train_feature_cache_local"),
    schema_path=exp237.get_nested(parent_config, "data.exp099_train_feature_schema_local"),
    required_columns=required,
    max_rows=None,
)
selector_frame, _, _ = exp237.add_feature_enrichment(
    selector_frame, parent_config, max_rows=None
)
selector_frame, _, _ = exp237.add_cluster_prior_confidence_features(
    selector_frame, parent_config, max_rows=None
)
selector_frame, _, _ = exp237.add_hmm_exp226_candidate_sources(
    selector_frame, parent_config
)
candidate_columns = [item.column for item in candidates]
candidate_names = [item.name for item in candidates]
if candidate_names != list(CONFIG["model"]["selector"]["candidates"]):
    raise ValueError("candidate bank changed after selector training")
runtime_outer, runtime_inner = engine.deterministic_outer_inner_splits(
    selector_frame,
    int(CONFIG["validation"]["outer_folds"]),
    int(CONFIG["validation"]["inner_folds"]),
)
fold_manifest = pd.read_csv(SELECTOR_DIR / f"{OUTPUT_PREFIX}_fold_manifest.csv")
if len(fold_manifest) != 20:
    raise ValueError("selector fold manifest count mismatch")
nested_outer = engine.load_nested_fold_contracts(
    SELECTOR_DIR, len(selector_frame), int(CONFIG["validation"]["outer_folds"])
)
fold_checks = [
    {
        "outer_fold": fold,
        "runtime_reconstructed_fold_match": bool(
            np.array_equal(train_rows, np.sort(runtime_outer[fold][0]))
            and np.array_equal(valid_rows, np.sort(runtime_outer[fold][1]))
        ),
    }
    for fold, (train_rows, valid_rows) in enumerate(nested_outer)
]
if not all(item["runtime_reconstructed_fold_match"] for item in fold_checks):
    raise ValueError("runtime outer folds differ from selector artifact folds")
outer = nested_outer
display(fold_manifest)
print(fold_checks)

# %% [markdown]
# ## 4. exp218 380-feature surface

# %%
exp218_config = yaml.safe_load(Path(exp218.__file__).with_name("config.yaml").read_text())
base_frame, base_feature_columns, _ = exp218.load_exp072_full_replay_cache_frame(
    cfg_get(exp218_config, "data.exp072_train_feature_cache_local"), max_rows=None
)
resolved_train_dir = exp218_settings.ExperimentPaths().train_data_dir
if not resolved_train_dir.exists():
    raise FileNotFoundError(
        f"resolved competition train directory does not exist: {resolved_train_dir}"
    )
base_frame, _ = exp218.add_anchor_columns(base_frame, resolved_train_dir)
projection_cfg = cfg_get(exp218_config, "model.u_projection") or {}
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
    cfg_get(exp218_config, "data.learned_likelihood_train_features_local"),
    schema_path=cfg_get(exp218_config, "data.learned_likelihood_train_feature_schema_local"),
    summary_path=cfg_get(exp218_config, "data.learned_likelihood_train_summary_local"),
)
learned, learned_groups, _ = exp218.build_learned_likelihood_features(
    learned_source,
    base_frame,
    cfg_get(exp218_config, "model.learned_likelihood_features") or {},
)
learned_columns = [column for column in learned if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    base_frame, learned.reset_index(drop=True), learned_columns
)
grwr, grwr_groups, _, _ = exp218.build_gr_wavelet_rotation_confidence_features(
    base_frame,
    train_dir=resolved_train_dir,
    config=cfg_get(exp218_config, "model.gr_wavelet_rotation_confidence_features") or {},
)
grwr_columns = [column for column in grwr if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    base_frame, grwr.reset_index(drop=True), grwr_columns
)
del projection, learned_source, learned, grwr
gc.collect()
feature_groups = {**projection_groups, **learned_groups, **grwr_groups}
parent_variant = next(
    item
    for item in cfg_get(exp218_config, "model.feature_ablation.active_variants")
    if item.get("name") == "gr_wavelet_rotation_confidence_addonly"
)
exp218_features = exp218.feature_columns_for_variant(
    base_feature_columns, feature_groups, parent_variant
)
if len(exp218_features) != 380:
    raise ValueError(f"exp218 base feature count changed: {len(exp218_features)}")
if not base_frame[engine.KEYS].astype(str).reset_index(drop=True).equals(
    selector_frame[engine.KEYS].astype(str).reset_index(drop=True)
):
    raise ValueError("exp218 and selector frames are not row aligned")
selector_min = selector_frame[
    [*engine.KEYS, "last_known_tvt", *candidate_columns]
].copy()
del selector_frame, runtime_inner
selector_frame = selector_min
gc.collect()
print(
    {
        "rows": len(base_frame),
        "wells": int(base_frame.well.nunique()),
        "base_features": len(exp218_features),
        "selector_features": 35,
        "total_features": 415,
    }
)

# %% [markdown]
# ## 5. Final 3 x 5 LightGBM training

# %%
mode = cfg_get(exp218_config, "model.training.modes.gpu_repro_guard_dp_threads8")
final_params = exp218.apply_mode_overrides(
    exp218.exp063_lgb_config_family(fast=False), mode
)
if len(final_params) != 3 or len(outer) != 5:
    raise ValueError("final training cost contract must remain 3 configs x 5 folds")
metrics, predictions, importance, model_manifest = engine.fit_final_nested_addonly(
    base_frame,
    selector_frame,
    exp218_features,
    outer,
    SELECTOR_DIR,
    candidate_columns,
    final_params,
    OUTPUT_DIR,
)
if len(model_manifest) != 15:
    raise AssertionError("final training did not produce 15 LightGBM models")
display(metrics)
display(importance.groupby("model").head(30))

# %% [markdown]
# ## 6. Historical exp238 OOF comparison

# %%
def find_historical_oof() -> Path:
    filename = (
        "exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218_"
        "final_oof_predictions.csv.gz"
    )
    configured = Path(CONFIG["data"]["historical_final_oof_local"])
    candidates_path = [
        configured,
        Path("/kaggle/input/exp238-nested-final-train/artifacts") / filename,
        Path("/kaggle/input/exp238-nested-final-train") / filename,
    ]
    resolved = next((path for path in candidates_path if path.exists()), None)
    if resolved is None and Path("/kaggle/input").exists():
        matches = list(Path("/kaggle/input").rglob(filename))
        resolved = matches[0] if matches else None
    if resolved is None:
        raise FileNotFoundError("historical exp238 final OOF is required for the final guard")
    return resolved


historical_path = find_historical_oof()
historical = pd.read_csv(historical_path, dtype={"id": str, "well": str})
historical = historical.rename(columns={"lgb_mean_pred_tvt": "historical_pred_tvt"})
if not historical[engine.KEYS].reset_index(drop=True).equals(
    predictions[engine.KEYS].astype(str).reset_index(drop=True)
):
    raise ValueError("historical exp238 OOF is not aligned with exp258 OOF")
truth = (
    predictions.last_known_tvt.to_numpy(np.float64)
    + predictions.target.to_numpy(np.float64)
)
current_pred = predictions.lgb_mean_pred_tvt.to_numpy(np.float64)
historical_pred = historical.historical_pred_tvt.to_numpy(np.float64)


def rmse(mask: np.ndarray, values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values[mask] - truth[mask]))))


comparison_rows = []


def add_comparison(scope: str, label: str, mask: np.ndarray) -> None:
    if not mask.any():
        return
    current = rmse(mask, current_pred)
    baseline = rmse(mask, historical_pred)
    comparison_rows.append(
        {
            "scope": scope,
            "label": label,
            "rows": int(mask.sum()),
            "exp258_rmse": current,
            "exp238_rmse": baseline,
            "delta_rmse": current - baseline,
        }
    )


all_rows = np.ones(len(predictions), dtype=bool)
add_comparison("overall", "all", all_rows)
md_since = base_frame.md_since.to_numpy(np.float32)
distance_bucket = pd.cut(
    md_since,
    bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
    labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
    include_lowest=True,
)
for label in distance_bucket.categories:
    add_comparison("distance_bucket", str(label), np.asarray(distance_bucket == label))
for outer_fold, (_, valid_rows) in enumerate(outer):
    mask = np.zeros(len(predictions), dtype=bool)
    mask[valid_rows] = True
    add_comparison("outer_fold", str(outer_fold), mask)

fold_assignment_path = exp237.find_artifact(exp237.EXP115_FOLD_ASSIGNMENTS)
fold_assignment = pd.read_csv(fold_assignment_path, dtype={"well_id": str})
role_columns = [
    "verification_like_spatial_role",
    "verification_like_typewell_purged_role",
]
role_lookup = fold_assignment.set_index("well_id")
prediction_wells = predictions.well.astype(str)
for column in role_columns:
    role = prediction_wells.map(role_lookup[column])
    add_comparison("hidden_like", column, role.eq("valid").to_numpy())

well_rows = []
for well, row_positions in predictions.groupby("well", sort=True).groups.items():
    mask = np.zeros(len(predictions), dtype=bool)
    mask[np.asarray(list(row_positions), dtype=np.int64)] = True
    current = rmse(mask, current_pred)
    baseline = rmse(mask, historical_pred)
    well_rows.append(
        {
            "well": str(well),
            "rows": int(mask.sum()),
            "exp258_rmse": current,
            "exp238_rmse": baseline,
            "delta_rmse": current - baseline,
        }
    )
comparison = pd.DataFrame(comparison_rows)
by_well = pd.DataFrame(well_rows)
display(comparison)
display(by_well.sort_values("delta_rmse", ascending=False).head(20))

# %% [markdown]
# ## 7. Artifacts and final guard

# %%
guard_cfg = CONFIG["validation"]["final_guard"]


def delta(scope: str, label: str) -> float:
    matched = comparison.loc[
        comparison.scope.eq(scope) & comparison.label.eq(label), "delta_rmse"
    ]
    if len(matched) != 1:
        raise ValueError(f"missing comparison scope={scope} label={label}")
    return float(matched.iloc[0])


hidden_delta = float(
    comparison.loc[comparison.scope.eq("hidden_like"), "delta_rmse"].max()
)
fold_nonworse = int(
    (comparison.loc[comparison.scope.eq("outer_fold"), "delta_rmse"] <= 0.0).sum()
)
worst_well_delta = float(by_well.delta_rmse.max())
checks = {
    "overall_nonworse": delta("overall", "all")
    <= float(guard_cfg["max_overall_delta_vs_exp238"]),
    "near_nonworse": delta("distance_bucket", "000_050")
    <= float(guard_cfg["max_near_delta_vs_exp238"]),
    "longtail_nonworse": delta("distance_bucket", "1000_plus")
    <= float(guard_cfg["max_longtail_delta_vs_exp238"]),
    "hidden_like_nonworse": hidden_delta
    <= float(guard_cfg["max_hidden_like_delta_vs_exp238"]),
    "worst_well_within_tolerance": worst_well_delta
    <= float(guard_cfg["max_worst_well_regression_vs_exp238"]),
    "three_of_five_folds_nonworse": fold_nonworse >= 3,
}
guard_pass = bool(all(checks.values()))
decision = {
    "guard_pass": guard_pass,
    "checks": checks,
    "overall_delta_rmse": delta("overall", "all"),
    "near_delta_rmse": delta("distance_bucket", "000_050"),
    "longtail_delta_rmse": delta("distance_bucket", "1000_plus"),
    "max_hidden_like_delta_rmse": hidden_delta,
    "worst_well_delta_rmse": worst_well_delta,
    "folds_nonworse": fold_nonworse,
}

metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_final_metrics.csv"
predictions_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_final_oof_predictions.csv.gz"
importance_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_feature_importance_mean.csv"
manifest_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_final_model_manifest.json"
comparison_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_final_comparison_vs_exp238.csv"
by_well_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_final_by_well_vs_exp238.csv"
metrics.to_csv(metrics_path, index=False)
predictions.to_csv(predictions_path, index=False, compression="gzip")
importance.to_csv(importance_path, index=False)
manifest_path.write_text(json.dumps(model_manifest, indent=2))
comparison.to_csv(comparison_path, index=False)
by_well.to_csv(by_well_path, index=False)
summary = {
    "status": (
        "final_guard_passed_inference_can_be_implemented"
        if guard_pass
        else "final_guard_failed_inference_forbidden"
    ),
    "selector_summary": selector_summary,
    "decision": decision,
    "metrics": metrics.to_dict(orient="records"),
    "rows": len(base_frame),
    "wells": int(base_frame.well.nunique()),
    "base_feature_count": len(exp218_features),
    "selector_feature_count": 35,
    "final_model_count": len(model_manifest),
    "historical_oof": str(historical_path),
    "sha256": {
        "metrics": engine._sha(metrics_path),
        "predictions_decompressed": engine._sha(predictions_path, decompressed=True),
        "feature_importance": engine._sha(importance_path),
        "model_manifest": engine._sha(manifest_path),
        "comparison": engine._sha(comparison_path),
        "by_well": engine._sha(by_well_path),
    },
}
summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_final_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

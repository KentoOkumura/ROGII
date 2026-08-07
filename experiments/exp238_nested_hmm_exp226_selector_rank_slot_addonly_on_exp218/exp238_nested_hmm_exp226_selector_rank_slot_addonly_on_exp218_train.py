# %% [markdown]
# # exp238 exp218 add-only train (GPU)

# %% [markdown]
# ## Contents
# 1. Imports and source resolution
# 2. Selector artifact and guard contract
# 3. Candidate and fold reconstruction
# 4. exp218 feature surface assembly
# 5. Fold-specific nested score loading
# 6. Final LightGBM training
# 7. Metrics and generated artifacts

# %%
from __future__ import annotations

import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path("experiments/exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218")
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_DIR = Path("/kaggle/working/artifacts") if Path("/kaggle/working").exists() else PACKAGE_DIR / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def import_file(name: str, candidates: list[Path], *, reset_settings: bool = False):
    path = next((p for p in candidates if p.exists()), None)
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


engine = import_file("exp238_engine", [PACKAGE_DIR / "nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py"])
exp237 = import_file("exp237_source", [
    Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/hmm_exp226_candidate_selector_on_exp183.py"),
    Path("/kaggle/input/exp237-hmm-exp226-candidate-selector-exp183-train/hmm_exp226_candidate_selector_on_exp183.py"),
    PACKAGE_DIR / "exp237_source/hmm_exp226_candidate_selector_on_exp183.py",
    PACKAGE_DIR / "hmm_exp226_candidate_selector_on_exp183.py",
], reset_settings=True)
exp218 = import_file("exp218_source", [
    Path("experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/gr_wavelet_rotation_confidence_features_on_exp148.py"),
    Path("/kaggle/input/exp218-gr-wavelet-rotation-exp148-train/gr_wavelet_rotation_confidence_features_on_exp148.py"),
    PACKAGE_DIR / "exp218_source/gr_wavelet_rotation_confidence_features_on_exp148.py",
    PACKAGE_DIR / "gr_wavelet_rotation_confidence_features_on_exp148.py",
], reset_settings=True)
exp218_settings = import_file("exp218_settings_source", [
    PACKAGE_DIR / "exp218_source/settings.py",
    Path("experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/settings.py"),
])


def cfg_get(config: dict, dotted_key: str):
    value = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value

# %% [markdown]
# ## 2. Selector artifact and guard contract

# %%
selector_dirs = [
    Path(CONFIG["data"]["selector_artifact_dir_local"]),
    Path("/kaggle/input/exp238-nested-selector-train/artifacts"),
    Path("/kaggle/input/exp238-nested-selector-train"),
    Path("/kaggle/input/notebooks/kentookumura/exp238-nested-selector-train/artifacts"),
    Path("/kaggle/input/notebooks/kentookumura/exp238-nested-selector-train"),
]
SELECTOR_DIR = next((p for p in selector_dirs if (p / f"{engine.OUTPUT_PREFIX}_selector_summary.json").exists()), None)
if SELECTOR_DIR is None and Path("/kaggle/input").exists():
    summary_matches = list(Path("/kaggle/input").rglob(f"{engine.OUTPUT_PREFIX}_selector_summary.json"))
    SELECTOR_DIR = summary_matches[0].parent if summary_matches else None
if SELECTOR_DIR is None:
    raise FileNotFoundError(f"selector artifact directory not found: {selector_dirs}")
selector_summary = json.loads((SELECTOR_DIR / f"{engine.OUTPUT_PREFIX}_selector_summary.json").read_text())
guard_override = bool(CONFIG["validation"]["guard"].get("allow_failure_for_final_train", False))
if not selector_summary["decision"]["guard_pass"] and not guard_override:
    raise RuntimeError("selector safety guard failed; final GPU training is forbidden")
print(json.dumps({"selector_status": selector_summary["status"], "decision": selector_summary["decision"], "guard_override": guard_override, "guard_override_reason": CONFIG["validation"]["guard"].get("override_reason"), "final_boosters": CONFIG["model"]["final_boosters"], "parent_control_retraining": False}, indent=2))

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
    required_columns=required, max_rows=None,
)
selector_frame, _, _ = exp237.add_feature_enrichment(selector_frame, parent_config, max_rows=None)
selector_frame, _, _ = exp237.add_cluster_prior_confidence_features(selector_frame, parent_config, max_rows=None)
selector_frame, _, _ = exp237.add_hmm_exp226_candidate_sources(selector_frame, parent_config)
candidate_columns = [item.column for item in candidates]
outer, inner = engine.deterministic_outer_inner_splits(selector_frame, int(CONFIG["validation"]["outer_folds"]), int(CONFIG["validation"]["inner_folds"]))
fold_manifest = pd.read_csv(SELECTOR_DIR / f"{engine.OUTPUT_PREFIX}_fold_manifest.csv")
if len(fold_manifest) != int(CONFIG["validation"]["outer_folds"]) * int(CONFIG["validation"]["inner_folds"]):
    raise ValueError("selector fold manifest count mismatch")
display(fold_manifest)

# %% [markdown]
# ## 4. exp218 feature surface assembly

# %%
exp218_config = yaml.safe_load(Path(exp218.__file__).with_name("config.yaml").read_text())
base_frame, base_feature_columns, _ = exp218.load_exp072_full_replay_cache_frame(cfg_get(exp218_config, "data.exp072_train_feature_cache_local"), max_rows=None)
resolved_train_dir = exp218_settings.ExperimentPaths().train_data_dir
if not resolved_train_dir.exists():
    raise FileNotFoundError(f"resolved competition train directory does not exist: {resolved_train_dir}")
print({"resolved_train_dir": str(resolved_train_dir), "horizontal_files": len(list(resolved_train_dir.glob("*__horizontal_well.csv")))})
base_frame, _ = exp218.add_anchor_columns(base_frame, resolved_train_dir)
projection_cfg = cfg_get(exp218_config, "model.u_projection") or {}
projection, projection_groups, _ = exp218.build_u_projection_features(base_frame, source_specs=dict(projection_cfg.get("sources") or {}), degree=int(projection_cfg.get("degree", 3)), robust_iters=int(projection_cfg.get("robust_iters", 3)), clip_sigma=float(projection_cfg.get("clip_sigma", 4.0)))
projection_columns = [c for c in projection if c not in {"id", "well"}]
exp218._assign_aligned_float32_columns(base_frame, projection.reset_index(drop=True), projection_columns)
learned_source, _ = exp218.load_learned_likelihood_ml_features(cfg_get(exp218_config, "data.learned_likelihood_train_features_local"), schema_path=cfg_get(exp218_config, "data.learned_likelihood_train_feature_schema_local"), summary_path=cfg_get(exp218_config, "data.learned_likelihood_train_summary_local"))
learned, learned_groups, _ = exp218.build_learned_likelihood_features(learned_source, base_frame, cfg_get(exp218_config, "model.learned_likelihood_features") or {})
learned_columns = [c for c in learned if c not in {"id", "well"}]
exp218._assign_aligned_float32_columns(base_frame, learned.reset_index(drop=True), learned_columns)
grwr, grwr_groups, _, _ = exp218.build_gr_wavelet_rotation_confidence_features(base_frame, train_dir=resolved_train_dir, config=cfg_get(exp218_config, "model.gr_wavelet_rotation_confidence_features") or {})
grwr_columns = [c for c in grwr if c not in {"id", "well"}]
exp218._assign_aligned_float32_columns(base_frame, grwr.reset_index(drop=True), grwr_columns)
del projection, learned_source, learned, grwr
gc.collect()
feature_groups = {**projection_groups, **learned_groups, **grwr_groups}
parent_variant = next(v for v in cfg_get(exp218_config, "model.feature_ablation.active_variants") if v.get("name") == "gr_wavelet_rotation_confidence_addonly")
exp218_features = exp218.feature_columns_for_variant(base_feature_columns, feature_groups, parent_variant)
print({"rows": len(base_frame), "wells": base_frame.well.nunique(), "exp218_features": len(exp218_features)})

# %% [markdown]
# ## 5. Fold-specific nested score contract

# %%
nested_outer = engine.load_nested_fold_contracts(
    SELECTOR_DIR,
    len(selector_frame),
    int(CONFIG["validation"]["outer_folds"]),
)
print([
    {
        "outer_fold": i,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "runtime_reconstructed_fold_match": bool(
            np.array_equal(train_rows, np.sort(outer[i][0]))
            and np.array_equal(valid_rows, np.sort(outer[i][1]))
        ),
    }
    for i, (train_rows, valid_rows) in enumerate(nested_outer)
])
# The selector artifact role is the authoritative fold contract. Runtime
# GroupKFold reconstruction is diagnostic only because sklearn versions may
# assign groups differently while still preserving group isolation.
outer = nested_outer

# Only these columns are needed once candidate generation and fold auditing
# finish.  Releasing the hundreds of selector context columns is essential
# before allocating LightGBM's 3M-row train matrix.
selector_min = selector_frame[[*engine.KEYS, "last_known_tvt", *candidate_columns]].copy()
del selector_frame, nested_outer, inner
selector_frame = selector_min
gc.collect()

# %% [markdown]
# ## 6. Final LightGBM training

# %%
mode = cfg_get(exp218_config, "model.training.modes.gpu_repro_guard_dp_threads8")
final_params = exp218.apply_mode_overrides(exp218.exp063_lgb_config_family(fast=False), mode)
metrics, predictions, importance, model_manifest = engine.fit_final_nested_addonly(base_frame, selector_frame, exp218_features, outer, SELECTOR_DIR, candidate_columns, final_params, OUTPUT_DIR)
display(metrics)
display(importance.groupby("model").head(30))

# %% [markdown]
# ## 7. Metrics and generated artifacts

# %%
metrics_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_final_metrics.csv"
predictions_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_final_oof_predictions.csv.gz"
importance_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_feature_importance_mean.csv"
manifest_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_final_model_manifest.json"
metrics.to_csv(metrics_path, index=False)
predictions.to_csv(predictions_path, index=False, compression="gzip")
importance.to_csv(importance_path, index=False)
manifest_path.write_text(json.dumps(model_manifest, indent=2))
summary = {
    "status": "nested_addonly_final_train_completed",
    "selector_summary": selector_summary,
    "metrics": metrics.to_dict(orient="records"),
    "rows": len(base_frame), "wells": int(base_frame.well.nunique()),
    "base_feature_count": len(exp218_features),
    "sha256": {"metrics": engine._sha(metrics_path), "predictions_decompressed": engine._sha(predictions_path, decompressed=True), "model_manifest": engine._sha(manifest_path)},
}
(OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_final_summary.json").write_text(json.dumps(summary, indent=2))

# %% [markdown]
# # exp238 final inference (GPU)

# %% [markdown]
# ## Contents
# 1. Imports and source resolution
# 2. Approval, selector, and model contracts
# 3. Current-test candidate and selector surface
# 4. exp218 current-test feature surface
# 5. In-memory saved-selector rank-slot features
# 6. Saved-booster inference
# 7. Submission and generated artifacts

# %%
from __future__ import annotations

import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from IPython.display import display

PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path("experiments/exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218")
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
STARTED_AT = time.time()
OUTPUT_DIR = Path("/kaggle/working/artifacts") if Path("/kaggle/working").exists() else PACKAGE_DIR / "artifacts"
SUBMISSION_PATH = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else PACKAGE_DIR / "submission.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def import_file(name: str, candidates: list[Path], *, reset_settings: bool = False):
    path = next((item for item in candidates if item.exists()), None)
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
    [PACKAGE_DIR / "nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py"],
)
exp218 = import_file(
    "exp218_source",
    [
        PACKAGE_DIR / "exp218_source/gr_wavelet_rotation_confidence_features_on_exp148.py",
        Path("experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/gr_wavelet_rotation_confidence_features_on_exp148.py"),
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
exp237_settings = import_file(
    "settings",
    [
        PACKAGE_DIR / "exp237_source/settings.py",
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/settings.py"),
    ],
    reset_settings=True,
)
exp237 = import_file(
    "hmm_exp226_candidate_selector_on_exp183",
    [
        PACKAGE_DIR / "exp237_source/hmm_exp226_candidate_selector_on_exp183.py",
        Path(
            "experiments/exp237_hmm_exp226_candidate_selector_on_exp183/"
            "hmm_exp226_candidate_selector_on_exp183.py"
        ),
    ],
)
rawtest = import_file(
    "exp237_rawtest_inference",
    [
        PACKAGE_DIR / "exp237_source/rawtest_inference.py",
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/rawtest_inference.py"),
    ],
)
exp145_settings = import_file(
    "settings",
    [
        PACKAGE_DIR / "exp145_source/settings.py",
        Path("experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/settings.py"),
    ],
    reset_settings=True,
)
exp145 = import_file(
    "exp145_dynamic_generator",
    [
        PACKAGE_DIR / "exp145_source/learned_likelihood_rawtest_feature_generator_parity.py",
        Path(
            "experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/"
            "learned_likelihood_rawtest_feature_generator_parity.py"
        ),
    ],
)
exp226 = import_file(
    "exp226_dynamic_inference",
    [
        PACKAGE_DIR / "exp226_source/connortynan_k16_reproduction.py",
        Path(
            "experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/"
            "connortynan_k16_reproduction.py"
        ),
    ],
)


def cfg_get(config: dict, dotted_key: str, default=None):
    value = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def find_one(filename: str) -> Path:
    local = PACKAGE_DIR / filename
    if local.exists():
        return local
    matches = list(Path("/kaggle/input").rglob(filename)) if Path("/kaggle/input").exists() else []
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {filename}, found {matches}")
    return matches[0]

# %% [markdown]
# ## 2. Approval, selector, and model contracts

# %%
if CONFIG.get("inference", {}).get("status") != "user_authorized_2026_07_13":
    raise RuntimeError("exp238 raw-test inference is not user-authorized")
selector_summary_path = find_one(f"{engine.OUTPUT_PREFIX}_selector_summary.json")
selector_manifest_path = find_one(f"{engine.OUTPUT_PREFIX}_selector_model_manifest.csv")
final_manifest_path = find_one(f"{engine.OUTPUT_PREFIX}_final_model_manifest.json")
selector_summary = json.loads(selector_summary_path.read_text())
selector_manifest = pd.read_csv(selector_manifest_path)
model_manifest = json.loads(final_manifest_path.read_text())
if int(selector_summary.get("selector_model_count", -1)) != 20:
    raise ValueError("selector train summary does not certify all 20 nested selector models")
if len(selector_manifest) != 20:
    raise ValueError(f"expected 20 saved selectors, got {len(selector_manifest)}")
if len(model_manifest) != int(CONFIG["model"]["final_boosters"]):
    raise ValueError(f"expected 15 final models, got {len(model_manifest)}")

candidate_columns = [str(value) for value in selector_summary["candidate_columns"]]
context_columns = [str(value) for value in selector_summary["context_columns"]]
expected_selector_features = [
    *context_columns,
    "candidate_code",
    "candidate_minus_anchor",
    "candidate_abs_minus_anchor",
]
outer_fold_count = int(CONFIG["validation"]["outer_folds"])
inner_fold_count = int(CONFIG["validation"]["inner_folds"])
resolved_selectors: dict[int, list[tuple[dict, Path]]] = {
    outer_fold: [] for outer_fold in range(outer_fold_count)
}
for raw_item in selector_manifest.to_dict(orient="records"):
    item = dict(raw_item)
    outer_fold = int(item["outer_fold"])
    filename = Path(str(item["file"])).name
    candidates = list(selector_manifest_path.parent.rglob(filename))
    if not candidates and Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob(filename))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one saved selector {filename}, found {candidates}")
    if engine._sha(candidates[0]) != str(item["sha256"]):
        raise ValueError(f"saved selector SHA mismatch: {filename}")
    booster = lgb.Booster(model_file=str(candidates[0]))
    if booster.feature_name() != expected_selector_features:
        raise ValueError(f"selector feature schema mismatch: {filename}")
    del booster
    resolved_selectors[outer_fold].append((item, candidates[0]))
for outer_fold, items in resolved_selectors.items():
    inner_folds = sorted(int(item[0]["inner_fold"]) for item in items)
    if inner_folds != list(range(inner_fold_count)):
        raise ValueError(f"outer fold {outer_fold} selector coverage mismatch: {inner_folds}")

resolved_models: list[tuple[dict, Path]] = []
for item in model_manifest:
    filename = Path(item["file"]).name
    candidates = list(final_manifest_path.parent.rglob(filename))
    if not candidates and Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob(filename))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one saved model {filename}, found {candidates}")
    if engine._sha(candidates[0]) != item["sha256"]:
        raise ValueError(f"saved model SHA mismatch: {filename}")
    resolved_models.append((item, candidates[0]))

first_booster = lgb.Booster(model_file=str(resolved_models[0][1]))
model_feature_columns = list(first_booster.feature_name())
selector_feature_columns = [column for column in model_feature_columns if column.startswith("nsel_")]
base_feature_columns = [column for column in model_feature_columns if not column.startswith("nsel_")]
if len(base_feature_columns) != 380 or len(selector_feature_columns) != 35:
    raise ValueError({"base_features": len(base_feature_columns), "selector_features": len(selector_feature_columns)})
del first_booster
print(json.dumps({
    "selector_train_status": selector_summary["status"],
    "selector_models": sum(len(items) for items in resolved_selectors.values()),
    "selector_inference_mode": "current_test_in_memory_no_fit",
    "final_models": len(resolved_models),
    "base_features": len(base_feature_columns),
    "selector_features": len(selector_feature_columns),
    "submission_requested": bool(CONFIG["inference"]["submission_requested"]),
}, indent=2))

# %% [markdown]
# ## 3. Current-test candidate and selector surface

# %%
try:
    from public_notebook_replay_audit import build_replay_test_frame, configure_public_runtime
except ModuleNotFoundError:
    from src.public_notebook_replay_audit import build_replay_test_frame, configure_public_runtime

paths = exp218_settings.ExperimentPaths()
if Path("/kaggle/input").exists():
    paths.require_kaggle_runtime()
paths.ensure_output_dirs()
exp218_config = yaml.safe_load((PACKAGE_DIR / "exp218_source/config.yaml").read_text())
configure_public_runtime(
    data_dir=paths.raw_data_dir,
    output_dir=OUTPUT_DIR,
    n_jobs=int(cfg_get(exp218_config, "runtime.num_workers", 8)),
    pf_seeds=int(cfg_get(exp218_config, "generator.rawtest_replay.pf_seeds", 128)),
    pf_particles=int(cfg_get(exp218_config, "generator.rawtest_replay.pf_particles", 500)),
    fast=False,
    use_gpu="auto",
)
base_test_frame, replay_meta = build_replay_test_frame()
base_test_frame["id"] = base_test_frame["id"].astype(str)
base_test_frame["well"] = base_test_frame["well"].astype(str)

parent_config = yaml.safe_load((PACKAGE_DIR / "exp237_source/config.yaml").read_text())
parent_config.setdefault("inference", {})["use_test_base_as_dense_auxiliary"] = True
selector_specs = exp237.candidate_specs_from_config(parent_config)
if [item.column for item in selector_specs] != candidate_columns:
    raise ValueError("saved selector candidate schema differs from current-test config")
selector_frame = rawtest._base_candidates(base_test_frame.copy())
selector_frame, hmm_meta = rawtest._attach_hmm_candidates(selector_frame, parent_config, paths)

exp226_config = yaml.safe_load((PACKAGE_DIR / "exp226_source/config.yaml").read_text())
exp226_work_dir = OUTPUT_DIR / "exp226_current_test"
exp226_work_dir.mkdir(parents=True, exist_ok=True)
exp226_paths = SimpleNamespace(
    train_data_dir=paths.train_data_dir,
    test_data_dir=paths.test_data_dir,
    sample_submission_path=paths.sample_submission_path,
    submission_path=exp226_work_dir / "submission.csv",
    artifacts_dir=exp226_work_dir / "artifacts",
    metrics_path=exp226_work_dir / "metrics.json",
)
exp226_meta = exp226.run_inference(exp226_paths, exp226_config)
exp226_values = pd.read_csv(exp226_paths.submission_path, dtype={"id": str})
exp226_values["well"] = exp226_values["id"].str.rsplit("_", n=1).str[0]
exp226_values = exp226_values.rename(
    columns={"tvt": "exp226_v6_k16_geometry_gr_u_projection"}
)
selector_frame = rawtest._merge_required(
    selector_frame,
    exp226_values[["id", "well", "exp226_v6_k16_geometry_gr_u_projection"]],
    name="dynamic exp226 K16 current-test",
)
selector_frame, multiobs_meta = rawtest._attach_multiobs(
    selector_frame, parent_config, paths
)
selector_frame, _, enrichment_meta = exp237.add_feature_enrichment(
    selector_frame, parent_config, max_rows=None
)
missing_context = [column for column in context_columns if column not in selector_frame.columns]
nonfinite_context_counts: dict[str, int] = {}
for column in context_columns:
    if column not in selector_frame.columns:
        selector_frame[column] = np.float32(np.nan)
        nonfinite_context_counts[column] = int(len(selector_frame))
        continue
    values = pd.to_numeric(selector_frame[column], errors="coerce").to_numpy(np.float32)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.float32(np.nan)
        selector_frame[column] = values
        nonfinite_context_counts[column] = int(bad.sum())
selector_frame, selector_candidate_values, _ = rawtest._test_candidate_features(
    selector_frame, selector_specs
)
if not np.isfinite(selector_candidate_values).all():
    raise ValueError("current-test selector candidate values contain non-finite values")
if not selector_frame[["id", "well"]].equals(base_test_frame[["id", "well"]]):
    raise ValueError("current-test selector surface changed replay row order")

selector_cache_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_current_test_selector_surface.csv.gz"
selector_frame.to_csv(selector_cache_path, index=False, compression="gzip")
learned_output_dir = OUTPUT_DIR / "exp145_current_test"
learned_generator_summary = exp145.run_generator(
    output_dir=learned_output_dir,
    mode="rawtest",
    train_cache_path=None,
    rawtest_cache_path=selector_cache_path,
    exp111_schema_path=None,
    exp111_manifest_path=None,
    exp112_schema_path=None,
    max_rows=None,
)
learned_source_path = Path(
    learned_generator_summary["outputs"]["rawtest_ml_features"]["path"]
)
learned_source = pd.read_csv(learned_source_path, dtype={"id": str, "well": str})
print({
    "current_test_rows": len(selector_frame),
    "current_test_wells": selector_frame.well.nunique(),
    "selector_candidates": len(candidate_columns),
    "selector_missing_context_columns": len(missing_context),
    "learned_schema_parity": learned_generator_summary["generated_schema"]["schema_parity_pass"],
})

# %% [markdown]
# ## 4. exp218 current-test feature surface

# %%
test_frame, anchor_meta = exp218.add_inference_anchor_columns(
    base_test_frame, paths.test_data_dir
)

projection_cfg = cfg_get(exp218_config, "model.u_projection", {}) or {}
projection, _, _ = exp218.build_u_projection_features(
    test_frame,
    source_specs=dict(projection_cfg.get("sources") or {}),
    degree=int(projection_cfg.get("degree", 3)),
    robust_iters=int(projection_cfg.get("robust_iters", 3)),
    clip_sigma=float(projection_cfg.get("clip_sigma", 4.0)),
)
projection_columns = [column for column in projection.columns if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(test_frame, projection.reset_index(drop=True), projection_columns)

if not exp218.learned_feature_keys_match(learned_source, test_frame):
    raise ValueError("dynamic exp145 learned-feature keys differ from current test")
learned, _, _ = exp218.build_learned_likelihood_features(
    learned_source,
    test_frame,
    cfg_get(exp218_config, "model.learned_likelihood_features", {}) or {},
)
learned_columns = [column for column in learned.columns if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(test_frame, learned.reset_index(drop=True), learned_columns)

grwr, _, _, grwr_meta = exp218.build_gr_wavelet_rotation_confidence_features(
    test_frame,
    train_dir=paths.test_data_dir,
    config=cfg_get(exp218_config, "model.gr_wavelet_rotation_confidence_features", {}) or {},
)
grwr_columns = [column for column in grwr.columns if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(test_frame, grwr.reset_index(drop=True), grwr_columns)
del projection, learned_source, learned, grwr, base_test_frame, exp226_values
gc.collect()
missing_base = [column for column in base_feature_columns if column not in test_frame.columns]
if missing_base:
    raise ValueError(f"raw-test exp218 surface missing model features: {missing_base[:40]}")
print({"test_rows": len(test_frame), "test_wells": test_frame.well.nunique(), "exp218_features": len(base_feature_columns)})

# %% [markdown]
# ## 5. In-memory saved-selector rank-slot features

# %%
def predict_outer_selector_features(outer_fold: int) -> pd.DataFrame:
    if len(selector_frame) != len(test_frame):
        raise ValueError("selector and exp218 current-test row counts differ")
    if not np.allclose(
        selector_frame["last_known_tvt"].to_numpy(np.float32),
        test_frame["last_known_tvt"].to_numpy(np.float32),
        atol=1e-3,
        rtol=0.0,
    ):
        raise ValueError(f"outer fold {outer_fold} selector and exp218 anchors differ")
    scores = np.zeros((len(selector_frame), len(candidate_columns)), dtype=np.float32)
    selector_rows = np.arange(len(selector_frame), dtype=np.int64)
    for item, model_path in resolved_selectors[outer_fold]:
        booster = lgb.Booster(model_file=str(model_path))
        scores += engine.predict_candidate_errors(
            booster,
            selector_frame,
            selector_rows,
            candidate_columns,
            context_columns,
            chunk_rows=int(CONFIG["model"]["selector"]["predict_chunk_rows"]),
        ) / np.float32(inner_fold_count)
        del booster
        gc.collect()
    if not np.isfinite(scores).all():
        raise ValueError(f"outer fold {outer_fold} selector scores contain non-finite values")
    extra = engine.rank_slot_features(
        selector_frame,
        np.arange(len(selector_frame), dtype=np.int64),
        scores,
        candidate_columns,
    ).reset_index(drop=True)
    if list(extra.columns) != selector_feature_columns:
        raise ValueError(f"outer fold {outer_fold} selector feature schema mismatch")
    if not np.isfinite(extra.to_numpy(np.float32)).all():
        raise ValueError(f"outer fold {outer_fold} selector features contain non-finite values")
    return extra

# %% [markdown]
# ## 6. Saved-booster inference

# %%
pred_delta = np.zeros(len(test_frame), dtype=np.float32)
loaded_models = []
for outer_fold in range(int(CONFIG["validation"]["outer_folds"])):
    extra = predict_outer_selector_features(outer_fold)
    x_matrix = pd.concat(
        [test_frame[base_feature_columns].reset_index(drop=True), extra], axis=1
    )
    if list(x_matrix.columns) != model_feature_columns:
        raise ValueError(f"outer fold {outer_fold} final feature order mismatch")
    if not np.isfinite(x_matrix.to_numpy(np.float32)).all():
        raise ValueError(f"outer fold {outer_fold} final feature matrix contains non-finite values")
    fold_models = [
        (item, model_path)
        for item, model_path in resolved_models
        if int(item["outer_fold"]) == outer_fold
    ]
    if len(fold_models) != int(CONFIG["model"]["final_configs"]):
        raise ValueError(f"outer fold {outer_fold} final model coverage mismatch")
    for item, model_path in fold_models:
        booster = lgb.Booster(model_file=str(model_path))
        if list(booster.feature_name()) != model_feature_columns:
            raise ValueError(f"model feature schema mismatch: {model_path.name}")
        pred_delta += booster.predict(
            x_matrix, num_iteration=int(item["best_iteration"])
        ).astype(np.float32) / np.float32(len(resolved_models))
        loaded_models.append({
            "model": item["model"],
            "outer_fold": outer_fold,
            "selector_score_outer_fold": outer_fold,
            "file": model_path.name,
            "sha256": item["sha256"],
            "best_iteration": item["best_iteration"],
        })
        del booster
        gc.collect()
    display(extra.head(3))
    del extra, x_matrix
    gc.collect()
pred_tvt = test_frame["last_known_tvt"].to_numpy(np.float32) + pred_delta
predictions = pd.DataFrame({
    "id": test_frame["id"].astype(str),
    "well": test_frame["well"].astype(str),
    "last_known_tvt": test_frame["last_known_tvt"].to_numpy(np.float32),
    "pred_delta": pred_delta,
    "pred_tvt": pred_tvt,
})

# %% [markdown]
# ## 7. Submission and generated artifacts

# %%
sample = pd.read_csv(paths.sample_submission_path, dtype={"id": str})
submission = sample[["id"]].merge(
    predictions[["id", "pred_tvt"]], on="id", how="left", validate="one_to_one"
).rename(columns={"pred_tvt": "tvt"})
if len(submission) != len(sample) or submission["tvt"].isna().any() or not np.isfinite(submission["tvt"]).all():
    raise ValueError("submission contract failed")
prediction_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_inference_test_predictions.csv.gz"
schema_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_inference_feature_schema.csv"
summary_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_inference_summary.json"
predictions.to_csv(prediction_path, index=False, compression="gzip")
pd.DataFrame({"feature_index": np.arange(len(model_feature_columns)), "feature": model_feature_columns}).to_csv(schema_path, index=False)
submission.to_csv(SUBMISSION_PATH, index=False)
summary = {
    "status": "hidden_safe_inference_completed_not_submitted",
    "runtime_seconds": float(time.time() - STARTED_AT),
    "rows": int(len(predictions)),
    "wells": int(predictions.well.nunique()),
    "model_count": len(loaded_models),
    "feature_count": len(model_feature_columns),
    "base_feature_count": len(base_feature_columns),
    "selector_feature_count": len(selector_feature_columns),
    "fallback_rows": 0,
    "prediction_min": float(pred_tvt.min()),
    "prediction_max": float(pred_tvt.max()),
    "prediction_mean": float(pred_tvt.mean()),
    "prediction_std": float(pred_tvt.std()),
    "selector_summary": selector_summary,
    "loaded_models": loaded_models,
    "sources": {
        "replay": replay_meta,
        "anchor": anchor_meta,
        "hmm": hmm_meta,
        "exp226_dynamic": exp226_meta,
        "multiobs": multiobs_meta,
        "selector_enrichment": enrichment_meta,
        "selector_missing_context_columns": missing_context,
        "selector_nonfinite_context_counts": nonfinite_context_counts,
        "learned_dynamic": learned_generator_summary,
        "grwr": exp218._jsonable(grwr_meta),
    },
    "sha256": {
        "predictions_decompressed": engine._sha(prediction_path, decompressed=True),
        "feature_schema": engine._sha(schema_path),
        "submission": engine._sha(SUBMISSION_PATH),
        "current_test_selector_surface_decompressed": engine._sha(
            selector_cache_path, decompressed=True
        ),
        "selector_model_manifest": engine._sha(selector_manifest_path),
        "train_model_manifest": engine._sha(final_manifest_path),
    },
    "notes": [
        "Predictions average all 15 saved nested-addonly LightGBM boosters.",
        "Each final LightGBM uses selector scores generated in memory from its matching four saved inner models.",
        "All 20 selectors are loaded from selector train; no selector is fitted during inference.",
        "All row-dependent base, HMM, exp226 K16, multiobs, exp145 learned, and GRWR features are regenerated from the current test.",
        "No public-test row artifact or precomputed selector-score CSV participates in prediction.",
        "No Kaggle competition submission is made by this notebook.",
    ],
}
summary_path.write_text(json.dumps(exp218._jsonable(summary), indent=2) + "\n")
print(json.dumps(exp218._jsonable(summary), indent=2))
display(submission.head(20))

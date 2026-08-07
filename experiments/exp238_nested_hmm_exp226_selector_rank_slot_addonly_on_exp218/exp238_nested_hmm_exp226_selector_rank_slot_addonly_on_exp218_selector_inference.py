# %% [markdown]
# # exp238 saved nested-selector inference (CPU)

# %% [markdown]
# ## Contents
# 1. Imports and source resolution
# 2. Approval and saved-model contracts
# 3. Raw-test candidate surface
# 4. Fold-specific saved-selector inference
# 5. Metrics and generated artifacts

# %%
from __future__ import annotations

import gc
import importlib.util
import json
import sys
from pathlib import Path

import lightgbm as lgb
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


def import_file(name: str, candidates: list[Path]):
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Cannot resolve {name}: {candidates}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_one(filename: str) -> Path:
    local = PACKAGE_DIR / filename
    if local.exists():
        return local
    matches = list(Path("/kaggle/input").rglob(filename)) if Path("/kaggle/input").exists() else []
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {filename}, found {matches}")
    return matches[0]


engine = import_file(
    "exp238_engine",
    [PACKAGE_DIR / "nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py"],
)
exp237_settings = import_file(
    "settings",
    [
        PACKAGE_DIR / "exp237_source/settings.py",
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/settings.py"),
    ],
)
exp237 = import_file(
    "hmm_exp226_candidate_selector_on_exp183",
    [
        PACKAGE_DIR / "exp237_source/hmm_exp226_candidate_selector_on_exp183.py",
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/hmm_exp226_candidate_selector_on_exp183.py"),
    ],
)
rawtest = import_file(
    "exp237_rawtest_inference",
    [
        PACKAGE_DIR / "exp237_source/rawtest_inference.py",
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/rawtest_inference.py"),
    ],
)

# %% [markdown]
# ## 2. Approval and saved-model contracts

# %%
if CONFIG.get("inference", {}).get("status") != "user_authorized_2026_07_13":
    raise RuntimeError("exp238 raw-test inference is not user-authorized")
parent_config = yaml.safe_load((PACKAGE_DIR / "exp237_source/config.yaml").read_text())
parent_config.setdefault("inference", {})["use_test_base_as_dense_auxiliary"] = True
paths = exp237_settings.ExperimentPaths()
if Path("/kaggle/input").exists():
    paths.require_kaggle_runtime()
paths.ensure_output_dirs()

selector_summary_path = find_one(f"{engine.OUTPUT_PREFIX}_selector_summary.json")
selector_manifest_path = find_one(f"{engine.OUTPUT_PREFIX}_selector_model_manifest.csv")
selector_summary = json.loads(selector_summary_path.read_text())
selector_manifest = pd.read_csv(selector_manifest_path)
outer_fold_count = int(CONFIG["validation"]["outer_folds"])
inner_fold_count = int(CONFIG["validation"]["inner_folds"])
expected_model_count = outer_fold_count * inner_fold_count
if len(selector_manifest) != expected_model_count:
    raise ValueError(f"expected {expected_model_count} saved selectors, got {len(selector_manifest)}")
if int(selector_summary.get("selector_model_count", -1)) != expected_model_count:
    raise ValueError("selector summary does not certify 20 saved models")
candidate_columns = [str(value) for value in selector_summary["candidate_columns"]]
context_columns = [str(value) for value in selector_summary["context_columns"]]
expected_feature_names = [
    *context_columns,
    "candidate_code",
    "candidate_minus_anchor",
    "candidate_abs_minus_anchor",
]

resolved_models: dict[int, list[tuple[dict, Path]]] = {
    outer_fold: [] for outer_fold in range(outer_fold_count)
}
loaded_model_audit: list[dict] = []
for raw_item in selector_manifest.to_dict(orient="records"):
    item = dict(raw_item)
    outer_fold = int(item["outer_fold"])
    inner_fold = int(item["inner_fold"])
    filename = Path(str(item["file"])).name
    candidates = list(selector_manifest_path.parent.rglob(filename))
    if not candidates and Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob(filename))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one saved selector {filename}, found {candidates}")
    model_path = candidates[0]
    if engine._sha(model_path) != str(item["sha256"]):
        raise ValueError(f"saved selector SHA mismatch: {filename}")
    manifest_feature_names = json.loads(str(item["feature_names_json"]))
    booster = lgb.Booster(model_file=str(model_path))
    if booster.feature_name() != expected_feature_names or manifest_feature_names != expected_feature_names:
        raise ValueError(f"selector feature schema mismatch: {filename}")
    del booster
    resolved_models[outer_fold].append((item, model_path))
    loaded_model_audit.append({
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "file": filename,
        "sha256": str(item["sha256"]),
        "best_iteration": int(item["best_iteration"]),
    })
for outer_fold, items in resolved_models.items():
    inner_folds = sorted(int(item[0]["inner_fold"]) for item in items)
    if inner_folds != list(range(inner_fold_count)):
        raise ValueError(f"outer fold {outer_fold} saved-selector coverage mismatch: {inner_folds}")
print(json.dumps({
    "selector_train_kernel_status": selector_summary["status"],
    "saved_selector_models": len(loaded_model_audit),
    "outer_folds": outer_fold_count,
    "inner_models_per_outer": inner_fold_count,
    "selector_training_in_this_notebook": False,
    "submission_requested": bool(CONFIG["inference"]["submission_requested"]),
}, indent=2))

# %% [markdown]
# ## 3. Raw-test candidate surface

# %%
candidates = exp237.candidate_specs_from_config(parent_config)
if [item.column for item in candidates] != candidate_columns:
    raise ValueError("saved selector candidate schema differs from raw-test config")
base_test, base_meta = rawtest._load_base_test_frame(parent_config, paths)
test_frame = rawtest._base_candidates(base_test)
test_frame, hmm_meta = rawtest._attach_hmm_candidates(test_frame, parent_config, paths)
test_frame, exp226_meta = rawtest._attach_exp226_candidate(test_frame, parent_config)
test_frame, multiobs_meta = rawtest._attach_multiobs(test_frame, parent_config, paths)
test_frame, _, test_enrichment_meta = exp237.add_feature_enrichment(
    test_frame, parent_config, max_rows=None
)
test_frame, test_candidate_values, _ = rawtest._test_candidate_features(test_frame, candidates)
if not np.isfinite(test_candidate_values).all():
    raise ValueError("raw-test candidate values contain non-finite values")

missing_context = [column for column in context_columns if column not in test_frame.columns]
nonfinite_context_counts: dict[str, int] = {}
for column in context_columns:
    if column not in test_frame.columns:
        test_frame[column] = np.float32(np.nan)
        nonfinite_context_counts[column] = int(len(test_frame))
        continue
    values = pd.to_numeric(test_frame[column], errors="coerce").to_numpy(np.float32)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.float32(np.nan)
        test_frame[column] = values
        nonfinite_context_counts[column] = int(bad.sum())
test_frame["target"] = np.float32(0.0)
print({
    "test_rows": len(test_frame),
    "test_wells": test_frame.well.nunique(),
    "available_context_features": len(context_columns) - len(missing_context),
    "missing_context_features_routed_as_lgb_missing": len(missing_context),
})
display(pd.DataFrame({"missing_context_feature": missing_context}).head(100))

# %% [markdown]
# ## 4. Fold-specific saved-selector inference

# %%
test_rows = np.arange(len(test_frame), dtype=np.int64)
score_columns = [f"pred_error__{name}" for name in candidate_columns]
score_artifacts: list[dict] = []
score_paths: list[Path] = []
for outer_fold in range(outer_fold_count):
    score_sum = np.zeros((len(test_frame), len(candidate_columns)), dtype=np.float32)
    for item, model_path in resolved_models[outer_fold]:
        booster = lgb.Booster(model_file=str(model_path))
        score_sum += engine.predict_candidate_errors(
            booster,
            test_frame,
            test_rows,
            candidate_columns,
            context_columns,
            chunk_rows=int(CONFIG["model"]["selector"]["predict_chunk_rows"]),
        ) / np.float32(inner_fold_count)
        del booster
        gc.collect()
    if not np.isfinite(score_sum).all():
        raise ValueError(f"outer fold {outer_fold} raw-test selector scores contain non-finite values")
    artifact = test_frame[[*engine.KEYS, "last_known_tvt", *candidate_columns]].copy()
    for index, name in enumerate(candidate_columns):
        artifact[f"pred_error__{name}"] = score_sum[:, index]
    score_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_rawtest_selector_scores_outer{outer_fold}.csv.gz"
    artifact.to_csv(score_path, index=False, compression="gzip")
    score_paths.append(score_path)
    score_artifacts.append({
        "outer_fold": outer_fold,
        "models": inner_fold_count,
        "file": score_path.name,
        "rows": int(len(artifact)),
        "sha256_decompressed": engine._sha(score_path, decompressed=True),
        "score_min": float(artifact[score_columns].to_numpy(np.float32).min()),
        "score_max": float(artifact[score_columns].to_numpy(np.float32).max()),
    })
    display(artifact.head(3))
    del artifact, score_sum
    gc.collect()

# %% [markdown]
# ## 5. Metrics and generated artifacts

# %%
loaded_manifest_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_rawtest_selector_model_manifest.json"
summary_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_rawtest_selector_summary.json"
loaded_manifest_path.write_text(json.dumps(loaded_model_audit, indent=2) + "\n")
summary = {
    "status": "rawtest_nested_selector_inference_completed_not_submitted",
    "rows": int(len(test_frame)),
    "wells": int(test_frame.well.nunique()),
    "candidate_columns": candidate_columns,
    "context_feature_count": len(context_columns),
    "selector_model_count": len(loaded_model_audit),
    "outer_fold_count": outer_fold_count,
    "inner_models_per_outer": inner_fold_count,
    "selector_training_executed": False,
    "missing_context_features": missing_context,
    "nonfinite_context_counts_routed_as_lgb_missing": nonfinite_context_counts,
    "score_artifacts": score_artifacts,
    "sources": {
        "saved_selector_summary": str(selector_summary_path),
        "saved_selector_manifest": str(selector_manifest_path),
        "base_test": base_meta,
        "hmm": hmm_meta,
        "exp226": exp226_meta,
        "multiobs": multiobs_meta,
        "test_enrichment": test_enrichment_meta,
    },
    "sha256": {
        "loaded_model_manifest": engine._sha(loaded_manifest_path),
        "train_selector_manifest": engine._sha(selector_manifest_path),
        "train_selector_summary": engine._sha(selector_summary_path),
    },
    "notes": [
        "No selector is fitted in this notebook; all 20 nested selector models are loaded from selector train.",
        "Each outer fold gets its own four-model averaged raw-test score surface.",
        "Unavailable raw-test context is left as NaN so LightGBM follows its learned missing-value routing.",
        "No Kaggle competition submission is made by this notebook.",
    ],
}
summary_path.write_text(json.dumps(exp237.to_jsonable(summary), indent=2, sort_keys=True) + "\n")
print(json.dumps(exp237.to_jsonable(summary), indent=2, sort_keys=True))

# %% [markdown]
# # exp238 nested selector train (CPU)

# %% [markdown]
# ## Contents
# 1. Imports and source resolution
# 2. Cost and input contract
# 3. Candidate surface assembly
# 4. Outer/inner fold contract
# 5. Bounded nested selector training
# 6. Safety guard
# 7. Fold-specific score artifacts and summary

# %%
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

# %% [markdown]
# ## 2. Cost and input contract

# %%
selector_cfg = CONFIG["model"]["selector"]
print(json.dumps({
    "runtime": "cpu",
    "selector_boosters": CONFIG["model"]["selector_boosters"],
    "max_train_long_rows_per_model": selector_cfg["max_train_long_rows_per_model"],
    "max_valid_long_rows_per_model": selector_cfg["max_valid_long_rows_per_model"],
    "predict_chunk_rows": selector_cfg["predict_chunk_rows"],
    "parent_control_retraining": False,
}, indent=2))

# %% [markdown]
# ## 3. Candidate surface assembly

# %%
parent_config = exp237.load_config()
# exp237's canonical config was later extended for raw-test inference. This is
# a train-side OOF rebuild, so never reuse the raw-test base as dense auxiliary.
parent_config.setdefault("inference", {})["use_test_base_as_dense_auxiliary"] = False
candidates = exp237.candidate_specs_from_config(parent_config)
required = exp237.build_required_columns(parent_config, candidates)
frame, source_meta = exp237.load_train_feature_cache(
    cache_path=exp237.get_nested(parent_config, "data.exp099_train_feature_cache_local"),
    schema_path=exp237.get_nested(parent_config, "data.exp099_train_feature_schema_local"),
    required_columns=required,
    max_rows=None,
)
frame, enrichment_columns, enrichment_meta = exp237.add_feature_enrichment(frame, parent_config, max_rows=None)
frame, cluster_columns, cluster_meta = exp237.add_cluster_prior_confidence_features(frame, parent_config, max_rows=None)
frame, external_columns, external_meta = exp237.add_hmm_exp226_candidate_sources(frame, parent_config)
frame, engineered_columns, candidate_values, oracle_labels = exp237.add_candidate_labels_and_features(frame, candidates, include_candidate_values=False)
context_columns = exp237.select_numeric_feature_columns(frame, parent_config, [*engineered_columns, *enrichment_columns, *cluster_columns, *external_columns])
candidate_columns = [item.column for item in candidates]
print({"rows": len(frame), "wells": frame.well.nunique(), "candidates": candidate_columns, "features": len(context_columns), "source": source_meta})

# %% [markdown]
# ## 4. Outer/inner fold contract

# %%
outer, inner = engine.deterministic_outer_inner_splits(frame, int(CONFIG["validation"]["outer_folds"]), int(CONFIG["validation"]["inner_folds"]))
fold_manifest_path = engine.save_fold_contract(OUTPUT_DIR, frame, outer, inner)
display(pd.read_csv(fold_manifest_path))

# %% [markdown]
# ## 5. Bounded nested selector training

# %%
nested, model_manifest = engine.fit_nested_selector_scores(
    frame, outer, inner, candidate_columns, context_columns,
    dict(selector_cfg["params"]), int(CONFIG["reproducibility"]["seed"]),
    output_dir=OUTPUT_DIR,
    max_train_long_rows=int(selector_cfg["max_train_long_rows_per_model"]),
    max_valid_long_rows=int(selector_cfg["max_valid_long_rows_per_model"]),
    predict_chunk_rows=int(selector_cfg["predict_chunk_rows"]),
)
model_manifest_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_selector_model_manifest.csv"
pd.DataFrame(model_manifest).to_csv(model_manifest_path, index=False)

# %% [markdown]
# ## 6. Safety guard

# %%
safety_parts, by_well_parts = [], []
for outer_fold, item in enumerate(nested):
    safety, by_well = engine.selector_safety_readout(frame, item["outer_valid"], item["valid_scores"], candidate_columns, "likpf_mean")
    safety["outer_fold"] = outer_fold
    by_well["outer_fold"] = outer_fold
    safety_parts.append(safety)
    by_well_parts.append(by_well)
safety = pd.concat(safety_parts, ignore_index=True)
by_well = pd.concat(by_well_parts, ignore_index=True)
guard = CONFIG["validation"]["guard"]
near_delta = float(safety.loc[safety.bucket.eq(guard["near_bucket"]), "delta_rmse"].mean())
global_delta = float(safety.loc[safety.bucket.eq("global"), "delta_rmse"].mean())
long_delta = float(safety.loc[safety.bucket.eq("1000_plus"), "delta_rmse"].mean())
worst = float(by_well.delta_rmse.max())
guard_pass = bool(near_delta <= float(guard["max_near_delta_rmse"]) and worst <= float(guard["max_worst_well_regression"]) and global_delta <= 0 and long_delta <= 0)
decision = {"guard_pass": guard_pass, "near_delta_rmse": near_delta, "global_delta_rmse": global_delta, "longtail_delta_rmse": long_delta, "worst_well_regression": worst}
print(json.dumps(decision, indent=2))

# %% [markdown]
# ## 7. Fold-specific score artifacts and summary

# %%
score_manifest = engine.save_nested_score_artifacts(OUTPUT_DIR, frame, nested, candidate_columns)
expected_score_sha = {
    int(key): str(value)
    for key, value in CONFIG["reproducibility"][
        "selector_v3_nested_score_sha256_decompressed"
    ].items()
}
actual_score_sha = {
    int(item["outer_fold"]): str(item["sha256_decompressed"])
    for item in score_manifest
}
if actual_score_sha != expected_score_sha:
    raise ValueError({
        "message": "saved-selector rerun differs from selector v3 scores used by final train",
        "expected": expected_score_sha,
        "actual": actual_score_sha,
    })
safety_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_selector_safety_metrics.csv"
by_well_path = OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_selector_by_well.csv"
safety.to_csv(safety_path, index=False)
by_well.to_csv(by_well_path, index=False)
summary = {
    "status": "selector_guard_passed_final_train_allowed" if guard_pass else "selector_guard_failed_final_train_forbidden",
    "rows": len(frame), "wells": int(frame.well.nunique()),
    "candidate_columns": candidate_columns, "context_feature_count": len(context_columns),
    "context_columns": context_columns, "selector_model_count": len(model_manifest),
    "selector_v3_nested_score_sha_contract": "pass",
    "decision": decision, "score_artifacts": score_manifest,
    "sha256": {"fold_manifest": engine._sha(fold_manifest_path), "selector_model_manifest": engine._sha(model_manifest_path), "safety": engine._sha(safety_path)},
}
(OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_selector_summary.json").write_text(json.dumps(summary, indent=2))
ax = safety.pivot(index="outer_fold", columns="bucket", values="delta_rmse").plot(kind="bar", figsize=(10, 4), title="Nested selector delta RMSE")
ax.axhline(0, color="black", linewidth=1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"{engine.OUTPUT_PREFIX}_selector_safety.png", dpi=140)
plt.show()
display(safety)
display(by_well.sort_values("delta_rmse", ascending=False).head(20))

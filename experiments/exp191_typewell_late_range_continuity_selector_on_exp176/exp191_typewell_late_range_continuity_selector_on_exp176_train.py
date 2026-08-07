# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp191_typewell_late_range_continuity_selector_on_exp176 train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and parent artifact contract
# 4. Continuity selector setup
# 5. Train-side posthoc audit
# 6. Metrics and generated artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from IPython.display import display
from settings import ExperimentPaths, get_nested, is_kaggle_runtime, load_config
from typewell_late_range_continuity_selector_on_exp176 import (
    EXP072_FEATURE_CACHE,
    EXP072_FEATURE_SCHEMA,
    EXP099_FEATURE_CACHE,
    EXP099_FEATURE_SCHEMA,
    EXP176_FEATURE_SCHEMA,
    EXP176_MANIFEST,
    candidate_specs_from_config,
    configured_raw_columns,
    find_artifact,
    read_typewell_late_range_context,
    run_typewell_late_range_continuity_selector,
    variant_specs_from_config,
)

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
paths = ExperimentPaths()
config = load_config()
output_dir = Path("/kaggle/working/artifacts") if is_kaggle_runtime() else paths.artifacts_dir
output_dir.mkdir(parents=True, exist_ok=True)

experiment = config["experiment"]
lineage = config["lineage"]
selector = config["selector"]
model = config["model"]

print("experiment:", experiment["name"])
print("route:", experiment["route"])
print("status:", experiment["status"])
print("parent:", lineage["parent"])
print("continuity_parent:", lineage["continuity_parent"])
print("output_dir:", output_dir)
print("kaggle_runtime:", is_kaggle_runtime())
print("new_boosters:", model["planned_boosters"])
print("saved_parent_boosters_used:", model["saved_parent_boosters_used"])

# %% [markdown]
# ## 3. Input and parent artifact contract

# %%
candidates = candidate_specs_from_config(config)
raw_columns = configured_raw_columns(config, candidates)
print("candidate_count:", len(candidates), [spec.name for spec in candidates])
print("required_source_columns:", len(raw_columns))
print("kernel_sources:", get_nested(config, "runtime.kaggle.kernel_sources"))

artifact_checks = [
    (
        "exp099_train_feature_cache",
        EXP099_FEATURE_CACHE,
        get_nested(config, "data.exp099_train_feature_cache_local"),
    ),
    (
        "exp099_train_feature_schema",
        EXP099_FEATURE_SCHEMA,
        get_nested(config, "data.exp099_train_feature_schema_local"),
    ),
    (
        "exp072_dense_feature_cache",
        EXP072_FEATURE_CACHE,
        get_nested(config, "data.exp072_train_feature_cache_local"),
    ),
    (
        "exp072_dense_feature_schema",
        EXP072_FEATURE_SCHEMA,
        get_nested(config, "data.exp072_feature_schema_local"),
    ),
    ("exp176_model_manifest", EXP176_MANIFEST, None),
    ("exp176_feature_schema", EXP176_FEATURE_SCHEMA, None),
]

resolved_rows = []
for label, filename, explicit_path in artifact_checks:
    try:
        resolved = find_artifact(
            filename,
            explicit_path=explicit_path,
            explicit_dir=get_nested(config, "data.exp176_artifact_dir_local")
            if label.startswith("exp176")
            else None,
        )
        resolved_rows.append({"label": label, "path": str(resolved), "exists": True})
    except FileNotFoundError as exc:
        resolved_rows.append({"label": label, "path": str(exc).splitlines()[0], "exists": False})

display(pd.DataFrame(resolved_rows))

main_cache = next(row for row in resolved_rows if row["label"] == "exp099_train_feature_cache")
if main_cache["exists"]:
    header = pd.read_csv(main_cache["path"], nrows=0).columns.tolist()
    missing = [column for column in raw_columns if column not in header]
    if missing:
        raise RuntimeError(f"main cache missing required columns: {missing}")
    print("main_cache_header_columns:", len(header))

# %% [markdown]
# ## 4. Continuity selector setup

# %%
prior = get_nested(config, "selector.typewell_late_range_prior") or {}
context, context_meta = read_typewell_late_range_context(
    train_dir=paths.train_data_dir,
    min_typewell_span=float(prior.get("min_typewell_span", 1.0)),
)
print("typewell_context:", json.dumps(context_meta, indent=2, sort_keys=True))
display(
    context[["well", "typewell_min", "typewell_max", "typewell_span", "known_last_pct"]]
    .sort_values("known_last_pct", ascending=False)
    .head(10)
)

viterbi_specs = variant_specs_from_config(config)
print("viterbi_variant_count:", len(viterbi_specs))
print("default_candidate:", selector["default_candidate"])
print("allowed_switch_candidates:", selector["allowed_switch_candidates"])
print("max_train_rows_per_fold_for_parent_long_medians:", selector["max_train_rows_per_fold"])
print(
    "long_row_feature_exclude_prefixes:",
    get_nested(config, "selector.long_models.row_feature_exclude_prefixes"),
)

# %% [markdown]
# ## 5. Train-side posthoc audit

# %%
summary = run_typewell_late_range_continuity_selector(
    output_dir=output_dir,
    cache_path=get_nested(config, "data.exp099_train_feature_cache_local"),
    schema_path=get_nested(config, "data.exp099_train_feature_schema_local"),
    max_rows=get_nested(config, "selector.max_rows"),
)

# %% [markdown]
# ## 6. Metrics and generated artifacts

# %%
metrics_path = output_dir / summary["artifacts"]["metrics"]
distribution_path = output_dir / summary["artifacts"]["selection_distribution"]
by_well_path = output_dir / summary["artifacts"]["by_well"]
score_summary_path = output_dir / summary["artifacts"]["score_summary"]

metrics = pd.read_csv(metrics_path)
distribution = pd.read_csv(distribution_path)
by_well = pd.read_csv(by_well_path)
score_summary = pd.read_csv(score_summary_path)

display(metrics.sort_values("rmse_tvt").head(20))
display(
    distribution.sort_values(["variant", "mode", "rate"], ascending=[True, True, False]).head(40)
)
display(by_well.sort_values("rmse_tvt", ascending=False).head(20))
display(score_summary)

print("best_viterbi_variant:", summary["best_viterbi_variant"])
print("best_viterbi_rmse_tvt:", summary["best_viterbi_rmse_tvt"])
print("delta_rmse_vs_likpf_mean:", summary["delta_rmse_vs_likpf_mean"])
print("recommendation:", summary["recommendation"])
print("summary_sha:", json.dumps(summary["sha256"], indent=2, sort_keys=True))

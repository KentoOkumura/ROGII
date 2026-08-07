# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp171_bimodal_posterior_pfbeam_candidate_audit train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and fixed candidate cache checks
# 4. Bimodal posterior candidate audit
# 5. Metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from bimodal_posterior_pfbeam_candidate_audit import (
    run_bimodal_posterior_pfbeam_candidate_audit,
)
from IPython.display import display
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

experiment_name = get_nested(config, "experiment.name")
route = get_nested(config, "experiment.route")
audit_config = get_nested(config, "audit") or {}
runtime_config = get_nested(config, "runtime.kaggle") or {}

print(f"experiment={experiment_name}")
print(f"route={route}")
print(f"train_dir={paths.train_data_dir}")
print(f"artifacts_dir={paths.artifacts_dir}")
print(f"mode={audit_config.get('mode')}")
print(f"gpu_enabled={runtime_config.get('enable_gpu')}")
print("filters=", [item.get("name") for item in audit_config.get("filters", [])])
print("temperatures=", audit_config.get("posterior_temperatures"))
print(
    "pfbeam_candidates=",
    [item.get("name") for item in audit_config.get("pfbeam_candidates", [])],
)

# %% [markdown]
# ## 3. Input and fixed candidate cache checks

# %%
train_dir = paths.train_data_dir
horizontal_files = sorted(train_dir.glob("*__horizontal_well.csv"))
typewell_files = sorted(train_dir.glob("*__typewell.csv"))

if not horizontal_files:
    raise FileNotFoundError(f"No horizontal well files found: {train_dir}")
if not typewell_files:
    raise FileNotFoundError(f"No typewell files found: {train_dir}")

display(
    {
        "horizontal_wells": len(horizontal_files),
        "typewells": len(typewell_files),
        "first_horizontal": str(horizontal_files[0]),
        "first_typewell": str(typewell_files[0]),
        "max_eval_rows_per_region_per_well": audit_config.get("max_eval_rows_per_region_per_well"),
        "shift_min_ft": audit_config.get("shift_min_ft"),
        "shift_max_ft": audit_config.get("shift_max_ft"),
        "shift_step_ft": audit_config.get("shift_step_ft"),
        "local_offsets_rows": audit_config.get("local_offsets_rows"),
        "candidate_cache": get_nested(config, "data.exp072_train_feature_cache_local"),
    }
)

# %% [markdown]
# ## 4. Bimodal posterior candidate audit

# %%
result = run_bimodal_posterior_pfbeam_candidate_audit(
    config=config,
    train_dir=paths.train_data_dir,
    output_dir=paths.artifacts_dir,
    metrics_path=paths.metrics_path,
)

summary = result["summary"]
candidate_metrics = result["candidate_metrics"]
bucket_metrics = result["bucket_metrics"]
well_metrics = result["well_metrics"]
gain_vs_commit = result["gain_vs_commit"]
input_summary = result["input_summary"]

print("summary")
display(summary)

# %% [markdown]
# ## 5. Metrics, diagnostics, and generated artifacts

# %%
print("candidate metrics")
display(candidate_metrics.head(120))

print("posterior gain vs commit")
display(gain_vs_commit.head(120))

print("bimodal bucket metrics")
display(
    bucket_metrics[bucket_metrics["bucket_type"].eq("bimodal_flag")]
    .sort_values(["eval_region", "bucket", "rmse_tvt", "candidate"])
    .head(120)
)

print("mode-separation bucket metrics")
display(
    bucket_metrics[bucket_metrics["bucket_type"].eq("mode_separation")]
    .sort_values(["eval_region", "bucket", "rmse_tvt", "candidate"])
    .head(120)
)

print("worst wells by candidate")
display(
    well_metrics.sort_values(
        ["candidate", "eval_region", "rmse_tvt"],
        ascending=[True, True, False],
    ).head(120)
)

print("input summary")
display(input_summary.head(40))

print("generated artifacts")
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")

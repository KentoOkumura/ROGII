# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp167_fft_denoised_gr_matching_audit train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input checks
# 4. FFT denoised GR matching audit
# 5. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from fft_denoised_gr_matching_audit import run_fft_denoised_gr_matching_audit
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

print(f"experiment={experiment_name}")
print(f"route={route}")
print(f"train_dir={paths.train_data_dir}")
print(f"artifacts_dir={paths.artifacts_dir}")
print(f"mode={audit_config.get('mode')}")
print("filters=", [item.get("name") for item in audit_config.get("filters", [])])

# %% [markdown]
# ## 3. Input checks

# %%
train_dir = paths.train_data_dir
horizontal_files = sorted(train_dir.glob("*__horizontal_well.csv"))
typewell_files = sorted(train_dir.glob("*__typewell.csv"))

if not horizontal_files:
    raise FileNotFoundError(f"No horizontal well files found: {train_dir}")
if not typewell_files:
    raise FileNotFoundError(f"No typewell files found: {train_dir}")

print(f"horizontal wells={len(horizontal_files)}")
print(f"typewells={len(typewell_files)}")
display(
    {
        "first_horizontal": str(horizontal_files[0]),
        "first_typewell": str(typewell_files[0]),
        "max_eval_rows_per_region_per_well": audit_config.get("max_eval_rows_per_region_per_well"),
        "shift_min_ft": audit_config.get("shift_min_ft"),
        "shift_max_ft": audit_config.get("shift_max_ft"),
        "shift_step_ft": audit_config.get("shift_step_ft"),
        "local_offsets_rows": audit_config.get("local_offsets_rows"),
    }
)

# %% [markdown]
# ## 4. FFT denoised GR matching audit

# %%
result = run_fft_denoised_gr_matching_audit(
    config=config,
    train_dir=paths.train_data_dir,
    output_dir=paths.artifacts_dir,
    metrics_path=paths.metrics_path,
)

summary = result["summary"]
filter_metrics = result["filter_metrics"]
bucket_metrics = result["bucket_metrics"]
well_metrics = result["well_metrics"]
gain_vs_raw = result["gain_vs_raw"]
input_summary = result["input_summary"]

print("summary")
display(summary)

# %% [markdown]
# ## 5. Metrics and generated artifacts

# %%
print("filter metrics")
display(filter_metrics)

print("raw vs denoised gain")
display(gain_vs_raw)

print("distance bucket metrics")
distance_buckets = bucket_metrics[bucket_metrics["bucket_type"].eq("distance")]
display(distance_buckets.sort_values(["eval_region", "filter", "bucket"]).head(80))

print("worst wells by filter/region")
display(
    well_metrics.sort_values(
        ["filter", "eval_region", "rmse_tvt"], ascending=[True, True, False]
    ).head(80)
)

print("input summary")
display(input_summary.head(20))

print("generated artifacts")
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")

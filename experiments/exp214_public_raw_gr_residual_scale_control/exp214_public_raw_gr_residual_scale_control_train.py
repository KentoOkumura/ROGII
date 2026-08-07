# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp214_public_raw_gr_residual_scale_control train
#
# Fixed public-like raw GR residual-scale likelihood-PF control on the
# exp072-compatible pseudo-tail surface.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and exp072 validation-surface checks
# 4. Public raw GR residual-scale PF / Beam control contract
# 5. Public-like raw GR residual-scale control generation
# 6. Metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from public_raw_gr_residual_scale_control import (
    parse_filter_specs,
    read_exp072_eval_cache,
    run_public_raw_gr_residual_scale_control,
    select_target_wells,
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
validation_surface_config = get_nested(config, "model.validation_surface") or {}
runtime_config = get_nested(config, "model.runtime") or {}
beam_config = get_nested(config, "model.beam") or {}
kaggle_runtime = get_nested(config, "runtime.kaggle") or {}
filters = parse_filter_specs(config)

print(f"experiment={experiment_name}")
print(f"route={route}")
print(f"train_dir={paths.train_data_dir}")
print(f"artifacts_dir={paths.artifacts_dir}")
display(
    {
        "parent": get_nested(config, "lineage.parent"),
        "references": get_nested(config, "lineage.references"),
        "mode": audit_config.get("mode"),
        "gpu_enabled": kaggle_runtime.get("enable_gpu"),
        "validation_surface": validation_surface_config,
        "pf_runtime": runtime_config,
        "beam": beam_config,
        "observation_variants": [spec.__dict__ for spec in filters],
        "primary_baseline": audit_config.get("primary_baseline"),
    }
)

# %% [markdown]
# ## 3. Input and exp072 validation-surface checks

# %%
train_dir = paths.train_data_dir
horizontal_files = sorted(train_dir.glob("*__horizontal_well.csv"))
typewell_files = sorted(train_dir.glob("*__typewell.csv"))

if not horizontal_files:
    raise FileNotFoundError(f"No horizontal well files found: {train_dir}")
if not typewell_files:
    raise FileNotFoundError(f"No typewell files found: {train_dir}")

validation_frame, validation_meta = read_exp072_eval_cache(config)
target_wells = select_target_wells(validation_frame, train_dir, config)

display(
    {
        "horizontal_wells": len(horizontal_files),
        "typewells": len(typewell_files),
        "validation_rows": len(validation_frame),
        "validation_wells": validation_frame["well"].nunique(),
        "target_wells": len(target_wells),
        "score_rows": get_nested(config, "validation.score_rows"),
        "exp072_cache": validation_meta["source"],
        "exp072_reference_candidates_present": validation_meta[
            "reference_candidates_present"
        ],
    }
)
display(target_wells.head(80))

if target_wells.empty:
    raise RuntimeError("No target wells selected for public raw GR residual-scale control.")

# %% [markdown]
# ## 4. Public raw GR residual-scale PF / Beam control contract

# %%
filter_names = [spec.name for spec in filters]
if filter_names[0] != "raw" or filters[0].transition != "classic":
    raise RuntimeError("The first observation variant must be raw classic baseline.")
if any(spec.kind != "raw" for spec in filters):
    raise RuntimeError("exp214 is a raw public-like control; calibrated variants are out of scope.")

display(
    {
        "observation_variants": [spec.__dict__ for spec in filters],
        "likelihood_scales": runtime_config.get("likelihood_scales"),
        "primary_scale": runtime_config.get("primary_scale"),
        "particles": runtime_config.get("particles"),
        "seed_count": runtime_config.get("seed_count"),
        "shared_seed_policy": get_nested(config, "reproducibility.seed_policy"),
        "stochastic_components": get_nested(config, "reproducibility.stochastic_components"),
        "expected_artifacts": audit_config.get("expected_train_artifacts"),
        "out_of_scope": get_nested(config, "notes"),
    }
)

# %% [markdown]
# ## 5. Public-like raw GR residual-scale control generation

# %%
result = run_public_raw_gr_residual_scale_control(config=config, paths=paths)

summary = result["summary"]
candidate_metrics = result["candidate_metrics"]
filter_delta_metrics = result["filter_delta_metrics"]
bucket_metrics = result["bucket_metrics"]
by_well = result["by_well"]
group_metrics = result["group_metrics"]
pf_diagnostics = result["pf_diagnostics"]
well_status = result["well_status"]

print("summary")
display(summary)

# %% [markdown]
# ## 6. Metrics, diagnostics, and generated artifacts

# %%
print("candidate metrics")
display(candidate_metrics.head(80))

print("candidate deltas versus primary public raw scale")
display(filter_delta_metrics.head(80))

print("PF diagnostics")
display(
    pf_diagnostics.groupby("filter", observed=True)
    .agg(
        wells=("well", "nunique"),
        rows=("rows", "sum"),
        gr_sigma=("gr_sigma", "mean"),
        ess_mean=("ess_mean", "mean"),
        resampling_rate=("resampling_rate", "mean"),
        log_likelihood_mean=("log_likelihood_mean", "mean"),
        seed_weight_max=("seed_weight_max", "mean"),
        primary_scale=("primary_scale", "first"),
    )
    .reset_index()
)

print("distance bucket metrics")
display(bucket_metrics.sort_values(["candidate", "distance_bucket"]).head(120))

print("group metrics")
display(group_metrics.sort_values(["group", "rmse"]).head(120))

print("worst wells")
display(
    by_well.sort_values(
        ["candidate", "delta_rmse_vs_primary_baseline"],
        ascending=[True, False],
        na_position="last",
    ).head(120)
)

print("well status")
display(
    well_status["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="wells")
)

print("generated artifacts")
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")

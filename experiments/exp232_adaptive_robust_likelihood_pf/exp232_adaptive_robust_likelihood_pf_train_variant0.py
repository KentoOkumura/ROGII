# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp232 adaptive robust PF — T=2 CPU split run
#
# This notebook runs the complete eligible-well surface for `temp_t2` only.
# It is intentionally independent from the T=4 notebook so both CPU kernels
# can execute concurrently without changing the scientific PF configuration.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Split-run selection and fixed PF contract
# 3. Raw data and exp209-reconstructed control checks
# 4. T=2 target-free likelihood / gate contract
# 5. T=2 PF generation with periodic progress
# 6. Metrics, coverage, and regression guards

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from copy import deepcopy

from IPython.display import display

from adaptive_robust_likelihood_pf import (
    activate_temperature_split_run,
    configured_temperature_variants,
    interval_well_set,
    read_exp072_eval_cache,
    run_adaptive_robust_likelihood_pf,
    select_target_wells,
    temperature_variants,
)
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Split-run selection and fixed PF contract

# %%
RUN_KEY = "temp_t2"
EXPECTED_NOTEBOOK_KIND = "train_variant0"

paths = ExperimentPaths()
config = deepcopy(load_config())
run_spec = activate_temperature_split_run(config, RUN_KEY)
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

runtime = get_nested(config, "model.runtime") or {}
gate = get_nested(config, "model.gate") or {}
interval_audit = get_nested(config, "model.interval_audit") or {}
validation = get_nested(config, "model.validation_surface") or {}
kaggle_runtime = get_nested(config, "runtime.kaggle") or {}
variants = temperature_variants(config)
configured_variants = configured_temperature_variants(config)
execution = get_nested(config, "execution") or {}

if run_spec.get("notebook_kind") != EXPECTED_NOTEBOOK_KIND:
    raise RuntimeError("temp_t2 split-run metadata does not match this notebook kind.")
if variants != [("temp_t2", 2.0)]:
    raise RuntimeError("train_variant0 must execute temp_t2 only.")

display(
    {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "parent": get_nested(config, "lineage.parent"),
        "split_run": RUN_KEY,
        "kernel_id": run_spec.get("kernel_id"),
        "title": run_spec.get("title"),
        "gpu_enabled": kaggle_runtime.get("enable_gpu"),
        "internet_enabled": kaggle_runtime.get("enable_internet"),
        "kernel_sources": kaggle_runtime.get("kernel_sources"),
        "particles": runtime.get("particles"),
        "seed_count": runtime.get("seed_count"),
        "seed_aggregation": runtime.get("seed_aggregation"),
        "temperature_variants": variants,
        "configured_temperature_variants": configured_variants,
        "execution": execution,
        "gate": gate,
        "interval_audit": interval_audit,
        "primary_control": get_nested(config, "audit.primary_baseline"),
    }
)

# %% [markdown]
# ## 3. Raw data and exp209-reconstructed control checks

# %%
train_dir = paths.train_data_dir
horizontal_files = sorted(train_dir.glob("*__horizontal_well.csv"))
typewell_files = sorted(train_dir.glob("*__typewell.csv"))
if not horizontal_files or not typewell_files:
    raise FileNotFoundError(f"Missing raw train horizontal/typewell files under {train_dir}")

validation_frame, validation_meta = read_exp072_eval_cache(config)
target_wells = select_target_wells(validation_frame, train_dir, config)
interval_wells = interval_well_set(target_wells, config)

display(
    {
        "horizontal_wells": len(horizontal_files),
        "typewells": len(typewell_files),
        "exp072_rows": len(validation_frame),
        "exp072_wells": validation_frame["well"].nunique(),
        "eligible_target_wells": len(target_wells),
        "interval_audit_wells": len(interval_wells),
        "exp072_cache": validation_meta["source"],
        "exp072_cache_decompressed_sha": validation_meta.get("source_decompressed_sha256"),
        "reference_candidates": validation_meta["reference_candidates_present"],
        "exp209_control": validation_meta.get("exp209_reconstructed_likpf_control"),
    }
)
display(target_wells.head(20))

if target_wells.empty:
    raise RuntimeError("No eligible exp072 pseudo-tail wells were selected.")
if get_nested(config, "model.validation_surface.max_target_wells") is not None:
    raise RuntimeError("exp232 must evaluate all eligible wells; max_target_wells must be null.")
if "likpf_mean" not in validation_meta["reference_candidates_present"]:
    raise RuntimeError("The exp209-reconstructed exp072 likpf_mean control is required.")

# %% [markdown]
# ## 4. T=2 target-free likelihood / gate contract

# %%
if configured_variants != [("temp_t2", 2.0), ("temp_t4", 4.0)]:
    raise RuntimeError("exp232 is scoped to the predeclared T=2 and T=4 variants only.")
if not bool(gate.get("enabled", False)):
    raise RuntimeError("The target-free adaptive gate must remain enabled.")
if get_nested(config, "model.outlier_mixture") is not None:
    raise RuntimeError("Outlier mixture belongs to its own backlog experiment and is forbidden in exp232.")
if runtime.get("particles") != 500 or runtime.get("seed_count") != 128:
    raise RuntimeError("Particles and seed_count are fixed to the exp072-compatible 500 x 128 setting.")
if runtime.get("seed_aggregation") != "mean":
    raise RuntimeError("Seed aggregation must remain the exp072 likpf_mean policy.")

display(
    {
        "control": "exp209-enriched reconstructed exp072_likpf_mean (T=1, no regeneration)",
        "new_particle_likelihood": "exp(-0.5 * residual^2 / 2) only on gated rows",
        "outside_gate": "T=1 exact baseline likelihood",
        "gate_inputs": [
            "pre-update normalized innovation",
            "raw GR change point",
            "short/long GR novelty",
            "pre-update ESS ratio",
            "pre-update max particle weight",
        ],
        "forbidden": [
            "true TVT / target / error / oracle gate",
            "global temperature",
            "outlier mixture",
            "transition/resampling/particle/seed changes",
            "inference and submission",
        ],
    }
)

# %% [markdown]
# ## 5. T=2 PF generation with periodic progress

# %%
result = run_adaptive_robust_likelihood_pf(
    config=config,
    paths=paths,
    validation_frame=validation_frame,
    validation_meta=validation_meta,
)
summary = result["summary"]
candidate_metrics = result["candidate_metrics"]
bucket_metrics = result["bucket_metrics"]
hidden_like_metrics = result["hidden_like_metrics"]
by_well = result["by_well"]
pf_diagnostics = result["pf_diagnostics"]
gate_diagnostics = result["gate_diagnostics"]
interval_metrics = result["interval_metrics"]
first_loss_by_well = result["first_loss_by_well"]
well_status = result["well_status"]

display(summary)

# %% [markdown]
# ## 6. Metrics, coverage, and regression guards

# %%
print("Overall candidate metrics")
display(candidate_metrics.head(20))

print("Distance buckets")
display(bucket_metrics.sort_values(["candidate", "distance_bucket"]).head(80))

print("Hidden-like subgroups")
if hidden_like_metrics.empty:
    print("Hidden-like split artifact was not available; see summary.hidden_like_metrics_available.")
else:
    display(hidden_like_metrics.sort_values(["subgroup", "candidate"]).head(40))

print("PF diagnostics")
display(
    pf_diagnostics.groupby(["variant", "temperature"], observed=True)
    .agg(
        wells=("well", "nunique"),
        rows=("rows", "sum"),
        gr_sigma=("gr_sigma", "mean"),
        ess_mean=("ess_mean", "mean"),
        resampling_rate=("resampling_rate", "mean"),
        log_likelihood_mean=("log_likelihood_mean", "mean"),
    )
    .reset_index()
)

print("Gate diagnostics")
display(
    gate_diagnostics.groupby(["variant", "temperature"], observed=True)
    .agg(
        wells=("well", "nunique"),
        rows=("rows", "sum"),
        gate_seed_fraction=("gate_seed_fraction_mean", "mean"),
        gate_any_seed_rows=("gate_any_seed_rows", "sum"),
        gate_all_seed_rows=("gate_all_seed_rows", "sum"),
    )
    .reset_index()
)

print("Sampled particle p05-p95 coverage")
display(interval_metrics.sort_values(["candidate", "slice"]))

print("First sampled particle loss by well")
display(first_loss_by_well[first_loss_by_well["ever_lost"]].head(80))

print("Worst well regressions versus exp072 likpf_mean")
display(
    by_well[by_well["candidate"].str.startswith("pf_temp_")]
    .sort_values("delta_rmse_vs_primary_baseline", ascending=False)
    .head(80)
)

print("Well execution status")
display(well_status["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="wells"))

print("Generated artifacts")
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")

print("Split-run execution state")
display(summary["execution"])

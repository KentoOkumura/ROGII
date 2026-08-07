# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp233_adaptive_outlier_mixture_likelihood_pf train
#
# Full exp072-compatible likelihood-PF audit. The exp209 enriched comparison
# cache reconstructs the Gaussian exp072 `likpf_mean` control; this notebook
# generates only the fixed, gated state-neutral Uniform-GR mixture variants.
# It never creates a submission.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Raw data and exp072 control checks
# 4. State-neutral outlier-mixture and gate contract
# 5. Full train-side PF generation
# 6. Metrics, coverage, and regression guards

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from IPython.display import display

from adaptive_outlier_mixture_likelihood_pf import (
    interval_well_set,
    outlier_mixture_variants,
    read_exp072_eval_cache,
    run_adaptive_outlier_mixture_likelihood_pf,
    select_target_wells,
)
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

runtime = get_nested(config, "model.runtime") or {}
gate = get_nested(config, "model.gate") or {}
mixture = get_nested(config, "model.outlier_mixture") or {}
interval_audit = get_nested(config, "model.interval_audit") or {}
temperature_comparison = get_nested(config, "comparison.temperature_experiment") or {}
kaggle_runtime = get_nested(config, "runtime.kaggle") or {}
variants = outlier_mixture_variants(config)

display(
    {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "parent": get_nested(config, "lineage.parent"),
        "temperature_comparison": "exp232_adaptive_robust_likelihood_pf",
        "status": get_nested(config, "experiment.status"),
        "gpu_enabled": kaggle_runtime.get("enable_gpu"),
        "internet_enabled": kaggle_runtime.get("enable_internet"),
        "kernel_sources": kaggle_runtime.get("kernel_sources"),
        "particles": runtime.get("particles"),
        "seed_count": runtime.get("seed_count"),
        "seed_aggregation": runtime.get("seed_aggregation"),
        "outlier_mixture": mixture,
        "mixture_variants": [spec.__dict__ for spec in variants],
        "gate": gate,
        "interval_audit": interval_audit,
        "primary_control": get_nested(config, "audit.primary_baseline"),
    }
)

# %% [markdown]
# ## 3. Raw data and exp072 control checks

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
        "exp209_reconstructed_control": validation_meta.get("exp209_reconstructed_likpf_control"),
        "temperature_row_candidates": temperature_comparison.get("row_candidates_filename"),
        "temperature_acceptance_required": temperature_comparison.get("required_for_acceptance"),
        "temperature_parallel_pending_allowed": temperature_comparison.get("allow_pending_during_parallel_train"),
    }
)
display(target_wells.head(20))

if target_wells.empty:
    raise RuntimeError("No eligible exp072 pseudo-tail wells were selected.")
if get_nested(config, "model.validation_surface.max_target_wells") is not None:
    raise RuntimeError("exp233 must evaluate all eligible wells; max_target_wells must be null.")
if "likpf_mean" not in validation_meta["reference_candidates_present"]:
    raise RuntimeError("The exp209-reconstructed exp072 likpf_mean control is required.")

# %% [markdown]
# ## 4. State-neutral outlier-mixture and gate contract

# %%
if not bool(gate.get("enabled", False)):
    raise RuntimeError("The exp232 target-free adaptive gate must remain enabled.")
if mixture.get("component") != "raw_gr_uniform_state_neutral":
    raise RuntimeError("Only the state-neutral raw-GR Uniform component is allowed.")
if [(spec.name, spec.epsilon) for spec in variants] != [
    ("mix_eps_0p02", 0.02),
    ("mix_eps_0p05", 0.05),
]:
    raise RuntimeError("exp233 is scoped to the predeclared epsilon=0.02/0.05 variants only.")
if (mixture.get("uniform_gr_min"), mixture.get("uniform_gr_max")) != (0.0, 500.0):
    raise RuntimeError("The predeclared state-neutral Uniform-GR support must remain [0, 500].")
if runtime.get("particles") != 500 or runtime.get("seed_count") != 128:
    raise RuntimeError("Particles and seed_count are fixed to the exp072-compatible 500 x 128 setting.")
if runtime.get("seed_aggregation") != "mean":
    raise RuntimeError("Seed aggregation must remain the exp072 likpf_mean policy.")

display(
    {
        "control": "exp209-reconstructed exp072_likpf_mean (Gaussian, no regeneration)",
        "gated_particle_likelihood": "(1-epsilon)*exp(-0.5*residual^2) + epsilon*L_uniform",
        "state_neutral_component": "L_uniform=sqrt(2*pi)*gr_sigma/(500-0), constant over particle state",
        "outside_gate": "exp(-0.5*residual^2) exact baseline likelihood",
        "gate_inputs": [
            "pre-update normalized innovation",
            "raw GR change point",
            "short/long GR novelty",
            "pre-update ESS ratio",
            "pre-update max particle weight",
        ],
        "parallel_execution_note": "exp232 artifact comparison may be pending in this initial run, but is mandatory before acceptance.",
        "forbidden": [
            "true TVT / target / error / oracle gate",
            "temperature",
            "global mixture",
            "broad Gaussian outlier component",
            "transition/resampling/particle/seed changes",
            "inference and submission",
        ],
    }
)

# %% [markdown]
# ## 5. Full train-side PF generation

# %%
result = run_adaptive_outlier_mixture_likelihood_pf(config=config, paths=paths)
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
display(candidate_metrics.head(30))

print("Distance buckets")
display(bucket_metrics.sort_values(["candidate", "distance_bucket"]).head(120))

print("Hidden-like subgroups")
if hidden_like_metrics.empty:
    print("Hidden-like split artifact was not available; see summary.hidden_like_metrics_available.")
else:
    display(hidden_like_metrics.sort_values(["subgroup", "candidate"]).head(80))

print("PF diagnostics")
display(
    pf_diagnostics.groupby(["variant", "epsilon"], observed=True)
    .agg(
        wells=("well", "nunique"),
        rows=("rows", "sum"),
        gr_sigma=("gr_sigma", "mean"),
        uniform_likelihood=("uniform_likelihood", "mean"),
        ess_mean=("ess_mean", "mean"),
        resampling_rate=("resampling_rate", "mean"),
        log_likelihood_mean=("log_likelihood_mean", "mean"),
    )
    .reset_index()
)

print("Gate and mixture diagnostics")
display(
    gate_diagnostics.groupby(["variant", "epsilon"], observed=True)
    .agg(
        wells=("well", "nunique"),
        rows=("rows", "sum"),
        gate_seed_fraction=("gate_seed_fraction_mean", "mean"),
        mixture_seed_fraction=("mixture_seed_fraction_mean", "mean"),
        gate_any_seed_rows=("gate_any_seed_rows", "sum"),
        mixture_any_seed_rows=("mixture_any_seed_rows", "sum"),
    )
    .reset_index()
)

print("Sampled particle p05-p95 coverage")
display(interval_metrics.sort_values(["candidate", "slice"]))

print("First sampled particle loss by well")
display(first_loss_by_well[first_loss_by_well["ever_lost"]].head(80))

print("Worst well regressions versus exp072 likpf_mean")
display(
    by_well[by_well["candidate"].str.startswith("pf_mix_")]
    .sort_values("delta_rmse_vs_primary_baseline", ascending=False)
    .head(80)
)

print("Temperature comparison status")
display(summary["temperature_comparison"])

print("Well execution status")
display(well_status["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="wells"))

print("Generated artifacts")
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")

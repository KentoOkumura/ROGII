# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp242 two-regime rate-noise PF train-side audit
#
# This notebook keeps the exp072-compatible Gaussian likelihood and PF runtime
# fixed.  It adds one sticky smooth/turn latent regime and broadens only the
# turn-state rate process noise.  It never generates raw-test predictions.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and fixed experiment contract
# 3. Input cache and raw-data checks
# 4. Two-regime PF contract
# 5. Full train-side PF generation
# 6. Metrics, regime diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

from IPython.display import display
from settings import ExperimentPaths, get_nested, load_config
from two_regime_rate_noise_pf import (
    read_exp072_eval_cache,
    run_two_regime_rate_noise_pf,
    select_target_wells,
    two_regime_variants,
)

# %% [markdown]
# The Numba PF kernel and cache/path helpers remain in the experiment's heavy
# PF helper.  This notebook owns configuration selection, input checks,
# orchestration, metric display, and artifact reporting.

# %% [markdown]
# ## 2. Configuration and fixed experiment contract

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

runtime = get_nested(config, "model.runtime") or {}
regime = get_nested(config, "model.regime") or {}
execution = get_nested(config, "execution") or {}
kaggle_runtime = get_nested(config, "runtime.kaggle") or {}
variants = two_regime_variants(config)

display(
    {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "status": get_nested(config, "experiment.status"),
        "parent": get_nested(config, "lineage.parent"),
        "implementation_source": get_nested(config, "lineage.implementation_source"),
        "variants": variants,
        "runtime": runtime,
        "regime": regime,
        "execution": execution,
        "gpu_enabled": kaggle_runtime.get("enable_gpu"),
        "internet_enabled": kaggle_runtime.get("enable_internet"),
        "kernel_sources": kaggle_runtime.get("kernel_sources"),
        "primary_control": get_nested(config, "audit.primary_baseline"),
    }
)

if variants != ["two_regime_k4"]:
    raise RuntimeError("Exactly one two_regime_k4 variant must be active.")
if runtime.get("particles") != 500 or runtime.get("seed_count") != 128:
    raise RuntimeError("The fixed runtime is 500 particles x 128 seeds.")
if runtime.get("seed_aggregation") != "mean":
    raise RuntimeError("Seed aggregation must remain mean.")
if runtime.get("velocity_noise") != 0.002:
    raise RuntimeError("Smooth velocity noise must remain exp072's 0.002.")
if runtime.get("position_noise") != 0.005:
    raise RuntimeError("Position noise must remain exp072's 0.005.")
if execution.get("active_variant_count") != 1:
    raise RuntimeError("exp242 must run one new PF variant.")
count_keys = ["lightgbm_config_count", "fold_count", "total_boosters"]
if any(execution.get(key) != 0 for key in count_keys):
    raise RuntimeError("LightGBM config, fold, and booster counts must all be zero.")

# %% [markdown]
# ## 3. Input cache and raw-data checks

# %%
train_dir = paths.train_data_dir
horizontal_files = sorted(train_dir.glob("*__horizontal_well.csv"))
typewell_files = sorted(train_dir.glob("*__typewell.csv"))
if not horizontal_files or not typewell_files:
    raise FileNotFoundError(f"Missing raw horizontal/typewell train files under {train_dir}")

validation_frame, validation_meta = read_exp072_eval_cache(config)
target_wells = select_target_wells(validation_frame, train_dir, config)

display(
    {
        "horizontal_files": len(horizontal_files),
        "typewell_files": len(typewell_files),
        "validation_rows": len(validation_frame),
        "validation_wells": validation_frame["well"].nunique(),
        "eligible_target_wells": len(target_wells),
        "exp072_cache": validation_meta.get("source"),
        "exp072_cache_decompressed_sha": validation_meta.get("source_decompressed_sha256"),
        "exp209_control": validation_meta.get("exp209_reconstructed_likpf_control"),
        "reference_candidates": validation_meta.get("reference_candidates_present"),
    }
)
display(target_wells.head(20))

if target_wells.empty:
    raise RuntimeError("No eligible exp072 pseudo-tail wells were selected.")
if "likpf_mean" not in validation_meta.get("reference_candidates_present", []):
    raise RuntimeError("The exp209-reconstructed exp072 likpf_mean control is required.")
if get_nested(config, "model.validation_surface.max_target_wells") is not None:
    raise RuntimeError("The canonical exp242 audit must cover all eligible wells.")

# %% [markdown]
# ## 4. Two-regime PF contract

# %%
expected_matrix = [[0.9998, 0.0002], [0.02, 0.98]]
if regime.get("states") != ["smooth", "turn"]:
    raise RuntimeError("Only smooth/turn regimes are allowed.")
if regime.get("transition_matrix") != expected_matrix:
    raise RuntimeError("The predeclared sticky transition matrix changed.")
if regime.get("initial_counts") != {"smooth": 495, "turn": 5}:
    raise RuntimeError("Initial regime counts must remain 495 smooth / 5 turn.")
if regime.get("turn_rate_noise_multiplier") != 4.0:
    raise RuntimeError("turn_rate_noise_multiplier must remain 4.0.")
if get_nested(config, "model.gate") is not None:
    raise RuntimeError("Target-free/adaptive gate changes are forbidden in exp242.")
if get_nested(config, "model.temperature_variants") is not None:
    raise RuntimeError("Adaptive observation temperature is forbidden in exp242.")

display(
    {
        "particle_state": "(position, rate, regime)",
        "smooth_rate_noise": 0.002,
        "turn_rate_noise": 0.008,
        "observation_likelihood": "exp(-0.5 * normalized_gr_residual**2)",
        "regime_switch_source": "fixed Markov transition and local RNG only",
        "resampling": "systematic; copies position, rate, and regime together",
        "forbidden": [
            "continuous acceleration",
            "position-noise changes",
            "target/error/oracle regime gate",
            "global high-noise rows",
            "adaptive likelihood",
            "particle-count increase",
            "raw-test inference or submission",
        ],
    }
)

# %% [markdown]
# ## 5. Full train-side PF generation

# %%
result = run_two_regime_rate_noise_pf(
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
regime_diagnostics = result["regime_diagnostics"]
well_status = result["well_status"]

display(summary)

# %% [markdown]
# ## 6. Metrics, regime diagnostics, and generated artifacts

# %%
print("Overall candidates")
display(candidate_metrics.head(20))

print("Distance buckets")
display(bucket_metrics.sort_values(["candidate", "distance_bucket"]).head(80))

print("Hidden-like groups")
if hidden_like_metrics.empty:
    print("Hidden-like fold assignments were not available.")
else:
    display(hidden_like_metrics.sort_values(["subgroup", "candidate"]).head(40))

print("PF diagnostics")
display(pf_diagnostics)

print("Regime diagnostics")
display(regime_diagnostics)

print("Worst well regressions")
display(
    by_well[by_well["candidate"].eq("pf_two_regime_k4_mean")]
    .sort_values("delta_rmse_vs_primary_baseline", ascending=False)
    .head(80)
)

print("Well status")
display(well_status["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="wells"))

print("Generated artifacts")
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")

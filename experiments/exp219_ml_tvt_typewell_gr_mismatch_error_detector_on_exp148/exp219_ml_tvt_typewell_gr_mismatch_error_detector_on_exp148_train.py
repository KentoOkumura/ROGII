# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148 train
#
# Diagnostic-only readout for using the exp148 ML-predicted TVT as a provisional
# typewell-GR alignment position, then measuring whether local GR mismatch
# separates high-error exp148 OOF rows.

# %% [markdown]
# ## Contents
#
# 1. Setup and configuration
# 2. Input contract
# 3. Execute ML-TVT GR mismatch readout
# 4. Metrics and generated artifacts

# %% [markdown]
# ## 1. Setup and configuration

# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from IPython.display import display

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
from ml_tvt_typewell_gr_mismatch_error_detector_on_exp148 import (
    OUTPUT_PREFIX,
    load_exp148_predictions,
    resolve_first_existing,
    run_readout,
)


def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value


paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", cfg_get(config, "experiment.route"))
print("Status:", cfg_get(config, "experiment.status"))
print("Parent:", cfg_get(config, "lineage.parent"))
print("Implementation:", cfg_get(config, "lineage.implementation_source"))
print("Kernel sources:", cfg_get(config, "runtime.kaggle.kernel_sources"))
print("Readout config:", json.dumps(cfg_get(config, "readout", {}), indent=2))
print("Feature config:", json.dumps(cfg_get(config, "model.ml_tvt_typewell_gr_mismatch_features", {}), indent=2))
print("Planned model training:", False)
print("Planned LightGBM boosters:", 0)

# %% [markdown]
# ## 2. Input contract

# %%
prediction_path = resolve_first_existing(cfg_get(config, "data.exp148_prediction_candidates", []))
print("exp148 OOF prediction source:", prediction_path)
print("Raw train dir:", paths.train_data_dir)

prediction_preview = load_exp148_predictions(prediction_path, config).head(5)
display(prediction_preview[["id", "well", "row_idx", "last_known_tvt", "target_tvt", "pred_tvt", "abs_error"]])

candidate_sources = cfg_get(config, "data.learned_likelihood_feature_candidates", [])
print("Optional candidate-disagreement sources:")
for item in candidate_sources:
    print("-", item)

hidden_sources = cfg_get(config, "data.hidden_like_fold_assignment_candidates", [])
print("Optional exp115 hidden-like fold assignment sources:")
for item in hidden_sources:
    print("-", item)

# %% [markdown]
# ## 3. Execute ML-TVT GR mismatch readout

# %%
summary = run_readout(
    output_dir=paths.artifacts_dir,
    train_dir=paths.train_data_dir,
    config=config,
)
print(json.dumps({
    "status": summary["status"],
    "rows": summary["rows"],
    "wells": summary["wells"],
    "feature_columns": summary["feature_columns"],
    "base_exp148_rmse_tvt": summary["base_exp148_rmse_tvt"],
    "primary_signal_auc_abs_error_gt10": summary["primary_signal_auc_abs_error_gt10"],
    "primary_high_signal_q90": summary["primary_high_signal_q90"],
    "correction_best_by_rmse": summary["correction_best_by_rmse"],
}, indent=2))

# %% [markdown]
# ## 4. Metrics and generated artifacts

# %%
metrics_path = Path("metrics.json")
summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"
signal_auc_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_signal_auc.csv"
signal_lift_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_signal_lift.csv"
bucket_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_distance_bucket_metrics.csv"
by_well_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv"
correction_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_correction_diagnostics.csv"
schema_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"

metrics = json.loads(metrics_path.read_text())
signal_auc = pd.read_csv(signal_auc_path)
signal_lift = pd.read_csv(signal_lift_path)
bucket_metrics = pd.read_csv(bucket_path)
by_well = pd.read_csv(by_well_path)
correction = pd.read_csv(correction_path)
schema = pd.read_csv(schema_path)

print("Metrics path:", metrics_path)
print("Summary path:", summary_path)
print("Feature cache:", paths.artifacts_dir / f"{OUTPUT_PREFIX}_features.csv.gz")
print("Next gate:", json.dumps(metrics["next_gate"], indent=2))
print("Feature schema rows:", len(schema))

display(signal_auc.sort_values("best_oriented_auc", ascending=False))
display(signal_lift.sort_values(["signal", "quantile"]).head(80))
display(bucket_metrics)
display(by_well.head(40))
display(correction.sort_values("rmse_tvt"))

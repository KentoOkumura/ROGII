# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp240 shrinkage residual-scale emission HMM on exp218 — train
#
# exp218 の保存済み `lgb_mean` OOF を Gaussian emission center に固定する。
# 最初の実行は必ず scalar `sigma=20` 対照を 1 本だけ生成する。対照結果を
# 記録した後の version だけが、事前固定した variance-shrinkage alpha を 1 つ実行できる。

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and stage contract
# 3. Input contracts
# 4. Shrinkage and exact-HMM execution helpers
# 5. Execute the selected train-side stage
# 6. Metrics and generated artifacts

# %%
from __future__ import annotations

import json
import os
from pathlib import Path

# Numba reads this value at import/JIT initialization. It must be fixed before
# importing the exact-HMM helper; changing it after the thread pool launches is invalid.
os.environ["NUMBA_NUM_THREADS"] = "1"

import pandas as pd
from IPython.display import display

from settings import ExperimentPaths, get_nested, load_config
from shrinkage_residual_scale_hmm_audit import run_shrinkage_residual_scale_hmm_audit

EXPERIMENT_NAME = "exp240_shrinkage_residual_scale_emission_hmm_on_exp218"


def resolve_existing(candidates: list[str]) -> Path:
    for value in candidates:
        path = Path(value)
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"No non-empty input candidate exists: {candidates}")


def selected_stage_contract(config: dict) -> tuple[str, dict]:
    shrinkage = dict(config.get("shrinkage") or {})
    selected = str(shrinkage.get("selected_stage") or "")
    stages = dict(shrinkage.get("stages") or {})
    if selected not in stages:
        raise ValueError(f"selected stage is not configured: {selected!r}")
    enabled = [name for name, stage in stages.items() if bool((stage or {}).get("enabled", False))]
    if enabled != [selected]:
        raise ValueError(f"Exactly selected_stage must be enabled: selected={selected} enabled={enabled}")
    return selected, dict(stages[selected])


# %% [markdown]
# ## 2. Configuration and stage contract

# %%
paths = ExperimentPaths()
paths.ensure_output_dirs()
config = load_config()
stage_name, stage = selected_stage_contract(config)

print(
    json.dumps(
        {
            "experiment": config["experiment"]["name"],
            "route": config["experiment"]["route"],
            "parent": config["lineage"]["parent"],
            "selected_stage": stage_name,
            "stage": stage,
            "formula": config["shrinkage"]["formula"],
            "allowed_alphas": config["shrinkage"]["allowed_alphas"],
            "lambda": config["lgb_emission"]["lambda_grid"],
            "gpu": get_nested(config, "runtime.kaggle.enable_gpu"),
            "internet": get_nested(config, "runtime.kaggle.enable_internet"),
            "hmm_variants_this_run": 1,
            "lightgbm_configs": 0,
            "lightgbm_boosters": 0,
            "parent_control_retraining": False,
        },
        indent=2,
        sort_keys=True,
    )
)

if stage_name != "scalar_control" and config["experiment"]["status"] == "implemented_scalar_control_not_run":
    raise ValueError("Scalar control must complete and be recorded before a shrinkage stage is enabled")
if bool(get_nested(config, "runtime.kaggle.enable_gpu")):
    raise ValueError("exp240 is CPU-only")
if bool(config["experiment"].get("inference_enabled", False)):
    raise ValueError("exp240 inference must remain disabled during train-side ablation")

# %% [markdown]
# ## 3. Input contracts
#
# Scalar control reads exp218 OOF and raw train/typewell inputs. Deferred shrinkage stages
# additionally read the exp072 row-context cache and cross-fit residual scale by well.

# %%
exp218_source = config["lgb_emission"]["sources"]["exp218_scalar"]
exp218_path = resolve_existing(list(exp218_source["candidates"]))
exp218_preview = pd.read_csv(exp218_path, nrows=5)
print({"exp218_oof": str(exp218_path), "preview_columns": exp218_preview.columns.tolist()})
display(exp218_preview)

train_dir = paths.train_data_dir
if not train_dir.is_dir():
    raise FileNotFoundError(f"raw train directory is missing: {train_dir}")

if stage["kind"] == "variance_shrinkage":
    context_path = resolve_existing(list(config["residual_scale"]["context_candidates"]))
    context_preview = pd.read_csv(context_path, nrows=5)
    print({"row_context": str(context_path), "preview_columns": context_preview.columns.tolist()})
    display(context_preview)
else:
    print("scalar_control: residual-scale fitting is intentionally skipped")

# %% [markdown]
# ## 4. Shrinkage and exact-HMM execution helpers
#
# Heavy exact-HMM/Numba recursion and well-GroupKFold residual-scale fitting remain in the
# experiment helpers. The selected-stage validation, inputs, formula, execution count, and
# generated outputs are exposed in this notebook.

# %%
max_wells_env = int(os.environ.get("N_WELLS", "0") or "0")
max_wells = max_wells_env or None
fast = bool(int(os.environ.get("FAST", "0") or "0"))
print({"max_wells": max_wells, "fast": fast, "artifacts_dir": str(paths.artifacts_dir)})

# %% [markdown]
# ## 5. Execute the selected train-side stage

# %%
summary = run_shrinkage_residual_scale_hmm_audit(max_wells=max_wells, fast=fast)

# %% [markdown]
# ## 6. Metrics and generated artifacts

# %%
print(json.dumps(summary, indent=2, sort_keys=True, default=str))
generated = sorted(path.name for path in paths.artifacts_dir.glob(f"{EXPERIMENT_NAME}*"))
print(json.dumps({"generated_artifacts": generated}, indent=2, sort_keys=True))

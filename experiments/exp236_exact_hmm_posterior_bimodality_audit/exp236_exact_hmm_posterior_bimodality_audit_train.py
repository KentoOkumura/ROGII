# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---
# %% [markdown]
# # exp236 exact HMM posterior bimodality audit train

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Setup and cost guard
# 4. Input and HMM-contract checks
# 5. Fixed posterior audit orchestration
# 6. Metrics, diagnostics, and generated artifacts

# %%
from __future__ import annotations

import json
from typing import Any

from exact_hmm_smoother import list_well_ids, resolve_existing_file, to_jsonable
from posterior_bimodality_audit import run_posterior_bimodality_audit
from settings import ExperimentPaths, get_nested, load_config


# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
def print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n## {title}")
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


def assert_cost_guard(config: dict[str, Any]) -> dict[str, Any]:
    training = get_nested(config, "model.training.modes.cpu_single_fixed_hmm_audit") or {}
    active = [
        item
        for item in (get_nested(config, "model.feature_ablation.active_variants") or [])
        if bool(item.get("enabled", False))
    ]
    if len(active) != 1:
        raise ValueError(f"expected exactly one active diagnostic variant, got {len(active)}")
    guard = {
        "active_variants": len(active),
        "lightgbm_configs": int(training.get("lightgbm_configs", -1)),
        "folds": int(training.get("folds", -1)),
        "boosters": int(training.get("boosters", -1)),
        "parent_control_retraining": bool(training.get("parent_control_retraining", True)),
        "use_gpu": bool(training.get("use_gpu", True)),
    }
    expected = {
        "active_variants": 1,
        "lightgbm_configs": 0,
        "folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
        "use_gpu": False,
    }
    if guard != expected:
        raise ValueError(f"cost guard changed: {guard}")
    return guard


def assert_fixed_exp221_contract(config: dict[str, Any]) -> dict[str, Any]:
    hmm = get_nested(config, "model.hmm") or {}
    lgb = get_nested(config, "model.lgb_emission") or {}
    expected_hmm = {
        "step": 0.35,
        "n_rates": 41,
        "sig_r": 0.002,
        "sig_p": 0.02,
        "mom": 0.998,
        "emission": "gauss",
        "rate_center": "zero",
    }
    mismatches = {key: (hmm.get(key), expected) for key, expected in expected_hmm.items() if hmm.get(key) != expected}
    if mismatches:
        raise ValueError(f"exp221 fixed HMM contract mismatch: {mismatches}")
    if float(lgb.get("sigma", -1.0)) != 20.0 or float(lgb.get("lambda", -1.0)) != 0.50:
        raise ValueError(f"exp221 selected emission is required, got {lgb}")
    return {"hmm": hmm, "lgb_emission": lgb}


# %% [markdown]
# ## 3. Setup and cost guard

# %%
paths = ExperimentPaths()
paths.ensure_output_dirs()
config = load_config()
experiment = get_nested(config, "experiment") or {}

print_json(
    "experiment",
    {
        "name": experiment.get("name"),
        "route": experiment.get("route"),
        "status": experiment.get("status"),
        "parent": get_nested(config, "lineage.parent"),
        "inference_enabled": experiment.get("inference_enabled"),
    },
)
print_json("cost guard", assert_cost_guard(config))
print_json("fixed exp221 HMM contract", assert_fixed_exp221_contract(config))


# %% [markdown]
# ## 4. Input and HMM-contract checks

# %%
oof_candidates = list(get_nested(config, "data.exp148_lgb_mean_oof_candidates") or [])
oof_path = resolve_existing_file(paths.root, oof_candidates)
wells = list_well_ids(paths.train_data_dir)
if not wells:
    raise FileNotFoundError(f"no train wells found under {paths.train_data_dir}")
print_json(
    "input contract",
    {
        "train_dir": str(paths.train_data_dir),
        "well_count": len(wells),
        "first_wells": wells[:5],
        "exp148_lgb_mean_oof": str(oof_path),
        "posterior_persisted": bool(get_nested(config, "execution.full_posterior_tensor_persisted")),
        "raw_test_inference": bool(get_nested(config, "execution.raw_test_inference")),
        "submission": bool(get_nested(config, "execution.submission")),
    },
)
if bool(get_nested(config, "execution.full_posterior_tensor_persisted")):
    raise ValueError("full posterior persistence is forbidden for this audit")
if bool(get_nested(config, "execution.raw_test_inference")) or bool(get_nested(config, "execution.submission")):
    raise ValueError("raw-test inference and submission are forbidden for this audit")


# %% [markdown]
# ## 5. Fixed posterior audit orchestration

# %%
paths.require_kaggle_runtime()
summary = run_posterior_bimodality_audit()
print_json("posterior audit summary", summary)


# %% [markdown]
# ## 6. Metrics, diagnostics, and generated artifacts

# %%
artifacts = summary.get("artifacts") or {}
print("Generated artifacts:")
for name, value in sorted(artifacts.items()):
    print(f"- {name}: {value}")
print_json(
    "decoder metrics",
    {
        "overall": summary.get("overall_decoder_metrics"),
        "bimodality": summary.get("bimodality"),
        "oracle_top2_diagnostic": summary.get("oracle_top2_diagnostic"),
    },
)

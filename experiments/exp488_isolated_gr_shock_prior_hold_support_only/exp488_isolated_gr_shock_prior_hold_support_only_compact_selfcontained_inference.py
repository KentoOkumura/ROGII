# %% [markdown]
# # exp488 isolated GR shock prior hold — inference guard candidate
#
# Stage A0/A1 is a train-side mechanism audit. Inference and submission remain
# disabled until a separately approved full-OOF Stage 1 passes every gate.

# %% [markdown]
# ## Contents
# 1. Imports and notebook-safe config loading
# 2. Disabled inference contract
# 3. Guarded orchestration

# %% [markdown]
# ## 1. Imports and notebook-safe config loading

# %%
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp488_isolated_gr_shock_prior_hold_support_only"
LINEAGE_PARENT = "exp482_isolated_gr_shock_prior_hold"
PACKAGE_DIR = Path.cwd()


def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return start


def load_config() -> dict[str, Any]:
    root = find_project_root()
    for path in (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ):
        if path.is_file():
            value = yaml.safe_load(path.read_text()) or {}
            if not isinstance(value, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            return value
    raise FileNotFoundError("exp488 config.yaml was not found")


# %% [markdown]
# ## 2. Disabled inference contract


# %%
def validate_inference_disabled(config: Mapping[str, Any]) -> dict[str, bool]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong exp488 config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp488 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != LINEAGE_PARENT:
        raise ValueError("exp488 lineage parent changed")
    contract = {
        "implementation_authorized": bool(
            get_nested(config, "execution.implementation_authorized", False)
        ),
        "canonical_notebook_adoption_authorized": bool(
            get_nested(
                config,
                "execution.canonical_notebook_adoption_authorized",
                False,
            )
        ),
        "kaggle_package_authorized": bool(
            get_nested(config, "execution.kaggle_package_authorized", False)
        ),
        "stage0_run_authorized": bool(get_nested(config, "execution.stage0_run_authorized", False)),
        "stage1_run_authorized": bool(get_nested(config, "execution.stage1_run_authorized", False)),
        "inference_authorized": bool(get_nested(config, "execution.inference_authorized", False)),
        "submission_authorized": bool(get_nested(config, "execution.submission_authorized", False)),
        "create_submission": bool(get_nested(config, "execution.create_submission", False)),
    }
    if not contract["implementation_authorized"]:
        raise RuntimeError("exp488 implementation authorization is missing")
    if contract["inference_authorized"]:
        raise RuntimeError(
            "exp488 inference cannot be enabled before separately approved Stage 1 promotion"
        )
    if contract["submission_authorized"] or contract["create_submission"]:
        raise RuntimeError("exp488 submission must remain disabled")
    return contract


def run_inference(config: Mapping[str, Any]) -> None:
    validate_inference_disabled(config)
    raise RuntimeError(
        "exp488 inference is disabled: the current authorization covers only the "
        "support-only Stage A0/A1 train run; full-OOF Stage 1, inference, and "
        "submission require separate approvals"
    )


# %% [markdown]
# ## 3. Guarded orchestration

# %%
if __name__ == "__main__":
    run_inference(load_config())

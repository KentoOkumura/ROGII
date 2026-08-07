# %% [markdown]
# # exp359 exp226 window likelihood on exp281 inference
#
# Inference is deliberately unavailable. exp359 currently implements only the
# train-side, truth-late Stage 0 rank audit. It does not contain an approved
# Stage 1 decoder, selected prediction, or submission contract.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe config loading
# 3. Fail-closed inference contract
# 4. Contract preview

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp359_exp226_window_likelihood_on_exp281"
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        return False
    return shell is not None


# %% [markdown]
# ## 2. Notebook-safe config loading

# %%
def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def project_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents, KAGGLE_WORKING_ROOT]
    for candidate in candidates:
        if (candidate / "experiments" / EXPERIMENT_NAME / "config.yaml").exists():
            return candidate
    return Path.cwd()


def load_config() -> dict[str, Any]:
    root = project_root()
    candidates = [
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        KAGGLE_WORKING_ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for path in candidates:
        if path.exists():
            return read_yaml(path)
    raise FileNotFoundError("exp359 config.yaml was not restored")


# %% [markdown]
# ## 3. Fail-closed inference contract

# %%
def validate_disabled_inference(config: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "experiment": get_nested(config, "experiment.name") == EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route") == "pf_beam",
        "stage_0_only": get_nested(config, "implementation.scope")
        == "stage_0_rank_audit_only",
        "stage_1_not_implemented": not bool(
            get_nested(config, "implementation.stage_1_implemented")
        ),
        "inference_disabled": not bool(get_nested(config, "inference.enabled")),
        "submission_disabled": not bool(get_nested(config, "inference.create_submission")),
        "run_inference_false": not bool(get_nested(config, "execution.run_inference")),
        "create_submission_false": not bool(
            get_nested(config, "execution.create_submission")
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise RuntimeError(f"exp359 disabled inference contract changed: {failed}")
    return {
        "experiment": EXPERIMENT_NAME,
        "status": "inference_disabled",
        "checks": checks,
        "reason": (
            "Stage 0 produces only a window-rank diagnostic. Stage 1 HMM, a "
            "selected prediction, and submission generation require separate approval."
        ),
    }


def refuse_inference(config: dict[str, Any]) -> None:
    contract = validate_disabled_inference(config)
    raise RuntimeError(contract["reason"])


# %% [markdown]
# ## 4. Contract preview

# %%
if in_notebook_runtime():
    CONFIG = load_config()
    INFERENCE_CONTRACT = validate_disabled_inference(CONFIG)
    print(json.dumps(INFERENCE_CONTRACT, indent=2, sort_keys=True))

# %%
if in_notebook_runtime():
    print(
        "No inference or submission was generated. "
        "This notebook is a fail-closed implementation candidate."
    )

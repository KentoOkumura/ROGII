# %% [markdown]
# # exp417 scale-5 seed aggregation promotion audit inference
#
# Fail-closed candidate. Stage A is a saved-OOF audit only. Raw-test batch
# inference must not be implemented until every Stage A guard passes and the
# user separately approves a same-experiment inference design.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Fail-closed inference contract
# 4. Setup and status

# %%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp417_scale5_seed_aggregation_promotion_audit"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP417_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime and configuration helpers


# %%
def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_config(package_dir: Path | None = None) -> dict[str, Any]:
    root = project_root()
    candidates = [
        package_dir,
        Path.cwd(),
        root / "experiments" / EXPERIMENT_NAME,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            path.parent
            for path in sorted(KAGGLE_INPUT_ROOT.glob("**/config.yaml"))
            if path.parent.name == EXPERIMENT_NAME
        )
    checked: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate / "config.yaml"
        checked.append(str(path))
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp417 config not found; checked={checked}")


# %% [markdown]
# ## 3. Fail-closed inference contract


# %%
def validate_inference_is_disabled(config: dict[str, Any]) -> dict[str, Any]:
    status = {
        "experiment": get_nested(config, "experiment.name"),
        "implementation_scope": get_nested(config, "implementation.scope"),
        "stage": get_nested(config, "execution.stage"),
        "stage_a_run_approved": bool(get_nested(config, "execution.stage_a_run_approved")),
        "inference_enabled": bool(get_nested(config, "inference.enabled")),
        "selected_candidate": get_nested(config, "inference.selected_candidate"),
        "inference_approved": bool(get_nested(config, "execution.inference_approved")),
        "submission_approved": bool(get_nested(config, "execution.submission_approved")),
    }
    if status["experiment"] != EXPERIMENT_NAME:
        raise ValueError("unexpected exp417 inference config")
    if status["implementation_scope"] != "train_side_saved_oof_promotion_audit_only":
        raise ValueError("unexpected exp417 implementation scope")
    if (
        status["inference_enabled"]
        or status["selected_candidate"] is not None
        or status["inference_approved"]
        or status["submission_approved"]
    ):
        raise RuntimeError(
            "exp417 inference must remain disabled until Stage A passes and "
            "a separate raw-test batch-inference design is approved"
        )
    return status


def fail_closed(config: dict[str, Any]) -> None:
    status = validate_inference_is_disabled(config)
    print(json.dumps(status, indent=2, sort_keys=True))
    raise RuntimeError(
        "exp417 inference is fail-closed: Stage A has not produced an approved "
        "promotion decision, and raw-test batch inference is out of scope"
    )


# %% [markdown]
# ## 4. Setup and status


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_config()
    fail_closed(CONFIG)

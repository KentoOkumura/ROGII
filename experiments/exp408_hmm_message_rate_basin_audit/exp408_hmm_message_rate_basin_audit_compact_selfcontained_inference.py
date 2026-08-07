# %% [markdown]
# # exp408 HMM message / rate basin audit — inference
#
# exp408 is a train-side internal-message diagnostic.  It deliberately creates
# no deployable prediction and cannot run inference or make a submission.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Config lookup
# 3. Fail-closed inference contract

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

EXPERIMENT_NAME = "exp408_hmm_message_rate_basin_audit"
PACKAGE_DIR = Path.cwd()


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def find_config() -> Path:
    candidates = (
        Path.cwd() / "config.yaml",
        Path.cwd() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp408 config.yaml was not found")


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(find_config().read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("exp408 config must be a mapping")
    return value


def validate_inference_disabled(config: Mapping[str, Any]) -> None:
    blockers = {
        "experiment.inference_enabled": get_nested(
            config, "experiment.inference_enabled"
        ),
        "execution.inference": get_nested(config, "execution.inference"),
        "execution.submission": get_nested(config, "execution.submission"),
        "inference.enabled": get_nested(config, "inference.enabled"),
        "inference.create_submission": get_nested(
            config, "inference.create_submission"
        ),
    }
    if any(bool(value) for value in blockers.values()):
        raise RuntimeError(f"exp408 inference contract was enabled: {blockers}")


# %%
if __name__ == "__main__":
    CONFIG = load_config()
    validate_inference_disabled(CONFIG)
    raise RuntimeError(
        "exp408 is an internal train-side HMM message audit; "
        "inference and submission are intentionally unavailable"
    )


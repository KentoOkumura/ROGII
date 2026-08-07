from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp259_coordinate_equivariance_path_warp_augmentation"


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def find_package_dir(start: str | Path | None = None) -> Path:
    candidates = [Path(start or Path.cwd()), Path.cwd()]
    candidates.extend(Path.cwd().parents)
    for candidate in candidates:
        if (candidate / "config.yaml").exists() and (candidate / "config.yaml").read_text().find(
            EXPERIMENT_NAME
        ) >= 0:
            return candidate
        nested = candidate / "experiments" / EXPERIMENT_NAME
        if (nested / "config.yaml").exists():
            return nested
    raise FileNotFoundError(f"could not locate package directory for {EXPERIMENT_NAME}")


def load_config(package_dir: str | Path | None = None) -> dict[str, Any]:
    path = find_package_dir(package_dir) / "config.yaml"
    with path.open() as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return value


def _resolve_data_dir(configured: str, package_dir: Path) -> Path:
    candidates = [
        Path(configured),
        package_dir / configured,
        package_dir.parents[1] / configured
        if len(package_dir.parents) >= 2
        else package_dir / configured,
        Path("/kaggle/input/rogii-wellbore-geology-prediction") / Path(configured).name,
        Path("/kaggle/input/rogii-wellbore-geology-prediction/data/raw") / Path(configured).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


@dataclass(frozen=True)
class ExperimentPaths:
    package_dir: Path
    train_dir: Path
    test_dir: Path
    output_dir: Path

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        package_dir: str | Path | None = None,
    ) -> ExperimentPaths:
        package = find_package_dir(package_dir)
        output = package / str(get_nested(config, "outputs.directory", "artifacts"))
        return cls(
            package_dir=package,
            train_dir=_resolve_data_dir(
                str(get_nested(config, "data.train_dir", "data/raw/train")), package
            ),
            test_dir=_resolve_data_dir(
                str(get_nested(config, "data.test_dir", "data/raw/test")), package
            ),
            output_dir=output,
        )


__all__ = ["EXPERIMENT_NAME", "ExperimentPaths", "find_package_dir", "get_nested", "load_config"]

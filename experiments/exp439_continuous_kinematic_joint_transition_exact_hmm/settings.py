"""Notebook-safe metadata for the implemented exp439 Stage 0 candidate."""

from __future__ import annotations

from pathlib import Path

EXPERIMENT_NAME = "exp439_continuous_kinematic_joint_transition_exact_hmm"
IMPLEMENTATION_STATUS = "stage0_implemented_unrun"
PACKAGE_DIR = Path.cwd()


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    """Find the repository root without relying on ``__file__``."""
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return start


def experiment_dir() -> Path:
    """Return the canonical local experiment directory."""
    return find_project_root() / "experiments" / EXPERIMENT_NAME


def require_stage0_run_approval() -> None:
    """Fail closed because implementation does not authorize Kaggle execution."""
    raise RuntimeError(
        f"{EXPERIMENT_NAME} Stage 0 is implemented but Kaggle execution "
        "requires separate user approval."
    )

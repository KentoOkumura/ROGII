from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    scripts_dir = ROOT / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "validate_experiment_for_tests",
            scripts_dir / "validate_experiment.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts_dir))


def test_current_experiment_requires_generated_directories(tmp_path: Path) -> None:
    validator = load_validator()
    errors: list[str] = []

    validator.validate_required_directories(
        tmp_path,
        legacy_layout=False,
        errors=errors,
    )

    assert errors == ["missing required directory: artifacts"]


def test_current_result_template_matches_validator_contract() -> None:
    validator = load_validator()
    result_template = (ROOT / "templates" / "experiment" / "result.md").read_text()

    assert validator.NEW_RESULT_HEADINGS <= validator.markdown_headings(result_template)


def test_legacy_experiment_warns_for_generated_directories(
    tmp_path: Path,
    capsys,
) -> None:
    validator = load_validator()
    errors: list[str] = []

    validator.validate_required_directories(
        tmp_path,
        legacy_layout=True,
        errors=errors,
    )

    assert not errors
    assert "WARNING: legacy experiment is missing generated directories" in capsys.readouterr().out

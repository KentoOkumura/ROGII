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


def test_current_requirements_template_matches_validator_contract() -> None:
    validator = load_validator()
    requirements_template = (ROOT / "templates" / "experiment" / "requirements.md").read_text()

    assert validator.NEW_REQUIREMENTS_HEADINGS <= validator.markdown_headings(requirements_template)


def test_current_experiment_requires_requirements_record(tmp_path: Path) -> None:
    validator = load_validator()
    errors: list[str] = []

    validator.validate_requirements_record(
        tmp_path,
        "exp002_current",
        "# exp002_current\n\n## 概要\n\n## 正の記録\n\n## 実行入口\n",
        new_layout=True,
        legacy_layout=False,
        allow_todo=True,
        errors=errors,
    )

    assert errors == ["missing required file: requirements.md"]


def test_explicit_legacy_contract_link_allows_missing_requirements(
    tmp_path: Path,
    capsys,
) -> None:
    validator = load_validator()
    errors: list[str] = []

    validator.validate_requirements_record(
        tmp_path,
        "exp002_legacy",
        (
            "# exp002_legacy\n\n## 概要\n\n## 正の記録\n\n"
            "[legacy](../../docs/legacy/steering/20260801-exp002-legacy/)\n\n"
            "## 実行入口\n"
        ),
        new_layout=True,
        legacy_layout=False,
        allow_todo=True,
        errors=errors,
    )

    assert errors == []
    assert "WARNING: legacy experiment has no requirements.md" in capsys.readouterr().out


def test_current_requirements_reject_unresolved_todo(tmp_path: Path) -> None:
    validator = load_validator()
    errors: list[str] = []
    headings = "\n\n".join(sorted(validator.NEW_REQUIREMENTS_HEADINGS))
    (tmp_path / "requirements.md").write_text(
        f"# exp002_current 要件と実装方法\n\n{headings}\n\n- 仮説: TODO\n"
    )

    validator.validate_requirements_record(
        tmp_path,
        "exp002_current",
        "# exp002_current\n\n## 概要\n\n## 正の記録\n\n[要件](requirements.md)\n\n## 実行入口\n",
        new_layout=True,
        legacy_layout=False,
        allow_todo=False,
        errors=errors,
    )

    assert errors == ["requirements.md still contains TODO values"]


def test_lineage_accepts_tracked_hypothesis_and_candidate() -> None:
    validator = load_validator()
    errors: list[str] = []

    validator.validate_lineage(
        {
            "lineage": {
                "hypothesis_id": "HYP-19000101-91",
                "backlog_candidate": "candidate_a",
            }
        },
        errors,
    )

    assert errors == []


def test_lineage_rejects_invalid_candidate_name() -> None:
    validator = load_validator()
    errors: list[str] = []

    validator.validate_lineage(
        {
            "lineage": {
                "hypothesis_id": "HYP-19000101-91",
                "backlog_candidate": "backlog/candidate-a.md",
            }
        },
        errors,
    )

    assert any("invalid lineage.backlog_candidate" in error for error in errors)


def test_lineage_requires_hypothesis_and_candidate_to_be_jointly_tracked() -> None:
    validator = load_validator()
    errors: list[str] = []

    validator.validate_lineage(
        {
            "lineage": {
                "hypothesis_id": "HYP-19000101-91",
                "backlog_candidate": "N/A",
            }
        },
        errors,
    )

    assert any("must both be tracked values or both be N/A" in error for error in errors)


def test_lineage_rejects_unregistered_hypothesis() -> None:
    validator = load_validator()
    errors: list[str] = []

    validator.validate_lineage(
        {
            "lineage": {
                "hypothesis_id": "HYP-19000101-91",
                "backlog_candidate": "candidate_a",
            }
        },
        errors,
        registered_ids={"HYP-19000101-92"},
    )

    assert any("unregistered lineage.hypothesis_id" in error for error in errors)


def test_allow_todo_still_rejects_completed_invalid_lineage() -> None:
    validator = load_validator()
    errors: list[str] = []

    validator.validate_config_contract(
        {
            "lineage": {
                "hypothesis_id": "HYP-19000101-91",
                "backlog_candidate": "backlog/candidate-a.md",
            }
        },
        allow_todo=True,
        project_defaults={},
        registered_ids={"HYP-19000101-91"},
        errors=errors,
    )

    assert any("invalid lineage.backlog_candidate" in error for error in errors)


def test_allow_todo_accepts_todo_lineage_values() -> None:
    validator = load_validator()
    errors: list[str] = []

    validator.validate_config_contract(
        {
            "lineage": {
                "hypothesis_id": "TODO",
                "backlog_candidate": "TODO",
            }
        },
        allow_todo=True,
        project_defaults={},
        registered_ids=set(),
        errors=errors,
    )

    assert errors == []


def test_allow_todo_still_rejects_unregistered_completed_hypothesis() -> None:
    validator = load_validator()
    errors: list[str] = []

    validator.validate_config_contract(
        {
            "lineage": {
                "hypothesis_id": "HYP-19000101-91",
                "backlog_candidate": "candidate_a",
            }
        },
        allow_todo=True,
        project_defaults={},
        registered_ids={"HYP-19000101-92"},
        errors=errors,
    )

    assert any("unregistered lineage.hypothesis_id" in error for error in errors)


def test_reproducibility_seed_override_requires_project_default_declaration() -> None:
    validator = load_validator()
    errors: list[str] = []

    validator.validate_config_contract(
        {"reproducibility": {"seed": 7}},
        allow_todo=False,
        project_defaults={"reproducibility": {"seed": 42}},
        registered_ids=set(),
        errors=errors,
    )

    assert "config value differs from project.yml without override: reproducibility.seed" in errors


def test_declared_reproducibility_seed_override_is_allowed() -> None:
    validator = load_validator()
    errors: list[str] = []

    validator.validate_config_contract(
        {
            "overrides": {"project_defaults": ["reproducibility.seed"]},
            "reproducibility": {"seed": 7},
        },
        allow_todo=False,
        project_defaults={"reproducibility": {"seed": 42}},
        registered_ids=set(),
        errors=errors,
    )

    assert not any("reproducibility.seed" in error for error in errors)


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

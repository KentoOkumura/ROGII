from pathlib import Path

import yaml

from scripts.config_utils import get_nested, load_project_config, project_experiment_defaults

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_experiment_files_exist() -> None:
    experiment_dir = ROOT / "experiments" / "exp001_baseline"
    required = [
        "README.md",
        "config.yaml",
        "settings.py",
        "exp001_baseline_train.ipynb",
        "exp001_baseline_inference.ipynb",
        "SESSION_NOTES.md",
        "result.md",
        "metrics.json",
    ]

    for filename in required:
        assert (experiment_dir / filename).exists()


def test_template_files_exist() -> None:
    template_dir = ROOT / "templates" / "experiment"
    required = [
        "README.md",
        "config.yaml",
        "settings.py",
        "{{EXPERIMENT_NAME}}_train.ipynb",
        "{{EXPERIMENT_NAME}}_inference.ipynb",
        "SESSION_NOTES.md",
        "result.md",
        "metrics.json",
    ]

    for filename in required:
        assert (template_dir / filename).exists()


def test_repository_control_files_exist() -> None:
    required = [
        "AGENTS.md",
        ".github/workflows/template-check.yml",
        "KAGGLE_DIRECTION.md",
        "Taskfile.yml",
        "Makefile",
        "uv.lock",
        "experiment_summary.md",
        "project.yml",
        "submissions/SUBMISSIONS.md",
        "docs/agent-playbooks.md",
        "app/streamlit_app.py",
        "app/oof_analysis_app.py",
        "docs/official/evaluation.md",
        ".pre-commit-config.yaml",
        "scripts/validate_project.py",
        "scripts/validate_experiment.py",
        "scripts/validate_submission.py",
        "scripts/execute_experiment_notebook.py",
        "scripts/prepare_kaggle_notebooks.py",
        "scripts/new_steering.py",
        "scripts/new_survey_report.py",
        "scripts/update_survey_index.py",
        "scripts/kaggle_download.py",
        "scripts/record_submission.py",
        "scripts/record_experiment.py",
        "scripts/compare_experiments.py",
        "scripts/update_experiment_summary.py",
        "templates/steering/requirements.md",
        "templates/steering/design.md",
        "templates/steering/tasklist.md",
        "templates/survey/report.md",
        "docs/surveys/README.md",
        "docs/01_competition.md",
        "docs/02_metric.md",
        "docs/03_validation.md",
        "docs/04_data.md",
        "docs/05_workflow.md",
        "docs/glossary.md",
    ]

    for filename in required:
        assert (ROOT / filename).exists()


def test_no_repository_local_codex_skills() -> None:
    skills_dir = ROOT / "skills"
    if not skills_dir.exists():
        return
    assert not any(skills_dir.glob("*/SKILL.md"))


def test_project_yml_supplies_experiment_defaults() -> None:
    project = load_project_config()
    defaults = project_experiment_defaults(project)

    assert get_nested(defaults, "validation.metric") == get_nested(project, "defaults.metric")
    assert get_nested(defaults, "validation.seed") == get_nested(project, "defaults.seed")
    assert get_nested(defaults, "data.id_column") == get_nested(project, "submission.id_column")
    assert get_nested(defaults, "data.sample_submission") == get_nested(
        project,
        "submission.sample_file",
    )
    assert get_nested(defaults, "data.submission_target_column") == "tvt"


def test_experiment_template_does_not_duplicate_project_defaults() -> None:
    config_path = ROOT / "templates" / "experiment" / "config.yaml"
    template_config = yaml.safe_load(config_path.read_text())

    assert template_config["validation"] == {}
    assert template_config["data"] == {}


def test_experiment_template_declares_route() -> None:
    config_path = ROOT / "templates" / "experiment" / "config.yaml"
    template_config = yaml.safe_load(config_path.read_text())

    assert template_config["experiment"]["route"] in {"ensemble", "ml_model", "pf_beam"}

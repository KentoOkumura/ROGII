from __future__ import annotations

import ast
import json
import re
import subprocess
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BARE_PYTHON_COMMAND = re.compile(
    r"(?:^\s*(?:\$\s*)?(?:nohup\s+)?|[|;&]\s*)python3?\s",
    re.MULTILINE,
)
GLOBAL_KAGGLE_INSTALL = re.compile(
    r"(?<![!\w])(?:uv\s+)?pip\s+install(?:\s+--upgrade)?\s+"
    r"(?:kaggle|kagglehub|playwright)\b",
    re.IGNORECASE,
)


def test_repo_skill_commands_do_not_depend_on_path_python() -> None:
    offenders = [
        path
        for path in (ROOT / ".agents" / "skills").rglob("*")
        if path.suffix in {".md", ".py", ".sh"}
        if BARE_PYTHON_COMMAND.search(path.read_text())
    ]
    assert not offenders


def test_repo_skills_do_not_install_kaggle_tools_outside_the_lock() -> None:
    offenders = [
        path
        for path in (ROOT / ".agents" / "skills").rglob("*")
        if path.suffix in {".md", ".py", ".sh"}
        if GLOBAL_KAGGLE_INSTALL.search(path.read_text())
    ]
    assert not offenders


def test_repo_skill_shell_helpers_use_the_locked_kaggle_cli() -> None:
    offenders: list[str] = []
    for path in (ROOT / ".agents" / "skills").rglob("*.sh"):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if re.search(r"(?:^|[=(;|&]\s*)kaggle\s", stripped):
                offenders.append(f"{path}:{line_number}: {stripped}")
    assert not offenders, "\n".join(offenders)


def test_repo_python_helpers_do_not_search_path_for_kaggle_cli() -> None:
    offenders = [
        path
        for path in (ROOT / ".agents" / "skills").rglob("*.py")
        if re.search(r"shutil\.which\([\"']kaggle[\"']\)", path.read_text())
    ]
    assert not offenders


def test_repo_python_helpers_use_the_current_interpreter_for_kaggle_cli() -> None:
    offenders: list[str] = []
    for path in (ROOT / ".agents" / "skills").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
                continue
            first = node.elts[0]
            if isinstance(first, ast.Constant) and first.value == "kaggle":
                offenders.append(f"{path}:{node.lineno}")
    assert not offenders, "\n".join(offenders)


def test_platform_has_one_credential_checker_entrypoint() -> None:
    legacy_checker = (
        ROOT / ".agents/skills/kaggle-platform/modules/registration/scripts/check_registration.py"
    )
    assert not legacy_checker.exists()

    offenders = [
        path
        for path in (ROOT / ".agents" / "skills" / "kaggle-platform").rglob("*")
        if path.suffix in {".md", ".py", ".sh"}
        if "check_registration.py" in path.read_text()
    ]
    assert not offenders


def test_repository_does_not_ship_unused_environment_example() -> None:
    gitignore = (ROOT / ".gitignore").read_text().splitlines()
    taskfile = (ROOT / "Taskfile.yml").read_text()
    makefile = (ROOT / "Makefile").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()
    checker = (
        ROOT / ".agents/skills/kaggle-platform/shared/check_all_credentials.py"
    ).read_text()
    mcp_client = (ROOT / ".agents/skills/kaggle-platform/shared/mcp_client.py").read_text()

    assert not (ROOT / ".env.example").exists()
    assert ".env" in gitignore
    assert ".env.*" in gitignore
    assert "/.kaggle/" in gitignore
    assert "dotenv:" not in taskfile
    assert "include .env" not in makefile
    assert '"python-dotenv"' not in pyproject
    assert "load_dotenv" not in checker
    assert "load_dotenv" not in mcp_client


def test_public_notebook_report_generator_uses_repository_command_entrypoint() -> None:
    script = (ROOT / "studies" / "public_notebook_catchup.py").read_text()

    assert "task fetch-kaggle-notebooks" in script
    assert "python3 .agents/skills/kaggle-notebook-fetch" not in script


def test_discussion_archive_uses_current_interpreter() -> None:
    script = (ROOT / "scripts" / "archive_kaggle_discussions.py").read_text()
    tree = ast.parse(script)
    path_cli_lists = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.List, ast.Tuple))
        and node.elts
        and isinstance(node.elts[0], ast.Constant)
        and node.elts[0].value == "kaggle"
    ]

    assert "sys.executable" in script
    assert '"python3"' not in script
    assert not path_cli_lists


def test_deprecated_top_level_output_directories_do_not_exist() -> None:
    deprecated = [ROOT / name for name in ("assets", "artifacts", "logs") if (ROOT / name).exists()]
    assert not deprecated


def test_repository_automation_keeps_caches_outside_the_worktree() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    taskfile = (ROOT / "Taskfile.yml").read_text()
    makefile = (ROOT / "Makefile").read_text()
    agents = (ROOT / "AGENTS.md").read_text()

    assert project["tool"]["ruff"]["cache-dir"] == "/tmp/ruff-cache"
    assert "no:cacheprovider" in project["tool"]["pytest"]["ini_options"]["addopts"]
    assert 'PYTHONDONTWRITEBYTECODE: "1"' in taskfile
    assert "PYTHONDONTWRITEBYTECODE ?= 1" in makefile
    assert "pytest -p no:cacheprovider" not in taskfile
    assert "pytest -p no:cacheprovider" not in makefile
    assert "managed sandboxでTask/Makeを経由せず`uv`を直接実行するとき" in agents
    assert "repo-local script、Kaggle CLI、`uv sync`などの用途を問わず" in agents
    assert not any(
        (ROOT / name).exists() for name in ("__pycache__", ".pytest_cache", ".ruff_cache")
    )


def test_survey_scaffolder_is_described_as_draft() -> None:
    taskfile = (ROOT / "Taskfile.yml").read_text()
    script = (ROOT / "scripts" / "new_survey_report.py").read_text()

    assert "draft investigation report" in taskfile
    assert "draft investigation report" in script
    assert "completed-investigation report" not in taskfile
    assert "completed-investigation report" not in script


def test_skill_change_classes_use_canonical_glossary_labels() -> None:
    skill_paths = [
        ROOT / ".agents" / "skills" / "kaggle-strategy" / "SKILL.md",
        ROOT / ".agents" / "skills" / "kaggle-review-exp" / "SKILL.md",
        ROOT / ".agents" / "skills" / "kaggle-oof-readout" / "SKILL.md",
    ]

    for path in skill_paths:
        source = path.read_text()
        assert "docs/glossary.md" in source
        assert "`add-only feature`" not in source
    combined = "\n".join(path.read_text() for path in skill_paths)
    assert "`parameter tuning`" not in combined


def test_skill_static_check_covers_all_repository_skills() -> None:
    taskfile = (ROOT / "Taskfile.yml").read_text()
    makefile = (ROOT / "Makefile").read_text()

    assert "ruff check .agents/skills" in taskfile
    assert "ruff check .agents/skills" in makefile
    assert "ruff check .agents/skills/kaggle-platform/shared" not in taskfile
    assert "ruff check .agents/skills/kaggle-platform/shared" not in makefile
    assert "check-skills:" in taskfile
    assert "check-skills:" in makefile
    assert "check-skill-modules:" in taskfile
    assert "check-skill-modules: check-skills" in makefile


def test_experiment_tests_are_not_stored_at_repository_test_root() -> None:
    assert not list((ROOT / "tests").glob("test_exp[0-9]*.py"))


def test_shared_module_tests_do_not_embed_single_experiment_contracts() -> None:
    experiment_names = {path.name for path in (ROOT / "experiments").glob("exp*") if path.is_dir()}
    offenders: list[str] = []

    for path in sorted((ROOT / "tests").glob("test_*.py")):
        source = path.read_text()
        if "from src." not in source and "import src." not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            function_source = ast.get_source_segment(source, node) or ""
            referenced = sorted(name for name in experiment_names if name in function_source)
            if len(referenced) == 1 and "experiments" in function_source:
                offenders.append(f"{path.name}:{node.lineno}: {referenced[0]}")

    assert not offenders, "\n".join(offenders)


def test_experiment_tests_are_not_stored_directly_under_experiment_root() -> None:
    direct_test_modules = [
        path
        for path in (ROOT / "experiments").glob("exp*/test*.py")
        if "def test_" in path.read_text() or "class Test" in path.read_text()
    ]
    assert not direct_test_modules


def test_experiment_test_directories_contain_python_tests() -> None:
    test_directories = list((ROOT / "experiments").glob("exp*/tests"))
    assert test_directories
    assert all(any(directory.glob("test*.py")) for directory in test_directories)


def test_experiment_specific_scripts_are_not_stored_at_repository_root() -> None:
    offenders = [
        path for path in ROOT.glob("*.py") if re.search(r"exp\d+", path.name, re.IGNORECASE)
    ]
    assert not offenders


def test_repository_root_does_not_use_conftest() -> None:
    assert not (ROOT / "conftest.py").exists()


def test_experiment_specific_generators_are_not_stored_in_shared_scripts() -> None:
    offenders = [
        path
        for path in (ROOT / "scripts").glob("*.py")
        if re.search(r"(?:^|_)exp\d+", path.name, re.IGNORECASE)
    ]
    assert not offenders


def test_experiment_specific_modules_are_not_stored_in_shared_src() -> None:
    offenders: list[str] = []
    for path in (ROOT / "src").glob("*.py"):
        if not re.search(r"(?:^|_)exp\d+", path.name, re.IGNORECASE):
            continue
        module = path.stem
        consumers: set[str] = set()
        for experiment in (ROOT / "experiments").glob("exp*"):
            if not experiment.is_dir():
                continue
            for candidate in experiment.rglob("*"):
                if not candidate.is_file() or candidate.suffix not in {".py", ".ipynb", ".md"}:
                    continue
                if module in candidate.read_text(encoding="utf-8", errors="replace"):
                    consumers.add(experiment.name)
                    break
        if len(consumers) < 2:
            offenders.append(f"{path.name}: {sorted(consumers)}")

    assert not offenders


def test_experiment_record_identities_match_directory_names() -> None:
    offenders: list[str] = []
    for directory in sorted((ROOT / "experiments").glob("exp*")):
        if not directory.is_dir():
            continue
        experiment = directory.name
        config_path = directory / "config.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text()) or {}
            config_name = (config.get("experiment") or {}).get("name")
            if config_name != experiment:
                offenders.append(f"{config_path}: {config_name!r}")
        metrics_path = directory / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("experiment") is not None and metrics.get("experiment") != experiment:
                offenders.append(f"{metrics_path}: {metrics.get('experiment')!r}")
        for filename in ("README.md", "result.md"):
            path = directory / filename
            if not path.exists():
                continue
            heading = re.search(r"^#\s+(.+)$", path.read_text(), flags=re.MULTILINE)
            expected_id = re.match(r"exp\d+", experiment, flags=re.IGNORECASE)
            assert expected_id is not None
            observed_id = (
                re.search(r"exp\d+", heading.group(1), flags=re.IGNORECASE)
                if heading is not None
                else None
            )
            if observed_id is None and filename == "result.md":
                continue
            if observed_id is None or observed_id.group(0).lower() != expected_id.group(0).lower():
                offenders.append(f"{path}: H1 mismatch")
    assert not offenders, "\n".join(offenders)


def test_kaggle_cli_tasks_use_the_locked_environment() -> None:
    taskfile = (ROOT / "Taskfile.yml").read_text()
    makefile = (ROOT / "Makefile").read_text()
    assert not re.search(r"^\s+-\s+kaggle\s", taskfile, flags=re.MULTILINE)
    assert not re.search(r"^\t+kaggle\s", makefile, flags=re.MULTILINE)
    assert "fetch-kaggle-notebooks:" in taskfile
    assert "fetch-kaggle-notebooks:" in makefile
    assert "archive-kaggle-discussions:" in taskfile
    assert "archive-kaggle-discussions:" in makefile


def test_competition_automation_reads_slug_from_project_config() -> None:
    taskfile = (ROOT / "Taskfile.yml").read_text()
    makefile = (ROOT / "Makefile").read_text()
    hardcoded_slug = "rogii-wellbore-geology-prediction"

    assert "scripts/project_value.py competition.slug" in taskfile
    assert "scripts/project_value.py competition.slug" in makefile
    assert hardcoded_slug not in taskfile
    assert hardcoded_slug not in makefile

    monitor_skill = (ROOT / ".agents" / "skills" / "kaggle-submit-monitor" / "SKILL.md").read_text()
    monitor_script = (
        ROOT / ".agents" / "skills" / "kaggle-submit-monitor" / "scripts" / "monitor_submission.py"
    ).read_text()
    fetch_skill = (ROOT / ".agents" / "skills" / "kaggle-notebook-fetch" / "SKILL.md").read_text()
    archive_skill = (
        ROOT / ".agents" / "skills" / "kaggle-discussion-archive" / "SKILL.md"
    ).read_text()

    assert "project.yml" in monitor_skill
    assert "read_project_competition" in monitor_script
    assert "--submission-ref" in monitor_skill
    assert "select_submission" in monitor_script
    assert "rows[0]" not in monitor_script
    assert "KAGGLE_COMPETITION" not in monitor_script
    assert ".kaggle_competition" not in monitor_script
    assert "COMPETITION=COMPETITION" not in fetch_skill
    assert "COMPETITION=COMPETITION" not in archive_skill


def test_kaggle_prepare_target_requires_push_ready_metadata() -> None:
    taskfile = (ROOT / "Taskfile.yml").read_text()
    makefile = (ROOT / "Makefile").read_text()

    assert "prepare_kaggle_notebooks.py --experiment {{.EXP}} --strict" in taskfile
    assert "prepare_kaggle_notebooks.py --experiment $(EXP) --strict" in makefile
    assert taskfile.count("scripts/validate_kaggle_metadata.py --package-dir") == 2
    assert makefile.count("scripts/validate_kaggle_metadata.py --package-dir") == 2


def test_standard_kaggle_notebook_flow_uses_generated_id_and_title() -> None:
    platform_skill = (ROOT / ".agents" / "skills" / "kaggle-platform" / "SKILL.md").read_text()
    review_skill = (ROOT / ".agents" / "skills" / "kaggle-review-exp" / "SKILL.md").read_text()
    session_template = (ROOT / "templates" / "experiment" / "SESSION_NOTES.md").read_text()

    standard_command = 'EXTRA_ARGS="--notebook train --run-on-push"'
    explicit_standard_command = (
        'EXTRA_ARGS="--notebook train --kernel-id username/expXXX-title-train'
    )
    platform_standard_flow = platform_skill.split(
        "#### Push 前の runtime resource / quota 確認", 1
    )[0]
    assert standard_command in platform_skill
    assert standard_command in review_skill
    assert "--notebook train --run-on-push" in session_template
    assert explicit_standard_command not in platform_standard_flow
    assert explicit_standard_command not in review_skill
    assert "通常は`--kernel-id`と`--title`を省略" in platform_skill


def test_experiment_templates_keep_each_record_in_one_source() -> None:
    template = ROOT / "templates" / "experiment"
    readme = (template / "README.md").read_text()
    result = (template / "result.md").read_text()
    session = (template / "SESSION_NOTES.md").read_text()
    metrics = json.loads((template / "metrics.json").read_text())

    assert "親実験:" not in readme
    assert "- 親:" not in result
    assert "- 検証:" not in result
    assert "- メトリック:" not in result
    assert "- シード:" not in result
    assert "feature content SHA: TODO" not in result
    assert "feature content SHA: TODO" not in session
    assert set(metrics["evidence"]) == {
        "kaggle",
        "artifacts",
        "submission_validation",
        "cpu_gpu_comparison",
        "reruns",
    }
    assert "feature_content_sha" in metrics["evidence"]["artifacts"]
    assert "oof_prediction_sha" in metrics["evidence"]["artifacts"]
    assert "test_prediction_content_sha" in metrics["evidence"]["artifacts"]
    assert "prediction_sha" not in metrics["evidence"]["artifacts"]
    assert "notebook_runtime_seconds" in metrics["evidence"]["kaggle"]
    assert "fallback_rows" in metrics["evidence"]["submission_validation"]
    assert "duplicate_id_count" in metrics["evidence"]["submission_validation"]
    assert "infinite_value_count" in metrics["evidence"]["submission_validation"]
    assert isinstance(metrics["evidence"]["reruns"], list)


def test_reproducibility_doc_references_the_canonical_record_source_split() -> None:
    reproducibility = (ROOT / "docs" / "06_reproducibility.md").read_text()

    assert "`metrics.json` または `SESSION_NOTES.md`" not in reproducibility
    assert "`metrics.json`の`evidence`へ機械可読に残す" in reproducibility
    assert "実験記録全体の役割分担は`AGENTS.md`を正とする" in reproducibility
    assert "同じ情報を複数ファイルへ手作業で転記していない" in reproducibility
    assert '"byte_identical_to_reference": true' in reproducibility


def test_steering_contract_is_not_duplicated_in_design() -> None:
    template = ROOT / "templates" / "steering"
    requirements = (template / "requirements.md").read_text()
    design = (template / "design.md").read_text()
    tasklist = (template / "tasklist.md").read_text()

    assert "## 判断履歴" in requirements
    assert "- input: TODO" in requirements
    assert "- target / objective: TODO" in requirements
    assert "- input: TODO" not in design
    assert "- target / objective: TODO" not in design
    assert "`requirements.md`だけに記録する" in tasklist
    assert "`requirements.md` と `design.md` に記録する" not in tasklist


def test_ci_uses_repository_automation_targets() -> None:
    workflow = (ROOT / ".github" / "workflows" / "template-check.yml").read_text()

    assert "make validate-template" in workflow
    assert "make test-common" in workflow
    assert "uv run python scripts/" not in workflow
    assert "uv run pytest" not in workflow


def test_external_tools_are_ignored_by_default() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", "tools/future-tool/README.md"],
        cwd=ROOT,
        check=False,
    )
    kept = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", "tools/README.md"],
        cwd=ROOT,
        check=False,
    )

    assert ignored.returncode == 0
    assert kept.returncode == 1


def test_generated_experiment_directories_keep_only_their_sentinel() -> None:
    generated_file = ROOT / "experiments" / "exp999_template_check" / "artifacts" / "report.csv"
    sentinel = generated_file.with_name(".gitkeep")

    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", str(generated_file)],
        cwd=ROOT,
        check=False,
    )
    kept = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", str(sentinel)],
        cwd=ROOT,
        check=False,
    )

    assert ignored.returncode == 0
    assert kept.returncode == 1

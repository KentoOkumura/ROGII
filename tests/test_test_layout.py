from __future__ import annotations

import ast
import json
import os
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
    checker = (ROOT / ".agents/skills/kaggle-platform/shared/check_all_credentials.py").read_text()
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


def test_public_notebook_reports_use_current_repository_commands() -> None:
    report_root = ROOT / "docs" / "notebooks" / "rogii-wellbore-geology-prediction"
    report_paths = [
        report_root / "public_notebook_catchup_after_self_improvements_2026-06-06.md",
        report_root / "public_notebook_catchup_2026-06-11.md",
        report_root / "eda_insights_summary.md",
        report_root / "latest_public_notebooks_20260603.md",
    ]

    for path in report_paths:
        source = path.read_text()
        assert "task fetch-kaggle-notebooks" in source
        assert not BARE_PYTHON_COMMAND.search(source)
        assert "scripts/public_notebook_catchup.py" not in source


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
    task_project_value = (
        "PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache "
        "uv run python scripts/project_value.py"
    )
    make_project_value = (
        "$(shell PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/project_value.py"
    )
    assert taskfile.count(task_project_value) == 2
    assert makefile.count(make_project_value) == 2
    assert "pytest -p no:cacheprovider" not in taskfile
    assert "pytest -p no:cacheprovider" not in makefile
    assert "managed sandboxでTask/Makeを経由せず`uv`を直接実行するとき" in agents
    assert "repo-local script、Kaggle CLI、`uv sync`などの用途を問わず" in agents
    cache_names = {"__pycache__", ".pytest_cache", ".ruff_cache"}
    cache_directories: list[Path] = []
    for current, directories, _ in os.walk(ROOT):
        if Path(current) == ROOT:
            directories[:] = [
                name for name in directories if name not in {".git", ".venv", "tools"}
            ]
        cache_directories.extend(
            Path(current) / name for name in directories if name in cache_names
        )
        directories[:] = [name for name in directories if name not in cache_names]
    assert not cache_directories


def test_kllm_shell_wrappers_share_sandbox_safe_uv_environment() -> None:
    scripts_dir = ROOT / ".agents/skills/kaggle-platform/modules/kllm/scripts"
    helper = (scripts_dir / "repo_uv_env.sh").read_text()
    wrappers = (
        "cli_competition.sh",
        "cli_download.sh",
        "cli_execute.sh",
        "cli_publish.sh",
        "network_check.sh",
        "poll_kernel.sh",
    )

    assert ': "${UV_CACHE_DIR:=/tmp/uv-cache}"' in helper
    assert ': "${PYTHONDONTWRITEBYTECODE:=1}"' in helper
    assert "export UV_CACHE_DIR PYTHONDONTWRITEBYTECODE" in helper
    for filename in wrappers:
        source = (scripts_dir / filename).read_text()
        assert 'source "${SCRIPT_DIR}/repo_uv_env.sh"' in source


def test_generated_submission_csv_locations_are_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text().splitlines()

    assert "/submission.csv" in gitignore
    assert "/experiments/*/submission.csv" in gitignore


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


def test_platform_uses_kaggle_notebook_term_without_kkb_shorthand() -> None:
    platform_root = ROOT / ".agents" / "skills" / "kaggle-platform"
    offenders = [
        path
        for path in platform_root.rglob("*")
        if path.suffix in {".md", ".py", ".sh"}
        if re.search(r"\bKKB\b|Kaggle Kernel Backend", path.read_text())
    ]

    assert not offenders


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


def test_viewer_automation_reads_raw_dir_from_project_config() -> None:
    taskfile = (ROOT / "Taskfile.yml").read_text()
    makefile = (ROOT / "Makefile").read_text()
    viewer_script = (ROOT / "scripts" / "run_rogii_viewer.py").read_text()

    assert "scripts/project_value.py data.raw_dir" in taskfile
    assert "scripts/project_value.py data.raw_dir" in makefile
    assert "VIEWER_DATA: '{{.VIEWER_DATA | default \"data/raw\"}}'" not in taskfile
    assert "VIEWER_DATA ?= data/raw" not in makefile
    assert 'project_path(load_project_config(), "data.raw_dir")' in viewer_script
    assert 'REPO_ROOT / "data" / "raw"' not in viewer_script


def test_strategy_and_glossary_distinguish_numeric_sources_and_legacy_terms() -> None:
    direction = (ROOT / "backlog/KAGGLE_DIRECTION.md").read_text()
    glossary = (ROOT / "docs" / "glossary.md").read_text()

    numeric_source = (
        "CV）、Public Leaderboard（Public LB）、Private Leaderboard（Private LB）の数値は"
        "各実験の `metrics.json` を正本"
    )
    assert numeric_source in direction
    assert "`SUBMISSIONS.md` は提出履歴の横断スナップショット" in direction
    assert "`metrics.json` と `result.md`" not in direction
    for term in ("`hidden-like`", "`public-core`", "`truth-late`", "`terminal close`"):
        assert term in glossary
    assert "`metrics.json`の実験statusではない" in glossary


def test_kaggle_prepare_target_requires_push_ready_metadata() -> None:
    taskfile = (ROOT / "Taskfile.yml").read_text()
    makefile = (ROOT / "Makefile").read_text()

    assert "prepare_kaggle_notebooks.py --experiment {{.EXP}} --strict" in taskfile
    assert "prepare_kaggle_notebooks.py --experiment $(EXP) --strict" in makefile
    assert taskfile.count("scripts/validate_kaggle_metadata.py --package-dir") == 1
    assert makefile.count("scripts/validate_kaggle_metadata.py --package-dir") == 1
    assert "task: push-kaggle-notebook" in taskfile
    assert "$(MAKE) push-kaggle-notebook" in makefile


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
    assert standard_command not in review_skill
    assert "--notebook train --run-on-push" in session_template
    assert explicit_standard_command not in platform_standard_flow
    assert explicit_standard_command not in review_skill
    assert "通常は`--kernel-id`と`--title`を省略" in platform_skill
    assert "このskillへコマンドを複製しない" in review_skill
    assert "task new-exp" not in platform_skill


def test_workflow_delegates_push_resource_checks_to_platform_skill() -> None:
    workflow = (ROOT / "docs" / "05_workflow.md").read_text()
    reproducibility = (ROOT / "docs" / "06_reproducibility.md").read_text()
    platform_skill = (ROOT / ".agents/skills/kaggle-platform/SKILL.md").read_text()
    platform_reference = (
        ROOT / ".agents/skills/kaggle-platform/modules/kllm/references/kaggle-knowledge.md"
    ).read_text()
    cli_reference = (
        ROOT / ".agents/skills/kaggle-platform/modules/kllm/references/cli-reference.md"
    ).read_text()

    assert "Push 前の runtime resource / quota 確認" in platform_skill
    assert "kaggle-platform/SKILL.md" in workflow
    assert "push前のruntime resource / quota確認" in workflow
    assert "uv run kaggle quota --format json" not in workflow
    assert "ユーザー指定の同時session上限" not in workflow
    assert "enable_tpu: false" not in workflow
    assert "kaggle-platform/SKILL.md" in reproducibility
    assert "uv run kaggle quota --format json" not in reproducibility
    assert "../../../SKILL.md" in platform_reference
    assert "this repository's notebook generation does not support TPU" not in platform_reference
    assert "`enable_tpu: true`を拒否する" not in platform_reference
    assert "TPU は未対応のため選択しない" not in cli_reference
    assert "TPU accelerator は選択しない" not in cli_reference
    assert "uv run kaggle quota --format json" not in platform_reference


def test_approved_experiment_entry_supports_direct_and_backlog_paths() -> None:
    agents = (ROOT / "AGENTS.md").read_text()
    workflow = (ROOT / "docs" / "05_workflow.md").read_text()
    requirements = (ROOT / "templates" / "experiment" / "requirements.md").read_text()
    review_skill = (ROOT / ".agents/skills/kaggle-review-exp/SKILL.md").read_text()
    review_ui = yaml.safe_load(
        (ROOT / ".agents/skills/kaggle-review-exp/agents/openai.yaml").read_text()
    )

    assert "backlog経由か直接承認かを問わず `kaggle-review-exp`" in agents
    assert "ユーザーが直接実験化を承認した場合は、形式的なbacklogを作らない" in workflow
    assert "直接承認では、依頼原文" in review_skill
    assert "## 実験化の入口・引き継ぎ・承認" in requirements
    assert "`直接承認`、またはbacklog時の状態" in requirements
    assert "when applicable" in review_ui["interface"]["default_prompt"]


def test_backlog_implementation_skill_uses_minimal_read_order() -> None:
    review_skill = (ROOT / ".agents/skills/kaggle-review-exp/SKILL.md").read_text()
    strategy_skill = (ROOT / ".agents/skills/kaggle-strategy/SKILL.md").read_text()
    ordered_steps = (
        "1. 対応する`backlog/<candidate>.md`",
        "2. 候補詳細から直接参照される根拠",
        "3. 親実験の`requirements.md`",
        "4. 親実験の`config.yaml`",
        "5. 変更対象コード",
    )
    positions = [review_skill.index(step) for step in ordered_steps]

    assert positions == sorted(positions)
    assert "`backlog/KAGGLE_DIRECTION.md`全体" in review_skill
    assert "全体戦略の収集手順を先に実行しない" in strategy_skill


def test_validation_scope_uses_agents_as_canonical_source() -> None:
    agents = (ROOT / "AGENTS.md").read_text()
    readme = (ROOT / "README.md").read_text()
    review_skill = (ROOT / ".agents/skills/kaggle-review-exp/SKILL.md").read_text()

    canonical_rule = "全実験のテストを収集する`task test`"
    assert canonical_rule in agents
    assert "`task test-common`" in agents
    assert "検証範囲の判断は`AGENTS.md`の運用ルールを正" in readme
    assert "検証範囲は`AGENTS.md`の運用ルールに従う" in review_skill
    assert canonical_rule not in readme
    assert canonical_rule not in review_skill


def test_actual_kaggle_submission_uses_agents_as_canonical_approval_source() -> None:
    agents = (ROOT / "AGENTS.md").read_text()
    readme = (ROOT / "README.md").read_text()
    workflow = (ROOT / "docs" / "05_workflow.md").read_text()
    platform_skill = (ROOT / ".agents/skills/kaggle-platform/SKILL.md").read_text()
    submit_check = (ROOT / ".agents/skills/kaggle-submit-check/SKILL.md").read_text()

    canonical_rule = "ユーザーがその提出を明示的に依頼または承認"
    assert canonical_rule in agents
    assert canonical_rule not in readme
    assert canonical_rule not in workflow
    assert canonical_rule not in platform_skill
    assert canonical_rule not in submit_check
    assert "submissionに必要なユーザー承認は`AGENTS.md`" in readme
    assert "submissionの承認条件は`AGENTS.md`" in workflow
    assert "`AGENTS.md`のsubmission承認条件" in platform_skill
    assert "`AGENTS.md`の承認条件" in submit_check
    assert "提出前検証の承認を、submission の承認として扱いません" in agents


def test_kaggle_output_download_policy_is_canonical_in_platform_skill() -> None:
    readme = (ROOT / "README.md").read_text()
    workflow = (ROOT / "docs" / "05_workflow.md").read_text()
    platform_skill = (ROOT / ".agents/skills/kaggle-platform/SKILL.md").read_text()
    review_skill = (ROOT / ".agents/skills/kaggle-review-exp/SKILL.md").read_text()

    assert "実ファイル確認が必要な場合だけ" in platform_skill
    assert platform_skill.count("実ファイル確認が必要な場合だけ") == 1
    assert "Kaggle outputを取得する条件" in readme
    assert "Kaggle outputを取得する条件" in workflow
    assert "output取得の判断" in review_skill
    assert "Kaggle output archive 取得の判断:" not in review_skill
    normalized_review = re.sub(r"\s+", "", review_skill)
    assert "CV評価だけなら" not in normalized_review
    assert "CV評価だけであれば" not in normalized_review
    assert "output取得を検討" not in normalized_review


def test_colab_outputs_are_stored_as_experiment_artifacts() -> None:
    colab_skill = (ROOT / ".agents/skills/colab-notebook-runner/SKILL.md").read_text()
    generator = (
        ROOT / ".agents/skills/colab-notebook-runner/scripts/create_colab_train_notebook.py"
    ).read_text()

    assert "experiments/<exp>/artifacts/colab_runs/<run_id>/" in colab_skill
    assert "experiments/<exp>/kaggle/output/colab_run_<run_id>/" not in colab_skill
    assert 'EXP_DIR / "artifacts" / "colab_runs"' in generator
    assert 'EXP_DIR / "colab_runs"' not in generator


def test_colab_reference_generator_is_exp092_specific() -> None:
    colab_skill = (ROOT / ".agents/skills/colab-notebook-runner/SKILL.md").read_text()
    generator = (
        ROOT / ".agents/skills/colab-notebook-runner/scripts/create_colab_train_notebook.py"
    ).read_text()

    assert "exp092-specific reference generator" in colab_skill
    assert 'EXPERIMENT = "exp092_u_projection_correction_disagreement_fullrun"' in generator
    assert 'parser.add_argument("--experiment"' not in generator


def test_experiment_workflow_only_requires_notebooks_in_the_experiment_contract() -> None:
    agents = (ROOT / "AGENTS.md").read_text()
    readme = (ROOT / "README.md").read_text()
    workflow = (ROOT / "docs/05_workflow.md").read_text()
    review_skill = (ROOT / ".agents/skills/kaggle-review-exp/SKILL.md").read_text()
    session_template = (ROOT / "templates/experiment/SESSION_NOTES.md").read_text()

    assert "実験契約に必要な Kaggle Notebook の実装・実行" in agents
    assert "実験契約に必要な Kaggle Notebook の実行" in readme
    assert "実験契約に必要な train、inference、audit、diagnostic" in workflow
    assert "train・inference・audit・diagnostic などの Notebook 実装" in review_skill
    assert "実験契約に必要な種類だけを実装・実行" in review_skill
    assert "実験契約に必要なnotebookだけを予定へ残します" in session_template


def test_readme_lists_full_test_command_once() -> None:
    readme = (ROOT / "README.md").read_text()

    assert readme.count("task test\n") == 1


def test_notebook_automation_accepts_contract_specific_notebook_kinds() -> None:
    taskfile = (ROOT / "Taskfile.yml").read_text()
    task_config = yaml.safe_load(taskfile)
    makefile = (ROOT / "Makefile").read_text()
    platform_skill = (ROOT / ".agents/skills/kaggle-platform/SKILL.md").read_text()
    prepare_script = (ROOT / "scripts/prepare_kaggle_notebooks.py").read_text()

    assert "push-kaggle-notebook:" in taskfile
    assert "push-kaggle-notebook" in task_config["tasks"]
    assert "push-kaggle-notebook:" in makefile
    assert "NOTEBOOK=audit" in platform_skill
    assert "EXTRA_NOTEBOOK_KINDS" not in prepare_script


def test_kaggle_skills_do_not_bypass_push_package_validation() -> None:
    platform_skill = (ROOT / ".agents/skills/kaggle-platform/SKILL.md").read_text()
    review_skill = (ROOT / ".agents/skills/kaggle-review-exp/SKILL.md").read_text()

    assert "validatorを迂回する直接の`kaggle kernels push`を使わない" in platform_skill
    assert "uv run kaggle kernels push" not in platform_skill
    assert "uv run kaggle kernels push" not in review_skill
    assert "uv run kaggle kernels logs" not in platform_skill
    assert "uv run kaggle kernels output" not in platform_skill
    assert "uv run kaggle competitions submit" not in platform_skill
    assert "task kaggle-logs KERNEL=owner/slug" in platform_skill
    assert "task kaggle-output" in platform_skill
    assert "task submit-code" in platform_skill
    assert "`--no-src`では`src/`を除外" in platform_skill


def test_transient_submission_monitor_does_not_reserve_root_logs_directory() -> None:
    gitignore = (ROOT / ".gitignore").read_text().splitlines()

    assert "/logs/" not in gitignore


def test_idea_forge_uses_plain_negative_evidence_descriptions() -> None:
    sources = [
        (ROOT / ".agents/skills/kaggle-idea-forge/SKILL.md").read_text(),
        (ROOT / "docs/surveys/rogii-top-solutions-agent-idea-workflow_20260806.md").read_text(),
    ]

    for source in sources:
        assert "instantiation closed" not in source
        assert "role closed" not in source
        assert "mechanism closed" not in source


def test_repository_setup_reviews_all_competition_specific_fields() -> None:
    platform_skill = (ROOT / ".agents/skills/kaggle-platform/SKILL.md").read_text()
    readme = (ROOT / "README.md").read_text()
    taskfile = (ROOT / "Taskfile.yml").read_text()
    makefile = (ROOT / "Makefile").read_text()

    for field_group in (
        "`competition`: `name`、`platform`、`slug`、`url`、`is_code_competition`",
        (
            "`data`: `raw_dir`、`train_dir`、`test_dir`、`processed_dir`、"
            "`target_column`、`group_column`、`score_rows`"
        ),
        "`defaults`: `seed`、`metric`、`primary_validation`、`n_folds`",
        (
            "`submission`: `sample_file`、`output_file`、`id_column`、"
            "`target_columns`、`allow_extra_columns`"
        ),
        "`metadata`: `owner`、`notes`",
        "`runtime.kaggle`: `enable_gpu`、`enable_internet`、`time_limit_hours`",
    ):
        assert field_group in platform_skill
    assert "competition、data、defaults、submission、metadata、runtime.kaggle" in readme
    assert "リポジトリ構成を変える場合だけ paths も更新" in readme
    for document in (
        "docs/01_competition.md",
        "docs/04_data.md",
        "docs/official/evaluation.md",
        "backlog/KAGGLE_DIRECTION.md",
        "backlog/",
        "SUBMISSIONS.md",
    ):
        assert f"`{document}`" in platform_skill
    assert "--expected-competition <competition-slug>" in platform_skill
    assert (
        'task validate-config VALIDATE_ARGS="--expected-competition <competition-slug>"' in readme
    )
    assert "competition、data、validation、submission、runtime" not in readme
    assert "scripts/validate_project.py --strict {{.VALIDATE_ARGS}}" in taskfile
    assert "scripts/validate_project.py --strict $(VALIDATE_ARGS)" in makefile


def test_current_study_readmes_use_repository_python_entrypoint() -> None:
    readmes = (
        ROOT / "studies/feature_replacement_audit/README.md",
        ROOT / "studies/exp226_offset_root_cause_audit_20260727/README.md",
        ROOT / "studies/candidate_path_blend_audit/README.md",
    )

    for path in readmes:
        source = path.read_text()
        assert "uv run python studies/" in source
        assert ".venv/bin/python studies/" not in source


def test_completed_studies_delegate_interpretation_to_surveys() -> None:
    study_contracts = {
        "prefix_extrapolation_reasonableness_20260705": (
            "prefix_extrapolation_reasonableness_20260705.md",
            ("## Findings", "## Decision"),
        ),
        "hmm_pf_exp226_well_pattern_readout_20260712": (
            "hmm_pf_exp226_well_pattern_readout_20260712.md",
            ("## Key Counts", "## Main Findings"),
        ),
        "feature_replacement_audit": (
            "exp148_exp092_feature_replacement_audit_20260704.md",
            ("## 結論", "## 推奨する次の ablation"),
        ),
        "exp148_feature_correlation": (
            "exp148_exp092_feature_replacement_audit_20260704.md",
            ("## 要約", "## 推奨 next experiment"),
        ),
        "exp226_offset_root_cause_audit_20260727": (
            "exp226_offset_root_cause_audit_20260727.md",
            ("主要結論:", "根本機構は"),
        ),
    }

    for study_name, (survey_name, forbidden_sections) in study_contracts.items():
        source = (ROOT / "studies" / study_name / "README.md").read_text()
        assert f"docs/surveys/{survey_name}" in source
        assert (ROOT / "docs" / "surveys" / survey_name).exists()
        for section in forbidden_sections:
            assert section not in source

    assert not (ROOT / "docs/analysis/exp226_offset_root_cause_audit_20260727.md").exists()


def test_post_deadline_backlog_candidates_do_not_advance_to_submission() -> None:
    direction = (ROOT / "backlog/KAGGLE_DIRECTION.md").read_text()
    candidate_paths = (
        ROOT / "backlog/candidate_rmse_prior_current_test_contract.md",
        ROOT / "backlog/outer_train_selector_well_risk_discriminator.md",
    )

    assert "現在は新しい最終提出の探索ではなく" in direction
    for path in candidate_paths:
        source = path.read_text()
        current_scope = source.split("## 移行前の記録（履歴）", maxsplit=1)[0]
        assert "分析結果にかかわらず行わない" in current_scope or (
            "監査結果にかかわらず行わない" in current_scope
        )
        assert "現行の実行条件ではない" in current_scope


def test_strategy_prompt_and_direction_respect_canonical_sources() -> None:
    strategy_ui = yaml.safe_load(
        (ROOT / ".agents/skills/kaggle-strategy/agents/openai.yaml").read_text()
    )
    direction = (ROOT / "backlog/KAGGLE_DIRECTION.md").read_text()
    workflow = (ROOT / "docs" / "05_workflow.md").read_text()

    assert "only when requested" in strategy_ui["interface"]["default_prompt"]
    assert "## 参照する前提" in direction
    assert "[`project.yml`](../project.yml)を正とする" in direction
    assert "## コンペ設定" not in direction
    assert "## 検証方針" not in direction
    assert "実験前の文献・公開Notebook調査" in workflow
    assert "単一実験の完了分析" in workflow
    assert "実験横断の完了調査" in workflow


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


def test_submission_history_uses_unambiguous_status_and_elapsed_time_keys() -> None:
    submissions = (ROOT / "SUBMISSIONS.md").read_text()

    assert "`submission_status`" in submissions
    assert "`notebook_runtime_seconds`" in submissions
    assert "`scoring_elapsed_minutes`" in submissions
    assert re.search(r"(?<![A-Za-z_])(?:status|runtime)=", submissions) is None


def test_experiment_requirements_consolidates_contract_and_implementation() -> None:
    requirements = (ROOT / "templates" / "experiment" / "requirements.md").read_text()

    assert "## 判断履歴" in requirements
    assert "## 手法契約" in requirements
    assert "## 実装方法" in requirements
    assert "## 受け入れ基準" in requirements
    assert "- input: TODO" in requirements
    assert "- target / objective: TODO" in requirements
    assert "- inputの実装箇所と変換: TODO" in requirements
    assert not (ROOT / "templates" / "steering").exists()


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

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strategy_context_prioritizes_canonical_and_recent_records(tmp_path: Path) -> None:
    collector = load_module(
        "collect_strategy_context",
        ROOT / ".agents/skills/kaggle-strategy/scripts/collect_strategy_context.py",
    )
    for relative in collector.CANONICAL_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    for experiment in ("exp002_old", "exp010_recent"):
        directory = tmp_path / "experiments" / experiment
        directory.mkdir(parents=True)
        for filename in collector.EXPERIMENT_RECORDS:
            (directory / filename).write_text(experiment)
    cache = tmp_path / ".pytest_cache" / "README.md"
    cache.parent.mkdir()
    cache.write_text("stale cache")

    selected = collector.candidate_files(tmp_path, max_files=7)
    relative = [path.relative_to(tmp_path).as_posix() for path in selected]

    assert relative[:4] == list(collector.CANONICAL_FILES)
    assert relative[4:] == [
        "experiments/exp010_recent/SESSION_NOTES.md",
        "experiments/exp010_recent/metrics.json",
        "experiments/exp010_recent/result.md",
    ]
    assert ".pytest_cache/README.md" not in relative


def test_experiment_reviewer_uses_current_canonical_paths(tmp_path: Path) -> None:
    reviewer = load_module(
        "review_exp_docs",
        ROOT / ".agents/skills/kaggle-review-exp/scripts/review_exp_docs.py",
    )
    for relative in reviewer.GLOBAL_RECORDS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("exp010 evidence")
    experiment = tmp_path / "experiments" / "exp010_candidate"
    experiment.mkdir(parents=True)
    (experiment / "result.md").write_text("# exp010 result")
    generated = experiment / "artifacts" / "stale.json"
    generated.parent.mkdir()
    generated.write_text("{}")
    obsolete = tmp_path / "submit" / "SUBMISSIONS.md"
    obsolete.parent.mkdir()
    obsolete.write_text("exp010 obsolete")

    selected = reviewer.candidate_files(tmp_path, "exp010")
    relative = {path.relative_to(tmp_path).as_posix() for path in selected}

    assert set(reviewer.GLOBAL_RECORDS) <= relative
    assert "experiments/exp010_candidate/result.md" in relative
    assert "experiments/exp010_candidate/artifacts/stale.json" not in relative
    assert "submit/SUBMISSIONS.md" not in relative


def test_experiment_reviewer_does_not_treat_global_context_as_target_evidence(
    tmp_path: Path,
) -> None:
    reviewer = load_module(
        "review_exp_docs_scopes",
        ROOT / ".agents/skills/kaggle-review-exp/scripts/review_exp_docs.py",
    )
    global_record = tmp_path / "KAGGLE_DIRECTION.md"
    global_record.write_text("objective baseline CV result artifact next action exp010")
    experiment = tmp_path / "experiments" / "exp010_candidate"
    experiment.mkdir(parents=True)
    result = experiment / "result.md"
    result.write_text("# exp010 result\n")

    reviews = reviewer.collect_reviews(
        tmp_path,
        reviewer.candidate_files(tmp_path, "exp010"),
    )
    scopes = {Path(review["path"]).name: review["scope"] for review in reviews}
    output = reviewer.render("exp010", tmp_path, reviews)

    assert scopes["KAGGLE_DIRECTION.md"] == "context"
    assert scopes["result.md"] == "target evidence"
    assert "Missing evidence in target experiment/steering records" in output
    assert "Core evidence categories are present in target" not in output


def test_experiment_reviewer_requires_canonical_target_records(tmp_path: Path) -> None:
    reviewer = load_module(
        "review_exp_docs_canonical",
        ROOT / ".agents/skills/kaggle-review-exp/scripts/review_exp_docs.py",
    )
    experiment = tmp_path / "experiments" / "exp010_candidate"
    package = experiment / "package"
    package.mkdir(parents=True)
    metadata = package / "metadata.json"
    metadata.write_text(
        '{"objective": "x", "baseline": "x", "cv": 1, "result": 1, "artifact": "x", "next": "x"}'
    )

    reviews = reviewer.collect_reviews(
        tmp_path,
        reviewer.candidate_files(tmp_path, "exp010"),
    )
    output = reviewer.render("exp010", tmp_path, reviews)
    has_target, missing = reviewer.target_evidence_summary(reviews)

    assert reviews[0]["scope"] == "supporting material"
    assert not has_target
    assert missing == list(reviewer.CHECKS)
    assert "No canonical target experiment or steering records were found." in output


def test_credential_checker_prefers_environment_token(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    credential_dir = tmp_path / ".kaggle"
    credential_dir.mkdir()
    token_file = credential_dir / "access_token"
    token_file.write_text("file-token-1234567890")
    token_file.chmod(0o600)

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "env-token-0987654321")
    monkeypatch.delenv("KAGGLE_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    checker = load_module(
        "check_all_credentials_env_priority",
        ROOT / ".agents/skills/kaggle-platform/shared/check_all_credentials.py",
    )

    assert checker.check_all_credentials()
    output = capsys.readouterr().out
    assert "from env" in output
    assert "env-token-0987654321" not in output
    assert "file-token-1234567890" not in output


def test_credential_checker_does_not_treat_kaggle_token_as_legacy_key(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KAGGLE_TOKEN", "ambiguous-token")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    checker = load_module(
        "check_all_credentials_unsupported_alias",
        ROOT / ".agents/skills/kaggle-platform/shared/check_all_credentials.py",
    )

    assert not checker.check_all_credentials()
    output = capsys.readouterr().out
    assert "rename it to KAGGLE_API_TOKEN" in output
    assert "KAGGLE_KEY" not in os.environ


def test_credential_checker_rejects_oauth_for_api_token_clients(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    credential_dir = tmp_path / ".kaggle"
    credential_dir.mkdir()
    oauth_file = credential_dir / "credentials.json"
    oauth_file.write_text("{}")
    oauth_file.chmod(0o600)

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    checker = load_module(
        "check_all_credentials_client_requirement",
        ROOT / ".agents/skills/kaggle-platform/shared/check_all_credentials.py",
    )

    assert checker.check_all_credentials(requirement="cli")
    assert not checker.check_all_credentials(requirement="api-token")
    output = capsys.readouterr().out
    assert "OAuth and legacy username/key credentials cannot authenticate MCP" in output


def test_credential_checker_requires_complete_legacy_pair(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KAGGLE_KEY", "legacy-key-only")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)

    checker = load_module(
        "check_all_credentials_legacy_pair",
        ROOT / ".agents/skills/kaggle-platform/shared/check_all_credentials.py",
    )

    assert not checker.check_all_credentials(requirement="cli")
    assert "unusable without KAGGLE_USERNAME" in capsys.readouterr().out


def test_credential_checker_accepts_legacy_for_python_api_but_not_mcp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "legacy-user")
    monkeypatch.setenv("KAGGLE_KEY", "legacy-key")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    checker = load_module(
        "check_all_credentials_legacy_clients",
        ROOT / ".agents/skills/kaggle-platform/shared/check_all_credentials.py",
    )

    assert checker.check_all_credentials(requirement="cli")
    assert checker.check_all_credentials(requirement="python-api")
    assert not checker.check_all_credentials(requirement="api-token")


def test_credential_checker_documents_selected_requirement_exit_codes() -> None:
    source = (ROOT / ".agents/skills/kaggle-platform/shared/check_all_credentials.py").read_text()

    assert "--require api-token" in source
    assert "--require python-api" in source
    assert "--require cli" in source
    assert "satisfying the selected requirement was found" in source
    assert "At least one usable credential source found" not in source


def test_kllm_docs_select_credentials_for_each_client() -> None:
    skill = (ROOT / ".agents/skills/kaggle-platform/SKILL.md").read_text()
    kllm = (ROOT / ".agents/skills/kaggle-platform/modules/kllm/README.md").read_text()
    hackathon = (
        ROOT / ".agents/skills/kaggle-platform/modules/kllm/hackathon/README.md"
    ).read_text()

    for source in (skill, kllm):
        assert "check_all_credentials.py --require cli" in source
        assert "check_all_credentials.py --require python-api" in source
        assert "check_all_credentials.py --require api-token" in source
    assert "check_all_credentials.py --require api-token" in hackathon


def test_competition_report_mcp_enrichment_requires_api_token() -> None:
    platform = ROOT / ".agents/skills/kaggle-platform"
    skill = (platform / "SKILL.md").read_text()
    comp_report = (platform / "modules/comp-report/README.md").read_text()
    kllm = (platform / "modules/kllm/README.md").read_text()
    overview = (platform / "modules/kllm/references/competition-overview.md").read_text()
    normalized_kllm = " ".join(kllm.split())

    assert "legacy username/keyだけの場合はMCP補完を省略する" in skill
    assert "`--require api-token`で追加確認" in skill
    assert "does not accept\n> legacy username/key credentials" in comp_report
    assert (
        "legacy username/key credentials alone cannot authenticate the MCP call"
        in normalized_kllm
    )
    assert "cannot authenticate this call" in overview


def test_hackathon_writeup_status_is_recorded_as_dated_evidence() -> None:
    module = ROOT / ".agents/skills/kaggle-platform/modules/kllm"
    paths = [
        module / "references/mcp-reference.md",
        module / "hackathon/README.md",
        module / "hackathon/references/hackathon-endpoints.md",
        module / "hackathon/scripts/list_writeups.py",
        module / "hackathon/scripts/fetch_writeup.py",
    ]
    sources = [path.read_text() for path in paths]
    stale_status_label = "known" + "-broken"

    assert all(stale_status_label not in source for source in sources)
    assert "2026-04-22 audit" in sources[0]
    assert "2026-05-04 retest" in sources[0]


def test_notebook_fetch_uses_kaggle_from_current_interpreter(monkeypatch) -> None:
    fetcher = load_module(
        "fetch_top_notebooks_locked_cli",
        ROOT / ".agents/skills/kaggle-notebook-fetch/scripts/fetch_top_notebooks.py",
    )
    observed: list[str] = []

    def fake_run(command, **kwargs):
        observed.extend(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="ref,title\nowner/example,Example\n",
        )

    monkeypatch.setattr(fetcher.subprocess, "run", fake_run)

    rows = fetcher.list_kernels("example-competition", 20, "voteCount")

    assert observed[:3] == [sys.executable, "-m", "kaggle"]
    assert rows == [{"ref": "owner/example", "title": "Example"}]


def test_badge_streak_single_run_reports_actions_but_does_not_verify_badges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scripts_dir = ROOT / ".agents/skills/kaggle-platform/modules/badge-collector/scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    for module_name in ("utils", "badge_registry", "badge_tracker"):
        sys.modules.pop(module_name, None)
    streaks = load_module(
        "badge_phase_5_streaks",
        scripts_dir / "phase_5_streaks.py",
    )

    statuses: list[tuple[str, str]] = []
    generated_script = tmp_path / "daily_streak.sh"
    monkeypatch.setattr(streaks, "should_attempt", lambda badge_id: True)
    monkeypatch.setattr(streaks, "get_kaggle_cli", lambda: ("kaggle",))
    monkeypatch.setattr(streaks, "_run_today", lambda command: True)
    monkeypatch.setattr(streaks, "_create_daily_script", lambda: generated_script)
    monkeypatch.setattr(streaks, "_print_scheduling_instructions", lambda path: None)
    monkeypatch.setattr(
        streaks,
        "set_status",
        lambda badge_id, status, details=None: statuses.append((badge_id, status)),
    )

    attempted, completed = streaks.run("owner")

    assert attempted == 1
    assert completed == 1
    assert {status for _, status in statuses} == {"attempting"}


def test_badge_registry_counts_match_documented_workflow_scope(monkeypatch) -> None:
    scripts_dir = ROOT / ".agents/skills/kaggle-platform/modules/badge-collector/scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    registry = load_module("badge_registry_counts", scripts_dir / "badge_registry.py")

    phase_counts = {phase: len(registry.get_badges_by_phase(phase)) for phase in (1, 2, 3, 4, 5)}
    assert len(registry.ALL_BADGES) == 55
    assert phase_counts == {1: 16, 2: 7, 3: 3, 4: 8, 5: 4}
    assert len(registry.get_workflow_badges()) == 38
    assert len(registry.get_automatable_badges()) == 30
    catalog = (
        ROOT / ".agents/skills/kaggle-platform/modules/badge-collector/references/badge-catalog.md"
    ).read_text()
    assert "| **Total** | **55** | **38** |" in catalog
    assert "## Not supported by the collector workflow (~17 badges)" in catalog
    assert "## Not Automatable" not in catalog


def test_comp_report_documents_playwright_as_optional() -> None:
    platform_dir = ROOT / ".agents/skills/kaggle-platform"
    skill = (platform_dir / "SKILL.md").read_text()
    readme = (platform_dir / "modules/comp-report/README.md").read_text()

    assert "Playwright MCP tools are not a prerequisite" in readme
    assert "Playwright MCP tools available in your agent" not in readme
    assert "場合だけ scraping を追加する" in skill
    assert "Playwright scraping、report 作成" not in skill


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["--phase", "1", "--status"],
            "--phase cannot be combined with --status or a manual status update",
        ),
        (
            ["--phase", "1", "--mark-verified", "python_coder"],
            "--phase cannot be combined with --status or a manual status update",
        ),
        (
            ["--details", "profile evidence"],
            "--details requires --mark-action-completed or --mark-verified",
        ),
    ],
)
def test_badge_orchestrator_rejects_ambiguous_arguments(
    arguments: list[str],
    message: str,
) -> None:
    orchestrator = (
        ROOT / ".agents/skills/kaggle-platform/modules/badge-collector/scripts/orchestrator.py"
    )

    result = subprocess.run(
        [sys.executable, str(orchestrator), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_badge_credentials_are_selected_by_phase(monkeypatch) -> None:
    scripts_dir = ROOT / ".agents/skills/kaggle-platform/modules/badge-collector/scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    utils = load_module("badge_phase_credentials", scripts_dir / "utils.py")

    assert utils.credential_requirement_for_phases([1]) == "python-api"
    assert utils.credential_requirement_for_phases([1, 2, 3, 4, 5]) == "python-api"
    assert utils.credential_requirement_for_phases([2, 3, 5]) == "cli"
    assert utils.credential_requirement_for_phases([4]) is None


def test_badge_pipeline_follows_live_logs_without_status_polling(monkeypatch) -> None:
    scripts_dir = ROOT / ".agents/skills/kaggle-platform/modules/badge-collector/scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    for module_name in ("utils", "badge_registry", "badge_tracker"):
        sys.modules.pop(module_name, None)
    phase = load_module("badge_phase_3_live_logs", scripts_dir / "phase_3_pipeline.py")
    observed: list[tuple[list[str], dict]] = []

    def fake_run(args, **kwargs):
        observed.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(phase, "run_kaggle_cli", fake_run)

    assert phase._follow_kernel_logs("owner", "kernel")
    assert observed == [
        (
            ["kernels", "logs", "-f", "owner/kernel"],
            {"check": False, "timeout": 600, "stream_output": True},
        )
    ]
    assert '["kernels", "status"' not in (scripts_dir / "phase_3_pipeline.py").read_text()


def test_badge_tracker_separates_action_completion_from_verification(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scripts_dir = ROOT / ".agents/skills/kaggle-platform/modules/badge-collector/scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    sys.modules.pop("badge_registry", None)
    sys.modules.pop("utils", None)
    tracker = load_module("badge_tracker_statuses", scripts_dir / "badge_tracker.py")
    progress_file = tmp_path / "badge-progress.json"
    progress_file.write_text(
        '{"python_coder": {"status": "earned"}, "stylish": {"status": "skipped"}}'
    )
    monkeypatch.setattr(tracker, "PROGRESS_FILE", progress_file)

    progress = tracker.load_progress()

    assert progress["python_coder"]["status"] == "verification_required"
    assert progress["stylish"]["status"] == "manual_required"
    with pytest.raises(ValueError, match="Unknown badge status"):
        tracker.set_status("python_coder", "earned")


def test_badge_phase_4_only_prints_guidance(monkeypatch) -> None:
    scripts_dir = ROOT / ".agents/skills/kaggle-platform/modules/badge-collector/scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    sys.modules.pop("badge_tracker", None)
    phase = load_module("badge_phase_4_manual", scripts_dir / "phase_4_manual.py")
    statuses: list[tuple[str, str]] = []
    monkeypatch.setattr(phase, "should_attempt", lambda badge_id: True)
    monkeypatch.setattr(
        phase,
        "set_status",
        lambda badge_id, status, details=None: statuses.append((badge_id, status)),
    )

    attempted, completed = phase.run("owner")

    assert attempted == 0
    assert completed == 0
    assert len(statuses) == 8
    assert {status for _, status in statuses} == {"manual_required"}
    source = (scripts_dir / "phase_4_manual.py").read_text()
    assert "sync_playwright" not in source
    assert "Data scientist and machine learning enthusiast" not in source


def test_badge_phases_do_not_record_actions_as_earned() -> None:
    scripts_dir = ROOT / ".agents/skills/kaggle-platform/modules/badge-collector/scripts"
    for phase_path in scripts_dir.glob("phase_*.py"):
        assert '"earned"' not in phase_path.read_text(), phase_path


def test_badge_daily_script_uses_gitignored_state_and_runtime_sample(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scripts_dir = ROOT / ".agents/skills/kaggle-platform/modules/badge-collector/scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    for module_name in ("utils", "badge_registry", "badge_tracker"):
        sys.modules.pop(module_name, None)
    streaks = load_module(
        "badge_phase_5_script",
        scripts_dir / "phase_5_streaks.py",
    )
    state_dir = tmp_path / ".badge-collector"
    script_path = state_dir / "daily_streak.sh"
    monkeypatch.setattr(streaks, "STATE_DIR", state_dir)
    monkeypatch.setattr(streaks, "DAILY_SCRIPT_PATH", script_path)

    created = streaks._create_daily_script()
    content = created.read_text()

    assert created == script_path
    assert "competitions download titanic" in content
    assert "gender_submission.csv" in content
    assert "submission_titanic.csv" not in content

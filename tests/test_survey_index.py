from pathlib import Path

import pytest

from scripts.update_survey_index import load_report, render_generated_index, update_index


def write_report(path: Path, *, status: str = "final", summary: str = "結論") -> None:
    path.write_text(
        "---\n"
        "title: selector調査\n"
        "date: 2026-08-06\n"
        "types:\n"
        "  - experiment_review\n"
        "  - oof_analysis\n"
        "experiments:\n"
        "  - exp238\n"
        "topics:\n"
        "  - selector\n"
        f"status: {status}\n"
        f"summary: {summary}\n"
        "---\n\n"
        "# selector調査\n"
    )


def test_load_report_parses_search_metadata(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    write_report(report_path)

    report = load_report(report_path)

    assert report.experiments == ("exp238",)
    assert report.types == ("experiment_review", "oof_analysis")
    assert report.topics == ("selector",)
    assert report.status == "final"


def test_load_report_rejects_missing_front_matter(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    report_path.write_text("# selector調査\n")

    with pytest.raises(ValueError, match="front matter"):
        load_report(report_path)


def test_load_report_rejects_todo_in_final_summary(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    write_report(report_path, summary="TODO")

    with pytest.raises(ValueError, match="must not contain TODO"):
        load_report(report_path)


def test_render_generated_index_groups_by_experiment_type_and_topic(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    write_report(report_path)
    report = load_report(report_path)

    rendered = render_generated_index([report])

    assert "[selector調査](selector.md)" in rendered
    assert "## 実験番号別" in rendered
    assert "## 種類別" in rendered
    assert "## トピック別" in rendered
    assert "`exp238`" in rendered
    assert "`oof_analysis`" in rendered
    assert "`selector`" in rendered


def test_update_index_check_detects_stale_readme(tmp_path: Path) -> None:
    surveys_dir = tmp_path / "surveys"
    surveys_dir.mkdir()
    report_path = surveys_dir / "selector.md"
    write_report(report_path)
    readme_path = surveys_dir / "README.md"
    readme_path.write_text(
        "# 調査レポート\n\n<!-- BEGIN AUTO SURVEY INDEX -->\n<!-- END AUTO SURVEY INDEX -->\n"
    )

    with pytest.raises(SystemExit, match="out of date"):
        update_index(surveys_dir=surveys_dir, readme_path=readme_path, check=True)

    assert update_index(surveys_dir=surveys_dir, readme_path=readme_path)
    assert not update_index(surveys_dir=surveys_dir, readme_path=readme_path, check=True)


def test_update_index_check_rejects_draft_report(tmp_path: Path) -> None:
    surveys_dir = tmp_path / "surveys"
    surveys_dir.mkdir()
    write_report(surveys_dir / "selector.md", status="draft", summary="TODO")
    readme_path = surveys_dir / "README.md"
    readme_path.write_text(
        "# 調査レポート\n\n<!-- BEGIN AUTO SURVEY INDEX -->\n<!-- END AUTO SURVEY INDEX -->\n"
    )

    with pytest.raises(SystemExit, match="draft survey reports"):
        update_index(surveys_dir=surveys_dir, readme_path=readme_path, check=True)

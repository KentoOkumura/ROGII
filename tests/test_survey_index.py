import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from scripts.update_survey_index import load_report, render_generated_index, update_index

ROOT = Path(__file__).resolve().parents[1]


def load_survey_generator():
    scripts_dir = ROOT / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "new_survey_report_for_tests",
            scripts_dir / "new_survey_report.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts_dir))


def write_report(
    path: Path,
    *,
    status: str = "final",
    summary: str = "結論",
    include_hypotheses: bool = True,
    include_body_declaration: bool = True,
    superseded_by: str | None = None,
) -> None:
    hypotheses = "hypotheses:\n  - HYP-19000101-91\n" if include_hypotheses else ""
    if not include_body_declaration:
        body_hypotheses = ""
    elif include_hypotheses:
        body_hypotheses = "\n- 対応する上位仮説: `HYP-19000101-91`\n"
    else:
        body_hypotheses = "\n- 対応する上位仮説: なし\n"
    replacement = f"superseded_by: {superseded_by}\n" if superseded_by else ""
    path.write_text(
        "---\n"
        "title: selector調査\n"
        "date: 2026-08-06\n"
        "types:\n"
        "  - experiment_review\n"
        "  - oof_analysis\n"
        f"{hypotheses}"
        "experiments:\n"
        "  - exp238\n"
        "topics:\n"
        "  - selector\n"
        f"status: {status}\n"
        f"{replacement}"
        f"summary: {summary}\n"
        "---\n\n"
        "# selector調査\n"
        f"{body_hypotheses}"
    )


def test_load_report_parses_search_metadata(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    write_report(report_path)

    report = load_report(report_path)

    assert report.hypotheses == ("HYP-19000101-91",)
    assert report.experiments == ("exp238",)
    assert report.types == ("experiment_review", "oof_analysis")
    assert report.topics == ("selector",)
    assert report.status == "final"


def test_new_survey_body_renders_hypothesis_declaration() -> None:
    generator = load_survey_generator()

    body = generator.render_body(
        "selector調査",
        "2026-08-12",
        ["HYP-19000101-91", "HYP-19000101-92"],
    )

    assert "- 対応する上位仮説: `HYP-19000101-91`, `HYP-19000101-92`" in body
    assert "{{ HYPOTHESES }}" not in body


def test_load_report_accepts_report_without_hypotheses_when_body_declares_none(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "selector.md"
    write_report(report_path, include_hypotheses=False)

    report = load_report(report_path)

    assert report.hypotheses == ()


def test_load_report_rejects_non_draft_without_hypothesis_declaration(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    write_report(
        report_path,
        include_hypotheses=False,
        include_body_declaration=False,
    )

    with pytest.raises(ValueError, match="must declare corresponding hypotheses"):
        load_report(report_path)


def test_load_report_rejects_invalid_hypothesis_id(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    write_report(report_path)
    report_path.write_text(report_path.read_text().replace("HYP-19000101-91", "HYP-1"))

    with pytest.raises(ValueError, match="HYP-YYYYMMDD-NN"):
        load_report(report_path)


def test_load_report_rejects_missing_front_matter(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    report_path.write_text("# selector調査\n")

    with pytest.raises(ValueError, match="front matter"):
        load_report(report_path)


def test_load_report_rejects_todo_in_final_summary(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    write_report(report_path, summary="TODO")

    with pytest.raises(ValueError, match="summary must not contain placeholders"):
        load_report(report_path)


def test_load_report_rejects_placeholder_in_final_body(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    write_report(report_path)
    report_path.write_text(report_path.read_text() + "\n## 結論\n\nTODO\n")

    with pytest.raises(ValueError, match="body must not contain placeholders"):
        load_report(report_path)


def test_load_report_rejects_body_hypothesis_metadata_mismatch(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    write_report(report_path)
    report_path.write_text(
        report_path.read_text().replace(
            "対応する上位仮説: `HYP-19000101-91`",
            "対応する上位仮説: `HYP-19000101-92`",
        )
    )

    with pytest.raises(ValueError, match="must match hypotheses metadata"):
        load_report(report_path)


def test_load_report_requires_none_for_non_hypothesis_body_declaration(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    write_report(report_path, include_hypotheses=False)
    report_path.write_text(
        report_path.read_text().replace("対応する上位仮説: なし", "対応する上位仮説: 未指定")
    )

    with pytest.raises(ValueError, match="対応する上位仮説: なし"):
        load_report(report_path)


def test_load_report_accepts_embedded_snapshot_with_matching_sha(tmp_path: Path) -> None:
    report_path = tmp_path / "snapshot.md"
    write_report(report_path, include_hypotheses=False)
    snapshot = "# 保存時点の文書\n\n変更しない本文。\n"
    snapshot_sha = hashlib.sha256(snapshot.encode()).hexdigest()
    report_path.write_text(
        report_path.read_text()
        + f"\n- 元ファイルSHA-256: `{snapshot_sha}`\n"
        + "\n## 移行前の全文\n\n"
        + snapshot
    )

    assert load_report(report_path).title == "selector調査"


def test_load_report_rejects_modified_embedded_snapshot(tmp_path: Path) -> None:
    report_path = tmp_path / "snapshot.md"
    write_report(report_path, include_hypotheses=False)
    report_path.write_text(
        report_path.read_text()
        + "\n- 元ファイルSHA-256: `"
        + "0" * 64
        + "`\n\n## 移行前の全文\n\n# 変更された本文\n"
    )

    with pytest.raises(ValueError, match="embedded snapshot SHA mismatch"):
        load_report(report_path)


def test_load_report_requires_snapshot_body_for_declared_sha(tmp_path: Path) -> None:
    report_path = tmp_path / "snapshot.md"
    write_report(report_path, include_hypotheses=False)
    report_path.write_text(report_path.read_text() + "\n- 元ファイルSHA-256: `" + "0" * 64 + "`\n")

    with pytest.raises(ValueError, match="requires an embedded 移行前の全文"):
        load_report(report_path)


def test_load_report_accepts_superseded_report_with_final_replacement(tmp_path: Path) -> None:
    replacement_path = tmp_path / "replacement.md"
    write_report(replacement_path)
    report_path = tmp_path / "selector.md"
    write_report(report_path, status="superseded", superseded_by=replacement_path.name)

    report = load_report(report_path)

    assert report.superseded_by == "replacement.md"
    assert "[後継](replacement.md)" in render_generated_index([report])


def test_load_report_accepts_superseded_chain_ending_in_final(tmp_path: Path) -> None:
    final_path = tmp_path / "final.md"
    write_report(final_path)
    middle_path = tmp_path / "middle.md"
    write_report(middle_path, status="superseded", superseded_by=final_path.name)
    report_path = tmp_path / "selector.md"
    write_report(report_path, status="superseded", superseded_by=middle_path.name)

    report = load_report(report_path)

    assert report.superseded_by == "middle.md"
    assert "[後継](middle.md)" in render_generated_index([report])


def test_load_report_rejects_superseded_report_without_replacement(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    write_report(report_path, status="superseded")

    with pytest.raises(ValueError, match="must set superseded_by"):
        load_report(report_path)


def test_load_report_rejects_superseded_chain_without_final_endpoint(tmp_path: Path) -> None:
    replacement_path = tmp_path / "replacement.md"
    write_report(replacement_path, status="draft", summary="TODO")
    report_path = tmp_path / "selector.md"
    write_report(report_path, status="superseded", superseded_by=replacement_path.name)

    with pytest.raises(ValueError, match="must terminate at a final report"):
        load_report(report_path)


def test_load_report_rejects_superseded_cycle(tmp_path: Path) -> None:
    report_path = tmp_path / "selector.md"
    replacement_path = tmp_path / "replacement.md"
    write_report(report_path, status="superseded", superseded_by=replacement_path.name)
    write_report(replacement_path, status="superseded", superseded_by=report_path.name)

    with pytest.raises(ValueError, match="contains a cycle"):
        load_report(report_path)


def test_render_generated_index_groups_by_hypothesis_experiment_type_and_topic(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "selector.md"
    write_report(report_path)
    report = load_report(report_path)

    rendered = render_generated_index([report])

    assert "[selector調査](selector.md)" in rendered
    assert "## 上位仮説別" in rendered
    assert "## 実験番号別" in rendered
    assert "## 種類別" in rendered
    assert "## トピック別" in rendered
    assert "`HYP-19000101-91`" in rendered
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


def test_update_index_check_can_allow_draft_report(tmp_path: Path) -> None:
    surveys_dir = tmp_path / "surveys"
    surveys_dir.mkdir()
    write_report(surveys_dir / "selector.md", status="draft", summary="TODO")
    readme_path = surveys_dir / "README.md"
    readme_path.write_text(
        "# 調査レポート\n\n<!-- BEGIN AUTO SURVEY INDEX -->\n<!-- END AUTO SURVEY INDEX -->\n"
    )

    assert update_index(surveys_dir=surveys_dir, readme_path=readme_path)
    assert not update_index(
        surveys_dir=surveys_dir,
        readme_path=readme_path,
        check=True,
        allow_draft=True,
    )

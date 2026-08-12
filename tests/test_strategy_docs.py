from pathlib import Path

from scripts.check_strategy_docs import validate_strategy_docs


def write_valid_strategy_docs(root: Path) -> None:
    backlog = root / "docs" / "backlog"
    backlog.mkdir(parents=True)
    (backlog / "candidate_a.md").write_text(
        "# candidate_a\n\n- 候補名: `candidate_a`\n- 状態: `検討メモ・設計不可`\n"
    )
    (root / "KAGGLE_DIRECTION.md").write_text(
        "# Kaggle 方針\n\n"
        "## アイデアバックログ\n\n"
        "### 未着手バックログ\n\n"
        "| 優先度 | アイデア | 短い要約 | 主な先行条件 / 依存 | 状態 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| P2 | [`candidate_a`](docs/backlog/candidate_a.md) | summary | dependency | "
        "`検討メモ・設計不可` |\n"
    )


def test_strategy_docs_accept_matching_index_and_detail(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)

    assert validate_strategy_docs(tmp_path) == []


def test_strategy_docs_reject_orphaned_detail(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    (tmp_path / "docs" / "backlog" / "orphan.md").write_text("# orphan\n")

    errors = validate_strategy_docs(tmp_path)

    assert any("unreferenced backlog detail files: orphan" in error for error in errors)


def test_strategy_docs_reject_state_mismatch(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    detail = tmp_path / "docs" / "backlog" / "candidate_a.md"
    detail.write_text(detail.read_text().replace("検討メモ・設計不可", "設計可能・実験化未承認"))

    errors = validate_strategy_docs(tmp_path)

    assert any("mismatched candidate state" in error for error in errors)

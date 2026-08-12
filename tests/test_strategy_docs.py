from pathlib import Path

from scripts.check_strategy_docs import validate_strategy_docs


def write_valid_strategy_docs(root: Path) -> None:
    backlog = root / "docs" / "backlog"
    backlog.mkdir(parents=True)
    (backlog / "candidate_a.md").write_text(
        "# candidate_a\n\n- 候補名: `candidate_a`\n"
        "- 状態: `検討メモ・設計不可`\n"
        "- 対応する上位仮説: `HYP-20260812-01`\n"
    )
    (root / "KAGGLE_DIRECTION.md").write_text(
        "# Kaggle 方針\n\n"
        "## アイデアバックログ\n\n"
        "### 検証中の仮説\n\n"
        "| 仮説ID | 仮説 | 対応する未着手候補 | 対応する実験 | 残っている問い |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `HYP-20260812-01` | hypothesis | "
        "[`candidate_a`](docs/backlog/candidate_a.md) | - | remaining |\n\n"
        "### 未着手バックログ\n\n"
        "| 優先度 | 対応仮説 | アイデア | 短い要約 | 主な先行条件 / 依存 | 状態 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| P2 | `HYP-20260812-01` | [`candidate_a`](docs/backlog/candidate_a.md) | "
        "summary | dependency | "
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


def test_strategy_docs_reject_missing_hypothesis(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    direction = tmp_path / "KAGGLE_DIRECTION.md"
    direction.write_text(
        direction.read_text().replace(
            "HYP-20260812-01` | hypothesis", "HYP-20260812-02` | hypothesis"
        )
    )

    errors = validate_strategy_docs(tmp_path)

    assert any("references missing hypothesis HYP-20260812-01" in error for error in errors)


def test_strategy_docs_reject_hypothesis_detail_mismatch(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    detail = tmp_path / "docs" / "backlog" / "candidate_a.md"
    detail.write_text(detail.read_text().replace("HYP-20260812-01", "HYP-20260812-02"))

    errors = validate_strategy_docs(tmp_path)

    assert any("mismatched hypothesis id" in error for error in errors)


def test_strategy_docs_reject_hypothesis_candidate_mapping_mismatch(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    direction = tmp_path / "KAGGLE_DIRECTION.md"
    direction.write_text(
        direction.read_text().replace(
            "[`candidate_a`](docs/backlog/candidate_a.md) | - | remaining",
            "- | - | remaining",
        )
    )

    errors = validate_strategy_docs(tmp_path)

    assert any("candidate mapping does not match backlog rows" in error for error in errors)


def test_strategy_docs_accept_unassigned_legacy_candidate(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    direction = tmp_path / "KAGGLE_DIRECTION.md"
    text = direction.read_text()
    hypothesis_row = (
        "| `HYP-20260812-01` | hypothesis | "
        "[`candidate_a`](docs/backlog/candidate_a.md) | - | remaining |\n"
    )
    text = text.replace(hypothesis_row, "")
    text = text.replace("`HYP-20260812-01` | [`candidate_a`]", "`未整理` | [`candidate_a`]")
    direction.write_text(text)
    detail = tmp_path / "docs" / "backlog" / "candidate_a.md"
    detail.write_text(detail.read_text().replace("HYP-20260812-01", "未整理"))

    assert validate_strategy_docs(tmp_path) == []

from pathlib import Path

from scripts.check_strategy_docs import validate_strategy_docs


def write_valid_strategy_docs(root: Path) -> None:
    backlog = root / "backlog"
    backlog.mkdir(parents=True)
    (backlog / "candidate_a.md").write_text(
        "# candidate_a\n\n- 候補名: `candidate_a`\n"
        "- 状態: `検討メモ・設計不可`\n"
        "- 対応する上位仮説: `HYP-19000101-91`\n"
        "- 優先度: P2\n"
        "- 優先度の理由: next candidate\n"
    )
    (root / "backlog/KAGGLE_DIRECTION.md").write_text(
        "# Kaggle 方針\n\n"
        "## アイデアバックログ\n\n"
        "### 検証中の仮説\n\n"
        "| 仮説ID | 仮説 | 対応する未着手候補 | 対応する実験 | 残っている問い |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `HYP-19000101-91` | hypothesis | "
        "[`candidate_a`](candidate_a.md) | - | remaining |\n\n"
        "### 未着手バックログ\n\n"
        "| 優先度 | 対応仮説 | アイデア | 短い要約 | 主な先行条件 / 依存 | 状態 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| P2 | `HYP-19000101-91` | [`candidate_a`](candidate_a.md) | "
        "summary | dependency | "
        "`検討メモ・設計不可` |\n"
    )


def make_candidate_design_ready(root: Path) -> None:
    detail = root / "backlog" / "candidate_a.md"
    detail.write_text(
        "# candidate_a\n\n"
        "- 候補名: `candidate_a`\n"
        "- 状態: `設計可能・実験化未承認`\n"
        "- 対応する上位仮説: `HYP-19000101-91`\n"
        "- 優先度: P2\n"
        "- 優先度の理由: next candidate\n"
        "- 親実験 / 比較対象: `exp001_baseline`\n\n"
        "## 観測事実と根拠\n\n"
        "- 実測済みの事実: result.mdで差を確認済み\n"
        "- 根拠ファイル / 一次資料: experiments/exp001_baseline/result.md\n\n"
        "## この候補が直接検証する仮説と範囲\n\n"
        "- 上位仮説のうちこの候補が検証する範囲: feature追加の寄与\n"
        "- この候補の具体的な仮説: featureを追加するとCVが改善する\n\n"
        "- 仮説が正しい場合に期待する観測: 同一foldでCV RMSEが改善する\n"
        "- 仮説を棄却する観測: 同一foldでCV RMSEが改善しない\n"
        "- この候補だけで上位仮説を判断できるか: いいえ\n"
        "- 上位仮説の判断に残る検証: 別modelでの再現\n\n"
        "## 親実験からの差分\n\n"
        "- 変更するもの: featureを1列追加する\n"
        "- 固定するもの: splitとmodelを固定する\n\n"
        "## 最小の反証可能な検証\n\n"
        "- 検証方法: 同じfoldで追加有無を比較する\n\n"
        "## 成功条件と停止条件\n\n"
        "- primary指標: CV RMSEが改善する\n"
        "- 成功条件: 親実験よりCV RMSEが0.01以上改善する\n"
        "- 失敗時の停止範囲: feature追加を停止する\n\n"
        "## 実行しないこと\n\n"
        "- 禁止する代替実装、proxy、同一OOF上の救済探索: threshold探索をしない\n"
        "- 壁打ちで採らなかった案と理由: model変更は差分が増えるため採らない\n\n"
        "## 未決事項\n\n"
        "- なし\n"
    )
    direction = root / "backlog/KAGGLE_DIRECTION.md"
    direction.write_text(
        direction.read_text().replace("`検討メモ・設計不可` |", "`設計可能・実験化未承認` |")
    )


def add_mapped_experiment(root: Path, *, hypothesis_id: str = "HYP-19000101-91") -> None:
    experiment_name = "exp001_candidate"
    experiment_dir = root / "experiments" / experiment_name
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "config.yaml").write_text(
        f"lineage:\n  hypothesis_id: {hypothesis_id}\n  backlog_candidate: candidate_a\n"
    )
    direction = root / "backlog/KAGGLE_DIRECTION.md"
    direction.write_text(
        direction.read_text().replace(
            "[`candidate_a`](candidate_a.md) | - | remaining",
            "[`candidate_a`](candidate_a.md) | "
            "[`exp001_candidate`](../experiments/exp001_candidate/) | remaining",
        )
    )


def add_unmapped_experiment(root: Path, hypothesis_id: str) -> None:
    experiment_dir = root / "experiments" / "exp002_unmapped"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "config.yaml").write_text(
        f"lineage:\n  hypothesis_id: {hypothesis_id}\n  backlog_candidate: candidate_a\n"
    )


def test_strategy_docs_accept_matching_index_and_detail(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)

    assert validate_strategy_docs(tmp_path) == []


def test_strategy_docs_accept_empty_backlog(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    direction = tmp_path / "backlog/KAGGLE_DIRECTION.md"
    text = direction.read_text()
    text = text.replace(
        "| `HYP-19000101-91` | hypothesis | "
        "[`candidate_a`](candidate_a.md) | - | remaining |\n",
        "",
    )
    text = text.replace(
        "| P2 | `HYP-19000101-91` | [`candidate_a`](candidate_a.md) | "
        "summary | dependency | `検討メモ・設計不可` |\n",
        "",
    )
    direction.write_text(text)
    (tmp_path / "backlog" / "candidate_a.md").unlink()

    assert validate_strategy_docs(tmp_path) == []


def test_strategy_docs_accept_completed_design_ready_detail(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    make_candidate_design_ready(tmp_path)

    assert validate_strategy_docs(tmp_path) == []


def test_strategy_docs_reject_incomplete_design_ready_detail(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    make_candidate_design_ready(tmp_path)
    detail = tmp_path / "backlog" / "candidate_a.md"
    detail.write_text(
        detail.read_text().replace("- 変更するもの: featureを1列追加する", "- 変更するもの: TODO")
    )

    errors = validate_strategy_docs(tmp_path)

    assert any("missing completed fields: 変更するもの" in error for error in errors)
    assert any("still contains placeholders" in error for error in errors)


def test_strategy_docs_reject_design_ready_detail_missing_hypothesis_boundary(
    tmp_path: Path,
) -> None:
    write_valid_strategy_docs(tmp_path)
    make_candidate_design_ready(tmp_path)
    detail = tmp_path / "backlog" / "candidate_a.md"
    detail.write_text(
        detail.read_text().replace(
            "- 上位仮説の判断に残る検証: 別modelでの再現\n",
            "",
        )
    )

    errors = validate_strategy_docs(tmp_path)

    assert any("missing completed fields: 上位仮説の判断に残る検証" in error for error in errors)


def test_strategy_docs_reject_invalid_candidate_only_decision(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    make_candidate_design_ready(tmp_path)
    detail = tmp_path / "backlog" / "candidate_a.md"
    detail.write_text(
        detail.read_text().replace(
            "- この候補だけで上位仮説を判断できるか: いいえ",
            "- この候補だけで上位仮説を判断できるか: 未確認",
        )
    )

    errors = validate_strategy_docs(tmp_path)

    assert any("must answer whether this candidate alone" in error for error in errors)


def test_strategy_docs_reject_design_ready_detail_with_unresolved_items(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    make_candidate_design_ready(tmp_path)
    detail = tmp_path / "backlog" / "candidate_a.md"
    detail.write_text(detail.read_text().replace("- なし\n", "- thresholdを確認する\n"))

    errors = validate_strategy_docs(tmp_path)

    assert any("unresolved items are not exactly '- なし'" in error for error in errors)


def test_strategy_docs_accept_matching_experiment_lineage(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    add_mapped_experiment(tmp_path)

    assert validate_strategy_docs(tmp_path) == []


def test_strategy_docs_reject_experiment_lineage_mismatch(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    add_mapped_experiment(tmp_path, hypothesis_id="HYP-19000101-92")

    errors = validate_strategy_docs(tmp_path)

    assert any(
        "experiment mapping does not match experiment config lineage" in error for error in errors
    )


def test_strategy_docs_reject_unregistered_experiment_hypothesis(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    add_unmapped_experiment(tmp_path, "HYP-19000101-92")

    errors = validate_strategy_docs(tmp_path)

    assert any("references unregistered hypothesis HYP-19000101-92" in error for error in errors)


def test_strategy_docs_accept_experiment_hypothesis_archived_in_final_survey(
    tmp_path: Path,
) -> None:
    write_valid_strategy_docs(tmp_path)
    add_unmapped_experiment(tmp_path, "HYP-19000101-92")
    surveys = tmp_path / "docs" / "surveys"
    surveys.mkdir(parents=True)
    (surveys / "archived.md").write_text(
        "---\n"
        "title: archived hypothesis\n"
        "date: 2026-08-12\n"
        "types:\n  - experiment_review\n"
        "hypotheses:\n  - HYP-19000101-92\n"
        "experiments: []\n"
        "topics:\n  - lineage\n"
        "status: final\n"
        "summary: archived\n"
        "---\n\n"
        "# archived hypothesis\n\n"
        "- 対応する上位仮説: `HYP-19000101-92`\n"
    )

    assert validate_strategy_docs(tmp_path) == []


def test_strategy_docs_reject_placeholder_hypothesis_content(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    direction = tmp_path / "backlog/KAGGLE_DIRECTION.md"
    direction.write_text(
        direction.read_text()
        .replace(
            "| `HYP-19000101-91` | hypothesis |",
            "| `HYP-19000101-91` | TODO |",
        )
        .replace("| - | remaining |", "| - | TODO |")
    )

    errors = validate_strategy_docs(tmp_path)

    assert any("placeholder statement" in error for error in errors)
    assert any("placeholder remaining question" in error for error in errors)


def test_strategy_docs_reject_braced_hypothesis_placeholders(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    direction = tmp_path / "backlog/KAGGLE_DIRECTION.md"
    direction.write_text(
        direction.read_text()
        .replace(
            "| `HYP-19000101-91` | hypothesis |",
            "| `HYP-19000101-91` | {{ HYPOTHESIS }} |",
        )
        .replace("| - | remaining |", "| - | {{ REMAINING }} |")
    )

    errors = validate_strategy_docs(tmp_path)

    assert any("placeholder statement" in error for error in errors)
    assert any("placeholder remaining question" in error for error in errors)


def test_strategy_docs_reject_braced_placeholder_in_design_ready_detail(
    tmp_path: Path,
) -> None:
    write_valid_strategy_docs(tmp_path)
    make_candidate_design_ready(tmp_path)
    detail = tmp_path / "backlog" / "candidate_a.md"
    detail.write_text(
        detail.read_text().replace(
            "- 変更するもの: featureを1列追加する",
            "- 変更するもの: {{ CHANGE }}",
        )
    )

    errors = validate_strategy_docs(tmp_path)

    assert any("still contains placeholders" in error for error in errors)


def test_strategy_docs_reject_orphaned_detail(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    (tmp_path / "backlog" / "orphan.md").write_text("# orphan\n")

    errors = validate_strategy_docs(tmp_path)

    assert any("unreferenced backlog detail files: orphan" in error for error in errors)


def test_strategy_docs_reject_state_mismatch(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    detail = tmp_path / "backlog" / "candidate_a.md"
    detail.write_text(detail.read_text().replace("検討メモ・設計不可", "設計可能・実験化未承認"))

    errors = validate_strategy_docs(tmp_path)

    assert any("mismatched candidate state" in error for error in errors)


def test_strategy_docs_reject_noncanonical_priority(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    direction = tmp_path / "backlog/KAGGLE_DIRECTION.md"
    direction.write_text(direction.read_text().replace("| P2 |", "| 中・P2 |"))

    errors = validate_strategy_docs(tmp_path)

    assert any("invalid candidate priority 中・P2" in error for error in errors)


def test_strategy_docs_reject_priority_mismatch(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    detail = tmp_path / "backlog" / "candidate_a.md"
    detail.write_text(detail.read_text().replace("- 優先度: P2", "- 優先度: P3"))

    errors = validate_strategy_docs(tmp_path)

    assert any("mismatched candidate priority" in error for error in errors)


def test_strategy_docs_reject_missing_priority_reason(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    detail = tmp_path / "backlog" / "candidate_a.md"
    detail.write_text(detail.read_text().replace("- 優先度の理由: next candidate\n", ""))

    errors = validate_strategy_docs(tmp_path)

    assert any("has no completed priority reason" in error for error in errors)


def test_strategy_docs_reject_legacy_combined_priority_reason(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    detail = tmp_path / "backlog" / "candidate_a.md"
    detail.write_text(detail.read_text() + "- 優先度と理由: P2・next candidate\n")

    errors = validate_strategy_docs(tmp_path)

    assert any("uses legacy combined priority/reason field" in error for error in errors)


def test_strategy_docs_reject_missing_hypothesis(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    direction = tmp_path / "backlog/KAGGLE_DIRECTION.md"
    direction.write_text(
        direction.read_text().replace(
            "HYP-19000101-91` | hypothesis", "HYP-19000101-92` | hypothesis"
        )
    )

    errors = validate_strategy_docs(tmp_path)

    assert any("references missing hypothesis HYP-19000101-91" in error for error in errors)


def test_strategy_docs_reject_hypothesis_detail_mismatch(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    detail = tmp_path / "backlog" / "candidate_a.md"
    detail.write_text(detail.read_text().replace("HYP-19000101-91", "HYP-19000101-92"))

    errors = validate_strategy_docs(tmp_path)

    assert any("mismatched hypothesis id" in error for error in errors)


def test_strategy_docs_reject_hypothesis_candidate_mapping_mismatch(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    direction = tmp_path / "backlog/KAGGLE_DIRECTION.md"
    direction.write_text(
        direction.read_text().replace(
            "[`candidate_a`](candidate_a.md) | - | remaining",
            "- | - | remaining",
        )
    )

    errors = validate_strategy_docs(tmp_path)

    assert any("candidate mapping does not match backlog rows" in error for error in errors)


def test_strategy_docs_accept_unassigned_legacy_candidate(tmp_path: Path) -> None:
    write_valid_strategy_docs(tmp_path)
    direction = tmp_path / "backlog/KAGGLE_DIRECTION.md"
    text = direction.read_text()
    hypothesis_row = (
        "| `HYP-19000101-91` | hypothesis | "
        "[`candidate_a`](candidate_a.md) | - | remaining |\n"
    )
    text = text.replace(hypothesis_row, "")
    text = text.replace("`HYP-19000101-91` | [`candidate_a`]", "`未整理` | [`candidate_a`]")
    direction.write_text(text)
    detail = tmp_path / "backlog" / "candidate_a.md"
    detail.write_text(detail.read_text().replace("HYP-19000101-91", "未整理"))

    assert validate_strategy_docs(tmp_path) == []

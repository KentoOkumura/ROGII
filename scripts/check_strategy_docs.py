from __future__ import annotations

import re
from pathlib import Path

import yaml

try:
    from .config_utils import ROOT
    from .update_survey_index import EXCLUDED_REPORTS, load_report
except ImportError:  # Direct execution: `uv run python scripts/check_strategy_docs.py`
    from config_utils import ROOT
    from update_survey_index import EXCLUDED_REPORTS, load_report


DIRECTION_PATH = Path("backlog/KAGGLE_DIRECTION.md")
BACKLOG_DIR = Path("backlog")
MAX_DIRECTION_BYTES = 50_000
MAX_DIRECTION_LINES = 220
MAX_LINE_LENGTH = 800
ALLOWED_STATES = {"検討メモ・設計不可", "設計可能・実験化未承認"}
ALLOWED_PRIORITIES = {"P0", "P1", "P2", "P3", "P4"}
DESIGN_READY_STATE = "設計可能・実験化未承認"
UNASSIGNED_HYPOTHESIS = "未整理"
HYPOTHESIS_ID_RE = re.compile(r"HYP-\d{8}-\d{2}")
EXPERIMENT_NAME_RE = re.compile(r"exp\d+_[a-z0-9_]+")
CANDIDATE_NAME_RE = re.compile(r"[a-z0-9_]+")
DETAIL_LINK_RE = re.compile(r"^\[`(?P<name>[a-z0-9_]+)`\]\((?P=name)\.md\)$")
DETAIL_LINK_FIND_RE = re.compile(r"\[`(?P<name>[a-z0-9_]+)`\]\((?P=name)\.md\)")
EXPERIMENT_LINK_RE = re.compile(r"^\[`(?P<name>exp\d+_[a-z0-9_]+)`\]\(\.\./experiments/(?P=name)/\)$")
DESIGN_READY_FIELDS = (
    "親実験 / 比較対象",
    "実測済みの事実",
    "根拠ファイル / 一次資料",
    "上位仮説のうちこの候補が検証する範囲",
    "この候補の具体的な仮説",
    "仮説が正しい場合に期待する観測",
    "仮説を棄却する観測",
    "この候補だけで上位仮説を判断できるか",
    "上位仮説の判断に残る検証",
    "変更するもの",
    "固定するもの",
    "検証方法",
    "primary指標",
    "成功条件",
    "失敗時の停止範囲",
    "禁止する代替実装、proxy、同一OOF上の救済探索",
    "壁打ちで採らなかった案と理由",
)
PLACEHOLDER_RE = re.compile(
    r"(?:TODO|TBD|FIXME)|\{\{[^{}\n]+\}\}",
    flags=re.IGNORECASE,
)


def hypothesis_rows(text: str) -> list[tuple[int, list[str]]]:
    heading = "### 検証中の仮説"
    start = text.find(heading)
    if start < 0:
        return []
    rows: list[tuple[int, list[str]]] = []
    in_table = False
    first_line = text[:start].count("\n") + 1
    for offset, line in enumerate(text[start:].splitlines(), start=first_line):
        if line.startswith("| 仮説ID | 仮説 |"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("### "):
            break
        if line.startswith("| ---"):
            continue
        if not line.startswith("| "):
            if rows:
                break
            continue
        rows.append((offset, line[2:-2].split(" | ")))
    return rows


def backlog_rows(text: str) -> list[tuple[int, list[str]]]:
    heading = "### 未着手バックログ"
    start = text.find(heading)
    if start < 0:
        return []
    rows: list[tuple[int, list[str]]] = []
    in_table = False
    first_line = text[:start].count("\n") + 1
    for offset, line in enumerate(text[start:].splitlines(), start=first_line):
        if line.startswith("| 優先度 | 対応仮説 | アイデア |"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("| ---"):
            continue
        if not line.startswith("| "):
            if rows:
                break
            continue
        cells = line[2:-2].split(" | ")
        rows.append((offset, cells))
    return rows


def experiment_names(cell: str) -> list[str] | None:
    if cell == "-":
        return []
    names: list[str] = []
    for part in cell.split("<br>"):
        match = EXPERIMENT_LINK_RE.fullmatch(part)
        if not match:
            return None
        names.append(match.group("name"))
    return names


def incomplete_table_value(value: str) -> bool:
    return not value.strip() or value.strip() == "-" or bool(PLACEHOLDER_RE.search(value))


def detail_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("- ") or ":" not in line:
            continue
        label, value = line[2:].split(":", maxsplit=1)
        fields[label.strip()] = value.strip()
    return fields


def section_body(text: str, heading: str) -> str | None:
    match = re.search(
        rf"(?ms)^## {re.escape(heading)}\s*\n(?P<body>.*?)(?=^## |\Z)",
        text,
    )
    return match.group("body").strip() if match else None


def validate_design_ready_detail(path: Path, detail: str, hypothesis_id: str) -> list[str]:
    relative_path = path
    errors: list[str] = []
    if hypothesis_id == UNASSIGNED_HYPOTHESIS:
        errors.append(f"{relative_path} is design-ready but has no tracked hypothesis id")

    fields = detail_fields(detail)
    missing = [
        field
        for field in DESIGN_READY_FIELDS
        if not fields.get(field) or fields[field] in {"TODO", "TBD", "FIXME"}
    ]
    if missing:
        errors.append(
            f"{relative_path} is design-ready but missing completed fields: " + ", ".join(missing)
        )
    if PLACEHOLDER_RE.search(detail):
        errors.append(f"{relative_path} is design-ready but still contains placeholders")
    if fields.get("この候補だけで上位仮説を判断できるか") not in {"はい", "いいえ"}:
        errors.append(
            f"{relative_path} must answer whether this candidate alone can decide the "
            "hypothesis with 'はい' or 'いいえ'"
        )
    if section_body(detail, "未決事項") != "- なし":
        errors.append(
            f"{relative_path} is design-ready but unresolved items are not exactly '- なし'"
        )
    return errors


def archived_hypothesis_ids(root: Path) -> set[str]:
    archived: set[str] = set()
    surveys_dir = root / "docs" / "surveys"
    for path in sorted(surveys_dir.glob("*.md")):
        if path.name in EXCLUDED_REPORTS:
            continue
        try:
            report = load_report(path)
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if report.status in {"final", "superseded"}:
            archived.update(report.hypotheses)
    return archived


def registered_hypothesis_ids(root: Path) -> set[str]:
    direction = root / DIRECTION_PATH
    active = set()
    if direction.is_file():
        for _, cells in hypothesis_rows(direction.read_text()):
            if not cells:
                continue
            hypothesis_id = cells[0].strip("`")
            if HYPOTHESIS_ID_RE.fullmatch(hypothesis_id):
                active.add(hypothesis_id)
    return active | archived_hypothesis_ids(root)


def load_experiment_lineage(config_path: Path) -> tuple[object, object]:
    config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    lineage = config.get("lineage") or {}
    if not isinstance(lineage, dict):
        raise ValueError("lineage must be a mapping")
    return lineage.get("hypothesis_id"), lineage.get("backlog_candidate")


def validate_strategy_docs(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    direction = root / DIRECTION_PATH
    if not direction.is_file():
        return [f"missing {DIRECTION_PATH}"]
    text = direction.read_text()
    encoded_size = len(text.encode("utf-8"))
    lines = text.splitlines()
    if encoded_size > MAX_DIRECTION_BYTES:
        errors.append(f"{DIRECTION_PATH} is {encoded_size} bytes; limit is {MAX_DIRECTION_BYTES}")
    if len(lines) > MAX_DIRECTION_LINES:
        errors.append(f"{DIRECTION_PATH} has {len(lines)} lines; limit is {MAX_DIRECTION_LINES}")
    for line_number, line in enumerate(lines, start=1):
        if len(line) > MAX_LINE_LENGTH:
            errors.append(
                f"{DIRECTION_PATH}:{line_number} has {len(line)} characters; "
                f"limit is {MAX_LINE_LENGTH}"
            )

    if "### 検証中の仮説" not in text:
        errors.append(f"{DIRECTION_PATH} has no '検証中の仮説' section")
    if "| 仮説ID | 仮説 | 対応する未着手候補 | 対応する実験 | 残っている問い |" not in text:
        errors.append(f"{DIRECTION_PATH} has no hypothesis index table")
    if "### 未着手バックログ" not in text:
        errors.append(f"{DIRECTION_PATH} has no '未着手バックログ' section")
    if "| 優先度 | 対応仮説 | アイデア | 短い要約 | 主な先行条件 / 依存 | 状態 |" not in text:
        errors.append(f"{DIRECTION_PATH} has no backlog index table")

    rows = backlog_rows(text)

    hypotheses: dict[str, tuple[int, set[str], set[str]]] = {}
    experiment_owners: dict[str, tuple[str, int]] = {}
    for line_number, cells in hypothesis_rows(text):
        if len(cells) != 5:
            errors.append(
                f"{DIRECTION_PATH}:{line_number} hypothesis row has {len(cells)} cells; expected 5"
            )
            continue
        hypothesis_cell, statement, candidate_cell, experiments_cell, remaining = cells
        hypothesis_id = hypothesis_cell.strip("`")
        if not HYPOTHESIS_ID_RE.fullmatch(hypothesis_id) or hypothesis_cell != f"`{hypothesis_id}`":
            errors.append(
                f"{DIRECTION_PATH}:{line_number} has invalid hypothesis id {hypothesis_cell}"
            )
            continue
        if incomplete_table_value(statement):
            errors.append(
                f"{DIRECTION_PATH}:{line_number} hypothesis {hypothesis_id} has an empty "
                "or placeholder statement"
            )
        if incomplete_table_value(remaining):
            errors.append(
                f"{DIRECTION_PATH}:{line_number} hypothesis {hypothesis_id} has an empty "
                "or placeholder remaining question"
            )
        if hypothesis_id in hypotheses:
            errors.append(
                f"{DIRECTION_PATH}:{line_number} duplicates hypothesis {hypothesis_id} "
                f"from line {hypotheses[hypothesis_id][0]}"
            )
            continue
        candidate_names = {
            match.group("name") for match in DETAIL_LINK_FIND_RE.finditer(candidate_cell)
        }
        declared_experiments = experiment_names(experiments_cell)
        if declared_experiments is None:
            errors.append(
                f"{DIRECTION_PATH}:{line_number} hypothesis {hypothesis_id} has invalid "
                "experiment links"
            )
            declared_experiments = []
        for experiment_name in declared_experiments:
            owner = experiment_owners.get(experiment_name)
            if owner is not None:
                errors.append(
                    f"{DIRECTION_PATH}:{line_number} experiment {experiment_name} is already "
                    f"mapped to {owner[0]} on line {owner[1]}"
                )
            else:
                experiment_owners[experiment_name] = (hypothesis_id, line_number)
        hypotheses[hypothesis_id] = (
            line_number,
            candidate_names,
            set(declared_experiments),
        )

    referenced: dict[str, int] = {}
    assigned_candidates: dict[str, set[str]] = {}
    for line_number, cells in rows:
        if len(cells) != 6:
            errors.append(
                f"{DIRECTION_PATH}:{line_number} backlog row has {len(cells)} cells; expected 6"
            )
            continue
        priority, hypothesis_cell, idea_link, _summary, _dependency, state_cell = cells
        if priority not in ALLOWED_PRIORITIES:
            errors.append(
                f"{DIRECTION_PATH}:{line_number} has invalid candidate priority {priority}; "
                "expected one of P0, P1, P2, P3, P4"
            )
            continue
        hypothesis_id = hypothesis_cell.strip("`")
        if hypothesis_cell != f"`{hypothesis_id}`" or (
            hypothesis_id != UNASSIGNED_HYPOTHESIS and not HYPOTHESIS_ID_RE.fullmatch(hypothesis_id)
        ):
            errors.append(
                f"{DIRECTION_PATH}:{line_number} has invalid candidate hypothesis {hypothesis_cell}"
            )
            continue
        if hypothesis_id != UNASSIGNED_HYPOTHESIS and hypothesis_id not in hypotheses:
            errors.append(
                f"{DIRECTION_PATH}:{line_number} references missing hypothesis {hypothesis_id}"
            )
            continue
        match = DETAIL_LINK_RE.fullmatch(idea_link)
        if not match:
            errors.append(f"{DIRECTION_PATH}:{line_number} has an invalid candidate detail link")
            continue
        name = match.group("name")
        if name in referenced:
            errors.append(
                f"{DIRECTION_PATH}:{line_number} duplicates candidate {name} "
                f"from line {referenced[name]}"
            )
        referenced[name] = line_number
        state = state_cell.strip("`")
        if state not in ALLOWED_STATES or state_cell != f"`{state}`":
            errors.append(
                f"{DIRECTION_PATH}:{line_number} has invalid candidate state {state_cell}"
            )
            continue
        detail_path = root / BACKLOG_DIR / f"{name}.md"
        if not detail_path.is_file():
            errors.append(f"{DIRECTION_PATH}:{line_number} missing {detail_path.relative_to(root)}")
            continue
        detail = detail_path.read_text()
        fields = detail_fields(detail)
        if f"- 候補名: `{name}`" not in detail:
            errors.append(f"{detail_path.relative_to(root)} has a mismatched candidate name")
        if f"- 状態: `{state}`" not in detail:
            errors.append(f"{detail_path.relative_to(root)} has a mismatched candidate state")
        if f"- 対応する上位仮説: `{hypothesis_id}`" not in detail:
            errors.append(f"{detail_path.relative_to(root)} has a mismatched hypothesis id")
        if fields.get("優先度") != priority:
            errors.append(f"{detail_path.relative_to(root)} has a mismatched candidate priority")
        if incomplete_table_value(fields.get("優先度の理由", "")):
            errors.append(f"{detail_path.relative_to(root)} has no completed priority reason")
        if "優先度と理由" in fields:
            errors.append(
                f"{detail_path.relative_to(root)} uses legacy combined priority/reason field"
            )
        if state == DESIGN_READY_STATE:
            errors.extend(
                validate_design_ready_detail(
                    detail_path.relative_to(root),
                    detail,
                    hypothesis_id,
                )
            )
        if hypothesis_id != UNASSIGNED_HYPOTHESIS:
            assigned_candidates.setdefault(hypothesis_id, set()).add(name)

    for hypothesis_id, (line_number, declared_candidates, _) in hypotheses.items():
        expected_candidates = assigned_candidates.get(hypothesis_id, set())
        if declared_candidates != expected_candidates:
            errors.append(
                f"{DIRECTION_PATH}:{line_number} hypothesis {hypothesis_id} candidate mapping "
                f"does not match backlog rows"
            )

    configured_experiments: dict[str, set[str]] = {
        hypothesis_id: set() for hypothesis_id in hypotheses
    }
    archived_hypotheses = archived_hypothesis_ids(root)
    for config_path in sorted((root / "experiments").glob("exp*/config.yaml")):
        try:
            hypothesis_id, backlog_candidate = load_experiment_lineage(config_path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            errors.append(f"{config_path.relative_to(root)} cannot be read: {error}")
            continue
        if hypothesis_id in hypotheses:
            experiment_name = config_path.parent.name
            configured_experiments[hypothesis_id].add(experiment_name)
            if not EXPERIMENT_NAME_RE.fullmatch(experiment_name):
                errors.append(f"{config_path.relative_to(root)} has an invalid experiment name")
            if not isinstance(backlog_candidate, str) or not CANDIDATE_NAME_RE.fullmatch(
                backlog_candidate
            ):
                errors.append(
                    f"{config_path.relative_to(root)} must set lineage.backlog_candidate "
                    "to the migrated candidate name"
                )
        elif (
            isinstance(hypothesis_id, str)
            and HYPOTHESIS_ID_RE.fullmatch(hypothesis_id)
            and hypothesis_id not in archived_hypotheses
        ):
            errors.append(
                f"{config_path.relative_to(root)} references unregistered hypothesis "
                f"{hypothesis_id}; register it in the active hypothesis table or a "
                "final/superseded survey"
            )

    for hypothesis_id, (line_number, _, declared_experiments) in hypotheses.items():
        expected_experiments = configured_experiments[hypothesis_id]
        if declared_experiments != expected_experiments:
            errors.append(
                f"{DIRECTION_PATH}:{line_number} hypothesis {hypothesis_id} experiment mapping "
                "does not match experiment config lineage"
            )

    backlog_dir = root / BACKLOG_DIR
    detail_names = {
        path.stem
        for path in backlog_dir.glob("*.md")
        if path.name not in {"_TEMPLATE.md", "KAGGLE_DIRECTION.md", "README.md"}
    }
    orphaned = sorted(detail_names - referenced.keys())
    if orphaned:
        errors.append("unreferenced backlog detail files: " + ", ".join(orphaned))
    return errors


def main() -> None:
    errors = validate_strategy_docs()
    if errors:
        raise SystemExit("\n".join(errors))
    print("Strategy documents passed")


if __name__ == "__main__":
    main()

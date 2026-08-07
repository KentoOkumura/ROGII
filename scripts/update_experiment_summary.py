from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from config_utils import ROOT

SUMMARY_PATH = ROOT / "experiment_summary.md"
EXPERIMENTS_DIR = ROOT / "experiments"
BEGIN_MARKER = "<!-- BEGIN AUTO EXPERIMENT SUMMARY -->"
END_MARKER = "<!-- END AUTO EXPERIMENT SUMMARY -->"
STATUS_LABELS = {
    "planned": "計画中",
    "running": "実行中",
    "usable": "利用可",
    "completed": "完了",
    "failed": "失敗",
    "discarded": "破棄",
    "deprecated": "非推奨",
    "leak-risk": "リーク注意",
    "debug_completed": "デバッグ完了",
    "scaffold_completed": "雛形完了",
}
ROUTE_LABELS = {
    "ensemble": "アンサンブル",
    "ml_model": "MLモデル",
    "pf_beam": "PF/Beam",
}


@dataclass(frozen=True)
class ExperimentRecord:
    name: str
    route: str
    parent: str
    status: str
    cv: str
    public_lb: str
    private_lb: str
    summary: str
    updated: str


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    return value if isinstance(value, dict) else {}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fp:
        value = json.load(fp)
    return value if isinstance(value, dict) else {}


def display_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def display_status(value: Any) -> str:
    status = display_value(value)
    return STATUS_LABELS.get(status, status)


def display_route(value: Any) -> str:
    route = display_value(value)
    return ROUTE_LABELS.get(route, route)


def record_from_experiment(experiment_dir: Path) -> ExperimentRecord | None:
    config = read_yaml(experiment_dir / "config.yaml")
    if not config:
        return None

    metrics = read_json(experiment_dir / "metrics.json")
    experiment = config.get("experiment", {})
    lineage = config.get("lineage", {})

    name = str(experiment.get("name") or experiment_dir.name)
    route = display_route(experiment.get("route") or metrics.get("route"))
    parent = display_value(lineage.get("parent"))
    status = display_status(metrics.get("status") or experiment.get("status"))
    cv = display_value(metrics.get("cv"))
    public_lb = display_value(metrics.get("public_lb"))
    private_lb = display_value(metrics.get("private_lb"))
    summary = display_value(
        metrics.get("key_idea") or experiment.get("description") or lineage.get("diff_summary")
    )
    updated = display_value(
        str(metrics.get("updated_at") or metrics.get("created_at") or "")[:10]
        or experiment.get("created_at")
    )

    return ExperimentRecord(
        name=name,
        route=route,
        parent=parent,
        status=status,
        cv=cv,
        public_lb=public_lb,
        private_lb=private_lb,
        summary=summary,
        updated=updated,
    )


def collect_records() -> list[ExperimentRecord]:
    records: list[ExperimentRecord] = []
    for experiment_dir in sorted(EXPERIMENTS_DIR.glob("exp*")):
        if not experiment_dir.is_dir():
            continue
        record = record_from_experiment(experiment_dir)
        if record is not None:
            records.append(record)
    return records


def render_mermaid(records: list[ExperimentRecord]) -> str:
    lines = ["```mermaid", "graph TD"]
    known_names = {record.name for record in records}

    for record in records:
        label = record.name.replace("-", "_")
        lines.append(f"    {label}[{record.name}]")

    for record in records:
        if record.parent in {"-", "None", "null"}:
            continue
        parent = record.parent.replace("-", "_")
        child = record.name.replace("-", "_")
        if record.parent not in known_names:
            lines.append(f"    {parent}[{record.parent}]")
        lines.append(f"    {parent} --> {child}")

    lines.append("```")
    return "\n".join(lines)


def render_table(records: list[ExperimentRecord]) -> str:
    lines = [
        "| 実験 | ルート | 親 | 状態 | CV | Public LB | Private LB | 要約 | 更新日 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record.name} | {record.route} | {record.parent} | {record.status} | {record.cv} | "
            f"{record.public_lb} | {record.private_lb} | {record.summary} | "
            f"{record.updated} |"
        )
    return "\n".join(lines)


def render_auto_block(records: list[ExperimentRecord]) -> str:
    return "\n\n".join(
        [
            BEGIN_MARKER,
            "## 実験のつながり",
            render_mermaid(records),
            "## スコア表",
            render_table(records),
            END_MARKER,
        ]
    )


def update_summary(existing: str, auto_block: str) -> str:
    if BEGIN_MARKER in existing and END_MARKER in existing:
        before, rest = existing.split(BEGIN_MARKER, 1)
        _, after = rest.split(END_MARKER, 1)
        return before.rstrip() + "\n\n" + auto_block + after

    manual_sections = "## 主な発見\n\n- TODO\n\n## 変更履歴\n\n- TODO\n"
    if "## 主な発見" in existing:
        manual_sections = existing.split("## 主な発見", 1)[1]
        manual_sections = "## 主な発見" + manual_sections
    elif "## Key Findings" in existing:
        manual_sections = existing.split("## Key Findings", 1)[1]
        manual_sections = "## 主な発見" + manual_sections

    return "# 実験サマリー\n\n" + auto_block + "\n\n" + manual_sections


def main() -> None:
    records = collect_records()
    auto_block = render_auto_block(records)
    existing = SUMMARY_PATH.read_text() if SUMMARY_PATH.exists() else ""
    SUMMARY_PATH.write_text(update_summary(existing, auto_block))
    print(f"Updated {SUMMARY_PATH.relative_to(ROOT)} with {len(records)} experiment(s)")


if __name__ == "__main__":
    main()

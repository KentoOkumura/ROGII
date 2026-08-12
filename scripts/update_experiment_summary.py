from __future__ import annotations

import argparse
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
    # New experiments keep status in metrics.json. The config fallback is for
    # existing experiments created before that source-of-truth rule.
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


def render_table(records: list[ExperimentRecord]) -> str:
    lines = [
        "| 実験 | ルート | 親 | 状態 | CV | Public LB | Private LB | 更新日 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record.name} | {record.route} | {record.parent} | {record.status} | {record.cv} | "
            f"{record.public_lb} | {record.private_lb} | {record.updated} |"
        )
    return "\n".join(lines)


def render_auto_block(records: list[ExperimentRecord]) -> str:
    return "\n\n".join(
        [
            BEGIN_MARKER,
            "## 実験比較",
            render_table(records),
            END_MARKER,
        ]
    )


def update_summary(existing: str, auto_block: str) -> str:
    del existing
    return (
        "# 実験サマリー\n\n"
        "このファイルは`task update-summary`で生成します。数値、status、構造化された実行証拠は各実験の`metrics.json`、"
        "証拠への参照、解釈、採否判断は`result.md`、時系列の作業履歴は`SESSION_NOTES.md`を正とします。"
        "手作業の戦略メモや変更履歴はここへ追記しません。\n\n"
        f"{auto_block}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate experiment_summary.md.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail without writing when experiment_summary.md is stale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_records()
    auto_block = render_auto_block(records)
    existing = SUMMARY_PATH.read_text() if SUMMARY_PATH.exists() else ""
    expected = update_summary(existing, auto_block)
    if args.check:
        if existing != expected:
            raise SystemExit("experiment_summary.md is stale; run `task update-summary`")
        print(f"experiment_summary.md is up to date ({len(records)} experiments)")
        return
    SUMMARY_PATH.write_text(expected)
    print(f"Updated {SUMMARY_PATH.relative_to(ROOT)} with {len(records)} experiment(s)")


if __name__ == "__main__":
    main()

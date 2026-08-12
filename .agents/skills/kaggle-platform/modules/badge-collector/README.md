# Badge Collector

Kaggle badge 55件のうち38件について、獲得条件となる操作または手動手順を5 phaseで扱う。
操作の成功とKaggleプロフィール上でのbadge確認を分けて記録する。Kaggle APIとの通信には
`kagglehub`とKaggle CLIを使い、リポジトリ内ではブラウザを自動操作しない。

## Quick Start

```bash
# 1. Verify credentials
uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py --require python-api

# 2. Dry run — see what will happen
uv run python .agents/skills/kaggle-platform/modules/badge-collector/scripts/orchestrator.py --dry-run

# 3. Run Phase 1 prerequisite actions (~16 badge workflows)
uv run python .agents/skills/kaggle-platform/modules/badge-collector/scripts/orchestrator.py --phase 1

# 4. Check progress
uv run python .agents/skills/kaggle-platform/modules/badge-collector/scripts/orchestrator.py --status

# 5. Run all phases
uv run python .agents/skills/kaggle-platform/modules/badge-collector/scripts/orchestrator.py --phase all
```

## Phases

| Phase | Name | Badge workflows | Method |
|-------|------|-----------------|--------|
| 1 | Instant API | 16 | Python API / CLIによる自動操作 |
| 2 | Competition | 7 | CLIによる提出・Notebook操作 |
| 3 | Pipeline | 3 | CLIによるpipeline操作 |
| 4 | Browser | 8 | ユーザーまたは明示的に許可されたhost agentによる手動操作 |
| 5 | Streaks | 4 | 初日の操作とhelper生成。badge確認には7日または30日必要 |

## Prerequisites

- Kaggle credentials configured according to [`../registration/references/kaggle-setup.md`](../registration/references/kaggle-setup.md)
- Phase 1はAPI tokenまたはlegacy username/keyが必要。OAuth-onlyの設定では実行しない
- Phase 2、3、5はKaggle CLIが利用できる認証が必要
- Phase 1–3はresource ownership用に`KAGGLE_USERNAME`を明示する。tokenから推測しない
- `uv sync --locked --extra kaggle-platform`
- Phase 4をhost agentへ依頼する場合、利用する対象と値をユーザーが選び、外部変更を明示的に許可する。このリポジトリへPlaywrightをinstallしない
- For Phase 2: Must accept competition rules at kaggle.com first

Phases 1–3 and 5 create Kaggle resources or submit to competitions. Run them only after the user explicitly requests those external actions. Otherwise use `--dry-run` or `--status` only.

## CLI Options

```
--phase N     Run phase N (1-5) or 'all'; combine only with --dry-run
--status      Show badge progress table
--dry-run     Show planned actions without executing
--mark-action-completed BADGE_ID  Record a completed prerequisite action
--mark-verified BADGE_ID          Record a badge only after checking the Kaggle profile
--details TEXT                    Attach evidence to either manual status update; not valid alone
```

## Resource Naming

All created resources are prefixed with `badge-collector-` and created as **private**.
A 5-second delay between API calls prevents throttling.

## Progress Tracking

Progress, the downloaded Titanic example submission, and the generated daily script are saved under the repository-root `.badge-collector/` directory, which is gitignored. Temporary download directories use the operating system temporary directory rather than the tracked skill tree.

- `action_completed`: 獲得条件となる操作が成功したが、badge表示は未確認
- `manual_required`: ユーザーまたは許可されたhost agentによる操作が必要
- `verified`: Kaggleプロフィール上でbadgeを確認済み

以前の`earned`状態は`verification_required`として読み込み、確認なしに`verified`へ移行しない。

## Scripts

- `scripts/orchestrator.py` — Main entry point
- `scripts/badge_registry.py` — All 55 badge definitions
- `scripts/badge_tracker.py` — JSON progress persistence
- `scripts/phase_1_instant_api.py` — Instant API badges
- `scripts/phase_2_competition.py` — Competition badges
- `scripts/phase_3_pipeline.py` — Pipeline badges (requires KKB)
- `scripts/phase_4_manual.py` — Browserで行う手動操作の案内
- `scripts/phase_5_streaks.py` — Streak automation setup
- `scripts/utils.py` — Shared utilities

## References

- [badge-catalog.md](references/badge-catalog.md) — Complete 55-badge catalog with earning criteria

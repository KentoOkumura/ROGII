#!/usr/bin/env python3
"""Badge Collector orchestrator — main entry point.

Usage (run from the repository root):
    uv run python \
        .agents/skills/kaggle-platform/modules/badge-collector/scripts/orchestrator.py \
        --phase 1
    uv run python \
        .agents/skills/kaggle-platform/modules/badge-collector/scripts/orchestrator.py \
        --status
    uv run python \
        .agents/skills/kaggle-platform/modules/badge-collector/scripts/orchestrator.py \
        --dry-run --phase 2
"""

import argparse
import sys
import traceback
from pathlib import Path

# Add scripts dir to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from badge_registry import get_badge_by_id, get_badges_by_phase
from badge_tracker import load_progress, print_status_table, set_status, should_attempt
from utils import check_credentials, get_username


def dry_run(phases: list[int]) -> None:
    """Show what would be done without executing."""
    print("\n[DRY RUN] Badge workflows that still need attention:\n")
    total = 0
    for phase in phases:
        badges = get_badges_by_phase(phase)
        actionable = [b for b in badges if should_attempt(b.id)]
        if not actionable:
            continue
        print(f"  Phase {phase}: {len(actionable)} badge(s)")
        for badge in actionable:
            print(f"    - {badge.name}: {badge.description}")
            total += 1
    print(f"\n  Total: {total} badge workflow(s) pending\n")


def run_phase(phase: int, username: str) -> tuple[int, int]:
    """Run a single phase and return automatic action attempt/success counts."""
    badges = get_badges_by_phase(phase)
    actionable = [b for b in badges if should_attempt(b.id)]

    if not actionable:
        print(f"\n  Phase {phase}: No prerequisite actions need to be repeated")
        return 0, 0

    print(f"\n{'='*60}")
    print(f"  Phase {phase}: Processing {len(actionable)} badge workflow(s)")
    print(f"{'='*60}\n")

    # Import the phase module (explicit imports for security auditability)
    if phase == 1:
        from phase_1_instant_api import run as phase_run
    elif phase == 2:
        from phase_2_competition import run as phase_run
    elif phase == 3:
        from phase_3_pipeline import run as phase_run
    elif phase == 4:
        from phase_4_manual import run as phase_run
    elif phase == 5:
        from phase_5_streaks import run as phase_run
    else:
        print(f"  Unknown phase: {phase}")
        return 0, 0

    attempted, succeeded = phase_run(username)
    print(f"\n  Phase {phase}: {succeeded}/{attempted} automatic actions completed\n")
    return attempted, succeeded


def update_confirmed_status(badge_id: str, status: str, details: str | None) -> None:
    """Record a user/agent confirmation without claiming an unverified badge."""
    badge = get_badge_by_id(badge_id)
    if badge is None:
        raise ValueError(f"Unknown badge id: {badge_id}")
    set_status(badge_id, status, details)
    print(f"{badge.name}: {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle Badge Collector")
    parser.add_argument("--phase", type=str, default=None,
                        help="Phase to run: 1-5, or 'all'")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true",
                      help="Show badge progress table")
    mode.add_argument("--dry-run", action="store_true",
                      help="Show planned actions without executing")
    mode.add_argument(
        "--mark-action-completed",
        metavar="BADGE_ID",
        help="Record that its prerequisite action completed; this does not verify the badge",
    )
    mode.add_argument(
        "--mark-verified",
        metavar="BADGE_ID",
        help="Record a badge only after it is visibly confirmed on the Kaggle profile",
    )
    parser.add_argument("--details", help="Evidence note for a manual status update")

    args = parser.parse_args()
    manual_update = bool(args.mark_action_completed or args.mark_verified)

    if args.phase is not None and (args.status or manual_update):
        parser.error(
            "--phase cannot be combined with --status or a manual status update"
        )
    if args.details is not None and not manual_update:
        parser.error(
            "--details requires --mark-action-completed or --mark-verified"
        )

    # --status: just show progress
    if args.status:
        print_status_table()
        return

    if args.mark_action_completed:
        update_confirmed_status(args.mark_action_completed, "action_completed", args.details)
        return
    if args.mark_verified:
        update_confirmed_status(args.mark_verified, "verified", args.details)
        return

    # Determine which phases to run. A bare --dry-run should preview the full plan.
    if args.phase is None and args.dry_run:
        phases = [1, 2, 3, 4, 5]
    elif args.phase is None and not args.status:
        parser.print_help()
        return
    elif args.phase == "all":
        phases = [1, 2, 3, 4, 5]
    else:
        try:
            phase = int(args.phase)
        except (ValueError, TypeError):
            parser.error(f"invalid phase {args.phase!r}; use 1-5 or 'all'")
        if phase not in range(1, 6):
            parser.error(f"invalid phase {phase}; use 1-5 or 'all'")
        phases = [phase]

    # --dry-run: show what would be done
    if args.dry_run:
        dry_run(phases)
        return

    # Check credentials
    print("Checking credentials required by the selected phases...")
    if not check_credentials(phases):
        print("\n[ERROR] Kaggle credentials not configured.")
        print("Follow modules/registration/references/kaggle-setup.md")
        sys.exit(1)

    username = get_username()
    if set(phases) & {1, 2, 3} and not username:
        print("\n[ERROR] Could not determine Kaggle username.")
        print("Set KAGGLE_USERNAME for badge resource ownership; do not infer it from a token.")
        sys.exit(1)

    if username:
        print(f"  Username: {username}")
    print(f"  Phases: {phases}")

    # Initialize progress file
    load_progress()

    # Run phases
    total_attempted = 0
    total_succeeded = 0

    for phase in phases:
        try:
            attempted, succeeded = run_phase(phase, username)
            total_attempted += attempted
            total_succeeded += succeeded
        except Exception as e:
            print(f"\n  [ERROR] Phase {phase} failed: {e}")
            traceback.print_exc()
            continue

    # Final summary
    print(f"\n{'='*60}")
    print(f"  AUTOMATIC ACTIONS: {total_succeeded}/{total_attempted} completed")
    print(f"{'='*60}")
    print_status_table()


if __name__ == "__main__":
    main()

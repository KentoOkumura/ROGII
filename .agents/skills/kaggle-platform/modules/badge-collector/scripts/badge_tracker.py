"""Badge action and verification tracker with JSON persistence.

Action completion and badge verification are deliberately separate. A Kaggle
resource or submission can satisfy an earning criterion without proving that
the badge is visible on the user's Kaggle profile.
"""

import json
from datetime import UTC, datetime

from badge_registry import ALL_BADGES
from utils import STATE_DIR

PROGRESS_FILE = STATE_DIR / "badge-progress.json"
VALID_STATUSES = {
    "pending",
    "attempting",
    "action_completed",
    "manual_required",
    "verification_required",
    "verified",
    "failed",
}
LEGACY_STATUS_MAP = {
    "earned": "verification_required",
    "skipped": "manual_required",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def load_progress() -> dict:
    """Load progress, normalizing statuses written by older collector versions."""
    data: dict = {}
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text())

    for info in data.values():
        status = info.get("status", "pending")
        info["status"] = LEGACY_STATUS_MAP.get(status, status)

    # Ensure all badges are tracked.
    for badge in ALL_BADGES:
        if badge.id not in data:
            data[badge.id] = {
                "status": "pending",
                "updated": None,
                "details": None,
            }
    return data


def save_progress(data: dict) -> None:
    """Save progress to disk."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(data, indent=2) + "\n")


def set_status(badge_id: str, status: str, details: str | None = None) -> None:
    """Update a badge's status."""
    if status not in VALID_STATUSES:
        allowed = ", ".join(sorted(VALID_STATUSES))
        raise ValueError(f"Unknown badge status {status!r}; expected one of: {allowed}")
    data = load_progress()
    if badge_id not in data:
        data[badge_id] = {}
    data[badge_id]["status"] = status
    data[badge_id]["updated"] = _now()
    if details:
        data[badge_id]["details"] = details
    save_progress(data)


def get_status(badge_id: str) -> str:
    """Get a badge's current status."""
    data = load_progress()
    return data.get(badge_id, {}).get("status", "pending")


def is_verified(badge_id: str) -> bool:
    """Check whether the badge was confirmed on the user's Kaggle profile."""
    return get_status(badge_id) == "verified"


def should_attempt(badge_id: str) -> bool:
    """Check if the prerequisite action should be attempted or retried."""
    status = get_status(badge_id)
    return status in ("pending", "failed")


def print_status_table() -> None:
    """Print a formatted status table of all badges."""
    data = load_progress()

    # Count by status
    counts: dict[str, int] = {}
    for info in data.values():
        s = info.get("status", "pending")
        counts[s] = counts.get(s, 0) + 1

    total = len(ALL_BADGES)
    verified = counts.get("verified", 0)

    print(f"\n{'='*60}")
    print(f"  Badge Progress: {verified}/{total} verified on Kaggle")
    print(f"{'='*60}")
    print(f"  Verified:              {counts.get('verified', 0)}")
    print(f"  Action completed:      {counts.get('action_completed', 0)}")
    print(f"  Verification required: {counts.get('verification_required', 0)}")
    print(f"  Manual action required: {counts.get('manual_required', 0)}")
    print(f"  Attempting:            {counts.get('attempting', 0)}")
    print(f"  Failed:                {counts.get('failed', 0)}")
    print(f"  Pending:               {counts.get('pending', 0)}")
    print(f"{'='*60}\n")

    # Group by phase
    for phase in [1, 2, 3, 4, 5, None]:
        phase_label = f"Phase {phase}" if phase else "Not Supported"
        phase_badges = [b for b in ALL_BADGES if b.phase == phase]
        if not phase_badges:
            continue

        print(f"  --- {phase_label} ---")
        for badge in phase_badges:
            info = data.get(badge.id, {})
            status = info.get("status", "pending")
            icon = {
                "verified": "[x]",
                "action_completed": "[a]",
                "verification_required": "[?]",
                "manual_required": "[m]",
                "attempting": "[~]",
                "failed": "[!]",
                "pending": "[ ]",
            }.get(status, "[ ]")
            details = info.get("details", "")
            detail_str = f" ({details})" if details else ""
            print(f"    {icon} {badge.name}{detail_str}")
        print()

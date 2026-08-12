"""Phase 4 guided steps for badges that require an authenticated browser.

This module never installs or imports Playwright and never changes a Kaggle
account. A user may perform the steps directly, or explicitly authorize a host
agent with browser tools. Profile text, external links, bookmarks, collections,
and account preferences remain user choices.
"""

from badge_tracker import set_status, should_attempt

MANUAL_STEPS = {
    "stylish": (
        "Stylish",
        "Open Kaggle account settings and choose the bio, location, occupation, "
        "and organization values yourself. A host agent must ask for the exact "
        "values and explicit authorization before editing them.",
    ),
    "vampire": (
        "Vampire",
        "Open Kaggle display settings and enable dark theme. A host agent must "
        "obtain explicit authorization before changing the preference.",
    ),
    "bookmarker": (
        "Bookmarker",
        "Choose a notebook, dataset, or competition and bookmark it in the Kaggle UI.",
    ),
    "collector": (
        "Collector",
        "Choose a Kaggle item and add it to a collection in the Kaggle UI.",
    ),
    "github_coder": (
        "GitHub Coder",
        "Choose a repository and link it to a Kaggle notebook in the Kaggle UI.",
    ),
    "colab_coder": (
        "Colab Coder",
        "Choose a Kaggle notebook and use its Open in Colab action.",
    ),
    "linked_dataset_creator": (
        "Linked Dataset Creator",
        "Choose an external URL and create a URL-linked dataset in the Kaggle UI.",
    ),
    "linked_model_creator": (
        "Linked Model Creator",
        "Choose an external source and create a linked model in the Kaggle UI.",
    ),
}


def run(_username: str) -> tuple[int, int]:
    """Print guided steps and return zero automatic action attempts/successes."""
    for badge_id, (badge_name, instructions) in MANUAL_STEPS.items():
        if not should_attempt(badge_id):
            continue
        print(f"\n  [MANUAL OR HOST AGENT] {badge_name}:")
        print(f"  {instructions}")
        set_status(badge_id, "manual_required", instructions)

    return 0, 0

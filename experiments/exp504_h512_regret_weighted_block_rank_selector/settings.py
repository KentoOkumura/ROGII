"""Execution approval state for the adopted exp504 train notebook."""

EXPERIMENT_NAME = "exp504_h512_regret_weighted_block_rank_selector"
IMPLEMENTATION_ENABLED = True
CANONICAL_NOTEBOOK_ADOPTED = True
KAGGLE_RUN_APPROVED = True


def main() -> None:
    """Expose the approval state without running scientific work from this helper."""
    if not CANONICAL_NOTEBOOK_ADOPTED or not KAGGLE_RUN_APPROVED:
        raise RuntimeError("exp504 canonical train notebook execution is not approved")
    print("exp504 canonical train notebook is adopted and Kaggle execution is approved")


if __name__ == "__main__":
    main()

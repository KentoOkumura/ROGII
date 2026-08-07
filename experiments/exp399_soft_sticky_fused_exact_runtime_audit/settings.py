"""Fail-closed settings entry point for the exp399 notebook experiment."""

EXPERIMENT_NAME = "exp399_soft_sticky_fused_exact_runtime_audit"
EXPERIMENT_STATUS = "technical_preflight_passed_full_oof_not_approved"


def main() -> None:
    raise RuntimeError(
        "exp399 is executed only through its self-contained Kaggle train notebook; "
        "full OOF, inference, and submission remain disabled."
    )


if __name__ == "__main__":
    main()

"""Fail-closed settings entry point for completed exp394.

The compact self-contained train candidate owns notebook-safe path resolution.
The fixed16 preflight failed its runtime gate, so all run stages are disabled.
"""

EXPERIMENT_NAME = "exp394_soft_sticky_exp226_k16_branch_hmm"
EXPERIMENT_STATUS = "technical_preflight_runtime_failed_closed"


def main() -> None:
    raise RuntimeError(
        "exp394 settings.py does not execute the notebook. "
        "The fixed16 runtime gate failed and full OOF is closed."
    )


if __name__ == "__main__":
    main()

# %% [markdown]
# # exp499 exp490 cross-fitted well application selector — inference guard
#
# Inference is intentionally disabled.  A train-side nested selector pass would
# only permit a separately approved inference-port design; it would not make a
# three-well hidden-test router safe automatically.

# %%
from __future__ import annotations

EXPERIMENT_NAME = "exp499_exp490_cross_fitted_well_application_selector"


def main() -> None:
    raise RuntimeError(
        f"{EXPERIMENT_NAME}: inference and submission are disabled until the "
        "technical, predictability, and safe-router gates pass and the user "
        "separately approves an inference port."
    )


# %%
if __name__ == "__main__":
    main()


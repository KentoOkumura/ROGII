# %% [markdown]
# # exp518 — LATE SUBMIT candidate inference is blocked
#
# The corrected training implementation did not reproduce the 3rd-place
# reported OOF RMSE. Per the user's instruction, hidden-test inference and
# submission generation must not run.

# %%
from __future__ import annotations


EXPERIMENT_NAME = "exp518_third_place_absolute_tvt_local_dtw_hmm_late_submit"
INFERENCE_BLOCKED = True
BLOCK_REASON = (
    "Inference is blocked: the 10-well training smoke RMSE was 11.1215, "
    "which did not reproduce the 3rd-place reported OOF RMSE 5.9703. "
    "Run a full OOF reproduction and obtain explicit user confirmation before inference."
)


def run_inference() -> None:
    """Stop before reading hidden-test data or creating a submission."""
    raise RuntimeError(BLOCK_REASON)


# %% [markdown]
# No competition test file is read and no `submission.csv` is created.

# %%
if __name__ == "__main__":
    run_inference()

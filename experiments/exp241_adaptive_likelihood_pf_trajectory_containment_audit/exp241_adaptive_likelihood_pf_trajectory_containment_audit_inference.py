# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp241 adaptive likelihood-PF trajectory containment audit — inference
#
# exp241 は train-side diagnostic 専用であり、raw-test inference と submission を生成しない。

# %%
from settings import get_nested, load_config

config = load_config()
if get_nested(config, "inference.mode") != "disabled_diagnostic_only":
    raise RuntimeError("exp241 inference mode must remain disabled_diagnostic_only")
raise RuntimeError(
    "exp241 is a train-side containment audit. Raw-test inference and submission are forbidden."
)

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp335 signed residual meta on exp264 — Stage D TVT train
#
# corrected exp264のclean 273特徴とsaved compact 74特徴を維持し、Stage Sで生成した
# strict-nested signed-residual 23特徴だけをadd-onlyする。新規学習は1 variant ×
# 3 LightGBM configs × 5 folds = 15 GPU boosters。saved exp264 Stage D v3 OOFを
# controlとして再利用し、controlの再学習は0とする。

# %% [markdown]
# ## Contents
#
# 1. Imports and Kaggle runtime helpers
# 2. Approval and 15-booster compute contract
# 3. Frozen Stage C, Stage S, and saved-control inputs
# 4. Clean273 source and hidden-like contracts
# 5. Stage D execution
# 6. Scientific-support and promotion gates
# 7. Feature-importance readout
# 8. Generated artifacts and reproducibility evidence

# %% [markdown]
# ## 1. Imports and Kaggle runtime helpers

# %%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from IPython.display import display

from src.candidate_selector_pipeline import read_yaml, resolve_existing_path, sha256_file
from src.signed_residual_meta import (
    resolve_saved_exp264_stage_c_root,
    run_stage_d,
    signed_compact_feature_names,
    stage_d_cost_contract,
    verify_saved_exp264_stage_c_root,
    verify_signed_stage_s_root,
)

EXPERIMENT_NAME = "exp335_signed_residual_meta_on_exp264"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


ROOT = find_project_root()


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def require_notebook_runtime() -> None:
    if is_kaggle_runtime() or os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "This Stage D notebook is Kaggle-first; local execution requires explicit approval."
    )


def resolve_config_path() -> Path:
    candidates = [
        Path.cwd() / "config.yaml",
        ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("config.yaml")


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(resolve_config_path().read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def competition_data_root() -> Path:
    local = ROOT / "data" / "raw"
    if not is_kaggle_runtime():
        return local
    project_path = ROOT / "project.yml"
    project = yaml.safe_load(project_path.read_text()) if project_path.exists() else {}
    slug = str((project or {}).get("competition", {}).get("slug", ""))
    candidates = [KAGGLE_INPUT_ROOT / slug]
    if slug:
        candidates.append(KAGGLE_INPUT_ROOT / "competitions" / slug)
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    for candidate in sorted(KAGGLE_INPUT_ROOT.iterdir()):
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError("competition train/test root was not found")


def resolve_pattern_list(patterns: list[str], search_roots: list[Path]) -> Path:
    direct = [Path(pattern) for pattern in patterns if Path(pattern).exists()]
    if direct:
        return sorted(direct)[0]
    return resolve_existing_path(
        [pattern for pattern in patterns if not Path(pattern).is_absolute()], search_roots
    )


def resolve_file_by_sha(
    patterns: list[str], search_roots: list[Path], expected_sha256: str
) -> Path:
    candidates: list[Path] = []
    for pattern in patterns:
        direct = Path(pattern)
        if direct.is_file():
            candidates.append(direct)
        if direct.is_absolute():
            continue
        for root in search_roots:
            if root.exists():
                candidates.extend(path for path in root.glob(pattern) if path.is_file())
    for candidate in dict.fromkeys(candidates):
        if sha256_file(candidate) == str(expected_sha256):
            return candidate
    raise FileNotFoundError(
        f"no artifact matches frozen SHA {expected_sha256}; checked={candidates[:40]}"
    )


def resolve_root_by_marker_sha(
    patterns: list[str], marker: str, expected_sha256: str, search_roots: list[Path]
) -> Path:
    candidates: list[Path] = []
    for pattern in patterns:
        direct = Path(pattern)
        if (direct / marker).is_file():
            candidates.append(direct)
        if direct.is_absolute():
            continue
        for root in search_roots:
            if root.exists():
                candidates.extend(path for path in root.glob(pattern) if path.is_dir())
    for root in search_roots:
        if root.exists():
            candidates.extend(path.parent for path in root.rglob(marker))
    for candidate in dict.fromkeys(candidates):
        marker_path = candidate / marker
        if marker_path.is_file() and sha256_file(marker_path) == str(expected_sha256):
            return candidate
    raise FileNotFoundError(f"no {marker} root matches frozen SHA")


require_notebook_runtime()
config = load_config()
data_root = competition_data_root()
raw_train_dir = data_root / "train"
output_dir = (
    KAGGLE_WORKING_ROOT / "artifacts"
    if is_kaggle_runtime()
    else ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Approval and 15-booster compute contract
#
# このnotebookはStage D専用であり、Stage S、saved compact、saved controlを再学習しない。
# T4 / internet off / double precision / deterministic flags / threads 8をexp264と同じに固定する。

# %%
if str(config["execution"]["stage"]) != "downstream_tvt_train":
    raise RuntimeError("Stage D notebook requires execution.stage=downstream_tvt_train")
if not bool(config["execution"]["downstream_train_approved"]):
    raise RuntimeError("Stage D implementation/train approval is missing")
if not bool(config["execution"]["run_downstream_train"]):
    raise RuntimeError("Stage D run flag is disabled")
if bool(config["execution"]["control_retraining"]):
    raise RuntimeError("saved exp264 control retraining is forbidden")
cost_contract = stage_d_cost_contract(config)
display(
    {
        "experiment": EXPERIMENT_NAME,
        "route": config["experiment"]["route"],
        "stage": config["execution"]["stage"],
        **cost_contract,
        "saved_exp264_control_retraining": 0,
        "inference": False,
        "submission": False,
        "gpu": True,
        "machine_shape": config["runtime"]["kaggle"]["tvt_train"]["machine_shape"],
        "internet": config["runtime"]["kaggle"]["enable_internet"],
    }
)
assert config["experiment"]["route"] == "ml_model"
assert cost_contract["planned_gpu_boosters"] == 15
assert cost_contract["saved_control_retraining_boosters"] == 0
assert config["runtime"]["kaggle"]["tvt_train"]["enable_gpu"] is True
assert config["runtime"]["kaggle"]["enable_internet"] is False
assert config["runtime"]["kaggle"]["gpu_use_dp"] is True
assert config["runtime"]["kaggle"]["deterministic"] is True
assert config["runtime"]["kaggle"]["force_col_wise"] is True
assert config["runtime"]["kaggle"]["num_threads"] == 8

# %% [markdown]
# ## 3. Frozen Stage C, Stage S, and saved-control inputs
#
# corrected Stage C v6の25 saved74 partition、Stage S v3の20 modelと25 signed23 partition、
# corrected Stage D v3のOOF/metricsをSHAで固定する。全partition/model SHA検証はfit直前に再実行する。

# %%
search_roots = [KAGGLE_INPUT_ROOT, KAGGLE_WORKING_ROOT, Path("/tmp"), ROOT]
candidate_contract_path = resolve_pattern_list(
    [str(item) for item in config["data"]["candidate_contract_patterns"]], search_roots
)
candidate_contract = read_yaml(candidate_contract_path)
if sha256_file(candidate_contract_path) != str(config["data"]["candidate_contract_file_sha256"]):
    raise ValueError("candidate contract SHA mismatch")
parent_stage_c_root = resolve_saved_exp264_stage_c_root(config, search_roots)
stage_s_root = resolve_root_by_marker_sha(
    [str(item) for item in config["data"]["stage_s_root_patterns"]],
    "signed_selector_metrics.json",
    str(config["data"]["stage_s_signed_selector_metrics_sha256"]),
    search_roots,
)
saved_parent_oof_path = resolve_file_by_sha(
    [str(item) for item in config["data"]["saved_exp264_stage_d_oof_patterns"]],
    search_roots,
    str(config["data"]["saved_exp264_stage_d_oof_sha256"]),
)
saved_parent_metrics_path = resolve_file_by_sha(
    [str(item) for item in config["data"]["saved_exp264_stage_d_metrics_patterns"]],
    search_roots,
    str(config["data"]["saved_exp264_stage_d_metrics_sha256"]),
)
parent_evidence = verify_saved_exp264_stage_c_root(
    parent_stage_c_root, config, verify_partition_sha=False
)
stage_s_evidence = verify_signed_stage_s_root(
    stage_s_root, config, verify_partition_sha=False, verify_model_sha=False
)
display(
    {
        "candidate_contract": str(candidate_contract_path),
        "candidate_count": len(config["model"]["selector"]["candidate_order"]),
        "saved_stage_c_root": str(parent_stage_c_root),
        "saved_compact_features": parent_evidence["compact_feature_count"],
        "saved_compact_partitions": parent_evidence["partition_count"],
        "stage_s_root": str(stage_s_root),
        "stage_s_models": stage_s_evidence["model_count"],
        "signed_features": stage_s_evidence["feature_count"],
        "signed_partitions": stage_s_evidence["partition_count"],
        "signed_feature_order": signed_compact_feature_names(candidate_contract),
        "saved_control_oof": str(saved_parent_oof_path),
        "saved_control_oof_sha256": sha256_file(saved_parent_oof_path),
        "saved_control_metrics": str(saved_parent_metrics_path),
    }
)

# %% [markdown]
# ## 4. Clean273 source and hidden-like contracts
#
# exp218 source/configから380-feature surfaceを同じ手順で再構築し、
# 監査済みallowlistで273列へ落とす。
# hidden-like assignmentは事後評価専用で、fitやearly stoppingには使わない。

# %%
exp218_source_path = resolve_pattern_list(
    [str(item) for item in config["data"]["exp218_source_patterns"]], search_roots
)
exp218_config_path = resolve_pattern_list(
    [str(item) for item in config["data"]["exp218_config_patterns"]], search_roots
)
clean_allowlist_path = resolve_pattern_list(
    [str(item) for item in config["data"]["clean_273_allowlist_patterns"]], search_roots
)
hidden_like_assignment_path = resolve_pattern_list(
    [str(item) for item in config["data"]["hidden_like_assignment_patterns"]], search_roots
)
if sha256_file(clean_allowlist_path) != str(config["data"]["clean_273_allowlist_sha256"]):
    raise ValueError("clean273 allowlist SHA mismatch")
if sha256_file(hidden_like_assignment_path) != str(
    config["data"]["hidden_like_assignment_sha256"]
):
    raise ValueError("hidden-like assignment SHA mismatch")
display(
    {
        "exp218_source": str(exp218_source_path),
        "exp218_source_sha256": sha256_file(exp218_source_path),
        "exp218_config": str(exp218_config_path),
        "exp218_config_sha256": sha256_file(exp218_config_path),
        "clean273_allowlist": str(clean_allowlist_path),
        "clean273_allowlist_sha256": sha256_file(clean_allowlist_path),
        "hidden_like_assignment": str(hidden_like_assignment_path),
        "hidden_like_assignment_sha256": sha256_file(hidden_like_assignment_path),
        "raw_train_dir": str(raw_train_dir),
    }
)

# %% [markdown]
# ## 5. Stage D execution
#
# fit前にStage C/S全partitionとStage S全20 modelのbyte SHAを検証する。
# その後、370列で15本だけ学習し、3-config meanのOOFを保存する。

# %%
stage_d_summary = run_stage_d(
    config=config,
    contract=candidate_contract,
    parent_stage_c_root=parent_stage_c_root,
    stage_s_root=stage_s_root,
    saved_parent_oof_path=saved_parent_oof_path,
    saved_parent_metrics_path=saved_parent_metrics_path,
    exp218_source_path=exp218_source_path,
    exp218_config_path=exp218_config_path,
    clean_allowlist_path=clean_allowlist_path,
    hidden_like_assignment_path=hidden_like_assignment_path,
    raw_train_dir=raw_train_dir,
    output_dir=output_dir,
)
assert stage_d_summary["model_count"] == 15
assert stage_d_summary["cost_contract"]["saved_control_retraining_boosters"] == 0
display(stage_d_summary)

# %% [markdown]
# ## 6. Scientific-support and promotion gates
#
# pooled `>=0.03 ft`、4/5 folds、near/mid/1000+、hidden-like 2面、by-well p95/worstをAND判定する。
# train-side promotionはさらにclean273比のworst-wellと+1/+3/+5 ft悪化well数の非増加を要求する。

# %%
fold_metrics = pd.read_csv(output_dir / "stage_d_fold_metrics.csv")
bucket_metrics = pd.read_csv(output_dir / "stage_d_bucket_metrics.csv")
hidden_metrics = pd.read_csv(output_dir / "stage_d_hidden_like_metrics.csv")
by_well = pd.read_csv(output_dir / "stage_d_by_well.csv")
display(stage_d_summary["scientific_support_gate"])
display(stage_d_summary["train_side_promotion_gate"])
display(fold_metrics[fold_metrics["model"].eq("lgb_mean")])
display(bucket_metrics)
display(hidden_metrics)
display(by_well.sort_values("new_minus_exp264_delta", ascending=False).head(80))

# %% [markdown]
# ## 7. Feature-importance readout
#
# signed23のgain/splitを全15 modelから保存する。重要度は補助証拠であり、
# RMSE/safety gateを代替しない。

# %%
importance = pd.read_csv(output_dir / "stage_d_feature_importance.csv")
signed_importance = (
    importance[
        importance["feature_group"].eq("signed_residual_compact")
        & importance["importance_type"].eq("gain")
    ]
    .groupby("feature", as_index=False)["importance"]
    .sum()
    .sort_values("importance", ascending=False)
)
display(signed_importance)
if len(signed_importance):
    ax = signed_importance.sort_values("importance").plot.barh(
        x="feature",
        y="importance",
        figsize=(11, 10),
        legend=False,
        title="exp335 Stage D signed-residual feature gain across 15 models",
    )
    ax.set_xlabel("summed gain importance")
    plt.tight_layout()
    plt.savefig(output_dir / "stage_d_signed_feature_importance.png", dpi=140)
    plt.show()

# %% [markdown]
# ## 8. Generated artifacts and reproducibility evidence
#
# model/OOF/metrics SHAを保存する。GPU bitwise再現は主張せず、inference/submissionは生成しない。

# %%
reproducibility = json.loads((output_dir / "reproducibility_manifest.json").read_text())
display(reproducibility)
print("Generated files")
for generated in sorted(output_dir.rglob("*")):
    if generated.is_file():
        print(generated.relative_to(output_dir), generated.stat().st_size)
print("Stage D scientific support passed:", stage_d_summary["scientific_support_gate"]["passed"])
print(
    "Stage D train-side promotion passed:",
    stage_d_summary["train_side_promotion_gate"]["passed"],
)
print("Inference executed: False")
print("Submission generated or submitted: False")

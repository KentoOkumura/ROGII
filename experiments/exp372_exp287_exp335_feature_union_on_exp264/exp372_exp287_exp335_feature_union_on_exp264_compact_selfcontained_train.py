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
# # exp372 exp287 + exp335 feature union on exp264 — train
#
# corrected exp264のclean273 + saved74を固定し、exp287の保存済みfold-safe
# formation74とexp335の保存済みstrict-nested signed23を同時にadd-onlyする。
# feature生成、control、standalone親、selectorは再実行せず、新規学習は
# 1 variant × 3 LightGBM configs × 5 folds = 15 GPU boostersだけとする。
#
# このJupytext sourceは正規train Notebookの編集元である。Kaggle version 1は
# booster開始前のparent compact loader key adapter不足でtechnical errorとなった。
# 現在のsource/pipelineは同adapterを修正済みで、2026-07-25の明示承認により
# version 2を同じ15-booster契約でtechnical retryする。

# %% [markdown]
# ## Contents
#
# 1. Imports and notebook-safe runtime helpers
# 2. Approval and 15-booster compute contract
# 3. Frozen exp264 / exp287 / exp335 input contracts
# 4. Pre-fit 444-feature schema freeze
# 5. Clean273 and saved-control inputs
# 6. Training orchestration
# 7. Fixed scientific gates and feature importance
# 8. Generated artifacts and reproducibility evidence

# %% [markdown]
# ## 1. Imports and notebook-safe runtime helpers

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from IPython.display import display

from src.candidate_selector_pipeline import sha256_file
from src.feature_union_pipeline import (
    freeze_union_feature_schema,
    load_clean_feature_contract,
    run_feature_union_train,
    union_cost_contract,
    verify_formation_root,
    verify_parent_compact_root,
    verify_signed_compact_root,
)

EXPERIMENT_NAME = "exp372_exp287_exp335_feature_union_on_exp264"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    candidates.extend(PACKAGE_DIR.rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    matches = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(matches) != 1:
        raise FileNotFoundError(f"exp372 config resolution is ambiguous: {matches}")
    return matches[0]


def find_competition_input_root() -> Path:
    preferred = [
        KAGGLE_INPUT_ROOT
        / "competitions"
        / "rogii-wellbore-geology-prediction",
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction",
    ]
    candidates = [
        path.resolve()
        for path in preferred
        if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
    ]
    if not candidates and KAGGLE_INPUT_ROOT.exists():
        candidates = [
            path.resolve()
            for path in KAGGLE_INPUT_ROOT.glob("*/*")
            if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
        ]
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"competition input with train/test directories was not unique: {candidates}"
        )
    return candidates[0]


def candidate_paths(patterns: list[str], search_roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for pattern in patterns:
        direct = Path(pattern)
        if direct.exists():
            candidates.append(direct)
        if direct.is_absolute():
            continue
        for root in search_roots:
            if root.exists():
                candidates.extend(root.glob(pattern))
    return list(dict.fromkeys(path.resolve() for path in candidates))


def resolve_file_by_sha(
    patterns: list[str],
    search_roots: list[Path],
    expected_sha256: str,
) -> Path:
    checked: list[str] = []
    for path in candidate_paths(patterns, search_roots):
        if not path.is_file():
            continue
        checked.append(str(path))
        if sha256_file(path) == str(expected_sha256):
            return path
    raise FileNotFoundError(
        f"no file matches frozen SHA {expected_sha256}; checked={checked[:80]}"
    )


def resolve_root_by_marker_sha(
    patterns: list[str],
    marker: str,
    expected_sha256: str,
    search_roots: list[Path],
) -> Path:
    checked: list[str] = []
    candidates = candidate_paths(patterns, search_roots)
    for search_root in search_roots:
        if search_root.exists():
            candidates.extend(path.parent.resolve() for path in search_root.rglob(marker))
    for root in dict.fromkeys(candidates):
        marker_path = root / marker
        if not marker_path.is_file():
            continue
        checked.append(str(root))
        if sha256_file(marker_path) == str(expected_sha256):
            return root
    raise FileNotFoundError(
        f"no {marker} root matches frozen SHA; checked={checked[:80]}"
    )


if not KAGGLE_INPUT_ROOT.exists() or not KAGGLE_WORKING_ROOT.exists():
    raise RuntimeError("Kaggle Notebook execution is authoritative for exp372")

config = read_yaml(find_config_path())
competition_root = find_competition_input_root()
raw_train_dir = competition_root / "train"
output_dir = KAGGLE_WORKING_ROOT / "artifacts"
output_dir.mkdir(parents=True, exist_ok=True)
search_roots = [KAGGLE_INPUT_ROOT, KAGGLE_WORKING_ROOT, Path("/tmp"), PACKAGE_DIR]

# %% [markdown]
# ## 2. Approval and 15-booster compute contract
#
# 実装承認とrun承認を分離する。現在のroot configは実装済み・run無効であるため、
# package/push前に別承認を受けた版だけがこのセルを通る。

# %%
if not bool(config["execution"]["implementation_approved"]):
    raise RuntimeError("exp372 implementation approval is missing")
if not bool(config["execution"]["train_run_approved"]):
    raise RuntimeError("exp372 train run approval is missing")
if not bool(config["execution"]["run_train"]):
    raise RuntimeError("exp372 run_train is disabled")
if bool(config["execution"]["inference_approved"]):
    raise RuntimeError("exp372 train candidate must not enable inference")
if bool(config["execution"]["submission_approved"]):
    raise RuntimeError("exp372 train candidate must not enable submission")

cost_contract = union_cost_contract(config)
display(
    {
        "experiment": EXPERIMENT_NAME,
        "route": config["experiment"]["route"],
        "stage": config["execution"]["stage"],
        **cost_contract,
        "saved_control_retraining": 0,
        "standalone_parent_retraining": 0,
        "selector_retraining": 0,
        "formation_or_signed_generation": 0,
        "inference": False,
        "submission": False,
        "gpu": config["runtime"]["kaggle"]["enable_gpu"],
        "machine_shape": config["runtime"]["kaggle"]["machine_shape"],
        "internet": config["runtime"]["kaggle"]["enable_internet"],
    }
)
assert cost_contract["planned_gpu_boosters"] == 15
assert cost_contract["parent_control_retraining_boosters"] == 0
assert cost_contract["standalone_parent_retraining_boosters"] == 0
assert config["runtime"]["kaggle"]["enable_gpu"] is True
assert config["runtime"]["kaggle"]["enable_internet"] is False
assert config["runtime"]["kaggle"]["gpu_use_dp"] is True
assert config["runtime"]["kaggle"]["deterministic"] is True
assert config["runtime"]["kaggle"]["force_col_wise"] is True
assert config["runtime"]["kaggle"]["num_threads"] == 8

# %% [markdown]
# ## 3. Frozen exp264 / exp287 / exp335 input contracts
#
# 3 manifest、schema、relationship audit、全partitionの存在をfit前に確認する。
# `run_feature_union_train`はこの後byte SHAとformation logical float32 SHAを再検証する。

# %%
data_config = config["data"]
parent_root = resolve_root_by_marker_sha(
    [str(value) for value in data_config["exp264_stage_c_root_patterns"]],
    "nested_compact_manifest.json",
    str(data_config["exp264_nested_compact_manifest_sha256"]),
    search_roots,
)
formation_root = resolve_root_by_marker_sha(
    [str(value) for value in data_config["exp287_formation_root_patterns"]],
    str(data_config["exp287_formation_manifest"]),
    str(data_config["exp287_formation_manifest_sha256"]),
    search_roots,
)
signed_root = resolve_root_by_marker_sha(
    [str(value) for value in data_config["exp335_stage_s_root_patterns"]],
    str(data_config["exp335_signed_compact_manifest"]),
    str(data_config["exp335_signed_compact_manifest_sha256"]),
    search_roots,
)
signed_schema_path = resolve_file_by_sha(
    [str(value) for value in data_config["exp335_signed_compact_schema_patterns"]],
    search_roots,
    str(data_config["exp335_signed_compact_schema_file_sha256"]),
)
parent_evidence = verify_parent_compact_root(
    parent_root, config, verify_partition_sha=False
)
signed_evidence = verify_signed_compact_root(
    signed_root,
    signed_schema_path,
    config,
    parent_evidence=parent_evidence,
    verify_partition_sha=False,
)
formation_evidence = verify_formation_root(
    formation_root,
    config,
    verify_partition_sha=False,
    verify_logical_content_sha=False,
)
display(
    {
        "exp264_root": str(parent_root),
        "exp264_manifest_sha256": parent_evidence["file_sha256"]["manifest"],
        "exp264_partitions": parent_evidence["partition_count"],
        "exp287_root": str(formation_root),
        "exp287_manifest_sha256": formation_evidence["manifest_sha256"],
        "exp287_partitions": formation_evidence["partition_count"],
        "exp335_root": str(signed_root),
        "exp335_manifest_sha256": signed_evidence["file_sha256"]["manifest"],
        "exp335_partitions": signed_evidence["partition_count"],
        "saved_feature_generation_runs": 0,
    }
)

# %% [markdown]
# ## 4. Pre-fit 444-feature schema freeze
#
# allowlistと保存schemaだけから
# `clean273 -> saved74 -> formation74 -> signed23`を凍結する。
# この時点ではtarget/error/OOF truthを開かない。

# %%
clean_allowlist_path = resolve_file_by_sha(
    [str(value) for value in data_config["clean_273_allowlist_patterns"]],
    search_roots,
    str(data_config["clean_273_allowlist_sha256"]),
)
clean_features, clean_contract = load_clean_feature_contract(
    clean_allowlist_path,
    expected_sha256=str(data_config["clean_273_allowlist_sha256"]),
)
final_features, feature_contract = freeze_union_feature_schema(
    clean_features=clean_features,
    parent_features=parent_evidence["features"],
    formation_features=formation_evidence["features"],
    signed_features=signed_evidence["features"],
    forbidden_columns=config["features"]["forbidden_columns"],
)
display(
    {
        "clean_allowlist": clean_contract,
        "frozen_order": feature_contract["frozen_order"],
        "feature_counts": feature_contract["feature_counts"],
        "final_feature_count": len(final_features),
        "feature_schema_sha256": feature_contract["feature_schema_sha256"],
        "truth_or_error_loaded_before_schema_freeze": 0,
    }
)
display(
    pd.DataFrame(
        {
            "position": range(len(final_features)),
            "feature": final_features,
            "group": [
                group
                for group, features in feature_contract["feature_groups"].items()
                for _ in features
            ],
        }
    )
)

# %% [markdown]
# ## 5. Clean273 and saved-control inputs
#
# exp218 source/configからclean273を親と同じ手順で組み立てる。
# exp264 / exp287 / exp335 OOFは比較専用で、444特徴には含めない。

# %%
exp218_source_path = resolve_file_by_sha(
    [str(value) for value in data_config["exp218_source_patterns"]],
    search_roots,
    str(data_config["exp218_source_sha256"]),
)
exp218_config_path = resolve_file_by_sha(
    [str(value) for value in data_config["exp218_config_patterns"]],
    search_roots,
    str(data_config["exp218_config_sha256"]),
)
hidden_like_assignment_path = resolve_file_by_sha(
    [str(value) for value in data_config["hidden_like_assignment_patterns"]],
    search_roots,
    str(data_config["hidden_like_assignment_sha256"]),
)
control_paths = {
    "exp264": resolve_file_by_sha(
        [str(value) for value in data_config["exp264_oof_patterns"]],
        search_roots,
        str(config["validation"]["saved_controls"]["exp264"]["oof_sha256"]),
    ),
    "exp287": resolve_file_by_sha(
        [str(value) for value in data_config["exp287_oof_patterns"]],
        search_roots,
        str(config["validation"]["saved_controls"]["exp287"]["oof_sha256"]),
    ),
    "exp335": resolve_file_by_sha(
        [str(value) for value in data_config["exp335_oof_patterns"]],
        search_roots,
        str(config["validation"]["saved_controls"]["exp335"]["oof_sha256"]),
    ),
}
display(
    {
        "exp218_source": str(exp218_source_path),
        "exp218_config": str(exp218_config_path),
        "raw_train_dir": str(raw_train_dir),
        "clean_allowlist": str(clean_allowlist_path),
        "hidden_like_assignment": str(hidden_like_assignment_path),
        "saved_control_oof": {
            name: str(path) for name, path in control_paths.items()
        },
        "saved_oof_used_as_features": False,
    }
)

# %% [markdown]
# ## 6. Training orchestration
#
# fit直前に全partition SHAとformation logical SHAを検証する。foldごとにだけ
# 444列matrixを保持し、15 model slotを埋めた後にOOFを保存する。

# %%
run_summary = run_feature_union_train(
    config=config,
    parent_root=parent_root,
    formation_root=formation_root,
    signed_root=signed_root,
    signed_schema_path=signed_schema_path,
    exp218_source_path=exp218_source_path,
    exp218_config_path=exp218_config_path,
    clean_allowlist_path=clean_allowlist_path,
    hidden_like_assignment_path=hidden_like_assignment_path,
    raw_train_dir=raw_train_dir,
    control_paths=control_paths,
    output_dir=output_dir,
)
assert run_summary["model_count"] == 15
assert run_summary["cost_contract"]["parent_control_retraining_boosters"] == 0
display(run_summary)

# %% [markdown]
# ## 7. Fixed scientific gates and feature importance
#
# technical / incremental utility / tail promotionを別々に表示し、全ANDのみをpromoteする。
# formation/signed重要度は補助証拠であり、RMSE/tail gateを上書きしない。

# %%
fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
bucket_metrics = pd.read_csv(output_dir / "bucket_metrics.csv")
hidden_metrics = pd.read_csv(output_dir / "hidden_like_metrics.csv")
by_well = pd.read_csv(output_dir / "by_well_metrics.csv")
importance = pd.read_csv(output_dir / "feature_importance.csv")
relationships = pd.read_csv(output_dir / "feature_relationship_audit.csv")
display(run_summary["technical_gate"])
display(run_summary["incremental_utility_gate"])
display(run_summary["tail_promotion_gate"])
display(run_summary["promotion_gate"])
display(fold_metrics[fold_metrics["model"].eq("lgb_mean")])
display(bucket_metrics)
display(hidden_metrics)
display(by_well.sort_values("union_minus_exp264_delta", ascending=False).head(80))
display(
    relationships.sort_values(
        ["exact_duplicate_count", "max_abs_pearson"],
        ascending=[False, False],
    ).head(100)
)

added_importance = (
    importance[
        importance["importance_type"].eq("gain")
        & importance["feature_group"].isin(
            ["fold_safe_formation", "signed_residual_compact"]
        )
    ]
    .groupby(["feature_group", "feature"], as_index=False)["importance"]
    .sum()
    .sort_values("importance", ascending=False)
)
display(added_importance)
for group in ["fold_safe_formation", "signed_residual_compact"]:
    selected = added_importance[added_importance["feature_group"].eq(group)].head(30)
    if len(selected):
        axis = selected.sort_values("importance").plot.barh(
            x="feature",
            y="importance",
            figsize=(11, 10),
            legend=False,
            title=f"exp372 {group} summed gain across 15 models",
        )
        axis.set_xlabel("summed gain importance")
        plt.tight_layout()
        plt.savefig(output_dir / f"{group}_feature_importance_top30.png", dpi=140)
        plt.show()

# %% [markdown]
# ## 8. Generated artifacts and reproducibility evidence
#
# model/OOF/feature schema/入力SHAを保存する。GPU bitwise再現は主張せず、
# inference/submissionは生成しない。

# %%
reproducibility = json.loads((output_dir / "reproducibility_manifest.json").read_text())
display(reproducibility)
print("Generated files")
for generated in sorted(output_dir.rglob("*")):
    if generated.is_file():
        print(generated.relative_to(output_dir), generated.stat().st_size)
print("Technical gate passed:", run_summary["technical_gate"]["passed"])
print(
    "Incremental utility gate passed:",
    run_summary["incremental_utility_gate"]["passed"],
)
print("Tail promotion gate passed:", run_summary["tail_promotion_gate"]["passed"])
print("Overall promotion passed:", run_summary["promotion_gate"]["passed"])
print("Inference executed: False")
print("Submission generated or submitted: False")

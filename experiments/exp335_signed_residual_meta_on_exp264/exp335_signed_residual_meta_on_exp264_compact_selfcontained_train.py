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
# # exp335 signed residual meta on exp264 — Stage S train
#
# corrected exp264 Stage C v6の88 selector特徴、12候補、outer 5 × inner 4 split、
# 保存済みcompact 74列を固定し、`true_tvt - candidate_tvt`だけを学ぶL2 headを追加する。
# Stage Sは20 CPU boostersを学習し、候補別12列・既存top-1注釈8列・分布3列の
# signed compact 23列をstrict nestedで生成する。保存済みexp264 selector/controlは再学習しない。
#
# このsourceは正規train notebookの編集元である。Kaggle package、preflight実行、
# 20-booster学習はいずれもconfig上の個別承認flagがない限り開始しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and notebook-safe runtime helpers
# 2. Configuration, stage, and compute contract
# 3. Frozen input, candidate, and feature contract
# 4. Zero-booster preflight contract
# 5. Strict-nested signed-residual execution
# 6. Metrics and feature-importance readout
# 7. Generated artifacts and next gate

# %% [markdown]
# ## 1. Imports and notebook-safe runtime helpers

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

from src.candidate_selector_pipeline import (
    audit_raw_context_availability,
    read_yaml,
    resolve_existing_path,
    resolve_exp263_cache_root,
    sha256_file,
    verify_exp263_root,
)
from src.signed_residual_meta import (
    resolve_saved_exp264_stage_c_root,
    run_stage_s,
    run_stage_s_preflight,
    signed_compact_feature_names,
    stage_s_cost_contract,
    verify_saved_exp264_stage_c_root,
    verify_stage_a_feature_contract,
)

EXPERIMENT_NAME = "exp335_signed_residual_meta_on_exp264"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else start
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
        "This notebook is Kaggle-first. Prepare and run the Kaggle notebook after approval; "
        "set EXPERIMENT_ALLOW_LOCAL=1 only for an explicitly approved local smoke run."
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
    matches = sorted(Path.cwd().rglob("config.yaml"))
    if not matches:
        raise FileNotFoundError("config.yaml")
    return matches[0]


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(resolve_config_path().read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def resolve_pattern_list(patterns: list[str], search_roots: list[Path]) -> Path:
    direct = [Path(pattern) for pattern in patterns if Path(pattern).exists()]
    if direct:
        return sorted(direct)[0]
    relative_patterns = [pattern for pattern in patterns if not Path(pattern).is_absolute()]
    return resolve_existing_path(relative_patterns, search_roots)


def competition_data_root() -> Path:
    local = ROOT / "data" / "raw"
    if not is_kaggle_runtime():
        return local
    project_path = ROOT / "project.yml"
    project = yaml.safe_load(project_path.read_text()) if project_path.exists() else {}
    slug = str((project or {}).get("competition", {}).get("slug", ""))
    configured_candidates = [KAGGLE_INPUT_ROOT / slug]
    if slug:
        configured_candidates.append(KAGGLE_INPUT_ROOT / "competitions" / slug)
    for candidate in configured_candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    for candidate in sorted(KAGGLE_INPUT_ROOT.iterdir()):
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    for sample_submission in sorted(KAGGLE_INPUT_ROOT.rglob("sample_submission.csv")):
        candidate = sample_submission.parent
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError("competition train/test input root was not found")


require_notebook_runtime()
config = load_config()
data_root = competition_data_root()
raw_train_dir = data_root / "train"
raw_test_dir = data_root / "test"
output_dir = (
    KAGGLE_WORKING_ROOT / "artifacts"
    if is_kaggle_runtime()
    else ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Configuration, stage, and compute contract
#
# `signed_selector_preflight`は0 boosterで、exp263 cache、corrected Stage Aの88特徴、
# corrected Stage C v6のmanifest/schema/25 partition SHAを検証する。
# `signed_selector_train`は1 objective × 5 outer × 4 inner = 20 CPU boostersだけを学習する。
# Stage D、saved control、既存2 objective、PF/HMM/Beam候補は再学習しない。

# %%
stage = str(config["execution"]["stage"])
allowed_stages = {str(item) for item in config["execution"]["allowed_stages"]}
if stage not in allowed_stages:
    raise ValueError(f"unknown exp335 execution stage: {stage}")
if not bool(config["execution"]["implementation_complete"]):
    raise RuntimeError("Stage S implementation_complete must be true")
if stage == "signed_selector_preflight":
    if not bool(config["execution"]["preflight_run_approved"]):
        raise RuntimeError(
            "signed_selector_preflight requires execution.preflight_run_approved=true"
        )
elif not (
    bool(config["execution"]["selector_train_approved"])
    and bool(config["execution"]["run_selector_train"])
):
    raise RuntimeError(
        "signed_selector_train requires selector_train_approved=true and "
        "run_selector_train=true after the 20-booster CPU cost is approved"
    )

cost_contract = stage_s_cost_contract(config)
display(
    {
        "experiment": EXPERIMENT_NAME,
        "route": config["experiment"]["route"],
        "stage": stage,
        **cost_contract,
        "effective_boosters_this_stage": (
            0 if stage == "signed_selector_preflight" else cost_contract["planned_cpu_boosters"]
        ),
        "saved_selector_retraining": 0,
        "saved_control_retraining": 0,
        "downstream_gpu_boosters": 0,
        "gpu": config["runtime"]["kaggle"]["enable_gpu"],
        "internet": config["runtime"]["kaggle"]["enable_internet"],
    }
)
assert config["experiment"]["route"] == "ml_model"
assert config["model"]["selector"]["objective"] == "regression_l2"
assert config["model"]["selector"]["target_formula"] == "true_tvt-candidate_tvt"
assert config["model"]["signed_compact"]["feature_count"] == 23
assert config["model"]["downstream_tvt"]["execution_count"]["control_retraining_boosters"] == 0
assert config["runtime"]["kaggle"]["enable_gpu"] is False
assert config["runtime"]["kaggle"]["enable_internet"] is False

print("Leakage contract")
for rule in config["validation"]["leakage_policy"]:
    print("-", rule)

# %% [markdown]
# ## 3. Frozen input, candidate, and feature contract
#
# candidate順と88-feature schemaはcorrected exp264 Stage A v4、既存top-1 scoreと74列は
# corrected Stage C v6を正とする。親compactは置換・再計算せず、各partitionをbyte SHAで固定する。

# %%
search_roots = [KAGGLE_INPUT_ROOT, KAGGLE_WORKING_ROOT, Path("/tmp"), ROOT]
candidate_contract_path = resolve_pattern_list(
    [str(item) for item in config["data"]["candidate_contract_patterns"]], search_roots
)
feature_schema_path = resolve_pattern_list(
    [str(item) for item in config["data"]["corrected_stage_a_feature_schema_patterns"]],
    search_roots,
)
feature_catalog_path = resolve_pattern_list(
    [str(item) for item in config["data"]["corrected_stage_a_feature_catalog_patterns"]],
    search_roots,
)
candidate_contract = read_yaml(candidate_contract_path)
if sha256_file(candidate_contract_path) != str(config["data"]["candidate_contract_file_sha256"]):
    raise ValueError("exp264 candidate contract file SHA mismatch")
cache_root = resolve_exp263_cache_root(config, search_roots)
parent_stage_c_root = resolve_saved_exp264_stage_c_root(config, search_roots)

cache_evidence = verify_exp263_root(cache_root, config)
feature_evidence = verify_stage_a_feature_contract(
    feature_schema_path, feature_catalog_path, config
)
parent_evidence = verify_saved_exp264_stage_c_root(
    parent_stage_c_root,
    config,
    verify_partition_sha=False,
)
raw_context_audit = audit_raw_context_availability(
    raw_train_dir,
    raw_test_dir,
    config["features"]["raw_context"]["horizontal_numeric_allowlist"],
)
display(
    {
        "candidate_contract": str(candidate_contract_path),
        "candidate_contract_sha256": sha256_file(candidate_contract_path),
        "candidate_order": config["model"]["selector"]["candidate_order"],
        "exp263_cache": cache_evidence,
        "feature_schema": str(feature_schema_path),
        "feature_count": feature_evidence["feature_count"],
        "feature_schema_sha256": feature_evidence["feature_schema_logical_sha256"],
        "saved_exp264_stage_c_root": str(parent_stage_c_root),
        "saved_compact_features": parent_evidence["compact_feature_count"],
        "saved_compact_partitions": parent_evidence["partition_count"],
        "new_signed_features": signed_compact_feature_names(candidate_contract),
    }
)
display(raw_context_audit)

# %% [markdown]
# ## 4. Zero-booster preflight contract
#
# preflightは全25 saved compact partitionのbyte SHAをfit前に検証し、
# 23列schemaと入力manifestを保存する。
# train stageも同じpreflightを先頭で再実行するため、未検証入力で1本目のLightGBM fitへ進まない。

# %%
stage_summary = None
if stage == "signed_selector_preflight":
    stage_summary = run_stage_s_preflight(
        config=config,
        contract=candidate_contract,
        cache_root=cache_root,
        parent_stage_c_root=parent_stage_c_root,
        feature_schema_path=feature_schema_path,
        feature_catalog_path=feature_catalog_path,
        output_dir=output_dir,
        verify_parent_partition_sha=True,
    )
    assert stage_summary["models_trained"] == 0
    assert stage_summary["passed"] is True
    display(stage_summary)
else:
    print("The train helper performs the same full-SHA preflight before fitting model 1/20.")

# %% [markdown]
# ## 5. Strict-nested signed-residual execution
#
# outer-train rowsにはinner OOF predictionを、outer-valid rowsにはouter-train内4 model平均を使う。
# signed label、true TVT、actual error/rank/oracleは監査scoreにだけ保存し、
# 23列feature partitionへは渡さない。

# %%
if stage == "signed_selector_train":
    stage_summary = run_stage_s(
        config=config,
        contract=candidate_contract,
        cache_root=cache_root,
        parent_stage_c_root=parent_stage_c_root,
        feature_schema_path=feature_schema_path,
        feature_catalog_path=feature_catalog_path,
        raw_train_dir=raw_train_dir,
        output_dir=output_dir,
    )
    assert stage_summary["model_count"] == 20
    assert stage_summary["technical_gate"]["passed"] is True
    display(stage_summary)
else:
    print("Preflight only: 0 selector models and 0 downstream models were trained.")

# %% [markdown]
# ## 6. Metrics and feature-importance readout
#
# Stage S gateはcandidate別outer-train mean signed residual priorに対し、pooled RMSEと
# 4/5 outer folds以上の両方で改善した場合だけPASSする。FAIL時はStage Dへ進まない。

# %%
if stage == "signed_selector_train":
    metrics = pd.read_csv(output_dir / "signed_selector_metrics.csv")
    candidate_metrics = pd.read_csv(output_dir / "signed_selector_candidate_metrics.csv")
    importance = pd.read_csv(output_dir / "signed_feature_importance_by_outer_inner.csv")
    display(metrics)
    display(
        candidate_metrics.groupby("candidate_id", as_index=False)[
            ["signed_residual_rmse", "prior_signed_residual_rmse", "sign_accuracy"]
        ].mean()
    )
    gain = importance[importance["importance_type"].eq("gain")].copy()
    gain_mean = (
        gain.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    display(gain_mean.head(30))
    if len(gain_mean):
        ax = gain_mean.head(30).sort_values("importance").plot.barh(
            x="feature",
            y="importance",
            figsize=(10, 10),
            legend=False,
            title="exp335 signed residual selector mean gain importance",
        )
        ax.set_xlabel("mean gain")
        plt.tight_layout()
        plt.savefig(output_dir / "signed_feature_importance_top30.png", dpi=140)
        plt.show()
else:
    display(pd.DataFrame([stage_summary["cost_contract"]]))

# %% [markdown]
# ## 7. Generated artifacts and next gate

# %%
generated = sorted(
    [
        {
        "path": str(path.relative_to(output_dir)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        }
        for path in output_dir.rglob("*")
        if path.is_file()
    ],
    key=lambda item: item["path"],
)
display(pd.DataFrame(generated))
print(json.dumps(stage_summary, indent=2, ensure_ascii=False))
if stage == "signed_selector_train" and stage_summary["stage_s_gate_passed"]:
    print(
        "Stage S PASS. Stop here and request separate approval before implementing or running "
        "the 15-GPU-booster Stage D downstream comparison."
    )
elif stage == "signed_selector_train":
    print("Stage S FAIL. Do not run Stage D and do not rescue this experiment with a target grid.")
else:
    print(
        "Preflight complete. Stop here; 20-booster Stage S training still needs "
        "separate approval."
    )

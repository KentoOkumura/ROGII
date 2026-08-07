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
# # exp413 scale-5 likPF full replacement on exp335 — train
#
# exp335の固定12候補にある`likpf_mean` semantic slotを、SHA固定済みexp404
# `likpf_scale_5_x1p0`へ同一IDのまま全面置換する。5 candidate slotを再計算し、
# 7 slotを固定する。replacement sourceからselector88、nested compact74、
# signed compact23、clean273、final370を依存順に再構築する。
#
# 新規学習はStage C 40 CPU、Stage S 20 CPU、Stage D 15 GPUの合計75 boosters。
# saved exp335 controlは再学習しない。各stageはconfigの個別承認flagとrun flagが
# 同時にtrueでなければ開始しない。推論・submissionはこのnotebookでは実装しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and notebook-safe runtime helpers
# 2. Frozen experiment, authorization, and compute contract
# 3. Input paths and SHA contracts
# 4. Stage 0 replacement cache and transitive-lineage preflight
# 5. Stage C strict-nested dual selector
# 6. Stage S strict-nested signed-residual selector
# 7. Stage D clean273 + compact74 + signed23 downstream
# 8. Metrics, feature importance, and generated artifacts
# 9. Reproducibility evidence and fixed stop

# %% [markdown]
# ## 1. Imports and notebook-safe runtime helpers

# %%
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from IPython.display import display

from src.candidate_selector_pipeline import (
    resolve_exp263_cache_root,
    run_stage_c,
    sha256_file,
    write_json,
)
from src.likpf_full_replacement import (
    downstream_runtime_config,
    replacement_cache_factory,
    replacement_cost_contract,
    require_stage_authorization,
    resolve_by_patterns,
    run_replacement_preflight,
    run_replacement_stage_d,
    stage_c_runtime_config,
    stage_s_runtime_config,
    validate_replacement_contract,
    verify_replacement_stage_0_root,
    verify_replacement_stage_c_root,
    verify_replacement_stage_s_root,
)
from src.signed_residual_meta import run_stage_s

EXPERIMENT_NAME = "exp413_scale5_likpf_full_replacement_on_exp335"
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
        "exp413 is Kaggle-first. Local execution requires an explicitly approved "
        "EXPERIMENT_ALLOW_LOCAL=1 smoke run."
    )


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def resolve_config_path() -> Path:
    candidates = [
        Path.cwd() / "config.yaml",
        ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("exp413 config.yaml")


def search_roots() -> list[Path]:
    return [KAGGLE_INPUT_ROOT, KAGGLE_WORKING_ROOT, Path("/tmp"), ROOT, Path.cwd()]


def resolve_file(spec: dict[str, Any], *, sha_key: str = "sha256") -> Path:
    path = resolve_by_patterns(
        [str(item) for item in spec["patterns"]],
        search_roots(),
        marker_sha256=str(spec.get(sha_key) or ""),
    )
    return path


def resolve_root(patterns: list[str], marker: str) -> Path:
    candidates: list[Path] = []
    for raw in patterns:
        direct = Path(raw)
        if (direct / marker).exists():
            candidates.append(direct)
        if direct.is_absolute():
            continue
        for root in search_roots():
            if root.exists():
                candidates.extend(
                    item for item in root.glob(raw) if (item / marker).exists()
                )
    for root in search_roots():
        if root.exists():
            candidates.extend(path.parent for path in root.rglob(marker))
    if not candidates:
        raise FileNotFoundError(f"no artifact root contains {marker}")
    return sorted(set(candidates))[0]


def competition_data_root() -> Path:
    local = ROOT / "data" / "raw"
    if not is_kaggle_runtime():
        return local
    project_path = ROOT / "project.yml"
    project = load_yaml(project_path) if project_path.exists() else {}
    slug = str(project.get("competition", {}).get("slug", ""))
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


require_notebook_runtime()
config = load_yaml(resolve_config_path())
data_root = competition_data_root()
raw_train_dir = data_root / "train"
output_dir = (
    KAGGLE_WORKING_ROOT / "artifacts"
    if is_kaggle_runtime()
    else ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Frozen experiment, authorization, and compute contract
#
# `replacement_preflight`は0 booster・PF 0。以降は40 CPU、20 CPU、15 GPUを
# 段階別に実行する。saved exp335 control再学習は全段0。

# %%
stage = str(config["execution"]["stage"])
allowed_stages = {str(item) for item in config["execution"]["allowed_stages"]}
if stage not in allowed_stages:
    raise ValueError(f"unknown exp413 stage: {stage}")
require_stage_authorization(config, stage)
cost_contract = replacement_cost_contract(config)

parent_exp264_config_path = resolve_file(config["data"]["parent_configs"]["exp264"])
parent_exp335_config_path = resolve_file(config["data"]["parent_configs"]["exp335"])
parent_exp264_config = load_yaml(parent_exp264_config_path)
parent_exp335_config = load_yaml(parent_exp335_config_path)
candidate_contract_path = resolve_file(config["data"]["candidate_contract"])
candidate_contract = load_yaml(candidate_contract_path)
contract_evidence = validate_replacement_contract(config, candidate_contract)

effective_boosters = {
    "replacement_preflight": 0,
    "nested_selector_train": 40,
    "signed_selector_train": 20,
    "downstream_gpu_train": 15,
}[stage]
display(
    {
        "experiment": EXPERIMENT_NAME,
        "route": config["experiment"]["route"],
        "status": config["experiment"]["status"],
        "stage": stage,
        "effective_boosters_this_stage": effective_boosters,
        "full_cost_contract": cost_contract,
        "saved_exp335_control_retraining": 0,
        "train_pf_well_runs": 0,
        "inference": False,
        "submission": False,
        "parent_exp264_config_sha256": sha256_file(parent_exp264_config_path),
        "parent_exp335_config_sha256": sha256_file(parent_exp335_config_path),
        "candidate_contract_sha256": sha256_file(candidate_contract_path),
    }
)
display(contract_evidence)
assert config["experiment"]["route"] == "ml_model"
assert cost_contract["total_boosters"] == 75
assert cost_contract["parent_control_retraining_boosters"] == 0
assert cost_contract["train_pf_well_runs"] == 0
assert config["execution"]["parent_control_retraining"] is False
assert config["execution"]["same_oof_rescue_enabled"] is False
assert config["execution"]["inference_enabled"] is False
assert config["execution"]["submission_enabled"] is False

print("Leakage contract")
for rule in config["validation"]["leakage_policy"]:
    print("-", rule)

# %% [markdown]
# ## 3. Input paths and SHA contracts
#
# exp404 predictionはtruth/fold/hidden-likeを読む前にraw/decompressed/logical/schema
# SHAを検証する。selector、signed selector、downstreamは同じoverlay cacheを使う。

# %%
parent_cache_root = resolve_exp263_cache_root(parent_exp264_config, search_roots())
frozen_prediction_path = resolve_file(
    config["data"]["exp404_scale5_train_prediction"],
    sha_key="expected_raw_sha256",
)
feature_schema_path = resolve_file(
    {
        "patterns": config["data"]["selector_contract"]["feature_schema_patterns"],
        "sha256": config["data"]["selector_contract"]["feature_schema_file_sha256"],
    }
)
feature_catalog_path = resolve_file(
    {
        "patterns": config["data"]["selector_contract"]["feature_catalog_patterns"],
        "sha256": config["data"]["selector_contract"]["feature_catalog_sha256"],
    }
)
display(
    {
        "parent_exp263_cache": str(parent_cache_root),
        "frozen_scale5_prediction": str(frozen_prediction_path),
        "frozen_scale5_raw_sha256": sha256_file(frozen_prediction_path),
        "selector_feature_schema": str(feature_schema_path),
        "selector_feature_catalog": str(feature_catalog_path),
        "raw_train_dir": str(raw_train_dir),
        "output_dir": str(output_dir),
    }
)

# %% [markdown]
# ## 4. Stage 0 replacement cache and transitive-lineage preflight
#
# parent fold identityへstrict joinし、replacement `likpf_mean` primitive partitionを
# 作る。旧meanは5 changed / 7 unchanged parity監査にだけ読み、overlay cache、
# selector probe、後段model viewへ渡さない。

# %%
stage_summary: dict[str, Any] | None = None
stage_0_evidence: dict[str, Any] | None = None
if stage == "replacement_preflight":
    stage_summary = run_replacement_preflight(
        config=config,
        contract=candidate_contract,
        parent_config=parent_exp264_config,
        parent_cache_root=parent_cache_root,
        frozen_prediction_path=frozen_prediction_path,
        feature_schema_path=feature_schema_path,
        feature_catalog_path=feature_catalog_path,
        raw_train_dir=raw_train_dir,
        output_dir=output_dir,
    )
    assert stage_summary["models_trained"] == 0
    assert stage_summary["pf_well_runs"] == 0
    assert stage_summary["passed"] is True
    stage_0_evidence = verify_replacement_stage_0_root(output_dir, config)
    display(stage_summary)
else:
    print("Stage 0 is read from a separately completed, SHA-recorded artifact root.")

# %% [markdown]
# ## 5. Stage C strict-nested dual selector
#
# replacement bankから固定88特徴を再構築し、2 objectives × outer 5 × inner 4 =
# 40 CPU boostersだけを学習する。saved exp264 selector/controlは再学習しない。

# %%
stage_0_root: Path | None = None
stage_c_root: Path | None = None
stage_s_root: Path | None = None
if stage != "replacement_preflight":
    stage_0_root = resolve_root(
        [str(item) for item in config["data"]["replacement_stage_0_root_patterns"]],
        "replacement_preflight.json",
    )
    replacement_root = stage_0_root / "replacement_candidate_cache"
    stage_0_evidence = verify_replacement_stage_0_root(stage_0_root, config)
    overlay_factory = replacement_cache_factory(replacement_root)
    display(
        {
            "stage_0_root": str(stage_0_root),
            "replacement_semantic_manifest_sha256": sha256_file(
                stage_0_root / "replacement_semantic_manifest.json"
            ),
            "replacement_cache_root": str(replacement_root),
            "stage_0_preflight_sha256": stage_0_evidence["preflight_sha256"],
        }
    )

if stage == "nested_selector_train":
    shutil.copy2(stage_0_root / "feature_schema.json", output_dir / "feature_schema.json")
    shutil.copy2(stage_0_root / "feature_catalog.csv", output_dir / "feature_catalog.csv")
    shutil.copy2(
        stage_0_root / "compact_meta_schema.json",
        output_dir / "compact_meta_schema.json",
    )
    shutil.copy2(
        stage_0_root / "replacement_semantic_manifest.json",
        output_dir / "replacement_semantic_manifest.json",
    )
    write_json(
        output_dir / "reproducibility_manifest.json",
        {
            "schema_version": "1.0.0",
            "status": "stage_c_inputs_frozen",
            "experiment": EXPERIMENT_NAME,
            "stage": stage,
            "seed": config["reproducibility"]["seed"],
            "deterministic_anchor": config["reproducibility"][
                "deterministic_anchor"
            ],
            "stage_0_preflight_sha256": stage_0_evidence["preflight_sha256"],
            "replacement_semantic_manifest_sha256": stage_0_evidence[
                "semantic_manifest_sha256"
            ],
            "replacement_frozen_prediction": stage_0_evidence[
                "frozen_prediction"
            ],
            "selector_feature_schema_sha256": sha256_file(
                output_dir / "feature_schema.json"
            ),
            "selector_feature_catalog_sha256": sha256_file(
                output_dir / "feature_catalog.csv"
            ),
            "compact_meta_schema_sha256": sha256_file(
                output_dir / "compact_meta_schema.json"
            ),
            "parent_exp264_config_sha256": sha256_file(
                parent_exp264_config_path
            ),
            "candidate_contract_sha256": sha256_file(
                candidate_contract_path
            ),
            "cost_contract": {
                "variants": 1,
                "objectives": 2,
                "outer_folds": 5,
                "inner_folds": 4,
                "cpu_boosters": 40,
                "control_retraining_boosters": 0,
                "pf_well_runs": 0,
            },
        },
    )
    stage_c_config = stage_c_runtime_config(config, parent_exp264_config)
    stage_summary = run_stage_c(
        config=stage_c_config,
        contract=candidate_contract,
        cache_root=parent_cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=output_dir,
        cache_factory=overlay_factory,
        hard_readout_enabled=False,
    )
    assert stage_summary["model_count"] == 40
    assert stage_summary["leakage_audit"]["passed"] is True
    write_json(
        output_dir / "replacement_stage_c_lineage.json",
        {
            "semantic_manifest_sha256": sha256_file(
                output_dir / "replacement_semantic_manifest.json"
            ),
            "stage_0_preflight_sha256": stage_0_evidence["preflight_sha256"],
            "value_source": "likpf_scale_5_x1p0",
            "frozen_raw_sha256": config["data"]["exp404_scale5_train_prediction"][
                "expected_raw_sha256"
            ],
            "frozen_decompressed_sha256": config["data"][
                "exp404_scale5_train_prediction"
            ]["expected_decompressed_sha256"],
            "frozen_logical_sha256": config["data"]["exp404_scale5_train_prediction"][
                "expected_logical_sha256"
            ],
            "frozen_schema_sha256": config["data"]["exp404_scale5_train_prediction"][
                "expected_schema_sha256"
            ],
            "old_mean_in_model_input": False,
            "models_trained": 40,
            "control_models_retrained": 0,
            "nested_compact_manifest_sha256": sha256_file(
                output_dir / "nested_compact_manifest.json"
            ),
        },
    )
    display(stage_summary)
elif stage not in {"replacement_preflight"}:
    print("Stage C is read from a separately completed, SHA-recorded artifact root.")

# %% [markdown]
# ## 6. Stage S strict-nested signed-residual selector
#
# replacement Stage C top-1注釈と同じcandidate bankを使い、
# `true_tvt - candidate_tvt`の1 objective × outer 5 × inner 4 = 20 CPU
# boostersを学習する。signed23はouter-trainにinner OOF、outer-validに4-model
# ensembleを使う。

# %%
if stage in {"signed_selector_train", "downstream_gpu_train"}:
    stage_c_root = resolve_root(
        [str(item) for item in config["data"]["replacement_stage_c_root_patterns"]],
        "replacement_stage_c_lineage.json",
    )
    if not (
        stage_c_root / "replacement_semantic_manifest.json"
    ).exists():
        raise FileNotFoundError("replacement Stage C semantic manifest is missing")
    stage_c_verify_config = stage_s_runtime_config(
        config, parent_exp335_config, stage_c_root
    )
    stage_c_evidence = verify_replacement_stage_c_root(
        stage_c_root, stage_c_verify_config
    )
    if (
        stage_c_evidence["replacement_semantic_manifest_sha256"]
        != stage_0_evidence["semantic_manifest_sha256"]
        or stage_c_evidence["stage_0_preflight_sha256"]
        != stage_0_evidence["preflight_sha256"]
    ):
        raise ValueError("replacement Stage C does not descend from selected Stage 0")
    display(
        {
            "stage_c_root": str(stage_c_root),
            "stage_c_lineage_sha256": sha256_file(
                stage_c_root / "replacement_stage_c_lineage.json"
            ),
            "stage_c_semantic_manifest_sha256": sha256_file(
                stage_c_root / "replacement_semantic_manifest.json"
            ),
            "stage_c_model_count": stage_c_evidence["model_count"],
        }
    )

if stage == "signed_selector_train":
    shutil.copy2(
        stage_0_root / "replacement_semantic_manifest.json",
        output_dir / "replacement_semantic_manifest.json",
    )
    stage_s_config = stage_s_runtime_config(
        config, parent_exp335_config, stage_c_root
    )
    stage_summary = run_stage_s(
        config=stage_s_config,
        contract=candidate_contract,
        cache_root=parent_cache_root,
        parent_stage_c_root=stage_c_root,
        feature_schema_path=stage_0_root / "feature_schema.json",
        feature_catalog_path=stage_0_root / "feature_catalog.csv",
        raw_train_dir=raw_train_dir,
        output_dir=output_dir,
        require_parent_score_guard=False,
        cache_factory=overlay_factory,
    )
    assert stage_summary["model_count"] == 20
    assert stage_summary["technical_gate"]["passed"] is True
    write_json(
        output_dir / "replacement_stage_s_lineage.json",
        {
            "semantic_manifest_sha256": sha256_file(
                output_dir / "replacement_semantic_manifest.json"
            ),
            "stage_c_lineage_sha256": sha256_file(
                stage_c_root / "replacement_stage_c_lineage.json"
            ),
            "stage_0_preflight_sha256": stage_0_evidence["preflight_sha256"],
            "value_source": "likpf_scale_5_x1p0",
            "old_mean_in_model_input": False,
            "models_trained": 20,
            "control_models_retrained": 0,
            "signed_compact_manifest_sha256": sha256_file(
                output_dir / "signed_compact_manifest.json"
            ),
        },
    )
    display(stage_summary)
elif stage == "downstream_gpu_train":
    print("Stage S is read from a separately completed, SHA-recorded artifact root.")

# %% [markdown]
# ## 7. Stage D clean273 + compact74 + signed23 downstream
#
# exp072 full replay sourceを再読込し、primitiveをscale5へ置換してからprojection、
# exp145 learned-likelihood transform、GRWRをすべて再構築する。allowlist上の
# `likpf_mean`名22列だけをpatchしない。final370で3 configs × 5 folds = 15
# GPU boostersを学習し、saved exp335 OOFと比較する。

# %%
if stage == "downstream_gpu_train":
    stage_s_root = resolve_root(
        [str(item) for item in config["data"]["replacement_stage_s_root_patterns"]],
        "replacement_stage_s_lineage.json",
    )
    stage_s_verify_config = downstream_runtime_config(
        config,
        parent_exp335_config,
        stage_c_root,
        stage_s_root,
    )
    stage_s_evidence = verify_replacement_stage_s_root(
        stage_s_root,
        stage_s_verify_config,
        stage_c_root=stage_c_root,
    )
    if (
        stage_s_evidence["replacement_semantic_manifest_sha256"]
        != stage_0_evidence["semantic_manifest_sha256"]
    ):
        raise ValueError("replacement Stage S does not descend from selected Stage 0")
    downstream_config = downstream_runtime_config(
        config,
        parent_exp335_config,
        stage_c_root,
        stage_s_root,
    )
    exp218_source_path = resolve_by_patterns(
        config["data"]["exp218_source"]["script_patterns"],
        search_roots(),
        marker_sha256=config["data"]["exp218_source"]["script_sha256"],
    )
    exp218_config_path = resolve_by_patterns(
        config["data"]["exp218_source"]["config_patterns"],
        search_roots(),
        marker_sha256=config["data"]["exp218_source"]["config_sha256"],
    )
    exp145_source_path = resolve_by_patterns(
        config["data"]["exp145_source"]["script_patterns"],
        search_roots(),
        marker_sha256=config["data"]["exp145_source"]["script_sha256"],
    )
    exp145_config_path = resolve_by_patterns(
        config["data"]["exp145_source"]["config_patterns"],
        search_roots(),
        marker_sha256=config["data"]["exp145_source"]["config_sha256"],
    )
    multiobs_source_path = resolve_by_patterns(
        config["data"]["exp145_source"]["multiobs_script_patterns"],
        search_roots(),
        marker_sha256=config["data"]["exp145_source"]["multiobs_script_sha256"],
    )
    exp099_source_path = resolve_by_patterns(
        config["data"]["exp099_train_feature_cache"]["patterns"],
        search_roots(),
        marker_sha256=config["data"]["exp099_train_feature_cache"][
            "expected_raw_sha256"
        ],
    )
    exp111_schema_path = resolve_by_patterns(
        config["data"]["exp111_saved_models"]["schema_patterns"],
        search_roots(),
        marker_sha256=config["data"]["exp111_saved_models"]["schema_sha256"],
    )
    exp111_manifest_path = resolve_by_patterns(
        config["data"]["exp111_saved_models"]["manifest_patterns"],
        search_roots(),
        marker_sha256=config["data"]["exp111_saved_models"]["manifest_sha256"],
    )
    clean_allowlist_path = resolve_file(config["data"]["clean_base_allowlist"])
    hidden_like_assignment_path = resolve_file(
        config["data"]["hidden_like_assignment"]
    )
    saved_parent_oof_path = resolve_by_patterns(
        config["data"]["exp335_saved_control"]["oof_patterns"],
        search_roots(),
        marker_sha256=config["data"]["exp335_saved_control"]["oof_sha256"],
    )
    saved_parent_metrics_path = resolve_by_patterns(
        config["data"]["exp335_saved_control"]["metrics_patterns"],
        search_roots(),
        marker_sha256=config["data"]["exp335_saved_control"]["metrics_sha256"],
    )
    saved_parent_model_manifest_path = resolve_by_patterns(
        config["data"]["exp335_saved_control"]["model_manifest_patterns"],
        search_roots(),
        marker_sha256=config["data"]["exp335_saved_control"][
            "model_manifest_sha256"
        ],
    )
    stage_summary = run_replacement_stage_d(
        config=config,
        runtime_config=downstream_config,
        contract=candidate_contract,
        stage_c_root=stage_c_root,
        stage_s_root=stage_s_root,
        saved_parent_oof_path=saved_parent_oof_path,
        saved_parent_metrics_path=saved_parent_metrics_path,
        saved_parent_model_manifest_path=saved_parent_model_manifest_path,
        hidden_like_assignment_path=hidden_like_assignment_path,
        frozen_prediction_path=frozen_prediction_path,
        exp218_source_path=exp218_source_path,
        exp218_config_path=exp218_config_path,
        exp099_source_path=exp099_source_path,
        exp145_source_path=exp145_source_path,
        exp145_config_path=exp145_config_path,
        multiobs_source_path=multiobs_source_path,
        exp111_schema_path=exp111_schema_path,
        exp111_manifest_path=exp111_manifest_path,
        clean_allowlist_path=clean_allowlist_path,
        raw_train_dir=raw_train_dir,
        output_dir=output_dir,
    )
    assert stage_summary["model_count"] == 15
    assert stage_summary["cost_contract"]["parent_control_retraining_boosters"] == 0
    display(stage_summary["primary_gate"])
else:
    print("Stage D was not executed in this stage.")

# %% [markdown]
# ## 8. Metrics, feature importance, and generated artifacts

# %%
if stage == "nested_selector_train":
    metrics = pd.read_csv(output_dir / "nested_selector_metrics.csv")
    importance = pd.read_csv(
        output_dir / "nested_feature_importance_by_objective_outer_inner.csv"
    )
    display(metrics)
    importance_mean = (
        importance[importance["importance_type"].eq("gain")]
        .groupby(["objective", "feature"], as_index=False)["importance"]
        .mean()
        .sort_values(["objective", "importance"], ascending=[True, False])
    )
    display(importance_mean.groupby("objective", group_keys=False).head(30))
elif stage == "signed_selector_train":
    metrics = pd.read_csv(output_dir / "signed_selector_metrics.csv")
    importance = pd.read_csv(
        output_dir / "signed_feature_importance_by_outer_inner.csv"
    )
    display(metrics)
    importance_mean = (
        importance[importance["importance_type"].eq("gain")]
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    display(importance_mean.head(30))
elif stage == "downstream_gpu_train":
    fold_metrics = pd.read_csv(output_dir / "stage_d_fold_metrics.csv")
    scope_metrics = pd.read_csv(output_dir / "stage_d_scope_metrics.csv")
    hidden_metrics = pd.read_csv(output_dir / "stage_d_hidden_like_metrics.csv")
    by_well = pd.read_csv(output_dir / "stage_d_by_well.csv")
    importance = pd.read_csv(output_dir / "stage_d_feature_importance.csv")
    display(fold_metrics[fold_metrics["model"].eq("lgb_mean")])
    display(scope_metrics)
    display(hidden_metrics)
    display(
        by_well.sort_values(
            "delta_rmse_replacement_minus_exp335", ascending=False
        ).head(80)
    )
    importance_mean = (
        importance[importance["importance_type"].eq("gain")]
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    display(importance_mean.head(100))
    top = importance_mean.head(30).sort_values("importance")
    if len(top):
        ax = top.plot.barh(
            x="feature",
            y="importance",
            figsize=(11, 11),
            legend=False,
            title="exp413 final370 mean gain importance",
        )
        ax.set_xlabel("mean gain across 15 replacement models")
        plt.tight_layout()
        plt.savefig(output_dir / "stage_d_feature_importance_top30.png", dpi=140)
        plt.show()
else:
    display(pd.DataFrame([stage_summary["contract"]["cost_contract"]]))

# %% [markdown]
# ## 9. Reproducibility evidence and fixed stop
#
# gzipはdecompressed content SHAを主証拠にする。GPU bitwise deterministicは
# 主張しない。gate PASSでも推論実装資格を得るだけで、このnotebookは必ず停止する。

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
if stage == "downstream_gpu_train":
    if stage_summary["primary_gate"]["passed"]:
        print(
            "exp413 train gate PASS. Stop here; current-test inference implementation, "
            "run, submission generation, and external submission require separate approval."
        )
    else:
        print(
            "exp413 train gate FAIL. Close without same-OOF scale, multiplier, candidate, "
            "feature, weight, threshold, or blend rescue."
        )
else:
    print(
        f"{stage} complete. Stop here; the next stage remains separately approval-gated."
    )
print("Inference executed: False")
print("Submission generated or submitted: False")

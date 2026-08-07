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
# # exp371 fixed13 selector — Stage D TVT train
#
# exp371のsaved fixed13 compact 77列を、監査済みexp218 clean 273列へadd-onlyする。
# 新規学習は1 variant × 3 LightGBM configs × 5 folds = 15 GPU boostersだけとする。
# 比較対象は保存済みexp264 Stage D v3（parent12 compact add-only）で、controlは再学習しない。
#
# Stage Cのfixed13 selectorはpooled RMSEを改善した一方、by-well p95 / worst安全gateには
# 不合格だった。このnotebookはユーザーの平均改善を根拠にした明示的な次段実行であり、
# 元の不合格判定をPASSへ変更しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and Kaggle runtime helpers
# 2. Override and 15-booster compute contract
# 3. Frozen exp371 Stage C input
# 4. Clean273, saved exp264, and hidden-like inputs
# 5. Stage D execution
# 6. Downstream scientific gate
# 7. Compact77 feature-importance readout
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

from src.candidate_selector_pipeline import resolve_existing_path, sha256_file
from src.exp333_fixed13_candidate_cache import (
    exp371_stage_d_cost_contract,
    run_exp371_fixed13_stage_d_addonly,
)

EXPERIMENT_NAME = "exp371_exp333_fixed13_dual_selector_on_exp264"
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
        [pattern for pattern in patterns if not Path(pattern).is_absolute()],
        search_roots,
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
    patterns: list[str],
    marker: str,
    expected_sha256: str,
    search_roots: list[Path],
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
search_roots = [KAGGLE_INPUT_ROOT, KAGGLE_WORKING_ROOT, Path("/tmp"), ROOT]
output_dir = (
    KAGGLE_WORKING_ROOT / "artifacts"
    if is_kaggle_runtime()
    else ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
)
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Override and 15-booster compute contract
#
# 許可された例外はStage Dの実行だけであり、Stage C safety gateの再分類ではない。
# T4 / internet off / double precision / deterministic flags / threads 8を固定する。

# %%
if str(config["execution"]["stage"]) != "downstream_tvt_train":
    raise RuntimeError("Stage D notebook requires execution.stage=downstream_tvt_train")
if not bool(config["execution"]["downstream_train_approved"]):
    raise RuntimeError("Stage D implementation/train approval is missing")
if not bool(config["execution"]["run_downstream_train"]):
    raise RuntimeError("Stage D run flag is disabled")
if bool(config["execution"]["control_retraining"]):
    raise RuntimeError("saved exp264 control retraining is forbidden")
if bool(config["outcome"]["scientific_gate_passed"]):
    raise RuntimeError("historical Stage C safety gate must remain failed")
if bool(config["outcome"]["selector_gate_reclassified"]):
    raise RuntimeError("Stage C safety gate must not be reclassified")
if not bool(config["outcome"]["selector_gate_override_received"]):
    raise RuntimeError("explicit selector-gate execution override is missing")
cost_contract = exp371_stage_d_cost_contract(config)
display(
    {
        "experiment": EXPERIMENT_NAME,
        "route": config["experiment"]["route"],
        "stage": config["execution"]["stage"],
        **cost_contract,
        "historical_selector_gate_passed": False,
        "explicit_stage_d_override": True,
        "saved_exp264_control_retraining": 0,
        "inference": False,
        "submission": False,
        "gpu": True,
        "machine_shape": config["runtime"]["kaggle"]["tvt_train"]["machine_shape"],
        "internet": config["runtime"]["kaggle"]["enable_internet"],
    }
)
assert config["experiment"]["route"] == "ml_model"
assert cost_contract["total_gpu_boosters"] == 15
assert cost_contract["control_retraining"] is False
assert config["runtime"]["kaggle"]["tvt_train"]["enable_gpu"] is True
assert config["runtime"]["kaggle"]["enable_internet"] is False
assert config["runtime"]["kaggle"]["gpu_use_dp"] is True
assert config["runtime"]["kaggle"]["deterministic"] is True
assert config["runtime"]["kaggle"]["force_col_wise"] is True
assert config["runtime"]["kaggle"]["num_threads"] == 8

# %% [markdown]
# ## 3. Frozen exp371 Stage C input
#
# 完了済みStage C v3の40 selector model、25 compact partition、compact77 schemaを
# file/logical SHAで固定する。全25 partitionのbyte SHAはfit直前に共通runner内で検証する。

# %%
stage_c_root = resolve_root_by_marker_sha(
    [str(item) for item in config["data"]["stage_c_artifact_root_patterns"]],
    "nested_selector_metrics.json",
    str(config["data"]["stage_c_expected_nested_selector_metrics_sha256"]),
    search_roots,
)
compact_schema_path = stage_c_root / "compact_meta_schema.json"
compact_schema = json.loads(compact_schema_path.read_text())
if sha256_file(compact_schema_path) != str(
    config["data"]["stage_c_expected_compact_meta_schema_file_sha256"]
):
    raise ValueError("exp371 compact schema file SHA mismatch")
if str(compact_schema["compact_meta_schema_sha256"]) != str(
    config["data"]["stage_c_expected_compact_meta_schema_logical_sha256"]
):
    raise ValueError("exp371 compact schema logical SHA mismatch")
compact_features = [str(item) for item in compact_schema["features"]]
if len(compact_features) != 77 or len(set(compact_features)) != 77:
    raise ValueError("exp371 Stage C must provide exactly 77 compact features")
display(
    {
        "stage_c_root": str(stage_c_root),
        "nested_selector_metrics_sha256": sha256_file(
            stage_c_root / "nested_selector_metrics.json"
        ),
        "nested_selector_model_manifest_sha256": sha256_file(
            stage_c_root / "nested_selector_model_manifest.json"
        ),
        "nested_compact_manifest_sha256": sha256_file(
            stage_c_root / "nested_compact_manifest.json"
        ),
        "compact_schema_file_sha256": sha256_file(compact_schema_path),
        "compact_schema_logical_sha256": compact_schema[
            "compact_meta_schema_sha256"
        ],
        "compact_feature_count": len(compact_features),
    }
)

# %% [markdown]
# ## 4. Clean273, saved exp264, and hidden-like inputs
#
# exp218 source/configから380列を同じ手順で再構築し、監査済みallowlistで273列へ落とす。
# 保存済みexp264 Stage D v3のmetrics/fold/bucket/hidden/by-wellはSHA固定して比較にだけ使う。
# hidden-like assignmentは事後評価専用で、fitやearly stoppingには使わない。

# %%
exp218_source_path = resolve_file_by_sha(
    [str(item) for item in config["data"]["exp218_source_patterns"]],
    search_roots,
    str(config["data"]["exp218_expected_source_sha256"]),
)
exp218_config_path = resolve_file_by_sha(
    [str(item) for item in config["data"]["exp218_config_patterns"]],
    search_roots,
    str(config["data"]["exp218_expected_config_sha256"]),
)
clean_allowlist_path = resolve_file_by_sha(
    [str(item) for item in config["data"]["exp218_clean_273_allowlist_patterns"]],
    search_roots,
    str(config["data"]["exp218_clean_273_allowlist_expected_sha256"]),
)
hidden_like_assignment_path = resolve_file_by_sha(
    [str(item) for item in config["data"]["hidden_like_assignment_patterns"]],
    search_roots,
    str(config["data"]["hidden_like_assignment_expected_sha256"]),
)
parent_cfg = config["data"]["parent_stage_d_v3"]
parent_reference_paths = {
    "metrics": resolve_file_by_sha(
        [str(item) for item in parent_cfg["metrics_patterns"]],
        search_roots,
        str(parent_cfg["expected_metrics_sha256"]),
    ),
    "fold_metrics": resolve_file_by_sha(
        [str(item) for item in parent_cfg["fold_metrics_patterns"]],
        search_roots,
        str(parent_cfg["expected_fold_metrics_sha256"]),
    ),
    "bucket_metrics": resolve_file_by_sha(
        [str(item) for item in parent_cfg["bucket_metrics_patterns"]],
        search_roots,
        str(parent_cfg["expected_bucket_metrics_sha256"]),
    ),
    "hidden_like_metrics": resolve_file_by_sha(
        [str(item) for item in parent_cfg["hidden_like_metrics_patterns"]],
        search_roots,
        str(parent_cfg["expected_hidden_like_metrics_sha256"]),
    ),
    "by_well": resolve_file_by_sha(
        [str(item) for item in parent_cfg["by_well_patterns"]],
        search_roots,
        str(parent_cfg["expected_by_well_sha256"]),
    ),
}
display(
    {
        "exp218_source": str(exp218_source_path),
        "exp218_source_sha256": sha256_file(exp218_source_path),
        "exp218_config": str(exp218_config_path),
        "exp218_config_sha256": sha256_file(exp218_config_path),
        "clean273_allowlist": str(clean_allowlist_path),
        "clean273_allowlist_sha256": sha256_file(clean_allowlist_path),
        "hidden_like_assignment": str(hidden_like_assignment_path),
        "hidden_like_assignment_sha256": sha256_file(
            hidden_like_assignment_path
        ),
        "saved_parent_stage_d_reference_sha256": {
            name: sha256_file(path) for name, path in parent_reference_paths.items()
        },
        "raw_train_dir": str(raw_train_dir),
    }
)

# %% [markdown]
# ## 5. Stage D execution
#
# clean273 + fixed13 compact77 = 350列で15本だけ学習する。
# Stage C selector、candidate generator、PF/HMM、保存済みcontrolは再学習しない。

# %%
stage_d_summary = run_exp371_fixed13_stage_d_addonly(
    config=config,
    stage_c_root=stage_c_root,
    exp218_source_path=exp218_source_path,
    exp218_config_path=exp218_config_path,
    base_feature_allowlist_path=clean_allowlist_path,
    hidden_like_assignment_path=hidden_like_assignment_path,
    raw_train_dir=raw_train_dir,
    parent_reference_paths=parent_reference_paths,
    output_dir=output_dir,
)
assert stage_d_summary["model_count"] == 15
assert stage_d_summary["cost_contract"]["control_retraining"] is False
display(stage_d_summary)

# %% [markdown]
# ## 6. Downstream scientific gate
#
# 保存済みparent12 compact add-only比で、pooled改善、3/5 folds、near、1000+、
# hidden-like 2面、by-well p95 / worstをAND判定する。Stage Cの過去gateとは別判定である。

# %%
fold_comparison = pd.read_csv(output_dir / "stage_d_parent_fold_comparison.csv")
bucket_comparison = pd.read_csv(output_dir / "stage_d_bucket_comparison.csv")
hidden_comparison = pd.read_csv(output_dir / "stage_d_hidden_like_comparison.csv")
by_well_comparison = pd.read_csv(output_dir / "stage_d_by_well_comparison.csv")
display(stage_d_summary["comparison_vs_parent12"])
display(fold_comparison)
display(bucket_comparison)
display(hidden_comparison)
display(
    by_well_comparison.sort_values(
        "delta_new13_minus_parent12", ascending=False
    ).head(80)
)

# %% [markdown]
# ## 7. Compact77 feature-importance readout
#
# fixed13 compact77のgain/splitを15 modelから保存する。重要度は補助証拠であり、
# RMSEと下流safety gateを代替しない。

# %%
importance = pd.read_csv(output_dir / "stage_d_feature_importance.csv")
compact_importance = (
    importance[
        importance["feature"].isin(compact_features)
        & importance["importance_type"].eq("gain")
    ]
    .groupby("feature", as_index=False)["importance"]
    .sum()
    .sort_values("importance", ascending=False)
)
display(compact_importance)
if len(compact_importance):
    top = compact_importance.head(35).sort_values("importance")
    ax = top.plot.barh(
        x="feature",
        y="importance",
        figsize=(12, 12),
        legend=False,
        title="exp371 Stage D fixed13 compact feature gain across 15 models",
    )
    ax.set_xlabel("summed gain importance")
    plt.tight_layout()
    plt.savefig(
        output_dir / "stage_d_fixed13_compact_feature_importance.png", dpi=140
    )
    plt.show()

# %% [markdown]
# ## 8. Generated artifacts and reproducibility evidence
#
# model/OOF/input/output SHAを保存する。GPU bitwise再現は主張せず、
# inference、submission、competition submitは実行しない。

# %%
reproducibility = json.loads(
    (output_dir / "exp371_stage_d_reproducibility_manifest.json").read_text()
)
display(reproducibility)
print("Generated files")
for generated in sorted(output_dir.rglob("*")):
    if generated.is_file():
        print(generated.relative_to(output_dir), generated.stat().st_size)
print(
    "Stage D parent12 comparison gate passed:",
    stage_d_summary["comparison_vs_parent12"]["passed"],
)
print("Historical Stage C safety gate passed: False")
print("Historical Stage C gate reclassified: False")
print("Inference executed: False")
print("Submission generated or submitted: False")

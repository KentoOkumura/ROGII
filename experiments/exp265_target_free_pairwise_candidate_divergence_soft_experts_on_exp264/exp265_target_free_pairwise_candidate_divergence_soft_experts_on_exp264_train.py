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
# # exp265 target-free pairwise candidate divergence soft experts — train
#
# exp263の6 primitive candidate path間の15 pair差分と非TVT raw context、target-free
# confidenceから512-row block fingerprintを作る。outer-trainだけでK=3 regimeをfitし、
# exp264 candidate score OOFはassignment確定後のseparability評価にだけ使う。
# Stage 0ではモデルを学習せず、soft expert化へ進む条件だけを0 boosterで監査する。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Stage and compute contract
# 3. Input and leakage contract
# 4. Pairwise block fingerprint
# 5. Outer-fold regime assignment
# 6. Post-assignment exp264 score audit
# 7. Artifacts and reproducibility evidence

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from settings import EXPERIMENT_NAME, ExperimentPaths, load_config

from src.candidate_pairwise_regime import (
    build_block_fingerprints,
    evaluate_regime_separability_from_parquet,
    expand_row_assignments,
    feature_columns,
    fit_outer_fold_regimes,
    load_primitive_fold,
    pair_ids,
    primitive_ids,
    read_contract,
    save_stage0_artifacts,
)
from src.candidate_selector_pipeline import (
    resolve_exp263_cache_root,
    sha256_file,
    verify_exp263_root,
)

paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()
output_dir = paths.artifacts_dir


def resolve_support_file(filename: str) -> Path:
    candidates = [
        Path.cwd() / filename,
        paths.experiment_dir / filename,
        Path("/kaggle/working") / filename,
    ]
    matches = [path for path in candidates if path.exists()]
    if not matches:
        matches = list(Path("/kaggle/working").rglob(filename))
    if not matches:
        raise FileNotFoundError(filename)
    return sorted(matches)[0]


def resolve_input(patterns: list[str], search_roots: list[Path]) -> Path:
    direct = [Path(pattern) for pattern in patterns if Path(pattern).exists()]
    if direct:
        return sorted(direct)[0]
    matches: list[Path] = []
    for pattern in patterns:
        if Path(pattern).is_absolute():
            continue
        for root in search_roots:
            if root.exists():
                matches.extend(root.glob(pattern))
    matches = sorted({path for path in matches if path.exists()})
    if not matches:
        raise FileNotFoundError(f"no input matches patterns={patterns}")
    return matches[0]


# %% [markdown]
# ## 2. Stage and compute contract
#
# Stage 0はfeature生成・KMeans・保存済みOOFの集計だけで、0 variant / 0 LightGBM config /
# 0 fold training / 0 booster。conditional Stage 1はK=3 × 2 objectives × 5 folds =
# 30 CPU boostersだが、このnotebookでは実行しない。

# %%
stage = str(config["execution"]["stage"])
if stage not in set(config["execution"]["allowed_stages"]):
    raise ValueError(f"unknown execution stage: {stage}")

cost_contract = {
    "experiment": EXPERIMENT_NAME,
    "route": config["experiment"]["route"],
    "stage": stage,
    "active_variants": config["execution"]["stage0_variants"],
    "lightgbm_configs": config["execution"]["stage0_lightgbm_configs"],
    "folds_trained": config["execution"]["stage0_folds"],
    "total_boosters": config["execution"]["stage0_total_boosters"],
    "conditional_stage1_boosters": config["execution"]["conditional_stage1_total_boosters"],
    "parent_or_control_retraining": False,
    "gpu": config["runtime"]["kaggle"]["enable_gpu"],
    "internet": config["runtime"]["kaggle"]["enable_internet"],
}
display(cost_contract)
assert config["experiment"]["route"] == "ensemble"
assert cost_contract["active_variants"] == 0
assert cost_contract["lightgbm_configs"] == 0
assert cost_contract["folds_trained"] == 0
assert cost_contract["total_boosters"] == 0
assert config["model"]["conditional_stage1"]["enabled"] is False
assert cost_contract["gpu"] is False
assert cost_contract["internet"] is False
if not bool(config["execution"]["run_approved"]):
    raise RuntimeError(
        "Stage 0 Kaggle run is not approved. Set execution.run_approved=true only after approval."
    )

# %% [markdown]
# ## 3. Input and leakage contract
#
# exp263のmanifest/catalog SHAを固定し、6 primitiveの値・availability・confidenceだけを読む。
# exp264 Stage Bはscore、model manifest、metricsの3 artifactが揃わない限りfail closedとする。
# scoreはregime fit前には読み込まない。

# %%
contract_path = resolve_support_file(str(config["data"]["regime_contract"]))
contract = read_contract(contract_path)
candidate_names = primitive_ids(contract)
assert len(candidate_names) == config["guards"]["technical"]["expected_primitives"]
assert len(pair_ids(candidate_names)) == config["guards"]["technical"]["expected_pairs"]

search_roots = [Path("/kaggle/input"), Path("/tmp"), paths.root]
cache_root = resolve_exp263_cache_root(config, search_roots)
cache_evidence = verify_exp263_root(cache_root, config)
score_path = resolve_input(
    [str(item) for item in config["data"]["exp264_candidate_score_patterns"]], search_roots
)
selector_manifest_path = resolve_input(
    [str(item) for item in config["data"]["exp264_selector_manifest_patterns"]], search_roots
)
selector_metrics_path = resolve_input(
    [str(item) for item in config["data"]["exp264_selector_metrics_patterns"]], search_roots
)
input_evidence = {
    "exp263": cache_evidence,
    "regime_contract_path": str(contract_path),
    "regime_contract_sha256": sha256_file(contract_path),
    "exp264_candidate_score_path": str(score_path),
    "exp264_candidate_score_sha256": sha256_file(score_path),
    "exp264_selector_manifest_path": str(selector_manifest_path),
    "exp264_selector_manifest_sha256": sha256_file(selector_manifest_path),
    "exp264_selector_metrics_path": str(selector_metrics_path),
    "exp264_selector_metrics_sha256": sha256_file(selector_metrics_path),
}
display(input_evidence)
assert (
    input_evidence["exp264_candidate_score_sha256"]
    == config["data"]["exp264_expected_candidate_score_sha256"]
)
assert (
    input_evidence["exp264_selector_manifest_sha256"]
    == config["data"]["exp264_expected_selector_manifest_sha256"]
)
assert (
    input_evidence["exp264_selector_metrics_sha256"]
    == config["data"]["exp264_expected_selector_metrics_sha256"]
)
assert cache_evidence["rows"] == config["guards"]["technical"]["expected_rows"]
assert cache_evidence["wells"] == config["guards"]["technical"]["expected_wells"]
assert paths.train_data_dir.exists()

print("Leakage contract")
for rule in config["validation"]["leakage_policy"]:
    print("-", rule)

# %% [markdown]
# ## 4. Pairwise block fingerprint
#
# blockはwell内のevaluation row先頭から固定512行。候補絶対TVTやlast-known TVTは保存せず、
# 15 pairのgap形状、bank rank/SVD構造、非TVT raw統計、target-free confidenceだけを作る。

# %%
block_parts: list[pd.DataFrame] = []
row_map_parts: list[pd.DataFrame] = []
for outer_fold in range(config["validation"]["n_folds"]):
    bundle = load_primitive_fold(cache_root, contract, outer_fold)
    fold_blocks, fold_row_map = build_block_fingerprints(
        bundle, paths.train_data_dir, config
    )
    block_parts.append(fold_blocks)
    row_map_parts.append(fold_row_map)
    print(
        f"fold={outer_fold}: rows={len(fold_row_map):,}, "
        f"wells={fold_row_map['well'].nunique():,}, blocks={len(fold_blocks):,}"
    )
    del bundle, fold_blocks, fold_row_map
    gc.collect()

blocks = pd.concat(block_parts, ignore_index=True).sort_values(
    ["outer_fold", "well", "block_id"], kind="stable"
).reset_index(drop=True)
row_map = pd.concat(row_map_parts, ignore_index=True).sort_values(
    ["outer_fold", "well", "well_row_idx"], kind="stable"
).reset_index(drop=True)
schema_columns = feature_columns(blocks)
assert len(row_map) == config["guards"]["technical"]["expected_rows"]
assert row_map["well"].nunique() == config["guards"]["technical"]["expected_wells"]
assert row_map["outer_fold"].nunique() == config["guards"]["technical"]["expected_folds"]
display(
    {
        "blocks": len(blocks),
        "rows": len(row_map),
        "wells": row_map["well"].nunique(),
        "features": len(schema_columns),
        "pair_features": sum(column.startswith("pair__") for column in schema_columns),
    }
)
display(blocks.head())

# %% [markdown]
# ## 5. Outer-fold regime assignment
#
# outer-train median補完、RobustScaler、KMeansをfoldごとにfitする。主seedと監査seedを
# centroid matchingし、outer-valid assignment一致率をstability guardにする。出力はhard label
# だけでなく3 regimeのsoft membershipを保持する。

# %%
block_assignments, centroids, stability = fit_outer_fold_regimes(blocks, config)
row_assignments = expand_row_assignments(row_map, block_assignments)
display(stability)
display(
    block_assignments.groupby(["outer_fold", "regime"])
    .agg(blocks=("block_key", "size"), wells=("well", "nunique"))
    .reset_index()
)

# %% [markdown]
# ## 6. Post-assignment exp264 score audit
#
# ここで初めてexp264 scoreをjoinする。candidate-long Parquetはbatch読込し、6 primitiveの
# regime×candidate集計だけを保持する。best family差またはcalibration bias差がなければ、
# 30-booster Stage 1へ進まない。

# %%
occupancy, candidate_metrics, audit_summary = evaluate_regime_separability_from_parquet(
    block_assignments=block_assignments,
    row_assignments=row_assignments,
    candidate_score_path=score_path,
    contract=contract,
    config=config,
    stability=stability,
)
calibration = audit_summary.pop("calibration")
display(occupancy)
display(candidate_metrics.query("outer_fold == 'all'").sort_values(["regime", "actual_rmse"]))
display(audit_summary)

# %%
figure, axes = plt.subplots(1, 2, figsize=(14, 4))
occupancy.pivot(index="outer_fold", columns="regime", values="block_share").plot.bar(
    ax=axes[0], title="OOF regime block share"
)
candidate_metrics.query("outer_fold == 'all'").pivot(
    index="candidate_id", columns="regime", values="actual_rmse"
).plot.bar(ax=axes[1], title="Primitive candidate RMSE by regime")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Artifacts and reproducibility evidence

# %%
stage0_summary = save_stage0_artifacts(
    output_dir=output_dir,
    blocks=blocks,
    block_assignments=block_assignments,
    row_assignments=row_assignments,
    centroids=centroids,
    stability=stability,
    occupancy=occupancy,
    candidate_metrics=candidate_metrics,
    calibration=calibration,
    summary=audit_summary,
    contract=contract,
    config=config,
    input_evidence=input_evidence,
)
display(stage0_summary)
print("Generated artifacts")
for artifact in config["artifacts"]["required"]:
    artifact_path = output_dir / artifact
    print(f"- {artifact}: exists={artifact_path.exists()}")
    assert artifact_path.exists()

if stage0_summary["stage0_guard_pass"]:
    print("Stage 0 PASS. Conditional Stage 1 still requires a separate approval.")
else:
    print("Stage 0 FAIL. Keep the saved exp264 global selector and do not train experts.")

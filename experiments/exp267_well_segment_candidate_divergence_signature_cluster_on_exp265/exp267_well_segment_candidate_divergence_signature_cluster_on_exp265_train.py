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
# # exp267 well-segment candidate divergence signature cluster — train
#
# exp263の6 primitive candidateがwell内の序盤・中盤・終盤でどの程度広がるかを、
# target-free 18次元well署名へ縮約する。outer-trainだけでK=3 clusterをfitし、
# 保存済みexp264 Stage B scoreはassignment確定後のadd-only前提監査にだけ使う。
# Stage Aは0 boosterであり、selector学習、inference、submissionは行わない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Stage and compute contract
# 3. Input, candidate, and leakage contract
# 4. Well-segment target-free signatures
# 5. Outer-fold semantic cluster assignment
# 6. Post-assignment exp264 score audit
# 7. Diagnostics, artifacts, and reproducibility evidence

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
    load_primitive_fold,
    pair_ids,
    primitive_ids,
    read_contract,
)
from src.candidate_selector_pipeline import (
    resolve_exp263_cache_root,
    sha256_file,
    verify_exp263_root,
)
from src.well_segment_candidate_divergence import (
    SEGMENTS,
    build_well_segment_signatures,
    evaluate_post_assignment_scores_from_parquet,
    fit_outer_fold_clusters,
    save_stage_a_artifacts,
    signature_feature_columns,
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
# Stage Aはsignature生成、KMeans、保存済みscoreの集計だけで、0 variant / 0 LightGBM
# config / 0 fold training / 0 booster。conditional Stage Bは18署名+3 membershipをexp264
# dual selectorへadd-onlyする10 CPU boostersだが、別承認までdisabledとする。

# %%
stage = str(config["execution"]["stage"])
if stage not in set(config["execution"]["allowed_stages"]):
    raise ValueError(f"unknown execution stage: {stage}")

cost_contract = {
    "experiment": EXPERIMENT_NAME,
    "route": config["experiment"]["route"],
    "stage": stage,
    "active_variants": config["execution"]["stage_a_variants"],
    "lightgbm_configs": config["execution"]["stage_a_lightgbm_configs"],
    "folds_trained": config["execution"]["stage_a_folds"],
    "total_boosters": config["execution"]["stage_a_total_boosters"],
    "conditional_stage_b_boosters": config["execution"][
        "conditional_stage_b_total_boosters"
    ],
    "parent_or_control_retraining": False,
    "gpu": config["runtime"]["kaggle"]["enable_gpu"],
    "internet": config["runtime"]["kaggle"]["enable_internet"],
}
display(cost_contract)
assert cost_contract["route"] == "ensemble"
assert cost_contract["active_variants"] == 0
assert cost_contract["lightgbm_configs"] == 0
assert cost_contract["folds_trained"] == 0
assert cost_contract["total_boosters"] == 0
assert config["model"]["conditional_stage_b"]["enabled"] is False
assert cost_contract["gpu"] is False
assert cost_contract["internet"] is False
if not bool(config["execution"]["run_approved"]):
    raise RuntimeError(
        "Stage A Kaggle run is not approved. Set execution.run_approved=true only after approval."
    )

# %% [markdown]
# ## 3. Input, candidate, and leakage contract
#
# exp263のmanifest/catalog SHAとexp264 Stage Bのscore/model manifest/metrics SHAを固定する。
# 6 primitiveの値はsignature生成に使うがabsolute値は保存せず、candidate間rangeと15 pair差だけを
# 集約する。exp264 scoreのParquet row dataはcluster assignment確定まで読まない。

# %%
contract_path = resolve_support_file(str(config["data"]["candidate_contract"]))
contract = read_contract(contract_path)
candidate_names = primitive_ids(contract)
assert len(candidate_names) == config["guards"]["technical"]["expected_primitives"]
assert len(pair_ids(candidate_names)) == config["guards"]["technical"]["expected_pairs"]

search_roots = [Path("/kaggle/input"), Path("/tmp"), paths.root]
cache_root = resolve_exp263_cache_root(config, search_roots)
cache_evidence = verify_exp263_root(cache_root, config)
score_path = resolve_input(
    [str(item) for item in config["data"]["exp264_candidate_score_patterns"]],
    search_roots,
)
selector_manifest_path = resolve_input(
    [str(item) for item in config["data"]["exp264_selector_manifest_patterns"]],
    search_roots,
)
selector_metrics_path = resolve_input(
    [str(item) for item in config["data"]["exp264_selector_metrics_patterns"]],
    search_roots,
)
input_evidence = {
    "exp263": cache_evidence,
    "candidate_contract_path": str(contract_path),
    "candidate_contract_sha256": sha256_file(contract_path),
    "exp264_candidate_score_path": str(score_path),
    "exp264_candidate_score_sha256": sha256_file(score_path),
    "exp264_selector_manifest_path": str(selector_manifest_path),
    "exp264_selector_manifest_sha256": sha256_file(selector_manifest_path),
    "exp264_selector_metrics_path": str(selector_metrics_path),
    "exp264_selector_metrics_sha256": sha256_file(selector_metrics_path),
}
display(input_evidence)
assert input_evidence["exp264_candidate_score_sha256"] == config["data"][
    "exp264_expected_candidate_score_sha256"
]
assert input_evidence["exp264_selector_manifest_sha256"] == config["data"][
    "exp264_expected_selector_manifest_sha256"
]
assert input_evidence["exp264_selector_metrics_sha256"] == config["data"][
    "exp264_expected_selector_metrics_sha256"
]
assert cache_evidence["rows"] == config["guards"]["technical"]["expected_rows"]
assert cache_evidence["wells"] == config["guards"]["technical"]["expected_wells"]

print("Leakage contract")
for rule in config["validation"]["leakage_policy"]:
    print("-", rule)

# %% [markdown]
# ## 4. Well-segment target-free signatures
#
# exp264と同じ`eval_position / eval_len`を使い、各wellをfixed thirdsへ分ける。
# 各区間でbank range mean/p90、effective rank、rank switch、15-pair absolute gap
# mean/p90をraw evaluation rowへ同じ重みを与えて集約し、18特徴とcoverageを保存する。

# %%
signature_parts: list[pd.DataFrame] = []
coverage_parts: list[pd.DataFrame] = []
for outer_fold in range(config["validation"]["n_folds"]):
    bundle = load_primitive_fold(cache_root, contract, outer_fold)
    fold_signatures, fold_coverage = build_well_segment_signatures(bundle, config)
    signature_parts.append(fold_signatures)
    coverage_parts.append(fold_coverage)
    print(
        f"fold={outer_fold}: rows={len(bundle.base):,}, "
        f"wells={len(fold_signatures):,}, features={len(signature_feature_columns())}"
    )
    del bundle, fold_signatures, fold_coverage
    gc.collect()

signatures = pd.concat(signature_parts, ignore_index=True).sort_values(
    ["outer_fold", "well"], kind="stable"
).reset_index(drop=True)
coverage = pd.concat(coverage_parts, ignore_index=True).sort_values(
    ["outer_fold", "well", "segment"], kind="stable"
).reset_index(drop=True)
assert len(signatures) == config["guards"]["technical"]["expected_wells"]
assert signatures["outer_fold"].nunique() == config["guards"]["technical"][
    "expected_folds"
]
assert len(signature_feature_columns()) == config["guards"]["technical"][
    "expected_features"
]
assert len(coverage) == len(signatures) * len(SEGMENTS)
display(signatures.head())
display(
    coverage.groupby("segment").agg(
        wells=("well", "nunique"),
        min_rows=("segment_rows", "min"),
        fallback_segments=("fallback_required", "sum"),
    )
)

# %% [markdown]
# ## 5. Outer-fold semantic cluster assignment
#
# outer-train median、RobustScaler、固定clip `[-10,10]`、KMeans K=3をfoldごとにfitする。
# centroidの18 scaled feature平均でlow/middle/highへsemantic化し、outer-validへsoft membershipを
# OOF付与する。別seedをHungarian matchingしてassignment stabilityを監査する。

# %%
(
    assignments,
    centroids,
    preprocessors,
    stability,
    occupancy,
    profiles,
    structure_summary,
) = fit_outer_fold_clusters(signatures, coverage, config)
display(stability)
display(occupancy)
display(profiles.query("segment == 'early'"))
display(structure_summary)

# %% [markdown]
# ## 6. Post-assignment exp264 score audit
#
# ここで初めてexp264 candidate-long score rowをstreaming読込する。6 primitiveについて
# cluster別actual MAE、predicted error calibration、fold別winner patternを集計する。
# worst clusterは最高error wellを1本除いても同じclusterかを確認する。

# %%
(
    candidate_metrics,
    calibration,
    well_score_metrics,
    score_summary,
) = evaluate_post_assignment_scores_from_parquet(
    assignments=assignments,
    candidate_score_path=score_path,
    contract=contract,
    config=config,
    structure_summary=structure_summary,
    batch_size=int(config["runtime"]["batch_size"]),
)
display(
    candidate_metrics.query("outer_fold == 'all'").sort_values(
        ["semantic_cluster", "actual_mae"]
    )
)
display(score_summary)

# %% [markdown]
# ## 7. Diagnostics, artifacts, and reproducibility evidence

# %%
figure, axes = plt.subplots(1, 3, figsize=(18, 4))
occupancy.query("outer_fold != 'all'").pivot(
    index="outer_fold", columns="semantic_cluster", values="well_share"
).plot.bar(ax=axes[0], title="OOF cluster well share")
profiles.pivot_table(
    index=["outer_fold", "segment"],
    columns="semantic_cluster",
    values="bank_range_mean",
).plot(ax=axes[1], marker="o", title="Bank range profile")
candidate_metrics.query("outer_fold == 'all'").pivot(
    index="candidate_id", columns="semantic_cluster", values="actual_mae"
).plot.bar(ax=axes[2], title="Primitive MAE by cluster")
plt.tight_layout()
plt.show()

# %%
stage_a_summary = save_stage_a_artifacts(
    output_dir=output_dir,
    signatures=signatures,
    coverage=coverage,
    assignments=assignments,
    centroids=centroids,
    preprocessors=preprocessors,
    stability=stability,
    occupancy=occupancy,
    profiles=profiles,
    candidate_metrics=candidate_metrics,
    calibration=calibration,
    well_score_metrics=well_score_metrics,
    structure_summary=structure_summary,
    score_summary=score_summary,
    contract=contract,
    config=config,
    input_evidence=input_evidence,
)
display(stage_a_summary)
print("Generated artifacts")
for artifact in config["artifacts"]["required"]:
    artifact_path = output_dir / artifact
    print(f"- {artifact}: exists={artifact_path.exists()}")
    assert artifact_path.exists()

if stage_a_summary["stage_a_guard_pass"]:
    print("Stage A PASS. Conditional 10-CPU-booster selector add-only still needs approval.")
else:
    print(
        "Stage A FAIL. Keep saved exp264 selector features and do not train "
        "the add-only variant."
    )

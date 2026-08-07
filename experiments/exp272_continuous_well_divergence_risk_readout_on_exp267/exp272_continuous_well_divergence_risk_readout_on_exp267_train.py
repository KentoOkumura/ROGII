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
# # exp272 continuous well divergence risk readout on exp267 — train
#
# exp267のtarget-free 18次元well署名をK=3へ離散化せず、outer-trainだけで正規化した
# continuous divergence axisとexp264 Stage B OOF candidate scoreのactual MAE / calibration
# biasの単調関係を0 boosterで監査する。primaryは事前固定12 range/gap特徴、PCA1は
# report-only sensitivityであり、selector学習、hard routing、inference、submissionは行わない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Compute, leakage, and decision contract
# 3. Fixed input and SHA checks
# 4. Outer-fold continuous divergence axes
# 5. Streaming exp264 candidate score aggregation
# 6. Spearman and stratified well-bootstrap readout
# 7. Diagnostics, artifacts, and reproducibility evidence

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from settings import EXPERIMENT_NAME, load_config

from src.candidate_selector_pipeline import (
    logical_frame_sha256,
    sha256_file,
)
from src.continuous_well_divergence_risk import (
    AXIS_COLUMNS,
    build_readout_tables,
    fit_oof_continuous_axes,
    primary_feature_columns,
    save_readout_artifacts,
    stream_candidate_well_metrics,
)
from src.well_segment_candidate_divergence import signature_feature_columns

EXECUTE_NOTEBOOK = os.environ.get("EXP272_IMPORT_ONLY", "0") != "1"


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def resolve_input(patterns: list[str], search_roots: list[Path]) -> Path:
    direct = sorted({Path(pattern) for pattern in patterns if Path(pattern).exists()})
    if direct:
        return direct[0]
    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            if Path(pattern).is_absolute():
                continue
            matches.extend(root.glob(pattern))
    matches = sorted({path for path in matches if path.exists() and path.stat().st_size > 0})
    if not matches:
        raise FileNotFoundError(f"no non-empty input matches patterns={patterns}")
    return matches[0]


def output_directory(root: Path) -> Path:
    if Path("/kaggle/working").exists():
        path = Path("/kaggle/working/artifacts")
    else:
        path = root / "experiments" / EXPERIMENT_NAME / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


# %% [markdown]
# ## 2. Compute, leakage, and decision contract
#
# 本readoutは0 variant / 0 config / 0 trained fold / 0 booster。primary axisの式・特徴・
# bootstrap thresholdはscoreを読む前にconfigで固定する。PCA1とcandidate別結果はreport-onlyで、
# primary guardを救済できない。

# %%
if EXECUTE_NOTEBOOK:
    config = load_config()
    root = project_root()
    output_dir = output_directory(root)
    compute_contract = {
        "experiment": EXPERIMENT_NAME,
        "route": config["experiment"]["route"],
        "stage": config["execution"]["stage"],
        "variants": int(config["execution"]["variants"]),
        "lightgbm_configs": int(config["execution"]["lightgbm_configs"]),
        "folds_trained": int(config["execution"]["folds_trained"]),
        "boosters": int(config["execution"]["total_boosters"]),
        "parent_control_retraining": bool(
            config["execution"]["parent_control_retraining"]
        ),
        "gpu": bool(config["runtime"]["kaggle"]["enable_gpu"]),
        "internet": bool(config["runtime"]["kaggle"]["enable_internet"]),
        "inference": bool(config["execution"]["inference_enabled"]),
        "submission": bool(config["execution"]["submission_enabled"]),
    }
    display(compute_contract)
    assert compute_contract["route"] == "ensemble"
    assert compute_contract["variants"] == 0
    assert compute_contract["lightgbm_configs"] == 0
    assert compute_contract["folds_trained"] == 0
    assert compute_contract["boosters"] == 0
    assert compute_contract["parent_control_retraining"] is False
    assert compute_contract["gpu"] is False
    assert compute_contract["internet"] is False
    assert compute_contract["inference"] is False
    assert compute_contract["submission"] is False
    assert config["axes"]["primary"] == "fixed_range_gap_axis"
    assert config["axes"]["sensitivity_decision_role"] == (
        "report_only_cannot_rescue_primary_guard"
    )
    assert config["guards"]["monotonic_risk"]["pca1_can_rescue"] is False
    assert config["guards"]["monotonic_risk"]["candidate_specific_can_rescue"] is False
    if not Path("/kaggle/input").exists() or not Path("/kaggle/working").exists():
        raise RuntimeError("The first full exp272 readout must run on Kaggle CPU.")
    if not bool(config["execution"]["run_approved"]):
        raise RuntimeError(
            "Kaggle readout is not approved. Set execution.run_approved=true only after approval."
        )
    print("Leakage contract")
    for rule in config["validation"]["leakage_policy"]:
        print("-", rule)


# %% [markdown]
# ## 3. Fixed input and SHA checks
#
# exp267 version 2のsignature byte/logical SHAと、immutable exp264 Stage B v2 score byte SHAを
# fail-closed照合する。K=3 assignment、centroid、membershipは読み込まない。

# %%
if EXECUTE_NOTEBOOK:
    search_roots = [Path("/kaggle/input"), Path("/tmp"), root]
    signature_path = resolve_input(
        [str(item) for item in config["data"]["exp267_signature_patterns"]],
        search_roots,
    )
    score_path = resolve_input(
        [str(item) for item in config["data"]["exp264_candidate_score_patterns"]],
        search_roots,
    )
    signature_sha = sha256_file(signature_path)
    score_sha = sha256_file(score_path)
    if signature_sha != config["data"]["exp267_expected_signature_sha256"]:
        raise ValueError(
            "exp267 signature SHA mismatch: "
            f"expected={config['data']['exp267_expected_signature_sha256']} actual={signature_sha}"
        )
    if score_sha != config["data"]["exp264_expected_candidate_score_sha256"]:
        raise ValueError(
            "exp264 candidate score SHA mismatch: "
            f"expected={config['data']['exp264_expected_candidate_score_sha256']} "
            f"actual={score_sha}"
        )
    signatures = pd.read_parquet(signature_path).sort_values(
        ["outer_fold", "well"], kind="stable"
    ).reset_index(drop=True)
    signature_logical_sha = logical_frame_sha256(signatures)
    if signature_logical_sha != config["data"][
        "exp267_expected_signature_logical_sha256"
    ]:
        raise ValueError(
            "exp267 signature logical SHA mismatch: "
            f"expected={config['data']['exp267_expected_signature_logical_sha256']} "
            f"actual={signature_logical_sha}"
        )
    input_evidence = {
        "exp267_signature_path": str(signature_path),
        "exp267_signature_sha256": signature_sha,
        "exp267_signature_logical_sha256": signature_logical_sha,
        "exp264_candidate_score_path": str(score_path),
        "exp264_candidate_score_sha256": score_sha,
        "signature_rows": int(len(signatures)),
        "signature_wells": int(signatures["well"].nunique()),
        "signature_folds": int(signatures["outer_fold"].nunique()),
        "signature_features": len(signature_feature_columns()),
        "primary_features": len(primary_feature_columns()),
    }
    display(input_evidence)
    display(signatures.head())


# %% [markdown]
# ## 4. Outer-fold continuous divergence axes
#
# `fixed_range_gap_axis`は12個の正方向range/gap特徴をouter-train median、RobustScaler、
# `[-10,10]` clip後に等重み平均する。`pca1_axis`は全18特徴を同じouter-trainだけでfitし、
# outer-train primaryとの相関が正になるよう符号を固定する。

# %%
if EXECUTE_NOTEBOOK:
    axes, preprocessors = fit_oof_continuous_axes(signatures, config)
    display(axes.head())
    display(
        axes.groupby("outer_fold")[list(AXIS_COLUMNS)].agg(["count", "mean", "std", "min", "max"])
    )
    preprocessor_overview = pd.DataFrame(
        [
            {
                "outer_fold": item["outer_fold"],
                "outer_train_wells": item["outer_train_wells"],
                "outer_valid_wells": item["outer_valid_wells"],
                "pca_explained_variance_ratio": item["pca_explained_variance_ratio"],
                "pca_orientation": item["pca_orientation"],
                "pca_train_spearman_after_orientation": item[
                    "pca_outer_train_spearman_after_orientation"
                ],
            }
            for item in preprocessors
        ]
    )
    display(preprocessor_overview)


# %% [markdown]
# ## 5. Streaming exp264 candidate score aggregation
#
# exp264 candidate-long Parquetを500k-row batchで読み、固定6 primitiveだけを
# well×candidateへ集約する。
# actual MAE / calibration biasはaxis fit完了後に初めて接続する。candidate-bank outcomeはwell内で
# 6 candidatesを等重み平均し、row数の多いwellが相関を支配しないようwell単位で評価する。

# %%
if EXECUTE_NOTEBOOK:
    candidate_metrics, stream_evidence = stream_candidate_well_metrics(
        score_path,
        signatures,
        [str(item) for item in config["candidate_bank"]["primitive_ids"]],
        batch_size=int(config["runtime"]["batch_size"]),
        expected_rows_per_candidate=int(
            config["guards"]["technical"]["expected_rows_per_candidate"]
        ),
    )
    if len(candidate_metrics) != int(
        config["guards"]["technical"]["expected_well_candidate_rows"]
    ):
        raise ValueError("unexpected well-candidate metric row count")
    display(stream_evidence)
    display(candidate_metrics.head(12))


# %% [markdown]
# ## 6. Spearman and stratified well-bootstrap readout
#
# fold/pooled Spearmanはprimary/PCA1 × actual/calibration × bank/candidateを全て保存する。
# guardはprimary candidate-bank meanだけを使い、actual正方向5/5、calibration負方向5/5、
# stratified bootstrap intervalの非自明な効果下限を全て要求する。

# %%
if EXECUTE_NOTEBOOK:
    tables = build_readout_tables(axes, candidate_metrics, config)
    primary_bank_correlations = tables["correlations"].query(
        "scope == 'candidate_bank_mean' and axis == 'fixed_range_gap_axis'"
    )
    display(primary_bank_correlations)
    display(tables["bootstrap"])
    display(tables["quantile_metrics"])
    display(tables["guard"])


# %% [markdown]
# ## 7. Diagnostics, artifacts, and reproducibility evidence

# %%
if EXECUTE_NOTEBOOK:
    by_well = tables["by_well"]
    quantile_metrics = tables["quantile_metrics"]
    correlations = tables["correlations"]
    figure, axes_plot = plt.subplots(2, 2, figsize=(15, 10))
    axes_plot[0, 0].scatter(
        by_well["fixed_range_gap_axis"], by_well["actual_mae"], s=12, alpha=0.45
    )
    axes_plot[0, 0].set_title("Primary divergence axis vs candidate-bank actual MAE")
    axes_plot[0, 0].set_xlabel("fixed_range_gap_axis")
    axes_plot[0, 0].set_ylabel("actual_mae")
    axes_plot[0, 1].scatter(
        by_well["fixed_range_gap_axis"],
        by_well["calibration_bias"],
        s=12,
        alpha=0.45,
    )
    axes_plot[0, 1].axhline(0.0, color="black", linewidth=1)
    axes_plot[0, 1].set_title("Primary divergence axis vs calibration bias")
    axes_plot[0, 1].set_xlabel("fixed_range_gap_axis")
    axes_plot[0, 1].set_ylabel("calibration_bias")

    primary_quantiles = quantile_metrics.query("axis == 'fixed_range_gap_axis'")
    axes_plot[1, 0].plot(
        primary_quantiles["axis_quantile"],
        primary_quantiles["actual_mae"],
        marker="o",
        label="actual MAE",
    )
    axes_plot[1, 0].plot(
        primary_quantiles["axis_quantile"],
        primary_quantiles["predicted_abs_error_mean"],
        marker="o",
        label="predicted abs error",
    )
    axes_plot[1, 0].set_title("Primary-axis decile readout")
    axes_plot[1, 0].set_xlabel("axis decile (low to high divergence)")
    axes_plot[1, 0].legend()

    fold_plot = correlations.query(
        "scope == 'candidate_bank_mean' and outer_fold != 'all'"
    ).copy()
    fold_plot["series"] = fold_plot["axis"] + " / " + fold_plot["outcome"]
    fold_pivot = fold_plot.pivot(index="outer_fold", columns="series", values="spearman")
    fold_pivot.plot.bar(ax=axes_plot[1, 1])
    axes_plot[1, 1].axhline(0.0, color="black", linewidth=1)
    axes_plot[1, 1].set_title("Fold Spearman — bank mean")
    axes_plot[1, 1].set_ylabel("Spearman")
    axes_plot[1, 1].legend(fontsize=7)
    plt.tight_layout()
    plot_path = output_dir / "continuous_well_divergence_risk_readout.png"
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.show()

# %%
if EXECUTE_NOTEBOOK:
    summary = save_readout_artifacts(
        output_dir=output_dir,
        axes=axes,
        preprocessors=preprocessors,
        tables=tables,
        input_evidence=input_evidence,
        stream_evidence=stream_evidence,
        config=config,
        plot_path=plot_path,
    )
    display(summary)
    print("Generated artifacts")
    for name in config["artifacts"]["required"]:
        path = output_dir / str(name)
        size = path.stat().st_size if path.exists() else 0
        print(f"- {name}: exists={path.exists()} bytes={size}")
        assert path.exists() and path.stat().st_size > 0
    if summary["guard"]["continuous_risk_guard_pass"]:
        print(
            "Primary continuous-risk guard PASS. This only supports designing a separate "
            "add-only experiment; no training or inference is allowed here."
        )
    else:
        print(
            "Primary continuous-risk guard FAIL. Keep the exp267 K=3 branch closed and do not "
            "rescue it with PCA1 or candidate-specific results."
        )

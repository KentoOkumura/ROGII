# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp238 OOF and selector-confidence plots
#
# Diagnostic visualization notebook derived from the exp083 v12 all-well plot.
# It replaces the exp148 OOF overlay with the saved exp238 final `lgb_mean` OOF
# and makes the strict nested selector decision visible row by row.
#
# The exp238 selector predicts candidate absolute error. Therefore:
#
# - selector top-1 = candidate with the **lowest predicted absolute error**;
# - confidence margin = second-lowest score minus lowest score;
# - a larger margin means the selector separated its preferred candidate more clearly.
#
# Only the score file in which a row has `role=valid` is used. The top-1 path is
# a diagnostic overlay, not the exp238 final prediction: exp238 uses selector
# rank/slot values as add-only LightGBM features. The historical selector safety
# guard failed on worst-well regression, so this notebook must not be read as a
# recommendation for direct top-1 replacement.

# %% [markdown]
# ## Contents
#
# 1. Imports and configuration
# 2. Path resolution and input contract
# 3. Base candidates and saved OOF loading
# 4. Strict outer-valid selector surface
# 5. Metrics and plot helpers
# 6. Generate all-well plots
# 7. Summary and generated outputs

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from IPython.display import display


EXPERIMENT_NAME = "exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218"
OUTPUT_PREFIX = f"{EXPERIMENT_NAME}_oof_selector_confidence_probe"

PFBEAM_FILENAME = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
EXP238_OOF_FILENAME = f"{EXPERIMENT_NAME}_final_oof_predictions.csv.gz"
SELECTOR_SUMMARY_FILENAME = f"{EXPERIMENT_NAME}_selector_summary.json"
SELECTOR_SCORE_TEMPLATE = f"{EXPERIMENT_NAME}_nested_scores_outer{{outer_fold}}.csv.gz"
EXP209_HMM_FILENAME = (
    "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_"
    "amerhu_exact_hmm_smoother_default_train_features.csv.gz"
)
EXP223_SELFGR_FILENAME = (
    "exp223_joint_typewell_self_gr_hmm_likelihood_probe_"
    "joint_typewell_self_gr_hmm_likelihood_probe_train_features.csv.gz"
)
EXP226_OOF_FILENAME = (
    "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_"
    "train_oof_predictions.csv.gz"
)
COMMON_TYPEWELL_ASSIGNMENTS_FILENAME = "common_typewell_cluster_assignments.csv"

EXP238_RECORDED_CV_RMSE = 7.936689853668213
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
COMMON_TYPEWELL_METHOD = "native_overlap"
COMMON_TYPEWELL_THRESHOLD = "0.999"
EXPECTED_COMMON_TYPEWELL_GROUPS = 54
EXPECTED_CANDIDATES = [
    "pf_ancc",
    "beam_mean",
    "likpf_mean",
    "sc_ens",
    "hyb",
    "tvt_dense",
    "tvt_densew",
    "tvt_dense50",
    "blend_likpf_hmm_w500",
    "hmm_selfgr_boost_only_a070_c100_mean_tvt",
    "exp226_v6_k16_geometry_gr_u_projection",
]
CANDIDATE_LABELS = {
    "pf_ancc": "PF ANCC",
    "beam_mean": "Beam mean",
    "likpf_mean": "Likelihood PF mean",
    "sc_ens": "SC ensemble",
    "hyb": "Hybrid",
    "tvt_dense": "Dense",
    "tvt_densew": "Dense weighted",
    "tvt_dense50": "Dense 50",
    "blend_likpf_hmm_w500": "LikPF/HMM 50:50",
    "hmm_selfgr_boost_only_a070_c100_mean_tvt": "Self-GR HMM",
    "exp226_v6_k16_geometry_gr_u_projection": "exp226 K16",
}
REFERENCE_LINE_COLORS = {
    # Keep the overlapping series identical to exp083 v12.
    "true_tvt": "black",
    "ml_oof": "#e11d48",
    "selector_top1": "#64748b",
    "pf_ancc": "#1f77b4",
    "beam_mean": "#ff7f0e",
    "likpf_mean": "#2ca02c",
    "exp226_k16": "#a16207",
    "exp209_hmm": "#7c3aed",
    "exp209_hmm_band": "#8b5cf6",
    "z_likpf_minmax": "#db2777",
    "confidence_margin": "#0f172a",
    "grid": "#e2e8f0",
    "caveat": "#7f1d1d",
}
CANDIDATE_COLOR_BY_NAME = {
    # Shared candidates use their exp083 v12 line colors.
    "pf_ancc": REFERENCE_LINE_COLORS["pf_ancc"],
    "beam_mean": REFERENCE_LINE_COLORS["beam_mean"],
    "likpf_mean": REFERENCE_LINE_COLORS["likpf_mean"],
    # Candidates absent from the exp083 main panel use stable Tableau colors.
    "sc_ens": "#9467bd",
    "hyb": "#8c564b",
    "tvt_dense": "#e377c2",
    "tvt_densew": "#7f7f7f",
    "tvt_dense50": "#bcbd22",
    "blend_likpf_hmm_w500": "#17becf",
    "hmm_selfgr_boost_only_a070_c100_mean_tvt": REFERENCE_LINE_COLORS[
        "exp209_hmm"
    ],
    "exp226_v6_k16_geometry_gr_u_projection": REFERENCE_LINE_COLORS[
        "exp226_k16"
    ],
}
CANDIDATE_COLORS = [CANDIDATE_COLOR_BY_NAME[name] for name in EXPECTED_CANDIDATES]

MAX_POINTS_PER_PLOT = 6000
READ_CHUNKSIZE = 300_000
ZIP_PLOTS = True
TVT_AXIS_INVERTED = True
MAX_PLOTS_ENV = os.environ.get("EXPERIMENT_MAX_PLOTS")
MAX_PLOTS = int(MAX_PLOTS_ENV) if MAX_PLOTS_ENV else None

print("Experiment:", EXPERIMENT_NAME)
print("Output prefix:", OUTPUT_PREFIX)
print("Recorded exp238 lgb_mean CV RMSE:", EXP238_RECORDED_CV_RMSE)
print("Selector top-1 definition: minimum predicted candidate absolute error")
print("Selector confidence margin: predicted error top2 - top1")
print("Selector score role: outer-valid only")
print("Direct selector replacement: no (diagnostic overlay only)")
print("Historical selector worst-well guard: failed")
print("Debug max plots override:", MAX_PLOTS)

# %% [markdown]
# ## 2. Path resolution and input contract

# %%
def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "experiment_summary.md").exists() and (
            candidate / "experiments"
        ).exists():
            return candidate
    return current


REPO_ROOT = find_repo_root(Path.cwd())
EXP_DIR = REPO_ROOT / "experiments" / EXPERIMENT_NAME
if not EXP_DIR.exists() and Path.cwd().name == EXPERIMENT_NAME:
    EXP_DIR = Path.cwd()

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
ARTIFACTS_DIR = (
    KAGGLE_WORKING_ROOT if KAGGLE_WORKING_ROOT.exists() else EXP_DIR
) / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_plots"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path
    return None


def resolve_input(
    *,
    filename: str,
    local_candidates: list[Path],
    preferred_slugs: list[str],
) -> Path:
    local = _existing(local_candidates)
    if local is not None:
        return local

    if KAGGLE_INPUT_ROOT.exists():
        preferred_roots = [KAGGLE_INPUT_ROOT / slug for slug in preferred_slugs]
        generic_roots = [
            path for path in sorted(KAGGLE_INPUT_ROOT.iterdir()) if path.is_dir()
        ]
        seen: set[Path] = set()
        for root in [*preferred_roots, *generic_roots]:
            if root in seen or not root.exists():
                continue
            seen.add(root)
            matches = sorted(root.rglob(filename))
            if matches:
                return matches[0]

    checked = "\n".join(str(path) for path in local_candidates)
    raise FileNotFoundError(
        f"{filename} not found. Checked local paths:\n{checked}\n"
        f"Kaggle slugs: {preferred_slugs}"
    )


pfbeam_path = resolve_input(
    filename=PFBEAM_FILENAME,
    local_candidates=[
        REPO_ROOT
        / "experiments"
        / "exp072_exp063_full_replay_feature_cache"
        / "artifacts"
        / PFBEAM_FILENAME,
        Path("/tmp/kaggle-output/exp072_exp063_full_replay_feature_cache/train/artifacts")
        / PFBEAM_FILENAME,
    ],
    preferred_slugs=["exp072-exp063-full-replay-feature-cache-train"],
)
exp238_oof_path = resolve_input(
    filename=EXP238_OOF_FILENAME,
    local_candidates=[
        EXP_DIR / "artifacts" / EXP238_OOF_FILENAME,
        EXP_DIR / "kaggle" / "output" / "train_v5" / "artifacts" / EXP238_OOF_FILENAME,
        Path("/tmp/kaggle-output") / EXPERIMENT_NAME / "train_v5" / "artifacts" / EXP238_OOF_FILENAME,
    ],
    preferred_slugs=["exp238-nested-rank-slot-exp218-train"],
)
selector_summary_path = resolve_input(
    filename=SELECTOR_SUMMARY_FILENAME,
    local_candidates=[
        EXP_DIR
        / "kaggle"
        / "output"
        / "selector_v3"
        / "artifacts"
        / SELECTOR_SUMMARY_FILENAME,
        EXP_DIR / "artifacts" / SELECTOR_SUMMARY_FILENAME,
    ],
    preferred_slugs=["exp238-nested-selector-train"],
)
selector_score_paths = [
    resolve_input(
        filename=SELECTOR_SCORE_TEMPLATE.format(outer_fold=outer_fold),
        local_candidates=[
            selector_summary_path.parent
            / SELECTOR_SCORE_TEMPLATE.format(outer_fold=outer_fold),
            EXP_DIR
            / "kaggle"
            / "output"
            / "selector_v3"
            / "artifacts"
            / SELECTOR_SCORE_TEMPLATE.format(outer_fold=outer_fold),
            EXP_DIR
            / "artifacts"
            / SELECTOR_SCORE_TEMPLATE.format(outer_fold=outer_fold),
        ],
        preferred_slugs=["exp238-nested-selector-train"],
    )
    for outer_fold in range(5)
]
exp209_hmm_path = resolve_input(
    filename=EXP209_HMM_FILENAME,
    local_candidates=[
        Path(
            "/tmp/kaggle-output/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/"
            "train_v5/artifacts"
        )
        / EXP209_HMM_FILENAME,
    ],
    preferred_slugs=["exp209-joint-exact-parity-train"],
)
exp223_selfgr_path = resolve_input(
    filename=EXP223_SELFGR_FILENAME,
    local_candidates=[
        Path("/tmp/kaggle-output/exp223-selfgr-hmm-train-v1/artifacts")
        / EXP223_SELFGR_FILENAME,
    ],
    preferred_slugs=["exp223-selfgr-hmm-train"],
)
exp226_oof_path = resolve_input(
    filename=EXP226_OOF_FILENAME,
    local_candidates=[
        Path(
            "/tmp/kaggle-output/"
            "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/"
            "train_v1/artifacts"
        )
        / EXP226_OOF_FILENAME,
    ],
    preferred_slugs=["exp226-k16-kappa-repro-train"],
)
common_typewell_assignments_path = resolve_input(
    filename=COMMON_TYPEWELL_ASSIGNMENTS_FILENAME,
    local_candidates=[
        REPO_ROOT
        / "experiments"
        / "exp065_typewell_supertype_cluster_cv_audit"
        / "artifacts"
        / COMMON_TYPEWELL_ASSIGNMENTS_FILENAME,
        Path("/tmp/kaggle-output/exp065-typewell-supertype-cluster-cv-audit-train/artifacts")
        / COMMON_TYPEWELL_ASSIGNMENTS_FILENAME,
    ],
    preferred_slugs=["exp065-typewell-supertype-cluster-cv-audit-train"],
)

print("Repo root:", REPO_ROOT)
print("Artifacts dir:", ARTIFACTS_DIR)
print("PF/Beam source:", pfbeam_path)
print("exp238 OOF source:", exp238_oof_path)
print("selector summary:", selector_summary_path)
print("selector score sources:", [str(path) for path in selector_score_paths])
print("exp209 exact-HMM source:", exp209_hmm_path)
print("exp223 self-GR HMM source:", exp223_selfgr_path)
print("exp226 K16 source:", exp226_oof_path)
print("Common typewell assignments:", common_typewell_assignments_path)

# %% [markdown]
# ## 3. Base candidates and saved OOF loading

# %%
def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    if decompressed and path.suffix == ".gz":
        stream_context = gzip.open(path, "rb")
    else:
        stream_context = path.open("rb")
    with stream_context as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse_values(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not bool(valid.any()):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(prediction[valid] - truth[valid]))))


def read_common_typewell_order(path: Path, wells: list[str]) -> pd.DataFrame:
    required = {
        "method",
        "threshold",
        "cluster_id",
        "well_id",
        "cluster_size",
        "representative_well_id",
    }
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(required.difference(header))
    if missing:
        raise ValueError(f"common typewell assignments are missing columns: {missing}")

    assignments = pd.read_csv(path, usecols=sorted(required), dtype=str)
    selected = assignments.loc[
        (assignments["method"] == COMMON_TYPEWELL_METHOD)
        & (assignments["threshold"] == COMMON_TYPEWELL_THRESHOLD)
    ].copy()
    if selected.empty:
        raise ValueError(
            "common typewell assignments do not contain "
            f"method={COMMON_TYPEWELL_METHOD}, threshold={COMMON_TYPEWELL_THRESHOLD}"
        )
    if selected["well_id"].duplicated().any():
        duplicate_wells = sorted(
            selected.loc[selected["well_id"].duplicated(keep=False), "well_id"].unique()
        )
        raise ValueError(f"common typewell assignments duplicate wells: {duplicate_wells[:10]}")

    expected_wells = {str(well) for well in wells}
    observed_wells = set(selected["well_id"].astype(str))
    if observed_wells != expected_wells:
        raise ValueError(
            {
                "message": "common typewell assignment coverage differs from plot wells",
                "missing_wells": sorted(expected_wells - observed_wells)[:20],
                "unexpected_wells": sorted(observed_wells - expected_wells)[:20],
                "expected_wells": len(expected_wells),
                "observed_wells": len(observed_wells),
            }
        )

    selected["cluster_size"] = pd.to_numeric(
        selected["cluster_size"], errors="raise"
    ).astype(np.int64)
    selected = selected.sort_values(
        ["cluster_id", "well_id"], kind="mergesort"
    ).reset_index(drop=True)
    cluster_ids = selected["cluster_id"].drop_duplicates().tolist()
    if len(cluster_ids) != EXPECTED_COMMON_TYPEWELL_GROUPS:
        raise ValueError(
            f"common typewell group count {len(cluster_ids)} != "
            f"{EXPECTED_COMMON_TYPEWELL_GROUPS}"
        )
    cluster_order = {
        cluster_id: order for order, cluster_id in enumerate(cluster_ids, start=1)
    }
    selected["typewell_order"] = (
        selected["cluster_id"].map(cluster_order).astype(np.int64)
    )
    selected["well_order_within_typewell"] = (
        selected.groupby("cluster_id", sort=False).cumcount() + 1
    ).astype(np.int64)
    selected["plot_order"] = np.arange(1, len(selected) + 1, dtype=np.int64)

    observed_cluster_sizes = selected.groupby("cluster_id")["well_id"].transform(
        "size"
    )
    if not np.array_equal(
        selected["cluster_size"].to_numpy(np.int64),
        observed_cluster_sizes.to_numpy(np.int64),
    ):
        raise ValueError("common typewell declared cluster_size does not match assignments")

    selected = selected.rename(
        columns={
            "method": "typewell_method",
            "threshold": "typewell_threshold",
            "cluster_id": "typewell_cluster_id",
            "cluster_size": "typewell_cluster_size",
            "representative_well_id": "typewell_representative_well_id",
            "well_id": "well",
        }
    )
    selected["plot_filename"] = [
        f"typewell_{typewell_order:04d}_{well}.png"
        for typewell_order, well in zip(
            selected["typewell_order"], selected["well"], strict=True
        )
    ]
    if selected["plot_filename"].duplicated().any():
        raise ValueError("common typewell plot filenames are not unique")
    return selected[
        [
            "plot_order",
            "typewell_order",
            "well_order_within_typewell",
            "typewell_method",
            "typewell_threshold",
            "typewell_cluster_id",
            "typewell_cluster_size",
            "typewell_representative_well_id",
            "well",
            "plot_filename",
        ]
    ]


def read_base_candidates(path: Path) -> pd.DataFrame:
    needed = {
        "id",
        "well",
        "target",
        "last_known_tvt",
        "md_since",
        "z",
        "dzdmd",
        "pf_ancc",
        "beam_mean_d",
        "likpf_mean_d",
        "sc_ens_d",
        "hyb_d",
        "tvt_dense_d",
        "tvt_densew_d",
        "tvt_dense50_d",
    }
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(needed.difference(header))
    if missing:
        raise ValueError(f"PF/Beam cache is missing columns: {missing}")
    frame = pd.read_csv(
        path,
        usecols=lambda column: column in needed,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    if frame.duplicated(["id", "well"]).any():
        raise ValueError("PF/Beam cache has duplicate id/well rows")
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in needed.difference({"id", "well"}):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    anchor = frame["last_known_tvt"].to_numpy(np.float32)
    frame["true_tvt"] = anchor + frame["target"].to_numpy(np.float32)
    frame["beam_mean"] = anchor + frame["beam_mean_d"].to_numpy(np.float32)
    frame["likpf_mean"] = anchor + frame["likpf_mean_d"].to_numpy(np.float32)
    frame["sc_ens"] = anchor + frame["sc_ens_d"].to_numpy(np.float32)
    frame["hyb"] = anchor + frame["hyb_d"].to_numpy(np.float32)
    frame["tvt_dense"] = anchor + frame["tvt_dense_d"].to_numpy(np.float32)
    frame["tvt_densew"] = anchor + frame["tvt_densew_d"].to_numpy(np.float32)
    frame["tvt_dense50"] = anchor + frame["tvt_dense50_d"].to_numpy(np.float32)
    return frame


def read_order_aligned_columns(
    path: Path,
    base: pd.DataFrame,
    *,
    source_name: str,
    value_columns: list[str],
) -> dict[str, np.ndarray]:
    usecols = ["id", "well", *value_columns]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(usecols).difference(header))
    if missing:
        raise ValueError(f"{source_name} is missing columns: {missing}")
    outputs = {
        column: np.full(len(base), np.nan, dtype=np.float32) for column in value_columns
    }
    base_ids = base["id"].to_numpy(dtype=str)
    base_wells = base["well"].to_numpy(dtype=str)
    offset = 0
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"id": str, "well": str},
        chunksize=READ_CHUNKSIZE,
        low_memory=False,
    ):
        stop = offset + len(chunk)
        if stop > len(base):
            raise ValueError(f"{source_name} contains more rows than the base cache")
        if not np.array_equal(chunk["id"].astype(str).to_numpy(), base_ids[offset:stop]):
            raise ValueError(f"{source_name} id order differs from the base cache at row {offset}")
        if not np.array_equal(
            chunk["well"].astype(str).to_numpy(), base_wells[offset:stop]
        ):
            raise ValueError(
                f"{source_name} well order differs from the base cache at row {offset}"
            )
        for column in value_columns:
            outputs[column][offset:stop] = pd.to_numeric(
                chunk[column], errors="coerce"
            ).to_numpy(np.float32)
        offset = stop
    if offset != len(base):
        raise ValueError(f"{source_name} row count {offset} != base row count {len(base)}")
    return outputs


def read_exp226_candidate(path: Path, base: pd.DataFrame) -> np.ndarray:
    usecols = ["well_id", "row_idx", "tvt_pred"]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(usecols).difference(header))
    if missing:
        raise ValueError(f"exp226 OOF is missing columns: {missing}")
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"well_id": str},
        chunksize=READ_CHUNKSIZE,
        low_memory=False,
    ):
        row_idx = pd.to_numeric(chunk["row_idx"], errors="coerce")
        if row_idx.isna().any():
            raise ValueError("exp226 row_idx contains non-numeric values")
        well = chunk["well_id"].astype(str)
        chunks.append(
            pd.DataFrame(
                {
                    "id": well + "_" + row_idx.astype(np.int64).astype(str),
                    "well": well,
                    "exp226_v6_k16_geometry_gr_u_projection": pd.to_numeric(
                        chunk["tvt_pred"], errors="coerce"
                    ).astype(np.float32),
                }
            )
        )
    candidate = pd.concat(chunks, ignore_index=True)
    if candidate.duplicated(["id", "well"]).any():
        raise ValueError("exp226 OOF has duplicate id/well rows")
    aligned = base[["id", "well"]].merge(
        candidate,
        on=["id", "well"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    values = aligned["exp226_v6_k16_geometry_gr_u_projection"].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("exp226 candidate does not fully cover the base cache")
    return values


base = read_base_candidates(pfbeam_path)
if len(base) != EXPECTED_ROWS or base["well"].nunique() != EXPECTED_WELLS:
    raise ValueError(
        {
            "message": "exp238 plot base row/well contract changed",
            "rows": len(base),
            "wells": int(base["well"].nunique()),
            "expected_rows": EXPECTED_ROWS,
            "expected_wells": EXPECTED_WELLS,
        }
    )
common_typewell_order = read_common_typewell_order(
    common_typewell_assignments_path,
    base["well"].dropna().astype(str).unique().tolist(),
)
print("Common typewell groups:", int(common_typewell_order["typewell_order"].nunique()))
print("First common-typewell plot files:", common_typewell_order["plot_filename"].head(10).tolist())

exp238_oof = read_order_aligned_columns(
    exp238_oof_path,
    base,
    source_name="exp238 final OOF",
    value_columns=["lgb_mean_pred_tvt"],
)
base["exp238_lgb_mean_oof_tvt"] = exp238_oof["lgb_mean_pred_tvt"]

exp209_values = read_order_aligned_columns(
    exp209_hmm_path,
    base,
    source_name="exp209 exact HMM",
    value_columns=["hmm_mean_tvt", "hmm_std"],
)
base["hmm_exact_mean_tvt"] = exp209_values["hmm_mean_tvt"]
base["hmm_exact_std"] = exp209_values["hmm_std"]
base["blend_likpf_hmm_w500"] = (
    0.5 * base["likpf_mean"].to_numpy(np.float32)
    + 0.5 * base["hmm_exact_mean_tvt"].to_numpy(np.float32)
).astype(np.float32)

exp223_values = read_order_aligned_columns(
    exp223_selfgr_path,
    base,
    source_name="exp223 self-GR HMM",
    value_columns=["hmm_selfgr_boost_only_a070_c100_mean_tvt"],
)
base["hmm_selfgr_boost_only_a070_c100_mean_tvt"] = exp223_values[
    "hmm_selfgr_boost_only_a070_c100_mean_tvt"
]
base["exp226_v6_k16_geometry_gr_u_projection"] = read_exp226_candidate(
    exp226_oof_path, base
)

for column in [*EXPECTED_CANDIDATES, "exp238_lgb_mean_oof_tvt"]:
    values = base[column].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} contains non-finite values")

print("Base rows:", len(base))
print("Base wells:", int(base["well"].nunique()))
print(
    "exp238 OOF RMSE:",
    rmse_values(base["true_tvt"].to_numpy(), base["exp238_lgb_mean_oof_tvt"].to_numpy()),
)

# %% [markdown]
# ## 4. Strict outer-valid selector surface

# %%
with selector_summary_path.open("r", encoding="utf-8") as stream:
    selector_summary = json.load(stream)

candidate_columns = [str(value) for value in selector_summary["candidate_columns"]]
if candidate_columns != EXPECTED_CANDIDATES:
    raise ValueError(
        {
            "message": "selector candidate order changed",
            "expected": EXPECTED_CANDIDATES,
            "actual": candidate_columns,
        }
    )

declared_score_contract = {
    int(item["outer_fold"]): {
        "file": str(item["file"]),
        "rows": int(item["rows"]),
        "valid_rows": int(item["valid_rows"]),
        "sha256_decompressed": str(item["sha256_decompressed"]),
    }
    for item in selector_summary["score_artifacts"]
}
if sorted(declared_score_contract) != list(range(5)):
    raise ValueError("selector summary does not declare outer folds 0..4")


def load_outer_valid_selector_surface(
    frame: pd.DataFrame,
    score_paths: list[Path],
    candidates: list[str],
) -> dict[str, np.ndarray]:
    n_rows = len(frame)
    top1_code = np.full(n_rows, -1, dtype=np.int16)
    top1_score = np.full(n_rows, np.nan, dtype=np.float32)
    confidence_margin = np.full(n_rows, np.nan, dtype=np.float32)
    outer_fold = np.full(n_rows, -1, dtype=np.int8)
    coverage = np.zeros(n_rows, dtype=np.uint8)
    base_ids = frame["id"].to_numpy(dtype=str)
    base_wells = frame["well"].to_numpy(dtype=str)
    score_columns = [f"pred_error__{name}" for name in candidates]

    for fold, path in enumerate(score_paths):
        expected_filename = declared_score_contract[fold]["file"]
        if path.name != expected_filename:
            raise ValueError(
                f"outer {fold} score filename {path.name} != declared {expected_filename}"
            )
        fold_valid_rows = 0
        for chunk in pd.read_csv(
            path,
            usecols=["row_index", "role", "id", "well", *score_columns],
            dtype={
                "row_index": np.int32,
                "role": str,
                "id": str,
                "well": str,
                **{column: np.float32 for column in score_columns},
            },
            chunksize=READ_CHUNKSIZE,
            low_memory=False,
        ):
            valid = chunk.loc[chunk["role"].eq("valid")]
            if valid.empty:
                continue
            rows = valid["row_index"].to_numpy(np.int64)
            if rows.min() < 0 or rows.max() >= n_rows:
                raise ValueError(f"outer {fold} selector row_index is out of range")
            if bool(np.any(coverage[rows] != 0)):
                raise ValueError(f"outer {fold} overlaps a previous outer-valid row")
            if not np.array_equal(valid["id"].astype(str).to_numpy(), base_ids[rows]):
                raise ValueError(f"outer {fold} selector id/row_index contract failed")
            if not np.array_equal(
                valid["well"].astype(str).to_numpy(), base_wells[rows]
            ):
                raise ValueError(f"outer {fold} selector well/row_index contract failed")
            scores = valid[score_columns].to_numpy(np.float32)
            if not np.isfinite(scores).all():
                raise ValueError(f"outer {fold} selector valid scores contain non-finite values")
            pair = np.argpartition(scores, kth=1, axis=1)[:, :2]
            pair_scores = np.take_along_axis(scores, pair, axis=1)
            swap = pair_scores[:, 1] < pair_scores[:, 0]
            first = np.where(swap, pair[:, 1], pair[:, 0])
            second = np.where(swap, pair[:, 0], pair[:, 1])
            first_score = scores[np.arange(len(scores)), first]
            second_score = scores[np.arange(len(scores)), second]
            margin = second_score - first_score
            if bool(np.any(margin < -1e-6)):
                raise ValueError(f"outer {fold} selector confidence margin became negative")
            top1_code[rows] = first.astype(np.int16)
            top1_score[rows] = first_score.astype(np.float32)
            confidence_margin[rows] = np.maximum(margin, 0.0).astype(np.float32)
            outer_fold[rows] = np.int8(fold)
            coverage[rows] = 1
            fold_valid_rows += len(rows)
        expected_valid_rows = declared_score_contract[fold]["valid_rows"]
        if fold_valid_rows != expected_valid_rows:
            raise ValueError(
                f"outer {fold} valid rows {fold_valid_rows} != declared {expected_valid_rows}"
            )
        print(f"outer {fold}: loaded {fold_valid_rows:,} strict valid rows")

    if not bool(np.all(coverage == 1)):
        values, counts = np.unique(coverage, return_counts=True)
        raise ValueError(
            f"selector outer-valid coverage failed: {dict(zip(values.tolist(), counts.tolist()))}"
        )
    if not np.isfinite(top1_score).all() or not np.isfinite(confidence_margin).all():
        raise ValueError("selector top1 score or confidence margin is incomplete")
    if top1_code.min() < 0 or top1_code.max() >= len(candidates):
        raise ValueError("selector top1 code is out of candidate range")
    return {
        "top1_code": top1_code,
        "top1_score": top1_score,
        "confidence_margin": confidence_margin,
        "outer_fold": outer_fold,
    }


selector_surface = load_outer_valid_selector_surface(
    base, selector_score_paths, candidate_columns
)
base["selector_top1_code"] = selector_surface["top1_code"]
base["selector_pred_error_top1"] = selector_surface["top1_score"]
base["selector_confidence_margin"] = selector_surface["confidence_margin"]
base["selector_outer_fold"] = selector_surface["outer_fold"]

selected_tvt = np.full(len(base), np.nan, dtype=np.float32)
for code, column in enumerate(candidate_columns):
    mask = selector_surface["top1_code"] == code
    selected_tvt[mask] = base.loc[mask, column].to_numpy(np.float32)
if not np.isfinite(selected_tvt).all():
    raise ValueError("selector top1 candidate TVT is incomplete")
base["selector_top1_tvt"] = selected_tvt

selector_counts = np.bincount(
    selector_surface["top1_code"], minlength=len(candidate_columns)
)
selector_distribution = pd.DataFrame(
    {
        "candidate_code": np.arange(len(candidate_columns), dtype=np.int16),
        "candidate": candidate_columns,
        "label": [CANDIDATE_LABELS[name] for name in candidate_columns],
        "rows": selector_counts,
        "share": selector_counts / len(base),
    }
).sort_values("rows", ascending=False)
display(selector_distribution)

# %% [markdown]
# ## 5. Metrics and plot helpers

# %%
def fmt_metric(value: Any, digits: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.{digits}f}"


def downsample_for_plot(group: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(group) <= max_points:
        return group
    positions = np.unique(
        np.linspace(0, len(group) - 1, max_points, dtype=np.int64)
    )
    return group.iloc[positions]


def minmax_negative_z_to_likpf(group: pd.DataFrame) -> np.ndarray:
    source = -pd.to_numeric(group["z"], errors="coerce").to_numpy(np.float64)
    target = pd.to_numeric(group["likpf_mean"], errors="coerce").to_numpy(np.float64)
    valid_source = np.isfinite(source)
    valid_target = np.isfinite(target)
    output = np.full(len(group), np.nan, dtype=np.float64)
    if not valid_source.any() or not valid_target.any():
        return output
    source_min = float(np.nanmin(source[valid_source]))
    source_max = float(np.nanmax(source[valid_source]))
    target_min = float(np.nanmin(target[valid_target]))
    target_max = float(np.nanmax(target[valid_target]))
    if source_max - source_min <= 1e-12:
        output[valid_source] = 0.5 * (target_min + target_max)
    else:
        output[valid_source] = target_min + (
            (source[valid_source] - source_min) / (source_max - source_min)
        ) * (target_max - target_min)
    return output


def x_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5], dtype=np.float64)
    middle = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (middle[0] - values[0])
    last = values[-1] + (values[-1] - middle[-1])
    return np.concatenate([[first], middle, [last]])


def candidate_distribution_for_group(codes: np.ndarray) -> dict[str, float]:
    counts = np.bincount(codes.astype(np.int16), minlength=len(candidate_columns))
    return {
        candidate_columns[index]: float(count / max(len(codes), 1))
        for index, count in enumerate(counts)
        if count > 0
    }


def plot_one_well(well_id: str, full_group: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    full_group = full_group.sort_values("md_since", kind="mergesort")
    group = downsample_for_plot(full_group, MAX_POINTS_PER_PLOT)
    x = group["md_since"].to_numpy(np.float64)
    true_tvt = group["true_tvt"].to_numpy(np.float64)
    exp238_tvt = group["exp238_lgb_mean_oof_tvt"].to_numpy(np.float64)
    selected = group["selector_top1_tvt"].to_numpy(np.float64)
    codes = group["selector_top1_code"].to_numpy(np.int16)
    margin = group["selector_confidence_margin"].to_numpy(np.float64)

    full_true = full_group["true_tvt"].to_numpy(np.float64)
    full_exp238 = full_group["exp238_lgb_mean_oof_tvt"].to_numpy(np.float64)
    full_selected = full_group["selector_top1_tvt"].to_numpy(np.float64)
    full_codes = full_group["selector_top1_code"].to_numpy(np.int16)
    full_margin = full_group["selector_confidence_margin"].to_numpy(np.float64)
    distribution = candidate_distribution_for_group(full_codes)
    dominant_code = int(np.argmax(np.bincount(full_codes, minlength=len(candidate_columns))))
    dominant_candidate = candidate_columns[dominant_code]
    dominant_share = distribution[dominant_candidate]
    switches = int(np.sum(full_codes[1:] != full_codes[:-1])) if len(full_codes) > 1 else 0

    exp238_rmse = rmse_values(full_true, full_exp238)
    selector_rmse = rmse_values(full_true, full_selected)
    likpf_rmse = rmse_values(
        full_true, full_group["likpf_mean"].to_numpy(np.float64)
    )
    exp226_rmse = rmse_values(
        full_true,
        full_group["exp226_v6_k16_geometry_gr_u_projection"].to_numpy(np.float64),
    )

    fig, (ax_tvt, ax_margin, ax_top1) = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [7.0, 1.5, 0.65], "hspace": 0.12},
    )

    ax_tvt.plot(
        x,
        true_tvt,
        color=REFERENCE_LINE_COLORS["true_tvt"],
        linewidth=2.3,
        label="true TVT",
        zorder=8,
    )
    ax_tvt.plot(
        x,
        exp238_tvt,
        color=REFERENCE_LINE_COLORS["ml_oof"],
        linewidth=2.0,
        label="exp238 lgb_mean OOF",
        zorder=7,
    )
    ax_tvt.plot(
        x,
        selected,
        color=REFERENCE_LINE_COLORS["selector_top1"],
        linewidth=1.7,
        linestyle="--",
        label="selector top-1 candidate (diagnostic)",
        zorder=6,
    )
    ax_tvt.plot(
        x,
        group["likpf_mean"],
        color=REFERENCE_LINE_COLORS["likpf_mean"],
        linewidth=1.1,
        alpha=0.75,
        label="Likelihood PF mean",
        zorder=4,
    )
    ax_tvt.plot(
        x,
        group["pf_ancc"],
        color=REFERENCE_LINE_COLORS["pf_ancc"],
        linewidth=0.95,
        alpha=0.6,
        label="PF ANCC",
        zorder=3,
    )
    ax_tvt.plot(
        x,
        group["beam_mean"],
        color=REFERENCE_LINE_COLORS["beam_mean"],
        linewidth=0.95,
        alpha=0.6,
        label="Beam mean",
        zorder=3,
    )
    ax_tvt.plot(
        x,
        group["exp226_v6_k16_geometry_gr_u_projection"],
        color=REFERENCE_LINE_COLORS["exp226_k16"],
        linewidth=1.1,
        alpha=0.75,
        label="exp226 K16 OOF",
        zorder=4,
    )
    hmm_mean = group["hmm_exact_mean_tvt"].to_numpy(np.float64)
    hmm_std = group["hmm_exact_std"].to_numpy(np.float64)
    ax_tvt.plot(
        x,
        hmm_mean,
        color=REFERENCE_LINE_COLORS["exp209_hmm"],
        linewidth=1.0,
        alpha=0.7,
        label="exp209 exact HMM",
        zorder=3,
    )
    ax_tvt.fill_between(
        x,
        hmm_mean - 2.0 * hmm_std,
        hmm_mean + 2.0 * hmm_std,
        color=REFERENCE_LINE_COLORS["exp209_hmm_band"],
        alpha=0.13,
        linewidth=0,
        label="exact HMM ±2σ",
        zorder=1,
    )
    ax_tvt.plot(
        x,
        minmax_negative_z_to_likpf(group),
        color=REFERENCE_LINE_COLORS["z_likpf_minmax"],
        linewidth=1.1,
        linestyle=":",
        alpha=0.55,
        label="-Z min-max to LikPF range",
        zorder=2,
    )
    if TVT_AXIS_INVERTED:
        ax_tvt.invert_yaxis()
    ax_tvt.set_ylabel("TVT (ft; depth increases downward)")
    ax_tvt.grid(
        True,
        color=REFERENCE_LINE_COLORS["grid"],
        linewidth=0.7,
        alpha=0.8,
    )
    ax_tvt.legend(loc="best", fontsize=8, ncol=2)

    ax_margin.plot(
        x,
        margin,
        color=REFERENCE_LINE_COLORS["confidence_margin"],
        linewidth=1.0,
    )
    ax_margin.fill_between(
        x,
        0.0,
        margin,
        color=REFERENCE_LINE_COLORS["confidence_margin"],
        alpha=0.12,
    )
    ax_margin.axhline(
        0.0, color=REFERENCE_LINE_COLORS["grid"], linewidth=0.8, alpha=0.8
    )
    ax_margin.set_ylabel("top2-top1\npred. error")
    ax_margin.grid(
        True,
        color=REFERENCE_LINE_COLORS["grid"],
        linewidth=0.7,
        alpha=0.8,
    )

    cmap = ListedColormap(CANDIDATE_COLORS)
    norm = BoundaryNorm(
        np.arange(-0.5, len(candidate_columns) + 0.5, 1.0), cmap.N
    )
    ax_top1.pcolormesh(
        x_edges(x),
        np.asarray([0.0, 1.0]),
        codes.reshape(1, -1),
        cmap=cmap,
        norm=norm,
        shading="flat",
    )
    ax_top1.set_yticks([])
    ax_top1.set_ylabel("top-1", rotation=0, labelpad=24, va="center")
    ax_top1.set_xlabel("MD since prediction start (ft)")
    ax_top1.set_title(
        "Selector top-1 candidate by row (lowest predicted absolute error)",
        fontsize=9,
        loc="left",
    )

    title = (
        f"{well_id} | exp238 OOF RMSE {fmt_metric(exp238_rmse)} | "
        f"selector top-1 RMSE {fmt_metric(selector_rmse)} | "
        f"LikPF {fmt_metric(likpf_rmse)} | exp226 {fmt_metric(exp226_rmse)}\n"
        f"dominant selector top-1: {CANDIDATE_LABELS[dominant_candidate]} "
        f"({100.0 * dominant_share:.1f}%) | mean confidence margin "
        f"{fmt_metric(np.mean(full_margin), 3)} | switches {switches}"
    )
    fig.suptitle(title, fontsize=11, y=0.995)
    candidate_handles = [
        Patch(color=CANDIDATE_COLORS[index], label=CANDIDATE_LABELS[name])
        for index, name in enumerate(candidate_columns)
    ]
    fig.legend(
        handles=candidate_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=4,
        fontsize=8,
        title="Selector top-1 color map",
        title_fontsize=8,
    )
    fig.text(
        0.01,
        0.01,
        "Caveat: top-1 is a diagnostic candidate path. exp238 final prediction is the rose "
        "LightGBM OOF line; the selector worst-well safety guard failed.",
        fontsize=8,
        color=REFERENCE_LINE_COLORS["caveat"],
    )
    fig.tight_layout(rect=[0.0, 0.09, 1.0, 0.95])
    fig.savefig(output_path, dpi=145, bbox_inches="tight")
    plt.close(fig)

    return {
        "well": well_id,
        "rows": int(len(full_group)),
        "plotted_rows": int(len(group)),
        "outer_fold": int(full_group["selector_outer_fold"].iloc[0]),
        "exp238_lgb_mean_oof_rmse": exp238_rmse,
        "selector_top1_rmse": selector_rmse,
        "likpf_mean_rmse": likpf_rmse,
        "exp226_k16_rmse": exp226_rmse,
        "dominant_selector_top1_candidate": dominant_candidate,
        "dominant_selector_top1_label": CANDIDATE_LABELS[dominant_candidate],
        "dominant_selector_top1_share": dominant_share,
        "selector_top1_switches": switches,
        "selector_confidence_margin_mean": float(np.mean(full_margin)),
        "selector_confidence_margin_p50": float(np.quantile(full_margin, 0.50)),
        "selector_confidence_margin_p90": float(np.quantile(full_margin, 0.90)),
        "selector_top1_distribution_json": json.dumps(distribution, sort_keys=True),
        "plot_path": str(output_path),
    }


global_metrics = {
    "rows": int(len(base)),
    "wells": int(base["well"].nunique()),
    "exp238_lgb_mean_oof_rmse": rmse_values(
        base["true_tvt"].to_numpy(), base["exp238_lgb_mean_oof_tvt"].to_numpy()
    ),
    "selector_top1_rmse": rmse_values(
        base["true_tvt"].to_numpy(), base["selector_top1_tvt"].to_numpy()
    ),
    "likpf_mean_rmse": rmse_values(
        base["true_tvt"].to_numpy(), base["likpf_mean"].to_numpy()
    ),
    "exp226_k16_rmse": rmse_values(
        base["true_tvt"].to_numpy(),
        base["exp226_v6_k16_geometry_gr_u_projection"].to_numpy(),
    ),
    "selector_confidence_margin_mean": float(
        base["selector_confidence_margin"].mean()
    ),
    "selector_confidence_margin_p50": float(
        base["selector_confidence_margin"].quantile(0.50)
    ),
    "selector_confidence_margin_p90": float(
        base["selector_confidence_margin"].quantile(0.90)
    ),
}
print(json.dumps(global_metrics, indent=2))

# %% [markdown]
# ## 6. Generate all-well plots

# %%
all_wells = common_typewell_order["well"].tolist()
plot_order_frame = (
    common_typewell_order.iloc[:MAX_PLOTS].copy()
    if MAX_PLOTS is not None
    else common_typewell_order.copy()
)
plot_wells = plot_order_frame["well"].tolist()
indices_by_well = base.groupby("well", sort=False).indices

plot_rows: list[dict[str, Any]] = []
for plot_meta in plot_order_frame.itertuples(index=False):
    plot_index = int(plot_meta.plot_order)
    well_id = str(plot_meta.well)
    positions = indices_by_well[well_id]
    group = base.iloc[positions].copy()
    fold_values = group["selector_outer_fold"].unique()
    if len(fold_values) != 1:
        raise ValueError(f"well {well_id} spans selector outer folds {fold_values.tolist()}")
    plot_path = PLOTS_DIR / str(plot_meta.plot_filename)
    plot_rows.append(
        {
            "plot_order": plot_index,
            "typewell_order": int(plot_meta.typewell_order),
            "well_order_within_typewell": int(plot_meta.well_order_within_typewell),
            "typewell_method": str(plot_meta.typewell_method),
            "typewell_threshold": str(plot_meta.typewell_threshold),
            "typewell_cluster_id": str(plot_meta.typewell_cluster_id),
            "typewell_cluster_size": int(plot_meta.typewell_cluster_size),
            "typewell_representative_well_id": str(
                plot_meta.typewell_representative_well_id
            ),
            "plot_filename": str(plot_meta.plot_filename),
            **plot_one_well(well_id, group, plot_path),
        }
    )
    if plot_index % 50 == 0 or len(plot_rows) == len(plot_wells):
        print(f"wrote {len(plot_rows)}/{len(plot_wells)} plots")

manifest = pd.DataFrame(plot_rows)
manifest_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_plot_manifest.csv"
manifest.to_csv(manifest_path, index=False)

distribution_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_selector_top1_distribution.csv"
selector_distribution.to_csv(distribution_path, index=False)

zip_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_plots.zip"
if ZIP_PLOTS:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for plot_filename in manifest["plot_filename"].astype(str):
            plot_path = PLOTS_DIR / plot_filename
            if not plot_path.is_file():
                raise FileNotFoundError(f"manifest plot does not exist: {plot_path}")
            archive.write(plot_path, arcname=plot_filename)

display(manifest.head(10))
display(
    manifest.sort_values("selector_top1_rmse", ascending=False)[
        [
            "well",
            "outer_fold",
            "exp238_lgb_mean_oof_rmse",
            "selector_top1_rmse",
            "dominant_selector_top1_label",
            "dominant_selector_top1_share",
            "selector_confidence_margin_mean",
        ]
    ].head(20)
)

# %% [markdown]
# ## 7. Summary and generated outputs

# %%
summary = {
    "status": "diagnostic_plots_completed_not_submitted",
    "created_at_utc": datetime.now(UTC).isoformat(),
    "experiment": EXPERIMENT_NAME,
    "notebook": f"{EXPERIMENT_NAME}_oof_selector_confidence_probe.ipynb",
    "reference_notebook": "kentookumura/exp083-v12-ml-oof-known-tvt-probe",
    "reference_script_version_id": 333830051,
    "scope": {
        "plot_wells": len(plot_wells),
        "all_wells": len(all_wells),
        "rows": len(base),
        "max_plots_override": MAX_PLOTS,
    },
    "typewell_ordering": {
        "source_experiment": "exp065_typewell_supertype_cluster_cv_audit",
        "method": COMMON_TYPEWELL_METHOD,
        "threshold": COMMON_TYPEWELL_THRESHOLD,
        "sort_keys": ["typewell_cluster_id", "well"],
        "groups": int(common_typewell_order["typewell_order"].nunique()),
        "wells": int(len(common_typewell_order)),
        "filename_template": "typewell_{typewell_order:04d}_{well}.png",
        "manifest_and_zip_follow_plot_order": True,
    },
    "selector_definition": {
        "score": "predicted candidate absolute error",
        "top1": "minimum predicted error within the 11 candidates",
        "confidence_margin": "second-lowest predicted error minus lowest predicted error",
        "score_role": "strict outer-valid only",
        "direct_replacement": False,
        "exp238_usage": "rank-slot selector values are add-only final LightGBM features",
        "historical_guard_pass": bool(
            selector_summary.get("decision", {}).get("guard_pass", False)
        ),
        "historical_worst_well_regression": selector_summary.get("decision", {}).get(
            "worst_well_regression"
        ),
    },
    "candidate_columns": candidate_columns,
    "plot_colors": {
        "reference": "exp083 v12 ml oof known tvt probe",
        "line_colors": REFERENCE_LINE_COLORS,
        "selector_top1_candidate_colors": CANDIDATE_COLOR_BY_NAME,
    },
    "global_metrics": global_metrics,
    "selector_top1_distribution": selector_distribution.to_dict(orient="records"),
    "inputs": {
        "pfbeam_cache": str(pfbeam_path),
        "exp238_final_oof": str(exp238_oof_path),
        "exp238_final_oof_sha256_decompressed": sha256_path(
            exp238_oof_path, decompressed=True
        ),
        "selector_summary": str(selector_summary_path),
        "selector_summary_sha256": sha256_path(selector_summary_path),
        "selector_scores": [
            {
                "outer_fold": fold,
                "path": str(selector_score_paths[fold]),
                **declared_score_contract[fold],
            }
            for fold in range(5)
        ],
        "exp209_exact_hmm": str(exp209_hmm_path),
        "exp223_selfgr_hmm": str(exp223_selfgr_path),
        "exp226_k16_oof": str(exp226_oof_path),
        "common_typewell_assignments": str(common_typewell_assignments_path),
        "common_typewell_assignments_sha256": sha256_path(
            common_typewell_assignments_path
        ),
    },
    "outputs": {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "selector_distribution": str(distribution_path),
        "selector_distribution_sha256": sha256_path(distribution_path),
        "plots_dir": str(PLOTS_DIR),
        "plots_zip": str(zip_path) if ZIP_PLOTS else None,
        "plots_zip_sha256": sha256_path(zip_path) if ZIP_PLOTS else None,
    },
    "notes": [
        "The selector top-1 candidate is the candidate with minimum predicted absolute error.",
        "Only role=valid rows from each outer-fold selector score artifact are used.",
        "The top-1 categorical strip and gray dashed path are diagnostic; they are not exp238 final predictions.",
        "The rose exp238 line is saved lgb_mean OOF from final train v5, matching the exp083 ML OOF color.",
        "Shared PF/Beam/LikPF/exp226/HMM series keep the exact exp083 v12 colors.",
        "PNG filenames, manifest rows, and zip members follow exp065 native-overlap common typewell order.",
        "The historical selector worst-well guard failed, so direct top-1 replacement is not authorized.",
        "No model training, PF/Beam regeneration, inference, submission generation, or competition submit occurs.",
    ],
}
summary_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("Manifest:", manifest_path)
print("Selector distribution:", distribution_path)
print("Plots directory:", PLOTS_DIR)
print("Plots zip:", zip_path if ZIP_PLOTS else None)
print("Summary:", summary_path)
print("Summary SHA256:", sha256_path(summary_path))

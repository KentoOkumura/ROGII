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
# # exp264 corrected OOF and selector-confidence plots
#
# Diagnostic visualization derived from the exp238 selector-confidence probe
# (`scriptVersionId=336248071`). The final overlay is the corrected exp264
# Stage D v3 compact-add-only OOF. The selector surface is the corrected Stage C
# v6 strict nested outer-valid score: four inner models score each outer-valid
# row without using that well for fit or early stopping.
#
# exp264 has two objectives and two legal candidate domains. This notebook keeps
# the selector result from the primary 11-candidate primitive/pair domain, but
# preserves the exp238 plot contract exactly: the same three panels, reference
# path types, colors, line styles, and exact-HMM mean plus/minus two sigma band.
# Fixed-domain and probability surfaces remain summary-only diagnostics.
#
# The corrected selector score guard passed, but hard top-1 failed. Stage D also
# retained a worst-well guard failure. Both caveats remain visible in every plot
# and in the summary.

# %% [markdown]
# ## Contents
#
# 1. Imports and configuration
# 2. Path resolution and input contracts
# 3. Corrected Stage D OOF and typewell ordering
# 4. Strict nested selector surface
# 5. Metrics and plot helpers
# 6. Generate all-well plots
# 7. Summary and generated outputs

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import display


EXPERIMENT_NAME = "exp264_exp263_candidate_confidence_dual_selector"
NOTEBOOK_KIND = "oof_selector_confidence_probe"
OUTPUT_PREFIX = f"{EXPERIMENT_NAME}_{NOTEBOOK_KIND}"

STAGE_D_OOF_FILENAME = "stage_d_oof_predictions.parquet"
STAGE_C_SCORE_FILENAME = "nested_outer_valid_candidate_score.parquet"
PFBEAM_FILENAME = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
EXP209_HMM_FILENAME = (
    "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation_"
    "amerhu_exact_hmm_smoother_default_train_features.csv.gz"
)
EXP226_OOF_FILENAME = (
    "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_"
    "train_oof_predictions.csv.gz"
)
COMMON_TYPEWELL_ASSIGNMENTS_FILENAME = "common_typewell_cluster_assignments.csv"
FINAL_OOF_COLUMN = "selector_compact_addonly__lgb_mean__pred_tvt"
CONTROL_OOF_COLUMN = "matched_control__lgb_mean__pred_tvt"

EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
EXPECTED_LONG_ROWS = EXPECTED_ROWS * 12
EXPECTED_STAGE_D_OOF_SHA256 = (
    "b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2"
)
EXPECTED_STAGE_C_SCORE_SHA256 = (
    "a10b7848127f01bef522f4b17dfd1640c9784956892dc24fc1159e3869500abc"
)
EXPECTED_TYPEWELL_SHA256 = (
    "dcda8588cc1dd9261bafae7de00c890393e38b8a0ca0eb86fbba18a2cffc4a50"
)
EXPECTED_FINAL_RMSE = 8.460811237612477
EXPECTED_HARD_PRIMARY_RMSE = 8.652531955610227
EXPECTED_COMMON_TYPEWELL_GROUPS = 54
COMMON_TYPEWELL_METHOD = "native_overlap"
COMMON_TYPEWELL_THRESHOLD = "0.999"

CANDIDATES = [
    "exp226_k16",
    "selfgr_hmm_a070",
    "likpf_mean",
    "exact_hmm",
    "pf_ancc",
    "beam_mean",
    "exp226_k16__selfgr_hmm_a070",
    "exp226_k16__exact_hmm",
    "exp226_k16__likpf_mean",
    "selfgr_hmm_a070__likpf_mean",
    "likpf_mean__exact_hmm",
    "exp226_w500_50_50",
]
PRIMARY_DOMAIN = CANDIDATES[:11]
FIXED_DOMAIN = [*CANDIDATES[:6], CANDIDATES[11]]
CANDIDATE_LABELS = {
    "exp226_k16": "exp226 K16",
    "selfgr_hmm_a070": "Self-GR HMM",
    "likpf_mean": "Likelihood PF mean",
    "exact_hmm": "Exact HMM",
    "pf_ancc": "PF ANCC",
    "beam_mean": "Beam mean",
    "exp226_k16__selfgr_hmm_a070": "K16 / Self-GR 50:50",
    "exp226_k16__exact_hmm": "K16 / exact-HMM 50:50",
    "exp226_k16__likpf_mean": "K16 / LikPF 50:50",
    "selfgr_hmm_a070__likpf_mean": "Self-GR / LikPF 50:50",
    "likpf_mean__exact_hmm": "LikPF / exact-HMM 50:50",
    "exp226_w500_50_50": "exp226 fixed w500 50:50",
}
REFERENCE_LINE_COLORS = {
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
    "exp226_k16": REFERENCE_LINE_COLORS["exp226_k16"],
    "selfgr_hmm_a070": REFERENCE_LINE_COLORS["exp209_hmm"],
    "likpf_mean": REFERENCE_LINE_COLORS["likpf_mean"],
    "exact_hmm": REFERENCE_LINE_COLORS["exp209_hmm"],
    "pf_ancc": REFERENCE_LINE_COLORS["pf_ancc"],
    "beam_mean": REFERENCE_LINE_COLORS["beam_mean"],
    "exp226_k16__selfgr_hmm_a070": "#9467bd",
    "exp226_k16__exact_hmm": "#8c564b",
    "exp226_k16__likpf_mean": "#e377c2",
    "selfgr_hmm_a070__likpf_mean": "#7f7f7f",
    "likpf_mean__exact_hmm": "#bcbd22",
    "exp226_w500_50_50": "#17becf",
}
CANDIDATE_COLORS = [CANDIDATE_COLOR_BY_NAME[name] for name in CANDIDATES]

MAX_POINTS_PER_PLOT = 6000
READ_CHUNKSIZE = 300_000
ZIP_PLOTS = True
TVT_AXIS_INVERTED = True
MAX_PLOTS_ENV = os.environ.get("EXPERIMENT_MAX_PLOTS")
MAX_PLOTS = int(MAX_PLOTS_ENV) if MAX_PLOTS_ENV else None

print("Experiment:", EXPERIMENT_NAME)
print("Final OOF: corrected Stage D v3 add-only")
print("Selector surface: corrected Stage C v6 strict nested outer-valid")
print("Primary domain:", len(PRIMARY_DOMAIN), "candidates")
print("Plot contract: exp238 exact structure, reference paths, colors, and HMM ±2σ")
print("Hard selector guard: FAIL; Stage D worst-well guard: FAIL")
print("Debug max plots override:", MAX_PLOTS)

# %% [markdown]
# ## 2. Path resolution and input contracts

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


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path
    return None


def resolve_input(
    *, filename: str, local_candidates: list[Path], preferred_slugs: list[str]
) -> Path:
    local = _existing(local_candidates)
    if local is not None:
        return local
    if KAGGLE_INPUT_ROOT.exists():
        preferred_roots = [
            KAGGLE_INPUT_ROOT / slug for slug in preferred_slugs
        ] + [
            KAGGLE_INPUT_ROOT / "notebooks" / "kentookumura" / slug
            for slug in preferred_slugs
        ]
        generic_roots = [
            path for path in sorted(KAGGLE_INPUT_ROOT.iterdir()) if path.is_dir()
        ]
        seen: set[Path] = set()
        for root in [*preferred_roots, *generic_roots]:
            if root in seen or not root.exists():
                continue
            seen.add(root)
            matches = sorted(
                path
                for path in root.rglob(filename)
                if path.is_file() and path.stat().st_size > 0
            )
            if matches:
                return matches[0]
    checked = "\n".join(str(path) for path in local_candidates)
    raise FileNotFoundError(
        f"{filename} not found. Checked:\n{checked}\nKaggle slugs: {preferred_slugs}"
    )


stage_d_oof_path = resolve_input(
    filename=STAGE_D_OOF_FILENAME,
    local_candidates=[
        EXP_DIR / "artifacts" / "stage_d_v3_corrected" / STAGE_D_OOF_FILENAME,
        EXP_DIR
        / "kaggle"
        / "output"
        / "stage_d_v3_corrected"
        / "artifacts"
        / STAGE_D_OOF_FILENAME,
        Path("/tmp/exp264-stage-d-v3-oof/artifacts") / STAGE_D_OOF_FILENAME,
    ],
    preferred_slugs=["exp264-exp263-confidence-dual-selector-tvt-train"],
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
stage_c_score_path = resolve_input(
    filename=STAGE_C_SCORE_FILENAME,
    local_candidates=[
        EXP_DIR / "artifacts" / "stage_c_v6" / STAGE_C_SCORE_FILENAME,
        EXP_DIR
        / "kaggle"
        / "output"
        / "stage_c_v6"
        / "artifacts"
        / STAGE_C_SCORE_FILENAME,
        Path("/tmp/exp264-stage-c-v6-outer-valid/artifacts")
        / STAGE_C_SCORE_FILENAME,
    ],
    preferred_slugs=["exp264-exp263-confidence-dual-selector-train"],
)
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
common_typewell_path = resolve_input(
    filename=COMMON_TYPEWELL_ASSIGNMENTS_FILENAME,
    local_candidates=[
        REPO_ROOT
        / "experiments"
        / "exp065_typewell_supertype_cluster_cv_audit"
        / "artifacts"
        / COMMON_TYPEWELL_ASSIGNMENTS_FILENAME,
    ],
    preferred_slugs=["exp065-typewell-supertype-cluster-cv-audit-train"],
)

input_sha = {
    "exp072_pfbeam_cache": sha256_path(pfbeam_path),
    "stage_d_oof": sha256_path(stage_d_oof_path),
    "stage_c_outer_valid_score": sha256_path(stage_c_score_path),
    "exp209_exact_hmm": sha256_path(exp209_hmm_path),
    "exp226_k16_oof": sha256_path(exp226_oof_path),
    "common_typewell_assignments": sha256_path(common_typewell_path),
}
expected_sha = {
    "stage_d_oof": EXPECTED_STAGE_D_OOF_SHA256,
    "stage_c_outer_valid_score": EXPECTED_STAGE_C_SCORE_SHA256,
    "common_typewell_assignments": EXPECTED_TYPEWELL_SHA256,
}
if {name: input_sha[name] for name in expected_sha} != expected_sha:
    raise ValueError({"message": "input SHA contract failed", "actual": input_sha})

print("exp072 PF/Beam cache:", pfbeam_path)
print("Stage D OOF:", stage_d_oof_path)
print("Stage C outer-valid score:", stage_c_score_path)
print("exp209 exact-HMM:", exp209_hmm_path)
print("exp226 K16 OOF:", exp226_oof_path)
print("Common typewell assignments:", common_typewell_path)

# %% [markdown]
# ## 3. Corrected Stage D OOF and typewell ordering

# %%
def rmse_values(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not bool(valid.any()):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(prediction[valid] - truth[valid]))))


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
    base_frame: pd.DataFrame,
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
        column: np.full(len(base_frame), np.nan, dtype=np.float32)
        for column in value_columns
    }
    base_ids = base_frame["id"].to_numpy(dtype=str)
    base_wells = base_frame["well"].to_numpy(dtype=str)
    offset = 0
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"id": str, "well": str},
        chunksize=READ_CHUNKSIZE,
        low_memory=False,
    ):
        stop = offset + len(chunk)
        if stop > len(base_frame):
            raise ValueError(f"{source_name} contains more rows than the base cache")
        if not np.array_equal(chunk["id"].astype(str).to_numpy(), base_ids[offset:stop]):
            raise ValueError(f"{source_name} id order differs from the base cache at row {offset}")
        if not np.array_equal(
            chunk["well"].astype(str).to_numpy(), base_wells[offset:stop]
        ):
            raise ValueError(f"{source_name} well order differs from the base cache at row {offset}")
        for column in value_columns:
            outputs[column][offset:stop] = pd.to_numeric(
                chunk[column], errors="coerce"
            ).to_numpy(np.float32)
        offset = stop
    if offset != len(base_frame):
        raise ValueError(
            f"{source_name} row count {offset} != base row count {len(base_frame)}"
        )
    return outputs


def read_exp226_candidate(path: Path, base_frame: pd.DataFrame) -> np.ndarray:
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
                    "exp226_k16_tvt": pd.to_numeric(
                        chunk["tvt_pred"], errors="coerce"
                    ).astype(np.float32),
                }
            )
        )
    candidate = pd.concat(chunks, ignore_index=True)
    if candidate.duplicated(["id", "well"]).any():
        raise ValueError("exp226 OOF has duplicate id/well rows")
    aligned = base_frame[["id", "well"]].merge(
        candidate,
        on=["id", "well"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    values = aligned["exp226_k16_tvt"].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("exp226 candidate does not fully cover the base cache")
    return values


base = read_base_candidates(pfbeam_path)
if len(base) != EXPECTED_ROWS or base["well"].nunique() != EXPECTED_WELLS:
    raise ValueError("exp072 PF/Beam base row/well contract changed")
if base["id"].duplicated().any():
    raise ValueError("exp072 PF/Beam base IDs are not unique")

stage_d_columns = [
    "id",
    "well",
    "md_since",
    "outer_fold",
    "actual_tvt",
    CONTROL_OOF_COLUMN,
    FINAL_OOF_COLUMN,
]
stage_d_schema = pq.read_schema(stage_d_oof_path).names
missing_stage_d = sorted(set(stage_d_columns).difference(stage_d_schema))
if missing_stage_d:
    raise ValueError(f"Stage D OOF missing columns: {missing_stage_d}")
stage_d = pd.read_parquet(stage_d_oof_path, columns=stage_d_columns)
stage_d["id"] = stage_d["id"].astype(str)
stage_d["well"] = stage_d["well"].astype(str)
if len(stage_d) != EXPECTED_ROWS or stage_d["well"].nunique() != EXPECTED_WELLS:
    raise ValueError("corrected Stage D OOF row/well contract changed")
stage_d_index = pd.Index(stage_d["id"])
if not stage_d_index.is_unique:
    raise ValueError("corrected Stage D OOF contains duplicate IDs")
stage_d_positions = stage_d_index.get_indexer(base["id"].to_numpy(dtype=str))
if bool(np.any(stage_d_positions < 0)) or len(np.unique(stage_d_positions)) != len(base):
    raise ValueError("corrected Stage D OOF does not cover the exp072 base one-to-one")
stage_d = stage_d.iloc[stage_d_positions].reset_index(drop=True)
if not np.array_equal(stage_d["well"].to_numpy(dtype=str), base["well"].to_numpy(dtype=str)):
    raise ValueError("corrected Stage D well alignment differs from the exp072 base")
if not np.allclose(
    stage_d["md_since"].to_numpy(np.float64),
    base["md_since"].to_numpy(np.float64),
    rtol=0.0,
    atol=1e-5,
):
    raise ValueError("corrected Stage D md_since differs from the exp072 base")
base["outer_fold"] = stage_d["outer_fold"].to_numpy(np.int8)
base["actual_tvt"] = stage_d["actual_tvt"].to_numpy(np.float32)
base[CONTROL_OOF_COLUMN] = stage_d[CONTROL_OOF_COLUMN].to_numpy(np.float32)
base[FINAL_OOF_COLUMN] = stage_d[FINAL_OOF_COLUMN].to_numpy(np.float32)
if not np.allclose(
    base["actual_tvt"].to_numpy(np.float64),
    base["true_tvt"].to_numpy(np.float64),
    rtol=0.0,
    atol=1e-5,
):
    raise ValueError("corrected Stage D truth differs from the exp072 replay truth")
del stage_d

exp209_values = read_order_aligned_columns(
    exp209_hmm_path,
    base,
    source_name="exp209 exact HMM",
    value_columns=["hmm_mean_tvt", "hmm_std"],
)
base["hmm_exact_mean_tvt"] = exp209_values["hmm_mean_tvt"]
base["hmm_exact_std"] = exp209_values["hmm_std"]
base["exp226_k16_tvt"] = read_exp226_candidate(exp226_oof_path, base)

required_finite = [
    "true_tvt",
    "pf_ancc",
    "beam_mean",
    "likpf_mean",
    "hmm_exact_mean_tvt",
    "hmm_exact_std",
    "exp226_k16_tvt",
    CONTROL_OOF_COLUMN,
    FINAL_OOF_COLUMN,
]
for column in required_finite:
    if not np.isfinite(base[column].to_numpy(np.float32)).all():
        raise ValueError(f"{column} contains non-finite values")

observed_final_rmse = rmse_values(base["true_tvt"], base[FINAL_OOF_COLUMN])
if not np.isclose(observed_final_rmse, EXPECTED_FINAL_RMSE, atol=1e-9):
    raise ValueError(
        f"corrected Stage D RMSE {observed_final_rmse} != {EXPECTED_FINAL_RMSE}"
    )


def read_common_typewell_order(path: Path, wells: list[str]) -> pd.DataFrame:
    required = {
        "method",
        "threshold",
        "cluster_id",
        "well_id",
        "cluster_size",
        "representative_well_id",
    }
    frame = pd.read_csv(path, usecols=sorted(required), dtype=str)
    selected = frame.loc[
        frame["method"].eq(COMMON_TYPEWELL_METHOD)
        & frame["threshold"].eq(COMMON_TYPEWELL_THRESHOLD)
    ].copy()
    if selected.empty or selected["well_id"].duplicated().any():
        raise ValueError("common typewell assignment selection is invalid")
    if set(selected["well_id"]) != set(wells):
        raise ValueError("common typewell assignment coverage differs from OOF wells")
    selected["cluster_size"] = pd.to_numeric(
        selected["cluster_size"], errors="raise"
    ).astype(np.int64)
    selected = selected.sort_values(
        ["cluster_id", "well_id"], kind="mergesort"
    ).reset_index(drop=True)
    cluster_ids = selected["cluster_id"].drop_duplicates().tolist()
    if len(cluster_ids) != EXPECTED_COMMON_TYPEWELL_GROUPS:
        raise ValueError("common typewell group count changed")
    cluster_order = {
        cluster_id: index for index, cluster_id in enumerate(cluster_ids, start=1)
    }
    selected["typewell_order"] = selected["cluster_id"].map(cluster_order)
    selected["well_order_within_typewell"] = (
        selected.groupby("cluster_id", sort=False).cumcount() + 1
    )
    selected["plot_order"] = np.arange(1, len(selected) + 1)
    selected["plot_filename"] = [
        f"typewell_{order:04d}_{well}.png"
        for order, well in zip(
            selected["typewell_order"], selected["well_id"], strict=True
        )
    ]
    return selected.rename(
        columns={
            "method": "typewell_method",
            "threshold": "typewell_threshold",
            "cluster_id": "typewell_cluster_id",
            "cluster_size": "typewell_cluster_size",
            "representative_well_id": "typewell_representative_well_id",
            "well_id": "well",
        }
    )


typewell_order = read_common_typewell_order(
    common_typewell_path, base["well"].drop_duplicates().tolist()
)
print("Base rows / wells:", len(base), "/", base["well"].nunique())
print("Final corrected Stage D v3 RMSE:", observed_final_rmse)
print("Common typewell groups:", typewell_order["typewell_order"].nunique())
display(base.head())

# %% [markdown]
# ## 4. Strict nested selector surface
#
# Each Parquet row group is validated as complete 12-candidate blocks. The
# primary domain minimizes predicted absolute error. The probability objective
# maximizes `p_within10`; its confidence margin is top1 minus top2.

# %%
def top_two_surface(
    scores: np.ndarray,
    values: np.ndarray,
    domain_indices: np.ndarray,
    *,
    maximize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    domain_scores = scores[:, domain_indices]
    ranking_source = -domain_scores if maximize else domain_scores
    pair = np.argpartition(ranking_source, kth=1, axis=1)[:, :2]
    pair_rank = np.take_along_axis(ranking_source, pair, axis=1)
    swap = pair_rank[:, 1] < pair_rank[:, 0]
    first_local = np.where(swap, pair[:, 1], pair[:, 0])
    second_local = np.where(swap, pair[:, 0], pair[:, 1])
    first_code = domain_indices[first_local]
    second_code = domain_indices[second_local]
    rows = np.arange(len(scores))
    first_score = scores[rows, first_code]
    second_score = scores[rows, second_code]
    margin = first_score - second_score if maximize else second_score - first_score
    if bool(np.any(margin < -1e-6)):
        raise ValueError("selector confidence margin became negative")
    return (
        first_code.astype(np.int16),
        values[rows, first_code].astype(np.float32),
        first_score.astype(np.float32),
        np.maximum(margin, 0.0).astype(np.float32),
    )


n_rows = len(base)
surface_arrays = {
    "primary_error_code": np.full(n_rows, -1, dtype=np.int16),
    "primary_error_tvt": np.full(n_rows, np.nan, dtype=np.float32),
    "primary_error_score": np.full(n_rows, np.nan, dtype=np.float32),
    "primary_error_margin": np.full(n_rows, np.nan, dtype=np.float32),
    "fixed_error_code": np.full(n_rows, -1, dtype=np.int16),
    "fixed_error_tvt": np.full(n_rows, np.nan, dtype=np.float32),
    "fixed_error_margin": np.full(n_rows, np.nan, dtype=np.float32),
    "primary_probability_code": np.full(n_rows, -1, dtype=np.int16),
    "primary_probability_tvt": np.full(n_rows, np.nan, dtype=np.float32),
    "primary_probability_margin": np.full(n_rows, np.nan, dtype=np.float32),
    "fixed_probability_code": np.full(n_rows, -1, dtype=np.int16),
    "fixed_probability_tvt": np.full(n_rows, np.nan, dtype=np.float32),
    "fixed_probability_margin": np.full(n_rows, np.nan, dtype=np.float32),
    "nested_model_count": np.full(n_rows, -1, dtype=np.int8),
}

score_columns = [
    "id",
    "well",
    "outer_fold",
    "candidate_id",
    "candidate_tvt",
    "pred_abs_error",
    "p_within10",
    "downstream_outer_fold",
    "nested_model_count",
]
score_file = pq.ParquetFile(stage_c_score_path)
if score_file.metadata.num_rows != EXPECTED_LONG_ROWS:
    raise ValueError("Stage C outer-valid candidate-long row count changed")
base_ids = base["id"].to_numpy(dtype=str)
base_wells = base["well"].to_numpy(dtype=str)
base_folds = base["outer_fold"].to_numpy(np.int8)
base_id_index = pd.Index(base_ids)
if not base_id_index.is_unique:
    raise ValueError("Stage D base IDs are not unique")
expected_candidate_block = np.asarray(CANDIDATES, dtype=str)
primary_indices = np.arange(len(PRIMARY_DOMAIN), dtype=np.int16)
fixed_indices = np.asarray([0, 1, 2, 3, 4, 5, 11], dtype=np.int16)
covered_rows = np.zeros(n_rows, dtype=bool)
processed_rows = 0

for row_group_index in range(score_file.metadata.num_row_groups):
    chunk = score_file.read_row_group(row_group_index, columns=score_columns).to_pandas()
    if len(chunk) % len(CANDIDATES) != 0:
        raise ValueError(f"row group {row_group_index} breaks candidate blocks")
    block_rows = len(chunk) // len(CANDIDATES)
    candidate_blocks = chunk["candidate_id"].astype(str).to_numpy().reshape(
        block_rows, len(CANDIDATES)
    )
    if not np.all(candidate_blocks == expected_candidate_block[None, :]):
        raise ValueError(f"row group {row_group_index} candidate order changed")
    ids = chunk["id"].astype(str).to_numpy().reshape(block_rows, -1)
    wells = chunk["well"].astype(str).to_numpy().reshape(block_rows, -1)
    folds = chunk["outer_fold"].to_numpy(np.int8).reshape(block_rows, -1)
    downstream_folds = chunk["downstream_outer_fold"].to_numpy(np.int8).reshape(
        block_rows, -1
    )
    model_counts = chunk["nested_model_count"].to_numpy(np.int8).reshape(
        block_rows, -1
    )
    if not (
        np.all(ids == ids[:, :1])
        and np.all(wells == wells[:, :1])
        and np.all(folds == folds[:, :1])
        and np.all(downstream_folds == downstream_folds[:, :1])
        and np.all(model_counts == 4)
    ):
        raise ValueError(f"row group {row_group_index} nested block contract failed")
    # Stage C is grouped by outer fold while Stage D is in global well/row order.
    # Align every strict nested row group by the fail-closed unique ID contract.
    row_positions = base_id_index.get_indexer(ids[:, 0])
    if bool(np.any(row_positions < 0)):
        missing_ids = ids[row_positions < 0, 0][:5].tolist()
        raise ValueError(
            f"row group {row_group_index} has IDs absent from Stage D: {missing_ids}"
        )
    if len(np.unique(row_positions)) != block_rows:
        raise ValueError(f"row group {row_group_index} repeats base IDs")
    if bool(np.any(covered_rows[row_positions])):
        raise ValueError(f"row group {row_group_index} overlaps an earlier row group")
    if not np.array_equal(wells[:, 0], base_wells[row_positions]):
        raise ValueError(f"row group {row_group_index} well order differs from Stage D")
    if not np.array_equal(folds[:, 0], base_folds[row_positions]):
        raise ValueError(f"row group {row_group_index} fold differs from Stage D")
    if not np.array_equal(folds[:, 0], downstream_folds[:, 0]):
        raise ValueError(f"row group {row_group_index} downstream fold is not outer-valid")

    values = chunk["candidate_tvt"].to_numpy(np.float32).reshape(block_rows, -1)
    error_scores = chunk["pred_abs_error"].to_numpy(np.float32).reshape(
        block_rows, -1
    )
    probability_scores = chunk["p_within10"].to_numpy(np.float32).reshape(
        block_rows, -1
    )
    if not (
        np.isfinite(values).all()
        and np.isfinite(error_scores).all()
        and np.isfinite(probability_scores).all()
        and bool(np.all((probability_scores >= 0.0) & (probability_scores <= 1.0)))
    ):
        raise ValueError(f"row group {row_group_index} contains invalid selector values")

    primary_error = top_two_surface(
        error_scores, values, primary_indices, maximize=False
    )
    fixed_error = top_two_surface(error_scores, values, fixed_indices, maximize=False)
    primary_probability = top_two_surface(
        probability_scores, values, primary_indices, maximize=True
    )
    fixed_probability = top_two_surface(
        probability_scores, values, fixed_indices, maximize=True
    )
    for prefix, result in [
        ("primary_error", primary_error),
        ("fixed_error", fixed_error),
        ("primary_probability", primary_probability),
        ("fixed_probability", fixed_probability),
    ]:
        surface_arrays[f"{prefix}_code"][row_positions] = result[0]
        surface_arrays[f"{prefix}_tvt"][row_positions] = result[1]
        if prefix == "primary_error":
            surface_arrays["primary_error_score"][row_positions] = result[2]
        surface_arrays[f"{prefix}_margin"][row_positions] = result[3]
    surface_arrays["nested_model_count"][row_positions] = model_counts[:, 0]
    covered_rows[row_positions] = True
    processed_rows += block_rows
    if row_group_index % 25 == 0 or row_group_index + 1 == score_file.metadata.num_row_groups:
        print(
            f"loaded strict nested surface {processed_rows:,}/{n_rows:,} base rows",
            flush=True,
        )

if processed_rows != n_rows:
    raise ValueError(f"strict nested surface rows {processed_rows} != {n_rows}")
if not bool(np.all(covered_rows)):
    raise ValueError(
        f"strict nested surface misses {int(np.count_nonzero(~covered_rows))} Stage D rows"
    )
for name, values in surface_arrays.items():
    if name.endswith("code") or name == "nested_model_count":
        if bool(np.any(values < 0)):
            raise ValueError(f"{name} is incomplete")
    elif not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")

for name, values in surface_arrays.items():
    base[f"selector_{name}"] = values

observed_hard_primary_rmse = rmse_values(
    base["actual_tvt"], base["selector_primary_error_tvt"]
)
if not np.isclose(observed_hard_primary_rmse, EXPECTED_HARD_PRIMARY_RMSE, atol=1e-9):
    raise ValueError(
        f"hard primary RMSE {observed_hard_primary_rmse} != {EXPECTED_HARD_PRIMARY_RMSE}"
    )

primary_counts = np.bincount(
    base["selector_primary_error_code"].to_numpy(np.int16), minlength=len(CANDIDATES)
)
primary_distribution = pd.DataFrame(
    {
        "candidate_code": np.arange(len(CANDIDATES), dtype=np.int16),
        "candidate": CANDIDATES,
        "label": [CANDIDATE_LABELS[name] for name in CANDIDATES],
        "rows": primary_counts,
        "share": primary_counts / len(base),
    }
).query("rows > 0").sort_values("rows", ascending=False)
display(primary_distribution)

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


def downsample_for_plot(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    positions = np.unique(np.linspace(0, len(frame) - 1, max_points, dtype=np.int64))
    return frame.iloc[positions]


def candidate_distribution(codes: np.ndarray) -> dict[str, float]:
    counts = np.bincount(codes.astype(np.int16), minlength=len(CANDIDATES))
    return {
        CANDIDATES[index]: float(count / max(len(codes), 1))
        for index, count in enumerate(counts)
        if count > 0
    }


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
        return np.asarray([values[0] - 0.5, values[0] + 0.5])
    middle = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        [[values[0] - (middle[0] - values[0])], middle, [values[-1] + (values[-1] - middle[-1])]]
    )


def plot_one_well(well_id: str, full_group: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    full_group = full_group.sort_values("md_since", kind="mergesort")
    group = downsample_for_plot(full_group, MAX_POINTS_PER_PLOT)
    x = group["md_since"].to_numpy(np.float64)
    truth = group["true_tvt"].to_numpy(np.float64)
    final_oof = group[FINAL_OOF_COLUMN].to_numpy(np.float64)
    primary_tvt = group["selector_primary_error_tvt"].to_numpy(np.float64)
    codes = group["selector_primary_error_code"].to_numpy(np.int16)
    margin = group["selector_primary_error_margin"].to_numpy(np.float64)

    full_truth = full_group["true_tvt"].to_numpy(np.float64)
    full_final = full_group[FINAL_OOF_COLUMN].to_numpy(np.float64)
    full_control = full_group[CONTROL_OOF_COLUMN].to_numpy(np.float64)
    full_primary = full_group["selector_primary_error_tvt"].to_numpy(np.float64)
    full_fixed = full_group["selector_fixed_error_tvt"].to_numpy(np.float64)
    full_codes = full_group["selector_primary_error_code"].to_numpy(np.int16)
    full_margin = full_group["selector_primary_error_margin"].to_numpy(np.float64)
    distribution = candidate_distribution(full_codes)
    dominant_code = int(np.argmax(np.bincount(full_codes, minlength=len(CANDIDATES))))
    dominant_candidate = CANDIDATES[dominant_code]
    dominant_share = distribution[dominant_candidate]
    switches = int(np.sum(full_codes[1:] != full_codes[:-1])) if len(full_codes) > 1 else 0

    final_rmse = rmse_values(full_truth, full_final)
    selector_rmse = rmse_values(full_truth, full_primary)
    likpf_rmse = rmse_values(
        full_truth, full_group["likpf_mean"].to_numpy(np.float64)
    )
    exp226_rmse = rmse_values(
        full_truth, full_group["exp226_k16_tvt"].to_numpy(np.float64)
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
        truth,
        color=REFERENCE_LINE_COLORS["true_tvt"],
        linewidth=2.3,
        label="true TVT",
        zorder=8,
    )
    ax_tvt.plot(
        x,
        final_oof,
        color=REFERENCE_LINE_COLORS["ml_oof"],
        linewidth=2.0,
        label="exp264 corrected Stage D v3 add-only OOF",
        zorder=7,
    )
    ax_tvt.plot(
        x,
        primary_tvt,
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
        group["exp226_k16_tvt"],
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
        0.0,
        color=REFERENCE_LINE_COLORS["grid"],
        linewidth=0.8,
        alpha=0.8,
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
        np.arange(-0.5, len(CANDIDATES) + 0.5, 1.0), cmap.N
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
        f"{well_id} | exp264 OOF RMSE {fmt_metric(final_rmse)} | "
        f"selector top-1 RMSE {fmt_metric(selector_rmse)} | "
        f"LikPF {fmt_metric(likpf_rmse)} | exp226 {fmt_metric(exp226_rmse)}\n"
        f"dominant selector top-1: {CANDIDATE_LABELS[dominant_candidate]} "
        f"({100.0 * dominant_share:.1f}%) | mean confidence margin "
        f"{fmt_metric(np.mean(full_margin), 3)} | switches {switches}"
    )
    fig.suptitle(title, fontsize=11, y=0.995)
    candidate_handles = [
        Patch(color=CANDIDATE_COLORS[index], label=CANDIDATE_LABELS[name])
        for index, name in enumerate(CANDIDATES)
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
        "Caveat: top-1 is a diagnostic candidate path. exp264 final prediction is the rose "
        "LightGBM OOF line; the selector hard top-1 and Stage D worst-well guards failed.",
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
        "final_oof_rmse": rmse_values(full_truth, full_final),
        "control_oof_rmse": rmse_values(full_truth, full_control),
        "primary_error_top1_rmse": rmse_values(full_truth, full_primary),
        "fixed_error_top1_rmse": rmse_values(full_truth, full_fixed),
        "primary_error_margin_mean": float(full_group["selector_primary_error_margin"].mean()),
        "primary_error_margin_p50": float(full_group["selector_primary_error_margin"].quantile(0.5)),
        "primary_error_margin_p90": float(full_group["selector_primary_error_margin"].quantile(0.9)),
        "primary_probability_margin_mean": float(full_group["selector_primary_probability_margin"].mean()),
        "dominant_primary_candidate": dominant_candidate,
        "dominant_primary_candidate_share": distribution[dominant_candidate],
        "primary_candidate_switches": switches,
        "final_squared_error_sum": float(np.square(full_final - full_truth).sum()),
        "control_squared_error_sum": float(np.square(full_control - full_truth).sum()),
        "primary_squared_error_sum": float(np.square(full_primary - full_truth).sum()),
        "fixed_squared_error_sum": float(np.square(full_fixed - full_truth).sum()),
        "plot_path": str(output_path),
    }

# %% [markdown]
# ## 6. Generate all-well plots

# %%
plot_order = typewell_order
if MAX_PLOTS is not None:
    plot_order = plot_order.head(MAX_PLOTS).copy()
indices_by_well = base.groupby("well", sort=False).indices
run_started = time.time()
plot_rows: list[dict[str, Any]] = []

for item in plot_order.itertuples(index=False):
    well_id = str(item.well)
    group = base.iloc[indices_by_well[well_id]].copy()
    output_path = PLOTS_DIR / str(item.plot_filename)
    metrics = plot_one_well(well_id, group, output_path)
    metrics.update(
        {
            "plot_order": int(item.plot_order),
            "typewell_order": int(item.typewell_order),
            "well_order_within_typewell": int(item.well_order_within_typewell),
            "typewell_method": str(item.typewell_method),
            "typewell_threshold": str(item.typewell_threshold),
            "typewell_cluster_id": str(item.typewell_cluster_id),
            "typewell_cluster_size": int(item.typewell_cluster_size),
            "typewell_representative_well_id": str(item.typewell_representative_well_id),
            "plot_filename": str(item.plot_filename),
        }
    )
    plot_rows.append(metrics)
    if len(plot_rows) % 25 == 0 or len(plot_rows) == len(plot_order):
        print(
            f"generated {len(plot_rows)}/{len(plot_order)} plots | elapsed {time.time() - run_started:.1f}s",
            flush=True,
        )

manifest = pd.DataFrame(plot_rows).sort_values("plot_order", kind="mergesort")
if len(manifest) != len(plot_order) or manifest["well"].nunique() != len(plot_order):
    raise ValueError("plot manifest coverage failed")
manifest_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_plot_manifest.csv"
distribution_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_primary_distribution.csv"
zip_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_plots.zip"
manifest.to_csv(manifest_path, index=False)
primary_distribution.to_csv(distribution_path, index=False)
if ZIP_PLOTS:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in manifest["plot_filename"]:
            archive.write(PLOTS_DIR / filename, arcname=filename)

display(manifest.head(10))
display(manifest.sort_values("primary_error_top1_rmse", ascending=False).head(20))

# %% [markdown]
# ## 7. Summary and generated outputs

# %%
processed_rows = int(manifest["rows"].sum())
global_metrics = {
    "rows": processed_rows,
    "wells": int(len(manifest)),
    "final_oof_rmse": float(np.sqrt(manifest["final_squared_error_sum"].sum() / processed_rows)),
    "control_oof_rmse": float(np.sqrt(manifest["control_squared_error_sum"].sum() / processed_rows)),
    "primary_error_top1_rmse": float(np.sqrt(manifest["primary_squared_error_sum"].sum() / processed_rows)),
    "fixed_error_top1_rmse": float(np.sqrt(manifest["fixed_squared_error_sum"].sum() / processed_rows)),
    "primary_error_margin_mean": float(base["selector_primary_error_margin"].mean()),
    "primary_error_margin_p50": float(base["selector_primary_error_margin"].quantile(0.5)),
    "primary_error_margin_p90": float(base["selector_primary_error_margin"].quantile(0.9)),
    "primary_probability_margin_mean": float(base["selector_primary_probability_margin"].mean()),
}
summary = {
    "status": "diagnostic_notebook_completed_not_submitted",
    "created_at_utc": datetime.now(UTC).isoformat(),
    "experiment": EXPERIMENT_NAME,
    "route": "ml_model",
    "notebook": f"{EXPERIMENT_NAME}_{NOTEBOOK_KIND}.ipynb",
    "reference_notebook": "kentookumura/exp238-oof-selector-confidence-probe",
    "reference_script_version_id": 336248071,
    "scope": {
        "plot_wells": len(plot_order),
        "all_wells": EXPECTED_WELLS,
        "rows": processed_rows,
        "runtime_seconds": round(time.time() - run_started, 3),
        "max_plots_override": MAX_PLOTS,
    },
    "selector_contract": {
        "surface": "corrected_stage_c_v6_strict_nested_outer_valid",
        "objectives": ["pred_abs_error", "p_within10"],
        "primary_domain": PRIMARY_DOMAIN,
        "fixed_domain": FIXED_DOMAIN,
        "plotted_surface": "primary_pred_abs_error_top1_and_top2_minus_top1_margin_only",
        "fixed_and_probability_surfaces_plotted": False,
        "all_12_single_hard_domain_used": False,
        "nested_model_count_per_outer_valid_score": 4,
        "score_guard_pass": True,
        "hard_primary_guard_pass": False,
        "hard_primary_top1_is_final_prediction": False,
    },
    "plot_contract": {
        "reference": "exp238 selector-confidence scriptVersionId=336248071",
        "panels": 3,
        "height_ratios": [7.0, 1.5, 0.65],
        "reference_paths_and_colors_unchanged": True,
        "exact_hmm_sigma_band": "mean_plus_minus_2sigma",
        "reference_line_colors": REFERENCE_LINE_COLORS,
    },
    "stage_d_contract": {
        "surface": "corrected_stage_d_v3_selector_compact_addonly_lgb_mean",
        "invalid_stage_d_v2_used": False,
        "worst_well_guard_pass": False,
        "worst_well_regression": 14.482873,
    },
    "global_metrics": global_metrics,
    "primary_candidate_distribution": primary_distribution.to_dict(orient="records"),
    "typewell_ordering": {
        "source_experiment": "exp065",
        "method": COMMON_TYPEWELL_METHOD,
        "threshold": COMMON_TYPEWELL_THRESHOLD,
        "groups": EXPECTED_COMMON_TYPEWELL_GROUPS,
        "filename_pattern": "typewell_{typewell_order:04d}_{well}.png",
    },
    "inputs": {
        "exp072_pfbeam_cache": str(pfbeam_path),
        "stage_d_oof": str(stage_d_oof_path),
        "stage_c_outer_valid_score": str(stage_c_score_path),
        "exp209_exact_hmm": str(exp209_hmm_path),
        "exp226_k16_oof": str(exp226_oof_path),
        "common_typewell_assignments": str(common_typewell_path),
        "sha256": input_sha,
    },
    "artifacts": {
        "plot_directory": str(PLOTS_DIR),
        "plot_manifest": str(manifest_path),
        "primary_distribution": str(distribution_path),
        "plots_zip": str(zip_path) if ZIP_PLOTS else None,
        "summary": str(ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_summary.json"),
    },
    "execution_contract": {
        "model_fits": 0,
        "lightgbm_boosters": 0,
        "candidate_regeneration": 0,
        "submission_generated": False,
        "competition_submitted": False,
        "device": "cpu",
        "internet_required": False,
    },
}
summary_path = ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
summary["artifact_sha256"] = {
    "plot_manifest": sha256_path(manifest_path),
    "primary_distribution": sha256_path(distribution_path),
    "plots_zip": sha256_path(zip_path) if ZIP_PLOTS else None,
    "summary": sha256_path(summary_path),
}

print(json.dumps(global_metrics, indent=2))
print("Generated outputs:")
for name, path in summary["artifacts"].items():
    print(f"- {name}: {path}")
print("Summary SHA256:", summary["artifact_sha256"]["summary"])

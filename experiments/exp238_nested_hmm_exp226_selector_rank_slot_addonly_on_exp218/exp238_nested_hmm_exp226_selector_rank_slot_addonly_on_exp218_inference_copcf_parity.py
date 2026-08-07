# %% [markdown]
# # exp238 hidden-safe copcf parity final inference (GPU)

# %% [markdown]
# ## Contents
# 1. Imports and source resolution
# 2. Current-test exp226 diagnostics
# 3. Frozen train-reference typewell assignment
# 4. Full-train typewell and spatial priors
# 5. Frozen cluster confidence features
# 6. Saved selector and final model contracts
# 7. Current-test context assembly and parity audit
# 8. exp218 current-test feature surface
# 9. Fold-matched selector and saved-final inference
# 10. Submission and generated artifacts

# %%
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from IPython.display import display

PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path("experiments/exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218")
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_DIR = (
    Path("/kaggle/working/artifacts")
    if Path("/kaggle/working").exists()
    else PACKAGE_DIR / "artifacts"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = (
    Path("/kaggle/working/submission.csv")
    if Path("/kaggle/working").exists()
    else PACKAGE_DIR / "submission.csv"
)
STARTED_AT = time.time()

OUTPUT_PREFIX = "exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218"
EXP099_TRAIN_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
EXP065_CLUSTER_ASSIGNMENTS = "common_typewell_cluster_assignments.csv"
EXP114_TRAIN_GEOMETRY = "exp114_spatial_neighbor_prior_signal_audit_well_geometry_summary.csv"
EXPECTED_EX226_DIAGNOSTICS = [
    "exp226_geop_tvt",
    "exp226_gr_delta",
    "exp226_geop_minus_pred",
    "exp226_geop_minus_pred_abs",
]


def import_file(name: str, candidates: list[Path], *, reset_settings: bool = False):
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Cannot resolve {name}: {candidates}")
    if reset_settings:
        sys.modules.pop("settings", None)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_one(filename: str) -> Path:
    local_candidates = [
        PACKAGE_DIR / filename,
        PACKAGE_DIR / "artifacts" / filename,
    ]
    local = [path for path in local_candidates if path.exists() and path.stat().st_size > 0]
    if local:
        return local[0]
    matches = (
        [path for path in Path("/kaggle/input").rglob(filename) if path.stat().st_size > 0]
        if Path("/kaggle/input").exists()
        else []
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {filename}, found {matches}")
    return matches[0]


def cfg_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


engine = import_file(
    "exp238_engine_copcf_parity",
    [PACKAGE_DIR / "nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py"],
)
exp218 = import_file(
    "exp218_features_copcf_parity_final",
    [
        PACKAGE_DIR / "exp218_source/gr_wavelet_rotation_confidence_features_on_exp148.py",
        Path(
            "experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/"
            "gr_wavelet_rotation_confidence_features_on_exp148.py"
        ),
    ],
    reset_settings=True,
)
exp218_settings = import_file(
    "exp218_settings_copcf_parity_final",
    [
        PACKAGE_DIR / "exp218_source/settings.py",
        Path("experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/settings.py"),
    ],
)
exp237_settings = import_file(
    "settings",
    [
        PACKAGE_DIR / "exp237_source/settings.py",
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/settings.py"),
    ],
    reset_settings=True,
)
exp237 = import_file(
    "hmm_exp226_candidate_selector_on_exp183",
    [
        PACKAGE_DIR / "exp237_source/hmm_exp226_candidate_selector_on_exp183.py",
        Path(
            "experiments/exp237_hmm_exp226_candidate_selector_on_exp183/"
            "hmm_exp226_candidate_selector_on_exp183.py"
        ),
    ],
)
rawtest = import_file(
    "exp237_rawtest_inference_copcf_parity",
    [
        PACKAGE_DIR / "exp237_source/rawtest_inference.py",
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/rawtest_inference.py"),
    ],
)
exp145_settings = import_file(
    "settings",
    [
        PACKAGE_DIR / "exp145_source/settings.py",
        Path(
            "experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/"
            "settings.py"
        ),
    ],
    reset_settings=True,
)
exp145 = import_file(
    "exp145_dynamic_generator_copcf_parity_final",
    [
        PACKAGE_DIR / "exp145_source/learned_likelihood_rawtest_feature_generator_parity.py",
        Path(
            "experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/"
            "learned_likelihood_rawtest_feature_generator_parity.py"
        ),
    ],
)
replay = import_file(
    "exp218_public_notebook_replay_audit_copcf_parity",
    [
        PACKAGE_DIR / "exp218_source/public_notebook_replay_audit.py",
        Path(
            "experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/"
            "public_notebook_replay_audit.py"
        ),
    ],
)
exp226 = import_file(
    "exp226_connortynan_k16_reproduction_copcf_parity",
    [
        PACKAGE_DIR / "exp226_source/connortynan_k16_reproduction.py",
        Path(
            "experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_"
            "reproduction/connortynan_k16_reproduction.py"
        ),
    ],
)


# %% [markdown]
# ## 2. Current-test exp226 diagnostics


# %%
def build_current_test_exp226_surface(paths) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit exp226 on full train once and expose its candidate plus four diagnostics."""
    config_candidates = [
        PACKAGE_DIR / "exp226_source/config.yaml",
        Path(
            "experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_"
            "reproduction/config.yaml"
        ),
    ]
    config_path = next((path for path in config_candidates if path.exists()), None)
    if config_path is None:
        raise FileNotFoundError(f"cannot resolve exp226 config: {config_candidates}")
    exp226_config = yaml.safe_load(config_path.read_text())
    params = exp226.params_from_config(exp226_config)
    max_train_wells = cfg_get(exp226_config, "inference.max_train_wells")
    max_test_wells = cfg_get(exp226_config, "inference.max_test_wells")
    if max_train_wells is not None or max_test_wells is not None:
        raise ValueError("exp226 debug well limits are forbidden for current-test inference")

    train_wells = exp226.load_train_wells(paths.train_data_dir, params)
    test_wells = exp226.load_test_wells(paths.test_data_dir, params)
    if not train_wells or not test_wells:
        raise FileNotFoundError(
            f"exp226 wells missing: train={len(train_wells)}, test={len(test_wells)}"
        )
    fields = exp226.build_fields(train_wells, params)
    kappa = exp226.fit_kappa(train_wells, fields, params)
    predictions = {
        well.wid: exp226.predict_well(well, fields, kappa, params) for well in test_wells
    }
    starts = {well.wid: int(well.s + 1) for well in test_wells}

    sample = pd.read_csv(paths.sample_submission_path, dtype={"id": str})
    parts = sample["id"].astype(str).str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("sample_submission id format must be '<well>_<row_idx>'")
    wells = parts[0].astype(str).to_numpy()
    row_indices = parts[1].astype(np.int64).to_numpy()
    values = {
        "exp226_v6_k16_geometry_gr_u_projection": np.empty(len(sample), dtype=np.float32),
        "exp226_geop_tvt": np.empty(len(sample), dtype=np.float32),
        "exp226_gr_delta": np.empty(len(sample), dtype=np.float32),
    }
    for row, (well_id, row_index) in enumerate(zip(wells, row_indices, strict=False)):
        result = predictions.get(str(well_id))
        if result is None:
            raise KeyError(f"exp226 has no current-test prediction for well {well_id}")
        offset = int(row_index) - starts[str(well_id)]
        if offset < 0 or offset >= len(result.pred):
            raise IndexError(
                f"exp226 row offset out of range: well={well_id}, row={row_index}, offset={offset}"
            )
        values["exp226_v6_k16_geometry_gr_u_projection"][row] = result.pred[offset]
        values["exp226_geop_tvt"][row] = result.geop[offset]
        values["exp226_gr_delta"][row] = result.delta[offset]

    surface = pd.DataFrame({"id": sample["id"].astype(str), "well": wells, **values})
    surface["exp226_geop_minus_pred"] = (
        surface["exp226_geop_tvt"].to_numpy(np.float32)
        - surface["exp226_v6_k16_geometry_gr_u_projection"].to_numpy(np.float32)
    ).astype(np.float32)
    surface["exp226_geop_minus_pred_abs"] = np.abs(
        surface["exp226_geop_minus_pred"].to_numpy(np.float32)
    ).astype(np.float32)
    checked = ["exp226_v6_k16_geometry_gr_u_projection", *EXPECTED_EX226_DIAGNOSTICS]
    if not np.isfinite(surface[checked].to_numpy(np.float32)).all():
        raise ValueError("current-test exp226 candidate/diagnostics contain non-finite values")
    return surface, {
        "mode": "full_train_fit_current_test_predict_with_diagnostics",
        "config": str(config_path),
        "train_wells": len(train_wells),
        "test_wells": len(test_wells),
        "rows": len(surface),
        "diagnostic_columns": EXPECTED_EX226_DIAGNOSTICS,
        "kappa": np.asarray(kappa, dtype=float).tolist(),
    }


# %% [markdown]
# ## 3. Frozen train-reference typewell assignment


# %%
@dataclass(frozen=True)
class NativeTypewellSeries:
    well_id: str
    tvt: np.ndarray
    gr: np.ndarray
    gr_quantized: np.ndarray
    median_tvt_step: float


def load_native_typewell_series(data_dir: Path, well_ids: list[str]) -> list[NativeTypewellSeries]:
    series: list[NativeTypewellSeries] = []
    for well_id in sorted(well_ids):
        path = data_dir / f"{well_id}__typewell.csv"
        if not path.exists():
            raise FileNotFoundError(f"typewell file is missing: {path}")
        frame = pd.read_csv(path, usecols=["TVT", "GR"])
        frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
        frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
        frame = frame.dropna(subset=["TVT", "GR"]).sort_values("TVT")
        tvt = frame["TVT"].to_numpy(np.float64)
        gr = frame["GR"].to_numpy(np.float64)
        if len(tvt) < 2:
            raise ValueError(f"typewell has fewer than two valid rows: {path}")
        series.append(
            NativeTypewellSeries(
                well_id=str(well_id),
                tvt=tvt,
                gr=gr,
                gr_quantized=np.rint(gr * 100.0).astype(np.int64),
                median_tvt_step=float(np.median(np.diff(tvt))),
            )
        )
    return series


def rolling_hashes(values: np.ndarray, k: int) -> np.ndarray:
    n = len(values)
    if n < k:
        return np.empty(0, dtype=np.uint64)
    base = 1_000_003
    offset = 2_147_483_647
    mask = (1 << 64) - 1
    hashes = np.empty(n - k + 1, dtype=np.uint64)
    current = 0
    power = 1
    for _ in range(k - 1):
        power = (power * base) & mask
    adjusted = values.astype(np.int64, copy=False)
    for idx in range(k):
        current = (current * base + int(adjusted[idx]) + offset) & mask
    hashes[0] = current
    for idx in range(k, n):
        left = int(adjusted[idx - k]) + offset
        right = int(adjusted[idx]) + offset
        current = ((current - left * power) * base + right) & mask
        hashes[idx - k + 1] = current
    return hashes


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def native_overlap_stats(
    train: NativeTypewellSeries,
    test: NativeTypewellSeries,
    row_lag_test_minus_train: int,
) -> dict[str, Any] | None:
    lag = int(row_lag_test_minus_train)
    train_start = max(0, -lag)
    test_start = train_start + lag
    overlap_rows = min(len(train.gr) - train_start, len(test.gr) - test_start)
    if overlap_rows <= 0:
        return None
    train_end = train_start + overlap_rows
    test_end = test_start + overlap_rows
    exact = train.gr_quantized[train_start:train_end] == test.gr_quantized[test_start:test_end]
    return {
        "train_well": train.well_id,
        "test_well": test.well_id,
        "row_lag_test_minus_train": lag,
        "overlap_rows": int(overlap_rows),
        "overlap_fraction_shorter": float(overlap_rows / max(min(len(train.gr), len(test.gr)), 1)),
        "exact_match_rate": float(exact.mean()),
        "longest_exact_run_rows": int(longest_true_run(exact)),
    }


def discover_test_train_native_overlaps(
    train_series: list[NativeTypewellSeries],
    test_series: list[NativeTypewellSeries],
    *,
    kgram_rows: int,
    max_hash_occurrences: int,
    min_kgram_hits: int,
    min_overlap_rows: int,
    min_overlap_fraction_shorter: float,
) -> pd.DataFrame:
    """Find only train-test edges; test-test edges can never affect hidden inference."""
    train_hashes: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for train_idx, item in enumerate(train_series):
        for position, value in enumerate(rolling_hashes(item.gr_quantized, kgram_rows)):
            train_hashes[int(value)].append((train_idx, position))

    lag_hits: dict[tuple[int, int, int], int] = defaultdict(int)
    for test_idx, item in enumerate(test_series):
        for test_position, value in enumerate(rolling_hashes(item.gr_quantized, kgram_rows)):
            occurrences = train_hashes.get(int(value), [])
            if not occurrences or len(occurrences) + 1 > max_hash_occurrences:
                continue
            for train_idx, train_position in occurrences:
                lag = int(test_position - train_position)
                lag_hits[(train_idx, test_idx, lag)] += 1

    rows: list[dict[str, Any]] = []
    for (train_idx, test_idx, lag), hits in lag_hits.items():
        if hits < min_kgram_hits:
            continue
        stats = native_overlap_stats(train_series[train_idx], test_series[test_idx], lag)
        if stats is None:
            continue
        if int(stats["overlap_rows"]) < min_overlap_rows:
            continue
        if float(stats["overlap_fraction_shorter"]) < min_overlap_fraction_shorter:
            continue
        stats["kgram_hits"] = int(hits)
        rows.append(stats)
    if not rows:
        return pd.DataFrame(
            columns=[
                "train_well",
                "test_well",
                "row_lag_test_minus_train",
                "overlap_rows",
                "overlap_fraction_shorter",
                "exact_match_rate",
                "kgram_hits",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "exact_match_rate",
                "overlap_rows",
                "overlap_fraction_shorter",
                "kgram_hits",
                "test_well",
                "train_well",
            ],
            ascending=[False, False, False, False, True, True],
        )
        .reset_index(drop=True)
    )


def read_cluster_assignments() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = find_one(EXP065_CLUSTER_ASSIGNMENTS)
    frame = pd.read_csv(path, dtype=str)
    required = {"method", "threshold", "cluster_id", "well_id", "cluster_size"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"cluster assignment columns are missing: {missing}")
    frame["well_id"] = frame["well_id"].astype(str)
    frame["cluster_id"] = frame["cluster_id"].astype(str)
    frame["cluster_size"] = (
        pd.to_numeric(frame["cluster_size"], errors="coerce").fillna(0).astype(int)
    )
    return frame, {
        "path": str(path),
        "sha256": engine._sha(path),
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
    }


def assign_test_to_frozen_clusters(
    test_wells: list[str],
    overlap_pairs: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    threshold: str,
    min_cluster_size: int,
) -> tuple[dict[str, str], dict[str, list[str]], pd.DataFrame]:
    subset = assignments[
        (assignments["method"].astype(str) == "native_overlap")
        & (assignments["threshold"].astype(str) == str(threshold))
        & (assignments["cluster_size"] >= int(min_cluster_size))
    ].copy()
    if subset.empty:
        raise ValueError(f"no frozen train clusters for native_overlap threshold={threshold}")
    train_to_cluster = dict(
        zip(subset["well_id"].astype(str), subset["cluster_id"].astype(str), strict=False)
    )
    cluster_to_train = {
        str(cluster): sorted(group["well_id"].astype(str).tolist())
        for cluster, group in subset.groupby("cluster_id", sort=False)
    }
    threshold_value = float(threshold)
    mapping: dict[str, str] = {}
    audit_rows: list[dict[str, Any]] = []
    for test_well in sorted(test_wells):
        edges = overlap_pairs[
            (overlap_pairs["test_well"].astype(str) == str(test_well))
            & (pd.to_numeric(overlap_pairs["exact_match_rate"], errors="coerce") >= threshold_value)
            & (overlap_pairs["train_well"].astype(str).isin(train_to_cluster))
        ].copy()
        if edges.empty:
            audit_rows.append(
                {
                    "threshold": str(threshold),
                    "test_well": str(test_well),
                    "selected_cluster": "",
                    "candidate_clusters": 0,
                    "matched_train_edges": 0,
                    "best_exact_match_rate": np.nan,
                    "best_overlap_rows": 0,
                }
            )
            continue
        edges["cluster_id"] = edges["train_well"].astype(str).map(train_to_cluster)
        ranked: list[dict[str, Any]] = []
        for cluster_id, group in edges.groupby("cluster_id", sort=True):
            order = group.sort_values(
                [
                    "exact_match_rate",
                    "overlap_rows",
                    "overlap_fraction_shorter",
                    "kgram_hits",
                    "train_well",
                ],
                ascending=[False, False, False, False, True],
            )
            best = order.iloc[0]
            ranked.append(
                {
                    "cluster_id": str(cluster_id),
                    "best_exact_match_rate": float(best["exact_match_rate"]),
                    "best_overlap_rows": int(best["overlap_rows"]),
                    "best_overlap_fraction": float(best["overlap_fraction_shorter"]),
                    "kgram_hits": int(best["kgram_hits"]),
                    "matched_train_edges": int(len(group)),
                }
            )
        ranked.sort(
            key=lambda item: (
                -item["best_exact_match_rate"],
                -item["best_overlap_rows"],
                -item["best_overlap_fraction"],
                -item["kgram_hits"],
                item["cluster_id"],
            )
        )
        selected = ranked[0]
        mapping[str(test_well)] = str(selected["cluster_id"])
        audit_rows.append(
            {
                "threshold": str(threshold),
                "test_well": str(test_well),
                "selected_cluster": str(selected["cluster_id"]),
                "candidate_clusters": int(len(ranked)),
                "matched_train_edges": int(selected["matched_train_edges"]),
                "best_exact_match_rate": float(selected["best_exact_match_rate"]),
                "best_overlap_rows": int(selected["best_overlap_rows"]),
            }
        )
    return mapping, cluster_to_train, pd.DataFrame(audit_rows)


# %% [markdown]
# ## 4. Full-train typewell and spatial priors


# %%
def load_train_prior_reference() -> tuple[
    pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, Any]
]:
    path = find_one(EXP099_TRAIN_CACHE)
    required = ["id", "well", "target", "last_known_tvt", "md_since"]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"exp099 train prior reference is missing: {missing}")
    frame = pd.read_csv(
        path,
        usecols=required,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in ["target", "last_known_tvt", "md_since"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    if frame[["target", "last_known_tvt", "md_since"]].isna().any().any():
        raise ValueError("exp099 train prior reference contains missing target/anchor/md")
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for well, group in frame.groupby("well", sort=False):
        order = np.argsort(group["md_since"].to_numpy(np.float32))
        arrays[str(well)] = {
            "md_since": group["md_since"].to_numpy(np.float32)[order],
            "true_delta": group["target"].to_numpy(np.float32)[order],
        }
    return (
        frame,
        arrays,
        {
            "path": str(path),
            "sha256": engine._sha(path),
            "sha256_decompressed": engine._sha(path, decompressed=True),
            "rows": int(len(frame)),
            "wells": int(frame["well"].nunique()),
        },
    )


def interp_neighbor_delta(
    query_md: np.ndarray,
    neighbor_md: np.ndarray,
    neighbor_delta: np.ndarray,
    *,
    require_in_range: bool,
) -> np.ndarray:
    finite = np.isfinite(neighbor_md) & np.isfinite(neighbor_delta)
    if finite.sum() < 2:
        return np.full(len(query_md), np.nan, dtype=np.float32)
    x = neighbor_md[finite].astype(np.float64)
    y = neighbor_delta[finite].astype(np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]
    if len(x) < 2:
        return np.full(len(query_md), np.nan, dtype=np.float32)
    left = np.nan if require_in_range else float(y[0])
    right = np.nan if require_in_range else float(y[-1])
    return np.interp(query_md.astype(np.float64), x, y, left=left, right=right).astype(np.float32)


def build_typewell_prior(
    test_frame: pd.DataFrame,
    train_arrays: dict[str, dict[str, np.ndarray]],
    test_to_cluster: dict[str, str],
    cluster_to_train: dict[str, list[str]],
    *,
    prior_name: str,
    require_in_range: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prior_delta = np.full(len(test_frame), np.nan, dtype=np.float32)
    prior_std = np.full(len(test_frame), np.nan, dtype=np.float32)
    prior_count = np.zeros(len(test_frame), dtype=np.int16)
    prior_neighbor_wells = np.zeros(len(test_frame), dtype=np.int16)
    coverage_rows: list[dict[str, Any]] = []
    for well, group in test_frame.groupby("well", sort=False):
        positions = group.index.to_numpy(np.int64)
        query_md = pd.to_numeric(group["md_since"], errors="coerce").to_numpy(np.float32)
        cluster = test_to_cluster.get(str(well))
        neighbors = [
            neighbor
            for neighbor in cluster_to_train.get(str(cluster), [])
            if neighbor in train_arrays
        ]
        neighbor_values: list[np.ndarray] = []
        for neighbor in neighbors:
            source = train_arrays[neighbor]
            values = interp_neighbor_delta(
                query_md,
                source["md_since"],
                source["true_delta"],
                require_in_range=require_in_range,
            )
            if np.isfinite(values).any():
                neighbor_values.append(values)
        if neighbor_values:
            stacked = np.vstack(neighbor_values)
            counts = np.isfinite(stacked).sum(axis=0)
            valid = counts >= 1
            if valid.any():
                prior_delta[positions[valid]] = np.nanmedian(stacked[:, valid], axis=0).astype(
                    np.float32
                )
                prior_std[positions[valid]] = np.nanstd(stacked[:, valid], axis=0).astype(
                    np.float32
                )
            prior_count[positions] = counts.astype(np.int16)
            prior_neighbor_wells[positions] = np.int16(len(neighbors))
        coverage_rows.append(
            {
                "prior": prior_name,
                "test_well": str(well),
                "cluster_id": "" if cluster is None else str(cluster),
                "train_neighbor_wells": int(len(neighbors)),
                "usable_neighbor_wells": int(len(neighbor_values)),
                "valid_rows": int(np.isfinite(prior_delta[positions]).sum()),
                "rows": int(len(positions)),
            }
        )
    last_known = test_frame["last_known_tvt"].to_numpy(np.float32)
    out = pd.DataFrame(
        {
            "id": test_frame["id"].astype(str),
            "well": test_frame["well"].astype(str),
            f"{prior_name}_prior_delta": prior_delta,
            f"{prior_name}_prior_tvt": (last_known + prior_delta).astype(np.float32),
            f"{prior_name}_prior_std": prior_std,
            f"{prior_name}_prior_count": prior_count,
            f"{prior_name}_neighbor_wells": prior_neighbor_wells,
        }
    )
    return out, {
        "prior": prior_name,
        "valid_rate": float(np.isfinite(prior_delta).mean()),
        "well_coverage": coverage_rows,
    }


def safe_span(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.nanmax(finite) - np.nanmin(finite))


def path_tortuosity(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if finite.sum() < 2:
        return float("nan")
    points = np.column_stack([x[finite], y[finite], z[finite]]).astype(np.float64)
    steps = float(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)).sum())
    chord = float(np.sqrt(np.sum((points[-1] - points[0]) ** 2)))
    return float(steps / chord) if chord > 0.0 else float("nan")


def circular_abs_diff(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return float(abs(math.atan2(math.sin(left - right), math.cos(left - right))))


def build_test_geometry_summary(
    test_frame: pd.DataFrame,
    test_data_dir: Path,
    typewell_cluster_0p999: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing_wells: list[str] = []
    for well, group in test_frame.groupby("well", sort=False):
        path = test_data_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            missing_wells.append(str(well))
            continue
        horizontal = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "TVT_input"])
        for column in horizontal.columns:
            horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
        row_idx = group["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(np.int64).to_numpy()
        if row_idx.max(initial=-1) >= len(horizontal):
            raise IndexError(f"test row index exceeds horizontal rows for {well}")
        tvt_input = horizontal["TVT_input"].to_numpy(np.float32)
        known = np.flatnonzero(np.isfinite(tvt_input))
        if len(known) == 0:
            missing_wells.append(str(well))
            continue
        anchor_idx = int(known[-1])
        start_idx = int(row_idx.min())
        end_idx = int(row_idx.max())
        segment = np.arange(start_idx, end_idx + 1, dtype=np.int64)
        md = horizontal["MD"].to_numpy(np.float64)
        x = horizontal["X"].to_numpy(np.float64)
        y = horizontal["Y"].to_numpy(np.float64)
        z = horizontal["Z"].to_numpy(np.float64)
        sx = x[segment]
        sy = y[segment]
        sz = z[segment]
        smd = md[segment]
        dx = float(x[end_idx] - x[start_idx])
        dy = float(y[end_idx] - y[start_idx])
        dz = float(z[end_idx] - z[start_idx])
        dmd = float(md[end_idx] - md[start_idx])
        azimuth = float(math.atan2(dy, dx)) if np.isfinite(dx) and np.isfinite(dy) else np.nan
        local_end = min(anchor_idx + 10, len(horizontal) - 1)
        local_dx = float(x[local_end] - x[anchor_idx])
        local_dy = float(y[local_end] - y[anchor_idx])
        local_azimuth = (
            float(math.atan2(local_dy, local_dx))
            if np.isfinite(local_dx) and np.isfinite(local_dy)
            else azimuth
        )
        rows.append(
            {
                "well": str(well),
                "typewell_cluster": typewell_cluster_0p999.get(str(well), ""),
                "rows": int(len(group)),
                "anchor_row_idx": anchor_idx,
                "eval_start_row_idx": start_idx,
                "eval_end_row_idx": end_idx,
                "centroid_x": float(np.nanmean(sx)),
                "centroid_y": float(np.nanmean(sy)),
                "centroid_z": float(np.nanmean(sz)),
                "start_x": float(x[start_idx]),
                "start_y": float(y[start_idx]),
                "start_z": float(z[start_idx]),
                "end_x": float(x[end_idx]),
                "end_y": float(y[end_idx]),
                "end_z": float(z[end_idx]),
                "bbox_dx": safe_span(sx),
                "bbox_dy": safe_span(sy),
                "bbox_dz": safe_span(sz),
                "md_span": float(np.nanmax(smd) - np.nanmin(smd)),
                "z_span": dz,
                "dz_dmd": dz / dmd if abs(dmd) > 1.0e-9 else np.nan,
                "azimuth": azimuth,
                "azimuth_sin": float(math.sin(azimuth)) if np.isfinite(azimuth) else np.nan,
                "azimuth_cos": float(math.cos(azimuth)) if np.isfinite(azimuth) else np.nan,
                "local_azimuth": local_azimuth,
                "local_azimuth_sin": (
                    float(math.sin(local_azimuth)) if np.isfinite(local_azimuth) else np.nan
                ),
                "local_azimuth_cos": (
                    float(math.cos(local_azimuth)) if np.isfinite(local_azimuth) else np.nan
                ),
                "tortuosity": path_tortuosity(sx, sy, sz),
                "prefix_tvt_range": safe_span(tvt_input[known]),
                "last_md": float(md[anchor_idx]),
                "last_x": float(x[anchor_idx]),
                "last_y": float(y[anchor_idx]),
                "last_z": float(z[anchor_idx]),
            }
        )
    summary = pd.DataFrame(rows)
    if missing_wells or len(summary) != test_frame["well"].nunique():
        raise ValueError(f"test geometry summary is incomplete: missing={missing_wells}")
    return summary, {
        "test_wells": int(len(summary)),
        "missing_wells": missing_wells,
        "test_test_neighbors_used": False,
    }


def read_train_geometry_summary() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = find_one(EXP114_TRAIN_GEOMETRY)
    frame = pd.read_csv(path, dtype={"well": str})
    if frame["well"].duplicated().any():
        raise ValueError("train geometry summary contains duplicate wells")
    for column in frame.columns:
        if column not in {"well", "typewell_cluster"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, {
        "path": str(path),
        "sha256": engine._sha(path),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
    }


def standardized_matrix(
    summaries: pd.DataFrame,
    wells: list[str],
    features: tuple[str, ...],
    *,
    mean: pd.Series | None = None,
    std: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    values = summaries.set_index("well").reindex(wells).loc[:, list(features)]
    values = values.apply(pd.to_numeric, errors="coerce")
    if mean is None:
        mean = values.mean(axis=0)
    if std is None:
        std = values.std(axis=0, ddof=0).replace(0.0, np.nan)
    filled = values.fillna(mean).fillna(0.0)
    scaled = (filled - mean.fillna(0.0)) / std.fillna(1.0)
    return scaled.to_numpy(np.float64), mean, std


def build_spatial_prior(
    test_frame: pd.DataFrame,
    train_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    train_arrays: dict[str, dict[str, np.ndarray]],
    *,
    prior_name: str,
    features: tuple[str, ...],
    top_k: int,
    min_neighbor_wells: int,
    min_row_neighbor_values: int,
    require_in_range: bool,
    distance_epsilon: float,
    distance_power: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_wells = [well for well in sorted(train_arrays) if well in set(train_summary["well"])]
    train_matrix, mean, std = standardized_matrix(train_summary, train_wells, features)
    train_by_well = train_summary.set_index("well", drop=False)
    test_by_well = test_summary.set_index("well", drop=False)
    prior_delta = np.full(len(test_frame), np.nan, dtype=np.float32)
    prior_std = np.full(len(test_frame), np.nan, dtype=np.float32)
    prior_count = np.zeros(len(test_frame), dtype=np.int16)
    prior_neighbor_wells = np.zeros(len(test_frame), dtype=np.int16)
    prior_distance_min = np.full(len(test_frame), np.nan, dtype=np.float32)
    prior_distance_mean = np.full(len(test_frame), np.nan, dtype=np.float32)
    prior_same_typewell_share = np.full(len(test_frame), np.nan, dtype=np.float32)
    prior_azimuth_mismatch = np.full(len(test_frame), np.nan, dtype=np.float32)
    prior_dz_dmd_mismatch = np.full(len(test_frame), np.nan, dtype=np.float32)
    audit_rows: list[dict[str, Any]] = []

    for well, group in test_frame.groupby("well", sort=False):
        positions = group.index.to_numpy(np.int64)
        query_matrix, _, _ = standardized_matrix(
            test_summary,
            [str(well)],
            features,
            mean=mean,
            std=std,
        )
        distances = np.sqrt(np.sum((train_matrix - query_matrix[0]) ** 2, axis=1))
        order = np.argsort(distances)[: max(int(top_k), 1)]
        neighbors = [train_wells[int(index)] for index in order]
        neighbor_distances = distances[order]
        query_md = pd.to_numeric(group["md_since"], errors="coerce").to_numpy(np.float32)
        values_list: list[np.ndarray] = []
        usable_neighbors: list[str] = []
        usable_distances: list[float] = []
        for neighbor, distance in zip(neighbors, neighbor_distances, strict=False):
            source = train_arrays[neighbor]
            values = interp_neighbor_delta(
                query_md,
                source["md_since"],
                source["true_delta"],
                require_in_range=require_in_range,
            )
            if np.isfinite(values).any():
                values_list.append(values)
                usable_neighbors.append(neighbor)
                usable_distances.append(float(distance))
        if len(usable_neighbors) >= int(min_neighbor_wells):
            stacked = np.vstack(values_list).astype(np.float32)
            distance_array = np.asarray(usable_distances, dtype=np.float64)
            weights = 1.0 / np.power(distance_array + distance_epsilon, distance_power)
            finite = np.isfinite(stacked)
            weighted = np.where(finite, stacked.astype(np.float64) * weights[:, None], 0.0)
            weight_sum = np.where(finite, weights[:, None], 0.0).sum(axis=0)
            counts = finite.sum(axis=0)
            valid = (counts >= int(min_row_neighbor_values)) & (weight_sum > 0.0)
            if valid.any():
                prior_delta[positions[valid]] = (
                    weighted[:, valid].sum(axis=0) / weight_sum[valid]
                ).astype(np.float32)
                prior_std[positions[valid]] = np.nanstd(stacked[:, valid], axis=0).astype(
                    np.float32
                )
            prior_count[positions] = counts.astype(np.int16)
            prior_neighbor_wells[positions] = np.int16(len(usable_neighbors))
            prior_distance_min[positions] = np.float32(np.nanmin(distance_array))
            prior_distance_mean[positions] = np.float32(np.nanmean(distance_array))
            query = test_by_well.loc[str(well)]
            neighbor_summary = train_by_well.loc[usable_neighbors]
            same_typewell = neighbor_summary["typewell_cluster"].astype(str).to_numpy() == str(
                query["typewell_cluster"]
            )
            prior_same_typewell_share[positions] = np.float32(np.mean(same_typewell))
            azimuth_mismatch = [
                circular_abs_diff(float(query["azimuth"]), float(value))
                for value in neighbor_summary["azimuth"].to_numpy()
            ]
            dz_mismatch = np.abs(
                pd.to_numeric(neighbor_summary["dz_dmd"], errors="coerce").to_numpy(np.float64)
                - float(query["dz_dmd"])
            )
            prior_azimuth_mismatch[positions] = np.float32(np.nanmean(azimuth_mismatch))
            prior_dz_dmd_mismatch[positions] = np.float32(np.nanmean(dz_mismatch))
        audit_rows.append(
            {
                "prior": prior_name,
                "test_well": str(well),
                "selected_train_neighbors": " ".join(usable_neighbors),
                "usable_neighbor_wells": int(len(usable_neighbors)),
                "valid_rows": int(np.isfinite(prior_delta[positions]).sum()),
                "rows": int(len(positions)),
            }
        )

    last_known = test_frame["last_known_tvt"].to_numpy(np.float32)
    out = pd.DataFrame(
        {
            "id": test_frame["id"].astype(str),
            "well": test_frame["well"].astype(str),
            f"{prior_name}_prior_delta": prior_delta,
            f"{prior_name}_prior_tvt": (last_known + prior_delta).astype(np.float32),
            f"{prior_name}_prior_std": prior_std,
            f"{prior_name}_prior_count": prior_count,
            f"{prior_name}_neighbor_wells": prior_neighbor_wells,
            f"{prior_name}_distance_min": prior_distance_min,
            f"{prior_name}_distance_mean": prior_distance_mean,
            f"{prior_name}_same_typewell_share": prior_same_typewell_share,
            f"{prior_name}_azimuth_mismatch": prior_azimuth_mismatch,
            f"{prior_name}_dz_dmd_mismatch": prior_dz_dmd_mismatch,
        }
    )
    return out, {
        "prior": prior_name,
        "features": list(features),
        "top_k": int(top_k),
        "train_reference_wells": int(len(train_wells)),
        "valid_rate": float(np.isfinite(prior_delta).mean()),
        "test_test_neighbors_used": False,
        "well_coverage": audit_rows,
    }


# %% [markdown]
# ## 5. Frozen cluster confidence features


# %%
def build_frozen_cluster_query_features(
    train_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    assignments: pd.DataFrame,
    test_to_cluster_1: dict[str, str],
    parent_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    method = str(cfg_get(parent_config, "cluster.assignment_method", "native_overlap"))
    threshold = str(cfg_get(parent_config, "cluster.assignment_threshold", "1"))
    min_cluster_size = int(cfg_get(parent_config, "cluster.min_cluster_size", 2))
    scale_floor = float(cfg_get(parent_config, "cluster.robust_scale_floor_ft", 250.0))
    nearby_k_values = [
        int(value) for value in cfg_get(parent_config, "cluster.nearby_k_values", [5, 8, 12])
    ]
    majority_min_share = float(cfg_get(parent_config, "cluster.nearby_majority_min_share", 0.5))
    subset = assignments[
        (assignments["method"].astype(str) == method)
        & (assignments["threshold"].astype(str) == threshold)
    ].copy()
    assignment_columns = ["well_id", "cluster_id", "cluster_size"]
    train = train_summary.merge(
        subset[assignment_columns].rename(columns={"well_id": "well"}),
        on="well",
        how="left",
        validate="one_to_one",
    )
    train["cluster_size"] = (
        pd.to_numeric(train["cluster_size"], errors="coerce").fillna(0).astype(int)
    )
    train["cluster_id"] = train["cluster_id"].astype("string")
    valid_train = train["cluster_id"].notna() & (train["cluster_size"] >= min_cluster_size)
    stats_rows: list[dict[str, Any]] = []
    for cluster_id, group in train[valid_train].groupby("cluster_id", sort=False):
        x = pd.to_numeric(group["centroid_x"], errors="coerce").to_numpy(np.float64)
        y = pd.to_numeric(group["centroid_y"], errors="coerce").to_numpy(np.float64)
        center_x = float(np.nanmedian(x))
        center_y = float(np.nanmedian(y))
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        stats_rows.append(
            {
                "cluster_id": str(cluster_id),
                "cluster_center_x": center_x,
                "cluster_center_y": center_y,
                "cluster_member_wells": int(len(group)),
                "cluster_dist_median": float(np.nanmedian(distance)),
                "cluster_dist_scale": float(exp237.robust_scale(distance, scale_floor)),
            }
        )
    cluster_stats = pd.DataFrame(stats_rows)
    centers = cluster_stats.set_index("cluster_id", drop=False)
    center_ids = cluster_stats["cluster_id"].astype(str).to_numpy()
    center_xy = cluster_stats[["cluster_center_x", "cluster_center_y"]].to_numpy(np.float64)
    train_xy = train[["centroid_x", "centroid_y"]].to_numpy(np.float64)
    train_clusters = train["cluster_id"].astype("string").to_numpy()

    rows: list[dict[str, Any]] = []
    for test_well in sorted(test_summary["well"].astype(str)):
        query = test_summary.set_index("well").loc[test_well]
        point = np.asarray([query["centroid_x"], query["centroid_y"]], dtype=np.float64)
        own_cluster = test_to_cluster_1.get(str(test_well))
        own_stats = centers.loc[own_cluster] if own_cluster in centers.index else None
        valid_cluster = bool(own_stats is not None)
        own_distance = (
            float(
                np.sqrt(
                    (point[0] - float(own_stats["cluster_center_x"])) ** 2
                    + (point[1] - float(own_stats["cluster_center_y"])) ** 2
                )
            )
            if valid_cluster and np.isfinite(point).all()
            else np.nan
        )
        own_z = (
            float(
                (own_distance - float(own_stats["cluster_dist_median"]))
                / float(own_stats["cluster_dist_scale"])
            )
            if valid_cluster and np.isfinite(own_distance)
            else np.nan
        )
        nearest_other_distance = np.nan
        if valid_cluster and len(center_xy) > 1 and np.isfinite(point).all():
            distance_to_centers = np.sqrt(np.sum((center_xy - point) ** 2, axis=1))
            distance_to_centers[center_ids == str(own_cluster)] = np.inf
            nearest_other_distance = float(np.min(distance_to_centers))
            if not np.isfinite(nearest_other_distance):
                nearest_other_distance = np.nan
        row: dict[str, Any] = {
            "well": str(test_well),
            "cluster_id": "" if own_cluster is None else str(own_cluster),
            "cluster_size": (int(own_stats["cluster_member_wells"]) if valid_cluster else 0),
            "copcf_cluster_feature_valid": np.float32(valid_cluster),
            "copcf_own_cluster_dist": np.float32(own_distance),
            "copcf_own_cluster_dist_z": np.float32(own_z),
            "copcf_nearest_other_cluster_dist": np.float32(nearest_other_distance),
            "copcf_nearest_other_closer": np.float32(
                np.isfinite(nearest_other_distance)
                and np.isfinite(own_distance)
                and nearest_other_distance < own_distance
            ),
        }
        distances = np.sqrt(np.sum((train_xy - point) ** 2, axis=1))
        for k in nearby_k_values:
            if not np.isfinite(distances).any():
                count = 0
                share = 0.0
                differs = False
            else:
                neighbor_index = np.argsort(distances)[:k]
                values = [
                    str(train_clusters[index])
                    for index in neighbor_index
                    if not pd.isna(train_clusters[index])
                ]
                if values:
                    counts = pd.Series(values).value_counts()
                    majority_cluster = str(counts.index[0])
                    count = int(counts.iloc[0])
                    share = float(count / max(len(values), 1))
                    differs = bool(
                        own_cluster is not None
                        and majority_cluster != str(own_cluster)
                        and share >= majority_min_share
                    )
                else:
                    count = 0
                    share = 0.0
                    differs = False
            row[f"copcf_nearby_majority_count_k{k}"] = np.float32(count)
            row[f"copcf_nearby_majority_share_k{k}"] = np.float32(share)
            row[f"copcf_nearby_majority_diff_k{k}"] = np.float32(differs)
        rows.append(row)
    output = pd.DataFrame(rows)
    return output, {
        "method": method,
        "threshold": threshold,
        "min_cluster_size": min_cluster_size,
        "train_reference_wells": int(len(train)),
        "train_clusters": int(len(cluster_stats)),
        "valid_test_wells": int(output["copcf_cluster_feature_valid"].sum()),
        "test_wells": int(len(output)),
        "test_test_neighbors_used": False,
    }


def prefix_prior_frame(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    rename = {
        column: f"copcf_{family}_{column}"
        for column in frame.columns
        if column not in {"id", "well"}
    }
    return frame.rename(columns=rename)


def merge_one_to_one(base: pd.DataFrame, extra: pd.DataFrame, name: str) -> pd.DataFrame:
    payload = [column for column in extra.columns if column not in {"well"}]
    keys = ["id"] if "id" in extra.columns else ["well"]
    if keys == ["id"] and "well" in extra.columns:
        keys = ["id", "well"]
        payload = list(extra.columns)
    before = len(base)
    out = base.merge(extra[payload], on=keys, how="left", validate="one_to_one")
    if len(out) != before:
        raise ValueError(f"{name} merge changed row count")
    return out


def add_current_test_copcf_features(
    frame: pd.DataFrame,
    parent_config: dict[str, Any],
    cluster_features: pd.DataFrame,
    prior_frames: list[tuple[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    out = frame.copy()
    for family, prior in prior_frames:
        prefixed = prefix_prior_frame(prior, family)
        out = out.merge(prefixed, on=["id", "well"], how="left", validate="one_to_one")
    out = out.merge(cluster_features, on="well", how="left", validate="many_to_one")

    settings = cfg_get(parent_config, "ranker.cluster_prior_features", {}) or {}
    priors = exp237.parse_prior_specs(parent_config)
    gates = exp237.parse_cluster_gates(parent_config)
    generated: dict[str, np.ndarray] = {}
    row_cluster_columns = [
        "copcf_cluster_feature_valid",
        "copcf_own_cluster_dist",
        "copcf_own_cluster_dist_z",
        "copcf_nearest_other_cluster_dist",
        "copcf_nearest_other_closer",
    ]
    for column in row_cluster_columns:
        generated[column] = exp237.numeric_array(out, column)
    for k in cfg_get(parent_config, "cluster.nearby_k_values", [5, 8, 12]):
        for suffix in ["count", "share", "diff"]:
            column = f"copcf_nearby_majority_{suffix}_k{int(k)}"
            generated[column] = exp237.numeric_array(out, column)

    for gate in gates:
        gate_column = f"copcf_gate_{gate.name}"
        generated[gate_column] = exp237.cluster_gate_mask(out, gate).astype(np.float32)
        out[gate_column] = generated[gate_column]
        ratio_column = f"copcf_well_gate_ratio_{gate.name}"
        generated[ratio_column] = (
            out.groupby("well", observed=True)[gate_column].transform("mean").to_numpy(np.float32)
        )
        out[ratio_column] = generated[ratio_column]
    gate_columns = [f"copcf_gate_{gate.name}" for gate in gates]
    generated["copcf_any_configured_gate"] = (
        np.column_stack([out[column].to_numpy(np.float32) for column in gate_columns])
        .max(axis=1)
        .astype(np.float32)
    )

    primary_typewell = str(settings.get("primary_typewell_prior"))
    primary_spatial = str(settings.get("primary_spatial_prior"))
    prior_lookup = {prior.name: prior for prior in priors}
    if primary_typewell not in prior_lookup or primary_spatial not in prior_lookup:
        raise ValueError("primary copcf priors are missing from parent config")
    typewell = prior_lookup[primary_typewell]
    spatial = prior_lookup[primary_spatial]
    typewell_column = exp237.prefixed_prior_column(typewell, typewell.prior_tvt)
    spatial_column = exp237.prefixed_prior_column(spatial, spatial.prior_tvt)
    if typewell_column not in out.columns or spatial_column not in out.columns:
        raise ValueError("primary copcf prior TVT columns are missing")
    agreement = exp237.numeric_array(out, typewell_column) - exp237.numeric_array(
        out, spatial_column
    )
    generated["copcf_typewell_spatial_prior_delta"] = agreement.astype(np.float32)
    generated["copcf_typewell_spatial_prior_abs_delta"] = np.abs(agreement).astype(np.float32)

    for prior in priors:
        prior_tvt_column = exp237.prefixed_prior_column(prior, prior.prior_tvt)
        prior_std_column = exp237.prefixed_prior_column(prior, prior.prior_std)
        prior_count_column = exp237.prefixed_prior_column(prior, prior.prior_count)
        prior_neighbors_column = exp237.prefixed_prior_column(prior, prior.neighbor_wells)
        if prior_tvt_column is None or prior_tvt_column not in out.columns:
            raise ValueError(f"current-test prior TVT is missing: {prior.name}")
        prior_tvt = exp237.numeric_array(out, prior_tvt_column)
        generated[f"copcf_{prior.name}_valid_prior"] = np.isfinite(prior_tvt).astype(np.float32)
        if prior_std_column is not None:
            generated[f"copcf_{prior.name}_prior_std"] = exp237.numeric_array(out, prior_std_column)
        if prior_count_column is not None:
            generated[f"copcf_{prior.name}_prior_count"] = exp237.numeric_array(
                out, prior_count_column
            )
        if prior_neighbors_column is not None:
            generated[f"copcf_{prior.name}_neighbor_wells"] = exp237.numeric_array(
                out, prior_neighbors_column
            )

    generated_columns = list(generated)
    for column, values in generated.items():
        numeric = np.asarray(values, dtype=np.float32)
        numeric[~np.isfinite(numeric)] = np.nan
        out[column] = numeric
    expected_count = int(CONFIG["inference"]["rawtest_copcf_parity"]["copcf_feature_count"])
    if len(generated_columns) != expected_count:
        raise ValueError(
            f"copcf feature count mismatch: expected={expected_count}, "
            f"actual={len(generated_columns)}"
        )
    return (
        out,
        generated_columns,
        {
            "feature_count": int(len(generated_columns)),
            "feature_columns": generated_columns,
            "all_nonfinite_columns": [
                column
                for column in generated_columns
                if not np.isfinite(exp237.numeric_array(out, column)).any()
            ],
            "partial_nonfinite_counts": {
                column: int((~np.isfinite(exp237.numeric_array(out, column))).sum())
                for column in generated_columns
                if (~np.isfinite(exp237.numeric_array(out, column))).any()
            },
        },
    )


# %% [markdown]
# ## 6. Saved selector and final model contracts

# %%
parity_config = CONFIG["inference"]["rawtest_copcf_parity"]
final_config = CONFIG["inference"]["copcf_parity_final"]
if bool(final_config["selector_training_during_inference"]):
    raise RuntimeError("copcf parity final inference must not train selectors")
if bool(final_config["final_training_during_inference"]):
    raise RuntimeError("copcf parity final inference must not train final boosters")
if bool(final_config["competition_submit_requested"]):
    raise RuntimeError("this notebook generates submission.csv but must not call the submit API")
if bool(parity_config["test_test_neighbors_allowed"]):
    raise RuntimeError("test-test neighbors would make features depend on hidden test size")
if bool(final_config["test_test_neighbors_allowed"]):
    raise RuntimeError("final inference must preserve the no-test-test-neighbor contract")
if bool(final_config["public_test_row_artifacts_allowed"]):
    raise RuntimeError("public-test row artifacts are forbidden in hidden-safe inference")

parent_config_candidates = [
    PACKAGE_DIR / "exp237_source/config.yaml",
    Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/config.yaml"),
]
parent_config_path = next((path for path in parent_config_candidates if path.exists()), None)
if parent_config_path is None:
    raise FileNotFoundError(f"cannot resolve exp237 config: {parent_config_candidates}")
parent_config = yaml.safe_load(parent_config_path.read_text())
parent_config.setdefault("inference", {})["use_test_base_as_dense_auxiliary"] = True
if not bool(cfg_get(parent_config, "ranker.cluster_prior_features.enabled", False)):
    raise ValueError("exp238 parity requires the original enabled copcf feature family")

paths = exp237_settings.ExperimentPaths()
if Path("/kaggle/input").exists():
    paths.require_kaggle_runtime()
paths.ensure_output_dirs()

selector_summary_path = find_one(f"{OUTPUT_PREFIX}_selector_summary.json")
selector_manifest_path = find_one(f"{OUTPUT_PREFIX}_selector_model_manifest.csv")
final_manifest_path = find_one(f"{OUTPUT_PREFIX}_final_model_manifest.json")
selector_summary = json.loads(selector_summary_path.read_text())
selector_manifest = pd.read_csv(selector_manifest_path)
final_model_manifest = json.loads(final_manifest_path.read_text())
outer_fold_count = int(CONFIG["validation"]["outer_folds"])
inner_fold_count = int(CONFIG["validation"]["inner_folds"])
expected_model_count = outer_fold_count * inner_fold_count
if len(selector_manifest) != expected_model_count:
    raise ValueError(
        f"expected {expected_model_count} saved selectors, got {len(selector_manifest)}"
    )
if int(selector_summary.get("selector_model_count", -1)) != expected_model_count:
    raise ValueError("selector summary does not certify 20 saved models")
if len(final_model_manifest) != int(final_config["saved_final_model_count"]):
    raise ValueError(
        f"expected {final_config['saved_final_model_count']} saved final models, "
        f"got {len(final_model_manifest)}"
    )
candidate_columns = [str(value) for value in selector_summary["candidate_columns"]]
context_columns = [str(value) for value in selector_summary["context_columns"]]
expected_context_count = int(parity_config["context_feature_count"])
if len(context_columns) != expected_context_count:
    raise ValueError(
        f"saved selector context mismatch: expected={expected_context_count}, "
        f"actual={len(context_columns)}"
    )
expected_feature_names = [
    *context_columns,
    "candidate_code",
    "candidate_minus_anchor",
    "candidate_abs_minus_anchor",
]

resolved_models: dict[int, list[tuple[dict[str, Any], Path]]] = {
    outer_fold: [] for outer_fold in range(outer_fold_count)
}
loaded_model_audit: list[dict[str, Any]] = []
for raw_item in selector_manifest.to_dict(orient="records"):
    item = dict(raw_item)
    outer_fold = int(item["outer_fold"])
    inner_fold = int(item["inner_fold"])
    filename = Path(str(item["file"])).name
    model_candidates = list(selector_manifest_path.parent.rglob(filename))
    if not model_candidates and Path("/kaggle/input").exists():
        model_candidates = list(Path("/kaggle/input").rglob(filename))
    if len(model_candidates) != 1:
        raise FileNotFoundError(f"expected one saved selector {filename}, found {model_candidates}")
    model_path = model_candidates[0]
    if engine._sha(model_path) != str(item["sha256"]):
        raise ValueError(f"saved selector SHA mismatch: {filename}")
    manifest_feature_names = json.loads(str(item["feature_names_json"]))
    booster = lgb.Booster(model_file=str(model_path))
    if (
        booster.feature_name() != expected_feature_names
        or manifest_feature_names != expected_feature_names
    ):
        raise ValueError(f"selector feature schema mismatch: {filename}")
    del booster
    resolved_models[outer_fold].append((item, model_path))
    loaded_model_audit.append(
        {
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "file": filename,
            "sha256": str(item["sha256"]),
            "best_iteration": int(item["best_iteration"]),
        }
    )
for outer_fold, items in resolved_models.items():
    inner_folds = sorted(int(item[0]["inner_fold"]) for item in items)
    if inner_folds != list(range(inner_fold_count)):
        raise ValueError(f"outer fold {outer_fold} selector coverage mismatch: {inner_folds}")

resolved_final_models: list[tuple[dict[str, Any], Path]] = []
for raw_item in final_model_manifest:
    item = dict(raw_item)
    filename = Path(str(item["file"])).name
    model_candidates = list(final_manifest_path.parent.rglob(filename))
    if not model_candidates and Path("/kaggle/input").exists():
        model_candidates = list(Path("/kaggle/input").rglob(filename))
    if len(model_candidates) != 1:
        raise FileNotFoundError(
            f"expected one saved final model {filename}, found {model_candidates}"
        )
    model_path = model_candidates[0]
    if engine._sha(model_path) != str(item["sha256"]):
        raise ValueError(f"saved final model SHA mismatch: {filename}")
    resolved_final_models.append((item, model_path))

first_final_booster = lgb.Booster(model_file=str(resolved_final_models[0][1]))
final_feature_columns = list(first_final_booster.feature_name())
del first_final_booster
selector_feature_columns = [
    column for column in final_feature_columns if column.startswith("nsel_")
]
base_feature_columns = [
    column for column in final_feature_columns if not column.startswith("nsel_")
]
if len(base_feature_columns) != int(final_config["base_feature_count"]):
    raise ValueError(f"base feature count mismatch: {len(base_feature_columns)}")
if len(selector_feature_columns) != int(final_config["selector_feature_count"]):
    raise ValueError(f"selector feature count mismatch: {len(selector_feature_columns)}")
if len(final_feature_columns) != int(final_config["final_feature_count"]):
    raise ValueError(f"final feature count mismatch: {len(final_feature_columns)}")
for outer_fold in range(outer_fold_count):
    fold_models = [
        item for item, _path in resolved_final_models if int(item["outer_fold"]) == outer_fold
    ]
    if len(fold_models) != int(CONFIG["model"]["final_configs"]):
        raise ValueError(f"outer fold {outer_fold} final model coverage mismatch")

print(
    json.dumps(
        {
            "mode": parity_config["mode"],
            "selector_train_status": selector_summary["status"],
            "selector_guard_pass": bool(
                selector_summary.get("decision", {}).get("guard_pass", False)
            ),
            "saved_selector_models": len(loaded_model_audit),
            "saved_final_models": len(resolved_final_models),
            "context_features": len(context_columns),
            "final_features": len(final_feature_columns),
            "selector_training_in_this_notebook": False,
            "final_training_in_this_notebook": False,
            "submission_generated": bool(final_config["submission_generated"]),
            "competition_submit_requested": False,
        },
        indent=2,
    )
)


# %% [markdown]
# ## 7. Current-test context assembly and parity audit

# %%
candidates = exp237.candidate_specs_from_config(parent_config)
if [item.column for item in candidates] != candidate_columns:
    raise ValueError("saved selector candidate schema differs from current-test config")

exp218_config_candidates = [
    PACKAGE_DIR / "exp218_source/config.yaml",
    Path("experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/config.yaml"),
]
exp218_config_path = next((path for path in exp218_config_candidates if path.exists()), None)
if exp218_config_path is None:
    raise FileNotFoundError(f"cannot resolve exp218 config: {exp218_config_candidates}")
exp218_config = yaml.safe_load(exp218_config_path.read_text())
replay.configure_public_runtime(
    data_dir=paths.raw_data_dir,
    output_dir=OUTPUT_DIR,
    n_jobs=int(cfg_get(exp218_config, "runtime.num_workers", 8)),
    pf_seeds=int(cfg_get(exp218_config, "generator.rawtest_replay.pf_seeds", 128)),
    pf_particles=int(cfg_get(exp218_config, "generator.rawtest_replay.pf_particles", 500)),
    fast=False,
    use_gpu="auto",
)
base_test, base_meta = replay.build_replay_test_frame()
base_test = base_test.reset_index(drop=True)
base_test["id"] = base_test["id"].astype(str)
base_test["well"] = base_test["well"].astype(str)
test_frame = rawtest._base_candidates(base_test.copy())
test_frame, hmm_meta = rawtest._attach_hmm_candidates(test_frame, parent_config, paths)
exp226_surface, exp226_meta = build_current_test_exp226_surface(paths)
test_frame = rawtest._merge_required(
    test_frame,
    exp226_surface,
    name="dynamic exp226 current-test candidate and diagnostics",
)
test_frame, multiobs_meta = rawtest._attach_multiobs(test_frame, parent_config, paths)
test_frame, _, enrichment_meta = exp237.add_feature_enrichment(
    test_frame, parent_config, max_rows=None
)
test_frame = test_frame.reset_index(drop=True)
if not test_frame[["id", "well"]].equals(base_test[["id", "well"]]):
    raise ValueError("candidate/enrichment assembly changed current-test row order")

assignments, assignment_meta = read_cluster_assignments()
train_reference_wells = sorted(assignments["well_id"].astype(str).unique().tolist())
test_wells = sorted(test_frame["well"].astype(str).unique().tolist())
train_typewells = load_native_typewell_series(paths.train_data_dir, train_reference_wells)
test_typewells = load_native_typewell_series(paths.test_data_dir, test_wells)
native_config = parity_config["native_overlap"]
overlap_pairs = discover_test_train_native_overlaps(
    train_typewells,
    test_typewells,
    kgram_rows=int(native_config["kgram_rows"]),
    max_hash_occurrences=int(native_config["max_hash_occurrences"]),
    min_kgram_hits=int(native_config["min_kgram_hits"]),
    min_overlap_rows=int(native_config["min_overlap_rows"]),
    min_overlap_fraction_shorter=float(native_config["min_overlap_fraction_shorter"]),
)
test_to_cluster: dict[str, dict[str, str]] = {}
cluster_to_train: dict[str, dict[str, list[str]]] = {}
cluster_audits: list[pd.DataFrame] = []
for threshold in [str(value) for value in native_config["thresholds"]]:
    mapping, sources, audit = assign_test_to_frozen_clusters(
        test_wells,
        overlap_pairs,
        assignments,
        threshold=threshold,
        min_cluster_size=int(native_config["min_cluster_size"]),
    )
    test_to_cluster[threshold] = mapping
    cluster_to_train[threshold] = sources
    cluster_audits.append(audit)
cluster_assignment_audit = pd.concat(cluster_audits, ignore_index=True)

_, train_arrays, train_prior_meta = load_train_prior_reference()
typewell_config = parity_config["typewell_prior"]
typewell_1, typewell_1_meta = build_typewell_prior(
    test_frame,
    train_arrays,
    test_to_cluster["1"],
    cluster_to_train["1"],
    prior_name="native_overlap_1",
    require_in_range=bool(typewell_config["require_in_range"]),
)
typewell_0p999, typewell_0p999_meta = build_typewell_prior(
    test_frame,
    train_arrays,
    test_to_cluster["0.999"],
    cluster_to_train["0.999"],
    prior_name="native_overlap_0p999",
    require_in_range=bool(typewell_config["require_in_range"]),
)

test_geometry, test_geometry_meta = build_test_geometry_summary(
    test_frame,
    paths.test_data_dir,
    test_to_cluster["0.999"],
)
train_geometry, train_geometry_meta = read_train_geometry_summary()
spatial_config = parity_config["spatial_prior"]
spatial_priors: dict[str, pd.DataFrame] = {}
spatial_meta: dict[str, Any] = {}
for prior_name, raw_variant in spatial_config["variants"].items():
    prior, meta = build_spatial_prior(
        test_frame,
        train_geometry,
        test_geometry,
        train_arrays,
        prior_name=str(prior_name),
        features=tuple(str(value) for value in raw_variant["features"]),
        top_k=int(raw_variant["top_k"]),
        min_neighbor_wells=int(spatial_config["min_neighbor_wells"]),
        min_row_neighbor_values=int(spatial_config["min_row_neighbor_values"]),
        require_in_range=bool(spatial_config["require_in_range"]),
        distance_epsilon=float(spatial_config["distance_epsilon"]),
        distance_power=float(spatial_config["distance_power"]),
    )
    spatial_priors[str(prior_name)] = prior
    spatial_meta[str(prior_name)] = meta

cluster_features, cluster_feature_meta = build_frozen_cluster_query_features(
    train_geometry,
    test_geometry,
    assignments,
    test_to_cluster["1"],
    parent_config,
)
test_frame, copcf_columns, copcf_meta = add_current_test_copcf_features(
    test_frame,
    parent_config,
    cluster_features,
    [
        ("typewell", typewell_1),
        ("typewell", typewell_0p999),
        ("spatial", spatial_priors["xy_only_k8"]),
        ("spatial", spatial_priors["xy_plus_trajectory_shape_k8"]),
    ],
)
test_frame, test_candidate_values, _ = rawtest._test_candidate_features(test_frame, candidates)
if not np.isfinite(test_candidate_values).all():
    raise ValueError("current-test candidate values contain non-finite values")
if not test_frame[["id", "well"]].equals(base_test[["id", "well"]]):
    raise ValueError("copcf/candidate assembly changed current-test row order")

missing_context = [column for column in context_columns if column not in test_frame.columns]
if missing_context:
    raise ValueError(f"current-test selector context is missing: {missing_context}")
copcf_context = [column for column in context_columns if column.startswith("copcf_")]
if len(copcf_context) != int(parity_config["copcf_feature_count"]):
    raise ValueError(
        f"saved selector copcf schema mismatch: expected={parity_config['copcf_feature_count']}, "
        f"actual={len(copcf_context)}"
    )
if set(copcf_context) != set(copcf_columns):
    raise ValueError("generated copcf columns differ from saved selector copcf schema")
missing_diagnostics = [
    column for column in EXPECTED_EX226_DIAGNOSTICS if column not in context_columns
]
if missing_diagnostics:
    raise ValueError(f"saved selector exp226 diagnostics are missing: {missing_diagnostics}")

partial_nonfinite_counts: dict[str, int] = {}
all_nonfinite_context: list[str] = []
for column in context_columns:
    values = pd.to_numeric(test_frame[column], errors="coerce").to_numpy(np.float32)
    bad_count = int((~np.isfinite(values)).sum())
    if bad_count:
        partial_nonfinite_counts[column] = bad_count
    if bad_count == len(values):
        all_nonfinite_context.append(column)
fully_finite_diagnostic_failures = [
    column
    for column in parity_config["required_fully_finite_features"]
    if column in partial_nonfinite_counts
]
if fully_finite_diagnostic_failures:
    raise ValueError(
        "required current-test diagnostics contain non-finite values: "
        f"{fully_finite_diagnostic_failures}"
    )
finite_copcf_columns = [
    column
    for column in copcf_context
    if np.isfinite(pd.to_numeric(test_frame[column], errors="coerce").to_numpy(np.float32)).any()
]
minimum_finite_copcf = int(parity_config["minimum_copcf_columns_with_any_finite"])
if len(finite_copcf_columns) < minimum_finite_copcf:
    raise ValueError(
        "too few copcf columns have generated finite values: "
        f"minimum={minimum_finite_copcf}, actual={len(finite_copcf_columns)}"
    )

context_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_rawtest_copcf_context.csv.gz"
schema_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_rawtest_copcf_schema.csv"
context_output = test_frame[["id", "well", *context_columns]].copy()
context_output.to_csv(context_path, index=False, compression="gzip")
schema_rows: list[dict[str, Any]] = []
for position, column in enumerate(context_columns):
    values = pd.to_numeric(test_frame[column], errors="coerce").to_numpy(np.float32)
    finite = values[np.isfinite(values)]
    schema_rows.append(
        {
            "position": position,
            "feature": column,
            "family": (
                "copcf"
                if column.startswith("copcf_")
                else "exp226_diagnostic"
                if column in EXPECTED_EX226_DIAGNOSTICS
                else "other"
            ),
            "rows": int(len(values)),
            "finite_count": int(len(finite)),
            "nonfinite_count": int(len(values) - len(finite)),
            "finite_rate": float(len(finite) / max(len(values), 1)),
            "min": float(np.min(finite)) if len(finite) else np.nan,
            "max": float(np.max(finite)) if len(finite) else np.nan,
            "mean": float(np.mean(finite)) if len(finite) else np.nan,
            "std": float(np.std(finite)) if len(finite) else np.nan,
        }
    )
schema_frame = pd.DataFrame(schema_rows)
schema_frame.to_csv(schema_path, index=False)
print(
    {
        "test_rows": len(test_frame),
        "test_wells": test_frame["well"].nunique(),
        "context_features": len(context_columns),
        "copcf_features": len(copcf_context),
        "exp226_diagnostics": len(EXPECTED_EX226_DIAGNOSTICS),
        "missing_context_columns": len(missing_context),
        "all_nonfinite_context_columns": len(all_nonfinite_context),
        "copcf_columns_with_any_finite": len(finite_copcf_columns),
        "partial_nonfinite_context_columns": len(partial_nonfinite_counts),
        "test_test_neighbors_used": False,
    }
)
display(schema_frame[schema_frame["family"].isin(["copcf", "exp226_diagnostic"])])


# %% [markdown]
# ## 8. exp218 current-test feature surface

# %%
selector_surface_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_copcf_parity_selector_surface.csv.gz"
test_frame.to_csv(selector_surface_path, index=False, compression="gzip")
learned_output_dir = OUTPUT_DIR / "exp145_current_test_copcf_parity"
learned_generator_summary = exp145.run_generator(
    output_dir=learned_output_dir,
    mode="rawtest",
    train_cache_path=None,
    rawtest_cache_path=selector_surface_path,
    exp111_schema_path=None,
    exp111_manifest_path=None,
    exp112_schema_path=None,
    max_rows=None,
)
learned_source_path = Path(
    learned_generator_summary["outputs"]["rawtest_ml_features"]["path"]
)
learned_source = pd.read_csv(learned_source_path, dtype={"id": str, "well": str})

final_test_frame, anchor_meta = exp218.add_inference_anchor_columns(
    base_test.copy(), paths.test_data_dir
)
projection_cfg = cfg_get(exp218_config, "model.u_projection", {}) or {}
projection, _, _ = exp218.build_u_projection_features(
    final_test_frame,
    source_specs=dict(projection_cfg.get("sources") or {}),
    degree=int(projection_cfg.get("degree", 3)),
    robust_iters=int(projection_cfg.get("robust_iters", 3)),
    clip_sigma=float(projection_cfg.get("clip_sigma", 4.0)),
)
projection_columns = [
    column for column in projection.columns if column not in {"id", "well"}
]
exp218._assign_aligned_float32_columns(
    final_test_frame, projection.reset_index(drop=True), projection_columns
)

if not exp218.learned_feature_keys_match(learned_source, final_test_frame):
    raise ValueError("dynamic exp145 learned-feature keys differ from current test")
learned, _, _ = exp218.build_learned_likelihood_features(
    learned_source,
    final_test_frame,
    cfg_get(exp218_config, "model.learned_likelihood_features", {}) or {},
)
learned_columns = [column for column in learned.columns if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    final_test_frame, learned.reset_index(drop=True), learned_columns
)

grwr, _, _, grwr_meta = exp218.build_gr_wavelet_rotation_confidence_features(
    final_test_frame,
    train_dir=paths.test_data_dir,
    config=cfg_get(exp218_config, "model.gr_wavelet_rotation_confidence_features", {})
    or {},
)
grwr_columns = [column for column in grwr.columns if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    final_test_frame, grwr.reset_index(drop=True), grwr_columns
)
missing_base = [
    column for column in base_feature_columns if column not in final_test_frame.columns
]
if missing_base:
    raise ValueError(f"raw-test exp218 surface missing model features: {missing_base[:40]}")
if not final_test_frame[["id", "well"]].equals(test_frame[["id", "well"]]):
    raise ValueError("selector and exp218 current-test row keys differ")
if not np.allclose(
    test_frame["last_known_tvt"].to_numpy(np.float32),
    final_test_frame["last_known_tvt"].to_numpy(np.float32),
    atol=1e-3,
    rtol=0.0,
):
    raise ValueError("selector and exp218 current-test anchors differ")
if not np.isfinite(final_test_frame[base_feature_columns].to_numpy(np.float32)).all():
    raise ValueError("exp218 current-test base features contain non-finite values")
print(
    {
        "test_rows": len(final_test_frame),
        "test_wells": int(final_test_frame["well"].nunique()),
        "exp218_base_features": len(base_feature_columns),
        "learned_schema_parity": learned_generator_summary["generated_schema"][
            "schema_parity_pass"
        ],
    }
)
del projection, learned_source, learned, grwr, base_test
gc.collect()


# %% [markdown]
# ## 9. Fold-matched selector and saved-final inference

# %%
test_frame["target"] = np.float32(0.0)
test_rows = np.arange(len(test_frame), dtype=np.int64)
score_columns = [f"pred_error__{name}" for name in candidate_columns]
score_artifacts: list[dict[str, Any]] = []
score_paths: list[Path] = []
pred_delta = np.zeros(len(final_test_frame), dtype=np.float32)
loaded_final_models: list[dict[str, Any]] = []
for outer_fold in range(outer_fold_count):
    score_sum = np.zeros((len(test_frame), len(candidate_columns)), dtype=np.float32)
    for _item, model_path in resolved_models[outer_fold]:
        booster = lgb.Booster(model_file=str(model_path))
        score_sum += engine.predict_candidate_errors(
            booster,
            test_frame,
            test_rows,
            candidate_columns,
            context_columns,
            chunk_rows=int(CONFIG["model"]["selector"]["predict_chunk_rows"]),
        ) / np.float32(inner_fold_count)
        del booster
        gc.collect()
    if not np.isfinite(score_sum).all():
        raise ValueError(f"outer fold {outer_fold} selector scores contain non-finite values")
    artifact = test_frame[[*engine.KEYS, "last_known_tvt", *candidate_columns]].copy()
    for index, name in enumerate(candidate_columns):
        artifact[f"pred_error__{name}"] = score_sum[:, index]
    score_path = OUTPUT_DIR / (
        f"{OUTPUT_PREFIX}_rawtest_copcf_selector_scores_outer{outer_fold}.csv.gz"
    )
    artifact.to_csv(score_path, index=False, compression="gzip")
    score_paths.append(score_path)
    score_artifacts.append(
        {
            "outer_fold": outer_fold,
            "models": inner_fold_count,
            "file": score_path.name,
            "rows": int(len(artifact)),
            "sha256_decompressed": engine._sha(score_path, decompressed=True),
            "score_min": float(artifact[score_columns].to_numpy(np.float32).min()),
            "score_max": float(artifact[score_columns].to_numpy(np.float32).max()),
        }
    )
    rank_slot = engine.rank_slot_features(
        test_frame,
        test_rows,
        score_sum,
        candidate_columns,
    ).reset_index(drop=True)
    if list(rank_slot.columns) != selector_feature_columns:
        raise ValueError(f"outer fold {outer_fold} selector feature schema mismatch")
    if not np.isfinite(rank_slot.to_numpy(np.float32)).all():
        raise ValueError(f"outer fold {outer_fold} selector features contain non-finite values")
    x_matrix = pd.concat(
        [final_test_frame[base_feature_columns].reset_index(drop=True), rank_slot],
        axis=1,
    )
    if list(x_matrix.columns) != final_feature_columns:
        raise ValueError(f"outer fold {outer_fold} final feature order mismatch")
    if not np.isfinite(x_matrix.to_numpy(np.float32)).all():
        raise ValueError(f"outer fold {outer_fold} final feature matrix contains non-finite values")
    fold_final_models = [
        (item, model_path)
        for item, model_path in resolved_final_models
        if int(item["outer_fold"]) == outer_fold
    ]
    if len(fold_final_models) != int(CONFIG["model"]["final_configs"]):
        raise ValueError(f"outer fold {outer_fold} final model coverage mismatch")
    for item, model_path in fold_final_models:
        booster = lgb.Booster(model_file=str(model_path))
        if list(booster.feature_name()) != final_feature_columns:
            raise ValueError(f"final model feature schema mismatch: {model_path.name}")
        pred_delta += booster.predict(
            x_matrix,
            num_iteration=int(item["best_iteration"]),
        ).astype(np.float32) / np.float32(len(resolved_final_models))
        loaded_final_models.append(
            {
                "model": str(item["model"]),
                "outer_fold": outer_fold,
                "selector_score_outer_fold": outer_fold,
                "file": model_path.name,
                "sha256": str(item["sha256"]),
                "best_iteration": int(item["best_iteration"]),
            }
        )
        del booster
        gc.collect()
    display(artifact.head(3))
    display(rank_slot.head(3))
    del artifact, score_sum, rank_slot, x_matrix
    gc.collect()

if len(loaded_final_models) != int(final_config["saved_final_model_count"]):
    raise ValueError(f"loaded final model count mismatch: {len(loaded_final_models)}")
pred_tvt = final_test_frame["last_known_tvt"].to_numpy(np.float32) + pred_delta
if not np.isfinite(pred_tvt).all():
    raise ValueError("final predictions contain non-finite values")
predictions = pd.DataFrame(
    {
        "id": final_test_frame["id"].astype(str),
        "well": final_test_frame["well"].astype(str),
        "last_known_tvt": final_test_frame["last_known_tvt"].to_numpy(np.float32),
        "pred_delta": pred_delta,
        "pred_tvt": pred_tvt,
    }
)


# %% [markdown]
# ## 10. Submission and generated artifacts

# %%
overlap_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_rawtest_copcf_typewell_overlap_edges.csv"
cluster_audit_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_rawtest_copcf_cluster_assignment.csv"
model_audit_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_rawtest_copcf_model_manifest.json"
prediction_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_copcf_parity_inference_test_predictions.csv.gz"
final_schema_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_copcf_parity_inference_feature_schema.csv"
summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_copcf_parity_inference_summary.json"
overlap_pairs.to_csv(overlap_path, index=False)
cluster_assignment_audit.to_csv(cluster_audit_path, index=False)
model_audit_path.write_text(json.dumps(loaded_model_audit, indent=2) + "\n")
predictions.to_csv(prediction_path, index=False, compression="gzip")
pd.DataFrame(
    {
        "feature_index": np.arange(len(final_feature_columns)),
        "feature": final_feature_columns,
    }
).to_csv(final_schema_path, index=False)

sample = pd.read_csv(paths.sample_submission_path, dtype={"id": str})
submission = sample[["id"]].merge(
    predictions[["id", "pred_tvt"]],
    on="id",
    how="left",
    validate="one_to_one",
).rename(columns={"pred_tvt": "tvt"})
if (
    len(submission) != len(sample)
    or submission["id"].duplicated().any()
    or submission["tvt"].isna().any()
    or not np.isfinite(submission["tvt"].to_numpy(np.float64)).all()
):
    raise ValueError("submission contract failed")
submission.to_csv(SUBMISSION_PATH, index=False)

summary = {
    "status": "hidden_safe_copcf_parity_final_inference_completed_not_submitted",
    "mode": final_config["mode"],
    "runtime_seconds": float(time.time() - STARTED_AT),
    "rows": int(len(predictions)),
    "wells": int(predictions["well"].nunique()),
    "candidate_columns": candidate_columns,
    "context_feature_count": len(context_columns),
    "copcf_feature_count": len(copcf_context),
    "exp226_diagnostic_feature_count": len(EXPECTED_EX226_DIAGNOSTICS),
    "selector_model_count": len(loaded_model_audit),
    "final_model_count": len(loaded_final_models),
    "outer_fold_count": outer_fold_count,
    "inner_models_per_outer": inner_fold_count,
    "final_models_per_outer": int(CONFIG["model"]["final_configs"]),
    "base_feature_count": len(base_feature_columns),
    "selector_feature_count": len(selector_feature_columns),
    "final_feature_count": len(final_feature_columns),
    "selector_training_executed": False,
    "final_training_executed": False,
    "final_inference_executed": True,
    "submission_generated": True,
    "competition_submit_executed": False,
    "prediction_min": float(pred_tvt.min()),
    "prediction_max": float(pred_tvt.max()),
    "prediction_mean": float(pred_tvt.mean()),
    "prediction_std": float(pred_tvt.std()),
    "fallback_rows": 0,
    "missing_context_columns": missing_context,
    "all_nonfinite_context_columns": all_nonfinite_context,
    "copcf_columns_with_any_finite": finite_copcf_columns,
    "partial_nonfinite_context_counts": partial_nonfinite_counts,
    "test_test_neighbors_used": False,
    "selector_guard_pass": bool(selector_summary.get("decision", {}).get("guard_pass", False)),
    "context_parity_contract": {
        "pass": bool(
            not missing_context
            and not fully_finite_diagnostic_failures
            and len(copcf_context) == int(parity_config["copcf_feature_count"])
            and len(finite_copcf_columns) >= minimum_finite_copcf
        ),
        "missing_context_columns": len(missing_context),
        "required_fully_finite_diagnostic_failures": fully_finite_diagnostic_failures,
        "copcf_columns": len(copcf_context),
        "copcf_columns_with_any_finite": len(finite_copcf_columns),
        "minimum_copcf_columns_with_any_finite": minimum_finite_copcf,
    },
    "score_artifacts": score_artifacts,
    "loaded_final_models": loaded_final_models,
    "sources": {
        "saved_selector_summary": str(selector_summary_path),
        "saved_selector_manifest": str(selector_manifest_path),
        "saved_final_manifest": str(final_manifest_path),
        "base_test": base_meta,
        "anchor": anchor_meta,
        "hmm": hmm_meta,
        "exp226": exp226_meta,
        "multiobs": multiobs_meta,
        "enrichment": enrichment_meta,
        "learned_dynamic": learned_generator_summary,
        "grwr": exp218._jsonable(grwr_meta),
        "candidate_features": {
            "rows": int(len(test_frame)),
            "candidates": candidate_columns,
            "all_finite": True,
        },
        "cluster_assignments": assignment_meta,
        "train_prior_reference": train_prior_meta,
        "train_geometry": train_geometry_meta,
        "test_geometry": test_geometry_meta,
        "frozen_cluster_features": cluster_feature_meta,
        "typewell_native_overlap_1": typewell_1_meta,
        "typewell_native_overlap_0p999": typewell_0p999_meta,
        "spatial_priors": spatial_meta,
        "copcf": copcf_meta,
    },
    "artifacts": {
        "context": context_path.name,
        "schema": schema_path.name,
        "typewell_overlap_edges": overlap_path.name,
        "cluster_assignment": cluster_audit_path.name,
        "loaded_model_manifest": model_audit_path.name,
        "selector_surface": selector_surface_path.name,
        "prediction": prediction_path.name,
        "final_feature_schema": final_schema_path.name,
        "submission": SUBMISSION_PATH.name,
        "summary": summary_path.name,
    },
    "sha256": {
        "context_decompressed": engine._sha(context_path, decompressed=True),
        "schema": engine._sha(schema_path),
        "typewell_overlap_edges": engine._sha(overlap_path),
        "cluster_assignment": engine._sha(cluster_audit_path),
        "loaded_model_manifest": engine._sha(model_audit_path),
        "train_selector_manifest": engine._sha(selector_manifest_path),
        "train_selector_summary": engine._sha(selector_summary_path),
        "train_final_manifest": engine._sha(final_manifest_path),
        "selector_surface_decompressed": engine._sha(
            selector_surface_path, decompressed=True
        ),
        "predictions_decompressed": engine._sha(prediction_path, decompressed=True),
        "final_feature_schema": engine._sha(final_schema_path),
        "submission": engine._sha(SUBMISSION_PATH),
    },
    "notes": [
        (
            "The original 184-feature exp238 selector schema is preserved; "
            "no train feature is removed."
        ),
        "All 41 copcf features are regenerated from frozen full-train references.",
        "Each test well is transformed independently; test-test edges and neighbors are forbidden.",
        (
            "Natural partial missing values are retained for LightGBM native missing routing, "
            "and their coverage is recorded without median/zero fallback."
        ),
        "All 20 saved selectors and 15 saved final LightGBM models are SHA/schema checked.",
        "No selector, final model, or parent/control is fitted during inference.",
        (
            "submission.csv is generated as notebook output, but the competition "
            "submit API is not called."
        ),
        (
            "The final model remains the original add-only exp238 model; direct "
            "candidate replacement is not introduced here."
        ),
        "Selector safety guard remains failed and is not relaxed by this parity correction.",
    ],
}
summary_path.write_text(json.dumps(exp237.to_jsonable(summary), indent=2, sort_keys=True) + "\n")
print(json.dumps(exp237.to_jsonable(summary), indent=2, sort_keys=True))
display(submission.head(20))

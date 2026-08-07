# %% [markdown]
# # exp282 long-tail prediction-zone self-GR loop-closure readout
#
# This zero-booster diagnostic freezes target-free, same-well GR loop-closure
# edges from 1000+ ft receiver rows to 0-500 ft prediction-zone donor rows. It
# hashes real and stable-shuffled edges before true TVT, folds, hidden-like
# roles, or the fixed exp263 OOF prediction can be attached. It never creates a
# corrected prediction, fitted model, inference output, or submission.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Scientific contract and canonical identity checks
# 4. Target-free GR preprocessing and multiscale matching
# 5. Segment support, confidence, and stable shuffled control
# 6. Post-freeze truth, exp263 fixed OOF, and hidden-like attachment
# 7. Precision, donor-transfer, scope, fold, and guard readouts
# 8. Full Kaggle CPU orchestration and generated artifacts
# 9. Setup and contract preview
# 10. Run diagnostic and report generated artifacts

# %% [markdown]
# ## 1. Imports and fixed experiment contract

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp282_longtail_prediction_zone_self_gr_loop_closure_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SCORE_STAGE_ALLOWED_COLUMNS = frozenset({"MD", "GR", "TVT_input"})
SCORE_STAGE_FORBIDDEN_COLUMNS = frozenset(
    {
        "TVT",
        "target",
        "tvt_true",
        "true_tvt",
        "error",
        "abs_error",
        "oracle",
        "oracle_label",
        "fold",
        "outer_fold",
        "candidate_tvt",
        "exp263_fixed",
        "tvt_pred",
    }
)
EDGE_CONTENT_COLUMNS = [
    "well",
    "receiver_row_idx",
    "receiver_md",
    "receiver_md_since_ft",
    "donor_row_idx",
    "donor_md",
    "donor_md_since_ft",
    "orientation",
    "primary_ncc",
    "primary_second_ncc",
    "primary_gap",
    "aux_half8_donor_row_idx",
    "aux_half8_orientation",
    "aux_half8_ncc",
    "aux_half15_donor_row_idx",
    "aux_half15_orientation",
    "aux_half15_ncc",
    "scale_agreement",
    "segment_run_length",
    "segment_supported",
    "orientation_flip_count",
    "ncc_percentile",
    "gap_percentile",
    "edge_confidence",
    "high_confidence",
    "shuffled_donor_row_idx",
    "truth_attached",
]


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP282_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n")


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp282 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def train_data_dir(config: Mapping[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
    return project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(to_jsonable(dict(value)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_content_sha(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in chosen:
        digest.update(column.encode())
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            array = np.ascontiguousarray(values.to_numpy())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        else:
            for value in values.astype(str):
                digest.update(value.encode())
                digest.update(b"\n")
    return digest.hexdigest()


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "content_sha256": dataframe_content_sha(frame),
    }


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            checked.append(str(path))
            if path.exists() and path.is_file() and path.stat().st_size > 0:
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            checked.append(str(path))
            if path.is_file() and path.stat().st_size > 0:
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def list_horizontal_wells(data_dir: Path) -> list[str]:
    return sorted(
        path.name.removesuffix("__horizontal_well.csv")
        for path in data_dir.glob("*__horizontal_well.csv")
    )


# %% [markdown]
# ## 3. Scientific contract and canonical identity checks


# %%
def validate_scientific_contract(config: Mapping[str, Any]) -> None:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp282 route must be pf_beam")
    if get_nested(config, "lineage.parent") != "exp090_lateral_self_gr_match_pseudotail_probe":
        raise ValueError("exp282 scientific parent must remain exp090")
    if not bool(get_nested(config, "matching.enabled")):
        raise ValueError("exp282 matching implementation must be enabled")
    fixed = {
        "matching.preprocessing.rolling_mean_rows": 5,
        "matching.search.primary_half_window_rows": 25,
        "matching.search.center_stride_rows": 3,
        "execution.active_variants": 1,
        "execution.lightgbm_config_count": 0,
        "execution.trained_fold_count": 0,
        "execution.total_boosters": 0,
        "execution.hmm_variants": 0,
        "execution.pf_variants": 0,
        "execution.receiver_chunk_size": 256,
    }
    for key, expected in fixed.items():
        actual = get_nested(config, key)
        if actual != expected:
            raise ValueError(f"fixed exp282 contract changed at {key}: {actual} != {expected}")
    if [
        int(value) for value in get_nested(config, "matching.search.auxiliary_half_window_rows")
    ] != [8, 15]:
        raise ValueError("exp282 fixes auxiliary half windows [8, 15]")
    if list(get_nested(config, "matching.search.orientations")) != ["forward", "reverse"]:
        raise ValueError("exp282 fixes forward/reverse orientation order")
    confidence = get_nested(config, "matching.confidence") or {}
    expected_confidence = {
        "ncc_percentile": 0.25,
        "best_second_gap_percentile": 0.25,
        "multiscale_agreement": 0.25,
        "segment_supported": 0.25,
    }
    if dict(confidence.get("components") or confidence) != expected_confidence:
        raise ValueError("exp282 fixes four equal-weight confidence components")
    weights = get_nested(config, "data.exp263_stage0.fixed_formula_weights") or {}
    if dict(weights) != {"exp226_k16": 0.5, "likpf_mean": 0.25, "exact_hmm": 0.25}:
        raise ValueError("exp282 fixes the exp263 0.50/0.25/0.25 formula")
    forbidden = (
        bool(get_nested(config, "execution.control_or_parent_retraining")),
        bool(get_nested(config, "execution.gpu")),
        bool(get_nested(config, "execution.inference")),
        bool(get_nested(config, "execution.submission")),
        bool(get_nested(config, "inference.enabled")),
        bool(get_nested(config, "inference.create_submission")),
        bool(get_nested(config, "execution.persist_full_pairwise_matrix")),
    )
    if any(forbidden):
        raise ValueError(
            "exp282 forbids retraining, GPU, inference, submission, and pairwise persistence"
        )


def validate_score_stage_columns(frame: pd.DataFrame) -> None:
    columns = set(frame.columns)
    leaked = sorted(columns.intersection(SCORE_STAGE_FORBIDDEN_COLUMNS))
    unexpected = sorted(columns - SCORE_STAGE_ALLOWED_COLUMNS)
    if leaked:
        raise ValueError(f"target-free score input contains forbidden columns: {leaked}")
    if unexpected:
        raise ValueError(f"target-free score input contains unexpected columns: {unexpected}")
    missing = sorted(SCORE_STAGE_ALLOWED_COLUMNS - columns)
    if missing:
        raise ValueError(f"target-free score input is missing columns: {missing}")


def resolve_exp263_cache_root(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = get_nested(config, "data.exp263_stage0") or {}
    manifest_path = resolve_existing(
        "cache_manifest.json", [str(value) for value in spec.get("manifest_candidates", [])]
    )
    actual_sha = sha256_path(manifest_path)
    expected_sha = str(spec.get("expected_cache_manifest_sha256") or "")
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"exp263 cache manifest SHA mismatch: {actual_sha} != {expected_sha}")
    manifest = json.loads(manifest_path.read_text())
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        int(manifest.get("rows", -1)) != expected_rows
        or int(manifest.get("wells", -1)) != expected_wells
    ):
        raise ValueError("exp263 cache manifest row/well coverage mismatch")
    expected_id_sha = str(spec.get("expected_canonical_id_sha256") or "")
    if expected_id_sha and str(manifest.get("canonical_id_sha256")) != expected_id_sha:
        raise ValueError("exp263 canonical ID SHA mismatch")
    return manifest_path.parent, {
        "name": "exp263_stage0_cache_manifest_target_free_identity",
        "path": str(manifest_path),
        "bytes": manifest_path.stat().st_size,
        "raw_sha256": actual_sha,
        "rows": int(manifest["rows"]),
        "wells": int(manifest["wells"]),
        "canonical_id_sha256": str(manifest["canonical_id_sha256"]),
    }


def exp263_partition_paths(cache_root: Path, candidate_id: str) -> list[Path]:
    paths = sorted((cache_root / "candidate_values" / candidate_id).glob("fold=*/*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no exp263 candidate partitions for {candidate_id} under {cache_root}"
        )
    return paths


def load_canonical_identity_stats(
    cache_root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    primitive = str(get_nested(config, "data.exp263_stage0.identity_primitive"))
    rows: list[dict[str, Any]] = []
    partition_rows = 0
    observed_folds: set[int] = set()
    for path in exp263_partition_paths(cache_root, primitive):
        fold_text = path.parent.name.split("=", maxsplit=1)[-1]
        fold = int(fold_text)
        observed_folds.add(fold)
        frame = pd.read_parquet(path, columns=["well", "well_row_idx"])
        frame["well"] = frame["well"].astype(str)
        frame["well_row_idx"] = pd.to_numeric(frame["well_row_idx"], errors="raise").astype(
            np.int32
        )
        partition_rows += len(frame)
        for well, group in frame.groupby("well", sort=False):
            indices = np.sort(group["well_row_idx"].to_numpy(np.int64))
            if len(indices) == 0 or not np.array_equal(
                indices, np.arange(indices[0], indices[-1] + 1)
            ):
                raise ValueError(f"exp263 canonical rows are not contiguous for {well}")
            rows.append(
                {
                    "well": str(well),
                    "first_row_idx": int(indices[0]),
                    "last_row_idx": int(indices[-1]),
                    "row_count": int(len(indices)),
                    "outer_fold_readout_only": fold,
                }
            )
    stats = pd.DataFrame(rows).sort_values("well", kind="mergesort").reset_index(drop=True)
    if stats["well"].duplicated().any():
        raise ValueError("an exp263 well spans more than one outer-fold partition")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if partition_rows != expected_rows or len(stats) != expected_wells:
        raise ValueError("exp263 identity partition row/well coverage mismatch")
    if sorted(observed_folds) != expected_folds:
        raise ValueError("exp263 identity partition fold coverage mismatch")
    score_safe_stats = stats.drop(columns="outer_fold_readout_only")
    manifest = {
        "name": "exp263_stage0_identity_partitions_score_safe",
        "path": str(cache_root / "candidate_values" / primitive),
        "rows": partition_rows,
        "wells": len(stats),
        "folds_verified_post_reader_boundary": sorted(observed_folds),
        "score_safe_columns": list(score_safe_stats.columns),
        "content_sha256": dataframe_content_sha(score_safe_stats),
    }
    return score_safe_stats, manifest


def validate_well_canonical_identity(
    well: str,
    prediction_start_row: int,
    horizontal_rows: int,
    canonical_stats: pd.DataFrame,
) -> None:
    row = canonical_stats.loc[canonical_stats["well"] == str(well)]
    if len(row) != 1:
        raise ValueError(f"canonical identity is missing or duplicated for well {well}")
    item = row.iloc[0]
    expected_count = horizontal_rows - prediction_start_row
    if (
        int(item["first_row_idx"]) != prediction_start_row
        or int(item["last_row_idx"]) != horizontal_rows - 1
        or int(item["row_count"]) != expected_count
    ):
        raise ValueError(f"raw suffix and exp263 canonical identity differ for well {well}")


# %% [markdown]
# ## 4. Target-free GR preprocessing and multiscale matching


# %%
def load_horizontal_score_safe(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(SCORE_STAGE_ALLOWED_COLUMNS))
    validate_score_stage_columns(frame)
    return frame


def normalize_window_matrix(
    signal: np.ndarray, centers: np.ndarray, half_window: int
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    positions = np.asarray(centers, dtype=np.int64)
    offsets = np.arange(-half_window, half_window + 1, dtype=np.int64)
    if len(positions) == 0:
        return np.empty((0, len(offsets)), dtype=np.float64)
    if positions.min() - half_window < 0 or positions.max() + half_window >= len(values):
        raise ValueError("window center is outside the signal boundary")
    windows = values[positions[:, None] + offsets[None, :]]
    means = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    normalized = (windows - means) / (std + 1.0e-6)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized GR windows must be finite")
    return normalized


def match_one_scale(
    signal: np.ndarray,
    donor_centers: np.ndarray,
    receiver_centers: np.ndarray,
    *,
    half_window: int,
    chunk_size: int,
) -> pd.DataFrame:
    donors = np.asarray(donor_centers, dtype=np.int64)
    receivers = np.asarray(receiver_centers, dtype=np.int64)
    if len(donors) == 0 or len(receivers) == 0:
        raise ValueError("matching requires at least one donor and receiver")
    donor_windows = normalize_window_matrix(signal, donors, half_window)
    candidates = np.concatenate([donor_windows, donor_windows[:, ::-1]], axis=0)
    candidate_donors = np.concatenate([donors, donors])
    candidate_orientation = np.asarray(
        ["forward"] * len(donors) + ["reverse"] * len(donors), dtype=object
    )
    rows: list[pd.DataFrame] = []
    denominator = float(2 * half_window + 1)
    for start in range(0, len(receivers), int(chunk_size)):
        chunk_centers = receivers[start : start + int(chunk_size)]
        receiver_windows = normalize_window_matrix(signal, chunk_centers, half_window)
        scores = receiver_windows @ candidates.T / denominator
        if not np.isfinite(scores).all():
            raise ValueError("NCC score matrix contains non-finite values")
        # Candidate order is forward donor-row ascending, then reverse donor-row
        # ascending. Stable descending sort therefore implements the frozen tie rule.
        order = np.argsort(-scores, axis=1, kind="stable")[:, :2]
        best_slot = order[:, 0]
        second_slot = order[:, 1]
        rows.append(
            pd.DataFrame(
                {
                    "receiver_row_idx": chunk_centers,
                    "donor_row_idx": candidate_donors[best_slot],
                    "orientation": candidate_orientation[best_slot],
                    "best_ncc": scores[np.arange(len(chunk_centers)), best_slot],
                    "second_ncc": scores[np.arange(len(chunk_centers)), second_slot],
                }
            )
        )
    output = pd.concat(rows, ignore_index=True)
    output["best_second_gap"] = output["best_ncc"] - output["second_ncc"]
    return output


def prepare_target_free_signal(frame: pd.DataFrame, rolling_rows: int) -> dict[str, Any]:
    validate_score_stage_columns(frame)
    md = pd.to_numeric(frame["MD"], errors="coerce").to_numpy(np.float64)
    gr_series = pd.to_numeric(frame["GR"], errors="coerce")
    tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce")
    if not np.isfinite(md).all():
        raise ValueError("horizontal MD must be finite")
    known_positions = np.flatnonzero(tvt_input.notna().to_numpy())
    if len(known_positions) == 0:
        raise ValueError("horizontal well has no finite TVT_input prefix")
    last_known_row = int(known_positions[-1])
    prediction_start_row = last_known_row + 1
    if prediction_start_row >= len(frame):
        raise ValueError("horizontal well has no prediction zone")
    interpolated = gr_series.interpolate(limit_direction="both")
    prefix_mean = float(gr_series.iloc[:prediction_start_row].mean())
    full_mean = float(gr_series.mean())
    fallback = prefix_mean if np.isfinite(prefix_mean) else full_mean
    if not np.isfinite(fallback):
        raise ValueError("horizontal well has no finite GR fallback")
    interpolated = interpolated.fillna(fallback)
    smoothed = interpolated.rolling(int(rolling_rows), center=True, min_periods=1).mean()
    signal = smoothed.to_numpy(np.float64)
    if not np.isfinite(signal).all():
        raise ValueError("preprocessed GR signal must be finite")
    last_known_md = float(md[last_known_row])
    return {
        "md": md,
        "signal": signal,
        "prediction_start_row": prediction_start_row,
        "last_known_row": last_known_row,
        "last_known_md": last_known_md,
        "md_since": md - last_known_md,
        "prefix_gr_mean": prefix_mean,
        "full_gr_mean": full_mean,
        "gr_missing_rows": int(gr_series.isna().sum()),
    }


def eligible_centers(prepared: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    md_since = np.asarray(prepared["md_since"], dtype=np.float64)
    row_idx = np.arange(len(md_since), dtype=np.int64)
    prediction_start = int(prepared["prediction_start_row"])
    primary_half = int(get_nested(config, "matching.search.primary_half_window_rows"))
    stride = int(get_nested(config, "matching.search.center_stride_rows"))
    donor_min = float(get_nested(config, "matching.donor_scope.minimum_md_since_ft"))
    donor_max = float(get_nested(config, "matching.donor_scope.maximum_md_since_ft_exclusive"))
    receiver_min = float(get_nested(config, "matching.receiver_scope.minimum_md_since_ft"))
    in_prediction = row_idx >= prediction_start
    fits_primary = (row_idx >= primary_half) & (row_idx + primary_half < len(row_idx))
    on_stride = (row_idx % stride) == 0
    donor_mask = (
        in_prediction & fits_primary & on_stride & (md_since >= donor_min) & (md_since < donor_max)
    )
    receiver_mask = in_prediction & fits_primary & on_stride & (md_since >= receiver_min)
    raw_longtail_mask = in_prediction & (md_since >= receiver_min)
    return {
        "donor_centers": row_idx[donor_mask],
        "receiver_centers": row_idx[receiver_mask],
        "eligible_receiver_centers": int(receiver_mask.sum()),
        "raw_longtail_rows": int(raw_longtail_mask.sum()),
        "excluded_receiver_boundary_or_stride_rows": int(
            raw_longtail_mask.sum() - receiver_mask.sum()
        ),
    }


def build_target_free_edges_for_well(
    well: str,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = prepare_target_free_signal(
        frame, int(get_nested(config, "matching.preprocessing.rolling_mean_rows"))
    )
    pools = eligible_centers(prepared, config)
    donors = pools["donor_centers"]
    receivers = pools["receiver_centers"]
    manifest = {
        "well": str(well),
        "horizontal_rows": len(frame),
        "prediction_start_row": int(prepared["prediction_start_row"]),
        "last_known_row": int(prepared["last_known_row"]),
        "last_known_md": float(prepared["last_known_md"]),
        "donor_centers": len(donors),
        "eligible_receiver_centers": int(pools["eligible_receiver_centers"]),
        "raw_longtail_rows": int(pools["raw_longtail_rows"]),
        "excluded_receiver_boundary_or_stride_rows": int(
            pools["excluded_receiver_boundary_or_stride_rows"]
        ),
        "gr_missing_rows": int(prepared["gr_missing_rows"]),
        "prefix_gr_mean": float(prepared["prefix_gr_mean"]),
        "full_gr_mean": float(prepared["full_gr_mean"]),
    }
    if len(receivers) == 0:
        manifest["generated_edges"] = 0
        return pd.DataFrame(columns=EDGE_CONTENT_COLUMNS), manifest
    if len(donors) == 0:
        raise ValueError(f"well {well} has eligible receivers but no donor centers")
    chunk_size = int(get_nested(config, "execution.receiver_chunk_size"))
    scales = [
        int(get_nested(config, "matching.search.primary_half_window_rows")),
        *[int(value) for value in get_nested(config, "matching.search.auxiliary_half_window_rows")],
    ]
    matches = {
        half: match_one_scale(
            np.asarray(prepared["signal"]),
            donors,
            receivers,
            half_window=half,
            chunk_size=chunk_size,
        )
        for half in scales
    }
    primary_half = scales[0]
    primary = matches[primary_half]
    md = np.asarray(prepared["md"], dtype=np.float64)
    md_since = np.asarray(prepared["md_since"], dtype=np.float64)
    edges = pd.DataFrame(
        {
            "well": str(well),
            "receiver_row_idx": primary["receiver_row_idx"].to_numpy(np.int32),
            "receiver_md": md[primary["receiver_row_idx"].to_numpy(np.int64)],
            "receiver_md_since_ft": md_since[primary["receiver_row_idx"].to_numpy(np.int64)],
            "donor_row_idx": primary["donor_row_idx"].to_numpy(np.int32),
            "donor_md": md[primary["donor_row_idx"].to_numpy(np.int64)],
            "donor_md_since_ft": md_since[primary["donor_row_idx"].to_numpy(np.int64)],
            "orientation": primary["orientation"].astype(str),
            "primary_ncc": primary["best_ncc"].to_numpy(np.float64),
            "primary_second_ncc": primary["second_ncc"].to_numpy(np.float64),
            "primary_gap": primary["best_second_gap"].to_numpy(np.float64),
        }
    )
    agreement_parts: list[np.ndarray] = []
    maximum_distance = int(
        get_nested(config, "matching.scale_agreement.maximum_primary_row_distance")
    )
    require_orientation = bool(
        get_nested(config, "matching.scale_agreement.require_same_orientation")
    )
    for half in scales[1:]:
        auxiliary = matches[half]
        prefix = f"aux_half{half}"
        edges[f"{prefix}_donor_row_idx"] = auxiliary["donor_row_idx"].to_numpy(np.int32)
        edges[f"{prefix}_orientation"] = auxiliary["orientation"].astype(str)
        edges[f"{prefix}_ncc"] = auxiliary["best_ncc"].to_numpy(np.float64)
        agrees = (
            np.abs(edges[f"{prefix}_donor_row_idx"].to_numpy() - edges["donor_row_idx"].to_numpy())
            <= maximum_distance
        )
        if require_orientation:
            agrees &= edges[f"{prefix}_orientation"].to_numpy() == edges["orientation"].to_numpy()
        agreement_parts.append(agrees.astype(np.float64))
    edges["scale_agreement"] = np.mean(np.column_stack(agreement_parts), axis=1)
    manifest["generated_edges"] = len(edges)
    return edges, manifest


# %% [markdown]
# ## 5. Segment support, confidence, and stable shuffled control


# %%
def add_segment_support(edges: pd.DataFrame, *, stride: int, minimum_centers: int) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    output = edges.sort_values(["well", "receiver_row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    run_lengths = np.ones(len(output), dtype=np.int32)
    total_flips = np.zeros(len(output), dtype=np.int32)
    for _, indices in output.groupby("well", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        receiver = output.loc[positions, "receiver_row_idx"].to_numpy(np.int64)
        donor = output.loc[positions, "donor_row_idx"].to_numpy(np.int64)
        orientation = output.loc[positions, "orientation"].astype(str).to_numpy()
        breaks = np.ones(len(positions), dtype=bool)
        for local in range(1, len(positions)):
            adjacent = receiver[local] - receiver[local - 1] == stride
            same_orientation = orientation[local] == orientation[local - 1]
            expected = stride if orientation[local] == "forward" else -stride
            aligned = abs((donor[local] - donor[local - 1]) - expected) <= stride
            breaks[local] = not (adjacent and same_orientation and aligned)
        run_id = np.cumsum(breaks)
        counts = pd.Series(run_id).value_counts().to_dict()
        run_lengths[positions] = np.asarray([counts[value] for value in run_id], dtype=np.int32)
        flip_count = int(
            np.sum((orientation[1:] != orientation[:-1]) & (np.diff(receiver) == stride))
        )
        total_flips[positions] = flip_count
    output["segment_run_length"] = run_lengths
    output["segment_supported"] = output["segment_run_length"] >= int(minimum_centers)
    output["orientation_flip_count"] = total_flips
    return output


def add_target_free_confidence(edges: pd.DataFrame, *, top_fraction: float) -> pd.DataFrame:
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    output = edges.copy()
    output["ncc_percentile"] = output.groupby("well", sort=False)["primary_ncc"].rank(
        method="average", pct=True, ascending=True
    )
    output["gap_percentile"] = output.groupby("well", sort=False)["primary_gap"].rank(
        method="average", pct=True, ascending=True
    )
    output["edge_confidence"] = 0.25 * (
        output["ncc_percentile"].to_numpy(np.float64)
        + output["gap_percentile"].to_numpy(np.float64)
        + output["scale_agreement"].to_numpy(np.float64)
        + output["segment_supported"].astype(np.float64).to_numpy()
    )
    output["high_confidence"] = False
    for _, indices in output.groupby("well", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        count = max(1, int(math.ceil(len(positions) * top_fraction)))
        local = output.loc[positions, ["edge_confidence", "receiver_row_idx"]].copy()
        chosen = local.sort_values(
            ["edge_confidence", "receiver_row_idx"],
            ascending=[False, True],
            kind="mergesort",
        ).index[:count]
        output.loc[chosen, "high_confidence"] = True
    return output


def add_stable_shuffled_control(
    edges: pd.DataFrame,
    *,
    experiment_name: str,
    seed: int,
) -> pd.DataFrame:
    output = edges.copy()
    output["shuffled_donor_row_idx"] = output["donor_row_idx"].to_numpy(np.int32)
    for well, indices in output.groupby("well", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        local_seed = stable_seed(experiment_name, f"seed={seed}", str(well), "donor_shuffle")
        rng = np.random.default_rng(local_seed)
        donors = output.loc[positions, "donor_row_idx"].to_numpy(np.int32)
        output.loc[positions, "shuffled_donor_row_idx"] = rng.permutation(donors)
    output["shuffled_donor_row_idx"] = output["shuffled_donor_row_idx"].astype(np.int32)
    output["truth_attached"] = False
    return output


def assert_frozen_edge_contract(edges: pd.DataFrame) -> str:
    missing = sorted(set(EDGE_CONTENT_COLUMNS) - set(edges.columns))
    if missing:
        raise ValueError(f"frozen edge schema is missing {missing}")
    leaked = sorted(set(edges.columns).intersection(SCORE_STAGE_FORBIDDEN_COLUMNS))
    if leaked:
        raise ValueError(f"frozen target-free edges contain forbidden columns: {leaked}")
    if bool(edges["truth_attached"].astype(bool).any()):
        raise ValueError("target-free edge table cannot have truth_attached=true")
    finite_columns = [
        "primary_ncc",
        "primary_second_ncc",
        "primary_gap",
        "aux_half8_ncc",
        "aux_half15_ncc",
        "scale_agreement",
        "edge_confidence",
    ]
    if not np.isfinite(edges[finite_columns].to_numpy(np.float64)).all():
        raise ValueError("target-free edge table contains non-finite scores")
    return dataframe_content_sha(edges, EDGE_CONTENT_COLUMNS)


# %% [markdown]
# ## 6. Post-freeze truth, exp263 fixed OOF, and hidden-like attachment


# %%
def require_frozen_edge_sha(frozen_edge_sha256: str) -> None:
    if not isinstance(frozen_edge_sha256, str) or len(frozen_edge_sha256) != 64:
        raise ValueError("post-freeze attachment requires a 64-character edge content SHA")


def needed_edge_keys(edges: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for column in ("receiver_row_idx", "donor_row_idx", "shuffled_donor_row_idx"):
        item = edges[["well", column]].rename(columns={column: "well_row_idx"})
        frames.append(item)
    keys = pd.concat(frames, ignore_index=True).drop_duplicates()
    keys["well"] = keys["well"].astype(str)
    keys["well_row_idx"] = pd.to_numeric(keys["well_row_idx"], errors="raise").astype(np.int32)
    return keys.sort_values(["well", "well_row_idx"], kind="mergesort").reset_index(drop=True)


def load_exp263_fixed_for_keys(
    cache_root: Path,
    keys: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    frozen_edge_sha256: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    require_frozen_edge_sha(frozen_edge_sha256)
    weights = {
        str(key): float(value)
        for key, value in dict(
            get_nested(config, "data.exp263_stage0.fixed_formula_weights") or {}
        ).items()
    }
    required_key_frame = keys[["well", "well_row_idx"]].drop_duplicates()
    merged: pd.DataFrame | None = None
    manifests: list[dict[str, Any]] = []
    for candidate_id, weight in weights.items():
        parts: list[pd.DataFrame] = []
        paths = exp263_partition_paths(cache_root, candidate_id)
        for path in paths:
            frame = pd.read_parquet(
                path,
                columns=["well", "well_row_idx", "outer_fold", "candidate_tvt"],
            )
            frame["well"] = frame["well"].astype(str)
            frame["well_row_idx"] = pd.to_numeric(frame["well_row_idx"], errors="raise").astype(
                np.int32
            )
            selected = frame.merge(
                required_key_frame,
                on=["well", "well_row_idx"],
                how="inner",
                validate="one_to_one",
            )
            if not selected.empty:
                parts.append(selected)
        if not parts:
            raise ValueError(f"exp263 candidate {candidate_id} selected zero required keys")
        candidate = pd.concat(parts, ignore_index=True)
        if candidate.duplicated(["well", "well_row_idx"]).any():
            raise ValueError(f"exp263 candidate {candidate_id} has duplicate selected keys")
        candidate = candidate.rename(columns={"candidate_tvt": candidate_id})
        if merged is None:
            merged = candidate[["well", "well_row_idx", "outer_fold", candidate_id]]
        else:
            merged = merged.merge(
                candidate[["well", "well_row_idx", "outer_fold", candidate_id]],
                on=["well", "well_row_idx", "outer_fold"],
                how="inner",
                validate="one_to_one",
            )
        file_rows = pd.DataFrame(
            {
                "path": [str(path) for path in paths],
                "bytes": [path.stat().st_size for path in paths],
                "raw_sha256": [sha256_path(path) for path in paths],
            }
        )
        manifests.append(
            {
                "name": f"exp263_stage0_candidate_{candidate_id}_post_freeze",
                "path": str(cache_root / "candidate_values" / candidate_id),
                "files": len(paths),
                "bytes": int(file_rows["bytes"].sum()),
                "raw_sha256": dataframe_content_sha(file_rows),
                "selected_rows": len(candidate),
                "formula_weight": weight,
            }
        )
    assert merged is not None
    if len(merged) != len(required_key_frame):
        raise ValueError(
            f"exp263 fixed coverage {len(merged)} != required keys {len(required_key_frame)}"
        )
    if not np.isfinite(merged[list(weights)].to_numpy(np.float64)).all():
        raise ValueError("exp263 fixed primitive values must be finite")
    fixed = np.zeros(len(merged), dtype=np.float32)
    for candidate_id, weight in weights.items():
        fixed += np.float32(weight) * merged[candidate_id].to_numpy(np.float32)
    merged["exp263_fixed"] = fixed.astype(np.float64)
    return merged[["well", "well_row_idx", "outer_fold", "exp263_fixed"]], manifests


def load_truth_for_keys(
    raw_dir: Path,
    keys: pd.DataFrame,
    *,
    frozen_edge_sha256: str,
) -> pd.DataFrame:
    require_frozen_edge_sha(frozen_edge_sha256)
    parts: list[pd.DataFrame] = []
    for well, group in keys.groupby("well", sort=True):
        path = raw_dir / f"{well}__horizontal_well.csv"
        truth = pd.read_csv(path, usecols=["TVT"])
        positions = group["well_row_idx"].to_numpy(np.int64)
        if len(positions) and (positions.min() < 0 or positions.max() >= len(truth)):
            raise ValueError(f"truth key outside raw horizontal range for {well}")
        values = pd.to_numeric(truth.iloc[positions]["TVT"], errors="raise").to_numpy(np.float64)
        parts.append(
            pd.DataFrame(
                {
                    "well": str(well),
                    "well_row_idx": positions.astype(np.int32),
                    "true_tvt": values,
                }
            )
        )
    output = pd.concat(parts, ignore_index=True)
    if (
        output.duplicated(["well", "well_row_idx"]).any()
        or not np.isfinite(output["true_tvt"]).all()
    ):
        raise ValueError("post-freeze raw truth must be unique and finite")
    return output


def load_hidden_like_assignments(
    config: Mapping[str, Any], *, frozen_edge_sha256: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_frozen_edge_sha(frozen_edge_sha256)
    spec = get_nested(config, "data.hidden_like") or {}
    path = resolve_existing(
        str(spec["filename"]), [str(value) for value in spec.get("candidates", [])]
    )
    actual_sha = sha256_path(path)
    expected_sha = str(spec.get("expected_sha256") or "")
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("hidden-like assignment SHA mismatch")
    frame = pd.read_csv(path, dtype={"well_id": str}).rename(columns={"well_id": "well"})
    role_columns = [str(value) for value in (spec.get("valid_role_columns") or {}).values()]
    required = {"well", *role_columns}
    if not required.issubset(frame.columns):
        raise ValueError(f"hidden-like assignments missing {sorted(required - set(frame.columns))}")
    if frame["well"].duplicated().any():
        raise ValueError("hidden-like assignment requires one row per well")
    return frame[["well", *role_columns]], {
        "name": "exp115_hidden_like_assignments_post_freeze",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
    }


def attach_role_lookup(
    edges: pd.DataFrame,
    lookup: pd.DataFrame,
    *,
    edge_key: str,
    value_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    right = lookup[["well", "well_row_idx", *value_columns]].rename(
        columns={
            "well_row_idx": edge_key,
            **{column: f"{prefix}_{column}" for column in value_columns},
        }
    )
    return edges.merge(right, on=["well", edge_key], how="left", validate="many_to_one")


def attach_post_freeze_readout(
    edges: pd.DataFrame,
    truth: pd.DataFrame,
    fixed: pd.DataFrame,
    hidden: pd.DataFrame,
    *,
    frozen_edge_sha256: str,
) -> pd.DataFrame:
    require_frozen_edge_sha(frozen_edge_sha256)
    if bool(edges["truth_attached"].astype(bool).any()):
        raise ValueError("edges were already truth-attached")
    output = edges.copy()
    for edge_key, prefix in (
        ("receiver_row_idx", "receiver"),
        ("donor_row_idx", "donor"),
        ("shuffled_donor_row_idx", "shuffled_donor"),
    ):
        output = attach_role_lookup(
            output,
            truth,
            edge_key=edge_key,
            value_columns=["true_tvt"],
            prefix=prefix,
        )
        output = attach_role_lookup(
            output,
            fixed,
            edge_key=edge_key,
            value_columns=["outer_fold", "exp263_fixed"],
            prefix=prefix,
        )
    numeric_required = [
        "receiver_true_tvt",
        "donor_true_tvt",
        "shuffled_donor_true_tvt",
        "receiver_exp263_fixed",
        "donor_exp263_fixed",
        "shuffled_donor_exp263_fixed",
        "receiver_outer_fold",
        "donor_outer_fold",
        "shuffled_donor_outer_fold",
    ]
    if not np.isfinite(output[numeric_required].to_numpy(np.float64)).all():
        raise ValueError("post-freeze truth/control attachment has missing values")
    if not (
        (output["receiver_outer_fold"] == output["donor_outer_fold"])
        & (output["receiver_outer_fold"] == output["shuffled_donor_outer_fold"])
    ).all():
        raise ValueError("same-well donor/receiver rows must share one outer fold")
    output["fold"] = output["receiver_outer_fold"].astype(np.int8)
    output = output.drop(
        columns=["receiver_outer_fold", "donor_outer_fold", "shuffled_donor_outer_fold"]
    )
    output = output.merge(hidden, on="well", how="left", validate="many_to_one")
    output["abs_delta_tvt_real"] = np.abs(output["receiver_true_tvt"] - output["donor_true_tvt"])
    output["abs_delta_tvt_shuffled"] = np.abs(
        output["receiver_true_tvt"] - output["shuffled_donor_true_tvt"]
    )
    output["baseline_squared_error"] = (
        output["receiver_exp263_fixed"] - output["receiver_true_tvt"]
    ) ** 2
    output["donor_transfer_squared_error"] = (
        output["donor_exp263_fixed"] - output["receiver_true_tvt"]
    ) ** 2
    output["shuffled_donor_transfer_squared_error"] = (
        output["shuffled_donor_exp263_fixed"] - output["receiver_true_tvt"]
    ) ** 2
    output["truth_attached"] = True
    output["frozen_edge_content_sha256"] = frozen_edge_sha256
    return output


# %% [markdown]
# ## 7. Precision, donor-transfer, scope, fold, and guard readouts


# %%
def precision_metric_row(frame: pd.DataFrame, *, scope: str, selection: str) -> dict[str, Any]:
    if frame.empty:
        raise ValueError(f"precision scope {scope}/{selection} selected zero edges")
    real = frame["abs_delta_tvt_real"].to_numpy(np.float64)
    shuffled = frame["abs_delta_tvt_shuffled"].to_numpy(np.float64)
    row: dict[str, Any] = {
        "scope": scope,
        "selection": selection,
        "edges": len(frame),
        "wells": int(frame["well"].nunique()),
        "real_abs_delta_mean": float(np.mean(real)),
        "real_abs_delta_median": float(np.median(real)),
        "real_abs_delta_rmse": float(np.sqrt(np.mean(real**2))),
        "real_abs_delta_p90": float(np.quantile(real, 0.90)),
        "real_abs_delta_p95": float(np.quantile(real, 0.95)),
        "shuffled_abs_delta_mean": float(np.mean(shuffled)),
        "shuffled_abs_delta_median": float(np.median(shuffled)),
        "shuffled_abs_delta_rmse": float(np.sqrt(np.mean(shuffled**2))),
        "shuffled_abs_delta_p90": float(np.quantile(shuffled, 0.90)),
        "shuffled_abs_delta_p95": float(np.quantile(shuffled, 0.95)),
    }
    for threshold in (2, 5, 10):
        real_precision = float(np.mean(real <= threshold))
        shuffled_precision = float(np.mean(shuffled <= threshold))
        row[f"real_within{threshold}"] = real_precision
        row[f"shuffled_within{threshold}"] = shuffled_precision
        row[f"within{threshold}_lift_vs_shuffled"] = real_precision - shuffled_precision
    return row


def donor_transfer_metric_row(frame: pd.DataFrame, *, scope: str, selection: str) -> dict[str, Any]:
    if frame.empty:
        raise ValueError(f"donor-transfer scope {scope}/{selection} selected zero edges")
    baseline_rmse = float(np.sqrt(frame["baseline_squared_error"].mean()))
    donor_rmse = float(np.sqrt(frame["donor_transfer_squared_error"].mean()))
    shuffled_rmse = float(np.sqrt(frame["shuffled_donor_transfer_squared_error"].mean()))
    return {
        "scope": scope,
        "selection": selection,
        "edges": len(frame),
        "wells": int(frame["well"].nunique()),
        "receiver_baseline_rmse": baseline_rmse,
        "matched_donor_transfer_rmse": donor_rmse,
        "shuffled_donor_transfer_rmse": shuffled_rmse,
        "matched_gain_vs_receiver_baseline_ft": baseline_rmse - donor_rmse,
        "matched_gain_vs_shuffled_donor_ft": shuffled_rmse - donor_rmse,
    }


def selection_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "all_edges": np.ones(len(frame), dtype=bool),
        "high_confidence": frame["high_confidence"].astype(bool).to_numpy(),
        "segment_supported": frame["segment_supported"].astype(bool).to_numpy(),
    }


def build_readout_tables(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    selections = selection_masks(frame)
    overall_rows = [
        precision_metric_row(frame.loc[mask], scope="overall", selection=name)
        for name, mask in selections.items()
        if bool(mask.any())
    ]
    fold_rows: list[dict[str, Any]] = []
    donor_rows: list[dict[str, Any]] = []
    for fold, part in frame.groupby("fold", sort=True):
        for name, mask in selection_masks(part).items():
            if not bool(mask.any()):
                continue
            row = precision_metric_row(part.loc[mask], scope=f"fold_{int(fold)}", selection=name)
            row["fold"] = int(fold)
            fold_rows.append(row)
            donor = donor_transfer_metric_row(
                part.loc[mask], scope=f"fold_{int(fold)}", selection=name
            )
            donor["fold"] = int(fold)
            donor_rows.append(donor)
    for name, mask in selections.items():
        if bool(mask.any()):
            donor_rows.append(
                donor_transfer_metric_row(frame.loc[mask], scope="overall", selection=name)
            )

    distance_specs = get_nested(config, "audit.distance_scopes") or {}
    scope_masks: dict[str, np.ndarray] = {}
    for name, spec in distance_specs.items():
        lower = float(spec["minimum_md_since_ft"])
        upper_raw = spec.get("maximum_md_since_ft")
        mask = frame["receiver_md_since_ft"].to_numpy(np.float64) >= lower
        if upper_raw is not None:
            mask &= frame["receiver_md_since_ft"].to_numpy(np.float64) < float(upper_raw)
        scope_masks[str(name)] = mask
    hidden_specs = get_nested(config, "data.hidden_like.valid_role_columns") or {}
    for name, role_column in hidden_specs.items():
        scope_masks[str(name)] = frame[str(role_column)].astype(str).eq("valid").to_numpy()
    scope_rows: list[dict[str, Any]] = []
    for scope, scope_mask in scope_masks.items():
        for selection, selection_mask in selections.items():
            mask = scope_mask & selection_mask
            if not bool(mask.any()):
                continue
            scope_rows.append(
                precision_metric_row(frame.loc[mask], scope=scope, selection=selection)
            )
            donor_rows.append(
                donor_transfer_metric_row(frame.loc[mask], scope=scope, selection=selection)
            )

    orientation_rows: list[dict[str, Any]] = []
    for orientation, part in frame.groupby("orientation", sort=True):
        for name, mask in selection_masks(part).items():
            if bool(mask.any()):
                orientation_rows.append(
                    precision_metric_row(part.loc[mask], scope=str(orientation), selection=name)
                )

    by_well_rows: list[dict[str, Any]] = []
    for well, part in frame.groupby("well", sort=True):
        for name, mask in selection_masks(part).items():
            if not bool(mask.any()):
                continue
            row = precision_metric_row(part.loc[mask], scope=str(well), selection=name)
            transfer = donor_transfer_metric_row(part.loc[mask], scope=str(well), selection=name)
            row.update(
                {
                    "well": str(well),
                    "fold": int(part["fold"].iloc[0]),
                    "receiver_baseline_rmse": transfer["receiver_baseline_rmse"],
                    "matched_donor_transfer_rmse": transfer["matched_donor_transfer_rmse"],
                    "matched_gain_vs_receiver_baseline_ft": transfer[
                        "matched_gain_vs_receiver_baseline_ft"
                    ],
                }
            )
            by_well_rows.append(row)
    return {
        "overall_metrics": pd.DataFrame(overall_rows),
        "fold_metrics": pd.DataFrame(fold_rows),
        "scope_metrics": pd.DataFrame(scope_rows),
        "orientation_metrics": pd.DataFrame(orientation_rows),
        "by_well_metrics": pd.DataFrame(by_well_rows),
        "donor_transfer_readout": pd.DataFrame(donor_rows),
    }


def evaluate_guard(
    frame: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    well_manifest: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    guards = get_nested(config, "validation.guards") or {}
    fold_metrics = tables["fold_metrics"]
    overall = tables["overall_metrics"]
    scope = tables["scope_metrics"]
    donor = tables["donor_transfer_readout"]
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    actual_folds = sorted(int(value) for value in frame["fold"].unique())
    generated_edges = int(len(frame))
    eligible_centers = int(well_manifest["eligible_receiver_centers"].sum())
    longtail_rows = int(well_manifest["raw_longtail_rows"].sum())
    high_edges = int(frame["high_confidence"].sum())
    finite_columns = [
        "primary_ncc",
        "primary_second_ncc",
        "primary_gap",
        "aux_half8_ncc",
        "aux_half15_ncc",
        "edge_confidence",
    ]
    finite_coverage = float(np.isfinite(frame[finite_columns].to_numpy(np.float64)).mean())
    edge_coverage = float(generated_edges / eligible_centers) if eligible_centers else 1.0
    high_receiver_coverage = float(high_edges / longtail_rows) if longtail_rows else 0.0

    def select_one(table: pd.DataFrame, scope_name: str, selection: str) -> pd.Series:
        selected = table.loc[
            (table["scope"].astype(str) == scope_name)
            & (table["selection"].astype(str) == selection)
        ]
        if len(selected) != 1:
            raise ValueError(f"expected one metric row for {scope_name}/{selection}")
        return selected.iloc[0]

    high_overall = select_one(overall, "overall", "high_confidence")
    all_folds = fold_metrics.loc[fold_metrics["selection"] == "all_edges"]
    high_folds = fold_metrics.loc[fold_metrics["selection"] == "high_confidence"]
    high_transfer_folds = donor.loc[
        donor["scope"].astype(str).str.startswith("fold_")
        & (donor["selection"] == "high_confidence")
    ]
    high_transfer_overall = select_one(donor, "overall", "high_confidence")
    hidden_checks: dict[str, bool] = {}
    for hidden_scope in (
        "verification_like_spatial",
        "verification_like_typewell_purged",
    ):
        row = select_one(scope, hidden_scope, "high_confidence")
        hidden_checks[f"{hidden_scope}_positive_within10_lift"] = bool(
            row["within10_lift_vs_shuffled"] > 0.0
        )
    scientific_checks = {
        "high_confidence_within10_precision": bool(
            high_overall["real_within10"]
            >= float(guards["minimum_high_confidence_within10_precision"])
        ),
        "all_edge_positive_lift_5_of_5_folds": bool(
            len(all_folds) == len(expected_folds)
            and (all_folds["within10_lift_vs_shuffled"] > 0.0).all()
        ),
        "high_confidence_positive_lift_5_of_5_folds": bool(
            len(high_folds) == len(expected_folds)
            and (high_folds["within10_lift_vs_shuffled"] > 0.0).all()
        ),
        "high_confidence_lower_median_delta_5_of_5_folds": bool(
            len(high_folds) == len(expected_folds)
            and (
                high_folds["real_abs_delta_median"] < high_folds["shuffled_abs_delta_median"]
            ).all()
        ),
        "high_confidence_receiver_coverage": bool(
            high_receiver_coverage >= float(guards["minimum_high_confidence_receiver_coverage"])
        ),
        "donor_transfer_improved_5_of_5_folds": bool(
            len(high_transfer_folds) == len(expected_folds)
            and (high_transfer_folds["matched_gain_vs_receiver_baseline_ft"] > 0.0).all()
        ),
        "pooled_donor_transfer_gain": bool(
            high_transfer_overall["matched_gain_vs_receiver_baseline_ft"]
            >= float(guards["minimum_pooled_donor_transfer_rmse_gain_ft"])
        ),
        **hidden_checks,
    }
    technical_checks = {
        "canonical_expected_folds": actual_folds == expected_folds,
        "eligible_receiver_edge_coverage": edge_coverage >= float(guards["required_edge_coverage"]),
        "finite_score_coverage": finite_coverage >= float(guards["required_finite_score_coverage"]),
        "forbidden_score_stage_columns": True,
        "truth_attachment_before_edge_freeze_zero": True,
    }
    return {
        "passed": bool(all(technical_checks.values()) and all(scientific_checks.values())),
        "technical_passed": bool(all(technical_checks.values())),
        "scientific_passed": bool(all(scientific_checks.values())),
        "technical_checks": technical_checks,
        "scientific_checks": scientific_checks,
        "actual_folds": actual_folds,
        "edge_coverage": edge_coverage,
        "finite_score_coverage": finite_coverage,
        "eligible_receiver_centers": eligible_centers,
        "generated_edges": generated_edges,
        "raw_longtail_receiver_rows": longtail_rows,
        "high_confidence_edges": high_edges,
        "high_confidence_receiver_coverage": high_receiver_coverage,
        "pooled_high_confidence_within10": float(high_overall["real_within10"]),
        "pooled_high_confidence_within10_lift": float(high_overall["within10_lift_vs_shuffled"]),
        "pooled_high_confidence_donor_transfer_gain_ft": float(
            high_transfer_overall["matched_gain_vs_receiver_baseline_ft"]
        ),
    }


# %% [markdown]
# ## 8. Full Kaggle CPU orchestration and generated artifacts


# %%
def run_full_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp282 readout must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 is reserved "
            "for an explicitly approved local smoke run."
        )
    if not bool(get_nested(config, "execution.kaggle_push_approved")):
        raise RuntimeError("exp282 Kaggle CPU execution is not approved")
    validate_scientific_contract(config)
    started = time.time()
    cache_root, cache_manifest = resolve_exp263_cache_root(config)
    canonical_stats, identity_manifest = load_canonical_identity_stats(cache_root, config)
    raw_dir = train_data_dir(config)
    wells = list_horizontal_wells(raw_dir)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(wells) != expected_wells or set(wells) != set(canonical_stats["well"]):
        raise ValueError("raw horizontal and exp263 canonical well sets differ")

    edge_parts: list[pd.DataFrame] = []
    well_rows: list[dict[str, Any]] = []
    progress_every = int(get_nested(config, "execution.progress_every_wells") or 25)
    for index, well in enumerate(wells, start=1):
        horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
        safe = load_horizontal_score_safe(horizontal_path)
        prepared = prepare_target_free_signal(
            safe, int(get_nested(config, "matching.preprocessing.rolling_mean_rows"))
        )
        validate_well_canonical_identity(
            well,
            int(prepared["prediction_start_row"]),
            len(safe),
            canonical_stats,
        )
        edges, manifest = build_target_free_edges_for_well(well, safe, config)
        manifest.update(
            {
                "horizontal_path": str(horizontal_path),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "canonical_identity_match": True,
            }
        )
        well_rows.append(manifest)
        if not edges.empty:
            edge_parts.append(edges)
        if index % progress_every == 0 or index == len(wells):
            print(f"target-free loop-closure wells={index}/{len(wells)}")
    if not edge_parts:
        raise ValueError("exp282 generated zero target-free edges")
    edges = pd.concat(edge_parts, ignore_index=True)
    edges = add_segment_support(
        edges,
        stride=int(get_nested(config, "matching.search.center_stride_rows")),
        minimum_centers=int(
            get_nested(config, "matching.segment_support.minimum_consecutive_centers")
        ),
    )
    edges = add_target_free_confidence(
        edges,
        top_fraction=float(get_nested(config, "validation.confidence_scope.top_fraction")),
    )
    edges = add_stable_shuffled_control(
        edges,
        experiment_name=EXPERIMENT_NAME,
        seed=int(get_nested(config, "validation.seed")),
    )
    edges = edges.sort_values(["well", "receiver_row_idx"], kind="mergesort").reset_index(drop=True)
    frozen_edge_sha = assert_frozen_edge_contract(edges)

    artifacts = artifact_dir()
    edge_contract = {
        "experiment": EXPERIMENT_NAME,
        "truth_attached": False,
        "forbidden_score_stage_columns_observed": [],
        "truth_attachment_before_freeze_count": 0,
        "matching": get_nested(config, "matching"),
        "confidence_scope": get_nested(config, "validation.confidence_scope"),
        "seed_policy": get_nested(config, "reproducibility.seed_policy"),
        "edge_content_columns": EDGE_CONTENT_COLUMNS,
        "edge_content_sha256": frozen_edge_sha,
    }
    edge_contract["scientific_contract_sha256"] = mapping_sha256(edge_contract)
    edge_contract_path = artifacts / f"{OUTPUT_PREFIX}_edge_contract.json"
    write_json(edge_contract_path, edge_contract)
    edge_artifact = write_csv_gzip(
        edges[EDGE_CONTENT_COLUMNS],
        artifacts / f"{OUTPUT_PREFIX}_target_free_edges.csv.gz",
    )
    edge_schema = pd.DataFrame(
        {
            "column": EDGE_CONTENT_COLUMNS,
            "dtype": [str(edges[column].dtype) for column in EDGE_CONTENT_COLUMNS],
            "truth_attached": False,
        }
    )
    edge_schema_path = artifacts / f"{OUTPUT_PREFIX}_edge_schema.csv"
    edge_schema.to_csv(edge_schema_path, index=False)

    # Outcome-like information is first read below, after every real and shuffled
    # edge has been persisted and the logical content SHA has been fixed.
    keys = needed_edge_keys(edges)
    truth = load_truth_for_keys(raw_dir, keys, frozen_edge_sha256=frozen_edge_sha)
    fixed, fixed_manifests = load_exp263_fixed_for_keys(
        cache_root, keys, config, frozen_edge_sha256=frozen_edge_sha
    )
    hidden, hidden_manifest = load_hidden_like_assignments(
        config, frozen_edge_sha256=frozen_edge_sha
    )
    readout = attach_post_freeze_readout(
        edges,
        truth,
        fixed,
        hidden,
        frozen_edge_sha256=frozen_edge_sha,
    )
    tables = build_readout_tables(readout, config)
    well_manifest = pd.DataFrame(well_rows).sort_values("well", kind="mergesort")
    guard = evaluate_guard(readout, tables, well_manifest, config)

    expected_names = {
        "overall_metrics": "overall_metrics.csv",
        "fold_metrics": "fold_metrics.csv",
        "scope_metrics": "scope_metrics.csv",
        "orientation_metrics": "orientation_metrics.csv",
        "by_well_metrics": "by_well_metrics.csv",
        "donor_transfer_readout": "donor_transfer_readout.csv",
    }
    table_paths: dict[str, Path] = {}
    for name, suffix in expected_names.items():
        path = artifacts / f"{OUTPUT_PREFIX}_{suffix}"
        tables[name].to_csv(path, index=False)
        table_paths[name] = path

    raw_manifest = {
        "name": "raw_horizontal_score_safe_files",
        "path": str(raw_dir),
        "rows": int(well_manifest["horizontal_rows"].sum()),
        "wells": len(well_manifest),
        "raw_sha256": dataframe_content_sha(well_manifest, ["well", "horizontal_raw_sha256"]),
    }
    input_manifest = pd.DataFrame(
        [cache_manifest, identity_manifest, raw_manifest, hidden_manifest, *fixed_manifests]
    )
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv"
    input_manifest.to_csv(input_manifest_path, index=False)
    well_manifest_path = artifacts / f"{OUTPUT_PREFIX}_well_manifest.csv"
    well_manifest.to_csv(well_manifest_path, index=False)

    output_paths = {
        **table_paths,
        "edge_contract": edge_contract_path,
        "edge_schema": edge_schema_path,
        "input_manifest": input_manifest_path,
        "well_manifest": well_manifest_path,
    }
    output_sha = {name: sha256_path(path) for name, path in output_paths.items()}
    high_overall = (
        tables["overall_metrics"]
        .loc[tables["overall_metrics"]["selection"] == "high_confidence"]
        .iloc[0]
    )
    high_transfer = (
        tables["donor_transfer_readout"]
        .loc[
            (tables["donor_transfer_readout"]["scope"] == "overall")
            & (tables["donor_transfer_readout"]["selection"] == "high_confidence")
        ]
        .iloc[0]
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "train_side_readout_completed_guard_passed"
        if guard["passed"]
        else "train_side_readout_completed_guard_failed",
        "route": get_nested(config, "experiment.route"),
        "runtime_seconds": time.time() - started,
        "canonical_rows": int(get_nested(config, "validation.expected_rows")),
        "wells": expected_wells,
        "eligible_receiver_centers": int(well_manifest["eligible_receiver_centers"].sum()),
        "generated_edges": len(readout),
        "active_audit_variants": 1,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "high_confidence": {
            "edges": int(readout["high_confidence"].sum()),
            "within10": float(high_overall["real_within10"]),
            "within10_lift_vs_shuffled": float(high_overall["within10_lift_vs_shuffled"]),
            "donor_transfer_gain_ft": float(high_transfer["matched_gain_vs_receiver_baseline_ft"]),
        },
        "guard": guard,
        "truth_attachment": {
            "stage": "after_real_and_shuffled_edges_persisted_and_hashed",
            "frozen_edge_content_sha256": frozen_edge_sha,
            "attachment_before_freeze_count": 0,
        },
        "artifacts": {
            "target_free_edges": edge_artifact,
            "file_sha256": output_sha,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decision": "consider_separate_soft_correction_experiment"
        if guard["passed"]
        else "close_branch_without_parameter_rescue",
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "route": get_nested(config, "experiment.route"),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": get_nested(config, "validation.metric"),
        "diagnostic": {
            "high_confidence": summary["high_confidence"],
            "guard": guard,
            "frozen_edge_content_sha256": frozen_edge_sha,
        },
        "active_variants": 1,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_variants": 0,
        "pf_variants": 0,
        "inference": False,
        "submission": False,
        "notes": "No corrected prediction, model, inference output, or submission is produced.",
    }
    write_json(metrics_output_path(), metrics)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Setup and contract preview


# %%
CONFIG: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment.name"),
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "fixed_prediction_reference": get_nested(
                    CONFIG, "lineage.fixed_prediction_reference"
                ),
                "donor_scope": get_nested(CONFIG, "matching.donor_scope"),
                "receiver_scope": get_nested(CONFIG, "matching.receiver_scope"),
                "half_windows": [
                    get_nested(CONFIG, "matching.search.primary_half_window_rows"),
                    *get_nested(CONFIG, "matching.search.auxiliary_half_window_rows"),
                ],
                "stride": get_nested(CONFIG, "matching.search.center_stride_rows"),
                "active_variants": get_nested(CONFIG, "execution.active_variants"),
                "lightgbm_configs": get_nested(CONFIG, "execution.lightgbm_config_count"),
                "trained_folds": get_nested(CONFIG, "execution.trained_fold_count"),
                "boosters": get_nested(CONFIG, "execution.total_boosters"),
                "hmm_variants": get_nested(CONFIG, "execution.hmm_variants"),
                "pf_variants": get_nested(CONFIG, "execution.pf_variants"),
                "parent_control_retraining": get_nested(
                    CONFIG, "execution.control_or_parent_retraining"
                ),
                "gpu": get_nested(CONFIG, "execution.gpu"),
                "inference": get_nested(CONFIG, "execution.inference"),
                "submission": get_nested(CONFIG, "execution.submission"),
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
            },
            indent=2,
        )
    )


# %% [markdown]
# ## 10. Run diagnostic and report generated artifacts


# %%
if EXECUTE_NOTEBOOK:
    assert CONFIG is not None
    EXP282_SUMMARY = run_full_experiment(CONFIG)

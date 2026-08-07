# %% [markdown]
# # exp339 missing-gap pseudo-mask uncertainty readout train
#
# Stage 0 is a deterministic, zero-HMM diagnostic. For every outer fold it
# learns the natural known-prefix GR missing-run histogram on outer-train
# wells, creates matched pseudo-gaps on finite GR blocks, freezes interpolation
# predictions before restoring hidden GR, and evaluates a fixed hierarchical
# interpolation-error variance table on outer-valid wells.

# %% [markdown]
# ## Contents
# 1. Imports and execution guard
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific contract and dependency preflight
# 4. Truth-free known-prefix and natural missing-run helpers
# 5. Deterministic real and circular pseudo-gap placement
# 6. Interpolation prediction freeze and late hidden-GR attachment
# 7. Hierarchical uncertainty table and Stage 0 gate
# 8. Metrics and generated artifacts
# 9. Setup and configuration preview
# 10. Run the approved Stage 0 readout only

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp339_missing_gap_pseudomask_uncertainty_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
RUN_LENGTH_BINS = ((1, 3), (4, 7), (8, 15), (16, 31), (32, 64))
ANCHOR_DISTANCE_BINS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32))


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP339_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
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
    raise FileNotFoundError(f"exp339 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        path = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def train_data_dir(config: Mapping[str, Any]) -> Path:
    configured = Path(str(get_nested(config, "data.train_dir") or "data/raw/train"))
    root = project_root()
    candidates = (
        configured,
        root / configured,
        KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
    )
    for path in candidates:
        if path.exists() and any(path.glob("*__horizontal_well.csv")):
            return path
    if KAGGLE_INPUT_ROOT.exists():
        matches = sorted(KAGGLE_INPUT_ROOT.glob("**/train/*__horizontal_well.csv"))
        if matches:
            return matches[0].parent
    raise FileNotFoundError(
        "raw train directory not found; checked=" + str([str(path) for path in candidates])
    )


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_gzip_csv(path: str | Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            if chunk:
                last_byte = chunk[-1:]
    line_count = newline_count + int(bool(last_byte) and last_byte != b"\n")
    return {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": digest.hexdigest(),
        "content_sha256": digest.hexdigest(),
        "data_rows": max(0, line_count - 1),
    }


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(to_jsonable(dict(value)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_content_sha(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
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


def stable_uint64(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big", signed=False)


def write_gzip_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    return inspect_gzip_csv(path)


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        paths = (
            candidate if candidate.name == filename else candidate / filename,
            root / candidate if candidate.name == filename else root / candidate / filename,
            Path.cwd() / candidate
            if candidate.name == filename
            else Path.cwd() / candidate / filename,
        )
        for path in paths:
            checked.append(str(path))
            if path.exists() and path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
    }


# %% [markdown]
# ## 3. Frozen scientific contract and dependency preflight


# %%
def validate_scientific_contract(
    config: Mapping[str, Any], *, require_run_approval: bool = False
) -> None:
    expected = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation",
        "implementation.enabled": True,
        "implementation.scope": "stage0_only",
        "implementation.canonical_notebook_adopted": True,
        "model.pseudo_gap.run_length_clip": [1, 64],
        "model.pseudo_gap.run_length_bins": [list(value) for value in RUN_LENGTH_BINS],
        "model.pseudo_gap.nearest_anchor_distance_bins": [
            list(value) for value in ANCHOR_DISTANCE_BINS
        ],
        "model.pseudo_gap.maximum_gaps_per_well_length_bin": 4,
        "model.pseudo_gap.require_finite_anchor_both_sides": True,
        "model.pseudo_gap.forbid_row_reuse_within_fold": True,
        "model.pseudo_gap.selection": "stable_sha256_fold_well_length_start",
        "model.pseudo_gap.length_matching": (
            "outer_train_exact_length_histogram_within_fixed_bins"
        ),
        "model.pseudo_gap.length_schedule": (
            "stable_sha256_fold_well_length_bin_slot_cdf"
        ),
        "model.pseudo_gap.control_placement": (
            "matched_length_and_count_candidate_rank_circular_rotation"
        ),
        "model.pseudo_gap.coverage_unit": (
            "outer_valid_wells_with_at_least_one_selected_gap"
        ),
        "model.pseudo_gap.interpolation": "exp209_linear_interpolate_both_directions",
        "model.uncertainty_table.value": "mean_squared_interpolation_error",
        "model.uncertainty_table.shrinkage_support_k": 200,
        "model.uncertainty_table.hierarchy": [
            "length_distance_cell",
            "length_bin",
            "outer_train_global",
        ],
        "model.uncertainty_table.gaussian_nll": (
            "half_log_2pi_variance_plus_squared_error_over_2variance"
        ),
        "model.uncertainty_table.primary": "two_dimensional_shrunk_variance",
        "model.uncertainty_table.control": "outer_train_global_constant_variance",
        "model.uncertainty_table.negative_control": (
            "stable_circular_pseudogap_placement"
        ),
        "execution_contract.scientific_readouts": 1,
        "execution_contract.control_readouts": 2,
        "execution_contract.hmm_well_runs": 0,
        "execution_contract.model_configs": 0,
        "execution_contract.trained_folds": 0,
        "execution_contract.boosters": 0,
        "execution_contract.parent_control_retraining": False,
        "execution.implementation_approved": True,
        "execution.implementation_approval_scope": "user_message_implement_exp339",
        "execution.run_inference": False,
        "execution.create_submission": False,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp339 fixed contract mismatch: {key} must be {value!r}")
    if get_nested(config, "model.forbidden") != [
        "hmm_decode",
        "tvt_prediction",
        "suffix_tvt",
        "adaptive_bins",
        "interpolation_method_grid",
        "inference",
        "submission",
    ]:
        raise ValueError("exp339 forbidden-operation contract changed")
    expected_validation = {
        "strategy": "outer_grouped_known_prefix_pseudogap_uncertainty_readout",
        "metric": "gaussian_nll",
        "n_folds": 5,
        "expected_folds": [0, 1, 2, 3, 4],
        "expected_rows": 3783989,
        "expected_wells": 773,
        "group_column": "well_id",
        "score_rows": "outer_valid_known_prefix_pseudogap_rows_only",
        "truth_attachment": (
            "hidden_raw_gr_after_pseudogap_identity_and_interpolation_content_sha_freeze"
        ),
    }
    for key, value in expected_validation.items():
        if get_nested(config, f"validation.{key}") != value:
            raise ValueError(f"exp339 fixed validation contract mismatch: {key}")
    expected_gate = {
        "minimum_pseudogap_coverage_each_fold": 0.90,
        "minimum_wells_each_fold": 140,
        "minimum_distinct_wells_pooled": 700,
        "require_pooled_nll_better_than_global": True,
        "minimum_improved_folds_nll_vs_global": 4,
        "pooled_variance_calibration_ratio": [0.80, 1.25],
        "fold_variance_calibration_ratio": [0.70, 1.40],
        "minimum_folds_calibrated": 4,
        "minimum_pooled_spearman_length_vs_sigma": 0.50,
        "minimum_positive_spearman_folds": 4,
        "require_real_better_than_circular_pooled": True,
        "minimum_folds_real_better_than_circular": 4,
    }
    if get_nested(config, "model.pass_requires_all") != expected_gate:
        raise ValueError("exp339 fixed Stage 0 promotion gate contract changed")
    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.run_stage_0"))
    ):
        raise RuntimeError("exp339 Stage 0 Kaggle package/push/run is not approved")


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "active_stage": "stage0_known_prefix_pseudogap_only",
        "unknown_suffix_tvt_read": False,
        "pseudo_gap": get_nested(config, "model.pseudo_gap"),
        "uncertainty_table": get_nested(config, "model.uncertainty_table"),
        "pass_requires_all": get_nested(config, "model.pass_requires_all"),
        "validation": {
            "strategy": get_nested(config, "validation.strategy"),
            "n_folds": get_nested(config, "validation.n_folds"),
            "truth_attachment": get_nested(config, "validation.truth_attachment"),
            "leakage_policy": get_nested(config, "validation.leakage_policy"),
        },
        "execution_counts": get_nested(config, "execution_contract"),
        "forbidden": get_nested(config, "model.forbidden"),
        "seed_policy": get_nested(config, "reproducibility.seed_policy"),
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def validate_raw_well_identity(config: Mapping[str, Any], raw_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for horizontal_path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.replace("__horizontal_well.csv", "")
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.exists():
            raise FileNotFoundError(typewell_path)
        rows.append(
            {
                "well_id": well,
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    frame = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    content_sha = dataframe_content_sha(
        frame,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if len(frame) != expected_wells or content_sha != expected_sha:
        raise ValueError("current raw train well-file identity mismatch")
    return {
        "path": str(raw_dir),
        "wells": len(frame),
        "well_ids": frame["well_id"].tolist(),
        "content_sha256": content_sha,
    }


def load_fold_map(
    path: Path,
    *,
    expected_rows: int,
    expected_wells: int,
    expected_folds: list[int],
) -> dict[str, int]:
    parts: list[pd.DataFrame] = []
    row_count = 0
    for chunk in pd.read_csv(
        path,
        usecols=["well_id", "fold"],
        dtype={"well_id": str},
        chunksize=250_000,
    ):
        chunk["fold"] = pd.to_numeric(chunk["fold"], errors="raise").astype(np.int64)
        row_count += len(chunk)
        if (chunk.groupby("well_id", sort=False)["fold"].nunique() > 1).any():
            raise ValueError("fold assignment changes inside a well")
        parts.append(chunk.drop_duplicates(["well_id", "fold"]))
    pairs = pd.concat(parts, ignore_index=True).drop_duplicates(["well_id", "fold"])
    if (pairs.groupby("well_id", sort=False)["fold"].nunique() > 1).any():
        raise ValueError("fold assignment changes across chunks inside a well")
    frame = pairs.drop_duplicates("well_id").sort_values("well_id", kind="mergesort")
    if (
        row_count != expected_rows
        or len(frame) != expected_wells
        or sorted(frame["fold"].unique().tolist()) != expected_folds
    ):
        raise ValueError("fold assignment row/well/fold coverage mismatch")
    return dict(zip(frame["well_id"], frame["fold"].astype(int), strict=True))


def preflight_dependencies(
    config: Mapping[str, Any], raw_dir: Path
) -> tuple[dict[str, Any], dict[str, int]]:
    raw_report = validate_raw_well_identity(config, raw_dir)
    fold_spec = get_nested(config, "data.fold_assignment") or {}
    fold_path = resolve_existing(
        str(fold_spec["filename"]),
        [str(value) for value in fold_spec.get("candidates", [])],
    )
    fold_report = inspect_gzip_csv(fold_path)
    if fold_report["decompressed_sha256"] != str(fold_spec["expected_decompressed_sha256"]):
        raise ValueError("exp226 fold assignment decompressed SHA mismatch")
    fold_map = load_fold_map(
        fold_path,
        expected_rows=int(get_nested(config, "validation.expected_rows")),
        expected_wells=int(get_nested(config, "validation.expected_wells")),
        expected_folds=[int(value) for value in get_nested(config, "validation.expected_folds")],
    )
    if sorted(fold_map) != sorted(raw_report["well_ids"]):
        raise ValueError("raw train and fold assignment well identity mismatch")
    return (
        {
            "raw_train": raw_report,
            "fold_assignment": {
                **fold_report,
                "source": fold_spec["source"],
                "well_ids": sorted(fold_map),
            },
        },
        fold_map,
    )


# %% [markdown]
# ## 4. Truth-free known-prefix and natural missing-run helpers


# %%
def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / f"{well}__horizontal_well.csv",
        usecols=["GR", "TVT_input"],
    )
    if list(frame.columns) != ["GR", "TVT_input"]:
        frame = frame[["GR", "TVT_input"]]
    return frame.reset_index(drop=True)


def build_known_prefix_profile(well: str, horizontal: pd.DataFrame) -> dict[str, Any]:
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    if np.isinf(tvt_input).any():
        raise ValueError(f"{well}: TVT_input contains infinity")
    known_index = np.flatnonzero(np.isfinite(tvt_input))
    if len(known_index) < 3 or not np.array_equal(known_index, np.arange(known_index[-1] + 1)):
        raise ValueError(f"{well}: TVT_input must be one contiguous known prefix")
    prefix_rows = int(known_index[-1] + 1)
    gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)[:prefix_rows]
    if np.isinf(gr).any():
        raise ValueError(f"{well}: GR contains infinity")
    return {
        "well_id": str(well),
        "prefix_rows": prefix_rows,
        "gr": gr,
        "finite_gr": np.isfinite(gr),
    }


def load_known_prefix_profiles(
    wells: Iterable[str], raw_dir: Path
) -> dict[str, dict[str, Any]]:
    return {
        str(well): build_known_prefix_profile(
            str(well), load_horizontal_without_truth(str(well), raw_dir)
        )
        for well in sorted(str(value) for value in wells)
    }


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.astype(int), stops.astype(int), strict=True))


def range_label(prefix: str, bounds: tuple[int, int]) -> str:
    return f"{prefix}{bounds[0]:02d}_{bounds[1]:02d}"


def value_bin_label(value: int, bins: tuple[tuple[int, int], ...], prefix: str) -> str:
    for bounds in bins:
        if bounds[0] <= int(value) <= bounds[1]:
            return range_label(prefix, bounds)
    raise ValueError(f"value {value} is outside frozen {prefix} bins")


def natural_missing_inventory(
    profiles: Mapping[str, Mapping[str, Any]],
    outer_train_wells: Iterable[str],
    outer_fold: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well in sorted(str(value) for value in outer_train_wells):
        finite = np.asarray(profiles[well]["finite_gr"], dtype=bool)
        for start, stop in contiguous_true_runs(~finite):
            raw_length = int(stop - start)
            clipped_length = int(np.clip(raw_length, 1, 64))
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "well_id": well,
                    "start_row": int(start),
                    "stop_row_exclusive": int(stop),
                    "raw_run_length": raw_length,
                    "clipped_run_length": clipped_length,
                    "length_bin": value_bin_label(
                        clipped_length, RUN_LENGTH_BINS, "L"
                    ),
                }
            )
    columns = [
        "outer_fold",
        "well_id",
        "start_row",
        "stop_row_exclusive",
        "raw_run_length",
        "clipped_run_length",
        "length_bin",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["well_id", "start_row"], kind="mergesort"
    ).reset_index(drop=True)


def build_natural_missing_histogram(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        raise ValueError("outer-train known prefixes contain no natural GR missing runs")
    histogram = (
        inventory.groupby(
            ["outer_fold", "length_bin", "clipped_run_length"],
            sort=True,
            observed=True,
        )
        .agg(natural_run_count=("well_id", "size"), natural_wells=("well_id", "nunique"))
        .reset_index()
    )
    histogram["natural_run_share_within_bin"] = histogram["natural_run_count"] / histogram.groupby(
        ["outer_fold", "length_bin"], sort=False
    )["natural_run_count"].transform("sum")
    return histogram.sort_values(
        ["outer_fold", "length_bin", "clipped_run_length"], kind="mergesort"
    ).reset_index(drop=True)


# %% [markdown]
# ## 5. Deterministic real and circular pseudo-gap placement


# %%
def target_length_schedule(
    histogram: pd.DataFrame,
    *,
    outer_fold: int,
    well_id: str,
    length_bin: str,
    maximum_slots: int,
) -> list[int]:
    subset = histogram.loc[
        histogram["outer_fold"].eq(outer_fold)
        & histogram["length_bin"].eq(length_bin)
    ].sort_values("clipped_run_length", kind="mergesort")
    if subset.empty:
        return []
    lengths = subset["clipped_run_length"].to_numpy(np.int64)
    counts = subset["natural_run_count"].to_numpy(np.float64)
    cumulative = np.cumsum(counts) / counts.sum()
    schedule: list[int] = []
    for slot in range(int(maximum_slots)):
        key = f"histogram_slot|{outer_fold}|{well_id}|{length_bin}|{slot}"
        quantile = (stable_uint64(key) + 0.5) / float(2**64)
        index = min(int(np.searchsorted(cumulative, quantile, side="right")), len(lengths) - 1)
        schedule.append(int(lengths[index]))
    return schedule


def eligible_gap_starts(finite_gr: np.ndarray, gap_length: int) -> np.ndarray:
    finite = np.asarray(finite_gr, dtype=bool)
    length = int(gap_length)
    if length < 1 or len(finite) < length + 2:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(1, len(finite) - length, dtype=np.int64)
    missing_prefix = np.r_[0, np.cumsum((~finite).astype(np.int64))]
    block_missing = missing_prefix[starts + length] - missing_prefix[starts]
    valid = (
        (block_missing == 0)
        & finite[starts - 1]
        & finite[starts + length]
    )
    return starts[valid]


def _placement_masks(
    gaps: list[dict[str, Any]], prefix_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.zeros(prefix_rows, dtype=bool)
    protected_anchor = np.zeros(prefix_rows, dtype=bool)
    for gap in gaps:
        start = int(gap["start_row"])
        stop = start + int(gap["gap_length"])
        hidden[start:stop] = True
        protected_anchor[start - 1] = True
        protected_anchor[stop] = True
    return hidden, protected_anchor


def _can_place(
    start: int,
    length: int,
    used_hidden: np.ndarray,
    protected_anchor: np.ndarray,
) -> bool:
    stop = int(start + length)
    return bool(
        not used_hidden[start:stop].any()
        and not protected_anchor[start:stop].any()
        and not used_hidden[start - 1]
        and not used_hidden[stop]
    )


def select_real_gap_plan(
    profile: Mapping[str, Any],
    histogram: pd.DataFrame,
    *,
    outer_fold: int,
    role: str,
    maximum_slots: int,
) -> list[dict[str, Any]]:
    well = str(profile["well_id"])
    finite = np.asarray(profile["finite_gr"], dtype=bool)
    used_hidden = np.zeros(len(finite), dtype=bool)
    protected_anchor = np.zeros(len(finite), dtype=bool)
    selected: list[dict[str, Any]] = []
    for bounds in RUN_LENGTH_BINS:
        length_bin = range_label("L", bounds)
        schedule = target_length_schedule(
            histogram,
            outer_fold=outer_fold,
            well_id=well,
            length_bin=length_bin,
            maximum_slots=maximum_slots,
        )
        for slot, gap_length in enumerate(schedule):
            starts = eligible_gap_starts(finite, gap_length)
            ordered = sorted(
                (int(value) for value in starts),
                key=lambda start: (
                    stable_uint64(f"{outer_fold}|{well}|{length_bin}|{start}"),
                    start,
                ),
            )
            chosen = next(
                (
                    start
                    for start in ordered
                    if _can_place(
                        start,
                        gap_length,
                        used_hidden,
                        protected_anchor,
                    )
                ),
                None,
            )
            if chosen is None:
                continue
            stop = chosen + gap_length
            used_hidden[chosen:stop] = True
            protected_anchor[chosen - 1] = True
            protected_anchor[stop] = True
            selected.append(
                {
                    "outer_fold": int(outer_fold),
                    "role": str(role),
                    "placement": "real",
                    "well_id": well,
                    "length_bin": length_bin,
                    "slot": int(slot),
                    "gap_id": f"{outer_fold}|{role}|{well}|{length_bin}|{slot}",
                    "gap_length": int(gap_length),
                    "start_row": int(chosen),
                    "stop_row_exclusive": int(stop),
                    "selection_sha256": hashlib.sha256(
                        f"{outer_fold}|{well}|{length_bin}|{chosen}".encode()
                    ).hexdigest(),
                    "control_unchanged": False,
                }
            )
    return selected


def build_circular_control_plan(
    real_plan: list[dict[str, Any]],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not real_plan:
        return []
    finite = np.asarray(profile["finite_gr"], dtype=bool)
    control = [{**gap, "placement": "circular"} for gap in real_plan]
    for index, real_gap in enumerate(real_plan):
        other = [gap for other_index, gap in enumerate(control) if other_index != index]
        used_hidden, protected_anchor = _placement_masks(other, len(finite))
        length = int(real_gap["gap_length"])
        starts = eligible_gap_starts(finite, length).astype(int).tolist()
        if not starts:
            raise RuntimeError("real pseudo-gap has no eligible circular-control placement")
        real_start = int(real_gap["start_row"])
        real_position = starts.index(real_start)
        if len(starts) > 1:
            key = (
                f"circular|{real_gap['outer_fold']}|{real_gap['well_id']}|"
                f"{real_gap['length_bin']}|{real_gap['slot']}"
            )
            offset = 1 + stable_uint64(key) % (len(starts) - 1)
        else:
            offset = 0
        candidate_order = [
            starts[(real_position + offset + shift) % len(starts)]
            for shift in range(len(starts))
        ]
        chosen = next(
            (
                start
                for start in candidate_order
                if _can_place(start, length, used_hidden, protected_anchor)
            ),
            None,
        )
        if chosen is None:
            raise RuntimeError("matched circular pseudo-gap placement could not preserve count")
        control[index] = {
            **control[index],
            "start_row": int(chosen),
            "stop_row_exclusive": int(chosen + length),
            "selection_sha256": hashlib.sha256(
                (
                    f"circular|{real_gap['outer_fold']}|{real_gap['well_id']}|"
                    f"{real_gap['length_bin']}|{chosen}"
                ).encode()
            ).hexdigest(),
            "control_unchanged": bool(chosen == real_start),
            "control_candidate_offset": int(offset),
        }
    validate_gap_plan(control, profile)
    return control


def validate_gap_plan(plan: list[dict[str, Any]], profile: Mapping[str, Any]) -> None:
    finite = np.asarray(profile["finite_gr"], dtype=bool)
    hidden = np.zeros(len(finite), dtype=bool)
    protected_anchor = np.zeros(len(finite), dtype=bool)
    per_bin: dict[str, int] = {}
    for gap in sorted(plan, key=lambda value: (str(value["length_bin"]), int(value["slot"]))):
        start = int(gap["start_row"])
        length = int(gap["gap_length"])
        stop = start + length
        if start <= 0 or stop >= len(finite):
            raise ValueError("pseudo-gap is not interior")
        if not finite[start - 1] or not finite[stop] or not finite[start:stop].all():
            raise ValueError("pseudo-gap requires finite hidden rows and both finite anchors")
        if not _can_place(start, length, hidden, protected_anchor):
            raise ValueError("pseudo-gap rows or anchors overlap another selected gap")
        hidden[start:stop] = True
        protected_anchor[start - 1] = True
        protected_anchor[stop] = True
        per_bin[str(gap["length_bin"])] = per_bin.get(str(gap["length_bin"]), 0) + 1
    if any(count > 4 for count in per_bin.values()):
        raise ValueError("pseudo-gap plan exceeds four gaps per well and length bin")


def validate_matched_plans(
    real_plan: list[dict[str, Any]], circular_plan: list[dict[str, Any]]
) -> None:
    columns = ("gap_id", "length_bin", "slot", "gap_length")
    real = sorted(tuple(gap[column] for column in columns) for gap in real_plan)
    circular = sorted(tuple(gap[column] for column in columns) for gap in circular_plan)
    if real != circular:
        raise ValueError("circular control did not preserve pseudo-gap identity, count, and length")


# %% [markdown]
# ## 6. Interpolation prediction freeze and late hidden-GR attachment


# %%
def build_interpolation_predictions(
    profile: Mapping[str, Any], plan: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_columns = [
        "outer_fold",
        "role",
        "placement",
        "well_id",
        "gap_id",
        "length_bin",
        "slot",
        "gap_length",
        "start_row",
        "row_idx",
        "nearest_anchor_distance",
        "distance_bin",
        "interpolated_gr",
    ]
    truth_columns = [
        "outer_fold",
        "role",
        "placement",
        "well_id",
        "gap_id",
        "row_idx",
        "hidden_raw_gr",
    ]
    if not plan:
        return pd.DataFrame(columns=prediction_columns), pd.DataFrame(columns=truth_columns)
    validate_gap_plan(plan, profile)
    raw_gr = np.asarray(profile["gr"], dtype=np.float64)
    masked = raw_gr.copy()
    for gap in plan:
        start = int(gap["start_row"])
        masked[start : start + int(gap["gap_length"])] = np.nan
    interpolated = (
        pd.Series(masked, dtype="float64")
        .interpolate(limit_direction="both")
        .to_numpy(np.float64)
    )
    prediction_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for gap in plan:
        start = int(gap["start_row"])
        length = int(gap["gap_length"])
        for offset in range(length):
            row_idx = start + offset
            distance = min(offset + 1, length - offset)
            if not np.isfinite(interpolated[row_idx]) or not np.isfinite(raw_gr[row_idx]):
                raise ValueError("pseudo-gap interpolation or hidden raw GR is non-finite")
            common = {
                "outer_fold": int(gap["outer_fold"]),
                "role": str(gap["role"]),
                "placement": str(gap["placement"]),
                "well_id": str(gap["well_id"]),
                "gap_id": str(gap["gap_id"]),
                "row_idx": int(row_idx),
            }
            prediction_rows.append(
                {
                    **common,
                    "length_bin": str(gap["length_bin"]),
                    "slot": int(gap["slot"]),
                    "gap_length": length,
                    "start_row": start,
                    "nearest_anchor_distance": int(distance),
                    "distance_bin": value_bin_label(
                        int(distance), ANCHOR_DISTANCE_BINS, "D"
                    ),
                    "interpolated_gr": float(interpolated[row_idx]),
                }
            )
            truth_rows.append({**common, "hidden_raw_gr": float(raw_gr[row_idx])})
    predictions = pd.DataFrame(prediction_rows, columns=prediction_columns).sort_values(
        ["outer_fold", "role", "placement", "well_id", "row_idx", "gap_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    truth = pd.DataFrame(truth_rows, columns=truth_columns).sort_values(
        ["outer_fold", "role", "placement", "well_id", "row_idx", "gap_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    forbidden = {"hidden_raw_gr", "interpolation_error", "squared_error", "TVT", "error"}
    if forbidden.intersection(predictions.columns):
        raise RuntimeError("hidden GR or target columns entered the prediction freeze frame")
    return predictions, truth


def attach_hidden_gr_after_prediction_freeze(
    predictions: pd.DataFrame,
    hidden_truth: pd.DataFrame,
    *,
    frozen_prediction_sha256: str,
) -> pd.DataFrame:
    if (
        not frozen_prediction_sha256
        or frozen_prediction_sha256 != dataframe_content_sha(predictions)
    ):
        raise ValueError("pseudo-gap prediction content must be frozen before hidden GR attachment")
    keys = ["outer_fold", "role", "placement", "well_id", "gap_id", "row_idx"]
    if predictions.duplicated(keys).any() or hidden_truth.duplicated(keys).any():
        raise ValueError("pseudo-gap row identity must be unique before late hidden GR join")
    output = predictions.merge(hidden_truth, on=keys, how="left", validate="one_to_one")
    if output["hidden_raw_gr"].isna().any():
        raise ValueError("late hidden GR join did not cover every pseudo-gap row")
    output["interpolation_error"] = (
        output["hidden_raw_gr"].to_numpy(np.float64)
        - output["interpolated_gr"].to_numpy(np.float64)
    )
    output["squared_error"] = output["interpolation_error"] ** 2
    return output


# %% [markdown]
# ## 7. Hierarchical uncertainty table and Stage 0 gate


# %%
def fit_uncertainty_table(
    audit_rows: pd.DataFrame,
    *,
    outer_fold: int,
    placement: str,
    support_k: int,
) -> pd.DataFrame:
    fit_rows = audit_rows.loc[
        audit_rows["outer_fold"].eq(outer_fold)
        & audit_rows["role"].eq("outer_train")
        & audit_rows["placement"].eq(placement)
    ].copy()
    if fit_rows.empty:
        raise ValueError("uncertainty table requires outer-train pseudo-gap errors")
    squared_error = fit_rows["squared_error"].to_numpy(np.float64)
    if not np.isfinite(squared_error).all():
        raise ValueError("outer-train squared interpolation errors must be finite")
    global_variance = float(np.mean(squared_error))
    if global_variance <= 0.0:
        raise ValueError("outer-train global interpolation variance must be positive")
    length_stats = fit_rows.groupby("length_bin", sort=True, observed=True)["squared_error"].agg(
        ["size", "mean"]
    )
    cell_stats = fit_rows.groupby(
        ["length_bin", "distance_bin"], sort=True, observed=True
    )["squared_error"].agg(["size", "mean"])
    rows: list[dict[str, Any]] = []
    for length_bounds in RUN_LENGTH_BINS:
        length_bin = range_label("L", length_bounds)
        if length_bin in length_stats.index:
            length_count = int(length_stats.loc[length_bin, "size"])
            length_mse = float(length_stats.loc[length_bin, "mean"])
        else:
            length_count = 0
            length_mse = global_variance
        length_weight = length_count / float(length_count + support_k)
        length_variance = (
            length_weight * length_mse + (1.0 - length_weight) * global_variance
        )
        for distance_bounds in ANCHOR_DISTANCE_BINS:
            distance_bin = range_label("D", distance_bounds)
            key = (length_bin, distance_bin)
            if key in cell_stats.index:
                cell_count = int(cell_stats.loc[key, "size"])
                cell_mse = float(cell_stats.loc[key, "mean"])
            else:
                cell_count = 0
                cell_mse = length_variance
            cell_weight = cell_count / float(cell_count + support_k)
            predicted_variance = (
                cell_weight * cell_mse + (1.0 - cell_weight) * length_variance
            )
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "placement": str(placement),
                    "length_bin": length_bin,
                    "distance_bin": distance_bin,
                    "global_count": int(len(fit_rows)),
                    "global_variance": global_variance,
                    "length_count": length_count,
                    "length_raw_mse": length_mse,
                    "length_shrinkage_weight": length_weight,
                    "length_variance": length_variance,
                    "cell_count": cell_count,
                    "cell_raw_mse": cell_mse,
                    "cell_shrinkage_weight": cell_weight,
                    "predicted_variance": predicted_variance,
                }
            )
    table = pd.DataFrame(rows)
    if len(table) != len(RUN_LENGTH_BINS) * len(ANCHOR_DISTANCE_BINS):
        raise AssertionError("uncertainty table must contain every fixed 5x6 cell")
    if not (table["predicted_variance"] > 0.0).all():
        raise ValueError("uncertainty table produced non-positive variance")
    return table


def apply_uncertainty_table(audit_rows: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    keys = ["outer_fold", "placement", "length_bin", "distance_bin"]
    selected = table[
        keys + ["global_variance", "length_variance", "predicted_variance"]
    ]
    output = audit_rows.merge(selected, on=keys, how="left", validate="many_to_one")
    required = ["global_variance", "length_variance", "predicted_variance"]
    if output[required].isna().any().any() or not (
        output[required].to_numpy(np.float64) > 0.0
    ).all():
        raise ValueError("uncertainty table failed to cover pseudo-gap audit rows")
    output["primary_nll"] = gaussian_nll_from_variance(
        output["interpolation_error"].to_numpy(np.float64),
        output["predicted_variance"].to_numpy(np.float64),
    )
    output["global_nll"] = gaussian_nll_from_variance(
        output["interpolation_error"].to_numpy(np.float64),
        output["global_variance"].to_numpy(np.float64),
    )
    return output


def gaussian_nll_from_variance(error: np.ndarray, variance: np.ndarray) -> np.ndarray:
    residual = np.asarray(error, dtype=np.float64)
    var = np.asarray(variance, dtype=np.float64)
    if not np.isfinite(residual).all() or not np.isfinite(var).all() or not (var > 0).all():
        raise ValueError("Gaussian NLL requires finite errors and positive finite variance")
    return 0.5 * (np.log(2.0 * np.pi * var) + residual * residual / var)


def spearman_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(np.asarray(left, dtype=np.float64)).rank(method="average").to_numpy()
    right_rank = pd.Series(np.asarray(right, dtype=np.float64)).rank(method="average").to_numpy()
    if len(left_rank) < 2 or np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def summarize_stage0_fold(
    scored_rows: pd.DataFrame,
    plans: pd.DataFrame,
    fold_map: Mapping[str, int],
    outer_fold: int,
) -> dict[str, Any]:
    valid_wells = sorted(well for well, fold in fold_map.items() if int(fold) == outer_fold)
    real_plan = plans.loc[
        plans["outer_fold"].eq(outer_fold)
        & plans["role"].eq("outer_valid")
        & plans["placement"].eq("real")
    ]
    real = scored_rows.loc[
        scored_rows["outer_fold"].eq(outer_fold)
        & scored_rows["role"].eq("outer_valid")
        & scored_rows["placement"].eq("real")
    ]
    circular = scored_rows.loc[
        scored_rows["outer_fold"].eq(outer_fold)
        & scored_rows["role"].eq("outer_valid")
        & scored_rows["placement"].eq("circular")
    ]
    if real.empty or circular.empty or len(real) != len(circular):
        raise ValueError("real and circular outer-valid audit rows must be non-empty and matched")
    selected_wells = int(real_plan["well_id"].nunique())
    calibration_ratio = float(real["predicted_variance"].mean() / real["squared_error"].mean())
    spearman = spearman_rank_correlation(
        real["gap_length"].to_numpy(np.float64),
        np.sqrt(real["predicted_variance"].to_numpy(np.float64)),
    )
    return {
        "outer_fold": int(outer_fold),
        "valid_wells": len(valid_wells),
        "selected_valid_wells": selected_wells,
        "well_coverage": selected_wells / float(len(valid_wells)),
        "real_valid_gaps": int(real_plan["gap_id"].nunique()),
        "real_valid_rows": len(real),
        "circular_valid_rows": len(circular),
        "primary_nll_mean": float(real["primary_nll"].mean()),
        "global_nll_mean": float(real["global_nll"].mean()),
        "primary_better_than_global": bool(real["primary_nll"].mean() < real["global_nll"].mean()),
        "circular_nll_mean": float(circular["primary_nll"].mean()),
        "real_better_than_circular": bool(
            real["primary_nll"].mean() < circular["primary_nll"].mean()
        ),
        "variance_calibration_ratio": calibration_ratio,
        "gap_length_sigma_spearman": spearman,
    }


def evaluate_stage0_gate(
    scored_rows: pd.DataFrame,
    plans: pd.DataFrame,
    fold_summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = get_nested(config, "model.pass_requires_all") or {}
    real = scored_rows.loc[
        scored_rows["role"].eq("outer_valid") & scored_rows["placement"].eq("real")
    ]
    circular = scored_rows.loc[
        scored_rows["role"].eq("outer_valid")
        & scored_rows["placement"].eq("circular")
    ]
    pooled_primary_nll = float(real["primary_nll"].mean())
    pooled_global_nll = float(real["global_nll"].mean())
    pooled_circular_nll = float(circular["primary_nll"].mean())
    pooled_ratio = float(real["predicted_variance"].mean() / real["squared_error"].mean())
    pooled_spearman = spearman_rank_correlation(
        real["gap_length"].to_numpy(np.float64),
        np.sqrt(real["predicted_variance"].to_numpy(np.float64)),
    )
    pooled_distinct_wells = int(
        plans.loc[
            plans["role"].eq("outer_valid") & plans["placement"].eq("real"), "well_id"
        ].nunique()
    )
    fold_ratio_low, fold_ratio_high = [
        float(value) for value in gates["fold_variance_calibration_ratio"]
    ]
    pooled_ratio_low, pooled_ratio_high = [
        float(value) for value in gates["pooled_variance_calibration_ratio"]
    ]
    checks = {
        "coverage_each_fold": bool(
            (
                fold_summary["well_coverage"]
                >= float(gates["minimum_pseudogap_coverage_each_fold"])
            ).all()
        ),
        "minimum_wells_each_fold": bool(
            (fold_summary["selected_valid_wells"] >= int(gates["minimum_wells_each_fold"])).all()
        ),
        "minimum_distinct_wells_pooled": bool(
            pooled_distinct_wells >= int(gates["minimum_distinct_wells_pooled"])
        ),
        "pooled_nll_better_than_global": bool(pooled_primary_nll < pooled_global_nll),
        "fold_nll_better_than_global": bool(
            int(fold_summary["primary_better_than_global"].sum())
            >= int(gates["minimum_improved_folds_nll_vs_global"])
        ),
        "pooled_variance_calibrated": bool(
            pooled_ratio_low <= pooled_ratio <= pooled_ratio_high
        ),
        "fold_variance_calibrated": bool(
            int(
                fold_summary["variance_calibration_ratio"].between(
                    fold_ratio_low, fold_ratio_high, inclusive="both"
                ).sum()
            )
            >= int(gates["minimum_folds_calibrated"])
        ),
        "pooled_length_sigma_spearman": bool(
            pooled_spearman >= float(gates["minimum_pooled_spearman_length_vs_sigma"])
        ),
        "positive_length_sigma_spearman_folds": bool(
            int((fold_summary["gap_length_sigma_spearman"] > 0.0).sum())
            >= int(gates["minimum_positive_spearman_folds"])
        ),
        "real_better_than_circular_pooled": bool(pooled_primary_nll < pooled_circular_nll),
        "real_better_than_circular_folds": bool(
            int(fold_summary["real_better_than_circular"].sum())
            >= int(gates["minimum_folds_real_better_than_circular"])
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "pooled": {
            "primary_nll_mean": pooled_primary_nll,
            "global_nll_mean": pooled_global_nll,
            "circular_nll_mean": pooled_circular_nll,
            "variance_calibration_ratio": pooled_ratio,
            "gap_length_sigma_spearman": pooled_spearman,
            "distinct_valid_wells": pooled_distinct_wells,
            "valid_rows": len(real),
        },
        "folds_primary_better_than_global": int(
            fold_summary["primary_better_than_global"].sum()
        ),
        "folds_variance_calibrated": int(
            fold_summary["variance_calibration_ratio"].between(
                fold_ratio_low, fold_ratio_high, inclusive="both"
            ).sum()
        ),
        "folds_positive_length_sigma_spearman": int(
            (fold_summary["gap_length_sigma_spearman"] > 0.0).sum()
        ),
        "folds_real_better_than_circular": int(
            fold_summary["real_better_than_circular"].sum()
        ),
        "exp341_enabled": False,
        "exp341_requires_all_gates_and_separate_approval": True,
    }


# %% [markdown]
# ## 8. Metrics and generated artifacts


# %%
def run_stage0_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=True)
    started = datetime.now(UTC)
    raw_dir = train_data_dir(config)
    preflight, fold_map = preflight_dependencies(config, raw_dir)
    profiles = load_known_prefix_profiles(fold_map, raw_dir)
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    maximum_slots = int(
        get_nested(config, "model.pseudo_gap.maximum_gaps_per_well_length_bin")
    )
    support_k = int(get_nested(config, "model.uncertainty_table.shrinkage_support_k"))
    inventory_parts: list[pd.DataFrame] = []
    histogram_parts: list[pd.DataFrame] = []
    plan_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    table_parts: list[pd.DataFrame] = []
    prediction_freezes: list[dict[str, Any]] = []
    table_freezes: list[dict[str, Any]] = []
    for outer_fold in expected_folds:
        outer_train_wells = sorted(
            well for well, fold in fold_map.items() if int(fold) != outer_fold
        )
        outer_valid_wells = sorted(
            well for well, fold in fold_map.items() if int(fold) == outer_fold
        )
        inventory = natural_missing_inventory(profiles, outer_train_wells, outer_fold)
        histogram = build_natural_missing_histogram(inventory)
        inventory_parts.append(inventory)
        histogram_parts.append(histogram)
        fold_plans: list[dict[str, Any]] = []
        fold_predictions: list[pd.DataFrame] = []
        fold_truth: list[pd.DataFrame] = []
        for role, wells in (
            ("outer_train", outer_train_wells),
            ("outer_valid", outer_valid_wells),
        ):
            for well in wells:
                real_plan = select_real_gap_plan(
                    profiles[well],
                    histogram,
                    outer_fold=outer_fold,
                    role=role,
                    maximum_slots=maximum_slots,
                )
                validate_gap_plan(real_plan, profiles[well])
                circular_plan = build_circular_control_plan(real_plan, profiles[well])
                validate_matched_plans(real_plan, circular_plan)
                fold_plans.extend(real_plan)
                fold_plans.extend(circular_plan)
                for plan in (real_plan, circular_plan):
                    prediction, truth = build_interpolation_predictions(profiles[well], plan)
                    fold_predictions.append(prediction)
                    fold_truth.append(truth)
        plan_frame = pd.DataFrame(fold_plans).sort_values(
            ["outer_fold", "role", "placement", "well_id", "length_bin", "slot"],
            kind="mergesort",
        ).reset_index(drop=True)
        predictions = pd.concat(fold_predictions, ignore_index=True).sort_values(
            ["outer_fold", "role", "placement", "well_id", "row_idx", "gap_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        hidden_truth = pd.concat(fold_truth, ignore_index=True).sort_values(
            ["outer_fold", "role", "placement", "well_id", "row_idx", "gap_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        prediction_sha = dataframe_content_sha(predictions)
        prediction_freezes.append(
            {
                "outer_fold": outer_fold,
                "rows": len(predictions),
                "content_sha256": prediction_sha,
                "hidden_gr_columns_present": False,
            }
        )
        audit = attach_hidden_gr_after_prediction_freeze(
            predictions,
            hidden_truth,
            frozen_prediction_sha256=prediction_sha,
        )
        fold_tables = []
        for placement in ("real", "circular"):
            fold_tables.append(
                fit_uncertainty_table(
                    audit,
                    outer_fold=outer_fold,
                    placement=placement,
                    support_k=support_k,
                )
            )
        table = pd.concat(fold_tables, ignore_index=True)
        for placement in ("real", "circular"):
            placement_table = table.loc[table["placement"].eq(placement)].sort_values(
                ["length_bin", "distance_bin"], kind="mergesort"
            ).reset_index(drop=True)
            table_freezes.append(
                {
                    "outer_fold": outer_fold,
                    "placement": placement,
                    "rows": len(placement_table),
                    "content_sha256": dataframe_content_sha(placement_table),
                }
            )
        scored = apply_uncertainty_table(audit, table)
        plan_parts.append(plan_frame)
        prediction_parts.append(predictions)
        audit_parts.append(scored)
        table_parts.append(table)
    inventory_all = pd.concat(inventory_parts, ignore_index=True)
    histogram_all = pd.concat(histogram_parts, ignore_index=True)
    plans_all = pd.concat(plan_parts, ignore_index=True)
    predictions_all = pd.concat(prediction_parts, ignore_index=True)
    scored_all = pd.concat(audit_parts, ignore_index=True)
    tables_all = pd.concat(table_parts, ignore_index=True).sort_values(
        ["outer_fold", "placement", "length_bin", "distance_bin"], kind="mergesort"
    ).reset_index(drop=True)
    fold_summary = pd.DataFrame(
        [
            summarize_stage0_fold(scored_all, plans_all, fold_map, outer_fold)
            for outer_fold in expected_folds
        ]
    )
    promotion_gate = evaluate_stage0_gate(
        scored_all, plans_all, fold_summary, config
    )
    contract = build_scientific_contract(config)
    output_dir = artifact_dir()
    fold_manifest = pd.DataFrame(
        sorted(fold_map.items()), columns=["well_id", "fold"]
    )
    artifact_frames = {
        "fold_manifest": fold_manifest,
        "natural_missing_inventory": inventory_all,
        "natural_missing_histogram": histogram_all,
        "pseudogap_plan": plans_all,
        "interpolation_predictions": predictions_all,
        "uncertainty_table": tables_all,
        "audit_rows": scored_all,
        "fold_summary": fold_summary,
    }
    artifact_reports: dict[str, Any] = {}
    for name, frame in artifact_frames.items():
        artifact_reports[name] = write_gzip_csv(
            output_dir / f"{OUTPUT_PREFIX}_{name}.csv.gz", frame
        )
        artifact_reports[name]["schema_sha256"] = mapping_sha256(
            {"columns": list(frame.columns), "dtypes": frame.dtypes.astype(str).to_dict()}
        )
    write_json(output_dir / f"{OUTPUT_PREFIX}_scientific_contract.json", contract)
    contract_report = {
        "path": str(output_dir / f"{OUTPUT_PREFIX}_scientific_contract.json"),
        "raw_sha256": sha256_path(output_dir / f"{OUTPUT_PREFIX}_scientific_contract.json"),
        "content_sha256": contract["scientific_contract_sha256"],
    }
    completed = datetime.now(UTC)
    status = (
        "stage0_all_gates_passed_exp341_still_requires_separate_approval"
        if promotion_gate["passed"]
        else "stage0_gate_failed_exp341_blocked"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "runtime_seconds": (completed - started).total_seconds(),
        "runtime_versions": runtime_versions(),
        "execution_counts": get_nested(config, "execution_contract"),
        "preflight": preflight,
        "scientific_contract": contract_report,
        "prediction_freezes_before_hidden_gr_join": prediction_freezes,
        "fold_table_freezes": table_freezes,
        "artifacts": artifact_reports,
        "promotion_gate": promotion_gate,
        "table_content_sha256": artifact_reports["uncertainty_table"][
            "decompressed_sha256"
        ],
        "exp341_enabled": False,
        "exp341_requires_separate_approval_even_after_pass": True,
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    summary["summary_raw_sha256"] = sha256_path(summary_path)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "stage": "stage0_known_prefix_pseudogap",
        "cv": {
            "metric": "gaussian_nll",
            **promotion_gate["pooled"],
            "gate_passed": promotion_gate["passed"],
        },
        "public_lb": None,
        "private_lb": None,
        "artifacts_generated": True,
        "summary_path": str(summary_path),
        "summary_raw_sha256": summary["summary_raw_sha256"],
    }
    write_json(metrics_output_path(), metrics)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Setup and configuration preview


# %%
CONFIG = load_experiment_config()
validate_scientific_contract(CONFIG)
SCIENTIFIC_CONTRACT = build_scientific_contract(CONFIG)

print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "parent": get_nested(CONFIG, "lineage.parent"),
            "active_stage": get_nested(CONFIG, "execution.active_stage"),
            "implementation_approved": get_nested(CONFIG, "execution.implementation_approved"),
            "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
            "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
            "execution_counts": get_nested(CONFIG, "execution_contract"),
            "run_length_bins": get_nested(CONFIG, "model.pseudo_gap.run_length_bins"),
            "anchor_distance_bins": get_nested(
                CONFIG, "model.pseudo_gap.nearest_anchor_distance_bins"
            ),
            "shrinkage_support_k": get_nested(
                CONFIG, "model.uncertainty_table.shrinkage_support_k"
            ),
            "scientific_contract_sha256": SCIENTIFIC_CONTRACT[
                "scientific_contract_sha256"
            ],
        },
        indent=2,
        sort_keys=True,
    )
)


# %% [markdown]
# ## 10. Run the approved Stage 0 readout only


# %%
if EXECUTE_NOTEBOOK:
    validate_scientific_contract(CONFIG, require_run_approval=True)
    STAGE0_SUMMARY = run_stage0_experiment(CONFIG)

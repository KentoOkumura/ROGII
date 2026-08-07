# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # exp294 calibrated Type Well gap-fill known-prefix self-GR HMM — Stage 0 train audit
#
# This compact self-contained notebook implements only the approved Stage 0
# pseudo-missing signal audit. It does not run the exp223 HMM, train a model,
# read suffix `TVT`, generate inference predictions, or write a submission.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, path, JSON, and SHA helpers
# 3. Input scan and stable pseudo-missing plan
# 4. Type Well preparation and robust affine calibration
# 5. Control interpolation and hybrid donor reconstruction
# 6. Stage 0 metrics and hard gates
# 7. Artifact generation and orchestration
# 8. Setup, cost guard, and execution switch

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
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SAFE_HORIZONTAL_COLUMNS = ("TVT_input", "GR")
FORBIDDEN_STAGE0_COLUMNS = frozenset(
    {
        "TVT",
        "target",
        "true_tvt",
        "tvt_true",
        "suffix_tvt",
        "error",
        "oracle",
        "candidate_tvt",
    }
)
QUANTILE_LABELS = ("q25", "q50", "q90")
QUANTILE_VALUES = (0.25, 0.50, 0.90)
MANIFEST_COLUMNS = (
    "well",
    "reporting_fold",
    "quantile_label",
    "requested_block_length",
    "block_id",
    "block_start_row",
    "block_end_row_exclusive",
    "row_position",
    "row_identity",
)


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


# %% [markdown]
# ## 2. Runtime, path, JSON, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
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


def print_json(title: str, value: Mapping[str, Any]) -> None:
    print(f"\n## {title}")
    print(json.dumps(to_jsonable(dict(value)), indent=2, sort_keys=True))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def package_dir() -> Path:
    root = project_root()
    candidates = (
        Path.cwd(),
        root / "experiments" / EXPERIMENT_NAME,
    )
    for candidate in candidates:
        config = read_yaml(candidate / "config.yaml")
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return candidate
    return root / "experiments" / EXPERIMENT_NAME


def load_experiment_config() -> dict[str, Any]:
    path = package_dir() / "config.yaml"
    config = read_yaml(path)
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise FileNotFoundError(f"exp294 config not found at {path}")
    return config


def resolve_train_dir(config: Mapping[str, Any]) -> Path:
    configured = str(get_nested(config, "data.train_dir", "data/raw/train"))
    root = project_root()
    candidates = [
        Path(configured),
        root / configured,
        package_dir() / configured,
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "data" / "raw" / "train",
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(path for path in KAGGLE_INPUT_ROOT.glob("**/train") if path.is_dir())
    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists() and next(candidate.glob("*__horizontal_well.csv"), None) is not None:
            return candidate.resolve()
    raise FileNotFoundError("raw train directory not found; checked:\n" + "\n".join(checked[:50]))


def resolve_artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        path = package_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(dict(value)), indent=2, sort_keys=True) + "\n")


def write_stable_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def stable_uint64(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def stable_reporting_fold(well: str, n_folds: int = 5) -> int:
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    digest = hashlib.sha256(str(well).encode("utf-8")).digest()
    return int(int.from_bytes(digest[:8], byteorder="big", signed=False) % int(n_folds))


def round_half_up(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"run-length quantile must be finite and non-negative: {value}")
    return int(math.floor(value + 0.5))


# %% [markdown]
# ## 3. Input scan and stable pseudo-missing plan


# %%
@dataclass(frozen=True)
class WellScan:
    well: str
    reporting_fold: int
    rows: int
    known_rows: int
    finite_known_gr_rows: int
    natural_missing_run_lengths: tuple[int, ...]
    horizontal_name: str
    typewell_name: str
    horizontal_sha256: str
    typewell_sha256: str


def list_well_ids(data_dir: Path) -> list[str]:
    wells: list[str] = []
    for path in sorted(data_dir.glob("*__horizontal_well.csv")):
        well = path.name.removesuffix("__horizontal_well.csv")
        if (data_dir / f"{well}__typewell.csv").exists():
            wells.append(well)
    return wells


def safe_horizontal(path: Path) -> pd.DataFrame:
    header = set(pd.read_csv(path, nrows=0).columns)
    forbidden = FORBIDDEN_STAGE0_COLUMNS.intersection(SAFE_HORIZONTAL_COLUMNS)
    if forbidden:
        raise AssertionError(
            f"internal safe column contract includes forbidden columns: {sorted(forbidden)}"
        )
    missing = sorted(set(SAFE_HORIZONTAL_COLUMNS) - header)
    if missing:
        raise ValueError(f"{path.name} is missing safe Stage 0 columns: {missing}")
    frame = pd.read_csv(path, usecols=list(SAFE_HORIZONTAL_COLUMNS))
    if FORBIDDEN_STAGE0_COLUMNS.intersection(frame.columns):
        raise AssertionError(f"truth-bearing columns entered Stage 0: {list(frame.columns)}")
    frame["TVT_input"] = pd.to_numeric(frame["TVT_input"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    return frame


def contiguous_true_run_lengths(mask: np.ndarray) -> tuple[int, ...]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("run-length mask must be one-dimensional")
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return tuple(int(end - start) for start, end in zip(edges[::2], edges[1::2], strict=True))


def scan_inputs(data_dir: Path, n_folds: int) -> list[WellScan]:
    scans: list[WellScan] = []
    wells = list_well_ids(data_dir)
    if not wells:
        raise ValueError(f"no horizontal/typewell pairs found under {data_dir}")
    for index, well in enumerate(wells, start=1):
        horizontal_path = data_dir / f"{well}__horizontal_well.csv"
        typewell_path = data_dir / f"{well}__typewell.csv"
        horizontal = safe_horizontal(horizontal_path)
        known = np.isfinite(horizontal["TVT_input"].to_numpy(np.float64))
        gr_finite = np.isfinite(horizontal["GR"].to_numpy(np.float64))
        runs = contiguous_true_run_lengths(known & ~gr_finite)
        scans.append(
            WellScan(
                well=well,
                reporting_fold=stable_reporting_fold(well, n_folds),
                rows=int(len(horizontal)),
                known_rows=int(known.sum()),
                finite_known_gr_rows=int((known & gr_finite).sum()),
                natural_missing_run_lengths=runs,
                horizontal_name=horizontal_path.name,
                typewell_name=typewell_path.name,
                horizontal_sha256=sha256_path(horizontal_path),
                typewell_sha256=sha256_path(typewell_path),
            )
        )
        if index % 100 == 0 or index == len(wells):
            print(f"input scan [{index}/{len(wells)}]", flush=True)
    return scans


def fold_external_run_lengths(
    scans: Sequence[WellScan],
    *,
    n_folds: int,
    quantiles: Sequence[float],
    clip_rows: Sequence[int],
    fallback_rows: Sequence[int],
) -> dict[int, dict[str, int]]:
    if len(quantiles) != len(QUANTILE_LABELS):
        raise ValueError("Stage 0 requires exactly q25/q50/q90 quantiles")
    if len(fallback_rows) != len(QUANTILE_LABELS):
        raise ValueError("Stage 0 fallback must contain exactly three run lengths")
    low, high = (int(clip_rows[0]), int(clip_rows[1]))
    output: dict[int, dict[str, int]] = {}
    for fold in range(int(n_folds)):
        external = [
            length
            for scan in scans
            if scan.reporting_fold != fold
            for length in scan.natural_missing_run_lengths
        ]
        if external:
            values = [
                int(np.clip(round_half_up(float(np.quantile(external, quantile))), low, high))
                for quantile in quantiles
            ]
        else:
            values = [int(np.clip(int(value), low, high)) for value in fallback_rows]
        output[fold] = dict(zip(QUANTILE_LABELS, values, strict=True))
    return output


def candidate_block_starts(
    *,
    known_mask: np.ndarray,
    finite_raw_gr: np.ndarray,
    typewell_in_range: np.ndarray,
    occupied: np.ndarray,
    block_length: int,
    edge_rows: int,
) -> list[int]:
    known_positions = np.flatnonzero(known_mask)
    if len(known_positions) == 0 or block_length <= 0:
        return []
    first = int(known_positions[0])
    last = int(known_positions[-1])
    lower = first + int(edge_rows)
    upper = last - int(edge_rows) - int(block_length) + 1
    if upper < lower:
        return []
    candidates: list[int] = []
    for start in range(lower, upper + 1):
        end = start + int(block_length)
        rows = slice(start, end)
        if not bool(np.all(known_mask[rows])):
            continue
        if not bool(np.all(finite_raw_gr[rows])):
            continue
        if not bool(np.all(typewell_in_range[rows])):
            continue
        if bool(np.any(occupied[rows])):
            continue
        left_anchor = bool(np.any(known_mask[:start] & finite_raw_gr[:start] & ~occupied[:start]))
        right_anchor = bool(np.any(known_mask[end:] & finite_raw_gr[end:] & ~occupied[end:]))
        if not (left_anchor and right_anchor):
            continue
        candidates.append(start)
    return candidates


def choose_stable_start(
    candidates: Sequence[int],
    *,
    fold: int,
    well: str,
    quantile_label: str,
) -> int:
    if not candidates:
        raise ValueError("cannot choose from an empty candidate list")
    offset = stable_uint64(EXPERIMENT_NAME, fold, well, quantile_label) % len(candidates)
    return int(candidates[int(offset)])


def build_pseudo_missing_manifest(
    data_dir: Path,
    scans: Sequence[WellScan],
    fold_lengths: Mapping[int, Mapping[str, int]],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_rows = int(
        get_nested(config, "stage0.pseudo_mask.minimum_distance_from_known_prefix_edge_rows")
    )
    manifest_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for scan_index, scan in enumerate(scans, start=1):
        horizontal = safe_horizontal(data_dir / scan.horizontal_name)
        raw_gr = horizontal["GR"].to_numpy(np.float64)
        tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
        known_mask = np.isfinite(tvt_input)
        finite_raw_gr = np.isfinite(raw_gr)
        occupied = np.zeros(len(horizontal), dtype=bool)
        try:
            typewell = pd.read_csv(data_dir / scan.typewell_name, usecols=["TVT", "GR"])
            typewell_tvt, typewell_gr = prepare_typewell_curve(typewell)
            sampled = interpolate_no_extrapolation(tvt_input, typewell_tvt, typewell_gr)
            typewell_in_range = np.isfinite(sampled)
            typewell_status = "ok"
        except (ValueError, KeyError) as exc:
            typewell_in_range = np.zeros(len(horizontal), dtype=bool)
            typewell_status = f"invalid_typewell:{type(exc).__name__}"

        for label in QUANTILE_LABELS:
            length = int(fold_lengths[scan.reporting_fold][label])
            candidates = candidate_block_starts(
                known_mask=known_mask,
                finite_raw_gr=finite_raw_gr,
                typewell_in_range=typewell_in_range,
                occupied=occupied,
                block_length=length,
                edge_rows=edge_rows,
            )
            if not candidates:
                selection_rows.append(
                    {
                        "well": scan.well,
                        "reporting_fold": scan.reporting_fold,
                        "quantile_label": label,
                        "requested_block_length": length,
                        "status": "skipped_no_candidate"
                        if typewell_status == "ok"
                        else typewell_status,
                        "candidate_count": 0,
                        "selected_start": np.nan,
                    }
                )
                continue
            start = choose_stable_start(
                candidates,
                fold=scan.reporting_fold,
                well=scan.well,
                quantile_label=label,
            )
            end = start + length
            occupied[start:end] = True
            block_id = f"{scan.well}:f{scan.reporting_fold}:{label}:{start}:{length}"
            selection_rows.append(
                {
                    "well": scan.well,
                    "reporting_fold": scan.reporting_fold,
                    "quantile_label": label,
                    "requested_block_length": length,
                    "status": "selected",
                    "candidate_count": len(candidates),
                    "selected_start": start,
                }
            )
            for row_position in range(start, end):
                manifest_rows.append(
                    {
                        "well": scan.well,
                        "reporting_fold": scan.reporting_fold,
                        "quantile_label": label,
                        "requested_block_length": length,
                        "block_id": block_id,
                        "block_start_row": start,
                        "block_end_row_exclusive": end,
                        "row_position": row_position,
                        "row_identity": f"{scan.well}_{row_position}",
                    }
                )
        if scan_index % 100 == 0 or scan_index == len(scans):
            print(f"pseudo-mask plan [{scan_index}/{len(scans)}]", flush=True)

    manifest = pd.DataFrame(manifest_rows, columns=list(MANIFEST_COLUMNS))
    selection = pd.DataFrame(selection_rows)
    if manifest.empty:
        raise ValueError("Stage 0 pseudo-missing manifest is empty")
    manifest = manifest.sort_values(["well", "row_position"]).reset_index(drop=True)
    selection = selection.sort_values(["well", "quantile_label"]).reset_index(drop=True)
    if manifest.duplicated(["well", "row_position"]).any():
        raise AssertionError("pseudo-missing blocks overlap within a well")
    if FORBIDDEN_STAGE0_COLUMNS.intersection(manifest.columns):
        raise AssertionError("truth-bearing columns entered the frozen pseudo-mask manifest")
    return manifest, selection


# %% [markdown]
# ## 4. Type Well preparation and robust affine calibration


# %%
@dataclass(frozen=True)
class AffineFit:
    valid: bool
    slope: float
    intercept: float
    pair_count: int
    typewell_gr_iqr: float
    residual_scale: float
    fit_rmse: float
    iterations: int
    converged: bool
    fallback_reason: str


def invalid_affine(reason: str, *, pair_count: int = 0, iqr: float = float("nan")) -> AffineFit:
    return AffineFit(
        valid=False,
        slope=float("nan"),
        intercept=float("nan"),
        pair_count=int(pair_count),
        typewell_gr_iqr=float(iqr),
        residual_scale=float("nan"),
        fit_rmse=float("nan"),
        iterations=0,
        converged=False,
        fallback_reason=reason,
    )


def prepare_typewell_curve(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = sorted({"TVT", "GR"} - set(typewell.columns))
    if missing:
        raise ValueError(f"Type Well is missing columns: {missing}")
    frame = pd.DataFrame(
        {
            "TVT": pd.to_numeric(typewell["TVT"], errors="coerce"),
            "GR": pd.to_numeric(typewell["GR"], errors="coerce"),
        }
    ).dropna()
    frame = frame[np.isfinite(frame["TVT"]) & np.isfinite(frame["GR"])]
    frame = frame.groupby("TVT", as_index=False, sort=True)["GR"].median()
    if len(frame) < 2:
        raise ValueError("Type Well requires at least two finite unique TVT/GR points")
    tvt = frame["TVT"].to_numpy(np.float64)
    gr = frame["GR"].to_numpy(np.float64)
    if not bool(np.all(np.diff(tvt) > 0)):
        raise ValueError("prepared Type Well TVT must be strictly increasing")
    return tvt, gr


def interpolate_no_extrapolation(
    query: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    output = np.full(query.shape, np.nan, dtype=np.float64)
    inside = (
        np.isfinite(query) & (query >= float(typewell_tvt[0])) & (query <= float(typewell_tvt[-1]))
    )
    output[inside] = np.interp(query[inside], typewell_tvt, typewell_gr)
    return output


def fit_huber_affine(
    x: np.ndarray,
    y: np.ndarray,
    *,
    minimum_pairs: int,
    minimum_iqr: float,
    huber_k: float,
    max_iterations: int,
    relative_tolerance: float,
    scale_floor: float,
) -> AffineFit:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    pair_count = int(len(x))
    if pair_count < int(minimum_pairs):
        return invalid_affine("insufficient_pairs", pair_count=pair_count)
    iqr = float(np.quantile(x, 0.75) - np.quantile(x, 0.25))
    if not math.isfinite(iqr) or iqr < float(minimum_iqr):
        return invalid_affine("typewell_gr_iqr_below_minimum", pair_count=pair_count, iqr=iqr)
    design = np.column_stack([x, np.ones(pair_count, dtype=np.float64)])
    if np.linalg.matrix_rank(design) < 2:
        return invalid_affine("singular_ordinary_least_squares", pair_count=pair_count, iqr=iqr)
    try:
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return invalid_affine("ordinary_least_squares_failure", pair_count=pair_count, iqr=iqr)
    if not bool(np.isfinite(coefficients).all()):
        return invalid_affine("nonfinite_ordinary_least_squares", pair_count=pair_count, iqr=iqr)

    converged = False
    residual_scale = float("nan")
    iterations = 0
    for iteration in range(1, int(max_iterations) + 1):
        residual = y - design @ coefficients
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        residual_scale = max(float(scale_floor), 1.4826 * mad)
        threshold = float(huber_k) * residual_scale
        abs_residual = np.abs(residual)
        weights = np.ones_like(abs_residual)
        tail = abs_residual > threshold
        weights[tail] = threshold / np.clip(abs_residual[tail], np.finfo(np.float64).eps, None)
        root_weights = np.sqrt(weights)
        weighted_design = design * root_weights[:, None]
        weighted_y = y * root_weights
        if np.linalg.matrix_rank(weighted_design) < 2:
            return invalid_affine("singular_huber_update", pair_count=pair_count, iqr=iqr)
        try:
            updated = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return invalid_affine("huber_update_failure", pair_count=pair_count, iqr=iqr)
        if not bool(np.isfinite(updated).all()):
            return invalid_affine("nonfinite_huber_update", pair_count=pair_count, iqr=iqr)
        denominator = max(float(np.linalg.norm(coefficients)), 1.0)
        relative_change = float(np.linalg.norm(updated - coefficients) / denominator)
        coefficients = updated
        iterations = iteration
        if relative_change <= float(relative_tolerance):
            converged = True
            break
    if not converged:
        return AffineFit(
            valid=False,
            slope=float(coefficients[0]),
            intercept=float(coefficients[1]),
            pair_count=pair_count,
            typewell_gr_iqr=iqr,
            residual_scale=residual_scale,
            fit_rmse=float(np.sqrt(np.mean((y - design @ coefficients) ** 2))),
            iterations=iterations,
            converged=False,
            fallback_reason="huber_not_converged",
        )
    fitted = design @ coefficients
    return AffineFit(
        valid=True,
        slope=float(coefficients[0]),
        intercept=float(coefficients[1]),
        pair_count=pair_count,
        typewell_gr_iqr=iqr,
        residual_scale=residual_scale,
        fit_rmse=float(np.sqrt(np.mean((y - fitted) ** 2))),
        iterations=iterations,
        converged=True,
        fallback_reason="none",
    )


def affine_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    prefix = "gapfill.affine"
    return {
        "minimum_pairs": int(get_nested(config, f"{prefix}.minimum_pairs")),
        "minimum_iqr": float(get_nested(config, f"{prefix}.minimum_typewell_gr_iqr")),
        "huber_k": float(get_nested(config, f"{prefix}.huber_k")),
        "max_iterations": int(get_nested(config, f"{prefix}.max_iterations")),
        "relative_tolerance": float(get_nested(config, f"{prefix}.coefficient_relative_tolerance")),
        "scale_floor": float(get_nested(config, f"{prefix}.residual_scale_floor")),
    }


# %% [markdown]
# ## 5. Control interpolation and hybrid donor reconstruction


# %%
def exp223_linear_interpolation(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64), dtype="float64")
    fill_value = float(np.nanmedian(series.to_numpy(np.float64))) if series.notna().any() else 0.0
    return series.interpolate(limit_direction="both").fillna(fill_value).to_numpy(np.float64)


def reconstruct_well(
    *,
    well: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    pseudo_rows: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_gr = horizontal["GR"].to_numpy(np.float64)
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    known_mask = np.isfinite(tvt_input)
    raw_missing_mask = ~np.isfinite(raw_gr)
    pseudo_mask = np.zeros(len(horizontal), dtype=bool)
    pseudo_rows = np.asarray(pseudo_rows, dtype=np.int64)
    if len(pseudo_rows):
        if int(pseudo_rows.min()) < 0 or int(pseudo_rows.max()) >= len(horizontal):
            raise IndexError(f"pseudo row outside {well} row range")
        pseudo_mask[pseudo_rows] = True
    if bool(np.any(pseudo_mask & (~known_mask | raw_missing_mask))):
        raise AssertionError(f"{well}: pseudo mask must contain finite known-prefix GR only")

    masked_gr = raw_gr.copy()
    masked_gr[raw_missing_mask | pseudo_mask] = np.nan
    typewell_tvt, typewell_gr = prepare_typewell_curve(typewell)
    sampled_typewell = interpolate_no_extrapolation(tvt_input, typewell_tvt, typewell_gr)
    fit_mask = known_mask & np.isfinite(masked_gr) & np.isfinite(sampled_typewell)
    pseudo_fit_overlap = int(np.sum(fit_mask & pseudo_mask))
    if pseudo_fit_overlap:
        raise AssertionError(f"{well}: pseudo-masked rows entered affine fit")
    fit = fit_huber_affine(sampled_typewell[fit_mask], masked_gr[fit_mask], **affine_spec(config))

    control = exp223_linear_interpolation(masked_gr)
    hybrid = control.copy()
    observed_after_mask = np.isfinite(masked_gr)
    hybrid[observed_after_mask] = masked_gr[observed_after_mask]
    gapfill_mask = (
        known_mask & ~observed_after_mask & np.isfinite(sampled_typewell) & bool(fit.valid)
    )
    if fit.valid:
        hybrid[gapfill_mask] = fit.slope * sampled_typewell[gapfill_mask] + fit.intercept

    observed_parity = bool(np.array_equal(hybrid[observed_after_mask], raw_gr[observed_after_mask]))
    raw_mask_after = raw_missing_mask.copy()
    raw_mask_parity = bool(np.array_equal(raw_missing_mask, raw_mask_after))
    target_fill_count = int(np.sum(gapfill_mask & ~known_mask))
    pseudo_finite = np.isfinite(control[pseudo_mask]) & np.isfinite(hybrid[pseudo_mask])
    finite_coverage = float(np.mean(pseudo_finite)) if len(pseudo_rows) else 1.0

    predictions = pd.DataFrame(
        {
            "well": well,
            "row_position": pseudo_rows,
            "true_gr": raw_gr[pseudo_rows],
            "control_gr": control[pseudo_rows],
            "variant_gr": hybrid[pseudo_rows],
            "typewell_gr": sampled_typewell[pseudo_rows],
            "affine_valid": bool(fit.valid),
            "affine_slope": fit.slope,
            "affine_intercept": fit.intercept,
        }
    )
    audit = {
        "well": well,
        **asdict(fit),
        "selected_pseudo_rows": int(len(pseudo_rows)),
        "natural_raw_missing_rows": int((known_mask & raw_missing_mask).sum()),
        "gapfilled_natural_rows": int((gapfill_mask & raw_missing_mask).sum()),
        "gapfilled_pseudo_rows": int((gapfill_mask & pseudo_mask).sum()),
        "observed_known_gr_exact_parity": observed_parity,
        "raw_missing_mask_exact_parity": raw_mask_parity,
        "pseudo_mask_fit_overlap_rows": pseudo_fit_overlap,
        "target_side_typewell_fill_count": target_fill_count,
        "pseudo_prediction_finite_coverage": finite_coverage,
    }
    return predictions, audit


def evaluate_frozen_manifest(
    data_dir: Path,
    scans: Sequence[WellScan],
    manifest: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    rows_by_well = {
        str(well): group["row_position"].to_numpy(np.int64)
        for well, group in manifest.groupby("well", sort=True)
    }
    for index, scan in enumerate(scans, start=1):
        horizontal = safe_horizontal(data_dir / scan.horizontal_name)
        typewell = pd.read_csv(data_dir / scan.typewell_name, usecols=["TVT", "GR"])
        pseudo_rows = rows_by_well.get(scan.well, np.array([], dtype=np.int64))
        predictions, audit = reconstruct_well(
            well=scan.well,
            horizontal=horizontal,
            typewell=typewell,
            pseudo_rows=pseudo_rows,
            config=config,
        )
        if not predictions.empty:
            prediction_frames.append(predictions)
        audit_rows.append(audit)
        if index % 100 == 0 or index == len(scans):
            print(f"truth-late Stage 0 evaluation [{index}/{len(scans)}]", flush=True)
    if not prediction_frames:
        raise ValueError("Stage 0 produced no held-out predictions")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.merge(
        manifest,
        on=["well", "row_position"],
        how="left",
        validate="one_to_one",
    )
    if predictions[list(MANIFEST_COLUMNS[1:])].isna().any().any():
        raise AssertionError("held-out predictions did not preserve frozen manifest identity")
    predictions = predictions.sort_values(["well", "row_position"]).reset_index(drop=True)
    audits = pd.DataFrame(audit_rows).sort_values("well").reset_index(drop=True)
    return predictions, audits


# %% [markdown]
# ## 6. Stage 0 metrics and hard gates


# %%
def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def mae(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.mean(np.abs(truth - prediction)))


def zncc(truth: np.ndarray, prediction: np.ndarray, *, minimum_length: int = 4) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[valid]
    prediction = prediction[valid]
    if len(truth) < int(minimum_length) or float(np.std(truth)) <= np.finfo(np.float64).eps:
        return float("nan")
    prediction_std = float(np.std(prediction))
    if prediction_std <= np.finfo(np.float64).eps:
        return 0.0
    truth_z = (truth - float(np.mean(truth))) / float(np.std(truth))
    prediction_z = (prediction - float(np.mean(prediction))) / prediction_std
    return float(np.mean(truth_z * prediction_z))


def derivative_ncc(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if len(truth) < 3:
        return float("nan")
    return zncc(np.diff(truth), np.diff(prediction), minimum_length=2)


def weighted_block_mean(blocks: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(blocks[column], errors="coerce").to_numpy(np.float64)
    weights = pd.to_numeric(blocks["block_length"], errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not bool(valid.any()):
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def build_block_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["well", "reporting_fold", "quantile_label", "block_id"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        truth = group["true_gr"].to_numpy(np.float64)
        control = group["control_gr"].to_numpy(np.float64)
        variant = group["variant_gr"].to_numpy(np.float64)
        control_rmse = rmse(truth, control)
        variant_rmse = rmse(truth, variant)
        control_zncc = zncc(truth, control)
        variant_zncc = zncc(truth, variant)
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "block_length": int(len(group)),
                "control_rmse": control_rmse,
                "variant_rmse": variant_rmse,
                "rmse_delta": variant_rmse - control_rmse,
                "control_mae": mae(truth, control),
                "variant_mae": mae(truth, variant),
                "control_zncc": control_zncc,
                "variant_zncc": variant_zncc,
                "zncc_delta": variant_zncc - control_zncc,
                "control_derivative_ncc": derivative_ncc(truth, control),
                "variant_derivative_ncc": derivative_ncc(truth, variant),
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def aggregate_metrics(
    predictions: pd.DataFrame,
    blocks: pd.DataFrame,
    *,
    group_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in predictions.groupby(group_column, sort=True):
        group_blocks = blocks[blocks[group_column] == key]
        truth = group["true_gr"].to_numpy(np.float64)
        control = group["control_gr"].to_numpy(np.float64)
        variant = group["variant_gr"].to_numpy(np.float64)
        control_rmse = rmse(truth, control)
        variant_rmse = rmse(truth, variant)
        control_zncc = weighted_block_mean(group_blocks, "control_zncc")
        variant_zncc = weighted_block_mean(group_blocks, "variant_zncc")
        rows.append(
            {
                group_column: key,
                "rows": int(len(group)),
                "blocks": int(len(group_blocks)),
                "control_rmse": control_rmse,
                "variant_rmse": variant_rmse,
                "rmse_delta": variant_rmse - control_rmse,
                "rmse_relative_improvement": (control_rmse - variant_rmse) / control_rmse,
                "control_mae": mae(truth, control),
                "variant_mae": mae(truth, variant),
                "control_zncc": control_zncc,
                "variant_zncc": variant_zncc,
                "zncc_delta": variant_zncc - control_zncc,
            }
        )
    return pd.DataFrame(rows).sort_values(group_column).reset_index(drop=True)


def evaluate_hard_gates(
    predictions: pd.DataFrame,
    blocks: pd.DataFrame,
    by_well: pd.DataFrame,
    by_fold: pd.DataFrame,
    audits: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    truth = predictions["true_gr"].to_numpy(np.float64)
    control = predictions["control_gr"].to_numpy(np.float64)
    variant = predictions["variant_gr"].to_numpy(np.float64)
    control_rmse = rmse(truth, control)
    variant_rmse = rmse(truth, variant)
    control_zncc = weighted_block_mean(blocks, "control_zncc")
    variant_zncc = weighted_block_mean(blocks, "variant_zncc")
    relative_improvement = (control_rmse - variant_rmse) / control_rmse
    zncc_delta = variant_zncc - control_zncc
    rmse_improving_folds = int((by_fold["rmse_delta"] < 0).sum())
    zncc_positive_folds = int((by_fold["zncc_delta"] > 0).sum())
    by_well_p95_delta = float(np.quantile(by_well["rmse_delta"].to_numpy(np.float64), 0.95))
    finite_coverage = float(
        np.mean(np.isfinite(predictions["control_gr"]) & np.isfinite(predictions["variant_gr"]))
    )

    gates_config = get_nested(config, "stage0.hard_gates") or {}
    checks = {
        "pooled_rmse_relative_improvement": bool(
            relative_improvement >= float(gates_config["pooled_rmse_relative_improvement_min"])
        ),
        "pooled_zncc_delta": bool(zncc_delta >= float(gates_config["pooled_zncc_delta_min"])),
        "rmse_improving_reporting_folds": bool(
            rmse_improving_folds >= int(gates_config["rmse_improving_reporting_folds_min"])
        ),
        "zncc_positive_reporting_folds": bool(
            zncc_positive_folds >= int(gates_config["zncc_positive_reporting_folds_min"])
        ),
        "by_well_rmse_p95_nonregression": bool(
            by_well_p95_delta <= float(gates_config["by_well_rmse_p95_delta_max"])
        ),
        "observed_known_gr_exact_parity": bool(audits["observed_known_gr_exact_parity"].all()),
        "raw_missing_mask_exact_parity": bool(audits["raw_missing_mask_exact_parity"].all()),
        "pseudo_mask_excluded_from_fit": bool((audits["pseudo_mask_fit_overlap_rows"] == 0).all()),
        "target_side_typewell_fill_count": bool(
            int(audits["target_side_typewell_fill_count"].sum())
            <= int(gates_config["target_side_typewell_fill_count_max"])
        ),
        "finite_output_coverage": bool(
            finite_coverage >= float(gates_config["finite_output_coverage_min"])
        ),
        "all_reporting_folds_present": bool(
            len(by_fold) == int(get_nested(config, "validation.reporting_folds"))
        ),
    }
    pooled = {
        "rows": int(len(predictions)),
        "blocks": int(len(blocks)),
        "wells": int(predictions["well"].nunique()),
        "reporting_folds": int(predictions["reporting_fold"].nunique()),
        "control_rmse": control_rmse,
        "variant_rmse": variant_rmse,
        "rmse_delta": variant_rmse - control_rmse,
        "rmse_relative_improvement": relative_improvement,
        "control_mae": mae(truth, control),
        "variant_mae": mae(truth, variant),
        "control_zncc": control_zncc,
        "variant_zncc": variant_zncc,
        "zncc_delta": zncc_delta,
        "rmse_improving_reporting_folds": rmse_improving_folds,
        "zncc_positive_reporting_folds": zncc_positive_folds,
        "by_well_rmse_p95_delta": by_well_p95_delta,
        "finite_output_coverage": finite_coverage,
    }
    return pooled, {"checks": checks, "passed": bool(all(checks.values()))}


# %% [markdown]
# ## 7. Artifact generation and orchestration


# %%
def validate_implementation_contract(config: Mapping[str, Any]) -> None:
    assertions = {
        "route": get_nested(config, "experiment.route") == "ensemble",
        "stage0_audit_variants": int(get_nested(config, "stage0.audit_variants")) == 1,
        "lightgbm_configs": int(get_nested(config, "model.lightgbm_configs")) == 0,
        "trained_folds": int(get_nested(config, "model.trained_folds")) == 0,
        "boosters": int(get_nested(config, "model.boosters")) == 0,
        "no_parent_control_regeneration": get_nested(config, "execution.regenerate_parent_control")
        is False,
        "no_stage1": get_nested(config, "execution.run_stage1") is False,
        "no_inference": get_nested(config, "execution.run_inference") is False,
        "no_submission": get_nested(config, "execution.write_submission") is False,
        "target_fill_disabled": get_nested(config, "gapfill.target_side_typewell_fill_enabled")
        is False,
        "observed_gr_preserved": get_nested(config, "gapfill.preserve_observed_gr_exactly") is True,
        "raw_mask_preserved": get_nested(config, "gapfill.preserve_raw_missing_mask") is True,
        "anchor_eligibility_preserved": get_nested(config, "gapfill.preserve_anchor_eligibility")
        is True,
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AssertionError(f"exp294 implementation contract failed: {failed}")


def input_manifest_frame(scans: Sequence[WellScan]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scan in scans:
        rows.extend(
            [
                {
                    "well": scan.well,
                    "reporting_fold": scan.reporting_fold,
                    "role": "raw_horizontal_safe_scan",
                    "file_name": scan.horizontal_name,
                    "raw_sha256": scan.horizontal_sha256,
                    "rows": scan.rows,
                },
                {
                    "well": scan.well,
                    "reporting_fold": scan.reporting_fold,
                    "role": "raw_typewell",
                    "file_name": scan.typewell_name,
                    "raw_sha256": scan.typewell_sha256,
                    "rows": np.nan,
                },
            ]
        )
    return pd.DataFrame(rows).sort_values(["well", "role"]).reset_index(drop=True)


def output_sha(path: Path) -> dict[str, Any]:
    value = {"file": path.name, "raw_sha256": sha256_path(path), "bytes": int(path.stat().st_size)}
    if path.suffix == ".gz":
        value["decompressed_content_sha256"] = sha256_gzip_decompressed(path)
    return value


def run_stage0(
    config: Mapping[str, Any] | None = None,
    *,
    train_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    config = dict(config or load_experiment_config())
    validate_implementation_contract(config)
    if not bool(get_nested(config, "execution.run_stage0", False)):
        raise RuntimeError(
            "Stage 0 execution switch is false; obtain user approval before Kaggle CPU execution"
        )
    train_dir = Path(train_dir or resolve_train_dir(config))
    output_dir = Path(output_dir or resolve_artifact_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    n_folds = int(get_nested(config, "validation.reporting_folds"))
    scans = scan_inputs(train_dir, n_folds)
    missing_spec = get_nested(config, "stage0.missing_run_lengths") or {}
    fold_lengths = fold_external_run_lengths(
        scans,
        n_folds=n_folds,
        quantiles=missing_spec["quantiles"],
        clip_rows=missing_spec["clip_rows"],
        fallback_rows=missing_spec["no_run_fallback_rows"],
    )

    input_manifest = input_manifest_frame(scans)
    input_manifest_path = output_dir / "stage0_input_manifest.csv"
    input_manifest.to_csv(input_manifest_path, index=False)

    pseudo_manifest, selection = build_pseudo_missing_manifest(
        train_dir, scans, fold_lengths, config
    )
    pseudo_manifest_path = output_dir / "stage0_pseudo_missing_manifest.csv.gz"
    selection_path = output_dir / "stage0_selection_summary.csv"
    write_stable_gzip_csv(pseudo_manifest, pseudo_manifest_path)
    selection.to_csv(selection_path, index=False)
    pseudo_manifest_sha = sha256_gzip_decompressed(pseudo_manifest_path)
    print(f"frozen pseudo-mask decompressed SHA256={pseudo_manifest_sha}")

    predictions, audits = evaluate_frozen_manifest(train_dir, scans, pseudo_manifest, config)
    block_metrics = build_block_metrics(predictions)
    by_well = aggregate_metrics(predictions, block_metrics, group_column="well")
    by_fold = aggregate_metrics(predictions, block_metrics, group_column="reporting_fold")
    pooled, hard_gates = evaluate_hard_gates(
        predictions,
        block_metrics,
        by_well,
        by_fold,
        audits,
        config,
    )

    predictions_path = output_dir / "stage0_heldout_predictions.csv.gz"
    affine_path = output_dir / "stage0_affine_fit_summary.csv"
    block_path = output_dir / "stage0_by_block_metrics.csv"
    well_path = output_dir / "stage0_by_well_metrics.csv"
    fold_path = output_dir / "stage0_fold_metrics.csv"
    summary_path = output_dir / "stage0_summary.json"
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    write_stable_gzip_csv(predictions, predictions_path)
    audits.to_csv(affine_path, index=False)
    block_metrics.to_csv(block_path, index=False)
    by_well.to_csv(well_path, index=False)
    by_fold.to_csv(fold_path, index=False)

    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_completed_pass" if hard_gates["passed"] else "stage0_completed_fail",
        "route": "ensemble",
        "stage": "stage0_pseudo_missing_signal_audit",
        "control": get_nested(config, "stage0.control"),
        "variant": get_nested(config, "stage0.variant"),
        "rows": int(len(predictions)),
        "blocks": int(len(block_metrics)),
        "wells": int(predictions["well"].nunique()),
        "input_wells": int(len(scans)),
        "fold_external_missing_run_lengths": fold_lengths,
        "selection_status_counts": selection["status"].value_counts(dropna=False).to_dict(),
        "affine_valid_wells": int(audits["valid"].sum()),
        "affine_fallback_reason_counts": audits["fallback_reason"]
        .value_counts(dropna=False)
        .to_dict(),
        "pooled": pooled,
        "hard_gates": hard_gates,
        "stage1_authorized": False,
        "stage1_next_action": (
            "request_separate_user_approval_before_stage1_implementation"
            if hard_gates["passed"]
            else "close_without_stage1_or_rescue_grid"
        ),
        "leakage_contract": {
            "horizontal_columns_read_before_prediction_freeze": list(SAFE_HORIZONTAL_COLUMNS),
            "suffix_tvt_read": False,
            "pseudo_manifest_truth_columns": [],
            "pseudo_mask_frozen_before_truth_join": True,
            "pseudo_mask_decompressed_content_sha256": pseudo_manifest_sha,
            "target_side_typewell_fill_count": int(audits["target_side_typewell_fill_count"].sum()),
        },
        "reproducibility": {
            "reporting_fold_policy": "sha256_utf8_first8_big_endian_uint64_mod_5",
            "mask_selection_policy": "sha256_experiment_fold_well_quantile_mod_sorted_candidates",
            "global_rng_used": False,
            "model_manifest": "not_applicable_no_trained_model",
            "submission_sha": "not_applicable_no_submission",
            "deterministic_anchor": False,
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "outputs": {
            "input_manifest": input_manifest_path.name,
            "pseudo_missing_manifest": pseudo_manifest_path.name,
            "selection_summary": selection_path.name,
            "heldout_predictions": predictions_path.name,
            "affine_fit_summary": affine_path.name,
            "by_block_metrics": block_path.name,
            "by_well_metrics": well_path.name,
            "fold_metrics": fold_path.name,
            "summary": summary_path.name,
            "artifact_manifest": artifact_manifest_path.name,
        },
    }
    write_json(summary_path, summary)

    artifact_paths = [
        input_manifest_path,
        pseudo_manifest_path,
        selection_path,
        predictions_path,
        affine_path,
        block_path,
        well_path,
        fold_path,
        summary_path,
    ]
    artifact_manifest = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage0_pseudo_missing_signal_audit",
        "input_file_count": int(len(input_manifest)),
        "input_manifest_content_sha256": sha256_path(input_manifest_path),
        "feature_schema_sha256": sha256_json(list(MANIFEST_COLUMNS)),
        "pseudo_mask_decompressed_content_sha256": pseudo_manifest_sha,
        "saved_exp223_control": "not_applicable_stage0_signal_only",
        "model_manifest": "not_applicable_no_trained_model",
        "submission": "not_applicable_no_submission",
        "artifacts": [output_sha(path) for path in artifact_paths],
    }
    write_json(artifact_manifest_path, artifact_manifest)
    print_json("Stage 0 summary", summary)
    print_json("Artifact manifest", artifact_manifest)
    return summary


# %% [markdown]
# ## 8. Setup, cost guard, and execution switch

# %%
CONFIG = load_experiment_config()
validate_implementation_contract(CONFIG)
print_json(
    "experiment and cost guard",
    {
        "experiment": get_nested(CONFIG, "experiment.name"),
        "status": get_nested(CONFIG, "experiment.status"),
        "route": get_nested(CONFIG, "experiment.route"),
        "parent": get_nested(CONFIG, "lineage.parent"),
        "stage0_audit_variants": get_nested(CONFIG, "stage0.audit_variants"),
        "stage1_hmm_variants": get_nested(CONFIG, "stage1.active_variants"),
        "stage1_hmm_well_runs": get_nested(CONFIG, "model.hmm_well_runs"),
        "lightgbm_configs": get_nested(CONFIG, "model.lightgbm_configs"),
        "trained_folds": get_nested(CONFIG, "model.trained_folds"),
        "boosters": get_nested(CONFIG, "model.boosters"),
        "parent_control_retraining": get_nested(CONFIG, "execution.regenerate_parent_control"),
        "gpu": get_nested(CONFIG, "runtime.gpu_enabled"),
        "run_stage0": get_nested(CONFIG, "execution.run_stage0"),
        "run_stage1": get_nested(CONFIG, "execution.run_stage1"),
        "run_inference": get_nested(CONFIG, "execution.run_inference"),
        "write_submission": get_nested(CONFIG, "execution.write_submission"),
    },
)
print_json(
    "Stage 0 scientific contract",
    {
        "safe_horizontal_columns": list(SAFE_HORIZONTAL_COLUMNS),
        "forbidden_columns": sorted(FORBIDDEN_STAGE0_COLUMNS),
        "reporting_fold_policy": get_nested(CONFIG, "validation.reporting_fold_policy"),
        "truth_join_policy": get_nested(CONFIG, "validation.truth_join_policy"),
        "gapfill_scope": get_nested(CONFIG, "gapfill.scope"),
        "affine": get_nested(CONFIG, "gapfill.affine"),
        "pseudo_mask": get_nested(CONFIG, "stage0.pseudo_mask"),
        "hard_gates": get_nested(CONFIG, "stage0.hard_gates"),
    },
)

if in_notebook_runtime():
    is_kaggle = KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()
    allow_local = os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1"
    if bool(get_nested(CONFIG, "execution.run_stage0", False)):
        if not is_kaggle and not allow_local:
            raise RuntimeError(
                "Stage 0 notebook execution is Kaggle-first; local execution is disabled"
            )
        STAGE0_SUMMARY = run_stage0(CONFIG)
    else:
        print("Stage 0 is implemented but execution.run_stage0=false; no audit was run.")
        print("Enable it only after explicit Kaggle CPU execution approval.")

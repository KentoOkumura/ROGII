# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp292 typewell GR warp-rate identifiability audit
#
# This train-side diagnostic ranks five already-saved exp268 candidate paths.
# Candidate construction, calibration, scoring, selection, and the shuffled
# negative control are target-free. Truth is loaded only after score and
# selection tables have been persisted and content-hashed.

# %% [markdown]
# ## Contents
# 1. Imports and immutable contract
# 2. Runtime, path, SHA, and input helpers
# 3. Target-free validation and parent preflight
# 4. Type Well calibration and forward-GR score
# 5. Deterministic fold, shuffle, and selection helpers
# 6. Post-freeze truth readout and success guards
# 7. Audit orchestration and artifacts
# 8. Execution

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

EXPERIMENT_NAME = "exp292_typewell_gr_warp_rate_identifiability_audit"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
CANDIDATES = [
    "hmm_ir_tail30",
    "hmm_ir_w32",
    "hmm_ir_w64",
    "hmm_ir_w128",
    "hmm_ir_w256",
]
SAFE_CANDIDATE = CANDIDATES[0]
HORIZONS = [128, 256, 512]
PRIMARY_HORIZON = 256
FORBIDDEN_TARGET_TOKENS = ("target", "true_tvt", "error", "oracle")
FORBIDDEN_EXACT_COLUMNS = {"TVT", "abs_error"}


# %% [markdown]
# ## 2. Runtime, path, SHA, and input helpers


# %%
def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = [
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.rglob("config.yaml")))
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp292 config not found; checked {len(candidates)} paths")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def train_data_dir(config: dict[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
        slug = "rogii-wellbore-geology-prediction"
        direct = (
            KAGGLE_INPUT_ROOT / slug / "train",
            KAGGLE_INPUT_ROOT / "competitions" / slug / "train",
        )
        for candidate in direct:
            if candidate.is_dir() and next(candidate.glob("*__horizontal_well.csv"), None):
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.rglob("train")):
            if candidate.is_dir() and next(candidate.glob("*__horizontal_well.csv"), None):
                return candidate
    return project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            checked.append(str(path))
            if path.exists() and path.stat().st_size > 0:
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.rglob(filename)):
            checked.append(str(path))
            if path.exists() and path.stat().st_size > 0:
                return path
    raise FileNotFoundError(f"Could not resolve {filename}; checked={checked}")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_content_sha256(frame: pd.DataFrame) -> str:
    canonical = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataframe_schema_sha256(frame: pd.DataFrame) -> str:
    schema = "\n".join(f"{column}:{frame[column].dtype}" for column in frame.columns) + "\n"
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, frame: pd.DataFrame, *, gzip_output: bool = False) -> None:
    if gzip_output:
        frame.to_csv(
            path,
            index=False,
            float_format="%.12g",
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
    else:
        frame.to_csv(path, index=False, float_format="%.12g")


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean((prediction[valid] - truth[valid]) ** 2)))


# %% [markdown]
# ## 3. Target-free validation and parent preflight


# %%
def validate_scientific_contract(config: dict[str, Any]) -> None:
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(get_nested(config, "experiment.route") == "pf_beam", "route must be pf_beam")
    require(get_nested(config, "execution.implementation") is True, "implementation flag")
    require(get_nested(config, "execution.active_audit_variants") == 1, "one audit variant")
    require(get_nested(config, "execution.lightgbm_config_count") == 0, "zero ML configs")
    require(get_nested(config, "execution.trained_fold_count") == 0, "zero trained folds")
    require(get_nested(config, "execution.total_boosters") == 0, "zero boosters")
    require(get_nested(config, "execution.hmm_pf_well_runs") == 0, "zero HMM/PF runs")
    require(get_nested(config, "execution.control_or_parent_retraining") is False, "no retraining")
    require(get_nested(config, "inference.enabled") is False, "inference disabled")
    require(get_nested(config, "inference.create_submission") is False, "submission disabled")
    require(get_nested(config, "candidate_bank.order") == CANDIDATES, "fixed candidate order")
    require(get_nested(config, "candidate_bank.safe_candidate") == SAFE_CANDIDATE, "safe tail30")
    require(get_nested(config, "candidate_bank.regenerate_candidates") is False, "no regeneration")
    require(get_nested(config, "audit.horizons_rows") == HORIZONS, "fixed horizons")
    require(get_nested(config, "audit.primary_horizon_rows") == PRIMARY_HORIZON, "primary H256")
    require(get_nested(config, "audit.prefix_calibration.maximum_rows") == 512, "prefix max 512")
    require(get_nested(config, "audit.prefix_calibration.minimum_pairs") == 40, "prefix min 40")
    require(
        get_nested(config, "audit.prefix_calibration.robust_iterations") == 2, "two robust fits"
    )
    require(get_nested(config, "audit.composite.weights") == [1 / 3, 1 / 3, 1 / 3], "equal weights")
    require(get_nested(config, "audit.shuffled_control.seed") == 42, "fixed shuffle seed")
    if failures:
        raise ValueError("exp292 scientific contract violation: " + "; ".join(failures))


def validate_target_free_frame(frame: pd.DataFrame) -> None:
    forbidden = []
    for column in frame.columns:
        lower = str(column).lower()
        if column in FORBIDDEN_EXACT_COLUMNS or any(
            token in lower for token in FORBIDDEN_TARGET_TOKENS
        ):
            forbidden.append(str(column))
    if forbidden:
        raise ValueError(f"target-free frame contains forbidden truth columns: {sorted(forbidden)}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        value = json.load(fp)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def preflight_exp268_aggregate(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = get_nested(config, "data.exp268_shards") or {}
    summary_path = resolve_existing(
        str(spec["aggregate_summary_filename"]), spec.get("aggregate_summary_candidates") or []
    )
    manifest_path = resolve_existing(
        str(spec["aggregate_manifest_filename"]), spec.get("aggregate_manifest_candidates") or []
    )
    summary = load_json(summary_path)
    expected_summary_sha = str(spec.get("expected_aggregate_summary_sha256") or "")
    expected_manifest_sha = str(spec.get("expected_aggregate_manifest_sha256") or "")
    if not expected_summary_sha or sha256_path(summary_path) != expected_summary_sha:
        raise RuntimeError("exp268 aggregate summary SHA mismatch")
    if not expected_manifest_sha or sha256_path(manifest_path) != expected_manifest_sha:
        raise RuntimeError("exp268 aggregate manifest SHA mismatch")
    expected_rows = int(spec["expected_candidate_rows"])
    expected_wells = int(spec["expected_wells"])
    if summary.get("status") != "completed_train_side_candidate_bank_audit_pending_review":
        raise RuntimeError(f"exp268 aggregate is not complete: status={summary.get('status')}")
    if (
        int(summary.get("rows", -1)) != expected_rows
        or int(summary.get("wells", -1)) != expected_wells
    ):
        raise RuntimeError("exp268 aggregate coverage does not match frozen exp292 contract")
    direct_candidates = [
        candidate
        for candidate in (summary.get("direct_candidates") or [])
        if candidate in CANDIDATES
    ]
    if direct_candidates != CANDIDATES:
        raise RuntimeError("exp268 aggregate candidate bank/order mismatch")
    rate_spread = summary.get("rate_spread") or {}
    if int(rate_spread.get("zero_rate_spread_wells", expected_wells)) >= expected_wells:
        raise RuntimeError("exp268 aggregate does not establish candidate diversity")
    expected_prediction_sha = str(spec.get("expected_prediction_content_sha256") or "")
    if summary.get("prediction_content_sha256") != expected_prediction_sha:
        raise RuntimeError("exp268 aggregate prediction content SHA mismatch")

    manifest = pd.read_csv(manifest_path).fillna("")
    shard_rows = manifest.loc[manifest["role"].astype(str).str.match(r"^shard[01]$")]
    if len(shard_rows) != int(spec["expected_shards"]):
        raise RuntimeError("exp268 aggregate manifest must contain exactly both frozen shards")
    for shard_spec in spec.get("shard_specs") or []:
        role = f"shard{int(shard_spec['shard_index'])}"
        row = shard_rows.loc[shard_rows["role"] == role]
        if len(row) != 1:
            raise RuntimeError(f"exp268 aggregate manifest missing {role}")
        actual = str(row.iloc[0].get("decompressed_sha256", ""))
        expected = str(shard_spec.get("expected_decompressed_sha256") or "")
        if not expected or actual != expected:
            raise RuntimeError(f"exp268 aggregate manifest SHA mismatch for {role}")
    records = [
        {
            "role": "exp268_aggregate_summary",
            "path": str(summary_path),
            "raw_sha256": sha256_path(summary_path),
            "decompressed_sha256": None,
            "rows": summary["rows"],
            "wells": summary["wells"],
        },
        {
            "role": "exp268_aggregate_manifest",
            "path": str(manifest_path),
            "raw_sha256": sha256_path(manifest_path),
            "decompressed_sha256": None,
            "rows": len(manifest),
            "wells": None,
        },
    ]
    return summary, records


def strict_same_ids(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    if reference["id"].duplicated().any() or other["id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate ids")
    reference_ids = reference["id"].astype(str).to_numpy()
    indexed = other.set_index(other["id"].astype(str), drop=False)
    if len(reference) != len(other) or set(reference_ids) != set(indexed.index):
        raise ValueError(f"{label} id coverage mismatch")
    return indexed.loc[reference_ids].reset_index(drop=True)


def load_target_free_candidate_bank(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    spec = get_nested(config, "data.exp268_shards") or {}
    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []
    shard_columns = ["id", "well", "row_idx", "prefix_rows", *CANDIDATES[1:]]
    for shard_spec in spec.get("shard_specs") or []:
        path = resolve_existing(str(shard_spec["filename"]), shard_spec["candidates"])
        actual_sha = sha256_gzip_decompressed(path)
        expected_sha = str(shard_spec.get("expected_decompressed_sha256") or "")
        if not expected_sha or actual_sha != expected_sha:
            raise ValueError(f"decompressed SHA mismatch for {path}")
        frame = pd.read_csv(path, usecols=shard_columns, dtype={"id": str, "well": str})
        if len(frame) != int(shard_spec["expected_rows"]):
            raise ValueError(f"row mismatch for shard{shard_spec['shard_index']}")
        if frame["well"].nunique() != int(shard_spec["expected_wells"]):
            raise ValueError(f"well mismatch for shard{shard_spec['shard_index']}")
        validate_target_free_frame(frame)
        frames.append(frame)
        manifest.append(
            {
                "role": f"exp268_shard{int(shard_spec['shard_index'])}",
                "path": str(path),
                "raw_sha256": sha256_path(path),
                "decompressed_sha256": actual_sha,
                "rows": len(frame),
                "wells": frame["well"].nunique(),
            }
        )
    if len(frames) != int(spec["expected_shards"]):
        raise ValueError("all configured exp268 shards are required")
    bank = pd.concat(frames, ignore_index=True)
    if bank["id"].duplicated().any():
        raise ValueError("exp268 shard union contains duplicate ids")
    bank = bank.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)
    if len(bank) != int(spec["expected_candidate_rows"]) or bank["well"].nunique() != int(
        spec["expected_wells"]
    ):
        raise ValueError("exp268 target-free shard union coverage mismatch")

    tail_spec = get_nested(config, "data.exp209_tail30_control") or {}
    tail_path = resolve_existing(str(tail_spec["filename"]), tail_spec["candidates"])
    tail_sha = sha256_gzip_decompressed(tail_path)
    if tail_sha != str(tail_spec["expected_decompressed_sha256"]):
        raise ValueError("exp209 tail30 decompressed SHA mismatch")
    tail = (
        pd.read_csv(
            tail_path,
            usecols=["id", "well", "last_known_tvt", "md_since", "hmm_mean_tvt"],
            dtype={"id": str, "well": str},
        )
        .sort_values(["well", "id"], kind="mergesort")
        .reset_index(drop=True)
    )
    bank = bank.sort_values(["well", "id"], kind="mergesort").reset_index(drop=True)
    tail = strict_same_ids(bank, tail, "exp209 tail30 control")
    if not np.array_equal(bank["well"].astype(str), tail["well"].astype(str)):
        raise ValueError("tail30 well labels differ from exp268 bank")
    bank[SAFE_CANDIDATE] = numeric_array(tail, "hmm_mean_tvt")
    bank["last_known_tvt"] = numeric_array(tail, "last_known_tvt")
    bank["md_since"] = numeric_array(tail, "md_since")
    bank = bank.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)
    validate_target_free_frame(bank)
    if not np.isfinite(bank[CANDIDATES].to_numpy(np.float64)).all():
        raise ValueError("candidate bank contains non-finite paths")
    manifest.append(
        {
            "role": "exp209_tail30_control",
            "path": str(tail_path),
            "raw_sha256": sha256_path(tail_path),
            "decompressed_sha256": tail_sha,
            "rows": len(tail),
            "wells": tail["well"].nunique(),
        }
    )
    return bank, manifest


def well_from_horizontal_path(path: Path) -> str:
    suffix = "__horizontal_well.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected horizontal filename: {path.name}")
    return path.name[: -len(suffix)]


def load_target_safe_horizontal(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = ["MD", "GR", "TVT_input"]
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"{path.name} missing target-free columns: {missing}")
    frame = pd.read_csv(path, usecols=required)
    validate_target_free_frame(frame)
    return frame


def load_truth_vector(path: Path) -> np.ndarray:
    if "TVT" not in pd.read_csv(path, nrows=0).columns:
        raise ValueError(f"{path.name} missing TVT truth")
    return pd.to_numeric(pd.read_csv(path, usecols=["TVT"])["TVT"], errors="coerce").to_numpy(
        np.float64
    )


def load_typewell(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted({"TVT", "GR"} - set(header))
    if missing:
        raise ValueError(f"{path.name} missing Type Well columns: {missing}")
    return pd.read_csv(path, usecols=["TVT", "GR"])


# %% [markdown]
# ## 4. Type Well calibration and forward-GR score


# %%
class AuditEligibilityError(ValueError):
    """A fixed-contract reason to use the safe candidate for one well/horizon."""


@dataclass(frozen=True)
class CalibrationResult:
    valid: bool
    reason: str
    slope: float
    intercept: float
    sigma: float
    derivative_sigma: float
    pairs: int
    retained_pairs: int
    typewell_gr_std: float
    prefix_rmse: float


def invalid_calibration(reason: str, pairs: int = 0) -> CalibrationResult:
    return CalibrationResult(
        valid=False,
        reason=reason,
        slope=float("nan"),
        intercept=float("nan"),
        sigma=float("nan"),
        derivative_sigma=float("nan"),
        pairs=pairs,
        retained_pairs=0,
        typewell_gr_std=float("nan"),
        prefix_rmse=float("nan"),
    )


def prepare_typewell_curve(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    clean = pd.DataFrame(
        {
            "TVT": pd.to_numeric(typewell["TVT"], errors="coerce"),
            "GR": pd.to_numeric(typewell["GR"], errors="coerce"),
        }
    ).dropna()
    clean = clean.sort_values("TVT", kind="mergesort")
    clean = clean.groupby("TVT", as_index=False, sort=True)["GR"].median()
    if len(clean) < 2:
        raise AuditEligibilityError("typewell_fewer_than_two_unique_points")
    tvt = clean["TVT"].to_numpy(np.float64)
    gr = clean["GR"].to_numpy(np.float64)
    if not np.all(np.diff(tvt) > 0):
        raise ValueError("Type Well TVT must be strictly increasing")
    return tvt, gr


def interpolate_no_extrapolation(
    query: np.ndarray, typewell_tvt: np.ndarray, typewell_gr: np.ndarray
) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    output = np.full(query.shape, np.nan, dtype=np.float64)
    inside = (
        np.isfinite(query) & (query >= float(typewell_tvt[0])) & (query <= float(typewell_tvt[-1]))
    )
    output[inside] = np.interp(query[inside], typewell_tvt, typewell_gr)
    return output


def typewell_gradient(
    midpoint_tvt: np.ndarray, typewell_tvt: np.ndarray, typewell_gr: np.ndarray
) -> np.ndarray:
    midpoint_tvt = np.asarray(midpoint_tvt, dtype=np.float64)
    output = np.full(midpoint_tvt.shape, np.nan, dtype=np.float64)
    inside = (
        np.isfinite(midpoint_tvt)
        & (midpoint_tvt >= typewell_tvt[0])
        & (midpoint_tvt <= typewell_tvt[-1])
    )
    indices = np.searchsorted(typewell_tvt, midpoint_tvt[inside], side="right") - 1
    indices = np.clip(indices, 0, len(typewell_tvt) - 2)
    denominator = typewell_tvt[indices + 1] - typewell_tvt[indices]
    output[inside] = (typewell_gr[indices + 1] - typewell_gr[indices]) / denominator
    return output


def median_mad(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    median = float(np.median(finite))
    return median, float(np.median(np.abs(finite - median)))


def robust_affine_calibration(
    horizontal_gr: np.ndarray,
    tvt_input: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    config: dict[str, Any],
) -> CalibrationResult:
    spec = get_nested(config, "audit.prefix_calibration") or {}
    maximum_rows = int(spec["maximum_rows"])
    minimum_pairs = int(spec["minimum_pairs"])
    reference = interpolate_no_extrapolation(tvt_input, typewell_tvt, typewell_gr)
    valid = np.isfinite(horizontal_gr) & np.isfinite(reference) & np.isfinite(tvt_input)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) > maximum_rows:
        valid_indices = valid_indices[-maximum_rows:]
    if len(valid_indices) < minimum_pairs:
        return invalid_calibration("prefix_pairs_below_minimum", len(valid_indices))
    x = reference[valid_indices]
    y = np.asarray(horizontal_gr, dtype=np.float64)[valid_indices]
    reference_std = float(np.std(x))
    if reference_std < float(spec["minimum_typewell_gr_std"]):
        return CalibrationResult(
            False,
            "prefix_typewell_gr_std_below_minimum",
            *(float("nan"),) * 4,
            len(x),
            len(x),
            reference_std,
            float("nan"),
        )
    keep = np.ones(len(x), dtype=bool)
    slope = intercept = float("nan")
    for _ in range(int(spec["robust_iterations"])):
        design = np.column_stack([x[keep], np.ones(int(keep.sum()))])
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = np.abs(y - (slope * x + intercept))
        cutoff = float(np.quantile(residual[keep], float(spec["trim_quantile"])))
        next_keep = residual <= cutoff
        if int(next_keep.sum()) < minimum_pairs:
            return CalibrationResult(
                False,
                "trimmed_prefix_pairs_below_minimum",
                slope,
                intercept,
                float("nan"),
                float("nan"),
                len(x),
                int(next_keep.sum()),
                reference_std,
                float("nan"),
            )
        keep = next_keep
    fit = slope * x + intercept
    residual = y - fit
    prefix_rmse = float(np.sqrt(np.mean(residual[keep] ** 2)))
    slope_low, slope_high = [float(value) for value in spec["slope_range"]]
    if not slope_low <= slope <= slope_high:
        return CalibrationResult(
            False,
            "calibration_slope_out_of_range",
            slope,
            intercept,
            float("nan"),
            float("nan"),
            len(x),
            int(keep.sum()),
            reference_std,
            prefix_rmse,
        )
    if prefix_rmse > float(spec["maximum_prefix_rmse"]):
        return CalibrationResult(
            False,
            "prefix_rmse_above_maximum",
            slope,
            intercept,
            float("nan"),
            float("nan"),
            len(x),
            int(keep.sum()),
            reference_std,
            prefix_rmse,
        )
    _, residual_mad = median_mad(residual[keep])
    sigma_low, sigma_high = [
        float(value) for value in get_nested(config, "audit.gaussian_component.sigma_clip")
    ]
    sigma = float(np.clip(1.4826 * residual_mad, sigma_low, sigma_high))
    retained_residual = residual[keep]
    _, diff_mad = median_mad(np.diff(retained_residual))
    derivative_low, derivative_high = [
        float(value) for value in get_nested(config, "audit.derivative_component.scale_clip")
    ]
    derivative_sigma = float(np.clip(1.4826 * diff_mad, derivative_low, derivative_high))
    return CalibrationResult(
        True,
        "ok",
        float(slope),
        float(intercept),
        sigma,
        derivative_sigma,
        len(x),
        int(keep.sum()),
        reference_std,
        prefix_rmse,
    )


def candidate_robust_zscore(values: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median, mad = median_mad(values)
    threshold = float(get_nested(config, "audit.composite.zero_mad_threshold"))
    if not math.isfinite(mad) or mad < threshold:
        return np.zeros_like(values)
    low, high = [float(value) for value in get_nested(config, "audit.composite.zscore_clip")]
    return np.clip((values - median) / (1.4826 * mad + 1.0e-6), low, high)


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) <= 0 or np.std(right) <= 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def score_candidate_horizon(
    observed_gr: np.ndarray,
    candidate_paths: dict[str, np.ndarray],
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    calibration: CalibrationResult,
    horizon_rows: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate_order = list(get_nested(config, "candidate_bank.order"))
    effective = min(int(horizon_rows), len(observed_gr))
    minimum_effective = int(get_nested(config, "audit.short_horizon_min_rows"))
    if not calibration.valid:
        raise AuditEligibilityError(calibration.reason)
    if effective < minimum_effective:
        raise AuditEligibilityError("effective_horizon_below_minimum")
    observed = np.asarray(observed_gr[:effective], dtype=np.float64)
    references: dict[str, np.ndarray] = {}
    derivative_references: dict[str, np.ndarray] = {}
    common = np.isfinite(observed)
    for candidate in candidate_order:
        path = np.asarray(candidate_paths[candidate][:effective], dtype=np.float64)
        raw_reference = interpolate_no_extrapolation(path, typewell_tvt, typewell_gr)
        references[candidate] = calibration.slope * raw_reference + calibration.intercept
        midpoint = (path[:-1] + path[1:]) / 2.0
        derivative_references[candidate] = (
            calibration.slope
            * typewell_gradient(midpoint, typewell_tvt, typewell_gr)
            * np.diff(path)
        )
        common &= np.isfinite(path) & np.isfinite(references[candidate])
    required = max(
        int(get_nested(config, "audit.common_pair_coverage.minimum_pairs")),
        int(
            math.ceil(
                float(
                    get_nested(
                        config, "audit.common_pair_coverage.minimum_fraction_of_effective_horizon"
                    )
                )
                * effective
            )
        ),
    )
    common_rows = int(common.sum())
    if common_rows < required:
        raise AuditEligibilityError("common_finite_pairs_below_minimum")
    common_delta = common[:-1] & common[1:] & np.isfinite(np.diff(observed))
    if int(common_delta.sum()) < max(1, required - 1):
        raise AuditEligibilityError("common_derivative_pairs_below_minimum")

    raw_rows: list[dict[str, Any]] = []
    min_std = float(get_nested(config, "audit.prefix_calibration.minimum_typewell_gr_std"))
    min_derivative = float(
        get_nested(
            config, "audit.derivative_component.minimum_median_absolute_forward_delta_gr_per_row"
        )
    )
    likelihood_clip = float(get_nested(config, "audit.gaussian_component.log_likelihood_clip"))
    for candidate_index, candidate in enumerate(candidate_order):
        reference = references[candidate]
        derivative_reference = derivative_references[candidate]
        forward_std = float(np.std(reference[common]))
        median_abs_derivative = float(np.median(np.abs(derivative_reference[common_delta])))
        if forward_std < min_std:
            raise AuditEligibilityError(f"candidate_forward_gr_std_below_minimum:{candidate}")
        if median_abs_derivative < min_derivative:
            raise AuditEligibilityError(f"candidate_forward_derivative_below_minimum:{candidate}")
        residual_z2 = ((observed[common] - reference[common]) / calibration.sigma) ** 2
        gaussian = float(np.mean(-0.5 * np.minimum(residual_z2, likelihood_clip)))
        ncc = pearson_correlation(observed[common], reference[common])
        derivative = float(
            -np.median(np.abs(np.diff(observed)[common_delta] - derivative_reference[common_delta]))
            / calibration.derivative_sigma
        )
        if not all(math.isfinite(value) for value in (gaussian, ncc, derivative)):
            raise AuditEligibilityError(f"non_finite_score_component:{candidate}")
        raw_rows.append(
            {
                "candidate": candidate,
                "candidate_index": candidate_index,
                "gaussian": gaussian,
                "ncc": ncc,
                "derivative": derivative,
                "forward_gr_std": forward_std,
                "median_abs_forward_delta_gr": median_abs_derivative,
            }
        )
    scores = pd.DataFrame(raw_rows)
    weights = np.asarray(get_nested(config, "audit.composite.weights"), dtype=np.float64)
    for component in ("gaussian", "ncc", "derivative"):
        scores[f"{component}_z"] = candidate_robust_zscore(scores[component].to_numpy(), config)
    scores["composite"] = scores[["gaussian_z", "ncc_z", "derivative_z"]].to_numpy() @ weights
    scores["eligible"] = True
    scores["eligibility_reason"] = "ok"
    meta = {
        "effective_horizon_rows": effective,
        "common_pairs": common_rows,
        "required_common_pairs": required,
    }
    return scores, meta


# %% [markdown]
# ## 5. Deterministic fold, shuffle, and selection helpers


# %%
def stable_rotation_offset(well: str, suffix_rows: int, config: dict[str, Any]) -> int | None:
    minimum = max(
        int(get_nested(config, "audit.shuffled_control.minimum_rotation_rows")),
        int(
            math.ceil(
                float(get_nested(config, "audit.shuffled_control.minimum_rotation_fraction"))
                * suffix_rows
            )
        ),
    )
    maximum = suffix_rows - minimum
    if minimum > maximum:
        return None
    seed = int(get_nested(config, "audit.shuffled_control.seed"))
    key = f"{EXPERIMENT_NAME}::{seed}::{well}".encode()
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    rng = np.random.default_rng(local_seed)
    return int(rng.integers(minimum, maximum + 1))


def assign_canonical_group_folds(wells: Iterable[str], n_splits: int = 5) -> dict[str, int]:
    ordered = np.asarray(sorted({str(well) for well in wells}), dtype=object)
    if len(ordered) < n_splits:
        raise ValueError("fewer wells than GroupKFold splits")
    folds: dict[str, int] = {}
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros((len(ordered), 1), dtype=np.float64)
    for fold, (_, valid_indices) in enumerate(splitter.split(dummy, groups=ordered)):
        for index in valid_indices:
            folds[str(ordered[index])] = fold
    return folds


def invalid_score_rows(reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate": candidate,
                "candidate_index": index,
                "gaussian": np.nan,
                "ncc": np.nan,
                "derivative": np.nan,
                "forward_gr_std": np.nan,
                "median_abs_forward_delta_gr": np.nan,
                "gaussian_z": np.nan,
                "ncc_z": np.nan,
                "derivative_z": np.nan,
                "composite": np.nan,
                "eligible": False,
                "eligibility_reason": reason,
            }
            for index, candidate in enumerate(CANDIDATES)
        ]
    )


def select_target_free(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["well", "fold", "horizon_rows", "control"]
    for key, group in scores.groupby(keys, sort=False, dropna=False):
        eligible = bool(group["eligible"].all()) and np.isfinite(group["composite"]).all()
        if eligible:
            ordered = group.sort_values(
                ["composite", "candidate_index"], ascending=[False, True], kind="mergesort"
            )
            selected = ordered.iloc[0]
            candidate = str(selected["candidate"])
            score = float(selected["composite"])
            reason = "score_top1"
        else:
            candidate = SAFE_CANDIDATE
            score = float("nan")
            reason = str(group["eligibility_reason"].iloc[0])
        rows.append(
            {
                "well": str(key[0]),
                "fold": int(key[1]),
                "horizon_rows": int(key[2]),
                "control": str(key[3]),
                "selected_candidate": candidate,
                "selected_score": score,
                "eligible": eligible,
                "selection_reason": reason,
                "effective_horizon_rows": int(group["effective_horizon_rows"].iloc[0]),
                "common_pairs": int(group["common_pairs"].iloc[0]),
                "truth_attached": False,
            }
        )
    return pd.DataFrame(rows).sort_values(keys, kind="mergesort").reset_index(drop=True)


def score_one_well_target_free(
    well: str,
    bank: pd.DataFrame,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    fold: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_target_free_frame(bank)
    validate_target_free_frame(horizontal)
    ordered = bank.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_idx = pd.to_numeric(ordered["row_idx"], errors="raise").to_numpy(np.int64)
    if len(row_idx) == 0 or row_idx.min() < 0 or row_idx.max() >= len(horizontal):
        raise ValueError(f"candidate row_idx outside horizontal frame for well={well}")
    observed_suffix = numeric_array(horizontal, "GR")[row_idx]
    candidate_paths = {candidate: numeric_array(ordered, candidate) for candidate in CANDIDATES}
    try:
        typewell_tvt, typewell_gr = prepare_typewell_curve(typewell)
        calibration = robust_affine_calibration(
            numeric_array(horizontal, "GR"),
            numeric_array(horizontal, "TVT_input"),
            typewell_tvt,
            typewell_gr,
            config,
        )
    except AuditEligibilityError as exc:
        typewell_tvt = np.asarray([0.0, 1.0])
        typewell_gr = np.asarray([0.0, 0.0])
        calibration = invalid_calibration(str(exc))
    rotation = stable_rotation_offset(well, len(observed_suffix), config)
    controls: dict[str, np.ndarray | None] = {
        "real": observed_suffix,
        "shuffled": None if rotation is None else np.roll(observed_suffix, rotation),
    }
    score_parts: list[pd.DataFrame] = []
    eligibility_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for control, observed in controls.items():
            meta = {
                "effective_horizon_rows": min(horizon, len(observed_suffix)),
                "common_pairs": 0,
                "required_common_pairs": max(
                    32, int(math.ceil(0.5 * min(horizon, len(observed_suffix))))
                ),
            }
            try:
                if observed is None:
                    raise AuditEligibilityError("shuffle_rotation_range_empty")
                part, meta = score_candidate_horizon(
                    observed,
                    candidate_paths,
                    typewell_tvt,
                    typewell_gr,
                    calibration,
                    horizon,
                    config,
                )
            except AuditEligibilityError as exc:
                part = invalid_score_rows(str(exc))
            part.insert(0, "control", control)
            part.insert(0, "horizon_rows", horizon)
            part.insert(0, "fold", fold)
            part.insert(0, "well", str(well))
            for key, value in meta.items():
                part[key] = value
            part["truth_attached"] = False
            score_parts.append(part)
            eligibility_rows.append(
                {
                    "well": str(well),
                    "fold": fold,
                    "horizon_rows": horizon,
                    "control": control,
                    **asdict(calibration),
                    "rotation_rows": rotation,
                    **meta,
                    "score_eligible": bool(part["eligible"].all()),
                    "score_reason": str(part["eligibility_reason"].iloc[0]),
                }
            )
    return pd.concat(score_parts, ignore_index=True), pd.DataFrame(eligibility_rows)


# %% [markdown]
# ## 6. Post-freeze truth readout and success guards


# %%
def best_candidate_labels(
    truth: np.ndarray, paths: dict[str, np.ndarray], horizon: int, atol: float
) -> dict[str, bool]:
    effective = min(horizon, len(truth))
    losses = {
        candidate: rmse(truth[:effective], paths[candidate][:effective]) for candidate in CANDIDATES
    }
    finite_losses = [value for value in losses.values() if math.isfinite(value)]
    if not finite_losses:
        return {candidate: False for candidate in CANDIDATES}
    best = min(finite_losses)
    return {
        candidate: math.isfinite(losses[candidate]) and abs(losses[candidate] - best) <= atol
        for candidate in CANDIDATES
    }


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return float("nan")
    return float(roc_auc_score(labels[valid].astype(int), scores[valid]))


def attach_truth_and_compute_metrics(
    bank: pd.DataFrame,
    score_table: pd.DataFrame,
    selection: pd.DataFrame,
    data_dir: Path,
    folds: dict[str, int],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    label_rows: list[dict[str, Any]] = []
    by_well_rows: list[dict[str, Any]] = []
    row_error_parts: list[pd.DataFrame] = []
    atol = float(get_nested(config, "audit.truth_readout.truth_rmse_tie_atol_ft"))
    real_primary_selection = selection.loc[
        (selection["control"] == "real") & (selection["horizon_rows"] == PRIMARY_HORIZON)
    ].set_index("well")
    for well, group in bank.groupby("well", sort=True):
        horizontal_path = data_dir / f"{well}__horizontal_well.csv"
        truth_all = load_truth_vector(horizontal_path)
        ordered = group.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
        row_idx = pd.to_numeric(ordered["row_idx"], errors="raise").to_numpy(np.int64)
        truth = truth_all[row_idx]
        if not np.isfinite(truth).all():
            raise ValueError(f"non-finite unknown-suffix truth for well={well}")
        paths = {candidate: numeric_array(ordered, candidate) for candidate in CANDIDATES}
        for horizon in HORIZONS:
            labels = best_candidate_labels(truth, paths, horizon, atol)
            for candidate in CANDIDATES:
                label_rows.append(
                    {
                        "well": str(well),
                        "horizon_rows": horizon,
                        "candidate": candidate,
                        "candidate_best": labels[candidate],
                        "candidate_rmse": rmse(
                            truth[: min(horizon, len(truth))],
                            paths[candidate][: min(horizon, len(truth))],
                        ),
                    }
                )
        selected_candidate = str(real_primary_selection.loc[str(well), "selected_candidate"])
        safe_prediction = paths[SAFE_CANDIDATE]
        selected_prediction = paths[selected_candidate]
        safe_rmse = rmse(truth, safe_prediction)
        selected_rmse = rmse(truth, selected_prediction)
        by_well_rows.append(
            {
                "well": str(well),
                "fold": int(folds[str(well)]),
                "rows": len(truth),
                "selected_candidate": selected_candidate,
                "selection_eligible": bool(real_primary_selection.loc[str(well), "eligible"]),
                "selected_rmse": selected_rmse,
                "safe_rmse": safe_rmse,
                "rmse_gain_ft": safe_rmse - selected_rmse,
            }
        )
        row_error_parts.append(
            pd.DataFrame(
                {
                    "well": str(well),
                    "fold": int(folds[str(well)]),
                    "md_since": numeric_array(ordered, "md_since"),
                    "selected_sq_error": (selected_prediction - truth) ** 2,
                    "safe_sq_error": (safe_prediction - truth) ** 2,
                }
            )
        )
    labels = pd.DataFrame(label_rows)
    scored = score_table.merge(
        labels, on=["well", "horizon_rows", "candidate"], how="left", validate="many_to_one"
    )
    scored["truth_attached"] = True
    primary = scored.loc[scored["horizon_rows"] == PRIMARY_HORIZON]
    auc_rows: list[dict[str, Any]] = []
    for control in ("real", "shuffled"):
        control_frame = primary.loc[(primary["control"] == control) & primary["eligible"]]
        for scope, fold in [("pooled", None), *[("fold", value) for value in range(5)]]:
            part = (
                control_frame if fold is None else control_frame.loc[control_frame["fold"] == fold]
            )
            auc_rows.append(
                {
                    "control": control,
                    "scope": scope,
                    "fold": fold,
                    "rows": len(part),
                    "wells": part["well"].nunique(),
                    "auc": safe_auc(
                        part["candidate_best"].to_numpy(), part["composite"].to_numpy()
                    ),
                }
            )
    auc_metrics = pd.DataFrame(auc_rows)
    by_well = pd.DataFrame(by_well_rows)
    row_errors = pd.concat(row_error_parts, ignore_index=True)
    rmse_rows: list[dict[str, Any]] = []
    for scope, fold in [("pooled", None), *[("fold", value) for value in range(5)]]:
        part = row_errors if fold is None else row_errors.loc[row_errors["fold"] == fold]
        selected_rmse = float(np.sqrt(part["selected_sq_error"].mean()))
        safe_rmse = float(np.sqrt(part["safe_sq_error"].mean()))
        rmse_rows.append(
            {
                "scope": scope,
                "fold": fold,
                "rows": len(part),
                "wells": part["well"].nunique(),
                "selected_rmse": selected_rmse,
                "safe_rmse": safe_rmse,
                "rmse_gain_ft": safe_rmse - selected_rmse,
            }
        )
    rmse_metrics = pd.DataFrame(rmse_rows)

    assignment_spec = get_nested(config, "data.hidden_like") or {}
    assignment_path = resolve_existing(
        Path(str(assignment_spec["fold_assignment_candidates"][0])).name,
        assignment_spec["fold_assignment_candidates"],
    )
    assignments = pd.read_csv(assignment_path, dtype={"well_id": str})
    subgroup_masks: dict[str, np.ndarray] = {
        "md_since_1000_plus": numeric_array(row_errors, "md_since") >= 1000.0
    }
    for subgroup, role_column in (assignment_spec.get("role_columns") or {}).items():
        if role_column not in assignments.columns:
            raise ValueError(f"hidden-like assignments missing {role_column}")
        valid_wells = set(
            assignments.loc[assignments[role_column].astype(str) == "valid", "well_id"].astype(str)
        )
        subgroup_masks[f"hidden_like_{subgroup}"] = (
            row_errors["well"].astype(str).isin(valid_wells).to_numpy()
        )
    subgroup_rows: list[dict[str, Any]] = []
    for subgroup, mask in subgroup_masks.items():
        if not mask.any():
            raise ValueError(f"subgroup {subgroup} has zero rows")
        part = row_errors.loc[mask]
        selected_rmse = float(np.sqrt(part["selected_sq_error"].mean()))
        safe_rmse = float(np.sqrt(part["safe_sq_error"].mean()))
        subgroup_rows.append(
            {
                "subgroup": subgroup,
                "rows": len(part),
                "wells": part["well"].nunique(),
                "selected_rmse": selected_rmse,
                "safe_rmse": safe_rmse,
                "rmse_gain_ft": safe_rmse - selected_rmse,
                "nonregression": selected_rmse <= safe_rmse,
            }
        )
    subgroup_metrics = pd.DataFrame(subgroup_rows)
    input_record = {
        "role": "hidden_like_assignments",
        "path": str(assignment_path),
        "raw_sha256": sha256_path(assignment_path),
        "decompressed_sha256": None,
        "rows": len(assignments),
        "wells": assignments["well_id"].nunique(),
    }
    return (
        scored,
        auc_metrics,
        rmse_metrics,
        subgroup_metrics,
        {"by_well": by_well, "input_record": input_record},
    )


def evaluate_success_guards(
    selection: pd.DataFrame,
    bank: pd.DataFrame,
    auc_metrics: pd.DataFrame,
    rmse_metrics: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    criteria = get_nested(config, "validation.success_criteria") or {}
    primary = selection.loc[
        (selection["control"] == "real") & (selection["horizon_rows"] == PRIMARY_HORIZON)
    ]
    eligible_wells = primary.loc[primary["eligible"], "well"].astype(str)
    eligible_set = set(eligible_wells)
    well_fraction = len(eligible_set) / max(1, primary["well"].nunique())
    row_fraction = float(bank["well"].astype(str).isin(eligible_set).mean())
    pooled_auc = auc_metrics.loc[auc_metrics["scope"] == "pooled"].set_index("control")["auc"]
    auc_lift = float(pooled_auc.get("real", np.nan) - pooled_auc.get("shuffled", np.nan))
    fold_auc = auc_metrics.loc[auc_metrics["scope"] == "fold"].pivot(
        index="fold", columns="control", values="auc"
    )
    positive_auc_folds = int(((fold_auc.get("real") - fold_auc.get("shuffled")) > 0).sum())
    pooled_rmse_gain = float(
        rmse_metrics.loc[rmse_metrics["scope"] == "pooled", "rmse_gain_ft"].iloc[0]
    )
    improved_rmse_folds = int(
        (rmse_metrics.loc[rmse_metrics["scope"] == "fold", "rmse_gain_ft"] > 0).sum()
    )
    subgroup_nonregression = dict(
        zip(subgroup_metrics["subgroup"], subgroup_metrics["nonregression"], strict=True)
    )
    guards = {
        "technical_validation_passed": True,
        "primary_eligible_well_fraction": well_fraction,
        "primary_eligible_row_fraction": row_fraction,
        "pooled_auc_lift_real_vs_shuffle": auc_lift,
        "positive_auc_lift_folds": positive_auc_folds,
        "pooled_rmse_gain_ft_top1_vs_safe": pooled_rmse_gain,
        "improved_rmse_folds": improved_rmse_folds,
        "md_since_1000_plus_nonregression": bool(
            subgroup_nonregression.get("md_since_1000_plus", False)
        ),
        "hidden_like_spatial_nonregression": bool(
            subgroup_nonregression.get("hidden_like_spatial", False)
        ),
        "hidden_like_typewell_purged_nonregression": bool(
            subgroup_nonregression.get("hidden_like_typewell_purged", False)
        ),
    }
    passed = (
        guards["technical_validation_passed"]
        and well_fraction >= float(criteria["minimum_primary_eligible_well_fraction"])
        and row_fraction >= float(criteria["minimum_primary_eligible_row_fraction"])
        and math.isfinite(auc_lift)
        and auc_lift >= float(criteria["minimum_pooled_auc_lift_real_vs_shuffle"])
        and positive_auc_folds >= int(criteria["minimum_positive_auc_lift_folds"])
        and pooled_rmse_gain >= float(criteria["minimum_pooled_rmse_gain_ft_top1_vs_safe"])
        and improved_rmse_folds >= int(criteria["minimum_improved_rmse_folds"])
        and guards["md_since_1000_plus_nonregression"]
        and guards["hidden_like_spatial_nonregression"]
        and guards["hidden_like_typewell_purged_nonregression"]
    )
    guards["passed"] = passed
    guards["decision"] = (
        "PASS_SAFE_PRESERVING_FOLLOWUP_ONLY" if passed else "FAIL_CLOSE_NO_RESCUE_GRID"
    )
    return guards


# %% [markdown]
# ## 7. Audit orchestration and artifacts


# %%
def run_audit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_experiment_config() if config is None else config
    validate_scientific_contract(config)
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp292 audit must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 is only "
            "for an explicitly approved smoke run."
        )
    started = time.time()
    aggregate_summary, manifest = preflight_exp268_aggregate(config)
    bank, bank_manifest = load_target_free_candidate_bank(config)
    manifest.extend(bank_manifest)
    folds = assign_canonical_group_folds(
        bank["well"].astype(str), int(get_nested(config, "validation.n_folds"))
    )
    fold_manifest = pd.DataFrame(
        [{"well": well, "fold": fold} for well, fold in sorted(folds.items())]
    )
    data_dir = train_data_dir(config)
    max_wells_env = int(os.environ.get("EXPERIMENT_MAX_WELLS", "0") or "0")
    selected_wells = sorted(bank["well"].astype(str).unique())
    if max_wells_env:
        selected_wells = selected_wells[:max_wells_env]
        bank = bank.loc[bank["well"].astype(str).isin(selected_wells)].reset_index(drop=True)
    score_parts: list[pd.DataFrame] = []
    eligibility_parts: list[pd.DataFrame] = []
    grouped_bank = bank.groupby("well", sort=True)
    for index, (raw_well, well_bank) in enumerate(grouped_bank, start=1):
        well = str(raw_well)
        horizontal_path = data_dir / f"{well}__horizontal_well.csv"
        typewell_path = data_dir / f"{well}__typewell.csv"
        horizontal = load_target_safe_horizontal(horizontal_path)
        typewell = load_typewell(typewell_path)
        manifest.extend(
            [
                {
                    "role": "raw_horizontal",
                    "well": well,
                    "path": str(horizontal_path),
                    "raw_sha256": sha256_path(horizontal_path),
                    "decompressed_sha256": None,
                    "rows": len(horizontal),
                    "wells": 1,
                },
                {
                    "role": "raw_typewell",
                    "well": well,
                    "path": str(typewell_path),
                    "raw_sha256": sha256_path(typewell_path),
                    "decompressed_sha256": None,
                    "rows": len(typewell),
                    "wells": 1,
                },
            ]
        )
        well_scores, well_eligibility = score_one_well_target_free(
            well,
            well_bank,
            horizontal,
            typewell,
            folds[well],
            config,
        )
        score_parts.append(well_scores)
        eligibility_parts.append(well_eligibility)
        if index % 50 == 0 or index == len(selected_wells):
            print(f"target-free scoring {index}/{len(selected_wells)}", flush=True)
    scores = pd.concat(score_parts, ignore_index=True)
    eligibility = pd.concat(eligibility_parts, ignore_index=True)
    selection = select_target_free(scores)
    validate_target_free_frame(scores.drop(columns=["truth_attached"]))
    validate_target_free_frame(selection.drop(columns=["truth_attached"]))

    artifacts = artifact_dir()
    paths = {
        "contract": artifacts / f"{OUTPUT_PREFIX}_contract.json",
        "input_manifest": artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv",
        "fold_manifest": artifacts / f"{OUTPUT_PREFIX}_fold_manifest.csv",
        "calibration_eligibility": artifacts / f"{OUTPUT_PREFIX}_calibration_eligibility.csv",
        "target_free_scores": artifacts / f"{OUTPUT_PREFIX}_target_free_scores.csv.gz",
        "target_free_selection": artifacts / f"{OUTPUT_PREFIX}_target_free_selection.csv",
        "auc_metrics": artifacts / f"{OUTPUT_PREFIX}_auc_metrics.csv",
        "rmse_metrics": artifacts / f"{OUTPUT_PREFIX}_rmse_metrics.csv",
        "subgroup_metrics": artifacts / f"{OUTPUT_PREFIX}_subgroup_metrics.csv",
        "by_well": artifacts / f"{OUTPUT_PREFIX}_by_well.csv",
        "sha_manifest": artifacts / f"{OUTPUT_PREFIX}_sha_manifest.csv",
        "summary": artifacts / f"{OUTPUT_PREFIX}_summary.json",
    }
    contract = {
        "experiment": EXPERIMENT_NAME,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate_order": CANDIDATES,
        "safe_candidate": SAFE_CANDIDATE,
        "horizons_rows": HORIZONS,
        "primary_horizon_rows": PRIMARY_HORIZON,
        "target_free_scoring": True,
        "selected_prediction_persisted": False,
        "aggregate_prediction_content_sha256": aggregate_summary["prediction_content_sha256"],
    }
    write_json(paths["contract"], contract)
    write_csv(paths["fold_manifest"], fold_manifest)
    write_csv(paths["calibration_eligibility"], eligibility)
    write_csv(paths["target_free_scores"], scores, gzip_output=True)
    write_csv(paths["target_free_selection"], selection)
    frozen_score_sha = dataframe_content_sha256(scores)
    frozen_selection_sha = dataframe_content_sha256(selection)
    frozen_score_schema_sha = dataframe_schema_sha256(scores)
    frozen_selection_schema_sha = dataframe_schema_sha256(selection)
    contract["target_free_score_content_sha256"] = frozen_score_sha
    contract["target_free_selection_content_sha256"] = frozen_selection_sha
    contract["target_free_score_schema_sha256"] = frozen_score_schema_sha
    contract["target_free_selection_schema_sha256"] = frozen_selection_schema_sha
    contract["fold_manifest_content_sha256"] = dataframe_content_sha256(fold_manifest)
    contract["fold_manifest_schema_sha256"] = dataframe_schema_sha256(fold_manifest)
    write_json(paths["contract"], contract)

    scored, auc_metrics, rmse_metrics, subgroup_metrics, extra = attach_truth_and_compute_metrics(
        bank, scores, selection, data_dir, folds, config
    )
    if (
        dataframe_content_sha256(scores) != frozen_score_sha
        or dataframe_content_sha256(selection) != frozen_selection_sha
    ):
        raise RuntimeError("target-free score or selection changed after truth attachment")
    by_well = extra["by_well"]
    manifest.append(extra["input_record"])
    write_csv(paths["auc_metrics"], auc_metrics)
    write_csv(paths["rmse_metrics"], rmse_metrics)
    write_csv(paths["subgroup_metrics"], subgroup_metrics)
    write_csv(paths["by_well"], by_well)
    write_csv(paths["input_manifest"], pd.DataFrame(manifest))
    guards = evaluate_success_guards(
        selection, bank, auc_metrics, rmse_metrics, subgroup_metrics, config
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_pass" if guards["passed"] else "completed_fail_closed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(bank),
        "wells": bank["well"].nunique(),
        "target_free_score_rows": len(scores),
        "target_free_selection_rows": len(selection),
        "truth_joined_score_rows_in_memory_only": len(scored),
        "target_free_score_content_sha256": frozen_score_sha,
        "target_free_selection_content_sha256": frozen_selection_sha,
        "target_free_score_schema_sha256": frozen_score_schema_sha,
        "target_free_selection_schema_sha256": frozen_selection_schema_sha,
        "success_guards": guards,
        "runtime_seconds": time.time() - started,
        "selected_prediction_persisted": False,
        "inference_enabled": False,
        "submission_created": False,
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    write_json(paths["summary"], summary)
    sha_rows = []
    for role, path in paths.items():
        if role == "sha_manifest" or not path.exists():
            continue
        sha_rows.append(
            {
                "role": role,
                "path": str(path),
                "raw_sha256": sha256_path(path),
                "decompressed_sha256": sha256_gzip_decompressed(path)
                if path.suffix == ".gz"
                else None,
            }
        )
    write_csv(paths["sha_manifest"], pd.DataFrame(sha_rows))
    metrics_path = artifacts.parent / "metrics.json"
    write_json(
        metrics_path,
        {
            "experiment": EXPERIMENT_NAME,
            "status": summary["status"],
            "metric": get_nested(config, "validation.metric"),
            "cv": None,
            "public_lb": None,
            "private_lb": None,
            "success_guards": guards,
            "summary": str(paths["summary"]),
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 8. Execution
#
# Import-only mode supports contract tests. A normal Kaggle notebook execution
# runs exactly one fixed CPU audit and never creates an inference artifact or a
# submission.

# %%
if os.environ.get("EXP292_IMPORT_ONLY", "0") != "1":
    RESULT = run_audit()

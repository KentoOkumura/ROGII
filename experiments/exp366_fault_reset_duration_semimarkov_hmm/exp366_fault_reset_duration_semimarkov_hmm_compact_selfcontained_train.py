# %% [markdown]
# # exp366 fault-reset duration semi-Markov HMM — Stage 0 train-side preflight
#
# This notebook implements only the frozen zero-HMM preflight. It discovers
# target-free trigger rows from raw GR change and the saved exp209 emission
# surprise, compares base plus twelve fixed reset-duration paths, freezes every
# trigger/path/score decision, and attaches suffix truth only for reporting.
# Stage 1 exact semi-Markov decoding remains disabled and unimplemented.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe runtime, configuration, path, and SHA helpers
# 3. Frozen scientific and execution contract
# 4. Saved exp209 and visible-prefix input preflight
# 5. Target-free trigger and fixed reset-branch generation
# 6. Reporting folds and pre-truth SHA freeze
# 7. Late truth and hidden-like attachment
# 8. Stage 0 metrics and promotion gates
# 9. Execution orchestration and generated artifacts
# 10. Setup and fail-closed execution selection

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import platform
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

try:
    from IPython import get_ipython
    from IPython.display import display
except ImportError:

    def get_ipython() -> Any:
        return None

    def display(value: Any) -> None:
        print(value)


EXPERIMENT_NAME = "exp366_fault_reset_duration_semimarkov_hmm"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
FORBIDDEN_PREFREEZE_HORIZONTAL_COLUMNS = {
    "TVT",
    "truth",
    "target",
    "error",
    "abs_error",
    "rmse",
}
BASE_BRANCH_ID = "base"

# %% [markdown]
# ## 2. Notebook-safe runtime, configuration, path, and SHA helpers

# %%
def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def locate_config() -> Path:
    relative = Path("experiments") / EXPERIMENT_NAME / "config.yaml"
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / relative,
        Path.cwd() / "config.yaml",
        Path.cwd() / relative,
        KAGGLE_WORKING_ROOT / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate exp366 config; checked={candidates}")


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or locate_config()
    value = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("exp366 config must be a YAML mapping")
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def mapping_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_gzip(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_content_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def dataframe_content_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = frame if columns is None else frame.loc[:, list(columns)]
    payload = selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def logical_dataframe_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in selected:
        digest.update(str(column).encode())
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


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def write_gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def inspect_artifact(path: Path) -> dict[str, Any]:
    report = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "raw_sha256": sha256_file(path),
    }
    if path.suffix == ".gz":
        report["decompressed_sha256"] = sha256_decompressed_gzip(path)
    return report


def _candidate_paths(filename: str, candidates: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in candidates:
        candidate = Path(str(raw))
        if candidate.name == filename:
            paths.append(candidate)
        else:
            paths.append(candidate / filename)
        if not candidate.is_absolute():
            paths.append(Path.cwd() / candidate / filename)
    return paths


def resolve_existing(
    filename: str,
    candidates: Iterable[str],
    patterns: Iterable[str] = (),
) -> Path:
    checked: list[str] = []
    for candidate in _candidate_paths(filename, candidates):
        checked.append(str(candidate))
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    search_roots = [Path.cwd()]
    if KAGGLE_INPUT_ROOT.exists():
        search_roots.append(KAGGLE_INPUT_ROOT)
    for root in search_roots:
        for pattern in [*patterns, f"**/{filename}"]:
            for candidate in sorted(root.glob(str(pattern))):
                checked.append(str(candidate))
                if (
                    candidate.exists()
                    and candidate.is_file()
                    and candidate.stat().st_size > 0
                ):
                    return candidate
    raise FileNotFoundError(
        f"Could not resolve required file {filename}; checked={checked[:40]}"
    )


def output_directory(config: Mapping[str, Any]) -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        root = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        root = (
            Path.cwd()
            / "experiments"
            / EXPERIMENT_NAME
            / "artifacts"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
    }


# %% [markdown]
# ## 3. Frozen scientific and execution contract

# %%
def fixed_branch_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    branch_config = get_nested(config, "validation.stage_0.branches") or {}
    jumps = sorted(
        (float(value) for value in branch_config["position_jumps_ft"]),
        key=lambda value: (abs(value), value > 0),
    )
    durations = sorted(int(value) for value in branch_config["horizons_rows"])
    specs: list[dict[str, Any]] = [
        {
            "branch_id": BASE_BRANCH_ID,
            "jump_ft": 0.0,
            "duration_rows": 0,
            "branch_order": 0,
        }
    ]
    for jump in jumps:
        sign = "p" if jump > 0 else "m"
        magnitude = str(abs(jump)).replace(".", "p")
        for duration in durations:
            specs.append(
                {
                    "branch_id": f"jump_{sign}{magnitude}_h{duration}",
                    "jump_ft": float(jump),
                    "duration_rows": int(duration),
                    "branch_order": len(specs),
                }
            )
    return specs


def stage0_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "implementation_scope": get_nested(config, "execution.implementation_scope"),
        "evaluation_horizon_rows": get_nested(
            config, "validation.stage_0.evaluation_horizon_rows"
        ),
        "trigger": get_nested(config, "validation.stage_0.trigger"),
        "branches": fixed_branch_specs(config),
        "branch_path_contract": get_nested(
            config, "validation.stage_0.branches.path_contract"
        ),
        "branch_score": get_nested(config, "validation.stage_0.branches.score"),
        "negative_control": get_nested(
            config, "validation.stage_0.negative_control"
        ),
        "bad_event_label": get_nested(
            config, "validation.stage_0.bad_event_label_after_freeze"
        ),
        "fixed_emission": get_nested(config, "model.fixed_emission"),
        "fixed_grid": get_nested(config, "model.fixed_grid"),
        "run_stage_1": get_nested(config, "execution.run_stage_1"),
        "run_inference": get_nested(config, "execution.run_inference"),
        "create_submission": get_nested(config, "execution.create_submission"),
        "stage_0_counts": get_nested(config, "execution.stage_0_counts"),
    }
    contract["contract_sha256"] = mapping_sha256(contract)
    return contract


def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    contract = stage0_contract(config)
    required = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "implementation_scope": "stage0_only",
        "evaluation_horizon_rows": 512,
        "negative_control": (
            "within_well_circular_shift_of_accepted_trigger_score_by_512_rows"
        ),
        "bad_event_label": "exp209_fixed_512_row_horizon_rmse_ge_10ft",
        "run_stage_1": False,
        "run_inference": False,
        "create_submission": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": contract.get(key)}
        for key, expected in required.items()
        if contract.get(key) != expected
    }
    expected_specs = [
        (BASE_BRANCH_ID, 0.0, 0),
        ("jump_m6p3_h128", -6.3, 128),
        ("jump_m6p3_h256", -6.3, 256),
        ("jump_m6p3_h512", -6.3, 512),
        ("jump_p6p3_h128", 6.3, 128),
        ("jump_p6p3_h256", 6.3, 256),
        ("jump_p6p3_h512", 6.3, 512),
        ("jump_m12p6_h128", -12.6, 128),
        ("jump_m12p6_h256", -12.6, 256),
        ("jump_m12p6_h512", -12.6, 512),
        ("jump_p12p6_h128", 12.6, 128),
        ("jump_p12p6_h256", 12.6, 256),
        ("jump_p12p6_h512", 12.6, 512),
    ]
    actual_specs = [
        (item["branch_id"], item["jump_ft"], item["duration_rows"])
        for item in contract["branches"]
    ]
    if actual_specs != expected_specs:
        mismatches["branches"] = {
            "expected": expected_specs,
            "actual": actual_specs,
        }
    trigger = contract["trigger"] or {}
    expected_trigger = {
        "gr_change_transform": "absolute_first_difference_then_median_1p4826mad_z",
        "gr_change_robust_z_quantile_from_known_prefix": 0.995,
        "emission_surprise": "exp209_gaussian_negative_log_emission",
        "exp209_emission_surprise_quantile_from_known_prefix": 0.995,
        "combine": "logical_and",
        "refractory_rows": 512,
        "negative_control_shift_rows": 512,
    }
    for key, expected in expected_trigger.items():
        if trigger.get(key) != expected:
            mismatches[f"trigger.{key}"] = {
                "expected": expected,
                "actual": trigger.get(key),
            }
    emission = contract["fixed_emission"] or {}
    expected_emission = {
        "family": "gaussian",
        "squared_z_clip": 600.0,
        "likelihood_weight": 1.0,
        "sigma_mode": "known_prefix_zero_fill_population_std",
        "sigma_clip": [10.0, 60.0],
        "missing_gr_policy": (
            "linear_interpolate_both_directions_then_typewell_mean"
        ),
    }
    for key, expected in expected_emission.items():
        if emission.get(key) != expected:
            mismatches[f"fixed_emission.{key}"] = {
                "expected": expected,
                "actual": emission.get(key),
            }
    if contract["fixed_grid"] != {
        "position_grid_step_ft": 0.35,
        "band_pad_ft": 100.0,
        "typewell_outer_pad_ft": 40.0,
    }:
        mismatches["fixed_grid"] = {
            "expected": {
                "position_grid_step_ft": 0.35,
                "band_pad_ft": 100.0,
                "typewell_outer_pad_ft": 40.0,
            },
            "actual": contract["fixed_grid"],
        }
    expected_counts = {
        "diagnostic_variants": 1,
        "fixed_branches": 13,
        "reporting_folds": 5,
        "semimarkov_hmm_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": 0,
    }
    if contract["stage_0_counts"] != expected_counts:
        mismatches["stage_0_counts"] = {
            "expected": expected_counts,
            "actual": contract["stage_0_counts"],
        }
    forbidden = set(get_nested(config, "model.forbidden") or [])
    required_forbidden = {
        "horizontal_gr_atlas",
        "true_error_or_oracle_trigger",
        "rate_prediction_from_prefix_or_geometry",
        "global_or_overlapping_branch_spawn",
        "unlimited_beam",
        "jump_horizon_margin_or_trigger_grid",
        "softmax_branch_average",
        "blend_or_selector",
        "parent_control_rerun",
    }
    if forbidden != required_forbidden:
        mismatches["model.forbidden"] = {
            "expected": sorted(required_forbidden),
            "actual": sorted(forbidden),
        }
    if mismatches:
        raise ValueError(f"Frozen exp366 Stage 0 contract mismatch: {mismatches}")
    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.kaggle_execution_approved"))
        and bool(get_nested(config, "execution.run_stage_0"))
    ):
        raise PermissionError("exp366 Kaggle Stage 0 run is not approved")
    return contract


@dataclass
class TruthAccessLedger:
    frozen: bool = False
    truth_rows_before_freeze: int = 0
    hidden_role_rows_before_freeze: int = 0
    truth_rows_after_freeze: int = 0
    hidden_role_rows_after_freeze: int = 0

    def guard_prefreeze_columns(
        self,
        columns: Iterable[str],
        rows: int,
        label: str,
    ) -> None:
        overlap = FORBIDDEN_PREFREEZE_HORIZONTAL_COLUMNS.intersection(columns)
        if overlap:
            if not self.frozen:
                self.truth_rows_before_freeze += int(rows)
            raise ValueError(
                f"{label}: forbidden pre-freeze columns requested: {sorted(overlap)}"
            )

    def freeze(self) -> None:
        if self.truth_rows_before_freeze or self.hidden_role_rows_before_freeze:
            raise RuntimeError("truth or hidden-like roles were accessed before SHA freeze")
        self.frozen = True

    def record_truth_late(self, rows: int) -> None:
        if not self.frozen:
            self.truth_rows_before_freeze += int(rows)
            raise RuntimeError("suffix truth cannot be read before SHA freeze")
        self.truth_rows_after_freeze += int(rows)

    def record_hidden_late(self, rows: int) -> None:
        if not self.frozen:
            self.hidden_role_rows_before_freeze += int(rows)
            raise RuntimeError("hidden-like roles cannot be read before SHA freeze")
        self.hidden_role_rows_after_freeze += int(rows)


# %% [markdown]
# ## 4. Saved exp209 and visible-prefix input preflight

# %%
def parse_row_index(ids: pd.Series) -> np.ndarray:
    values = ids.astype(str).str.rsplit("_", n=1).str[-1]
    return pd.to_numeric(values, errors="raise").to_numpy(np.int64)


def inspect_gzip_csv(path: Path) -> dict[str, Any]:
    rows = 0
    with gzip.open(path, "rt") as stream:
        header = stream.readline().rstrip("\n").split(",")
        for _ in stream:
            rows += 1
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": sha256_decompressed_gzip(path),
        "data_rows": rows,
        "columns": header,
    }


def load_saved_exp209_target_free(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.saved_exp209") or {}
    path = resolve_existing(
        str(spec["filename"]),
        [str(value) for value in spec.get("candidates", [])],
        [str(value) for value in spec.get("patterns", [])],
    )
    report = inspect_gzip_csv(path)
    if report["decompressed_sha256"] != str(spec["expected_decompressed_sha256"]):
        raise ValueError("saved exp209 decompressed SHA mismatch")
    safe_columns = [str(value) for value in spec["safe_columns"]]
    if set(safe_columns) != {"id", "well", "hmm_mean_tvt", "hmm_prefix_sigma"}:
        raise ValueError("exp366 saved exp209 allowlist changed")
    frame = pd.read_csv(path, usecols=safe_columns, dtype={"id": str, "well": str})
    frame = frame.rename(columns={"well": "well_id"})
    frame["row_idx"] = parse_row_index(frame["id"])
    for column in ("hmm_mean_tvt", "hmm_prefix_sigma"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("saved exp209 keys are duplicated")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(frame) != expected_rows or frame["well_id"].nunique() != expected_wells:
        raise ValueError("saved exp209 row/well coverage mismatch")
    if not np.isfinite(
        frame[["hmm_mean_tvt", "hmm_prefix_sigma"]].to_numpy(np.float64)
    ).all():
        raise ValueError("saved exp209 target-free columns contain non-finite values")
    report["loaded_columns"] = safe_columns
    report["loaded_rows"] = int(len(frame))
    report["loaded_wells"] = int(frame["well_id"].nunique())
    report["truth_or_error_columns_loaded"] = 0
    return frame, report


def discover_raw_train(config: Mapping[str, Any]) -> tuple[Path, list[str], dict[str, Any]]:
    configured = Path(str(get_nested(config, "data.train_dir")))
    candidates = [
        Path.cwd() / configured,
        configured,
        KAGGLE_INPUT_ROOT
        / "competitions"
        / "rogii-wellbore-geology-prediction"
        / "train",
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            path.parent
            for path in sorted(
                KAGGLE_INPUT_ROOT.glob("**/*__horizontal_well.csv")
            )[:20]
        )
    train_dir = next(
        (
            path
            for path in candidates
            if path.exists() and any(path.glob("*__horizontal_well.csv"))
        ),
        None,
    )
    if train_dir is None:
        raise FileNotFoundError(f"Could not locate raw train directory: {candidates}")
    rows: list[dict[str, str]] = []
    for horizontal in sorted(train_dir.glob("*__horizontal_well.csv")):
        well = horizontal.name.removesuffix("__horizontal_well.csv")
        typewell = train_dir / f"{well}__typewell.csv"
        if not typewell.exists():
            raise FileNotFoundError(typewell)
        rows.append(
            {
                "well_id": well,
                "horizontal_raw_sha256": sha256_file(horizontal),
                "typewell_raw_sha256": sha256_file(typewell),
            }
        )
    identity = pd.DataFrame(rows).sort_values("well_id", kind="mergesort")
    identity_sha = logical_dataframe_sha256(
        identity,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if len(identity) != expected_wells:
        raise ValueError(
            f"raw train well count mismatch: {len(identity)} != {expected_wells}"
        )
    if identity_sha != expected_sha:
        raise ValueError(
            f"raw train well identity mismatch: expected={expected_sha} "
            f"actual={identity_sha}"
        )
    return train_dir, identity["well_id"].tolist(), {
        "path": str(train_dir),
        "wells": int(len(identity)),
        "content_sha256": identity_sha,
    }


def load_prefreeze_well(
    well_id: str,
    train_dir: Path,
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    horizontal_path = train_dir / f"{well_id}__horizontal_well.csv"
    typewell_path = train_dir / f"{well_id}__typewell.csv"
    horizontal_columns = ["MD", "Z", "GR", "TVT_input"]
    ledger.guard_prefreeze_columns(horizontal_columns, 0, f"{well_id} horizontal")
    horizontal = pd.read_csv(horizontal_path, usecols=horizontal_columns)
    ledger.guard_prefreeze_columns(
        horizontal.columns, len(horizontal), f"{well_id} horizontal"
    )
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    return horizontal, typewell, horizontal_path, typewell_path


def prepare_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    frame = typewell.copy()
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.sort_values("TVT", kind="mergesort")
    frame["GR"] = frame["GR"].ffill().bfill()
    valid = frame["TVT"].notna() & frame["GR"].notna()
    tvt = frame.loc[valid, "TVT"].to_numpy(np.float64)
    gr = frame.loc[valid, "GR"].to_numpy(np.float64)
    if len(tvt) < 2 or np.any(np.diff(tvt) < 0):
        raise ValueError("typewell TVT/GR contract is invalid")
    return tvt, gr


def exp209_prefix_sigma(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    clip: Sequence[float],
) -> float:
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
        np.float64
    )
    known = np.isfinite(tvt_input)
    known_index = np.flatnonzero(known)
    if len(known_index) == 0 or not np.array_equal(
        known_index, np.arange(len(known_index))
    ):
        raise ValueError("one contiguous visible prefix is required")
    raw_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    expected = np.interp(tvt_input[known], typewell_tvt, typewell_gr)
    residual = np.where(np.isfinite(raw_gr[known]), raw_gr[known], 0.0) - expected
    sigma = float(np.nanstd(residual, ddof=0))
    if not math.isfinite(sigma):
        sigma = float(clip[1])
    return float(np.clip(sigma, float(clip[0]), float(clip[1])))


def exp209_position_grid_bounds(
    last_known_tvt: float,
    typewell_tvt: np.ndarray,
    *,
    band_pad_ft: float,
    typewell_outer_pad_ft: float,
    step_ft: float,
) -> tuple[float, float]:
    grid_min = max(
        float(np.min(typewell_tvt)) - float(typewell_outer_pad_ft),
        float(last_known_tvt) - float(band_pad_ft),
    )
    grid_limit = min(
        float(np.max(typewell_tvt)) + float(typewell_outer_pad_ft),
        float(last_known_tvt) + float(band_pad_ft),
    )
    grid = np.arange(grid_min, grid_limit + float(step_ft), float(step_ft))
    if len(grid) < 2:
        raise ValueError("exp209 position grid contains fewer than two cells")
    return float(grid[0]), float(grid[-1])


def gaussian_log_emission(
    observed_gr: np.ndarray,
    path_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    sigma: float,
    squared_z_clip: float = 600.0,
) -> np.ndarray:
    expected_gr = np.interp(
        np.asarray(path_tvt, np.float64), typewell_tvt, typewell_gr
    )
    zscore = (np.asarray(observed_gr, np.float64) - expected_gr) / float(sigma)
    return -0.5 * np.minimum(np.square(zscore), float(squared_z_clip))


def robust_location_scale(
    values: np.ndarray,
    floor: float = 1.0e-6,
) -> tuple[float, float]:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 4:
        raise ValueError("at least four finite prefix values are required")
    location = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - location)))
    return location, max(scale, float(floor))


def robust_zscore(
    values: np.ndarray,
    location: float,
    scale: float,
) -> np.ndarray:
    return (np.asarray(values, np.float64) - float(location)) / float(scale)


# %% [markdown]
# ## 5. Target-free trigger and fixed reset-branch generation

# %%
def apply_refractory(
    candidate_mask: np.ndarray,
    refractory_rows: int,
) -> np.ndarray:
    candidates = np.asarray(candidate_mask, dtype=bool)
    accepted = np.zeros(len(candidates), dtype=bool)
    next_allowed = 0
    for index in np.flatnonzero(candidates):
        if int(index) < next_allowed:
            continue
        accepted[int(index)] = True
        next_allowed = int(index) + int(refractory_rows)
    return accepted


def circular_shift_trigger_score(
    accepted_score: np.ndarray,
    shift_rows: int,
) -> np.ndarray:
    values = np.asarray(accepted_score, np.float64)
    if len(values) <= 1:
        return np.zeros_like(values)
    shift = int(shift_rows) % len(values)
    if shift == 0:
        shift = 1
    return np.roll(values, shift)


def reset_branch_path(
    base_path: np.ndarray,
    jump_ft: float,
    duration_rows: int,
    *,
    grid_min_tvt: float | None = None,
    grid_max_tvt: float | None = None,
) -> np.ndarray:
    path = np.asarray(base_path, np.float64).copy()
    active = min(max(int(duration_rows), 0), len(path))
    if active:
        path[:active] += float(jump_ft)
        if grid_min_tvt is not None and grid_max_tvt is not None:
            path[:active] = np.clip(
                path[:active], float(grid_min_tvt), float(grid_max_tvt)
            )
    return path


def ranked_branch_ids(
    score_by_branch: Mapping[str, float],
    specs: Sequence[Mapping[str, Any]],
) -> list[str]:
    priority = {
        str(spec["branch_id"]): int(spec["branch_order"]) for spec in specs
    }
    return sorted(
        (str(branch_id) for branch_id in score_by_branch),
        key=lambda branch_id: (
            -float(score_by_branch[branch_id]),
            priority[branch_id],
        ),
    )


def build_prefreeze_rows_for_well(
    well_id: str,
    saved_well: pd.DataFrame,
    train_dir: Path,
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    horizontal, typewell, horizontal_path, typewell_path = load_prefreeze_well(
        well_id, train_dir, ledger
    )
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
        np.float64
    )
    known_mask = np.isfinite(tvt_input)
    suffix_index = np.flatnonzero(~known_mask)
    known_index = np.flatnonzero(known_mask)
    if (
        len(known_index) < 4
        or len(suffix_index) == 0
        or not np.array_equal(known_index, np.arange(len(known_index)))
    ):
        raise ValueError(f"well={well_id} lacks one valid visible prefix and suffix")
    saved = saved_well.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(saved["row_idx"].to_numpy(np.int64), suffix_index):
        raise ValueError(f"well={well_id} saved exp209 row identity mismatch")
    horizon = int(get_nested(config, "validation.stage_0.evaluation_horizon_rows"))
    eligible_count = max(0, len(suffix_index) - horizon + 1)
    typewell_tvt, typewell_gr = prepare_typewell(typewell)
    emission = get_nested(config, "model.fixed_emission") or {}
    fixed_grid = get_nested(config, "model.fixed_grid") or {}
    sigma = exp209_prefix_sigma(
        horizontal,
        typewell_tvt,
        typewell_gr,
        emission["sigma_clip"],
    )
    saved_sigma = saved["hmm_prefix_sigma"].to_numpy(np.float64)
    sigma_max_abs_diff = float(np.max(np.abs(saved_sigma - sigma)))
    if sigma_max_abs_diff > 1.0e-5:
        raise ValueError(f"well={well_id} exp209 prefix sigma parity mismatch")
    grid_min_tvt, grid_max_tvt = exp209_position_grid_bounds(
        float(tvt_input[known_index[-1]]),
        typewell_tvt,
        band_pad_ft=float(fixed_grid["band_pad_ft"]),
        typewell_outer_pad_ft=float(fixed_grid["typewell_outer_pad_ft"]),
        step_ft=float(fixed_grid["position_grid_step_ft"]),
    )
    raw_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    filled_gr = (
        pd.Series(raw_gr)
        .interpolate(limit_direction="both")
        .fillna(float(np.mean(typewell_gr)))
        .to_numpy(np.float64)
    )
    absolute_change = np.full(len(raw_gr), np.nan, dtype=np.float64)
    adjacent = np.isfinite(raw_gr[1:]) & np.isfinite(raw_gr[:-1])
    absolute_change[1:] = np.where(
        adjacent,
        np.abs(raw_gr[1:] - raw_gr[:-1]),
        np.nan,
    )
    change_location, change_scale = robust_location_scale(
        absolute_change[known_mask]
    )
    change_z = robust_zscore(absolute_change, change_location, change_scale)
    trigger = get_nested(config, "validation.stage_0.trigger") or {}
    change_quantile = float(
        trigger["gr_change_robust_z_quantile_from_known_prefix"]
    )
    change_threshold = float(
        np.quantile(change_z[known_mask & np.isfinite(change_z)], change_quantile)
    )
    known_surprise = -gaussian_log_emission(
        filled_gr[known_mask],
        tvt_input[known_mask],
        typewell_tvt,
        typewell_gr,
        sigma,
        float(emission["squared_z_clip"]),
    )
    surprise_threshold = float(
        np.quantile(
            known_surprise,
            float(trigger["exp209_emission_surprise_quantile_from_known_prefix"]),
        )
    )
    suffix_change_z = change_z[suffix_index]
    suffix_raw_observed = np.isfinite(raw_gr[suffix_index])
    suffix_surprise = -gaussian_log_emission(
        filled_gr[suffix_index],
        saved["hmm_mean_tvt"].to_numpy(np.float64),
        typewell_tvt,
        typewell_gr,
        sigma,
        float(emission["squared_z_clip"]),
    )
    change_denominator = max(change_threshold, 1.0e-12)
    surprise_denominator = max(surprise_threshold, 1.0e-12)
    trigger_strength = np.minimum(
        suffix_change_z / change_denominator,
        suffix_surprise / surprise_denominator,
    )
    trigger_strength[~np.isfinite(trigger_strength)] = 0.0
    trigger_strength[~suffix_raw_observed] = 0.0
    eligible = np.arange(len(suffix_index)) < eligible_count
    candidate = eligible & (trigger_strength >= 1.0)
    accepted = apply_refractory(candidate, int(trigger["refractory_rows"]))
    accepted_score = np.where(accepted, trigger_strength, 0.0)
    circular_score = circular_shift_trigger_score(
        accepted_score[:eligible_count],
        int(trigger["negative_control_shift_rows"]),
    )
    md = pd.to_numeric(horizontal["MD"], errors="raise").to_numpy(np.float64)
    last_known_md = float(md[known_index[-1]])
    trigger_frame = pd.DataFrame(
        {
            "id": saved["id"].iloc[:eligible_count].astype(str).to_numpy(),
            "well_id": well_id,
            "row_idx": suffix_index[:eligible_count].astype(np.int32),
            "suffix_offset": np.arange(eligible_count, dtype=np.int32),
            "md_since": (md[suffix_index[:eligible_count]] - last_known_md).astype(
                np.float32
            ),
            "gr_change_robust_z": suffix_change_z[:eligible_count].astype(
                np.float32
            ),
            "exp209_emission_surprise": suffix_surprise[:eligible_count].astype(
                np.float32
            ),
            "trigger_score": accepted_score[:eligible_count].astype(np.float32),
            "accepted_trigger": accepted[:eligible_count],
            "circular_trigger_score": circular_score.astype(np.float32),
        }
    )
    specs = fixed_branch_specs(config)
    base_all = saved["hmm_mean_tvt"].to_numpy(np.float64)
    suffix_observed_gr = filled_gr[suffix_index]
    branch_records: list[dict[str, Any]] = []
    for suffix_offset in np.flatnonzero(accepted[:eligible_count]):
        start = int(suffix_offset)
        stop = start + horizon
        base_path = base_all[start:stop]
        observed = suffix_observed_gr[start:stop]
        score_by_branch: dict[str, float] = {}
        rows_for_event: list[dict[str, Any]] = []
        for spec in specs:
            path = reset_branch_path(
                base_path,
                float(spec["jump_ft"]),
                int(spec["duration_rows"]),
                grid_min_tvt=grid_min_tvt,
                grid_max_tvt=grid_max_tvt,
            )
            score = float(
                gaussian_log_emission(
                    observed,
                    path,
                    typewell_tvt,
                    typewell_gr,
                    sigma,
                    float(emission["squared_z_clip"]),
                ).sum()
            )
            branch_id = str(spec["branch_id"])
            score_by_branch[branch_id] = score
            rows_for_event.append(
                {
                    "event_id": f"{well_id}_{int(suffix_index[start])}",
                    "well_id": well_id,
                    "trigger_row_idx": int(suffix_index[start]),
                    "trigger_suffix_offset": start,
                    "branch_id": branch_id,
                    "branch_order": int(spec["branch_order"]),
                    "jump_ft": float(spec["jump_ft"]),
                    "duration_rows": int(spec["duration_rows"]),
                    "evaluation_horizon_rows": horizon,
                    "grid_min_tvt": grid_min_tvt,
                    "grid_max_tvt": grid_max_tvt,
                    "cumulative_gr_log_emission": score,
                    "branch_path_content_sha256": array_content_sha256(path),
                }
            )
        ranking = ranked_branch_ids(score_by_branch, specs)
        rank_by_branch = {
            branch_id: rank for rank, branch_id in enumerate(ranking, start=1)
        }
        selected = ranking[0]
        base_score = score_by_branch[BASE_BRANCH_ID]
        for row in rows_for_event:
            row["gr_score_margin_vs_base"] = (
                float(row["cumulative_gr_log_emission"]) - base_score
            )
            row["evidence_rank"] = rank_by_branch[str(row["branch_id"])]
            row["selected_branch"] = str(row["branch_id"]) == selected
            branch_records.append(row)
    branch_frame = pd.DataFrame(branch_records)
    manifest = {
        "well_id": well_id,
        "status": "ok",
        "prefix_rows": int(len(known_index)),
        "suffix_rows": int(len(suffix_index)),
        "eligible_trigger_rows": int(eligible_count),
        "candidate_trigger_rows": int(candidate.sum()),
        "accepted_trigger_rows": int(accepted.sum()),
        "trigger_row_fraction": (
            float(accepted.sum() / eligible_count) if eligible_count else 0.0
        ),
        "gr_change_location": change_location,
        "gr_change_scale": change_scale,
        "gr_change_z_q995": change_threshold,
        "emission_surprise_q995": surprise_threshold,
        "exp209_prefix_sigma": sigma,
        "saved_sigma_max_abs_diff": sigma_max_abs_diff,
        "grid_min_tvt": grid_min_tvt,
        "grid_max_tvt": grid_max_tvt,
        "horizontal_raw_sha256": sha256_file(horizontal_path),
        "typewell_raw_sha256": sha256_file(typewell_path),
    }
    return trigger_frame, branch_frame, manifest


# %% [markdown]
# ## 6. Reporting folds and pre-truth SHA freeze

# %%
def assign_group_folds(
    trigger_ledger: pd.DataFrame,
    requested_splits: int,
) -> pd.DataFrame:
    frame = trigger_ledger.copy()
    wells = int(frame["well_id"].nunique())
    splits = min(int(requested_splits), wells)
    if splits < 2:
        frame["fold"] = 0
        return frame
    groups = frame["well_id"].astype(str).to_numpy()
    fold_values = np.full(len(frame), -1, dtype=np.int16)
    splitter = GroupKFold(n_splits=splits)
    dummy = np.zeros((len(frame), 1), dtype=np.uint8)
    for fold, (_, valid_index) in enumerate(splitter.split(dummy, groups=groups)):
        fold_values[valid_index] = int(fold)
    if np.any(fold_values < 0):
        raise RuntimeError("GroupKFold left trigger rows unassigned")
    frame["fold"] = fold_values
    return frame


def attach_event_folds(
    branch_ledger: pd.DataFrame,
    trigger_ledger: pd.DataFrame,
) -> pd.DataFrame:
    if branch_ledger.empty:
        return branch_ledger.assign(fold=pd.Series(dtype=np.int16))
    event_folds = trigger_ledger.loc[
        trigger_ledger["accepted_trigger"], ["well_id", "row_idx", "fold"]
    ].rename(columns={"row_idx": "trigger_row_idx"})
    return branch_ledger.merge(
        event_folds,
        on=["well_id", "trigger_row_idx"],
        how="left",
        validate="many_to_one",
    )


def freeze_prefreeze_artifacts(
    output_dir: Path,
    prefix: str,
    input_manifest: pd.DataFrame,
    trigger_ledger: pd.DataFrame,
    branch_ledger: pd.DataFrame,
    contract: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> dict[str, Any]:
    paths = {
        "input_manifest": output_dir / f"{prefix}_input_manifest.csv",
        "trigger_ledger": output_dir / f"{prefix}_trigger_ledger.csv.gz",
        "branch_ledger": output_dir / f"{prefix}_branch_ledger.csv.gz",
        "freeze_manifest": output_dir / f"{prefix}_freeze_manifest.json",
    }
    write_csv(paths["input_manifest"], input_manifest)
    write_gzip_csv(paths["trigger_ledger"], trigger_ledger)
    write_gzip_csv(paths["branch_ledger"], branch_ledger)
    artifact_reports = {
        key: inspect_artifact(path)
        for key, path in paths.items()
        if key != "freeze_manifest"
    }
    freeze_manifest = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_0",
        "contract_sha256": contract["contract_sha256"],
        "truth_rows_before_freeze": ledger.truth_rows_before_freeze,
        "hidden_role_rows_before_freeze": ledger.hidden_role_rows_before_freeze,
        "trigger_rows": int(len(trigger_ledger)),
        "accepted_triggers": int(trigger_ledger["accepted_trigger"].sum()),
        "branch_rows": int(len(branch_ledger)),
        "reporting_folds": sorted(
            int(value) for value in trigger_ledger["fold"].unique()
        ),
        "artifacts": artifact_reports,
    }
    write_json(paths["freeze_manifest"], freeze_manifest)
    reread = json.loads(paths["freeze_manifest"].read_text())
    for key, report in reread["artifacts"].items():
        path = paths[key]
        if sha256_file(path) != report["raw_sha256"]:
            raise RuntimeError(f"pre-truth raw SHA readback failed for {key}")
        if path.suffix == ".gz" and (
            sha256_decompressed_gzip(path) != report["decompressed_sha256"]
        ):
            raise RuntimeError(f"pre-truth decompressed SHA readback failed for {key}")
    ledger.freeze()
    return freeze_manifest


# %% [markdown]
# ## 7. Late truth and hidden-like attachment

# %%
def forward_window_mse(squared_error: np.ndarray, horizon: int) -> np.ndarray:
    values = np.asarray(squared_error, np.float64)
    if len(values) < int(horizon):
        return np.empty(0, dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    return (cumulative[horizon:] - cumulative[:-horizon]) / float(horizon)


def load_hidden_like_late(
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like") or {}
    path = resolve_existing(
        str(spec["filename"]),
        [str(value) for value in spec.get("candidates", [])],
        [str(value) for value in spec.get("patterns", [])],
    )
    actual_sha = sha256_file(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    role_columns = [str(value) for value in spec["role_columns"].values()]
    frame = pd.read_csv(path, usecols=["well_id", *role_columns], dtype={"well_id": str})
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment has duplicate wells")
    ledger.record_hidden_late(len(frame))
    renamed = frame.rename(
        columns={
            role_column: scope
            for scope, role_column in spec["role_columns"].items()
        }
    )
    return renamed, {
        "path": str(path),
        "raw_sha256": actual_sha,
        "rows": int(len(frame)),
    }


def build_late_readouts(
    train_dir: Path,
    saved_exp209: pd.DataFrame,
    trigger_ledger: pd.DataFrame,
    branch_ledger: pd.DataFrame,
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not ledger.frozen:
        raise RuntimeError("late readout requires a completed pre-truth SHA freeze")
    horizon = int(get_nested(config, "validation.stage_0.evaluation_horizon_rows"))
    specs = fixed_branch_specs(config)
    order = {
        str(spec["branch_id"]): int(spec["branch_order"]) for spec in specs
    }
    trigger_parts: list[pd.DataFrame] = []
    event_records: list[dict[str, Any]] = []
    for well_id, well_trigger in trigger_ledger.groupby("well_id", sort=True):
        horizontal_path = train_dir / f"{well_id}__horizontal_well.csv"
        truth_frame = pd.read_csv(horizontal_path, usecols=["TVT"])
        ledger.record_truth_late(len(truth_frame))
        truth = pd.to_numeric(truth_frame["TVT"], errors="coerce").to_numpy(
            np.float64
        )
        saved = (
            saved_exp209.loc[saved_exp209["well_id"].eq(well_id)]
            .sort_values("row_idx", kind="mergesort")
            .reset_index(drop=True)
        )
        row_idx = saved["row_idx"].to_numpy(np.int64)
        base = saved["hmm_mean_tvt"].to_numpy(np.float64)
        suffix_truth = truth[row_idx]
        if not np.isfinite(suffix_truth).all():
            raise ValueError(f"well={well_id} suffix truth contains non-finite values")
        base_mse = forward_window_mse(np.square(base - suffix_truth), horizon)
        ordered_trigger = well_trigger.sort_values(
            "suffix_offset", kind="mergesort"
        ).copy()
        if len(ordered_trigger) != len(base_mse):
            raise ValueError(f"well={well_id} eligible horizon row mismatch")
        ordered_trigger["base_horizon_mse"] = base_mse
        ordered_trigger["base_horizon_rmse"] = np.sqrt(base_mse)
        ordered_trigger["bad_event"] = (
            ordered_trigger["base_horizon_rmse"] >= 10.0
        )
        trigger_parts.append(ordered_trigger)
        accepted = ordered_trigger.loc[ordered_trigger["accepted_trigger"]]
        for trigger_row in accepted.itertuples(index=False):
            start = int(trigger_row.suffix_offset)
            stop = start + horizon
            base_path = base[start:stop]
            truth_path = suffix_truth[start:stop]
            branches = branch_ledger.loc[
                branch_ledger["event_id"].eq(
                    f"{well_id}_{int(trigger_row.row_idx)}"
                )
            ].copy()
            if len(branches) != len(specs):
                raise ValueError(f"well={well_id} event branch coverage mismatch")
            mse_by_branch: dict[str, float] = {}
            for branch in branches.itertuples(index=False):
                candidate = reset_branch_path(
                    base_path,
                    float(branch.jump_ft),
                    int(branch.duration_rows),
                    grid_min_tvt=float(branch.grid_min_tvt),
                    grid_max_tvt=float(branch.grid_max_tvt),
                )
                if array_content_sha256(candidate) != str(
                    branch.branch_path_content_sha256
                ):
                    raise RuntimeError("post-freeze branch path SHA reconstruction mismatch")
                mse_by_branch[str(branch.branch_id)] = float(
                    np.mean(np.square(candidate - truth_path))
                )
            oracle = min(
                mse_by_branch,
                key=lambda branch_id: (mse_by_branch[branch_id], order[branch_id]),
            )
            evidence_rank = {
                str(row.branch_id): int(row.evidence_rank)
                for row in branches.itertuples(index=False)
            }
            selected = str(
                branches.loc[branches["selected_branch"], "branch_id"].iloc[0]
            )
            alternative_within10 = any(
                branch_id != BASE_BRANCH_ID and math.sqrt(mse) <= 10.0
                for branch_id, mse in mse_by_branch.items()
            )
            event_records.append(
                {
                    "event_id": f"{well_id}_{int(trigger_row.row_idx)}",
                    "well_id": well_id,
                    "trigger_row_idx": int(trigger_row.row_idx),
                    "trigger_suffix_offset": start,
                    "fold": int(trigger_row.fold),
                    "md_since": float(trigger_row.md_since),
                    "selected_branch_id": selected,
                    "oracle_branch_id": oracle,
                    "base_mse": mse_by_branch[BASE_BRANCH_ID],
                    "selected_mse": mse_by_branch[selected],
                    "oracle_mse": mse_by_branch[oracle],
                    "base_rmse": math.sqrt(mse_by_branch[BASE_BRANCH_ID]),
                    "selected_rmse": math.sqrt(mse_by_branch[selected]),
                    "oracle_rmse": math.sqrt(mse_by_branch[oracle]),
                    "alternative_within10": alternative_within10,
                    "evidence_reciprocal_rank": 1.0 / evidence_rank[oracle],
                    "base_first_reciprocal_rank": 1.0 / (order[oracle] + 1),
                    "selected_is_oracle": selected == oracle,
                }
            )
    trigger_readout = (
        pd.concat(trigger_parts, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    event_readout = pd.DataFrame(event_records)
    if event_readout.empty:
        event_readout = pd.DataFrame(
            columns=[
                "event_id",
                "well_id",
                "trigger_row_idx",
                "trigger_suffix_offset",
                "fold",
                "md_since",
                "selected_branch_id",
                "oracle_branch_id",
                "base_mse",
                "selected_mse",
                "oracle_mse",
                "base_rmse",
                "selected_rmse",
                "oracle_rmse",
                "alternative_within10",
                "evidence_reciprocal_rank",
                "base_first_reciprocal_rank",
                "selected_is_oracle",
            ]
        )
    hidden, hidden_report = load_hidden_like_late(config, ledger)
    trigger_readout = trigger_readout.merge(
        hidden, on="well_id", how="left", validate="many_to_one"
    )
    event_readout = event_readout.merge(
        hidden, on="well_id", how="left", validate="many_to_one"
    )
    return trigger_readout, event_readout, hidden_report


# %% [markdown]
# ## 8. Stage 0 metrics and promotion gates

# %%
def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float | None:
    label = np.asarray(labels, dtype=bool)
    score = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(score)
    label = label[valid]
    score = score[valid]
    positive = int(label.sum())
    negative = int((~label).sum())
    if positive == 0 or negative == 0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy(np.float64)
    rank_sum_positive = float(ranks[label].sum())
    return float(
        (
            rank_sum_positive - positive * (positive + 1) / 2.0
        )
        / (positive * negative)
    )


def trigger_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "eligible_rows": 0,
            "accepted_triggers": 0,
            "trigger_row_fraction": None,
            "bad_event_fraction": None,
            "trigger_bad_event_auc": None,
            "circular_bad_event_auc": None,
            "auc_gain_over_circular": None,
        }
    real_auc = roc_auc_binary(
        frame["bad_event"].to_numpy(bool),
        frame["trigger_score"].to_numpy(np.float64),
    )
    circular_auc = roc_auc_binary(
        frame["bad_event"].to_numpy(bool),
        frame["circular_trigger_score"].to_numpy(np.float64),
    )
    return {
        "eligible_rows": int(len(frame)),
        "accepted_triggers": int(frame["accepted_trigger"].sum()),
        "trigger_row_fraction": float(frame["accepted_trigger"].mean()),
        "bad_event_fraction": float(frame["bad_event"].mean()),
        "trigger_bad_event_auc": real_auc,
        "circular_bad_event_auc": circular_auc,
        "auc_gain_over_circular": (
            float(real_auc - circular_auc)
            if real_auc is not None and circular_auc is not None
            else None
        ),
    }


def event_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "events": 0,
            "event_wells": 0,
            "alternative_branch_within10_coverage": None,
            "evidence_mrr": None,
            "base_first_mrr": None,
            "mrr_gain_vs_base_first": None,
            "base_rmse": None,
            "selected_rmse": None,
            "selected_branch_rmse_gain_vs_base_ft": None,
            "oracle_rmse": None,
            "selected_oracle_rate": None,
        }
    base_rmse = float(np.sqrt(frame["base_mse"].mean()))
    selected_rmse = float(np.sqrt(frame["selected_mse"].mean()))
    evidence_mrr = float(frame["evidence_reciprocal_rank"].mean())
    base_first_mrr = float(frame["base_first_reciprocal_rank"].mean())
    return {
        "events": int(len(frame)),
        "event_wells": int(frame["well_id"].nunique()),
        "alternative_branch_within10_coverage": float(
            frame["alternative_within10"].mean()
        ),
        "evidence_mrr": evidence_mrr,
        "base_first_mrr": base_first_mrr,
        "mrr_gain_vs_base_first": evidence_mrr - base_first_mrr,
        "base_rmse": base_rmse,
        "selected_rmse": selected_rmse,
        "selected_branch_rmse_gain_vs_base_ft": base_rmse - selected_rmse,
        "oracle_rmse": float(np.sqrt(frame["oracle_mse"].mean())),
        "selected_oracle_rate": float(frame["selected_is_oracle"].mean()),
    }


def scope_metrics(
    scope: str,
    trigger_frame: pd.DataFrame,
    event_frame: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "scope": scope,
        **trigger_metrics(trigger_frame),
        **event_metrics(event_frame),
    }


def build_scope_and_fold_metrics(
    trigger_readout: pd.DataFrame,
    event_readout: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes = [
        scope_metrics("overall", trigger_readout, event_readout),
    ]
    for scope in ("hidden_like_spatial", "hidden_like_typewell_purged"):
        trigger_mask = trigger_readout[scope].astype(str).eq("valid")
        event_mask = (
            event_readout[scope].astype(str).eq("valid")
            if not event_readout.empty
            else np.zeros(0, dtype=bool)
        )
        scopes.append(
            scope_metrics(
                scope,
                trigger_readout.loc[trigger_mask],
                event_readout.loc[event_mask],
            )
        )
    scope_frame = pd.DataFrame(scopes)
    folds: list[dict[str, Any]] = []
    for fold in [int(value) for value in get_nested(config, "validation.expected_folds")]:
        folds.append(
            scope_metrics(
                f"fold_{fold}",
                trigger_readout.loc[trigger_readout["fold"].eq(fold)],
                event_readout.loc[event_readout["fold"].eq(fold)],
            )
            | {"fold": fold}
        )
    return scope_frame, pd.DataFrame(folds)


def _at_least(value: Any, threshold: float) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) >= threshold


def evaluate_stage0_gate(
    scope_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = get_nested(config, "validation.stage_0.all_required") or {}
    overall = scope_frame.loc[scope_frame["scope"].eq("overall")].iloc[0]
    event_range = [
        float(value) for value in gates["trigger_row_fraction_range"]
    ]

    def row_checks(row: pd.Series) -> dict[str, bool]:
        fraction = row["trigger_row_fraction"]
        return {
            "trigger_auc": _at_least(
                row["trigger_bad_event_auc"],
                float(gates["minimum_trigger_bad_event_auc"]),
            ),
            "auc_gain_over_circular": _at_least(
                row["auc_gain_over_circular"],
                float(gates["minimum_auc_gain_over_circular"]),
            ),
            "trigger_row_fraction": (
                fraction is not None
                and event_range[0] <= float(fraction) <= event_range[1]
            ),
            "alternative_coverage": _at_least(
                row["alternative_branch_within10_coverage"],
                float(gates["minimum_alternative_branch_within10_coverage"]),
            ),
            "mrr_gain": _at_least(
                row["mrr_gain_vs_base_first"],
                float(gates["minimum_selected_branch_mrr_gain_vs_base_first"]),
            ),
        }

    overall_checks = row_checks(overall)
    fold_rows: list[dict[str, Any]] = []
    for row in fold_frame.to_dict(orient="records"):
        checks = row_checks(pd.Series(row))
        fold_rows.append(
            {
                "fold": int(row["fold"]),
                **checks,
                "passed": bool(all(checks.values())),
            }
        )
    passing_folds = sum(int(row["passed"]) for row in fold_rows)
    hidden_checks: dict[str, bool] = {}
    for scope in ("hidden_like_spatial", "hidden_like_typewell_purged"):
        row = scope_frame.loc[scope_frame["scope"].eq(scope)].iloc[0]
        hidden_checks[scope] = _at_least(
            row["selected_branch_rmse_gain_vs_base_ft"], 0.0
        ) and float(row["selected_branch_rmse_gain_vs_base_ft"]) > 0.0
    checks = {
        **overall_checks,
        "minimum_passing_folds": passing_folds
        >= int(gates["minimum_passing_folds"]),
        "positive_hidden_like_spatial_direction": hidden_checks[
            "hidden_like_spatial"
        ],
        "positive_hidden_like_typewell_purged_direction": hidden_checks[
            "hidden_like_typewell_purged"
        ],
    }
    passed = bool(all(checks.values()))
    return {
        "stage": "stage_0",
        "technical_gate_passed": True,
        "scientific_gate_passed": passed,
        "stage_1_eligible": passed,
        "checks": checks,
        "fold_checks": fold_rows,
        "passing_folds": passing_folds,
        "required_passing_folds": int(gates["minimum_passing_folds"]),
        "decision": (
            "stage0_pass_wait_for_separate_stage1_approval"
            if passed
            else "stage0_failed_close_without_semimarkov_hmm"
        ),
    }


# %% [markdown]
# ## 9. Execution orchestration and generated artifacts

# %%
def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=True)
    started = time.time()
    output_dir = output_directory(config)
    prefix = str(get_nested(config, "audit.output_prefix"))
    ledger = TruthAccessLedger()
    saved_exp209, saved_report = load_saved_exp209_target_free(config)
    train_dir, wells, raw_report = discover_raw_train(config)
    if sorted(wells) != sorted(saved_exp209["well_id"].unique().tolist()):
        raise ValueError("raw train and saved exp209 well identity mismatch")
    trigger_parts: list[pd.DataFrame] = []
    branch_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for well_id in sorted(wells):
        trigger, branches, manifest = build_prefreeze_rows_for_well(
            well_id,
            saved_exp209.loc[saved_exp209["well_id"].eq(well_id)],
            train_dir,
            config,
            ledger,
        )
        trigger_parts.append(trigger)
        if not branches.empty:
            branch_parts.append(branches)
        manifest_rows.append(manifest)
    trigger_ledger = pd.concat(trigger_parts, ignore_index=True)
    branch_ledger = (
        pd.concat(branch_parts, ignore_index=True)
        if branch_parts
        else pd.DataFrame()
    )
    trigger_ledger = assign_group_folds(
        trigger_ledger,
        int(get_nested(config, "validation.n_folds")),
    )
    branch_ledger = attach_event_folds(branch_ledger, trigger_ledger)
    input_manifest = pd.DataFrame(manifest_rows).sort_values(
        "well_id", kind="mergesort"
    )
    contract = validate_scientific_contract(config)
    freeze_manifest = freeze_prefreeze_artifacts(
        output_dir,
        prefix,
        input_manifest,
        trigger_ledger,
        branch_ledger,
        contract,
        ledger,
    )
    trigger_readout, event_readout, hidden_report = build_late_readouts(
        train_dir,
        saved_exp209,
        trigger_ledger,
        branch_ledger,
        config,
        ledger,
    )
    scope_frame, fold_frame = build_scope_and_fold_metrics(
        trigger_readout, event_readout, config
    )
    gate = evaluate_stage0_gate(scope_frame, fold_frame, config)
    generated_paths = {
        "trigger_readout": output_dir / f"{prefix}_trigger_readout.csv.gz",
        "event_readout": output_dir / f"{prefix}_event_readout.csv.gz",
        "scope_metrics": output_dir / f"{prefix}_scope_metrics.csv",
        "fold_metrics": output_dir / f"{prefix}_fold_metrics.csv",
        "gate_report": output_dir / f"{prefix}_gate_report.json",
        "summary": output_dir / f"{prefix}_summary.json",
    }
    write_gzip_csv(generated_paths["trigger_readout"], trigger_readout)
    write_gzip_csv(generated_paths["event_readout"], event_readout)
    write_csv(generated_paths["scope_metrics"], scope_frame)
    write_csv(generated_paths["fold_metrics"], fold_frame)
    write_json(generated_paths["gate_report"], gate)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_0",
        "decision": gate["decision"],
        "technical_gate_passed": gate["technical_gate_passed"],
        "scientific_gate_passed": gate["scientific_gate_passed"],
        "runtime_seconds": time.time() - started,
        "rows": int(len(saved_exp209)),
        "wells": int(saved_exp209["well_id"].nunique()),
        "eligible_trigger_rows": int(len(trigger_ledger)),
        "accepted_triggers": int(trigger_ledger["accepted_trigger"].sum()),
        "branch_rows": int(len(branch_ledger)),
        "semimarkov_hmm_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": 0,
        "truth_rows_before_freeze": ledger.truth_rows_before_freeze,
        "hidden_role_rows_before_freeze": ledger.hidden_role_rows_before_freeze,
        "truth_rows_after_freeze": ledger.truth_rows_after_freeze,
        "hidden_role_rows_after_freeze": ledger.hidden_role_rows_after_freeze,
        "contract_sha256": contract["contract_sha256"],
        "saved_exp209": saved_report,
        "raw_train": raw_report,
        "hidden_like": hidden_report,
        "freeze_manifest": freeze_manifest,
        "scope_metrics": scope_frame.to_dict(orient="records"),
        "fold_metrics": fold_frame.to_dict(orient="records"),
        "gate": gate,
        "runtime_versions": runtime_versions(),
    }
    write_json(generated_paths["summary"], summary)
    summary["generated_artifacts"] = {
        key: inspect_artifact(path)
        for key, path in generated_paths.items()
        if key != "summary"
    }
    write_json(generated_paths["summary"], summary)
    return summary


# %% [markdown]
# ## 10. Setup and fail-closed execution selection

# %%
CONFIG = load_config()
CONTRACT = validate_scientific_contract(CONFIG)
SETUP = {
    "experiment": EXPERIMENT_NAME,
    "route": get_nested(CONFIG, "experiment.route"),
    "status": get_nested(CONFIG, "experiment.status"),
    "implementation_scope": get_nested(CONFIG, "execution.implementation_scope"),
    "contract_sha256": CONTRACT["contract_sha256"],
    "fixed_branches": len(CONTRACT["branches"]),
    "evaluation_horizon_rows": CONTRACT["evaluation_horizon_rows"],
    "reporting_folds": get_nested(CONFIG, "validation.n_folds"),
    "semimarkov_hmm_well_runs": get_nested(
        CONFIG, "execution.stage_0_counts.semimarkov_hmm_well_runs"
    ),
    "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
    "run_stage_1": get_nested(CONFIG, "execution.run_stage_1"),
    "run_inference": get_nested(CONFIG, "execution.run_inference"),
    "create_submission": get_nested(CONFIG, "execution.create_submission"),
}
display(pd.DataFrame([SETUP]))

# %%
if bool(get_nested(CONFIG, "execution.run_stage_0")) and get_ipython() is not None:
    STAGE0_SUMMARY = run_stage0(CONFIG)
    display(pd.DataFrame(STAGE0_SUMMARY["scope_metrics"]))
    display(pd.DataFrame(STAGE0_SUMMARY["fold_metrics"]))
    display(STAGE0_SUMMARY["gate"])
elif bool(get_nested(CONFIG, "execution.run_stage_0")):
    print(
        "exp366 Stage 0 is approved and will run in the canonical Kaggle "
        "Notebook. Direct module import remains side-effect free."
    )
else:
    print(
        "exp366 Stage 0 implementation is ready, but execution.run_stage_0=false. "
        "Kaggle package/push/run, Stage 1, inference, and submission remain disabled."
    )

# %% [markdown]
# # exp426 RSD-binned pattern absolute reanchor — Stage A
#
# This deterministic zero-model readout tests whether Wu et al. (2019)-style
# 0.5-ft RSD bin averaging plus Pearson pattern matching can identify a fixed
# absolute offset of the saved fold-safe exp226 final path.  The target-free
# score bank, support, ranks, top-3 candidates, manifests, and logical SHA are
# frozen before true TVT or hidden-like roles are read.  Stage B, Stage C,
# inference, and submission remain disabled.

# %% [markdown]
# ## Contents
# 1. Imports and execution guard
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen Stage A scientific contract
# 4. Target-free exp226, horizontal, and Type Well loaders
# 5. RSD-binned and matched-control score helpers
# 6. Per-well 512-row / 13-offset score-bank generation
# 7. Target-free freeze and independent probe parity
# 8. Post-freeze truth, oracle, scope, and fold readout
# 9. Technical and scientific gates
# 10. Metrics and generated artifacts
# 11. Setup, configuration preview, and approved execution

# %%
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import os
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

EXPERIMENT_NAME = "exp426_rsd_binned_pattern_absolute_reanchor"
OUTPUT_PREFIX = EXPERIMENT_NAME
IMPORT_ONLY_ENV = "EXP426_IMPORT_ONLY"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
OFFSETS_FT = np.asarray(
    [-80.0, -40.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    dtype=np.float64,
)
TIE_ORDER_FT = np.asarray(
    [0.0, -2.0, 2.0, -5.0, 5.0, -10.0, 10.0, -20.0, 20.0, -40.0, 40.0, -80.0, 80.0],
    dtype=np.float64,
)
VARIANTS = (
    "rsd_binned_pearson",
    "raw_pointwise_pearson",
    "raw_gaussian",
    "stable_permutation",
)
PRIMARY_VARIANT = "rsd_binned_pearson"
CONTROL_VARIANTS = (
    "raw_pointwise_pearson",
    "raw_gaussian",
    "stable_permutation",
)
SAFE_EXP226_COLUMNS = ("well_id", "row_idx", "suffix_offset", "fold", "tvt_pred")
FORBIDDEN_PRE_FREEZE_COLUMNS = {
    "TVT",
    "tvt_true",
    "actual_tvt",
    "target",
    "error",
    "abs_error",
    "oracle_offset_ft",
    "persistent_episode",
    "persistent_cause",
    "verification_like_spatial_role",
    "verification_like_typewell_purged_role",
}
SCORE_LOGICAL_COLUMNS = [
    "well_id",
    "fold",
    "block_id",
    "block_start_suffix_offset",
    "block_end_suffix_offset",
    "block_start_row_idx",
    "block_end_row_idx",
    "block_row_count",
    "md_since_min_ft",
    "md_since_max_ft",
    "md_since_mid_ft",
    "raw_finite_gr_points",
    "observed_gr_share",
    "offset_slot",
    "offset_ft",
    "rsd_bin_score",
    "rsd_pearson",
    "rsd_cosine",
    "rsd_spearman",
    "rsd_paired_bins",
    "rsd_valid",
    "rsd_rank",
    "rsd_top3",
    "raw_pearson_score",
    "raw_pearson_pairs",
    "raw_pearson_valid",
    "raw_pearson_rank",
    "raw_pearson_top3",
    "raw_gaussian_score",
    "raw_gaussian_valid",
    "raw_gaussian_rank",
    "raw_gaussian_top3",
    "permutation_score",
    "permutation_valid",
    "permutation_rank",
    "permutation_top3",
]


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get(IMPORT_ONLY_ENV, "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
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


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file() and (candidate / "experiments").is_dir():
            return candidate
    return start


def experiment_dir() -> Path:
    candidate = project_root() / "experiments" / EXPERIMENT_NAME
    return candidate if candidate.is_dir() else Path.cwd().resolve()


def load_experiment_config() -> dict[str, Any]:
    candidates = (Path.cwd() / "config.yaml", experiment_dir() / "config.yaml")
    for path in candidates:
        value = read_yaml(path)
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return value
    raise FileNotFoundError(f"exp426 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if KAGGLE_WORKING_ROOT.is_dir()
        else experiment_dir() / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return experiment_dir() / "metrics.json"


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        to_jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_frame_for_hash(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column, dtype in normalized.dtypes.items():
        if isinstance(dtype, pd.StringDtype):
            normalized[column] = normalized[column].astype(object)
    return normalized


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> str:
    selected = frame if columns is None else frame[list(columns)]
    selected = _normalize_frame_for_hash(selected)
    digest = hashlib.sha256()
    digest.update("|".join(selected.columns).encode())
    digest.update("|".join(str(dtype) for dtype in selected.dtypes).encode())
    hashes = pd.util.hash_pandas_object(selected, index=False, categorize=True)
    digest.update(hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    normalized = _normalize_frame_for_hash(frame)
    schema = [(column, str(dtype)) for column, dtype in normalized.dtypes.items()]
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": sha256_gzip_decompressed(path),
        "content_sha256": dataframe_content_sha(frame),
        "schema_sha256": dataframe_schema_sha(frame),
    }


def expand_existing_paths(patterns: Sequence[str]) -> list[Path]:
    root = project_root()
    found: dict[str, Path] = {}
    for raw in map(str, patterns):
        path = Path(raw)
        direct = path if path.is_absolute() else root / path
        if direct.is_file() and direct.stat().st_size > 0:
            found[str(direct.resolve())] = direct
            continue
        searches = [raw]
        if not path.is_absolute():
            searches.append(str(root / raw))
        for search in searches:
            for match in glob.glob(search, recursive=True):
                candidate = Path(match)
                if candidate.is_file() and candidate.stat().st_size > 0:
                    found[str(candidate.resolve())] = candidate
    return list(found.values())


def resolve_file(
    patterns: Sequence[str],
    *,
    label: str,
    expected_file_sha256: str | None = None,
    expected_decompressed_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for candidate in expand_existing_paths(patterns):
        row: dict[str, Any] = {
            "path": str(candidate),
            "bytes": candidate.stat().st_size,
            "file_sha256": sha256_path(candidate),
        }
        if expected_decompressed_sha256 is not None:
            row["decompressed_sha256"] = sha256_gzip_decompressed(candidate)
        evidence.append(row)
        if expected_file_sha256 is not None and row["file_sha256"] != expected_file_sha256:
            continue
        if (
            expected_decompressed_sha256 is not None
            and row["decompressed_sha256"] != expected_decompressed_sha256
        ):
            continue
        return candidate, row
    raise FileNotFoundError(f"Could not resolve {label} with fixed SHA: {evidence[:8]}")


def stable_uint64(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def peak_rss_gb() -> float:
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


# %% [markdown]
# ## 3. Frozen Stage A scientific contract


# %%
def assert_no_forbidden_columns(columns: Iterable[str]) -> None:
    present = set(map(str, columns)).intersection(FORBIDDEN_PRE_FREEZE_COLUMNS)
    if present:
        raise ValueError(f"truth/error columns are forbidden before freeze: {sorted(present)}")


def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("unexpected experiment name")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp426 route must remain pf_beam")
    if not bool(get_nested(config, "implementation.enabled")):
        raise ValueError("exp426 Stage A implementation must be enabled")
    if get_nested(config, "implementation.scope") != "stage_a_implementation_ready":
        raise ValueError("only the Stage A implementation scope is allowed")
    if not bool(get_nested(config, "model.stage_a.enabled")):
        raise ValueError("Stage A must be enabled")
    if bool(get_nested(config, "model.stage_b.enabled")) or bool(
        get_nested(config, "model.stage_c.enabled")
    ):
        raise ValueError("Stage B and Stage C must remain disabled")

    candidate = get_nested(config, "model.common_candidate_contract") or {}
    observed_offsets = np.asarray(candidate.get("offsets_ft", ()), dtype=np.float64)
    observed_ties = np.asarray(candidate.get("tie_order_ft", ()), dtype=np.float64)
    if not np.array_equal(observed_offsets, OFFSETS_FT):
        raise ValueError("the frozen 13-offset bank changed")
    if not np.array_equal(observed_ties, TIE_ORDER_FT):
        raise ValueError("the deterministic tie order changed")
    expected_candidate = {
        "block_size_rows": 512,
        "block_overlap_rows": 0,
        "rsd_bin_width_ft": 0.5,
        "rsd_bin_origin_ft": 0.0,
        "minimum_raw_finite_gr_points": 32,
        "minimum_paired_occupied_bins": 16,
        "minimum_std": 1.0e-6,
        "correlation_clip_epsilon": 1.0e-6,
    }
    for key, expected in expected_candidate.items():
        observed = candidate.get(key)
        if isinstance(expected, float):
            if float(observed) != expected:
                raise ValueError(f"candidate contract changed: {key}={observed}")
        elif int(observed) != expected:
            raise ValueError(f"candidate contract changed: {key}={observed}")
    if candidate.get("base_path") != "exp226_final_tvt_pred":
        raise ValueError("Stage A must use saved exp226 final tvt_pred")
    if candidate.get("typewell_interpolation") != "finite_linear_no_extrapolation_no_endpoint_hold":
        raise ValueError("RSD Type Well interpolation contract changed")
    if candidate.get("horizontal_gr_imputation") != "none":
        raise ValueError("RSD horizontal GR imputation must remain none")

    counts = get_nested(config, "execution_contract") or {}
    required_zero = {
        "stage_a_b_model_configs": 0,
        "stage_a_b_trained_folds": 0,
        "stage_a_b_boosters": 0,
        "stage_a_b_hmm_runs": 0,
        "stage_a_b_pf_runs": 0,
        "stage_a_b_beam_runs": 0,
        "gpu_runs": 0,
    }
    for key, expected in required_zero.items():
        if int(counts.get(key, -1)) != expected:
            raise ValueError(f"zero-model Stage A contract changed: {key}")
    if bool(counts.get("parent_control_retraining")):
        raise ValueError("parent control retraining is forbidden")
    if int(counts.get("stage_a_score_variants", -1)) != 1:
        raise ValueError("Stage A requires one primary score variant")
    if int(counts.get("stage_a_controls", -1)) != 3:
        raise ValueError("Stage A requires three matched controls")
    if get_nested(config, "validation.scope_block_assignment") != "md_since_mid_ft":
        raise ValueError("Stage A scopes must use the frozen block midpoint assignment")

    if not bool(get_nested(config, "execution.implementation_authorized")):
        raise ValueError("Stage A implementation is not authorized")
    forbidden_true = (
        "execution.run_stage_b",
        "execution.run_stage_c_sentinel",
        "execution.run_stage_c_full",
        "execution.run_inference",
        "execution.create_submission",
        "implementation.inference_enabled",
        "implementation.submission_enabled",
    )
    if any(bool(get_nested(config, key)) for key in forbidden_true):
        raise ValueError("Stage B/C, inference, and submission must remain disabled")
    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_package_authorized"))
        and bool(get_nested(config, "execution.kaggle_push_authorized"))
        and bool(get_nested(config, "execution.kaggle_execution_authorized"))
        and bool(get_nested(config, "execution.run_stage_a"))
    ):
        raise RuntimeError("exp426 Stage A Kaggle package/push/run is not approved")

    contract = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_a_absolute_datum_identifiability",
        "base_path": "exp226_final_tvt_pred",
        "offsets_ft": OFFSETS_FT.tolist(),
        "tie_order_ft": TIE_ORDER_FT.tolist(),
        "block_size_rows": 512,
        "rsd_bin_width_ft": 0.5,
        "variants": list(VARIANTS),
        "truth_attachment": "after_target_free_score_rank_top3_manifest_and_sha_freeze",
        "execution_counts": {
            "primary_scores": 1,
            "descriptive_scores": 2,
            "matched_controls": 3,
            "model_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "hmm_runs": 0,
            "pf_runs": 0,
            "beam_runs": 0,
            "gpu_runs": 0,
        },
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


class TruthAccessLedger:
    def __init__(self) -> None:
        self.frozen_score_content_sha256: str | None = None
        self.truth_rows_before_freeze = 0
        self.truth_rows_after_freeze = 0
        self.hidden_role_rows_before_freeze = 0
        self.hidden_role_rows_after_freeze = 0

    @property
    def frozen(self) -> bool:
        return self.frozen_score_content_sha256 is not None

    def mark_frozen(self, score_content_sha256: str) -> None:
        if len(score_content_sha256) != 64:
            raise ValueError("target-free freeze requires a SHA256")
        self.frozen_score_content_sha256 = score_content_sha256

    def register_truth_access(self, rows: int) -> None:
        if not self.frozen:
            self.truth_rows_before_freeze += int(rows)
            raise RuntimeError("truth access attempted before target-free freeze")
        self.truth_rows_after_freeze += int(rows)

    def register_hidden_role_access(self, rows: int) -> None:
        if not self.frozen:
            self.hidden_role_rows_before_freeze += int(rows)
            raise RuntimeError("hidden-like role access attempted before target-free freeze")
        self.hidden_role_rows_after_freeze += int(rows)


# %% [markdown]
# ## 4. Target-free exp226, horizontal, and Type Well loaders


# %%
def load_exp226_safe(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    spec = get_nested(config, "data.exp226") or {}
    path, evidence = resolve_file(
        [str(value) for value in spec["patterns"]],
        label="exp226 final OOF",
        expected_decompressed_sha256=str(spec["expected_decompressed_sha256"]),
    )
    safe_columns = [str(value) for value in spec["safe_columns"]]
    if tuple(safe_columns) != SAFE_EXP226_COLUMNS:
        raise ValueError("exp226 safe-column allowlist changed")
    frame = pd.read_csv(path, usecols=safe_columns, dtype={"well_id": str})
    assert_no_forbidden_columns(frame.columns)
    frame["well_id"] = frame["well_id"].astype(str)
    for column in ("row_idx", "suffix_offset", "fold"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["tvt_pred"] = pd.to_numeric(frame["tvt_pred"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 safe OOF has duplicate well_id/row_idx")
    if not np.isfinite(frame["tvt_pred"].to_numpy(np.float64)).all():
        raise ValueError("exp226 final tvt_pred must be finite")
    if len(frame) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp226 OOF row inventory changed")
    if frame["well_id"].nunique() != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("exp226 OOF well inventory changed")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if sorted(frame["fold"].unique().tolist()) != expected_folds:
        raise ValueError("exp226 OOF fold inventory changed")
    if not frame.groupby("well_id")["fold"].nunique().eq(1).all():
        raise ValueError("each exp226 well must have exactly one fold")
    return (
        frame,
        path,
        {
            "name": "exp226_final_oof_safe_columns",
            **evidence,
            "rows": len(frame),
            "wells": int(frame["well_id"].nunique()),
            "folds": sorted(int(value) for value in frame["fold"].unique()),
            "safe_columns": safe_columns,
        },
    )


def resolve_train_root(config: Mapping[str, Any]) -> Path:
    for raw in get_nested(config, "data.train_root_candidates") or ():
        path = Path(str(raw))
        candidate = path if path.is_absolute() else project_root() / path
        if candidate.is_dir() and any(candidate.glob("*__horizontal_well.csv")):
            return candidate
    raise FileNotFoundError("could not resolve raw train root")


def load_horizontal_safe(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["MD", "GR", "TVT_input"])
    assert_no_forbidden_columns(frame.columns)
    for column in ("MD", "GR", "TVT_input"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame["MD"].to_numpy(np.float64)).all():
        raise ValueError(f"{path} contains non-finite MD")
    return frame


def _aggregate_finite_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell must contain TVT and GR")
    frame = typewell[["TVT", "GR"]].copy()
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["TVT"]) & np.isfinite(frame["GR"])]
    frame = (
        frame.groupby("TVT", as_index=False, sort=True)["GR"]
        .mean()
        .sort_values("TVT", kind="mergesort")
    )
    if len(frame) < 2:
        raise ValueError("typewell requires at least two finite unique TVT/GR rows")
    return (
        frame["TVT"].to_numpy(np.float64),
        frame["GR"].to_numpy(np.float64),
    )


def prepare_typewell_gaussian(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    frame = typewell[["TVT", "GR"]].copy()
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["TVT"])].sort_values("TVT", kind="mergesort")
    frame["GR"] = frame["GR"].ffill().bfill()
    frame = frame.loc[np.isfinite(frame["GR"])]
    frame = frame.groupby("TVT", as_index=False, sort=True)["GR"].mean()
    if len(frame) < 2:
        raise ValueError("Gaussian control requires two finite Type Well samples")
    return frame["TVT"].to_numpy(np.float64), frame["GR"].to_numpy(np.float64)


def load_exp226_truth(
    path: Path,
    ledger: TruthAccessLedger,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["well_id", "row_idx", "tvt_true"],
        dtype={"well_id": str},
    )
    ledger.register_truth_access(len(frame))
    frame["well_id"] = frame["well_id"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["tvt_true"] = pd.to_numeric(frame["tvt_true"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 truth rows must be unique")
    if not np.isfinite(frame["tvt_true"].to_numpy(np.float64)).all():
        raise ValueError("exp226 truth must be finite")
    return frame


def load_hidden_like_assignments(
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like_assignment") or {}
    path, evidence = resolve_file(
        [str(value) for value in spec["patterns"]],
        label="hidden-like assignment",
        expected_file_sha256=str(spec["expected_sha256"]),
    )
    role_columns = [str(value) for value in spec["role_columns"].values()]
    frame = pd.read_csv(path, usecols=["well_id", *role_columns], dtype={"well_id": str})
    ledger.register_hidden_role_access(len(frame))
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment must have one row per well")
    for column in role_columns:
        frame[column] = frame[column].astype(str)
    return frame, {
        "name": "hidden_like_assignment",
        **evidence,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
    }


# %% [markdown]
# ## 5. RSD-binned and matched-control score helpers


# %%
def finite_pearson(
    left: np.ndarray,
    right: np.ndarray,
    *,
    minimum_pairs: int,
    minimum_std: float,
) -> tuple[float, int, bool]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    count = int(mask.sum())
    if count < minimum_pairs:
        return 0.0, count, False
    x = left[mask]
    y = right[mask]
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= minimum_std or y_std <= minimum_std:
        return 0.0, count, False
    correlation = float(np.mean((x - x.mean()) * (y - y.mean())) / (x_std * y_std))
    if not math.isfinite(correlation):
        return 0.0, count, False
    return float(np.clip(correlation, -1.0, 1.0)), count, True


def average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy()


def rsd_binned_pattern_score(
    raw_gr: np.ndarray,
    candidate_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    bin_width_ft: float,
    bin_origin_ft: float,
    minimum_raw_points: int,
    minimum_paired_bins: int,
    minimum_std: float,
    correlation_clip_epsilon: float,
) -> dict[str, Any]:
    raw_gr = np.asarray(raw_gr, dtype=np.float64)
    candidate_tvt = np.asarray(candidate_tvt, dtype=np.float64)
    raw_mask = np.isfinite(raw_gr) & np.isfinite(candidate_tvt)
    raw_count = int(raw_mask.sum())
    invalid = {
        "score": 0.0,
        "pearson": 0.0,
        "cosine": 0.0,
        "spearman": 0.0,
        "raw_finite_points": raw_count,
        "paired_bins": 0,
        "valid": False,
    }
    if raw_count < minimum_raw_points:
        return invalid

    rsd = candidate_tvt[raw_mask]
    observed = raw_gr[raw_mask]
    bin_id = np.floor((rsd - bin_origin_ft) / bin_width_ft).astype(np.int64)
    unique_bins, inverse = np.unique(bin_id, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=observed)
    horizontal_mean = sums / counts
    bin_centers = bin_origin_ft + (unique_bins.astype(np.float64) + 0.5) * bin_width_ft
    in_typewell_range = (bin_centers >= typewell_tvt[0]) & (bin_centers <= typewell_tvt[-1])
    paired_centers = bin_centers[in_typewell_range]
    paired_horizontal = horizontal_mean[in_typewell_range]
    paired_bins = int(len(paired_centers))
    if paired_bins < minimum_paired_bins:
        invalid["paired_bins"] = paired_bins
        return invalid
    paired_typewell = np.interp(paired_centers, typewell_tvt, typewell_gr)
    pearson, _, valid = finite_pearson(
        paired_horizontal,
        paired_typewell,
        minimum_pairs=minimum_paired_bins,
        minimum_std=minimum_std,
    )
    if not valid:
        invalid["paired_bins"] = paired_bins
        return invalid
    clipped = float(
        np.clip(pearson, -1.0 + correlation_clip_epsilon, 1.0 - correlation_clip_epsilon)
    )
    fisher_z = float(np.arctanh(clipped))
    score = 0.5 * float(np.sign(fisher_z)) * fisher_z**2
    denominator = float(np.sqrt(np.sum(paired_horizontal**2) * np.sum(paired_typewell**2)))
    cosine = (
        float(np.dot(paired_horizontal, paired_typewell) / denominator)
        if denominator > 0.0
        else 0.0
    )
    spearman, _, spearman_valid = finite_pearson(
        average_ranks(paired_horizontal),
        average_ranks(paired_typewell),
        minimum_pairs=minimum_paired_bins,
        minimum_std=minimum_std,
    )
    return {
        "score": float(score),
        "pearson": float(pearson),
        "cosine": float(cosine),
        "spearman": float(spearman if spearman_valid else 0.0),
        "raw_finite_points": raw_count,
        "paired_bins": paired_bins,
        "valid": True,
    }


def raw_pointwise_pearson_score(
    raw_gr: np.ndarray,
    candidate_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    minimum_pairs: int,
    minimum_std: float,
) -> dict[str, Any]:
    candidate_tvt = np.asarray(candidate_tvt, dtype=np.float64)
    expected = np.full(len(candidate_tvt), np.nan, dtype=np.float64)
    native = (candidate_tvt >= typewell_tvt[0]) & (candidate_tvt <= typewell_tvt[-1])
    expected[native] = np.interp(candidate_tvt[native], typewell_tvt, typewell_gr)
    score, pairs, valid = finite_pearson(
        np.asarray(raw_gr, dtype=np.float64),
        expected,
        minimum_pairs=minimum_pairs,
        minimum_std=minimum_std,
    )
    return {"score": float(score), "pairs": int(pairs), "valid": bool(valid)}


def prepare_raw_gaussian_control(
    horizontal_safe: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    typewell_tvt, typewell_gr = prepare_typewell_gaussian(typewell)
    known = horizontal_safe.loc[horizontal_safe["TVT_input"].notna()]
    if len(known) < 4:
        raise ValueError("Gaussian control needs at least four known-prefix rows")
    known_tvt = known["TVT_input"].to_numpy(np.float64)
    known_gr = known["GR"].fillna(0.0).to_numpy(np.float64)
    expected_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - expected_known
    sigma_clip = get_nested(
        config,
        "model.common_candidate_contract.raw_gaussian_control.sigma_clip",
    )
    sigma = float(np.clip(np.std(residual), float(sigma_clip[0]), float(sigma_clip[1])))
    fill_value = float(np.mean(typewell_gr))
    all_gr = (
        horizontal_safe["GR"]
        .interpolate(limit_direction="both")
        .fillna(fill_value)
        .to_numpy(np.float64)
    )
    if not math.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(all_gr).all():
        raise ValueError("Gaussian control preparation is non-finite")
    return {
        "typewell_tvt": typewell_tvt,
        "typewell_gr": typewell_gr,
        "sigma": sigma,
        "all_gr": all_gr,
        "known_rows": len(known),
        "known_residual_std_unclipped": float(np.std(residual)),
    }


def raw_gaussian_block_score(
    observed_gr: np.ndarray,
    candidate_tvt: np.ndarray,
    prepared: Mapping[str, Any],
    *,
    log_likelihood_clip: float,
) -> float:
    expected = np.interp(
        np.asarray(candidate_tvt, dtype=np.float64),
        np.asarray(prepared["typewell_tvt"], dtype=np.float64),
        np.asarray(prepared["typewell_gr"], dtype=np.float64),
    )
    zscore = (np.asarray(observed_gr, dtype=np.float64) - expected) / float(prepared["sigma"])
    score = float(np.mean(-0.5 * np.minimum(zscore**2, log_likelihood_clip)))
    if not math.isfinite(score):
        raise ValueError("raw Gaussian score is non-finite")
    return score


def tie_priority_by_slot(offsets: np.ndarray = OFFSETS_FT) -> np.ndarray:
    priority_by_value = {float(value): rank for rank, value in enumerate(TIE_ORDER_FT)}
    return np.asarray([priority_by_value[float(value)] for value in offsets], dtype=np.int16)


def rank_scores(scores: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if scores.shape != OFFSETS_FT.shape or valid.shape != OFFSETS_FT.shape:
        raise ValueError("ranking requires one score and support flag per fixed offset")
    if not np.isfinite(scores).all():
        raise ValueError("ranking requires finite score storage")
    tie_priority = tie_priority_by_slot()
    order = np.asarray(
        sorted(
            range(len(scores)),
            key=lambda slot: (
                not bool(valid[slot]),
                -float(scores[slot]),
                int(tie_priority[slot]),
            ),
        ),
        dtype=np.int16,
    )
    ranks = np.empty(len(scores), dtype=np.int16)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int16)
    return ranks, order


def stable_score_label_permutation(
    scores: np.ndarray,
    valid: np.ndarray,
    *,
    well_id: str,
    block_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(
        stable_uint64(EXPERIMENT_NAME, "stage_a_score_label_permutation", well_id, block_id)
    )
    permutation = rng.permutation(len(OFFSETS_FT))
    return (
        np.asarray(scores, dtype=np.float64)[permutation],
        np.asarray(valid, dtype=bool)[permutation],
    )


# %% [markdown]
# ## 6. Per-well 512-row / 13-offset score-bank generation


# %%
def score_well_target_free(
    oof_safe: pd.DataFrame,
    horizontal_safe: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert_no_forbidden_columns(oof_safe.columns)
    assert_no_forbidden_columns(horizontal_safe.columns)
    required_oof = set(SAFE_EXP226_COLUMNS)
    if not required_oof.issubset(oof_safe.columns):
        raise ValueError(f"safe OOF missing {sorted(required_oof - set(oof_safe.columns))}")
    if oof_safe.empty or oof_safe["well_id"].nunique() != 1 or oof_safe["fold"].nunique() != 1:
        raise ValueError("score_well_target_free requires one non-empty well and fold")
    oof = oof_safe.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_idx = oof["row_idx"].to_numpy(np.int64)
    suffix_offset = oof["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(suffix_offset, np.arange(len(oof), dtype=np.int64)):
        raise ValueError("suffix_offset must be contiguous from zero")
    if row_idx.min() < 0 or row_idx.max() >= len(horizontal_safe):
        raise ValueError("exp226 row_idx is outside the raw horizontal frame")
    if horizontal_safe.iloc[row_idx]["TVT_input"].notna().any():
        raise ValueError("exp226 OOF must align only to unknown-suffix rows")

    candidate_contract = get_nested(config, "model.common_candidate_contract") or {}
    block_rows = int(candidate_contract["block_size_rows"])
    minimum_raw = int(candidate_contract["minimum_raw_finite_gr_points"])
    minimum_bins = int(candidate_contract["minimum_paired_occupied_bins"])
    minimum_std = float(candidate_contract["minimum_std"])
    typewell_tvt, typewell_gr = _aggregate_finite_typewell(typewell)
    gaussian = prepare_raw_gaussian_control(horizontal_safe, typewell, config)
    log_clip = float(candidate_contract["raw_gaussian_control"]["log_likelihood_clip"])

    raw_gr_all = horizontal_safe["GR"].to_numpy(np.float64)
    raw_gr = raw_gr_all[row_idx]
    md = horizontal_safe["MD"].to_numpy(np.float64)
    known_positions = np.flatnonzero(horizontal_safe["TVT_input"].notna().to_numpy())
    if not len(known_positions):
        raise ValueError("well has no known TVT_input prefix")
    last_known = int(known_positions[-1])
    md_since = md[row_idx] - md[last_known]
    base_path = oof["tvt_pred"].to_numpy(np.float64)
    block_ids = suffix_offset // block_rows
    well_id = str(oof["well_id"].iloc[0])
    fold = int(oof["fold"].iloc[0])
    rows: list[dict[str, Any]] = []

    for block_id in np.unique(block_ids):
        block_mask = block_ids == block_id
        positions = np.flatnonzero(block_mask)
        base_block = base_path[block_mask]
        raw_block = raw_gr[block_mask]
        gaussian_gr_block = np.asarray(gaussian["all_gr"], dtype=np.float64)[row_idx[block_mask]]
        rsd_scores = np.zeros(len(OFFSETS_FT), dtype=np.float64)
        rsd_valid = np.zeros(len(OFFSETS_FT), dtype=bool)
        rsd_pearson = np.zeros(len(OFFSETS_FT), dtype=np.float64)
        rsd_cosine = np.zeros(len(OFFSETS_FT), dtype=np.float64)
        rsd_spearman = np.zeros(len(OFFSETS_FT), dtype=np.float64)
        rsd_paired_bins = np.zeros(len(OFFSETS_FT), dtype=np.int32)
        raw_pearson_scores = np.zeros(len(OFFSETS_FT), dtype=np.float64)
        raw_pearson_pairs = np.zeros(len(OFFSETS_FT), dtype=np.int32)
        raw_pearson_valid = np.zeros(len(OFFSETS_FT), dtype=bool)
        gaussian_scores = np.zeros(len(OFFSETS_FT), dtype=np.float64)
        gaussian_valid = np.ones(len(OFFSETS_FT), dtype=bool)

        for slot, offset in enumerate(OFFSETS_FT):
            candidate = base_block + offset
            rsd = rsd_binned_pattern_score(
                raw_block,
                candidate,
                typewell_tvt,
                typewell_gr,
                bin_width_ft=float(candidate_contract["rsd_bin_width_ft"]),
                bin_origin_ft=float(candidate_contract["rsd_bin_origin_ft"]),
                minimum_raw_points=minimum_raw,
                minimum_paired_bins=minimum_bins,
                minimum_std=minimum_std,
                correlation_clip_epsilon=float(candidate_contract["correlation_clip_epsilon"]),
            )
            point = raw_pointwise_pearson_score(
                raw_block,
                candidate,
                typewell_tvt,
                typewell_gr,
                minimum_pairs=minimum_raw,
                minimum_std=minimum_std,
            )
            rsd_scores[slot] = rsd["score"]
            rsd_valid[slot] = rsd["valid"]
            rsd_pearson[slot] = rsd["pearson"]
            rsd_cosine[slot] = rsd["cosine"]
            rsd_spearman[slot] = rsd["spearman"]
            rsd_paired_bins[slot] = rsd["paired_bins"]
            raw_pearson_scores[slot] = point["score"]
            raw_pearson_pairs[slot] = point["pairs"]
            raw_pearson_valid[slot] = point["valid"]
            gaussian_scores[slot] = raw_gaussian_block_score(
                gaussian_gr_block,
                candidate,
                gaussian,
                log_likelihood_clip=log_clip,
            )

        permutation_scores, permutation_valid = stable_score_label_permutation(
            rsd_scores,
            rsd_valid,
            well_id=well_id,
            block_id=int(block_id),
        )
        ranks: dict[str, np.ndarray] = {}
        top3: dict[str, np.ndarray] = {}
        for variant, values, valid in (
            (PRIMARY_VARIANT, rsd_scores, rsd_valid),
            ("raw_pointwise_pearson", raw_pearson_scores, raw_pearson_valid),
            ("raw_gaussian", gaussian_scores, gaussian_valid),
            ("stable_permutation", permutation_scores, permutation_valid),
        ):
            variant_ranks, order = rank_scores(values, valid)
            ranks[variant] = variant_ranks
            valid_order = [int(slot) for slot in order if bool(valid[slot])]
            top3_mask = np.zeros(len(OFFSETS_FT), dtype=bool)
            top3_mask[valid_order[:3]] = True
            top3[variant] = top3_mask

        for slot, offset in enumerate(OFFSETS_FT):
            rows.append(
                {
                    "well_id": well_id,
                    "fold": fold,
                    "block_id": int(block_id),
                    "block_start_suffix_offset": int(suffix_offset[positions[0]]),
                    "block_end_suffix_offset": int(suffix_offset[positions[-1]]),
                    "block_start_row_idx": int(row_idx[positions[0]]),
                    "block_end_row_idx": int(row_idx[positions[-1]]),
                    "block_row_count": int(block_mask.sum()),
                    "md_since_min_ft": float(np.min(md_since[block_mask])),
                    "md_since_max_ft": float(np.max(md_since[block_mask])),
                    "md_since_mid_ft": float(np.mean(md_since[block_mask])),
                    "raw_finite_gr_points": int(np.isfinite(raw_block).sum()),
                    "observed_gr_share": float(np.isfinite(raw_block).mean()),
                    "offset_slot": int(slot),
                    "offset_ft": float(offset),
                    "rsd_bin_score": float(rsd_scores[slot]),
                    "rsd_pearson": float(rsd_pearson[slot]),
                    "rsd_cosine": float(rsd_cosine[slot]),
                    "rsd_spearman": float(rsd_spearman[slot]),
                    "rsd_paired_bins": int(rsd_paired_bins[slot]),
                    "rsd_valid": bool(rsd_valid[slot]),
                    "rsd_rank": int(ranks[PRIMARY_VARIANT][slot]),
                    "rsd_top3": bool(top3[PRIMARY_VARIANT][slot]),
                    "raw_pearson_score": float(raw_pearson_scores[slot]),
                    "raw_pearson_pairs": int(raw_pearson_pairs[slot]),
                    "raw_pearson_valid": bool(raw_pearson_valid[slot]),
                    "raw_pearson_rank": int(ranks["raw_pointwise_pearson"][slot]),
                    "raw_pearson_top3": bool(top3["raw_pointwise_pearson"][slot]),
                    "raw_gaussian_score": float(gaussian_scores[slot]),
                    "raw_gaussian_valid": bool(gaussian_valid[slot]),
                    "raw_gaussian_rank": int(ranks["raw_gaussian"][slot]),
                    "raw_gaussian_top3": bool(top3["raw_gaussian"][slot]),
                    "permutation_score": float(permutation_scores[slot]),
                    "permutation_valid": bool(permutation_valid[slot]),
                    "permutation_rank": int(ranks["stable_permutation"][slot]),
                    "permutation_top3": bool(top3["stable_permutation"][slot]),
                }
            )

    scores = pd.DataFrame(rows)[SCORE_LOGICAL_COLUMNS].sort_values(
        ["well_id", "block_id", "offset_slot"],
        kind="mergesort",
    )
    manifest = {
        "well_id": well_id,
        "fold": fold,
        "horizontal_rows": len(horizontal_safe),
        "evaluation_rows": len(oof),
        "blocks": int(scores["block_id"].nunique()),
        "known_rows": int(gaussian["known_rows"]),
        "last_known_row_idx": last_known,
        "raw_eval_gr_share": float(np.isfinite(raw_gr).mean()),
        "gaussian_sigma": float(gaussian["sigma"]),
        "gaussian_known_residual_std_unclipped": float(gaussian["known_residual_std_unclipped"]),
        "primary_supported_blocks": int(
            scores.groupby("block_id", sort=False)["rsd_valid"].any().sum()
        ),
    }
    return scores.reset_index(drop=True), manifest


def generate_target_free_score_bank(
    oof_safe: pd.DataFrame,
    train_root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    score_parts: list[pd.DataFrame] = []
    well_manifests: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for well_id, well_oof in oof_safe.groupby("well_id", sort=True):
        horizontal_path = train_root / f"{well_id}__horizontal_well.csv"
        typewell_path = train_root / f"{well_id}__typewell.csv"
        if not horizontal_path.is_file() or not typewell_path.is_file():
            raise FileNotFoundError(f"missing raw files for {well_id}")
        horizontal = load_horizontal_safe(horizontal_path)
        typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        scores, manifest = score_well_target_free(well_oof, horizontal, typewell, config)
        score_parts.append(scores)
        well_manifests.append(manifest)
        input_rows.append(
            {
                "well_id": str(well_id),
                "horizontal_path": str(horizontal_path),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_path": str(typewell_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    score_bank = pd.concat(score_parts, ignore_index=True).sort_values(
        ["well_id", "block_id", "offset_slot"], kind="mergesort"
    )
    return (
        score_bank.reset_index(drop=True),
        pd.DataFrame(well_manifests).sort_values("well_id", kind="mergesort"),
        input_rows,
    )


# %% [markdown]
# ## 7. Target-free freeze and independent probe parity


# %%
def score_bank_logical_sha(score_bank: pd.DataFrame) -> str:
    return dataframe_content_sha(
        score_bank.sort_values(
            ["well_id", "block_id", "offset_slot"], kind="mergesort"
        ).reset_index(drop=True),
        SCORE_LOGICAL_COLUMNS,
    )


def validate_score_bank_structure(score_bank: pd.DataFrame) -> dict[str, bool]:
    required = set(SCORE_LOGICAL_COLUMNS)
    checks: dict[str, bool] = {
        "required_columns": required.issubset(score_bank.columns),
        "duplicate_identity_zero": not score_bank.duplicated(
            ["well_id", "block_id", "offset_slot"]
        ).any(),
    }
    if not checks["required_columns"]:
        return checks
    sorted_bank = score_bank.sort_values(
        ["well_id", "block_id", "offset_slot"], kind="mergesort"
    ).reset_index(drop=True)
    checks["canonical_order"] = sorted_bank[SCORE_LOGICAL_COLUMNS].equals(
        score_bank.reset_index(drop=True)[SCORE_LOGICAL_COLUMNS]
    )
    numeric_score_columns = (
        "rsd_bin_score",
        "rsd_pearson",
        "rsd_cosine",
        "rsd_spearman",
        "raw_pearson_score",
        "raw_gaussian_score",
        "permutation_score",
    )
    checks["finite_score_storage"] = bool(
        np.isfinite(score_bank[list(numeric_score_columns)].to_numpy(np.float64)).all()
    )
    groups = score_bank.groupby(["well_id", "block_id"], sort=False)
    checks["thirteen_offsets_per_block"] = bool(groups.size().eq(len(OFFSETS_FT)).all())
    checks["fixed_offset_order"] = all(
        np.array_equal(frame["offset_ft"].to_numpy(np.float64), OFFSETS_FT) for _, frame in groups
    )
    rank_columns = (
        "rsd_rank",
        "raw_pearson_rank",
        "raw_gaussian_rank",
        "permutation_rank",
    )
    expected_ranks = np.arange(1, len(OFFSETS_FT) + 1)
    checks["rank_permutations"] = all(
        np.array_equal(np.sort(frame[column].to_numpy(np.int64)), expected_ranks)
        for _, frame in groups
        for column in rank_columns
    )
    checks["top3_masks_consistent"] = all(
        int(frame[top3_column].sum()) == min(3, int(frame[valid_column].sum()))
        for _, frame in groups
        for valid_column, top3_column in (
            ("rsd_valid", "rsd_top3"),
            ("raw_pearson_valid", "raw_pearson_top3"),
            ("raw_gaussian_valid", "raw_gaussian_top3"),
            ("permutation_valid", "permutation_top3"),
        )
    )
    return checks


def build_target_free_freeze(
    score_bank: pd.DataFrame,
    well_manifest: pd.DataFrame,
    input_manifest: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    runtime_seconds: float,
    peak_memory_gb: float,
    probe_logical_sha_match: bool,
    truth_ledger: TruthAccessLedger,
    strict_inventory: bool = True,
) -> dict[str, Any]:
    structure = validate_score_bank_structure(score_bank)
    block_support = score_bank.groupby(["well_id", "block_id"], sort=False)["rsd_valid"].any()
    supported_block_fraction = float(block_support.mean()) if len(block_support) else 0.0
    supported_wells = block_support.groupby("well_id").any()
    supported_well_fraction = float(supported_wells.mean()) if len(supported_wells) else 0.0
    technical = get_nested(config, "model.stage_a.technical_gate") or {}
    inventory_checks = {
        "expected_rows": int(well_manifest["evaluation_rows"].sum())
        == int(get_nested(config, "validation.expected_rows")),
        "expected_wells": int(score_bank["well_id"].nunique())
        == int(get_nested(config, "validation.expected_wells")),
        "expected_folds": sorted(score_bank["fold"].unique().tolist())
        == [int(value) for value in get_nested(config, "validation.expected_folds")],
    }
    if not strict_inventory:
        inventory_checks = {key: True for key in inventory_checks}
    checks = {
        **structure,
        **inventory_checks,
        "truth_reads_before_freeze_zero": truth_ledger.truth_rows_before_freeze == 0,
        "hidden_role_reads_before_freeze_zero": (truth_ledger.hidden_role_rows_before_freeze == 0),
        "supported_block_fraction": supported_block_fraction
        >= float(technical["minimum_supported_block_fraction"]),
        "supported_well_fraction": supported_well_fraction
        >= float(technical["minimum_supported_well_fraction"]),
        "probe_logical_sha_match": bool(probe_logical_sha_match),
        "runtime": runtime_seconds <= float(technical["maximum_runtime_seconds"]),
        "peak_memory": peak_memory_gb <= float(technical["maximum_peak_rss_gb"]),
    }
    return {
        "technical_passed": bool(all(checks.values())),
        "technical_checks": checks,
        "supported_block_fraction": supported_block_fraction,
        "supported_well_fraction": supported_well_fraction,
        "runtime_seconds_before_truth": float(runtime_seconds),
        "peak_rss_gb_before_truth": float(peak_memory_gb),
        "score_content_sha256": score_bank_logical_sha(score_bank),
        "score_schema_sha256": dataframe_schema_sha(score_bank[SCORE_LOGICAL_COLUMNS]),
        "well_manifest_content_sha256": dataframe_content_sha(well_manifest),
        "input_manifest_content_sha256": dataframe_content_sha(input_manifest),
        "config_content_sha256": mapping_sha256(config),
    }


def rerun_fixed_probe(
    score_bank: pd.DataFrame,
    oof_safe: pd.DataFrame,
    train_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    probe_well = str(get_nested(config, "validation.fixed_probe_well"))
    if probe_well not in set(oof_safe["well_id"]):
        raise ValueError(f"fixed probe well {probe_well} is missing")
    expected = score_bank.loc[score_bank["well_id"].eq(probe_well)].reset_index(drop=True)
    horizontal = load_horizontal_safe(train_root / f"{probe_well}__horizontal_well.csv")
    typewell = pd.read_csv(
        train_root / f"{probe_well}__typewell.csv",
        usecols=["TVT", "GR"],
    )
    observed, _ = score_well_target_free(
        oof_safe.loc[oof_safe["well_id"].eq(probe_well)],
        horizontal,
        typewell,
        config,
    )
    expected_sha = score_bank_logical_sha(expected)
    observed_sha = score_bank_logical_sha(observed)
    return {
        "well_id": probe_well,
        "expected_logical_sha256": expected_sha,
        "rerun_logical_sha256": observed_sha,
        "match": expected_sha == observed_sha,
    }


# %% [markdown]
# ## 8. Post-freeze truth, oracle, scope, and fold readout


# %%
def tie_resolved_minimum_slot(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != OFFSETS_FT.shape or not np.isfinite(values).all():
        raise ValueError("oracle SSE requires 13 finite values")
    tie_priority = tie_priority_by_slot()
    return min(
        range(len(values)),
        key=lambda slot: (float(values[slot]), int(tie_priority[slot])),
    )


def variant_columns(variant: str) -> tuple[str, str, str]:
    mapping = {
        PRIMARY_VARIANT: ("rsd_valid", "rsd_rank", "rsd_top3"),
        "raw_pointwise_pearson": (
            "raw_pearson_valid",
            "raw_pearson_rank",
            "raw_pearson_top3",
        ),
        "raw_gaussian": (
            "raw_gaussian_valid",
            "raw_gaussian_rank",
            "raw_gaussian_top3",
        ),
        "stable_permutation": (
            "permutation_valid",
            "permutation_rank",
            "permutation_top3",
        ),
    }
    return mapping[variant]


def _role_is_active(value: Any) -> bool:
    normalized = str(value).strip().lower()
    return normalized not in {"", "0", "false", "none", "nan", "train"}


def build_post_freeze_block_readout(
    score_bank: pd.DataFrame,
    oof_safe: pd.DataFrame,
    truth: pd.DataFrame,
    hidden_like: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    joined = oof_safe.merge(
        truth,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    if joined["tvt_true"].isna().any() or len(joined) != len(oof_safe):
        raise ValueError("post-freeze truth coverage is incomplete")
    block_size = int(get_nested(config, "model.common_candidate_contract.block_size_rows"))
    joined["block_id"] = joined["suffix_offset"] // block_size
    role_columns = get_nested(config, "data.hidden_like_assignment.role_columns") or {}
    role_by_well = hidden_like.set_index("well_id")
    rows: list[dict[str, Any]] = []

    score_groups = {
        (str(well), int(block)): frame.reset_index(drop=True)
        for (well, block), frame in score_bank.groupby(["well_id", "block_id"], sort=False)
    }
    for (well_id, block_id), block_rows in joined.groupby(["well_id", "block_id"], sort=True):
        key = (str(well_id), int(block_id))
        scores = score_groups[key]
        base = block_rows["tvt_pred"].to_numpy(np.float64)
        actual = block_rows["tvt_true"].to_numpy(np.float64)
        candidate_sse = np.asarray(
            [np.sum(np.square(base + offset - actual)) for offset in OFFSETS_FT],
            dtype=np.float64,
        )
        oracle_slot = tie_resolved_minimum_slot(candidate_sse)
        oracle_offset = float(OFFSETS_FT[oracle_slot])
        metadata = scores.iloc[0]
        hidden_flags = {
            scope: (
                str(well_id) in role_by_well.index
                and _role_is_active(role_by_well.loc[str(well_id), column])
            )
            for scope, column in role_columns.items()
        }
        primary_block_supported = bool(scores["rsd_valid"].any())
        for variant in VARIANTS:
            valid_column, rank_column, top3_column = variant_columns(variant)
            valid = scores[valid_column].to_numpy(bool)
            variant_score_supported = bool(valid.any())
            top1_slots = np.flatnonzero(
                valid & scores[rank_column].to_numpy(np.int64).astype(int).__eq__(1)
            )
            selected_slot = (
                int(top1_slots[0]) if len(top1_slots) else int(np.flatnonzero(OFFSETS_FT == 0.0)[0])
            )
            selected_offset = float(OFFSETS_FT[selected_slot])
            top3_offsets = set(
                scores.loc[scores[top3_column] & scores[valid_column], "offset_ft"]
                .astype(float)
                .tolist()
            )
            parent_sse = float(np.sum(np.square(base - actual)))
            replay_sse = float(np.sum(np.square(base + selected_offset - actual)))
            rows.append(
                {
                    "well_id": str(well_id),
                    "fold": int(block_rows["fold"].iloc[0]),
                    "block_id": int(block_id),
                    "variant": variant,
                    # All variants use the primary RSD support mask so matched
                    # comparisons have the same block denominator.
                    "supported": primary_block_supported,
                    "variant_score_supported": variant_score_supported,
                    "row_count": len(block_rows),
                    "md_since_min_ft": float(metadata["md_since_min_ft"]),
                    "md_since_max_ft": float(metadata["md_since_max_ft"]),
                    "md_since_mid_ft": float(metadata["md_since_mid_ft"]),
                    "observed_gr_share": float(metadata["observed_gr_share"]),
                    "oracle_offset_ft": oracle_offset,
                    "selected_offset_ft": selected_offset,
                    "top1_exact": selected_slot == oracle_slot,
                    "top3_coverage": oracle_offset in top3_offsets,
                    "nonzero_oracle": oracle_offset != 0.0,
                    "direction_correct": (
                        oracle_offset != 0.0 and np.sign(selected_offset) == np.sign(oracle_offset)
                    ),
                    "parent_sse": parent_sse,
                    "replay_sse": replay_sse,
                    **hidden_flags,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["variant", "well_id", "block_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def scope_mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "pooled":
        return np.ones(len(frame), dtype=bool)
    if scope == "distance_0_50":
        return frame["md_since_mid_ft"].to_numpy(np.float64) < 50.0
    if scope == "distance_50_100":
        values = frame["md_since_mid_ft"].to_numpy(np.float64)
        return (values >= 50.0) & (values < 100.0)
    if scope == "distance_1000_plus":
        return frame["md_since_mid_ft"].to_numpy(np.float64) >= 1000.0
    if scope == "raw_gr_observed":
        return frame["observed_gr_share"].to_numpy(np.float64) == 1.0
    if scope == "raw_gr_missing":
        return frame["observed_gr_share"].to_numpy(np.float64) < 1.0
    if scope == "high_missing":
        return frame["observed_gr_share"].to_numpy(np.float64) < 0.5
    if scope in {"hidden_like_spatial", "hidden_like_typewell_purged"}:
        return frame[scope].to_numpy(bool)
    if scope == "by_well":
        return np.ones(len(frame), dtype=bool)
    raise ValueError(f"unknown report scope {scope}")


def summarize_readout(frame: pd.DataFrame, *, scope: str) -> dict[str, Any]:
    scoped = frame.loc[scope_mask(frame, scope)]
    identified = scoped.loc[scoped["supported"]]
    nonzero = identified.loc[identified["nonzero_oracle"]]
    # Unsupported blocks use the fixed zero-offset fallback, so replay RMSE is
    # evaluated over the complete scope while identifiability rates use only
    # blocks where that variant has a valid score.
    rows = int(scoped["row_count"].sum())
    parent_sse = float(scoped["parent_sse"].sum())
    replay_sse = float(scoped["replay_sse"].sum())
    parent_rmse = math.sqrt(parent_sse / rows) if rows else None
    replay_rmse = math.sqrt(replay_sse / rows) if rows else None
    return {
        "scope": scope,
        "blocks": len(scoped),
        "supported_blocks": len(identified),
        "rows": rows,
        "wells": int(scoped["well_id"].nunique()),
        "top1_exact": (float(identified["top1_exact"].mean()) if len(identified) else None),
        "top3_coverage": (float(identified["top3_coverage"].mean()) if len(identified) else None),
        "nonzero_oracle_blocks": len(nonzero),
        "direction_accuracy": (
            float(nonzero["direction_correct"].mean()) if len(nonzero) else None
        ),
        "parent_rmse": parent_rmse,
        "replay_rmse": replay_rmse,
        "replay_gain_ft": (
            float(parent_rmse - replay_rmse)
            if parent_rmse is not None and replay_rmse is not None
            else None
        ),
    }


def build_scope_and_fold_metrics(
    readout: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scope_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    scopes = [str(value) for value in get_nested(config, "validation.report_scopes")]
    for variant in VARIANTS:
        variant_rows = readout.loc[readout["variant"].eq(variant)]
        for scope in scopes:
            if scope == "by_well":
                continue
            scope_rows.append({"variant": variant, **summarize_readout(variant_rows, scope=scope)})
        for fold, fold_frame in variant_rows.groupby("fold", sort=True):
            fold_rows.append(
                {
                    "variant": variant,
                    "fold": int(fold),
                    **summarize_readout(fold_frame, scope="pooled"),
                }
            )
        for well_id, well_frame in variant_rows.groupby("well_id", sort=True):
            well_rows.append(
                {
                    "variant": variant,
                    "well_id": str(well_id),
                    "fold": int(well_frame["fold"].iloc[0]),
                    **summarize_readout(well_frame, scope="pooled"),
                }
            )
    return (
        pd.DataFrame(scope_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(well_rows),
    )


# %% [markdown]
# ## 9. Technical and scientific gates


# %%
def _metric_row(
    frame: pd.DataFrame,
    *,
    variant: str,
    scope: str | None = None,
    fold: int | None = None,
) -> pd.Series:
    mask = frame["variant"].eq(variant)
    if scope is not None:
        mask &= frame["scope"].eq(scope)
    if fold is not None:
        mask &= frame["fold"].eq(fold)
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(
            f"expected one metric row for variant={variant}, scope={scope}, fold={fold}"
        )
    return selected.iloc[0]


def finite_metric(value: Any) -> float:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return float("-inf")
    return float(value)


def evaluate_stage_a_gate(
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    freeze: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = get_nested(config, "model.stage_a.scientific_gate") or {}
    primary = _metric_row(scope_metrics, variant=PRIMARY_VARIANT, scope="pooled")
    fold_direction_count = sum(
        finite_metric(
            _metric_row(fold_metrics, variant=PRIMARY_VARIANT, fold=fold)["direction_accuracy"]
        )
        > 0.5
        for fold in range(5)
    )
    control_checks: dict[str, Any] = {}
    for control, config_key in (
        ("raw_pointwise_pearson", "minimum_top1_gain_vs_raw_pearson"),
        ("raw_gaussian", "minimum_top1_gain_vs_raw_gaussian"),
        ("stable_permutation", "minimum_top1_gain_vs_permutation"),
    ):
        control_pooled = _metric_row(scope_metrics, variant=control, scope="pooled")
        gain = finite_metric(primary["top1_exact"]) - finite_metric(control_pooled["top1_exact"])
        improving_folds = sum(
            finite_metric(
                _metric_row(fold_metrics, variant=PRIMARY_VARIANT, fold=fold)["top1_exact"]
            )
            > finite_metric(_metric_row(fold_metrics, variant=control, fold=fold)["top1_exact"])
            for fold in range(5)
        )
        control_checks[control] = {
            "top1_gain": gain,
            "minimum_gain": float(gate[config_key]),
            "improving_folds": improving_folds,
            "passed": gain >= float(gate[config_key])
            and improving_folds >= int(gate["minimum_control_improvement_folds"]),
        }

    scoped_checks: dict[str, Any] = {}
    for scope in gate["scoped_direction_must_exceed_half"]:
        row = _metric_row(scope_metrics, variant=PRIMARY_VARIANT, scope=str(scope))
        direction = finite_metric(row["direction_accuracy"])
        gain = finite_metric(row["replay_gain_ft"])
        scoped_checks[str(scope)] = {
            "direction_accuracy": direction,
            "replay_gain_ft": gain,
            "passed": direction > 0.5 and gain >= -float(gate["maximum_scoped_rmse_regression_ft"]),
        }

    checks = {
        "technical_gate": bool(freeze["technical_passed"]),
        "parent_rmse_matches_exp226": abs(
            finite_metric(primary["parent_rmse"])
            - float(get_nested(config, "data.exp226.expected_rmse"))
        )
        <= float(get_nested(config, "validation.parent_rmse_atol_ft")),
        "minimum_top1_discrete_oracle_exact": finite_metric(primary["top1_exact"])
        >= float(gate["minimum_top1_discrete_oracle_exact"]),
        "minimum_top3_oracle_coverage": finite_metric(primary["top3_coverage"])
        >= float(gate["minimum_top3_oracle_coverage"]),
        "minimum_nonzero_oracle_direction_accuracy": finite_metric(primary["direction_accuracy"])
        >= float(gate["minimum_nonzero_oracle_direction_accuracy"]),
        "minimum_direction_folds_above_half": fold_direction_count
        >= int(gate["minimum_direction_folds_above_half"]),
        "matched_controls": all(item["passed"] for item in control_checks.values()),
        "minimum_independent_top1_rmse_gain_ft": finite_metric(primary["replay_gain_ft"])
        >= float(gate["minimum_independent_top1_rmse_gain_ft"]),
        "minimum_independent_top1_improvement_folds": sum(
            finite_metric(
                _metric_row(fold_metrics, variant=PRIMARY_VARIANT, fold=fold)["replay_gain_ft"]
            )
            > 0.0
            for fold in range(5)
        )
        >= int(gate["minimum_independent_top1_improvement_folds"]),
        "scoped_tail_and_hidden_like": all(item["passed"] for item in scoped_checks.values()),
    }
    passed = bool(all(checks.values()))
    result = {
        "passed": passed,
        "checks": checks,
        "pooled_primary": {
            "top1_exact": finite_metric(primary["top1_exact"]),
            "top3_coverage": finite_metric(primary["top3_coverage"]),
            "direction_accuracy": finite_metric(primary["direction_accuracy"]),
            "parent_rmse": finite_metric(primary["parent_rmse"]),
            "replay_rmse": finite_metric(primary["replay_rmse"]),
            "replay_gain_ft": finite_metric(primary["replay_gain_ft"]),
        },
        "direction_folds_above_half": fold_direction_count,
        "matched_control_checks": control_checks,
        "scoped_checks": scoped_checks,
    }
    decision = {
        "action": (
            "request_separate_stage_b_implementation_authorization"
            if passed
            else "close_stage_a_without_score_bin_block_offset_or_support_rescue"
        ),
        "stage_b_implemented": False,
        "stage_c_implemented": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    return result, decision


# %% [markdown]
# ## 10. Metrics and generated artifacts


# %%
def save_final_artifacts(
    *,
    score_bank: pd.DataFrame,
    well_manifest: pd.DataFrame,
    input_manifest: pd.DataFrame,
    block_readout: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    by_well_metrics: pd.DataFrame,
    freeze: Mapping[str, Any],
    probe: Mapping[str, Any],
    gate: Mapping[str, Any],
    decision: Mapping[str, Any],
    input_evidence: Sequence[Mapping[str, Any]],
    scientific_contract: Mapping[str, Any],
    ledger: TruthAccessLedger,
    elapsed_seconds: float,
) -> dict[str, Any]:
    output = artifact_dir()
    artifact_evidence = {
        "target_free_score_bank": write_csv_gzip(
            score_bank,
            output / f"{OUTPUT_PREFIX}_stage_a_target_free_score_bank.csv.gz",
        ),
        "well_manifest": write_csv_gzip(
            well_manifest,
            output / f"{OUTPUT_PREFIX}_stage_a_well_manifest.csv.gz",
        ),
        "input_manifest": write_csv_gzip(
            input_manifest,
            output / f"{OUTPUT_PREFIX}_stage_a_input_manifest.csv.gz",
        ),
        "block_readout": write_csv_gzip(
            block_readout,
            output / f"{OUTPUT_PREFIX}_stage_a_block_readout.csv.gz",
        ),
    }
    scope_path = output / f"{OUTPUT_PREFIX}_stage_a_scope_metrics.csv"
    fold_path = output / f"{OUTPUT_PREFIX}_stage_a_fold_metrics.csv"
    well_path = output / f"{OUTPUT_PREFIX}_stage_a_by_well_metrics.csv"
    scope_metrics.to_csv(scope_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    by_well_metrics.to_csv(well_path, index=False)
    artifact_evidence["scope_metrics"] = {
        "path": str(scope_path),
        "rows": len(scope_metrics),
        "raw_sha256": sha256_path(scope_path),
        "content_sha256": dataframe_content_sha(scope_metrics),
    }
    artifact_evidence["fold_metrics"] = {
        "path": str(fold_path),
        "rows": len(fold_metrics),
        "raw_sha256": sha256_path(fold_path),
        "content_sha256": dataframe_content_sha(fold_metrics),
    }
    artifact_evidence["by_well_metrics"] = {
        "path": str(well_path),
        "rows": len(by_well_metrics),
        "raw_sha256": sha256_path(well_path),
        "content_sha256": dataframe_content_sha(by_well_metrics),
    }

    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_a_absolute_datum_identifiability",
        "status": "stage_a_passed" if gate["passed"] else "stage_a_failed_closed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scientific_contract": scientific_contract,
        "input_evidence": list(input_evidence),
        "target_free_freeze": freeze,
        "probe_rerun": probe,
        "truth_access_ledger": {
            "truth_rows_before_freeze": ledger.truth_rows_before_freeze,
            "truth_rows_after_freeze": ledger.truth_rows_after_freeze,
            "hidden_role_rows_before_freeze": ledger.hidden_role_rows_before_freeze,
            "hidden_role_rows_after_freeze": ledger.hidden_role_rows_after_freeze,
            "frozen_score_content_sha256": ledger.frozen_score_content_sha256,
        },
        "stage_a_gate": gate,
        "decision": decision,
        "elapsed_seconds": float(elapsed_seconds),
        "peak_rss_gb": peak_rss_gb(),
        "artifacts": artifact_evidence,
    }
    summary_path = output / f"{OUTPUT_PREFIX}_stage_a_summary.json"
    write_json(summary_path, summary)
    summary["artifacts"]["summary"] = {
        "path": str(summary_path),
        "raw_sha256": sha256_path(summary_path),
    }
    metrics = {
        "status": summary["status"],
        "stage": summary["stage"],
        "route": "pf_beam",
        "metric": "absolute_offset_identifiability_and_blockwise_replay_rmse",
        "stage_a_gate": gate,
        "decision": decision,
        "target_free_freeze": freeze,
        "probe_rerun": probe,
        "truth_access_ledger": summary["truth_access_ledger"],
        "elapsed_seconds": float(elapsed_seconds),
        "artifacts": summary["artifacts"],
    }
    write_json(metrics_output_path(), metrics)
    return metrics


def save_technical_failure_artifacts(
    *,
    score_bank: pd.DataFrame,
    well_manifest: pd.DataFrame,
    input_manifest: pd.DataFrame,
    freeze: Mapping[str, Any],
    probe: Mapping[str, Any],
    exp226_evidence: Mapping[str, Any],
    scientific_contract: Mapping[str, Any],
    ledger: TruthAccessLedger,
    elapsed_seconds: float,
) -> dict[str, Any]:
    output = artifact_dir()
    artifacts = {
        "target_free_score_bank": write_csv_gzip(
            score_bank,
            output / f"{OUTPUT_PREFIX}_stage_a_target_free_score_bank.csv.gz",
        ),
        "well_manifest": write_csv_gzip(
            well_manifest,
            output / f"{OUTPUT_PREFIX}_stage_a_well_manifest.csv.gz",
        ),
        "input_manifest": write_csv_gzip(
            input_manifest,
            output / f"{OUTPUT_PREFIX}_stage_a_input_manifest.csv.gz",
        ),
    }
    decision = {
        "action": "close_stage_a_on_technical_gate_without_truth_read_or_rescue",
        "stage_b_implemented": False,
        "stage_c_implemented": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_a_absolute_datum_identifiability",
        "status": "stage_a_technical_failed_closed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scientific_contract": scientific_contract,
        "input_evidence": [exp226_evidence],
        "target_free_freeze": freeze,
        "probe_rerun": probe,
        "truth_access_ledger": {
            "truth_rows_before_freeze": ledger.truth_rows_before_freeze,
            "truth_rows_after_freeze": ledger.truth_rows_after_freeze,
            "hidden_role_rows_before_freeze": ledger.hidden_role_rows_before_freeze,
            "hidden_role_rows_after_freeze": ledger.hidden_role_rows_after_freeze,
        },
        "stage_a_gate": {
            "passed": False,
            "technical_passed": False,
            "scientific_evaluation_skipped": True,
        },
        "decision": decision,
        "elapsed_seconds": float(elapsed_seconds),
        "peak_rss_gb": peak_rss_gb(),
        "artifacts": artifacts,
    }
    summary_path = output / f"{OUTPUT_PREFIX}_stage_a_summary.json"
    write_json(summary_path, summary)
    artifacts["summary"] = {
        "path": str(summary_path),
        "raw_sha256": sha256_path(summary_path),
    }
    metrics = {
        "status": summary["status"],
        "stage": summary["stage"],
        "route": "pf_beam",
        "stage_a_gate": summary["stage_a_gate"],
        "decision": decision,
        "target_free_freeze": freeze,
        "probe_rerun": probe,
        "truth_access_ledger": summary["truth_access_ledger"],
        "elapsed_seconds": float(elapsed_seconds),
        "artifacts": artifacts,
    }
    write_json(metrics_output_path(), metrics)
    return metrics


def run_stage_a_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    scientific_contract = validate_scientific_contract(
        config,
        require_run_approval=True,
    )
    started = time.perf_counter()
    ledger = TruthAccessLedger()
    oof_safe, exp226_path, exp226_evidence = load_exp226_safe(config)
    train_root = resolve_train_root(config)
    score_bank, well_manifest, raw_input_rows = generate_target_free_score_bank(
        oof_safe,
        train_root,
        config,
    )
    probe = rerun_fixed_probe(score_bank, oof_safe, train_root, config)
    input_manifest = pd.DataFrame(raw_input_rows).sort_values("well_id", kind="mergesort")
    before_truth_runtime = time.perf_counter() - started
    freeze = build_target_free_freeze(
        score_bank,
        well_manifest,
        input_manifest,
        config=config,
        runtime_seconds=before_truth_runtime,
        peak_memory_gb=peak_rss_gb(),
        probe_logical_sha_match=bool(probe["match"]),
        truth_ledger=ledger,
    )
    if not freeze["technical_passed"]:
        return save_technical_failure_artifacts(
            score_bank=score_bank,
            well_manifest=well_manifest,
            input_manifest=input_manifest,
            freeze=freeze,
            probe=probe,
            exp226_evidence=exp226_evidence,
            scientific_contract=scientific_contract,
            ledger=ledger,
            elapsed_seconds=time.perf_counter() - started,
        )
    ledger.mark_frozen(str(freeze["score_content_sha256"]))

    truth = load_exp226_truth(exp226_path, ledger)
    hidden_like, hidden_evidence = load_hidden_like_assignments(config, ledger)
    block_readout = build_post_freeze_block_readout(
        score_bank,
        oof_safe,
        truth,
        hidden_like,
        config,
    )
    scope_metrics, fold_metrics, by_well_metrics = build_scope_and_fold_metrics(
        block_readout,
        config,
    )
    gate, decision = evaluate_stage_a_gate(
        scope_metrics,
        fold_metrics,
        freeze,
        config,
    )
    return save_final_artifacts(
        score_bank=score_bank,
        well_manifest=well_manifest,
        input_manifest=input_manifest,
        block_readout=block_readout,
        scope_metrics=scope_metrics,
        fold_metrics=fold_metrics,
        by_well_metrics=by_well_metrics,
        freeze=freeze,
        probe=probe,
        gate=gate,
        decision=decision,
        input_evidence=[
            exp226_evidence,
            hidden_evidence,
            {
                "name": "raw_train_horizontal_and_typewell_files",
                "root": str(train_root),
                "wells": len(input_manifest),
                "logical_content_sha256": dataframe_content_sha(input_manifest),
            },
        ],
        scientific_contract=scientific_contract,
        ledger=ledger,
        elapsed_seconds=time.perf_counter() - started,
    )


# %% [markdown]
# ## 11. Setup, configuration preview, and approved execution


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    CONTRACT = validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            to_jsonable(
                {
                    "experiment": get_nested(CONFIG, "experiment"),
                    "lineage": get_nested(CONFIG, "lineage"),
                    "implementation": get_nested(CONFIG, "implementation"),
                    "execution_contract": get_nested(CONFIG, "execution_contract"),
                    "execution": get_nested(CONFIG, "execution"),
                    "stage_a": get_nested(CONFIG, "model.stage_a"),
                    "scientific_contract": CONTRACT,
                }
            ),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    display(
        pd.DataFrame(
            {
                "variant": VARIANTS,
                "role": ["primary", "matched control", "matched control", "negative control"],
                "model_configs": [0, 0, 0, 0],
                "boosters": [0, 0, 0, 0],
                "pf_runs": [0, 0, 0, 0],
            }
        )
    )


# %% [markdown]
# The execution guard below requires all four explicit Stage A Kaggle
# package/push/run flags.  The implementation-ready repository state keeps
# those flags false, so importing or converting this source cannot start work.


# %%
if EXECUTE_NOTEBOOK:
    METRICS = run_stage_a_experiment(CONFIG)
    print(json.dumps(to_jsonable(METRICS), indent=2, sort_keys=True, ensure_ascii=False))

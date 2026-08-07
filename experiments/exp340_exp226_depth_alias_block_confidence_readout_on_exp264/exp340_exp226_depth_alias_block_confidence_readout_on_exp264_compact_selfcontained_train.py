# %% [markdown]
# # exp340 exp226 depth-alias block confidence readout on exp264
#
# Stage 0 is a deterministic, zero-booster diagnostic. It consumes the frozen
# exp280 13-shift likelihood bank, creates seven preregistered target-free block
# risk families, freezes their fold-wise quartile boundaries and content SHA,
# and only then attaches saved exp264/exp226 OOF errors. It never changes a
# candidate, prediction, selector, decoder, HMM, inference path, or submission.

# %% [markdown]
# ## Contents
# 1. Imports and execution guard
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific and execution contract
# 4. Exp280 target-free score and contract loader
# 5. Seven depth-alias risk families and circular control
# 6. Fold-wise quantile and target-free freeze
# 7. Post-freeze exp264/exp226 truth and hidden-like loaders
# 8. Block RMSE, row-weighted AUC, scopes, and fixed gate
# 9. Metrics and generated artifacts
# 10. Setup and configuration preview
# 11. Run the approved Stage 0 readout only

# %%
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from IPython.display import display

EXPERIMENT_NAME = "exp340_exp226_depth_alias_block_confidence_readout_on_exp264"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
EXPECTED_SHIFTS = np.asarray(
    [-80.0, -40.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    dtype=np.float64,
)
FAMILIES = (
    "top1_top2_margin",
    "softmax_entropy",
    "likelihood_weighted_shift_std",
    "zero_shift_rank",
    "absolute_top1_shift",
    "top1_shift_jump_from_previous_block",
    "three_block_sign_inconsistency",
)
SEQUENCE_FAMILIES = (
    "top1_shift_jump_from_previous_block",
    "three_block_sign_inconsistency",
)
FORBIDDEN_PRE_FREEZE_COLUMNS = {
    "TVT",
    "tvt_true",
    "actual_tvt",
    "target",
    "error",
    "abs_error",
    "tvt_pred",
    "prediction",
    "selector_compact_addonly__lgb_mean__pred_tvt",
}


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP340_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


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
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "experiments").is_dir():
            return candidate
    return start


def experiment_dir() -> Path:
    candidate = project_root() / "experiments" / EXPERIMENT_NAME
    return candidate if candidate.exists() else Path.cwd().resolve()


def load_experiment_config() -> dict[str, Any]:
    candidates = (
        Path.cwd() / "config.yaml",
        experiment_dir() / "config.yaml",
    )
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp340 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
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
        to_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_frame_for_hash(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column, dtype in normalized.dtypes.items():
        if isinstance(dtype, pd.StringDtype):
            normalized[column] = normalized[column].astype(object)
    return normalized


def dataframe_content_sha(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
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
    raise FileNotFoundError(
        f"Could not resolve {label} with its fixed SHA contract: {evidence[:8]}"
    )


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
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
    }


def stable_uint64(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


# %% [markdown]
# ## 3. Frozen scientific and execution contract


# %%
def assert_no_forbidden_columns(columns: Iterable[str]) -> None:
    present = set(map(str, columns)).intersection(FORBIDDEN_PRE_FREEZE_COLUMNS)
    if present:
        raise ValueError(f"truth/error columns are forbidden before freeze: {sorted(present)}")


def validate_scientific_contract(
    config: Mapping[str, Any], *, require_run_approval: bool = False
) -> None:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("unexpected experiment name")
    if get_nested(config, "experiment.route") != "ensemble":
        raise ValueError("exp340 route must remain ensemble")
    if not bool(get_nested(config, "implementation.enabled")):
        raise ValueError("exp340 Stage 0 implementation must be enabled")
    if tuple(get_nested(config, "model.target_free_families") or ()) != FAMILIES:
        raise ValueError("the seven target-free family contract changed")
    shifts = np.asarray(get_nested(config, "data.shifts_ft"), dtype=np.float64)
    if not np.array_equal(shifts, EXPECTED_SHIFTS):
        raise ValueError("the fixed 13-shift bank changed")
    if int(get_nested(config, "data.block_size")) != 512:
        raise ValueError("exp340 requires fixed non-overlapping H512 blocks")
    if list(get_nested(config, "validation.expected_folds")) != [0, 1, 2, 3, 4]:
        raise ValueError("the five-fold readout contract changed")
    counts = get_nested(config, "execution_contract") or {}
    expected_counts = {
        "readout_families": 7,
        "controls": 1,
        "models": 0,
        "hmm_well_runs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
    }
    if counts != expected_counts:
        raise ValueError(f"zero-booster execution contract changed: {counts}")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise ValueError("Stage 0 implementation is not approved")
    forbidden_true = (
        "execution.run_inference",
        "execution.create_submission",
        "inference.enabled",
        "inference.create_submission",
        "implementation.inference_enabled",
        "implementation.submission_enabled",
    )
    if any(bool(get_nested(config, key)) for key in forbidden_true):
        raise ValueError("inference and submission must remain disabled")
    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.run_stage_0"))
        and bool(get_nested(config, "runtime.kaggle.train_run_on_push"))
    ):
        raise RuntimeError("exp340 Kaggle package/push/run is not approved")


class TruthAccessLedger:
    def __init__(self) -> None:
        self.frozen = False
        self.count_before_freeze = 0

    def mark_frozen(self) -> None:
        self.frozen = True

    def register_truth_access(self) -> None:
        if not self.frozen:
            self.count_before_freeze += 1
            raise ValueError("truth access attempted before target-free freeze")


# %% [markdown]
# ## 4. Exp280 target-free score and contract loader


# %%
def load_exp280_target_free_scores(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    spec = get_nested(config, "data.exp280_source")
    score_path, score_evidence = resolve_file(
        spec["score_patterns"],
        label="exp280 target-free shift score",
        expected_decompressed_sha256=str(spec["score_decompressed_sha256"]),
    )
    contract_path, contract_evidence = resolve_file(
        spec["contract_patterns"],
        label="exp280 score contract",
    )
    contract = json.loads(contract_path.read_text())
    if bool(contract.get("truth_attached")):
        raise ValueError("exp280 score contract must be truth-free")
    if contract.get("target_free_score_content_sha256") != str(spec["score_content_sha256"]):
        raise ValueError("exp280 target-free score content declaration changed")
    if contract.get("scientific_contract_sha256") != str(spec["scientific_contract_sha256"]):
        raise ValueError("exp280 scientific contract SHA changed")
    if list(map(float, contract.get("shift_bank_ft", []))) != EXPECTED_SHIFTS.tolist():
        raise ValueError("exp280 score contract shift bank changed")
    if int(contract.get("block_rows", -1)) != 512:
        raise ValueError("exp280 block contract changed")

    scores = pd.read_csv(score_path, dtype={"well_id": str})
    assert_no_forbidden_columns(scores.columns)
    required = {
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
        "observed_gr_share",
        "shift_slot",
        "shift_ft",
        "likelihood_mean",
        "likelihood_rank",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"exp280 score table missing {missing}")
    integer_columns = (
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "block_start_row_idx",
        "block_end_row_idx",
        "block_row_count",
        "shift_slot",
        "likelihood_rank",
    )
    for column in integer_columns:
        scores[column] = pd.to_numeric(scores[column], errors="raise").astype(np.int64)
    numeric_columns = (
        "md_since_min_ft",
        "md_since_max_ft",
        "md_since_mid_ft",
        "observed_gr_share",
        "shift_ft",
        "likelihood_mean",
    )
    for column in numeric_columns:
        scores[column] = pd.to_numeric(scores[column], errors="raise").astype(np.float64)
    scores["well_id"] = scores["well_id"].astype(str)
    scores = scores.sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    ).reset_index(drop=True)

    expected_blocks = int(get_nested(config, "validation.expected_blocks"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(scores) != expected_blocks * len(EXPECTED_SHIFTS):
        raise ValueError("exp280 score row count changed")
    if scores["well_id"].nunique() != expected_wells:
        raise ValueError("exp280 score well count changed")
    group_size = scores.groupby(["well_id", "block_id"], sort=False).size()
    if not group_size.eq(len(EXPECTED_SHIFTS)).all() or len(group_size) != expected_blocks:
        raise ValueError("each exp280 block must contain exactly 13 shifts")
    observed_shifts = scores["shift_ft"].to_numpy(np.float64).reshape(
        expected_blocks, len(EXPECTED_SHIFTS)
    )
    expected_matrix = np.broadcast_to(EXPECTED_SHIFTS, observed_shifts.shape)
    if not np.array_equal(observed_shifts, expected_matrix):
        raise ValueError("exp280 shift order changed")
    if not np.isfinite(scores["likelihood_mean"].to_numpy(np.float64)).all():
        raise ValueError("exp280 likelihood contains non-finite values")
    return scores, [
        {"name": "exp280_target_free_shift_scores", **score_evidence},
        {
            "name": "exp280_score_contract",
            **contract_evidence,
            "declared_target_free_content_sha256": contract[
                "target_free_score_content_sha256"
            ],
            "scientific_contract_sha256": contract["scientific_contract_sha256"],
        },
    ]


# %% [markdown]
# ## 5. Seven depth-alias risk families and circular control


# %%
def pairwise_sign_inconsistency(values: np.ndarray) -> np.ndarray:
    shifts = np.asarray(values, dtype=np.float64)
    output = np.zeros(len(shifts), dtype=np.float64)
    signs = np.sign(shifts).astype(np.int8)
    for index in range(len(signs)):
        window = signs[max(0, index - 2) : index + 1]
        nonzero = window[window != 0]
        if len(nonzero) < 2:
            output[index] = 0.0
            continue
        disagreements = 0
        pairs = 0
        for left in range(len(nonzero)):
            for right in range(left + 1, len(nonzero)):
                pairs += 1
                disagreements += int(nonzero[left] != nonzero[right])
        output[index] = disagreements / pairs
    return output


def stable_nonzero_rotation(well_id: str, block_count: int) -> int:
    if block_count < 2:
        return 0
    return 1 + stable_uint64(EXPERIMENT_NAME, "circular_control", well_id) % (
        block_count - 1
    )


def sequence_features(top1_shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(top1_shift, dtype=np.float64)
    jump = np.zeros(len(values), dtype=np.float64)
    if len(values) > 1:
        jump[1:] = np.abs(np.diff(values))
    inconsistency = pairwise_sign_inconsistency(values)
    return jump, inconsistency


def build_target_free_block_features(scores: pd.DataFrame) -> pd.DataFrame:
    assert_no_forbidden_columns(scores.columns)
    ordered = scores.sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    ).reset_index(drop=True)
    group_columns = ["well_id", "block_id"]
    block_count = ordered.groupby(group_columns, sort=False).ngroups
    likelihood = ordered["likelihood_mean"].to_numpy(np.float64).reshape(
        block_count, len(EXPECTED_SHIFTS)
    )
    likelihood_ordered = np.sort(likelihood, axis=1)[:, ::-1]
    top_slot = np.argmax(likelihood, axis=1)
    top_shift = EXPECTED_SHIFTS[top_slot]
    centered = likelihood - likelihood.max(axis=1, keepdims=True)
    weights = np.exp(centered)
    weights /= weights.sum(axis=1, keepdims=True)
    entropy = -np.sum(weights * np.log(np.clip(weights, 1e-300, None)), axis=1)
    weighted_mean = np.sum(weights * EXPECTED_SHIFTS[None, :], axis=1)
    weighted_std = np.sqrt(
        np.sum(weights * np.square(EXPECTED_SHIFTS[None, :] - weighted_mean[:, None]), axis=1)
    )
    zero_slot = int(np.flatnonzero(EXPECTED_SHIFTS == 0.0)[0])
    zero_rank = (
        ordered["likelihood_rank"]
        .to_numpy(np.int64)
        .reshape(block_count, len(EXPECTED_SHIFTS))[:, zero_slot]
        .astype(np.float64)
        - 1.0
    ) / (len(EXPECTED_SHIFTS) - 1)

    first = ordered.groupby(group_columns, sort=False, as_index=False).first()
    features = first[
        [
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
            "observed_gr_share",
        ]
    ].copy()
    features["top1_shift_ft"] = top_shift
    features["top1_top2_margin"] = likelihood_ordered[:, 0] - likelihood_ordered[:, 1]
    features["softmax_entropy"] = entropy
    features["likelihood_weighted_shift_std"] = weighted_std
    features["zero_shift_rank"] = zero_rank
    features["absolute_top1_shift"] = np.abs(top_shift)
    features["top1_shift_jump_from_previous_block"] = 0.0
    features["three_block_sign_inconsistency"] = 0.0
    features["control__top1_shift_jump_from_previous_block"] = np.nan
    features["control__three_block_sign_inconsistency"] = np.nan

    for well, positions in features.groupby("well_id", sort=True).indices.items():
        position_array = np.asarray(positions, dtype=np.int64)
        position_array = position_array[
            np.argsort(features.iloc[position_array]["block_id"].to_numpy(), kind="mergesort")
        ]
        real_shifts = features.iloc[position_array]["top1_shift_ft"].to_numpy(np.float64)
        jump, inconsistency = sequence_features(real_shifts)
        rotation = stable_nonzero_rotation(str(well), len(position_array))
        control_shifts = np.roll(real_shifts, rotation)
        control_jump, control_inconsistency = sequence_features(control_shifts)
        features.loc[position_array, "top1_shift_jump_from_previous_block"] = jump
        features.loc[position_array, "three_block_sign_inconsistency"] = inconsistency
        features.loc[
            position_array, "control__top1_shift_jump_from_previous_block"
        ] = control_jump
        features.loc[
            position_array, "control__three_block_sign_inconsistency"
        ] = control_inconsistency

    for family in FAMILIES:
        raw = features[family].to_numpy(np.float64)
        risk = -raw if family == "top1_top2_margin" else raw
        features[f"risk__{family}"] = risk
    for family in SEQUENCE_FAMILIES:
        features[f"control_risk__{family}"] = features[f"control__{family}"].to_numpy(
            np.float64
        )
    required_finite = [f"risk__{family}" for family in FAMILIES]
    if not np.isfinite(features[required_finite].to_numpy(np.float64)).all():
        raise ValueError("target-free real risk families must be finite")
    control_columns = [f"control_risk__{family}" for family in SEQUENCE_FAMILIES]
    if not np.isfinite(features[control_columns].to_numpy(np.float64)).all():
        raise ValueError("target-free circular controls must be finite")
    return features.sort_values(["well_id", "block_id"], kind="mergesort").reset_index(drop=True)


# %% [markdown]
# ## 6. Fold-wise quantile and target-free freeze


# %%
def fit_fold_quantile_boundaries(
    features: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    assert_no_forbidden_columns(features.columns)
    quantiles = list(map(float, get_nested(config, "model.attribution.quantiles")))
    if quantiles != [0.25, 0.75]:
        raise ValueError("exp340 fixes Q1/Q4 at 0.25/0.75")
    rows: list[dict[str, Any]] = []
    for fold, part in features.groupby("fold", sort=True):
        for family in FAMILIES:
            values = part[f"risk__{family}"].to_numpy(np.float64)
            rows.append(
                {
                    "fold": int(fold),
                    "family": family,
                    "q25_risk_boundary": float(np.quantile(values, 0.25)),
                    "q75_risk_boundary": float(np.quantile(values, 0.75)),
                    "blocks": len(part),
                    "finite_coverage": float(np.isfinite(values).mean()),
                    "risk_direction": "higher_is_more_alias_risk",
                    "control_applicable": family in SEQUENCE_FAMILIES,
                }
            )
    output = pd.DataFrame(rows).sort_values(["family", "fold"], kind="mergesort")
    if len(output) != len(FAMILIES) * 5:
        raise ValueError("fold-wise family quantile coverage changed")
    return output.reset_index(drop=True)


def attach_frozen_quartile_flags(
    features: pd.DataFrame, boundaries: pd.DataFrame
) -> pd.DataFrame:
    output = features.copy()
    for family in FAMILIES:
        selected = boundaries.loc[boundaries["family"].eq(family)].set_index("fold")
        q25 = output["fold"].map(selected["q25_risk_boundary"]).to_numpy(np.float64)
        q75 = output["fold"].map(selected["q75_risk_boundary"]).to_numpy(np.float64)
        risk = output[f"risk__{family}"].to_numpy(np.float64)
        output[f"q1__{family}"] = risk <= q25
        output[f"q4__{family}"] = risk >= q75
    return output


def freeze_target_free_bundle(
    features: pd.DataFrame,
    boundaries: pd.DataFrame,
    input_manifest: list[dict[str, Any]],
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> dict[str, Any]:
    assert_no_forbidden_columns(features.columns)
    artifacts = artifact_dir()
    feature_path = artifacts / f"{OUTPUT_PREFIX}_target_free_block_features.parquet"
    quantile_path = artifacts / f"{OUTPUT_PREFIX}_fold_quantile_boundaries.csv"
    schema_path = artifacts / f"{OUTPUT_PREFIX}_feature_schema.json"
    input_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.json"
    contract_path = artifacts / f"{OUTPUT_PREFIX}_contract.json"
    freeze_path = artifacts / f"{OUTPUT_PREFIX}_freeze_manifest.json"
    features.to_parquet(feature_path, index=False, compression="zstd")
    boundaries.to_csv(quantile_path, index=False)
    schema = {
        "columns": [
            {"name": column, "dtype": str(dtype)}
            for column, dtype in features.dtypes.items()
        ],
        "families": list(FAMILIES),
        "sequence_families": list(SEQUENCE_FAMILIES),
        "forbidden_pre_freeze_columns": sorted(FORBIDDEN_PRE_FREEZE_COLUMNS),
    }
    write_json(schema_path, schema)
    write_json(input_path, {"inputs": input_manifest})
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "ensemble",
        "truth_attached": False,
        "readout_families": list(FAMILIES),
        "circular_control_families": list(SEQUENCE_FAMILIES),
        "shift_bank_ft": EXPECTED_SHIFTS.tolist(),
        "block_rows": 512,
        "quantiles": [0.25, 0.75],
        "models": 0,
        "hmm_well_runs": 0,
        "boosters": 0,
        "prediction_changes": 0,
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    write_json(contract_path, contract)
    freeze = {
        "experiment": EXPERIMENT_NAME,
        "frozen": True,
        "truth_access_count_before_freeze": ledger.count_before_freeze,
        "truth_columns_loaded_before_freeze": [],
        "blocks": len(features),
        "wells": int(features["well_id"].nunique()),
        "feature_schema_sha256": dataframe_schema_sha(features),
        "feature_content_sha256": dataframe_content_sha(features),
        "quantile_content_sha256": dataframe_content_sha(boundaries),
        "file_sha256": {
            "target_free_block_features": sha256_path(feature_path),
            "fold_quantile_boundaries": sha256_path(quantile_path),
            "feature_schema": sha256_path(schema_path),
            "input_manifest": sha256_path(input_path),
            "contract": sha256_path(contract_path),
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(freeze_path, freeze)
    ledger.mark_frozen()
    return {
        "manifest": freeze,
        "manifest_path": freeze_path,
        "feature_path": feature_path,
        "quantile_path": quantile_path,
        "schema_path": schema_path,
        "input_path": input_path,
        "contract_path": contract_path,
    }


def verify_freeze(freeze: Mapping[str, Any], ledger: TruthAccessLedger) -> None:
    manifest = freeze["manifest"]
    if not bool(manifest["frozen"]) or int(manifest["truth_access_count_before_freeze"]) != 0:
        raise ValueError("target-free freeze contract is invalid")
    if not ledger.frozen or ledger.count_before_freeze != 0:
        raise ValueError("truth ledger is not cleanly frozen")
    file_map = {
        "target_free_block_features": freeze["feature_path"],
        "fold_quantile_boundaries": freeze["quantile_path"],
        "feature_schema": freeze["schema_path"],
        "input_manifest": freeze["input_path"],
        "contract": freeze["contract_path"],
    }
    for name, path in file_map.items():
        if sha256_path(path) != manifest["file_sha256"][name]:
            raise ValueError(f"frozen file changed after freeze: {name}")


# %% [markdown]
# ## 7. Post-freeze exp264/exp226 truth and hidden-like loaders


# %%
def aggregate_exp226_blocks(path: Path, *, block_size: int) -> pd.DataFrame:
    partials: list[pd.DataFrame] = []
    usecols = [
        "well_id",
        "row_idx",
        "suffix_offset",
        "fold",
        "tvt_true",
        "tvt_pred",
    ]
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"well_id": str},
        chunksize=250_000,
    ):
        for column in ("row_idx", "suffix_offset", "fold"):
            chunk[column] = pd.to_numeric(chunk[column], errors="raise").astype(np.int64)
        for column in ("tvt_true", "tvt_pred"):
            chunk[column] = pd.to_numeric(chunk[column], errors="raise").astype(np.float64)
        if not np.isfinite(chunk[["tvt_true", "tvt_pred"]].to_numpy(np.float64)).all():
            raise ValueError("exp226 OOF contains non-finite values")
        chunk["block_id"] = chunk["suffix_offset"].to_numpy(np.int64) // block_size
        chunk["exp226_squared_error"] = np.square(chunk["tvt_pred"] - chunk["tvt_true"])
        grouped = chunk.groupby(["well_id", "block_id"], sort=False, observed=True)
        partials.append(
            grouped.agg(
                fold_min=("fold", "min"),
                fold_max=("fold", "max"),
                exp226_rows=("row_idx", "size"),
                exp226_first_row_idx=("row_idx", "min"),
                exp226_last_row_idx=("row_idx", "max"),
                exp226_first_suffix_offset=("suffix_offset", "min"),
                exp226_last_suffix_offset=("suffix_offset", "max"),
                exp226_squared_error_sum=("exp226_squared_error", "sum"),
                exp226_truth_sum=("tvt_true", "sum"),
                exp226_truth_min=("tvt_true", "min"),
                exp226_truth_max=("tvt_true", "max"),
            ).reset_index()
        )
    combined = pd.concat(partials, ignore_index=True)
    grouped = combined.groupby(["well_id", "block_id"], sort=True, observed=True)
    blocks = grouped.agg(
        fold_min=("fold_min", "min"),
        fold_max=("fold_max", "max"),
        exp226_rows=("exp226_rows", "sum"),
        exp226_first_row_idx=("exp226_first_row_idx", "min"),
        exp226_last_row_idx=("exp226_last_row_idx", "max"),
        exp226_first_suffix_offset=("exp226_first_suffix_offset", "min"),
        exp226_last_suffix_offset=("exp226_last_suffix_offset", "max"),
        exp226_squared_error_sum=("exp226_squared_error_sum", "sum"),
        exp226_truth_sum=("exp226_truth_sum", "sum"),
        exp226_truth_min=("exp226_truth_min", "min"),
        exp226_truth_max=("exp226_truth_max", "max"),
    ).reset_index()
    if not blocks["fold_min"].eq(blocks["fold_max"]).all():
        raise ValueError("exp226 fold changes within a block")
    blocks["fold"] = blocks.pop("fold_min").astype(np.int64)
    blocks = blocks.drop(columns="fold_max")
    if not np.array_equal(
        blocks["exp226_rows"].to_numpy(np.int64),
        (
            blocks["exp226_last_suffix_offset"]
            - blocks["exp226_first_suffix_offset"]
            + 1
        ).to_numpy(np.int64),
    ):
        raise ValueError("exp226 suffix offsets are not contiguous within a block")
    by_well = blocks.groupby("well_id", sort=True, observed=True).agg(
        rows=("exp226_rows", "sum"),
        minimum_suffix=("exp226_first_suffix_offset", "min"),
        maximum_suffix=("exp226_last_suffix_offset", "max"),
        fold_count=("fold", "nunique"),
    )
    if (
        not by_well["minimum_suffix"].eq(0).all()
        or not by_well["maximum_suffix"].eq(by_well["rows"] - 1).all()
        or not by_well["fold_count"].eq(1).all()
    ):
        raise ValueError("exp226 well-level suffix/fold contract changed")
    return blocks


def aggregate_exp264_blocks(
    path: Path,
    *,
    prediction_column: str,
    block_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    required = {
        "id",
        "well",
        "outer_fold",
        "actual_tvt",
        "md_since",
        prediction_column,
    }
    missing = sorted(required.difference(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"exp264 Stage D OOF missing {missing}")
    partials: list[pd.DataFrame] = []
    first_row_by_well: dict[str, int] = {}
    last_row_by_well: dict[str, int] = {}
    outer_fold_by_well: dict[str, int] = {}
    total_rows = 0
    total_squared_error = 0.0
    for batch in parquet.iter_batches(
        batch_size=250_000,
        columns=[
            "id",
            "well",
            "outer_fold",
            "actual_tvt",
            "md_since",
            prediction_column,
        ],
    ):
        chunk = batch.to_pandas()
        chunk["id"] = chunk["id"].astype(str)
        chunk["well"] = chunk["well"].astype(str)
        split_id = chunk["id"].str.rsplit("_", n=1, expand=True)
        if not np.array_equal(
            split_id[0].to_numpy(dtype=str), chunk["well"].to_numpy(dtype=str)
        ):
            raise ValueError("exp264 ID prefix differs from well")
        chunk["row_idx"] = pd.to_numeric(split_id[1], errors="raise").astype(np.int64)
        suffix_offset = np.empty(len(chunk), dtype=np.int64)
        for well, positions in chunk.groupby("well", sort=False).indices.items():
            position_array = np.asarray(positions, dtype=np.int64)
            row_index = chunk.iloc[position_array]["row_idx"].to_numpy(np.int64)
            if len(row_index) > 1 and not np.all(np.diff(row_index) == 1):
                raise ValueError(f"exp264 rows are not consecutive within well {well}")
            if well in last_row_by_well and row_index[0] != last_row_by_well[well] + 1:
                raise ValueError(f"exp264 well {well} reappeared out of row order")
            first_row_by_well.setdefault(str(well), int(row_index[0]))
            last_row_by_well[str(well)] = int(row_index[-1])
            folds = chunk.iloc[position_array]["outer_fold"].to_numpy(np.int64)
            if len(np.unique(folds)) != 1:
                raise ValueError(f"exp264 outer fold changes within well {well}")
            previous_fold = outer_fold_by_well.setdefault(str(well), int(folds[0]))
            if previous_fold != int(folds[0]):
                raise ValueError(f"exp264 outer fold changes across batches for {well}")
            suffix_offset[position_array] = row_index - first_row_by_well[str(well)]
        chunk["suffix_offset"] = suffix_offset
        chunk["block_id"] = suffix_offset // block_size
        actual = pd.to_numeric(chunk["actual_tvt"], errors="raise").to_numpy(np.float64)
        prediction = pd.to_numeric(chunk[prediction_column], errors="raise").to_numpy(
            np.float64
        )
        md_since = pd.to_numeric(chunk["md_since"], errors="raise").to_numpy(np.float64)
        if not np.isfinite(np.c_[actual, prediction, md_since]).all():
            raise ValueError("exp264 Stage D OOF contains non-finite values")
        error = prediction - actual
        chunk["exp264_squared_error"] = np.square(error)
        chunk["exp264_bad10"] = np.abs(error) >= 10.0
        grouped = chunk.groupby(["well", "block_id"], sort=False, observed=True)
        partials.append(
            grouped.agg(
                exp264_outer_fold_min=("outer_fold", "min"),
                exp264_outer_fold_max=("outer_fold", "max"),
                exp264_rows=("row_idx", "size"),
                exp264_first_row_idx=("row_idx", "min"),
                exp264_last_row_idx=("row_idx", "max"),
                exp264_first_suffix_offset=("suffix_offset", "min"),
                exp264_last_suffix_offset=("suffix_offset", "max"),
                min_md_since=("md_since", "min"),
                max_md_since=("md_since", "max"),
                exp264_squared_error_sum=("exp264_squared_error", "sum"),
                exp264_bad10_rows=("exp264_bad10", "sum"),
                exp264_truth_sum=("actual_tvt", "sum"),
                exp264_truth_min=("actual_tvt", "min"),
                exp264_truth_max=("actual_tvt", "max"),
            ).reset_index()
        )
        total_rows += len(chunk)
        total_squared_error += float(np.square(error).sum())
    combined = pd.concat(partials, ignore_index=True)
    grouped = combined.groupby(["well", "block_id"], sort=True, observed=True)
    blocks = grouped.agg(
        exp264_outer_fold_min=("exp264_outer_fold_min", "min"),
        exp264_outer_fold_max=("exp264_outer_fold_max", "max"),
        exp264_rows=("exp264_rows", "sum"),
        exp264_first_row_idx=("exp264_first_row_idx", "min"),
        exp264_last_row_idx=("exp264_last_row_idx", "max"),
        exp264_first_suffix_offset=("exp264_first_suffix_offset", "min"),
        exp264_last_suffix_offset=("exp264_last_suffix_offset", "max"),
        min_md_since=("min_md_since", "min"),
        max_md_since=("max_md_since", "max"),
        exp264_squared_error_sum=("exp264_squared_error_sum", "sum"),
        exp264_bad10_rows=("exp264_bad10_rows", "sum"),
        exp264_truth_sum=("exp264_truth_sum", "sum"),
        exp264_truth_min=("exp264_truth_min", "min"),
        exp264_truth_max=("exp264_truth_max", "max"),
    ).reset_index()
    blocks = blocks.rename(columns={"well": "well_id"})
    if not blocks["exp264_outer_fold_min"].eq(blocks["exp264_outer_fold_max"]).all():
        raise ValueError("exp264 outer fold changes within a final block")
    blocks["exp264_outer_fold"] = blocks.pop("exp264_outer_fold_min").astype(np.int64)
    blocks = blocks.drop(columns="exp264_outer_fold_max")
    if not np.array_equal(
        blocks["exp264_rows"].to_numpy(np.int64),
        (
            blocks["exp264_last_suffix_offset"]
            - blocks["exp264_first_suffix_offset"]
            + 1
        ).to_numpy(np.int64),
    ):
        raise ValueError("exp264 suffix offsets are not contiguous within a block")
    evidence = {
        "rows": total_rows,
        "wells": len(first_row_by_well),
        "outer_folds": sorted(set(outer_fold_by_well.values())),
        "rmse": float(np.sqrt(total_squared_error / total_rows)),
    }
    return blocks, evidence


def load_post_freeze_block_metrics(
    config: Mapping[str, Any],
    freeze: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    verify_freeze(freeze, ledger)
    ledger.register_truth_access()
    exp226_spec = get_nested(config, "data.exp226_source")
    exp226_path, exp226_evidence = resolve_file(
        exp226_spec["patterns"],
        label="exp226 OOF",
        expected_decompressed_sha256=str(exp226_spec["expected_decompressed_sha256"]),
    )
    exp264_spec = get_nested(config, "data.exp264_source")
    exp264_path, exp264_evidence = resolve_file(
        exp264_spec["patterns"],
        label="corrected exp264 Stage D OOF",
        expected_file_sha256=str(exp264_spec["expected_sha256"]),
    )

    prediction_column = str(exp264_spec["prediction_column"])
    block_size = int(get_nested(config, "data.block_size"))
    exp226_blocks = aggregate_exp226_blocks(exp226_path, block_size=block_size)
    exp264_blocks, exp264_aggregate_evidence = aggregate_exp264_blocks(
        exp264_path,
        prediction_column=prediction_column,
        block_size=block_size,
    )
    if exp264_aggregate_evidence["outer_folds"] != [0, 1, 2, 3, 4]:
        raise ValueError("exp264 outer-fold provenance changed")
    blocks = exp226_blocks.merge(
        exp264_blocks,
        on=["well_id", "block_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not blocks["_merge"].eq("both").all():
        raise ValueError("exp264/exp226 block identity coverage failed")
    blocks = blocks.drop(columns="_merge")
    identity_checks = {
        "rows": np.array_equal(
            blocks["exp226_rows"].to_numpy(np.int64),
            blocks["exp264_rows"].to_numpy(np.int64),
        ),
        "first_row": np.array_equal(
            blocks["exp226_first_row_idx"].to_numpy(np.int64),
            blocks["exp264_first_row_idx"].to_numpy(np.int64),
        ),
        "last_row": np.array_equal(
            blocks["exp226_last_row_idx"].to_numpy(np.int64),
            blocks["exp264_last_row_idx"].to_numpy(np.int64),
        ),
        "first_suffix": np.array_equal(
            blocks["exp226_first_suffix_offset"].to_numpy(np.int64),
            blocks["exp264_first_suffix_offset"].to_numpy(np.int64),
        ),
        "last_suffix": np.array_equal(
            blocks["exp226_last_suffix_offset"].to_numpy(np.int64),
            blocks["exp264_last_suffix_offset"].to_numpy(np.int64),
        ),
    }
    if not all(identity_checks.values()):
        raise ValueError(f"exp264/exp226 block identity differs: {identity_checks}")
    truth_mean_difference = np.abs(
        blocks["exp264_truth_sum"] / blocks["exp264_rows"]
        - blocks["exp226_truth_sum"] / blocks["exp226_rows"]
    )
    truth_atol = float(get_nested(config, "validation.truth_alignment_atol_ft"))
    if float(truth_mean_difference.max()) > truth_atol:
        raise ValueError("exp264/exp226 block truth mean differs")
    blocks["rows"] = blocks["exp264_rows"].astype(np.int64)
    blocks["exp264_block_rmse"] = np.sqrt(
        blocks["exp264_squared_error_sum"] / blocks["rows"]
    )
    blocks["exp226_block_rmse"] = np.sqrt(
        blocks["exp226_squared_error_sum"] / blocks["rows"]
    )
    blocks["exp264_bad10_rate"] = blocks["exp264_bad10_rows"] / blocks["rows"]
    blocks["exp264_block_rmse_ge_10ft"] = blocks["exp264_block_rmse"] >= 10.0
    blocks["exp226_beats_exp264_by_0p25ft"] = (
        blocks["exp226_block_rmse"] + 0.25 <= blocks["exp264_block_rmse"]
    )
    blocks["exp226_benefit_ft"] = (
        blocks["exp264_block_rmse"] - blocks["exp226_block_rmse"]
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        int(blocks["rows"].sum()) != expected_rows
        or blocks["well_id"].nunique() != expected_wells
    ):
        raise ValueError("post-freeze row/well contract changed")
    observed_rmse = float(exp264_aggregate_evidence["rmse"])
    if not np.isclose(
        observed_rmse, float(exp264_spec["expected_rmse"]), rtol=0.0, atol=1e-9
    ):
        raise ValueError(f"exp264 RMSE contract changed: {observed_rmse}")
    return blocks, [
        {"name": "exp226_oof_post_freeze", **exp226_evidence},
        {
            "name": "exp264_corrected_stage_d_v3_post_freeze",
            **exp264_evidence,
            "observed_rmse": observed_rmse,
            "aggregate_evidence": exp264_aggregate_evidence,
            "maximum_block_truth_mean_difference_ft": float(
                truth_mean_difference.max()
            ),
        },
    ]


def load_hidden_like_sets(
    config: Mapping[str, Any],
    freeze: Mapping[str, Any],
    ledger: TruthAccessLedger,
    valid_wells: set[str],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    verify_freeze(freeze, ledger)
    spec = get_nested(config, "data.hidden_like_assignment")
    path, evidence = resolve_file(
        spec["patterns"],
        label="exp115 hidden-like assignment",
        expected_file_sha256=str(spec["expected_sha256"]),
    )
    frame = pd.read_csv(path, dtype={str(spec["well_column"]): str})
    well_column = str(spec["well_column"])
    required = {well_column, *map(str, spec["role_columns"].values())}
    missing = sorted(required.difference(frame.columns))
    if missing or frame[well_column].duplicated().any():
        raise ValueError(f"hidden-like assignment contract failed: missing={missing}")
    output: dict[str, set[str]] = {}
    for scope, role_column in spec["role_columns"].items():
        selected = set(
            frame.loc[frame[str(role_column)].astype(str).eq("valid"), well_column].astype(str)
        )
        if not selected or not selected.issubset(valid_wells):
            raise ValueError(f"hidden-like scope {scope} has invalid well membership")
        output[str(scope)] = selected
    return output, {"name": "exp115_hidden_like_assignment", **evidence}


# %% [markdown]
# ## 8. Block RMSE, row-weighted AUC, scopes, and fixed gate


# %%
def build_post_freeze_block_readout(
    frozen_features: pd.DataFrame,
    post_freeze_blocks: pd.DataFrame,
    hidden_sets: Mapping[str, set[str]],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    readout = frozen_features.merge(
        post_freeze_blocks,
        on=["well_id", "fold", "block_id"],
        how="left",
        validate="one_to_one",
    )
    if readout["rows"].isna().any() or len(readout) != len(frozen_features):
        raise ValueError("frozen block identity does not cover post-freeze block metrics")
    identity_checks = {
        "row_count": np.array_equal(
            readout["block_row_count"].to_numpy(np.int64), readout["rows"].to_numpy(np.int64)
        ),
        "first_row": np.array_equal(
            readout["block_start_row_idx"].to_numpy(np.int64),
            readout["exp264_first_row_idx"].to_numpy(np.int64),
        ),
        "last_row": np.array_equal(
            readout["block_end_row_idx"].to_numpy(np.int64),
            readout["exp264_last_row_idx"].to_numpy(np.int64),
        ),
        "first_suffix": np.array_equal(
            readout["block_start_suffix_offset"].to_numpy(np.int64),
            readout["exp264_first_suffix_offset"].to_numpy(np.int64),
        ),
        "last_suffix": np.array_equal(
            readout["block_end_suffix_offset"].to_numpy(np.int64),
            readout["exp264_last_suffix_offset"].to_numpy(np.int64),
        ),
        "min_md_since": np.allclose(
            readout["md_since_min_ft"].to_numpy(np.float64),
            readout["min_md_since"].to_numpy(np.float64),
            rtol=0.0,
            atol=1e-5,
        ),
    }
    if not all(identity_checks.values()):
        raise ValueError(f"exp280/exp264 block identity mismatch: {identity_checks}")
    readout["distance_1000_plus"] = readout["min_md_since"] >= 1000.0
    for scope, wells in hidden_sets.items():
        readout[str(scope)] = readout["well_id"].astype(str).isin(wells)
    expected_blocks = int(get_nested(config, "validation.expected_blocks"))
    if len(readout) != expected_blocks:
        raise ValueError("post-freeze block count changed")
    return readout


def weighted_block_auc(
    scores: np.ndarray, positive_counts: np.ndarray, negative_counts: np.ndarray
) -> float | None:
    score = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive_counts, dtype=np.float64)
    negative = np.asarray(negative_counts, dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(positive) & np.isfinite(negative)
    if not valid.all() or np.any(positive < 0) or np.any(negative < 0):
        raise ValueError("AUC requires aligned finite nonnegative counts")
    total_positive = float(positive.sum())
    total_negative = float(negative.sum())
    if total_positive <= 0 or total_negative <= 0:
        return None
    work = pd.DataFrame(
        {"score": score, "positive": positive, "negative": negative}
    ).sort_values("score", kind="mergesort")
    tied = work.groupby("score", sort=True, observed=True)[["positive", "negative"]].sum()
    cumulative_negative = tied["negative"].cumsum().shift(fill_value=0.0)
    numerator = np.sum(
        tied["positive"].to_numpy()
        * (cumulative_negative.to_numpy() + 0.5 * tied["negative"].to_numpy())
    )
    return float(numerator / (total_positive * total_negative))


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    label = np.asarray(labels, dtype=bool)
    return weighted_block_auc(scores, label.astype(float), (~label).astype(float))


def summarize_family_scope(
    frame: pd.DataFrame,
    *,
    family: str,
    scope: str,
) -> dict[str, Any]:
    risk = frame[f"risk__{family}"].to_numpy(np.float64)
    q1 = frame[f"q1__{family}"].to_numpy(bool)
    q4 = frame[f"q4__{family}"].to_numpy(bool)
    positive_counts = frame["exp264_bad10_rows"].to_numpy(np.float64)
    negative_counts = frame["rows"].to_numpy(np.float64) - positive_counts
    q1_rmse = frame.loc[q1, "exp264_block_rmse"]
    q4_rmse = frame.loc[q4, "exp264_block_rmse"]
    control_auc: float | None = None
    if family in SEQUENCE_FAMILIES:
        control_auc = weighted_block_auc(
            frame[f"control_risk__{family}"].to_numpy(np.float64),
            positive_counts,
            negative_counts,
        )
    real_auc = weighted_block_auc(risk, positive_counts, negative_counts)
    q1_rows = float(frame.loc[q1, "rows"].sum())
    q4_rows = float(frame.loc[q4, "rows"].sum())
    q1_bad = float(frame.loc[q1, "exp264_bad10_rows"].sum())
    q4_bad = float(frame.loc[q4, "exp264_bad10_rows"].sum())
    return {
        "family": family,
        "scope": scope,
        "blocks": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "rows": int(frame["rows"].sum()),
        "feature_finite_coverage": float(np.isfinite(risk).mean()),
        "q1_blocks": int(q1.sum()),
        "q4_blocks": int(q4.sum()),
        "q1_q4_overlap_blocks": int(np.logical_and(q1, q4).sum()),
        "q1_mean_exp264_block_rmse": float(q1_rmse.mean()) if len(q1_rmse) else np.nan,
        "q4_mean_exp264_block_rmse": float(q4_rmse.mean()) if len(q4_rmse) else np.nan,
        "q4_minus_q1_mean_exp264_block_rmse": (
            float(q4_rmse.mean() - q1_rmse.mean()) if len(q1_rmse) and len(q4_rmse) else np.nan
        ),
        "q1_median_exp264_block_rmse": float(q1_rmse.median()) if len(q1_rmse) else np.nan,
        "q4_median_exp264_block_rmse": float(q4_rmse.median()) if len(q4_rmse) else np.nan,
        "q4_minus_q1_median_exp264_block_rmse": (
            float(q4_rmse.median() - q1_rmse.median())
            if len(q1_rmse) and len(q4_rmse)
            else np.nan
        ),
        "q1_abs_error_ge_10ft_rate": q1_bad / q1_rows if q1_rows else np.nan,
        "q4_abs_error_ge_10ft_rate": q4_bad / q4_rows if q4_rows else np.nan,
        "row_weighted_abs_error_ge_10ft_auc": real_auc,
        "alias_like_failure_auc": binary_auc(
            frame["exp226_beats_exp264_by_0p25ft"].to_numpy(bool), risk
        ),
        "control_applicable": family in SEQUENCE_FAMILIES,
        "circular_control_row_weighted_auc": control_auc,
        "real_minus_circular_auc": (
            float(real_auc - control_auc)
            if real_auc is not None and control_auc is not None
            else np.nan
        ),
    }


def build_family_metrics(
    readout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope_masks = {
        "pooled": np.ones(len(readout), dtype=bool),
        "distance_1000_plus": readout["distance_1000_plus"].to_numpy(bool),
        "hidden_like_spatial": readout["hidden_like_spatial"].to_numpy(bool),
        "hidden_like_typewell_purged": readout[
            "hidden_like_typewell_purged"
        ].to_numpy(bool),
    }
    scope_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for scope, mask in scope_masks.items():
            part = readout.loc[mask]
            if part.empty:
                raise ValueError(f"scope {scope} is empty")
            scope_rows.append(summarize_family_scope(part, family=family, scope=scope))
        for fold, part in readout.groupby("fold", sort=True):
            fold_rows.append(
                {
                    **summarize_family_scope(part, family=family, scope=f"fold_{int(fold)}"),
                    "fold": int(fold),
                }
            )
    return pd.DataFrame(scope_rows), pd.DataFrame(fold_rows)


def evaluate_fixed_gate(
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    boundaries: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    guards = get_nested(config, "model.pass_requires_all_per_family")
    minimum_coverage = float(guards["minimum_feature_coverage"])
    minimum_delta = float(guards["minimum_q4_minus_q1_mean_rmse_ft"])
    minimum_positive_folds = int(guards["minimum_positive_folds"])
    minimum_auc = float(guards["minimum_pooled_auc_abs_error_ge_10ft"])
    minimum_auc_folds = int(guards["minimum_folds_auc_gt_half"])
    minimum_control_folds = int(guards["minimum_folds_real_better_than_circular"])
    gate_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        scopes = scope_metrics.loc[scope_metrics["family"].eq(family)].set_index("scope")
        folds = fold_metrics.loc[fold_metrics["family"].eq(family)].sort_values("fold")
        pooled = scopes.loc["pooled"]
        positive_delta_folds = int(
            (folds["q4_minus_q1_mean_exp264_block_rmse"] > 0.0).sum()
        )
        auc_above_half_folds = int(
            (folds["row_weighted_abs_error_ge_10ft_auc"] > 0.5).sum()
        )
        stress_pass = all(
            float(scopes.loc[name, "q4_minus_q1_mean_exp264_block_rmse"]) > 0.0
            for name in (
                "distance_1000_plus",
                "hidden_like_spatial",
                "hidden_like_typewell_purged",
            )
        )
        family_boundaries = boundaries.loc[boundaries["family"].eq(family)]
        separated_quantiles = bool(
            (
                family_boundaries["q75_risk_boundary"]
                > family_boundaries["q25_risk_boundary"]
            ).all()
        )
        if family in SEQUENCE_FAMILIES:
            control_fold_count = int((folds["real_minus_circular_auc"] > 0.0).sum())
            control_pooled_pass = bool(float(pooled["real_minus_circular_auc"]) > 0.0)
            control_fold_pass = control_fold_count >= minimum_control_folds
        else:
            control_fold_count = 5
            control_pooled_pass = True
            control_fold_pass = True
        checks = {
            "coverage": float(pooled["feature_finite_coverage"]) >= minimum_coverage,
            "quantile_separation_all_folds": separated_quantiles,
            "pooled_q4_minus_q1_mean_rmse": (
                float(pooled["q4_minus_q1_mean_exp264_block_rmse"]) >= minimum_delta
            ),
            "pooled_q4_minus_q1_median_positive": (
                float(pooled["q4_minus_q1_median_exp264_block_rmse"]) > 0.0
            ),
            "positive_delta_folds": positive_delta_folds >= minimum_positive_folds,
            "stress_scopes_positive": stress_pass,
            "pooled_abs_error_ge_10ft_auc": (
                float(pooled["row_weighted_abs_error_ge_10ft_auc"]) >= minimum_auc
            ),
            "auc_above_half_folds": auc_above_half_folds >= minimum_auc_folds,
            "real_better_than_circular_pooled": control_pooled_pass,
            "real_better_than_circular_folds": control_fold_pass,
        }
        gate_rows.append(
            {
                "family": family,
                "passed": all(checks.values()),
                "positive_delta_folds": positive_delta_folds,
                "auc_above_half_folds": auc_above_half_folds,
                "real_better_than_circular_folds": control_fold_count,
                **{f"check__{name}": value for name, value in checks.items()},
            }
        )
    family_gate = pd.DataFrame(gate_rows)
    passed_families = family_gate.loc[family_gate["passed"], "family"].astype(str).tolist()
    decision = {
        "technical_passed": True,
        "scientific_passed": bool(passed_families),
        "passed_families": passed_families,
        "action": (
            "qualify_separately_designed_addonly_experiment_no_correction_here"
            if passed_families
            else "close_depth_alias_confidence_branch_without_rescue"
        ),
    }
    return family_gate, decision


# %% [markdown]
# ## 9. Metrics and generated artifacts


# %%
def save_final_artifacts(
    readout: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    family_gate: pd.DataFrame,
    freeze: Mapping[str, Any],
    post_freeze_evidence: list[dict[str, Any]],
    decision: Mapping[str, Any],
    started_at: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = artifact_dir()
    readout_path = artifacts / f"{OUTPUT_PREFIX}_post_freeze_block_readout.csv.gz"
    scope_path = artifacts / f"{OUTPUT_PREFIX}_family_scope_metrics.csv"
    fold_path = artifacts / f"{OUTPUT_PREFIX}_family_fold_metrics.csv"
    gate_path = artifacts / f"{OUTPUT_PREFIX}_family_gate.csv"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    sha_path = artifacts / f"{OUTPUT_PREFIX}_sha_manifest.csv"
    readout_evidence = write_csv_gzip(readout, readout_path)
    scope_metrics.to_csv(scope_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    family_gate.to_csv(gate_path, index=False)
    pooled = scope_metrics.loc[scope_metrics["scope"].eq("pooled")].to_dict(orient="records")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "stage_0_completed_guard_passed"
            if decision["scientific_passed"]
            else "stage_0_completed_guard_failed"
        ),
        "route": get_nested(config, "experiment.route"),
        "runtime_seconds": time.time() - started_at,
        "rows": int(get_nested(config, "validation.expected_rows")),
        "wells": int(readout["well_id"].nunique()),
        "blocks": len(readout),
        "readout_families": len(FAMILIES),
        "controls": 1,
        "models": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "parent_control_retraining": False,
        "decision": dict(decision),
        "family_gate": family_gate.to_dict(orient="records"),
        "pooled_family_metrics": pooled,
        "freeze": freeze["manifest"],
        "post_freeze_inputs": post_freeze_evidence,
        "artifacts": {
            "post_freeze_block_readout": readout_evidence,
            "family_scope_metrics": str(scope_path),
            "family_fold_metrics": str(fold_path),
            "family_gate": str(gate_path),
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(summary_path, summary)
    output_paths = {
        "contract": freeze["contract_path"],
        "input_manifest": freeze["input_path"],
        "feature_schema": freeze["schema_path"],
        "target_free_block_features": freeze["feature_path"],
        "fold_quantile_boundaries": freeze["quantile_path"],
        "freeze_manifest": freeze["manifest_path"],
        "post_freeze_block_readout": readout_path,
        "family_scope_metrics": scope_path,
        "family_fold_metrics": fold_path,
        "family_gate": gate_path,
        "summary": summary_path,
    }
    sha_frame = pd.DataFrame(
        [
            {"name": name, "path": str(path), "sha256": sha256_path(path)}
            for name, path in output_paths.items()
        ]
    )
    sha_frame.to_csv(sha_path, index=False)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "route": "ensemble",
        "stage": "stage_0",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "diagnostic": {
            "technical_passed": decision["technical_passed"],
            "scientific_passed": decision["scientific_passed"],
            "passed_families": decision["passed_families"],
            "family_gate": family_gate.to_dict(orient="records"),
            "feature_content_sha256": freeze["manifest"]["feature_content_sha256"],
            "quantile_content_sha256": freeze["manifest"]["quantile_content_sha256"],
        },
        "notes": (
            "Zero-booster readout only; no prediction, inference, or submission was generated."
        ),
    }
    write_json(metrics_output_path(), metrics)
    return summary


def run_stage_0_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=True)
    started_at = time.time()
    ledger = TruthAccessLedger()
    scores, pre_freeze_evidence = load_exp280_target_free_scores(config)
    features = build_target_free_block_features(scores)
    expected_blocks = int(get_nested(config, "validation.expected_blocks"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(features) != expected_blocks or features["well_id"].nunique() != expected_wells:
        raise ValueError("target-free feature block/well contract changed")
    boundaries = fit_fold_quantile_boundaries(features, config)
    features = attach_frozen_quartile_flags(features, boundaries)
    freeze = freeze_target_free_bundle(
        features, boundaries, pre_freeze_evidence, config, ledger
    )
    post_freeze_blocks, post_freeze_evidence = load_post_freeze_block_metrics(
        config, freeze, ledger
    )
    hidden_sets, hidden_evidence = load_hidden_like_sets(
        config,
        freeze,
        ledger,
        set(features["well_id"].astype(str)),
    )
    post_freeze_evidence.append(hidden_evidence)
    readout = build_post_freeze_block_readout(
        features, post_freeze_blocks, hidden_sets, config
    )
    scope_metrics, fold_metrics = build_family_metrics(readout)
    family_gate, decision = evaluate_fixed_gate(
        scope_metrics, fold_metrics, boundaries, config
    )
    summary = save_final_artifacts(
        readout,
        scope_metrics,
        fold_metrics,
        family_gate,
        freeze,
        post_freeze_evidence,
        decision,
        started_at,
        config,
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 10. Setup and configuration preview


# %%
CONFIG = load_experiment_config()
validate_scientific_contract(CONFIG)
SETUP_PREVIEW = {
    "experiment": get_nested(CONFIG, "experiment.name"),
    "route": get_nested(CONFIG, "experiment.route"),
    "parent": get_nested(CONFIG, "lineage.parent"),
    "score_source": get_nested(CONFIG, "data.exp280_source.experiment"),
    "readout_families": list(FAMILIES),
    "sequence_control_families": list(SEQUENCE_FAMILIES),
    "folds": get_nested(CONFIG, "validation.expected_folds"),
    "expected_blocks": get_nested(CONFIG, "validation.expected_blocks"),
    "active_stage": get_nested(CONFIG, "execution.active_stage"),
    "implementation_approved": get_nested(CONFIG, "execution.implementation_approved"),
    "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
    "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
    "models": get_nested(CONFIG, "execution_contract.models"),
    "trained_folds": get_nested(CONFIG, "execution_contract.trained_folds"),
    "boosters": get_nested(CONFIG, "execution_contract.boosters"),
    "hmm_well_runs": get_nested(CONFIG, "execution_contract.hmm_well_runs"),
}
print(json.dumps(to_jsonable(SETUP_PREVIEW), indent=2, sort_keys=True))


# %% [markdown]
# ## 11. Run the approved Stage 0 readout only
#
# The repository implementation intentionally leaves Kaggle push/run disabled.
# A later explicit approval must set all three execution switches before this
# cell can consume mounted parent outputs.


# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_stage_0_experiment(CONFIG)
    POOLED_PREVIEW = pd.DataFrame(SUMMARY["pooled_family_metrics"])[
        [
            "family",
            "q4_minus_q1_mean_exp264_block_rmse",
            "row_weighted_abs_error_ge_10ft_auc",
            "real_minus_circular_auc",
        ]
    ]
    display(POOLED_PREVIEW)

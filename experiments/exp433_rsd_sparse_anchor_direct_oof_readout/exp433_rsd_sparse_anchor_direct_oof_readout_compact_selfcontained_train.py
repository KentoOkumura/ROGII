# %% [markdown]
# # exp433 RSD sparse-anchor direct OOF readout
#
# This deterministic zero-model readout consumes the frozen exp426 Stage-A
# score bank without regenerating scores, support, ranks, or top-3 candidates.
# One preregistered Viterbi decoder projects sparse absolute-datum anchors onto
# the saved exp226 OOF prediction.  Input, support, datum-path, and prediction
# hashes are frozen and independently reproduced before truth, hidden-like
# roles, or persistent-offset episodes are read.

# %% [markdown]
# ## Contents
# 1. Imports and execution guard
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific and execution contract
# 4. Frozen exp426 and safe exp226 input loaders
# 5. Score-bank and support diagnostics
# 6. Fixed sparse-anchor Viterbi and row projection
# 7. Target-free prediction freeze and independent rerun
# 8. Post-freeze truth, raw-GR, hidden-like, and episode joins
# 9. Scope, fold, by-well, and mechanism readouts
# 10. Technical and scientific gates
# 11. Metrics and generated artifacts
# 12. Setup, configuration preview, and approved execution

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

EXPERIMENT_NAME = "exp433_rsd_sparse_anchor_direct_oof_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
IMPORT_ONLY_ENV = "EXP433_IMPORT_ONLY"
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
OFFSETS_FT = np.asarray(
    [-80.0, -40.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    dtype=np.float64,
)
TIE_ORDER_FT = np.asarray(
    [0.0, -2.0, 2.0, -5.0, 5.0, -10.0, 10.0, -20.0, 20.0, -40.0, 40.0, -80.0, 80.0],
    dtype=np.float64,
)
SAFE_EXP226_COLUMNS = ("well_id", "row_idx", "suffix_offset", "fold", "tvt_pred")
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
SCORE_BOOLEAN_COLUMNS = (
    "rsd_valid",
    "rsd_top3",
    "raw_pearson_valid",
    "raw_pearson_top3",
    "raw_gaussian_valid",
    "raw_gaussian_top3",
    "permutation_valid",
    "permutation_top3",
)
SCORE_INTEGER_COLUMNS = (
    "fold",
    "block_id",
    "block_start_suffix_offset",
    "block_end_suffix_offset",
    "block_start_row_idx",
    "block_end_row_idx",
    "block_row_count",
    "raw_finite_gr_points",
    "offset_slot",
    "rsd_paired_bins",
    "rsd_rank",
    "raw_pearson_pairs",
    "raw_pearson_rank",
    "raw_gaussian_rank",
    "permutation_rank",
)
DATUM_LOGICAL_COLUMNS = [
    "well_id",
    "fold",
    "block_id",
    "block_center_suffix_offset",
    "supported",
    "valid_offset_count",
    "selected_offset_slot",
    "selected_offset_ft",
    "selected_emission_score",
    "cumulative_objective",
    "blockwise_top1_offset_ft",
]
PREDICTION_LOGICAL_COLUMNS = [
    "well_id",
    "row_idx",
    "suffix_offset",
    "fold",
    "base_tvt_pred",
    "datum_correction_ft",
    "primary_tvt_pred",
    "blockwise_correction_ft",
    "blockwise_tvt_pred",
]
FORBIDDEN_PRE_FREEZE_COLUMNS = {
    "TVT",
    "tvt_true",
    "actual_tvt",
    "target",
    "error",
    "abs_error",
    "oracle_offset_ft",
    "persistent_episode",
    "verification_like_spatial_role",
    "verification_like_typewell_purged_role",
}


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
    raise FileNotFoundError(f"exp433 config not found in {[str(path) for path in candidates]}")


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
        "logical_content_sha256": dataframe_content_sha(frame),
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


def peak_rss_gb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


def assert_no_forbidden_columns(columns: Iterable[str]) -> None:
    present = set(map(str, columns)).intersection(FORBIDDEN_PRE_FREEZE_COLUMNS)
    if present:
        raise ValueError(f"truth/error columns are forbidden before freeze: {sorted(present)}")


# %% [markdown]
# ## 3. Frozen scientific and execution contract


# %%
def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("unexpected experiment name")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp433 route must remain pf_beam")
    if not bool(get_nested(config, "implementation.enabled")):
        raise ValueError("exp433 implementation must be enabled")
    if (
        get_nested(config, "implementation.scope")
        != "compact_selfcontained_train_implementation_ready"
    ):
        raise ValueError("only the compact self-contained train implementation is allowed")
    if not bool(get_nested(config, "execution.implementation_authorized")):
        raise ValueError("exp433 implementation is not authorized")

    candidate = get_nested(config, "model.frozen_candidate_contract") or {}
    primary = get_nested(config, "model.primary") or {}
    if not np.array_equal(
        np.asarray(candidate.get("offsets_ft", ()), dtype=np.float64),
        OFFSETS_FT,
    ):
        raise ValueError("the frozen exp426 offset bank changed")
    if not np.array_equal(
        np.asarray(candidate.get("tie_order_ft", ()), dtype=np.float64),
        TIE_ORDER_FT,
    ):
        raise ValueError("the deterministic tie order changed")
    expected_candidate = {
        "block_size_rows": 512,
        "block_overlap_rows": 0,
        "score_column": "rsd_bin_score",
        "valid_column": "rsd_valid",
        "score_support_rank_regeneration": "forbidden",
    }
    for key, expected in expected_candidate.items():
        if candidate.get(key) != expected:
            raise ValueError(f"frozen candidate contract changed: {key}")
    expected_primary = {
        "initial_center_ft": 0.0,
        "initial_sigma_ft": 5.0,
        "transition_sigma_ft": 10.0,
        "maximum_initial_step_ft": 20.0,
        "maximum_adjacent_step_ft": 40.0,
        "unsupported_block_emission": "zero_all_states_transition_only_carry",
        "partially_invalid_candidate_emission": "negative_infinity",
        "row_projection": "linear_boundary_zero_through_block_centers_then_last_hold",
    }
    for key, expected in expected_primary.items():
        if primary.get(key) != expected:
            raise ValueError(f"primary decoder contract changed: {key}")

    counts = get_nested(config, "execution_contract") or {}
    expected_counts = {
        "primary_decoders": 1,
        "diagnostic_replays": 1,
        "wells_decoded": 773,
        "reporting_folds": 5,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_runs": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    for key, expected in expected_counts.items():
        if int(counts.get(key, -1)) != expected:
            raise ValueError(f"execution contract changed: {key}")
    if bool(counts.get("parent_control_regeneration")) or bool(counts.get("score_regeneration")):
        raise ValueError("parent/control/score regeneration is forbidden")

    frozen_exp426 = get_nested(config, "data.exp426") or {}
    expected_frozen_artifacts = {
        "score_bank": {
            "logical_content_sha256": (
                "463aa32bef9a1045469466e2cf5fd68e038258e75f11fc88153fd9ca7f8dd2fd"
            ),
            "decompressed_sha256": (
                "6adb009b83c884681fa64e29c03bc05c6dac15d3bb6826df1a000961c8bbe575"
            ),
            "schema_sha256": ("6e86f76bf7df038e5b3b8077db80888c737bfa3880377ac24fe74d236706e9bd"),
        },
        "well_manifest": {
            "logical_content_sha256": (
                "6af41b07945527af423405a860bd04ef356656266cfb264b80c471a72e7d0266"
            ),
            "decompressed_sha256": (
                "e36e3b7dae4625fa44bb39370daa0112612ab27f16fd4440807e4e19b9fe69fc"
            ),
            "schema_sha256": ("47a74386f8e528605c95ae4aa07519cc5864735435640148ba628a14e94aaf6f"),
        },
        "input_manifest": {
            "logical_content_sha256": (
                "7933f0f2babaa382ee23ae64db096db0dcc775035fc399254e64e7b30fe7656b"
            ),
            "decompressed_sha256": (
                "899beb7b1fcc4dfcd6777398abfbc6fb15fa75d74dc250950b7e2a705b30e435"
            ),
            "schema_sha256": ("d21ab7618bb3075014310c4398f93b5ce726065017a3834566ccfa52e53fc56a"),
        },
    }
    for artifact_name, expected_hashes in expected_frozen_artifacts.items():
        artifact_spec = frozen_exp426.get(artifact_name) or {}
        for hash_name, expected_hash in expected_hashes.items():
            if artifact_spec.get(hash_name) != expected_hash:
                raise ValueError(f"frozen exp426 {artifact_name} contract changed: {hash_name}")

    forbidden_true = (
        "implementation.inference_enabled",
        "implementation.submission_enabled",
        "execution.run_inference",
        "execution.create_submission",
    )
    if any(bool(get_nested(config, key)) for key in forbidden_true):
        raise ValueError("inference and submission are disabled")
    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_package_authorized"))
        and bool(get_nested(config, "execution.kaggle_push_authorized"))
        and bool(get_nested(config, "execution.kaggle_execution_authorized"))
        and bool(get_nested(config, "execution.run_train"))
    ):
        raise RuntimeError("exp433 Kaggle package/push/run is not approved")

    contract = {
        "experiment": EXPERIMENT_NAME,
        "stage": "frozen_sparse_anchor_direct_oof_readout",
        "score_source": "exp426_version_1_immutable_artifact",
        "base_prediction": "exp226_final_tvt_pred",
        "offsets_ft": OFFSETS_FT.tolist(),
        "tie_order_ft": TIE_ORDER_FT.tolist(),
        "block_size_rows": 512,
        "truth_attachment": "after_input_support_datum_path_prediction_and_rerun_sha_freeze",
        "frozen_exp426_artifacts": expected_frozen_artifacts,
        "execution_counts": expected_counts,
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


class TruthAccessLedger:
    def __init__(self) -> None:
        self.frozen_prediction_sha256: str | None = None
        self.truth_rows_before_freeze = 0
        self.truth_rows_after_freeze = 0
        self.hidden_role_rows_before_freeze = 0
        self.hidden_role_rows_after_freeze = 0
        self.episode_rows_before_freeze = 0
        self.episode_rows_after_freeze = 0

    @property
    def frozen(self) -> bool:
        return self.frozen_prediction_sha256 is not None

    def mark_frozen(self, prediction_sha256: str) -> None:
        if len(prediction_sha256) != 64:
            raise ValueError("prediction freeze requires a SHA256")
        self.frozen_prediction_sha256 = prediction_sha256

    def _register(self, rows: int, kind: str) -> None:
        before_name = f"{kind}_rows_before_freeze"
        after_name = f"{kind}_rows_after_freeze"
        if not self.frozen:
            setattr(self, before_name, int(getattr(self, before_name)) + int(rows))
            raise RuntimeError(
                f"{kind.replace('_', '-')} access attempted before prediction freeze"
            )
        setattr(self, after_name, int(getattr(self, after_name)) + int(rows))

    def register_truth_access(self, rows: int) -> None:
        self._register(rows, "truth")

    def register_hidden_role_access(self, rows: int) -> None:
        self._register(rows, "hidden_role")

    def register_episode_access(self, rows: int) -> None:
        self._register(rows, "episode")

    def as_dict(self) -> dict[str, Any]:
        return {
            "frozen_prediction_sha256": self.frozen_prediction_sha256,
            "truth_rows_before_freeze": self.truth_rows_before_freeze,
            "truth_rows_after_freeze": self.truth_rows_after_freeze,
            "hidden_role_rows_before_freeze": self.hidden_role_rows_before_freeze,
            "hidden_role_rows_after_freeze": self.hidden_role_rows_after_freeze,
            "episode_rows_before_freeze": self.episode_rows_before_freeze,
            "episode_rows_after_freeze": self.episode_rows_after_freeze,
        }


# %% [markdown]
# ## 4. Frozen exp426 and safe exp226 input loaders


# %%
def _coerce_score_bank(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(SCORE_LOGICAL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"exp426 score bank missing columns: {missing}")
    result = frame[SCORE_LOGICAL_COLUMNS].copy()
    result["well_id"] = result["well_id"].astype(str)
    for column in SCORE_INTEGER_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(np.int64)
    for column in SCORE_BOOLEAN_COLUMNS:
        if result[column].dtype != bool:
            normalized = result[column].astype(str).str.strip().str.lower()
            if not normalized.isin({"true", "false"}).all():
                raise ValueError(f"invalid boolean values in {column}")
            result[column] = normalized.eq("true")
    float_columns = [
        column
        for column in SCORE_LOGICAL_COLUMNS
        if column not in {"well_id", *SCORE_INTEGER_COLUMNS, *SCORE_BOOLEAN_COLUMNS}
    ]
    for column in float_columns:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(np.float64)
    return result.sort_values(["well_id", "block_id", "offset_slot"], kind="mergesort").reset_index(
        drop=True
    )


def validate_score_bank_structure(score_bank: pd.DataFrame) -> dict[str, bool]:
    checks = {
        "required_columns": set(SCORE_LOGICAL_COLUMNS).issubset(score_bank.columns),
        "duplicate_identity_zero": not score_bank.duplicated(
            ["well_id", "block_id", "offset_slot"]
        ).any(),
    }
    if not checks["required_columns"]:
        return checks
    canonical = score_bank.sort_values(
        ["well_id", "block_id", "offset_slot"], kind="mergesort"
    ).reset_index(drop=True)
    checks["canonical_order"] = canonical[SCORE_LOGICAL_COLUMNS].equals(
        score_bank.reset_index(drop=True)[SCORE_LOGICAL_COLUMNS]
    )
    numeric_scores = (
        "rsd_bin_score",
        "rsd_pearson",
        "rsd_cosine",
        "rsd_spearman",
        "raw_pearson_score",
        "raw_gaussian_score",
        "permutation_score",
    )
    checks["finite_score_storage"] = bool(
        np.isfinite(score_bank[list(numeric_scores)].to_numpy(np.float64)).all()
    )
    groups = score_bank.groupby(["well_id", "block_id"], sort=False)
    checks["thirteen_offsets_per_block"] = bool(groups.size().eq(len(OFFSETS_FT)).all())
    checks["fixed_offset_order"] = all(
        np.array_equal(part["offset_ft"].to_numpy(np.float64), OFFSETS_FT) for _, part in groups
    )
    expected_ranks = np.arange(1, len(OFFSETS_FT) + 1)
    checks["rsd_rank_permutation"] = all(
        np.array_equal(np.sort(part["rsd_rank"].to_numpy(np.int64)), expected_ranks)
        for _, part in groups
    )
    checks["rsd_rank_one_is_valid_or_block_unsupported"] = all(
        (
            int((part["rsd_valid"] & part["rsd_rank"].eq(1)).sum()) == 1
            if bool(part["rsd_valid"].any())
            else True
        )
        for _, part in groups
    )
    checks["top3_mask_immutable_contract"] = all(
        int(part["rsd_top3"].sum()) == min(3, int(part["rsd_valid"].sum()))
        and not bool((part["rsd_top3"] & ~part["rsd_valid"]).any())
        for _, part in groups
    )
    return checks


def load_frozen_exp426_inputs(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    spec = get_nested(config, "data.exp426") or {}
    evidence: list[dict[str, Any]] = []

    score_spec = spec["score_bank"]
    score_path, score_evidence = resolve_file(
        [str(value) for value in score_spec["patterns"]],
        label="frozen exp426 score bank",
        expected_decompressed_sha256=str(score_spec["decompressed_sha256"]),
    )
    score_bank = _coerce_score_bank(pd.read_csv(score_path, dtype={"well_id": str}))
    score_evidence.update(
        {
            "name": "exp426_score_bank",
            "rows": len(score_bank),
            "blocks": int(score_bank[["well_id", "block_id"]].drop_duplicates().shape[0]),
            "producer_logical_content_sha256": str(score_spec["logical_content_sha256"]),
            "post_read_logical_content_sha256": dataframe_content_sha(
                score_bank,
                SCORE_LOGICAL_COLUMNS,
            ),
            "schema_sha256": dataframe_schema_sha(score_bank[SCORE_LOGICAL_COLUMNS]),
        }
    )
    evidence.append(score_evidence)

    manifest_frames: dict[str, pd.DataFrame] = {}
    for key in ("well_manifest", "input_manifest"):
        item = spec[key]
        path, item_evidence = resolve_file(
            [str(value) for value in item["patterns"]],
            label=f"frozen exp426 {key}",
            expected_decompressed_sha256=str(item["decompressed_sha256"]),
        )
        frame = (
            pd.read_csv(path, dtype={"well_id": str})
            .sort_values("well_id", kind="mergesort")
            .reset_index(drop=True)
        )
        item_evidence.update(
            {
                "name": f"exp426_{key}",
                "rows": len(frame),
                "producer_logical_content_sha256": str(item["logical_content_sha256"]),
                "post_read_logical_content_sha256": dataframe_content_sha(frame),
                "schema_sha256": dataframe_schema_sha(frame),
            }
        )
        evidence.append(item_evidence)
        manifest_frames[key] = frame

    structure = validate_score_bank_structure(score_bank)
    expected_score_checks = {
        "score_rows": len(score_bank) == int(score_spec["rows"]),
        "score_blocks": score_evidence["blocks"] == int(score_spec["blocks"]),
        "score_decompressed_sha": (
            score_evidence["decompressed_sha256"] == str(score_spec["decompressed_sha256"])
        ),
        "score_schema_sha": (score_evidence["schema_sha256"] == str(score_spec["schema_sha256"])),
        "well_manifest_rows": len(manifest_frames["well_manifest"])
        == int(spec["well_manifest"]["rows"]),
        "well_manifest_decompressed_sha": (
            evidence[1]["decompressed_sha256"] == str(spec["well_manifest"]["decompressed_sha256"])
        ),
        "well_manifest_schema_sha": (
            evidence[1]["schema_sha256"] == str(spec["well_manifest"]["schema_sha256"])
        ),
        "input_manifest_rows": len(manifest_frames["input_manifest"])
        == int(spec["input_manifest"]["rows"]),
        "input_manifest_decompressed_sha": (
            evidence[2]["decompressed_sha256"] == str(spec["input_manifest"]["decompressed_sha256"])
        ),
        "input_manifest_logical_sha": (
            evidence[2]["post_read_logical_content_sha256"]
            == str(spec["input_manifest"]["logical_content_sha256"])
        ),
        "input_manifest_schema_sha": (
            evidence[2]["schema_sha256"] == str(spec["input_manifest"]["schema_sha256"])
        ),
        **structure,
    }
    if not all(expected_score_checks.values()):
        failed = [key for key, passed in expected_score_checks.items() if not passed]
        raise ValueError(f"frozen exp426 input contract failed: {failed}")
    score_evidence["contract_checks"] = expected_score_checks
    return (
        score_bank,
        manifest_frames["well_manifest"],
        manifest_frames["input_manifest"],
        evidence,
    )


def load_exp226_safe(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    spec = get_nested(config, "data.exp226") or {}
    path, evidence = resolve_file(
        [str(value) for value in spec["patterns"]],
        label="exp226 final OOF",
        expected_decompressed_sha256=str(spec["decompressed_sha256"]),
    )
    safe_columns = [str(value) for value in spec["safe_columns_before_freeze"]]
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
        raise ValueError("exp226 tvt_pred must be finite")
    checks = {
        "rows": len(frame) == int(get_nested(config, "validation.expected_rows")),
        "wells": frame["well_id"].nunique() == int(get_nested(config, "validation.expected_wells")),
        "folds": sorted(frame["fold"].unique().tolist())
        == [int(value) for value in get_nested(config, "validation.expected_folds")],
        "one_fold_per_well": bool(frame.groupby("well_id")["fold"].nunique().eq(1).all()),
        "contiguous_suffix": all(
            np.array_equal(
                part.sort_values("row_idx")["suffix_offset"].to_numpy(np.int64),
                np.arange(len(part), dtype=np.int64),
            )
            for _, part in frame.groupby("well_id", sort=False)
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"exp226 safe OOF contract failed: {checks}")
    evidence.update(
        {
            "name": "exp226_final_oof_safe_columns",
            "rows": len(frame),
            "wells": int(frame["well_id"].nunique()),
            "folds": sorted(int(value) for value in frame["fold"].unique()),
            "safe_columns": safe_columns,
            "contract_checks": checks,
        }
    )
    return frame, path, evidence


# %% [markdown]
# ## 5. Score-bank and support diagnostics


# %%
def distance_bucket(values: pd.Series) -> pd.Series:
    numeric = values.to_numpy(np.float64)
    labels = np.select(
        [
            numeric < 50.0,
            numeric < 100.0,
            numeric < 500.0,
            numeric < 1000.0,
        ],
        [
            "distance_0_50",
            "distance_50_100",
            "distance_100_500",
            "distance_500_1000",
        ],
        default="distance_1000_plus",
    )
    return pd.Series(labels, index=values.index, dtype=object)


def build_support_diagnostics(
    score_bank: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    minimum_raw = 32
    minimum_bins = 16
    rows: list[dict[str, Any]] = []
    for (well_id, block_id), part in score_bank.groupby(["well_id", "block_id"], sort=True):
        part = part.reset_index(drop=True)
        valid = part["rsd_valid"].to_numpy(bool)
        raw_points = int(part["raw_finite_gr_points"].iloc[0])
        paired = part["rsd_paired_bins"].to_numpy(np.int64)
        invalid = ~valid
        invalid_raw = invalid & (raw_points < minimum_raw)
        invalid_bins = invalid & ~invalid_raw & (paired < minimum_bins)
        invalid_residual = invalid & ~invalid_raw & ~invalid_bins
        valid_offsets = set(part.loc[part["rsd_valid"], "offset_ft"].astype(float))
        symmetric_pairs = sum(
            positive in valid_offsets and -positive in valid_offsets
            for positive in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
        )
        rows.append(
            {
                "well_id": str(well_id),
                "fold": int(part["fold"].iloc[0]),
                "block_id": int(block_id),
                "md_since_mid_ft": float(part["md_since_mid_ft"].iloc[0]),
                "block_row_count": int(part["block_row_count"].iloc[0]),
                "observed_gr_share": float(part["observed_gr_share"].iloc[0]),
                "valid_offset_count": int(valid.sum()),
                "supported": bool(valid.any()),
                "zero_offset_valid": 0.0 in valid_offsets,
                "symmetric_valid_pair_count": int(symmetric_pairs),
                "all_thirteen_valid": bool(valid.all()),
                "invalid_raw_finite_gr_count": int(invalid_raw.sum()),
                "invalid_paired_bin_count": int(invalid_bins.sum()),
                "invalid_low_variance_or_correlation_count": int(invalid_residual.sum()),
            }
        )
    diagnostics = (
        pd.DataFrame(rows)
        .sort_values(["well_id", "block_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    diagnostics["distance_bucket"] = distance_bucket(diagnostics["md_since_mid_ft"])

    well_rows: list[dict[str, Any]] = []
    for well_id, part in diagnostics.groupby("well_id", sort=True):
        supported_ids = part.loc[part["supported"], "block_id"].to_numpy(np.int64)
        gaps = np.maximum(np.diff(supported_ids) - 1, 0)
        gap_median = float(np.median(gaps)) if len(gaps) else 0.0
        gap_p90 = float(np.quantile(gaps, 0.90)) if len(gaps) else 0.0
        gap_max = int(gaps.max()) if len(gaps) else 0
        well_rows.append(
            {
                "well_id": str(well_id),
                "well_supported_block_fraction": float(part["supported"].mean()),
                "supported_gap_median_blocks": gap_median,
                "supported_gap_p90_blocks": gap_p90,
                "supported_gap_max_blocks": gap_max,
            }
        )
    diagnostics = diagnostics.merge(
        pd.DataFrame(well_rows), on="well_id", how="left", validate="many_to_one"
    )
    by_distance = []
    for bucket, part in diagnostics.groupby("distance_bucket", sort=True):
        by_distance.append(
            {
                "scope": str(bucket),
                "blocks": len(part),
                "rows": int(part["block_row_count"].sum()),
                "supported_block_fraction": float(part["supported"].mean()),
                "mean_valid_offset_count": float(part["valid_offset_count"].mean()),
            }
        )
    expected_blocks = int(get_nested(config, "validation.expected_blocks"))
    summary = {
        "blocks": len(diagnostics),
        "expected_blocks": expected_blocks,
        "supported_blocks": int(diagnostics["supported"].sum()),
        "supported_block_fraction": float(diagnostics["supported"].mean()),
        "supported_wells": int(diagnostics.groupby("well_id", sort=False)["supported"].any().sum()),
        "wells": int(diagnostics["well_id"].nunique()),
        "zero_offset_valid_fraction": float(diagnostics["zero_offset_valid"].mean()),
        "all_thirteen_valid_fraction": float(diagnostics["all_thirteen_valid"].mean()),
        "valid_offset_count_distribution": {
            str(int(key)): int(value)
            for key, value in diagnostics["valid_offset_count"].value_counts().sort_index().items()
        },
        "by_distance": by_distance,
        "logical_content_sha256": dataframe_content_sha(diagnostics),
        "coverage_is_gate": False,
    }
    if len(diagnostics) != expected_blocks:
        raise ValueError("support diagnostic block inventory changed")
    return diagnostics, summary


# %% [markdown]
# ## 6. Fixed sparse-anchor Viterbi and row projection


# %%
def tie_priority_by_slot() -> np.ndarray:
    priority_by_offset = {float(offset): priority for priority, offset in enumerate(TIE_ORDER_FT)}
    return np.asarray(
        [priority_by_offset[float(offset)] for offset in OFFSETS_FT],
        dtype=np.int64,
    )


def choose_max_slot(values: np.ndarray, slots: np.ndarray | None = None) -> int:
    values = np.asarray(values, dtype=np.float64)
    candidate_slots = (
        np.arange(len(values), dtype=np.int64)
        if slots is None
        else np.asarray(slots, dtype=np.int64)
    )
    if not len(candidate_slots):
        raise ValueError("cannot choose from an empty state set")
    candidate_values = values[candidate_slots]
    maximum = float(np.max(candidate_values))
    tied = candidate_slots[candidate_values == maximum]
    priorities = tie_priority_by_slot()[tied]
    return int(tied[np.argmin(priorities)])


def blockwise_top1_offset(part: pd.DataFrame) -> float:
    selected = part.loc[part["rsd_valid"] & part["rsd_rank"].eq(1)]
    if selected.empty:
        if bool(part["rsd_valid"].any()):
            raise ValueError("supported block has no immutable valid rank-1 score")
        return 0.0
    if len(selected) != 1:
        raise ValueError("supported block must have one immutable rank-1 score")
    return float(selected["offset_ft"].iloc[0])


def decode_well_viterbi(
    well_scores: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    primary = get_nested(config, "model.primary") or {}
    block_size = int(get_nested(config, "model.frozen_candidate_contract.block_size_rows"))
    scores = well_scores.sort_values(["block_id", "offset_slot"], kind="mergesort").reset_index(
        drop=True
    )
    block_ids = scores["block_id"].drop_duplicates().to_numpy(np.int64)
    if not np.array_equal(block_ids, np.arange(len(block_ids), dtype=np.int64)):
        raise ValueError("score-bank block ids must be contiguous from zero")

    emissions: list[np.ndarray] = []
    supported: list[bool] = []
    valid_counts: list[int] = []
    top1_offsets: list[float] = []
    metadata: list[pd.Series] = []
    for block_id, part in scores.groupby("block_id", sort=True):
        part = part.reset_index(drop=True)
        if not np.array_equal(part["offset_ft"].to_numpy(np.float64), OFFSETS_FT):
            raise ValueError(f"block {block_id} offset order changed")
        valid = part["rsd_valid"].to_numpy(bool)
        score = part["rsd_bin_score"].to_numpy(np.float64)
        if valid.any():
            emission = np.where(valid, score, -np.inf)
        else:
            emission = np.zeros(len(OFFSETS_FT), dtype=np.float64)
        emissions.append(emission)
        supported.append(bool(valid.any()))
        valid_counts.append(int(valid.sum()))
        top1_offsets.append(blockwise_top1_offset(part))
        metadata.append(part.iloc[0])

    initial_center = float(primary["initial_center_ft"])
    initial_sigma = float(primary["initial_sigma_ft"])
    transition_sigma = float(primary["transition_sigma_ft"])
    maximum_initial = float(primary["maximum_initial_step_ft"])
    maximum_adjacent = float(primary["maximum_adjacent_step_ft"])
    state_count = len(OFFSETS_FT)
    block_count = len(emissions)
    dp = np.full((block_count, state_count), -np.inf, dtype=np.float64)
    back = np.full((block_count, state_count), -1, dtype=np.int64)

    initial_allowed = np.abs(OFFSETS_FT - initial_center) <= maximum_initial
    dp[0, initial_allowed] = emissions[0][initial_allowed] - 0.5 * np.square(
        (OFFSETS_FT[initial_allowed] - initial_center) / initial_sigma
    )
    if not np.isfinite(dp[0]).any():
        raise ValueError("no legal initial Viterbi state")

    for block_index in range(1, block_count):
        for current_slot, current_offset in enumerate(OFFSETS_FT):
            if not np.isfinite(emissions[block_index][current_slot]):
                continue
            eligible = np.flatnonzero(
                np.isfinite(dp[block_index - 1])
                & (np.abs(OFFSETS_FT - current_offset) <= maximum_adjacent)
            )
            if not len(eligible):
                continue
            candidate = dp[block_index - 1].copy()
            candidate[eligible] -= 0.5 * np.square(
                (current_offset - OFFSETS_FT[eligible]) / transition_sigma
            )
            predecessor = choose_max_slot(candidate, eligible)
            dp[block_index, current_slot] = (
                candidate[predecessor] + emissions[block_index][current_slot]
            )
            back[block_index, current_slot] = predecessor
        if not np.isfinite(dp[block_index]).any():
            raise ValueError(f"no legal Viterbi state at block {block_index}")

    path_slots = np.empty(block_count, dtype=np.int64)
    path_slots[-1] = choose_max_slot(dp[-1])
    for block_index in range(block_count - 1, 0, -1):
        predecessor = int(back[block_index, path_slots[block_index]])
        if predecessor < 0:
            raise ValueError("Viterbi backpointer is incomplete")
        path_slots[block_index - 1] = predecessor

    rows = []
    for block_index, slot in enumerate(path_slots):
        item = metadata[block_index]
        rows.append(
            {
                "well_id": str(item["well_id"]),
                "fold": int(item["fold"]),
                "block_id": int(item["block_id"]),
                "block_center_suffix_offset": float(
                    int(item["block_id"]) * block_size + block_size / 2
                ),
                "supported": supported[block_index],
                "valid_offset_count": valid_counts[block_index],
                "selected_offset_slot": int(slot),
                "selected_offset_ft": float(OFFSETS_FT[slot]),
                "selected_emission_score": (
                    float(emissions[block_index][slot]) if supported[block_index] else 0.0
                ),
                "cumulative_objective": float(dp[block_index, slot]),
                "blockwise_top1_offset_ft": top1_offsets[block_index],
            }
        )
    return pd.DataFrame(rows)[DATUM_LOGICAL_COLUMNS]


def project_datum_to_rows(
    suffix_offset: np.ndarray,
    datum_path: pd.DataFrame,
    *,
    block_size_rows: int,
) -> np.ndarray:
    offsets = np.asarray(suffix_offset, dtype=np.int64)
    if not np.array_equal(offsets, np.arange(len(offsets), dtype=np.int64)):
        raise ValueError("suffix_offset must be contiguous from zero")
    path = datum_path.sort_values("block_id", kind="mergesort")
    expected_centers = path["block_id"].to_numpy(np.float64) * block_size_rows + block_size_rows / 2
    if not np.array_equal(
        path["block_center_suffix_offset"].to_numpy(np.float64),
        expected_centers,
    ):
        raise ValueError("fixed block-center projection contract changed")
    xp = np.r_[0.0, expected_centers]
    fp = np.r_[0.0, path["selected_offset_ft"].to_numpy(np.float64)]
    return np.interp(offsets.astype(np.float64), xp, fp)


def decode_all_wells(
    score_bank: pd.DataFrame,
    oof_safe: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    block_size = int(get_nested(config, "model.frozen_candidate_contract.block_size_rows"))
    score_by_well = {
        str(well_id): part.reset_index(drop=True)
        for well_id, part in score_bank.groupby("well_id", sort=True)
    }
    datum_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    maximum_slope = 0.0
    for well_id, safe_part in oof_safe.groupby("well_id", sort=True):
        well_id = str(well_id)
        if well_id not in score_by_well:
            raise ValueError(f"score bank missing well {well_id}")
        safe_part = safe_part.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
        datum = decode_well_viterbi(score_by_well[well_id], config)
        expected_blocks = safe_part["suffix_offset"].to_numpy(np.int64) // block_size
        if datum["block_id"].tolist() != np.unique(expected_blocks).tolist():
            raise ValueError(f"score/OOF block inventory mismatch for {well_id}")
        correction = project_datum_to_rows(
            safe_part["suffix_offset"].to_numpy(np.int64),
            datum,
            block_size_rows=block_size,
        )
        if len(correction) > 1:
            maximum_slope = max(
                maximum_slope,
                float(np.max(np.abs(np.diff(correction)))),
            )
        top1_by_block = datum.set_index("block_id")["blockwise_top1_offset_ft"]
        blockwise = top1_by_block.loc[expected_blocks].to_numpy(np.float64)
        base = safe_part["tvt_pred"].to_numpy(np.float64)
        prediction_parts.append(
            pd.DataFrame(
                {
                    "well_id": well_id,
                    "row_idx": safe_part["row_idx"].to_numpy(np.int64),
                    "suffix_offset": safe_part["suffix_offset"].to_numpy(np.int64),
                    "fold": safe_part["fold"].to_numpy(np.int64),
                    "base_tvt_pred": base,
                    "datum_correction_ft": correction,
                    "primary_tvt_pred": base + correction,
                    "blockwise_correction_ft": blockwise,
                    "blockwise_tvt_pred": base + blockwise,
                }
            )
        )
        datum_parts.append(datum)
    datum_path = (
        pd.concat(datum_parts, ignore_index=True)
        .sort_values(["well_id", "block_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    probe_well = str(get_nested(config, "validation.fixed_probe_well"))
    probe_predictions = predictions.loc[predictions["well_id"].eq(probe_well)]
    evidence = {
        "datum_rows": len(datum_path),
        "prediction_rows": len(predictions),
        "datum_logical_sha256": dataframe_content_sha(datum_path, DATUM_LOGICAL_COLUMNS),
        "prediction_logical_sha256": dataframe_content_sha(predictions, PREDICTION_LOGICAL_COLUMNS),
        "maximum_row_correction_slope_ft": maximum_slope,
        "duplicate_prediction_identity_rows": int(
            predictions.duplicated(["well_id", "row_idx"]).sum()
        ),
        "fixed_probe_well": probe_well,
        "fixed_probe_rows": len(probe_predictions),
        "fixed_probe_prediction_logical_sha256": (
            dataframe_content_sha(probe_predictions, PREDICTION_LOGICAL_COLUMNS)
            if len(probe_predictions)
            else None
        ),
    }
    return datum_path, predictions, evidence


# %% [markdown]
# ## 7. Target-free prediction freeze and independent rerun


# %%
def build_prediction_freeze(
    *,
    config: Mapping[str, Any],
    score_bank: pd.DataFrame,
    well_manifest: pd.DataFrame,
    input_manifest: pd.DataFrame,
    support: pd.DataFrame,
    datum_path: pd.DataFrame,
    predictions: pd.DataFrame,
    first_evidence: Mapping[str, Any],
    rerun_evidence: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> dict[str, Any]:
    probe_well = str(get_nested(config, "validation.fixed_probe_well"))
    probe_first = predictions.loc[predictions["well_id"].eq(probe_well)]
    checks = {
        "score_regeneration_zero": True,
        "score_rows": len(score_bank) == int(get_nested(config, "validation.expected_score_rows")),
        "datum_rows": len(datum_path) == int(get_nested(config, "validation.expected_blocks")),
        "prediction_rows": len(predictions) == int(get_nested(config, "validation.expected_rows")),
        "prediction_wells": predictions["well_id"].nunique()
        == int(get_nested(config, "validation.expected_wells")),
        "duplicate_prediction_identity_zero": int(
            first_evidence["duplicate_prediction_identity_rows"]
        )
        == 0,
        "full_prediction_sha_rerun_match": (
            first_evidence["prediction_logical_sha256"]
            == rerun_evidence["prediction_logical_sha256"]
        ),
        "full_datum_sha_rerun_match": (
            first_evidence["datum_logical_sha256"] == rerun_evidence["datum_logical_sha256"]
        ),
        "fixed_probe_present": len(probe_first) > 0,
        "fixed_probe_prediction_sha_rerun_match": (
            first_evidence["fixed_probe_prediction_logical_sha256"]
            == rerun_evidence["fixed_probe_prediction_logical_sha256"]
        ),
        "truth_reads_before_freeze_zero": ledger.truth_rows_before_freeze == 0,
        "hidden_role_reads_before_freeze_zero": ledger.hidden_role_rows_before_freeze == 0,
        "episode_reads_before_freeze_zero": ledger.episode_rows_before_freeze == 0,
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"target-free prediction freeze failed: {failed}")
    payload = {
        "config_content_sha256": mapping_sha256(config),
        "score_content_sha256": dataframe_content_sha(score_bank, SCORE_LOGICAL_COLUMNS),
        "well_manifest_content_sha256": dataframe_content_sha(well_manifest),
        "input_manifest_content_sha256": dataframe_content_sha(input_manifest),
        "support_content_sha256": dataframe_content_sha(support),
        "datum_path_content_sha256": first_evidence["datum_logical_sha256"],
        "prediction_content_sha256": first_evidence["prediction_logical_sha256"],
        "fixed_probe_well": probe_well,
        "fixed_probe_prediction_sha256": first_evidence["fixed_probe_prediction_logical_sha256"],
        "independent_rerun": {
            "datum_path_content_sha256": rerun_evidence["datum_logical_sha256"],
            "prediction_content_sha256": rerun_evidence["prediction_logical_sha256"],
            "fixed_probe_prediction_sha256": rerun_evidence[
                "fixed_probe_prediction_logical_sha256"
            ],
        },
        "checks": checks,
    }
    payload["freeze_contract_sha256"] = mapping_sha256(payload)
    ledger.mark_frozen(str(payload["prediction_content_sha256"]))
    return payload


# %% [markdown]
# ## 8. Post-freeze truth, raw-GR, hidden-like, and episode joins


# %%
def load_exp226_truth(path: Path, ledger: TruthAccessLedger) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["well_id", "row_idx", "tvt_true"],
        dtype={"well_id": str},
    )
    ledger.register_truth_access(len(frame))
    frame["well_id"] = frame["well_id"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["tvt_true"] = pd.to_numeric(frame["tvt_true"], errors="raise").astype(np.float64)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("truth contains duplicate row identity")
    if not np.isfinite(frame["tvt_true"].to_numpy(np.float64)).all():
        raise ValueError("truth must be finite")
    return frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)


def load_hidden_like_assignments(
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like_assignment") or {}
    path, evidence = resolve_file(
        [str(value) for value in spec["patterns"]],
        label="exp115 hidden-like assignment",
        expected_file_sha256=str(spec["expected_sha256"]),
    )
    role_columns = [str(value) for value in spec["role_columns"].values()]
    frame = pd.read_csv(path, usecols=["well_id", *role_columns], dtype={"well_id": str})
    ledger.register_hidden_role_access(len(frame))
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment must be one row per well")
    evidence.update({"name": "hidden_like_assignment", "rows": len(frame)})
    return frame, evidence


def load_persistent_episodes(
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.persistent_episode") or {}
    path, evidence = resolve_file(
        [str(value) for value in spec["patterns"]],
        label="exp226 persistent-offset episodes",
        expected_file_sha256=str(spec["expected_sha256"]),
    )
    required = [
        "well_id",
        "fold",
        "start_suffix_offset",
        "end_suffix_offset_exclusive",
        "rows",
        "episode_sse",
    ]
    frame = pd.read_csv(path, usecols=required, dtype={"well_id": str})
    ledger.register_episode_access(len(frame))
    for column in (
        "fold",
        "start_suffix_offset",
        "end_suffix_offset_exclusive",
        "rows",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["episode_sse"] = pd.to_numeric(frame["episode_sse"], errors="raise").astype(np.float64)
    if (
        not (frame["end_suffix_offset_exclusive"] - frame["start_suffix_offset"])
        .eq(frame["rows"])
        .all()
    ):
        raise ValueError("persistent episode interval/row contract changed")
    evidence.update(
        {
            "name": "persistent_offset_episodes",
            "episodes": len(frame),
            "wells": int(frame["well_id"].nunique()),
            "rows": int(frame["rows"].sum()),
        }
    )
    return (
        frame.sort_values(["well_id", "start_suffix_offset"], kind="mergesort").reset_index(
            drop=True
        ),
        evidence,
    )


def resolve_train_root(config: Mapping[str, Any]) -> Path:
    for raw in get_nested(config, "data.raw_train.root_candidates") or ():
        path = Path(str(raw))
        candidate = path if path.is_absolute() else project_root() / path
        if candidate.is_dir() and any(candidate.glob("*__horizontal_well.csv")):
            return candidate
    raise FileNotFoundError("could not resolve raw train root")


def load_raw_row_context(
    predictions: pd.DataFrame,
    input_manifest: pd.DataFrame,
    score_bank: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_root = resolve_train_root(config)
    manifest = input_manifest.set_index("well_id")
    score_blocks = (
        score_bank.groupby(["well_id", "block_id"], sort=False)
        .first()[["observed_gr_share"]]
        .reset_index()
    )
    block_size = int(get_nested(config, "model.frozen_candidate_contract.block_size_rows"))
    parts: list[pd.DataFrame] = []
    checked_files = 0
    for well_id, prediction in predictions.groupby("well_id", sort=True):
        well_id = str(well_id)
        if well_id not in manifest.index:
            raise ValueError(f"input manifest missing {well_id}")
        path = train_root / f"{well_id}__horizontal_well.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_sha = str(manifest.loc[well_id, "horizontal_raw_sha256"])
        if sha256_path(path) != expected_sha:
            raise ValueError(f"raw horizontal SHA changed for {well_id}")
        checked_files += 1
        raw = pd.read_csv(path, usecols=["MD", "GR", "TVT_input"])
        for column in ("MD", "GR", "TVT_input"):
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        known = np.flatnonzero(raw["TVT_input"].notna().to_numpy())
        if not len(known):
            raise ValueError(f"{well_id} has no known prefix")
        row_idx = prediction["row_idx"].to_numpy(np.int64)
        if row_idx.min() < 0 or row_idx.max() >= len(raw):
            raise ValueError(f"{well_id} row_idx is outside raw horizontal")
        if raw.iloc[row_idx]["TVT_input"].notna().any():
            raise ValueError(f"{well_id} OOF rows overlap the known prefix")
        context = prediction[["well_id", "row_idx"]].copy()
        context["md_since_ft"] = raw["MD"].to_numpy(np.float64)[row_idx] - float(
            raw["MD"].iloc[int(known[-1])]
        )
        context["raw_gr_missing"] = ~np.isfinite(raw["GR"].to_numpy(np.float64)[row_idx])
        context["block_id"] = prediction["suffix_offset"].to_numpy(np.int64) // block_size
        parts.append(context)
    result = pd.concat(parts, ignore_index=True)
    result = result.merge(
        score_blocks,
        on=["well_id", "block_id"],
        how="left",
        validate="many_to_one",
    )
    if result["observed_gr_share"].isna().any():
        raise ValueError("raw row context is missing score-block GR support")
    computed = (
        result.assign(raw_gr_observed=~result["raw_gr_missing"])
        .groupby(["well_id", "block_id"], sort=False)["raw_gr_observed"]
        .mean()
        .rename("computed")
        .reset_index()
        .merge(score_blocks, on=["well_id", "block_id"], validate="one_to_one")
    )
    maximum_share_difference = float(
        np.max(np.abs(computed["computed"] - computed["observed_gr_share"]))
    )
    if maximum_share_difference > 1.0e-12:
        raise ValueError("raw GR support does not reproduce the frozen exp426 score bank")
    return result, {
        "train_root": str(train_root),
        "horizontal_files_sha_checked": checked_files,
        "maximum_block_observed_gr_share_difference": maximum_share_difference,
    }


def role_is_active(value: Any) -> bool:
    normalized = str(value).strip().lower()
    return normalized not in {"", "0", "false", "none", "nan", "train"}


def attach_post_freeze_evaluation(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    hidden_like: pd.DataFrame,
    raw_context: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    frame = predictions.merge(
        truth,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
    ).merge(
        raw_context,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    if len(frame) != len(predictions) or frame["tvt_true"].isna().any():
        raise ValueError("post-freeze truth coverage is incomplete")
    role_columns = get_nested(config, "data.hidden_like_assignment.role_columns") or {}
    roles = hidden_like.set_index("well_id")
    for scope, role_column in role_columns.items():
        active = {
            str(well_id): role_is_active(value)
            for well_id, value in roles[str(role_column)].items()
        }
        frame[str(scope)] = frame["well_id"].map(active).fillna(False).astype(bool)
    frame["base_error"] = frame["base_tvt_pred"] - frame["tvt_true"]
    frame["primary_error"] = frame["primary_tvt_pred"] - frame["tvt_true"]
    frame["blockwise_error"] = frame["blockwise_tvt_pred"] - frame["tvt_true"]
    if not np.isfinite(
        frame[["base_error", "primary_error", "blockwise_error"]].to_numpy(np.float64)
    ).all():
        raise ValueError("post-freeze prediction errors must be finite")
    return frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)


# %% [markdown]
# ## 9. Scope, fold, by-well, and mechanism readouts


# %%
def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def original_episode_mask(
    evaluation: pd.DataFrame,
    episodes: pd.DataFrame,
) -> np.ndarray:
    mask = np.zeros(len(evaluation), dtype=bool)
    position = pd.Series(
        np.arange(len(evaluation), dtype=np.int64),
        index=pd.MultiIndex.from_frame(evaluation[["well_id", "suffix_offset"]]),
    )
    for row in episodes.itertuples(index=False):
        offsets = np.arange(
            int(row.start_suffix_offset),
            int(row.end_suffix_offset_exclusive),
            dtype=np.int64,
        )
        keys = pd.MultiIndex.from_arrays([np.repeat(str(row.well_id), len(offsets)), offsets])
        selected = position.reindex(keys)
        if selected.isna().any():
            raise ValueError(f"persistent episode rows missing for {row.well_id}")
        indices = selected.to_numpy(np.int64)
        if mask[indices].any():
            raise ValueError("persistent episode intervals overlap")
        mask[indices] = True
    if int(mask.sum()) != int(episodes["rows"].sum()):
        raise ValueError("persistent episode row inventory changed")
    return mask


def build_episode_readout(
    evaluation: pd.DataFrame,
    episodes: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    original_mask = original_episode_mask(evaluation, episodes)
    detail_rows: list[dict[str, Any]] = []
    original_base_sse = 0.0
    original_primary_sse = 0.0
    improved_wells = 0
    episode_wells = 0

    episode_by_well = {
        str(well_id): part for well_id, part in episodes.groupby("well_id", sort=False)
    }
    for well_id, part in evaluation.groupby("well_id", sort=True):
        well_id = str(well_id)
        well_episodes = episode_by_well.get(well_id)
        if well_episodes is None:
            continue
        base_error = part["base_error"].to_numpy(np.float64)
        primary_error = part["primary_error"].to_numpy(np.float64)
        well_base_sse = 0.0
        well_primary_sse = 0.0
        for episode_index, episode in enumerate(well_episodes.itertuples(index=False)):
            start = int(episode.start_suffix_offset)
            end = int(episode.end_suffix_offset_exclusive)
            base_sse = float(np.sum(np.square(base_error[start:end])))
            primary_sse = float(np.sum(np.square(primary_error[start:end])))
            well_base_sse += base_sse
            well_primary_sse += primary_sse
            detail_rows.append(
                {
                    "episode_kind": "original_exp226",
                    "episode_id": f"{well_id}:original:{episode_index:03d}",
                    "well_id": well_id,
                    "fold": int(episode.fold),
                    "start_suffix_offset": start,
                    "end_suffix_offset_exclusive": end,
                    "rows": end - start,
                    "base_sse": base_sse,
                    "primary_sse": primary_sse,
                    "new_rows": 0,
                    "new_primary_sse": 0.0,
                    "improved": primary_sse < base_sse,
                }
            )
        original_base_sse += well_base_sse
        original_primary_sse += well_primary_sse
        episode_wells += 1
        improved_wells += int(well_primary_sse < well_base_sse)

    threshold = float(get_nested(config, "data.persistent_episode.threshold_ft"))
    minimum_rows = int(get_nested(config, "data.persistent_episode.minimum_rows"))
    new_episode_sse = 0.0
    corrected_episode_count = 0
    for well_id, part in evaluation.groupby("well_id", sort=True):
        error = part["primary_error"].to_numpy(np.float64)
        global_indices = part.index.to_numpy(np.int64)
        for episode_index, (start, end) in enumerate(
            contiguous_true_runs(np.abs(error) >= threshold)
        ):
            if end - start < minimum_rows:
                continue
            corrected_episode_count += 1
            global_window = global_indices[start:end]
            new_mask = ~original_mask[global_window]
            run_new_sse = float(np.sum(np.square(error[start:end][new_mask])))
            new_episode_sse += run_new_sse
            detail_rows.append(
                {
                    "episode_kind": "corrected_detected",
                    "episode_id": f"{well_id}:corrected:{episode_index:03d}",
                    "well_id": str(well_id),
                    "fold": int(part["fold"].iloc[0]),
                    "start_suffix_offset": start,
                    "end_suffix_offset_exclusive": end,
                    "rows": end - start,
                    "base_sse": float("nan"),
                    "primary_sse": float(np.sum(np.square(error[start:end]))),
                    "new_rows": int(new_mask.sum()),
                    "new_primary_sse": run_new_sse,
                    "improved": False,
                }
            )
    corrected_total_sse = float(np.sum(np.square(evaluation["primary_error"].to_numpy(np.float64))))
    summary = {
        "original_episodes": len(episodes),
        "original_episode_wells": episode_wells,
        "original_episode_rows": int(original_mask.sum()),
        "original_base_sse": original_base_sse,
        "original_primary_sse": original_primary_sse,
        "persistent_episode_sse_reduction": (
            (original_base_sse - original_primary_sse) / original_base_sse
            if original_base_sse
            else None
        ),
        "persistent_episode_wells_improved": improved_wells,
        "persistent_episode_well_improvement_fraction": (
            improved_wells / episode_wells if episode_wells else None
        ),
        "corrected_detected_episodes": corrected_episode_count,
        "new_corrected_episode_sse": new_episode_sse,
        "corrected_total_sse": corrected_total_sse,
        "new_episode_sse_fraction_of_corrected_total": (
            new_episode_sse / corrected_total_sse if corrected_total_sse else None
        ),
        "new_episode_definition": (
            "corrected abs-error >=10 ft contiguous runs >=128 rows; only rows outside "
            "the frozen exp226 episode union contribute new-episode SSE"
        ),
    }
    return pd.DataFrame(detail_rows), summary, original_mask


def scope_mask(
    frame: pd.DataFrame,
    scope: str,
    persistent_mask: np.ndarray,
) -> np.ndarray:
    if scope == "pooled":
        return np.ones(len(frame), dtype=bool)
    distance = frame["md_since_ft"].to_numpy(np.float64)
    if scope == "distance_0_50":
        return distance < 50.0
    if scope == "distance_50_100":
        return (distance >= 50.0) & (distance < 100.0)
    if scope == "distance_100_500":
        return (distance >= 100.0) & (distance < 500.0)
    if scope == "distance_500_1000":
        return (distance >= 500.0) & (distance < 1000.0)
    if scope == "distance_1000_plus":
        return distance >= 1000.0
    if scope == "raw_gr_observed":
        return ~frame["raw_gr_missing"].to_numpy(bool)
    if scope == "raw_gr_missing":
        return frame["raw_gr_missing"].to_numpy(bool)
    if scope == "high_missing":
        return frame["observed_gr_share"].to_numpy(np.float64) < 0.5
    if scope in {"hidden_like_spatial", "hidden_like_typewell_purged"}:
        return frame[scope].to_numpy(bool)
    if scope == "persistent_episode":
        return np.asarray(persistent_mask, dtype=bool)
    if scope == "by_well":
        return np.ones(len(frame), dtype=bool)
    raise ValueError(f"unknown report scope {scope}")


def prediction_metric_row(
    frame: pd.DataFrame,
    *,
    scope: str,
    persistent_mask: np.ndarray,
) -> dict[str, Any]:
    mask = scope_mask(frame, scope, persistent_mask)
    scoped = frame.loc[mask]
    rows = len(scoped)
    if not rows:
        return {
            "scope": scope,
            "rows": 0,
            "wells": 0,
            "base_rmse": None,
            "primary_rmse": None,
            "primary_gain_ft": None,
            "primary_delta_rmse_ft": None,
            "blockwise_rmse": None,
            "blockwise_gain_ft": None,
        }
    base_rmse = float(np.sqrt(np.mean(np.square(scoped["base_error"]))))
    primary_rmse = float(np.sqrt(np.mean(np.square(scoped["primary_error"]))))
    blockwise_rmse = float(np.sqrt(np.mean(np.square(scoped["blockwise_error"]))))
    return {
        "scope": scope,
        "rows": rows,
        "wells": int(scoped["well_id"].nunique()),
        "base_rmse": base_rmse,
        "primary_rmse": primary_rmse,
        "primary_gain_ft": base_rmse - primary_rmse,
        "primary_delta_rmse_ft": primary_rmse - base_rmse,
        "blockwise_rmse": blockwise_rmse,
        "blockwise_gain_ft": base_rmse - blockwise_rmse,
    }


def build_prediction_metrics(
    evaluation: pd.DataFrame,
    persistent_mask: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scope_rows = []
    for scope in get_nested(config, "validation.report_scopes"):
        if str(scope) not in {"fold", "by_well"}:
            scope_rows.append(
                prediction_metric_row(
                    evaluation,
                    scope=str(scope),
                    persistent_mask=persistent_mask,
                )
            )
    fold_rows = []
    for fold, part in evaluation.groupby("fold", sort=True):
        local_persistent = persistent_mask[part.index.to_numpy(np.int64)]
        fold_rows.append(
            {
                "fold": int(fold),
                **prediction_metric_row(
                    part.reset_index(drop=True),
                    scope="pooled",
                    persistent_mask=local_persistent,
                ),
            }
        )
    well_rows = []
    for well_id, part in evaluation.groupby("well_id", sort=True):
        base_rmse = float(np.sqrt(np.mean(np.square(part["base_error"]))))
        primary_rmse = float(np.sqrt(np.mean(np.square(part["primary_error"]))))
        blockwise_rmse = float(np.sqrt(np.mean(np.square(part["blockwise_error"]))))
        well_rows.append(
            {
                "well_id": str(well_id),
                "fold": int(part["fold"].iloc[0]),
                "rows": len(part),
                "base_rmse": base_rmse,
                "primary_rmse": primary_rmse,
                "primary_gain_ft": base_rmse - primary_rmse,
                "primary_delta_rmse_ft": primary_rmse - base_rmse,
                "blockwise_rmse": blockwise_rmse,
                "blockwise_gain_ft": base_rmse - blockwise_rmse,
            }
        )
    return pd.DataFrame(scope_rows), pd.DataFrame(fold_rows), pd.DataFrame(well_rows)


def choose_min_slot(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float64)
    minimum = float(np.min(values))
    tied = np.flatnonzero(values == minimum)
    priorities = tie_priority_by_slot()[tied]
    return int(tied[np.argmin(priorities)])


def build_block_diagnostic_metrics(
    score_bank: pd.DataFrame,
    evaluation: pd.DataFrame,
    datum_path: pd.DataFrame,
) -> pd.DataFrame:
    score_groups = {
        (str(well_id), int(block_id)): part.reset_index(drop=True)
        for (well_id, block_id), part in score_bank.groupby(["well_id", "block_id"], sort=False)
    }
    datum = datum_path.set_index(["well_id", "block_id"])
    rows = []
    for (well_id, block_id), part in evaluation.groupby(["well_id", "block_id"], sort=True):
        key = (str(well_id), int(block_id))
        scores = score_groups[key]
        base = part["base_tvt_pred"].to_numpy(np.float64)
        truth = part["tvt_true"].to_numpy(np.float64)
        sse = np.asarray(
            [np.sum(np.square(base + offset - truth)) for offset in OFFSETS_FT],
            dtype=np.float64,
        )
        oracle_slot = choose_min_slot(sse)
        valid = scores["rsd_valid"].to_numpy(bool)
        supported = bool(valid.any())
        top1_slot = (
            int(np.flatnonzero(valid & scores["rsd_rank"].eq(1).to_numpy(bool))[0])
            if supported
            else int(np.flatnonzero(OFFSETS_FT == 0.0)[0])
        )
        top3_offsets = set(
            scores.loc[scores["rsd_valid"] & scores["rsd_top3"], "offset_ft"].astype(float).tolist()
        )
        primary_offset = float(datum.loc[key, "selected_offset_ft"])
        oracle_offset = float(OFFSETS_FT[oracle_slot])
        rows.append(
            {
                "well_id": str(well_id),
                "fold": int(part["fold"].iloc[0]),
                "block_id": int(block_id),
                "rows": len(part),
                "supported": supported,
                "oracle_offset_ft": oracle_offset,
                "blockwise_top1_offset_ft": float(OFFSETS_FT[top1_slot]),
                "viterbi_offset_ft": primary_offset,
                "top1_exact": bool(supported and top1_slot == oracle_slot),
                "top3_coverage": bool(supported and oracle_offset in top3_offsets),
                "blockwise_direction_correct": bool(
                    supported
                    and oracle_offset != 0.0
                    and np.sign(OFFSETS_FT[top1_slot]) == np.sign(oracle_offset)
                ),
                "viterbi_direction_correct": bool(
                    oracle_offset != 0.0 and np.sign(primary_offset) == np.sign(oracle_offset)
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_block_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    supported = frame.loc[frame["supported"]]
    nonzero = supported.loc[supported["oracle_offset_ft"].ne(0.0)]
    return {
        "blocks": len(frame),
        "supported_blocks": len(supported),
        "supported_block_fraction": float(frame["supported"].mean()),
        "blockwise_top1_exact": (float(supported["top1_exact"].mean()) if len(supported) else None),
        "blockwise_top3_coverage": (
            float(supported["top3_coverage"].mean()) if len(supported) else None
        ),
        "nonzero_oracle_supported_blocks": len(nonzero),
        "blockwise_direction_accuracy": (
            float(nonzero["blockwise_direction_correct"].mean()) if len(nonzero) else None
        ),
        "viterbi_direction_accuracy": (
            float(nonzero["viterbi_direction_correct"].mean()) if len(nonzero) else None
        ),
    }


# %% [markdown]
# ## 10. Technical and scientific gates


# %%
def metric_row(frame: pd.DataFrame, scope: str) -> pd.Series:
    selected = frame.loc[frame["scope"].eq(scope)]
    if len(selected) != 1:
        raise ValueError(f"expected one metric row for scope={scope}")
    return selected.iloc[0]


def finite_metric(value: Any, *, default: float = float("-inf")) -> float:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return default
    return float(value)


def evaluate_gates(
    *,
    config: Mapping[str, Any],
    input_evidence: Sequence[Mapping[str, Any]],
    freeze: Mapping[str, Any],
    first_decode: Mapping[str, Any],
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    by_well_metrics: pd.DataFrame,
    episode_summary: Mapping[str, Any],
    ledger: TruthAccessLedger,
    elapsed_seconds: float,
    peak_memory_gb: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    technical_spec = get_nested(config, "gates.technical") or {}
    pooled = metric_row(scope_metrics, "pooled")
    parent_rmse = finite_metric(pooled["base_rmse"])
    expected_parent = float(get_nested(config, "validation.parent_rmse"))
    technical_checks = {
        "input_sha_and_inventory": all(
            all(item.get("contract_checks", {"resolved": True}).values()) for item in input_evidence
        ),
        "prediction_freeze_checks": all(freeze["checks"].values()),
        "parent_rmse_parity": abs(parent_rmse - expected_parent)
        <= float(get_nested(config, "validation.parent_rmse_atol_ft")),
        "truth_reads_before_freeze_zero": ledger.truth_rows_before_freeze == 0,
        "hidden_role_reads_before_freeze_zero": ledger.hidden_role_rows_before_freeze == 0,
        "episode_reads_before_freeze_zero": ledger.episode_rows_before_freeze == 0,
        "maximum_row_correction_slope": finite_metric(
            first_decode["maximum_row_correction_slope_ft"]
        )
        <= float(technical_spec["maximum_row_correction_slope_ft"]),
        "runtime": elapsed_seconds <= float(technical_spec["maximum_runtime_seconds"]),
        "peak_memory": peak_memory_gb <= float(technical_spec["maximum_peak_rss_gb"]),
    }
    technical_passed = bool(all(technical_checks.values()))

    scientific_spec = get_nested(config, "gates.scientific") or {}
    fold_improvement_count = int((fold_metrics["primary_gain_ft"] > 0.0).sum())
    guarded_scopes = {}
    for scope in scientific_spec["guarded_scopes"]:
        row = metric_row(scope_metrics, str(scope))
        delta = finite_metric(row["primary_delta_rmse_ft"], default=float("inf"))
        guarded_scopes[str(scope)] = {
            "primary_delta_rmse_ft": delta,
            "maximum_regression_ft": float(scientific_spec["maximum_scoped_regression_ft"]),
            "passed": delta <= float(scientific_spec["maximum_scoped_regression_ft"]),
        }
    thousand = metric_row(scope_metrics, "distance_1000_plus")
    by_well_delta = by_well_metrics["primary_delta_rmse_ft"].to_numpy(np.float64)
    p95 = float(np.quantile(by_well_delta, 0.95))
    worst = float(np.max(by_well_delta))
    scientific_checks = {
        "minimum_rmse_gain_vs_exp226": finite_metric(pooled["primary_gain_ft"])
        >= float(scientific_spec["minimum_rmse_gain_vs_exp226_ft"]),
        "minimum_improvement_folds": fold_improvement_count
        >= int(scientific_spec["minimum_improvement_folds"]),
        "minimum_distance_1000_plus_gain": finite_metric(thousand["primary_gain_ft"])
        >= float(scientific_spec["minimum_distance_1000_plus_gain_ft"]),
        "minimum_persistent_episode_sse_reduction": finite_metric(
            episode_summary["persistent_episode_sse_reduction"]
        )
        >= float(scientific_spec["minimum_persistent_episode_sse_reduction"]),
        "minimum_persistent_well_improvement_fraction": finite_metric(
            episode_summary["persistent_episode_well_improvement_fraction"]
        )
        >= float(scientific_spec["minimum_persistent_well_improvement_fraction"]),
        "guarded_scope_non_regression": all(item["passed"] for item in guarded_scopes.values()),
        "maximum_new_episode_sse_fraction": finite_metric(
            episode_summary["new_episode_sse_fraction_of_corrected_total"],
            default=float("inf"),
        )
        <= float(scientific_spec["maximum_new_episode_sse_fraction_of_corrected_total"]),
        "maximum_by_well_delta_rmse_p95": p95
        <= float(scientific_spec["maximum_by_well_delta_rmse_p95_ft"]),
        "maximum_worst_well_regression": worst
        <= float(scientific_spec["maximum_worst_well_regression_ft"]),
    }
    scientific_passed = bool(technical_passed and all(scientific_checks.values()))
    gate = {
        "technical_passed": technical_passed,
        "scientific_passed": scientific_passed,
        "all_passed": scientific_passed,
        "technical_checks": technical_checks,
        "scientific_checks": scientific_checks,
        "pooled": pooled.to_dict(),
        "improvement_folds": fold_improvement_count,
        "distance_1000_plus": thousand.to_dict(),
        "guarded_scopes": guarded_scopes,
        "by_well_delta_rmse_p95_ft": p95,
        "worst_well_regression_ft": worst,
        "episode_summary": dict(episode_summary),
    }
    if not technical_passed:
        action = "technical_fail_close_without_scientific_claim_or_rescue"
    elif not scientific_passed:
        action = "scientific_fail_close_sparse_anchor_branch_without_rescue"
    else:
        action = "train_side_pf_beam_candidate_requires_separate_inference_design"
    decision = {
        "action": action,
        "exp426_decision_remains": "technical_failed_closed",
        "same_oof_rescue_allowed": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    return gate, decision


# %% [markdown]
# ## 11. Metrics and generated artifacts


# %%
def save_results(
    *,
    support: pd.DataFrame,
    datum_path: pd.DataFrame,
    predictions: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    by_well_metrics: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    block_diagnostic_metrics: pd.DataFrame,
    block_diagnostic_summary: Mapping[str, Any],
    input_evidence: Sequence[Mapping[str, Any]],
    raw_context_evidence: Mapping[str, Any],
    support_summary: Mapping[str, Any],
    freeze: Mapping[str, Any],
    scientific_contract: Mapping[str, Any],
    gate: Mapping[str, Any],
    decision: Mapping[str, Any],
    ledger: TruthAccessLedger,
    elapsed_seconds: float,
    peak_memory_gb: float,
) -> dict[str, Any]:
    output = artifact_dir()
    artifacts: dict[str, Any] = {
        "support_diagnostics": {},
        "datum_path": write_csv_gzip(
            datum_path,
            output / f"{OUTPUT_PREFIX}_datum_path.csv.gz",
        ),
        "row_predictions": write_csv_gzip(
            predictions,
            output / f"{OUTPUT_PREFIX}_row_predictions.csv.gz",
        ),
    }
    plain_frames = {
        "support_diagnostics": support,
        "scope_metrics": scope_metrics,
        "fold_metrics": fold_metrics,
        "by_well_metrics": by_well_metrics,
        "episode_metrics": episode_metrics,
        "block_diagnostic_metrics": block_diagnostic_metrics,
    }
    for label, frame in plain_frames.items():
        path = output / f"{OUTPUT_PREFIX}_{label}.csv"
        frame.to_csv(path, index=False)
        artifacts[label] = {
            "path": str(path),
            "rows": len(frame),
            "raw_sha256": sha256_path(path),
            "logical_content_sha256": dataframe_content_sha(frame),
            "schema_sha256": dataframe_schema_sha(frame),
        }

    status = (
        "completed_all_gates_passed_train_side_candidate"
        if gate["all_passed"]
        else (
            "completed_scientific_failed_closed"
            if gate["technical_passed"]
            else "completed_technical_failed_closed"
        )
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "stage": "frozen_sparse_anchor_direct_oof_readout",
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scientific_contract": scientific_contract,
        "input_evidence": list(input_evidence),
        "raw_context_evidence": dict(raw_context_evidence),
        "support_summary": dict(support_summary),
        "block_diagnostic_summary": dict(block_diagnostic_summary),
        "prediction_freeze": freeze,
        "truth_access_ledger": ledger.as_dict(),
        "gate": gate,
        "decision": decision,
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_gb": peak_memory_gb,
        "artifacts": artifacts,
    }
    summary_path = output / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    artifacts["summary"] = {
        "path": str(summary_path),
        "raw_sha256": sha256_path(summary_path),
    }
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "metric": "direct_full_oof_rmse_and_fixed_tail_gates",
        "cv": gate["pooled"]["primary_rmse"],
        "parent_cv": gate["pooled"]["base_rmse"],
        "execution_counts": {
            "primary_decoders": 1,
            "diagnostic_replays": 1,
            "wells_decoded": int(predictions["well_id"].nunique()),
            "reporting_folds": int(predictions["fold"].nunique()),
            "model_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "hmm_runs": 0,
            "pf_runs": 0,
            "beam_runs": 0,
            "gpu_runs": 0,
            "parent_control_regeneration": False,
            "score_regeneration": False,
        },
        "support_summary": support_summary,
        "block_diagnostic_summary": block_diagnostic_summary,
        "scope_metrics": scope_metrics.to_dict(orient="records"),
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "prediction_freeze": freeze,
        "truth_access_ledger": ledger.as_dict(),
        "gate": gate,
        "decision": decision,
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_gb": peak_memory_gb,
        "artifacts": artifacts,
        "inference": {"status": "disabled"},
        "submission": {"status": "disabled"},
    }
    write_json(metrics_output_path(), metrics)
    return metrics


def run_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    scientific_contract = validate_scientific_contract(config, require_run_approval=True)
    ledger = TruthAccessLedger()
    (
        score_bank,
        well_manifest,
        input_manifest,
        exp426_evidence,
    ) = load_frozen_exp426_inputs(config)
    oof_safe, exp226_path, exp226_evidence = load_exp226_safe(config)
    support, support_summary = build_support_diagnostics(score_bank, config)

    datum_path, predictions, first_decode = decode_all_wells(score_bank, oof_safe, config)
    _, rerun_predictions, rerun_decode = decode_all_wells(score_bank, oof_safe, config)
    freeze = build_prediction_freeze(
        config=config,
        score_bank=score_bank,
        well_manifest=well_manifest,
        input_manifest=input_manifest,
        support=support,
        datum_path=datum_path,
        predictions=predictions,
        first_evidence=first_decode,
        rerun_evidence=rerun_decode,
        ledger=ledger,
    )
    del rerun_predictions

    truth = load_exp226_truth(exp226_path, ledger)
    hidden_like, hidden_evidence = load_hidden_like_assignments(config, ledger)
    episodes, episode_evidence = load_persistent_episodes(config, ledger)
    raw_context, raw_context_evidence = load_raw_row_context(
        predictions,
        input_manifest,
        score_bank,
        config,
    )
    evaluation = attach_post_freeze_evaluation(
        predictions,
        truth,
        hidden_like,
        raw_context,
        config,
    )
    episode_metrics, episode_summary, persistent_mask = build_episode_readout(
        evaluation,
        episodes,
        config,
    )
    scope_metrics, fold_metrics, by_well_metrics = build_prediction_metrics(
        evaluation,
        persistent_mask,
        config,
    )
    block_diagnostic_metrics = build_block_diagnostic_metrics(
        score_bank,
        evaluation,
        datum_path,
    )
    block_diagnostic_summary = summarize_block_diagnostics(block_diagnostic_metrics)
    elapsed_seconds = time.perf_counter() - started
    peak_memory_gb = peak_rss_gb()
    input_evidence = [
        *exp426_evidence,
        exp226_evidence,
        hidden_evidence,
        episode_evidence,
    ]
    gate, decision = evaluate_gates(
        config=config,
        input_evidence=input_evidence,
        freeze=freeze,
        first_decode=first_decode,
        scope_metrics=scope_metrics,
        fold_metrics=fold_metrics,
        by_well_metrics=by_well_metrics,
        episode_summary=episode_summary,
        ledger=ledger,
        elapsed_seconds=elapsed_seconds,
        peak_memory_gb=peak_memory_gb,
    )
    return save_results(
        support=support,
        datum_path=datum_path,
        predictions=predictions,
        scope_metrics=scope_metrics,
        fold_metrics=fold_metrics,
        by_well_metrics=by_well_metrics,
        episode_metrics=episode_metrics,
        block_diagnostic_metrics=block_diagnostic_metrics,
        block_diagnostic_summary=block_diagnostic_summary,
        input_evidence=input_evidence,
        raw_context_evidence=raw_context_evidence,
        support_summary=support_summary,
        freeze=freeze,
        scientific_contract=scientific_contract,
        gate=gate,
        decision=decision,
        ledger=ledger,
        elapsed_seconds=elapsed_seconds,
        peak_memory_gb=peak_memory_gb,
    )


# %% [markdown]
# ## 12. Setup, configuration preview, and approved execution


# %%
CONFIG = load_experiment_config()
SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)

if EXECUTE_NOTEBOOK:
    print("exp433 frozen sparse-anchor direct OOF readout")
    print(
        {
            "route": get_nested(CONFIG, "experiment.route"),
            "status": get_nested(CONFIG, "experiment.status"),
            "parent": get_nested(CONFIG, "lineage.parent"),
            "base_prediction": get_nested(CONFIG, "lineage.base_prediction"),
            "active_stage": get_nested(CONFIG, "execution_contract.active_stage"),
            "primary_decoders": get_nested(CONFIG, "execution_contract.primary_decoders"),
            "diagnostic_replays": get_nested(CONFIG, "execution_contract.diagnostic_replays"),
            "model_configs": get_nested(CONFIG, "execution_contract.model_configs"),
            "boosters": get_nested(CONFIG, "execution_contract.boosters"),
            "hmm_pf_beam_gpu": [
                get_nested(CONFIG, "execution_contract.hmm_runs"),
                get_nested(CONFIG, "execution_contract.pf_runs"),
                get_nested(CONFIG, "execution_contract.beam_runs"),
                get_nested(CONFIG, "execution_contract.gpu_runs"),
            ],
        }
    )
    display(pd.DataFrame([SCIENTIFIC_CONTRACT]))


# %%
if EXECUTE_NOTEBOOK:
    if bool(get_nested(CONFIG, "execution.run_train")):
        METRICS = run_experiment(CONFIG)
        display(pd.DataFrame([METRICS["gate"]["pooled"]]))
        display(pd.DataFrame(METRICS["fold_metrics"]))
        display(pd.DataFrame(METRICS["scope_metrics"]))
        display(pd.DataFrame([METRICS["support_summary"]]))
        display(pd.DataFrame([METRICS["block_diagnostic_summary"]]))
        print(METRICS["artifacts"])
        print(METRICS["decision"])
    else:
        print(
            "Implementation is ready. Canonical notebook replacement and Kaggle "
            "package/push/run remain disabled pending separate authorization."
        )

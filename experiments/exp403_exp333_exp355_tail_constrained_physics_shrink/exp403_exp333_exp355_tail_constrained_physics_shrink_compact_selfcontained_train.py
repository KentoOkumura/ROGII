# %% [markdown]
# # exp403 exp333/exp355 tail-constrained physics shrink — train-side readout
#
# This saved-OOF-only notebook replaces the two frozen exp263 physics
# components, selects one scalar shrink coefficient on outer-train wells, and
# applies it to the held-out exp226 reporting fold.  All source predictions,
# formulas, identities, and content hashes are frozen before suffix truth or
# hidden-like roles are read.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable contract
# 2. Notebook-safe configuration, hashing, and serialization
# 3. Execution and truth-access guards
# 4. Saved-input resolution and target-free source loaders
# 5. Streaming source assembly and prediction freeze
# 6. Late truth and hidden-like attachment
# 7. Outer-train lambda calibration
# 8. Metrics, persistent-offset diagnostics, and promotion gate
# 9. Orchestration and generated artifacts
# 10. Setup and guarded execution

# %% [markdown]
# ## 1. Imports and immutable contract

# %%
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import os
import resource
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp403_exp333_exp355_tail_constrained_physics_shrink"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
KEY_COLUMNS = ("well_id", "row_idx")
SOURCE_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    "reporting_fold",
    "exp263_generation_fold",
    "md_since",
    "exp226_k16",
    "likpf_mean",
    "exp209_exact_hmm",
    "exp333_stage1",
    "exp355_stage1",
    "exp263_control",
    "full_replacement",
    "correction",
)
EXPECTED_LAMBDAS = (
    0.0,
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
    0.75,
    1.0,
)
EXPECTED_COMPONENT_WEIGHTS = {
    "exp226_k16": 0.50,
    "likpf_mean": 0.25,
    "exp209_exact_hmm": 0.25,
}
CROSSFIT_CANDIDATE = "crossfit_shrink"
CONTROL_CANDIDATE = "exp263_control"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP403_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


# %% [markdown]
# ## 2. Notebook-safe configuration, hashing, and serialization

# %%
def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file() and (candidate / "AGENTS.md").is_file():
            return candidate
    return start


def experiment_dir() -> Path:
    candidate = project_root() / "experiments" / EXPERIMENT_NAME
    return candidate if candidate.is_dir() else Path.cwd()


def resolve_config_path() -> Path:
    candidates = (
        Path.cwd() / "config.yaml",
        experiment_dir() / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
    )
    for path in candidates:
        if not path.is_file():
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if isinstance(value, Mapping) and get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return path
    raise FileNotFoundError(f"config.yaml for {EXPERIMENT_NAME} was not found")


def read_config() -> dict[str, Any]:
    value = yaml.safe_load(resolve_config_path().read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
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
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
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


def dataframe_content_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256(canonical_json_bytes({"columns": selected}))
    row_hash = pd.util.hash_pandas_object(
        frame.loc[:, selected],
        index=False,
        categorize=True,
    ).to_numpy(np.uint64)
    digest.update(np.ascontiguousarray(row_hash).tobytes())
    return digest.hexdigest()


def portable_prediction_surface_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    selected = (
        frame.loc[:, list(columns)]
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "algorithm": "keyed_float_hex_v1",
                "columns": list(columns),
            }
        )
    )
    for values in selected.itertuples(index=False, name=None):
        encoded: list[str] = []
        for column, value in zip(columns, values, strict=True):
            if column == "well_id":
                encoded.append(str(value))
            elif column in {"row_idx", "outer_fold", "reporting_fold"}:
                encoded.append(str(int(value)))
            else:
                encoded.append(float(value).hex())
        digest.update(("\t".join(encoded) + "\n").encode())
    return digest.hexdigest()


class PartitionContentHasher:
    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)
        self._digest = hashlib.sha256(
            canonical_json_bytes({"columns": list(self.columns)})
        )
        self.rows = 0

    def update(self, frame: pd.DataFrame) -> None:
        if tuple(frame.columns) != self.columns:
            raise ValueError("partition columns differ from the frozen source schema")
        row_hash = pd.util.hash_pandas_object(
            frame,
            index=False,
            categorize=True,
        ).to_numpy(np.uint64)
        self._digest.update(np.ascontiguousarray(row_hash).tobytes())
        self.rows += len(frame)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def dataframe_schema_sha256(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    return hashlib.sha256(canonical_json_bytes(schema)).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_file(path),
    }


def write_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
    return {
        "path": str(path),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_file(path),
        "content_sha256": dataframe_content_sha256(frame),
        "schema_sha256": dataframe_schema_sha256(frame),
    }


def write_gzip_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
        compression={"method": "gzip", "mtime": 0},
    )
    return {
        "path": str(path),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": sha256_gzip_decompressed(path),
        "content_sha256": dataframe_content_sha256(frame),
        "schema_sha256": dataframe_schema_sha256(frame),
    }


def output_artifacts_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        return KAGGLE_WORKING_ROOT / "artifacts"
    return experiment_dir() / "artifacts"


def peak_rss_gb() -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


# %% [markdown]
# ## 3. Execution and truth-access guards

# %%
def validate_contract(
    config: Mapping[str, Any],
    *,
    require_run_authorization: bool = False,
) -> dict[str, Any]:
    lambdas = tuple(
        float(value) for value in get_nested(config, "candidate.lambda.candidates", [])
    )
    execution_count = dict(get_nested(config, "candidate.execution_count", {}))
    run_train = bool(get_nested(config, "execution.run_train"))
    canonical_notebook_adopted = bool(
        get_nested(config, "implementation.canonical_notebook_adopted")
    )
    training_enabled = bool(get_nested(config, "implementation.training_enabled"))
    if run_train:
        train_lifecycle_state = all(
            (
                canonical_notebook_adopted,
                training_enabled,
                bool(get_nested(config, "execution.kaggle_package_approved")),
                bool(get_nested(config, "execution.kaggle_train_run_approved")),
                bool(
                    get_nested(
                        config,
                        "execution.canonical_notebook_adoption_approved",
                    )
                ),
            )
        )
    elif canonical_notebook_adopted:
        train_lifecycle_state = all(
            (
                not training_enabled,
                not bool(get_nested(config, "execution.kaggle_train_run_approved")),
                get_nested(config, "results.promotion_gate_passed") is not None,
            )
        )
    else:
        train_lifecycle_state = all(
            (
                not training_enabled,
                not bool(get_nested(config, "execution.kaggle_train_run_approved")),
            )
        )
    checks = {
        "experiment_name": get_nested(config, "experiment.name") == EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route") == "ensemble",
        "implementation_enabled": bool(get_nested(config, "implementation.enabled")),
        "jupytext_source_created": bool(
            get_nested(config, "implementation.jupytext_source_created")
        ),
        "train_lifecycle_state": train_lifecycle_state,
        "inference_disabled": not bool(
            get_nested(config, "implementation.inference_enabled")
        ),
        "submission_disabled": not bool(
            get_nested(config, "implementation.submission_enabled")
        ),
        "fixed_lambdas": lambdas == EXPECTED_LAMBDAS,
        "selection_rule": (
            get_nested(config, "candidate.lambda.selection")
            == "largest_positive_eligible_lambda"
        ),
        "zero_fallback": (
            float(get_nested(config, "candidate.lambda.no_positive_fallback")) == 0.0
        ),
        "reporting_folds": execution_count.get("reporting_folds") == 5,
        "scientific_policy": execution_count.get("scientific_policies") == 1,
        "model_configs": execution_count.get("model_configs") == 0,
        "lightgbm_configs": execution_count.get("lightgbm_configs") == 0,
        "trained_folds": execution_count.get("trained_folds") == 0,
        "boosters": execution_count.get("boosters") == 0,
        "pf_runs": execution_count.get("pf_well_runs") == 0,
        "hmm_runs": execution_count.get("hmm_well_runs") == 0,
        "beam_runs": execution_count.get("beam_well_runs") == 0,
        "parent_reruns": execution_count.get("parent_control_reruns") == 0,
        "run_inference_false": not bool(get_nested(config, "execution.run_inference")),
        "create_submission_false": not bool(
            get_nested(config, "execution.create_submission")
        ),
    }
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise ValueError(f"exp403 frozen contract failed: {failed}")
    if require_run_authorization:
        authorized = bool(get_nested(config, "execution.kaggle_train_run_approved"))
        if not authorized or not run_train:
            raise RuntimeError(
                "exp403 train-side readout is implemented but Kaggle execution is "
                "fail-closed until a separate run approval sets both run flags"
            )
    return {
        "checks": checks,
        "lambda_candidates": list(lambdas),
        "execution_count": execution_count,
        "run_authorized": bool(
            get_nested(config, "execution.kaggle_train_run_approved")
            and get_nested(config, "execution.run_train")
        ),
    }


def contains_truth_token(column: str) -> bool:
    normalized = str(column).strip().lower()
    tokens = ("truth", "true_tvt", "tvt_true", "target", "error", "oracle")
    return any(token in normalized for token in tokens)


def assert_target_free_columns(columns: Iterable[str], stage: str) -> None:
    forbidden = sorted(column for column in columns if contains_truth_token(column))
    if forbidden:
        raise ValueError(f"{stage} requested truth/error columns before freeze: {forbidden}")


@dataclass
class AccessLedger:
    prediction_frozen: bool = False
    truth_columns_read_before_freeze: int = 0
    late_truth_rows: int = 0
    late_hidden_rows: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def pre_freeze_read(self, label: str, columns: Iterable[str], rows: int) -> None:
        selected = [str(column) for column in columns]
        forbidden = [column for column in selected if contains_truth_token(column)]
        if forbidden:
            self.truth_columns_read_before_freeze += len(forbidden)
            raise RuntimeError(f"{label} crossed the truth boundary: {forbidden}")
        self.events.append(
            {"stage": "pre_freeze", "label": label, "columns": selected, "rows": rows}
        )

    def freeze(self, evidence: Mapping[str, Any]) -> None:
        if self.truth_columns_read_before_freeze:
            raise RuntimeError("cannot freeze after an early truth/error read")
        required = {
            "rows",
            "wells",
            "source_schema_sha256",
            "source_content_sha256",
            "formula_sha256",
        }
        missing = required - set(evidence)
        if missing:
            raise RuntimeError(f"freeze evidence is incomplete: {sorted(missing)}")
        self.prediction_frozen = True
        self.events.append({"stage": "prediction_freeze", **dict(evidence)})

    def truth_late(self, label: str, rows: int) -> None:
        if not self.prediction_frozen:
            self.truth_columns_read_before_freeze += 1
            raise RuntimeError("suffix truth requires frozen source predictions")
        self.late_truth_rows += rows
        self.events.append({"stage": "late_truth", "label": label, "rows": rows})

    def hidden_late(self, label: str, rows: int) -> None:
        if not self.prediction_frozen:
            raise RuntimeError("hidden-like roles require frozen source predictions")
        self.late_hidden_rows += rows
        self.events.append({"stage": "late_hidden", "label": label, "rows": rows})


# %% [markdown]
# ## 4. Saved-input resolution and target-free source loaders

# %%
def _candidate_paths(filename: str, configured: Iterable[str]) -> list[Path]:
    root = project_root()
    candidates: list[Path] = []
    for raw in configured:
        pattern = str(raw)
        if any(token in pattern for token in ("*", "?", "[")):
            matches = (
                [Path(path) for path in glob.glob(pattern, recursive=True)]
                if Path(pattern).is_absolute()
                else list(root.glob(pattern))
            )
        else:
            candidate = Path(pattern)
            if not candidate.is_absolute():
                candidate = root / candidate
            matches = [candidate]
        for candidate in matches:
            path = candidate if candidate.name == filename else candidate / filename
            if path.is_file():
                candidates.append(path)
    for search_root in (KAGGLE_INPUT_ROOT, Path("/tmp"), root):
        if search_root.exists():
            candidates.extend(search_root.glob(f"**/{filename}"))
    unique = {str(path.resolve()): path for path in candidates if path.is_file()}
    return sorted(unique.values(), key=str)


def resolve_sha_matched_file(
    *,
    filename: str,
    configured: Iterable[str],
    expected_raw_sha256: str | None,
    expected_decompressed_sha256: str | None,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    candidates = _candidate_paths(filename, configured)
    reports: list[dict[str, Any]] = []
    for path in candidates:
        raw_sha = sha256_file(path)
        if expected_raw_sha256 and raw_sha != expected_raw_sha256:
            continue
        decompressed_sha = (
            sha256_gzip_decompressed(path) if path.suffix == ".gz" else raw_sha
        )
        if expected_decompressed_sha256 and decompressed_sha != expected_decompressed_sha256:
            continue
        reports.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "raw_sha256": raw_sha,
                "decompressed_sha256": decompressed_sha,
            }
        )
    if not reports:
        raise FileNotFoundError(
            f"{label} with the frozen SHA was not found; candidates={len(candidates)}"
        )
    signatures = {
        (item["raw_sha256"], item["decompressed_sha256"]) for item in reports
    }
    if len(signatures) != 1:
        raise ValueError(f"multiple content-distinct {label} inputs matched")
    return Path(str(reports[0]["path"])), reports[0]


def resolve_exp263_cache_root(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = dict(get_nested(config, "data.exp263_cache"))
    filename = str(spec["cache_manifest_filename"])
    expected = str(spec["expected_manifest_sha256"])
    roots: list[Path] = []
    for raw in spec.get("candidates", []):
        path = Path(str(raw))
        if not path.is_absolute():
            path = project_root() / path
        if path.name == filename:
            path = path.parent
        if (path / filename).is_file():
            roots.append(path)
    for search_root in (KAGGLE_INPUT_ROOT, Path("/tmp"), project_root()):
        if search_root.exists():
            roots.extend(path.parent for path in search_root.glob(f"**/{filename}"))
    matches = sorted(
        {
            str(path.resolve()): path
            for path in roots
            if (path / filename).is_file()
            and sha256_file(path / filename) == expected
        }.values(),
        key=str,
    )
    if not matches:
        raise FileNotFoundError("exp263 cache with the frozen manifest SHA was not found")
    required = tuple(dict(spec["candidate_partitions"]))
    materialized = [
        path
        for path in matches
        if all((path / "candidate_values" / candidate).is_dir() for candidate in required)
    ]
    if not materialized:
        raise FileNotFoundError("exp263 cache lacks one or more required primitives")
    signatures = {
        tuple(
            sorted(
                item.relative_to(path).as_posix()
                for candidate in required
                for item in (path / "candidate_values" / candidate).glob(
                    "fold=*/part-*.parquet"
                )
            )
        )
        for path in materialized
    }
    if len(signatures) != 1:
        raise ValueError("multiple structurally different exp263 caches matched")
    root = materialized[0]
    return root, {
        "label": "exp263_cache_manifest",
        "path": str(root / filename),
        "raw_sha256": expected,
    }


def _partition_specifications(
    cache_root: Path,
    candidate: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((cache_root / "cache_manifest.json").read_text())
    specifications = manifest.get("candidate_value_partitions", {}).get(candidate, [])
    expected = {
        "/".join(Path(str(item["path"])).parts[-3:]): item for item in specifications
    }
    return manifest, expected


def load_exp263_partition(
    cache_root: Path,
    candidate: str,
    generation_fold: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths = sorted(
        (cache_root / "candidate_values" / candidate / f"fold={generation_fold}").glob(
            "part-*.parquet"
        )
    )
    if not paths:
        raise FileNotFoundError(
            f"exp263 cache has no {candidate} partition for fold {generation_fold}"
        )
    _, expected_by_suffix = _partition_specifications(cache_root, candidate)
    reports: list[dict[str, Any]] = []
    columns = ["id", "well", "well_row_idx", "outer_fold", "md_since", "candidate_tvt"]
    frames: list[pd.DataFrame] = []
    for path in paths:
        suffix = "/".join(path.parts[-3:])
        expected = expected_by_suffix.get(suffix)
        if expected is None:
            raise ValueError(f"unexpected exp263 partition: {suffix}")
        actual_sha = sha256_file(path)
        if actual_sha != str(expected["file_sha256"]):
            raise ValueError(f"exp263 partition SHA mismatch: {suffix}")
        frame = pd.read_parquet(path, columns=columns)
        frames.append(frame)
        reports.append(
            {
                "candidate": candidate,
                "generation_fold": generation_fold,
                "path": str(path),
                "raw_sha256": actual_sha,
                "manifest_content_sha256": str(expected["content_sha256"]),
                "rows": len(frame),
            }
        )
    frame = pd.concat(frames, ignore_index=True).rename(
        columns={
            "well": "well_id",
            "well_row_idx": "row_idx",
            "outer_fold": "exp263_generation_fold",
            "candidate_tvt": candidate,
        }
    )
    frame["id"] = frame["id"].astype(str)
    frame["well_id"] = frame["well_id"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int32)
    frame["exp263_generation_fold"] = pd.to_numeric(
        frame["exp263_generation_fold"], errors="raise"
    ).astype(np.int8)
    frame["md_since"] = pd.to_numeric(frame["md_since"], errors="raise").astype(
        np.float64
    )
    frame[candidate] = pd.to_numeric(frame[candidate], errors="raise").astype(
        np.float64
    )
    if not frame["exp263_generation_fold"].eq(generation_fold).all():
        raise ValueError(f"exp263 {candidate} partition fold label changed")
    if frame.duplicated(list(KEY_COLUMNS)).any() or frame["id"].duplicated().any():
        raise ValueError(f"exp263 {candidate} partition contains duplicate identity")
    return frame, reports


def _well_code_map(wells: Iterable[str]) -> dict[str, int]:
    return {well: index for index, well in enumerate(sorted(set(wells)))}


def global_row_key(
    wells: Iterable[str],
    row_idx: Sequence[int] | np.ndarray | pd.Series,
    well_codes: Mapping[str, int],
) -> np.ndarray:
    codes = np.fromiter(
        (well_codes.get(str(well), -1) for well in wells),
        dtype=np.int64,
    )
    if (codes < 0).any():
        raise KeyError("row identity contains an unknown well")
    rows = np.asarray(row_idx, dtype=np.int64)
    if (rows < 0).any() or (rows >= 2**32).any():
        raise ValueError("row_idx is outside the uint32 key contract")
    return (codes.astype(np.uint64) << np.uint64(32)) | rows.astype(np.uint64)


@dataclass
class BranchPredictions:
    frame: pd.DataFrame
    well_codes: dict[str, int]
    source_reports: list[dict[str, Any]]
    exp333_prediction_sha256: str
    exp355_selected_content_sha256: str
    used: np.ndarray


def load_branch_predictions(
    config: Mapping[str, Any],
    ledger: AccessLedger,
) -> BranchPredictions:
    exp333_spec = dict(get_nested(config, "data.exp333_oof"))
    exp355_spec = dict(get_nested(config, "data.exp355_oof"))
    exp333_path, exp333_report = resolve_sha_matched_file(
        filename=str(exp333_spec["filename"]),
        configured=exp333_spec.get("candidates", []),
        expected_raw_sha256=str(exp333_spec["expected_raw_sha256"]),
        expected_decompressed_sha256=str(exp333_spec["expected_decompressed_sha256"]),
        label="exp333 saved OOF",
    )
    exp355_path, exp355_report = resolve_sha_matched_file(
        filename=str(exp355_spec["filename"]),
        configured=exp355_spec.get("candidates", []),
        expected_raw_sha256=str(exp355_spec["expected_raw_sha256"]),
        expected_decompressed_sha256=str(exp355_spec["expected_decompressed_sha256"]),
        label="exp355 saved OOF",
    )

    exp333_columns = list(exp333_spec["allowed_pre_freeze_columns"])
    exp355_columns = list(exp355_spec["allowed_pre_freeze_columns"])
    assert_target_free_columns(exp333_columns, "exp333")
    assert_target_free_columns(exp355_columns, "exp355")
    exp333_header = set(pd.read_csv(exp333_path, nrows=0).columns)
    exp355_header = set(pd.read_csv(exp355_path, nrows=0).columns)
    if not set(exp333_columns).issubset(exp333_header):
        raise ValueError("exp333 OOF schema does not satisfy the frozen allowlist")
    if not set(exp355_columns).issubset(exp355_header):
        raise ValueError("exp355 OOF schema does not satisfy the frozen allowlist")
    forbidden333 = set(exp333_spec.get("forbidden_pre_freeze_columns", []))
    forbidden355 = set(exp355_spec.get("forbidden_pre_freeze_columns", []))
    if forbidden333 & set(exp333_columns) or forbidden355 & set(exp355_columns):
        raise ValueError("a saved OOF allowlist contains forbidden truth/error columns")

    exp333 = pd.read_csv(exp333_path, usecols=exp333_columns)
    ledger.pre_freeze_read("exp333_oof", exp333_columns, len(exp333))
    exp333_sha = portable_prediction_surface_sha256(
        exp333,
        ("well_id", "row_idx", "outer_fold", "tvt_pred_stage1"),
    )

    exp355 = pd.read_csv(exp355_path, usecols=exp355_columns)
    ledger.pre_freeze_read("exp355_oof", exp355_columns, len(exp355))
    selected355 = (
        "well_id",
        "row_idx",
        str(exp355_spec["reporting_fold_column"]),
        str(exp355_spec["prediction_column"]),
    )
    exp355_selected_sha = portable_prediction_surface_sha256(exp355, selected355)

    exp333 = exp333.rename(
        columns={
            str(exp333_spec["reporting_fold_column"]): "reporting_fold",
            str(exp333_spec["prediction_column"]): "exp333_stage1",
            str(exp333_spec["exp226_parity_column"]): "exp333_exp226_parity",
        }
    )
    exp355 = exp355.rename(
        columns={
            str(exp355_spec["reporting_fold_column"]): "reporting_fold",
            str(exp355_spec["prediction_column"]): "exp355_stage1",
        }
    )
    keep333 = [
        "well_id",
        "row_idx",
        "reporting_fold",
        "exp333_exp226_parity",
        "exp333_stage1",
    ]
    keep355 = ["well_id", "row_idx", "reporting_fold", "exp355_stage1"]
    exp333 = exp333[keep333].copy()
    exp355 = exp355[keep355].copy()
    for frame in (exp333, exp355):
        frame["well_id"] = frame["well_id"].astype(str)
        frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(
            np.int32
        )
        frame["reporting_fold"] = pd.to_numeric(
            frame["reporting_fold"], errors="raise"
        ).astype(np.int8)
        frame.sort_values(list(KEY_COLUMNS), kind="mergesort", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError("branch OOF contains duplicate global keys")
    if len(exp333) != len(exp355):
        raise ValueError("exp333 and exp355 row counts differ")
    for column in KEY_COLUMNS:
        if not np.array_equal(exp333[column].to_numpy(), exp355[column].to_numpy()):
            raise ValueError(f"exp333/exp355 key mismatch in {column}")
    if not np.array_equal(
        exp333["reporting_fold"].to_numpy(np.int8),
        exp355["reporting_fold"].to_numpy(np.int8),
    ):
        raise ValueError("exp333 and exp355 reporting folds differ")
    exp333["exp355_stage1"] = pd.to_numeric(
        exp355["exp355_stage1"], errors="raise"
    ).to_numpy(np.float64)
    del exp355
    for column in ("exp333_exp226_parity", "exp333_stage1"):
        exp333[column] = pd.to_numeric(exp333[column], errors="raise").astype(
            np.float64
        )
    if not np.isfinite(
        exp333[
            ["exp333_exp226_parity", "exp333_stage1", "exp355_stage1"]
        ].to_numpy(np.float64)
    ).all():
        raise ValueError("branch predictions contain non-finite values")
    well_codes = _well_code_map(exp333["well_id"])
    keys = global_row_key(exp333["well_id"], exp333["row_idx"], well_codes)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("branch global uint64 key is not unique")
    exp333.index = pd.Index(keys, name="global_row_key")
    return BranchPredictions(
        frame=exp333,
        well_codes=well_codes,
        source_reports=[
            {
                "label": "exp333_oof",
                **exp333_report,
                "upstream_producer_prediction_sha256": str(
                    exp333_spec["expected_prediction_content_sha256"]
                ),
                "upstream_producer_hash_role": (
                    "provenance_only_pandas_hash_pandas_dtype_version_dependent"
                ),
                "portable_selected_surface_sha256": exp333_sha,
                "portable_selected_surface_hash_algorithm": "keyed_float_hex_v1",
            },
            {
                "label": "exp355_oof",
                **exp355_report,
                "upstream_prediction_logical_sha256": str(
                    exp355_spec["expected_prediction_logical_sha256"]
                ),
                "upstream_producer_hash_role": (
                    "provenance_only_dtype_dependent_array_hash"
                ),
                "portable_selected_surface_sha256": exp355_selected_sha,
                "portable_selected_surface_hash_algorithm": "keyed_float_hex_v1",
            },
        ],
        exp333_prediction_sha256=exp333_sha,
        exp355_selected_content_sha256=exp355_selected_sha,
        used=np.zeros(len(exp333), dtype=bool),
    )


# %% [markdown]
# ## 5. Streaming source assembly and prediction freeze

# %%
def _align_exp263_component(
    base: pd.DataFrame,
    component: pd.DataFrame,
    candidate: str,
) -> pd.DataFrame:
    identity = ("id", "exp263_generation_fold", "md_since")
    merged = base.merge(
        component[[*KEY_COLUMNS, *identity, candidate]],
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
        suffixes=("", f"_{candidate}"),
    )
    if len(merged) != len(base):
        raise ValueError(f"exp263 {candidate} join lost rows")
    for column in identity:
        other = f"{column}_{candidate}"
        left = merged[column].to_numpy()
        right = merged.pop(other).to_numpy()
        equal = (
            np.array_equal(left, right, equal_nan=True)
            if column == "md_since"
            else np.array_equal(left, right)
        )
        if not equal:
            raise ValueError(f"exp263 {candidate} identity mismatch in {column}")
    return merged


def assemble_source_partition(
    component_frames: Mapping[str, pd.DataFrame],
    branch: BranchPredictions,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = tuple(EXPECTED_COMPONENT_WEIGHTS)
    if set(component_frames) != set(required):
        raise ValueError("exp263 component family differs from the frozen formula")
    base = component_frames[required[0]].copy()
    for candidate in required[1:]:
        base = _align_exp263_component(base, component_frames[candidate], candidate)
    keys = global_row_key(base["well_id"], base["row_idx"], branch.well_codes)
    positions = branch.frame.index.get_indexer(keys)
    if (positions < 0).any():
        raise ValueError("one or more exp263 rows are absent from branch OOF")
    if branch.used[positions].any():
        raise ValueError("one or more global rows appeared in multiple exp263 partitions")
    branch.used[positions] = True
    upgrades = branch.frame.iloc[positions].reset_index(drop=True)
    if not np.array_equal(
        base["well_id"].astype(str).to_numpy(),
        upgrades["well_id"].astype(str).to_numpy(),
    ) or not np.array_equal(
        base["row_idx"].to_numpy(np.int32),
        upgrades["row_idx"].to_numpy(np.int32),
    ):
        raise ValueError("branch global-key lookup changed row identity")
    exp226_parity = np.abs(
        base["exp226_k16"].to_numpy(np.float64)
        - upgrades["exp333_exp226_parity"].to_numpy(np.float64)
    )
    exp226_max_abs = float(exp226_parity.max(initial=0.0))
    exp226_tolerance = float(
        get_nested(config, "guards.technical.require_exp226_parity_atol_ft")
    )
    if exp226_max_abs > exp226_tolerance:
        raise ValueError(
            f"exp226 cache/exp333 parity failed: {exp226_max_abs} > {exp226_tolerance}"
        )

    exp226 = base["exp226_k16"].to_numpy(np.float64)
    likpf = base["likpf_mean"].to_numpy(np.float64)
    exact_hmm = base["exp209_exact_hmm"].to_numpy(np.float64)
    exp333 = upgrades["exp333_stage1"].to_numpy(np.float64)
    exp355 = upgrades["exp355_stage1"].to_numpy(np.float64)
    control = 0.50 * exp226 + 0.25 * likpf + 0.25 * exact_hmm
    control_float32 = (
        np.float32(0.50) * exp226.astype(np.float32)
        + np.float32(0.25) * likpf.astype(np.float32)
        + np.float32(0.25) * exact_hmm.astype(np.float32)
    ).astype(np.float32)
    formula_parity = float(
        np.max(np.abs(control - control_float32.astype(np.float64)), initial=0.0)
    )
    formula_spacing = np.abs(np.spacing(control_float32)).astype(np.float64)
    formula_ulp_distance = np.divide(
        np.abs(control - control_float32.astype(np.float64)),
        formula_spacing,
        out=np.zeros_like(control),
        where=formula_spacing > 0.0,
    )
    formula_parity_max_ulps = float(formula_ulp_distance.max(initial=0.0))
    formula_tolerance = float(
        get_nested(config, "guards.technical.require_exp263_formula_parity_atol_ft")
    )
    formula_ulp_tolerance = float(
        get_nested(
            config,
            "guards.technical.require_exp263_formula_parity_max_float32_ulps",
        )
    )
    if (
        formula_parity > formula_tolerance
        and formula_parity_max_ulps > formula_ulp_tolerance + 1e-12
    ):
        raise ValueError(
            "exp263 float64/float32 formula parity failed: "
            f"{formula_parity} ft / {formula_parity_max_ulps} ULP"
        )
    full = 0.50 * exp333 + 0.25 * likpf + 0.25 * exp355
    correction = full - control
    finite = np.column_stack((control, full, correction))
    if not np.isfinite(finite).all():
        raise ValueError("assembled source formulas contain non-finite values")

    source = pd.DataFrame(
        {
            "id": base["id"].astype(str).to_numpy(),
            "well_id": base["well_id"].astype(str).to_numpy(),
            "row_idx": base["row_idx"].to_numpy(np.int32),
            "reporting_fold": upgrades["reporting_fold"].to_numpy(np.int8),
            "exp263_generation_fold": base["exp263_generation_fold"].to_numpy(
                np.int8
            ),
            "md_since": base["md_since"].to_numpy(np.float64),
            "exp226_k16": exp226,
            "likpf_mean": likpf,
            "exp209_exact_hmm": exact_hmm,
            "exp333_stage1": exp333,
            "exp355_stage1": exp355,
            "exp263_control": control,
            "full_replacement": full,
            "correction": correction,
        },
        columns=list(SOURCE_COLUMNS),
    )
    assert_target_free_columns(source.columns, "assembled_source")
    return source, {
        "rows": len(source),
        "exp226_parity_max_abs_ft": exp226_max_abs,
        "exp263_formula_parity_max_abs_ft": formula_parity,
        "exp263_formula_parity_max_float32_ulps": formula_parity_max_ulps,
    }


@dataclass(frozen=True)
class FrozenSourceSurface:
    partition_paths: tuple[Path, ...]
    partition_manifest: pd.DataFrame
    fold_ledger: pd.DataFrame
    fold_cross_tab: pd.DataFrame
    source_schema_sha256: str
    source_content_sha256: str
    formula_sha256: str
    rows: int
    wells: int


def freeze_source_surface(
    cache_root: Path,
    branch: BranchPredictions,
    config: Mapping[str, Any],
    artifacts: Path,
    ledger: AccessLedger,
) -> FrozenSourceSurface:
    freeze_root = artifacts / f"{OUTPUT_PREFIX}_frozen_sources"
    if freeze_root.exists():
        shutil.rmtree(freeze_root)
    freeze_root.mkdir(parents=True)
    content_hasher = PartitionContentHasher(SOURCE_COLUMNS)
    partition_rows: list[dict[str, Any]] = []
    well_ledgers: list[pd.DataFrame] = []
    partition_paths: list[Path] = []
    schema_sha: str | None = None
    total_rows = 0

    for generation_fold in range(5):
        component_frames: dict[str, pd.DataFrame] = {}
        exp263_reports: list[dict[str, Any]] = []
        for source_name, candidate in (
            ("exp226_k16", "exp226_k16"),
            ("likpf_mean", "likpf_mean"),
            ("exp209_exact_hmm", "exact_hmm"),
        ):
            frame, reports = load_exp263_partition(
                cache_root, candidate, generation_fold
            )
            if candidate != source_name:
                frame = frame.rename(columns={candidate: source_name})
            component_frames[source_name] = frame
            exp263_reports.extend(reports)
            ledger.pre_freeze_read(
                f"exp263:{source_name}:fold={generation_fold}",
                frame.columns,
                len(frame),
            )
        source, parity = assemble_source_partition(component_frames, branch, config)
        del component_frames
        current_schema = dataframe_schema_sha256(source)
        if schema_sha is None:
            schema_sha = current_schema
        elif schema_sha != current_schema:
            raise ValueError("frozen source schema changed across generation folds")
        content_hasher.update(source)
        fold_path = freeze_root / f"fold={generation_fold}" / "part-000.parquet"
        fold_path.parent.mkdir(parents=True)
        source.to_parquet(fold_path, index=False)
        partition_paths.append(fold_path)
        total_rows += len(source)
        well_ledger = source[
            ["well_id", "reporting_fold", "exp263_generation_fold"]
        ].drop_duplicates()
        if well_ledger["well_id"].duplicated().any():
            raise ValueError("a fold label is not constant within one well")
        well_ledgers.append(well_ledger)
        partition_rows.append(
            {
                "generation_fold": generation_fold,
                "path": str(fold_path),
                "rows": len(source),
                "wells": int(source["well_id"].nunique()),
                "file_sha256": sha256_file(fold_path),
                "content_sha256": dataframe_content_sha256(source),
                "schema_sha256": current_schema,
                **parity,
                "exp263_partition_files": len(exp263_reports),
                "exp263_partition_file_shas": mapping_sha256(
                    {"reports": exp263_reports}
                ),
            }
        )
        del source

    if not branch.used.all():
        raise ValueError(
            f"branch OOF has {int((~branch.used).sum())} rows absent from exp263 cache"
        )
    fold_ledger = (
        pd.concat(well_ledgers, ignore_index=True)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if fold_ledger["well_id"].duplicated().any():
        raise ValueError("one well appears in multiple exp263 generation folds")
    expected_support = set(int(value) for value in get_nested(config, "validation.expected_folds"))
    for column in ("reporting_fold", "exp263_generation_fold"):
        observed = set(int(value) for value in fold_ledger[column].unique())
        if observed != expected_support:
            raise ValueError(f"{column} support differs from the frozen fold contract")
    mismatch = fold_ledger["reporting_fold"].ne(
        fold_ledger["exp263_generation_fold"]
    )
    mismatch_wells = int(mismatch.sum())
    expected_mismatch = int(
        get_nested(config, "validation.expected_independent_fold_label_mismatch_wells")
    )
    if mismatch_wells != expected_mismatch:
        raise ValueError(
            f"independent fold mismatch changed: {mismatch_wells} != {expected_mismatch}"
        )
    cross_tab = pd.crosstab(
        fold_ledger["reporting_fold"],
        fold_ledger["exp263_generation_fold"],
    ).reindex(index=range(5), columns=range(5), fill_value=0)
    rows_expected = int(get_nested(config, "validation.expected_rows"))
    wells_expected = int(get_nested(config, "validation.expected_wells"))
    if total_rows != rows_expected or len(fold_ledger) != wells_expected:
        raise ValueError(
            f"frozen source dimensions changed: rows={total_rows}, wells={len(fold_ledger)}"
        )
    formula_contract = {
        "control_formula": get_nested(config, "candidate.control_formula"),
        "full_replacement_formula": get_nested(
            config, "candidate.full_replacement_formula"
        ),
        "correction_formula": get_nested(config, "candidate.correction_formula"),
        "output_formula": get_nested(config, "candidate.output_formula"),
        "lambda_candidates": list(EXPECTED_LAMBDAS),
    }
    frozen = FrozenSourceSurface(
        partition_paths=tuple(partition_paths),
        partition_manifest=pd.DataFrame(partition_rows),
        fold_ledger=fold_ledger,
        fold_cross_tab=cross_tab,
        source_schema_sha256=str(schema_sha),
        source_content_sha256=content_hasher.hexdigest(),
        formula_sha256=mapping_sha256(formula_contract),
        rows=total_rows,
        wells=len(fold_ledger),
    )
    ledger.freeze(
        {
            "rows": frozen.rows,
            "wells": frozen.wells,
            "source_schema_sha256": frozen.source_schema_sha256,
            "source_content_sha256": frozen.source_content_sha256,
            "formula_sha256": frozen.formula_sha256,
        }
    )
    return frozen


# %% [markdown]
# ## 6. Late truth and hidden-like attachment

# %%
def resolve_raw_train_dir(config: Mapping[str, Any], wells: Iterable[str]) -> Path:
    expected_samples = tuple(sorted(set(str(well) for well in wells))[:3])
    if not expected_samples:
        raise ValueError("raw truth resolver received no well identities")
    filename = f"{expected_samples[0]}__horizontal_well.csv"
    candidates = [
        Path(str(get_nested(config, "data.train_dir"))),
        project_root() / str(get_nested(config, "data.train_dir")),
    ]
    if KAGGLE_INPUT_ROOT.is_dir():
        candidates.extend(path.parent for path in KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    valid: list[Path] = []
    for path in sorted(set(candidates), key=str):
        if not path.is_dir():
            continue
        complete = True
        for well in expected_samples:
            horizontal_path = path / f"{well}__horizontal_well.csv"
            if not horizontal_path.is_file():
                complete = False
                break
            header = set(pd.read_csv(horizontal_path, nrows=0).columns)
            if not {"TVT", "TVT_input"}.issubset(header):
                complete = False
                break
        if complete:
            valid.append(path)
    if not valid:
        raise FileNotFoundError(
            "raw train directory with TVT/TVT_input truth schema and exp403 "
            "well identities was not found"
        )
    return min(
        valid,
        key=lambda path: (
            "rogii-wellbore-geology-prediction" not in path.as_posix(),
            "/train" not in path.as_posix(),
            path.as_posix(),
        ),
    )


def load_hidden_like_late(
    config: Mapping[str, Any],
    ledger: AccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = dict(get_nested(config, "data.hidden_like_assignment"))
    path, report = resolve_sha_matched_file(
        filename=str(spec["filename"]),
        configured=spec.get("candidates", []),
        expected_raw_sha256=str(spec["expected_sha256"]),
        expected_decompressed_sha256=None,
        label="exp115 hidden-like assignment",
    )
    roles = dict(spec["valid_role_columns"])
    columns = ["well_id", *roles.values()]
    frame = pd.read_csv(path, usecols=columns, dtype={"well_id": str})
    ledger.hidden_late("exp115_hidden_like", len(frame))
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment contains duplicate wells")
    frame = frame.rename(
        columns={
            roles["verification_like_spatial"]: "hidden_like_spatial_role",
            roles[
                "verification_like_typewell_purged"
            ]: "hidden_like_typewell_purged_role",
        }
    )
    frame["hidden_like_spatial"] = frame["hidden_like_spatial_role"].astype(str).eq(
        "valid"
    )
    frame["hidden_like_typewell_purged"] = frame[
        "hidden_like_typewell_purged_role"
    ].astype(str).eq("valid")
    return frame[
        ["well_id", "hidden_like_spatial", "hidden_like_typewell_purged"]
    ], {"label": "hidden_like_assignment", **report, "rows": len(frame)}


def attach_truth_after_freeze(
    frozen: FrozenSourceSurface,
    config: Mapping[str, Any],
    ledger: AccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not ledger.prediction_frozen:
        raise RuntimeError("late truth attachment requires the source freeze")
    raw_dir = resolve_raw_train_dir(config, frozen.fold_ledger["well_id"])
    hidden, hidden_report = load_hidden_like_late(config, ledger)
    frames: list[pd.DataFrame] = []
    for partition_path in frozen.partition_paths:
        source = pd.read_parquet(
            partition_path,
            columns=[
                "id",
                "well_id",
                "row_idx",
                "reporting_fold",
                "exp263_generation_fold",
                "md_since",
                "exp263_control",
                "full_replacement",
                "correction",
            ],
        )
        truth_rows: list[pd.DataFrame] = []
        for well, group in source.groupby("well_id", sort=True, observed=True):
            horizontal = pd.read_csv(
                raw_dir / f"{well}__horizontal_well.csv",
                usecols=["TVT", "TVT_input"],
            )
            row_idx = group["row_idx"].to_numpy(np.int64)
            if row_idx.max(initial=-1) >= len(horizontal):
                raise ValueError(f"{well}: frozen row exceeds raw truth")
            tvt_input = pd.to_numeric(
                horizontal.iloc[row_idx]["TVT_input"], errors="coerce"
            )
            if tvt_input.notna().any():
                raise ValueError(f"{well}: frozen rows are not wholly in the unknown suffix")
            true_tvt = pd.to_numeric(
                horizontal.iloc[row_idx]["TVT"], errors="raise"
            ).to_numpy(np.float64)
            ledger.truth_late(f"suffix_truth:{well}", len(group))
            truth_rows.append(
                pd.DataFrame(
                    {
                        "well_id": str(well),
                        "row_idx": row_idx.astype(np.int32),
                        "true_tvt": true_tvt,
                    }
                )
            )
        truth = pd.concat(truth_rows, ignore_index=True)
        attached = (
            source.merge(
                truth,
                on=list(KEY_COLUMNS),
                how="inner",
                validate="one_to_one",
            )
            .merge(hidden, on="well_id", how="left", validate="many_to_one")
        )
        if len(attached) != len(source):
            raise ValueError("late truth join lost frozen source rows")
        for column in ("hidden_like_spatial", "hidden_like_typewell_purged"):
            attached[column] = attached[column].fillna(False).astype(bool)
        finite = attached[
            [
                "md_since",
                "exp263_control",
                "full_replacement",
                "correction",
                "true_tvt",
            ]
        ].to_numpy(np.float64)
        if not np.isfinite(finite).all():
            raise ValueError("late score surface contains non-finite values")
        frames.append(attached)
    evaluation = (
        pd.concat(frames, ignore_index=True)
        .sort_values(list(KEY_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
    )
    if len(evaluation) != frozen.rows:
        raise ValueError("late score surface row count differs from source freeze")
    return evaluation, {
        "raw_train_dir": str(raw_dir),
        "truth_rows": ledger.late_truth_rows,
        "truth_columns_read_before_freeze": ledger.truth_columns_read_before_freeze,
        "hidden_like": hidden_report,
        "source_content_sha256_before_truth": frozen.source_content_sha256,
    }


# %% [markdown]
# ## 7. Outer-train lambda calibration

# %%
def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    if len(actual_array) == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(predicted_array - actual_array))))


def _scope_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    distance = frame["md_since"].to_numpy(np.float64)
    return {
        "near_0_250": (distance >= 0.0) & (distance < 250.0),
        "mid_250_1000": (distance >= 250.0) & (distance < 1000.0),
        "1000_plus": distance >= 1000.0,
        "hidden_like_spatial": frame["hidden_like_spatial"].to_numpy(bool),
        "hidden_like_typewell_purged": frame[
            "hidden_like_typewell_purged"
        ].to_numpy(bool),
    }


def lambda_metrics(
    frame: pd.DataFrame,
    lambda_value: float,
) -> dict[str, Any]:
    truth = frame["true_tvt"].to_numpy(np.float64)
    control = frame["exp263_control"].to_numpy(np.float64)
    candidate = control + float(lambda_value) * frame["correction"].to_numpy(
        np.float64
    )
    control_rmse = rmse(truth, control)
    candidate_rmse = rmse(truth, candidate)
    row: dict[str, Any] = {
        "lambda": float(lambda_value),
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "control_rmse": control_rmse,
        "candidate_rmse": candidate_rmse,
        "pooled_gain_ft": control_rmse - candidate_rmse,
    }
    for scope, mask in _scope_masks(frame).items():
        row[f"{scope}_rows"] = int(mask.sum())
        if mask.any():
            row[f"{scope}_control_rmse"] = rmse(truth[mask], control[mask])
            row[f"{scope}_candidate_rmse"] = rmse(truth[mask], candidate[mask])
            row[f"{scope}_delta_rmse_ft"] = (
                row[f"{scope}_candidate_rmse"] - row[f"{scope}_control_rmse"]
            )
        else:
            row[f"{scope}_control_rmse"] = math.nan
            row[f"{scope}_candidate_rmse"] = math.nan
            row[f"{scope}_delta_rmse_ft"] = math.inf
    by_well_rows: list[dict[str, Any]] = []
    working = frame[["well_id", "true_tvt", "exp263_control"]].copy()
    working["candidate"] = candidate
    for well, group in working.groupby("well_id", sort=True, observed=True):
        by_well_rows.append(
            {
                "well_id": str(well),
                "delta": rmse(group["true_tvt"], group["candidate"])
                - rmse(group["true_tvt"], group["exp263_control"]),
            }
        )
    deltas = pd.DataFrame(by_well_rows)["delta"].to_numpy(np.float64)
    row["by_well_delta_p95_ft"] = float(np.quantile(deltas, 0.95))
    row["worst_well_delta_rmse_ft"] = float(np.max(deltas))
    return row


def lambda_is_eligible(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if float(metrics["lambda"]) <= 0.0:
        return False
    gate = dict(
        get_nested(config, "candidate.lambda.positive_eligibility_outer_train")
    )
    checks = (
        float(metrics["pooled_gain_ft"])
        >= float(gate["minimum_pooled_rmse_gain_ft"]),
        float(metrics["near_0_250_delta_rmse_ft"])
        <= float(gate["maximum_near_0_250_delta_rmse_ft"]),
        float(metrics["mid_250_1000_delta_rmse_ft"])
        <= float(gate["maximum_mid_250_1000_delta_rmse_ft"]),
        float(metrics["1000_plus_delta_rmse_ft"])
        <= float(gate["maximum_1000_plus_delta_rmse_ft"]),
        float(metrics["by_well_delta_p95_ft"])
        <= float(gate["maximum_by_well_delta_p95_ft"]),
        float(metrics["worst_well_delta_rmse_ft"])
        <= float(gate["maximum_worst_well_delta_rmse_ft"]),
    )
    return bool(all(checks))


def calibrate_lambdas(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    lambdas = [
        float(value) for value in get_nested(config, "candidate.lambda.candidates")
    ]
    for outer_fold in range(5):
        outer_train = frame.loc[frame["reporting_fold"].ne(outer_fold)].copy()
        eligible: list[float] = []
        for lambda_value in lambdas:
            row = lambda_metrics(outer_train, lambda_value)
            row["outer_valid_fold"] = outer_fold
            row["calibration_scope"] = "reporting_fold_not_equal_outer_valid"
            row["eligible"] = lambda_is_eligible(row, config)
            metric_rows.append(row)
            if row["eligible"]:
                eligible.append(lambda_value)
        selected = max(eligible) if eligible else 0.0
        selection_rows.append(
            {
                "outer_valid_fold": outer_fold,
                "lambda_fold": selected,
                "positive_eligible_count": len(eligible),
                "eligible_lambdas": json.dumps(eligible),
                "fallback_to_zero": not bool(eligible),
                "selection_rule": "largest_positive_eligible_lambda",
                "outer_valid_rows_used_for_selection": 0,
                "outer_train_rows": len(outer_train),
                "outer_train_wells": int(outer_train["well_id"].nunique()),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    selection = pd.DataFrame(selection_rows)
    return metrics, selection


def apply_crossfit_lambdas(
    frame: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    lambda_map = selection.set_index("outer_valid_fold")["lambda_fold"].to_dict()
    output = frame.copy()
    output["lambda_fold"] = output["reporting_fold"].map(lambda_map).astype(
        np.float64
    )
    if output["lambda_fold"].isna().any():
        raise ValueError("cross-fit lambda map did not cover every reporting fold")
    output[CROSSFIT_CANDIDATE] = (
        output["exp263_control"]
        + output["lambda_fold"] * output["correction"]
    )
    return output


# %% [markdown]
# ## 8. Metrics, persistent-offset diagnostics, and promotion gate

# %%
def build_scope_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    masks = {"pooled": np.ones(len(frame), dtype=bool), **_scope_masks(frame)}
    rows: list[dict[str, Any]] = []
    for scope, mask in masks.items():
        group = frame.loc[mask]
        control_rmse = rmse(group["true_tvt"], group[CONTROL_CANDIDATE])
        candidate_rmse = rmse(group["true_tvt"], group[CROSSFIT_CANDIDATE])
        rows.append(
            {
                "scope": scope,
                "rows": len(group),
                "wells": int(group["well_id"].nunique()),
                "control_rmse": control_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_rmse_ft": candidate_rmse - control_rmse,
                "gain_ft": control_rmse - candidate_rmse,
            }
        )
    return pd.DataFrame(rows)


def build_fold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, group in frame.groupby("reporting_fold", sort=True):
        control_rmse = rmse(group["true_tvt"], group[CONTROL_CANDIDATE])
        candidate_rmse = rmse(group["true_tvt"], group[CROSSFIT_CANDIDATE])
        rows.append(
            {
                "reporting_fold": int(fold),
                "lambda_fold": float(group["lambda_fold"].iloc[0]),
                "rows": len(group),
                "wells": int(group["well_id"].nunique()),
                "control_rmse": control_rmse,
                "candidate_rmse": candidate_rmse,
                "gain_ft": control_rmse - candidate_rmse,
                "improved": candidate_rmse < control_rmse,
            }
        )
    return pd.DataFrame(rows)


def build_by_well_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True, observed=True):
        control_rmse = rmse(group["true_tvt"], group[CONTROL_CANDIDATE])
        candidate_rmse = rmse(group["true_tvt"], group[CROSSFIT_CANDIDATE])
        rows.append(
            {
                "well_id": str(well),
                "reporting_fold": int(group["reporting_fold"].iloc[0]),
                "exp263_generation_fold": int(
                    group["exp263_generation_fold"].iloc[0]
                ),
                "lambda_fold": float(group["lambda_fold"].iloc[0]),
                "rows": len(group),
                "control_rmse": control_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_rmse_ft": candidate_rmse - control_rmse,
            }
        )
    return pd.DataFrame(rows)


def persistent_offset_episodes(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = dict(get_nested(config, "audit.persistent_offset"))
    threshold = float(spec["error_threshold_ft"])
    minimum_rows = int(spec["minimum_consecutive_rows"])
    return_threshold = float(spec["return_threshold_ft"])
    horizons = [int(value) for value in spec["recovery_horizons_rows"]]
    rows: list[dict[str, Any]] = []
    for candidate in (CROSSFIT_CANDIDATE, CONTROL_CANDIDATE):
        for well, group in frame.groupby("well_id", sort=True, observed=True):
            group = group.sort_values("row_idx", kind="mergesort")
            error = np.abs(
                group[candidate].to_numpy(np.float64)
                - group["true_tvt"].to_numpy(np.float64)
            )
            bad = error > threshold
            padded = np.r_[False, bad, False]
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            ends = np.flatnonzero(padded[:-1] & ~padded[1:])
            row_index = group["row_idx"].to_numpy(np.int64)
            for start, end in zip(starts, ends, strict=True):
                if end - start < minimum_rows:
                    continue
                confirmed = start + minimum_rows - 1
                recovery = np.flatnonzero(error[confirmed + 1 :] <= return_threshold)
                recovery_rows = int(recovery[0] + 1) if len(recovery) else None
                row: dict[str, Any] = {
                    "candidate": candidate,
                    "well_id": str(well),
                    "reporting_fold": int(group["reporting_fold"].iloc[0]),
                    "episode_start_row_idx": int(row_index[start]),
                    "confirmed_row_idx": int(row_index[confirmed]),
                    "consecutive_rows_above_threshold": int(end - start),
                    "peak_abs_error_ft": float(np.max(error[start:end])),
                    "recovery_rows_after_confirmation": recovery_rows,
                }
                for horizon in horizons:
                    row[f"recovered_within_{horizon}"] = bool(
                        recovery_rows is not None and recovery_rows <= horizon
                    )
                rows.append(row)
    columns = [
        "candidate",
        "well_id",
        "reporting_fold",
        "episode_start_row_idx",
        "confirmed_row_idx",
        "consecutive_rows_above_threshold",
        "peak_abs_error_ft",
        "recovery_rows_after_confirmation",
        *[f"recovered_within_{horizon}" for horizon in horizons],
    ]
    episodes = pd.DataFrame(rows, columns=columns)
    summaries: list[dict[str, Any]] = []
    for candidate in (CROSSFIT_CANDIDATE, CONTROL_CANDIDATE):
        group = (
            episodes.loc[episodes["candidate"].eq(candidate)]
            if not episodes.empty
            else episodes
        )
        row: dict[str, Any] = {"candidate": candidate, "episodes": len(group)}
        for horizon in horizons:
            column = f"recovered_within_{horizon}"
            row[f"{column}_count"] = int(group[column].sum()) if len(group) else 0
            row[f"{column}_rate"] = (
                float(group[column].mean()) if len(group) else math.nan
            )
        summaries.append(row)
    return episodes, pd.DataFrame(summaries)


def evaluate_promotion_gate(
    selection: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    recovery: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = dict(get_nested(config, "guards.promotion"))
    selected_lambdas = selection["lambda_fold"].to_numpy(np.float64)
    lambda_test = float(np.median(selected_lambdas))
    positive_folds = int((selected_lambdas > 0.0).sum())
    pooled = scope_metrics.set_index("scope").loc["pooled"]
    protected_scopes = (
        "near_0_250",
        "mid_250_1000",
        "1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    )
    scope_delta = scope_metrics.set_index("scope")["delta_rmse_ft"]
    deltas = by_well["delta_rmse_ft"].to_numpy(np.float64)
    p95_delta = float(np.quantile(deltas, 0.95))
    worst_index = int(np.argmax(deltas))
    worst = by_well.iloc[worst_index]
    recovery_wide = recovery.set_index("candidate")
    candidate_episodes = int(
        recovery_wide.loc[CROSSFIT_CANDIDATE, "episodes"]
    )
    control_episodes = int(recovery_wide.loc[CONTROL_CANDIDATE, "episodes"])
    episode_delta = candidate_episodes - control_episodes
    recovery_column = "recovered_within_512_rate"
    candidate_recovery = float(
        recovery_wide.loc[CROSSFIT_CANDIDATE, recovery_column]
    )
    control_recovery = float(
        recovery_wide.loc[CONTROL_CANDIDATE, recovery_column]
    )
    recovery_delta = (
        math.inf
        if candidate_episodes == 0
        else candidate_recovery - control_recovery
    )
    checks = {
        "positive_lambda_folds": positive_folds
        >= int(criteria["minimum_positive_lambda_folds"]),
        "lambda_test": lambda_test
        >= float(criteria["minimum_current_test_lambda"]),
        "pooled_gain": float(pooled["gain_ft"])
        >= float(criteria["minimum_pooled_rmse_gain_vs_exp263_ft"]),
        "improved_folds": int(fold_metrics["improved"].sum())
        >= int(criteria["minimum_improved_folds"]),
        "protected_scopes": all(
            float(scope_delta.loc[scope])
            <= float(criteria["maximum_scope_delta_rmse_vs_exp263_ft"])
            for scope in protected_scopes
        ),
        "by_well_p95": p95_delta
        <= float(criteria["maximum_by_well_delta_p95_ft"]),
        "worst_well": float(worst["delta_rmse_ft"])
        <= float(criteria["maximum_worst_well_delta_rmse_ft"]),
        "persistent_episode_count": episode_delta
        <= int(criteria["maximum_persistent_offset_episode_count_delta"]),
        "recovery_within_512": candidate_episodes == 0
        or recovery_delta
        >= float(criteria["minimum_recovery_within_512_rate_delta"]),
    }
    passed = bool(all(checks.values()))
    return {
        "passed": passed,
        "checks": checks,
        "observed": {
            "positive_lambda_folds": positive_folds,
            "lambda_fold": selected_lambdas.tolist(),
            "lambda_test": lambda_test,
            "pooled_control_rmse": float(pooled["control_rmse"]),
            "pooled_candidate_rmse": float(pooled["candidate_rmse"]),
            "pooled_gain_ft": float(pooled["gain_ft"]),
            "improved_folds": int(fold_metrics["improved"].sum()),
            "scope_delta_rmse_ft": {
                scope: float(scope_delta.loc[scope]) for scope in protected_scopes
            },
            "by_well_delta_p95_ft": p95_delta,
            "worst_well_id": str(worst["well_id"]),
            "worst_well_delta_rmse_ft": float(worst["delta_rmse_ft"]),
            "persistent_episode_count": {
                "candidate": candidate_episodes,
                "control": control_episodes,
                "delta": episode_delta,
            },
            "recovery_within_512_rate": {
                "candidate": candidate_recovery,
                "control": control_recovery,
                "delta": recovery_delta,
            },
        },
        "decision": (
            get_nested(config, "guards.decision.pass_action")
            if passed
            else get_nested(config, "guards.decision.fail_action")
        ),
    }


# %% [markdown]
# ## 9. Orchestration and generated artifacts

# %%
def run_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_contract(config, require_run_authorization=True)
    started = time.perf_counter()
    artifacts = output_artifacts_dir()
    artifacts.mkdir(parents=True, exist_ok=True)
    ledger = AccessLedger()

    cache_root, cache_report = resolve_exp263_cache_root(config)
    branch = load_branch_predictions(config, ledger)
    input_manifest_rows = [cache_report, *branch.source_reports]
    frozen = freeze_source_surface(
        cache_root,
        branch,
        config,
        artifacts,
        ledger,
    )
    branch_report = {
        "exp333_prediction_content_sha256": branch.exp333_prediction_sha256,
        "exp355_selected_content_sha256": branch.exp355_selected_content_sha256,
        "all_branch_rows_consumed": bool(branch.used.all()),
    }
    del branch

    evaluation, late_attachment = attach_truth_after_freeze(
        frozen,
        config,
        ledger,
    )
    lambda_metrics_frame, selection = calibrate_lambdas(evaluation, config)
    crossfit = apply_crossfit_lambdas(evaluation, selection)
    scope_metrics = build_scope_metrics(crossfit)
    fold_metrics = build_fold_metrics(crossfit)
    by_well = build_by_well_metrics(crossfit)
    episodes, recovery = persistent_offset_episodes(crossfit, config)
    promotion = evaluate_promotion_gate(
        selection,
        scope_metrics,
        fold_metrics,
        by_well,
        recovery,
        config,
    )
    runtime_seconds = time.perf_counter() - started
    truth_values = evaluation["true_tvt"].to_numpy(np.float64)
    rebuilt_control_rmse = rmse(
        truth_values,
        evaluation["exp263_control"].to_numpy(np.float64),
    )
    full_replacement_rmse = rmse(
        truth_values,
        evaluation["full_replacement"].to_numpy(np.float64),
    )
    reference_control_rmse = float(
        get_nested(
            config,
            "validation.full_replacement_reference.exploratory_rebuilt_control_rmse_ft",
        )
    )
    reference_full_rmse = float(
        get_nested(
            config,
            "validation.full_replacement_reference.exploratory_rmse_ft",
        )
    )
    reference_metric_tolerance = float(
        get_nested(
            config,
            "guards.technical.require_design_reference_metric_atol_ft",
        )
    )
    technical = {
        "all_input_sha_matches": True,
        "rows": len(crossfit),
        "wells": int(crossfit["well_id"].nunique()),
        "reporting_fold_support": sorted(
            int(value) for value in crossfit["reporting_fold"].unique()
        ),
        "generation_fold_support": sorted(
            int(value) for value in crossfit["exp263_generation_fold"].unique()
        ),
        "fold_mismatch_wells": int(
            frozen.fold_ledger["reporting_fold"]
            .ne(frozen.fold_ledger["exp263_generation_fold"])
            .sum()
        ),
        "truth_columns_read_before_freeze": ledger.truth_columns_read_before_freeze,
        "late_truth_rows": ledger.late_truth_rows,
        "finite_prediction_coverage": float(
            np.isfinite(
                crossfit[
                    [
                        CONTROL_CANDIDATE,
                        "full_replacement",
                        CROSSFIT_CANDIDATE,
                    ]
                ].to_numpy(np.float64)
            ).mean()
        ),
        "runtime_seconds": runtime_seconds,
        "peak_rss_gb": peak_rss_gb(),
        "execution_count_match": True,
        "rebuilt_control_rmse": rebuilt_control_rmse,
        "reference_rebuilt_control_rmse": reference_control_rmse,
        "rebuilt_control_rmse_abs_diff": abs(
            rebuilt_control_rmse - reference_control_rmse
        ),
        "full_replacement_rmse": full_replacement_rmse,
        "reference_full_replacement_rmse": reference_full_rmse,
        "full_replacement_rmse_abs_diff": abs(
            full_replacement_rmse - reference_full_rmse
        ),
    }
    technical_checks = {
        "expected_rows": technical["rows"]
        == int(get_nested(config, "guards.technical.require_expected_rows")),
        "expected_wells": technical["wells"]
        == int(get_nested(config, "guards.technical.require_expected_wells")),
        "fold_support": technical["reporting_fold_support"]
        == list(get_nested(config, "guards.technical.require_expected_folds"))
        and technical["generation_fold_support"]
        == list(get_nested(config, "guards.technical.require_expected_folds")),
        "fold_mismatch": technical["fold_mismatch_wells"]
        == int(
            get_nested(
                config,
                "validation.expected_independent_fold_label_mismatch_wells",
            )
        ),
        "truth_boundary": technical["truth_columns_read_before_freeze"]
        == int(
            get_nested(
                config,
                "guards.technical.maximum_truth_error_hidden_reads_before_freeze",
            )
        ),
        "finite": technical["finite_prediction_coverage"]
        == float(
            get_nested(
                config,
                "guards.technical.require_source_prediction_finite_coverage",
            )
        ),
        "runtime": technical["runtime_seconds"]
        <= float(get_nested(config, "guards.technical.maximum_runtime_seconds")),
        "peak_rss": technical["peak_rss_gb"]
        <= float(get_nested(config, "guards.technical.maximum_peak_rss_gb")),
        "execution_count": technical["execution_count_match"],
        "design_reference_metrics": (
            technical["rebuilt_control_rmse_abs_diff"]
            <= reference_metric_tolerance
            and technical["full_replacement_rmse_abs_diff"]
            <= reference_metric_tolerance
        ),
    }
    technical["checks"] = technical_checks
    technical["passed"] = bool(all(technical_checks.values()))
    if not technical["passed"]:
        promotion["passed"] = False
        promotion["decision"] = "technical_gate_failed_close_without_scientific_decision"

    prefix = OUTPUT_PREFIX
    reports: dict[str, dict[str, Any]] = {}
    reports["contract"] = write_json(
        artifacts / f"{prefix}_scientific_contract.json",
        {
            "experiment": EXPERIMENT_NAME,
            "route": "ensemble",
            "formula": {
                "control": get_nested(config, "candidate.control_formula"),
                "full": get_nested(config, "candidate.full_replacement_formula"),
                "output": get_nested(config, "candidate.output_formula"),
            },
            "lambda": get_nested(config, "candidate.lambda"),
            "execution_count": get_nested(config, "candidate.execution_count"),
            "forbidden": get_nested(config, "guards.forbidden"),
        },
    )
    reports["input_manifest"] = write_csv(
        artifacts / f"{prefix}_input_manifest.csv",
        pd.DataFrame(input_manifest_rows),
    )
    reports["partition_manifest"] = write_csv(
        artifacts / f"{prefix}_frozen_source_partition_manifest.csv",
        frozen.partition_manifest,
    )
    reports["fold_ledger"] = write_csv(
        artifacts / f"{prefix}_fold_ledger.csv",
        frozen.fold_ledger,
    )
    reports["fold_cross_tab"] = write_csv(
        artifacts / f"{prefix}_fold_cross_tab.csv",
        frozen.fold_cross_tab.rename_axis("reporting_fold").reset_index(),
    )
    reports["lambda_metrics"] = write_csv(
        artifacts / f"{prefix}_outer_train_lambda_metrics.csv",
        lambda_metrics_frame,
    )
    reports["lambda_selection"] = write_csv(
        artifacts / f"{prefix}_lambda_selection.csv",
        selection,
    )
    reports["scope_metrics"] = write_csv(
        artifacts / f"{prefix}_scope_metrics.csv",
        scope_metrics,
    )
    reports["fold_metrics"] = write_csv(
        artifacts / f"{prefix}_fold_metrics.csv",
        fold_metrics,
    )
    reports["by_well"] = write_csv(
        artifacts / f"{prefix}_by_well_metrics.csv",
        by_well,
    )
    reports["episodes"] = write_csv(
        artifacts / f"{prefix}_persistent_offset_episodes.csv",
        episodes,
    )
    reports["recovery"] = write_csv(
        artifacts / f"{prefix}_persistent_offset_recovery.csv",
        recovery,
    )
    prediction_output = crossfit[
        [
            "id",
            "well_id",
            "row_idx",
            "reporting_fold",
            "exp263_generation_fold",
            "md_since",
            "exp263_control",
            "full_replacement",
            "correction",
            "lambda_fold",
            CROSSFIT_CANDIDATE,
        ]
    ]
    reports["predictions"] = write_gzip_csv(
        artifacts / f"{prefix}_crossfit_oof_predictions.csv.gz",
        prediction_output,
    )
    reports["promotion_gate"] = write_json(
        artifacts / f"{prefix}_promotion_gate.json",
        promotion,
    )
    freeze_manifest = {
        "source_schema_sha256": frozen.source_schema_sha256,
        "source_content_sha256": frozen.source_content_sha256,
        "formula_sha256": frozen.formula_sha256,
        "rows": frozen.rows,
        "wells": frozen.wells,
        "truth_columns_read_before_freeze": ledger.truth_columns_read_before_freeze,
        "branch": branch_report,
        "late_attachment": late_attachment,
        "access_events_sha256": mapping_sha256({"events": ledger.events}),
    }
    reports["freeze_manifest"] = write_json(
        artifacts / f"{prefix}_freeze_manifest.json",
        freeze_manifest,
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed" if technical["passed"] else "technical_gate_failed",
        "route": "ensemble",
        "decision": promotion["decision"],
        "technical_gate_passed": technical["passed"],
        "promotion_gate_passed": promotion["passed"],
        "technical": technical,
        "promotion": promotion,
        "rows": len(crossfit),
        "wells": int(crossfit["well_id"].nunique()),
        "runtime_seconds": runtime_seconds,
        "source_freeze": freeze_manifest,
        "generated_artifacts": reports,
        "deterministic_anchor": False,
        "inference_approved": False,
        "submission_approved": False,
    }
    reports["summary"] = write_json(
        artifacts / f"{prefix}_summary.json",
        summary,
    )
    evidence_rows = []
    for name, report in reports.items():
        evidence_rows.append({"artifact": name, **report})
    write_csv(
        artifacts / f"{prefix}_sha_manifest.csv",
        pd.DataFrame(evidence_rows),
    )
    if KAGGLE_WORKING_ROOT.is_dir():
        write_json(KAGGLE_WORKING_ROOT / "metrics.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 10. Setup and guarded execution

# %%
if EXECUTE_NOTEBOOK:
    CONFIG = read_config()
    CONTRACT_PREVIEW = validate_contract(CONFIG, require_run_authorization=False)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "status": get_nested(CONFIG, "experiment.status"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "k16_upgrade": get_nested(CONFIG, "lineage.k16_upgrade"),
                "hmm_upgrade": get_nested(CONFIG, "lineage.hmm_upgrade"),
                "lambda_candidates": list(EXPECTED_LAMBDAS),
                "execution_count": get_nested(
                    CONFIG, "candidate.execution_count"
                ),
                "contract": CONTRACT_PREVIEW,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if bool(get_nested(CONFIG, "execution.run_train")):
        run_experiment(CONFIG)
    else:
        print(
            "Implementation-only mode: exp403 Kaggle train-side execution remains "
            "fail-closed pending separate approval.",
            flush=True,
        )

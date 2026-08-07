# %% [markdown]
# # exp498 geometry mean-reversion tail-regime physics readout
#
# This notebook implements one pre-registered, saved-full-OOF diagnostic over
# exp490.  It never regenerates a prediction, runs an HMM/PF/Beam, trains a
# model, blends candidates, or creates a selector.  Phase A freezes one
# target-free well table; only Phase B may read fold and saved outcomes.

# %% [markdown]
# ## Contents
# 1. Imports and immutable experiment contract
# 2. Notebook-safe paths, SHA helpers, and leakage ledger
# 3. SHA-pinned exp490 and raw-input resolution
# 4. Chunked target-free prediction and segment aggregation
# 5. Visible-prefix GR physics features
# 6. Fixed buckets, primary regime, and feature freeze
# 7. Truth-late fold, by-well, and episode readout
# 8. Fixed all-AND gate and secondary descriptive tables
# 9. Generated artifacts, metrics, and guarded execution

# %% [markdown]
# ## 1. Imports and immutable experiment contract

# %%
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import platform
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp498_geometry_mean_reversion_tail_regime_physics_readout"
PARENT_EXPERIMENT = "exp490_geometry_centered_mean_reverting_offset_hmm"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

PREDICTION_SAFE_COLUMNS = (
    "well",
    "row_idx",
    "suffix_offset",
    "tvt_geop",
    "geometry_mean_reverting_delta_mean",
    "geometry_mean_reverting_hmm_std",
    "dmd",
    "k16_segment_id",
    "k16_segment_span",
    "rho",
    "exp226_pred",
    "md_since",
)
PREDICTION_FORBIDDEN_BEFORE_FREEZE = frozenset(
    {
        "fold",
        "true_tvt_readout_only",
        "candidate_error",
        "parent_error",
        "exp226_error",
        "exp357_parent_prediction",
    }
)
HORIZONTAL_SAFE_COLUMNS = ("TVT_input", "MD", "Z", "GR")
HORIZONTAL_READ_COLUMNS = ("TVT_input", "GR")
TYPEWELL_READ_COLUMNS = ("TVT", "GR")
PRIMARY_REGIME_COLUMN = "weak_gr_geometry_conflict"

FEATURE_VALUE_COLUMNS = (
    "suffix_horizon_md",
    "k16_median_segment_span_ft",
    "prefix_gr_sigma",
    "prefix_gr_information_ratio",
    "geometry_disagreement_median_ft",
    "early_abs_offset_ft",
    "state_uncertainty_median_ft",
)
FEATURE_BUCKET_COLUMNS = (
    "suffix_horizon_bucket",
    "k16_median_segment_span_bucket",
    "prefix_gr_sigma_bucket",
    "prefix_gr_information_ratio_bucket",
    "geometry_disagreement_bucket",
    "early_abs_offset_bucket",
    "state_uncertainty_bucket",
)
FEATURE_FLAG_COLUMNS = (
    "weak_observation",
    "geometry_conflict",
    "material_early_offset",
    PRIMARY_REGIME_COLUMN,
)


def get_nested(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def validate_immutable_config(config: Mapping[str, Any]) -> None:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("unexpected experiment name")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp498 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp498 parent changed")
    if not bool(get_nested(config, "implementation.enabled")):
        raise ValueError("exp498 implementation must be enabled")
    if bool(get_nested(config, "implementation.inference_enabled")):
        raise ValueError("exp498 inference is out of scope")
    if bool(get_nested(config, "implementation.submission_enabled")):
        raise ValueError("exp498 submission is out of scope")
    if bool(get_nested(config, "model.prediction_generation")):
        raise ValueError("exp498 may not generate predictions")
    expected_safe = tuple(get_nested(config, "data.inputs.predictions.phase_a_safe_columns"))
    if expected_safe != PREDICTION_SAFE_COLUMNS:
        raise ValueError("Phase A prediction allowlist changed")
    forbidden = set(expected_safe).intersection(PREDICTION_FORBIDDEN_BEFORE_FREEZE)
    if forbidden:
        raise ValueError(f"Phase A prediction allowlist leaks {sorted(forbidden)}")
    if bool(get_nested(config, "data.horizontal_target_read_allowed")):
        raise ValueError("horizontal suffix TVT must remain forbidden")
    counts = get_nested(config, "execution_contract.if_separately_implemented_and_run")
    forbidden_counts = (
        "new_hmm_well_runs",
        "new_predictions",
        "model_configs",
        "trained_folds",
        "boosters",
        "pf_runs",
        "beam_runs",
        "gpu_runs",
    )
    if any(int(counts[name]) != 0 for name in forbidden_counts):
        raise ValueError("exp498 execution contract contains forbidden work")
    if int(counts["readouts"]) != 1:
        raise ValueError("exp498 must contain exactly one scientific readout")


# %% [markdown]
# ## 2. Notebook-safe paths, SHA helpers, and leakage ledger


# %%
def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "experiments").is_dir():
            return candidate
    return start


def config_path() -> Path:
    root = find_project_root()
    candidates = (
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        PACKAGE_DIR / "config.yaml"
        if PACKAGE_DIR.name == EXPERIMENT_NAME
        else Path("/nonexistent-exp498-config"),
        KAGGLE_WORKING_ROOT / "config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp498 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    selected = path or config_path()
    with selected.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("exp498 config must be a mapping")
    validate_immutable_config(value)
    return value


def artifacts_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = find_project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def metrics_path() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return find_project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
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


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stable_object_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_csv(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_frame_sha256(frame: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> dict[str, Any]:
    data = json.dumps(
        to_jsonable(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data + b"\n")
    return {"path": str(path), "sha256": sha256_file(path)}


def write_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return {
        "path": str(path),
        "rows": len(frame),
        "sha256": sha256_file(path),
        "logical_sha256": logical_frame_sha256(frame),
    }


def peak_rss_gib() -> float:
    rss_kib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return rss_kib / (1024.0**3)
    return rss_kib / (1024.0**2)


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
    }


@dataclass
class TruthLateLedger:
    expected_wells: int
    phase: str = "phase_a_target_free"
    safe_reads: dict[str, int] = field(default_factory=dict)
    outcome_reads_before_freeze: dict[str, int] = field(default_factory=dict)
    post_freeze_reads: dict[str, int] = field(default_factory=dict)
    feature_content_sha256: str = ""
    feature_contract_sha256: str = ""

    @property
    def frozen(self) -> bool:
        return self.phase == "phase_b_truth_late"

    def record_safe(self, label: str, rows: int) -> None:
        if self.frozen:
            raise RuntimeError(f"target-free read {label} occurred after feature freeze")
        self.safe_reads[label] = self.safe_reads.get(label, 0) + int(rows)

    def freeze_features(self, frame: pd.DataFrame, contract_sha256: str) -> None:
        if self.frozen:
            raise RuntimeError("features were already frozen")
        if len(frame) != self.expected_wells or frame["well"].nunique() != self.expected_wells:
            raise ValueError("feature freeze does not cover the expected wells")
        if not contract_sha256:
            raise ValueError("feature contract SHA is empty")
        self.feature_content_sha256 = logical_frame_sha256(frame)
        self.feature_contract_sha256 = str(contract_sha256)
        self.phase = "phase_b_truth_late"

    def record_outcome(self, label: str, rows: int) -> None:
        if not self.frozen:
            self.outcome_reads_before_freeze[label] = self.outcome_reads_before_freeze.get(
                label, 0
            ) + int(rows)
            raise RuntimeError(f"outcome {label} was read before feature freeze")
        self.post_freeze_reads[label] = self.post_freeze_reads.get(label, 0) + int(rows)


# %% [markdown]
# ## 3. SHA-pinned exp490 and raw-input resolution
#
# File hashing is permitted before freeze, but outcome CSVs are not parsed until
# Phase B.  The four exp490 shard decoder manifests are also SHA-pinned because
# they are the source of the per-well horizontal/typewell file hashes.


# %%
def _deduplicate_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted({path.resolve() for path in paths if path.is_file()})


def resolve_exact_file(filename: str, candidates: Sequence[str]) -> Path:
    root = find_project_root()
    matches: list[Path] = []
    for raw in candidates:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file() and candidate.name == filename:
            matches.append(candidate)
        elif (candidate / filename).is_file():
            matches.append(candidate / filename)
    if KAGGLE_INPUT_ROOT.is_dir():
        matches.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    unique = _deduplicate_paths(matches)
    if not unique:
        raise FileNotFoundError(f"could not resolve exact input {filename}")
    return unique[0]


def verify_file_sha(path: Path, expected: str, *, label: str) -> str:
    observed = sha256_file(path)
    if observed != str(expected):
        raise ValueError(f"{label} SHA changed: expected={expected}, observed={observed}")
    return observed


def resolve_and_verify_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    merge_candidates = [
        str(item) for item in get_nested(config, "data.exp490_merge_source.candidates")
    ]
    inputs = get_nested(config, "data.inputs")
    resolved: dict[str, Any] = {}
    for label in (
        "predictions",
        "by_well_metrics",
        "episode_metrics",
        "segment_contract",
        "well_manifest",
    ):
        spec = inputs[label]
        path = resolve_exact_file(str(spec["filename"]), merge_candidates)
        expected = spec.get("raw_gzip_sha256", spec.get("sha256"))
        observed = verify_file_sha(path, str(expected), label=label)
        entry: dict[str, Any] = {"path": str(path), "sha256": observed}
        if label == "predictions":
            decompressed = sha256_decompressed_csv(path)
            if decompressed != str(spec["decompressed_sha256"]):
                raise ValueError("exp490 prediction decompressed content SHA changed")
            entry["raw_gzip_sha256"] = observed
            entry["decompressed_sha256"] = decompressed
        resolved[label] = entry

    raw_contract = inputs["raw_sha_manifests"]
    expected_scientific = str(inputs["scientific_contract"]["sha256"])
    raw_sha_by_well: dict[str, dict[str, str]] = {}
    shard_entries: list[dict[str, Any]] = []
    for shard in raw_contract["shards"]:
        path = resolve_exact_file(
            str(shard["filename"]),
            [str(item) for item in shard["candidates"]],
        )
        observed = verify_file_sha(
            path, str(shard["sha256"]), label=f"raw_sha_manifest_{shard['shard_index']}"
        )
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if str(payload.get("scientific_contract_sha256")) != expected_scientific:
            raise ValueError(f"shard {shard['shard_index']} scientific contract changed")
        shard_map = payload.get("raw_input_sha256_by_well")
        if not isinstance(shard_map, dict):
            raise ValueError(f"shard {shard['shard_index']} raw SHA map is missing")
        overlap = set(raw_sha_by_well).intersection(shard_map)
        if overlap:
            raise ValueError(f"raw SHA manifests overlap on {sorted(overlap)[:3]}")
        for well, hashes in shard_map.items():
            if set(hashes) != {"horizontal_sha256", "typewell_sha256"}:
                raise ValueError(f"{well}: unexpected raw SHA contract")
            raw_sha_by_well[str(well)] = {
                "horizontal_sha256": str(hashes["horizontal_sha256"]),
                "typewell_sha256": str(hashes["typewell_sha256"]),
            }
        shard_entries.append(
            {
                "shard_index": int(shard["shard_index"]),
                "path": str(path),
                "sha256": observed,
                "wells": len(shard_map),
                "scientific_contract_sha256": expected_scientific,
            }
        )
    expected_wells = int(raw_contract["expected_combined_wells"])
    if len(raw_sha_by_well) != expected_wells:
        raise ValueError(
            f"raw SHA contract covers {len(raw_sha_by_well)} wells, expected {expected_wells}"
        )
    resolved["raw_sha_manifests"] = shard_entries
    resolved["raw_sha_by_well"] = raw_sha_by_well
    resolved["raw_sha_contract_sha256"] = stable_object_sha256(raw_sha_by_well)
    resolved["scientific_contract_sha256"] = expected_scientific
    return resolved


def train_data_dir(config: Mapping[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.is_dir():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        first = next(KAGGLE_INPUT_ROOT.glob("**/*__horizontal_well.csv"), None)
        if first is not None:
            return first.parent
    local = find_project_root() / str(get_nested(config, "data.train_dir"))
    if not local.is_dir():
        raise FileNotFoundError(f"raw train directory not found: {local}")
    return local


def read_identity_manifest(path: Path, expected_rows: int, ledger: TruthLateLedger) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["well", "rows"], dtype={"well": str})
    ledger.record_safe("well_manifest", len(frame))
    if len(frame) != expected_rows or frame["well"].nunique() != expected_rows:
        raise ValueError("exp490 well manifest identity changed")
    frame["rows"] = pd.to_numeric(frame["rows"], errors="raise").astype(np.int64)
    if (frame["rows"] <= 0).any():
        raise ValueError("well manifest contains non-positive row counts")
    return frame.sort_values("well", kind="mergesort").reset_index(drop=True)


# %% [markdown]
# ## 4. Chunked target-free prediction and segment aggregation


# %%
def _prediction_feature_row(well: str, frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_idx = frame["row_idx"].to_numpy(np.int64)
    suffix_offset = frame["suffix_offset"].to_numpy(np.int64)
    if len(np.unique(row_idx)) != len(frame) or np.any(np.diff(row_idx) <= 0):
        raise ValueError(f"{well}: prediction row_idx is not strictly increasing")
    if not np.array_equal(suffix_offset, np.arange(len(frame), dtype=np.int64)):
        raise ValueError(f"{well}: suffix_offset is not contiguous from zero")
    numeric_columns = (
        "tvt_geop",
        "geometry_mean_reverting_delta_mean",
        "geometry_mean_reverting_hmm_std",
        "exp226_pred",
    )
    values = frame.loc[:, numeric_columns].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{well}: target-free prediction features are non-finite")
    early = frame.loc[frame["suffix_offset"].between(0, 31), "geometry_mean_reverting_delta_mean"]
    if len(early) != min(32, len(frame)):
        raise ValueError(f"{well}: early-offset window identity changed")
    return {
        "well": str(well),
        "prediction_rows": int(len(frame)),
        "first_row_idx": int(row_idx[0]),
        "last_row_idx": int(row_idx[-1]),
        "geometry_disagreement_median_ft": float(
            np.median(
                np.abs(
                    frame["exp226_pred"].to_numpy(np.float64)
                    - frame["tvt_geop"].to_numpy(np.float64)
                )
            )
        ),
        "early_abs_offset_ft": float(abs(np.median(early.to_numpy(np.float64)))),
        "state_uncertainty_median_ft": float(
            np.median(frame["geometry_mean_reverting_hmm_std"].to_numpy(np.float64))
        ),
    }


def stream_prediction_features(
    path: Path,
    *,
    expected_rows: int,
    chunk_rows: int,
    ledger: TruthLateLedger,
) -> pd.DataFrame:
    safe_columns = list(PREDICTION_SAFE_COLUMNS)
    if set(safe_columns).intersection(PREDICTION_FORBIDDEN_BEFORE_FREEZE):
        raise ValueError("prediction Phase A allowlist contains outcomes")
    rows: list[dict[str, Any]] = []
    seen_wells: set[str] = set()
    current_well: str | None = None
    current_parts: list[pd.DataFrame] = []
    total_rows = 0
    last_global_key: tuple[str, int] | None = None

    def finalize_current() -> None:
        nonlocal current_well, current_parts
        if current_well is None:
            return
        if current_well in seen_wells:
            raise ValueError(f"prediction well is non-contiguous: {current_well}")
        combined = pd.concat(current_parts, ignore_index=True)
        rows.append(_prediction_feature_row(current_well, combined))
        seen_wells.add(current_well)
        current_well = None
        current_parts = []

    for chunk in pd.read_csv(
        path,
        usecols=safe_columns,
        dtype={"well": str},
        chunksize=int(chunk_rows),
    ):
        total_rows += len(chunk)
        for column in safe_columns:
            if column != "well":
                chunk[column] = pd.to_numeric(chunk[column], errors="raise")
        first_key = (str(chunk.iloc[0]["well"]), int(chunk.iloc[0]["row_idx"]))
        if last_global_key is not None and first_key <= last_global_key:
            raise ValueError("prediction file is not globally sorted by well,row_idx")
        ordered = chunk.sort_values(["well", "row_idx"], kind="mergesort")
        if (
            not ordered[["well", "row_idx"]]
            .reset_index(drop=True)
            .equals(chunk[["well", "row_idx"]].reset_index(drop=True))
        ):
            raise ValueError("prediction chunk is not sorted by well,row_idx")
        last_global_key = (str(chunk.iloc[-1]["well"]), int(chunk.iloc[-1]["row_idx"]))
        for well, group in chunk.groupby("well", sort=False, observed=True):
            well = str(well)
            if current_well is None:
                current_well = well
                current_parts = [group]
            elif well == current_well:
                current_parts.append(group)
            else:
                finalize_current()
                current_well = well
                current_parts = [group]
    finalize_current()
    ledger.record_safe("prediction_safe_columns", total_rows)
    if total_rows != int(expected_rows):
        raise ValueError(f"prediction rows changed: {total_rows}")
    return pd.DataFrame(rows).sort_values("well", kind="mergesort").reset_index(drop=True)


def load_segment_features(
    path: Path,
    *,
    expected_rows: int,
    ledger: TruthLateLedger,
) -> pd.DataFrame:
    usecols = ["well", "k16_segment_id", "dmd_span"]
    segments = pd.read_csv(path, usecols=usecols, dtype={"well": str})
    ledger.record_safe("k16_segment_contract", len(segments))
    if len(segments) != int(expected_rows):
        raise ValueError("K16 segment contract row count changed")
    segments["k16_segment_id"] = pd.to_numeric(segments["k16_segment_id"], errors="raise").astype(
        np.int64
    )
    segments["dmd_span"] = pd.to_numeric(segments["dmd_span"], errors="raise").astype(np.float64)
    if segments.duplicated(["well", "k16_segment_id"]).any():
        raise ValueError("K16 segment keys are not unique")
    if not np.isfinite(segments["dmd_span"]).all() or (segments["dmd_span"] <= 0.0).any():
        raise ValueError("K16 segment spans must be finite and positive")
    counts = segments.groupby("well", sort=True, observed=True).size()
    if not (counts == 16).all():
        raise ValueError("every well must have exactly 16 K16 segments")
    grouped = segments.groupby("well", sort=True, observed=True)["dmd_span"]
    features = grouped.agg(
        suffix_horizon_md="sum",
        k16_median_segment_span_ft="median",
    ).reset_index()
    return features.sort_values("well", kind="mergesort").reset_index(drop=True)


# %% [markdown]
# ## 5. Visible-prefix GR physics features
#
# Horizontal `TVT` is excluded at `read_csv(usecols=...)`.  `prefix_gr_sigma`
# exactly reuses exp490's residual standard deviation and [10, 60] clipping.


# %%
def prefix_gr_features(
    horizontal: pd.DataFrame, typewell: pd.DataFrame
) -> tuple[float, float, int]:
    if "TVT" in horizontal.columns:
        raise ValueError("horizontal suffix truth entered Phase A")
    if set(horizontal.columns) != set(HORIZONTAL_READ_COLUMNS):
        raise ValueError("horizontal safe schema changed")
    typewell = typewell.sort_values("TVT", kind="mergesort").reset_index(drop=True)
    typewell_tvt = pd.to_numeric(typewell["TVT"], errors="raise").to_numpy(np.float64)
    typewell_gr = (
        pd.to_numeric(typewell["GR"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    )
    if (
        len(typewell_tvt) < 2
        or not np.isfinite(typewell_tvt).all()
        or not np.isfinite(typewell_gr).all()
    ):
        raise ValueError("typewell interpolation support is invalid")
    if np.any(np.diff(typewell_tvt) < 0.0):
        raise ValueError("typewell TVT must be monotone after stable sort")
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    if len(known) < 4:
        raise ValueError("visible prefix requires at least four rows")
    known_tvt = pd.to_numeric(known["TVT_input"], errors="raise").to_numpy(np.float64)
    known_gr = pd.to_numeric(known["GR"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    if not np.isfinite(known_tvt).all() or not np.isfinite(known_gr).all():
        raise ValueError("visible-prefix TVT/GR is non-finite")
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - typewell_at_known
    sigma = float(np.clip(np.std(residual), 10.0, 60.0))
    p05, p95 = np.percentile(typewell_at_known, [5.0, 95.0])
    information_ratio = float((p95 - p05) / sigma)
    if not np.isfinite(sigma) or not np.isfinite(information_ratio) or information_ratio < 0.0:
        raise ValueError("visible-prefix GR physics feature is invalid")
    return sigma, information_ratio, int(len(known))


def aggregate_raw_prefix_features(
    wells: Sequence[str],
    *,
    raw_dir: Path,
    expected_sha_by_well: Mapping[str, Mapping[str, str]],
    ledger: TruthLateLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_files = 0
    raw_rows = 0
    for well in sorted(str(item) for item in wells):
        if well not in expected_sha_by_well:
            raise ValueError(f"{well}: raw SHA contract is missing")
        horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well}__typewell.csv"
        hashes = expected_sha_by_well[well]
        verify_file_sha(horizontal_path, hashes["horizontal_sha256"], label=f"{well}_horizontal")
        verify_file_sha(typewell_path, hashes["typewell_sha256"], label=f"{well}_typewell")
        horizontal = pd.read_csv(horizontal_path, usecols=list(HORIZONTAL_READ_COLUMNS))
        typewell = pd.read_csv(typewell_path, usecols=list(TYPEWELL_READ_COLUMNS))
        sigma, information_ratio, prefix_rows = prefix_gr_features(horizontal, typewell)
        rows.append(
            {
                "well": well,
                "prefix_gr_sigma": sigma,
                "prefix_gr_information_ratio": information_ratio,
                "visible_prefix_rows": prefix_rows,
            }
        )
        raw_files += 2
        raw_rows += len(horizontal) + len(typewell)
    ledger.record_safe("raw_visible_prefix_and_typewell", raw_rows)
    frame = pd.DataFrame(rows).sort_values("well", kind="mergesort").reset_index(drop=True)
    manifest = {
        "raw_dir": str(raw_dir),
        "well_count": len(frame),
        "file_count": raw_files,
        "parsed_rows": raw_rows,
        "horizontal_read_columns": list(HORIZONTAL_READ_COLUMNS),
        "horizontal_forbidden_columns": ["TVT"],
        "typewell_read_columns": list(TYPEWELL_READ_COLUMNS),
        "expected_raw_sha_contract_sha256": stable_object_sha256(expected_sha_by_well),
    }
    return frame, manifest


# %% [markdown]
# ## 6. Fixed buckets, primary regime, and feature freeze


# %%
def _cut_left_closed(values: pd.Series, bins: Sequence[float], labels: Sequence[str]) -> pd.Series:
    effective_bins = [float(item) for item in bins]
    if math.isfinite(effective_bins[-1]):
        effective_bins[-1] = float(np.nextafter(effective_bins[-1], math.inf))
    result = pd.cut(
        values, bins=effective_bins, labels=labels, right=False, include_lowest=True, ordered=True
    )
    if result.isna().any():
        raise ValueError("left-closed fixed bucket left values unassigned")
    return result.astype("string")


def _cut_right_closed(values: pd.Series, bins: Sequence[float], labels: Sequence[str]) -> pd.Series:
    result = pd.cut(values, bins=bins, labels=labels, right=True, include_lowest=True, ordered=True)
    if result.isna().any():
        raise ValueError("right-closed fixed bucket left values unassigned")
    return result.astype("string")


def apply_fixed_buckets_and_regime(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    readout = get_nested(config, "readout")
    features = readout["features"]
    primary = readout["primary_regime"]
    result = frame.copy()
    result["suffix_horizon_bucket"] = _cut_right_closed(
        result["suffix_horizon_md"],
        features["suffix_horizon_md"]["bins"],
        features["suffix_horizon_md"]["labels"],
    )
    result["k16_median_segment_span_bucket"] = _cut_right_closed(
        result["k16_median_segment_span_ft"],
        features["k16_median_segment_span_ft"]["bins"],
        features["k16_median_segment_span_ft"]["labels"],
    )
    result["prefix_gr_sigma_bucket"] = _cut_left_closed(
        result["prefix_gr_sigma"],
        features["prefix_gr_sigma"]["bins"],
        features["prefix_gr_sigma"]["labels"],
    )
    result["prefix_gr_information_ratio_bucket"] = _cut_left_closed(
        result["prefix_gr_information_ratio"],
        features["prefix_gr_information_ratio"]["bins"],
        features["prefix_gr_information_ratio"]["labels"],
    )
    result["geometry_disagreement_bucket"] = _cut_left_closed(
        result["geometry_disagreement_median_ft"],
        features["geometry_disagreement_median_ft"]["bins"],
        features["geometry_disagreement_median_ft"]["labels"],
    )
    result["early_abs_offset_bucket"] = _cut_left_closed(
        result["early_abs_offset_ft"],
        features["early_abs_offset_ft"]["bins"],
        features["early_abs_offset_ft"]["labels"],
    )
    result["state_uncertainty_bucket"] = _cut_left_closed(
        result["state_uncertainty_median_ft"],
        features["state_uncertainty_median_ft"]["bins"],
        features["state_uncertainty_median_ft"]["labels"],
    )
    result["weak_observation"] = (
        result["prefix_gr_sigma"] >= float(primary["prefix_gr_sigma_minimum"])
    ) | (
        result["prefix_gr_information_ratio"]
        < float(primary["prefix_gr_information_ratio_maximum_exclusive"])
    )
    result["geometry_conflict"] = result["geometry_disagreement_median_ft"] >= float(
        primary["geometry_disagreement_minimum_ft"]
    )
    result["material_early_offset"] = result["early_abs_offset_ft"] >= float(
        primary["early_abs_offset_minimum_ft"]
    )
    result[PRIMARY_REGIME_COLUMN] = (
        result["weak_observation"] & result["geometry_conflict"] & result["material_early_offset"]
    )
    result = result.sort_values("well", kind="mergesort").reset_index(drop=True)
    return result


def assemble_target_free_well_features(
    prediction: pd.DataFrame,
    segment: pd.DataFrame,
    prefix: pd.DataFrame,
    manifest: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    frame = manifest.merge(prediction, on="well", how="left", validate="one_to_one")
    frame = frame.merge(segment, on="well", how="left", validate="one_to_one")
    frame = frame.merge(prefix, on="well", how="left", validate="one_to_one")
    if not np.array_equal(
        frame["rows"].to_numpy(np.int64), frame["prediction_rows"].to_numpy(np.int64)
    ):
        raise ValueError("prediction rows do not match exp490 well manifest")
    numeric = frame.loc[:, list(FEATURE_VALUE_COLUMNS)].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("target-free well feature coverage is not finite 1.0")
    frame = apply_fixed_buckets_and_regime(frame, config)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(frame) != expected_wells or frame["well"].nunique() != expected_wells:
        raise ValueError("target-free feature identity coverage changed")
    if frame.loc[:, list(FEATURE_BUCKET_COLUMNS)].isna().any().any():
        raise ValueError("one or more fixed buckets are unassigned")
    for column in FEATURE_FLAG_COLUMNS:
        frame[column] = frame[column].astype(bool)
    columns = (
        ["well", "rows", "prediction_rows", "first_row_idx", "last_row_idx", "visible_prefix_rows"]
        + list(FEATURE_VALUE_COLUMNS)
        + list(FEATURE_BUCKET_COLUMNS)
        + list(FEATURE_FLAG_COLUMNS)
    )
    return frame.loc[:, columns].sort_values("well", kind="mergesort").reset_index(drop=True)


def build_feature_contract(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "experiment": EXPERIMENT_NAME,
        "phase": "phase_a_target_free_frozen_before_fold_or_outcome",
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "schema": [
            {"column": column, "dtype": str(frame[column].dtype)} for column in frame.columns
        ],
        "feature_content_sha256": logical_frame_sha256(frame),
        "prediction_safe_columns": list(PREDICTION_SAFE_COLUMNS),
        "prediction_forbidden_before_freeze": sorted(PREDICTION_FORBIDDEN_BEFORE_FREEZE),
        "horizontal_read_columns": list(HORIZONTAL_READ_COLUMNS),
        "horizontal_suffix_truth_read": False,
        "typewell_read_columns": list(TYPEWELL_READ_COLUMNS),
        "fixed_features_and_buckets": get_nested(config, "readout.features"),
        "primary_regime": get_nested(config, "readout.primary_regime"),
        "source_contract": {
            "prediction_decompressed_sha256": source_manifest["predictions"]["decompressed_sha256"],
            "segment_sha256": source_manifest["segment_contract"]["sha256"],
            "well_manifest_sha256": source_manifest["well_manifest"]["sha256"],
            "raw_sha_contract_sha256": source_manifest["raw_sha_contract_sha256"],
            "scientific_contract_sha256": source_manifest["scientific_contract_sha256"],
        },
        "new_prediction_count": 0,
        "new_hmm_count": 0,
        "model_count": 0,
        "pf_count": 0,
        "beam_count": 0,
    }
    return {**core, "feature_contract_sha256": stable_object_sha256(core)}


# %% [markdown]
# ## 7. Truth-late fold, by-well, and episode readout


# %%
def load_fold_by_well(
    prediction_path: Path,
    *,
    expected_rows: int,
    chunk_rows: int,
    ledger: TruthLateLedger,
) -> pd.DataFrame:
    ledger.record_outcome("prediction_fold", expected_rows)
    fold_sets: dict[str, set[int]] = {}
    row_counts: dict[str, int] = {}
    total = 0
    for chunk in pd.read_csv(
        prediction_path,
        usecols=["well", "row_idx", "fold"],
        dtype={"well": str},
        chunksize=int(chunk_rows),
    ):
        total += len(chunk)
        chunk["fold"] = pd.to_numeric(chunk["fold"], errors="raise").astype(np.int64)
        for well, group in chunk.groupby("well", sort=False, observed=True):
            key = str(well)
            fold_sets.setdefault(key, set()).update(int(item) for item in group["fold"].unique())
            row_counts[key] = row_counts.get(key, 0) + len(group)
    if total != int(expected_rows):
        raise ValueError("truth-late fold read row count changed")
    rows: list[dict[str, Any]] = []
    for well in sorted(fold_sets):
        folds = fold_sets[well]
        if len(folds) != 1:
            raise ValueError(f"{well}: prediction rows span multiple folds")
        rows.append({"well": well, "fold": next(iter(folds)), "fold_rows": row_counts[well]})
    return pd.DataFrame(rows)


def load_saved_outcomes(
    by_well_path: Path,
    episode_path: Path,
    *,
    expected_wells: int,
    expected_episodes: int,
    ledger: TruthLateLedger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger.record_outcome("saved_by_well_metrics", expected_wells)
    by_well = pd.read_csv(by_well_path, dtype={"well": str})
    required_by_well = {
        "well",
        "fold",
        "rows",
        "candidate_rmse_ft",
        "exp357_parent_rmse_ft",
        "candidate_minus_parent_rmse_ft",
    }
    if set(by_well.columns) != required_by_well:
        raise ValueError("saved by-well outcome schema changed")
    if len(by_well) != expected_wells or by_well["well"].nunique() != expected_wells:
        raise ValueError("saved by-well outcome identity changed")
    by_well_numeric = [
        "fold",
        "rows",
        "candidate_rmse_ft",
        "exp357_parent_rmse_ft",
        "candidate_minus_parent_rmse_ft",
    ]
    for column in by_well_numeric:
        by_well[column] = pd.to_numeric(by_well[column], errors="raise")
    if not np.isfinite(by_well.loc[:, by_well_numeric].to_numpy(np.float64)).all():
        raise ValueError("saved by-well outcomes contain non-finite values")
    by_well["fold"] = by_well["fold"].astype(np.int64)
    by_well["rows"] = by_well["rows"].astype(np.int64)
    if (by_well["rows"] <= 0).any():
        raise ValueError("saved by-well outcomes contain non-positive row counts")
    ledger.record_outcome("saved_episode_metrics", expected_episodes)
    episodes = pd.read_csv(episode_path, dtype={"well": str, "episode_id": str})
    required_episode = {
        "episode_id",
        "well",
        "start_row_idx",
        "end_row_idx_exclusive",
        "rows",
        "parent_sse",
        "candidate_sse",
        "parent_recovered_within_256",
        "candidate_recovered_within_256",
        "parent_recovered_within_512",
        "candidate_recovered_within_512",
    }
    if set(episodes.columns) != required_episode or len(episodes) != expected_episodes:
        raise ValueError("saved episode outcome schema or row count changed")
    if episodes["episode_id"].nunique() != len(episodes):
        raise ValueError("saved episode IDs are not unique")
    episode_numeric = [
        "start_row_idx",
        "end_row_idx_exclusive",
        "rows",
        "parent_sse",
        "candidate_sse",
    ]
    for column in episode_numeric:
        episodes[column] = pd.to_numeric(episodes[column], errors="raise")
    if not np.isfinite(episodes.loc[:, episode_numeric].to_numpy(np.float64)).all():
        raise ValueError("saved episode outcomes contain non-finite values")
    if (episodes["rows"] <= 0).any() or (
        episodes[["parent_sse", "candidate_sse"]] < 0.0
    ).any().any():
        raise ValueError("saved episode rows/SSE violate the non-negative contract")
    for column in (
        "parent_recovered_within_256",
        "candidate_recovered_within_256",
        "parent_recovered_within_512",
        "candidate_recovered_within_512",
    ):
        if episodes[column].dtype != bool:
            normalized = (
                episodes[column].astype(str).str.lower().map({"true": True, "false": False})
            )
            if normalized.isna().any():
                raise ValueError(f"saved episode boolean column changed: {column}")
            episodes[column] = normalized.astype(bool)
    return by_well, episodes


def attach_truth_late_outcomes(
    features: pd.DataFrame,
    folds: pd.DataFrame,
    by_well: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    frame = features.merge(folds, on="well", how="left", validate="one_to_one")
    frame = frame.merge(
        by_well, on="well", how="left", validate="one_to_one", suffixes=("_manifest", "_outcome")
    )
    if not np.array_equal(
        frame["fold_rows"].to_numpy(np.int64), frame["prediction_rows"].to_numpy(np.int64)
    ):
        raise ValueError("truth-late fold rows do not match frozen prediction rows")
    if not np.array_equal(
        frame["rows_outcome"].to_numpy(np.int64), frame["prediction_rows"].to_numpy(np.int64)
    ):
        raise ValueError("saved by-well rows do not match frozen prediction rows")
    if not np.array_equal(
        frame["fold_manifest"].to_numpy(np.int64),
        frame["fold_outcome"].to_numpy(np.int64),
    ):
        raise ValueError("prediction fold and saved by-well fold disagree")
    frame = frame.rename(columns={"fold_manifest": "fold"}).drop(columns=["fold_outcome"])
    expected_folds = set(int(item) for item in get_nested(config, "validation.expected_folds"))
    if set(frame["fold"].astype(int)) != expected_folds:
        raise ValueError("truth-late fold coverage changed")
    outcomes = get_nested(config, "readout.outcomes")
    frame["harmful_well"] = frame["candidate_minus_parent_rmse_ft"] > float(
        outcomes["harmful_well_delta_rmse_threshold_ft"]
    )
    frame["catastrophic_tail_well"] = frame["candidate_minus_parent_rmse_ft"] > float(
        outcomes["catastrophic_tail_delta_rmse_threshold_ft"]
    )
    return frame.sort_values("well", kind="mergesort").reset_index(drop=True)


# %% [markdown]
# ## 8. Fixed all-AND gate and secondary descriptive tables


# %%
def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if float(denominator) <= 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


def _group_readout(
    frame: pd.DataFrame, *, scope: str, fold: int | None, regime: bool
) -> dict[str, Any]:
    selected = frame.loc[frame[PRIMARY_REGIME_COLUMN].eq(regime)]
    return {
        "scope": scope,
        "fold": fold,
        "group": "regime" if regime else "complement",
        "wells": int(len(selected)),
        "harmful_wells": int(selected["harmful_well"].sum()),
        "harmful_rate": _safe_rate(int(selected["harmful_well"].sum()), len(selected)),
        "catastrophic_tail_wells": int(selected["catastrophic_tail_well"].sum()),
        "catastrophic_tail_rate": _safe_rate(
            int(selected["catastrophic_tail_well"].sum()), len(selected)
        ),
        "mean_delta_rmse_ft": float(selected["candidate_minus_parent_rmse_ft"].mean())
        if len(selected)
        else float("nan"),
        "median_delta_rmse_ft": float(selected["candidate_minus_parent_rmse_ft"].median())
        if len(selected)
        else float("nan"),
    }


def episode_readout(episodes: pd.DataFrame, well_frame: pd.DataFrame) -> list[dict[str, Any]]:
    flags = well_frame.loc[:, ["well", "fold", PRIMARY_REGIME_COLUMN]]
    joined = episodes.merge(flags, on="well", how="left", validate="many_to_one")
    if joined[PRIMARY_REGIME_COLUMN].isna().any():
        raise ValueError("episode well is absent from frozen feature identity")
    output: list[dict[str, Any]] = []
    for regime in (False, True):
        selected = joined.loc[joined[PRIMARY_REGIME_COLUMN].eq(regime)]
        parent_sse = float(selected["parent_sse"].sum())
        candidate_sse = float(selected["candidate_sse"].sum())
        output.append(
            {
                "group": "regime" if regime else "complement",
                "episodes": int(len(selected)),
                "wells": int(selected["well"].nunique()),
                "parent_sse": parent_sse,
                "candidate_sse": candidate_sse,
                "candidate_minus_parent_sse": candidate_sse - parent_sse,
                "candidate_to_parent_sse_ratio": _safe_rate(candidate_sse, parent_sse),
                "sse_reduction_fraction": _safe_rate(parent_sse - candidate_sse, parent_sse),
                "parent_recovery_256": float(
                    selected["parent_recovered_within_256"].astype(bool).mean()
                )
                if len(selected)
                else float("nan"),
                "candidate_recovery_256": float(
                    selected["candidate_recovered_within_256"].astype(bool).mean()
                )
                if len(selected)
                else float("nan"),
                "parent_recovery_512": float(
                    selected["parent_recovered_within_512"].astype(bool).mean()
                )
                if len(selected)
                else float("nan"),
                "candidate_recovery_512": float(
                    selected["candidate_recovered_within_512"].astype(bool).mean()
                )
                if len(selected)
                else float("nan"),
            }
        )
    return output


def evaluate_primary_regime(
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    thresholds = get_nested(config, "readout.physics_regime_requires_all")
    pooled_rows = [
        _group_readout(frame, scope="pooled", fold=None, regime=value) for value in (False, True)
    ]
    fold_rows: list[dict[str, Any]] = []
    expected_folds = [int(item) for item in get_nested(config, "validation.expected_folds")]
    for fold in expected_folds:
        selected = frame.loc[frame["fold"].eq(fold)]
        fold_rows.extend(
            _group_readout(selected, scope="fold", fold=fold, regime=value)
            for value in (False, True)
        )
    by_fold = pd.DataFrame(fold_rows)
    pooled = {row["group"]: row for row in pooled_rows}
    regime = pooled["regime"]
    complement = pooled["complement"]
    if complement["harmful_rate"] == 0.0:
        harm_rate_ratio = float("inf") if regime["harmful_rate"] > 0.0 else float("nan")
    else:
        harm_rate_ratio = float(regime["harmful_rate"] / complement["harmful_rate"])
    mean_delta_difference = float(regime["mean_delta_rmse_ft"] - complement["mean_delta_rmse_ft"])
    catastrophic_total = int(frame["catastrophic_tail_well"].sum())
    catastrophic_capture = _safe_rate(regime["catastrophic_tail_wells"], catastrophic_total)
    regime_coverage = _safe_rate(regime["wells"], len(frame))
    supported_folds = 0
    folds_harm_rate_higher = 0
    folds_mean_delta_positive = 0
    fold_comparisons: list[dict[str, Any]] = []
    for fold in expected_folds:
        selected = by_fold.loc[by_fold["fold"].eq(fold)].set_index("group")
        fold_regime = selected.loc["regime"]
        fold_complement = selected.loc["complement"]
        supported = int(fold_regime["wells"]) >= int(
            thresholds["minimum_regime_wells_per_supported_fold"]
        )
        harm_higher = bool(
            np.isfinite(fold_regime["harmful_rate"])
            and np.isfinite(fold_complement["harmful_rate"])
            and float(fold_regime["harmful_rate"]) > float(fold_complement["harmful_rate"])
        )
        delta_difference = float(
            fold_regime["mean_delta_rmse_ft"] - fold_complement["mean_delta_rmse_ft"]
        )
        mean_positive = bool(np.isfinite(delta_difference) and delta_difference > 0.0)
        supported_folds += int(supported)
        folds_harm_rate_higher += int(harm_higher)
        folds_mean_delta_positive += int(mean_positive)
        fold_comparisons.append(
            {
                "fold": fold,
                "regime_wells": int(fold_regime["wells"]),
                "supported": supported,
                "harm_rate_difference": float(
                    fold_regime["harmful_rate"] - fold_complement["harmful_rate"]
                ),
                "harm_rate_higher": harm_higher,
                "mean_delta_rmse_difference_ft": delta_difference,
                "mean_delta_rmse_difference_positive": mean_positive,
            }
        )
    checks = {
        "coverage_and_fold_support": bool(
            regime["wells"] >= int(thresholds["minimum_total_regime_wells"])
            and supported_folds >= int(thresholds["minimum_supported_folds"])
        ),
        "pooled_harmful_rate_ratio": bool(
            np.isfinite(harm_rate_ratio)
            and harm_rate_ratio >= float(thresholds["pooled_harmful_rate_ratio_minimum"])
        )
        or bool(math.isinf(harm_rate_ratio) and harm_rate_ratio > 0.0),
        "fold_harmful_rate_direction": bool(
            folds_harm_rate_higher >= int(thresholds["minimum_folds_regime_harm_rate_higher"])
        ),
        "pooled_mean_delta_difference": bool(
            np.isfinite(mean_delta_difference)
            and mean_delta_difference
            >= float(thresholds["pooled_mean_delta_rmse_regime_minus_complement_minimum_ft"])
        ),
        "fold_mean_delta_direction": bool(
            folds_mean_delta_positive
            >= int(thresholds["minimum_folds_mean_delta_rmse_difference_positive"])
        ),
        "catastrophic_capture_with_bounded_coverage": bool(
            np.isfinite(catastrophic_capture)
            and catastrophic_capture
            >= float(thresholds["catastrophic_tail_capture_minimum_fraction"])
            and regime_coverage <= float(thresholds["regime_coverage_maximum_fraction"])
        ),
    }
    passed = bool(all(checks.values()))
    summary = {
        "primary_regime": PRIMARY_REGIME_COLUMN,
        "pooled": pooled,
        "harmful_rate_ratio_regime_over_complement": harm_rate_ratio,
        "mean_delta_rmse_regime_minus_complement_ft": mean_delta_difference,
        "catastrophic_tail_wells_total": catastrophic_total,
        "catastrophic_tail_capture_fraction": catastrophic_capture,
        "regime_coverage_fraction": regime_coverage,
        "supported_folds": supported_folds,
        "folds_regime_harm_rate_higher": folds_harm_rate_higher,
        "folds_mean_delta_difference_positive": folds_mean_delta_positive,
        "fold_comparisons": fold_comparisons,
        "checks": checks,
        "passed": passed,
        "decision": (
            get_nested(config, "readout.pass_action")
            if passed
            else get_nested(config, "readout.fail_action")
        ),
        "parent_terminal_close_preserved": True,
        "episode_readout": episode_readout(episodes, frame),
    }
    return by_fold, summary


def build_secondary_bucket_summary(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    bucket_to_feature = {
        "suffix_horizon_bucket": "suffix_horizon_md",
        "k16_median_segment_span_bucket": "k16_median_segment_span_ft",
        "prefix_gr_sigma_bucket": "prefix_gr_sigma",
        "prefix_gr_information_ratio_bucket": "prefix_gr_information_ratio",
        "geometry_disagreement_bucket": "geometry_disagreement_median_ft",
        "early_abs_offset_bucket": "early_abs_offset_ft",
        "state_uncertainty_bucket": "state_uncertainty_median_ft",
    }
    feature_contract = get_nested(config, "readout.features")
    rows: list[dict[str, Any]] = []
    for column in FEATURE_BUCKET_COLUMNS:
        for bucket in feature_contract[bucket_to_feature[column]]["labels"]:
            selected = frame.loc[frame[column].eq(str(bucket))]
            wells_by_fold = {
                str(int(fold)): int(count)
                for fold, count in selected.groupby("fold", sort=True, observed=True).size().items()
            }
            rows.append(
                {
                    "feature": column,
                    "bucket": str(bucket),
                    "wells": int(len(selected)),
                    "folds_present": int(selected["fold"].nunique()),
                    "minimum_wells_per_present_fold": int(min(wells_by_fold.values()))
                    if wells_by_fold
                    else 0,
                    "wells_by_fold_json": json.dumps(
                        wells_by_fold, sort_keys=True, separators=(",", ":")
                    ),
                    "mean_delta_rmse_ft": float(selected["candidate_minus_parent_rmse_ft"].mean()),
                    "median_delta_rmse_ft": float(
                        selected["candidate_minus_parent_rmse_ft"].median()
                    ),
                    "harmful_rate": float(selected["harmful_well"].mean()),
                    "catastrophic_tail_rate": float(selected["catastrophic_tail_well"].mean()),
                    "primary_regime_rate": float(selected[PRIMARY_REGIME_COLUMN].mean()),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["feature", "bucket"], kind="mergesort")
        .reset_index(drop=True)
    )


# %% [markdown]
# ## 9. Generated artifacts, metrics, and guarded execution


# %%
def run_readout(config: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    chunk_rows = int(get_nested(config, "runtime.chunk_rows"))
    ledger = TruthLateLedger(expected_wells=expected_wells)
    source_manifest = resolve_and_verify_inputs(config)
    raw_sha_by_well = source_manifest.pop("raw_sha_by_well")
    paths = {
        key: Path(value["path"])
        for key, value in source_manifest.items()
        if isinstance(value, Mapping) and "path" in value
    }
    manifest = read_identity_manifest(
        paths["well_manifest"],
        int(get_nested(config, "data.inputs.well_manifest.expected_rows")),
        ledger,
    )
    if set(manifest["well"]) != set(raw_sha_by_well):
        raise ValueError("well manifest and raw SHA contracts disagree")
    prediction_features = stream_prediction_features(
        paths["predictions"], expected_rows=expected_rows, chunk_rows=chunk_rows, ledger=ledger
    )
    segment_features = load_segment_features(
        paths["segment_contract"],
        expected_rows=int(get_nested(config, "data.inputs.segment_contract.expected_rows")),
        ledger=ledger,
    )
    prefix_features, raw_manifest = aggregate_raw_prefix_features(
        manifest["well"].tolist(),
        raw_dir=train_data_dir(config),
        expected_sha_by_well=raw_sha_by_well,
        ledger=ledger,
    )
    features = assemble_target_free_well_features(
        prediction_features, segment_features, prefix_features, manifest, config
    )
    output = artifacts_dir()
    prefix = EXPERIMENT_NAME
    feature_artifact = write_csv(output / f"{prefix}_target_free_well_features.csv", features)
    feature_contract = build_feature_contract(features, config, source_manifest)
    feature_contract_artifact = write_json(
        output / f"{prefix}_feature_contract.json", feature_contract
    )
    if feature_artifact["logical_sha256"] != feature_contract["feature_content_sha256"]:
        raise RuntimeError("written target-free feature content differs from frozen contract")
    ledger.freeze_features(features, feature_contract["feature_contract_sha256"])
    if ledger.feature_content_sha256 != feature_contract["feature_content_sha256"]:
        raise RuntimeError("leakage ledger feature content differs from frozen contract")
    if ledger.outcome_reads_before_freeze:
        raise RuntimeError("truth-late contract was violated before freeze")
    input_manifest = {
        "experiment": EXPERIMENT_NAME,
        "phase_order": get_nested(config, "readout.phase_order"),
        "source_inputs": source_manifest,
        "raw_input": raw_manifest,
        "feature_artifact": feature_artifact,
        "feature_contract_artifact": feature_contract_artifact,
        "leakage_ledger_at_freeze": asdict(ledger),
        "scientific_execution_counts": get_nested(
            config, "execution_contract.if_separately_implemented_and_run"
        ),
    }
    input_manifest_artifact = write_json(output / f"{prefix}_input_manifest.json", input_manifest)

    folds = load_fold_by_well(
        paths["predictions"], expected_rows=expected_rows, chunk_rows=chunk_rows, ledger=ledger
    )
    by_well, episodes = load_saved_outcomes(
        paths["by_well_metrics"],
        paths["episode_metrics"],
        expected_wells=int(get_nested(config, "data.inputs.by_well_metrics.expected_rows")),
        expected_episodes=int(get_nested(config, "data.inputs.episode_metrics.expected_rows")),
        ledger=ledger,
    )
    truth_late = attach_truth_late_outcomes(features, folds, by_well, config)
    by_fold, primary = evaluate_primary_regime(truth_late, episodes, config)
    buckets = build_secondary_bucket_summary(truth_late, config)
    by_fold_artifact = write_csv(output / f"{prefix}_by_fold.csv", by_fold)
    bucket_artifact = write_csv(output / f"{prefix}_bucket_summary.csv", buckets)
    technical_checks = {
        "all_fixed_input_sha_match": True,
        "prediction_rows_match": len(truth_late) == expected_wells
        and int(truth_late["prediction_rows"].sum()) == expected_rows,
        "well_identity_one_to_one": len(truth_late) == expected_wells
        and truth_late["well"].nunique() == expected_wells,
        "finite_feature_coverage": bool(
            np.isfinite(features.loc[:, list(FEATURE_VALUE_COLUMNS)].to_numpy(np.float64)).all()
        ),
        "bucket_assignment_complete": bool(
            not features.loc[:, list(FEATURE_BUCKET_COLUMNS)].isna().any().any()
        ),
        "outcome_reads_before_freeze_zero": not bool(ledger.outcome_reads_before_freeze),
        "horizontal_suffix_truth_reads_zero": "TVT" not in HORIZONTAL_READ_COLUMNS,
        "new_prediction_hmm_model_pf_beam_gpu_zero": True,
    }
    technical_passed = bool(all(technical_checks.values()))
    if not technical_passed:
        raise RuntimeError(f"technical all-AND failed: {technical_checks}")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_saved_full_oof_physics_readout",
        "route": "pf_beam",
        "rows": expected_rows,
        "wells": expected_wells,
        "scientific_readouts": 1,
        "technical": {"checks": technical_checks, "passed": technical_passed},
        "physics_regime": primary,
        "leakage_ledger_final": asdict(ledger),
        "artifacts": {
            "input_manifest": input_manifest_artifact,
            "feature_contract": feature_contract_artifact,
            "target_free_well_features": feature_artifact,
            "by_fold": by_fold_artifact,
            "bucket_summary": bucket_artifact,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_gib": peak_rss_gib(),
            "cpu_only": True,
            "internet_enabled": False,
            "versions": runtime_versions(),
        },
        "inference": False,
        "submission": False,
        "parent_terminal_close_preserved": True,
    }
    summary_artifact = write_json(output / f"{prefix}_summary.json", summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "route": "pf_beam",
        "stage": "saved_full_oof_truth_late_physics_regime_diagnostic",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": get_nested(config, "validation.metric"),
        "implementation": {
            "enabled": True,
            "train_notebook": "compact_selfcontained_implemented_and_executed",
            "inference_enabled": False,
            "submission_enabled": False,
        },
        "result": primary,
        "technical": summary["technical"],
        "artifacts": {**summary["artifacts"], "summary": summary_artifact},
        "runtime": summary["runtime"],
        "notes": "Saved-full-OOF diagnostic only. exp490 remains terminal fail-closed.",
    }
    write_json(metrics_path(), metrics)
    return summary


# %%
CONFIG = load_config()
EXECUTION_PREVIEW = {
    "experiment": get_nested(CONFIG, "experiment.name"),
    "route": get_nested(CONFIG, "experiment.route"),
    "parent": get_nested(CONFIG, "lineage.parent"),
    "status": get_nested(CONFIG, "experiment.status"),
    "run_readout": bool(get_nested(CONFIG, "execution.run_readout")),
    "planned_execution": get_nested(CONFIG, "execution_contract.if_separately_implemented_and_run"),
    "primary_regime": get_nested(CONFIG, "readout.primary_regime"),
    "planned_artifacts": get_nested(CONFIG, "planned_artifacts"),
}
print(json.dumps(to_jsonable(EXECUTION_PREVIEW), indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    if bool(get_nested(CONFIG, "execution.run_readout")):
        if not bool(get_nested(CONFIG, "implementation.kaggle_run_approved")):
            raise RuntimeError("run_readout=true requires separate Kaggle run approval")
        READOUT_SUMMARY = run_readout(CONFIG)
        print(json.dumps(to_jsonable(READOUT_SUMMARY), indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "Implementation is ready; scientific readout remains disabled until "
            "separate Kaggle run approval."
        )

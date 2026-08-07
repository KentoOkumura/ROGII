# %% [markdown]
# # exp504 H512 regret-weighted block rank selector train
#
# This notebook keeps the exp293 fixed12 candidate bank and non-overlapping
# H512 blocks unchanged.  It rebuilds only the corrected exp264 target-free
# 88-column candidate-long surface, aggregates it by candidate and block, and
# fits one regret-weighted pairwise CPU LightGBM for each outer fold.  A held
# block receives exactly one candidate through antisymmetrized Borda scoring
# and the frozen anchor guard.  No PF/HMM/Beam candidate is regenerated.

# %% [markdown]
# ## Contents
# 1. Imports and immutable scientific contract
# 2. Notebook-safe paths, hashes, and serialization
# 3. Exp263 fixed12 cache and corrected exp264 row features
# 4. H512 candidate-block aggregation and target-free freeze
# 5. Pair labels, regret weights, and pair feature matrices
# 6. Pairwise LightGBM and antisymmetrized Borda selection
# 7. Truth-late fold evaluation and promotion gates
# 8. Metrics, feature importance, and generated artifacts
# 9. Setup and execution orchestration

# %% [markdown]
# ## 1. Imports and immutable scientific contract

# %%
from __future__ import annotations

import gc
import glob
import gzip
import hashlib
import json
import math
import platform
import resource
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError:  # Contract tests do not install LightGBM locally.
    LGBMClassifier = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Plotting is required on Kaggle, optional for tests.
    plt = None


EXPERIMENT_NAME = "exp504_h512_regret_weighted_block_rank_selector"
PARENT_EXPERIMENT = "exp293_physics_only_candidate_bank_headroom_contract"
FEATURE_PARENT = "exp264_exp263_candidate_confidence_dual_selector"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
OUTPUT_PREFIX = EXPERIMENT_NAME

KEY_COLUMNS = ["id", "well", "well_row_idx", "outer_fold", "md_since"]
BLOCK_COLUMNS = [
    "id",
    "well",
    "well_row_idx",
    "outer_fold",
    "md_since",
    "well_code",
    "h128_group",
    "h256_group",
    "h512_group",
    "whole_well_group",
]
COMMON_CONFIDENCE_SLOTS = [
    "sigma_tvt",
    "loglik_per_row",
    "entropy",
    "score_margin",
    "support_count",
    "ess_fraction",
    "fallback_rate",
]
AGGREGATION_NAMES = (
    "finite_fraction",
    "mean",
    "population_std_ddof0",
    "q10_numpy_linear",
    "q50_numpy_linear",
    "q90_numpy_linear",
    "first_finite",
    "last_finite",
    "last_minus_first_finite",
)
BLOCK_CONTEXT_NAMES = (
    "row_count",
    "is_partial_block",
    "block_start_md_since",
    "block_end_md_since",
    "block_start_evaluation_progress",
    "block_end_evaluation_progress",
)
EXPECTED_CANDIDATE_ORDER = (
    "exp226_k16",
    "selfgr_hmm_a070",
    "likpf_mean",
    "exact_hmm",
    "pf_ancc",
    "beam_mean",
    "exp226_k16__selfgr_hmm_a070",
    "exp226_k16__exact_hmm",
    "exp226_k16__likpf_mean",
    "selfgr_hmm_a070__likpf_mean",
    "likpf_mean__exact_hmm",
    "exp226_w500_50_50",
)
PRIMITIVE_CANDIDATES = EXPECTED_CANDIDATE_ORDER[:6]
ANCHOR_CANDIDATE = "exp226_w500_50_50"
ANCHOR_INDEX = EXPECTED_CANDIDATE_ORDER.index(ANCHOR_CANDIDATE)
PAIR_LEFT, PAIR_RIGHT = np.triu_indices(len(EXPECTED_CANDIDATE_ORDER), k=1)

CANDIDATE_SPECS: dict[str, dict[str, Any]] = {
    "exp226_k16": {"kind": "primitive", "family": "geometry"},
    "selfgr_hmm_a070": {"kind": "primitive", "family": "self_gr_hmm"},
    "likpf_mean": {"kind": "primitive", "family": "likelihood_pf"},
    "exact_hmm": {"kind": "primitive", "family": "exact_hmm"},
    "pf_ancc": {"kind": "primitive", "family": "particle_filter"},
    "beam_mean": {"kind": "primitive", "family": "beam"},
    "exp226_k16__selfgr_hmm_a070": {
        "kind": "pair_mean_50",
        "family": "geometry_self_gr_hmm_pair",
        "parents": ["exp226_k16", "selfgr_hmm_a070"],
        "weights": [0.5, 0.5],
    },
    "exp226_k16__exact_hmm": {
        "kind": "pair_mean_50",
        "family": "geometry_exact_hmm_pair",
        "parents": ["exp226_k16", "exact_hmm"],
        "weights": [0.5, 0.5],
    },
    "exp226_k16__likpf_mean": {
        "kind": "pair_mean_50",
        "family": "geometry_likelihood_pf_pair",
        "parents": ["exp226_k16", "likpf_mean"],
        "weights": [0.5, 0.5],
    },
    "selfgr_hmm_a070__likpf_mean": {
        "kind": "pair_mean_50",
        "family": "self_gr_hmm_likelihood_pf_pair",
        "parents": ["selfgr_hmm_a070", "likpf_mean"],
        "weights": [0.5, 0.5],
    },
    "likpf_mean__exact_hmm": {
        "kind": "pair_mean_50",
        "family": "likelihood_pf_exact_hmm_pair",
        "parents": ["likpf_mean", "exact_hmm"],
        "weights": [0.5, 0.5],
    },
    "exp226_w500_50_50": {
        "kind": "named_fixed",
        "family": "geometry_likelihood_pf_exact_hmm_fixed",
        "parents": ["exp226_k16", "likpf_mean", "exact_hmm"],
        "weights": [0.5, 0.25, 0.25],
    },
}


def get_nested(mapping: Mapping[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def validate_immutable_config(config: Mapping[str, Any]) -> None:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("unexpected experiment name")
    if get_nested(config, "experiment.route") != "ensemble":
        raise ValueError("exp504 route must remain ensemble")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp504 parent changed")
    if get_nested(config, "lineage.feature_contract_parent") != FEATURE_PARENT:
        raise ValueError("exp504 feature parent changed")
    order = tuple(get_nested(config, "data.candidate_bank.candidate_order"))
    if order != EXPECTED_CANDIDATE_ORDER:
        raise ValueError("fixed12 candidate order changed")
    if int(get_nested(config, "data.block_assignment.horizon_rows")) != 512:
        raise ValueError("exp504 is H512-only")
    if int(get_nested(config, "data.row_feature_schema.feature_count")) != 88:
        raise ValueError("corrected exp264 schema must contain 88 features")
    execution = get_nested(config, "execution_contract")
    expected_counts = {
        "scientific_variants": 1,
        "rank_configs": 1,
        "outer_folds": 5,
        "total_cpu_models": 5,
        "total_boosters": 5,
        "parent_control_retrains": 0,
        "candidate_regeneration_runs": 0,
        "pf_runs": 0,
        "hmm_runs": 0,
        "beam_runs": 0,
        "gpu_models": 0,
        "inference_runs": 0,
        "submission_files": 0,
    }
    for key, expected in expected_counts.items():
        if int(execution[key]) != expected:
            raise ValueError(f"execution contract changed: {key}")
    model = get_nested(config, "ranking.model")
    fixed_model = {
        "learning_rate": 0.03,
        "n_estimators": 800,
        "num_leaves": 31,
        "min_child_samples": 100,
        "max_depth": -1,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "max_bin": 255,
        "random_state": 42,
        "n_jobs": 4,
    }
    for key, expected in fixed_model.items():
        if model[key] != expected:
            raise ValueError(f"rank model setting changed: {key}")
    if bool(model["early_stopping"]) or bool(model["pair_subsampling"]):
        raise ValueError("early stopping and pair subsampling are forbidden")
    if bool(get_nested(config, "implementation.inference_enabled")):
        raise ValueError("inference remains outside the approved scope")
    if bool(get_nested(config, "implementation.submission_enabled")):
        raise ValueError("submission remains outside the approved scope")


# %% [markdown]
# ## 2. Notebook-safe paths, hashes, and serialization


# %%
def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file() and (candidate / "experiments").is_dir():
            return candidate
    return start


def config_path() -> Path:
    root = find_project_root()
    candidates = (
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        PACKAGE_DIR / "config.yaml"
        if PACKAGE_DIR.name == EXPERIMENT_NAME
        else Path("/nonexistent"),
        KAGGLE_WORKING_ROOT / "config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp504 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or config_path()).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("config must be a mapping")
    validate_immutable_config(config)
    return config


def artifacts_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = find_project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def work_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        output = KAGGLE_WORKING_ROOT / ".exp504_work"
    else:
        output = Path("/tmp") / "exp504_h512_rank_work"
    output.mkdir(parents=True, exist_ok=True)
    return output


def runtime_metrics_path() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return find_project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
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


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def decompressed_gzip_sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def frame_content_sha256(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    selected = frame if columns is None else frame[list(columns)]
    selected = selected.copy()
    for column in selected.select_dtypes(include=["string"]).columns:
        selected[column] = selected[column].astype(object)
    digest = hashlib.sha256()
    digest.update("|".join(selected.columns).encode())
    digest.update("|".join(str(dtype) for dtype in selected.dtypes).encode())
    row_hashes = pd.util.hash_pandas_object(selected, index=False, categorize=True)
    digest.update(row_hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def array_bundle_sha256(names: Sequence[str], *arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(names), separators=(",", ":")).encode())
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def keyed_prediction_sha256(ids: Sequence[str], prediction: np.ndarray) -> str:
    if len(ids) != len(prediction):
        raise ValueError("prediction key/value length mismatch")
    digest = hashlib.sha256()
    for value in ids:
        digest.update(str(value).encode())
        digest.update(b"\0")
    digest.update(np.asarray(prediction, dtype="<f4").tobytes())
    return digest.hexdigest()


def expand_paths(patterns: Sequence[str]) -> list[Path]:
    root = find_project_root()
    found: dict[str, Path] = {}
    for raw in patterns:
        raw_path = Path(str(raw))
        direct = raw_path if raw_path.is_absolute() else root / raw_path
        if direct.exists():
            found.setdefault(str(direct.resolve()), direct)
        for pattern in (str(raw), str(root / str(raw)) if not raw_path.is_absolute() else str(raw)):
            for match in glob.glob(pattern, recursive=True):
                path = Path(match)
                if path.exists():
                    found.setdefault(str(path.resolve()), path)
    return list(found.values())


def resolve_file(
    patterns: Sequence[str], *, label: str, expected_sha256: str | None = None
) -> Path:
    root = find_project_root()
    # Most repo and Kaggle contracts list canonical exact paths first.  Return
    # an exact SHA-matching path before considering recursive fallbacks; a
    # recursive scan of hundreds of experiment outputs is otherwise slower
    # than the actual fixed-file preflight.
    for raw in patterns:
        if any(token in str(raw) for token in ("*", "?", "[")):
            continue
        raw_path = Path(str(raw))
        direct = raw_path if raw_path.is_absolute() else root / raw_path
        if direct.is_file() and (not expected_sha256 or sha256_file(direct) == expected_sha256):
            return direct
    candidates = [path for path in expand_paths(patterns) if path.is_file()]
    if expected_sha256:
        matched = [path for path in candidates if sha256_file(path) == expected_sha256]
        if matched:
            return sorted(matched, key=lambda item: (len(str(item)), str(item)))[0]
        if candidates:
            evidence = {str(path): sha256_file(path) for path in candidates}
            raise ValueError(f"{label} SHA mismatch: {evidence}")
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"{label} not found from patterns={list(patterns)}")
    return sorted(candidates, key=lambda item: (len(str(item)), str(item)))[0]


def resolve_directory(patterns: Sequence[str], *, label: str) -> Path:
    candidates = [path for path in expand_paths(patterns) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"{label} not found from patterns={list(patterns)}")
    return sorted(candidates, key=lambda item: (len(str(item)), str(item)))[0]


def deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    frame.to_csv(
        path,
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
    )
    return {
        "path": str(path),
        "rows": len(frame),
        "file_sha256": sha256_file(path),
        "decompressed_content_sha256": decompressed_gzip_sha256(path),
        "logical_content_sha256": frame_content_sha256(frame),
    }


# %% [markdown]
# ## 3. Exp263 fixed12 cache and corrected exp264 row features


# %%
@dataclass
class FoldBundle:
    base: pd.DataFrame
    values: np.ndarray
    available: np.ndarray
    confidence: dict[str, pd.DataFrame]
    candidate_ids: tuple[str, ...] = EXPECTED_CANDIDATE_ORDER
    specs: dict[str, dict[str, Any]] = field(default_factory=lambda: CANDIDATE_SPECS)


def _artifact_path_from_manifest(manifest_path: Path, item: Mapping[str, Any]) -> Path:
    raw = str(item["path"])
    marker = "/artifacts/"
    if marker in raw:
        candidate = manifest_path.parent / raw.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    direct = Path(raw)
    if direct.exists():
        return direct
    suffix = Path(raw).parts[-4:]
    for root in manifest_path.parents:
        candidate = root.joinpath(*suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"manifest partition is missing: {raw}")


def resolve_exp263_root(config: Mapping[str, Any]) -> Path:
    spec = get_nested(config, "data.exp263_cache")
    for candidate in expand_paths(spec["root_patterns"]):
        if (candidate / "cache_manifest.json").is_file():
            return candidate
    manifest = resolve_file(
        spec["manifest_patterns"],
        label="exp263 cache manifest",
        expected_sha256=str(spec["expected_manifest_file_sha256"]),
    )
    return manifest.parent


class Exp263Cache:
    def __init__(self, root: Path, config: Mapping[str, Any]):
        self.root = Path(root)
        self.manifest_path = self.root / "cache_manifest.json"
        self.catalog_path = self.root / "candidate_catalog.json"
        if not self.manifest_path.is_file() or not self.catalog_path.is_file():
            raise FileNotFoundError("exp263 cache manifest/catalog are missing")
        spec = get_nested(config, "data.exp263_cache")
        if sha256_file(self.manifest_path) != str(spec["expected_manifest_file_sha256"]):
            raise ValueError("exp263 manifest SHA mismatch")
        if sha256_file(self.catalog_path) != str(spec["expected_catalog_file_sha256"]):
            raise ValueError("exp263 catalog SHA mismatch")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("canonical_id_sha256") != spec["expected_canonical_id_sha256"]:
            raise ValueError("exp263 canonical key SHA changed")
        expected = get_nested(config, "technical_gate")
        for key, manifest_key in (
            ("expected_rows", "rows"),
            ("expected_wells", "wells"),
            ("expected_folds", "folds"),
        ):
            if int(self.manifest.get(manifest_key, -1)) != int(expected[key]):
                raise ValueError(f"exp263 manifest {manifest_key} changed")
        self.input_evidence: list[dict[str, Any]] = [
            {
                "source": "exp263_manifest",
                "path": str(self.manifest_path),
                "file_sha256": sha256_file(self.manifest_path),
            },
            {
                "source": "exp263_catalog",
                "path": str(self.catalog_path),
                "file_sha256": sha256_file(self.catalog_path),
            },
        ]

    def _read_partition(self, kind: str, candidate_id: str, fold: int) -> pd.DataFrame:
        manifest_key = f"candidate_{kind}_partitions"
        items = self.manifest[manifest_key][candidate_id]
        matching = [item for item in items if f"fold={fold}" in str(item["path"])]
        if not matching:
            raise FileNotFoundError(f"{manifest_key}/{candidate_id}/fold={fold}")
        frames: list[pd.DataFrame] = []
        for item in matching:
            path = _artifact_path_from_manifest(self.manifest_path, item)
            actual_sha = sha256_file(path)
            if actual_sha != str(item["file_sha256"]):
                raise ValueError(f"exp263 partition SHA mismatch: {path}")
            frame = pd.read_parquet(path)
            if len(frame) != int(item["rows"]):
                raise ValueError(f"exp263 partition row mismatch: {path}")
            forbidden = {
                column
                for column in frame.columns.astype(str)
                if any(
                    token in column.lower() for token in ("true_tvt", "abs_error", "oracle_label")
                )
            }
            if forbidden:
                raise ValueError(f"target fields in exp263 partition: {sorted(forbidden)}")
            frames.append(frame)
            self.input_evidence.append(
                {
                    "source": f"exp263_{kind}::{candidate_id}::fold{fold}",
                    "path": str(path),
                    "rows": len(frame),
                    "file_sha256": actual_sha,
                    "logical_content_sha256": str(item.get("content_sha256", "")),
                    "schema_sha256": str(item.get("schema_sha256", "")),
                }
            )
        output = pd.concat(frames, ignore_index=True)
        return output.sort_values(["well", "well_row_idx"], kind="stable").reset_index(drop=True)

    @staticmethod
    def _assert_keys(left: pd.DataFrame, right: pd.DataFrame) -> None:
        if len(left) != len(right):
            raise ValueError("exp263 candidate partition row mismatch")
        for column in KEY_COLUMNS:
            a = left[column].to_numpy()
            b = right[column].to_numpy()
            same = (
                np.array_equal(a, b, equal_nan=True)
                if column == "md_since"
                else np.array_equal(a, b)
            )
            if not same:
                raise ValueError(f"exp263 candidate key mismatch: {column}")

    def load_fold(self, fold: int) -> FoldBundle:
        primitive_frames: dict[str, pd.DataFrame] = {}
        confidence: dict[str, pd.DataFrame] = {}
        for candidate_id in PRIMITIVE_CANDIDATES:
            value = self._read_partition("value", candidate_id, fold)
            conf = self._read_partition("confidence", candidate_id, fold)
            self._assert_keys(value, conf)
            primitive_frames[candidate_id] = value
            confidence[candidate_id] = conf
        reference = primitive_frames[PRIMITIVE_CANDIDATES[0]]
        for candidate_id in PRIMITIVE_CANDIDATES[1:]:
            self._assert_keys(reference, primitive_frames[candidate_id])
        base = reference[[*KEY_COLUMNS, "last_known_tvt"]].copy()
        by_id: dict[str, np.ndarray] = {}
        available_by_id: dict[str, np.ndarray] = {}
        for candidate_id, frame in primitive_frames.items():
            values = pd.to_numeric(frame["candidate_tvt"], errors="coerce").to_numpy(np.float32)
            available = frame["candidate_available"].astype(bool).to_numpy() & np.isfinite(values)
            values[~available] = np.nan
            by_id[candidate_id] = values
            available_by_id[candidate_id] = available
        for candidate_id in EXPECTED_CANDIDATE_ORDER[6:]:
            spec = CANDIDATE_SPECS[candidate_id]
            parents = [str(value) for value in spec["parents"]]
            weights = np.asarray(spec["weights"], dtype=np.float32)
            if candidate_id == ANCHOR_CANDIDATE:
                combined = (
                    np.float32(0.5) * by_id["exp226_k16"]
                    + np.float32(0.25) * by_id["likpf_mean"]
                    + np.float32(0.25) * by_id["exact_hmm"]
                ).astype(np.float32)
            else:
                combined = (np.float32(0.5) * (by_id[parents[0]] + by_id[parents[1]])).astype(
                    np.float32
                )
            by_id[candidate_id] = combined
            available_by_id[candidate_id] = np.logical_and.reduce(
                [available_by_id[parent] for parent in parents]
            ) & np.isfinite(combined)
        values = np.column_stack([by_id[name] for name in EXPECTED_CANDIDATE_ORDER]).astype(
            np.float32
        )
        available = np.column_stack(
            [available_by_id[name] for name in EXPECTED_CANDIDATE_ORDER]
        ).astype(bool)
        if not available.all() or not np.isfinite(values).all():
            raise ValueError("fixed12 candidate coverage is not complete")
        return FoldBundle(base=base, values=values, available=available, confidence=confidence)


def resolve_raw_train_dir(config: Mapping[str, Any]) -> Path:
    spec = get_nested(config, "data.raw_train")
    candidates = [path for path in expand_paths(spec["directory_patterns"]) if path.is_dir()]
    expected_wells = int(get_nested(config, "technical_gate.expected_wells"))
    for path in sorted(candidates, key=lambda item: (len(str(item)), str(item))):
        if len(list(path.glob(str(spec["horizontal_glob"])))) == expected_wells:
            return path
    raise FileNotFoundError("raw train directory with 773 horizontal wells was not found")


def _raw_horizontal_path(raw_dir: Path, well: str) -> Path:
    path = raw_dir / f"{well}__horizontal_well.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _raw_typewell_path(raw_dir: Path, well: str) -> Path:
    path = raw_dir / f"{well}__typewell.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_raw_context(base: pd.DataFrame, raw_dir: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    generator = get_nested(config, "features.generator")
    allowlist = [str(value) for value in generator["horizontal_numeric_allowlist"]]
    if allowlist != ["MD", "X", "Y", "Z", "GR"]:
        raise ValueError("raw context allowlist changed")
    context = pd.DataFrame(index=np.arange(len(base)))
    context["ctx__md_since"] = pd.to_numeric(base["md_since"], errors="raise").to_numpy(np.float32)
    context["ctx__last_known_tvt"] = pd.to_numeric(base["last_known_tvt"], errors="raise").to_numpy(
        np.float32
    )
    context["ctx__well_row_idx"] = pd.to_numeric(base["well_row_idx"], errors="raise").to_numpy(
        np.float32
    )
    for column in allowlist:
        context[f"ctx__raw__{column.lower()}"] = np.nan
        context[f"ctx__raw_delta_last__{column.lower()}"] = np.nan
    typewell_columns = (
        "ctx__typewell__rows",
        "ctx__typewell__tvt_min",
        "ctx__typewell__tvt_max",
        "ctx__typewell__gr_mean",
        "ctx__typewell__gr_std",
        "ctx__typewell__gr_min",
        "ctx__typewell__gr_max",
        "ctx__typewell__row_gr_z",
    )
    for column in typewell_columns:
        context[column] = np.nan
    eval_len = base.groupby("well", sort=False)["id"].transform("size").to_numpy(np.float32)
    eval_position = base.groupby("well", sort=False).cumcount().to_numpy(np.float32) + 1.0
    context["ctx__eval_len"] = eval_len
    context["ctx__evaluation_progress"] = eval_position / np.maximum(eval_len, 1.0)

    for well, positions in base.groupby("well", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        raw = pd.read_csv(_raw_horizontal_path(raw_dir, str(well)))
        row_index = pd.to_numeric(base.iloc[pos]["well_row_idx"], errors="raise").to_numpy(np.int64)
        if row_index.min(initial=0) < 0 or row_index.max(initial=-1) >= len(raw):
            raise ValueError(f"raw row index out of bounds for well={well}")
        selected = raw.iloc[row_index]
        known = pd.to_numeric(raw["TVT_input"], errors="coerce").notna().to_numpy()
        previous_known = np.flatnonzero(
            known & (np.arange(len(raw)) < row_index.min(initial=len(raw)))
        )
        last_known_idx = (
            int(previous_known[-1]) if len(previous_known) else max(int(row_index.min()) - 1, 0)
        )
        for column in allowlist:
            if column not in raw:
                raise ValueError(f"raw-test-safe column is absent: {well}/{column}")
            current = pd.to_numeric(selected[column], errors="coerce").to_numpy(np.float32)
            context.loc[pos, f"ctx__raw__{column.lower()}"] = current
            anchor = float(pd.to_numeric(raw.iloc[last_known_idx][column], errors="coerce"))
            context.loc[pos, f"ctx__raw_delta_last__{column.lower()}"] = current - np.float32(
                anchor
            )
        typewell = pd.read_csv(_raw_typewell_path(raw_dir, str(well)))
        tw_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64)
        tw_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64)
        finite_tvt = tw_tvt[np.isfinite(tw_tvt)]
        finite_gr = tw_gr[np.isfinite(tw_gr)]
        gr_mean = float(np.mean(finite_gr)) if len(finite_gr) else np.nan
        gr_std = float(np.std(finite_gr)) if len(finite_gr) else np.nan
        summaries = {
            "ctx__typewell__rows": float(len(typewell)),
            "ctx__typewell__tvt_min": float(np.min(finite_tvt)) if len(finite_tvt) else np.nan,
            "ctx__typewell__tvt_max": float(np.max(finite_tvt)) if len(finite_tvt) else np.nan,
            "ctx__typewell__gr_mean": gr_mean,
            "ctx__typewell__gr_std": gr_std,
            "ctx__typewell__gr_min": float(np.min(finite_gr)) if len(finite_gr) else np.nan,
            "ctx__typewell__gr_max": float(np.max(finite_gr)) if len(finite_gr) else np.nan,
        }
        for column, value in summaries.items():
            context.loc[pos, column] = value
        row_gr = pd.to_numeric(selected["GR"], errors="coerce").to_numpy(np.float64)
        context.loc[pos, "ctx__typewell__row_gr_z"] = (row_gr - gr_mean) / max(gr_std, 1e-6)
    return context.astype(np.float32)


@dataclass
class ShapeState:
    values: np.ndarray
    previous: np.ndarray
    group_start: np.ndarray
    cumulative_abs_step: np.ndarray

    @classmethod
    def from_bundle(cls, base: pd.DataFrame, values: np.ndarray) -> "ShapeState":
        n_rows = len(base)
        previous = np.arange(n_rows, dtype=np.int64) - 1
        group_start = np.zeros(n_rows, dtype=np.int64)
        cumulative = np.zeros_like(values, dtype=np.float32)
        for positions in base.groupby("well", sort=False).indices.values():
            pos = np.asarray(positions, dtype=np.int64)
            start = int(pos[0])
            group_start[pos] = start
            previous[start] = start
            steps = np.zeros((len(pos), values.shape[1]), dtype=np.float32)
            if len(pos) > 1:
                steps[1:] = np.abs(np.diff(values[pos], axis=0)).astype(np.float32)
            cumulative[pos] = np.cumsum(steps, axis=0, dtype=np.float32)
        return cls(values, previous, group_start, cumulative)

    def extract(self, indices: np.ndarray, windows: Sequence[int]) -> dict[str, np.ndarray]:
        idx = np.asarray(indices, dtype=np.int64)
        prev = np.maximum(self.previous[idx], self.group_start[idx])
        step = self.values[idx] - self.values[prev]
        prev2 = np.maximum(self.previous[prev], self.group_start[idx])
        previous_step = self.values[prev] - self.values[prev2]
        output = {
            "cand__step": step.astype(np.float32),
            "cand__curvature": (step - previous_step).astype(np.float32),
        }
        for window in windows:
            lag = np.maximum(idx - int(window), self.group_start[idx])
            span = np.maximum(idx - lag, 1).astype(np.float32)[:, None]
            net = self.values[idx] - self.values[lag]
            slope = net / span
            prev_idx = np.maximum(idx - 1, self.group_start[idx])
            prev_lag = np.maximum(prev_idx - int(window), self.group_start[idx])
            prev_span = np.maximum(prev_idx - prev_lag, 1).astype(np.float32)[:, None]
            previous_slope = (self.values[prev_idx] - self.values[prev_lag]) / prev_span
            path = self.cumulative_abs_step[idx] - self.cumulative_abs_step[lag]
            output[f"cand__slope_{window}"] = slope.astype(np.float32)
            output[f"cand__curvature_{window}"] = (slope - previous_slope).astype(np.float32)
            output[f"cand__straightness_{window}"] = (np.abs(net) / np.maximum(path, 1e-6)).astype(
                np.float32
            )
        return output


def load_feature_schema(config: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    spec = get_nested(config, "data.row_feature_schema")
    path = resolve_file(
        spec["source_patterns"],
        label="corrected exp264 feature schema",
        expected_sha256=str(spec["source_file_sha256"]),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = [str(value) for value in payload["features"]]
    if len(features) != int(spec["feature_count"]) or len(set(features)) != len(features):
        raise ValueError("corrected exp264 feature schema count/uniqueness changed")
    if payload.get("feature_schema_sha256") != spec["feature_schema_logical_sha256"]:
        raise ValueError("corrected exp264 logical feature schema SHA changed")
    if tuple(payload.get("candidate_order", [])) != EXPECTED_CANDIDATE_ORDER:
        raise ValueError("corrected exp264 schema candidate order changed")
    if bool(payload.get("ordinal_candidate_index", True)):
        raise ValueError("ordinal candidate index is forbidden")
    forbidden_tokens = tuple(
        str(value).lower()
        for value in get_nested(config, "features.generator.forbidden_feature_tokens")
    )
    forbidden = [
        name for name in features if any(token in name.lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise ValueError(f"target/readout fields in feature schema: {forbidden}")
    return features, {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "logical_sha256": payload["feature_schema_sha256"],
        "feature_count": len(features),
    }


def _confidence_numeric_fields(confidence: Mapping[str, pd.DataFrame]) -> list[str]:
    excluded = set(KEY_COLUMNS) | {
        "candidate_id",
        "confidence_source",
        "confidence_valid",
        "confidence_missing_fields",
    }
    fields: set[str] = set()
    for frame in confidence.values():
        for column in frame.columns:
            if column in excluded:
                continue
            if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(
                frame[column]
            ):
                fields.add(str(column))
    return sorted(fields)


def build_selected_feature_cube(
    bundle: FoldBundle,
    raw_context: pd.DataFrame,
    indices: np.ndarray,
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    *,
    shape_state: ShapeState | None = None,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    n_rows = len(idx)
    n_candidates = len(EXPECTED_CANDIDATE_ORDER)
    if shape_state is None:
        shape_state = ShapeState.from_bundle(bundle.base, bundle.values)
    windows = [int(value) for value in get_nested(config, "features.generator.shape_windows")]
    values = bundle.values[idx]
    anchor = pd.to_numeric(bundle.base.iloc[idx]["last_known_tvt"], errors="raise").to_numpy(
        np.float32
    )
    feature_arrays: dict[str, np.ndarray] = {}

    context = raw_context.iloc[idx]
    for name in context.columns:
        feature_arrays[name] = np.broadcast_to(
            context[name].to_numpy(np.float32)[:, None], (n_rows, n_candidates)
        )

    feature_arrays["cand__tvt"] = values
    feature_arrays["cand__minus_last"] = values - anchor[:, None]
    feature_arrays.update(shape_state.extract(idx, windows))

    bank_median = np.nanmedian(values, axis=1)
    bank_range = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
    bank_std = np.nanstd(values, axis=1)
    repeated_median = np.broadcast_to(bank_median[:, None], values.shape)
    feature_arrays["bank__median"] = repeated_median
    feature_arrays["bank__range"] = np.broadcast_to(bank_range[:, None], values.shape)
    feature_arrays["bank__std"] = np.broadcast_to(bank_std[:, None], values.shape)
    feature_arrays["bank__candidate_minus_median"] = values - repeated_median
    feature_arrays["bank__candidate_abs_minus_median"] = np.abs(values - repeated_median)
    feature_arrays["bank__candidate_rank_fraction"] = np.argsort(
        np.argsort(values, axis=1), axis=1
    ).astype(np.float32) / np.float32(n_candidates - 1)
    feature_arrays["bank__candidate_is_min"] = (
        values == np.nanmin(values, axis=1)[:, None]
    ).astype(np.float32)
    feature_arrays["bank__candidate_is_max"] = (
        values == np.nanmax(values, axis=1)[:, None]
    ).astype(np.float32)
    feature_arrays["bank__candidate_mean_abs_disagreement"] = np.mean(
        np.abs(values[:, :, None] - values[:, None, :]), axis=2
    ).astype(np.float32)
    id_to_index = {name: index for index, name in enumerate(EXPECTED_CANDIDATE_ORDER)}
    for domain in ("primary", "fixed"):
        domain_ids = [
            str(value) for value in get_nested(config, f"features.generator.{domain}_domain")
        ]
        matrix = values[:, [id_to_index[name] for name in domain_ids]]
        feature_arrays[f"bank__{domain}_median"] = np.broadcast_to(
            np.nanmedian(matrix, axis=1)[:, None], values.shape
        )
        feature_arrays[f"bank__{domain}_range"] = np.broadcast_to(
            (np.nanmax(matrix, axis=1) - np.nanmin(matrix, axis=1))[:, None], values.shape
        )
        feature_arrays[f"bank__{domain}_std"] = np.broadcast_to(
            np.nanstd(matrix, axis=1)[:, None], values.shape
        )

    for position, candidate_id in enumerate(EXPECTED_CANDIDATE_ORDER):
        matrix = np.zeros((n_rows, n_candidates), dtype=np.float32)
        matrix[:, position] = 1.0
        feature_arrays[f"id__candidate__{candidate_id}"] = matrix
    for kind in ("pair_mean_50", "primitive", "named_fixed"):
        membership = np.asarray(
            [CANDIDATE_SPECS[name]["kind"] == kind for name in EXPECTED_CANDIDATE_ORDER],
            dtype=np.float32,
        )
        feature_arrays[f"id__kind__{kind}"] = np.broadcast_to(membership, (n_rows, n_candidates))

    native_fields = _confidence_numeric_fields(bundle.confidence)
    native = {
        name: np.full((n_rows, n_candidates), np.nan, dtype=np.float32) for name in native_fields
    }
    confidence_valid = np.zeros((n_rows, n_candidates), dtype=np.float32)
    primitive_valid: dict[str, np.ndarray] = {}
    primitive_slots: dict[tuple[str, str], np.ndarray] = {}
    for position, candidate_id in enumerate(PRIMITIVE_CANDIDATES):
        frame = bundle.confidence[candidate_id].iloc[idx]
        valid = frame["confidence_valid"].astype(bool).to_numpy()
        primitive_valid[candidate_id] = valid
        confidence_valid[:, position] = valid.astype(np.float32)
        for name in native_fields:
            if name in frame:
                native[name][:, position] = pd.to_numeric(frame[name], errors="coerce").to_numpy(
                    np.float32
                )
        for slot in COMMON_CONFIDENCE_SLOTS:
            if slot in frame:
                primitive_slots[(candidate_id, slot)] = pd.to_numeric(
                    frame[slot], errors="coerce"
                ).to_numpy(np.float32)
    for name, matrix in native.items():
        feature_arrays[f"conf__native__{name}"] = matrix

    parent_valid_count = np.full((n_rows, n_candidates), np.nan, dtype=np.float32)
    component_range = np.full_like(parent_valid_count, np.nan)
    component_std = np.full_like(parent_valid_count, np.nan)
    direction_agreement = np.full_like(parent_valid_count, np.nan)
    weight_max = np.full_like(parent_valid_count, np.nan)
    weight_entropy = np.full_like(parent_valid_count, np.nan)
    for position, candidate_id in enumerate(EXPECTED_CANDIDATE_ORDER):
        parents = [str(value) for value in CANDIDATE_SPECS[candidate_id].get("parents", [])]
        if not parents:
            continue
        parent_positions = [id_to_index[parent] for parent in parents]
        components = values[:, parent_positions]
        component_range[:, position] = np.ptp(components, axis=1)
        component_std[:, position] = np.std(components, axis=1)
        directions = np.sign(components - anchor[:, None])
        direction_agreement[:, position] = np.all(directions == directions[:, :1], axis=1)
        valids = [primitive_valid.get(parent, np.zeros(n_rows, dtype=bool)) for parent in parents]
        parent_valid_count[:, position] = np.sum(valids, axis=0)
        confidence_valid[:, position] = np.logical_or.reduce(valids).astype(np.float32)
        weights = np.asarray(CANDIDATE_SPECS[candidate_id]["weights"], dtype=np.float64)
        weight_max[:, position] = float(weights.max())
        weight_entropy[:, position] = float(-np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0))))
    feature_arrays["conf__native_valid"] = confidence_valid
    feature_arrays["formula__parent_valid_count"] = parent_valid_count
    feature_arrays["formula__component_range"] = component_range
    feature_arrays["formula__component_std"] = component_std
    feature_arrays["formula__parent_direction_agreement"] = direction_agreement
    feature_arrays["formula__weight_max"] = weight_max
    feature_arrays["formula__weight_entropy"] = weight_entropy

    for parent in PRIMITIVE_CANDIDATES:
        membership = np.asarray(
            [
                parent in CANDIDATE_SPECS[name].get("parents", [])
                for name in EXPECTED_CANDIDATE_ORDER
            ],
            dtype=bool,
        )
        member_matrix = np.broadcast_to(membership, (n_rows, n_candidates))
        valid = primitive_valid.get(parent, np.zeros(n_rows, dtype=bool))[:, None]
        feature_arrays[f"formula__parent__{parent}__confidence_valid"] = (
            member_matrix & valid
        ).astype(np.float32)
        for slot in COMMON_CONFIDENCE_SLOTS:
            source = primitive_slots.get((parent, slot), np.full(n_rows, np.nan, dtype=np.float32))[
                :, None
            ]
            feature_arrays[f"formula__parent__{parent}__{slot}"] = np.where(
                member_matrix, source, np.nan
            ).astype(np.float32)

    missing = [name for name in feature_names if name not in feature_arrays]
    if missing:
        raise ValueError(f"corrected exp264 features cannot be generated: {missing}")
    cube = np.stack([feature_arrays[name] for name in feature_names], axis=2).astype(np.float32)
    if np.isinf(cube).any():
        raise ValueError("selected exp264 feature cube contains infinity")
    return cube


# %% [markdown]
# ## 4. H512 candidate-block aggregation and target-free freeze


# %%
def load_block_assignment(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.block_assignment")
    patterns = [
        str(spec["source_file"]),
        "/kaggle/input/notebooks/kentookumura/exp293-physics-bank-headroom-audit-train/artifacts/exp293_physics_only_candidate_bank_headroom_contract_block_assignment.csv.gz",
        "/kaggle/input/exp293-physics-bank-headroom-audit-train/artifacts/exp293_physics_only_candidate_bank_headroom_contract_block_assignment.csv.gz",
        "**/exp293_physics_only_candidate_bank_headroom_contract_block_assignment.csv.gz",
    ]
    path = resolve_file(
        patterns, label="exp293 H512 block assignment", expected_sha256=str(spec["file_sha256"])
    )
    if decompressed_gzip_sha256(path) != str(spec["decompressed_content_sha256"]):
        raise ValueError("exp293 block assignment decompressed SHA mismatch")
    dtype = {
        "id": str,
        "well": str,
        "well_row_idx": "int32",
        "outer_fold": "int8",
        "md_since": "float32",
        "well_code": "int32",
        "h128_group": "int32",
        "h256_group": "int32",
        "h512_group": "int32",
        "whole_well_group": "int32",
    }
    frame = pd.read_csv(path, dtype=dtype)
    if list(frame.columns) != BLOCK_COLUMNS:
        raise ValueError("exp293 block assignment columns changed")
    technical = get_nested(config, "technical_gate")
    if len(frame) != int(technical["expected_rows"]):
        raise ValueError("block assignment row count changed")
    if frame["well"].nunique() != int(technical["expected_wells"]):
        raise ValueError("block assignment well count changed")
    if frame["h512_group"].nunique() != int(technical["expected_h512_queries"]):
        raise ValueError("H512 query count changed")
    if frame["id"].duplicated().any():
        raise ValueError("block assignment row keys are duplicated")
    return frame, {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "decompressed_content_sha256": decompressed_gzip_sha256(path),
        # Exact decompressed bytes are the hard gate.  Recomputing pandas row
        # hashes over 3.78M string-key rows is needlessly expensive and can
        # differ across pandas releases, so the upstream logical SHA is chained
        # only after the exact file/decompressed checks pass.
        "logical_content_sha256": str(spec["logical_content_sha256"]),
        "logical_sha_verification": "chained_from_exact_decompressed_content",
    }


def candidate_bank_content_sha256(
    keys: pd.DataFrame, values: np.ndarray, expected_key_sha: str, chunk_rows: int = 100_000
) -> tuple[str, str]:
    if list(keys.columns[: len(KEY_COLUMNS)]) != KEY_COLUMNS:
        raise ValueError("candidate key column order changed")
    if keys["id"].duplicated().any():
        raise ValueError("candidate keys are duplicated")
    # The key SHA is chained from the exact exp293 block-assignment bytes,
    # whose file and decompressed SHA are verified before this function.
    key_sha = str(expected_key_sha)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(EXPECTED_CANDIDATE_ORDER), separators=(",", ":")).encode())
    digest.update(key_sha.encode())
    for position, candidate_id in enumerate(EXPECTED_CANDIDATE_ORDER):
        digest.update(candidate_id.encode())
        for start in range(0, len(keys), chunk_rows):
            end = min(start + chunk_rows, len(keys))
            digest.update(np.asarray(values[start:end, position], dtype="<f4").tobytes())
    return digest.hexdigest(), key_sha


def aggregate_feature_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("block feature matrix must be non-empty and two-dimensional")
    finite = np.isfinite(values)
    finite_fraction = finite.mean(axis=0, dtype=np.float64)
    masked = np.where(finite, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            mean = np.nanmean(masked, axis=0)
            std = np.nanstd(masked, axis=0, ddof=0)
            quantiles = np.nanquantile(masked, [0.10, 0.50, 0.90], axis=0, method="linear")
    first = np.full(values.shape[1], np.nan, dtype=np.float32)
    last = np.full(values.shape[1], np.nan, dtype=np.float32)
    for column in range(values.shape[1]):
        indices = np.flatnonzero(finite[:, column])
        if len(indices):
            first[column] = values[indices[0], column]
            last[column] = values[indices[-1], column]
    output = np.stack(
        [
            finite_fraction,
            mean,
            std,
            quantiles[0],
            quantiles[1],
            quantiles[2],
            first,
            last,
            last - first,
        ],
        axis=1,
    )
    return output.astype(np.float32).reshape(-1)


def block_feature_names(feature_names: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    shared = [name for name in feature_names if name.startswith("ctx__")]
    candidate = [name for name in feature_names if not name.startswith("ctx__")]
    candidate_aggregated = [
        f"{name}__agg__{operator}" for name in candidate for operator in AGGREGATION_NAMES
    ]
    shared_aggregated = [
        f"{name}__agg__{operator}" for name in shared for operator in AGGREGATION_NAMES
    ]
    pair_names = (
        [f"pair__left_minus_right__{name}" for name in candidate_aggregated]
        + [f"pair__abs_left_minus_right__{name}" for name in candidate_aggregated]
        + [f"pair__left_plus_right_div2__{name}" for name in candidate_aggregated]
        + [f"pair__shared__{name}" for name in shared_aggregated]
        + [f"pair__block__{name}" for name in BLOCK_CONTEXT_NAMES]
    )
    return candidate_aggregated, shared_aggregated, pair_names


@dataclass
class TargetFreeSurface:
    blocks: pd.DataFrame
    candidate_features: np.ndarray
    shared_features: np.ndarray
    block_context: np.ndarray
    candidate_values: np.memmap
    candidate_feature_names: list[str]
    shared_feature_names: list[str]
    pair_feature_names: list[str]
    feature_schema_sha256: str
    feature_content_sha256: str
    candidate_bank_sha256: str
    candidate_key_sha256: str
    input_evidence: list[dict[str, Any]]


def _block_bounds(block_assignment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected = np.arange(block_assignment["h512_group"].nunique(), dtype=np.int32)
    seen: list[int] = []
    for block_id, group in block_assignment.groupby("h512_group", sort=True):
        positions = group.index.to_numpy(np.int64)
        if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
            raise ValueError("H512 block rows are not contiguous")
        if group["well"].nunique() != 1 or group["outer_fold"].nunique() != 1:
            raise ValueError("H512 block spans well/fold boundaries")
        seen.append(int(block_id))
        rows.append(
            {
                "h512_group": int(block_id),
                "well": str(group["well"].iloc[0]),
                "outer_fold": int(group["outer_fold"].iloc[0]),
                "row_start": int(positions[0]),
                "row_stop_exclusive": int(positions[-1] + 1),
                "row_count": int(len(group)),
            }
        )
    if not np.array_equal(np.asarray(seen, dtype=np.int32), expected):
        raise ValueError("H512 block IDs are not dense and ordered")
    return pd.DataFrame(rows)


def build_target_free_surface(config: Mapping[str, Any]) -> TargetFreeSurface:
    output = artifacts_dir()
    scratch = work_dir()
    block_assignment, block_evidence = load_block_assignment(config)
    feature_names, schema_evidence = load_feature_schema(config)
    candidate_feature_names, shared_feature_names, pair_feature_names = block_feature_names(
        feature_names
    )
    expected_shared = int(
        get_nested(config, "features.candidate_block_aggregation.shared_context_feature_count")
    )
    expected_candidate = int(
        get_nested(config, "features.candidate_block_aggregation.candidate_specific_feature_count")
    )
    if len(shared_feature_names) != expected_shared * len(AGGREGATION_NAMES):
        raise ValueError("shared context feature count changed")
    if len(candidate_feature_names) != expected_candidate * len(AGGREGATION_NAMES):
        raise ValueError("candidate-specific feature count changed")
    if len(pair_feature_names) != int(get_nested(config, "runtime.max_pair_feature_columns")):
        raise ValueError("pair feature width changed")

    blocks = _block_bounds(block_assignment)
    n_rows = len(block_assignment)
    n_blocks = len(blocks)
    n_candidates = len(EXPECTED_CANDIDATE_ORDER)
    bank_path = scratch / f"{OUTPUT_PREFIX}_candidate_bank.f32"
    candidate_values = np.memmap(
        bank_path, mode="w+", dtype="float32", shape=(n_rows, n_candidates)
    )
    candidate_values[:] = np.nan
    candidate_block = np.full(
        (n_blocks, n_candidates, len(candidate_feature_names)), np.nan, dtype=np.float32
    )
    shared_block = np.full((n_blocks, len(shared_feature_names)), np.nan, dtype=np.float32)
    block_context = np.full((n_blocks, len(BLOCK_CONTEXT_NAMES)), np.nan, dtype=np.float32)

    cache = Exp263Cache(resolve_exp263_root(config), config)
    raw_dir = resolve_raw_train_dir(config)
    global_index = pd.Index(block_assignment["id"].astype(str))
    feature_index = {name: position for position, name in enumerate(feature_names)}
    shared_names = [name for name in feature_names if name.startswith("ctx__")]
    candidate_names = [name for name in feature_names if not name.startswith("ctx__")]
    shared_pos = [feature_index[name] for name in shared_names]
    candidate_pos = [feature_index[name] for name in candidate_names]
    batch_size = int(get_nested(config, "features.generator.block_batch_size"))

    for fold in range(int(get_nested(config, "validation.outer_folds"))):
        print(f"Building target-free exp264 H512 features for source fold {fold}", flush=True)
        bundle = cache.load_fold(fold)
        global_positions = global_index.get_indexer(bundle.base["id"].astype(str))
        if np.any(global_positions < 0) or len(np.unique(global_positions)) != len(
            global_positions
        ):
            raise ValueError("exp263 fold keys do not map one-to-one to exp293 rows")
        if not np.all(
            block_assignment.iloc[global_positions]["outer_fold"].to_numpy(np.int8) == fold
        ):
            raise ValueError("exp263/exp293 fold identity mismatch")
        candidate_values[global_positions] = bundle.values
        raw_context = build_raw_context(bundle.base, raw_dir, config)
        shape_state = ShapeState.from_bundle(bundle.base, bundle.values)
        local_block_ids = block_assignment.iloc[global_positions]["h512_group"].to_numpy(np.int32)
        unique_blocks = np.unique(local_block_ids)
        for batch_start in range(0, len(unique_blocks), batch_size):
            batch_blocks = unique_blocks[batch_start : batch_start + batch_size]
            mask = np.isin(local_block_ids, batch_blocks)
            local_indices = np.flatnonzero(mask)
            cube = build_selected_feature_cube(
                bundle, raw_context, local_indices, feature_names, config, shape_state=shape_state
            )
            batch_group = local_block_ids[local_indices]
            for block_id in batch_blocks:
                row_mask = batch_group == block_id
                block_cube = cube[row_mask]
                shared_cube = block_cube[:, :, shared_pos]
                if not np.allclose(shared_cube, shared_cube[:, :1, :], equal_nan=True):
                    raise ValueError(f"ctx__ features differ across candidates in block {block_id}")
                for candidate in range(n_candidates):
                    candidate_block[block_id, candidate] = aggregate_feature_matrix(
                        block_cube[:, candidate, :][:, candidate_pos]
                    )
                shared_block[block_id] = aggregate_feature_matrix(shared_cube[:, 0, :])
                local_rows = local_indices[row_mask]
                md_since = raw_context.iloc[local_rows]["ctx__md_since"].to_numpy(np.float32)
                progress = raw_context.iloc[local_rows]["ctx__evaluation_progress"].to_numpy(
                    np.float32
                )
                block_context[block_id] = np.asarray(
                    [
                        len(local_rows),
                        float(len(local_rows) < 512),
                        md_since[0],
                        md_since[-1],
                        progress[0],
                        progress[-1],
                    ],
                    dtype=np.float32,
                )
            del cube
        del bundle, raw_context, shape_state
        gc.collect()
    candidate_values.flush()
    if not np.isfinite(candidate_values).all():
        raise ValueError("fixed12 candidate bank is incomplete")
    if (
        np.isinf(candidate_block).any()
        or np.isinf(shared_block).any()
        or not np.isfinite(block_context).all()
    ):
        raise ValueError("block features contain invalid infinity/context values")
    if np.isnan(candidate_block).all(axis=(1, 2)).any() or np.isnan(shared_block).all(axis=1).any():
        raise ValueError("a block has no finite aggregated features")

    bank_spec = get_nested(config, "data.candidate_bank")
    bank_sha, key_sha = candidate_bank_content_sha256(
        block_assignment,
        candidate_values,
        str(bank_spec["key_content_sha256"]),
    )
    if bank_sha != str(bank_spec["content_sha256"]):
        raise ValueError(f"exp293 fixed12 candidate bank SHA mismatch: {bank_sha}")
    feature_schema_sha = json_sha256(
        {
            "row_features": feature_names,
            "candidate_block_features": candidate_feature_names,
            "shared_block_features": shared_feature_names,
            "block_context": list(BLOCK_CONTEXT_NAMES),
            "pair_features": pair_feature_names,
        }
    )
    feature_content_sha = array_bundle_sha256(
        ["candidate_block", "shared_block", "block_context"],
        candidate_block,
        shared_block,
        block_context,
    )
    np.save(output / f"{OUTPUT_PREFIX}_candidate_block_features.npy", candidate_block)
    np.save(output / f"{OUTPUT_PREFIX}_shared_block_features.npy", shared_block)
    np.save(output / f"{OUTPUT_PREFIX}_block_context.npy", block_context)
    blocks.to_parquet(output / f"{OUTPUT_PREFIX}_block_metadata.parquet", index=False)
    schema_payload = {
        "row_feature_count": len(feature_names),
        "shared_row_feature_count": len(shared_names),
        "candidate_specific_row_feature_count": len(candidate_names),
        "candidate_block_feature_count": len(candidate_feature_names),
        "shared_block_feature_count": len(shared_feature_names),
        "block_context_feature_count": len(BLOCK_CONTEXT_NAMES),
        "pair_feature_count": len(pair_feature_names),
        "candidate_block_features": candidate_feature_names,
        "shared_block_features": shared_feature_names,
        "block_context_features": list(BLOCK_CONTEXT_NAMES),
        "pair_features": pair_feature_names,
        "feature_schema_sha256": feature_schema_sha,
        "feature_content_sha256": feature_content_sha,
    }
    write_json(output / f"{OUTPUT_PREFIX}_block_feature_schema.json", schema_payload)
    input_evidence = [block_evidence, schema_evidence, *cache.input_evidence]
    freeze_payload = {
        "experiment": EXPERIMENT_NAME,
        "status": "target_free_h512_pair_surface_frozen",
        "rows": n_rows,
        "wells": int(block_assignment["well"].nunique()),
        "h512_queries": n_blocks,
        "candidate_blocks": n_blocks * n_candidates,
        "candidate_order": list(EXPECTED_CANDIDATE_ORDER),
        "candidate_bank_content_sha256": bank_sha,
        "candidate_key_content_sha256": key_sha,
        "row_feature_schema_sha256": schema_evidence["logical_sha256"],
        "block_feature_schema_sha256": feature_schema_sha,
        "block_feature_content_sha256": feature_content_sha,
        "input_evidence_sha256": json_sha256(input_evidence),
        "truth_columns_read_before_freeze": [],
        "truth_access_count_before_freeze": 0,
    }
    write_json(output / f"{OUTPUT_PREFIX}_target_free_freeze.json", freeze_payload)
    return TargetFreeSurface(
        blocks=blocks,
        candidate_features=candidate_block,
        shared_features=shared_block,
        block_context=block_context,
        candidate_values=candidate_values,
        candidate_feature_names=candidate_feature_names,
        shared_feature_names=shared_feature_names,
        pair_feature_names=pair_feature_names,
        feature_schema_sha256=feature_schema_sha,
        feature_content_sha256=feature_content_sha,
        candidate_bank_sha256=bank_sha,
        candidate_key_sha256=key_sha,
        input_evidence=input_evidence,
    )


# %% [markdown]
# ## 5. Pair labels, regret weights, and pair feature matrices


# %%
@dataclass
class PairTargets:
    block_ids: np.ndarray
    left: np.ndarray
    right: np.ndarray
    label: np.ndarray
    sample_weight: np.ndarray
    canonical_table: pd.DataFrame


def build_pair_targets(
    block_mse: np.ndarray,
    row_count: np.ndarray,
    block_ids: np.ndarray,
    *,
    tie_tolerance: float,
) -> PairTargets:
    selected_blocks = np.asarray(block_ids, dtype=np.int32)
    left = np.tile(PAIR_LEFT.astype(np.int8), len(selected_blocks))
    right = np.tile(PAIR_RIGHT.astype(np.int8), len(selected_blocks))
    repeated_blocks = np.repeat(selected_blocks, len(PAIR_LEFT))
    left_mse = block_mse[repeated_blocks, left]
    right_mse = block_mse[repeated_blocks, right]
    delta = left_mse - right_mse
    keep = np.abs(delta) > float(tie_tolerance)
    canonical_block = repeated_blocks[keep]
    canonical_left = left[keep]
    canonical_right = right[keep]
    canonical_label = (delta[keep] < 0.0).astype(np.int8)
    raw_weight = row_count[canonical_block].astype(np.float64) * np.log1p(np.abs(delta[keep]))
    if not np.isfinite(raw_weight).all() or np.any(raw_weight <= 0.0):
        raise ValueError("pair regret weights are invalid")
    normalized = raw_weight / raw_weight.mean()
    canonical = pd.DataFrame(
        {
            "h512_group": canonical_block,
            "candidate_left": canonical_left,
            "candidate_right": canonical_right,
            "label_left_better": canonical_label,
            "raw_weight": raw_weight,
            "normalized_weight": normalized,
        }
    )
    return PairTargets(
        block_ids=np.concatenate([canonical_block, canonical_block]).astype(np.int32),
        left=np.concatenate([canonical_left, canonical_right]).astype(np.int8),
        right=np.concatenate([canonical_right, canonical_left]).astype(np.int8),
        label=np.concatenate([canonical_label, 1 - canonical_label]).astype(np.int8),
        sample_weight=np.concatenate([normalized / 2.0, normalized / 2.0]).astype(np.float32),
        canonical_table=canonical,
    )


def assemble_pair_features(
    candidate_features: np.ndarray,
    shared_features: np.ndarray,
    block_context: np.ndarray,
    block_ids: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    block_ids = np.asarray(block_ids, dtype=np.int64)
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if not (len(block_ids) == len(left) == len(right)):
        raise ValueError("pair feature identity lengths differ")
    left_value = candidate_features[block_ids, left]
    right_value = candidate_features[block_ids, right]
    output = np.concatenate(
        [
            left_value - right_value,
            np.abs(left_value - right_value),
            (left_value + right_value) / np.float32(2.0),
            shared_features[block_ids],
            block_context[block_ids],
        ],
        axis=1,
    ).astype(np.float32)
    if np.isinf(output).any():
        raise ValueError("pair feature matrix contains infinity")
    return output


def pair_feature_memmap(
    surface: TargetFreeSurface, targets: PairTargets, path: Path, chunk_rows: int = 4096
) -> np.memmap:
    matrix = np.memmap(
        path,
        mode="w+",
        dtype="float32",
        shape=(len(targets.block_ids), len(surface.pair_feature_names)),
    )
    for start in range(0, len(targets.block_ids), chunk_rows):
        end = min(start + chunk_rows, len(targets.block_ids))
        matrix[start:end] = assemble_pair_features(
            surface.candidate_features,
            surface.shared_features,
            surface.block_context,
            targets.block_ids[start:end],
            targets.left[start:end],
            targets.right[start:end],
        )
    matrix.flush()
    return matrix


# %% [markdown]
# ## 6. Pairwise LightGBM and antisymmetrized Borda selection


# %%
def make_rank_model(config: Mapping[str, Any]) -> Any:
    if LGBMClassifier is None:
        raise ModuleNotFoundError("LightGBM is required for the Kaggle exp504 train run")
    model = get_nested(config, "ranking.model")
    return LGBMClassifier(
        objective=str(model["objective"]),
        metric=str(model["metric"]),
        boosting_type=str(model["boosting_type"]),
        learning_rate=float(model["learning_rate"]),
        n_estimators=int(model["n_estimators"]),
        num_leaves=int(model["num_leaves"]),
        min_child_samples=int(model["min_child_samples"]),
        max_depth=int(model["max_depth"]),
        subsample=float(model["subsample"]),
        subsample_freq=int(model["subsample_freq"]),
        colsample_bytree=float(model["colsample_bytree"]),
        reg_alpha=float(model["reg_alpha"]),
        reg_lambda=float(model["reg_lambda"]),
        max_bin=int(model["max_bin"]),
        random_state=int(model["random_state"]),
        deterministic=bool(model["deterministic"]),
        force_col_wise=bool(model["force_col_wise"]),
        device=str(model["device"]),
        n_jobs=int(model["n_jobs"]),
        verbosity=-1,
    )


def select_from_pair_probabilities(
    pair_probability: np.ndarray,
    *,
    anchor_index: int = ANCHOR_INDEX,
    tie_tolerance: float = 1.0e-12,
    guard_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(pair_probability, dtype=np.float64)
    n_blocks = probabilities.shape[0]
    n_candidates = len(EXPECTED_CANDIDATE_ORDER)
    if probabilities.shape != (n_blocks, len(PAIR_LEFT)):
        raise ValueError("pair probability shape changed")
    win = np.full((n_blocks, n_candidates, n_candidates), np.nan, dtype=np.float64)
    rows = np.arange(n_blocks)
    for pair_index, (left, right) in enumerate(zip(PAIR_LEFT, PAIR_RIGHT, strict=True)):
        value = probabilities[:, pair_index]
        win[rows, left, right] = value
        win[rows, right, left] = 1.0 - value
    diagonal = np.arange(n_candidates)
    win[:, diagonal, diagonal] = 0.5
    borda = (win.sum(axis=2) - 0.5) / float(n_candidates - 1)
    provisional = np.empty(n_blocks, dtype=np.int8)
    selected = np.empty(n_blocks, dtype=np.int8)
    fallback = np.zeros(n_blocks, dtype=bool)
    for block in range(n_blocks):
        maximum = float(np.max(borda[block]))
        tied = np.flatnonzero(np.abs(borda[block] - maximum) <= tie_tolerance)
        winner = int(anchor_index if anchor_index in tied else tied[0])
        provisional[block] = winner
        if winner != anchor_index and not (win[block, winner, anchor_index] > guard_threshold):
            selected[block] = anchor_index
            fallback[block] = True
        else:
            selected[block] = winner
    return selected, provisional, fallback, borda.astype(np.float32)


def predict_valid_blocks(
    model: Any,
    surface: TargetFreeSurface,
    valid_blocks: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    batch_size = int(get_nested(config, "runtime.pair_prediction_block_batch_size"))
    canonical_probability = np.full((len(valid_blocks), len(PAIR_LEFT)), np.nan, dtype=np.float32)
    for batch_start in range(0, len(valid_blocks), batch_size):
        batch = np.asarray(valid_blocks[batch_start : batch_start + batch_size], dtype=np.int32)
        repeated_blocks = np.repeat(batch, len(PAIR_LEFT))
        left = np.tile(PAIR_LEFT.astype(np.int8), len(batch))
        right = np.tile(PAIR_RIGHT.astype(np.int8), len(batch))
        forward = assemble_pair_features(
            surface.candidate_features,
            surface.shared_features,
            surface.block_context,
            repeated_blocks,
            left,
            right,
        )
        reverse = assemble_pair_features(
            surface.candidate_features,
            surface.shared_features,
            surface.block_context,
            repeated_blocks,
            right,
            left,
        )
        q_forward = model.predict_proba(forward)[:, 1]
        q_reverse = model.predict_proba(reverse)[:, 1]
        antisymmetric = 0.5 * (q_forward + 1.0 - q_reverse)
        canonical_probability[batch_start : batch_start + len(batch)] = antisymmetric.reshape(
            len(batch), len(PAIR_LEFT)
        )
    selected, provisional, fallback, borda = select_from_pair_probabilities(
        canonical_probability,
        anchor_index=ANCHOR_INDEX,
        tie_tolerance=float(get_nested(config, "ranking.inference.tie_tolerance")),
        guard_threshold=float(get_nested(config, "ranking.inference.guard_probability_threshold")),
    )
    return {
        "pair_probability": canonical_probability,
        "selected": selected,
        "provisional": provisional,
        "anchor_guard_fallback": fallback,
        "borda": borda,
    }


# %% [markdown]
# ## 7. Truth-late fold evaluation and promotion gates


# %%
@dataclass
class TruthAccessLedger:
    target_free_frozen: bool = False
    prediction_frozen_folds: set[int] = field(default_factory=set)
    records: list[dict[str, Any]] = field(default_factory=list)

    def freeze_target_free(self) -> None:
        self.target_free_frozen = True

    def authorize_outer_train(self, valid_fold: int, folds: Sequence[int], rows: int) -> None:
        if not self.target_free_frozen:
            raise RuntimeError("truth requested before target-free feature freeze")
        if valid_fold in set(int(value) for value in folds):
            raise RuntimeError("outer-valid truth requested for pair training")
        self.records.append(
            {
                "valid_fold": valid_fold,
                "role": "outer_train_label",
                "folds": list(folds),
                "rows": rows,
            }
        )

    def freeze_prediction(self, valid_fold: int, prediction_sha256: str) -> None:
        self.prediction_frozen_folds.add(int(valid_fold))
        self.records.append(
            {
                "valid_fold": valid_fold,
                "role": "outer_valid_prediction_freeze",
                "prediction_sha256": prediction_sha256,
            }
        )

    def authorize_outer_valid(self, valid_fold: int, rows: int) -> None:
        if valid_fold not in self.prediction_frozen_folds:
            raise RuntimeError("outer-valid truth requested before prediction freeze")
        self.records.append(
            {
                "valid_fold": valid_fold,
                "role": "outer_valid_readout",
                "folds": [valid_fold],
                "rows": rows,
            }
        )


def load_truth_for_folds(
    block_assignment: pd.DataFrame,
    raw_dir: Path,
    folds: Sequence[int],
) -> np.ndarray:
    allowed = set(int(value) for value in folds)
    truth = np.full(len(block_assignment), np.nan, dtype=np.float32)
    subset = block_assignment[block_assignment["outer_fold"].isin(allowed)]
    for well, group in subset.groupby("well", sort=True):
        raw = pd.read_csv(_raw_horizontal_path(raw_dir, str(well)), usecols=["TVT"])
        indices = group["well_row_idx"].to_numpy(np.int64)
        values = pd.to_numeric(raw.iloc[indices]["TVT"], errors="raise").to_numpy(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"truth contains non-finite suffix values: {well}")
        truth[group.index.to_numpy(np.int64)] = values
    expected_mask = block_assignment["outer_fold"].isin(allowed).to_numpy()
    if not np.isfinite(truth[expected_mask]).all() or np.isfinite(truth[~expected_mask]).any():
        raise ValueError("truth fold projection violated the requested boundary")
    return truth


def compute_block_mse(
    surface: TargetFreeSurface,
    truth: np.ndarray,
    block_ids: np.ndarray,
) -> np.ndarray:
    output = np.full((len(surface.blocks), len(EXPECTED_CANDIDATE_ORDER)), np.nan, dtype=np.float64)
    for block_id in np.asarray(block_ids, dtype=np.int32):
        row = surface.blocks.iloc[int(block_id)]
        start = int(row["row_start"])
        stop = int(row["row_stop_exclusive"])
        actual = truth[start:stop]
        if not np.isfinite(actual).all():
            raise ValueError(f"truth is missing for H512 block {block_id}")
        error = surface.candidate_values[start:stop].astype(np.float64) - actual[:, None]
        output[block_id] = np.mean(np.square(error), axis=0)
    return output


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(prediction - actual))))


def evaluate_rank_readout(
    block_mse: np.ndarray,
    block_ids: np.ndarray,
    prediction: Mapping[str, np.ndarray],
    row_count: np.ndarray,
    tie_tolerance: float,
) -> dict[str, Any]:
    block_ids = np.asarray(block_ids, dtype=np.int32)
    selected = np.asarray(prediction["selected"], dtype=np.int8)
    borda = np.asarray(prediction["borda"], dtype=np.float64)
    pair_probability = np.asarray(prediction["pair_probability"], dtype=np.float64)
    true_mse = block_mse[block_ids]
    oracle = np.argmin(true_mse, axis=1)
    top1_accuracy = float(np.mean(selected == oracle))
    ranks = np.argsort(np.argsort(true_mse, axis=1), axis=1)
    relevance = len(EXPECTED_CANDIDATE_ORDER) - ranks
    ndcg_at_1 = float(
        np.mean(
            relevance[np.arange(len(selected)), selected] / float(len(EXPECTED_CANDIDATE_ORDER))
        )
    )
    top3_hit = []
    for index in range(len(block_ids)):
        order = np.lexsort((np.arange(len(EXPECTED_CANDIDATE_ORDER)), -borda[index]))
        top3_hit.append(int(oracle[index]) in set(order[:3].tolist()))

    left_mse = true_mse[:, PAIR_LEFT]
    right_mse = true_mse[:, PAIR_RIGHT]
    delta = left_mse - right_mse
    keep = np.abs(delta) > tie_tolerance
    labels = delta < 0.0
    pair_choice = pair_probability > 0.5
    correct = pair_choice == labels
    raw_weight = row_count[block_ids, None] * np.log1p(np.abs(delta))
    return {
        "h512_top1_exact_accuracy": top1_accuracy,
        "weighted_pair_accuracy": float(
            np.sum(raw_weight[keep] * correct[keep]) / np.sum(raw_weight[keep])
        ),
        "unweighted_pair_accuracy": float(np.mean(correct[keep])),
        "ndcg_at_1": ndcg_at_1,
        "top3_oracle_coverage": float(np.mean(top3_hit)),
        "anchor_guard_fallback_blocks": int(np.sum(prediction["anchor_guard_fallback"])),
        "anchor_guard_fallback_rate": float(np.mean(prediction["anchor_guard_fallback"])),
        "selected_anchor_blocks": int(np.sum(selected == ANCHOR_INDEX)),
        "selected_anchor_rate": float(np.mean(selected == ANCHOR_INDEX)),
    }


def load_hidden_like_sets(
    config: Mapping[str, Any], wells: set[str]
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like_assignment")
    path = resolve_file(
        spec["patterns"],
        label="hidden-like assignment",
        expected_sha256=str(spec["expected_file_sha256"]),
    )
    frame = pd.read_csv(path)
    output: dict[str, set[str]] = {}
    for scope, column in spec["roles"].items():
        selected = set(
            frame.loc[frame[str(column)].eq("valid"), str(spec["well_column"])].astype(str)
        )
        if selected - wells:
            raise ValueError(f"hidden-like assignment has unknown wells for {scope}")
        output[str(scope)] = selected
    return output, {"path": str(path), "file_sha256": sha256_file(path), "rows": len(frame)}


def build_scope_metrics(frame: pd.DataFrame, hidden_sets: Mapping[str, set[str]]) -> pd.DataFrame:
    masks = {
        "pooled": np.ones(len(frame), dtype=bool),
        "md_since_0_250": frame["md_since"].to_numpy(np.float64) < 250.0,
        "md_since_250_1000": frame["md_since"].to_numpy(np.float64) >= 250.0,
        "md_since_1000_plus": frame["md_since"].to_numpy(np.float64) >= 1000.0,
        "hidden_like_spatial": frame["well"].isin(hidden_sets["hidden_like_spatial"]).to_numpy(),
        "hidden_like_typewell_purged": frame["well"]
        .isin(hidden_sets["hidden_like_typewell_purged"])
        .to_numpy(),
    }
    masks["md_since_250_1000"] &= frame["md_since"].to_numpy(np.float64) < 1000.0
    records: list[dict[str, Any]] = []
    for scope, mask in masks.items():
        actual = frame.loc[mask, "true_tvt_readout_only"].to_numpy(np.float64)
        selected = frame.loc[mask, "selected_tvt"].to_numpy(np.float64)
        anchor = frame.loc[mask, "anchor_tvt"].to_numpy(np.float64)
        selected_rmse = rmse(actual, selected)
        anchor_rmse = rmse(actual, anchor)
        records.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "wells": int(frame.loc[mask, "well"].nunique()),
                "selected_rmse_ft": selected_rmse,
                "anchor_rmse_ft": anchor_rmse,
                "delta_vs_anchor_ft": selected_rmse - anchor_rmse,
            }
        )
    return pd.DataFrame(records)


def build_by_well_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=True):
        actual = group["true_tvt_readout_only"].to_numpy(np.float64)
        selected_rmse = rmse(actual, group["selected_tvt"].to_numpy(np.float64))
        anchor_rmse = rmse(actual, group["anchor_tvt"].to_numpy(np.float64))
        records.append(
            {
                "well": str(well),
                "outer_fold": int(group["outer_fold"].iloc[0]),
                "rows": len(group),
                "selected_rmse_ft": selected_rmse,
                "anchor_rmse_ft": anchor_rmse,
                "delta_vs_anchor_ft": selected_rmse - anchor_rmse,
            }
        )
    return pd.DataFrame(records)


def count_inter_block_switches(block_selection: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for well, group in block_selection.groupby("well", sort=True):
        ordered = group.sort_values("h512_group", kind="stable")
        selected = ordered["selected_candidate_index"].to_numpy(np.int8)
        records.append(
            {
                "well": str(well),
                "blocks": len(ordered),
                "inter_block_switch_count": int(np.sum(selected[1:] != selected[:-1])),
            }
        )
    return pd.DataFrame(records)


def evaluate_promotion_gates(
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    technical: Mapping[str, bool],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = get_nested(config, "promotion_gate")
    pooled = scope_metrics.set_index("scope").loc["pooled"]
    fixed_scopes = scope_metrics[scope_metrics["scope"].ne("pooled")]
    deltas = by_well["delta_vs_anchor_ft"].to_numpy(np.float64)
    checks = {
        "technical_all_passed": bool(all(technical.values())),
        "pooled_gain_at_least_0p05_ft": bool(
            -float(pooled["delta_vs_anchor_ft"]) >= float(gate["pooled_gain_vs_anchor_min_ft"])
        ),
        "pooled_rmse_within_fixed_max": bool(
            float(pooled["selected_rmse_ft"]) <= float(gate["pooled_rmse_max"])
        ),
        "nonworse_folds_at_least_4_of_5": bool(
            int(
                (
                    fold_metrics["delta_vs_anchor_ft"] <= float(gate["fold_nonworse_tolerance_ft"])
                ).sum()
            )
            >= int(gate["nonworse_fold_min"])
        ),
        "all_fixed_scopes_within_0p02_ft": bool(
            (fixed_scopes["delta_vs_anchor_ft"] <= float(gate["every_scope_delta_max_ft"])).all()
        ),
        "by_well_delta_p95_within_0p25_ft": bool(
            float(np.quantile(deltas, 0.95)) <= float(gate["by_well_delta_p95_max_ft"])
        ),
        "worst_well_delta_within_0p25_ft": bool(
            float(np.max(deltas)) <= float(gate["worst_well_delta_max_ft"])
        ),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": (
            "PASS_REQUIRE_NEW_APPROVAL_BEFORE_DOWNSTREAM_OR_INFERENCE"
            if all(checks.values())
            else "FAIL_TERMINAL_CLOSE_WITHOUT_HORIZON_LOSS_WEIGHT_OR_THRESHOLD_RESCUE"
        ),
    }


# %% [markdown]
# ## 8. Metrics, feature importance, and generated artifacts


# %%
def _model_booster(model: Any) -> Any:
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise RuntimeError("trained LightGBM booster is unavailable")
    return booster


def save_feature_importance(
    importance_rows: list[pd.DataFrame], output: Path
) -> tuple[pd.DataFrame, str | None]:
    importance = pd.concat(importance_rows, ignore_index=True)
    importance.to_csv(output / f"{OUTPUT_PREFIX}_feature_importance_by_fold.csv", index=False)
    mean = (
        importance.groupby("feature", as_index=False)["gain"]
        .mean()
        .sort_values("gain", ascending=False, kind="stable")
    )
    plot_sha: str | None = None
    if plt is not None and len(mean):
        plot_path = output / f"{OUTPUT_PREFIX}_feature_importance_top30.png"
        ax = (
            mean.head(30)
            .sort_values("gain")
            .plot.barh(
                x="feature",
                y="gain",
                figsize=(10, 11),
                legend=False,
                title="exp504 mean pair-rank gain importance",
            )
        )
        ax.set_xlabel("mean gain")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=140)
        plt.close()
        plot_sha = sha256_file(plot_path)
    return mean, plot_sha


# %% [markdown]
# ## 9. Setup and execution orchestration


# %%
def run_train(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config())
    validate_immutable_config(config)
    started = time.perf_counter()
    output = artifacts_dir()
    scratch = work_dir()
    block_assignment, block_evidence = load_block_assignment(config)
    surface = build_target_free_surface(config)
    ledger = TruthAccessLedger()
    ledger.freeze_target_free()
    raw_dir = resolve_raw_train_dir(config)
    n_rows = len(block_assignment)
    n_blocks = len(surface.blocks)
    row_count = surface.blocks["row_count"].to_numpy(np.int32)
    row_block = block_assignment["h512_group"].to_numpy(np.int32)
    block_fold = surface.blocks["outer_fold"].to_numpy(np.int8)
    tie_tolerance = float(get_nested(config, "ranking.target.tie_tolerance_mse"))

    oof_prediction = np.full(n_rows, np.nan, dtype=np.float32)
    oof_truth = np.full(n_rows, np.nan, dtype=np.float32)
    selected_by_block = np.full(n_blocks, -1, dtype=np.int8)
    provisional_by_block = np.full(n_blocks, -1, dtype=np.int8)
    fallback_by_block = np.zeros(n_blocks, dtype=bool)
    borda_by_block = np.full((n_blocks, len(EXPECTED_CANDIDATE_ORDER)), np.nan, dtype=np.float32)
    pair_probability = np.full((n_blocks, len(PAIR_LEFT)), np.nan, dtype=np.float32)
    fold_metric_rows: list[dict[str, Any]] = []
    rank_metric_rows: list[dict[str, Any]] = []
    model_manifest: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    prediction_freezes: list[dict[str, Any]] = []
    pair_table_hashes: list[dict[str, Any]] = []

    all_folds = list(range(int(get_nested(config, "validation.outer_folds"))))
    for valid_fold in all_folds:
        train_folds = [fold for fold in all_folds if fold != valid_fold]
        train_row_mask = block_assignment["outer_fold"].isin(train_folds).to_numpy()
        ledger.authorize_outer_train(valid_fold, train_folds, int(train_row_mask.sum()))
        train_truth = load_truth_for_folds(block_assignment, raw_dir, train_folds)
        train_blocks = np.flatnonzero(block_fold != valid_fold).astype(np.int32)
        valid_blocks = np.flatnonzero(block_fold == valid_fold).astype(np.int32)
        train_mse = compute_block_mse(surface, train_truth, train_blocks)
        targets = build_pair_targets(
            train_mse, row_count, train_blocks, tie_tolerance=tie_tolerance
        )
        pair_table_hashes.append(
            {
                "outer_valid_fold": valid_fold,
                "unordered_pairs_after_ties": len(targets.canonical_table),
                "ordered_examples": len(targets.block_ids),
                "logical_sha256": frame_content_sha256(targets.canonical_table),
                "normalized_unordered_weight_mean": float(
                    targets.canonical_table["normalized_weight"].mean()
                ),
                "ordered_weight_sum": float(targets.sample_weight.sum()),
            }
        )
        matrix_path = scratch / f"pair_features_outer_valid_fold{valid_fold}.f32"
        train_matrix = pair_feature_memmap(surface, targets, matrix_path)
        model = make_rank_model(config)
        print(
            f"Outer fold {valid_fold}: fitting {len(targets.block_ids):,} ordered pairs x "
            f"{len(surface.pair_feature_names):,} features",
            flush=True,
        )
        model.fit(
            train_matrix,
            targets.label,
            sample_weight=targets.sample_weight,
            feature_name=surface.pair_feature_names,
        )
        booster = _model_booster(model)
        model_path = output / f"{OUTPUT_PREFIX}_outer_fold{valid_fold}.txt"
        booster.save_model(str(model_path))
        model_sha = sha256_file(model_path)
        model_manifest.append(
            {
                "outer_valid_fold": valid_fold,
                "model_path": str(model_path),
                "model_sha256": model_sha,
                "trees": int(booster.num_trees()),
                "pair_feature_count": len(surface.pair_feature_names),
                "ordered_train_examples": len(targets.block_ids),
            }
        )
        importance_rows.append(
            pd.DataFrame(
                {
                    "outer_valid_fold": valid_fold,
                    "feature": surface.pair_feature_names,
                    "gain": booster.feature_importance(importance_type="gain"),
                    "split": booster.feature_importance(importance_type="split"),
                }
            )
        )

        valid_prediction = predict_valid_blocks(model, surface, valid_blocks, config)
        selected_by_block[valid_blocks] = valid_prediction["selected"]
        provisional_by_block[valid_blocks] = valid_prediction["provisional"]
        fallback_by_block[valid_blocks] = valid_prediction["anchor_guard_fallback"]
        borda_by_block[valid_blocks] = valid_prediction["borda"]
        pair_probability[valid_blocks] = valid_prediction["pair_probability"]
        valid_row_mask = block_assignment["outer_fold"].eq(valid_fold).to_numpy()
        valid_rows = np.flatnonzero(valid_row_mask)
        selected_index_rows = selected_by_block[row_block[valid_rows]]
        valid_row_prediction = surface.candidate_values[
            valid_rows, selected_index_rows.astype(np.int64)
        ].astype(np.float32)
        oof_prediction[valid_rows] = valid_row_prediction
        freeze_sha = keyed_prediction_sha256(
            block_assignment.iloc[valid_rows]["id"].astype(str).tolist(), valid_row_prediction
        )
        fold_block_frame = surface.blocks.iloc[valid_blocks][
            ["h512_group", "well", "outer_fold", "row_count"]
        ].copy()
        fold_block_frame["provisional_candidate_index"] = valid_prediction["provisional"]
        fold_block_frame["selected_candidate_index"] = valid_prediction["selected"]
        fold_block_frame["provisional_candidate_id"] = [
            EXPECTED_CANDIDATE_ORDER[index] for index in valid_prediction["provisional"]
        ]
        fold_block_frame["selected_candidate_id"] = [
            EXPECTED_CANDIDATE_ORDER[index] for index in valid_prediction["selected"]
        ]
        fold_block_frame["anchor_guard_fallback"] = valid_prediction["anchor_guard_fallback"]
        for index, candidate_id in enumerate(EXPECTED_CANDIDATE_ORDER):
            fold_block_frame[f"borda__{candidate_id}"] = valid_prediction["borda"][:, index]
        freeze_path = output / f"{OUTPUT_PREFIX}_outer_fold{valid_fold}_prediction_freeze.csv.gz"
        freeze_evidence = deterministic_gzip_csv(fold_block_frame, freeze_path)
        freeze_evidence["row_prediction_content_sha256"] = freeze_sha
        prediction_freezes.append(freeze_evidence)
        ledger.freeze_prediction(valid_fold, freeze_sha)

        ledger.authorize_outer_valid(valid_fold, int(valid_row_mask.sum()))
        valid_truth = load_truth_for_folds(block_assignment, raw_dir, [valid_fold])
        oof_truth[valid_rows] = valid_truth[valid_rows]
        valid_mse = compute_block_mse(surface, valid_truth, valid_blocks)
        selected_rmse = rmse(valid_truth[valid_rows], valid_row_prediction)
        anchor_prediction = surface.candidate_values[valid_rows, ANCHOR_INDEX]
        anchor_rmse = rmse(valid_truth[valid_rows], anchor_prediction)
        fold_metric_rows.append(
            {
                "outer_fold": valid_fold,
                "rows": len(valid_rows),
                "wells": int(block_assignment.loc[valid_row_mask, "well"].nunique()),
                "blocks": len(valid_blocks),
                "selected_rmse_ft": selected_rmse,
                "anchor_rmse_ft": anchor_rmse,
                "delta_vs_anchor_ft": selected_rmse - anchor_rmse,
                "prediction_sha256": freeze_sha,
                "model_sha256": model_sha,
            }
        )
        rank_row = evaluate_rank_readout(
            valid_mse,
            valid_blocks,
            valid_prediction,
            row_count,
            tie_tolerance,
        )
        rank_row["outer_fold"] = valid_fold
        rank_row["blocks"] = len(valid_blocks)
        rank_metric_rows.append(rank_row)

        del train_matrix
        try:
            matrix_path.unlink()
        except FileNotFoundError:
            pass
        del model, booster, train_truth, train_mse, valid_truth, valid_mse, targets
        gc.collect()

    if not np.isfinite(oof_prediction).all() or not np.isfinite(oof_truth).all():
        raise ValueError("outer-valid OOF coverage is incomplete")
    if (
        np.any(selected_by_block < 0)
        or np.isnan(pair_probability).any()
        or np.isnan(borda_by_block).any()
    ):
        raise ValueError("block selection coverage is incomplete")

    selected_candidate_rows = selected_by_block[row_block]
    anchor_prediction = np.asarray(surface.candidate_values[:, ANCHOR_INDEX], dtype=np.float32)
    oof = block_assignment[
        ["id", "well", "well_row_idx", "outer_fold", "md_since", "h512_group"]
    ].copy()
    oof["selected_candidate_index"] = selected_candidate_rows
    oof["selected_candidate_id"] = [
        EXPECTED_CANDIDATE_ORDER[index] for index in selected_candidate_rows
    ]
    oof["selected_tvt"] = oof_prediction
    oof["anchor_tvt"] = anchor_prediction
    oof["true_tvt_readout_only"] = oof_truth
    oof_path = output / f"{OUTPUT_PREFIX}_oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)

    block_selection = surface.blocks[["h512_group", "well", "outer_fold", "row_count"]].copy()
    block_selection["provisional_candidate_index"] = provisional_by_block
    block_selection["selected_candidate_index"] = selected_by_block
    block_selection["provisional_candidate_id"] = [
        EXPECTED_CANDIDATE_ORDER[index] for index in provisional_by_block
    ]
    block_selection["selected_candidate_id"] = [
        EXPECTED_CANDIDATE_ORDER[index] for index in selected_by_block
    ]
    block_selection["anchor_guard_fallback"] = fallback_by_block
    for index, candidate_id in enumerate(EXPECTED_CANDIDATE_ORDER):
        block_selection[f"borda__{candidate_id}"] = borda_by_block[:, index]
    block_selection.to_parquet(output / f"{OUTPUT_PREFIX}_block_selection.parquet", index=False)
    pair_probability_frame = pd.DataFrame(
        pair_probability,
        columns=[
            f"p__{EXPECTED_CANDIDATE_ORDER[left]}__beats__{EXPECTED_CANDIDATE_ORDER[right]}"
            for left, right in zip(PAIR_LEFT, PAIR_RIGHT, strict=True)
        ],
    )
    pair_probability_frame.insert(0, "h512_group", np.arange(n_blocks, dtype=np.int32))
    pair_probability_frame.to_parquet(
        output / f"{OUTPUT_PREFIX}_pair_probability_oof.parquet", index=False
    )

    hidden_sets, hidden_evidence = load_hidden_like_sets(config, set(oof["well"].astype(str)))
    scope_metrics = build_scope_metrics(oof, hidden_sets)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    rank_metrics = pd.DataFrame(rank_metric_rows)
    by_well = build_by_well_metrics(oof)
    switches = count_inter_block_switches(block_selection)
    scope_metrics.to_csv(output / f"{OUTPUT_PREFIX}_scope_metrics.csv", index=False)
    fold_metrics.to_csv(output / f"{OUTPUT_PREFIX}_fold_metrics.csv", index=False)
    rank_metrics.to_csv(output / f"{OUTPUT_PREFIX}_rank_metrics.csv", index=False)
    by_well.to_csv(output / f"{OUTPUT_PREFIX}_by_well.csv", index=False)
    switches.to_csv(output / f"{OUTPUT_PREFIX}_inter_block_switches.csv", index=False)
    importance_mean, importance_plot_sha = save_feature_importance(importance_rows, output)

    pair_contract = {
        "candidate_order": list(EXPECTED_CANDIDATE_ORDER),
        "canonical_pair_left": PAIR_LEFT.tolist(),
        "canonical_pair_right": PAIR_RIGHT.tolist(),
        "tie_tolerance_mse": tie_tolerance,
        "raw_weight_formula": "row_count * log1p(abs(mse_left - mse_right))",
        "fold_normalization": "unordered raw weight mean equals one",
        "orientation_weight": "half normalized unordered weight per direction",
        "pair_feature_schema_sha256": json_sha256(surface.pair_feature_names),
        "fold_pair_tables": pair_table_hashes,
    }
    pair_contract_sha = write_json(output / f"{OUTPUT_PREFIX}_pair_contract.json", pair_contract)
    model_manifest_payload = {
        "model_count": len(model_manifest),
        "models": model_manifest,
        "model_manifest_sha256": json_sha256(model_manifest),
    }
    model_manifest_sha = write_json(
        output / f"{OUTPUT_PREFIX}_model_manifest.json", model_manifest_payload
    )

    anchor_rmse = rmse(oof_truth, anchor_prediction)
    selected_rmse = rmse(oof_truth, oof_prediction)
    anchor_expected = float(get_nested(config, "validation.primary_control.oof_rmse"))
    technical = {
        "input_candidate_bank_sha_match": surface.candidate_bank_sha256
        == str(get_nested(config, "data.candidate_bank.content_sha256")),
        "input_candidate_key_sha_match": surface.candidate_key_sha256
        == str(get_nested(config, "data.candidate_bank.key_content_sha256")),
        "block_assignment_file_sha_match": block_evidence["file_sha256"]
        == str(get_nested(config, "data.block_assignment.file_sha256")),
        "block_assignment_decompressed_sha_match": block_evidence["decompressed_content_sha256"]
        == str(get_nested(config, "data.block_assignment.decompressed_content_sha256")),
        "row_feature_schema_sha_match": any(
            evidence.get("logical_sha256")
            == str(get_nested(config, "data.row_feature_schema.feature_schema_logical_sha256"))
            for evidence in surface.input_evidence
        ),
        "target_free_feature_freeze_exists": (
            output / f"{OUTPUT_PREFIX}_target_free_freeze.json"
        ).is_file(),
        "row_coverage_exact": len(oof) == int(get_nested(config, "technical_gate.expected_rows")),
        "well_coverage_exact": oof["well"].nunique()
        == int(get_nested(config, "technical_gate.expected_wells")),
        "block_coverage_exact": len(block_selection)
        == int(get_nested(config, "technical_gate.expected_h512_queries")),
        "candidate_block_count_exact": len(block_selection) * len(EXPECTED_CANDIDATE_ORDER)
        == int(get_nested(config, "technical_gate.expected_candidate_blocks")),
        "unique_row_keys": not oof["id"].duplicated().any(),
        "unique_block_keys": not block_selection["h512_group"].duplicated().any(),
        "outer_valid_prediction_frozen_before_truth": all(
            any(
                record.get("role") == "outer_valid_prediction_freeze"
                and int(record["valid_fold"]) == fold
                for record in ledger.records[:index]
            )
            for index, record in enumerate(ledger.records)
            if record.get("role") == "outer_valid_readout"
            for fold in [int(record["valid_fold"])]
        ),
        "five_models_saved": len(model_manifest)
        == int(get_nested(config, "execution_contract.total_cpu_models")),
        "all_model_sha_recorded": all(len(item["model_sha256"]) == 64 for item in model_manifest),
        "pair_feature_schema_recorded": len(surface.feature_schema_sha256) == 64,
        "oof_prediction_sha_recorded": all(
            len(item["prediction_sha256"]) == 64 for item in fold_metric_rows
        ),
        "anchor_parity_within_1e_3_ft": abs(anchor_rmse - anchor_expected) <= 1.0e-3,
    }
    gates = evaluate_promotion_gates(scope_metrics, fold_metrics, by_well, technical, config)
    deltas = by_well["delta_vs_anchor_ft"].to_numpy(np.float64)
    choice_counts = (
        block_selection["selected_candidate_id"]
        .value_counts()
        .reindex(EXPECTED_CANDIDATE_ORDER, fill_value=0)
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed" if all(technical.values()) else "technical_failed",
        "route": get_nested(config, "experiment.route"),
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "accelerator": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        },
        "inputs": {
            "rows": len(oof),
            "wells": int(oof["well"].nunique()),
            "h512_queries": len(block_selection),
            "candidate_bank_sha256": surface.candidate_bank_sha256,
            "candidate_key_sha256": surface.candidate_key_sha256,
            "block_feature_schema_sha256": surface.feature_schema_sha256,
            "block_feature_content_sha256": surface.feature_content_sha256,
            "hidden_like_assignment": hidden_evidence,
        },
        "headline": {
            "selected_oof_rmse_ft": selected_rmse,
            "anchor_oof_rmse_ft": anchor_rmse,
            "delta_vs_anchor_ft": selected_rmse - anchor_rmse,
            "nonworse_folds": int((fold_metrics["delta_vs_anchor_ft"] <= 0.0).sum()),
            "by_well_improved": int((deltas < 0.0).sum()),
            "by_well_worsened": int((deltas > 0.0).sum()),
            "by_well_delta_p50_ft": float(np.quantile(deltas, 0.50)),
            "by_well_delta_p90_ft": float(np.quantile(deltas, 0.90)),
            "by_well_delta_p95_ft": float(np.quantile(deltas, 0.95)),
            "worst_well_delta_ft": float(np.max(deltas)),
            "worst_well": str(by_well.loc[by_well["delta_vs_anchor_ft"].idxmax(), "well"]),
            "anchor_guard_fallback_blocks": int(fallback_by_block.sum()),
            "anchor_guard_fallback_rate": float(fallback_by_block.mean()),
            "inter_block_switch_count": int(switches["inter_block_switch_count"].sum()),
        },
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "scope_metrics": scope_metrics.to_dict(orient="records"),
        "rank_metrics": rank_metrics.to_dict(orient="records"),
        "candidate_choice_count": {str(key): int(value) for key, value in choice_counts.items()},
        "feature_importance_top30": importance_mean.head(30).to_dict(orient="records"),
        "technical_gate": technical,
        "technical_passed": bool(all(technical.values())),
        "promotion_gate": gates,
        "truth_access_ledger": ledger.records,
        "prediction_freezes": prediction_freezes,
        "reproducibility": {
            "pair_contract_file_sha256": pair_contract_sha,
            "model_manifest_file_sha256": model_manifest_sha,
            "model_manifest_logical_sha256": model_manifest_payload["model_manifest_sha256"],
            "oof_prediction_file_sha256": sha256_file(oof_path),
            "oof_prediction_content_sha256": keyed_prediction_sha256(
                oof["id"].astype(str).tolist(), oof_prediction
            ),
            "feature_importance_plot_sha256": importance_plot_sha,
            "config_file_sha256": sha256_file(config_path()),
        },
        "execution_actual": {
            "scientific_variants": 1,
            "rank_configs": 1,
            "outer_folds": 5,
            "cpu_models": len(model_manifest),
            "boosters": len(model_manifest),
            "parent_control_retrains": 0,
            "candidate_regeneration_runs": 0,
            "pf_runs": 0,
            "hmm_runs": 0,
            "beam_runs": 0,
            "gpu_models": 0,
            "inference_runs": 0,
            "submission_files": 0,
        },
    }
    summary_path = output / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "route": "ensemble",
        "cv": summary["headline"],
        "gates": {"technical": technical, "promotion": gates},
        "sha256": summary["reproducibility"],
        "kaggle": {"kernel_id": None, "version": None, "url": None},
        "public_lb": None,
        "private_lb": None,
        "submission": None,
    }
    write_json(runtime_metrics_path(), metrics)
    print("\nFold metrics\n", fold_metrics.to_string(index=False), flush=True)
    print("\nScope metrics\n", scope_metrics.to_string(index=False), flush=True)
    print("\nRank metrics\n", rank_metrics.to_string(index=False), flush=True)
    print("\nHeadline", json.dumps(to_jsonable(summary["headline"]), sort_keys=True), flush=True)
    print("Promotion", json.dumps(to_jsonable(gates), sort_keys=True), flush=True)
    print("Artifacts", output, flush=True)
    return summary


# %%
if __name__ == "__main__":
    CONFIG = load_config()
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(CONFIG, "experiment.route"))
    print("Parent:", get_nested(CONFIG, "lineage.parent"))
    print("Candidate order:", list(EXPECTED_CANDIDATE_ORDER))
    print("Execution contract: 1 variant / 1 config / 5 outer folds / 5 CPU boosters")
    RUN_SUMMARY = run_train(CONFIG)

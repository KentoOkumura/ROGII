# %% [markdown]
# # exp394 soft-sticky exp226/K16 branch HMM — train
#
# This notebook implements the frozen exp394 contract.  It keeps one
# group-safe exp226 geometry state (`E`) beside the complete exp209
# absolute-TVT × residual-rate trellis (`H`) and marginalizes both branches in
# one MD-aware forward-backward pass.  Canonical version 1 completed the
# fixed16 preflight but failed its runtime projection, so `config.yaml`
# disables every run stage and full OOF remains fail closed.

# %% [markdown]
# ## Contents
# 1. Imports and frozen execution contract
# 2. Notebook-safe paths, SHA helpers, and leakage ledger
# 3. Target-free input and immutable dependency checks
# 4. Frozen exp355 K16 relative-rate schedule
# 5. Exp209 observation and state-grid preparation
# 6. Soft-sticky transition and exact forward-backward helpers
# 7. Per-well joint decoding and diagnostics
# 8. Fixed 16-well technical preflight
# 9. Prediction freeze and late scientific readout
# 10. Kaggle CPU orchestration and generated artifacts
# 11. Setup and configuration preview
# 12. Fail-closed post-preflight entry point

# %% [markdown]
# ## 1. Imports and frozen execution contract

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import numba
    from numba import njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - validation-only fallback
    numba = None
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args: Any, **kwargs: Any):
        del kwargs
        if args and callable(args[0]):
            return args[0]

        def decorator(function: Any) -> Any:
            return function

        return decorator

    def set_num_threads(value: int) -> None:
        del value


EXPERIMENT_NAME = "exp394_soft_sticky_exp226_k16_branch_hmm"
OUTPUT_PREFIX = EXPERIMENT_NAME
CANDIDATE_NAME = "soft_sticky_exp226_k16_full_grid_exact_hmm"
SAFE_GEOMETRY_COLUMNS = ["well_id", "row_idx", "suffix_offset", "fold", "tvt_geop"]
KEY_COLUMNS = ("well_id", "row_idx")
FIXED_FORMULA_WEIGHTS = {
    "exp226_k16": 0.50,
    "likpf_mean": 0.25,
    "exact_hmm": 0.25,
}
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
PACKAGE_DIR = Path.cwd()
EXECUTE_NOTEBOOK = os.environ.get("EXP394_IMPORT_ONLY") != "1"


def get_nested(
    config: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    run_stage: str | None = None,
) -> dict[str, Any]:
    expected = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "implementation.enabled": True,
        "implementation.compact_selfcontained_source_created": True,
        "implementation.canonical_notebook_adopted": True,
        "validation.n_folds": 5,
        "validation.expected_rows": 3_783_989,
        "validation.expected_wells": 773,
        "validation.truth_attachment": (
            "after_schedule_branch_posterior_prediction_and_content_sha_freeze"
        ),
        "model.initial_regime_probability.E_exp226_geometry": 0.5,
        "model.initial_regime_probability.H_free_exact_hmm": 0.5,
        "model.h_branch.retain_all_tvt_grid_states": True,
        "model.h_branch.retain_all_rate_states": True,
        "model.h_branch.transition_mean_schedule.k_segments": 16,
        "model.h_branch.fixed_hmm.step_ft": 0.35,
        "model.h_branch.fixed_hmm.n_rates": 41,
        "model.h_branch.fixed_hmm.residual_rate_span": 0.10,
        "model.h_branch.fixed_hmm.sig_r": 0.002,
        "model.h_branch.fixed_hmm.sig_p": 0.02,
        "model.h_branch.fixed_hmm.emission": "gaussian",
        "model.h_branch.fixed_hmm.sigma_mode": "std",
        "model.h_branch.fixed_hmm.start_sig": 0.75,
        "model.h_branch.fixed_hmm.r0_sig": 0.01,
        "model.h_branch.fixed_hmm.band_pad_ft": 100.0,
        "model.h_branch.fixed_hmm.momentum": 0.998,
        "model.soft_sticky.base_switching_length_md_ft": 1000.0,
        "model.soft_sticky.h_to_e_docking.sigma_ft": 6.0,
        "model.inference.algorithm": "exact_log_space_forward_backward",
        "execution.full_oof_counts.scientific_variants": 1,
        "execution.full_oof_counts.reporting_folds": 5,
        "execution.full_oof_counts.switching_hmm_well_runs": 773,
        "execution.full_oof_counts.lightgbm_configs": 0,
        "execution.full_oof_counts.trained_folds": 0,
        "execution.full_oof_counts.boosters": 0,
        "execution.full_oof_counts.parent_control_reruns": 0,
        "runtime.device": "cpu",
        "runtime.use_gpu": False,
        "runtime.num_workers": 1,
        "runtime.numba_num_threads": 2,
        "execution.run_inference": False,
        "execution.create_submission": False,
    }
    for dotted_key, expected_value in expected.items():
        actual = get_nested(config, dotted_key)
        if actual != expected_value:
            raise ValueError(
                f"frozen exp394 contract changed for {dotted_key}: {actual!r} != {expected_value!r}"
            )
    forbidden = set(get_nested(config, "model.forbidden", []))
    required_forbidden = {
        "low_rank_3d_geologic_field",
        "exp226_gr_delta_or_final_tvt_pred_or_u_projection",
        "manually_enumerated_hmm_modes",
        "marginal_map_viterbi_or_topk_mode_bank",
        "hard_router_or_rowwise_selector",
        "posthoc_blend_or_oracle_branch_choice",
        "parameter_grid_or_same_oof_rescue",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("exp394 forbidden-method contract changed")
    counts = {
        "scientific_variants": 1,
        "reporting_folds": 5,
        "switching_hmm_well_runs": 773,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": 0,
        "gpu": False,
    }
    if run_stage is not None:
        if run_stage not in {"technical_preflight", "full_oof"}:
            raise ValueError(f"unknown exp394 run stage: {run_stage}")
        if not bool(get_nested(config, "execution.kaggle_package_approved")):
            raise RuntimeError("exp394 Kaggle package is not approved")
        if not bool(get_nested(config, "execution.kaggle_push_approved")):
            raise RuntimeError("exp394 Kaggle push is not approved")
        if run_stage == "technical_preflight":
            if not bool(get_nested(config, "execution.technical_preflight_approved")):
                raise RuntimeError("exp394 technical preflight is not approved")
            if not bool(get_nested(config, "execution.run_technical_preflight")):
                raise RuntimeError("exp394 run_technical_preflight is disabled")
        else:
            if not bool(get_nested(config, "execution.full_oof_approved")):
                raise RuntimeError("exp394 full OOF is not separately approved")
            if not bool(get_nested(config, "execution.run_full_oof")):
                raise RuntimeError("exp394 run_full_oof is disabled")
            preflight_sha = get_nested(
                config,
                "validation.full_oof.required_preflight_summary_sha256",
            )
            if not isinstance(preflight_sha, str) or len(preflight_sha) != 64:
                raise RuntimeError("exp394 full OOF requires the frozen PASS preflight summary SHA")
    return counts


# %% [markdown]
# ## 2. Notebook-safe paths, SHA helpers, and leakage ledger


# %%
def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "AGENTS.md").is_file() and (candidate / "project.yml").is_file():
            return candidate
    return start


def config_path() -> Path:
    root = find_project_root()
    candidates = [
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp394 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    resolved = config_path() if path is None else path
    value = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a YAML mapping")
    return value


def output_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        root = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        root = find_project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def metrics_path() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return find_project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


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
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_csv(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_frame_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in selected:
        digest.update(str(column).encode())
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            array = np.ascontiguousarray(series.to_numpy())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        else:
            for value in series.astype(str):
                digest.update(value.encode())
                digest.update(b"\n")
    return digest.hexdigest()


def schema_sha256(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    return hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n")
    return {"path": str(path), "raw_sha256": sha256_file(path)}


def write_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_file(path),
        "content_sha256": logical_frame_sha256(frame),
        "schema_sha256": schema_sha256(frame),
    }


def write_gzip_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": sha256_decompressed_csv(path),
        "content_sha256": logical_frame_sha256(frame),
        "schema_sha256": schema_sha256(frame),
    }


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 if platform.system() == "Linux" else 1024.0**2
    return value / divisor / 1024.0


def runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": getattr(numba, "__version__", None),
        "numba_available": NUMBA_AVAILABLE,
    }


@dataclass
class RoleReadLedger:
    prediction_frozen: bool = False
    suffix_truth_rows_before_freeze: int = 0
    error_rows_before_freeze: int = 0
    hidden_role_rows_before_freeze: int = 0
    suffix_truth_rows_after_freeze: int = 0
    hidden_role_rows_after_freeze: int = 0
    reads: list[dict[str, Any]] = field(default_factory=list)

    def target_free(
        self,
        label: str,
        columns: Sequence[str],
        rows: int,
        forbidden: Sequence[str],
    ) -> None:
        overlap = sorted(set(columns).intersection(forbidden))
        if overlap:
            self.suffix_truth_rows_before_freeze += int(rows)
            raise ValueError(f"{label}: forbidden pre-freeze columns read: {overlap}")
        self.reads.append(
            {
                "label": label,
                "phase": "target_free",
                "columns": list(columns),
                "rows": int(rows),
            }
        )

    def freeze(self) -> None:
        if (
            self.suffix_truth_rows_before_freeze
            or self.error_rows_before_freeze
            or self.hidden_role_rows_before_freeze
        ):
            raise RuntimeError("truth/error/hidden-like data was read before prediction freeze")
        self.prediction_frozen = True

    def truth_late(self, label: str, rows: int) -> None:
        if not self.prediction_frozen:
            self.suffix_truth_rows_before_freeze += int(rows)
            raise RuntimeError(f"{label}: suffix truth cannot be read before freeze")
        self.suffix_truth_rows_after_freeze += int(rows)
        self.reads.append({"label": label, "phase": "truth_late", "rows": int(rows)})

    def hidden_late(self, label: str, rows: int) -> None:
        if not self.prediction_frozen:
            self.hidden_role_rows_before_freeze += int(rows)
            raise RuntimeError(f"{label}: hidden-like roles cannot be read before freeze")
        self.hidden_role_rows_after_freeze += int(rows)
        self.reads.append({"label": label, "phase": "hidden_late", "rows": int(rows)})


def resolve_unique_file(
    *,
    filename: str,
    configured_candidates: Iterable[str],
    patterns: Iterable[str],
    label: str,
) -> Path:
    root = find_project_root()
    candidates: list[Path] = []
    for raw in configured_candidates:
        path = Path(str(raw))
        if not path.is_absolute():
            path = root / path
        candidates.append(path if path.name == filename else path / filename)
    search_roots = [root, Path("/tmp")]
    if KAGGLE_INPUT_ROOT.is_dir():
        search_roots.insert(0, KAGGLE_INPUT_ROOT)
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for pattern in patterns:
            candidates.extend(search_root.glob(str(pattern)))
    matches = sorted(
        {str(path.resolve()): path for path in candidates if path.is_file()}.values(),
        key=str,
    )
    if not matches:
        raise FileNotFoundError(f"{label} was not found: {filename}")
    identities = {
        (sha256_decompressed_csv(path) if path.suffix == ".gz" else sha256_file(path))
        for path in matches
    }
    if len(identities) != 1:
        raise ValueError(f"multiple non-identical {label} files were found")
    return matches[0]


def train_data_dir(config: Mapping[str, Any]) -> Path:
    expected = int(get_nested(config, "validation.expected_wells"))
    local = find_project_root() / str(get_nested(config, "data.train_dir"))
    if local.is_dir() and len(list(local.glob("*__horizontal_well.csv"))) == expected:
        return local
    if KAGGLE_INPUT_ROOT.is_dir():
        for candidate in sorted(KAGGLE_INPUT_ROOT.rglob("train")):
            if len(list(candidate.glob("*__horizontal_well.csv"))) == expected:
                return candidate
    return local


# %% [markdown]
# ## 3. Target-free input and immutable dependency checks


# %%
def list_well_ids(raw_dir: Path) -> list[str]:
    wells = [
        path.name.removesuffix("__horizontal_well.csv")
        for path in sorted(raw_dir.glob("*__horizontal_well.csv"))
    ]
    for well in wells:
        if not (raw_dir / f"{well}__typewell.csv").is_file():
            raise FileNotFoundError(raw_dir / f"{well}__typewell.csv")
    return wells


def validate_raw_identity(
    config: Mapping[str, Any],
    raw_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for well in list_well_ids(raw_dir):
        horizontal = raw_dir / f"{well}__horizontal_well.csv"
        typewell = raw_dir / f"{well}__typewell.csv"
        rows.append(
            {
                "well_id": well,
                "horizontal_raw_sha256": sha256_file(horizontal),
                "typewell_raw_sha256": sha256_file(typewell),
            }
        )
    frame = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    content_sha = logical_frame_sha256(frame)
    if len(frame) != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("raw well count differs from exp394 contract")
    if content_sha != str(get_nested(config, "data.expected_raw_well_identity_sha256")):
        raise ValueError("raw well identity SHA differs from exp355/exp394 contract")
    return frame, {
        "path": str(raw_dir),
        "wells": len(frame),
        "content_sha256": content_sha,
    }


def load_exp226_geometry(
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = dict(get_nested(config, "data.exp226_geometry_oof"))
    path = resolve_unique_file(
        filename=str(spec["filename"]),
        configured_candidates=spec.get("candidates", []),
        patterns=spec.get("patterns", []),
        label="exp226 group-safe geometry OOF",
    )
    actual_sha = sha256_decompressed_csv(path)
    if actual_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp226 geometry decompressed SHA mismatch")
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(SAFE_GEOMETRY_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"exp226 geometry is missing columns: {missing}")
    frame = pd.read_csv(
        path,
        usecols=SAFE_GEOMETRY_COLUMNS,
        dtype={
            "well_id": str,
            "row_idx": "int32",
            "suffix_offset": "int32",
            "fold": "int8",
            "tvt_geop": "float64",
        },
    )
    ledger.target_free(
        "exp226_geometry",
        list(frame.columns),
        len(frame),
        spec.get("forbidden_columns", []),
    )
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("exp226 geometry contains duplicate well/row keys")
    if not np.isfinite(frame["tvt_geop"].to_numpy(np.float64)).all():
        raise ValueError("exp226 tvt_geop must be finite")
    if len(frame) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp226 geometry row count mismatch")
    if frame["well_id"].nunique() != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("exp226 geometry well count mismatch")
    if not frame.groupby("well_id", sort=False)["fold"].nunique().eq(1).all():
        raise ValueError("each exp226 well must belong to exactly one outer fold")
    return frame, {
        "path": str(path),
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "decompressed_sha256": actual_sha,
        "safe_content_sha256": logical_frame_sha256(frame),
        "safe_schema_sha256": schema_sha256(frame),
        "columns_loaded": list(frame.columns),
    }


def load_target_free_well(
    well: str,
    raw_dir: Path,
    ledger: RoleReadLedger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
    typewell_path = raw_dir / f"{well}__typewell.csv"
    horizontal = pd.read_csv(
        horizontal_path,
        usecols=["MD", "Z", "GR", "TVT_input"],
    )
    ledger.target_free(
        f"raw_horizontal:{well}",
        list(horizontal.columns),
        len(horizontal),
        ("TVT", "target", "truth", "error", "abs_error"),
    )
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    ledger.target_free(
        f"typewell:{well}",
        ["typewell_TVT", "typewell_GR"],
        len(typewell),
        ("target", "truth", "error", "abs_error"),
    )
    return horizontal, typewell


# %% [markdown]
# ## 4. Frozen exp355 K16 relative-rate schedule


# %%
def exp209_initial_rate(horizontal: pd.DataFrame, tail_n: int = 30) -> dict[str, Any]:
    known = horizontal.loc[horizontal["TVT_input"].notna(), ["MD", "Z", "TVT_input"]]
    tail = known.tail(tail_n)
    delta_md = np.diff(tail["MD"].to_numpy(np.float64))
    delta_u = np.diff(tail["TVT_input"].to_numpy(np.float64) + tail["Z"].to_numpy(np.float64))
    valid = np.isfinite(delta_md) & np.isfinite(delta_u) & (delta_md > 0.0)
    value = float(np.median(delta_u[valid] / delta_md[valid])) if valid.sum() >= 3 else 0.0
    return {
        "initial_rate": value,
        "known_rows": len(known),
        "tail_rows": len(tail),
        "valid_steps": int(valid.sum()),
        "fallback": bool(valid.sum() < 3),
    }


def k16_segment_ids(n_rows: int, k_segments: int = 16) -> np.ndarray:
    if n_rows <= 0 or k_segments <= 0:
        raise ValueError("K16 segmentation requires positive row and segment counts")
    edges = np.linspace(0.0, float(n_rows), k_segments + 1)
    step_index = np.arange(1.0, n_rows + 1.0)
    return np.clip(
        np.searchsorted(edges[1:], step_index, side="left"),
        0,
        k_segments - 1,
    ).astype(np.int16)


def segment_step_rates(
    md: np.ndarray,
    u: np.ndarray,
    segment_ids: np.ndarray,
    k_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    md = np.asarray(md, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    segment_ids = np.asarray(segment_ids, dtype=np.int16)
    rates = np.full(k_segments, np.nan, dtype=np.float64)
    counts = np.zeros(k_segments, dtype=np.int32)
    if len(md) < 2:
        return rates, counts
    delta_md = np.diff(md)
    delta_u = np.diff(u)
    valid = np.isfinite(delta_md) & np.isfinite(delta_u) & (delta_md > 0.0)
    step_rate = np.full(len(delta_md), np.nan, dtype=np.float64)
    step_rate[valid] = delta_u[valid] / delta_md[valid]
    destination_segment = segment_ids[1:]
    for segment_id in range(k_segments):
        selected = step_rate[valid & (destination_segment == segment_id) & np.isfinite(step_rate)]
        counts[segment_id] = len(selected)
        if len(selected):
            rates[segment_id] = float(np.median(selected))
    return rates, counts


def validate_well_alignment(
    well: str,
    geometry: pd.DataFrame,
    horizontal: pd.DataFrame,
) -> np.ndarray:
    row_index = geometry["row_idx"].to_numpy(np.int64)
    suffix_offset = geometry["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(suffix_offset, np.arange(len(geometry), dtype=np.int64)):
        raise ValueError(f"{well}: suffix_offset is not stable contiguous row order")
    unknown_index = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    if not np.array_equal(row_index, unknown_index):
        raise ValueError(f"{well}: exp226 rows do not match the raw unknown suffix")
    if len(row_index) == 0 or row_index[0] == 0:
        raise ValueError(f"{well}: no known-prefix anchor")
    if not horizontal.loc[: row_index[0] - 1, "TVT_input"].notna().all():
        raise ValueError(f"{well}: known prefix is not contiguous")
    return row_index


def build_well_rate_schedule(
    well: str,
    geometry: pd.DataFrame,
    horizontal: pd.DataFrame,
    *,
    k_segments: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    geometry = geometry.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_index = validate_well_alignment(well, geometry, horizontal)
    segment_ids = k16_segment_ids(len(geometry), k_segments)
    md = horizontal.loc[row_index, "MD"].to_numpy(np.float64)
    z = horizontal.loc[row_index, "Z"].to_numpy(np.float64)
    geometry_tvt = geometry["tvt_geop"].to_numpy(np.float64)
    if not (np.isfinite(md).all() and np.isfinite(z).all() and np.isfinite(geometry_tvt).all()):
        raise ValueError(f"{well}: K16 schedule inputs must be finite")

    rate_audit = exp209_initial_rate(horizontal)
    prefix_rate = float(rate_audit["initial_rate"])
    geometry_rate, valid_steps = segment_step_rates(
        md,
        geometry_tvt + z,
        segment_ids,
        k_segments,
    )
    first_geometry_rate = float(geometry_rate[0])
    first_valid = math.isfinite(first_geometry_rate)
    mu_by_segment = np.full(k_segments, prefix_rate, dtype=np.float64)
    delta_by_segment = np.zeros(k_segments, dtype=np.float64)
    fallback_by_segment = np.ones(k_segments, dtype=bool)
    if first_valid:
        valid_segments = np.isfinite(geometry_rate)
        delta_by_segment[valid_segments] = geometry_rate[valid_segments] - first_geometry_rate
        mu_by_segment[valid_segments] = prefix_rate + delta_by_segment[valid_segments]
        fallback_by_segment[valid_segments] = False

    anchor_index = int(row_index[0] - 1)
    anchor_md = float(horizontal.loc[anchor_index, "MD"])
    anchor_z = float(horizontal.loc[anchor_index, "Z"])
    anchor_tvt = float(horizontal.loc[anchor_index, "TVT_input"])
    delta_md = np.diff(np.r_[anchor_md, md])
    if not np.isfinite(delta_md).all() or np.any(delta_md <= 0.0):
        raise ValueError(f"{well}: MD must increase strictly across the suffix")
    row_mu = mu_by_segment[segment_ids]
    schedule = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in row_index],
            "well_id": well,
            "row_idx": row_index.astype(np.int32),
            "suffix_offset": geometry["suffix_offset"].to_numpy(np.int32),
            "fold": geometry["fold"].to_numpy(np.int8),
            "segment_id": segment_ids,
            "md": md,
            "z": z,
            "delta_md": delta_md,
            "md_since": md - anchor_md,
            "tvt_geop": geometry_tvt,
            "prefix_rate": prefix_rate,
            "geometry_segment_rate": geometry_rate[segment_ids],
            "geometry_delta_rate": delta_by_segment[segment_ids],
            "mu_rate": row_mu,
            "geometry_fallback": fallback_by_segment[segment_ids],
            "anchor_tvt": anchor_tvt,
            "anchor_z": anchor_z,
        }
    )
    ledger = pd.DataFrame(
        {
            "well_id": well,
            "fold": int(geometry["fold"].iloc[0]),
            "segment_id": np.arange(k_segments, dtype=np.int16),
            "row_count": np.bincount(segment_ids, minlength=k_segments).astype(np.int32),
            "valid_geometry_steps": valid_steps,
            "prefix_rate": prefix_rate,
            "first_segment_geometry_rate": first_geometry_rate,
            "geometry_segment_rate": geometry_rate,
            "geometry_delta_rate": delta_by_segment,
            "mu_rate": mu_by_segment,
            "geometry_fallback": fallback_by_segment,
        }
    )
    fallback = {
        "well_id": well,
        "fold": int(geometry["fold"].iloc[0]),
        "rows": len(schedule),
        "prefix_rate": prefix_rate,
        "prefix_rate_fallback": bool(rate_audit["fallback"]),
        "prefix_rate_valid_steps": int(rate_audit["valid_steps"]),
        "first_geometry_segment_valid": first_valid,
        "fallback_segments": int(fallback_by_segment.sum()),
        "fallback_rows": int(fallback_by_segment[segment_ids].sum()),
    }
    return schedule, ledger, fallback


@dataclass(frozen=True)
class FrozenSchedule:
    rows: pd.DataFrame
    segments: pd.DataFrame
    fallback: pd.DataFrame
    content_sha256: str
    segment_sha256: str


def build_and_freeze_schedule(
    geometry: pd.DataFrame,
    raw_dir: Path,
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
    *,
    require_full: bool = True,
) -> FrozenSchedule:
    schedules: list[pd.DataFrame] = []
    segments: list[pd.DataFrame] = []
    fallback_rows: list[dict[str, Any]] = []
    k_segments = int(get_nested(config, "model.h_branch.transition_mean_schedule.k_segments"))
    for well, group in geometry.groupby("well_id", sort=True, observed=True):
        horizontal, _ = load_target_free_well(str(well), raw_dir, ledger)
        schedule, segment, fallback = build_well_rate_schedule(
            str(well),
            group,
            horizontal,
            k_segments=k_segments,
        )
        schedules.append(schedule)
        segments.append(segment)
        fallback_rows.append(fallback)
    rows = (
        pd.concat(schedules, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    segment_frame = (
        pd.concat(segments, ignore_index=True)
        .sort_values(["well_id", "segment_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    fallback_frame = (
        pd.DataFrame(fallback_rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    )
    forbidden = {"TVT", "true_tvt", "target", "error", "abs_error", "gr_delta", "tvt_pred"}
    if forbidden.intersection(rows.columns):
        raise RuntimeError("truth or forbidden exp226 output entered the frozen schedule")
    if require_full and len(rows) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("frozen schedule row count mismatch")
    return FrozenSchedule(
        rows=rows,
        segments=segment_frame,
        fallback=fallback_frame,
        content_sha256=logical_frame_sha256(rows),
        segment_sha256=logical_frame_sha256(segment_frame),
    )


# %% [markdown]
# ## 5. Exp209 observation and state-grid preparation


# %%
def exp209_prefix_scale(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
) -> dict[str, Any]:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    known_tvt = pd.to_numeric(known["TVT_input"], errors="raise").to_numpy(np.float64)
    known_gr_series = pd.to_numeric(known["GR"], errors="coerce")
    known_gr = known_gr_series.fillna(0.0).to_numpy(np.float64)
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - typewell_at_known
    raw_sigma = float(np.nanstd(residual, ddof=0))
    sigma = float(np.clip(raw_sigma, 10.0, 60.0))
    finite = np.isfinite(typewell_at_known) & np.isfinite(known_gr)
    if finite.sum() >= 3 and np.std(typewell_at_known[finite]) > 0.0:
        affine_a, affine_b = np.polyfit(
            typewell_at_known[finite],
            known_gr[finite],
            1,
        )
    else:
        affine_a, affine_b = 1.0, 0.0
    # exp209 sigma_mode=std evaluates the raw typewell curve (a=1, b=0).
    return {
        "sigma_gr_raw": raw_sigma,
        "sigma_gr": sigma,
        "known_prefix_rows": int(len(known)),
        "missing_known_gr_rows": int(known_gr_series.isna().sum()),
        "sigma_mode": "std",
        "missing_known_gr_fill": 0.0,
        "affine_a_diagnostic": float(affine_a),
        "affine_b_diagnostic": float(affine_b),
        "emission_affine_a": 1.0,
        "emission_affine_b": 0.0,
    }


def gaussian_emissions(
    observed_gr: np.ndarray,
    state_gr: np.ndarray,
    e_state_gr: np.ndarray,
    sigma_gr: float,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed_gr, dtype=np.float64)
    state = np.asarray(state_gr, dtype=np.float64)
    e_state = np.asarray(e_state_gr, dtype=np.float64)
    if observed.ndim != 1 or state.ndim != 1 or e_state.shape != observed.shape:
        raise ValueError("emission inputs have inconsistent shape")
    safe_observed = np.where(np.isfinite(observed), observed, 0.0)
    h_zscore = (safe_observed[:, None] - state[None, :]) / float(sigma_gr)
    e_zscore = (safe_observed - e_state) / float(sigma_gr)
    return (
        (-0.5 * np.minimum(h_zscore**2, 600.0)).astype(np.float64),
        (-0.5 * np.minimum(e_zscore**2, 600.0)).astype(np.float64),
    )


def prepare_switching_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    schedule: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    hmm = dict(get_nested(config, "model.h_branch.fixed_hmm"))
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known_index = np.flatnonzero(np.isfinite(tvt_input))
    eval_index = np.flatnonzero(~np.isfinite(tvt_input))
    if len(known_index) == 0 or len(eval_index) == 0:
        raise ValueError("switching HMM requires a visible prefix and unknown suffix")
    if not np.array_equal(known_index, np.arange(len(known_index))):
        raise ValueError("visible prefix must be contiguous")
    if not np.array_equal(eval_index, np.arange(eval_index[0], len(horizontal))):
        raise ValueError("unknown suffix must be contiguous")
    schedule = schedule.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(schedule["row_idx"].to_numpy(np.int64), eval_index):
        raise ValueError("K16 schedule row identity differs from raw suffix")

    last_index = int(known_index[-1])
    last_tvt = float(tvt_input[last_index])
    typewell_tvt = pd.to_numeric(typewell["TVT"], errors="raise").to_numpy(np.float64)
    typewell_gr = (
        pd.to_numeric(typewell["GR"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    )
    band_pad = float(hmm["band_pad_ft"])
    grid_min = max(float(typewell_tvt.min()) - 40.0, last_tvt - band_pad)
    grid_max = min(float(typewell_tvt.max()) + 40.0, last_tvt + band_pad)
    step = float(hmm["step_ft"])
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    state_gr = np.interp(grid, typewell_tvt, typewell_gr)
    e_state_gr = np.interp(
        schedule["tvt_geop"].to_numpy(np.float64),
        typewell_tvt,
        typewell_gr,
    )
    observed_gr = (
        pd.to_numeric(horizontal["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
        .to_numpy(np.float64)[eval_index]
    )
    md_all = pd.to_numeric(horizontal["MD"], errors="raise").to_numpy(np.float64)
    z_all = pd.to_numeric(horizontal["Z"], errors="raise").to_numpy(np.float64)
    md = md_all[eval_index]
    z = z_all[eval_index]
    delta_md = np.maximum(np.diff(np.r_[md_all[last_index], md]), 1.0)
    delta_z = np.diff(np.r_[z_all[last_index], z])
    mu_rate = schedule["mu_rate"].to_numpy(np.float64)
    rates = np.linspace(
        -float(hmm["residual_rate_span"]),
        float(hmm["residual_rate_span"]),
        int(hmm["n_rates"]),
        dtype=np.float64,
    )
    prefix_rate = float(exp209_initial_rate(horizontal)["initial_rate"])
    initial_residual_rate = prefix_rate - float(mu_rate[0])
    if abs(initial_residual_rate) > float(hmm["residual_rate_span"]) + 1.0e-12:
        raise ValueError("initial residual rate is outside the frozen 41-state grid")
    scale = exp209_prefix_scale(horizontal, typewell)
    h_emission, e_emission = gaussian_emissions(
        observed_gr,
        state_gr,
        e_state_gr,
        float(scale["sigma_gr"]),
    )
    return {
        "eval_index": eval_index,
        "grid": grid,
        "rates": rates,
        "h_emission": h_emission,
        "e_emission": e_emission,
        "observed_gr": observed_gr,
        "state_gr": state_gr,
        "e_state_gr": e_state_gr,
        "delta_md": delta_md,
        "delta_z": delta_z,
        "effective_delta_z": delta_z - mu_rate * delta_md,
        "q_e": schedule["tvt_geop"].to_numpy(np.float64),
        "mu_rate": mu_rate,
        "start_position_index": float((last_tvt - grid_min) / step),
        "initial_residual_rate": initial_residual_rate,
        "prefix_rate": prefix_rate,
        "prefix_scale": scale,
        "anchor_tvt": last_tvt,
    }


# %% [markdown]
# ## 6. Soft-sticky transition and exact forward-backward helpers


# %%
def rate_transition_probabilities(
    rates: np.ndarray,
    source_index: int,
    delta_md: float,
    sig_r: float,
    momentum: float,
) -> list[tuple[int, float]]:
    rates = np.asarray(rates, dtype=np.float64)
    if len(rates) < 2:
        raise ValueError("residual-rate grid requires at least two states")
    rate_step = float(rates[1] - rates[0])
    variance_cells = (float(sig_r) * math.sqrt(float(delta_md)) / rate_step) ** 2
    mean_move = -(1.0 - float(momentum)) * float(rates[source_index]) * float(delta_md) / rate_step
    p_plus = max(0.5 * (variance_cells + mean_move), 1.0e-12)
    p_minus = max(0.5 * (variance_cells - mean_move), 1.0e-12)
    if p_plus + p_minus > 0.9:
        scale = 0.9 / (p_plus + p_minus)
        p_plus *= scale
        p_minus *= scale
    raw = [
        (source_index - 1, p_minus),
        (source_index, 1.0 - p_plus - p_minus),
        (source_index + 1, p_plus),
    ]
    valid = [(index, value) for index, value in raw if 0 <= index < len(rates)]
    total = sum(value for _, value in valid)
    return [(index, value / total) for index, value in valid]


def position_transition_probabilities(
    grid: np.ndarray,
    source_tvt: float,
    destination_rate: float,
    delta_md: float,
    effective_delta_z: float,
    sig_p: float,
) -> list[tuple[int, float]]:
    grid = np.asarray(grid, dtype=np.float64)
    step = float(grid[1] - grid[0])
    sigma_position = max(float(sig_p), 0.35 * step)
    target = (
        float(source_tvt) + float(destination_rate) * float(delta_md) - float(effective_delta_z)
    )
    center = int(math.floor((target - float(grid[0])) / step + 0.5))
    raw: list[tuple[int, float]] = []
    for destination in range(center - 2, center + 3):
        if 0 <= destination < len(grid):
            distance = float(grid[destination]) - target
            raw.append(
                (
                    destination,
                    math.exp(-0.5 * (distance / sigma_position) ** 2),
                )
            )
    if not raw:
        nearest = int(np.clip(center, 0, len(grid) - 1))
        return [(nearest, 1.0)]
    total = sum(value for _, value in raw)
    return [(index, value / total) for index, value in raw]


def build_dense_joint_transition(
    grid: np.ndarray,
    rates: np.ndarray,
    *,
    delta_md: float,
    effective_delta_z: float,
    sig_r: float,
    sig_p: float,
    momentum: float,
    q_e_previous: float,
    q_e_current: float,
    switching_length_md_ft: float,
    docking_sigma_ft: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reference transition used by exhaustive small-trellis tests.

    State 0 is E; state `1 + position * R + rate` is H.  Every H source row
    first forms the same normalized exp209 rate/position kernel, then assigns
    `h*dock` to H→E and the remaining mass to that complete H kernel.
    """
    grid = np.asarray(grid, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    position_count = len(grid)
    rate_count = len(rates)
    state_count = 1 + position_count * rate_count
    transition = np.zeros((state_count, state_count), dtype=np.float64)
    hazard = 1.0 - math.exp(-float(delta_md) / float(switching_length_md_ft))
    transition[0, 0] = 1.0 - hazard

    zero_rate_index = int(np.argmin(np.abs(rates)))
    injection = np.zeros((position_count, rate_count), dtype=np.float64)
    for destination_rate, rate_probability in rate_transition_probabilities(
        rates,
        zero_rate_index,
        delta_md,
        sig_r,
        momentum,
    ):
        for destination_position, position_probability in position_transition_probabilities(
            grid,
            q_e_previous,
            rates[destination_rate],
            delta_md,
            effective_delta_z,
            sig_p,
        ):
            injection[destination_position, destination_rate] += (
                rate_probability * position_probability
            )
    injection /= injection.sum()
    for position in range(position_count):
        for rate in range(rate_count):
            transition[0, 1 + position * rate_count + rate] = hazard * injection[position, rate]

    docking = np.zeros((position_count, rate_count), dtype=np.float64)
    for source_position in range(position_count):
        for source_rate in range(rate_count):
            kernel = np.zeros((position_count, rate_count), dtype=np.float64)
            for destination_rate, rate_probability in rate_transition_probabilities(
                rates,
                source_rate,
                delta_md,
                sig_r,
                momentum,
            ):
                for (
                    destination_position,
                    position_probability,
                ) in position_transition_probabilities(
                    grid,
                    grid[source_position],
                    rates[destination_rate],
                    delta_md,
                    effective_delta_z,
                    sig_p,
                ):
                    kernel[destination_position, destination_rate] += (
                        rate_probability * position_probability
                    )
            kernel /= kernel.sum()
            dock_weight = np.exp(
                -0.5 * ((grid - float(q_e_current)) / float(docking_sigma_ft)) ** 2
            )
            docking_score = float((kernel * dock_weight[:, None]).sum())
            docking[source_position, source_rate] = docking_score
            switch_probability = hazard * docking_score
            source = 1 + source_position * rate_count + source_rate
            transition[source, 0] = switch_probability
            for destination_position in range(position_count):
                for destination_rate in range(rate_count):
                    destination = 1 + destination_position * rate_count + destination_rate
                    transition[source, destination] = (1.0 - switch_probability) * kernel[
                        destination_position, destination_rate
                    ]
    row_error = float(np.max(np.abs(transition.sum(axis=1) - 1.0)))
    return transition, {
        "hazard": hazard,
        "injection": injection,
        "docking": docking,
        "transition_row_sum_max_abs_error": row_error,
    }


def exact_dense_forward_backward(
    initial_probability: np.ndarray,
    transitions: Sequence[np.ndarray],
    log_emission: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Log-space reference implementation for small exhaustive parity tests."""
    initial = np.asarray(initial_probability, dtype=np.float64)
    emission = np.asarray(log_emission, dtype=np.float64)
    if emission.ndim != 2 or initial.shape != (emission.shape[1],):
        raise ValueError("dense forward-backward shape mismatch")
    if len(transitions) != emission.shape[0] - 1:
        raise ValueError("dense transition count mismatch")
    if not np.isclose(initial.sum(), 1.0):
        raise ValueError("initial probability must sum to one")
    tiny = np.finfo(np.float64).tiny
    log_initial = np.log(np.maximum(initial, tiny))
    alpha = np.full_like(emission, -np.inf)
    alpha[0] = log_initial + emission[0]
    for row in range(1, len(emission)):
        log_transition = np.log(np.maximum(transitions[row - 1], tiny))
        for destination in range(emission.shape[1]):
            values = alpha[row - 1] + log_transition[:, destination]
            maximum = float(np.max(values))
            alpha[row, destination] = (
                emission[row, destination]
                + maximum
                + math.log(float(np.exp(values - maximum).sum()))
            )
    maximum = float(np.max(alpha[-1]))
    log_partition = maximum + math.log(float(np.exp(alpha[-1] - maximum).sum()))
    beta = np.zeros_like(alpha)
    for row in range(len(emission) - 2, -1, -1):
        log_transition = np.log(np.maximum(transitions[row], tiny))
        for source in range(emission.shape[1]):
            values = log_transition[source] + emission[row + 1] + beta[row + 1]
            maximum = float(np.max(values))
            beta[row, source] = maximum + math.log(float(np.exp(values - maximum).sum()))
    posterior = np.exp(alpha + beta - log_partition)
    posterior /= posterior.sum(axis=1, keepdims=True)
    return posterior, float(log_partition)


@njit(cache=True, inline="always")
def _logaddexp(left: float, right: float) -> float:
    if left == -np.inf:
        return right
    if right == -np.inf:
        return left
    maximum = left if left >= right else right
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


@njit(cache=True)
def _rate_kernel(
    rates: np.ndarray,
    delta_md: float,
    sig_r: float,
    momentum: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    rate_count = len(rates)
    rate_step = rates[1] - rates[0]
    destinations = np.full((rate_count, 3), -1, np.int64)
    log_probability = np.full((rate_count, 3), -np.inf, np.float64)
    maximum_error = 0.0
    for source in range(rate_count):
        variance_cells = (sig_r * math.sqrt(delta_md) / rate_step) ** 2
        mean_move = -(1.0 - momentum) * rates[source] * delta_md / rate_step
        p_plus = max(0.5 * (variance_cells + mean_move), 1.0e-12)
        p_minus = max(0.5 * (variance_cells - mean_move), 1.0e-12)
        if p_plus + p_minus > 0.9:
            scale = 0.9 / (p_plus + p_minus)
            p_plus *= scale
            p_minus *= scale
        raw = np.array((p_minus, 1.0 - p_plus - p_minus, p_plus))
        total = 0.0
        for slot in range(3):
            destination = source + slot - 1
            if 0 <= destination < rate_count:
                destinations[source, slot] = destination
                total += raw[slot]
        normalized_sum = 0.0
        for slot in range(3):
            if destinations[source, slot] >= 0:
                probability = raw[slot] / total
                log_probability[source, slot] = math.log(probability)
                normalized_sum += probability
        maximum_error = max(maximum_error, abs(normalized_sum - 1.0))
    return destinations, log_probability, maximum_error


@njit(cache=True, parallel=True)
def _position_kernel(
    grid: np.ndarray,
    rates: np.ndarray,
    delta_md: float,
    effective_delta_z: float,
    sig_p: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    position_count = len(grid)
    rate_count = len(rates)
    step = grid[1] - grid[0]
    sigma_position = max(sig_p, 0.35 * step)
    destinations = np.full((rate_count, position_count, 5), -1, np.int64)
    log_probability = np.full(
        (rate_count, position_count, 5),
        -np.inf,
        np.float64,
    )
    row_error = np.zeros((rate_count, position_count), dtype=np.float64)
    for destination_rate in prange(rate_count):
        move = rates[destination_rate] * delta_md - effective_delta_z
        offset_center = int(math.floor(move / step + 0.5))
        raw = np.empty(5, dtype=np.float64)
        for slot in range(5):
            delta = (offset_center - 2 + slot) * step - move
            raw[slot] = math.exp(-0.5 * (delta / sigma_position) ** 2)
        for source_position in range(position_count):
            total = 0.0
            for slot in range(5):
                destination = source_position + offset_center - 2 + slot
                if 0 <= destination < position_count:
                    destinations[destination_rate, source_position, slot] = destination
                    total += raw[slot]
            normalized_sum = 0.0
            for slot in range(5):
                if destinations[destination_rate, source_position, slot] >= 0:
                    probability = raw[slot] / total
                    log_probability[destination_rate, source_position, slot] = math.log(probability)
                    normalized_sum += probability
            row_error[destination_rate, source_position] = abs(normalized_sum - 1.0)
    return destinations, log_probability, float(np.max(row_error))


@njit(cache=True)
def _continuous_injection(
    grid: np.ndarray,
    rates: np.ndarray,
    rate_destinations: np.ndarray,
    rate_log_probability: np.ndarray,
    delta_md: float,
    effective_delta_z: float,
    sig_p: float,
    source_tvt: float,
) -> np.ndarray:
    position_count = len(grid)
    rate_count = len(rates)
    step = grid[1] - grid[0]
    sigma_position = max(sig_p, 0.35 * step)
    zero_rate_source = int(np.argmin(np.abs(rates)))
    injection = np.full((position_count, rate_count), -np.inf, np.float64)
    for rate_slot in range(3):
        destination_rate = rate_destinations[zero_rate_source, rate_slot]
        if destination_rate < 0:
            continue
        target = source_tvt + rates[destination_rate] * delta_md - effective_delta_z
        center = int(math.floor((target - grid[0]) / step + 0.5))
        raw = np.empty(5, dtype=np.float64)
        total = 0.0
        for slot in range(5):
            destination_position = center - 2 + slot
            if 0 <= destination_position < position_count:
                distance = grid[destination_position] - target
                raw[slot] = math.exp(-0.5 * (distance / sigma_position) ** 2)
                total += raw[slot]
            else:
                raw[slot] = 0.0
        for slot in range(5):
            destination_position = center - 2 + slot
            if 0 <= destination_position < position_count and raw[slot] > 0.0:
                value = rate_log_probability[zero_rate_source, rate_slot] + math.log(
                    raw[slot] / total
                )
                injection[destination_position, destination_rate] = _logaddexp(
                    injection[destination_position, destination_rate],
                    value,
                )
    maximum = -np.inf
    for position in range(position_count):
        for rate in range(rate_count):
            maximum = max(maximum, injection[position, rate])
    total = 0.0
    for position in range(position_count):
        for rate in range(rate_count):
            total += math.exp(injection[position, rate] - maximum)
    normalizer = maximum + math.log(total)
    for position in range(position_count):
        for rate in range(rate_count):
            injection[position, rate] -= normalizer
    return injection


@njit(cache=True, parallel=True)
def _docking_scores(
    grid: np.ndarray,
    rates: np.ndarray,
    rate_destinations: np.ndarray,
    rate_log_probability: np.ndarray,
    position_destinations: np.ndarray,
    position_log_probability: np.ndarray,
    q_e_current: float,
    docking_sigma: float,
) -> np.ndarray:
    position_count = len(grid)
    rate_count = len(rates)
    docking = np.zeros((position_count, rate_count), dtype=np.float64)
    dock_weight = np.exp(-0.5 * ((grid - q_e_current) / docking_sigma) ** 2)
    for source_position in prange(position_count):
        for source_rate in range(rate_count):
            value = 0.0
            for rate_slot in range(3):
                destination_rate = rate_destinations[source_rate, rate_slot]
                if destination_rate < 0:
                    continue
                rate_probability = math.exp(rate_log_probability[source_rate, rate_slot])
                for position_slot in range(5):
                    destination_position = position_destinations[
                        destination_rate,
                        source_position,
                        position_slot,
                    ]
                    if destination_position < 0:
                        continue
                    position_probability = math.exp(
                        position_log_probability[
                            destination_rate,
                            source_position,
                            position_slot,
                        ]
                    )
                    value += (
                        rate_probability * position_probability * dock_weight[destination_position]
                    )
            docking[source_position, source_rate] = min(max(value, 0.0), 1.0)
    return docking


@njit(cache=True, parallel=True)
def _forward_h_transition(
    source: np.ndarray,
    source_log_stay: np.ndarray,
    rates: np.ndarray,
    rate_destinations: np.ndarray,
    rate_log_probability: np.ndarray,
    position_log_probability: np.ndarray,
    delta_md: float,
    effective_delta_z: float,
    step: float,
) -> np.ndarray:
    position_count, rate_count = source.shape
    rate_mixed = np.full((position_count, rate_count), -np.inf, np.float64)
    for source_position in prange(position_count):
        for source_rate in range(rate_count):
            base = (
                source[source_position, source_rate]
                + source_log_stay[
                    source_position,
                    source_rate,
                ]
            )
            if base == -np.inf:
                continue
            for rate_slot in range(3):
                destination_rate = rate_destinations[source_rate, rate_slot]
                if destination_rate >= 0:
                    rate_mixed[source_position, destination_rate] = _logaddexp(
                        rate_mixed[source_position, destination_rate],
                        base + rate_log_probability[source_rate, rate_slot],
                    )
    output = np.full((position_count, rate_count), -np.inf, np.float64)
    for destination_position in prange(position_count):
        for destination_rate in range(rate_count):
            move = rates[destination_rate] * delta_md - effective_delta_z
            offset_center = int(math.floor(move / step + 0.5))
            value = -np.inf
            for position_slot in range(5):
                source_position = destination_position - (offset_center - 2 + position_slot)
                if 0 <= source_position < position_count:
                    log_probability = position_log_probability[
                        destination_rate,
                        source_position,
                        position_slot,
                    ]
                    if log_probability != -np.inf:
                        value = _logaddexp(
                            value,
                            rate_mixed[source_position, destination_rate] + log_probability,
                        )
            output[destination_position, destination_rate] = value
    return output


@njit(cache=True, parallel=True)
def _backward_h_transition(
    destination_value: np.ndarray,
    source_log_stay: np.ndarray,
    rate_destinations: np.ndarray,
    rate_log_probability: np.ndarray,
    position_destinations: np.ndarray,
    position_log_probability: np.ndarray,
) -> np.ndarray:
    position_count, rate_count = source_log_stay.shape
    output = np.full((position_count, rate_count), -np.inf, np.float64)
    for source_position in prange(position_count):
        for source_rate in range(rate_count):
            value = -np.inf
            for rate_slot in range(3):
                destination_rate = rate_destinations[source_rate, rate_slot]
                if destination_rate < 0:
                    continue
                for position_slot in range(5):
                    destination_position = position_destinations[
                        destination_rate,
                        source_position,
                        position_slot,
                    ]
                    if destination_position < 0:
                        continue
                    value = _logaddexp(
                        value,
                        rate_log_probability[source_rate, rate_slot]
                        + position_log_probability[
                            destination_rate,
                            source_position,
                            position_slot,
                        ]
                        + destination_value[destination_position, destination_rate],
                    )
            output[source_position, source_rate] = (
                source_log_stay[source_position, source_rate] + value
            )
    return output


@njit(cache=True)
def _matrix_logsumexp(values: np.ndarray) -> float:
    maximum = -np.inf
    position_count, rate_count = values.shape
    for position in range(position_count):
        for rate in range(rate_count):
            maximum = max(maximum, values[position, rate])
    if maximum == -np.inf:
        return maximum
    total = 0.0
    for position in range(position_count):
        for rate in range(rate_count):
            total += math.exp(values[position, rate] - maximum)
    return maximum + math.log(total)


@njit(cache=True)
def _joint_logsumexp(e_value: float, h_values: np.ndarray) -> float:
    maximum = max(e_value, np.max(h_values))
    total = math.exp(e_value - maximum)
    position_count, rate_count = h_values.shape
    for position in range(position_count):
        for rate in range(rate_count):
            total += math.exp(h_values[position, rate] - maximum)
    return maximum + math.log(total)


@njit(cache=True)
def _soft_sticky_forward_backward(
    h_emission: np.ndarray,
    e_emission: np.ndarray,
    grid: np.ndarray,
    rates: np.ndarray,
    delta_md: np.ndarray,
    effective_delta_z: np.ndarray,
    q_e: np.ndarray,
    step: float,
    sig_r: float,
    sig_p: float,
    start_position_index: float,
    start_sig: float,
    initial_residual_rate: float,
    r0_sig: float,
    momentum: float,
    switching_length: float,
    docking_sigma: float,
    initial_e_probability: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    """Scaled log-space exact FB over E plus the full H joint trellis."""
    row_count, position_count = h_emission.shape
    rate_count = len(rates)
    if row_count == 0:
        raise ValueError("soft-sticky HMM received zero rows")
    alpha_h = np.full(
        (row_count, position_count, rate_count),
        -np.inf,
        np.float32,
    )
    alpha_e = np.full(row_count, -np.inf, np.float64)
    log_scale = np.zeros(row_count, np.float64)
    transition_error = 0.0

    initial_h = np.full((position_count, rate_count), -np.inf, np.float64)
    for position in range(position_count):
        position_delta = (position - start_position_index) * step
        position_log = -0.5 * (position_delta / start_sig) ** 2
        if position_log < -60.0:
            continue
        for rate in range(rate_count):
            rate_delta = (rates[rate] - initial_residual_rate) / r0_sig
            initial_h[position, rate] = position_log - 0.5 * rate_delta**2
    initial_h -= _matrix_logsumexp(initial_h)

    rate_destination, rate_logp, rate_error = _rate_kernel(
        rates,
        delta_md[0],
        sig_r,
        momentum,
    )
    position_destination, position_logp, position_error = _position_kernel(
        grid,
        rates,
        delta_md[0],
        effective_delta_z[0],
        sig_p,
    )
    transition_error = max(transition_error, rate_error, position_error)
    no_stay_adjustment = np.zeros((position_count, rate_count), np.float64)
    h_current = _forward_h_transition(
        initial_h,
        no_stay_adjustment,
        rates,
        rate_destination,
        rate_logp,
        position_logp,
        delta_md[0],
        effective_delta_z[0],
        step,
    )
    log_h_prior = math.log(1.0 - initial_e_probability)
    log_e_prior = math.log(initial_e_probability)
    for position in range(position_count):
        for rate in range(rate_count):
            h_current[position, rate] += log_h_prior + h_emission[0, position]
    e_current = log_e_prior + e_emission[0]
    scale = _joint_logsumexp(e_current, h_current)
    log_scale[0] = scale
    alpha_e[0] = e_current - scale
    for position in range(position_count):
        for rate in range(rate_count):
            alpha_h[0, position, rate] = np.float32(h_current[position, rate] - scale)

    for row in range(1, row_count):
        hazard = 1.0 - math.exp(-delta_md[row] / switching_length)
        rate_destination, rate_logp, rate_error = _rate_kernel(
            rates,
            delta_md[row],
            sig_r,
            momentum,
        )
        position_destination, position_logp, position_error = _position_kernel(
            grid,
            rates,
            delta_md[row],
            effective_delta_z[row],
            sig_p,
        )
        docking = _docking_scores(
            grid,
            rates,
            rate_destination,
            rate_logp,
            position_destination,
            position_logp,
            q_e[row],
            docking_sigma,
        )
        h_source = alpha_h[row - 1].astype(np.float64)
        h_log_stay = np.empty((position_count, rate_count), np.float64)
        row_mixture_error = 0.0
        for position in range(position_count):
            for rate in range(rate_count):
                switch_probability = hazard * docking[position, rate]
                h_log_stay[position, rate] = math.log1p(-switch_probability)
                row_mixture_error = max(
                    row_mixture_error,
                    abs(switch_probability + math.exp(h_log_stay[position, rate]) - 1.0),
                )
        h_propagated = _forward_h_transition(
            h_source,
            h_log_stay,
            rates,
            rate_destination,
            rate_logp,
            position_logp,
            delta_md[row],
            effective_delta_z[row],
            step,
        )
        injection = _continuous_injection(
            grid,
            rates,
            rate_destination,
            rate_logp,
            delta_md[row],
            effective_delta_z[row],
            sig_p,
            q_e[row - 1],
        )
        injection_sum = 0.0
        for position in range(position_count):
            for rate in range(rate_count):
                injection_sum += math.exp(injection[position, rate])
                from_e = (
                    alpha_e[row - 1]
                    + math.log(hazard)
                    + injection[
                        position,
                        rate,
                    ]
                )
                h_current[position, rate] = (
                    _logaddexp(
                        h_propagated[position, rate],
                        from_e,
                    )
                    + h_emission[row, position]
                )
        h_to_e = -np.inf
        for position in range(position_count):
            for rate in range(rate_count):
                switch_probability = hazard * docking[position, rate]
                if switch_probability > 0.0:
                    h_to_e = _logaddexp(
                        h_to_e,
                        h_source[position, rate] + math.log(switch_probability),
                    )
        e_current = (
            _logaddexp(
                alpha_e[row - 1] + math.log1p(-hazard),
                h_to_e,
            )
            + e_emission[row]
        )
        scale = _joint_logsumexp(e_current, h_current)
        log_scale[row] = scale
        alpha_e[row] = e_current - scale
        for position in range(position_count):
            for rate in range(rate_count):
                alpha_h[row, position, rate] = np.float32(h_current[position, rate] - scale)
        transition_error = max(
            transition_error,
            rate_error,
            position_error,
            row_mixture_error,
            abs(injection_sum - 1.0),
            abs(hazard + (1.0 - hazard) - 1.0),
        )

    gamma_e = np.zeros(row_count, np.float64)
    h_position_posterior = np.zeros((row_count, position_count), np.float64)
    h_conditional_mean = np.zeros(row_count, np.float64)
    h_conditional_std = np.zeros(row_count, np.float64)
    joint_mean = np.zeros(row_count, np.float64)
    joint_std = np.zeros(row_count, np.float64)
    expected_switch = np.zeros(row_count, np.float64)
    docking_probability = np.zeros(row_count, np.float64)
    normalization_error = 0.0

    beta_h_next = np.zeros((position_count, rate_count), np.float64)
    beta_e_next = 0.0
    last_joint_log = _joint_logsumexp(
        alpha_e[row_count - 1] + beta_e_next,
        alpha_h[row_count - 1].astype(np.float64) + beta_h_next,
    )
    gamma_e[row_count - 1] = math.exp(alpha_e[row_count - 1] + beta_e_next - last_joint_log)
    for position in range(position_count):
        value = 0.0
        for rate in range(rate_count):
            value += math.exp(
                float(alpha_h[row_count - 1, position, rate])
                + beta_h_next[position, rate]
                - last_joint_log
            )
        h_position_posterior[row_count - 1, position] = value

    for row in range(row_count - 1, 0, -1):
        hazard = 1.0 - math.exp(-delta_md[row] / switching_length)
        rate_destination, rate_logp, _ = _rate_kernel(
            rates,
            delta_md[row],
            sig_r,
            momentum,
        )
        position_destination, position_logp, _ = _position_kernel(
            grid,
            rates,
            delta_md[row],
            effective_delta_z[row],
            sig_p,
        )
        docking = _docking_scores(
            grid,
            rates,
            rate_destination,
            rate_logp,
            position_destination,
            position_logp,
            q_e[row],
            docking_sigma,
        )
        injection = _continuous_injection(
            grid,
            rates,
            rate_destination,
            rate_logp,
            delta_md[row],
            effective_delta_z[row],
            sig_p,
            q_e[row - 1],
        )
        destination_value = np.empty((position_count, rate_count), np.float64)
        e_to_h_inner = -np.inf
        for position in range(position_count):
            for rate in range(rate_count):
                destination_value[position, rate] = (
                    h_emission[row, position] + beta_h_next[position, rate]
                )
                e_to_h_inner = _logaddexp(
                    e_to_h_inner,
                    injection[position, rate] + destination_value[position, rate],
                )
        beta_e_previous = (
            _logaddexp(
                math.log1p(-hazard) + e_emission[row] + beta_e_next,
                math.log(hazard) + e_to_h_inner,
            )
            - log_scale[row]
        )
        h_log_stay = np.empty((position_count, rate_count), np.float64)
        for position in range(position_count):
            for rate in range(rate_count):
                h_log_stay[position, rate] = math.log1p(-hazard * docking[position, rate])
        h_stay_value = _backward_h_transition(
            destination_value,
            h_log_stay,
            rate_destination,
            rate_logp,
            position_destination,
            position_logp,
        )
        beta_h_previous = np.empty((position_count, rate_count), np.float64)
        for position in range(position_count):
            for rate in range(rate_count):
                switch_probability = hazard * docking[position, rate]
                to_e = -np.inf
                if switch_probability > 0.0:
                    to_e = math.log(switch_probability) + e_emission[row] + beta_e_next
                beta_h_previous[position, rate] = (
                    _logaddexp(
                        to_e,
                        h_stay_value[position, rate],
                    )
                    - log_scale[row]
                )

        log_eh = alpha_e[row - 1] + math.log(hazard) + e_to_h_inner - log_scale[row]
        log_he = -np.inf
        for position in range(position_count):
            for rate in range(rate_count):
                switch_probability = hazard * docking[position, rate]
                if switch_probability > 0.0:
                    log_he = _logaddexp(
                        log_he,
                        float(alpha_h[row - 1, position, rate])
                        + math.log(switch_probability)
                        + e_emission[row]
                        + beta_e_next
                        - log_scale[row],
                    )
        expected_switch[row] = min(
            1.0,
            max(0.0, math.exp(log_eh) + math.exp(log_he)),
        )

        previous_joint_log = _joint_logsumexp(
            alpha_e[row - 1] + beta_e_previous,
            alpha_h[row - 1].astype(np.float64) + beta_h_previous,
        )
        gamma_e[row - 1] = math.exp(alpha_e[row - 1] + beta_e_previous - previous_joint_log)
        docking_numerator = 0.0
        h_probability = 0.0
        for position in range(position_count):
            position_probability = 0.0
            for rate in range(rate_count):
                probability = math.exp(
                    float(alpha_h[row - 1, position, rate])
                    + beta_h_previous[position, rate]
                    - previous_joint_log
                )
                position_probability += probability
                docking_numerator += probability * docking[position, rate]
            h_position_posterior[row - 1, position] = position_probability
            h_probability += position_probability
        if h_probability > 0.0:
            docking_probability[row] = docking_numerator / h_probability
        beta_e_next = beta_e_previous
        beta_h_next = beta_h_previous

    for row in range(row_count):
        h_probability = float(np.sum(h_position_posterior[row]))
        row_total = gamma_e[row] + h_probability
        normalization_error = max(normalization_error, abs(row_total - 1.0))
        if row_total <= 0.0:
            raise RuntimeError("soft-sticky posterior row has zero mass")
        gamma_e[row] /= row_total
        h_position_posterior[row] /= row_total
        h_probability /= row_total
        conditional_first = 0.0
        conditional_second = 0.0
        if h_probability > 0.0:
            for position in range(position_count):
                conditional = h_position_posterior[row, position] / h_probability
                conditional_first += conditional * grid[position]
                conditional_second += conditional * grid[position] ** 2
        else:
            conditional_first = q_e[row]
            conditional_second = q_e[row] ** 2
        h_conditional_mean[row] = conditional_first
        h_conditional_std[row] = math.sqrt(max(conditional_second - conditional_first**2, 0.0))
        joint_first = gamma_e[row] * q_e[row]
        joint_second = gamma_e[row] * q_e[row] ** 2
        for position in range(position_count):
            joint_first += h_position_posterior[row, position] * grid[position]
            joint_second += h_position_posterior[row, position] * grid[position] ** 2
        joint_mean[row] = joint_first
        joint_std[row] = math.sqrt(max(joint_second - joint_first**2, 0.0))
    log_partition = float(np.sum(log_scale))
    return (
        gamma_e,
        h_position_posterior,
        h_conditional_mean,
        h_conditional_std,
        joint_mean,
        joint_std,
        expected_switch,
        docking_probability,
        log_scale,
        log_partition,
        normalization_error,
        transition_error,
    )


def run_soft_sticky_hmm(
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    hmm = dict(get_nested(config, "model.h_branch.fixed_hmm"))
    sticky = dict(get_nested(config, "model.soft_sticky"))
    (
        gamma_e,
        h_position_posterior,
        h_mean,
        h_std,
        joint_mean,
        joint_std,
        expected_switch,
        docking_probability,
        log_scale,
        log_partition,
        normalization_error,
        transition_error,
    ) = _soft_sticky_forward_backward(
        np.asarray(prepared["h_emission"], dtype=np.float64),
        np.asarray(prepared["e_emission"], dtype=np.float64),
        np.asarray(prepared["grid"], dtype=np.float64),
        np.asarray(prepared["rates"], dtype=np.float64),
        np.asarray(prepared["delta_md"], dtype=np.float64),
        np.asarray(prepared["effective_delta_z"], dtype=np.float64),
        np.asarray(prepared["q_e"], dtype=np.float64),
        float(hmm["step_ft"]),
        float(hmm["sig_r"]),
        float(hmm["sig_p"]),
        float(prepared["start_position_index"]),
        float(hmm["start_sig"]),
        float(prepared["initial_residual_rate"]),
        float(hmm["r0_sig"]),
        float(hmm["momentum"]),
        float(sticky["base_switching_length_md_ft"]),
        float(sticky["h_to_e_docking"]["sigma_ft"]),
        float(get_nested(config, "model.initial_regime_probability.E_exp226_geometry")),
    )
    gamma_h = 1.0 - gamma_e
    h_expected_emission = np.zeros(len(gamma_e), dtype=np.float64)
    for row in range(len(gamma_e)):
        if gamma_h[row] > 0.0:
            conditional_position = h_position_posterior[row] / gamma_h[row]
            h_expected_emission[row] = float(
                conditional_position @ np.asarray(prepared["h_emission"], dtype=np.float64)[row]
            )
    return {
        "gamma_e": gamma_e,
        "gamma_h": gamma_h,
        "h_position_posterior": h_position_posterior,
        "h_conditional_mean": h_mean,
        "h_conditional_std": h_std,
        "joint_mean": joint_mean,
        "joint_std": joint_std,
        "expected_switch_probability": expected_switch,
        "docking_probability": docking_probability,
        "e_emission_loglik": np.asarray(prepared["e_emission"], dtype=np.float64),
        "h_expected_emission_loglik": h_expected_emission,
        "row_log_normalizer": log_scale,
        "log_partition": float(log_partition),
        "posterior_normalization_max_abs_error": float(normalization_error),
        "transition_row_sum_max_abs_error": float(transition_error),
        "h_full_grid_coverage": 1.0,
    }


# %% [markdown]
# ## 7. Per-well joint decoding and diagnostics


# %%
def decode_well(
    well: str,
    raw_dir: Path,
    schedule: pd.DataFrame,
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
    prepared = prepare_switching_inputs(horizontal, typewell, schedule, config)
    result = run_soft_sticky_hmm(prepared, config)
    ordered = schedule.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    prediction = ordered[
        [
            "id",
            "well_id",
            "row_idx",
            "suffix_offset",
            "fold",
            "segment_id",
            "md_since",
            "delta_md",
            "mu_rate",
            "geometry_fallback",
            "tvt_geop",
        ]
    ].copy()
    prediction[CANDIDATE_NAME] = np.asarray(result["joint_mean"], dtype=np.float64)
    prediction["joint_std"] = np.asarray(result["joint_std"], dtype=np.float64)
    prediction["gamma_E"] = np.asarray(result["gamma_e"], dtype=np.float64)
    prediction["gamma_H"] = np.asarray(result["gamma_h"], dtype=np.float64)
    prediction["h_conditional_mean"] = np.asarray(
        result["h_conditional_mean"],
        dtype=np.float64,
    )
    prediction["h_conditional_std"] = np.asarray(
        result["h_conditional_std"],
        dtype=np.float64,
    )
    prediction["expected_switch_probability"] = np.asarray(
        result["expected_switch_probability"],
        dtype=np.float64,
    )
    prediction["docking_probability"] = np.asarray(
        result["docking_probability"],
        dtype=np.float64,
    )
    prediction["e_emission_loglik"] = np.asarray(
        result["e_emission_loglik"],
        dtype=np.float64,
    )
    prediction["h_expected_emission_loglik"] = np.asarray(
        result["h_expected_emission_loglik"],
        dtype=np.float64,
    )
    prediction["row_log_normalizer"] = np.asarray(
        result["row_log_normalizer"],
        dtype=np.float64,
    )
    finite_columns = [
        CANDIDATE_NAME,
        "joint_std",
        "gamma_E",
        "gamma_H",
        "h_conditional_mean",
        "h_conditional_std",
        "expected_switch_probability",
        "docking_probability",
        "e_emission_loglik",
        "h_expected_emission_loglik",
        "row_log_normalizer",
        "mu_rate",
        "tvt_geop",
    ]
    if not np.isfinite(prediction[finite_columns].to_numpy(np.float64)).all():
        raise RuntimeError(f"{well}: switching-HMM output contains non-finite values")
    branch_sum_error = float(
        np.max(
            np.abs(
                prediction["gamma_E"].to_numpy(np.float64)
                + prediction["gamma_H"].to_numpy(np.float64)
                - 1.0
            )
        )
    )
    elapsed = time.perf_counter() - started
    total_md = float(prediction["delta_md"].sum())
    expected_switches = float(prediction["expected_switch_probability"].sum())
    return prediction, {
        "well_id": well,
        "fold": int(ordered["fold"].iloc[0]),
        "rows": len(prediction),
        "grid_positions": len(prepared["grid"]),
        "rate_states": len(prepared["rates"]),
        "joint_h_states_per_row": len(prepared["grid"]) * len(prepared["rates"]),
        "state_time_units": len(prediction) * len(prepared["grid"]) * len(prepared["rates"]),
        "elapsed_seconds": elapsed,
        "peak_rss_gb": peak_rss_gb(),
        "prefix_rate": float(prepared["prefix_rate"]),
        "initial_residual_rate": float(prepared["initial_residual_rate"]),
        "sigma_gr": float(prepared["prefix_scale"]["sigma_gr"]),
        "log_partition": float(result["log_partition"]),
        "finite_prediction_coverage": float(
            np.isfinite(prediction[CANDIDATE_NAME].to_numpy(np.float64)).mean()
        ),
        "h_full_grid_coverage": float(result["h_full_grid_coverage"]),
        "posterior_normalization_max_abs_error": max(
            float(result["posterior_normalization_max_abs_error"]),
            branch_sum_error,
        ),
        "transition_row_sum_max_abs_error": float(result["transition_row_sum_max_abs_error"]),
        "e_occupancy": float(prediction["gamma_E"].mean()),
        "h_occupancy": float(prediction["gamma_H"].mean()),
        "expected_switch_count": expected_switches,
        "expected_switches_per_1000_md_ft": (
            expected_switches / total_md * 1000.0 if total_md > 0.0 else 0.0
        ),
    }


def decode_scope(
    wells: Sequence[str],
    raw_dir: Path,
    schedule: FrozenSchedule,
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    for index, well in enumerate(sorted(str(value) for value in wells), start=1):
        local_schedule = schedule.rows.loc[schedule.rows["well_id"].astype(str).eq(well)]
        prediction, runtime = decode_well(
            well,
            raw_dir,
            local_schedule,
            config,
            ledger,
        )
        predictions.append(prediction)
        runtime_rows.append(runtime)
        print(
            f"[{index:03d}/{len(wells):03d}] {well} "
            f"rows={runtime['rows']} grid={runtime['grid_positions']} "
            f"seconds={runtime['elapsed_seconds']:.2f}",
            flush=True,
        )
    prediction_frame = (
        pd.concat(predictions, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    runtime_frame = (
        pd.DataFrame(runtime_rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    )
    if prediction_frame.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("decoded scope contains duplicate well/row keys")
    return prediction_frame, runtime_frame


# %% [markdown]
# ## 8. Fixed 16-well technical preflight


# %%
def select_preflight_wells(
    geometry: pd.DataFrame,
    *,
    longest_per_fold: int = 3,
    expected_folds: Sequence[int] = (0, 1, 2, 3, 4),
) -> pd.DataFrame:
    length = (
        geometry.groupby(["well_id", "fold"], sort=True, observed=True)
        .size()
        .rename("suffix_rows")
        .reset_index()
    )
    selected_rows: list[dict[str, Any]] = []
    selected: set[str] = set()
    for fold in expected_folds:
        candidates = length.loc[length["fold"].astype(int).eq(int(fold))].sort_values(
            ["suffix_rows", "well_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        if len(candidates) < longest_per_fold:
            raise ValueError(f"fold {fold} has fewer than {longest_per_fold} wells")
        for rank, row in enumerate(
            candidates.head(longest_per_fold).itertuples(index=False),
            start=1,
        ):
            selected.add(str(row.well_id))
            selected_rows.append(
                {
                    "well_id": str(row.well_id),
                    "fold": int(row.fold),
                    "suffix_rows": int(row.suffix_rows),
                    "selection_reason": "fold_longest",
                    "selection_rank": rank,
                }
            )
    median_length = float(length["suffix_rows"].median())
    median_candidates = length.loc[~length["well_id"].astype(str).isin(selected)].copy()
    median_candidates["median_distance"] = (
        median_candidates["suffix_rows"].astype(float) - median_length
    ).abs()
    median_candidates = median_candidates.sort_values(
        ["median_distance", "well_id"],
        kind="mergesort",
    )
    if median_candidates.empty:
        raise ValueError("no duplicate-free global median-length well remains")
    median_row = median_candidates.iloc[0]
    selected_rows.append(
        {
            "well_id": str(median_row["well_id"]),
            "fold": int(median_row["fold"]),
            "suffix_rows": int(median_row["suffix_rows"]),
            "selection_reason": "global_median_length_duplicate_free",
            "selection_rank": 1,
        }
    )
    result = pd.DataFrame(selected_rows)
    if len(result) != 16 or result["well_id"].nunique() != 16:
        raise RuntimeError("exp394 technical preflight selection must contain 16 unique wells")
    return result


def estimate_full_state_time_units(
    geometry: pd.DataFrame,
    raw_dir: Path,
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> int:
    hmm = dict(get_nested(config, "model.h_branch.fixed_hmm"))
    step = float(hmm["step_ft"])
    band_pad = float(hmm["band_pad_ft"])
    rate_count = int(hmm["n_rates"])
    total = 0
    for well, group in geometry.groupby("well_id", sort=True, observed=True):
        horizontal, typewell = load_target_free_well(str(well), raw_dir, ledger)
        known = pd.to_numeric(horizontal["TVT_input"], errors="coerce").dropna()
        if known.empty:
            raise ValueError(f"{well}: no known prefix for state-size projection")
        last_tvt = float(known.iloc[-1])
        typewell_tvt = pd.to_numeric(typewell["TVT"], errors="raise").to_numpy(np.float64)
        grid_min = max(float(typewell_tvt.min()) - 40.0, last_tvt - band_pad)
        grid_max = min(float(typewell_tvt.max()) + 40.0, last_tvt + band_pad)
        grid_count = len(np.arange(grid_min, grid_max + step, step))
        total += len(group) * grid_count * rate_count
    return int(total)


def evaluate_preflight_gate(
    prediction: pd.DataFrame,
    runtime: pd.DataFrame,
    ledger: RoleReadLedger,
    *,
    projected_runtime_seconds: float,
    projected_peak_rss_gb: float,
    input_identity_passed: bool,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = dict(get_nested(config, "validation.technical_preflight"))
    technical = {
        "completed_wells": int(runtime["well_id"].nunique()),
        "required_completed_wells": int(gates["require_completed_wells"]),
        "finite_prediction_coverage": float(
            np.isfinite(prediction[CANDIDATE_NAME].to_numpy(np.float64)).mean()
        ),
        "required_finite_prediction_coverage": float(gates["require_finite_prediction_coverage"]),
        "h_full_grid_coverage": float(runtime["h_full_grid_coverage"].min()),
        "required_h_full_grid_coverage": float(gates["require_h_full_grid_coverage"]),
        "posterior_normalization_max_abs_error": float(
            runtime["posterior_normalization_max_abs_error"].max()
        ),
        "maximum_posterior_normalization_error": float(
            gates["maximum_posterior_normalization_error"]
        ),
        "transition_row_sum_max_abs_error": float(
            runtime["transition_row_sum_max_abs_error"].max()
        ),
        "maximum_transition_row_sum_error": float(gates["maximum_transition_row_sum_error"]),
        "projected_runtime_seconds": float(projected_runtime_seconds),
        "maximum_projected_runtime_seconds": float(gates["maximum_projected_runtime_seconds"]),
        "projected_peak_rss_gb": float(projected_peak_rss_gb),
        "maximum_projected_peak_rss_gb": float(gates["maximum_projected_peak_rss_gb"]),
        "input_identity_passed": bool(input_identity_passed),
        "truth_rows_before_freeze": int(ledger.suffix_truth_rows_before_freeze),
        "error_rows_before_freeze": int(ledger.error_rows_before_freeze),
        "hidden_role_rows_before_freeze": int(ledger.hidden_role_rows_before_freeze),
        "scientific_score_used_as_gate": False,
    }
    checks = {
        "completed_wells": (technical["completed_wells"] == technical["required_completed_wells"]),
        "finite_prediction_coverage": (
            technical["finite_prediction_coverage"]
            == technical["required_finite_prediction_coverage"]
        ),
        "h_full_grid_coverage": (
            technical["h_full_grid_coverage"] == technical["required_h_full_grid_coverage"]
        ),
        "posterior_normalization": (
            technical["posterior_normalization_max_abs_error"]
            <= technical["maximum_posterior_normalization_error"]
        ),
        "transition_row_sum": (
            technical["transition_row_sum_max_abs_error"]
            <= technical["maximum_transition_row_sum_error"]
        ),
        "runtime_projection": (
            technical["projected_runtime_seconds"] <= technical["maximum_projected_runtime_seconds"]
        ),
        "rss_projection": (
            technical["projected_peak_rss_gb"] <= technical["maximum_projected_peak_rss_gb"]
        ),
        "input_identity": technical["input_identity_passed"],
        "leakage_guard": (
            technical["truth_rows_before_freeze"] == 0
            and technical["error_rows_before_freeze"] == 0
            and technical["hidden_role_rows_before_freeze"] == 0
        ),
    }
    passed = bool(all(checks.values()))
    return {
        "stage": "technical_preflight",
        "passed": passed,
        "checks": checks,
        "technical": technical,
        "decision": (
            "technical_preflight_passed_full_oof_requires_separate_approval"
            if passed
            else "technical_blocker_not_scientific_negative_result"
        ),
        "rmse_used_as_gate": False,
    }


# %% [markdown]
# ## 9. Prediction freeze and late scientific readout


# %%
@dataclass(frozen=True)
class FrozenPrediction:
    frame: pd.DataFrame
    prediction_content_sha256: str
    branch_posterior_content_sha256: str
    schedule_content_sha256: str
    truth_rows_before_freeze: int
    hidden_rows_before_freeze: int


def freeze_predictions(
    prediction: pd.DataFrame,
    schedule: FrozenSchedule,
    ledger: RoleReadLedger,
) -> FrozenPrediction:
    forbidden = {"TVT", "true_tvt", "target", "error", "abs_error", "oracle"}
    if forbidden.intersection(prediction.columns):
        raise RuntimeError("truth/error/oracle columns entered pre-freeze predictions")
    if (
        ledger.suffix_truth_rows_before_freeze
        or ledger.error_rows_before_freeze
        or ledger.hidden_role_rows_before_freeze
    ):
        raise RuntimeError("late-read ledger is non-zero before prediction freeze")
    prediction_columns = [
        "well_id",
        "row_idx",
        CANDIDATE_NAME,
    ]
    branch_columns = [
        "well_id",
        "row_idx",
        "gamma_E",
        "gamma_H",
        "h_conditional_mean",
        "h_conditional_std",
        "joint_std",
        "expected_switch_probability",
        "docking_probability",
        "e_emission_loglik",
        "h_expected_emission_loglik",
        "mu_rate",
        "geometry_fallback",
    ]
    frozen = FrozenPrediction(
        frame=prediction.copy(),
        prediction_content_sha256=logical_frame_sha256(
            prediction,
            prediction_columns,
        ),
        branch_posterior_content_sha256=logical_frame_sha256(
            prediction,
            branch_columns,
        ),
        schedule_content_sha256=schedule.content_sha256,
        truth_rows_before_freeze=ledger.suffix_truth_rows_before_freeze,
        hidden_rows_before_freeze=ledger.hidden_role_rows_before_freeze,
    )
    ledger.freeze()
    return frozen


def resolve_exp263_cache_root(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.exp263_fixed_physical_candidate"))
    expected_manifest_sha = str(spec["expected_stage0_manifest_sha256"])
    root = find_project_root()
    candidates: list[Path] = []
    for raw in spec.get("cache_candidates", []):
        path = Path(str(raw))
        if not path.is_absolute():
            path = root / path
        if path.name == "cache_manifest.json":
            path = path.parent
        if (path / "cache_manifest.json").is_file():
            candidates.append(path)
    for search_root in (KAGGLE_INPUT_ROOT, Path("/tmp"), root):
        if search_root.exists():
            candidates.extend(
                manifest.parent for manifest in search_root.glob("**/cache_manifest.json")
            )
    matches = sorted(
        {
            str(path.resolve()): path
            for path in candidates
            if (path / "cache_manifest.json").is_file()
            and sha256_file(path / "cache_manifest.json") == expected_manifest_sha
        }.values(),
        key=str,
    )
    if not matches:
        raise FileNotFoundError("exp263 fixed candidate cache was not found")
    materialized = [
        path
        for path in matches
        if all(
            (path / "candidate_values" / candidate).is_dir() for candidate in FIXED_FORMULA_WEIGHTS
        )
    ]
    if not materialized:
        raise FileNotFoundError("exp263 cache lacks fixed-formula primitive partitions")
    signatures = {
        tuple(
            sorted(
                item.relative_to(path).as_posix()
                for candidate in FIXED_FORMULA_WEIGHTS
                for item in (path / "candidate_values" / candidate).glob("fold=*/part-*.parquet")
            )
        )
        for path in materialized
    }
    if len(signatures) != 1:
        raise ValueError("multiple structurally different exp263 caches were found")
    return materialized[0]


def load_candidate_partition_family(root: Path, candidate: str) -> pd.DataFrame:
    paths = sorted((root / "candidate_values" / candidate).glob("fold=*/part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"exp263 cache has no {candidate} partitions")
    manifest = json.loads((root / "cache_manifest.json").read_text())
    specifications = manifest.get("candidate_value_partitions", {}).get(candidate, [])
    expected_by_suffix = {
        "/".join(Path(str(item["path"])).parts[-3:]): item for item in specifications
    }
    if len(expected_by_suffix) != len(paths):
        raise ValueError(f"exp263 {candidate} partition count differs from manifest")
    for path in paths:
        suffix = "/".join(path.parts[-3:])
        specification = expected_by_suffix.get(suffix)
        if specification is None:
            raise ValueError(f"exp263 {candidate} unexpected partition: {suffix}")
        if sha256_file(path) != str(specification["file_sha256"]):
            raise ValueError(f"exp263 {candidate} partition SHA mismatch: {suffix}")
    columns = [
        "id",
        "well",
        "well_row_idx",
        "outer_fold",
        "md_since",
        "candidate_tvt",
    ]
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in paths],
        ignore_index=True,
    ).rename(
        columns={
            "id": "exp263_id",
            "well": "well_id",
            "well_row_idx": "row_idx",
            "outer_fold": "exp263_fold",
            "md_since": "exp263_md_since",
            "candidate_tvt": candidate,
        }
    )
    frame["well_id"] = frame["well_id"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int32)
    frame["exp263_fold"] = pd.to_numeric(
        frame["exp263_fold"],
        errors="raise",
    ).astype(np.int8)
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"exp263 {candidate} contains duplicate keys")
    return frame


def load_exp263_fixed_late(
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not ledger.prediction_frozen:
        raise RuntimeError("exp263 baseline may only be joined after prediction freeze")
    root = resolve_exp263_cache_root(config)
    identity = ["exp263_id", "exp263_fold", "exp263_md_since"]
    base = load_candidate_partition_family(root, "exp226_k16")
    for candidate in ("likpf_mean", "exact_hmm"):
        frame = load_candidate_partition_family(root, candidate)
        base = base.merge(
            frame[[*KEY_COLUMNS, *identity, candidate]],
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
            suffixes=("", f"_{candidate}"),
        )
        for column in identity:
            other = f"{column}_{candidate}"
            left = base[column].to_numpy()
            right = base.pop(other).to_numpy()
            if column == "exp263_md_since":
                equal = np.array_equal(left, right, equal_nan=True)
            else:
                equal = np.array_equal(left, right)
            if not equal:
                raise ValueError(f"exp263 {candidate} {column} identity mismatch")
    fixed = np.zeros(len(base), dtype=np.float64)
    for candidate, weight in FIXED_FORMULA_WEIGHTS.items():
        fixed += float(weight) * pd.to_numeric(
            base[candidate],
            errors="raise",
        ).to_numpy(np.float64)
    base["exp263_fixed"] = fixed
    return base[
        [
            *KEY_COLUMNS,
            *identity,
            "exp226_k16",
            "likpf_mean",
            "exact_hmm",
            "exp263_fixed",
        ]
    ], {
        "path": str(root),
        "manifest_sha256": sha256_file(root / "cache_manifest.json"),
        "rows": len(base),
        "wells": int(base["well_id"].nunique()),
    }


def load_truth_late(
    frozen: FrozenPrediction,
    raw_dir: Path,
    ledger: RoleReadLedger,
) -> pd.DataFrame:
    if not ledger.prediction_frozen:
        raise RuntimeError("suffix truth requires a frozen prediction")
    records: list[pd.DataFrame] = []
    for well, group in frozen.frame.groupby("well_id", sort=True):
        truth = pd.read_csv(
            raw_dir / f"{well}__horizontal_well.csv",
            usecols=["TVT"],
        )
        ledger.truth_late(f"suffix_truth:{well}", len(group))
        row_index = group["row_idx"].to_numpy(np.int64)
        if row_index.max(initial=-1) >= len(truth):
            raise ValueError(f"{well}: prediction row exceeds raw truth")
        records.append(
            pd.DataFrame(
                {
                    "well_id": str(well),
                    "row_idx": row_index.astype(np.int32),
                    "true_tvt": pd.to_numeric(
                        truth.iloc[row_index]["TVT"],
                        errors="raise",
                    ).to_numpy(np.float64),
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def load_hidden_like_late(
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not ledger.prediction_frozen:
        raise RuntimeError("hidden-like roles require a frozen prediction")
    spec = dict(get_nested(config, "data.hidden_like"))
    path = resolve_unique_file(
        filename=str(spec["filename"]),
        configured_candidates=spec.get("candidates", []),
        patterns=spec.get("patterns", []),
        label="exp115 hidden-like assignments",
    )
    actual_sha = sha256_file(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    role_columns = list(spec["role_columns"])
    frame = pd.read_csv(path, usecols=["well_id", *role_columns], dtype={"well_id": str})
    ledger.hidden_late("exp115_hidden_like", len(frame))
    for column in role_columns:
        frame[column] = frame[column].astype(str).eq("valid")
    return frame.drop_duplicates("well_id"), {
        "path": str(path),
        "raw_sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
    }


def attach_late_readout(
    frozen: FrozenPrediction,
    raw_dir: Path,
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth = load_truth_late(frozen, raw_dir, ledger)
    baseline, baseline_report = load_exp263_fixed_late(config, ledger)
    hidden, hidden_report = load_hidden_like_late(config, ledger)
    frame = (
        frozen.frame.merge(
            truth,
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        .merge(
            baseline,
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        .merge(
            hidden,
            on="well_id",
            how="left",
            validate="many_to_one",
        )
    )
    role_columns = list(get_nested(config, "data.hidden_like.role_columns"))
    for column in role_columns:
        frame[column] = frame[column].fillna(False).astype(bool)
    if len(frame) != len(frozen.frame):
        raise RuntimeError("late truth/baseline join lost prediction rows")
    if not np.array_equal(
        frame["fold"].to_numpy(np.int8),
        frame["exp263_fold"].to_numpy(np.int8),
    ):
        raise RuntimeError("exp394 and exp263 outer-fold identities differ")
    if not np.allclose(
        frame["md_since"].to_numpy(np.float64),
        frame["exp263_md_since"].to_numpy(np.float64),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError("exp394 and exp263 md_since identities differ")
    finite_columns = [
        CANDIDATE_NAME,
        "true_tvt",
        "exp263_fixed",
        "exp226_k16",
        "exact_hmm",
    ]
    if not np.isfinite(frame[finite_columns].to_numpy(np.float64)).all():
        raise RuntimeError("late readout contains non-finite score inputs")
    return frame, {
        "prediction_content_sha256_before_truth": (frozen.prediction_content_sha256),
        "branch_posterior_content_sha256_before_truth": (frozen.branch_posterior_content_sha256),
        "truth_rows_attached": len(truth),
        "baseline": baseline_report,
        "hidden_like": hidden_report,
        "ledger": to_jsonable(ledger.__dict__),
    }


def rmse(truth: Sequence[float], prediction: Sequence[float]) -> float:
    actual = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(actual) & np.isfinite(estimate)
    if not finite.any():
        return math.nan
    return float(np.sqrt(np.mean(np.square(actual[finite] - estimate[finite]))))


def paired_scope_metric(
    frame: pd.DataFrame,
    mask: np.ndarray,
    scope: str,
    scope_value: str,
) -> dict[str, Any]:
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError(f"metric scope is empty: {scope}/{scope_value}")
    truth = selected["true_tvt"].to_numpy(np.float64)
    candidate = rmse(truth, selected[CANDIDATE_NAME])
    baseline = rmse(truth, selected["exp263_fixed"])
    return {
        "scope": scope,
        "scope_value": scope_value,
        "rows": len(selected),
        "wells": int(selected["well_id"].nunique()),
        "candidate_rmse": candidate,
        "exp263_rmse": baseline,
        "delta_candidate_minus_exp263": candidate - baseline,
        "gain_vs_exp263": baseline - candidate,
    }


def build_scope_metrics(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    records = [
        paired_scope_metric(
            frame,
            np.ones(len(frame), dtype=bool),
            "overall",
            "all",
        )
    ]
    for fold in get_nested(config, "validation.expected_folds"):
        mask = frame["fold"].to_numpy(np.int64) == int(fold)
        records.append(paired_scope_metric(frame, mask, "fold", str(int(fold))))
    md_since = frame["md_since"].to_numpy(np.float64)
    records.append(
        paired_scope_metric(
            frame,
            (md_since >= 0.0) & (md_since < 250.0),
            "distance",
            "000_250",
        )
    )
    records.append(
        paired_scope_metric(
            frame,
            md_since >= 1000.0,
            "distance",
            "1000_plus",
        )
    )
    for column in get_nested(config, "data.hidden_like.role_columns"):
        records.append(
            paired_scope_metric(
                frame,
                frame[column].to_numpy(bool),
                "hidden_like",
                str(column),
            )
        )
    return pd.DataFrame(records)


def build_by_well_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well, group in frame.groupby("well_id", sort=True):
        truth = group["true_tvt"].to_numpy(np.float64)
        candidate = rmse(truth, group[CANDIDATE_NAME])
        baseline = rmse(truth, group["exp263_fixed"])
        rows.append(
            {
                "well_id": str(well),
                "fold": int(group["fold"].iloc[0]),
                "rows": len(group),
                "candidate_rmse": candidate,
                "exp263_rmse": baseline,
                "delta_candidate_minus_exp263": candidate - baseline,
            }
        )
    return pd.DataFrame(rows)


def build_reporting_references(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    truth = frame["true_tvt"].to_numpy(np.float64)
    expected = dict(get_nested(config, "validation.full_oof.reporting_references"))
    actual = {
        "exp226_oof_rmse_ft": rmse(truth, frame["exp226_k16"]),
        "exp209_oof_rmse_ft": rmse(truth, frame["exact_hmm"]),
        "exp263_oof_rmse_ft": rmse(truth, frame["exp263_fixed"]),
        "exp394_oof_rmse_ft": rmse(truth, frame[CANDIDATE_NAME]),
    }
    parity = {
        "exp226_absolute_difference_ft": abs(
            actual["exp226_oof_rmse_ft"] - float(expected["exp226_oof_rmse_ft"])
        ),
        "exp209_absolute_difference_ft": abs(
            actual["exp209_oof_rmse_ft"] - float(expected["exp209_oof_rmse_ft"])
        ),
    }
    return {
        "actual": actual,
        "saved_references": {
            **expected,
            "exp263_oof_rmse_ft": float(
                get_nested(
                    config,
                    "validation.full_oof.primary_baseline.saved_oof_rmse_ft",
                )
            ),
        },
        "saved_only_exp355_not_rerun": float(expected["exp355_oof_rmse_ft"]),
        "parity": parity,
    }


def persistent_offset_episodes(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = dict(get_nested(config, "audit.persistent_offset"))
    threshold = float(spec["error_threshold_ft"])
    minimum_rows = int(spec["minimum_consecutive_rows"])
    return_threshold = float(spec["return_threshold_ft"])
    horizons = [int(value) for value in spec["recovery_horizons_rows"]]
    candidates = [CANDIDATE_NAME, "exp263_fixed"]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for well, group in frame.groupby("well_id", sort=True):
            group = group.sort_values("row_idx", kind="mergesort")
            error = np.abs(
                group[candidate].to_numpy(np.float64) - group["true_tvt"].to_numpy(np.float64)
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
                row = {
                    "candidate": candidate,
                    "well_id": str(well),
                    "fold": int(group["fold"].iloc[0]),
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
    episode_columns = [
        "candidate",
        "well_id",
        "fold",
        "episode_start_row_idx",
        "confirmed_row_idx",
        "consecutive_rows_above_threshold",
        "peak_abs_error_ft",
        "recovery_rows_after_confirmation",
        *[f"recovered_within_{horizon}" for horizon in horizons],
    ]
    episodes = pd.DataFrame(rows, columns=episode_columns)
    summaries = []
    for candidate in candidates:
        group = (
            episodes.loc[episodes["candidate"].eq(candidate)] if not episodes.empty else episodes
        )
        row: dict[str, Any] = {"candidate": candidate, "episodes": len(group)}
        for horizon in horizons:
            column = f"recovered_within_{horizon}"
            row[f"{column}_count"] = int(group[column].sum()) if len(group) else 0
            row[f"{column}_rate"] = float(group[column].mean()) if len(group) else math.nan
        summaries.append(row)
    return episodes, pd.DataFrame(summaries)


def evaluate_promotion_gate(
    frame: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    runtime: pd.DataFrame,
    recovery: pd.DataFrame,
    frozen: FrozenPrediction,
    ledger: RoleReadLedger,
    runtime_seconds: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = dict(get_nested(config, "validation.full_oof.promotion_requires_all"))
    overall = scope_metrics.loc[
        scope_metrics["scope"].eq("overall") & scope_metrics["scope_value"].eq("all")
    ].iloc[0]
    folds = scope_metrics.loc[scope_metrics["scope"].eq("fold")]
    near = scope_metrics.loc[
        scope_metrics["scope"].eq("distance") & scope_metrics["scope_value"].eq("000_250")
    ].iloc[0]
    long_tail = scope_metrics.loc[
        scope_metrics["scope"].eq("distance") & scope_metrics["scope_value"].eq("1000_plus")
    ].iloc[0]
    hidden = scope_metrics.loc[scope_metrics["scope"].eq("hidden_like")].set_index("scope_value")
    spatial_column = "verification_like_spatial_role"
    typewell_column = "verification_like_typewell_purged_role"
    improved_folds = int((folds["gain_vs_exp263"] > 0.0).sum())
    by_well_delta = by_well["delta_candidate_minus_exp263"].to_numpy(np.float64)
    improved_or_equal_fraction = float(np.mean(by_well_delta <= 0.0))
    paired_p95 = float(np.quantile(by_well_delta, 0.95))
    worst_well_row = by_well.loc[by_well["delta_candidate_minus_exp263"].idxmax()]
    total_md = float(frame["delta_md"].sum())
    expected_switch_count = float(frame["expected_switch_probability"].sum())
    switches_per_1000 = expected_switch_count / total_md * 1000.0 if total_md > 0.0 else 0.0
    e_occupancy = float(frame["gamma_E"].mean())
    h_occupancy = float(frame["gamma_H"].mean())
    occupancy_low, occupancy_high = [
        float(value) for value in criteria["branch_occupancy_each_range"]
    ]
    recovery_wide = recovery.set_index("candidate")
    candidate_episodes = int(recovery_wide.loc[CANDIDATE_NAME, "episodes"])
    baseline_episodes = int(recovery_wide.loc["exp263_fixed", "episodes"])
    episode_delta = candidate_episodes - baseline_episodes
    recovery_column = "recovered_within_512_rate"
    candidate_recovery = float(recovery_wide.loc[CANDIDATE_NAME, recovery_column])
    baseline_recovery = float(recovery_wide.loc["exp263_fixed", recovery_column])
    if candidate_episodes == 0:
        recovery_delta = math.inf
    else:
        recovery_delta = candidate_recovery - baseline_recovery
    expected_baseline = float(
        get_nested(
            config,
            "validation.full_oof.primary_baseline.saved_oof_rmse_ft",
        )
    )
    baseline_parity_difference = abs(float(overall["exp263_rmse"]) - expected_baseline)
    technical = {
        "rows": len(frame),
        "expected_rows": int(get_nested(config, "validation.expected_rows")),
        "wells": int(frame["well_id"].nunique()),
        "expected_wells": int(get_nested(config, "validation.expected_wells")),
        "finite_prediction_coverage": float(
            np.isfinite(frame[CANDIDATE_NAME].to_numpy(np.float64)).mean()
        ),
        "duplicate_rows": int(frame.duplicated(list(KEY_COLUMNS)).sum()),
        "posterior_normalization_max_abs_error": float(
            runtime["posterior_normalization_max_abs_error"].max()
        ),
        "transition_row_sum_max_abs_error": float(
            runtime["transition_row_sum_max_abs_error"].max()
        ),
        "truth_rows_before_freeze": frozen.truth_rows_before_freeze,
        "hidden_rows_before_freeze": frozen.hidden_rows_before_freeze,
        "prediction_content_sha256": frozen.prediction_content_sha256,
        "branch_posterior_content_sha256": (frozen.branch_posterior_content_sha256),
        "schedule_content_sha256": frozen.schedule_content_sha256,
        "baseline_parity_absolute_difference_ft": baseline_parity_difference,
        "baseline_parity_tolerance_ft": 1.0e-5,
        "runtime_seconds": runtime_seconds,
        "runtime_limit_seconds": float(get_nested(config, "runtime.maximum_seconds")),
    }
    technical_checks = {
        "row_count": technical["rows"] == technical["expected_rows"],
        "well_count": technical["wells"] == technical["expected_wells"],
        "finite_prediction": technical["finite_prediction_coverage"] == 1.0,
        "unique_identity": technical["duplicate_rows"] == 0,
        "posterior_normalization": (technical["posterior_normalization_max_abs_error"] <= 1.0e-8),
        "transition_row_sum": (technical["transition_row_sum_max_abs_error"] <= 1.0e-10),
        "late_truth_guard": (
            technical["truth_rows_before_freeze"] == 0
            and technical["hidden_rows_before_freeze"] == 0
            and ledger.error_rows_before_freeze == 0
        ),
        "baseline_parity": (
            technical["baseline_parity_absolute_difference_ft"]
            <= technical["baseline_parity_tolerance_ft"]
        ),
        "runtime": (technical["runtime_seconds"] <= technical["runtime_limit_seconds"]),
    }
    scientific = {
        "candidate_rmse": float(overall["candidate_rmse"]),
        "exp263_rmse": float(overall["exp263_rmse"]),
        "gain_vs_exp263_ft": float(overall["gain_vs_exp263"]),
        "improved_folds": improved_folds,
        "near_000_250_delta_ft": float(near["delta_candidate_minus_exp263"]),
        "long_1000_plus_delta_ft": float(long_tail["delta_candidate_minus_exp263"]),
        "hidden_like_spatial_delta_ft": float(
            hidden.loc[spatial_column, "delta_candidate_minus_exp263"]
        ),
        "hidden_like_typewell_purged_delta_ft": float(
            hidden.loc[typewell_column, "delta_candidate_minus_exp263"]
        ),
        "improved_or_equal_well_fraction": improved_or_equal_fraction,
        "paired_by_well_delta_p95_ft": paired_p95,
        "worst_well_id": str(worst_well_row["well_id"]),
        "worst_well_regression_ft": float(worst_well_row["delta_candidate_minus_exp263"]),
        "e_branch_occupancy": e_occupancy,
        "h_branch_occupancy": h_occupancy,
        "expected_switch_count": expected_switch_count,
        "expected_switches_per_1000_md_ft": switches_per_1000,
        "persistent_episode_count": {
            "candidate": candidate_episodes,
            "exp263": baseline_episodes,
            "delta": episode_delta,
        },
        "recovery_within_512_rate": {
            "candidate": candidate_recovery,
            "exp263": baseline_recovery,
            "delta": recovery_delta,
        },
    }
    scientific_checks = {
        "minimum_rmse_gain": (
            scientific["gain_vs_exp263_ft"] >= float(criteria["minimum_rmse_gain_vs_exp263_ft"])
        ),
        "minimum_improved_folds": (improved_folds >= int(criteria["minimum_improved_folds"])),
        "near_000_250_guard": (
            scientific["near_000_250_delta_ft"]
            <= float(criteria["maximum_near_0_250_regression_ft"])
        ),
        "long_1000_plus_guard": (
            scientific["long_1000_plus_delta_ft"]
            <= float(criteria["maximum_1000_plus_regression_ft"])
        ),
        "hidden_like_spatial_guard": (
            scientific["hidden_like_spatial_delta_ft"]
            <= float(criteria["maximum_hidden_like_spatial_regression_ft"])
        ),
        "hidden_like_typewell_purged_guard": (
            scientific["hidden_like_typewell_purged_delta_ft"]
            <= float(criteria["maximum_hidden_like_typewell_purged_regression_ft"])
        ),
        "improved_or_equal_well_fraction": (
            improved_or_equal_fraction
            > float(criteria["minimum_improved_or_equal_well_fraction_exclusive"])
        ),
        "paired_by_well_p95": (
            paired_p95 <= float(criteria["maximum_paired_by_well_delta_p95_ft"])
        ),
        "worst_well": (
            scientific["worst_well_regression_ft"]
            <= float(criteria["maximum_worst_well_regression_ft"])
        ),
        "branch_occupancy": (
            occupancy_low <= e_occupancy <= occupancy_high
            and occupancy_low <= h_occupancy <= occupancy_high
        ),
        "expected_switch_rate": (
            switches_per_1000
            > float(criteria["minimum_expected_switches_per_1000_md_ft_exclusive"])
            and switches_per_1000 <= float(criteria["maximum_expected_switches_per_1000_md_ft"])
        ),
        "persistent_episode_count": (
            episode_delta <= int(criteria["maximum_persistent_offset_episode_count_delta"])
        ),
        "recovery_within_512": (
            candidate_episodes == 0
            or recovery_delta >= float(criteria["minimum_recovery_within_512_rate_delta"])
        ),
    }
    passed = bool(all(technical_checks.values()) and all(scientific_checks.values()))
    return {
        "passed": passed,
        "decision": (
            "promotion_supported_no_automatic_inference_or_submission"
            if passed
            else "promotion_rejected_no_parameter_rescue_blend_selector_inference_or_submission"
        ),
        "technical": technical,
        "technical_checks": technical_checks,
        "scientific": scientific,
        "scientific_checks": scientific_checks,
    }


# %% [markdown]
# ## 10. Kaggle CPU orchestration and generated artifacts


# %%
def require_kaggle_or_explicit_local() -> None:
    if not KAGGLE_WORKING_ROOT.is_dir() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "exp394 notebook execution must run first on Kaggle; "
            "local execution requires explicit user approval"
        )


def scientific_contract(config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "stage": stage,
        "candidate": CANDIDATE_NAME,
        "lineage": get_nested(config, "lineage"),
        "validation": get_nested(config, "validation"),
        "model": get_nested(config, "model"),
        "execution_counts": validate_scientific_contract(config),
        "truth_attached": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def run_technical_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_or_explicit_local()
    validate_scientific_contract(config, run_stage="technical_preflight")
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    started = time.perf_counter()
    artifacts = output_dir()
    ledger = RoleReadLedger()
    raw_dir = train_data_dir(config)
    raw_manifest, raw_report = validate_raw_identity(config, raw_dir)
    geometry, geometry_report = load_exp226_geometry(config, ledger)
    selection = select_preflight_wells(
        geometry,
        expected_folds=get_nested(config, "validation.expected_folds"),
    )
    selected_wells = selection["well_id"].astype(str).tolist()
    selected_geometry = geometry.loc[geometry["well_id"].astype(str).isin(selected_wells)]
    schedule = build_and_freeze_schedule(
        selected_geometry,
        raw_dir,
        config,
        ledger,
        require_full=False,
    )
    prediction, runtime = decode_scope(
        selected_wells,
        raw_dir,
        schedule,
        config,
        ledger,
    )
    selected_units = float(runtime["state_time_units"].sum())
    full_units = float(
        estimate_full_state_time_units(
            geometry,
            raw_dir,
            config,
            ledger,
        )
    )
    decode_seconds = float(runtime["elapsed_seconds"].sum())
    projected_runtime = (
        decode_seconds / selected_units * full_units * 1.15 if selected_units > 0.0 else math.inf
    )
    projected_peak_rss = float(runtime["peak_rss_gb"].max())
    frozen = freeze_predictions(prediction, schedule, ledger)
    contract = scientific_contract(config, "technical_preflight")
    gate = evaluate_preflight_gate(
        prediction,
        runtime,
        ledger,
        projected_runtime_seconds=projected_runtime,
        projected_peak_rss_gb=projected_peak_rss,
        input_identity_passed=True,
        config=config,
    )

    prefix = f"{OUTPUT_PREFIX}_preflight"
    reports = {
        "contract": write_json(
            artifacts / f"{prefix}_scientific_contract.json",
            contract,
        ),
        "raw_identity": write_csv(
            artifacts / f"{prefix}_raw_well_identity.csv",
            raw_manifest,
        ),
        "selection": write_csv(
            artifacts / f"{prefix}_well_ledger.csv",
            selection,
        ),
        "schedule": write_gzip_csv(
            artifacts / f"{prefix}_rate_schedule.csv.gz",
            schedule.rows,
        ),
        "predictions": write_gzip_csv(
            artifacts / f"{prefix}_predictions.csv.gz",
            prediction,
        ),
        "runtime": write_csv(
            artifacts / f"{prefix}_runtime.csv",
            runtime,
        ),
        "gate": write_json(
            artifacts / f"{prefix}_gate.json",
            gate,
        ),
    }
    runtime_seconds = time.perf_counter() - started
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "technical_preflight_passed_full_oof_not_approved"
            if gate["passed"]
            else "technical_preflight_blocked_not_scientific_negative"
        ),
        "stage": "technical_preflight",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime_seconds,
        "selected_wells": selected_wells,
        "rows": len(prediction),
        "wells": int(prediction["well_id"].nunique()),
        "projected_full_runtime_seconds": projected_runtime,
        "projected_peak_rss_gb": projected_peak_rss,
        "prediction_content_sha256": frozen.prediction_content_sha256,
        "branch_posterior_content_sha256": (frozen.branch_posterior_content_sha256),
        "schedule_content_sha256": frozen.schedule_content_sha256,
        "scientific_contract_sha256": contract["scientific_contract_sha256"],
        "input_identity": {
            "raw": raw_report,
            "geometry": geometry_report,
        },
        "gate": gate,
        "runtime_versions": runtime_versions(),
        "truth_rows_read": 0,
        "rmse_computed": False,
        "models": 0,
        "boosters": 0,
        "control_reruns": 0,
        "inference_enabled": False,
        "submission_created": False,
        "reports": reports,
    }
    summary_report = write_json(
        artifacts / f"{prefix}_summary.json",
        summary,
    )
    write_json(
        metrics_path(),
        {
            "experiment": EXPERIMENT_NAME,
            "route": "pf_beam",
            "status": summary["status"],
            "stage": "technical_preflight",
            "cv": None,
            "public_lb": None,
            "private_lb": None,
            "metric": "rmse",
            "technical_gate": gate,
            "prediction_sha256": frozen.prediction_content_sha256,
            "branch_posterior_sha256": (frozen.branch_posterior_content_sha256),
            "summary_path": summary_report["path"],
            "notes": (
                "16-well resource/numerical preflight only; RMSE is not a gate "
                "and full OOF still requires separate approval."
            ),
        },
    )
    print(json.dumps(to_jsonable(gate), indent=2, sort_keys=True))
    return summary


def load_preflight_pass_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    filename = f"{OUTPUT_PREFIX}_preflight_summary.json"
    path = resolve_unique_file(
        filename=filename,
        configured_candidates=[],
        patterns=[f"**/{filename}"],
        label="exp394 technical preflight PASS summary",
    )
    expected_sha = str(
        get_nested(
            config,
            "validation.full_oof.required_preflight_summary_sha256",
        )
    )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError("exp394 preflight summary SHA mismatch")
    summary = json.loads(path.read_text())
    if not bool(get_nested(summary, "gate.passed")):
        raise RuntimeError("exp394 full OOF requires a passing technical preflight")
    if summary.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("preflight summary belongs to another experiment")
    return {
        "path": str(path),
        "raw_sha256": actual_sha,
        "status": summary.get("status"),
        "gate_passed": True,
    }


def run_full_oof(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_or_explicit_local()
    validate_scientific_contract(config, run_stage="full_oof")
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    started = time.perf_counter()
    artifacts = output_dir()
    preflight_evidence = load_preflight_pass_evidence(config)
    ledger = RoleReadLedger()
    raw_dir = train_data_dir(config)
    raw_manifest, raw_report = validate_raw_identity(config, raw_dir)
    geometry, geometry_report = load_exp226_geometry(config, ledger)
    schedule = build_and_freeze_schedule(
        geometry,
        raw_dir,
        config,
        ledger,
        require_full=True,
    )
    wells = sorted(geometry["well_id"].astype(str).unique())
    prediction, runtime = decode_scope(
        wells,
        raw_dir,
        schedule,
        config,
        ledger,
    )
    frozen = freeze_predictions(prediction, schedule, ledger)
    prediction_freeze_seconds = time.perf_counter() - started
    frame, late_report = attach_late_readout(
        frozen,
        raw_dir,
        config,
        ledger,
    )
    scope_metrics = build_scope_metrics(frame, config)
    by_well = build_by_well_metrics(frame)
    reporting_references = build_reporting_references(frame, config)
    episodes, recovery = persistent_offset_episodes(frame, config)
    runtime_seconds = time.perf_counter() - started
    gate = evaluate_promotion_gate(
        frame,
        scope_metrics,
        by_well,
        runtime,
        recovery,
        frozen,
        ledger,
        runtime_seconds,
        config,
    )
    contract = scientific_contract(config, "full_oof")
    prefix = OUTPUT_PREFIX
    oof_columns = [
        "id",
        "well_id",
        "row_idx",
        "suffix_offset",
        "fold",
        "md_since",
        CANDIDATE_NAME,
    ]
    posterior_columns = [
        "id",
        "well_id",
        "row_idx",
        "gamma_E",
        "gamma_H",
        "h_conditional_mean",
        "h_conditional_std",
        "joint_std",
        "expected_switch_probability",
        "docking_probability",
        "e_emission_loglik",
        "h_expected_emission_loglik",
        "mu_rate",
        "geometry_fallback",
    ]
    reports = {
        "contract": write_json(
            artifacts / f"{prefix}_scientific_contract.json",
            contract,
        ),
        "raw_identity": write_csv(
            artifacts / f"{prefix}_raw_well_identity.csv",
            raw_manifest,
        ),
        "schedule": write_gzip_csv(
            artifacts / f"{prefix}_rate_schedule.csv.gz",
            schedule.rows,
        ),
        "oof_predictions": write_gzip_csv(
            artifacts / f"{prefix}_oof_predictions.csv.gz",
            prediction[oof_columns],
        ),
        "branch_posterior": write_gzip_csv(
            artifacts / f"{prefix}_branch_posterior.csv.gz",
            prediction[posterior_columns],
        ),
        "runtime": write_csv(
            artifacts / f"{prefix}_by_well_runtime.csv",
            runtime,
        ),
        "scope_metrics": write_csv(
            artifacts / f"{prefix}_scope_metrics.csv",
            scope_metrics,
        ),
        "by_well": write_csv(
            artifacts / f"{prefix}_by_well_metrics.csv",
            by_well,
        ),
        "reporting_references": write_json(
            artifacts / f"{prefix}_reporting_references.json",
            reporting_references,
        ),
        "episodes": write_csv(
            artifacts / f"{prefix}_recovery_episodes.csv",
            episodes,
        ),
        "recovery": write_csv(
            artifacts / f"{prefix}_recovery_summary.csv",
            recovery,
        ),
        "gate": write_json(
            artifacts / f"{prefix}_promotion_gate.json",
            gate,
        ),
    }
    overall = scope_metrics.loc[
        scope_metrics["scope"].eq("overall") & scope_metrics["scope_value"].eq("all")
    ].iloc[0]
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "full_oof_promotion_supported_no_inference"
            if gate["passed"]
            else "full_oof_rejected_no_rescue"
        ),
        "route": "pf_beam",
        "stage": "full_oof",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime_seconds,
        "prediction_freeze_seconds": prediction_freeze_seconds,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "cv": float(overall["candidate_rmse"]),
        "exp263_cv": float(overall["exp263_rmse"]),
        "gain_vs_exp263_ft": float(overall["gain_vs_exp263"]),
        "reporting_references": reporting_references,
        "gate": gate,
        "preflight_evidence": preflight_evidence,
        "input_identity": {
            "raw": raw_report,
            "geometry": geometry_report,
        },
        "late_attachment": late_report,
        "prediction_content_sha256": frozen.prediction_content_sha256,
        "branch_posterior_content_sha256": (frozen.branch_posterior_content_sha256),
        "schedule_content_sha256": frozen.schedule_content_sha256,
        "scientific_contract_sha256": contract["scientific_contract_sha256"],
        "runtime_versions": runtime_versions(),
        "execution_counts": validate_scientific_contract(config),
        "models": 0,
        "boosters": 0,
        "control_reruns": 0,
        "inference_enabled": False,
        "submission_created": False,
        "reports": reports,
    }
    summary_report = write_json(
        artifacts / f"{prefix}_summary.json",
        summary,
    )
    write_json(
        metrics_path(),
        {
            "experiment": EXPERIMENT_NAME,
            "route": "pf_beam",
            "status": summary["status"],
            "stage": "full_oof",
            "cv": summary["cv"],
            "public_lb": None,
            "private_lb": None,
            "metric": "rmse",
            "exp263_cv": summary["exp263_cv"],
            "gain_vs_exp263_ft": summary["gain_vs_exp263_ft"],
            "promotion_gate": gate,
            "schedule_sha256": frozen.schedule_content_sha256,
            "prediction_sha256": frozen.prediction_content_sha256,
            "branch_posterior_sha256": frozen.branch_posterior_content_sha256,
            "submission_sha256": None,
            "summary_path": summary_report["path"],
            "notes": (
                "One frozen 773-well switching-HMM candidate; no same-OOF "
                "rescue, blend, selector, inference, or submission."
            ),
        },
    )
    print(scope_metrics.to_string(index=False))
    print(json.dumps(to_jsonable(gate), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 11. Setup and configuration preview

# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_config()
    CONTRACT_COUNTS = validate_scientific_contract(CONFIG)
    print("Experiment:", get_nested(CONFIG, "experiment.name"))
    print("Route:", get_nested(CONFIG, "experiment.route"))
    print("Status:", get_nested(CONFIG, "experiment.status"))
    print("Parent:", get_nested(CONFIG, "lineage.parent"))
    print("Regimes:", get_nested(CONFIG, "model.regimes"))
    print("Execution counts:", json.dumps(CONTRACT_COUNTS, sort_keys=True))
    print(
        "Technical preflight enabled:",
        get_nested(CONFIG, "execution.run_technical_preflight"),
    )
    print("Full OOF enabled:", get_nested(CONFIG, "execution.run_full_oof"))
    print("Inference enabled:", get_nested(CONFIG, "execution.run_inference"))


# %% [markdown]
# ## 12. Fail-closed post-preflight entry point

# %%
if EXECUTE_NOTEBOOK:
    if bool(get_nested(CONFIG, "execution.run_full_oof")):
        EXP394_SUMMARY = run_full_oof(CONFIG)
        print(json.dumps(to_jsonable(EXP394_SUMMARY), indent=2, sort_keys=True))
    elif bool(get_nested(CONFIG, "execution.run_technical_preflight")):
        EXP394_SUMMARY = run_technical_preflight(CONFIG)
        print(json.dumps(to_jsonable(EXP394_SUMMARY), indent=2, sort_keys=True))
    else:
        print(
            "exp394 has no enabled run stage. Full OOF remains disabled "
            "pending separate approval."
        )

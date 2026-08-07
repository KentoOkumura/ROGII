# %% [markdown]
# # exp490 geometry-centered mean-reverting offset HMM
#
# This full-OOF notebook implements the same one frozen mechanism candidate. It keeps the
# exp357 Huber residual-offset exact HMM fixed and changes only the transition
# centers:
#
# `rho_t = 2 ** (-dMD_t / destination_K16_segment_MD_span)`
#
# `q_center_t = 0.998 * rho_t * q_(t-1)`
#
# `delta_center_t = rho_t * delta_(t-1) + q_t * dMD_t`
#
# Four operational well shards decode all 773 wells without reading role, fold,
# truth, saved exp357 predictions, or persistent-episode boundaries. The
# aggregate notebook accepts only SHA-pinned shard outputs and attaches truth
# after the global candidate root has been frozen. Inference and submission
# remain disabled.

# %% [markdown]
# ## Contents
# 1. Imports and immutable execution/scientific contracts
# 2. Notebook-safe paths, SHA helpers, and leakage ledger
# 3. Target-free full-well identity and exp226/raw inputs
# 4. K16 segment half-life and fixed exp357 Huber input preparation
# 5. Geometry-centered exact forward-backward decoder
# 6. Target-free per-well prediction and decoder-contract freeze
# 7. Truth-late saved-parent and persistent-episode readout
# 8. Reusable Stage 0 metric helpers
# 9. Four full-OOF shards and strict truth-late aggregate
# 10. Configuration preview and guarded execution

# %% [markdown]
# ## 1. Imports and immutable execution/scientific contracts

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import io
import json
import math
import os
import platform
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from numba import njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - import fallback for static inspection
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **kwargs: Any):  # type: ignore[misc]
        del args, kwargs

        def decorator(function):
            return function

        return decorator

    def prange(*args: int):  # type: ignore[misc]
        return range(*args)

    def set_num_threads(_: int) -> None:
        return None


EXPERIMENT_NAME = "exp490_geometry_centered_mean_reverting_offset_hmm"
RUN_KIND_OVERRIDE = "shard2"
FULL_SHARD_COUNT = 4
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
FORBIDDEN_CANDIDATE_COLUMNS = {
    "TVT",
    "tvt_true",
    "true_tvt_readout_only",
    "error",
    "abs_error",
    "role",
    "fold",
    "episode_id",
}


def get_nested(
    mapping: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    value: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_run_authorization: bool,
) -> dict[str, Any]:
    expected_stage0 = {
        "scientific_variants": 1,
        "candidate_hmm_well_runs": 32,
        "saved_parent_hmm_well_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    observed = dict(get_nested(config, "execution_contract.stage_0_if_separately_approved"))
    if observed != expected_stage0:
        raise ValueError(f"Stage 0 cost contract changed: {observed}")
    expected_stage1 = {
        "scientific_variants": 1,
        "candidate_hmm_well_runs": 773,
        "saved_parent_hmm_well_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "operational_cpu_shards": 4,
        "merge_hmm_well_runs": 0,
    }
    observed_stage1 = dict(
        get_nested(config, "execution_contract.stage_1_if_separately_approved")
    )
    if observed_stage1 != expected_stage1:
        raise ValueError(f"Stage 1 cost contract changed: {observed_stage1}")
    if bool(get_nested(config, "execution.run_stage_0", False)):
        raise ValueError("Stage 0 and Stage 1 cannot run together")
    if require_run_authorization and bool(
        get_nested(config, "execution.run_inference", False)
    ):
        raise ValueError("exp490 inference is disabled")
    if require_run_authorization and bool(
        get_nested(config, "execution.create_submission", False)
    ):
        raise ValueError("exp490 submission is disabled")
    if require_run_authorization:
        if not bool(get_nested(config, "execution.run_full_oof", False)):
            raise RuntimeError("exp490 full OOF run flag is disabled")
        if not bool(get_nested(config, "execution.full_run_approved", False)):
            raise RuntimeError("exp490 full OOF execution is not approved")
        if not bool(get_nested(config, "implementation.stage_1_override_approved", False)):
            raise RuntimeError("exp490 Stage 0 failure override is not approved")
    return {
        **expected_stage1,
        "reporting_folds": 5,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
        "control_rerun": False,
        "stage_0_all_pass": False,
        "execution_basis": "explicit_user_override_after_stage0_fail",
    }


def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "experiment": get_nested(config, "experiment.name") == EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route") == "pf_beam",
        "parent": (
            get_nested(config, "lineage.parent")
            == "exp357_exp226_huber_emission_independent_audit"
        ),
        "variant": get_nested(config, "model.mean_reversion.active_variants")
        == ["k16_segment_span_half_life"],
        "k16": int(get_nested(config, "model.k16_segment.count")) == 16,
        "half_life": (
            float(get_nested(config, "model.mean_reversion.half_life_segments"))
            == 1.0
        ),
        "destination_segment_owner": (
            get_nested(config, "model.k16_segment.boundary_transition_owner")
            == "destination_row_segment"
        ),
        "delta_min": float(get_nested(config, "model.hmm.delta_min_ft")) == -80.0,
        "delta_max": float(get_nested(config, "model.hmm.delta_max_ft")) == 80.0,
        "step": float(get_nested(config, "model.hmm.step_ft")) == 0.35,
        "rates": int(get_nested(config, "model.hmm.n_rates")) == 41,
        "rate_span": float(get_nested(config, "model.hmm.rate_span")) == 0.10,
        "sig_r": float(get_nested(config, "model.hmm.sig_r")) == 0.002,
        "sig_p": float(get_nested(config, "model.hmm.sig_p")) == 0.02,
        "momentum": float(get_nested(config, "model.hmm.mom")) == 0.998,
        "start_delta": float(get_nested(config, "model.hmm.start_delta_ft")) == 0.0,
        "start_sigma": float(get_nested(config, "model.hmm.start_sig_ft")) == 0.75,
        "start_rate": float(get_nested(config, "model.hmm.initial_rate")) == 0.0,
        "start_rate_sigma": (
            float(get_nested(config, "model.hmm.initial_rate_sig")) == 0.01
        ),
        "position_support": (
            int(get_nested(config, "model.hmm.position_kernel_support_cells")) == 5
        ),
        "posterior_mean_output": (
            get_nested(config, "model.hmm.output")
            == "forward_backward_posterior_mean"
        ),
        "huber": float(get_nested(config, "model.emission.huber_delta")) == 1.345,
        "likelihood_weight": float(get_nested(config, "model.emission.lam")) == 1.0,
        "sigma_mode": (
            get_nested(config, "model.emission.sigma_mode")
            == "known_prefix_residual_std"
        ),
        "sigma_clip": list(get_nested(config, "model.emission.sigma_clip"))
        == [10.0, 60.0],
        "missing_gr_policy": (
            get_nested(config, "model.emission.missing_gr_policy")
            == "interpolate_both_directions_then_typewell_mean"
        ),
        "additional_clip_disabled": (
            get_nested(config, "model.emission.additional_likelihood_clip")
            == "none"
        ),
        "hard_reset_disabled": not bool(
            get_nested(config, "model.mean_reversion.hard_reset")
        ),
        "gr_gate_disabled": not bool(
            get_nested(config, "model.mean_reversion.gr_confidence_gate")
        ),
        "full_oof_enabled": bool(get_nested(config, "execution.run_full_oof")),
        "stage1_override_approved": bool(
            get_nested(config, "implementation.stage_1_override_approved")
        ),
        "inference_disabled_or_explicitly_approved": (
            not bool(get_nested(config, "inference.enabled"))
            or bool(get_nested(config, "implementation.inference_override_approved", False))
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"exp490 scientific contract changed: {failed}")
    return {
        "experiment": EXPERIMENT_NAME,
        "parent": get_nested(config, "lineage.parent"),
        "route": "pf_beam",
        "stage": "stage_1_full_oof_four_target_free_shards_then_truth_late_merge",
        "candidate_variants": 1,
        "coordinate": {
            "prediction": "exp226_tvt_geop_t + residual_offset_t",
            "rate_center": "0.998 * rho_t * q_previous",
            "offset_center": (
                "rho_t * residual_offset_previous + q_t * positive_dMD_t"
            ),
        },
        "rho": {
            "formula": "2 ** (-dMD_t / destination_K16_segment_MD_span)",
            "boundary_owner": "destination_row_segment",
            "half_life_segments": 1.0,
        },
        "hmm": dict(get_nested(config, "model.hmm")),
        "emission": dict(get_nested(config, "model.emission")),
        "forbidden_rescue": list(get_nested(config, "model.forbidden_rescue")),
        "truth_attachment": get_nested(config, "validation.truth_attachment"),
        "control": "SHA-pinned saved exp357 prediction; zero HMM reruns",
    }


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
        else Path("/nonexistent-exp490-package-config"),
        KAGGLE_WORKING_ROOT / "config.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("exp490 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    selected = path or config_path()
    with selected.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("exp490 config must be a mapping")
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


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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


def array_bundle_sha256(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        to_jsonable(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
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


def write_deterministic_gzip_csv(
    path: Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=6,
            mtime=0,
        ) as compressed:
            compressed.write(csv_bytes)
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_gzip_sha256": sha256_file(path),
        "decompressed_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "logical_sha256": logical_frame_sha256(frame),
    }


def peak_rss_gib() -> float:
    rss_kib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return rss_kib / (1024.0**3)
    return rss_kib / (1024.0**2)


def runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
    }
    try:
        import numba

        versions["numba"] = numba.__version__
    except Exception:
        versions["numba"] = "unavailable"
    return versions


def resolve_bootstrap_asset(filename: str, local_path: str) -> Path:
    candidates = (
        KAGGLE_WORKING_ROOT / "assets" / filename,
        PACKAGE_DIR / "assets" / filename,
        find_project_root() / local_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if KAGGLE_INPUT_ROOT.is_dir():
        matches = sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"asset not found: {filename}")


def resolve_unique_file(
    *,
    filename: str,
    candidates: Sequence[str],
    patterns: Sequence[str],
) -> Path:
    root = find_project_root()
    matches: list[Path] = []
    for raw in candidates:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file() and candidate.name == filename:
            matches.append(candidate)
        elif candidate.is_dir() and (candidate / filename).is_file():
            matches.append(candidate / filename)
    search_roots = (root, KAGGLE_INPUT_ROOT, Path("/tmp"))
    for pattern in patterns:
        if pattern.startswith("/"):
            candidate = Path(pattern)
            if candidate.is_file():
                matches.append(candidate)
            continue
        for search_root in search_roots:
            if search_root.exists():
                matches.extend(search_root.glob(pattern))
    unique = sorted({path.resolve() for path in matches if path.is_file()})
    if not unique:
        raise FileNotFoundError(f"could not resolve {filename}")
    exact = [path for path in unique if path.name == filename]
    if not exact:
        raise FileNotFoundError(f"resolved files do not contain exact {filename}")
    return exact[0]


@dataclass
class LeakageLedger:
    expected_wells: int
    identity_rows: int = 0
    target_free_rows: int = 0
    frozen_wells: set[str] = field(default_factory=set)
    well_prediction_sha256: dict[str, str] = field(default_factory=dict)
    decoder_contract_sha256: str = ""
    forbidden_reads_before_freeze: dict[str, int] = field(default_factory=dict)
    post_freeze_reads: dict[str, int] = field(default_factory=dict)

    @property
    def all_predictions_frozen(self) -> bool:
        return len(self.frozen_wells) == self.expected_wells

    @property
    def all_frozen(self) -> bool:
        return self.all_predictions_frozen and bool(self.decoder_contract_sha256)

    def record_identity(self, rows: int) -> None:
        self.identity_rows += int(rows)

    def record_target_free(self, rows: int) -> None:
        self.target_free_rows += int(rows)

    def freeze_prediction(self, well: str, prediction_sha256: str) -> None:
        if well in self.frozen_wells:
            raise ValueError(f"duplicate candidate freeze for well={well}")
        if not prediction_sha256:
            raise ValueError("candidate prediction SHA is empty")
        self.frozen_wells.add(str(well))
        self.well_prediction_sha256[str(well)] = str(prediction_sha256)

    def freeze_decoder_contract(self, decoder_contract_sha256: str) -> None:
        if not self.all_predictions_frozen:
            raise RuntimeError("decoder contract cannot freeze before all predictions")
        if not decoder_contract_sha256:
            raise ValueError("decoder contract SHA is empty")
        self.decoder_contract_sha256 = str(decoder_contract_sha256)

    def record_forbidden(self, label: str, rows: int) -> None:
        if not self.all_frozen:
            self.forbidden_reads_before_freeze[label] = (
                self.forbidden_reads_before_freeze.get(label, 0) + int(rows)
            )
            raise RuntimeError(f"{label} was read before candidate/contract freeze")
        self.post_freeze_reads[label] = self.post_freeze_reads.get(label, 0) + int(rows)


# %% [markdown]
# ## 3. Identity-only fixed32 and target-free exp226/raw inputs
#
# Before freeze, the fixed manifest is opened with `usecols=["well"]` only.
# Candidate generation never receives role, fold, truth, error, or episode data.

# %%
def load_fixed32_identity(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[list[str], dict[str, Any]]:
    spec = get_nested(config, "data.stage_0_manifest")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_file(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(f"fixed32 manifest SHA changed: {observed}")
    identity = pd.read_csv(path, usecols=["well"], dtype={"well": str})
    if len(identity) != 32 or identity["well"].nunique() != 32:
        raise ValueError("fixed32 identity must contain 32 unique wells")
    wells = sorted(identity["well"].astype(str).tolist())
    ledger.record_identity(len(wells))
    return wells, {
        "path": str(path),
        "sha256": observed,
        "rows": len(wells),
        "identity_only_columns": ["well"],
    }


def load_fixed32_readout_after_freeze(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    if not ledger.all_frozen:
        raise RuntimeError("fixed32 readout requires complete freeze")
    spec = get_nested(config, "data.stage_0_manifest")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    frame = pd.read_csv(
        path,
        dtype={"well": str, "matched_persistent_well": str},
    )
    ledger.record_forbidden("manifest_role_fold_and_matching", len(frame))
    if len(frame) != 32 or frame["well"].nunique() != 32:
        raise ValueError("fixed32 readout coverage changed")
    if frame["role"].value_counts().to_dict() != {
        "persistent": 16,
        "control": 16,
    }:
        raise ValueError("fixed32 role counts changed")
    if set(frame["fold"].astype(int)) != {0, 1, 2, 3, 4}:
        raise ValueError("fixed32 no longer covers all folds")
    return frame.sort_values("well", kind="mergesort").reset_index(drop=True)


def load_exp226_target_free(
    config: Mapping[str, Any],
    target_wells: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_oof")
    path = resolve_unique_file(
        filename=str(spec["filename"]),
        candidates=[str(item) for item in spec["candidates"]],
        patterns=[str(item) for item in spec["patterns"]],
    )
    decompressed = sha256_decompressed_csv(path)
    if decompressed != str(spec["expected_decompressed_sha256"]):
        raise ValueError(f"exp226 decompressed SHA changed: {decompressed}")
    safe_columns = [str(item) for item in spec["safe_candidate_columns"]]
    forbidden = FORBIDDEN_CANDIDATE_COLUMNS.intersection(safe_columns)
    if forbidden:
        raise ValueError(f"exp226 candidate schema contains {sorted(forbidden)}")
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=safe_columns,
        dtype={"well_id": str},
        chunksize=200_000,
    ):
        selected = chunk.loc[chunk["well_id"].isin(target_wells)]
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise ValueError("exp226 target-free selection is empty")
    frame = pd.concat(pieces, ignore_index=True)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["suffix_offset"] = pd.to_numeric(
        frame["suffix_offset"], errors="raise"
    ).astype(np.int64)
    frame["tvt_geop"] = pd.to_numeric(
        frame["tvt_geop"], errors="raise"
    ).astype(np.float64)
    frame = frame.sort_values(
        ["well_id", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    if frame["well_id"].nunique() != len(target_wells):
        raise ValueError("exp226 target-free cache does not cover requested wells")
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 target-free keys are not unique")
    if not np.isfinite(frame["tvt_geop"]).all():
        raise ValueError("exp226 target-free geometry contains non-finite values")
    ledger.record_target_free(len(frame))
    return frame, {
        "path": str(path),
        "raw_gzip_sha256": sha256_file(path),
        "decompressed_sha256": decompressed,
        "selected_rows": len(frame),
        "selected_wells": int(frame["well_id"].nunique()),
        "candidate_columns": safe_columns,
    }


def train_data_dir(config: Mapping[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.is_dir():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT
            / "competitions"
            / "rogii-wellbore-geology-prediction"
            / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        first = next(KAGGLE_INPUT_ROOT.glob("**/*__horizontal_well.csv"), None)
        if first is not None:
            return first.parent
    return find_project_root() / str(get_nested(config, "data.train_dir"))


def load_target_free_well(
    well: str,
    raw_dir: Path,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
    typewell_path = raw_dir / f"{well}__typewell.csv"
    horizontal = pd.read_csv(
        horizontal_path,
        usecols=lambda column: str(column) != "TVT",
    )
    forbidden = FORBIDDEN_CANDIDATE_COLUMNS.intersection(horizontal.columns)
    if forbidden:
        raise ValueError(f"{well}: decoder input contains {sorted(forbidden)}")
    typewell = (
        pd.read_csv(typewell_path)
        .sort_values("TVT", kind="mergesort")
        .reset_index(drop=True)
    )
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError(f"{well}: typewell requires TVT and GR")
    ledger.record_target_free(len(horizontal) + len(typewell))
    return horizontal, typewell, {
        "horizontal_sha256": sha256_file(horizontal_path),
        "typewell_sha256": sha256_file(typewell_path),
    }


# %% [markdown]
# ## 4. K16 segment half-life and fixed exp357 Huber input preparation
#
# K16 uses the exact exp226 equal-row-count segmentation: edges are
# `linspace(0, unknown_rows, 17)` and row numbers 1..N belong to the destination
# segment selected with `searchsorted(..., side="left")`. The first unknown row
# uses the positive MD difference from the last known-prefix row.

# %%
def huber_log_likelihood(zscore: np.ndarray, delta: float) -> np.ndarray:
    z = np.asarray(zscore, dtype=np.float64)
    abs_z = np.abs(z)
    loss = np.where(
        abs_z <= float(delta),
        0.5 * z * z,
        float(delta) * abs_z - 0.5 * float(delta) ** 2,
    )
    return -loss


def robust_initial_rate(
    known_prefix: pd.DataFrame,
    window_rows: int = 30,
    *,
    min_valid_steps: int = 3,
    fallback_rate: float = 0.0,
) -> tuple[float, int, int]:
    tail = known_prefix.tail(int(window_rows))
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    dtvt = np.diff(tvt)
    dz = np.diff(z)
    dmd = np.diff(md)
    valid = np.isfinite(dtvt) & np.isfinite(dz) & np.isfinite(dmd) & (dmd > 0.0)
    valid_steps = int(valid.sum())
    if valid_steps < int(min_valid_steps):
        return float(fallback_rate), int(len(tail)), valid_steps
    rate = float(np.median((dtvt[valid] + dz[valid]) / dmd[valid]))
    if not np.isfinite(rate):
        rate = float(fallback_rate)
    return rate, int(len(tail)), valid_steps


def prefix_stats(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> tuple[float, float, int, int]:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    known_gr = (
        pd.to_numeric(known["GR"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    known_tvt = pd.to_numeric(
        known["TVT_input"], errors="raise"
    ).to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - typewell_at_known
    sigma = float(np.clip(np.std(residual), 10.0, 60.0))
    if not np.isfinite(sigma):
        raise ValueError("known-prefix GR residual sigma is not finite")
    initial_rate, effective_rows, valid_steps = robust_initial_rate(known, 30)
    return sigma, initial_rate, effective_rows, valid_steps


def k16_segment_half_life(
    unknown_md: np.ndarray,
    *,
    last_known_md: float,
    segment_count: int = 16,
) -> dict[str, np.ndarray]:
    md = np.asarray(unknown_md, dtype=np.float64)
    if md.ndim != 1 or len(md) < int(segment_count):
        raise ValueError("K16 requires at least one destination row per segment")
    if not np.isfinite(md).all() or not np.isfinite(last_known_md):
        raise ValueError("K16 MD values must be finite")
    dmd = np.diff(np.concatenate([[float(last_known_md)], md]))
    if not np.all(dmd > 0.0):
        raise ValueError("every transition-entering dMD must be strictly positive")
    edges = np.linspace(0.0, float(len(md)), int(segment_count) + 1)
    destination_rows = np.arange(1, len(md) + 1, dtype=np.float64)
    segment_id = np.clip(
        np.searchsorted(edges[1:], destination_rows, side="left"),
        0,
        int(segment_count) - 1,
    ).astype(np.int16)
    segment_span = np.bincount(
        segment_id.astype(np.int64),
        weights=dmd,
        minlength=int(segment_count),
    ).astype(np.float64)
    segment_rows = np.bincount(
        segment_id.astype(np.int64),
        minlength=int(segment_count),
    ).astype(np.int64)
    if np.any(segment_rows <= 0) or np.any(segment_span <= 0.0):
        raise ValueError("every K16 destination segment must be non-empty and positive")
    rho = np.power(2.0, -dmd / segment_span[segment_id])
    if not np.isfinite(rho).all() or np.any(rho <= 0.0) or np.any(rho > 1.0):
        raise ValueError("rho must be finite and in (0, 1]")
    cumulative = np.asarray(
        [
            np.prod(rho[segment_id == segment])
            for segment in range(int(segment_count))
        ],
        dtype=np.float64,
    )
    return {
        "dmd": dmd,
        "segment_id": segment_id,
        "segment_span": segment_span,
        "segment_rows": segment_rows,
        "rho": rho,
        "segment_cumulative_rho": cumulative,
        "edges": edges,
    }


def zero_state_geometry_identity(
    dmd: np.ndarray,
    rho: np.ndarray,
) -> dict[str, Any]:
    dmd_values = np.asarray(dmd, dtype=np.float64)
    rho_values = np.asarray(rho, dtype=np.float64)
    rate_center = 0.998 * rho_values * np.zeros_like(dmd_values)
    offset_center = rho_values * np.zeros_like(dmd_values) + rate_center * dmd_values
    maximum_abs_offset = float(np.max(np.abs(offset_center)))
    return {
        "pass": bool(maximum_abs_offset == 0.0),
        "maximum_abs_offset_ft": maximum_abs_offset,
        "rows": len(dmd_values),
    }


def prepare_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    geop_tvt: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if "TVT" in horizontal.columns:
        raise ValueError("candidate HMM preparation forbids horizontal truth")
    required_horizontal = {"MD", "Z", "GR", "TVT_input"}
    if not required_horizontal.issubset(horizontal.columns):
        missing = sorted(required_horizontal - set(horizontal.columns))
        raise ValueError(f"horizontal input is missing {missing}")
    typewell_tvt = pd.to_numeric(
        typewell["TVT"], errors="raise"
    ).to_numpy(np.float64)
    typewell_gr = (
        pd.to_numeric(typewell["GR"], errors="coerce")
        .ffill()
        .bfill()
        .to_numpy(np.float64)
    )
    if (
        len(typewell_tvt) < 2
        or not np.isfinite(typewell_tvt).all()
        or not np.isfinite(typewell_gr).all()
    ):
        raise ValueError("typewell TVT/GR must contain finite interpolation support")
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    evaluation = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4 or len(evaluation) == 0:
        raise ValueError("well needs four known rows and a non-empty unknown suffix")
    geop = np.asarray(geop_tvt, dtype=np.float64)
    if len(geop) != len(evaluation) or not np.isfinite(geop).all():
        raise ValueError("exp226 tvt_geop must align and be finite")

    sigma, initial_rate_diagnostic, rate_rows, valid_steps = prefix_stats(
        horizontal,
        typewell_tvt,
        typewell_gr,
    )
    last_known = known.iloc[-1]
    md = pd.to_numeric(evaluation["MD"], errors="raise").to_numpy(np.float64)
    segment = k16_segment_half_life(
        md,
        last_known_md=float(last_known["MD"]),
        segment_count=int(get_nested(config, "model.k16_segment.count")),
    )
    gr_fill = float(np.mean(typewell_gr))
    gr = (
        pd.to_numeric(horizontal["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)[evaluation.index]
    )
    hmm = get_nested(config, "model.hmm")
    step = float(hmm["step_ft"])
    grid = np.arange(
        float(hmm["delta_min_ft"]),
        float(hmm["delta_max_ft"]) + 0.5 * step,
        step,
        dtype=np.float64,
    )
    rates = np.linspace(
        -float(hmm["rate_span"]),
        float(hmm["rate_span"]),
        int(hmm["n_rates"]),
        dtype=np.float64,
    )
    absolute_tvt_states = geop[:, None] + grid[None, :]
    gr_grid = np.interp(absolute_tvt_states, typewell_tvt, typewell_gr)
    zscore = (gr[:, None] - gr_grid) / sigma
    emission = get_nested(config, "model.emission")
    emission_ll = huber_log_likelihood(
        zscore,
        float(emission["huber_delta"]),
    ).astype(np.float32)
    if not np.isfinite(emission_ll).all():
        raise ValueError("Huber emission contains non-finite values")
    start_delta = float(hmm["start_delta_ft"])
    native_typewell = (absolute_tvt_states >= float(typewell_tvt.min())) & (
        absolute_tvt_states <= float(typewell_tvt.max())
    )
    return {
        "emission_ll": emission_ll,
        "grid": grid,
        "rates": rates,
        "dmd": segment["dmd"],
        "segment_id": segment["segment_id"],
        "segment_span": segment["segment_span"],
        "segment_rows": segment["segment_rows"],
        "rho": segment["rho"],
        "segment_cumulative_rho": segment["segment_cumulative_rho"],
        "segment_edges": segment["edges"],
        "eval_index": evaluation.index.to_numpy(np.int64),
        "start_p": float((start_delta - grid[0]) / step),
        "prefix_rows": int(len(known)),
        "prefix_sigma": sigma,
        "prefix_initial_rate_diagnostic_only": initial_rate_diagnostic,
        "initial_rate_effective_rows": int(rate_rows),
        "initial_rate_valid_steps": int(valid_steps),
        "emission_finite_coverage": float(np.isfinite(emission_ll).mean()),
        "native_typewell_state_coverage": float(native_typewell.mean()),
        "zero_quantization_error_ft": float(np.min(np.abs(grid - start_delta))),
        "zero_state_identity": zero_state_geometry_identity(
            segment["dmd"],
            segment["rho"],
        ),
    }


# %% [markdown]
# ## 5. Geometry-centered exact forward-backward decoder
#
# The exp357 three-cell rate kernel and five-cell position kernel are retained.
# Their centers alone receive `rho_t`. Position transition probabilities are
# normalized on the same five-cell support for each source offset and
# destination rate, then used identically by forward and backward messages.

# %%
@njit(cache=True, nogil=True, parallel=True)
def _hmm2_fb_geometry_mean_reversion(
    emission_ll,
    dmd,
    rho,
    step,
    rates,
    sig_r,
    sig_p,
    start_p,
    start_sig,
    initial_rate,
    initial_rate_sig,
    lam,
    momentum,
):
    time_count, position_count = emission_ll.shape
    rate_count = len(rates)
    rate_step = rates[1] - rates[0]
    zero_position = 80.0 / step
    negative = np.float32(-1.0e18)
    alpha = np.full(
        (time_count, position_count, rate_count),
        negative,
        np.float32,
    )
    previous = np.full((position_count, rate_count), negative, np.float32)
    for position_index in range(position_count):
        delta_position = (position_index - start_p) * step
        position_prior = -0.5 * (delta_position / start_sig) ** 2
        if position_prior < -60.0:
            continue
        for rate_index in range(rate_count):
            delta_rate = (rates[rate_index] - initial_rate) / initial_rate_sig
            previous[position_index, rate_index] = np.float32(
                position_prior - 0.5 * delta_rate * delta_rate
            )

    after_rate = np.empty((position_count, rate_count), np.float32)
    current = np.empty((position_count, rate_count), np.float32)
    for time_index in range(time_count):
        row_rho = rho[time_index]
        sigma_rate_step = sig_r * np.sqrt(dmd[time_index])
        rate_variance_cells = (sigma_rate_step / rate_step) ** 2

        for position_index in prange(position_count):
            for destination_rate in range(rate_count):
                best = negative
                source_start = max(destination_rate - 1, 0)
                source_end = min(destination_rate + 1, rate_count - 1)
                for source_rate in range(source_start, source_end + 1):
                    mean_move = (
                        momentum * row_rho * rates[source_rate]
                        - rates[source_rate]
                    ) / rate_step
                    probability_plus = max(
                        0.5 * (rate_variance_cells + mean_move),
                        1.0e-12,
                    )
                    probability_minus = max(
                        0.5 * (rate_variance_cells - mean_move),
                        1.0e-12,
                    )
                    probability_total = probability_plus + probability_minus
                    if probability_total > 0.9:
                        probability_plus *= 0.9 / probability_total
                        probability_minus *= 0.9 / probability_total
                    move = destination_rate - source_rate
                    if move == -1:
                        log_probability = np.log(probability_minus)
                    elif move == 0:
                        log_probability = np.log(
                            1.0 - probability_plus - probability_minus
                        )
                    else:
                        log_probability = np.log(probability_plus)
                    value = (
                        previous[position_index, source_rate] + log_probability
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for source_rate in range(source_start, source_end + 1):
                        mean_move = (
                            momentum * row_rho * rates[source_rate]
                            - rates[source_rate]
                        ) / rate_step
                        probability_plus = max(
                            0.5 * (rate_variance_cells + mean_move),
                            1.0e-12,
                        )
                        probability_minus = max(
                            0.5 * (rate_variance_cells - mean_move),
                            1.0e-12,
                        )
                        probability_total = probability_plus + probability_minus
                        if probability_total > 0.9:
                            probability_plus *= 0.9 / probability_total
                            probability_minus *= 0.9 / probability_total
                        move = destination_rate - source_rate
                        if move == -1:
                            log_probability = np.log(probability_minus)
                        elif move == 0:
                            log_probability = np.log(
                                1.0 - probability_plus - probability_minus
                            )
                        else:
                            log_probability = np.log(probability_plus)
                        total += np.exp(
                            previous[position_index, source_rate]
                            + log_probability
                            - best
                        )
                    after_rate[position_index, destination_rate] = np.float32(
                        best + np.log(total)
                    )
                else:
                    after_rate[position_index, destination_rate] = negative

        sigma_position = max(sig_p, 0.35 * step)
        for destination_rate in prange(rate_count):
            rate_displacement_cells = (
                rates[destination_rate] * dmd[time_index] / step
            )
            for destination_position in range(position_count):
                lower_source = zero_position + (
                    destination_position
                    - 2.5
                    - zero_position
                    - rate_displacement_cells
                ) / row_rho
                upper_source = zero_position + (
                    destination_position
                    + 2.5
                    - zero_position
                    - rate_displacement_cells
                ) / row_rho
                source_start = max(int(np.ceil(lower_source)), 0)
                source_end = min(int(np.ceil(upper_source)) - 1, position_count - 1)
                best = negative
                for source_position in range(source_start, source_end + 1):
                    center = (
                        zero_position
                        + row_rho * (source_position - zero_position)
                        + rate_displacement_cells
                    )
                    base = int(np.floor(center + 0.5))
                    kernel_index = destination_position - (base - 2)
                    if kernel_index < 0 or kernel_index >= 5:
                        continue
                    kernel = np.empty(5)
                    for offset in range(5):
                        residual = (base - 2 + offset - center) * step
                        kernel[offset] = -0.5 * (
                            residual / sigma_position
                        ) ** 2
                    kernel_max = np.max(kernel)
                    log_normalizer = kernel_max + np.log(
                        np.sum(np.exp(kernel - kernel_max))
                    )
                    value = (
                        after_rate[source_position, destination_rate]
                        + kernel[kernel_index]
                        - log_normalizer
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for source_position in range(source_start, source_end + 1):
                        center = (
                            zero_position
                            + row_rho * (source_position - zero_position)
                            + rate_displacement_cells
                        )
                        base = int(np.floor(center + 0.5))
                        kernel_index = destination_position - (base - 2)
                        if kernel_index < 0 or kernel_index >= 5:
                            continue
                        kernel = np.empty(5)
                        for offset in range(5):
                            residual = (base - 2 + offset - center) * step
                            kernel[offset] = -0.5 * (
                                residual / sigma_position
                            ) ** 2
                        kernel_max = np.max(kernel)
                        log_normalizer = kernel_max + np.log(
                            np.sum(np.exp(kernel - kernel_max))
                        )
                        total += np.exp(
                            after_rate[source_position, destination_rate]
                            + kernel[kernel_index]
                            - log_normalizer
                            - best
                        )
                    current[destination_position, destination_rate] = np.float32(
                        best
                        + np.log(total)
                        + lam * emission_ll[time_index, destination_position]
                    )
                else:
                    current[destination_position, destination_rate] = negative

        for position_index in range(position_count):
            for rate_index in range(rate_count):
                alpha[time_index, position_index, rate_index] = current[
                    position_index, rate_index
                ]
                previous[position_index, rate_index] = current[
                    position_index, rate_index
                ]

    last_best = np.max(alpha[time_count - 1])
    last_total = np.sum(np.exp(alpha[time_count - 1] - last_best))
    log_likelihood = float(last_best) + np.log(last_total)
    position_posterior = np.zeros((time_count, position_count), np.float64)
    beta_next = np.zeros((position_count, rate_count), np.float32)
    values = alpha[time_count - 1] + beta_next
    best = np.max(values)
    total = 0.0
    for position_index in range(position_count):
        accumulator = 0.0
        for rate_index in range(rate_count):
            accumulator += np.exp(values[position_index, rate_index] - best)
        position_posterior[time_count - 1, position_index] = accumulator
        total += accumulator
    position_posterior[time_count - 1] /= total

    beta_current = np.empty((position_count, rate_count), np.float32)
    beta_after_position = np.empty((position_count, rate_count), np.float32)
    for time_index in range(time_count - 1, 0, -1):
        row_rho = rho[time_index]
        sigma_position = max(sig_p, 0.35 * step)
        for source_position in prange(position_count):
            for destination_rate in range(rate_count):
                rate_displacement_cells = (
                    rates[destination_rate] * dmd[time_index] / step
                )
                center = (
                    zero_position
                    + row_rho * (source_position - zero_position)
                    + rate_displacement_cells
                )
                base = int(np.floor(center + 0.5))
                kernel = np.empty(5)
                for offset in range(5):
                    residual = (base - 2 + offset - center) * step
                    kernel[offset] = -0.5 * (residual / sigma_position) ** 2
                kernel_max = np.max(kernel)
                log_normalizer = kernel_max + np.log(
                    np.sum(np.exp(kernel - kernel_max))
                )
                best = negative
                for offset in range(5):
                    destination_position = base - 2 + offset
                    if 0 <= destination_position < position_count:
                        value = (
                            kernel[offset]
                            - log_normalizer
                            + lam
                            * emission_ll[time_index, destination_position]
                            + beta_next[destination_position, destination_rate]
                        )
                        if value > best:
                            best = value
                if best > negative / 2:
                    total = 0.0
                    for offset in range(5):
                        destination_position = base - 2 + offset
                        if 0 <= destination_position < position_count:
                            total += np.exp(
                                kernel[offset]
                                - log_normalizer
                                + lam
                                * emission_ll[time_index, destination_position]
                                + beta_next[destination_position, destination_rate]
                                - best
                            )
                    beta_after_position[
                        source_position, destination_rate
                    ] = np.float32(best + np.log(total))
                else:
                    beta_after_position[
                        source_position, destination_rate
                    ] = negative

        sigma_rate_step = sig_r * np.sqrt(dmd[time_index])
        rate_variance_cells = (sigma_rate_step / rate_step) ** 2
        for source_position in prange(position_count):
            for source_rate in range(rate_count):
                mean_move = (
                    momentum * row_rho * rates[source_rate]
                    - rates[source_rate]
                ) / rate_step
                probability_plus = max(
                    0.5 * (rate_variance_cells + mean_move),
                    1.0e-12,
                )
                probability_minus = max(
                    0.5 * (rate_variance_cells - mean_move),
                    1.0e-12,
                )
                probability_total = probability_plus + probability_minus
                if probability_total > 0.9:
                    probability_plus *= 0.9 / probability_total
                    probability_minus *= 0.9 / probability_total
                best = negative
                destination_start = max(source_rate - 1, 0)
                destination_end = min(source_rate + 1, rate_count - 1)
                for destination_rate in range(
                    destination_start, destination_end + 1
                ):
                    move = destination_rate - source_rate
                    if move == -1:
                        log_probability = np.log(probability_minus)
                    elif move == 0:
                        log_probability = np.log(
                            1.0 - probability_plus - probability_minus
                        )
                    else:
                        log_probability = np.log(probability_plus)
                    value = (
                        log_probability
                        + beta_after_position[source_position, destination_rate]
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for destination_rate in range(
                        destination_start, destination_end + 1
                    ):
                        move = destination_rate - source_rate
                        if move == -1:
                            log_probability = np.log(probability_minus)
                        elif move == 0:
                            log_probability = np.log(
                                1.0 - probability_plus - probability_minus
                            )
                        else:
                            log_probability = np.log(probability_plus)
                        total += np.exp(
                            log_probability
                            + beta_after_position[
                                source_position, destination_rate
                            ]
                            - best
                        )
                    beta_current[source_position, source_rate] = np.float32(
                        best + np.log(total)
                    )
                else:
                    beta_current[source_position, source_rate] = negative

        values = alpha[time_index - 1] + beta_current
        best = np.max(values)
        total = 0.0
        for position_index in range(position_count):
            accumulator = 0.0
            for rate_index in range(rate_count):
                accumulator += np.exp(values[position_index, rate_index] - best)
            position_posterior[time_index - 1, position_index] = accumulator
            total += accumulator
        position_posterior[time_index - 1] /= total
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                beta_next[position_index, rate_index] = beta_current[
                    position_index, rate_index
                ]

    maximum_normalization_error = 0.0
    for time_index in range(time_count):
        row_error = abs(np.sum(position_posterior[time_index]) - 1.0)
        if row_error > maximum_normalization_error:
            maximum_normalization_error = row_error
    return position_posterior, log_likelihood, maximum_normalization_error


def run_geometry_mean_reverting_hmm(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    geop_tvt: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = prepare_hmm_inputs(horizontal, typewell, geop_tvt, config)
    hmm = get_nested(config, "model.hmm")
    emission = get_nested(config, "model.emission")
    posterior, log_likelihood, normalization_error = (
        _hmm2_fb_geometry_mean_reversion(
            prepared["emission_ll"],
            prepared["dmd"].astype(np.float64),
            prepared["rho"].astype(np.float64),
            float(hmm["step_ft"]),
            prepared["rates"].astype(np.float64),
            float(hmm["sig_r"]),
            float(hmm["sig_p"]),
            float(prepared["start_p"]),
            float(hmm["start_sig_ft"]),
            float(hmm["initial_rate"]),
            float(hmm["initial_rate_sig"]),
            float(emission["lam"]),
            float(hmm["mom"]),
        )
    )
    grid = prepared["grid"]
    delta_mean = posterior @ grid
    variance = posterior @ (grid**2) - delta_mean**2
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    del posterior
    gc.collect()
    return {
        **prepared,
        "delta_mean": np.asarray(delta_mean, dtype=np.float64),
        "mean": np.asarray(geop_tvt, dtype=np.float64)
        + np.asarray(delta_mean, dtype=np.float64),
        "std": np.asarray(standard_deviation, dtype=np.float64),
        "log_likelihood": float(log_likelihood),
        "maximum_posterior_normalization_error": float(normalization_error),
    }


# %% [markdown]
# ## 6. Target-free per-well prediction and decoder-contract freeze

# %%
@dataclass
class FrozenWell:
    well: str
    row_idx: np.ndarray
    suffix_offset: np.ndarray
    tvt_geop: np.ndarray
    prediction: np.ndarray
    delta_mean: np.ndarray
    posterior_std: np.ndarray
    dmd: np.ndarray
    segment_id: np.ndarray
    segment_span: np.ndarray
    segment_rows: np.ndarray
    segment_cumulative_rho: np.ndarray
    rho: np.ndarray
    prefix_rows: int
    prefix_sigma: float
    maximum_posterior_normalization_error: float
    zero_state_identity_pass: bool
    emission_finite_coverage: float
    elapsed_seconds: float
    prediction_sha256: str
    raw_input_sha256: dict[str, str]


def freeze_target_free_well(
    *,
    well: str,
    raw_dir: Path,
    exp226_well: pd.DataFrame,
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> FrozenWell:
    horizontal, typewell, raw_sha = load_target_free_well(well, raw_dir, ledger)
    evaluation_index = horizontal.index[
        horizontal["TVT_input"].isna()
    ].to_numpy(np.int64)
    source = exp226_well.sort_values("row_idx", kind="mergesort")
    source_rows = source["row_idx"].to_numpy(np.int64)
    suffix_offset = source["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(source_rows, evaluation_index):
        raise ValueError(f"{well}: exp226/raw row identity mismatch")
    if not np.array_equal(suffix_offset, np.arange(len(source), dtype=np.int64)):
        raise ValueError(f"{well}: exp226 suffix offsets are not contiguous from zero")
    geop = source["tvt_geop"].to_numpy(np.float64)
    started = time.perf_counter()
    result = run_geometry_mean_reverting_hmm(
        horizontal,
        typewell,
        geop,
        config,
    )
    elapsed = float(time.perf_counter() - started)
    finite_arrays = (
        geop,
        result["mean"],
        result["delta_mean"],
        result["std"],
        result["dmd"],
        result["rho"],
    )
    if not all(np.isfinite(array).all() for array in finite_arrays):
        raise ValueError(f"{well}: non-finite candidate output")
    prediction_sha = array_bundle_sha256(
        row_idx=source_rows,
        suffix_offset=suffix_offset,
        tvt_geop=geop,
        prediction=np.asarray(result["mean"], dtype=np.float64),
        delta_mean=np.asarray(result["delta_mean"], dtype=np.float64),
        posterior_std=np.asarray(result["std"], dtype=np.float64),
        dmd=np.asarray(result["dmd"], dtype=np.float64),
        segment_id=np.asarray(result["segment_id"], dtype=np.int16),
        rho=np.asarray(result["rho"], dtype=np.float64),
    )
    ledger.freeze_prediction(well, prediction_sha)
    return FrozenWell(
        well=str(well),
        row_idx=source_rows,
        suffix_offset=suffix_offset,
        tvt_geop=geop,
        prediction=np.asarray(result["mean"], dtype=np.float64),
        delta_mean=np.asarray(result["delta_mean"], dtype=np.float64),
        posterior_std=np.asarray(result["std"], dtype=np.float64),
        dmd=np.asarray(result["dmd"], dtype=np.float64),
        segment_id=np.asarray(result["segment_id"], dtype=np.int16),
        segment_span=np.asarray(result["segment_span"], dtype=np.float64),
        segment_rows=np.asarray(result["segment_rows"], dtype=np.int64),
        segment_cumulative_rho=np.asarray(
            result["segment_cumulative_rho"], dtype=np.float64
        ),
        rho=np.asarray(result["rho"], dtype=np.float64),
        prefix_rows=int(result["prefix_rows"]),
        prefix_sigma=float(result["prefix_sigma"]),
        maximum_posterior_normalization_error=float(
            result["maximum_posterior_normalization_error"]
        ),
        zero_state_identity_pass=bool(result["zero_state_identity"]["pass"]),
        emission_finite_coverage=float(result["emission_finite_coverage"]),
        elapsed_seconds=elapsed,
        prediction_sha256=prediction_sha,
        raw_input_sha256=raw_sha,
    )


def prediction_frame(frozen_wells: Sequence[FrozenWell]) -> pd.DataFrame:
    pieces = []
    for item in frozen_wells:
        pieces.append(
            pd.DataFrame(
                {
                    "id": [
                        f"{item.well}_{int(row)}" for row in item.row_idx
                    ],
                    "well": item.well,
                    "row_idx": item.row_idx,
                    "suffix_offset": item.suffix_offset,
                    "tvt_geop": item.tvt_geop,
                    "geometry_mean_reverting_hmm": item.prediction,
                    "geometry_mean_reverting_delta_mean": item.delta_mean,
                    "geometry_mean_reverting_hmm_std": item.posterior_std,
                    "dmd": item.dmd,
                    "k16_segment_id": item.segment_id,
                    "k16_segment_span": item.segment_span[item.segment_id],
                    "rho": item.rho,
                }
            )
        )
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )


def segment_contract_frame(
    frozen_wells: Sequence[FrozenWell],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in frozen_wells:
        for segment in range(len(item.segment_span)):
            mask = item.segment_id == segment
            rows.append(
                {
                    "well": item.well,
                    "k16_segment_id": segment,
                    "destination_start_suffix_offset": int(
                        item.suffix_offset[mask][0]
                    ),
                    "destination_end_suffix_offset_exclusive": int(
                        item.suffix_offset[mask][-1] + 1
                    ),
                    "rows": int(mask.sum()),
                    "dmd_span": float(item.segment_span[segment]),
                    "rho_product": float(
                        item.segment_cumulative_rho[segment]
                    ),
                    "rho_product_abs_error_vs_half": float(
                        abs(item.segment_cumulative_rho[segment] - 0.5)
                    ),
                    "minimum_rho": float(np.min(item.rho[mask])),
                    "maximum_rho": float(np.max(item.rho[mask])),
                    "positive_dmd": bool(np.all(item.dmd[mask] > 0.0)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["well", "k16_segment_id"], kind="mergesort"
    )


def build_decoder_manifest(
    config: Mapping[str, Any],
    scientific_contract: Mapping[str, Any],
    frozen_wells: Sequence[FrozenWell],
    segment_contract: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_1_target_free_full_shard",
        "scientific_contract": scientific_contract,
        "scientific_contract_sha256": hashlib.sha256(
            stable_json_bytes(scientific_contract)
        ).hexdigest(),
        "state_order": {
            "offset_grid": "ascending_minus80_to_plus80_step0p35",
            "rate_grid": "ascending_minus0p10_to_plus0p10_41_states",
            "well": "lexicographic",
            "row": "ascending_raw_row_idx",
            "segment": "ascending_0_to_15",
        },
        "transition_order": [
            "destination-row K16 segment and positive dMD",
            "rho_t = 2 ** (-dMD_t / destination segment dMD span)",
            "three-cell rate transition centered at 0.998 * rho_t * q_previous",
            "five-cell offset transition centered at rho_t * delta_previous + q_t * dMD_t",
            "Huber delta=1.345 GR emission",
            "forward-backward posterior mean",
        ],
        "segment_contract_rows": len(segment_contract),
        "segment_contract_logical_sha256": logical_frame_sha256(segment_contract),
        "raw_input_sha256_by_well": {
            item.well: item.raw_input_sha256 for item in frozen_wells
        },
        "prediction_sha256_by_well": {
            item.well: item.prediction_sha256 for item in frozen_wells
        },
        "truth_role_fold_episode_available": False,
        "control_hmm_reruns": 0,
    }


# %% [markdown]
# ## 7. Truth-late saved-parent and persistent-episode readout

# %%
def load_saved_exp357_after_freeze(
    config: Mapping[str, Any],
    target_wells: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not ledger.all_frozen:
        raise RuntimeError("saved exp357 control requires complete freeze")
    spec = get_nested(config, "data.exp357_saved_parent")
    path = resolve_unique_file(
        filename=str(spec["filename"]),
        candidates=[str(item) for item in spec["candidates"]],
        patterns=[str(item) for item in spec["patterns"]],
    )
    decompressed = sha256_decompressed_csv(path)
    if decompressed != str(spec["expected_decompressed_sha256"]):
        raise ValueError(f"saved exp357 decompressed SHA changed: {decompressed}")
    prediction_column = str(spec["prediction_column"])
    truth_column = str(spec["truth_column"])
    columns = [
        "id",
        "well",
        "row_idx",
        "fold",
        "tvt_geop",
        prediction_column,
        truth_column,
        "exp226_pred",
        "md_since",
    ]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype={"id": str, "well": str},
        chunksize=200_000,
    ):
        selected = chunk.loc[chunk["well"].isin(target_wells)]
        if not selected.empty:
            pieces.append(selected)
    frame = pd.concat(pieces, ignore_index=True).rename(
        columns={
            prediction_column: "exp357_parent_prediction",
            truth_column: "true_tvt_readout_only",
        }
    )
    ledger.record_forbidden("saved_exp357_prediction_fold_and_truth", len(frame))
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    numeric = [
        "fold",
        "tvt_geop",
        "exp357_parent_prediction",
        "true_tvt_readout_only",
        "exp226_pred",
        "md_since",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values(
        ["well", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    if frame["well"].nunique() != len(target_wells) or frame.duplicated(["well", "row_idx"]).any():
        raise ValueError("saved exp357 coverage or identity changed")
    if not np.isfinite(frame[numeric].to_numpy(np.float64)).all():
        raise ValueError("saved exp357 readout is non-finite")
    return frame, {
        "path": str(path),
        "raw_gzip_sha256": sha256_file(path),
        "decompressed_sha256": decompressed,
        "declared_candidate_content_sha256": str(
            spec["expected_content_sha256"]
        ),
        "selected_rows": len(frame),
        "selected_wells": int(frame["well"].nunique()),
        "loaded_after_freeze": True,
    }


def load_persistent_episodes_after_freeze(
    config: Mapping[str, Any],
    persistent_wells: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not ledger.all_frozen:
        raise RuntimeError("persistent episodes require complete freeze")
    spec = get_nested(config, "data.persistent_episodes")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_file(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError(f"persistent episode SHA changed: {observed}")
    columns = [
        "episode_id",
        "well",
        "start_row_idx",
        "end_row_idx_exclusive",
    ]
    frame = pd.read_csv(path, usecols=columns, dtype={"well": str})
    frame = frame.loc[frame["well"].isin(persistent_wells)].copy()
    ledger.record_forbidden("persistent_episode_boundaries", len(frame))
    if frame.empty or set(frame["well"]) != persistent_wells:
        raise ValueError("fixed persistent wells are missing episode boundaries")
    return frame.sort_values(
        ["well", "start_row_idx"], kind="mergesort"
    ).reset_index(drop=True), {
        "path": str(path),
        "sha256": observed,
        "selected_rows": len(frame),
        "selected_wells": int(frame["well"].nunique()),
        "loaded_after_freeze": True,
    }


def attach_truth_late_readout(
    candidate: pd.DataFrame,
    parent: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    merged = candidate.merge(
        parent,
        on=["id", "well", "row_idx"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_saved"),
    )
    if len(merged) != len(candidate):
        raise ValueError("saved exp357 join changed candidate row count")
    if not np.allclose(
        merged["tvt_geop"].to_numpy(np.float64),
        merged["tvt_geop_saved"].to_numpy(np.float64),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("candidate and saved exp357 geometry identities differ")
    readout = manifest[["well", "role", "fold", "prefix_rows", "suffix_rows"]]
    merged = merged.merge(
        readout,
        on="well",
        how="left",
        validate="many_to_one",
        suffixes=("_saved", "_manifest"),
    )
    if not np.array_equal(
        merged["fold_saved"].to_numpy(np.int64),
        merged["fold_manifest"].to_numpy(np.int64),
    ):
        raise ValueError("saved exp357 and fixed32 folds differ")
    merged["fold"] = merged.pop("fold_saved").astype(np.int8)
    merged = merged.drop(columns=["fold_manifest", "tvt_geop_saved"])
    if merged["role"].isna().any():
        raise ValueError("fixed32 role join is incomplete")
    merged["candidate_error"] = (
        merged["geometry_mean_reverting_hmm"]
        - merged["true_tvt_readout_only"]
    )
    merged["parent_error"] = (
        merged["exp357_parent_prediction"]
        - merged["true_tvt_readout_only"]
    )
    if not np.isfinite(
        merged[
            [
                "candidate_error",
                "parent_error",
                "geometry_mean_reverting_hmm",
                "exp357_parent_prediction",
                "true_tvt_readout_only",
            ]
        ].to_numpy(np.float64)
    ).all():
        raise ValueError("truth-late readout contains non-finite values")
    return merged.sort_values(
        ["well", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)


def rmse(error: np.ndarray | pd.Series) -> float:
    values = np.asarray(error, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("RMSE requires non-empty finite errors")
    return float(np.sqrt(np.mean(np.square(values))))


def build_well_metrics(readout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, part in readout.groupby("well", sort=True):
        parent_rmse = rmse(part["parent_error"])
        candidate_rmse = rmse(part["candidate_error"])
        rows.append(
            {
                "well": str(well),
                "role": str(part["role"].iloc[0]),
                "fold": int(part["fold"].iloc[0]),
                "rows": len(part),
                "parent_rmse_ft": parent_rmse,
                "candidate_rmse_ft": candidate_rmse,
                "candidate_minus_parent_rmse_ft": candidate_rmse - parent_rmse,
            }
        )
    return pd.DataFrame(rows)


def contiguous_episode_count(
    error: np.ndarray,
    *,
    threshold_ft: float,
    minimum_rows: int,
) -> int:
    mask = np.abs(np.asarray(error, dtype=np.float64)) >= float(threshold_ft)
    count = 0
    start = 0
    while start < len(mask):
        if not mask[start]:
            start += 1
            continue
        end = start + 1
        while end < len(mask) and mask[end]:
            end += 1
        if end - start >= int(minimum_rows):
            count += 1
        start = end
    return count


def build_episode_metrics(
    episodes: pd.DataFrame,
    readout: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    episode_config = get_nested(config, "model.persistent_episode")
    recovery_threshold = float(episode_config["recovery_threshold_ft"])
    horizons = [int(item) for item in episode_config["recovery_horizons_rows"]]
    grouped = {
        str(well): part.sort_values("row_idx", kind="mergesort")
        for well, part in readout.groupby("well", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        well = str(episode.well)
        part = grouped[well]
        start = int(episode.start_row_idx)
        end = int(episode.end_row_idx_exclusive)
        window = part.loc[
            part["row_idx"].ge(start) & part["row_idx"].lt(end)
        ]
        if window.empty:
            raise ValueError(f"{episode.episode_id}: fixed episode window is empty")
        row = {
            "episode_id": str(episode.episode_id),
            "well": well,
            "start_row_idx": start,
            "end_row_idx_exclusive": end,
            "rows": len(window),
            "parent_sse": float(np.square(window["parent_error"]).sum()),
            "candidate_sse": float(np.square(window["candidate_error"]).sum()),
        }
        for horizon in horizons:
            recovery = part.loc[
                part["row_idx"].ge(end)
                & part["row_idx"].lt(end + horizon)
            ]
            row[f"parent_recovered_within_{horizon}"] = bool(
                (recovery["parent_error"].abs() <= recovery_threshold).any()
            )
            row[f"candidate_recovered_within_{horizon}"] = bool(
                (recovery["candidate_error"].abs() <= recovery_threshold).any()
            )
        rows.append(row)
    return pd.DataFrame(rows)


# %% [markdown]
# ## 8. Frozen Stage 0 technical/mechanism gates and generated artifacts

# %%
def evaluate_stage0_gates(
    *,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    frozen_wells: Sequence[FrozenWell],
    readout: pd.DataFrame,
    well_metrics: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    segment_contract: pd.DataFrame,
    ledger: LeakageLedger,
    candidate_elapsed_seconds: float,
) -> dict[str, Any]:
    technical_config = get_nested(
        config,
        "validation.stage_0.technical_requires_all",
    )
    mechanism_config = get_nested(
        config,
        "validation.stage_0.mechanism_requires_all",
    )
    total_rows = int(sum(len(item.row_idx) for item in frozen_wells))
    finite_rows = int(
        sum(np.isfinite(item.prediction).sum() for item in frozen_wells)
    )
    finite_coverage = finite_rows / total_rows
    rho_values = np.concatenate([item.rho for item in frozen_wells])
    dmd_values = np.concatenate([item.dmd for item in frozen_wells])
    maximum_half_life_error = float(
        segment_contract["rho_product_abs_error_vs_half"].max()
    )
    maximum_normalization_error = max(
        item.maximum_posterior_normalization_error for item in frozen_wells
    )
    runtime_projection = float(candidate_elapsed_seconds * 773.0 / 32.0)
    manifest_roles = manifest["role"].value_counts().to_dict()
    technical = {
        "manifest_sha_match": True,
        "manifest_32_unique_16_plus_16": bool(
            len(manifest) == 32
            and manifest["well"].nunique() == 32
            and manifest_roles == {"persistent": 16, "control": 16}
        ),
        "segment_coverage": bool(
            len(segment_contract) == 32 * 16
            and (segment_contract.groupby("well").size() == 16).all()
        ),
        "positive_dmd_and_segment_span": bool(
            np.all(dmd_values > 0.0)
            and (segment_contract["dmd_span"] > 0.0).all()
        ),
        "rho_finite_coverage": bool(
            np.isfinite(rho_values).mean()
            >= float(technical_config["required_rho_finite_coverage"])
        ),
        "rho_bounds": bool(
            np.all(rho_values > float(technical_config["rho_lower_exclusive"]))
            and np.all(
                rho_values
                <= float(technical_config["rho_upper_inclusive"])
            )
        ),
        "segment_cumulative_half_life": bool(
            maximum_half_life_error
            <= float(technical_config["segment_cumulative_rho_atol"])
        ),
        "zero_state_geometry_identity": bool(
            np.mean([item.zero_state_identity_pass for item in frozen_wells])
            >= float(technical_config["required_zero_state_geometry_identity"])
        ),
        "posterior_normalization": bool(
            maximum_normalization_error
            <= float(technical_config["posterior_normalization_atol"])
        ),
        "prediction_finite_coverage": bool(
            finite_coverage
            >= float(technical_config["required_prediction_finite_coverage"])
        ),
        "forbidden_reads_before_freeze": bool(
            sum(ledger.forbidden_reads_before_freeze.values())
            <= int(
                technical_config[
                    "required_truth_role_fold_episode_reads_before_freeze"
                ]
            )
        ),
        "runtime_projection": bool(
            runtime_projection
            <= float(technical_config["full_runtime_projection_max_seconds"])
        ),
        "peak_rss": bool(
            peak_rss_gib() <= float(technical_config["peak_rss_max_gib"])
        ),
    }

    persistent_wells = well_metrics.loc[
        well_metrics["role"].eq("persistent")
    ]
    controls = well_metrics.loc[well_metrics["role"].eq("control")]
    persistent_improved_wells = int(
        (persistent_wells["candidate_minus_parent_rmse_ft"] < 0.0).sum()
    )
    persistent_fold_metrics: list[dict[str, Any]] = []
    persistent_rows = readout.loc[readout["role"].eq("persistent")]
    for fold in range(5):
        part = persistent_rows.loc[persistent_rows["fold"].eq(fold)]
        parent_fold_rmse = rmse(part["parent_error"])
        candidate_fold_rmse = rmse(part["candidate_error"])
        persistent_fold_metrics.append(
            {
                "fold": fold,
                "rows": len(part),
                "parent_rmse_ft": parent_fold_rmse,
                "candidate_rmse_ft": candidate_fold_rmse,
                "candidate_minus_parent_rmse_ft": (
                    candidate_fold_rmse - parent_fold_rmse
                ),
                "improved": bool(candidate_fold_rmse < parent_fold_rmse),
            }
        )
    persistent_improved_folds = sum(
        row["improved"] for row in persistent_fold_metrics
    )
    control_rows = readout.loc[readout["role"].eq("control")]
    control_parent_rmse = rmse(control_rows["parent_error"])
    control_candidate_rmse = rmse(control_rows["candidate_error"])
    control_regression = control_candidate_rmse - control_parent_rmse
    control_p95 = float(
        np.quantile(
            controls["candidate_minus_parent_rmse_ft"].to_numpy(np.float64),
            0.95,
        )
    )
    parent_sse = float(episode_metrics["parent_sse"].sum())
    candidate_sse = float(episode_metrics["candidate_sse"].sum())
    episode_sse_reduction = (
        1.0 - candidate_sse / parent_sse if parent_sse > 0.0 else math.nan
    )
    episode_config = get_nested(config, "model.persistent_episode")
    threshold = float(episode_config["error_threshold_ft"])
    minimum_rows = int(episode_config["minimum_consecutive_rows"])
    parent_episode_count = 0
    candidate_episode_count = 0
    for _, part in persistent_rows.groupby("well", sort=True):
        parent_episode_count += contiguous_episode_count(
            part["parent_error"].to_numpy(np.float64),
            threshold_ft=threshold,
            minimum_rows=minimum_rows,
        )
        candidate_episode_count += contiguous_episode_count(
            part["candidate_error"].to_numpy(np.float64),
            threshold_ft=threshold,
            minimum_rows=minimum_rows,
        )
    episode_count_delta = candidate_episode_count - parent_episode_count
    recovery_diagnostics: dict[str, Any] = {}
    recovery_checks: list[bool] = []
    for horizon in episode_config["recovery_horizons_rows"]:
        horizon = int(horizon)
        parent_rate = float(
            episode_metrics[f"parent_recovered_within_{horizon}"].mean()
        )
        candidate_rate = float(
            episode_metrics[f"candidate_recovered_within_{horizon}"].mean()
        )
        delta = candidate_rate - parent_rate
        minimum_delta = float(
            mechanism_config[f"recovery_rate_{horizon}_delta_min"]
        )
        recovery_diagnostics[str(horizon)] = {
            "parent_rate": parent_rate,
            "candidate_rate": candidate_rate,
            "delta": delta,
            "minimum_delta": minimum_delta,
        }
        recovery_checks.append(delta >= minimum_delta)
    mechanism = {
        "persistent_episode_sse_reduction": bool(
            math.isfinite(episode_sse_reduction)
            and episode_sse_reduction
            >= float(mechanism_config["persistent_episode_sse_reduction_min"])
        ),
        "persistent_improved_wells": bool(
            persistent_improved_wells
            >= int(mechanism_config["persistent_improved_wells_min"])
        ),
        "persistent_improved_folds": bool(
            persistent_improved_folds
            >= int(mechanism_config["persistent_improved_folds_min"])
        ),
        "matched_control_pooled_rmse": bool(
            control_regression
            <= float(
                mechanism_config[
                    "matched_control_pooled_rmse_regression_max_ft"
                ]
            )
        ),
        "matched_control_by_well_p95": bool(
            control_p95
            <= float(
                mechanism_config[
                    "matched_control_by_well_delta_rmse_p95_max_ft"
                ]
            )
        ),
        "persistent_episode_count": bool(
            episode_count_delta
            <= int(mechanism_config["persistent_episode_count_delta_max"])
        ),
        "recovery_rates_256_and_512": bool(all(recovery_checks)),
    }
    return {
        "technical": technical,
        "mechanism": mechanism,
        "all_pass": bool(all(technical.values()) and all(mechanism.values())),
        "diagnostics": {
            "rows": total_rows,
            "wells": 32,
            "persistent_wells": 16,
            "matched_control_wells": 16,
            "finite_coverage": finite_coverage,
            "rho_min": float(np.min(rho_values)),
            "rho_max": float(np.max(rho_values)),
            "maximum_segment_half_life_abs_error": maximum_half_life_error,
            "maximum_posterior_normalization_error": (
                maximum_normalization_error
            ),
            "candidate_runtime_seconds": candidate_elapsed_seconds,
            "full_773_runtime_projection_seconds": runtime_projection,
            "peak_rss_gib": peak_rss_gib(),
            "persistent_episode_sse_reduction_fraction": (
                episode_sse_reduction
            ),
            "persistent_improved_wells": persistent_improved_wells,
            "persistent_improved_folds": persistent_improved_folds,
            "persistent_fold_metrics": persistent_fold_metrics,
            "matched_control_parent_rmse_ft": control_parent_rmse,
            "matched_control_candidate_rmse_ft": control_candidate_rmse,
            "matched_control_rmse_regression_ft": control_regression,
            "matched_control_by_well_delta_rmse_p95_ft": control_p95,
            "parent_persistent_episode_count": parent_episode_count,
            "candidate_persistent_episode_count": candidate_episode_count,
            "persistent_episode_count_delta": episode_count_delta,
            "recovery": recovery_diagnostics,
            "stage_0_is_cv": False,
        },
    }


def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP490_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "exp490 Stage 0 must run on Kaggle CPU; local execution is disabled"
    )


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for the exp490 exact HMM")
    execution_contract = validate_execution_contract(
        config,
        require_run_authorization=True,
    )
    scientific_contract = validate_scientific_contract(config)
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    started = time.perf_counter()
    ledger = LeakageLedger(expected_wells=32)
    wells, manifest_identity = load_fixed32_identity(config, ledger)
    exp226, exp226_input = load_exp226_target_free(
        config,
        set(wells),
        ledger,
    )
    exp226_groups = {
        str(well): part
        for well, part in exp226.groupby("well_id", sort=False)
    }
    raw_dir = train_data_dir(config)
    frozen_wells: list[FrozenWell] = []
    hard_runtime = float(get_nested(config, "runtime.kaggle.runtime_limit_seconds"))
    hard_rss = float(
        get_nested(
            config,
            "validation.stage_0.technical_requires_all.peak_rss_max_gib",
        )
    )
    for well_index, well in enumerate(wells, start=1):
        if well not in exp226_groups:
            raise ValueError(f"{well}: exp226 candidate input is missing")
        frozen = freeze_target_free_well(
            well=well,
            raw_dir=raw_dir,
            exp226_well=exp226_groups[well],
            config=config,
            ledger=ledger,
        )
        frozen_wells.append(frozen)
        elapsed = float(time.perf_counter() - started)
        if elapsed > hard_runtime:
            raise RuntimeError(f"exp490 Stage 0 hard runtime exceeded: {elapsed}")
        if peak_rss_gib() > hard_rss:
            raise MemoryError(f"exp490 Stage 0 peak RSS exceeded: {peak_rss_gib()}")
        print(
            json.dumps(
                {
                    "event": "exp490_stage0_candidate_progress",
                    "well_index": well_index,
                    "well_count": 32,
                    "well": well,
                    "suffix_rows": len(frozen.row_idx),
                    "hmm_seconds": frozen.elapsed_seconds,
                    "half_life_max_abs_error": float(
                        np.max(np.abs(frozen.segment_cumulative_rho - 0.5))
                    ),
                    "posterior_normalization_max_abs_error": (
                        frozen.maximum_posterior_normalization_error
                    ),
                    "elapsed_seconds": elapsed,
                    "peak_rss_gib": peak_rss_gib(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    candidate_elapsed = float(time.perf_counter() - started)
    if not ledger.all_predictions_frozen:
        raise RuntimeError("all fixed32 candidate predictions were not frozen")

    predictions = prediction_frame(frozen_wells)
    segments = segment_contract_frame(frozen_wells)
    decoder_manifest = build_decoder_manifest(
        config,
        scientific_contract,
        frozen_wells,
        segments,
    )
    decoder_contract_sha = hashlib.sha256(
        stable_json_bytes(decoder_manifest)
    ).hexdigest()
    ledger.freeze_decoder_contract(decoder_contract_sha)
    if not ledger.all_frozen:
        raise RuntimeError("candidate and decoder contract freeze is incomplete")

    output = artifacts_dir()
    prediction_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_target_free_predictions.csv.gz",
        predictions,
    )
    segment_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_k16_segment_contract.csv",
        segments,
    )
    decoder_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0_decoder_manifest.json",
        decoder_manifest,
    )

    # Truth-bearing inputs begin here, after prediction and decoder SHA freeze.
    manifest = load_fixed32_readout_after_freeze(config, ledger)
    saved_parent, parent_input = load_saved_exp357_after_freeze(
        config,
        set(wells),
        ledger,
    )
    readout = attach_truth_late_readout(predictions, saved_parent, manifest)
    for item in frozen_wells:
        expected = manifest.loc[manifest["well"].eq(item.well)].iloc[0]
        if len(item.row_idx) != int(expected["suffix_rows"]):
            raise ValueError(f"{item.well}: fixed suffix row count changed")
        if item.prefix_rows != int(expected["prefix_rows"]):
            raise ValueError(f"{item.well}: fixed prefix row count changed")
    persistent_wells = set(
        manifest.loc[manifest["role"].eq("persistent"), "well"].astype(str)
    )
    episodes, episode_input = load_persistent_episodes_after_freeze(
        config,
        persistent_wells,
        ledger,
    )
    well_metrics = build_well_metrics(readout)
    episode_metrics = build_episode_metrics(episodes, readout, config)
    gates = evaluate_stage0_gates(
        config=config,
        manifest=manifest,
        frozen_wells=frozen_wells,
        readout=readout,
        well_metrics=well_metrics,
        episode_metrics=episode_metrics,
        segment_contract=segments,
        ledger=ledger,
        candidate_elapsed_seconds=candidate_elapsed,
    )

    readout_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_truth_late_rows.csv.gz",
        readout,
    )
    well_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_well_metrics.csv",
        well_metrics,
    )
    episode_artifact = write_csv(
        output / f"{EXPERIMENT_NAME}_stage0_episode_metrics.csv",
        episode_metrics,
    )
    input_manifest = {
        "fixed32_identity": manifest_identity,
        "exp226_target_free": exp226_input,
        "saved_exp357_post_freeze": parent_input,
        "persistent_episodes_post_freeze": episode_input,
        "raw_train_dir": str(raw_dir),
        "decoder_contract_sha256": decoder_contract_sha,
        "leakage_ledger": {
            "identity_rows": ledger.identity_rows,
            "target_free_rows": ledger.target_free_rows,
            "frozen_wells": len(ledger.frozen_wells),
            "forbidden_reads_before_freeze": ledger.forbidden_reads_before_freeze,
            "post_freeze_reads": ledger.post_freeze_reads,
        },
    }
    input_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0_input_manifest.json",
        input_manifest,
    )
    status = (
        "stage0_all_pass_pending_separate_stage1_approval"
        if gates["all_pass"]
        else "stage0_fail_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "stage": "stage_0_fixed32_mechanism_preflight_not_cv",
        "execution_contract": execution_contract,
        "scientific_contract_sha256": hashlib.sha256(
            stable_json_bytes(scientific_contract)
        ).hexdigest(),
        "decoder_contract_sha256": decoder_contract_sha,
        "prediction_content_sha256": prediction_artifact[
            "decompressed_sha256"
        ],
        "gates": gates,
        "runtime": {
            "candidate_elapsed_seconds": candidate_elapsed,
            "total_elapsed_seconds": float(time.perf_counter() - started),
            "peak_rss_gib": peak_rss_gib(),
            "versions": runtime_versions(),
            "cpu_only": True,
            "numba_threads": int(get_nested(config, "runtime.numba_num_threads")),
        },
        "leakage": input_manifest["leakage_ledger"],
        "artifacts": {
            "target_free_predictions": prediction_artifact,
            "k16_segment_contract": segment_artifact,
            "decoder_manifest": decoder_artifact,
            "truth_late_rows": readout_artifact,
            "well_metrics": well_artifact,
            "episode_metrics": episode_artifact,
            "input_manifest": input_artifact,
        },
        "stage_1": {
            "implemented": False,
            "execution_approved": False,
            "requires_stage_0_all_pass": True,
            "requires_separate_user_approval": True,
        },
        "inference": False,
        "submission": False,
    }
    summary_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0_summary.json",
        summary,
    )
    summary["artifacts"]["summary"] = summary_artifact
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "stage": "stage_0_fixed32_mechanism_preflight_not_cv",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "execution_contract": execution_contract,
        "technical_gates": gates["technical"],
        "mechanism_gates": gates["mechanism"],
        "stage_0_all_pass": gates["all_pass"],
        "result": gates["diagnostics"],
        "decoder_contract_sha256": decoder_contract_sha,
        "prediction_content_sha256": prediction_artifact[
            "decompressed_sha256"
        ],
        "artifacts": summary["artifacts"],
        "stage_1_implemented": False,
        "inference": False,
        "submission": False,
    }
    write_json(metrics_path(), metrics)
    print(json.dumps(to_jsonable(summary), sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 9. Four full-OOF shards and strict truth-late aggregate

# %%
def stable_full_well_shard(well: str, shard_count: int = FULL_SHARD_COUNT) -> int:
    key = f"exp490::full_well_shard::{well}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return int(value % int(shard_count))


def load_full_well_manifest(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_oof")
    path = resolve_unique_file(
        filename=str(spec["filename"]),
        candidates=[str(item) for item in spec["candidates"]],
        patterns=[str(item) for item in spec["patterns"]],
    )
    decompressed = sha256_decompressed_csv(path)
    if decompressed != str(spec["expected_decompressed_sha256"]):
        raise ValueError(f"exp226 decompressed SHA changed: {decompressed}")
    counts: dict[str, int] = {}
    for chunk in pd.read_csv(
        path,
        usecols=["well_id"],
        dtype={"well_id": str},
        chunksize=200_000,
    ):
        for well, rows in chunk.groupby("well_id", sort=False).size().items():
            counts[str(well)] = counts.get(str(well), 0) + int(rows)
    manifest = pd.DataFrame(
        {
            "well": sorted(counts),
            "rows": [counts[well] for well in sorted(counts)],
        }
    )
    manifest["shard_index"] = manifest["well"].map(stable_full_well_shard).astype(np.int8)
    expected_wells = int(get_nested(config, "validation.expected_full_wells"))
    expected_rows = int(get_nested(config, "validation.expected_full_rows"))
    if len(manifest) != expected_wells or int(manifest["rows"].sum()) != expected_rows:
        raise ValueError("full well manifest coverage changed")
    expected = pd.DataFrame(get_nested(config, "data.stage_1_shards.expected"))
    observed = (
        manifest.groupby("shard_index", sort=True)
        .agg(wells=("well", "size"), rows=("rows", "sum"))
        .reset_index()
    )
    compare = expected[["shard_index", "wells", "rows"]].astype(np.int64)
    if not observed.astype(np.int64).equals(compare):
        raise ValueError(f"full shard allocation changed: {observed.to_dict('records')}")
    return manifest, {
        "path": str(path),
        "raw_gzip_sha256": sha256_file(path),
        "decompressed_sha256": decompressed,
        "rows": expected_rows,
        "wells": expected_wells,
        "shard_policy": "sha256_exp490_full_well_shard_modulo_4",
        "manifest_logical_sha256": logical_frame_sha256(manifest),
    }


def full_shard_artifact_names(shard_index: int) -> dict[str, str]:
    prefix = f"{EXPERIMENT_NAME}_stage1_shard{int(shard_index)}"
    return {
        "predictions": f"{prefix}_target_free_predictions.csv.gz",
        "segments": f"{prefix}_k16_segment_contract.csv",
        "well_manifest": f"{prefix}_well_manifest.csv",
        "decoder": f"{prefix}_decoder_manifest.json",
        "input": f"{prefix}_input_manifest.json",
        "summary": f"{prefix}_summary.json",
    }


def run_full_shard(config: Mapping[str, Any], shard_index: int) -> dict[str, Any]:
    require_kaggle_runtime()
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for the exp490 exact HMM")
    if shard_index not in range(FULL_SHARD_COUNT):
        raise ValueError("full shard index must be in [0, 3]")
    execution_contract = validate_execution_contract(config, require_run_authorization=True)
    scientific_contract = validate_scientific_contract(config)
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    started = time.perf_counter()
    full_manifest, exp226_identity = load_full_well_manifest(config)
    selected_manifest = full_manifest.loc[
        full_manifest["shard_index"].astype(int).eq(int(shard_index))
    ].reset_index(drop=True)
    wells = selected_manifest["well"].astype(str).tolist()
    ledger = LeakageLedger(expected_wells=len(wells))
    ledger.record_identity(len(wells))
    exp226, exp226_input = load_exp226_target_free(config, set(wells), ledger)
    exp226_groups = {
        str(well): part for well, part in exp226.groupby("well_id", sort=False)
    }
    raw_dir = train_data_dir(config)
    frozen_wells: list[FrozenWell] = []
    hard_runtime = float(get_nested(config, "runtime.kaggle.runtime_limit_seconds"))
    hard_rss = float(
        get_nested(config, "validation.stage_0.technical_requires_all.peak_rss_max_gib")
    )
    for well_index, well in enumerate(wells, start=1):
        if well not in exp226_groups:
            raise ValueError(f"{well}: exp226 candidate input is missing")
        frozen = freeze_target_free_well(
            well=well,
            raw_dir=raw_dir,
            exp226_well=exp226_groups[well],
            config=config,
            ledger=ledger,
        )
        frozen_wells.append(frozen)
        elapsed = float(time.perf_counter() - started)
        if elapsed > hard_runtime:
            raise RuntimeError(f"exp490 full shard hard runtime exceeded: {elapsed}")
        if peak_rss_gib() > hard_rss:
            raise MemoryError(f"exp490 full shard peak RSS exceeded: {peak_rss_gib()}")
        print(
            json.dumps(
                {
                    "event": "exp490_stage1_shard_progress",
                    "shard_index": shard_index,
                    "well_index": well_index,
                    "well_count": len(wells),
                    "well": well,
                    "suffix_rows": len(frozen.row_idx),
                    "hmm_seconds": frozen.elapsed_seconds,
                    "elapsed_seconds": elapsed,
                    "peak_rss_gib": peak_rss_gib(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    candidate_elapsed = float(time.perf_counter() - started)
    if not ledger.all_predictions_frozen:
        raise RuntimeError("full shard candidate predictions were not all frozen")
    predictions = prediction_frame(frozen_wells)
    segments = segment_contract_frame(frozen_wells)
    decoder_manifest = build_decoder_manifest(
        config, scientific_contract, frozen_wells, segments
    )
    decoder_manifest.update(
        {
            "stage": "stage_1_target_free_full_shard",
            "shard_index": int(shard_index),
            "shard_count": FULL_SHARD_COUNT,
            "well_manifest_logical_sha256": logical_frame_sha256(selected_manifest),
        }
    )
    decoder_contract_sha = hashlib.sha256(stable_json_bytes(decoder_manifest)).hexdigest()
    ledger.freeze_decoder_contract(decoder_contract_sha)
    if not ledger.all_frozen:
        raise RuntimeError("full shard prediction and decoder freeze is incomplete")
    if ledger.forbidden_reads_before_freeze:
        raise RuntimeError("truth-bearing data was read during target-free shard generation")
    total_rows = int(selected_manifest["rows"].sum())
    if len(predictions) != total_rows or predictions["well"].nunique() != len(wells):
        raise ValueError("full shard prediction coverage changed")
    if predictions["id"].duplicated().any():
        raise ValueError("full shard prediction ids are not unique")
    maximum_half_life_error = float(
        segments["rho_product_abs_error_vs_half"].max()
    )
    maximum_normalization_error = float(
        max(item.maximum_posterior_normalization_error for item in frozen_wells)
    )
    technical = {
        "prediction_finite": bool(
            np.isfinite(
                predictions[
                    [
                        "geometry_mean_reverting_hmm",
                        "geometry_mean_reverting_delta_mean",
                        "geometry_mean_reverting_hmm_std",
                        "rho",
                        "dmd",
                    ]
                ].to_numpy(np.float64)
            ).all()
        ),
        "row_and_well_coverage": True,
        "stable_shard_assignment": bool(
            selected_manifest["well"].map(stable_full_well_shard).eq(shard_index).all()
        ),
        "segment_half_life": maximum_half_life_error <= 1.0e-10,
        "posterior_normalization": maximum_normalization_error <= 1.0e-6,
        "zero_state_geometry_identity": bool(
            all(item.zero_state_identity_pass for item in frozen_wells)
        ),
        "forbidden_reads_before_freeze": not ledger.forbidden_reads_before_freeze,
        "runtime": candidate_elapsed <= hard_runtime,
        "peak_rss": peak_rss_gib() <= hard_rss,
    }
    if not all(technical.values()):
        raise RuntimeError(f"exp490 full shard technical contract failed: {technical}")
    names = full_shard_artifact_names(shard_index)
    output = artifacts_dir()
    artifacts = {
        "predictions": write_deterministic_gzip_csv(output / names["predictions"], predictions),
        "segments": write_csv(output / names["segments"], segments),
        "well_manifest": write_csv(output / names["well_manifest"], selected_manifest),
        "decoder": write_json(output / names["decoder"], decoder_manifest),
    }
    input_manifest = {
        "exp226_full_identity": exp226_identity,
        "exp226_target_free_shard": exp226_input,
        "raw_train_dir": str(raw_dir),
        "shard_index": int(shard_index),
        "shard_count": FULL_SHARD_COUNT,
        "wells": len(wells),
        "rows": total_rows,
        "forbidden_reads_before_freeze": ledger.forbidden_reads_before_freeze,
        "post_freeze_truth_reads": ledger.post_freeze_reads,
    }
    artifacts["input"] = write_json(output / names["input"], input_manifest)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": "stage_1_full_shard_complete_awaiting_strict_merge",
        "stage": "stage_1_target_free_full_shard",
        "execution_basis": "explicit_user_override_after_stage0_fail",
        "shard_index": int(shard_index),
        "shard_count": FULL_SHARD_COUNT,
        "wells": len(wells),
        "rows": total_rows,
        "execution_contract": execution_contract,
        "scientific_contract_sha256": hashlib.sha256(
            stable_json_bytes(scientific_contract)
        ).hexdigest(),
        "decoder_contract_sha256": decoder_contract_sha,
        "technical": technical,
        "maximum_segment_half_life_abs_error": maximum_half_life_error,
        "maximum_posterior_normalization_error": maximum_normalization_error,
        "runtime": {
            "candidate_elapsed_seconds": candidate_elapsed,
            "total_elapsed_seconds": float(time.perf_counter() - started),
            "peak_rss_gib": peak_rss_gib(),
            "cpu_only": True,
            "numba_threads": int(get_nested(config, "runtime.numba_num_threads")),
            "versions": runtime_versions(),
        },
        "artifacts": artifacts,
        "inference": False,
        "submission": False,
    }
    summary_artifact = write_json(output / names["summary"], summary)
    write_json(metrics_path(), {**summary, "summary_artifact": summary_artifact})
    print(json.dumps(to_jsonable({**summary, "summary_artifact": summary_artifact}), sort_keys=True), flush=True)
    return summary


def resolve_exact_named_file(filename: str) -> Path:
    matches: list[Path] = []
    for root in (KAGGLE_INPUT_ROOT, find_project_root(), Path("/tmp")):
        if root.exists():
            matches.extend(root.glob(f"**/{filename}"))
    unique = sorted({path.resolve() for path in matches if path.is_file() and path.name == filename})
    if len(unique) != 1:
        raise FileNotFoundError(f"expected one {filename}, found {len(unique)}")
    return unique[0]


def load_pinned_full_shards(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], str]:
    configured = list(get_nested(config, "data.stage_1_shards.outputs"))
    if len(configured) != FULL_SHARD_COUNT:
        raise ValueError("full merge requires four configured shard outputs")
    candidate_parts: list[pd.DataFrame] = []
    segment_parts: list[pd.DataFrame] = []
    manifest_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    root_items: list[dict[str, Any]] = []
    for expected_index, pin in enumerate(configured):
        shard_index = int(pin["shard_index"])
        if shard_index != expected_index:
            raise ValueError("full shard pins must be ordered 0..3")
        required_pins = [
            "prediction_raw_gzip_sha256",
            "prediction_decompressed_sha256",
            "summary_sha256",
            "well_manifest_sha256",
        ]
        if any(not isinstance(pin.get(key), str) or len(str(pin.get(key))) != 64 for key in required_pins):
            raise RuntimeError(f"shard {shard_index} output SHA pins are incomplete")
        names = full_shard_artifact_names(shard_index)
        prediction_path = resolve_exact_named_file(names["predictions"])
        summary_path = resolve_exact_named_file(names["summary"])
        manifest_path = resolve_exact_named_file(names["well_manifest"])
        segment_path = resolve_exact_named_file(names["segments"])
        if sha256_file(prediction_path) != str(pin["prediction_raw_gzip_sha256"]):
            raise ValueError(f"shard {shard_index} raw prediction SHA mismatch")
        if sha256_decompressed_csv(prediction_path) != str(pin["prediction_decompressed_sha256"]):
            raise ValueError(f"shard {shard_index} decompressed prediction SHA mismatch")
        if sha256_file(summary_path) != str(pin["summary_sha256"]):
            raise ValueError(f"shard {shard_index} summary SHA mismatch")
        if sha256_file(manifest_path) != str(pin["well_manifest_sha256"]):
            raise ValueError(f"shard {shard_index} well manifest SHA mismatch")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("stage") != "stage_1_target_free_full_shard"
            or int(summary.get("shard_index", -1)) != shard_index
            or int(summary.get("shard_count", -1)) != FULL_SHARD_COUNT
            or not all(summary.get("technical", {}).values())
        ):
            raise ValueError(f"shard {shard_index} summary contract mismatch")
        if sha256_file(segment_path) != str(summary["artifacts"]["segments"]["sha256"]):
            raise ValueError(f"shard {shard_index} segment SHA mismatch")
        candidate = pd.read_csv(prediction_path, dtype={"id": str, "well": str})
        manifest = pd.read_csv(manifest_path, dtype={"well": str})
        segments = pd.read_csv(segment_path, dtype={"well": str})
        if len(candidate) != int(summary["rows"]) or candidate["well"].nunique() != int(summary["wells"]):
            raise ValueError(f"shard {shard_index} prediction coverage mismatch")
        if not manifest["shard_index"].astype(int).eq(shard_index).all():
            raise ValueError(f"shard {shard_index} manifest assignment mismatch")
        if not manifest["well"].map(stable_full_well_shard).eq(shard_index).all():
            raise ValueError(f"shard {shard_index} stable assignment mismatch")
        candidate["shard_index"] = np.int8(shard_index)
        candidate_parts.append(candidate)
        segment_parts.append(segments)
        manifest_parts.append(manifest)
        summaries.append(summary)
        root_items.append(
            {
                "shard_index": shard_index,
                "prediction_decompressed_sha256": str(pin["prediction_decompressed_sha256"]),
                "summary_sha256": str(pin["summary_sha256"]),
                "decoder_contract_sha256": str(summary["decoder_contract_sha256"]),
            }
        )
    candidate = pd.concat(candidate_parts, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    segments = pd.concat(segment_parts, ignore_index=True).sort_values(
        ["well", "k16_segment_id"], kind="mergesort"
    ).reset_index(drop=True)
    manifest = pd.concat(manifest_parts, ignore_index=True).sort_values(
        "well", kind="mergesort"
    ).reset_index(drop=True)
    expected_rows = int(get_nested(config, "validation.expected_full_rows"))
    expected_wells = int(get_nested(config, "validation.expected_full_wells"))
    if len(candidate) != expected_rows or candidate["well"].nunique() != expected_wells:
        raise ValueError("strict shard union coverage mismatch")
    if candidate["id"].duplicated().any() or manifest["well"].duplicated().any():
        raise ValueError("strict shard union contains duplicate identity")
    if len(manifest) != expected_wells or int(manifest["rows"].sum()) != expected_rows:
        raise ValueError("strict shard manifest union coverage mismatch")
    if not candidate.groupby("well")["shard_index"].first().astype(int).equals(
        manifest.set_index("well")["shard_index"].astype(int)
    ):
        raise ValueError("candidate and manifest shard identities differ")
    if len(segments) != expected_wells * int(get_nested(config, "model.k16_segment.count")):
        raise ValueError("strict shard segment union coverage mismatch")
    candidate_root = hashlib.sha256(stable_json_bytes(root_items)).hexdigest()
    return candidate, segments, manifest, summaries, candidate_root


def load_hidden_like_assignments(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like")
    path = resolve_unique_file(
        filename=str(spec["filename"]),
        candidates=[str(item) for item in spec["candidates"]],
        patterns=[f"**/{spec['filename']}"],
    )
    observed = sha256_file(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {"well_id", *[str(value) for value in spec["role_columns"].values()]}
    if not required.issubset(frame.columns) or frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment contract changed")
    return frame, {
        "path": str(path),
        "sha256": observed,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
    }


def load_all_persistent_episodes_after_freeze(
    config: Mapping[str, Any], ledger: LeakageLedger
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not ledger.all_frozen:
        raise RuntimeError("persistent episodes require global candidate freeze")
    spec = get_nested(config, "data.persistent_episodes")
    path = resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))
    observed = sha256_file(path)
    if observed != str(spec["expected_sha256"]):
        raise ValueError("persistent episode SHA changed")
    columns = ["episode_id", "well", "start_row_idx", "end_row_idx_exclusive"]
    frame = pd.read_csv(path, usecols=columns, dtype={"well": str})
    ledger.record_forbidden("persistent_episode_boundaries", len(frame))
    if frame.empty or frame["episode_id"].duplicated().any():
        raise ValueError("persistent episode contract changed")
    return frame.sort_values(["well", "start_row_idx"], kind="mergesort").reset_index(drop=True), {
        "path": str(path), "sha256": observed, "rows": len(frame),
        "wells": int(frame["well"].nunique()), "loaded_after_global_freeze": True,
    }


def attach_full_truth_late_readout(candidate: pd.DataFrame, parent: pd.DataFrame) -> pd.DataFrame:
    merged = candidate.merge(
        parent,
        on=["id", "well", "row_idx"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_saved"),
    )
    if len(merged) != len(candidate) or merged["true_tvt_readout_only"].isna().any():
        raise ValueError("full truth-late parent join is incomplete")
    if not np.array_equal(
        merged["tvt_geop"].to_numpy(np.float64),
        merged["tvt_geop_saved"].to_numpy(np.float64),
    ):
        raise ValueError("candidate and saved exp357 geometry identities differ")
    merged = merged.drop(columns=["tvt_geop_saved"])
    merged["fold"] = merged["fold"].astype(np.int8)
    merged["candidate_error"] = (
        merged["geometry_mean_reverting_hmm"] - merged["true_tvt_readout_only"]
    )
    merged["parent_error"] = (
        merged["exp357_parent_prediction"] - merged["true_tvt_readout_only"]
    )
    merged["exp226_error"] = merged["exp226_pred"] - merged["true_tvt_readout_only"]
    required = [
        "geometry_mean_reverting_hmm", "exp357_parent_prediction", "exp226_pred",
        "true_tvt_readout_only", "candidate_error", "parent_error", "exp226_error",
    ]
    if not np.isfinite(merged[required].to_numpy(np.float64)).all():
        raise ValueError("full truth-late readout contains non-finite values")
    return merged.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)


def score_prediction(truth: pd.Series, prediction: pd.Series) -> dict[str, Any]:
    error = prediction.to_numpy(np.float64) - truth.to_numpy(np.float64)
    if len(error) == 0 or not np.isfinite(error).all():
        raise ValueError("score requires non-empty finite arrays")
    return {
        "rows": len(error),
        "rmse": rmse(error),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "within5": float(np.mean(np.abs(error) <= 5.0)),
        "within10": float(np.mean(np.abs(error) <= 10.0)),
    }


FULL_CANDIDATES = {
    "geometry_mean_reverting_hmm": "geometry_mean_reverting_hmm",
    "exp357_parent": "exp357_parent_prediction",
    "exp226_final": "exp226_pred",
}


def build_full_candidate_metrics(readout: pd.DataFrame) -> pd.DataFrame:
    truth = readout["true_tvt_readout_only"]
    return pd.DataFrame(
        [
            {"candidate": name, **score_prediction(truth, readout[column])}
            for name, column in FULL_CANDIDATES.items()
        ]
    )


def build_full_fold_metrics(readout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, part in readout.groupby("fold", sort=True):
        for name, column in FULL_CANDIDATES.items():
            rows.append(
                {
                    "fold": int(fold),
                    "candidate": name,
                    **score_prediction(part["true_tvt_readout_only"], part[column]),
                }
            )
    return pd.DataFrame(rows)


def build_full_scope_metrics(
    readout: pd.DataFrame,
    hidden: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    scopes: list[tuple[str, pd.DataFrame]] = [
        ("overall", readout),
        ("md_1000_plus", readout.loc[readout["md_since"] >= 1000.0]),
    ]
    role_by_well = hidden.set_index("well_id")
    for scope, role_column in get_nested(config, "data.hidden_like.role_columns").items():
        valid_wells = set(
            role_by_well.index[
                role_by_well[str(role_column)].astype(str).eq("valid")
            ].astype(str)
        )
        scopes.append((str(scope), readout.loc[readout["well"].isin(valid_wells)]))
    rows: list[dict[str, Any]] = []
    for scope, part in scopes:
        if part.empty:
            raise ValueError(f"scope {scope} selected zero rows")
        candidate = rmse(part["candidate_error"])
        parent = rmse(part["parent_error"])
        rows.append(
            {
                "scope": scope,
                "rows": len(part),
                "wells": int(part["well"].nunique()),
                "candidate_rmse_ft": candidate,
                "exp357_parent_rmse_ft": parent,
                "candidate_minus_parent_rmse_ft": candidate - parent,
            }
        )
    return pd.DataFrame(rows)


def build_full_by_well_metrics(readout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, part in readout.groupby("well", sort=True):
        candidate = rmse(part["candidate_error"])
        parent = rmse(part["parent_error"])
        rows.append(
            {
                "well": str(well),
                "fold": int(part["fold"].iloc[0]),
                "rows": len(part),
                "candidate_rmse_ft": candidate,
                "exp357_parent_rmse_ft": parent,
                "candidate_minus_parent_rmse_ft": candidate - parent,
            }
        )
    return pd.DataFrame(rows)


def evaluate_full_stage1_gate(
    config: Mapping[str, Any],
    readout: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    episode_metrics: pd.DataFrame,
) -> dict[str, Any]:
    gate = get_nested(config, "validation.stage_1.promotion_requires_all")
    metrics = candidate_metrics.set_index("candidate")
    candidate_rmse = float(metrics.loc["geometry_mean_reverting_hmm", "rmse"])
    parent_rmse = float(metrics.loc["exp357_parent", "rmse"])
    exp226_rmse = float(metrics.loc["exp226_final", "rmse"])
    folds = fold_metrics.pivot(index="fold", columns="candidate", values="rmse")
    improved_folds = int(
        (folds["geometry_mean_reverting_hmm"] < folds["exp357_parent"]).sum()
    )
    scope_table = scope_metrics.set_index("scope")
    required_scopes = [str(value) for value in gate["required_nonworse_scopes"]]
    scope_deltas = {
        scope: float(scope_table.loc[scope, "candidate_minus_parent_rmse_ft"])
        for scope in required_scopes
    }
    well_delta = by_well["candidate_minus_parent_rmse_ft"].to_numpy(np.float64)
    p95 = float(np.quantile(well_delta, 0.95))
    worst_index = int(np.argmax(well_delta))
    worst_well = str(by_well.iloc[worst_index]["well"])
    worst_delta = float(well_delta[worst_index])
    parent_sse = float(episode_metrics["parent_sse"].sum())
    candidate_sse = float(episode_metrics["candidate_sse"].sum())
    episode_sse_reduction = 1.0 - candidate_sse / parent_sse
    episode_config = get_nested(config, "model.persistent_episode")
    persistent_wells = set(episode_metrics["well"].astype(str))
    persistent_rows = readout.loc[readout["well"].isin(persistent_wells)]
    parent_episode_count = 0
    candidate_episode_count = 0
    for _, part in persistent_rows.groupby("well", sort=True):
        parent_episode_count += contiguous_episode_count(
            part["parent_error"].to_numpy(np.float64),
            threshold_ft=float(episode_config["error_threshold_ft"]),
            minimum_rows=int(episode_config["minimum_consecutive_rows"]),
        )
        candidate_episode_count += contiguous_episode_count(
            part["candidate_error"].to_numpy(np.float64),
            threshold_ft=float(episode_config["error_threshold_ft"]),
            minimum_rows=int(episode_config["minimum_consecutive_rows"]),
        )
    episode_count_delta = candidate_episode_count - parent_episode_count
    recovery: dict[str, Any] = {}
    recovery_pass = True
    for horizon in episode_config["recovery_horizons_rows"]:
        horizon = int(horizon)
        parent_rate = float(episode_metrics[f"parent_recovered_within_{horizon}"].mean())
        candidate_rate = float(episode_metrics[f"candidate_recovered_within_{horizon}"].mean())
        delta = candidate_rate - parent_rate
        recovery[str(horizon)] = {
            "parent_rate": parent_rate,
            "candidate_rate": candidate_rate,
            "delta": delta,
        }
        recovery_pass = recovery_pass and delta >= float(gate[f"recovery_rate_{horizon}_delta_min"])
    finite_coverage = float(
        np.isfinite(
            readout[
                [
                    "geometry_mean_reverting_hmm",
                    "geometry_mean_reverting_delta_mean",
                    "geometry_mean_reverting_hmm_std",
                ]
            ].to_numpy(np.float64)
        ).mean()
    )
    saved = get_nested(config, "validation.saved_references")
    parity_atol = float(gate["saved_control_rmse_parity_atol_ft"])
    checks = {
        "exp357_parent_rmse_parity": abs(parent_rmse - float(saved["exp357_huber_parent_rmse"])) <= parity_atol,
        "exp226_final_rmse_parity": abs(exp226_rmse - float(saved["exp226_final_rmse"])) <= parity_atol,
        "overall_gain_vs_exp357": parent_rmse - candidate_rmse >= float(gate["minimum_rmse_gain_vs_exp357_ft"]),
        "improved_folds": improved_folds >= int(gate["minimum_improved_folds"]),
        "required_scopes_nonworse": all(
            delta <= float(gate["maximum_scope_rmse_regression_ft"])
            for delta in scope_deltas.values()
        ),
        "by_well_p95_nonworse": p95 <= float(gate["maximum_by_well_delta_rmse_p95_ft"]),
        "worst_well_regression": worst_delta <= float(gate["maximum_worst_well_rmse_regression_ft"]),
        "persistent_episode_sse": episode_sse_reduction >= float(gate["persistent_episode_sse_reduction_min"]),
        "persistent_episode_count": episode_count_delta <= int(gate["persistent_episode_count_delta_max"]),
        "persistent_recovery": bool(recovery_pass),
        "gain_vs_exp226_final": float(saved["exp226_final_rmse"]) - candidate_rmse >= float(gate["minimum_rmse_gain_vs_exp226_final_ft"]),
        "maximum_candidate_rmse": candidate_rmse <= float(gate["maximum_candidate_rmse"]),
        "row_identity_coverage": len(readout) == int(get_nested(config, "validation.expected_full_rows")),
        "finite_coverage": finite_coverage >= float(gate["required_finite_coverage"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate_rmse": candidate_rmse,
        "exp357_parent_rmse": parent_rmse,
        "exp226_final_rmse": exp226_rmse,
        "gain_vs_exp357_ft": parent_rmse - candidate_rmse,
        "gain_vs_exp226_final_ft": float(saved["exp226_final_rmse"]) - candidate_rmse,
        "improved_folds": improved_folds,
        "scope_delta_rmse_vs_exp357": scope_deltas,
        "by_well_p95_delta_rmse_vs_exp357": p95,
        "worst_well": worst_well,
        "worst_well_delta_rmse_vs_exp357": worst_delta,
        "persistent_episode_sse_reduction_fraction": episode_sse_reduction,
        "persistent_episode_count_delta": episode_count_delta,
        "recovery": recovery,
        "finite_coverage": finite_coverage,
        "execution_basis": "explicit_user_override_after_stage0_fail",
        "decision": "stage_1_passed_awaiting_inference_approval" if all(checks.values()) else "stage_1_failed_close_without_rescue",
    }


def run_full_aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    execution_contract = validate_execution_contract(config, require_run_authorization=True)
    scientific_contract = validate_scientific_contract(config)
    started = time.perf_counter()
    candidate, segments, well_manifest, shard_summaries, candidate_root = load_pinned_full_shards(config)
    ledger = LeakageLedger(expected_wells=0)
    ledger.freeze_decoder_contract(candidate_root)
    parent, parent_input = load_saved_exp357_after_freeze(
        config, set(candidate["well"].astype(str)), ledger
    )
    readout = attach_full_truth_late_readout(candidate, parent)
    hidden, hidden_input = load_hidden_like_assignments(config)
    episodes, episode_input = load_all_persistent_episodes_after_freeze(config, ledger)
    candidate_metrics = build_full_candidate_metrics(readout)
    fold_metrics = build_full_fold_metrics(readout)
    scope_metrics = build_full_scope_metrics(readout, hidden, config)
    by_well = build_full_by_well_metrics(readout)
    episode_metrics = build_episode_metrics(episodes, readout, config)
    gate = evaluate_full_stage1_gate(
        config, readout, candidate_metrics, fold_metrics, scope_metrics, by_well, episode_metrics
    )
    output = artifacts_dir()
    prefix = f"{EXPERIMENT_NAME}_stage1_full_oof"
    artifacts = {
        "predictions": write_deterministic_gzip_csv(output / f"{prefix}_predictions.csv.gz", readout),
        "segments": write_csv(output / f"{prefix}_k16_segment_contract.csv", segments),
        "well_manifest": write_csv(output / f"{prefix}_well_manifest.csv", well_manifest),
        "candidate_metrics": write_csv(output / f"{prefix}_candidate_metrics.csv", candidate_metrics),
        "fold_metrics": write_csv(output / f"{prefix}_fold_metrics.csv", fold_metrics),
        "scope_metrics": write_csv(output / f"{prefix}_scope_metrics.csv", scope_metrics),
        "by_well_metrics": write_csv(output / f"{prefix}_by_well_metrics.csv", by_well),
        "episode_metrics": write_csv(output / f"{prefix}_episode_metrics.csv", episode_metrics),
        "gate": write_json(output / f"{prefix}_gate.json", gate),
    }
    input_manifest = {
        "candidate_root_sha256": candidate_root,
        "shards": [
            {
                "shard_index": int(summary["shard_index"]),
                "rows": int(summary["rows"]),
                "wells": int(summary["wells"]),
                "scientific_contract_sha256": summary["scientific_contract_sha256"],
                "decoder_contract_sha256": summary["decoder_contract_sha256"],
            }
            for summary in shard_summaries
        ],
        "saved_exp357_post_freeze": parent_input,
        "hidden_like_post_freeze": hidden_input,
        "persistent_episodes_post_freeze": episode_input,
        "post_freeze_reads": ledger.post_freeze_reads,
    }
    artifacts["input_manifest"] = write_json(output / f"{prefix}_input_manifest.json", input_manifest)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": "stage_1_full_oof_passed_awaiting_inference_approval" if gate["passed"] else "stage_1_full_oof_failed_closed",
        "stage": "stage_1_full_oof_truth_late_strict_merge",
        "execution_basis": "explicit_user_override_after_stage0_fail",
        "rows": len(readout),
        "wells": int(readout["well"].nunique()),
        "shards": FULL_SHARD_COUNT,
        "execution_contract": execution_contract,
        "scientific_contract_sha256": hashlib.sha256(stable_json_bytes(scientific_contract)).hexdigest(),
        "candidate_root_sha256": candidate_root,
        "gate": gate,
        "runtime": {
            "merge_elapsed_seconds": float(time.perf_counter() - started),
            "peak_rss_gib": peak_rss_gib(),
            "cpu_only": True,
            "hmm_well_runs": 0,
            "versions": runtime_versions(),
        },
        "artifacts": artifacts,
        "inference": False,
        "submission": False,
    }
    summary_artifact = write_json(output / f"{prefix}_summary.json", summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "route": "pf_beam",
        "stage": summary["stage"],
        "cv": gate["candidate_rmse"],
        "public_lb": None,
        "private_lb": None,
        "gate": gate,
        "candidate_root_sha256": candidate_root,
        "artifacts": {**artifacts, "summary": summary_artifact},
        "inference": False,
        "submission": False,
    }
    write_json(metrics_path(), metrics)
    print(json.dumps(to_jsonable({**summary, "summary_artifact": summary_artifact}), sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 10. Configuration preview and guarded execution

# %%
CONFIG = load_config()
SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
EXECUTION_CONTRACT = validate_execution_contract(CONFIG, require_run_authorization=False)
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "status": get_nested(CONFIG, "experiment.status"),
            "route": get_nested(CONFIG, "experiment.route"),
            "active_stage": get_nested(CONFIG, "execution.active_stage"),
            "run_kind": RUN_KIND_OVERRIDE,
            "run_full_oof": bool(get_nested(CONFIG, "execution.run_full_oof")),
            "execution_contract": EXECUTION_CONTRACT,
            "stage_0_all_pass": False,
            "execution_basis": "explicit_user_override_after_stage0_fail",
            "inference": False,
            "submission": False,
        },
        indent=2,
        sort_keys=True,
    )
)

SUMMARY = None
if __name__ == "__main__" and bool(get_nested(CONFIG, "execution.run_full_oof", False)):
    if RUN_KIND_OVERRIDE.startswith("shard"):
        SUMMARY = run_full_shard(CONFIG, int(RUN_KIND_OVERRIDE.removeprefix("shard")))
    elif RUN_KIND_OVERRIDE == "aggregate":
        SUMMARY = run_full_aggregate(CONFIG)
    else:
        raise ValueError(f"unknown exp490 full run kind: {RUN_KIND_OVERRIDE}")

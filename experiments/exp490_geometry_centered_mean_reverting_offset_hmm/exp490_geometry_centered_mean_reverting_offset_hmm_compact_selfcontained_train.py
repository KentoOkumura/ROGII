# %% [markdown]
# # exp490 geometry-centered mean-reverting offset HMM
#
# This Stage 0 notebook implements one frozen mechanism candidate. It keeps the
# exp357 Huber residual-offset exact HMM fixed and changes only the transition
# centers:
#
# `rho_t = 2 ** (-dMD_t / destination_K16_segment_MD_span)`
#
# `q_center_t = 0.998 * rho_t * q_(t-1)`
#
# `delta_center_t = rho_t * delta_(t-1) + q_t * dMD_t`
#
# Stage 0 decodes the identity-only fixed32 scope. Role, fold, truth, saved
# exp357 predictions, and persistent-episode boundaries are unavailable until
# all 32 candidate predictions and the decoder contract SHA have been frozen.
# Stage 0 is a mechanism preflight, not CV. Stage 1, inference, and submission
# remain unimplemented and disabled.

# %% [markdown]
# ## Contents
# 1. Imports and immutable execution/scientific contracts
# 2. Notebook-safe paths, SHA helpers, and leakage ledger
# 3. Identity-only fixed32 and target-free exp226/raw inputs
# 4. K16 segment half-life and fixed exp357 Huber input preparation
# 5. Geometry-centered exact forward-backward decoder
# 6. Target-free per-well prediction and decoder-contract freeze
# 7. Truth-late saved-parent and persistent-episode readout
# 8. Frozen Stage 0 technical/mechanism gates and generated artifacts
# 9. Configuration preview and guarded execution

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
    expected = {
        "scientific_variants": 1,
        "candidate_hmm_well_runs": 32,
        "saved_parent_hmm_well_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    observed = dict(get_nested(config, "execution_contract.stage_0_if_separately_approved"))
    if observed != expected:
        raise ValueError(f"Stage 0 cost contract changed: {observed}")
    if bool(get_nested(config, "execution.run_stage_1", False)):
        raise ValueError("exp490 Stage 1 is not implemented or authorized")
    if require_run_authorization and bool(
        get_nested(config, "execution.run_inference", False)
    ):
        raise ValueError("exp490 inference is disabled")
    if require_run_authorization and bool(
        get_nested(config, "execution.create_submission", False)
    ):
        raise ValueError("exp490 submission is disabled")
    if require_run_authorization:
        if not bool(get_nested(config, "execution.run_stage_0", False)):
            raise RuntimeError("exp490 Stage 0 run flag is disabled")
        if not bool(get_nested(config, "implementation.kaggle_run_approved", False)):
            raise RuntimeError("exp490 Kaggle Stage 0 execution is not approved")
    return {
        **expected,
        "reporting_folds": 5,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
        "control_rerun": False,
        "stage_0_is_cv": False,
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
        "stage1_disabled": not bool(get_nested(config, "execution.run_stage_1")),
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
        "stage": "stage_0_fixed32_mechanism_preflight_not_cv",
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
        raise ValueError("exp226 fixed32 target-free selection is empty")
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
    if frame["well_id"].nunique() != 32:
        raise ValueError("exp226 target-free cache does not cover fixed32")
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
        "stage": "stage_0_fixed32_mechanism_preflight_not_cv",
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
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values(
        ["well", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    if frame["well"].nunique() != 32 or frame.duplicated(["well", "row_idx"]).any():
        raise ValueError("saved exp357 fixed32 coverage or identity changed")
    if not np.isfinite(frame[numeric].to_numpy(np.float64)).all():
        raise ValueError("saved exp357 fixed32 readout is non-finite")
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
# ## 9. Configuration preview and guarded execution
#
# The config prints the frozen 1-variant / 32-well / zero-control-rerun cost
# contract. A Stage 0 run starts only from notebook execution (`__main__`) after
# both Kaggle run guards are explicitly enabled; importing this source for
# contract tests remains side-effect free.

# %%
CONFIG = load_config()
SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
EXECUTION_CONTRACT = validate_execution_contract(
    CONFIG,
    require_run_authorization=False,
)
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "status": get_nested(CONFIG, "experiment.status"),
            "route": get_nested(CONFIG, "experiment.route"),
            "active_stage": get_nested(CONFIG, "execution.active_stage"),
            "implementation_approved": bool(
                get_nested(CONFIG, "implementation.implementation_approved")
            ),
            "kaggle_run_approved": bool(
                get_nested(CONFIG, "implementation.kaggle_run_approved")
            ),
            "run_stage_0": bool(get_nested(CONFIG, "execution.run_stage_0")),
            "execution_contract": EXECUTION_CONTRACT,
            "stage_0_is_cv": False,
            "stage_1_implemented": False,
            "inference": False,
            "submission": False,
        },
        indent=2,
        sort_keys=True,
    )
)

SUMMARY = None
if __name__ == "__main__" and bool(
    get_nested(CONFIG, "execution.run_stage_0", False)
):
    SUMMARY = run_stage0(CONFIG)

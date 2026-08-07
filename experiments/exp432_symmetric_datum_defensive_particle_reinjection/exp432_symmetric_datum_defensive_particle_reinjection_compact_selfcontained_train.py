# %% [markdown]
# # exp432 symmetric datum defensive particle reinjection — Stage 0
#
# This compact self-contained notebook implements the approved train-side
# mechanism preflight only. It freezes the first exp412 persistent rate-gap
# event from an unchanged exp209 HMM pass, then compares the exp404-compatible
# likelihood-PF with a one-transition 80/10/10 base/minus/plus position
# proposal. The original PF target is retained with an unclipped, log-domain
# `p0/q` correction. Truth, cause roles, and folds are attached only after all
# target-free schedules, predictions, particle-support ledgers, and SHAs have
# been frozen.
#
# Kaggle Stage 0 version 1 completed fail-closed. Rerun, full OOF, inference,
# and submission are locked in the post-run `config.yaml`.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable execution contract
# 2. Notebook-safe paths, SHA, and leakage ledger
# 3. Fixed32 scope and target-free input checks
# 4. Exact exp209 first-pass HMM and frozen event schedule
# 5. Exp404 likelihood-PF input preparation
# 6. Symmetric defensive proposal and log-domain importance correction
# 7. Target-free Stage 0 prediction freeze
# 8. Truth-late mechanism readout and fail-closed gates
# 9. Generated artifacts and guarded execution

# %% [markdown]
# ## 1. Imports and immutable execution contract

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from numba import get_num_threads, njit, prange, set_num_threads

EXPERIMENT_NAME = "exp432_symmetric_datum_defensive_particle_reinjection"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
LOG_IMPORTANCE_MAX = math.log(1.25)
COMPONENT_NAMES = ("base", "minus_datum", "plus_datum")
FORBIDDEN_PRE_FREEZE_COLUMNS = {
    "TVT",
    "tvt_true",
    "truth",
    "error",
    "abs_error",
    "cause",
    "role",
    "fold",
}


def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_run_authorization: bool,
) -> dict[str, int]:
    design = get_nested(config, "design")
    execution = get_nested(config, "execution")
    stage0 = get_nested(config, "run_contract.stage0")
    full = get_nested(config, "run_contract.full")
    expected = {
        "active_scientific_variants": 1,
        "stage_0_hmm_trigger_well_runs": 32,
        "stage_0_baseline_pf_well_runs": 32,
        "stage_0_treatment_pf_well_runs": 32,
        "stage_0_total_pf_well_runs": 64,
        "stage_0_seed_well_trajectories": 8192,
        "stage_0_particle_starts": 4096000,
        "full_hmm_trigger_well_runs": 773,
        "full_treatment_pf_well_runs": 773,
        "full_seed_well_trajectories": 98944,
        "full_particle_starts": 49472000,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    observed = {key: int(execution[key]) for key in expected}
    if observed != expected:
        raise ValueError(f"exp432 execution counts changed: {observed}")
    if not bool(design["implementation_allowed_now"]):
        raise ValueError("exp432 implementation approval is missing")
    if bool(design["full_execution_allowed"]):
        raise ValueError("full execution must remain locked before Stage 0 passes")
    if bool(design["inference_allowed"]) or bool(design["submission_allowed"]):
        raise ValueError("inference and submission must remain locked")
    if int(stage0["total_pf_well_runs"]) != 64:
        raise ValueError("Stage 0 must rerun baseline 32 plus treatment 32 PF wells")
    if int(full["parent_independent_full_pf_reruns"]) != 0:
        raise ValueError("full must reuse the saved exp404 control")
    if require_run_authorization:
        if not bool(design["canonical_notebook_adoption_allowed"]):
            raise RuntimeError("exp432 canonical train notebook adoption is not approved")
        if not bool(execution["kaggle_execution_authorized"]):
            raise RuntimeError("exp432 Kaggle Stage 0 run is not approved")
        if not bool(design["kaggle_stage0_push_allowed"]):
            raise RuntimeError("exp432 Kaggle Stage 0 push is not approved")
    return observed


def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    trigger = get_nested(config, "trigger")
    proposal = get_nested(config, "proposal")
    pf = get_nested(config, "pf")
    hmm = get_nested(config, "hmm")
    expected_trigger = {
        "denominator_floor": 0.005,
        "absolute_z_threshold": 2.0,
        "rolling_window_rows": 16,
        "qualifying_rows_min": 8,
        "same_sign_fraction_min": 0.75,
        "tie_policy": "inactive",
        "maximum_events_per_well": 1,
        "beta_sign_used_by_treatment": False,
        "freeze_before_truth": True,
    }
    observed_trigger = {key: trigger[key] for key in expected_trigger}
    if observed_trigger != expected_trigger:
        raise ValueError(f"exp432 trigger contract changed: {observed_trigger}")
    masses = np.asarray(
        [
            float(proposal["components"]["base"]["mass"]),
            float(proposal["components"]["minus_datum"]["mass"]),
            float(proposal["components"]["plus_datum"]["mass"]),
        ],
        dtype=np.float64,
    )
    if not np.array_equal(masses, np.asarray([0.80, 0.10, 0.10])):
        raise ValueError("exp432 proposal mass contract changed")
    if str(proposal["components"]["minus_datum"]["position_shift_ft"]) != "-datum":
        raise ValueError("minus datum shift contract changed")
    if str(proposal["components"]["plus_datum"]["position_shift_ft"]) != "+datum":
        raise ValueError("plus datum shift contract changed")
    if proposal["importance_clip"] is not None:
        raise ValueError("importance clipping is forbidden")
    if bool(proposal["materialize_ratio_for_weighting"]):
        raise ValueError("importance weights must be updated in log space")
    if (int(pf["particles"]), int(pf["seeds"]), float(pf["scale"])) != (
        500,
        128,
        1.0,
    ):
        raise ValueError("exp432 particles/seeds/scale contract changed")
    dynamics = pf["dynamics"]
    if (
        float(dynamics["momentum"]),
        float(dynamics["rate_noise"]),
        float(dynamics["position_noise"]),
        float(dynamics["rough_position_ft"]),
        float(dynamics["rough_rate"]),
    ) != (0.998, 0.002, 0.005, 0.10, 0.001):
        raise ValueError("exp404 PF dynamics contract changed")
    if float(pf["readout"]["temperature"]) != 5.0:
        raise ValueError("exp404 seed aggregation temperature changed")
    if (
        float(hmm["sig_r"]),
        float(hmm["sig_p"]),
        float(hmm["momentum"]),
    ) != (0.002, 0.02, 0.998):
        raise ValueError("exp209 first-pass HMM contract changed")
    contract = {
        "experiment": EXPERIMENT_NAME,
        "trigger": observed_trigger,
        "proposal_mass": masses.tolist(),
        "datum_formula": str(proposal["datum_ft"]),
        "importance": {
            "formula": str(proposal["importance_ratio"]),
            "density": str(proposal["mixture_density_evaluation"]),
            "weight_update": str(proposal["numerical_weight_update"]),
            "clip": None,
            "log_upper_bound": LOG_IMPORTANCE_MAX,
        },
        "pf": {
            "particles": int(pf["particles"]),
            "seeds": int(pf["seeds"]),
            "temperature": float(pf["readout"]["temperature"]),
            "dynamics": dict(dynamics),
        },
        "hmm": dict(hmm),
        "forbidden": list(get_nested(config, "forbidden")),
    }
    contract["sha256"] = hashlib.sha256(stable_json_bytes(contract)).hexdigest()
    return contract


# %% [markdown]
# ## 2. Notebook-safe paths, SHA, and leakage ledger

# %%
def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file() and (candidate / "AGENTS.md").is_file():
            return candidate
    return start


def experiment_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        return KAGGLE_WORKING_ROOT
    root = find_project_root()
    candidate = root / "experiments" / EXPERIMENT_NAME
    return candidate if candidate.is_dir() else PACKAGE_DIR


def config_path() -> Path:
    local = experiment_dir() / "config.yaml"
    return local if local.is_file() else PACKAGE_DIR / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    selected = path or config_path()
    value = yaml.safe_load(selected.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def artifacts_dir() -> Path:
    path = experiment_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_path() -> Path:
    return experiment_dir() / "metrics.json"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


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


def array_bundle_sha256(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(stable_json_bytes(array.shape))
        digest.update(array.tobytes())
    return digest.hexdigest()


def logical_frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    data = normalized.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )
    return {"path": str(path), "sha256": sha256_file(path)}


def write_deterministic_gzip_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = frame.to_csv(index=False, lineterminator="\n").encode()
    with path.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            compressed.write(raw)
    return {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "logical_sha256": logical_frame_sha256(frame),
        "rows": len(frame),
    }


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 if platform.system() != "Darwin" else 1024.0**2
    return value / divisor / 1024.0


def runtime_versions() -> dict[str, str]:
    import numba

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "numba_threads": str(get_num_threads()),
    }


def resolve_bootstrap_asset(filename: str, local_path: str) -> Path:
    candidates = [
        experiment_dir() / "assets" / filename,
        PACKAGE_DIR / "assets" / filename,
        find_project_root() / local_path,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"could not resolve bootstrap asset {filename}")


def resolve_unique_file(
    filename: str,
    candidates: Sequence[str],
    patterns: Sequence[str],
) -> Path:
    root = find_project_root()
    matches: list[Path] = []
    for item in candidates:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            matches.append(candidate)
        elif (candidate / filename).is_file():
            matches.append(candidate / filename)
    if KAGGLE_INPUT_ROOT.is_dir():
        for pattern in patterns:
            matches.extend(KAGGLE_INPUT_ROOT.glob(pattern))
    unique = sorted({path.resolve() for path in matches if path.is_file()})
    if not unique:
        raise FileNotFoundError(f"could not resolve {filename}")
    hashes = {sha256_file(path) for path in unique}
    if len(hashes) != 1:
        raise RuntimeError(f"multiple non-identical files found for {filename}")
    return unique[0]


@dataclass
class LeakageLedger:
    expected_wells: int = 32
    frozen_wells: set[str] = field(default_factory=set)
    target_free_rows: int = 0
    truth_rows_before_all_freeze: int = 0
    role_rows_before_all_freeze: int = 0
    truth_rows_after_all_freeze: int = 0
    role_rows_after_all_freeze: int = 0

    @property
    def all_frozen(self) -> bool:
        return len(self.frozen_wells) == self.expected_wells

    def record_target_free(self, rows: int) -> None:
        self.target_free_rows += int(rows)

    def freeze(self, well: str) -> None:
        self.frozen_wells.add(str(well))

    def record_truth_late(self, rows: int) -> None:
        if not self.all_frozen:
            self.truth_rows_before_all_freeze += int(rows)
            raise RuntimeError("truth was read before all fixed32 artifacts froze")
        self.truth_rows_after_all_freeze += int(rows)

    def record_roles_late(self, rows: int) -> None:
        if not self.all_frozen:
            self.role_rows_before_all_freeze += int(rows)
            raise RuntimeError("cause roles/folds were read before all artifacts froze")
        self.role_rows_after_all_freeze += int(rows)


# %% [markdown]
# ## 3. Fixed32 scope and target-free input checks

# %%
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


def fixed32_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    manifest_spec = get_nested(config, "data.stage_0_manifest")
    metadata_spec = get_nested(config, "data.stage_0_manifest_metadata")
    manifest_path = resolve_bootstrap_asset(
        str(manifest_spec["filename"]), str(manifest_spec["local"])
    )
    metadata_path = resolve_bootstrap_asset(
        str(metadata_spec["filename"]), str(metadata_spec["local"])
    )
    if sha256_file(manifest_path) != str(manifest_spec["expected_sha256"]):
        raise ValueError("fixed32 manifest SHA changed")
    if sha256_file(metadata_path) != str(metadata_spec["expected_sha256"]):
        raise ValueError("fixed32 metadata SHA changed")
    return manifest_path, metadata_path


def load_fixed32_scope(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> tuple[list[str], dict[str, Any]]:
    manifest_path, metadata_path = fixed32_paths(config)
    scope = pd.read_csv(manifest_path, usecols=["well"], dtype={"well": str})
    if len(scope) != 32 or scope["well"].nunique() != 32:
        raise ValueError("fixed32 scope must contain 32 unique wells")
    ledger.record_target_free(len(scope))
    return sorted(scope["well"].astype(str)), {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "scope_sha256": logical_frame_sha256(scope.sort_values("well")),
    }


def load_fixed32_roles_after_freeze(
    config: Mapping[str, Any],
    ledger: LeakageLedger,
) -> pd.DataFrame:
    manifest_path, _ = fixed32_paths(config)
    ledger.record_roles_late(32)
    frame = pd.read_csv(manifest_path, dtype={"well": str}, keep_default_na=False)
    if frame["role"].value_counts().to_dict() != {
        "control": 16,
        "backward_cause": 8,
        "forward_cause": 8,
    }:
        raise ValueError("fixed32 cause-role counts changed")
    if set(frame["fold"].astype(int)) != set(range(5)):
        raise ValueError("fixed32 reporting folds changed")
    return frame


def load_target_free_well(
    well: str,
    raw_dir: Path,
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
    typewell_path = raw_dir / f"{well}__typewell.csv"
    horizontal = pd.read_csv(
        horizontal_path,
        usecols=lambda column: str(column) != "TVT",
    )
    forbidden = FORBIDDEN_PRE_FREEZE_COLUMNS.intersection(horizontal.columns)
    if forbidden:
        raise ValueError(f"{well}: forbidden decoder columns {sorted(forbidden)}")
    typewell = pd.read_csv(typewell_path).sort_values("TVT").reset_index(drop=True)
    ledger.record_target_free(len(horizontal) + len(typewell))
    return horizontal, typewell


def cache_row_indices(frame: pd.DataFrame, well_column: str = "well") -> np.ndarray:
    output = np.empty(len(frame), dtype=np.int64)
    for offset, (well, identifier) in enumerate(
        zip(frame[well_column].astype(str), frame["id"].astype(str), strict=True)
    ):
        prefix = f"{well}_"
        if not identifier.startswith(prefix) or not identifier[len(prefix) :].isdigit():
            raise ValueError(f"invalid cache id {identifier}")
        output[offset] = int(identifier[len(prefix) :])
    return output


def load_saved_control_subset(
    config: Mapping[str, Any],
    key: str,
    wells: set[str],
    ledger: LeakageLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, f"data.{key}")
    path = resolve_unique_file(
        str(spec["filename"]),
        [str(value) for value in spec["candidates"]],
        [str(value) for value in spec.get("patterns", [])],
    )
    if spec.get("expected_raw_sha256") and sha256_file(path) != str(
        spec["expected_raw_sha256"]
    ):
        raise ValueError(f"{key} raw SHA changed")
    if sha256_decompressed_csv(path) != str(spec["expected_decompressed_sha256"]):
        raise ValueError(f"{key} decompressed SHA changed")
    header = pd.read_csv(path, compression="gzip", nrows=0)
    well_column = "well" if "well" in header.columns else "well_id"
    usecols = ["id", well_column, str(spec["prediction_column"])]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=usecols,
        dtype={"id": str, well_column: str},
        chunksize=200_000,
    ):
        selected = chunk.loc[chunk[well_column].isin(wells)]
        if not selected.empty:
            pieces.append(selected)
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.rename(
        columns={
            well_column: "well",
            str(spec["prediction_column"]): "saved_prediction",
        }
    )
    frame["row_idx"] = cache_row_indices(frame)
    frame["saved_prediction"] = pd.to_numeric(frame["saved_prediction"], errors="raise")
    frame = frame.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well", "row_idx"]).any():
        raise ValueError(f"{key} contains duplicate row keys")
    ledger.record_target_free(len(frame))
    return frame, {
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": sha256_decompressed_csv(path),
        "rows": len(frame),
    }


# %% [markdown]
# ## 4. Exact exp209 first-pass HMM and frozen event schedule

# %%
def robust_initial_rate(
    known_prefix: pd.DataFrame,
    window_rows: int = 30,
) -> float:
    tail = known_prefix.tail(window_rows)
    dtvt = np.diff(tail["TVT_input"].to_numpy(np.float64))
    dz = np.diff(tail["Z"].to_numpy(np.float64))
    dmd = np.diff(tail["MD"].to_numpy(np.float64))
    valid = np.isfinite(dtvt) & np.isfinite(dz) & np.isfinite(dmd) & (dmd > 0)
    if int(valid.sum()) < 3:
        return 0.0
    return float(np.median((dtvt[valid] + dz[valid]) / dmd[valid]))


def prepare_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    if "TVT" in horizontal.columns:
        raise ValueError("suffix truth reached HMM preparation")
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].ffill().bfill().to_numpy(np.float64)
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    evaluation = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4 or evaluation.empty:
        raise ValueError("expected visible prefix and non-empty suffix")
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    step = float(hmm["step"])
    grid_min = max(typewell_tvt.min() - 40.0, last_tvt - float(hmm["band_pad"]))
    grid_max = min(typewell_tvt.max() + 40.0, last_tvt + float(hmm["band_pad"]))
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    gr_grid = np.interp(grid, typewell_tvt, typewell_gr)
    known_tvt = known["TVT_input"].to_numpy(np.float64)
    known_gr = known["GR"].fillna(0.0).to_numpy(np.float64)
    residual = known_gr - np.interp(known_tvt, typewell_tvt, typewell_gr)
    gr_sigma = float(np.clip(np.nanstd(residual), 10.0, 60.0))
    filled_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(typewell_gr)))
    )
    eval_indices = evaluation.index.to_numpy(np.int64)
    md = evaluation["MD"].to_numpy(np.float64)
    z = evaluation["Z"].to_numpy(np.float64)
    dm = np.maximum(np.diff(np.r_[float(last["MD"]), md]), 1.0)
    dz = np.diff(np.r_[float(last["Z"]), z])
    zscore = (
        filled_gr.to_numpy(np.float64)[eval_indices, None] - gr_grid[None, :]
    ) / gr_sigma
    emission = (-0.5 * np.minimum(zscore**2, 600.0)).astype(np.float32)
    initial_rate = robust_initial_rate(known)
    rate_span = max(float(hmm["rate_span"]), abs(initial_rate) + 0.04)
    return {
        "emission_ll": emission,
        "dm": dm,
        "dz": dz,
        "grid": grid,
        "rates": np.linspace(
            -rate_span, rate_span, int(hmm["n_rates"]), dtype=np.float64
        ),
        "start_p": float((last_tvt - grid_min) / step),
        "r0": initial_rate,
        "eval_indices": eval_indices,
    }


@njit(cache=True, nogil=True)
def rate_kernel_probabilities(
    rates: np.ndarray,
    dm: float,
    sig_r: float,
    momentum: float,
) -> np.ndarray:
    count = len(rates)
    rate_step = rates[1] - rates[0]
    variance_cells = (sig_r * np.sqrt(dm) / rate_step) ** 2
    kernel = np.empty((count, 3), np.float64)
    for index in range(count):
        mean_move = -(1.0 - momentum) * rates[index] * dm / rate_step
        plus = max(0.5 * (variance_cells + mean_move), 1.0e-12)
        minus = max(0.5 * (variance_cells - mean_move), 1.0e-12)
        total = plus + minus
        if total > 0.9:
            plus *= 0.9 / total
            minus *= 0.9 / total
        kernel[index, 0] = minus
        kernel[index, 1] = 1.0 - plus - minus
        kernel[index, 2] = plus
    return kernel


@njit(cache=True, nogil=True, parallel=True)
def _exact_hmm_messages(
    emission,
    dm,
    dz,
    step,
    rates,
    sig_r,
    sig_p,
    start_p,
    start_sig,
    r0,
    r0_sig,
    lam,
    momentum,
):
    rows, positions = emission.shape
    rate_count = len(rates)
    neg = np.float32(-1.0e18)
    alpha = np.full((rows, positions, rate_count), neg, np.float32)
    previous = np.full((positions, rate_count), neg, np.float32)
    for position_index in range(positions):
        dpos = (position_index - start_p) * step
        lp0 = -0.5 * (dpos / start_sig) ** 2
        if lp0 < -60:
            continue
        for rate_index in range(rate_count):
            dr = (rates[rate_index] - r0) / r0_sig
            previous[position_index, rate_index] = np.float32(lp0 - 0.5 * dr * dr)
    temporary = np.empty_like(previous)
    predictive = np.empty_like(previous)
    current = np.empty_like(previous)
    filtered_rate_mean = np.empty(rows, np.float64)
    filtered_rate_second = np.empty(rows, np.float64)
    filtered_position_mean = np.empty(rows, np.float64)
    filtered_position_second = np.empty(rows, np.float64)
    maximum_normalization_error = 0.0
    for row in range(rows):
        rate_kernel = rate_kernel_probabilities(rates, dm[row], sig_r, momentum)
        log_rate_kernel = np.log(rate_kernel)
        for p_index in prange(positions):
            for r2 in range(rate_count):
                best = neg
                start = max(r2 - 1, 0)
                stop = min(r2 + 1, rate_count - 1)
                for r1 in range(start, stop + 1):
                    value = previous[p_index, r1] + log_rate_kernel[
                        r1, r2 - r1 + 1
                    ]
                    best = max(best, value)
                total = 0.0
                if best > neg / 2:
                    for r1 in range(start, stop + 1):
                        total += np.exp(
                            previous[p_index, r1]
                            + log_rate_kernel[r1, r2 - r1 + 1]
                            - best
                        )
                    temporary[p_index, r2] = np.float32(best + np.log(total))
                else:
                    temporary[p_index, r2] = neg
        position_sigma = max(sig_p, 0.35 * step)
        for r2 in prange(rate_count):
            mean = rates[r2] * dm[row] - dz[row]
            center = int(np.floor(mean / step + 0.5))
            log_kernel = np.empty(5, np.float64)
            for offset in range(5):
                delta = (center - 2 + offset) * step - mean
                log_kernel[offset] = -0.5 * (delta / position_sigma) ** 2
            kernel_max = np.max(log_kernel)
            log_kernel -= kernel_max + np.log(np.sum(np.exp(log_kernel - kernel_max)))
            for p2 in range(positions):
                best = neg
                for offset in range(5):
                    p1 = p2 - (center - 2 + offset)
                    if 0 <= p1 < positions:
                        best = max(best, temporary[p1, r2] + log_kernel[offset])
                total = 0.0
                if best > neg / 2:
                    for offset in range(5):
                        p1 = p2 - (center - 2 + offset)
                        if 0 <= p1 < positions:
                            total += np.exp(
                                temporary[p1, r2] + log_kernel[offset] - best
                            )
                    pre_emission = best + np.log(total)
                    predictive[p2, r2] = np.float32(pre_emission)
                    current[p2, r2] = np.float32(
                        pre_emission + lam * emission[row, p2]
                    )
                else:
                    predictive[p2, r2] = neg
                    current[p2, r2] = neg
        filtered_best = np.max(current)
        filtered_total = 0.0
        rate1 = 0.0
        rate2 = 0.0
        position1 = 0.0
        position2 = 0.0
        for p_index in range(positions):
            for r_index in range(rate_count):
                probability = np.exp(current[p_index, r_index] - filtered_best)
                filtered_total += probability
                rate1 += probability * rates[r_index]
                rate2 += probability * rates[r_index] * rates[r_index]
                position_ft = p_index * step
                position1 += probability * position_ft
                position2 += probability * position_ft * position_ft
        filtered_rate_mean[row] = rate1 / filtered_total
        filtered_rate_second[row] = rate2 / filtered_total
        filtered_position_mean[row] = position1 / filtered_total
        filtered_position_second[row] = position2 / filtered_total
        check = 0.0
        for p_index in range(positions):
            for r_index in range(rate_count):
                alpha[row, p_index, r_index] = current[p_index, r_index]
                previous[p_index, r_index] = current[p_index, r_index]
                check += np.exp(current[p_index, r_index] - filtered_best) / filtered_total
        maximum_normalization_error = max(
            maximum_normalization_error, abs(check - 1.0)
        )
    final_best = np.max(alpha[rows - 1])
    final_total = 0.0
    for p_index in range(positions):
        for r_index in range(rate_count):
            final_total += np.exp(
                alpha[rows - 1, p_index, r_index] - final_best
            )
    log_likelihood = float(final_best) + np.log(final_total)
    posterior_position = np.zeros((rows, positions), np.float64)
    beta_next = np.zeros((positions, rate_count), np.float32)
    beta_current = np.empty_like(beta_next)
    beta_temporary = np.empty_like(beta_next)
    for row in range(rows - 1, -1, -1):
        values = alpha[row] + beta_next
        best = np.max(values)
        total = 0.0
        for p_index in range(positions):
            probability = 0.0
            for r_index in range(rate_count):
                probability += np.exp(values[p_index, r_index] - best)
            posterior_position[row, p_index] = probability
            total += probability
        for p_index in range(positions):
            posterior_position[row, p_index] /= total
            for r_index in range(rate_count):
                alpha[row, p_index, r_index] = np.float32(
                    np.exp(values[p_index, r_index] - best) / total
                )
        if row == 0:
            break
        rate_kernel = rate_kernel_probabilities(
            rates, dm[row], sig_r, momentum
        )
        log_rate_kernel = np.log(rate_kernel)
        position_sigma = max(sig_p, 0.35 * step)
        for r2 in prange(rate_count):
            mean = rates[r2] * dm[row] - dz[row]
            center = int(np.floor(mean / step + 0.5))
            log_kernel = np.empty(5, np.float64)
            for offset in range(5):
                delta = (center - 2 + offset) * step - mean
                log_kernel[offset] = -0.5 * (delta / position_sigma) ** 2
            kernel_max = np.max(log_kernel)
            log_kernel -= kernel_max + np.log(np.sum(np.exp(log_kernel - kernel_max)))
            for p1 in range(positions):
                best = neg
                for offset in range(5):
                    p2 = p1 + (center - 2 + offset)
                    if 0 <= p2 < positions:
                        best = max(
                            best,
                            log_kernel[offset]
                            + lam * emission[row, p2]
                            + beta_next[p2, r2],
                        )
                total = 0.0
                if best > neg / 2:
                    for offset in range(5):
                        p2 = p1 + (center - 2 + offset)
                        if 0 <= p2 < positions:
                            total += np.exp(
                                log_kernel[offset]
                                + lam * emission[row, p2]
                                + beta_next[p2, r2]
                                - best
                            )
                    beta_temporary[p1, r2] = np.float32(best + np.log(total))
                else:
                    beta_temporary[p1, r2] = neg
        for p_index in prange(positions):
            for r1 in range(rate_count):
                best = neg
                start = max(r1 - 1, 0)
                stop = min(r1 + 1, rate_count - 1)
                for r2 in range(start, stop + 1):
                    best = max(
                        best,
                        log_rate_kernel[r1, r2 - r1 + 1]
                        + beta_temporary[p_index, r2],
                    )
                total = 0.0
                if best > neg / 2:
                    for r2 in range(start, stop + 1):
                        total += np.exp(
                            log_rate_kernel[r1, r2 - r1 + 1]
                            + beta_temporary[p_index, r2]
                            - best
                        )
                    beta_current[p_index, r1] = np.float32(best + np.log(total))
                else:
                    beta_current[p_index, r1] = neg
        beta_next[:, :] = beta_current
    smoothed_rate_mean = np.zeros(rows, np.float64)
    posterior_error = 0.0
    for row in range(rows):
        row_total = 0.0
        for p_index in range(positions):
            for r_index in range(rate_count):
                probability = float(alpha[row, p_index, r_index])
                row_total += probability
                smoothed_rate_mean[row] += probability * rates[r_index]
        posterior_error = max(posterior_error, abs(row_total - 1.0))
        smoothed_rate_mean[row] /= row_total
    filtered_rate_std = np.sqrt(
        np.maximum(filtered_rate_second - filtered_rate_mean**2, 0.0)
    )
    filtered_position_std = np.sqrt(
        np.maximum(filtered_position_second - filtered_position_mean**2, 0.0)
    )
    return (
        posterior_position,
        log_likelihood,
        filtered_rate_mean,
        filtered_rate_std,
        filtered_position_std,
        smoothed_rate_mean,
        max(maximum_normalization_error, posterior_error),
    )


def run_first_pass_hmm(
    prepared: Mapping[str, Any],
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    outputs = _exact_hmm_messages(
        np.asarray(prepared["emission_ll"], dtype=np.float32),
        np.asarray(prepared["dm"], dtype=np.float64),
        np.asarray(prepared["dz"], dtype=np.float64),
        float(hmm["step"]),
        np.asarray(prepared["rates"], dtype=np.float64),
        float(hmm["sig_r"]),
        float(hmm["sig_p"]),
        float(prepared["start_p"]),
        float(hmm["start_sig"]),
        float(prepared["r0"]),
        float(hmm["r0_sig"]),
        float(hmm["lam"]),
        float(hmm["momentum"]),
    )
    (
        posterior_position,
        log_likelihood,
        filtered_rate_mean,
        filtered_rate_std,
        filtered_position_std,
        smoothed_rate_mean,
        normalization_error,
    ) = outputs
    prediction = posterior_position @ np.asarray(prepared["grid"], dtype=np.float64)
    return {
        "prediction": prediction,
        "log_likelihood": float(log_likelihood),
        "filtered_rate_mean": filtered_rate_mean,
        "filtered_rate_std": filtered_rate_std,
        "filtered_position_std": filtered_position_std,
        "smoothed_rate_mean": smoothed_rate_mean,
        "normalization_error": float(normalization_error),
        "elapsed_seconds": time.perf_counter() - started,
        "message_sha256": array_bundle_sha256(
            filtered_rate_mean=filtered_rate_mean,
            filtered_rate_std=filtered_rate_std,
            filtered_position_std=filtered_position_std,
            smoothed_rate_mean=smoothed_rate_mean,
        ),
    }


def beta_filter_activation_schedule(
    smoothed_rate_mean: np.ndarray,
    filtered_rate_mean: np.ndarray,
    filtered_rate_std: np.ndarray,
    trigger: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    smoothed = np.asarray(smoothed_rate_mean, dtype=np.float64)
    filtered = np.asarray(filtered_rate_mean, dtype=np.float64)
    std = np.asarray(filtered_rate_std, dtype=np.float64)
    z_beta = (smoothed - filtered) / np.maximum(
        std, float(trigger["denominator_floor"])
    )
    rows = len(z_beta)
    direction = np.zeros(rows, dtype=np.int8)
    qualifying_count = np.zeros(rows, dtype=np.int16)
    majority_fraction = np.zeros(rows, dtype=np.float64)
    window = int(trigger["rolling_window_rows"])
    threshold = float(trigger["absolute_z_threshold"])
    minimum = int(trigger["qualifying_rows_min"])
    fraction_min = float(trigger["same_sign_fraction_min"])
    for row in range(rows):
        values = z_beta[max(0, row - window + 1) : row + 1]
        qualifying = values[np.abs(values) >= threshold]
        qualifying_count[row] = len(qualifying)
        if len(qualifying) < minimum:
            continue
        positive = int(np.sum(qualifying > 0))
        negative = int(np.sum(qualifying < 0))
        majority_fraction[row] = max(positive, negative) / len(qualifying)
        if positive == negative or majority_fraction[row] < fraction_min:
            continue
        direction[row] = 1 if positive > negative else -1
    return {
        "z_beta": z_beta,
        "active_direction": direction,
        "qualifying_count": qualifying_count,
        "majority_fraction": majority_fraction,
    }


def first_persistent_activation_event(active_direction: np.ndarray) -> int:
    active = np.asarray(active_direction, dtype=np.int8) != 0
    entered = active & ~np.r_[False, active[:-1]]
    indices = np.flatnonzero(entered)
    return int(indices[0]) if len(indices) else -1


# %% [markdown]
# ## 5. Exp404 likelihood-PF input preparation

# %%
def uniform_typewell_grid(
    tvt: np.ndarray,
    gr: np.ndarray,
    *,
    step: float,
) -> tuple[np.ndarray, float, float]:
    minimum = float(np.nanmin(tvt))
    maximum = float(np.nanmax(tvt))
    grid = np.arange(minimum, maximum + step, step, dtype=np.float64)
    return np.interp(grid, tvt, gr), minimum, step


def exp072_base_gr_scale(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    clip: tuple[float, float],
) -> float:
    known = horizontal["TVT_input"].notna()
    residual = (
        horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64)
        - np.interp(
            horizontal.loc[known, "TVT_input"].to_numpy(np.float64),
            typewell_tvt,
            typewell_gr,
        )
    )
    raw_scale = float(np.nanstd(residual))
    if not math.isfinite(raw_scale):
        raise ValueError("known-prefix GR residual scale is not finite")
    return float(np.clip(raw_scale, clip[0], clip[1]))


def prepare_pf_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    dynamics: Mapping[str, Any],
) -> dict[str, Any]:
    if "TVT" in horizontal.columns:
        raise ValueError("suffix truth reached PF preparation")
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_mean = float(typewell["GR"].mean())
    if not math.isfinite(typewell_mean):
        raise ValueError("Type Well GR mean is not finite")
    typewell_gr = typewell["GR"].fillna(typewell_mean).to_numpy(np.float64)
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    evaluation_mask = ~known_mask
    known = horizontal.loc[known_mask]
    evaluation = horizontal.loc[evaluation_mask]
    if known.empty or evaluation.empty:
        raise ValueError("likelihood-PF requires prefix and suffix")
    last = known.iloc[-1]
    eval_indices = np.flatnonzero(evaluation_mask).astype(np.int64)
    filled_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(typewell_mean)
        .to_numpy(np.float64)
    )
    grid_gr, grid_minimum, grid_step = uniform_typewell_grid(
        typewell_tvt,
        typewell_gr,
        step=float(dynamics["typewell_grid_step_ft"]),
    )
    clip = tuple(float(value) for value in dynamics["gr_sigma_clip"])
    return {
        "eval_indices": eval_indices,
        "eval_md": evaluation["MD"].to_numpy(np.float64),
        "eval_z": evaluation["Z"].to_numpy(np.float64),
        "eval_gr": filled_gr[eval_indices],
        "raw_gr_missing": evaluation["GR"].isna().to_numpy(bool),
        "last_known_position": float(last["TVT_input"] + last["Z"]),
        "last_known_tvt": float(last["TVT_input"]),
        "last_known_md": float(last["MD"]),
        "initial_rate": robust_initial_rate(known),
        "grid_gr": grid_gr,
        "grid_minimum": grid_minimum,
        "grid_step": grid_step,
        "gr_scale": exp072_base_gr_scale(
            horizontal, typewell_tvt, typewell_gr, clip
        ),
    }


def exp404_seed_base(well: str) -> int:
    payload = f"likpf::train::{well}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % 2_147_483_647 + 1


def component_draws(
    well: str,
    *,
    seeds: int,
    particles: int,
    event_row: int,
) -> np.ndarray:
    """Stable component IDs keyed by experiment/well/seed/event/particle.

    SplitMix64 is used only as a deterministic hash expander. The immutable
    SHA-256 prefix includes experiment, well, seed, and event; each particle
    index is then mixed independently, so particle order, shard, and resume
    order do not alter the draw.
    """
    result = np.empty((seeds, particles), dtype=np.int8)
    mask = (1 << 64) - 1
    for seed_index in range(seeds):
        prefix = (
            f"{EXPERIMENT_NAME}::{well}::{seed_index}::{event_row}".encode()
        )
        base = int.from_bytes(hashlib.sha256(prefix).digest()[:8], "little")
        for particle_index in range(particles):
            value = (base ^ particle_index) & mask
            value = (value + 0x9E3779B97F4A7C15) & mask
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
            value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
            value ^= value >> 31
            uniform = ((value >> 11) & ((1 << 53) - 1)) / float(1 << 53)
            result[seed_index, particle_index] = (
                0 if uniform < 0.80 else (1 if uniform < 0.90 else 2)
            )
    return result


# %% [markdown]
# ## 6. Symmetric defensive proposal and log-domain importance correction

# %%
@njit(cache=True)
def _interp1(grid: np.ndarray, value: float, minimum: float, step: float) -> float:
    index = int((value - minimum) / step)
    if index < 0:
        return grid[0]
    final = len(grid) - 1
    if index >= final:
        return grid[final]
    fraction = (value - minimum) / step - index
    return grid[index] * (1.0 - fraction) + grid[index + 1] * fraction


@njit(cache=True)
def _normal_logpdf(value: float, mean: float, sigma: float) -> float:
    zscore = (value - mean) / sigma
    return -0.5 * zscore * zscore - np.log(sigma) - 0.9189385332046727


@njit(cache=True)
def _logsumexp3(a: float, b: float, c: float) -> float:
    maximum = max(a, b, c)
    return maximum + np.log(
        np.exp(a - maximum) + np.exp(b - maximum) + np.exp(c - maximum)
    )


@njit(cache=True)
def symmetric_position_log_importance(
    innovation: float,
    datum: float,
    sigma: float,
) -> tuple[float, float]:
    """Return `(log(p0/q), log(q))` without clipping or ratio materialization."""
    log_p0 = _normal_logpdf(innovation, 0.0, sigma)
    log_minus = _normal_logpdf(innovation, -datum, sigma)
    log_plus = _normal_logpdf(innovation, datum, sigma)
    log_q = _logsumexp3(
        np.log(0.80) + log_p0,
        np.log(0.10) + log_minus,
        np.log(0.10) + log_plus,
    )
    return log_p0 - log_q, log_q


def importance_quadrature_contract(
    *,
    datum: float = 0.35,
    sigma: float = 0.005,
    points: int = 200_001,
) -> dict[str, float | bool]:
    grid = np.linspace(-datum - 8 * sigma, datum + 8 * sigma, points)
    log_p0 = -0.5 * (grid / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(
        2 * math.pi
    )
    log_minus = (
        -0.5 * ((grid + datum) / sigma) ** 2
        - math.log(sigma)
        - 0.5 * math.log(2 * math.pi)
    )
    log_plus = (
        -0.5 * ((grid - datum) / sigma) ** 2
        - math.log(sigma)
        - 0.5 * math.log(2 * math.pi)
    )
    maximum = np.maximum.reduce(
        [np.log(0.8) + log_p0, np.log(0.1) + log_minus, np.log(0.1) + log_plus]
    )
    log_q = maximum + np.log(
        np.exp(np.log(0.8) + log_p0 - maximum)
        + np.exp(np.log(0.1) + log_minus - maximum)
        + np.exp(np.log(0.1) + log_plus - maximum)
    )
    log_importance = log_p0 - log_q
    corrected_density = np.exp(log_q + log_importance)
    mass = float(np.trapezoid(corrected_density, grid))
    mean = float(np.trapezoid(corrected_density * grid, grid))
    variance = float(np.trapezoid(corrected_density * grid**2, grid))
    return {
        "mass": mass,
        "mean": mean,
        "variance": variance,
        "target_variance": sigma**2,
        "max_log_importance": float(np.max(log_importance)),
        "finite_log_importance": bool(np.isfinite(log_importance).all()),
        "pass": bool(
            abs(mass - 1.0) <= 1.0e-8
            and abs(mean) <= 1.0e-10
            and abs(variance - sigma**2) <= 1.0e-9
            and float(np.max(log_importance)) <= LOG_IMPORTANCE_MAX + 1.0e-12
        ),
    }


@njit(cache=True, nogil=True)
def _pf_symmetric_allseeds(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    grid_gr: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    gr_scale: float,
    last_position: float,
    initial_rate: float,
    particles: int,
    seeds: int,
    seed_base: int,
    momentum: float,
    rate_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
    initial_spread: float,
    initial_rate_spread: float,
    event_index: int,
    datum: float,
    event_components: np.ndarray,
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
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows = len(md_v)
    predictions = np.empty((seeds, rows), np.float64)
    log_likelihoods = np.empty(seeds, np.float64)
    support_min = np.empty((seeds, rows), np.float32)
    support_max = np.empty((seeds, rows), np.float32)
    ancestry = np.zeros((rows, 3), np.float64)
    component_counts = np.zeros((seeds, 3), np.int64)
    log_importance_min = np.zeros(seeds, np.float64)
    log_importance_max = np.zeros(seeds, np.float64)
    log_q_min = np.zeros(seeds, np.float64)
    resampling_counts = np.zeros(seeds, np.int64)
    minimum_ess = np.full(seeds, float(particles))
    clip_counts = np.zeros(seeds, np.int64)
    final_position = np.empty((seeds, particles), np.float64)
    final_rate = np.empty((seeds, particles), np.float64)
    final_weights = np.empty((seeds, particles), np.float64)
    grid_maximum = grid_minimum + len(grid_gr) * grid_step
    for seed_index in range(seeds):
        np.random.seed(seed_base + seed_index)
        position = np.empty(particles, np.float64)
        rate = np.empty(particles, np.float64)
        weights = np.ones(particles, np.float64) / particles
        labels = np.zeros(particles, np.int8)
        event_log_importance = np.zeros(particles, np.float64)
        event_log_q = np.zeros(particles, np.float64)
        for particle in range(particles):
            position[particle] = last_position + initial_spread * np.random.randn()
            rate[particle] = initial_rate + initial_rate_spread * np.random.randn()
        log_likelihood = 0.0
        previous_md = md_v[0] - 1.0
        for row in range(rows):
            delta_md = max(md_v[row] - previous_md, 1.0)
            for particle in range(particles):
                rate[particle] = momentum * rate[particle] + rate_noise * np.random.randn()
                innovation = position_noise * np.random.randn()
                if row == event_index:
                    label = event_components[seed_index, particle]
                    labels[particle] = label
                    if label == 1:
                        innovation -= datum
                    elif label == 2:
                        innovation += datum
                    component_counts[seed_index, label] += 1
                    (
                        event_log_importance[particle],
                        event_log_q[particle],
                    ) = symmetric_position_log_importance(
                        innovation, datum, position_noise
                    )
                position[particle] += rate[particle] * delta_md + innovation
                tvt_value = position[particle] - z_v[row]
                if tvt_value < grid_minimum - 100.0:
                    tvt_value = grid_minimum - 100.0
                    clip_counts[seed_index] += 1
                if tvt_value > grid_maximum + 100.0:
                    tvt_value = grid_maximum + 100.0
                    clip_counts[seed_index] += 1
                position[particle] = tvt_value + z_v[row]
            row_min = 1.0e300
            row_max = -1.0e300
            for particle in range(particles):
                tvt_value = position[particle] - z_v[row]
                row_min = min(row_min, tvt_value)
                row_max = max(row_max, tvt_value)
                ancestry[row, labels[particle]] += 1.0
            support_min[seed_index, row] = row_min
            support_max[seed_index, row] = row_max
            if row == event_index:
                log_values = np.empty(particles, np.float64)
                minimum_log_importance = 1.0e300
                maximum_log_importance = -1.0e300
                minimum_log_q = 1.0e300
                for particle in range(particles):
                    log_importance = event_log_importance[particle]
                    log_q = event_log_q[particle]
                    expected_gr = _interp1(
                        grid_gr,
                        position[particle] - z_v[row],
                        grid_minimum,
                        grid_step,
                    )
                    zscore = (gr_v[row] - expected_gr) / gr_scale
                    squared = min(zscore * zscore, 600.0)
                    log_emission = max(-0.5 * squared, math.log(1.0e-300))
                    log_values[particle] = (
                        np.log(max(weights[particle], 1.0e-300))
                        + log_importance
                        + log_emission
                    )
                    minimum_log_importance = min(
                        minimum_log_importance, log_importance
                    )
                    maximum_log_importance = max(
                        maximum_log_importance, log_importance
                    )
                    minimum_log_q = min(minimum_log_q, log_q)
                maximum = np.max(log_values)
                normalizer = maximum + np.log(np.sum(np.exp(log_values - maximum)))
                for particle in range(particles):
                    weights[particle] = np.exp(log_values[particle] - normalizer)
                log_likelihood += normalizer
                log_importance_min[seed_index] = minimum_log_importance
                log_importance_max[seed_index] = maximum_log_importance
                log_q_min[seed_index] = minimum_log_q
            else:
                average_likelihood = 0.0
                for particle in range(particles):
                    expected_gr = _interp1(
                        grid_gr,
                        position[particle] - z_v[row],
                        grid_minimum,
                        grid_step,
                    )
                    zscore = (gr_v[row] - expected_gr) / gr_scale
                    squared = min(zscore * zscore, 600.0)
                    likelihood = max(np.exp(-0.5 * squared), 1.0e-300)
                    average_likelihood += weights[particle] * likelihood
                    weights[particle] *= likelihood
                average_likelihood = max(average_likelihood, 1.0e-300)
                log_likelihood += np.log(average_likelihood)
                weight_sum = 0.0
                for particle in range(particles):
                    weight_sum += weights[particle]
                if weight_sum > 0:
                    for particle in range(particles):
                        weights[particle] /= weight_sum
                else:
                    for particle in range(particles):
                        weights[particle] = 1.0 / particles
            inverse_ess = 0.0
            for particle in range(particles):
                inverse_ess += weights[particle] * weights[particle]
            ess = 1.0 / inverse_ess
            minimum_ess[seed_index] = min(minimum_ess[seed_index], ess)
            if ess < resample_fraction * particles:
                cumulative = np.empty(particles, np.float64)
                cumulative_value = 0.0
                for particle in range(particles):
                    cumulative_value += weights[particle]
                    cumulative[particle] = cumulative_value
                initial_uniform = np.random.uniform(0.0, 1.0 / particles)
                new_position = np.empty(particles, np.float64)
                new_rate = np.empty(particles, np.float64)
                new_labels = np.empty(particles, np.int8)
                cursor = 0
                for particle in range(particles):
                    uniform = initial_uniform + particle / particles
                    while cursor < particles - 1 and cumulative[cursor] < uniform:
                        cursor += 1
                    new_position[particle] = (
                        position[cursor] + rough_position * np.random.randn()
                    )
                    new_rate[particle] = rate[cursor] + rough_rate * np.random.randn()
                    new_labels[particle] = labels[cursor]
                position[:] = new_position
                rate[:] = new_rate
                labels[:] = new_labels
                weights[:] = 1.0 / particles
                resampling_counts[seed_index] += 1
            estimate = 0.0
            for particle in range(particles):
                estimate += weights[particle] * (position[particle] - z_v[row])
            predictions[seed_index, row] = estimate
            previous_md = md_v[row]
        log_likelihoods[seed_index] = log_likelihood
        final_position[seed_index] = position
        final_rate[seed_index] = rate
        final_weights[seed_index] = weights
    ancestry /= float(seeds * particles)
    return (
        predictions,
        log_likelihoods,
        support_min,
        support_max,
        ancestry,
        component_counts,
        log_importance_min,
        log_importance_max,
        log_q_min,
        resampling_counts,
        minimum_ess,
        clip_counts,
        final_position,
        final_rate,
        final_weights,
    )


def aggregate_seed_predictions(
    predictions: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    centered = log_likelihoods - float(np.max(log_likelihoods))
    weights = np.exp(centered / float(temperature))
    weights /= float(weights.sum())
    return (weights[:, None] * predictions).sum(axis=0), weights


def run_pf(
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    well: str,
    event_index: int,
    datum: float,
    component_event_row: int | None = None,
) -> dict[str, Any]:
    pf = get_nested(config, "pf")
    dynamics = pf["dynamics"]
    seeds = int(pf["seeds"])
    particles = int(pf["particles"])
    components = component_draws(
        well,
        seeds=seeds,
        particles=particles,
        event_row=int(
            event_index if component_event_row is None else component_event_row
        ),
    )
    started = time.perf_counter()
    outputs = _pf_symmetric_allseeds(
        np.asarray(prepared["eval_md"], dtype=np.float64),
        np.asarray(prepared["eval_z"], dtype=np.float64),
        np.asarray(prepared["eval_gr"], dtype=np.float64),
        np.asarray(prepared["grid_gr"], dtype=np.float64),
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["gr_scale"]),
        float(prepared["last_known_position"]),
        float(prepared["initial_rate"]),
        particles,
        seeds,
        exp404_seed_base(well),
        float(dynamics["momentum"]),
        float(dynamics["rate_noise"]),
        float(dynamics["position_noise"]),
        float(dynamics["rough_position_ft"]),
        float(dynamics["rough_rate"]),
        float(dynamics["resample_threshold_fraction"]),
        float(dynamics["initial_position_spread_ft"]),
        float(dynamics["initial_rate_spread"]),
        int(event_index),
        float(datum),
        components,
    )
    (
        seed_predictions,
        log_likelihoods,
        support_min,
        support_max,
        ancestry,
        component_counts,
        log_importance_min,
        log_importance_max,
        log_q_min,
        resampling_counts,
        minimum_ess,
        clip_counts,
        final_position,
        final_rate,
        final_weights,
    ) = outputs
    prediction, seed_weights = aggregate_seed_predictions(
        seed_predictions,
        log_likelihoods,
        temperature=float(pf["readout"]["temperature"]),
    )
    if event_index >= 0:
        if not (
            np.isfinite(log_importance_min).all()
            and np.isfinite(log_importance_max).all()
            and np.isfinite(log_q_min).all()
        ):
            raise RuntimeError("non-finite defensive-proposal log density")
        if float(log_importance_max.max()) > LOG_IMPORTANCE_MAX + 1.0e-12:
            raise RuntimeError("p0/q theoretical upper bound violated")
    diagnostics = {
        "elapsed_seconds": time.perf_counter() - started,
        "event_index": int(event_index),
        "component_event_row": int(
            event_index if component_event_row is None else component_event_row
        ),
        "datum_ft": float(datum if event_index >= 0 else 0.0),
        "log_importance_min": float(log_importance_min.min()),
        "log_importance_max": float(log_importance_max.max()),
        "log_q_min": float(log_q_min.min()),
        "component_counts": component_counts.sum(axis=0).tolist(),
        "resampling_count": int(resampling_counts.sum()),
        "minimum_ess": float(minimum_ess.min()),
        "position_clip_count": int(clip_counts.sum()),
        "seed_weight_min": float(seed_weights.min()),
        "seed_weight_max": float(seed_weights.max()),
        "seed_weight_sum": float(seed_weights.sum()),
    }
    return {
        "prediction": prediction,
        "seed_predictions": seed_predictions,
        "log_likelihoods": log_likelihoods,
        "support_min": support_min,
        "support_max": support_max,
        "ancestry": ancestry,
        "diagnostics": diagnostics,
        "prediction_sha256": array_bundle_sha256(
            row_idx=np.asarray(prepared["eval_indices"], dtype=np.int64),
            prediction=np.asarray(prediction, dtype=np.float32),
        ),
        "support_sha256": array_bundle_sha256(
            support_min=support_min, support_max=support_max
        ),
        "ancestry_sha256": array_bundle_sha256(ancestry=ancestry),
        "component_sha256": array_bundle_sha256(components=components),
        "particle_state_sha256": array_bundle_sha256(
            final_position=final_position,
            final_rate=final_rate,
            final_weights=final_weights,
        ),
    }


def synthetic_no_event_parity(config: Mapping[str, Any]) -> dict[str, Any]:
    prepared = {
        "eval_indices": np.arange(8, dtype=np.int64),
        "eval_md": np.arange(1.0, 9.0),
        "eval_z": np.linspace(0.0, 0.7, 8),
        "eval_gr": np.asarray([50, 52, 54, 53, 51, 50, 49, 51], np.float64),
        "grid_gr": np.linspace(40.0, 70.0, 151),
        "grid_minimum": 90.0,
        "grid_step": 0.2,
        "gr_scale": 20.0,
        "last_known_position": 100.0,
        "initial_rate": 0.01,
    }
    reduced = json.loads(json.dumps(config))
    reduced["pf"]["particles"] = 24
    reduced["pf"]["seeds"] = 4
    first = run_pf(prepared, reduced, well="synthetic", event_index=-1, datum=0.0)
    second = run_pf(prepared, reduced, well="synthetic", event_index=-1, datum=0.0)
    return {
        "prediction_bitwise": bool(
            np.array_equal(first["prediction"], second["prediction"])
        ),
        "seed_prediction_bitwise": bool(
            np.array_equal(first["seed_predictions"], second["seed_predictions"])
        ),
        "support_bitwise": bool(
            np.array_equal(first["support_min"], second["support_min"])
            and np.array_equal(first["support_max"], second["support_max"])
        ),
        "particle_state_bitwise": bool(
            first["particle_state_sha256"] == second["particle_state_sha256"]
        ),
        "pass": bool(
            np.array_equal(first["prediction"], second["prediction"])
            and np.array_equal(first["seed_predictions"], second["seed_predictions"])
            and np.array_equal(first["support_min"], second["support_min"])
            and np.array_equal(first["support_max"], second["support_max"])
            and first["particle_state_sha256"] == second["particle_state_sha256"]
        ),
    }


# %% [markdown]
# ## 7. Target-free Stage 0 prediction freeze

# %%
@dataclass
class FrozenWell:
    well: str
    row_idx: np.ndarray
    ids: np.ndarray
    raw_gr_missing: np.ndarray
    event_index: int
    datum_ft: float
    hmm_prediction: np.ndarray
    baseline_prediction: np.ndarray
    treatment_prediction: np.ndarray
    baseline_support_min: np.ndarray
    baseline_support_max: np.ndarray
    treatment_support_min: np.ndarray
    treatment_support_max: np.ndarray
    z_beta: np.ndarray
    active_direction: np.ndarray
    filtered_position_std: np.ndarray
    ancestry: np.ndarray
    hmm_saved_max_abs_diff: float
    pf_saved_max_abs_diff: float
    pre_event_prediction_max_abs_diff: float
    hmm_message_sha256: str
    schedule_sha256: str
    baseline_prediction_sha256: str
    treatment_prediction_sha256: str
    baseline_support_sha256: str
    treatment_support_sha256: str
    branch_ancestry_sha256: str
    component_sha256: str
    baseline_particle_state_sha256: str
    treatment_particle_state_sha256: str
    baseline_diagnostics: dict[str, Any]
    treatment_diagnostics: dict[str, Any]
    role: str = ""
    fold: int = -1


def align_saved_prediction(
    saved: pd.DataFrame,
    well: str,
    row_idx: np.ndarray,
) -> np.ndarray:
    frame = saved.loc[saved["well"].eq(well)].sort_values(
        "row_idx", kind="mergesort"
    )
    if not np.array_equal(frame["row_idx"].to_numpy(np.int64), row_idx):
        raise ValueError(f"{well}: saved prediction row identity mismatch")
    return frame["saved_prediction"].to_numpy(np.float64)


def freeze_target_free_well(
    *,
    well: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
    saved_hmm: pd.DataFrame,
    saved_pf: pd.DataFrame,
    ledger: LeakageLedger,
) -> FrozenWell:
    hmm_prepared = prepare_hmm_inputs(horizontal, typewell, get_nested(config, "hmm"))
    pf_prepared = prepare_pf_inputs(
        horizontal, typewell, get_nested(config, "pf.dynamics")
    )
    if not np.array_equal(
        hmm_prepared["eval_indices"], pf_prepared["eval_indices"]
    ):
        raise ValueError(f"{well}: HMM/PF suffix identities differ")
    row_idx = np.asarray(pf_prepared["eval_indices"], dtype=np.int64)
    ids = np.asarray([f"{well}_{int(row)}" for row in row_idx], dtype=str)
    hmm = run_first_pass_hmm(hmm_prepared, get_nested(config, "hmm"))
    schedule = beta_filter_activation_schedule(
        hmm["smoothed_rate_mean"],
        hmm["filtered_rate_mean"],
        hmm["filtered_rate_std"],
        get_nested(config, "trigger"),
    )
    event_index = first_persistent_activation_event(schedule["active_direction"])
    datum = (
        max(float(hmm["filtered_position_std"][event_index]), 0.35)
        if event_index >= 0
        else 0.0
    )
    baseline = run_pf(
        pf_prepared,
        config,
        well=well,
        event_index=-1,
        datum=0.0,
    )
    treatment = run_pf(
        pf_prepared,
        config,
        well=well,
        event_index=event_index,
        datum=datum,
        component_event_row=(
            int(row_idx[event_index]) if event_index >= 0 else -1
        ),
    )
    saved_hmm_values = align_saved_prediction(saved_hmm, well, row_idx)
    saved_pf_values = align_saved_prediction(saved_pf, well, row_idx)
    hmm_diff = float(
        np.max(
            np.abs(
                np.asarray(hmm["prediction"], dtype=np.float32).astype(np.float64)
                - np.asarray(saved_hmm_values, dtype=np.float32).astype(np.float64)
            )
        )
    )
    pf_diff = float(
        np.max(
            np.abs(
                np.asarray(baseline["prediction"], dtype=np.float32).astype(np.float64)
                - np.asarray(saved_pf_values, dtype=np.float32).astype(np.float64)
            )
        )
    )
    parity_stop = event_index if event_index >= 0 else len(row_idx)
    pre_event_diff = (
        float(
            np.max(
                np.abs(
                    baseline["seed_predictions"][:, :parity_stop]
                    - treatment["seed_predictions"][:, :parity_stop]
                )
            )
        )
        if parity_stop > 0
        else 0.0
    )
    schedule_sha = array_bundle_sha256(
        row_idx=row_idx,
        z_beta=schedule["z_beta"],
        active_direction=schedule["active_direction"],
        event_index=np.asarray([event_index], dtype=np.int64),
        datum=np.asarray([datum], dtype=np.float64),
    )
    ledger.freeze(well)
    return FrozenWell(
        well=well,
        row_idx=row_idx,
        ids=ids,
        raw_gr_missing=np.asarray(pf_prepared["raw_gr_missing"], dtype=bool),
        event_index=event_index,
        datum_ft=datum,
        hmm_prediction=np.asarray(hmm["prediction"], dtype=np.float64),
        baseline_prediction=np.asarray(baseline["prediction"], dtype=np.float64),
        treatment_prediction=np.asarray(treatment["prediction"], dtype=np.float64),
        baseline_support_min=np.asarray(baseline["support_min"], dtype=np.float32),
        baseline_support_max=np.asarray(baseline["support_max"], dtype=np.float32),
        treatment_support_min=np.asarray(treatment["support_min"], dtype=np.float32),
        treatment_support_max=np.asarray(treatment["support_max"], dtype=np.float32),
        z_beta=np.asarray(schedule["z_beta"], dtype=np.float64),
        active_direction=np.asarray(schedule["active_direction"], dtype=np.int8),
        filtered_position_std=np.asarray(
            hmm["filtered_position_std"], dtype=np.float64
        ),
        ancestry=np.asarray(treatment["ancestry"], dtype=np.float64),
        hmm_saved_max_abs_diff=hmm_diff,
        pf_saved_max_abs_diff=pf_diff,
        pre_event_prediction_max_abs_diff=pre_event_diff,
        hmm_message_sha256=str(hmm["message_sha256"]),
        schedule_sha256=schedule_sha,
        baseline_prediction_sha256=str(baseline["prediction_sha256"]),
        treatment_prediction_sha256=str(treatment["prediction_sha256"]),
        baseline_support_sha256=str(baseline["support_sha256"]),
        treatment_support_sha256=str(treatment["support_sha256"]),
        branch_ancestry_sha256=str(treatment["ancestry_sha256"]),
        component_sha256=str(treatment["component_sha256"]),
        baseline_particle_state_sha256=str(baseline["particle_state_sha256"]),
        treatment_particle_state_sha256=str(treatment["particle_state_sha256"]),
        baseline_diagnostics=dict(baseline["diagnostics"]),
        treatment_diagnostics=dict(treatment["diagnostics"]),
    )


def target_free_prediction_frame(frozen: Sequence[FrozenWell]) -> pd.DataFrame:
    pieces = []
    for item in frozen:
        pieces.append(
            pd.DataFrame(
                {
                    "id": item.ids,
                    "well": item.well,
                    "row_idx": item.row_idx,
                    "hmm_first_pass": item.hmm_prediction.astype(np.float32),
                    "baseline_pf": item.baseline_prediction.astype(np.float32),
                    "treatment_pf": item.treatment_prediction.astype(np.float32),
                }
            )
        )
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="mergesort"
    )


def target_free_schedule_frame(frozen: Sequence[FrozenWell]) -> pd.DataFrame:
    pieces = []
    for item in frozen:
        pieces.append(
            pd.DataFrame(
                {
                    "well": item.well,
                    "row_idx": item.row_idx,
                    "suffix_offset": np.arange(len(item.row_idx), dtype=np.int64),
                    "z_beta": item.z_beta,
                    "active": item.active_direction != 0,
                    "event": np.arange(len(item.row_idx)) == item.event_index,
                    "event_index": item.event_index,
                    "filtered_position_std_ft": item.filtered_position_std,
                    "datum_ft": item.datum_ft,
                    "base_ancestry_fraction": item.ancestry[:, 0],
                    "minus_ancestry_fraction": item.ancestry[:, 1],
                    "plus_ancestry_fraction": item.ancestry[:, 2],
                }
            )
        )
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="mergesort"
    )


# %% [markdown]
# ## 8. Truth-late mechanism readout and fail-closed gates

# %%
def load_suffix_truth(
    item: FrozenWell,
    raw_dir: Path,
    ledger: LeakageLedger,
) -> np.ndarray:
    frame = pd.read_csv(
        raw_dir / f"{item.well}__horizontal_well.csv",
        usecols=["TVT", "TVT_input"],
    )
    truth = frame.loc[frame["TVT_input"].isna(), "TVT"].to_numpy(np.float64)
    if len(truth) != len(item.row_idx) or not np.isfinite(truth).all():
        raise ValueError(f"{item.well}: truth identity/coverage changed")
    ledger.record_truth_late(len(truth))
    return truth


def majority_seed_outside_support(
    truth: np.ndarray,
    support_min: np.ndarray,
    support_max: np.ndarray,
) -> np.ndarray:
    outside = (truth[None, :] < support_min) | (truth[None, :] > support_max)
    return outside.mean(axis=0) > 0.5


def attach_roles(
    frozen: Sequence[FrozenWell],
    manifest: pd.DataFrame,
) -> None:
    lookup = manifest.set_index("well")
    for item in frozen:
        item.role = str(lookup.loc[item.well, "role"])
        item.fold = int(lookup.loc[item.well, "fold"])


def truth_late_readout(
    frozen: Sequence[FrozenWell],
    raw_dir: Path,
    ledger: LeakageLedger,
    window_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_pieces: list[pd.DataFrame] = []
    well_rows: list[dict[str, Any]] = []
    for item in frozen:
        truth = load_suffix_truth(item, raw_dir, ledger)
        baseline_error = item.baseline_prediction - truth
        treatment_error = item.treatment_prediction - truth
        baseline_outside = majority_seed_outside_support(
            truth, item.baseline_support_min, item.baseline_support_max
        )
        treatment_outside = majority_seed_outside_support(
            truth, item.treatment_support_min, item.treatment_support_max
        )
        window = np.zeros(len(truth), dtype=bool)
        if item.event_index >= 0:
            stop = min(item.event_index + int(window_rows), len(truth))
            window[item.event_index : stop] = True
        row_pieces.append(
            pd.DataFrame(
                {
                    "well": item.well,
                    "role": item.role,
                    "fold": item.fold,
                    "row_idx": item.row_idx,
                    "event_window": window,
                    "truth": truth,
                    "baseline_prediction": item.baseline_prediction,
                    "treatment_prediction": item.treatment_prediction,
                    "baseline_squared_error": baseline_error**2,
                    "treatment_squared_error": treatment_error**2,
                    "baseline_majority_seed_outside_support": baseline_outside,
                    "treatment_majority_seed_outside_support": treatment_outside,
                }
            )
        )
        well_rows.append(
            {
                "well": item.well,
                "role": item.role,
                "fold": item.fold,
                "rows": len(truth),
                "event_present": item.event_index >= 0,
                "event_index": item.event_index,
                "datum_ft": item.datum_ft,
                "baseline_rmse_ft": float(np.sqrt(np.mean(baseline_error**2))),
                "treatment_rmse_ft": float(np.sqrt(np.mean(treatment_error**2))),
                "rmse_delta_ft": float(
                    np.sqrt(np.mean(treatment_error**2))
                    - np.sqrt(np.mean(baseline_error**2))
                ),
                "hmm_saved_max_abs_diff_ft": item.hmm_saved_max_abs_diff,
                "pf_saved_max_abs_diff_ft": item.pf_saved_max_abs_diff,
                "pre_event_prediction_max_abs_diff_ft": (
                    item.pre_event_prediction_max_abs_diff
                ),
            }
        )
    rows = pd.concat(row_pieces, ignore_index=True)
    return rows, pd.DataFrame(well_rows)


def safe_reduction(baseline: float, treatment: float) -> float:
    return 1.0 - treatment / baseline if baseline > 0 else math.nan


def evaluate_stage0_gates(
    *,
    config: Mapping[str, Any],
    frozen: Sequence[FrozenWell],
    rows: pd.DataFrame,
    wells: pd.DataFrame,
    ledger: LeakageLedger,
    parity: Mapping[str, Any],
    quadrature: Mapping[str, Any],
    schedule_readback_pass: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    technical_config = get_nested(config, "technical_gate")
    mechanism_config = get_nested(config, "mechanism_gate")
    event_items = [item for item in frozen if item.event_index >= 0]
    event_rows = rows.loc[rows["event_window"]]
    baseline_sse = float(event_rows["baseline_squared_error"].sum())
    treatment_sse = float(event_rows["treatment_squared_error"].sum())
    sse_reduction = safe_reduction(baseline_sse, treatment_sse)
    baseline_outside = float(
        event_rows["baseline_majority_seed_outside_support"].mean()
    )
    treatment_outside = float(
        event_rows["treatment_majority_seed_outside_support"].mean()
    )
    support_reduction = baseline_outside - treatment_outside
    fold_rows = []
    for fold in range(5):
        frame = event_rows.loc[event_rows["fold"].eq(fold)]
        fold_baseline = float(frame["baseline_squared_error"].sum())
        fold_treatment = float(frame["treatment_squared_error"].sum())
        fold_rows.append(
            {
                "fold": fold,
                "rows": len(frame),
                "baseline_sse": fold_baseline,
                "treatment_sse": fold_treatment,
                "nonworse": bool(
                    len(frame) > 0 and fold_treatment <= fold_baseline
                ),
            }
        )
    nonworse_folds = int(sum(item["nonworse"] for item in fold_rows))
    control_rows = rows.loc[rows["role"].eq("control")]
    control_baseline_rmse = float(
        np.sqrt(control_rows["baseline_squared_error"].mean())
    )
    control_treatment_rmse = float(
        np.sqrt(control_rows["treatment_squared_error"].mean())
    )
    control_delta = control_treatment_rmse - control_baseline_rmse
    control_worst = float(
        wells.loc[wells["role"].eq("control"), "rmse_delta_ft"].max()
    )
    maximum_log_importance = max(
        item.treatment_diagnostics["log_importance_max"] for item in frozen
    )
    finite_diagnostics = all(
        math.isfinite(float(value))
        for item in frozen
        for value in (
            item.treatment_diagnostics["log_importance_min"],
            item.treatment_diagnostics["log_importance_max"],
            item.treatment_diagnostics["log_q_min"],
        )
    )
    no_event_items = [item for item in frozen if item.event_index < 0]
    no_event_bitwise = all(
        item.baseline_prediction_sha256 == item.treatment_prediction_sha256
        and item.baseline_support_sha256 == item.treatment_support_sha256
        and item.baseline_particle_state_sha256
        == item.treatment_particle_state_sha256
        for item in no_event_items
    )
    maximum_hmm_parity = max(item.hmm_saved_max_abs_diff for item in frozen)
    maximum_pf_parity = max(item.pf_saved_max_abs_diff for item in frozen)
    maximum_pre_event_diff = max(
        item.pre_event_prediction_max_abs_diff for item in frozen
    )
    event_once = all(
        int(np.sum(item.active_direction != 0)) >= 0
        and item.event_index == first_persistent_activation_event(
            item.active_direction
        )
        for item in frozen
    )
    runtime_projection = elapsed_seconds * 773.0 / 32.0
    technical = {
        "truth_reads_before_freeze_zero": ledger.truth_rows_before_all_freeze == 0,
        "cause_role_reads_before_freeze_zero": ledger.role_rows_before_all_freeze == 0,
        "event_maximum_once_and_first_entry": event_once,
        "no_event_bitwise_parent_parity": no_event_bitwise,
        "synthetic_no_event_parity": bool(parity["pass"]),
        "finite_log_q_and_importance": finite_diagnostics,
        "importance_upper_bound": bool(
            maximum_log_importance <= LOG_IMPORTANCE_MAX + 1.0e-12
        ),
        "importance_moment_contract": bool(quadrature["pass"]),
        "trigger_schedule_sha_readback": bool(schedule_readback_pass),
        "hmm_saved_parent_parity": bool(
            maximum_hmm_parity
            <= float(get_nested(config, "hmm.saved_parent_prediction_tolerance_ft"))
        ),
        "pf_saved_parent_parity": bool(
            maximum_pf_parity
            <= float(technical_config["saved_parent_prediction_tolerance_ft"])
        ),
        "base_common_random_pre_event_parity": maximum_pre_event_diff == 0.0,
        "runtime_projection": runtime_projection
        <= float(get_nested(config, "runtime.hard_runtime_limit_seconds")),
        "peak_rss": peak_rss_gb()
        <= float(get_nested(config, "runtime.peak_rss_limit_gb")),
    }
    mechanism = {
        "triggered_wells": len(event_items)
        >= int(mechanism_config["triggered_wells_min"]),
        "support_fraction_reduction": bool(
            math.isfinite(support_reduction)
            and support_reduction
            >= float(
                mechanism_config[
                    "majority_seed_truth_outside_support_fraction_absolute_reduction_min"
                ]
            )
        ),
        "triggered_window_sse_reduction": bool(
            math.isfinite(sse_reduction)
            and sse_reduction
            >= float(mechanism_config["triggered_window_sse_reduction_min"])
        ),
        "nonworse_reporting_folds": nonworse_folds
        >= int(mechanism_config["nonworse_reporting_folds_min"]),
        "matched_control_pooled_rmse": control_delta
        <= float(mechanism_config["matched_control_pooled_rmse_delta_max_ft"]),
        "matched_control_worst_well": control_worst
        <= float(mechanism_config["matched_control_worst_well_rmse_delta_max_ft"]),
    }
    diagnostics = {
        "triggered_wells": len(event_items),
        "no_event_wells": len(no_event_items),
        "baseline_majority_seed_outside_support_fraction": baseline_outside,
        "treatment_majority_seed_outside_support_fraction": treatment_outside,
        "support_fraction_absolute_reduction": support_reduction,
        "triggered_window_baseline_sse": baseline_sse,
        "triggered_window_treatment_sse": treatment_sse,
        "triggered_window_sse_reduction": sse_reduction,
        "folds": fold_rows,
        "nonworse_folds": nonworse_folds,
        "control_baseline_rmse_ft": control_baseline_rmse,
        "control_treatment_rmse_ft": control_treatment_rmse,
        "control_rmse_delta_ft": control_delta,
        "control_worst_well_rmse_delta_ft": control_worst,
        "maximum_hmm_saved_parent_diff_ft": maximum_hmm_parity,
        "maximum_pf_saved_parent_diff_ft": maximum_pf_parity,
        "maximum_pre_event_seed_prediction_diff_ft": maximum_pre_event_diff,
        "maximum_log_importance": maximum_log_importance,
        "runtime_projection_seconds": runtime_projection,
        "peak_rss_gb": peak_rss_gb(),
    }
    return {
        "technical": technical,
        "mechanism": mechanism,
        "diagnostics": diagnostics,
        "technical_pass": bool(all(technical.values())),
        "mechanism_pass": bool(all(mechanism.values())),
        "full_eligible": bool(all(technical.values()) and all(mechanism.values())),
    }


# %% [markdown]
# ## 9. Generated artifacts and guarded execution

# %%
def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.is_dir():
        return
    if os.environ.get("EXP432_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError("exp432 Stage 0 must run on Kaggle CPU")


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    validate_execution_contract(config, require_run_authorization=True)
    contract = validate_scientific_contract(config)
    started = time.perf_counter()
    set_num_threads(int(get_nested(config, "runtime.numba_threads_per_worker")))
    ledger = LeakageLedger()
    wells, manifest_evidence = load_fixed32_scope(config, ledger)
    well_set = set(wells)
    saved_hmm, saved_hmm_evidence = load_saved_control_subset(
        config, "exp209_saved_control", well_set, ledger
    )
    saved_pf, saved_pf_evidence = load_saved_control_subset(
        config, "exp404_saved_control", well_set, ledger
    )
    raw_dir = train_data_dir(config)
    parity = synthetic_no_event_parity(config)
    quadrature = importance_quadrature_contract(
        datum=0.35,
        sigma=float(get_nested(config, "pf.dynamics.position_noise")),
    )
    frozen: list[FrozenWell] = []
    for index, well in enumerate(wells, start=1):
        horizontal, typewell = load_target_free_well(well, raw_dir, ledger)
        item = freeze_target_free_well(
            well=well,
            horizontal=horizontal,
            typewell=typewell,
            config=config,
            saved_hmm=saved_hmm,
            saved_pf=saved_pf,
            ledger=ledger,
        )
        frozen.append(item)
        print(
            f"[{index:02d}/32] {well} event={item.event_index} "
            f"datum={item.datum_ft:.6f} "
            f"hmm_parity={item.hmm_saved_max_abs_diff:.3g} "
            f"pf_parity={item.pf_saved_max_abs_diff:.3g}",
            flush=True,
        )
    if not ledger.all_frozen:
        raise RuntimeError("not all fixed32 target-free artifacts were frozen")
    prediction_frame = target_free_prediction_frame(frozen)
    schedule_frame = target_free_schedule_frame(frozen)
    output = artifacts_dir()
    prediction_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_target_free_predictions.csv.gz",
        prediction_frame,
    )
    schedule_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_trigger_schedule.csv.gz",
        schedule_frame,
    )
    schedule_readback_pass = (
        sha256_decompressed_csv(Path(schedule_artifact["path"]))
        == schedule_artifact["decompressed_sha256"]
    )
    target_free_sha_rows = []
    for item in frozen:
        target_free_sha_rows.append(
            {
                "well": item.well,
                "event_index": item.event_index,
                "datum_ft": item.datum_ft,
                "hmm_message_sha256": item.hmm_message_sha256,
                "schedule_sha256": item.schedule_sha256,
                "baseline_prediction_sha256": item.baseline_prediction_sha256,
                "treatment_prediction_sha256": item.treatment_prediction_sha256,
                "baseline_support_sha256": item.baseline_support_sha256,
                "treatment_support_sha256": item.treatment_support_sha256,
                "branch_ancestry_sha256": item.branch_ancestry_sha256,
                "component_sha256": item.component_sha256,
                "baseline_particle_state_sha256": (
                    item.baseline_particle_state_sha256
                ),
                "treatment_particle_state_sha256": (
                    item.treatment_particle_state_sha256
                ),
            }
        )
    sha_artifact = write_json(
        output / f"{EXPERIMENT_NAME}_stage0_target_free_sha_ledger.json",
        {"wells": target_free_sha_rows},
    )
    manifest = load_fixed32_roles_after_freeze(config, ledger)
    attach_roles(frozen, manifest)
    truth_rows, well_metrics = truth_late_readout(
        frozen,
        raw_dir,
        ledger,
        int(get_nested(config, "validation.evaluation_window_rows_inclusive")),
    )
    elapsed = time.perf_counter() - started
    gates = evaluate_stage0_gates(
        config=config,
        frozen=frozen,
        rows=truth_rows,
        wells=well_metrics,
        ledger=ledger,
        parity=parity,
        quadrature=quadrature,
        schedule_readback_pass=schedule_readback_pass,
        elapsed_seconds=elapsed,
    )
    well_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_well_metrics.csv.gz",
        well_metrics,
    )
    truth_artifact = write_deterministic_gzip_csv(
        output / f"{EXPERIMENT_NAME}_stage0_truth_late_rows.csv.gz",
        truth_rows,
    )
    payload = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "stage0_pass_eligible_for_separate_full_approval"
            if gates["full_eligible"]
            else "stage0_fail_closed"
        ),
        "route": "pf_beam",
        "stage0": gates,
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "executed": True,
        "scientific_contract": contract,
        "scientific_contract_sha256": contract["sha256"],
        "execution_contract": validate_execution_contract(
            config, require_run_authorization=False
        ),
        "manifest": manifest_evidence,
        "inputs": {
            "saved_hmm": saved_hmm_evidence,
            "saved_pf": saved_pf_evidence,
        },
        "leakage_ledger": {
            "frozen_wells": len(ledger.frozen_wells),
            "target_free_rows": ledger.target_free_rows,
            "truth_rows_before_all_freeze": ledger.truth_rows_before_all_freeze,
            "role_rows_before_all_freeze": ledger.role_rows_before_all_freeze,
            "truth_rows_after_all_freeze": ledger.truth_rows_after_all_freeze,
            "role_rows_after_all_freeze": ledger.role_rows_after_all_freeze,
        },
        "contracts": {
            "synthetic_no_event_parity": parity,
            "importance_quadrature": quadrature,
        },
        "artifacts": {
            "predictions": prediction_artifact,
            "schedule": schedule_artifact,
            "well_metrics": well_artifact,
            "truth_late_rows": truth_artifact,
            "sha_ledger": sha_artifact,
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_rss_gb": peak_rss_gb(),
            "versions": runtime_versions(),
        },
    }
    write_json(metrics_path(), payload)
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), flush=True)
    return payload


CONFIG = load_config()
EXECUTION_COUNTS = validate_execution_contract(
    CONFIG, require_run_authorization=False
)
SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "status": get_nested(CONFIG, "experiment.status"),
            "selected_stage": get_nested(CONFIG, "execution.selected_stage"),
            "execution_counts": EXECUTION_COUNTS,
            "scientific_contract_sha256": SCIENTIFIC_CONTRACT["sha256"],
        },
        indent=2,
        sort_keys=True,
    ),
    flush=True,
)

if os.environ.get("EXP432_IMPORT_ONLY", "0") != "1":
    selected_stage = get_nested(CONFIG, "execution.selected_stage")
    if selected_stage is None:
        print(
            "exp432 implementation is ready; Kaggle Stage 0 remains locked.",
            flush=True,
        )
    elif selected_stage == "stage0_fixed32":
        run_stage0(CONFIG)
    else:
        raise ValueError(f"unsupported exp432 execution stage: {selected_stage}")
